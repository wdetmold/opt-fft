# L36_mixedradix — strategy record (ICE LAKE panel)

Continuation of this entry's records from the single-thread CLX panel
(`bench/geom/strategies/L36_mixedradix.md`, rounds panel_r1..r11 — the full
do-not-retry list, the bit-class rules, the probe designs live there).  This
file records what happens on the Ice Lake panel (bare-metal Xeon Gold 6326,
2×512-bit FMA pipes, 48 KB L1d / 1.25 MB L2 / ~24 MB L3, graded chain
workload `cases.txt` 36:8:64 — 64 unitary steps over 8 volumes, three
5.97 MB buffers L3-resident, driver-side 1/√V scale pass rereading and
rewriting the whole output after EVERY step).

## Round ice_r2

### Where this round started

My ice_r1 agent crashed in the worker-crash storm (exit 134) before writing
anything, so the ice_r1 board scored the untouched geom panel_r11 code:
**119.530 us/xform, 12% spread, pick `v1-cached-pf0`** — a dead tie with
L36_pfa (119.163, 1.4%) and ahead of L36_pencilfused (123.594) and MKL
(160.688).  The r11 zy bet got its ice answer off the r1 description string:
probe pf0=116.9 vs zy=137.9, **+18% — zy is dead on a 2×512-pipe part** (the
y-call FMAs it donates to the z-transpose window were the second pipe's
food, not slack).

### What the round's measurements established (tryout.sh = the node itself,
### one leased core; windows range quiet to heavily contended — MKL on the
### same case/core is the load normalizer, 160–192 us across the day)

1. **Where the time goes** (rdtsc phase splits via the new `-DFFT36_TSC`
   hook; per-call floors from the 232-FMA/57-shuffle op count at 2.9 GHz,
   two FMA pipes):
   - B=1 chain (everything cache-resident): execute ≈ 94.5 us/vol.  Split
     z/y/p2 = 37.3/39.5/23.2% → **336 cyc/call in the y-subloop vs 197 in
     its identical-arithmetic x twin** (floor 144.5); z 317 (floor 216).
   - B=8 graded: execute ≈ 110 us/vol.  Split 44.3/33.1/22.6 — z grows +18
     us (src streamed from dirty L3), y +10 (dst RFO from L3), p2 +2.
   - The driver scale pass costs ~18–20 us/xform on top of execute —
     identical for every backend, not addressable.
2. **The y-subloop anomaly at B=1 is 4K store→load aliasing, and on ICX it
   is enormous.**  Pinning (pout − pl) mod 4096 = 2112 (the old CLX
   anti-alias knob, r8, where it priced 0 to −1.2%) read **−22% end-to-end
   on the node's graded B=1 chain: 122.48 → 95.39 us/xform**, MKL steady
   (140.5/139.6) across the pair, arena pf0 102.0 → 82.5, y-share of the
   tsc split 38.4 → 32.7%.  Both subloop bodies load all 36 vectors before
   storing any (so within-call aliasing is impossible); the collisions are
   across consecutive calls — load row k of call zb+1 vs in-flight store
   row j of call zb, colliding when pind ≡ 64·(1+9(k−j)) mod 4096.
   2112 = 64·(1+9·32) puts the colliding pairs 32 rows apart (those stores
   are ~36 stores old, long retired) — the CLX plateau center lands on the
   analytically sensible residue.
   At B=8 the same race prices FLAT (111.7–112.1 across {2112,−1,1536,
   2688}): the y-subloop is RFO-drain-bound there and the alias stalls hide
   under the store misses.  Default is now 2112 everywhere + a plan-time
   stage-B race over {2112, −1, 1536, 2688} (bits-neutral knob; env/-D
   overrides stay absolute).
3. **Window adaptivity is worth more than any single mechanism.**  In quiet
   windows every prefetch mechanism sits within ~2% of pf0 (pf1 the
   consistent −1.5..−3%); under core-lease contention the write-intent
   family wins hugely: pf0 162.0 vs pfw 130.4, pfin-pfw 122.2, **l1-pfw
   119.3** us/vol in-arena (MKL 192 that window).  The chain-shaped tuner
   runs under the same conditions as the imminent run, so the pick adapts.
4. **pf1 ≤ pf0 in every window measured** (ten runs, quiet and loaded) —
   pf1 is now the first-listed incumbent; pf0 must beat it by 3%.

### What was changed (in order of measured effect)

1. **pind = 2112 default + stage-B residue race** (−22% at the node's B=1
   chain; flat at B=8; see above).  This is the round's headline number.
2. **Chain-shaped tuner** (adopted from L17_matrixsimd ice_r1 stage 1g,
   attributed): cached-regime candidates are timed under the driver's own
   loop — nt = min(batch, 8) volumes, dst unitarily scaled after every step
   and ping-ponged back as src, two full-size destination arenas, only the
   execute() spans accumulated (the scale pass shapes cache/dirty state but
   is the same additive constant for every candidate).  The old fresh-src
   nt≤4 arena measured a milder regime and produced ice_r1's 12%-spread
   pf0 pick.
3. **Clock-settle spin** (~150 ms dependent scalar FP at the top of
   create(); adopted from L17_winograd via L17_matrixsimd ice_r1,
   attributed) — the schedutil governor otherwise ranks candidates on an
   unramped core.
4. **Pool retuned for ICX**: order pf1, pf0, pfin, pfw, pfin-pfw, l1-pfw
   (V1 only when AVX-512 exists — L17 measured every 256-bit variant at
   +34% on this machine; V0 is the no-AVX-512 fallback).  pfw ungated at
   every batch (the CLX batch≥2 gate assumed B=1 dst stays L2-resident; on
   the graded chain it does not — dst was last touched a full step ago).
   pfw-only unbundled shape from L36_pfa r11, attributed.
5. **New `l1` composite (exec codes 8/9)**: far T1 cursor (32 KB ahead,
   L3→L2) + near T0 cursor (2 z-blocks ahead, L2→L1) over the phase-1 src
   stream, plus the next y-call's 36 scratch lines prefetched T0 (half of
   pl has been evicted to L2 by the y-subloop's own store stream by then).
   Best-in-window under contention (119.3 vs pfin-pfw 122.2); ~pf1-equal in
   quiet windows.  Fielded as l1-pfw only.
6. **zy retired** (ice_r1 node probe +18%, above); codes 5/6 now carry the
   nta bodies (kept compiled for FORCEPICK A/Bs, not fielded — see below).
7. **Dev hooks that survive**: `-DFFT36_VERBOSE` (tuner table to stderr),
   `-DFFT36_TSC` (phase-split rdtsc accounting), `-DFFT36_FORCEPICK=<0..9>`
   (bypass the pick), `-DFFT36_PIND_DEF=<bytes>` (absolute pin, skips the
   race).  tryout.sh cannot pass env through ssh (L17's lesson), so these
   are compile hooks; the env overrides (FFT36_PIND/PFIN/PFW/NTA) remain
   for the monitor's interactive A/Bs.  `-DFFT36_PMU` exists but
   perf_event_open is blocked on the node (paranoid=4) despite the brief —
   rdtsc is the instrument.

### Operation count

Unchanged from geom r10/r11: 232 FMA-port vector ops + 57 port-5 shuffles
per 36-point line over PW lanes (DFT4 8+1 ×9, n1_9 DFT9 40+12 ×4), plus 144
transpose shuffles per z-call; 724 real flops/line, 2 814 912 flops/volume.
Two-pipe port floor ≈ 56.5 us/vol at 2.9 GHz (z-calls port-5-bound at 216
cyc, y/x calls balanced at 144.5).  This round adds zero arithmetic — all
changes are plan-time or prefetch/aliasing mechanisms.

### What was measured on the node (tryout.sh, graded chain 36:8:64 and the
### B=1 diagnostic; min/median per-xform; MKL same case/core as normalizer)

- Round start (r1 board, quiet scoring window): 119.530 us/xform.
- Baseline re-run in a noisy dev window: min 137.7 / median 212 (sd 13%),
  MKL 163.9.
- + settle + chain tuner: min 128.5 / median 137.3, MKL 160.2 (ratio 0.802).
- + pind + pool retune (semi-quiet window): **min 127.6 / median 127.7
  (sd 2.6%), MKL 168.9 — ratio 0.756, the day's best**; pick v1-pf1;
  in-arena pf1 110.3, l1-pfw 109.9, pf0 112.5.
- B=1 chain with pind: **min 95.39 / median 95.44 us/xform (sd 0.16%)**,
  MKL 139.6 — vs 122.5 without pind in the same window.
- Correctness everywhere: rel_l2 = 3.586e-16 (B=8) / 3.591e-16 (B=1) vs
  numpy (tol 1e-12), chain-64 closed-form check 1.261e-14 (tol 8e-12),
  bit-repeatable across runs, single bit class in every pool.
- Setup 0.54 s at B=8, 0.19 s at B=1 (settle 0.15 s + admission + chain
  tournament + pind race).

### What was tried and did NOT work — with the number that killed it

1. **Pre-RA scheduling pragma** (`schedule-insns`,`sched-pressure` — L17's
   −7.7% cure): MKL-normalized ratio 0.866 vs 0.802 without it (windows
   shifted between runs; the normalized read is the honest one).  Corpus
   §10 predicted exactly this for the 36/45/64 codelet class.  Kept as a
   dormant `-DFFT36_SCHED` hook, default off.
2. **Bare nta (PREFETCHNTA L2-bypass on src)**: null in every window
   (112.20 vs pf0 112.33 quiet; 164.8 vs 162.0 loaded).  The theory (keep
   the once-read src out of L2 so dst stays resident for phase 2) loses
   what it wins: NTA lines skip L2, so anything evicted from the 48 KB L1
   before use comes back at L3 latency.  Not fielded; body kept for
   FORCEPICK.
3. **Bare l1 (T0-only near cursor, first design)**: worst-in-pool under
   load (174.5 vs pf0 171.7) — short-lead T0 alone cannot hide L3 latency
   and just adds uops.  The redesigned far-T1+near-T0 composite is what
   shipped.
4. **PMU counters**: perf_event_open fails, kernel.perf_event_paranoid = 4
   on a80n0 — the brief's "PMU exposed" does not hold for implementer
   leases.  rdtsc phase accounting (FFT36_TSC) is the replacement.
5. The pind race at B=8: flat (111.7–112.1 over four residues) — recorded
   so nobody reads the B=1 −22% as a B=8 promise.  The knob is kept for
   the B=1-like cells and because it costs ~0.3 s of setup.

### Borrowed this round

- Chain-shaped in-plan tuning, the clock-settle spin, the "env doesn't
  cross ssh, use -D hooks" lesson, and the NT-stores-are-dead-on-L3-chains
  confirmation: **L17_matrixsimd ice_r1**.
- pfw-unbundled-from-pfin candidate shape: **L36_pfa panel_r11**.
- The two-FMA-pipe port model and the "restructuring beats prefetch"
  prior: corpus §10; its bare-metal provenance note (second pipe genuinely
  feedable) is what made the y-subloop's 336-vs-197 cyc/call reading
  actionable rather than dismissed as feed-cap noise.

### Predictions for the node (stated so they can be scored)

- The scoring window is quieter than any dev window today.  Expect pick
  `v1-pf1` (or l1-pfw if the window has any load), pind=2112, and
  **~113–121 us/xform at the graded cell** (from 127.6 in a 168.9-MKL
  window, scaled by the quiet-window MKL of ~160).  Spread should drop
  from r1's 12% to ≤3% (settle spin + chain-shaped arena).
- If L36_pfa's agent also ran this round, the tie likely persists at the
  mechanism level — the differentiators here are pind (B=1-class cells)
  and the adaptive pfw pool (loaded windows).
- rel_l2 fingerprints 3.586e-16/3.591e-16 unchanged (the n1_9 bit class).

### Next round

1. **The remaining gap is structural, not mechanistic**: quiet B=8 execute
   ≈ 110 us vs the 56.5 us port floor, with z at 44% of it.  Every read
   prefetch instrument is now within ±2% — the z-subloop's ~18 us of
   un-overlapped L3 streaming appears to be an L2-miss-queue sharing
   effect (three concurrent miss streams: src reads, dst RFOs, p2
   refetches), not a latency-coverage problem.  The only untried in-core
   lever with a positive paper estimate is the extract-store transpose-out
   (−72 port-5 shuffles per z-call for +108 16-B stores, ~−4 us/vol at the
   z-call port balance) — L17's store-forwarding post-mortem says the risk
   is the LAST z-call's stores being reread as zmm by the first y-call;
   everything earlier is thousands of cycles retired.  Worth one build.
2. Re-read the y-subloop RFO question at B=8 with a pfw-distance race
   (L36_pfa's pfwd=1296 won on their cell; mine still uses the 2592
   default — only matters in loaded windows, so low priority).
3. If the monitor's leaderboard shows the spread still >3%, the residual
   is window noise in the SCORING run, not the plan — ask for median-based
   reads before building anything.
4. Do NOT retry: zy (+18% ice_r1), bare nta / bare-T0 l1 (numbers above),
   sched pragma (0.866 vs 0.802 normalized), NT stores (L17: 2× on
   L3-resident chains, six rounds of geom rejections).

## Round ice_r4

(No ice_r3 section exists: that round's agent left no record and the r3
board scored the unchanged ice_r2 code at 117.417 us/xform, third behind
L36_pencilfused 109.6 and L36_pfa 111.6.)

### The task changed: own the chain or pay for an unfused map

The graded step is now the rivals' full step, state <- (z+c)/(1+|z+c|)
with z the RAW FFT of the state, and the driver times an exported
`fft3d_chain(plan, x0, c, final_out, m)` weak symbol for the whole m=64
chain.  Everything this round is that entry point; fft3d_execute is now
correctness-only, and its whole apparatus -- the mechanism tournament
(exec codes 2..9), the unitary-chain tuner, the streaming-regime arena,
the PMU probe -- was deleted, not ported.  Rivals' L=36 mark: 0.059 s /
512 transforms = 115.2 us/xform.

### What was built (in dependency order)

1. **fft3d_chain, PER-VOLUME and IN-PLACE.**  Volume b runs all m steps
   before b+1 (volume-resident order: corpus SS10 SS3, via L23_matrixsimd
   / L23_rader ice_r4).  Steps run in place: MY pass structure already
   drains each x-plane fully into the L1 plane scratch before the y-lines
   rewrite it, and phase 2 was always in-place -- the same in-place-
   legality argument L23_rader made for their X pass.  No ping-pong
   buffer exists at all.
2. **Lazy map fused at the z-subloop loads** (the rival pipelines'
   winning shape, corpus SS10 SS2): every step leaves a raw spectrum in
   the state buffer; the NEXT step's z-subloop loads z and c at identical
   offsets, maps in registers, and feeds the transpose.  Each point is
   read exactly once there, so one fused site covers the volume; a
   pointwise trailing pass finishes step m straight into final_out.
3. **Split-form pair map**: two vunpck{lo,hi}pd per 128-bit lane pull two
   interleaved vectors into 8 re + 8 im, so |w|^2 and the Newton ladders
   run once per 8 points, and the same unpacks restore the layout (shape
   = L17_winograd ice_r4's mx8 crossed with L23_rader ice_r4's pair
   compression, both attributed).  Styles, all raced chain-shaped:
   - mA: vrsqrt14pd + 2 Newtons, d = fma(h,y,1) (L23_matrixsimd's trick),
     vrcp14pd + 2 Newtons.  Divider-free, ~25 vector ops / 8 pts, and a
     max(h, 1e-300) clamp against rsqrt14(0) = inf (their clamp too).
   - mB: one vsqrtpd instead of the 9-op rsqrt ladder (~18 ops + one
     divider op / 8 pts).
   - mS: semi-fused control -- standalone style-B sweep over the volume,
     then the unmapped body.
   Precision arithmetic (do not tier down): seed 2^-14 -> one Newton
   3.7e-9 FAILS the 1e-13/step budget -> two Newtons 1.4e-17 = full
   double.  Measured whole-chain drift m=64: **1.211e-14 (B=8) /
   1.296e-14 (B=1) vs tol 6.4e-12**, ~500x margin.  The rivals'
   float-seed tier is nominally legal at (36, 64) but buys nothing: the
   expensive part here is placement, not the seed.
4. **A plan-owned 2 MB chain arena** (MAP_HUGETLB, THP-madvise fallback;
   hugepage precedent: L64_blocked): state at +0, the current volume's c
   at +1 MB + 2048 (fixed 4K skew), c copied in per volume (~1/64 of a
   step; memoized by pointer at B=1).  Two wins: the L2 set behavior
   stops depending on the driver buffers' page-coloring lottery, and the
   create()-time race runs in the EXACT buffers the scored run uses.
   Before the arena, tuner said 118-127 while scored runs flapped
   127.4/127.3/128.0 vs 159.3/152.0 at B=1 (MKL steady ~325 both modes);
   after it, tuner and measurement agree within ~3% in every window
   observed (155.9->160.5, 112.1->112.8, 124.2->124.6).
5. **Chain-shaped tuner in create()**: admission of every candidate
   against an exact scalar-map reference chain (tol 1e-13; a rejected
   candidate cannot ship, the fallback is the scalar ref step), then a
   6-round interleaved race of {mA,mB,mS} x {pf0,pf1} over 11-step
   in-arena chains, 3% hysteresis in listed order, then the pind residue
   race {2112,-1,1536,2688} re-run under the chain shape.  Setup 0.25 s
   (B=1) / 0.32-0.39 s (B=8).

### Operation count

FFT unchanged: 232 FMA-port ops + 57 shuffles per 36-point line over PW
lanes, 2,814,912 flops/volume.  The map adds, per volume, 5832 pairs x
(mB: ~18 vector ops incl. 4 port-5 unpacks + 1 vsqrtpd) ~= 105k vector
ops + 5832 divider ops (~19 us of divider occupancy at 9.5 cyc rtp --
which is why divider-free mA stays in the pool), or mA: ~25 ops, 146k.
Map issue floor ~16-25 us/vol on the 2-port model; the measured fused
increment over the unmapped step is ~+16-22 us/step, i.e. the map rides
at its issue cost without a divider penalty showing.

### Measured on the node (tryout.sh, graded map-chain 36:8:64 and B=1;
### the W= workaround below; same-window MKL through the driver fallback)

- **B=8 QUIET: min 112.78 / median 112.84 us/xform (sd 0.03%)**, pick
  mB-pf1 (in-arena mB-pf1 112.1); second quiet run 113.08 (in-arena
  118.2).  MKL same case/core 289-291 -> **2.56x**; rivals' mark 115.2
  beaten in every quiet window.  Contended-lease windows: 157.7-160.5
  steady (MKL unchanged ~288 -- the contention hits an L2-resident chain
  ~40% and the DRAM-bound fallback ~0%; L17_winograd's clk probe note
  says 2.90 vs 3.3-3.5 GHz is part of that).
- **B=1: min 124.61 (sd 0.02%) / 129.36 across windows**, picks mB-pf1 /
  mA-pf1 (window-dependent, both admitted, arithmetically a wash).
- Correctness: single transform 3.586e-16 (B=8) / 3.591e-16 (B=1);
  whole-chain drift above; out.bin bit-repeatable (execute path is a
  fixed v1-pf1 pick).  Chain bits can differ across processes iff the
  race flips mA<->mB (different rounding, both ~500x inside budget; the
  harness only ever compares to numpy).
- Phase split (TSC, one CONTENDED window -- re-measure quiet before
  believing it): z=68.8% y=13.9% p2=17.3%.

### What did NOT work / what to know, with numbers

1. **tryout.sh is broken this round** exactly as L17_winograd ice_r4
   documented: prefix `W=$PWD/build/tryout/<name> ./tryout.sh ...` (the
   script expands $W two lines early under set -u), and the remote
   check.py map-chain call still crashes on an unexpanded '$W/c.bin' ->
   run check.py by hand on the shared FS after every run (also recovers
   the silently-skipped repeatability cmp).
2. **mS (semi-fused sweep) never beat fused mB**: 147.2 vs 144.3 and
   116.9 vs 118.2 in matched windows -- inside noise, so the extra
   read+write of the state buys nothing; kept as a race column since it
   costs one body.
3. **-DFFT36_SCHED again**: contended-window A/B, min 122.0 vs plain
   113.1 right after (MKL 288.6/289.2) -- consistent with ice_r2's 0.866
   vs 0.802 and corpus SS10.  Third strike; stop testing it.
4. **The pre-arena bimodality chase**: B=1 flapped 127 vs 152-159 with
   sd 0.05% inside each run and MKL steady.  The arena removed the
   tuner-vs-reality buffer mismatch (numbers in item 4 above); the
   residual 124.6 vs 129.4 spread tracks the lease window (clock +
   neighbor), not the plan.  Do not re-diagnose this as a code problem.
5. The pind race under the chain shape prices ~flat (117.7-119.3,
   145.6-146.3 across residues, 2112 kept by hysteresis) -- the fused
   z-subloop changed the alias geometry the ice_r2 -22% lived in.  Keep
   the knob, expect nothing from it here.

### Borrowed this round, named

- Map recipe skeleton (double-precision seeds + 2 Newtons, bias/clamp,
  "fuse where c streams"), the W= tryout workaround, and the run-check-
  by-hand protocol: **L17_winograd ice_r4**.
- Pair compression (one sqrt/reciprocal per 8 points) and the in-place
  chain-legality argument, per-volume order: **L23_rader ice_r4**.
- d = fma(m2, rsqrt, 1.0), the 1e-300 clamp, and the one-vs-two-Newton
  budget arithmetic: **L23_matrixsimd ice_r4**.
- Lazy map and volume-resident chaining doctrine: the rival pipelines
  (corpus SS10 SS2/SS3, ext/reference/fft_v4_solutions).
- Hugepage scratch: **L64_blocked** (ice_r1/r2 lineage).

### Predictions for the scoring window (so they can be scored)

- Quiet node -> **~110-114 us/xform at 36:8:64** (best dev 112.78), pick
  mB-pf1, pind 2112, spread <=1%.  If L36_pencilfused/L36_pfa did not
  ship a fused chain this round, the fallback map costs them ~+35 us and
  this entry takes the cell; if they did, expect all three within ~5 us
  again.
- B=1 cell ~124-128, pick mB-pf1 or mA-pf1 (a wash).
- rel_l2 fingerprints unchanged (3.586e-16/3.591e-16); chain drift
  ~1.2e-14.

### Next round

1. **Re-measure the TSC phase split in a QUIET window.**  If z really is
   ~69% fused, the map's 25/18 ops per 8 pts on top of the z-call's
   433-uop floor is the target: try the EAGER deferred-pair variant
   (map at phase 2's stores via a 2-deep SD pairing rotation -- phase 2
   has no shuffles and ~50 cyc/call of load/store slack) and A/B it
   against lazy.
2. The extract-store transpose-out (-72 port-5 shuffles per z-call, from
   the ice_r2 plan) is now MORE valuable: the fused z-subloop is the fat
   pass and the map added 4 port-5 unpacks per 8 points.  One build.
3. Adopt L17's clk512 probe into the description string so window class
   is readable off the leaderboard.
4. A T0 near-cursor on the c/S streams in the fused z-subloop (l1-style)
   -- the arena is 1.5 MB against 1.25 MB L2, so ~250 KB/step comes from
   L3 unprefetched.
5. Do NOT retry: SCHED (three strikes), mS-as-default (races say fused),
   float-seed map tiers (nothing to win, gate margin to lose), zy / nta
   / NT stores (unchanged verdicts).

## Round ice_r5

### Where it started, and the diagnosis that drove everything

r4 board: 111.425 us/xform, first by 0.5 us over L36_pencilfused.  Round
start re-measure (quiet window): 113.88 scored, MKL 289.5, ratio 0.393.
The r4 plan's first item -- re-measure the TSC phase split quiet -- gave
z=68.5% y=13.8% p2=17.7%: y and p2 sit AT their port floors (140 / 180
cyc/call vs ~144), the fused z-subloop runs ~700 cyc/call against a ~273
port-5 floor.  Three probes then eliminated every conventional suspect:
extract-store transpose-out (-72 p5/call) LOST, T0 near-cursors on the
S/c streams LOST, and mS/mP staging variants that move ~2 MB of extra
traffic per step priced within ~4 us of fused -- so the z stall was not
ports, not bandwidth, not prefetchable latency.  A 4-way TSC split (map
sweep in its own bucket, style mP forced) found it: the map sweep alone
is 36.5% of the step (~36-40 us), against a ~9 us issue floor.  5832
vsqrtpd-zmm per volume at ~18-20 cyc reciprocal throughput = 36-40 us of
DIVIDER occupancy.  The r4 map was divider-bound, and no placement of
the same arithmetic (fused mB, sweep mS, plane-staged mP) could hide it;
it had been masquerading as the "un-overlapped z stall" since r4.

### What ships (in dependency order)

1. **Hybrid map pairs** (mQ/mH, then folded into nF): alternate pairs
   between style B (vsqrtpd on the divider) and style A (rsqrt14+2N /
   rcp14+2N on the FMA ports), so the two units run CONCURRENTLY and the
   map floor drops from ~36-40 to ~19-20 us.  Staged hybrid mQ-pf1 beat
   mB-pf1 by 4.3% in-arena (119.0 vs 124.4) the first time it raced.
2. **The map moved to phase 2's stores** ("new protocol", styles
   nE/nF/nF2): dft36_xm applies the map via a 2-deep deferred-pair
   rotation at the x-pass's SDST sites (outputs stash one-deep, map two
   at a time, straight-line so GCC folds the branches).  c is read from
   **cperm, a per-volume PERMUTED copy in phase-2 store order** (built
   once per volume by cpfill, ~1.5% of a chain), so the fused map's c
   reads stream strictly sequentially -- this is what makes eager viable
   where L36_pencilfused ice_r4's eager attempt died on strided c (their
   143 vs 113).  Consequences: the map pass disappears as a pass; each
   step is ONE body call (FFT-then-map); the next step's z-subloop reads
   a single already-mapped stream with no map and no c; there is no
   firstfn/mpass split; the last step writes straight into final_out
   (saves the 746 KB copy).  nF-pf1 (1:1 hybrid at phase 2) = 114.5 vs
   mQ-pf1 121.1 vs mB-pf1 124.1 in one window; 124.7/135.5/141.5 in
   another -- an ~8% step win over the r4 shape, consistent.
3. **Arena grown to 4 MB**: S (+0), natural c (+1 MB+2048), cperm
   (+2 MB+2048) in one span; a chain touches S + one c form = 1.5 MB as
   before.  The tuner races BOTH protocols in-arena (prot[] flag per
   candidate; new-protocol candidates race with cperm, old with natural).
4. **Pool-order fix**: mB-pf1 (now nF-pf1) listed first.  This round
   caught a live 3%-hysteresis mis-pick: mA-pf1 kept at 127.88 vs
   mB-pf1 124.09 -- the better candidate missed the bar by 0.05%.
5. Pool shipped (order = hysteresis): nF-pf1 first after mQ/mQ2
   incumbents... final order mQ-pf1, mQ2-pf1, nF-pf1, nF2-pf1, nE-pf1,
   nF-pf0, mQ-pf0, mH-pf1, mB-pf1, mA-pf1; nF-pf1 won every window
   raced (5 of 5, margins 0.5-10%).

### Operation count

FFT unchanged: 232 FMA-port ops + 57 shuffles per 36-point line over PW
lanes, 2,814,912 flops/volume.  Map (nF), per volume at phase 2's
stores: 2916 pairs style B (~12 FMA-port + 4 p5 unpacks + 1 vsqrtpd +
1 rcp14) + 2916 pairs style A (~21 FMA-port + 4 p5 + rsqrt14 + rcp14).
Divider occupancy halves to ~2916 x 18-20 cyc ~= 18-20 us/vol, running
concurrently with the A-ladder FMA work.  Plus one cpfill per volume
per chain (11664 vector moves, 746 KB permuted copy, amortized /64).
Precision unchanged: both styles are double-seed + 2 Newtons, ~2-3 ulp
per application; measured whole-chain drift m=64: 1.229e-14 (B=8) /
1.352e-14 (B=1) vs tol 6.4e-12, ~500x margin.  (Pair-style assignment
differs across nE/nF/nF2, so chain bits can differ across processes iff
the pick flips -- same situation as r4's mA/mB, all inside budget.)

### Measured on the node (tryout.sh; W= workaround still needed, check.py
### still run by hand -- the r4 breakage is unchanged in r5)

- Round start baseline: 113.88 min / MKL 289.5 (quiet), ratio 0.393;
  contended windows 128.9-129.8 at MKL 282-295.
- **Best scored this round: 103.26 min (B=8, quiet moment, MKL 288.7,
  ratio 0.358)**, pick nF-pf1; contended-class windows 116.6-117.0 at
  MKL 286.6-306.7 (ratio 0.38-0.41).  In-arena race, matched windows:
  nF-pf1 114.2-124.7 vs mQ-pf1 119.6-135.5 vs mB-pf1 124.1-141.5.
- B=1: 129.1 scored (MKL 326.4, loaded window; r4 was 124.6 at MKL 325
  in a quieter one), pick nF-pf1 (in-arena 123.4, next candidate +4%).
- TSC under nF (loaded window): z=32.8% y=14.5% p2=50.8% -- phase 2 now
  carries FFT + map + both streams at ~465 cyc/call vs ~200 floor.
- Correctness: single transform 3.586e-16 (B=8) / 3.591e-16 (B=1),
  chain drift above, repeatable across runs.  Setup 0.30-0.45 s.

### What did NOT work, with the number that killed it

1. **Extract-store transpose-out (xs)**, the r2/r4 plan's pet idea, is
   dead in EVERY shape: fused mBx-pf1 130.88 vs mB-pf1 124.09 (+5.5%),
   mAx-pf1 134.11, staged mQx-pf1 152.42 vs mQ-pf1 146.78 (+3.8%).
   Codegen verified correct (432 memory-form vextractf64x2 in the
   binary), so the mechanism is real: the z-subloop is stall-bound, not
   port-5-bound, and 144 16-B stores/call eat the store-buffer entries
   the OOO window needs to hide those stalls.  Do not retry.
2. **T0 near-cursors on the fused z-subloop's S/c streams (zpf)**:
   mBz-pf1 140.78 vs mB-pf1 124.09 (+13%), mBxz-pf1 146.72.  Third
   panel confirmation that prefetch is a tax on cache-resident chains
   (after ice_r3's and L36_pencilfused r4's).  The stall it aimed at
   turned out to be the divider anyway.
3. **Plane staging WITHOUT the hybrid (mP)**: a wash vs fused mB
   (134.62 vs 137.21 one window, 126.40 vs 125.52 another) -- staging
   alone buys nothing, confirming the divider diagnosis.
4. **1:2 divider:ladder ratio (mQ2/nF2)**: window-dependent wash at the
   sweep (mQ2 144.81 vs mQ 146.78 once, then 120.7 vs 121.1), slightly
   worse at phase 2 (nF2 115.0-115.8 vs nF 114.2-114.5 at B=8, 128.5 vs
   123.4 at B=1).  1:1 ships first; both stay in the pool.
5. mst residue pinning for the staged shapes: no measurable move; kept
   by analysis (2112 puts the colliding rows 25 apart for the z-shape).

### Borrowed this round, named

- The plane-staged map shape (mapplane through an L1 scratch) and the
  eager-map-fails-on-strided-c post-mortem that motivated cperm:
  **L36_pencilfused ice_r4**.  Their record also saved me from
  re-testing eager-with-strided-c ("143 vs 113").
- The divider/FMA-ladder port-balancing and the deferred-pair rotation
  at SDST: own analysis (the rotation idea was queued in my r4 record);
  the split/pair map machinery it runs on is r4's, itself assembled
  from L17_winograd / L23_rader / L23_matrixsimd ice_r4.

### Predictions for the scoring window (so they can be scored)

- Quiet node -> **~100-106 us/xform at 36:8:64** (best dev 103.26 at
  MKL 288.7), pick nF-pf1 (nE-pf1/nF2-pf1 flips possible, both within
  1%), prot=p2map-cperm in the description, pind 2112, spread <=1%.
- If L36_pencilfused ships only their r4 shape (~112), this takes the
  cell by ~8 us; the rivals' 115.2 falls by ~12 us.
- B=1 cell ~123-129 (window-dependent), pick nF-pf1.
- rel_l2 fingerprints unchanged (3.586e-16 / 3.591e-16); chain drift
  1.229e-14 (B=8) / 1.352e-14 (B=1).

### Next round

1. **Phase 2 is the frontier now: 50.8% of the step, ~465 cyc/call vs a
   ~200 floor.**  The ~250-cycle stall/call sits on the strided
   in-place x-pass now carrying the map.  Candidates: software-pipeline
   pairs of x-calls (the map's ladder latency wants independent work;
   note geom-r8's sp2 rejection was for the PURE FFT balance -- the map
   changed it); split the deferred pairs across a longer rotation so
   MAPPAIR issues further from its stores; check whether the 36
   x-streams' L2-set behavior under the new store pattern wants a
   different pf distance (only pf=1/0 are compiled).
2. The plain z-subloop reads the mapped state at ~300 cyc/call vs a 201
   p5 floor -- second-order, but it is the only other slack left.
3. cpfill is a strided-read sweep (~1.5% of a chain); vectorizing it
   per-plane or folding it into step 1's z-subloop is small but free.
4. Do NOT retry: xs in any shape (numbers above), prefetch cursors on
   chain streams (three strikes panel-wide), SCHED (three strikes),
   1-Newton / float-seed map tiers (budget arithmetic unchanged), zy /
   nta / NT stores.
