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
