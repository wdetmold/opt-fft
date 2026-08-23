# L23_matrixsimd — strategy record (ICE LAKE panel)

Continuation of this entry's records from the single-thread panel
(`bench/geom/strategies/L23_matrixsimd.md`, rounds panel_r6..r11) and the
multicore panel (`bench/mt/strategies/L23_matrixsimd.md`).  Those files are
the full history — the operation-count argument that settled the arithmetic
(L23_rader r6), the bit-class rules, the do-not-retry list (NT stores,
pf=1, uniform pipelining, tail-paced schedule — all raced and rejected on
the CLX node), and the sbw bandwidth instrument.  This file records what
happens on the Ice Lake panel (bare-metal Xeon Gold 6326, 2×512-bit FMA
pipes, graded chain workload, `cases.txt` 23:16:165).

NOTE ON ice_r1: this entry's ice_r1 agent died in the round's worker crash
storm (verdict §3: 15 of 19 agents killed at launch by host memory
exhaustion), so ice_r1 scored the UNMODIFIED panel_r11 geom code and no ice
record was written.  ice_r1 measured it at 39.584 us/transform on the
graded chain (B=16, m=165), a statistical dead heat with L23_rader (39.142,
1.01×), both ~5.6× ahead of the best library (ducc0 220.216).  ice_r2 below
is therefore this entry's FIRST round of actual ice-panel work.

## Round ice_r2

### Where this round started

39.584 us/t on the graded chain (spread 1.8%), picked variant za, tuner
telemetry `tune[pick=37.74 inc=40.33 us/t nv=8]`.  Diagnosis, from the
ice_r1 verdict and L17_matrixsimd's ice_r1 record (the round's only
measured playbook on this node):

1. **The tuner never saw the graded workload.**  The scored unit is a
   chain: execute(B=16) → driver-side unitary scale of the whole 2.97 MiB
   output → output becomes the next input, ping-ponging two destination
   buffers, all three buffers L3-resident (8.9 MiB against 1.25 MB L2 /
   24 MB L3).  My stage-1 arena was 8 volumes, fixed src→dst, no scale
   pass: pick=37.74 predicted for a cell that scored 39.584 (+4.9%
   optimism, verdict §4a).  L17's chain-shaped tuner was accurate to +1.3%
   (§4b).
2. **The whole plan was selected on an unramped core.**  My clk telemetry
   read clk512/256 = 2.90/2.90 GHz — the base clock — while ramped runs on
   the same silicon read 3.30/3.50 (verdict §0a).  schedutil + a short
   create() = rankings taken while the core ramps.
3. **Candidate lists shaped for CLX.**  Head of the resident walk was flat
   (v8) from CLX/node-r10 evidence; the ice_r1 node arena picked za (v40)
   by −6.4%.

### What was changed (no exec variant, kernel, table or bit class changed)

1. **150 ms clock-settle spin** (`l23_settle`, 8 independent 512-bit FMA
   chains) at the top of the tuner, so rankings and the clk probes run on a
   ramped core.  Adopted from L17_matrixsimd ice_r1, originally
   L17_winograd's protocol.
2. **Chain-shaped tuner stage (stage 1ch, 9 ≤ batch < 64 — the graded
   cell).**  Candidates are timed under the driver's own RUN_UNIT loop:
   nv = min(batch,16) volumes, 6 chained steps per timed unit, unitary
   scale of the whole destination after every execute, output fed back as
   the next source, ping-ponging two destination arenas (a third tuner
   buffer `tq` was added).  Licence warmup ≥ 1.5 ms per candidate, min of
   3 units, two fixed-order sweeps, 2% hysteresis — the r8 discipline
   unchanged, only the timed unit is now the scored one.  The walk list is
   the 512-bit cached family plus one NT and one 256-bit sentinel row
   (mirroring the streaming walk's design; the settled families would eat
   2/3 of the walk).  The (pf,pw) knob grid on the winner also runs
   chain-shaped (same rows, 3% margin).  Old stage 1 + stage 2 now cover
   batch < 9 only; the batch ≥ 64 streaming walk is untouched.  Adopted
   from L17_matrixsimd ice_r1 (its stage 1g).
3. **Walk head moved to za** (fastest-known-head rule, my r10 lesson): za
   beat flat in every ice_r2 chain walk on the node (43.98 vs 45.43, 48.92
   vs 50.50, 55.75 vs 57.44 us/t) and in ice_r1's arena (−6.4%).  za+dz
   straddles the 2% margin against za (won one window by 2.2%, lost
   another by 1.2%); with za at the head the near-tie resolves to za in
   every process.
4. Housekeeping: `-DL23_VERBOSE_BUILD` hook (tryout.sh cannot pass env
   through ssh — L17's hook), chain-stage telemetry tagged `tune[ch ...]`
   so the leaderboard shows which stage picked.

### Operation count

Unchanged from panel_r11: 594 real flop/line, 943 kflop/volume, 297 vector
FP ops per chunk, 409 zmm chunks flat (414 za).  On this 2-pipe machine the
port floor is ~74k cycles/volume ≈ 26 us at 2.9 GHz (Y/Z chunks are
port-5-bound: 59 p5-only shuffles per chunk against 308 FP ops wanting
ports 0+5), so the 40.4 us dev-window cell still carries ~35% of
issue/feed slack — see Next.

### Measured on the node (tryout.sh = a80n0, leased core, graded chain
### L=23 B=16 m=165 unless stated; dev windows, not the scoring quiet)

- ice_r1 code re-based reference: scored 39.584 (quiet window).
- Settle + chain tuner + scheduling pragma (first build): min 41.899,
  median 46.736, sd 5.49% — then with verbose in a cleaner window: pick
  za+dz at 47.83 in-walk, graded min 42.639 / median 42.668, sd 0.18%.
- **Pinned-variant pragma A/B/A** (forced za, back-to-back windows, all
  sd ≤ 0.04%): sched 41.872 / no-sched 41.104 / sched 41.760 us/t —
  **the pragma costs +1.7%; REJECTED** (see below).
- **FINAL SHIPPED STATE** (no pragma, za head): chain walk picked za 43.98
  (flat 45.43, za+dz 46.41, tail-paced 47.77, NT 70.02, 256-bit 55.65);
  knob grid 00/01/20/21 = 47.71/45.57/45.31/46.16 → pf=0 pw=1 that window;
  graded run **min 40.416 / median 40.530 us/t, sd 0.29%**; rel_l2
  3.798e-16, chain-165 closed-form 9.288e-15 (tol 1.3e-11);
  bit-repeatable.  MKL same case/core: 230.8 us/t (we are 5.7×).
- B=1 chain m=165 (batch<9 path, untouched picks): **38.580 us/t**, sd
  0.01%, rel_l2 3.767e-16, chain 9.331e-15, repeatable; MKL 228.6.
- B=64 chain m=165 (streaming stage, 35.7 MiB > L3): min 48.364 / median
  49.857, sd 3.66%, rel_l2 3.803e-16, repeatable.
- Dev-window offset caution (L17 ice_r1's, confirmed): same binary+pick
  read 40.4–42.7 across today's windows; ice_r1's quiet-window score for
  the same za pick was 39.58.  Within-window contrasts are the trustworthy
  ones; expect ~38.5–40.5 scored.

### What did NOT work / negative results with numbers

- **Pre-RA scheduling pragma (schedule-insns + sched-pressure): +1.7% on
  the graded cell — the SIGN FLIPS relative to L17_matrixsimd's −7.7% on
  the same node and toolchain.**  Pinned A/B/A above; also visible
  unpinned (42.64 sched vs 40.77 no-sched, different picks).  Mechanism:
  L17's chunk source is phase-serial ROLLED loops the scheduler can
  profitably mix; my hot kernel (chunk23p) is fully-unrolled straight-line
  code whose 23 independent accumulator chains already expose the ILP, so
  sched1's remix only adds register pressure.  Kept as an opt-in
  `-DL23_SCHED` hook.  Lesson for the panel: corpus §10's "~20% on prime
  passes" GCC cure is shape-dependent — test pinned before adopting.
  (Side cost while it was in: compile time roughly doubled.)
- **NT planes on the L3-resident chain: catastrophic**, 70.02–110.10 vs
  43.98–55.75 us/t in the same walks — matches L17's 28.9-vs-14.29 finding;
  seventh round running that NT pays only at true DRAM streaming.  The
  sentinel row stays (it costs ~40 ms of tuner time).
- **256-bit on this 2×512-pipe machine loses ~13–16%** (55.65 vs 43.98;
  64.61 vs 55.75 in the no-sched walk) — consistent with the verdict §4.8
  ruling; sentinel row only.
- **pw straddles in-regime** (pw=1 won one knob grid by 4.5%, lost two
  others by 3.5% and 3.4%): unlike the CLX streaming cells where pw=1 was
  worth ~12%, the chain's L3-resident out-plane RFO is only marginally
  hideable.  The grid is per-process and changes no bits, so the straddle
  is cosmetic; L13_rader's "pw only past L3" gate would have pinned it to
  0 — consider adopting if the monitor's runs show pw flip-flopping in the
  telemetry.
- **The chain-walk absolute level is window-sensitive in dev slots**
  (candidate tables shifted ~8 us/t between two adjacent processes with
  identical code), and tuner absolutes sit ABOVE the graded run in the
  same process (43.98 in-walk vs 40.42 graded: 6-step units pay
  per-candidate cold starts the 165-step graded chain amortizes).  Use
  in-walk numbers for ranking only, never as level predictions.

### Borrowed this round

- Chain-shaped in-regime tuning, clock-settle spin, verbose-build hook:
  **L17_matrixsimd ice_r1** (which credits L17_winograd for the settle
  protocol).
- The scheduling pragma experiment (rejected here): **L17_matrixsimd
  ice_r1** via corpus §10 GCC cure #5.
- pw-gate caution under the chain: **L13_rader ice_r1** via verdict §4d.
- Fastest-known-head + hysteresis discipline: my own geom r8/r10 record.

### Next round

- **The kernel is the remaining ~35%.**  Port floor ≈ 26 us/volume at 2.9
  GHz (with the port-5 shuffle tax; ~21 us without it), graded cell ≈ 40.
  Two candidate mechanisms, in order:
  1. Port-5 relief in the Y/Z chunks: 48 tile-transpose shuffles + 11 MULI
     swaps per chunk are p5-only on ICX and displace the second FMA pipe
     (~30 cycles/chunk ≈ 2.9 us/volume of theoretical tax).  The verdict
     recommends L17_rader's ymm 4×4 tile transposes (dual-issue p1/p5) at
     L=17; port here only after it wins there, and heed L13_rader's
     warning (p5 relief that lengthens load chains loses, +1.36 us).
  2. ZIPP-style merged P/R sweep (load each input pair once, −22
     loads/chunk, hand the issue queue explicitly interleaved chains).
     Needs the k-blocked shape (~31 live registers unblocked = spill
     cliff), and the pragma result above says do the interleave BY HAND in
     the generator/source, not via sched1.
- Chain-shape the batch ≥ 64 streaming walk too if that cell is ever
  scored (B=64 chain = 35.7 MiB is genuinely streaming AND chained — a
  regime neither existing tuner stage matches).
- If the monitor's telemetry shows the za head or pw knob flip-flopping
  across its three runs, pin pw by the L3 gate and re-check.
