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

(NOTE: this entry's ice_r3 agent left no record — ice_r3 scored the
unmodified ice_r2 code at 39.761 us/t, second to L23_rader 39.502, with
tune[ch pick=50.73 inc=50.73 nv=16] and clk512/256 = 2.90/2.90 in the
description: the settle spin ran but the clk probes still read base clock
in the scoring window.  The pick held za as designed.)

## Round ice_r4

### The task changed under us; this round is fft3d_chain

The graded step is now the full rival step, state <- (z+c)/(1+|z+c|) with
z the RAW (unnormalized) FFT of the state — no driver-side unitary scale
in map mode — and the driver times an exported `fft3d_chain(plan, x0, c,
final_out, m)` weak symbol as the whole unit.  Entries without it pay
fft3d_execute plus a driver-side map pass; MKL (fallback-only by
construction) measured that pass at +31..35 us/t on the graded cell
(231 -> 262–266) in this round's windows.  Everything below is about
owning the chain; no exec variant, kernel, table, tuner stage or bit
class of the fft3d_execute path changed.

### What was built

1. **Volume-resident chain order** (corpus §10 §3 consensus: "iterate each
   volume through all m steps while cache-resident; never sweep passes
   across volumes").  fft3d_chain runs volume b through all m=165 steps
   before touching volume b+1: state ping-pong (2 x 190 KiB plan buffers,
   skewed 1.5 KiB apart), the volume's own c (190 KiB) and t1 (199 KiB)
   all stay L2-resident (~780 KiB against 1.25 MB), where the old
   batch-per-step order sloshed 8.9 MiB through L3.  Legal because both
   the FFT and the map act volume-locally, so the reorder changes no
   values.  Step m's Z pass writes its mapped planes straight into
   final_out's volume: no epilogue, no copy, x0 never written.
2. **FFT body per step**: exec_zf's per-volume interior verbatim (za
   layout, pinned kernel, X-first, no prefetch knobs — every knob has
   only ever been uop tax on L2-resident lines).
3. **Map fused at Z-CHUNK granularity, L1-hot**: Z chunk t finalises
   out-plane rows ky = f0..f0+3 = one contiguous 184-double span (exactly
   23 zmm), which is mapped in place immediately, so span t's divides
   issue between chunk t+1's FMA bursts.  The off23 overlap row (ky=19)
   is stored raw over its mapped copy by the last chunk and re-mapped
   bit-identically (+4.3% map vectors; keeps every span whole-vector).
4. **Map arithmetic (mv=0, the shipped default)**: w = z+c; |w|^2
   duplicated into both lanes (1 vpermilpd + 1 add); clamped to 1e-300
   (rsqrt14(0)=inf would NaN the Newton ladder; the clamp is exact for
   the result since sqrt(1e-300) vanishes in 1+|w|); vrsqrt14pd seed + 2
   Newton on the FMA pipes; d = fma(m2,r,1.0); ONE exact vdivpd w/d.
   This is corpus §10 §2's 4/7-convergent shape (1000f989's mapF,
   8dc1a96d): burn the divider once per point, Newton for the rest.
   **Precision arithmetic for the gate (m=165, tol 1.65e-11)**: seed err
   6.1e-5 -> 1 Newton 5.6e-9 (FAILS the 1e-13/step budget, do not tier
   down) -> 2 Newtons 4.7e-17 = full double, ~2-3 ulp/application.
   Measured chain-165: **3.63e-14, ~450x margin**.  The rivals'
   float-seed tier is legal at this (L,m) but buys ~nothing once the
   divider is the marginal resource; not taken.
5. mv is COMPILE-TIME (-DL23_MV): the variants are different bit classes,
   so a plan-time wall-clock race could flip bits across processes.

### Operation count

FFT unchanged (943 kflop/vol, za = 138 X + 276 plane chunks).  Map adds
per volume 3174 zmm map-vectors (529/plane x 23 planes + 4.3% overlap)
x (~14 vector ops + 1 vdivpd) ~= 44k vector ops on the pipes + 3174
divides.  Divider occupancy ~= 3174 x ~16 cyc ~= 51k cyc ~= 17.5 us/vol
at 2.9 GHz — more than a third of the step — which is why WHERE the
divides sit in program order decided the round (below).

### Measured on the node (tryout.sh = a80n0, leased core, graded chain
### L=23 B=16 m=165, dev windows; same-window contrasts only)

- **SHIPPED STATE: min 44.81–45.52 us/t across four windows (best-window
  sd 0.03%), B=1 44.51 us/t (quiet window)**; single-transform rel_l2
  3.798e-16, map-chain-165 rel_l2 3.631e-14 (tol 1.65e-11), single AND
  chain outputs bit-identical across processes.  MKL same case/core:
  262–266 us/t through the fallback (we are 5.9x).
- Map placement ladder (mv=0): per-plane epilogue after the Z pass
  46.00/46.15 (A/B/A against mv=1 47.56) -> chunk-granular fusion
  44.81/45.25: **-1.2 us/t for moving the SAME divides between the FMA
  bursts instead of a 133-divide block the divider serialises.**
- Map variant race (chunk-fused, same windows): mv=0 (rsqrt-Newton + one
  vdivpd) 45.25 vs mv=1 (all-Newton, rcp14 ladder, no divider) 46.61
  (+3.0%); as plane epilogue 46.0–46.1 vs 47.6–48.0 (+3.3%); mv=2
  (hardware vsqrtpd + rcp ladder) 53.38 (+18%).  The consensus shape
  wins here too; rsqrt14/rcp14 are NOT microcoded-slow on this
  bare-metal tier (1760b1bf's VM claim does not transfer — mv=1 loses
  on FMA-port pressure, not seed latency).
- d = fma(m2,r,1.0) merge (saves one add/vector): 45.52 in its own
  window, kept as a >= 0 change (one op fewer, same error class, chain
  residual unchanged at 3.631e-14).

### What did NOT work / traps hit, with numbers

- **Hardware vsqrtpd for |w| (mv=2): 53.38 vs 45.25 us/t.**  Two divider
  ops per point saturate the one divider; the sqrt is the op to Newton
  away, the divide is the one to keep.
- **All-Newton (mv=1) loses ~1.4 us/t in both placements** even though
  it frees the divider entirely: the second (rcp14) ladder spends more
  FMA-pipe slots than the hidden divide costs.
- **A per-plane map epilogue costs +1.2 us/t vs chunk-granular fusion**:
  133 back-to-back divides stall the divider (~2.1k cyc/plane) faster
  than OoO can reach the next plane's FMA work.  Placement, not count.
- **Dev-window trap (cost ~40 min): a B=1..4 vs B=8..16 "cliff" (51.5 vs
  45.2 us/t, sd 0.02%!) that was pure core-lease contention** — MKL in
  the same windows showed the same +37 us/t at B=1, and a fresh window
  read B=1 at 44.51.  A tiny sd does NOT certify a window: it can be a
  STABLY contended neighbor.  Cross-B (= cross-window) comparisons lie;
  only same-window A/B counts (the r2 lesson, now with a sharper edge).
- **tryout.sh has two bugs this round** (cannot fix, script is shared):
  line 36 reads $W before line 38 sets it (set -u aborts) — work around
  with `W=<ice>/build/tryout/<name> ./tryout.sh ...`; and the remote
  check.py receives a literally-unexpanded '$W/c.bin' for --cin, so the
  map-chain check inside tryout always dies with FileNotFoundError —
  run check.py yourself against build/tryout/<name>/{in,out,c}.bin
  (shared FS), and do the two-run repeatability cmp manually on a
  leased core (the && chain skips it after the check.py crash).

### Borrowed this round

- The map shape (rsqrt-Newton magnitude + ONE exact vdivpd) and the
  divider-hiding doctrine: **corpus §10 §2 [consensus 4/7]** (1000f989's
  mapF, 8dc1a96d); "the exact final divide protects the gate" is theirs.
- Volume-resident chain order: **corpus §10 §3 [consensus]**.
- The zero-clamp before rsqrt14: own derivation (rsqrt14(0)=inf NaNs the
  ladder; the corpus never mentions it — their graders' data never hit
  w = 0, ours must not be able to).
- Skewing the two state buffers apart: corpus §10 §3 4K-aliasing hygiene.
- Do-the-budget-arithmetic-per-(L,m) before precision tiering:
  PANEL_BRIEF ice_r4 task section (and it came out "don't tier").

### Next round

- **Map cost is now ~5.2 us/t of the ~45; the FFT body is the rest.**
  The r2 diagnosis stands: ~35% issue slack against the ~26 us port
  floor.  Port-5 relief in the Y/Z chunks (48 tile shuffles + 11 MULI
  p5-only ops per chunk) is worth ~2.9 us/vol of theoretical tax and now
  ALSO frees pipe slots the map's Newton ladder competes for — the two
  compound.  Check L17_rader's ymm 4x4 tile transposes at L=17 first
  (r2 next-item 1 unchanged).
- Fuse the map into the Z-chunk STORE TAIL (values still in registers):
  saves the 23-vector store->reload round trip per span, at the price of
  ~4 map temps against 23 live accumulators (spill risk).  Worth one
  pinned A/B; expected < 1 us.
- Race dz (deferred-Z) and flat-vs-za INSIDE the chain body: the chain's
  L2-resident economics differ from every regime those were tuned in.
- If the quiet window still shows the divider exposed (score well above
  ~44), split each 23-vector map span into two half-spans issued around
  the next Z chunk.

### Late addendum (same round): deferred-Z inside the chain — RACED, LOSES

The "race dz inside the chain" next-item was cheap enough to do now.
-DL23_CDZ (exec_zfd's schedule: Y(x+1) between Y(x) and Z(x), pb
double-buffered) A/B/A/B on the graded cell: CDZ 61.60 (contended
window, discard) / plain 44.86 / CDZ 45.72 / plain 45.34 us/t — dz loses
~0.4–0.9% in the comparable adjacent windows.  Consistent with the map
now sitting between the Z chunks: the store->load junction dz was built
to break is already separated by ~390 map uops per span.  Hook kept
(-DL23_CDZ) for forced experiments; default stays plain.  The flat-vs-za
race inside the chain remains untried (windows too noisy today to
resolve an expected <1% contrast; do it in a quieter round).

## Round ice_r5

### Where this round started

ice_r4 scored 44.872 us/t at the graded cell -- second, 18% behind
L23_rader's 38.105, after being a dead heat with it for three FFT-only
rounds.  The whole gap was map handling: rader's ice_r4 record lays out the
mechanism list plainly, so this round is deliberate adoption, not invention.

### What was changed (chain path only; no exec variant, kernel, table,
### tuner stage or bit class of fft3d_execute changed)

1. **Map fused into the Z pass's TILE STORES, register-level** (adopted from
   L23_rader ice_r4; my own r4 "Next" list had it unexecuted).  chunk23p
   grew compile-time (cq, dc, mp) params; mp=1 swaps the store tail for
   TILEM, which maps each transposed tile pair against c in registers and
   stores mapped values once.  The ice_r4 store-raw-span/reload/map-in-place
   pass is deleted -- ~380 KiB of L1/L2 store+reload round-trip traffic per
   step gone.  Row/column overlap slots recompute bit-identical raw values,
   so their re-mapped stores are bit-identical (cmp-verified).
2. **Pair-compressed map arithmetic** (rader ice_r4, extending the rivals'
   PW_CORE): |w|^2 of two output vectors compressed into ONE via
   unpacklo/unpackhi/add, ONE divider-class op per 8 points (ice_r4 burned
   one vdivpd per 4), expansion via 2 vpermilpd-imm + 2 muls.
3. **Map variant raced pinned, and the ranking FLIPPED vs rader's**: the
   shipped default is L23_MAPV=1 = rsqrt14+2-Newton |w| ladder (zero-clamped)
   + dn=fma(m2,r,1)+ ONE exact vdivpd -- against rader's winner (MAPV=0,
   hw vsqrtpd + rcp14+2-Newton).  Same-lease A/B/A x3, sd <= 0.02%:
   v1 36.90/36.98/37.09 vs v0 38.33/38.54/38.60 us/t, v1 by 1.5 us 3/3.
   Rader's own race in ITS tail read hw-sqrt 39.05 vs rsqrt-ladder 39.8.
   Mechanism (best reading): in this tail the vsqrtpd->rcp-ladder chain is
   one serial step longer, while the rsqrt ladder's FMA ops interleave into
   the adjacent chunks' bursts and the single late vdivpd hits an otherwise
   idle divider.  Panel lesson: map-variant rankings are STORE-TAIL-SHAPE
   dependent; pin-race them in situ, do not copy a ranking.
4. **In-place chain on ONE za-padded state volume** (rader's in-place +
   state-padding findings, fused and taken further).  State rows padded
   23->24 complex, planes 1104 doubles = 138 whole lines: EVERY X-pass load
   of steps 2..m is 64-byte aligned (rader padded only plane bases, -1.2 us
   there; at row level the per-chunk 3-of-4 split-line load tax vanishes),
   and 5 of 6 Z-pass tile stores into the state are aligned too.  Steps
   2..m-1 run in place (X pass drains the state into t1 before the plane
   loop rewrites it -- rader's legality argument), so the ping/pong pair is
   gone and the per-volume working set drops ~790 -> ~600 KiB of L2.
   Three chain-step instantiations keep every stride compile-time
   (flat->padded, padded->padded in place, padded->flat; L45_pfa r8
   lea-spill rule via rader's three-instantiation pattern).
5. The -DL23_CDZ chain hook was removed by the restructure (it measured a
   0.4-0.9% loss in my r4 addendum and the map now sits exactly where its
   mechanism needed a gap).  -DL23_MAPV=0 kept as the A/B hook; the old
   L23_MV span-map knob is gone with the span map.

### Operation count

FFT body unchanged (943 kflop/vol; za = 138 X + 276 plane chunks).  Map:
1656 map2st calls/volume (72/plane x 23; 13248 mapped point-slots vs 12167
unique, +8.9% overlap re-map), each ~17 FMA-port ops + 4 port-5 shuffles +
1 vdivpd per 8 points ~= 28k pipe ops + 1656 divides/volume, vs ice_r4's
~44k pipe ops + 3174 divides + the full store+reload round trip.

### Measured on the node (tryout.sh = a80n0, leased cores, graded chain
### L=23 m=165; same-window contrasts unless noted)

- **SHIPPED STATE: B=16 36.886/36.909 us/t (two runs, one lease, sd <=
  0.04%); B=1 36.870/36.928; B=64 37.035.**  Single rel_l2 3.798e-16;
  map-chain-165 rel_l2 3.824e-14 (B=16) / 2.901e-14 (B=1) / 2.854e-14
  (B=64), tol 1.65e-11 -- ~430x margin.  Single AND chain outputs
  bit-identical across runs at B=1 and B=16.  MKL same case/core through
  the fallback: 266-267 us/t (we are 7.2x).
- Ladder within the round (graded cell): r4 shipped 44.81-45.52 ->
  map@store + compression + in-place padded state (MAPV=0) 38.38-38.68 ->
  MAPV=1 flip 36.89-37.09.  Total -17.7% vs the r4 scored 44.872; crosses
  L23_rader's scored 38.105 and the rivals' 39.0-equivalent (0.103 s
  graded point ~= 0.098 s projected).
- Batch-invariance is a design property now (per-volume L2-resident chain):
  36.9 flat across B=1/16/64.

### What did NOT work / traps hit, with numbers

- **Rader's map arithmetic (MAPV=0) in MY store tail: +1.5 us/t, 3/3
  pinned.**  See item 3 -- the second sign-flip this entry has documented
  when transplanting a tuning result between differently-shaped code
  (ice_r2's scheduling pragma was the first).
- **B=1 "cliff" (43.5 vs 38.4, sd 0.02%) in the first B=1 window was
  contamination, not the rader mod-4096 pathology**: MKL in the same window
  read +30 us above its known level, and a same-lease B=1/B=16/B=1 A/B/A
  resolved B=1 to 38.74 (later 36.87-36.93).  The r4 lesson holds with its
  sharper edge: a tiny sd does not certify a window; MKL-in-window is the
  cheap contamination probe.
- **tryout.sh still has the two r4 bugs** (shared file, cannot fix): W
  unbound at line 36 under set -u -- prefix `W=<ice>/build/tryout/<name>`;
  and the remote map-chain check always dies on a literally-unexpanded
  '$W/c.bin' (single quotes inside the local command substitution survive
  to the remote shell) -- run check.py yourself on the shared FS and do the
  two-run cmp manually on a leased core.  Also (rader's note confirmed):
  tryout regenerates in.bin/c.bin at whatever batch you pass -- check the
  chain BEFORE running a different batch, or keep per-batch copies.

### Borrowed this round

- Register-level Z-store map fusion, pair-compressed |w|^2, in-place state,
  state padding, three-instantiation stride discipline: **L23_rader
  ice_r4** (which credits the rivals' 1760b1bf PW_CORE for the fusion
  shape).  This round is that record, executed here, plus two extensions.
- Row-level (not just plane-base) za padding of the state volume: own
  extension, reusing this entry's existing za/t1 machinery (zoff24 X pass).
- The MAPV pinned-race discipline: my own ice_r2 pragma lesson, reapplied.

### Next round

- **Fully aligned Z stores into the padded state**: replace the m0=19
  overlap tile with an m0=20 tile whose 4th column is a synthesized
  duplicate (pad column, finite, never read), making 6/6 tile stores
  aligned on padded-output steps.  Untried; expect < 0.5 us.
- Port-5 relief in the Y/Z chunks (r2 next-item, still open): the map
  added 4 p5 shuffles per 8 points on top of the 48 tile shuffles + 11
  MULI per chunk; L17_rader's ymm 4x4 tile transposes remain the candidate.
- The X pass still loads each input pair twice (u and w sweeps) --
  rader's next-item 2; a single-load X kernel would halve X-pass L1
  traffic, now that all its loads are aligned.
- If rader adopts the same set and the margin closes to noise, the FFT
  body itself (~32 us of the 36.9) is where the round after goes: the
  ~26 us port floor still leaves ~20% issue slack.
