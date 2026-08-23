# L23_rader — strategy record (ice panel)

Note on missing history: this entry carried its full record in the multicore
panel (`bench/mt/strategies/L23_rader.md`, rounds panel_r1–r11; the kernel
header summarizes them).  In ice rounds r1–r3 the implementation ran
unchanged and no ice-panel record was written; the numbers are on the round
leaderboards: r1 39.142, r2 39.214, r3 39.502 us/transform at the graded
cell (B=16, chain m=165), first place at L=23 both rounds it shared the cell
with L23_matrixsimd.  The arithmetic question (Rader-23 folds to the
cyclic-11 pair = the dense conj-folded kernel; no sub-quadratic length-11
convolution beats 121 fused FMAs on FMA hardware) was settled in panel_r1–r2
and has not moved.

## Round ice_r4 — own the graded step: fft3d_chain with the map fused

**The task change**: the graded chain is now the rivals' full step
`state <- (z+c)/(1+|z+c|), z = FFT(state)`, and the driver detects an
optional `fft3d_chain` entry point.  Through the fallback (our FFT + the
driver's unfused map pass) this entry measured **71.6 us/transform** on the
node — against the rivals' 39.0 (their 0.103 s / (165·16)).  Everything this
round is about deleting that unfused pass.

### What changed

1. **`fft3d_chain` exported, chains run PER VOLUME.**  Volume b runs all
   m=165 steps while its working set stays L2-resident: state 196 KB + t1
   196 KB + c volume 190 KB + plane buffer 18 KB ≈ 0.6 MB of the 1.25 MB L2
   (the brief's "iterate a volume through steps while it is cache-resident").
   The fallback instead swept the whole 6 MB batch through every step.
   Steps 1..m-2 run **in place** on the state buffer — legal because the
   X pass fully drains the state into t1 before the plane loop rewrites it —
   so there is no ping/pong pair, just one padded state volume.

2. **Map fused into the Z pass's TILE stores.**  Every output point of a
   step is produced exactly once in the Z pass (up to the bit-identical
   overlap chunk), so the map is applied to the transposed tile vectors just
   before their stores, with c loaded at the matching plane offsets
   (identical (y,z) addressing, c always at the standard 1058-double plane
   stride).  The raw FFT value never round-trips through memory.

3. **Pair-compressed map arithmetic** (`L23R_MAP2`): for two output vectors
   (8 points), the |t|² lane-pair sums are compressed into ONE vector by two
   two-source shuffles + one add, so the expensive part runs once per 8
   points: one `vsqrtpd`, one reciprocal, then two expand shuffles and two
   application muls.  Cost per 8 points: 15 FMA-port ops + 4 port-5 shuffles
   + 1 vsqrtpd; per volume ≈ 22.8k vector ops + 1521 sqrts ≈ +4 us of issue
   work on top of the FFT's ~121k vector ops.  This *extends* the rivals'
   `PW_CORE` (ext/reference/fft_v4_solutions/1760b1bf_score0.96/
   generator.py), which ran sqrt on pair-duplicated sums (one per 4 points);
   the compression halves the divider-unit demand and the Newton work.

4. **Reciprocal = `vrcp14pd` seed + 2 Newton steps (default, `zmap=nr`).**
   Seed error 2⁻¹⁴ → 3.7e-9 → 1.4e-17: sub-ulp, so the map is exact to
   ~1 ulp per application, like the hardware divide.  Chain drift measured
   **3.57e-14 at m=165** (budget 1.7e-11, 480× margin) — we keep full
   correctness where the rivals' fast path drifts to 1.28e-8 on long chains.
   NOT taken: their float `rcp_ps` seed (needs 2 zmm↔ymm converts per use;
   `vrcp14pd` is one op with a better seed on AVX-512).

5. **Chain state planes padded to 1064 doubles** (the r7 t1-padding trick
   applied to the state volume, possible only because the chain owns its
   intermediate buffers): every X-pass line load starts on an aligned plane
   base instead of 3/4 split cache lines.  Node, same core, 3/3: 37.9 vs
   39.2–39.4 us/t (**−1.2 us**).  Needs three step instantiations (x0→st,
   st→st, st→fout) so every kernel call site keeps compile-time-constant
   strides (the L45_pfa r8 lea-spill rule).

### Measured on the node (a80n0, leased cores, graded cell B=16 m=165)

| configuration | us/transform |
|---|---|
| fallback (r3 code + driver map) | 71.6 |
| fused chain, map = hw vdivpd (mp=1) | 45.7–45.8 |
| fused chain, map = rcp14+2NR (mp=2) | 39.05–39.4 |
| + padded state planes | **37.9–38.8** |
| rivals' best (their 0.103 s) | 39.0 |
| MKL through the fallback | 262–266 |

B=1: 38.0 us/t, B=64: 38.5 us/t (same code path; see phase note below).
Single-transform rel L2 3.8e-16; chain rel L2 3.6e-14 (B=16), 2.7e-14 (B=1),
2.8e-14 (B=64).  Chain and single outputs bit-identical across processes.
Projected full graded point: ~100 ms vs rivals' 103 ms.

### What did NOT work, with the number that killed it

- **Hardware `vdivpd` for the reciprocal (mp=1)**: 45.7 vs 39.05.  The
  divider-unit cycles (~30k/volume for sqrt+div serialized per pair) do NOT
  hide behind the Z group's FMA work — the div sits at the end of the
  dependence chain, right before the stores.
- **All-FMA sqrt via `vrsqrt14pd` + 2 Newton (mp=3)**: 39.8 vs 39.05.  The
  sqrt, unlike the div, starts early enough that the OoO window hides it;
  trading its hidden divider cycles for ~6 more FMA-port ops per 8 points
  loses.  Both alternatives stay compiled in, selectable via `L23R_MAP`.
- **Deferred-Z schedule inside the chain step** (Y(x+1) between Y(x) and
  Z(x), from L17_matrixsimd r6): a wash, 39.35/39.49 vs 39.17/39.44 same
  core — consistent with plain being the node's checked B<64 pick since r7.
  Dropped from the build.
- **State-base offset sweep (`L23R_STOFF`, 0–56 cache lines)**: ±1.5% only.
  The real phase story is below.

### The mod-4096 phase pathology (watch this, all entries)

With the first fused build, B=1 measured **44 us** vs B=16's 37.9 — 16%
slower per step, identical per-volume work, reproducible across cores and
map styles.  Mechanism: at B=1 every driver buffer is page-aligned and every
step sees the same relative mod-4096 geometry between the driver's buffers
and my block's t1/st/pb (4K-aliasing false dependencies between c loads /
state stores at equal page offsets); at B=16 the 16 volumes cycle through
16 phases (volume stride 194672 ≡ 2160 mod 4096) and dilute the bad ones.
Growing the plan block by 520 doubles (slack added for the offset knob)
shifted the whole block's allocation phase and the pathology vanished at
every batch (B=1: 44 → 38.0).  The lesson: **a 16% per-step effect can live
entirely in the malloc phase of your scratch block**, and it is invisible at
the graded batch until it isn't.  The `L23R_STOFF` env knob (0–63 cache
lines added to the state base) stays in as a probe; allocation is
deterministic per (L, batch, block size), so the shipped layout is what the
monitor will measure.  A plan-time self-tuned base phase is the clean fix if
this ever resurfaces.

### Harness notes for whoever reads this next (round ice_r4 tryout bugs)

- `tryout.sh` line 36 references `$W` before line 38 defines it; under
  `set -u` every chain-mode tryout dies with "W: unbound variable".
  Workaround that touches nothing:
  `W=$PWD/build/tryout/<name> ./tryout.sh <name> 23 16`.
- The remote map-check inside tryout still fails (`--cin '/c.bin'`): the
  `$W` inside the command substitution is single-quoted, survives to the
  remote shell, and expands empty there.  Run the check yourself on the
  shared filesystem:
  `python3 check.py --input $D/in.bin --output $D/out.bin --L 23 --batch 16
  --map-check 165 --cin $D/c.bin` — the driver writes `out.bin.chain` next
  to `out.bin`.
- tryout regenerates `in.bin`/`c.bin` at whatever batch you pass, so a B=1
  tryout clobbers the B=16 files; generate per-batch copies for manual A/Bs.

### Borrowed this round

- The Z-store map fusion + Newton-on-FMA-pipes recipe and the lazy-map idea:
  the rivals' 1760b1bf generator (`PW_CORE`, `pw_full_fast`) via the brief's
  §10 pointer — with the compression and the `vrcp14pd` seed as our own
  changes, and 2 Newton steps kept so the chain gate passes by design.
- Per-volume cache-resident chaining: the brief's own directive.
- State-plane padding: L23_matrixsimd r6's unexecuted "Next" item via my r7
  t1 padding.

### Next round

1. Plan-time base-phase self-tuning (race 4–8 block phase offsets on the
   chain in fft3d_create) to make the mod-4096 story robust by construction
   rather than by luck of the block size.
2. The X pass still loads each state point twice (u and w sweeps).  A
   single-load X kernel (krn_il's load structure with the two-sweep's pinned
   constants) would cut the X pass's L1 traffic ~half; it lost as a general
   kernel (spills), but the X pass alone has the fewest live values.
3. The map's 4 shuffles per 8 points land on port 5 next to the tile
   transposes; a store-order variant that maps pre-transpose columns (c
   gathered once per plane into a transposed scratch at plan… no — c is
   step-invariant, so a ONE-TIME transposed copy of c per volume would let
   the map run before the transpose entirely) is worth one experiment.
4. If the monitor's scored number lands near 38.0, L=23 beats the rivals'
   0.103 s at full double precision; the remaining gap to the FFT-only 35 us
   is the map's ~3 us — only a cheaper-but-still-exact reciprocal moves it.

## Round ice_r5 — eager-vs-lazy settled by experiment; chain made phase-robust by construction

Scored r4: **38.105 us/t**, first at L=23 (matrixsimd 44.87, MKL fallback
262, rivals' target 39.0).  The cumulative mandate this round: test the two
techniques other entries' records point at — L13_direct's staged lazy map
(their r4 winner) and L17_matrixsimd's negative on the same idea — and
execute my own r4 "Next" items.  Everything below is node-measured
(tryout leased core, graded cell B=16 m=165, SAME-WINDOW contrasts only;
`min` is the statistic — this round's windows often carried a stably
contended phase where the median lies, the r4 matrixsimd lesson).

### What shipped: chain arm `e` (eager Z-store map + owned, staggered buffers)

1. **Padded c copy (`cp`)**: c is pad-copied once per volume-chain into a
   1064-double plane-stride buffer inside my block (a 190 KB sweep
   amortized over 165 steps).  Every hot-loop load and store of the chain
   — st, t1, pb, cp — now lives in MY allocation; the driver's buffers are
   touched only at step 0 (x0 read) and step m−1 (fout write).
2. **Deterministic mod-4096 staggering (`L23R_PH4K`)**: t1, st and cp
   bases are placed at intra-block page phases 0 / 1344 / 2688, thirds of
   a page apart, so no two same-plane-index streams can collide mod 4096
   regardless of block size.  This is the r4 "Next" item 1 executed in its
   deterministic form: with the hot loops owning their buffers (item 1),
   intra-block phases are the only ones that matter, and they are now a
   property of the code, not of malloc luck.  B=1 measured 38.26 us/t —
   the r4 44-us pathology cell stays closed.  (The `L23R_STOFF` probe knob
   survives on top.)
3. Same-window A/B vs r4's chain (`n`): **e 37.81/37.89 vs n 37.87/38.01**
   — ~0.1 us, inside noise but consistent in direction (2/2), and the
   structural determinism is the real purchase.  Shipped default `e`;
   `L23R_CH` env forces any arm.

### Measured (node, graded cell, min us/transform)

| configuration | us/t |
|---|---|
| shipped `e` (4 runs, 3 windows) | **37.71–37.94** |
| r4 chain `n` (same windows) | 37.85–38.06 |
| lazy staged+pipelined `l` | 41.55–42.25 |
| `e` with row+plane-padded c (1152/48) | 38.23–38.25 |
| `-DL23R_SQ2` |t|² form | 37.85–38.11 |
| B=1 / B=64, shipped | 38.26 / 37.96 |
| MKL through the fallback | 262.1–262.3 |

Correctness: single 3.798e-16; **chain m=165 rel L2 3.569e-14 (B=16),
2.717e-14 (B=1), 2.838e-14 (B=64), tol 1.7e-11** — full-double tier
unchanged, ~480x margin.  Three processes bit-identical (.chain cmp), all
arms cmp-verified BIT-IDENTICAL to each other (the map's per-point
arithmetic is lane-wise, so where it runs cannot change bits — verified,
never assumed).  AVX2-only build (w2 eager path) PASSES both gates.

### What did NOT work, with the number that killed it

- **The lazy staged X-pass map: 41.55/42.25 vs 37.81/37.89 (+10%).**
  Built exactly as L13_direct's r4 winner: Z stores RAW z, the map runs at
  the next step's X pass through a 2x1.5 KB rotating L1 stage, depth-1
  pipelined (map of chunk i+1 issued between chunk i's stage stores and
  its FFT), last step maps eagerly at Z-store; solo-tail plane 22 mapped
  in-vector.  L17_matrixsimd's r4 verdict ("our X pass IS the first pass —
  the mechanism is latency exposure") holds at L=23 EVEN WITH the pipeline
  they didn't have: my eager Z-store map is already hidden behind the tile
  transposes' port-5 work (that is what mp=2's early-sqrt shape was built
  for), so staging pays the map's full issue cost plus ~34 extra L1
  accesses per chunk and buys back nothing.  Between L13 (win), L17
  (lose), L23 (lose), the panel's rule is now: **lazy pays only where the
  eager placement was divider- or spill-bound; if the eager map already
  hides, the stage is pure overhead.**
- **Row+plane-padded c (rows 23→24 complex, plane 1152): 38.23/38.25 vs
  38.06/37.85.**  Killing the 3/4 split-line Z-map c loads is worth less
  than the +3.8% c-line footprint (138 vs 133 lines/plane) and the lost
  line-sharing between adjacent ky rows — ICX absorbs a split load at ~1
  extra L1 access, cheaper than an extra resident line.  The kernel now
  carries a separate c-lane-stride parameter (cbs) from this experiment;
  it costs nothing (compile-time constant) and stays.
- **|t|² via deinterleave + 2 FMA (L17_matrixsimd s6's sub-shape,
  -DL23R_SQ2): 37.85/38.11 vs 37.71/37.79.**  One uop and one dependence
  level saved per 8 points is invisible at a cell that is not map-issue-
  bound.  Different bit class, so compile-time only; not shipped.
- **Single-load X kernel for the chain (r4 "Next" item 2): killed by
  arithmetic before any build.**  Single-load data + pinned constants
  needs 23 accumulators + 22 constants = 45 registers; the table-operand
  form (krn_il) does 23 data + 242 coefficient = 265 L1 accesses/chunk vs
  the two-sweep's 46, and the second sweep's reloads hit L1 anyway.  Do
  not rediscover this.
- The r4 tryout bugs persist unchanged (`W=$PWD/build/tryout/<name>`
  prefix; run check.py by hand — the remote `--cin '$W/c.bin'` still
  expands empty on the node).

### Borrowed this round

- The two-phase staged map + depth-1 pipeline: **L13_direct ice_r4**
  (items 1/3), built faithfully, rejected here with numbers.
- The prediction that it would lose and why: **L17_matrixsimd ice_r4**
  ("What did NOT work" v4) — confirmed at L=23, now with the pipelined
  variant they left untested.
- The |t|² deinterleave shape: **L17_matrixsimd ice_r4** s6 (tested, wash).
- The "same-window contrasts only / a tiny sd can be a stably contended
  neighbor" protocol: **L23_matrixsimd ice_r4**.

### Next round

1. The chain is now ~37.8 = FFT skeleton ~35 + map ~3, at ~1.5x the
   ~25 us port floor; the headroom is in the FFT body's junction latency,
   which five rounds of scheduling variants have not moved.  The one
   unbuilt structural idea left: a Z-group store layout that lands
   map-friendly WITHOUT tile transposes (L17_matrixsimd's costed sketch) —
   trades ~13 k port-5 shuffles/volume against strided map addressing.
   Big rewrite, uncertain payoff; it is the only >1 us lever I can name.
2. If the monitor's number lands ~37.8-38.0, the margin over the rivals'
   0.103 s is ~1.08x at full double precision; defense is adequate and
   further rounds here have lower expected value than the thin-margin
   geometries (L=8/13/36).

## Round ice_r6 — matrixsimd's two r5 extensions tested in MY tail: the map flip is a WASH here, the aligned layout ships

Scored r5: **37.780**, second — L23_matrixsimd took first at 36.903 by
executing my r4/r5 recipe plus two extensions of their own (their record
says so plainly).  This round is the mirror of their r5: adopt their two
extensions, race them in situ, keep what survives.  Node protocol as in
r5 (min statistic, MKL-in-window contamination probe, same-window or
same-PROCESS contrasts).  The node carried heavy intermittent contention
all round (several windows with median 40-43 over a clean min; one arm-e
window read a flat 43.1, discarded against its 4 clean 37.9s).

### 1. Their MAPV=1 map (mp=4) — transplanted, raced, REJECTED as a wash

Added map style mp=4 = rsqrt14+2-Newton for the sqrt + ONE hardware
vdivpd for the reciprocal (chain arm `g`).  This was the one cell of the
(sqrt, recip) 2x2 I had never raced: r4 measured div+sqrt (mp=1) 45.7,
sqrt+rcp-ladder (mp=2) 39.05, ladder+ladder (mp=3) 39.8; matrixsimd's
r5 pinned race won with ladder+div by 1.5 us IN THEIR TAIL.  In mine:

- **In-plan same-process race** (new `chAB` telemetry, below): e vs g
  across 8 windows: e 43.66-44.46 vs g 43.73-44.57 — g wins some windows
  by 0.1-0.2, loses others by the same; no consistent direction.
- Driver level: e 37.85-37.95 vs g 37.94 (adjacent windows).

Their own warning ("map-variant rankings are STORE-TAIL-SHAPE dependent;
pin-race them in situ, do not copy a ranking") is confirmed in the
opposite direction.  Both tails are Z-store tile maps, so tail SHAPE
alone does not explain the 1.5-us flip they measured; the residual
difference is the surrounding chunk schedule (their za body vs my
two-sweep).  mp=4 stays compiled (`L23R_MAP=4`) and permanently raced by
the telemetry.

### 2. Arm `r` SHIPPED: the fully aligned chain layout (their item 4, taken further)

State AND t1 rows padded 23 -> 24 complex (row stride 48 dbl = 384 B =
6 whole lines; plane 1104 dbl, a 64-byte multiple), with two extensions
their version does not have:

- **Pad-column chunk set `off23a` = {0,4,8,12,16,20}** for the X and Y
  passes: the tail chunk starts at 20 (320 B, aligned) and computes one
  garbage lane over the finite pad column instead of overlapping at 19.
  Every X and Y chunk base is aligned; no overlap rewrite.
- **m0=20 synthesized-pad tile** in a new tr=2 store tail: the last Z
  tile writes columns 20..22 plus the pad column (duplicate of P1-R1,
  never read), so **6/6 tile stores are aligned** on padded-output steps
  — this executes L23_matrixsimd r5's own unexecuted "Next" item 1.

Alignment ledger per mid-chain step vs arm e: Z tile stores ~3/4 split
(lane stride 368 B) -> 0; Y t1 loads ~3/4 split -> 0; X chunk bases
all-but-tail -> all.  c stays flat-padded 1064/46 — killing its split
LOADS was measured a net loss in r5 (footprint + lost line sharing) and
loads are the cheap half of a split; the kernel's separate `cbs` stride
made this free to keep.  pb grew to 24 rows (the off23a Y chunk writes a
scratch row 23).  FP cost: zero — chunk counts and kernel calls are
unchanged (138 X chunks were already the rp count; 6 Y + 6 Z per plane
unchanged; the pad tile replaces the overlap tile 1:1).

**Bit-identity**: lanes are independent lines and the map is lane-wise,
so regrouping chunks and writing pad columns changes no real output bit
— cmp-VERIFIED: arm r's out.bin.chain is byte-identical to arm e's (and
to the r5-era file), at B=16; correctness values identical to r5's at
all batches (chain rel L2 3.569e-14 B=16 / 2.717e-14 B=1 / 2.838e-14
B=64, tol 1.7e-11; single 3.798e-16).

**Measured (node, graded cell B=16 m=165, clean-window mins)**:

| configuration | us/t |
|---|---|
| arm r (shipped) | **37.328 / 37.368 / 37.420 / 37.627** |
| arm e (r5 ship), same day | 37.952 (+ one contended 43.1, discarded) |
| arm g (mp=4) | 37.94 |
| in-plan chAB, r vs e | r faster **8/8 windows**, -0.3 to -0.8 |
| B=64, arm r | 37.532 (chAB r 39.02 < g 39.17 < e 39.63) |
| MKL fallback, same cases | 261.9-262.5 |

**B=1 caution for whoever reads the r6 board**: arm r reads 42.2-42.7 at
B=1 today vs r5's 38.26 — but the UNMODIFIED r5 exemplar itself reads
42.96 at B=1 in the same windows (verified by building
exemplars/ice_r5/L23_rader.c directly).  The whole B=1 cell is elevated
on the node right now (mechanism unknown; MKL-in-window reads clean, so
it is not gross contamination — possibly a persistent neighbor with L3
pressure that the B=16 cell hides by touching more distinct pages, or a
node THP/zone state).  It is NOT a property of this round's code, and r
remains the fastest arm within today's B=1 state.

### 3. Fixed a page-phase collision my own stride change introduced

The r5 stagger (t1/st/cp at intra-block page phases 0/1344/2688) assumed
all three walk 320 B/plane mod 4096 (stride 1064), making phase
differences constant.  Arm r's st and t1 walk 640/plane (stride 1104),
and with cp still at 320/plane the difference is no longer constant:
at the old cp anchor 2688, st plane 17 and cp plane 17 BOTH land at
phase 4032 — the Z pass's c loads alias its state stores on that plane.
New cp anchor 1152: solved for no same-plane-index collision against st
or t1 for x in [0,23) at EITHER stride pair (arm e and arm r).  Lesson
for the panel: **a deterministic stagger is only as good as the stride
assumption baked into it — re-derive it whenever any plane stride
changes.**  (No clean same-window contrast isolated this fix's value;
it is correctness of the stagger design, kept on principle.)

### 4. New instrument: in-plan chain A/B telemetry (`chAB`)

fft3d_create now races chain arms e/g/r on the tuner arena (nv<=8
volumes, m=12, licence-honest warmup, 2 alternating sweeps, min) and
bakes ", chAB e=X g=Y r=Z" into the description — telemetry ONLY, never
a pick: map styles are different bit classes, and a timing-based pick
would let output flip across processes (the r8 timed!=checked lesson
applied to the chain).  Every leaderboard line, including the monitor's
scored one, now carries the node's own same-process arm contrast.  Dev
builds add -DL23R_ABPRINT to print it through tryout's ssh hop (env does
not survive the hop; -D flags do).  Extends the L36_pfa r8 in-plan probe
pattern.

### What did NOT work / what was not done, with numbers

- **mp=4 (ladder+div map)**: see item 1 — e 37.85-37.95 vs g 37.94
  driver-level, chAB no consistent direction across 8 windows.  Do not
  re-transplant map rankings between bodies; race them.
- **Row-padded c** stays dead (r5: 38.23-38.25 vs 38.06/37.85) — not
  retried; arm r deliberately keeps c at 1064/46.
- **st planes at 1152 (24 rows) + off23a Z lanes** (would align the Z
  pass's 529 split pb tail-chunk loads/step): not built — costed as ~2%
  of loads (the cheap kind) against +4.3% st footprint, the exact
  tradeoff r5 measured as a loss on c.  Wrote it down so the next round
  does not rediscover it without new evidence.
- The tryout r4/r5 bugs persist unchanged (W= prefix workaround; manual
  check.py; per-batch in/c copies).

### Borrowed this round

- mp=4 map arithmetic + the demand that it be pin-raced in situ:
  **L23_matrixsimd ice_r5** (items 3 and its "panel lesson").
- Row+plane-padded state: **L23_matrixsimd ice_r5** item 4 (itself my
  r4/r5 padding taken further), extended here with off23a and the tr=2
  pad tile.
- m0=20 synthesized pad tile: **L23_matrixsimd ice_r5** "Next" item 1,
  executed here first.
- In-plan probe telemetry shape: **L36_pfa r8** via my r9 tuner note.

### Next round

1. If the scored number lands ~37.3-37.6, matrixsimd (36.9) holds first
   by ~0.5 us.  The remaining structural difference between us is the
   FFT body, not the map or the layout: both bodies sit ~1.4x above the
   ~26 us port floor.  The one unbuilt >1 us idea remains the
   transpose-free Z-group store layout (r5 "Next" 1); a cheaper new one:
   interleave TWO volume chains at the plane level (true cross-volume
   independence at every Y->Z junction, where deferred-Z's same-volume
   filler was a wash) — working set 2x (~1.2 MB) sits exactly at L2
   capacity, so it needs the L1-stage discipline of L6_unrolled's
   pair-interleaved chains, which WIN at L=6.
2. Re-measure B=1 next round before believing any B=1 delta: today's
   node B=1 state is +4.7 us for code that is bit-identical to its r5
   self (see the exemplar test above).

## Round ice_r7 — the 1.3-us gap to matrixsimd found by objdump, not by theory

Scored r6: **37.172**, second (L23_matrixsimd 35.888).  This round's mandate
is mining the competition; at L=23 the ranking of sources to mine came out:
(1) the v6 Hartley-split H23 kernel — read and DISMISSED by arithmetic
(below); (2) the rivals re-benchmarked on our node — best L=23 is 0.099 s
= 37.5 us/t (1760b1bf), already behind both of us, nothing to take; (3)
L23_matrixsimd's r6 exemplar — where the whole round's win came from.

### The diagnostic that worked: same-window exemplar race + instruction diff

Their exemplar built against my binaries on the same leased core, same
inputs, same window: **matrixsimd 36.017 vs my arm r 37.193** — the gap is
real code, not node state.  Since r6 our chain steps are structural twins
(same two-sweep pinned kernel, same aligned za layout, same Z-store map
fusion, same 138+138+138 chunks/step), so instead of more theory I
disassembled both step bodies and diffed the instruction mix
(`objdump -d`, per-mnemonic counts of their `chain_pp_w4` = 1786 insns vs
my `chstep_rm_w4`/`chstep_hm_w4` = 1823/1866):

| where | theirs | mine (r6) |
|---|---|---|
| map compress/expand | 12 vunpcklpd + 12 vunpckhpd + 57 vpermilpd-IMM | 12 vpermt2pd + 12 vpermi2pd + 28 vpermpd (INDEX-VECTOR forms) |
| index-vector loads | 0 | **35 vmovdqa64** |
| reg-pressure spills | 202 vmovapd | **247 vmovapd** |

GCC compiles the generic `SHUF2(q0,q1, 0,2,4,6,8,10,12,14)` /
`SHUF1(r8, 0,0,1,1,...)` builtins as two-source/lane-crossing permutes that
need index vectors in registers; the indexes cost loads, occupy zmm regs
next to the 23 live accumulators, and push ~45 extra spills — all inside
the Z kernel body that executes 138x per step.

### What shipped (three changes, all node-raced, chain outputs cmp-verified)

1. **MAP2 compress/expand rewritten in immediate form** (adopted from
   their `l23_map2st`): |t|² compression = `vunpcklpd + vunpckhpd + add`
   (point-interleaved order {t0p0,t1p0,t0p1,...} instead of half-split),
   expansion = `vmovddup` / `vpermilpd imm 0xFF`.  Every ladder op is
   elementwise, so per-point values are BIT-identical (cmp-verified,
   single and chain, old vs new binaries).  The index loads vanish
   (objdump: 0 vmovdqa64, spills 247→235).  Same window: **36.18 vs
   37.04 us/t (−0.86)**.
2. **Arm 'h' = arm r's aligned layout + the mp=4 map (rsqrt14+2NR sqrt
   ladder + ONE late vdivpd), SHIPPED as the compile-time default.**  The
   r6 "mp=4 is a wash" verdict was measured in the OLD arm-e tail; their
   own store-tail-shape warning cuts both ways, and in arm r's aligned
   tail the same map wins ~0.4–0.5 us: chAB h<r in EVERY window this
   round (8+ windows, −0.36 to −0.46 in-plan), driver A/B/A/B h
   36.21/36.56 vs r 37.10/37.14.  chAB telemetry extended to four arms
   (e/g/r/h); default remains compile-time (bit classes).
3. **TILE stores each mapped pair before mapping the next** (their TILEM
   program order): yy[0..1] die before yy[2..3]'s ladder allocates.
   Alternating A/B 3/3: 35.94/36.11/36.02 vs 36.36/36.32/36.09
   (−0.08..−0.41), bit-identical.

### Measured on the node (a80n0 leased cores, graded cell B=16 m=165)

| configuration | us/t |
|---|---|
| shipped (arm h + imm shuffles + early-store tiles) | **35.88–36.39 across 5 windows** |
| r6 code, same windows | 37.19–37.24 |
| matrixsimd r6 exemplar, same windows | 35.93–36.77 |
| best rival on this node (1760b1bf) | 37.5 (0.099 s) |
| MKL fallback | 262.4 |

B=1 36.17, B=64 36.35 (the r6 B=1 elevation is gone — it was node state,
as the r6 note predicted).  Correctness: single 3.798e-16; chain m=165
rel L2 3.824e-14 (B=16) / 2.901e-14 (B=1) / 2.854e-14 (B=64), tol 1.7e-11
— ~440x margin.  Chain outputs repeatable (cmp) across processes; the
imm-shuffle and early-store changes are cmp-verified bit-identical to the
r6 arm-h arithmetic; arm h vs r remain different bit classes (reciprocal
realization), so the default stays compile-time.

### What did NOT work / was dismissed, with the number or argument

- **v6 Hartley-split H23 kernel (fft_v5v6_solutions/v6_f40c5e25, H23=s65)
  — dismissed by op count before building.**  gen_hartley emits the SAME
  folded-symmetric arithmetic we settled in panel_r1 (fold s_j/d_j, then
  4 FMAs per (j,k) on split re/im = 4h² = 484 FMAs per 8 pencils + 44
  fold adds + 44 output combines ≈ 596 ops/8 pts vs our 594/8 pts) —
  their "decisive win over the primes" was over MKL, not over this
  panel's kernels.  Its SoA 8-volumes-per-zmm packaging removes shuffles
  but costs the layout transposes elsewhere; their own best L=23 on our
  node (0.118 s, gate-FAILED at that) is 20% behind us.  Do not re-mine.
- **The rivals' L=23 fastest (0.099 s) offers nothing structural we lack**
  — the 1760b1bf generator's PW_CORE fusion has been in this entry since
  ice_r4, and its precision-tiered map fails our gate philosophy anyway.
- The remaining static-insn gap to their step body after the fix is +35
  (1821 vs 1786, ~33 spill moves).  Racing at parity now; the next 0.2 us
  is register-allocation luck, not a named mechanism.
- tryout.sh cannot run this session at all: no slurm client on this dev
  host, so `reserve.sh --status` fails and tryout exits at its gate even
  though the node reservation is alive.  Worked around by following
  tryout's protocol manually (slot_lease acquire → the exact ssh
  build/run/check commands → release).  The r4/r5 in-script bugs are
  moot under this workaround; check.py must run on the node (no local
  numpy).

### Borrowed this round

- Immediate-form map shuffle shapes (`vunpck*` compress, `vpermilpd`-imm
  expand) and the early-store TILEM program order: **L23_matrixsimd ice_r5
  exemplar source** — found by diffing our step bodies' instruction mixes,
  which is itself the transferable method: when two entries are
  structural twins and one is 1 us faster, objdump the two hot bodies
  and diff the mnemonic histograms before theorizing.
- The mp=4 map arithmetic (raced anew in the new tail): **L23_matrixsimd
  ice_r5** (MAPV=1), with their own "pin-race in situ" rule applied to a
  tail shape that did not exist in r6.

### Next round

1. We are at parity with L23_matrixsimd (both ~35.9–36.4 in shared
   windows) and both ~4% under the best rival re-benchmarked on this
   node.  If they adopt nothing, expect a scored dead heat; the panel's
   L=23 cell is safe either way.
2. The only unbuilt >0.5 us idea remains the two-volume plane-interleaved
   chain (r6 Next 1) — but deferred-Z's repeated wash (r4 here, r4
   addendum there) says the Y→Z junction it targets is already cheap, and
   its 1.2 MB working set sits at L2 capacity.  Low expected value; build
   only if a round has nothing better.
3. Both step bodies now roofline ~1.38x above the ~26 us port floor with
   no named mechanism left in the schedule space.  If the panel wants
   more at L=23, it needs a new structural idea (the transpose-free
   Z-store layout sketch remains costed-not-built), not another knob.
