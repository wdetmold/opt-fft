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
