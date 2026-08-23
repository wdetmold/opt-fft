# L8_batchsimd — strategy record (Ice Lake panel)

Lineage: rounds 1–11 of this implementation's history live in
`bench/geom/strategies/L8_batchsimd.md` (the single-thread geometry panel; the
code carried into this panel is that file's round-11 state).  The round ice_r1
agent for this entry crashed (bun runtime abort, exit 134, see
`results/ice_r1/agents/exits.txt`) before writing a record here — this file
therefore starts at ice_r2, with the ice_r1 node results reconstructed from
the scored JSONs.

## Round ice_r1 (reconstructed — no agent record was written)

Code was the geom-panel r11 state, unmodified.  Node (a80n0, Xeon Gold 6326,
ICX) scored the graded chain (B=64, m=2572, `--unitary`):

* leaderboard: **0.550 µs/transform, 1st** (fusedaxes 0.556, radix8 0.561,
  MKL 0.623 = 1.13×), spread 3.2%.
* pick string all 3 runs: `mode=FUSED nt=0 pf=s0 alloc=r8(a64,si512)`.
* arena{} (in-plan, min-of-7): FUSEDAA/s0 **ahead of** FUSED/s0 in **3/3
  runs** — 0.426/0.487/0.414 vs 0.429/0.493/0.417 — i.e. exactly branch (b)
  of the r11 prediction table ("arena shows AA ahead consistently ⇒ next
  round drops the hysteresis for AA").  fusedaxes' file read the same
  ordering with their new depth-3 rows in front: AA2 0.409 < AA 0.412 <
  fused 0.415.

## Round ice_r2

### What changed

1. **Ported `aa_perm2_tab` verbatim from L8_fusedaxes (their round panel_r11
   "fusedAA2")** as `MODE_FUSEDAA2`: same `fusedaa_volume` kernel, same
   execute-time sigma≡48 scratch base, but the pass-B ky-iteration row is
   brute-forced collision-free at store-buffer depths 1, 2 AND 3 (the
   56-entry store buffer holds ~3 iterations of stores in flight; the depth-1
   rows I ported in r11 still carried 0–4 depth-2 and 0–2 depth-3 residual
   collisions depending on c — an allocation lottery).  Store ORDER only;
   output bit-identical (verified: rel_l2 2.267e-16 unchanged to the digit,
   `repeatable: identical output across runs`).
2. **Mid-regime (the graded B=64 cell) default and hysteresis anchor moved
   FUSED+s0 → FUSEDAA2+s0** — executing my own r11 branch plan (b) on the
   three-run ice_r1 arena evidence above plus fusedaxes' independent
   confirmation.  Candidate set is now {FUSEDAA2+s0 (anchor), FUSEDAA2+none,
   FUSEDAA+s0 (depth-1 control), FUSED+s0, FUSED+none}; 6% hysteresis now
   protects AA2, so FUSED must beat it clearly to displace.  AA2+none is
   offered because corpus §10 says bare-metal ICX hardware prefetchers are
   strong — let the node price the 128-uop/volume s0 tax itself.
3. B=1 (fixed FUSED/SI520 path) and both streaming regimes: byte-identical,
   untouched.

### Operation count

Unchanged: 1248 vector FP (24 × 52-op split-complex radix-8 codelet) + 896
shuffles + 256 loads + 256 stores per volume.  AA2 is a memory-schedule
change only.

### ICX port math (why I did not chase shuffle reduction)

On ICX the second 512-bit FMA pipe lives on **port 5, shared with all 512-bit
shuffles**, so the machine floor is (1248 FP + 896 shuf)/2 ports ≈ 1072
cycles/volume; the scored 0.550 µs ≈ 1595 cycles at 2.9 GHz *including* the
driver's per-step unitary-scale pass over the whole batch (~130–190
cycles/volume, paid identically by every backend) → we run at ~1.3× the
port-05 floor.  Two shuffle-reduction structures were priced on paper and
rejected — recorded so the next generation does not rediscover them:

* **lanes-resident z-DFT via masked cross-lane butterflies** (the L=64
  z-split trick applied at L=8): ~48 shuffles + ~110 FP per ky-iteration vs
  the current transpose-pair route's 96 + 52 — port-05 total goes UP (158 vs
  148), and the output interleave is still owed.
* **interleaved-complex arithmetic for the y/x axes** (kill the 128
  deinterleaves): an interleaved DFT8 on 4 complex/lane costs ~32 FP + 5
  shuffles per 4 pencils; per x-plane that is 74 port-05 uops vs the split
  path's 68.  The ±i renames that are free in split-complex become real
  swap+FMA work interleaved.

### Measured (all numbers from the NODE via tryout.sh — this panel's tryout
runs on a80n0 itself, one leased core, co-tenants active)

| configuration | B=64 chain min / median | B=1 chain |
|---|---|---|
| before (r11 state, same session) | 0.591 / 0.705 µs (sd 5.8%) | — |
| after (FUSEDAA2 anchor) | **0.562 / 0.635 µs** (sd 6.0%) | 0.554 µs (sd 0.04%) |
| MKL same core, same runs | 0.633–0.648 µs (sd ≤0.3%) | 0.620 µs |

The B=64 sd ~6% is co-tenant L2/ring noise (B=1 on the same silicon shows sd
0.04%); the scoring window drains all leases, so the scored number should sit
near the min.  The cleaner instrument is the in-plan arena under identical
conditions: **AA2/s0 0.465 < AA/s0 0.468 < FUSED/s0 0.476 < FUSED/none
0.479** (µs, min-of-7, this round's tuner on the node) — AA2 −2.3% vs FUSED,
same sign and similar size as fusedaxes' independent arena (−1.4%).  Tuner
pick confirmed via `--json`: `mode=FUSEDAA2 nt=0 pf=s0`.
Correctness: rel_l2 = 2.267e-16 (B=64), 2.269e-16 (B=1); chain check
rel_l2 = 1.390e-13 vs tol 5.1e-11; repeatable bit-identical across runs.

### What did not work / was not retried

* Nothing shipped this round failed.  The two shuffle-reduction structures
  above were rejected on paper (numbers in the port-math section).
* JOIN_FMA stays closed (L8_radix8's r10 node probe: +0.5–0.7%; and it was
  already +4% worse on a 2-FMA part) — not retried on ICX.
* GCC spill audit of the ICX build (gcc 11.4, `-march=native`): **zero** zmm
  stack moves in `faa_run_*`/`f_run_*` — the corpus §10 spill disease does
  not afflict this kernel; no cure needed.

### Borrowed

* **L8_fusedaxes panel_r11**: the depth-3 `aa_perm2_tab` (verbatim, 8 rows)
  and its collision-depth analysis.  My FOUT/AA addressing is line-for-line
  the model their table was solved for, so it transfers without
  re-derivation — same as the r11 depth-1 port.

### Next

1. Read the scored arena{}: if AA2 ≤ AA ≤ FUSED holds in the quiet window and
   the pick is AA2 3/3, the fused-family structure question is CLOSED on ICX
   at B=64, and the remaining ~1.3× over the port-05 floor is scheduling/
   memory residue, not shape.
2. The brief says the node's PMU is exposed (`perf_event_open` works,
   `perf_event_paranoid` permitting) — fusedaxes' dormant `-DL8_PMC=1` probe
   pattern could price port-5 saturation and `ld_blocks_partial.address_alias`
   directly next round if the residue needs attribution.
3. If the scored B=64 number does not move despite the arena gap: the gap
   between arena (0.43–0.47) and driver chain (0.55+) is the driver's
   unitary-scale pass plus chain ping-pong effects — consider a chain-shaped
   surrogate in the tuner (alternate two dst buffers + a scale sweep between
   steps) so the tuner ranks candidates under scored conditions.

## Round ice_r3

Standing after ice_r2: 0.544 µs at the graded cell, a 0.12% tie for first
with fusedaxes (0.544), radix8 0.565, MKL 0.628 → 1.15×.  The VERDICT calls
L=8 "the weakest geometry on the board", the PMU blocked (`EACCES` in both
dev and scored contexts), and the alias hazard permutation-invariant — the
levers left are structure and TIMING.

### What changed

1. **FUSEDAA3 — depth-4 AA rows sized to THIS machine's store buffer, and
   the mid anchor moves to them.**  Ice Lake-SP's store buffer is 72 entries,
   not the 56 of the SKX/CLX parts the panel's depth-3 tables were implicitly
   sized for, so ~4.5 iterations × 16 stores are in flight.  Scored per c,
   the ice_r2 AA2 rows carry (d4,d5) residual collisions of
   (1,3)(2,1)(1,2)(2,3)(1,1)(2,1)(1,1)(1,3) for c=0..7 on this machine.  I
   re-ran fusedaxes' brute-force one depth deeper: **16 rows per c are
   collision-free at depths 1–4; depth-5-free is PROVABLY infeasible**
   (exhaustive over all 8! per c: best residual is exactly 1 for every c).
   The shipped rows' single depth-5 collision sits at position 6 — stores 80
   entries back, beyond even this buffer.  Same kernel, store order only,
   output bit-identical (node-verified: rel_l2 = 2.267e-16 unchanged to the
   digit, repeatable bit-identical).
2. **Delta-aware row selection — the boundary "timing lever".**  All 16
   depth-4-free rows per c are kept in the binary, and `aa3_pick()` chooses
   at aa_setup time, when δ = (out−in) mod 16 lines is finally known, the
   row whose LAST-stored pass-B iterations collide with the LATEST-loaded
   pass-A pencils of the next volume (weights 1/2/4 on tail positions
   5/6/7).  This attacks the one channel fusedaxes proved no permutation
   removes (pass-A loads of volume v+1 vs pass-B out-stores of volume v, 63
   boundaries per graded call) with the only lever their proof leaves:
   drain slack.  Scorer has real range to exploit (e.g. c=7, δ=5: worst row
   scores 3, best 36).  Zero hot-path cost — the choice is a table row.
3. **Chain-shaped tuner + clock-settle spin** for the mid regime, executing
   my own ice_r2 "Next" item 3 — and ported from L8_fusedaxes ice_r2, who
   took it from L17_matrixsimd ice_r1.  The old surrogate timed bare
   transforms on one (src,dst) pair (1 MiB, L2-resident, no scale pass —
   the 22–34% optimism the VERDICT flagged: my ice_r2 arena read 0.473
   against a 0.544 score).  The new `chain_unit` replays the graded unit:
   prime from ti, then ta↔tb with a ×1/√512 pass after every step.
   Published `chain{...}` numbers INCLUDE the scale pass and are
   leaderboard-comparable (the VERDICT's mixed-units complaint, fixed):
   this round they read 0.566 in-plan vs 0.564 measured same-process.
   Candidates {AA3+s0 (anchor, 6% hysteresis), AA3+none, AA2+s0 (control),
   FUSED+s0}; NT and pfw excluded on VERDICT §0c / L13_rader's +7.4%.
4. **Two-entry aa_setup cache** (fusedaxes ice_r2 item 5): the graded chain
   alternates (out,pong)/(pong,out) every call, so the single-entry cache
   recomputed ~50 scalar setup ops per call; steady state is now 2–4
   pointer compares.
5. B=1 and streaming: paths untouched (B=1 keeps fixed FUSED/SI520; the
   publish-only b1_ab now reports `ab{fused,fusedAA3}`).

### Operation count

Unchanged: 1248 vector FP (24 × 52-op split-complex radix-8) + 896 shuffles
+ 256 loads + 256 stores per volume; port-05 floor (1248+896)/2 = 1072
cy/vol.  AA3 changes only store order and the setup-time row choice.  Spill
audit on the icelake-server build: zero zmm stack moves (re-checked after
the refactor).

### Measured (all on a80n0 via tryout.sh, one leased core, co-tenants active)

| case | this round | MKL same window | correctness |
|---|---|---|---|
| B=64 graded chain (m=2572) | **min 0.570 µs**, driver-JSON best 0.564; contended windows 0.616/0.630 | 0.628 | rel_l2 2.267e-16, chain 1.390e-13 (tol 5.1e-11), repeatable |
| B=1 chain | **0.555 µs**, sd 0.06% (unchanged from ice_r2's 0.554) | 0.621 | rel_l2 2.269e-16 |
| B=2048 streaming | 1.473 µs (path untouched, sanity) | 1.785 | rel_l2 2.271e-16 |

In-plan chain{} tables (min-of-7, scale pass included, three windows):

* quiet-ish:  AA3/s0 **0.566** = AA2/s0 0.566 < FUSED/s0 0.575 < AA3/none 0.584
* contended:  AA2/s0 0.630 < AA3/s0 0.632 < FUSED/s0 0.644 < AA3/none 0.650
* contended:  AA2/s0 0.634 < AA3/s0 0.637 < FUSED/s0 0.651 < AA3/none 0.657

Read: the AA-family vs FUSED gap (~2–2.5%) reproduces in every window; s0
is worth ~3% in-chain (fusedaxes measured the same sign); **AA3 vs AA2 is a
dev-window statistical tie** (−0.0/+0.3/+0.5%, resolution ~1–2% here).
That matches radix8's ice_r2 finding that deeper AA schedules buy
determinism rather than dev-window mean.  AA3 ≥ AA2 in collision structure
by construction at identical arithmetic, so the anchor move is
downside-free; the depth-4 and boundary margins, if real, are quiet-window
effects — the scored table will say.  Tuner calibration: in-plan 0.566 vs
same-process driver 0.564 — the optimism is gone.

### What did not work / was rejected, with numbers

* **Depth-5-free rows do not exist** — exhaustive search, best residual 1
  for every c.  Recorded so nobody re-runs the search.
* **Cross-volume software pipelining (interleave pass A of v+1 into pass B
  of v) — analysed, not built.**  It needs ping-pong scratch (the WAR
  hazard on the shared scratch), and two 8 KiB scratch buffers are ≡ 0 mod
  4096 apart at any padding: pass-B loads occupy line classes {ky, ky+8}
  mod 16 while every pass-A 16-line store window covers ALL mod-16 classes,
  so 2 load classes per in-flight A-window collide — permutation- and
  offset-invariant, the same invariance fusedaxes proved for the in↔out
  channel, now on the scr↔scr channel.  Building it buys the port-balance
  overlap but re-opens a structural alias channel the AA lineage spent
  three rounds closing.  Not worth it while the margins are <1%.
* PMU still blocked (fusedaxes' probe returned EACCES in both contexts in
  ice_r2; nothing changed harness-side this round).

### Borrowed, plainly

* **Chain-shaped tuner + clock-settle spin**: L17_matrixsimd (ice_r1), via
  L8_fusedaxes' ice_r2 port and the VERDICT's endorsement.
* **Two-entry aa_setup cache**: L8_fusedaxes ice_r2 item 5.
* **The invariance proof** that motivated the δ-aware tail selection:
  L8_fusedaxes ice_r2 (their §4 "deferred with analysis" item — the timing
  lever is theirs by analysis, the row-freedom implementation is mine).
* NT/pfw exclusions: ice_r2 VERDICT §0c and L13_rader's in-chain +7.4%.

### Next round

1. Read the scored chain{} tables in the drained window: if AA3/s0 ≤ AA2/s0
   in 3/3 runs the depth-4 story is confirmed on quiet silicon; if they
   still tie, the residual ~0.19 µs over the 0.37 µs port floor is latency
   the schedule cannot reach and L=8 should stop spending rounds on alias
   work (the VERDICT's "at most ~20% left in this structure").
2. If B=1 lands in cases.txt (the VERDICT asked twice): the B=1 cell is
   currently fixed FUSED/SI520 at 0.555; the b1_ab published numbers will
   say whether fusedAA3 should take it (deterministic vs the FUSED alias
   lottery — the same argument that won B=64).
3. The one unpriced idea left in my file: pass-A x-order rotation combined
   with the δ-aware row (the load side of the boundary channel).  Needs the
   PMU to be worth the risk.

## Round ice_r4 (2026-08-23)

### The task changed: own the chain or pay the unfused map

The graded step is now `state <- (z+c)/(1+|z+c|)`, z = **raw unnormalised**
FFT(state) (verified in driver.c/check.py: no unitary scale in map mode —
the map itself keeps the state bounded), timed through the optional weak
`fft3d_chain` symbol.  No L=8 entry had one when this round started; MKL
through the driver fallback reads **2.12 µs/xform** at the graded cell
(B=64, m=2572) — the unfused map pass costs more than the entire FFT.
This entry now exports fft3d_chain and owns all 2572 steps.

### What shipped

1. **fft3d_chain with the map fused EAGERLY into pass B** (transposing the
   rivals' lazy-map idea, corpus §10 §2, adopted via L13_rader/L13_direct's
   ice_r4 records): after the z-axis DFT the registers already ARE split
   re/im vectors of 8 points, so `w = z+c`, `t = wr²+wi²`, `|w|`, and the
   scale run with **zero pairing permutes** (L13_direct pays 4 permutes per
   8 points for exactly this; the split-complex layout gets it free).  The
   ping-pong buffer always holds the mapped STATE, every step is the same
   call, and step m writes final_out directly — no standalone map pass, no
   raw/mapped bookkeeping, and the driver's old per-step scale pass is gone
   entirely.  c is pre-split once (first chain call, cached by pointer)
   into the exact (register = kz, lane = kx.SW) shape pass B consumes:
   1 KiB streamed per iteration, zero shuffles in the hot loop.
2. **Map ladder = rsqrt14 + 2 Newton on the FMA pipes + ONE exact vdivpd
   per 8 points** (`-DL8_MAPV=0`), i.e. L13_rader's MAPSTYLE=1 form, NOT
   L13_direct's hw-sqrt v2 — my A/B inverted theirs, see negatives.  Full
   double: seed 2^-14 → 5.6e-9 → 4.7e-17 (below rounding), divide exact,
   ~3 ulp/application.  t clamped ≥1e-300 (rsqrt14(0)=inf ⇒ NaN).  Node
   drift: **1.66e-11 at B=64/m=2572 vs tol 2.6e-10** (15×); B=1 7.1e-13
   (360×).  The float-seed tier stays banned at L=8 per the brief.
3. **The chain owns its buffers**: ping-pong is final_out ↔ an internal
   page-aligned buffer whose base is slid 0–15 lines at setup to maximise
   the aa3 boundary score in BOTH directions (owning the buffer turns the
   ice_r3 "timing lever" into a design variable).  With the natural row
   winning (below) the slide is currently moot; it ships because it is
   free and any future AA-row pick uses it.
4. **A mapped-chain race in create()** replaces the ice_r3 unitary-scale
   chain tuner (its regime no longer exists in the grading): 6 candidates,
   all output-bit-identical (row order / prefetch only) — eager {nat/s0,
   nat/sc, aa2/s0, aa3/s0} and a full **lazy family** {lnat/s0, laa3/s0}
   (map fused into the next step's pass A, raw z between steps, one
   in-place sweep at the end; built to price the rivals' original shape
   honestly).  3% hysteresis toward eager-nat/s0.
5. fft3d_execute, B=1 fixed path, streaming tuner: untouched.  Non-AVX512
   builds route fft3d_chain through execute + a scalar map (verified PASS
   under L8_EMU8, and eager vs lazy chain outputs cmp bit-identical).

### Operation count (per volume per step, on top of the ice_r3 kernel)

Map: 64 evals × (12 FMA-port ops + 1 vdivpd + 2 c-loads) = 768 p05 uops
(384 cy at 2 ports) + 64 divider ops (~512 cy, overlapped) + 8 KiB c
stream.  FFT unchanged: 1248 FP + 896 shuffles + 256/256 mem.  Working set
grows 1.0 → 1.5 MiB (state ×2 + csplit) — past this core's 1.25 MiB L2,
so every step pays some L2 miss traffic no layout can remove.  Spill
audit (gcc 11.4, icelake-server): **zero** zmm stack moves in fm_run_*.

### Measured on the NODE (a80n0; MKL = same window through the driver
fallback, since it cannot export fft3d_chain)

| config | µs/xform | notes |
|---|---|---|
| graded B=64 m=2572, SHIPPED (MAPV=0, race → nat/s0) | **min 0.778–0.800**, median 0.845–0.903 (sd 5–7%, co-tenants) | MKL 2.12 ⇒ **2.7×** |
| B=1 m=2572 | **0.861**, sd 0.03% | MKL 2.27 ⇒ 2.6× |
| identity-map control (`-DL8_MAPV=9`, timing only) | 0.631 | ladder+div cost = ~0.15–0.17 |
| ice_r3 FFT-only cell, for scale | 0.544 | c-stream+footprint ≈ +0.09 |

In-plan race (min-of-7, one window): nat/s0 **0.903** < laa3/s0 0.959 <
lnat/s0 0.967 < aa2/s0 0.996 < nat/sc 1.001 < aa3/s0 1.007.  Correctness:
single-transform rel_l2 2.267e-16 (B=64) / 2.269e-16 (B=1); whole-chain
above; chain output bit-identical across processes (cmp of .chain).
Rivals' full-task L=8 target: 0.115 s ⇒ 0.699 µs/xform — we are ~12%
behind their point but 2.7× ahead of every library through the fallback.

### What did NOT work, with the number that killed it

1. **Hardware vsqrtpd, in ANY role: MAPV=2 (hw sqrt + rcp14+2N) 0.922,
   MAPV=1 (hw sqrt + float-rcpps seed) 0.951, MAPV=3 (sqrt+div) 1.048 vs
   MAPV=0's 0.785–0.800.**  This INVERTS L13_direct's flavor sweep (their
   v2 won): in my pass B the sqrt's ~20 cy divider occupancy per iteration
   lands right before the store tail with only ~40 p05 uops after it, so
   it serializes; their two-phase stage gave it a full unit of distance.
   Port folklore does not transfer across kernel shapes — race it.
2. **MAPV=4 hybrid (alternate all-divider / all-FMA per register pair,
   corpus 0f45aeae's trick): 0.909 vs 0.800.**  Half the sqrt serialization
   is still serialization.
3. **AA rows in the mapped pass B: aa3/s0 1.007 vs nat/s0 0.903 (+12%).**
   Two mechanisms, both new this round: the permuted ky order breaks the
   csplit stream's sequentiality, and the rows' collision schedule was
   solved for a store tail that no longer exists (~100 map uops + a divide
   now sit between the scratch loads and the out stores).  Three rounds of
   AA lineage are simply not load-bearing under the map — the race keeps
   them priced in case a future shape revives them.
4. **The lazy family (rivals' own shape): laa3/s0 0.959 in-race (~2% of
   which is race bias from the 1/8-amortised sweep), forced driver run
   0.845 vs eager 0.800.**  Eager wins here because the split-complex
   pass-B map is free of pairing shuffles and saves the extra final sweep;
   the lazy code stays compiled as a race lane.
5. **c-stream prefetch (PF_SC): nat/sc 1.001 vs nat/s0 0.903 (+11%).**
   L13_direct's +7.4% finding, reproduced independently — the panel rule
   ("dedicated prefetch loses where real work already covers the latency")
   now has a fifth confirmation.
6. **tryout.sh is still broken for chain cases** (the `$W`-before-def bug
   L13 documented): worked around with `W=<path> ./tryout.sh ...` and by
   running check.py + the repeat-cmp manually.  All numbers above went
   through the manual gate.

### Borrowed this round (attribution)

* **Rival pipelines via corpus §10 §2**: the fused-map chain concept, the
  one-divider-op-per-point budget, and the eager/lazy design space.
* **L13_rader ice_r4**: the MAPSTYLE=1 ladder that won here, the
  bit-identity argument that makes the chain race adoption-legal, and the
  tryout $W workaround.  **L13_direct ice_r4**: the flavor-sweep method
  (v2 itself lost here), the sweep-bias warning, and the c-prefetch
  negative.  **0f45aeae (corpus)**: the hybrid idea (lost).
* Their float-seed fast tier was NOT copied (fails our gate by design).

### For next round

1. The remaining ~0.1 µs to the rivals' 0.699 is map-latency exposure:
   the two-phase software pipeline (L13_direct's +4% mechanism — consume
   iteration k's divides during iteration k+1's DFT, staging w/v in L1)
   is the one unpriced structural idea.  Their negative #1 (do NOT inline
   into the accumulate DAG without a stage) marks the trap.
2. The B=64 graded cell timing has sd 5–7% in dev windows from co-tenants;
   the scored drained window should land near the min.  If the scored
   number sits ≥0.85, believe the median instead and re-examine.
3. L8_fusedaxes/L8_radix8 will port this next round (cumulative mandate).
   The differentiator left is chain plumbing and the pipeline in item 1.
