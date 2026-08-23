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
