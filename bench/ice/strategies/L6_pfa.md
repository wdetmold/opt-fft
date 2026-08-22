# L6_pfa — strategy record (ice panel)

Lineage: this implementation arrived in the ice panel carrying eleven rounds of
history from the single-core CLX panel (panel_r1..r11); that history — the PFA
codelet derivation, the d2 codelet flip, zp-vs-ascending x-order, the deleted
512-bit and _rot families — is summarized in the header comment of
`impl/L6_pfa.c` and in `bench/mt/strategies/L6_pfa.md`'s ancestors. This file
starts at ice_r2 because ice_r1 ran the CLX-tuned code unchanged (scored 0.220
µs/xform, variant=fused_pf_d2, second place behind L6_unrolled's 0.219 by 0.5%,
with a 13.5% run spread).

## Round ice_r2

### What changed

1. **512-bit reopened and won — z512x is the new pick.** The panel_r7
   falsification of AVX-512 ("zero picks in eight cells at equal licence
   clock") was CLX-specific: the Gold 5218 has ONE 512-bit FMA pipe, so zmm
   halved the instruction count but not the port-cycle floor. The ice node's
   Gold 6326 has TWO (PANEL_BRIEF: "the second FMA pipe is genuinely
   feedable"), so I rebuilt the family:
   - `z512x` / `z512x_pf`: zmm x-pass + the node-proven ymm fused y/z (d2
     codelet). The 36 x-lines are free to group any way, so lanes = 4
     consecutive (y,z) plane indices give **9 groups, every load/store 64B
     aligned, zero tail** — 162 FP instructions where ymm needs 324. Total
     per volume: 810 FP-shaped instr (162 zmm + 648 ymm) vs 972 all-ymm; the
     arithmetic (44 real flops per DFT6, Good-Thomas optimum) is unchanged.
   - `z512yz` / `z512yz_pf`: zmm x-pass + fully-512 fused y/z (6 zmm rows +
     6 ymm tails per plane, y-DFT, an 18-shuffle vpermt2pd/vshuff64x2
     transpose to y-lanes, z-DFT, 18 shuffles back). **Lost** — see below.
   - `z512xy` / `z512xy_pf`: zmm y-DFT inside the ymm z-stage. **Lost badly**
     — see below.
2. **Chain-aware scratch placement.** The graded chain's steady state
   ping-pongs (out,pong)/(pong,out), but the 4K-aliasing placement ran once,
   on the first call's (in,out) — stale for the other 4855 steps.
   `fft3d_execute` now re-places when the buffer pair's 4096-residues change;
   `cyc4k(-d)==cyc4k(d)` means one placement serves both directions of a
   pair, so a chain re-places exactly once (at step 2) and is then stable.
3. **Chain-faithful race.** The plan-time tournament now times ping-pong
   pairs (a→b, b→a) instead of a one-way stream, matching the scored access
   pattern. z512x wins under both semantics, so the pick did not move, but
   future candidates will be ranked under the pattern that is actually scored.

### Measured (tryout.sh on the reserved ICX node a80n0; two clock regimes)

The node has two clock regimes and dev runs land in either: **quiet** (probes
read clkS256=3.50, kclk=3.30 GHz) and **busy** (everything pinned at 2.90,
other implementers' core leases active). Race numbers scale by exactly
3.3/2.9 between regimes; rankings are identical in both.

- Quiet run, race (us/vol, B=64): z512x **0.1562** ← chosen, z512x_pf 0.1568,
  fused_pf_d2 (r1 incumbent) 0.1646, z512yz 0.1741. **z512x −5.1% vs
  incumbent.** Graded chain B=64 m=4856: **min 0.213 / median 0.214
  µs/xform, sd 0.32%** (r1 score was 0.220). MKL same case: 0.340.
- Busy runs, race: z512x 0.1785–0.1794 vs fused_pf_d2 0.1876–0.1877
  (−4.4…−4.9%); chain median 0.243–0.244 (that's 0.214 × 3.3/2.9 — pure
  clock). B=1: z512x chosen, 0.218–0.219 µs/xform, sd 0.09%.
- Correctness: rel_l2 2.428e-16 (B=64 single), 2.904e-13 whole chain
  (tol 7.0e-11), repeatable bit-identical across runs. 512 licence cost on
  this node: 3.50 → 3.30 GHz (−5.7%), fully covered by the −17% instruction
  count on the x-pass.
- **The r1 mystery explained:** my r1 kclk=2.90 and 13.5% run spread match
  the busy/quiet clock ratio (3.3/2.9 = 13.8%) — clock environment, not
  code. L6_unrolled's r1 kclk=3.30 was a quiet-window create(); same
  silicon, same licence.

### What did not work, with the numbers that killed it

- `z512yz` (fully-512 fused y/z): 0.1987 vs z512x 0.1794 busy (+11%), 0.1741
  vs 0.1562 quiet. Not spills (checked the gcc 11.4 asm: zero stack traffic
  in the kernel body). Suspects: the 96-byte row stride makes every other
  zmm row load/store cache-line-split (3 split loads + 3 split stores per
  plane), and the per-plane y-DFT → 18-shuffle transpose → z-DFT chain is
  serial where the ymm z-stage overlaps chunks. A padded-row scratch (16
  doubles/row) would fix the loads but pushes split stores into the x-pass;
  not attempted this round.
- `z512xy` (zmm y-DFT feeding the ymm z-stage through 6 vextractf64x4):
  0.2181 — worse than plain `fused` (0.1934). The P[18] array of
  cast/extract results is materialized on the stack by gcc 11.4 (store +
  reload per element) instead of staying in registers. A register-named
  rewrite might rescue it, but z512yz's cleaner version of the same idea
  already lost, so this is parked.

### Borrowed

- The two-FMA-pipe fact and the licence-clock numbers that justified
  reopening 512-bit: PANEL_BRIEF / corpus §10 (the rival pipelines'
  Ice Lake forensics, provenance-corrected for bare metal).
- The zmm codelet is the radix-2-first d2 graph adopted from L6_unrolled in
  panel_r9 (store-feeding FMAs), transliterated to 512-bit.
- Nothing else applicable: the only other ice strategy records this round
  (L13/L17/L23/L36) attack Rader/dense-matrix problems that don't map to a
  6-point PFA.

### Operation count (per 6³ volume, z512x)

x-pass: 9 zmm codelets × (18 FP + 2 vpermilpd) = 162 FP + 18 shuffles, all
64B-aligned, no tail. Fused y/z: unchanged ymm, 648 FP + ~288 shuffles.
4752 real flops/volume, the Good-Thomas optimum since round 1 — everything
since is ports, addresses, and clocks.

### What I would do next

1. The chain gap: race says 0.156 µs/vol (quiet) but the graded chain scores
   ~0.214. ~0.02 µs is the driver's unitary scale pass (identical for
   everyone); the rest is per-step state I'd like to see in PMU counters —
   the perf tool is absent but perf_event_open works, so an in-plan counter
   probe (L2 misses, split loads, port 5 pressure) is the next instrument.
2. Padded-row (16-double) scratch to give a full-512 y/z stage aligned rows:
   costs ~36 extra 32B stores in the x-pass, removes 3 split loads/plane.
   Only worth it if the PMU says z512yz's loss is split-dominated.
3. If a future round adds a quiet-clock guarantee to create(), re-check
   z512x vs z512x_pf: they are within 0.4% and the pf twin may matter if the
   scored window ever runs the chain cold.
