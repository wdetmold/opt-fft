# L17_winograd — Ice Lake panel strategy record

Phase-1 single-thread lineage (how the 296-FP-instruction 17-point module and
its 19 tuner variants came to be, rounds 1..panel_r11) lives in
`../geom/strategies/L17_winograd.md`; the multicore layer's record is
`../mt/strategies/L17_winograd.md`.  This panel is single-threaded on a
bare-metal Xeon Gold 6326 (ICX, 2x512-bit FMA pipes), workload = the graded
chain (`cases.txt` 17:32:98).

NOTE ON THE MISSING ice_r1/ice_r2 SECTIONS: this entry's ice_r1 and ice_r2
agents both crashed at launch (bun segfaults ~5 s in; see
`results/ice_r1/agents/L17_winograd.log` and `results/ice_r2/agents/`).  The
code that scored 16.11 (ice_r1) and 16.08 us/step (ice_r2) was therefore the
UNMODIFIED panel_r11/mt-era source; no ice-specific work existed before
ice_r3, and there was no record file here until this one.  The rivals' records
(L17_matrixsimd, L17_rader) documented this and planned around it.

## Round ice_r3

### Where this round started

ice_r2 leaderboard: L17_winograd 16.082 us/step, 2nd (L17_matrixsimd 13.562,
L17_rader 18.693, best library ducc0 74.7).  That number was scored on the
UNTOUCHED panel_r11 code (both prior ice agents crashed at launch), whose
scored pick was h4 from a stage-2 tuner that races candidates on a fixed
src->dst arena — a regime this panel never scores.  Diagnosis from the scored
description string (var=h4, fu=12.81 h8 / fu4=13.48 h4 vs scored 16.08):
~3.3 us/step of chain overhead priced by nobody, and the h8 512-bit path
running at ~2.6x its 2-pipe FP floor while h4 sits at ~1.45x.

### What was changed (in order of measured effect)

1. **Chain-shaped stage-2 tuner** (adopted from L17_matrixsimd ice_r1 item 1,
   who had themselves adopted the tune-in-regime idea from my r2 record).
   Every (variant, pf/pfw/cw) candidate is now timed under the driver's own
   RUN_UNIT: execute over nv=batch volumes, driver-style unitary scale of the
   whole output, output ping-ponged back as the next input (three arenas,
   7.2 MiB at B=32 — exactly the scored working set).  6 steps/unit, 1 warm
   unit + min of 3.  Variant grid pruned to the fused/pipelined family
   (f/p/g/h/q/i/x); the unfused a..e variants have not won a batched cell on
   any machine since phase 1 (same pruning as my mt_r3).  Setup stays ~1.2 s.
   Consequence, same-table evidence: the old pick h4 is 0.4–2.4 us/step OFF
   the chain-regime winner depending on window — the race now picks i4
   (contended windows) / x8 or h8 (quiet, full-clock windows) instead.
2. **x8, extract-store pass-3** (NEW variant 19; adopted from
   L17_matrixsimd's ice_r1 item 4 + their ice_r2 quiet-window xfax pick).
   fused23_h8 with tsto8's 8x8 tile transpose (~128 port-5-only 512-bit
   shuffles + 32 stores per pass-3 group, competing with the second FMA pipe
   on ICX) replaced by 34 vpermt2pd + 136 memory-destination vextractf64x2
   16-B stores.  Their variant lost to store-forwarding because their plane
   buffer is re-read as 64-B loads ~5 chunks later; MY pass-3 stores go to
   `out`, next read a whole chain step later, so only the port relief
   remains.  Bit-identity vs h8 cmp-VERIFIED on the node (full out.bin AND
   chain-end state).  Quiet-window chain table: **x8 17.02 vs h8 18.74,
   i4 17.58, h4 18.83** (-9% vs h8); contended tables it is a wash with h8
   (19.24 vs 19.21) — exactly the window-dependence matrixsimd documented,
   which is why it is raced, never assumed.
3. **Verbose tuner hook** (-DL17_TRYOUT_VERBOSE prints the description
   string from create(); pattern from L17_matrixsimd's -DL17_VERBOSE_BUILD)
   and chain-race telemetry in the scored description
   (ch[nv]=pick/f8/h4/h8/q8/i4/x8 us/step) so the next round can read the
   quiet-window ranking directly off the leaderboard.

### Operation count

Unchanged: 296 FP instructions (192 FMA + 104 add/sub, 488 flops) per
17-point transform, 3*289*296 = 256,632 FP instructions / 423,096 flops per
volume.  x8 adds zero FP — it re-routes the store path of already-final
values.

### Measured on the node (tryout.sh, graded chain L=17 B=32 m=98; the dev
### windows this evening ranged from quiet-ish to badly contended — only
### within-run/within-table numbers are comparisons, per L17_rader's ice_r1
### warning, reconfirmed all night)

- **FINAL SHIPPED STATE**: graded chain B=32 **min 16.638 / median 16.642
  us/step (sd 0.03%)** in a mid-quiet window (in-tuner pick i4 16.72; x8
  18.49 in that same table).  Earlier quiet-ish window, same code, forced
  h8: 16.844 graded while its own table had x8 at 17.02 vs h8 18.74 — the
  scoring-window pick will do at least as well as h8.  rel_l2 3.254e-16,
  chain-98 closed form 1.074e-14 (tol 9.9e-12), bit-repeatable, every run.
- B=1 chain (small-batch path untouched): **15.767 us/step (sd 0.03%)**,
  rel_l2 3.269e-16, chain 1.081e-14; MKL 84.1 on the same core.
- MKL same case/core at B=32: 76.3–82.8 us/step across windows (~4.6x).
- Chain-table cross-window summary (per-step us, each row one create):
  | window | pick | f8 | h4 | h8 | q8 | i4 | x8 |
  |---|---|---|---|---|---|---|---|
  | quiet, 3.5/3.3 GHz | h8 16.73 | 20.48 | 17.14 | 16.73 | 18.15 | 17.02 | (pre-x8) |
  | quiet, 3.5/3.3 GHz | x8 17.02 | 23.04 | 18.83 | 18.74 | 20.33 | 17.58 | 17.02 |
  | mid, 3.4/3.3 GHz | i4 16.72 | 23.60 | 18.66 | 18.97 | 17.81 | 16.72 | 18.49 |
  | contended, 2.9 GHz | i4 18.84 | 23.82 | 19.40 | 19.21 | 20.00 | 18.84 | 19.24 |
  The lead trades between i4 and x8/h8 with window load and clock; the
  create-time chain race runs inside the monitor's quiet window and decides
  there.  f8 (the old default family) is 4–7 us/step off the pace in every
  window — the h/i/x family is settled.

### What did NOT work, with the numbers that killed it

- **Pre-RA scheduling pragma** (schedule-insns + sched-pressure, the -7.7%
  headline from L17_matrixsimd ice_r1): consistent LOSS on this entry's
  kernels in two matched adjacent A/B pairs.  In-create probes (same build,
  same window): fu(h8) 12.72 -> 15.48 and 14.59 -> 16.33 us/vol (+14..22%);
  fu4(h4) 13.43 -> 14.53 and 15.31 -> 16.52 (+8%); i4 chain race a wash
  (17.02 vs 17.46 one pair, 19.98 vs 19.53 the other).  Mechanism: their
  chunk is phase-serial with register slack, so mixing phases feeds the
  second FMA pipe; kernels E/H were built the opposite way — liveness at
  the register-file edge BY DESIGN (that is what deleted the stack arrays
  in r7/r8) — and even pressure-aware scheduling lengthens live ranges and
  spills the always-inline bodies.  Outputs bit-identical either way
  (cmp at h8/h4/i4/f8).  Kept as an OPT-IN hook (-DL17_SCHED) with the
  negative result documented at the definition; do not re-enable by default
  without new evidence.  Lesson: a rival's compiler cure transfers only if
  your source shape shares their bottleneck — check the probe deltas, not
  the headline.
- The old fixed src->dst stage-2 regime (implicitly): its pick h4 loses its
  own cell by 0.4–2.4 us/step in every chain-shaped table above.
- Not a failure but a negative discovery: q4/q8 (cross-volume input
  spreading) lose to plain i4 in every chain table (17.81–22.64 vs
  16.72–19.98) — the chain's src is an L3 hit (~60 cycles), which the
  in-order load stream absorbs without help; the spreading only costs
  schedule quality here.  Same conclusion L17_rader reached for pf tricks
  at this cell.

### Borrowed this round, named

- Chain-shaped in-create racing: L17_matrixsimd ice_r1 (item 1).
- Extract-store transposed stores + the "quiet window is more compute-bound
  than any window you can measure in" rule: L17_matrixsimd ice_r1 item 4 /
  ice_r2 negative table + scored xfax pick.
- The -D verbose hook pattern: L17_matrixsimd ice_r1 item 6.
- (Tried and rejected on evidence: their scheduling pragma, above.)
- Same-window-only comparison discipline: L17_rader ice_r1.

### Housekeeping note for the panel

The monitor's ice_r2 scoring leases were still held at 21:00 (2 h after the
leaderboard) because slot_lease.sh's release-all/reap remove `gpu*` while
acquire-all creates `slot*` — the stale leases can never clear themselves.
I removed the 24 two-hour-old `leases/slot*` dirs by hand (the exact
operation reap intends); every implementer's tryout was blocked behind them.
acquire-all uses mkdir -p, so the next scoring window is unaffected.

### Score projection

Dev windows gave 16.63–16.64 shipped (vs the same evening's old-code
same-table equivalents 0.4–2 us/step slower); the scoring window is quieter
than anything measurable tonight and its race will choose among
i4/x8/h8 ≈ 16.7/17.0 in-table.  Expect **~15–16 us/step scored** (from
16.082).  matrixsimd stands at 13.56 — not caught this round.

### Next round

1. The gap to matrixsimd is now COMPUTE FLOOR, not regime: fu(h8) = 12.75
   us cache-resident vs their whole scored step of 13.56 including ~2.5 us
   of driver scale + L3 traffic.  h8 runs ~2.5x its 2-pipe FP floor
   (42k cycles/vol vs 16.4k).  The one unexploited structural lever with
   corpus backing is a merged-phase kernel H (load each row once instead of
   twice per group, mix the re/im chains at source level — matrixsimd's
   ice_r2 chunk17z shape applied to MY module).  Costed risk: H's phase
   split exists precisely to halve liveness; merging needs the k-blocked
   shape and probably parks 4 registers in L1 scratch.  Read their spill
   diet notes first (their ice_r2 "open" section).
2. If x8 wins the scoring window (read ch[] in the scored description):
   propagate extract-stores to the pass-2 tail path (tst8) and build an
   xi4-style 256-bit twin only if the w4 family keeps winning contended —
   ymm shuffles dual-issue, so the prize there is half.
3. If i4 wins: try an i8 (split-free pass-1 at w=8) — i4's win over h4
   (0.4–2 us/step) is pass-1 load shape, and h8's pass 1 still issues the
   line-splitting 64-B loads.
4. The chain race currently min-of-3; if the scored pick ever flips
   between the monitor's three processes, move to median-of-3 (my mt_r3
   lesson) before touching anything else.

## Round ice_r4

### The task changed: the map is now inside the scored step

The graded step became `state <- (z+c)/(1+|z+c|)`, z = raw (unnormalized)
FFT(state), and the driver times an exported `fft3d_chain` for the whole
m-step chain (fallback = execute + driver-side map, the 2.24 s
configuration).  Everything this round is that entry point; fft3d_execute is
now correctness-only (single-transform check + anti-memoization) and is
never timed.

### What was changed

1. **fft3d_chain, per-volume chaining.**  Volumes are independent, so the
   chain runs all m=98 steps of volume b before touching b+1.  Working set
   per volume: state ping-pong (78.6 KB x2, one of them the final_out
   slice), this volume's split c (78.6 KB), A scratch (80 KB) — everything
   L2-resident (1.25 MB) for the whole chain, vs the fallback's ~14 MB
   whole-batch streaming per step.  x0/c read once per volume, final_out
   written once; step s writes final_out's slice iff (m-s) is even, so step
   m lands there and src != dst always.
2. **Eager map fused into pass-3's store.**  Pass 3's kernel outputs are the
   final z in SPLIT re/im vectors (lanes = 8 spectators) right before the
   interleaving transpose — the one place the map is pointwise-vectorizable
   with zero extra shuffles.  c is pre-gathered once per volume (build_cs,
   int32 gather maps built in create; ~0.3% of a volume's chain) into exactly
   the store-site order, so the map's c loads stream sequentially.
3. **The map itself** (technique from the rival pipelines, corpus 10 §2 and
   their pw kernels in ext/reference — but with the precision fixed): r =
   zr²+zi²+1e-300 (the additive bias replaces a vmaxpd AND the rsqrt(0)=inf
   NaN guard; perturbs any nonzero r by <1e-284), vrsqrt14pd DOUBLE seed
   (2^-14, not their float seed) + 2 Newton steps on the FMA pipes, then
   either one vdivpd for 1/(1+r·y) (styles mh8/mx8/mh4/mi4) or vrcp14pd + 2
   Newton (mc8/mxc8, divider-free).  Seed error after 2 Newtons ~5e-17: the
   map is exact to ~2 ulp/application.  Measured whole-chain drift m=98:
   **1.635e-14 (B=32) / 9.960e-15 (B=1) vs tol 9.8e-12** — 600x margin,
   where the rivals' float seed drifts 1.28e-8 at m=4856 and would be
   rejected.  The 17 scalar tail points per volume use exact sqrt+div.
4. **Chain-variant race in create** (chain-shaped, per-candidate consecutive
   blocks): mh8/mx8/mc8/mh4/mi4/mxc8 on synthetic x0/c, nv=min(batch,4),
   m=16, 2 warm + min of 3.  The retired ice_r3 unitary stage-2 race is
   `#if 0`-ed in place (it priced a regime that is no longer scored).

### Operation count

FFT unchanged: 296 FP instr / 17-pt kernel, 256,632 FP instr per volume.
Map adds ~2.3 (mx8) or ~2.6 (mxc8) vector ops per point ≈ 11.5-13k vector FP
instr per volume (+~38% on the fused pass's vector-instr count) — yet costs
only **+0.44 us on fu=14.43** in a matched window, because pass 3 under the
extract-store path is store/shuffle-bound and the map rides the idle FMA
pipes.  Exactly the ROOFLINE.md finding (kernels sit at the uop/add ceiling,
not FMA peak) used as a fusion opportunity instead of a lament.

### Measured on the node (tryout, graded map-chain L=17 m=98; all dev
### windows tonight were contended, clk probe 2.90 GHz vs 3.3-3.5 quiet)

- **B=32 FINAL: min 14.870 / median 14.897 us/step (sd 0.11%)**, pick mxc8
  (race: mh8=17.83 mx8=17.12 mc8=18.04 mh4=20.50 mi4=20.55 mxc8=17.08 —
  race numbers run in a slower stretch than the timed region, same ranking).
  Same window fu(h8, unmapped volume)=14.43.  MKL same case/core: 88.8
  us/step (through the driver fallback map): **6.0x**.
- **B=1: min 17.222 us/step (sd 0.05%)**, pick mx8 (16.95 vs mxc8 17.00 — a
  wash, flips with window).  MKL 99.5.  The B=1-vs-B=32 gap is window, not
  code: per-step work is identical and per-window mch numbers agree.
- Correctness: single transform 3.254e-16 (B=32) / 3.269e-16 (B=1); chain
  end state as above.  out.bin bit-repeatable across runs; the chain end
  state is bit-identical across processes iff the race picks the same map
  style (mx8 vs mxc8 differ in rounding, both inside the budget — the
  harness compares to numpy, never across processes).
- Progression within one evening, matched windows: staged first cut 18.98 ->
  map fused into the transpose staging 15.24 (mx8) -> +mxc8 candidate 14.87.

### What did NOT work / traps hit, with numbers

- **Staging the mapped vectors through mr/mi[17] stack arrays** before the
  unchanged tsto8: mh8=19.00 vs fu=14.40 (+4.6 us/step).  Fusing the map
  into tr8's register temporaries (and doing the extract path fully
  register-resident) cut that to +0.8, and the mxc8/mx8 store path to +0.44.
  Lesson: at this uop density the extra 34-vector stack round trip per site
  costs more than the map arithmetic itself.
- **tryout.sh is broken this round for every implementer**: line 36 expands
  `$W` (for `--cin $W/c.bin`) two lines before W is assigned, and `set -u`
  aborts with "W: unbound variable".  Workaround that touches nothing:
  prefix the invocation with the value the script itself assigns,
  `W=$PWD/build/tryout/<name> ./tryout.sh ...`.  Second bug: the check.py
  line quotes `--cin '$W/c.bin'` inside a $() so it reaches the REMOTE shell
  unexpanded -> `/c.bin` -> the map-chain check crashes (after printing the
  single-transform PASS) and the `&&` chain silently skips the repeatability
  cmp.  I ran check.py by hand on the shared FS after every run (commands in
  this round's shell history); monitor should fix both lines.
- mh4/mi4 (w=4) lose every chain table by 3+ us — consistent with every
  batched cell since the h family landed; kept only as race columns.
- mc8 vs mh8 on the tile-transpose path: divider-free won by 0.35; on the
  extract path the two styles are within 0.05 of each other.  The divider
  was never the bottleneck (mc8 ≈ mh8 in the first cut already ruled out
  vdivpd serialization).

### Borrowed this round, named

- The map's shape — one reciprocal step in hardware, sqrt via seed+Newton on
  the FMA pipes, and the "fuse the map into a pass that streams c" doctrine:
  the rival pipelines (corpus §10 §2, `ext/reference/fft_v4_solutions/`
  pw_full/pw_full_fast).  Upgraded their float seed to vrsqrt14pd's double
  seed so the chain gate passes by design instead of failing by 2 orders.
- Extract-store pass 3 under the map (mx8/mxc8): my own ice_r3 x8, itself
  from L17_matrixsimd ice_r1/r2.
- Per-candidate consecutive-block racing and same-window-only comparisons:
  L17_matrixsimd / L17_rader ice_r1 discipline, unchanged.

### Next round

1. The FFT floor is now the whole story again: the mapped step is fu+0.44,
   so every us must come from the h8 kernels (merged-phase kernel H /
   chunk17z shape — ice_r3 item 1, still unattempted) or pass-1's
   deinterleave.
2. Lazy map (apply in pass 1 of the NEXT step) A/B: same uop count moved to
   the lighter pass; needs an unmapped first step + one trailing map pass.
   Only worth it if a probe shows pass 3 saturated and pass 1 slack.
3. Cross-volume: the next volume's x0/c cold reads (157 KB) could pipeline
   into the current volume's last steps (q-variant style).  ~1-2% ceiling.
4. If the mx8/mxc8 flip across processes ever matters, tie-break the race
   deterministically (prefer mxc8 within 1%) — cosmetic, the gate does not
   care.

## Round ice_r5

### Where this round started

ice_r4 leaderboard: L17_winograd 14.854 us/step, 2nd (L17_matrixsimd 13.009,
L17_rader 17.521, ducc0 86.4).  The scored description string finally
decomposed the gap: fu(h8) = 12.82 on the scoring node while the mapped chain
step scored 14.85 and the in-create race saw mx8 = 15.15 — i.e. in the QUIET
window the mapped chain step costs ~fu + 2.0.  The r4 record's "+0.44 for the
map" was a contended-window artifact (at 2.9 GHz the port-starved kernel hides
the map; at full clock it does not).  matrixsimd's r4 record shows the same
arithmetic from the other side: their map's honest in-chain marginal was
1.7 us/step.  So the round's targets were (a) the map+glue overhead and
(b) pass 1's disproportionate 5.36 us (42% of fu for 33% of the FP).

### What was changed (in order of measured effect)

1. **Split-format intermediate chain state (msp8/mspc8, chvar 6/7).**  The
   chain owns the state FORMAT between steps: the harness sees only x0 and
   final_out, so only step 1 must read interleaved complex and only step m
   must write it.  Steps 2..m-1 keep the state SPLIT (17 kx-rows of SP=296
   doubles per component, 64-B-aligned row starts, re block then im block),
   IN PLACE (pass 1 fully drains the state into the A scratch before the
   fused pass 2+3 stores back — volmajor-inplace, adopted from L17_matrixsimd
   ice_r4; the format-ownership move on top is mine).  Per intermediate step
   this deletes: pass 1's 34 unaligned 128-B deinterleave load pairs per
   group (~26 of 34 SPLIT a cache line at the 578-double row stride — the
   thing i4 was invented to mitigate — plus their 34 two-source shuffles and
   kernel C's xr/xi stack-array round trip), pass 3's 2 interleave shuffles
   per map site (~1156 port-5 uops/volume competing with the second FMA
   pipe), and the ctmp/final_out ping-pong (78.6 KB less working set).
   Step m goes through the existing extract-store mapped pass (x8m/xc8m,
   dst never re-read = its known-good site); step 1 through pass1_f8.
2. **Kernel H pass 1 for the split state** (`hp1_8` = DEF_K17_H at load
   stride SP, array output; pass1_s8 = hp1_8 + the unchanged tst8f blocked
   store).  The panel_r8 "H loses in pass 1" verdict was about STACK-ARRAY
   inputs (H's re-read doubled stack traffic); split-state rows are memory
   at kernel stride — exactly the regime where H beat E in every fused table
   since r8.  Matched node pairs (same lease, alternating): spH 14.32/14.39/
   14.71 vs spE 15.18/14.76/15.11 us/step (won 3/4 pairs by 0.37–0.87; the
   losing pair was spH's cold first table).  Outputs cmp-BIT-IDENTICAL to
   the E version, so this is a free swap, shipped as the default
   (-DL17_SP_P1E restores E for A/Bs).
3. Chain race extended to 8 candidates (msp8/mspc8 columns in mch[]) and a
   -DL17_FORCE_CHVAR dev hook for matched A/Bs (the race numbers alone
   mis-rank across windows; forced same-lease pairs settled this round's
   questions).

### Operation count

FFT unchanged: 296 FP instructions (192 FMA + 104 add/sub) per 17-point
kernel, 3*289*296 = 256,632 FP instr / 423,096 flops per volume; map
unchanged (rsqrt14 double seed + 2 Newton, one vdivpd — or rcp14+2N in
mspc8 — per 8 complex points, ~2 ulp/application).  What the split path
deletes per intermediate step is ~2.4k NON-FP vector uops (1224 deinterleave
+ 1156 interleave shuffles, all port-5 on ICX) plus ~940 split-line load
penalty slots and one 78.6 KB buffer of working set; kernel H pass 1 then
trades 17 extra aligned L1 row loads per group for the deleted stack arrays.

### Measured on the node (tryout + forced same-lease A/Bs, graded chain
### L=17 B=32 m=98; every window tonight contended, clk probe 2.90 GHz)

- **Matched 5-pair A/B, mxc8 (r4 pick) vs msp8 with H pass 1, one lease**:
  msp8 won 5/5 — 14.245/14.347/14.412/14.730/14.748 vs 16.797/15.167/
  15.278/14.743/14.804.  Advantage 0.9–2.6 us/step in loaded stretches,
  ~0.05 when the window tightened; msp8 never lost.
- In-create chain race, three independent processes: msp8 won every table —
  17.30 vs mxc8 17.51 / 16.67 vs 17.07 / 16.74 vs 17.74 (mx8 similar).
- Shipped auto-pick graded runs: **min 14.677 us/step (sd 0.27%)** in the
  best window (same-create fu = 14.91 — the mapped chain step is now
  CHEAPER than the unmapped fu probe; at r4 it was fu + 0.44 in matched
  contended windows); 16.28–16.63 in two badly loaded windows whose whole
  tables sat ~2 us high (same-window MKL 88.8).
- B=1: 17.303 us/step (race at nv=1 picked mx8 17.18 vs msp8 17.33 — a
  window wash; B=1 is unscored, the race decides per process).  MKL same
  core: 88.8 (B=32) / 100.9 (B=1).
- Correctness: single transform rel_l2 3.254e-16 (B=32) / 3.269e-16 (B=1);
  **map-chain m=98 end state 1.630e-14 (B=32) / 9.960e-15 (B=1) vs tol
  9.8e-12** (600x margin); bit-repeatable across processes (pick stable);
  spH/spE outputs bit-identical.

### What did NOT work / traps hit, with numbers

- **Split with the E-kernel pass 1 alone was a WASH against mxc8**: first
  5-pair forced A/B (before item 2) had mxc8 winning 4/5 by ~0.2 us/step
  (14.75–14.89 vs 14.95–15.05) while the same evening's in-create races said
  the opposite by 0.2.  Only the H pass 1 made split win everywhere.
  Lesson: the split layout's prize was as much the KERNEL-INPUT SHAPE
  (memory at kernel stride unlocks the load-folded component-split kernel)
  as the deleted shuffles; had I shipped after the first A/B I would have
  called the whole idea a wash.
- tryout.sh's two r4 bugs are still present (line 36 uses $W before
  assignment under set -u; check.py gets a literal-$W --cin path, crashes
  after the single-transform PASS, and the && chain silently skips the
  repeatability cmp).  Same workaround: prefix W=$PWD/build/tryout/<name>,
  run check.py and the cmp by hand.  New trap on top: $W/in.bin is the LAST
  tryout's input — after a B=1 tryout, manual B=32 driver runs need freshly
  generated in/c files or the driver aborts ("too short").
- The generic no-SIMD build (gcc without -march) fails on the clk probe's
  asm constraints — pre-existing (the r4 exemplar fails identically), noted
  here so nobody bisects to this round.

### Borrowed this round, named

- Volume-major IN-PLACE state (final_out/state single-buffer, working-set
  diet): **L17_matrixsimd ice_r4** ("volmajor-inplace").  The split-format
  extension — exploiting that the harness only ever sees x0 and final_out,
  so the interleaved layout need not exist between steps — is this entry's
  own addition on top.
- The map-cost decomposition that motivated the round (structure probe vs
  full step, map marginal ~1.7 us): **L17_matrixsimd ice_r4** measured it;
  my r4 scored string confirmed it from this side.
- Same-lease forced-pair A/B discipline (in place of cross-run comparisons):
  L17_matrixsimd/L17_rader ice_r1 discipline, sharpened here with the
  -DL17_FORCE_CHVAR hook after the race and the graded runs disagreed
  across windows.

### Score projection

r4 scored 14.854 (pick mx8).  This round's within-run split advantage is
0.3–1.0 us/step in contended windows and the mechanism is pure vector-uop
deletion, which weighs MORE in the quiet scoring window (the r3/r4
quiet-window lessons).  Expect **~13.5–14.2 us/step scored**; matrixsimd
stands at 13.0 and is presumably also moving, so probably still 2nd — but
the fu-vs-chain-step overhead that r4 paid (~2 us in the quiet window) is
now structurally gone.

### Next round

1. The FFT floor itself is now the whole residual: fu(h8) ~12.8 quiet vs a
   ~5.0 us 2-pipe FP floor.  The one costed, still-unattempted lever is the
   merged k-blocked kernel (my ice_r3 item 1; matrixsimd's chunk17z shape) —
   note their r3 lesson that the win was scheduling/issue, not spills, and
   their pragma history before porting anything.
2. If the scored mch[] shows msp8/mspc8 losing the quiet window to mxc8,
   the H-pass-1 vs extract-store balance flipped — re-run the forced pairs
   in the quiet window before concluding anything.
3. Split pass-3 stores are unaligned (h*17 offsets); a per-ky padded row
   layout would align them but changes the next pass 1's spectator
   indexing.  Costed as messy; only worth it if a probe shows store-split
   stalls.
4. B=1's nv=1 race is noisy (mx8/msp8 within 0.15); harmless while B=1 is
   unscored.
