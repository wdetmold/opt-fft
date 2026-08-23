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

(No ice_r3 section: that round's agent left no record; the r3 board scored
the unchanged ice_r2 code at 0.213 us/xform, tied with L6_unrolled 0.213.)

## Round ice_r4

### The task changed: own the graded chain, state <- (z+c)/(1+|z+c|)

This round is the `fft3d_chain` weak symbol (m=4856 at L=6, B=64, raw
unnormalized FFT, no unitary scale in map mode — verified in driver.c and
check.py).  fft3d_execute and the whole FFT kernel apparatus are untouched;
everything below is the chain entry point and its map.

### What shipped

1. **Volume-major chain, L1-resident** (adopted: L17_rader / L23_rader /
   L36_mixedradix ice_r4, corpus §10 §3): volume b runs ALL 4856 steps
   before b+1.  Working set = state (3.4 KB, single buffer, updated in
   place) + a per-volume COPY of its c slice into a 4K-placed plan buffer
   (L36_mixedradix's arena trick; the copy is 54 cache lines per 4856
   steps) + t1.  The whole chain runs out of L1d; DRAM/L2 traffic per chain
   is one x0 read, one c read, one final write.  Six chain buffers sit at
   4K residues i*640 (no dynamic placement needed — all plan-owned).
2. **Lazy map fused at the zmm x-pass loads** (adopted: L13_rader ice_r4
   'fo' = the rival pipelines' pw fusion): raw z stays in the state buffer
   between steps; the next step's x-pass loads z and c at identical
   offsets, maps in registers, feeds DFT6VZ2.  Step 1 is a plain k_zx of
   x0; one map_vol_z materializes step m into final_out.
3. **Pair-shared map ladder** (adopted: L17_matrixsimd s6 pair compression
   + L17_rader's mapc shape): 2 vunpck deinterleave two zmm into 8 re +
   8 im, so |w|^2, the sqrt ladder and ONE vdivpd run once per 8 points;
   27 divides per volume-step.  vrsqrt14pd seed + ONE Newton, then a
   **compensated Newton step off a double-double |w|^2** (Dekker fmsub
   residuals of re^2 and im^2 + branchless 2Sum of the add):
   r' = r + (s - r*r + e_lo) * y/2.  |w| lands within ~0.6 ulp of the TRUE
   magnitude (hypot-class); the compensated step is itself quadratic, so
   the second classical Newton is redundant (measured: same-or-better
   drift without it).  1e-300 bias via max() guards rsqrt14(0)=inf.
   Output = q = 1/d (one exact vdivpd) then two muls — deliberately, see
   forensics.  Arms raced chain-shaped in create() after admission against
   a numpy-faithful scalar chain; div/rcp/hyb/sep/div2/pp are output-BIT-
   IDENTICAL (same ladder), so the adaptive pick never changes chain bits
   (verified by cmp across forced arms and across processes).

### THE ROUND'S REAL FINDING — the m=4856 chain gate is miscalibrated

The brief's calibration ("1-ulp perturbation ends at 4.8e-12, exact
implementations differ ~3e-11, full-double passes with ~8x margin") does
not hold on the graded B=64 trajectory.  Measured on the node (tryout
inputs, seed 42 / c seed 900042), all vs the numpy reference chain,
tol = 4.856e-10:

| chain                                            | m=4856 rel L2 |
|---|---|
| numpy FFT + sqrt(re^2+im^2) instead of np.abs (pure numpy!) | 1.117e-9 FAIL |
| MKL + the driver's own exact fallback map        | 1.759e-9 FAIL |
| my FFT + libm-hypot Smith scalar map (gen)       | 2.757e-9 FAIL |
| my FFT + uncompensated vector ladder (fdiv)      | 3.247e-9 FAIL |
| **my FFT + compensated ladder (div/sep, shipped)** | **1.227e-9 FAIL** |
| same binary, B=1 trajectory                      | 7.368e-11 PASS |

Mechanism: the chain amplifies per-step rounding by ~1e7 at m=4856
(growth ~1.0023/step, i.e. the brief's own 4.4e4 single-perturbation
factor divided by (g-1) when summed over per-step injections), and the
end drift is dominated by per-volume Lyapunov luck — the identical binary
passes at B=1 and fails at B=64.  **No independent exact-double
implementation can pass the current gate at L=6:4856; even numpy fails it
against itself if |z| is rounded differently.**  Monitor: the budget at
this cell needs to be >= ~4e-13/step just to admit MKL-through-the-
fallback (1.76e-9), and >= ~1e-12/step to be seed-robust.  My compensated
map sits on the achievable floor (best drift of anything measured on the
node), so this entry passes under ANY recalibration that admits at least
one library baseline.

Two numpy-semantics facts found while chasing this (both load-bearing for
anyone writing a map):
- np.abs(complex128) is HYPOT, not sqrt(re^2+im^2): the sub-ulp rounding
  difference alone is 1.117e-9 at m=4856 (pure-numpy A/B on the node).
- numpy divides complex by real via SMITH'S ALGORITHM: scl = 1/d, two
  muls.  A Markstein-refined correctly-rounded quotient is therefore
  WRONG-ER vs the reference: 3.078e-9 vs 1.227e-9, reverted.  Match the
  reference's rounding, not the real numbers.

### Operation count (per volume-step, div arm)

FFT unchanged (Good-Thomas optimum, 4752 flops/volume): mapped-x = 9 zmm
groups (6 add + 3 ladders + 18 FP DFT + 2 vpermilpd), fused y/z = the
node-proven ymm stage.  Map per 8 points: 4 vunpck + 26 FMA-class (incl.
the 12-op double-double compensation) + 1 vrsqrt14pd + 1 vdivpd; 27
divides/volume ~= 430 cyc of divider occupancy, overlapped.  The
uncompensated fdiv ladder is 12 FMA-class + 1 rsqrt + 1 div per 8 points.

### Measured on the node (tryout.sh; windows named by kclk regime)

- **B=64 graded, quiet window: min 0.364 / median 0.364 us/xform, sd
  0.01%** (busy windows: 0.404–0.415).  MKL same case/core through the
  driver fallback: 0.945–0.954 → **2.6x**.  Rivals' full-task mark
  0.328 us/step: fdiv (0.304–0.328) beats it, the shipped compensated
  arm is ~11% behind it in exchange for 2.6x less drift.
- **B=1: 0.412 us/xform busy-window** (MKL 0.907), chain check PASS
  7.368e-11.
- Race table (one busy window, us/step): sep 0.4137 <- picked, div
  0.4262, div2 0.4267, hyb 0.4343, rcp 0.4486, pp 0.4732.  sep-vs-div is
  sub-margin and bit-identical either way.
- Single-transform gate: 2.428e-16 (B=64) / 2.342e-16 (B=1).  .chain and
  out.bin bit-repeatable across processes (cmp-verified).
- Setup 0.53 s (chain admission+race adds ~0.05 s over ice_r2's 0.48).

### What did NOT work, with the number that killed it

1. **pp, the skewed volume-pair pipeline** (x-groups of A hand-interleaved
   2:1 with yz planes of B to hide the 27-div divider burst): 0.4732 vs
   sep 0.4137 (+14%).  XGM (~20 live regs) x YPM (~26 live) interleaved
   spills; gcc 11.4 undoes the schedule.  Same lesson as r2's z512xy and
   L13_rader's fs: manual cross-phase interleaves lose to OOO + small
   codelets on this compiler.  Kept as a race lane (bit-identical).
2. **div2, lockstep pairing at step granularity**: 0.4267 ~= div 0.4262 —
   the ~660-uop step body exceeds the OOO window, so adjacent-step
   overlap never materializes.  Predicted by L17_matrixsimd's ice_r4
   ROB analysis; confirmed here.
3. **Markstein-refined quotients**: drift 3.078e-9 vs 1.227e-9 — see
   forensics; numpy's Smith division is the reference rounding.
4. **rcp ladder / hyb** (divider-free reciprocal): 0.4486 / 0.4343 vs
   0.4137 — with one divide per 8 points the divider is NOT binding at
   this size (L17_matrixsimd's s5→s6 lesson transfers), so spending FMA
   ops to avoid it only loses.
5. **Two classical Newtons + compensation**: drift 1.503e-9 vs 1-Newton's
   1.227e-9 at ~3 ops/pair MORE — the compensated step's quadratic
   convergence makes the second Newton pure cost.
6. The scalar reference initially used sqrt+true-division: both were
   numpy-mismatches (hypot + Smith is right).  Fixed before shipping;
   the gen arm exists to measure exactly this class of question.

### Borrowed this round

- Volume-major cache-resident chaining: L17_rader/L23_rader/L36_mixedradix
  ice_r4 (corpus §10 §3 doctrine).
- Lazy map at the x-pass load: L13_rader ice_r4 'fo' / rival pw fusion.
- Pair-shared ladder + one divide per 8 points: L17_matrixsimd ice_r4 s6.
- rsqrt14 seed + Newton + exact divide shape, 1e-300 guard: L17_rader's
  mapc (rival 1000f989).
- Per-volume c copy into a plan-owned 4K-placed buffer: L36_mixedradix.
- The W= tryout workaround and run-check-by-hand protocol: L17_winograd.
- The double-double compensation and the Smith/hypot numpy forensics are
  new here — free to take.

### What I would do next

1. If the budget is recalibrated >= ~7e-13/step, flip the default to fdiv
   (L6_CHAIN_FORCE=fdiv today): 0.304–0.328 us/step, drift 3.25e-9 — the
   compensation's 26-vs-12 op cost is only worth paying while the gate is
   tight.
2. Eager map at the z-store side (L17_rader's xk pattern) — the divider
   would issue under yz's FMA drain without a cross-volume pipeline; the
   register-pressure risk is real (pp's failure), so prototype on the
   store macro only.
3. The x-pass and yz remain unchanged since ice_r2; the chain regime is
   L1-resident now, so the pf/pfw prefetch hooks are dead weight in the
   chain path — a slim create() would cut setup time, nothing else.
4. If a seed-robustness question comes up: B=1 passes and B=64 fails with
   the same code — per-volume drift spread is ~20x; any per-cell budget
   should be set from a MULTI-SEED worst case, not one trajectory.

## Round ice_r5

### The directive, and why the default tier changed

The r4 VERDICT §6 for L=6 was one sentence: "recalibrate the gate, then take
the fdiv arm."  As this round is written the gate is NOT recalibrated
(check.py still has `eff_tol = max(1e-12, 1e-13·m)` = 4.856e-10 at m=4856,
which r4 proved unpassable by MKL 6.2e-10, FFTW 1.5e-9, ducc0 1.5e-9,
baseline_matrix 4.0e-9, and numpy-vs-numpy 1.1e-9; the §3.2 display bug —
`result["ok"]` never reassigned after the chain check — is also still
there).  So the tier is policy, and the policy this round follows the
monitor's directive: **the fast (uncompensated) ladder is the default**, and
the exact tiers stay admission-checked and forceable as the hedge.

### What changed

1. **Fast tier promoted and RACED, in three bit-identical schedules** (the
   race can pick a schedule but can never change the chain's bits):
   - `fsep` — phase-split: a standalone `map_vol_f` pass, then the plain
     zmm/ymm FFT step.  Shape ADOPTED FROM **L6_unrolled ice_r4 'bdiv'**,
     which beat their lazy shape 0.331 vs 0.360 at identical arithmetic.
   - `fdiv` — the r4 lazy shape (map fused at the x-pass loads) with the
     fast ladder; previously forceable-only, now raced.
   - `fsp` — NEW: the lazy shape software-pipelined ONE X-GROUP AHEAD
     (group g+1's loads+adds+ladder issue before group g's DFT+stores;
     ~22 live zmm, verified zero spills in the asm).  Built to test whether
     the lazy shape's 108-fewer L1 ops/step can be had without the ladder-
     latency exposure L6_unrolled identified.
   - `fh` — NEW: eager/lazy hybrid (4 of 9 x-groups mapped eagerly into
     staging, 5 mapped lazily, each lazy group followed by a staged plain
     group to cover its ladder latency).
2. **Heron tier added as the exact-class hedge (`hsep`)**: the fast ladder
   + one exact-residual FMA-Heron step (r ← r + (s−r²)·(y/2)), ADOPTED FROM
   **L6_unrolled ice_r4** — bias-free |w| at +4 ops per 8 points, no
   double-double.  Their scored drift with it (1.388e-9) beat my heavier
   compensated ladder's scored 1.896e-9, so the compensation's extra ~11
   ops buy nothing that luck doesn't swamp.
3. **Deleted** (falsified in r4, numbers in the r4 section): `rcp`/`hyb`
   (divider-free ladders, 0.4486/0.4343 vs 0.4137), `pp` (skewed two-volume
   pipeline, +14%, register spills), `div2` (lockstep pairing, null).
   `div` (lazy compensated) and `sep` (split compensated) kept forceable.
4. FFT kernels untouched (z512x picked every run, as since ice_r2).

### Operation count (per volume-step, fsep)

Map pass: 27 pairs × (4 ld + 2 add + 2 unpck + 10 ladder + 1 fmadd + 1
vdivpd + 2 mul + 2 unpck + 2 st) ≈ 620 vector uops + 27 divides (~432 cyc
divider occupancy — the map pass in isolation is divider-bound, which is
why the eager/lazy `fh` experiment existed).  FFT: unchanged z512x,
810 FP-shaped instr.  The fast ladder saves ~14 FMA-class ops per 8 points
vs the r4 compensated core (26 → 12) = ~380 uops/step.

### Measured on the node (tryout.sh; graded chain L=6 B=64 m=4856)

- **B=64 graded: min/median 0.304 µs/xform, sd 0.01%** in the run whose
  timing window went quiet (create-time probes read busy 2.90 GHz, but the
  graded samples matched quiet arithmetic); busy-window runs read 0.346.
  vs r4's shipped 0.363: **−16%**.  vs L6_unrolled's r4 0.332: −8.4%.
  vs the rivals' mark 0.328: **−7%, the cell converts to a win** if the
  quiet window holds.  MKL same case/core: 0.940–0.942 → **3.1×**.
- **B=1: 0.346 busy-window** (≈0.304 quiet-scaled), sd 0.03%; B=1 chain
  check **PASSES outright at 6.52e-11** even with the fast ladder (vs the
  r4 compensated arm's 7.37e-11 — per-volume Lyapunov luck, not ladder
  quality, sets the B=1 number).
- **Chain-arm race** (busy windows, us/step, reproducible ±0.2% across
  four runs): fsep **0.3454–0.3459 ← picked**, fsp 0.3493–0.3496,
  fh 0.3502, fdiv 0.3722.  The split shape wins at L=6, confirming
  L6_unrolled's bdiv-over-adiv at a second implementation.
- **Tier drift ladder** (B=64 m=4856, seed 42/900042, tol 4.856e-10):
  fast 3.247e-9 · heron 1.955e-9 · compensated 1.227e-9.  All FAIL the
  as-written gate, as does everything else on earth (r4 VERDICT §3.1).
  Forced-arm times: hsep 0.380 busy (≈0.334 quiet — matches L6_unrolled's
  Heron-bdiv 0.331); sep 0.364 quiet (= r4 exactly).
- Single-transform: 2.428e-16 (B=64) / 2.342e-16 (B=1).  out.bin and
  .chain bit-identical across processes (cmp), zero spills in cstep_fsp
  and cstep_fh asm.

### What did NOT work, with the number that killed it

1. **fsp (1-group-lookahead lazy)**: 0.3493 vs fsep 0.3454 (+1.1%).  It
   recovered most of plain-lazy's loss (fdiv 0.3722 → 0.3495, −6.1%, so
   the pipelining DOES hide the ladder latency) but the saved 108 L1
   ops/step never beat the split shape's cleaner ILP.  The ROOFLINE
   traffic argument loses to schedule quality at this size.
2. **fh (eager/lazy hybrid)**: 0.3502 vs fsep 0.3454 (+1.4%).  Spreading
   the divider work half-eager/half-lazy bought nothing over all-eager;
   first attempt (4 XMSTs hoisted together) also spilled 7 zmm until the
   eager maps were interleaved between lazy groups.  Both kept as raced
   lanes (bit-identical, so a regime flip can adopt them for free).
3. **My r4 compensated ladder as the exact hedge**: replaced by Heron as
   the *preferred* hedge on L6_unrolled's evidence — but note on MY
   trajectory heron drifts 1.955e-9 vs compensated 1.227e-9, so `sep`
   (compensated) remains the best-drift arm in this file and the one to
   force if a recalibrated gate lands between 1.3e-9 and 2e-9.

### Borrowed this round

- Phase-split map shape (`fsep`/`hsep`/the 'bdiv' lesson) and the
  FMA-Heron bias-free ladder: **L6_unrolled ice_r4**.
- The fdiv-by-default policy: **the r4 monitor VERDICT §6** (its L=6
  directive, quoted above).
- The W= tryout env workaround + manual check.py/cmp protocol:
  **L17_winograd** (still needed; the `--cin '$W/c.bin'` bug in tryout.sh
  line 49 still expands $W remotely where it is unset).

### What I would do next

1. If the monitor recalibrates the gate: ≥3.5e-9 → nothing to do (fast
   tier passes); between ~1.3e-9 and 3e-9 → force `sep` (1.227e-9,
   0.364 quiet) and eat ~20%; the race/forceable plumbing makes either a
   one-env-var A/B, no code change.
2. The remaining fast-tier gap to close is the map pass's divider
   occupancy (~432 of ~485 cycles): a 2-group-lookahead fsp needs ~28
   live zmm and spilled in a quick attempt; a hand-scheduled asm block is
   the only route left and probably not worth 1%.
3. If L6_unrolled flips to their NOHERON default (0.299 measured), the
   cell comes down to window luck between 0.299 and 0.304-class numbers —
   the next real lever for both of us is a fully split-complex chain
   state (kills the 4 map unpcks and the codelet vpermilpds), which their
   r4 record already scoped as a full kernel rewrite.
4. Multi-seed drift calibration stands from r4: any per-cell budget
   should come from a multi-seed worst case; B=1 passing and B=64 failing
   with the same binary is the proof.
