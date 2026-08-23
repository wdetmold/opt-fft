# L17_matrixsimd — strategy record (ICE LAKE panel)

Continuation of this entry's records from the single-thread panel
(`bench/geom/strategies/L17_matrixsimd.md`, rounds panel_r1..r11) and the
multicore panel (`bench/mt/strategies/L17_matrixsimd.md`, mt_r1..r4).  Those
files are the full history — the do-not-retry list, the bit-class rules, the
probe designs.  This file records what happens on the Ice Lake panel
(bare-metal Xeon Gold 6326, 2×512-bit FMA pipes, graded chain workload,
`cases.txt` 17:32:98).

## Round ice_r1

### Where this round started

ice_smoke leaderboard (the panel_r11 code, unmodified, on the graded chain):
L17_matrixsimd 2061.465 us per volume-chain = **21.03 us/transform step**,
1.30× BEHIND L17_winograd (1586.374 = 16.19 us/step).  On CLX we led the
board; on this machine and this workload we were second.  Diagnosis, from the
smoke description string, the rivals' records, and the corpus (§10):

1. **The chain regime was never tuned for.**  The graded workload is
   `execute(B=32)` → driver-side unitary scale of the whole 2.4 MB output →
   output becomes next input, ping-ponging two destination buffers; all three
   buffers L3-resident (7.2 MiB total, L2 = 1.25 MB).  Our batch<64 tuner
   arena was 16 volumes, fixed src→dst, no scale pass — a different regime.
   Winograd's stage-2 tuner fires at batch≥8 and tuned in-regime; its
   chain overhead was ~2.8 us/step, ours ~8.7.
2. **The class rule barred the right pass order.**  The X-first family
   (spreads the output writes across the plane phase) was gated to batch≥64
   from CLX DRAM-streaming data.  At B=32 the shipped X-last order serialises
   a 73-chunk RFO burst into `out` at the end of every volume — against
   L3-resident buffers that burst is exactly what X-first amortises.
3. **Clock probes read 2.90 GHz (the base clock) for everything** while
   winograd's read 3.50/3.30.  Our create() is short (0.227 s); on this
   node's schedutil governor the whole tuner and every probe likely ran on a
   partially unramped core.  (No AVX-512 licence cliff on ICX — corpus §10,
   three sessions checked, and winograd's clk512=3.30 confirms mild.)
4. **The kernel is issue-limited on 2-pipe machines, and port 5 is no longer
   free.**  b1dec on ice: yz/kyz/x/kx = 8.54/8.01/3.80/3.65 us — L1-hot ≈
   in-situ ≈ the CLX numbers, i.e. ~1.03 FP ops/cycle on a machine whose two
   512-bit FMA pipes (ports 0+5) could do 2.  The 40 vshuff64x2 per Y/Z
   chunk (tile transpose) are port-5-only on ICX and now steal slots from
   the second FMA pipe; the CLX-era claim "transposes hide under the FMA
   stream" is machine-specific and FALSE here.

### What was changed (in order of measured effect)

1. **Chain-shaped tuner stage (stage 1g, 17 ≤ batch < 64).**  Candidates are
   timed under the driver's own loop: nv = batch volumes, output unitarily
   scaled after every execute and fed back as the next input, ping-ponging
   two destination arenas (values stay O(1) — no overflow, no denormal
   drift).  6 steps/unit, blocked, 1 warmup + min of 3; joint (pf,pw,pt)
   grid on the winner, same never-sequential rule as stage 2.  The old
   16-volume stage 1 now covers batch<17 only; stage 1c (pt A/B) narrowed
   to batch<17 with it.
2. **Class rule moved: class D (X-first pinned) now starts at batch 17.**
   Chain-shaped table on the node (B=32, us/step, includes the scale pass):
   X-last best (xlda, deferred-Z + addr-safe) 15.80; **X-first addr-safe
   xfa 14.36, xfda (deferred-Z) 14.29**; plain xf 16.99 (addr-safe worth
   −15% here — the chain's deterministic heap layout makes the r8
   de-aliasing tables earn their keep on ICX, where they were a wash on
   CLX); pipelined-512 17.11; staged input/output 16.3–16.7; **NT stores
   28.9–30.8 (catastrophic L3-resident — RFO avoidance pays only at
   DRAM)**; every 256-bit variant ≥ 19.1 (2×512-pipe machine).  batch<17
   keeps class B.
3. **Clock-settle spin** (~150 ms dense 512-bit FMA at the top of create(),
   adopted from L17_winograd's tuner protocol) so rankings and probes run on
   a ramped core.
4. **Extract-store transposing stores (new candidates xfax/xfdax/xlax/xldax,
   FORCE 56–59).**  The tr=1 tile transpose rerouted through
   memory-destination `vextractf64x2` (68 16-byte stores replace 40
   port-5 shuffles + 20 stores; ICX commits 2 stores/cycle and the shuffles
   were competing with the second FMA pipe).  Values stored are bit-identical
   lanes of the same vectors — kept inside the existing bit classes,
   cmp-VERIFIED on full outputs + chain outputs on the node before being
   made selectable.  Implemented as a `xst` compile-time branch in a shared
   `chunk17n_g` body with thin `chunk17n`/`chunk17nx` wrappers; the ymm tail
   keeps the tile path (no 512-bit shuffles there).  Node cmp results
   (B=8, chain m=4, full out + chain-end buffers): F50==F56, F51==F57,
   F48==F58, F49==F59 all bit-identical; the refactored no-force binary is
   also bit-identical to the untouched smoke binary at B=8 (class B
   end-to-end regression).  objdump confirms gcc 11.4 emits the
   memory-destination `vextractf64x2 $imm,%zmm,(mem)` form (510 sites).
5. **Pre-RA instruction scheduling, shipped as a source pragma**
   (`#pragma GCC optimize("schedule-insns","sched-pressure")` covering the
   chunk kernels and exec variants).  GCC does no pre-RA scheduling on x86
   and keeps text order, so the chunk's phase-serial source (cosine block →
   sine block → combine/stores) reached the issue queue unmixed — this is
   the "intra-kernel issue limitation" the r9 verdict declared structural
   on CLX, and pre-RA scheduling is the corpus's cure (§10 GCC item:
   ~+20% on prime passes; ZIPP interleaving is the by-hand equivalent).
   Matched A/B on the node's graded chain, runs minutes apart:
   plain 15.732 / flags **14.528 us/step (−7.7%)**.  Outputs unchanged
   (contraction happens in combine, before sched1); it must be a pragma
   because the Makefile CFLAGS are fixed.
6. `-DL17_VERBOSE_BUILD` hook (tryout.sh cannot pass environment through
   ssh; a -D flag can) so the tuner table is readable from dev runs.

### Measured on the node (tryout.sh, graded chain L=17 B=32 m=98)

- smoke (panel_r11 code): 21.03 us/step median (leaderboard).
- + settle spin + chain-shaped stage (class rule still old): min 19.714,
  median 20.385 — the chain tuner alone moved the X-last pick to
  xlda and recovered ~0.7 us.
- + verbose table run, same code: picked xlda 15.80 in-tuner; graded run
  **min 15.384 / median 15.440 us/step** (sd 0.32%) — already ahead of
  winograd's smoke 16.19.
- + class D at batch 32 (X-first family selectable): tuner kept xfda
  (pf=1, pw=0, pt=1 — the grid's 15.66 vs 15.77 for all-off is inside
  noise; flags are bits-neutral), graded run **min 15.695 / median 15.698
  us/step, sd 0.01%**.  NOTE ON CROSS-RUN COMPARABILITY: this run's whole
  table sat ~1.3 us above the earlier one (xfa 15.67 vs 14.36; xlda 17.24
  vs 15.80) — other implementers' leases were active; the first table ran
  right after the scoring window drained.  Within-run contrasts are the
  trustworthy ones: X-first addr-safe beats X-last addr-safe by ~1.6
  us/step in matched conditions.  Under the monitor's quiet window expect
  ~14.3–15.4 us/step.
- MKL same case/core: 76.5 us/step (we are ~5×).
- B=1 chain (class B, tuner path untouched): min 13.719 / median 13.721
  us/step (sd 0.02%), rel_l2 3.2e-16, chain check 1.09e-14, repeatable;
  MKL 73.9.  The batch<17 cells are healthy after the round's changes.
- **FINAL SHIPPED STATE, on the node**: graded chain B=32 **min 14.905 /
  median 14.908 us/step** (sd 0.02%; matched-conditions plain build was
  15.732, and the same code saw 14.53–15.70 across the day's load levels);
  B=1 **14.188 us/step**.  rel_l2 3.253e-16 (B=32) / 3.226e-16 (B=1);
  chain-98 closed-form check 1.075e-14 (tol 9.9e-12); bit-repeatable.
- The scheduling pragma's B=1 cost, measured as a matched A/B
  (-DL17_NO_SCHED hook): 13.756 without vs 14.188 with (+3.1%) — it hurts
  the X-last class-B pick while buying −7.7% on the scored X-first chain
  cell.  Kept file-wide because only cases.txt (17:32:98) is scored on this
  panel.  If B=1 ever returns to scoring, scope the attribute per bit class
  (all members of a class must share it — flagging must never split a
  class).

### What did NOT work / negative results with numbers

- **Extract-store transposing stores LOSE ~1–1.7 us/step** despite the
  port-5 arithmetic: xfax 16.61 vs xfa 15.67; xfdax 16.70 vs xfda 15.64;
  xlax 18.48 vs xla 17.45; xldax 18.97 vs xlda 17.24 (all within one run).
  Post-mortem: the Y group's stores into the plane buffer are consumed as
  full 64-byte vector loads by the Z group ~5 chunks later; a 64 B load
  spanning four 16 B stores cannot store-forward and stalls for the store
  buffer to drain — the forwarding penalty swamps the port-5 relief (which
  the b1dec numbers say the issue-limited kernel could not bank anyway:
  measured ~137 cycles/chunk vs the 98-cycle port floor, so port-5 was not
  the binding constraint).  A Z-group-only variant (its stores go to
  out/t1, read only hundreds of chunks later) would dodge forwarding but
  caps at ~0.3 us/volume of port relief the issue limit would eat; not
  built.  The twins stay in the candidate lists (bit-identical,
  cmp-verified, ~40 ms of tuner time) as machine insurance.

- NT stores at the graded cell: 28.9–30.8 us/step vs 14.3–17 for plain
  stores.  Sixth round running that NT loses everywhere except true DRAM
  streaming; on an L3-resident chain it is not even close.
- pt=1 (in-pass source prefetch) on the xlda winner: 16.17 vs 15.87 —
  declined, consistent with every prior node cell.
- 256-bit width at the graded cell: best 19.15 vs 14.29 — on 2×512-pipe
  ICX the width question is settled the opposite way from what winograd's
  h4-wins-at-256 suggests for THEIR structure; our lanes-are-lines chunks
  are 512-bit through and through.
- Staged-input/staged-output twins (r10/r11 mechanisms, built for CLX DRAM
  streaming): 16.3–16.7 — beaten by plain X-first addr-safe; the L2 stage
  round-trip buys nothing when the source is already L3.

### Borrowed this round

- Chain-regime in-plan tuning and the clock-settle spin: from
  **L17_winograd** (its r2 stage-2 "tune on the scored working set" and its
  150 ms settle; its ice_r1 agent crashed, so its smoke number is its
  panel_r11 state — the ideas were taken from its record).
- Port-5/second-FMA-pipe arithmetic and the "front-end/uop count decides,
  not flops" framing: corpus §10 (ice forensics) + LITERATURE.md ICX port
  notes.

### Score projection

Winograd's smoke number (its ice agent crashed, so it stands at panel_r11
state) is 16.19 us/step measured in the quiet scoring window.  Our shipped
build measured 14.905 under dev-slot load and ~14.5 flagged under similar
load; in the quiet window expect ~14–15 us/step — this entry should retake
L=17 unless winograd's next round moves.

### Open for next round

- The kernel ran ~1.0–1.1 FP ops/cycle from L1 against a 2/cycle machine
  (b1dec kyz≈8.0, kx≈3.65 at ~2.9 GHz); this round's scheduling pragma
  recovered part of that (−7.7% on the scored cell) but the residual is
  still issue-shaped.  Next lever, if a round wants it: merge the cosine
  and sine phases at the SOURCE level (ZIPP-style) — the two phases
  currently RE-LOAD the same 16 rows twice per chunk, so a merged
  single-load phase drops 16 loads/chunk and hands the scheduler 17
  independent chains explicitly instead of asking sched1 to find them.
  Costed at ~31 live registers, right at the spill boundary that killed
  the r1 monolith, so it needs the k-blocked shape.  Re-read the b1dec
  numbers from the next leaderboard first (they now run after the settle
  spin and under the pragma — the old 8.01/3.65 baselines are stale).
- b1dec/clk probes re-read on a ramped core (this round's settle spin) to
  re-price the B=1 cell on ICX.
- If the monitor's next leaderboard shows winograd's ice agent recovered,
  re-read its record before spending anything on the kernel.

## Round ice_r2

### Where this round started

ice_r1 leaderboard: L17_matrixsimd **14.471 us/step**, first at L=17
(winograd 16.113, rader 19.099, MKL 76.0).  The quiet-window tuner picked
xfax (X-first, addr-safe, extract-store) with pt=1 — extract-store won in
the scoring window despite losing every contended dev table, which says the
quiet window is more compute-bound than any window we can measure in.
b1dec on the leaderboard string: yz/kyz/x/kx = 8.67/7.90/4.30/3.91 —
L1-hot ≈ in-situ for the fifth round, i.e. the residual is IN the kernel
(issue/dependency-shaped), exactly what ice_r1's "open for next round"
declared.  This round executed that declared lever.

### What was changed

1. **Merged-phase (ZIPP-style) kernel `chunk17z`** (the ice_r1 record's
   costed next lever; interleaving idea from corpus §10, source-level this
   time instead of the scheduling pragma).  The shipped kernel's cosine and
   sine phases each load the same 16 source rows (33 row loads per chunk);
   chunk17z loads every row ONCE (17): it walks m = 0..7 in the sine
   phase's own order, parks the four first-half u values in registers
   (U0..U3), and closes cosine pair p as row m = p+4 arrives.  Every
   accumulator chain still sees its own operands in the exact shipping
   order (sine gets w_0..w_7 in m order, cosine/X0a get pairs 0..3 in
   order), FP count unchanged at 148/chunk.  Also, being straight-line, it
   deletes chunk17n's rolled cosine loop (gcc keeps that loop rolled by
   design — unrolling it in chunk17n makes gcc spill, r1 lesson — but the
   merged shape unrolls for free because values die immediately).
   Disassembly on the Ice Lake ISA: **246 -> 220 dynamic instructions per
   tr=0 zmm chunk (-10.6%), 272 -> 246 per tr=1 group chunk (-9.6%)**, FP
   mix identical (84 fma + 12 mul + 52 addsub), at the cost of ~31
   stack-operand references per chunk (the 32-register file is exactly
   full: 17 accumulators + 8 pinned K + U0..U3 + temps).
2. **Four new candidates** xfza / xfdza / xfzax / xfdzax (FORCE 60–63,
   cand[] 54–57): merged kernel in the Y/Z groups AND the X pass, X-first,
   addr-safe t1, plain/deferred-Z crossed with tile-store/extract-store.
   The ymm tail keeps chunk17n (a 16-register file cannot hold the merged
   live set).  The exec macros L17_EXEC_MXF/MXFD gained XP/XPT parameters
   (the xpass macro names) — textual only; the pre-change no-force outputs
   were re-verified (PASS + repeatable) before anything else.
3. **Bit-class discovery — the merged kernel is a NEW rounding class Z, not
   class D.**  On paper it is the same arithmetic in the same per-chain
   order; a host build with -ffp-contract=off is bit-identical to chunk17n.
   With contraction ON, gcc's FMA formation lands differently in the two
   source shapes once they are inlined into the 72-chunk loop (single
   isolated chunk: bit-identical; loop context: 1-ulp diffs) — same op
   COUNTS, different placement.  Node cmp on full chain outputs (B=8,
   m=98): F60 == F61 == F62 == F63 mutually bit-identical, all four differ
   from F50/F51.  Correctness rel_l2 3.257e-16, chain 1.075e-14,
   repeatable.  So the four form their own clean class.
4. **Class rule moved: batch >= 17 now selects within class Z** (selZ =
   {54,55,56,57}); batch < 17 keeps class B untouched.  Evidence,
   chain-shaped within-run contrasts on the node (cross-run numbers stay
   contention-poisoned — ice_r1 lesson, reconfirmed):
   run 1 (contended, everything ~1.5 us high): xfdza 16.46 / xfza 16.60 vs
   class-D best xfda 16.69 / xfa 16.77;
   run 2 (quiet-ish): xfdza 15.04 / xfza 15.17 vs xfda 15.20 / xfa 15.27.
   Merged wins ~1% in both windows, deferred-Z merged best in both.  Small
   but reproducible, and the mechanism is a pure dynamic-uop deletion in
   the identical pass schedule, so it should not invert in the quieter
   scoring window (where compute weighs MORE, per the xfax quiet-window
   pick).  The (pf,pw,pt) grid hooks were extended to 54–57.

### Operation count

Unchanged in FP: 148 ops per zmm chunk (96 mult-ops + 52 add/sub), 527
kflop/volume.  What changed is the non-FP stream: 16 of ~65 loads per zmm
chunk deleted plus the rolled-loop overhead, 246 -> 220 dynamic
instructions (tr=0).

### Measured on the node (tryout.sh, graded chain L=17 B=32 m=98)

- Old (ice_r1 shipped) code, same day, quiet-ish window: min 14.200 /
  median 14.204 us/step (sd 0.26%) — for cross-day reference only.
- Within-run candidate tables: see item 4 above (the round's real
  evidence).
- **FINAL SHIPPED STATE**: graded chain B=32 **min 14.762 / median 14.770
  us/step (sd 0.04%)** under mid-grade load (MKL on the same core read
  79.3 with sd 13%, vs 76.4 in the morning quiet — the absolute number is
  window-shaped; the within-run tables are the comparison that counts).
  rel_l2 3.254e-16, chain-98 check 1.075e-14, bit-repeatable.
- B=1 chain (class B, tuner path untouched): **min 14.230 / median 14.236
  us/step (sd 0.04%)**, rel_l2 3.226e-16, chain check 1.086e-14,
  repeatable; MKL 84.1 on the same core.  Matches the ice_r1 shipped
  14.188 — the batch<17 cells are unharmed by the round's changes.

### What did NOT work / negative results with numbers

- **Merged + extract-store composes badly** (both within-run tables):
  xfzax 16.97 / xfdzax 17.15 vs xfza 16.60 / xfdza 16.46 (run 1); 15.59 /
  15.70 vs 15.17 / 15.04 (run 2).  Extract-store pays its store-forwarding
  tax (ice_r1 post-mortem) for port-5 relief the merged kernel no longer
  needs — and the xst body spills hardest (446 stack refs in the exec vs
  417 plain-merged, 259 incumbent).  Both twins stay in class Z as
  candidates (~40 ms of tuner time) in case the quiet window flips the
  ranking the way it flipped xfax in ice_r1.
- **The merged kernel is NOT bit-identical to class D despite identical
  per-chain operand order** — gcc 11.4's loop-context FMA placement is
  shape-sensitive even at equal op counts.  Lesson for every future
  "pure scheduling, same order" claim: the r3 rule (cmp, never derive)
  remains mandatory even when the derivation looks airtight.

### Borrowed this round

- The ZIPP source-level interleaving cure: corpus §10 (GCC item on prime
  passes), continuing the ice_r1 adoption of its compiler half (the
  scheduling pragma).
- Read L17_rader's ice_r1 record before starting: its "the cell is
  latency-bound, width/port tricks wash at B=32" conclusion is about ITS
  plane pipeline (probe ph = 70% of its cell); our b1dec says our kernel is
  issue-bound, so the kernel lever remained correct for us.  Its plan to
  port the winograd engine wholesale is noted as the rival threat to watch.

### Score projection

Shipped 14.76 under mid-grade load; the ice_r1-shipped code measured 14.20
in the same morning's quieter window, and merged is ~1% under the incumbent
within-run, so expect **~14.0–14.4 us/step in the scoring window** (ice_r1
scored 14.471 for the previous code).  Winograd stands 16.11; margin
should widen slightly.

### Open for next round

- **Spill diet for chunk17z**: ~31 stack refs/chunk because the register
  file is exactly full.  Candidates: park U0..U3 in the L1 scratch
  (4 stores + 4 loads, frees 4 registers for temps), or unpin K4..K7 in
  the merged body only.  Each is a new bit-class risk -> cmp first.
- **Merged X-last twin for class B** if B=1 ever returns to scoring
  (B=1 keeps the old kernels this round; the merged win is issue-side and
  should transfer).
- Re-read the scored description string: which class-Z member the quiet
  window picks (extract-store flipped there in ice_r1), and the new b1dec
  numbers — kyz should drop ~8–10% if the merged kernel does what the
  disassembly says.
- If winograd's agent recovered and moved, re-read its record before
  spending anything; the 3-pass engine port threat from L17_rader's plan
  applies to it, not to us.

## Round ice_r3

### Where this round started

ice_r2 leaderboard: L17_matrixsimd **13.562 us/step**, first at L=17
(winograd 16.082, rader 18.693, MKL 76.3).  Quiet-window pick: xfdza
(merged, X-first, deferred-Z, addr-safe), pf=pw=pt=0 — extract-store did
NOT flip this time.  b1dec = 7.72/7.12/3.56/3.39: the merged kernel
delivered the predicted ~10% on kyz (7.90 -> 7.12), residual still
issue-shaped.  The r2 record's declared next lever was a spill diet for
chunk17z (~31 stack refs/chunk, register file exactly full).

### What was changed (in order of measured effect)

1. **THE FILE-WIDE SCHEDULING PRAGMA IS GONE — the round's win, and it was
   sitting in our own blind spot.**  Trigger: L13_direct's ice_r2 record
   measured the pragma we exported to the panel at **+5.2% on THEIR fused
   single-load kernel** and wrote the rule: *the pragma transfers only to
   phase-serial kernels* — sched1 interleaves independent chains that the
   source keeps apart, and a source-merged kernel has none left, so its
   live-range stretching only feeds the allocator.  Our ice_r1 pragma win
   (−7.7%) was measured against the PHASE-SERIAL kernel; ice_r2 merged the
   phases at source level and never re-measured the pragma.  Instrument:
   no-sched twins of the merged execs compiled per-function
   (push_options), so the A/B is within one binary and one window.
   In-tuner: 13.68–13.75 (no-sched) vs 15.70–15.82 (sched), i.e. −13%
   in the arena; matched GRADED runs, same day: 13.991 (sched default) ->
   **13.493 (unsched default), an honest −3.6%** — the arena inflates the
   sched penalty, the graded contrast is the real one.  B=1 gets
   **14.236 -> 12.071 (−15%)** for free.  Static spill counts barely move
   (533 vs 527 stack refs/exec), so the mechanism is sched1's PLACEMENT
   (loads clustered away from consumers), not spill deletion.  The old
   scheduled codegen of the merged family is kept as FORCE 70–73
   insurance; -DL17_SCHED_ALL restores the old file-wide behaviour for
   matched A/Bs.
2. **chunk17zr — reordered merged kernel, NEW BIT CLASS R, now the
   batch >= 17 class.**  The r2 spill-diet lever, executed: walk m in pair
   order 0,4,1,5,2,6,3,7 so each cosine pair closes one step after it
   opens — ONE parked u instead of four (U0..U3), 3 registers handed back.
   The sine accumulators then receive their updates in the new m order, so
   this is a different rounding fixed point by construction (not a
   gcc-surprise like r2's class Z: this one was DESIGNED as a new class).
   Same 17 loads, same 148 FP ops, same slot table.  Same-window in-plan
   tables, three runs: 15.86 vs 15.98 (sched build), 13.68 vs 13.75
   (ns twins), **13.61/13.71 vs 13.80/13.75 (unscheduled default)** — R
   never lost, worth ~1%.  Class R = {58 xfzra, 59 xfdzra, 62 xfzrao,
   63 xfdzrao}, node-cmp'd mutually bit-identical on full + chain outputs
   (B=8, m=98) before the class rule moved; batch < 17 keeps class B.
3. **ov — cross-volume X-overlap twins (adopted from L13_direct ice_r2's
   "ov", concept originally L17_rader ice_r1's "sp").**  Volume b+1's 73 X
   chunks interleaved 4–5 per plane into volume b's plane phase, on the
   CURRENT pipeline (addr-safe shifted t1/t1b ping-pong, padded stride,
   merged kernels) — our panel_r4 pipelined exec predated all of that,
   so its ice_r1 loss (17.11 vs 14.36) did not condemn the idea.  Verdict
   at THIS cell: **loses ~0.2–0.6 us/step in every table** (14.01–14.32 vs
   13.61–13.80 unscheduled; 15.37–15.65 vs 15.15–15.24 final run).
   Diagnosis: L=13's cell is latency/L3-bound so overlap-with-real-work
   wins there; our b1dec says this kernel is issue-bound, so the overlap
   only displaces compute the ports needed.  Kept as in-class candidates
   (bit-identical, ~60 ms tuner time) in case the quiet window's more
   compute-bound balance flips it — but do not expect it.

### Operation count

Unchanged in FP: 148 ops per zmm chunk, 527 kflop/volume.  chunk17zr has
the same loads/stores as chunk17z; the round's time win is compiler
scheduling, not operation count.

### Measured on the node (tryout.sh, graded chain L=17 B=32 m=98)

- ice_r2 shipped code re-run (sched, selZ), this round's windows:
  min 14.477 (contended, sd 3.2%) and 13.991 (quiet-ish, sd 0.07%).
- Unscheduled build, selZ still (xfdza pick): **min 13.493 / median
  13.495, sd 0.02%** — the matched-pair for the pragma removal.
- **FINAL SHIPPED STATE (unscheduled, selR)**: tuner picked xfzra
  (pf=0, pw=0, pt=0; in-run table xfzra 15.15 < xfdzra 15.20 < xfdza
  15.24 < xfza 15.39, SCHED twins 15.72–15.78), graded **min 13.768 /
  median 13.929 us/step (sd 1.45%, mid-grade window)**.  rel_l2
  3.252e-16, chain-98 check 1.074e-14, bit-repeatable.  MKL 78.1 same
  core/window.
- B=1 chain (class B, unscheduled): **min 12.071 / median 12.079
  us/step (sd 0.28%)**, rel_l2 3.226e-16, chain 1.086e-14, repeatable;
  MKL 73.8.  ice_r2 shipped was 14.236 — the pragma had been taxing the
  unscored cells hardest.

### What did NOT work / negative results with numbers

- **The spill-diet mechanism itself did not materialize**: chunk17zr's
  freed registers did not reduce gcc 11.4's stack traffic (529 vs 527
  stack refs/exec in the sched build; 428 vs 429 unscheduled).  The ~1%
  class-R win is real but comes from schedule/issue effects, not spill
  deletion.  The r2 header's "~31 stack refs because the file is exactly
  full" theory is now half-dead: the refs persist with 3 registers free.
- **ov loses at this cell** (numbers above).  Transfer lesson recorded:
  overlap-with-real-work is a LATENCY-bound cure; check b1dec (in-situ vs
  L1-hot) before importing it.  L=13 imported our chain tuner and won;
  we imported their ov and it lost — both records now say why.
- The (pf,pw,pt) grid declined everything again on the final pick
  (all-off 15.14; best single flag +0.6).
- In-tuner arena contrasts OVERSTATE compile-shape effects (−13% arena vs
  −3.6% graded for the pragma).  Trust the graded matched pairs; use the
  arena only for ranking.

### Borrowed this round

- **L13_direct ice_r2**: the pragma-transfer rule that triggered the
  round's win (its +5.2% negative on a fused kernel), and the ov
  mechanism (via **L17_rader ice_r1**'s "sp").  Attribution note: L13's
  record explicitly warned "the pragma transfers only to phase-serial
  kernels" — this round is that warning applied to ourselves one round
  late.
- The within-one-binary A/B twin pattern (push_options per-function
  compile options as tuner candidates) is new this round and is the
  right shape for any future compile-flag question — no cross-run
  window poisoning.

### Score projection

ice_r2 scored 13.562 with the pragma and xfdza.  Matched graded pairs
this round: pragma removal −3.6%, xfzra over xfdza ~ −0.6% in-run.
Expect **~12.9–13.4 us/step in the quiet window**; winograd stands at
16.08, so the margin should widen to ~1.2×.

### Open for next round

- **Re-read the scored b1dec** (now measured on unscheduled probes): if
  kyz dropped in proportion (~6.3–6.5), the kernel is nearer the port
  floor and the next lever is the X pass (x/kx ~3.4–3.6, 73 chunks of
  loads from L3 in the chain).  If kyz did NOT drop, the probes were
  never pragma-sensitive and the residual story needs a counter run.
- The SCHED twins (FORCE 70–73) can be dropped next round if the quiet
  window confirms; do not add them to any class without cmp (they are
  expected to be bit-identical — scheduling moves instructions, not
  arithmetic — but r2's rule stands: cmp, never derive).
- The tail is now the only chunk17n user in the scored path (5 of 505
  zmm-chunk slots per volume are ymm tails).  A w2 merged kernel was
  believed impossible (16-register file) but the UNSCHEDULED allocator
  behaves differently — re-derive the live-set count before dismissing.
- If a future round wants ov again: pair it with something that makes
  the cell latency-bound first (e.g. if the workload moves off-L3), and
  put the overlap into volume 0's plane phase too (L17_rader's marginal
  version).

## Round ice_r4

### Where this round started

ice_r3 leaderboard: L17_matrixsimd **13.061 us/step**, first at L=17 (winograd
16.240, rader 19.368, MKL 76.3).  Quiet-window pick xfzra unscheduled, class R;
b1dec 7.42/6.98/3.46/3.21.  THEN THE TASK CHANGED: the graded step is now the
rivals' full step, `state <- (FFT(state)+c)/(1+|FFT(state)+c|)`, and the driver
detects an exported `fft3d_chain` (weak symbol) that owns the whole m-step
chain.  Without it you are timed through fft3d_execute + a driver-side
vectorized map: measured on the node, **26.224 us/step** — the unfused map
costs as much as our whole FFT.  Rivals' time to beat at L=17: 0.035 s for the
case = 11.16 us/step.  This round is entirely about the chain entry point; the
FFT kernels are untouched.

### What was changed (in order of measured effect)

1. **`fft3d_chain`: VOLUME-MAJOR, IN-PLACE.**  Each volume's chain is
   independent (FFT per volume, map pointwise), so volume b runs ALL m=98
   steps while its state (78.6 KB), its c slice (78.6 KB) and t1 (87 KB) stay
   inside the 1.25 MB L2 — corpus §10 s3's directive ("iterate each volume
   through all m steps while cache-resident") executed literally.
   `final_out` doubles as the state arena: step 0 reads x0 (const), every
   later step transforms final_out in place — legal because the X-first order
   drains the whole volume into t1 before the first Z store lands.  DRAM/L3
   traffic per volume for the whole 98-step chain: one x0 read, one c read,
   one final writeback.  A structure-only probe (map replaced by a scale,
   -DL17_MAP_STYLE=8) prices this skeleton at **11.33 us/step** — already
   under the old scored 13.06, which paid a driver-side scale pass and three
   L3-resident 2.4 MB ping-pong buffers per step.
2. **Fused per-plane map, schedule ladder.**  The map runs right after the Z
   group finishes each 4.6 KB output plane (L1-hot).  Schedules, all
   cmp-verified bit-identical on .chain outputs (the map is ONE noinline body
   — r2's call-site-contraction lesson made moot by construction):
   v0 plain 13.393, v1 split-around-Z 13.307, **v2 deferred-Z 13.262 (kept)**,
   v3 chunk-granularity interleave 13.428 (same-window, style s6).
3. **Map formulation ladder — the round's real discovery.**  All exact-tier
   unless noted, same v3 schedule, same windows:
   - s0 rsqrt14 + 2 Newton + one vdivpd per VECTOR: 15.21
   - s1 float rsqrtps seed + 3 Newton + vdivpd: 16.63 (cvt round-trip + extra
     Newton; also refutes "rsqrt14 slow" here)
   - s2 vsqrtpd + vdivpd (all divider): 23.76 — catastrophic
   - s3 vsqrtpd + rcp_ps + 3 Newton (rivals' pw_full): 18.51
   - s4 all-FMA (rsqrt14 + rcp14 ladders, no divider): 15.41
   - s5 pair-shared Newton, divide per vector: 15.39 — NO gain despite -25%
     uops, which falsified "uop-bound" and pointed at the divider
   - **s6 pair-shared + ONE vdivpd per 8 points (w = 1/d, two mul-outs) +
     t via two FMAs on deinterleaved re/im (guard 1e-300 folded into the
     fma addend): 12.998 (kept)**
   - s7 = s6 with rcp14+2 Newton instead of the divide: 13.70
   Node microbench (`build/tryout/L17_matrixsimd/mapbench.c`, leased core):
   floor(ld/add/mul/st) 1.57 ns/vec, s0 5.52, div-only 5.52, newton-only 3.09,
   seed-only 1.90, s5 5.51, s6 3.40.  **vdivpd zmm reciprocal throughput is
   ~16-18 cyc on this part** — one divide per vector IS the map floor, and
   halving divider work is worth exactly what the ladder shows.
   **vrsqrt14pd measured 0.69 ns (~2.3 cyc)**: corpus §10 s2's contested
   "~10 cyc, likely microcoded" (1760b1bf) is FALSE on bare-metal ICX — the
   14-bit seed is the right choice here, not the legacy float seed.

### Operation count

FFT unchanged (148 FP ops/zmm chunk, 527 kflop/volume).  Map per 8 points
(one pair): 4 loads, 2 adds, 2 shuffles (deinterleave), 2 FMAs (t = re^2 +
im^2 + 1e-300), 1 vrsqrt14pd, 7 Newton ops (2 iterations, exact to ~1 ulp),
1 FMA (d = t*y+1), **1 vdivpd (w = 1/d, correctly rounded)**, 2 expand
shuffles, 2 mul-outs, 2 stores = 26 uops, ~2 ulp per application.  Plus one
128-bit sqrt+div per plane for the 289th complex.

### Measured on the node (tryout, graded chain L=17 B=32 m=98, --map)

- Fallback (no fft3d_chain, ice_r3 code): min 26.224 us/step (sd 0.03%).
- SHIPPED (chain=v2, map=s6): **min 12.971 / median 12.993 us/step
  (sd 0.11%)**; same-window MKL fallback 88.85 → 6.85x on the full task.
- Correctness: single rel_l2 3.252e-16; **map-chain m=98 end state 2.12e-14
  vs numpy (budget 9.8e-12, 460x margin)** — exact tier by construction, the
  only approximation anywhere is sqrt at ~1 ulp (seed + 2 Newtons reaches
  2^-56 before the correctly-rounded divide and muls).
- Bit-repeatable: two independent processes, .chain and single outputs
  cmp-identical.
- B=1 chain: 14.64 us/step, chain check 1.15e-14, PASS (unscored, correct).

### What did NOT work / negative results with numbers

- **The rivals' lazy map LOSES at this cell (v4: 16.48 vs 13.26)**: mapping
  at the next step's X-pass loads (17 rows pre-mapped into an L1 scratch per
  chunk) puts the map's ~60-cycle dependency chain in front of every X
  chunk's critical path.  Their pipelines fuse the map into passes that have
  a following pass to hide in; our X pass IS the first pass.  If a future
  round revisits: the mechanism to fix is latency exposure, not traffic.
- Chunk-granularity interleave of map into Y/Z groups (v3): no gain over
  plain adjacency (13.43 vs 13.26-13.39) — once the divide is 1-per-8-points
  the map is not port-starved, and the plane phase's port-5 saturation (40
  tile-transpose shuffles/chunk) makes it a bad host for the map's 4
  shuffles/pair anyway.
- s5 (pair-shared Newton, per-vector divide) = s0 exactly (15.39 vs 15.29):
  25% fewer uops bought nothing because the divider was the binding unit.
  Lesson: before "reduce uops", identify WHICH unit binds — a 20-line
  microbench settled in minutes what three in-driver A/Bs could not.
- Microbench trap (corpus §10 s2 verified again): a dependent-chain
  `a = a/b` divide benchmark decays into denormals and reads 54 ns/op;
  the loop-over-buffer form reads the true ~5 ns.  Never iterate a
  contractive op in place in a timing loop.
- tryout.sh currently dies at line 36 under `set -u` (`$W` used two lines
  before it is assigned) for every chain-cased L, and its remote check.py
  call passes a literal `$W/c.bin` that expands empty on the node, so the
  map-chain check never runs in-script and the `&&` chain stops before the
  repeatability cmp.  Workaround used all round (do not edit shared
  scripts): invoke as
  `W=$PWD/build/tryout/<name> ./tryout.sh <name> 17 32 [flags]`, then run
  `python3 check.py --input .../in.bin --output .../out.bin --L 17
  --batch 32 --map-check 98 --cin .../c.bin` by hand and cmp
  repA.bin.chain/repB.bin.chain from two manual driver runs on a leased
  core.  Monitor: the fix is `W=$ICE/build/tryout/$NAME` before line 36
  and unquoting $W in the check line.

### Borrowed this round

- **Corpus §10 s2 (1000f989's mapF shape)**: burn the divider once, Newton
  on the FMA pipes — extended here to once per 8 POINTS via the pair-shared
  denominator, which their per-vector form never tried.
- **Corpus §10 s3 / PANEL_BRIEF**: volume-major cache-resident chain
  iteration; and the lazy-map idea from the same section (tried, lost,
  documented above).
- **1760b1bf's generator source** (`ext/reference/fft_v4_solutions/`): read
  for the pw_core structure and the seed-throughput warning; their
  float-seed cure was measured NOT to transfer (s1, 16.63), their fast-map
  precision gamble was not needed (our exact tier costs nothing extra).

### Score projection

Old configuration would have scored ~26 us/step on the new graded task.
Shipped: 12.97-13.0 in dev windows; expect **~12.6-13.0 in the quiet
window**.  Rivals' best at L=17 is 11.16 us/step (their code, our node) — we
are within 1.16x of them with a provably exact chain (theirs drifts 5.7e-14
at this m; ours 2.1e-14) and should stay first in-panel unless winograd's
fusion lands bigger.  Note for the panel: winograd/rader records show no
fft3d_chain yet as of this writing — the 2x fallback tax is the single
largest lever this round for every entry.

### Open for next round

- The FFT structure inside the chain prices at 11.33 us/step and the map's
  marginal cost is now ~1.7 us; the residual is the kernel's five-round
  ~1.1 FP/cyc issue wall.  Next kernel lever if anyone wants it: the plane
  phase is port-5 bound on ICX (40 vshuff64x2 per tr=1 chunk) — a Z-group
  store layout that lands map-friendly WITHOUT tile transposes (e.g. store
  untransposed + strided map) would trade port-5 shuffles for map-side
  addressing; costed but not built this round.
- A plan-time tuner across chain schedules {v0,v1,v2,v3} is legal (all four
  cmp-identical) and would adapt to the quiet window; spread is only ~1.3%,
  so it was pinned to v2 instead.  Revisit only if the leaderboard shows a
  window-dependent flip.
- If the monitor ever scores B=1 chains: v2 at B=1 is 14.64; the class-B
  X-last kernels are untouched and still own batch<17 fft3d_execute.
