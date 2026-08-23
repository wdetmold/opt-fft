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

## Round ice_r5

### Where this round started

ice_r4 leaderboard: L17_matrixsimd **13.009 us/step**, first at L=17 (winograd
14.854, rader 17.521, best library ducc0 86.4 → 6.64x).  The r4 VERDICT's
directive for this geometry: "attack the 17-point kernel, not the chain" —
the FFT skeleton prices at 11.33 us/step with the map fully hidden, the
rivals' mark is 11.16, so the whole 1.17x deficit is the kernel's six-round
~1.1-FP/cyc issue wall.  b1dec on the scored string: 7.44/7.07/3.48/3.21.
This round: one win (sign-folded sine constants, −1.3%) and three cleanly
killed negatives (register-fused map, Z-only extract-stores, in-chain
prefetch), all matched-pair measured.

### What was changed

1. **SIGN-FOLDED SINE CONSTANTS — the round's win, bit-identical by IEEE
   sign algebra.**  Every sine multiply in every kernel is
   `S +-= K * MULI(a-b)` with MULI = swap-re/im + vpxorq of the odd lanes.
   On ICX the 8 vpxorq per chunk issue to ports 0/5 — exactly the two FMA
   pipes the kernel is bound on.  Folded the odd-lane sign into the sine
   TABLES/PINS instead: stab8/4 and nts8/4 now hold alternating-sign splats
   (+k,-k,+k,-k,...), MULI is just the swap.  (-k)*x is bit-identical to
   k*(-x) (sign is XOR'd in IEEE multiply, FMAs included), so every bit
   class survives intact — cmp on full chain outputs on the node confirmed
   BIT-IDENTICAL before anything was timed.  Deletes 8 uops/chunk from the
   binding ports across all 243 chunk slots (~1.9k p0/p5 uops/volume).
   Matched A/B, alternating binaries 3x each on one leased core:
   folded 12.777/12.808/12.820 vs XOR 12.982/12.979/12.924 us/step —
   **wins every pair, −1.3%**.  `-DL17_NO_SFOLD` restores the old form
   (the whole fold is a table change + one macro, so the hook is cheap).
   First fold-vs-window trap for the record: my first single measurement
   said the fold LOST (14.615 vs 13.102) — that was the documented bimodal
   window (MKL steady, us swinging); only the within-lease alternation is
   evidence.

2. **chunk17zrm + chain v5 — register-fused map, a designed-and-killed
   NEGATIVE (+1.7 us/step).**  Executed the r4 "open" idea: the s6 map
   applied to the Z chunk's 17 output vectors IN REGISTERS before the tile
   transpose (map commutes with the transpose pointwise), reading a
   ky<->kz-transposed copy of c built once per volume-chain (p->ct, 78.6 KB,
   amortized over m=98 to ~0.03 us/step).  Pairs chosen SYMMETRIC
   ((E1,E16)(E2,E15)...(E8,E9) + E0 solo) so each pair kills one (C_n,S_n)
   accumulator pair at map time — consecutive pairing would keep all 17
   live.  ymm tail runs FIRST (rows 15,16 unmapped), the f0=12 chunk
   re-maps row 15, rows ky=16 finish through the same noinline helpers
   (map pairs 34,35 + tail — exactly row 16, the pair grid aligns).
   Correct (chain 2.197e-14) — and **LOSES: 14.468/14.479 vs 12.820/12.742
   matched**.  Spills are NOT the cause (450 vs 423 stack refs in
   fft3d_chain).  Post-mortem: the ladder (~70 cycles: adds → shuffles →
   rsqrt14 → 2 Newtons → fma → vdivpd(~23c) → shuffle → mul) lands on
   every chunk's STORE path, and the fused chunk is a ~400-uop monolith
   the ROB (~352) cannot span — cross-chunk OoO overlap dies.  The
   noinline map loop it replaced pipelines ~12 independent 28-uop
   iterations in the same window.  **Together with r4's v3 (chunk-granular
   interleave, no gain) and v4 (lazy map, +3.2), the map-placement question
   is now CLOSED: the plane-level noinline map after Z (v2) is optimal;
   the 1.66 us map marginal is irreducible issue cost, not schedulable
   slack.**  Code kept, forceable via -DL17_CHAIN_VAR=5.

3. **Z-group-only extract-stores in the chain (-DL17_CHAIN_ZXST=1) — the
   r1 postmortem's untested arm, NEGATIVE (+0.55).**  r1 predicted Z-only
   dodges the Y->Z store-forwarding tax and priced ~0.3 us of port-5
   relief.  Built: 13.277/13.353/13.293 vs 12.778/12.724/12.769 matched,
   bit-identical outputs.  The forwarding WAS clean (map reads land
   hundreds of cycles after the stores commit); the loss is the +48 store
   uops per chunk colliding with the map pass's own load/store traffic on
   the same plane.  Extract-store is now 0-for-3 at this cell (xfax r1,
   merged+xst r2, Z-only r5); do not retry in any form.

4. **In-chain plane prefetch (-DL17_CHAIN_PF=1/2/3) — NEGATIVE.**  The
   exec tuner's pf/pw/pt knobs never existed inside fft3d_chain, and b1dec
   shows ~0.6 us/volume of L2-fill gap (yz 7.44 vs kyz 7.07, x 3.48 vs kx
   3.21), so this was the one untested latency lever in the volume-major
   regime.  pf1 (t0 of the next Y source plane) +0.6; pf2 (prefetchw of
   the next out plane) +0.5; pf3 (both) +1.3 us.  The 73-uop bursts
   displace issue slots the kernel needed: the cell is issue-bound, not
   latency-bound, now proven inside the chain regime too (fifth
   consecutive prefetch decline at L=17).

### Operation count

FFT: 148 FP vector ops per zmm chunk unchanged, but the 8 vpxorq/chunk are
deleted — per volume ~243 chunk slots x 8 = ~1.9k uops off ports 0/5.  Map
unchanged (s6: 26 uops per 8 points, one vdivpd per 8 points).  527
kflop/volume unchanged.

### Measured on the node (tryout, graded chain L=17 B=32 m=98, --map)

- r4 shipped code re-run this round's windows: 12.92-13.03 quiet-ish,
  13.10 (session start), 14.9 contended (bimodal, MKL steady 88.8 — the
  r4-documented co-tenant L3 swing).
- **FINAL SHIPPED STATE (sign-fold + chain v2, all A/B hooks off): min
  12.768 / median 12.769 us/step (sd 0.01%, MKL 88.8 same window)**.
  Single rel_l2 3.252e-16; map-chain m=98 end state **2.123e-14** vs numpy
  (budget 9.8e-12) — bit-identical to the r4 chain, as the fold's algebra
  demands.  Bit-repeatable across two processes (cmp on .chain outputs).
- B=1 chain: 14.403 us/step (sd 0.07%), chain check 1.152e-14, PASS;
  MKL 99.6.  (r4: 14.64 — the fold helps the unscored cells too.)
- tryout.sh line-36 $W bug still present; the whole round used the r4
  workaround (env W=... prefix) + manual check.py + manual repeatability.

### What did NOT work / negative results with numbers

All three negatives above, each with matched-pair numbers: v5 register-fused
map (+1.7 us, latency-on-store-path + ROB-defeat, NOT spills), Z-only
extract-store (+0.55, store-uop collision with the map pass), in-chain
prefetch (+0.5 to +1.3, issue displacement).  Transfer lessons: (a) fusing a
long-latency ladder INTO a store path of an issue-bound kernel loses even
when it deletes memory traffic — keep independent work in its own small
loop the OoO can pipeline; (b) the "map is free" finding of r4 holds only
for the noinline-loop placement, not for register fusion.

### Borrowed this round

- The r4 VERDICT's L=17 directive (attack the kernel, the map is optimal)
  framed the round; its map-marginal table justified killing v5 quickly.
- The within-one-lease alternating-binary A/B protocol is this entry's r3
  pattern, now upgraded to alternate FULL BINARIES via the driver's --json
  (needed because the bimodal window poisons single runs — see item 1).

### Score projection

r4 scored 13.009 from dev 12.97-13.0; this round's dev floor is 12.74-12.82
under the same conditions, so expect **~12.6-12.8 us/step in the quiet
window**.  Winograd stood 14.854 with no fft3d_chain in its r4 record — if
its agent ships fusion this round it lands ~mid-13s at best from its 16.2
FFT, so first place should hold.  Rivals' mark 11.16 (gap now ~1.13-1.15x,
was 1.17x, with a provably exact chain: 2.1e-14 vs their 5.7e-14 at this m).

### Open for next round

- The kernel wall stands at ~1.1 FP-instr/cyc after SEVEN rounds and every
  cheap mechanism class is now measured (scheduling, addresses, spills,
  width, prefetch, store form, map placement, uop deletion).  The only
  untried lever left with real headroom is ALGORITHMIC: the verdict's
  §4.2(b) symmetric/antisymmetric convolution split, or a Winograd-style
  negacyclic-8 that trades the sine block's 64 FMAs for fewer mults + more
  adds — but ROOFLINE.md's add-port warning says count TOTAL vector uops
  before building anything (the dense sine block is 64 uops; a Winograd
  split must beat that including its pre/post adds, which L17_winograd's
  record says no split of x^8+1 does).  Read their record first.
- If a future task moves the working set off-L2, re-run pf/pw/pt in the
  chain (they are one -D flag away) — the declines here are issue-bound
  verdicts, not universal ones.
- The X pass measures ~145 cyc/chunk against an ~80-cyc port floor with
  L1-hot sources (kx probe) — the residual is dependency/issue shape, not
  memory.  If anyone wants it: the 17 strided stores (t1, 87 KB, misses L1)
  are in BOTH probe arms, so a store-side experiment (t1 in a 4-plane
  rotating L1 window is impossible — Y needs whole planes; but a 2-plane
  X/Y overlap does exist in the exec ov twins and LOST — see r3) has no
  obvious remaining form.  Treat X as closed unless a counter run says
  otherwise.

## Round ice_r6

### Where this round started

ice_r5 leaderboard: L17_matrixsimd **12.736 us/step**, first at L=17 (winograd
14.412, rader 15.052, best library ducc0 86.2 -> 6.77x, the largest library
margin on the board).  The r5 VERDICT's L=17 directive: "attack the 17-point
module's operation count; scheduling is finished", naming LITERATURE §4.2(b)'s
symmetric/antisymmetric convolution split.  This round EXAMINED that directive
and rejected it with arithmetic before building anything: our kernel already
IS the sym/antisym split (the conjugate-pair fold's u/v halves feeding real
cosine/sine blocks -- §01 §8's construction applies it to Rader's length-16
convolution, which we don't use), L17_winograd's record has already counted
every split of x^8+1 as a loser, and decisively: **the rivals' 11.16 us/step
existence proof (1760b1bf, read this round in ext/reference/) uses MORE FP
per line than we do** -- a dense 8x8 cosine block (64 FMAs, no
cyclic/negacyclic split), phase-split with stack spill arrays, ~168 FP
ops/chunk vs our 148.  Their speed is not arithmetic; it is STRUCTURE.  From
their generator: rows padded 17->20 complex so every strided access is one
aligned cache line; transforms run ALONG strided directions IN PLACE (zero
shuffles in 2 of 3 passes, no t1-style buffer at all); slab stride forced to
an odd line count for L1 set spread; the state lives in the padded arena for
the whole chain (unpack once, pack once).  Their issue rate ~2.3 uops/cyc vs
our engine's ~1.6 on the same silicon.  This round rebuilt our chain engine
in that shape while keeping our cheaper kernel arithmetic.

### What was changed (chain v6, now the AVX-512 default; the exec/B=1 ABI
### paths and the kernels' arithmetic are untouched)

1. **chunk17zri: in-place twin of chunk17zr.**  Same 17 loads in the same
   pair order, same accumulator update order, same slot table, but ONE
   non-restrict data pointer (chunk17zr with src==dst is UB by its restrict
   qualifiers).  All loads precede all stores in program order and the
   pointers may alias, so in-place is well-defined.  Plain-store path only.

2. **Padded per-volume arenas** (p->pa state, p->ca c field): rows 17->20
   complex (320 B), slab stride 808 doubles = 6464 B = 101 lines (odd -- the
   rivals' set-spread trick), 110 KB each, both L2-resident together.
   Padding lanes are zeros: zero in -> zero out through every linear pass,
   map(0+0) == 0, 1e-300 guard keeps rsqrt14 finite -- pads never
   contaminate, never denormalize.  Unpack x0+c per volume-chain, pack the
   (already mapped) end state once -- both amortized by m=98.

3. **Step = pass b + slab-resident phase.**
   - b (axis j0): 85 in-place chunks, elements at the 101-line stride.
   - per slab (6.5 KB): a1 (axis j1, 5 in-place chunks at stride 40); a2
     (axis j2): four 4-row groups, each 5 TRANSP4-tiles in -> the SAME kernel
     on a 20-vector stack io[] -> 5 tiles out (64B stores forward cleanly to
     64B loads -- not r1's 16B->64B trap); the s6 map interleaved at GROUP
     granularity (item 5).
   - row-16 fringe (item 6) and a1 pipelining (item 7) on top.
   Axis order is unchanged from the shipped X-first engine (j0,j1,j2), and
   per-lane arithmetic is identical -- v6's .chain outputs were expected in
   the same rounding class and the harness gates confirm correctness
   directly (single 3.252e-16, chain 2.055e-14).

4. **Profile that drove the rest** (-DL17_V6_PROF=1 rdtsc split, TSC kcyc
   @2.9 GHz ref, first working v6 with a flat per-slab map sweep):
   b=6.08 a1=6.69 a2=11.56 map=11.72.  Two findings: **the strided in-place
   passes crushed the old engine** -- pass b runs ~81 core-cyc/chunk vs the
   old X pass's ~145 with the same kernel (alignment + in-place + set
   spread: the six-round "issue wall" was never the kernel alone, it was the
   pass shape) -- and **the flat map attribution exploded to ~4 us** vs the
   old engine's 1.7 marginal.  The FFT won ~2.1 us; the map placement gave
   half back.  (Matched mins that round-trip: flat v6 12.61-12.83 vs v2
   12.74-12.85 -- barely ahead.)

5. **Map interleaved at a2-group granularity** (after group g stores rows
   4g..4g+3, map exactly those rows: same noinline body, same values, same
   order -- bit-identical, call sites only).  The ladders and the c stream
   issue under the next group's port-5-heavy transposes.  Matched 3/3:
   12.516/12.546/12.527 vs flat 12.645/12.863/12.657 (-0.12 us).

6. **Cross-slab row-16 fringe** (-DL17_V6_XS=0 restores per-slab): the 17
   row-16 lines are independent lines at the common slab stride, so 4 full
   groups + 1 overlap group (store slab 16's lane only) replace 17 per-slab
   recompute groups -- 73 a2 groups/volume instead of 85.  Co-lane rawness
   is irrelevant: every vector op is lanewise and TRANSP4 only moves complex
   units.  Matched 3/3: 12.186/12.098/12.118 vs 12.516/12.567/12.484
   (**-0.39 us**), .chain bit-identical.

7. **a1(s+1) software-pipelined into a2(s)'s groups** (-DL17_V6_P1=0
   restores): a1 is load/FMA work, a2 is shuffle-heavy -- port mix, plus
   independent uops inside a2's >ROB-sized group+map bodies.  Matched 3/3:
   11.928/11.948/11.927 vs 12.094/12.053/12.287 (**-0.15 to -0.36 us**),
   bit-identical.

### Operation count

Per volume-step: 243 zmm kernel chunks (85 b + 85 a1 + 73 a2) x 148 FP =
36.0k vector FP ops -- numerically the same slot count as the old engine's
243, but ALL-zmm covering 972 lane-transforms (867 needed; +12% pad/overlap
waste is the price of padded alignment, the rivals pay the same) where the
old engine's 209 zmm + 34 ymm covered 904.  Shuffles: 73x80 = 5.8k TRANSP4
uops + 8 MULI/chunk (was 6.8k tile + MULI).  DELETED: the 87 KB t1
round-trip per step, both plane buffers, every unaligned/line-split access,
the addr-safe shift machinery (v6 needs none), and the separate map sweep.
Map unchanged: s6 exact tier (rsqrt14+2N, one vdivpd per 8 points), now 714
pair-iters + 17 single-vector calls per volume (+17% points from row pads,
more than paid for by placement).

### Measured on the node (tryout + same-lease alternating-binary A/Bs,
### graded chain L=17 B=32 m=98; every A/B above is 3 matched pairs)

- **FINAL SHIPPED STATE (chain=v6 xs1 p1, defaults, no -D flags): B=32
  min 11.905 / median 11.966 us/step**; same-window MKL 88.83 -> 7.46x.
  Same-day matched v2 baseline: 12.74-12.85 (r5 scored 12.736), so the
  round is worth ~-0.85 us (-6.6%) in matched conditions.
- Correctness: single rel_l2 3.252e-16; **map-chain m=98 end state 2.055e-14
  vs numpy (budget 9.8e-12, 477x margin)** -- exact tier, only sqrt at
  ~1 ulp + correctly rounded divide.  Bit-repeatable across two processes
  (.bin and .bin.chain both cmp-identical).
- B=1 chain: **13.610 us/step (sd 0.04%)**, chain check 1.163e-14, PASS;
  MKL 99.7.  (r5: 14.403 -- v6 helps the unscored cell too.)
- tryout.sh's line-36 $W bug still present; same env-prefix workaround +
  manual check.py + manual repeatability as r4/r5.

### What did NOT work / negatives with numbers

- **Flat per-slab map sweep in v6: the map attributed ~4.0 us/step** (profile
  item 4) vs 1.7 in the old engine -- a placement that follows ALL the a2
  groups instead of interleaving them leaves the ladders un-overlapped.
  Fixed by item 5; recorded because "the map is free after the FFT" is
  engine-shape-dependent, not a law.
- **Split-complex state layout: costed and rejected on paper, do not build.**
  For OUR lanes-are-lines kernel the 8x8 real transposes cost 0.375
  shuffle/double vs 4x4 complex tiles' 0.29, MULI is already just a swap
  (sign-folded), and FP count is unchanged -- the arithmetic says it loses
  before any of L17_winograd's msp8 deinterleave savings apply to us (their
  kernel is component-split by construction; ours is not).
- rdtsc profiling itself costs ~1-2.5 us/step (52 probe pairs/step) -- the
  profiled binary's absolute numbers are for SPLITS only.

### Borrowed this round (named, per the round's brief)

- **Rival pipeline 1760b1bf** (ext/reference/fft_v4_solutions/
  1760b1bf_score0.96/generator.py): the whole v6 pass structure -- padded
  aligned rows, in-place transforms along strided axes, odd-line-count slab
  stride, arena custody across the chain, the a2 tile-io shape, and the
  "leftover row as its own group" idea (their fft17_row2, upgraded here to
  cross-slab zmm groups instead of their 128-bit scalar codelet).
- **L17_winograd ice_r5**: the format-ownership principle ("the harness sees
  only x0 and final_out, so the interleaved layout need not exist between
  steps") that legitimizes the padded arena custody.
- **L64_blocked ice_r5 / r5 VERDICT §5**: the custody-layout precedent and
  the "restructure for residency, never fuse within it" synthesis -- v6 is
  that rule applied at L1/line granularity (residency + alignment, with the
  map kept as its own small loops).

### Score projection

Dev floor 11.90-11.97 under mid-grade load, matched-window advantage over
the r5 code -0.85 us; expect **~11.6-11.9 us/step in the quiet window**
(r5 scored 12.736 from dev 12.77).  Rivals' full-task mark is 11.16 on this
node: the gap closes from 1.14x to ~1.05x, with a provably exact chain
(2.06e-14 vs their 5.7e-14 at this m).  Winograd's r5 record projects
13.5-14.2 for its round; first place should hold.

### Open for next round

- **Fringe/b overlap**: pass b of step s+1 needs only rows j1<=15 mapped
  for 80 of its 85 chunks; the current step's cross-slab fringe (5 groups +
  17 row-16 maps) could interleave with them.  Est ~0.2-0.3 us, moderate
  restructure.
- **Lazy map into pass b's loads** (the rivals' run_A shape): now that the
  first pass is short-strided and near floor, the r4 v4 failure may not
  transfer -- but map-in-a2 already overlaps well, so expected gain is
  small; try only if the fringe/b overlap is built anyway.
- The b/a1 padding waste (3 of 20 lanes) is structural; every alternative
  costed worse (ymm 5th chunk: no port-time gain on 2x512 pipes; gather
  regrouping: transpose cost exceeds the waste).  Treat as closed.
- If the quiet window flips any hook: -DL17_V6_XS=0, -DL17_V6_P1=0,
  -DL17_CHAIN_VAR=2 are one flag each, all cmp-verified classes.
- The in-plan b1dec probes still measure the OLD exec engine (scored string
  carries them): they no longer describe the scored chain path.  If a future
  round wants in-chain diagnostics, port the probe to v6's passes
  (-DL17_V6_PROF is the dev-only version of that).

## Round ice_r7

### Where this round started

ice_r6 leaderboard: L17_matrixsimd **11.935 us/step**, SECOND at L=17 for the
first time since ice_r1 -- L17_winograd shipped its rotating-lane fused chain
and took the cell at 11.649 (rader 12.284, ducc0 86.3).  The round's brief
opened the rival campaigns' sources and the honest per-attempt re-benchmarks
on THIS node (`results/rivals_icelake/`): best rival at L=17 is **1760b1bf at
0.0335 s = 10.68 us/step** (not the 11.16 planning number), so the real gap
was 1.12x to the rivals and 1.02x to winograd.

### What was changed: chain v7 -- VOLUME-SoA LANES (4 volumes per zmm)

One structural idea, taken from the rival corpus and then made cheaper than
their own form.  The v5/v6 campaigns' decisive layout at sizes 6-23
(v6_f40c5e25's gen.py, `fft_v5v6_solutions/`) puts VOLUMES in the SIMD lanes
("SoA 8-volumes-per-zmm"), so a transform along ANY axis is a plain strided
in-place sweep -- zero shuffles in the transform, by construction.  Their
form is 8-volume SPLIT-complex, which needs ~336 vector ops per 8 pencils
(each real scalar op becomes one vector op) -- and notably their SoA
attempts measure SLOWER at L=17 on our node (0.038-0.041 s) than 1760b1bf's
within-volume padded engine (0.0335).  Chain v7 keeps INTERLEAVED complex
with 4 volumes per zmm instead: our merged-reordered chunk17zri (148 FP ops
per 4 pencils = 296 per 8, +8 sign-folded MULI swaps) runs VERBATIM -- the
lanes just mean volumes -- so we get the rivals' zero-shuffle structure at
our lower op count, in half their L2 footprint.

Mechanics (see the chain v7 block in the source):
- Arena: point q = j0*289+j1*17+j2 holds the 4 volumes' complex values in
  one zmm at pa7[8q]; 4913 x 64 B = 314 KB state + 314 KB c (ca7), both
  L2-resident for the whole m=98 group-chain.  Every access in every pass
  is one full aligned 64-B line.  pa7/ca7 base congruence mod 4K held 3200 B
  apart for L1 set spread on the paired map streams.
- Step = pass j0 (289 in-place chunks, row stride 2312 doubles = 289 lines,
  mod-4K = 33 lines, odd -> set spread; consecutive chunks walk adjacent
  columns so the 17 row streams read sequentially out of L2), then a
  slab-resident phase per 18.5 KB slab: pass j1 (17 chunks, stride 136 = 17
  lines, odd), pass j2 (17 chunks, stride 8 = contiguous) with the s6 map
  interleaved at 4-pencil granularity (v6's proven placement, same noinline
  map bodies -- l17_map_vecs pairs + l17_map_vec1 for the odd vector).
- Unpack x0+c into SoA / pack the mapped end state back: 16-B lane copies
  once per group-chain, amortized by m (arena-custody legitimacy:
  L17_winograd ice_r5's format-ownership doctrine, as in v6).
- batch%4 remainder volumes run the untouched v6 engine (l17_chain_v6 grew
  an nbv volume-count parameter); B=1 therefore stays on v6.
  -DL17_CHAIN_VAR=6 restores v6 wholesale for matched A/Bs.

What this deletes per volume-step vs v6: ALL tile transposes (73 A2G groups
x 80 vpermt2pd = 5.8k port-5 uops plus 40-vector stack round trips), ALL
padding work (v6 ran 972 lane-transforms / 243 chunks for 867 needed; v7
runs exactly 867/4 = 216.75 chunk-equivalents), the +17% map points on row
pads (v7 maps exactly 4913 points), and the cross-slab fringe.  Rough total
vector-uop count per volume-step: ~75k -> ~58k (-23%).

### Operation count

FFT arithmetic unchanged: 148 FP vector ops per chunk, but now 867 chunks
per 4-VOLUME group per step (216.75/volume, was 243) = 32.1k vector FP
ops/volume-step (was 36.0k), zero transpose uops (was ~5.8k), zero pad
lanes.  Map unchanged in form (s6 exact tier: rsqrt14 + 2 Newton, ONE
vdivpd per 8 points): 2456.5 pair-iters + 17 single-vector calls per group
(614 pair-iters/volume, was 714 pairs + 17 singles on padded points).

### Measured on the node (manual tryout protocol, graded chain L=17 B=32
### m=98 --map; the r4 $W workaround is still needed AND reserve.sh --status
### cannot see slurm from wallaby this round -- replicated tryout.sh's exact
### ssh steps by hand on a leased core instead)

- Matched same-lease alternating-binary pairs, 3/3, --samples 4:
  **v7 9.040 / 9.045 / 9.042 vs v6 11.953 / 11.950 / 12.012 us/step
  (-24.4%)** -- and the .chain outputs are BIT-IDENTICAL between v6 and v7
  on the node's gcc 11.4 (same kernel + map bodies, same per-point order;
  verified by cmp, not assumed -- r2's rule).
- **FINAL SHIPPED STATE: B=32 min 9.053 / median 9.056 us/step (sd 0.75%)**;
  MKL same case/core 89.0 -> 9.8x on the full task.  Single rel_l2
  3.252e-16; map-chain m=98 end state **2.055e-14 vs numpy (budget 9.8e-12,
  477x margin)** -- numerically the same end state as r6, as bit-identity
  demands.  Bit-repeatable across two processes (.bin and .bin.chain).
- B=1 chain (v6 remainder path): **min 11.876 / median 11.878 us/step
  (sd 0.02%)**, rel_l2 3.226e-16, chain 1.163e-14, PASS; the r6-shipped
  code measured 13.610 at this cell -- the gain is window/lease conditions,
  not code (the B=1 path is untouched v6).
- Correctness also verified locally (wallaby has AVX-512) before the node
  ever saw the code: B=1/5/6/8/32, m=3 and m=98, all PASS; B=5/6 exercise
  the group+remainder seam.

### What did NOT work / negatives with numbers

- **v7-P1 (slab i+1's j1 chunks software-pipelined into slab i's j2+map
  phase, v6's winning P1 mechanism transplanted): LOSES 3/3 matched pairs,
  9.409/9.241/9.225 vs 9.045/9.045/9.040.**  v6's P1 won on PORT MIX (a1
  is FMA-heavy, a2 shuffle-heavy); v7 has no shuffle-heavy phase left, so
  the interleave only displaces issue slots from an already FP-saturated
  stream.  Same lesson class as r3's ov loss: overlap tricks need a
  complementary resource to fill.  Kept as -DL17_V7_P1=1 (bit-identical,
  cmp-verified on node) for machine insurance; default 0.
- Note for the panel: the 8-volume SPLIT-complex SoA form (the rivals'
  literal layout) was REJECTED on arithmetic before building: ~336 vector
  ops per 8 pencils vs our 296+8, double the L2 footprint (1.26 MB for
  state+c vs 628 KB), and the rivals' own SoA attempts are slower than
  1760b1bf at L=17 on our node.  Interleaved-complex 4-volume SoA is the
  form that composes with a conjugate-fold interleaved kernel.

### Borrowed this round (named, per the round's brief)

- **v6_f40c5e25** (fft_v5v6_solutions/v6_f40c5e25_score0.91/
  dev_generators/gen.py): the volumes-in-lanes SoA layout for prime sizes
  -- the whole idea of chain v7 -- adapted from their 8-volume
  split-complex to 4-volume interleaved complex.
- **results/rivals_icelake/rivals.json**: the honest target (1760b1bf
  10.68 us/step on this node, not the 11.16 planning number).
- v6's slab phase shape, map placement, and arena custody carry over from
  our own r6 (itself from 1760b1bf and L17_winograd r5).

### Score projection

r6 scored 11.935 from dev 11.90-11.97; this round's dev floor is
9.040-9.056 under the same protocol, so expect **~9.0-9.3 us/step in the
quiet window** = ~0.0283 s for the graded point.  That would retake L=17
from winograd (11.649) and pass the best rival on this node (1760b1bf,
0.0335 s = 10.68) by ~15%, chain-true at 2.06e-14 vs their 5.95e-14 at
this m.  Whole-panel note: L=17 would move from the last remaining rival-
losing cell (1.05x) to ~0.85x.

### Open for next round

- The kernel wall is now the whole story: 9.05 us/step = ~29.9 kcyc/volume
  against a ~21-22 kcyc all-port floor (867 chunks x ~74 cyc / 4 volumes +
  map FMA work).  The divider is NOT binding (2456 vdivpd x ~16 cyc = 39
  kcyc/group vs 119 kcyc measured -- comfortably hidden).  If anyone wants
  the residual: pass j0 is the only phase reading from L2 at distance;
  a j0-block-transposed custody (keep the state j0-major half the time) is
  the shape that would L1-localize it, but it reintroduces transposes --
  price it against the 5.8k-uop lesson before building.
- The v7 arena supports any group count; if a future task scores B=1, a
  1-volume SoA is meaningless -- B=1 stays v6 (11.88), and the only B=1
  lever left is the v6 open list from r6.
- tryout.sh $W bug STILL live (r4 note has the fix); additionally this
  round reserve.sh --status could not see slurm from wallaby (sbatch/squeue
  not in PATH even after env.sh), so the whole session ran the manual ssh
  protocol on a slot lease.  Monitor: either fix the PATH in env.sh or
  make tryout.sh trust an existing RESERVATION whose node still answers
  ssh.

## Round ice_r8

### Where this round started

ice_r7 leaderboard: L17_matrixsimd **9.035 us/step**, FIRST at L=17 again
(winograd 10.922, rader 11.919, ducc0 86.2 -> 9.54x) -- chain v7 (volume-SoA
4/zmm) retook the cell and passed the best measured rival (1760b1bf, 10.68 on
this node).  The r8 brief: (1) the gate is now TWO-PART (one-step 1e-14 +
chain at 300x the honest library divergence -- docs/GRADER.md); (2) mine the
warm cohort (`fft_warm_solutions/`, best 0.99 = r~0.145).  Read the 0.99's
README + generator config: its prime engine's levers (volumes-in-SIMD-lanes
SoA, fused per-plane sweeps, consumption-ordered c, lazy map into the next
step's first pass, batch tail-splitting) are ALL either already in chain v7
(r7 took the layout from v6_f40c5e25) or measured losers at this cell (lazy
map: r4 v4 +3.2; its 8-vol split-complex form: rejected on op count in r7).
Nothing structurally new to take, so this round went where the r7 record
pointed: the v7 step's own residual, measured instead of guessed.

### Gate work first (rejection insurance, found by running the new gate shape)

- **`driver --map --chain 1` SEGFAULTS every chain-exporting entry**:
  driver.c allocates `pong` only when chain > 1, then passes it (NULL) as
  fft3d_chain's destination; at chain==1 it also never writes the .chain
  file.  So the one-step gate can only be run as `--chain 2` (which is why
  check.py's one-step branch accepts m <= 2, tol 1.5e-14*m).  Our entry now
  NULL-guards final_out (writes to internal scratch instead of crashing) --
  one branch per chain call, zero cost on real runs.
- **check.py is missing `import math`** since the two-part-gate edit: every
  map-check with m > 2 dies with NameError AFTER printing the single-transform
  PASS (the map-chain verdict, repeatability cmp and MKL reference never
  run).  Worked around with a fixed private copy
  (build/tryout/L17_matrixsimd/check_fixed.py).  MONITOR: one-word fix at
  check.py line 6.  (tryout.sh's r4 `$W` bug is ALSO still live; same
  env-prefix workaround as r4-r7, though this round replicated the ssh steps
  manually anyway because reserve.sh --status still cannot see slurm from
  wallaby.)
- Gate posture of the shipped entry, measured on the node: one-step-tier
  (m=2) rel_l2 **1.212e-15** vs tol 3.0e-14 (25x margin); chain m=98
  **2.118e-14** vs tol 1.0e-10 (anchor 3.0e-14).  The exact-tier map keeps
  paying for itself under the corrected gate.

### The profile that drove the round (-DL17_V7_PROF, new rdtsc split)

v7 step phases, tsc kcyc/group-step (x1.14 for core cyc at 3.3 GHz):
**j0 = 20.97, j1 = 21.64, j2+map = 60.96** (unpack/pack amortized ~1% of a
step).  j0 and j1 sit AT the 82-cyc/chunk port floor (156 p05 uops / 2
pipes) -- six rounds of kernel work have left nothing there.  The whole
residual is the map: ~39 tsc kcyc/group-step = 16-18 cyc/pair, ~2x the
18-p05-uop port floor, and (decisive, from matched A/Bs + mapbench):
**the map runs at its STANDALONE rate in-chain** -- a 17-pair burst is ~510
uops > ROB(352), so the OoO cannot interpenetrate map bursts with kernel
chunks at all; "the map hides under the FFT" was never true in v7, it just
ran fast standalone.  The map body is ILP/latency-bound (one ~16-stage
dependency chain per pair), not divider-bound, not c-stream-bound.

### What was changed (all matched same-lease alternating-binary pairs, B=32 m=98)

1. **Map interleave granularity 4 -> 2 chunks (L17_V7_MG=2)**: 9.020/9.030/
   9.030 vs 9.052/9.062/9.090 (3/3, -0.35%).  Bit-identical (same body,
   same ascending pair order; cmp-verified, and MG=4 twin cmp'd == the r7
   baseline binary).  MG=1 LOSES (+0.15: 9.17 vs 9.02 3/3) -- map loads land
   right behind the just-issued chunk stores.  Re-tested MG=4 after the x2
   change (34 pairs = exactly 17 x2-iterations, no odd tail): still loses
   (8.90/8.91 vs 8.84/8.86).
2. **Quartic (4th-order Householder) rsqrt refinement instead of two Newton
   steps (L17_MAP_Q4=1, default)**: y += y*e*(1/2 + e*(3/8 + e*5/16)),
   e = 1 - t*y^2.  One op fewer AND a 2-stage-shorter chain; residual
   (35/128)e^4 ~ 4e-17 from the 2^-14 seed, sqrt lands ~2-3 ulp -- exact
   tier.  **8.899/8.893/8.888 vs 9.013/9.014/9.015 (3/3, -1.3%)** -- more
   than the op count predicts, consistent with the body being
   latency-bound.  New rounding class by construction; all gates re-run
   PASS (m=2 1.212e-15, chain 2.118e-14, bit-repeatable).
3. **Two map pairs per iteration, hand-interleaved at source
   (L17_MAP_X2=1, default)**: gcc does no pre-RA scheduling on x86 (the r1
   pragma saga's root cause, striking again), so the one-pair loop reached
   the backend as one serial chain; the x2 body interleaves two.  mapbench
   isolated: 2.68 -> 2.49 ns/vec; in-chain **8.845-8.859 vs 8.888-9.213,
   4/4 pairs (~-0.5%)**.  BIT-IDENTICAL (every point sees the same scalar
   ops in the same order regardless of which points share a pack vector;
   cmp-verified).  x3/x4 tested in mapbench: 2.79/2.78 vs x2's 2.83
   ns/vec -- not worth the code for ~0.02 us/step; not wired.
4. NULL-guard in fft3d_chain (gate work above) + the v7 profile hook
   (-DL17_V7_PROF) + description string now carries map=s6q4x2 mg2.

### Operation count

FFT unchanged (148 FP vector ops/chunk, 867 chunks/group-step).  Map per 8
points: was 26 uops with a 7-op rsqrt refinement; now 25 uops with a 6-op
quartic refinement, still ONE vdivpd per 8 points (correctly rounded divide
and output muls; the only approximate op is sqrt at ~2-3 ulp).  614
pair-iters + 17 singles per volume-step unchanged.

### Measured on the node (manual r7 ssh protocol, core 2 lease)

- **FINAL SHIPPED STATE: B=32 min 8.831 / median 8.834 us/step (sd 0.02%)**;
  same-window MKL 88.88 -> **10.06x on the full graded task**.  Same-day
  matched r7-equivalent binaries: 9.05-9.09, so the round is worth ~-2.3%
  matched.  Single rel_l2 3.252e-16; chain m=98 2.118e-14 (tol 1e-10);
  one-step-tier m=2 1.212e-15 (tol 3e-14); bit-repeatable across two
  processes (.bin and .bin.chain).
- B=1 (v6 remainder path, inherits the map changes): **13.359 us/step
  (sd 0.03%)**, chain 1.239e-14, PASS, repeatable.  (r6 measured 13.610,
  r7's 11.876 was a quiet-window fluke per its own note.)
- B=6 (group+remainder seam): 9.837, PASS, repeatable.
- Window health: several A/B first-runs-after-build read 1.2-1.4 us high
  (page-cache/ramp artifact); every conclusion above discards those and
  uses within-lease alternation, per the r5 protocol.

### What did NOT work / negatives with numbers

- **s7 for the reciprocal (rcp14 + 2 Newton, no divider): +0.4 us**
  (9.470/9.470/9.471 vs 9.065-9.113, 3/3) even though **vrcp14pd zmm
  measures 0.69 ns = ~2.3 cyc on this part** (mapbench, new) -- NOT
  microcoded; corpus IMPLEMENTATION_NOTES' "~10 cyc" caution is FALSE here,
  same verdict as r4 gave for vrsqrt14pd.  The +4 FMA-port ops simply cost
  more than the divide's marginal slot: the divider was never binding.
- **Mixed divider-offload map (every Nth pair via vsqrtpd+vdivpd,
  L17_MAP_MIX): mix=4 +0.72 us, mix=2 +1.7 us** -- vsqrtpd zmm is ~2x a
  vdivpd and serializes behind the shared divider.  Confirms r4's s2
  verdict from the other direction; the hook stays in the source as the
  record.
- **c-stream residency probe (-DL17_V7_CPROBE, timing-only)**: reusing slab
  0's c for all slabs (kills the per-step 314 KB c read) gains only
  0.05 us/step -> c-prefetch/layout work is capped there; line closed.
- The MG=1 and MG=4-with-x2 losses above.

### Borrowed this round

- **Warm cohort d43251c2 (0.99)** was read as directed; its prime-engine
  levers are already in v7 or previously measured losers (list in "Where
  this round started") -- adopted nothing new, and that conclusion is
  itself the round's mining result: our engine already embodies the 0.99
  attempt's structural ideas at lower op count.
- The x2 interleave is the corpus ZIPP/text-order lesson (gcc does no
  pre-RA scheduling on x86) applied to our OWN map body -- the same blind
  spot as r3's pragma episode, one level down.

### Score projection

r7 scored 9.035 from dev 9.04-9.06; this round's dev floor is 8.83-8.86
under the same protocol, so expect **~8.8-9.0 us/step in the quiet window**
(~0.0277 s for the graded point).  Winograd's r7 stands at 10.922 with 17%
run spread (its agent should fix that before it threatens); best measured
rival 10.68.  First place should hold with margin.

### Open for next round

- **The map is now the measured, understood wall**: ~16.5 cyc/pair
  standalone-rate with an ~8.5-9 cyc port floor, and NO overlap with kernel
  chunks is possible (burst 510 uops > ROB; fusing into the chunk = r5's
  +1.7 register-fusion loss).  The only shapes that could beat it: (a) a
  map body with a fundamentally shorter chain (none found: Q4 is one op off
  optimal, divider/rcp14/sqrt alternatives all measured), or (b) making
  bursts ROB-small AND chunks smaller so they interpenetrate -- that means
  splitting the 148-op kernel chunk, a codegen experiment someone could
  price with objdump before building.
- j0/j1 are AT the port floor; the FFT side is closed unless the op count
  itself drops (seven rounds say it does not).
- If the monitor reruns rivals: compare against the warm cohort's node
  numbers when results/rivals_icelake grows them; our 8.83 vs their
  VM-tier r=0.145 should stay ahead, but verify, don't assume.
- MONITOR bugs live this round: check.py missing `import math` (line 6);
  tryout.sh $W (r4 note); driver --map --chain 1 NULL pong (use --chain 2
  for the one-step gate, or allocate pong at chain==1).
