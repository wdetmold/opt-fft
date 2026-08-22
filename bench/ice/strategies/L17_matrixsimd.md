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
