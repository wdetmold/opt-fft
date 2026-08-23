# L36_pencilfused — ice-panel strategy record

Full lineage (rounds 1–panel_r11, CLX/SPR era) lives at
`bench/geom/strategies/L36_pencilfused.md` and is the authoritative history of
this file's kernels, modes, and dead ends. This record starts at the ice panel.

## Round ice_r1

No entry: the implementer agent was killed at launch by the host's
worker-crash storm (15 of 19 agents died in the first minute; verdict §3).
The scored 123.594 µs/xform (B=8 chain, 3rd of 3 panel entries, 1.04× behind
L36_pfa's 119.163) is the untouched panel_r11 CLX code re-measured on the
Ice Lake node.

## Round ice_r2

### Where the round started, and the diagnosis

Scored ice_r1 (untouched CLX code): 123.594 µs/xform at the graded cell
(B=8, chain m=64, unitary), 3rd behind L36_pfa 119.163 and L36_mixedradix
119.530. Three verdict facts framed everything done here:

1. **The plan race is bigger than the gap.** Identical binaries swung −11.1%
   between scoring windows (my own probe read 116.0 in one window, 99.4 in
   another, on the same bytes); the verdict's rule: "nothing under ~12% is a
   result unless the plan is pinned."
2. **Every non-chain arena was systematically optimistic** (verdict §4a):
   my in-arena probe ip4=99.4 vs 123.6 scored (+24%); pfa's fu=81.0 vs
   119.2 (+47%). The graded chain re-scales the whole batch inside the
   timed unit and ping-pongs buffers; no tuner modelled that. The one entry
   whose probe matched its score to 2.2% (L36_mixedradix) took nothing from
   its kernel — it just wasn't lied to by its arena.
3. **The machine changed under the code**: on ICX the second 512-bit FMA
   pipe lives on port 5, so the 1296 transpose shuffles per plane in pass A
   (free riders on CLX) now displace FMA work (the verdict's L=17 port-5
   mechanism, §5).

First act of the round: a forced-mode A/B of the EXISTING modes under the
real graded chain on the node itself (tryout.sh runs `--chain 64 --unitary`
— nobody in this file's lineage had ever measured the chain directly):

| mode (pw4, forced) | min µs/xform | median | note |
|---|---|---|---|
| 0 inplace (y-first) | 129.4 | 140.1 | winner of the old set |
| 11 inplace-cs | 129.9 | 129.9 | tied min, far stabler median |
| 7 istream (z-first, T1 cursor) | 134.6 | 134.7 | cursor HELPS z-first under the chain |
| 8 istream+pfw | 134.9 | 134.9 | prefetchw buys nothing (out is L3-resident) |
| 12 istream0 (z-first, pf0) | 138.0 | 138.1 | pfa's B=1 shape loses MY chain cell |
| 1 scratch | 154.3 | 158.6 | dead at chain scale |

(Same window, MKL reference 168.4—170.4 µs; the ice_r1 scoring window had
MKL at 160.7, so absolute numbers here are ~5% pessimistic across the board.)

### Technique (round ice_r2) — five changes

1. **Transpose-free broadcast pass A — modes 13 BCST0 / 14 BCST+PF**
   (ADOPTED from **L36_pfa's ice_r2 tr=1 kernel**, in their file when this
   round opened; translated onto my pass shapes). Every lane-transposed
   input vector is assembled as 4 masked `vbroadcastf64x2` loads (one
   load-port µop each, zero port-5; corpus S10's bare-metal note — ICX
   folds broadcasts into the load µop, 2×64 B loads/cycle); both subloops
   then store plain full vectors. Subloop A: z-transform, lanes=y,
   broadcasts straight off the in-plane rows, plain stores into pp[kz][y].
   Subloop B: y-transform, lanes=kz, broadcasts off pp's 72-double row
   stride, plain stores to the out plane rows ky. This deletes ALL 1296
   transpose shuffles per plane AND both 36-vd staging arrays — the r11
   spill surgery — in one move: objdump confirms 288 memory-folded
   vbroadcastf64x2, zero transpose shuffles (the 2×57 kernel CSWAPs
   remain), and rsp-relative vmovs drop 114 → 52 in the pass-A body.
   Mode 13 = prefetch-free twin, mode 14 = mode 7's paced T1 cursor +
   PFNX. PW=4 only. L13_rader's CLX warning about merge-broadcast loads
   (+1.36 µs there) does not bite: their chain was 8-deep on one vector,
   this is 4-deep with 36 independent vectors per PFA36 call in flight.
2. **Chain-shaped tuner arena** (idea from **L17_matrixsimd ice_r1**, who
   tuned in-regime and was accurate to 1.3%). The timed unit in
   fft3d_create() is now a chain step: exec(src,dst), then the driver's
   own unitary scale (×1/216 exactly — sqrt(36³) = 216) over dst as a
   separate pass, then ping-pong through three buffers (din/dout/dpong),
   byte-faithful to driver.c RUN_UNIT. Chain state persists across
   candidates (the transform is norm-preserving under the scale — no decay,
   no denormals); one untimed warm step per visit keeps the r10 self-warm
   discipline. Candidates are now ranked by the scored workload, and the
   description-string probes are directly comparable to leaderboard
   numbers: probe bc4=127.8 vs 128.5 measured end-to-end in the same
   window — 0.5% accuracy, against +24% last round.
3. **Clock-settle spin** (~120 ms of dependent FP work) before any
   candidate is timed (from **L17_matrixsimd ice_r1**: schedutil probes an
   unramped core at 2.90 GHz and mis-ranks plans).
4. **Gates re-keyed to the detected L3** (`_SC_LEVEL3_CACHE_SIZE`, ×1.25):
   prefetchw and NT/XV/pipe candidates admitted only past L3 —
   **L13_rader ice_r1** (pfw on L3-resident out = +7.4% tax; confirmed
   here: mode 8 = 134.9 vs 129.4) and **L17_matrixsimd ice_r1** (NT in the
   L3-resident chain = 2× catastrophe). SCRATCH additionally excluded
   below the NT threshold on this round's own 154.3-vs-129.4 measurement.
5. **Anti-phase traversal** (this file's own mechanism; y-first-only after
   measurement — see the dead-end list): the driver's chain-scale pass
   walks dst volume 0→7, plane 0→35, immediately before the next execute
   reads it back as src; so the y-first modes (0, 11) now process volumes
   7→0, planes 35→0, starting each execute on the lines the scale pass
   touched last (still L2-resident) and finishing exactly where the
   driver's next forward scale begins. Bit-identical output;
   `-DFFT36PF_FWD` restores forward order. Cursor modes and the bcst modes
   keep forward order — the bcst pass A's whole advantage is its
   sequential cold read, and reversal was measured to defeat it.

### Operation count

Line kernel unchanged: 232 FMA-port ops + 57 port-5 swaps per 36-point line
over PW lanes (PFA 4×9, n1_9 DAG); 225 504 FMA-port vector ops/volume at
PW=4. What changed is the port-5 total: pass A's transpose shuffles
(46 656/volume) are gone in the bcst modes, so the two-pipe port floor
drops from (225 504 + 102 060)/2 = 163 782 cyc = 56.5 µs to
(225 504 + 55 404)/2 = 140 454 cyc = 48.4 µs at 2.90 GHz — identical to
pfa's tr=1 arithmetic, as it must be (same kernel, same trick). The cost:
loads triple in pass A (2 592 broadcast µops/plane vs 648 vector loads),
riding ports the pass leaves idle (bare-metal ICX does 2×64 B loads/cyc).

### What was measured (all on the NODE, a80n0, graded chain B=8 m=64
unitary via tryout.sh, leased core; MKL on the same core quoted as the
window reference; rel_l2 vs numpy and the m=64 closed-form chain check PASS
on every run listed, and every run printed "repeatable: identical output")

* **Headline, quiet window (MKL 167.5–168.4):** forced bcst0 **118.8** min /
  129.1 median; forced bcst+pf **122.3/122.3**; forced mode 0 **127.2/127.3**;
  **auto 118.3 min / 118.4 median, sd 0.05%** — the chain-shaped tuner
  installed pw4/bcst0 through the 3% cross-class guard on a genuine ~7%
  in-arena margin. Same-window reference points: ice_r1 scored this file at
  123.6 (window MKL 160.7) and pfa at 119.2.
* **Verbose in-plan tournament (separate, slightly busier window):**
  pw4: bcst0 127.8, bcst+pf 132.2, inplace-cs 137.4, inplace 137.9,
  istream0 145.2, istream 147.1; pw2: 164–174. End-to-end same run: 128.5.
  The chain probe now predicts the end-to-end number to 0.5%.
* **Final-state confirmation, three consecutive auto runs at B=8** (after
  the mode-13 reversal revert): min 118.1 (window MKL 168.5), 118.5 (MKL
  193.9), 129.7 (MKL 185.4) — the tuner installed pw4/bcst0 with the
  3.586e-16 fingerprint in ALL runs, quiet and contended alike; the pick
  is deterministic even where the timing is not.
* **B=1** (not a graded cell; sanity): auto 120.7 µs/xform vs MKL 139.9,
  fingerprint 3.591e-16 — the deterministic B≤2 class gate installed the
  z-first class on this 1.25 MB-L2 machine as designed. **B=32**: auto
  174.6 vs MKL 236.4, setup 1.77 s, sd 0.05%.
* Fingerprints: bcst modes are the z-then-y bit class, 3.586e-16 at B=8
  (chain 1.261e-14), same family as modes 7/8/12; y-first modes unchanged
  at 3.763e-16.

### What was tried and did NOT work — with the number that killed it

1. **Anti-phase reversal on the bcst mode: measured harmful, reverted.**
   Same-session forced A/B at B=8: mode 13 reversed = 142.5 vs forward =
   123.0 µs/xform min, with the MKL reference drifting only 7% between the
   runs (191.5 vs 179.2) — ~8% real damage. Mechanism: descending-plane
   order breaks the ascending sequential read stream that IS the broadcast
   pass A's advantage on cold/L3 input (pages straddling the 20.25 KB
   plane boundary get revisited with backward jumps, and the L2 streamer
   only tracks ascending within a page). Mode 13 shipped forward.
   On the y-first modes (strided reads, no stream to defeat) the same A/B
   was unresolvable above the leased-core noise — mode 0 rev-vs-fwd 136.2
   vs 134.8 min while MKL itself swung 170.1 → 205.0 (19 agents active).
   Kept there: bit-identical, mechanically motivated, and those modes are
   no longer the pick, so it only touches the ip4/cs4 probe values;
   `-DFFT36PF_FWD` is the one-flag revert.
2. **prefetchw at the graded cell** (mode 8): 134.9 vs 129.4 — L13_rader's
   ICX rule confirmed at L=36; the gate now excludes it below L3.
3. **SCRATCH at the graded cell** (mode 1): 154.3 vs 129.4 (−19%) — the
   private mid buys nothing L3-resident and costs a third volume pass.
4. **First auto run after the tuner rewrite landed at 157.7** — not a code
   defect: that window's MKL read 180.2 and the next quiet-window auto
   read 118.3 with an identical binary. Recorded as a live demonstration
   of why single-window numbers on leased cores decide nothing (the
   verdict's ~12% rule, reproduced within one hour).

### Attribution

* Broadcast transpose-free pass A: **L36_pfa ice_r2** (`tr=1`), verbatim
  mechanism, re-derived onto this file's plane-fused pass shapes (their
  phase-1 layout differs; my subloop A stores [kz][y] so BOTH subloops get
  plain stores). Their objdump verification protocol reused.
* Chain-shaped/in-regime tuning and the clock-settle spin:
  **L17_matrixsimd ice_r1**.
* L3-keyed prefetchw gate: **L13_rader ice_r1**. NT-past-L3-only:
  **L17_matrixsimd ice_r1**.
* Anti-phase traversal: this file, this round (novel, unproven).
* The forced-mode chain A/B protocol (measure the graded chain directly
  with FFT36PF_FORCE_MODE before believing any arena): this file, this
  round — recommended to every entry; tryout.sh makes it free.

### Predictions for the node (stated so they can be scored)

* Pick: `pw=4 mode=bcst0`, deterministic (in-arena margin ~7% over the
  best y-first candidate, far outside the 3% guard and the arena's ~0.5%
  repeatability). Fingerprint at B=8 moves 3.763e-16 → **3.586e-16**; the
  description reads `chain probe us ip4=… cs4=… bc4=…` with bc4 lowest.
* **B=8 chain: 113–119 µs/xform** in a scoring window whose MKL lands near
  160.7 (my quiet-window 118.3 came with MKL at 167.9; scaling by the MKL
  ratio gives ~113.5, but I do not trust cross-window scaling to better
  than a couple of µs — hence the band). Anything ≥123 = the bcst pick
  failed to transfer, read the probe string first.
* pfa ships the same tr=1 mechanism, so parity or a photo finish is the
  honest expectation there; the differentiators are my chain-shaped pick
  (their tournament still ranks by non-chain arena) and the anti-phase
  order.
* Setup ≤ 0.7 s at B=8 (measured 0.58–0.65).

### Next

1. **Pass B is now the port-5 frontier**: 57 CSWAPs per group against 232
   FMA ops (floor share 16.2 µs/vol of the 48.4). The swaps implement ±i
   in interleaved complex; killing them needs split-complex lanes through
   pass B (load-time deinterleave via the same masked-broadcast trick,
   store-time re-interleave), worth a measured attempt only if the p5
   counter confirms pass B binds there — read the PMU (the brief says
   perf_event_open works) before building it.
2. If the scored bc4 probe ≈ end-to-end but both sit ≥4 µs above pfa's
   tr=1 number, the residual is their phase-1 store layout ([y][kz] mid,
   sequential mid stores) vs my [kz][y] pp scatter — port the store order,
   not the whole phase.
3. The B=32/B=256 streaming cells still run mode 8 (istream+pfw) with the
   OLD pass A; bcst twins with the write-intent cursor (a bcst+pfw mode)
   are one dispatch branch away if those cells are ever graded here.
4. Chain-shape the B≤2 arena's deterministic gate reading: at B=1 the gate
   chose z-first on L2-overflow physics and bcst0 then won the class
   tournament — but nobody has verified y-first-vs-bcst at B=1 under the
   chain with forced runs. One tryout pair settles it if a B=1 cell ever
   appears in cases.txt.

## Round ice_r3

### Where the round started

Scored ice_r2: **110.477 µs/xform, 1st** (pfa 112.727, mixedradix 116.814,
MKL 161.140). The chain probe bc4=110.0 predicted the score to 0.4% — the
arena is now trustworthy, so this round's decisions were made off it plus
forced node A/Bs. The margin over pfa is 2%, and their record says their
next move is split-complex, so standing still was not an option.

### Technique (round ice_r3) — volume-only anti-phase reversal shipped;
### two overlap mechanisms built, measured, and rejected by the node

Five new tuner modes on the bcst (transpose-free broadcast) pass A, all
PW=4, all gated to L3-resident chain batches:

1. **mode 17 `bcst+vr` — volume-only reversal. THIS IS THE ONE THAT WON
   AND SHIPS.** The r2 anti-phase idea, minus the part that killed it: the
   driver's unitary-scale pass walks dst volume 0→7 ascending immediately
   before my execute reads it back as src, and reads my dst volume 0
   ascending immediately after I return. Mode 17 processes volumes 7→0
   while keeping every plane and every intra-volume access ascending
   (the r2 full reversal descended through PLANES too, breaking the
   sequential cold-read stream — 142.5 vs 123.0; volume-granular reversal
   keeps each volume's read one ascending 746 KB stream). Execute starts
   on the volume the scale pass left L2-warm and finishes at volume 0,
   exactly where the scale pass reads next. In-arena vs the bcst0
   incumbent: **120.7 vs 122.4** (window 1), **121.9 vs 123.7** (window
   2) — a consistent −1.4/−1.5%, picked through the 1% in-regime band in
   both windows. Bit-identical output (3.586e-16 fingerprint unchanged).
2. **modes 15/16 `bcst+xv[+vr]` — cross-volume L2 staging: REJECTED by
   the node, +9.6%.** The theory: pass B is compute-bound on the
   L2-resident volume (every out line is loaded once, stored once, then
   dead), so its ~20 µs memory-idle window could stage the next volume's
   input into L2 with paced prefetcht1 (36 lines per line group × 324
   groups = exactly one volume) — mode 3's XV relocated to the chain
   regime where no NT drain can drop the prefetches. Measured in-arena:
   xv 134.2/135.7, xv+vr 132.3/134.1 vs bcst0 122.4/123.7. The eviction
   side of the ledger wins: 746 KB of incoming stream thrashes the same
   1.25 MB L2 that must hold the out volume pass B is still reading, and
   the sequential in-read was already HW-prefetch-covered under pass A's
   compute. Same L2-capacity arithmetic kills any pass-A/pass-B
   interleave across volumes (pass B's working set is one FULL volume by
   construction — the x-transform reads one line from all 36 planes per
   group — so any concurrent stream overflows L2); recorded so nobody
   builds the demand-load version of this.
3. **modes 18/19 `bcst+nta[+vr]` — L2-bypassing paced NTA in-read:
   REJECTED by the node, +7.8% at B=8, +5.2% at B=1.** Theory from the
   phase split (below): pass A's read-once in stream evicts the freshly
   written out volume from L2, so pass B re-reads dirty L3 lines and
   doubles writeback traffic; prefetchnta (fills L1, bypasses L2 on this
   core) with pfa r6's discipline — constant 4 KB lead, issued at exactly
   the consumption rate, subloop A only — should keep out L2-resident
   through both passes. Measured: nta 133.3, nta+vr 131.1 vs bcst0 123.7
   (B=8); nta 120.2 vs bcst0 114.3 (B=1). With this, every prefetch
   flavor is now priced on this chain cell and ALL are taxes: T1 cursor
   (+3.5%), prefetchw (r2, +4%), XV-t1 (+9.6%), NTA (+7.8%), pfa's
   pf=1/2/7 (+4-7%). The mechanism-level reading: the in-read is one
   ascending HW-prefetchable stream that the L2 streamer already handles
   at L3 bandwidth, and everything software adds either burns uops or
   displaces L2 lines that were about to be used.
4. `-DFFT36PF_VERBOSE_CC` — compile-time twin of the env verbose flag,
   because env does not cross tryout's ssh (L17_matrixsimd's lesson via
   mixedradix's record). This is what made the in-plan tournament
   readable on the node this round; recommended to anyone still
   env-gating diagnostics.

### The phase split (new protocol this round: forced mode + SKIPA/SKIPB
### through tryout's extra-gcc-flags argument — values shrink 6^-64 over
### the chain but never denormalize, so the timing is clean)

Same busier window (MKL 171–172): full mode-17 chain step 129.4; pass A +
driver scale only (SKIPB) **102.3**; by subtraction pass B ≈ **27 µs/vol**
against its 16.1 µs two-pipe floor, pass A ≈ 82–84 against its 32.3, scale
~18–20 (mixedradix's number, same driver). Per-volume L3 traffic is
compulsory-ish: 746 KB in-read + 746 KB out-RFO + 746 KB writeback +
~1.5 MB scale-pass RMW ≈ 3.7 MB, which at this core's ~30 GB/s L3 is
~120 µs — the cell is **L3-bandwidth-bound end to end**, which is why
every latency-hiding mechanism above failed: there is no idle bandwidth to
schedule into, only idle cycles. The remaining headroom is traffic
reduction, and a two-pass shape with distinct in/out has none left to cut.

### Operation count

Unchanged from ice_r2: PFA 4×9 line kernel (232 FMA-port ops + 57 port-5
swaps per 36-point line over PW lanes), bcst pass A shuffle-free, 225,504
FMA-port vector ops + 55,404 port-5 swaps per volume at PW=4, two-pipe
floor 48.4 µs/vol at 2.90 GHz. Modes 15–19 add zero arithmetic (prefetch
and ordering only); mode 17's output is bit-identical to bcst0.

### What was measured on the node (tryout.sh, graded chain B=8 m=64
### unitary unless stated; MKL same core as window normalizer; rel_l2 =
### 3.586e-16 vs numpy and chain check 1.261e-14 PASS on every run, all
### runs "repeatable: identical output")

* **Quiet window: auto pick pw4/bcst+vr, min 109.852 / median 109.9,
  sd 0.34%, MKL 159.3** — best state this round; MKL-normalized 0.690 vs
  the ice_r2 scored ratio 0.686 with vr's −1.5% on top.
* Second verbose window: auto min 109.993 (median noisy, sd 10%), MKL
  159.6; in-arena table: bcst+vr 120.7 < bcst0 122.4 < bcst+pf 126.9 <
  inplace-cs 131.9 < inplace 132.3 < xv+vr 132.3 < xv 134.2 < istream0
  137.9 < istream 140.8; pw2 158–171.
* Third window (busier): auto 122.381 min, MKL 166.1 — the ~12%
  window-swing rule reproduced again; the pick is stable even where the
  timing is not.
* Forced mode 17 (setup 0.17 s): 129.4 min in the MKL-172 window; SKIPB
  102.3; SKIPA run lost its timing line (uninitialized-buffer chain —
  admission correctly rejected the candidate; use SKIPB + subtraction).
* B=1 (not graded): auto 103.070, sd 0.03%, MKL 157.2 (quiet); verbose
  window pick bcst0 114.3 in-arena over inplace 119.7, nta 120.2.
  B=32: auto 169.828, MKL 229.5 (ice_r2: 174.6 / 236.4). Neither cell
  regressed; setup 0.82 s at B=8, 1.86 s at B=32.

### What was tried and did NOT work — with the number that killed it

1. **XV cross-volume L2 staging (modes 15/16)**: 134.2/135.7 vs 122.4/
   123.7 in-arena at B=8 (+9.6%). L2-capacity physics, not pacing —
   see above. Do not rebuild as a demand-load interleave; same capacity
   wall.
2. **Paced-NTA in-read (modes 18/19)**: 133.3 vs 123.7 at B=8, 120.2 vs
   114.3 at B=1. pfa's B=1 L2-eviction diagnosis does not transfer to
   this pass shape even at B=1.
3. **mixedradix's pind 4K-alias pin: considered, not built.** Their own
   record measures it FLAT at B=8 (111.7–112.1 over four residues) —
   the subloop-B stores that would alias are RFO-drain-bound at the
   graded cell, hiding the stalls. Their −22% is a B=1 effect and my
   B=1 cell is not graded. One build saved by reading their record.
4. The SKIPA diagnostic produces no timing line (see above): admission
   runs candidates against a pass-B-only reference on a buffer the
   skipped pass A never initialized, and the malloc garbage differs
   between the v2 reference and v4 candidate runs, so everything fails
   admission. SKIPB + subtraction is the working half of the protocol.

### Borrowed this round

* pfa r6's NTA pacing discipline (constant short lead, consumption-rate
  issue, first-consumer subloop only) for modes 18/19 — mechanism
  rejected here, discipline sound.
* The "env doesn't cross ssh, use -D hooks" lesson: L17_matrixsimd via
  mixedradix's ice_r2 record.
* mixedradix's B=8-flat pind measurement, used as a do-not-build.
* Volume-only reversal is this file's own mechanism (refinement of my
  ice_r2 anti-phase traversal, restricted to the granularity that does
  not break the bcst read stream).

### Predictions for the node (stated so they can be scored)

* Pick: `pw=4 mode=bcst+vr` at B=8, deterministic (two windows, both
  −1.4/−1.5% in-arena, band 1%). Fingerprint stays 3.586e-16; the
  description reads `... bc4=… xv4=… xr4=… vr4=… nt4=… nv4=…` with vr4
  lowest of the bcst family.
* **B=8 chain: 108–112 µs/xform** in a ~160-MKL scoring window (quiet
  measurement 109.85 at MKL 159.3). Anything ≥118 = the vr pick failed
  to transfer; read the probe string first.
* pfa ships split-complex if their round landed; if it works their
  port-5 floor drops ~10 µs/vol and they may retake the cell — the
  counter is not prefetch (all priced, all taxes) but their same trick
  ported here.

### Next

1. **The only structural door left at this cell is traffic**: the
   ~3.7 MB/vol L3 traffic is compulsory for two passes + driver scale.
   A single-pass shape does not exist for 3 axes, but a HALF-VOLUME
   z-split (process x-planes 0–17 through both passes while planes
   18–35 of the NEXT... does not work either: pass B needs all 36
   x-planes per line. Written down so the next round doesn't re-derive
   it: pass B's full-volume working set is a theorem, not a choice.
2. **Split-complex through pass B is net-negative here**: boundary
   deint/reint costs 144 port-5 shuffles per 8 lanes against the 114
   swaps it deletes (pass A's broadcast loads cannot assemble split
   vectors without doubling load uops to 576/call, which becomes the
   new bound at 288 cyc > 232). Done on paper this round so nobody
   builds it; pfa's version must eat the same boundary unless they
   split the WHOLE pipeline including the mid layout — watch their
   record, not their probe.
3. If the scored number sits ≥4 µs above the 109.85 quiet measurement,
   it is window drift, not code: ask the monitor for the probe string
   (vr4 vs bc4) before changing anything.
4. The B=1 cell (if ever graded): bcst0 wins it now (103.1 quiet); the
   deterministic B≤2 class gate + tournament already handle it.

## Round ice_r4

### The task changed: the map is the battleground

The graded step became `state <- (z+c)/(1+|z+c|)`, z = RAW (unnormalized)
FFT(state) — **no unitary scale in map mode**, the map bounds the state — and
the driver times an exported `fft3d_chain(plan, x0, c, final_out, m)` weak
symbol as the whole m-step unit (fallback = fft3d_execute + a driver-side
unfused map, the 2.24 s configuration).  Everything this round is that entry
point; fft3d_execute and its whole tuner survive untouched as the
single-transform correctness path.  Rivals' mark at L=36: 0.059 s / (64·8) =
**115.2 µs/step**.

### What ships

1. **fft3d_chain, volume-resident, in-place, ONE state buffer.**  Volume b
   runs all m=64 steps before b+1 (the doctrine all three L17/L23 ice_r4
   records converged on).  My pass A is plane-local — the whole source plane
   drains through mp/pp before its rewrite — and pass B was already the
   in-place path, so a step needs NO ping-pong: S (746 KB, = p->mid) owns L2
   for the whole chain.  Step 1 reads x0 (const, unmapped — x0 IS a state);
   the chain ends with one whole-volume map into final_out.  restrict was
   REMOVED from both pass-A functions' rsrc/rdst (the in-place call was UB
   with it; pp keeps its restrict, which is the aliasing fact that matters).
2. **Lazy map (the rivals' doctrine, corpus §10 §2), staged per plane.**  S
   holds RAW z between steps; `mapplane` converts plane x through a 20 KB L1
   scratch mp that immediately feeds passA_bcst, so S and c both stream
   SEQUENTIALLY.  The mapped state never exists in memory outside mp.
3. **Pair-compressed map arithmetic** (`map2`, ADOPTED from **L23_rader
   ice_r4's L23R_MAP2**): two vectors' |w|² pair-sums compressed into ONE
   vector (2 two-source vpermt2pd + 1 add), ONE vrsqrt14pd + 2 Newton ladder
   and ONE hardware vdivpd per 8 points, two expand shuffles + 2 muls to
   apply.  The +1e-300 additive bias folded into the square (L17_winograd's
   NaN-guard trick) replaces both a clamp and the rsqrt14(0)=inf hazard.
   d = fma(|w|², r, 1.0) merge from L23_matrixsimd.  ~15 FMA-port ops +
   4 shuffles + 1 rsqrt + 1 div per 8 points ≈ 5832 pairs/volume.
4. **Chain width chosen by ISA (`chainpw`), never by a race** — the chain's
   bits are process-deterministic by construction (md5 of out.bin AND
   out.bin.chain identical across runs, verified).  PW=2 fallback (non-AVX512
   machines only) keeps a scalar-exact per-vector map.
5. mp placed at mid+MIDSKIP+16: MIDSKIP alone lands mp on S plane 6's exact
   mod-4096 phase (L23_rader's pathology); +128 B is maximally distant from
   all 16 plane phases.  Measured a wash on the node (113.3/113.3/113.0
   A/B/A) — kept as the principled default.

### Operation count

FFT unchanged (PFA 4×9, 232 FMA + 57 swaps per line over PW lanes, 225,504
FMA-port vector ops/volume at PW=4).  Map per volume: 5832 pairs × (~15
FMA-port + 4 p5 shuffles) ≈ 111k vector ops (~19 µs issue at two pipes) +
5832 vdivpd on the otherwise-idle divider (~17 µs, overlapped).  THE ROUND'S
ARITHMETIC LESSON: the naive per-vector map is ~24 ops × **11664** vectors
(not 2916 — my first estimate was 4× low) ≈ 48 µs/step of issue, and it
measured 55–70 µs/step in EVERY placement tried.  Count first.

### Measured on the node (tryout.sh = a80n0 leased core, graded map-chain
### L=36 B=8 m=64; windows drifted MKL 287→332 over the session, same-window
### contrasts only; single-transform rel_l2 = 3.586e-16 and map-chain m=64
### rel_l2 = 1.240e-14 vs tol 6.4e-12 PASS on every configuration listed)

Progression, one evening: fallback-shape estimate ≥150 → eager first cut
143.0 → lazy staged (per-vector map) 130.6 → **pair-compressed lazy 114.8 →
113.3/112.9 final** (best quiet windows; MKL 288–307 same windows; rivals'
mark 115.2).  B=1: 112.1 (chain 1.296e-14); B=32: 113.1 (chain 1.423e-14) —
per-volume chaining makes the step batch-invariant.  Setup 0.79–0.93 s.
Knob A/Bs, same windows: MAPRCP (rcp14+2NR instead of vdivpd) 116.6 vs 114.8
— the divider is FREE in this pass, L23_matrixsimd's consensus shape wins
here too; CPFIN (t1 c at consumption) +1.0; SPF (S one-plane-ahead t0) +1.5.
Phase splits that drove the round: SKIPB (mapplane+passA) 116.4, NOPA
(mapplane+passB) 79.9, SKIPB+NOC 108.7, passA alone 50.9, passB ≈ 14–18.

### What did NOT work, with the number that killed it

1. **Eager map at pass B's strided store sites, c t0-prefetched: 143.0.**
   c's 746 KB L2 transit rotates S+c = 1.5 MB through the 1.25 MB L2 — the
   cyclic-sweep LRU pathology, every line evicted just before reuse, whole
   step at L3 speed.
2. **NTA on the strided c (eager): 192.**  c's 20736-B stride advances only
   4 L1 sets per line → NTA lines self-evict pre-use, AND NTA never
   allocates in L3 on this core, so c re-reads from DRAM.
3. **NTA on the sequential c (lazy): 167.**  Same L3 demotion: NTA hits
   don't refresh L3 age, c ages out, every step re-reads DRAM at ~12-LFB
   latency-bound ~8 GB/s (SKIPB split: 149 vs 51).  **NTA is only safe for
   data you will never re-read**; c is re-read every step.
4. **Map fused into pass A's broadcast load sites (passA_bcstm): 182 stable
   (SKIPB 124.5 vs staged 116.4).**  36 map expansions inside SA_'s live
   u[36] set spill hard; deleted from the file.
5. **Staging the map as its own pass with PER-VECTOR mapv: ~60 µs/step** for
   what should be 48 — L17_winograd's staging-trap at my scale.  Pair
   compression, not placement, was the fix (their +0.44 µs fusion is a
   different regime: their pass 3 is shuffle-bound with idle FMA pipes).
6. **Cursor-staging c one plane ahead through pass A's pfc: +3 µs** (133.8
   vs 130.6).  The step is L2/L3-bandwidth-shaped, not latency-shaped —
   ice_r3's "every prefetch is a tax on cache-resident chains" survives the
   task change intact.
7. tryout.sh is broken this round (W unbound at line 36; remote map-check
   gets `--cin /c.bin`): workaround `W=$PWD/build/tryout/<name> ./tryout.sh
   ...` and run check.py by hand — found independently, matches L23_rader's
   and L17_winograd's notes.

### Borrowed this round, named

* Lazy map + fuse-where-c-streams doctrine: the rival pipelines (corpus §10
  §2) via the brief.  Volume-resident chaining: L23_matrixsimd /
  L23_rader / L17_winograd ice_r4 (all three, independently).
* Pair-compressed map: **L23_rader ice_r4** (L23R_MAP2), with my rsqrt
  ladder kept on the FMA pipes.  One-hardware-divide reciprocal:
  **L23_matrixsimd ice_r4** (mv=0), confirmed against its rcp14 twin here.
* +1e-300 additive NaN-guard bias: **L17_winograd ice_r4**.
  d = fma(m2,r,1.0): **L23_matrixsimd ice_r4**.
* mod-4096 scratch-phase discipline: **L23_rader ice_r4** (a wash here, but
  placed by rule).  vrsqrt14pd-not-microcoded evidence: L23_matrixsimd.

### Predictions for the node (stated so they can be scored)

* **B=8 graded cell: 110–116 µs/step** (best quiet 112.9 at MKL-307,
  113.3 at MKL-289).  Rivals' 115.2 falls if the window is fair; the FFT
  itself is unchanged, so any regression ≥5 µs is window, not code.
* Description reads `fchain pw=4 volres inplace lazymap2 rsqrt14+2NR+vdiv`.
  Chain end state bit-identical across processes; chain rel_l2 1.24e-14.
* Setup ≤1 s (the exec tuner still runs; it prices a path that is no longer
  scored but costs only setup time).

### Next

1. **passA is now the frontier again: 50.9 of the 113** (floor ~32 + loads).
   The mp round trip could go if map2 wrote pp-shaped output directly and a
   bcst-variant subloop A read it — a HALF-fusion (map into subloop A's
   consumption order, not into SA_ registers) that avoids the bcstm spill
   trap because the map runs before the FFT's live range opens.
2. passB (14–18 vs 16.1 floor) and mapvol are done; the c read (~5–10 in
   the final shape) is compulsory.
3. If a rival record shows the map cheaper than ~19 µs/step at full double,
   read their reciprocal: the remaining map cost is the ladder + divide,
   not the data movement.
4. The exec-path tuner is vestigial for scoring; if setup time ever
   matters, gate it behind batch>threshold or drop to the two known picks.
