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

## Round ice_r5

### Where the round started

Scored ice_r4: **111.962 µs/step, 2nd** at the graded cell (36:8:64 map
chain), 0.5% behind L36_mixedradix's 111.425 (L36_pfa 128.551, MKL 283.3).
My r4 record's own "Next" list said passA+map is the frontier; the round
context added one hard fact from the leader's record: their mB map style
(vsqrtpd on the divider unit) beat their ladder style by ~3.5 µs in-arena.
Both threads were pulled this round — one shipped, one died instructively.

### Technique (round ice_r5): ROW-GROUP-INTERLEAVED lazy map — this ships

The r4 chain staged the map per PLANE: `mapplane` read S+c and wrote a
20.25 KB `mp`, then passA_bcst's subloop A re-read mp through broadcasts.
Two structural costs, invisible until listed together: (a) mp + pp =
40.5 KB against the 48 KB L1, with S/c lines transiting on top — by the
time subloop A's broadcasts returned to mp's early rows they had been
evicted to L2, a ~20 KB/plane (746 KB/step) staging re-read; (b) mapplane
is a tight ~15-op/pair loop in which the 5832 vdivpd/vol have nothing to
hide under.

New shape (`maprows` + `passA_bmap`): row group yb+4 (4 rows = 2.25 KB =
18 map2 pairs) is mapped into a 2 × 2.25 KB ping-pong immediately BEFORE
the FFT of row group yb (one-group software pipeline; plane's group 0 in a
short prologue). Consequences: the staging set (4.5 KB) is always L1-hot;
the map's divides and Newton ladders issue into the same OOO window as
PFA36's ~470 µops of FMA/shuffle/load work — the exact mechanism that
makes mixedradix's fused-at-load map cheap, applied at CHUNK granularity
instead of register granularity, which is what avoids the r4 bcstm spill
trap (182 µs/step). The ~400-µop store-to-load distance also keeps the
group's map stores clear of its own 16-B broadcast reloads. Map
arithmetic is unchanged per point, so the chain output is BIT-IDENTICAL
to r4 (verified: chain rel_l2 1.240e-14 exactly as r4, and out.bin cmp
across processes identical). s=0 reads x0 rows directly, unmapped, as
before; PW=2 fallback keeps the old whole-plane flow.

`-DFFT36PF_CHV0` rebuilds the r4 whole-plane flow for A/B.

### Operation count

Unchanged: PFA 4×9 line kernel (232 FMA-port + 57 port-5 per line over PW
lanes), 225,504 FMA-port vector ops + 55,404 swaps per volume at PW=4;
map = 5832 pairs × (~15 FMA-port + 4 p5 + 1 rsqrt14 + 1 vdivpd). The
round moved zero arithmetic; it moved WHERE the map issues.

### Measured on the node (tryout.sh = leased core on a80n0, graded map
### chain 36:8:64; same-window contrasts only, MKL same case/core quoted;
### single rel_l2 = 3.586e-16 and chain m=64 rel_l2 = 1.240e-14 PASS on
### every configuration listed; chain output bit-repeatable across runs)

* **Headline same-window A/B: new 110.040 vs r4-flow (CHV0) 113.005
  µs/step** (MKL 289.3/289.4) — −2.6%. Best readings of the session:
  109.376/109.504/109.578 (MKL 288.6–289.9), i.e. **−3.4 µs vs the r4
  shape's 113.0–113.4 in the same windows**. r4 scored 111.962 in a
  283-MKL window; this shape should score ~108–110 if the window is fair.
* Window classes reproduced: identical binary read 125.9/120.7 min in two
  contended leases (MKL flat ~285–289 — contention taxes an L2-resident
  chain, not the DRAM-bound fallback; mixedradix documented the same
  mode), then 110.944 min in the next run. Min-of-quiet-windows is the
  number; medians today ranged to 126 with sd up to 9%.
* B=1 (not graded): 123.4 min in loaded windows (MKL 322); per-volume
  chaining keeps the step batch-shape-invariant, no quiet B=1 window
  appeared all session.

### What was tried and did NOT work — with the number that killed it

1. **L36_mixedradix's mB map (vsqrtpd + rcp14+2NR) ported into my staged
   map: 158.0 vs 113.3 µs/step, same window, MKL flat (289.0/288.8) — a
   +40% catastrophe for the exact arithmetic that WINS their cell.**
   Mechanism: placement, not op count. Their map is fused into the
   z-subloop's loads, where ~433 µops/call of FFT port work hide the
   sqrt's divider occupancy; my r4 map was a STAGED tight loop whose
   whole per-pair port cost is ~7.5 cyc, so 5832 vsqrtpd/vol (zmm
   throughput ~2–4× vdivpd's on this part) became the serial bottleneck —
   +45 µs ≈ the sqrt-vs-divide occupancy gap. Kept compiled under
   `-DFFT36PF_MAPSQ`. THE LESSON FOR EVERYONE: divider-unit ops are only
   cheap where other work covers them; when adopting a rival's map
   arithmetic, adopt (or reproduce) its PLACEMENT first. (Corollary
   measured the same session: after the row-group interleave landed, the
   ladder+vdivpd shape now has FFT work covering the divide, so the r4
   MAPRCP/divider question is closed in vdivpd's favor for free.)
2. **Chain pass B without its one-line-ahead read prefetch
   (`-DFFT36PF_BNOPF`): 115.530 vs 113.442, same window.** The 11664
   prefetch µops/step are load-bearing even with S L2-resident: 36
   concurrent 20736-B-stride streams are exactly what the L1 DCU
   prefetcher cannot track, and OOO alone does not cover 36 L2-latency
   loads per group. ice_r3's "every prefetch is a tax on cache-resident
   chains" does NOT extend to this one; it is the exception, now priced.
3. **Cross-plane group-0 lookahead (map plane x+1's first group during
   plane x's subloop B, legal since pass B runs after all planes' pass A):
   123.2 vs 109.1, same window (MKL 283.7/283.4), +13% for a ~2 µs
   theoretical prize.** Leading explanation: maprows is always_inline, so
   the branch + ~500-instruction map body lands INSIDE subloop B's hot
   group loop and pushes its PFA36 body past the DSB, taxing all 9
   iterations (the same code-size cliff mode 11 was built against in
   panel_r9), possibly compounded by nzp/ncp live across both subloops
   pressuring the register allocator. Kept under `-DFFT36PF_XP`; the
   default build compiles none of it.
4. **t0 c-cursor one group ahead inside maprows (`-DFFT36PF_CPF`):
   109.376 vs 109.578 — a wash** (and slightly negative MKL-normalized:
   0.3790 vs 0.3780). The interleave already gives c's demand loads a
   full FFT call of slack. Off by default, knob kept.
5. tryout.sh remains broken exactly as r4 documented: `W=$PWD/build/
   tryout/<name> ./tryout.sh ...` and run check.py by hand (which also
   recovers the silently-skipped repeatability cmp — do it, it is the
   only bit-identity check that runs at all in map mode).

### Borrowed this round, named

* The interleave-map-with-FFT-port-work PRINCIPLE: **L36_mixedradix
  ice_r4** (their fused-at-z-subloop-loads placement), reimplemented at
  row-group chunk granularity to fit my broadcast pass A without register
  fusion. Their mB ARITHMETIC was tried verbatim and rejected here with
  the mechanism above — both halves of that sentence are the borrow.
* The window-class diagnosis (L2-resident chain +40% under lease
  contention while MKL sits flat): **L36_mixedradix ice_r4**'s
  observation, reproduced and leaned on all session.

### Predictions for the scoring window (so they can be scored)

* Description reads `fchain pw=4 volres inplace lazymap2-rgi
  rsqrt14+2NR+vdiv`. Chain fingerprints unchanged from r4: single
  3.586e-16, chain m=64 1.240e-14, bit-repeatable.
* **B=8 graded cell: 107–112 µs/step** in a quiet window (best dev
  109.4–109.5 at MKL 288–290; r4's 112.9-quiet scored 111.962). If the
  score lands ≥115, it was a contended window — ask for the min across
  processes before reading anything into the code.
* If mixedradix stands still (~111.4), this takes the cell by ~2%.

### Next

1. **passA is still the frontier.** The step is ~109.5 vs a ~48 µs
   two-pipe FFT floor + ~19 µs map issue + ~16 µs pass B. A quiet-window
   SKIPB/NOPA phase split of the NEW shape (never taken — every window
   this session was too noisy to subtract reliably) should come first
   next round; the r4 splits no longer describe the code.
2. The subloop-B store order into S ([ky][kz] rows) vs pass B's
   [stride-NPLANE] reads is the remaining known-unpriced junction; pfa's
   pind-style mod-4096 probe of pp/S relative phase under the CHAIN shape
   has never been run on this file.
3. If someone wants the mB arithmetic here anyway, the road is register
   fusion into BCLD sites with the map SPLIT across both subloops to
   halve live pressure (pfa's sp=1 does divider splitting for the same
   reason) — but the r4 bcstm spill number (182) says do not go there
   without a new register budget.
4. Do NOT retry: vsqrtpd in any staged/tight map loop (158.0), removing
   pass B's read prefetch (115.5), inlined map work inside subloop B's
   group loop (123.2), c prefetch in maprows (wash), plus everything on
   the r3/r4 lists (XV staging, NTA on c or S, eager map at pass-B
   stores, whole-plane SPF/CPFIN cursors).

## Round ice_r6

### Where the round started

Scored ice_r5: **108.631 µs/step, 2nd** at the graded cell (36:8:64 map
chain), 4.7 µs behind L36_mixedradix's 103.888 (33.1% spread on their
number; L36_pfa 115.437, MKL 283.2). Round-start re-measure of the r5 code:
110.262 min, sd 0.09%, MKL 288.5 — a quiet session; nearly every contrast
below is MKL-flat (288–290), so same-window minima are directly comparable.
This round was explicitly cumulative, and the decisive input was reading
L36_mixedradix's code and record: their ice_r6 was ALREADY in their file
(nF3, divider:ladder 2:1, 4/4 windows) when I started.

### The four-way phase split that framed everything (new protocol pieces:
### -DFFT36PF_NOMAP = pass A with no map/staging, -DFFT36PF_MAPNOP = staging
### movement with the ladder/divide replaced by one add; both wrong-answer
### diagnostics, both under SKIPB[+NOC]; all one quiet window, MKL 288–290)

| configuration | µs/step | increment = |
|---|---|---|
| pass A alone (NOMAP) | 52.8 | pass A: ~19 above its ~33 port model |
| + staging movement (NOC+MAPNOP) | 66.0 | **13.1 = the r5 mr round trip, pure overhead** |
| + map arithmetic (NOC) | 84.8 | 18.9 ≈ the map's issue floor — already optimal |
| + c bytes (SKIPB) | 90.8 | 6.0 compulsory L3 stream |
| + pass B (full) | 110.26 | 19.4 vs 16.1 floor |

The 13.1 µs staging round trip was the round's target.

### What ships

1. **EAGER map fused at pass B's store sites** (`passB_mape`): a 2-deep
   deferred-pair rotation at PFA36's 36 stores maps (z+c) while z is still
   in registers; S holds MAPPED state between steps; pass A reads it
   directly (the s==0 path every step, no maprows/mr/mp); the last step's
   pass B writes final_out directly (mapvol gone from the default path).
   ADOPTED from **L36_mixedradix ice_r5's nF "new protocol"** — whose record
   credits this file's r4 eager-on-strided-c post-mortem (143 vs 113) for
   the enabling trick: **cperm**, a per-volume copy of c permuted into
   pass-B store order (`cpfill`, sequential read / full-line 64-B scatter
   writes, once per volume per chain, amortized /64 and measured free:
   SKIPB with cpfill 51.65 vs NOMAP without 52.8). The map2 pairing changes
   but every per-point value is lanewise, so eager-with-style-D bits are
   IDENTICAL to r5 (verified: chain rel_l2 1.240e-14 exactly).
2. **2:1 divider:ladder map hybrid** (ADOPTED from **L36_mixedradix
   ice_r6's nF3** and their diagnosis verbatim: the map-carrying pass is
   UOP-COUNT-bound, so trade FMA-port ladder ops for divider ops until the
   divider nears saturation; their measured budget 12 sqrts/call rides
   free, 18 loses). map2 grew a compile-time style: B = vsqrtpd + rcp14+2NR
   (~11 FMA-port ops/pair, divider), D = rsqrt14+2NR + vdivpd (~16, the
   r4/r5 shape), A = both ladders (~21, divider-free). The 18 pairs/call
   cycle B,B,D — 12 sqrts + 6 divs ≈ 264 divider cyc/call. Same-window
   race: all-D 110.24, BBA 109.18/109.19, **BBD 107.95/107.92**, 5:1
   B:D 109.39 (divider saturates, exactly their nE lesson). Note the
   asymmetry against their nF3 (their third style is A): MY D beats A in
   the third slot — with only 6 non-B pairs the divider still has room, so
   vdivpd's 5-op saving over the rcp ladder wins; their nFD result (D no
   better than A) was at 1:1 where the divider is fuller.
3. **map2 compress/expand on IMMEDIATE-controlled shuffles** (vshufpd imm /
   vpermpd imm instead of two-source __builtin_shuffle → vpermt2pd): the
   index VECTORS were registers pinned across the fused body. Frame
   1096 → 968 B; ~neutral alone (the spills are u[36]-inherent), kept for
   the register hygiene. Lanewise → bit-neutral.
4. **Chain arena on 2 MB pages** (MAP_HUGETLB attempt, THP-madvise real
   path — verified AnonHugePages 4096 kB on the node; posix_memalign last
   resort; munmap in destroy). Idea from **L36_mixedradix ice_r4** /
   **L64_blocked**. Measured ~neutral here (dTLB theory: pass B touches
   ~72 4K pages/group vs the 64-entry L1 dTLB — the STLB apparently covers
   it), kept because it also fixes the arena's page-phase lottery.
5. New knobs: FFT36PF_LAZYRGI (r5 flow), FFT36PF_ESTASH / FFT36PF_MBBA /
   FFT36PF_M51 / FFT36PF_MALLD (rejected twins), FFT36PF_PPOFF (pp/S
   phase), FFT36PF_CQOFF (cperm phase, default +256 B = maximally distant
   from S's 512-B plane-phase grid), FFT36PF_NOMAP / FFT36PF_MAPNOP
   (diagnostics). BNOPF now also gates passB_mape's read prefetch.

### Operation count

FFT unchanged (PFA 4×9 n1_9, 232 FMA-port + 57 p5 per 36-line over PW
lanes, 225,504 FMA-port vector ops/volume at PW=4). Map per volume: 5832
pairs — per 36-output call, 12 style-B pairs (~11 FMA-port + 4 p5 + 1
vsqrtpd) + 6 style-D (~16 FMA-port + 4 p5 + 1 vdivpd) ≈ 228 FMA-port ops
(was 270 all-D, −13.6k ops/step) + ~264 divider cyc/call, concurrent with
the FFT's port work. Staging movement deleted: −13.1 µs measured, of which
the rotation gives back ~9 µs as register-pressure stalls in the fused
store phase (the carrier runs ~500 cyc/call vs ~333 p05 floor; frame ~1 KB
of u[36] spills) — net eager win ~2.3 µs, + ~1.3 µs from the 2:1 ratio.

### Measured on the node (tryout.sh = leased core on a80n0, graded map
### chain 36:8:64; W= workaround + check.py by hand still required exactly
### as r4/r5 documented; single rel_l2 = 3.586e-16 every run; chain m=64
### rel_l2 = 1.191e-14 (BBD bits) vs tol 6.4e-12; chain output
### bit-repeatable across processes, verified by explicit double-run cmp)

* **Headline, quiet windows (MKL 288.1–288.6): 107.945 / 107.922 µs/step
  min, sd 0.07–0.20%**, vs the r5 shape's 110.262 same session — −2.1%.
  Contended-window final check: 109.567 min at MKL 313.6.
* B=1: 121.585 (MKL 311.6, loaded window; not graded). B=32: **107.496**
  (MKL 333.0), chain 1.416e-14 — per-volume chaining stays batch-invariant.
  Setup 0.78–1.04 s at B=8.
* Every intermediate step is in the technique section above with its
  same-window MKL.

### What was tried and did NOT work — with the number that killed it

1. **Group-stash eager** (`passB_mste`, kept under -DFFT36PF_ESTASH): FFT
   stores raw to a 2304-B L1 stash, tight pair-map loop does the strided
   stores. **121.9 vs 110.2** (MKL flat 288.9). With r5's 13.1 µs mr
   staging and this, the lesson is now measured twice: ANY memory round
   trip for the map costs ~13–14 µs regardless of buffer size or layout;
   only the register rotation avoids it.
2. **The r5-lazy hybrid borrow** (all-rcp MAPRCP in the rgi shape): 113.02
   vs 110.26 — removing ALL divider work bought nothing, i.e. the divider
   was already fully hidden in the lazy interleave; vdivpd zmm is ~8 cyc
   rtp here, not the 16+ I had assumed. Saved building the staged hybrid.
3. **PFA36X2 stage-interleaved pass B** (-DFFT36PF_PAIRB under LAZYRGI),
   the every-round-since-r1 pairability idea, finally priced: 112.15 vs
   110.26. The doubled live set's spills eat the dual-pipe win. Retire it.
4. **Hugepage arena as a SPEEDUP**: ~neutral (110.7 vs 110.2 window-shifted;
   THP confirmed materialized). The dTLB-thrash theory of the carrier's
   stalls is dead; kept for page-phase control only.
5. **Carrier without its read prefetch** (BNOPF): 110.64 vs 107.95. Still
   load-bearing even though the pass is uop-bound (36 streams beat the L1
   DCU prefetcher, r5's finding survives the fusion).
6. **pp/S mod-4096 phase race** (PPOFF; the r5 next-list junction, now
   priced): default phase 1024 = 107.95, phases {512,1536,2624,3136,3584,
   64} all 107.9–109.4 — a wide plateau — but **phase 2112 = 137.5
   (+27%)**: a violent alias hole between subloop-B's S stores and pp
   broadcast reloads, one page-phase wide. Default 0 ships; if the arena
   layout ever changes, re-run this race first.
7. 5:1 sqrt:div ratio: 109.39 — the divider budget really is ~12 sqrts per
   ~500-cyc call, exactly as mixedradix's nE/nF3 numbers say.
8. Immediate-shuffle map2 alone (before the hybrid): 111.7 at MKL 289.9 ≈
   wash normalized — index-register pressure was not the spill driver;
   u[36] is.
9. BBD pattern permutations (post-record addendum, knobs kept): D-first
   (-DFFT36PF_MDF) 108.63 at MKL 284.8 = 0.3814 normalized vs interleaved
   0.3740; B-first (-DFFT36PF_MBF) landed a contended window (109.04 at
   MKL 321.6), not better. Interleaved B,B,D ships; next-list item 2 is
   answered — the divider wants steady feeding, not front-loading.

### Borrowed this round, named

* Eager map at the final pass's store sites, the 2-deep deferred-pair
  rotation, cperm + cpfill, and last-step-writes-final_out:
  **L36_mixedradix ice_r5 (nF)** — a mutual borrow, since their record
  credits this file's r4 strided-c post-mortem for cperm.
* The uop-count-bound carrier diagnosis, the divider-ratio lever, and the
  12-sqrt budget: **L36_mixedradix ice_r6 (nF3)**, read from their file
  mid-round as this cumulative format intends. Their nFD note also warned
  me off 1:1 with style D; the 2:1 D-third refinement is this file's own.
* Hugepage arena: **L36_mixedradix ice_r4**, ultimately **L64_blocked**.
* The four-way split protocol extends my own r5 SKIPB/NOC machinery.

### Predictions for the scoring window (so they can be scored)

* Description reads `fchain: volres inplace EAGER map@passB-stores cperm
  hyb12B:6D`. Fingerprints: single 3.586e-16 (B=8), chain m=64
  **1.191e-14**, bit-repeatable.
* **B=8 graded cell: 105–110 µs/step** in a quiet window (best dev 107.92
  at MKL 288.1). ≥114 means a contended window — ask for the cross-process
  min before reading anything into the code.
* L36_mixedradix ships nF3 (their file already says so): expect them at
  ~100–102 and this entry 2nd again, gap narrowed from 4.7 to ~4–6 µs.
  If they somehow stood still at nF, this is a photo finish.

### Next

1. **Pass A is now the biggest block: 51.65 µs vs ~41 realistic model**
   (and vs mixedradix's 48.9 for the same two axes). Unpriced ideas: the
   4-deep dependent merge-broadcast chains in BCLD (a maskz+vorpd tree
   halves the depth for +36 p05 uops/call — probably a wash, measure not
   argue); their pind-style discipline applied INSIDE pass A (subloop-A
   pp-stores vs S-row loads — note the PPOFF race above only moves pp
   globally, both subloops at once).
2. The carrier's remaining ~170 cyc/call over floor is u[36]-spill
   latency in the store phase. Ideas spent: stash (121.9), per-vector map
   (r4: issue-bound), style shuffling (done). What's left is surgical:
   force the 6 D-pairs to the FIRST SB_ blocks — DONE this round (dead
   end 9): the interleaved pattern wins; the surgical road left is
   splitting SB_'s map bursts across smaller register windows, which
   likely needs hand-scheduling beyond what gcc's allocator honors.
3. Do NOT retry: any staged/stash map placement (two independent 13-14 µs
   measurements), PAIRB, 5:1+ divider ratios, prefetch removal, NTA/t0/t1
   cursors anywhere on chain streams (r3/r4/r5 lists), hugepages-for-speed.
4. tryout.sh line 36 still expands $W before defining it; keep the
   `W=$PWD/build/tryout/<name>` prefix and run check.py by hand (which is
   also the only chain-repeatability check that exists in map mode).

## Round ice_r7

### Where the round started

Scored ice_r6: **106.908 µs/step, 3rd** at the graded cell (36:8:64 map
chain; L36_mixedradix 100.801, L36_pfa 106.249, MKL 284.3).  Two context
facts framed the round: (1) the new `results/rivals_icelake/` table shows
the best GATE-PASSING rival at L=36 on our own node is 1760b1bf at 0.0587 s
= 114.7 µs/step — every rival is behind this entry; the target is
mixedradix.  (2) Their r6 record decomposes their 100.5 quiet step as
z=31.7% y=15.9% p2=51.9%, i.e. phase 1 ≈ 47.6 µs vs my pass A 51.65 and
their y-subloop at its EXACT 144.5-cyc p05 floor.  Round-start re-measure
of the r6 code: 108.1–108.6 min, sd 0.02–0.06%, MKL 286.8–289.8 — every
contrast below is same-window, MKL-flat, and the session was unusually
quiet (sd ≤0.5% on nearly every run).

Bookkeeping note found this round: the r6 leaderboard description string
still read `lazymap2-rgi` because fft3d_create()'s snprintf was never
updated when r6 shipped the eager flow — fixed; the string now names the
actual chain shape, so the monitor can read the pick again.

### The round's bet, and how it died: FUSED-BOUNDARY CUSTODY

ADOPTED from **L64_radix8 ice_r6's ckind=2** (which took the L=64 cell
after L64_blocked's record declared the fusion impossible): the three axis
transforms of one FFT commute, so the chain ... M·X·Y·Z · M·X·Y·Z ...
regroups into alternating SINGLE sweeps, each completing step k and
beginning step k+1:

    P: [Z_k Y_k · M_k · Z_{k+1} Y_{k+1}]   plane-local (map at subloop-B
       stores into the 20.25 KB mp, c NATURAL — subloop B's store index IS
       the row-major c plane offset; then the plain broadcast pass A
       re-reads mp for Z_{k+1} Y_{k+1})
    F: [X_k · M_k · X_{k+1}]               strided (map via the r6 rotation
       into a register array w[36], indices constant-folded; FFT#2 consumes
       w directly — no memory round trip, honoring the r6 ESTASH lesson)

Schedule: sweep 0 = plain pass A on x0; k=1..m-1 alternate F (odd) / P
(even); final half sweep = passB_mape (m odd) or a planePhalf_map into
final_out (m even; the graded m=64 ends P-type).  Same arithmetic (6 axis
stages + 2 maps per 2 steps), ONE S read+write per step instead of two.
Both gates PASS (single 3.586e-16; chain m=64 rel_l2 1.291e-14 — bits
differ from r6 by legal axis reassociation), bit-repeatable.

**REJECTED by the node: 111.5–111.8 vs the r6 flow 108.2–108.5 µs/step,
2/2 alternations, MKL 286.8 (+3.0%).**  noinline + `#pragma GCC unroll 1`
fences on the fused bodies (the mode-11 cure) changed nothing (111.9/115.1
vs 108.6/112.2 in a noisier window).  The SKF/SKP phase splits priced the
sweeps: F ≈ 79 µs/sweep (707 cyc/call vs a ~440 port floor), P ≈ 146
µs/sweep, so custody = 225 µs/2-steps vs r6's 215.7.  The accounting
mistake in the bet, written down so nobody repeats it: **at this
L2-resident cell the r6 two-sweep flow is already load/store-minimal** —
per 2 steps r6 does 4 S-reads + 4 S-writes, custody does 3 + 3 (the mp
round trip replaces one of them 1:1) — so the fusion deletes only ~one
746 KB L2 round trip per 2 steps (~4 µs) while adding ~18 µs/2-steps of
fused-body register/ROB pressure (u[36]+w[36] live in F; the slB map
rotation in P costs more than the same rotation in the carrier, where the
36 plain strided loads pressure the OOO window less than 144 broadcast
merges).  L64's custody win lives on an 8 MB/volume L3-scale state where a
deleted sweep is real bandwidth; it does NOT transfer to a 746 KB
L2-resident state.  Kept compiled under `-DFFT36PF_CUSTODY` (diagnostics
`-DFFT36PF_SKF`/`-DFFT36PF_SKP`); the default chain is the r6 flow.

### Also priced and rejected, with the numbers that killed them

1. **Split merge-broadcast builder (`-DFFT36PF_BCOR`)** — r6 next-list
   item 1: the 4-deep dependent vbroadcastf64x2 merge chain rebuilt as two
   2-deep maskz halves + one vorpd (+36 p05/call, half the depth).
   **109.1–109.9 vs 108.1–108.6, 3/3 alternations, MKL 288–290** — the
   merge depth was NOT the pass-A binder; the vorpd is a pure p05 tax.
   All six builder sites now share one BCB4 macro (default = 4-deep).
   CAUTIONARY TALE, free to the panel: the FIRST version of this knob
   "won" by 1.5 µs — with a mask bug (the first half broadcast r0 into
   all 8 lanes and only merged r1, so the OR corrupted lanes 4–7).  The
   tells were a changed setup time (0.50 vs 0.78 s — the exec tuner's
   admission gate silently rejecting every corrupted bcst candidate) and
   `cmp` on the chain outputs.  A/B twins that should be bit-identical
   MUST be cmp'd before their timing is believed.
2. **Transposed-pp pass-A hybrid (`-DFFT36PF_TPP`)** — grafting
   mixedradix's y-at-floor mechanism: subloop A stages its 36 outputs
   (Wv), 4x4-transposes them (72 p5/call) and stores pp[y][kz]; subloop B
   then feeds PFA36 with 36 PLAIN 64-B loads, zero broadcasts.
   **112.0–112.3 vs 107.9–108.9, 3/3, MKL 288** — the Wv[36] staging
   spills (the exact thing the r2 bcst design deleted) cost ~4 µs, more
   than the 108 deleted broadcast loads/call pay back.  Structural fact
   derived on the way (write it down, it kills a family of ideas): PFA's
   index scatter makes the staging UNAVOIDABLE for any transpose-fed
   PFA36 — OX(k1,k2) mod 4 == k1 and IX(n1,n2) mod 4 == n1, so every
   consecutive-k transpose block needs one output from each of the four
   SB_ calls and every consecutive-j block feeds four different SA_
   calls.  Mixedradix pays this staging inside their ~300-cyc z-calls;
   grafting only their cheap half is not possible.
3. Custody map-style variants (all-B in the F sweep on its doubled
   divider budget) were built as knobs (`-DFFT36PF_MALLB`) but not raced
   — custody lost by 3.3 µs before style tuning could matter.

### What ships

The ice_r6 eager flow, bit-identical output and fingerprints (single
3.586e-16, chain m=64 1.191e-14), through the refactored shared BCB4
builder (4-deep default — instruction-identical to r6).  Plus: the fixed
description string, and the three knob-gated rejected twins for future
rounds.  End-state same-window numbers on the node: **B=8 107.9–108.9
min (MKL 288–290); B=32 108.786 (chain 1.416e-14); B=1 107.8–121.9
window-dependent, twins bit-identical (chain 1.197e-14)**; B=8 chain
output identical across runs (explicit cmp).  Setup 0.78 s (B=8).

### Operation count

Default path unchanged from ice_r6 (PFA 4×9 n1_9: 232 FMA-port + 57 p5
per 36-line over PW lanes; 225,504 FMA-port vector ops + 55,404 swaps per
volume at PW=4; map 12 style-B + 6 style-D pairs per carrier call).  The
custody twin moves zero arithmetic (same 6 stages + 2 maps per 2 steps);
BCOR adds 36 vorpd/call; TPP trades 108 broadcast loads/call for 72 p5
shuffles + a 36-vd staging array.  All three rejected on measurement, not
on count — the counts said custody and TPP should win or tie.

### Borrowed this round, named

* Fused-boundary custody (one sweep per step, axis-role alternation):
  **L64_radix8 ice_r6 (ckind=2)** — mechanism verbatim, translated to
  4×9 PFA sweeps; rejected here by the L2-residency arithmetic above.
* The phase-1 decomposition target (y at p05 floor) that motivated TPP:
  **L36_mixedradix ice_r6's** TSC splits, read from their file as this
  format intends.
* The noinline/unroll-fence code-size cure tried on the F sweep:
  **this file's panel_r9 mode 11** machinery.

### Predictions for the scoring window (so they can be scored)

* Description reads `fchain pw=4 volres inplace EAGER map@passB cperm
  hyb12B:6D r7[cu111.5 or109.1 tpp112.0 vs 108.1]` — the r7 bracket is
  this round's rejected-twin scoreboard riding the string for the
  monitor.  Fingerprints are the r6 ones: single 3.586e-16, chain m=64
  **1.191e-14**, bit-repeatable — if the scored chain fingerprint reads
  1.291e-14 instead, a `-DFFT36PF_CUSTODY` flag leaked into the scored
  build; flag it.
* **B=8 graded cell: 106–110 µs/step** in a quiet window (this session's
  quiet band 107.9–108.9 at MKL 288–290; r6 scored 106.908 from the same
  code in a 284-MKL window).  Any move ≥4 µs either way is window, not
  code — the shipped bits are r6's.
* mixedradix ships from ~100.5; expect 2nd again, gap ~6 µs.  The honest
  read after this round: their remaining edge is phase-1 shape (their z
  absorbs the transpose staging better than my broadcasts absorb their
  rebuild), and closing it needs their w-file kernel form, not a knob.

### Next

1. **The three cheap structural doors at this cell are now all measured
   shut** (custody +3.0%, BCOR +1%, TPP +3.7%, on top of r6's stash/
   PAIRB/prefetch/ratio dead ends).  What remains is hand-shaping the
   two hot bodies below gcc's allocator: (a) the carrier's ~170 cyc/call
   over floor (u[36] spill placement in the store phase), (b) pass A's
   ~86 cyc/call over floor.  Both are asm-level or
   scheduling-pragma-level work; nothing else on the idea list survives
   a count.
2. If a future round has PMU time (perf_event_open works on this node),
   read UOPS_DISPATCHED.PORT_5 and DSB coverage on the carrier FIRST —
   the r6/r7 overhead attribution (ROB/retire vs ports) is still
   inference, and two rounds of ideas died on it.
3. The custody machinery is correct, gate-passing, and knob-live; if the
   graded cell ever moves to an L3-scale state (bigger L, bigger B, or a
   cache-hostile c), re-race `-DFFT36PF_CUSTODY` before anything else —
   the arithmetic that kills it here is exactly what reverses there.
4. Do NOT retry: everything on the r3–r6 lists, plus one-sweep custody
   at L2-resident cells, broadcast-depth splitting, and transpose-fed
   PFA36 without accepting the staging array (the mod-4 scatter proof
   above).

## Round ice_r8

### Where the round started

Scored ice_r7: **107.009 µs/step, 3rd** at the graded cell (36:8:64 map
chain; L36_mixedradix 99.809, L36_pfa 100.296, MKL 283.3).  Round-start
re-measure of the r7 code: 106.665 min, sd 0.05%, MKL 281.4 — a quiet
window; every headline contrast below is same-window, MKL-flat 281–286.
The round context's decisive input: **L36_pfa's ice_r7 record**, which
took them from 106.2 to 100.3 with two mechanisms measured on this exact
cell and the same PFA 4×9 codelet — the one-tile-lagged map (their mix=7)
and the "phase 1 un-broadcast" (TRNC, their own finding, worth 5.8 µs).
The warm cohort offered nothing at L=36 (their best cell is 0.0530 s =
103.5 µs/step, behind our leaders; mixedradix's r7 record already mined
and closed the rival sources for this size).  New tryout breakage this
round: check.py dies at line 94 (`math` never imported) in the m>2
map-check branch, on top of the r4–r7 $W/squeue/out2.bin issues — run the
chain gate by hand (inline numpy replicating the 300×-anchor arithmetic),
and the two-run cmp by hand as before.

### What ships: TRNC pass A in the chain (the "phase 1 un-broadcast")

ADOPTED from **L36_pfa ice_r7 (P1ZT+P1YT)**, translated onto my pass
shapes: `passA_trnc` replaces the broadcast pass A in the default chain
path.  Both subloops build their 36 lane-transposed input vectors with 36
full-width 64-B loads + 9 register-transpose quads (my existing
CTRANSPOSE butterfly, 8 shuffles each = 72 p5/call) into a `vd[36]`
staging array (`Zt`/`Yv`), instead of the r2 BCB4 builder's 144 4-deep
merge-mask broadcasts per call.  Subloop A: rows y=yb..yb+3 × columns
zq*4..+3 loaded full-width, transposed in registers, PFA36 consumes
Zt[j]; stores to pp unchanged.  Subloop B: same on pp rows (kz × y).
Mechanism (pfa's r7 diagnosis, confirmed here): the r2 port arithmetic
(broadcasts keep port 5 free for the second FMA pipe) was measured on
shapes whose pass A carried real port pressure; the eager chain's pass A
is map-free and STALL-bound — the 4-deep merge-mask dependence chains
cost more than the 72 shuffles displace.  Why this is NOT my r7 TPP
rejection (112.0): TPP transposed the OUTPUT side, so its Wv[36] staging
was live ACROSS the codelet on top of u[36]; here Zt/Yv die as u[36]
fills — input-side staging replaces the loads' register demand instead of
adding to the stores'.  The exec path keeps BCB4 (not scored;
`-DFFT36PF_BCSTA` rebuilds the r7 broadcast chain pass A for A/B).
Values are bit-identical to the broadcast builder's (shuffles are exact),
so ALL fingerprints carry over from r6/r7 unchanged.

### Operation count

FFT and map arithmetic unchanged (PFA 4×9 n1_9: 232 FMA-port + 57 p5 per
36-line over PW lanes; 225,504 FMA-port vector ops + 55,404 codelet swaps
per volume at PW=4; map 12 style-B + 6 style-D pairs per carrier call).
Pass A per call: 144 broadcast-merge load µops → 36 full loads + 72 p5
shuffles + ~36 st/36 ld of L1 staging traffic; per volume that is +46,656
p5 shuffles (the exact count the r2 bcst design deleted — and it is STILL
the right trade, because the binder moved from ports to stalls when the
map left pass A in r6).  Two-pipe port floor rises ~8k cyc/vol; measured
time fell 4.4 µs/step: this round was stalls, not ports, exactly as pfa
wrote.

### Measured on the node (tryout.sh with the W= workaround + PATH shim;
### chain gate and two-run cmp by hand — check.py m>2 branch is broken
### this round; graded map chain 36:8:64 unless stated)

* **Headline: 102.245 min / 102.397 median, sd 0.07%, MKL 282.7** vs the
  r7 code's 106.665 same session (MKL 281.4) — **−4.2%**.  Confirmations
  of the shipping bits: 103.380 (MKL 285.5), 102.631 (MKL 285.6; median
  136.9 — a contended lease, min-of-windows per the r5 rule).  One
  contended-class reading in between: 116.9 at MKL flat 282.6 (the known
  +13%-on-us/flat-MKL mode; code identical to the 102.6 run).
* **Phase split (SKIPB, pass A alone): 46.833 vs r7's 51.65** — the whole
  end-to-end win localizes in pass A, as designed.  Carrier by
  subtraction ≈ 55.4 µs, unchanged, now the clear frontier again.
* B=1 (not graded): 115.456, sd 0.04%, MKL 309.8 — slow-clock window,
  ratio 0.373, consistent with r7's window-dependent 107.8–121.9 band.
* **Correctness, all by hand on the node**: single 3.586e-16 (B=8) /
  3.591e-16 (B=1); **two-step precision gate (m=2): 1.549e-15 vs tol
  3e-14** (the new scored gate — ~19× margin); chain m=64 rel_l2
  1.191e-14 vs anchor-based tol 1e-10 (anchor 1.227e-14); chain output
  **bit-identical to the r7 broadcast flow** (cmp of .chain files across
  the two builds) and identical across two fresh runs of the shipping
  binary.  Setup 0.83–0.97 s.

### What was tried and did NOT work — with the number that killed it

1. **One-group-LAGGED carrier (`-DFFT36PF_MAPL`, kept compiled)** — pfa's
   mix=7 / mixedradix's style L on my carrier: FFT stores raw to the
   2304-B stash, the next group's call flushes the previous stash at its
   top (adjacent-pair order, stash+cq both sequential), tail flush ends
   the plane; flush styles B,B,A (pfa's lagged choice; `-DFFT36PF_MLBBD`
   builds B,B,D).  **110.921 vs 102.245 same window, MKL 282.8/282.7 —
   +8.5%, REJECTED.**  This settles the panel's split evidence for my
   body: it matches mixedradix's nL verdict (−8) and not pfa's (+2.8),
   and lands within 1.5 µs of my r6 immediate-stash number (121.9 vs
   110.2 — the same cyclic instruction stream cut at the other loop
   boundary, as it should).  Conclusion for L=36 carriers: the lag only
   pays where the rotation was NOT already fused into the store phase
   (pfa's rotation-era carrier ran ~2.8 µs worse than mine relative to
   floor); with the r6 rotation in place there is nothing for the lag to
   recover and the stash round trip is pure cost.  Do not retry; the
   style-pattern variant cannot recover 8.7 µs.
2. The B,B,D flush variant was built but not raced (lost cause after
   item 1's margin).
3. Not re-raced on judgment: the pp/S mod-4096 phase (PPOFF) — the r6
   race found a one-phase-wide alias hole at 2112 with the BROADCAST
   subloop-B reloads; TRNC changes pass A's read pattern (full-width
   loads), so the hole may have moved, but the default phase measures
   healthy (102.2 at quiet) and the plateau was 6-wide in r6.  If a
   future round changes the arena layout, re-run the PPOFF race with the
   TRNC build first.

### Borrowed this round, named

* **TRNC pass A (the whole shipped win): L36_pfa ice_r7's "phase 1
  un-broadcast" (P1ZT+P1YT)** — mechanism and diagnosis verbatim, ported
  onto my plane-fused pass shapes with my own CTRANSPOSE butterfly
  (their VSH-based TRNC and my CTRSTEP compile to the same 8-shuffle
  quad).  Their record's step ladder (105.8 → 101.7 → 100.0) predicted
  my −4.4 µs almost exactly.
* The lagged-carrier candidate: **L36_pfa ice_r7 (mix=7)** and
  **L36_mixedradix ice_r7 (style L)** — rejected here with the number
  above; my measurement resolves their split verdict in mixedradix's
  favor for rotation-fused carriers.
* The min-of-windows discipline and window classing: this file's r5/r6
  machinery, unchanged.

### Predictions for the scoring window (so they can be scored)

* Description reads `fchain pw=4 volres inplace TRNC-passA(r8) EAGER
  map@passB cperm hyb12B:6D r8[mapl110.9 vs 102.2]`.  Fingerprints
  UNCHANGED from r6/r7 (bit-identical chain): single 3.586e-16, chain
  m=64 1.191e-14, two-step 1.549e-15, bit-repeatable across processes.
* **B=8 graded cell: 101–105 µs/step** in a quiet window (dev minima
  102.2–103.4 at MKL 281–286; r7's 107.9-quiet scored 107.009).  ≥110
  means a contended window — ask for the cross-process min first.
* vs rivals: mixedradix ~100–102 (they queued no big lever in their r7
  record), pfa ~100–102.  This entry lands in the same band for the
  first time since r5 — expect a three-way photo finish decided by the
  window, not the code.

### Next

1. **The carrier is the frontier again and it is the LAST fat block:
   ~55.4 µs by subtraction vs a ~41 µs model (16.1 FFT floor + ~19 map
   issue + ~6 c stream).**  Every placement idea is now priced (rotation
   r6, stash r6, lag r8, per-vector r4, staged r5); what remains is
   hand-shaping the store-phase register pressure below gcc's allocator
   (u[36] spill placement) — asm-level work, or the PMU attribution
   first (perf_event_open works on this node; UOPS_DISPATCHED.PORT_5 and
   DSB coverage on the carrier would settle ROB-vs-ports in one run).
2. Pass A at 46.8 vs a ~34 port-model: the residual is now the same
   dependent-latency class as pfa's post-TRNC p1 (145k cyc vs 94k
   floor); their next-list points at memory-side effects (mid-store RFO
   from L2).  A pp-store RFO experiment (prefetchw on pp lines once per
   plane, NOT per group) is the one un-priced idea left there — but
   every pass-A prefetch since r3 has been a tax; price it only with the
   PMU first.
3. Split-complex through the whole pipeline: pfa's r7 next-list note
   stands — with TRNC's 72 shuffles/call now in pass A, the boundary
   deint/reint arithmetic from my r3 rejection has changed; re-do that
   count before dismissing it a third time.
4. Do NOT retry: the lagged carrier (110.9), anything on the r3–r7
   lists.  tryout.sh: W= prefix + PATH squeue shim still required, and
   check.py's m>2 map-check now needs the by-hand numpy replica (missing
   `import math`).
