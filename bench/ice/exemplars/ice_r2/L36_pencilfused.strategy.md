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
