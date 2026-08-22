# VERDICT — round panel_r3

Monitor's report. Measured on `p55n3`, Intel Xeon Gold 5218 (Cascade Lake, 2.30 GHz base,
AVX-512 with one 512-bit FMA unit, 1 MB L2/core, 22 MB L3), gcc 11.4, `powersave`,
slurm job 438477, 2026-08-21T19:47. Three runs per (backend, L, batch); every number
below is the min-over-runs per-transform time, the same statistic `leaderboard.py` ranks on.
Setup/plan time is excluded from the score, as always.

Eleven implementations were submitted, eleven built, eleven ran, eleven passed correctness
in all four of their batch cells. `build_errors.txt` is empty and there is no
`failures.txt`. Nothing is missing.

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

"Best library" is the fastest of MKL 2022, MKL 2026, FFTW ×3 and ducc0 in that cell.
`baseline_matrix` (the harness floor) is excluded.

### L = 6 (216 points/volume)

| case | fastest panel entry | best library | speedup |
|---|---|---|---|
| **non-batched** | **L6_pfa 0.220 µs** (38.09 GF/s) — L6_unrolled 0.220, a dead heat | mkl_dfti 0.370 µs | **1.68×** |
| B = 64 | **L6_unrolled 0.214 µs** (39.09 GF/s) | mkl_dfti 0.392 µs | **1.83×** |
| B = 4096 | **L6_unrolled 0.392 µs** (21.34 GF/s) | mkl_dfti 0.557 µs | **1.42×** |
| B = 32768 | **L6_unrolled 0.563 µs** (14.87 GF/s) | mkl_dfti 0.706 µs | **1.25×** |

The panel owns every L=6 cell and has since round 1.

### L = 8 (512 points/volume)

| case | fastest panel entry | best library | speedup |
|---|---|---|---|
| **non-batched** | **L8_batchsimd 0.570 µs** (40.45 GF/s) | mkl_dfti 0.652 µs | **1.14×** |
| B = 64 | **L8_fusedaxes 0.642 µs** (35.90 GF/s) | mkl_dfti 0.710 µs | **1.11×** |
| B = 2048 | **L8_radix8 1.243 µs** (18.54 GF/s) | mkl_dfti 1.335 µs | **1.07×** |
| B = 16384 | **L8_fusedaxes 1.580 µs** (14.58 GF/s) | mkl2026_dfti 1.777 µs | **1.12×** |

Three different entries hold the four cells. All three beat both MKLs everywhere.
Caveat on the B=1 baseline: `mkl_dfti` posted a 61.9 % run spread in that cell; its min
(0.652) agrees with r1 (0.651) and r2 (0.651), so the ratio stands, but the cell is noisy
for the library, not for us.

### L = 17 (4913 points/volume)

| case | fastest panel entry | best library | speedup |
|---|---|---|---|
| **non-batched** | **L17_matrixsimd 16.386 µs** (18.38 GF/s) | fftw3_patient 81.723 µs | **4.99×** |
| B = 8 | **L17_matrixsimd 17.930 µs** (16.80 GF/s) | fftw3_estimate 81.921 µs | **4.57×** |
| B = 256 | **L17_matrixsimd 21.444 µs** (14.05 GF/s) | fftw3_patient 83.515 µs | **3.89×** |
| B = 2048 | **L17_matrixsimd 22.697 µs** (13.27 GF/s) | fftw3_measure 84.000 µs | **3.70×** |

A clean sweep for `L17_matrixsimd`, which had lost both batched cells in r2. This is the
largest margin over the state of the art anywhere on the board: MKL needs ~100 µs here.

### L = 36 (46656 points/volume)

| case | fastest panel entry | best library | speedup |
|---|---|---|---|
| **non-batched** | **L36_mixedradix 118.626 µs** (30.50 GF/s) | mkl_dfti 162.419 µs | **1.37×** |
| B = 4 | **L36_pencilfused 127.304 µs** (28.42 GF/s) | mkl_dfti 175.342 µs | **1.38×** |
| B = 32 | **L36_pfa 218.351 µs** (16.57 GF/s) | mkl_dfti 220.506 µs | **1.01×** |
| B = 256 | **L36_pfa 227.497 µs** (15.90 GF/s) | mkl_dfti 247.568 µs | **1.09×** |

The three L=36 entries are within 1.2 % of each other at B=1 and B=4 — those orderings are
inside the run spread and should not be read as wins. B=32 is the weakest cell on the whole
board: a 1.01× margin over MKL, and a regression (see §2).

---

## 2. What changed since panel_r2, per geometry

The cross-round table below is best-panel-entry vs best-library in each cell, so a change in
which entry leads does not hide a change in the cell.

| cell | r1 best | r2 best | r3 best | Δ r2→r3 | r3 vs MKL |
|---|---|---|---|---|---|
| L6 B=1 | 0.219 | 0.218 | 0.220 | +0.9 % | 1.68× |
| L6 B=64 | 0.219 | 0.214 | 0.214 | 0 % | 1.83× |
| L6 B=4096 | 0.392 | 0.384 | 0.392 | +2.1 % | 1.42× |
| L6 B=32768 | 0.631 | 0.572 | **0.563** | −1.6 % | 1.25× |
| L8 B=1 | 0.570 | 0.573 | **0.570** | −0.5 % | 1.14× |
| L8 B=64 | 0.625 | 0.636 | 0.642 | +0.9 % | 1.11× |
| L8 B=2048 | 1.432 | 1.205 | 1.243 | **+3.2 %** | 1.07× |
| L8 B=16384 | 1.782 | 1.557 | 1.580 | **+1.5 %** | 1.12× |
| L17 B=1 | 16.880 | 16.751 | **16.386** | −2.2 % | 4.99× |
| L17 B=8 | 20.532 | 18.661 | **17.930** | −3.9 % | 4.57× |
| L17 B=256 | 25.984 | 24.031 | **21.444** | −10.8 % | 3.89× |
| L17 B=2048 | 27.863 | 24.603 | **22.697** | −7.7 % | 3.70× |
| L36 B=1 | 118.441 | 119.266 | 118.626 | −0.5 % | 1.37× |
| L36 B=4 | 128.353 | 128.460 | **127.304** | −0.9 % | 1.38× |
| L36 B=32 | 204.655 | 202.746 | 218.351 | **+7.7 %** | 1.01× |
| L36 B=256 | 247.435 | 238.796 | **227.497** | −4.7 % | 1.09× |

**L = 6 — flat; the geometry is finished.** Every cell moved by less than the run spread
except B=32768, which improved 1.6 %. `L6_unrolled`'s headline feature (write-intent
`prefetchw` on the next volume's output) *was* selected by the node tuner
(`variant=fused_pfw` at B=4096 and B=32768) and bought 1.6 % at B=32768 and −2 % at B=4096,
against the 1.41× it measured on its dev machine. `L6_pfa` regressed 3.4 % at B=32768
(0.616 → 0.637) while correctly predicting its own pick (v5 = 3-pass + prefetch T0,
normal stores). NT stores were rejected by the node tuner in every L=6 cell for the third
consecutive round, by both entries independently.

The reason L=6 is finished: at B=32768 the compulsory traffic is 6912 B/volume, so 0.563 µs
is **12.3 GB/s of compulsory read+write** (18.4 GB/s counting the write-allocate RFO that
normal stores pay). That is the fastest single-core stream anything on this board has
demonstrated on this node. Nothing further is available at large batch without cutting
traffic, and the traffic-cutting move (NT stores) is the one the node keeps rejecting.

**L = 8 — the cell leadership rotated and two batched cells regressed.**
`L8_batchsimd` restructured LANEX from three passes to two (deleting a full L1 scratch round
trip). It won B=1 back — 0.598 → 0.570, first place, and it predicted 0.57–0.59 — but it
**lost all three batched cells it had held in r2**: B=64 0.636 → 0.663 (+4.2 %),
B=2048 1.205 → 1.283 (+6.5 %), B=16384 1.557 → 1.748 (+12.3 %). Its own predictions for
those cells were ≤1.20 and 1.50–1.56. It shipped the new structure as a replacement rather
than as an additional tuner candidate, so the node had no way to choose the old one.

Meanwhile `L8_radix8` went the *opposite* structural direction — 2-pass → 3-pass, adding
an L1 round trip to make the output stores sequential — and improved B=2048 1.526 → 1.243
(−18.5 %, taking the cell) and B=16384 1.778 → 1.647 (−7.4 %). `L8_fusedaxes` fixed a real
r2 bug (`pf = nt`, which prefetched an L3-resident input) and improved B=2048 1.503 → 1.291
(−14 %) and B=16384 1.614 → 1.580, taking that cell. Both hit every prediction they made.

Net: the panel still beats MKL in all four cells, but B=2048 and B=16384 are worse than r2's
numbers. That is a genuine regression and it is attributable to one entry's structural swap.

**L = 17 — the round's clear win.** `L17_matrixsimd` improved every cell and swept the
geometry, gaining 10.8 % at B=256 and 7.7 % at B=2048 from two changes: pinned sine constants
(−64 loads/chunk) and X-first pass reordering (spreading output writes across the volume's
compute instead of bursting them at the end). Both were adopted from other entries' records
with attribution; the X-first reorder is the one that moved the batched cells.
`L17_winograd` was essentially flat (18.247 → 18.169, 24.031 → 23.905, 24.603 → 24.470,
i.e. ≤0.5 % everywhere) — see §4, its fused-pass change did not transfer.
`L17_rader` **regressed at batch**: B=256 24.394 → 26.205 (+7.4 %), B=2048 26.635 → 27.114
(+1.8 %), while improving B=1 19.212 → 18.491. It is now third in all four cells. Its
description strings show why it is the suspect: it is the only L=17 entry whose tuner
selected `pf=1` (cross-volume input prefetch) at B=256, where both rivals' tuners chose
`pf=0`.

**L = 36 — the round went backwards at B=32 and nothing was learned from it.**
All three entries built the same thing this round: a cross-volume prefetch pipeline aimed
at the r2 verdict's 1.9× memory-overlap prize. On the node, **two of the three tuners
rejected their own headline feature**, readably:
`L36_mixedradix` picked `v1-nt-pf1` (not `…-xv`) at B=32 and B=256, and
`L36_pencilfused` picked `mode=scratch+nt` (not `nt+xv`) at both. `L36_pfa` cannot be
checked — see §3.

Outcomes: `L36_pencilfused` was the round's biggest improver (B=4 151.5 → 127.3, −16 %;
B=32 241.4 → 221.2, −8.4 %; B=256 283.9 → 236.8, −16.6 %) and went from last in every cell
to first at B=4 and second at B=32/B=256. `L36_mixedradix` was flat-to-worse
(B=32 231.4 → 233.4, B=256 261.5 → 264.5). `L36_pfa` improved B=256 by 4.7 % against a
predicted 150–175 µs, and **regressed 7.7 % at B=32** (202.7 → 218.4) against a predicted
145–175 µs.

The B=32 regression is real and not a library artifact. MKL's L=36 batched numbers swung
+18 %/+25 % in r2 and returned to their r1 values in r3 (B=32: 220.6 → 259.8 → 220.5;
B=256: 246.3 → 307.8 → 247.6) on a fixed binary, so r2's *ratios* at L=36 were inflated.
But the panel's own B=32 numbers were stable across r1 and r2 (204.7, 202.7) and are now
218.4. Measured against the stable r1/r3 MKL baseline, the panel's L=36 B=32 margin has gone
**1.08× (r1) → 1.01× (r3)**. That cell is the one place the panel is at risk of losing to a
library outright.

---

## 3. Correctness, builds, crashes, omissions — the adversarial pass

**Nothing failed and nothing is missing.** The evidence:

* `build_errors.txt` is present and empty; no `failures.txt` was written.
* `agents/exits.txt` records `exit=0` for all eleven implementers.
* `impl/` holds eleven implementations plus `baseline_matrix.c`; `strategies/` holds eleven
  records; `prompts/` holds eleven prompts. Every implementation appears in all four cells
  of its geometry, with three timed runs each (verified from the raw `t_*.json`).
* `check.log` contains **156 PASS and zero FAIL** — exactly the expected count
  (L=6: 9 backends × 4 cells; L=8/17/36: 10 backends × 4 cells each). `check.py` compares
  the full batched output volume against `numpy.fft.fftn` at tolerance 1e-12; measured
  `rel_l2` ranged 1.3e-16 … 8.4e-16 (the 8.4e-16 is `baseline_matrix`, the O(L⁴) floor).
* `timing.err` contains only "does not support L=36"-class lines from geometry-specialised
  entries being offered the wrong L — expected, and the driver records those as
  `supported: false` rather than as failures. No signals, no timeouts, no aborts in
  `timing.log`.

**No fast wrong answer survived, and none is hiding.** Beyond the numpy check, I verified
that no winner is faster than its own arithmetic can be — the check a no-op, a cached
output, or a silently-skipped axis would fail:

| geometry | winner | vector FP ops/volume (implementer's own count) | port floor @2.30 GHz | measured B=1 | ratio |
|---|---|---|---|---|---|
| L=6 | L6_unrolled | 972 (2 × 256-bit FMA ports) | 0.211 µs | 0.220 | 1.04× |
| L=8 | L8_batchsimd | 1248 (1 × 512-bit FMA unit) | 0.543 µs | 0.570 | 1.05× |
| L=17 | L17_matrixsimd | 35 964 | 15.6 µs | 16.386 | 1.05× |
| L=36 | L36_mixedradix | 241 056 | 105 µs | 118.6 | 1.13× |

Every entry sits *above* its floor by a plausible 4–13 %. Nothing is returning an answer it
did not compute. (The floors assume base clock; if the node turbos above 2.30 GHz under
AVX-512 the true floors are lower and these margins are upper bounds — which is exactly the
unresolved measurement in §6.)

**One reportability failure, and it matters.** `L36_pfa` is the only entry of the eleven
that does not report its tuner's pick through `fft3d_description()`. Its string is the same
in every cell: *"…x in place or via reused scratch + NT stores, variant autotuned in
create()"*. Its own round-3 record asks the monitor for exactly this readout ("Ask the
monitor: which (pw, mode, pf) the node tuner chose per batch"). It is also the entry that
regressed 7.7 % at B=32 with a prefetch-only round, and the leading hypothesis — that its
12-candidate tournament selected `pf=1` at B=32, where `in` is 23.9 MB against a 22 MB L3
and `L36_mixedradix` independently measured cross-volume prefetch losing 12 % in that
marginal regime — **cannot be tested from this round's artifacts**. Every other L=36 entry
plumbed the pick as the r2 verdict asked. This is not a correctness failure, but it cost the
round its diagnosis of the worst regression on the board.

**Tuner instability is now costing measurable time.** The three runs per cell do not always
select the same candidate:

* `L8_radix8` at B=64 picked `avx512-3p` in run 1 (0.671 µs) and `avx512-3p-pf` in runs 2
  and 3 (0.716, 0.715) — a 6.7 % penalty from a coin-flip at plan time. The leaderboard
  reports the min, so it shows 0.671, but two runs in three shipped the slower plan.
* `L36_mixedradix` at B=1 picked `pf4`/`pf0`/`pf1` across the three runs (123.2 / 118.6 /
  119.7 µs) — a 3.9 % spread caused by the tuner, not the machine.
* `L36_mixedradix` at B=32/B=256 also flipped `pf0`/`pf1` between runs.

Reporting the min over runs flatters entries whose tuners are unstable. Next round should
report the *median* pick's time alongside the min, or entries should widen their hysteresis.

---

## 4. Claimed numbers vs measured — and how much is the machine

Implementers develop on `wallaby` (Xeon Gold 6448Y, Sapphire Rapids: 2 FMA units, full-clock
AVX-512, 2 MB L2/core, 60 MB L3, shared and frequently contended). They are scored here on
Cascade Lake with one FMA unit, licence-downclocked AVX-512, 1 MB L2 and 22 MB L3. MKL alone
spans ~2.9× between the two, so absolute claims are not comparable and I do not hold them
against anyone. The measured absolute ratio this round was **1.8–2.4× for compute-bound
cells and up to 2.9× for the DRAM-bound L=8 cells** — e.g. `L17_matrixsimd` 8.88 → 16.386 µs
(1.85×), `L36_pfa` 104.4 → 227.5 µs (2.18×), `L8_fusedaxes` 0.441 → 1.291 µs (2.93×). All
of that is the machine.

What is *not* explained by the machine, and is the substantive finding of this round, is
that the **deltas** — the improvement each change bought — largely failed to transfer:

| entry | change | dev-machine gain | node gain |
|---|---|---|---|
| L6_unrolled | `prefetchw` on next volume's output | **1.41×** at 113 MiB | **1.6 %** (B=32768), −2 % (B=4096) |
| L17_winograd | fuse passes 2+3, delete the 157 KB L2 round trip | **−6.2 %** (B=1), −6.3 % (B=256) | **−0.4 %** / **−0.5 %** |
| L36_mixedradix | cross-volume prefetch ("xv") | **−13 … −23 %** (B=256) | **rejected by the node's own tuner** |
| L36_pencilfused | cross-volume pipeline (XV) | **−8.7 %** (B=256) | **not selected** (plain NT won) |
| L36_pfa | paced input prefetch + cross-volume pre-coverage | **−24 %** (B=256) | −4.7 % (B=256), **+7.7 % worse** (B=32) |
| L8_batchsimd | 3-pass → 2-pass LANEX2 | −4 … −7 % batched | **+4 … +12 % worse** batched |
| L8_radix8 | 2-pass → 3-pass, sequential stores | +25 % vs rival at B=5632 | **−18.5 %** at B=2048 ✓ |
| L8_fusedaxes | fix `pf = nt` bug; tune store policy on node | predicted ≤1.35 / 1.55–1.62 | **1.291 / 1.580** ✓ |

Attribution: the four badly-missed predictions are all **memory-system** changes tuned
against a 2 MB L2 / 60 MB L3 / fat-DRAM part and scored on a 1 MB L2 / 22 MB L3 part. On
wallaby a 256-volume L=36 working set (40 MB) is L3-resident; on the node it streams. A
prefetch scheme whose whole value is hiding DRAM latency behind compute is exactly the class
of change that inverts across that boundary, and three entries said so in advance in their
own records. So: **machine difference, as expected, and predicted by the implementers
themselves** — with one exception. `L8_batchsimd`'s batched regression is *not* a machine
effect in the same sense: it deleted the structure that was winning and shipped only the
replacement, so the node's tuner could not fall back. Its record even argues, correctly,
that a fused structure can measure worse on a 2-FMA part and still be right for a 1-FMA
part — but that argument justifies *keeping both candidates*, not replacing one with the
other.

The two entries whose predictions landed (`L8_fusedaxes`, `L8_radix8`) are precisely the two
that shipped both structures as tuner candidates and predicted a range rather than a point.
That is the round's transferable lesson, and it is a process lesson, not an algorithmic one:
**on this hardware pair, add candidates; do not replace structures.**

Two prediction scorecards worth recording in full, because they were stated to be scored:

* `L17_winograd` predicted the node would pick `f4` (its 256-bit fused variant) in the
  batched cells and that B=1 would land ~17.1 µs. **The pick prediction was right** —
  `var=f4, pf=0` at B=256 and B=2048 — and the time prediction was wrong: B=1 landed 18.169
  with the node choosing `var=a8`, i.e. it declined the fused variant entirely at B=1.
* `L36_pencilfused` predicted `pw=4` everywhere, INPLACE at B=1/B=4, NT+XV at B=32/B=256.
  Three of four right: `pw=4 mode=inplace` at B=1 and B=4, `pw=4 mode=scratch+nt` — **no
  XV** — at B=32 and B=256.

Finally, plan time is growing: 1.0–1.6 s at the largest batches for four entries. It is not
scored, which creates a standing incentive to over-tune. It has not been abused yet — the
worst case is ~32× a single execute call and these are plan-once/run-many designs — but it
should be watched.

---

## 5. Which open question from LITERATURE.md §4 moved

### §4.3 — "Is axis fusion worth 3× or 3%?" — moved decisively, and the answer is 3 %.

§07 gap 7 names this "the single most important number this project will have to measure for
itself," and notes there is no published CPU study comparing fused against multi-pass
L1-resident work for small cubes. This round produced the controlled experiment, twice, on
the same node in the same job:

**At L = 8, a matched pair in opposite directions.** `L8_batchsimd` fused three passes into
two, deleting one full L1 scratch round trip (128 loads + 128 stores per volume) with the
arithmetic byte-identical: it gained 4.7 % in the compute-bound cell (B=1: 0.598 → 0.570)
and *lost* 4–12 % in the batched cells. `L8_radix8` did the reverse on an isomorphic kernel
— split two passes into three, *adding* 128 loads + 128 stores to an L1-resident scratch, in
order to make the output store stream sequential — and gained 18.5 % at B=2048 and 7.4 % at
B=16384. Same node, same round, same op count, opposite structural moves, and the pass-count
change is worth a few percent while the *store order* change is worth 18 %.

**At L = 17, the same answer from the other side.** `L17_winograd` fused passes 2+3 and
deleted a 157 KB per-volume L2 round trip — the literal experiment §4.3 asks for — for
−6.2 % on Sapphire Rapids and **−0.4 to −0.5 % on the node**. It had reasoned that a smaller,
slower L2 should make the deleted round trip worth *more*; the measurement says the opposite.
The node's tuner then declined the fused variant altogether at B=1.

**Verdict on §4.3:** for L1/L2-resident intermediates in the 6³–36³ range on this class of
CPU, fusing passes is worth **single-digit percent, and can be negative**. TurboFNO's 3–5 %
prior is the right one; §07 §2.1's Hong–Kung "strictly worse by a factor equal to the number
of passes" does not describe this regime, because the traffic being deleted never leaves the
cache hierarchy. Tolmachev's rule from §07 §1.6 — *payoff = passes avoided × the bandwidth
gap between the two levels involved* — survives intact and now has CPU numbers attached: at
an L1↔L2 gap the payoff is ~5 %, at an L2↔L3 gap ~0.5 %, and what actually pays at these
sizes is the *order* traffic is issued in (sequential vs strided stores, write-spreading),
not its volume. `L17_matrixsimd`'s X-first reorder — which changed no arithmetic and no
traffic volume, only *when* the writes happen — bought 10.8 %, more than every fusion
attempt on the board combined.

### §4.6 — "Model versus search" — reinforced, with a new cost attached.

Search won again, and by a wide margin: the node's picks contradicted the dev machine's in
almost every batched cell (`f4` vs `f8`, plain-stores vs NT, `pf=0` vs `pf=1`, xv rejected
twice). §06's position — that the schedule is "the primary thing to search" at every one of
our sizes, since none of 6, 8, 17, 36 is a power of 4 — is now supported by three rounds of
node data. **The new datum is that search has a price:** unstable tuners cost 3.9–6.7 % in
the cells listed in §3, and reporting min-over-runs conceals it.

### Not moved, and one flag away

**§4.5 (does L = 8 need padding, and where)** was not settled. `L8_radix8` padded its
scratch plane stride 128 → 144 doubles on the set-conflict argument, found the dev-machine
A/B inconclusive, shipped it anyway, and explicitly asked for one node A/B
(`-DL8R_SCRX=128` at B=2048/16384). That A/B did not happen. It is one compile flag on the
machine where the 32 KB L1 argument actually applies, and it would close a corpus
disagreement outright. Do it next round.

---

## 6. The single highest-value thing the next round should attack, per geometry

The ranking is by *quantified distance to a bound*, using the node's own demonstrated
streaming rate of ~12 GB/s single-core (compulsory read+write per volume ÷ measured time)
and each entry's own FP-port floor.

**L = 6 — stop optimising it; measure the clock, then redeploy.**
B=32768 is at 12.3 GB/s compulsory, the fastest stream on the board — there is nothing left
without cutting traffic, and NT stores (the only traffic cut available) have now been
rejected by the node's tuners three rounds running. B=1 is 1.04× its own FP-port floor at
base clock. So the highest-value L=6 action is not an implementation change at all: it is the
**`perf stat -e cycles,ref-cycles` on an L=6 B=1 run** that four entries across two
geometries have now asked for in writing. It decides whether B=1 has ≤4 % of headroom or
~40 % (0.220 µs is 506 cycles at 2.30 GHz but ~750 at single-core turbo). If it shows no
headroom, L=6 is done at 1.25–1.83× MKL and one of the two L=6 implementers should be moved
to L=36, where 1.83× is sitting unclaimed.

**L = 8 — recover the batched regression by making both structures tuner-selectable.**
`L8_batchsimd` must ship its r2 three-pass LANEX *and* its r3 two-pass LANEX2 as candidates
and let the node choose per batch size. On this round's evidence that recovers ~1.205 µs at
B=2048 and ~1.557 at B=16384 while keeping the 0.570 it won at B=1 — the single largest
guaranteed gain available anywhere on the board for the effort, because both numbers have
already been measured on this node. Bundle the free `-DL8R_SCRX=128` A/B at B=2048/16384 to
close §4.5 in the same job. For context on how much room is left afterwards: at B=16384 the
compulsory traffic is 16 KB/volume ≈ 1.365 µs at 12 GB/s against a measured 1.580, so L=8
large-batch is already within 16 % of the bandwidth bound and this cell is nearly closed.

**L = 17 — close the batched overlap gap; the arithmetic is finished.**
Compulsory traffic is 157 KB/volume ≈ 13.1 µs at 12 GB/s, and the compute floor is 16.4 µs,
so at B=2048 memory should hide entirely under compute and the ceiling is **~16.4 µs against
a measured 22.697** — 6.3 µs/volume, 1.39×, of un-overlapped memory time. This is now the
second-largest quantified gap on the board and it is `L17_matrixsimd`'s own named next step:
software-pipeline across volumes, interleaving volume b+1's X pass into volume b's plane
phase, which stays inside its verified bit-equivalence class D. Three rounds of evidence say
no further op-count work will pay here (r2: −11.9 % FP ops → −0.8 % time), so no one should
spend another round on the kernel.

**L = 36 — instrument before optimising, and fix the B=32 regression.**
Compulsory traffic is 1.49 MB/volume ≈ 124 µs at 12 GB/s against a compute floor of
~119 µs, so the ceiling at B=256 is **~124 µs against a measured 227.5** — **1.83×, still
the largest unclaimed prize on the board**, essentially unchanged from r2's 1.9×. All three
entries spent this round throwing prefetch variants at it and collectively bought 4.7 % in
one cell and lost 7.7 % in another. That approach is exhausted as a blind search. What is
missing is a measurement of *where* the 103 µs/volume goes: `L36_pencilfused` already ships
`-DFFT36PF_SKIPA/SKIPB` phase-split builds and `L36_pfa` ships `FFT36_FORCE_PF`,
`FFT36_NT` and `FFT36_XV` overrides, so a single control job of forced-variant runs at
B=32 and B=256 would produce the node's own pf=0/pf=1 and NT=0/NT=1 deltas — the numbers
every entry has been guessing at from wallaby. Preconditions, both mandatory:
**(a) `L36_pfa` must report its tuner pick in `fft3d_description()`** like every other entry
— its regression is currently undiagnosable; **(b) the forced-variant control run at B=32
must be scheduled**, since B=32 is where the panel's margin over MKL has fallen to 1.01×.
Only after that is the two-volume ping-pong pipeline (costed and deferred by `L36_pfa` on
L2-residency grounds) worth building.

---

## 7. What to keep

Against `docs/CURATION.md`:

**Promoted:**

1. **`L6_unrolled`** — *criterion 1.* Fastest at L=6 in three of four cells and tied at the
   fourth; holds the board's fastest demonstrated single-core stream (12.3 GB/s at
   B=32768). Its record also carries the round's cleanest transfer failure with numbers
   (`prefetchw`: 1.41× on SPR, 1.6 % here), which is why the code goes with it.
2. **`L8_batchsimd`** — *criteria 1 and 3.* Fastest at L=8 B=1 (0.570 µs), and the round's
   most instructive failure: a fusion that won the compute cell and lost all three batched
   cells because it was shipped as a replacement rather than a candidate. Both numbers are
   measured and documented in its record.
3. **`L8_fusedaxes`** — *criterion 1.* Fastest at B=64 and B=16384; the only entry on the
   board that hit every prediction it made, and the entry whose `pf = nt` bug fix produced
   the node's plain-vs-NT store-policy crossover.
4. **`L8_radix8`** — *criteria 1 and 2.* Fastest at B=2048, and structurally the opposite of
   the other two (three passes, sequential output stores). It is half of the §4.3 evidence
   and the next panel needs to read it beside `L8_batchsimd`, not in prose.
5. **`L17_matrixsimd`** — *criterion 1.* Swept all four L=17 cells; 3.70–4.99× the best
   library, the largest margin on the board. The X-first write-spreading reorder is the
   single most valuable technique the round produced.
6. **`L17_winograd`** — *criterion 2.* Structurally different runner-up at 1.08–1.13×, well
   inside the ~20 % bar, and it carries the fused-pass f4/f8 implementation that answers
   §4.3 at L=17.
7. **`L36_pfa`** — *criterion 1.* Fastest at B=32 and B=256, within 1.2 % at the other two.
   Promoted with the round note that it regressed 7.7 % at B=32 and that it must plumb its
   tuner pick before the next round.
8. **`L36_pencilfused`** — *criterion 2.* Fastest at B=4, second at B=32/B=256, the round's
   largest improvement (−16.6 % at B=256), and structurally distinct: mode-keyed pass-A
   variants (z-first transpose-on-load when streaming, y-first when cached) that neither
   rival has.

**Not promoted, with reasons:**

* **`L6_pfa`** — near-duplicate of `L6_unrolled` (same PFA 2×3 arithmetic). Its 0.0003 µs
  lead at B=1 is inside a 1.6–3.2 % run spread and is not a win, and it regressed 3.4 % at
  B=32768. Its distinct contribution this round (the T1 prefetch column, the race-arena cap
  raise) is already in its strategy record, which is tracked regardless.
* **`L17_rader`** — third in all four cells, regressed 7.4 % at B=256, and no longer
  structurally distinct: it now runs `L17_winograd`'s kernel by its own attribution, so it
  is a layout variant of an already-promoted entry rather than the independent Rader
  alternative §4.2(a) wanted. It was promoted in panel_r1 and dropped in panel_r2; nothing
  this round argues for bringing it back. Its record — that Rader with a shared kernel loses
  13–22 % at L=17 on this hardware — stands as the answer to §4.2(a) and stays tracked.
* **`L36_mixedradix`** — same PFA 4×9 two-sweep structure as `L36_pfa`; its 1.2 % lead at
  B=1 is inside the run spread, and both its batched cells regressed while its headline
  feature was rejected by the node's own tuner. Its two genuinely reusable findings (the
  machine-relative 2.5×L3 tuner arena, and the measured cost of cross-volume prefetch when
  the input is L3-resident) live in its strategy record, which is enough.

PROMOTE: L6_unrolled L8_batchsimd L8_fusedaxes L8_radix8 L17_matrixsimd L17_winograd L36_pfa L36_pencilfused
