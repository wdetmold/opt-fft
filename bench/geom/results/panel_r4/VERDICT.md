# VERDICT — panel round `panel_r4`

Monitor's judgement on the measurements taken on `p55n3` (Xeon Gold 5218, Cascade Lake,
2×16c, exclusive, slurm job 438480, 2026-08-21T20:51), gcc 11.4.0, `-O3 -march=native`.
Roster: 11 implementations, all built, all ran, all correct. Sources for this round are in
`impl/` (see the provenance note at the end).

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are per transform, minimum across three independent processes, as reported by
`leaderboard.txt`. "Best library" is whichever of FFTW ×3 / MKL 2022 / MKL 2026 / ducc0
was fastest in that exact cell.

### L = 6 (volume 216)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L6_unrolled 0.219 µs** (L6_pfa 0.219, dead heat) | mkl_dfti 0.369 µs | **1.69×** |
| B=64 (0.42 MiB) | **L6_unrolled 0.214 µs** | mkl_dfti 0.392 µs | **1.83×** |
| B=4096 (27 MiB) | **L6_pfa 0.387 µs** | mkl_dfti 0.557 µs | **1.44×** |
| B=32768 (216 MiB) | **L6_unrolled 0.566 µs** | mkl_dfti 0.707 µs | **1.25×** |

### L = 8 (volume 512)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L8_batchsimd 0.570 µs = L8_radix8 0.570 µs** | mkl_dfti 0.654 µs | **1.15×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.623 µs** | mkl_dfti 0.712 µs | **1.14×** |
| B=2048 (32 MiB) | **L8_radix8 1.136 µs** | mkl2026_dfti 1.324 µs | **1.17×** |
| B=16384 (256 MiB) | **L8_radix8 1.418 µs** | mkl2026_dfti 1.804 µs | **1.27×** |

### L = 17 (volume 4913)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 16.431 µs** | fftw3_estimate 81.705 µs | **4.97×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 18.008 µs** | fftw3_patient 81.957 µs | **4.55×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.626 µs** | fftw3_estimate 83.497 µs | **3.86×** |
| B=2048 (307 MiB) | **L17_matrixsimd 22.290 µs** | fftw3_patient 84.054 µs | **3.77×** |

L=17 remains the board's largest margin, and it is the one geometry where the libraries
are all clustered (FFTW 81.7–88.6, MKL 98.8–102.1, ducc0 103.9–113.0) — i.e. nobody's
prime-size path is special, and ours is 3.8–5.0× all of them.

### L = 36 (volume 46656)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L36_mixedradix 119.021 µs** | mkl_dfti 162.171 µs | **1.36×** |
| B=4 (5.70 MiB) | **L36_mixedradix 129.921 µs** | mkl_dfti 175.427 µs | **1.35×** |
| B=32 (45.6 MiB) | **L36_pfa 174.226 µs** | mkl_dfti 221.910 µs | **1.27×** |
| B=256 (364 MiB) | **L36_pfa 218.899 µs** | mkl_dfti 248.867 µs | **1.14×** |

**Every one of the 11 panel entries beat every library baseline in its own geometry in
every scored cell this round.** The narrowest such margin on the whole board is
L36_mixedradix at B=32: 221.602 µs against mkl_dfti's 221.910 µs — 0.14%, a statistical
tie, and the recovery of the one cell where a panel entry was behind a library in
`panel_r3` (mixedradix 233.434 vs mkl 220.506).

Consequence for curation: promotion ground 4 ("anything that beat a library baseline")
selects the entire roster this round and therefore discriminates nothing. It is recorded
here as a fact about the round rather than used as a promotion criterion.

---

## 2. What changed since `panel_r3`

### Cell-level best (fastest correct entry in the cell, r3 → r4)

| L | B | r3 best | r4 best | Δ |
|---|---|---|---|---|
| 6 | 1 | 0.220 | 0.219 | −0.5% |
| 6 | 64 | 0.214 | 0.214 | flat |
| 6 | 4096 | 0.392 | 0.387 | −1.3% |
| 6 | 32768 | 0.563 | 0.566 | +0.5% (inside 1.6% spread) |
| 8 | 1 | 0.570 | 0.570 | flat |
| 8 | 64 | 0.642 | 0.623 | **−3.0%** |
| 8 | 2048 | 1.243 | 1.136 | **−8.6%** |
| 8 | 16384 | 1.580 | 1.418 | **−10.3%** |
| 17 | 1 | 16.386 | 16.431 | +0.3% (inside 0.9% spread) |
| 17 | 8 | 17.930 | 18.008 | +0.4% |
| 17 | 256 | 21.444 | 21.626 | +0.8% |
| 17 | 2048 | 22.697 | 22.290 | −1.8% |
| 36 | 1 | 118.626 | 119.021 | +0.3% |
| 36 | 4 | **127.304** | **129.921** | **+2.1% — the round's only cell-level regression** |
| 36 | 32 | 218.351 | 174.226 | **−20.2%** |
| 36 | 256 | 227.497 | 218.899 | −3.8% |

**L = 6 — stalled, as intended.** Both entries deliberately ran narrow rounds. L6_pfa
closed its 13% B=32768 deficit (0.637 → 0.570) by adopting L6_unrolled's write-intent
prefetch, and took B=4096 (0.398 → 0.387). L6_unrolled is flat in all four cells: its
headline change, the split-z-store kernel shapes (`_s`), was **never selected by the node
tuner in any of 12 invocations** despite winning 12% on the development machine. Nothing
regressed beyond run spread.

**L = 8 — the round's cleanest technical win.** L8_radix8 adopted L8_fusedaxes' spread
prefetch (128 lines of volume b+1 issued 6/5/5 per pass iteration instead of 16 in a
pass-1 burst) and took both streaming cells: B=2048 1.243 → 1.136 (−8.6%) and B=16384
1.647 → 1.418 (−13.9%). Both are the best numbers ever recorded in those cells, beating
even L8_batchsimd's `panel_r2` marks (1.205 / 1.557). The node picked `avx512-3p-pfs`,
i.e. **plain stores plus spread prefetch — the NT variant lost again.** L8_fusedaxes
improved B=64 (0.642 → 0.623) and regressed slightly at B=2048 (1.291 → 1.313, +1.7%,
inside its 2.7% spread) after its tuner selected the new `seq3` shape; L8_batchsimd is
flat-to-better everywhere but see §4.

**L = 17 — frozen at the top, moving underneath.** L17_matrixsimd has now held the
geometry for three rounds with B=1 unchanged inside noise across r2/r3/r4 (16.751 /
16.386 / 16.431). Its only real move is B=2048 (−1.8%). The movement is in second place:
L17_rader improved in all four cells (B=1 18.491 → 17.742, −4.1%; B=256 26.205 → 25.202,
−3.8%; B=2048 27.114 → 25.704, −5.2%) and passed L17_winograd to take second at every
batch size, narrowing the gap to the leader from 1.13× to 1.08× at B=1. L17_winograd is
flat or marginally worse everywhere; its software-pipelined `p4`/`p8` variants — the
round's entire delta for that entry — were **not selected by the node tuner in a single
cell** (picks: `a8` at B=1, `f4` at B=8/256/2048).

**L = 36 — one large win, one regression, and a lot of rejected machinery.** L36_pfa's
B=32 cell went 218.351 → 174.226 µs (−20.2%), which is the largest single move of the
round and takes the cell from a statistical tie with MKL (1.01× in r3) to 1.27× ahead.
L36_mixedradix recovered B=256 (264.531 → 228.743, −13.5%) and B=32 (233.434 → 221.602,
−5.1%) with its paced input prefetch. L36_pencilfused went the other way in two cells:
B=256 236.824 → 242.231 (+2.3%, against a 0.7% run spread — a real regression) and B=4
127.304 → 132.360 (+4.0%, inside its 5.5% spread but enough to lose the cell it led).
L36_pfa also gave up 2.4% at B=4 (129.242 → 132.347) against a 0.5% spread.

### The regressions, named

Only two per-entry regressions this round fall outside their own cell's run-to-run spread:

* **L36_pencilfused at B=256** — 236.824 → 242.231 µs (+2.3%; spread 0.7%). Its record
  states the streaming path was extended with three new modes; the node selected
  `scratch+nt` (mode 2, the r3 shipping config), so the regression is not a mode change
  and is not explained by its own record.
* **L36_pfa at B=4** — 129.242 → 132.347 µs (+2.4%; spread 0.5%). Its record asserts the
  B=1/B=4 INPLACE path reproduces r3's address streams exactly; the pick string confirms
  the same `pw=4 mode=inplace pf=0`. The refactor of phase 1/phase 2 into per-plane
  functions is the only candidate cause and it was not re-measured at B=4 on the node.

Everything else that got slower did so by less than the cell's own three-process spread
and should not be read as a regression.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed to build.** `build_errors.txt` is present and empty (0 bytes).

**Nothing crashed, hung, or timed out.** `failures.txt` does not exist, which for
`sweep.sh` means no backend invocation exited non-zero across 16 cases × 3 runs.
`results/panel_r4/agents/exits.txt` records `exit=0` for all 11 implementers.

**Nothing is missing.** All 11 entries in `impl/` appear in the leaderboard, all 11 have
strategy records in `strategies/`, and all 11 have correctness verdicts at all four of
their scored batch sizes (44 `c_*.json` files, one per entry per cell).

**Nothing failed correctness.** Every panel entry passes at every scored batch size, with
relative L2 error against numpy between 1.4e-16 and 4.0e-16 against a 1e-12 tolerance —
i.e. four orders of magnitude inside the gate, which is where a genuinely correct
double-precision transform lands. `check.py` compares the full `(B, L, L, L)` array
element-by-element, not a sampled volume, so there is no room for an entry to be right
only where it is looked at. There is no fast wrong answer in this round.

Three things the measurement did *not* establish, which the record should say plainly:

**(a) For two cells, the number in the leaderboard was produced by a plan variant whose
output was never checked.** `sweep.sh` runs each backend three times, reports the
*minimum*, and then runs `check.py` on the output file left by the *last* run. Six cells
this round had plan-time tuners that selected different variants in different processes,
and in two of them the fastest process was not the checked one:

| cell | reported time (run) | variant timed | variant checked (run 3) |
|---|---|---|---|
| L17_matrixsimd B=1 | 16.431 µs (r1) | `512-bit, pinned sines` | `512-bit, C parked, pinned sines` |
| L8_radix8 B=64 | 0.680 µs (r1) | `avx512-2p (tuned)` | `avx512-3p (default)` |

Both implementers independently verified their candidate sets — L17_matrixsimd by `cmp`
on full outputs across its bit-equivalence classes, L8_radix8 by passing correctness on
both kernel shapes — so I do not believe either number is wrong. But the harness did not
prove it, and that is a hole worth closing: check the output of the *fastest* run, or
check every run.

**(b) The other four pick flips, and what they cost.** L17_matrixsimd B=256
(`pf=0` ↔ `pipelined`), L17_rader B=256 (`512t pin` ↔ `512t`), L6_unrolled B=4096
(`fused_pf` ↔ `fused_pfw`), L8_batchsimd B=64 (`LANEX2` ↔ `LANEX2S`), L8_fusedaxes B=2048
(`seq3` ↔ `seq3+pf_t0`). Spread across the three processes in those cells is 0.8–3.7%,
which is *not worse* than the spread in cells with stable picks (up to 6.6% at
L8_radix8 B=1, 5.0% at L6_pfa B=1). The `panel_r3` verdict measured unstable tuners
costing 3.9–6.7%; the hysteresis bands, minimum-of-N sampling and simplest-candidate
tie-breaks that six entries shipped this round have contained that. This is a solved
problem and should stop consuming implementer rounds.

**(c) Two pairs of entries are computing bit-identical arithmetic.** The correctness
JSONs are a fingerprint: `rel_l2`, `max_abs` and `rel_max` all agree to the last ULP
between —

* **L8_radix8 and L8_batchsimd, at all four batch sizes** (e.g. `rel_l2` =
  1.9155724583698214e-16 at B=16384, identical in all 17 digits over 8.4M complex
  doubles). These are not two algorithms; they are two memory schedules of one algorithm
  — the same radix-8 split-complex codelet in the same C,A,B axis order. L8_batchsimd's
  own record predicted exactly this ("the axis-order last-digit effect"). The timings do
  differ (0.665 vs 0.680 at B=64, 1.215 vs 1.136 at B=2048, 1.642 vs 1.418 at B=16384),
  so the *schedule* is what is being measured — but for curation purposes they are near
  duplicates, and only one of them should go into `exemplars/`.
* **L36_pfa and L36_pencilfused, at B=32 and B=256 only** (identical), while remaining
  distinct at B=1 and B=4. Their streaming paths have converged on the same PFA-4×9
  arithmetic order; their small-batch INPLACE paths have not.

L=6 (2.374e-16 vs 2.511e-16 at B=1) and all three L=17 entries (3.195 / 3.233 /
3.279e-16) are numerically distinct from each other — at L=17 the panel really is running
three different algorithms, which is what §4.2 of the literature asks for.

**No rule violations found.** No entry references FFTW, MKL, DFTI, ducc0 or any vendor DFT
outside of comments; no OpenMP pragma, `omp_*` call or pthread anywhere in `impl/`; and
`git diff` against HEAD shows `driver.c`, `fft3d_api.h`, `Makefile`, `sweep.sh`,
`check.py`, `leaderboard.py` and `gen_input.py` untouched.

---

## 4. Claimed numbers versus measured numbers

The development machine (`wallaby`, Xeon Gold 6448Y, Sapphire Rapids, 2 MB L2/core,
60 MB L3, two 512-bit FMA units) is not the scoring machine (Gold 5218, Cascade Lake,
1 MB L2/core, 22 MB L3, one 512-bit FMA unit). The size of that gap is calibrated by the
libraries themselves: the same MKL case at L=36 B=1 was recorded at **80.7 and 150.6
µs/vol in two runs of one wallaby session** and at **162.2 µs on the node** — a 2.0× span
in MKL alone, and the panel's standing calibration puts the full span at ~2.9×. Any
wallaby-to-node comparison therefore carries roughly a 3× uncertainty band, and the
divergences below are attributed accordingly.

**Ratios of claimed (wallaby) to measured (node), same entry, same cell:**

| entry | cell | claimed | measured | ratio |
|---|---|---|---|---|
| L6_unrolled | B=1 | 0.114 µs | 0.219 µs | 1.92× |
| L8_radix8 | B=1 / B=16384 | 0.308 / 0.575 | 0.570 / 1.418 | 1.85× / 2.47× |
| L8_batchsimd | B=1 / B=16384 | 0.305 / 0.597 | 0.570 / 1.642 | 1.87× / 2.75× |
| L8_fusedaxes | B=1 / B=16384 | 0.345 / 0.620 | 0.579 / 1.585 | 1.68× / 2.56× |
| L17_matrixsimd | B=1 / B=2048 | 9.76 / 13.45 | 16.431 / 22.290 | 1.68× / 1.66× |
| L17_winograd | B=1 / B=2048 | ~9.3 / 18.39 | 18.325 / 24.221 | 1.97× / 1.32× |
| L36_pfa | B=1 / B=256 | 51.2 / 107.6 | 121.866 / 218.899 | 2.38× / 2.03× |
| L36_mixedradix | B=1 / B=256 | 50.4 / 98.4 | 119.021 / 228.743 | 2.36× / 2.32× |
| L36_pencilfused | B=1 / B=256 | 52.7 / 97.9 | 122.478 / 242.231 | 2.32× / 2.47× |

These ratios are all inside the machine band and none of them is an implementer error.
But three specific claims deserve to be scored, because their *direction*, not their
magnitude, failed to transfer:

1. **L17_matrixsimd's headline "−22% at B=2048" is a wallaby-only result, and by
   construction.** Its record claims 17.19 → 13.45 µs/t on wallaby and attributes most of
   it honestly to an L3-scaled tuner arena that let non-temporal stores be selected. That
   arena is `min(batch, clamp(2.5·L3/157 KB, 384, 1024))`, which on the node's 22 MB L3
   evaluates to 384 — *the same value r3 used*. The record says so explicitly ("node
   behaviour is bit-for-bit identical to r3's tuner"). The node measured −1.8%. This is
   not a machine-difference surprise; it is a change that was designed not to affect the
   scoring machine, and the −22% should not be carried forward as a panel result.

2. **L36_mixedradix's paced prefetch claimed −23% on wallaby and delivered −13.5% on the
   node**, which is the healthiest claim/measure pair of the round: it predicted B=256 in
   200–230 µs and landed 228.7, and B=32 in 205–225 and landed 221.6. Attribute the
   shortfall to the machine: its mechanism is hiding DRAM read latency, and the node's
   read rate and L2 capacity are both lower.

3. **L36_pencilfused's `pipeseq` (mode 6) claimed −11.3% against its own shipping config
   on wallaby and was not selected on the node in any cell.** The node picked
   `scratch+nt` at B=32 and B=256. Its companion negative result (mode 5, sequential NT
   stores without pipelining, 168.2 vs 110.3 µs/vol) is the more valuable half and is
   discussed in §6.

**The single largest claim/measure divergence in the round is in the panel's favour and
nobody predicted it: L36_pfa's B=32 cell.** Its record predicted 203–212 µs on the
hypothesis that hysteresis would flip the pick from `pf=1` to `pf=0`. The node measured
**174.226 µs with `pw=4 mode=inplace pf=1`** — the pick did *not* flip, and the cell came
in 15% below the entry's own optimistic bound and 20% below r3. Because r3 shipped no
pick reporting, the r3 configuration for that cell is unknown, so the cause cannot be
attributed from the data we have. The two live candidates are (i) the 3% simplest-wins
hysteresis moving the mode (e.g. off a scratch/NT mode onto `inplace`), or (ii) the
per-plane refactor of phase 1/phase 2 changing code layout and scheduling. This is the
round's best unexplained result and §6 makes isolating it the L=36 priority.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

**§4.8 gap 6 — "No AVX-512 measurement anywhere in this corpus… licence-based
downclocking on Skylake-SP is severe… Measure it on the node."** This round moved it
further than any previous round, in two independent ways.

**(a) The node's sustained clock is measured, and it is 3.89 GHz, not 2.30 GHz.**
L6_unrolled shipped a core-clock probe inside `fft3d_create()` — a serially dependent
256-bit FMA chain (latency 4 on Cascade Lake), timed after the plan-time tournament has
warmed the core, ~10 ms, unscored — and formats the result into `fft3d_description()`.
Every L=6 cell in this round's leaderboard therefore carries `clk=3.89GHz`, identically
across all four batch sizes and all three processes. That is essentially the Gold 5218's
3.9 GHz maximum turbo, which is what an exclusive node with one busy core should give,
and it is *not* what the corpus's downclocking warning (§04 §8.1–8.2, Gold 5120:
2.7 → 2.3 → 1.6 GHz) had led three rounds of the panel's own cycle accounting to assume.

The consequence is immediate and it reopens a geometry the previous verdict had closed.
L6_unrolled's B=1 cell at 0.219 µs is **852 cycles at 3.89 GHz**, against its own
486-cycle two-FP-port floor: **1.75× the floor, not the 1.04× that "L=6 is finished" was
based on.** Roughly 366 cycles per volume — 43% of the runtime — are not FP-port-bound
and are currently unaccounted for. Every cycle-per-volume figure in every strategy record
computed at 2.30 GHz is off by 1.69× and should be re-derived.

It also re-frames the machine gap. Wallaby's probe reports 4.10 GHz against the node's
3.89 GHz — a 5% clock difference — yet L6_unrolled runs 0.114 µs on wallaby (467 cycles,
at or just under its floor) against 0.219 µs on the node (852 cycles). **At L=6 the
~1.9× machine gap is almost entirely per-cycle throughput, not clock.** Item 4's standard
attribution to "the machine difference" is correct but should stop being read as a clock
difference; at this geometry it is microarchitecture, and it is therefore something the
node data can actually be reasoned about.

**(b) The Gold 5218's single 512-bit FMA unit is confirmed, and it is exploitable.**
L17_rader made an explicit, falsifiable bet on §04 §8.1's claim that some Skylake-SP SKUs
carry one 512-bit FMA unit while retaining two 256-bit ports: it built mixed-width plane
passes (two zmm blocks plus one ymm tail block) as a plan-time candidate, and recorded
that the shape measured **3–4% worse on wallaby** (two 512-bit units), so it could only
be confirmed on the node. The node tuner selected the mixed-width variant (`xl 512t`) in
**all four cells and all three processes**, and the entry improved 2.5–5.2% across the
board. This is the first direct evidence in the project — and, per gap 6, the first
anywhere in the corpus — that the width-versus-port trade on this exact hardware is real,
has the sign the corpus predicted, and is worth several percent. It also demonstrates the
inverse: a kernel shape tuned on Sapphire Rapids will *reject* the shape that wins on
Cascade Lake.

The gap is not closed. The probe measured the **AVX2** clock; every L=17 kernel is zmm,
and no one has measured the AVX-512 licence clock on this node. L17_rader's own cycle
arithmetic (5.2k cycles saved ≈ 2.2 µs "at 2.30 GHz") is built on the wrong clock: at
3.89 GHz that prediction is 1.34 µs and the measured gain was 0.75 µs, so the mechanism
delivered about 55% of its clock-corrected prediction rather than 34% of its
base-clock one. Getting the AVX-512 clock is what turns that from a ratio into a model.

**Also moved, more weakly: §4.3 (is axis fusion worth 3× or 3%?).** L=8 now has a direct
answer at two residencies. L8_fusedaxes (one fused L1-resident pass) wins B=64 by 9%
(0.623 vs 0.680); L8_radix8 (three passes, sequential stores) wins B=2048 by 6.5% and
B=16384 by 10.5%. Fusion helps where the volume is cache-resident and *loses* where the
batch streams — which is Tolmachev's rule (payoff = passes avoided × the bandwidth gap
between the levels involved) coming out on the correct side of the argument, and squarely
in TurboFNO's 3–10% range rather than §07's 3×.

**And §4.6 (model versus search) is settled in practice, if not in principle.** Every
entry now ships a plan-time tournament; six of them shipped hysteresis or wider sampling
this round; the resulting pick instability costs less than the run-to-run spread (§3b).
Search wins, but the round's evidence is that *what you put in the candidate set* is now
the binding constraint, not the search: five separate cross-volume pipelining schemes
were built this round (L17_matrixsimd exec18–20, L17_winograd p4/p8, L36_pfa `M_PIPE`,
L36_pencilfused PIPE/SEQNT/PIPESEQ) and the node tuner selected **none of them in any
cell**, while the round's two real wins came from a prefetch placement change and an
unexplained mode/layout change.

---

## 6. The single highest-value thing the next round should attack, per geometry

**L = 6 — re-open B=1 with the real clock.** The geometry was declared finished on a
cycle count that was wrong by 1.69×; it is at 852 cycles against a 486-cycle FP floor.
Find the missing ~366 cycles. The specific experiment: L6_unrolled's split-store `_s`
shapes won 12% on wallaby by removing 216 port-5 shuffles per volume and were never
selected on the node in 12 invocations — force them via the existing compile-time switch
and A/B them on the node, and run `perf stat` for port pressure and uop counts on the
node rather than inferring it from wallaby. If port 5 is not the answer on Cascade Lake,
that is a publishable negative result for the panel and it redirects the geometry.

**L = 8 — attack the B=64 L2 cliff.** B=64 is a working set of 1.00 MiB, which is
*exactly* the node's per-core L2. Every entry is slower per transform at B=64 than at
B=1 (best: 0.623 vs 0.570, +9.3%) even though a single volume is 8 KiB and should be
trivially resident either way. That is a capacity cliff sitting precisely on a scored
cell, and nobody has blocked the batch loop into sub-L2 groups to step around it. This is
also where §4.5's untouched padding question lives: the L=8 volume-to-volume stride is
8192 B ≡ 0 mod 64 L1 sets, so consecutive volumes collide in every set.

**L = 17 — measure the AVX-512 licence clock, then re-derive.** Extend L6_unrolled's
clock probe (its record explicitly offers it, ~30 lines) to report *both* a 256-bit and a
512-bit sustained clock, and put it in the L=17 description strings. Every L=17 kernel is
zmm; the mixed-width tail result of this round is the first confirmed port-level lever at
this geometry, and it cannot be modelled — only stumbled into — without knowing what a
zmm block actually costs on this part. Second priority, cheap and already asked for by
the implementer: A/B `-DL17R_XF_CUT=64` on the node, since X-first is worth ~10% to
L17_matrixsimd and 11–60% *against* L17_rader on wallaby, and only the node can say which
transfers.

**L = 36 — isolate the 20% at B=32 before building anything else.** L36_pfa's B=32 cell
improved 20.2% for a reason its own record predicted incorrectly, and the entry now leads
its two rivals by 27% in the same cell. With pick reporting live and `FFT36_PW` /
`FFT36_MODE` / `FFT36_PF` forcing available, run the forced-variant matrix at B=32 on the
node and attribute the gain to (i) the hysteresis mode change, (ii) the per-plane
refactor, or (iii) min-of-5 sampling. Until that is done the panel does not know whether
it has a transferable technique or a lucky pick, and two implementers are working in a
cell where the answer would redirect them. Explicitly *not* the priority: more
cross-volume pipelining. Four entries built five such schemes this round and the node
selected zero.

---

## Provenance note

This round's sources are in `bench/geom/impl/`, which is a plain directory (not the
`impl → impl_N` symlink `run_rounds.sh` now creates) and is excluded by `.gitignore`.
`setup_impl_dir()` will migrate it to `impl_4/` — which *is* tracked — at the start of
round 5, so the code is preserved provided round 5 starts through `run_rounds.sh`. Until
then, `exemplars/panel_r4/` is the only versioned copy of this round's code, which raises
the stakes on the promotions below relative to a normal round.

---

## Promotion

Against `docs/CURATION.md`:

1. **Fastest correct entry per geometry.** `L6_unrolled` (fastest in three of four L=6
   cells, tied in the fourth's leader). `L8_radix8` (tied fastest at B=1, fastest at
   B=2048 and B=16384). `L17_matrixsimd` (fastest in all four cells). At L=36 the
   geometry splits cleanly by regime and the brief scores batched and non-batched
   separately, so both winners are kept: `L36_mixedradix` (B=1, B=4) and `L36_pfa`
   (B=32, B=256).
2. **Structurally different runner-up, close behind.** `L17_rader` — 1.08× the leader at
   B=1, a genuinely different algorithm (verified: its output differs numerically from
   both rivals), second at every batch size, and the carrier of the round's mixed-width
   AVX-512 result. `L8_fusedaxes` — not merely a runner-up but the outright winner at
   B=64, and the single-fused-pass arm of the §4.3 question that L8_radix8's three-pass
   structure sits opposite.
3. **Instructive failure.** `L36_pencilfused` — the most elaborate overlap machinery on
   the board (PIPE, SEQNT, PIPESEQ), −11.3% on the development machine, selected in zero
   node cells, and regressed in two. Its record carries the round's cleanest decomposition
   with the numbers that killed it: sequential NT stores *alone* cost 53% (168.2 vs 110.3
   µs/vol) and only win once the drain is hidden under compute (97.8) — store order and
   drain placement are one lever, not two. Promoted so the next panel reads that before
   building a sixth pipelining scheme.
4. **Beat a library.** Selects the whole roster this round (§1) and so is not used.

Not promoted, with reasons: `L6_pfa` (near-duplicate of L6_unrolled — same PFA 2×3
codelet, same variant taxonomy, and its round consisted of adopting the promoted entry's
prefetch); `L8_batchsimd` (produces **bit-identical output** to L8_radix8 at all four
batch sizes — the same arithmetic in a different schedule — and finishes third in three
of four cells; its instructive datum, that shipping its own r2 node-winning configuration
as the tuner default still landed 5.5% short of that number at B=16384, 1.642 vs 1.557,
is recorded here instead); `L17_winograd` (already in `exemplars/panel_r2` and
`panel_r3`, no node movement this round, its `p4`/`p8` variants selected in zero cells,
and its kernel survives inside the promoted `L17_rader`).

PROMOTE: L6_unrolled L8_radix8 L8_fusedaxes L17_matrixsimd L17_rader L36_pfa L36_mixedradix L36_pencilfused
