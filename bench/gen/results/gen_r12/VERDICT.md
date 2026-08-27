# gen_r12 — monitor's verdict

Measured on **a80n0.lqcd.mit**, exclusive reservation, slurm job 438947,
2026-08-26T21:19:44-04:00. CPU **Intel Xeon Gold 6326** (Ice Lake-SP, 2.90 GHz,
AVX-512, 1.25 MB L2/core, 24 MB L3), gcc 11.4.0, `-O3 -march=native`.

## 0. Three corrections to the monitor brief, up front

The brief I was given describes a different campaign, and the mismatch matters for
every number below, so it is stated first rather than quietly worked around.

1. **The geometries are not 6, 8, 17, 36.** Those are the *previous* (`bench/geom`)
   campaign's fixed sizes, which is also why `docs/CURATION.md` and
   `docs/LITERATURE.md` §4 are written in terms of them. The GENERALIZE campaign's
   acceptance suite is frozen in `bench/gen/cases.txt` and is
   **L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100**. There is no L=6, 8, 17 or 36
   cell in `leaderboard.txt`. I report the eleven cells that were actually measured.
2. **The scoring host is Ice Lake-SP, not Cascade Lake.** `environment.txt` reads
   `Xeon Gold 6326`. Cascade Lake and Sapphire Rapids are *advisory* cross-arch hosts
   (`XARCH.md`, `results/xarch_clx_r*`, `xarch_spr_r5`); implementers also develop on
   wallaby (SPR) and on this same Ice Lake node under held leases. So the brief's
   "they develop on SPR, you score on CLX" framing is inverted: most claimed-vs-measured
   gaps this round are **same-machine window effects**, not silicon. Section 4 separates
   the two honestly.
3. **There is no batched/non-batched pair per geometry.** `cases.txt` fixes exactly one
   `(L, B, m)` per size and has been frozen since round 1. Ten cells are batched
   (B = 64…4); **L = 100 is the only non-batched (B = 1) cell**. Section 1 gives the
   batched headline for the ten and the non-batched headline for L=100, and says so
   rather than inventing a second column.

---

## 1. Headline per geometry — fastest correct panel entry vs best library

`per-transform` µs, min of 3 runs. "Best library" = fastest non-`gen_*` entry excluding
`baseline_matrix`; because `fftw3_custom*` are our own genfft-codelet baselines rather
than a shipped library, the strict-library (FFTW/MKL/ducc0) column is given too.

### Batched cells

| L | B | fastest panel entry | µs | best library | µs | ratio | fastest *shipped* library (FFTW/MKL/ducc0) | µs | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 64 | **gen_pfa_small** | 1.116 | fftw3_custom_soa | 4.545 | **4.07×** | mkl_dfti | 4.566 | **4.09×** |
| 12 | 64 | **gen_batchlane** | 1.912 | mkl_dfti | 7.737 | **4.05×** | mkl_dfti | 7.737 | **4.05×** |
| 15 | 32 | **gen_pfa_small** | 4.324 | fftw3_custom_soa | 15.628 | **3.61×** | mkl_dfti | 16.450 | **3.80×** |
| 20 | 32 | **gen_pfa_small** | 12.552 | fftw3_custom_soa | 41.485 | **3.30×** | fftw3_patient | 44.870 | **3.57×** |
| 25 | 16 | **gen_powp** | 31.698 | fftw3_custom_soa | 75.339 | **2.38×** | fftw3_patient | 107.777 | **3.40×** |
| 27 | 16 | **gen_powp** | 44.031 | fftw3_custom_soa | 96.416 | **2.19×** | mkl_dfti | 144.132 | **3.27×** |
| 31 | 16 | **gen_rader** | 84.753 | fftw3_custom_soa | 207.161 | **2.44×** | ducc0_c2c | 715.666 | **8.44×** |
| 32 | 8 | **gen_pow2** | 54.851 | mkl_dfti | 171.758 | **3.13×** | mkl_dfti | 171.758 | **3.13×** |
| 40 | 8 | **gen_pfa_large** | 159.253 | mkl2026_dfti | 403.913 | **2.54×** | mkl2026_dfti | 403.913 | **2.54×** |
| 50 | 4 | **gen_pfa_large** | 413.440 | mkl_dfti | 947.464 | **2.29×** | mkl_dfti | 947.464 | **2.29×** |

At L=31 MKL and all three FFTW planner levels collapse to 848–883 µs (prime axis, no
codelet); `fftw3_custom_soa` is our own 31-point genfft codelet, which is why the two
columns differ by 4×. That cell's 8.44× over ducc0 remains the campaign's widest margin.

### Non-batched cell

| L | B | fastest panel entry | µs | best library | µs | ratio |
|---|---|---|---|---|---|---|
| 100 | 1 | **gen_batchlane** | 4084.110 | mkl_dfti | 7791.419 | **1.91×** |

`gen_powp` is 4090.543 at the same cell — a 0.16 % gap, i.e. a tie — but it is a
near-verbatim port of `gen_batchlane`'s own r11 engine by their record's own account,
not an independent second answer. The nearest *structurally distinct* entry is
`gen_pfa_large` at 4562.497 (1.12×).

**Every one of the eleven cells is a correct panel win over every library.** All entries
report `ok` at both gates; `check.log` contains zero non-PASS lines.

---

## 2. What changed since gen_r11, per geometry

First, a measurement calibration that the round handed me for free. `timing.log` contains
**two complete sweeps of the same binaries** — an earlier pass and the scored pass. Comparing
them per entry per cell gives a direct read of cross-pass reproducibility on this node:

> same binary, two passes, agreement within **±3 %** at 106 of 110 comparable cells.
> The four exceptions are gen_layout@40 (+6.1 %), gen_planner@27 (+6.4 %),
> gen_pfa_small@50 (+4.7 %) — and **gen_race, +18 % to +100 % at all ten cells** (§3).

So: treat any r11→r12 delta under ~3 % as window, and anything over ~6 % as real.

| L | r11 leader | r12 leader | Δ at the top | verdict |
|---|---|---|---|---|
| 10 | gen_pfa_small 1.121 | gen_pfa_small 1.116 | −0.4 % | flat |
| 12 | gen_batchlane 1.915 | gen_batchlane 1.912 | −0.2 % | flat |
| 15 | gen_batchlane 4.340 | gen_pfa_small 4.324 | −0.4 % | flat (leader swap inside a tie) |
| 20 | gen_batchlane 12.597 | gen_pfa_small 12.552 | −0.4 % | flat (leader swap inside a tie) |
| 25 | gen_powp 30.857 | gen_powp 31.698 | +2.7 % | flat/window |
| 27 | gen_powp 43.484 | gen_powp 44.031 | +1.3 % | flat |
| 31 | gen_race 84.549 | gen_rader 84.753 | +0.2 % | flat |
| 32 | gen_pow2 53.809 | gen_pow2 54.851 | +1.9 % | flat |
| 40 | gen_race 159.534 | gen_pfa_large 159.253 | −0.2 % | flat |
| 50 | gen_race 410.975 | gen_pfa_large 413.440 | +0.6 % | flat |
| 100 | gen_race 4071.340 | gen_batchlane 4084.110 | +0.3 % | flat |

**The board did not move at a single scored geometry.** Twelve implementers, one round,
eleven cells, zero measurable improvement at the top of any of them. Note also that every
r11 cell whose leader was `gen_race` is now led by the class entry it was forwarding to —
that is not the classes improving, that is gen_race falling out of the picture (§3).

Where the round's work actually landed, all of it outside the scored cells:

* **gen_pow2** — `GP128_ZF` (next step's z-rows fused into x-stage-2's L1-hot row
  completion): **−8…−15 % at L=128**, ratio vs MKL 1.75× → 1.92×; and the same fusion
  **flips the r11 L=64 verdict to −12 %**. Both sizes unscored. L=32 stays bit-identical
  for the sixth round.
* **gen_rader** — 4-wide dense chunks with an E/O phase split at p ≥ 59:
  **−12.2 % at p=127, −9.6 % at 107, −8.6 % at 83, −5.7 % at 59**, mechanism confirmed by
  counters (port_2_3 −20 %/step, fills flat). All unscored primes; L=31 measured a wash
  and ships untouched.
* **gen_pfa_small** — the within-volume SoA chain adopted from gen_batchlane r11 and
  extended to **every** generic size: −16…−25 % at 14/21/28/36/44, −24 % at L=50
  (726.6 → 568.5) and −15.6 % at L=100 (6066 → 5120). Real and large, but from 1.77×
  and 1.49× off the cell leaders to 1.38× and 1.25× — it closes distance without taking
  a cell.
* **gen_twiddle** — radix-20 leaf as PFA 4×5: **L=20 32.903 → 26.181, −20.4 %**, the
  single largest scored-cell improvement by any entry this round. It is still 2.09× behind
  gen_pfa_small at that cell, so it does not move the board, but it beats every library
  there (1.59× over fftw3_custom_soa, 2.20× over MKL 2026).
* **gen_batchlane** — the one-sweep fused L=100 chain step (CT 10×10 across the step
  boundary) ships: LLC-loads −30 %, LLC-load-misses −28 %, chain drift improved
  4.42e-14 → 2.85e-14. Wall-clock payout ≈ 1 %.

### Regressions

* **gen_race: catastrophic, at ten of eleven cells.** Full treatment in §3.
* **gen_powp at L=50: 413.898 → 480.616, +16.1 %.** Real, and not a kernel change —
  gen_powp's record says L=50 is untouched and measures parity at 418.76. The cause is in
  `results/wisdom_a80n0.json`: the scoring window's cold race banked
  `gen_powp/chain9/L50/B4 → l50-ipm0, tie:1, margin:-0.0188, us:484.842`. The noise gate
  let a **tie with a negative margin** displace the incumbent, and the board number
  (480.616, setup 0.002 s = warm hit on that key) is that banked candidate. A gate that
  can install a candidate it measured as *slower* is a defect, not bad luck.
* **gen_layout at L=40: 329.358 → 348.766, +5.9 %.** Cross-pass artefact: the same r12
  binary read 328.745 in the earlier pass and 348.766 in the scored one. Not code.
  gen_layout's own regression sweep checked 12/32/50 and missed 40.
* **gen_pow2 at L=32: +1.9 %**, and **gen_powp at 25/27: +2.7 %/+1.3 %** — all inside the
  ±3 % cross-pass band, all on paths their records prove bit-identical to r11. Window.

---

## 3. Adversarial pass — failures, missing work, and one fast answer that is not what it looks like

Nothing failed correctness. Every entry at every cell reports `ok`, single-transform
rel-L2 ≤ 3e-15 against tol 1e-12, chain rel-L2 within tol 1e-10, and `check.log` has no
non-PASS line. No panel entry crashed, hung, or failed to build. That part of the round is
clean and I could not break it.

The problems are elsewhere.

### 3.1 gen_race — its board line is an artefact, and the artefact is its own bug

This is the round's biggest single finding, so here is the evidence chain in full.

**Symptom.** gen_race regressed at every cell it contests:

| L | r11 | r12 scored | Δ |
|---|---|---|---|
| 10 | 1.124 | 1.329 | +18 % |
| 12 | 1.915 | 2.353 | +23 % |
| 15 | 4.346 | 5.241 | +21 % |
| 20 | 12.694 | 17.662 | +39 % |
| 25 | 31.164 | 40.961 | +31 % |
| 27 | 43.545 | 60.041 | +38 % |
| 31 | 84.549 | 138.751 | **+64 %** |
| 32 | 54.041 | 108.994 | **+102 %** |
| 40 | 159.534 | 236.639 | +48 % |
| 50 | 410.975 | 545.065 | +33 % |
| 100 | 4071.340 | 4179.005 | +2.6 % |

In r12 its numbers sit on top of `gen_planner`'s at every cell except L=100 — i.e. it ran
its own self engine and never forwarded to the class winner.

**It is not the code.** The *same binary* in the earlier pass of `timing.log` reads
1.122 / 1.917 / 4.335 / 12.546 / 31.404 / 42.842 / 84.908 / 54.619 / 159.306 / 413.068 —
i.e. exactly the class winners' numbers. The cross-class routing works.

**It is not a bad race verdict.** `results/wisdom_a80n0.json` holds the correct
`eng12` winner for every cell, several of them decisive:

```
gen_race/eng12/L32/B8  -> gen_pow2.2ff9737a      margin 0.8451
gen_race/eng12/L31/B16 -> gen_rader.d5bd17fd     margin 0.2696
gen_race/eng12/L40/B8  -> gen_pfa_large.d302f221 margin 0.2702
gen_race/eng12/L25/B16 -> gen_powp.75712edc      margin 0.4420
gen_race/eng12/L50/B4  -> gen_pfa_large.d302f221 margin 0.1131
```

The L=32 margin of 0.845 is the race correctly measuring gen_pow2 as 1.85× faster than
self — and the board's own ratio, 108.994 / 54.851 = 1.99×, confirms the race was right.
It banked the right answer and then did not use it.

**The mechanism.** The dlopen cache is `build/<host>/race_eng/<name>.<hash8>.so`. On this
node those eight `.so` files are timestamped **01:01–01:03 on Aug 27** — built during the
L=100 cell (00:53–01:03), the *last* cell of the sweep. Every earlier cell (L=10 at 21:28
through L=50 at 00:12) ran with that directory empty, because the harness rebuilds
`build/<host>/` at sweep start while `results/wisdom_<host>.json` survives. And
`impl/gen_race.c` gates the compile phase on wisdom, not on cache readiness:

* `gen_race.c:2632` — *"Compile phase, skipped when wisdom already answers"*; `have_wisdom`
  is set purely by a successful `gr_wisdom_lookup`, with no `grx_so_ready()` check.
* `gen_race.c:2724` — `if (!grx_materialize(&ec[ew])) ew = 0; /* fall back */` — the
  dlopen failure is swallowed and self ships, silently.

So: stale-but-correct wisdom + cold `.so` cache ⇒ no compile fires ⇒ dlopen fails ⇒ silent
2× degradation. At L=100 the key happened to re-hash (new `_f0` variant arm), `have_wisdom`
came out false, the 32.847 s prefetch-all fired, and that one cell routed correctly.

**Judgement.** gen_race's r12 record anticipates this and asks the monitor for a prewarm
("one throwaway create on the scoring host before the suite"); the harness did not do it,
and that is a harness gap I am recording. But a tuning layer whose documented failure mode
is *"lose 2× silently and ship the wrong engine"* is carrying a real defect. The four-line
fix is in its own code comment: `have_wisdom &= all_arms_ready()`, or emit a warning on
`grx_materialize` failure so the fallback is visible in `timing.err` instead of invisible
in the leaderboard. **The r12 board line for gen_race does not represent gen_race's code
and must not be read as one.**

### 3.2 gen_bluestein — a no-op round with a missing record and a 116-minute lease

* `results/gen_r12/agents/gen_bluestein.log` is **1 byte** (a bare newline). Every other
  implementer's log is 0.9–2.3 KB.
* `strategies/gen_bluestein.md` ends at *"### What I would do next (gen_r12)"* — **there is
  no `## Round gen_r12` section.** It is the only entry of twelve with no record for this
  round.
* It nonetheless changed its source (111 diff lines vs impl_11: a `BST_NOSTG` staging
  rewrite in `last_scatter_tr`, which is its own r11 next-list item #1). Undocumented,
  unmeasured, unattributed. Its L=100 number moved 13536.8 → 13499.7, i.e. −0.3 %, versus
  the ~−3 % its r11 record projected for this change.
* `logs/rounds.log`: *"still 1 lease(s) held after 1800s; reaping and proceeding — reaping
  stale lease slot0 held by gen_bluestein_r12 for 116 min"*. It held a benchmark-node lease
  for nearly two hours, delayed the scoring window by 30 minutes, and produced no record.
* It is also the only panel entry that **never beats a library at any cell**: at L=10 it is
  2.77× *slower* than MKL, at L=100 1.73× slower.

`docs/CURATION.md` is explicit — *"Do not promote … entries whose strategy record is
missing — the record is what makes the code useful later."* Not promoted. The source is
preserved in `impl_12/` regardless, which is exactly what per-round directories are for.

### 3.3 gen_rader — a latent undefined-behaviour warning, unchanged from r11

`build_errors.txt` (the only build diagnostic this round, and a warning not an error):

```
impl/gen_rader.c:2648:23: warning: iteration 1152921504606846976 invokes undefined
behavior [-Waggressive-loop-optimizations]   double re = zp[2 * i] + cp[2 * i];
```

gcc has proved that `for (; i < npts; ++i) … zp[2*i]` overflows `2*i` before the loop
bound is reached, which licenses it to do anything with the loop. It is the same warning at
the same function in r11 (line 2429 then), gen_rader passes every gate, and its L=31 output
is bit-identical to r4's — so this is not a correctness failure *today*. But it is a
standing licence for gcc to delete or transform that tail on any future flag or version
change, in the entry that owns the campaign's widest margin. Fix is one line
(`size_t i` / hoist the bound). Flagged for r13.

### 3.4 The wisdom cache is round-persistent state that outlives the binaries

Both §3.1 and the gen_powp L=50 regression are the same class of bug: `results/wisdom_*.json`
survives a round boundary and a `build/` rebuild, so a verdict banked under contention (or
against a cache that no longer exists) is replayed as a warm hit at scoring time with
setup ≈ 0.002 s and no way to tell from the leaderboard. Three implementers
(gen_pfa_large, gen_powp, gen_race) independently strip their own dev keys at round end —
that discipline is good and should stay — but it does not protect against verdicts banked
by the *scoring* sweep itself. Harness recommendation for r13: run the scored sweep from a
clean wisdom file, or prewarm `build/<host>/race_eng/` before the first cell.

---

## 4. Claimed vs measured

The brief expects cross-architecture drift to explain the gaps. On this round it mostly does
not: the implementers develop on wallaby (SPR) *and* on this same Ice Lake node under held
leases, and they quote both separately. Almost every claim lands.

| entry | cell | claimed | measured | gap | attribution |
|---|---|---|---|---|---|
| gen_batchlane | 100 | 4025–4079 (quiet) | 4084.1 | +0.1 % | — |
| gen_powp | 100 | 4074.2 / 4070.0 | 4090.5 | +0.4 % | — |
| gen_pfa_small | 100 | 5178–5186 | 5120.0 | −1.1 % | — |
| gen_pfa_small | 50 | 551–563 | 568.5 | +1.0 % | — |
| gen_rader | 31 | 85.11 | 84.75 | −0.4 % | — |
| gen_dense_prime | 31 | 109.5–111.2 | 111.3 | +0.1 % | — |
| gen_pow2 | 32 | 54.14–54.93 | 54.85 | in range | — |
| gen_layout | 100 | 9047.6 | 9092.3 | +0.5 % | — |
| gen_planner | 100 | 4439.9–4560.5 | 4560.3 | top of range | — |
| gen_twiddle | 20 | 26.99 | 26.18 | −3.0 % | — |
| gen_twiddle | 100 | 6333.8 | 6468.0 | +2.1 % | window |
| gen_pfa_large | 100 | 4428.8 (min) | 4562.5 | **+3.0 %** | window; their own record flags "median hit a churn burst" |
| **gen_powp** | **50** | **418.76 (parity)** | **480.6** | **+15 %** | **not the machine — banked race verdict, §2** |
| **gen_race** | **50** | **414.15** | **545.1** | **+32 %** | **not the machine — routing failure, §3.1** |
| **gen_race** | **12** | **2.019** | **2.353** | **+17 %** | **same** |

**The two large gaps are both machinery, not silicon**, and both are diagnosable from
artefacts in this results directory. That is a good outcome for the round's evidence
discipline and a bad one for the tuning layer.

Where genuine architecture difference *does* appear, the implementers already attributed it
correctly and I confirm their framing:

* gen_pow2 measured the same L=128 fused chain at **7.02 ms on SPR vs 10.74 ms on ICX** —
  a 1.53× machine gap on a DRAM-regime cell, with both new defaults transferring in sign.
* gen_rader reads its L=31 B=1 chain at **96.6 µs on a80n0 and 85.2–85.3 µs on a81n2** on
  bit-identical output — a same-architecture *node/window* gap of 13 %, which is a useful
  caution: not every cross-host discrepancy is an ISA story.
* `XARCH.md` remains the standing cross-arch datum: all eleven cells still win on Cascade
  Lake, but the ratios compress ~30 % at the compute-bound cells (L=31: 8.48× → 6.15×)
  from 512-bit licence downclocking, while the memory-bound cells (40, 100) barely move.
  Nothing this round contradicts it, and several of the round's new knobs
  (`BL_FUSE100`, `GP64_FUSE`, `GWVS*`, `RP_PFT1`, `GDP_YPIPE`) exist specifically so the
  CLX/SPR races can re-decide.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

**§4.3, "Is axis fusion worth 3× or 3%?" — specifically the clause re-opened by §08 §1.9,
the untested L2↔DRAM regime.** That clause reads:

> "The untested case is **L2↔DRAM**, where the gap is 7× … tile the batch so a tile fits L2,
> then run all three axes inside the tile. That is not the same experiment the panel ran,
> and it is the largest untried structural move on the board."

Six entries ran exactly that experiment this round, on six different engines, with counters.
The results do not agree on sign, and the disagreement is the finding:

| entry | fusion built | wall-clock | traffic |
|---|---|---|---|
| gen_pow2, L=128 | z-rows into x-stage-2 (`GP128_ZF`) | **−8…−15 %** | instructions −9.3 %, L1 fills −11 %, LLC flat |
| gen_pow2, L=64 | one-sweep + ZF | **−12 %** (flips the r11 verdict) | LLC-loads −86 % |
| gen_pow2, L=128 | y+stage-1 slab fusion (`GP128_YF`) | **+2 %** | l2_lines_in −18 % but LLC-load-**misses** +33 % |
| gen_batchlane, L=100 | one-sweep CT 10×10 across the step seam | **−0.3…−1.2 %** | LLC-loads −30 %, LLC-load-misses −28 % |
| gen_pfa_large, L=100 | one-sweep `ipw1` | **+16…+25 %** | demand LLC-load-misses only 0.4 MB/step under an 80 MB/step stream |
| gen_planner, L=100 | within-volume `pln_wv` | **+6.6 %** | l1d.replacement **−31 %** (target met), stores +45 % |
| gen_dense_prime, L=31 | cross-plane pipelined y-pass | **+7 %** | port uops identical, l1d.replacement **+33 %** |

**The reconciliation the round produced, and it is a genuine sharpening of Tolmachev's rule
as §4.3 states it.** The corpus's rule prices a fusion at *passes avoided × bandwidth gap
between the levels involved*. Four entries independently measured that the pass count and
the paper MB/step are both the wrong inputs:

1. **Only *demand* traffic is worth restructuring against.** gen_pfa_large's counters are
   the cleanest demonstration in the campaign: an 80 MB/step DRAM stream with **0.4 MB/step
   of demand LLC-load-misses** — the rest is hardware-prefetch-covered and therefore already
   free. gen_batchlane's independently reached the same reading at the same cell
   (~2 MB/step demand DRAM under a 16.8 MB state that lives in the 24 MB LLC). Both
   entries explicitly **retract** their own r11 "L=100 is ~88 % DRAM-BW-bound" corollary.
   That is what turns gen_batchlane's −30 % LLC-load cut into a 1 % wall-clock win instead
   of gen_pow2's 14 %: the bandwidth gap in Tolmachev's product must be the *stalled* gap,
   not the nominal one.
2. **A fusion is only free if it keeps the pass count flat.** gen_pfa_large: *"a pass
   'deleted' by splitting another pass in two is a net ADD"* — their `ipw1` split the x
   transform across the step seam and paid an extra volume-level tail pass, +16…25 %.
   gen_pow2's winning shape shares one tile visit between stage-2 and stage-1 and adds
   nothing.
3. **A fusion is only free if its walk stays streamer-friendly.** gen_pow2's `GP128_YF`
   cut L3→L2 traffic 18 % and *lost* 2 %, because its fused walk is 128-B touches at 2112-B
   stride which no prefetcher tracks: `l2_lines_in` down while wall-clock up is the
   signature.
4. **A fusion is only free if the concurrent hot set still fits the level you are trying to
   exploit.** gen_dense_prime's y-pipe is the calibration point: 55–60 KB concurrent against
   a 48 KB L1D ⇒ +33 % fills, +7 % time, with *identical port uops to four digits*. They
   propose the resulting design gate — cost candidates in concurrent L1-resident bytes,
   adopt only under ~40 KB — and gen_rader used it at design time this round and won −12.2 %.

So §4.3's re-opened L2↔DRAM clause is now **answered with CPU counters, and the answer is
"it depends, and here is the discriminator"**: run
`LLC-loads` / `LLC-load-misses` *before* building any traffic restructure; `l1d.replacement`
and paper MB/step both lie about what the machine is waiting for. Three entries state that
rule in their records in almost the same words, having reached it separately. I regard the
clause as settled in the same partial sense §4.3's existing note settles the L1↔L2 case, and
recommend the LITERATURE §4.3 block be amended with the seven-row table above.

Two smaller movements worth logging:

* **§4.6 (model versus search for the instruction schedule)** gains a sharp datum on the
  search side: gen_batchlane measured **+4 % from runtime-constant knob dispatch on
  bit-identical output** — routing a tail through a per-column function changed gcc-11's
  codegen enough to lose 4 %. gen_twiddle independently measured **+1.3…+6 % at L=12 from
  an `if (p->pf0)` gate at an inlined call site, with the gate OFF**, fixed by moving the
  gated arm into its own `noinline` function. §4.6's "you should not need a search phase"
  position looks worse each round; "code shape is a measured variable, re-race after any
  refactor even a transparent one" is the panel's operating rule now.
* **§4.8 item 6 (AVX-512 on server parts)** is not moved. The Ice Lake-SP figures continue
  to accumulate and `XARCH.md`'s ~30 % CLX ratio compression is consistent with
  licence-based downclocking, but no new primary measurement was taken.

---

## 6. Highest-value target for r13, per geometry

| L | the one thing |
|---|---|
| **10, 12, 15** | **Nothing in the kernels — fix the routing.** These three cells are converged to within 0.5 % across gen_pfa_small / gen_batchlane / (working) gen_race and have not moved in three rounds. The only measurable loss here in r12 was gen_race shipping self instead of the winner (§3.1). Spend the round on `have_wisdom &= all_arms_ready()` and a visible fallback warning, not on butterflies. |
| **20** | Race gen_twiddle's new **radix-20 PFA-4×5 leaf** (−20 % on its own engine, zero internal twiddles, one whole level deleted) against gen_pfa_small's and gen_batchlane's incumbent 4×5 two-stage pencils. It is the only structurally new DFT20 shape produced in six rounds and it has never met the cell leaders. |
| **25, 27** | **Delete uops, not fills.** gen_layout's r10 dashboard has both cells running *at* the node's dispatch cap, so traffic work cannot pay here. The transferable method is gen_rader's r12 result: 4-wide chunks with an E/O phase split bought −12.2 % purely by deleting broadcast dispatch (port_2_3 −20 %/step) with fills flat. Apply the same widening to gen_powp's `l25-soa`/`l27-soa` pencils. |
| **31** | **Closed on Ice Lake; take it cross-arch.** gen_rader is bit-identical since r4 and port-saturated; gen_dense_prime's last structural lever (the pipelined y-pass) died at +7 % this round with a mechanism; their own remaining option (asm 32-accumulator GEMM) is priced at single digits on a cell they lose by 1.31×. Meanwhile L=31 is the board's worst CLX compression, 8.48× → 6.15×. Race a 256-bit (ymm) variant of the Rader conv kernels on CLX, where they may clear the 512-bit downclock. |
| **32** | **Does the r12 fusion reach down one regime?** gen_pow2 flipped `GP64_FUSE` on at L=64 for −12 % using the new z-row fusion, and reports "nothing at L=32" — but the record contains no evidence that `ZF` was ever *tried* at G=4. L=32 has been bit-identical for six rounds; this is the only untested structural move left on it. |
| **40** | The second-worst library ratio on the board (2.54×) and the pick has been `l40-ip0` since r9. Given §5's finding that the demand-traffic budget at these sizes is near zero, the lever is overlap quality: gen_pfa_large's own item — move compute-phase staging round trips onto the idle 256-bit access class — plus a first honest cross-class race of gen_pfa_small's r12 within-volume engine at 40, which has never been run there. |
| **50** | **16 % is sitting on the floor for zero kernel work.** Fix the noise gate so a tie with a *negative* margin cannot displace an incumbent (`gen_powp/chain9/L50/B4 → l50-ipm0, tie:1, margin:-0.0188`). That alone restores gen_powp from 480.6 to ~414. Then re-run the powp/pfa_large playoff in a quiet window — these two are within 0.6 % and the cell's leader has flipped twice. |
| **100** | **Latency, not traffic.** Three entries independently proved this round that the cell is prefetch-covered and not DRAM-BW-bound; gen_batchlane's −30 % LLC-load cut bought 1 %. The measured residual is instruction count and overlap: gen_pfa_small reads 24.7 M insn/step against gen_batchlane's ~17.3 M at the same traffic. Attack gen_batchlane's own next-list — the hybrid register-form verticals (half of `BL_F100RV`'s pressure) and software-pipelining the head across sites, now that halved DRAM crossings have left slack under it. |

**And one cross-cutting item that outranks all eleven:** the round produced *zero* movement
at *every* scored cell while producing large, well-measured wins at unscored sizes
(64, 128, 14–44, 59–127). Twelve implementers spent a round on `L=100` per the brief and
moved it 0.3 % in the wrong direction. Either the acceptance suite should be re-pointed at
the sizes where the class engines are still improving, or r13 should be the consolidation
round: land the two machinery fixes (§3.1, §2) and re-score, rather than adding a
thirteenth engine.

---

## 7. Curation

Applying `docs/CURATION.md`'s four grounds in order.

**1 — fastest correct entry per geometry (mandatory).**
`gen_pfa_small` (10, 15, 20), `gen_batchlane` (12, 100), `gen_powp` (25, 27),
`gen_rader` (31), `gen_pow2` (32), `gen_pfa_large` (40, 50). Six entries, eleven cells.

**2 — structurally different runner-up.**
`gen_dense_prime` at L=31 (111.333 vs 84.753, 1.31×). Wider than the ~20 % the criterion
suggests, but it is the documented other side of the campaign's central dense-vs-Rader
crossover, and it qualifies overwhelmingly on ground 3 below. Kept.
*Not* kept on this ground: `gen_powp` at L=100 — a 0.16 % gap but a near-verbatim port of
gen_batchlane's engine by its own record, i.e. exactly the "near-duplicate" the criterion
excludes. (It is promoted anyway as the 25/27 winner.)

**3 — instructive failures, whose record carries the number that killed them.**
* `gen_dense_prime` — cross-plane pipelined y-pass, **+7 %, 4/4**, with port uops identical
  to four digits and `l1d.replacement +33 %`, plus a footprint-minimised variant at +5 %
  proving the loss monotone in added footprint. It yields a reusable design gate
  (concurrent L1-resident bytes < ~40 KB) that gen_rader used the same round to win −12.2 %.
  This is the most useful negative on the board.
* `gen_planner` — `pln_wv` built, iterated through three forms from +29 % down to +6.6 %,
  hitting the round's stated traffic target (`l1d.replacement −31 %`, lower FMA-port work
  than the incumbent) and still losing on wall-clock. "Traffic goal met, time not won,
  here are the counters" is precisely the record the next panel must not have to re-earn.

**4 — anything that beat a library baseline, regardless of rank.**
* `gen_twiddle` — 1.59× over the best baseline and 2.20× over MKL 2026 at L=20, off a
  **−20.4 %** radix-20 leaf that is the largest scored-cell improvement in the round; and it
  is the library layer `gen_bluestein` and `gen_pow2` both build against.
* `gen_layout` — beats MKL at nine of eleven cells (4.18× at L=31) and is the layer carried
  by three of the four fastest L=100 entries; its r12 doctrine ("accumulator kernels are
  latency-tolerant — `stalls_l1d_miss` is correlation, not causation") was borrowed and
  confirmed by `gen_rader` inside the same round.
* `gen_race` — beats every library at every cell **even in its degraded state** (L=32
  108.994 vs MKL 171.758), is adopted by `gen_planner`, `gen_pfa_large` and `gen_powp` on a
  frozen API for the eighth round, and delivered the variant-arm mechanism `gen_batchlane`
  asked for. Promoted **with the §3.1 defect named in the round note** — its r12 board line
  is not a measurement of its code, and the next panel must be told that in the same breath
  as it is handed the source.

**Not promoted.** `gen_bluestein` — no `## Round gen_r12` section in its strategy record
(the only entry of twelve), a 1-byte agent log, an undocumented 111-line source change, a
116-minute stale lease that delayed the scoring window, and no library beaten at any cell.
`docs/CURATION.md`: *"Do not promote … entries whose strategy record is missing."* Its
source remains in `impl_12/` for provenance; it is simply not on the reading list.

PROMOTE: gen_pfa_small gen_batchlane gen_powp gen_rader gen_pow2 gen_pfa_large gen_dense_prime gen_planner gen_twiddle gen_layout gen_race
