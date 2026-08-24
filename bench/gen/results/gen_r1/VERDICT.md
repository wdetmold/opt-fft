# VERDICT — round gen_r1 (GENERALIZE campaign, round 1 of 6)

Monitor: judgement pass over the timing pass already written to this directory.
Nothing was re-measured; every number below is quoted from `leaderboard.txt`,
`c_*.json`, `o_*.json`, `failures.txt`, `build_errors.txt`, `agents/exits.txt` or
`timing.err` in this directory, or from `strategies/*.md`.

---

## 0. Three corrections to the monitoring brief, before the numbers

These matter because two of them would have made the report wrong, so they are
stated first rather than buried.

**0.1 The geometries are not L = 6, 8, 17, 36.** Those are the *previous*
campaign's sizes (`bench/geom`, rounds `panel_r1..r11` + `graded_icelake`, tuned
cubes 6, 8, 13, 17, 23, 36, 45, 64). This is the GENERALIZE campaign, whose
acceptance suite is `bench/gen/cases.txt` and is deliberately disjoint from it —
its header says so: *"none of these sizes was ever tuned by any prior round."*
The eleven scored cases are

| L | 10 | 12 | 15 | 20 | 25 | 27 | 31 | 32 | 40 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B | 64 | 64 | 32 | 32 | 16 | 16 | 16 | 8 | 8 | 4 | 1 |
| m | 1000 | 600 | 600 | 256 | 256 | 200 | 140 | 250 | 128 | 128 | 64 |

I report on all eleven. There is no L = 6, 8, 17 or 36 measurement in this round
to report on.

**0.2 The scoring machine is Ice Lake-SP, not Cascade Lake.** `environment.txt`
records `a80n0.lqcd.mit`, **Intel Xeon Gold 6326 @ 2.90 GHz** — Ice Lake-SP
(Sunny Cove): 1.25 MiB L2 per core, two 512-bit FMA units, and no licence-based
AVX-512 downclock at the one-active-core count these single-threaded runs use.
It is not a Gold 5218 / Cascade Lake (1 MiB L2, one 512-bit FMA on that SKU).
`CAMPAIGN_GENERALIZE.md` and `PANEL_BRIEF.md` both specify Ice Lake as the
scoring node, so this is the intended machine, not a mis-run. The `docs/`
Cascade Lake material — including `LITERATURE.md` §4.8 item 6, which is written
about the Gold 5218 specifically — describes the *previous* campaign's node.

**0.3 The implementers developed on this same node, not on Sapphire Rapids.**
`PANEL_BRIEF.md`: *"Development on wallaby via ./tryout.sh (leased node
cores)"*, i.e. leased cores on a80n0 itself. Every implementer's headline table
is labelled accordingly — e.g. `gen_planner.md`: *"Measured on the reserved Ice
Lake node (a80n0) via tryout.sh, graded chain"*; `gen_pfa_large.md`: *"Measured
on the node (a80n0, Gold 6326 Ice Lake, graded chain m, min)"*. Consequently
there is **no machine-difference correction to apply in §4** — the claims and the
measurements agree to a few percent. The one implementer who did report dev-host
numbers alongside node numbers (`gen_powp`) measured the gap directly, and it is
**1.29–1.52x**, not the 2.9x MKL span quoted in the brief. Details in §4.

**Measurement method as run** (from `sweep.sh`): 3 runs × 20 samples, 5 warmup
iterations, ≥20 ms per sample, **min over runs**, spread reported; compile, plan
and warmup excluded from the timing, plan time reported separately as `setup`.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

"Best library" = the fastest of {MKL 2022, MKL 2026, FFTW3 estimate/measure/
patient, ducc0} in that case. Times are µs per transform (per volume), min over
3 runs. All entries listed passed all three gates (§3).

### The batched cases (ten of eleven)

| L | B | fastest panel entry | µs | GF/s | best library | µs | **speed-up** |
|---|---|---|---|---|---|---|---|
| 10 | 64 | `gen_batchlane` | **1.174** | 42.44 | mkl_dfti | 4.548 | **3.87x** |
| 12 | 64 | `gen_batchlane` | **1.995** | 46.57 | mkl_dfti | 7.744 | **3.88x** |
| 15 | 32 | `gen_batchlane` | **4.686** | 42.21 | mkl_dfti | 16.445 | **3.51x** |
| 20 | 32 | `gen_pfa_small` | **16.920** | 30.65 | fftw3_measure | 44.893 | **2.65x** |
| 25 | 16 | `gen_powp` | **48.778** | 22.31 | fftw3_patient | 108.664 | **2.23x** |
| 27 | 16 | `gen_powp` | **66.078** | 21.25 | mkl_dfti | 144.425 | **2.19x** |
| 31 | 16 | `gen_rader` | **94.170** | 23.51 | ducc0_c2c | 725.494 | **7.70x** |
| 32 | 8 | `gen_pow2` | **63.683** | 38.59 | mkl_dfti | 171.437 | **2.69x** |
| 40 | 8 | `gen_pfa_large` | **200.515** | 25.48 | mkl2026_dfti | 405.425 | **2.02x** |
| 50 | 4 | `gen_powp` | **473.484** | 22.35 | mkl_dfti | 948.870 | **2.00x** |

### The non-batched case (one of eleven)

| L | B | fastest panel entry | µs | GF/s | best library | µs | **speed-up** |
|---|---|---|---|---|---|---|---|
| 100 | 1 | `gen_powp` | **5026.755** | 19.83 | mkl_dfti | 7817.717 | **1.56x** |

**The suite has no geometry measured both batched and non-batched.** Each case in
`cases.txt` fixes one B, and only L=100 draws B=1. So the batched/non-batched
split asked for cannot be given per geometry from this round's scored data; it is
one case at B=1 and ten at B ≥ 4. That is a real gap in the acceptance suite, not
a gap in the measurement, and it is dangerous because **round 6 draws three
unannounced sizes and scores the assembled trunk** — at whatever batch the
monitor picks. The advisory B=1 numbers the implementers self-report on their own
cores are the only evidence on the question, and they are alarming for the
small-L winner:

| entry | B=1, self-reported | MKL B=1, same source | verdict |
|---|---|---|---|
| `gen_batchlane` L=10/12/15 | 10.6 / 18.0 / 41.3 µs | 4.5 / 7.4 / 16.3 | **2.3–2.5x SLOWER than MKL** |
| `gen_pfa_small` L=10/12/15/20 plain | 3.75 / 5.30 / 15.06 / 30.94 | 1.48 / 2.13 / 6.39 / 31.08 | loses at 10/12/15, ties at 20 |
| `gen_powp` L=25/27/50 | 48.7 / 68.3 / 615 | 133.1 / 165.8 / 940 | wins 1.53–2.73x |
| `gen_pfa_large` L=40/50 | 236.3 / 557.5 | — | (no MKL pair given) |
| `gen_planner` L=10/25/32 | 9.07 / 151.3 / 304.1 | — | per-transform cost ≈ batched; no remainder penalty |

Summary of §1: **the panel beat the best library at every one of the eleven
acceptance sizes.** Geomean speed-up over best library **2.82x**; worst case
**1.56x** (L=100, the memory-pressure case); best case **7.70x** (L=31, prime).
Against MKL 2022 alone the geomean is 2.96x (range 1.56x–9.01x). Every scored case is a win with
margin, and the two-part gate held everywhere (§3).

Panel entries that beat the best library but did **not** win their case — these
matter for §7:

- L=10/12/15: `gen_pfa_small` (1.21x / 1.25x / 1.32x behind `gen_batchlane`, still 3.2x/3.1x/2.7x over MKL)
- L=31: **four** further entries beat ducc0's 725.494 — `gen_dense_prime` 175.591 (4.13x over best library), `gen_bluestein` 405.309 (1.79x), `gen_layout` 466.299 (1.56x), `gen_planner` 606.815 (1.20x). Every library on the node is between 2.51 and 3.05 GF/s at L=31; the panel's *fallback path* is faster than all of them.
- L=50: `gen_pfa_large` 481.372 — 1.7% behind the winner, 1.97x over MKL.
- L=100: `gen_pfa_large` 5086.365 — 1.2% behind the winner, 1.54x over MKL.

---

## 2. What changed since the previous round — and did anything regress?

**Nothing regressed, because nothing could have.** `results/.rounds_state` reads
`1 6`: gen_r1 is the *first* round of this campaign, and `results/` contains
exactly one round directory. There is no previous gen round to regress against,
and no per-geometry delta to compute.

The only prior data on this same node is the previous campaign's final grading,
`bench/geom/results/graded_icelake` (a80n0, Gold 6326, 2026-08-22 — 26 hours
before this round). **Its sizes are disjoint from this suite** by construction
(6, 8, 13, 17, 23, 36, 45, 64 vs 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100),
so there is no size-for-size comparison to make. What can honestly be compared is
the achievement level, and it is the round's most important result:

| | previous campaign, final graded round | this round |
|---|---|---|
| sizes | 8, each tuned for 11 rounds | 11, **none ever tuned** |
| per-size winners | `L6_unrolled`, `L8_fusedaxes`, `L13_direct`, `L17_matrixsimd`, `L23_rader`, `L36_pfa`, `L45_mixedradix`, **`mkl_dfti` at L=64** | six class engines, no per-size code |
| margin over best library | 1.16x – 5.55x, **and a loss at L=64** | 1.56x – 7.70x, **no losses** |
| coverage per entry | one size | a factorization class (and any L for planner/bluestein) |

(The two rounds used different workload configurations — `graded_icelake` was the
rival-grader configuration, this is the chain suite of `cases.txt` — so treat the
margin row as comparable in kind, not to three digits.)

So the change since the previous round is that the *class* formulation did not
cost margin: eleven never-tuned sizes served by six generic class engines beat
every library at every size, where the eight-size hand-tuned campaign had lost one
of its eight to MKL. That is the campaign's central bet paying off in round 1.

**The one thing that plainly did not happen: layer adoption is zero.** Four of the
twelve roster entries (`gen_planner`, `gen_race`, `gen_twiddle`, `gen_layout`) are
library layers whose scoring is *by adoption* (`PANEL_BRIEF.md`: "the monitor
credits them when class entries win using their layers"). Grepping every entry in
`impl/` for `#include "gen_*.c"` or the documented `GEN_*_LIB_ONLY` guards finds
**not one adoption by any class entry**; the only cross-file include in the whole
tree is `gen_twiddle.c` including `gen_layout.c` (one layer using another). All
four layers therefore **score zero adoption in r1**, and every class winner shipped
its own file-local twiddles, its own allocation, and no race. Each class owner's
record says "adopt as they land" and defers to r2. This is the round's biggest
structural miss and it is on the critical path: r5 assembles a trunk from parts
that currently share no code, and r6 scores only that trunk.

Real code transfer *did* happen, peer-to-peer rather than through the layers, and
it is worth crediting because it is what the exemplar mechanism is for:

- `gen_powp` adopted `gen_pfa_large`'s engine shell **wholesale** mid-round, and records the number: its own first cut ran L=100 at **12,729 µs** against `gen_pfa_large`'s concurrent 5,030; adopting the `ip*` in-place family took L=100 to ~5,000 and L=50 from 789 to 478.
- `gen_rader` adopted `gen_dense_prime`'s volume-resident fused chain and s6 map, and records **110.4 → 94.3 µs/step (−16 µs)** from making state + c L2-resident at 953 KB.
- `gen_batchlane` credits nearly all of its structure to the previous campaign's `L8_fusedaxes` bl8 (which itself credits rival reconstructions v5_cb7847fb / 8dc1a96d).

---

## 3. Adversarial pass: correctness, builds, crashes, omissions

**Every gate was checked independently from the round's own artifacts, not taken
from the implementers' self-reports.** Result: **no entry failed any gate, and no
fast-wrong answer is in the standings.**

### 3.1 Gate 1 — single call, rel L2 < 1e-12 vs numpy
152 `c_*.json` files, one per (backend, case) cell that produced a timed output.
Audited all 152 programmatically: `ok: true` in **152/152**. Worst single-call
residual anywhere in the round is 9.4e-16 (`baseline_matrix` L=100) — three
orders of magnitude inside tolerance. `check.log` contains **zero** `FAIL` lines.

### 3.2 Gate 2 — the two-step precision gate (m=2, 3e-14 = 1.5e-14/step)
This is the gate `PANEL_BRIEF.md` calls the one that *"catches every shortcut"*,
and it is easy to miss because it is not in the `c_*.json` files. `sweep.sh`
lines 110–120 run a separate `--chain 2 --map` pass and write it to
`o_*.json`. **153 files, `ok && one_ok` true in 153/153.** Worst two-step
residual in the entire round: **4.858e-15** (`baseline_matrix` L=100) against a
3e-14 tolerance — a 6.2x margin at the very worst cell, and 16–36x at the class
winners:

| winner | single-call rel L2 | two-step rel L2 (tol 3e-14) |
|---|---|---|
| `gen_batchlane` L=10 | 2.59e-16 | 8.35e-16 |
| `gen_pfa_small` L=20 | 3.19e-16 | 1.32e-15 |
| `gen_powp` L=25 | 3.60e-16 | 1.54e-15 |
| `gen_rader` L=31 | 4.06e-16 | 1.87e-15 |
| `gen_pow2` L=32 | 2.87e-16 | 1.34e-15 |
| `gen_pfa_large` L=40 | 3.59e-16 | 1.78e-15 |
| `gen_powp` L=100 | 4.52e-16 | 2.45e-15 |

Nobody is anywhere near this gate. No entry bought speed with precision.

### 3.3 Gate 3 — chain end vs 300x the honest reference divergence
`chain_ok: true` in 152/152. Spot-checking the ratio to the anchor rather than to
the tolerance (the meaningful test): the winners land at 1.0–2.0x their own
honest anchor — e.g. `gen_batchlane` L=10 `chain_rel_l2` 9.43e-14 against
`anchor_rel_l2` 8.05e-14 (1.17x), `gen_rader` L=31 2.8e-14 against 2.0e-14. Drift
is *at* the anchor, i.e. exact tier, not merely inside a 300x cheat-catcher.

### 3.4 Builds
`build_errors.txt` contains **no errors** — only warnings, all four from one
entry. `agents/exits.txt`: all twelve implementers `exit=0`. Every roster entry
built and ran.

**But the warnings name the L=20 winner, and they are UB warnings, so they get
called out.** `gen_pfa_small.c:324`, four instances (`soa_step_10/12/15/20`):

```
warning: iteration 2305843009213693951 invokes undefined behavior
         [-Waggressive-loop-optimizations]
  324 |         double a = zr[i] + cr[i], b = zi[i] + ci[i];
  323 |     for (; i < n; ++i) {
```

That is the scalar tail of `map_span`, whose own comment (`gen_pfa_small.c:288`)
reads *"n need not be a multiple of 8"* — i.e. this is precisely the code path
that serves batches not divisible by 8, via the entry's documented `B%8` split
path. `n` is an unbounded `ptrdiff_t`, so GCC proves the loop could index past
any object (2^61−1 = PTRDIFF_MAX/8) and is thereby licensed to assume it exits
early. **All four acceptance sizes this entry owns run at B ∈ {64, 64, 32, 32} —
every one a multiple of 8 — so the warned path is not exercised by any scored
case in this round.** It passed correctness because the code GCC was told it may
mangle never ran. This is a latent, not an actual, defect; it is also exactly the
kind of thing round 6 finds, since round 6 draws (L, B) that nobody announced.
Fix the bound, don't re-measure and shrug.

### 3.5 Crashes, hangs and timeouts
`failures.txt` contains exactly three lines, all the same cell:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout 600` (sweep.sh:88). This is the harness's library-free
reference floor, not a panel contender, and the timeout is arithmetic rather than
a bug: `baseline_matrix` is O(L⁴)/volume/axis, it already runs 33,904 µs per
transform at L=50, and the L=100 chain is m=64 — the 600 s wall cannot be met.
The evidence that it is a timeout and not a fault: the same binary at the same
size **passes both correctness gates** in the shorter m=2 pass
(`o_baseline_matrix_L100_B1.json`: single-call 9.40e-16, two-step 4.86e-15). It
is therefore the only cell in the round with no timing and no `c_*.json`, and
L=100's leaderboard section correctly shows 13 rows rather than 14.

`timing.err` was scanned line by line: **every** non-blank line is a clean
`<entry>: does not support L=<n>` decline. Zero segfaults, zero SIGILLs, zero
aborts, zero unexpected diagnostics in the whole round.

### 3.6 Missing entries — checked against the roster, not against the leaderboard
The `does not support` declines are what `PANEL_BRIEF.md` *requires* ("must
`support()` exactly the acceptance sizes your class covers (declining the
rest)"), so absence from a leaderboard section is only a fault if the size is in
the entry's own column. Cross-checking the brief's roster table against the 152
result cells:

| entry | acceptance sizes owned | present | verdict |
|---|---|---|---|
| `gen_pfa_small` | 10, 12, 15, 20 | 10, 12, 15, 20 | complete |
| `gen_pfa_large` | 40, 50, 100 | 40, 50, 100 | complete |
| `gen_pow2` | 32 | 32 | complete |
| `gen_powp` | 25, 27, 50, 100 | 25, 27, 50, 100 | complete |
| `gen_dense_prime` | 31 (+ helps 10..20) | 31, 10, 12, 15, 20 | complete, over-delivered |
| `gen_rader` | 31 | 31 | complete |
| `gen_bluestein` | none scored; **must run everywhere** | all 11 | complete |
| `gen_batchlane` | 10, 12, 15 at B≥8 | 10, 12, 15 | complete |
| `gen_planner` / `gen_race` / `gen_twiddle` / `gen_layout` | all (layers) | all 11 each | complete |

**No entry is missing a size it owns.** No entry is missing a strategy record
(twelve `strategies/*.md`, all updated during the round). One record-quality
complaint, which bears on §4: `gen_pfa_large.md`'s headline table reports L=40 at
B=8 and B=1 but its "Raw execute (no chain)" line quotes only B=1 numbers, so
its most-cited figure (136.6 µs) is not comparable to any graded cell — a reader
could easily mistake it for a chained result. Fixed by labelling, not by work.

### 3.7 Plausibility check on the winners (is anything too fast to be real?)
`leaderboard.py:110` computes GF/s against a nominal 5·V·log₂(V)·B·m yardstick.
Single-core peak on this part is 2 × 512-bit FMA × 8 doubles × 2 flop × 2.90 GHz
= **92.8 GF/s**. The fastest cell in the round, `gen_batchlane` at L=12, is
46.57 GF/s = **50.2% of peak**; `gen_pow2` at L=32 is 41.6%. Nothing exceeds
peak, nothing is close enough to peak to be suspicious, and the twiddle-free PFA
entries sitting highest is the expected ordering. No entry's speed requires
explaining away.

### 3.8 Plan-time budget (60 s cold / 50 ms warm)
Largest `setup` of any panel entry in the round: `gen_powp` at L=100,
**4.163 s** — 14x inside the 60 s cold budget. Next: `gen_pfa_large` L=100
3.347 s, `gen_race` L=40 1.954 s. Ten of the eleven `gen_planner` cells are
≤ 1 ms. No violations. (For scale, `fftw3_patient` needs 17.575 s at L=100; it
is a library and exempt.) **The ≤ 50 ms warm/wisdom budget is essentially
untested**, because the only entry with a wisdom cache is the one nobody adopted:
`results/wisdom_a80n0.json` has 12 entries, all `gen_race/exec/...`, and the only
visible warm hit is `gen_race`'s 0.004 s setup at L=100.

---

## 4. Claimed versus measured

Because of §0.3 — the implementers measured on the scoring node itself, through
leased cores — **claimed and measured agree to within a few percent essentially
everywhere, and there is nothing for the machine difference to explain.** The
strongest case is `gen_planner`, which published all eleven cells:

| case | `gen_planner.md` claim (µs) | measured (µs) | Δ |
|---|---|---|---|
| L=10 B=64 | 8.59 | 8.594 | +0.0% |
| L=12 B=64 | 14.14 | 14.145 | +0.0% |
| L=15 B=32 | 29.26 | 28.937 | −1.1% |
| L=20 B=32 | 62.12 | 61.658 | −0.7% |
| L=25 B=16 | 137.4 | 137.546 | +0.1% |
| L=27 B=16 | 216.9 | 216.159 | −0.3% |
| L=31 B=16 | 609.4 | 606.815 | −0.4% |
| L=32 B=8 | 307.1 | 306.017 | −0.4% |
| L=40 B=8 | 669.8 | 666.323 | −0.5% |
| L=50 B=4 | 1543.7 | 1535.440 | −0.5% |
| L=100 B=1 | 15266 | 15044.6 | −1.5% |

And the class winners:

| entry | claim | measured | Δ | note |
|---|---|---|---|---|
| `gen_rader` L=31 | 94.3 | 94.170 | −0.1% | exact |
| `gen_pow2` L=32 | 63 | 63.683 | +1.1% | exact |
| `gen_dense_prime` L=31 | 177.9 | 175.591 | −1.3% | exact |
| `gen_powp` L=25/27/50/100 | 50.0 / 66.1 / 478 / 5000 | 48.778 / 66.078 / 473.484 / 5026.755 | −2.4% / 0.0% / −0.9% / +0.5% | exact |
| `gen_pfa_large` L=40/50/100 | 206.7 / 497.0 / 5247.6 | 200.515 / 481.372 / 5086.365 | −3.0% / −3.1% / −3.1% | **under**-claimed |
| `gen_pfa_small` L=12/15/20 | 2.50–2.72 / 6.44–6.56 / 17.05–17.40 | 2.500 / 6.178 / 16.920 | at / −4.1% / −0.8% | **under**-claimed at 15 |
| `gen_batchlane` L=15 | 4.64–4.77 typical | 4.686 | in range | exact |

Every divergence larger than 1.5% is in the *conservative* direction: the entry
measured **faster** than its own record claimed. Not one entry over-claimed.
Two apparent discrepancies are units mismatches rather than claim failures, and
should not be read as over-claims:

- `gen_bluestein.md`'s "~29 Gflop/s real on one Ice Lake core" against a
  leaderboard 3.18–5.97 GF/s: the leaderboard yardstick is *nominal* 5·V·log₂V,
  while Bluestein's chirp-Z does several times that many real flops (its
  power-of-two convolution length exceeds L). The two numbers are consistent.
- `gen_pfa_large.md`'s "136.6 µs (40 B=1)" against a measured 200.515 at L=40:
  that claim is **raw execute, no chain, B=1**; the scored cell is chained at
  B=8. The same record makes the gap explicit — *"L=40 B=1 raw 136.6 µs but
  chained 377"* — and its own chained claim (206.7) matches the measurement to
  3%. This is the labelling problem from §3.6, not a discrepancy.

**Where a machine difference *does* exist, one implementer measured it, and it is
not the gap the brief describes.** `gen_powp.md` reports the same source built and
run on both hosts: dev host (wallaby) **38.8 / 47.8 / 343 / 3296 µs** at L=25/27/
50/100 versus node **50.0 / 66.1 / 478 / 5000** — the dev host is **1.29x–1.52x
faster**, not 2.9x. The 2.9x MKL span quoted in the brief is a Cascade-Lake–to–
Sapphire-Rapids figure and does not describe this round, whose scoring node is Ice
Lake (§0.2) and whose development happened on leased cores of that same node
(§0.3). Nothing in this round needs attributing to hardware.

The genuinely important part of that datum is not the ratio but the inversion
`gen_powp` records alongside it: *"wallaby INVERTS several picks (f* wins 25/27
there too, but ipf wins 50 and ip1 wins 100) — the per-host race is doing real
work; do not hardcode a family."* The winning **variant** differs between hosts at
two of four sizes. That is a direct, measured justification for `gen_race`'s
wisdom cache — the layer that scored zero adoption (§2) — and for the r2
cross-architecture check.

### 4.1 The m-calibration duty (this round owned it)
`cases.txt` and `CAMPAIGN_GENERALIZE.md` both charge the round-1 monitor with
calibrating m once against measured MKL (~0.4 s of MKL chain per case) and then
freezing the suite. Measured MKL 2022 per-call (= chain) seconds as run, and the
m that would hit 0.4 s exactly:

| L | m as run | MKL chain (s) | m for 0.4 s | winner's chain (s) |
|---|---|---|---|---|
| 10 | 1000 | 0.291 | 1374 | 0.075 |
| 12 | 600 | 0.297 | 807 | 0.077 |
| 15 | 600 | 0.316 | 760 | 0.090 |
| 20 | 256 | 0.475 | 216 | 0.139 |
| 25 | 256 | 0.495 | 207 | 0.200 |
| 27 | 200 | 0.462 | 173 | 0.211 |
| **31** | **140** | **1.900** | **29** | 0.211 |
| 32 | 250 | 0.343 | 292 | 0.127 |
| 40 | 128 | 0.416 | 123 | 0.205 |
| 50 | 128 | 0.486 | 105 | 0.242 |
| 100 | 64 | 0.500 | 51 | 0.322 |

Ten of eleven cases are already within 1.4x of the 0.4 s target — the provisional
m values were well chosen. The exception is **L=31, at 4.75x the target**, and it
is not an error in m: MKL collapses to 2.61 GF/s at the prime, so the same m buys
1.9 s of MKL chain.

**Recommendation, deliberately not executed:** freeze the suite **exactly as
run** and do not rescale L=31. Rescaling it to m=29 would shrink the *winner's*
chain from 211 ms to 44 ms, and at that length the ≤ 2.6% spreads this round
achieved are no longer credible — an MKL-anchored m is the wrong anchor precisely
where we are 7.7x faster than MKL. The better fix, if Will wants one, is to
anchor m to the *best library* rather than to MKL. Either way this changes a
number that all six rounds and the r6 acceptance test depend on, so it is
recorded here as a monitor recommendation for Will's decision rather than written
into `cases.txt` by me.

Timing quality as run supports freezing: run spread ≤ 4.3% for every panel entry
(worst `gen_pfa_small` L=20 at 4.3%; the class winners are 0.4–2.8%), against
library spreads reaching 8.0% (`fftw3_patient` L=27) and 7.9% (`fftw3_measure`
L=100).

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### Primary: §4.3, "Is axis fusion worth 3x or 3%?" — moved, in the regime §4.3 named as untested
§4.3's standing answer (from `panel_r3/VERDICT.md` §5) was *single-digit percent,
sometimes negative*, and its addendum is explicit that every panel experiment so
far fused across an **L1↔L2** boundary, where the bandwidth gap is 2.6x, while
the untested case is **L2↔DRAM** at a 7x gap — and that the recommended
construction there is: *"tile the batch so a tile fits L2, then run all three
axes inside the tile. That is not the same experiment the panel ran, and it is
the largest untried structural move on the board."*

**This round ran that experiment, twice, and it paid.**

1. `gen_rader` (`gen_rader.md`) made state + c **L2-resident at 953 KB** against
   this part's 1.25 MiB L2, and measured **110.4 → 94.3 µs/step, −16 µs =
   −14.5%** at L=31, B=16 — an order of magnitude more than the L1↔L2 single
   digits, and in the direction Intel's manual, Alappat et al. and the L3-Fusion
   result all predicted. It is the reason `gen_rader` is the round's largest
   margin (7.70x).
2. The counter-experiment is in the same round and confirms the mechanism by its
   failure. `gen_pfa_small` at L=20 has Q+CQ = **2 MB against a 1.25 MiB L2**, so
   c streams from L3 every step — the *unfixed* version of `gen_rader`'s
   condition. Its attempt to paper over it with software prefetch instead of
   tiling cost **+14%** (19.5–20.1 µs vs 17.05–17.5 control). And L=20 is exactly
   where the round's GF/s curve breaks: 42.4 / 46.6 / 42.2 GF/s at L=10/12/15,
   then **30.7** at L=20. Same engine family, same layout, same codelet style —
   the only thing that changed is that the working set left L2.

So §4.3's addendum is confirmed on the L2↔DRAM side: it is worth ~14%, not ~3%,
and the lever is residency, not pass count. **Tolmachev's rule survives with a
second set of CPU numbers attached.**

A secondary, weaker datum on §4.3's original pass-count question: `gen_batchlane`
(fused L1 zy-sweep + x-pass) versus `gen_pfa_small` (three separate in-place
passes) at identical L and B, both twiddle-free PFA on split-complex SoA-8 —
**+21% / +25% / +32%** at L=10/12/15 for the three-pass form. That is far above
the r3 single digits, but it is **not** a clean A/B (different codelets, different
slot algebra, different map placement), so it should be logged as suggestive and
settled by one implementer building both forms of the same kernel, not treated as
a measurement.

### Secondary: §4.2, "dense-symmetric vs Rader" — moved, transposed from p=17 to p=31
§4.2's open question (a) is *"which of dense-symmetric and Rader wins on this
hardware, batch-vectorised"*, with §02 §7 on record that Rader *"is not the lever
at L=17."* This round fought that exact crossover at **p=31** with both kernels
written, correct, batch-vectorised at B=16, and both beating every library:

- `gen_rader` **94.170 µs** — 7.70x over the best library
- `gen_dense_prime` **175.591 µs** — 4.13x over the best library
- **Rader wins by 1.86x.**

And it wins *via* §4.2's open question (b) — the structural suggestion §01 made
and nobody had tested. `gen_rader`'s description is precisely the
symmetric/antisymmetric split: *"conjugate fold → cyclic-15 (cos) +
negacyclic-15 (sin; odd-N sign-twist to cyclic), each via Winograd-C3 × dense-C5
(100 FMA + 65 add)."* So (b) is answered affirmatively: the fold is worth having,
on top of Rader, at a real prime, with the number attached.

**What this does not settle**, and the record must say so: 31 is not 17. Rader-31
folds to a cyclic length of 15 = 3×5, which decomposes beautifully; Rader-17
folds to 8 = 2³. The crossover is now measured for primes whose p−1 is smooth and
odd-after-folding; the p=17 case and the p=29 case remain open, and `PANEL_BRIEF`
already schedules *"dense-vs-Rader crossover measured (p = 29, 31 both ways)"*
for r2–r3. Half of that assignment is done.

### Also moved, incidentally: §4.8 item 6 (AVX-512 measurement)
§4.8 item 6 closes with *"there is no primary measurement in the corpus for Ice
Lake-SP or later server parts, which is the hardware most likely to be in a
current LQCD cluster. **Measure it on the node.**"* This round is 152 gate-passing
cells of primary Ice Lake-SP (Gold 6326) single-core AVX-512 measurement across
eleven sizes and six library baselines, with the peak-fraction accounting in
§3.7. That gap is now closed for this microarchitecture and should be written up
into §4.8 rather than left as an open item.

### Not moved
§4.1 (AVX2 register spill at L=6/8) — out of scope, AVX2 and those sizes are not
in this campaign. §4.4 (split vs interleaved) was already closed by §08 §5.4;
this round is consistent with it — every class winner ships split-complex, and
the one that does not (`gen_rader`, "all-real constants on interleaved zmm") is
interleaved only in its constant tables. §4.5 (L=8 padding) — no L=8 case; though
`gen_batchlane`'s shipped "plane stride 256 mod 4096" and `gen_pfa_large`'s
odd-cache-line pitches are §4.5's rule in production, unmeasured here. §4.7
(vector-radix) — nobody tried it; correctly.

---

## 6. The single highest-value thing the next round should attack, per geometry

Ranked by measured headroom, taken from the round's own numbers and the
implementers' own floor estimates.

| geometry | attack this | the number that says so |
|---|---|---|
| **L=10, 12, 15** | **The B<8 path.** The winner is a batch-lane engine that wastes 7 of 8 lanes at B=1 and is **2.3–2.5x slower than MKL** there (10.6/18.0/41.3 vs 4.5/7.4/16.3 µs). Either give it a per-volume split path or make the planner dispatch away from it below B=8. | This is the only place in the round where we lose to a library, and r6 draws unannounced (L, B). |
| **L=20** | **L2 residency.** Q+CQ = 2 MB against 1.25 MiB L2; c streams from L3 every step. Apply `gen_rader`'s batch/tile-to-L2 construction (worth −14.5% at L=31), *not* prefetch (measured **+14%**, already reverted — do not rediscover). Fix the `map_span` UB warning (§3.4) in the same pass. | 30.65 GF/s at L=20 against 42.2–46.6 at L=10/12/15 with the same engine family: ~1.4x of pure residency loss. |
| **L=25, 27** | **Close the twiddled-CT penalty.** The prime-power CT path — the campaign's declared centre of gravity — runs at 22.31 / 21.25 GF/s where the twiddle-free 2^k path gets 38.59 at L=32. That is ~55% of the achievable rate for the same class of work. `gen_powp`'s own lever: state layout so each phase touches longer contiguous spans. | 1.73x between the twiddled and twiddle-free rates on the same node. |
| **L=31** | **Finish the crossover, then close the port gap.** (a) p=29 was never measured; the brief assigns "29 and 31 both ways" to r2–r3 and only 31 is done. (b) `gen_rader`'s own port-floor estimate is 65–70 µs against 94.170 shipped — ~25–30 µs, of which it attributes 8–10 µs to killing the extra state write per step. (c) `gen_dense_prime`, 1.86x behind, has 2 × ~50 µs of issue gap in its fold passes by its own accounting. | 1.34–1.45x remaining against a self-derived floor, on the round's biggest win. |
| **L=32** | **Port efficiency against its own ceiling.** `gen_pow2` puts this structure's ceiling-of-merit at ~45 µs and shipped 63.683. Then extend to the unscored 16/64/128 — r6 can draw any of them and only 32 is exercised today. | 1.41x from its own floor; three of four sizes in the class unmeasured. |
| **L=40** | **Fuse the map into the last sweep.** The chain plumbing, not the codelets, is the cost: a full-volume map pass plus a memcpy per unit takes raw 136.6 µs to 377 chained at B=1 (200.515 at B=8). `gen_pfa_small` and `gen_rader` already fuse; copy them. Codelet floor is separately estimated at 50–60 µs. | 2.8x between raw and chained at B=1 is pure plumbing. |
| **L=50, 100** | **Bandwidth, not algorithm.** The top two entries are within **1.2–1.7%** of each other (`gen_powp` 473.484 / 5026.755 vs `gen_pfa_large` 481.372 / 5086.365) across two structurally different decompositions — so the decomposition is not the lever. L=100 is the round's worst margin (**1.56x**) and its memory-pressure case (30.5 MiB vs 54 MB L3). Apply §4.3's L2↔DRAM tiling here, where the 7x gap actually bites. Also the round's largest plan time (4.163 s). | Two different algorithms landing within 1.2% is the signature of a bandwidth bound. |

**And one cross-cutting item that outranks every row above, because r5 and r6
depend on it: make the layers get adopted.** Four layers scored zero adoption in
r1 (§2). r2's first order of business is that each class winner actually
`#include`s `gen_twiddle`, `gen_layout` and `gen_race` — not because the layers
are faster today (they are not; their demo entries sit at the O(L⁴) floor) but
because r5 assembles `fft3d_general` from parts that at present share no line of
code, and r6 scores *only* that trunk, with a zero for any drawn size that misses
a gate or busts the plan budget. Related and unmeasured: the ≤ 50 ms warm/wisdom
budget has essentially no evidence behind it (§3.8), and `gen_powp`'s measured
host-inversion of the winning variant at 2 of 4 sizes (§4) says the race is not
optional.

---

## 7. What to keep

Applying `docs/CURATION.md`'s four grounds, in order.

**Ground 1 — the fastest correct entry for each geometry, one per L, always.**
Six entries cover all eleven acceptance sizes:

- `gen_batchlane` — L=10, 12, 15 (3.87x / 3.88x / 3.51x)
- `gen_pfa_small` — L=20 (2.65x)
- `gen_powp` — L=25, 27, 50, 100 (2.23x / 2.19x / 2.00x / 1.56x)
- `gen_rader` — L=31 (7.70x)
- `gen_pow2` — L=32 (2.69x)
- `gen_pfa_large` — L=40 (2.02x)

**Ground 2 — a structurally different runner-up when it is close.**

- **L=50 and L=100 are dead heats**: `gen_pfa_large` is 1.7% and 1.2% behind
  `gen_powp`, via a genuinely different axis decomposition (twiddle-free PFA
  25×2 versus twiddled CT 25×4). Both are already in on ground 1 — and the
  near-tie across two different structures is itself the §6 finding that the
  bound is bandwidth.
- **L=31 is the case CURATION's own example was written for**: the winner is
  Rader and `gen_dense_prime`, the dense conjugate-symmetric alternative, is the
  runner-up. At 1.86x it is outside the "~20%" bar, but it is the other half of
  `LITERATURE.md` §4.2's crossover, the brief schedules the rematch at p=29 for
  r2–r3, and it beats the best library by **4.13x** on its own. The next panel
  needs it written down, not described. **Keep `gen_dense_prime`.**
- L=10/12/15: `gen_pfa_small` is 21–32% behind `gen_batchlane` and structurally
  distinct (three passes vs fused zy-sweep — the §4.3 pass-count pair). Already
  in on ground 1.

**Ground 3 — instructive entries whose record documents the number that killed
them.**

- **`gen_planner`** — never wins a size, and that is the point. It is the only
  single generic engine that ran all eleven acceptance sizes *and* L=2..128 (worst
  residual 1.1e-15 at L=113) at 5.8–8.4 GF/s with ≤ 1 ms setup, it beat the best
  library at L=31 (606.815 vs 725.494), and its record carries the
  candidate-ordering measurements the next round would otherwise pay for again:
  Rader-31 551 vs Bluestein-64 906 vs dense-31 1376; gt-PFA losing 23% to CT in
  this executor (574 vs 468 at L=40); PLN_TI=8 losing 6–15% everywhere. It is
  also the backbone r5's trunk is assembled from. **Keep.**
- **`gen_bluestein`** — the mandated existence fallback, the only entry besides
  the layers that ran and gate-passed at all eleven sizes, and the entry that
  quantifies the literature's warning. `LITERATURE.md` §07 §6.3 warns of a
  107–1315x Bluestein penalty; measured here it is **4.1–13.3x** off the class
  winner, and at L=31 it **beats the best library by 1.79x** (405.309 vs
  725.494). A fallback that outruns MKL and ducc0 at a prime is a load-bearing
  fact for r6, where any unplannable size scores zero for the whole library.
  **Keep.**

**Ground 4 — anything that beat a library baseline, regardless of rank.** This
fires for every entry above, and for one more: `gen_layout` beat ducc0 at L=31
(466.299 vs 725.494). **Declining that one**, with the reason on the record: by
its own description that entry is an *"any-L dense matrixsimd demo of the
layer"* — an O(L⁴) floor-class demo — and its L=31 rank measures how badly every
library handles the prime (all six are 2.51–3.05 GF/s there), not how good the
layout layer is. Since no class entry adopted the layer, there is no measured
layout result to preserve; promoting the demo would put a dense O(L⁴) kernel in
the exemplar reading list on the strength of an artifact. Same reasoning declines
`gen_race` (32–78x off the winner, never beats a library, deliberately an O(L⁴)
test bench) and `gen_twiddle` (5.3–15.4x off, 1120.143 vs 725.494 at L=31 — it
does not clear the bar at all).

Declining those three costs nothing: `CURATION.md` tracks `impl_<N>/` in full and
`strategies/*.md` unconditionally, so all four layers' code and records survive
in `impl_1/` and `strategies/` regardless of promotion. What they do not get is a
place in `exemplars/gen_r1/`, and the signal that sends to r2 is the correct one —
**the layers earn their exemplar slot by being adopted, which is r2's stated
first priority (§2, §6).**

**Not promoted, for completeness:** `baseline_matrix` (harness floor, not a
contender; timed out at L=100 as §3.5 documents), `gen_race`, `gen_layout`,
`gen_twiddle` (above).

**One condition on `gen_pfa_small`'s promotion**, to be recorded in
`exemplars/gen_r1/NOTES.md`: the four `-Waggressive-loop-optimizations` UB
warnings at `gen_pfa_small.c:324` (§3.4) sit in its `B%8` tail path, which no
scored case in this round exercised. It is promoted for its L=20 win and its
three-pass structure, and the next panel must be told that the tail bound is
unfixed before anyone copies the file.

---

### Round summary

Eleven never-before-tuned sizes; eleven wins over the best of MKL 2022, MKL 2026,
FFTW3 ×3 planners and ducc0; geomean **2.82x**, worst case **1.56x**, best case
**7.70x**. All three gates pass in 152/152 and 153/153 cells with 6–36x margin on
the precision contract. Zero build errors, zero crashes, zero missing roster
coverage, one timeout (the O(L⁴) reference floor at L=100, expected). Not one
implementer over-claimed. `CAMPAIGN_GENERALIZE.md`'s stated expectation —
4–10x on prime-heavy sizes, 1.5–3x on smooth/large — is met at every case
(L=31: 7.70x; L=40/50/100: 2.02x/2.00x/1.56x), and r1's actual charge, which was
*"every class entry gets its acceptance sizes RUNNING and gate-passing (speed
secondary)"*, is met with the speed thrown in.

The round's two failures are both structural rather than numerical: **all four
library layers scored zero adoption**, and **B=1 is a hole** — the small-L winner
loses to MKL there by 2.3–2.5x. Both are r2 work, and r6 will price them.
