# VERDICT — panel round d1_r2

Monitor report. Measured 2026-09-03T02:33 on `a80n0.lqcd.mit`, slurm job 440424, seed
20262847, 3 runs × 12 samples per cell, 52 cells (13 sizes × 4 regimes).

---

## 0. Two corrections to the brief I was given, before anything else

**(a) The geometries are not L = 6, 8, 17, 36.** Those are the *3D cube* campaign's sizes
(`bench/geom`, `docs/LITERATURE.md`). This is the 1D campaign (`bench/d1`,
`docs/literature_1d/00-SURVEY.md`), whose case list is thirteen 1D lengths:

> 13, 31 (small primes) · 32, 64, 128 (small pow2) · 60 (composite 4·3·5) ·
> 1024, 4096, 16384 (large pow2) · 1021, 10007, 65537, 100003 (large primes)

I have reported on the sizes that were actually measured. Nothing named 6, 8, 17 or 36
exists in `cases.txt`, in `impl/`, or in either leaderboard.

**(b) The scoring machine is not a Cascade Lake Xeon Gold 5218.** `environment.txt` records
**Intel Xeon Gold 6326 — Ice Lake-SP**, 2.90 GHz base / 3.5 GHz max turbo, **1.25 MB L2 per
core**, two 512-bit FMA units, governor `schedutil`. That matters for §4 below, because the
port-balance and L2-capacity conclusions differ from Cascade Lake, and because the
turbo ratio 3.5/2.9 turns out to be the dominant error term in this round (§3.1).

Development machine, per every strategy record: **wallaby, Xeon Gold 6448Y (Sapphire
Rapids), 2 MB L2**, `schedutil` swinging 2.1–4.1 GHz. So the dev/score gap is
SPR → ICL, not SPR → CLX. The measured MKL span between the two machines in this round's
cells is **1.5–1.8×**, not 2.9× (§4).

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

Library set = FFTW 3.3.10 `{estimate, measure, patient}` + oneMKL DFTI, per PANEL_BRIEF.
(`fftw1d_custom*` are genfft codelets compiled in our harness, not stock library calls, so
they are excluded from "library" here; `baseline_dft` is the harness floor.) Batched B is
512 at L ≤ 1024, then 256 / 64 / 16 / 8 as L grows. Ratio > 1 means the panel is ahead.

`win` / `loss` = the verdict survives the ±20.7 % clock-state band established in §3.1.
`tie*` = **the verdict does NOT survive it** — the cell is undecided by this measurement.

| L | regime | best panel | µs/xform | best library | µs/xform | ratio | |
|---|---|---|---|---|---|---|---|
| 13 | B=1 m=1 | d1_batchlane | 0.0229 | fftw-patient | 0.0219 | 0.96x | tie* |
| 13 | B=1 m>1 | d1_race | 0.0394 | fftw-measure | 0.0674 | 1.71x | **win** |
| 13 | B=512 m=1 | d1_prime | 0.0206 | fftw-measure | 0.0140 | 0.68x | **loss** |
| 13 | B=512 m>1 | d1_prime | 0.0154 | fftw-patient | 0.0513 | 3.33x | **win** |
| 31 | B=1 m=1 | d1_prime | 0.0643 | MKL | 0.2795 | 4.35x | **win** |
| 31 | B=1 m>1 | d1_prime | 0.0582 | MKL | 0.3194 | 5.49x | **win** |
| 31 | B=512 m=1 | d1_race | 0.0539 | fftw-estimate | 0.2634 | 4.89x | **win** |
| 31 | B=512 m>1 | d1_prime | 0.0483 | MKL | 0.3153 | 6.53x | **win** |
| 32 | B=1 m=1 | d1_race | 0.0205 | MKL | 0.0254 | 1.24x | **win** |
| 32 | B=1 m>1 | d1_batchlane | 0.0791 | MKL | 0.1310 | 1.66x | **win** |
| 32 | B=512 m=1 | d1_race | 0.0154 | MKL | 0.0153 | 0.99x | tie* |
| 32 | B=512 m>1 | d1_race | 0.0384 | MKL | 0.1122 | 2.92x | **win** |
| 60 | B=1 m=1 | d1_composite | 0.0534 | MKL | 0.0622 | 1.16x | tie* |
| 60 | B=1 m>1 | d1_batchlane | 0.1235 | fftw-measure | 0.2332 | 1.89x | **win** |
| 60 | B=512 m=1 | d1_race | 0.0516 | MKL | 0.0433 | 0.84x | tie* |
| 60 | B=512 m>1 | d1_race | 0.0578 | MKL | 0.2280 | 3.94x | **win** |
| 64 | B=1 m=1 | d1_race | 0.0483 | MKL | 0.0427 | 0.88x | tie* |
| 64 | B=1 m>1 | d1_race | 0.1358 | MKL | 0.2369 | 1.74x | **win** |
| 64 | B=512 m=1 | d1_race | 0.0372 | MKL | 0.0383 | 1.03x | tie* |
| 64 | B=512 m>1 | d1_race | 0.0783 | MKL | 0.2365 | 3.02x | **win** |
| 128 | B=1 m=1 | d1_race | 0.0988 | MKL | 0.0911 | 0.92x | tie* |
| 128 | B=1 m>1 | d1_race | 0.2130 | MKL | 0.4756 | 2.23x | **win** |
| 128 | B=512 m=1 | d1_race | 0.1562 | MKL | 0.1433 | 0.92x | tie* |
| 128 | B=512 m>1 | d1_race | 0.2305 | MKL | 0.5381 | 2.33x | **win** |
| 1021 | B=1 m=1 | d1_rader | 7.788 | MKL | 9.413 | 1.21x | **win** |
| 1021 | B=1 m>1 | d1_rader | 7.851 | MKL | 11.40 | 1.45x | **win** |
| 1021 | B=256 m=1 | d1_rader | 7.604 | MKL | 10.02 | 1.32x | **win** |
| 1021 | B=256 m>1 | d1_race | 7.737 | MKL | 12.38 | 1.60x | **win** |
| 1024 | B=1 m=1 | d1_pow2 | 1.366 | MKL | 1.079 | 0.79x | **loss** |
| 1024 | B=1 m>1 | d1_pow2 | 2.726 | MKL | 4.165 | 1.53x | **win** |
| 1024 | B=512 m=1 | d1_race | 1.821 | MKL | 1.624 | 0.89x | tie* |
| 1024 | B=512 m>1 | d1_race | 2.747 | MKL | 4.932 | 1.80x | **win** |
| 4096 | B=1 m=1 | d1_pow2 | 9.120 | MKL | 6.000 | 0.66x | **loss** |
| 4096 | B=1 m>1 | d1_twiddle | 14.27 | MKL | 18.34 | 1.29x | **win** |
| 4096 | B=256 m=1 | d1_pow2 | 13.50 | fftw-patient | 10.38 | 0.77x | **loss** |
| 4096 | B=256 m>1 | d1_pow2 | 13.05 | MKL | 22.94 | 1.76x | **win** |
| 10007 | B=1 m=1 | d1_bluestein | 184.3 | fftw-patient | 198.7 | 1.08x | tie* |
| 10007 | B=1 m>1 | d1_bluestein | 199.2 | fftw-patient | 228.4 | 1.15x | tie* |
| 10007 | B=64 m=1 | d1_bluestein | 195.8 | fftw-patient | 208.5 | 1.07x | tie* |
| 10007 | B=64 m>1 | d1_planner | 205.5 | fftw-patient | 238.9 | 1.16x | tie* |
| 16384 | B=1 m=1 | d1_race | 44.76 | fftw-patient | 32.14 | 0.72x | **loss** |
| 16384 | B=1 m>1 | d1_pow2 | 69.61 | fftw-patient | 82.34 | 1.18x | tie* |
| 16384 | B=64 m=1 | d1_race | 63.32 | fftw-patient | 44.74 | 0.71x | **loss** |
| 16384 | B=64 m>1 | d1_pow2 | 68.16 | fftw-patient | 99.27 | 1.46x | **win** |
| 65537 | B=1 m=1 | d1_race | 1157 | fftw-patient | 1468 | 1.27x | **win** |
| 65537 | B=1 m>1 | d1_race | 1047 | fftw-patient | 1629 | 1.56x | **win** |
| 65537 | B=16 m=1 | d1_race | 1502 | fftw-patient | 1544 | 1.03x | tie* |
| 65537 | B=16 m>1 | d1_rader | 1085 | fftw-patient | 1772 | 1.63x | **win** |
| 100003 | B=1 m=1 | d1_race | 3089 | fftw-patient | 2713 | 0.88x | tie* |
| 100003 | B=1 m>1 | d1_race | 3068 | fftw-patient | 3126 | 1.02x | tie* |
| 100003 | B=8 m=1 | d1_race | 3601 | fftw-patient | 2975 | 0.83x | **loss** |
| 100003 | B=8 m>1 | d1_bluestein | 3635 | fftw-measure | 3240 | 0.89x | tie* |

**Score: 27 robust wins, 7 robust losses, 18 cells undecided.** By regime:

| regime | robust win | undecided | robust loss |
|---|---|---|---|
| B=1, m=1 (benchFFT cell) | 4 | 6 | 3 |
| B>1, m=1 | 2 | 7 | 4 |
| B=1, m>1 (chained) | 10 | 3 | 0 |
| B>1, m>1 (chained) | 11 | 2 | 0 |

The brief's prediction holds exactly: **the chained regimes are won (21 of 26 cells, none
lost), the m=1 regimes are not (6 wins, 7 losses, 13 undecided).** Every robust loss is a
single-call cell, and all but one is at large pow2 (1024/4096/16384) or 100003 — the
sizes where FFTW's and MKL's tuning is deepest and where our fused-chain and lane-fill
levers do not apply.

---

## 2. What changed since d1_r1, per geometry

Same host, same seed, same case list, same harness for both rounds, so the comparison is
clean at the level the measurement supports. Read this section together with §3.1.

| L | direction | detail |
|---|---|---|
| **13** | mixed / net worse | `d1_prime` targeted the two m=1 cells it lost in r1 and **lost ground in both**: B=512 m=1 ratio 0.84x → **0.68x**, B=1 m=1 1.01x → 0.96x. Chained cells unchanged (1.71x, 3.33x). |
| **31** | mixed | chained cells flat-to-better (B=512 chain 5.77x → 6.53x). `d1_prime` B=1 m=1 regressed 0.0527 → 0.0643 but still wins 4.35x — the cell was never in doubt. |
| **32** | better | batched chain **1.85x → 2.92x** (`d1_race` picking `d1_batchlane`'s SoA path, +58 %). B=1 chain 1.54x → 1.66x. m=1 cells flat. |
| **60** | better | batched chain 3.35x → 3.94x; B=1 chain 1.75x → 1.89x; B=1 m=1 0.89x → 1.16x (`d1_composite`'s full-unroll fix, the only m=1 cell recovered this round). Batched m=1 still 0.84x. |
| **64** | much better | batched chain **1.88x → 3.02x** (+61 %); B=1 chain 1.59x → 1.74x. m=1 cells flat. |
| **128** | much better | B=1 chain **1.62x → 2.23x** (+38 %); batched chain 1.81x → 2.33x; B=1 m=1 0.84x → 0.92x. |
| **1021** | transformed | all four cells taken, from two losses. B=1 m=1 **0.60x → 1.21x** (13.74 → 7.79 µs); B=256 m=1 0.67x → 1.32x. This is the round's largest single gain, and it is `d1_rader`'s intrinsic radix-17. |
| **1024** | better | B=1 chain 1.21x → 1.53x; batched chain 1.59x → 1.80x. m=1 cells unchanged (still 0.79x / 0.89x). |
| **4096** | slightly better | chains +5–15 %; the three m=1 losses are unmoved (0.66x / 0.77x). |
| **10007** | slightly better | all four cells improved 4–6 %, ratios 1.07–1.16x — but every one is inside the clock band, so 10007 is **not** demonstrably won. |
| **16384** | better | B=64 m=1 0.62x → 0.71x, B=1 m=1 0.64x → 0.72x, chains +3–6 %. Still two robust losses. |
| **65537** | **REGRESSED** | both chained cells went backwards: B=1 chain 910 → 1047 µs (**ratio 1.79x → 1.56x**), B=16 chain 949 → 1085 µs (**1.86x → 1.63x**). The m=1 cells improved slightly. |
| **100003** | better | B=1 m=1 0.75x → 0.88x, B=1 chain 0.83x → 1.02x. Still the panel's weakest size — three of four cells not won. |

### Regressions, named

1. **`d1_rader` at 65537, both chained cells, −13 %.** 910.4 → 1051.9 µs (B=1) and
   948.7 → 1084.9 µs (B=16). Their own record reports it honestly (`65537 chains 985/1028
   vs r1 910/949`) without explaining it. The round's work at 65537 went into `st17`/`st3`
   vectorisation aimed at 1021; the 65537 chain path lost. Spread is 0.1 % and 1.8 %, and
   `d1_race` regressed by the same amount at the same cells, so this is real, not noise.
2. **`d1_prime` at 13, both m=1 cells, −24 % and −29 %.** 0.0216 → 0.0286 (B=1) and
   0.0146 → 0.0206 (B=512). Unchanged code in the same cells moved −12 % to −17 %, so
   roughly half of this is clock state (§3.1) and roughly half is real. Either way the
   **ratio against the same-round library worsened**, 0.84x → 0.68x, in the one cell the
   entry explicitly set out to win. Attribution in §4.
3. **`d1_bluestein` at 65537, all four cells, −13 % to −31 %.** 2679.7 → 3868.3 µs at
   B=1 m=1, against a claimed *improvement* to 1836 µs on wallaby. This is an L2-capacity
   regression and it is the round's cleanest instructive failure — see §4.
4. **`d1_race` mis-picks (new this round).** `d1_race` is *slower than a sibling that was
   available to it* in 10 of 52 cells: 10007 B=64 m=1 258.0 vs `d1_bluestein` 195.8
   (**+32 %**), 4096 B=256 m=1 16.72 vs `d1_pow2` 13.50 (+24 %), 16384 B=1 chain 80.11 vs
   `d1_pow2` 69.61 (+15 %), 1021 B=1 chain 8.89 vs `d1_rader` 7.85 (+13 %), plus six
   smaller. The race layer's whole job is to pick the winner; at large L it is picking on
   too few samples (or from stale wisdom — `results/wisdom1d_a80n0.json` is dirty in git).
   Against that, it *beat every sibling* in 15 cells by up to 15 %, so the mechanism works
   where it is sampled adequately.

No geometry lost a cell it had previously won robustly. The two ratio reversals (13 B=1
m=1 1.01x → 0.96x, 13 B=512 m=1 0.84x → 0.68x) are both in cells that were already
undecided or lost.

---

## 3. Adversarial review: correctness, builds, crashes, omissions

### 3.1 The finding that dominates this round: an unrecorded 20.7 % clock-state error

`baseline_dft` is byte-identical between `impl_1` and `impl_2` (verified by `diff`), was
compiled by the same Makefile, and ran on the same host. It is **exactly 1.2070× slower in
r2 than in r1** at seven separate cells — 13 B=1 m=1, 13 B=512 m=1, 13 B=1 chain, 31 B=1
m=1, 31 B=512 m=1, 32 B=1 m=1, 32 B=512 m=1 — and 1.2072× at a eighth (60 B=512 m=1).

**3.5 GHz / 2.9 GHz = 1.20690.** That is the Gold 6326's max-turbo-to-base ratio. Sixteen
unchanged-code cell/backend pairs land in [1.200, 1.215]; the recurring "20.7 % / 20.8 %"
entries in the `run spread` column of *both* leaderboards are the same thing — a set of
three runs in which some landed on turbo and some on base, with the leaderboard reporting
the **min**, i.e. whichever run happened to catch turbo.

Consequences, all of which I have applied above:

- The error is **per process invocation and uncorrelated between backends within a round**
  — at 13 B=1 m=1, `fftw1d_estimate` moved −17.4 %, `fftw1d_patient` **+20.5 %**, and
  `mkl1d_dfti` 0.0 %, in the same cell of the same sweep. So it does not cancel in a ratio.
- It is confined almost entirely to the **m=1 cells**: median drift of unchanged code is
  −0.1 % in chained cells (which run ~0.1 s and settle to base clock) and reaches ±20.7 %
  in single-call cells. Median reported spread: 1.3 % chained vs 6.7 % single-call, p90
  15.9 % vs 26.9 %.
- **Any m=1 ratio between 0.83x and 1.21x is undecided by this measurement.** That is 18
  of 52 cells, including every "we nearly beat MKL" claim at 32/64/128/60 batched-single,
  all four of 10007, and three of four at 100003.
- `environment.txt` records the *governor* but never the *achieved clock*. The harness
  cannot currently detect this, and did not.

This is a measurement defect, not an implementer defect, and it is the single highest-value
thing to fix (§6).

### 3.2 Correctness

**One FAILED row in the entire round, and it is the harness's own floor, not a panel
entry:**

```
1024|non-batched, chain m=4000   baseline_dft   FAILED ch=1.9e-10/1e-10
```

The dense O(L²) reference exceeded the chained-gate tolerance at L=1024, m=4000 — expected
for a 1024×1024 matvec chained 4000 times, but it means **that cell has no validated
floor**. Worse, the gate is barely discriminating there for anybody: at 1024 B=1 m=4000 the
best margin any backend achieves is **9.1×** (`d1_twiddle`, 1.1e-11 vs 1e-10), with
`d1_pow2` at 6.2×, MKL at 5.6× and `fftw1d_estimate` at 4.2×. A gate with less than one
decade of headroom cannot distinguish a correct implementation from a slightly wrong one.
Every other cell in the round has ≥ 22× margin. **The 1024:1:4000 chain gate should be
re-anchored before the next round.**

Every panel entry passed everywhere else: single-call rel-L2 in 9.3e-17 … 1.4e-15 against
a 1e-12 gate; chained gates passed with 22× to 10⁵× margin. No fast-wrong-answer survived,
because there was no fast wrong answer.

### 3.3 Builds, crashes, missing entries

- **`build_errors.txt` contains no errors.** It is 40 lines of `-Wformat-truncation`
  warnings from `impl/d1_race.c` (`snprintf` into a 600-byte `so`/`tmp` path buffer at
  lines 991, 995, 1406). All nine backends built. Worth fixing — a truncated `.so` path in
  the race layer would silently mis-target a dlopen — but it did not bite this round.
- **No `failures.txt`** — nothing crashed or hung.
- **`agents/exits.txt`: all nine implementers exit=0.**
- **No entry is missing relative to r1.** The presence matrix is identical between rounds:
  every backend appears in exactly the same 52-cell subset it did in r1. `d1_prime` at
  13/31 only, `d1_composite` at 60 only, `d1_pow2` at pow2 only, `d1_rader` at 13/31/1021/
  65537 only — these are declared class scopes, not silent drop-outs.
- **No panel entry calls a library.** I grepped all ten sources for `fftw|mkl|DFTI|cblas|
  ipp`; the only hits are in `d1_race.c` and are false positives (the substring `ipp` in
  `skipped`). `d1_race`'s `dlopen` targets are `build/race1d/<host>/<label>_<hash>.so`,
  built from sibling panel sources — our own code throughout.
- **Setup cost is excluded from timing and should be read.** `d1_race` pays 11.4 s at
  13 B=1 m=1 and 2.3 s at 13 B=512 m=1 (it compiles and fork-gates arms); `fftw1d_patient`
  pays 67.9 s at 16384 B=64 and 44.7 s at 4096 B=256. Both are legitimate under the brief
  (timing is after compile and warmup), but a `d1_race` that needs 11 s of plan time to win
  a 0.023 µs transform is not a shippable production plan and the next round should say so.

### 3.4 One claim I could not corroborate

`d1_rader`'s record states the Ice Lake reservation **was live** during r2 development
(`icehold 440371 on a80n0`) and that all their numbers are from the scoring node. Every
other implementer states the same job was **dead** for their whole session. Both cannot be
literally true of the same job. I lean toward believing `d1_rader`, because their claimed
numbers reconcile with the measured leaderboard to within a few percent (§4) in a way
nobody else's do — but the panel should note that one implementer had node access others
did not, which is an unfair-round issue independent of who was right.

---

## 4. Claimed vs. measured, and what the machine explains

Eight of the nine implementers developed on **wallaby (SPR Gold 6448Y, 2 MB L2, 2.1–4.1 GHz
schedutil)** and scored on **a80n0 (ICL Gold 6326, 1.25 MB L2, 2.9/3.5 GHz)**. The measured
MKL span between the two, from the implementers' own same-core MKL runs against this
leaderboard, is **1.5–1.8×** (13 B=1: 0.015 → 0.0259; 32 B=1: 0.014 → 0.0254; 60 B=1:
0.036 → 0.0622; 128 B=512: 0.095 → 0.1433). Panel entries span **1.3–2.1×** over the same
gap. So the machine difference is real and large, but it is **not uniform** — and the
divergences below are exactly the cases where a panel entry's span exceeds MKL's.

| entry / cell | claimed (wallaby) | measured (a80n0) | span | verdict |
|---|---|---|---|---|
| `d1_composite` 60 B=1 m=1 | 0.033, beats MKL 0.036 | 0.0534 vs MKL 0.0622 | 1.62× | **carried** — win survived, 1.16x |
| `d1_composite` 60 B=512 m=1 | 0.032, beats MKL 0.034 | 0.0684 vs MKL 0.0433 | **2.14×** | **did NOT carry** — 0.63x loss |
| `d1_batchlane` 13 B=512 m=1 | 0.011, beats MKL 0.013 | 0.0230 vs MKL 0.0192 | 2.09× | did not carry |
| `d1_batchlane` 128 B=512 m=1 | 0.185 vs MKL 0.095 (1.95× behind) | 0.2462 vs 0.1433 (1.72× behind) | 1.33× | carried, as they predicted |
| `d1_prime` 13 B=512 m=1 | 30.4 GF/s, "projects to a clear win" | 11.67 GF/s, **0.68x loss** | — | **prediction inverted** |
| `d1_bluestein` 65537 B=1 | 1836 (from 2113–2526) | **3868** (from 2680) | 2.11× | **prediction inverted** |
| `d1_bluestein` 10007 B=1 | 97.0 | 184.3 | 1.90× | direction right, magnitude not |
| `d1_pow2` 1024 B=1 m=1 | 1.014 | 1.366 | 1.35× | carried |
| `d1_pow2` 16384 B=1 m=1 | 27.0 | 45.97 | 1.70× | carried |
| `d1_twiddle` 16384 B=1 m=1 | 31.2 | 51.10 | 1.64× | carried |
| `d1_planner` 65537 B=1 | 813 | 1417 | 1.74× | carried |
| `d1_rader` 1021 B=1 m=1 | **7.85–7.99** | **7.788** | **1.00×** | exact |
| `d1_rader` 13 B=1 / B=512 | 0.056 / 0.051 | 0.0562 / 0.0511 | 1.00× | exact |
| `d1_rader` 65537 B=16 m=1 | 1489 quiet, 1653 noisy | 1661.7 | — | the *noisy* number was the true one |

**`d1_rader` is the control.** They are the only entry whose claims land on the measurement
(1.00× span), and they are the only entry that measured on the scoring node. Everyone
else's absolute numbers are off by 1.3–2.1×. That is not a criticism of the others — it is
the strongest possible argument that **wallaby numbers cannot be scored, only ranked**, and
even the ranking fails in two cases below.

### The two inversions, and why they are not "just the machine"

**`d1_composite` at 60 B=512 m=1 (claimed win, measured 0.63x loss).** MKL spans 1.27× over
this gap; `d1_composite` spans 2.14×. So this is *not* the machine moving both equally —
their kernel degrades disproportionately on Ice Lake. Their own record predicts the
mechanism and names the experiment: they ship three kernel variants (`ymm2` default,
`zmm2x2`, `zmm4`) and measured on SPR that `zmm2x2` **loses despite 26 % fewer instructions
per transform** (415 vs 565) because at 512-bit every shuffle/blend/FMA uop shares p0+p5,
while a 256-bit mix spreads across p0/p1/p5. That balance differs on ICL, and the variant
choice was never A/B'd there. This is a plausible-cause attribution, not a proven one; the
A/B is one afternoon on the node.

**`d1_prime` at 13 (claimed clear win, measured 0.68x and a 29 % regression).** Their round
was built on one change: an `__asm__("" : "+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi))` barrier
forcing the folded u/v rows through memory, so gcc emits `{1to8}` broadcast-memory FMA
operands instead of port-5 `vpermpd`. On SPR that was worth 21.5 → 27.6 GF/s. On the node
it went backwards. I do not think the machine alone explains this, because MKL moved 0 %
in that exact cell between rounds. The likelier cause is documented *by another
implementer in this same round*: `d1_batchlane` found that kernel vector loads spanning
freshly-written mixed vector+scalar stores cause store-forward stalls, and that this cost
them 0.023 → 0.036 µs at exactly L=13 until they went AoS-direct. `d1_prime`'s barrier
creates precisely that store-then-vector-load pattern. **Two entries, same size, opposite
conclusions about routing broadcasts through memory — and only one of them measured on the
scoring node.** This is the most valuable disagreement in the round and it is settled by
one A/B on the node.

**`d1_bluestein` at 65537 (claimed 1836, measured 3868, a 31 % regression from r1).** They
switched 65537 to an Agarwal–Cooley 8192×25 decomposition, tuned against wallaby's **2 MB**
L2. Their own record flags the risk twice — "wallaby's 2 MB L2 flatters M=20480 in
particular" and "~1 MB is at L2 capacity". The scoring node has **1.25 MB**. The
decomposition that fits on the dev machine does not fit on the scoring machine, and the
cell fell off a cliff. This one is cleanly attributable to the machine, and it is the
round's best instructive failure: **an L2-capacity-tuned structural choice made on a 2 MB
machine and scored on a 1.25 MB machine.** `d1_pow2` avoided the identical trap by gating
their SoA chain path at L ≤ 2048 for exactly this reason ("at 4096 it fit wallaby's 2 MB L2
(4 % win) but not the scoring node's 1.25 MB; gated out") — and `d1_pow2`'s 4096 numbers
carried while `d1_bluestein`'s 65537 numbers did not.

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

§4 is written for the 3D cube campaign, so §4.1 (L=6/8 register liveness), §4.3 (axis
fusion), §4.5 (L=8 padding) and §4.7 (vector-radix) have no 1D analogue and this round
cannot touch them. Three items did move, one of them substantially.

### §4.4 — split vs. interleaved complex: **advanced, and the granule is now measured**

§4.4 was "strongly motivated but unproven" with no published head-to-head; §08 §5.4 closed
it in the corpus's favour and left "only the *granule*" open, answering 8 volumes per
granule from the cache line. **This round supplies the CPU head-to-head and a measured
failure mode for the wrong granule.**

`d1_pow2` replaced its interleaved-AoS pipeline above the codelets with a blocked
split-complex radix-8 engine and A/B'd three separate design points:

- **Split beats interleaved, but only in the chained regime.** Non-chained cells came out
  at parity to +3 %; the chained cells moved hard: 128 B=1 chain 0.231 → 0.173,
  1024 B=1 chain 2.169 → 1.716, 16384 B=1 chain 49.7 → 41.7 µs. The scoring node
  confirms the direction — `d1_pow2` now takes 1024 B=1 chain at 1.53x over MKL and both
  16384 chained cells. The stated mechanism is §04's: the middle stages became pure
  vertical FMA with **zero shuffles** and broadcast twiddles.
- **The granule is decisive, and "two big planes" is the wrong one.** Split-as-two-L-sized-
  planes **lost 42 %** to blocked `[8 re | 8 im]` at 4096 B=1 (9.6 vs 6.7 µs). The
  diagnosis is new and is not in the corpus: each radix-8 stage ran 16 read + 16 write
  streams at 4K-multiple strides, exhausting L1 fill buffers — and **a 64 B im-plane pad
  did not fix it (9.56 µs), so it is stream count, not set conflicts.** §08 §1.10's
  8-per-granule answer is confirmed, with the reason sharpened.
- **The AoS→split conversion must be fused into an existing transpose, or it eats the
  win.** A dedicated conversion stage cost 1.24 vs 1.01 µs at 1024 B=1; fusing it into the
  stride-1 first stage's store transpose costs literally zero, because that transpose is
  8 permutes per 16 complexes in *either* output format.
- Bonus datum for the same section: buffer **count** is first-order at L1-boundary sizes —
  a second scratch buffer cost +20 % at 1024 B=1 (L1 working set 48 → 64 KB).

Corroborating from other entries: `d1_batchlane`'s across-batch split-complex SoA is what
won every chained cell at 32/60/64/128, and `d1_pow2` adopted it wholesale. Counter-datum,
worth recording: `d1_rader` measured an 8-wide split map **losing** to scalar (0.110 vs
0.099 µs/step) because Ice Lake zmm `vsqrtpd`/`vdivpd` throughput is so poor — split does
not help when the kernel is divider-bound rather than shuffle-bound.

### §4.8 item 6 — AVX-512 on a modern server part: **the corpus's conclusion does not transfer to Ice Lake-SP**

§4.8 item 6 says "there is no primary measurement in the corpus for Ice Lake-SP or later
server parts, which is the hardware most likely to be in a current LQCD cluster. **Measure
it on the node.**" It then concludes, from the Gold 5218's spec, that 512-bit is "strictly
preferable on this part (half the instructions …) at **zero** frequency cost."

This round measured it on a Gold 6326 and **that conclusion inverts**:

- `d1_composite`: `zmm2x2` at **415 instructions/transform loses to `ymm2` at 565** —
  0.037 vs 0.032 µs — and still loses after the load/store side was de-port-5'd
  (256-bit halves + `vinsertf64x4` loads, `vextractf64x4` stores). Their conclusion, which
  I endorse as the round's most transferable microarchitecture lesson: *at 512-bit all
  shuffle/blend/FMA uops share p0+p5, so instruction count is not the bound; the 256-bit
  mix spreads across p0/p1/p5 and wins despite ~35 % more instructions.* `zmm4` likewise
  went from an r1 tie to a clear loss (0.038 vs 0.032) once both were fully unrolled.
- `d1_rader`: `vgatherdpd` entry is **worse than hand-assembled `vinsertf64x2` + permutes**
  on L1/L2-resident data (2.8 vs 2.0 µs inside a 9.56 µs transform) — Ice Lake gathers are
  microcoded. `vscatterdpd` fused exit cost 2.4 µs vs ~0.5 for a staged scalar store.
- `d1_rader`: an 8-wide zmm map loses to a scalar map (0.110 vs 0.099 µs/step), Ice Lake
  zmm sqrt/div throughput.
- **And the corpus's own tool is wrong here:** `d1_pow2` records that
  `llvm-mca -mcpu=icelake-server` models **one** 512-bit FMA unit and therefore piles all
  512-bit FP on port 0, rating their AoS and split radix-8 stages identical. The Gold 6326
  has **two**. Use mca for uop counts and dependency chains, not for p0/p5 balance on this
  machine.

So: §4.8 item 6's frequency claim is fine (this part does not licence-downclock), but its
*inference* — "512-bit is strictly preferable, half the instructions" — is contradicted by
three independent measurements on the actual node. **Port distribution, not instruction
count, is the binding constraint.** That is a real closure of an item the corpus explicitly
asked to be measured.

### §4.6 — model vs. search for the instruction schedule: **advanced, with a new caveat**

§4.6 says the schedule is "the primary thing to search" at non-power-of-4 sizes and that
the search is cheap. `d1_race` is a literal implementation of that: a plan-time fork-gated
race across sibling implementations with per-host wisdom. The measured verdict is
two-sided and is new information:

- Search **wins**: `d1_race` beat every hand-picked sibling in **15 of 52 cells**, by up to
  15 % (60 B=512 m=1: 0.0516 vs `d1_composite` 0.0684; 100003 B=1 m=1: 3089 vs
  `d1_bluestein` 3613; 65537 B=16 m=1: 1502 vs `d1_rader` 1662).
- Search **loses when under-sampled**: it picked a *worse* arm than one available to it in
  **10 of 52 cells**, by up to 32 % (10007 B=64 m=1). All ten are at L ≥ 1021, where a race
  sample is expensive and the race evidently buys too few.

§4.6 does not have this caveat. The corpus assumes search is free at small sizes and says
nothing about what happens when the search budget is small relative to the measurement
noise — which, per §3.1, is exactly this machine's problem. Add to §4.6: **a search whose
per-arm sample is shorter than the machine's clock-state settling time is worse than a
fixed good choice.**

Two compiler-search data points also belong under §4.6, both of which cost real time to
find and are cheap to reuse: `_Pragma("GCC unroll N")` is required to make gcc 11.4
straight-line a fixed-trip loop indexed by a constant table even at `-O3 -funroll-loops`
(`d1_composite`, and it was most of their m=1 gap); and `optimize("no-tree-vectorize,
no-tree-slp-vectorize")` is required on hand-written register codelets, where gcc's SLP
pass turned a 12-point codelet into 47 `vpermt2pd` + 8 `vgatherdpd` and made it 2.9×
slower (`d1_rader`, 0.163 → 0.056 µs).

### §4.2, by analogy — dense-symmetric vs. Rader at small primes

Not a 1D question as written (it is about L=17), but the round answers it decisively at
the nearest 1D sizes, in the direction §02 §7 predicted ("Rader is not the lever at L=17"):
at **L=13** `d1_prime`'s dense conjugate-symmetric fold is **2.5× faster** than
`d1_rader`'s Rader-with-CRT-codelet (0.0206 vs 0.0511, batched single call), and at
**L=31** it is **3.3× faster** (0.0643 vs 0.2103). `d1_rader` concedes it in their own
record. Worth carrying back to §4.2 as supporting evidence.

---

## 6. Highest-value thing for the next round

**Cross-cutting, and it outranks every per-geometry item: fix the measurement.** Record the
achieved clock (`aperf`/`mperf`, or `perf stat -e cycles,ref-cycles`) per timed region into
`environment.txt` and into every `t_*.json`, and stop scoring m=1 cells with min-over-three
processes. Eighteen of 52 cells are currently undecided for a reason the harness cannot
even see, and four implementers spent this round optimising against those cells. Pinning to
a fixed P-state, or reporting median-with-clock rather than min, would convert 18 undecided
cells into real results at zero implementer cost. Nothing else on this list is worth as
much.

Then, per geometry:

| L | attack | why |
|---|---|---|
| **13, 31** | **A/B `d1_prime`'s `"+m"` broadcast barrier on the node**, then take the batched-single cell with across-batch SoA lanes rather than more AoS port-5 surgery. | 13 B=512 m=1 is the panel's only robust loss at a small prime, it got *worse* after being targeted, and two entries hold documented opposite conclusions about routing broadcasts through memory (§4). Settling that disagreement is one afternoon and unblocks both entries. |
| **32, 64, 128** | **The batched-single cell** (0.99x / 1.03x / 0.92x — all undecided). `d1_batchlane` names the structural cause: ~160 KB per group through a 48 KB L1 in the multi-pass SoA path. Try `d1_pow2`'s AoS 4-complex-per-zmm per transform, with PMU evidence that port 5 is actually the wall. | Three cells, all within a hair, all in the regime the brief says is our natural advantage. The chained cells here are already won 1.7–3.0x; this is the remaining half. |
| **60** | **A/B `ymm2` vs `zmm2x2` vs `zmm4` on Ice Lake.** All three variants already ship in-file behind `-D` switches; the choice was made on SPR and never re-tested. | The sharpest dev-vs-score divergence in the round: a claimed 1.06x win measured as a 0.63x loss, with `d1_composite` spanning 2.14× where MKL spans 1.27×. One build sweep, high expected payoff. |
| **1024, 4096, 16384** | **The non-batched single cell** (0.79x / 0.66x / 0.72x — the only robust losses at pow2). Four-step 128×128 at 16384 is `d1_pow2`'s own next step; at 4096 the gap is 1.52x to MKL. | These are the four robust losses that are not at 100003. Also the sizes where the corpus is most confident we *should* lose, so a win here is the round's most citable result. |
| **1021** | Consolidate, then software-pipeline two batch elements at B=256 and replace zmm sqrt/div in `st17_chain` with rsqrt14+Newton. | All four cells won this round, from two losses — the largest gain of r2. Protect it: re-verify it survives once the clock recording lands, since three of the four margins are 1.21–1.45x. |
| **10007** | **The untried nested Rader for the inner 5003** (5003−1 = 2·41·61, both smooth). All four margins are 1.07–1.16x, i.e. inside the clock band. | This is the one size where nobody has tried the structural move the survey recommends, and it is currently the panel's least-secure "win". |
| **65537** | **Recover the r1 chain numbers first** (910/949 → 1052/1085 is a real 13 % regression), then take B=16 m=1 (1502 vs 1544, undecided). Vectorise the last scalar stage, `st8` at s=4. | The only geometry that went backwards this round. Fix the regression before adding anything. |
| **100003** | The panel's weakest size — 3 of 4 cells not won, one robustly lost. Nested Rader for 2381, or an AC decomposition sized for **1.25 MB** L2 rather than 2 MB. | `d1_bluestein`'s 65537 collapse (§4) is the direct warning: size the decomposition for the scoring machine's L2, not the dev machine's. |

---

## 7. Curation decision

Applying `docs/CURATION.md`'s four grounds, in order. Rule 1 is *per L*, not per cell, so a
cell won by 1–4 % inside the noise band does not by itself earn a promotion.

**Promoted (7):**

| entry | ground | evidence |
|---|---|---|
| `d1_prime` | 1 — fastest at L=13 and L=31 | 31: 4.35x / 5.49x / 6.53x over MKL; 13 batched chain 3.33x. Dense symmetric-pair real-coefficient fold. |
| `d1_pow2` | 1 — fastest at L=1024, 4096, 16384 | 1024 B=1 chain 1.53x, 4096 B=256 chain 1.76x, 16384 B=64 chain 1.46x. Also ground 3: carries the *negative* result that split-as-two-planes loses 42 % to blocked split. |
| `d1_batchlane` | 1 at 32/60/64 B=1 chained + **2 — structurally different runner-up** | Across-batch SoA split-complex, within 1–2 % of `d1_pow2`'s in-register AoS at 64/128 chained while being a completely different construction. The panel needs both written down. Also carries the store-forward-stall record (§4). |
| `d1_composite` | 1 — fastest at L=60 | Only PFA/Good-Thomas entry; 0.0534 vs MKL 0.0622 at B=1 m=1, the one m=1 cell recovered this round. Ground 3 as well: the `zmm2x2` and `zmm4` losses are documented with the numbers that killed them. |
| `d1_rader` | 1 — fastest at L=1021 (all four cells) and 65537 batched chain; **4** | 1021 B=1 0.60x → 1.21x, the round's largest gain. Also the only entry whose claims reconcile with the measurement, which makes its record the reference for how to measure here. |
| `d1_bluestein` | 1 — fastest at L=10007 and 100003; **3 — instructive failure** | Owns the two awkward primes. Promoted *equally* for the 65537 AC-8192×25 collapse (2680 → 3868 µs), which is the clearest documented case of tuning to the dev machine's L2 — the next panel should read it before choosing any decomposition. |
| `d1_race` | 2 and **3** | Beat every sibling in 15 of 52 cells (up to 15 %) — the fork-gated race plus per-host wisdom finds configurations no sibling ships by default. Promoted *with* the counter-record: it mis-picks in 10 cells by up to 32 %, all at L ≥ 1021, and pays 11.4 s of plan time for a 0.023 µs transform. Both halves are the lesson. |

**Not promoted (2), with reasons:**

- **`d1_twiddle`** — wins exactly one cell (4096 B=1 chain, 14.27 vs `d1_pow2` 14.87, a 4 %
  margin on a 14.2 % spread) and is not the fastest entry at any L. Its real contribution,
  the quadrant-exact long-double twiddle tables, was adopted by `d1_batchlane` (chain gates
  improved up to 10× at zero runtime cost) and `d1_pow2` — both promoted — so the technique
  is preserved in code. Its strategy record stays tracked regardless.
- **`d1_planner`** — wins exactly one cell (10007 B=64 chain, 205.5 vs `d1_bluestein` 205.7,
  a **0.1 %** margin) and is mid-pack in the other 51. Improved 2–6× over r1 and is a
  genuinely useful generic fallback across all 13 sizes, but it is not the fastest entry at
  any L, is not a close structural runner-up anywhere, and its record documents no
  instructive failure. Reconsider next round if the generic path closes on a specialist.

Neither exclusion loses anything: both entries' strategy records remain tracked under
`strategies/`, which is where the reusable content lives.

---

PROMOTE: d1_prime d1_pow2 d1_batchlane d1_composite d1_rader d1_bluestein d1_race
