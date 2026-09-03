# VERDICT — panel round d1_r4

Monitor pass over `results/d1_r4/`. Judgement only: no timing was re-run. Sources graded are
`impl_4/` (= `impl` symlink), 9 panel entries plus `baseline_dft`, against 7 library
configurations, over 52 graded cells.

---

## 0. Four corrections to the brief, before any numbers

(a)–(c) are the same corrections the r3 monitor recorded in `results/d1_r3/VERDICT.md` §0.
They have not been fixed in the brief, so they are restated here rather than silently worked
around. (d) is new to this round.

**(a) The geometries are not L = 6, 8, 17, 36.** Those are the *3D* campaign's cube edges
(`bench/geom/`). This is the 1D campaign, and its graded set is
**L = 13, 31, 32, 60, 64, 128, 1021, 1024, 4096, 16384, 10007, 65537, 100003** — thirteen
lengths × {B=1, batched} × {single call, chained} = 52 cells. `leaderboard.txt` contains no
L=6/8/17/36 row and never has. §1 below reports all thirteen.

**(b) The scoring node is not a Cascade Lake Xeon Gold 5218.** `environment.txt` reads
**Intel Xeon Gold 6326 @ 2.90 GHz, 64 cores** with `avx512_vnni avx512_bitalg avx512_vbmi
avx512_ifma avx512_vpopcntdq` in the ISA list — that combination is **Ice Lake-SP**, not
Cascade Lake (which has none of `vbmi`, `bitalg`, `vpopcntdq` or `ifma`).
1.25 MB L2 per core, 24 MB shared L3, and Ice Lake-SP does *not* carry Skylake/Cascade Lake's
severe AVX-512 licence downclocking. The Gold 5218 in the brief is the part
`docs/LITERATURE.md` §4.8 item 6 discusses; it is not this machine. This matters for §4 and §5:
the panel develops on **wallaby, Xeon Gold 6448Y (Sapphire Rapids), 2 MB L2, ~4.1 GHz** and is
scored on **a80n0, Gold 6326 (Ice Lake-SP), 1.25 MB L2, 2.9 GHz**, and the two behave
differently in a way this round measured directly.

**(c) `docs/LITERATURE.md` §4 is the 3D corpus's open-questions section.** Three of its eight
items (§4.3 axis fusion, §4.5 L=8 padding, §4.7 vector-radix) are 3D-geometry questions a 1D
round cannot address. §5 answers against the four items that are hardware- and
kernel-structure questions, which this round does bear on.

**(d) Two statistics are in play and the brief does not say which.** Per commit `5fc787a5`,
r3 ranked on **min-over-3-runs**; r4 ranks on **median-over-3-runs** and prints the min
alongside. Every r3→r4 comparison in §2 is therefore made **min-against-min**, which is the
only apples-to-apples pairing. `RESCORE_PLAN.md` records that r1–r4 are all still to be topped
up to 9 runs on single-call cells; until that happens the single-call verdicts here carry the
measured ±9% median / ±22% p90 inter-process band.

---

## 1. Headline per geometry — fastest correct panel entry vs best library

Median µs/transform, the round's ranking statistic. Spread is the in-cell run spread over 3
runs. "*(inside noise)*" means `leaderboard.py` flagged the panel-vs-library gap as smaller
than the measurement spread — those cells are ties, not results, in either direction.
`baseline_dft` (the harness's O(L²) floor) is excluded from "library".

| L | cell | best panel (median) | spread | best library (median) | spread | verdict |
|---|---|---|---|---|---|---|
| **13**   | B=1 m=1 | d1_race 0.0148 | ±2% | mkl1d_dfti 0.0260 | ±3% | **WIN 1.76x** |
|          | B=1 chain m=200000 | d1_planner 0.0394 | ±14% | fftw1d_measure 0.0674 | ±0% | **WIN 1.71x** |
|          | B=512 m=1 | d1_prime 0.0106 | ±12% | fftw1d_patient 0.0140 | ±2% | **WIN 1.32x** |
|          | B=512 chain m=2000 | d1_race 0.0146 | ±0% | fftw1d_custom_soa 0.0299 | ±1% | **WIN 2.05x** |
| **31**   | B=1 m=1 | d1_prime 0.0471 | ±22% | fftw1d_custom 0.2158 | ±20% | **WIN 4.58x** |
|          | B=1 chain m=100000 | d1_prime 0.0534 | ±14% | fftw1d_custom 0.2095 | ±1% | **WIN 3.92x** |
|          | B=512 m=1 | d1_race 0.0425 | ±2% | fftw1d_custom_soa 0.1054 | ±21% | **WIN 2.48x** |
|          | B=512 chain m=1200 | d1_race 0.0465 | ±5% | fftw1d_custom_soa 0.0780 | ±1% | **WIN 1.68x** |
| **32**   | B=1 m=1 | d1_twiddle 0.0192 | ±13% | mkl1d_dfti 0.0257 | ±8% | **WIN 1.34x** |
|          | B=1 chain m=100000 | d1_race 0.0600 | ±17% | fftw1d_custom 0.1217 | ±1% | **WIN 2.03x** |
|          | B=512 m=1 | d1_twiddle 0.0174 | ±0% | **mkl1d_dfti 0.0153** | ±14% | **loss 0.88x** |
|          | B=512 chain m=1000 | d1_batchlane 0.0340 | ±14% | fftw1d_custom_soa 0.0667 | ±0% | **WIN 1.96x** |
| **60**   | B=1 m=1 | d1_composite 0.0451 | ±15% | mkl1d_dfti 0.0616 | ±4% | **WIN 1.37x** |
|          | B=1 chain m=60000 | d1_composite 0.1104 | ±0% | mkl1d_dfti 0.2371 | ±0% | **WIN 2.15x** |
|          | B=512 m=1 | d1_race 0.0565 | ±36% | mkl1d_dfti 0.0514 | ±4% | loss 0.91x *(inside noise)* |
|          | B=512 chain m=600 | d1_race 0.0706 | ±10% | fftw1d_custom_soa 0.1367 | ±29% | **WIN 1.94x** |
| **64**   | B=1 m=1 | d1_race 0.0427 | ±16% | mkl1d_dfti 0.0503 | ±5% | **WIN 1.18x** |
|          | B=1 chain m=60000 | d1_race 0.0848 | ±14% | mkl1d_dfti 0.2696 | ±0% | **WIN 3.18x** |
|          | B=512 m=1 | d1_planner 0.0421 | ±19% | mkl1d_dfti 0.0483 | ±29% | **WIN 1.15x** |
|          | B=512 chain m=500 | d1_batchlane 0.0677 | ±1% | fftw1d_custom_soa 0.1501 | ±4% | **WIN 2.22x** |
| **128**  | B=1 m=1 | d1_race 0.1126 | ±14% | **mkl1d_dfti 0.1041** | ±4% | **loss 0.92x** |
|          | B=1 chain m=30000 | d1_batchlane 0.1850 | ±14% | fftw1d_measure 0.4965 | ±14% | **WIN 2.68x** |
|          | B=512 m=1 | d1_pow2 0.1789 | ±1% | **mkl1d_dfti 0.1584** | ±3% | **loss 0.89x** |
|          | B=512 chain m=250 | d1_race 0.1873 | ±15% | fftw1d_custom_soa 0.4476 | ±3% | **WIN 2.39x** |
| **1021** | B=1 m=1 | d1_race 7.3968 | ±16% | mkl1d_dfti 8.2955 | ±14% | **WIN 1.12x** |
|          | B=1 chain m=2000 | d1_rader 7.1277 | ±15% | mkl1d_dfti 12.9606 | ±14% | **WIN 1.82x** |
|          | B=256 m=1 | d1_race 7.3156 | ±31% | mkl1d_dfti 8.8778 | ±15% | **WIN 1.21x** |
|          | B=256 chain m=400 | d1_race 7.1998 | ±2% | mkl1d_dfti 12.3972 | ±0% | **WIN 1.72x** |
| **1024** | B=1 m=1 | d1_race 1.2598 | ±16% | **fftw1d_measure 1.1596** | ±14% | **loss 0.92x** |
|          | B=1 chain m=4000 | d1_pow2 2.4313 | ±14% | mkl1d_dfti 4.1837 | ±0% | **WIN 1.72x** |
|          | B=512 m=1 | d1_race 1.9513 | ±19% | mkl1d_dfti 1.8566 | ±5% | loss 0.95x *(inside noise)* |
|          | B=512 chain m=2000 | d1_race 2.5856 | ±0% | mkl1d_dfti 4.9167 | ±1% | **WIN 1.90x** |
| **4096** | B=1 m=1 | d1_pow2 8.8882 | ±3% | **mkl1d_dfti 6.8279** | ±0% | **loss 0.77x** |
|          | B=1 chain m=1000 | d1_race 11.1503 | ±16% | fftw1d_patient 19.2318 | ±7% | **WIN 1.72x** |
|          | B=256 m=1 | d1_pow2 10.3932 | ±1% | fftw1d_patient 11.0852 | ±3% | **WIN 1.07x** |
|          | B=256 chain m=400 | d1_race 10.1376 | ±4% | mkl1d_dfti 22.8917 | ±1% | **WIN 2.26x** |
| **10007** | B=1 m=1 | d1_race 124.69 | ±20% | fftw1d_patient 204.05 | ±5% | **WIN 1.64x** |
|          | B=1 chain m=400 | d1_bluestein 168.36 | ±51% | fftw1d_patient 227.07 | ±0% | **WIN 1.35x** |
|          | B=64 m=1 | d1_race 183.43 | ±39% | fftw1d_patient 204.78 | ±4% | win 1.12x *(inside noise)* |
|          | B=64 chain m=80 | d1_bluestein 145.94 | ±0% | fftw1d_patient 235.55 | ±2% | **WIN 1.61x** |
| **16384** | B=1 m=1 | d1_pow2 35.29 | ±8% | **fftw1d_patient 32.85** | ±3% | **loss 0.93x** |
|          | B=1 chain m=250 | d1_race 49.53 | ±20% | fftw1d_patient 82.84 | ±0% | **WIN 1.67x** |
|          | B=64 m=1 | d1_twiddle 57.63 | ±13% | **fftw1d_patient 48.84** | ±1% | **loss 0.85x** |
|          | B=64 chain m=150 | d1_race 50.12 | ±3% | fftw1d_patient 97.60 | ±3% | **WIN 1.95x** |
| **65537** | B=1 m=1 | d1_race 852.6 | ±1% | fftw1d_patient 1466.7 | ±0% | **WIN 1.72x** |
|          | B=1 chain m=60 | d1_race 724.9 | ±5% | fftw1d_patient 1636.5 | ±0% | **WIN 2.26x** |
|          | B=16 m=1 | d1_race 1328.4 | ±41% | fftw1d_patient 1547.7 | ±1% | win 1.17x *(inside noise)* |
|          | B=16 chain m=20 | d1_race 774.4 | ±2% | fftw1d_patient 1741.7 | ±1% | **WIN 2.25x** |
| **100003** | B=1 m=1 | d1_race 2323.7 | ±32% | fftw1d_measure 2851.5 | ±8% | **WIN 1.23x** |
|          | B=1 chain m=40 | d1_race 2422.5 | ±1% | fftw1d_patient 3111.4 | ±1% | **WIN 1.28x** |
|          | B=8 m=1 | d1_bluestein 2472.5 | ±39% | fftw1d_patient 2845.7 | ±3% | win 1.15x *(inside noise)* |
|          | B=8 chain m=15 | d1_bluestein 2828.8 | ±0% | fftw1d_patient 3229.4 | ±3% | **WIN 1.14x** |

### Score: 43 of 52 cells to the panel, and the split is entirely along one axis

* **Chained cells: 26 of 26 to the panel.** Every single one, by 1.14× to 3.92×, geometric
  mean **1.95×**. The libraries do not have a competitive chained path anywhere on the board.
* **Single-call cells: 17 of 26.** Geometric mean **1.22×**. All nine library wins are m=1
  cells, and four of the nine are marked inside the noise.
* Aggregate over all 52 cells, geometric mean of best-library ÷ best-panel: **1.543×**
  (r3 under the same median statistic: 1.436×). `RESCORE_PLAN.md`'s "r4 1.617× so far" was a
  mid-round figure; 1.543× is the final board.

The nine library-won cells, and whether the loss is real:

| L | cell | library | best panel | gap | resolved? |
|---|---|---|---|---|---|
| 4096 | B=1 m=1 | mkl 6.828 | d1_pow2 8.888 | 0.77× | **real, and the largest hole on the board** |
| 16384 | B=64 m=1 | fftw_patient 48.84 | d1_twiddle 57.63 | 0.85× | **real** |
| 32 | B=512 m=1 | mkl 0.0153 | d1_twiddle 0.0174 | 0.88× | **real** |
| 128 | B=512 m=1 | mkl 0.1584 | d1_pow2 0.1789 | 0.89× | **real** |
| 60 | B=512 m=1 | mkl 0.0514 | d1_race 0.0565 | 0.91× | inside noise (±36%) |
| 128 | B=1 m=1 | mkl 0.1041 | d1_race 0.1126 | 0.92× | real but marginal |
| 1024 | B=1 m=1 | fftw_measure 1.1596 | d1_race 1.2598 | 0.92× | real but marginal |
| 16384 | B=1 m=1 | fftw_patient 32.85 | d1_pow2 35.29 | 0.93× | real but marginal |
| 1024 | B=512 m=1 | mkl 1.8566 | d1_race 1.9513 | 0.95× | inside noise |

---

## 2. What changed since d1_r3, per geometry

All comparisons min-of-3 against min-of-3 (§0(d)). **Calibration first, because it governs
every claim in this section:** across the 270 library and `baseline_dft` rows — *unchanged
code, rebuilt from an identical source* (`diff impl_3/baseline_dft.c impl_4/baseline_dft.c`
is empty) — the r3→r4 shift in min-of-3 has p50 **0.7%**, p90 **17.2%**, max **62.6%**, and
28% of rows move by more than 10%. **Nothing under ~17% on a single-call cell is a result.**
Chained cells are the trustworthy regime: p50 1.1%, p90 16.2% per `RESCORE_PLAN.md`.

### Improvements, by geometry

| L | what improved | mechanism, from the entries' own records |
|---|---|---|
| **13, 31** | `d1_batchlane` −24% to −44% at all four batched/single cells; `d1_planner` −54% to −84% at all eight | Both **adopted `d1_prime`'s r3 interleaved-pair kernels wholesale**. batchlane's record: "the ENTIRE interleaved-pair kernel design for 13/31 (exec13p/exec13p_b2/exec31p, ported with renamed macros)". This is the round's largest single mass of improvement and it is *diffusion*, not invention. |
| **13, 31** | `d1_bluestein` −17% to −41% everywhere | Zero-pruned + chirp-fused entry and pruned chirp-fused exit for small M. Still 6–9× off the pace at these sizes; the improvement is real and irrelevant to the ranking. |
| **32, 64** | `d1_twiddle` −30% to −45% at six of eight cells; took 32 B=1 m=1 outright | Ported `d1_pow2`'s in-register L=32/64 codelets, execute **and** register-resident chains. Again adoption. |
| **60** | `d1_composite` −12% at B=1 m=1 (0.0489→0.0430), MKL gap 1.20×→1.12× at B=512 m=1 | The **port-5 diet**: 415→287 instructions/transform, ~60 fewer p5 µops, broadcast-fed signed-FMA stage A, memory-destination `vextractf64x2` pure-store exits. See §5 — this is the round's most transferable result. |
| **1024, 4096, 16384** | `d1_pow2` −22% at 1024 B=1 m=1, **−38%** at 16384 B=1 m=1, −33% at 16384 B=1 chain, −19% at 16384 B=64 chain | **Compact (c,s)-pair twiddle tables at L≥1024, halving the L2 footprint.** Their diagnosis, from on-node PMU: "the ICX large-L bottleneck was table-bloated L1/L2 fill traffic, not ports." Plus first-stage-pair fusion through an L1 tile at L≥4096. |
| **16384** | `d1_race` −22% to −32% at three of four cells; `d1_twiddle` −37% at B=64 m=1 | race's ship-time placement probe (§3.4) plus pow2's table compaction reaching twiddle by port. |
| **10007** | `d1_race` −22%, `d1_bluestein` −20%, `d1_planner` −28% at B=1 m=1 | bluestein's one-pass fused middle (fwd-last × kernel-mul × inv-entry in a single pass) and paired-p zmm radix-8 at the narrow s=4 stage. race and planner inherit it. |
| **65537** | `d1_rader` −14% at B=1 m=1 (1014→868) | The **fused mid pass**: fwd-last butterfly + kernel multiply + inv-first butterfly in one pass, 9 passes not 10 — the spectrum is never materialised. |
| **100003** | `d1_bluestein` −22% at B=1 m=1 | Agarwal–Cooley coprime 2D convolution with chirp/CRT-fused entry and exit, minimal M split over odd 3^a5^b. |

**Two cells flipped from library to panel and none flipped the other way**: L=64 B=1 m=1
(r3 mkl 0.0465 → r4 d1_race 0.0427 median / 0.0385 min) and L=4096 B=256 m=1
(r3 mkl 10.147 → r4 d1_pow2 10.393 median). Library wins went 11 → 9.

### Regressions

Presented against the ±17% single-call noise band. The **chained** regressions are the only
ones I treat as established, since chained noise is p90 16.2% and these cells' in-cell
spreads are ≤1%.

**Established (chained regime):**

| entry | cell | r3 min | r4 min | change | assessment |
|---|---|---|---|---|---|
| `d1_batchlane` | 32 B=1 chain | 0.0591 | 0.0698 | **+18%** | Lost the cell it held in r3. Its record claims 0.070 for this cell and does not mention a regression — the r3 number is simply gone. Its "natural-row four-step" B=1 chain rewrite at 32/64/128 is the suspect. |
| `d1_batchlane` | 64 B=1 chain | 0.0848 | 0.0965 | **+14%** | Same rewrite. `d1_race`, which ships batchlane's arm here, measured 0.0848 in the same sweep — so batchlane's *own* build lost 12% that race's shipped copy did not (§3.4). |
| `d1_batchlane` | 13 B=1 chain | 0.0392 | 0.0451 | **+15%** | Lost the cell to `d1_planner`. Record claims 0.045 — the regression is disclosed as a number but not as a regression. |
| `d1_pow2` | 128 B=512 chain | 0.1870 | 0.2122 | **+13%** | Not mentioned in its record, which claims 0.187 for this cell. Its own claim is the r3 number. |
| `d1_bluestein` | 16384 B=1 chain | 121.7 | 140.5 | **+15%** | Off-target size for this entry (3.5× off the pace); cosmetic. |
| `d1_prime` | 13 B=512 chain | 0.0154 | 0.0171 | +11% | Just inside the band; its record claims 0.015 flat. Probably noise. |

**And one accountability finding on top of the numbers.** `d1_batchlane`'s three regressions
above are all *present as numbers* in its own r4 table — it reports 0.045 / 0.070 / 0.098 at
13 / 32 / 64 B=1 chained, which are exactly the regressed values — while its
"What did NOT work" section asserts **"Nothing else regressed this round."** The r3 values for
those same cells (0.0392 / 0.0591 / 0.0848) are in the r3 leaderboard the record cites
elsewhere. Two of them were cells `d1_batchlane` itself held in r3. Reporting the new number
without comparing it to the one you previously held is how a regression survives a round; this
is the round's clearest instance and it is the reason `d1_batchlane`'s promotion in §7 is
conditioned on the regression being on the record.

**Not established (single-call, inside the noise band):** `d1_rader` 31 B=512 +14%,
`d1_pow2` 32 B=512 +14% and 1024 B=512 +13%, `d1_batchlane` 64 B=512 +19% and 128 B=512 +15%,
`d1_twiddle` 128 B=1 +20%, `d1_race` 64 B=512 +16%, `d1_planner` 1021 B=256 +14%,
`d1_bluestein` 16384 B=64 +22%. For four of these the entry's own same-window interleaved A/B
against the r3 binary reports **no** regression — `d1_rader` explicitly: "31 B=512 … 0.063 →
0.063", flat. I attribute these to inter-process placement variance, not to code.

**The one apparent regression that is a scoring artefact, not a regression:** `d1_composite`
at 60 B=512 chain reads median 0.0740 against r3's 0.0592, which looks like +25%. Its
**min** is 0.0589 — identical to r3 — with a 30% in-cell spread. Its record says this path
was **untouched** this round ("B=512 m=600 | 0.044 | 0.044 (untouched)"). The median moved;
the code did not.

---

## 3. Adversarial pass — correctness, builds, crashes, gaps

### 3.1 Nothing failed correctness, and I checked that the check can fail

* `check.log`: **zero** occurrences of `FAIL`, `mismatch` or `error`, case-insensitive,
  across the whole file. Every graded configuration reports `PASS`.
* Two-sided, not one-sided: the log contains **two distinct gate types** that both had to
  fire — the single-call `rel_l2 … (tol 1e-12)` gate and the chained `map-chain m=… (anchor …,
  tol 1.0e-10)` gate, plus a `map-2-step … (tol 3.0e-14)` strict gate. A run that produced no
  output would show as missing rows, not as passes; the leaderboard's `correctness` column is
  populated on all 530 rows.
* The reported errors are physically sane and *differentiated*, which a stub cannot fake:
  1.3e-16 at L=32 rising monotonically to 1.3e-15 at L=65537, and the O(L²)
  `baseline_dft` sitting consistently 2–4× worse than the FFT entries at the same L
  (4.0e-16 vs 1.6e-16 at L=13). `d1_rader` at L=13 reports 3.2e-16 against `d1_prime`'s
  1.6e-16 — the convolution path is measurably less accurate than the dense one, exactly as
  it should be. A harness that was not comparing against a reference would not produce that
  structure.
* Chain gates carry real margin, not threshold-hugging: the worst on the board is
  `d1_pow2` 32 B=512 chain at 1.2e-11 against a 1e-10 tolerance (8× margin), and the median
  chained residual is ~1e-13 against 1e-10.

**No fast wrong answers.** I looked for the specific shape — an entry that is anomalously fast
at one cell *and* anomalously inaccurate there — and there is none. The fastest entries are
also among the most accurate.

### 3.2 `build_errors.txt` contains no errors — but one warning is a latent bug

The file is 84 lines of **warnings only**; all ten sources compiled and all ten linked
(`agents/exits.txt`: nine `exit=0`, and `baseline_dft` is unchanged from r3). Nothing failed
to build. `failures.txt` does not exist, which in this harness means no entry crashed or hung;
consistent with all 530 leaderboard rows being populated.

One warning should not be left in the record as cosmetic:

```
impl/d1_batchlane.c:169:39: warning: iteration 2147483648 invokes undefined behavior
  169 |     for (; j < L; ++j) { re[j] = aos[2*j]; im[j] = aos[2*j + 1]; }
```

`j` is `int` and `2*j` **signed-overflows at j = 2³¹**, which is genuine UB that GCC is
entitled to exploit. It is unreachable at every graded L (max 100003) so it did not affect a
single number this round, and it is not a correctness finding for r4. It should be fixed
anyway — the same pattern appears at three call sites (`deint8`, `chain32_reg`, `chain64_reg`)
and `-Waggressive-loop-optimizations` firing means the optimiser is already reasoning about it.

### 3.3 Missing entries: none unexplained, and coverage did not shrink

No entry present in r3 is absent from r4 at any cell — I checked all 52 cells against both
leaderboards. Per-cell absences are all deliberate specialisation and match each entry's
declared scope in the `backends:` block:

* `d1_composite` serves **only L=60** (Good-Thomas PFA 60=4×3×5). It is the fastest entry
  there in both non-batched cells. Serving one size is its design, not a gap.
* `d1_prime` serves 13 and 31 only; `d1_rader` serves 13, 31, 1021, 65537 — the primes with
  smooth N−1. At 10007 (N−1 = 2·5003, 5003 prime) and 100003 Rader has no cheap convolution
  length and correctly declines; Bluestein covers those.
* `d1_pow2` and `d1_twiddle` decline the non-power-of-two sizes; `d1_batchlane` declines
  L>128.
* `baseline_dft` is absent from L=128 B=512 and from every L≥1021 cell. At L=1021 it runs
  2191 µs/transform (296× the winner); omitting it above that is a harness time decision, not
  a failure. Its presence at 11 of 13 lengths is enough to keep the floor honest.

### 3.4 `d1_race` wins 25 of 52 cells, and what that number does and does not mean

`d1_race` is not an FFT. Its own `backends:` line says it: "plan-time race + per-host wisdom;
demo entry fork-gates and races the sibling class entries per (L,B) and ships the winner by
vtable." It wins **25 cells — nearly half the board — by shipping other entries' kernels.**
Three things follow, and they cut in different directions.

**(a) Most of its wins are tautological and must not be read as an algorithmic result.** At
L=65537 it ships `d1_rader`'s fused-mid-pass conv and lands within 2% of it in all four cells
(852.6 vs 869.4; 724.9 vs 740.7; 774.4 vs 776.8). At 13 B=512 chain it ties `d1_rader` to four
digits (0.0146 vs 0.0146). Reporting "race won 65537" instead of "Rader won 65537" would
misattribute the work. **Every headline in §1 where `d1_race` appears should be read as "the
sibling race selected correctly", and §6/§7 credit the sibling.**

**(b) Its setup cost is excluded from every timing, and at small L it is grotesque — but in
aggregate it is not the worst offender on the board.** Summed over all 52 cells `d1_race`
pays **46.19 s** of plan time. The distribution is what matters: **14.34 s at L=13 B=1 m=1**
— to plan a 13-point transform whose execution is 14.8 *nanoseconds* — and 12.20 s at 13
B=512, i.e. **57% of its entire plan budget goes to the two smallest cells on the board**.
A user paying 14 s to save 0.4 ns per transform needs ~3×10¹⁰ transforms to break even. Every
other cell is ≤3.6 s.

For fairness the same column has to be read for the libraries, and it embarrasses the
complaint: **`fftw1d_patient` pays 845.6 s** across the board — 18× `d1_race`'s total —
including 67.7 s at 16384 B=64 and 45.4 s at 4096 B=256, and it is a graded baseline that
wins two cells. So plan-time racing is *not* a worse citizen than `FFTW_PATIENT`; the
criticism that survives is narrower and still worth making: at L=13 the cost is
disproportionate to any plausible use, and the panel brief does not score setup, so the
caveat has to travel with the number. It does not travel with the number today.

**(c) One part of it *is* a genuine, independent result, and it is the round's sleeper.**
At several cells `d1_race` beats **every** standalone sibling, including the one whose kernel
it ships — and the gap is not noise, because it appears in the *chained* regime with in-cell
spreads under 1%:

| cell | d1_race (min) | best standalone sibling (min) | gap | siblings clustered at |
|---|---|---|---|---|
| 32 B=1 chain | **0.0596** | d1_planner 0.0694 | **+14.1%** | pow2 0.0706, batchlane 0.0698, twiddle 0.0706 — *four* entries at 0.070 |
| 4096 B=1 chain | **9.840** | d1_pow2 11.472 | **+14.2%** | — |
| 64 B=1 chain | **0.0848** | d1_batchlane 0.0965 | **+12.1%** | — |

Its record names the mechanism: `+al64` **variant lanes** — the same source, relinked with
different alignment/placement flags, raced at ship time under the driver's own scoring
conditions. At L=32 B=1 chained, four structurally different implementations all measure
0.070 and the same code under a different placement measures 0.060. **Code and data placement
is worth ~14% on Ice Lake-SP, and it is currently a confound in every other entry's number.**
That is a measurement about the machine, not about `d1_race`, and it is the most reusable
thing in this round.

### 3.5 The round's real process failure: `d1_planner` shipped 992 new lines with no record

`d1_planner.c` grew **2149 → 3141 lines, 1012 changed lines — the largest diff of any entry
this round** — and `strategies/d1_planner.md` has **no `## Round d1_r4` section at all.** Its
record ends at r3. Every other entry has one.

This is not cosmetic. `d1_planner` improved by −54% to −84% at eight cells and **took two
cells outright** (13 B=1 chain at 0.0394; 64 B=512 m=1 at 0.0421), and there is no document
anywhere saying what it did. Reading its r3 record plus the `backends:` line, the improvements
are almost certainly adoption — "adopted codelets: d1_prime pair kernels at 13/31, d1_pow2
register codelets at 32/64" — which would mean its two cell wins are `d1_prime`'s and
`d1_pow2`'s kernels in a planner vehicle. But that is my inference from a stale record, not a
claim anyone made, and `CURATION.md` is explicit that "the record is what makes the code
useful later." See §7.

Separately: `exemplars/d1_r3/NOTES.md` was promoted with its four sections still holding the
`(Fill in: …)` skeleton. r3's promotion happened; its round note did not get written. Worth
fixing while r3's monitor's `VERDICT.md` is still on disk to write it from.

---

## 4. Claimed numbers vs measured — where they diverge, and why

The dominant effect is which machine the implementer used. Five entries measured their r4
numbers **on the scoring node** (`d1_batchlane`, `d1_bluestein`, `d1_pow2`, `d1_prime`,
`d1_rader`) via a leased core; two measured on **wallaby, Sapphire Rapids Gold 6448Y**
(`d1_composite`, `d1_race`) because the Ice Lake reservation was unavailable to them;
`d1_twiddle` reports both. That split predicts the divergences almost perfectly.

**On-node claims hold up, and two entries are exemplary.** `d1_pow2`'s twelve chained claims
match measurement to 1–3% at every cell (32: 0.071/0.039 claimed vs 0.0706/0.0387 measured;
1024: 2.16/2.58 vs 2.157/2.593; 16384: 52.4/51.5 vs 51.25/51.31). `d1_rader` matches at ten of
twelve cells, several exactly (31 B=1 claimed 0.175, measured 0.1750; 13 B=1 claimed 0.047,
measured 0.0471). `d1_prime` matches at six of eight within 3%. Their m=1 claims run ~10–19%
optimistic, which is precisely the min-of-N-on-a-leased-core effect §0(d) describes and not a
credibility problem.

**Off-node claims are optimistic by a consistent factor, and the factor is the machine.**

| entry | cell | claimed (wallaby SPR) | measured (a80n0 ICL) | ratio |
|---|---|---|---|---|
| `d1_composite` | 60 B=1 m=1 | 0.029 | 0.0430 | **1.48×** |
| `d1_composite` | 60 B=512 m=1 | 0.037 | 0.0531 | **1.44×** |
| `d1_composite` | 60 B=1 chain | 0.084 | 0.1101 | 1.31× |
| `d1_composite` | 60 B=512 chain | 0.044 | 0.0589 | 1.34× |
| `d1_race` | 65537 B=16 m=1 | 684 | 1119 | **1.64×** |
| `d1_race` | 100003 B=8 m=1 | 1584 | 2474 | **1.56×** |
| `d1_race` | 16384 B=64 m=1 | 34.7 | 55.0 | **1.59×** |
| `d1_race` | 1024 B=1 m=1 | 1.022 | 1.240 | 1.21× |

**This is attributable to the machine and I am attributing it there.** The band —
**1.2× to 1.6×** — matches `d1_pow2`'s independently documented cross-machine degradation
("AoS degraded 1.54× wallaby→node while shuffle-free degraded 1.32×"), and the mechanism is
known: 2.9 GHz vs ~4.1 GHz, 1.25 MB L2 vs 2 MB, and on Ice Lake-SP every 512-bit shuffle
lands on port 5, which is also one of the two FMA ports. **The libraries degrade at least as
much**, which is why the panel's *relative* position improves on the node: `d1_composite`
measured MKL at 0.036–0.037 on wallaby at L=60 B=1 against 0.0612 measured here, a **1.66×**
MKL degradation against composite's 1.48× — and that alone is why composite's MKL gap closed.
(I cannot reproduce the brief's "MKL spans 2.9× between those machines" figure from this
round's data; 1.66× at L=60 B=1 is the largest MKL wallaby→node span I can substantiate here.
The 2.9× figure should be sourced or dropped.)

**Two divergences are not explained by the machine and should be checked:**

* `d1_twiddle` claims 1024 B=1 chain 2.16 and 4096 chains 10.8/10.3, measured
  **3.137 / 15.11 / 14.99** — 45%, 40% and 46% optimistic. Its own r4 record contains the
  reason it should have caught this: "the node DRIFTS between invocations … two runs of the
  SAME cell read 57.4 then 67.6 µs. Only interleaved same-window A/B was trustworthy." Those
  three numbers appear to be non-interleaved readings, i.e. exactly what the entry's own
  lesson forbids.
* `d1_bluestein` claims 10007 B=64 m=1 at 147.2, measured **186.9 min / 190.1 median** —
  27% optimistic, on the scoring node, while its ten other claims match. Its chained claims at
  the same size are exact (136.3/146.2 claimed vs 137.7/145.3 measured), so this looks like one
  bad window rather than a systematic problem.

---

## 5. Which open question this round moved

Per §0(c), `docs/LITERATURE.md` §4 is the 3D corpus's section. This round bears on two of its
items and moves both; a third is replicated rather than moved.

### §4.8 item 6 — "no primary measurement in the corpus for Ice Lake-SP or later server parts … Measure it on the node." **MOVED, twice, and the two results are complementary**

Item 6 was rewritten once already, when Intel's turbo table for the Gold 5218 inverted the
downclocking assumption and left the corpus concluding that 512-bit is strictly preferable at
zero frequency cost — while noting there is no primary measurement for **Ice Lake-SP**. This
round supplies two.

**(a) On Ice Lake-SP the AVX-512 shuffle port really does bind, and the effect is invisible on
Sapphire Rapids.** `d1_composite` ran the clean experiment: a port-5 diet on an otherwise
unchanged kernel — 415 → 287 instructions/transform, ~60 fewer p5 µops, achieved by
broadcast-fed signed-FMA stage A and memory-destination `vextractf64x2` pure stores.

* On wallaby (SPR, **two** 512-bit FMA ports): **0.037 before, 0.037 after — zero change.**
  And they killed the frequency explanation by measurement: `scaling_cur_freq` sampled
  mid-run read 3.97–4.10 GHz for *both* binaries, so it is not downclocking.
* On a80n0 (Ice Lake-SP, shuffles share port 5 with one of two FMA ports): the same change
  took L=60 B=1 m=1 from 0.0489 to **0.0430 (−12%)** and closed the MKL gap at B=512 m=1 from
  1.20× to **1.12×**.

This is `docs/LITERATURE.md` §4.4/§04 §8.1's port-5-sharing prediction confirmed on a server
Ice Lake part, with the machine as the isolating variable. **The corollary is the transferable
one and belongs in the corpus: an AVX-512 shuffle diet is worth ~10% on Ice Lake-SP and
exactly nothing on Sapphire Rapids, so SPR is the wrong stopwatch for this entire class of
optimisation.** `d1_composite`'s own record states the kill criterion in advance — "if the r4
leaderboard shows zmm2x2-v2 LOSING to MKL by more than r3's base did, flip
`-DUSE_YMM2_BATCH` into the default" — and the criterion was not met, so the 512-bit default
stands on measured grounds.

**(b) At large L on Ice Lake-SP the binding resource is L2 *table* footprint, not ports.**
`d1_pow2` diagnosed this from on-node PMU and acted: compact (c,s)-pair twiddle tables at
L≥1024, halving the tables' L2 footprint. Measured: **−22%** at 1024 B=1 m=1, **−38%** at
16384 B=1 m=1, −33% at 16384 B=1 chain, −19% at 16384 B=64 chain. Its record states it
plainly: "the ICX large-L bottleneck was table-bloated L1/L2 fill traffic, not ports." On a
part with 1.25 MB L2 the twiddle table is a first-class competitor with the data for cache,
which the corpus nowhere anticipates. The negative half is equally worth having: a **full
Bailey four-step engine was built, measured and removed**, killing the survey's and
`d1_planner`'s standing recommendation for 16384.

### §4.6 — "Model versus search for the instruction schedule." **MOVED, and the answer includes an axis §4.6 lists but nobody had priced**

§4.6's open question ends: "a near-exhaustive search over {schedule variant × unroll depth ×
batch-loop placement × copy-or-not × **compiler flags**} costs minutes. Do it." This round did
the last item, at ship time rather than plan time, and §3.4(c) is the price tag: **the same
source relinked under different alignment/placement flags is worth ~14% in three independent
chained cells**, and at L=32 B=1 chained it beats four structurally different implementations
that all cluster at 0.070. So search wins on an axis that is not the schedule at all, and the
figure to carry forward is that **placement noise of ~14% currently sits underneath every
number on this board**. §4.6's framing — that the schedule is "the primary thing to search" —
is not contradicted, but it is now demonstrably not the *only* thing.

### §4.4 — "Split vs interleaved complex." **Replicated, not moved**

r3 established the reconciliation (split-across-batch wins when the boundary transpose
amortises; interleaved-in-lane wins when it cannot) from `d1_prime`'s kernels. r4 replicates it
in **two independent codebases**: `d1_batchlane` ported the interleaved-pair design and went
−43% at 13 B=512 m=1 and −24% at 31 B=512 m=1; `d1_planner` did the same and went −80%.
Independent replication in a different vehicle is worth recording, but the conclusion is r3's.

### Not moved

§4.1 (register liveness in batch-vectorised codelets), §4.2(b) and (c) (the
symmetric/antisymmetric convolution split; the Winograd-17 op count), and the three
3D-geometry items §4.3/§4.5/§4.7. §4.2(a) stays settled as r3 left it, and r4 adds one
confirmation at the crossover: at 31 B=512 m=1 the dense conjugate-symmetric `d1_prime`
(0.0484) still beats Rader/Agarwal–Cooley `d1_rader` (0.0717) by 1.48×, while at 1021
`d1_rader` takes the non-batched chain outright at 7.128.

---

## 6. The single highest-value thing r5 should attack, per geometry

Grouped by structure, because that is how the board actually behaves. One item each, and each
is the largest *resolved* gap for that group rather than the largest nominal one.

| geometry | attack this | why it is the highest-value move |
|---|---|---|
| **L = 4096** | **`d1_pow2`'s 0.77× at B=1 m=1** (8.888 vs MKL 6.828) | The largest hole on the board by a wide margin — every other loss is ≥0.85×. It is a single-call, non-batched, unbatched cell where MKL is 30% ahead with an in-cell spread of 0.0%, so it is not a noise artefact. The Bailey four-step was already built and rejected here, so the next move is *not* another pass-count scheme: it is the one lever pow2's table compaction has not yet reached at 4096, since 4096's tables are the size where compaction paid least. |
| **L = 16384** | **`d1_twiddle`'s 0.85× at B=64 m=1** (57.63 vs fftw_patient 48.84) | Second-largest resolved gap. `d1_pow2` is 10% better than twiddle in the same cell (57.87 min vs 55.58 — effectively tied) and both lose, so this is a *shared* structural deficit at a 32 MB working set, not one entry's bug. twiddle's own next-round item 1 names the fix and calls it a full-round port: pow2's blocked split-complex `[8re|8im]` engine for L≥128. Do that once, in pow2, and let the others adopt — the r4 pattern shows adoption is cheap and effective. |
| **L = 32, 128** (small pow2, batched single-call) | **The three resolved m=1 losses to MKL** (32 B=512 0.88×, 128 B=512 0.89×, 128 B=1 0.92×) | These are the only cells where MKL holds a *resolved* lead at small L, and §3.4(c) says why they may be reachable without new algorithms: placement is worth 14% and these gaps are 8–12%. Cheapest credible experiment on the board — run `d1_race`'s `+al64` variant lanes against `d1_pow2`/`d1_twiddle` standalone at exactly these three cells before writing any kernel. |
| **L = 13, 31** (small primes) | **`d1_batchlane`'s three B=1 chain regressions** (13 +15%, 32 +18%, 64 +14%) | The only established regressions of the round, all from one rewrite (the "natural-row four-step" B=1 chains), all undisclosed as regressions in its record. It cost the panel two cells it held in r3. Revert-or-fix is a bounded task and recovers held ground, which beats any new attempt at these sizes — `d1_prime` is already 3.9–4.6× over the best library at 31. |
| **L = 60** | **Finish the port-5 diet** — the cell is now 1.12× from MKL, from 1.20× | `d1_composite` states the remaining fat exactly: 37 irreducible `vpermilpd` and 120 two-deep masked-broadcast merge chains, with a specific A/B proposed (32 B load + `vinsertf64x4`-mem = 1 load + 1 p05 blend). It is the only cell on the board where the mechanism is understood, the next step is written down, and the gap is under 15%. |
| **L = 64** | **Resolve it, do not optimise it** | All four cells are panel wins but three are single-call with ±16–29% spreads and the B=512 m=1 "win" (planner 0.0421 vs mkl 0.0483) has `d1_pow2` at 0.0448 marked inside the noise. This is one of the 9-run top-ups `RESCORE_PLAN.md` schedules; it needs runs, not code. |
| **L = 1021, 65537** (primes with smooth N−1) | **Nothing structural — bank it** | `d1_rader` is 1.12–2.26× over the best library in all eight cells and its fused mid pass took another 14% this round. The remaining item in its own record is B=16 batched at 65537 (1328 median against a 1119 min — a 41% spread), which is again a run-count problem. |
| **L = 10007, 100003** (awkward primes) | **`d1_bluestein`'s batched single-call cells** (10007 B=64, 100003 B=8) | Both are the only two cells where a claimed number diverged 27% with no machine explanation (§4), *and* both are marked inside the noise at ±39%. Establish whether the 1.12–1.15× batched wins are real before investing in the Agarwal–Cooley split further; the non-batched cells are solid at 1.23–1.64×. |
| **L = 1024** | **The two marginal single-call losses** (B=1 0.92×, B=512 0.95× inside noise) | Lowest-value of the set and listed for completeness: `d1_pow2` already improved 22% here and both gaps are within a run-count top-up of parity. |

**And one item that is not per-geometry but outranks most of the above:** finish
`RESCORE_PLAN.md`. Nine of this round's 52 verdicts turn on single-call cells whose noise band
is wider than the gap being reported, and four are explicitly marked as ties. 1.7 hours of
sharded compute converts about nine of those into decisions and demonstrates that the rest are
genuine ties. No amount of implementer effort substitutes for it.

---

## 7. Curation — what this round keeps

Against `docs/CURATION.md`'s four grounds, in its order.

**Ground 1 — the fastest correct entry per geometry.** Thirteen lengths, 52 cells; the entries
that are fastest somewhere are `d1_race` (25), `d1_bluestein` (4), `d1_prime` (3),
`d1_batchlane` (3), `d1_planner` (2), `d1_composite` (2), `d1_pow2` (2), `d1_twiddle` (1),
`d1_rader` (1). Discounting `d1_race`'s wins to the sibling whose kernel it ships (§3.4(a)),
the algorithm owners of the board are: **`d1_prime`** (13, 31), **`d1_composite`** (60),
**`d1_pow2`** (1024, 4096, 16384 and the compaction that moved them), **`d1_rader`** (1021,
65537), **`d1_bluestein`** (10007, 100003), **`d1_batchlane`** (32, 64, 128 chained).

**Ground 2 — a structurally different runner-up when it is close.** `d1_rader` at L=13/31 is
1.5–4.6× behind `d1_prime` and is *not* close — but it is close at 1021 (within 1.2% of
`d1_race`, which ships it) and it owns 65537 outright. It qualifies on ground 1, not 2.
`d1_bluestein` is the structural alternative at 65537 and 1021 (chirp-Z vs Rader conv) and is
1.3–1.8× behind — outside the 20% window, so it is kept on ground 1 for 10007/100003 rather
than as a runner-up.

**Ground 3 — instructive failures whose record documents the number that killed them.** Two
are worth the shelf space. `d1_pow2` **built, measured and removed a full Bailey four-step
engine** at 16384, with numbers — that stops the next panel spending a round on the survey's
own recommendation. `d1_composite` recorded the *negative* half of the port-5 diet: 415→287
instructions and ~60 fewer p5 µops bought **zero** on Sapphire Rapids, with the frequency
hypothesis killed by direct `scaling_cur_freq` measurement, and 12% on the node. Both entries
qualify on ground 1 anyway; the failures are noted here so `NOTES.md` records them.

**Ground 4 — anything that beat a library.** 43 of 52 cells, so this ground selects almost
everything and does not discriminate. It does rescue `d1_twiddle`: it beats a library at 32
B=1 m=1 (1.34×) and is the panel's best entry at 16384 B=64.

**Promoted — 8 of 9.**

| entry | ground | note |
|---|---|---|
| `d1_prime` | 1 | Fastest at 31 in both non-batched cells at **3.92–4.58× over the best library** — the widest margins on the board. Its interleaved-pair kernel design was adopted by three other entries this round and is the single most-copied artefact in the campaign. |
| `d1_pow2` | 1, 3 | The round's best-evidenced work: PMU-driven diagnosis, the compact-table fix worth 22–38% at large L, a documented rejected alternative, and claims that match measurement to 1–3% at twelve chained cells. The model for how to write a round. |
| `d1_rader` | 1 | Owns 65537 (1.72–2.26× over `fftw1d_patient`) and 1021's non-batched chain; the fused mid pass — 9 passes not 10, spectrum never materialised — is a genuinely new structure and its claims verify at ten of twelve cells. |
| `d1_bluestein` | 1 | Owns 10007 and 100003 chained, where nothing else applies. Agarwal–Cooley coprime 2D convolution with fused chirp/CRT entry and exit; also the source of the split-complex Stockham core three other entries build on. |
| `d1_composite` | 1, 3 | Fastest at 60 non-batched, and carries §5(a) — the port-5 diet A/B'd across two machines, with the kill criterion stated before the run and the frequency confound eliminated by measurement. |
| `d1_batchlane` | 1, 3 | Fastest at 32/64/128 batched chained (1.96–2.68× over library). Promoted **with the regression on the record**: its B=1 chain rewrite cost 14–18% at three cells and lost two cells it held in r3, and the exemplar is the code that did it. That is exactly what ground 3 is for. |
| `d1_twiddle` | 1, 4 | Marginal, and promoted for the artefact rather than the ranking: the exact 1D twiddle-table library (`d1tw_cexp` quadrant-exact ~1 ulp, `d1tw_chirp` integer-reduced, `d1tw_stage` v1/v2 broadcast-pair and lane-major formats) is a distinct thing other entries consume, and it is the panel's best entry at 16384 B=64. Its own 1024/4096 chained claims are 40–46% optimistic (§4) and the record should be read with that flagged. |
| `d1_race` | 3 | Promoted on ground 3, not ground 1 — its 25 cell wins are its siblings' (§3.4(a)), and the instructive failure is its plan-cost distribution: **14.34 s to plan a 14.8 ns transform**, 57% of its 46 s total spent on the two smallest cells on the board (while `fftw1d_patient` pays 846 s across the same board, so the criticism is about proportion, not about racing). But §3.4(c) is a real and independent finding worth the next panel's attention: the `+al64` ship-time placement probe beats every standalone sibling by ~14% at three chained cells, which means placement is a confound under every number on this board. |

**Not promoted — `d1_planner`.** It is fastest at two of 52 cells, which under ground 1 would
normally carry it. It is excluded on the ground `CURATION.md` states as an exclusion rather
than a criterion: **"do not promote … entries whose strategy record is missing — the record is
what makes the code useful later."** `d1_planner.c` changed by 1012 lines this round, the
largest diff of any entry, and `strategies/d1_planner.md` has no r4 section at all (§3.5).
Two further considerations point the same way: both of its wins are marginal — 64 B=512 m=1
has `d1_pow2` 0.0448 within the noise band, and 13 B=1 chain is a cell `d1_batchlane` held in
r3 and regressed out of — and its own r3 record describes it as adopting `d1_prime`'s and
`d1_pow2`'s codelets, both of which are promoted, which places it under "do not promote
near-duplicates of an already-promoted entry."

**Nothing is lost by the exclusion:** `impl_4/d1_planner.c` is versioned per
`CURATION.md`'s per-round-source rule, so the code is preserved and a future round can diff
it. What is missing is the prose, and promoting the code without it would put an
unexplainable artefact on the reading list. **If `d1_planner`'s implementer writes the r4
section, this decision should be revisited before r5's panel reads the exemplars** — the entry
is close to qualifying and it is the only routing-layer alternative to `d1_race`.

---

*Monitor note: `results/d1_r3/leaderboard.txt` and `results/d1_r4/leaderboard.txt` use
different column sets and different ranking statistics; §2's comparisons are min-against-min.
`exemplars/d1_r3/NOTES.md` is still an unfilled skeleton and should be written from r3's
`VERDICT.md` while that file is at hand.*

PROMOTE: d1_prime d1_pow2 d1_rader d1_bluestein d1_composite d1_batchlane d1_twiddle d1_race
