# VERDICT — panel round d1_r5

Monitor pass over `results/d1_r5/`. Judgement only; no timing was re-run. Sources graded are
`impl_5/` (= the `impl` symlink), 9 panel entries plus `baseline_dft`, against 7 library
configurations, over 52 graded cells / 530 leaderboard rows.

---

## 0. Corrections to the brief, before any numbers

(a)–(c) are the same three corrections the r3 and r4 monitors recorded. They are still not
fixed in the brief, so they are restated rather than silently worked around. (d) is new.

**(a) The geometries are not L = 6, 8, 17, 36.** Those are the *3D* campaign's cube edges
(`bench/geom/`). This is the 1D campaign and its graded set is
**L = 13, 31, 32, 60, 64, 128, 1021, 1024, 4096, 10007, 16384, 65537, 100003** — thirteen
lengths × {B=1, batched} × {single call, chained} = 52 cells. `leaderboard.txt` contains no
L=6/8/17/36 row and never has. §1 reports all thirteen.

**(b) The scoring node is not a Cascade Lake Xeon Gold 5218.** `environment.txt` reads
**Intel Xeon Gold 6326 @ 2.90 GHz** with `avx512_vnni avx512_bitalg avx512_vbmi avx512ifma
avx512_vpopcntdq` in the ISA list — that combination is **Ice Lake-SP**, not Cascade Lake
(which has none of those four). 1.25 MB L2/core, and Ice Lake-SP does *not* carry
Skylake/Cascade Lake's severe AVX-512 licence downclocking. The panel develops largely on
**wallaby, Gold 6448Y (Sapphire Rapids), 2 MB L2, ~4.1 GHz** and is scored on **a80n0, Gold
6326 (Ice Lake-SP), 1.25 MB L2, 2.9 GHz base**. That difference matters in §4 and §5, and this
round it was measured directly rather than inferred.

I again could not substantiate the brief's "MKL alone spans 2.9× between those machines" from
this round's data. The largest wallaby→node MKL span any entry documents this round is
**1.26×** (`d1_composite`, L=60 B=512: 0.043–0.044 on the node vs the wallaby figure implied by
its own SPR sanity run). The 2.9× figure should be sourced or dropped from the brief.

**(c) `docs/LITERATURE.md` §4 is the 3D corpus's open-questions section.** Three of its eight
items (§4.3 axis fusion, §4.5 L=8 padding, §4.7 vector-radix) are 3D-geometry questions a 1D
round cannot address. §5 answers against the items that are hardware- and kernel-structure
questions, which this round bears on heavily.

**(d) The ranking statistic changed between r4 and r5, per `RESCORE_PLAN.md`.** Single-call
cells now get **9 runs** instead of 3; the statistic is the median in both rounds. Chained
cells remain 3 runs. All r4→r5 comparisons in §2 are therefore **median-against-median**
(unlike r4's verdict, which compared min-against-min because r3→r4 was a statistic change
rather than a run-count change). Where the median is contaminated I say so and give the min
alongside — §2.0 explains why that matters more this round than in any previous one.

---

## 1. Headline per geometry — fastest correct panel entry vs best library

Median µs/transform, the round's ranking statistic; "spread" is the in-cell run spread.
"*(tie)*" means `leaderboard.py` flags the panel-vs-library gap as smaller than the
resolvability band (which per commit `1dddd3db` is now based on standard error, not the
observed range) — those cells are demonstrated ties, not results, in either direction.
`baseline_dft` (the harness's O(L²) floor) is excluded from "library".

| L | cell | best panel (median) | spread | best library (median) | spread | verdict |
|---|---|---|---|---|---|---|
| **13** | B=1 m=1 | d1_planner 0.0169 | 31.9% | mkl1d_dfti 0.0259 | 26.6% | **WIN 1.53x** |
|  | B=1 chain m=200000 | d1_race 0.0338 | 0.0% | fftw1d_custom 0.0626 | 21.1% | **WIN 1.85x** |
|  | B=512 m=1 | d1_prime 0.0093 | 14.3% | fftw1d_patient 0.0141 | 16.6% | **WIN 1.52x** |
|  | B=512 chain m=2000 | d1_planner 0.0155 | 0.2% | fftw1d_custom_soa 0.0267 | 14.3% | **WIN 1.72x** |
| **31** | B=1 m=1 | d1_race 0.0536 | 26.6% | fftw1d_custom 0.2149 | 22.3% | **WIN 4.01x** |
|  | B=1 chain m=100000 | d1_race 0.0510 | 0.2% | fftw1d_custom 0.2090 | 20.8% | **WIN 4.10x** |
|  | B=512 m=1 | d1_prime 0.0484 | 16.6% | fftw1d_custom_soa 0.1014 | 36.8% | **WIN 2.10x** |
|  | B=512 chain m=1200 | d1_race 0.0459 | 1.9% | fftw1d_custom_soa 0.0776 | 0.0% | **WIN 1.69x** |
| **32** | B=1 m=1 | d1_race 0.0192 | 32.8% | mkl1d_dfti 0.0255 | 62.4% | **WIN 1.33x** |
|  | B=1 chain m=100000 | d1_batchlane 0.0599 | 0.4% | mkl1d_dfti 0.1310 | 0.0% | **WIN 2.19x** |
|  | B=512 m=1 | d1_planner 0.0154 | 13.9% | mkl1d_dfti 0.0174 | 0.1% | **WIN 1.13x** |
|  | B=512 chain m=1000 | d1_twiddle 0.0368 | 14.0% | fftw1d_custom_soa 0.0665 | 0.2% | **WIN 1.81x** |
| **60** | B=1 m=1 | d1_composite 0.0429 | 19.4% | mkl1d_dfti 0.0617 | 24.1% | **WIN 1.44x** |
|  | B=1 chain m=60000 | d1_race 0.1253 | 0.0% | fftw1d_patient 0.2415 | 0.1% | **WIN 1.93x** |
|  | B=512 m=1 | d1_planner 0.0529 | 18.0% | mkl1d_dfti 0.0514 | 11.5% | loss 0.97x *(tie)* |
|  | B=512 chain m=600 | d1_composite 0.0605 | 17.9% | fftw1d_custom_soa 0.1655 | 24.7% | **WIN 2.74x** |
| **64** | B=1 m=1 | d1_twiddle 0.0449 | 19.2% | mkl1d_dfti 0.0438 | 21.5% | loss 0.98x *(tie)* |
|  | B=1 chain m=60000 | d1_batchlane 0.0848 | 13.8% | mkl1d_dfti 0.2370 | 0.1% | **WIN 2.79x** |
|  | B=512 m=1 | d1_planner 0.0382 | 33.9% | mkl1d_dfti 0.0422 | 43.4% | win 1.10x *(tie)* |
|  | B=512 chain m=500 | d1_race 0.0775 | 0.2% | fftw1d_custom_soa 0.1514 | 2.2% | **WIN 1.95x** |
| **128** | B=1 m=1 | d1_race 0.1103 | 16.5% | mkl1d_dfti 0.1044 | 9.3% | loss 0.95x *(tie)* |
|  | B=1 chain m=30000 | d1_race 0.1651 | 13.8% | fftw1d_patient 0.4852 | 1.7% | **WIN 2.94x** |
|  | B=512 m=1 | d1_race 0.1495 | 16.3% | mkl1d_dfti 0.1419 | 11.5% | loss 0.95x *(tie)* |
|  | B=512 chain m=250 | d1_race 0.1972 | 8.1% | fftw1d_custom_soa 0.3999 | 0.7% | **WIN 2.03x** |
| **1021** | B=1 m=1 | d1_rader 7.4897 | 14.6% | mkl1d_dfti 9.4066 | 14.0% | **WIN 1.26x** |
|  | B=1 chain m=2000 | d1_rader 7.0898 | 1.2% | mkl1d_dfti 12.98 | 3.6% | **WIN 1.83x** |
|  | B=256 m=1 | d1_planner 8.4150 | 5.7% | mkl1d_dfti 8.7543 | 21.9% | win 1.04x *(tie)* |
|  | B=256 chain m=400 | d1_race 6.6615 | 1.6% | mkl1d_dfti 12.40 | 0.0% | **WIN 1.86x** |
| **1024** | B=1 m=1 | d1_twiddle 1.2823 | 2.0% | mkl1d_dfti 1.0867 | 15.0% | **LOSS 0.85x** |
|  | B=1 chain m=4000 | d1_race 2.2806 | 13.9% | mkl1d_dfti 4.1669 | 14.0% | **WIN 1.83x** |
|  | B=512 m=1 | d1_twiddle 1.7436 | 0.9% | fftw1d_patient 1.8943 | 6.2% | **WIN 1.09x** |
|  | B=512 chain m=2000 | d1_race 2.3351 | 0.2% | mkl1d_dfti 4.8984 | 1.0% | **WIN 2.10x** |
| **4096** | B=1 m=1 | d1_race 7.2191 | 15.5% | mkl1d_dfti 6.0417 | 1.4% | **LOSS 0.84x** |
|  | B=1 chain m=1000 | d1_pow2 9.8061 | 1.5% | fftw1d_patient 19.21 | 0.1% | **WIN 1.96x** |
|  | B=256 m=1 | d1_race 9.9015 | 12.9% | mkl1d_dfti 10.07 | 38.2% | win 1.02x *(tie)* |
|  | B=256 chain m=400 | d1_race 10.23 | 0.6% | mkl1d_dfti 22.73 | 2.6% | **WIN 2.22x** |
| **10007** | B=1 m=1 | d1_bluestein 110.64 | 14.8% | fftw1d_patient 200.03 | 3.5% | **WIN 1.81x** |
|  | B=1 chain m=400 | d1_race 122.42 | 14.3% | fftw1d_patient 230.58 | 3.9% | **WIN 1.88x** |
|  | B=64 m=1 | d1_race 113.10 | 34.0% | fftw1d_patient 208.31 | 3.4% | **WIN 1.84x** |
|  | B=64 chain m=80 | d1_race 130.68 | 3.6% | fftw1d_patient 237.21 | 1.6% | **WIN 1.82x** |
| **16384** | B=1 m=1 | d1_race 33.83 | 47.3% | fftw1d_patient 32.50 | 3.0% | loss 0.96x *(tie)* |
|  | B=1 chain m=250 | d1_pow2 50.33 | 1.3% | fftw1d_patient 82.15 | 0.8% | **WIN 1.63x** |
|  | B=64 m=1 | d1_pow2 44.44 | 23.8% | fftw1d_patient 46.08 | 13.0% | win 1.04x *(tie)* |
|  | B=64 chain m=150 | d1_pow2 51.08 | 0.8% | fftw1d_patient 98.41 | 1.2% | **WIN 1.93x** |
| **65537** | B=1 m=1 | d1_race 752.50 | 9.8% | fftw1d_patient 1467 | 0.9% | **WIN 1.95x** |
|  | B=1 chain m=60 | d1_race 683.92 | 4.3% | fftw1d_patient 1628 | 0.3% | **WIN 2.38x** |
|  | B=16 m=1 | d1_rader 933.86 | 14.5% | fftw1d_patient 1539 | 2.1% | **WIN 1.65x** |
|  | B=16 chain m=20 | d1_rader 744.65 | 2.2% | fftw1d_patient 1763 | 1.8% | **WIN 2.37x** |
| **100003** | B=1 m=1 | d1_bluestein 2564 | 35.4% | fftw1d_patient 2715 | 1.9% | win 1.06x *(tie)* |
|  | B=1 chain m=40 | d1_bluestein 2313 | 0.3% | fftw1d_patient 3116 | 1.5% | **WIN 1.35x** |
|  | B=8 m=1 | d1_race 2399 | 37.0% | fftw1d_patient 2806 | 1.5% | **WIN 1.17x** |
|  | B=8 chain m=15 | d1_bluestein 2728 | 0.9% | fftw1d_patient 3169 | 1.6% | **WIN 1.16x** |

### Score: 45 of 52 cells, and only two library wins survive resolution

* **Chained cells: 26 of 26 to the panel**, by 1.16× to 4.10×, geometric mean **2.01×**. As in
  r3 and r4, the libraries have no competitive chained path anywhere on the board.
* **Single-call cells: 19 of 26**, geometric mean **1.27×**.
* Aggregate over all 52 cells, geometric mean of best-library ÷ best-panel: **1.602×**.

Computed the same way for all three rounds, the campaign trend is monotone:

| round | panel cells | library cells | of which **resolved** losses | geomean |
|---|---|---|---|---|
| r3 | 41 | 11 | 11 | 1.436× |
| r4 | 43 | 9 | 7 | 1.543× |
| **r5** | **45** | **7** | **2** | **1.602×** |

The seven library-won cells, and whether the loss is real:

| L | cell | library | best panel | gap | resolved? |
|---|---|---|---|---|---|
| 4096 | B=1 m=1 | mkl 6.042 | d1_race 7.219 | 0.84× | **REAL — the largest hole on the board** |
| 1024 | B=1 m=1 | mkl 1.087 | d1_twiddle 1.282 | 0.85× | **REAL — and it moved the wrong way** |
| 16384 | B=1 m=1 | fftw_patient 32.50 | d1_race 33.83 | 0.96× | tie (panel spread 47.3%) |
| 128 | B=512 m=1 | mkl 0.1419 | d1_race 0.1495 | 0.95× | tie |
| 128 | B=1 m=1 | mkl 0.1044 | d1_race 0.1103 | 0.95× | tie |
| 64 | B=1 m=1 | mkl 0.0438 | d1_twiddle 0.0449 | 0.98× | tie |
| 60 | B=512 m=1 | mkl 0.0514 | d1_planner 0.0529 | 0.97× | tie |

**Four of r4's nine library wins were converted outright**, and three of those were r4's own
named next-round targets: L=32 B=512 (0.88× → **1.13× win**), L=16384 B=64 (0.85× →
**1.04× win**), L=128 B=512 (0.89× → 0.95× tie), L=1024 B=512 (0.95× → **1.09× win**). L=4096
B=1 narrowed from 0.77× to 0.84× but is still the board's largest gap. Only **L=1024 B=1**
went backwards, from 0.92× to 0.85×.

---

## 2. What changed since d1_r4, per geometry

### 2.0 First, a measurement finding, because it governs every claim in this section

**The scoring node's per-process execution rate is quantised into three discrete states, and
the round-over-round comparison is worthless until that is accounted for.**

Calibration on the 135 library and `baseline_dft` rows — unchanged code, rebuilt from
identical source — gives r4→r5 median shifts of p50 2.0% / p90 20.3% on single-call cells and
p50 0.4% / p90 12.8% on chained cells. That is broadly the band `RESCORE_PLAN.md` predicted.
But the *shape* of the noise is not what the plan assumes, and it is not a continuum.

Taking every (backend, cell) unit's runs and dividing each run's median execute time by that
unit's fastest run, the 530-row ratio distribution has **two razor-sharp modes**:

| mode | n | mean ratio | sd |
|---|---|---|---|
| A | 348 | **1.1401** | 0.0090 |
| B | 200 | **1.2090** | 0.0081 |

The chained max/min histogram makes the same point visually: 26 cells at 1.14, 10 at 1.21, and
essentially nothing between 1.16 and 1.19. Four things establish that this is the *machine*
and not any entry's code:

1. It appears identically in `mkl1d_dfti`, all five FFTW configurations, `baseline_dft` and
   all nine panel entries.
2. It is **independent of working set**, from 0.4 KB (L=13 B=1) to 32 MB (L=4096 B=256). A
   placement or cache effect scales with the access pattern; a rate scaling does not.
3. It is **not a warm-up artefact**: the fraction of "slow" runs is 32/29/26% for run indices
   1/2/3 and 33–44% for runs 4–9 — no index trend.
4. It is **per-process, not per-window**: only 29 of 312 (L,B,m,run-index) units have every
   backend agreeing on the mode, so it is selected independently per invocation.

The leading hypothesis is a per-core P-state: `environment.txt` records
**`governor: schedutil`**, not `performance`; 1/1.1401 × 3.3 GHz = **2.894 GHz**, which is the
Gold 6326's base clock to three figures; and `d1_composite` independently sampled
`scaling_cur_freq` at **3.3 GHz** in its good windows. I am not asserting this — an uncore or
memory-clock step would look the same — but it is one command to settle: log
`/sys/.../scaling_cur_freq` (or an `IA32_APERF/MPERF` pair) per run in `driver.c` and correlate.
**State that kill criterion before the next sweep, not after.**

The consequence for §2 is direct: **a change of +13% to +15% on this board is one mode step and
is not evidence of anything.** Everything below is filtered against that.

### 2.1 Improvements, by geometry

| L | what improved | mechanism, from the entries' own records |
|---|---|---|
| **60** | `d1_planner` **−67% / −62%** at both m=1 cells (0.1356→0.0446, 0.1341→0.0504 min); `d1_batchlane` −34% / −19%; `d1_composite` took B=1 m=1 to 0.0429 | All three adopted `d1_composite`'s PFA-60 kernels. Composite's own change is the round's best-evidenced result and is discussed in §5. This is the largest single mass of improvement on the board and it is *diffusion*, as in r4. |
| **128** | `d1_planner` **−39% / −52%** (min) at the two m=1 cells; `d1_batchlane` −21% at B=512 | planner from the paired-p zmm stages; batchlane from a **new** `fft128_codelet` of its own — radix-4/4/8 with branch-grouped intermediates, the only genuinely new kernel written this round outside the library layers. |
| **16384** | `d1_pow2` **−24%** at B=64 m=1 (57.87→44.13 min) and took the cell from `fftw1d_patient`; `d1_bluestein` −43%; `d1_twiddle` −27%; `d1_planner` −30% | pow2's **deterministic huge-page arena** (adopted from bluestein r2/r3 via planner r3) killed its own 44.6/64.4 µs cross-invocation bimodality — see §5(b). twiddle's register-resident fused first-stage pair. |
| **4096** | `d1_race` −25%, `d1_planner` −39% at B=256, `d1_twiddle` −19–24%, `d1_pow2` −16% at B=1 | pow2's arena (−9.5% at B=256, stated and measured); twiddle's `vsx44` register-resident stage-pair fusion, which **beat pow2's own tiled version's gate** (−12% at 1024 where pow2's tile lost 8%). |
| **1024** | `d1_twiddle` −22% / −24% at the two m=1 cells, taking B=512 from FFTW; `d1_pow2` −22% at B=512 | twiddle's compact (c,s) v3 tables + single-scratch ping-pong; pow2's batch-gated first-stage fusion (fused wins at B=1, **loses 6% at B=512** because the fused reads are quad-scattered once input streams from L3 — a gate, not a revert). |
| **65537** | `d1_rader` **−32%** at B=16 m=1 (1370.6→930.9 min); chains −5 to −8% | The round's cleanest surprise: **exit-scatter write prefetch (ET0)**. The random 16-byte output stores were paying a cold RFO per pair *even at B=1*, because the transform's own ping-pong traffic evicts the output. Plus radix-64 middle stages, 7 M-length passes instead of 9. |
| **10007** | `d1_bluestein` −24% at B=1 m=1 and **−40%** at B=64 m=1; `d1_race` −16% (min) at B=64 | radix-16 two-layer L1-tile stages ported from `d1_rader`, in flavor-specific schedules, plus `stmid8` fusing the trailing-8 row schedule and realigned NT batched exits. |
| **1021, 100003, 65537 (planner)** | `d1_planner` −17 to −20% almost everywhere | The paired-p zmm stages `st8@s4` / `st3@s4` / `st5@s12` fixed the one stage every plan ≥1024 was running at half vector width. Single highest-leverage line-item of the round for a single entry. |
| **13, 31** | `d1_prime` −19% at 13 B=512 chain, −20% at 13 B=1 chain (via race) | The latency-shaped divider-free map: Goldschmidt sqrt + early-seeded `rcp14`, taking all four chained cells. |

### 2.2 Regressions

Filtered against §2.0. A change in the 1.13–1.16 band with no corroborating code change is
scored as a mode step, not a regression.

**Established — three, all in `d1_race`, all the same failure:**

| entry | cell | r4 median | r5 median | r4 min | r5 min | assessment |
|---|---|---|---|---|---|---|
| `d1_race` | 1021 B=256 m=1 | 7.316 | **9.538** (+30%) | 7.260 | 8.041 (+11%) | Real: **both** statistics moved, and it lost the cell to `d1_planner`. |
| `d1_race` | 100003 B=1 m=1 | 2323.7 | **3004.3** (+29%) | 2319.5 | 2267.5 (−2%) | The min *improved*; the median blew up with a 32.8% spread. The shipped kernel is fine; the placement probe failed to converge in most processes. |
| `d1_race` | 10007 B=1 m=1 | 124.69 | **142.56** (+14%) | 124.15 | 109.39 (−12%) | Identical signature: best draw 12% better, median 14% worse. |

This is exactly the failure mode `d1_race` set out to fix this round, and its own record opens
by naming it: *"the verdicts are right; the MEDIANS lose the cells."* It **succeeded** where it
targeted — 100003 B=8 m=1 went 3398.5 → 2398.7 (−29%) and is now a 1.17× resolved panel win —
and **regressed two adjacent cells the same way**. Its wisdom-referenced probe is therefore
demonstrated to work and demonstrated not to generalise across (L, B) yet. See also §3.4.

**Not established — the L=32/60/64/128 chained block, which is the mode:** at L=32 B=1 chain
`d1_planner` +14.0%, `d1_twiddle` +13.9%; at L=60 B=1 chain `d1_composite` +13.8%, `d1_race`
+13.7%, `d1_bluestein` +14.2%; at L=64 B=1 chain `d1_pow2` +13.7%, `d1_race` +13.8%,
`d1_twiddle` +13.7%; at L=32/64 B=512 chain `d1_batchlane` +14.4%/+14.3%, `d1_race` +15.0%.
Twelve "regressions" across eight unrelated codebases, every one within 0.5% of mode A, in
cells three of the entries state in writing they did not touch. Raw per-run data confirms it
directly: at L=32 B=1 chain `d1_pow2` measured 0.0804 / 0.0804 / 0.0706 µs across its three
runs — the same binary, two modes, 13.9% apart — while `d1_twiddle` drew 0.0804 three times
and `d1_batchlane` drew 0.0599 twice. **The median picked the mode, not the code.**

**And one correction to the r4 verdict, which the r5 data overturns.** r4 §2 certified three
`d1_batchlane` chained regressions as "established (chained regime)" and conditioned its
promotion on them. All three have returned in r5 to their r3 values, to three or four
significant figures, in code `d1_batchlane`'s r5 record states was untouched:

| cell | r3 | r4 | r5 | r4/r3 |
|---|---|---|---|---|
| 13 B=1 chain | 0.0392 | 0.0451 | **0.0392** | 1.150 |
| 32 B=1 chain | 0.0591 | 0.0699 | **0.0599** | 1.183 |
| 64 B=1 chain | 0.0848 | 0.0965 | **0.0848** | 1.138 |

Three ratios in the mode band, exact reversion, and no code motion. **`d1_batchlane`'s r4
"natural-row four-step regression" did not happen.** The accountability finding built on it in
r4 §2 — that batchlane reported the regressed numbers while asserting nothing regressed — should
be withdrawn: the entry reported what it measured, and what it measured was the machine. The
same applies to `d1_rader`'s apparent +19% at 13 and 31 B=1 m=1, where the medians are 0.0568 /
0.2113 (r3), 0.0471 / 0.2112 (r4), 0.0563 / 0.2112 (r5) — r4 was the outlier, and at 31 the
median is identical in all three rounds.

---

## 3. Adversarial pass — correctness, builds, crashes, gaps

### 3.1 Nothing failed correctness, and I checked that the check can fail

* `check.log`: **zero** occurrences of `FAIL`, `mismatch`, `error`, `nan` or `inf`,
  case-insensitive, across all 1325 lines. Every graded configuration reports `PASS`.
* **Three distinct gate types all had to fire**, not one: 795 `rel_l2 … (tol 1e-12)` gates,
  265 `map-chain m=… (anchor …, tol 1.0e-10)` gates, and 265 strict `map-2-step … (tol
  3.0e-14)` gates. A backend that produced no output would show as missing leaderboard rows,
  not as passes; all 530 rows carry a populated `correctness` column.
* The reported errors are **physically sane and differentiated**, which a stub cannot fake:
  1.3e-16 at L=13/32 rising monotonically to 1.4e-15 at L=65537, and the O(L²) `baseline_dft`
  sitting 2–4× worse than the FFT entries at the same L. At L=13 B=1, `d1_rader`'s
  convolution path reports 2.7e-16 against `d1_prime`'s dense 1.9e-16 — the conv route is
  measurably less accurate than the direct one, exactly as it must be.
* **Margins are not threshold-hugging.** Worst single-call residual on the board is 1.4e-15
  against a 1e-12 tolerance (700× margin); worst chained residual is `baseline_dft` at 3.8e-12
  against 1e-10 (26× margin), and the worst *panel* chained residual is `d1_rader` at 3.2e-12
  (31× margin). Median chained residual is ~1e-13.

**No fast wrong answers.** I looked specifically for the shape — an entry anomalously fast at a
cell *and* anomalously inaccurate there. Nine cells have the fastest entry also carrying the
cell's largest error, but every one is a ≤1.2× error difference three decades inside tolerance,
and the pattern is algorithmically expected (at 65537 and 1021 the winners are Rader/Bluestein
convolution routes at 1.4e-15 against direct FFTW's 1.2e-15). The screen is not one-sided:
`mkl1d_dfti` trips the same flag at 4096 B=1. The largest error-vs-cell-best ratios on the
whole board belong to **FFTW's own rows** (18.5× at L=32 B=1 chain, 1.7e-14 vs 9.2e-16), and
the O(L²) floor is worse still at 77×. Nothing panel-side is anomalous.

### 3.2 Nothing failed to build, nothing crashed, nothing is missing — but one warning is now two rounds old

* `build_errors.txt` is **84 lines of warnings, zero errors**; all ten sources compiled and
  linked. `agents/exits.txt` shows nine `exit=0`.
* `failures.txt` does not exist, which in this harness means nothing crashed or hung —
  consistent with all 530 rows being populated.
* `timing.err` contains only `<entry>: does not support L=<n>` declines, which match every
  entry's declared scope (see below).

**The `d1_batchlane` signed-overflow UB is still there, and r4 flagged it.** r4 §3.2 recorded
it at `impl/d1_batchlane.c:169`; it is now at lines **184 and 195**, in three call sites
(`deint8`, `chain32_reg`, `chain64_reg`):

```
impl/d1_batchlane.c:184:39: warning: iteration 2147483648 invokes undefined behavior
  184 |     for (; j < L; ++j) { re[j] = aos[2*j]; im[j] = aos[2*j + 1]; }
```

`j` is `int`, `2*j` signed-overflows at j = 2³¹. Unreachable at every graded L, so it affected
no number this round either — but `-Waggressive-loop-optimizations` firing means GCC is already
reasoning about the overflow, and a monitor flagging the same latent bug in consecutive rounds
is the point at which it should stop being cosmetic. One `size_t`.

**Missing entries: none unexplained, and coverage did not shrink.** Every entry present in r4 is
present in r5 at every cell it served. Per-cell absences are all declared specialisation:
`d1_composite` serves only L=60 (4/52) and is fastest there in both non-batched cells;
`d1_prime` serves 13 and 31 (8/52); `d1_rader` serves 13, 31, 1021, 65537 — the primes with a
smooth N−1 — and correctly declines 10007 (N−1 = 2·5003, 5003 prime) and 100003, which
`d1_bluestein` covers; `d1_pow2` and `d1_twiddle` decline non-powers-of-two;
`d1_batchlane` declines L>128. `d1_bluestein`, `d1_planner` and `d1_race` serve all 52.
`baseline_dft` is absent above L=1021 batched — a harness time decision (it runs 2188 µs at
1021 B=1, 292× the winner), present at 11 of 13 lengths, enough to keep the floor honest.

### 3.3 The one substantive process problem: `d1_race` did not verify on the scoring node

Eight of nine entries state explicitly that the a80n0 reservation (job 440424) was alive all
session and that their numbers are from a leased core on the scoring node. `d1_race` is the
exception. Its record's measurement header reads *"Measured (wallaby, nice'd core 100, load
~0.5; all machinery checks — the node re-races fresh)"* and every number in its table is a
wallaby number. Its correctness claims are also wallaby claims.

This matters more for `d1_race` than it would for any other entry, because `d1_race` is a
*routing layer* whose entire value is picking the right arm and the right placement **on the
host it runs on**, and its own r5 change #4 is a roster trim justified by which lanes "won on
the node" in r4. Tuning a per-host mechanism on the wrong host is the specific hazard, and §2.2
shows exactly that outcome: the probe fixed the cell it was developed against and broke two
neighbours. The graded correctness of the shipped binary is not in doubt — the driver's own
`check.log` covers `d1_race` at all 52 cells with `PASS` — but the *tuning* is untested on the
scoring node, and that is the difference between its 100003 B=8 win and its 100003 B=1 loss.

### 3.4 `d1_race` wins 24 of 52 cells, and one r4 finding it built on is now retracted

`d1_race` is not an FFT; it races the sibling entries per (L,B) and ships the winner by vtable.
Its 24 wins (down from 29 in r4) are its siblings' kernels, and **every headline in §1 naming
`d1_race` should be read as "the race selected correctly", with §6/§7 crediting the sibling.**
At 65537 B=1 it ships `d1_rader` and lands within 4% of it; at 60 B=1 chain it ships
`d1_composite` and lands within 0.3%.

**r4's most-promoted finding is withdrawn by its own author.** r4 §3.4(c) called the `+al64`
variant lane "the most reusable thing in this round" — the same source relinked under different
alignment flags, worth ~14% at three chained cells, concluding "code and data placement is worth
~14% on Ice Lake-SP." `d1_race`'s r5 record settles it on the node and the answer is mostly no:
it **dropped** `d1_batchlane+al64` and `d1_composite+al64` because *"base won every contested
cell on the node; the r4 wallaby wins did not transfer"*, and dropped `d1_planner+lane16k` and
`d1_composite+zmm2x2` on the same grounds. One lane survived — `d1_pow2+al64`, which won 16384
B=64 and both 1024 chains on the node, and which pow2's owner could adopt outright as
`-falign-functions=64 -falign-loops=32`.

Two of r4's three headline figures for that finding were 1.164× and 1.166×, which sit on the
edge of the §2.0 mode band, and the third (64 B=1 chain, 1.138×) is mode A exactly. In r5 the
same cell it celebrated — L=32 B=1 chain, where race was 14% *ahead* of four clustered siblings
— has race at 0.0700 and `d1_batchlane` at 0.0599, i.e. **17% behind**. The honest statement is
that the r4 result was mostly the machine, that placement search is worth real money in exactly
one measured case, and that the cheap check r4 recommended (run the al64 lanes at the three
losing MKL cells) was run and came back negative. That is a good outcome for the panel — a
speculative lever priced and mostly closed in one round — but it must not stay on the books as
a 14% result.

**Setup cost, for the record.** `d1_race` pays **2.82 s** of plan time across all 52 cells this
round, down from 46.19 s in r4 (its 14.34 s at L=13 B=1 is gone — the wisdom-referenced probe
now stops immediately on a good draw). `fftw1d_patient` pays **843 s** across the same board —
299× more — including 66.9 s at 16384 B=64 and 45.6 s at 4096 B=256, and `fftw1d_measure` pays
46.5 s. Plan-time racing is no longer even the third most expensive planner on the board, and
r4's proportionality criticism of it is now spent.

### 3.5 Process items from r4 that were and were not closed

* **Closed: `d1_planner`'s missing record.** r4 excluded it from promotion solely because
  `strategies/d1_planner.md` had no r4 section for a 1012-line diff. The r5 session wrote both a
  **reconstructed r4 stub** (correctly labelled as a reconstruction, deriving the r4 changes
  from the diff and the impl header) and a full r5 section with a 24-row measured table and a
  "what did not work" block. r4's stated condition for revisiting the decision is met; §7 acts
  on it.
* **Not closed: `exemplars/d1_r3/NOTES.md` and `exemplars/d1_r4/NOTES.md` are still unfilled
  skeletons**, both still carrying `(Fill in: …)` markers, for the third round running. Both
  rounds' `VERDICT.md` files are on disk and contain everything needed to write them.

---

## 4. Claimed numbers vs measured

**The machine-difference explanation that dominated r4 barely applies this round, because eight
of nine entries measured on the scoring node.** Only `d1_race` reports wallaby numbers (§3.3).
The divergences that remain have a different and more interesting cause.

**The claims are right and the board's medians are what diverge.** `d1_pow2` is the clean case —
its claimed on-node minima match the board's **min** column almost exactly, while the board's
**median** sits one or two mode steps above:

| cell | pow2 claimed (node min) | board min | board median | median/claim |
|---|---|---|---|---|
| 4096 B=1 | 7.29–7.49 | **7.3225** | 8.2854 | 1.13× |
| 4096 B=256 | 9.36–9.43 | **9.3684** | 11.886 | 1.27× |
| 1024 B=512 | 1.79 | **1.7629** | 2.0798 | 1.16× |
| 1024 B=1 | 1.23–1.25 | 1.4096 | 1.4164 | 1.14× |
| **16384 B=64** | **43.4–44.3 stable** | **44.13** | **44.44** | **1.01×** |

The last row is the whole story. 16384 B=64 is the one cell where `d1_pow2` explicitly
eliminated its own bimodality with the huge-page arena, and it is the one cell where the median
equals the claim — and it is the cell pow2 **won**, taking it from `fftw1d_patient`. Everywhere
it did not, the median is a mode above the claim. `d1_twiddle` shows the same pattern (1024 B=1
claimed 1.121, board min 1.2738 = 1.14× — and its record independently states "node drift
between windows was again 10–15%"), as does `d1_planner`, whose 24 claimed cells all match the
board's min to within one mode and match exactly at 60, 128, 16384, 65537, 10007 and 100003.
`d1_rader`'s claims verify directly (65537 B=16 claimed 992, board min 930.9 — measured better
than claimed), and `d1_composite`'s four-cell table matches the board mins at three of four.

**So the honest attribution this round is not "they develop on a faster machine."** It is: the
implementers measured carefully on the right machine, most of them with interleaved same-window
A/B against the r4 binary, and the scoring harness is reporting a statistic that averages over a
machine state the implementers were controlling for and the harness is not. Four entries
(`d1_twiddle`, `d1_prime`, `d1_composite`, `d1_rader`) say so in their own records, in almost the
same words — "only within-window ratios were believed", "the node DRIFTS between invocations".
They are right, and §6 says what to do about it.

**One divergence is not explained by the mode and is a genuine finding.**
`d1_batchlane` ported `d1_composite`'s **`fft60_zmm2x2`** kernel for batched pairs at L=60,
citing "their node-race evidence that zmm2x2 beats the 256-bit form on ICX" — while
`d1_composite`, working concurrently in the same round, **flipped its own default away from
zmm2x2** after the on-node A/B showed the 256-bit `ymm1` per-transform loop beats it by 13%
(§5). The board shows the cost exactly: at L=60 B=512 m=1, `d1_composite` (ymm1) reads 0.0533
median / 0.0504 min, and `d1_batchlane` (zmm2x2) reads **0.0650 / 0.0564** — 22% behind the
donor at a cell it had just imported from the donor. `d1_batchlane` claimed 0.061 and measured
0.0650; the claim is honest, the kernel choice was stale by hours. **Concurrent adoption from a
record that is being rewritten in the same round is a real hazard of this panel format**, and
this is its first clean instance. It costs `d1_batchlane` nothing on the board (it is not the
cell winner either way) and it is a one-line dispatch flip next round.

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

Per §0(c), §4 is the 3D corpus's section. This round bears on three items and moves one of them
hard — **against** the corpus's current text and against r4's own conclusion.

### §4.8 item 6 — AVX-512 on Ice Lake-SP. **MOVED, and it reverses r4**

Item 6 currently concludes that on a modern server part "512-bit is strictly preferable …
(half the instructions, 32 registers instead of 16, 2× L1 and 1.7× L2 load bandwidth, free
embedded broadcast) at **zero** frequency cost." r4 §5(a) supported that with `d1_composite`'s
port-5 diet: 415 → 287 instructions/transform, worth −12% on the node and nothing on Sapphire
Rapids.

**r5 ran the direct on-node interleaved A/B that r4 could only infer, and the 256-bit kernel
wins.** `d1_composite`, at L=60 B=512, on a leased a80n0 core, warm, alternating runs:

| kernel | node min µs/xform | instructions/xform | width |
|---|---|---|---|
| **`ymm1` loop, per transform** | **0.045–0.046** | 505 | ≤256-bit |
| `zmm2x2` (the r4 default) | 0.052–0.053 | 287 | 512-bit |
| `ymm2` (the r3 default) | 0.053–0.055 | 565 | 256-bit pairs |
| MKL, same core, interleaved | 0.043–0.044 | — | — |

**And the frequency explanation was killed by measurement, not argued away**: `scaling_cur_freq`
sampled mid-run reads **3.3 GHz for both binaries** and for the 512-bit chain step. So this is
not licence downclocking; it is ports. On Ice Lake-SP every 512-bit FMA, add and shuffle
dispatches on p0 (+p1 fused) and p5 *only*, so `zmm2x2`'s ~132 FMA-class ops/transform floor at
~66 cycles on two ports with its broadcasts, stores and `vpermilpd` competing for the same pair;
the 256-bit mix fills p0/p1/p5 three-wide.

The transferable rule, and it belongs in the corpus verbatim: **on Ice Lake-SP, a lower
instruction count only wins if the instructions are ≤256-bit. A 512-bit "diet" kernel must get
under the 256-bit version's *port-weighted µop* count, not just its instruction count.**
That directly qualifies item 6's "strictly preferable", and it retires r4's own prediction that
512-bit "should win on the decode-bound node."

**Independently replicated in the same round, in a different codebase.** `d1_pow2` built an AoS
in-register codelet for L=128, measured it, and gated it off: a wash at B=1 and **−25% at
B=512**, diagnosed as "(b) stage 2 pays one `vpermilpd` per cmul where the split engine's SS8
middle stage has ZERO shuffles — I reintroduced exactly the port-5 pressure r2 removed."
`d1_batchlane`'s own `fft128_codelet`, which does the same thing in AoS form, wins at B=512
(0.155 vs 0.169) because it replaced ~192 *cross-lane* shuffles with ~156 mostly *in-lane* µops
— consistent with the same rule. Two entries, opposite outcomes, one mechanism.

### §4.6 — model versus search for the instruction schedule. **MOVED, and it moves back toward the model**

§4.6 ends with "a near-exhaustive search over {schedule variant × unroll depth × batch-loop
placement × copy-or-not × compiler flags} costs minutes. Do it." r4 priced the compiler-flag
axis at ~14% and called it the round's sleeper. r5 ran that search *on the scoring node* and
**four of the five variant lanes lost** (§3.4); the r4 wallaby wins did not transfer, and the one
survivor is a single entry's alignment flags. Combined with §2.0 — two of r4's three headline
figures for the finding sit on the mode band — the flag axis is now priced at roughly zero
except for `d1_pow2`, and §4.6's claim that the *schedule* is "the primary thing to search"
is neither confirmed nor contradicted, because nobody searched the schedule.

What *did* pay this round was structural adoption, not search: the seven largest improvements
in §2.1 are all one entry taking a named kernel from another entry's record.

### §4.4 — split vs interleaved complex. **A qualification worth recording**

§4.4 is marked CLOSED in the corpus's favour (split wins, "the split data layout eliminates
permutations"). `d1_twiddle` adds a limit at the twiddle-table level: **for q-vectorised passes
the interleaved v1 table already *is* the compact format** — consumed with two `vbroadcastsd`
and `u*w = fmaddsub(u, set1(c), mul(permute_pd(u,0x55), set1(s)))`, identical op count to the
broadcast idiom at 2/3 the bytes — so in interleaved-AoS form the table compaction is **free**,
whereas in `d1_pow2`'s split format it costs two port-5 dups (which is why pow2 gates it at
L ≥ 1024). Split still wins on the data layout; it does not automatically win on the tables,
and on a 1.25 MB-L2 part the tables are a first-class competitor for cache.

### Not moved

§4.1 (register liveness), §4.2 (Winograd-17 op count and the symmetric/antisymmetric split), and
the three 3D-geometry items §4.3/§4.5/§4.7. §4.2(a) is reconfirmed rather than moved: at 31
B=512 m=1 the dense conjugate-symmetric `d1_prime` (0.0484) still beats Rader/Agarwal–Cooley
`d1_rader` (0.0634) by 1.31×, while at 1021 and 65537 `d1_rader` owns the board.

---

## 6. The single highest-value thing r6 should attack, per geometry

**One item outranks every entry in the table below, and it is not an implementation item.**

> **Instrument the P-state, then decide whether `RESCORE_PLAN.md` is still the right plan.**
> §2.0 shows the board's noise is not a continuum to be averaged down but three discrete states
> with 14% and 21% steps, selected per process. `RESCORE_PLAN.md` proposes ~3.5 h of compute to
> add six runs per single-call cell across four rounds, on the assumption that more samples
> shrink a Gaussian band by √N. Against a discrete mode-occupancy distribution that assumption
> is wrong: nine samples estimate the *mode occupancy*, not the code. Adding one field to
> `driver.c` — `scaling_cur_freq` (or `APERF/MPERF`) sampled inside the timed region and written
> to each `t_*.json` — costs minutes and settles it. If the modes are frequency, the fix is to
> normalise or to discard off-mode runs, which turns a ±14–21% band into the ~0.5% *within*-process
> sd the driver already reports, and makes 3 runs sufficient. If they are not frequency, the
> top-up proceeds as planned and the campaign has learned something anyway. **State the kill
> criterion before running it.** Nine of this round's 52 verdicts are marked ties whose gaps are
> smaller than one mode step; no amount of implementer effort substitutes for fixing this.

Per geometry, each item being the largest *resolved* gap for that group rather than the largest
nominal one:

| geometry | attack this | why it is the highest-value move |
|---|---|---|
| **L = 4096** | **The 0.84× at B=1 m=1** (d1_race 7.219 / `d1_pow2` min 7.32 vs MKL 6.042) | Still the largest hole on the board, and the only one where MKL's spread is 1.4% so the target is solid. r4 named it and it moved 0.77→0.84×. The Bailey four-step is already built and rejected here, and pow2's arena is now in — the untried lever is `d1_twiddle`'s **register-resident** stage-pair fusion (`vsx44`), which beat pow2's tiled version by 12% at 1024 where pow2's tile *lost* 8%. Port it into pow2 at 4096 B=1 before anything else. |
| **L = 1024** | **The 0.85× at B=1 m=1** — the round's only cell that went backwards (0.92× → 0.85×) | `d1_twiddle` holds it at 1.2823 median / 1.2738 min against a claim of **1.121**, i.e. one clean mode step: this cell may already be at 0.94× and be unresolvable until the item above is done. Do the instrumentation first, re-read the cell, and only then write code. Ranked here because acting on the board number without that could waste a round chasing a P-state. |
| **L = 60** | **`d1_batchlane`'s stale kernel** (§4): flip its batched dispatch from `zmm2x2` to `composite`'s `ymm1` loop | One dispatch line, worth ~0.0650 → ~0.0533 by the donor's own measurement, and it removes the panel's only instance of two entries shipping opposite conclusions about the same kernel. The cell itself (B=512 m=1, 0.97×) is a demonstrated tie and `d1_composite` states it considers the cell closed at ~1.03× of MKL with no structural lever left below PFA's ~204 FMA/xform — I agree; do not spend a round on it. |
| **L = 128** | **The two 0.95× ties, with `d1_batchlane`'s new `fft128_codelet` as the vehicle** | The only geometry where a genuinely new kernel was written this round and it is already at parity (0.155 vs MKL 0.1584 by its own A/B). `d1_pow2`'s record names the one variant worth building — a **split-form** codelet with an `SCMUL` stage 2 — and explicitly says do not retry the AoS form. Two entries have now converged on the same next step; have one of them take it. |
| **L = 16384** | **Adopt `d1_pow2`'s huge-page arena into `d1_twiddle` and `d1_planner`** | pow2 turned the board's worst bimodality (44.6/64.4 µs modes) into 43.5–44.1 stable and took the cell from FFTW. It also recorded **two measured traps** — the skew must be 16 KB+192 B, not 32 KB+192 B, and the SoA chain buffers must stay *out* of the arena (+7% if co-located). That is a transferable fix with the landmines already mapped, and it is the only mechanism on the board demonstrated to remove a mode. |
| **L = 65537, 1021** | **Bank them; port ET0 outward instead** | `d1_rader` is 1.65–2.38× over the best library in all four 65537 cells and owns 1021 non-batched. Its **exit-scatter write prefetch** was worth −32% at B=16 for a runtime-gated prefetch hint — the cheapest win of the round by a wide margin — and it was found because the random output stores were paying a cold RFO *even at B=1*. Every entry with a scatter-shaped exit (`d1_bluestein`'s CRT exit, `d1_planner`'s Rader path) should test one ET0 hint before writing anything. |
| **L = 10007, 100003** | **`d1_race`'s two median regressions** (§2.2), not the kernels | `d1_bluestein` owns both lengths and improved 24–40%. The exposed problem is that at 100003 B=1 and 10007 B=1 the *best* draw improved 2–12% while the median got 14–29% worse. Race's wisdom-referenced probe demonstrably works (100003 B=8, −29%) and demonstrably does not generalise; and it was tuned on wallaby (§3.3). Re-tune it on the node, on the two cells it broke. |
| **L = 13, 31** | **Nothing structural — this geometry is finished** | `d1_prime` is 1.52–4.10× over the best library in all eight cells, the widest margins on the board, and took all four chained cells with the latency-shaped map. Its own record says the one contested cell (13 B=1 m=1) "is a PLACEMENT contest around one kernel" — which §2.0 says is a machine question, not a code question. |
| **L = 32, 64** | **Resolve, do not optimise** | Five of eight cells are wins and the three that are not are ties (64 B=1 0.98×, 64 B=512 1.10×, both with 19–43% spreads). These are the cells the instrumentation item exists for. |

---

## 7. Curation — what this round keeps

Against `docs/CURATION.md`'s four grounds, in its order.

**Ground 1 — the fastest correct entry per geometry.** Cell wins are `d1_race` 24,
`d1_planner` 6, `d1_twiddle` 4, `d1_rader` 4, `d1_pow2` 4, `d1_bluestein` 4, `d1_prime` 2,
`d1_batchlane` 2, `d1_composite` 2. Discounting `d1_race`'s wins to the sibling whose kernel it
ships (§3.4), the algorithm owners of the board are **`d1_prime`** (13, 31), **`d1_composite`**
(60), **`d1_batchlane`** (32/64 B=1 chained), **`d1_twiddle`** (32 B=512 chain, 1024 both m=1
cells), **`d1_pow2`** (4096 and 16384 chained, 16384 B=64), **`d1_rader`** (1021, 65537),
**`d1_bluestein`** (10007, 100003), **`d1_planner`** (13, 32, 60, 64 and 1021 batched m=1).

**Ground 2 — a structurally different runner-up when it is close.** Two qualify and both are
already carried on ground 1: at L=60 B=512 m=1, `d1_planner` (0.0529) and `d1_composite`
(0.0533) are within 0.8% with identical mins — the same PFA kernel in two vehicles, so this is a
near-duplicate rather than a structural alternative, and `CURATION.md` excludes it as such. At
1021, `d1_bluestein`'s chirp-Z (8.39 chained) is 18% behind `d1_rader`'s conv — outside the ~20%
window only just, and it is kept on ground 1 for 10007/100003 anyway.

**Ground 3 — instructive failures whose record documents the number that killed them.** This
round is unusually rich, and three are worth the shelf space on this ground alone:

* `d1_composite` **falsified r4's own conclusion** with an interleaved on-node A/B, killed the
  frequency confound by direct `scaling_cur_freq` measurement rather than argument, and killed a
  second idea (shallower Newton map) on arithmetic before writing it — 1 NR leaves ~5.6e-9
  relative error per step against a chain gate already at 6.6e-13 of a 1e-10 budget.
* `d1_pow2` **built, measured and gated off** an AoS L=128 codelet (−25% at B=512) with the
  mechanism diagnosed to the instruction, and recorded **two negative arena results with the
  numbers that killed them** (32 KB skew: +2%; SoA buffers in the arena: +7%, reproduced with
  THP off, "do not clean this up without re-measuring"). Its record is again the model for the
  round.
* `d1_rader` recorded **three** measured negatives at the site: entry-gather prefetch at B=1
  (+1.7%), middle-stage stream prefetch (+6%), and two ideas skipped on op-count grounds with
  the arithmetic shown (the 13 B=512 8×8 transpose saves ~8% of port-5 where ~26% is needed).

**Ground 4 — anything that beat a library.** 45 of 52 cells, so this ground selects nearly
everything and does not discriminate.

### Promoted — all 9

| entry | ground | note |
|---|---|---|
| `d1_prime` | 1 | 1.52–4.10× over the best library at all eight of its cells — the widest margins on the board — and it took all four chained cells with the latency-shaped divider-free map (Goldschmidt sqrt + early-seeded `rcp14`). Two measured negatives recorded (fold-ahead pipelining at 31 B=512, −2–4%; in-file `aligned(64)` attributes, reverted). |
| `d1_composite` | 1, 3 | Fastest at 60 B=1 in both cells, and carries §5 — the result that reverses r4's port-5 conclusion and qualifies `LITERATURE.md` §4.8 item 6, with the frequency confound eliminated by measurement and the kill criterion stated in advance. The most transferable thing in the round for a 41-line diff. |
| `d1_pow2` | 1, 3 | Owns 4096/16384 chained and took 16384 B=64 from FFTW. The **huge-page arena is the only mechanism on this board demonstrated to remove a timing mode** (44.6/64.4 → 43.5–44.1 stable), and it ships with both of its traps measured. Its claims match the board's min column to 1–3% at every cell. |
| `d1_rader` | 1, 3 | Owns 65537 (1.65–2.38× over `fftw1d_patient`) and 1021 non-batched. **ET0 exit-scatter prefetch, −32% at B=16 for one runtime-gated hint**, is the round's cheapest win and generalises to every scatter-shaped exit on the board. Three measured negatives recorded. |
| `d1_bluestein` | 1 | Owns 10007 and 100003, where nothing else applies, improving 24–43% at the batched single-call cells. Agarwal–Cooley coprime 2D convolution with fused chirp/CRT entry and exit; also the source of the split-complex Stockham core and the arena that three other entries build on. Radix-15 built and killed by measurement. |
| `d1_twiddle` | 1, 3, 4 | Took **both L=1024 m=1 cells** and the 32 B=512 chain, and its **register-resident `vsx44` stage-pair fusion beat `d1_pow2`'s own tiled version's gate** (−12% at 1024 where the tile lost 8%) — a rare case of the adopter improving on the donor, and §6's lever for the 4096 hole. Also carries the §4.4 qualification: the interleaved v1 twiddle table already *is* the compact format. |
| `d1_batchlane` | 1, 3 | Owns 32/64 B=1 chained. Wrote **`fft128_codelet`, the only genuinely new kernel outside the library layers this round** (−8% at 128 B=512, correct first try), and recorded two honest washes at B=1. Promoted **with the stale-kernel finding on the record** (§4): it imported `d1_composite`'s `zmm2x2` while the donor was abandoning it, and reads 22% behind the donor at that cell. That is precisely what ground 3 is for, and the fix is one dispatch line. r4's three "established regressions" against this entry are withdrawn (§2.2). |
| `d1_planner` | 1, 3 | **Promoted, reversing r4's exclusion.** r4 excluded it solely for a missing record; the r5 session wrote both a labelled r4 reconstruction and a full r5 section with a 24-cell measured table, so `CURATION.md`'s exclusion no longer applies. On the numbers it earns it independently: 6 cell wins, −62% to −67% at L=60 and −39% to −52% at L=128, and the paired-p zmm `st8@s4`/`st3@s4`/`st5@s12` fix — one stage every plan ≥1024 was running at half vector width — improved it 13–39% at *nine* lengths. Its record names every borrowing and lists two flat cells as flat rather than dressing them up. |
| `d1_race` | 3 | Promoted on ground 3, not ground 1 — its 24 cell wins are its siblings' (§3.4). The instructive material is the **negative**: it retired four of five variant lanes because r4's wallaby placement wins did not transfer to the node, which prices and mostly closes the axis r4 called its most reusable finding; and it demonstrates the limit of median-targeted tuning by fixing one cell (100003 B=8, −29%) and breaking two neighbours the same way. Its plan cost fell 46.19 s → 2.82 s. Promoted **with §3.3 on the record**: it is the only entry that did not verify or tune on the scoring node, and it is the entry for which that matters most. |

Nothing is excluded. No entry failed correctness, failed to build, crashed, is missing, or lacks
a strategy record; no promoted entry is a near-duplicate of another (`d1_planner` and
`d1_race` are both routing layers but reach the board by different mechanisms — factorisation
planning versus plan-time racing — and §6 depends on both being legible to the next panel).

---

*Monitor notes for whoever writes `NOTES.md`: (1) `exemplars/d1_r3/NOTES.md` and
`exemplars/d1_r4/NOTES.md` are still unfilled skeletons and both rounds' `VERDICT.md` are on
disk to write them from. (2) `RESCORE_PLAN.md` should be read against §2.0 before step 3 is
executed. (3) The brief's L=6/8/17/36 geometries, its Cascade Lake 5218, and its unsourced "MKL
spans 2.9×" have now been corrected by three consecutive monitors; fixing the brief would be
cheaper than a fourth.*

PROMOTE: d1_prime d1_composite d1_pow2 d1_rader d1_bluestein d1_twiddle d1_batchlane d1_planner d1_race
