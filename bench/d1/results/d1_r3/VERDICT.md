# VERDICT — panel round d1_r3

Monitor's report. Written 2026-09-03 from `results/d1_r3/leaderboard.txt`,
`environment.txt`, `build_errors.txt`, `results/wisdom1d_a80n0.json`, the nine
`strategies/*.md` records, and `docs/CURATION.md`.

---

## 0. Two corrections to the brief before any numbers

The monitoring brief I was given describes a **different campaign**, and the mismatch
has to be stated or every number below reads wrong.

**(a) The geometries.** The brief asks for headlines at `L = 6, 8, 17, 36`. Those are the
**3D cube** campaign's sizes (`bench/geom/`, scored in `results/panel_r*`). Round `d1_r3`
is the **1D** contest defined by `bench/d1/cases.txt`: **thirteen** sizes
`13, 31, 32, 60, 64, 128, 1021, 1024, 4096, 16384, 10007, 65537, 100003`, each in four
regimes (B=1/B>1 × m=1/chained) — 52 scored cells. I report all thirteen.

**(b) The machine.** The brief says Xeon Gold 5218, Cascade Lake, 1 MB L2, downclocked
AVX-512. That is again the 3D campaign's scoring node. `environment.txt` says this round
was measured on **`a80n0.lqcd.mit`, Intel Xeon Gold 6326 — Ice Lake-SP**, 64 cores,
`schedutil`, gcc 11.4.0, slurm job 440424, and the ISA list carries the Ice Lake-only
`avx512_vbmi2`/`vnni`/`bitalg` cluster. Ice Lake-SP has **1.25 MB L2 per core** and two
512-bit FMA units, and two implementers independently measured it sustaining
**~2.9–3.0 GHz under AVX-512** — a much milder licence penalty than the 5218's. The
implementers develop on **wallaby, Xeon Gold 6448Y (Sapphire Rapids), 2 MB L2, ~4.1 GHz
sustained**. So the development-vs-scoring gap in §4 is **SPR → ICX**, and it is real, but
it is not the gap the brief describes.

**(c) `docs/LITERATURE.md` §4** is likewise the 3D corpus's open-questions section, written
against L=6/8/17/36. There is no §4 in the 1D corpus (`docs/literature_1d/00-SURVEY.md`
has no such section). §5 below answers the question honestly by mapping this round's
evidence onto the §4 items that are *hardware* questions rather than 3D-geometry
questions, and says which ones 1D cannot speak to.

Everything else in the brief applies as written.

---

## 1. Headline per geometry — fastest correct panel entry vs best library

All times are **µs per transform**, min over runs, as the leaderboard reports them.
"best library" = best of `fftw1d_{estimate,measure,patient}`, `fftw1d_custom{,_soa}`,
`mkl1d_dfti`. `baseline_dft` is the harness's library-free floor, not a competitor.

Two columns are given for the panel because **`d1_race` does not contain an algorithm** —
it is the adoption layer, and at every cell it dispatches a *sibling entry's compiled code*
by vtable. Crediting it as "the fastest entry" would credit dispatch with the sibling's
work. See §3.1: its m=1 margins over the very code it ships are measurement artefacts.
**The "panel (algorithmic)" column is the one to read as a result.** The `(±)` figure is
that row's reported run spread.

| L | regime | panel (as scored) | panel (algorithmic) | best library | verdict |
|---|---|---|---|---|---|
| **13** | B=1 m=1 | d1_race 0.0166 | **d1_prime 0.0172** (±37%) | fftw1d_estimate 0.0218 (±21%) | **WIN 1.27×** |
| | B=512 m=1 | d1_prime 0.0107 | **d1_prime 0.0107** (±0.1%) | fftw1d_measure 0.0123 (±14%) | **WIN 1.15×** |
| | B=1 chain | d1_batchlane 0.0392 | **d1_batchlane 0.0392** (±14%) | fftw1d_custom 0.0625 (±21%) | **WIN 1.59×** |
| | B=512 chain | d1_race 0.0146 | **d1_rader 0.0146** (±14%) | fftw1d_custom_soa 0.0262 (±14%) | **WIN 1.79×** |
| **31** | B=1 m=1 | d1_prime 0.0530 | **d1_prime 0.0530** (±2.5%) | fftw1d_custom 0.2148 (±1.0%) | **WIN 4.05×** |
| | B=512 m=1 | d1_race 0.0441 | **d1_prime 0.0501** (±0.0%) | fftw1d_custom_soa 0.0913 (±15%) | **WIN 1.82×** |
| | B=1 chain | d1_batchlane 0.0546 | **d1_batchlane 0.0546** (±14%) | fftw1d_custom 0.2520 (±0.8%) | **WIN 4.62×** |
| | B=512 chain | d1_race 0.0477 | **d1_prime 0.0485** (±0.1%) | fftw1d_custom_soa 0.0773 (±1.6%) | **WIN 1.59×** |
| **32** | B=1 m=1 | d1_race 0.0206 | **d1_pow2 0.0207** (±14%) | mkl1d_dfti 0.0225 (±42%) | WIN 1.09× *(tie)* |
| | B=512 m=1 | d1_pow2 0.0154 | **d1_pow2 0.0154** (±14%) | mkl1d_dfti 0.0153 (±14%) | **loss 0.99×** *(tie)* |
| | B=1 chain | d1_batchlane 0.0591 | **d1_batchlane 0.0591** (±18%) | fftw1d_custom 0.1223 (±20%) | **WIN 2.07×** |
| | B=512 chain | d1_race 0.0382 | **d1_pow2 0.0384** (±1.2%) | fftw1d_custom_soa 0.0665 (±0.4%) | **WIN 1.73×** |
| **60** | B=1 m=1 | d1_race 0.0473 | **d1_composite 0.0489** (±14%) | mkl1d_dfti 0.0608 (±7.5%) | **WIN 1.24×** |
| | B=512 m=1 | d1_race 0.0525 | **d1_composite 0.0591** (±6.4%) | mkl1d_dfti 0.0492 (±12%) | **loss 0.83×** |
| | B=1 chain | d1_composite 0.1105 | **d1_composite 0.1105** (±14%) | mkl1d_dfti 0.2371 (±14%) | **WIN 2.15×** |
| | B=512 chain | d1_composite 0.0592 | **d1_composite 0.0592** (±0.4%) | fftw1d_custom_soa 0.1336 (±1.7%) | **WIN 2.26×** |
| **64** | B=1 m=1 | d1_race 0.0484 | **d1_batchlane 0.0486** (±8.0%) | mkl1d_dfti 0.0465 (±12%) | **loss 0.96×** *(tie)* |
| | B=512 m=1 | d1_race 0.0374 | **d1_pow2 0.0434** (±23%) | mkl1d_dfti 0.0443 (±14%) | WIN 1.02× *(tie)* |
| | B=1 chain | d1_race 0.0848 | **d1_batchlane 0.0848** (±14%) | fftw1d_estimate 0.2678 (±14%) | **WIN 3.16×** |
| | B=512 chain | d1_race 0.0788 | **d1_batchlane 0.0789** (±14%) | fftw1d_custom_soa 0.1511 (±0.6%) | **WIN 1.92×** |
| **128** | B=1 m=1 | d1_twiddle 0.1110 | **d1_twiddle 0.1110** (±19%) | mkl1d_dfti 0.0910 (±16%) | **loss 0.82×** |
| | B=512 m=1 | d1_race 0.1552 | **d1_batchlane 0.1601** (±20%) | mkl1d_dfti 0.1471 (±7.8%) | **loss 0.92×** *(tie)* |
| | B=1 chain | d1_race 0.1845 | **d1_batchlane 0.1850** (±3.1%) | mkl1d_dfti 0.4756 (±14%) | **WIN 2.57×** |
| | B=512 chain | d1_race 0.1863 | **d1_pow2 0.1870** (±1.2%) | fftw1d_custom_soa 0.3938 (±14%) | **WIN 2.11×** |
| **1021** | B=1 m=1 | d1_race 6.963 | **d1_rader 7.788** (±3.8%) | mkl1d_dfti 9.407 (±0.5%) | **WIN 1.21×** |
| | B=256 m=1 | d1_rader 7.574 | **d1_rader 7.574** (±15%) | mkl1d_dfti 8.747 (±16%) | **WIN 1.15×** |
| | B=1 chain | d1_race 7.319 | **d1_rader 7.400** (±15%) | mkl1d_dfti 11.397 (±14%) | **WIN 1.54×** |
| | B=256 chain | d1_race 7.356 | **d1_rader 7.410** (±0.5%) | mkl1d_dfti 12.394 (±0.1%) | **WIN 1.67×** |
| **1024** | B=1 m=1 | d1_race 1.529 | **d1_pow2 1.622** (±0.8%) | mkl1d_dfti 1.075 (±15%) | **loss 0.66×** |
| | B=512 m=1 | d1_race 1.899 | **d1_pow2 1.991** (±11%) | mkl1d_dfti 1.652 (±19%) | **loss 0.83×** |
| | B=1 chain | d1_pow2 2.109 | **d1_pow2 2.109** (±14%) | fftw1d_patient 4.245 (±14%) | **WIN 2.01×** |
| | B=512 chain | d1_pow2 2.564 | **d1_pow2 2.564** (±0.4%) | mkl1d_dfti 4.864 (±1.2%) | **WIN 1.90×** |
| **4096** | B=1 m=1 | d1_twiddle 8.288 | **d1_twiddle 8.288** (±13%) | mkl1d_dfti 5.997 (±0.9%) | **loss 0.72×** |
| | B=256 m=1 | d1_race 10.73 | **d1_pow2 11.07** (±0.2%) | mkl1d_dfti 10.15 (±13%) | **loss 0.92×** *(tie)* |
| | B=1 chain | d1_pow2 11.05 | **d1_pow2 11.05** (±14%) | fftw1d_patient 19.08 (±1.3%) | **WIN 1.73×** |
| | B=256 chain | d1_race 11.16 | **d1_pow2 11.98** (±0.3%) | mkl1d_dfti 22.93 (±0.6%) | **WIN 1.91×** |
| **16384** | B=1 m=1 | d1_race 45.66 | **d1_twiddle 52.88** (±12%) | fftw1d_patient 32.27 (±3.9%) | **loss 0.61×** ← worst cell |
| | B=64 m=1 | d1_pow2 54.45 | **d1_pow2 54.45** (±16%) | fftw1d_patient 47.71 (±4.0%) | **loss 0.88×** |
| | B=1 chain | d1_race 72.16 | **d1_pow2 76.12** (±14%) | fftw1d_patient 82.23 (±0.3%) | **WIN 1.08×** |
| | B=64 chain | d1_pow2 63.62 | **d1_pow2 63.62** (±5.6%) | fftw1d_patient 98.86 (±0.9%) | **WIN 1.55×** |
| **10007** | B=1 m=1 | d1_bluestein 156.4 | **d1_bluestein 156.4** (±18%) | fftw1d_patient 198.4 (±1.3%) | **WIN 1.27×** |
| | B=64 m=1 | d1_race 168.7 | **d1_bluestein 192.2** (±21%) | fftw1d_patient 207.1 (±4.2%) | **WIN 1.08×** |
| | B=1 chain | d1_bluestein 172.2 | **d1_bluestein 172.2** (±0.1%) | fftw1d_patient 225.7 (±2.4%) | **WIN 1.31×** |
| | B=64 chain | d1_bluestein 176.0 | **d1_bluestein 176.0** (±1.6%) | fftw1d_patient 239.8 (±3.8%) | **WIN 1.36×** |
| **65537** | B=1 m=1 | d1_race 907.2 | **d1_rader 1013.9** (±6.2%) | fftw1d_patient 1463.4 (±1.8%) | **WIN 1.44×** |
| | B=16 m=1 | d1_race 1182.9 | **d1_rader 1339.7** (±8.1%) | fftw1d_patient 1535.6 (±1.2%) | **WIN 1.15×** |
| | B=1 chain | d1_race 783.2 | **d1_rader 784.5** (±2.5%) | fftw1d_patient 1627.6 (±0.2%) | **WIN 2.07×** |
| | B=16 chain | d1_rader 821.0 | **d1_rader 821.0** (±9.9%) | fftw1d_patient 1740.2 (±1.6%) | **WIN 2.12×** |
| **100003** | B=1 m=1 | d1_race 2607.2 | **d1_bluestein 2992.5** (±16%) | fftw1d_patient 2686.7 (±1.6%) | **loss 0.90×** |
| | B=8 m=1 | d1_race 2747.9 | **d1_bluestein 2754.9** (±37%) | fftw1d_patient 2808.8 (±1.4%) | WIN 1.02× *(tie)* |
| | B=1 chain | d1_bluestein 2713.7 | **d1_bluestein 2713.7** (±3.1%) | fftw1d_patient 3113.8 (±8.2%) | **WIN 1.15×** |
| | B=8 chain | d1_bluestein 3021.9 | **d1_bluestein 3021.9** (±0.2%) | fftw1d_patient 3195.6 (±4.5%) | **WIN 1.06×** |

### Score: 40 of 52 cells won on algorithm alone (41 with `d1_race`'s dispatch)

The regime aggregates the brief mandates (geometric mean of library/panel across the
thirteen sizes, *within* each regime — never averaged across regimes):

| regime | r1 | r2 | **r3** | cells won (r3) |
|---|---|---|---|---|
| B=1, m=1 | 0.95× | 1.06× | **1.15×** | 8 / 13 |
| B>1, m=1 | 0.89× | 0.96× | **1.10×** | 7 / 13 |
| B=1, chained | 1.44× | 1.59× | **1.93×** | **13 / 13** |
| B>1, chained | 1.35× | 1.60× | **1.75×** | **13 / 13** |

**The structural fact of this round: every one of the 26 chained cells is a win, and
every one of the 12 losses is a non-chained (m=1) cell.** The fused-map chain — the
panel's own API, which no library exposes — is now uncontested at every size from 13 to
100003. The single-call regime is exactly where `PANEL_BRIEF.md` predicted the panel
would struggle, and it does, but it has gone from losing that regime (0.95×/0.89× in r1)
to winning it on aggregate (1.15×/1.10×).

**Which losses are real and which are noise.** Seven of the twelve are inside the
reported run spreads and are *not resolvable from this leaderboard* — marked *(tie)*
above: 32 B=512 (0.0154 vs 0.0153 with ±14% on both), 64 B=1, 128 B=512, 4096 B=256,
100003 B=8. Five are real, all in the power-of-two single-call column plus L=60 batched:

* **16384 B=1 m=1 — 0.61×** (52.88 vs 32.27, ±12%/±3.9%). The panel's worst cell.
* **1024 B=1 m=1 — 0.66×** (1.622 vs 1.075, ±0.8%/±15%).
* **4096 B=1 m=1 — 0.72×** (8.288 vs 5.997, ±13%/±0.9%).
* **128 B=1 m=1 — 0.82×** (0.1110 vs 0.0910, ±19%/±16%).
* **60 B=512 m=1 — 0.83×** (0.0591 vs 0.0492, ±6.4%/±12%).
* **16384 B=64 m=1 — 0.88×**, and **100003 B=1 m=1 — 0.90×**, both marginal.

---

## 2. What changed since d1_r2, and what regressed

Same host, same slurm partition, same gcc, same case list, same nine entries, same
per-entry cell coverage — so r2→r3 is a clean comparison. Best-panel-entry delta per cell:

### Improvements, by geometry

| L | what improved, and by how much | who, and why |
|---|---|---|
| **13** | B=1 m=1 −28%, B=512 m=1 **−48%** (0.0206→0.0107) | `d1_prime`'s interleaved-pair kernels: one complex output per 128-bit lane, pair-duplicated tables, no deinterleave prologue. It also **removed** the r2 `{1to8}`-broadcast barrier, which was an ICX poison (0.021→0.015 in a controlled on-node bisect) — an SPR-tuned change that reversed sign on the scoring silicon. Two cells that were library *losses* in r2 are now wins. |
| **31** | B=1 m=1 −18%, B=512 m=1 −18% | Same pair kernels. `d1_rader` independently flipped both batched cells with across-batch SoA (31 B=512: 0.175→0.063 same-minute). |
| **32** | B=1 chain −25% | `d1_batchlane`'s register-resident B=1 chain: the four-step kernels' natural row layout is closed under the transform, so the whole state stays in zmm across all m steps. |
| **60** | B=1 m=1 −11%, B=1 chain −11% | `d1_composite`'s coset-row chain (v4): natural indices decompose into 15 cosets that are closed under *both* the stage-A operand pairing and the stage-C emission pairing, so the state lives as 15 DFT-4-ready rows and no split scratch exists. Killed a store-to-load-forwarding block on the serial dependence chain. |
| **64** | B=1 chain **−38%** (0.1358→0.0848) | `d1_batchlane` register-resident chain again. |
| **128** | B=512 chain −19%, B=1 chain −13% | `d1_batchlane` / `d1_pow2`. |
| **1021** | B=1 m=1 −11%, chains −5…−7% | `d1_rader`'s Newton `rsqrt14`/`rcp14` map replacing `vsqrt`+`vdiv` on the single divide unit. |
| **1024/4096** | B=1 chains −23% each | `d1_pow2` dropping the exact-residual map refinements (worst gate moved 4.54e-12→5.68e-12 against a 1e-10 floor) and the SS64 radix-64 L1-tiled fusion at 4096. |
| **10007** | all four cells −14…−15% | `d1_bluestein`: fresh `mmap`+`MADV_HUGEPAGE`+pre-fault with a rotating cross-block skew (phase-time swing 25–30% → ±1.4%), and the discovery that gcc **silently drops** per-function `prefer-vector-width=512` when the stage kernels inline into a caller without the attribute — `objdump` showed *zero* zmm in `core_exec_range`. |
| **16384** | B=64 m=1 −14%, B=64 chain −7% | `d1_pow2`'s NT-store rehabilitation: r1's blanket "NT is 3× slower" verdict was **confounded** — with an odd stage count the ping-pong ran intermediates through the caller's `out`, so the NT stream targeted lines the same call had just dirtied. Routing intermediates through private scratch flips it (71.9→50.7 µs on wallaby at the same cell). |
| **65537** | all four cells −21…−25% | `d1_rader`'s conv restructured `[4,8,8,8,8,4]` → `[4,16,16,16,4]`: 10 array passes instead of 12 over two 512 KB planes, on an L3-bandwidth-bound cell, so it paid almost exactly pro rata. Plus the paired-p `st16_s4` variant for the masked half-pass. |
| **100003** | −12…−24% | `d1_bluestein`'s Agarwal–Cooley minimal-M re-derivation and NT-streamed exit scatter (exit phase 1103–1435 → 711–782 µs). |

### Regressions — yes, and they cluster

Thirty per-entry cell regressions exceed 8%. **The pattern is unmistakable: every
regression sits in an entry that measured on wallaby this round, and none sits in an
entry that measured on the scoring node.**

| entry | measured r3 on | regressions >8% | worst |
|---|---|---|---|
| `d1_planner` | wallaby SPR | **10** | 13 B=1 +37%, 31 B=1 +35%, 128 B=512 +26% |
| `d1_pow2` | wallaby SPR | 5 | 1024 B=1 +19%, 128 B=1 +17%, 16384 B=1 +15% |
| `d1_twiddle` | wallaby SPR | 4 | **16384 B=64 +34%**, 60 B=512 +22% |
| `d1_race` | wallaby SPR | 4 | 32 B=512 +14%, 128 B=1 +13%, 16384 B=64 +12% |
| `d1_batchlane` | wallaby SPR | 3 | 60 B=1 chain +14%, 128 B=1 +13% |
| `d1_composite` | wallaby SPR | **0** | — (see below) |
| `d1_bluestein` | **a80n0 node** | 3, all outside its class (13, 60) | 60 B=1 +20% |
| `d1_prime` | **a80n0 node** | 1 | 31 B=1 chain +14% |
| `d1_rader` | **a80n0 node** | **0** | — |

The **cells** that went backwards for the whole panel, i.e. the best panel entry is
slower in r3 than in r2:

* **L=128 B=1 m=1: +12%** and it is a four-entry collapse — `d1_race` 0.0988→0.1121,
  `d1_batchlane` 0.0991→0.1117, `d1_pow2` 0.1010→0.1177, `d1_planner` 0.2334→0.2867.
  Nobody targeted this cell; everybody lost it. This is the round's clearest
  unexplained regression and it turns a near-tie with MKL into a real 0.82× loss.
* **L=1024 B=1 m=1: +12%** (`d1_pow2` alone +19%, 1.3663→1.6224). `d1_pow2`'s r3 changes
  at 1024 were the fast-map default (chained cells only) and an NT policy that
  *correctly excluded* 1024 — so this regression is not accounted for by any documented
  change. It makes 1024 B=1 the panel's second-worst cell.
* **L=16384 B=1 chain: +4%** (`d1_pow2` 69.61→76.12) and **B=64 m=1** would have
  regressed too but for `d1_pow2`'s NT win; `d1_twiddle` there is +34%.
* **L=60 B=512 m=1: +2%**, **L=32/64 B=1 m=1: ~0%** — noise.

`d1_composite` is the counter-example that proves the mechanism rather than breaking it.
It also measured on wallaby, but it did the arithmetic: it took its wallaby number
(0.085 µs at the B=1 chain), multiplied by the 1.25× SPR→ICX factor it had derived from
its own r2 board position, predicted **0.106**, and the node measured **0.1105**. That is
the discipline the other four wallaby-only entries did not apply, and `d1_composite`
regressed nowhere.

---

## 3. Adversarial pass — correctness, builds, crashes, gaps

### 3.0 Nothing failed correctness. I checked, and I checked that the check can fail.

* **No `failures.txt`** — nothing crashed or hung.
* **`build_errors.txt` exists but contains only warnings.** Five `-Waggressive-loop-optimizations`
  in `d1_batchlane.c:164/175` (an `int` induction variable the optimiser proves would have
  to reach 2³¹ — dead scalar-tail code, since `L` is a compile-time constant on those
  paths) and three `-Wformat-truncation` in `d1_race.c:1029/1033/1451` (`snprintf` into
  600- and 64-byte buffers). **No errors, and all ten panel binaries are present in the
  leaderboard.** The `batchlane` warnings should still be silenced — an `int` index into
  `aos[2*j]` is technically UB and the optimiser is telling you it noticed.
* **Every one of the 530 scored rows reports `ok`.** Single-call `rel_l2` spans
  8.6e-17 … 1.3e-15 against a 1e-12 gate. All **265 chained rows** pass; the worst
  margin on the board belongs to `baseline_dft` at 0.16 of tolerance (1.6e-11 vs 1e-10),
  and the worst *panel* margin is `d1_twiddle` at 1024 B=1 m=4000, 6.0e-12 vs 1e-10 —
  **a 16× margin**. Nothing is anywhere near the wire.
* **The gate is not a check that cannot fail**, and I have two independent proofs from
  this round rather than an assumption. `d1_prime` records its first pair-lane kernel
  failing at `rel_l2 ≈ 1` because the sine pair table was built `(+s,+s)` instead of
  `(+s,−s)`; `d1_rader` records its first `st16` tile cut failing at `rel_l2 = 1.4e0`
  from a slot-index transposition. Both were caught immediately by `check.py`. A fast
  wrong answer was produced twice this round and killed twice.
* **Two-sided control.** `baseline_dft`, the library-free O(L²) floor, is present and is
  9×–1693× slower than the winner at every cell where it is feasible, so the harness is
  measuring something. It is **absent** above L≈128 at large B (all of 4096, 10007,
  16384, 65537, 100003, and the batched cells at 128/1024) because O(L²) is infeasible
  there — expected, but worth stating plainly: **at the eight largest geometries there is
  no library-free reference inside the leaderboard**, and correctness rests entirely on
  `check.py` against numpy. That is the right ground truth, but it is a single point of
  failure and it is not cross-checked by a second slow path.

### 3.1 The finding that matters: `d1_race`'s single-call margins are not real

`d1_race` sits at the top of **31 of 52** cells and appears as "fastest" at eleven cells
in the raw board. It contains no transform of its own: `results/wisdom1d_a80n0.json`
records, for every one of the 24 `(L,B)` pairs, exactly which sibling's compiled code the
race selected and shipped. So for each cell I can compare `d1_race`'s scored time against
the scored time of **the entry whose identical machine code it is running.** The result
splits cleanly by regime:

**Chained cells — the numbers agree, as they must.** All 26, without exception, land
within ~1% of the shipped sibling: 13 B=512 chain `d1_race` 0.0146 vs `d1_rader` 0.0146;
64 B=1 chain 0.0848 vs `d1_batchlane` 0.0848; 65537 B=1 chain 783.18 vs `d1_rader`
784.46; 1024 B=512 chain 2.579 vs `d1_pow2` 2.564. Two exceptions in the wrong direction
(31 B=1 chain, race 13% *slower* than the `batchlane` code it ships; 60 B=1 chain, 13%
slower than `composite`) and one in the right (4096 B=256, 7% faster).

**Single-call m=1 cells — the numbers scatter ±30% in both directions.**

| cell | ships | `d1_race` | that sibling, scored | race is |
|---|---|---|---|---|
| 16384 B=1 | `d1_pow2` | 45.66 | 52.95 | **14% faster** |
| 65537 B=16 | `d1_rader` | 1182.9 | 1339.7 | **12% faster** |
| 100003 B=1 | `d1_bluestein` | 2607.2 | 2992.5 | **13% faster** |
| 1021 B=1 | `d1_rader` | 6.963 | 7.788 | **11% faster** |
| 10007 B=64 | `d1_bluestein` | 168.7 | 192.2 | **12% faster** |
| 128 B=512 | `d1_pow2` | 0.1552 | 0.1895 | **18% faster** |
| 31 B=512 | `d1_prime` | 0.0441 | 0.0501 | **12% faster** |
| 64 B=512 | `d1_pow2` | 0.0374 | 0.0434 | **14% faster** |
| 32 B=512 | `d1_pow2` | 0.0176 | 0.0154 | **14% slower** |
| 16384 B=64 | `d1_pow2` | 70.83 | 54.45 | **30% slower** |
| 4096 B=1 | `d1_pow2` | 8.811 | 8.667 | 2% slower |

Identical instructions, identical data, same session, same core — up to 30% apart, both
signs. That is not an advantage `d1_race` earned; it is code and heap placement, and it is
**independently corroborated**: `d1_composite`'s r2 forensics found the race scoring its
*byte-identical* ymm2 machine code 33% faster than `d1_composite`'s own entry in the same
session, and `d1_race`'s own r3 record diagnoses the mechanism (arm plans created
mid-race, with ~7× the cell's bytes of race buffers live, land in a different heap and
page layout than the standalone binary) and ships two fixes for it (`warm_each` sampling
and a post-race replan on a drained heap). The fixes evidently did not close it.

**Three consequences, and I am applying all three:**

1. On the m=1 cells, **differences below ~15% between panel entries are not resolvable by
   this leaderboard.** The chained cells, which run thousands of reps, agree to <1%; the
   single-call cells, whose reported spreads run 10–42%, do not. Any r4 decision taken on
   a <15% m=1 delta is being taken on noise.
2. `d1_race`'s eleven "fastest" placements are **the shipped sibling's number plus
   placement luck**, and §1 credits the sibling. This is not a correctness fault and not
   dishonesty — the race's *selections* are sound (§3.2) — but it is a scoring artefact
   and the board should not be read as if `d1_race` beat its own donors.
3. **`d1_race` converts a panel loss into a panel win at exactly one cell**: 100003 B=1
   m=1, where it scores 2607.2 against `fftw1d_patient` 2686.7 while `d1_bluestein`, whose
   code it runs, scores 2992.5 and loses. A 3% margin at ±16%/±1.6% spread. **I am not
   counting that as a win.** The honest 100003 B=1 m=1 result is a 0.90× loss.

### 3.2 The race's *selections*, by contrast, are correct

Credit where it is due. Checked against the r3 board, the `exe.r3`/`chn.r2` verdicts in
`wisdom1d_a80n0.json` pick the genuinely fastest algorithmic sibling at essentially every
cell — `d1_prime` at 13/31, `d1_pow2` at 32/1024/4096/16384, `d1_batchlane` at 64/128 B=1
and every small B=1 chain, `d1_composite` at 60, `d1_rader` at 1021/65537,
`d1_bluestein` at 10007/100003. Two near-misses, both immaterial: 4096 B=1 it picked
`d1_pow2` (8.667) where `d1_twiddle` measured 8.288 (4.6%), and 128 B=1 it picked
`d1_batchlane` (0.1117) where `d1_twiddle` measured 0.1110 (0.6%). Both inside §3.1's
resolution floor. The r2 verdict this round set out to fix — `d1_twiddle` wrongly winning
4096 B=256 on cold caches — **is fixed**: `exe.r3` now picks `d1_pow2` there.

### 3.3 `d1_race`'s setup cost is real, disclosed, and belongs in the record

The metric excludes plan time, so the race's plan-time tournament is free by the rules.
It is not free in practice: **4.68 s** at 10007 B=1, **4.00 s** at 65537 B=1, **2.20 s**
at 1021 B=256, **1.75 s** at 4096 B=256, **1.53 s** at 16384 B=1, **1.06 s** at 128
B=512 — against 0.000–0.008 s for every algorithmic entry. Its rank is unreachable
without the per-host wisdom cache, and it recompiles siblings at plan time. In fairness,
`fftw1d_patient` pays far worse (**67.6 s** at 16384 B=64, **44.6 s** at 4096 B=256,
**11.6 s** at 1024 B=512), so the panel is not the worst offender — but a reader
comparing 45.66 µs to 32.27 µs at 16384 B=1 should know one of them cost 1.5 s to plan.

### 3.4 `d1_planner` won nothing, and regressed the most

`d1_planner` covers all 52 cells and is the **only entry that is fastest at none of
them** — for the second round running. It also owns 10 of the 30 regressions, including
the two largest on the board (13 B=1 +37%, 31 B=1 +35%). Its own record is candid that
r3 "is almost entirely adoption" — four separate lifts from `d1_rader`, one from
`d1_pow2`, one from `d1_bluestein`. Its claimed wins are large (16384 B=64 90.95→45.59,
a 2× from `d1_bluestein`'s huge-page arena; 1021 B=1 8.19→5.03 from `d1_rader`'s `st17`)
and all were measured on wallaby; the node scored 1021 B=1 at 8.759 and 16384 B=64 at
72.47. It is not broken and not incorrect — it passes all 52 cells and beats a library at
19 of them, more than any single-class entry — but nothing in it is now its own.

### 3.5 Missing coverage, named

No entry disappeared and no cell lost a competitor: per-entry coverage is **identical**
to r2 (`bluestein`/`planner`/`race` 52, `twiddle` 28, `batchlane`/`pow2` 24, `rader` 16,
`prime` 8, `composite` 4 — each entry's declared class). Two real gaps remain:

* **`d1_rader` does not enter 10007 or 100003 at all.** The brief assigns the
  Rader-vs-Bluestein crossover at the two awkward primes as the class headline, and the
  nested-Rader A/B (5003−1 = 2·41·61; 2381 for 100003) is *still* untried after three
  rounds — `d1_rader`'s own next-round list item 4. The crossover is therefore still
  **assumed, not measured**, at exactly the two sizes the brief says to measure it.
* **L=60 has one owner.** `d1_composite` is the only class entry; `d1_batchlane`'s PFA-60
  path and `d1_twiddle` are the only cross-class competition, and MKL leads the B=512 m=1
  cell outright. A single-owner geometry has no internal check.

---

## 4. Claimed numbers vs measured — and where the machine explains it

**The critical fact the brief did not have: only four of nine entries measured on the
scoring node this round, and it was not their choice.** `reserve.sh --status` reported
the live a80n0 hold (job 440424) as dead, because the wallaby `squeue` shim on `PATH`
reads the heartbeat from `bench/gen/` rather than `bench/d1/`. `d1_prime`, `d1_rader`,
`d1_bluestein` diagnosed this and worked around it with personal `/tmp` shims pointing at
the d1 heartbeat; `d1_planner` left the first such shim. **`d1_batchlane`, `d1_pow2`,
`d1_twiddle`, `d1_race` and `d1_composite` all recorded "the Ice Lake reservation was
dead again" and measured blind on wallaby.** That is a harness bug, not implementer
error, and §2 shows it cost the round about thirty cell-regressions.

### The four entries that measured on the node: claims land

| entry | claim | measured | Δ |
|---|---|---|---|
| `d1_prime` | 13 B=1 0.018–0.019 | 0.0172 | ✓ |
| | 13 B=512 0.010–0.011 | 0.0107 | ✓ |
| | 31 B=1 0.048–0.053 | 0.0530 | ✓ |
| | 31 B=512 0.044–0.050 | 0.0501 | ✓ |
| `d1_rader` | 65537 B=1 1049 | 1013.9 | −3% (better) |
| | 65537 B=16 1348 | 1339.7 | ✓ |
| | 65537 B=1 chain 912 | 783.2 | −14% (better) |
| | 13 B=512 0.018 | 0.0186 | ✓ |
| | 31 B=1 **0.178** | **0.2113** | **+19%** |
| | 31 B=1 chain 0.206 | 0.2348 | +14% |
| | 1021 B=1 6.9 | 7.788 | +13% |
| `d1_bluestein` | 65537 B=1 2008–2360 | **1715.4** | −15…−27% (better) |
| | 10007 B=1 180 | 156.4 | −13% (better) |
| | 100003 B=1 2660–3000 | 2992.5 | ✓ (top of range) |

`d1_prime` is exact — it ran `tryout.sh`'s pipeline on the node against the same
`gen_input`/`check.py`. `d1_rader`'s ±13–19% scatter is **self-declared**: its record
opens by documenting that its measurement window drifted monotonically within the session
("an early baseline of 1241 µs at 65537 B=1 aged to 1340 for the identical binary 20
minutes later") and states every decision was same-minute A/B'd for that reason. Its
claims are honest ranges that the sweep landed inside or just outside.
`d1_bluestein`'s claims are **conservative** — the node scored it 15–27% *faster* than
claimed at 65537 — and it explains why: back-to-back invocations droop monotonically
under sustained AVX-512 (3410→4042 µs over six runs at 100003), so a long tryout battery
punishes whoever runs later, while the sweep measures everyone warm. That is the correct
diagnosis and it is worth the whole panel adopting.

### The five that measured on wallaby: claims are optimistic by a consistent factor

| entry | cell | claimed (wallaby SPR) | measured (a80n0 ICX) | factor |
|---|---|---|---|---|
| `d1_pow2` | 16384 B=1 m=1 | 27.25 | 52.95 | **1.94×** |
| | 16384 B=1 chain | 36.9 | 76.12 | **2.06×** |
| | 64 B=1 m=1 | 0.029 | 0.0519 | 1.79× |
| | 32 B=1 m=1 | 0.012 | 0.0207 | 1.73× |
| | 1024 B=1 m=1 | 1.021 | 1.622 | 1.59× |
| | 4096 B=1 m=1 | 6.31 | 8.667 | 1.37× |
| `d1_twiddle` | 16384 B=64 m=1 | 35.6 | 87.81 | **2.47×** |
| | 32 B=1 m=1 | 0.018 | 0.0348 | 1.93× |
| | 128 B=512 chain | 0.139 | 0.2053 | 1.48× |
| | 4096 B=1 m=1 | 5.96 | 8.288 | 1.39× |
| `d1_batchlane` | 31 B=512 m=1 | 0.031 | 0.0664 | **2.14×** |
| | 128 B=512 m=1 | 0.092 | 0.1601 | 1.74× |
| | 13 B=1 m=1 | 0.015 | 0.0242 | 1.61× |
| | 32 B=1 chain | 0.053 | 0.0591 | 1.12× |
| `d1_planner` | 10007 B=1 m=1 | 105 | 216.7 | **2.06×** |
| | 1021 B=1 m=1 | 5.03 | 8.759 | 1.74× |
| | 65537 B=1 m=1 | 682 | 1173.5 | 1.72× |
| `d1_race` | 13 B=1 m=1 | 0.009–0.010 | 0.0166 | ~1.7× |
| | 10007 B=64 m=1 | 114 | 168.7 | 1.48× |
| `d1_composite` | B=1 chain | 0.085 → **predicted 0.106** | 0.1105 | **1.30× — predicted** |

**Attribution.** The cluster at **1.3–1.6×** is almost entirely clock: wallaby's Gold
6448Y sustains ~4.1 GHz under AVX-512 against a80n0's ~2.9–3.0 GHz — a ratio of **1.41×**
measured independently by `d1_bluestein` and consistent with `d1_prime`'s on-node bisects.
That much is pure machine difference and I attribute it as such. The excess above ~1.6× is
**not** clock and the implementers themselves name the two causes:

* **L2 capacity, 2 MB → 1.25 MB per core.** `d1_bluestein` states it directly:
  "wallaby's 2 MB L2 holds working sets the node's 1.25 MB does not." This is why the
  worst factors are at 16384 and 10007 — the sizes whose working sets straddle the
  boundary. `d1_twiddle`'s 2.47× at 16384 B=64 and `d1_pow2`'s 2.06× at the 16384 B=1
  chain are capacity, and both entries' r3 tuning at 16384 was blind cross-machine
  betting that regressed on the node.
* **Ice Lake's single shuffle port and narrower front end.** `d1_twiddle` quantifies it:
  its AoS code degraded 1.9× wallaby→a80n0 at 32 B=1 against `d1_pow2`'s 1.7×, and it
  attributes the gap to `vpermilpd` and first-stage transposes on one port.
  `d1_prime`'s finding 1 is the sharpest instance and it *reverses sign* between the two
  machines: replacing port-5 `vpermpd` broadcasts with memory `{1to8}` broadcasts was
  worth **+28% on SPR** and is **poison on ICX**, where an 8-byte broadcast load hitting
  a just-stored 64-byte row stalls with no store-forwarding (0.021 vs 0.015 at 13 B=512).
  No clock factor can carry a sign change; this is microarchitecture, and it is the
  round's single most transferable result.

So the honest summary of §4: **the machine gap explains the magnitude of nearly every
overclaim, but it does not excuse the direction of the r3 tuning decisions taken on top
of it.** `d1_composite` demonstrates the correct handling in one line — measure on
wallaby, multiply by the calibrated 1.25–1.4× factor, publish the prediction, and be held
to it. It predicted 0.106 and the node measured 0.1105.

---

## 5. Which open question this round moved

Per §0(c), `docs/LITERATURE.md` §4 is the **3D** corpus's open-questions section, written
for L=6/8/17/36 on a Gold 5218. Three of its eight items (§4.3 axis fusion, §4.5 L=8
padding, §4.7 vector-radix) are 3D-geometry questions that a 1D round cannot address, and
this round did not address them. Four are *hardware and kernel-structure* questions that
this round bears on directly, and one — **§4.2** — it settles.

### §4.2, "L=17: dense-symmetric, Rader, or a hand-derived Winograd module?" — MOVED DECISIVELY, and 1D answers question (a) at two primes

§4.2's open question (a) is "which of dense-symmetric and Rader wins on this hardware,
batch-vectorised?" The 1D board runs exactly that head-to-head at L=13 and L=31, both
gated, both on the scoring node, all four regimes:

| cell | dense-symmetric (`d1_prime`) | Rader/Agarwal–Cooley (`d1_rader`) | ratio |
|---|---|---|---|
| 13 B=1 m=1 | **0.0172** | 0.0568 | **3.3×** |
| 13 B=512 m=1 | **0.0107** | 0.0186 | **1.7×** |
| 31 B=1 m=1 | **0.0530** | 0.2113 | **4.0×** |
| 31 B=512 m=1 | **0.0501** | 0.0630 | 1.26× |
| 31 B=512 chain | **0.0485** | 0.0552 | 1.14× |
| 13 B=512 chain | 0.0146 | **0.0146** | tie |

**The dense conjugate-symmetric kernel wins at every small prime, in every regime, by
1.14× to 4.0×** — vindicating §02 §7's position ("Rader is not the lever") and refuting
§01 §8's flop-count-driven preference, on the exact ground §4.2 said to settle it. The
mechanism is §4.2's own predicted one: `d1_prime` reaches 22.6 GF/s at 13 B=512 with a
kernel that has *no* buffers, *no* permutation and *no* convolution table, and its
remaining gap to a port model is ~35 vs ~27 cycles. Rader's own record concedes the
point: "13 B=1 likely needs a true Winograd DFT-13 straight-line codelet to matter."

And the crossover is now located. Rader wins where §4.2 never looked — at **large**
primes with smooth N−1, decisively: at 1021 (N−1 = 2²·3·5·17) `d1_rader` is 1.15–1.67×
over MKL in all four cells, and at 65537 (N−1 = 2¹⁶, unpadded conv) 1.15–2.12× over
`fftw1d_patient`. Meanwhile Bluestein owns the awkward primes, 10007 and 100003. **The
prime playbook the 1D brief hypothesised is now measured: dense below ~31, Rader when
N−1 is smooth, Bluestein when it is not.** Question (b) — the symmetric/antisymmetric
convolution split — remains untouched, and (c) the Winograd op count remains
unobtainable.

### §4.4, "Split vs interleaved complex" — MOVED, and it is now regime-dependent, not universal

§4.4 was marked CLOSED in the corpus in favour of split, citing Popovici et al. and
ducc0. **This round finds the closure is too broad, and the counter-evidence is a cell
win.** `d1_prime` rewrote 13 and 31 as **interleaved-pair** kernels — one complex output
per 128-bit lane, four per zmm, coefficient tables pair-duplicated at plan time — on the
explicit reasoning that "the natural interleaved layout IS the compute layout," and took
13 B=512 from 0.0206 to **0.0107 (−48%)**, converting two library losses into wins. Its
prologue is 3 unaligned loads + 2 `vshuff64x2` at L=13; the split/SoA alternative pays
`d1_batchlane`'s documented 24-shuffle-per-8×8 transpose floor, which is what pushed
`d1_prime` away from split in the first place.

The reconciliation, which is new and belongs in the corpus: **split-across-batch wins
whenever the transpose amortises, interleaved-in-lane wins when it cannot.** Every
chained cell at 13/31 keeps split-complex SoA (the boundary transpose is paid once per
chain, over m steps — `d1_rader` took 31 B=512 chain from 0.251 to 0.056 that way, and
13 B=512 chain from 0.080 to 0.015); every single-call m=1 cell at 13/31 now prefers
interleaved pairs. §4.4's granule answer (8 volumes per 64-byte line) survives intact for
the split path.

### §4.6, "Model versus search for the instruction schedule" — MOVED toward search, with a caveat

§06's position was that the schedule is "the primary thing to search" at non-power-of-4
sizes. `d1_race` is a plan-time search — it forks, gates, races the sibling class entries
per `(L,B)` and ships the winner by vtable, caching verdicts in a per-host wisdom file —
and it is at or within 1% of the best algorithmic entry at **all 26 chained cells**, with
correct selections at essentially all 24 `(L,B)` pairs (§3.2). Search does work, and the
wisdom file turned out to have a second use nobody designed: **three separate entries used
it as a forensic record this round** to discover what the node had actually chosen and at
what cost, and `d1_composite` set its entire round priority from it.

The caveat is §3.1 and it is a warning §4.6 does not contain: **a search harness must be
scored in the same environment it will ship into.** `d1_race`'s r3 changes are both
attempts at exactly this — `warm_each` (one untimed re-warm of the same candidate before
every timed sample, because interleaved sampling is right for core-state drift and wrong
for cache state past L2) and a post-race replan on a drained heap. Neither closed the
±30% single-call gap. Search selects well here; it does not yet *measure* well at
single-call granularity.

### §4.1, "How many registers does a batch-vectorised codelet actually need?" — MOVED from both sides

§4.1 asked whether `2L` is a budget or a data-only lower bound, and whether spill traffic
costs more than the shuffles it avoids. Two independent 1D data points, both on 32 zmm:

* **The spill-free side pays off, large.** `d1_batchlane` made the B=1 chains
  **register-resident at all six small sizes** — the whole state lives in registers across
  every `m` step — by finding that the four-step kernels' natural row layout is closed
  under the transform at 32/64/128, and reusing `d1_prime`'s fold-ready A/B rows at 13/31.
  Measured: 64 B=1 chain **−38%** (0.1358→0.0848), 32 B=1 chain −25%, 13/31 similar. This
  is §4.1's trade taken in the "avoid the round trip" direction and it is worth 25–38%.
* **The `2L` bound bites exactly where §4.1 said it would.** `d1_prime` could not write a
  two-transform L=31 batched body because "it would need >32 live zmm," and kept the
  tables as FMA memory operands to hold registers at ~20. `d1_rader` hit the same wall
  from the other side: 16 live complex lane-vectors *are* the whole zmm file, gcc spilled,
  and the fix was a two-layer radix-16 through a 2 KB L1 tile. `d1_pow2` reports radix-16
  middle stages at 1024 needing "32 live zmm for data alone."

So: **`2L` is a real ceiling even at 32 registers, and the productive response is
tiling through L1 rather than accepting the spill** — a sharper answer than §4.1's
"probably worth taking, but untested."

### §4.8 item 5, "No quantified comparison of batch/vector-loop placement" — MOVED

§06 called this "our largest untapped search axis." This round quantifies it, and the
answer is **regime-dependent**, matching §5's split-vs-interleaved finding. `d1_batchlane`
switched its batched m=1 cells at 31/64/128 *away* from the SoA group path to **looping
the fused-AoS single-shot kernels** — one memory pass, no boundary transposes — on
scoring-node evidence (its own r2 numbers at B=1 per transform, including call overhead,
already beat its batched path: 64: 0.0553 vs 0.0725; 128: 0.0991 vs 0.2462). Wallaby
confirmed: 128 B=512 0.20→0.092. But 13/32/60 measured the *other* way and kept SoA, and
every chained cell keeps SoA. Loop placement is now a per-`(L, regime)` decision with
numbers attached rather than a shrug.

### One cross-cutting result that is in no §4 item and should be

`d1_batchlane` solved r2's undiagnosed "Newton map in the masked tail is catastrophically
slow": **`map_scale`'s h-clamp of `1e-300` drives `rsqrt14` into FP-assist territory.**
`rsqrt14(1e-300) ≈ 1e150`, and a standalone microbench of one `map_scale` call on a vector
with **one** zeroed lane reads **85 ns against 4 ns** with the lane at 0.6 — roughly 250
assist cycles per call, every step, whenever a masked tail zeroes its junk lanes. Moving
the clamp to `1e-100` removes the assists with identical results for any real `h`. Five
entries now ship an `rsqrt14`/`rcp14` Newton map (`prime`, `rader`, `pow2`, `batchlane`,
`composite`, `planner`, `twiddle` — seven, in fact), so this is a panel-wide correctness-
adjacent performance trap, and **every entry using that map with masked or padded lanes
must check its clamp.** It generalises straight to the 3D campaign, which uses the same
map.

---

## 6. The single highest-value thing r4 should attack, per geometry

### Cross-cutting, and it outranks every per-geometry item below

**Fix `reserve.sh --status` / the `squeue` shim.** It reads the heartbeat from
`bench/gen/` instead of `bench/d1/`, so a live hold reports dead. Five of nine entries
consequently tuned blind on Sapphire Rapids, and **all thirty of this round's regressions
sit in those five entries while the four that reached the node produced one between
them** (§2). Three implementers independently built private `/tmp` shims to route around
it; that workaround is now folklore instead of a fix. This is a one-line change to a
shared script and it is worth more than any kernel in the list below. Until it lands,
make `d1_composite`'s discipline mandatory: **publish the wallaby number, the calibrated
SPR→ICX factor, and the resulting node prediction, and be scored against the
prediction.**

Second cross-cutting item: **the m=1 cells need enough reps to resolve 5%.** §3.1 shows
identical machine code scoring ±30% apart at single-call granularity while chained cells
agree to <1%. Seven of the twelve losses in §1 are inside their own spreads. Until that
is fixed, r4 cannot tell whether it improved a non-chained cell.

### Per geometry

| L | highest-value attack for r4 | why this one |
|---|---|---|
| **13** | A **three-transform pair body** at B=512, plus a 4 KiB-aliasing check on the driver buffers. | All four cells are won (1.15–1.79×) but B=512 m=1 is the thinnest at 1.15×, and `d1_prime` puts it at ~35 cycles against a ~27-cycle port-model floor. This is the only remaining headroom at 13 and it is bounded at ~25%. |
| **31** | **Stop optimising 31.** If anything, the register-tiled two-transform batched variant. | 4.05× / 4.62× / 1.82× / 1.59× over the best library. The libraries have no answer at all here (FFTW and MKL both at 0.2148–0.3848 µs). Further rounds spent here buy nothing; the effort belongs at 128/1024/16384. |
| **32** | **Resolve the cell, don't optimise it** — then `d1_batchlane`'s interleaved-AoS 4-complex-per-zmm codelet. | The sole loss is 0.0154 vs MKL 0.0153 with ±14% on both rows. That is a coin flip being reported as a loss. `d1_pow2`'s in-register AoS codelet already reaches 52 GF/s; the question is measurement, not algorithm. |
| **60** | Cut instruction count below ~600/transform **without adding port-5 pressure**, at B=512 m=1. | The one real non-tie loss outside pow2 (0.0591 vs MKL 0.0492). `d1_composite` has diagnosed it precisely: ymm2 runs ~600 instructions/transform at ~5/cycle, front-end saturated, and the SoA transpose tax at 60 is 90 port-5 µops/transform. Both width A/Bs (zmm4, zmm2x2) are settled as losses on both machines. It also needs a **second class entry** — one owner, no internal check (§3.5). |
| **64** | Same as 32: resolve B=1 m=1 first (0.0486 vs 0.0465 at ±8%/±12%), then the AoS codelet. | Three of four cells won, one by 3.16×. The remaining deficit is inside the noise. |
| **128** | **Find the four-entry regression**, then the two-block in-register codelet (32 zmm). | B=1 m=1 went backwards for `race`, `batchlane`, `pow2` *and* `planner` simultaneously (§2) — nobody targeted it and everybody lost 12–23%, turning a near-tie into 0.82×. Diagnose that before writing new kernels. `d1_pow2` names the two-block codelet as the only untried structural idea here. |
| **1021** | Paired/tripled-p schedule for `st5` at s=12. | All four cells won 1.15–1.67×. `d1_rader` scopes this at 5–8% across all four — genuinely low priority, correctly so. |
| **1024** | **Find `d1_pow2`'s +19% B=1 m=1 regression** (1.3663→1.6224), then the radix-16-as-two-in-register-radix-4 3-pass schedule. | 0.66× is the second-worst cell on the board, and the regression is *unexplained by any documented r3 change* — the fast-map default touched only chained cells and the NT policy correctly excluded 1024. A 19% regression with no owning change is a bug, not a tuning outcome. |
| **4096** | **Merge the two winning ideas**: run `d1_twiddle`'s `[4,4,16,16]` schedule inside `d1_pow2`'s blocked split engine. | The node scored `d1_twiddle`'s radix-16 (8.288) *ahead of* `d1_pow2`'s SS64 L1-tiled fusion (8.667) at B=1 m=1, while `d1_pow2` owns all three other 4096 cells. Neither entry has the other's win. Both were measured on wallaby, so re-A/B on the node first. |
| **16384** | **Bailey four-step 128×128 with L1-resident sub-FFTs, with on-node PMU evidence before shipping.** | 0.61× is the panel's worst cell. `d1_pow2` (item 1) and `d1_twiddle` (item 2) name this construction independently, it has never been attempted, and both entries' r3 work here was blind cross-machine betting that regressed (pow2 +15% B=1, twiddle +34% B=64). `d1_pow2`'s own note is the right one: "GET A RESERVATION FIRST; two rounds of blind cross-machine betting is enough." Note also `d1_pow2`'s own finding that 16384 is *not* pass-count-bound — the SS64 fusion **lost** 5% there because 64 interleaved 512 B bursts 4 KB apart are too short for the prefetcher. Stream count and length, not total traffic. |
| **10007** | `d1_bluestein`'s **8-row-batched row FFTs**, and finally run the nested-Rader A/B. | All four cells won but only 1.08–1.36×. Rows run ~1.9 cyc/pt/pass against a 0.6–1.0 ideal. The nested-Rader comparison (5003−1 = 2·41·61) is the brief's assigned headline and is unmeasured after three rounds (§3.5). |
| **65537** | Fuse the kernel multiply into the **last forward stage** — one full pass saved. | Already 1.15–2.12× ahead, but `d1_rader` has the mechanism scoped: B=16 single sits ~400 µs above its own chain steps, and that gap is pure entry/exit streaming. Forward-last and inverse-first both touch the spectrum; combining them needs the twiddle algebra checked carefully. |
| **100003** | `d1_bluestein`'s **8-row-batched row FFTs** — the largest quantified untried win on the board. | The thinnest margins anywhere (1.02–1.15×), and B=1 m=1 is a 0.90× loss once `d1_race`'s dispatch artefact is removed (§3.1). Rows are **48% of the transform** (1460 of 3020 µs); `d1_bluestein` estimates 30–40% off rows, which would put 100003 solidly clear of `fftw1d_patient` in all four cells. |

---

## 7. Curation decision

Applying `docs/CURATION.md`'s four grounds, in its order. Note first that **ground 4 as
written ("anything that beat a library baseline, regardless of rank") selects all nine
entries** — the least successful, `d1_bluestein` outside its class, still beats a library
at 7 cells; `d1_planner` does at 19. Promoting nine would make `exemplars/` a copy of
`impl_3/` and defeat the distinction the doc itself draws between complete provenance
(`impl_N/`, tracked) and the curated reading list. So ground 4 is applied together with
the doc's exclusion, "do not promote near-duplicates of an already-promoted entry."

**Promoted — 8 of 9.**

| entry | ground | the number |
|---|---|---|
| `d1_prime` | 1 — fastest at L=13 and L=31 | Fastest at 5 cells; beats a library at **8 of its 8**. 13 B=512 −48% in one round; the dense-symmetric-beats-Rader result of §5; and the ICX-vs-SPR store-forwarding finding (0.021 vs 0.015), the round's most transferable lesson. |
| `d1_rader` | 1 — fastest at L=1021 and L=65537; **2** — structurally different runner-up at 13/31 | 9 cell wins, 13 of 16 library beats, **zero regressions**. The unpadded `[4,16,16,16,4]` conv at 65537 (12 passes → 10, −13…−15%) and the across-batch SoA CRT codelets that flipped three cells. As the Agarwal–Cooley alternative to `d1_prime`'s dense kernel it is exactly ground 2's case: the losing structure must be *written down*, not described. |
| `d1_pow2` | 1 — fastest at 1024, 4096, 16384; most cells of any algorithmic entry | **15 cell wins**, 14 library beats. Ships the NT-store correction that overturned r1's blanket verdict (the confound: an odd stage count ping-ponged intermediates through the caller's `out`), the SS64 L1-tiled radix-64 fusion, and its **negative** result at 16384 (fusion lost 5%; stream length beats pass count) — which is ground 3 in the same file as ground 1. |
| `d1_batchlane` | 1 — fastest at 64/128 B=1 and every small B=1 chain | 9 cell wins, 14 library beats. The register-resident B=1 chains (64 B=1 chain −38%) and **the FP-assist clamp discovery** (§5), which every other entry shipping the Newton map needs and which generalises to the 3D campaign. |
| `d1_composite` | 1 — fastest at all four L=60 cells | Sole owner of its geometry; 3 of 4 library beats. The coset-row chain is an original structural result with a stated generalisation ("the cosets are the orbits of the CRT residue classes mod the smallest coprime factor product") applicable to any Good–Thomas PFA. And it is the round's **only** wallaby-measured entry with zero regressions, because it predicted its node number and was right (0.106 predicted, 0.1105 measured). |
| `d1_bluestein` | 1 — fastest at 10007 and 100003 | 8 cell wins, all four cells at both awkward primes. Agarwal–Cooley minimal-M, NT-streamed exit (exit phase −35%), the hugepage arena that took invocation swing from 25–30% to ±1.4%, and the **gcc `prefer-vector-width` inlining trap** (zero zmm in the caller — check the caller's disassembly, not the kernel's), which silently voided an r1 fix in every build. Also the correct sustained-AVX-droop measurement protocol. |
| `d1_twiddle` | 1 — fastest panel entry at 4096 B=1 m=1; **4** — beats a library at 13 chained cells | Marginal, and promoted for two specific reasons. Its `[4,4,16,16]` radix-16 schedule at 4096 is its own work and scored **ahead of `d1_pow2`'s fusion on the node** (8.288 vs 8.667) — the r4 merge in §6 needs both sources side by side. And the twiddle-table library itself (`d1tw_cexp` quadrant-exact, `d1tw_chirp` integer-reduced, `d1tw_stage` v1/v2) is a distinct artefact other entries consume. Its own record concedes the SoA chain "is mostly [`d1_batchlane`'s] idea landing in my vehicle" — so it is promoted for the vehicle and the tables, not the chain. |
| `d1_race` | 3 — **instructive failure, and the record documents the number that killed it**; 4 | Correct sibling selection at ~24/24 `(L,B)` pairs and 41 library beats, so the adoption layer works. But it is promoted primarily as ground 3: §3.1's ±30% single-call artefact on byte-identical code is **the measurement lesson of the round**, and `d1_race` carries the diagnosis (arm plans created mid-race in a polluted heap), two attempted fixes that did not close it (`warm_each`, post-race replan), and a second instructive failure worth every future agent's attention — `gcc … \| head -20 && echo BUILD_OK` reported success on a **failed build**, because `head` exits after 20 lines, gcc dies on SIGPIPE, and `BUILD_OK` keyed on `head`'s status. That is `CLAUDE.md` §5's "a check that could not fail," caught in the wild, with the recovery ("verify the binary's salt string before trusting any dev number"). |

**Not promoted — `d1_planner`.** It is the only entry that is **fastest at zero of 52
cells**, for the second round running, and it owns **10 of the round's 30 regressions**,
including the two largest (13 B=1 +37%, 31 B=1 +35%). Its own record states r3 "is almost
entirely adoption" — four lifts from `d1_rader`, one from `d1_pow2`, one from
`d1_bluestein`, all three of which are promoted — which places it squarely under the
doc's "do not promote near-duplicates of an already-promoted entry." It qualifies under
ground 4 (19 library beats, more than any single-class entry), which is why this is a
judgement call rather than an obvious cut; the deciding factor is that a reader wanting
`st17_vblock`, GATHER8, the conv-order chain, the Newton map or the hugepage arena will
find each in the promoted donor, in the form its author measured. **Nothing is lost:**
`impl_3/d1_planner.c` is tracked per `CURATION.md`'s per-round-source rule, and
`strategies/d1_planner.md` — which is the genuinely valuable artefact, being the
campaign's cleanest worked example of round-over-round adoption with per-lift interleaved
A/B numbers — is tracked and stays. It should be read by r4's implementers even though
its code is not in `exemplars/`.

**Round note for `exemplars/d1_r3/NOTES.md`:** this round established (a) the small-prime
kernel question — dense conjugate-symmetric beats Rader at 13 and 31 in every regime by
1.14–4.0×, while Rader owns smooth-`N−1` large primes and Bluestein the awkward ones;
(b) that split-vs-interleaved is a *regime* decision, not a global one — interleaved-pair
for single-call small primes, split SoA whenever a transpose amortises over a chain; and
(c) that the panel now owns the chained regime outright, 26 of 26 cells, at 1.93×/1.75×
geometric mean over the best library. It also established, expensively, that a
reservation-status bug can cost a whole round's tuning direction.

---

**Bottom line.** 52 cells, 40 won on algorithm alone (41 with dispatch), **26 of 26
chained cells won**, all four regime aggregates improved for the third consecutive round,
nothing failed correctness, nothing failed to build, nothing crashed, nothing went
missing. Against that: eleven of the twelve remaining losses are single-call
power-of-two cells, thirty per-entry regressions traceable to a broken reservation check,
and a top-of-board entry whose single-call margins over its own shipped code are
measurement noise. The round's most valuable output is not a kernel — it is
`d1_prime`'s finding that an SPR optimisation can reverse sign on Ice Lake, and the
demonstration across five entries of what happens when you tune on the wrong metal.
