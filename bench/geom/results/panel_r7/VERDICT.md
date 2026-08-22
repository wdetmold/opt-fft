# VERDICT — panel round `panel_r7`

Monitor's judgement on the measurements taken on `p55n3` (Intel Xeon Gold 5218, Cascade
Lake, 2×16c, exclusive, slurm job 438522, 2026-08-22T01:15), gcc 11.4.0, `-O3
-march=native`, `governor=powersave`. Roster: **19 implementations, all built, all ran,
all correct.** Sources for this round are in `bench/geom/impl_7/` (`impl` → `impl_7`).

**Two things about this round's shape before any numbers.**

1. **The previous *timed* round is `panel_r5`, not `panel_r6`.** Round 6 was halted between
   its development and timing phases (`results/panel_r6_abandoned_no_timing/WHY.md`); its
   code survives as `impl_6/` and became this round's starting point. So every number below
   prices **two rounds of work at once**, and every r6 prediction in every record is scored
   here for the first time. Several implementers structured their records around this and
   the attribution below respects it.
2. **Four geometries are on the board for the first time.** `cases.txt` gained a "wave 2":
   L = 13, 23, 45, 64. There is no previous round to compare them against, and §2 says so
   explicitly rather than inventing a baseline. The brief asks for headlines at L = 6, 8,
   17, 36; §1 gives those in full and adds a compact table for the four new ones, because
   the promotion decision in §7 needs them.

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are per transform, minimum across three independent processes, as reported by
`leaderboard.txt`. "Best library" is whichever of FFTW ×3 / MKL 2022 / MKL 2026 / ducc0
was fastest **in that exact cell**.

### L = 6 (volume 216)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L6_unrolled 0.219 µs** (L6_pfa 0.223) | mkl_dfti 0.369 µs | **1.68×** |
| B=64 (0.42 MiB) | **L6_unrolled 0.214 µs** (L6_pfa 0.226) | mkl_dfti 0.392 µs | **1.83×** |
| B=4096 (27 MiB) | **L6_pfa 0.381 µs** (L6_unrolled 0.397) | mkl_dfti 0.548 µs | **1.44×** |
| B=32768 (216 MiB) | **L6_unrolled 0.563 µs** (L6_pfa 0.572) | mkl_dfti 0.696 µs | **1.24×** |

**Read the B=1 cell down.** L6_unrolled's three runs are 0.2253 / 0.2253 / **0.2187**;
L6_pfa's are 0.2249 / 0.2234 / 0.2234. On the run distributions **L6_pfa owns the
non-batched cell by ~0.8%**, not the other way round, exactly as it did in r5 (where the
outlier was L6_unrolled's too). This is used in §7.

### L = 8 (volume 512)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L8_batchsimd 0.558 µs** (fusedaxes 0.571, radix8 0.572) | mkl_dfti 0.655 µs | **1.17×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.587 µs** (batchsimd 0.588) | mkl_dfti 0.701 µs | **1.19×** |
| B=2048 (32 MiB) | **L8_fusedaxes 0.930 µs** (batchsimd 0.945) | mkl2026_dfti 1.340 µs | **1.44×** |
| B=16384 (256 MiB) | **L8_batchsimd 1.232 µs** (fusedaxes 1.234) | mkl2026_dfti 1.785 µs | **1.45×** |

**B=1 moved for the first time in six rounds** (0.570 → 0.558, −2.1%), and not by the
mechanism anyone predicted — see §2 and §5.

### L = 17 (volume 4913)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 15.227 µs** (winograd 16.567) | fftw3_measure 81.730 µs | **5.37×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 16.715 µs** (winograd 18.080) | fftw3_estimate 81.936 µs | **4.90×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.437 µs** (winograd 21.793) | fftw3_patient 83.398 µs | **3.89×** |
| B=2048 (307 MiB) | **L17_matrixsimd 21.661 µs** (winograd 22.157) | fftw3_measure 84.013 µs | **3.88×** |

Still the board's largest margin, and still the geometry where the libraries are clustered
(FFTW 81.7–88.1, MKL 98.8–102.1, ducc0 100.4–113.9): nobody's prime-size path is special.
The leader is flat for a fourth round; the *runner-up* moved 8–10% (§2).

### L = 36 (volume 46656)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L36_mixedradix 118.532 µs** (pfa 120.252) | mkl_dfti 161.737 µs | **1.36×** |
| B=4 (5.70 MiB) | **L36_mixedradix 129.562 µs** (pfa 129.764) | mkl_dfti 175.399 µs | **1.35×** |
| B=32 (45.6 MiB) | **L36_mixedradix 166.444 µs** (pencilfused 169.577) | mkl_dfti 219.448 µs | **1.32×** |
| B=256 (364 MiB) | **L36_pfa 183.529 µs** (mixedradix 183.579) | mkl_dfti 246.452 µs | **1.34×** |

**The r5 verdict's MKL warning is resolved, and it was right.** r5 flagged that both MKL
builds got 17–25% *slower* at L=36 B=32/B=256 between r4 and r5, refused to bank the
resulting 1.54×/1.70× margins, and told the monitor to re-measure. They have reverted:
`mkl_dfti` B=32 **260.097 → 219.448** (r4 was 221.910) and B=256 **309.806 → 246.452**
(r4: 248.867); `mkl2026_dfti` likewise, 269.798 → 227.646 and 319.787 → 254.640. FFTW and
ducc0 at the same cells moved −1% to +4% over the same interval. So the r5 anomaly was
transient, specific to MKL, and is still unexplained. **The honest L=36 batched margins are
1.32× and 1.34×, not 1.54× and 1.70×**; anyone quoting the r5 figures should stop.

### The four new geometries (first node measurement, no prior round)

| L | case | fastest panel entry | best library | margin |
|---|---|---|---|---|
| 13 | B=1 | L13_direct **5.901 µs** | mkl2026_dfti 7.520 | 1.27× |
| 13 | B=16 | L13_direct **6.089 µs** | mkl2026_dfti 7.602 | 1.25× |
| 13 | B=512 | L13_direct **8.396 µs** | mkl2026_dfti 9.106 | **1.08×** |
| 23 | B=1 | L23_matrixsimd **47.717 µs** | fftw3_patient 260.825 | 5.47× |
| 23 | B=4 | L23_rader **49.232 µs** | fftw3_estimate 261.661 | 5.31× |
| 23 | B=128 | L23_rader **65.197 µs** | fftw3_patient 265.700 | 4.08× |
| 45 | B=1 | L45_mixedradix **328.012 µs** | mkl_dfti 606.851 | 1.85× |
| 45 | B=2 | L45_mixedradix **325.296 µs** | mkl_dfti 602.605 | 1.85× |
| 45 | B=16 | L45_mixedradix **406.571 µs** | mkl_dfti 679.245 | 1.67× |
| 64 | B=1 | L64_radix8 **995.782 µs** | mkl_dfti 1197.119 | 1.20× |
| 64 | B=2 | L64_radix8 **1022.973 µs** | mkl_dfti 1230.694 | 1.20× |
| 64 | B=8 | L64_radix8 **1245.366 µs** | mkl_dfti 1943.526 | 1.56× |

**Promotion ground 4 discriminates this round, for the first time in three rounds.** 18 of
19 entries beat every library in every scored cell of their own geometry. The exception is
named in §3: **L13_rader at B=512 is fourth, behind `mkl2026_dfti` and `mkl_dfti`.**

### Distance from each geometry's own port floor, at the now-measured 2.89 GHz

Each entry's floor is its own published figure; this is the round's cleanest cross-geometry
picture and it drives §6.

| L | B=1 best | cycles @2.89 GHz | published floor | ratio |
|---|---|---|---|---|
| 23 | 47.717 µs | — | 41.9 µs | **1.14×** |
| 8 | 0.558 µs | 1612 | 1248–1296 cy | 1.24–1.29× |
| 13 | 5.901 µs | — | ~4.7 µs | 1.26× |
| 6 | 0.219 µs | 632 | 486 cy | 1.30× |
| 17 | 15.227 µs | 44.0k | 33.4k cy (11.55 µs) | 1.32× |
| 36 | 118.532 µs | — | ~83 µs | 1.43× |
| 45 | 328.012 µs | — | 199 µs | 1.65× |
| 64 | 995.782 µs | — | ~550 µs | 1.81× |

---

## 2. What changed since `panel_r5`, per geometry

### Cell-level best (fastest correct entry in the cell, r5 → r7)

| L | B | r5 best | r7 best | Δ |
|---|---|---|---|---|
| 6 | 1 | 0.219 (pfa) | 0.219 (unrolled) | flat (5th round at 0.219–0.220) |
| 6 | 64 | 0.215 (unrolled) | 0.214 (unrolled) | −0.5% (inside spread) |
| 6 | 4096 | 0.391 (pfa) | 0.381 (pfa) | −2.6% |
| 6 | 32768 | 0.566 (unrolled) | 0.563 (unrolled) | −0.5% (inside spread) |
| 8 | 1 | 0.570 (radix8) | **0.558 (batchsimd)** | **−2.1% — first movement in six rounds** |
| 8 | 64 | 0.594 (fusedaxes) | 0.587 (fusedaxes) | −1.2% |
| 8 | 2048 | 0.910 (fusedaxes) | 0.930 (fusedaxes) | **+2.2% — cell-level regression** |
| 8 | 16384 | 1.254 (fusedaxes) | 1.232 (batchsimd) | −1.8% |
| 17 | 1 | 15.223 | 15.227 | flat |
| 17 | 8 | 16.658 | 16.715 | +0.3% (inside spread) |
| 17 | 256 | 21.198 | 21.437 | **+1.1% — cell-level regression** |
| 17 | 2048 | 21.983 | 21.661 | −1.5% |
| 36 | 1 | 120.358 (pfa) | **118.532 (mixedradix)** | **−1.5%** |
| 36 | 4 | 129.295 (pfa) | 129.562 (mixedradix) | +0.2% (inside spread) |
| 36 | 32 | 168.565 (pfa) | 166.444 (mixedradix) | −1.3% |
| 36 | 256 | 182.598 (pfa) | 183.529 (pfa) | +0.5% (inside spread) |

**L = 13, 23, 45, 64: no comparison exists.** These are first measurements. Nothing
regressed because nothing was measured before.

### L = 6 — flat for a third round, and now for a *settled* reason

Both entries spent the round on the same bet, from opposite directions, and both lost it
cleanly. L6_pfa shipped `fused_zx` (zmm x-pass + the node-proven ymm fused y/z stage, 1546
vs 1834 uops per volume, −16%); L6_unrolled shipped `zxf` (−17%) and `zff` (−25%, with the
aligned-load construction that fixes r5's `z2s` line-splitting). **The node selected none of
them, in any of eight cells, across all three processes.** Both picked `fused` / `fused_pf`
/ `fused_pf(w)` / `fused_pfw(_xa)` — ymm, as in r5 and r4.

What makes this round's rejection different from r5's is that **the confound r5 could not
exclude is gone**. Both entries independently implemented per-candidate licence dwell
(~0.5–0.7 ms untimed in the candidate's own licence immediately before each timed slice),
precisely because CLX licence state persists ~670 µs and a round-robin race therefore
systematically penalises a minority-licence candidate. And both measured `kclk` — the
licence clock *in the chosen kernel's own context* — at **2.89 GHz**. So there is no
licence transition to pay, the race was fair, and a 17–25% uop reduction still lost. See
§5: this is the round's headline corpus result and it is a *falsification*, not a win.

Cell numbers: B=4096 improved 2.6% (L6_pfa's `fused_pf_xa`, its r6 ascending-x-pass twin,
finally scored); everything else is inside spread. B=1 has now read 0.219–0.223 for five
consecutive rounds.

### L = 8 — the first B=1 movement in six rounds, and it came from the entry nobody expected

**L8_batchsimd took B=1 (0.558) and B=16384 (1.232)**, its first cells ever won. The
mechanism at B=1 is not its own: it is `MODE_FUSED`, a wholesale port of L8_fusedaxes'
fused structure with L8_batchsimd's own 52-op FMA codelet substituted (48 FP instructions
lighter). Its three B=1 runs picked LANEX3 / FUSED / FUSED — the two FUSED runs read 0.5577
and 0.5647, both under every other entry's best. **Its own prediction for this cell was
"0.570–0.574 stands, no change expected."** It beat its own prediction by adopting a rival's
shape into a regime it had never offered it in.

L8_fusedaxes held B=64 (0.594 → 0.587) and B=2048, but **B=2048 regressed 0.910 → 0.930
(+2.2%)** — the round's clearest cell-level regression at L=8. Its code at that cell is not
byte-identical to r5's (the arena and candidate sets changed around it) but its pick string
is the same `fused+pfs+pfw`; the three runs are 0.9724 / 0.9445 / 0.9299, a 4.6% spread
against r5's much tighter distribution, so part of this is measurement noise and part is not
explained. Its round's actual bet — the `fusedAA` / `seq3AA` anti-4K-aliasing shapes, with a
brute-forced permuted-k1 order and execute-time scratch base selection — was **declined in
every cell**; B=1 kept `fused` at 0.571 against a predicted `seq3AA` at 0.50–0.55.

L8_radix8 is third in all four cells (0.572 / 0.612 / 0.980 / 1.277) but produced the
round's most useful L=8 result; see §5 (§4.3).

### L = 17 — the leader is frozen, the runner-up moved 9%

**L17_winograd is the round's biggest single improvement anywhere on the board:**
B=1 18.177 → **16.567 (−8.9%)**, B=8 19.668 → **18.080 (−8.1%)**, B=256 23.933 →
**21.793 (−8.9%)**, B=2048 24.567 → **22.157 (−9.8%)**. It is now second in all four cells
(it was third at B=1/B=8), and at B=256/B=2048 it is within **1.7% / 2.3%** of the leader.
The mechanism is `g8`: laying the fused mini-buffer **ky-major** instead of
spectator-major, which deletes the pass-2 transposed store entirely (−48 shuffles, −16
scalar stores, −~68 stack uops per w=8 group), plus `ES`, which stores each output vector
the moment it is produced because the store is now plain. It predicted "~16.3–17.0 and pick
g8" and measured 16.567 with `g8` picked. Its two companion experiments both failed and
both are recorded with numbers: kernel E alone (input-load folding) was a wash, and
two-group unroll-and-jam cost +3–5%. Its own synthesis is the transferable one: *at ~850
uops/group, neither deferring work nor pairing groups pays; only deleting uops does.*

**L17_matrixsimd is flat and lost one cell.** B=1 15.223 → 15.227, B=8 16.658 → 16.715,
B=256 21.198 → **21.437 (+1.1%, a regression)**, B=2048 21.983 → 21.661 (−1.5%). Its r6
`deferred-Z` was selected at B=256 and B=2048 and *not* at B=1/B=8 — and the cell where it
was selected is the cell that regressed. Its r7 lever, the cosine-resident `cr` variants,
turned out **not bit-identical** to its class representatives (gcc 11 contracts the unrolled
straight-line form differently — the fourth documented instance) and so was **not selectable
by its own tuner**; it ships forceable but unmeasured. Its round therefore produced no
scoreable kernel, honestly documented.

**L17_rader is flat and now third in all four cells** (17.156 / 18.586 / 24.467 / 25.537
against r5's 17.098 / 18.605 / 24.843 / 25.500). Its round's structural bet — `dz`, a
one-plane-deep software pipeline removing all three store→load junctions per plane, adopted
from matrixsimd's deferred-Z — was **selected in zero cells**, and neither was its r6
`pfw`-at-B=1 (all four picks read `xl 512t, pf=0, pfw=0`). That is the **second** scheduling
attack from this entry rejected by the node in two rounds (`ov` in r5, `dz` here), which is
itself the finding: see §6.

### L = 36 — a lead change, and two mechanisms closed by null

**L36_mixedradix took B=1, B=4 and B=32**; L36_pfa retains B=256 by 0.05 µs (0.03% — a tie).
mixedradix B=1 122.755 → **118.532 (−3.4%)**, recovering the cell it lost in r5 and going
below the r4 best (119.021) for the first time; B=32 177.726 → 166.444 (−6.3%); B=256
215.882 → 183.579 (−15.0%). L36_pencilfused improved hugely at streaming (B=32 −9.3%, B=256
−19.0%) by finally scoring its r6 `istream+pfw` port, but **regressed at B=1, 121.255 →
123.987 (+2.3%)**, and is third in all four cells. L36_pfa is flat everywhere (B=1 −0.1%,
B=4 +0.4%, B=32 **+0.9%**, B=256 +0.5%) — it lost three cells without getting slower.

**Both of the round's L=36 bets were rejected on the node, in every entry, in every cell
where they could physically apply.** (i) NTA-hinted read prefetch, built to fix pfa's
L2-thrash diagnosis (in+out = 1.5 MB against a 1 MB L2), shipped by all three entries in
three different forms (pfa's `pf=4`, mixedradix's `nta`, pencilfused's mode 9 `istream+nta`
and mode 10 `inplace+nta` with a column-order cursor): **zero picks**. (ii) pfa's r6
quiet-window finding that `pfw` wins at B=4 (−8%), which mixedradix reproduced at −13–15%
on wallaby and both admitted at B≥2: **rejected at B=4 by all three entries** — the r5
in-arena rejection stands and the wallaby B=4 measurement did not transfer. (iii) pfa's
`scratch+NT pf=4`, the never-before-fielded combination that was to give NT the read-side
protection it always lacked: **rejected**. Every implementer pre-registered the null branch
with numbers, so all three records now close cleanly (§5).

### The regressions, named

* **L8_fusedaxes at B=2048 — 0.910 → 0.930 µs, +2.2%.** The round's largest cell-level
  regression, in a cell this entry has owned since r5 and with the same pick string
  (`fused+pfs+pfw`). Run spread 4.6% (0.9724 / 0.9445 / 0.9299) against r5's much tighter
  distribution. Partly noise; not fully explained. The candidate sets and arena around this
  path did change (`fusedAA`/`seq3AA` added, anchors re-based) — the same class of
  unexplained regression the r4 and r5 verdicts pinned on refactors around an untouched hot
  path.
* **L17_matrixsimd at B=256 — 21.198 → 21.437 µs, +1.1% against a 1.3% run spread.**
  Marginal, but it is the cell where its r6 `deferred-Z` was newly selected. If deferred-Z
  is what did this, the mechanism is negative at B=256 and its selection is a tuner
  hysteresis artifact; the entry should force the A/B before deepening it.
* **L36_pencilfused at B=1 — 121.255 → 123.987 µs, +2.3% against a 0.9% run spread.** It
  fell *outside its own predicted null band* (119–123). Its r5 B=1 number was already
  flagged in the r5 verdict as resting on a monotone warm-up across processes (134.058 /
  125.565 / 121.255, 10.6% spread); this round's distribution is tight (123.988 / 125.140 /
  124.410), so the honest reading is that r5's 121.255 was the outlier and this entry has
  been at ~124 all along. Either way it is now third at B=1.
* **L36_pfa at B=32 — 168.565 → 170.083 µs, +0.9% against a 1.0% run spread.** Inside
  spread, and see §3(a): the reported minimum came from the one run in three that picked
  `pw=4` while the other two (and the checked run) picked `pw=2`.
* **L17_rader at B=1/B=2048 and L6_pfa at B=64** moved +0.3% / +0.1% / +4.1% — L6_pfa's
  B=64 is the one that deserves a word: 0.217 → 0.226. Its r5 number was itself flagged in
  the r5 verdict as an outlier (0.226 / 0.226 / **0.217**); this round it reads 0.2257 /
  0.2291 / 0.2256. **On typical runs L6_pfa's B=64 is unchanged across three rounds at
  ≈0.226**, and the r5 "−2.3% gain" the verdict declined to bank was indeed not real.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed to build.** `build_errors.txt` is present and empty (0 bytes).

**Nothing crashed, hung, or timed out.** `failures.txt` does not exist, which for
`sweep.sh` means no backend invocation exited non-zero across 28 cases × 3 runs.
`agents/exits.txt` records `exit=0` for all 19 implementers.

**Nothing is missing.** All 19 entries in `impl_7/` appear in the leaderboard; all 19 have
strategy records in `strategies/` with a `panel_r7` section (2871 lines added across the
19 files); all 19 have a correctness verdict at every scored batch size of their own
geometry, with no gaps. Entries that decline a foreign geometry report
`supported: false` and are correctly absent from those tables (1392 lines of
`"does not support"` in `timing.err` are this, not failures).

**Nothing failed correctness.** Every panel entry passes at every scored batch size, with
relative L2 error against numpy between **1.28e-16 and 4.46e-16** against a 1e-12 tolerance
— four orders of magnitude inside the gate, which is where a genuinely correct
double-precision transform lands. `check.py` compares the full `(B, L, L, L)` array
element-by-element, not a sample. **There is no fast wrong answer in this round.**

**No rule violations found.** `git status --porcelain` is empty for `driver.c`,
`fft3d_api.h`, `Makefile`, `sweep.sh`, `check.py`, `leaderboard.py`, `gen_input.py`,
`cases.txt`, `tryout.sh`, `PANEL_BRIEF.md`, `promote.sh`, `panel_round.js` and `sota/` —
the measurement apparatus is untouched, including the wave-2 case list which was committed
in r5. No `#pragma omp`, `omp_*` call, pthread symbol, `dlopen`, or `system()` appears
anywhere in `impl_7/`. **Zero** occurrences of `fftw_*`, `DftiC*`, `DFTI_*` or `ducc`
symbols anywhere in `impl_7/` — not even in comments this round.

### The one entry beaten by a library

**`L13_rader` at B=512 is fourth in its cell at 9.469 µs, behind `mkl2026_dfti` (9.106) and
`mkl_dfti` (9.237).** It is the only panel entry in this round — and, by my reading of the
r4/r5 verdicts, the first in three rounds — to lose a cell to a vendor library. Evidence:
`leaderboard.txt` L=13 B=512 table; runs 9.5393 / 9.4686 / 9.5458 (0.8% spread, so this is
not a bad draw). It is 1.13× behind its own stablemate `L13_direct` (8.396) in the same
cell, having been 1.03× behind at B=1 — i.e. it degrades with batch. Its record predicted
the opposite ("expect the pfw win to exceed wallaby-B2048's −6%"). This does not disqualify
it from promotion (§7) — it beats both MKL builds at B=1 and B=16 and it is the only Rader
arm at L=13 — but the cell is named here so the record carries the number.

### L = 23 is running **one** algorithm, not two

This is the round's sharpest adversarial finding and it changes a promotion.

`L23_matrixsimd` and `L23_rader` produce **bit-identical output at all three scored batch
sizes**. The correctness JSONs agree to the last digit of `rel_l2`, `max_abs` *and*
`rel_max` simultaneously — e.g. `rel_l2 = 3.8208093982081066e-16` at B=1,
`3.8128132902489255e-16` at B=4, `3.8008643630267765e-16` at B=128 — over 12167·B complex
doubles. Two genuinely different summation orders cannot do that.

The records confirm it rather than contradict it. `L23_rader`'s own r6 work "settled the
algorithm question by operation count: no realization of the cyclic-11 convolution pair
(Winograd/CRT, Karatsuba, FFT-22) beats the conjugate-folded direct form on hardware where
FMA = add = one FP-port cycle." Both entries now publish the **same op count** — 297 vector
FP ops per chunk, 409 zmm chunks per volume, 594 real flops per line — and both this round
independently adopted X-first pass order, the 1058→1064-double t1 plane pad, deferred-Z and
paced `prefetchw`. Their measured times differ by 0.3% (47.717 vs 47.854), 0.4% and 0.6%.
**"rader23" is a name, not an algorithm**; L=23 is one dense conjugate-folded kernel
implemented twice. `CURATION.md` rule 2 ("a *structurally different* runner-up") therefore
does not select a second L=23 entry, and one of the two must go. §7 decides which and why.

A second consequence: **the correctness check at L=23 is effectively one independent
verification, not two.** Two bit-identical implementations agreeing tells you nothing extra.
The check against numpy still holds, so the result stands, but the redundancy the panel
usually gets from two entries per geometry is absent here.

### (a) In seven cells the number in the leaderboard was produced by a plan variant whose output was never checked

`sweep.sh` runs each backend three times, reports the **minimum**, and runs `check.py` on
the output left by the **last** run. Fifteen cells had a tuner that picked differently
across processes; in seven of them the fastest process was not the checked one. Stripping
clock-probe digits (which differ harmlessly between runs), the genuine cases are:

| cell | reported time (run) | variant **timed** | variant **checked** (run 3) |
|---|---|---|---|
| L23_matrixsimd B=1 | 47.717 (r1) | `pinned X-first` | `pinned X-first **pipelined**` |
| L23_matrixsimd B=4 | 49.440 (r2) | `pinned X-first` | `pinned**+park** X-first` |
| L23_matrixsimd B=128 | 65.593 (r1) | `pinned X-first pf=0 **pw=1**` | `pinned**+park** X-first pf=0 **pw=0**` |
| L17_matrixsimd B=2048 | 21.661 (r2) | `X-first pf=0 pw=0` | `X-first **deferred-Z** pf=0 pw=0` |
| L36_pfa B=32 | 170.083 (r1) | `**pw=4** inplace pf=2` | `**pw=2** inplace pf=2` |

**`L23_matrixsimd` is the worst offender on the board: all three of its cells** report a
number produced by a variant that was never checked, and at B=128 the *prefetch policy*
differs between the timed and checked runs. Mitigations, stated fairly: the entry
cmp-verified all 18 selectable X-first variants bit-identical on full outputs at B=4 and
re-checked five at B=64, its shipped bit-class discipline is documented in the file, and —
usefully — its outputs are bit-identical to `L23_rader`'s, which *was* checked in its own
right and picked different variants. I do not believe the numbers are wrong. **The harness
did not prove them**, and this entry's exposure is systematic rather than incidental.

L17_matrixsimd's B=2048 exposure is the same one r5 recorded and it holds for the same
reason (its class members are cmp-verified against representatives). L36_pfa's B=32 flip is
a prefetch-width difference (`pw=4` vs `pw=2`) that changes the schedule but not the
arithmetic; the reported 170.083 is the pw=4 run and the two pw=2 runs read 170.656 /
171.701, so reading the cell as ≈170.7 changes nothing about its rank.

The eight benign flips (timed variant == checked variant, or clock-probe digits only) are
L6_pfa B=64, L8_batchsimd B=1 and B=64, L8_fusedaxes B=1/B=64/B=16384, L17_matrixsimd
B=1/B=256, L17_winograd B=8, L23_rader B=1, and L45_pfa B=16.

### (b) Two headline cells rest on a minimum that is an outlier against their own other two runs

Minimum-of-three is the standing convention and I am not overriding it, but these should
not be read as confirmed:

* **L6_unrolled B=1**: 0.2253 / 0.2253 / **0.2187** (min 3.0% below the other two). Reported
  0.219. **L6_pfa's B=1 is 0.2249 / 0.2234 / 0.2234**, so on the run distributions L6_pfa
  reproducibly owns the L=6 non-batched cell by ~0.8%, not the 1.8% deficit the leaderboard
  shows. This is the *third* round in which L=6's B=1 ordering is decided by which entry
  drew the lucky process. Used in §7.
* **L8_batchsimd B=2048**: 0.9450 / 0.9842 / 0.9876 (min 4.2% below the other two).
  Reported 0.945, second in cell. Read honestly at ≈0.984 it is still second (L8_radix8's
  typical run is ≈1.002), so the rank survives; the *number* does not.

L8_batchsimd's B=1 win, by contrast, survives scrutiny: its min (0.5577) and its checked run
(0.5647) are the same `FUSED` variant, and both beat every other entry's best in the cell.

### (c) Bit-identity fingerprints, fourth round running

When `rel_l2`, `max_abs` and `rel_max` all agree to the last digit, two entries are
producing bit-identical output over millions of complex doubles.

* **L23_matrixsimd ≡ L23_rader at all three batch sizes** — the finding above.
* **L36_pfa ≡ L36_pencilfused at B=32 and B=256**, distinct at B=1 and B=4 — unchanged from
  r4/r5, and still explained: pencilfused's `istream` mode is an acknowledged translation of
  pfa's in-place path. Arithmetic-order identity, not code duplication.
* **L8_batchsimd ≡ L8_radix8 at B=64 and B=16384** (r5: B=1, B=2048 and B=16384). They now
  **differ at B=1 and B=2048** — batchsimd's B=1 pick is the ported `FUSED` shape (which
  carries fusedaxes' axis order) and radix8's B=2048 pick is `3p`. The fingerprint
  independently confirms that batchsimd's B=1 lead change is a real structural change and
  not a relabel, and that the two entries genuinely diverged this round.
* **All three L=17 entries remain numerically distinct** (3.2865 / 3.1192 / 3.2719 e-16 at
  B=1). **The two L=13 entries are distinct** (2.9118 vs 4.0243 e-16) — dense-folded vs
  Rader-13 really are two algorithms. **The two L=45 entries are distinct** (4.0973 vs
  4.0355 e-16) despite heavy mutual borrowing, and **the two L=64 entries are distinct**
  (4.1682 vs 4.4622 e-16). So of the eight geometries, only L=23 has collapsed to one
  algorithm.

### (d) The harness floor is absent from the five largest cells

`sweep.sh`'s rule skipping `baseline_matrix` when `L³·B > 2e6` now removes the library-free
reference from L=6 B=32768, L=8 B=16384, L=17 B=2048, L=36 B=256 **and L=64 B=8**. Nothing
scored depends on it, but the round's five biggest cells have no harness floor. Noted, as
in r5; not a change this round.

### (e) Provenance note

`git status` shows `D bench/geom/impl/baseline_matrix.c`. This is not a deletion: `impl` was
converted from a tracked directory to a symlink (`impl → impl_7`) and the tracked file moved
into the untracked `impl_7/`. `impl_6/`, `impl_7/` and `results/panel_r7/` are all untracked
at the time of writing. **Commit them before round 8 starts** — this is the exact exposure
`CURATION.md` records as having destroyed panel_r1's eleven implementations.

---

## 4. Claimed numbers versus measured numbers

The development machine (`wallaby`, Xeon Gold 6448Y, Sapphire Rapids, two 512-bit FMA units,
2 MB L2/core, 60 MB L3) is not the scoring machine (Gold 5218, Cascade Lake, one 512-bit FMA
unit, 1 MB L2/core, 22 MB L3), and wallaby additionally swings ~1.95× between its base and
turbo clock states between sessions (established in r5 and re-confirmed by four entries this
round). The panel's standing calibration puts the full wallaby-to-node span at ~2.9×.

### Ratios of claimed (wallaby) to measured (node), same entry, B=1

| entry | claimed (wallaby) | measured (node) | ratio |
|---|---|---|---|
| L6_unrolled | 0.108 | 0.219 | 2.03× |
| L6_pfa | 0.129 | 0.223 | 1.73× |
| L8_batchsimd | 0.305 | 0.558 | 1.83× |
| L8_fusedaxes | 0.319–0.320 | 0.571 | 1.79× |
| L8_radix8 | 0.308 (slow state) | 0.572 | 1.86× |
| L13_direct | 3.102 | 5.901 | 1.90× |
| L13_rader | 3.21 | 6.054 | 1.89× |
| L17_matrixsimd | 8.71–9.86 | 15.227 | 1.54–1.75× |
| L17_winograd | 8.540 | 16.567 | 1.94× |
| L17_rader | 8.889 | 17.156 | 1.93× |
| L23_matrixsimd | 21.32 | 47.717 | 2.24× |
| L23_rader | 20.89–21.67 | 47.854 | 2.21–2.29× |
| L36_mixedradix | 54.2–55.2 | 118.532 | 2.15–2.19× |
| L36_pfa | 51.8 | 120.252 | 2.32× |
| L36_pencilfused | 53.2 | 123.987 | 2.33× |
| L45_mixedradix | 167.7 | 328.012 | 1.96× |
| L45_pfa | 171.2 | 343.316 | 2.01× |
| L64_blocked | 658.8–665.5 | 1087.455 | 1.63–1.65× |
| L64_radix8 | 512.0–544.0 | 995.782 | 1.83–1.95× |

**Every ratio lies inside the 1.5–2.4× machine band, comfortably inside the ~2.9× that MKL
alone spans between these two machines. None of these is an implementer error**, and the
largest (L=23 at 2.2–2.3×) is
exactly the geometry whose scratch working set (t1 + plane buffers ≈ 8.5 KB × 23 planes)
straddles the 1 MB / 2 MB L2 difference between the two machines. Where a claim deserves
scoring it is on **direction**, not magnitude.

### The predictions that were right, and they are the round's best work

1. **L17_winograd went four-for-four and named the pick.** It predicted B=1 "~16.3–17.0 and
   pick `g8`" and measured **16.567 with `g8` picked**; predicted B=8 and B=256 would move to
   `g` and follow wallaby's −3%/−10% and measured **−8.1% / −8.9% with `g4`/`g8` picked**;
   predicted B=2048 would be mostly hidden under DRAM and it moved −9.8% on `g4`. It also
   predicted correctly that this would *narrow, not take*, the cell against matrixsimd. This
   is the strongest prediction sheet since L36_pfa's four-for-four in r5, and the reasoning
   behind it — a disassembly cost model of the group loop (296 FP against ~850 total uops,
   with 108 stack uops and 96 transposed-store uops named) — is the r5 verdict's §6 ask
   ("nobody has costed the kernels' ~90 constant loads plus ~60 stack moves per block")
   executed literally and then acted on.
2. **L64_radix8 predicted a cross-machine sign reversal and got it.** Its own wallaby tuner
   picked NT stores in every regime (B=1 552.3, B=8 510.3, B=64 653.3, all beating
   `pfw+slabpf` by 5–37%). It predicted that *the node* would reject NT and pick
   `pfw+slabpf` anyway, on the strength of the r5 verdict's L=36 rule. The node picked
   **`plain+slabpf` at B=1 and `pfw+slabpf` at B=2 and B=8**. Predicting the opposite of your
   own machine's measurement, from another geometry's node evidence, and being right, is the
   most transferable thing in this round's records.
3. **L23_rader's aliasing bet.** It isolated the t1 plane-stride pad with a stride-only A/B
   (identical binaries, same window, alternating: 1058 doubles → 30.5–31.1 µs, 1064 doubles
   → 21.0–23.8 µs), attributed it to 4K aliasing, and predicted node B=1 "near 45–50 µs ≈
   1.1–1.2× the 41.9 µs floor". Measured **47.854** (cell best 47.717 = **1.14× floor**, the
   tightest floor ratio on the board).
4. **Both L=6 entries called `kclk = 2.89` and both pre-registered the falsifying branch.**
   L6_pfa: "If kclk=2.89 AND zx is rejected → uops are simply not the B=1 limiter (third
   falsification)." L6_unrolled: "If zmm rejected AND kclk=2.89 → the uop theory is dead with
   clean attribution." Measured kclk = 2.89 on two independent probes, zmm rejected 8/8.
   Getting the value of a round from a measurement you designed to be able to lose is the
   right way to spend a round on a hypothesis you cannot test locally, and both did it.
5. **All three L=36 entries pre-registered the NTA null with numbers** ("pick stays
   `inplace/cached pf=0/1` at 119–123", "a clean null that closes NTA for L=36"), and the
   null fired: mixedradix 118.532 with `v1-cached-pf0`, pfa 120.252 with `inplace pf=0`,
   pencilfused 123.987 with `inplace`. Three files, one closed question, no ambiguity.
6. **L23_matrixsimd beat its own range**: predicted ~53–58 µs at B=1, measured **47.717**.
7. **L8_radix8 hit three of four** (B=1 0.570 → 0.5719; B=64 0.615–0.625 with pick `1f-pfs`
   → 0.6122 with `1f-pfs`; B=16384 1.25–1.32 → 1.2767 with `1f-pfs-pfw`).

### The predictions whose direction failed

8. **L8_fusedaxes' entire round was declined, and it is the second consecutive round in
   which this entry's headline mechanism did not transfer.** It predicted "B=1: pick
   `seq3AA`, 0.50–0.55 µs" on a careful 4K-alias enumeration (14 blocked loads/volume in
   phase A for *any* scratch placement; a brute-forced permuted-k1 order giving 0 phase-B
   collisions; execute-time scratch base selection) plus a ROB argument (fused phase B ≈280
   uops/iteration against a 224-entry ROB). The node picked **`fused`** and measured
   **0.571**, i.e. its own stated "if NOTHING transfers" branch. It was honest about that
   branch in advance and the mechanism is not disproved — but it is unmeasured, because a
   tuner declining a candidate inside a 1% hysteresis band is not the same as the candidate
   losing. See §6: the counter run it asked for is the cheapest open item at L=8.
9. **L45_pfa claimed a same-window lead over its rival in every batched cell and landed
   behind in all of them — the second precisely-controlled cross-machine reversal the project
   has recorded.** Its record states: "Same-window rival comparison (back-to-back, this
   session): … **I now lead every batched cell (182.6 vs 192.8, 234.9 vs 249.1)** and am
   within noise at B=1." On the node it is **4.7% behind at B=1** (343.316 vs 328.012),
   **8.9% behind at B=2** (354.242 vs 325.296) and **4.2% behind at B=16** (423.732 vs
   406.571). Its B=1 range (225–275 µs) missed by 25%. This is the same shape as
   L36_pencilfused's r5 failure (parity on wallaby, 10.9% short on the node) with the sign of
   the wallaby comparison reversed, and it reinforces the same rule: **a same-window
   dev-machine A/B between two entries does not predict their node ordering.** Its `pf`
   prediction also failed — it expected `pf1` or `pf3` at B=1 on a 1 MB-L2 argument and the
   node picked **`pf0`**, so the PF45 poke is pure overhead on the scoring machine too.
10. **Every first-contact geometry's port-floor prediction was optimistic, except L=23.**
    L13_direct predicted 4.7–5.5 µs at B=1 and measured 5.901 (+7% over the top);
    L13_rader predicted 4.5–5.2 and measured 6.054 (+16%); L45_mixedradix predicted 255–310
    and measured 328.0 (+6%); L45_pfa predicted 225–275 and measured 343.3 (+25%);
    L64_radix8 predicted 0.55–0.75 ms and measured **0.996 ms (+33%)**. The one that landed
    (L23_rader) predicted from a *measured* mechanism's effect size, not from a floor. **A
    port floor is a lower bound and the non-FP residue on this node is 14–81% depending on
    the geometry (§1); it is not predictable from the dev machine.** Next round's predictions
    at these four sizes should be anchored on this round's measured ratio, not re-derived.
11. **L36_pencilfused fell outside its own null band** at B=1 (predicted 119–123 if NTA was
    rejected; measured 123.987) and **L64_blocked's `st=1` fork resolved on the dead branch**
    (it predicted batched cells might flip to `st1 nt` on the node's 1 MB L2; the node kept
    `st0 cached` in all three cells, so by its own words "the 2-sweep idea is dead on both
    machines and should be recorded as such"). Recorded.

### A methodology finding worth more than most kernels: the build-flag gap

L45_pfa discovered, by objdump-diffing its rival's kernel against its own, that **the scored
build does not carry `-funroll-loops`** (which `tryout.sh` does), and that its code was **10%
slower without it** (202 µs vs 222–234 in a paired A/B/A/B with an MKL sentinel held flat).
It pinned the codegen with a file-level `#pragma GCC optimize("unroll-loops")`. The same
diff found its own `phase1_pw4` carrying **758 scalar instructions against 42 in its rival's
entire transform**, because gcc materialised a 48-entry offset table and reloaded an offset
into `%rbx` before every vector load; hoisting one base pointer per block took it to 217
scalar instructions and −14% at B=1. **Both findings apply to every entry in the panel and
neither is geometry-specific.** Any entry writing codelet macros with runtime offsets should
count the scalar instructions in its hot function before touching anything else.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### The primary move: §4.8 item 6 — AVX-512 on the scoring part. **Closed, and the r5 synthesis is itself refuted.**

r5 settled `clk512 = 2.89 GHz` on four of five probes but left `clk256` open, with three
probes reading 3.89, one 2.89 and one 3.27, and called it "the more consequential half …
one process and ~5 ms of work". **That measurement was made and the disagreement is
resolved: it was probe density.**

| entry | probe | reads |
|---|---|---|
| L6_pfa (design), L6_unrolled (adopted) | sparse chain / dense chain / 512-bit / **kernel context** | clkS256 **3.89**, clkD256 **2.89**, clkS512 **2.89**, **kclk 2.89** |
| L17_matrixsimd | sparse + dense 256 | clk256 3.89, **d256 3.89**, clk512 2.89 |
| L17_winograd | saturating dual-width | 2.89 / 2.89 |
| L8_fusedaxes | serial + saturating, both widths, one process | 3.27 / 2.42 (256 s/p), 2.43 / 2.42 (512 s/p) |

Reading: a **sparse** FMA chain reads the non-AVX licence at 3.89 GHz; a **dense** one reads
the AVX2/AVX-512 licence at 2.89. That is exactly Intel's table for the Gold 5218 (non-AVX
3.9, AVX2 2.9, AVX-512 2.9) and it is precisely the structured explanation r5 hypothesised
but could not confirm. L8_fusedaxes' probe reads 0.84× low at *every* width and density
while preserving the ratio (2.42/3.27 = 0.740), so it is a calibration defect in that probe,
not a machine disagreement; L17_matrixsimd's `d256 = 3.89` is the one genuinely discordant
reading and its chain is presumably still too sparse to trip the licence.

**The decisive number is `kclk`: the licence clock measured inside the chosen kernel's own
context, 2.89 GHz.** L6_pfa designed it in r6 and L6_unrolled adopted it; both report it in
every cell of the leaderboard. So the panel's ymm kernels run at 2.89, not 3.89, and
**L=6's B=1 is 632 cycles against a 486-cycle two-FP-port floor — 1.30×, ~147 cycles of
unexplained overhead, not the 366 that two entries spent r4 and r5 chasing.** The prize is
less than half what it was thought to be. Every cycle-per-volume figure in every record
should be re-derived at 2.89 GHz for both widths.

**And with the clock known, the width question was answered behaviourally — against the
corpus *and* against the r5 verdict.** §4.8 item 6 says 512-bit is "strictly preferable on
this part … at **zero** frequency cost". The r5 verdict, seeing eight zmm candidates
rejected at L=6, proposed instead that "the cost is the licence transition, not the width."
**Both are now refuted by the same experiment.** This round:

* Three new zmm families were built at L=6 across two entries — L6_pfa's `fused_zx` (−16%
  uops/volume), L6_unrolled's `zxf` (−17%) and `zff` (−25%, with an aligned-load
  construction that removes r5's line-splitting `loadu` defect and the t2 round trip). All
  have the *identical* 486-cycle FP-port floor by design (one 512-bit FMA unit ⇒ zmm at
  1/cy, ymm at 2/cy); they trade nothing at the floor and delete only front-end and
  load/store uops.
* Both entries independently added **per-candidate licence dwell** — the exact fix for the
  bias the r5 reading rested on. There is no transition to pay and no table-order penalty.
* **The node rejected all three families in all eight cells, across all three processes.**

So: at *equal* licence clock, with the race made licence-fair, a 17–25% uop reduction still
loses at L=6. The corpus's rule and the r5 verdict's rule are both too general. **The
measured statement is: width buys nothing where the kernel is not front-end-bound, and at
L=6 it is not.** The complementary r5 result stands and is unaffected — at L=17, where every
kernel is already zmm, mixing a ymm tail in was selected again in every cell.

### Also moved, and this is the round's largest single mechanism: §4.5 — padding and 4K aliasing

§4.5's stated open question — measure the padding effect and read
`ld_blocks_partial.address_alias` — has never been touched by anyone. It was attacked from
two directions this round and the results are opposite, which is more informative than
either alone.

* **L=23: a one-constant change worth −25 to −30%.** L23_rader padded the t1 plane stride
  1058 → 1064 doubles (8464 B → 8512 B = 133 whole cache lines) and isolated it with a
  stride-only A/B — identical binaries but for the constant, same window, pinned,
  alternating: **30.5–31.1 µs → 21.0–23.8 µs at B=1**. Its diagnosis is 4K aliasing rather
  than line-splitting: the X pass read `in` at plane stride 8464 B and wrote t1 at *the same*
  8464 B stride, so every one of the 23 loads sat in a fixed mod-4096 relation to every one
  of the 23 stores of the same chunk, for all 133 chunks, every volume. Padding decorrelates
  the two stride sequences (272 vs 320 mod 4096) *and* aligns everything to 64 B.
  L23_matrixsimd shipped the same pad independently. On the node, **L=23 B=1 landed at 1.14×
  its port floor — the tightest floor ratio of any geometry on the board**, against 1.24–1.32×
  for L=6/8/13/17 and 1.43–1.81× for L=36/45/64. This is §04's rule ("with a one-cache-line
  granule, pad so every stride is an *odd* number of cache lines") and §08's ducc0
  `make_noncritical()` guard, confirmed on the scoring machine at a size nobody had padded.
* **L=8: the same model, built more carefully, and declined.** L8_fusedaxes enumerated its
  residues in Python (14 blocked loads/volume in phase A for *any* scratch placement — a
  spacing-8 comb; 12–16 in phase B set by an allocation lottery; 0 to ~128 in seq3's pass
  B2), built `fusedAA`/`seq3AA` to drive all of them to zero with brute-forced permuted-k1
  orders, and the node picked plain `fused` in all four cells.

**The reconciliation is the useful part**, and it should go into the corpus: **L=23's
aliasing was self-inflicted — its own scratch stride, which the plan owns — while L=8's is
between the driver's `in` and `out` buffers, whose addresses no entry controls.** §4.5's
second half (§08 §1.8: L=8's 8192-byte volume stride makes 4K aliasing "maximally
degenerate") predicts an effect the panel cannot fix from inside a plan. That is a
conclusion about where to look, not a null result — and the counter has *still* never been
read (§6).

### Also moved: §4.5's store-policy half — "hide the RFO, don't avoid it" now holds at **five** geometries, including two on first contact

* **L=36**: NTA read-prefetch rejected in all three entries; `scratch+NT pf=4` — the
  combination built specifically to give NT the read-side protection every previous NT
  attempt lacked — **rejected**. `pfw` selected at both streaming cells in all three entries.
* **L=23**, first contact: wallaby's tuner picked "512-bit pinned X-first **NT planes**" at
  streaming; the node picked **`pw=1`** at B=128 in both entries.
* **L=64**, first contact: L64_radix8's wallaby tuner picked NT in every regime; the node
  picked `plain+slabpf` at B=1 and **`pfw+slabpf`** at B=2 and B=8. L64_blocked picked
  `cached`, never NT.
* **L=8 and L=6**: `pfw` in every streaming pick, as in r5.

**Non-temporal stores are now 0-for-everything on this node across five rounds, eight
geometries and nineteen entries' own tournaments** — including, this round, the one variant
designed to remove the specific defect that was blamed for their previous losses. On the
Gold 5218 this can be stated as a rule rather than an observation: *hide the RFO with
`prefetchw`; do not avoid it with NT stores.* The converse also held again — prefetchw on
cache-resident lines was rejected at every B=1 and, this round, at every L=36 B=4 as well.

### Also moved, and **reversed back**: §4.3 — is axis fusion worth 3× or 3%?

r5 recorded that a single fused pass won all three batched L=8 cells by 11–20% and flagged
the experiment as confounded, because L8_fusedaxes had changed pass count *and* store policy
in the same round. **The r5 verdict made the isolating experiment §6's L=8 priority, two
entries ran it, and the answer is `pfw`, not fusion.**

L8_radix8 built `{1f, 3p} × {pfw, no-pfw}` into one same-process tuner table:

| B (wallaby) | 1f-pfs-pfw | 3p-pfs-pfw | 3p-pfs | 1f-pfs | 3p-pf |
|---|---|---|---|---|---|
| 5632 | 0.6164 | **0.6030** | 0.8543 | 0.9538 | 0.8365 |
| 16384 | **0.7405** | 0.7375 | 1.0248 | 1.1898 | 0.9807 |

`pfw` is worth −27% to −38% over its no-pfw twin at both shapes; with `pfw` on both, `3p`
and `1f` are statistical twins (≤2%, either side); `1f` *without* `pfw` is the worst
candidate in both tables. **The node's own picks agree and are the stronger evidence**:
L8_radix8's node picks are **`3p-pfs-pfw` at B=2048** and **`1f-pfs-pfw` at B=16384** — it
split across shapes while keeping `pfw` in both. So r5's apparent overturning of §4.3 is
withdrawn: **the r3 answer ("single-digit percent, sometimes negative") stands**, Tolmachev's
rule survives, and the 0.910 µs r5 headline was the RFO hiding.

Fusion is not worthless — L8_batchsimd took B=1 by porting the fused shape into a regime it
had never offered it in (§2) — but it is worth single digits, not 20%, and the panel should
stop tuning three-pass structures against a fused winner as though the pass count were the
variable.

### §4.6 — model versus search: the search protocol was debugged again, and the theme is now *licence fairness*

Third consecutive round in which tuner-protocol fixes were worth more than kernel work:

* **Per-candidate licence dwell** (L6_pfa's design, independently built by L6_unrolled,
  adopted by both L=23 entries from L17_rader r6). CLX licence state persists ~670 µs —
  comparable to a whole trial — so in a round-robin race a ymm candidate timed straight after
  a zmm one ran up to a third of its trial at the lower clock, and the bias hits the same
  candidates every round so per-candidate minima do not wash it out. This is what makes this
  round's zmm rejection interpretable where r5's was not.
* **Two-sweep candidate ranking** (L17_rader). Even with r5's settle spin and r6's
  per-candidate warmups, one wallaby table spanned **27 → 10 µs/t monotonically** down the
  table for near-identical work. Two full fixed-order sweeps with a per-candidate min across
  both fixed it. Any fixed-order tuner on a powersave-governor machine needs this.
* **The licence tail** (L6_unrolled r6, adopted by L6_pfa): a `create()` that *ends* with a
  512-bit probe hands the driver a licence-degraded core, and at B=1 the whole sample set can
  fit inside the recovery window. Both entries now end `create()` by dwelling in the chosen
  kernel.
* **Build-flag parity** (L45_pfa, §4) — worth 10% and invisible from every wallaby table.
* Two independent re-triggerings of the raw-ssh measurement trap (L17_matrixsimd, L23_rader):
  a remote command without a leading `cd` silently produces nothing, `grep PASS` eats the
  error, and a subsequent `cp` copies a *stale* output file — so a `cmp` loop "verifies" four
  copies of one file. L17_matrixsimd's first two "BIT-IDENTICAL" results this round were this
  artifact and the truth (DIFFERS) only appeared once every forced run asserted its own PASS
  line. **Any wallaby number without an adjacent PASS line is void.**

Search still beats model, and for the fourth round running the binding constraint is whether
the measurement inside the search is honest.

### §4.2 — L=17 dense vs Rader vs Winograd, and a new data point at L=13/L=23

§4.2 asks which of dense-symmetric and Rader wins, batch-vectorised. The panel's answer is
now three-sizes deep and consistent:

* **L=17**: dense nested cyclic/negacyclic (matrixsimd) leads, the hand-derived 17-point
  module (winograd) is second and closing to within 1.7% at batch, Rader-17 is third — all
  three numerically distinct, so the comparison is real.
* **L=23**: the question is *settled by counting and the panel converged* — L23_rader's own
  r6 analysis found no realization of the cyclic-11 convolution pair (Winograd/CRT,
  Karatsuba, FFT-22) beats the conjugate-folded direct form where FMA = add = one FP-port
  cycle, and both entries now run that form (§3). At L=23, **Rader loses on op count before
  any measurement**.
* **L=13**, the interpolating case both L=23 records asked the panel for: dense conj-folded
  (L13_direct) beats Rader-13 in all three cells, by 2.6% at B=1 widening to 13–20% at batch,
  and Rader-13 is the round's only entry beaten by a library. **The dense form wins at 13,
  17 and 23**; §02 §7's "Rader is not the lever at L=17" now reads as understated.

---

## 6. The single highest-value thing the next round should attack, per geometry

### L = 6 — stop shipping kernels. The uop theory is dead; run the counter.

Three zmm families across two entries, 17–25% fewer uops, raced with the licence bias
removed, at a now-*measured* equal licence clock: **zero picks in eight cells**. Combined
with r4's port-5/uop falsification and r5's out-of-order-window falsification, that is three
independent mechanisms falsified *on the node* for the same 147 cycles. B=1 has read
0.219–0.223 for five rounds and the geometry is 1.68× the best library.

**The single thing: one `perf stat` run, which r5 also asked for and which did not happen.**
`perf stat -e uops_issued.any,cycles,resource_stalls.rob,resource_stalls.rs,cycle_activity.stalls_mem_any,ld_blocks.store_forward`
on forced `L6_FORCE=fused` vs `zff` vs `fused_zx` at B=1. Both entries ship the forcing
switch specifically so the monitor can do this, and both records name the same remaining
suspect that no uop count can see: **the t1 store→load joint at the pass boundary**. Without
it, round 8 at L=6 is a fourth round of guessing. Secondary, and a real question for the
panel rather than the implementers: with the prize now known to be ~147 cycles and the
mechanism list exhausted, is a second L=6 entry still earning its slot?

### L = 8 — read `ld_blocks_partial.address_alias`, then decide whether the mechanism is even reachable.

L=8 B=1 is 1612 cycles against a 1248–1296 cycle floor. L8_fusedaxes built the most detailed
model any entry has produced of where that residue is (14 structurally-unavoidable blocked
loads per volume in phase A; 12–16 more set by an allocation lottery in phase B) and the
node's tuner declined it inside a 1% hysteresis band — **which is not the same as the
mechanism losing, and nobody has measured it forced.** Meanwhile L=23 just showed the same
mechanism worth −25–30% when the offending stride was one the plan owned.

**The single thing: `perf stat -e ld_blocks_partial.address_alias,cycles` on forced
`-DL8_VARIANT=0` vs `-DL8_VARIANT=12` at B=1** — the entry's own ask, verbatim, one run per
binary. Its model predicts ~26–30 blocked loads per volume against ~2. If the counter
confirms it and forced AA still does not win, the panel learns that CLX's per-stall cost is
small and can close the mechanism. If the counter shows the aliasing is between the
*driver's* `in` and `out` buffers rather than the entry's scratch, the panel learns that L=8's
alias exposure is **structurally out of reach from inside a plan** and three entries can stop
spending rounds on it. Either answer is worth more than another kernel. Streaming at L=8 is
converged (three entries within 3.6% on one technique) — stop tuning it.

### L = 17 — take `g8` into the leader, and force `cr`.

The only mechanism that has moved an L=17 cell in three rounds is **uop deletion**, it paid
8–10% in four cells, and **it is in the runner-up, not the leader.** L17_winograd's `g8`
(ky-major mini-buffer + direct store, deleting a whole 8×8 transposed store per pass-2 group)
took it from third to second and to within 1.7% at B=256. Against that, the node has now
declined *three* scheduling attacks at this geometry: rader's `ov` (r5, 0/4), rader's `dz`
(r7, 0/4) and matrixsimd's `deferred-Z` at the small cells (0/2, and the one cell where it
*was* selected regressed 1.1%). Winograd's own jam experiment cost +3–5%. The evidence is
one-directional.

**The single thing: apply the ky-major / direct-store surgery to L17_matrixsimd's structure**
— its B=1 is 44.0k cycles against its own 33.4k-cycle mixed-shape floor and has not moved in
three rounds, and winograd's record already names the identical deletable transpose in the
three-pass `b8` variant. Cheap and owed alongside it: **force matrixsimd's `cr` pairs on the
node** (`-DL17_FORCE=38` vs `44`, and `42` vs `45`, at B=1; `43` vs `47` at B=2048), because
that lever is built, verified and *unmeasurable by its own tuner* — the cr variants fall
outside its bit-selection class through a gcc contraction difference, so the entry cannot
score them without a node number to justify moving the class rule. One forced same-window
pair decides it. Also worth one forced A/B: matrixsimd's `deferred-Z` at B=256, where its
selection coincided with the round's regression.

### L = 36 — the counter run, third time of asking; then stop tuning the batched cells.

Two mechanisms were closed by null this round (NTA in three forms, `scratch+NT pf=4`) and one
wallaby finding failed to transfer (pfw at B=4). The geometry sits at 118.5 µs = **1.43× its
~83 µs port floor**, the worst ratio of the four original geometries, and the diagnosis that
two entries built rejected mechanisms on — pfa's L2-thrash story (in+out = 1.5 MB against a
1 MB L2) — is now **unsupported by any positive result**. It may still be right; NTA may
simply be the wrong instrument. It cannot be settled from wallaby, whose 2 MB L2 holds both
buffers, and all three records have now requested the same measurement in three consecutive
rounds.

**The single thing: `perf stat -e l2_rqsts.all_demand_miss,LLC-loads,idq.dsb_uops,idq.mite_uops`
on one B=1 run.** It discriminates the L2-thrash story from a front-end story, both of which
are pre-registered forks in the implementers' own records, and it costs one run. Do it before
anyone writes another L=36 candidate. Secondary: B=256 is at its own modelled traffic floor
(183.5 measured against 175–183 modelled) and B=32/B=256 moved ≤0.5% for the cell in two
rounds — **stop tuning the batched cells**. Also for the monitor: watch the MKL L=36 batched
baselines, which moved −16% and −20% back to r4 levels this round after an unexplained r5
excursion (§1).

### The four new geometries, in one line each

* **L = 13 — the margin is 1.08× at B=512 and one entry lost to MKL there.** This is the
  thinnest cell on the board. L13_direct ships `-DL13_FORCE=10` (staged-Z burst copy)
  specifically because the node's B=512 truly streams (36 MB > 22 MB L3) and wallaby cannot
  reproduce that regime; run that one A/B. If the margin does not open, L=13 is the geometry
  where the panel has the least to add.
* **L = 23 — algorithmically finished at 1.14× floor; the open item is the counter that
  explains *why*.** `perf stat -e ld_blocks_partial.address_alias` on the 1058-vs-1064
  binaries at B=1 would convert the round's best mechanism from a hypothesis into a rule the
  other seven geometries can apply. The remaining kernel lever (row-padded t1 to fix the
  Y-pass load splits, at +1.2% volume FP) is one tuner-gated experiment.
* **L = 45 — 1.65× floor, the second-worst ratio on the board, and both entries agree the
  bulk is phase-1 data movement.** The standing structural idea (fuse the z-store-transpose
  with the y-load) is now justified by the floor ratio; also run the `-DPPITCH=48` and
  `FFT45_PF=0/1` one-flag A/Bs both entries asked for, and propagate L45_pfa's scalar-count
  audit (§4) to every entry with runtime-offset macros.
* **L = 64 — 1.81× floor, the worst on the board, and the largest untried structural move in
  the corpus applies here.** LITERATURE §4.3's re-opened case is L2↔DRAM tiling ("tile the
  batch so a tile fits L2, then run all three axes inside the tile"), where §08 measures a 7×
  bandwidth gap against the 2.6× the panel has always fused across. At 8 MiB per volume, L=64
  is the only geometry where that experiment is even meaningful. The 2-sweep question is
  closed (`st=1` rejected on both machines); hugepages are worth +3.3–3.7% and should be
  swept on the node's smaller STLB (`FFT64B_NOHP`).

---

## 7. Promotion

Against `docs/CURATION.md`, in its stated order. Sources are in `bench/geom/impl_7/`.

**1. Fastest correct entry per geometry.** The brief scores non-batched and batched
separately, so where a geometry splits cleanly by regime, both cell-owners qualify.

* **L = 6 — both.** `L6_unrolled` holds B=1, B=64 and B=32768 on the leaderboard minima;
  `L6_pfa` holds B=4096 and — on the run distributions, which is the honest statistic here —
  **also B=1** (0.2234 typical against 0.2253, §3b). This is the same 2–2 regime split the r5
  verdict recorded with the entries swapped, and it is decided by process luck three rounds
  running. Both also carry a *measured negative* that closes a question (rule 3): three zmm
  families, 17–25% fewer uops, licence-fair race, kclk = 2.89, **zero picks in eight cells**.
  L6_pfa additionally originated `kclk` and the per-candidate licence dwell that the rest of
  the panel adopted.
* **L = 8 — `L8_batchsimd` and `L8_fusedaxes`.** batchsimd takes B=1 (0.558, the first
  movement in that cell in six rounds) and B=16384 (1.232); fusedaxes takes B=64 (0.587) and
  B=2048 (0.930). This reverses the r5 verdict's decision not to promote batchsimd: it is no
  longer bit-identical to L8_radix8 at B=1 or B=2048 (§3c), and it won its cells with a
  cross-entry port whose distinct fingerprint proves the transfer was real.
* **L = 13 — `L13_direct`**, fastest in all three cells at this geometry's first measurement.
* **L = 17 — `L17_matrixsimd`**, fastest in all four cells for the fifth round running.
* **L = 23 — `L23_rader`** (see the ruling below). It holds B=4 and B=128 on the minima and
  on typical runs, and B=1 is a 0.3% tie inside spread.
* **L = 36 — `L36_mixedradix`**, fastest at B=1, B=4 and B=32, and **`L36_pfa`**, which holds
  B=256 (by 0.03%, a tie) after four cells in r5.
* **L = 45 — `L45_mixedradix`**, fastest in all three cells.
* **L = 64 — `L64_radix8`**, fastest in all three cells.

**2. Structurally different runner-up, close behind.**

* **`L17_winograd`** — the round's largest improvement anywhere (−8.1% to −9.8% in all four
  cells), second in all four, **within 1.7% and 2.3% of the leader at B=256/B=2048**, and
  numerically distinct from both rivals (3.2719e-16 vs 3.2865 and 3.1192 at B=1). It also has
  the round's only four-for-four prediction sheet and the disassembly cost model behind it.
* **`L13_rader`** — 2.6% behind at B=1 and *verified* structurally different by fingerprint
  (4.0243e-16 vs 2.9118e-16), so L=13 really is running Rader against a dense conj-folded
  kernel. Keeping it is what makes L=13 the interpolating prime the L=23 records asked the
  panel for. Its B=512 loss to MKL (§3) is recorded in the promotion note, not hidden.
* **`L64_blocked`** — 5–13% behind on a genuinely different structure (8×8 two-stage,
  three-sweep, hugepage odd-line-padded scratch, distinct fingerprint), at a geometry on
  first contact. It also carries this round's hugepage measurement (+3.3–3.7% at B=8, three
  non-overlapping build pairs) and the `st=1` two-sweep kill on both machines.

**3. Instructive failures — slower for a documented and measured reason.**

* **`L8_radix8`** — third in all four cells (+2.5% to +5.4%) and the entry that **answered
  §4.3**. Its crossed `{1f, 3p} × {pfw, no-pfw}` table and its split node picks
  (`3p-pfs-pfw` at B=2048, `1f-pfs-pfw` at B=16384) are what withdrew r5's "fusion wins"
  reading. Keeping both arms in the exemplar set is what makes the L=8 counter experiment in
  §6 reproducible from the reading list alone.
* **`L45_pfa`** — behind in all three cells, and the reason it is worth a slot is precisely
  that: it claimed a **same-window** wallaby lead over its rival in every batched cell
  (182.6 vs 192.8, 234.9 vs 249.1) and landed **4–9% behind on the node**. That is the second
  precisely-controlled cross-machine reversal the project has recorded and the first with the
  sign of the dev-machine comparison inverted. Its record also carries the two most portable
  findings of the round: the **`-funroll-loops` build-flag gap** (worth 10%, invisible from
  every wallaby table) and the **scalar-instruction audit** (758 scalar instructions vs 42 in
  a rival's whole transform, fixed by hoisting one base pointer per block, −14% at B=1).

**4. Anything that beat a library.** Selects 18 of 19 — every entry except `L13_rader` at
B=512 (§3). Recorded as a fact about the round; it discriminates only that one cell, which is
already accounted for above.

### Not promoted, with reasons

* **`L23_matrixsimd`** — **not a structurally different runner-up: it produces bit-identical
  output to `L23_rader` at all three batch sizes** (§3), publishes the identical op count, and
  converged on the same X-first / padded-t1 / deferred-Z / `pw` configuration. `CURATION.md`
  forbids near-duplicates and rule 2 does not select it. Between the two, `L23_rader` is
  promoted because (i) it owns two of three cells on both the minima and the typical runs,
  (ii) its record contains the **isolated stride-only A/B** that is the round's most
  transferable mechanism (−25–30% at B=1, §5), where matrixsimd shipped the same pad as one
  of five simultaneous changes with no isolating measurement, and (iii) matrixsimd is the
  board's worst timed-≠-checked offender, with **all three** of its cells reporting a number
  from an unchecked variant (§3a). Its genuine contributions — the deferred-Z port, the
  licence warmup, and the r6 padding suspicion that rader confirmed — are credited in rader's
  record and recorded in §5 here.
* **`L17_rader`** — third in all four cells and flat (+0.3% / −0.1% / −1.5% / +0.1%). Its
  round's structural bet `dz` was **selected in zero cells**, as `ov` was in r5: two
  independent scheduling attacks from this entry rejected by the node in two rounds. That is
  a real finding and it is recorded in §2, §5 and §6 — but the entry is already in
  `exemplars/panel_r5`, its 17-point kernel is winograd's and survives inside the promoted
  `L17_winograd`, and the *positive* L=17 lesson this round (delete uops, don't reschedule
  them) is carried by winograd. Its two-sweep tuner-ranking fix is recorded in §5.
* **`L36_pencilfused`** — third in all four cells, **the round's clearest B=1 regression**
  (121.255 → 123.987, +2.3%, outside its own predicted null band), and bit-identical to
  `L36_pfa` at B=32 and B=256. Its instructive content — the `B ≤ 2` physics gate for NTA with
  the numbers that forced it (forced mode 10 at B=4: 136.8 vs ~76, **+80%**) — duplicates the
  NTA verdict already carried by both promoted L=36 entries. Its r5 exemplar remains the
  reference for the cross-machine transfer lesson.

---

## Provenance and housekeeping

`impl_6/`, `impl_7/` and `results/panel_r7/` are **untracked** at the time of writing, and
`impl/baseline_matrix.c` shows as deleted only because `impl` became a symlink to `impl_7`
(§3e). `CURATION.md` records that panel_r1's eleven implementations were lost exactly this
way. **Commit `impl_6/`, `impl_7/`, `results/panel_r7/`, the strategy records and the
promoted exemplars before round 8 starts.**

Outstanding monitor-side measurements, consolidated from §6 and the records, in cost order
(each is one or two runs, and between them they adjudicate five mechanisms):

1. `perf stat -e ld_blocks_partial.address_alias,cycles` — L8 `-DL8_VARIANT=0` vs `12` at
   B=1, and L23 `L23_T1P` 1058 vs 1064 at B=1.
2. `perf stat -e l2_rqsts.all_demand_miss,LLC-loads,idq.dsb_uops,idq.mite_uops` — L36 at B=1.
3. `perf stat -e uops_issued.any,cycles,resource_stalls.rob,cycle_activity.stalls_mem_any` —
   L6 `L6_FORCE=fused` vs `zff` vs `fused_zx` at B=1.
4. Forced pairs: L17 `-DL17_FORCE=38/44` and `42/45` at B=1; L17_matrixsimd deferred-Z on/off
   at B=256; L13 `-DL13_FORCE=10` at B=512; L45 `-DPPITCH=48` and `FFT45_PF=0/1` at B=1;
   L64 `FFT64B_NOHP`.

PROMOTE: L6_pfa L6_unrolled L8_batchsimd L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pfa L45_mixedradix L45_pfa L64_blocked L64_radix8
