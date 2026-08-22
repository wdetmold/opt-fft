# VERDICT — panel round `panel_r5`

Monitor's judgement on the measurements taken on `p55n3` (Xeon Gold 5218, Cascade Lake,
2×16c, exclusive, slurm job 438486, 2026-08-21T21:59), gcc 11.4.0, `-O3 -march=native`.
Roster: 11 implementations, all built, all ran, all correct. Sources for this round are in
`bench/geom/impl/`.

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are per transform, minimum across three independent processes, as reported by
`leaderboard.txt`. "Best library" is whichever of FFTW ×3 / MKL 2022 / MKL 2026 / ducc0
was fastest in that exact cell.

### L = 6 (volume 216)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L6_pfa 0.219 µs** (L6_unrolled 0.220) | mkl_dfti 0.375 µs | **1.71×** |
| B=64 (0.42 MiB) | **L6_unrolled 0.215 µs** | mkl_dfti 0.393 µs | **1.83×** |
| B=4096 (27 MiB) | **L6_pfa 0.391 µs** | mkl_dfti 0.558 µs | **1.43×** |
| B=32768 (216 MiB) | **L6_unrolled 0.566 µs** | mkl_dfti 0.693 µs | **1.22×** |

### L = 8 (volume 512)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L8_radix8 0.570 µs** (fusedaxes 0.573, batchsimd 0.574) | mkl_dfti 0.648 µs | **1.14×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.594 µs** | mkl_dfti 0.720 µs | **1.21×** |
| B=2048 (32 MiB) | **L8_fusedaxes 0.910 µs** | mkl2026_dfti 1.333 µs | **1.47×** |
| B=16384 (256 MiB) | **L8_fusedaxes 1.254 µs** | mkl2026_dfti 1.788 µs | **1.43×** |

### L = 17 (volume 4913)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 15.223 µs** | fftw3_patient 81.710 µs | **5.37×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 16.658 µs** | fftw3_estimate 81.914 µs | **4.92×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.198 µs** | fftw3_estimate 83.430 µs | **3.94×** |
| B=2048 (307 MiB) | **L17_matrixsimd 21.983 µs** | fftw3_measure 88.373 µs | **4.02×** |

Still the board's largest margin, and still the geometry where the libraries are clustered
(FFTW 81.7–88.9, MKL 98.9–103.5, ducc0 101.8–114.7): nobody's prime-size path is special.

### L = 36 (volume 46656)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L36_pfa 120.358 µs** | mkl_dfti 161.844 µs | **1.34×** |
| B=4 (5.70 MiB) | **L36_pfa 129.295 µs** (mixedradix 129.742) | mkl_dfti 175.388 µs | **1.36×** |
| B=32 (45.6 MiB) | **L36_pfa 168.565 µs** | mkl_dfti 260.097 µs | **1.54×** |
| B=256 (364 MiB) | **L36_pfa 182.598 µs** | mkl_dfti 309.806 µs | **1.70×** |

**Do not bank the L=36 batched margins.** Both MKL builds got substantially *slower*
between r4 and r5 in exactly those two cells, by the same amount: `mkl_dfti` B=32
221.910 → 260.097 (+17.2%) and B=256 248.867 → 309.806 (+24.5%); `mkl2026_dfti`
230.079 → 269.798 (+17.3%) and 256.835 → 319.787 (+24.5%). Over the same interval FFTW
moved +1.8% to +5.3% and ducc0 moved −1.2% to −2.2%, and MKL at L=36 B=1/B=4 and at
L=17 B=2048 (307 MiB) and L=8 B=16384 (256 MiB) is flat to slightly faster. So this is
not a working-set effect and not a general node effect; it is specific to MKL at L=36
batched, run spreads are tight in both rounds (0.1–0.5%), and it is unexplained. The
panel-side gain at B=256 is the 16.6% L36_pfa earned; the rest of the margin change is
the baseline moving. Re-measure the MKL L=36 batched cells before anyone cites 1.70×.

**All 11 panel entries beat every library baseline in their own geometry in every scored
cell, for the second round running.** Promotion ground 4 ("anything that beat a library")
therefore again selects the whole roster and discriminates nothing; it is recorded as a
fact about the round rather than used as a criterion.

---

## 2. What changed since `panel_r4`

### Cell-level best (fastest correct entry in the cell, r4 → r5)

| L | B | r4 best | r5 best | Δ |
|---|---|---|---|---|
| 6 | 1 | 0.219 (unrolled) | 0.219 (pfa) | flat |
| 6 | 64 | 0.214 (unrolled) | 0.215 (unrolled) | +0.5% (inside spread) |
| 6 | 4096 | 0.387 (pfa) | 0.391 (pfa) | +1.0% (inside spread) |
| 6 | 32768 | 0.566 (unrolled) | 0.566 (unrolled) | flat |
| 8 | 1 | 0.570 | 0.570 (radix8) | flat |
| 8 | 64 | 0.623 (fusedaxes) | 0.594 (fusedaxes) | **−4.7%** |
| 8 | 2048 | 1.136 (radix8) | **0.910** (fusedaxes) | **−19.9%** |
| 8 | 16384 | 1.418 (radix8) | 1.254 (fusedaxes) | **−11.6%** |
| 17 | 1 | 16.431 | **15.223** | **−7.4%** |
| 17 | 8 | 18.008 | 16.658 | **−7.5%** |
| 17 | 256 | 21.626 | 21.198 | −2.0% |
| 17 | 2048 | 22.290 | 21.983 | −1.4% |
| 36 | 1 | **119.021** (mixedradix) | **120.358** (pfa) | **+1.1% — the round's only cell-level regression** |
| 36 | 4 | 129.921 (mixedradix) | 129.295 (pfa) | −0.5% |
| 36 | 32 | 174.226 (pfa) | 168.565 (pfa) | −3.2% |
| 36 | 256 | 218.899 (pfa) | **182.598** (pfa) | **−16.6%** |

**L = 6 — flat for a second round, and now for a documented reason.** Both entries ran
their most ambitious rounds yet and the node rejected both mechanisms where it mattered.
L6_unrolled built eight mixed-width AVX-512 candidates whose uop counts are 15–27% below
its ymm incumbents (`z2s` 1464, `z3t` 1266, vs ymm-fused 1728 per volume); the node tuner
selected **none of them in any of four cells**, keeping `fused` / `fused_pf` /
`fused_pf` / `fused_pfw`. L6_pfa's out-of-order-window hypothesis (`fused_sp`,
`fused_sp2`, `_u2`) was rejected at B=1 — the cell it was built for — and selected only
at B=64. B=1 has now read 0.219–0.220 for four consecutive rounds. The round's product at
this geometry is a measurement, not a kernel: the licence-clock pair (§5), which explains
the zmm rejection quantitatively.

**L = 8 — the round's largest win and a lead change.** L8_fusedaxes took all three
batched cells: B=64 0.623 → 0.594, B=2048 **1.313 → 0.910** (−31% for the entry, −19.9%
for the cell), B=16384 1.585 → 1.254. Its node picks are `fused+pfs+pfw` in both
streaming cells — the first time write-intent prefetch has been selected at L=8, and it
beat the plain `pfs` configuration that won r4. Underneath, everyone improved by adopting
somebody else's node-proven mechanism: L8_radix8's B=64 fell 0.680 → 0.619 on the `1f`
fused shape it ported from L8_fusedaxes (pick `avx512-1f-pfs`, 3/3), and L8_batchsimd's
streaming cells fell 1.215 → 1.096 and 1.642 → 1.388 on the spread prefetch it ported
from L8_radix8 (pick `LANEX3 nt=0 pf=s0`, 3/3). Note the pattern: in all three cases the
entry that *invented* the mechanism is not the entry that won with it.

**L = 17 — the first real movement at the top in three rounds.** L17_matrixsimd adopted
L17_rader's r4 mixed 512-bit+ymm-tail shape; the node selected it in all four cells and
it moved B=1 16.431 → 15.223 (−7.4%) and B=8 18.008 → 16.658 (−7.5%). The batched cells
moved only −2.0% / −1.4%, which is consistent with their overhead being memory-side.
L17_rader improved in all four cells (−3.6% / −3.6% / −1.4% / −0.8%) — but its headline
bet, the `ov` overlapped-shuffle execs, was **selected in zero cells**; the gain came
from a tuner-protocol fix (a clock-settle spin before ranking) that changed which
incumbent wins at B=1 (`xl 512t pin` → `xl 512t`). L17_winograd is flat (18.325 → 18.177,
−0.8%) and its `d8`/`e8` deferred-transpose variants were **selected in zero cells**.
Ordering is unchanged: matrixsimd everywhere, rader 2nd at B=1/B=8, winograd 2nd at
B=256/B=2048.

**L = 36 — one entry now owns the geometry outright, and the other two converged on its
structure.** L36_pfa took all four cells for the first time. B=256 218.899 → 182.598
(−16.6%) and B=32 174.226 → 168.565 (−3.2%) on the new `pf=2` write-intent prefetch
(picks `inplace pf=2`, 3/3 in both cells), and it recovered the B=4 regression the r4
verdict pinned on its per-plane refactor (132.347 → 129.295) by making those functions
`always_inline` — the r4 attribution was correct and the fix landed. L36_mixedradix
(B=32 −19.8%, B=256 −5.6%) and L36_pencilfused (−14.9%, −4.9%) both improved by porting
pfa's in-place + paced-prefetch structure into their own files, and both remain behind
the original while running it.

### The regressions, named

* **L36_mixedradix at B=1 — 119.021 → 122.755 µs, +3.1% against a 0.3% run spread.** The
  round's only cell-level regression, and it cost the entry a cell it had held for two
  rounds. The pick string is **identical to r4's** (`v1-cached-pf0`, arena 1 vol,
  ntpolicy=0), so the executed path is nominally unchanged; the only difference at B=1 is
  that the candidate pool grew 9 → 12 with a new execute body that B=1 never runs. That
  makes code layout the leading suspect — the same class of unexplained regression the r4
  verdict pinned on L36_pfa's per-plane refactor, and which L36_pfa fixed this round by
  restoring inlining. mixedradix should apply the same discipline before adding anything.
* **L6_pfa at B=64 — reported −2.3% (0.222 → 0.217), but see §3(b).** Its three runs read
  0.226 / 0.226 / 0.217. On the typical run it is +1.8% against r4. The `fused_sp2_pf`
  plane pipeline should not be recorded as a confirmed gain on this evidence.
* **L8_radix8 lost both streaming cells it won in r4** (B=2048 1st → 3rd, B=16384 1st →
  3rd), but its own numbers improved (1.136 → 1.116, 1.418 → 1.402). It was overtaken,
  not regressed.
* L36_pencilfused at B=4 (132.360 → 132.656, +0.2%) is inside spread. Everything else
  improved or moved inside its cell's run-to-run spread.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed to build.** `build_errors.txt` is present and empty (0 bytes).

**Nothing crashed, hung, or timed out.** `failures.txt` does not exist, which for
`sweep.sh` means no backend invocation exited non-zero across 16 cases × 3 runs.
`results/panel_r5/agents/exits.txt` records `exit=0` for all 11 implementers.

**Nothing is missing.** All 11 entries in `impl/` appear in the leaderboard, all 11 have
strategy records in `strategies/`, and all 11 have correctness verdicts at all four of
their scored batch sizes — 44 `c_*.json` files, one per entry per cell, with no gaps.

**Nothing failed correctness.** Every panel entry passes at every scored batch size, with
relative L2 error against numpy between **1.26e-16 and 3.96e-16** against a 1e-12
tolerance — roughly four orders of magnitude inside the gate, which is where a genuinely
correct double-precision transform lands. `check.py` compares the full `(B, L, L, L)`
array element-by-element, not a sample, so there is no room for an entry to be right only
where it is looked at. **There is no fast wrong answer in this round.**

**No rule violations found.** `git status --porcelain` is empty for `driver.c`,
`fft3d_api.h`, `Makefile`, `sweep.sh`, `check.py`, `leaderboard.py` and `gen_input.py` —
the measurement apparatus is untouched. No OpenMP pragma, `omp_*` call or pthread symbol
appears anywhere in `impl/`. Ten grep hits for FFTW/MKL exist across five files and every
one is a comment citing a published operation count (Burrus T7.1, FFTW's `n1_6`/`n1_8`/
`dft-generic-17`) or a recorded MKL comparison number; there is no vendor DFT call.

**One harness change *is* in this round, committed and benign, with one consequence worth
stating.** `sweep.sh` gained a `cases.txt` hook and a rule skipping `baseline_matrix`
when `L³·B > 2e6` ("too expensive to be informative"). The effect is that the library-free
floor is **absent from the four largest cells** — L=6 B=32768, L=8 B=16384, L=17 B=2048,
L=36 B=256 — where r4 had it (12 `c_baseline_matrix` files this round, 16 last round).
Nothing scored depends on it, but the round's four biggest cells have no harness floor.

Four things the measurement did *not* establish:

**(a) In two cells the number in the leaderboard was produced by a plan variant whose
output was never checked.** `sweep.sh` runs each backend three times, reports the
*minimum*, and runs `check.py` on the output left by the *last* run. Five cells had a
tuner that picked differently across processes; in two of them the fastest process was
not the checked one:

| cell | reported time (run) | variant timed | variant checked (run 3) |
|---|---|---|---|
| L17_matrixsimd B=1 | 15.223 µs (r2) | `512-bit+ymm tail, pinned sines` | `512-bit+ymm tail, C parked, pinned sines` |
| L8_batchsimd B=64 | 0.610 µs (r1) | `LANEX3 nt=0 pf=s0` | `LANEX2S nt=0 pf=s0` |

L17_matrixsimd `cmp`-verified all four mixed variants against their class representatives
on full output files at B=8 and B=256 before shipping them, and states the residual
exposure honestly (bit-identity is a gcc codegen property verified under SPR
`-march=native`, while the node compiles Cascade Lake). The same exposure existed in r4
and held. I do not believe the number is wrong, but the harness did not prove it.

**L8_batchsimd B=64 is worse than a bookkeeping note and should be read down.** The
reported 0.610 comes from the one run in three that picked `LANEX3`; the two runs that
picked its shipped default `LANEX2S` measured **0.660 and 0.655**. The entry's own record
says to "treat any single B=64 structure pick as noise." Read this cell as ≈0.655, which
places it **fourth** in the cell behind L8_fusedaxes (0.594), L8_radix8 (0.619) and
mkl_dfti (0.720) — not second. The three benign flips are L17_winograd B=1 (a8/b8/a8; the
min came from run 3, which is the checked run), L6_unrolled B=4096 (fused_pf/pfw/pf; min
from run 1, same variant as checked), and L8_fusedaxes B=16384, whose "flip" is only its
clock probe reading 3.26 vs 3.27 GHz.

**(b) Three headline cells rest on a min that is an outlier against their own other two
runs.** Minimum-of-three is the standing convention and I am not overriding it, but these
should not be read as confirmed:

* **L6_pfa B=64**: 0.226 / 0.226 / **0.217**. Reported −2.3% vs r4; typical run is +1.8%.
* **L6_unrolled B=1**: 0.227 / 0.227 / **0.220**. Reported 0.220; typical run is 0.227.
  L6_pfa's B=1 is 0.219 in all three runs (0.0% spread), so on the run distributions
  L6_pfa reproducibly owns the L=6 non-batched cell by ~3.5%, not the 0.5% the
  leaderboard shows. This matters for promotion and is used in §7.
* **L36_pencilfused B=1**: 134.058 / 125.565 / **121.255** — a monotone warm-up across
  processes, 10.6% spread. Its 2nd place at B=1 rests on the third run alone.

**(c) Bit-identity fingerprints, third round running.** The correctness JSONs are a
fingerprint: when `rel_l2`, `max_abs` and `rel_max` all agree to the last digit, two
entries are producing bit-identical output over millions of complex doubles.

* **L8_radix8 ≡ L8_batchsimd at B=1, B=2048 and B=16384** (e.g. `rel_l2` =
  1.9149126868005188e-16 at B=2048, identical in all 17 digits). Same radix-8
  split-complex arithmetic in the same axis order, two memory schedules — as in r4. **At
  B=64 they now differ** (radix8 2.287e-16 vs batchsimd 1.9155e-16), because radix8's
  B=64 pick is the `1f` shape it ported from L8_fusedaxes, which carries fusedaxes' y,x,z
  axis order. The fingerprint independently confirms that port is a real structural
  change and not a relabel.
* **L36_pfa ≡ L36_pencilfused at B=32 and B=256**, distinct at B=1 and B=4 — same as r4,
  but now explained rather than merely observed: pencilfused's new mode 7 (`istream`) is
  an acknowledged translation of pfa's `inplace pf=1`. This is *arithmetic-order*
  identity, not code duplication — a whitespace-insensitive diff of the two files differs
  on 1604 of 1686 lines, and their measured times differ by 10.9% at B=32.
* **All three L=17 entries remain numerically distinct** (3.274 / 3.114 / 3.305 e-16 at
  B=1). At L=17 the panel really is running three different algorithms, which is what
  LITERATURE §4.2 asks for. It is the only geometry where that is still true.

**(d) The clock probes disagree with each other, and it is not a minor discrepancy.** Five
independent probes shipped this round; see §5. Three read 3.89/2.89 GHz, one reads
2.89/2.89, one reads 3.27/2.43. The 512-bit figure is robust; the 256-bit figure is not,
and every cycle-per-volume number in this round's records depends on which is right.

---

## 4. Claimed numbers versus measured numbers

The development machine (`wallaby`, Xeon Gold 6448Y, Sapphire Rapids, two 512-bit FMA
units, 2 MB L2/core, 60 MB L3) is not the scoring machine (Gold 5218, Cascade Lake, one
512-bit FMA unit, 1 MB L2/core, 22 MB L3). The panel's standing calibration puts the full
wallaby-to-node span at ~2.9×, anchored on MKL measuring 80.7–150.6 µs on wallaby against
162.2 on the node for one L=36 B=1 case.

**That calibration now has a named cause, and it is the round's most useful methodology
result.** L17_winograd's dual-width probe reads **2.10 GHz in a contended wallaby window
and 4.10 GHz in an idle one** — the Gold 6448Y's base and max-turbo clocks exactly. So
wallaby's long-documented "bimodality" is a provable **1.95× clock swing**, not cache
noise, and a large share of the historical machine gap is the dev machine's clock
lottery. The record carries two concrete casualties: `a8` measured 19.069 µs with an sd of
**0.10%** in one window and 9.727 µs in another (a tight sd made the wrong number look
trustworthy), and a first `d8`-vs-`a8` A/B that read "−45%" and collapsed to +10% once
re-run same-window. Every wallaby table in rounds 1–4, from every entry, should be read
with this in mind, and same-window A/B is now the only defensible dev-machine statistic.

**Ratios of claimed (wallaby) to measured (node), same entry, same cell:**

| entry | cell | claimed | measured | ratio |
|---|---|---|---|---|
| L6_unrolled | B=1 / B=32768 | 0.108 / 0.249 | 0.220 / 0.566 | 2.04× / 2.27× |
| L6_pfa | B=1 / B=64 | 0.129 / 0.128 | 0.219 / 0.217 | 1.70× / 1.70× |
| L8_radix8 | B=1 / B=16384 | 0.308 / 0.604 | 0.570 / 1.402 | 1.85× / 2.32× |
| L8_batchsimd | B=1 / B=16384 | 0.305 / 0.597 | 0.574 / 1.388 | 1.88× / 2.32× |
| L8_fusedaxes | B=1 / B=16384 | 0.343 / 0.626 | 0.573 / 1.254 | 1.67× / 2.00× |
| L17_matrixsimd | B=1 / B=2048 | 8.64 / 13.22 | 15.223 / 21.983 | 1.76× / 1.66× |
| L17_rader | B=1 / B=2048 | 9.465 / 16.26 | 17.098 / 25.500 | 1.81× / 1.57× |
| L17_winograd | B=1 / B=256 | 9.31 / 11.21 | 18.177 / 23.933 | 1.95× / 2.13× |
| L36_pfa | B=1 / B=256 | 51.4 / 101.6 | 120.358 / 182.598 | 2.34× / 1.80× |
| L36_mixedradix | B=1 / B=256 | 51.59 / 101.2 | 122.755 / 215.882 | 2.38× / 2.13× |
| L36_pencilfused | B=1 / B=256 | 51.45 / 105.0 | 121.255 / 230.243 | 2.36× / 2.19× |

Every ratio is inside the machine band; none is an implementer error. Six specific claims
deserve to be scored on *direction*, three because they were right and three because they
were not.

**The three that were right, and they are the round's best work.**

1. **L17_matrixsimd's port-floor model predicted a node cell to within 0.2%.** From a
   static instruction count alone — mixed shape = 208 zmm + 35 ymm chunks = 33 374 cycles
   against the pure-512 floor of 35 964 — it predicted B=1 "16.43 → ~15.3 µs" and B=8
   "~16.7". Measured **15.223** and **16.658**. Its batched predictions (~20.5 / ~21.2)
   came in 3.4% / 3.7% optimistic, which is exactly the memory-bound component the model
   explicitly does not cover, and it said so in advance. This is the first time in five
   rounds that a panel entry has predicted a node number from first principles rather
   than scaled one from wallaby, and it is a direct dividend of having the clock.
2. **L36_pfa went four-for-four on its own stated ranges** — B=1 119–122 → 120.358;
   B=4 128–132 → 129.295; B=32 158–172 → 168.565; B=256 190–210 → **182.598**, better
   than its own optimistic bound. It also predicted the *pick* correctly in every cell,
   including the negative prediction that `pf=2` would be rejected at B=1/B=4 (where its
   own in-arena data measured prefetchw at +13%/+11% on cache-resident lines) and
   selected at B=32/B=256. The one thing it got wrong is in the panel's favour: it
   expected `inplace-pf2` at B=256 and pw=4; the node picked `pw=2 inplace pf=2`.
3. **L8_batchsimd beat all three of its batched predictions** (B=64 0.62–0.66 → 0.610;
   B=2048 1.13–1.18 → 1.096; B=16384 1.42–1.50 → 1.388). Its round was a single-mechanism
   round built entirely on reading r4's pick strings and attributing the whole
   1.215-vs-1.136 streaming gap to prefetch *placement*; the attribution was correct and
   slightly conservative. (Its B=64 number is separately unreliable — §3a.)

**The three whose direction failed.**

4. **L8_fusedaxes under-predicted its own headline by 20%, and the cause is a repeatable
   methodology error rather than the machine.** It predicted B=2048 1.13–1.20 and B=16384
   1.40–1.48, expecting the node to pick `seq3+pfs` and match L8_radix8's r4 numbers.
   The node picked **`fused+pfs+pfw`** and measured **0.910** and **1.254**. Its own
   record contains the number that predicted this — `plain+pfs+pfw 0.637 vs plain+pfs
   1.107` (pfw −42%) at wallaby B=5632 — but it also contains `fused+pfs+pfw 0.463 vs
   fused+pfs 0.449` at wallaby B=2048, and it generalised from the second. The error is
   that it compared wallaby B=2048 (0.53× wallaby's 60 MB L3) with node B=2048 (1.45× the
   node's 22 MB L3): those are different regimes. When the dev machine's L3 is 2.7× the
   node's, batch sizes must be mapped by L3-relative working set, not by B. Two entries
   (L8_radix8, L8_batchsimd) already ship L3-relative arena clamps for exactly this
   reason; this is the same lesson on the prediction side.
5. **L36_pencilfused's `istream` port was validated at parity on wallaby and landed 10.9%
   short on the node.** It predicted B=32 170–185 and B=256 210–225; measured **186.903**
   and **230.243**, both just outside. The prediction was anchored on a careful
   calibration: it built L36_pfa's r4 exemplar privately, forced it to `inplace pf=1`, and
   measured **pfa 157.1 vs its own mode 7 at 156.6** in the same wallaby window — parity,
   to 0.3%. On the node the same two structures measure 168.565 and 186.903. This is the
   most precisely controlled cross-machine transfer failure the project has recorded, and
   its lesson is sharp: **a structure transplant verified at parity on the development
   machine can be 11% short on the scoring machine.** Cross-entry ports must be
   re-validated on the node.
6. **L36_mixedradix's B=1 claim of "unchanged, 118–121"** measured 122.755 — outside the
   range, +3.1% against a 0.3% spread, with a pick string identical to r4's. Not a
   machine difference: it is a same-machine, same-pick, round-over-round regression in a
   path the entry believed it had not touched. See §2.

**Two conditional predictions resolved on their falsifying branch, and both were honest
about it in advance.** L6_unrolled wrote: "If B ≈ 3.5–3.9 [GHz], I expect `z2s_pf` or
`z3t` to take B=1 at 0.18–0.21 µs. If B ≤ 3.2, the margin plus the clock loss will keep
`fused` and the cell stays ≈0.219." Measured clk512 = 2.89 and the cell measured 0.219
with `fused`. L17_winograd wrote a symmetric fork for `d8`. Both entries got the value of
the round from the measurement rather than the kernel, which is the right way to spend a
round on a hypothesis you cannot test locally.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### The primary move: §4.8 gap 6 — AVX-512 on the scoring part

Five independent clock probes shipped on the node this round, from five different
implementers, all reporting through `fft3d_description()` into the leaderboard JSONs.
This is the first AVX-512 licence-clock data in the corpus.

| entry | probe design | clk256 | clk512 |
|---|---|---|---|
| L6_unrolled | 1 serially dependent FMA chain (~0.25 FMA/cy), after tournament | **3.89** | **2.89** |
| L17_matrixsimd | 4 parallel latency-4 chains (1 FMA/cy), end of `create()` | **3.89** | **2.89** |
| L17_rader | serial chain, best of 5, after tournament + settle | **3.89** | **2.89** |
| L17_winograd | 4 parallel chains per width, 256 issued *before* 512 | **2.89** | **2.89** |
| L8_fusedaxes | running max of 5 ms chunks, 100 ms stagnation stop | 3.27 | 2.43 |

**What is settled: `clk512` = 2.89 GHz.** Four of five probes, three independent
implementations, agree exactly. That is the AVX-512 licence clock of the Gold 5218 at one
active core on an exclusive node, and it matches Intel's Specification Update figure
(2.9 GHz, 1–8 active cores) quoted in §4.8 item 6. **The corpus's corrected 512-bit
number is confirmed by measurement.** Every cycle-per-volume figure in every strategy
record should now be re-derived at 2.89 GHz for zmm kernels; the numbers change
materially — L17_matrixsimd's B=1 is 44.0k cycles against its own 33.4k-cycle mixed-shape
floor (1.32×, not the 1.05× the record assumed), L17_winograd's is 52.5k against its 32k
floor (1.64×), L17_rader's is 49.4k against ~35.9k (1.38×), and L36_pfa's B=1 is 348k
against a ~241k-cycle single-FMA-port floor (1.44×).

**What is not settled, and it is the more consequential half: `clk256`.** Three probes
read 3.89, one reads 2.89, one reads 3.27. The disagreement is structured, not random —
the *ratio* clk512/clk256 is 0.743 for both L6_unrolled (2.89/3.89) and L8_fusedaxes
(2.43/3.27), to three digits — and the plausible reading is that the probes differ in
whether their 256-bit chain is dense enough to engage the AVX2 licence. L17_winograd's is
explicitly designed to saturate a single FMA unit at both widths and issues 256-bit
first so licence dwell cannot leak backwards; it reads 2.89/2.89, i.e. AVX2 and AVX-512
at the same clock, which is precisely Intel's table (non-AVX 3.9, AVX2 2.9, AVX-512 2.9).
The sparser chains would then be reading the non-AVX licence at 3.89. But
L17_matrixsimd's probe is also four parallel chains at 1 FMA/cycle and reads 3.89/2.89,
so probe density does not cleanly explain the split, and nobody has run the two probe
designs back to back in one process. **This must be closed next round; it is one process
and ~5 ms of work.**

**Why it matters more than the 512-bit number: it is the difference between L=6 being
nearly finished and L=6 having a large unexplained stall.** L=6's picked kernels are all
ymm. At 3.89 GHz its B=1 of 0.219 µs is 852 cycles against a 486-cycle two-FP-port floor
— 1.75×, the "366 missing cycles" the r4 verdict reopened the geometry over. At 2.89 GHz
the same measurement is **633 cycles, 1.30× the floor**, and the unexplained overhead
falls from 43% of runtime to 23%. Two implementers spent this round chasing that gap on
the 3.89 reading. It may be less than half the size they think it is.

**The panel's kernels answered the underlying question behaviourally, which is stronger
evidence than any probe, and the answer is not what §4.8 item 6 says.** That item's
correction concludes 512-bit is "strictly preferable on this part … at **zero** frequency
cost." The round tested that directly and the result splits:

* **Converting a 256-bit kernel to 512-bit loses.** L6_unrolled built eight zmm
  candidates with 15–27% fewer uops per volume and shipped them in a tournament that
  already prefers the simplest candidate. The node rejected **all eight, in all four
  cells, across all three processes**, keeping ymm. A 15–27% uop reduction losing to its
  ymm twin is what a ~26% licence penalty predicts and is the opposite of what "zero
  frequency cost" predicts.
* **Mixing ymm into a kernel that is already 512-bit wins, and handsomely.** At L=17,
  where every kernel is zmm and the core is already in the AVX-512 licence, the mixed
  512+ymm-tail shape was selected in every cell for the second entry running and paid
  **−7.4% at B=1** for L17_matrixsimd, on top of the −2.5…−5.2% it paid L17_rader in r4.
  Once you are inside the licence, the second 256-bit port is free throughput.

**The synthesis, which the corpus should absorb: on the Gold 5218 the cost is the licence
transition, not the width.** §4.8 item 6's blanket "512-bit is strictly preferable at zero
frequency cost" is refuted as stated; the correct rule is *do not enter 512-bit for a
kernel that does not need it, and once you are in it, mix widths freely.* That is a
measured, two-geometry, two-sign result and it is the round's contribution to the corpus.

### Also moved: §4.5 — padding, store policy and the RFO

The round produced a clean answer to the store-side question, at two geometries, from two
independent implementations, in the same direction:

* **L8_fusedaxes**: `fused+pfs+pfw` selected 3/3 in both streaming cells; B=2048
  1.313 → 0.910, B=16384 1.585 → 1.254. First selection of write-intent prefetch at L=8.
* **L36_pfa**: `pf=2` (paced `prefetchw` on the phase-1 in-place store stream) selected
  3/3 at B=32 and B=256; B=256 218.899 → 182.598. Its in-arena decomposition attributes
  it precisely — `inplace-pf2 90.5 vs inplace-pf1 156.6` (−42%) — "the phase-1 RFO was
  indeed the dominant exposed cost of the mode the node runs."
* Meanwhile **non-temporal stores lost on the node for the fourth consecutive round**, at
  every geometry, in every streaming cell, in every entry's own tournament.

So the rule for Cascade Lake, now measured rather than argued: **hide the RFO
(`prefetchw`) rather than avoid it (NT stores)**. Both entries also independently
confirmed the converse — prefetchw on cache-resident lines is pure µop tax (L36_pfa
+13%/+11% in-arena at B=1/B=4; L8_fusedaxes +3% at an L3-resident wallaby cell) — so the
mechanism belongs in a gated tournament, never as a default. §4.5's actual open question
(measure L=8 at (8,8) vs (9,9) padding, and check
`ld_blocks_partial.address_alias`) is still untouched by anyone.

### Also moved, and reversed: §4.3 — is axis fusion worth 3× or 3%?

The r4 verdict recorded a tidy answer at L=8: fusion wins where the volume is
cache-resident and loses where the batch streams, which is Tolmachev's rule behaving. **r5
overturns the second half.** The single fused pass now wins *all three* batched cells
including B=16384 at a 256 MiB working set, and by 11–20%. Two independent confirmations:
L8_fusedaxes won them outright, and L8_radix8's B=64 cell fell 0.680 → 0.619 on the `1f`
fused shape it ported from fusedaxes.

**But the experiment is confounded and the panel should not bank it.** L8_fusedaxes
changed pass count *and* store policy in the same round: its r4 B=2048 pick was `seq3`
(three passes) and its r5 pick is `fused+pfs+pfw`. The three-pass entries that beat it in
r4 still have no pfw candidate. So the −20% could be fusion, could be pfw, could be their
interaction, and nothing in this round separates them. §6 makes that the L=8 priority.

### §4.6 — model versus search: the search protocol itself is now the thing being debugged

Three separate tuner-protocol bugs were found and fixed this round, each worth more than
the kernel work in the same record:

* **L36_pencilfused: predecessor-state poisoning.** With `inner=1`, each candidate
  inherits the previous candidate's cache state; `istream` ran immediately after an
  NT-flushing mode and measured **167.4 µs/vol in-arena against a true 89.8** — an 86%
  phantom penalty that would have buried the round's main candidate. Fixed with one
  untimed warm exec per candidate. **Any in-arena ranking in any record where a candidate
  followed an NT mode carries this taint**, which is a panel-wide retraction.
* **L17_rader: rank-order handicap on a ramping clock.** Without a settle spin, `xl 512t`
  timed 20.96 µs/t ranked third and 11.88 two slots later — bit-identical work, **76%
  apart, purely table order**. Adding a ~150 ms settle collapsed the table to 11.5–12.1,
  and this alone is what produced its −3.6% at B=1.
* **L17_winograd: the clock lottery** (§4), which makes any cross-window dev-machine A/B
  meaningless regardless of its standard deviation.

Search still wins over model, but for the third round running the binding constraint is
neither the model nor the search — it is whether the measurement inside the search is
honest.

---

## 6. The single highest-value thing the next round should attack, per geometry

**L = 6 — settle `clk256`, then profile. Stop shipping kernels.** Both hypotheses for
B=1's overhead have now been falsified *on the node*: port-5/uop count (L6_unrolled's
eight zmm candidates, 15–27% fewer uops, zero picks in four cells) and out-of-order
window pressure (L6_pfa's `fused_sp`/`fused_sp2`, rejected at the cell they were built
for). B=1 has read 0.219–0.220 for four rounds and the geometry is 1.71× the best library.
Two measurements, both cheap, both already requested by the implementers and neither of
which needs an implementer round: **(i)** run L17_winograd's saturating dual-width probe
and L6_unrolled's sparse-chain probe back to back in one process on the node and settle
whether the panel's ymm kernels run at 3.89 or 2.89 GHz — this alone decides whether the
unexplained overhead at B=1 is 366 cycles or 147, i.e. whether there is anything left to
find; **(ii)** `perf stat -e uops_issued.any,cycles,resource_stalls.rob,resource_stalls.rs,cycle_activity.stalls_total`
on forced `L6_FORCE=fused` vs `fused_sp` vs `z2s` at B=1. Both entries shipped variant
forcing switches specifically so the monitor could do this. Without it, round 6 at L=6 is
a third round of guessing.

**L = 8 — separate fusion from `pfw` before anyone optimises the wrong variable.** The
round's headline (0.910 at B=2048, −19.9%) moved two variables at once, and two of the
three L=8 entries are currently tuning three-pass structures against a fused winner
without knowing which change earned the win. The isolating experiment is four forced
runs and no new code: on L8_fusedaxes, `fused+pfs` (no pfw) and `seq3+pfs+pfw`, via the
`-DL8_VARIANT=0..9` switch it already ships; and add a `pfw` candidate to L8_radix8's
`3p-pfs` and L8_batchsimd's `LANEX3+s0`. If pfw carries it, the three-pass entries
recover most of the gap and §4.3's r4 answer stands; if fusion carries it, §4.3 flips at
L=8 and the panel should fuse everywhere. Secondary, and now much cheaper than it looks:
the B=64 "L2 cliff" narrowed from +9.3% over B=1 to +4.2% this round without anyone
blocking the batch loop, so the remaining cliff may not be capacity at all — and
§4.5's `ld_blocks_partial.address_alias` counter, which L=8's 8192-byte volume stride
makes maximally degenerate, has still never been read.

**L = 17 — force `ov` and `d8` on the node and get their numbers.** B=1 is now 15.223 µs
= 44.0k cycles at the measured 2.89 GHz against L17_matrixsimd's 33.4k-cycle mixed-shape
floor: **1.32× the floor, ~10.6k cycles/volume of non-FP time**, quantified correctly for
the first time. All three records independently locate the same suspect — the per-plane
deinterleave and transpose traffic that runs serialized between kernel blocks (rader
estimates ~19k µops/volume; winograd ~120 port-5 uops per group). Both entries built a
mechanism to attack it (rader's `ov`, which emits the shuffles inside the zmm drain;
winograd's `d8`, which defers the transposed store one group late) and the node tuner
declined both in all four cells — but **neither was ever measured forced on the node,
only offered against an incumbent inside a 3% hysteresis band.** A tuner declining a
candidate is not the same as a candidate losing, and the difference here is a ~10.6k-cycle
prize. Force both, publish the numbers. If both genuinely lose, the transposes are not
where the time is and the next suspect is the kernels' own ~90 constant loads plus ~60
stack moves per block, which nobody has costed.

**L = 36 — explain the 10.9%, and do it by diffing two files rather than writing a
third.** All three entries now run L36_pfa's in-place + paced-prefetch structure at
streaming batch. The two ports land 5.4% (mixedradix at B=32) and 10.9% (pencilfused at
B=32) behind the original *on the node*, while pencilfused's careful same-window
calibration measured its port at **parity** with pfa's own exemplar on wallaby (156.6 vs
157.1). Same structure, same machine, 11% apart, with both source files in hand and the
candidate differences already enumerated by the implementer (PFNX depth, phase-2 prefetch
pattern, pass-A ordering, plane-scratch handling). That is the most precisely localised
unexplained number on the whole board and it is cheap to close. Do it before adding any
mechanism. Second, named independently by two records and untouched since round 1: B=1 is
348k cycles at 2.89 GHz against a ~241k-cycle single-512-bit-FMA port floor — 1.44× — and
the only untried compute-side lever, software-pipelining two line groups so the DFT3
latency chains overlap across groups, has been on somebody's "Next" list for five rounds
and has never been measured by anyone. Third, for the monitor rather than an implementer:
re-measure the MKL L=36 batched baselines (§1).

---

## 7. Promotion

Against `docs/CURATION.md`, in its stated order.

**1. Fastest correct entry per geometry.**

* **L = 6 — both, because the geometry splits cleanly by regime and the split is
  reproducible.** `L6_pfa` owns B=1 (0.219 in all three runs, 0.0% spread, against
  L6_unrolled's 0.227/0.227/0.220) and B=4096 (0.391–0.394 against 0.397–0.407);
  `L6_unrolled` owns B=64 and B=32768 in every run. The brief scores non-batched and
  batched separately and this is a 2–2 split on the run distributions, not on a single
  minimum. This reverses r4's judgement that L6_pfa is a near-duplicate not worth
  promoting; the reversal is on the numbers (it reproducibly holds the non-batched cell)
  and on content (its `fused_sp`/`fused_sp2` double-buffered *plane* pipeline is a
  mechanism L6_unrolled does not have, and it is distinct from the cross-*volume*
  pipelining the r4 verdict closed — the node selected it at B=64, the only new mechanism
  selected at L=6 all round). Its B=64 number is not a confirmed gain (§3b) and the
  promotion note should say so.
* **L = 8 — `L8_fusedaxes`**, fastest in three of four cells including all three batched
  ones, and by 18% at B=2048. Plus **`L8_radix8`** under rule 1 for the non-batched cell
  (0.570, fastest by 0.5% in a three-way tie) and under rule 2 below.
* **L = 17 — `L17_matrixsimd`**, fastest in all four cells for the fourth round running,
  now with the round's only first-principles prediction to its name.
* **L = 36 — `L36_pfa`**, fastest in all four cells. The r4 regime split is gone; it took
  B=1 and B=4 back and extended B=256 by 16.6%.

**2. Structurally different runner-up, close behind.**

* **`L17_rader`** — 1.12× the leader at B=1 and B=8, second in both, and *verified*
  structurally different: its output fingerprint (3.114e-16) differs from both rivals', so
  the panel really is running three algorithms at L=17. It also carries the round's
  clock-settle-spin finding (76% mis-ranking from table order alone), which every entry
  with a fixed-order tournament needs to read.
* **`L8_radix8`** — the three-pass arm of the §4.3 axis-fusion question that
  L8_fusedaxes' single fused pass sits opposite, fastest at B=1, and the carrier of the
  `1f` cross-entry port whose distinct output fingerprint at B=64 proves the transfer was
  real. Keeping both arms is what makes §6's isolating experiment possible from the
  exemplars alone.

**3. Instructive failures.**

* **`L36_pencilfused`** — promoted again, for a better reason than r4's. Its record now
  contains the cleanest cross-machine transfer failure the project has: it built the node
  winner's exemplar privately, forced it to the winning configuration, measured its own
  translation at **parity on wallaby (156.6 vs 157.1)**, and landed **10.9% behind on the
  node**. That single pair of numbers is worth a round to anyone about to transplant a
  structure between entries. It also carries the predecessor-state poisoning diagnosis
  (167.4 in-arena vs 89.8 true, an 86% phantom penalty) and the self-warming fix, which
  retroactively taints in-arena rankings across several entries' records.

**4. Anything that beat a library.** Selects all 11 entries (§1); not used.

**Not promoted, with reasons.**

* **`L8_batchsimd`** — produces **bit-identical output to L8_radix8 at three of four
  batch sizes** for the third round running, its entire round was adopting L8_radix8's
  spread prefetch, and its one nominally-second-place cell (B=64, 0.610) is an unstable
  pick whose honest value is ≈0.655 and fourth place (§3a). Its instructive datum — that
  reading r4's pick strings and attributing a 7% streaming gap to prefetch *placement*
  alone was correct and slightly conservative, beating all three of its own predictions —
  is recorded here in §4 instead.
* **`L17_winograd`** — already in `exemplars/panel_r2` and `panel_r3`, flat on the node
  (−0.8% at B=1), its `d8`/`e8` deferral selected in zero cells, and its 17-point module
  survives inside the promoted `L17_rader`. Its two real contributions are measurements,
  not code, and they are recorded in §4 and §5: the proof that wallaby swings 2.10 ↔ 4.10
  GHz between sessions (with the `a8` 19.069-µs-at-0.10%-sd casualty), and the only
  saturating dual-width clock probe on the board, whose 2.89/2.89 reading is the crux of
  the unresolved `clk256` question.
* **`L36_mixedradix`** — second in three cells but **not structurally different** (PFA 4×9
  two-sweep, and this round it explicitly ported L36_pfa's in-place path), so rule 2 does
  not select it; a third near-identical PFA 4×9 file adds nothing to the reading list that
  `L36_pfa` and the structurally distinct `L36_pencilfused` do not already carry. It also
  produced the round's only cell-level regression (B=1, +3.1%, identical pick string).
  Its genuine result — that when its tuner was finally given both store policies in one
  tournament it independently chose *cached* over NT on a second pass structure,
  reproducing L36_pfa's r4 finding — is recorded in §5 and is the reason §4.5's answer
  can be stated as a rule rather than as one entry's observation.

---

## Provenance note

Round 5's sources are in `bench/geom/impl/`, which is again a plain directory rather than
the `impl → impl_N` symlink `run_rounds.sh` creates; `impl_4/` does not exist, so r4's
code is preserved only in `exemplars/panel_r4/`. The same exposure applies to this round.
Promote and commit before round 6 starts, and check that `setup_impl_dir()` actually
migrates `impl/` to `impl_5/`.

PROMOTE: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L17_matrixsimd L17_rader L36_pfa L36_pencilfused
