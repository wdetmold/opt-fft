# Round ice_r7 — monitor's verdict

Scored on `a80n0.lqcd.mit`, slurm job 438637, 2026-08-23T12:59:51-04:00.
19 panel entries + `baseline_matrix` + 6 library backends, 8 geometries, 3 independent
processes each, 12 samples per process. Ranking metric is min-over-processes of each
process's min sample.

---

## 0. Three corrections to the brief, before any number is read

These are not new. The ice_r5 and ice_r6 monitors each raised 0.1 and 0.2; ice_r6 raised
0.3. Nothing was fixed between rounds, so they are restated here with this round's evidence
and then not belaboured.

**0.1 The scoring machine is not a Cascade Lake Gold 5218. Third round of this correction.**
`environment.txt` reads `Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz`, and the ISA string
carries `avx512_vbmi avx512ifma avx512_bitalg avx512_vpopcntdq`. Those four are Ice Lake-SP
(Sunny Cove); Cascade Lake has none of them. This is a dual Gold 6326 (2 × 16C/32T = the 64
reported logical cores): **1.25 MiB L2 per core, not 1 MiB, and two 512-bit FMA pipes, not
one.** The round tag `ice` is right; the brief and `docs/LITERATURE.md` §4.8 item 6 are the
stale artifacts. §4.8 item 6's own closing instruction — "there is no primary measurement in
the corpus for Ice Lake-SP or later *server* parts … **Measure it on the node**" — has now
been carried out seven times by this panel and the corpus still has not been updated.

**0.2 There is no Sapphire Rapids development machine, so item 4 of the brief has no
subject.** All 19 strategy records state their measurements were taken on `a80n0` — the
scoring node — through `tryout.sh` or a hand-driven core lease. "Sapphire" appears once in
19 records, in `L6_unrolled.md`, and not as a dev host. There is no cross-machine transfer
in this round and no MKL spanning 2.9×: same-window MKL controls quoted inside the records
read 0.94 / 2.10 / 89.0 / 286.8 µs at L = 6/8/17/36 against scored MKL rows of 0.939 /
2.112 / 88.832 / 283.287 — agreement to about 1%. §4 below is therefore a real
claim-versus-measured audit rather than a machine-difference apology.

**0.3 This round contains no non-batched measurement. Seventh round running.**
`cases.txt` — "THE GRADED CONFIGURATION: `<L>:<batch>:<chain>`. Fixed, identical for
solver/base/SOTA" — has eight rows, `6:64:4856` through `64:2:134`, and no `B=1` row.
`sweep.sh` iterates exactly that file, so every `t_*.json` in this directory carries one
batch per L. `grep -c non-batched` over `results/ice_r*/leaderboard.txt` returns 0 for all
seven rounds. `fft3d_api.h` says batched and non-batched "are separate measurements"; the
ice series has never measured the second one.

**This gap is now materially worse than it was, because of what the round did.** The
round's dominant lever (§5) puts *volumes* in the SIMD lanes and requires B ≥ 4 or B ≥ 8;
below that threshold every entry that adopted it falls back to its previous engine. §1
therefore gives the scored batched figure and, separately and **clearly labelled unscored**,
the implementers' own B=1 readings — and at L = 6 and L = 17 those show the round's headline
gain does not exist at B=1 at all. Adding the `B=1` rows to `cases.txt` is a one-line change
and it is now the highest-value harness item after §3.1.

---

## 1. Headline per geometry

µs per transform, min over three processes. Ratio is best library ÷ best panel entry.

### L = 6 (V = 216, B = 64, m = 4856)

| | batched B=64 (scored) | non-batched B=1 (**unscored**, implementer's own) |
|---|---|---|
| best panel | **`L6_unrolled` 0.238 µs** (35.19 GF/s) | 0.330 µs (`L6_unrolled`, 3.30-mode) |
| best library | `mkl_dfti` 0.939 µs (8.92 GF/s) | — not measured — |
| ratio | **3.95×** | — |

Runner-up `L6_pfa` 0.320 (1.34×). **This cell is not validly gated — see §3.1.** The B=1
figure is 39% worse than the batched one: the SoA-8 group chain that won the cell needs
eight volumes.

### L = 8 (V = 512, B = 64, m = 2572)

| | batched B=64 (scored) | non-batched B=1 (**unscored**) |
|---|---|---|
| best panel | **`L8_fusedaxes` 0.511 µs** (45.10 GF/s) | 0.559 µs (`L8_fusedaxes`) |
| best library | `mkl_dfti` 2.112 µs (10.91 GF/s) | — not measured — |
| ratio | **4.13×** | — |

`L8_radix8` 0.551 (1.08×), `L8_batchsimd` 0.555 (1.09×). Highest GF/s on the board.

### L = 17 (V = 4913, B = 32, m = 98)

| | batched B=32 (scored) | non-batched B=1 (**unscored**) |
|---|---|---|
| best panel | **`L17_matrixsimd` 9.035 µs** (33.34 GF/s) | 11.876 µs (`L17_matrixsimd`, v6 fallback) |
| best library | `ducc0_c2c` 86.235 µs (3.49 GF/s) | — not measured — |
| ratio | **9.54×** | — |

`L17_winograd` 10.922 (1.21×), `L17_rader` 11.919 (1.32×). Largest library margin on the
board. The B=1 path is 31% slower than the batched one and is *literally the r6 code*:
`L17_matrixsimd`'s chain v7 packs 4 volumes per zmm, and "1-volume SoA is meaningless — B=1
stays v6." The geometry's 24% gain is a batched-only gain.

### L = 36 (V = 46656, B = 8, m = 64)

| | batched B=8 (scored) | non-batched B=1 (**unscored**) |
|---|---|---|
| best panel | **`L36_mixedradix` 99.809 µs** (36.25 GF/s) | 100.259 µs (`L36_mixedradix`) |
| best library | `mkl_dfti` 283.287 µs (12.77 GF/s) | — not measured — |
| ratio | **2.84×** | — |

`L36_pfa` 100.296 (1.00×) — **a tie, not a win: 0.5% apart**, and the winner's leaderboard
row carries a 7.2% inter-process spread. `L36_pencilfused` 107.009 (1.07×). Smallest library
margin on the board apart from L=64.

### The other four geometries, for completeness

| L | best panel | best library | ratio | runner-up |
|---|---|---|---|---|
| 13 | `L13_direct` 3.471 | `mkl2026_dfti` 11.798 | **3.40×** | `L13_rader` 3.688 (1.06×) |
| 23 | `L23_matrixsimd` 33.876 | `ducc0_c2c` 249.083 | **7.35×** | `L23_rader` 35.919 (1.06×) |
| 45 | `L45_mixedradix` 225.571 | `mkl_dfti` 757.588 | **3.36×** | `L45_pfa` 231.220 (1.03×) |
| 64 | `L64_blocked` 608.213 | `mkl_dfti` 1722.819 | **2.83×** | `L64_radix8` 614.640 (1.01×) |

**Series total** (sum of the eight graded per-call times, best panel entry per cell):
**0.7916 s**, against 0.9390 s in ice_r6 (**−15.7%**) and 3.1932 s for the best library at
each cell (**4.03×**). Every one of the 19 entries beats every library at its own geometry,
so CURATION criterion 4 does not discriminate this round.

---

## 2. What changed since ice_r6, per geometry

The libraries are the control and they reproduce: `mkl_dfti` moved 0.940→0.939, 2.120→2.112,
88.825→88.832, 284.283→283.287 µs at L = 6/8/17/36 — under 0.4% at every cell, and the same
holds at 13/23/45/64. **The panel deltas below are code, not machine drift.**

| L | entry | r6 | r7 | Δ |
|---|---|---|---|---|
| 6 | `L6_unrolled` | 0.291 | **0.238** | **−18.2%** |
| 6 | `L6_pfa` | 0.296 | 0.320 | **+8.1%** |
| 8 | `L8_fusedaxes` | 0.555 | **0.511** | −7.9% |
| 8 | `L8_radix8` | 0.569 | 0.551 | −3.2% |
| 8 | `L8_batchsimd` | 0.575 | 0.555 | −3.5% |
| 13 | `L13_direct` | 5.287 | **3.471** | **−34.3%** |
| 13 | `L13_rader` | 5.377 | 3.688 | **−31.4%** |
| 17 | `L17_matrixsimd` | 11.935 | **9.035** | **−24.3%** |
| 17 | `L17_winograd` | 11.649 | 10.922 | −6.2% |
| 17 | `L17_rader` | 12.284 | 11.919 | −3.0% |
| 23 | `L23_matrixsimd` | 35.888 | **33.876** | −5.6% |
| 23 | `L23_rader` | 37.172 | 35.919 | −3.4% |
| 36 | `L36_mixedradix` | 100.801 | **99.809** | −1.0% |
| 36 | `L36_pfa` | 106.249 | 100.296 | −5.6% |
| 36 | `L36_pencilfused` | 106.908 | 107.009 | **+0.1%** |
| 45 | `L45_mixedradix` | 264.977 | **225.571** | −14.9% |
| 45 | `L45_pfa` | 264.111 | 231.220 | −12.5% |
| 64 | `L64_blocked` | 638.757 | **608.213** | −4.8% |
| 64 | `L64_radix8` | 660.301 | 614.640 | −6.9% |

**Two cells changed hands.** L=17 went `L17_winograd` → `L17_matrixsimd` (the largest lead
change of the series, 1.02× behind to 1.21× ahead). L=45 went `L45_pfa` → `L45_mixedradix`,
which had been 0.3% behind and is now 2.5% ahead.

### Did anything regress?

**Two entries got slower, and only one of them is a regression.**

* **`L6_pfa`, +8.1% (0.296 → 0.320). Not a regression — a mandated accuracy purchase, and
  the price it quoted is the price it paid.** The r7 brief made the exact/Heron map tier
  policy at m=4856 after r6's gate rejected both fast tiers. `L6_pfa` flipped its default
  from the fast in-place arm to the Heron arm and recorded the cost as "+8.8% over fast
  (hip 0.3659 vs fip 0.3364)". The scored delta is +8.1%. **The comparison that matters is
  that `L6_unrolled` was handed the identical mandate and came out 18% faster anyway**, by
  spending the round on a structural change (§5) rather than absorbing the tier premium.
  That is the cleanest controlled pairing in the round.
* **`L36_pencilfused`, +0.1% (106.908 → 107.009). A genuine flat round, honestly reported.**
  Its record states plainly that "the shipped bits are r6's". Its one structural bet —
  fused-boundary custody, adopted from `L64_radix8`'s ckind=2 — was built, passed both
  correctness gates, and was **rejected by the node at 111.5–111.8 vs 108.2–108.5 µs/step,
  2/2 alternations, +3.0%**. That is a well-documented cross-geometry negative (the L=64
  lever does not transfer to L=36) and it is worth what the round cost, but the entry
  shipped no gain.

**Nothing else regressed. Nothing lost accuracy except `L17_winograd`, deliberately — §3.4.**

---

## 3. Adversarial pass: correctness, builds, crashes, omissions

### 3.1 Every backend at L=6 failed the chain gate, and the leaderboard printed "ok" for all nine. Second round running.

This is the round's most serious finding and it is a repeat.

`results/ice_r7/check.log` lines 1–18 record nine consecutive `FAIL map-chain m=4856`
verdicts at L=6 — one for every backend that ran the cell, including all six libraries and
the harness's own O(L⁴) `baseline_matrix` reference:

| backend | chain rel_l2 | gate (`1e-13·m` = 4.86e-10) |
|---|---|---|
| `L6_pfa` | 1.036e-9 | **FAIL** (2.1× over) |
| `L6_unrolled` | 1.322e-9 | **FAIL** (2.7×) |
| `mkl2026_dfti` | 1.692e-9 | **FAIL** (3.5×) |
| `ducc0_c2c` | 2.103e-9 | **FAIL** (4.3×) |
| `fftw3_estimate/measure/patient` | 2.145e-9 | **FAIL** (4.4×) |
| `mkl_dfti` | 3.108e-9 | **FAIL** (6.4×) |
| `baseline_matrix` | 2.192e-8 | **FAIL** (45×) |

Yet the L=6 block of `leaderboard.txt` prints `ok 2.4e-16` for all nine and ranks them.
The mechanism is a two-line defect and both lines are still exactly as ice_r6 described
them:

1. `check.py:67` computes `ok = ok and chain_ok` but **never writes the result back**;
   `result["ok"]` was fixed at line 41, before the chain check ran. The JSON therefore
   says `{"ok": true, … "chain_ok": false}` — verifiable in any `c_*_L6_B64.json`.
2. `leaderboard.py` reads only `chk["ok"]` and never looks at `chain_ok`.

And the ranked number *is* the chain: per-transform is `per_call/(B·m)`, so at L=6 the
leaderboard is ranking 4856-step chains whose end states it has been told are wrong.

**Two things make this the harness's fault and not the panel's, and both should be recorded
before anyone reaches for a disqualification.** First, the two panel entries are the *two
most accurate backends in the cell* — more accurate than every library and 17–21× more
accurate than the reference floor. Second, `L6_unrolled`'s record contains the mechanism, in
numbers, from pure-numpy experiments with no C anywhere: drift grows linearly (~4e-16·m) to
m ≈ 2500 and then **exponentially with a ~450-step e-fold** (2.5e-11 → 2.9e-10 → 1.4e-9 at
m = 2572 → 3600 → 4856). `e^(4856/450) ≈ 4.8e4` reproduces the brief's own "1-ulp ends at
4.8e-12" conditioning constant. The gate's `1e-13·m` is the wrong *functional form* for
L=6, not merely the wrong constant, and it anchors to pocketfft's and glibc-hypot's specific
bit patterns rather than to accuracy: substituting an 80-bit correctly-rounded `|z|` — more
accurate than numpy's own — drifts 1.56e-9 and also fails.

The cheat the gate exists to catch is still caught by every proposed replacement: a
float-seeded Newton map's 1e-12/application bias explodes to 1.2e-8+ under the same
amplification, four orders clear of the exact arms at any mid-chain checkpoint m ≤ 2572.

**The L=6 cell in §1 is reported as measured and should be treated as provisional until the
gate is fixed.** I am not disqualifying either entry: the reference implementation fails the
same gate by 45×, and an entry cannot be required to beat a bar that the definition of
correct does not clear.

### 3.2 Nothing failed to build, nothing crashed, nothing is missing

* `agents/exits.txt`: 19 lines, all `exit=0`.
* `build_errors.txt` exists but contains **no errors** — one `-Wmacro-redefined` warning in
  `L36_mixedradix.c` (`MAPPAIR_D` redefined at line 1175 against line 1203). Cosmetic; the
  entry built and scored. Worth cleaning so that a real error is not lost in the noise next
  round, since `sweep.sh` only greps this file for `error`.
* **`failures.txt` does not exist** — `sweep.sh` writes it only on a non-zero, non-3 exit,
  so its absence means no backend crashed, hung or hit the 600 s timeout.
* `timing.err` contains 405 lines, all of the form `<entry>: does not support L=<n>`. Every
  one is a geometry-specialist correctly declining a cell it does not claim; `leaderboard.py`
  skips `supported: false` records. No entry declined its own geometry.
* All 19 entries appear in `impl_7/`, all 19 have a strategy record with an
  `## Round ice_r7` section, and all 19 appear on the board.

### 3.3 Every entry passes the single-transform gate; 17 of 19 pass the chain gate

Single-transform `rel_l2` spans 2.27e-16 to 4.46e-16 against `tol 1e-12` — six to seven
orders of margin, tighter than `baseline_matrix` everywhere. The two chain failures are
`L6_pfa` and `L6_unrolled`, both under §3.1, both alongside the entire library field.

Three bit-identity facts, from full-precision JSON rather than the leaderboard's one digit:

* **`L8_radix8` and `L8_batchsimd` are bit-identical**, both `2.266437449855183e-16` single
  and `3.726217514247466e-12` chain — all 16 digits. `L8_fusedaxes` matches on the single
  transform and diverges on the chain (2.239e-12), so it has changed chain reassociation
  only. ice_r6 found all three identical; **the geometry has recovered one bit of structural
  diversity and lost none.** It bears on curation: the L=8 runner-up choice is between two
  binaries that produce the same bits.
* All three L=36 entries are single-transform bit-identical (`3.5858992906413332e-16`) and
  differ only in chain reassociation (1.265–1.300e-14).
* `L64_blocked` and `L64_radix8` are single-transform bit-identical
  (`4.456896580611933e-16`) and their chains agree to four digits.

### 3.4 `L17_winograd` degraded its chain accuracy 93× in one round, and lost the cell anyway

This is the round's one accuracy movement in the wrong direction and it deserves naming.

| | ice_r6 | ice_r7 |
|---|---|---|
| `L17_winograd` chain rel_l2 (m=98) | 1.726e-14 | **1.612e-12** |
| `L17_matrixsimd` | 1.892e-14 | 1.529e-14 |
| `L17_rader` | — | 1.529e-14 |

The cause is in its own record: the map's two Newton steps (7 ops, 2⁻⁵⁶) were replaced by
**one Halley step** (5 ops, 2⁻⁴²), plus a cubic step for the reciprocal. The entry did this
with eyes open, quoted a per-application error of ~5.7e-13, and predicted an end state of
1.039e-12 at "9.4× margin".

Three things to hold it to. **(a) The prediction was 55% optimistic**: measured 1.612e-12,
so the real margin is **6.1×, not 9.4×** — the thinnest on the board and thinner than the
L=64 pair. The gap is seed sensitivity, which is exactly the failure mode §3.1 documents at
L=6, so a 6.1× margin should not be read as comfortable. **(b) It bought this with ~2 ops
per 8 points and finished 21% behind** `L17_matrixsimd`, which is exact at 1.53e-14 and
faster. The trade did not pay. **(c) It is the only entry on the board that got materially
less accurate this round.** It passes the gate and is not disqualified, but the next round
should be told to reclaim the Newton tier — its own record already notes the exact arm is
kept and costs little, "take it if a future round ever needs the margin back". It needs it
back.

### 3.5 The L=64 pair still has no accurate arm, and the r6 warning stands unaddressed

`L64_blocked` 1.624e-12 and `L64_radix8` 1.624e-12 against a 1.34e-11 budget: **8.2×
margin, and 57× worse than `mkl_dfti`'s 2.86e-14 on the identical case.** Both improved
slightly from r6's 1.93e-12, so the trend is not worsening, but ice_r6 §3.4's finding is
unchanged: `L64_radix8` adopted `L64_blocked`'s cubic reciprocal ladder in r6 and the
geometry has had no conservative arm since. **If the L=64 chain is ever lengthened, both
entries fail together.** `L64_radix8` retains `-DFFT64R_MAPEXACT` and gates the tier at
create time, which is the right engineering; the board-level exposure is still real.

### 3.6 Two spread anomalies, one of them worth a re-run and neither of them kernel instability

The leaderboard's "run spread" column is inter-process — `(max−min)/min` over three
processes' minima — so it does not distinguish a noisy kernel from one disturbed process.
Per-run data separates them:

* **`L17_winograd`, 17.1%: one bad process, not a noisy kernel.** r1 34252.7, r2 **40102.8**,
  r3 34299.7 µs, and *within* each run the sd is 0.02–0.05%. Run 2 ran 17% slow at a
  perfectly steady state for all 12 samples. The min-of-mins metric discarded it correctly,
  but a whole process settling into a 17%-slower steady state is a reproducibility hazard,
  not a scheduling blip. `L45_mixedradix` (3.2%) and `L13_direct` (4.0% within r2) show the
  same signature more weakly. Related and probably the same cause: the entries self-probe the
  core clock at setup and **disagree across processes of the same binary** —
  `L17_winograd` reported `clk256/clk512 = 2.90/2.90` in runs 1 and 2 and `3.50/3.30` in
  run 3; `L36_pfa` reported `clk=3.50, 2.90, 3.50`; `L6_unrolled` `kclk=3.30, 2.90, 3.30`.
  Entries that select code variants off that probe are selecting off a coin flip.
* **`L64_blocked`: recurring intra-run tail spikes in all three processes.** min/max per run
  are 163001/190887, 163965/184766, 163739/192478 µs — sd 3.7–6.4% *inside* every run, with
  a stable floor. That is a periodic stall in the kernel's own steady state, not a co-tenant,
  and it is the only entry on the board with this shape. It does not affect the ranking
  (min-of-mins) but it is ~17% of tail latency that nobody has looked at.
* Minor: `L36_pfa`'s leaderboard number (51351.5 µs) comes from run 2, whose *median* was
  64363.6 — a lucky minimum inside a badly disturbed process. It is only 0.2% below the two
  clean runs' minima, so the ranking is unaffected, but the L=36 top two being 0.5% apart
  with this kind of noise underneath means **L=36 is a tie and should be reported as one.**

### 3.7 The round's seed is still not recorded

`environment.txt`, `leaderboard.txt` and `sweep.out` contain no `--seed` value — same as
ice_r6. `CURATION.md` justifies discarding the several hundred raw `t_*`/`c_*` files on the
grounds that "a round is re-runnable from its seed"; that guarantee remains unbacked. It is
not academic: this round's L=6 chain drifts are 5–8× *lower* than ice_r6's for the identical
backends on the identical machine (`mkl_dfti` 8.84e-9 → 3.11e-9, FFTW 7.95e-9 → 2.15e-9),
purely from the seed change. §3.1's whole calibration problem is seed-dominated, and the
seeds are not written down.

---

## 4. Claimed versus measured

**The brief's premise for this section does not hold (§0.2), and the honest result is that
there is nothing to attribute to a machine difference: the implementers develop on the
scoring node, and every prediction landed.**

| entry | claimed in its ice_r7 record | scored | Δ |
|---|---|---|---|
| `L6_unrolled` | 0.238 (3.30-mode) / 0.271 (2.90-mode) | 0.238 | exact, fast mode |
| `L6_pfa` | 0.3659 (`hip`) / 0.364 (`cst`) | 0.320 | −12% (faster) |
| `L8_fusedaxes` | 0.511 min and median | 0.511 | **exact** |
| `L8_radix8` | 0.553–0.554 | 0.551 | −0.4% |
| `L8_batchsimd` | 0.548 | 0.555 | +1.3% |
| `L13_direct` | 3.476 / 3.504 / 3.505 / 3.667 | 3.471 | −0.1% |
| `L13_rader` | 3.688 / 3.692 / 3.694 / 3.696 | 3.688 | **exact** |
| `L17_matrixsimd` | 9.053 min / 9.056 median | 9.035 | −0.2% |
| `L17_rader` | 11.890–11.969 (7 pairs) | 11.919 | in range |
| `L17_winograd` | 11.351–11.473 | 10.922 | −3.8% (faster) |
| `L23_matrixsimd` | 33.995 / 34.007 / 34.163 | 33.876 | −0.4% |
| `L23_rader` | 35.88–36.39 (5 windows) | 35.919 | in range |
| `L36_pfa` | 100.039 / 100.201 / 100.294 | 100.296 | +0.0% |
| `L45_mixedradix` | 225.678 (sd 0.10%) | 225.571 | −0.05% |
| `L45_pfa` | 233.678 (sd 0.05%) | 231.220 | −1.1% |
| `L64_blocked` | 607 | 608.213 | +0.2% |
| `L64_radix8` | 620.2 | 614.640 | −0.9% |

Seventeen of seventeen quoted predictions land within ~1% or inside their own quoted
window, in both directions. **The one genuinely bimodal entry is `L6_unrolled`, and the
split is within this machine, not across machines**: it quotes 0.271 in "2.90-mode" and
0.238 in "3.30-mode" for the same binary across five sessions, and the scored number is the
fast mode. The L=6 headline in §1 is therefore a best-case reading; the same code will read
+14% in a session where the core sits at 2.90 GHz, and §3.6 shows the node visits both
states within a single round. **This bimodality — turbo/licence residency varying per
process on an exclusive node — is the only "machine difference" in the round, and it is one
machine.** Two consequences: the panel's self-probed-clock variant selection is unsound
(§3.6), and any cross-round comparison at L=6 tighter than 15% is meaningless.

The one claim that should be marked down is not a timing claim: `L17_winograd`'s **accuracy**
prediction missed by 55% in the unsafe direction (§3.4).

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

**§4.4 (split vs interleaved complex) moved decisively, and it moved by having its actual
variable identified. The lever is not the complex layout. It is what lives in the SIMD
lanes.**

ice_r6 reopened §4.4 against the corpus with `L45_pfa`'s bit-exact split-complex build,
which lost by 30–38% because the y↔z corner turn's gathers degraded from 16-byte to 8-byte
granules — "deleting the port-5 shuffle term buys you a larger port-2/3 term". That verdict
ended with a boundary condition: *split-complex pays when the vectorised axis is the batch,
and loses when the vectorised axis is spatial.* **This round tested the first half of that
sentence at seven geometries, and it is the correct half.**

Seven entries rebuilt their chain state so that the SIMD lanes hold **volumes** rather than
points along a spatial axis — 4 or 8 volumes per zmm, the transform along any axis becoming
a plain strided vertical sweep with zero shuffles by construction, and the corner turn
simply not existing. The correlation with the round's results is as clean as this panel ever
gets: **the seven entries whose ice_r7 record discusses lanes-are-volumes are exactly the
seven largest movers of the round**, in order.

| rank | entry | Δ | lanes-are-volumes? |
|---|---|---|---|
| 1 | `L13_direct` | −34.3% | yes (`soa8`, B≥8) |
| 2 | `L13_rader` | −31.4% | yes (8 vols/zmm, split) |
| 3 | `L17_matrixsimd` | −24.3% | yes (4 vols/zmm, **interleaved**) |
| 4 | `L6_unrolled` | −18.2% | yes (`g8`, 8 vols/zmm, split) |
| 5 | `L45_mixedradix` | −14.9% | yes (lane-packed) |
| 6 | `L45_pfa` | −12.5% | yes |
| 7 | `L8_fusedaxes` | −7.9% | yes (`bl8` batch-lane) |
| 8–18 | everything else | −6.9% … +8.1% | no |

**The sharpest result is `L17_matrixsimd`'s, because it separates the two variables the
corpus had conflated.** The rival generators' form is 8-volume *split*-complex; their own
SoA attempts measure 0.038–0.041 s at L=17 on this node, *slower* than their within-volume
padded engine at 0.0335. `L17_matrixsimd` kept **interleaved** complex with 4 volumes per
zmm, so its existing merged-reordered `chunk17zri` codelet (148 FP ops per 4 pencils) runs
verbatim — the lanes just mean volumes — and took the cell at 9.035 in half the rivals' L2
footprint, deleting all tile transposes (73 A2G groups × 80 `vpermt2pd` ≈ 5.8k port-5 uops),
all padding work, and the cross-slab fringe: ~75k → ~58k vector uops per volume-step. It
verified bit-identity of the v6 and v7 chain outputs by `cmp` rather than assuming it.

**What §4.4 should now say.** Split-complex was never the operative variable; it was
correlated with the operative one in the corpus's sources because `DFT_n ⊗ I_ν` kernels put
the batch in the lanes as a side effect. The transferable statement is: **put volumes in the
lanes and the corner turn disappears — that is worth 8–34%. Whether the complex parts are
split or interleaved is a second-order choice, and at L=17 interleaved won it.** ice_r6's
load-port conservation law survives intact and is now explained: it was measuring the cost
of *not* having the batch in the lanes.

**This also closes, in its own terms, §4.8 item 5** — "no quantified comparison of
batch/vector-loop placement for batched small 3D transforms … our largest untapped search
axis, and the literature only gestures at it." It is now quantified at seven geometries on
one node in one round, with a magnitude (8–34%), a mechanism (corner-turn elimination), and
a boundary (§0.3: it requires B ≥ 4, and evaporates at B=1).

**Secondary, §4.3 (axis fusion / pass count):** `L36_pencilfused`'s rejected fused-boundary
custody adds a fourth measured negative. Regrouping the chain so each sweep completes step
*k* and begins step *k+1* — halving the state read/write traffic at identical arithmetic —
measured **+3.0%** at L=36 despite winning the L=64 cell in r6 for `L64_radix8`. Pass-count
reduction remains not the variable; ice_r6's "group uop count and balance are" stands.

**Not moved:** §4.1 (register liveness — nobody counted stack traffic again), §4.2 (L=17
arithmetic — the round won L=17 on layout, not op count, for the second round running),
§4.5 (the literal (8,8)-vs-(9,9) L=8 experiment), §4.6, §4.7. **§4.8 item 6 remains factually
wrong about the scoring hardware for the third round (§0.1), and the PMU harness that
ice_r6 called "the highest-leverage infrastructure item after the L=6 gate" was not built —
`L8_fusedaxes` still reports `pmc=na` and `L8_radix8` still lists `perf_event_open` as the
only instrument left.**

---

## 6. The single highest-value thing the next round should attack, per geometry

**L = 6 — fix the gate, then score `B=1`. Fourth round of asking, and it is harness work,
not implementer work.** The cell is not validly gated (§3.1) and both entries have been
requesting a policy decision since ice_r4. Two changes: (1) write `chain_ok` back into
`result["ok"]` in `check.py` — one line, without it the ranking cannot see a chain failure
and it has now ranked nine of them twice; (2) set `eff_tol` from a **multi-seed worst case
over the six libraries and `baseline_matrix`, announced before the round starts**, or move
the gate to a mid-chain checkpoint at m ≤ 2572 where the exact and float-seeded maps are
still four orders apart. The data to choose with is in `L6_unrolled.md`. Both L=6 entries
are one environment variable from either policy, so re-scoring costs nothing once a floor
exists. Until then, do not spend implementer rounds tuning L=6: the last two rounds bought
26% at a cell whose correctness criterion the reference implementation fails by 45×.

**L = 8 — build the PMU harness. It is now the binding constraint at this geometry and it
serves the whole panel.** All three entries sit at ~1.2× a port-pool floor they cannot
decompose further, all three have asked for counters, `L8_fusedaxes` has a dormant
`-DL8_PMC=1` probe waiting, and `L8_radix8` has the specific hypothesis (MITE-vs-DSB
delivery: ~5.5k instructions per step against ICX's ~2.3k-uop DSB). `perf` is absent but
bare-metal counters work; one `perf_event_open` harness settles it. Second: §3.3 shows
`L8_radix8` and `L8_batchsimd` are bit-identical, so one of the three seats is spent on a
duplicate schedule — a seeded-from-scratch fourth design is worth more than a fourth
schedule, as ice_r6 also concluded.

**L = 17 — take the lanes-are-volumes state into the other two engines, and put the Newton
tier back into `L17_winograd`.** `L17_matrixsimd` won by 21% with chain v7 while
`L17_winograd` and `L17_rader` gained 6.2% and 3.0% from tuning their existing shapes; the
lever is proven at this exact geometry and is not adopted by two of three entries. Do that
before anything else. Then §3.4: `L17_winograd` traded 93× of chain accuracy for ~2 ops and
lost the cell — reclaim it, its own record says the exact arm is available. **And note what
the 24% did not buy: the B=1 path at L=17 is untouched r6 code at 11.876, so the geometry's
headline gain is batched-only.** If B=1 is ever scored (§0.3), L=17 regresses to r6 there.

**L = 36 — the cell is a 0.5% three-way tie among three PFA 4×9 implementations and the
structural search inside that decomposition is exhausted.** Three entries, one algorithm,
99.809 / 100.296 / 107.009, all bit-identical on the single transform, and the round's one
new structural idea (fused-boundary custody) measured +3.0%. Meanwhile L=36 has the second-
smallest library margin on the board (2.84×) and is **the one geometry where lanes-are-
volumes was not tried by anyone** — its working set is 11.39 MiB at B=8, so this is the
L2↔DRAM regime that §4.3's own annotation calls "the largest untried structural move on the
board" (tile the batch to fit L2, then run all three axes inside the tile). Direct one entry
at a volume-lane or L2-tiled chain state and stop the third round of ratio-tuning the same
eager-map protocol.

**The other four, one line each.** **L=13:** the round's biggest winner (−34%) and the lever
is fresh — push `soa8` down into the B=1 and remainder paths, which still run the
interleaved `mz` engine at 5.29. **L=23:** the only geometry where nobody tried
lanes-are-volumes and both arms gained under 6%; at B=16 with a 5.94 MiB working set it is
the best remaining candidate for the lever. **L=45:** −15% and the lead changed hands; the
two arms are 2.5% apart and both now lane-packed, so the next move is the L2 tiling question,
same as L=36. **L=64:** get an accurate arm back (§3.5) — 8.2× margin across a
bit-identical pair with no fallback is one chain-length change away from losing the whole
geometry — and look at `L64_blocked`'s recurring 17% tail spike (§3.6), which is the only
periodic intra-run stall on the board.

**Panel-wide, above all of the above:** add the eight `B=1` rows to `cases.txt`. §0.3. The
round's dominant lever requires a batch, and the series has never measured the case where it
is absent.

---

## 7. Curation

Applying `docs/CURATION.md`'s four grounds in order.

**Promoted (16).**

*Criterion 1 — fastest correct entry per geometry, one per L, always (8):*
`L6_unrolled` (0.238), `L8_fusedaxes` (0.511), `L13_direct` (3.471), `L17_matrixsimd`
(9.035), `L23_matrixsimd` (33.876), `L36_mixedradix` (99.809), `L45_mixedradix` (225.571),
`L64_blocked` (608.213). The L=6 pair is promoted under the §3.1 reasoning: it fails a gate
that the reference implementation fails by 45× and it is the most accurate backend in its
cell.

*Criterion 2 — structurally different runner-up, close (7):*

* **`L13_rader`** (3.688, 1.06×) — Rader-13 against a dense 13×13; genuinely different
  algorithm, 6% apart, and it drove the round's second-largest gain independently.
* **`L23_rader`** (35.919, 1.06×) — same relationship at L=23, 6% apart.
* **`L45_pfa`** (231.220, 1.03×) — GT-PFA 9×5 against mixed-radix 9×5; 2.5% apart, and this
  is the entry whose ice_r6 bit-exact split build produced the load-port conservation law
  that §5 has now completed.
* **`L64_radix8`** (614.640, 1.01×) — radix-8²/axis split-complex against a blocked z-split;
  1% apart and the source of the ckind=2 fused-boundary custody idea, whose *failure* to
  transfer to L=36 is one of the round's better negatives.
* **`L36_pfa`** (100.296, 1.00×) — a statistical tie with the winner (§3.6) and it delivered
  precisely what ice_r6 §6 asked L=36 for: the lagged map (mix=7) plus phase-1
  un-broadcast, "two stall cures worth 7%". A directive answered on demand is worth keeping.
* **`L8_radix8`** (0.551, 1.08×) — the faster of the two runner-up schedules at L=8 and the
  source of the geometry's documented negatives (v3 fused +15%, lazy map +11%, unbalanced
  half-pass +10%).
* **`L17_winograd`** (10.922, 1.21×) — at the outer edge of "close", and promoted on
  structure rather than speed: it is the only non-dense-matrix engine still standing at
  L=17 and the panel should not lose the Winograd module. **Promoted with the §3.4 finding
  attached** — its 93× accuracy degradation and 55%-optimistic margin prediction are part of
  what the record is for.

*Criterion 3 — instructive failure whose record documents the number that killed it (1):*

* **`L6_pfa`** (0.320, 1.34× — outside the criterion-2 band, promoted on this ground
  instead). Its +8.1% is the exact-tier mandate priced to within 0.7% of its own
  prediction, and its round produced two measured negatives worth more than its ranking:
  the **SQRT tier falsified at +46%** (`qip` 0.5455 / `qpd` 0.5480 / `qsp` 0.5326 against
  `hip` 0.3659 — 27 `vsqrtpd` + 27 `vdivpd` per volume-step is divider-bound on this core in
  every schedule tried, including one built specifically to spread the divider under DFT
  work), and the finding that the Heron premium is latency and irreducible. It is also the
  control half of the round's cleanest pairing (§2).

*Criterion 4 — anything that beat a library:* satisfied by all 19; does not discriminate.

**Not promoted (3), with reasons.**

* **`L8_batchsimd`** (0.555, 1.09%) — §3.3 shows it is **bit-identical to `L8_radix8` to all
  16 digits on both the single transform and the chain**, and it is slower. Near-duplicate
  rule; same exclusion as ice_r6.
* **`L17_rader`** (11.919, 1.32×) — 32% behind, the geometry's smallest mover (−3.0%), and
  ice_r6 already judged it "a near-duplicate of the union of two promoted entries" on its
  own description. Nothing this round changed that. Its artifacts survive in
  `strategies/L17_rader.md`, which is tracked independently of `exemplars/`.
* **`L36_pencilfused`** (107.009, 1.07×, **+0.1%**) — the round's only entry to ship no gain,
  and the third implementation of PFA 4×9 with the same eager-map protocol; the
  near-duplicate rule allows two and it is the slowest. It has a real criterion-3 claim —
  the +3.0% rejection of fused-boundary custody is the round's best cross-geometry negative
  — but promoting a third near-identical L=36 source duplicates code the next panel already
  receives twice, and the finding is fully preserved in `strategies/L36_pencilfused.md`.
  Same exclusion as ice_r6, and as ice_r5 applied to `L36_pfa`. **The next panel should be
  pointed at that record explicitly**, since §6 asks L=36 for a structural change and this
  is the record of the last one that failed.

**`exemplars/ice_r7/NOTES.md` should record:** that §4.4's real variable is lanes-are-volumes
and not split-vs-interleaved, with the seven-entry rank correlation and `L17_matrixsimd`'s
interleaved counter-example (§5); that the lever requires B ≥ 4 and evaporates at B=1, which
makes the seven-round-old missing `B=1` rows a substantive gap rather than a cosmetic one
(§0.3); that the L=6 cell was ranked ungated for the second consecutive round, with
`baseline_matrix` failing its own gate by 45× (§3.1); that `L17_winograd` traded 93× of chain
accuracy for ~2 ops and lost the cell anyway (§3.4); that L=64 still has no accurate arm
(§3.5); that the fused-boundary custody lever does not transfer from L=64 to L=36, measured
at +3.0%; and that the series total moved 0.9390 s → 0.7916 s, 4.03× the best library at
every cell.

PROMOTE: L6_unrolled L6_pfa L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pfa L45_mixedradix L45_pfa L64_blocked L64_radix8
