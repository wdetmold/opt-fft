# VERDICT — round gen_r5

Monitor's judgement on the measured round. Source of record:
`results/gen_r5/leaderboard.txt`, `environment.txt`, `failures.txt`,
`build_errors.txt`, the 164 `c_*.json` correctness records,
`results/wisdom_a80n0.json` (as written by the scoring race),
`strategies/*.md` §"Round gen_r5", `bench/gen/cases.txt`, `PANEL_BRIEF.md`
and `docs/CURATION.md`.

---

## 0. Three corrections to the framing I was given

Same three as `gen_r4/VERDICT.md` §0. They still hold, so they go first.

**0.1 The geometries are not 6, 8, 17, 36.** Those belong to the *previous*,
fixed-size campaign (`bench/geom/`, rounds `panel_r1`…`panel_r11`) and they are
also the hard-coded fallback case list in `sweep.sh`. This is the GENERALIZE
campaign in `bench/gen/`, whose acceptance suite is frozen in `cases.txt` and is
**L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100**. All eleven were measured and
I report on those. The only r5 data touching the named sizes is unscored:
`gen_dense_prime` measured L=17 at **27.58 µs** (B=4 m=8, −17% on the exact-tile
GEMM) and `gen_pfa_small`'s generic coprime engine still covers 36. Neither is on
the board.

**0.2 The scoring machine is not Cascade Lake, and there is no second machine in
this round.** `environment.txt` reads `Intel(R) Xeon(R) Gold 6326 @ 2.90GHz` with
`avx512_vbmi / avx512_ifma / avx512_bitalg / avx512_vpopcntdq` — that combination
is **Ice Lake-SP** (node a80n0), which `PANEL_BRIEF.md` §"Scoring" names as the
scoring host. Cascade Lake (Gold 5218 — the part `docs/LITERATURE.md` §4.8 item 6
analyses) and Sapphire Rapids are *advisory* reruns due after this board;
`XARCH.md` does not exist yet (checked). L2 on this part is **1.25 MB**, not the
1 MB of Cascade Lake, and its AVX-512 downclock is mild, not severe — several
implementers size their L2 gates against 1.25 MB explicitly and are right to.

**0.3 There is no batched/non-batched pair per geometry.** `cases.txt` scores each
L at exactly one batch: B=64 at 10/12, B=32 at 15/20, B=16 at 25/27/31, B=8 at
32/40, B=4 at 50, and **B=1 at L=100 only**. Where implementers measured B=1 at a
batched cell those numbers live in the strategy records; I quote them in §1.1 as
unscored.

The consequence for item 4 of my brief is in §4: implementers develop via
`tryout.sh` on **leased cores of a80n0 itself**, so a claimed-vs-measured gap here
cannot be attributed to a machine difference. This round the largest such gap has
a completely different and fully documented cause, and it is the sharpest finding
on the board.

---

## 1. Headline per geometry

Fastest correct panel entry vs. the strongest library configuration in the same
cell (`baseline_matrix` is the harness floor, not a library). Every backend at
every cell reported `ok`; see §3.1 for what that does and does not cover.

| L | B | winner | µs/transform | GF/s | spread | best library | µs | win vs library |
|---|---|---|---|---|---|---|---|---|
| 10 | 64 | **gen_pfa_small** | 1.152 | 43.25 | 13.8% | mkl_dfti | 4.570 | **3.97×** |
| 12 | 64 | **gen_pfa_small** | 1.914 | 48.54 | 13.8% | mkl_dfti | 7.761 | **4.05×** |
| 15 | 32 | **gen_pfa_small** | 4.406 | 44.89 | 0.1% | mkl_dfti | 16.456 | **3.73×** |
| 20 | 32 | **gen_pfa_small** | 13.072 | 39.67 | 0.1% | fftw3_measure | 44.944 | **3.44×** |
| 25 | 16 | **gen_powp** | 31.352 | 34.72 | 2.8% | fftw3_measure | 109.191 | **3.48×** |
| 27 | 16 | **gen_powp** | 51.444 | 27.29 | 0.1% | mkl_dfti | 144.359 | **2.81×** |
| 31 | 16 | **gen_rader** | 84.668 | 26.15 | 1.1% | ducc0_c2c | 716.947 | **8.47×** |
| 32 | 8 | **gen_pow2** | 56.472 | 43.52 | 1.7% | mkl_dfti | 172.480 | **3.05×** |
| 40 | 8 | **gen_pfa_large** | 160.240 | 31.88 | 1.4% | mkl2026_dfti | 405.524 | **2.53×** |
| 50 | 4 | **gen_powp** | 415.637 | 25.46 | 1.9% | mkl_dfti | 947.620 | **2.28×** |
| 100 | 1 | **gen_pfa_large** | 4531.445 | 21.99 | 2.1% | mkl_dfti | 7790.545 | **1.72×** |

**Geomean over the eleven cells: 3.31× the best library (r4: 3.24×); 3.01×
excluding L=31 (r4: 3.01×).** Worst cell is L=100 at 1.72× (r4: 1.62×); best real
cell is L=12 at 4.05×.

Four qualifications, all adverse:

- **L=31's 8.47× is a property of the size, not of the entry.** Every library
  collapses on a prime edge: MKL 2022/2026 and all three FFTW planners sit at
  833–883 µs, 10× off `gen_rader`, and ducc0 is only marginally better. *All seven*
  gen entries at L=31 beat the best library there, `gen_bluestein` (289 µs) and
  `gen_twiddle`'s demo (267 µs) included. As in r4 I do not credit "beat a library
  at L=31" as an achievement anywhere in §7.
- **L=50 is a near-tie for the third round running.** `gen_powp` 415.637 vs
  `gen_pfa_large` 420.957 is **1.28%**. The lead flipped back from r4, and both
  engines moved for the same reason (`map_step_pair`, authored by `gen_pfa_large`
  and adopted verbatim by `gen_powp`). Neither should claim the cell.
- **L=100 separated, but not decisively.** `gen_pfa_large` 4531.445 vs `gen_powp`
  4618.710 is **1.93%** — up from r4's 0.013%, and `gen_pfa_large`'s spread (2.1%)
  is a quarter of `gen_powp`'s (9.3%). The rank is now consistent across two
  boards and both entries' own quiet-window floors agree with it, so I record
  `gen_pfa_large` as the winner while noting the gap is inside the runner-up's own
  run spread.
- **The 13.8% run spreads at L=10 and L=12 are the winner's own.** `gen_pfa_small`
  reports 13.8% spread at both, against 0.2–0.7% for MKL in the same windows. The
  reported minimum is a min-of-runs on a bimodal core, which is the confound the
  panel itself characterized in r4. `gen_batchlane` is 0.5% and 0.05% behind at
  those two cells with 11.5% and 0.7% spread. **At L=12 the 0.05% lead means
  nothing; treat 10 and 12 as shared cells.**

### 1.1 B=1 outside L=100 (unscored, from the records)

Reported here because round 6 draws unknown batches and scores the assembled
trunk. `gen_pfa_small` 3.447 / 4.761 / 12.422 / 28.650 µs at 10/12/15/20 (split
path); `gen_batchlane` 10.65 / 17.59 / 40.53 / 133.97 at the same sizes;
`gen_planner` 3.96 / 5.89 / 11.68 (per-volume path — the split group never engages
below B=8), 68.0 at 25, 203.5 at 31; `gen_powp` 39.96 / 50.35 / 416.87 at 25/27/50;
`gen_pfa_large` 185.8 / 419.2 at 40/50; `gen_pow2` 55.45 at 32 (≈ its batched
number); `gen_rader` 85.54 and `gen_dense_prime` 121.60 at 31 (both ≈ batched).

Two facts stand out. First, **the B=1 hole at 10/12/15 is unchanged for the fifth
round**: `gen_pfa_small`'s own record puts MKL at ~1.5/2.1/6.4 µs there, i.e. the
panel is 2–2.3× *behind* a library at B=1 in exactly the cells where it is 4×
ahead at B=64. Second, **at L=50 batching now buys nothing**: `gen_powp` reads
416.87 at B=1 against 415.637 scored at B=4, and `gen_pfa_large` 419.2 vs 420.957.
That is a capacity statement about the cell, and it is the basis of §6's L=50 item.

---

## 2. What changed since gen_r4, per geometry

Deltas are r4 board → r5 board: same node, same slurm job (438682), same harness,
same seed, five hours apart.

### The round's real content

**The generic layer, at every cell below L=32 — `gen_planner`'s split-group batch
engine.** This is the round by a wide margin. At B≥8, groups of 8 volumes now run
site-major **split**-complex with the batch as the vector dimension: full-width
pencil DFTs, broadcast-scalar twiddles, **zero shuffle-class instructions
(asm-audited per codelet)**, no masked tails, no transposes, and a map ladder that
sees 8 distinct sites per zmm instead of 4. Arithmetic is op-for-op the
interleaved form (`fmsub`/`fmadd` pairs reproduce `fmaddsub`'s per-lane rounding),
so outputs are **bit-identical to the per-volume path on the same tree** — which
is what makes the measurement a clean layout A/B rather than an algorithm change.

| cell | gen_planner r4 → r5 | speedup | gen_race tracking |
|---|---|---|---|
| L=10 | 3.460 → **1.408** | **2.46×** | 3.463 → 1.408 |
| L=12 | 5.112 → **2.463** | **2.08×** | 5.107 → 2.474 |
| L=15 | 11.634 → **5.648** | **2.06×** | 11.638 → 5.734 |
| L=20 | 24.685 → **18.349** | 1.35× | 24.032 → 18.405 |
| L=25 | 60.130 → **40.398** | 1.49× | 60.130 → 40.223 |
| L=31 | 202.978 → **139.998** | 1.45× | 195.606 → 139.795 |

And the boundary, which is as valuable as the win: the race **rejected** the group
engine at 27, 32 and 40 (a 3.1% loss and two honest ties), and it does not engage
at all at 50/100 (B<8). `gen_planner`'s own account: past vol ≈ 20k the group
working set (16·vol doubles vs 2·vol) trades L2 residency for lane width. So the
generic layer moved 0 to +1% at 32/40/50/100. See §5 — this is the measurement
`docs/LITERATURE.md` §4.4 has been asking for.

**L=40 — `gen_pfa_large` −15.1%** (188.718 → 160.240), the round's best tuned-size
gain, from the pair-packed map ladder (`map_step_pair`): the ~19-op Newton ladder
ran on vectors where every |z|² sat duplicated in both complex lanes, so the 8
distinct |z|² of a *pair* of vectors now pack into one zmm, one ladder serves both,
and reciprocals unpack pair-duplicated. 38 arith + 2 shuffles → 21 + 4 per pair,
a 45% cut in map arithmetic, and **bit-identical per element** — every r4 gate
value reproduced exactly, which is the strongest available regression check.

**L=50 — `gen_powp` −12.1%** (472.873 → 415.637) and **`gen_pfa_large` −9.7%**
(466.040 → 420.957), both from the same adopted ladder. The map was ~41% of step
arithmetic at this cell.

**L=100 — `gen_pfa_large` −6.1%** (4827.566 → 4531.445), `gen_powp` −4.3%. Same
ladder plus a rank reorder putting `ipp1` ahead of `ip*`.

**L=31 — the class winner did not move, and the rest of the field closed hard.**
`gen_rader` 84.603 → 84.668 (parity; the L=31 binary path is bit-identical to r4)
and `gen_dense_prime` 120.490 → 120.287. Both engines re-confirmed the crossover
is arithmetic: `gen_rader` measured its remaining structural idea (`R31_ZMIX`) at
+2.5% and +7% and closed the cell. Meanwhile `gen_twiddle` took **−42.3%**
(463.249 → 267.265) with conjugate-fold prime butterflies, `gen_planner` −31.0%,
and `gen_layout` −12.3%. The real r5 work at this class happened **off the board**:
`gen_rader` generalized the Winograd-C3-over-dense-blocks construction to runtime
tables for h = 3m primes, measuring **−34.5% at p=43, −40.9% at 67, −31% at 79,
−16% at 103**. That is round-6 insurance and the biggest arithmetic lever the
panel produced this round.

**L=32 — `gen_pow2` +1.3%** (55.746 → 56.472) on a shipped change it measured as a
wall-clock wash; see §2 regressions and §5.

**L=20 — `gen_dense_prime` −11.8%** (44.159 → 38.949) from the exact-tile GEMM;
still 3.0× off the class winner, which did not move (13.059 → 13.072).

**L=12 — lead change, `gen_batchlane` → `gen_pfa_small`** (1.915 vs 1.914). This is
a 0.05% "lead" and I do not treat it as one. The interesting part is *why*
`gen_pfa_small` moved −2.8%: diffing `gen_batchlane`'s source against its own, it
found that its r2 adoption of the bl8 map ladder had been a **transcription rather
than a copy** — one extra multiply per site inside the rsqrt Newtons, and `set1`
intrinsics where the original used static-const vectors. On the corrected body the
div-vs-rcp verdict at L=12 *flips*. Four rounds of A/B verdicts at this cell were
being set on a subtly wrong kernel.

### Regressions

Six cells went backwards against a flat or faster library baseline. That is the
test I apply: a cell is a regression only if the rest of the field held still.

| entry | cell | r4 → r5 | libraries in the same cell | verdict |
|---|---|---|---|---|
| **gen_powp** | L=27 B=16 | 44.281 → **51.444 (+16.2%)** | mkl −0.02%, mkl2026 +0.06%, fftw3_measure +0.08%, ducc0 −0.8% | **real, and fully diagnosed — §3.4** |
| **gen_layout** | L=10 B=64 | 4.904 → **5.259 (+7.2%)** | mkl +0.4%, fftw3_guru 0.0%, fftw3_estimate +0.1% | **real, mechanism in its own record** |
| **gen_dense_prime** | L=10 B=64 | 5.282 → **5.583 (+5.7%)** | as above | **real — §3.5** |
| **gen_dense_prime** | L=15 B=32 | 14.264 → **14.875 (+4.3%)** | mkl −0.6%, fftw3_estimate −0.1% | **real, and self-predicted** |
| **gen_layout** | L=20 B=32 | 39.905 → **41.215 (+3.3%)** | fftw3_measure +0.1%, fftw3_patient +0.2%, mkl +0.2% | **real** |
| **gen_layout** | L=25 B=16 | 93.358 → **95.488 (+2.3%)** | mkl +0.01%, fftw3_measure +0.5% | **real** |

`gen_layout` is the interesting case because its record contains the mechanism and
then draws the wrong conclusion from it. Fusing the graded map into the axis-2
exit means pencils issue `ceil(L/8)` map calls instead of `L/8`, so any
L mod 8 ∈ {1..4} pays dead-lane divides — the record quantifies this as **+33% map
divides at L=12, +28% at 25, +12% at 50**. L=10, 12, 20 and 25 are all in that
class. Their session ran 3–6% hot, so their same-window A/B (fused vs
`-DGL_DEMO_NOMAPFUSE=1`) legitimately showed the fusion beating its own control —
but they then wrote *"every SCORED cell wins or washes, and one code path beats a
size-conditional fusion nobody can maintain"* and shipped it ungated. **The board
disagrees at four of eleven scored cells.** The libraries in those cells did not
move, so "hot window" is not available as a defence for the *board*. The
gating knob already exists; the residency rule the rest of the panel applies to
every other fusion this campaign (`gen_twiddle`'s `32·L³ > 1.25 MB`,
`gen_bluestein`'s `BST_MAPFUSE_MAX_MIB`, `gen_planner`'s `L³ > 1728`) is exactly
what is missing here. Against that, the same change bought **−12.2% at L=32,
−10.0% at L=40** (recovering r4's unremarked L=40 regression), −12.3% at 31 and
−8.3% at 50 — so the shape of the result is right and only the gate is missing.

`gen_pow2`'s L=32 +1.3% I do **not** count as a regression: mkl2026 read +3.2% and
mkl +0.7% in the same cell, the entry's spread is 1.7%, and its own 16 interleaved
same-core rounds put the shipped `GP2_FTW=1` within ±0.3% of the r4 arithmetic.
It is noise on a change the entry itself declared a wash.

Smaller moves I judge to be noise, because the record contains a same-core
interleaved A/B on unchanged or bit-identical code and the board falls inside the
implementer's measured range: `gen_batchlane` L=10 +0.3% / L=20 +0.9% (paths
bit-identical to r4), `gen_pfa_small` L=20 +0.1%, `gen_layout` L=12 +4.1% (the two
FFTW planners read +3.1% and +6.2% in that same cell — the only cell where the
library field genuinely moved), `gen_bluestein` L=12/27/40/50 within ±0.6%.

### Lead changes

- **L=12: `gen_batchlane` → `gen_pfa_small`**, 0.05%. A tie.
- **L=50: `gen_pfa_large` → `gen_powp`**, 1.28%. A tie, and the flip is entirely
  attributable to `gen_powp` adopting `gen_pfa_large`'s own ladder.
- **L=100: no change**, but the gap widened from 0.013% to 1.93%.

---

## 3. Adversarial pass: failures, correctness, builds, absences

### 3.1 Correctness — clean on what was checked, but one of the three gates was not checked by the harness

Every one of the 16 backends at every one of the 11 cells reports `ok`. I read all
**164** `c_*.json` records rather than trusting the leaderboard column: zero have
`ok:false` or `chain_ok:false`. Single-call residuals sit at 7e-16…3.4e-13 against
a 1e-12/1e-10 tolerance; nothing is near a limit and nothing is suspiciously
*exact*. Independent corroboration in the records is unusually strong this round
because so many changes were designed to be bit-transparent: `gen_pfa_large`,
`gen_powp`, `gen_planner`, `gen_dense_prime`, `gen_bluestein`, `gen_twiddle`,
`gen_layout`, `gen_rader` and `gen_pow2` each verified chain outputs **bit-identical
to their own r4 binary** on the unchanged paths, and `gen_dense_prime` did it at
all 15 supported sizes. **No fast wrong answer is present on this board.**

**But the harness only runs two of the campaign's three gates.** `PANEL_BRIEF.md`
§"gates" requires (1) single call rel L2 < 1e-12, (2) **the two-step m=2 fused
precision gate at 3e-14 — "the precision contract; catches every shortcut"**, and
(3) the chain-end 300× divergence check. The `c_*.json` schema is
`{ok, rel_l2, max_abs, rel_max, tol, L, batch, chain_ok, chain_rel_l2,
anchor_rel_l2, chain_tol}` — gates 1 and 3 only, and `check.py` has no two-step
mode. **Gate 2 is self-reported by implementers on their own binaries and is not
verified by the scoring sweep.** All twelve records do report it, on the node, by
hand, with 11–30× margin (9.2e-16 … 5.0e-15 against 3e-14), and gate 2 is the one
designed to catch a shortcut that gates 1 and 3 would forgive. I have no evidence
any entry abused this. But the gate the brief calls the shortcut-catcher is
currently outside the monitor's chain of custody, and the fix is a `--twostep` leg
in `check.py` driven from `sweep.sh`. **This is a standing hole in the scoring
apparatus and it should be closed before round 6 scores an assembled trunk.**

### 3.2 Failures — one, and it is the reference floor

`failures.txt` contains exactly three lines, all the same entry, identical to r4:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout`. `baseline_matrix` is the library-free O(L⁴)/volume/axis
reference; at L=100 that is 10⁶ points × 100 per axis, and it already cost
34.1 ms/transform at L=50. It timed out on all three runs and is correctly absent
from the L=100 table. This is not a panel entry and not a defect — but **L=100
remains the one cell with no library-free harness floor**, so its correctness rests
entirely on the numpy gate, which did run and did pass for all fourteen backends
present.

**No panel entry crashed, hung, or was killed at any cell.**

### 3.3 Build — the r4 promotion condition was not met

`build_errors.txt` contains no errors. It contains two warnings, and they are
**the same two warnings, in the same two files, as round 4**:

```
impl/gen_dense_prime.c:1679:23: warning: iteration 1152921504606846976 invokes
    undefined behavior [-Waggressive-loop-optimizations]
impl/gen_rader.c:1284:23: warning: iteration 1152921504606846976 invokes
    undefined behavior [-Waggressive-loop-optimizations]
```

`gen_r4/VERDICT.md` §3.3 named these, established that the defect had **propagated
by adoption** (`gen_rader`'s own backend line says "s6 map adopted from
gen_dense_prime"; the two loops are character-identical), and made fixing them an
explicit **condition on promotion**: *"the fix is one line each and should land
before either is used as exemplar material… Named here so the next panel does not
copy it a third time."* Both entries were promoted. Neither line was touched. I
verified the current sources directly: `impl_5/gen_dense_prime.c:1678` and
`impl_5/gen_rader.c:1283` are the same `for (; i < npts; ++i)` tail, byte for byte.

One correction to r4's diagnosis, because it changes the severity. r4 called this
signed-integer overflow. The index is declared `size_t i` (`gen_dense_prime.c:1631`),
so unsigned wraparound is defined; what GCC is actually reasoning about is
**pointer-arithmetic overflow** — `zp + 2*i` leaves the object at i ≈ 2⁶⁰, which is
UB, so GCC infers the loop cannot reach that trip count. There is no
signed-overflow exploit path, `npts` is bounded by L³·B ≈ 10⁶, and the warning
does not fire today. So this **does not invalidate any number on this board**, and
it is less dangerous than r4 stated. It is still a diagnostic that GCC emits on
every build of two entries that are being handed to the next panel as exemplars,
and a monitor's promotion condition that was ignored for a full round is a process
finding regardless of the severity of the underlying bug. The condition is
**restated, unchanged, in §7**.

### 3.4 The round's decisive failure: L=27 was lost to the race layer, not to a kernel

`gen_powp` regressed **+16.2%** at L=27 (44.281 → 51.444), the largest single move
on the board in either direction, in a cell where every library held still to
within 0.1% and where the entry's own run spread is **0.1%** — so this is a stable
slow state, not window noise. Its record claims **44.567 µs**, a wash against r4,
backed by 3/3 same-core paired A/Bs showing the new paired-`vdivpd` map at −1.3%.

`results/wisdom_a80n0.json`, as written by the scoring race, says what happened:

```
gen_powp/chain5/L27/B16#2e449ac0 -> l27-ip0   widx=0  tie=1  us=58.6274  margin=-0.005567
gen_powp/chain5/L25/B16#6bb92654 -> l25-soa   widx=6  tie=0  us=31.1923  margin= 0.31
```

At L=25 the race found the `soa` engine by a 31% margin and the cell scored
31.352 — matching the record to 0.2%. At L=27 the race measured the rank-0 `ip0`
candidate **0.56% slower** than the best trial and the 3% hysteresis band installed
it anyway. The scored chain then ran the `ip0` engine, 15.4% behind the `soa` arm
the implementer measured. Corroboration, four ways:

1. The `gen_powp` backend blurb in the r5 leaderboard reads *"pick: l25-soa
   (B=16)"*. In r4 it read *"pick: l27-soa (B=16)"*. The L=27 pick changed engines.
2. **`gen_powp` documented this exact failure happening to itself in dev, at L=25**:
   *"My first cold race ran on a contended core… the race stored l25-ip0 and the
   graded chain shipped 41.1 µs instead of 31.4."* Same engine pair, same
   direction, ratio 1.31 there and 1.15 here.
3. Their record names the mechanism: *"interleaved candidates evict the SoA arena
   and i-cache between trial rounds; 256-step graded chains hide this, 2-step
   trials do not."* The race's trial is not representative of the workload it is
   selecting for, and it is biased **against** exactly the arena-resident SoA
   engines that win the graded cells.
4. Every library in the cell is flat; the spread is 0.1%.

This is the same class of failure `gen_r4/VERDICT.md` §3.5 recorded — where
`gen_powp` lost L=50 by 1.47% to a −1.82%-margin hysteresis install, and the record
declared the tax openly. **The tax has now been paid at 10× the size.** A 0.56%
trial margin selected a 15.4% slower engine. Two conclusions, both actionable:
the hysteresis band must not span structurally different arms whose trial and
graded costs are known to diverge, and the trial must be long enough to be
representative. Round 6 scores the assembled trunk **through this layer**, at three
sizes nobody has raced before. This is the highest-priority defect on the board.

### 3.5 The instructive failure of the round: bit-identical output, different speed

Three entries independently hit the same effect, and one of them isolated it
cleanly enough that the other two now have their explanation.

`gen_twiddle` put its new conjugate-fold prime butterfly inline as the default
case of `twd_butterfly`. L=31 went −42% as designed — and **L=12, 50 and 100
regressed +1…+4.5%, 4/4 same-core pairs each, with `cmp`-identical outputs, on
code paths that never execute the fold.** GCC had de-inlined and re-scheduled the
hot r=2..5 paths inside the now-bloated dispatch function. Moving the fold body
into a separate `noinline` function did not merely restore parity: the final binary
**beats** r4 at those sizes (12: 13.97–14.11 vs 14.83–15.25; 50: 1275–1321 vs
1363–1380; 100: 10734–10862 vs 11362–11464), because `twd_butterfly` ends up
*smaller* than r4's. Their own statement of the rule is the right one and is
panel-transferable verbatim: *"keep rare heavy cases out of hot dispatch
functions; if outputs are bit-identical and time moved, it is code layout, and an
A/B against the previous binary is the only detector."*

That rule explains two of §2's regressions that were otherwise unattributed.
`gen_dense_prime`'s exact-tile GEMM adds ~20 tail instantiations per GEMM, of which
only 2–4 are hot per size. It cut d-side FMA slots by 25% at L=10 and yet the
board went **+5.7%** there and **+4.3%** at L=15 — and their own record already
guessed the mechanism without being able to prove it: *"Strictly fewer FMA slots
yet no win — most plausibly DSB/code-layout: the tail bodies double the hot-loop
footprint at those sizes."* `gen_twiddle`'s isolation converts that guess into a
diagnosis with a known fix. This is the panel's cross-validation working exactly as
the strategy records are supposed to make it work, and it is the single most
reusable result of the round.

### 3.6 Two provenance incidents, both self-reported, both repaired — and I verified the repairs

**`impl_4/gen_twiddle.c` was hard-linked to `impl_5/gen_twiddle.c` at round start**,
so early in-place edits landed in the **archived r4 source** until a later
whole-file write split the inodes — which also silently dropped two edits from the
live file. Detected via an r4 control build failing strangely; repaired with
`git checkout -- impl_4/gen_twiddle.c`. This is an attack on precisely the thing
`docs/CURATION.md` created `impl_N/` to protect: *"a regression can be diffed
against the exact source that produced the previous round's numbers."* **Verified:**
`git status --porcelain bench/gen/impl_4/` is empty (archive matches HEAD), and
`stat -c %h` reports link count 1 for all 26 files in `impl_4` and `impl_5`. The
archive is intact. `gen_twiddle`'s warning to the panel — `stat -c %h` your
`impl_4` file before trusting an r4 control build — should go in the r6 brief, and
the round-seeding step should `cp` rather than hard-link.

**`results/wisdom_a80n0.json` was corrupted mid-session** by `gen_pfa_large` doing
a round-end strip with a bare Python rewrite that ignored the `.lock` file and the
nested `{host, format, entries}` layout; a concurrent `gen_planner` write was
orphaned and the file was unparseable until repaired under `flock`. Their record
also flags a 40-entry drop at 20:42 that was not their write. All of this predates
the 21:32 scoring sweep, every entry strips its own keys at round end so the
monitor cold-races, and the file I read carries 54 entries all written by the
scoring sweep itself. **The scoring picks are the sweep's own and are sound.** The
protocol survived because of the round-end strip, and `gen_race` shipped
`gr_wisdom_drop_prefix()` this round precisely to make the strip a single
flock-safe operation. Adopt it panel-wide.

### 3.7 Nothing is missing

I checked every entry against the class table in `PANEL_BRIEF.md` by enumerating
the 164 correctness files per cell, not by reading the leaderboard.

| entry | cells present | class coverage | verdict |
|---|---|---|---|
| gen_pfa_small | 10,12,15,20 | PFA coprime small | correct |
| gen_batchlane | 10,12,15,20 | SoA batch-lane 10/12/15 + adopted 20 | correct |
| gen_dense_prime | 10,12,15,20,31 | dense prime p≤31 + small composites | correct |
| gen_powp | 25,27,50,100 | p^k CT | correct |
| gen_pfa_large | 40,50,100 | PFA coprime large | correct |
| gen_pow2 | 32 | 2^k | correct |
| gen_rader | 31 | Rader primes | correct |
| gen_bluestein | all 11 | must run everywhere | correct |
| gen_planner / gen_race / gen_layout / gen_twiddle | all 11 | library layers | correct |

**No entry is silently absent from a cell it owns.** Every declined cell is a
declared class boundary. Clean sheet, and worth stating explicitly because a
silent absence is the failure mode this check exists for.

---

## 4. Claimed vs. measured

**The machine-difference premise does not apply to this round, and I will not
invent one.** Per `PANEL_BRIEF.md`, implementers develop via `tryout.sh` on **leased
cores of a80n0 itself** — the same Ice Lake-SP node the score is taken on. Cascade
Lake and Sapphire Rapids are advisory reruns that have not happened (`XARCH.md`
does not exist). The agreement across records confirms it: **twenty-two of the
twenty-four claimed numbers I could pair with a scored cell land within 4%**, most
within 2%, which no cross-machine comparison could produce. MKL's own board-to-board
movement between r4 and r5 is 0.0–3.3% at every cell — nothing like the 2.9× span
the framing anticipated, because there is only one machine here.

| entry | cell | claimed in record | scored | Δ |
|---|---|---|---|---|
| gen_batchlane | 15 | 4.410–4.413 | 4.411 | 0.0% |
| gen_pfa_small | 12 | 1.916–1.920 | 1.914 | −0.1% |
| gen_pfa_small | 15 | 4.418–4.425 | 4.406 | −0.3% |
| gen_powp | 25 | 31.420 | 31.352 | −0.2% |
| gen_bluestein | 31 | 289.0 | 289.292 | +0.1% |
| gen_twiddle | 100 | 10735.11 | 10725.350 | −0.1% |
| gen_layout | 31 | 196.6–201.0 | 195.931 | −0.3% |
| gen_layout | 32 | 199.0 | 199.953 | +0.5% |
| gen_rader | 31 | 85.27–86.05 | 84.668 | −0.7% |
| gen_planner | 31 | 141.1 | 139.998 | −0.8% |
| gen_pfa_large | 40 | 162.8 | 160.240 | −1.6% |
| gen_pfa_large | 50 | 426.6 | 420.957 | −1.3% |
| gen_dense_prime | 20 | 39.52–39.88 | 38.949 | −1.4% |
| gen_dense_prime | 31 | 119.89–122.38 | 120.287 | in range |
| gen_pow2 | 32 | 55.27–55.42 | 56.472 | +1.9% |
| gen_powp | 50 | 426.098 | 415.637 | −2.5% |
| gen_pfa_large | 100 | 4657 (quiet floor) | 4531.445 | −2.7% |
| gen_powp | 100 | 4762 (paired floor) | 4618.710 | −3.0% |
| gen_layout | 100 | 14976 | 14529.744 | −3.0% |
| gen_planner | 100 | 5953 | 5519.543 | −7.3% |
| **gen_powp** | **27** | **44.567** | **51.444** | **+15.4%** |

Where a gap exists it runs, as in r4, in the **opposite direction to a machine
penalty** — the scored number is usually *faster* than what the implementer could
measure — and the cause is the one the panel diagnosed in r4 and has now
institutionalized: `tryout.sh` acquires a fresh core lease per invocation, so
consecutive runs land on different cores in different turbo/neighbour states, and
the scoring sweep runs in a genuinely quiet window that no leased core reproduces.
`gen_planner`'s −7.3% at L=100 is the extreme, and their record already annotates
it (*"MKL +3.7% same window"*). Every one of the twelve records this round used
the held-lease same-core alternation protocol for every keep/kill verdict; that
methodology is now universal and it shows in the table above.

**The one real outlier is `gen_powp` at L=27, and it is not a machine difference,
a window, or a measurement error. It is the race layer installing the wrong
engine.** §3.4 has the receipt. I flag it here as well because it is the only cell
on the board where the implementer's claimed number is right about their code and
wrong about what shipped.

**One overclaim worth correcting.** `gen_twiddle`'s record states its demo *"now
beats ducc0 at every acceptance size and fftw3_measure at 100."* The fftw3_measure
claim holds (10725.350 vs 10938.892). The ducc0 claim holds at 6 of 11 sizes and
fails at five: L=10 (10.359 vs 9.611), L=27 (218.155 vs 188.698), L=32 (337.387 vs
308.586), L=40 (611.968 vs 593.594) and L=50 (1272.112 vs 1269.370). Their windows
may well have read differently; the board is the board. The underlying achievement
— a −42% fold at L=31 and a demo that is no longer an embarrassment — stands.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### §4.4 — "Split vs interleaved complex: strongly motivated, unproven" — this round supplies the missing measurement

§4.4's complaint is precise: *"there is **no published head-to-head of split versus
interleaved complex for a batched small FFT on AVX2/AVX-512**… Treat §2's verdict
as strongly motivated but unproven, and settle it with a microbenchmark before
committing all four sizes to one layout."* The §08 update marks the *direction*
closed (Popovici/Franchetti/Low's 1.3–2× on `DFT_n ⊗ I_ν` kernels; FFTc 2.0's IPC
0.13 → 2.59; ducc0 shipping split) and narrows the residual to the **granule**,
which §08 §1.10 answers from the cache line: *"8 volumes per granule for split
complex double, so every vector access is a whole 64-byte line."*

**`gen_planner` ran exactly that experiment, at exactly that granule, and this is
the cleanest version of it the campaign can produce**, because the two arms are
*not* different algorithms. Same candidate tree, same per-point op count, same
rounding — the split codelets use `fmsub`/`fmadd` pairs that reproduce
`fmaddsub`'s per-lane rounding, so **outputs are bit-identical between the split
and interleaved arms**, verified by `cmp` through the whole graded chain. The only
thing that changes is the layout. And the mechanism §04 predicted is confirmed
directly rather than inferred: **shuffle-class instructions in every split codelet
= 0, asm-audited**, versus the interleaved path's per-volume 4×4-block transposes.

The measured answer, from the board, at the 8-volume granule §08 recommends:

| L | interleaved per-volume (r4) | split 8-granule (r5) | speedup |
|---|---|---|---|
| 10 | 3.460 | 1.408 | **2.46×** |
| 12 | 5.112 | 2.463 | **2.08×** |
| 15 | 11.634 | 5.648 | **2.06×** |
| 25 | 60.130 | 40.398 | 1.49× |
| 31 | 202.978 | 139.998 | 1.45× |
| 20 | 24.685 | 18.349 | 1.35× |

**And, equally importantly, the crossover — which no source in the corpus gives.**
The race rejected the split group at L=27 (+3.1%) and called honest ties at L=32
and L=40; it does not engage below B=8, so L=50 and L=100 are untouched. The
mechanism is stated: at volume ≳ 20k the group working set is 16·vol doubles
against the per-volume path's 2·vol, so **the split granule buys lane width at the
cost of L2 residency, and past L≈25–27 on a 1.25 MB L2 that trade stops paying.**

So the number §4.4 asked for, on CPU, double precision, batched, AVX-512, at the
8-volume granule: **2.0–2.5× where the group fits L2, 1.35–1.5× at the L2
boundary, and zero to negative past it.** The 2.0–2.5× at the small sizes sits
right on §04's predicted 4/4-vs-2/4 vectorisation-efficiency ratio and inside
Popovici et al.'s measured 1.3–2× band. §4.4 can be marked closed with a CPU
number and a residency boundary attached; the surviving open piece is the
sub-8 granule (G=4/ymm) for B<8, which nobody has built — see §6.

Note that this also **retires §4.8 item 5** in part (*"No quantified comparison of
batch/vector-loop placement for batched small 3D transforms… our largest untapped
search axis"*), which r4 partially closed with volume-major vs step-major. Split
8-granule vs per-volume is the same axis one level up, and it is now quantified
with bit-identical outputs at six sizes.

### Secondary movement

**§4.5 — "Padding: does L=8 need it, and where?" — moved by a controlled
experiment, though still not by the counter §4.5 asks for.** §4.5's operative rule
is §04's: *"with a one-cache-line element granule, pad so every stride is an odd
number of cache lines."* `gen_pow2` tested removing that padding on a live kernel:
`GP2_KS=64` shrinks the custody row pad (S 578 → 526 KB, S+C 1.11 → 1.06 MB against
a 1.25 MB L2), produces **bit-identical output**, and is consistently **+0.3–0.5%
slower** in every rotated-order round. Two results follow. (a) The L2-capacity-edge
theory of the x-pass slack is dead — shrinking the working set made it *worse*.
(b) They re-derived the hazard audit before building and found the odd-line rule's
value at L=32 **is not the 4K store-load proofing** §08 predicts: with KS=64 this
engine has no store→load pair at equal address mod 4K in any phase. It is **L1-set
uniformity** — stride 9 lines walks all 64 L1 sets (gcd(9,64)=1) while stride 8
hits 16 sets at 4 lines each and loses to conflict pressure from H/c/stack sharing
those sets. That is §05's framing of the hazard, not §04's, arrived at
independently on a real kernel. `gen_rader` tested the complementary move — bump
row strides off 0 mod 4 in 64B units to spread 4K phases — and killed it: **+1.5%
at p=127 (3/3), +4% at p=61 (3/3)**. §4.5's standing complaint that
`ld_blocks_partial.address_alias` *"has never been checked"* **still stands**: both
entries reasoned from structure rather than reading the counter. One `perf stat`
run would convert two careful arguments into evidence.

**§4.6 — "Model versus search for the instruction schedule" — moved substantively,
in favour of §06.** §01's implementer takeaway was *"you should not need a search
phase"*; §06 §6.1 corrects that none of our sizes is a power of 4 so genfft's
spill-optimality proof does not hold, and concludes the schedule is *"the primary
thing to search."* Two independent r5 results say §06 is right and §01 is wrong,
and both are unusually clean because outputs were bit-identical:

- `gen_twiddle`'s case-bloat tax (§3.5): **+1…+4.5% at three sizes that never
  execute the changed code**, purely from GCC's inlining and scheduling decisions
  in a dispatch function, reversed by one `noinline`. This is §06 §3.4's ">3×
  runtime spread across flag combinations" and §01 §1.5's "50–100% from lexical
  scoping alone" reproduced on 2026 hardware and a 2022 compiler at ~5% amplitude.
- `gen_pow2`'s proof by subtraction: `GP2_FTW=1` removes **7.1% of FMA-port ops
  per 32-point line** (420 → 390) via the literature-11 dual-select FMA twiddle
  fold; the profile confirms the ops genuinely leave (x-phase 87.8K → 82.3K cyc,
  z+y 78.6K → 76.4K); **the wall does not move, ±0.3% across 16 interleaved
  same-core rounds across three sessions.** The op count is not the model at L=32.
  They then name the wall with an accounting: ~110K cyc port floor ∥ ~44K cyc L2
  transfer against a measured 158–166K, i.e. **the engine is at [port ∥ L2] with
  <8% residual**, and neither cutting ops 7% nor shrinking the set 5% touches it.

**§4.3 — "Is axis fusion worth 3× or 3%?" — refined again, in the same direction
r4 established.** r4's answer was that the payoff is set by which cache level the
deleted round trip lived at, and must be gated on residency. r5 adds three
data points and one falsified prediction. `gen_layout` predicted **−15…25% at
L=100** from deleting a 32 MB/step `zt` round trip and measured **−0.7%**,
correcting its own model in the record: after r4's NT-store work the chain at
L=100 is no longer bandwidth-exposed, and what binds is inside the FFT passes.
The same change was worth **−12.3% at L=31, −12.2% at L=32, −10.0% at L=40** — the
L2/L3 middle — and **+3.7% at L=12 B=1**, L1-resident, where the deleted traffic
was free and the dead-lane divides were not. `gen_bluestein` reached the same
boundary from the other side (*"the deleted work is L2-resident, so the win is
uop-count-sized, not traffic-sized"*) and added a sharper sub-result: its
`BST_CSTRIDED` arm, built specifically to be refuted, proves that **layout, not
placement, was the blocker** — fusing the map into the axis-1 scatter while
reading `c` strided is the *worst* of three arms at L=50 (1817–2139 vs 1772–1834),
while fusing it with `c` in custody order wins. The doctrine to carry into r6:
fusion pays in the L2/L3 middle, is a wash at DRAM sizes once the store side is
fixed, costs at L1-resident sizes, and requires the operand to be in the consuming
pass's walk order before it pays at all.

**Also worth recording, though not a §4 item:** the campaign's brief asked for a
first validation of `docs/literature/11-post2020-untested-ideas.md` Tier 1 in
performant software. Two entries delivered it for the **dual-select FMA twiddle
form** (Bergach; Linzer–Feig), independently: `gen_twiddle` built the library form
(`tw_cis_ds`, `tw_fill_ct_ds_split`, audited for |t| ≤ 1 and ≤ 0.51 ulp) and
`gen_pow2` built it into a shipping codelet. Both verdicts agree and both are
honest: **accuracy-neutral at fp64, op count genuinely reduced, wall-clock a wash
on this host.** `gen_pow2` shipped it as default anyway on the grounds that the op
headroom is free and a port-bound host may cash it, and made it a cross-arch race
axis. That is the right disposition, and "free, not faster, here" is a publishable
result about a citation that promised free accuracy.

---

## 6. The single highest-value thing the next round should attack, per geometry

| L | attack | why it is the highest-value one |
|---|---|---|
| **10, 12** | **The B=1 lane-spatial engine, built once and shared.** | Fifth round on three separate next-lists. The panel is **2–2.3× behind MKL at B=1** at 10/12/15 while being 4× ahead at B=64, and there are now *three* engines stuck at the same 3.4–4.0 µs at L=10 B=1 (`gen_pfa_small` 3.447, `gen_planner` per-volume 3.96, `gen_batchlane` 10.65). `gen_planner`'s split group explicitly does not engage below B=8, so r5's headline win does not reach here. Round 6 draws unknown batches and scores the assembled trunk; a size that misses a gate or plans badly scores **zero for the whole library**. This is the one hole that can zero a cell, and the panel has been circling it for five rounds. Build the ice `L6_pfa` interleaved-complex z-turn once; whoever moves first, the others adopt. |
| **15** | **Make the map-tail knobs a per-host race axis, then stop tuning the cell.** | `gen_pfa_small` (4.406) and `gen_batchlane` (4.411) are a dead heat and both are ~15% off a ~3.8 µs port floor whose remaining lever (the 3-FMA DFT5 lifting form) both records price at ~0.5%. The live finding is different and more useful: **two honest same-core sessions on two cores of the same node produced opposite div-vs-rcp verdicts at ±3%**, and `gen_pfa_small` separately showed the ladder *body* and the *tail* choice are coupled. That is not a tuning question, it is evidence the knob must be raced per host — which is exactly what CLX/SPR and round 6's surprise sizes will demand. Cost: a few extra `gen_race` candidates. |
| **20** | **A narrower group granule (G=4 / ymm lanes, or 2-site zmm).** | This is the only small cell where the split-group win was modest (−26% vs −52% at 15), and `gen_planner`'s own diagnosis names capacity: the 8-volume group is 16·vol doubles. Independently, `gen_pfa_small` closed its last L=20 idea on an *arithmetic* capacity argument — BB(1.07 MiB) + S(1.07 MiB) still exceeds the 1.25 MB L2, so the bounce buffer cannot fix capacity and adds 2.14 MiB/step of traffic — and declared the cell closed at ~13.6. Both engines therefore point at the same single fact from opposite directions. Halving the granule is the one move that changes it, and it is the same primitive L=50 needs (below), so one build serves two cells. |
| **25** | **A Winograd-style split DFT25 module inside `gen_planner`'s fused codelet.** | Nothing to do on the class winner: `gen_powp` is at 3.48×, moved −2.1%, and leads the generic layer by 29%. But the generic layer closed from 1.87× behind to 1.29× behind in one round, and `gen_planner` names the remaining gap precisely — `gen_powp`'s ~218-op hand-derived real-factor lines against its own ~500. Closing that is round-6 insurance for every composite the draw might land on, and it is arithmetic, which is the one lever the split-group work did not pull. |
| **27** | **Fix the race trial and the hysteresis band. This is the panel's top priority, not just L=27's.** | §3.4: a **0.56% trial margin installed an engine that ran 15.4% slower on the graded chain**, costing the cell a 16.2% board regression and dropping it from 3.26× to 2.81× vs MKL. The diagnosis is already written down by the entry that suffered it — 2-step trials let interleaved candidates evict the SoA arena; 256-step graded chains hide it — so the trial is *systematically biased against* arena-resident engines, which are the ones that win. Two concrete fixes: lengthen the trial until trial rank tracks graded rank (verify at L=25, where the two disagree by 31%), and do not let the 3% hysteresis band span structurally different arms. Round 6 scores the assembled trunk through this layer at three unraced sizes. Only after that: `gen_powp`'s x-pass two-column pipelining, now **five** rounds on its own list. |
| **31** | **h-even primes, and 127.** | The cell itself is closed and both sides now agree in writing: `gen_rader` is at its issue-port model (85 µs ≈ 250k cycles) and killed its last idea at +2.5%/+7%; `gen_dense_prime`'s 1.42× gap is arithmetic. r4 told them to port the effort to generic primes and **they did** — outer-C3 Rader delivered −34.5% at 43, −40.9% at 67, −31% at 79, −16% at 103. The construction requires h = (p−1)/2 odd and 3∤(h/3), so it leaves **{37, 41, 53, 61, 73, 89, 97, 101, 109, 113}** (h even — the E/cos system is still a cyclic-h correlation for any h; only the O/sin system goes negacyclic, and a negacyclic C2 split is 3-block Karatsuba) and **127** (h = 63 = 7×9, needs Agarwal–Cooley C7×C9). Together those are most of the primes a round-6 draw in 14..127 can land on, and the panel currently serves them at dense-engine speed. |
| **32** | **A different step factorization, or concede the cell.** | `gen_pow2` did the rare thing and proved the negative: 7% fewer FMA-port ops moved the wall 0.0%, 5% less working set made it 0.3–0.5% *worse*, and the phase accounting (~110K cyc port ∥ ~44K cyc L2 vs 158–166K measured) leaves <8% residual. It also killed its own L=64 next-step on paper (the proposed x(s)+z(s+1) fusion is traffic-neutral). The only remaining lever it names is literature 11 Tier 2's **two-axes-per-pass with in-register y×z tiles** — which is the *same* structural idea `gen_powp` queues for L=100. One entry should burn a round on it and both adopt; nobody should burn two. |
| **40** | **The p1 staging spill/PMU audit.** | `gen_pfa_large` took the round's best tuned gain here (−15.1%) and the map is now ~30% of step ops rather than 46%, so by its own accounting **the FFT codelets are the whole step at 40/50**. The four-round-old audit of p1's Zv/Wv/T_ staging (80–300 live vectors per line) is finally the top lever rather than a deferred one. The r4 regressions at this cell are both resolved without further work (`gen_race` −1.0%, `gen_layout` −10.0%). |
| **50** | **The 2-volume-pair / G=4 half-group schedule at B=4.** | Queued by three entries for two rounds and measured by none. The cell is a 1.28% dead heat, and there is now a hard number that says batching buys nothing here: `gen_powp` reads **416.87 µs at B=1 against 415.637 scored at B=4**, and `gen_pfa_large` 419.2 vs 420.957. B=4 is below `gen_planner`'s group threshold, so the split-group win — the largest structural result of the round — cannot reach this cell at all. A narrower granule is the only thing that changes any of that, and it is the same build L=20 wants. |
| **100** | **A PMU session, before any more chain candidates. Second round of asking.** | Still the worst cell (1.72×) and now the cell with the worst model-to-measurement record on the board. This round produced: `ipk1` winning the interleaved race by 1.9% at `gen_pfa_large` and losing 4/4 held-lease pairs by up to +21%; the same candidate splitting 2-2 at `gen_powp`; `PREFETCHNTA` losing 9–13% at both entries; and `gen_layout` predicting −15…25% and measuring −0.7%. **Four models, four falsifications, zero counters read.** r4 asked for exactly this and nobody ran it. Attribute the residual to port 5 versus DRAM before proposing another chain family — otherwise r6 will produce a fifth candidate and a fifth contradiction. |
| **apparatus** | **Put gate 2 in the scoring harness.** | §3.1: the two-step m=2 precision gate — the one `PANEL_BRIEF.md` calls the shortcut-catcher — is self-reported and not run by `check.py`. Round 6 scores an assembled trunk at three sizes drawn by the monitor. A `--twostep` leg is a small change to `check.py` and `sweep.sh` and it closes the only hole in the correctness chain of custody. |

---

## 7. Curation decision

Applying `docs/CURATION.md` in its stated order. Two framing notes first. `impl_5/`
is complete provenance and every strategy record is tracked regardless of
promotion, so declining an entry does not lose its lesson — `exemplars/gen_r5/`
should stay a *reading list*, not an archive. And the promotion list is large by
construction, not by indulgence: the panel is twelve entries, four of which are
library layers scored by adoption and eight of which own **disjoint** size classes,
so rule 1 alone claims six. The discrimination has to happen at the margin, and it
does.

**Rule 1 — the fastest correct entry for each geometry, always.** Eleven cells
collapse to five distinct class entries: `gen_pfa_small` (10, 12, 15, 20),
`gen_powp` (25, 27, 50), `gen_rader` (31), `gen_pow2` (32), `gen_pfa_large`
(40, 100). All five promote. `gen_powp` promotes on its L=25 and L=50 numbers; its
L=27 cell was lost to the race layer, not to the kernel (§3.4), and the record
documents the engine that should have run.

**Rule 2 — a structurally different runner-up when it is close.**
- **`gen_dense_prime` at L=31 (1.42×).** Outside the "~20%" guideline, but this is
  the crossover the brief explicitly commissioned (*"gen_rader — 31 (crossover
  fight vs dense)"*), the alternative is genuinely structural (conjugate-pair dense
  fold vs Rader cyclic convolution), and both sides have now independently and
  publicly re-confirmed the gap is arithmetic and permanent. It also produced the
  round's most reusable generic result — the exact-tile GEMM, −32% FMA slots at
  17/19, −37% at 20, verified bit-identical at all 15 supported sizes — which
  `gen_rader` adopted as a discipline. **Promote.**
- The other runners-up (`gen_powp` at 100, `gen_pfa_large` at 50) are inside 2% and
  already promoted under rule 1.

**Rule 3 — instructive failures whose record documents the number that killed
them.** Four qualify, and each documents a different class of mistake.
- **`gen_twiddle`.** Declined in r4 for gaining no adopter; it earns promotion now.
  The case-bloat tax (§3.5) is the round's most panel-transferable finding —
  +1…4.5% at three sizes on **bit-identical output**, isolated to code layout,
  fixed by one `noinline`, and it retroactively explains `gen_dense_prime`'s two
  otherwise-unattributed board regressions. It also delivered −42.3% at L=31,
  rejected `gen_pfa_large`'s pair-packed ladder **on its own engine** with a
  boundary that generalizes (*"pack the ladder where the map is a standalone or
  FMA-bound pass, never inside a shuffle-bound exit"*), and produced one of the two
  first performant validations of literature 11 Tier 1. **Promote.**
- **`gen_race`.** It carries the round's most consequential single number, and it
  is a number *against itself*: the L=27 wisdom receipt (§3.4). It also fixed all
  three of r4 §4.1's complaints — the L=100 tile pick is now `t32` by a confident
  4.9% (r4 installed `t64` and lost 3.2%), L=40 is `t32` and now runs 0.2% *ahead*
  of `gen_planner`'s default, and the layer is ahead at 4 of 11 cells rather than
  behind at 3. Round 6 scores the trunk through this layer, so the next panel needs
  both the fix and the remaining defect in front of it. **Promote.**
- **`gen_layout`.** Four scored-cell regressions shipped under a stated judgement
  (*"every SCORED cell wins or washes"*) that the board falsifies at 10, 20 and 25
  — while the record itself contains the mechanism (dead-lane map divides at
  L mod 8 ∈ {1..4}, +33% at L=12) and the missing gate is one line. Against that:
  −12.2% at 32, −10.0% at 40 (recovering its own unremarked r4 regression), −12.3%
  at 31, and a **corrected prediction** at L=100 (forecast −15…25%, measured −0.7%)
  that is real evidence in §5. It also shipped `gl_map16` as a library primitive
  **with a measured negative verdict attached** so the next entry reads a number
  instead of burning a window — which is precisely the behaviour this repository
  exists to reward. **Promote.**
- **`gen_bluestein`.** Wins nothing and is last or near-last everywhere, but it is
  the **any-L existence guarantee that round 6's surprise draw rests on** (correct
  at all eleven cells and across a 2..128 sweep, plus L=101 and 127), and its
  `BST_CSTRIDED` arm was built specifically to be refuted and was — establishing
  that layout, not placement, was the blocker. It also declined `gen_layout`'s NT
  stores on a *structural* argument rather than a burned window, and recorded one
  honest non-identity at L=101 tail groups rather than hiding it. **Promote.**

**Rule 4 — anything that beat a library baseline.** Applied with judgement, because
at L=31 every gen entry beats every library by virtue of the size (§1); L=31 alone
carries no promotion.
- **`gen_planner`** beat the best library at **all eleven** cells for the second
  round running and delivered the round's headline — the split-group batch engine,
  1.35–2.46× at six cells with bit-identical outputs — which is also what moved
  §4.4. It is the round-6 trunk. **Promote.**
- `gen_layout` beat the best library at 20, 25, 27 and 31 on its own merits;
  `gen_race` at 10, 12, 15, 20, 25, 27, 40, 50 and 100; `gen_twiddle` at 12, 15,
  20, 25, 31 and 100 (against ducc0) and at 100 against fftw3_measure. All already
  promoted.

**Declined: `gen_batchlane`**, under `CURATION.md`'s explicit prohibition on
near-duplicates of an already-promoted entry. This is a close call and I want the
reasoning on the record, because r4 promoted it. It wins no cell (0.05%, 0.1%,
0.5% and 1.5% behind `gen_pfa_small` at 12, 15, 10 and 20). Its two shipped r5
changes are both **adoptions from `gen_pfa_small`** — the memory-form L=15 pencil,
taken on `gen_pfa_small`'s r4 evidence and its ≤32-live-zmm rule, and the map tail
that followed from it. Its one original structural idea, the hybrid form, it built
and killed itself at +0.35%. Its 10/12/20 paths are bit-identical to r4. The two
engines now share the `map8` body **verbatim**, and `gen_pfa_small` strictly
dominates on coverage (it owns L=20 outright and carries the generic coprime engine
for 14/18/21/36/56 that `gen_batchlane` lacks). The convergence is near-total and
promoting both would put two copies of one engine in front of the next panel.

What is lost by declining is one paragraph, and it survives intact in the tracked
record: **the div-vs-rcp verdict at L=15 is core-state-dependent on this node** —
two honest held-lease same-core sessions on different cores of a80n0 produced
opposite verdicts at ±3%, six consecutive pairs each with in-run sd < 0.1%. That is
a methodological finding, not a code artifact; it needs no source in front of the
next panel to be useful, and §6 acts on it. Three clean negatives (hybrid form
+0.35%, sched-pressure on the memory form +10.9%, rcp on the memory form +4.1%)
likewise survive in `strategies/gen_batchlane.md`.

**Recommendation for the r6 brief, arising from that decline:** `gen_batchlane` and
`gen_pfa_small` should be **merged, or explicitly differentiated**. Four rounds of
mutual adoption have converged them onto one kernel family, and the round-5 record
of that convergence — `gen_pfa_small` discovering its own four-round-old *mis-
transcription* of `gen_batchlane`'s ladder by diffing the two sources — is an
argument for one maintained engine with a raced knob set, not two. The panel slot
would be better spent on the B=1 lane-spatial engine that §6 lists first for both.

**Conditions on promotion.**
1. **The UB warning in the shared map tail (§3.3) must be fixed** in
   `impl_5/gen_dense_prime.c:1678` and `impl_5/gen_rader.c:1283` before either is
   put in front of the next panel as exemplar material. This was already a
   condition on their r4 promotion, it was not actioned, and both were promoted
   anyway. It is one line each (`zp[i2]` with `i2 += 2`, or index the pointers
   directly). Exemplars are read to be copied, and this one has already propagated
   once by adoption.
2. **`gen_powp`'s L=27 exemplar must ship with the wisdom receipt** from §3.4
   alongside its record. The code is not what lost the cell and the next panel must
   not read the 51.444 µs board number as a property of the kernel.
3. **`gen_layout`'s exemplar must carry the four-cell board regression** (§2) next
   to its `-DGL_DEMO_NOMAPFUSE` knob, so the next adopter of exit-fused mapping
   reads the residency gate as required rather than optional.

---

PROMOTE: gen_pfa_small gen_powp gen_rader gen_pow2 gen_pfa_large gen_dense_prime gen_planner gen_race gen_layout gen_twiddle gen_bluestein
