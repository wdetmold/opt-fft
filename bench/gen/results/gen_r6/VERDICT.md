# Round gen_r6 — monitor's verdict

Measured 2026-08-25T03:21:23-04:00, host a80n0.lqcd.mit, slurm job 438682.
All numbers below are from `results/gen_r6/leaderboard.txt` unless explicitly
labelled as an implementer self-report.

---

## 0. Three corrections to the monitor brief, before any judgement

The brief I was handed (preserved verbatim in `monitor_prompt.txt`) carries three
premises that this round's own files contradict. I am recording them first because
two of them would silently corrupt items 1 and 4 if followed.

**(a) The geometries are wrong.** The brief asks for headlines at L = 6, 8, 17, 36.
**None of those four is in the round.** The scored case list is
L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100. L = 6/8/17/36 are the geometries of
the older `bench/geom` `panel_r*` series (they appear in `docs/CURATION.md`'s
promotion example as `L17_rader L8_batchsimd L36_pfa`). I score the eleven
geometries actually measured. Nothing at L = 6, 8, 17 or 36 exists to report.

**(b) The machine is wrong, and this removes item 4's explanation.** The brief says
the timing pass ran on a Xeon Gold 5218 (Cascade Lake) and that implementers develop
on Sapphire Rapids, so claimed-vs-measured gaps should be attributed to a
cross-machine difference spanning 2.9× on MKL alone. `environment.txt` says
**Intel Xeon Gold 6326 @ 2.90 GHz — Ice Lake-SP**, 64 logical cores, gcc 11.4.0.
Neither Cascade Lake nor Sapphire Rapids is in play: `results/xarch_clx_r4/` and
`results/xarch_spr_r5/` are *separate* cross-architecture rounds. And every strategy
record puts development on **a80n0 itself** — held `slot_lease` cores on the same
host that scored them (100 references to a80n0 across the twelve records; `wallaby`
appears only as the AVX2-without-AVX-512 correctness cross-check). So the
claimed-vs-measured gaps in item 4 **cannot** be attributed to a machine
difference. They are same-host, same-microarchitecture window and core-state
effects, which is exactly what the records themselves claim. Item 4 is answered on
that basis.

**(c) There are no non-batched cells to headline, except one.** The brief asks for
both the non-batched and the batched case per geometry. The board measures exactly
one mode per geometry: L = 10…50 batched (B = 64 down to 4), L = 100 non-batched
(B = 1). There is no B = 1 board measurement at any L below 100. I give the batched
board numbers, the single non-batched board number at L = 100, and — clearly
separated and clearly labelled — the B = 1 figures the implementers self-report,
which are *not* round measurements and were not taken under the harness's
correctness gate.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

Batched, from the board. "Best library" is the fastest of
{mkl_dfti, mkl2026_dfti, fftw3_×4, ducc0_c2c} in that cell.

| geometry | mode | fastest correct panel entry | best library | speedup |
|---|---|---|---|---|
| L=10 | B=64, m=1000 | **gen_batchlane 1.155 µs** (43.13 GF/s) | mkl_dfti 4.569 µs | **3.96×** |
| L=12 | B=64, m=600 | **gen_batchlane 1.911 µs** (48.62 GF/s) | mkl_dfti 7.765 µs | **4.06×** |
| L=15 | B=32, m=600 | **gen_pfa_small 4.406 µs** (44.89 GF/s) | mkl_dfti 16.557 µs | **3.76×** |
| L=20 | B=32, m=256 | **gen_batchlane 12.866 µs** (40.31 GF/s) | fftw3_measure 44.810 µs | **3.48×** |
| L=25 | B=16, m=256 | **gen_powp 31.971 µs** (34.04 GF/s) | fftw3_measure 108.193 µs | **3.38×** |
| L=27 | B=16, m=200 | **gen_powp 44.544 µs** (31.52 GF/s) | mkl_dfti 144.394 µs | **3.24×** |
| L=31 | B=16, m=140 | **gen_rader 84.801 µs** (26.11 GF/s) | ducc0_c2c 719.293 µs | **8.48×** |
| L=32 | B=8, m=250 | **gen_pow2 56.524 µs** (43.48 GF/s) | mkl_dfti 172.086 µs | **3.04×** |
| L=40 | B=8, m=128 | **gen_pfa_large 160.560 µs** (31.82 GF/s) | mkl2026_dfti 404.415 µs | **2.52×** |
| L=50 | B=4, m=128 | **gen_powp 415.524 µs** (25.47 GF/s) | mkl_dfti 943.923 µs | **2.27×** |
| L=100 | **B=1 (non-batched)**, m=64 | **gen_pfa_large 4570.267 µs** (21.81 GF/s) | mkl_dfti 7797.120 µs | **1.71×** |

Every geometry is won by a panel entry, and the panel's margin over the best library
decays monotonically with volume — 4.1× at L=12 down to 1.71× at L=100 — except at
L=31, where the libraries have no prime-length path and collapse to 8.5× behind.

**Four of the eleven headlines are statistical ties, not orderings.** At L=10 the
top two are 1.155 vs 1.156 µs inside a **14.0 %** run spread; at L=15, 4.406 vs
4.412; at L=20, 12.866 vs 12.867 inside 1.5 %; at L=12, 1.911 vs 1.917.
gen_batchlane and gen_pfa_small are indistinguishable at all four cells they share,
and gen_batchlane's own record concedes their L=20 paths "are now algorithmically
identical … that cell is window luck, not code". Do not read the L=10/12/15/20 row
order as a result. Similarly L=50 (gen_powp 415.524 vs gen_pfa_large 420.018, 1.1 %)
and L=100 (gen_pfa_large 4570.267 vs gen_powp 4617.510, 1.0 %, the latter with an
8.3 % spread) are ties.

### Non-batched (B = 1) — one board number, the rest self-reported

L=100 above is the only non-batched cell on the board. The following B = 1 figures
come from implementer records, on the same host but outside the harness; they are
**not** round measurements and carry no correctness gate:

| geometry | entry | self-reported B=1 µs/transform | source |
|---|---|---|---|
| L=12 | gen_race demo | 4.560 | `gen_race.md` r6 table |
| L=12 | gen_planner | 4.50 | `gen_planner.md` |
| L=12 | gen_layout | 9.57 | `gen_layout.md` |
| L=25 | gen_powp | 40.055 | `gen_powp.md` |
| L=27 | gen_powp | 50.453 | `gen_powp.md` |
| L=27 | gen_planner | 110.4 | `gen_planner.md` |
| L=31 | gen_rader | 84.80 | `gen_rader.md` |
| L=31 | gen_planner | 190.0 | `gen_planner.md` |
| L=32 | gen_pow2 | 56.32 | `gen_pow2.md` |
| L=32 | gen_layout | 201.3 | `gen_layout.md` |
| L=50 | gen_powp | 486.5 (hot, sd 3.9 %) | `gen_powp.md` |
| L=50 | gen_planner | 574.2 | `gen_planner.md` |

The shape worth noting: gen_powp at L=27 goes 44.5 → 50.5 µs from B=16 to B=1
(+13 %) and gen_planner 60.4 → 110.4 (+83 %), while gen_pow2 at L=32 is flat
(56.5 → 56.3) and gen_rader at L=31 is flat (84.8 → 84.8). The batch-lane and
split-group forms depend on batch to fill lanes; the custody/fold forms do not.
**If a future round is graded at B = 1, the mid-L standings reorder.** That is an
untested exposure, not a finding.

---

## 2. What changed since gen_r5, per geometry

No geometry regressed structurally. The panel improved at seven of eleven cells and
was flat at four. Best panel entry, r5 → r6:

| geometry | r5 best | r6 best | delta |
|---|---|---|---|
| L=10 | gen_pfa_small 1.152 | gen_batchlane 1.155 | flat (tie, inside 14 % spread) |
| L=12 | gen_pfa_small 1.914 | gen_batchlane 1.911 | flat (tie) |
| L=15 | gen_pfa_small 4.406 | gen_pfa_small 4.406 | flat (bit-identical code) |
| L=20 | gen_pfa_small 13.072 | gen_batchlane 12.866 | **−1.6 %** |
| L=25 | gen_powp 31.352 | gen_powp 31.971 | +2.0 % (window; code = r5) |
| L=27 | gen_powp 51.444 | gen_powp 44.544 | **−13.4 %** |
| L=31 | gen_rader 84.668 | gen_rader 84.801 | flat (bit-identical code) |
| L=32 | gen_pow2 56.472 | gen_pow2 56.524 | flat (bit-identical code) |
| L=40 | gen_pfa_large 160.240 | gen_pfa_large 160.560 | flat (code unchanged) |
| L=50 | gen_powp 415.637 | gen_powp 415.524 | flat (code = r5) |
| L=100 | gen_pfa_large 4531.445 | gen_pfa_large 4570.267 | +0.9 % (window) |

The frontier barely moved. **All the round's real movement is in the second and
third ranks**, and it is large:

| entry | cells improved | range |
|---|---|---|
| **gen_twiddle** | 10 of 11 | **−28 % to −45 %** |
| gen_bluestein | 2 (L=20, L=40) | −20 %, −42 % |
| gen_planner | 5 (27/32/40/50/100) | −8 % to −23 % |
| gen_race demo | 5 (27/32/40/50/100) | −5 % to −26 % |
| gen_layout | 4 (32/40/50/100) | −13 % to −17 % |
| gen_powp | 1 (L=27) | −13.4 % |

gen_twiddle is the round's outlier and deserves the sentence: its
register-resident whole-level codelets for r = 2/3/4/5 moved it from last-or-near-last
at every cell to **beating the best library at seven of eleven** (L=20, 25, 27, 31,
40, 50, 100), including L=100 where it now edges MKL 2022 (7734.618 vs 7797.120 µs).
It went from 10.359 → 6.249 at L=10, 337.387 → 195.573 at L=32, 611.968 → 339.232 at
L=40, 10725 → 7735 at L=100. Its record claims the change is bit-identical; the board
confirms L=31 as the untouched control (267.265 → 268.617, wash) exactly as designed.

### Regressions

**No structural regression.** Every adverse move is ≤ 4 %, is in a cell whose code
did not change, and was pre-flagged by its own implementer as window heat:

| entry | cell | r5 → r6 | status |
|---|---|---|---|
| gen_race demo | L=20 | 18.405 → 19.145 (+4.0 %) | worst adverse move on the board |
| gen_race demo | L=25 | 40.223 → 41.248 (+2.5 %) | record pre-flagged "25/31 read flat-to-worse in my windows … I claim window heat, not regression" |
| gen_layout | L=31 | 195.931 → 200.777 (+2.5 %) | record: "odd: unchanged — no code-layout regression (checked)" |
| gen_powp | L=25 | 31.352 → 31.971 (+2.0 %) | record: "window; code = r5" |
| gen_layout | L=27 | 122.481 → 124.086 (+1.3 %) | record: "odd: unchanged" |
| gen_bluestein | L=10 | 13.275 → 13.410 (+1.0 %) | parity control, path untouched |
| gen_pfa_large | L=100 | 4531.4 → 4570.3 (+0.9 %) | code unchanged at instruction level |

Each of these was predicted in the record *before* the board was drawn, which is the
strongest form of the claim available. I accept all of them as window effects.

### The largest fact about this round is what it did **not** measure

Nine of the twelve records state that round 6 was expected to score **"three
never-announced surprise sizes in 14..127"** against the assembled library rather
than the r5 acceptance cells. The scored case list is **identical to r5's** — same
eleven L, same B, same chain m. So the round's dominant engineering effort is
entirely absent from the board:

| entry | r6 work | scored? |
|---|---|---|
| gen_pfa_small | module set → 19 modules + nested GT-PFA; **53 new sizes**, 68 total | **no** |
| gen_pfa_large | **16 new coverage sizes** (44…117), DFTODDM + DFT8M modules, GEN_LEAN mode | **no** |
| gen_rader | **even-h split Rader for 13 primes** (p ≡ 1 mod 4, 13…113), −14 % to −41 % | **no** |
| gen_batchlane | DFT7 module + safe placement → **14/21/28/35** at batch-lane speed | **no** |
| gen_powp | 49/81/121/125 | **no** |
| gen_pow2 | **L=16 −3.5 %** (its entire r6 change; 32/64 shipped bit-identical) | **no** |
| gen_planner / gen_race | surprise drills at 14/45/48/54/61/63/96/104/127 | **no** |
| gen_bluestein | non-pow₂ M grid (M = 48/80/96/160/192) | **partly** — visible at L=20 (−20 %) and L=40 (−42 %), both confirmed |
| gen_layout, gen_twiddle, gen_dense_prime | all-L work | yes |

Roughly half the round is unscored, and gen_pow2's *only* r6 change is invisible.
The largest single claim on the table — gen_rader's −25 % to −41 % across thirteen
primes — has no independent measurement whatsoever. This is a harness/protocol
failure, not an implementer failure, and it is the subject of item 6.

---

## 3. Adversarial pass: correctness, builds, crashes, absences

I went looking for a fast wrong answer. There isn't one on this board.

**Correctness: clean, and verified independently of the leaderboard's own column.**
`check.log` is 658 lines, **0 occurrences of FAIL**, all PASS. Single-call rel L2
across all entries and cells is 2.3e-16…5.5e-16 against tol 1e-12 (≥ 1800× margin);
graded map-chains 2.0e-14…3.6e-13 against tol 1e-10; two-step gates 7.4e-16…3.2e-15
against tol 3e-14. Every leaderboard row reads `ok`. The chain errors track volume
and chain length in the expected way and no entry is anomalously *accurate* either
(which would suggest a short-circuited kernel). **No entry failed correctness.**

**Builds: nothing failed to build.** `build_errors.txt` exists but contains **only
two warnings and zero errors** — the filename is misleading this round:

```
impl/gen_dense_prime.c:1925  iteration 1152921504606846976 invokes undefined
impl/gen_rader.c:1620        behaviour [-Waggressive-loop-optimizations]
```

I checked both sites rather than reporting the warning text. They are the *same*
code: the scalar remainder loop after the AVX-512 map body,
`for (; i < npts; ++i) { ... zp[2*i] ... }` with `size_t i`
(`gen_dense_prime.c:1924-1930`, `gen_rader.c:1619-1625`). 1152921504606846976 = 2⁶⁰;
at that iteration `zp + 2*i` overflows any object, so gcc-11 proves the loop cannot
legally reach it and uses that to bound the trip count. The inferred bound is
`npts < 2⁶⁰`; the largest `npts` in this round is 10⁶. **The UB is unreachable, gcc
does not delete the loop, and both entries pass every correctness gate — including
gen_rader at L=31, which is a headline winner.** So: not a defect that affects any
measured number, but it is live UB in the winner's map tail and it should be silenced
(index with a plain `size_t` byte offset, or hoist `2*i` into its own induction
variable) rather than left for a future compiler to exploit differently. Flagging it,
not disqualifying on it.

**Crashes/hangs: one, and it is the harness floor, not an implementation.**
`failures.txt` contains exactly three lines:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout`. `baseline_matrix` is the deliberately-O(L⁴) library-free
reference; at volume 10⁶ it projects to ~3.4 s per transform (extrapolating its
L=50 rate) against a chain of 64, so the timeout is expected and it is correctly
absent from the L=100 block. **Consequence worth stating: the L=100 block has no
library-free floor.** Every L=100 number is validated only against the numpy/
long-double oracle in `check.py`, not against the in-harness reference. That is
adequate but it is one fewer independent check at the largest geometry.

**Absences: all 54 are declared `supports()` declines, none is a crash.**
`timing.err` is 220 lines, every one of the form `<entry>: does not support L=<n>`.
Cross-checking the leaderboard against `sort -u timing.err` accounts for every
missing row exactly, with no residue:

| entry | declines | cells present |
|---|---|---|
| gen_pfa_large | 10, 12, 15, 20, 25, 27, 31, 32 | 40, 50, 100 |
| gen_pow2 | all but 32 | 32 |
| gen_rader | all but 31 | 31 |
| gen_powp | 10, 12, 15, 20, 31, 32, 40 | 25, 27, 50, 100 |
| gen_batchlane | 25, 27, 31, 32, 40, 50, 100 | 10, 12, 15, 20 |
| gen_pfa_small | 25, 27, 31, 32, 40, 50, 100 | 10, 12, 15, 20 |
| gen_dense_prime | 25, 27, 32, 40, 50, 100 | 10, 12, 15, 20, 31 |
| gen_layout, gen_planner, gen_race, gen_twiddle, gen_bluestein | none | all 11 |

Class entries declining out-of-class geometries is by design and the harness records
it honestly. **Nothing crashed. Nothing is silently missing. No entry is unaccounted
for.** The one thing a decline *does* hide is that a class entry cannot be embarrassed
outside its class — a point that matters for item 6.

**One methodological hazard, and it is the harness's, not an entry's.**
gen_powp's record documents that its r5 board number at L=27 (51.444 µs) was **the
scoring race installing a losing candidate**: "the monitor's quiet-window cold race
installed l27-ip0 (wisdom: 58.63 µs/vol trial, margin −0.6 %, 'tie'), i.e. soa was
mis-ranked OUT in the scoring race itself." The r6 board confirms the diagnosis —
44.544 µs, −13.4 %, after they added a playoff that races on the graded chain instead
of a 2-step trial. Combined with gen_race's round-end `gr_wisdom_drop_prefix` (both
hosts stripped to 0 entries, so the monitor **cold-races** every verdict at scoring
time), this means: **a mis-pick in the scoring window is a live failure mode for
every raced entry on this board, and it cost 13 % at one cell last round.** It is
invisible in the leaderboard — a mis-picked candidate still reports `ok`. This is the
one place where "a fast wrong answer" has an analogue here: not a wrong *answer*, but
a wrong *plan* scored as if it were the entry's best.

---

## 4. Claimed vs. measured

Per item 0(b), the brief's Sapphire-Rapids-vs-Cascade-Lake explanation does not
apply: implementers and monitor ran on the same a80n0 Ice Lake-SP host. The
attribution below is therefore same-host window and core-state effects, which is
what the records themselves claim and, in several cases, predicted in advance.

**Agreement is unusually good. There is no large unexplained discrepancy.**

| entry | cell | claimed | board | gap |
|---|---|---|---|---|
| gen_twiddle | all 11 | −28 % … −44 % vs r5 | −28 % … −45 % | **every cell reproduces** |
| gen_planner | L=27 | 60.2–60.3 | 60.373 | +0.1 % |
| gen_planner | L=32 | 108.3–110.0 | 105.717 | board 2.4 % faster |
| gen_planner | L=50 | 565.0–568.3 | 566.668 | exact |
| gen_powp | L=27 | 45.254 | 44.544 | board 1.6 % faster |
| gen_powp | L=25 | 32.103 | 31.971 | 0.4 % |
| gen_pfa_small | L=10/12/15/20 | 1.156 / 1.916 / 4.42 / 12.99 | 1.156 / 1.917 / 4.406 / 12.867 | ≤ 1 % |
| gen_pow2 | L=32 | 55.40 | 56.524 | board 2.0 % slower |
| gen_layout | all | 5.17 … 12357 | 5.200 … 12099.9 | ≤ 2 %, mostly board-faster |
| gen_rader | L=31 | 84.82/84.95/86.02 | 84.801 | exact, bit-identical to r5 as claimed |
| gen_bluestein | L=20 / L=40 | 84.5 / 682 | 82.906 / 620.182 | board 2 % / 9 % faster |

Two systematic patterns, both benign:

**(i) gen_race's demo reads 5–13 % slower in its own windows than on the board** —
L=31 149.50 claimed vs 138.928 measured, L=40 268.02 vs 236.997, L=100 5565.4 vs
5200.083, L=32 111.33 vs 105.534. The direction is the honest one (they
under-promised). Its record names the cause: "31/25 read flat-to-worse vs their r5
boards in my windows (sd up to 22 % at 31; MKL moved with it: 866–1026 vs the board's
857)". The board's own MKL at L=31 is 848.537, i.e. gen_race's *library control*
moved with its entry, which is the signature of a hot window rather than a code
difference. Accepted as window heat, and the record's decision to publish the
pessimistic number rather than a quiet-core best is the right practice.

**(ii) gen_pow2 reports bimodal L=16 windows** — "~6.4–6.6 vs ~7.3–7.6 µs states,
flipping mid-round twice". L=16 is unscored so this cannot be checked, but the same
core-state bimodality is the most plausible source of the board's own 14.0 % spread
at L=10 (gen_batchlane) and 13.8 % (gen_race), and of the 8.3 % at L=100 (gen_powp).
On this host, sub-2 % orderings are not resolvable in a single sweep.

**Two claims that do NOT reproduce, both small and both in the same entry:**

- **gen_layout claims L=32 is its "first MKL win at 32 for this floor"** (176.9 vs
  MKL 184.1 in its window). On the board it is **172.732 vs mkl_dfti 172.086 — MKL
  wins by 0.4 %.** A tie, not a win. Their absolute number was if anything better
  than claimed; MKL's was 6 % better in the scoring window. The claim is
  window-dependent and should not be quoted.
- **gen_layout's L=50 "beats MKL"** survives only as a tie: 941.267 vs 943.923, a
  0.3 % margin inside a 1.6 % spread. Their L=40 (391.754 vs 405.615, −3.4 %),
  L=25, L=27 and L=31 MKL wins are real and large.

**One claim worth confirming because it was aggressive:** gen_twiddle's "the demo now
BEATS MKL at 100" reproduces — 7734.618 vs 7797.120 µs — but by **0.8 %**. It is a
win; it is not a margin to build on.

---

## 5. Which LITERATURE.md §4 open question moved

### §4.6 — *model versus search for the instruction schedule*: moved hardest, and the panel's answer is now "search, but search the workload you are graded on"

§4.6 sets up FFTW's claim that a cache-oblivious schedule needs no search phase
against §06 §6.1's rebuttal (genfft's spill-optimality proof holds only at powers of
4) and §06 §3.4's ">3× runtime spread across 2000 random flag combinations, and -O3
sometimes slower than -O1". This round supplies four CPU data points, all on the
rebuttal's side:

1. **Hand-set boundaries were wrong by 10–11 % on their first race.** gen_race added
   a third race stage over the shipped engine's runtime `fusemap` boundary — an
   ICX-tuned constant — and it "caught two wrong boundaries on its FIRST outing":
   L=12 B=1 beats the `L>12` default by **11 %**, L=14 B=8 beats the `L³>1728`
   default by **10.6 %**, both non-tie. Elsewhere it confirmed the default with 3.5–10 %
   margins. A model-chosen constant, tuned on a different microarchitecture, was
   wrong at the boundary by more than any arithmetic change in the round.
2. **Search found a tree no model proposed.** gen_planner's new @s4 split-group level
   took L=27 by −24 % (board: 78.4 → 60.4 µs), and at L=40 "the race found a NEW
   tree under the fused exit". The winning L=27 plan `c3(c3(d3))@s4` retires r5's
   verdict that pv beat @s2 at 27.
3. **The compiler silently withdrew an optimisation nobody had asked for, worth
   5–9 %.** gen_planner: growing `pln_s8_step` pushed it past gcc-11's inlining budget,
   which stopped constprop-cloning `pln_foldw` into map/no-map forms, costing **5–9 %
   at L=31 with bit-identical outputs and no change to the hot path.** This is §06
   §3.4's phenomenon one level deeper — not a *flag*, but an inlining decision — and
   it is invisible to any model of the schedule. Their lesson generalises and I
   endorse it: *after growing any dispatch function, diff the symbol table against
   the previous build.*
4. **The counter-weight, which is the round's real contribution here.** gen_powp's
   L=27 cell shows search *losing* 13 % because the race measured a 2-step trial
   while the board graded a 200-step chain: interleaved trials evict the SoA arenas
   between rounds, so SoA re-paid a refill every trial round that the graded chain
   pays once. Their fix — a playoff on the graded shape — recovered it (board:
   51.444 → 44.544). So §4.6's "search the schedule" is right, and this round adds
   the necessary qualifier: **a search is only as good as the proxy it times, and a
   short-trial proxy can invert the ranking of a cache-resident plan.**

### §4.3 — *is axis fusion worth 3× or 3 %*: the re-opened L2↔DRAM case was tested twice, and it is **not** a general win

§4.3's re-opened paragraph names one construction as "the largest untried structural
move on the board": *tile the batch so a tile fits L2, then run all three axes inside
the tile*, on the strength of a 7× L2↔DRAM bandwidth gap. Two entries built it
independently this round and the verdict is narrow:

- **gen_planner's @s4 is exactly that construction** — blocks of P pencils staged
  through an L1 buffer, one volume sweep per axis instead of @s2's two, no ping-pong
  buffer. It wins **−24 % at L=27** (volume 19 683, working set 9.61 MiB, i.e. above
  L2) — a real, large, reproduced win (same-core alternating A/B: 59.8–60.1 vs 77.7–77.8
  forced-pv, "1.30×, every round"). But it **loses at 32, 40, 45, 48, 54, 63, 96**:
  "above vol ~20k the group's 16× working set loses to the pv volume-major chain's L2
  residency no matter how few sweeps the group makes."
- **gen_dense_prime tested the same idea and rejected it at every size**, with the
  mechanism named: custody costs +2 % at L=17 growing to +4–8 % at L=29, and lazy-map
  +3–10 % everywhere. Their diagnosis is the sharper of the two: *"the custody z-phase
  must walk the volume in the x-pass's direction — L rows per block at PLANE stride —
  where the whole-volume z-pass streams rows contiguously … custody pays only when it
  deletes a MISS stream, never inside a cache level the OoO already covers."*

So the answer to §4.3's re-opened case, on Ice Lake-SP: **the fusion win is not the
bandwidth gap, it is the number of volume sweeps you delete.** @s4 wins at L=27 not
because a tile fits L2 but because it halves the per-axis sweeps of the form it
replaced; where the un-tiled form already touches the volume once, tiling costs
working set and loses. Tolmachev's rule (§07 §1.6 — avoided passes × bandwidth gap)
survives a second round with CPU numbers attached, and the "tile for L2" framing does
not add to it. Both entries also record the negative explicitly so the next panel
does not re-derive it, which is the behaviour §4.3 asks for.

### §4.5 — *padding*: confirmed again, now applied at plan birth

§4.5's memorable rule (*with a one-cache-line granule, pad so every stride is an odd
number of cache lines*) is now built into plan construction rather than per-size
tables: gen_planner's @s4 stage row stride is "16P + 8 doubles = an odd number of
cache lines, so pass 2's `m*srs` stride never lands 4K-uniform … applied at plan
birth", and gen_batchlane's four new sizes "all keep plane bytes == 256 (mod 4096)".
Confirmatory, not new evidence — but the rule has now survived being generalised from
per-size constants to a plan constructor, which is a stronger form of the claim.

### Untouched

**§4.1** (register pressure in batch-vectorised L=6/L=8 codelets) — L=6 and L=8 are
not in this round at all. **§4.2** (L=17: dense-symmetric vs Rader vs Winograd) —
*newly buildable but still unmeasured*: gen_dense_prime now reports L=17 at
27.5–28.7 µs (B=4, m=8) for the dense conjugate-fold, and gen_rader's new even-h
split covers p ≡ 1 mod 4 including 17 — so for the first time both sides of §4.2(a)
exist in code, and neither is on a board. **§4.7** (vector-radix) — nobody attempted
it, correctly.

---

## 6. The single highest-value thing the next round should attack

**Before anything per-geometry: score what was built.** Half of gen_r6's engineering
is unmeasured (item 2), because nine records prepared for a surprise-size,
assembled-library round and the sweep re-ran r5's eleven acceptance cells. The
highest-value action available is not new code — it is **a sweep over the coverage
sizes the panel already shipped and gated**: gen_rader's thirteen even-h primes
(37/41/53/61/73/89/101/103/113, claimed −14 % to −41 %), gen_pfa_small's composites
with prime factors 17…31 (it claims L=34 is 4.7× faster than MKL, which if true is
the largest library margin anyone has claimed at a composite), gen_pfa_large's
sixteen new sizes, gen_batchlane's 14/21/28/35, and gen_pow2's L=16. Every one of
those claims is currently a self-report with no independent measurement. A second
action, nearly free: **fix the race-proxy exposure** by making the scoring race time
the graded chain (gen_powp's playoff, generalised), since a mis-pick already cost
13 % at one cell and is invisible on the board.

Per geometry, given that:

- **L=10, 12 (B=64) — stop tuning; separate the twins or merge them.**
  gen_batchlane and gen_pfa_small are inside run spread at all four shared cells and
  their own records call L=20 "algorithmically identical". Two implementers are
  spending rounds re-measuring each other. The productive split is by *contribution*:
  gen_batchlane owns **safe placement** (the stage-1 store permutation that makes
  stage-2 in-place at any coprime (P,Q) — the idea that makes new sizes mechanical),
  gen_pfa_small owns **breadth** (68 sizes). Have one of them attack the thing neither
  has: at 43 GF/s against ~48 at L=12, L=10's 14 % run spread says the cell is
  bound by something outside the codelet. Measure it (`ld_blocks_partial.address_alias`
  per §4.5, plus L1 miss counters) rather than tuning the kernel again.
- **L=15, 20 (B=32) — attack gen_twiddle's mechanism, not the leader.** The leader is
  flat and tied. But gen_twiddle went 55.4 → 36.0 at L=20 with register-resident
  whole-level codelets and now beats FFTW, MKL, gen_layout and gen_dense_prime there.
  That mechanism has not been tried inside the PFA leaders. Whether whole-level
  register residency composes with twiddle-free PFA pencils is the open question at
  these two cells and it is worth one round.
- **L=25, 27 (B=16) — the split-group boundary.** L=27 was the round's biggest
  frontier move (−13.4 %) and also where @s4 won by 24 %. gen_powp leads at 44.5 with
  a CT/SoA form; gen_planner's @s4 group form is at 60.4. The two have never been
  crossed: run @s4's staged single-sweep construction on gen_powp's p^k trees. The
  measured boundary ("@s4 loses above vol ~20k") sits between L=27 (19 683) and
  L=32 (32 768) — nail it, because L=25 (15 625) should be on the winning side and
  gen_powp has not tried it.
- **L=31 (B=16) — settle §4.2's dense-vs-Rader crossover, which is now buildable.**
  gen_rader (84.8) leads gen_dense_prime (120.7) by 1.42×, and this is the only cell
  where a genuinely different algorithm is the runner-up. gen_rader's new even-h split
  covers p ≡ 1 mod 4 (5m² conv FMA vs 8m²) but 31 ≡ 3 mod 4 and keeps the r5 path,
  bit-identical two rounds running. Two concrete moves: apply the even-h machinery's
  Karatsuba/CRT structure to the odd-h case at 31, and put dense-fold and Rader
  head-to-head at **17, 19, 23, 29** where dense already leads by 15–20 % — that is
  §4.2(a), on a board, for the first time.
- **L=32 (B=8) — gen_pow2 is at its documented wall; the round's own record says so.**
  56.5 µs, 3.04× MKL, bit-identical for two rounds, with a named ceiling
  ([port ∥ L2], < 8 % residual). Do not spend a round on 8 %. The valuable exposure
  is that gen_pow2's *only* r6 work (L=16, −3.5 %) is unscored and L=64 sits at an
  L3-bandwidth wall (32.5 GB/s). **Score 16 and 64.** If the frontier at 32 must be
  attacked, the lever is gen_twiddle's −42 % at this cell (337 → 196), which came
  from register-resident whole-level codelets — the one structural idea gen_pow2's
  custody engine has not absorbed.
- **L=40 (B=8) — the gap between the leader and the movers is now the story.**
  gen_pfa_large has been flat at 160.5 for two rounds (its own code "ships UNCHANGED,
  verified at the instruction level") while gen_planner came −15 % to 239.9,
  gen_twiddle −45 % to 339.2, gen_bluestein −42 % to 620.2 and gen_layout −14 % to
  391.8. Everything is converging on a leader that has stopped moving and whose setup
  cost is **1.176 s** — 400× any other entry at this cell, and by far the worst on the
  board. That setup cost is unexamined and is the one number at L=40 that would
  disqualify the leader in any latency-sensitive use.
- **L=50 (B=4), L=100 (B=1) — the library margin is thinnest here and the ties are
  unbroken.** 2.27× and 1.71× over MKL, versus ~4× at small L. Both cells are
  two-way ties (gen_powp/gen_pfa_large, 1.0–1.1 % apart, one with 8.3 % spread) that
  three rounds have failed to resolve — stop trying to resolve them and attack the
  margin instead. Concretely: **L=100 is the one cell where the fused-map exit
  *loses* (+8–10 %, 3/3 pairs, gated off above L=80) and where the fused-exit boundary
  81…99 is entirely unmeasured.** gen_planner's own diagnosis — the blockwise exit
  re-reads the c plane L/4 times per plane, 160 KB × 25 at L=100 — points at a
  c-traversal order problem, not a fusion problem. Fix the walk and the gate may
  disappear. This is also the only geometry with no library-free floor (item 3), so
  restoring a reference at L=100, even a slow one with a longer timeout, is worth
  doing for its own sake.

---

## 7. What to keep

Against `docs/CURATION.md`'s four grounds. This is a wide promotion — eleven of
twelve — and that needs justifying rather than glossing: gen_r6 was a coverage and
library round in which the frontier barely moved but the *second rank* moved 28–45 %,
and nearly every entry either won a geometry, beat a library by a wide margin, or
produced a documented negative result with a mechanism. I would rather hand the next
panel one clearly-labelled exclusion than eleven near-misses.

**Ground 1 — fastest correct entry per geometry (always).** Six entries:
`gen_batchlane` (L=10, 12, 20), `gen_pfa_small` (L=15), `gen_powp` (L=25, 27, 50),
`gen_rader` (L=31), `gen_pow2` (L=32), `gen_pfa_large` (L=40, 100).
gen_batchlane and gen_pfa_small are tied at all four of their cells (item 1) but are
not near-duplicates *as contributions*: safe placement vs. 68-size breadth. Both kept,
with the tie recorded so the next panel does not treat the row order as a result.

**Ground 2 — structurally different runner-up.** `gen_dense_prime` at L=31 (120.708,
1.42× behind gen_rader) is the only cell where a different algorithm — dense
conjugate-fold GEMM against Rader convolution — holds second place. It is outside
CURATION's "~20 %" example, but it is precisely the situation that example describes,
it is the code that makes §4.2 settleable, and it qualifies independently on grounds
3 and 4. Kept. `gen_planner` is the structural runner-up at L=27/32/40/50 (1.32–1.49×)
and, more importantly, is the substrate that produced most of the round's movement.

**Ground 3 — instructive failures whose record documents the killing number.**
- `gen_dense_prime` — the round's best negative result: custody +2 % (L=17) → +8 %
  (L=29) and lazy-map +3–10 % at five sizes, rejected with the mechanism named
  (*walk direction, not store-retire distance*) and the machinery preserved behind
  knobs. This is what stops the next panel spending a round rediscovering it.
- `gen_planner` — three measured negatives: the fused exit at L=100 (+8–10 %, 3/3,
  gate 12 < L ≤ 80), @s4 losing above vol ~20k, and the gcc-11 un-inlining incident
  (5–9 % at L=31 from a source change outside the hot path).
- `gen_pfa_small` — module 15 nested-PFA rejected (+10 % at L=30) while module 21
  nested wins (−12 % at 42), i.e. the op-count cut loses to the fold's straight-line
  FMA stream below a threshold. A clean, size-dependent negative.

**Ground 4 — anything that beat a library baseline.** `gen_twiddle` beat the best
library at **seven** cells (L=20, 25, 27, 31, 40, 50, 100) after a −28 % to −45 %
round, and its record reproduces on the board cell for cell — the round's most
honest and largest single-entry improvement. `gen_layout` beat the best library at
L=20, 25, 27, 31, 40 and tied at 32/50, and is the layer nearly everyone `#include`s
(THP arenas, the measured pitch picker). `gen_bluestein` beat every library at L=31,
and its non-pow₂ convolution grid (M = 48/80/96/160/192, cutting M up to 37.5 % below
next_pow2) delivered −20 % at L=20 and −42 % at L=40, both confirmed on the board; it
is also the only any-L entry and the library's universal fallback, which is worth
having written down even though it is last at most cells.

**Excluded: `gen_race`.** Two independent grounds, both from its own record. (i) Its
library is **frozen — "zero changes, no `gr_*` signature or wisdom-format change"** —
so promoting it puts no new code in front of the next panel. (ii) Its demo *is*
gen_planner's r6 trunk; on the board the two are within 1 % at eight of eleven cells
(L=10 1.424/1.418, L=12 2.471/2.467, L=15 5.591/5.518, L=31 138.928/138.210, L=32
105.534/105.717, L=50 567.466/566.668), which is CURATION's near-duplicate exclusion
almost exactly. Its genuinely valuable finding this round — the `fm` race catching two
hand-set boundaries by 11 % and 10.6 % — is a *result*, recorded in §5 above and in
`strategies/gen_race.md`, and it is preserved there whether or not the code is
promoted. Its exclusion is not a judgement on the work.

No entry is excluded for failing correctness, failing to build, or a missing strategy
record: all twelve records exist and all are substantial (649–825 lines).

---

*Round monitored 2026-08-25. Verdict is on the eleven geometries actually measured
(L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100) on Intel Xeon Gold 6326
(Ice Lake-SP), not on the L = 6/8/17/36 Cascade Lake premise in the monitor brief —
see §0.*
