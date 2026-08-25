# Round gen_r7 — monitor's verdict

Measured 2026-08-25T10:21:56-04:00, host a80n0.lqcd.mit, slurm job 438682.
All numbers below are from `results/gen_r7/leaderboard.txt` unless explicitly
labelled as an implementer self-report.

---

## 0. Three corrections to the monitor brief, before any judgement

The brief I was handed (preserved verbatim in `monitor_prompt.txt`) is the same
stale template that was handed to the gen_r6 monitor, and it carries the same
three premises this round's own files contradict. Recording them first because
two of them would silently corrupt items 1 and 4 if followed.

**(a) The geometries are wrong.** The brief asks for headlines at L = 6, 8, 17, 36.
**None of those four is in the round.** `cases.txt` — the frozen generalize
acceptance suite — is L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100, and the
board measures exactly those eleven. L = 6/8/17/36 are the geometries of the
older `bench/geom` `panel_r*` series (`bench/geom/cases.txt` line 4: `6:1 6:64
6:4096 6:32768` …; they also appear in `docs/CURATION.md`'s promotion example as
`L17_rader L8_batchsimd L36_pfa`). I score the eleven geometries actually
measured. Nothing at L = 6, 8, 17 or 36 exists to report.

**(b) The machine is wrong, and this removes item 4's explanation.** The brief
says the timing pass ran on a Xeon Gold 5218 (Cascade Lake) and that implementers
develop on Sapphire Rapids, so claimed-vs-measured gaps should be attributed to a
cross-machine difference. `environment.txt` says **Intel Xeon Gold 6326 @ 2.90 GHz
— Ice Lake-SP**, 64 logical cores, gcc 11.4.0. This is not a discrepancy to
adjudicate: `PANEL_BRIEF.md` § "Scoring and the endgame" states the score *is*
Ice Lake and that Cascade Lake and Sapphire Rapids are **advisory** reruns, and
those reruns are separate rounds on disk — `results/xarch_clx_r6/` (p52n1, Gold
5218) and `results/xarch_spr_r5/` (wallaby, Gold 6448Y). Every strategy record
puts this round's development on **a80n0 itself**, under held `slot_lease` cores
on the same host that scored it. So the claimed-vs-measured gaps in item 4
**cannot** be attributed to a machine difference. They are same-host,
same-microarchitecture window and core-state effects. Item 4 is answered on that
basis.

**(c) There are no non-batched cells to headline, except one.** The brief asks
for both the non-batched and the batched case per geometry. `cases.txt` fixes
exactly one (L, B) per geometry: L = 10…50 batched (B = 64 down to 4), L = 100
non-batched (B = 1). There is no B = 1 board measurement at any L below 100 —
confirmed against the raw `t_*_L*_B*_r*.json` set, which contains exactly the
eleven `L*_B*` combinations and no others. I give the batched board numbers, the
single non-batched board number at L = 100, and — clearly separated and clearly
labelled — the B = 1 figures the implementers self-report, which are *not* round
measurements and were not taken under the harness's correctness gate. This
matters more than usual this round: `PANEL_BRIEF.md`'s standing rules say
"batched and B=1 both matter", and the round's single largest measured
improvement (gen_pfa_small, −23 to −37 %) is entirely in the B = 1 path the
board cannot see.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

Batched, from the board. "Best library" is the fastest of
{mkl_dfti, mkl2026_dfti, fftw3_×4, ducc0_c2c} in that cell.

| geometry | mode | fastest correct panel entry | best library | speedup |
|---|---|---|---|---|
| L=10 | B=64, m=1000 | **gen_batchlane 1.147 µs** (43.44 GF/s) | mkl_dfti 4.574 µs | **3.99×** |
| L=12 | B=64, m=600 | **gen_batchlane 1.915 µs** (48.53 GF/s) | mkl2026_dfti 7.760 µs | **4.05×** |
| L=15 | B=32, m=600 | **gen_batchlane 4.381 µs** (45.15 GF/s) | mkl_dfti 16.459 µs | **3.76×** |
| L=20 | B=32, m=256 | **gen_batchlane 12.855 µs** (40.34 GF/s) | fftw3_patient 44.912 µs | **3.49×** |
| L=25 | B=16, m=256 | **gen_powp 30.882 µs** (35.24 GF/s) | fftw3_patient 108.582 µs | **3.52×** |
| L=27 | B=16, m=200 | **gen_powp 43.966 µs** (31.93 GF/s) | mkl_dfti 144.414 µs | **3.28×** |
| L=31 | B=16, m=140 | **gen_rader 84.544 µs** (26.19 GF/s) | ducc0_c2c 714.391 µs | **8.45×** |
| L=32 | B=8, m=250 | **gen_pow2 56.378 µs** (43.59 GF/s) | mkl_dfti 170.679 µs | **3.03×** |
| L=40 | B=8, m=128 | **gen_pfa_large 159.959 µs** (31.94 GF/s) | mkl2026_dfti 405.006 µs | **2.53×** |
| L=50 | B=4, m=128 | **gen_pfa_large 413.958 µs** (25.56 GF/s) | mkl_dfti 947.956 µs | **2.29×** |
| L=100 | **B=1 (non-batched)**, m=64 | **gen_pfa_large 4475.279 µs** (22.27 GF/s) | mkl_dfti 7803.986 µs | **1.74×** |

Every geometry is won by a panel entry. The margin over the best library still
decays monotonically with volume — 4.05× at L=12 down to 1.74× at L=100 — except
at L=31, where the libraries have no prime-length path and collapse to 8.45×
behind. Against r6 the ratios are flat to slightly better everywhere (L=100:
1.71× → 1.74×; L=25: 3.38× → 3.52×).

**Five of the eleven headlines are statistical ties, not orderings.**
At L=12 the top two are **1.915 vs 1.915 µs** — identical to the third decimal;
at L=10, 1.147 vs 1.155 inside a **14.0 %** run spread at the winner; at L=15,
4.381 vs 4.407 (0.6 %); at L=20, 12.855 vs 12.929 (0.6 %); at L=50,
gen_pfa_large 413.958 vs gen_powp 415.066 (0.3 %, inside a 6.1 % spread).
L=100 (4475.3 vs gen_powp 4549.6, 1.7 %) is close to one.

Two "leader changes" this round are ties and should not be reported as results:
**L=15 flipped gen_pfa_small → gen_batchlane** (0.6 %) and **L=50 flipped
gen_powp → gen_pfa_large** (0.3 %). gen_batchlane and gen_pfa_small are
indistinguishable at all four cells they share, for the third round running.

### Non-batched (B = 1) — one board number, the rest self-reported

L=100 above is the only non-batched cell on the board. The following B = 1
figures come from implementer records, on the same host but outside the harness;
they are **not** round measurements and carry no correctness gate:

| geometry | entry | self-reported B=1 µs/transform | r6 self-report | source |
|---|---|---|---|---|
| L=10 | gen_pfa_small | **2.572–2.582** (MKL 4.34–4.93 same core → 1.7×) | 3.877 | `gen_pfa_small.md` |
| L=10 | gen_planner | 2.94–2.98 | 3.17 | `gen_planner.md` |
| L=12 | gen_pfa_small | **3.357–3.375** (MKL 7.32 → 2.2×) | 5.305 | `gen_pfa_small.md` |
| L=12 | gen_race demo | 3.817 | 4.560 | `gen_race.md` |
| L=12 | gen_layout | 9.04 | 9.57 | `gen_layout.md` |
| L=12 | gen_twiddle | 10.104 | — | `gen_twiddle.md` |
| L=15 | gen_pfa_small | **10.761–10.791** (MKL 15.84–16.68 → 1.5×) | 14.031 | `gen_pfa_small.md` |
| L=20 | gen_pfa_small | **21.690–22.133** (MKL 55.05–55.24 → 2.5×) | 32.330 | `gen_pfa_small.md` |
| L=20 | gen_planner | 17.8–18.3 | 18.9 | `gen_planner.md` |
| L=25 | gen_powp | 40.02 | 40.06 | `gen_powp.md` |
| L=27 | gen_powp | 50.55 | 50.45 | `gen_powp.md` |
| L=31 | gen_rader | 85.63 (held lease) | 84.80 | `gen_rader.md` |
| L=31 | gen_dense_prime | 121.4 | — | `gen_dense_prime.md` |
| L=32 | gen_pow2 | 55.82–56.19 | 56.32 | `gen_pow2.md` |
| L=32 | gen_layout | 176.3 | 201.3 | `gen_layout.md` |
| L=40 | gen_pfa_large | 185.5 | — | `gen_pfa_large.md` |
| L=50 | gen_powp | 421.2 | 486.5 (hot) | `gen_powp.md` |
| L=50 | gen_pfa_large | 433.9 | — | `gen_pfa_large.md` |

The r6 verdict flagged the B = 1 exposure as "untested, not a finding". It is
now partly tested, by the one entry that spent its round there: gen_pfa_small's
B = 1 path went from **losing to MKL since r1** to beating it 1.5–2.5× at all
four of its sizes, and the batch-dependence gap it was closing is real (its own
r6 B=1/B=64 ratio at L=10 was 3.36×; it is now 2.24×). **This is the single
largest measured improvement in the round and the board cannot see any of it.**

---

## 2. What changed since gen_r6, per geometry

**No geometry regressed structurally.** The frontier moved at every cell but by
small amounts; all the round's large movement is again in the second and third
ranks. Best panel entry, r6 → r7:

| geometry | r6 best | r7 best | delta |
|---|---|---|---|
| L=10 | gen_batchlane 1.155 | gen_batchlane 1.147 | −0.7 % (inside a 14.0 % spread) |
| L=12 | gen_batchlane 1.911 | gen_batchlane 1.915 | +0.2 % (bit-identical path) |
| L=15 | gen_pfa_small 4.406 | gen_batchlane 4.381 | **−0.6 %** |
| L=20 | gen_batchlane 12.866 | gen_batchlane 12.855 | −0.1 % |
| L=25 | gen_powp 31.971 | gen_powp 30.882 | **−3.4 %** |
| L=27 | gen_powp 44.544 | gen_powp 43.966 | **−1.3 %** |
| L=31 | gen_rader 84.801 | gen_rader 84.544 | −0.3 % (bit-identical code) |
| L=32 | gen_pow2 56.524 | gen_pow2 56.378 | −0.3 % (bit-identical code) |
| L=40 | gen_pfa_large 160.560 | gen_pfa_large 159.959 | −0.4 % (code unchanged) |
| L=50 | gen_powp 415.524 | gen_pfa_large 413.958 | **−0.4 %** (tie flip) |
| L=100 | gen_pfa_large 4570.267 | gen_pfa_large 4475.279 | **−2.1 %** |

The frontier's real moves are **L=25 (−3.4 %)** and **L=100 (−2.1 %)**; the rest
is inside window noise or explicitly bit-identical code. Second and third rank:

| entry | cells improved | range | driver (per its record) |
|---|---|---|---|
| **gen_layout** | 7 of 11 | **−2.2 % to −22.8 %** | third-level k-fold ("quad kernel") at 4∣L + exit-map packing |
| **gen_race demo** | 8 of 11 | −1.1 % to −8.0 % | fresh salted races over planner's new fused-GT engine |
| **gen_planner** | 8 of 11 | −1.0 % to −5.9 % | fused Good-Thomas codelets + DFT7 as a hard leaf |
| **gen_bluestein** | 2 (L=50, L=100) | **−21.1 %, −7.9 %** | 7·2^k convolution grid (M = 112 / 224) |
| gen_powp | 2 (L=25, L=27) | −3.4 %, −1.3 % | 3-shear lifted twiddle rotations |
| gen_batchlane | 3 (10, 15, 20) | −0.1 % to −0.7 % | lifted DFT5 v-pair (φ · sin 36° = sin 72°) |

**gen_layout is the round's outlier and deserves the sentence.** Its quad kernel
halves the dense kernel's j-sweep again at 4∣L, and the board reads
−22.8 % at L=100 (12 099.9 → 9 343.2), −13.4 % at L=40, −10.8 % at L=32,
−8.9 % at L=10, −6.1 % at L=20. It now **beats the best library at seven of
eleven cells** (20, 25, 27, 31, 32, 40, 50) — up from five plus two ties in r6 —
and its isolated `-DGL_DEMO_NOQUAD=1` A/B attributes −13.6 % (L=32) and −23.1 %
(L=100) to the quad kernel alone. Notably, **the r6 claim that failed on the r6
board is now real**: gen_layout at L=32 is 154.051 vs mkl_dfti 170.679, a 10.8 %
win, where r6 gave it 172.732 vs 172.086 (a 0.4 % loss).

**gen_bluestein's L=50 result is the cheapest win of the round and worth
recording as method, not just as a number.** Its r6 record had *declined* the
7·2^k convolution grid on the argument that the slice "contains no graded size".
Checking that claim against `cases.txt` rather than against memory showed L=50
and L=100 both sit inside 7-slices. Re-running the same r6 machinery on the
corrected grid gave −21.1 % and −7.9 % on the board, confirmed 3/3 in its own
same-core A/B (1 394.9 vs control 1 797.9). One arithmetic re-check of a
declined idea outperformed most of the round's new engineering.

### Regressions

**No structural regression, and one adverse move I do not accept as noise.**

| entry | cell | r6 → r7 | status |
|---|---|---|---|
| **gen_race demo** | **L=32** | **105.534 → 108.463 (+2.8 %)** | **not explained by its record — see below** |
| gen_twiddle | L=12 | 8.714 → 8.954 (+2.8 %) | scored cells never execute the new path by construction; its record measured this cell a wash |
| gen_layout | L=25 | 96.057 → 98.515 (+2.6 %) | quad kernel cannot apply (L ≢ 0 mod 4); record pre-flagged "odd: +1.3 % drift" |
| gen_layout | L=15 | 18.810 → 19.126 (+1.7 %) | odd L, path unchanged; record pre-flagged |
| gen_twiddle | L=20 | 35.991 → 36.511 (+1.4 %) | bit-identical outputs to r6 (cmp-verified) |
| gen_layout | L=31 | 200.777 → 203.360 (+1.3 %) | its designed unchanged-path control; read flat (200.7 → 200.6) in its own window |
| gen_bluestein | L=20 | 82.906 → 83.990 (+1.3 %) | byte-identical code path (only L 49–56, 97–112 changed) |
| gen_planner | L=25 | 39.681 → 40.162 (+1.2 %) | record: tie |
| gen_twiddle | L=10/32 | +1.2 % each | bit-identical |
| gen_dense_prime | L=12 | 7.958 → 8.010 (+0.7 %) | ships r6 source (comment-only edit) |
| gen_pfa_small | L=20 | 12.867 → 12.929 (+0.5 %) | batched chain cmp-identical to r6 |

Every row but the first is in a cell whose code did not change, or was
pre-flagged by its own implementer, and none exceeds 2.8 %. I accept all of them
as window and code-layout effects, which on this host are worth 1–3 % routinely
(gen_layout, gen_twiddle and gen_planner all documented layout-drift incidents
this round independently).

**The exception is gen_race at L=32, and the evidence is positional, not
absolute.** gen_race's demo *is* gen_planner's trunk plus the raced picks, so the
two are directly comparable at every cell:

| cell | r7 race | r7 planner | race vs planner | r6 race vs planner |
|---|---|---|---|---|
| L=20 | 17.618 | 17.166 | **+2.6 %** | +5.7 % |
| **L=32** | **108.463** | **105.662** | **+2.7 %** | **−0.2 %** |
| L=31 | 140.403 | 138.925 | +1.1 % | +0.5 % |
| L=15 | 5.320 | 5.518 | −3.6 % | +1.3 % |
| L=40 | 224.856 | 236.694 | −5.0 % | −1.2 % |
| L=50 | 541.745 | 557.487 | −2.8 % | +0.1 % |

At L=32 the raced trunk went from 0.2 % *ahead* of the unraced planner default to
2.7 % *behind* it, and that is the cell where gen_race installed a brand-new arm
this round: `c4(d8)@t48`, from the widened pv tile race, which its record
reports as "**t48's first win, +1.6 % non-tie**". The board says the opposite. At
L=20 the same pattern is older: the race's "tie doctrine" picks the pv arm, and
the board puts pv 2.6 % behind planner's `@s1` group form for the second round
running. **These are two live instances of the race-proxy exposure the gen_r6
verdict named** — a mis-picked plan is scored as if it were the entry's best, and
it reports `ok`. Net the race layer is still positive (it beats its own substrate
at six of eleven cells, by up to 5.0 %), but two of its picks are measurably
wrong on the scoring board and neither is acknowledged in its record.

### What the board again did not measure

Roughly half of gen_r7's engineering is off-board, for a *different* reason than
in r6. There is no protocol failure this time — `cases.txt` says the acceptance
suite is **FROZEN for the whole campaign**, by design, and the round honoured it.
But the consequence is the same:

| entry | r7 work | scored? |
|---|---|---|
| gen_pfa_small | B=1 rotation step, **−23 to −37 %**, now 1.5–2.5× MKL | **no** (board is batched at all four of its cells) |
| gen_rader | rolled convs + dead-arm removal + prefetch, **−7 to −21 %** at 61/89/101/113 | **no** (31 ships bit-identical) |
| gen_twiddle | fused fold-combine for radices 7/11/13, **−6 to −8 %** at 49/77/91/98/121 | **no** (scored cells never execute it) |
| gen_planner | DFT7 hard leaf: L=14 **5×**, L=21 −38 %, L=35 −36 %, L=63 −27 % | **no** |
| gen_batchlane | 14/21/28/35 at batch-lane speed | **no** |
| gen_powp | 49/81/121/125 | **no** |
| gen_pow2 | L=16, L=64 | **no** (second round running) |
| gen_race | surprise drills at 21/44/96/61 (L=21 now **3.22×** MKL, was 1.99×) | **no** |
| gen_layout, gen_bluestein, gen_dense_prime | all-L / graded-cell work | yes |

The r6 addendum's surprise-draw mechanism is currently the only instrument that
reaches any of this. Its one measured effect this round is large and confirms it
works: the r6 surprise round priced the missing 7-point module, gen_planner built
it, and L=21 went 1.99× → 3.22× over MKL.

---

## 3. Adversarial pass: correctness, builds, crashes, absences

I went looking for a fast wrong answer. There isn't one on this board.

**Correctness: clean, complete, and verified independently of the leaderboard's
own column.** `check.log` is 658 lines with **0 occurrences of FAIL**. The
composition is exactly right for full coverage: **164 `map-chain` PASS lines
against 164 leaderboard rows** — a 1:1 match, so every scored row was
chain-gated — plus 165 `map-2-step` and 329 `single` PASS lines. Single-call
rel L2 spans 1.98e-16…5.51e-16 against tol 1e-12 (≥ 1 800× margin); graded
map-chains 2.96e-14…3.41e-13 against tol 1e-10; two-step gates 9.0e-16…3.0e-15
against tol 3e-14 (the 1.5e-14/step precision contract, ≥ 10× margin
everywhere). Every leaderboard row reads `ok`. Chain errors track volume and
chain length in the expected way; the largest (gen_bluestein, 2.6e-13 at L=10)
is the entry whose algorithm legitimately accumulates the most rounding, and no
entry is anomalously *accurate* either — which would suggest a short-circuited
kernel. **No entry failed correctness.**

Three entries changed arithmetic this round and all three re-gated honestly
rather than claiming bit-identity they did not have: gen_batchlane's lifted DFT5
(gates re-run at eight sizes, worst two-step 1.8e-15), gen_powp's 3-shear
twiddles (~2 ulp vs ~1 per twiddle, two-step margin moved by ~5 % of an 18×
margin), and gen_bluestein's new M = 112/224 tails (18 sizes single-gated
including both slice boundaries and the L = M/2 edge). Everyone else's claim of
bit-identity is backed by a `cmp` in the record.

**Builds: nothing failed to build.** `build_errors.txt` exists but contains
**two warnings and zero errors** — the filename is misleading for the second
round running:

```
impl/gen_dense_prime.c:1943  iteration 1152921504606846976 invokes undefined
impl/gen_rader.c:1705        behaviour [-Waggressive-loop-optimizations]
```

I checked both sites rather than trusting the r6 disposition. **They are the
same finding as r6, in the same code, at shifted line numbers** (r6: 1925 and
1620): the scalar remainder loop after the AVX-512 map body,
`for (; i < npts; ++i) { double re = zp[2*i] + cp[2*i]; ... }` with `size_t i`.
1152921504606846976 = 2⁶⁰; at that iteration `zp + 2*i` overflows any object, so
gcc-11 proves the loop cannot legally reach it and bounds the trip count at
`npts < 2⁶⁰`. The largest `npts` this round is 10⁶. **The UB is unreachable, gcc
does not delete the loop, and both entries pass every correctness gate —
including gen_rader at L=31, a headline winner.** Flagging, not disqualifying —
but it is now live UB in a headline winner's map tail that has survived a full
round after being written up in `gen_r6/VERDICT.md` §3, and neither owner
silenced it. One line each (hoist `2*i` into its own induction variable).

**Crashes/hangs: one, and it is the harness floor, not an implementation.**
`failures.txt` contains exactly three lines:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout`. `baseline_matrix` is the deliberately-O(L⁴) library-free
reference; at volume 10⁶ it projects to seconds per transform against a chain of
64. **Consequence, unchanged from r6 and now two rounds old: the L=100 block has
no library-free floor.** Every L=100 number is validated only against the
numpy/long-double oracle in `check.py`, not against the in-harness reference.
Adequate, but one fewer independent check at the largest geometry — and L=100 is
where the library margin is thinnest.

**Absences: all 54 are declared `supports()` declines, none is a crash.**
`timing.err` is 220 lines, **every one** of the form `<entry>: does not support
L=<n>` (0 lines of any other kind). I cross-checked the leaderboard against the
decline set programmatically: **every gen entry missing from every cell is
accounted for by a declared decline, with zero residue.**

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

Class entries declining out-of-class geometries is by design. **Nothing crashed.
Nothing is silently missing. No entry is unaccounted for.** All twelve agents
exited 0 (`agents/exits.txt`), and all twelve strategy records exist and are
substantial (894–1 151 lines).

**Two non-correctness hazards worth naming, because both are invisible in the
`ok` column.**

1. **The race mis-picks** (item 2). A mis-picked plan gate-passes and reports
   `ok`; the only symptom is the number. Two measured this round (L=32 +2.7 %,
   L=20 +2.6 % vs the unraced substrate). gen_pfa_large's own record concedes the
   general form: "the challenger playoff CAN still install ipk1 in a noisy
   window … the verdict gate-passed, so it is a performance coin-flip, not a
   correctness risk."
2. **Plan time moved by two to three orders of magnitude at four cells, and it
   bought ≤ 3.4 %.** From the board's `setup` column:

   | cell | entry | r6 setup | r7 setup | exec delta bought |
   |---|---|---|---|---|
   | L=25 | gen_powp | 0.005 s | **0.535 s** | −3.4 % |
   | L=27 | gen_powp | 0.005 s | **0.509 s** | −1.3 % |
   | L=50 | gen_powp | 0.004 s | **1.446 s** | −0.1 % |
   | L=100 | gen_powp | 0.003 s | **3.752 s** | −1.5 % |
   | L=40 | gen_race | 0.010 s | **0.377 s** | −5.1 % |
   | L=40 | gen_pfa_large | 1.176 s | **1.253 s** | −0.4 % |

   **This is not a violation**: `PANEL_BRIEF.md` sets a 60 s cold-create budget
   for a never-seen (L, B) and both records state their measured cold setup
   inside it (gen_powp 0.47–5.9 s; gen_pfa_large "cold create at 100 went
   ~4.4 → 4.8 s"). But the board strips wisdom at round end, so it scores the
   cold path, and the panel is now paying seconds of plan time at four cells for
   sub-4 % of throughput. gen_pfa_large's 1.253 s at L=40 is the worst on the
   board, 400× the next entry at that cell, was flagged in the r6 verdict, and
   got 6.5 % *worse*. Someone should decide whether that trade is wanted before
   it grows again.

---

## 4. Claimed vs. measured

Per item 0(b) the brief's Sapphire-Rapids-vs-Cascade-Lake explanation does not
apply: implementers and monitor ran on the same a80n0 Ice Lake-SP host, and the
campaign's own brief makes Ice Lake the scoring machine. The attribution below
is therefore same-host window and core-state effects.

**Agreement is the best it has been. There is one contradicted claim.**

| entry | cell | claimed | board | gap |
|---|---|---|---|---|
| **gen_layout** | L=100 | 9 277–9 417 | 9 343.190 | **inside the claimed range** |
| gen_layout | L=32 | 151.4–156.9 | 154.051 | inside |
| gen_layout | L=40 | 336.4 | 339.126 | +0.8 % |
| gen_layout | L=25 / L=27 | 98.9 / 126.5 | 98.515 / 124.942 | ≤ 1.3 %, board faster |
| gen_powp | L=25 | 30.868–31.564 (min-of-mins 30.87) | 30.882 | **exact** |
| gen_powp | L=27 | 43.075–44.293 | 43.966 | inside |
| gen_rader | L=31 | 84.82–85.63, tryout 84.818 | 84.544 | exact; bit-identical to r6 as claimed |
| gen_bluestein | L=50 | 1 406.9 (fresh core), 1 394.9 (A/B) | 1 397.610 | ≤ 0.7 % |
| gen_bluestein | L=100 | 14 379.9 | 14 333.454 | 0.3 % |
| gen_pow2 | L=32 | 55.51–56.80 | 56.378 | inside |
| gen_dense_prime | L=31 | 119.5–120.9 over 11 windows | 120.347 | inside |
| gen_twiddle | L=12 | 8.927 | 8.954 | 0.3 % |
| gen_batchlane | L=10 / L=15 | −0.8 % / −0.7…0.9 % | −0.7 % / −0.6 % | reproduces |

Three patterns, two benign:

**(i) gen_planner and gen_race both read systematically slower in their own
windows than on the board** — planner L=27 64.09–64.38 claimed vs 59.766
measured, L=31 144.7–150.0 vs 138.925, L=100 5 168–5 180 vs 4 998.545; race
L=100 5 222.4 vs 4 966.850, L=40 234.57 vs 224.856, L=25 41.671 vs 39.613. The
direction is the honest one (they under-promised), the offset is 2–9 %, and both
entries share the same substrate and the same held-lease measurement conditions.
This is the same signature the r6 verdict accepted, with the magnitude roughly
halved — gen_race's record explains why, having quantified the effect this round:
"the first invocation on a freshly-leased core reads 6–13 % slow", so every
number in its tables is a second-invocation reading. That is a genuine protocol
improvement and it shows in the agreement.

**(ii) gen_batchlane's L=20 delta did not fully reproduce** — claimed −1.0 %
from 4/4 same-core pairs, board −0.1 %. Its own record pre-flagged the window
("20: 13.53 … window busier than the r6 board's 12.87; the A/B delta is the
honest number"). Small, pre-declared, accepted.

**(iii) One claim the board contradicts, and it is a claimed *win*.**
**gen_race's `t48` at L=32, reported as "t48's first win, +1.6 % non-tie".** On
the board the t48-picked demo is 108.463 against gen_planner's unraced 105.662 —
**2.7 % behind its own substrate**, having been 0.2 % ahead of it in r6. The
race's own absolute number moved barely at all in its window (111.33 → 111.01),
so this is not visible from its side; it is only visible positionally, against
the entry the race is supposed to improve on. This is the one place where "a
fast wrong answer" has an analogue here: not a wrong answer, but a wrong plan
scored as the entry's best, exactly as the r6 verdict warned.

Two softer notes:

- **gen_twiddle's r6 headline "the demo now BEATS MKL at 100" has decayed to a
  tie.** Board: 7 797.055 vs mkl_dfti 7 803.986 — a **0.09 %** margin, down from
  0.8 %. Its L=100 code is bit-identical to r6, so nothing regressed; the claim
  simply is not quotable any more. Its six other library wins (20, 25, 27, 31,
  40, 50) are real and large (1.11–2.67×).
- **gen_layout under-claimed twice.** It called L=50 a +1.6 % drift "in an
  MKL-fast window"; the board gives it a clean library win (920.627 vs mkl_dfti
  947.956). It called L=25 an MKL win "-ish"; the board gives it 10 %. Both
  cases are the conservative direction and the right practice.

---

## 5. Which LITERATURE.md §4 open question moved

### §4.3 — *is axis fusion worth 3× or 3 %*: the re-opened case is now **closed in the negative**, from four independent directions

§4.3's re-opened paragraph named the L2↔DRAM regime "the largest untried
structural move on the board". gen_r6 tested one construction (tile the batch to
L2, run all three axes inside) and found it narrow. **gen_r7 attacked the
sibling construction — two-axes-per-pass fusion, `docs/literature` 11 Tier 2 —
and four entries independently say no.** This is the round's principal literature
result and it is settled to a standard the corpus asked for:

1. **gen_twiddle built the entire program and measured it losing at all twelve
   cells, 4/4 pairs each, with bit-identical outputs.** Split intermediate
   volume, axis 1 fed directly from split plane rows (zero gather copy), axes 1+2
   fused through one in-register 8×8 transpose, axis 2 direct-fed. Result: **+6 %
   to +35 %**, monotone in L (+8 % at L=10, +16 % at 25, +26 % at 50, +35 % at
   100). The mechanism is the transferable part and it inverts the premise:
   *the gathers were never overhead — they are the software prefetch and the L1
   staging*. The old gather walks pencil rows sequentially once; direct feed
   replaces that with the recursion's own DIT-decimated access (leaf s reads rows
   s, s+m, s+2m, …), widely-strided 64 B touches the hardware prefetchers do not
   follow. And the claimed deleted volume stream does not exist: both forms run
   five DRAM streams, but the split variant touches 48 MB/step at L=100 against
   32 MB, blowing a 24 MB L3. Their rule for the corpus: *if your engine stages
   pencils through a gather, the gather is load-bearing; delete the shuffles
   inside it, never the sequential pass itself.*
2. **gen_pfa_large closed it by DRAM accounting, without code.** Its engine has
   fused two axes per DRAM pass since r1. The literature's proposed re-cut has
   **bit-identical DRAM accounting per step** at L=100 (pass A 16r state + 16r c
   + 16w; pass B 16r + 16w — the same ~80 MB/step), and a one-pass 3-axis step is
   impossible at volume ≫ cache because the x-stage DFT25 alone couples 25
   planes. **Two passes is the floor and two-axes-per-pass is already the optimal
   pass shape**; the only reducible term is pass B's DRAM share via L3 custody of
   the state, which is the r5 c-bypass family, not a fusion re-cut.
3. **gen_dense_prime declined it on its own r6 measured boundary** (every cell
   L2-resident; fusion inside a covered cache level buys nothing), and
4. **gen_pow2 paper-checked it at 32 and 64** (at 32 the tile materializes at L2
   anyway; at 64 two sweeps stay two sweeps).

The composite answer, on Ice Lake-SP: **Tolmachev's rule survives a third round
— the payoff is the number of avoided passes times the bandwidth gap — and at
L2↔DRAM there are no passes left to avoid in a 3D transform.** §4.3 should be
marked closed for this hardware, with §07 gap 7's "TurboFNO's 3–5 % is probably
the better prior" upgraded from prior to measurement, and with a sign attached:
here it is negative.

### §4.6 — *model versus search for the instruction schedule*: moved again, and this round supplies the counter-weight

Four new data points, three for search and one sharply against:

1. **A stale verdict would have shipped the slow form.** gen_race's salt bump
   forced fresh races against gen_planner's new engine, and **the tree pick
   changed at 8 of 12 graded cells**; the fused-map boundary **flipped at L=15**
   (r6 `fm1` → r7 `fm0`, **+4.2 % non-tie**). §06 §3.4's phenomenon, one
   generation later: a search result is only valid for the engine it was run on.
2. **The compiler re-litigated two claims, both worth double digits.**
   gen_rader instrumented before choosing and found `-funroll-loops` had peeled
   its blocked convolution to **10 820 instructions / ~75 KB per chunk** — past a
   32 KB L1I, so every chunk refetched its own code from L2. Pinning the loop
   rolled bought **−7 % to −21 %** at 89/101/113. The consequence for the record
   is sharper than the speedup: *"the r6 record's unroll-vs-blocked boundary was
   raced against MANGLED blocked codegen."* Separately, gen_planner's new DFT7
   leaf, added inline to two dispatchers, cost **+1–3 % at L=40 — a cell whose
   tree contains no factor 7** (constprop clones grew 3 198 → 3 867 bytes), and
   with the race on it compounded into a **mis-picked plan** until the leaf was
   made a noinline call. Both were caught by `nm -S` symbol-size diffs, not by
   benchmarks.
3. **A model-predicted win that measurement inverted, with the mechanism.**
   gen_pow2 built Garrido's constant-per-site twiddle routing (lit 11 Tier 1) in
   its strongest x86 form: **−4.6 % instructions, vbroadcastsd 33 → 7, identical
   FMA count — and +0.7…2.7 % wall, 5/5 and 6/6 rotated pairs.** The mechanism is
   the finding: the runtime tables' per-iteration broadcast reloads were
   compiler-enforced *rematerialization*, keeping each constant's live range one
   group long; compiling the constants in stretched those live ranges across a
   body already at the register-file edge. First measurement of this citation in
   the corpus, and the verdict is "structurally already present in any
   fully-unrolled FTW-style codelet; the final literal-constant step is a small
   loss on ICL."
4. **The counter-weight, and it is this round's addition.** Search's *own*
   picks were wrong twice on the scoring board (item 2): `t48` at L=32 and the
   pv arm at L=20, each ~2.7 % behind the unraced default, both reported as wins
   or ties by the race. And gen_race's new `p47` stage "wins nowhere on ICX". So
   the r6 qualifier — *a search is only as good as the proxy it times* — now has
   a second, independent instance, and a third form: **a search that races a
   knob the host does not care about pays its cold-create cost for nothing and
   occasionally installs a loser.**

### §4.8 item 5 — *no quantified comparison of batch/vector-loop placement for batched small 3D transforms* ("our largest untapped search axis"): partly moved

gen_pfa_small's B=1 rotation step is the first clean measurement of this axis in
the campaign. It isolated the two effects separately, same-core interleaved:
per-slab half-turn beats the r6 transpose-sandwich + separate map pass by
**9–14 %**, and the full rotated store beats the half-turn by another **10–26 %**
where per-slab blocking wastes lanes — with the block count as the mechanism
(13 vs 20 blocks at L=10) and the port-5 bill as the receipt (tr8 per volume-step
at L=10: **160 → 52**). §4.8 item 5 can be downgraded from "the literature only
gestures at it" to "measured once, on the split path, and worth 23–37 %".

### §3 / §4.2 — the corpus's "direct O(n²) is preferable up to n ≈ 20" position is **refuted on AVX-512**, by three independent measurements

§03 §6.4 and §06 §6.4a cite McFarlin et al. and FFTW's generic Θ(n²) plan to
argue for direct computation at small n; lit 11 Tier 2 restates it as a
dense-GEMM crossover at L ≤ 16. Three entries priced it this round and it is dead
at every size we score:

- **gen_pfa_small built it** (`dense3_slab`, compiled DFT-matrix tables,
  zero-padded columns, masked stores, zero shuffles — the claim's *most*
  favourable regime, the B=1 cross-lane pass) and measured **+35 % / +60 % /
  +43 % / +63 %** against the structured form at 10/12/15/20.
- **gen_dense_prime is that design** and the board shows it 3.4–4.7× behind
  PFA/batch-lane at 10/12/15.
- **gen_batchlane and gen_pow2 declined it on arithmetic** with the same number:
  a dense 10-point matrix apply costs 4L² = 400 FMA/pencil against the PFA's 84,
  on an engine already running 43–48 GF/s of a ~93 GF/s FMA peak. The crossover
  cannot exist where the FMA ports are the binding resource.

This does not settle §4.2 (which is about L=17 specifically, and 17 is still not
on any board), but it removes one of §4.2's four positions for the size range we
do measure.

### Untouched

**§4.1** (register pressure in batch-vectorised L=6/L=8 codelets) — those
geometries are not in this campaign. The *phenomenon* was measured at a different
size, though: gen_batchlane built a fused DFT7×2 stage-2 at L=14 and it lost
**+6–10 %** because 28 slot loads plus two DFT7 cores' ~24 temps each spill past
32 zmm — and gen_pfa_small's r4 register-budget rule **predicted it in advance**.
That is §4.1's "2L is a data-only lower bound, not a budget" confirmed on
AVX-512, with the boundary now measured from both sides (DFT5×2 wins ~1 %,
DFT7×2 loses 6–10 %). **§4.7** (vector-radix) — nobody attempted it, correctly.

---

## 6. The single highest-value thing the next round should attack, per geometry

**Before anything per-geometry: fix the race's arm selection at L=20 and L=32.**
It is the cheapest 2.6–2.7 % on the board, it is a bug in the layer everyone
depends on rather than a kernel change, and it is currently *invisible* to the
entry that owns it — gen_race's own windows read L=32 flat. Concretely: the
scoring board's cold race should compare its pick against the unraced planner
default and refuse to install an arm that loses to it. The r6 verdict asked for
the race proxy to time the graded chain; gen_powp and gen_pfa_large both built
playoffs that do so, and this round's two mis-picks are in the *tile* and
*tie-doctrine* stages, which those playoffs do not cover.

Per geometry:

- **L=10, 12 (B=64) — stop tuning the kernel; explain the L=10 spread.**
  The twins are now tied to the third decimal at L=12 (1.915 vs 1.915) and within
  0.7 % at the other three shared cells, for the third round running. The DFT5
  lift was the last queued op-count cut and it bought 0.7 %. The one unexplained
  number at these cells is that **L=10 carries a 14.0 % run spread at 43.4 GF/s
  while L=12 carries 0.1 % at 48.5 GF/s** — same engine, same layout rule, third
  round with the same spread, and nobody has measured its cause.
  `perf` is unavailable on both hosts (`perf_event_paranoid=4`, no binary), so the
  instrument is gen_dense_prime's microbench substitute (`prof2.c`-style per-phase
  timers, plus a 4K-aliasing probe per §4.5). One window, and it either finds a
  placement bug worth 10 % or closes the cell for good.
- **L=15, 20 (B=32) — the arm selection above is worth more than any kernel
  change proposed for these cells.** Beyond it, gen_twiddle's whole-level
  register-resident codelets (r6, −45 % at L=20) still have not been tried inside
  the PFA leaders; that carry from the r6 verdict is untouched and remains the
  only structural idea on the table here.
- **L=25, 27 (B=16) — close the planner's arithmetic gap, and price the plan
  time.** gen_powp declares 27 saturated ("protect, don't chase") and its record
  is right: with twiddle ops now cut 4–6 %, the next lever at 27 is memory shape,
  which two independent negatives (its own two-column rejection, gen_dense_prime's
  custody results) say the OoO window already covers. The live lever is
  gen_planner's own item 2: **Winograd/real-factor DFT9 and DFT25 modules in the
  fused codelets**, which planner names as "my largest arithmetic gaps to powp"
  (1.28× at 25, 1.36× at 27). Separately: gen_powp's cold setup at these two
  cells went 0.005 s → ~0.52 s for ≤ 3.4 %. Decide whether that is wanted.
- **L=31 (B=16) — do not spend the round on the frontier; put dense-vs-Rader on
  a board instead.** 84.8 µs for three rounds, bit-identical. gen_dense_prime has
  now *decomposed* the plateau rather than guessing at it: the map is
  divider-throughput-locked (3 844 vdivpd/volume × ~16 cy ≈ 21 µs floor, and the
  lazy-map lever nets 2.6 µs and cannot net more); both GEMM passes run ~1.0
  zmm-FMA/cy against the machine's 2.0; 25–30 % of even the isolated GEMM is
  tile-block boundary drain; and the boundary count is **invariant** — fixed by
  the 32-register file. Three bit-identical restructurings all lost (+0.5–1 %,
  +2.3 %, wash). The residual is register-file/ROB physics at 240-FMA drain
  granularity and "not reachable from C intrinsics"; the only lever left is a
  hand-scheduled 32-accumulator asm kernel (their estimate: 15–25 %, full round,
  high risk). **The higher-value move is §4.2(a), for the third round running:
  put dense-fold and Rader head-to-head at 17, 19, 23, 29 on an actual board.**
  Both engines have covered those primes since r6 and neither has ever been
  scored there.
- **L=32 (B=8) — the cell is closed; the value is in the two unscored sizes.**
  gen_pow2 has now measured out every cheap theory against its documented
  [port ∥ L2] ceiling with < 8 % residual: constant-per-site routing (+1 %),
  huge pages / TLB (wash, 6 rounds), seven prefetch variants, alignment, L2
  capacity, op-count cuts (r5: −7 % ops, zero wall). Do not spend a round on
  8 %. **Score L=16 and L=64** — gen_pow2's only r6 change (L=16, −3.5 %) and its
  L=64 L3-bandwidth wall have now been unmeasured for two consecutive rounds.
- **L=40 (B=8) — the leader has been flat for three rounds and its plan time is
  the outlier on the board.** 159.959 µs (−0.4 %), against challengers that moved
  −5.1 % (race, via `@s4` with a fused-GT child) and −13.4 % (layout). Its setup
  is **1.253 s**, 400× the next entry at this cell, flagged in the r6 verdict and
  now 6.5 % worse. Its own next-list item 1 — PMU attribution of the p1 phase —
  has been queued four rounds and is blocked on `perf` being unavailable on both
  hosts. **Either get PMU access on a80n0 or adopt the microbench substitute and
  close the item**; four rounds of "queued" on the weakest phase of a
  three-round-flat leader is the real finding at this cell.
- **L=50, 100 (B=4, B=1) — the ties are unbreakable and the margin is the
  target; gen_bluestein just showed how.** Both cells are ties (0.3 % and 1.7 %)
  that four rounds have failed to resolve, and both carry the thinnest library
  margins on the board (2.29×, 1.74×). Stop resolving the ties. The round's
  cheapest win came from re-checking a *declined* idea against `cases.txt`, and
  gen_bluestein's next M-cut is already named and unspent: **M = 208 (13·2^k) for
  L = 97…104, worth another ~7 % of convolution data at L=100**, gated on an
  honest DFT-13 tail budget. Two carries from the r6 verdict are also still open
  at L=100 and both are cheap: the **fused-exit boundary 81…99 is entirely
  unmeasured**, and **L=100 still has no library-free floor** (baseline_matrix
  timed out 3/3 for the second round running — a longer timeout, or a blocked
  reference, restores one independent check at the largest geometry).

---

## 7. What to keep

Against `docs/CURATION.md`'s four grounds. This is again a wide promotion —
eleven of twelve — and it needs justifying rather than glossing. gen_r7 was a
"spend the literature backlog" round: the frontier moved 0.1–3.4 % while the
second rank moved up to 23 %, and the round's most valuable products are
**measured negatives with named mechanisms**. Ground 4 read literally would
promote all twelve (every gen entry beats a library somewhere), so the
discriminating criterion is CURATION's near-duplicate exclusion, and I apply it
to exactly one entry.

**Ground 1 — fastest correct entry per geometry (always).** Five entries:
`gen_batchlane` (L=10, 12, 15, 20), `gen_powp` (L=25, 27), `gen_rader` (L=31),
`gen_pow2` (L=32), `gen_pfa_large` (L=40, 50, 100). The L=15 and L=50 leader
flips are ties (0.6 %, 0.3 %) and are recorded as such in item 1 so the next
panel does not read the row order as a result.

**Ground 2 — structurally different runner-up.**
- `gen_dense_prime` at L=31 (120.347, 1.42× behind gen_rader) — still the only
  cell where a genuinely different algorithm (dense conjugate-fold GEMM vs Rader
  convolution) holds second place, and now the entry that has *explained* the
  crossover rather than just losing it.
- `gen_planner` is the structural runner-up at L=25/27/32/40/50/100 (1.28–1.87×)
  and is the substrate that produced most of the round's board movement (fused
  Good-Thomas codelets, DFT7 as a hard leaf). It beat the best library at **all
  eleven cells**.
- `gen_pfa_small` is within 1 % of gen_batchlane at all four shared cells, which
  looks like the near-duplicate case — but the two are now clearly distinct *as
  contributions*, more so than in r6. gen_batchlane's r7 work is batched
  arithmetic (the lifted DFT5); gen_pfa_small's r7 work is **entirely in the B=1
  split path** (the rotation step, −23 to −37 %, first MKL win there since r1),
  a path no other entry attacked, on an axis `PANEL_BRIEF.md`'s standing rules
  call co-equal with batched. Its rotation step is also offered as adoptable
  wholesale, with gen_batchlane's own eight-size B=1 gap named as the target.
  Kept, with the batched tie recorded.

**Ground 3 — instructive failures whose record documents the killing number.**
This is the round's strongest category and it is why the promotion is wide.
- `gen_twiddle` — the round's best negative: the split-custody two-axes-per-pass
  direct feed, built whole, **12 cells × 4 pairs, all losses, +6 % to +35 %**,
  with bit-identical outputs proving the arithmetic was preserved, and a
  transferable mechanism (*the staging gather is the software prefetch; never
  delete the sequential pass*). This closes a §4.3 question the corpus called
  the largest untried move on the board.
- `gen_pow2` — Garrido constant-per-site routing built in its strongest x86 form
  and raced out (−4.6 % instructions, +1 % wall, 5/5 and 6/6), plus the TLB
  theory of its x-pass residual killed (wash, 6 rotated rounds). First
  measurement of that citation in the corpus, with the register-pressure
  mechanism named. Ships bit-identical to r6.
- `gen_dense_prime` — three bit-identical levers, all measured losers
  (`-falign-loops=32` wash-to-negative, pair-pipelined z-phase +0.5–1 %, merged
  C+S GEMM +2.3 % at 0.75 vs 0.5 loads/FMA), plus the first counters-free
  decomposition of the L=31 plateau. It converts a six-round mystery into an
  explained register-file/ROB bound with a named remaining lever.
- `gen_pfa_small` — the dense-GEMM crossover implemented honestly in its most
  favourable regime and killed at four sizes (+35 % to +63 %), with the
  arithmetic that makes it impossible on this ISA.
- `gen_batchlane` — DFT7×2 fusion at L=14 (+6–10 %), which puts a measured
  number on the register-budget boundary from the losing side, and `BL_SAFE15`
  re-raced under the lift and closed again.
- `gen_pfa_large` — `ipk1`-rank-first tried on gen_powp's evidence and **refuted
  4/5 on this engine**, directly contradicting gen_powp's 5/5 the other way. The
  finding is the boundary itself: *rank evidence does not transfer even between
  engines sharing a shell; race machinery does.*
- `gen_planner` — the DFT7-inline case-bloat incident: +1–3 % at a cell whose
  tree contains no factor 7, compounding into a race mis-pick, fixed by making
  the leaf a noinline call, and caught by `nm -S` before any benchmark.

**Ground 4 — anything that beat a library baseline.**
- `gen_layout` beat the best library at **seven of eleven cells** (20, 25, 27,
  31, 32, 40, 50) after a −2.2 % to −22.8 % round, its quad kernel isolated at
  −13.6 % / −23.1 % by a clean A/B knob, and its r6 claim that failed on the r6
  board (an MKL win at L=32) is now real with a 10.8 % margin. Its claimed
  numbers reproduce cell-for-cell — the best claim/board agreement on the board.
  It is also the layer nearly everyone `#include`s, frozen for the third round
  running by deliberate doctrine, and the quad identity is written up as
  adoptable by gen_dense_prime and gen_rader with a measured reference.
- `gen_bluestein` beat every library at L=31, moved **−21.1 % at L=50** and
  −7.9 % at L=100 on the board, and did it by re-checking a coverage argument it
  had got wrong in r6 — a method lesson worth as much as the number. It remains
  the only any-L entry and the library's universal fallback.
- `gen_twiddle` beat the best library at six cells (20, 25, 27, 31, 40, 50) and
  ties at 100; see ground 3.

**Excluded: `gen_race`.** Same two grounds as the r6 verdict, both still holding,
and this round adds a third.
1. Its library is **frozen — "zero changes, no `gr_*` signature or wisdom-format
   change" — for the fourth round running**, so promoting it puts no new library
   code in front of the next panel.
2. Its demo *is* gen_planner's r7 trunk. CURATION's near-duplicate exclusion is
   the only criterion that discriminates in a round where ground 4 would promote
   everyone, and this is the pair it describes.
3. Its two genuinely new r7 stages did not earn their keep on the board: the
   `p47` stage "wins nowhere on ICX" by its own account, and the widened tile
   race's `t48` pick at L=32 is **2.7 % behind the unraced substrate** (item 2).
   The stages that *did* win at 15/40/50 are its r3/r6 stages (`fm`, `@s4`,
   tile), already on record.

   Its genuinely valuable results this round are all *findings*, and findings
   survive without the code: the fm boundary flipping at L=15 (+4.2 %, a stale
   verdict would have shipped the slow form), the four-round cross-architecture
   receipt table (the L=20 tree verdict flipping on Cascade Lake's 1 MB L2,
   exactly as predicted in r5), the quantified first-invocation-on-a-fresh-lease
   warmup effect (6–13 %), and the L=21 surprise cell going 1.99× → 3.22×. All
   are recorded in §2, §4 and §5 above and in `strategies/gen_race.md`. **The
   exclusion is not a judgement on the work** — and the two mis-picks I found are
   a reason for its owner to act, not a reason to bury the layer.

No entry is excluded for failing correctness, failing to build, crashing, or a
missing strategy record: all twelve records exist and all are substantial
(894–1 151 lines), and the board is clean on all four counts.

---

*Round monitored 2026-08-25. Verdict is on the eleven geometries actually
measured (L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100) on Intel Xeon Gold
6326 (Ice Lake-SP), not on the L = 6/8/17/36 Cascade Lake premise in the monitor
brief — see §0. That premise has now been wrong for two consecutive rounds;
`panel_round.js` should be pointed at `cases.txt` and `environment.txt` when it
writes `monitor_prompt.txt`.*
