# ice_r3 — monitor's verdict

Measured on `a80n0.lqcd.mit`, 2026-08-22T21:28 → 22:50, slurm job 438572.
75 backend×geometry cells, 3 runs each (225 supported timing runs). `build_errors.txt` is
empty; `failures.txt` was never created. All 20 binaries built, all 19 panel entries and the
`baseline_matrix` floor produced a scored line at their geometry, and **all 75 cells pass both
the single-call and the whole-chain correctness gate**.

## 0. Correction to the brief before anything else

The brief this round was written against the wrong machine, on both sides, and the error is
not cosmetic — it is the single fact that moved the most numbers.

* **Scoring node.** `environment.txt` reads `Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` —
  Ice Lake-SP (ICX), **two** 512-bit FMA pipes, 1.25 MiB L2. Not a Xeon Gold 5218 (Cascade
  Lake, **one** 512-bit FMA pipe, 1 MiB L2). The round is named `ice` for this reason.
* **Dev machine.** The ice panel's implementers develop on *the scoring node itself*, via
  `tryout.sh`. `strategies/L6_unrolled.md` heads its section "Round ice_r2 (dev machine = the
  reserved Ice Lake node itself, via tryout.sh)"; `strategies/L13_direct.md` says outright
  "this panel does not measure on wallaby". `strategies/L13_rader.md` quotes its dev numbers
  as "(a80n0, Xeon Gold 6326, gcc 11.4, `./tryout.sh L13_rader 13 32`)".

The Sapphire-Rapids-dev / Cascade-Lake-score framing, and the 2.9× MKL span that goes with it,
describe the earlier `bench/geom` panel — which is these entries' lineage, and which several
records explicitly say they are correcting for. **No ice_r3 discrepancy can be attributed to a
dev-versus-score machine difference: it is the same silicon.** §4 below gives the causes that
actually apply. The two-FMA-pipe difference matters directly: it is what reopened AVX-512 this
round (§5).

## 1. Headline per geometry

### Batched — the graded cell, as measured

The graded configuration in `cases.txt` is one `L:batch:chain` triple per geometry, and the
timed unit is the full *m*-step map chain, not a bare transform. `per-transform` =
per-call / (m·B). Figures are the leaderboard's min across three runs.

| L | cell | fastest correct panel entry | best library | margin |
|---|---|---|---|---|
| 6 | B=64, m=4856 | **L6_unrolled 0.213 µs** (39.37 GF/s) — dead tie with L6_pfa 0.213 (39.28) | mkl_dfti 0.340 µs (24.63) | **1.60×** |
| 8 | B=64, m=2572 | **L8_fusedaxes 0.544 µs** (42.31) — L8_batchsimd 0.547 (42.13), L8_radix8 0.567 | mkl_dfti 0.626 µs (36.81) | **1.15×** |
| 17 | B=32, m=98 | **L17_matrixsimd 13.061 µs** (23.06) — L17_winograd 16.240, L17_rader 19.368 | ducc0_c2c 74.554 µs (4.04); mkl_dfti 76.271 | **5.71×** (5.84× vs MKL) |
| 36 | B=8, m=64 | **L36_pencilfused 109.619 µs** (33.01) — L36_pfa 111.568, L36_mixedradix 117.417 | mkl_dfti 160.669 µs (22.52) | **1.47×** |

The other four geometries, for completeness: L=13 `L13_rader` 4.619 vs mkl2026 6.043 (1.31×);
L=23 `L23_rader` 39.502 vs ducc0 219.765 (5.56×); L=45 `L45_mixedradix` 259.287 vs mkl_dfti
520.823 (2.01×); L=64 `L64_blocked` 902.885 vs mkl_dfti 1019.974 (1.13×). The panel leads at
all eight geometries.

Note that at L=17 the best library is **ducc0, not MKL** — MKL and all three FFTW planner
levels sit in a 76–80 µs band that barely moves between planner levels, the signature of a
generic prime-size path.

### Non-batched — NOT MEASURED THIS ROUND

`cases.txt` contains no `B=1` row, and no `*_B1_*` artifact exists in `ice_r1`, `ice_r2` or
`ice_r3`. `sweep.sh` still carries the comment "Cases: batch=1 is the NON-BATCHED problem,
reported separately from the batched ones" and a default case list containing `6:1 8:1 17:1
36:1`, but `cases.txt` overrides it. <cc-memory filenames="fft-benchmark-methodology.md">The
standing rule for this project is that batched and non-batched are reported separately, so the
non-batched half of every headline has now been unscored for three consecutive ice
rounds.</cc-memory> Two entries have asked for it in writing; `L8_batchsimd` notes "if B=1
lands in `cases.txt` (the VERDICT asked twice)".

I will not invent the numbers. Below are the implementers' own B=1 chain readings, taken on
**this node** via `tryout.sh` with the in-window MKL they quote alongside. These are dev
measurements, not scored results, and the window caveats of §4a apply to every row.

| L | best panel entry at B=1 (unscored) | MKL, same window | margin |
|---|---|---|---|
| 6 | L6_unrolled 0.188 µs (sd 0.06%); L6_pfa 0.218–0.219 (busy window) | not recorded | unquantified |
| 8 | L8_fusedaxes 0.554 µs; L8_batchsimd 0.555 | **0.546** | **MKL ahead by 1.5%** |
| 17 | L17_matrixsimd 12.071 µs (sd 0.28%); L17_winograd 15.767; L17_rader 15.83 | 73.8 | 6.11× |
| 36 | L36_mixedradix 95.39 µs (sd 0.16%); L36_pfa 106.287; L36_pencilfused 103.1 (quiet) | 139.6 / 138.4 | 1.46× / 1.30× |

Two things in that table matter more than the batched leaderboard:

1. **L=8 is the one geometry where a library is ahead non-batched.** MKL 0.546 vs 0.554. The
   1.15× batched lead does not survive to B=1, and nobody is scoring the cell where it fails.
2. **The L=36 ranking inverts.** Batched: pencilfused < pfa < mixedradix. Non-batched:
   mixedradix < pencilfused < pfa. `L36_mixedradix` is last in the graded cell and first in the
   ungraded one, by 8%. A single graded batch depth is picking the winner.

At L=6, B=1 (0.188) is *faster per transform* than B=64 (0.213) — the batched cell is the
slower one, which says the 0.42 MiB batched working set is not buying what the batch was
meant to buy.

## 2. What changed since ice_r2 — and the noise floor that decides it

**Only 8 of 19 entries shipped changed code.** Diffing `impl_2` against `impl_3` (`impl` →
`impl_3`): changed are `L6_unrolled` (361 lines), `L8_batchsimd` (424), `L8_fusedaxes` (183),
`L13_direct` (122), `L13_rader` (191), `L17_matrixsimd` (529), `L17_winograd` (269),
`L36_pencilfused` (185). Byte-identical to r2: `L6_pfa`, `L8_radix8`, `L17_rader`,
`L23_matrixsimd`, `L23_rader`, `L36_mixedradix`, `L36_pfa`, `L45_mixedradix`, `L45_pfa`,
`L64_blocked`, `L64_radix8`, `baseline_matrix`.

That gives the round a free calibration: **eleven entries whose r2→r3 delta is pure
measurement.** Excluding `L17_rader` (whose r2 run carried 12.47% within-run sd), those eleven
span **−1.5% … +1.5%**. Any delta inside that band is not a result.

| L | entry | r1 | r2 | r3 | r2→r3 | code | read |
|---|---|---|---|---|---|---|---|
| 6 | L6_unrolled | 0.219 | 0.213 | 0.213 | +0.0% | revised | no change |
| 6 | L6_pfa | 0.220 | 0.213 | 0.213 | +0.0% | identical | noise |
| 8 | L8_fusedaxes | 0.556 | 0.544 | 0.544 | +0.0% | revised | no change |
| 8 | L8_batchsimd | 0.550 | 0.544 | 0.547 | +0.6% | revised | noise |
| 8 | L8_radix8 | 0.561 | 0.565 | 0.567 | +0.4% | identical | noise |
| 13 | L13_rader | 6.131 | 4.680 | 4.619 | −1.3% | revised | noise |
| 13 | L13_direct | 4.661 | 4.662 | 4.623 | −0.8% | revised | noise |
| 17 | **L17_matrixsimd** | 14.471 | 13.562 | **13.061** | **−3.7%** | revised | **real** |
| 17 | L17_winograd | 16.113 | 16.082 | 16.240 | +1.0% | revised | noise |
| 17 | L17_rader | 19.099 | 18.693 | 19.368 | +3.6% | identical | measurement only |
| 23 | L23_rader | 39.142 | 39.214 | 39.502 | +0.7% | identical | noise |
| 23 | L23_matrixsimd | 39.584 | 39.223 | 39.761 | +1.4% | identical | noise |
| 36 | L36_pencilfused | 123.594 | 110.477 | 109.619 | −0.8% | revised | noise |
| 36 | L36_pfa | 119.163 | 112.727 | 111.568 | −1.0% | identical | noise |
| 36 | L36_mixedradix | 119.530 | 116.814 | 117.417 | +0.5% | identical | noise |
| 45 | L45_mixedradix | 284.202 | 263.344 | 259.287 | −1.5% | identical | noise |
| 45 | L45_pfa | 295.054 | 297.097 | 293.720 | −1.1% | identical | noise |
| 64 | L64_blocked | 1205.721 | 891.973 | 902.885 | +1.2% | identical | noise |
| 64 | L64_radix8 | 1184.003 | 935.296 | 949.492 | +1.5% | identical | noise |

**Verdict on the round: a plateau.** Eight entries were revised; exactly one produced a
measurable gain. `L17_matrixsimd` −3.7% at 0.03–0.05% within-run sd, from removing a
scheduling pragma (−3.6% in matched graded pairs) plus `xfzra` over `xfdzra`. Every other
revision landed inside the identical-code band — including three (`L6_unrolled`,
`L8_fusedaxes`, `L36_pencilfused`) whose records describe substantial structural work.

**Did anything regress? No — not once you subtract the noise floor.** The four largest
apparent regressions (`L17_rader` +3.6%, `L64_radix8` +1.5%, `L23_matrixsimd` +1.4%,
`L64_blocked` +1.2%) are all on **byte-identical source**, so by construction they are
measurement. No revised entry lost ground outside the band.

Per geometry: **L=6** and **L=8** are frozen at their r2 numbers to three digits. **L=13**
consolidated (both entries improved ~1%, inside noise; the r1→r2 Rader fix of −24% holds).
**L=17** is the only geometry that moved. **L=23/45/64** were not touched by anyone — no agent
revised a line — and their r2 numbers stand. **L=36** improved 0.8% on revised code, i.e. not
at all.

## 3. Failures, missing entries, and the adversarial pass

### Nothing failed correctness, and I checked the place a fast wrong answer would hide

The leaderboard's `correctness` column shows only `rel_l2` — the **single-call** error. That is
precisely the surface `13c92ac` documented the rival pipeline exploiting: a precision-tiered
map that is exact at step 1 (2.8e-16, so any single-call grader sees a perfect answer by
construction) and drifts to 1.28e-8 by the end of a 4856-step chain. So I read the field the
leaderboard does not print. Every `c_*.json` carries `chain_ok` and `chain_rel_l2` from
`check.py --map-check`, gated at `max(1e-12, 1e-13·m)`.

**All 75 cells: `ok=true` AND `chain_ok=true`.** No entry is exact-where-graders-look and
drifting elsewhere. The decisive cell is L=6, m=4856 — the longest chain, and the one
`PANEL_BRIEF` §10 forbids precision tiering in:

| L=6, m=4856 (gate 4.86e-10) | chain_rel_l2 |
|---|---|
| L6_unrolled, L6_pfa | 2.947e-13 |
| mkl_dfti | 2.934e-13 |
| mkl2026_dfti | 2.471e-13 |
| fftw3 ×3 | 2.462e-13 |
| ducc0_c2c | 4.958e-13 |
| baseline_matrix | 2.517e-12 |

Both panel entries sit within 0.4% of MKL's accumulated drift and *below* ducc0's — the
signature of exact double arithmetic reassociated differently, four to five orders away from
the 1.28e-8 a float-seeded 2-Newton map produces at this chain length, and 1650× inside the
gate. The same pattern holds at L=8 (1.399e-13 vs MKL 1.387e-13) and L=13 (`L13_direct`
2.844e-14, the *most* accurate entry in its cell, MKL 1.665e-13). At every geometry, every
panel entry's chain drift is at or below its library peers'. **Nothing was bought with
precision this round.**

### Missing: `strategies/L23_rader.md` does not exist — third round running

`L23_rader` is the fastest L=23 entry (39.502 µs, 5.56× over the best library) and has **never
had a strategy record**. `git log -- strategies/L23_rader.md` returns nothing; the file is
absent from `strategies/`, which holds records for 18 of 19 entries. `ice_r2`'s verdict flagged
this and withheld promotion; nothing changed, and `L23_rader.c` is byte-identical to r2 (its
agent has not run since r1). CURATION is explicit — "do not promote … entries whose strategy
record is missing" — so **it is again not promotable**, and the L=23 slot again goes to
`L23_matrixsimd`, 0.65% behind. The panel is now carrying its L=23 winner as an
undocumented binary for the third consecutive round.

### Three entries shipped revised code with no ice_r3 record section

`L6_unrolled`, `L13_direct` and `L13_rader` changed 361 / 122 / 191 lines against `impl_2`, but
their records' last section is "Round ice_r2" — `grep -c "Round ice_r3"` returns 0 for all
three, against 5 for the entries that did write up (`L17_matrixsimd`, `L17_winograd`,
`L36_pencilfused`, `L8_batchsimd`, `L8_fusedaxes`).

That `impl_3` is what was scored is verifiable, not assumed: `impl_3/L6_unrolled.c` contains
the description string `"...zxf incumbent, chain-replace..."`, `impl_2/L6_unrolled.c` does not,
and that exact string appears in `t_L6_unrolled_L6_B64_r*.json`. So the measured binary is
`impl_3`'s, and **the L=6 winner's shipped code is undocumented at the r3 level.**

Mitigating, and the reason I am not treating this as disqualifying: the r2 records *prescribe*
these changes as their own next step, and the shipped descriptions match. `L6_unrolled.md`'s
"Next" reads "If zxf is picked and wins the cell, flip zxf to incumbent in r3 … and delete the
zff family"; the r3 description string reads `variant=zxf … zxf incumbent, chain-replace` with
no zff family present. `L13_direct`'s r3 pick is `ov(inc)`, the incumbency flip its r2 record
set up. The changes are the documented plan executed — but a plan is not a record of what was
measured, and the next panel reading `strategies/L6_unrolled.md` will not find the round that
produced the number it is being shown. **Fix at promotion time: these three owe an ice_r3
section before their `impl_3` code goes into `exemplars/`.**

### Measurement quality: the leaderboard's `run spread` column understates the noise

`run spread` is the spread across the three run **minima**, not within-run variance, and the
reported figure is a min-of-min. Two consequences that change how the table should be read:

* **`L6_unrolled`'s 13.9% is a clock regime, not code.** Its three runs: r1 `kclk=2.90GHz` →
  0.242 µs (within-run sd 0.05%); r2/r3 `kclk=3.30GHz` → 0.213 (sd 0.05%). 0.242/0.213 = 1.139,
  exactly the 3.3/2.9 = 13.8% signature `L6_pfa.md` documents ("Race numbers scale by exactly
  3.3/2.9 between regimes; rankings are identical in both"). Run 1 executed entirely in the
  base-clock regime. Its r2 record predicted "≤1% run spread in the drained window" — the
  window was not drained.
* **A 0.3% spread can hide 6.7% bimodality.** `L6_pfa` reports 0.3% spread, but its r2/r3 runs
  have within-run sd 6.11% / 6.68% with **medians at 0.244 / 0.243 against minima of 0.214** —
  each of those runs contains both clock regimes and min-of-min kept only the fast samples.
  Its own in-plan probe read `kclk=2.90GHz` in all three runs while the execution ran ramped,
  so the probe and the kernel disagreed.
* **The worst case is a library.** `mkl2026_dfti` at L=36: r1 203.459, r2 215.575, r3 163.653.
  The published 163.653 is the best of a 1.32× spread; its median-of-medians is 203.5. Had this
  round reported medians, mkl2026 would read 1.86× behind `L36_pencilfused` rather than 1.49×.

**Does this bias panel against library?** No, and I checked rather than assumed. Transient
~6–7% within-run episodes hit libraries about as often as panel entries (`mkl_dfti` L=6 r2 6.54%,
`mkl2026` L=8 r1 7.26%, `fftw3_patient` L=8 r1 7.76%, `ducc0` L=36 r2 5.99%, `fftw3_estimate`
L=17 r1 2.29%), and every library got at least one clean window at every geometry. The
published *ratios* are fast-window against fast-window and stand. What does not stand is
reading them as sustained throughput: they are best-of-three-windows on a node that was
demonstrably not drained.

## 4. Claimed versus measured

Per §0, "the machine difference" is not available as an explanation this round — dev and
scoring are the same node. The gaps have three causes, all of which the records themselves
identify and quantify.

**(a) Clock regime, ±14–21%.** The node has two regimes: quiet reads 3.50 GHz at 256-bit and
3.30 GHz at 512-bit; busy pins everything at the 2.90 GHz base. `L17_winograd.md` tabulates all
three states (quiet 3.5/3.3, mid 3.4/3.3, contended 2.9) and its own pick moves 16.73 → 18.84
across them. Implementers therefore quote MKL per window as the calibration handle, and it
works: `L17_matrixsimd` and `L17_winograd` both record "MKL 84.1 on the same core" against a
scored MKL of 76.271, so their dev window ran 10% slow; scaling their dev claims by
76.271/84.1 = 0.907 lands both on their scored numbers (matrixsimd dev 14.5–14.9 → 13.2–13.5,
scored 13.061; winograd quiet 16.73 → 15.2, scored 16.240). **The AVX-512 licence cost on this
part is −5.7% (3.50 → 3.30 GHz), not a cliff** — see §5.

**(b) In-plan arena price vs the graded cell, +5…+30%.** The `pick=` and `arena{}` strings in
the leaderboard's backend column are in-plan tournament prices at small `nvol`, not predictions
of the graded cell. Largest gap: `L8_fusedaxes` publishes `fusedAA2+pfs=0.419` against a scored
0.544 (+30%). Its record already closes this: the 0.419 row is the *quiet-window arena* table,
and the calibrated chain-shaped figure is "in-plan `fused+pfs=0.563` vs driver-forced 0.568
same window — the 22–34% optimism the VERDICT flagged is gone." Same shape at `L8_radix8`
(0.480 → 0.567, +18%) and `L23_rader` (33.50 → 39.502, +18%), neither of which ported the
chain-shaped tuner.

**(c) Driver-side work outside the tuner's timed interval, +7…+26%.** The graded unit is
transform + a driver-side unitary scale of the whole destination + a ping-pong between two
buffers; tuners that time only the kernel interval miss the scale pass. Quantified three
independent ways: `L13_direct.md` — "the graded cell runs ~4.2 µs kernel + ~0.5 µs driver-side
unitary scale" (11% of the cell); `L36_mixedradix.md` — "The driver scale pass costs ~18–20
µs/xform on top of execute — identical for every backend, not addressable"; `L36_pfa.md` — "the
driver's own unitary-scale pass (~1.5 MB RMW per volume, identical for all backends, not
removable)". This is most of `L13_rader`'s 3.657 → 4.619 (+26%) and `L36_mixedradix`'s l1w
109.7 → 117.4 (+7%).

**The entries that ported the chain-shaped tuner predict the scored cell to ~1%:**
`L36_pencilfused` `vr4=110.3` → 109.619 (−0.6%); `L17_winograd` `pick=16.37` → 16.240 (−0.8%);
`L8_batchsimd` `FUSEDAA3/s0=0.555` → 0.547 (−1.4%); `L17_matrixsimd`'s written projection
"~12.9–13.4 µs/step in the quiet window" → 13.061, inside its own band. **This is a methodology
result, not a machine result: the tuner shape, not the silicon, decides whether a claim
survives scoring.**

Two entries were *pessimistic*, both quoting loaded-window arena prices: `L23_matrixsimd`
`pick=50.73` vs scored 39.761 (−22%) and `L45_pfa` `312.0` vs 293.720 (−6%). Worth naming
because an implementer who under-claims by 22% will discard candidates that would have won.

## 5. Which LITERATURE §4 open question moved

### Primary: §4.8 item 6 — AVX-512, and specifically "no primary measurement for Ice Lake-SP server"

§4.8.6 closes on an instruction: *"there is no primary measurement in the corpus for Ice
Lake-SP or later server parts, which is the hardware most likely to be in a current LQCD
cluster. **Measure it on the node.**"* This round is that measurement, and it **inverts a
result the previous panel had settled 0-for-20.**

1. **The part is not the part §4.8.6 reasons about.** §4.8.6 builds its case on the Gold 5218,
   which "has **one** 512-bit FMA unit, so 512-bit and 256-bit code have *identical* peak FP
   throughput". The Gold 6326 has **two**. `L6_pfa.md` states the consequence exactly: "The
   panel_r7 falsification of AVX-512 ('zero picks in eight cells at equal licence clock') was
   CLX-specific: the Gold 5218 has ONE 512-bit FMA pipe, so zmm halved the instruction count
   but not the port-cycle floor. The ice node's Gold 6326 has TWO."
2. **Licence cost measured: −5.7%.** 3.50 → 3.30 GHz, two independent in-plan probes across
   two entries, versus §4.8.6's inherited fear of a Skylake-SP-style collapse (Gold 5120: 2.7 →
   1.6 GHz). At L=6 that −5.7% is "fully covered by the −17% instruction count on the x-pass".
3. **L=6 flipped.** Both entries independently reopened 512-bit and both picked it:
   `L6_unrolled` ships `variant=zxf` with `zwd=-2.9%` (zmm ahead of the best ymm shape),
   `L6_pfa` ships `variant=z512x` at −5.1% against its own ymm incumbent. Two different
   implementers, same conclusion, both landing on 0.213 µs. `L6_unrolled.md` wrote the
   pre-registered prediction — "I bet 55/45 on `zxf` being picked … if the pick is `zxf`, `zwd`
   reads ≤ −2.5%" — and the scored line resolved it at −2.9%.
4. **L=17 flipped harder.** `L17_matrixsimd.md`: "256-bit width at the graded cell: best 19.15
   vs 14.29 — on 2×512-pipe ICX the width question is settled the opposite way." 512-bit is 34%
   ahead at the geometry with the largest absolute margin on the board.

§4.8.6's inherited conclusion — that 512-bit wins on half the instruction count at near-zero
clock cost — is now confirmed on Ice Lake-SP server silicon, *and* it gains a second mechanism
the 5218 does not have. That paragraph can be rewritten from measurement rather than from
Intel's turbo tables.

### Secondary, and the more interesting result: §4.5 moved in both directions

§4.5 asks whether L=8 needs padding and where. This round answers it at L=8 **and** finds the
effect somewhere nobody was looking, with the two answers differing by 44×.

* **At L=8 it is closed as negligible, by timing.** `L8_fusedaxes` built the deferral A/B and
  read quiet `0.418 / 0.418 / 0.411 / 0.420 / 0.414` and contended `0.491 / 0.492 / 0.481 /
  0.495 / 0.485`, concluding: "**the ~195-cycle residual over the 1072-cycle port floor is NOT
  volume-boundary 4K aliasing.** … their cost is under ~0.5% and L=8 should stop spending
  rounds on aliasing." With the earlier permutation-invariance proof, §4.5 at L=8 is closed
  from both ends — the collisions cannot be scheduled away, and deferring them buys nothing.
  §4.5's requested counter (`ld_blocks_partial.address_alias`) is still unread: `perf_event_open`
  returns `EACCES` in both dev and scored contexts, unchanged for three rounds.
* **At L=36, B=1, it is worth 22%.** `L36_mixedradix` pinned `(pout − pl) mod 4096 = 2112` and
  read **122.48 → 95.39 µs/xform end-to-end** (in-arena pf0 102.0 → 82.5, y-share of the tsc
  split 38.4% → 32.7%), with MKL steady at 140.5/139.6 across the pair. The same knob priced 0
  to −1.2% on Cascade Lake. The mechanism is diagnosed: both subloop bodies load all 36 vectors
  before storing any, so within-call aliasing is impossible — the collisions are *across
  consecutive calls*, load row *k* of call *zb*+1 against an in-flight store row *j* of call
  *zb*, colliding when `pind ≡ 64·(1+9(k−j)) mod 4096`.

So §4.5's answer is not "L=8 needs padding". It is that **4K store→load aliasing here is a
cross-call effect whose size is set by the volume stride and the batch depth**: negligible
where a deep batch keeps many independent calls in flight, large where one volume's consecutive
calls collide. §08's `make_noncritical()` advice is right, but the place to apply it is the
non-batched cell at L=36, not the batched cell at L=8 — and the non-batched cell is the one
this round did not grade.

### Also moved

* **§4.2 (L=17: dense-symmetric, Rader, or Winograd).** The batched ranking is now
  dense-symmetric **13.061** < Winograd **16.240** < Rader **19.368**, so §02 §7's "Rader is not
  the lever at L=17" is confirmed on this hardware at 1.48×, and §4.2(b)'s symmetric/
  antisymmetric convolution split — `L17_winograd`'s module, whose kernel `L17_rader` also
  borrows — does not overturn it from either side. Caveat, and it is why I call this a partial
  move: `L17_rader.c` is byte-identical to r2 and carries no ice_r3 section, so §4.2(a) was
  answered by the dense side *improving*, not by a fair contest between three revised kernels.
  §4.2(c) (the exact op count for a full 17-point Winograd module) remains open — still no
  journal access, still TR 8105.
* **§4.1 (register pressure) is retired rather than answered.** The premise — batch-major L=6/8
  "will spill a little" on 16 registers — does not apply on a 32-register machine, and
  `L6_unrolled.md` says so: "the premise that blocked AVX-512 … is dead on a 32-register
  machine." Both L=8 entries audited the ICX build and found **zero** zmm stack traffic
  (`L8_fusedaxes`: "ax0_d\* bodies are 110 instructions, 0 rsp refs"; `L8_batchsimd`: "**zero**
  zmm stack moves"). The spill *price* did get measured, from the wrong end: forcing a 16-arm
  always-inline merge with ~40 live zmm "spills catastrophically", confirming gcc 11.4 cannot
  hold that live set — but at our shapes there is nothing to spill.
* **§4.3 (axis fusion) did not move.** Its reopened clause names one construction — tile the
  batch so a tile fits L2, then run all three axes inside the tile, across the 7× L2↔DRAM gap.
  No entry built it. `L36_pencilfused` fuses y+z then runs a strided x pass over the whole
  volume; `L36_pfa` and `L36_mixedradix` are two-sweep. See §6.
* **§4.6 (model vs search)** accumulated one negative: `L17_matrixsimd` gained 3.6% by
  *removing* a scheduling pragma, and `L8_fusedaxes` declined to try it because the same pragma
  measured +1.7% on `L23_matrixsimd` and +5.2% on `L13_direct`. Schedule search remains
  worthwhile; schedule *hints* on fully-unrolled straight-line code are a tax.

## 6. Highest-value attack for the next round, per geometry

**L=6 — fix the measurement, not the code.** There is no code question left to answer until the
number is trustworthy: the two entries are 0.2% apart, both now pick 512-bit, and 0.213 is a
fast-window minimum from a sweep where one run of the winner executed entirely at base clock
(§3). Get one drained-window sweep, or record `kclk` per sample and report medians alongside
minima, then re-read the tie. Then put **B=1 in `cases.txt`**: at 0.188 vs 0.213 the ungraded
cell is the *faster* one per transform, which means the graded cell is measuring something
other than the kernel.

**L=8 — the front end.** Aliasing is now excluded at <0.5% (§5) and there are zero spills, so
the ~18% over the 1072-cycle port floor is instruction delivery and dependency latency.
`L8_fusedaxes` names the one untried lever that fits gcc's register limits: split each phase-B
iteration's two `dft8s` calls across **adjacent y-iterations** — a 2-stage software pipeline at
codelet granularity, ~24 live zmm rather than ~40 — to cover the `trans8 → dft8s → untrans`
chain. It is cheap to prototype and the arena prices it in one run. Second, and arguably first:
L=8 is the **only** geometry where a library leads non-batched (MKL 0.546 vs 0.554), and that
cell has never been scored. Grade it before claiming L=8.

**L=17 — the X pass.** This is the geometry with the largest margin (5.71×) and the only real
gain of the round, so keep the momentum on the winner and do not re-litigate Rader (1.48%
behind at 1.48×, kernel already shared with the Winograd entry). `L17_matrixsimd`'s own
`b1dec` decomposition puts yz/kyz at ~7.0–8.0 µs against x/kx at ~3.2–3.6 µs, with the 73 X
chunks loading from L3 inside the chain; its record's instruction is to read the scored `b1dec`
and, if kyz dropped toward the port floor, attack X. Do exactly that.

**L=36 — build the L2 batch tile.** This is the one construction §4.3's reopened clause names,
it crosses the 7× L2↔DRAM gap rather than the 2.6× L1↔L2 gap every previous panel experiment
crossed, and **no entry has built it**. The working set is 11.39 MiB against 1.25 MiB L2, and
both leaders pass over the whole volume. `L36_pencilfused` has already written down why its own
shape cannot escape — "pass B's full-volume working set is a theorem, not a choice" — which is
precisely the argument that the remaining move is a *batch* tile, not a volume tile: tile the
batch so a tile fits L2, run all three axes inside the tile, price it against 109.619. Cheaper
second item: port `L36_mixedradix`'s `pind = 2112` anti-alias knob into `pencilfused` and `pfa`
and check whether the 22% it bought at B=1 survives at B=8. The honest prior is that it does
not — mixedradix itself gained 0.5% at B=8 — but it costs one run and the B=1 payoff is the
largest single number this round produced.

**Cross-cutting, and I would rank it above three of the four items above: put `B=1` rows in
`cases.txt`.** Three ice rounds have graded only the batched cell. The consequences are already
visible in the dev data: a library leads at L=8 non-batched, and the L=36 ranking inverts. Two
entries have asked for it twice. It costs one line per geometry in `cases.txt`.

## 7. What to keep

Against CURATION's four grounds, in order.

**1 — Fastest correct entry per geometry, always.** `L6_unrolled` (0.213), `L8_fusedaxes`
(0.544), `L13_rader` (4.619), `L17_matrixsimd` (13.061), `L36_pencilfused` (109.619),
`L45_mixedradix` (259.287), `L64_blocked` (902.885). At **L=23** the fastest entry is
`L23_rader` (39.502) and it is barred — no strategy record, third round running (§3) — so the
slot goes to `L23_matrixsimd` (39.761, 0.65% behind), exactly as ice_r2 ruled.

**2 — Structurally different runner-up when close.**
* `L8_batchsimd` (0.547, 0.55% behind) — radix-8 split against fusedaxes' fused-axes shape.
  Both were revised this round and each produced a distinct measured result (fusedaxes the
  §4.5 closure, batchsimd the depth-4 AA3 A/B), so the near-duplicate bar that ice_r2 waived
  by exception is cleared on merit this time.
* `L13_direct` (4.623, 0.09% behind) — dense conjugate-folded 13×13 against Rader-13 CRT. A
  genuine dead heat between the two structures §4.2 argues about, at a different prime. Both
  revised.
* `L17_winograd` (16.240, 1.24×) — the hand-derived module, and the only written-down instance
  of §4.2(b)'s symmetric/antisymmetric split. It is the datum that makes the §4.2 answer above
  a measurement instead of an assertion, and it was revised this round.
* `L64_radix8` (949.492, 1.05×) — radix-8²/axis **split-complex**, against blocked's 8×8
  two-stage. It is the only split-complex entry on the board, and both L=36 records name
  split-complex as their next structural lever while pointing at it ("L64_radix8 already runs
  split-complex here"). Keep it as the §4.4 reference the next round has to read.

**3 — Instructive, with the number.** `L36_mixedradix` (117.417, third in the graded cell) is
promoted on this ground and rule 4. It carries the round's largest single measured effect — the
22% 4K-aliasing win at B=1, `pind = 2112`, 122.48 → 95.39 with MKL held steady across the pair
and the cross-call collision mechanism derived — and it **leads the non-batched cell at L=36 by
8%** while placing last in the batched one. That inversion is the strongest argument in this
verdict for grading B=1, and it needs to be in front of the next panel as code, not prose.

Deliberately **not** promoted as runners-up:
* `L6_pfa` (0.213, dead tie) — structurally the same Good-Thomas PFA 2×3 as `L6_unrolled`, so
  CURATION's near-duplicate bar applies, as it did in ice_r1. Its clock-regime forensics are
  the most useful thing it produced this round and they survive in `strategies/L6_pfa.md`,
  which is tracked regardless of promotion.
* `L36_pfa` (111.568, 1.02×) — the two-sweep PFA 4×9 slot at L=36, but `L36_mixedradix` is the
  same structural family and is the one that moved a number. `L36_pfa.c` is byte-identical to
  r2 and is already in `exemplars/ice_r2`; re-promoting it would add a duplicate directory and
  nothing else.
* `L45_pfa` (293.720, 1.13×) — also PFA 9×5, 13% behind, code unrevised.
* `L17_rader` (19.368, 1.48×) — the instructive failure that answers §4.2(a), but its code is
  unchanged since r1, its number is unchanged, and it already sits in `exemplars/ice_r1` with
  its record. Cited in §5 rather than re-promoted.
* `L23_rader` (39.502, fastest at L=23) — barred by the missing record.

**4 — Anything that beat a library.** All nineteen panel entries beat the best library at their
geometry in the batched cell, so this ground does not discriminate. It is worth recording that
the panel leads at all eight geometries for the second consecutive round — and that the one
place a library is ahead, L=8 non-batched, is not on the leaderboard.

### Operational notes before running `promote.sh`

* All fourteen names below have a `strategies/<name>.md`, so `promote.sh` will not warn.
* **Blocking, before `exemplars/ice_r3` is committed:** `L6_unrolled`, `L13_direct` and
  `L13_rader` must add an ice_r3 section to their records (§3). Promoting `impl_3` code under an
  ice_r2 record is precisely the failure CURATION's record rule exists to prevent, and
  `L6_unrolled` is the L=6 winner.
* `exemplars/ice_r3/NOTES.md` should record: (a) that `L23_matrixsimd`, `L64_radix8` and
  `L36_mixedradix` are byte-identical carry-overs re-measured but not revised, so their r3
  leaderboard rows are drift probes, not results; (b) that `L23_rader` is again the fastest
  L=23 entry and again withheld for want of a record — the third round, and it should now be
  treated as a standing debt rather than a note; (c) that the round's r2→r3 noise floor is
  −1.5%…+1.5%, measured from eleven byte-identical entries, and that any future round's deltas
  should be read against it; (d) that `run spread` in the leaderboard is min-of-min and hides
  within-run bimodality up to 6.7%.
* The brief for ice_r4 needs §0's correction: the node is a Xeon Gold 6326 (Ice Lake-SP, two
  512-bit FMA pipes, 1.25 MiB L2), implementers develop on it directly, and the
  Sapphire-Rapids/Cascade-Lake dev-vs-score framing no longer describes this panel.
