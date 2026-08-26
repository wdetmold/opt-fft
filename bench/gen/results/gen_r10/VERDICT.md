# VERDICT — round gen_r10

Monitor's judgement on the measured board at
`bench/gen/results/gen_r10/leaderboard.txt`.

---

## 0. Two corrections to the monitoring brief, before any numbers

The brief I was given describes a different panel and a different machine. Both are
checkable against the artefacts in this directory, and both change what the report can say.

**0.1 — The geometries are not L = 6, 8, 17, 36.**
Those are the `bench/geom` panel's four fixed cubes (rounds `panel_r1`…`panel_r11`), and
they are the geometries `docs/CURATION.md` and `docs/LITERATURE.md` are written around —
CURATION.md's own worked example is `./promote.sh panel_r1 L17_rader L8_batchsimd L36_pfa`.

This round is `gen_r10` in `bench/gen`, whose case list is pinned by
`bench/gen/cases.txt`, headed *"THE GENERALIZE ACCEPTANCE SUITE — none of these sizes was
ever tuned by any prior round"*:

```
10:64:1000  12:64:600  15:32:600  20:32:256  25:16:256  27:16:200
31:16:140   32:8:250   40:8:128   50:4:128   100:1:64
```

Eleven cells: L ∈ {10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100}. `sweep.sh` still carries a
hard-coded `6:1 8:1 17:1 36:1 …` fallback in its `else` branch, but `cases.txt` exists, so
that branch never ran. I report on the eleven cells that were actually measured.

**0.2 — This board was scored on Ice Lake, not Cascade Lake.**
`environment.txt` reads:

```
host: a81n2.lqcd.mit   slurm_job: 438854
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: … avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq …   (Ice Lake-SP)
```

The Cascade Lake Gold 5218 is host `p52n1`, used for the cross-architecture advisory round
`xarch_clx_r6`. It is not this round's scoring host. The `avx512_vbmi`/`vnni`/`vpopcntdq`
flags above are Ice Lake-only and are the cheap discriminator.

This matters for item 4, because the correct translation is **SPR dev host → ICL scoring
host**, not SPR → CLX. Measured on this campaign's own boards, oneMKL 2022 spans:

| L | SPR (wallaby, 6448Y) | ICL (a81n2, 6326) | CLX (p52n1, 5218) | SPR→ICL | SPR→CLX |
|---|---|---|---|---|---|
| 10 | 3.608 | 4.556 | 5.816 | 1.26× | 1.61× |
| 20 | 43.013 | 57.781 | 72.765 | 1.34× | 1.69× |
| 31 | 575.757 | 848.710 | 1224.066 | 1.47× | 2.13× |
| 32 | 138.325 | 169.544 | 209.486 | 1.23× | 1.69× |
| 100 | 5380.666 | 7822.982 | 10222.787 | 1.45× | 1.90× |

So the working rule for reading a wallaby number onto this board is **×1.25–1.5**, and the
widest span anywhere in the campaign is 2.13× (SPR→CLX at L=31), not 2.9×.

**0.3 — "batched and non-batched for each geometry" does not exist in this suite.**
Each L is scored at exactly one batch size. The only non-batched cell on the board is
**L=100, B=1**. Where implementers measured B=1 themselves I quote it and mark it
*unscored*. Section 6 asks for this gap to be closed.

Everything below is from the measured board unless explicitly marked as an implementer's
own dev-window reading.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

All entries below carry `ok` correctness on the board. Two library columns are given
because `fftw3_custom` and `fftw3_custom_soa` are *project-built* genfft codelet baselines,
not stock library calls; the "stock" column is the best of MKL 2022 / MKL 2026 / FFTW
estimate-measure-patient-guru / ducc0.

| L | B | fastest correct entry | µs/xform | best library (any) | best **stock** library | speedup vs any / vs stock |
|---|---|---|---|---|---|---|
| 10 | 64 | **gen_pfa_small** | 1.115 | fftw3_custom_soa 4.457 | mkl_dfti 4.556 | **4.00×** / 4.09× |
| 12 | 64 | **gen_race** | 1.914 | mkl_dfti 7.734 | mkl_dfti 7.734 | **4.04×** / 4.04× |
| 15 | 32 | **gen_pfa_small** | 4.325 | fftw3_custom_soa 15.396 | mkl_dfti 16.460 | **3.56×** / 3.81× |
| 20 | 32 | **gen_batchlane** | 12.571 | fftw3_custom_soa 41.777 | fftw3_measure 44.928 | **3.32×** / 3.57× |
| 25 | 16 | **gen_powp** | 30.863 | fftw3_custom_soa 76.782 | fftw3_patient 107.868 | **2.49×** / 3.50× |
| 27 | 16 | **gen_race** | 42.837 | fftw3_custom_soa 95.794 | mkl_dfti 144.186 | **2.24×** / 3.37× |
| 31 | 16 | **gen_race** | 84.519 | fftw3_custom_soa 208.612 | ducc0_c2c 714.181 | **2.47×** / 8.45× |
| 32 | 8 | **gen_pow2** | 54.478 | mkl_dfti 169.544 | mkl_dfti 169.544 | **3.11×** / 3.11× |
| 40 | 8 | **gen_pfa_large** | 159.584 | mkl2026_dfti 403.529 | mkl2026_dfti 403.529 | **2.53×** / 2.53× |
| 50 | 4 | **gen_powp** | 417.201 | mkl_dfti 948.098 | mkl_dfti 948.098 | **2.27×** / 2.27× |
| **100** | **1** | **gen_pfa_large** | **4529.429** | mkl_dfti 7822.982 | mkl_dfti 7822.982 | **1.73×** / 1.73× |

**The panel beats every library at every one of the eleven cells**, by 1.73× (L=100, the
one non-batched cell, DRAM-resident) to 10.04× (L=31 against MKL, where MKL and all four
FFTW planners fall off a cliff at the prime axis).

Three of the eleven cells are won by `gen_race`, which is a race/routing layer, not a
kernel. Its banked verdicts (`results/wisdom_a81n2.json`) show what it actually ran:

* L=12 → `gen_batchlane` (tie, margin −0.6%)
* L=27 → `gen_powp` (margin +54.6%)
* L=31 → `gen_rader` (margin +28.9%)

So the *kernels* behind the eleven headlines are: gen_pfa_small ×2, gen_batchlane ×2,
gen_powp ×3, gen_rader ×1, gen_pow2 ×1, gen_pfa_large ×2.

Non-batched side-readings the board does not score, from implementer records (dev windows,
not comparable to the table above): L=12 B=1 5.625 µs (gen_race, sd 20%); L=32 B=1 55.43
(gen_pow2); L=31 B=1 85.31 (gen_rader); L=40 B=1 176.3 and L=50 B=1 465.0 (gen_pfa_large).
The B=1 path runs roughly 1.5–3× behind the batched path at the small cells and is
**untested by the acceptance suite below L=100**.

---

## 2. What changed since gen_r9, per geometry

Deltas are r9 board → r10 board, same host (a81n2), same suite, same seed protocol. The
board's own run-spread column bounds what counts as signal; anything under ~1.5% at these
cells is weather.

| L | cell winner r9 → r10 | Δ | the round's real movement in this cell |
|---|---|---|---|
| 10 | 1.121 (batchlane) → **1.115** (pfa_small) | −0.5% | **gen_pfa_small −2.4%** (1.142→1.115) — adopted gen_batchlane's r9 factor swap and took the cell. gen_batchlane bit-identical (1.121→1.121). |
| 12 | 1.913 → **1.914** | +0.05% | Nothing. Three engines inside 0.16%. gen_dense_prime **−5.5%** (8.147→7.697) off the pace. gen_twiddle +1.1% (its own documented code-layout tax). |
| 15 | 4.326 → **4.325** | −0.0% | gen_pfa_small **−1.4%** and takes the cell. gen_dense_prime **−3.7%**. |
| 20 | 12.549 → **12.571** | +0.2% | gen_pfa_small **−3.2%** (SWAP20, see §4). gen_batchlane −0.8%. **gen_race +1.7% — regression, see §3.** |
| 25 | 31.054 → **30.863** | −0.6% | gen_race −1.4%. **gen_planner +2.8% — regression.** |
| 27 | 43.607 → **42.837** | −1.8% | gen_race **−2.0%** and takes the cell from gen_powp (+0.3%, noise). |
| 31 | 84.694 → **84.519** | −0.2% | Flat everywhere. gen_rader −0.4% on a bit-identical binary; gen_dense_prime −0.1%. The cell is frozen. |
| 32 | 53.999 → **54.478** | +0.9% | **gen_twiddle −20.7%** (196.497→155.828) — the round's largest single gain, and it now **beats MKL** (169.544). gen_pow2 −1.2% on a *byte-identical* binary. **gen_race +1.7% — regression.** |
| 40 | 159.721 → **159.584** | −0.1% | **gen_twiddle −14.9%** (336.718→286.426). gen_planner −1.6% (under its claim, see §4). |
| 50 | 417.400 → **417.201** | −0.0% | **gen_pfa_large −4.3%** (437.059→418.457) — the r9 anomaly resolved, exactly as their record predicted. |
| 100 | 4516.288 → **4529.429** | +0.3% | gen_powp **−3.1%** (4807→4660) after their pick fix, but short of target. gen_twiddle −3.1%. **gen_race +3.1% and gen_layout +2.1% — regressions.** |

### Did anything regress?

Yes — four panel regressions above the noise floor, all diagnosable:

1. **`gen_race` L=100: 4516.3 → 4655.9 (+3.1%).** Not a code regression. The banked
   `eng10/L100/B1` verdict is `gen_powp` with `tie=1, margin=−0.002258` — an honest 0.2%
   tie between gen_powp and gen_pfa_large. It routed to gen_powp, and gen_powp scored
   4659.8 while gen_pfa_large scored 4529.4. A 0.2% tie in the race window paid a 2.8%
   penalty in the scored window. gen_race's own record predicted exactly this ("the
   documented honest ~0.7% tie at 100"). **The routing layer is correct; the tie is real
   and the cell's measurement noise is larger than the tie.**
2. **`gen_race` L=32: 54.0 → 54.9 (+1.7%) and L=20: 12.55 → 12.77 (+1.7%).** Both route
   correctly (`eng10/L32` → gen_pow2 margin +70%; `eng10/L20` → gen_batchlane, tie) and
   both land ~0.8–1.6% above the engine they forward to. This is forwarding overhead plus
   window, not a mis-pick. Worth one look at the vtable-forwarding path.
3. **`gen_planner` L=25: 39.589 → 40.688 (+2.8%).** No planner change touches L=25 (their
   record: every shipped path bit-identical to r8). Banked tree is unchanged
   (`c5(d5)@s1`). Attributed to window; the 2.6% run spread on the r10 read covers it.
4. **`gen_layout` L=100: 9131 → 9327 (+2.1%).** Their only behaviour change this round is
   the L=10 tail path (`GL_M4T_MAX` → 0); L=100 executes identical code. Their own dev
   reading was 9365.5 against an r9 board of 9355 — i.e. they measured it flat. Window.

For calibration on how much of this is weather: **`gen_pow2` shipped a binary that is
byte-identical to r9 (`cmp` of the compiled executables, their record) and its board number
moved −1.2%.** That is the honest floor on cell-to-cell reproducibility here, and it is
large enough to have flipped the L=32 ordering between gen_race and gen_pow2 between rounds.

Library-side movements worth noting so nobody reads them as panel effects: `fftw3_measure`
+8.2% at L=25 and +9.8% at L=100, `mkl2026_dfti` +2.6% at L=32, `fftw3_estimate` +2.5% at
L=40 — the libraries drifted too, in both directions.

---

## 3. Correctness, build, crash and coverage audit — adversarial

**No entry failed correctness.** `check.log` is 734 lines with **zero** `FAIL`, `ERROR` or
`MISMATCH` tokens; every one of the 176 board rows carries `ok`, and every row's chain drift
sits 1.0–2.5× its honest anchor against a `1e-10` tolerance, with single-transform `rel_l2`
between 2.3e-14 and 4.9e-16 against a `1e-12` tolerance. There is no fast-and-wrong entry to
strike.

Named findings, with evidence:

**3.1 `baseline_matrix` — MISSING from the L=100 cell, killed by timeout.**
`failures.txt`, verbatim and complete:
```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```
Exit 124 is `timeout 600`. All three runs hit the wall, so the L=100 board has 14 rows where
every other cell has 15–18, and the harness floor is absent at the one cell where the floor
would be most informative. This is the harness, not an entry: `sweep.sh` already skips
`baseline_matrix` at some sizes as *"too expensive to be informative"* but does not skip it
at 100³, so the round burned 30 minutes of exclusive node time producing nothing. Fix the
skip predicate before the next round.

**3.2 `gen_rader` — the only compiler diagnostic in the round, and it is an
undefined-behaviour warning.** `build_errors.txt` contains no errors, but it is not empty:
```
impl/gen_rader.c: In function ‘map_volume.constprop’:
impl/gen_rader.c:2402:23: warning: iteration 1152921504606846976 invokes undefined
                          behavior [-Waggressive-loop-optimizations]
 2402 |         double re = zp[2 * i] + cp[2 * i];
      |                       ^
impl/gen_rader.c:2401:14: note: within this loop
 2401 |     for (; i < npts; ++i) {
```
`1152921504606846976` = 2^60, i.e. GCC has proven that `2 * i` overflows before `i` reaches
`npts` on some constant-propagated instantiation of `map_volume`, and is therefore licensed
to assume the loop terminates earlier — or not at all. It did not misbehave: gen_rader
passed every gate, its L=31 output is bit-identical to r8/r9 by `cmp`, and the board reads
84.838 µs. But this is a real UB flag on a promoted entry's map tail, and `-Waggressive-loop-optimizations`
is precisely the class of warning that turns into a wrong answer under a compiler change.
**gen_rader owes the next round an `i`-type widening or an explicit trip-count bound.** I am
promoting the entry — the warning is in a constant-propagated clone of a tail loop that the
gates exercise, and every measured output is bit-identical to three prior rounds — but it is
named here so it cannot be quietly inherited.

**3.3 Nothing crashed, hung, or vanished silently.** `agents/exits.txt` shows all twelve
implementers `exit=0`. `timing.err` contains 63 lines and every one of them is a declared
`"<entry>: does not support L=<n>"` refusal — no signal, no abort, no timeout. Cross-checked
against the board: every absence is a declared class boundary (gen_rader is prime-only,
gen_pow2 is 2^k-only, gen_pfa_small/gen_batchlane stop at 20, gen_dense_prime stops at 31,
gen_pfa_large starts at 40, gen_powp is prime-power). gen_bluestein, gen_layout,
gen_planner, gen_race and gen_twiddle appear at all eleven cells. `sweep.out` ends
`== round gen_r10 complete ==`.

**3.4 One entry's whole round is invisible to this board.** `gen_batchlane` spent gen_r10
extending its factor swap to L ∈ {21, 22, 28, 33, 35, 44, 55} and measured −1.7% to −4.7%
across that family on this host. **The acceptance suite contains none of those sizes.**
Their scored cells (10/12/15/20) shipped deliberately bit-identical to r9 and the board
confirms it to three decimals (10: 1.121→1.121; 12: 1.917→1.917). So their claims are
uncorroborated by measurement I control — not disputed, just unscored. Same for
gen_bluestein's L=63/96 radix-16 gate work (−7.7%/−9.8%) and gen_rader's L=97 adoption
(−4…6%). This is a suite-coverage gap, not an implementer failure.

---

## 4. Implementer claims vs. what was measured

The dev host is wallaby (Xeon Gold 6448Y, Sapphire Rapids, 2 MB L2/core, two 512-bit FMA
pipes). The scoring host is a81n2 (Xeon Gold 6326, Ice Lake-SP, 1.25 MB L2/core, one
512-bit FMA pipe). Per §0.2 the MKL span between them is 1.23–1.47×, so absolute wallaby
numbers translate at roughly ×1.25–1.5 — but the interesting divergences this round are
**sign inversions**, which no scaling factor fixes.

**4.1 `gen_pfa_small` — the round's cleanest SPR→ICL inversion, and the bet paid.**
The Ice Lake nodes were queued-busy for their whole session, so every number in their record
is wallaby. They claimed, on SPR:

| cell | their SPR delta | board delta (ICL) |
|---|---|---|
| L=10 | −1.6% | **−2.4%** |
| L=15 | −0.6% | **−1.4%** |
| L=20 | **+1.6% — "SPR FLIPS IT"** | **−3.2%** |

They shipped `SWAP20=1` anyway, explicitly overriding their own dev host on the strength of
gen_batchlane's 5/5 ICL race, and recorded the reasoning: *"llvm-mca's ICL model calls it a
wash… the model cannot veto the node."* **The board vindicates them at all three cells**,
and L=20 is a genuine sign flip between microarchitectures — not a magnitude difference. The
plausible cause is the one their record names: the swapped stage-2 fuses the map into a
narrower codelet, which trades register pressure against ILP, and SPR's second FMA pipe and
larger L2 change that trade's sign. This is a per-host knob (`-DSWAP20=0` is the SPR revert),
correctly identified as such.

**4.2 `gen_layout` — the same inversion, running the other way.**
Their r9 SPR advisory said the 4-lane ymm exit tail (`m4t`) wins −3…4% at L=10. The ICL A/B
this round measured the zmm form winning both pairs (−2.2%, −0.8%), so they flipped
`GL_M4T_MAX` to 0 and documented it as *"the first clean case this campaign of an SPR-advisory
A/B verdict INVERTING on ICX at the same cell."* Board: L=10 4.952 → 4.972 (+0.4%), inside
the ±1.5% code-layout band they stated. Claim and measurement agree. Two independent SPR→ICL
inversions in one round, on opposite sides, is the strongest evidence yet that **wallaby A/Bs
on port-mix and vector-width questions are hypothesis generators only.**

**4.3 `gen_twiddle` — SPR understated the ICL win by ~7 points, same sign.**
Claimed on the node: −21.4% at L=32 (7/7 pairs), −14.5% at L=40 (6/6). Board: **−20.7%** and
**−14.9%**. Excellent agreement. Their *wallaby advisory*, run first, predicted −13%/−9% —
i.e. the SPR machine understated the ICL benefit of deleting a whole radix-8 combine level by
about 8 and 6 percentage points. That is the expected direction: deleting a pass removes an
L1 round trip per pencil group, and ICL's smaller L2 and single FMA pipe make that round trip
proportionally more expensive. Their claim to now beat MKL at L=32 (158.256 vs 177.279 in
their window) is confirmed and improved on the board: **155.828 vs 169.544.**

**4.4 `gen_powp` — right diagnosis, right fix, target missed by 2.4% to measurement noise.**
Their round was a forensic fix to the L=100 pick: r9 banked `l100-ipk1` through a 3% rank
hysteresis even though its own challenger playoff had put `ipp1` 1.6% ahead. They rebuilt the
playoff (authoritative decided verdicts, leader playoff even when rank 0 wins, cache-regime
warm-step fidelity) and asked me to *"verify on the r10 board that L=100 recovered to ~4550
and that wisdom holds a plain-name chain8 l100-ipp1."*

Both halves check out on the pick, one half misses on the number:
```
entries/gen_powp/chain8/L100/B1  →  winner = l100-ipp1, tie = 1, us = 4511.8, margin = 0.0065
```
**The machinery worked**: the scoring window's cold race picked `ipp1`, and the race's own
timing of it was 4511.8 µs — squarely on their ~4550 target and level with gen_pfa_large.
The board nonetheless scored gen_powp at **4659.795 with a 10.9% run spread**, the widest
spread of any panel entry anywhere on the board. So the residual 2.4% is not a pick error and
not a machine difference; it is the L=100 B=1 cell being the noisiest thing we measure. See
§6.

**4.5 `gen_planner` — the round's one shipped win did not land, and the wisdom file says
why.** They claimed the tile playoff (`t16` vs the fixed `t32` since r3) is worth −2.5…4% at
L=40 and −2.6…3% at L=50, confirmed 3/3 on the node. Board: **−1.6% at 40 and −0.3% at 50.**
The banked trees explain it:
```
entries/gen_planner/tree/L40/g8  →  c2(gt(d4,d5))      ← no @t16
entries/gen_planner/tree/L50     →  c5(gt(d2,d5))      ← no @t16
```
Neither cell carries the `@t16` tag, so the scoring-window cold create kept `t32`. Their own
record predicted this exact failure mode and recorded it honestly: *"in a CONTENDED window a
cold create at 40/50 may keep t32 (settled-vs-capped varies with neighbor load)."* The
adoption gate requires the playoff to settle at `max(2%, 2× trial noise)`; at 40/50 it capped
out instead. **A measured, built, bit-neutral 2.5–4% win is sitting on the table uncollected
because the scoring window was not quiet enough for the playoff to settle.** Not machine
difference — measurement fidelity. (gen_race's independent `tile10` race *did* bank `t16` at
both cells, which is the same verdict arriving through a different layer.)

**4.6 `gen_pfa_large` — claim confirmed, and it retires an r9 mystery.** They asserted the r9
L=50 read of 437.06 was *"a sustained-slow scoring-slot artifact, not a pick error"*, refuted
the re-ranking hypothesis 4/5 pairs plus a nine-arm family sweep, and put the true floor at
410–417. Board: **418.457, −4.3%**, with the banked `l50-ip1.ch` pick unchanged. Confirmed.
Their L=40/L=100 claims (floors 159.8 and 4501.2) also land: board 159.584 and 4529.429.

**4.7 Same-host claims that simply agree** (no machine difference to attribute): gen_rader
claimed 85.20 min at L=31, board 84.838; gen_bluestein claimed 170.3/272.9/601.7/1383.0/14132
across five cells and the board reads within 1–2.5% of every one; gen_dense_prime claimed
−1.3% at L=20 and −3…5% at 17/19/23/29, board −0.9% at 20 (their 17/19/23/29 cells are
off-suite); gen_batchlane claimed bit-identity at all four scored cells, board confirms.

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

### Primary: **§4.3 — "Is axis fusion worth 3× or 3%?"** — moved decisively, in the one regime the corpus called untested.

§4.3's own addendum, after `panel_r3` settled the L1↔L2 case at single-digit percent, says
the L2↔DRAM case is *"not the same experiment the panel ran, and it is the largest untried
structural move on the board"*, and names the construction: tile the batch so a tile fits L2,
then run all three axes inside the tile.

`gen_r10` ran that experiment at L=100 (30.5 MiB working set, DRAM-resident, the suite's one
non-batched cell) and the answer is **no, and now for two independent reasons**:

1. **gen_planner rebuilt the fused exit correctly and it still loses.** Their r6 fused exit
   lost at L>80 and they had blamed c-plane re-reads. Re-analysis found the real cause was
   walk order — the z0-outer exit *strides* both DRAM streams while streaming the L2-resident
   scratch. They built the repaired y0-outer form (bit-identical output) and it recovers most
   of the pathology: 5722–6018 → 5146–5589 µs. **The separate, unfused, pair-packed map pass
   still wins 3/3** (5073/5414/5466 vs 5146/5424/5589, 0.2–2.2%). Their stated mechanism:
   *"two sequential full-plane sweeps ride the prefetcher at effectively no cost, and the
   exit's strided P reads + vdivpd beat the 32 MB of traffic fusion deletes."* So the
   r6 mechanism story was wrong and the verdict survived anyway.
2. **gen_pfa_large showed there is no DRAM traffic left for fusion to delete.** Their traffic
   re-accounting at L=100 finds `ipp1` already at the 80 MB/step floor: phase 1 is
   per-x-plane, so the y-subpass writes exactly the plane lines the prepass just read
   (160 KB, L2/L3-hot) — there is no DRAM RFO to eliminate. They killed a planned NT-store
   hybrid on that basis before writing it, and measured the NT variant at **+64%** for good
   measure.

The board corroborates both: gen_planner ships `fm0` (unfused) at L=100 and scores 4987.5,
while `gen_race`'s independent three-way `fm10` race banked `fm0` at L=100 (margin 4.1%)
and `fm1` (fused) at 25/27/31/32/40/50. **That is the regime boundary, measured:
fusion wins while the tile is L2-resident and loses once the volume is DRAM-resident.**
Which is Tolmachev's rule from §07 §1.6 — payoff = avoided passes × the bandwidth gap between
the levels involved — coming out the *other* way than the GPU literature predicted, because
at L2↔DRAM the prefetcher already gets the sequential passes for free.

`gen_race` also recorded a live cross-architecture flip on the same question: the same
three-way race picks `fm1` at L=100 on wallaby (SPR) through a confirmed noise-gate upset.
So the boundary is host-dependent, and the campaign now has the knob and the banking layer
to carry that per host.

### Secondary, both moved:

* **§4.6 — "Model versus search for the instruction schedule."** §4.6 concludes the schedule
  is *"the primary thing to search"* at every size. Three independent results this round say
  search, and specifically *node* search, is the only arbiter that holds. gen_pfa_small:
  llvm-mca's Ice Lake model scores the L=20 swap a wash (58860 vs 58864 cycles) while the
  node reproduces the win 5/5 and the board delivers −3.2% — *"use the model for port
  pressure, never to veto a clean node race."* gen_dense_prime: a cost model predicted 5–7%
  from paired divides at every generic cell and measured −1.3…−3% and washes, because
  *"cost models built on a pass timed IN ISOLATION overstate what deleting its bottleneck
  buys in situ."* gen_layout: their PMU dashboard shows L=25/32 running at the node's ~2.1
  all-port uops/cycle dispatch cap with p0+p5 under 1.0, so *"stop rebalancing ports and
  start deleting uops."* Three different entries, three different modelling failures, all
  caught only by measurement.

* **§4.8 item 6 — the AVX-512 gap** — extended past its own closure. gen_pfa_large's
  `portcal3` microbenchmark, run this round on the scoring node, measures 8 zmm FMA chains +
  K ymm streams at **exactly (8+K)/2 cycles** (K = 2, 4, 8, 12 → 5.05, 6.00, 8.00, 10.00).
  256-bit FP steals 512-bit FMA slots 1:1 on Ice Lake, identically to SPR. Combined with
  gen_layout's ~2.1 uops/cycle global dispatch ceiling, this **permanently closes the
  "port 1 idles, side-work co-issues nearly free" avenue** — four separate entries had it on
  their lists and all four struck it this round without spending a build.

Not moved: §4.2 (L=17 dense-vs-Rader — off-suite here, though gen_dense_prime's −3…5% at
17/19/23/29 is adjacent evidence), §4.4 (split-vs-interleaved, already closed in §08 §5.4),
§4.5 (L=8 padding — off-suite), §4.7 (vector-radix — untouched, correctly).

---

## 6. The single highest-value thing the next round should attack, per geometry

Ranked by measured headroom, not by interest.

| L | highest-value attack | why, with the number |
|---|---|---|
| **10** | **Score a B=1 cell.** | Three engines within 0.7% and the winner moved −0.5% in a round. The batched cell is done. The unscored B=1 path is where pfa_small's r7/r8 split-chain work lives and no board has ever measured it. |
| **12** | **Same — B=1.** | Board spread across gen_race/pfa_small/batchlane is 0.16%. The only in-cell idea left on any record is a swapped-12 pencil under a *different* port structure; both batchlane and pfa_small measured swap-12 losing 3.5–4.7% here. That is CLX/SPR material, not an ICL round. |
| **15** | **Same — B=1.** | Winner moved 0.0%. gen_dense_prime gained 3.7% and is still 3.4× off. Converged. |
| **20** | **Fix `gen_race`'s forwarding cost.** | It routes correctly to gen_batchlane and still lands 1.6% above it (12.767 vs 12.571), and regressed 1.7% from r9. Same symptom at L=32. Cheap, and it is currently costing the routing layer cells it has already won. |
| **25** | **Winograd / real-factor DFT25 module in gen_planner's fused codelets.** | gen_planner is 1.32× behind gen_powp (40.688 vs 30.863) and its own record has named this as the top unspent arithmetic item for **four rounds**. The general-L path pays this gap at every prime-power size. |
| **27** | **Same — DFT9/DFT27 module in gen_planner.** | 1.40× behind gen_powp (59.771 vs 42.837), same root cause, same four-round-old item. 25 and 27 are one piece of work. |
| **31** | **Adopt gen_rader's folded cyclic-convolution module as a gen_planner `@s` level.** | The class itself is converged — gen_rader has been at its port model since r4 and gen_dense_prime's PMU (1.44/2.0 p0+p5, 0.15 l1d lines/cyc) says the residual is drain structure, not traffic. But gen_planner is **1.65×** behind at this cell, and it is the only entry that covers arbitrary L. Move the general path, not the champion. |
| **32** | **Port level-deletion into gen_planner's 2^k path.** | gen_planner sits **1.94× behind gen_pow2** (105.877 vs 54.478) — the largest gap between the general planner and a class champion anywhere on the board. gen_twiddle just proved the lever on this exact size: one radix-8 level deleted was worth **−20.7%** and took its demo past MKL. |
| **40** | **Make gen_planner's tile playoff settle in a cold scoring create.** | §4.5: `t16` is built, bit-neutral, measured at −2.5…4% on the node, and **was not banked** (`tree/L40/g8` has no `@t16`). This is finished work not being collected. Either lower the settle threshold for bit-neutral knobs or have the monitor run a quiet pre-suite create. |
| **50** | **Same playoff fix, plus accept the cell.** | `tree/L50` also lacks `@t16`, and gen_powp / gen_pfa_large / gen_race are within 0.7% of each other (417.2 / 418.5 / 420.2). After the playoff lands, this cell is converged. |
| **100** | **Fix the measurement, not the code.** | This cell decided two of the round's four regressions. gen_powp scored with a **10.9% run spread**, the worst on the board; gen_race paid 2.8% for routing on a 0.2% tie; and `baseline_matrix` timed out three times at 600 s. Raise `m` and `--samples` at L=100 (or split it into its own longer job), and skip `baseline_matrix` there. Until the cell's noise is below its ties, no engine work at L=100 is measurable — and gen_pfa_large has independently shown the traffic is already at its 80 MB/step floor. |

**Two cross-cutting items, above every per-cell entry:**

1. **The acceptance suite has one non-batched cell out of eleven.** L=100 B=1 is the only
   place the board sees the unbatched problem, and implementer side-readings put B=1 at
   1.5–3× the batched cost at small L. Every "B=1 lane-spatial split engine" proposal on the
   panel (gen_planner has carried it for nine rounds; gen_batchlane and gen_pfa_small both
   defer it to "if a scored case ever lands at B<8") is unbuilt precisely because nothing
   scores it. **Add B=1 cells at 10/20/32 to `cases.txt`.** Note the suite is documented as
   FROZEN for the campaign, so this is a deliberate decision, not a tweak.
2. **Harness debt that eight rounds of implementers have reported.** `tryout.sh`'s remote
   map-check leg dies on an unexpanded `'$W/c.bin'` (reported every round since r1/r2 — one
   `sed` on the `CH=` line); `reserve.sh --status` fails on wallaby with
   `squeue: error: Unrecognized option: icehold`, which makes `tryout.sh` refuse to run for
   *every* implementer and forced most of this round's work through hand-replicated ssh
   sessions; `/tmp/perf` is node-local and has to be re-staged after every node move
   (`ext/tools/perf-install/perf-bin` → `a81n2:/tmp/perf`). These three fixes are worth more
   panel throughput than any per-cell item above.

---

## 7. Curation decision

Applying `docs/CURATION.md` in its stated order.

**1 — Fastest correct entry per geometry (always).** Eleven cells, six distinct kernels plus
the routing layer that won three:
`gen_pfa_small` (10, 15), `gen_race` (12, 27, 31), `gen_batchlane` (20), `gen_powp` (25, 50),
`gen_pow2` (32), `gen_pfa_large` (40, 100). — **6 entries.**

**2 — Structurally different runner-up, when close.**
`gen_rader` at L=31: 84.838 vs 84.519, **0.4% behind**, and it is the engine gen_race actually
routes to. It is the Rader/Winograd cyclic-convolution alternative to everything else on the
board and CURATION's worked example is this exact situation. — **+1 (7).**

**3 — Instructive failures whose record documents the number that killed them.**
`gen_dense_prime` at L=31: 109.865, **1.30× behind gen_rader**, and it is the *dense folded
GEMM* side of the crossover this class exists to settle. Its record now carries the PMU
number that closes it (1.44/2.0 p0+p5 dispatch, l1d.replacement 0.15 lines/cyc — drain
structure, not traffic), plus this round's paired-divide boundary: pairing pays only where
the map is a standalone max-ILP pass (−1.3…−3% at 17–29), and loses **+1.5%, 5/5** where the
divide was already in a compute pass's OoO shadow. That corollary was independently confirmed
against gen_layout's r5 result on a second engine. It also beats every library at L=31 by
1.9×. — **+1 (8).**

`gen_planner`: never wins a cell and is 1.10–1.94× off the pace everywhere, but it beats
every library at every one of the eleven cells, is the only entry covering arbitrary
2 ≤ L ≤ 128, and its gen_r10 record is the densest set of *measured refutations* in the round
— ZST transpose stores refuted on ICL (+2…6%, 3/3, with the 16-byte-alignment mechanism);
c-line custody refuted (+6…11% at L=100, 3/3) on the exact node its own sysfs heuristic had
predicted a win; the fused exit rebuilt correctly and still losing (§5). Those three
negatives are worth a round each to the next panel. — **+1 (9).**

**4 — Anything that beat a library baseline, regardless of rank.**
Read literally this promotes all twelve, since every panel entry beats some library
somewhere; read as intended it catches low-ranked entries that nonetheless cleared a library
bar. `gen_twiddle` is the case: it wins no cell, but it delivered the round's largest single
gain (**−20.7% at L=32, −14.9% at L=40**, by deleting a whole radix-8 combine level) and in
doing so **took its demo past MKL 2022 at L=32** (155.828 vs 169.544) from 1.14× behind in
r9. Its record also contributes the boundary on the round's most-borrowed trick:
extract-to-memory stores pay only where the consumer is far away, and cost +0.7–1% at a
staging buffer where the wide store *is* the store-forwarding path. — **+1 (10).**

**Build closure.** The entries above are not standalone C files; `impl_10` has a real include
graph:
```
gen_batchlane, gen_powp, gen_rader, gen_twiddle  →  #include "gen_layout.c"  (LIB_ONLY)
gen_powp, gen_pfa_large, gen_planner             →  #include "gen_race.c"    (LIB_ONLY)
gen_race (demo)                                  →  #include "gen_planner.c"
gen_bluestein                                    →  #include "gen_twiddle.c"
```
So `gen_layout` must be promoted for four of the ten above to compile at all. It also earns
its place on its own merits: it is the adopted arena/`gl_map`/`gl_tr8x8` layer under four
promoted entries, and its gen_r10 contribution is the PMU dashboard that reframed the
panel's optimisation target — L=25/32 running at the node's **~2.1 all-port uops/cycle** cap
with p0+p5 under 1.0, i.e. *"stop rebalancing ports and start deleting uops"* — plus the
cleanest SPR→ICL A/B inversion in the campaign (§4.2). — **+1 (11).**

**Not promoted: `gen_bluestein`.** It wins no cell; it is never within 20% of one (3.2×
behind at L=31, 11.5× at L=10); it *loses to stock MKL* at six of the eleven scored cells
(10, 12, 15, 20, 25, 27); no promoted entry depends on it; and its entire gen_r10 output —
the radix-16 fusion gate at L=63/96 (−7.7%/−9.8%) and the masked nv<8 tail pipeline — landed
on sizes the acceptance suite does not contain. Its record is good and stays tracked in
`strategies/`, which is where its negatives (2,8-split-radix deleted on gen_pow2's evidence;
the gather-side restructure declined on a static uops.info audit) remain available. If the
suite ever adds a non-smooth prime or a size like 101, revisit it immediately — it is the
only any-L chirp-Z code in the campaign.

**Round note for `NOTES.md` at promotion time:** gen_r10 was a *consolidation* round. The
board moved less than 1% at eight of eleven cells; the two real gains (gen_twiddle's radix-8
at 32/40, gen_pfa_small's factor-swap adoption at 10/15/20) came from adopting a peer's
mechanism rather than inventing one, which is the panel structure working as designed. The
round's most valuable output is not a speed: it is §5's regime boundary on axis fusion, the
permanent closure of the port-1 co-issue avenue on both graded architectures, and two
documented cases of an SPR advisory verdict *inverting* on the Ice Lake scoring host.

---

PROMOTE: gen_pfa_small gen_batchlane gen_powp gen_pow2 gen_pfa_large gen_rader gen_dense_prime gen_race gen_planner gen_layout gen_twiddle
