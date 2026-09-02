# VERDICT — round gen_r13

Monitor report. Measured on `a80n0.lqcd.mit`, slurm job 439820, 2026-09-01,
Xeon Gold 6326 (Ice Lake-SP), gcc 11.4.0, exclusive node.

---

## 0. Two corrections to the monitor brief before anything else

The brief I was given (`monitor_prompt.txt`, byte-identical to the one issued) is a
**stale template** on two points of fact. Both change what this document can say, so
they are recorded first.

1. **The geometries in the brief do not exist in this round.** The brief asks for
   headlines at `L = 6, 8, 17, 36`. Round gen_r13 scored **thirteen cells at eleven
   distinct L**: 10 (B=1 *and* B=64), 12 (B=1 *and* B=64), 15, 20, 25, 27, 31, 32, 40,
   50, 100. None of 6, 8, 17 or 36 is on the board, in this round or in gen_r11/r12.
   6/8/17/36 are the `bench/geom/panel_r*` geometry set, a different campaign.
   I report the geometries that were actually measured.

2. **The machine in the brief is the wrong machine.** The brief says "Xeon Gold 5218,
   Cascade Lake" and instructs me to attribute claim/measurement gaps to a
   Sapphire-Rapids-versus-Cascade-Lake difference spanning 2.9x in MKL.
   `environment.txt` says **Xeon Gold 6326, Ice Lake-SP** — 1.25 MB L2/core, and not
   a part with Cascade Lake's AVX-512 licence downclock. The implementers' own records
   agree and use the right name throughout ("the ICX race verdicts", "the scoring
   host", "SPR builds want `-DGL_M4T_MAX=16`").

   More importantly, **the premise behind the instruction no longer holds**: the
   implementers this round did *not* develop-and-claim on Sapphire Rapids. They took
   their A/B numbers on the scoring node itself via held slot leases on individual
   cores. `gen_pfa_small`'s record quotes `mkl_dfti 4.330-4.339` at 10:1 and the board
   reads **4.331**; it quotes `7.319-7.367` at 12:1 and the board reads **7.319**.
   That is a same-machine measurement, and it is why §4 below has almost no gaps to
   explain rather than the large ones the brief anticipated.

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

"Best library" = best of `mkl_dfti`, `mkl2026_dfti`, `ducc0_c2c`, `fftw3_{estimate,
measure,patient,guru}`. Per the project's standing rule, the two **custom-built**
FFTW configurations (`fftw3_custom`, `fftw3_custom_soa` — FFTW rebuilt with `genfft`
codelets generated for our exact sizes, which no real user ships) are reported in a
**separate column** rather than folded into "best library": the stock column answers
"what a user actually gets", the custom column answers "how good could the library be".
Where the custom build is the stronger of the two, both ratios are given.

Only L=10 and L=12 have both a non-batched and a batched cell; L=100 is non-batched
only; the rest are batched only. That is the board's shape, not an omission.

| L | case | fastest correct panel entry | best **stock** library | speedup | best **custom** FFTW | speedup |
|---|---|---|---|---|---|---|
| 10 | **B=1**, m=16384 | **gen_pfa_small 1.943 µs** (gen_race 1.943, same engine) | mkl_dfti 4.331 | **2.23x** | fftw3_custom 5.464 | 2.81x |
| 10 | B=64, m=1000 | **gen_pfa_small / gen_batchlane 1.147 µs** | mkl_dfti 4.652 | **4.06x** | fftw3_custom_soa 4.481 | 3.91x |
| 12 | **B=1**, m=12288 | **gen_pfa_small 2.999 µs** (gen_race 3.001) | mkl_dfti 7.319 | **2.44x** | fftw3_custom 8.627 | 2.88x |
| 12 | B=64, m=600 | **gen_race 1.947 µs** (pfa_small 1.952, batchlane 1.955) | mkl_dfti 7.912 | **4.06x** | fftw3_custom_soa 8.178 | 4.20x |
| 15 | B=32, m=600 | **gen_batchlane 4.409 µs** (race 4.415, pfa_small 4.420) | mkl_dfti 16.844 | **3.82x** | fftw3_custom_soa 15.939 | 3.62x |
| 20 | B=32, m=256 | **gen_race 12.846 µs** (batchlane 12.849, pfa_small 12.849) | fftw3_patient 44.871 | **3.49x** | fftw3_custom_soa 42.805 | 3.33x |
| 25 | B=16, m=256 | **gen_powp 31.369 µs** (race 31.378) | fftw3_patient 109.537 | **3.49x** | fftw3_custom_soa 76.278 | 2.43x |
| 27 | B=16, m=200 | **gen_powp 43.893 µs** (race 44.106) | mkl_dfti 146.844 | **3.35x** | fftw3_custom_soa 97.955 | 2.23x |
| 31 | B=16, m=140 | **gen_rader 86.176 µs** (race 86.909) | ducc0_c2c 726.558 | **8.43x** | fftw3_custom_soa 209.645 | 2.43x |
| 32 | B=8, m=250 | **gen_pow2 55.600 µs** (see §3 on gen_race's 55.189) | mkl_dfti 174.979 | **3.15x** | fftw3_custom_soa 176.191 | 3.17x |
| 40 | B=8, m=128 | **gen_pfa_large 161.068 µs** (race 162.172) | mkl2026_dfti 409.863 | **2.54x** | fftw3_custom_soa 424.893 | 2.64x |
| 50 | B=4, m=128 | **gen_pfa_large 418.260 µs** (powp 420.549) | mkl_dfti 956.976 | **2.29x** | fftw3_custom 1708.697 | 4.09x |
| 100 | **B=1**, m=64 | **gen_batchlane 4089.562 µs** (race 4113.269) | mkl_dfti 7861.152 | **1.92x** | — (not run at 100) | — |

**Every one of the thirteen cells is won by a panel entry, over both the stock and the
custom-codelet library columns.** The narrowest margin over a stock library is L=100 at
1.92x; the widest is L=31 at 9.86x over MKL (Rader-class primes remain the libraries'
worst geometry by a wide margin).

Both **new** cells this round — 10:1 and 12:1, added because benchFFT exposed a real
B=1 small-L weakness in round 12 — land at 2.2x and 2.4x over MKL. The weakness the
previous round found is closed as far as this harness measures it. See §6 for the part
that is *not* closed.

---

## 2. What changed since gen_r12 — and the drift control that interprets it

### The control comes first, because without it this round reads as a regression

Nine of the eleven repeated cells show the r13 winner **1–3% slower** than the r12
winner. Before attributing that to code, I checked the two backends that **cannot have
changed**: `baseline_matrix` (the library-free harness floor, source untouched) and the
two MKL builds (fixed binaries).

| cell | `baseline_matrix` r12 → r13 | `mkl_dfti` r12 → r13 |
|---|---|---|
| 10:64 | 54.687 → 56.299 (**+2.9%**) | +1.9% |
| 12:64 | 112.125 → 114.728 (**+2.3%**) | +2.3% |
| 15 | 272.409 → 278.560 (**+2.3%**) | +2.4% |
| 20 | 850.293 → 868.176 (**+2.1%**) | +1.9% |
| 25 | 2061.289 → 2101.963 (**+2.0%**) | +2.4% |
| 27 | 2799.025 → 2852.948 (**+1.9%**) | +1.9% |
| 31 | 4853.226 → 4931.723 (**+1.6%**) | +0.2% |
| 32 | 5752.899 → 5834.739 (**+1.4%**) | +1.9% |
| 40 | 13466.071 → 13650.081 (**+1.4%**) | +1.2% |
| 50 | 33723.072 → 34775.435 (**+3.1%**) | +1.0% |

**Unchanged code is uniformly 1.4–3.1% slower in the r13 window than in the r12
window.** This round's board sits on a node running roughly 2% slower than round 12's —
the reason is not visible from here (governor is `schedutil` in both rounds; ambient
thermal or firmware state are the obvious candidates) but the fact is not in doubt: it
is two independent fixed backends moving together, at every cell, in the same direction.

This is the two-sided control the project's verification discipline asks for. Applying it:

| cell | winner r12 → r13 | raw | drift | **net** | reading |
|---|---|---|---|---|---|
| 10:1 | — → gen_pfa_small 1.943 | — | — | — | **new cell** |
| 10:64 | 1.116 → 1.147 | +2.8% | +2.9% | **−0.1%** | flat |
| 12:1 | — → gen_pfa_small 2.999 | — | — | — | **new cell** |
| 12:64 | 1.912 → 1.947 | +1.8% | +2.3% | **−0.5%** | flat |
| 15 | 4.324 → 4.409 | +2.0% | +2.3% | **−0.3%** | flat |
| 20 | 12.552 → 12.846 | +2.3% | +2.1% | **+0.2%** | flat |
| 25 | 31.698 → 31.369 | −1.0% | +2.0% | **−3.0%** | gain |
| 27 | 44.031 → 43.893 | −0.3% | +1.9% | **−2.2%** | gain |
| 31 | 84.753 → 86.176 | +1.7% | +1.6% | **+0.1%** | flat |
| 32 | 54.851 → 55.600 | +1.4% | +1.4% | **0.0%** | flat |
| 40 | 159.253 → 161.068 | +1.1% | +1.4% | **−0.3%** | flat |
| 50 | 413.440 → 418.260 | +1.2% | +3.1% | **−1.9%** | gain |
| 100 | 4084.110 → 4089.562 | +0.1% | +0.9%¹ | **−0.8%** | flat |

¹ `baseline_matrix` timed out at L=100 (see §3); MKL used as the control instead.

**Nothing regressed.** No cell winner moved outside ±0.5% of the node's own drift except
in the favourable direction. The only entry anywhere near a real regression is
`gen_layout` at L=32 (150.228 → 155.728, +3.7% raw, ≈ +2.3% net) — inside its own 1.7%
run spread plus drift, and it is a library layer scored by adoption, not a contender at
that cell. Not material.

### Where the round's real movement was

The round was scoped as "two new B=1 small-L cells; everyone else protect your cells",
and that is exactly the shape of the result. The gains are concentrated in entries that
were **not** cell winners:

| entry | cell | r12 → r13 | net of drift |
|---|---|---|---|
| **gen_race** | 32 | 108.994 → 55.189 | **−50%** |
| gen_race | 31 | 138.751 → 86.909 | **−38%** |
| gen_race | 40 | 236.639 → 162.172 | **−32%** |
| gen_race | 27 | 60.041 → 44.106 | **−28%** |
| gen_race | 20 | 17.662 → 12.846 | **−29%** |
| gen_race | 25 | 40.961 → 31.378 | **−25%** |
| gen_race | 50 | 545.065 → 423.261 | **−25%** |
| **gen_dense_prime** | 10:64 | 5.079 → 3.055 | **−42%** |
| gen_dense_prime | 12:64 | 7.673 → 4.812 | **−39%** |
| gen_dense_prime | 15 | 14.760 → 10.888 | **−28%** |
| gen_dense_prime | 20 | 38.428 → 35.075 | **−11%** |
| **gen_twiddle** | 12:64 | 7.902 → 5.471 | **−32%** |
| gen_twiddle | 15 | 17.823 → 14.716 | **−19%** |
| **gen_bluestein** | 10:64 | 12.646 → 8.826 | **−32%** |
| gen_bluestein | 12:64 | 18.751 → 13.654 | **−29%** |
| gen_bluestein | 20 | 80.712 → 65.389 | **−21%** |
| **gen_powp** | 50 | 480.616 → 420.549 | **−15%** |

Three of these are genuine arithmetic/structural wins (`gen_dense_prime`'s compile-time
instantiation, `gen_twiddle`'s new PFA leaf12/leaf15 levels, `gen_bluestein`'s sub-32
convolution grid). One — `gen_race` — is not; see §3.

`gen_powp`'s L=50 recovery deserves separate mention because it is the round's most
transferable record. Its r12 number (480.616) was **not** a code regression: its own
plan-time race banked a biased winner from one noisy scoring window, and its
authoritative-override rule then locked that choice in, shipping 480 where ~415 was
available — a self-inflicted 16% loss on a scored cell **with the kernel arithmetic
untouched**. r13 fixed it by trimming the candidate pool to families with a recorded win
at that size, not by changing a rule. Their own summary is the lesson: *"a decision rule
cannot out-measure its window; what CAN be fixed deterministically is who is allowed on
the ballot."*

### Setup cost (excluded from scored time, reported separately per standing rule)

Two large moves, in opposite directions, both at L=100: `gen_race` **32.847 s → 0.040 s**,
`gen_powp` **0.019 s → 3.943 s**. `gen_race`'s salt bump (see §3) invalidated every
banked wisdom entry this round, so all plan-time races ran cold; `gen_powp` pays 3.9 s of
plan time for a 4.1 ms transform at L=100. Neither affects a scored number. Both should
be watched: a plan cost of seconds is real for a user who plans once per call.

---

## 3. Adversarial review — failures, absences, and one result that must not stand

### 3.1 `gen_race` — its record for this round does not exist. **Excluded from promotion.**

This is the round's most serious finding, and it concerns the entry that improved the most.

**Evidence.** Eleven of twelve strategy records were appended this round
(1,760 lines total: batchlane 147, bluestein 190, dense_prime 185, layout 148,
pfa_large 126, pfa_small 129, planner 203, pow2 133, powp 177, rader 150, twiddle 172).
`strategies/gen_race.md` received **zero lines** — `git diff --stat` shows no
modification, and the file still ends at the r12 section headed *"What I would do next
(gen_r13 / campaign close)"*.

Its agent log (`agents/gen_race.log`) is 245 bytes and consists of a single line:

> *"I'm waiting on the node's compile queue to drain... Once quiet, I'll take the
> quotable measurements at 10:1 and 12:1, verify the L=12 gates, spot-check a batched
> cell, and then write the strategy record."*

The agent stopped there. It never took its measurements and never wrote its record.
Meanwhile it **shipped 62 changed lines of code** that produced the largest deltas on the
board, and `agents/exits.txt` records `gen_race exit=0`.

That `exit=0` is a check that cannot fail — every one of the twelve agents reports
`exit=0`, including this one. The round runner's success signal does not detect an agent
that halted mid-task. That is worth fixing before the next round.

**What it actually shipped** (reconstructed by me from
`diff impl_12/gen_race.c impl_13/gen_race.c`, since no record exists): two changes.
(1) the engine race's chain-length cap `es.m` raised **64 → 4096**, because the new
cells grade at m in the thousands and a cap of 64 charged once-per-chain setup costs
200–256x more heavily than the score does; (2) a salt bump `*12 → *13` on all nine
wisdom keys (`chain13/tile13/chaingate13/fm13/cf13/p413/alt13/wv13/eng13`), invalidating
every banked pick and forcing a cold re-race everywhere.

So `gen_race`'s −25% to −50% across seven cells is **the search finding the right arm,
not new arithmetic**. That is a legitimate and valuable result. It is also entirely
undocumented, and 13 lines of in-source comment are not the record CURATION.md requires.

**Consequence, and it is the rule's plain text:** *"Do not promote... entries whose
strategy record is missing — the record is what makes the code useful later."*
`gen_race` is excluded. It should be asked to write the round's record retroactively —
including the measurements it never took — before `impl_14` is seeded from `impl_13`.

### 3.2 `gen_race`'s cell "wins" are not wins. L=32 is credited to `gen_pow2`.

`gen_race` is a library layer that dlopens the other classes' engines, races them, and
ships the winner by vtable forwarding. Its top-of-table appearances are therefore
**the class engine's own number with a wrapper's name on it**:

| cell | gen_race | class engine | margin | engine's own run spread |
|---|---|---|---|---|
| 32 | 55.189 | gen_pow2 55.600 | **0.74%** | 0.8% |
| 25 | 31.378 | gen_powp 31.369 | −0.03% | 2.4% |
| 31 | 86.909 | gen_rader 86.176 | −0.85% | 0.8% |
| 40 | 162.172 | gen_pfa_large 161.068 | −0.69% | 1.7% |
| 12:64 | 1.947 | gen_pfa_small 1.952 | 0.26% | 0.8% |
| 20 | 12.846 | gen_batchlane 12.849 | 0.02% | 5.0% |
| 10:1 | 1.943 | gen_pfa_small 1.943 | 0.00% | 14.0% |

At L=32 `gen_race` nominally tops the table by 0.74% — **inside `gen_pow2`'s own 0.8%
run spread, while running `gen_pow2`'s code**. Recording that as "the fastest entry at
L=32 is gen_race" would be a measurement artifact of adoption presented as an
algorithmic result. §1 credits L=32 to **`gen_pow2`**, and every other `gen_race`
appearance above is a tie with the engine it is executing, not an independent result.

### 3.3 `gen_rader` — a real compiler diagnostic, in the map, unaddressed for three rounds

`build_errors.txt` is not empty. It contains one diagnostic, and it is the only compiler
output on the whole board:

```
impl/gen_rader.c: In function 'map_volume.constprop':
impl/gen_rader.c:2967:23: warning: iteration 1152921504606846976 invokes undefined
behavior [-Waggressive-loop-optimizations]
```

`1152921504606846976` is 2^60. The site (`impl_13/gen_rader.c:2966-2972`) is the scalar
tail of `map_volume` following an AVX-512 loop that runs while `i + 8 <= npts`, so the
tail can execute **at most 7 iterations**. In a `constprop` clone GCC has lost that
relationship and derived a theoretical trip count that would index far past the object.

**Assessment: a false positive in effect, but not one to leave sitting.** It is the
*map* — the one function every chain gate depends on, and the only non-transform
arithmetic in the timed path. `-Waggressive-loop-optimizations` means GCC believes it may
reason on an assumption that does not hold. It is not new: the identical warning appears
in `gen_r11` (line 2429) and `gen_r12` (line 2648), same function, same loop. Three
rounds, three appearances, no record mentions it.

The correctness evidence says it is not firing — `gen_rader` passes both gates at L=31
(`ch=4.0e-14` against a `1e-10` budget, `1s=2e-15` against the `1e-14` contract), and did
so in r11 and r12. But "it hasn't bitten yet" is not the standard. **One line — bounding
the tail with `npts & 7` or hoisting the vector-loop postcondition — removes it.**
`gen_rader` should do that next round.

### 3.4 `baseline_matrix` timed out at L=100 — expected, but it costs the control

`failures.txt` names three entries, all the same one:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout`. `baseline_matrix` is O(L⁴)/volume/axis; at L=100 it is ~200x the
work of the next-slowest entry and cannot finish in the harness window. Not a panel
failure and not a correctness event — but it is the reason the L=100 row of §2's drift
table had to fall back to MKL for its control. Consider giving that one cell a longer
timeout, or accept that L=100 has a weaker control than the other twelve.

### 3.5 Nothing failed correctness. Nothing failed to build. Nothing is missing.

I checked all three, adversarially.

**Correctness.** Every entry in every cell reports `ok` on both gates. The margins are
not marginal:

* **One-step gate** (the precision contract; chaos cannot touch one step): every
  backend on the board lands `1s = 7e-16 … 4e-15`, against a `1e-14` contract. The
  worst is `gen_bluestein` at L=100 (`4e-15`) — still 2.5x inside. Nothing is running an
  fp32 interior or a shortcut map; that gate is what catches those, and it is clean
  everywhere.
* **Chain-end gate**: worst ratio anywhere on the board is `gen_layout` at 10:1
  (`1.2e-05` against a `3e-03` budget) — **0.4% of budget**, better than two orders of
  margin. Typical is 0.05%.

One thing that looks alarming and is not: the new B=1 cells carry budgets of `3e-03`
(10:1, m=16384) and `1e-05` (12:1, m=12288), seven and five orders looser than the
batched cells' `1e-10`. That is correct by design — these are the longest chains on the
board and the budget is set from the measured divergence of the reference pair on that
very chain, not from a constant. The precision contract is carried by the one-step gate,
which at those cells reads `8e-16 … 1e-15` for every entry. A loose chain budget with a
tight one-step gate is the intended configuration, not a hole.

**Builds.** All twelve implementations compiled. The only diagnostic is §3.3's warning.

**Coverage.** No entry vanished from any cell it contested in r12. Per-cell entry counts
are identical r12 → r13 (8 at 10/12/15/20, 6 at 25/27/32/40, 7 at 31, 9 at 50/100), and
absences are class restrictions, all legitimate: `gen_pow2` is 2^k only, `gen_powp` p^k
only (its `supports()` is verified p^k in-record), `gen_rader` and `gen_dense_prime`
prime-only, `gen_pfa_small`/`gen_batchlane` need a coprime P×Q factorisation. No entry
declined a cell it can serve.

**One measurement-quality caveat, raised rather than buried.** The two new B=1 cells show
unusually wide run spread at the top: `gen_pfa_small` and `gen_race` both report **14.0%**
at 10:1 and **13.8%** at 12:1, while `gen_batchlane` at the same cells is at 0.1–0.6%. The
leaderboard reports the min across processes, so a 14%-wide distribution reported at its
floor deserves scepticism. Here it is corroborated: `gen_pfa_small`'s independent
same-core measurement, taken under a held slot lease in a quiet window, reads
**1.945–1.950** at 10:1 (board 1.943) and **2.996–3.004** at 12:1 (board 2.999). The
board number is real; the spread is process-to-process placement, not a lucky sample.
The same check applies to `gen_pfa_small`'s 1.147 at 10:64 (11.1% spread) — there,
`gen_batchlane` reaches the identical 1.147 at **0.1%** spread, so §1 names both and the
batchlane number is the one to trust.

---

## 4. Claimed versus measured

As established in §0, implementers measured on the scoring node this round, so the
machine-difference explanation the brief anticipated is largely not needed. Agreement is
good. Four items are worth recording.

**Confirmed, closely.**

| entry | claim | measured (net of drift) |
|---|---|---|
| `gen_pfa_small` | 1.945–1.950 at 10:1, 2.996–3.004 at 12:1 | 1.943 / 2.999 — **within 0.2%** |
| `gen_pfa_small` | MKL 4.330–4.339 / 7.319–7.367 in-window | board 4.331 / 7.319 — **exact** |
| `gen_dense_prime` | "−38..−42% at the B=1 and batched 10/12 cells" | −42% at 10:64, −39% at 12:64 — **holds** |
| `gen_twiddle` | leaf12/leaf15 delete a whole combine level | −32% at 12:64, −19% at 15 — **holds, large** |
| `gen_powp` | ~415 was available at L=50 | 420.549 shipped — **holds** |
| `gen_rader` | 31 unchanged, control 84.76–85.56 | board 86.176, net +0.1% — **holds** |
| `gen_pow2` | "scored chain paths are UNTOUCHED (verified)" | L=32 net 0.0% — **holds exactly** |

**Gap 1 — `gen_bluestein`'s sub-32 grid underdelivers at the top of its claimed range.**
The record claims the new grid covers `L=25..28` with `M 64→56`, −12.5% convolution
data. Measured net: **−3.4% at L=25 and −3.9% at L=27**, against **−32% at 10:64,
−29% at 12:64 and −21% at 20**. The claim is not false — it is a *data* reduction, not a
wall-clock one — but at 25/27 the wall payoff is under a third of the data saving, where
at 10/12/20 it exceeds it. The plausible cause is not the machine: at L=25/27 the working
set is 7.6–9.6 MiB and `gen_layout`'s own r10 PMU audit found these sizes traffic-bound,
so deleting convolution *arithmetic* does not shorten a chain whose cost is elsewhere.
At 10/12 the volume is L1/L2-resident and the pass deletion converts directly. This is
worth stating because it is a fair prediction for the next round: arithmetic-reduction
work will keep paying at 10–20 and keep disappointing at 25+.

**Gap 2 and 3 — two entries' entire rounds are invisible to the board, and both know it.**
Neither is a discrepancy in the usual sense; both are claims that are almost certainly
true and simply cannot be checked here.

* `gen_rader` shipped compile-time engines for **p ≤ 13** and reports −40..−57%
  (p=13: 8.393 → 4.2 µs at B=1, 1.44x over MKL; p=5 0.669 → 0.313, −53%). **No prime
  below 31 is scored.** Its own record says r13 is bit-identical to r12 at p ≥ 17 and 31,
  and the board confirms it (net +0.1%).
* `gen_pow2` shipped `GP2_XFE` execute-path conversion fusion and reports −19/−19/−24/−13%
  at single-call L=16/32/64/128 (L=32: 75.86 → 61.60 µs). **The board scores L=32 only as
  a B=8 chain**, where the record says the path is untouched — and the board confirms that
  too (net 0.0%).

Both claims are internally well-evidenced (same-core rotated pairs, 3/3 rounds, cmp-verified
bit-identity). I have no reason to doubt either. But two of twelve implementers spent a
full round on work this harness does not measure, and neither the −53% nor the −19%
appears anywhere in §1. That is a scoring-coverage problem, not an implementer problem;
it is the substance of §6's recommendation for those two cells.

**No entry's claim was contradicted by measurement anywhere on the board.**

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

Two moved with numbers attached, and a third moved sideways in an unexpected and
useful direction.

### §4.1 — "How many registers does a batch-vectorised codelet actually need?" **Moved, and answered in the direction the corpus feared.**

§4.1's framing is that `2L` (§04 §1.3, §07 §4.3) is a *data-only lower bound, not a
budget*; §01 §7.2 measured real peak liveness including temporaries at 17 for n=6 and 19
for n=8 and predicted the batch-major form "will spill a little"; and the open question
is whether the spill traffic costs more than the shuffles it avoids.

`gen_pfa_small` built the experiment without setting out to. Its pass-3/pass-1 fusion is
structurally exact — the mapped pass-3 output registers of a block *are* the next step's
pass-1 pencil — and it deletes a whole volume round trip per step. It **lost by +12% at
L=10 and +13% at L=12**, 3/3 rotated same-core rounds. The mechanism is §4.1's, measured:
the fused block holds **2L live site registers across L map ladders and a second DFT**,
violating their standing `2L + module temps ≤ ~32` rule exactly where it bites, and the
spill traffic exceeded the saved streaming loads and stores.

So the corpus's `2L` bound is confirmed as a floor that is *reached and exceeded* the
moment anything is fused across it, on AVX-512 with 32 registers — not merely on AVX2
with 16. And Frigo's UltraSPARC price tag for getting it wrong (50–100%, "entirely
register spills and reloads") is, on this hardware and at this size, 12–13%.

A second, sharper result came out of the same experiment and belongs in the record:
**the fused form is not bit-identical to the unfused one even though it runs the same
macro arithmetic in the same order** — GCC contracts shared products into FMA differently
per inlining context (`t=mul; add/sub t` versus `fma + fnma`). Divergence 3.3e-16 relative
at m=2, 1-ulp class, all gates re-passed with ≥30x margin. Their rule, which every entry
should adopt: *any claim of bit-identity across a refactor that changes inlining context
must be `cmp`-verified, not assumed* — form 3 happened to match, form 4 did not.

### §4.3 — "Is axis fusion worth 3× or 3%?" **Moved, and Tolmachev's rule gains a third term.**

§4.3 was settled in part by `panel_r3` for the **L1↔L2** boundary (single-digit percent),
and explicitly **re-opened** by §08 §1.9 for the **L2↔DRAM** boundary, where the
bandwidth gap is 7x and where "tile the batch so a tile fits L2, then run all three axes
inside the tile" is named *"the largest untried structural move on the board"*.

Two independent measurements this round bound the answer from both sides.

**From below — fusion that saves hits is worthless.** `gen_pfa_small`'s +12/+13% loss
above is the L1 case, and their conclusion is the crisp form of Tolmachev's rule:
*"a fusion that saves L1-HIT traffic but adds spills is a net loss; fusion pays when it
saves misses, not hits."* The saved pass-1 loads were L1-hot by construction — pass 1
read what pass 3 had just stored.

**From above — and this is the round's most transferable finding — fusion that saves
misses can still lose, to the page tables.** `gen_pow2` fused its custody conversions
into the execute path at L=128, deleting **~138 MB of the call's ~240 MB of DRAM
traffic**. Result: a **wash** (17.2–17.4 ms fused vs 17.1–18.2 ms control, mixed signs,
3 rotated rounds). Mechanism: the custody block has been huge-paged since r12, so the
unfused strided x-pass stores cost ~8 stores per 2 MB page — but the *fused* stores land
in the driver's natural buffer, which is 4 KB-paged, at 256 KB stride: **one page walk
per store line**. The TLB cost consumed the entire traffic win. The load-side-only form
ships instead, at −13%.

That is a direct, quantified counter-example to the pure traffic model §4.3 is built on.
Tolmachev's rule — payoff = avoided passes × bandwidth gap between the two levels — is
necessary and **not sufficient**. `gen_pow2`'s formulation deserves to go into §4.3
verbatim: *"a fusion that redirects strided STORES from your own (huge-paged) buffer to a
caller's 4-KB-paged buffer can lose to page walks everything it wins in traffic — check
the store target's page size before fusing, not just the walk order."*

Note what this does **not** settle: §4.3's re-opened L2↔DRAM question asks about tiling
the batch so a tile fits L2 and running all three axes inside it. Nobody built that this
round. It remains the largest untried structural move, now with a known hazard attached.

### §4.6 — "Model versus search for the instruction schedule." **Moved sideways: search is the primary lever *and* the primary risk.**

§06 concluded that the schedule is "the primary thing to search" at our sizes, and §4.6
frames the cost as cheap ("costs minutes"). Round 13 supplies both halves of an amendment.

The **upside**: `gen_race`'s entire board-wide −25% to −50% came from changing *one search
parameter* — the race's chain-length cap, 64 → 4096 — so that plan-time A/B ran at a
chain length representative of the graded one. No arithmetic changed. Search was worth
more this round than every kernel change combined.

The **downside**, from `gen_powp`: its own race banked a biased pick from a single noisy
scoring window and its override rule then made that pick authoritative, shipping a **16%
regression on a scored cell with the kernel code untouched**. Their record notes this is
the **third** observed instance of the same boundary (r8's l25-ip0, r11's dev-window
l25-ip0, now r12's scoring window itself).

So §4.6's "just search it, it costs minutes" needs the caveat that a search which cannot
out-measure its own window will confidently ship the wrong arm and then defend the
choice. `gen_powp`'s fix is the generalisable one and is not a better decision rule but a
better *ballot*: restrict the default pool to families with a recorded win at that size on
some host.

**Not moved:** §4.2 (L=17), §4.5 (L=8 padding), §4.7 (vector-radix) — none of those L
values is on this board. §4.8 item 6 continues to accumulate Ice Lake-SP server data,
which the corpus still lists as absent, but this round added no new AVX-512 finding beyond
prior rounds.

---

## 6. The single highest-value thing the next round should attack, per geometry

**L=10 and L=12, B=1 (the new cells) — attack the plain single-call `execute()`, not the
chain.** This is the round's own unfinished business and `gen_pfa_small` names it
directly: *"the driver's chain shape is kinder to us than benchFFT's plain-transform
convention — the libraries pay a separate map pass per step; plain `execute()` at B=1 is
still weak."* The 2.2x/2.4x in §1 is measured on a 16384-step chain; the community
harness times repeated single calls, where our per-call setup and custody conversions are
unamortised. `gen_pow2` has already built the fix for its own class and measured it
(`GP2_XFE`, −19..−24% at single-call 16/32/64) — **port conversion fusion to the 10/12
split chain's execute path.** Highest value because it closes the gap between what we
score and what the outside world would score.

**L=10/12 B=64, L=15, L=20 — stop optimising arithmetic; delete uops.** These four cells
are at 40–48 GF/s, three entries are tied inside 0.4%, and the net movement over two
rounds is zero. The arithmetic is done: the map is settled (`gen_pfa_small`'s rcp tail
lost +4.3/+5.6%, their *sixth* independent confirmation the divider is free here), the
DFT5 is φ-lifted to 6 ops, the leaves are twiddle-free. What is left is front-end and
port pressure — `gen_layout`'s PMU dashboard measured these sizes running **at the node's
~2.1 uops/cycle dispatch cap with loads the largest port class**, and its r13 pencil-lane
kit (`gl_tr8x8_ld` insert-load transpose, `gl_pack8_ld` at −33% port-5 entry pack,
`gl_map8s`/`gl_map4s` split-form maps) is built, bit-identical and **unadopted at these
batched cells**. Adopt it and measure the port distribution, not the flops.

**L=25 and L=27 — get a second algorithm on the board.** `gen_powp` leads both by 1.30x
over the next entry, and that next entry is `gen_race` running `gen_powp`. There is **no
structurally different implementation of 5² or 3³ within 30%** — which means CURATION
criterion 2 cannot be satisfied here, and more practically that `gen_powp`'s plan-time
race has nothing to cross-check against. Given §2's account of how a race can ship 16%
off its own best, that is the exposure to close. `gen_pfa_small` already has a module-25
CT from r11; race it at B=16.

**L=31 — score the tiny primes, or stop working on them.** `gen_rader` holds the board's
widest margin (9.86x over MKL) and spent the entire round on p ≤ 13, which the board does
not score. Its probe found a genuinely sharp class boundary — everything from p=17 up
beats the libraries 3.4–6.4x, everything at p ≤ 13 *lost* to MKL by 1.4–2.0x — and it
fixed the losing side (p=13: 8.393 → 4.2 µs, now 1.44x over MKL). **Add p=13 as a scored
cell.** It is the size where MKL has a tuned small radix and we were behind, it is now
the class's most interesting result, and at present none of that work can be seen. If the
board will not score it, then L=31 itself has been flat for two rounds and should be the
target instead.

**L=32 — the B=8 chain has not been attacked in two rounds.** Same diagnosis as L=31:
`gen_pow2`'s round went entirely to single-call B=1 at 16/32/64/128 and its scored cell
moved by exactly the node drift (net 0.0%). Meanwhile the entry's own biggest structural
win — the r11/r12 one-sweep fused step — was flipped on at 64 (−11%) and 128 (−14%) and
**L=32 still runs the plain two-sweep step**. Try the fused sweep at 32, with §5's page-size
warning applied to the store target before, not after.

**L=40, L=50, L=100 — build the L2↔DRAM tile.** These are the traffic-bound sizes
(`gen_layout` r10 measured 50/100 traffic-bound; `gen_layout` r11 found the driver's
32 MB buffers are 4 KB-backed on this kernel, `THP=madvise` notwithstanding, and 5.15 has
no `MADV_COLLAPSE`). They are also the cells where §4.3's re-opened question lives, and
where the corpus, Intel's manual, Alappat et al. and the L3-Fusion result independently
recommend the same untried construction: **tile the batch so a tile fits L2, then run all
three axes inside the tile.** L=100 is the weakest cell on the board (1.92x over MKL,
against 2.2–4.1x everywhere else) and has been flat two rounds running. This is the
largest single structural move available anywhere on the board — and `gen_pow2`'s TLB
inversion is the map of the minefield to cross first.

---

## 7. What to keep

Applying `docs/CURATION.md` in its stated order.

**1. Fastest correct entry per geometry — always.** Six distinct entries win the
thirteen cells: `gen_pfa_small` (10:1, 10:64, 12:1, 12:64), `gen_batchlane` (15, 100, and
tied at 10:64 and 20), `gen_powp` (25, 27), `gen_rader` (31), `gen_pow2` (32),
`gen_pfa_large` (40, 50). All six are promoted on this criterion alone.

**2. A structurally different runner-up when it is close.** `gen_dense_prime` at L=31 is
1.30x off `gen_rader` — outside the rule's "~20%" example, but it is a completely
different algorithm (folded dense O(p²) register-tiled GEMM versus Rader cyclic
convolution), it is the *only* alternative to Rader on the board at any prime, and it
qualifies independently under criterion 4 below. At 10/12/15/20 the top three are within
0.4% of each other and two of them — `gen_pfa_small` (interleaved-site SoA) and
`gen_batchlane` (split batch-lane SoA) — are genuinely different structures; both are
already promoted, so criterion 2 is satisfied there without additions.

**3. Instructive failures whose record documents the number that killed them.** Three
this round, all inside entries already being promoted, and each is a reason to keep that
entry's record rather than just its code:
* `gen_pfa_small` — pass-3/pass-1 fusion, structurally exact, **+12%/+13%**, killed by
  2L live site registers spilling across the map ladders (§5, §4.1). Plus the
  bit-identity-across-inlining discovery.
* `gen_pow2` — full execute-path fusion at L=128, deleting ~138 MB of ~240 MB DRAM
  traffic, a **wash**, killed by one page walk per store line into the caller's
  4 KB-paged buffer (§5, §4.3).
* `gen_powp` — its own race shipping **480.6 where 415 was available**, a 16% loss on a
  scored cell with the kernel untouched, and the third instance of the same boundary
  (§2, §4.6).

**4. Anything that beat a library baseline, regardless of rank.** `gen_dense_prime`
qualifies emphatically: 1.88x over the best library at L=31 and 1.4–1.5x over MKL at
10/12/15/20. It is also the round's second-largest genuine win (−42%/−39% at 10:64/12:64
from compile-time instantiation of the small composites) — and that result **converges
with `gen_rader`'s independent finding at p ≤ 13**, that runtime-table engines are
all fixed cost at small sizes and literal-shape instantiation removes it. Two entries
arriving at the same structural conclusion from different classes is the round's main
transferable finding, and both halves should be on the reading list.

**Deliberately not promoted, with reasons:**

* **`gen_race` — excluded on the explicit rule, despite improving the most.** Its
  strategy record for this round does not exist (§3.1): zero lines appended, agent halted
  mid-task with its last words being *"and then write the strategy record"*. CURATION is
  unambiguous — *"entries whose strategy record is missing"* are not promoted, because the
  record is what makes the code useful later. Compounding it, its apparent cell wins are
  0.02–0.74% margins over the class engines it is literally executing (§3.2), so there is
  no independent result to preserve either. **Action for the round runner: have
  `gen_race` write its r13 record, including the measurements it never took, before
  `impl_14` is seeded.** Its actual change is small and valuable (chain-cap 64 → 4096,
  salt bump) and I have reconstructed it in §3.1 so the next round is not blind.
* `gen_twiddle`, `gen_layout`, `gen_planner` — library layers scored by adoption. Real
  work this round (`gen_twiddle`'s leaf12/leaf15 is worth −32% at 12:64), but they are
  mid-table everywhere, win nothing, and their value is already realised *inside* the
  promoted entries. Promoting them would be promoting near-duplicates of code already
  kept.
* `gen_bluestein` — the any-L fallback, last among panel entries at every cell (3.2–8.4x
  off the winners). Its r13 sub-32 grid is a genuine −32%/−29%/−21% at 10/12/20, but it
  beats a stock library at exactly one cell (L=31, over MKL) and satisfies no other
  criterion. Its record is intact and stays in `strategies/` where the next round will
  read it.

**Promoting seven** — six cell winners plus one structural alternative that also clears
the library criterion.

---

## 8. One process note for the round runner

`agents/exits.txt` reports `exit=0` for all twelve agents, including the one that stopped
mid-sentence waiting on a compile queue and shipped code without a record. A success
signal that reports success for an agent that halted before finishing is the
"check that could not fail" shape this project's verification discipline warns about.
Cheapest two-sided fix: after each round, assert that every agent whose
`impl_N/<name>.c` differs from `impl_{N-1}/<name>.c` also appended lines to
`strategies/<name>.md`. That single check would have caught this round's one real
process failure automatically.
