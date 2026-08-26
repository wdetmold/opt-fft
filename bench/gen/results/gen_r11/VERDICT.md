# VERDICT — round gen_r11

Monitor's judgement on the measured board in `results/gen_r11/`.
Scored on **a80n0.lqcd.mit**, slurm job 438881, 2026-08-26T12:13:40-04:00.

---

## 0. Three corrections to the monitor brief, up front

The brief I was given is a stale template. It is wrong on three points of fact, and
since every number below depends on them, they go first.

**0.1 — The geometries are not L = 6, 8, 17, 36.** Those were `panel_r1`'s cells (they
survive in `docs/CURATION.md`'s worked promote example: `L17_rader L8_batchsimd L36_pfa`).
The `gen_*` campaign scores **eleven** cells: **L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50,
100**. There is no L=6, L=8, L=17 or L=36 data in this round, and no entry supports being
scored at them. I report the eleven cells that were actually measured.

**0.2 — The scoring host is not a Cascade Lake Xeon Gold 5218.** `environment.txt` reads
**Xeon Gold 6326 @ 2.90 GHz** — Ice Lake-SP, 1.25 MB L2/core — and the ISA line proves it
(`avx512_vbmi`, `avx512_vnni`, `avx512ifma`, `avx512_bitalg`, `avx512_vpopcntdq` are Ice
Lake features Cascade Lake does not have). Every one of the eleven `gen_r*` rounds was
scored on a 6326. A Gold 5218 *does* exist in this project — it is `p52n1`, the Cascade Lake
**cross-arch advisory** host behind `results/xarch_clx_r4/` and `xarch_clx_r6/` — but it is
not the scorer. The "downclocked AVX-512" framing also does not apply: `gen_batchlane`
measured license level 2 for ~97% of cycles at L=100 with no frequency penalty visible in
the throughput, and `gen_dense_prime` sustained 3.0 vector uops/cycle in a calibration
microbenchmark on this node.

**0.3 — Implementers did not develop on Sapphire Rapids this round; they developed on the
scoring node.** Every r11 strategy record quotes numbers taken on **a80n0 itself**, under
held slot leases with same-core interleaved pairs. `wallaby` (Xeon Gold 6448Y, Sapphire
Rapids) was used only for pre-reads and knob A/Bs, and is always labelled as such. The
consequence for §4 below is large: there is essentially **no claimed-vs-measured gap left to
attribute to machine difference**, and I decline to invoke one where the data does not
support it. For the record, the MKL span across the three hosts on comparable boards is
**~1.5x–2.1x** (L=31: CLX 1224.1 µs, ICL 848.8, SPR 575.8 = 2.13x end to end), not the 2.9x
the brief asserts.

One further methodological caveat that shapes everything: **r10 was scored on a81n2, r11 on
a80n0.** Both are 6326s but they are not the same machine, and two entries measured that
directly. `gen_dense_prime` shipped a binary whose `.text` is byte-identical to r10
(`objcopy` cmp) and its board numbers still moved **−3.3% to +3.3%** across five cells;
their own node read of the L=31 cell was 113.9 on a80n0 against 109.9 on a81n2. So the
cross-round noise floor here is ~±3%, and I treat sub-3% r10→r11 moves as weather unless an
entry's own record contradicts that.

---

## 1. Headline per geometry

### 1.1 The board does not resolve a single winner at any cell

Before the table: at **every one of the eleven cells the top two entries are separated by
less than their own reported run spread.** Margins and spreads:

| L | co-leaders | margin | leaders' run spread |
|---|---|---|---|
| 10 | gen_pfa_small / gen_batchlane | 0.09% | 0.1% / 0.1% |
| 12 | gen_batchlane / gen_race / gen_pfa_small | 0.02% / 0.05% | 0.3% / 14.2% / 0.2% |
| 15 | gen_batchlane / gen_race / gen_pfa_small | 0.14% / 0.18% | 1.8% / 14.3% / 0.4% |
| 20 | gen_batchlane / gen_race | 0.77% | 5.8% / 4.8% |
| 25 | gen_powp / gen_race | 1.00% | 2.2% / 1.8% |
| 27 | gen_powp / gen_race | 0.14% | 5.6% / 1.7% |
| 31 | gen_race / gen_rader | 0.27% | 0.8% / 1.0% |
| 32 | gen_pow2 / gen_race | 0.43% | 1.8% / 3.4% |
| 40 | gen_race / gen_pfa_large | 0.31% | 3.8% / 3.5% |
| 50 | gen_race / gen_powp / gen_pfa_large | 0.71% / 1.70% | 5.4% / 5.5% / 4.9% |
| 100 | gen_race / gen_batchlane | 0.02% | 4.6% / 4.7% |

I therefore report **co-leaders**, and the ordering inside a tie should not be read as a
result. This is not a defect of the round — it is what convergence looks like — but a
"winner" whose lead is 1/20th of its own variance is not a winner.

The `gen_race` ties are structural, not coincidental: `gen_race` is a plan-time racing layer
that compiles peer entries as `.so` and forwards to the winner by vtable. **I audited this
against the correctness digits, which the racer cannot fake**, and the routing is confirmed
at every cell: at L=100 `gen_race` and `gen_batchlane` both report `ch=2.8e-14 1s=3e-15`; at
31 `gen_race` and `gen_rader` both `3.4e-14/2e-15`; at 40 both `2.1e-14`; at 32 both
`2.9e-14`; at 25/27 `gen_race` matches `gen_powp` exactly (`4.8e-14`, `5.5e-14`); at 50
`gen_race` matches `gen_powp`/`gen_pfa_large` at `3.3e-14` and **not** `gen_batchlane`'s
`3.1e-14` — i.e. it correctly declined batchlane's new engine at the one large cell where
batchlane lost. The racer is honest.

### 1.2 Fastest correct panel entry vs best library

Best library = minimum over all `sota/` backends (FFTW ×5 planner/guru/custom variants,
MKL 2022, MKL 2026, ducc0). MKL 2022 given separately as the canonical reference.

| L | mode | fastest correct panel (µs/transform) | co-leader within spread | best library | speedup vs best lib | vs MKL 2022 |
|---|---|---|---|---|---|---|
| 10 | batched B=64 | **gen_pfa_small 1.121** | gen_batchlane 1.122, gen_race 1.124 | fftw3_custom_soa 4.451 | **3.97×** | 4.06× (4.553) |
| 12 | batched B=64 | **gen_batchlane 1.915** | gen_race 1.915, gen_pfa_small 1.916 | mkl_dfti 7.740 | **4.04×** | 4.04× |
| 15 | batched B=32 | **gen_batchlane 4.340** | gen_race 4.346, gen_pfa_small 4.348 | fftw3_custom_soa 15.414 | **3.55×** | 3.80× (16.490) |
| 20 | batched B=32 | **gen_batchlane 12.597** | gen_race 12.694 | fftw3_custom_soa 41.407 | **3.29×** | 4.59× (57.878) |
| 25 | batched B=16 | **gen_powp 30.857** | gen_race 31.164 | fftw3_custom_soa 75.145 | **2.44×** | 3.91× (120.773) |
| 27 | batched B=16 | **gen_powp 43.484** | gen_race 43.545 | fftw3_custom_soa 96.257 | **2.21×** | 3.32× (144.451) |
| 31 | batched B=16 | **gen_race 84.549** | gen_rader 84.776 | fftw3_custom_soa 213.361 | **2.52×** | **10.04×** (848.750) |
| 32 | batched B=8 | **gen_pow2 53.809** | gen_race 54.041 | mkl_dfti 170.663 | **3.17×** | 3.17× |
| 40 | batched B=8 | **gen_race 159.534** | gen_pfa_large 160.033 | mkl_dfti 404.658 | **2.54×** | 2.54× |
| 50 | batched B=4 | **gen_race 410.975** | gen_powp 413.898, gen_pfa_large 417.960 | mkl_dfti 946.334 | **2.30×** | 2.30× |
| 100 | **non-batched B=1** | **gen_race 4071.340** | gen_batchlane 4072.315 | mkl_dfti 7812.381 | **1.92×** | 1.92× |

Every cell is won by a panel entry, over every library, by ≥1.9×. The board's weakest cell
is still L=100 and its strongest margin is still L=31 (10× MKL, because MKL falls off a
cliff at prime 31 while our Rader engine does not).

### 1.3 On "both the non-batched and the batched cases"

**The harness gave exactly one mode per geometry**, so this cannot be answered from the
board: L=10…50 are scored batched (B = 64/64/32/32/16/16/16/8/8/4) and **L=100 is the only
non-batched cell (B=1)**. There is no batched/non-batched pair at any single L in
`leaderboard.txt`, and I will not manufacture one.

What does exist is off-board B=1 reads the implementers took on this node, which I record
here clearly labelled as **implementer-measured, not harness-scored**:

| L | scored (batched) | implementer B=1 read on a80n0 | source |
|---|---|---|---|
| 10 | 4.251 (gen_twiddle, B=64) | 4.265 | gen_twiddle r11 |
| 31 | 84.776 (gen_rader, B=16) | 96.1–96.7 (fresh-core ramp signature) | gen_rader r11 |
| 31 | 111.117 (gen_dense_prime, B=16) | 111.41 | gen_dense_prime r11 |
| 32 | 53.809 (gen_pow2, B=8) | 53.835 | gen_pow2 r11 |
| 40 | 160.033 (gen_pfa_large, B=8) | 185.1 | gen_pfa_large r11 |
| 50 | 417.960 (gen_pfa_large, B=4) | 419.1 | gen_pfa_large r11 |
| 100 | 4072.315 (gen_batchlane, B=1) | 4606.8 at B=2 | gen_powp r11 |

Reading: batching is worth ~13% at L=40, ~0 at 32/50, and ~11% at 31 for the Rader engine.
**If the campaign wants a batched/non-batched contrast it has to be added to the sweep**;
this is a harness gap, not an implementer gap, and it is the cheapest missing measurement on
the board.

---

## 2. What changed since gen_r10, per geometry

All-hands round: the brief put every implementer on L=100. That is exactly where the board
moved, and the small cells paid a small tax for it.

| L | r10 leader | r11 leader | Δ leader | verdict |
|---|---|---|---|---|
| 10 | gen_pfa_small 1.115 | gen_pfa_small 1.121 | +0.5% | flat (inside spread) |
| 12 | gen_race 1.914 | gen_batchlane 1.915 | +0.1% | flat |
| 15 | gen_pfa_small 4.325 | gen_batchlane 4.340 | +0.3% | flat |
| 20 | gen_batchlane 12.571 | gen_batchlane 12.597 | +0.2% | flat leader, **runner-up regressed** |
| 25 | gen_powp 30.863 | gen_powp 30.857 | −0.0% | flat |
| 27 | gen_race 42.837 | gen_powp 43.484 | +1.5% | flat (r10's race lead was a warm-hit artifact its own record flagged) |
| 31 | gen_race 84.519 | gen_race 84.549 | +0.0% | flat |
| 32 | gen_pow2 54.478 | gen_pow2 53.809 | **−1.2%** | small real gain |
| 40 | gen_pfa_large 159.584 | gen_race 159.534 | −0.0% | flat |
| 50 | gen_powp 417.201 | gen_race 410.975 | **−1.5%** | small real gain |
| 100 | gen_pfa_large 4529.429 | gen_race 4071.340 | **−10.1%** | **the round's result** |

**L=100 is where the round happened**, and it moved on five independent engines at once:

| entry | r10 | r11 | Δ | mechanism (own record) |
|---|---|---|---|---|
| gen_race | 4655.9 | **4071.3** | **−12.6%** | routes gen_batchlane's new within-volume engine |
| gen_batchlane | — | **4072.3** | NEW | within-volume SoA: 8 x-planes of one volume per zmm; shuffle-free fused z+y slab sweeps; PFA 4×25 with DFT25 = 5×5 CT (the file's first twiddles) |
| gen_powp | 4659.8 | 4465.2 | −4.2% | THP re-home of chain-hot streams, adopted from gen_layout; dtlb walks −98% |
| gen_planner | 4987.5 | 4570.5 | −8.4% | alternating (x,y,z)↔(x,z,y) layout → **one** in-place transpose per plane instead of two; dtlb walks −88% |
| gen_pfa_small | — | **6066.2** | NEW | slab-fused split step + in-place pass 1 + single L2-resident scratch slab; DRAM traffic 188→28 MB/step |
| gen_twiddle | 7247.2 | 6451.4 | −11.0% | radix-10 PFA 2⊗5 codelet levels; one whole level pass deleted, tw muls 155→90 |
| gen_bluestein | 14132.1 | 13536.8 | −4.2% | conv_mid14 hand-respilled: 566→479 insns/block, rsp accesses 102→1 |
| gen_layout | 9326.7 | 9255.8 | −0.8% | zero-copy chain-state re-home into a THP arena |
| gen_pfa_large | 4529.4 | 4554.7 | **+0.6%** | shipped `ipa1` prefetch-ahead — **did not show on the board** (§3.3) |

The cell went from **1.73× MKL to 1.92× MKL** and from one engine under 4.6 ms to five. It
is still the board's weakest cell.

Elsewhere: **L=50 gained two new entrants** (gen_batchlane 475.5, gen_pfa_small 726.6) —
both honest losses their own records own; and **gen_twiddle improved −18.7% at L=10 and
−11.5% at L=50** off its radix-10 factorizer, which is the largest non-L=100 gain of the
round and moved it to beating MKL at 9 of 11 cells.

### Did anything regress? Yes — two entries, both from code layout, on bit-identical output.

**2.1 `gen_pfa_small` at L=20: 12.609 → 13.275 µs, +5.3%.** This is real and it is
undetected. Their record asserts "ALL pre-r11 paths … ship **BIT-IDENTICAL** to r10" and
lists `cmp`-verified chain files — but the list covers **20 B=1**, and the graded cell is
**B=32**. Bit-identical output is not identical performance: the round added 449 lines
(`gstep_slab`, `gdft25`) to the same translation unit, which displaces the small-L hot text.
Their own 0.1% run spread at this cell rules out weather. **Cost: the L=20 runner-up slot.**
They were 0.3% behind the leader in r10; they are 5.4% behind now.

**2.2 `gen_planner` at L=20: 17.226 → 18.134 µs, +5.3%** (undocumented) **and at L=32:
105.877 → 108.888, +2.8%** (documented). The L=32 tax they found, chased through three
mitigation attempts, and accepted with a full write-up — `nm -S` showed gcc-11 outlining
`pln_map_span` once it gained a second caller; restoring the function verbatim recovered
most of it and they recorded the residual ±2–3% address sensitivity honestly. **L=20 is not
in their A/B list** (they checked 50/40/27/25/10) and carries the same signature. Same
mechanism, same round, one caught and one missed.

**2.3 Not regressions:** `gen_dense_prime` L=15 +3.3% and L=31 +1.1% on a `.text`-identical
binary — this is the a81n2→a80n0 host change, and this entry is the calibration for it.
`gen_layout` L=31 +2.4%, `gen_race` L=27 +1.7%, `gen_twiddle` L=12 +1.8% (this last one
documented by them at +0.3…+2.6%, 7/7 rotated pairs) — all inside the ±3% cross-host floor
or explicitly owned. Library-side moves (`fftw3_custom_soa` +8.8% at L=12, +6.9% at L=32;
`fftw3_guru` +4.0% at L=32) are baseline weather across hosts and do not implicate the panel.

---

## 3. Correctness, build, crash, and missing-entry audit

Adversarial pass. **No fast wrong answer survives this round — because there is no wrong
answer to catch.** The evidence:

**3.1 Correctness: 187/187 timed rows pass.** Every row in `leaderboard.txt` reads `ok`,
and independently all 187 `c_*.json` files carry `"ok": true` and `"chain_ok": true` with
no exceptions. Single-call `rel_l2` spans 7e-16…4e-15 against tol 1e-12 (≥250× margin);
graded-chain `rel_l2` spans 2.0e-14…2.5e-13 against tol 1e-10 (≥400× margin), and each is
within ~2× of its honest anchor. The two entries that shipped **new twiddled stages** this
round — `gen_batchlane`'s and `gen_pfa_small`'s DFT25 = 5×5 Cooley-Tukey, the first twiddles
either file has ever contained — cost nothing measurable: batchlane reads `ch=2.8e-14` at
L=100 against a 2.4e-14 anchor, and `gen_twiddle` actually **improved** L=100 single accuracy
to 4.612e-16 from 4.817e-16 by deleting a twiddled level.

**3.2 Builds: zero failures. One warning, pre-existing.** `build_errors.txt` contains a
single `-Waggressive-loop-optimizations` warning in `gen_rader.c:2429` — the scalar tail of
the fused map ladder, `for (; i < npts; ++i)`, where gcc reports "iteration
1152921504606846976 invokes undefined behavior". It is **not** an error, the object links and
passes every gate, and the identical warning is in `gen_r10/build_errors.txt` at line 2402 —
so this is ten-round-old noise, not an r11 event. Mechanically it is benign: the AVX-512 body
above it (lines 2409–2426) leaves at most 7 iterations, so the 2^60 case gcc is reasoning
about is unreachable. But it is a UB-derived warning in a hot path of the **L=31
co-leader**, and gcc is entitled to act on the inference. **This is a one-line fix (bound
`npts` explicitly, or make the induction variable `size_t`) and it should be taken in r12**
rather than carried an eleventh round.

**3.3 Crashes/hangs: one, and it is the harness floor, not a panel entry.**
`failures.txt` names only `baseline_matrix L=100 B=1` timing out (`exit 124`) on all three
runs. `baseline_matrix` is the library-free O(L⁴)/axis reference; at L=100 with chain m=64
that is ~3×10⁸ complex ops per volume per axis per step, and it cannot finish inside the
per-case timeout. **This is expected and pre-existing** — `gen_r10/failures.txt` is
byte-for-byte the same three lines. It costs the round nothing: the reference is validated at
the other ten cells, all `ok`. No panel entry crashed, hung, or exited nonzero; `agents/exits.txt`
records `exit=0` for all twelve.

**3.4 Missing entries: none silent.** I built the full presence matrix against r10 and **no
entry disappeared from any cell it occupied in r10**. Absences are all class scope, and the
three that matter at the all-hands cell were each **costed on paper and struck with numbers**
rather than quietly skipped:

- `gen_dense_prime` at L=100 — struck at the whiteboard: folded-dense 100-point axes are
  ~37.5M zmm FMA/volume = **6.5 ms at perfect port saturation vs a 4.5 ms incumbent**. Same
  arithmetic kills L=40 (166 µs floor vs 160 board) and L=50 (404 floor vs 410 board).
- `gen_rader` at L=100 — struck: the only Rader-shaped angle is Rader-25 on Z₂₅\* inside a
  PFA(4×25), and a 25-point via two twiddled 5-stages already costs ~404 vector FP with far
  better glue ratio; the cell is traffic-bound, not arithmetic-bound.
- `gen_pow2` at L=100 — out of class (not 2^k); it contributed the L=128 transfer note instead.

Two entries also **declined L=40 by arithmetic and said so**: `gen_batchlane` (B=8 group
working set 16.4 MB models to ~200 µs vs 159.6 measured) and `gen_pfa_small` (the SoA-8
8-volume arena is the bandwidth-bound regime their r6 measured at 0.95–1.1× MKL). Both
declines are correct: `gen_pfa_large` holds the cell at 160.0.

**3.5 The one real accountability finding: `gen_race` shipped 276 lines of code with no r11
strategy record.** `impl_11/gen_race.c` is 2639 lines against r10's 2363, and its backend
description in `leaderboard.txt` documents substantial new work (the ALT one-transpose chain
own-gated and raced per host; cross-class engine routing that puts `gen_batchlane`'s
within-volume engine at 50/100; new salts `chain11/tile11/chaingate11/fm11/cf11/p411/alt11/eng11`).
But `strategies/gen_race.md` gained **zero lines** this round — `git diff --stat` shows all
eleven other records gaining 147–215 lines each and gen_race gaining none — and its final
section is still r10's forward-looking "What I would do next (gen_r11 / campaign close)".
`agents/gen_race.log` is a single newline, while every other agent log carries a summary.
The agent exited 0 and its code demonstrably works — it is the fastest correct entry at four
of eleven cells — so this is a **documentation failure, not a correctness or performance
failure**. It matters anyway: `docs/CURATION.md` is explicit that "the record is what makes
the code useful later," and `gen_race` is precisely the entry whose value is its method
rather than its kernels. **I promote it (rule 1 makes it mandatory at four cells) and record
the missing r11 record as a blocking item on the promotion commit.**

**3.6 A harness item three entries reported independently.** `tryout.sh`'s chain map-check
leg passes the literal string `'$W/c.bin'` to `check.py` and has since r1 — `gen_batchlane`,
`gen_pfa_small`, `gen_pfa_large`, `gen_powp`, `gen_twiddle` and `gen_rader` all ran their map
gates by hand this round and all six flagged it. `gen_dense_prime` additionally found the
reservation heartbeat writer dead while the slurm job lived, which makes `reserve.sh --status`
and therefore `tryout.sh` refuse to run for the whole panel. And `--chain 1` segfaults in the
driver in every binary including r10's (`gen_planner`) / is a silent no-op (`gen_rader`).
None of these affected the scored run — every graded m ≥ 64 — but they cost the panel
several sessions and they are all cheap. **Nine to eleven rounds is too long to carry a
quoting bug.**

---

## 4. Claimed vs measured

**This is the section the brief expected to be full and it is nearly empty, for a good
reason: implementers measured on the scoring host.** Every r11 record quotes a80n0 numbers
taken under held slot leases with same-core interleaved pairs — the same machine, often the
same cores, as the scoring pass. Agreement is correspondingly tight.

| entry | cell | claimed (own record) | measured board | gap |
|---|---|---|---|---|
| gen_batchlane | 100 | 4059.2 (quiet, sd 0.1%) | 4072.3 | **+0.3%** |
| gen_batchlane | 50 | 475.2–475.8 | 475.5 | **in range** |
| gen_pow2 | 32 | 53.686 B=8 (sd 0.03%) | 53.809 | **+0.2%** |
| gen_rader | 31 | 85.448 (sd 0.08%) | 84.776 | −0.8% |
| gen_twiddle | 10 | 4.283 | 4.251 | −0.7% |
| gen_twiddle | 12 | 7.70–8.09 | 7.771 | in range |
| gen_pfa_small | 15 | 4.354 | 4.348 | −0.1% |
| gen_pfa_small | 50 | 721–735 | 726.6 | in range |
| gen_pfa_small | 100 | 6009 quiet best | 6066.2 | +0.9% |
| gen_bluestein | 100 | 13586.9 best / 13691.5 | 13536.8 | −0.4% |
| gen_layout | 100 | 9190.9 min-of-mins | 9255.8 | +0.7% |
| gen_powp | 100 | 4417.4–4439.5 best, 4590.3 ship arm | 4465.2 | in range |
| gen_planner | 100 | 4478.2–4499.1 quiet | 4570.5 | +1.6…+2.1% |
| gen_dense_prime | 31 | 113.93 (a80n0 core 2) | 111.1 | −2.5% |
| gen_twiddle | 100 | 6626.4 ship / 6306.9 quiet PMU | 6451.4 | between the two |

Twelve of fifteen land inside ±1%. **There is no case in this round that requires a
machine-difference explanation, and I decline to supply one.** The two classes of genuine
divergence are both same-machine effects:

**4.1 Busy-window dev reads, self-flagged.** `gen_powp` quotes 40.97 / 51.42 / 436.57 at
25/27/50 against board 30.86 / 43.48 / 413.90 — **+33% / +18% / +5.5%**. Their record labels
these verbatim as "busy dev windows" on code paths untouched below their re-home gate, and
they are right: this is twelve implementers compiling on one node, not architecture.
`gen_pfa_large` quotes L=100 `ipa1` at 4730.3 (race floor) / 4754.7 (pair min-of-mins) and
the board reads **4554.7 — 3.7–4.2% faster than claimed**, because their session windows
were hot ("late session unusable", "ctrl 4661–5147 in one lease"). `gen_race`'s r10 record
made the same point with an extreme case: 12–27 concurrent gcc jobs had MKL reading 2666.9
at L=31 against its 848.8 board. **The panel's own protocol — held-lease same-core
interleaved pairs, min-of-mins, sd-gated — is what kept the rankings right while the
absolutes drifted, and it worked.**

**4.2 One claimed win that did not survive scoring: `gen_pfa_large`'s `ipa1`.** They report
`ipa1` beating `ipp1` in **5/5 cold interleaved races** (margins 0.8–1.8%) and 4/6 held-lease
pairs, and shipped it as rank 0 at L=100 only. The board reads `gen_pfa_large` at 4554.7 vs
its r10 4529.4 — **+0.6%, i.e. the claimed ~1% gain is not visible.** To their considerable
credit their record predicted exactly this: "a consistent sub-6% quiet-floor winner cannot
displace rank 0 through the r9 noise gate **BY DESIGN** — the rank must carry it." A 1%
effect is below what this board can resolve; the entry fell from #1 to #4 at L=100 not
because it got slower but because four other engines got much faster. **Recorded as
unresolved, not as a false claim.**

**4.3 What the cross-host span actually is, for the record.** Since the brief asserted 2.9×,
here is the measured MKL 2022 span across the project's three hosts:

| L | CLX (Gold 5218, p52n1) | ICL (Gold 6326, a80n0 — scorer) | SPR (Gold 6448Y, wallaby) | CLX/SPR |
|---|---|---|---|---|
| 31 | 1224.066 | 848.750 | 575.757 | **2.13×** |
| 32 | 209.486 | 170.663 | 138.325 | 1.51× |
| 40 | 523.347 | 404.658 | 300.668 | 1.74× |
| 50 | 1208.676 | 946.334 | 663.213 | 1.82× |

The scorer sits between the two, 1.44× faster than Cascade Lake and 1.47× slower than
Sapphire Rapids at L=31. **1.5–2.1×, not 2.9×** — and irrelevant to this round's claims
anyway, since nobody scored their r11 numbers off-host.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

The round's own declared arbitration item — "is the engine uop-saturated at L=100?" — is not
a §4 question; it is a local dispute from `gen_pfa_large`'s r7/r8 accounting. It was settled
comprehensively (all nine engines that measured it report 1.04–2.03 all-port uops/cycle, and
the answer is *per-engine*: traffic-bound at 1.0–1.6, cap-bound at 2.0). But three genuine
§4 items moved, one of them decisively.

### 5.1 §4.3 — "Is axis fusion worth 3× or 3%?" — the re-opened L2↔DRAM case is now measured. This is the round's largest literature result.

§4.3's inset is unambiguous about what was left open: the r3 verdict (single-digit percent,
sometimes negative) covered only **L1↔L2** fusion at a 2.6× bandwidth gap, and the untested
case was **L2↔DRAM** at a 7× gap, where "Intel's own manual, Alappat et al. and the
L3-Fusion result all independently recommend the construction: tile the batch so a tile fits
L2, then run all three axes inside the tile. That is not the same experiment the panel ran,
and it is **the largest untried structural move on the board**."

**Three entries built it this round, at three different sizes, and the answer is a regime
boundary with a number on each side.**

- **`gen_pow2`, the clean experiment.** Split the x-FFT into its two radix stages so a chain
  step becomes ONE volume sweep — `x-stage-2(s) + map(s) + z(s+1) + y(s+1) + x-stage-1(s+1)`
  — with the tile resident across all five phases. Bit-identical output to the two-sweep
  engine (`cmp`-verified), so this is a pure structural A/B. **L=128 (DRAM regime, 68 MB
  working set): −9…−15% wall, 5/5 pairs, demand DRAM reads (LLC-load-misses) 42.0M → 16.4M =
  −61%, IPC 0.70 → 0.85.** **L=64 (L3 regime, 8.7 MB): +2% LOSS, 3/3 pairs**, *even though
  the traffic goal landed* (LLC-loads −36%).
- **`gen_pfa_small`**: slab fusion of passes 2+3 per L2-resident x-slab gave −13% on its own;
  replacing the ping-pong volume with one reused 160 KB L2-resident scratch slab took L=100
  from 12,684 → 6,244 µs. **DRAM traffic 188 → 28 MB/step (−6.7×), L3→L2 116 → 51 MB.**
- **`gen_batchlane`**: within-volume SoA gets fused z+y per ~1.3 MB slab "for free" (both
  axes are pure elementwise batch-lane code) and took the cell.

**The answer: double-digit percent when the fused tile deletes a DRAM crossing, ~0 or
negative when the volume is already LLC-resident.** Tolmachev's rule from §07 §1.6 (payoff =
avoided passes × bandwidth gap between the levels involved) survives with CPU numbers
attached at the DRAM gap for the first time. Two sharpenings the corpus does not have:

1. **`l1d.replacement` is the wrong dashboard for this question.** `gen_pow2` states it
   flatly: at L=64 the traffic cut landed and the time went the wrong way; the discriminating
   counter is **demand vs prefetch traffic at the LLC boundary** (`LLC-loads` /
   `LLC-load-misses`), not L1 fills. `gen_planner` independently found 79% of its L1 fill
   traffic at L=100 was hardware-prefetch and RFO amplification, not demand.
2. **A fused operand must take the FUSED consumer's walk order, not the original pass's.**
   `gen_pow2`'s first build kept x-fastest `c` and **lost** (+4% at 64, +10% at 128); storing
   `c` in tile-consumption order flipped L=128 from +10% loss to −14% win. That is a
   reusable rule and it is worth more than the fusion result itself.

### 5.2 §4.1 — "How much spill traffic do batch-vectorised codelets generate, and does it cost more than the shuffles it avoids?" — answered with the exact check §07 §7.8 prescribed.

§07 §7.8's cheap check is: build them, then **count stack traffic in the generated assembly
(`vmovupd` against `%rsp`/`%rbp`) before believing any timing.** `gen_bluestein` did precisely
that on its shipping r10 `conv_mid14`: **566 instructions per 14-block, of which 102 are
`rsp` spill accesses (~20%)** — ~40 live values against 32 registers. Rewriting as five
low-liveness phases staged through the block's own L1-hot slots: **566 → 479 insns,
`rsp` accesses 102 → 1**, same ops in the same order, **outputs bit-identical**, measured
**−3% at L=100, 6/7 pairs**, with counters confirming the mechanism (loads −1.09G, stores
−0.53G, FP ports unchanged).

The answer to §4.1 is **conditional on saturation, and that is the new part.** Their own r8
record had declined this fix on exactly §4.1's reasoning — "spills ride the idle load/store
ports for free" — and that was **right at 1.2 uops/cycle and wrong at 2.03**. At the cap
every spill uop is wall time. So: *2L is a data-only lower bound as §4.1 says; the spill
traffic is real and large (20% of a codelet's instruction stream); and whether it costs
anything depends on whether the engine is dispatch-saturated.* Frigo's 50–100% UltraSPARC
datum is not the price tag here — 3% is — but the direction is confirmed and the measurement
protocol is now proven on this hardware.

### 5.3 §4.8 item 6 — "No primary measurement in the corpus for Ice Lake-SP server parts. Measure it on the node." — measured, and it retires the panel's own port arithmetic.

`gen_dense_prime` shipped **no code** this round (`.text` byte-identical to r10, `objcopy`
cmp) and spent the round on `r11dev/ubcap.c`, a calibration microbenchmark. Findings, on the
scoring node:

- **There is no ~2.1 total-dispatch cap.** The node sustains **3.00** vector uops/cycle with
  demand met exactly (2 FMA + 1 load), and 2.98 at 1 FMA + 1 shuffle + 1 load. The "~2.1
  under mixed loads" figure in `TOOLS.md` describes a coincidence zone, not a wall.
- **The real wall is a 512-bit L1 access ceiling of ~1.12/cycle, loads and stores POOLED.**
  Dead-load asm measures 1.124 zmm loads/cy (72 B/cy) regardless of address pattern;
  **ymm loads run the full 2.0/cy**; adding 512-bit stores shares the same pool.
- **Consequence: zmm FMA saturation is only reachable below 0.56 zmm loads/FMA**, not 1.0.

This is directly responsive to §4.8 item 6, and it **inverts that item's own conclusion in a
second place.** The item argues 512-bit is "strictly preferable on this part … 2× L1 …
load bandwidth … at **zero** frequency cost." The frequency half holds (license level 2 at
~97% of cycles with no visible penalty). The bandwidth half is **measured false on this
SKU**: two ymm halves move 64 B/cy against zmm's 72 B/cy — only 12% behind — and they come
from a *different* budget. `gen_rader` immediately used this to explain a four-round-old
mystery ("dense engine runs 2× above its load-port model"), and `gen_dense_prime` re-derived
`gen_pfa_large`'s disputed 2.11-uops/cy `zsub` reading as ~89% of the *access* ceiling rather
than a total-uop cap — which flips the actionable conclusion. **Every port floor this panel
has computed, and everything `llvm-mca`/`uiCA`/`OSACA` will tell it, assumed 2×512-bit
loads/cycle on ICX. The part does half that.** Run `ubcap`'s `ldonly`/`ldsame`/`ldonly256`
trio on any new host before computing a port floor on it.

### 5.4 §4.5 — "Padding: does L=8 need it, and where?" — moved sideways, in the §08 direction.

§4.5's inset names 4K store→load aliasing as "the bigger hazard … and neither has ever been
checked." `gen_dense_prime` checked it, in miniature: two same-size `aligned_alloc`s put a
microbenchmark's load and store buffers in overlapping 4K phase and cost **+29% cycles and
+0.9G replay uops** versus the phase-disjoint version. Separately, the panel's standing
house rule — pad so slab bytes ≡ 256 mod 4096 — is §04's odd-cache-line rule in production
across `gen_batchlane`, `gen_pfa_small` and `gen_rader` (which adds a +2048 B phase offset
specifically to miss the page-aligned driver `c` volume). The named counter
(`ld_blocks_partial.address_alias`) is still unrun. **Partial credit: the mechanism is
confirmed and priced, the specific L=8 question is untouched because L=8 is not a scored cell.**

---

## 6. The single highest-value thing r12 should attack, per geometry

| L | attack | why this and not something else |
|---|---|---|
| **10, 12, 15** | **Hot-text isolation / link-order discipline, as a shared harness rule.** | Three entries are within 0.3% at 4.0×/4.0×/3.6× the best library; there is no algorithmic gap left. What *does* move these cells is code layout: `gen_twiddle` +0.3…+2.6% at 12 (7/7 pairs) and `gen_pfa_small` +5.3% at 20 both on **bit-identical output**. The cells now lose more to text displacement from work done elsewhere in the file than to any remaining idea. Cheapest fix with the widest reach: separate TU or section placement for the small-L hot paths so a large-L feature cannot tax them. |
| **20** | **Recover `gen_pfa_small`'s 5.3%, then build `gen_twiddle`'s radix-20 (PFA 4⊗5) leaf codelet.** | The regression is §2.1 and is pure layout. The structural item is `gen_twiddle`'s own #2: 20 = [4,5] still pays a combine level that a 20-point PFA leaf deletes outright — the same derivation that just bought −18% at L=10 and −11% at L=100, with no fold and pure exact constants. It pays at **20 and 40** (40 = 2·20). |
| **25, 27** | **A shared low-op DFT25 (and DFT9) module — Winograd or better-scheduled CT.** | `gen_powp` leads at 2.44×/2.21× and the residual is arithmetic: `gen_planner` has carried "Winograd DFT9/DFT25" for five rounds and the 25/27 gap to `gen_powp` still stands. The leverage is that **DFT25 is now on the critical path at four cells** — 25, 50 (2×25), 100 (4×25) — and three entries wrote it independently this round (`gen_pfa_large` r1's shape, `gen_pfa_small`'s `gdft25`, `gen_batchlane`'s 9 compiled-in w25 constants). One better module improves four cells. Cost the candidate in `gen_dense_prime`'s corrected currency: max(FMA/2.0, 512b-accesses/1.12), not the old port floor. |
| **31** | **`gen_dense_prime`'s asm-pipelined drain — or declare the cell closed and redeploy.** | This is the board's strongest cell (2.52× the best library, **10.04× MKL**) and `gen_rader` calls the class converged, with the residue quantified as chunk-local L1 thrash from working sets that exceed L1D by construction (83 KB/chunk against 48 KB) — and shrinking them meaningfully saves only 3–13% against a 2× overshoot. The one unplayed card is `gen_dense_prime`'s asm drain (their own estimate 15–25%), a full-round bet on a cell they trail by 1.31×. **Also: fix the `gen_rader.c:2429` UB warning (§3.2) — one line, eleven rounds old, in the co-leader's hot path.** |
| **32** | **Nothing algorithmic. Guard the ±3% address sensitivity.** | `gen_pow2` leads at 3.17× MKL, has been bit-identical for five rounds, and calls the cell closed with attribution (`[port ‖ L2]`, r8). The measurable risk is that L=32's fused-exit loop is address-sensitive at ±2–3% (`gen_planner` documented this exhaustively) — so this cell's r12 item is the same hot-text discipline as 10/12/15, not a kernel. |
| **40** | **Delete a level: radix-20/radix-8 leaf (40 = 8×5 = 2·20).** | Two engines are tied at 2.54× MKL and `gen_pfa_large`'s own class list has nothing left at 40 (their remaining items are xarch and coverage holes at 60/84/90/96/105/108/120/126). Crucially, `gen_pow2` measured that **fusion will not pay here**: L=40's 15 MB working set is the L=64 regime where their fused sweep *lost* 2%. So attack arithmetic/level count, not traffic — and check `LLC-load-misses` before spending a window on any traffic idea at this size. |
| **50** | **FP per site, via the shared DFT25 above.** | Three engines converge at 411–418 µs (2.30× MKL) and `gen_batchlane` measured *why* its new engine lost 14% here: at L=50 the cell is compute-bound (1.65 uops/cy, IPC 1.45, never leaves L3) and the winner simply runs leaner arithmetic — **2.17 vs 2.52 FP/site**. They explicitly say do not chase it with knobs. That makes 50 an arithmetic cell, and the arithmetic is DFT25. `gen_pfa_small`'s slab transfer (their #2: race `GSLAB_MIN` down through 34/38/42/44, where they measured −21% and −28%) is the secondary item. |
| **100** | **One-sweep fusion with tile-order `c`, `gen_pow2`'s L=128 construction, ported to 4×25. Nobody has tried it here.** | This is the clearest open lever on the board. `gen_pfa_large`'s accounting says L=100 is **~88% DRAM-BW-bound end to end** (80 MB/step at the measured 19–20 GB/s single-core ceiling = ~4.0 ms of a 4.53 ms step), so **only traffic cuts can pay** — and they closed the overlap axis from three directions (both-streams prefetch +5.3…7.1%, T2 +3.9…4.7%, c-flush +4…8%, NTA +12.7%). What nobody has built at 100 is the one construction that cut demand DRAM reads **−61% for −14% wall at the analogous L=128 regime**: fusing the x-stage *across the step boundary* with `c` stored in tile-consumption order. `gen_batchlane` fused z+y **within** a slab and `gen_pfa_small` fused passes 2+3 **within** a step; neither crossed the step boundary, and neither reordered `c` to the fused consumer's walk order — the change that flipped `gen_pow2`'s L=128 from +10% loss to −14% win. `gen_pow2` wrote the transfer note ("L=100 B=1 is the L=128 regime; the 4×25 axis split gives tiles of 4 or 25 planes and the same label algebra applies") and no one acted on it. **Secondary:** `gen_twiddle`'s axis-0 stream-count tiling — the axis-0 pass walks ~100 concurrent 128 B row streams against ~16 the prefetchers track, and their `l1d.replacement` sat at 226 MB/step against a 128 MB floor and did not move when they deleted 37% of load dispatch. |

**Cross-cutting, one item:** every entry's port floor and every "we are saturated" claim in
the strategy records predates §5.3 and was computed in the wrong currency. **Re-cost the
large cells as max(FMA/2.0, 512-bit-accesses/1.12) before designing anything**, and note that
the **256-bit access class is an unused 2/cycle budget** — the obvious home for staging
round-trips that currently compete with the FMA feeds. `gen_dense_prime` names the two
entries whose stage shapes should be re-checked against the 0.56 loads/FMA boundary first
(`gen_powp`, `gen_batchlane`), and `gen_pfa_large`'s `zsub`/`ysub` accounting explicitly
needs redoing.

---

## 7. What to keep, and why

Applying `docs/CURATION.md`'s four grounds in order. **All twelve entries qualify**, which is
what a converged round-11 panel looks like when every entry beats every library at every cell
it enters — so the value of this list is the *annotation*, not the selection. Ordered by
ground:

**Ground 1 — fastest correct entry per geometry (mandatory, one per L):**

- **`gen_pfa_small`** — L=10 (1.121 µs, 3.97× best library). Also the round's second engine
  at L=100 (6066.2, 1.29× MKL) with the traffic teardown that priced the whole cell:
  188 → 28 MB/step DRAM.
- **`gen_batchlane`** — L=12 (1.915), L=15 (4.340), L=20 (12.597), and the engine behind the
  L=100 co-lead (4072.3). Carries the round's winning structural idea: within-volume SoA at
  B=1, 8 x-planes of one volume per zmm lane-slot — approach #4, previously untried by anyone.
- **`gen_powp`** — L=25 (30.857), L=27 (43.484), plus −4.2% at L=100 from the first adoption
  of `gen_layout`'s THP re-home (dtlb walks −98%).
- **`gen_race`** — L=31 (84.549), L=40 (159.534), L=50 (410.975), L=100 (4071.340). The
  racing/wisdom layer, verified honest against the correctness digits (§1.1).
  **Promotion condition: the missing r11 strategy record (§3.5) must be written before the
  commit.**
- **`gen_pow2`** — L=32 (53.809, 3.17× MKL), and it also carries the round's most valuable
  negative (below).

**Ground 2 — structurally different runner-up, close:**

- **`gen_rader`** — L=31 at +0.27%, and it *is* the engine `gen_race` routes at 31; without
  it the board's strongest cell has no source code behind it. Independently instructive: it
  built the zero-copy THP re-home and **reverted it on node evidence** (below).
- **`gen_pfa_large`** — L=40 at +0.31% (the engine `gen_race` routes at 40), L=50 at +1.7%,
  and at L=100 a structurally distinct two-sweep + prefetch-ahead shape 11.9% off the lead
  with the 88%-DRAM-bound accounting that scoped the entire round.
- **`gen_planner`** — L=100 at +12.3% with a construction no other entry has written down:
  the alternating (x,y,z)↔(x,z,y) layout that halves transposes to one in-place square
  transpose per plane. Also the only entry with a create()-time gate that falls back to the
  classic chain if the new machinery disagrees.

**Ground 3 — instructive failures whose record documents the number that killed them:**

- **`gen_pow2`** — the one-sweep fused chain step **ships at L=128 (−14% wall, −61% demand
  DRAM reads) and measured OUT at L=64 (+2%, 3/3 pairs)**, with counters that killed its own
  four-round-old "32.5 GB/s L3-bandwidth wall" theory. Two negatives with numbers: the
  x-fastest `c` layout (+4%/+10%) and the L=64 fusion itself. This record stops r12 from
  rebuilding fusion at 40/50.
- **`gen_rader`** — built both of `gen_layout`'s zero-copy changes, raced them, and
  **reverted both**: memcpy deletion **+0.6% at 127 m=2 (4/4)**, THP re-home **+4.1% (4/4
  non-overlapping)**. The mechanism is the round's best-explained surprise — an in-place
  custody chain pays no RFO, so a "zero-copy" exit to a cold `final_out` *adds* a 32 MB RFO
  read, and glibc's 32 MB memcpy writes NT. "Fewer sweeps" accounting that ignores RFO and
  NT gets the sign wrong. It also contributed the regime boundary back to `gen_layout`.
- **`gen_dense_prime`** — shipped **no code change at all** (`.text` byte-identical) and
  produced the round's highest-leverage result anyway: `ubcap.c`, which retires the ~2.1-uop
  cap, establishes the **1.12/cycle pooled 512-bit access ceiling** and the 0.56 loads/FMA
  saturation boundary, and invalidates every port floor in every strategy record (§5.3). Plus
  three whiteboard-struck cross-class entries costed with numbers so nobody re-costs them
  (100/40/50) and a reproduced 4K-aliasing hazard (+29% cycles). Wins no cell; keep it for
  the measurement.

**Ground 4 — beat a library baseline, regardless of rank:**

- **`gen_twiddle`** — beats MKL 2022 at **9 of 11 cells** (10, 20, 25, 27, 31, 32, 40, 50,
  100) despite ranking mid-board, and delivered the round's largest non-L=100 gain: radix-10
  PFA 2⊗5 codelet levels, one whole level pass deleted at seven sizes, **−11% at 100, −12% at
  50, −18% at 10, all 4/4 pairs**, with L=100 single accuracy *improving* as a side effect. It
  is also the twiddle library three other entries depend on.
- **`gen_layout`** — beats **every** library at L=31 (203.3 vs 213.4) and MKL at L=50, and
  its finding drove the round: THP mode is `madvise` on kernel 5.15, so the driver's
  `posix_memalign` buffers get **zero** huge pages, verified with a new `gl_thp_bytes` smaps
  primitive. Three entries adopted it (`gen_powp` −2%, `gen_planner` inside its −10%,
  `gen_rader` rejected it with a boundary). Kept also for the honesty: they measured the
  mechanism at **−96% page walks** and reported the time win as **−0.5%, not the ~2% the raw
  counters suggest**, because the OoO window was already hiding most walk latency.
- **`gen_bluestein`** — beats MKL by **3.1×** at L=31 and 8 of 9 libraries there, from last
  place on the board elsewhere. Kept primarily for §4.1's answer (§5.2): the objdump spill
  census, the 102 → 1 `rsp` fix, and the retraction of its own r8 reasoning with the
  condition attached ("spills are free below the cap, wall time at it"). Also documents a new
  degree of freedom nobody had used — chirp periodicity makes cyclic convolution exact at
  M = any multiple of 2L, so M ≥ 2L−1 is not the only legal grid.

**Exclusions I checked and rejected:** nothing failed correctness (187/187 `ok`); nothing
failed to build (one pre-existing warning); nothing crashed except the harness floor at
L=100 (pre-existing, expected); no strategy record is absent (`gen_race`'s is stale for r11,
which is a blocking condition on its promotion rather than a disqualification); and no pair
is a near-duplicate — `gen_pfa_small` and `gen_batchlane` converge at 10/12/15/20 but
diverged sharply this round at 50/100 (slab-fused split step vs within-volume SoA, 726.6 vs
475.5 and 6066.2 vs 4072.3).

---

*Monitor's note: this round's real deliverable is not the −10% at L=100. It is that three
entries independently measured the machine instead of modelling it — the 512-bit access
ceiling, the THP/smaps verdict, and the demand-vs-prefetch discriminator — and each of the
three overturned an accounting the panel had been reasoning from for four or more rounds. The
L=100 gain is downstream of that. r12 should keep the counter protocol mandatory.*
