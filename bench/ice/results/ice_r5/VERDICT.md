# Round ice_r5 — monitor's verdict

Scored on `a80n0.lqcd.mit`, slurm job 438633, 2026-08-23T05:10:18-04:00.
19 panel entries + `baseline_matrix` + 6 library backends, 8 geometries, 3 processes each.

## 0. Two corrections to the brief, before any number is read

**0.1 The scoring machine is not a Cascade Lake Gold 5218.** `environment.txt` reports
`Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` with `avx512_vbmi avx512ifma avx512_bitalg
avx512_vpopcntdq` in the ISA string. Those four are Ice Lake-SP (Sunny Cove) features;
Cascade Lake has none of them. The node is a dual Gold 6326 (2 × 16C/32T = the reported 64
logical cores), which means **1.25 MiB L2 per core, not 1 MiB, and two 512-bit FMA units,
not one.** The round tag `ice` has been right all along; the brief and `docs/LITERATURE.md`
are the things that are stale.

This is not cosmetic. `docs/LITERATURE.md` §4.8 item 6 and the §08 corrections throughout
§3 build their whole clock-and-port model on Intel's turbo table for the **Gold 5218** —
"one 512-bit FMA unit, so 512-bit and 256-bit code have *identical* peak FP throughput" and
"2.9 GHz flat from 1 to 8 active cores". Neither statement describes this part. On a 6326,
512-bit is genuinely ~2× the FP peak (not merely half the instruction count at equal peak),
and single-core turbo reaches ~3.5 GHz. Several entries have already worked this out
empirically from the node rather than the docs — `L17_matrixsimd` this round targets "ports
0/5, the two FMA pipes the kernel is bound on", and the L=6/L=13/L=17 records all carry
measured `clk512/clk256 = 3.30/3.50 GHz` probes that contradict the documented 2.90 flat
model. **`docs/LITERATURE.md` §3.1's "batch tile for the 1 MiB L2" row and §4.8 item 6
should be reissued against the Gold 6326 before the next round tunes to them.**

**0.2 This round contains no non-batched measurement.** `cases.txt` — "THE GRADED
CONFIGURATION, fixed, identical for solver/base/SOTA" — has eight rows, `6:64:4856`
through `64:2:134`, and no `B=1` row. `sweep.sh` iterates exactly that file. So although
`fft3d_api.h` states that "Batched (B>1) and non-batched (B=1) are separate measurements",
ice_r5 measured only the batched chain, as did ice_r1 through ice_r4. The headline below
therefore gives the scored batched figure, and for B=1 reports the implementers' own
self-measured numbers, clearly labelled as unscored. **Adding B=1 rows to `cases.txt` is a
one-line change and would close a gap the API contract has promised for five rounds.**

---

## 1. Headline per geometry

Scored figures are µs per transform, minimum over three independent processes, from
`leaderboard.txt`. B=1 figures are from the implementers' strategy records, measured on the
same node under lease but **not** scored by the harness.

### L = 6 (V = 216, B = 64, m = 4856)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L6_pfa 0.304 µs** (27.58 GF/s) | mkl_dfti 0.939 µs | **3.09×** |
| non-batched (unscored, implementer-reported) | L6_unrolled 0.330 µs; L6_pfa 0.346 busy-window (≈0.304 quiet-scaled) | not measured at B=1 by either record | — |

> **Carry this caveat with the L=6 headline.** The 6.3 % gap between `L6_pfa` (0.304) and
> `L6_unrolled` (0.323) is a *precision-tier choice*, not a speed result — see §3.3. At
> matched tier the order reverses in both directions.

### L = 8 (V = 512, B = 64, m = 2572)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L8_radix8 0.570 µs** (40.41 GF/s) | mkl_dfti 2.095 µs | **3.67×** |
| non-batched (unscored, implementer-reported) | L8_radix8 0.570 µs (batch-invariant by design; B=2048 reads 0.581) | not paired at B=1 | — |

The fastest cell on the board in GF/s terms, and the only geometry where the winner's code
is byte-for-byte the r4 winner.

### L = 17 (V = 4913, B = 32, m = 98)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L17_matrixsimd 12.736 µs** (23.65 GF/s) | ducc0_c2c 86.194 µs | **6.77×** |
| non-batched (unscored, implementer-reported) | L17_matrixsimd 14.403 µs | MKL 99.6 µs, same window | 6.9× |

The largest margin over any library at any geometry, and it grew again (6.64× → 6.77×).
Every library agrees to within 5 % here — `ducc0` 86.2, MKL 88.5/88.8, FFTW 90.1 at all
three planner levels — which is the corpus's "neither library has a good 17" result
reproduced for the fifth consecutive round.

### L = 36 (V = 46656, B = 8, m = 64)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L36_mixedradix 103.888 µs** (34.83 GF/s) | mkl_dfti 283.228 µs | **2.73×** |
| non-batched (unscored, implementer-reported) | L36_pencilfused 123.4 µs (MKL 322, same window → 2.6×); L36_mixedradix 129.1 µs (MKL 326.4 → 2.5×) | | |

Note the inversion at B=1: `L36_pencilfused` is ahead of `L36_mixedradix` in the
implementers' own unscored B=1 runs, while `L36_mixedradix` wins the scored B=8 cell. That
alone is an argument for scoring B=1.

### The other four geometries, for completeness

| L | panel best | best library | margin |
|---|---|---|---|
| 13 | L13_direct 5.401 µs | mkl2026_dfti 11.801 µs | 2.18× |
| 23 | L23_matrixsimd 36.903 µs | ducc0_c2c 248.047 µs | 6.72× |
| 45 | L45_mixedradix 262.959 µs | mkl_dfti 758.065 µs | 2.88× |
| 64 | L64_blocked 639.508 µs | mkl_dfti 1721.297 µs | 2.69× |

**Every panel entry at every geometry beats every library backend.** The slowest panel
entry on the board (`L36_pfa`, 3rd of 3 at L=36) still beats the best library at its
geometry by 2.45×. `CURATION.md` promotion ground 4 ("anything that beat a library
baseline") is therefore vacuous this round and plays no part in §7.

---

## 2. What changed since ice_r4, per geometry

Per-transform µs, r4 scored → r5 scored.

| L | entry | r4 | r5 | Δ |
|---|---|---|---|---|
| 6 | **L6_pfa** | 0.363 | **0.304** | **−16.3 %** |
| 6 | L6_unrolled | 0.332 | 0.323 | −2.7 % |
| 8 | **L8_radix8** | **0.564** | 0.570 | **+1.1 %** ← see below |
| 8 | L8_fusedaxes | 0.744 | 0.585 | −21.4 % |
| 8 | L8_batchsimd | 0.779 | 0.596 | −23.5 % |
| 13 | L13_direct | 5.837 | 5.401 | −7.5 % |
| 13 | L13_rader | 6.363 | 5.808 | −8.7 % |
| 17 | L17_matrixsimd | 13.009 | 12.736 | −2.1 % |
| 17 | L17_winograd | 14.854 | 14.412 | −3.0 % |
| 17 | L17_rader | 17.521 | 15.052 | −14.1 % |
| 23 | L23_matrixsimd | 44.872 | 36.903 | −17.8 % |
| 23 | L23_rader | 38.105 | 37.780 | −0.9 % |
| 36 | L36_mixedradix | 111.425 | 103.888 | −6.8 % |
| 36 | L36_pencilfused | 111.962 | 108.631 | −3.0 % |
| 36 | L36_pfa | 128.551 | 115.437 | −10.2 % |
| 45 | L45_mixedradix | 283.800 | 262.959 | −7.3 % |
| 45 | L45_pfa | 283.339 | 263.807 | −6.9 % |
| 64 | **L64_blocked** | 1048.605 | **639.508** | **−39.0 %** |
| 64 | L64_radix8 | 1042.956 | 1011.066 | −3.1 % |

### Did anything regress?

**No entry regressed in code.** One entry regressed on paper:

* **L=8, `L8_radix8`: 0.564 → 0.570, +1.1 %.** This is not a regression. Its record states
  v2 shipped unchanged and, decisively, its scored chain drift is `2.599e-11` — identical
  to r4 to every printed digit, which it can only be if the chain is bit-for-bit the r4
  chain. Its own in-lease runs read 0.570 steadily. The 0.006 µs is the node's documented
  bimodal timing window (§4), and 1.1 % sits inside the 1.7 % run spread the leaderboard
  itself reports for the entry. Nothing to answer for.

Two **rank changes** that are not regressions:

* **L=6:** `L6_unrolled` → `L6_pfa`. Both got faster; the lead changed hands. But the
  flip is confounded by a precision-tier difference (§3.3) and reverses at matched tier.
* **L=23:** `L23_rader` → `L23_matrixsimd`. `L23_rader` did not slow down (−0.9 %); it was
  overtaken by an 17.8 % improvement in the dense arm. A clean pass.

### The round's structural story

Sorting the movers by mechanism gives an unusually clean result, and it is the same result
at every scale:

* **Everything that gained more than 10 % did so by changing where the chain state lives
  between steps, or by unblocking a saturated functional unit.** `L64_blocked` (−39 %)
  moved the whole m=134 chain into an L3-resident split-complex custody layout, one volume
  at a time, cutting per-step traffic from ~30 MB DRAM-touching to ~22.3 MB all-L3 at
  *unchanged* op count. `L23_matrixsimd` (−17.8 %), `L8_batchsimd` (−23.5 %) and
  `L8_fusedaxes` (−21.4 %) all went volume-major / per-volume-resident. `L36_mixedradix`
  (−6.8 %) found by 4-way TSC split that its map was **divider-bound** — 5832 `vsqrtpd` per
  volume at ~18–20 cy reciprocal throughput, 36 % of the step against a ~9 µs issue floor —
  and halved it by alternating pairs between the divider and the FMA-port `rsqrt14+2N`
  ladder so both units run concurrently.
* **Everything that tried to gain by deleting memory traffic inside an already-resident
  chain lost.** See §5.

---

## 3. Adversarial pass: correctness, builds, crashes, omissions

### 3.1 Nothing is missing, nothing failed to build, nothing crashed

* `build_errors.txt` is **0 bytes**. No entry failed to build.
* `failures.txt` **does not exist**. No entry crashed or hung.
* `agents/exits.txt` records **exit=0 for all 19** implementer agents.
* All 19 implementations present in `impl_5/` (+ `baseline_matrix.c`), all 19 appear in
  `leaderboard.txt`, and all 19 strategy records were updated this round
  (`git diff --stat strategies/` = 2671 insertions across 19 files). No entry is promotable
  but recordless.

### 3.2 Every panel entry passes both correctness gates

I checked the raw `c_*.json` rather than trusting the leaderboard column, because the
leaderboard column does not report the chain gate (see §3.4). All 19 panel entries pass
**both** the single-transform check (`rel_l2` between 2.29e-16 and 4.46e-16, tol 1e-12) and
the m-step map-chain check. Margins against `eff_tol = max(1e-12, 1e-13·m)`:

| L | tol | tightest panel entry | margin |
|---|---|---|---|
| 6 | 4.86e-10 | L6_pfa 2.54e-10 | **1.9×** ← see §3.3 |
| 8 | 2.57e-10 | all three 1.62e-11 | 16× |
| 13 | 1.28e-10 | L13_direct 1.12e-13 | 1140× |
| 17 | 9.8e-12 | L17_matrixsimd 1.56e-14 | 628× |
| 23 | 1.65e-11 | L23_matrixsimd 3.47e-14 | 475× |
| 36 | 6.4e-12 | L36_pfa 1.61e-14 | 398× |
| 45 | 1.77e-11 | L45_mixedradix 3.60e-14 | 492× |
| 64 | 1.34e-11 | **L64_blocked 1.80e-12** | **7.4×** |

No fast wrong answer survived. But two of those margins want naming.

### 3.3 The L=6 gate certifies seed luck, and the L=6 ranking is a precision-tier artifact

This is the round's most serious finding and it is not a defect in either entry.

**The gate.** At L=6, m=4856, the panel passes at 2.47–2.54e-10 against a 4.86e-10
tolerance — under 2× headroom — and `mkl2026_dfti` passes at **4.16e-10, which is 86 % of
tolerance**. Meanwhile *both* L=6 implementers independently measured, on their own seeds,
that essentially every backend on the machine **fails** this gate: `L6_pfa`'s record gives
MKL 6.2e-10, FFTW 1.5e-9, ducc0 1.5e-9, `baseline_matrix` 4.0e-9, and numpy-vs-numpy
1.1e-9; `L6_unrolled` reports its own shipped chain at 1.403e-9. The scored run and the dev
runs differ only in the seed (`sweep.sh` derives it as `SEED + L*1000 + B`). A 6× spread in
end-state drift from the seed alone, on a chaotic map iterated 4856 times, is exactly what
you would expect — and it means **the L=6 chain check is currently measuring which
trajectory the seed picked, not whether the arithmetic is sound.** Both implementers said so
in their records, in r4 and again in r5. The r4 monitor's L=6 directive was "recalibrate the
gate, then take the fdiv arm." The gate was not recalibrated. `check.py` still reads
`eff_tol = max(1e-12, 1e-13 * m)`.

**The consequence for the ranking.** Given a gate that does not bite, the two L=6 entries
made *opposite* policy choices, and the leaderboard scored them as if they were comparable:

* `L6_pfa` shipped the **uncompensated ("fast") ladder** as its default — 0.304 µs. Its own
  record measures that tier drifting **3.247e-9 on seeds 42/900042**, i.e. 6.7× over
  tolerance. Its exact arm (`sep`, compensated, drift 1.227e-9) measures **0.364 µs**.
* `L6_unrolled` shipped the **exact FMA-Heron ladder** — 0.323 µs — explicitly reasoning
  that if the monitor recalibrated to a tight "admit ≥1 exact implementation" floor,
  dropping Heron "converts my cell from best-exact-entry into a rejection." Its fast
  (`NOHERON`) arm measures **0.299 µs**.

So the scored 6.3 % margin **reverses at either matched tier**:

| tier | L6_pfa | L6_unrolled | winner |
|---|---|---|---|
| fast / uncompensated | 0.304 (shipped) | 0.299 (`-DL6_MAP_NOHERON`) | L6_unrolled |
| exact | 0.364 (`sep`) | 0.323 (shipped) | L6_unrolled, by 13 % |
| **as scored** | **0.304** | 0.323 | L6_pfa |

`L6_pfa` takes the headline: it was the fastest correct entry under the gate as written and
as run, and it passed. I am not disqualifying it — the gate is the harness's problem, not
the entry's, and `L6_pfa` documented its tier choice openly and kept the exact arm one
environment variable away. But the L=6 cell is **not a measurement of engineering quality
this round**, and it must not be read as one. Both entries are promoted, and the next round
must not attack L=6 speed until the gate is fixed (§6).

### 3.4 `baseline_matrix` at L=6 failed the chain gate and the leaderboard printed "ok"

`check.log` line 2:

```
FAIL map-chain m=4856: rel_l2=1.687e-09 (tol 4.9e-10)
```

`c_baseline_matrix_L6_B64.json` confirms: `"chain_ok": false, "chain_rel_l2":
1.6867684131893238e-09`. Yet `leaderboard.txt` line 19 shows
`baseline_matrix ... ok 6.0e-16 24.16x`.

**Root cause, and it is a live hole in the scoring path.** `check.py` line 39 computes `ok`
from the single transform and stores it at line 41. Line 67 then does `ok = ok and
chain_ok` — but only `chain_ok` and `chain_rel_l2` are written back into `result` (lines
69–70); **`result["ok"]` is never reassigned.** The process exit code is correct
(line 105), but `leaderboard.py` line 84 reads `chk["ok"]` from the JSON, so a chain
failure is structurally incapable of reaching the ranking. Line 6 of `leaderboard.py`
promises "a backend that failed correctness is shown but never ranked" — for chain failures
that promise is not kept.

This round it bit only `baseline_matrix`, which is the harness floor and not a competitor,
so no panel result is invalidated. **But this is precisely the path by which a fast wrong
answer wins a future round, and it has now been reported by implementers in two consecutive
rounds' records without being fixed.** It is the single highest-priority harness change.

### 3.5 `L64_blocked`'s accuracy margin is the thinnest real one on the board

`L64_blocked` passes at `1.80e-12` against a `1.34e-11` tolerance — 7.4×. Its rival
`L64_radix8` passes the identical case at `3.90e-14`, i.e. **46× more accurate**. The
difference is `L64_blocked`'s all-FMA `MAPDIV=0` reciprocal ladder, and its own record is
candid: "8× margin is already the floor I am comfortable signing." At m=134 that is fine and
the entry is correct. But `L64_blocked` is the round's biggest win by a wide margin, and if
the L=64 chain is ever lengthened, its margin scales the wrong way. Flagged, not penalised.

### 3.6 `L36_mixedradix`'s winning margin is smaller than its own run-to-run spread

The leaderboard reports 33.1 % run spread for the L=36 winner. The three process minima are
**138.22 / 103.89 / 104.14 µs**; the leaderboard takes the min, so the headline 103.888 is
the best of a set containing one 33 % outlier. Its own record identifies the cause — the
first run drew a contended window ("contended-class windows 116.6–117.0", "128.9–129.8"), a
mode every L=36 record documents.

The ranking survives the scrutiny: under **median** of runs it is `L36_mixedradix` 104.14
vs `L36_pencilfused` 110.02, still a 5.3 % win, and `L36_pencilfused`'s own three runs
(108.63/110.54/110.02) are tight. So the verdict stands — but the 4.4 % headline margin at
L=36 is inside the winner's spread and should be reported as "ahead, not separated".

The same caution applies harder at **L=45**, where `L45_mixedradix` 262.959 and `L45_pfa`
263.807 are 0.32 % apart for the second round running — and the order **inverts under
median** (264.73 vs 264.57). L=45 is a tie, not a result. Both are promoted on that basis.

---

## 4. Claimed versus measured

**The premise in the brief does not apply to this round.** Every strategy record states its
measurements were taken on `a80n0` — the scoring node — via `tryout.sh` or a direct core
lease ("Measured on the node (a80n0, tryout.sh)", "tryout = leased core on a80n0", "a80n0
leased cores"). There is no Sapphire-Rapids-to-Cascade-Lake transfer in ice_r5, and no MKL
spanning 2.9×: same-window MKL controls in the records read 0.94, 2.11, 88.8, 288 µs at
L=6/8/17/36, matching the scored MKL rows to within a few percent. Accordingly there is
**no entry whose claim is far from what was measured.**

| entry | implementer's claim | scored | Δ |
|---|---|---|---|
| L6_pfa | 0.304 (quiet window) | 0.304 | 0.0 % |
| L6_unrolled | 0.323 (quiet window) | 0.323 | 0.0 % |
| L8_radix8 | 0.570 steady in-lease | 0.570 | 0.0 % |
| L8_fusedaxes | 0.585–0.600 | 0.585 | bottom of range |
| L8_batchsimd | 0.596–0.597 | 0.596 | 0.0 % |
| L13_direct | 5.411 / 5.407 | 5.401 | −0.1 % |
| L13_rader | predicted 5.8–6.15 | 5.808 | in range |
| L17_matrixsimd | 12.768 min; projected 12.6–12.8 | 12.736 | in range |
| L17_winograd | 14.677 min (contended) | 14.412 | scored 1.8 % faster |
| L17_rader | 15.143 min measured | 15.052 | −0.6 % |
| L23_matrixsimd | 36.886 / 36.909 | 36.903 | 0.0 % |
| L23_rader | 37.96–38.26 | 37.780 | −0.5 % |
| L36_mixedradix | 103.26 best quiet | 103.888 | +0.6 % |
| L36_pencilfused | 107–112, best dev 109.4 | 108.631 | in range |
| L36_pfa | 115.254 min | 115.437 | +0.2 % |
| L45_mixedradix | ~272.6 | 262.959 | scored 3.5 % faster |
| L45_pfa | 265.909 | 263.807 | scored 0.8 % faster |
| L64_blocked | 667.5 (best window 660.1) | 639.508 | scored 3.1 % faster |
| L64_radix8 | 1044.7 / 1173.3 in two windows | 1011.066 | scored at/below range |

The residual scatter is ±4 %, it goes in **both** directions, and where it is largest the
scored number is *faster* than the developer's — the opposite of what a hostile-machine
story predicts. The cause is documented independently by at least six records: **the node
sits in one of two whole-run clock modes, ramped ~3.30 GHz or loaded ~2.90 GHz, and the
readings convert between them by exactly ×3.3/2.9.** `L6_unrolled` verified this
same-binary-same-core with only the window changing (0.323 ↔ 0.367, 0.330 ↔ 0.376);
`L36_pfa` shows 117.4 × 3.3/2.9 = 133.6. Taking the min over three processes filters most
of it, which is why the table above is as tight as it is.

**One prediction, not one measurement, missed.** `L17_rader` projected "~13.0–14.0 µs/step
scored" on the reasoning that having ported `L17_matrixsimd`'s chain engine it would inherit
that entry's window. It scored 15.052. Its own dev measurement was 15.143 — so the
measurement was right and only the extrapolation was wrong. Recorded so the next round does
not repeat the inference that a borrowed engine transfers a borrowed number.

**Secondary note.** The clock probes disagree across entries: L=17/L=13/L=23 records report
`clk512/256 = 3.30/3.50 GHz`, while L=6/L=36/L=45/L=64 report `clk=2.90`. `L6_unrolled`
changed its own reading from 3.30 (r4) to 2.90 (r5). With `governor: schedutil` and a
bimodal window, a create-time probe is sampling whichever mode it landed in. Entries that
derive port floors from a probed clock are deriving them from a coin flip; the floors in
§08-style analyses should be quoted at both 2.90 and 3.30 until a pinned-governor
measurement exists.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

**§4.3 — "Is axis fusion worth 3× or 3%?" — moved decisively, and it is now answerable in
both of its regimes.**

§4.3 was marked "SETTLED IN PART by panel_r3 §5, and re-opened in one regime by §08 §1.9".
The re-opened regime was **L2↔DRAM**, where the bandwidth gap is 7× rather than the 2.6× the
panel had been fusing across, and §08's recommended construction was: *tile so a tile fits
the level, then run all three axes inside the tile*. §4.3 called that "the largest untried
structural move on the board."

This round tried it, and simultaneously produced three independent falsifications of the
naive version. Both halves matter.

**(a) Inside a chain already resident in the target level, fusion is negative — and the
mechanism is the reorder buffer, not traffic.** Three entries built it and killed it with
matched-pair numbers:

* `L8_radix8` built "v3", a genuine single-pass fused chain step deleting the 8 KiB state
  round-trip — **256 of 640 memory ops per step, 40 % of L1 traffic**. It lost by **+15 %**
  (0.654 vs 0.570 same-window), robust to spills, unrolling and the step-loop branch. Its
  post-mortem is the transferable part: the fused iteration is ~420 uops against a
  352-entry ROB, so consecutive iterations stop overlapping *at allocation*; and the work
  fusion absorbed was exactly the eight small independent ~90-uop phase-A iterations the
  OoO engine had been using as ILP filler between the long-latency ladder chains. "Fusion
  removed the filler and lengthened the groups: both directions wrong."
* `L6_unrolled` built 2-pass fusion in **both** directions — store-side (0.380) and,
  in r4, load-side (0.360–0.408) — against 3 passes with a flat map (0.323–0.333). The
  bisect that explains it is the round's cleanest single number: with the ladders stripped,
  the 2-pass skeleton runs at **0.216 µs, i.e. the 2-pass structure itself is free**. The
  loss is entirely that a ~74-cycle ladder chunked behind a ~190-uop plane body cannot be
  hidden. Op-count changes had *inverted sign* in that shape — the signature of a
  latency-bound, not throughput-bound, arrangement.
* `L17_matrixsimd` built the register-fused map (v5) — mathematically elegant, map commutes
  with the tile transpose, correct at 2.197e-14 — and lost **+1.7 µs/step**. Spills ruled
  out (450 vs 423 stack refs). Same diagnosis: a ~400-uop monolith the ROB cannot span.

**(b) Across a level boundary the working set does not yet fit, restructuring for residency
is worth 39 % — at unchanged operation count.** `L64_blocked` executed §08 §1.9's
construction literally: the chain state moved into a split-complex padded custody layout and
**each volume iterates all m=134 steps while resident**, one volume at a time. Working set
per volume 8.9 MB, fully L3-resident, against r4's ~30 MB that forced NT stores and DRAM
round trips. Per-step traffic ~22.3 MB, all L3. Its own op count is "essentially r4's
count — the round's entire win is traffic and residency". Measured on one build in adjacent
windows: **667 custody vs 1226 r4-pipeline**. Scored: 1048.6 → 639.5, **−39 %**, and the
L=64 margin over the best library went 1.65× → 2.69×.

**The synthesis, which is Tolmachev's rule with CPU numbers at a second boundary.** The
payoff of fusion is the number of avoided passes times the bandwidth gap of the level you
actually cross — *and nothing more*. When the chain is already resident in that level the
gap is zero, the traffic argument evaporates, and what is left is a pure cost: fused bodies
exceed the ROB and destroy the cross-iteration overlap the OoO engine was living on. So:
**fuse to achieve residency, never fuse within it.** §4.3 should be updated to record that
its re-opened L2/L3↔DRAM regime is now closed in the corpus's favour (39 %), that the
L1↔L2 regime is closed against it with a mechanism (ROB width, ~352 uops, measured three
independent ways), and that `L8_radix8`'s v3 is the specific negative nobody at L=8 needs to
repeat.

**Secondary movement.** §4.8 item 6 asked for a primary AVX-512 measurement on an
Ice Lake-SP *server* part, noting "there is no primary measurement in the corpus" and
"Measure it on the node." Five ice rounds on a Gold 6326 are that measurement — but §0.1
above means the corpus is currently filing those numbers under the wrong SKU, with the wrong
L2 size and the wrong FMA-unit count. Closing item 6 requires reissuing it against the
6326, not just citing the rounds.

**§4.2 (dense-symmetric vs Rader at a prime) also moved, as a by-product.** `L17_rader`
ported `L17_matrixsimd`'s chain engine wholesale, so for the first time the two arms differ
only in the 17-point module: dense-symmetric 12.736 vs Rader 15.052, **dense wins by 18 %**
with the engine confound removed. At L=23 the same question flipped this round in the same
direction — `L23_rader` led r4 at 38.105, `L23_matrixsimd` now leads at 36.903. Two primes,
same answer, consistent with §02 §7's "Rader is not the lever at L=17". §4.2(b) — whether
the symmetric/antisymmetric convolution split adds anything on top — remains untried.

---

## 6. The single highest-value thing the next round should attack, per geometry

**L = 6 — fix the gate; do not tune. (Harness work, not implementer work.)** The cell is
currently decided by a chain check that certifies seed luck (§3.3) and by which accuracy
tier each entry happened to choose. Recalibrate `eff_tol` from a **multi-seed worst case**
across at least the six library backends — both L=6 records independently recommend exactly
this, and `L6_pfa`'s multi-seed data suggests a floor near 3.5e-9 admits the fast tier while
~1.5–2e-9 admits only the exact one; that choice is a policy decision the monitor owes the
panel *before* the round starts. Then re-score both entries at the tier the new gate
implies. Everything else at L=6 is worth ~1 %: both records agree the only lever left is a
fully split-complex chain state, and both price it as a full kernel rewrite.

**L = 8 — port `L64_blocked`'s custody layout down.** Pass-fusion is now dead at this
geometry with a measured number (+15 %) and a mechanism, and the joint (σ, phase-B perm)
sweep came back flat, so the scheduling axis is exhausted. What remains is that
`L8_radix8`'s v2 still executes **384 shuffles per step** to maintain an interleaved-ish
state. `L64_blocked` just demonstrated, at the same node, that carrying the chain state in a
split-complex custody layout across all m steps makes the steady-state map **completely
shuffle-free** and deletes even the per-step `c` deinterleave. That is the one untried
structural move at L=8 and it is now de-risked by a working implementation two geometries up.

**L = 17 — attack the 17-point module's operation count; scheduling is finished.**
`L17_matrixsimd` has now closed map placement (v3, v4, v5 all lose), extract-store (0-for-3
across r1/r2/r5) and prefetch (0-for-5). It states the remaining 1.13–1.15× deficit is "the
kernel's six-round ~1.1-FP/cyc issue wall" — an arithmetic problem, not a scheduling one.
`docs/LITERATURE.md` §4.2(b) names the untried lever: the symmetric/antisymmetric
convolution split that makes each sub-transform real-input and half cost, which §01 §8
argues is *orthogonal* to the dense-vs-Rader question this round just settled. **Secondary
concern the next round should be told about: L=17 has three entries and one engine.**
`L17_rader` now runs `L17_matrixsimd`'s chain on `L17_winograd`'s kernel. The geometry with
the largest library margin on the board has lost its structural diversity, and a
seeded-from-scratch third design would be worth more than a third tuning of the same one.

**L = 36 — kill the last of the divider.** `L36_mixedradix`'s own 4-way TSC split says the
map still costs ~19–20 µs of a ~104 µs step after the hybrid halved it, and phase 2 now runs
at ~465 cy/call against a ~200 cy floor while carrying 50.8 % of the step. The hybrid still
puts one `vsqrtpd` on the divider for every other pair. `L64_blocked` ships a **fully
all-FMA reciprocal ladder** (`MAPDIV=0`) that clears its gate with 7.4× margin, and L=36's
gate margin is 398× — there is a factor of fifty of unused accuracy budget to spend here.
Porting that ladder is a concrete cross-entry transfer worth a large fraction of the
remaining 19 µs.

**The other four, one line each.** **L=13:** `L13_direct` leads by 7.5 % with both arms
improving in lockstep; the dense-vs-Rader question is answered, so the lever is the shared
chain machinery, where `L13_rader`'s record already prices the rivals' plumbing at 4.01 µs
against our 5.40. **L=23:** the two arms are 2.4 % apart after the dense arm gained 17.8 %
in one round — re-race them before assuming the flip is stable. **L=45:** a 0.32 % tie that
inverts under median for the second consecutive round; the next round should stop
optimising and instead get a statistic that can separate them (more processes, or a
pinned governor), or accept the tie and merge effort. **L=64:** `L64_blocked` now sits *on*
its traffic roofline by its own audit (~33 GB/s of L3 movement, arithmetic hiding
underneath) — so the next move is arithmetic reduction or a smaller custody working set,
not more restructuring.

**And one harness item that outranks all eight:** wire `chain_ok` into `result["ok"]` in
`check.py` (§3.4). One line. Without it the ranking cannot see a chain failure.

---

## 7. Curation

Applying `docs/CURATION.md`. Ground 4 ("anything that beat a library baseline") is vacuous
this round — all 19 entries beat all 6 libraries at every geometry — so the decisions rest on
grounds 1–3. All 19 have current strategy records, so nothing is excluded for a missing
record, and none failed correctness.

**Ground 1 — fastest correct entry per geometry (8):** `L6_pfa`, `L8_radix8`, `L13_direct`,
`L17_matrixsimd`, `L23_matrixsimd`, `L36_mixedradix`, `L45_mixedradix`, `L64_blocked`.

**Ground 2 — structurally different runner-up, close (7):**

* `L6_unrolled` — 6.3 % behind as scored, but **ahead at both matched accuracy tiers**
  (§3.3), and a genuinely different map shape (pair-interleaved in-place vs phase-split).
  Given that the L=6 cell is a tier artifact, promoting only the winner would put the wrong
  code in front of the next panel.
* `L8_fusedaxes` — 2.6 % behind, and a real structural alternative (fused 3-axis AA2 vs
  radix-8 split phase-A/B).
* `L13_rader` — 7.5 % behind; Rader-13 CRT against a dense conj-folded matrix, the §4.2
  question at a second prime.
* `L17_winograd` — 13 % behind, but it owns the distinct artifact: the 296-instruction
  17-point cyclic+negacyclic module, which is the kernel `L17_rader` also borrows.
* `L23_rader` — 2.4 % behind, and the arms **traded places this round**; both must survive
  until the flip is confirmed.
* `L36_pencilfused` — 4.6 % behind, inside the winner's own run spread (§3.6), different map
  placement, and *ahead* of the winner in the unscored B=1 runs.
* `L45_pfa` — 0.32 % behind and ahead under median (§3.6). This is a tie; promoting one arm
  would be reading noise as a result.

**Ground 3 — instructive, kept for the record (1):** `L64_radix8`. It is 58 % behind, so it
does not qualify as a close runner-up, but it is the only structural alternative at L=64
(split-complex radix-8² per axis), it is the design `L8_radix8` should be borrowing from
next round, and leaving the geometry with a single promoted design after a round in which
that geometry moved 39 % would be the wrong reading list.

**Not promoted (3), with reasons:**

* **`L8_batchsimd`** (0.596, 4.6 % behind) — improved 23.5 %, entirely creditably, but it is
  radix-8-split like the promoted `L8_radix8` and adds no structure the promoted pair lacks.
  Near-duplicate rule.
* **`L17_rader`** (15.052, 18 % behind) — the round's 4th-biggest mover and the source of a
  genuinely valuable §4.2(a) result. Excluded anyway because its kernel is `L17_winograd`'s
  and its chain engine is `L17_matrixsimd`'s, both promoted: it is a near-duplicate of the
  union of the two. The §4.2(a) number it produced is preserved in
  `strategies/L17_rader.md`, which is tracked independently of `exemplars/`.
* **`L36_pfa`** (115.437, 11.1 % behind) — improved 10.2 %, but it is the third
  implementation of PFA 4×9 at this geometry and two better ones are promoted.

The round note for `exemplars/ice_r5/NOTES.md` should record: the fusion/residency
synthesis (§5), that the L=6 cell is void pending gate recalibration (§3.3), and that
`check.py`'s `result["ok"]` bug (§3.4) let a chain-failing backend print "ok".

PROMOTE: L6_pfa L6_unrolled L8_radix8 L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pencilfused L45_mixedradix L45_pfa L64_blocked L64_radix8
