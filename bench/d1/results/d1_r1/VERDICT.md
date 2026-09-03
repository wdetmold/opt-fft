# d1_r1 — monitor's verdict

Measured on **a80n0.lqcd.mit**, slurm job 440371, 2026-09-02T18:30:52-04:00.
CPU **Intel Xeon Gold 6326 @ 2.90 GHz (Ice Lake SP), 64 cores**, AVX-512 (incl. VBMI/VNNI/IFMA),
gcc 11.4.0, governor `schedutil`. 52 cells, 3 timing runs each, min-of-runs reported with spread.

## 0. Three corrections to the tasking, before any numbers

The brief I was given describes a different campaign in three respects, and the verdict below
follows the data, not the brief.

1. **The geometries are not L = 6, 8, 17, 36.** Those are the 3D `bench/gen`/`geom` campaign's
   cubes. This is the 1D contest: **L = 13, 31, 32, 60, 64, 128, 1021, 1024, 4096, 10007, 16384,
   65537, 100003** — thirteen sizes × four regimes = 52 cells (`cases.txt`).
2. **The scoring machine is not a Cascade Lake Xeon Gold 5218.** `environment.txt` records a
   **Xeon Gold 6326, Ice Lake SP**. That matters for §4: Ice Lake does *not* have Cascade Lake's
   severe AVX-512 licence downclock, and has 1.25 MB L2/core, not 1 MB. The developers' machine
   (wallaby, **Xeon Gold 6448Y, Sapphire Rapids**, 2 MB L2/core) is as described. The claimed
   "MKL alone spans 2.9× between those machines" is **not reproducible from anything in this
   tree** — there is no MKL measurement from wallaby here — so I do not use it as an attribution
   constant. I use a measured substitute instead (§4).
3. **There is no previous round to regress against.** Per `PANEL_BRIEF.md`, the earlier r1–r3
   was lost to a harness error and *this* r1 is the real first round. §2 says what can honestly
   be compared instead.

## 1. Headline per geometry — best stock library vs best correct panel entry

Denominator is the brief's own: the best of `fftw1d_estimate` / `fftw1d_measure` /
`fftw1d_patient` / `mkl1d_dfti` at **that cell**. Panel column excludes `d1_race`, which ships a
sibling's kernel rather than computing anything itself (§3.4). All figures are µs per transform.
`>1.00×` means the panel is faster. **T** marks a result inside the run spread — a tie, not a win.

### Non-batched (B = 1)

| L | single call (m=1): best lib | panel | ratio | chained: best lib | panel | ratio |
|---|---|---|---|---|---|---|
| 13 | fftw_est 0.0218 | d1_prime 0.0216 | **1.01 T** | mkl 0.0665 | d1_prime 0.0406 | **1.64** |
| 31 | fftw_est 0.2600 | d1_prime 0.0527 | **4.93** | fftw_meas 0.3475 | d1_prime 0.0582 | **5.97** |
| 32 | mkl 0.0224 | d1_pow2 0.0183 | **1.22** | mkl 0.1309 | d1_pow2 0.0852 | **1.54** |
| 60 | mkl 0.0612 | d1_composite 0.0690 | 0.89 | mkl 0.2371 | d1_composite 0.1357 | **1.75** |
| 64 | mkl 0.0489 | d1_pow2 0.0489 | 1.00 T | mkl 0.2369 | d1_pow2 0.1493 | **1.59** |
| 128 | mkl 0.1035 | d1_pow2 0.1232 | 0.84 | mkl 0.4757 | d1_batchlane 0.2935 | **1.62** |
| 1021 | mkl 8.277 | d1_rader 13.917 | 0.59 | mkl 12.966 | d1_rader 11.666 | **1.11** |
| 1024 | mkl 1.083 | d1_pow2 1.410 | 0.77 | mkl 4.190 | d1_pow2 3.483 | **1.20** |
| 4096 | mkl 5.994 | d1_pow2 10.216 | 0.59 | fftw_pat 19.128 | d1_pow2 14.965 | **1.28** |
| 10007 | fftw_pat 195.18 | d1_bluestein 195.01 | 1.00 T | fftw_pat 223.43 | d1_bluestein 211.68 | **1.06** |
| 16384 | mkl 32.42 | d1_pow2 50.87 | 0.64 | fftw_pat 82.66 | d1_pow2 73.48 | **1.12** |
| 65537 | fftw_pat 1464.9 | d1_rader 1202.2 | **1.22** | fftw_pat 1629.1 | d1_rader 910.4 | **1.79** |
| 100003 | fftw_pat 2689.5 | d1_bluestein 3574.4 | 0.75 | fftw_pat 3132.4 | d1_bluestein 3765.1 | 0.83 |

### Batched (B = 512, or 256/64/16/8 at the large sizes per `cases.txt`)

| L | single call (m=1): best lib | panel | ratio | chained: best lib | panel | ratio |
|---|---|---|---|---|---|---|
| 13 | fftw_meas 0.0123 | d1_prime 0.0146 | 0.84 | fftw_meas 0.0510 | d1_prime 0.0154 | **3.31** |
| 31 | fftw_pat 0.2631 | d1_prime 0.0552 | **4.77** | mkl 0.3152 | d1_prime 0.0550 | **5.73** |
| 32 | mkl 0.0153 | d1_pow2 0.0154 | 0.99 T | mkl 0.1122 | d1_batchlane 0.0607 | **1.85** |
| 60 | mkl 0.0498 | d1_composite 0.0647 | 0.77 | mkl 0.2293 | d1_composite 0.0684 | **3.35** |
| 64 | mkl 0.0383 | d1_pow2 0.0372 | 1.03 T | mkl 0.2369 | d1_batchlane 0.1258 | **1.88** |
| 128 | mkl 0.1472 | d1_pow2 0.1918 | 0.77 | mkl 0.5352 | d1_batchlane 0.2956 | **1.81** |
| 1021 | mkl 8.750 | d1_bluestein 13.106 | 0.67 | mkl 12.393 | d1_rader 11.617 | **1.07** |
| 1024 | mkl 1.677 | d1_pow2 1.843 | 0.91 | mkl 4.887 | d1_pow2 3.077 | **1.59** |
| 4096 | fftw_pat 10.865 | d1_pow2 15.877 | 0.68 | mkl 22.795 | d1_pow2 15.001 | **1.52** |
| 10007 | fftw_pat 203.30 | d1_bluestein 204.30 | 1.00 T | fftw_pat 234.76 | d1_bluestein 218.00 | **1.08** |
| 16384 | fftw_pat 47.47 | d1_pow2 76.83 | 0.62 | fftw_pat 96.58 | d1_pow2 72.66 | **1.33** |
| 65537 | fftw_pat 1518.3 | d1_rader 1549.0 | 0.98 T | fftw_pat 1762.2 | d1_rader 948.7 | **1.86** |
| 100003 | fftw_pat 2827.7 | d1_bluestein 3696.6 | 0.76 | fftw_pat 3201.8 | d1_bluestein 3890.1 | 0.82 |

### The four regime aggregates (geometric mean of ratios across the 13 sizes — never averaged across regimes)

| regime | vs stock libraries | vs stock + the genfft codelet baselines | wins (>1.05) | ties | losses |
|---|---|---|---|---|---|
| R1 non-batched, non-chained | **0.974×** | 0.946× | 3 | 3 | 7 |
| R2 batched, non-chained | **0.943×** | 0.868× | 1 | 4 | 8 |
| R3 non-batched, chained | **1.505×** | 1.439× | 12 | 0 | 1 |
| R4 batched, chained | **1.811×** | 1.344× | 12 | 0 | 1 |

**The one-sentence result of this round: the panel owns the chained regimes and does not yet own
the non-chained ones.** R3/R4 win 24 of 26 cells; R1/R2 win 4 of 26. That is exactly the shape
`PANEL_BRIEF.md` predicted ("the non-batched non-chained cell is genuinely hard and may not win
at pow2"), and the panel did not fake the cells it could not win.

Two results stand out as genuinely hard wins rather than fused-chain wins:

- **L=31, all four regimes, 4.8–6.0× over the best library.** `d1_prime`'s symmetric-pair
  real-coefficient dense DFT beats FFTW and MKL in the latency cell too, which is the cell the
  brief called the libraries' home turf.
- **L=65537 non-batched single call, 1.22× over `fftw1d_patient`** (1202 vs 1465 µs), rising to
  **1.79–1.86× in the chained cells**. `d1_rader` used the unpadded 2^16 Rader convolution the
  1D survey predicted would beat a Bluestein-defaulting library, and it did — against an FFTW
  plan that paid **58 s** of `patient` planning time to get its number.

## 2. What changed since the previous round

**Nothing, because there is no previous panel round.** `PANEL_BRIEF.md`'s restart note records
that the earlier r1–r3 was destroyed by a harness error (the impl tree was emptied mid-round and
the emptiness copied forward), so no panel entry from any prior round survives to compare
against. **No entry can have regressed, and this file is the regression baseline for r2.**

What *can* be compared is the library side, because `results/library_baseline/` holds the same 52
cells measured four hours earlier on **a81n2** — a different node of the same CPU model. Over
**244 paired library measurements**: median ratio **1.000**, 15 cells faster by >15%, 9 slower by
>15%, extremes **1.38×** (`fftw1d_estimate`, L=16384 B=64) and **0.80×** (`mkl1d_dfti`, same
cell). So the libraries did not systematically drift, but **individual cells move by up to ~1.4×
node-to-node on identical hardware and identical binaries.** That number is the noise floor for
everything in §4, and it is why I marked five cells in §1 as ties rather than wins.

## 3. Adversarial review — failures, absences, and things that are not what they look like

**No entry failed correctness.** `leaderboard.txt` contains zero `FAILED` and zero `unchecked`
verdicts across all 52 cells. Non-chained residuals are 8.7e-17 to 1.4e-15 against a 1e-12 gate —
three to four orders of margin. The chained two-part gate passes everywhere. I looked for a fast
wrong answer and there is not one in this round.

**No entry failed to build, crashed, or hung.** `build_errors.txt` contains only two
`-Wformat-truncation` warnings from `snprintf` in `d1_race.c:962/966` — no errors. There is no
`failures.txt`, and `sweep.out` contains no failure, abort, timeout or signal; its single `error`
line is a benign `slurmstepd: _is_a_lwp: open() /proc/2362390/status failed` from job teardown.
All 52 cells are present.

**Every absence from the leaderboard is an explicit, self-declared decline, not a silent skip.**
This is the check worth spelling out, because a missing entry that was quietly dropped after
producing a wrong answer would look identical in the leaderboard. It is not what happened:
entries outside their class emit `{"name":...,"L":31,"batch":1,"supported":false}` and
`leaderboard.py:44` skips exactly on that flag; a correctness failure is *shown* with `FAILED`
and merely denied a rank (`leaderboard.py:158-168`). Verified concretely at L=31 B=1 m=1:
`t_d1_composite_...json` and `t_d1_pow2_...json` both carry `"supported":false` and have no
correctness file, while `t_d1_prime_...json` carries `"supported":true` plus a full timing record
and `c_d1_prime_...json` shows `rel_l2 = 2.96e-16`. The coverage is therefore:

| entry | sizes covered (of 13) | notes |
|---|---|---|
| d1_bluestein, d1_planner, d1_race | 13/13 | full sweep |
| d1_pow2 | 6 (32/64/128/1024/4096/16384) | its class |
| d1_twiddle | 7 | its class |
| d1_batchlane | 6 (13…128) | **see below** |
| d1_rader | 4 (13/31/1021/65537) | its class |
| d1_prime | 2 (13/31) | its class |
| d1_composite | 1 (L=60) | its class, as `cases.txt` defines it |

**`d1_batchlane` is absent from every batched cell at L ≥ 1021.** This is the one absence I
consider a substantive gap rather than a clean class boundary. `d1_batchlane` exists to win the
batched regime — "SoA 8-lane zmm batch-lane engine" — and R2 is the regime the panel lost most
badly (geomean 0.943×). It declines all seven large sizes, and at the six sizes where it does run
it never wins a single R2 cell; its wins are all in R4, where the fused chain is doing the work.
Its own record concedes the mechanism: "Scalar-lane fallback for B=1 is 3–5× off the libraries".

**`d1_race` is a dispatcher, and its wins must not be counted as an independent result.** It is
the fastest panel entry in 20 of 52 cells, but its own description says it "fork-gates and races
the sibling class entries per (L,B) and ships the winner by vtable". The measurement confirms it:
in **every** cell its correctness residual is identical to the sibling it ships (L=60 B=1 m=1:
race 2.4e-16 = `d1_composite` 2.4e-16; L=16384 B=64 chained: race `ch=2.7e-14` = `d1_pow2`
`ch=2.7e-14`), and its time sits within a few percent of that sibling's — sometimes a hair below
it, which is min-of-three noise on the same code, not a speedup. It is promoted below as
infrastructure, not as a kernel.

**`d1_race`'s setup column understates its true plan cost.** `results/wisdom1d_a80n0.json` exists
and persists across the sweep, so after the first key is written the "race" is a cache lookup.
Reported setups are 0.003–0.024 s. That is a warm-wisdom number, and it is not comparable to
`fftw1d_patient`'s honestly-paid 58 s at L=65537 or 67 s at L=16384. None of the §1 ratios depend
on setup time, but a future round that scores plan cost must not read this column as the race's
price.

**The three "adoption-scored library layers" scored essentially zero adoption.** `d1_planner` is
the fastest panel entry in **0/52** cells; `d1_twiddle` in **0/52**. Grepping `impl/` for
cross-entry use: `d1tw_*` appears in `d1_twiddle.c` and nowhere else (0 uses in all eight
siblings); `d1_planner` symbols appear only in `d1_planner.c` and `d1_race.c`; `gr_*` appears
only in `d1_race.c`. The layers were built but not consumed. That is the finding, and it is why
two of them are not promoted.

**One gate is closer to saturation than it looks.** At **L=1024, B=1, m=4000**, `d1_bluestein`'s
chain residual is `3.0e-11` against a `1e-10` gate — **30% of the gate**, the tightest margin in
the round. `fftw1d_measure` (1.4e-11), `d1_twiddle` (1.3e-11) and even `baseline_dft` (1.3e-11)
sit in the same band, so this is chain-length accumulation at that cell rather than one entry
misbehaving. Nobody failed. But that cell has ~3× of headroom where every other cell has 10–10⁴×,
so if r2 lengthens the chain at L=1024 the gate will start rejecting *correct* implementations.
Raise m elsewhere or re-derive the anchor before that happens.

## 4. Claimed numbers versus measured — and what the machine explains

Absolute times came in **1.5–1.9× slower than the implementers' own records** almost across the
board, while the *ratios against the libraries* mostly survived. That combination is the
signature of a clock/machine difference rather than an error, and it is what I attribute it to.

| entry | claimed (own record, on wallaby) | measured (a80n0) | factor | verdict |
|---|---|---|---|---|
| d1_rader | 65537 B=1 m=1: **703 µs** | 1202 µs | 1.71× | machine + wallaby noise; **the win survived** (1.22× over FFTW) |
| d1_rader | 1021 B=1 m=1: **7.43 µs** | 13.92 µs | 1.87× | machine; **the win did not survive** — MKL takes the cell at 8.28 µs |
| d1_bluestein | 10007 B=1: **~110–122 µs**, "win ~1.9–2×" | 195.0 µs | 1.6–1.8× | machine; the claimed 2× win **collapsed to a dead tie** with `fftw1d_patient` (195.2) |
| d1_bluestein | 16384 direct: **50.5 µs** | 107.6 µs | 2.1× | largest single gap in the round |
| d1_prime | 31 B=1 m=1: 0.034 µs, "4.9× win" | 0.0527 µs, **4.93× win** | 1.55× on time, **1.00× on ratio** | textbook machine scaling; claim fully vindicated |
| d1_composite | 60 B=1 m=1: 0.045–0.046 µs | 0.0690 µs | 1.5× | machine |
| d1_composite | "B=512 m=1: parity with MKL" | 0.77× of MKL | — | **claim not vindicated**; the batched latency cell is a loss, not parity |
| d1_pow2 | 4096 B=1: "~0.9× of FFTW patient" | 0.66× of FFTW patient | — | worse than claimed even allowing for the clock |
| d1_planner | 16384 B=1: 80–160 µs | 116 µs | — | in range |

**The attribution, stated carefully.** Development is on **wallaby, Xeon Gold 6448Y (Sapphire
Rapids)**: newer core, 2 MB L2 per core, higher sustained AVX-512 clocks. Scoring is on **a80n0,
Xeon Gold 6326 (Ice Lake SP)**, 2.90 GHz base, 1.25 MB L2 per core. A 1.5–1.9× gap is more than
that architectural difference alone should buy — but the records themselves supply the rest, and
they are candid about it: `d1_composite` measured wallaby swinging **2.7–4.1 GHz between runs**
and calls a "3× apparent regression" mid-session "pure" frequency noise; `d1_planner` records
wallaby cores differing **2× in effective speed**; `d1_bluestein` worked while the node was at
"load 25–30, 51 users"; `d1_rader` saw *the same binary* read 703 µs in a quiet window and
1300–1600 µs an hour later. Implementers reported best-case minima from a contended shared login
node. The scoring node is exclusive. **The honest reading is: architecture plus contention, with
contention probably the larger term, and the panel's ratio claims holding up far better than its
absolute ones.** I cannot decompose it further, because — contrary to the tasking — there is no
cross-machine library measurement in this tree to calibrate against. What I *can* offer as a
floor is §2's number: **1.4× cell-to-cell variation was observed between two nodes of the same
model running the same library binaries**, so any claimed-vs-measured gap below ~1.4× is not
evidence of anything.

## 5. Which open question this round moved

`docs/LITERATURE.md` §4 is written for the 3D campaign's cubes (L = 6, 8, 17, 36), so most of its
questions are not addressable from a 1D round. Two are, and one is moved decisively.

### §4.2 — "L = 17: dense-symmetric, Rader, or Winograd?" — moved, toward dense-symmetric, and extended past the corpus's own range

§4.2(a) asks which of dense-symmetric and Rader wins on this hardware, batch-vectorised. §02 §7
says "Rader is not the lever"; §03 §6.4 and §06 §6.4a say direct O(n²) is preferable up to
n ≈ 20. This round ran both, as independent implementations, on the same node, at the two prime
sizes bracketing 17:

| cell | `d1_prime` (dense symmetric-pair) | `d1_rader` (Rader conv) | dense wins by |
|---|---|---|---|
| L=13 B=1 m=1 | 0.0216 µs | 0.1128 µs | **5.22×** |
| L=13 B=512 m=1 | 0.0146 µs | 0.1151 µs | **7.88×** |
| L=31 B=1 m=1 | 0.0527 µs | 0.3861 µs | **7.33×** |
| L=31 B=512 m=1 | 0.0552 µs | 0.3907 µs | **7.08×** |

Dense-symmetric wins all eight small-prime cells by **3.4–7.9×**. The corpus's position is
confirmed and **extended from n ≲ 20 to n = 31** — McFarlin's crossover is not where the useful
boundary sits on Ice Lake. And the round *locates* the crossover from the other side: Rader beats
Bluestein by **2.2–2.9× at L=65537** (where N−1 = 2¹⁶ makes the convolution a clean power of two)
and by only 1.00–1.23× at L=1021. **The dense→Rader crossover for a prime DFT on this hardware
lies between L=31 and L=1021, and Rader's advantage is governed by the factorisation of N−1, not
by N** — exactly the prediction in `PANEL_BRIEF.md`, now with numbers. This is the round's most
transferable result, and it bears directly on how L=17 should be built in the 3D campaign.

### §4.6 — "model versus search for the instruction schedule" — supported, cheaply

§06 argues the schedule is "the primary thing to search" at sizes that are not powers of 4.
`d1_pow2`'s record measures the spread directly: mixed radix schedule vs pure radix-4 at 4096 is
**6.77 vs 8.09 µs**, and vs greedy radix-8 at 1024 is **0.896 vs 1.334 µs** — a **1.49× swing
from schedule choice alone**, at fixed algorithm and fixed compiler. `d1_twiddle` adds a second
datum at L=60: radix order [4,3,5] at ~0.14 µs versus [3,5,4] at 0.20 µs, a **1.43× spread**.
Search is worth its minutes; §01's "you should not need a search phase" does not hold here.

### A third result the corpus asked for but §4 does not itemise

The 1D survey names across-batch split-complex lane vectorisation "the top under-used lever", and
this round contains an unusually clean controlled measurement of it. `fftw1d_custom` and
`fftw1d_custom_soa` are **the same genfft monolithic codelet on the same split arrays**, differing
only in whether the datatype is `double` (scalar DAG + autovectorisation) or `v8` (8 transforms
per vector lane) — one `-DCUSTOM_SOA` flag (`sota/fftw1d_custom.c:9-10`). Across the 12 batched
cells where both ran, the lane layout is worth **1.07–2.29×, geometric mean 1.47×**, and the gain
is systematically larger in the chained cells (1.56–2.29×) than the single-call ones (1.07–1.25×).
This is a same-generator, same-layout, single-variable A/B of the kind §4.4 complains has never
been published, and it isolates *lane assignment* rather than split-vs-interleaved storage.

## 6. The single highest-value thing r2 should attack, per geometry

Ordered by how much is on the table.

| L | attack | why this one |
|---|---|---|
| **100003** | one-level **nested Rader** (N−1 = 2·3·7·2381, and 2381−1 = 2²·5·7·17 is smooth) | the **only size losing all four cells** (0.75/0.76/0.83/0.82). It is also the size the survey explicitly flags as where FFTW's planner bails to Bluestein — so the structural opening is real and untried. `d1_planner`'s record already names it "the weak headline cell". |
| **1021** | make Rader's *non-chained* path competitive: the claimed 7.43 µs became 13.92 µs | the largest claimed-vs-measured collapse in the round, and the difference between winning and losing the cell (MKL 8.28 µs). Establish first whether the 1.87× is machine or a wallaby-only artifact — measure on the scoring node before touching code. |
| **16384, 4096** | the **four-step / six-step 128×128 decomposition** with L1-resident sub-FFTs | the worst R2 cells in the round (0.62×, 0.68×). `d1_pow2` never implemented it and its own record lists it as the missing move; at these sizes the batched working set (16–32 MB) is past L2, so this is a traffic problem, not a port-dispatch problem. |
| **128, 60, 1024** | the **batched non-chained (R2)** cell specifically | all three lose R2 (0.77/0.77/0.91) while winning R4 by 1.6–3.4×. The chain fusion is carrying entries whose single-call batched kernel is not competitive; the fixed per-call cost is the target. `d1_batchlane` owns this and currently wins zero R2 cells. |
| **32, 64** | hold, and convert the R1/R2 ties into wins | already at 0.99–1.22× of MKL with 13–30% spread. These are the cheapest remaining R1/R2 wins on the board, and they need tighter measurement (more runs) as much as more code. |
| **13** | a hand-rolled **Rader-13 / Winograd-13** as `d1_prime`'s own record proposes | the only size where dense-symmetric merely ties in R1 (1.01×) and loses R2 (0.84×) to the library codelets. |
| **65537, 10007, 31** | **defend** | 31 wins all four cells by 4.8–6.0×; 65537 wins three of four including the hard R1 cell; 10007 ties R1/R2 and wins R3/R4. Spend r2's effort elsewhere and re-measure these to confirm they hold. |

**Cross-cutting, and above all of the above: R2 is the regime the panel lost (geomean 0.943×,
one win in thirteen).** Every per-size row above that mentions R2 is the same underlying problem —
a batched single-call kernel whose fixed cost is not amortised by a chain. If r2 fixes one thing,
fix that; the §5 batch-lane A/B says the layout is worth 1.47× and the panel is not yet
collecting it outside the chained cells.

## 7. What to keep, and why

Applying `docs/CURATION.md`'s four grounds.

**Rule 1 — the fastest correct entry per geometry** yields six distinct kernels, since no single
entry wins everywhere: `d1_prime` (13, 31), `d1_pow2` (32, 64, 128, 1024, 4096, 16384),
`d1_composite` (60), `d1_rader` (1021, 65537), `d1_bluestein` (10007, 100003), and `d1_batchlane`,
which takes the R4 cells at 32, 64 and 128 and the R3 cell at 128 outright.

**Rule 2 — a structurally different runner-up that came close** is satisfied twice over by that
same set, and deliberately so. `d1_batchlane` (across-batch SoA lanes) against `d1_pow2`
(in-register Stockham) at 32/64/128 is precisely the "keep both, the next panel needs to see the
alternative written down" case — they trade the cell depending on regime. `d1_rader` against
`d1_bluestein` at 1021/65537/10007 is the second: they are within 1.00–1.23× at 1021 and 10007
and 2.2–2.9× apart at 65537, which is the crossover measurement of §5 and is only legible if both
sources survive.

**Rule 4 — anything that beat a library** adds nothing new; all six already did.

**`d1_race` is promoted on separate grounds: infrastructure, not speed.** It computes nothing of
its own (§3.4) and I have not counted any of its 20 cells as a panel result. But it is the
mechanism by which the per-(L,B) winner actually ships, its record is substantial, and the brief
scores it for adoption. Keep it so r2 inherits the race-and-wisdom machinery as code.

**`d1_planner` and `d1_twiddle` are held back.** Both are library layers; both are the fastest
panel entry in **0/52** cells; and neither is used by any sibling — `d1tw_*` has zero uses outside
`d1_twiddle.c`, and `d1_planner` zero outside itself and `d1_race`. `CURATION.md` warns against
diluting the reading list, and an unadopted layer that wins nothing is dilution. This is *not* a
correctness judgement — both are correct in all 52 and 28 cells respectively — and neither
qualifies as rule 3's "instructive failure", because what their records document (exact twiddle
accuracy at 1.24 × 2⁻⁵³, a radix-order A/B) is preserved in `impl_1/` and cited in §5 above
regardless. If r2 adopts either layer, promote it then. `baseline_dft` is the harness floor, not
a competitor, and is tracked separately by `CURATION.md`.

Seven entries, six kernels plus the dispatcher, covering all thirteen geometries and both sides
of the two structural questions this round opened.

PROMOTE: d1_prime d1_pow2 d1_batchlane d1_composite d1_rader d1_bluestein d1_race
