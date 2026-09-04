# Round d1_r6 — monitor's verdict

Scored on the exclusive benchmark nodes **a80n0** and **a81n2** (slurm 440424 / 440513),
both `Intel Xeon Gold 6326 @ 2.90 GHz`, 64 cores, AVX-512 (**Ice Lake-SP**, not Cascade
Lake — see §0), gcc 11.4.0, governor `schedutil`.
Ranking statistic: median of per-process medians over 9 runs (m=1) or 3 runs (chained);
`?` marks a gap inside 2σ of the combined standard error of the two medians.

---

## 0. Two corrections to the monitor brief, before any numbers

**The geometries are not L = 6, 8, 17, 36.** Those are the *3D* panel's cubes
(`bench/geom`). This is the **1D** panel (`bench/d1`), whose brief specifies thirteen
lengths — 13, 31, 32, 60, 64, 128 (the original six), 1024, 4096, 16384 (added mid-r1),
and the four contrast primes 1021, 10007, 65537, 100003 — each in **four** regimes
(B=1/batched × m=1/chained), 52 cells. This verdict reports all thirteen.

**The scoring machine is not a Cascade Lake Xeon Gold 5218 with 1 MB L2.** Both grading
nodes are Xeon Gold 6326 (Ice Lake-SP, 1.25 MB L2/core). The 5218 in the brief comes from
`docs/LITERATURE.md` §4.8 item 6, which quotes Intel's turbo table for that part; the same
item says there is *no* primary measurement in the corpus for Ice Lake-SP or later server
parts. That matters for §4 and §5 below, because it means the "downclocked AVX-512"
premise had to be re-measured this round rather than assumed — and it was.

**The implementers did not develop on Sapphire Rapids this round.** Every strategy record
states its numbers were taken *on the scoring node itself* (`a80n0`, leased core, via the
r6 reservation), with wallaby/SPR quoted only as a secondary data point. So where a claimed
number differs from the board, the machine difference is **not** the available explanation;
§4 gives the ones that are.

---

## 1. Headline per geometry — fastest correct panel entry vs best stock library

"Best stock library" = best correct of `mkl1d_dfti`, `fftw1d_{estimate,measure,patient}`.
`fftw1d_custom` / `fftw1d_custom_soa` (our own genfft-codelet harnesses) are excluded from
the library column and noted separately where they beat stock. Ratio > 1 = panel wins.

| L | batch | chain | fastest correct panel entry (µs/xf) | best stock library | r6 ratio | r5 ratio |
|---|---|---|---|---|---|---|
| **13** | B=1 | m=1 | `d1_rader` 0.0163 | `mkl` 0.0224 | **1.37×** | 1.53× |
| **13** | B=1 | m=200000 | `d1_rader` 0.0337 | `fftw-measure` 0.0674 | **2.00×** | 1.97× |
| **13** | B=512 | m=1 | `d1_rader` 0.0095 | `fftw-measure` 0.0139 | **1.46×** | 1.52× |
| **13** | B=512 | m=2000 | `d1_prime` 0.0137 | `fftw-measure` 0.0513 | **3.74×** | 3.30× |
| **31** | B=1 | m=1 | `d1_batchlane` 0.0478 | `fftw-measure` 0.2633 | **5.51×** | 4.32× |
| **31** | B=1 | m=100000 | `d1_batchlane` 0.0520 | `fftw-measure` 0.3492 | **6.72×** | 6.83× |
| **31** | B=512 | m=1 | `d1_prime` 0.0480 | `mkl` 0.2267 | **4.72×** | 4.69× |
| **31** | B=512 | m=1200 | `d1_race` 0.0453 | `mkl` 0.3153 | **6.96×** | 6.87× |
| **32** | B=1 | m=1 | `d1_twiddle` 0.0192 | `mkl` 0.0263 | **1.37×** | 1.33× |
| **32** | B=1 | m=100000 | `d1_race` 0.0573 | `mkl` 0.1311 | **2.29×** | 2.19× |
| **32** | B=512 | m=1 | `d1_twiddle` 0.0174 | `mkl` 0.0153 | 0.88× ✗ | 1.13× |
| **32** | B=512 | m=1000 | `d1_race` 0.0331 | `mkl` 0.1122 | **3.39×** | 3.05× |
| **60** | B=1 | m=1 | `d1_batchlane` 0.0422 | `mkl` 0.0620 | **1.47×** | 1.44× |
| **60** | B=1 | m=60000 | `d1_composite` 0.0828 | `mkl` 0.2371 | **2.86×** | 1.93× |
| **60** | B=512 | m=1 | `d1_planner` 0.0504 | `mkl` 0.0437 | 0.87× ✗ | 0.97× |
| **60** | B=512 | m=600 | `d1_race` 0.0548 | `mkl` 0.2293 | **4.18×** | 3.85× |
| **64** | B=1 | m=1 | `d1_race` 0.0442 | `mkl` 0.0473 | **1.07×** | 0.98× |
| **64** | B=1 | m=60000 | `d1_race` 0.0801 | `fftw-estimate` 0.2680 | **3.35×** | 2.79× |
| **64** | B=512 | m=1 | `d1_batchlane` 0.0370 | `mkl` 0.0421 | **1.14×** | 1.10× |
| **64** | B=512 | m=500 | `d1_race` 0.0650 | `mkl` 0.2385 | **3.67×** | 3.07× |
| **128** | B=1 | m=1 | `d1_batchlane` 0.1107 | `mkl` 0.1042 | 0.94× ✗ | 0.95× |
| **128** | B=1 | m=30000 | `d1_batchlane` 0.1537 | `mkl` 0.5414 | **3.52×** | 2.94× |
| **128** | B=512 | m=1 | `d1_race` 0.1521 | `mkl` 0.1440 | 0.95× ✗ | 0.95× |
| **128** | B=512 | m=250 | `d1_pow2` 0.1874 | `mkl` 0.5381 | **2.87×** | 2.73× |
| **1021** | B=1 | m=1 | `d1_planner` 6.707 | `mkl` 8.297 | **1.24×** | 1.26× |
| **1021** | B=1 | m=2000 | `d1_race` 7.128 | `mkl` 12.981 | **1.82×** | 1.83× |
| **1021** | B=256 | m=1 | `d1_race` 9.036 | `mkl` 8.745 | 0.97× ✗ | 1.04× |
| **1021** | B=256 | m=400 | `d1_planner` 6.627 | `mkl` 12.394 | **1.87×** | 1.86× |
| **1024** | B=1 | m=1 | `d1_twiddle` 1.277 | `mkl` 1.083 | 0.85× ✗ | 0.85× |
| **1024** | B=1 | m=4000 | — | — | **cell lost** | — |
| **1024** | B=512 | m=1 | `d1_twiddle` 1.782 | `fftw-patient` 1.871 | **1.05×** | 1.09× |
| **1024** | B=512 | m=2000 | `d1_planner` 2.352 | `mkl` 4.913 | **2.09×** | 2.10× |
| **4096** | B=1 | m=1 | `d1_race` 7.006 | `mkl` 5.995 | 0.86× ✗ | 0.84× |
| **4096** | B=1 | m=1000 | `d1_race` 9.771 | `fftw-patient` 19.207 | **1.97×** | 1.96× |
| **4096** | B=256 | m=1 | `d1_race` 9.745 | `fftw-patient` 11.082 | **1.14×** | 1.02× |
| **4096** | B=256 | m=400 | `d1_pow2` 10.124 | `mkl` 22.623 | **2.23×** | 2.22× |
| **10007** | B=1 | m=1 | `d1_planner` 107.8 | `fftw-patient` 201.0 | **1.87×** | 1.81× |
| **10007** | B=1 | m=400 | `d1_race` 122.4 | `fftw-patient` 230.9 | **1.89×** | 1.88× |
| **10007** | B=64 | m=1 | `d1_race` 135.2 | `fftw-patient` 206.8 | **1.53×** | 1.84× |
| **10007** | B=64 | m=80 | `d1_planner` 125.4 | `fftw-patient` 239.9 | **1.91×** | 1.82× |
| **16384** | B=1 | m=1 | `d1_pow2` 32.952 | `fftw-patient` 32.871 | 1.00×? (tie) | 0.96× |
| **16384** | B=1 | m=250 | `d1_pow2` 49.263 | `fftw-patient` 82.181 | **1.67×** | 1.63× |
| **16384** | B=64 | m=1 | `d1_pow2` 44.196 | `fftw-patient` 47.223 | **1.07×** | 1.04× |
| **16384** | B=64 | m=150 | `d1_pow2` 49.121 | `fftw-patient` 102.2 | **2.08×** | 1.93× |
| **65537** | B=1 | m=1 | `d1_race` 753.0 | `fftw-patient` 1468.5 | **1.95×** | 1.95× |
| **65537** | B=1 | m=60 | `d1_rader` 680.8 | `fftw-patient` 1629.6 | **2.39×** | 2.38× |
| **65537** | B=16 | m=1 | `d1_planner` 1020.0 | `fftw-patient` 1539.4 | **1.51×** | 1.65× |
| **65537** | B=16 | m=20 | — | — | **cell lost** | — |
| **100003** | B=1 | m=1 | `d1_race` 2222.4 | `fftw-patient` 2713.1 | **1.22×** | 1.06× |
| **100003** | B=1 | m=40 | `d1_bluestein` 2320.6 | `fftw-patient` 3129.0 | **1.35×** | 1.35× |
| **100003** | B=8 | m=1 | `d1_race` 2399.2 | `fftw-patient` 2849.9 | **1.19×** | 1.17× |
| **100003** | B=8 | m=15 | `d1_bluestein` 2752.0 | `fftw-patient` 3220.5 | **1.17×** | 1.16× |

**Per-regime geometric means** (the brief forbids a single aggregate number):

| regime | geomean vs best stock library | range | sizes |
|---|---|---|---|
| non-batched, single call (B=1, m=1) | **1.37×** | 0.85× – 5.51× | 13 |
| batched, single call (m=1) | **1.26×** | 0.87× – 4.72× | 13 |
| non-batched, chained | **2.41×** | 1.35× – 6.72× | 12 |
| batched, chained | **2.71×** | 1.17× – 6.96× | 12 |

**The shape of the round in one sentence: every chained cell is won, and every loss is a
single-call cell.** 50 of 52 cells were measured; of those, 42 are panel wins, 1 is a
statistical tie (16384 B=1 m=1, 1.00×?), and **7 are losses — all of them m=1**:
32 B=512 (0.88×), 60 B=512 (0.87×), 128 B=1 (0.94×), 128 B=512 (0.95×), 1021 B=256
(0.97×), 1024 B=1 (0.85×), 4096 B=1 (0.86×).

Those 7 are not one phenomenon. Splitting them by whether the *best* observed run also
loses separates genuine deficits from variance losses:

| cell | panel med / best | library med / best | median ratio | **best-vs-best ratio** | reading |
|---|---|---|---|---|---|
| 4096 B=1 m=1 | 7.006 / 6.958 (18%) | 5.995 / 5.976 (14%) | 0.86× | **0.86×** | **genuine deficit**, stable both sides |
| 32 B=512 m=1 | 0.0174 / 0.0173 (0.5%) | 0.0153 / 0.0153 (14%) | 0.88× | **0.88×** | **genuine deficit** for the stable entry |
| 128 B=1 m=1 | 0.1107 / 0.1102 (2.7%) | 0.1042 / 0.0911 (22%) | 0.94× | **0.83×** | **genuine deficit**; MKL has a fast mode we have no answer to |
| 1024 B=1 m=1 | 1.277 / 1.123 (16%) | 1.083 / 1.075 (15%) | 0.85× | 0.96× | mostly variance (turbo licence, §4) |
| 128 B=512 m=1 | 0.1521 / 0.1479 (16%) | 0.1440 / 0.1426 (11%) | 0.95× | 0.96× | near-parity |
| 60 B=512 m=1 | 0.0504 / 0.0442 (20%) | 0.0437 / 0.0432 (15%) | 0.87× | 0.98× | near-parity on bests, lost on median |
| 1021 B=256 m=1 | 9.036 / 7.272 (33%) | 8.745 / 8.734 (16%) | 0.97× | **1.20×** | **variance loss** — the code is 1.2× faster, the median is not |

So there are exactly **three genuinely slower cells** (4096 B=1, 32 B=512, 128 B=1) and
four cells lost to run-to-run dispersion. That distinction drives §6.

---

## 2. What changed since d1_r5, per geometry

Round-over-round is on *panel-best median*, so a change can come from a new entry taking
the cell as well as from an entry improving.

**Improved materially (>5%):**

| L | cell | r5 → r6 | driver |
|---|---|---|---|
| 60 | B=1 chain | `d1_race` 0.1253 → `d1_composite` **0.0828** (**1.51×**) | composite's chain-step v5: `vbroadcastf64x2` builds the stage-A pairs with zero cross-lane shuffles, collapsing the 30-ymm live set to 15 zmm and deleting 15 `vinsertf64x4` p5 uops from the critical path. Cell margin vs MKL 1.93× → 2.86×. Biggest single-cell gain of the round, and bit-identical output to v4. |
| 100003 | B=1 m=1 | `d1_bluestein` 2563.9 → `d1_race` **2222.4** (1.15×) | 1.06× → **1.22×** vs FFTW-patient |
| 64 | B=512 chain | 0.0775 → **0.0650** (1.19×) | batchlane/prime latency-shaped map |
| 31 | B=1 m=1 | 0.0536 → **0.0478** (1.12×) | `d1_batchlane` takes the cell |
| 1021 | B=1 m=1 | `d1_rader` 7.490 → `d1_planner` **6.707** (1.12×) | planner's radix-16/64 tile schedules |
| 13 | B=512 chain | 0.0155 → **0.0137** (1.13×) | prime's Goldschmidt/early-rcp map |
| 32/60/64/128 | batched chain | +5–11% across the board | the same map, ported into five entries |

**Regressed.** Four cells went backwards on the scored median; two of them flipped from a
win to a loss:

1. **L=32, B=512, m=1 — 1.13× win → 0.88× loss. The worst regression of the round.**
   In r5 `d1_planner` held this cell at a **median** of 0.0154. In r6 planner's *best* is
   unchanged (0.0153) but its median is 0.0175, and the panel's most stable entry there
   (`d1_twiddle`, 0.0174 at 0.5 % spread) is genuinely 14 % slower than MKL. MKL
   simultaneously improved 0.0174 → 0.0153. Net: a cell the panel owned is now MKL's.
2. **L=1021, B=256, m=1 — 1.04× win → 0.97× loss.** `d1_race`'s best run (7.272) would
   win the cell by 1.20×; its median (9.036, **33.4 % spread**) loses it. Nothing got
   slower; the dispersion got wider.
3. **L=10007, B=64, m=1 — 1.84× → 1.53×.** `d1_race` 113.1 → 135.2 median, best 109.8.
   Same shape: the fast mode is intact, the median moved.
4. **L=65537, B=16, m=1 — 1.65× → 1.51×.** r5's `d1_rader` 933.9 → r6's `d1_planner`
   1020.0 (best 936.5); `d1_rader` itself went 933.9 → 1022.5. Note that d1_rader's r6
   record claims a −3.1 % *improvement* at this cell from its buffer-stagger change,
   A/B'd against a leased-core baseline of 993 — a baseline that never matched the r5
   board number of 933.9 in the first place.

Everything else moved by ≤ 2 %, i.e. inside the round's noise (13 B=512 m=1 0.979×,
31 B=1 chain 0.981×, 128 B=1/B=512 m=1 0.996×/0.983×, 1024 B=512 m=1 0.979×).

**Three of the four regressions are median-not-best regressions.** Two of them
(1021 B=256, 65537 B=16) are on cells that §3 shows were measured on *both* grading nodes.

---

## 3. Adversarial review: failures, crashes, missing entries, and evidence quality

### 3.1 Nothing failed correctness, and the check can discriminate

All 9 panel entries built (`build_errors_{1,2}.txt` are byte-identical and contain
**only warnings** — `-Waggressive-loop-optimizations` on `d1_batchlane.c:184/195` and
`-Wformat-truncation` on three `snprintf`s in `d1_race.c`; no errors). All 9 appear in the
standings. `check_{1,2}.log` contain **1253 PASS and 0 FAIL** (the remaining 10 lines are the two
`FileNotFoundError` tracebacks of §3.3 — the checker died loudly on the missing input
rather than passing an absent file, which is the right failure mode).

A clean sheet is exactly the shape that should be distrusted, so: *does this check
discriminate?* Two arguments that it does. (a) The reported `rel_l2` is **not constant
across backends** — it tracks algorithm, ordering as error-growth theory predicts:
split-complex Stockham entries sit at 1.4–2.1e-16, Rader/Bluestein at 2.5–3.4e-16, and the
dense O(L²) `baseline_dft` floor at 3.2–4.9e-16, at every L. A stub returning garbage, or
a check reading a stale file, could not produce that ordering. (b) The harness checks both
a `one_` single-step output and the timed `out_` file, and the chained gate is a
self-calibrating chaotic chain, so a no-op chain fails by construction. What is **missing**
is a deliberately-wrong positive control in the round's own artifacts; the discrimination
argument above is inferential, not a two-sided control. Recommend r7 add one broken entry.

### 3.2 One accuracy hypothesis raised, tested, and killed

Five entries adopted `d1_prime`'s "latency-shaped map" (Goldschmidt sqrt + early-seeded
reciprocal) this round, and chain accuracy at several cells got 7–17× worse
(`d1_race`/`d1_prime` at 31 B=512 chain: 1.3e-12 → 1.1e-11; `d1_rader` at 13 B=512 chain:
3.5e-14 → 5.8e-13). That looked like the round's fast-but-less-right finding.

**It is not.** The two-sided control kills it: at the *same* cells, `mkl1d_dfti` moved
8.4e-13 → 5.9e-12 (7.0×), FFTW ×3 moved 1.1e-14 → 8.2e-14 (7.5×), and `baseline_dft` moved
9.3e-14 → 9.3e-13 (10×). The libraries do not contain our map. The shift is the chaotic
chain's sensitivity to this round's regenerated input seed, not a panel regression. Every
entry retains ≥ 1 decade of margin to its gate. *Residual:* `d1_rader` at 13 B=512 chain
(5.8e-13) is ~20× looser than its rivals in that cell in absolute terms and moved by more
than the co-moving baseline; worth one line in r7's record, not a kill.

### 3.3 **Two cells were destroyed by a harness bug, not by any implementation**

`failures_1.txt` and `failures_2.txt` list 49 `exited 2` entries. They are **not**
implementation crashes. Every backend in each affected cell died — including MKL and all
three FFTW planner levels — and `timing_{1,2}.err` gives the cause verbatim:

```
results/d1_r6/in_L1024_B1.bin:  No such file or directory     (37×, shard 1)
results/d1_r6/in_L65537_B16.bin: No such file or directory    (30×, shard 2)
```

The mechanism, confirmed in the harness: `sweep.sh:212` names the input
`$OUT/in_L${L}_B${B}.bin` — a name with **no shard component** — and `sweep.sh:270`
`rm -f`s it when a cell finishes. r6 is the **first round to grade on two nodes**
(r5 lost its second node to the `reserve.sh --extra` bug fixed in 1dddd3db), and the shard
split was not disjoint: comparing `sweep_1.out` and `sweep_2.out`, **11 of the 26 (L,B)
pairs ran on both nodes simultaneously**, into the same shared output directory. Each
shard deleted the other's input mid-cell.

Casualties, both chained cells, both silently absent from the board:
* **L=1024, B=1, chain m=4000** — only `baseline_dft` survived, `unchecked`.
* **L=65537, B=16, chain m=20** — only `d1_bluestein` survived, `unchecked`, 2 runs.

**Neither of those two surviving numbers may be quoted.** They are unchecked, unranked
(`--` in the ratio column), and in the 65537 case a 2-run median. In particular
`d1_bluestein 1835.7 µs` at 65537 B=16 chain is **not** a result; r5's number for that cell
was `d1_rader` 744.6, so quoting 1835.7 would misread a 2.5× *regression* into the record.

**This is the round's most serious defect and it is in the apparatus, not the entries.**
Fix before r7: put the shard id (or the pid) in the input filename, or make the shard split
disjoint by (L,B), or both.

### 3.4 The same bug has a quieter second consequence: overwritten measurements

The per-run result files are named `t_<backend>_L<L>_B<B>_m<M>_r<N>.json` — also with no
shard component. For the 11 (L,B) pairs that ran on both nodes, both shards wrote `r1…r9`
into the *same* filenames, so the leaderboard's "9r" for those cells is **whichever node
wrote each slot last**, not nine draws from one machine. The two nodes are the same CPU
model on exclusive allocations, so this does not bias the level much — but it silently
mixes two machines' page-colouring and, for one entry, two different tuning states:

**`d1_race` keeps per-host wisdom, and a81n2's wisdom was cold.** `results/wisdom1d_a80n0.json`
is 131 KB and carries five rounds of accumulated verdicts; `results/wisdom1d_a81n2.json` is
**new this round and 23.8 KB**. So on the second node d1_race raced from an empty cache.
Two of the four median regressions in §2 — **1021 B=256 and 65537 B=16** — are on cells in
the overlapping set, which makes cold-wisdom draws the leading explanation for both. This
is a confound on `d1_race`'s (and to a lesser extent `d1_planner`'s) numbers specifically;
`d1_prime`'s placement probe is per-process and is unaffected.

### 3.5 Entries that are absent from cells — by design, verified

`timing_*.err` is dominated by lines of the form `d1_rader: does not support L=32`. These
are the brief's declared-class opt-outs (`d1_pow2` owns 2^k, `d1_prime`/`d1_rader` the small
primes, `d1_composite` 60, and so on), not failures: the brief says explicitly *"win where
you can win; do not fake the cells you cannot."* Coverage is `d1_bluestein`, `d1_planner`,
`d1_race` at all 13 lengths; the class entries where they claim a class. Nobody is missing
from a cell they claimed.

`fftw1d_custom_soa: fft1d_create failed for L=<n> batch=1` (78 lines) is likewise by
design — the SoA 8-transform genfft harness has no B=1 form.

### 3.6 `d1_race` is legitimate, with two caveats that must travel with its numbers

I checked the obvious cheat. **No panel source contains `DftiCreate`, `fftw_plan`, or any
library header** — the whole panel is from-scratch, as `fft-project-approach` requires.
`d1_race` does `system()` a gcc invocation and `dlopen` the result at plan time, but the
arm table (`d1_race.c:972–979`) is exactly the eight *sibling panel entries*, compiled from
`impl/*.c` with `-lm`. That is the declared adoption-scored library layer, not a library.

Two caveats:
* **`d1_race`'s copy of a rival kernel is not the rival's binary.** It builds arms
  `-shared -fPIC -Wl,-Bsymbolic -fno-semantic-interposition`; its own source comment admits
  "the steady node-only in-process gaps at 32/64 B=512 are the remaining suspects." This is
  visibly two-sided on the board: at 32 B=1 chain race ships 0.0573 against `d1_batchlane`'s
  own 0.0674 for what should be the same kernel, and at 32 B=512 race reads 0.0176 against
  `d1_planner`'s best of 0.0153. Race's number is evidence about *race*, not about the arm.
* **Setup cost.** `d1_race` reports setup up to **2.8 s** (64 B=512 chain), 2.5 s, 2.3 s,
  1.6 s at other chained cells, versus MKL's 0.002 s. The brief scores timing after
  compile+warmup and reports setup separately, so this is within the rules — but no
  deployment absorbs a 2.8 s plan, and the record should say so.

**Nothing this round was fast and wrong. The thing that had to be caught was fast and
*deleted* (§3.3), and a number that would have been quoted as a result but is unchecked
(§3.3).**

---

## 4. Claimed numbers versus measured

Because implementers measured *on the grading node* (§0), machine difference is not
available as the explanation. Three mechanisms cover every gap, and the implementers
themselves supplied the evidence for all three.

**(a) Min-on-a-leased-core versus median-of-nine-processes.** This is by far the most
common. In every one of these the board's *best* column reproduces the claim almost
exactly and the median does not:

| entry | cell | claimed | board median | board best |
|---|---|---|---|---|
| `d1_planner` | 16384 B=1 m=1 | 41.7–44 | 51.63 | **41.69** |
| `d1_planner` | 16384 B=64 m=1 | 47.7–57.4 | 62.50 | **47.76** |
| `d1_planner` | 65537 B=16 m=1 | 936.0 | 1020.0 | **936.5** |
| `d1_twiddle` | 1024 B=1 m=1 | 1.121 | 1.277 | **1.123** |
| `d1_prime` | 13 B=1 m=1 | 0.015 | 0.0187 | **0.0150** |
| `d1_batchlane` | 60 B=512 m=1 | 0.044–0.047 ("MKL parity") | 0.0545 | **0.0521** (MKL best 0.0432) |

`d1_batchlane`'s "my last library-losing m=1 cell is now at MKL parity" is the sharpest
case: on *bests* it is right (0.0442 panel vs 0.0432 MKL = 0.98×); on the scored median it
is 0.87×. The claim is true about the code and false about the statistic.

**(b) AVX-512 turbo licensing, measured on this exact part.** `d1_pow2` ran this to ground
and their record is the round's best piece of methodology: same binary, same core,
alternating 1.24 / 1.42 µs at 1024 B=1 across invocations; `perf` shows **3.04 GHz in the
fast mode and 2.86 GHz in the slow mode** — the heavy-512-FMA licence — with frequency
explaining ~6 % and retired cycles the other ~6 %. Crucially they ran the **fairness
check**: pinned to the same core, the pow2/MKL ratio is 1.14–1.15× *whether the core is in
the fast state (1.234/1.084) or the slow state (1.419/1.232)*. **Both throttle together, so
the scoring is fair** and the 9-run median absorbs it. This is why the 1024 B=1 median
ratio (0.85×) overstates the real deficit (~0.88× best-vs-best, ~1.14× in pow2's pinned
measurement).

**(c) Allocation and code-placement lottery.** `d1_pow2` also found and fixed a real
zero-result: the "deterministic huge-page arena" adopted across four entries since r5 was
getting **`AnonHugePages = 0 kB` at every size** — an anonymous `mmap` is only 4 KB-aligned,
so a 2 MB arena straddling a 2 MB boundary gets no huge page at all, and
`madvise(MADV_HUGEPAGE)` had been a no-op for the whole campaign. After align-and-trim,
2048 kB. Independently, `d1_prime`'s in-file probe measured **2.7–5.0 % spread within a
single process across byte-identical-arithmetic code copies, with a different winner per
process** — the same magnitude as the board gaps it was built to close.

**One claim that did not reproduce.** `d1_bluestein` reported that hugepaging the last
4K-paged table collapsed the 100003 B=8 bimodality — "now reads 2379–2395 across processes
(r5 board: median 3249, best 2377 — the median should collapse to the best)." The board:
median **2872.5**, best 2404.5, **spread 38.6 %**. It improved (3249 → 2872) but the median
did **not** collapse to the best; the bimodality survives at that cell. Their 65537 B=1
claim (1485–1517) fares better against a board median of 1531.7, though the 41.5 % spread
there says that mode is not fully dead either.

**One claim the board beat.** `d1_planner` claimed 852.1 at 65537 B=1 and 2841 at
100003 B=1; the board reads 792.4 and 2815.5. `d1_rader` claimed 861 at 65537 B=1 and the
board reads 813.3 (best 734.4). Under-claiming is the correct direction to be wrong in.

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

**§4.8 item 6 — "no primary measurement in the corpus for Ice Lake-SP or later *server*
parts... **Measure it on the node.**" This round measured it, and the answer is benign.**

The corpus's fear was Skylake-SP-style licence-based downclocking (the Gold 5120's
2.7 → 2.3 → 1.6 GHz), which would make 512-bit code a trap. On the Xeon Gold 6326 the
measured licence step is **3.04 → 2.86 GHz, about 6 %** (`d1_pow2`), it is *bimodal per
process* rather than a sustained penalty, and — the part that actually settles the
question — **MKL and the panel entries transition together, so the measured ratio is
invariant to which mode the core is in** (1.14–1.15× either way). Consequence for the
project: 512-bit is not penalised on this part, and the licence is a *variance* problem for
the scoring statistic, not a *level* problem for the code. That is a new, primary, on-node
number where the corpus had none.

The round also put a sharp CPU-specific qualifier on §4.4's "go split-complex, go 512-bit"
verdict, via `d1_composite` (confirmed and adopted by `d1_batchlane`): **on Ice Lake-SP
every 512-bit FMA-class op dispatches on p0+p5 only**, so at L=60 a 287-instruction zmm
"diet" kernel *loses* to a three-wide 256-bit ymm mix. Their formulation deserves to go
into §4.4 verbatim: *"a 512-bit diet kernel must get under the 256-bit version's
port-weighted uop count, not its instruction count."* The counter-case is in the same
round and is equally instructive: at the L=60 **B=1 chain**, which is latency-bound rather
than port-bound, the *same* widening to 512-bit wins by 25 % — "the same width that LOSES
in execute WINS here." So §4.4's answer in 1D is regime-dependent, not global.

Second-order, and worth recording against **§4.6 (model versus search for the instruction
schedule)**: the corpus's dispute is whether a *search phase* is needed. This round says
yes, but relocates the search target. `d1_prime` showed 2.7–5.0 % within-process spread
across candidates with **identical arithmetic** and a different winner per process — so
what needs searching is code and data *placement*, not the schedule. And `d1_race` supplies
the constraint the corpus never states: **the search must optimise the same statistic the
score uses.** Their r6 record diagnoses r5's losses precisely — min-based tie-breaking
shipped arms that the driver's medians had 14–21 % slower (32 B=512: shipped pow2 0.0176 vs
planner 0.0154), and min-of-3 60 ms bursts accepted "burst-fast/steady-slow" instances
(10007 B=1: ref 111.8, shipped median 142.6). That is a genuinely new methodological result
and it generalises well beyond this project.

**§4.3 (axis fusion) and §4.2 (L=17) were not touched** — they are 3D questions and 17 is
not in the 1D size list.

---

## 6. The single highest-value thing r7 should attack, per geometry

Grouped, because several geometries share one answer.

| L | highest-value target | why this and not something else |
|---|---|---|
| **13, 31** | **Stop working here. Re-differentiate instead.** | All eight cells won by 1.37×–6.96×. But `d1_prime`'s interleaved-pair kernel is now the winner *under four different names* (prime, rader, batchlane, planner, race, all within 15 %). The panel has no structural alternative left at small primes. If anything: build the one thing nobody has — the symmetric/antisymmetric split (§4.2b) — or reassign the implementer. |
| **32** | **Recover the r5 median at B=512, m=1.** | The only cell the panel *lost that it previously held*. The panel's best (0.0153) already equals MKL's median; `d1_twiddle`'s 0.0174 is stable and genuinely slower. This is a make-the-fast-mode-the-median problem, i.e. placement/licence hardening, not a new kernel. |
| **60** | **Same: make 0.0442 the median at B=512, m=1.** | Best-vs-best is 0.98×, median 0.87×, spread 20.3 %. `d1_composite`'s r5 verdict that no structural lever remains below PFA's ~204 FMA/xform still stands, so the entire remaining gap is dispersion. |
| **64** | Harden B=1 m=1 (1.07×?, inside the noise band). | The only pow2 B=1 cell the panel holds at all. It is one bad window from being lost. |
| **128** | **Fewer flops at B=1.** | The one small-pow2 cell with a *real* algorithmic gap: MKL's best is 0.0911 against our 0.1102 — **1.21× on bests**, and our spread is only 2.7 %, so there is no fast mode to find. Split-radix / conjugate-pair butterflies, per `d1_pow2`'s own diagnosis. |
| **1021** | **Kill the 33 % spread at B=256.** | `d1_race`'s best run wins the cell by 1.20×; its median loses it by 0.97×. This is the purest variance loss on the board and the cheapest cell to flip. Start with the cold-wisdom confound in §3.4. |
| **1024** | Nothing structural — it is 0.96× on bests. | Pooled with 4096 below; the shared lever is the same. |
| **4096** | ⭐ **Reduce the flop count of the B=1 radix-8 core. This is the single most valuable item on the whole board.** | 0.86× median *and* 0.86× best-vs-best, spreads 18 %/14 %, unchanged from r5 — no variance story, no placement story, no licence story. `d1_pow2` has already ruled out the alternatives (three array passes, L2-resident, compact tables) and names the remaining lever: conjugate-pair split-radix. It is a high-risk core rewrite, which is exactly why it needs a whole round rather than a corner of one. It also covers 1024 and 16384. |
| **10007, 100003** | **Collapse the median onto the best at the batched cells.** | Won everywhere but by the panel's thinnest large-prime margins (1.17–1.35× at 100003). `d1_bluestein`'s claimed bimodality kill did **not** reproduce at 100003 B=8 (§4): median 2872 vs best 2404, 38.6 % spread. ~16 % is sitting there for free. |
| **16384** | One push flips B=1 m=1 from 1.00×? to a win. | Dead tie with `fftw1d_patient`, tight spreads both sides. The 4096 flop-reduction work lands here too. |
| **65537** | Re-measure before optimising. | Won at 1.51×–2.39×, but the B=16 m=1 median regressed 933.9 → 1020.0 while `d1_rader` claimed a −3.1 % *gain* against a leased-core baseline (993) that never matched the board (933.9). The claim and the board disagree about the starting point; settle that first. |

**Two cross-cutting items that outrank most of the per-geometry work:**

1. **Fix the shard input-file race (§3.3) before r7 runs.** Two cells were destroyed and
   eleven more were measured with interleaved writes from two machines. Put the shard id in
   `in_L*_B*.bin` and in `t_*_r*.json`, or make the shard split disjoint by (L,B).
   A round that silently loses cells cannot be trusted to detect a regression.
2. **The dominant loss mechanism at m=1 is now dispersion, not speed.** Four of the seven
   lost cells, and three of the four regressions, are median-not-best. Two independent,
   already-measured mechanisms exist (turbo licence 6 %, placement lottery 2.7–5 %), one
   fix has already been found to have been silently doing nothing for a whole campaign
   (the huge-page arena, `AnonHugePages = 0`), and one entry has built a working probe
   (`d1_prime`). Making that probe panel-wide is worth more than any single kernel.

---

## 7. Curation — what this round keeps

Applying `docs/CURATION.md` in its stated order.

**Rule 1 (fastest correct entry per geometry — one per L, always).** Across the 50
measured cells the outright winners are: 13 → `d1_rader`, `d1_prime`; 31 → `d1_batchlane`,
`d1_prime`, `d1_race`; 32 → `d1_twiddle`, `d1_race`; 60 → `d1_batchlane`, `d1_composite`,
`d1_planner`, `d1_race`; 64 → `d1_race`, `d1_batchlane`; 128 → `d1_batchlane`, `d1_race`,
`d1_pow2`; 1021 → `d1_planner`, `d1_race`; 1024 → `d1_twiddle`, `d1_planner`; 4096 →
`d1_race`, `d1_pow2`; 10007 → `d1_planner`, `d1_race`; 16384 → `d1_pow2` (all four cells);
65537 → `d1_race`, `d1_rader`, `d1_planner`; 100003 → `d1_race`, `d1_bluestein`.

**The union is all nine entries.** Rule 4 (anything that beat a library) reaches the same
set independently — every entry beats a stock library at the cell it owns.

**Rule 2 (structurally different runner-up when close)** is what makes several of these
worth keeping on their own terms rather than as cell-winners:
* `d1_composite` owns exactly one cell (60 B=1 chain) but it is the round's largest single
  gain, 1.51× round-over-round and 2.86× over MKL, from a genuinely new construction
  (zero-shuffle broadcast-built pairs, output bit-identical to the version it replaces).
* `d1_prime` is the *source* of the small-prime kernel that four other entries now ship,
  and of the first-call placement probe that §6 recommends generalising.
* `d1_bluestein` and `d1_rader` are the structural alternatives to each other at the large
  primes, which is the contrast the brief's prime selection exists to test.

**Rule 3 (instructive failures with a documented, measured reason)** independently keeps
two entries whose *negative* results are the most useful things in the round:
* `d1_twiddle` built `d1_pow2`'s huge-page arena faithfully, measured it as a **pure loss
  on its engine** (+8–9 % at 16384 B=1, reproduced under four variants), and shipped it
  gated off with the A/B flags intact. That record stops the next panel re-adopting it
  blind. Its 128 B=1 fused-pair gate (fused wins at B=1, loses 9 % at B=512) is a second
  measured negative.
* `d1_pow2` contributes two: the `AnonHugePages = 0` finding that invalidates a
  campaign-wide assumption, and the turbo-licence fairness check that keeps the whole
  board's ratios credible.

**Rule 5 (do not promote near-duplicates).** The one place this bites is 13/31, where
`d1_rader`'s r6 win is `d1_prime`'s kernel ported near-verbatim, by their own account. I am
keeping both, because `d1_rader`'s value is not the small-prime kernel — it is the unpadded
65537 = 2^16 Rader convolution with the fused mid pass, which owns 65537 B=1 chain and is
the panel's clearest large-prime structural result. But **this is a warning for r7's
curation**: promoting all nine has now happened four rounds running, and `exemplars/` is
converging on a copy of `impl_N`, which is exactly what `CURATION.md` says it must not
become. If r7 does not re-differentiate 13/31, the next monitor should drop an entry there.

Not promoted: `baseline_dft` (the tracked harness floor, not a competitor).

**When writing `exemplars/d1_r6/NOTES.md`, record: the seven lost m=1 cells split into three
real deficits and four dispersion losses (§1); the shard input-file race that destroyed two
cells (§3.3); and that `d1_bluestein 1835.7 µs` at 65537 B=16 chain is unchecked and must
not be quoted.**
