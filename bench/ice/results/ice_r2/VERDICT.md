# Round ice_r2 — monitor's verdict

Scored on `a80n0.lqcd.mit`, slurm job 438572, 2026-08-22T19:06:56-04:00.
Sources: `results/ice_r2/leaderboard.txt`, `environment.txt`, `sweep.out`, `build_errors.txt`
(empty, 0 bytes), `failures.txt` (does not exist), the 75 `c_*.json` and 624 `t_*.json`
raw files, `agents/exits.txt` and the 19 agent logs, `strategies/*.md`, `docs/CURATION.md`,
`docs/LITERATURE.md` §4, and `results/ice_r1/{leaderboard.txt,VERDICT.md}` as the baseline.

---

## 0. Three corrections to the round's own framing, before any numbers

**0a. The scoring machine is Ice Lake-SP, not Cascade Lake.** `environment.txt` reads
`Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` — Ice Lake-SP: **two** 512-bit FMA pipes,
**1.25 MB** L2 per core, **~24 MB** L3 per socket, and no AVX-512 licence cliff. It is not a
Gold 5218 (Cascade Lake, one 512-bit pipe, 1 MB L2). This is the same correction ice_r1's
verdict made, and this round supplies the licence cost as a *number* rather than an absence:
`L6_pfa` measured `3.50 → 3.30 GHz` on 512-bit code, **−5.7%**, "fully covered by the −17%
instruction count on the x-pass." `L36_mixedradix` independently reports the cache hierarchy
it measured on the node as `48 KB L1d / 1.25 MB L2 / ~24 MB L3`. Every conclusion below is an
Ice Lake conclusion.

**0b. There is no Sapphire-Rapids-versus-Cascade-Lake gap to attribute anything to.** The
brief's framing — implementers develop on Sapphire Rapids, the monitor scores on Cascade
Lake, "MKL alone spans 2.9× between those machines" — does not describe this round.
`PANEL_BRIEF.md`'s `tryout.sh` builds and runs **on the reserved node itself**, and every
strategy record in `strategies/` says so explicitly ("dev machine = the reserved Ice Lake
node itself, via tryout.sh"; "all numbers from the NODE via tryout.sh"). The cross-machine
2.9× figure cannot explain anything measured here, and MKL in fact reproduces between ice_r1
and ice_r2 to **±0.8% at every one of the eight geometries** (§4). §4 re-attributes the
claimed-vs-measured gaps to their actual cause, which is co-tenant L3 contention in the dev
windows.

**0c. `L2↔DRAM` is not reachable from `cases.txt`.** ice_r1's verdict named the §4.3
`L2↔DRAM` batch-tiling construction as the highest-value L=36 move, on the basis that the
11.39 MiB working set is a ~9× overflow of the 1.25 MB L2. That is true of L2 but the working
set is **L3-resident** on a ~24 MB L3, so the boundary actually crossed at the graded cell is
`L2↔L3` (gap ~2.6–3×), not `L2↔DRAM` (gap 7×). Three independent measurements this round
corroborate that nothing on the board is DRAM-bound: non-temporal stores are catastrophic at
every size — L=6 `3pass_nt_pf` **0.765 vs 0.220** µs (3.5×), L=17 NT **28.9–30.8 vs 14.3**
µs/step, L=36 `SCRATCH` mode **154.3 vs 129.4** µs (−19%), L=64 `nt pf0` **1514.9 vs 1155.4**
µs — and RFO avoidance only pays at DRAM. §6 redirects L=36 accordingly.

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

### Batched — the graded chain (this round's measurement)

`cases.txt` is `6:64:4856  8:64:2572  13:32:1278  17:32:98  23:16:165  36:8:64  45:4:177
64:2:134`. Per-transform times are per-call ÷ (B × m).

| L | B, chain m | fastest correct panel entry | best library | panel speed-up |
|---|---|---|---|---|
| **6** | B=64, m=4856 | **L6_unrolled 0.213 µs/xform, 39.34 GF/s** (spread 0.1%) | mkl_dfti 0.341 µs, 24.59 GF/s | **1.60×** |
| **8** | B=64, m=2572 | **L8_batchsimd 0.544 µs/xform, 42.39 GF/s** (spread 1.5%) | mkl_dfti 0.628 µs, 36.71 GF/s | **1.15×** |
| **17** | B=32, m=98 | **L17_matrixsimd 13.562 µs/xform, 22.21 GF/s** (spread 1.9%) | ducc0_c2c 74.733 µs, 4.03 GF/s | **5.51×** (5.60× vs mkl2026) |
| **36** | B=8, m=64 | **L36_pencilfused 110.477 µs/xform, 32.75 GF/s** (spread 0.8%) | mkl_dfti 161.140 µs, 22.45 GF/s | **1.46×** |

Two of the four are statistical ties for first and should be read as such: L=6
`L6_unrolled 0.213` vs `L6_pfa 0.213` (0.17% apart) and L=8 `L8_batchsimd 0.544` vs
`L8_fusedaxes 0.544` (0.12% apart).

The other four graded geometries: L=13 `L13_direct` **4.662** vs mkl2026 6.027 (**1.29×**);
L=23 `L23_rader` **39.214** vs ducc0 218.175 (**5.56×**); L=45 `L45_mixedradix` **263.344**
vs mkl_dfti 520.811 (**1.98×**); and **L=64, where the board's one standing loss has
flipped** — `L64_blocked` **891.973** vs mkl_dfti 1021.687, i.e. **the panel is now 1.15×
ahead at every one of the eight geometries.** In ice_r1 MKL led L=64 by 1.16×.

### Non-batched — NOT MEASURED THIS ROUND, for the second round running

`cases.txt` still contains no `B=1` case, and no `t_*_B1.json` or `c_*_B1.json` exists in
`results/ice_r2/`. ice_r1's verdict asked, in bold, for B=1 rows to be added; that did not
happen, so **this half of the scorecard does not exist as a measurement and I will not
manufacture it.** ice_r1's stand-in (`geom/results/xarch_icelake`) is no longer usable
either: 16 of the 19 sources changed this round, so those binaries are gone.

What does exist is each implementer's own B=1 probe, taken on the node through `tryout.sh`
on a leased core with co-tenants active — *not* in the drained scoring window. Indicative
only, and where two implementers probed the same cell they disagree:

| L | panel entries at B=1 (own probes) | MKL at B=1 (own probes) | note |
|---|---|---|---|
| 6 | L6_unrolled **0.188** (sd 0.06%); L6_pfa 0.218–0.219 (sd 0.09%) | not quoted by either | 16% spread between two entries that tie exactly at B=64 |
| 8 | L8_fusedaxes **0.554** (sd 0.06%); L8_batchsimd **0.554** (sd 0.04%) | **0.546** (fusedaxes) vs **0.620** (batchsimd) | see below |
| 17 | L17_matrixsimd 13.7–14.2 (sd 0.02–0.04%) | not cleanly quoted | close to its own B=32 score of 13.562 |
| 36 | L36_mixedradix **95.39** (sd 0.16%); L36_pfa 106.287; L36_pencilfused 120.7 | 139.6 / 138.4 / 139.9 | **ranking inverts — see below** |
| 13 | L13_rader 5.369 | 5.734 | |
| 64 | L64_blocked **827.7** | 978.3 | |

Two things in that table are load-bearing.

**The L=8 non-batched result is contradicted by the panel's own data.** Two implementers
measured MKL at the same size, on the same node, on the same day, and got **0.546 and 0.620
µs** — a 13.5% disagreement on the baseline. On the first reading MKL is 1.5% *ahead* of the
panel at B=1; on the second the panel is 1.12× ahead. L=8 is already the thinnest batched
margin on the board (1.15×), so which of those is true matters, and no private probe can
settle it. **This is the clearest possible argument for putting B=1 in `cases.txt`.**

**At L=36 the non-batched ranking is the exact reverse of the batched one, and here the data
is trustworthy**: all three implementers' independent MKL B=1 readings agree to ±1%
(138.4 / 139.6 / 139.9), so their own numbers are comparable. The batched winner
`L36_pencilfused` (110.477, 1st) is **last** at B=1 (120.7, 1.16× over MKL); the batched
third-place `L36_mixedradix` (116.814, 3rd) is **first** at B=1 (95.39, 1.46× over MKL) —
a 21% inversion. ice_r1 found the same inversion with the same two entries. It has now
survived a full rewrite of all three L=36 sources, which makes it a structural property of
the pass shapes, not a plan race. If `bench/geom/best/` is ever to serve the non-batched
case it needs a different entry at L=36 than the one this leaderboard crowns.

---

## 2. What changed since ice_r1 — and did anything regress?

### The drift floor this round is ±2%, not ice_r1's ±12%

ice_r1's verdict established that "nothing under ~12% is a result on this harness unless the
entry's plan is pinned," because unmodified binaries swung up to −11.9% between scoring
windows. **That is no longer true, and the improvement is the round's most important
methodological result.** Two independent probes:

* **Untouched binaries.** Three sources are byte-identical to `impl_1` (`cmp` clean):
  `L17_winograd` **−0.19%**, `L17_rader` **−2.13%**, `L23_rader` **+0.18%**. Band:
  −2.1% … +0.2%.
* **Libraries.** Across all eight geometries, `mkl_dfti` moved **−0.15% … +0.8%** and
  `mkl2026_dfti` **−0.6% … +8.8%** (the +8.8% is mkl2026 at L=8, its one outlier).

So this round was taken in a properly drained window with plans pinned, and **a delta beyond
about ±3% is a real result.** The mechanism is that ice_r1's two prescriptions — the
clock-settle spin and the chain-shaped tuner stage — were adopted by essentially every
surviving implementer, and the run spreads show it: `L6_pfa` **13.5% → 0.3%**,
`L8_fusedaxes` **16.6% → 0.5%**, `L45_pfa` **12.0% → 0.9%**, `L36_mixedradix`
**12.0% → 3.7%**, `L6_unrolled` **3.2% → 0.1%**.

### Round-over-round, all nineteen entries

| entry | ice_r1 µs/xform | ice_r2 µs/xform | Δ | source edited? |
|---|---|---|---|---|
| L64_blocked | 1205.721 | **891.973** | **−26.0%** | yes (198 lines) |
| L13_rader | 6.131 | **4.680** | **−23.7%** | yes (2249 lines) |
| L64_radix8 | 1184.003 | **935.296** | **−21.0%** | yes (497 lines) |
| L36_pencilfused | 123.594 | **110.477** | **−10.6%** | yes (313 lines) |
| L45_mixedradix | 284.202 | **263.344** | **−7.3%** | yes (469 lines) |
| L17_matrixsimd | 14.471 | **13.562** | **−6.3%** | yes (391 lines) |
| L36_pfa | 119.163 | **112.727** | **−5.4%** | yes (240 lines) |
| L6_pfa | 0.220 | 0.213 | −3.2% | yes (319 lines) |
| L6_unrolled | 0.219 | 0.213 | −2.7% | yes (567 lines) |
| L36_mixedradix | 119.530 | 116.814 | −2.3% | yes (732 lines) |
| L17_rader | 19.099 | 18.693 | −2.1% | **no — identical binary** |
| L8_fusedaxes | 0.556 | 0.544 | −2.2% | yes (260 lines) |
| L8_batchsimd | 0.550 | 0.544 | −1.1% | yes (87 lines) |
| L23_matrixsimd | 39.584 | 39.223 | −0.9% | yes (246 lines) |
| L17_winograd | 16.113 | 16.082 | −0.2% | **no — identical binary** |
| L13_direct | 4.661 | 4.662 | +0.02% | yes (327 lines) |
| L23_rader | 39.142 | 39.214 | +0.2% | **no — identical binary** |
| L45_pfa | 295.054 | 297.097 | +0.7% | yes (578 lines) |
| L8_radix8 | 0.561 | 0.565 | +0.7% | yes (248 lines) |

### Per geometry

* **L = 6 — real but small, and the plan race is dead.** Both entries improved ~3% and
  landed on **exactly the same number, 0.213**, from opposite directions: `L6_unrolled`
  raced its way to `zxf` (zmm x-pass + ymm fused y+z) and `L6_pfa` to `z512x` (the same
  shape by another name). ice_r1's L=6 instruction was "kill the plan race first"; it is
  killed — spread 13.5% → 0.3% and 3.2% → 0.1%, and `L6_pfa` explains ice_r1's 13.5% swing
  exactly: `3.3/2.9 = 13.8%`, a clock-regime artifact of an unramped `create()` probe, not
  code. The two implementations have now **converged structurally**, which is why only one
  is promoted (§7). 1.60× over MKL, up from 1.56×.
* **L = 8 — the weakest geometry on the board, and it barely moved.** −1.1% / −2.2% / +0.7%,
  all inside drift; the three entries (0.544 / 0.544 / 0.565) are a statistical tie as they
  were in ice_r1. The margin over MKL went 1.13× → 1.15×, which is a *library* movement
  (+0.8% on mkl_dfti), not a panel gain. ice_r1's single L=8 prescription — read
  `ld_blocks_partial.address_alias` — was built and armed and then **blocked by the
  platform** (§3). L=8 is the one headline geometry where this round produced no measurable
  progress, and the reason is not the implementer's.
* **L = 17 — a real −6.3%, and the mechanism is not the one that was asked for.** ice_r1's
  verdict told `L17_matrixsimd` to port `L17_rader`'s ymm-tile port-5-relief transposes. It
  did the equivalent — rerouting the tile transpose through memory-destination
  `vextractf64x2` (68 16-byte stores replacing 40 port-5 shuffles + 20 stores, bit-identical
  outputs, `objdump`-verified at 510 sites) — and **it lost, by 1–1.7 µs/step**. The −6.3%
  came from somewhere else entirely: a `#pragma GCC optimize("schedule-insns",
  "sched-pressure")` on the chunk kernels, matched A/B on the node at **15.732 → 14.528
  µs/step, −7.7%**. GCC does no pre-RA scheduling on x86 by default, so the phase-serial
  kernel source reached the issue queue unmixed. `L8_fusedaxes` independently analysed the
  ymm-tile port-5 relief and declined to port it, with port math: its p0/p5 pool is already
  balanced (896 shuffles against a 1072-cycle floor), so halving the width doubles the op
  count for zero relief. **The port-5 lever is spent at L=17; the schedule lever is not.**
* **L = 36 — the largest headline gain, −10.6%, and the ranking changed hands.** ice_r1 had
  `L36_pfa` first and `L36_pencilfused` third and 1.04× behind; ice_r2 has `pencilfused`
  first at 110.477 with the tightest spread of the three (0.8%). All three sources were
  rewritten (313 / 240 / 732 diff lines). The gains are store-mode and prefetch-protocol
  work inside the existing pass structures (`bcst0` broadcast mode at pw=4 for pencilfused;
  `tr=1` transposed store worth 7.7% for pfa; pinned-D prefetch for mixedradix), plus the
  chain-shaped tuner. 1.35× → 1.46× over MKL.

### Off the headline four

* **L = 64 — the board's one loss is gone.** `L64_blocked` −26.0% and `L64_radix8` −21.0%;
  the panel now leads MKL by 1.15× where it trailed by 1.16×. Both gains are the same
  finding: under chain semantics NT stores cost 20–25%, and the whole ice_r1 deficit "and
  then some" was the store mode. `L64_blocked` predicted the flip in its own record
  ("~903 vs ~1007: the one loss should flip to a win") and measured 891.973 vs 1021.687.
* **L = 13 — the largest relative gain of the round, −23.7%.** `L13_rader` was rebuilt
  wholesale (2249 diff lines) on top of `L13_direct`'s `lanes=lines` pipeline and went from
  1.32× behind it to a dead tie (4.680 vs 4.662). `L13_direct` itself moved +0.02% for 327
  edited lines.
* **L = 23 — nothing happened.** The winner `L23_rader` is an untouched binary (+0.2%, a
  drift reading); `L23_matrixsimd` edited 246 lines for −0.9%. The dead heat from ice_r1
  (39.214 vs 39.223, 0.02% apart) persists.
* **L = 45 — `L45_mixedradix` −7.3%**, `L45_pfa` +0.7% for 578 edited lines.

### Did anything regress?

**No entry regressed beyond the drift floor, and no entry lost correctness.** The three
nominal regressions — `L8_radix8` +0.7%, `L45_pfa` +0.7%, `L13_direct` +0.02% — are all
inside the ±2% band that untouched binaries define. What *is* worth recording is **effort
spent for no measurable return**: `L45_pfa` (578 diff lines → +0.7%), `L13_direct` (327 →
+0.02%), `L23_matrixsimd` (246 → −0.9%) and `L8_radix8` (248 → +0.7%) all landed inside
noise. In three of those four the record says why, with a number, which is what makes them
worth keeping anyway (§5, §7).

One stability regression: **`L64_blocked`, this round's L=64 winner, has the worst run spread
of any panel entry (4.0%, up from 2.3%)**, and its own record is candid that "my own
identical build read 903–1288 µs across four windows while MKL read 1007–1101." Its 891.973
is a genuine minimum but its plan is not yet pinned the way L=6/L=8/L=45 now are.

---

## 3. Failures, missing entries, and correctness — the adversarial pass

**Correctness: all 26 backends pass at every geometry, under both gates. No fast wrong
answer is hiding in this leaderboard.** Verified directly from the 75 `c_*.json` files, not
from the leaderboard's `ok` column:

* Single-transform `rel_l2` ranges **1.3e-16 … 8.42e-16** against `tol = 1e-12`; the maximum
  belongs to `baseline_matrix` at L=17, i.e. the O(L⁴) reference, as expected.
* Chain `rel_l2` ranges up to **2.494e-12**, which *exceeds* the printed `tol` of 1e-12 —
  I checked whether that is a smuggled failure. It is not: `check.py:64` sets
  `eff_tol = tol * max(1, m**0.5)` because roundoff accumulates as √m, so at L=6 (m=4856)
  the chain tolerance is **6.97e-11** and the 2.494e-12 is a factor 28 under it. The
  offender is `baseline_matrix`; every panel entry is well inside the library pack
  (L=6: `L6_unrolled`/`L6_pfa` **2.920e-13** against `mkl_dfti` 2.902e-13, FFTW 2.436e-13,
  ducc0 4.966e-13). **No panel entry is an accuracy outlier at any geometry.**
* `chain_ok` and `ok` are both `true` in all 75 files. The chain gate is the strong one: it
  compounds an error at any one of up to 4856 steps into the end state.

**Nothing is missing from the measurement.** 624 `t_*.json` = 26 backends × 8 cases × 3 runs,
exactly; 399 are `supported:false` (each `L<n>_*` entry correctly declines the other seven
geometries) and 225 = 75 × 3 are supported timings, so no leaderboard block drops a row. 75
`c_*.json` = 19 panel + 8 `baseline_matrix` + 48 library, exactly. `build_errors.txt` is
0 bytes, `failures.txt` does not exist, and `sweep.out` contains no error, warning, skip,
timeout, signal or NaN. **Nothing crashed, hung, or failed to build.**

### Three implementer agents died — named, with evidence

`agents/exits.txt`: **16 of 19 exited 0** (against 4 of 19 in ice_r1 — the worker-launch fix
largely worked). Three did not:

> **`L17_rader` exit=139**, **`L17_winograd` exit=139**, **`L23_rader` exit=139**

All three are agent-harness crashes, not FFT bugs, and the failure mode has *changed* from
ice_r1's out-of-memory storm:

* `L17_rader.log`: `panic(main thread): Segmentation fault at address 0xBBADBEEF` /
  `oh no: Bun has crashed. This indicates a bug in Bun, not your code.` — after
  `Elapsed: 260ms`, `Peak: 0.34 GB`.
* `L17_winograd.log`: identical Bun 1.4.0 panic after `Elapsed: 5488ms`, `Peak: 0.38 GB`.
* `L23_rader.log`: same panic, plus `oh no: multiple threads are crashing`.

Peak RSS of 0.34–0.38 GB on a 0.14 TB machine rules out ice_r1's memory exhaustion; this is
a Bun 1.4.0 segfault. The corroboration that these three produced **nothing** is independent
of the exit codes: `cmp` confirms `impl_2/{L17_rader,L17_winograd,L23_rader}.c` are
**byte-identical to `impl_1/`** — and byte-identical to the copies already sitting in
`exemplars/ice_r1/`. Their leaderboard rows (18.693, 16.082, 39.214) are re-measurements of
ice_r1 binaries, and their −2.1% / −0.2% / +0.2% deltas *are* this round's drift probe (§2).

**Two of the nineteen have no strategy record at all: `L17_winograd` and `L23_rader`.**
`strategies/` holds 17 files for 19 entries, and those two are the gap. Their agents have now
crashed in both ice rounds, so no ice-panel record has ever been written for either. Per
CURATION ("entries whose strategy record is missing" must not be promoted) this bars both
from promotion — including `L23_rader`, which is the *fastest* entry at L=23. §7 handles that
conflict explicitly.

### A curation defect carried over from ice_r1, still open

ice_r1's verdict closed with an operational note: eight of its twelve promotions had no ice
record, and the `geom` records were to be copied in before promoting. **That was not done.**
`exemplars/ice_r1/` contains twelve `.c` files and only **four** `.strategy.md` files
(`L13_direct`, `L13_rader`, `L17_matrixsimd`, `L17_rader`). Eight promoted exemplars are
currently shipping without the record that CURATION says is what makes the code useful
later. This should be fixed at the same time as this round's promotion.

---

## 4. Claimed numbers vs. measured numbers

**The framing to attribute gaps to — Sapphire Rapids development versus Cascade Lake
scoring — does not apply (§0b).** Development and scoring were the same Ice Lake node, and
MKL reproduces between the two rounds to ±0.8%. More to the point, **the direction of the
error has reversed since ice_r1**, which is itself the round's headline finding about method.

ice_r1 found a systematic **+22% … +47% optimism** in every private probe, and diagnosed it:
the private arenas timed a bare transform while the graded unit is a chain step including the
driver's unitary scale pass and a buffer ping-pong. Its prescription was "tune in the chain
regime." That prescription was adopted almost universally — `L6_unrolled`, `L6_pfa`,
`L8_fusedaxes`, `L8_batchsimd`, `L8_radix8`, `L13_rader`, `L17_matrixsimd`,
`L36_{pfa,pencilfused,mixedradix}`, `L45_mixedradix`, `L64_{blocked,radix8}` all cite it by
name — and **the optimism is gone. Fourteen of sixteen working entries are now accurate to a
few percent, and conservative:**

| entry | own claim / prediction | measured | error |
|---|---|---|---|
| L6_unrolled | 0.213 (auto graded-chain median) | **0.213** | **0.0%** |
| L6_pfa | 0.213 min / 0.214 median | **0.213** | **0.0%** |
| L8_radix8 | 1faa 0.566 / 0.567 in-chain | **0.565** | −0.2% |
| L45_mixedradix | "expect ~255–263 scored" | **263.344** | top of own range |
| L8_fusedaxes | 0.553 quiet-window tuned | **0.544** | −1.6% (conservative) |
| L64_blocked | "~903 vs MKL ~1007, ratio 0.90" | **891.973**, ratio 0.873 | −1.2% (conservative) |
| L17_matrixsimd | "expect ~14.0–14.4 µs/step" | **13.562** | −3.1% (conservative) |
| L8_batchsimd | 0.562 min (contended window) | **0.544** | −3.2% (conservative) |
| L64_radix8 | 967.9–1020.7 quiet windows | **935.296** | −3.4% (conservative) |
| L36_pfa | 120.825 new build at B=8 | **112.727** | −6.7% (conservative) |
| L36_mixedradix | "~113–121 at the graded cell" | **116.814** | inside own range |
| L36_pencilfused | "113–119 in a scoring window near MKL 160.7" | **110.477** | −2.2% (conservative) |

**The one entry far from its own number is `L13_rader`, and it is 30% *faster* than it
claimed.** Its record's table reads `graded chain B=32 m=1278: 6.687 µs/xform` with a
window range of 6.05–6.88, and its explicit prediction is "expected node standings: graded
6.4–6.9 µs/xform (MKL 6.39 ± 0.05)." It measured **4.680**. This is not a machine difference
and not a correctness problem (rel_l2 3.0e-16, chain gate passed). The cause is visible in
the scored run's own description string: the in-plan arena read `zs:4209` ns/vol — the entry
itself measured 4.2 µs in the drained window, so the code did not change between prediction
and score; the *window* did.

That mechanism generalises, and it is the thing to write down:

> **MKL is not a valid same-window normaliser for these kernels.** Several implementers
> quoted their numbers as ratios against MKL measured on the same core in the same window,
> on the reasonable assumption that MKL absorbs contention the same way they do. It does
> not. `L8_fusedaxes`: "identical final binaries measured 0.553 / 0.745 / 0.781 / 0.565
> across ~an hour while MKL held 0.632–0.660 — my 3 MiB chain working set is far more
> exposed to co-tenant L3 traffic than MKL's." `L64_blocked`: "my own identical build read
> 903–1288 µs across four windows while MKL read 1007–1101." `L13_rader`'s windows had MKL
> at 6.29–6.46 against the scored 6.245 — a 2.3% baseline difference that cannot explain a
> 30% swing in its own time. **A panel kernel's dev-window number is a lower bound on its
> scored performance by up to 30%, and normalising by MKL hides that rather than correcting
> for it.** The scored drained window is the only comparable measurement; within-window A/B
> is the only usable dev instrument. `L23_matrixsimd` and `L45_mixedradix` state this rule
> explicitly in their records and are the two most accurate predictors in the table.

Two smaller reporting notes:

* **ice_r1's `backends:` description defect is not fixed.** The description string still
  need not come from the run that supplied the reported minimum, and it still mixes units:
  `L8_fusedaxes` prints `chain-arena{fusedAA2+pfs=0.411}` next to a measured 0.544, and
  `L8_batchsimd` prints `arena{FUSEDAA2/s0=0.473}` next to 0.544. Both are in-plan *kernel*
  prices excluding the driver's scale pass, not comparable to the leaderboard column beside
  them. `leaderboard.py` should carry the description from the min-supplying run.
* **`mkl2026_dfti` at L=36 has a 31.6% run spread**, the worst number anywhere on the board,
  against `mkl_dfti`'s 0.1% in the same block. Its minimum (163.507) is fine and agrees with
  ice_r1 to 0.01%, but any conclusion drawn from mkl2026's L=36 median is worthless.

---

## 5. Which `LITERATURE.md` §4 open question this round moved

### §4.6 — "Model versus search for the instruction schedule" — MOVED, and settled to a rule with the sign measured four times

This is the round's cleanest literature result, and it is what produced the L=17 gain. §4.6
sets §01's claim ("you should not need a search phase for L = 6, 8, 17, 36") against §06's
correction that genfft's spill-optimality proof holds only for powers of 4 and that the
schedule is therefore "the primary thing to search at every one of our sizes," and calls the
question "cheap to settle."

It was settled. Five entries A/B-tested `#pragma GCC optimize("schedule-insns",
"sched-pressure")` on the same node with the same toolchain (gcc 11.4, `-march=native`), and
**the sign flips with the shape of the kernel source**:

| entry | kernel source shape | Δ from pre-RA scheduling |
|---|---|---|
| **L17_matrixsimd** | phase-serial **rolled** loops (cosine block, then sine block, re-loading rows) | **−7.7%** (15.732 → 14.528 µs/step, matched A/B) |
| L13_direct | already a fused single-load sweep | **+5.2%** (race 4460 vs 4240 ns/vol) |
| L23_matrixsimd | fully unrolled straight-line, 23 independent accumulator chains | **+1.7%** (pinned A/B/A; also visible unpinned, 42.64 vs 40.77) |
| L36_mixedradix | PFA 4×9 codelet class | **worse** (MKL-normalised 0.866 vs 0.802) |
| L45_mixedradix | same shape as L23_matrixsimd | not attempted, on L23's evidence |

The answer §4.6 wanted: **yes, the schedule is worth searching, it costs exactly one build,
and the payoff is up to −7.7% — but the predictor is source shape, not size.** GCC performs
no pre-RA scheduling on x86 by default, so for a kernel whose source is phase-serial the
"search" is a single pragma; for a kernel that is already fully unrolled with independent
dependence chains, the scheduler has no ILP left to expose and its register-pressure
heuristic costs more than it earns. That is a sharper statement than §06's "search
everything" and it should be written back into the section, together with the observation
that §06's own cheap test (`-fno-schedule-insns`) points the wrong way on x86, where the
default is already off.

### §4.1 — "How many registers does a batch-vectorised codelet actually need?" — CLOSED on this machine, by the exact check §07 §7.8 prescribed

§4.1's open question is "how much spill traffic do the AVX2 batch-vectorised L=6 and L=8
codelets actually generate," with §01 measuring 17 live registers for n=6 and 19 for n=8 and
concluding AVX2 "will spill a little." Three entries ran §07 §7.8's check — count stack
traffic in the generated assembly before believing any timing — and all three found **zero**:

* `L6_pfa`: "checked the gcc 11.4 asm: **zero stack traffic**."
* `L8_batchsimd`: "GCC spill audit of the ICX build: **zero** zmm stack moves in
  `faa_run_*`/`f_run_*` — the corpus §10 spill disease does [not apply]."
* `L8_fusedaxes`: "**zero spills** in every shipping kernel (`vol_aa_s`, `vol_p_s`,
  `vol_aa` verified by `objdump` on the icelake-server build)."

**§01's premise is dead on this hardware**: 32 zmm registers, not 16 ymm, and the 2L
data-only bound (12 for L=6, 16 for L=8) plus temporaries now fits with room. The corollary
matters more than the answer: the spill argument was the thing blocking wider batch granules
and 512-bit codelets, and with it removed both L=6 entries reinstated their AVX-512 variants
— and the zmm variants **still lost**, for a different reason. `L6_unrolled`'s `zff` ("more
512-bit is better") measured 0.239–0.272 against `zxf`'s 0.243 and was rejected because its
512-bit `vpermt2pd`/`valignq` transposes are **port-5-only on ICX**, the same port the second
FMA pipe lives on. `L6_pfa`'s `z512yz` lost by 11% and explicitly ruled out spills as the
cause. **The scarce resource at L=6, as at L=17, is port-5 issue slots — not registers, not
multiplications.**

### §4.5 — "Padding: does L = 8 need it, and where?" — NOT moved, and now blocked with a named cause

This was ice_r1's single highest-value L=8 ask: read `ld_blocks_partial.address_alias`,
because L=8's volume stride of 8192 B = 2 × 4096 makes every load from `in` falsely alias a
recent store to `out` in the low 12 bits. `L8_fusedaxes` built the probe properly — three
~2 ms measurements in `create()`, at `(out−in) ≡ 0 mod 4096` and at +32 lines, plus a forced
variant at the degenerate placement — and it returned nothing:

> "Under tryout it reports `pmc=na`: **`perf_event_open` returns EACCES for the leader**
> (ssh session, not the monitor's slurm context)."

The scored leaderboard line reads **`pmc=na`** too. So the PMU is unavailable in the scored
slurm context as well, and **the brief's claim that `perf_event_open` works on this node is
false in both contexts.** §4.5 cannot be settled, at L=8 or anywhere, until that is fixed at
the harness level — and this is now blocking the weakest geometry on the board for a second
round. `L8_fusedaxes` did contribute one thing the counter cannot: a proof that the hazard is
**permutation-invariant** — "every phase-A 16-line load window covers all 16 mod-16 line
classes, and each in-flight out-store iteration occupies exactly 2 of them" — so no
k1/y reordering can remove it and the only levers are timing or padding.

### §4.3 — axis fusion, and the `L2↔DRAM` case — NOT moved, and not reachable from `cases.txt`

Nobody built the batch-tiled all-three-axes-in-L2 construction. Per §0c the reason is
structural rather than a failure of will: the graded L=36 cell's 11.39 MiB working set is
**L3-resident** on a ~24 MB L3, so the graded configuration cannot cross the `L2↔DRAM`
boundary the section names, and the panel's own NT-store measurements (§0c) confirm nothing
on the board is DRAM-bound. The nearest datum is `L36_pencilfused`'s off-graded B=32 probe —
**174.6 vs MKL 236.4** at a ~45 MiB working set — which is in the right regime but is not the
fusion experiment. To test §4.3 the graded cell would need B ≥ 32 at L=36; to *keep* testing
it at B=8 is impossible.

### §4.4 — split vs interleaved complex — not moved, but three implementers independently converged on it as the next lever

§4.4 is already CLOSED in the corpus's favour (§08 §5.4: Popovici et al. measure `DFT_n ⊗ I_ν`
split kernels — literally our kernel shape — at 1.3–2× FFTW). This round adds no measurement,
but all three L=36 records name split-complex as their single remaining structural attack,
with port-5 as the stated mechanism: `L36_pfa` — "the only structural attack is
split-complex (SoA) … and `L64_radix8` already runs split-complex here. Big job; right shape
for a dedicated [round]"; `L36_pencilfused` — the swaps "are in interleaved complex; killing
them needs split-complex lanes through pass B." `L64_radix8` is the only entry on the board
already running split, and it just gained 21%. That convergence is what §6 acts on.

### Also closed, from ice_r1's open item

§4.8 item 6 (AVX-512 on Ice Lake-SP) was closed in ice_r1; ice_r2 supplies the licence cost
as a number: **512-bit costs 3.50 → 3.30 GHz, −5.7%, against a −17% instruction count** on
`L6_pfa`'s x-pass, i.e. a net win. There is no AVX-512 cliff on this part, and 256-bit
variants continue to lose everywhere they are raced.

---

## 6. The single highest-value thing the next round should attack

**Above all four geometries, two harness items that no implementer can fix.**

1. **Get `perf_event_open` working in the scored context, or say it is unavailable.** Two
   rounds of verdicts have now named a PMU counter as the top priority at L=8; the probe
   exists, is armed by default, degrades cleanly, and returns `EACCES` in both the dev ssh
   session and the scored slurm job. `L6_pfa`, `L6_unrolled`, `L36_pfa` and `L8_batchsimd`
   all name PMU counters as their next instrument. This one fix unblocks four geometries.
2. **Add `B=1` rows to `cases.txt`.** Asked for in ice_r1, not done, and this round produced
   the proof it matters: two implementers measured MKL at L=8 B=1 as 0.546 and 0.620 (§1), so
   the panel does not know whether it wins or loses the non-batched case at its thinnest
   geometry; and at L=36 the non-batched ranking is the *reverse* of the batched one, for the
   second round running and across a full rewrite of all three sources.

Then, per geometry:

* **L = 6 — *(1.60× over MKL; two entries converged on one number.)* Stop tuning and change
  the layout.** The plan race is dead (spreads 0.1% / 0.3%), §4.1 is closed with zero spills,
  and both entries independently raced their zmm shapes and landed within 0.2% of each other
  at 0.213 — the search space of *this* structure is exhausted. Both records also identify
  the same unexplained residue: the in-plan race reads 0.156 µs/vol quiet while the graded
  chain scores 0.213, of which only ~0.02 µs is the driver's scale pass. The one structural
  lever named and not built is `L6_pfa`'s **padded-row (16-double) scratch**, which costs ~36
  extra 32-byte stores in the x-pass to remove 3 split loads per plane and give the y/z stage
  aligned full-512 rows — and it is exactly §4.5's odd-cache-line padding rule applied at
  L=6. Build it; it does not need the PMU to be decisive, only to be explained.
* **L = 8 — *(1.15× over MKL, still the thinnest margin on the board, and it did not move.)*
  Pad the stride rather than waiting for the counter.** The `address_alias` read is blocked
  and may stay blocked. `L8_fusedaxes` has already proved the collision count is
  permutation-invariant, so the hypothesis is testable without a counter by the direct
  method: apply ducc0's one-line guard (`if ((dstride & 256) == 0) dstride += 16;` /
  `make_noncritical()`, §08 §5.5), which the corpus reports as production practice, and
  measure the graded cell with and without. If the time moves, §4.5 is answered by timing;
  if it does not, the hypothesis is dead and L=8 can stop spending rounds on it. Everything
  else at L=8 is now a tie inside 0.2% between three radix-8 cousins, and the port math in
  `L8_fusedaxes`'s record says its kernel is already at 1.18–1.24× a 1072-cycle p0/p5 floor
  — there is at most ~20% left in this structure.
* **L = 17 — *(5.51× over MKL, the panel's biggest win; the port-5 lever is spent, the
  schedule lever is not.)* Push §4.6 further, and merge the phases at source level.** The
  two things this round proved at L=17 are that memory-destination `vextractf64x2` port-5
  relief **loses** 1–1.7 µs/step, and that pre-RA scheduling on phase-serial source **wins**
  7.7%. `L13_direct`'s record explains why the second worked and names the follow-on: "their
  record's next lever, merge the phases at source level, is a thing L=13 has had since
  panel_r9." Do that — hand-interleave the cosine and sine blocks in `chunk17n_g` so the
  scheduler starts from mixed source (the corpus's ZIPP construction, §10, ~+20% on prime
  passes) — and A/B it against the pragma, since the two are substitutes rather than
  complements. This is the only lever on the board with a measured 7.7% already banked and a
  named mechanism for more.
* **L = 36 — *(1.46× over MKL, the largest headline gain; redirect from §4.3 to §4.4.)*
  Build the split-complex (SoA) codelet lanes.** ice_r1 pointed L=36 at §4.3's `L2↔DRAM`
  batch tiling; §0c retires that instruction — the graded cell is L3-resident, the boundary
  is `L2↔L3`, and NT stores lose by 19% here for exactly that reason. What replaced it is
  better evidenced: all three L=36 records independently name split-complex as the sole
  remaining structural attack, they agree on the mechanism (the interleaved layout's swaps
  are port-5 work and port 5 is the measured bottleneck at L=6, L=17 and L=36 alike), §4.4
  is closed in split's favour with Popovici et al. measuring 1.3–2× on our exact kernel
  shape, §08 §1.10 prescribes the granule (8 volumes for split double, so every vector
  access is a whole 64-byte line), and `L64_radix8` is already running split on this node and
  just gained 21%. `L36_pfa` calls it a "big job; right shape for a dedicated [round]" —
  give it one, seeded from `L64_radix8`'s split machinery rather than from scratch. Secondary
  and cheap: the batched/non-batched inversion (§1) means whichever entry wins B=8 is the
  wrong one to ship for B=1; if `cases.txt` gains a B=1 row, `L36_mixedradix`'s pinned-D
  prefetch path (95.39 vs MKL 139.6) is the entry to develop for it.

---

## 7. Curation — what to keep, and why

Applying `docs/CURATION.md`'s four grounds in order.

**1 — Fastest correct entry per geometry (mandatory, one per L).**
`L6_unrolled` (0.213), `L8_batchsimd` (0.544), `L13_direct` (4.662), `L17_matrixsimd`
(13.562), `L36_pencilfused` (110.477), `L45_mixedradix` (263.344), `L64_blocked` (891.973).
All correct at 2.3e-16 … 4.5e-16 with the chain gate passed.

**The one exception, stated openly: L=23.** The fastest entry is `L23_rader` (39.214), but
its agent SIGSEGV'd (§3), its source is byte-identical to `impl_1` **and** to the copy
already in `exemplars/ice_r1/`, and **no ice-panel strategy record has ever been written for
it** — CURATION forbids promoting an entry whose record is missing, and re-promoting a
byte-identical file would ship a literal duplicate with no record. I promote
**`L23_matrixsimd`** (39.223, **0.02% behind** — a dead heat well inside the ±2% drift floor)
in its place: it is in the fastest class at L=23, it did real work this round, and it has a
fresh record. Ground 1's purpose is served; the rule against recordless promotions is not
broken.

**2 — A structurally different runner-up that came close.**
* `L13_rader` (1.00×, 4.680 vs 4.662) — Rader-13 CRT against a dense 13×13 conj-folded
  kernel, a **dead tie**, and the round's largest relative gain (−23.7%). Its 2249 changed
  lines make it new code, not the ice_r1 exemplar of the same name.
* `L36_pfa` (1.02×, 112.727) — GT-PFA 4×9 two-sweep against the winner's plane-fused
  y+z-then-strided-x pass structure. Genuinely different pass shape, 2% behind.
* `L64_radix8` (1.05×, 935.296) — radix-8² split-complex per axis against the winner's 8×8
  two-stage blocked form. Kept for a second reason under ground 4 below: it is **the only
  split-complex entry on the board**, which is the lever §6 sends L=36 after.

**Deliberately not promoted as runners-up, per the near-duplicate rule:** `L6_pfa` (0.213 —
a literal tie, but it and `L6_unrolled` have now *converged* on the same zmm-x-pass + ymm-y/z
shape, so the second copy shows nothing the first does not; its asm spill audit survives in
`strategies/L6_pfa.md`, which CURATION tracks independently); `L36_mixedradix` (1.06%, also
PFA 4×9 two-sweep, a near-duplicate of the promoted `L36_pfa`); `L45_pfa` (1.13×, outside the
band and also PFA 9×5); `L17_winograd` and `L17_rader` (untouched binaries already in
`exemplars/ice_r1`, and `L17_winograd` still has no record at all).

**3 — Instructive failures whose record documents the number that killed them.**
* `L23_matrixsimd` — already in on ground 1, and it carries the round's most transferable
  negative: the pre-RA scheduling pragma at **+1.7%, the sign flipping against
  `L17_matrixsimd`'s −7.7% on the same node and toolchain**, with the mechanism (fully
  unrolled straight-line code with 23 independent accumulator chains has no ILP left for the
  scheduler to expose). That pair of numbers is half of this round's §4.6 answer, and
  `L45_mixedradix` already cites it to avoid repeating the experiment.
* `L8_fusedaxes` (1.00×, 0.544 — a 0.12% tie for first) — promoted for its record rather
  than its rank. It is the only entry that built the §4.5 apparatus ice_r1 asked for, it
  documents the platform failure that blocked it (`perf_event_open` → `EACCES`, `pmc=na` in
  both dev and scored contexts), it carries the permutation-invariance proof that tells the
  next round the alias hazard cannot be scheduled away, **and** it records the analysed
  rejection of the ymm-tile port-5 relief with the port math that explains why
  `L17_matrixsimd`'s equivalent attempt lost. Structurally it is a radix-8 cousin of
  `L8_batchsimd`, which would normally bar it; the dead tie for first and the fact that L=8
  is the geometry the next round must attack make the exception worth taking.

**4 — Anything that beat a library baseline.** Every promoted entry beats the best library at
its geometry, which is new: **the panel now leads at all eight geometries.** `L64_blocked`
and `L64_radix8` are the entries that closed the board's one standing loss (MKL 1.16× ahead
in ice_r1 → panel 1.15× ahead now), and both are promoted.

### Operational notes before running `promote.sh`

* `strategies/` has records for 17 of 19 entries. **All twelve names promoted below have
  one**, so `promote.sh` will not warn on any of them.
* **Fix the ice_r1 debt at the same time (§3):** `exemplars/ice_r1/` ships twelve `.c` files
  and only four `.strategy.md` files. Copy the `bench/geom/strategies/<name>.md` records in
  for the eight that are missing, as ice_r1's verdict directed.
* `exemplars/ice_r2/NOTES.md` should record that `L17_winograd`, `L17_rader` and `L23_rader`
  are byte-identical carry-overs re-measured but not revised (their agents crashed), so their
  ice_r2 leaderboard rows are the round's drift probe and not results; and that `L23_rader`
  is the fastest L=23 entry but is deliberately not promoted for want of a record.

PROMOTE: L6_unrolled L8_batchsimd L8_fusedaxes L13_direct L13_rader L17_matrixsimd L23_matrixsimd L36_pencilfused L36_pfa L45_mixedradix L64_blocked L64_radix8
