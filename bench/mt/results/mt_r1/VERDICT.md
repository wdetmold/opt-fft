# mt_r1 — monitor's verdict

Round: `mt_r1`, the first multicore round (phase 2).
Scored on: `p51n1`, Intel Xeon Gold 5218 (Cascade Lake, 2 sockets × 16 cores, 1 MiB L2/core,
22 MiB L3/socket, 1× 512-bit FMA unit, DDR4-2666), 32 threads, `OMP_PROC_BIND=close`,
`OMP_PLACES=cores`, governor `powersave`, gcc 11.4.0, slurm job 438534.
Comparison baseline for "what changed": `../../../geom/results/panel_r11/leaderboard.txt`
(the same node, single-threaded).

Statistic used by the harness: **minimum over three independent processes of each process's
minimum sample.** This matters a great deal this round and §3 is mostly about it.

---

## 0. Bottom line

* **Nothing failed correctness, nothing failed to build, nothing crashed, nothing is missing.**
  All 216 (backend × case) correctness records pass at `rel_l2` 1.2e-16 … 4.5e-16 against
  `numpy.fft` with tol 1e-12, and every leaderboard row has a matching `c_*.json`. No fast
  wrong answers to strike.
* The panel still beats the best threaded library at **21 of 24 scored cells**, by 1.17× to
  5.44×. It **loses three cells**: L=8 B=32768 (0.91×), L=13 B=8192 (0.81×),
  L=64 B=8 (0.81×). All three are large-working-set or low-batch-large-volume cells.
  Phase 1 won every cell; that clean sweep is over.
* The lead **widened at every batched cell** (L=6 B=4096 1.43×→2.24×, L=17 B=256 3.99×→5.36×,
  L=23 B=128 4.13×→5.44×, L=45 B=16 1.74×→2.20×) and **narrowed at every B=1 cell where the
  libraries had threading headroom** (L=17 5.42×→3.21×, L=23 5.48×→3.32×, L=45 1.99×→1.44×).
  MKL threads a single 17³ volume 4.33× and a single 23³ volume 7.43×; the panel's best
  entries get 2.12× and 4.03×. **Single-volume decomposition is the panel's weak axis and it
  is the axis the libraries improved most.**
* The round's largest methodological problem is not any one entry: it is that **nine entries
  return a different create-time plan pick from different processes**, with 1.04×–1.71× spread,
  while within-process sample `sd` is 0.03–0.9%. Min-of-min then scores each entry on its
  luckiest process. Several headline margins in the leaderboard are therefore not reproducible.

---

## 1. Headline per geometry — fastest correct panel entry vs best library

### The four brief geometries

| L | case | fastest panel entry | best library | ratio |
|---|---|---|---|---|
| **6** | B=1 | **L6_pfa 0.210 µs** | mkl_dfti 0.371 µs | **1.77×** |
| 6 | B=4096 | **L6_pfa 0.0094 µs/t** (38.59 µs/call) | mkl_dfti 0.021 (86.45) | **2.24×** |
| 6 | B=65536 | **L6_pfa 0.0395 µs/t** (2586 µs/call) | fftw3_estimate 0.121 (7951) | **3.07×** |
| **8** | B=1 | **L8_fusedaxes 0.554 µs** (L8_batchsimd ties at 0.554) | mkl_dfti 0.658 | **1.19×** |
| 8 | B=2048 | **L8_fusedaxes 0.0262 µs/t** (53.72 µs/call) | mkl_dfti 0.0367 (75.21) | **1.40×** |
| 8 | B=32768 | L8_radix8 0.173 µs/t (5654 µs/call) | **fftw3_patient 0.161 (5261)** | **0.93× — LOSS** |
| **17** | B=1 | **L17_matrixsimd 7.112 µs** | mkl_dfti 22.820 | **3.21×** |
| 17 | B=256 | **L17_matrixsimd 0.726 µs/t** (185.9 µs/call) | fftw3_measure 3.893 (996.6) | **5.36×** |
| 17 | B=4096 | **L17_winograd 1.222 µs/t** (5005 µs/call) | fftw3_patient 3.925 (16078) | **3.21×** |
| **36** | B=1 | **L36_pfa 25.659 µs** | mkl_dfti 38.156 | **1.49×** |
| 36 | B=32 | **L36_mixedradix 5.589 µs/t** (178.9 µs/call) | mkl_dfti 7.510 (240.3) | **1.34×** |
| 36 | B=512 | **L36_mixedradix 14.170 µs/t** (7255 µs/call) | fftw3_patient 19.969 (10224) | **1.41×** |

### The four extra geometries the round also measured

| L | case | fastest panel entry | best library | ratio |
|---|---|---|---|---|
| 13 | B=1 | **L13_direct 5.670 µs** | mkl2026_dfti 7.632 | 1.35× |
| 13 | B=512 | **L13_direct 0.308 µs/t** | mkl2026_dfti 0.360 | 1.17× |
| 13 | B=8192 | L13_rader 0.976 µs/t (7998) | **fftw3_patient 0.787 (6450)** | **0.81× — LOSS** |
| 23 | B=1 | **L23_matrixsimd 11.835 µs** | fftw3_patient 39.332 | 3.32× |
| 23 | B=128 | **L23_matrixsimd 2.287 µs/t** | fftw3_patient 12.437 | 5.44× |
| 23 | B=2048 | **L23_rader 6.052 µs/t** (12394) | fftw3_patient 12.466 (25531) | 2.06× (see §3.2) |
| 45 | B=1 | **L45_mixedradix 58.317 µs** | mkl2026_dfti 83.911 | 1.44× |
| 45 | B=16 | **L45_mixedradix 16.873 µs/t** | mkl2026_dfti 37.056 | 2.20× |
| 45 | B=256 | **L45_pfa 26.897 µs/t** (6886) | fftw3_patient 57.708 (14773) | 2.15× |
| 64 | B=1 | **L64_blocked 127.995 µs** | mkl_dfti 153.235 | 1.20× |
| 64 | B=8 | L64_radix8 91.521 µs/t (732.2) | **mkl_dfti 73.742 (589.9)** | **0.81× — LOSS** |
| 64 | B=128 | **L64_blocked 95.686 µs/t** (12248) | fftw3_patient 184.357 (23598) | 1.93× |

**Best result of the round:** `L17_winograd` at L=17 B=4096 — 5004.7 / 5020.2 / 5012.1 µs
across three processes, `sd` 0.14–0.20%. 644 MB of compulsory traffic in 5.005 ms = **129 GB/s**,
3.21× the best library and 1.72× the next panel entry, with no measurable variance. It is the
only large-batch number in the round that is both fast and boring.

---

## 2. What changed since the previous round, per geometry

Phase 1's timings are single-core, so the honest comparison has two parts: parallel efficiency
against `panel_r11` on the same node, and the change in the ratio to the best library.

### Parallel efficiency (32 threads vs panel_r11 single thread, same node, same cell)

| L | B | panel_r11 1T | mt_r1 32T | speedup | vs 25.4× ceiling* |
|---|---|---|---|---|---|
| 6 | 1 | 0.208 | 0.210 | 0.99× | serial by design |
| 6 | 4096 | 0.394 | 0.0094 | **41.8×** | **165%** |
| 8 | 1 | 0.551 | 0.554 | 0.99× | serial by design |
| 8 | 2048 | 0.922 | 0.0262 | **35.2×** | **139%** |
| 13 | 1 | 5.733 | 5.670 | 1.01× | serial by design |
| 13 | 512 | 8.075 | 0.308 | 26.2× | 103% |
| 17 | 1 | 15.066 | 7.112 | 2.12× | 8% |
| 17 | 256 | 20.903 | 0.726 | 28.8× | 113% |
| 23 | 1 | 47.648 | 11.835 | 4.03× | 16% |
| 23 | 128 | 64.157 | 2.287 | 28.1× | 111% |
| 36 | 1 | 114.561 | 25.659 | 4.47× | 18% |
| 36 | 32 | 163.380 | 5.589 | 29.2× | 115% |
| 45 | 1 | 312.948 | 58.317 | 5.37× | 21% |
| 45 | 16 | 405.188 | 16.873 | 24.0× | 95% |
| 64 | 1 | 952.743 | 127.995 | 7.44× | 29% |
| 64 | 8 | 1247.663 | 91.521 | 13.6× | 54% |

\* The 32× ideal is not reachable: the panel's own in-plan clock probes report
`clk512/256 = 2.29/2.79 GHz` with 32 cores active (L17_matrixsimd, L23_matrixsimd description
strings) against `2.89/3.89 GHz` at one core in `panel_r11`. A 512-bit kernel's clock-adjusted
ceiling is therefore **32 × 2.29/2.89 = 25.4×**. Six cells exceed it — see §5.

**The shape of the result is a three-band structure, and it is the same at every L:**

* **B=1 — 1.0× to 7.4×.** No geometry parallelises a single volume well. Below L=17 nobody
  even tries: every L=6/8/13 entry ships B=1 serial, each on a measured dispatch cost
  (L8_batchsimd's empty-job pool round trip 0.407/0.620/1.684 µs at T=2/8/32; L17_rader's
  OpenMP region 2.7–8.3 µs; L23_matrixsimd's GOMP fork+join 6.2–8.2 µs). Those are correct
  decisions with numbers attached, not surrenders.
* **Working set 27–48 MiB — 24× to 42×, i.e. *superlinear*.** L=6 B=4096 (27 MiB), L=8 B=2048
  (32 MiB), L=17 B=256 (38 MiB), L=36 B=32 (46 MiB), L=23 B=128 (48 MiB). One core cannot hold
  these and is fill-buffer-limited; 32 cores hold them in 32 MiB of aggregate private L2 plus
  44 MiB of combined L3. This band is where phase 2 pays.
* **Working set ≥ 500 MiB — 13× to 26×, and falling.** Achieved DRAM rates: 64 GB/s (L=23
  B=2048), 72 (L=13 B=8192), 88 (L=64 B=128), 93 (L=8 B=32768), 105 (L=36 B=512),
  108 (L=45 B=256), 129 (L=17 B=4096), 175 (L=6 B=65536). All three library losses are in
  this band.

### Per geometry

**L=6.** No regression. `L6_pfa` widened its batched lead (B=4096 1.43×→2.24× over MKL) and
is the round's superlinearity champion at 41.8×. `L6_unrolled` is 1.05×/1.24×/1.83× behind at
the three cells but is the *stable* one at B=4096 (47.67/49.40/49.40 µs, `sd` ≤0.42%) where
`L6_pfa` returns 38.59/38.83/59.02. B=1 is unchanged from phase 1 by design.

**L=8.** The one geometry with a genuine regression relative to phase 1's clean sweep:
**fftw3_patient now takes B=32768.** Within-band detail matters (§3.2): fftw's min is
5261 µs but its median is 9686/9644/5592 across the three processes, while `L8_fusedaxes`
returns 5779 µs at `sd` 0.25%. FFTW really does reach 5.26 ms; it reaches it in a minority of
samples. B=1 and B=2048 are unchanged or slightly better.

**L=17.** The strongest geometry in the round and the one that changed most structurally.
At B=1 and B=256 `L17_matrixsimd` (dense conjugate-symmetric) wins as it did in phase 1; at
B=4096 `L17_winograd` **overtakes it by 1.72×** (1.222 vs 2.106 µs/t) having been 1.03× behind
it at B=256 in phase 1. The ranking inside L=17 is now regime-dependent. Against the libraries
the batched lead widened (3.99×→5.36× at B=256) and the B=1 lead narrowed sharply
(5.42×→3.21×) because MKL threads a single volume 4.33× and the panel gets 2.12×.

**L=36.** No entry regressed against a library, but the internal ranking moved and one entry
fell behind FFTW. `L36_pfa` takes B=1 (25.659 µs, `sd` 0.09%, 1.49× MKL — the round's most
robust B=1 result). `L36_mixedradix` takes B=32 (5.589, with `L36_pencilfused` at 5.644, a tie)
and B=512. **`L36_pencilfused` regressed from a 1.02× tie at B=32 to 1.73× behind at B=512**,
placing it 4th, behind `fftw3_patient`. Its own record predicted 13–18 µs/t for that cell and
5–9 µs for B=1; it measured 24.5 and 28.5. This is the round's cleanest single-entry
demonstration that the fused construction pays inside aggregate cache and loses at the DRAM
wall.

**L=13.** `L13_direct` holds B=1 and B=512 by 2–3% over `L13_rader`. **B=8192 is lost to
fftw3_patient by 1.24×**, and unlike the L=8 loss this one survives on medians (fftw 6.49 ms
vs L13_rader 8.01 ms). `L13_direct` is a further 1.31× behind at that cell.

**L=23.** No library loss; the lead widened at B=128 (4.13×→5.44×) and narrowed at B=1
(5.48×→3.32×, MKL threading a single volume 7.43× against L23_matrixsimd's 4.03×).
`L23_rader` and `L23_matrixsimd` are within 1.03× at B=128 and tie at B=2048 on representative
runs (§3.2).

**L=45.** `L45_mixedradix` takes B=1 and B=16; **it then collapses at B=256 to 79.067 µs/t —
2.94× behind its sibling `L45_pfa` (26.897) and 1.37× behind fftw3_patient.** Dead stable
across processes (20241/20259/20278 µs, `sd` ≤0.09%), so this is deterministic, not noise.
Diagnosis is in its own record: it shipped the wallaby pick `vol32-v2-pfpp`, whose in-plan
arena read **25.4 µs/t against the driver's 79.07** — a 3.1× arena mis-pricing of the streaming
regime, the exact trap the record says it had fixed by raising the arena to 4× L3.

**L=64.** `L64_blocked` holds B=1 (1.20× MKL) and B=128 (1.93× FFTW). **B=8 is lost to
mkl_dfti by 1.24×** (`L64_radix8` 732.2 vs 589.9 µs/call, both `sd` < 0.5% — a real, stable
loss). This loss was pre-registered: `L64_radix8`'s record measured MKL 1.21× ahead at B=8 on
its dev machine, predicted 60–110 µs/t on the node, and measured 91.5. L=64 B=1 is the only
cell in the round where a 16-thread single-socket team won: **both** L=64 entries independently
picked `nth=16`.

---

## 3. Adversarial audit

### 3.1 Correctness, builds, crashes, omissions — clean

* `build_errors.txt` is present and **empty (0 bytes)**. Every one of the 20 implementations
  and 6 library backends compiled with `-O3 -march=native` on the benchmark node
  (`slurm-438534.out`).
* `failures.txt` **does not exist** — no entry crashed or hung.
* I cross-checked the leaderboard against the raw records programmatically: **24 cases,
  216 leaderboard rows, 216 matching `c_*.json`, zero with `ok != true`, zero with
  `rel_l2 > 1e-13`.** Worst observed `rel_l2` is 8.4e-16 (`baseline_matrix` at L=17, expected
  for an O(L⁴) matvec).
* No panel entry is absent from any case its geometry was scored at: all 19 implementations
  appear in all three cells of their L. The only systematic absence is `baseline_matrix` from
  the largest cell of each geometry, which is a deliberate harness skip (it would run for
  minutes), consistent across all eight geometries.
* The correctness witness is `numpy.fft` on the same input file, with numpy's own agreement
  against `python/slow_dft.py` established separately. That chain is sound.

**Nothing here disqualifies any entry.** What follows are not correctness failures, but they
do change how several leaderboard lines should be read.

### 3.2 Non-reproducible plan picks — the round's real defect

`leaderboard.py` ranks on min-over-processes of min-over-samples. Within-process `sd` this
round is 0.03–0.93% almost everywhere. So the "run spread" column is *not* sampling noise: it
is **different processes installing different create-time plans, or landing on different page
placements**. Nine entries are affected:

| entry | cell | the three process minima (µs/call) | worst/best |
|---|---|---|---|
| L8_radix8 | L=8 B=32768 | 9674 / 5662 / **5654** | **1.71×** |
| L36_mixedradix | L=36 B=512 | 11766 / 7413 / **7255** | **1.62×** |
| L6_pfa | L=6 B=4096 | 59017 ns / 38831 / **38593** | **1.53×** |
| L36_pencilfused | L=36 B=512 | 17589 / 12724 / **12567** | 1.40× |
| L17_rader | L=17 B=1 | 12.629 / **9.134** / 12.613 | 1.38× |
| L64_blocked | L=64 B=128 | 12401 / **12248** / 15740 | 1.28× |
| L23_rader | L=23 B=2048 | 14719 / **12394** / 14705 | 1.19× |
| L17_matrixsimd | L=17 B=256 | 199.2 / 208.5 / **185.9** | 1.12× |
| L45_pfa | L=45 B=256 | 7189 / 7087 / **6886** | 1.04× |

Two leaderboard headlines **depend on the lucky process** and should not be quoted as the
entry's number:

* **`L17_rader` at L=17 B=1 (9.134 µs).** Two of three processes read 12.61–12.63 µs at
  `sd` ≤0.46%. Its representative time is 12.6 µs, which puts it *behind* `L17_winograd`
  (8.998, stable) and `L17_matrixsimd` (7.112, stable), not ahead of them by the 1.28×/1.09×
  the table implies. Its own record documents a mode-2 team-size race at B=1 that is exactly
  the mechanism.
* **`L23_rader` at L=23 B=2048 (6.052 µs/t).** Two of three processes read 14.70–14.72 ms
  against `L23_matrixsimd`'s 14.817/14.818/14.827 ms at `sd` ≤0.11%. The honest verdict at that
  cell is a **tie at ~14.7–14.8 ms (7.2 µs/t)**, and the panel's lead over fftw3_patient there
  is **1.74×, not 2.06×**.

One more that is close to load-bearing: **`L36_mixedradix` at B=512.** Its 1.41× win over
fftw3_patient holds in two processes (7255, 7413) and inverts in the third (11766, which is
slower than FFTW's 10224). The win is real but it is a 2-in-3 win.

`L6_pfa`'s own record also flags a `tryout.sh` **NOT REPEATABLE** bit-compare at unscored
B=8/B=33 caused by cross-process pick flips choosing bit-differently-rounded kernels. All
scored sizes pass the stricter check, so this is not a correctness finding — but it is the same
instability showing up in the output bits rather than the clock, and it is worth remembering
before the next round trusts a single process's pick.

### 3.3 The min-of-min statistic also flatters the libraries

Being adversarial in both directions:

* **L=8 B=32768.** `fftw3_patient` minima are 5277/5268/5261 µs but its **medians are
  9686/9644/5592** (`sd` 35–40%). It alternates between a ~5.27 ms mode and a ~9.7 ms mode and
  is scored on the fast mode. `L8_fusedaxes` returns 5795/5779/5792 with medians 5807/5799/5801.
  So FFTW genuinely wins on the harness's statistic by 1.10×, and on any typical-case statistic
  `L8_fusedaxes` wins by up to 1.67×. Both facts belong in the record; the loss stands as
  scored.
* **L=13 B=8192.** `fftw3_patient` also shows `sd` 43–46%, but its medians are 6497/7363/6487 —
  it is typically fast. That loss (1.24×, and 1.23× on medians) is **fully real**.
* **L=64 B=8.** `mkl_dfti` 589.9/591.8/590.6 at `sd` 0.3–0.5%; `L64_radix8` 732.2/733.2/733.4
  at `sd` ≤0.4%. **Fully real**, no statistical escape.
* `fftw3_patient` at L=17 B=4096 carries a 76.3% run spread and at L=36 B=512 a 50.2% one; both
  are far enough behind that it changes nothing.

### 3.4 The one entry that is stably, diagnosably wrong-tuned

**`L8_batchsimd` at L=8 B=32768: 9698 / 9706 / 9674 µs, `sd` 0.03–0.05%.** Its two siblings
run the same cell on the same node at 5654 and 5779 µs. It is 1.68× behind them and it sits at
**exactly** the number shared by `mkl_dfti` (9675), `mkl2026_dfti` (9669), `fftw3_measure`
(9664) and `fftw3_estimate` (9596) — 786 MB of traffic including the write-allocate RFO at
81 GB/s. That is the cached-store DRAM roof. Its siblings deleted the RFO with non-temporal
stores and got 93 GB/s of useful traffic instead. So this is not a machine limit: its
create-time race failed to select NT at the streaming cell, and it did so reproducibly. Its
own next-round list asked exactly this question ("does nt-s0 hold at B=2048 in the node's
streaming set") and `L8_radix8`'s record pre-registered the mechanism ("the arena cap gives the
race more L3 residency than the scored run has, which biases the race toward cached stores").
The mechanism is confirmed. It is an instructive failure with a documented number, not a
disqualification.

---

## 4. Claimed versus measured

Every implementer developed on **wallaby** (Xeon Gold 6448Y, Sapphire Rapids, 32 cores on one
socket, 2 MiB L2/core, ~60 MiB L3, **two** 512-bit FMA units, DDR5) and was scored on
**Cascade Lake** (2×16 cores, 1 MiB L2, 22 MiB L3/socket, **one** 512-bit FMA unit, DDR4, and
32 threads spanning two sockets). The prompt's note that MKL alone spans 2.9× between the two
machines is the right calibration: **the median wallaby→node factor across all 19 entries and
all cells this round is ≈1.9×, ranging 1.13× to 3.5×.** Almost every gap is the machine.

Attributable to the machine — no criticism implied:

| entry | cell | claimed (wallaby) | measured (node) | factor |
|---|---|---|---|---|
| L17_winograd | B=4096 | 1.082 µs/t | 1.222 | **1.13×** (best transfer of the round) |
| L6_pfa | B=65536 | 0.0336 | 0.0395 | 1.18× |
| L36_mixedradix | B=512 | 9.97 | 14.170 | 1.42× |
| L45_pfa | B=256 | 16.9 | 26.897 | 1.59× |
| L6_unrolled | B=65536 | 0.0432 | 0.072 | 1.67× |
| L8_fusedaxes | B=2048 | 0.0154 | 0.0262 | 1.70× |
| L64_blocked | B=128 | 52.6 | 95.686 | 1.82× |
| L36_pfa | B=1 | 13.50 | 25.659 | 1.90× |
| L17_winograd | B=1 | 4.60–4.90 | 8.998 | 1.9× |
| L64_radix8 | B=8 | 41.5 | 91.521 | 2.20× |
| L17_matrixsimd | B=4096 | 0.84 | 2.106 | 2.51× |
| L13_rader | B=8192 | 0.360 | 0.976 | 2.71× |
| L23_matrixsimd | B=2048 | 2.05 | 7.235 | 3.53× |

Several entries pre-registered node bands and hit them, which is the behaviour to reward:
`L8_fusedaxes` predicted 0.082 µs/t at 200 GB/s or ~0.16 if socket-0 residency halved the
bandwidth and measured 0.176 — the pessimistic branch, correctly identified in advance.
`L64_radix8` predicted 95–160 / 60–110 / 90–160 µs/t at B=1/8/128 and measured 135.7 / 91.5 /
146.9 — three for three. `L64_blocked` predicted 60–140 and 90–160 and measured 128.0 and 95.7.
`L36_mixedradix` predicted 20–30 µs at B=1 and 15–25 µs/t at B=512 and measured 28.9 and 14.2.

**Three gaps are *not* explained by the machine**, because a sibling entry ran the same cell on
the same node much faster:

1. **`L8_batchsimd`, L=8 B=32768:** claimed 0.058 µs/t at B=16384, measured 0.295. Roughly
   2.5–3× of that is the machine; the remaining **1.68×** is a tuner mis-pick, proven by
   `L8_radix8` (0.173) and `L8_fusedaxes` (0.176) on the same node and cell. §3.4.
2. **`L45_mixedradix`, L=45 B=256:** claimed 25.1 µs/t, measured 79.067. Inside its own stated
   node band (45–110), so the *prediction* was honest — but its **in-plan arena on the node
   read 25.4 µs/t against the driver's 79.07**, a 3.1× mis-pricing of the very regime the arena
   exists to price, and `L45_pfa` runs the same cell at 26.897. This is an arena-fidelity
   failure, not a machine effect.
3. **`L36_pencilfused`:** predicted 5–9 µs at B=1 (measured 28.5, 3.2–5.7× off) and 13–18 µs/t
   at B=512 (measured 24.5). Its record stated that "≥11 µs means the barrier scan is pricier
   on two sockets than one — try a 16-thread single-socket pool pick there." The node says the
   two-socket cost is far larger than it allowed. Its siblings run B=512 at 14.2 and 19.5.

**A shared premise the node falsified.** Every entry built 16-thread candidates on the
reasoning that the driver `fread`s/`memset`s both caller buffers on its main thread (which I
confirmed in `driver.c:105-118` — `aligned_or_die` then `memset(out)` then `fread(in)`, all
serial, and `sweep.sh` sets no `numactl` policy), so all caller pages live on socket 0 and the
far 16 threads pay UPI for nothing. **At every batched cell the node picked T=32**, often
decisively: `L36_pfa` publishes `ip32=6.27` vs `ip16=19.93` µs/vol at B=32 (3.2×);
`L36_pencilfused` publishes `t32=15.71` vs `t16=24.01` at B=512 (1.53×); `L13_rader` picked
`ntb=32/32`. T=16 won exactly once, at **L=64 B=1**, where both entries independently chose it.
And L=6 B=65536 sustains **175 GB/s**, which is 88% of this node's *dual-socket* practical
bandwidth and well above a single socket's ~100 GB/s. So the far socket is contributing memory
bandwidth, not merely paying interconnect — which means the caller's pages are **not** all
effectively on socket 0 by the time the timed samples run. The leading explanation is Linux
AutoNUMA page migration during the multi-second timing loop, which would also explain the
across-process bimodality in §3.2 (tiny within-sample `sd`, large between-process spread, and
FFTW's within-process two-mode behaviour). **This is a hypothesis, not a measurement** — the
cheap check is `cat /proc/sys/kernel/numa_balancing`, `numastat -p` on one long run, and one
geometry re-run under `numactl --interleave=all` vs `--membind=0`. It should be the first thing
next round does, because a great deal of design effort this round was spent on a NUMA model
that may not describe the machine.

---

## 5. Which LITERATURE §4 open question moved

**§4.3 — "Is axis fusion worth 3× or 3%?", specifically the re-opened L2↔DRAM clause.** This is
the question the round moved, and it moved a long way. §4.3's re-opening note says the panel's
r3 answer was measured entirely across an **L1↔L2** boundary (2.6× bandwidth gap) and that the
untested case is **L2↔DRAM** (7× gap), naming "tile the batch so a tile fits L2, then run all
three axes inside the tile" as "the largest untried structural move on the board."

What mt_r1 supplies:

1. **The crossover, measured, on a CPU, for cubes in our range.** Working sets of 27–48 MiB —
   which fit 32 cores' aggregate 32 MiB L2 plus 44 MiB combined L3, but not one core's
   1 MiB L2 — scale at **24–42×** on 32 cores, i.e. 95–165% of the clock-adjusted 25.4×
   ceiling. Working sets ≥ 500 MiB scale at **13–26×** and sit on a DRAM roof of 64–175 GB/s.
   The 32-core machine does not just add FLOPs; it adds a cache level of 76 MiB, and the entire
   payoff of phase 2 lives in the band that level covers. §4.3's Tolmachev rule survives with
   CPU numbers attached, and the numbers are large at this gap, not single-digit percent.
2. **Store order beats pass count again, and by much more.** §4.3's r3 finding was "pass count
   worth 5%, store order worth 18%." At the L2↔DRAM gap the same asymmetry is worth
   **20–72%**: non-temporal stores, rejected in all eleven single-core rounds, invert at 32
   cores, and the inversion was measured independently by six entries (L23_matrixsimd first,
   then L36_pfa −17% at B=512, L17_rader −14% at B=4096, L8_fusedaxes −30% at B=32768,
   L45_pfa −26%, L64_* 20–30%). The three cells where the panel *lost* to a library are the
   three cells where a panel entry failed to make that switch (§3.4) or had no RFO left to
   delete. And the flip is regime-gated in both directions: NT is 2.5–6× *worse* at
   cache-resident batch (L6_pfa 0.0218 vs 0.0088; L8_batchsimd's node arena `T32/nt-s0=0.1807`
   vs `T32/s0w=0.0297`).
3. **A direct A/B of two L=17 schedules across the boundary.** `L17_winograd`'s three rotating
   passes lose to `L17_matrixsimd`'s pipelined X-first by 1.27× at B=1 (0.15 MiB, L2-resident)
   and **win by 1.72× at B=4096** (614 MiB, DRAM-bound), at 129 GB/s with 0.2% variance. Same
   geometry, same node, same round, opposite verdicts on the two sides of the gap. That is the
   regime-dependence §4.3 predicted, now with a number on each side.
4. **The corollary the panel discovered for itself and should keep:** a create-time tuning
   arena must exceed the *aggregate* cache of the whole team, not one socket's L3. `L6_pfa`
   calls this "the single biggest lesson of the round" (raced 0.0159 vs real 0.0434 µs/vol
   before the cap was raised, −21% recovered); `L8_fusedaxes` measured a 27% wrong pick from the
   same cause; `L45_pfa` measured 9.6 in-arena against 25.7 in the driver; `L36_pencilfused`
   measured 4.49 in-arena against 10.3 real. `L45_mixedradix` and `L8_batchsimd` are what
   happens when the correction is incomplete.

**Also moved: §4.8 item 6, the AVX-512 downclock.** The corpus closed this with Intel's turbo
table for *1–8 active cores* (2.9 GHz for both AVX2 and AVX-512) and explicitly dismissed the
Gold 5120's 1.6 GHz figure as a 9+-core number that "does not describe a single-threaded run."
This round supplies the 32-active-core number for the Gold 5218 itself: the panel's own in-plan
probes report **clk512/256 = 2.29/2.79 GHz** against 2.89/3.89 GHz at one core — a 0.79×
licence penalty on 512-bit and 0.72× on 256-bit, with the 512/256 ratio *improving* from 0.74
to 0.82. The practical consequence is the 25.4× ceiling used throughout §2: **~21% of every
"missing core" in this round's efficiency numbers is licence clock, not the code.** 512-bit
remains strictly preferable, as §4.8 concluded, and is now preferable by a slightly larger
margin at scale.

**Moved sideways: §4.8 item 5**, "no quantified comparison of batch/vector-loop placement…
our largest untapped search axis." The round searched (team size × schedule × store mode)
grids at every geometry and the answer is nearly uniform: **contiguous static volume blocks at
T=32**. Dynamic scheduling lost everywhere it was published (`L8_batchsimd` dyn2 0.0888 vs
static 0.0352 at B=64; `L64_blocked` S-dyn 63.6 vs 54.8 at B=8). That axis is now largely
settled for this harness and should stop consuming tuner surface.

**Not moved: §4.5 (L=8 padding, 4 KiB aliasing, and the `ld_blocks_partial.address_alias`
counter).** No L=8 entry ran a padding A/B or that counter this round; all three spent the
round on threading. The question is exactly where §4.5 left it, and L=8 is now the geometry
with a library loss.

---

## 6. The single highest-value thing the next round should attack

**Before anything else, one harness-level item that dominates all of them.** Settle the page
placement (§4 last paragraph) and then fix it *in the harness*, identically for every backend —
either first-touch the caller's buffers in parallel or set an explicit policy — and make the
timing pass report medians alongside minima. Two consequences: the ≥500 MiB band stops being
scored on page-migration luck, and the nine entries in §3.2 stop being scored on their luckiest
process. `L36_mixedradix` raised the related question of whether an entry may call
`move_pages(2)` on the caller's buffers during warmup. **My ruling: no** — it mutates state the
driver owns and that every other backend shares, and it is not something a caller would want a
library to do silently. It should be measured as a diagnostic and reported, but the fix belongs
in the harness, where it applies to MKL and FFTW too.

Per geometry:

* **L=6 — make the B=4096 pick reproducible.** `L6_pfa` is 2.24× ahead of MKL when its race
  lands and 1.46× ahead when it does not (38.6 vs 59.0 µs). The kernel is done; the plan
  lottery is the whole remaining gap. Fix the race, then the entry is uncontroversially the
  fastest thing on the machine at every L=6 cell.
* **L=8 — take B=32768 back, and it is a tuning fix, not a kernel fix.** `L8_fusedaxes` at
  5779 µs (`sd` 0.25%) is 1.10× behind FFTW's best sample and 1.67× ahead of its typical one.
  Two concrete moves: raise every L=8 create-time arena so it streams past the *aggregate*
  76 MiB (which is what `L8_batchsimd` failed to do and `L8_radix8` does only 2 times in 3),
  and then close §4.5 — L=8's volume stride is exactly 8192 B = 2 × 4096, the one geometry
  where the 4 KiB-aliasing store-forwarding penalty is maximally degenerate, and nobody has
  ever read `ld_blocks_partial.address_alias` on it. Highest-value single counter in the
  project.
* **L=17 — parallelise one volume properly.** MKL threads a single 17³ volume 4.33×; the
  panel's best gets 2.12×. `L17_matrixsimd`'s own decomposition probe says where it goes:
  ~1 µs handshake+barrier, ~0.5 µs plane-granularity skew, and **~2 µs of all-to-all coherence
  traffic** (87 KiB of the intermediate crossing cores between phases). Its named fix —
  locality-aware phase-2 chunk order, so each thread starts on the columns of the planes it
  wrote, at zero extra traffic — is cheap and untried. Closing half that gap turns 3.21× over
  MKL into ~4.5×. Secondary: merge `L17_winograd`'s B=4096 store schedule into
  `L17_matrixsimd`, since the 1.72× reversal at that cell is a schedule difference on the same
  arithmetic.
* **L=36 — settle B=512 and stop losing it 1 time in 3.** `L36_mixedradix` wins 1.41× in two
  processes and loses to FFTW in the third; `L36_pfa` is stable but 1.37× slower. The right
  move is the one §4.3 names and no L=36 entry has actually built: **tile the batch so a tile
  fits one core's 1 MiB L2, then run all three axes inside the tile.** `L36_pencilfused` fuses
  planes but streams whole volumes; `L36_pfa` and `L36_mixedradix` are volume-parallel
  row-column. A genuine L2-tiled fused variant at L=36 is the largest untried structural move
  on the board and this is the geometry to try it on.
* **L=13** — close the 1.24× to `fftw3_patient` at B=8192. `L13_rader` achieves 72 GB/s where
  FFTW gets 89 and where L=6 reaches 175: this cell is not at the machine's roof, so there is
  traffic to delete. NT stores are the obvious first swing (its race reads `nt!:463` against
  `i:353` at a cache-resident surrogate — i.e. the arena is still mis-pricing the regime).
* **L=23** — B=1. 4.03× against MKL's 7.43× on a single volume is the largest single-volume
  deficit in the round; the plane-granularity imbalance (23 planes over 32 threads) is named in
  the record and unaddressed.
* **L=45** — `L45_mixedradix` must adopt `L45_pfa`'s streaming construction or its arena. A
  2.94× intra-geometry gap at B=256 with a 3.1× arena mis-pricing is a solved problem sitting
  in the sibling file.
* **L=64** — B=8, the one loss where the panel is behind a library on a *stable* number.
  `L64_radix8`'s record already names the residual: the gang-internal all-to-all, where each
  lane reads 3/4 of its second-pass input from sibling lanes, and the fix is a lane-blocked
  first-pass store layout that scatters by second-pass owner. One round of real work, clearly
  scoped, and it is the only route to taking B=8 from MKL.

---

## 7. Curation decision

Applying `docs/CURATION.md`'s four grounds in order. Every entry below is either the fastest
correct entry at its geometry, the fastest correct entry at some scored *cell*, a structurally
distinct runner-up within the rule-2 margin, or the documented instructive failure of a
construction the next round needs to see. All 19 strategy records stay in `strategies/`
regardless; this is the reading list.

**Promoted (16):**

| entry | ground |
|---|---|
| `L6_pfa` | fastest at all three L=6 cells; carries the aggregate-cache race-arena lesson (−21%) |
| `L8_fusedaxes` | fastest L=8 at B=1 and B=2048; the fused construction; the robust number at the DRAM wall |
| `L8_radix8` | structurally distinct 3-pass; fastest L=8 at B=32768; shows the NT/cached pick flip live (5654 vs 9674) |
| `L13_direct` | fastest L=13 at B=1 and B=512 |
| `L13_rader` | fastest L=13 at B=8192 by 1.31×; different structure; the entry that must close the FFTW gap |
| `L17_matrixsimd` | fastest L=17 at B=1 and B=256; the spin-pool and NT-inversion source most of the panel borrowed |
| `L17_winograd` | fastest L=17 at B=4096 by 1.72×; the round's most robust large-batch result (129 GB/s, sd 0.2%) |
| `L23_matrixsimd` | fastest L=23 at B=1 and B=128; the round's most-cited record (pool, flag barrier, NT, arena fill) |
| `L23_rader` | rule 2 verbatim: structurally distinct (Rader vs dense conj-symmetric), within 1.03× at B=128 |
| `L36_pfa` | fastest L=36 at B=1, 1.49× MKL, sd 0.09% — the most robust B=1 result in the round |
| `L36_mixedradix` | fastest L=36 at B=32 and B=512 |
| `L36_pencilfused` | rule 3: the fused construction at 1.01× parity inside aggregate cache and 1.73× behind at the DRAM wall, with a pre-registered prediction that missed 3× — the §4.3 datum |
| `L45_pfa` | fastest L=45 at B=256 by 2.94× over its sibling |
| `L45_mixedradix` | fastest L=45 at B=1 and B=16; rule 3 at B=256 (arena 25.4 vs driver 79.07 µs/t) |
| `L64_blocked` | fastest L=64 at B=1 and B=128 |
| `L64_radix8` | fastest L=64 at B=8; pre-registered the MKL loss and reproduced it on the node |

**Not promoted (3), all on the near-duplicate clause:**

* **`L6_unrolled`** — same Good-Thomas PFA 2×3 as `L6_pfa`, slower at all three cells
  (1.05×/1.24×/1.83×), no distinct structure. Its one distinguishing datum is worth recording
  here rather than in an exemplar: it is the *stable* L=6 B=4096 entry (47.67/49.40/49.40 µs,
  `sd` ≤0.42%) against `L6_pfa`'s 38.59/38.83/59.02, which is how the pick instability in §3.2
  was diagnosed.
* **`L8_batchsimd`** — identical arithmetic and op count to `L8_radix8` (1248 vector FP + 896
  shuffles), slower at all three cells, and its lesson is already carried by `L8_radix8`'s
  record, which pre-registered the exact failure mode. The killing number, for the record:
  **9674–9706 µs at L=8 B=32768, `sd` 0.05%, sitting exactly on MKL's 9675 and 1.68× behind
  both siblings, because its create-time race never selected NT stores at the streaming cell.**
* **`L17_rader`** — its own description states the kernel is taken from `L17_winograd`, so it
  is a near-duplicate of a promoted entry; it is fastest at no cell and its B=1 headline is a
  1-in-3 lucky process (§3.2). Its useful contribution is a negative one worth stating: at
  32 cores, as at one, **Rader-17 loses to both the dense conjugate-symmetric form and the
  hand-derived Winograd module** (1.28× / 1.09× / 2.38× behind across the three cells), which
  closes LITERATURE §4.2(a) in the same direction phase 1 did, in the threaded regime as well.

PROMOTE: L6_pfa L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_pfa L36_mixedradix L36_pencilfused L45_pfa L45_mixedradix L64_blocked L64_radix8
