# VERDICT — panel round `panel_r2`

Monitor's report. Measured on the exclusive benchmark node `p55n3`
(Intel Xeon Gold 5218, Cascade Lake-SP, 2.30 GHz base, AVX-512 with licence
downclocking, 1 MiB L2/core, 22 MiB L3, 64 cores, `powersave`, gcc 11.4.0),
SLURM job 438476, 2026-08-21T18:46:15-04:00. Previous round `panel_r1` was
measured on `p51n1` — same CPU model, **different physical node**. That matters;
see §2.

Correctness for every backend was checked by `check.py` against `numpy.fft.fftn`
on the output of the **timed** run itself (`sweep.sh:79`), i.e. the same process,
the same binary and the same plan that produced the score. There is no separate
"check mode" an entry could pass while its timed path did something else.

---

## 1. Headline per geometry

Fastest correct panel entry vs. the best library baseline, per case.
"Best library" is the fastest of FFTW ×3 planner levels, MKL 2022.0.2, MKL 2026.1
and ducc0 0.41 in that same case.

### L = 6 (216 complex doubles/volume)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | `L6_unrolled` **0.218 µs** (38.35 GF/s) | `mkl_dfti` 0.369 µs | **1.69×** |
| batched B=64 | `L6_unrolled` **0.214 µs**/transform (39.19 GF/s) | `mkl_dfti` 0.392 µs | **1.83×** |
| batched B=4096 | `L6_unrolled` **0.384 µs**/transform (21.79 GF/s) | `mkl_dfti` 0.548 µs | **1.43×** |
| batched B=32768 | `L6_unrolled` **0.572 µs**/transform (14.65 GF/s) | `mkl_dfti` 0.684 µs | **1.20×** |

`L6_unrolled` took first place in all four cases. `L6_pfa` is second everywhere
(0.221 / 0.223 / 0.393 / 0.616 µs) and also beats every library in every case.

### L = 8 (512 complex doubles/volume)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | `L8_fusedaxes` **0.573 µs** (40.22 GF/s) | `mkl_dfti` 0.651 µs | **1.14×** |
| batched B=64 | `L8_batchsimd` **0.636 µs**/transform (36.21 GF/s) | `mkl_dfti` 0.687 µs | **1.08×** |
| batched B=2048 | `L8_batchsimd` **1.205 µs**/transform (19.11 GF/s) | `mkl_dfti` 1.325 µs | **1.10×** |
| batched B=16384 | `L8_batchsimd` **1.557 µs**/transform (14.79 GF/s) | `mkl2026_dfti` 1.784 µs | **1.15×** |

B=2048 and B=16384 are the news: in `panel_r1` MKL held both (1.338 and 1.772,
ahead of every panel entry). This is the first round in which the panel leads
L = 8 in **every** regime. B=1 remains a near-tie — 0.573 / 0.583 / 0.598 across
the three entries against run spreads of 1.2 / 1.3 / 0.8 %.

### L = 17 (4913 complex doubles/volume)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | `L17_matrixsimd` **16.751 µs** (17.98 GF/s) | `fftw3_measure` 81.761 µs | **4.88×** |
| batched B=8 | `L17_matrixsimd` **18.661 µs**/transform (16.14 GF/s) | `fftw3_measure` 81.951 µs | **4.39×** |
| batched B=256 | `L17_winograd` **24.031 µs**/transform (12.53 GF/s) | `fftw3_measure` 83.412 µs | **3.47×** |
| batched B=2048 | `L17_winograd` **24.603 µs**/transform (12.24 GF/s) | `fftw3_patient` 84.077 µs | **3.42×** |

Against MKL specifically the margin is 5.90× (B=1) to 4.14× (B=2048). L = 17 is by
a wide margin the geometry where the panel's advantage over every library is largest.

### L = 36 (46656 complex doubles/volume)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | `L36_pfa` **119.266 µs** (30.34 GF/s) | `mkl_dfti` 162.396 µs | **1.36×** |
| batched B=4 | `L36_pfa` **128.460 µs**/transform (28.17 GF/s) | `mkl_dfti` 174.138 µs | **1.36×** |
| batched B=32 | `L36_pfa` **202.746 µs**/transform (17.85 GF/s) | `mkl_dfti` 259.771 µs | **1.28×** |
| batched B=256 | `L36_pfa` **238.796 µs**/transform (15.15 GF/s) | `mkl_dfti` 307.771 µs | **1.29×** |

`L36_pfa` swept all four. In `panel_r1` MKL held B=256 outright (246.319 vs the best
panel entry's 247.435); the panel now leads L = 36 in every regime too. **Caveat: read
§2 before trusting the size of the B=32 / B=256 margin** — MKL itself moved by +18 %
and +25 % between the two rounds in exactly those two cells.

---

## 2. What changed since `panel_r1`, per geometry — and what regressed

### The node changed, and it is not neutral

`panel_r1` ran on `p51n1`, `panel_r2` on `p55n3`. MKL is a fixed binary that cannot
have changed between rounds, so it is a control. It is flat almost everywhere:

| control cell | r1 | r2 | Δ |
|---|---|---|---|
| `mkl_dfti` L=6 B=32768 | 0.698 | 0.684 | −2.0 % |
| `mkl_dfti` L=8 B=16384 | 1.811 | 1.826 | +0.8 % |
| `mkl_dfti` L=17 B=2048 | 108.269 | 101.753 | −6.0 % |
| `mkl_dfti` L=36 B=1 | 163.554 | 162.396 | −0.7 % |
| `mkl_dfti` L=36 B=4 | 174.408 | 174.138 | −0.2 % |
| **`mkl_dfti` L=36 B=32** | **220.552** | **259.771** | **+17.8 %** |
| **`mkl_dfti` L=36 B=256** | **246.319** | **307.771** | **+24.9 %** |
| **`mkl2026_dfti` L=36 B=32** | **228.248** | **268.035** | **+17.4 %** |
| **`mkl2026_dfti` L=36 B=256** | **251.739** | **318.020** | **+26.3 %** |

FFTW and ducc0 in the same two cells are flat (`fftw3_patient` B=32 −1.0 %, B=256
+3.7 %; `fftw3_measure` +0.6 % / +0.3 %; `ducc0` +2.1 % / −0.4 %) — but they run at
377–707 µs there, 1.5–2.7× slower than MKL, so they are not near whatever ceiling
MKL is hitting. **Conclusion: there is a real, node-localised, L=36-batched-only
shift of roughly +18…+25 % affecting the fastest backends only.** It is not a
DRAM-bandwidth story in general (L=17 B=2048 streams 307 MiB and is unaffected);
L = 36 is the only geometry whose single volume, 746 KB, is L2-scale on this part.
Every L = 36 B=32 / B=256 cross-round comparison below is contaminated by it and I
flag each one.

### L = 6 — clean sweep, no regressions

| entry | B=1 | B=64 | B=4096 | B=32768 |
|---|---|---|---|---|
| `L6_unrolled` | 0.219 → **0.218** (−0.5 %) | 0.219 → **0.214** (−2.3 %) | 0.514 → **0.384** (**−25.3 %**) | 0.760 → **0.572** (**−24.7 %**) |
| `L6_pfa` | 0.219 → 0.221 (+0.9 %) | 0.223 → 0.223 (0 %) | 0.392 → 0.393 (+0.3 %) | 0.631 → 0.616 (−2.4 %) |

`L6_unrolled` closed a 31 % / 20 % large-batch deficit and took the lead outright.
`L6_pfa`'s +0.9 % at B=1 is inside its 1.9 % run spread — not a regression.
**No regressions at L = 6.**

### L = 8 — batched solved, non-batched went slightly backwards

| entry | B=1 | B=64 | B=2048 | B=16384 |
|---|---|---|---|---|
| `L8_batchsimd` | 0.573 → **0.598** (**+4.4 %**) | 0.640 → 0.636 (−0.6 %) | 1.432 → **1.205** (−15.9 %) | 1.782 → **1.557** (−12.6 %) |
| `L8_fusedaxes` | 0.570 → 0.573 (+0.5 %) | 0.625 → 0.638 (+2.1 %) | 1.548 → 1.503 (−2.9 %) | 1.899 → 1.614 (−15.0 %) |
| `L8_radix8` | 0.576 → 0.583 (+1.2 %) | 0.682 → 0.706 (**+3.5 %**) | 1.526 → 1.526 (0.0 %) | 2.018 → 1.778 (−11.9 %) |

**Regressions: `L8_batchsimd` at B=1 (+4.4 %, against run spreads of 2.2 % and 0.8 % —
real), and `L8_radix8` at B=64 (+3.5 % against a 3.4 % spread — marginal).** Both
entries spent the round on *arithmetic and register-copy elimination* — batchsimd
removed the `vpermt2pd` index-vector network, radix8 cut 54 → 52 codelet instructions
and eliminated 192 register copies per volume — and both got *slower* at B=1 on the
scoring machine. See §4.

### L = 17 — improved in all twelve cells, no regressions

| entry | B=1 | B=8 | B=256 | B=2048 |
|---|---|---|---|---|
| `L17_matrixsimd` | 16.880 → **16.751** (−0.8 %) | 21.626 → **18.661** (−13.7 %) | 27.356 → 26.067 (−4.7 %) | 29.281 → 27.266 (−6.9 %) |
| `L17_winograd` | 18.259 → 18.247 (−0.1 %) | 20.532 → 20.211 (−1.6 %) | 26.737 → **24.031** (−10.1 %) | 30.586 → **24.603** (**−19.6 %**) |
| `L17_rader` | 20.883 → 19.212 (−8.0 %) | 21.803 → 20.365 (−6.6 %) | 25.984 → 24.394 (−6.1 %) | 27.863 → 26.635 (−4.4 %) |

`L17_rader` also improved its accuracy, 4.1e-16 → 3.1e-16, because it replaced its
own kernel with `L17_winograd`'s. **The three L = 17 entries have now converged on
one 296-instruction module** (see §5). `L17_matrixsimd`'s r1 B=8 figure carried a
7.7 % spread, so its −13.7 % is partly the r1 number being soft.

### L = 36 — the winner and the runner-up swapped, and the ex-winner regressed

| entry | B=1 | B=4 | B=32 | B=256 |
|---|---|---|---|---|
| `L36_pfa` | 225.342 → **119.266** (**−47.1 %**) | 228.956 → **128.460** (−43.9 %) | 294.632 → **202.746** (−31.2 %) | 312.002 → **238.796** (−23.5 %) |
| `L36_pencilfused` | 194.172 → 125.066 (−35.6 %) | 198.740 → 151.519 (−23.8 %) | 329.308 → 241.425 (−26.7 %) | 354.218 → 283.910 (−19.8 %) |
| `L36_mixedradix` | 118.441 → **120.322** (+1.6 %) | 128.353 → **129.745** (+1.1 %) | 204.655 → **231.373** (**+13.1 %**) | 247.435 → **261.463** (+5.7 %) |

**`L36_mixedradix` regressed in all four cells and lost first place in all four.**
The B=32 regression (+13.1 % against a 0.8 % run spread) is far outside noise. Its
B=1 and B=4 code paths were *unchanged* this round by its own record, so +1.6 % /
+1.1 % there is node noise or the node change. Its B=32 / B=256 cells are precisely
the ones where the new NT-store path was enabled **and** the ones where MKL moved
+17.8 % / +24.9 %. I cannot separate the two from this round's data. What I can say:
mixedradix predicted B=256 → 170–185 µs and B=32 → 150–165 µs and delivered 261.5 and
231.4, so the change did not do on the node what it did on wallaby regardless of which
factor dominates. **This needs one control run next round** — the entry ships an
`FFT36_NT` env override for exactly this; run B=32 and B=256 with `FFT36_NT=0` and
`FFT36_NT=1` back to back on the scoring node.

`L36_pfa`'s 47 % gain at B=1 came from throwing away its three-pass slab-blocked
architecture for a two-sweep per-x-plane structure. `L36_pencilfused`'s 36 % came from
abandoning split-complex for interleaved.

---

## 3. Failures, non-builds, crashes and missing entries

`build_errors.txt` is present and **empty**. `failures.txt` **does not exist**, meaning
no backend exited non-zero and none hit the 600 s `timeout` in any of its runs.
`results/panel_r2/agents/exits.txt` records **exit=0 for all eleven implementer agents**.

**Correctness: 156 of 156 `c_*.json` verdicts are `"ok": true`.** That is 11 panel
entries + 7 library/floor backends × 4 batch cases × 4 geometries. Every panel entry
was checked at all four of its batch sizes and passed. Worst relative L2 error across
all panel entries is 4.0e-16 (`L36_mixedradix`), against a tolerance of 1e-12 and
against the library-free floor `baseline_matrix` at 3.9e-16 … 8.4e-16. Nothing is
suspiciously accurate-but-fast or fast-but-marginal:

* the most accurate entry, `L8_radix8` at 1.3e-16, is also **not** the fastest at L = 8,
  so accuracy is not being traded for rank;
* the fastest entry at each geometry (`L6_unrolled` 2.4e-16, `L8_batchsimd` 1.9e-16,
  `L17_matrixsimd` 3.3e-16, `L36_pfa` 3.7e-16) is in every case at or better than the
  library median for that geometry.

**No fast wrong answer survived, because none was submitted.**

### One entry is missing outright: `L6_batchsimd`

The panel roster in `panel_round.js:130` defines **twelve** implementers. Eleven ran.

```
{ file: 'L6_batchsimd', L: 6, brief: `Strategy: BATCH-MAJOR SIMD (vectorize ACROSS volumes).
```

Evidence that it is absent, not merely slow:

* no `impl/L6_batchsimd.c` — `find` over the whole repo returns nothing;
* no `strategies/L6_batchsimd.md`;
* no `results/panel_r2/agents/L6_batchsimd.log`, and no line for it in
  `agents/exits.txt` (which lists exactly eleven names) — so the agent did not fail,
  it produced no record at all;
* no row in either round's leaderboard (`grep -c` returns 0 for `panel_r1` and
  `panel_r2`), and no `c_L6_batchsimd_*.json`;
* the string `L6_batchsimd` occurs in exactly one file in the entire repository:
  `panel_round.js`.

**This has now happened silently for two consecutive rounds.** The mechanism is
`panel_round.js:266`, `const entries = built.filter(Boolean)` — `parallel()` yields
`null` for an agent that dies on a terminal error, and `.filter(Boolean)` discards it
without a `log()`. `sweep.sh` then only benchmarks binaries that exist in
`build/<host>/bin`, so a missing source file produces neither a build error nor a
failure line. The round reports as complete.

The cost is not cosmetic: `L6_batchsimd` was the panel's only batch-major
vectorise-across-volumes entry at L = 6, which is one of the two configurations
LITERATURE.md §4.1 asks for by name. **Fix before the next round:** `log()` every
dropped entry in `panel_round.js`, and have `sweep.sh`/`leaderboard.py` diff the roster
against the backends actually built so a missing implementer appears in
`build_errors.txt` rather than nowhere.

---

## 4. Claimed numbers vs. measured numbers

Implementers develop on **wallaby** (Xeon Gold 6448Y, Sapphire Rapids: two 512-bit FMA
units, no meaningful AVX-512 licence penalty, 2 MB L2/core, 60 MB L3, 2-socket, and
shared with the other implementers). They are scored on **Cascade Lake**: one 512-bit
FMA unit, severe AVX-512 licence downclocking, 1 MB L2, 22 MB L3, exclusive. MKL alone
spans ~2.9× between the two, so absolute transfer is not expected. The entries below
are the ones where the *direction* or the *magnitude* is worth recording.

**`L8_batchsimd` — the largest claim/measurement gap of the round, and it is not
explained by the machine.** Claimed: the non-destructive `vshuff64x2` transpose network
took B=1 from 0.629 → 0.306 µs on wallaby, "2.05×", and predicted "**~0.42–0.50 µs at
B=1/B=64**" on the node. Measured on the node: **0.598 µs at B=1, a 4.4 % regression
against its own r1 0.573 µs**, and 0.636 at B=64 (flat). The record does anticipate
part of this — it notes the removed copies are rename-eliminated on real SPR hardware
and only cost front-end slots on the node — but a *2× wallaby win producing a small
node loss* means the mechanism was mostly a Sapphire-Rapids/GCC-codegen artifact, not a
port-pressure win. Attribute to the machine difference **only as far as sign**: the
node has no headroom left on this path, and the prediction was wrong by 20–40 %.

**`L8_radix8` — same story, smaller numbers.** Claimed: 52-instruction codelet
(−48 FP instructions/volume) plus 192 removed register copies gives a new p0 bound of
1248 cycles ≈ 0.543 µs, predicting **0.545–0.555 µs** at B=1. Measured **0.583 µs**,
which is 0.7 % *worse* than its r1 0.576 µs. Its own r1 measurement had already put it
at "~98 % of the port-0 bound", and cutting the bound by 3.7 % moved nothing. Plausible
machine contribution: on wallaby's 2-FMA part the copy removal is free; on the node
the copies were apparently never the limiter either. **The honest reading across both
entries: the p0-instruction-count model does not predict L = 8 B=1 on Cascade Lake.**

**`L8_fusedaxes` — the one L = 8 prediction that landed.** Claimed ~+21 % at the
DRAM-bound size from `prefetcht1` + an L3-relative NT gate, predicting **1.55–1.60 µs**
at B=16384. Measured **1.614 µs** (from 1.899). Its B=2048 prediction ("≤1.43 µs")
missed at 1.503, and it correctly flagged that as the open one.

**`L17_matrixsimd` — a −11.9 % instruction cut bought −0.8 %.** Claimed: the nested
cyclic/negacyclic kernel drops 168 → 148 vector FP ops per chunk, worth ~8 % measured on
wallaby, and — because r1 measured this kernel at 1.045 FP ops/cycle on the node's single
FMA unit — "expect close to the full 12 % there (16.99 → **~15.0 µs**)". Measured
**16.751 µs**, −0.8 %. This is the reverse of the usual machine-difference excuse: the
entry argued the *node* would show more than wallaby, and it showed far less. The
conclusion the next round has to accept is that **L = 17 B=1 on this node is not
FMA-port-bound**; the residual is elsewhere (its own record puts the transposes at
~2.4 µs and counts 94–103 stack references per kernel). The same change did land at
batch: −13.7 % at B=8, −4.7 % / −6.9 % at B=256 / B=2048.

**`L17_rader` — direction right, magnitude short, and the machine explains it.**
Claimed −19.6 % instructions (368 → 296) and projected **~17.5 µs** at B=1. Measured
**19.212 µs**, −8.0 %. The record predicted this discrepancy itself and gave the reason
before the fact: its static counts show the FP saving is partly re-spent in spills and
constant loads (72 → 237 stack moves), and it noted 8 % on wallaby against 19.6 % of FP.
It expected the 1-FMA node to convert more of the FP saving; it converted the same 8 %.

**`L17_winograd` — prediction wrong, outcome good.** It predicted the node's tuner would
pick variant **c4** (256-bit, register-resident) at B=1 and land in 15–17 µs, beating
matrixsimd's 16.88. The node's tuner picked **c8** and scored 18.247 µs — essentially
unchanged. Its *batched* prediction was right and then some: B=2048 30.586 → 24.603.
See §5 for what the tuner actually chose; this is the entry that made its picks readable.

**`L36_pfa` — the round's best-calibrated prediction.** Claimed the rewrite would put
node B=1 at **110–125 µs** ("parity or slightly better" than mixedradix's 118.4).
Measured **119.266 µs**. It also predicted pw4 would still win on the node despite the
licence clock. Both correct.

**`L36_mixedradix` — predicted improvement, measured regression.** Claimed B=256
→ ~170–185 µs and B=32 → ~150–165 µs from the new streaming-store path (1.4× measured
on wallaby, 179.6 → 126.6). Measured **261.463** and **231.373**, i.e. slower than its
own r1 code. Partial attribution to the machine is defensible here and only here —
MKL moved +24.9 % / +17.8 % in the identical cells — but even after crediting the full
MKL-sized shift the entry did not gain what it claimed. Settle it with the `FFT36_NT`
control run.

**`L6_pfa` — the round's one fully closed prediction loop, and it is the model to
copy.** It cross-compiled with `-march=cascadelake`, grepped the emitted assembly
(86 `ymm16–31` references, 1 `rsp` reference = spill-free under EVEX), argued from
front-end slots rather than FP ports that the *fused* variant should take B=1 on the
node by ~4 % — an effect wallaby's 6-wide front end structurally cannot show — and
stated that the leaderboard's `variant=` string would settle it. The node's tuner
picked `variant=2` (fused) at B=1. `L6_unrolled` independently picked `fused` at B=1
too. Prediction stated in advance, mechanism identified, confirmed by two entries.

---

## 5. Which LITERATURE.md §4 open question moved

### Primary: §4.8 gap 6 — "No AVX-512 measurement anywhere in this corpus. Measure it on the node."

This round produced the corpus's first direct, same-kernel, width-A/B measurement on a
downclocking AVX-512 server part, because `L17_winograd` and `L6_pfa` plumbed their
plan-time tuner's pick into `fft3d_description()`. Reading the per-case picks out of the
raw `t_*.json`:

| entry | B | picked variant | width |
|---|---|---|---|
| `L17_winograd` | 1 | `c8` | **512-bit** |
| `L17_winograd` | 8 | `a8` | **512-bit** |
| `L17_winograd` | 256 | `a4` | **256-bit** |
| `L17_winograd` | 2048 | `b4` | **256-bit** |
| `L17_matrixsimd` | 1 | 512-bit, C parked in L1 | 512-bit |
| `L17_matrixsimd` | 8 / 256 / 2048 | 512-bit, C live | 512-bit |

**The answer is regime-dependent and it inverts.** On Cascade Lake with one 512-bit FMA
unit, 512-bit wins while the working set is L1/L2-resident and the kernel is
compute-bound (B=1, B=8), and **256-bit wins once the kernel streams** (B=256 at 38 MiB,
B=2048 at 307 MiB) — exactly the shape §4.8 gap 6 predicts from licence downclocking,
measured for the first time here. `L8_radix8`'s r1 result (the node's self-timer chose
the AVX-512 backend at L = 8, 0.576 µs) still stands, and `L36_pfa` predicted and got
pw4 (512-bit) on the node. Note the corpus's own framing — "the shift is not simply 2×
wider" — is vindicated: this is not a width preference, it is a residency-dependent one.

The same tuner reporting settled two more node-vs-wallaby inversions:

* **Non-temporal stores lose at L = 6 on the node, at every batch size.** `L6_unrolled`
  picked `fused` (B=1), `fused_pf` (B=64, B=4096) and `3pass_pf` (B=32768) — no NT
  variant anywhere, despite `3pass_nt_pf` winning B=32768 on wallaby by 1.6× and the
  record explicitly predicting "an NT+pf variant to win there". `L6_pfa` picked
  `variant=2` (B=1) and `variant=6` = fused+prefetch-1 with **normal stores**
  (B=64/4096/32768), where wallaby chose `v8` = fused+**nt**+pf1. Two independent
  entries, same inversion.
* **Cross-volume prefetch loses at L = 17 on the node.** `L17_winograd` reports
  **`pf=0` at every batch size including B=2048**, where the same candidate was worth a
  clean −4.4 % on wallaby. Its 19.6 % gain at B=2048 therefore came entirely from the
  stage-2 tuner fix, not from the prefetch — and at B=256 the node picked `a4`, a
  *round-1* kernel, so that cell's −10.1 % is pure search, no new code.

### Also moved

**§4.1 (register pressure of batch-vectorised codelets).** Answered at L = 6 and L = 8,
by the method §07 §7.8 asked for — count stack traffic in the generated assembly before
believing any timing. `L6_pfa`: 1 `rsp` reference at `-march=cascadelake` (the
callee-save restore), spill-free. `L8_batchsimd`: "zero stack traffic in both kernels".
`L8_radix8`: "0 `vmovapd zmm,zmm`, 0 stack spills/reloads". §01's "for AVX2 the
batch-major form of even n = 6 and n = 8 will spill a little" **does not hold on any
AVX512VL part even in 256-bit mode**, because `ymm16–31` are available. The `2L`
data-only bound was the right budget after all, for the wrong reason. And the binding
resource at L = 6 B=1 turned out to be **front-end slots** (fused 1834 vs 3-pass 2050
uops/volume against a 486-cycle FP floor on a 4-wide machine), which is why `fused`
won the node and tied on wallaby.

**§4.2 (L = 17: dense-symmetric, Rader, or a hand-derived Winograd module?).** Moved
decisively — and the panel destroyed a data point doing it. Sub-question (b), "does the
symmetric/antisymmetric convolution split add anything on top?", is answered **yes**:
`L17_winograd`'s 296-instruction cyclic/negacyclic module was adopted verbatim by
`L17_rader` (368 → 296, −19.6 %) and in nested form by `L17_matrixsimd` (168 → 148 ops
per chunk, −11.9 %), and both improved. §4.2's sub-question (c) — no exact op count for
a full 17-point Winograd module exists in the corpus — is now moot for our purposes:
we have a measured 296 instructions / 488 flops, hand-derived and verified.

But **sub-question (a), "which of dense-symmetric and Rader-17 wins on this hardware,
batch-vectorised?", can no longer be answered by this panel.** `L17_rader` is Rader in
name only; its own description string now reads "kernel adopted from L17_winograd", and
its two-FFT16 Rader kernel — the thing the roster asked it to build — has been deleted.
All three L = 17 entries now compute the same 296-instruction module and differ only in
pass structure and data layout (16.751 / 18.247 / 19.212 µs at B=1). That convergence is
itself the finding — arithmetic at L = 17 is closed, and the remaining 15 % spread is
layout — but the next panel should know the Rader comparison is gone, not settled.

**§4.3 (is axis fusion worth 3× or 3 %?).** The most quantitatively useful movement
after §4.8. `L36_pfa` replaced three passes through an 830 KB slab-blocked intermediate
with two fused sweeps and went **225.342 → 119.266 µs at B=1, 1.89×**; its own r1 phase
table attributes 108 of 304 µs to the `T` write and read. That is far above TurboFNO's
3–5 % prior. But it confirms Tolmachev's rule from §07 §1.6 rather than overturning
§07 gap 7: what was deleted was a *volume round-trip through L2/L3*, not a fusion of
already-L1-resident passes. The L1-resident case measured the other way — at L = 6,
where all three passes are L1-resident, `fused` beats `3pass` by only a few percent at
B=1 and **loses to `3pass_pf` at B=32768**, because the separate z pass emits its stores
in tight bursts. **So: ~1.9× when fusion deletes a cache-level round trip, and ±a few
percent with a sign flip when it does not.** Both halves of the corpus were right about
different regimes.

**§4.4 (split vs interleaved complex).** The corpus asked for one microbenchmark before
committing all four sizes to one layout; this round produced a geometry-resolved answer
instead. At L = 36, `L36_pencilfused` rewrote from split to interleaved and went
194.172 → 125.066 µs at B=1, calling split "this round's headline documented dead end
for L = 36"; `L36_pfa` independently went interleaved and won. At L = 8, all three
entries use split-complex and all three beat every library. **The layout decision is
geometry-specific, not global** — split where the batch is batch-major and 8 divides the
vector width, interleaved where lanes are a spectator axis and the vector length does
not divide the transform length. §04's blanket "split-complex batch-minor across the
board" recommendation is now measured to be wrong at L = 36.

**§4.6 (model versus search for the instruction schedule).** Search won, twice, with
numbers: `L17_winograd`'s B=256 went 26.737 → 24.031 µs with **no kernel change** (the
node picked `a4`, a round-1 variant) purely by fixing *when and on what working set* the
tuner measures. The round also produced two reusable tuner-protocol failures, both
documented with the number that killed them: sequential/blocked candidate timing
mis-ranks under drifting load (`L8_batchsimd` picked a 1.5× slower mode at B=64;
`L6_pfa` measured a 21 % mis-pick), and a tuning arena that fits the dev machine's L3
mis-picks the store policy for a streaming run (`L36_pfa`: 16-volume arena chose cached
stores, the real run wanted NT, 45.5 vs 32.4 ms).

### Did not move

**§4.5 (does L = 8 need padding, and where?).** Untouched. Nobody measured
(Nx,Ny) = (8,8) against (9,9) at NB = 8, or NB = 4, or ran
`perf stat -e L1-dcache-load-misses`. `L8_batchsimd` explicitly reclassified the
question as BATCH-mode-only and therefore "dead unless" its mode machinery survives.
This is the loudest unmeasured claim in the corpus at exactly the geometry where the
panel is now closest to a wall — see §6.

**§4.7 (vector-radix).** Not attempted by anyone; the corpus's verdict stands unchallenged
and nothing this round contradicts it.

---

## 6. The single highest-value thing the next round should attack, per geometry

### L = 6 — establish the clock, then decide whether L = 6 is finished

B=1 has sat at 0.218–0.219 µs for two rounds across two independent implementations,
and the batched regime is now solved (all four cells swept, NT stores ruled out on the
node). Whether anything remains depends entirely on a number nobody has measured: the
node's actual core frequency under this code. The two L = 6 records disagree about it
and reach opposite conclusions — `L6_pfa` computes 0.219 µs × 2.3 GHz = 504 cycles
against a 486-cycle FP-port floor and concludes ≤4 % headroom; `L6_unrolled` observes
that if the node turbos to ~3.9 GHz it is ~850 cycles against the same floor, i.e. ~40 %
headroom and a real structural target. **One `perf stat -e cycles,ref-cycles` (or an
`aperf/mperf` read) on the node during an L = 6 B=1 run settles whether to spend another
round here at all.** It costs one job. Do it before any further L = 6 work.

### L = 8 — measure the padding question (§4.5), and stop cutting instructions

This is the geometry where the round's effort most clearly failed. Three entries cut FP
instructions and register copies; B=1 moved from 0.570 to 0.573 and one entry regressed
4.4 %. Three implementations within 1.2 % for two consecutive rounds is a wall, and two
rounds of evidence now say it is not the FP port. The one structural hypothesis at
L = 8 that has never been tested is the corpus's loudest: the naive 8³ z-stride is
**exactly 4096 B = one L1 set**, Bailey's `E = 1/R = 1/64` worst case, which §04 §7.3
calls padding "mandatory" for — and §05 identifies the volume-to-volume stride
(8192 B ≡ 0 mod 64 sets) as a second instance of the same pathology. **Build
(Nx,Ny) = (9,9) alongside (8,8), at NB = 8 and NB = 4, and run
`perf stat -e L1-dcache-load-misses,cycles` at B = 1, 64, 2048.** If padding does
nothing, L = 8 B=1 is genuinely at its floor and the panel should say so and move on;
if it does something, it is the only untapped lever left.

### L = 17 — fuse passes 2+3, delete the scratch round-trip

`L17_matrixsimd` cut 11.9 % of vector FP operations and gained 0.8 % at B=1. That is a
model failure, not a tuning problem: L = 17 B=1 is not FMA-port-bound on this node.
Meanwhile the round's single largest measured gain anywhere — `L36_pfa`'s 1.89× — came
from deleting one volume round-trip through the cache hierarchy. **`L17_winograd`'s own
"Next" item 2 is the same move at L = 17: fuse passes 2 and 3 over `kx` blocks, removing
~314 KB/volume of scratch traffic through the node's 1 MiB L2.** Its record estimates
~5 % at B=1 and more at batch; given what the identical structural change bought at
L = 36, that estimate is probably low. Do that before any further arithmetic. Secondary,
and cheap: `L17_rader`'s two requested one-flag node A/Bs (`L17R_FORCE_VW=8`, and
`-DL17R_PIN_CONST` — the latter is specifically a 2-load-port-Cascade-Lake hypothesis
that a 3-load-port Sapphire Rapids cannot test).

### L = 36 — first the control run, then overlap the streaming regime

Two things, in order.

**(a) Settle the `L36_mixedradix` regression, one job.** Run B=32 and B=256 on the
scoring node with `FFT36_NT=0` and `FFT36_NT=1` back to back. If `FFT36_NT=0` recovers
r1's numbers, the new store path is at fault and should be reverted or re-gated; if it
does not, the +13 % is the node and the MKL control explains it. Right now the round
cannot tell, and leaving it unresolved risks the next round optimising against a phantom.

**(b) The real optimisation target is streaming overlap, and it is worth ~1.9×.**
L = 36 compulsory traffic is 1.49 MB/volume (in + out). `L36_pfa` at B=256 achieves
238.796 µs = **6.25 GB/s** single-core. In the same round, `L6_unrolled` at B=32768
achieves **12.1 GB/s** and `L8_batchsimd` at B=16384 achieves **10.5 GB/s** on the same
node. L = 36 is leaving roughly half the achievable single-core streaming rate on the
floor. Its B=1 time is 119.3 µs and its B=256 time is 238.8 µs, so 119.5 µs per volume
is pure un-overlapped memory time; if the reads and writes were hidden under the
compute at the rate L = 6 already demonstrates, the ceiling is
max(119.3, 1.49 MB / 12.1 GB/s = 123 µs) ≈ **123 µs**, i.e. ~1.9× available at B=256.
Nobody has built a cross-volume software pipeline at L = 36 — `L36_pfa`'s prefetch is
one cache line ahead *within* phase 2. **Overlap volume b+1's phase-1 reads with volume
b's phase-2 compute.** That is the largest single quantified gap on the whole board.

### Cross-cutting, and cheap

1. **Fix the silent implementer drop** (§3). Two rounds down one of twelve, unreported.
2. **Report every tuner's pick through `fft3d_description()`.** Six of eleven entries
   ship a plan-time tuner and explicitly asked the monitor to read its verdict off the
   leaderboard — all three L = 36 entries and all three L = 8 entries — and none of them
   plumbed the pick through. Their strings are static. The entries that *did*
   (`L6_pfa`, `L6_unrolled`, `L17_winograd`, and partly `L17_matrixsimd`) produced this
   round's only decisive answers, including the entire §4.8-gap-6 result in §5. This is
   the highest-value-per-line change available to the panel.
3. **`leaderboard.py:33` overwrites `descriptions[name]` per case**, so even a
   correctly-reporting entry shows only one arbitrary case's pick in `leaderboard.txt`;
   the per-case strings survive only in the raw `t_*.json`, which the round deletes
   nothing of but the leaderboard never surfaces. Print the description per case, or
   have the monitor emit a picks table.
4. **Pin the physical node across rounds**, or record it prominently. The `p51n1` →
   `p55n3` move cost this round a clean read on the L = 36 batched cells (§2).

---

## 7. What to keep

Applying `docs/CURATION.md`. Note that `exemplars/` is currently empty — nothing was
promoted from `panel_r1` — so this is the panel's first promotion and there is no
prior exemplar for the near-duplicate rule to measure against. Every entry below has a
complete strategy record, and every one beats the best library baseline in every case
of its geometry.

**Rule 1 — fastest correct entry per geometry (always):**

* **`L6_unrolled`** — first in all four L = 6 cases; 1.20–1.83× over MKL.
* **`L8_batchsimd`** — first at B=64, B=2048, B=16384; the first panel entry ever to
  take L = 8 B=2048 from MKL (1.205 vs 1.325).
* **`L17_matrixsimd`** — first at B=1 (16.751 µs) and B=8; 5.90× over MKL at B=1.
* **`L36_pfa`** — first in all four L = 36 cases; 1.28–1.36× over MKL, and the first
  panel entry to take L = 36 B=256 from MKL.

**Rule 2 — a structurally different runner-up that is close:**

* **`L8_fusedaxes`** — leads B=1 outright (0.573 vs 0.598), and within 2 % at B=64.
  Genuinely different structure: all three axes fused in L1 with the spatial axis in the
  lanes, versus batchsimd's pass-structured lane-per-x. It also owns the round's one
  landed L = 8 prediction (`prefetcht1` vs `prefetcht0`, +21 % at the DRAM size) and the
  measurement that `t0` and `t1` invert between the two machines — a fact the next panel
  must not rediscover.
* **`L17_winograd`** — leads B=256 (24.031) and B=2048 (24.603, −19.6 % on the round),
  within 9 % at B=1. Different structure from matrixsimd: split re/im with three rotating
  passes and a six-variant × two-width tuner, versus interleaved lanes-are-lines nested
  matvec. It is also the origin of the 296-instruction module the whole L = 17 field now
  runs, and the only entry that reported its per-case tuner picks — which is where the
  §4.8-gap-6 result in §5 comes from.

**Rule 3 — instructive failure, whose record documents the number that killed it:**

* **`L36_mixedradix`** — round 1's winner at every L = 36 case; this round it regressed
  in all four (+1.6 / +1.1 / +13.1 / +5.7 %) and lost every one. Its record documents a
  streaming-store path measured at 1.4× on wallaby (179.6 → 126.6 µs) that delivered
  231.4 and 261.5 on the node against a predicted 150–165 and 170–185. It is still within
  1 % of the winner at B=1 and B=4, so it is also a close runner-up. Keeping the code is
  what makes the `FFT36_NT=0` control run in §6 possible; keeping the record is what
  stops the next panel re-deriving a dev-machine NT threshold.

**Not promoted, with reasons:**

* **`L6_pfa`** — second in all four cases and beats every library, but it is now a
  near-duplicate of `L6_unrolled`: same PFA 2×3, same 2-complex/ymm interleaved lanes,
  same in-register transposes, same next-volume prefetch, same round-robin tuner — the
  two entries borrowed from each other in both directions this round. `L6_unrolled` beats
  it in every cell. Its outstanding contribution (the cross-compiled spill audit →
  front-end argument → confirmed node variant pick, §4) lives in
  `strategies/L6_pfa.md`, which is tracked regardless of promotion.
* **`L8_radix8`** — third or fourth in every cell, and by its own attribution section it
  now runs `L8_batchsimd`'s 52-instruction codelet and `L8_fusedaxes`'s transpose network.
  Promoting it would be promoting a hybrid of the two entries already promoted. Its
  distinctive results (the copy-free interleave-source swap; the enumeration proof that
  the perfect-shuffle lane order is unreachable from a 3-stage non-destructive network)
  are preserved in `strategies/L8_radix8.md`.
* **`L17_rader`** — never first in any cell, and structurally no longer distinct: its
  description string now reads "kernel adopted from L17_winograd" and its own record says
  the module was taken "their derivation, their constants, verbatim". Promoting it would
  be a third copy of the same 296-instruction kernel. Its instructive negatives — the
  plane-pair slab losing in all three variants (11.53 / 11.34 / 11.37 vs 11.01 µs) and
  the `KREG` constant-pinning costing 5 % — are in `strategies/L17_rader.md`. **Flag for
  the roster: the Rader-with-a-16-point-convolution assignment
  (`panel_round.js:171`) no longer has an implementation. If §4.2(a) is still wanted,
  it has to be re-briefed explicitly.**
* **`L36_pencilfused`** — 5 / 18 / 19 / 19 % behind `L36_pfa`, and its structural
  distinction is thinner than it looks now that both entries are two-pass and
  interleaved-complex. Its one durable contribution — split-complex measured as a dead
  end at L = 36 (194.172 → 125.066 µs at B=1 on the rewrite), which is half of the §4.4
  answer in §5 — is a record, not a codebase, and the record is tracked.

Promotion command:

```bash
cd bench/geom
./promote.sh panel_r2 L6_unrolled L8_batchsimd L8_fusedaxes L17_matrixsimd \
                      L17_winograd L36_pfa L36_mixedradix
git add exemplars/panel_r2 strategies results/panel_r2 && git commit
```
