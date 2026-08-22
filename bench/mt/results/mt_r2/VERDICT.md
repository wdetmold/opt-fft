# mt_r2 — monitor's verdict

Round: `mt_r2`, the second multicore round (phase 2).
Scored on: `p55n3`, Intel Xeon Gold 5218 (Cascade Lake, 2 sockets × 16 cores, 1 MiB L2/core,
22 MiB L3/socket, 1× 512-bit FMA unit, DDR4-2666), 32 threads, `OMP_PROC_BIND=close`,
governor `powersave`, gcc 11.4.0, slurm job 438551.
Comparison baseline: `results/mt_r1/leaderboard.txt` (same CPU model, different node `p51n1`).

**The two nodes are interchangeable for this purpose.** Across all 48 MKL cells the r2/r1 ratio
has median 1.0001 and spans 0.968–1.022. Every panel change reported below is therefore code,
not machine. (FFTW spans 0.33–1.57× between the rounds; that is FFTW's planner, and it is why
FFTW is quoted with medians wherever it is load-bearing.)

Statistic used by the harness: **minimum over three independent processes of each process's
minimum sample.** As in mt_r1 this is the single most important thing to know about the round,
and §3.2 is mostly about it.

---

## 0. Bottom line

* **Nothing failed correctness. Nothing crashed. One entry failed to build and is absent from
  the leaderboard: `L13_rader`** (§3.1). All 213 (backend × case) correctness records pass at
  `rel_l2` 1.3e-16 … 8.4e-16 against `numpy.fft` with tol 1e-12; every leaderboard row has a
  matching `c_*.json`; no fast wrong answers to strike.
* The panel beats the best threaded library at **21 of 24 scored cells**, by 1.17× to 5.56×,
  and loses the **same three cells as mt_r1**: L=8 B=32768 (0.92×), L=13 B=8192 (0.61×),
  L=64 B=8 (0.95×). L=64 B=8 nearly closed (0.81×→0.95×); L=13 B=8192 got materially worse
  (0.81×→0.61×) and that is directly attributable to the missing `L13_rader`.
* **Fourteen of nineteen entries improved somewhere; six regressed on stable numbers.** The
  real regressions are `L45_pfa` at B=256 (1.69×), `L36_pfa` at B=512 (1.62×),
  `L36_pencilfused` at B=512 (1.39×), `L17_matrixsimd` at B=4096 (1.36×), `L17_winograd` at
  B=256 (1.31×) and, on typical-case behaviour rather than the scored minimum, `L6_pfa` at
  B=65536 (2.05× on medians).
* **The round's central finding, and it is a good one.** `L8_fusedaxes` shipped an execute-time
  placement probe and published it: at L=8 B=32768, in all three processes,
  `gov{fr=0,nb=1,…}` — the driver's 512 MiB of caller buffers are **100 % on socket 0 and stay
  there** for the whole timed loop, with `numa_balancing` enabled. That **falsifies the mt_r1
  verdict's leading hypothesis** (AutoNUMA migration during timing) *for that cell*, and it
  explains the loss: all three L=8 entries picked a **16-thread single-socket team**, all three
  land within 0.13 % of each other at **94 GB/s**, which is one socket's DRAM. Meanwhile
  L=6 B=65536 sustains **200 GB/s** — two sockets' worth — and `L6_pfa` brackets it on the same
  binary at the same cell: **T=32 → 200 GB/s, T=16 → 85 GB/s**. Two placement regimes exist on
  this machine and the create-time tuners cannot tell which one they are in (§5).

---

## 1. Headline per geometry — fastest correct panel entry vs best library

### The four brief geometries

| L | case | fastest panel entry | best library | ratio |
|---|---|---|---|---|
| **6** | B=1 | **L6_pfa 0.220 µs** (L6_unrolled 0.222, a tie) | mkl_dfti 0.371 µs | **1.69×** |
| 6 | B=4096 | **L6_unrolled 0.0094 µs/t** (38.34 µs/call; L6_pfa 38.37, a tie) | mkl_dfti 0.0210 (85.83) | **2.24×** |
| 6 | B=65536 | **L6_unrolled 0.0346 µs/t** (2269.3 µs/call) | fftw3_patient 0.0665 (4357.1) | **1.92×** (3.50× on medians) |
| **8** | B=1 | **L8_batchsimd 0.557 µs** (L8_fusedaxes 0.558 — a 0.2 % tie) | mkl_dfti 0.652 | **1.17×** |
| 8 | B=2048 | **L8_fusedaxes 0.0263 µs/t** (53.87 µs/call) | mkl_dfti 0.0364 (74.65) | **1.39×** |
| 8 | B=32768 | L8_batchsimd 0.1737 µs/t (5690.8; fusedaxes 5691.0, radix8 5697.8 — a three-way tie) | **fftw3_patient 0.1591 (5214.9)** | **0.92× — LOSS** (1.68× on medians) |
| **17** | B=1 | **L17_matrixsimd 6.163 µs** | mkl_dfti 22.632 | **3.67×** |
| 17 | B=256 | **L17_matrixsimd 0.755 µs/t** (193.3 µs/call) | fftw3_estimate 3.906 (999.9) | **5.17×** |
| 17 | B=4096 | **L17_winograd 1.220 µs/t** (4998.1 µs/call) | fftw3_patient 4.066 (16653.2) | **3.33×** |
| **36** | B=1 | **L36_mixedradix 23.012 µs** | mkl2026_dfti 38.297 | **1.66×** |
| 36 | B=32 | **L36_mixedradix 5.213 µs/t** (166.8 µs/call; L36_pfa 167.5, a tie — and L36_pfa wins on medians) | mkl_dfti 7.465 (238.9) | **1.43×** |
| 36 | B=512 | **L36_mixedradix 9.989 µs/t** (5114.5 µs/call) | fftw3_patient 19.677 (10074.4) | **1.97× — but a 1-in-3 result; 1.07× typical** (§3.2) |

### The four extra geometries the round also measured

| L | case | fastest panel entry | best library | ratio |
|---|---|---|---|---|
| 13 | B=1 | **L13_direct 5.734 µs** | mkl2026_dfti 7.637 | 1.33× |
| 13 | B=512 | **L13_direct 0.309 µs/t** | mkl2026_dfti 0.360 | 1.17× |
| 13 | B=8192 | L13_direct 0.984 µs/t (8061.9) | **fftw3_patient 0.603 (4942.2)** | **0.61× — LOSS** (0.81× on medians) |
| 17 | — | (see above) | | |
| 23 | B=1 | **L23_matrixsimd 11.628 µs** | fftw3_patient 39.272 | 3.38× |
| 23 | B=128 | **L23_rader 2.242 µs/t** (L23_matrixsimd 2.284, 1.02× behind) | fftw3_patient 12.461 | 5.56× |
| 23 | B=2048 | **L23_matrixsimd 5.934 µs/t** (12153.2) | fftw3_patient 12.707 (26024.7) | 2.14× |
| 45 | B=1 | **L45_pfa 57.197 µs** (L45_mixedradix 57.549, a tie) | mkl2026_dfti 84.225 | 1.47× |
| 45 | B=16 | **L45_pfa 15.751 µs/t** | mkl_dfti 37.669 | 2.39× |
| 45 | B=256 | **L45_pfa 45.446 µs/t** (11634.1) | fftw3_patient 56.952 (14579.7) | 1.25× |
| 64 | B=1 | **L64_blocked 126.988 µs** | mkl_dfti 153.300 | 1.21× |
| 64 | B=8 | L64_blocked 76.722 µs/t (613.8) | **mkl_dfti 72.988 (583.9)** | **0.95× — LOSS** |
| 64 | B=128 | **L64_radix8 69.494 µs/t** (8895.2) | fftw3_patient 164.390 (21042.0) | 2.37× |

**Best result of the round:** `L64_radix8` at L=64 B=128 — 8895 µs, 2.11× faster than its own
mt_r1 number, 2.37× the best library, and it did it by changing the gang width from 8 to 4
(`gang-T32-g4-fused-nt` versus r1's `g8`). Runner-up for the honour is `L6_unrolled` at
B=65536: 2269.3 / 2274.6 / 2277.3 µs across three processes, `sd` ≤ 0.4 %, **200 GB/s** —
2.08× its own r1 number and the only large-batch cell in the round that is both fast *and*
reproducible.

---

## 2. What changed since mt_r1, per geometry

**L=6 — the leadership changed hands and the stability argument won.** `L6_unrolled` improved
1.24× at B=4096 and **2.08× at B=65536** (4724 → 2269 µs) by taking `L6_pfa`'s
aggregate-cache arena lesson and shipping `3pass_nt_pf` at T=32; it now takes both batched
cells and it is dead stable (three processes within 0.4 %). `L6_pfa` did not regress on the
scored statistic (B=4096 38.59 → 38.37; B=65536 2586 → 2582) and its B=4096 pick lottery from
r1 is **fixed** — but at B=65536 it got worse where it matters: two of three processes now race
themselves onto **T=16** and read 5327/5330 µs against the winning process's 2582. Its median
went 2598 → 5335, a 2.05× typical-case regression, on a cell its own record flagged as
"page-luck" and declined to fix. B=1 is a 1 % tie between the two and unchanged.

**L=8 — the r1 instructive failure was repaired, the cell is still lost, and we now know why.**
`L8_batchsimd` fixed exactly the thing mt_r1 convicted it of: 9674 → 5691 µs at B=32768, a
**1.70× improvement**, and it now sits at the top of a three-way tie with its siblings
(5690.8 / 5691.0 / 5697.8 — a 0.13 % spread). `L8_radix8` improved 1.24× at B=2048.
`L8_fusedaxes` is flat and shipped the governor. The loss to `fftw3_patient` narrowed
0.93× → 0.92× on minima but the *reason* is now measured, not guessed: see §5.

**L=13 — the geometry went backwards because half of it is missing.** `L13_direct` improved
1.30× at B=8192 (10444 → 8062 µs) and hit its own pre-registered band exactly (predicted
0.85–1.0 µs/vol, measured 0.984). But `L13_rader`, which was the panel's best entry at that
cell in r1 (7998 µs), **failed to build** (§3.1), so the geometry's loss to `fftw3_patient`
widened from 1.24× to 1.63× on minima. B=1 and B=512 are unchanged and still won.

**L=17 — every entry improved at B=1, two regressed elsewhere, and the three-way ranking is now
clean.** B=1: matrixsimd 7.112 → 6.163 (1.15×), rader 9.134 → 6.776 (1.35×, and its r1 number
was a lucky process — the honest improvement is 12.6 → 6.8), winograd 8.998 → 7.604 (1.18×).
`L17_matrixsimd` hit its pre-registered B=1 band (5.5–6.5, measured 6.163) and collapsed its
r1 B=256 cross-process spread as promised. Against that: **`L17_matrixsimd` regressed 1.36× at
B=4096** (2.106 → 2.858 µs/t) and **`L17_winograd` regressed 1.31× at B=256** (0.824 → 1.078),
both on stable numbers. `L17_winograd` holds B=4096 at 1.220 µs/t — identical to r1 — but has
**lost the reproducibility that made it r1's best result**: 8662 / 5029 / 4998 µs, a 1.73×
lottery where r1 had `sd` 0.2 %.

**L=23 — the round's cleanest geometry.** `L23_matrixsimd` improved 1.22× at B=2048 (14817 →
12153) and **took the cell back on a stable number**, resolving r1's "honest tie at ~14.7 ms"
in its favour; it hit all three of its pre-registered bands (B=2048 6.0–6.3 → 5.934;
B=128 2.2–2.3 → 2.284; B=1 10.5–12 → 11.628). `L23_rader` improved 1.19× at B=1 and takes
B=128 by 1.02×. Its apparent 1.18× "regression" at B=2048 is not one — r1's 6.052 was the
lucky process; on medians the two rounds are identical.

**L=36 — internal ranking upended; two of three entries regressed hard at B=512.**
`L36_mixedradix` improved 1.26× at B=1 and now takes all three cells on the scored statistic;
it hit its B=1 ("low 20s", measured 23.012) and B=32 (5.3–5.5, measured 5.213) predictions.
But **`L36_pfa` regressed 1.62× at B=512** (19.465 → 31.569 µs/t) and **`L36_pencilfused`
regressed 1.39×** (24.544 → 34.043), both stable across all three processes. And
`L36_mixedradix`'s B=512 headline is a 1-in-3 result (§3.2).

**L=45 — the r1 instructive failure was repaired and the r1 winner broke.**
`L45_mixedradix` did exactly what mt_r1 told it to: 79.067 → 50.216 µs/t at B=256, a
**1.57× improvement**, closing the arena-fidelity failure. `L45_pfa` **regressed 1.69×** at the
same cell (26.897 → 45.446) while replacing its OpenMP two-sweep with a pinned spin pool. It
still wins all three L=45 cells, and it improved 1.35× at B=16, but B=256 is the round's
largest single regression on a stable number.

**L=64 — the biggest wins of the round.** `L64_radix8` improved **2.11× at B=128** (18808 →
8895) and took the cell. `L64_blocked` improved **1.46× at B=8** (893.8 → 613.8) and all but
closed the r1 loss to MKL (0.81× → 0.95×). Both entries hit every one of their pre-registered
node bands — `L64_radix8` three for three (B=1 120–140 → 136.2; B=8 70–90 → 90.6;
B=128 65–100 → 69.5), `L64_blocked` three for three (B=1 ~128 → 127.0; B=8 ~84 → 76.7;
B=128 70–95 → 90.5). The one blemish is `L64_blocked`'s B=128 pick lottery (§3.2).

---

## 3. Adversarial audit

### 3.1 One entry is missing: `L13_rader` failed to build

**The evidence, in order.**

1. `results/mt_r2/build_errors.txt` — `impl/L13_rader.c` fails at link:
   `undefined reference to l13r_ntcopy`, `l13r_pool_new`, `l13r_pool_del`, `l13r_pool_nap`,
   `l13r_dwell` (11 relocations), preceded by
   `warning: 'l13r_pool_new' used but never defined` at `L13_rader.c:1035-1039` and
   `implicit declaration of function 'l13r_ntcopy'` at `L13_rader.c:1861`.
   `make: *** [Makefile:48: build/p55n3/bin/L13_rader] Error 1`. It is the only build error in
   the file; every other implementation and all six library backends compiled.
2. `results/mt_r2/agents/exits.txt` — `L13_rader exit=134`. It is the only non-zero exit of the
   nineteen. `agents/L13_rader.log` ends with
   `ASSERTION FAILED: MemoryExhaustion: Crash intentionally because memory is exhausted` from
   the agent runtime's allocator — the implementer process was killed mid-edit.
3. `impl_2/L13_rader.c` grew 1684 → 1904 lines against `impl_1`. The five undefined symbols are
   exactly the create-time spin pool its own mt_r1 record's "Next round" item 1 asked for
   ("a create()-time **spin-pool with a sense-reversing barrier** to beat libgomp region
   entry"). The call sites and forward declarations landed; the definitions never did.
4. `strategies/L13_rader.md` **has no `Round mt_r2` section** — it is the only strategy record
   in the directory that was not updated this round. There is no r2 measurement, no prediction,
   and no account of what was attempted.
5. Consequence in the leaderboard: 213 correctness records instead of 216; `L13_rader` is
   absent from all three L=13 cells; the geometry's B=8192 loss to `fftw3_patient` widened from
   1.24× to 1.63×.

This is a tooling casualty, not a design failure, but the round has to score what it can build.
`L13_rader` is **not promoted**: CURATION forbids promoting an entry whose record is missing,
and promoting a non-compiling source file into `exemplars/` would put broken code in front of
the next panel. Its mt_r1 exemplar stands and is where the next implementer should start.

### 3.2 Correctness, crashes, omissions — otherwise clean

* I cross-checked the leaderboard against the raw records programmatically: **24 cases, 213
  correctness records, zero with `ok != true`, zero with `rel_l2 > 1e-13`.** Worst observed is
  8.4e-16 (`baseline_matrix` at L=17 — expected for an O(L⁴) matvec). Best panel entry is
  1.3e-16 (`L8_radix8` at B=1).
* `failures.txt` does not exist — nothing crashed or hung at run time.
* Record count reconciles exactly: 6 libraries × 24 cells = 144, plus `baseline_matrix` at 15
  cells (the harness deliberately skips it at the largest cell of each geometry, and at L=64 it
  runs only at B=1), plus 18 built panel entries × 3 cells = 54. 144 + 15 + 54 = 213. Nothing
  else is silently absent.
* Every timing record has all three processes; no truncated runs.

### 3.3 The pick lottery is worse than in mt_r1, and it now decides three headlines

mt_r1 named this the round's real defect. It was fixed in some places (`L6_pfa` at B=4096,
`L17_matrixsimd` at B=256, `L23_rader` at B=128 all tightened) and got **worse in others**.
Within-process `sd` is 0.1–0.5 % almost everywhere, so the run-spread column is not sampling
noise — it is different processes installing different create-time plans, and the description
strings prove it:

| entry | cell | three process minima (µs/call) | worst/best | the pick that flipped |
|---|---|---|---|---|
| **L6_pfa** | L=6 B=65536 | **2582** / 5327 / 5330 | **2.06×** | `T=32` → `T=16` (same variant `fused_pf_nt_xa_d2`) |
| **L36_mixedradix** | L=36 B=512 | 9866 / 9867 / **5115** | **1.93×** | `v1-vol16-sntp` → `v1-vol32-sntp` |
| **L17_winograd** | L=17 B=4096 | 8662 / 5029 / **4998** | **1.73×** | `var=g4 dyn=2` → `var=i4/h4 dyn=0` |
| **L64_blocked** | L=64 B=128 | **11582** / 17353 / 16459 | **1.50×** | `G=2 dyn=0 sb=0` → `G=16 dyn=1 sb=1` |
| L36_mixedradix | L=36 B=32 | **166.8** / 207.7 / 192.1 | 1.25× | 3 distinct picks |
| L64_radix8 | L=64 B=128 | 10518 / **8895** / 10480 | 1.18× | `g8` → `g4` |
| L36_pfa | L=36 B=32 | 190.3 / **167.5** / 169.8 | 1.14× | 3 distinct picks |
| L36_pencilfused | L=36 B=32 | **176.5** / 199.4 / 199.8 | 1.13× | 3 distinct picks |
| L8_batchsimd | L=8 B=2048 | 62.7 / **56.9** / 58.3 | 1.10× | 3 distinct picks |
| L64_blocked | L=64 B=8 | 667.6 / 622.1 / **613.8** | 1.09× | 3 distinct picks |
| L23_rader | L=23 B=128 | 311.3 / **286.9** / 288.2 | 1.08× | 2 distinct picks |
| L17_winograd | L=17 B=256 | 278.6 / 295.9 / **276.0** | 1.07× | 3 distinct picks |
| L45_pfa | L=45 B=256 | 12324 / **11634** / 12359 | 1.06× | `mtn-pfi` → `mtn-pf0` |

**One headline in §1 must not be quoted as the entry's number:**

* **`L36_mixedradix` at L=36 B=512 (9.989 µs/t, "1.97× over fftw3_patient").** Two of three
  processes read 9866 and 9867 µs (`sd` 0.3–6.6 %) against the scored 5115. The representative
  number is **9.87 ms**, and `fftw3_patient`'s representative at that cell is 10.6 ms, so the
  honest verdict is **a narrow 1.07× win, not 1.97×**. This is the *same* entry, at the *same*
  cell, with the *same* 2-in-3-versus-1-in-3 problem mt_r1 flagged — the polarity has merely
  inverted (r1: 2 good, 1 bad; r2: 1 good, 2 bad).

Two more are load-bearing but survive:

* **`L6_pfa` at B=65536.** Representative 5.33 ms, not 2.58 ms. It still beats every library at
  that cell (MKL 8.01 ms), but at **1.50×, not 3.10×**, and `L6_unrolled` beats it 2.3× on the
  representative number. This is why `L6_unrolled`, not `L6_pfa`, is the L=6 entry to trust.
* **`L17_winograd` at B=4096.** Representative 5.03 ms; the cell is won by 3.3× either way. But
  the entry that mt_r1 called "the only large-batch number in the round that is both fast and
  boring" is no longer boring, and it lost that property while adding `dyn` to its tuner grid —
  `dyn=2` is precisely what the slow process picked.

Being adversarial in both directions: **the statistic also flatters the libraries at two of the
three lost cells.** At L=8 B=32768 `fftw3_patient`'s minima are 5215/5222/5230 but its medians
are 9647/9629/9636 — it alternates between a ~5.2 ms mode and a ~9.6 ms mode and is scored on
the fast one; `L8_batchsimd` returns 5713 µs at the median. On any typical-case statistic the
panel wins that cell by 1.68×. The loss stands as scored, and both facts belong in the record.
At L=13 B=8192 `fftw3_patient` is 4942/4956/10390 with medians 6509/6460/10410 — typically
fast, so that loss is real at 1.24× on medians (1.63× as scored). At L=64 B=8 `mkl_dfti` reads
583.9/586.5/590.2 at `sd` ≤ 0.4 % against `L64_blocked`'s 613.8/622.1/667.6 — **fully real, no
statistical escape**, and now down to 5 %.

### 3.4 Regressions that are stable, reproducible, and self-inflicted

These are not lotteries. All three processes agree; the code got slower.

1. **`L36_pfa`, L=36 B=512: 19.465 → 31.569 µs/t (1.62× worse).** Its r1 winning pick was
   `T=16 pw=4 scratch+nt pf=1`, published in-arena at `nt16=20.30` µs/vol against a driver
   reading of 19.47 — a well-priced pick. This round it dropped the T=16 rows on the reasoning
   that they were "expected dead" and shipped `T=32 inplace pf=7`, published at
   `ip7=24.97` and measured at 31.57. Its own arena's *best* T=32 row (24.14–24.97) is worse
   than the T=16 row it deleted. Pre-registered band: 10–16 µs/vol. Missed by 2×.
2. **`L45_pfa`, L=45 B=256: 26.897 → 45.446 µs/t (1.69× worse).** It replaced the r1 OpenMP
   two-sweep with a pinned spin pool. The interesting detail is that **its arena is now honest**
   — it publishes `mtn-pfi=42.4/45.8` against a driver reading of 45.4, where in r1 the same
   family read 16.9 on wallaby. So the round bought arena fidelity and paid 1.69× of runtime
   for it. Pre-registered band: 25–27 µs/vol. Missed by 1.7×.
3. **`L36_pencilfused`, L=36 B=512: 24.544 → 34.043 µs/t (1.39× worse).** It added
   `istream+pfw+ntad` to the r1 `scratch+nt` pick. Its probe reads **`t32=15.72`** — essentially
   identical to r1's `t32=15.71` — while the driver went 24.5 → 34.0. The probe cannot see the
   change it made: **arena-to-driver mis-pricing is now 2.2×**, up from 1.6× in r1.
   Pre-registered band: 10–14 µs/vol. Missed by 2.4×.
4. **`L17_matrixsimd`, L=17 B=4096: 2.106 → 2.858 µs/t (1.36× worse).** It moved off r1's
   `C parked, pipelined, NT store` to `pw=1, ar=ot` with no NT flag in the description.
   Pre-registered band: 1.2–1.4 µs/t. Missed by 2×.
5. **`L17_winograd`, L=17 B=256: 0.824 → 1.078 µs/t (1.31× worse)**, stable in both rounds
   (r1 `sd` 0.9 %, r2 `sd` 0.3–0.4 %). It moved from `var=h8` to `var=h4` with `dyn=1`.
   Predicted "the win held"; it fell from 3rd-place-by-1.14× to 3rd-place-by-1.43×.
6. **`L64_blocked`, L=64 B=8: still 0.95× behind MKL** — a 1.46× improvement, so this is
   listed for completeness, not as a regression. It is the one cell where a library holds a
   stable lead over the panel on both minima and medians.

### 3.5 One shared mechanism behind items 1–3 and most of §3.3

Every one of the regressions above, and every entry in the §3.3 table, is a **create-time
tuner** choosing between team shapes at a **DRAM-streaming cell**. In each case the arena either
mis-prices the wide team, or prices it correctly only sometimes. §5 gives what I believe is the
mechanism, with the measurement that supports it.

---

## 4. Claimed versus measured

Every implementer develops on **wallaby** (Xeon Gold 6448Y, Sapphire Rapids, 32 cores on one
socket, 2 MiB L2/core, ~60 MiB L3, two 512-bit FMA units, DDR5) and is scored on **Cascade
Lake** (2×16 cores, 1 MiB L2, 22 MiB L3/socket, one 512-bit FMA unit, DDR4, 32 threads across
two sockets). MKL alone spans 2.9× between them, so a 1.5–2.5× wallaby→node factor is the
expected background and is not a criticism.

**Attributable to the machine — no criticism implied:**

| entry | cell | claimed (wallaby) | measured (node) | factor |
|---|---|---|---|---|
| L8_radix8 | B=32768 | 0.0756–0.0767 µs/t | 0.1737 | 2.27× (DDR5 vs DDR4; both at their wall) |
| L8_fusedaxes | B=32768 | 0.0748 | 0.1737 | 2.32× |
| L6_unrolled | B=65536 | 0.0309–0.0315 | 0.0346 | **1.11× — best transfer of the round** |
| L6_pfa | B=65536 | 0.0339 | 0.0394 (scored 0.0339 in one process) | 1.16× |
| L6_pfa | B=4096 | 0.0062–0.0066 | 0.0094 | 1.45× |
| L17_winograd | B=4096 | 0.829 | 1.220 | 1.47× |
| L23_matrixsimd | B=2048 | 2.01–2.06 | 5.934 | 2.9× |
| L36_mixedradix | B=1 | 12.04 µs | 23.012 | 1.91× |
| L45_pfa | B=1 | 28.7–35.8 µs | 57.197 | 1.7× |
| L64_blocked | B=1 | 67.1 µs | 126.988 | 1.89× |
| L13_direct | B=8192 | 0.340 µs/t | 0.984 | 2.89× — **and pre-registered as 0.85–1.0** |

**Predictions that landed, which is the behaviour to reward.** Three entries went three for
three on pre-registered node bands: `L64_radix8` (120–140 / 70–90 / 65–100 → 136.2 / 90.6 /
69.5), `L64_blocked` (~128 / ~84 / 70–95 → 127.0 / 76.7 / 90.5) and `L23_matrixsimd`
(10.5–12 / 2.2–2.3 / 6.0–6.3 → 11.628 / 2.284 / 5.934, with the B=2048 number *beating* the
band). `L13_direct` called B=8192 at 0.85–1.0 µs/vol from a wallaby number 2.9× away and
measured 0.984. `L8_fusedaxes` pre-registered three branching scenarios keyed on a quantity it
had not yet measured, published the quantity (`fr=0`), correctly identified branch (b), and
predicted 0.165–0.174 µs/t on that branch — measured **0.1737**. That is the single most
disciplined piece of forecasting in either round.

**Gaps that are *not* explained by the machine**, because a sibling ran the same cell on the
same node much faster, or because the entry's own arena disagreed with the driver:

1. **`L36_pencilfused`, B=512.** Its own probe read 15.72 µs/vol; the driver read 34.04. A
   **2.2× arena mis-pricing**, on the same probe number that read 15.71 in r1 for a 24.5 µs/vol
   driver result — so the probe has now been wrong by 1.6× and then 2.2× at the same cell, in
   the same direction, twice. This is not a Cascade Lake effect; it is an instrument that
   cannot see the regime it is supposed to price.
2. **`L36_pfa`, B=512.** Arena 24.97, driver 31.57 (1.26× mis-pricing) *and* a sibling
   (`L36_mixedradix`) runs the cell at 9.87 ms typical / 5.11 ms best against its 16.16 ms.
3. **`L45_mixedradix`, B=256.** Predicted 24–30 µs/vol, measured 50.216 — outside the band by
   1.7×, even though the entry improved 1.57×. Its arena reads `vnt32-v2-pfpk=53.3` against a
   driver 50.2, so the arena is now *honest*; the prediction, which was extrapolated from
   wallaby's 17.3–18.3, was not. `L45_pfa` at 45.4 on the same cell is only 1.10× ahead, so
   this is a machine-factor miss, not a mis-tune.
4. **`L17_matrixsimd` and `L17_rader` at B=4096**, both predicted 0.9–1.4 µs/t and both
   measured ~2.2–2.9. `L17_winograd` runs the same geometry at the same cell at 1.220 on the
   same node, so ~2× of each gap is schedule, not machine.
5. **`L23_rader`, B=1**: predicted 6–10 µs, measured 11.854; **B=2048**: predicted
   5.5–6.0 µs/vol, measured 7.167. Its sibling reads 11.628 and 5.934, so the B=2048 gap in
   particular is a real 1.21× intra-geometry deficit.

---

## 5. Which `docs/LITERATURE.md` §4 open question moved, and how

### Primary: **§4.3, "Is axis fusion worth 3× or 3%?" — the re-opened L2↔DRAM clause**

mt_r1 moved this question by supplying the CPU numbers for Tolmachev's rule at the L2↔DRAM gap
and concluding "store order beats pass count, and by much more." **mt_r2 supersedes that with a
sharper and more uncomfortable answer: at the DRAM boundary, neither pass count nor store order
is the lever. Which memory controllers you use is.** The evidence:

1. **Three structurally different L=8 memory schedules converge to within 0.13 %.** At
   L=8 B=32768, `L8_fusedaxes` (one fused pass, `seq3AA` variant), `L8_batchsimd` (fused, split
   radix-8) and `L8_radix8` (one-pass `1f` with 3-pass probes) return 5691.0, 5690.8 and
   5697.8 µs. All three deleted the RFO with non-temporal stores. All three land at
   **94.3 GB/s**. The pass-count-and-store-order axis, which was worth 20–72 % in mt_r1, is
   worth **0.13 %** here — because all three had already taken it, and something else is
   binding.
2. **What is binding is placement, and it was measured.** `L8_fusedaxes` shipped an
   execute-time governor that reads the page home of the *driver's own* `in`/`out` with
   `get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` (a pure read — the mt_r1 verdict banned migration
   and asked for exactly this diagnostic) and re-scans every 48 calls so a late migration is not
   missed. All three processes publish `gov{fr=0,nb=1,…}`: **zero percent of the 512 MiB of
   caller buffers is remote, in every process, for the whole timed loop, with
   `numa_balancing` enabled.** `driver.c:105-118` explains it — `aligned_or_die`, then
   `memset(out)`, then `fread(in)`, all on the main thread. And all three L=8 entries picked a
   **16-thread single-socket team** at that cell (`seq3AA-nt+pfs/nt16`, `mt{T=16 run=nt-s0}`,
   `mth-1f-nt-pfs` where `mth` is literally `(g_team+1)/2`). One socket's pages, one socket's
   threads, one socket's DRAM: 94 GB/s, and `fftw3_patient`'s 32-thread plan gets 103 GB/s. The
   panel loses that cell because it chose to use half the machine.
3. **The other regime exists and the same binary brackets it.** L=6 B=65536 sustains
   **199.6 GB/s** — more than one socket of DDR4-2666 can deliver — and `L6_pfa` measures both
   sides at that cell: its T=32 process reads 2582 µs (**200 GB/s**) and its T=16 processes read
   5327/5330 µs (**85 GB/s**). Same code, same cell, same node, **2.35× from team width alone**.
   So on this machine the caller's pages are *sometimes* effectively spread and sometimes not,
   and the difference is worth more than every schedule decision in the round put together.
4. **This falsifies mt_r1's leading explanation and replaces it with a better one.** mt_r1
   hypothesised AutoNUMA migration during the timed loop. At L=8 B=32768, with `nb=1`, `fr`
   stays 0 for the entire run — under a 16-thread socket-0 team, which gives AutoNUMA nothing to
   migrate *toward*. The coherent reading of all the evidence is: **the create-time arena races
   its candidates in the pre-migration/pre-spread regime, where a wide team looks bad (32
   threads contending for one socket's controllers over UPI) and a narrow team looks good; the
   scored run is long enough to reach the other regime, where the wide team would have won —
   but the tuner has already locked itself out of it.** That is one mechanism that explains
   `L6_pfa`'s T=32→T=16 flip, `L36_mixedradix`'s vol32→vol16 flip, `L64_blocked`'s
   G=2→G=16/dyn=1 flip, and the L=8 sweep, all at once. It is inference from measurement, not a
   measurement: **nobody has yet read `fr` under a 32-thread team at a streaming cell**, and
   that is the one experiment that settles it (§6).

**Net effect on §4.3.** Tolmachev's rule survives, but the "bandwidth gap between the two levels
involved" must be read as *the gap the running configuration actually reaches*, and on a
two-socket node that is a property of team placement rather than of the transform's schedule.
The r1 finding (NT stores worth 20–72 % at 32 cores) is unchanged and now universally adopted;
it has simply stopped being the binding constraint.

### Secondary: **§4.2, "L=17: dense-symmetric, Rader, or a hand-derived Winograd module?"**

mt_r1 closed §4.2(a) tentatively — its Rader entry's headline was a 1-in-3 lucky process.
mt_r2 gives the same answer on stable numbers at all three cells, with all three entries
improved at B=1:

| cell | dense conj-symmetric | Rader-17 | Winograd module |
|---|---|---|---|
| B=1 | **6.163** | 6.776 | 7.604 |
| B=256 | **0.755** | 0.792 | 1.078 |
| B=4096 | 2.858 | 2.200 | **1.220** |

**Rader-17 is second at every cell and first at none, in both rounds, at one thread and at 32.**
§02 §7's "Rader is not the lever at L=17" is now as settled as this project can settle it. The
regime split between the dense form (cache-resident) and the hand-derived Winograd module
(DRAM-bound) is confirmed and has *widened* at B=4096, from 1.72× to 2.34×.

### Also moved, in a direction worth recording: **§4.6, "model versus search for the schedule"**

§06's position — that the schedule is "the primary thing to search" at all our sizes — is not
challenged; every gain in this round came from search. What mt_r2 adds is a **precondition**
nobody in the corpus states: *a search is only as good as its arena's fidelity to the scored
regime.* Six entries this round were beaten by their own tuners (§3.3, §3.4), with arena-to-
driver mis-pricings of 1.26× (`L36_pfa`), 2.2× (`L36_pencilfused`) and a 2.06× three-way pick
lottery (`L6_pfa`); and the two entries whose arenas became *honest* this round (`L45_pfa`,
`L45_mixedradix`) both moved toward the driver's number, one of them by getting 1.69× slower.
Search remains right; racing a surrogate is not searching.

### **Not moved: §4.5 (L=8 padding, 4 KiB aliasing, `ld_blocks_partial.address_alias`)**

`L8_fusedaxes` compiled the counter behind `-DL8_PMC=1` and could not read it —
`perf_event_paranoid` on the node forbids it. Its `seq3AA` variant, motivated by shaving
volume-boundary alias stalls, won the pick in 2 of 3 processes and moved the cell 0.176 → 0.174
µs/t: **~1 %.** That is a weak negative datum, not an answer. §4.5 is where mt_r1 left it, and
L=8 is still the geometry with a library loss.

---

## 6. The single highest-value thing the next round should attack

**Before anything else, one item that dominates every per-geometry item below.** The round has
now *measured* what mt_r1 could only hypothesise, and the finding is that create-time arenas
race in a different regime from the one the driver scores (§5). Two concrete moves, in order:

1. **Read `fr` under a wide team.** `L8_fusedaxes`'s governor already does the scan; port it (or
   the four lines of `get_mempolicy` it wraps) into the L=6 and L=36 tuners and publish `fr` at
   L=6 B=65536 and L=36 B=512 under both T=16 and T=32. If `fr > 0` under T=32 and `fr = 0`
   under T=16, the mechanism in §5 is confirmed and the fix is mechanical: **race wide-team
   candidates only after a migration-settling warmup, or make wide-team the incumbent at any
   cell whose working set exceeds aggregate L3 and require a candidate to beat it by a margin.**
   This is one instrument, already written, applied to four more entries.
2. **Fix page placement in the harness, identically for every backend**, as mt_r1 recommended
   and this round did not do — parallel first-touch of the caller's buffers, or an explicit
   policy — **and make the timing pass report medians alongside minima.** The mt_r1 ruling
   stands: an entry may *read* page homes as a diagnostic but may not `move_pages` the driver's
   buffers. Three of this round's twelve headline cells are decided by a 1-in-3 process (§3.3);
   that has to stop before finer distinctions mean anything.

Per geometry:

* **L=6 — make `L6_pfa`'s B=65536 team-width race reproducible, or concede the geometry to
  `L6_unrolled`.** The gap is entirely the T=32-vs-T=16 flip: 200 GB/s versus 85 GB/s on the
  same binary. `L6_unrolled` already ships T=32 at that cell in all three processes and is
  2.08× faster than its own r1 number. The cheapest correct move for `L6_pfa` is to stop racing
  team width at streaming cells and hard-pin T=32 there, which its sibling's stability proves is
  right. L=6 is otherwise finished: 1.69×/2.24×/1.92× over the best library and the highest
  achieved bandwidth in the project.
* **L=8 — use the whole machine. This is the round's clearest single action.** All three
  entries independently chose a 16-thread single-socket team at B=32768 and all three landed at
  94 GB/s against `fftw3_patient`'s 32-thread 103 GB/s. The cell is not a kernel problem — the
  three schedules are within 0.13 % of each other — it is a team-width problem, and `fr=0` plus
  L=6's 200 GB/s says the second socket's controllers are reachable. Force T=32 with a
  socket-aware chunk split at that cell and re-measure. Only if that fails is §4.5 the next
  swing (and `ld_blocks_partial.address_alias` still needs `perf_event_paranoid` lowered — worth
  one request to whoever administers the node, since it is the highest-value single counter in
  the project and has now been blocked for two rounds).
* **L=17 — merge, don't diverge.** `L17_winograd` owns B=4096 by 2.34× over the dense form;
  `L17_matrixsimd` owns B=1 and B=256 by 1.10× and 1.05× over Rader. Both regressed this round
  at the *other* entry's cell (matrixsimd 1.36× at B=4096, winograd 1.31× at B=256), which
  means each broke something while chasing the other's regime. The action is to make the regime
  switch explicit rather than raced: a single entry that runs the dense conj-symmetric X-first
  schedule below the aggregate-cache threshold and the Winograd rotating-pass NT schedule above
  it, with the threshold set from the working set rather than from an arena. Also: `dyn≠0` was
  the losing pick in every process where it was chosen (`L17_winograd` `dyn=2` at 8662 vs
  `dyn=0` at 4998; `L64_blocked` `dyn=1` at 17353 vs `dyn=0` at 11582). mt_r1 already concluded
  dynamic scheduling loses on this harness; it should be removed from the tuner surface, not
  re-raced.
* **L=36 — this is where the untried structural move still sits, and now two of three entries
  have regressed at the cell it targets.** `L36_pfa` (1.62× worse) and `L36_pencilfused`
  (1.39× worse) both went backwards at B=512, and `L36_mixedradix`'s 1.97× headline is really
  1.07× typical. The geometry is at 77 GB/s on the representative number against L=6's 200 and
  L=17's 129 — there is more headroom here than anywhere else in the round. §4.3's named
  construction is still unbuilt: **tile the batch so a tile fits one core's 1 MiB L2, then run
  all three axes inside the tile.** `L36_pencilfused` fuses planes but streams whole volumes;
  the other two are volume-parallel row–column. Build the L2-tiled fused variant, and — first —
  restore `L36_pfa`'s deleted `T=16 scratch+nt pf=1` row, which is a 1.62× regression sitting in
  its own r1 source.
* **L=13 — rebuild `L13_rader`.** The entry that held the panel's best B=8192 number is a link
  error away from existing; its missing symbols are named in `build_errors.txt` and its r1
  source is in `impl_1/`. Then close the 1.24×-on-medians gap to `fftw3_patient`: `L13_direct`
  achieves 71 GB/s where FFTW gets 89 and L=6 gets 200, so the cell is nowhere near the roof.
* **L=23 — B=1.** `L23_matrixsimd` went three-for-three on its predictions and took B=2048 back
  on a stable number; the geometry is in good shape everywhere except single-volume
  decomposition, where 11.628 µs against MKL's 42.5 is a 3.38× win that still represents only
  ~4× parallel efficiency. The plane-granularity imbalance (23 planes over 32 threads) is named
  in its record and remains unaddressed.
* **L=45 — find out what the spin pool cost at B=256.** `L45_pfa` regressed 1.69× while
  replacing OpenMP with a pinned pool, and its arena now agrees with the driver, which means the
  r1 26.897 was a real number that the r2 code cannot reproduce. `impl_1/L45_pfa.c` versus
  `impl_2/L45_pfa.c` is a clean A/B, and 64 GB/s at that cell is the second-lowest in the round.
  `L45_mixedradix`'s 1.57× repair shows the geometry responds to arena fidelity.
* **L=64 — take B=8 from MKL; it is 5 % away.** `L64_blocked` closed 0.81× → 0.95× in one
  round. Its record's named residual — the gang-internal all-to-all where each lane reads 3/4 of
  its second-pass input from sibling lanes — is unchanged and is the remaining route. Secondary:
  `L64_blocked`'s B=128 pick lottery (`G=2 dyn=0` 11582 vs `G=16 dyn=1` 17353) is a 1.50× loss
  to a knob its own record put on the dead list.

---

## 7. Curation decision

Applying `docs/CURATION.md`'s four grounds in order. All 19 strategy records stay in
`strategies/` regardless; `exemplars/mt_r2/` is the reading list. Ties within 0.5 % are treated
as ties, not wins, since that is below the round's within-process sample spread.

**Promoted (16):**

| entry | ground |
|---|---|
| `L6_unrolled` | rule 1 — fastest L=6 at B=4096 and B=65536, and the only one of the pair that is reproducible (2269/2275/2277 µs, 200 GB/s); 2.08× better than its own r1 number |
| `L6_pfa` | rule 4 (beats every library at all three cells) + rule 3 — the round's cleanest documented pick lottery, T=32 200 GB/s vs T=16 85 GB/s on the same binary at the same cell |
| `L8_fusedaxes` | rule 1 — fastest L=8 at B=2048 by 1.06×, tied at the other two; and it carries the `gov{fr,nb}` placement instrument, the single most valuable diagnostic produced this round |
| `L8_radix8` | rule 2 — structurally distinct (`1f`/`3p` half-team) and within 0.13 % at B=32768; improved 1.24× at B=2048 |
| `L13_direct` | rule 1 — fastest L=13 at all three cells; hit a 2.9×-extrapolated B=8192 prediction to within 3 % |
| `L17_matrixsimd` | rule 1 — fastest L=17 at B=1 (3.67× MKL) and B=256 (5.17×) |
| `L17_winograd` | rule 1 at cell + rule 2 — fastest L=17 at B=4096 by 2.34× over the dense form, 129 GB/s; structurally the hand-derived Winograd module |
| `L23_matrixsimd` | rule 1 — fastest L=23 at B=1 and B=2048; three-for-three on pre-registered bands, the round's best-calibrated record |
| `L23_rader` | rule 1 at cell + rule 2 — fastest L=23 at B=128, structurally distinct from the dense form |
| `L36_mixedradix` | rule 1 — fastest L=36 at all three cells as scored (with §3.3's caveat at B=512 recorded in the note) |
| `L36_pfa` | rule 3 — 1.62× regression at B=512, stable across all three processes, caused by deleting its own r1-winning `T=16 scratch+nt` row on wallaby reasoning; the record documents the arena number (24.97) and the driver number (31.57) |
| `L36_pencilfused` | rule 3 + the §4.3 datum — the only fused-plane construction at L=36; its probe read `t32=15.72` for a 34.04 µs/vol driver result, a 2.2× arena mis-pricing at the same cell where it was 1.6× wrong in r1 |
| `L45_pfa` | rule 1 — fastest L=45 at all three cells; also rule 3 at B=256, where a 1.69× regression bought arena honesty (arena 42.4 vs driver 45.4, against r1's 16.9-vs-26.9) |
| `L45_mixedradix` | rule 3 resolved — the repair of mt_r1's named arena-fidelity failure, 79.07 → 50.22 µs/t (1.57×); the worked example of how to fix the defect §5 says dominates the round |
| `L64_blocked` | rule 1 — fastest L=64 at B=1 and B=8; three-for-three on pre-registered bands; 1.46× better at B=8 |
| `L64_radix8` | rule 1 at cell — fastest L=64 at B=128, 2.11× better than its own r1 number and 2.37× the best library; three-for-three on pre-registered bands |

**Not promoted (3):**

* **`L13_rader`** — **failed to build** (§3.1: five undefined spin-pool symbols, implementer
  agent `exit=134` on memory exhaustion mid-edit) and its strategy record has **no mt_r2
  section**. CURATION forbids promoting an entry whose record is missing, and a non-compiling
  source file must not go in front of the next panel. Its mt_r1 exemplar stands. The number that
  matters for the record: its absence widened L=13 B=8192's loss to `fftw3_patient` from 1.24×
  to 1.63×.
* **`L8_batchsimd`** — near-duplicate of `L8_radix8` (identical arithmetic, 1248 vector FP +
  896 shuffles), nominally fastest at B=1 and B=32768 but by 0.2 % and 0.004 %, which is a tie,
  not a win. Its lesson is worth recording here rather than in an exemplar: **it repaired
  mt_r1's disqualifying failure completely** — 9674 → 5691 µs at B=32768, a 1.70× improvement —
  by rebuilding its streaming candidate set on the node's own arena tables instead of wallaby's.
  The instructive-failure record from mt_r1 did its job.
* **`L17_rader`** — its own description still states the kernel is taken from `L17_winograd`,
  so it is a near-duplicate of a promoted entry, and it is fastest at no cell. It improved
  1.35× at B=1 and 1.32× at B=4096, and its contribution is the negative one that closes
  §4.2(a): **Rader-17 is second at all three cells and first at none, in both rounds, at one
  thread and at 32** (6.776 / 0.792 / 2.200 against the winners' 6.163 / 0.755 / 1.220).

PROMOTE: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L13_direct L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pfa L36_pencilfused L45_pfa L45_mixedradix L64_blocked L64_radix8
