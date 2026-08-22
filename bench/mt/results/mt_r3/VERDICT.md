# mt_r3 — monitor's verdict

Round: `mt_r3`, the third multicore round.
Scored on: `p55n3`, Intel Xeon Gold 5218 (Cascade Lake, 2 sockets × 16 cores, 1 MiB L2/core,
22 MiB L3/socket, 1× 512-bit FMA unit, DDR4-2666), 32 threads, `OMP_PROC_BIND=close`,
governor `powersave`, gcc 11.4.0, slurm job 438566.
Comparison baseline: `results/mt_r2/leaderboard.txt` (same node, `p55n3`, job 438551).

**The two rounds are directly comparable.** Across all 24 cells the r3/r2 ratio for `mkl_dfti`
has median 0.9994 and spans 0.954–1.023; `mkl2026_dfti` 0.9987 / 0.956–1.009; `fftw3_patient`
0.9985 / 0.824–1.083; `ducc0_c2c` 0.9982 / 0.977–1.058; `baseline_matrix` 1.0000 /
0.995–1.002. The machine did not move. **Every panel change reported below is code.** That
matters more than usual this round, because the round's biggest number went the wrong way and
the machine is not available as an excuse for it.

Statistic used by the harness: **minimum over three independent processes of each process's
minimum sample.** The harness was not changed this round, so the mt_r2 recommendation to report
medians alongside minima was *not* implemented; I compute medians from the raw `t_*.json`
(`per_execute_seconds.median`) wherever the minimum is load-bearing, and quote both.

---

## 0. Bottom line

* **Nothing failed correctness. Nothing failed to build. Nothing crashed or hung. Nothing is
  missing.** `build_errors.txt` is empty (0 bytes), `failures.txt` does not exist, all 19
  implementer agents exited 0, and the record count reconciles exactly: **216 correctness
  records, zero with `ok != true`, zero with `rel_l2 > 1e-13`**, every leaderboard row backed by
  a matching `c_*.json`, every timing record complete at three processes. mt_r2's one casualty,
  `L13_rader`, **builds and scores** (§3.1). There are no fast wrong answers to strike.
* The panel beats the best threaded library at **20 of 24 scored cells**, by 1.19× to 5.49×, and
  loses four: L=6 B=65536 (**0.84×, new**), L=8 B=32768 (0.93×), L=13 B=8192 (0.61×), L=64 B=8
  (0.94×). On medians the panel wins 23 of 24 and only L=64 B=8 is a real loss.
* **The round's central result is a negative one, and it is the most valuable thing produced in
  three rounds.** The mt_r2 verdict's §5 mechanism — *create-time arenas race in a pre-migration
  page-placement regime, so the tuner locks a narrow team and forecloses a two-socket regime the
  scored loop would have reached* — is **refuted**. Six entries independently built the
  `get_mempolicy` instrument mt_r2 ordered and ran it under 32-thread teams at streaming cells.
  Every one of them reads **`fr = 0`**: the caller's buffers never migrate. Where `fr` is not
  zero it makes no difference (`L36_mixedradix` at B=512: 150.9 / 150.5 / 149.5 GB/s at
  `fr` = 0 / 7 / 23). And four entries raced team width **on the caller's real buffers**, after
  settling warmups of up to 0.9 s and dwells of up to 161 calls, and all four found the **half
  team faster** (§5). "Use the whole machine" was measured, four ways, and it is wrong here.
* **What did move bandwidth is the kernel's memory schedule — LITERATURE.md §4.3's re-opened
  L2↔DRAM tile construction, which the corpus calls "the largest untried structural move on the
  board." It got built this round, and it worked.** `L36_pencilfused` put one 729 KB volume
  through a 1 MiB-L2-resident scratch tile with all three axes inside it and went **43.9 →
  137.5 GB/s (3.13×)**. `L36_mixedradix` reached **150.9 GB/s in all three processes** where r2
  managed it in one. `L45_pfa` 64.2 → 109.0, `L17_rader` 71.5 → 122.0. Four geometries, four
  independent repairs, none of them a team-width or page-placement change.
* **The pick lottery — mt_r1's and mt_r2's named defect — is largely fixed.** r2 had four cells
  decided by a ≥1.5× cross-process flip and thirteen at ≥1.06×. r3 has two: `L13_rader` at
  B=512 (1.50×) and B=1 (1.39×), plus `L64_blocked` at B=128 (1.24×) and `L36_mixedradix` at
  B=1 (1.13×). Everything else is within 3% across processes. The mechanism was
  working-set-gated deterministic installation instead of racing, and it worked everywhere it
  was applied.
* **Two large, stable, self-inflicted regressions.** `L6_unrolled` at B=65536 went **2269 →
  6290 µs (2.77× worse)** with a byte-identical execute path, the same shipped pick, and
  `fr=0/0` — and it took the L=6 B=65536 cell from a 1.92× win to a 0.84× loss (§3.3).
  `L64_radix8` at B=128 went **8895 → 18217 µs (2.05× worse)** on an *identical* pick string
  whose own arena price barely moved (§3.4).

---

## 1. Headline per geometry — fastest correct panel entry vs best library

### The four brief geometries

| L | case | fastest panel entry | best library | ratio (min) | ratio (median) |
|---|---|---|---|---|---|
| **6** | **B=1 (non-batched)** | **L6_pfa 0.211 µs** | mkl_dfti 0.372 µs | **1.76×** | 1.72× |
| 6 | B=4096 | **L6_pfa 0.0092 µs/t** (37.746 µs/call) | mkl_dfti 0.0205 (84.093) | **2.23×** | 2.28× |
| 6 | B=65536 | L6_pfa 0.0788 µs/t (5161.4 µs/call) | **fftw3_patient 0.0660 (4326.7)** | **0.84× — LOSS** | 1.55× |
| **8** | **B=1 (non-batched)** | **L8_fusedaxes 0.551 µs** | mkl_dfti 0.657 µs | **1.19×** | 1.19× |
| 8 | B=2048 | **L8_fusedaxes 0.0261 µs/t** (53.496 µs/call) | mkl_dfti 0.0361 (73.980) | **1.38×** | 1.39× |
| 8 | B=32768 | L8_fusedaxes 0.1723 µs/t (5646.4) | **fftw3_patient 0.1595 (5224.5)** | **0.93× — LOSS** | 1.39× |
| **17** | **B=1 (non-batched)** | **L17_matrixsimd 5.976 µs** | mkl_dfti 23.148 µs | **3.87×** | 3.89× |
| 17 | B=256 | **L17_matrixsimd 0.756 µs/t** (193.459 µs/call) | fftw3_patient 3.900 (998.504) | **5.16×** | 5.15× |
| 17 | B=4096 | **L17_winograd 1.219 µs/t** (4991.7) | fftw3_estimate 4.096 (16777.9) | **3.36×** | 3.36× |
| **36** | **B=1 (non-batched)** | **L36_mixedradix 23.027 µs** | mkl_dfti 38.107 µs | **1.65×** | 1.65× |
| 36 | B=32 | **L36_pfa 5.212 µs/t** (166.785 µs/call) | mkl_dfti 7.411 (237.148) | **1.42×** | 1.41× |
| 36 | B=512 | **L36_mixedradix 9.896 µs/t** (5066.5) | fftw3_patient 19.397 (9931.0) | **1.96×** | 2.02× |

All twelve of these are stable across processes (cross-process spread ≤ 3%) except
`L36_mixedradix` at B=1 (23.0 / 23.2 / 25.9 µs, 1.13×; it wins on any of the three).
**Every one of the L=36 B=512 and L=17 B=4096 headlines is now a 3-of-3 result** — that was the
thing mt_r2 said had to stop before finer distinctions meant anything, and it stopped.

### The four extra geometries the round also measured

| L | case | fastest panel entry | best library | ratio (min) |
|---|---|---|---|---|
| 13 | B=1 | **L13_direct 5.868 µs** | mkl2026_dfti 7.642 | 1.30× |
| 13 | B=512 | L13_rader 154.6 µs — **1-in-3, do not quote** (§3.2); honest winner **L13_direct 156.05** | mkl2026_dfti 184.40 | 1.18× (on L13_direct) |
| 13 | B=8192 | L13_direct 0.980 µs/t (8030.9) | **fftw3_patient 0.603 (4938.7)** | **0.61× — LOSS** (1.19× on medians) |
| 17 | — | (see above) | | |
| 23 | B=1 | **L23_rader 11.865 µs** (L23_matrixsimd 11.956, a 0.8% tie) | fftw3_patient 38.890 | 3.28× |
| 23 | B=128 | **L23_matrixsimd 2.268 µs/t** (290.2) | fftw3_estimate 12.446 (1593.1) | 5.49× |
| 23 | B=2048 | **L23_rader 5.810 µs/t** (11898.0; matrixsimd 12069.2, a 1.4% tie) | fftw3_patient 12.529 (25658.4) | 2.16× |
| 45 | B=1 | **L45_pfa 56.569 µs** | mkl2026_dfti 83.718 | 1.48× |
| 45 | B=16 | **L45_pfa 14.742 µs/t** (235.9) | mkl2026_dfti 37.526 (600.4) | 2.55× |
| 45 | B=256 | **L45_pfa 26.763 µs/t** (6851.3) | fftw3_patient 59.326 (15187.5) | 2.22× |
| 64 | B=1 | **L64_blocked 128.691 µs** | mkl_dfti 152.760 | 1.19× |
| 64 | B=8 | L64_blocked 77.127 µs/t (617.0) | **mkl_dfti 72.366 (578.9)** | **0.94× — LOSS** |
| 64 | B=128 | **L64_blocked 73.715 µs/t** (9435.5) | fftw3_patient 178.077 (22793.8) | 2.42× |

**Best result of the round:** `L36_mixedradix` at L=36 B=512 — 5066.5 / 5079.6 / 5111.4 µs
across three processes (0.9% spread), **150.9 GB/s**, 1.96× the best library. It is the fastest
streaming cell in the project's history *on a reproducible number*, and it replaces r2's
1-in-3 5114 µs with the same number in every process. Runner-up: `L36_pencilfused`'s 3.13×
repair at the same cell — the one entry that built the construction the literature named.

---

## 2. What changed since mt_r2, per geometry. Did anything regress?

**L=6 — the geometry went backwards, and it is the round's worst news.** `L6_pfa`'s B=65536
pick lottery is **fixed and the fix is worthless**: r2 read 2582 / 5327 / 5330 µs (a 2.06×
lottery); r3 reads **5161 / 5173 / 5165** — a 0.2% spread at the *slow* level. Its own
first-execute re-race on the caller's buffers shipped `3pass_pf_nt_xa_d2, T=16` in two
processes and `fused_pf_nt_xa_d2, T=32` in one, and **all three landed within 0.2% of each
other at 87.7 GB/s** — so the re-race proved that T=16 and T=32 are now the same speed at that
cell, which is itself a clean measurement and the direct refutation of the "team width is worth
2.35×" reading r2 built on this exact entry. Against that, its pre-registration ("three
near-identical runs at ~39 ns/vol or better") half-landed: the spread prediction was right, the
level was 2× off. B=1 improved 1.04× (0.220 → 0.211) and B=4096 1.02×; it now takes all three
L=6 cells from its sibling.

`L6_unrolled` **regressed 2.77× at B=65536** (2269.3 → 6290.2 µs, all three processes within
0.1%), which is the largest single regression in either multicore round, and it is why L=6
B=65536 is now a loss. Details in §3.3. At B=4096 it is 0.99× (flat) and at B=1 flat.

**L=8 — the cell is still lost by 7%, and the round answered *why* in the negative.** All three
entries built execute-time governors that race team width on the caller's real buffers, and all
three published the answer: the wide 32-thread team is **19–35% slower** than the 16-thread
single-socket team, measured post-settling with `fr = 0` throughout (§5). `L8_fusedaxes`
improved 1.01% at B=32768 and 1.01× at B=2048 and takes all three cells; `L8_batchsimd`
improved 1.04× at B=2048 (56.9 → 54.7, its epoch-word pool replacing the tree collect its own
r2 record pre-registered for reversion); `L8_radix8` flat. Nothing regressed at L=8. The panel
now wins the cell 1.39× on medians (6980 vs fftw's 9685) and loses it 0.93× on minima; both
facts stand.

**L=13 — the missing entry is back and the geometry did not move.** `L13_rader` builds
(§3.1) and is competitive on minima at all three cells, but two of its three cells are decided
by a pick lottery it created this round (§3.2), so its representative numbers *lose* to MKL at
B=1 and B=512. `L13_direct` is flat everywhere: B=1 5.734 → 5.868 (a 2.3% regression — its
governor locked `t2g2` in all three processes after measuring it 10% faster on the real buffers,
and the cell got slower; the third consecutive round in which the `t2g2` instrument reads high,
and the first in which it was acted on), B=512 158.0 → 156.1, B=8192 8061.9 → 8030.9. The
B=8192 loss to `fftw3_patient` is unchanged at 0.61× on minima — but note the honest picture:
FFTW hits its 4939 µs mode in **one** of three processes and reads 10382 / 10359 in the other
two, so on medians the panel *wins* that cell 1.19×. Both L=13 entries sit at 69–72 GB/s where
`L17_winograd` gets 129 on the same node; the geometry is nowhere near its roof.

**L=17 — the best-executed geometry of the round, and one 2.39× puzzle.** `L17_winograd` did
exactly what it pre-registered: it deleted `dyn` from the tuner surface, sized its arena to
aggregate rather than per-socket L3, owner-touched it, and switched to a median-of-3 pick
statistic — and its B=4096 cell went from a 1.73× three-process lottery (8662/5029/4998) to
**4991.7 / 5013.5 / 4997.0, 0.4% spread**, at the same 1.219 µs/t. It also repaired its r2
B=256 regression, 1.078 → 0.823 µs/t (1.31×). `L17_matrixsimd` improved 1.03× at B=1 (still
first, 3.87× MKL), held B=256, and **regressed 2% at B=4096 to 11924 µs while running
L17_winograd's engine verbatim** (§4.4) — its working-set gate fired exactly as designed
(`ws=614MB aggl3=44MB`), the lottery is dead by construction (1.5% spread), and the engine
still runs **2.39× slower in its process than in winograd's**. `L17_rader` improved **1.71× at
B=4096** (9010 → 5280, 71.5 → 122.0 GB/s) by pinning the wide team and refusing near-tie NT
picks — and is second at all three cells for the third consecutive round.

**L=23 — flat and clean, and the two entries have converged.** `L23_rader` improved **1.23× at
B=2048** (14678 → 11898) with the aggregate-cache arena it borrowed from its sibling, and takes
B=1 (by 0.8%) and B=2048 (by 1.4%) — both ties. `L23_matrixsimd` held B=2048 (12153 → 12069)
and B=128 (292 → 290), and regressed 3% at B=1 (11.628 → 11.956). Its new dynamic
volume-claiming knob was raced and, correctly, not installed at B=128. Nothing regressed
materially. Both entries are stuck at 65–67 GB/s, the lowest of any geometry's streaming cell
in the round.

**L=36 — the round's best geometry, and the §4.3 result.** `L36_pencilfused` improved **3.13×
at B=512** (17430 → 5561; 43.9 → 137.5 GB/s), reversing the r2 mechanism it had built on ("NT
stores never fault, so `out` stays on the driver's socket") after measuring the premise dead,
and shipping the L2-resident tile construction. `L36_pfa` improved **1.63× at B=512** (16163 →
9893) by restoring the `T=16 scratch+nt` row this verdict's predecessor ordered it to restore,
and takes B=32 outright (167.5 → 166.8). `L36_mixedradix` converted its r2 1-in-3 5114 into
5066 / 5080 / 5111 by pinning `v1-vol32-sntp` at deep-streaming cells instead of racing it, and
holds B=1 and B=512; it lost B=32 to `L36_pfa` by 3% (166.8 → 171.6). Nothing at L=36
regressed. The internal spread is now the geometry's whole story: 150.9 / 137.5 / 77.3 GB/s for
three entries all describing themselves as scratch+NT (§6).

**L=45 — the r2 regression is fully repaired and then some.** `L45_pfa` improved **1.70× at
B=256** (11634 → 6851), which beats its own mt_r1 number (6885 µs) — so the r1 result was real
and the r2 code's failure to reproduce it is now explained by two things its record names: a
`Pt` scratch pointer that was 16-mod-64 aligned (so every 64 B plane-scratch access split a
cache line, invisible on Sapphire Rapids, expensive on Cascade Lake), and the r1 OpenMP
two-sweep reinstated next to the spin pool. 64.2 → 109.0 GB/s. It also improved 1.07× at B=16.
`L45_mixedradix` improved 1.05× at B=256 and 1.10× at B=16 and regressed 2% at B=1 — and is now
**1.78× behind its sibling at B=256 on the same geometry and the same node** (61.1 vs
109.0 GB/s), which makes it the round's cleanest negative control (§6).

**L=64 — the leadership changed hands on a 2.05× regression.** `L64_blocked` improved **1.23× at
B=128** (11582 → 9435) and takes the cell at 113.8 GB/s; it holds B=8 (613.8 → 617.0, still
0.94× behind MKL, now the panel's only loss on medians) and regressed 1.3% at B=1 (127.0 →
128.7, still first). `L64_radix8` **regressed 2.05× at B=128** (8895 → 18217, all three
processes within 1%) and lost the cell it won in r2 (§3.4). Its `scs16` NUMA-correct shared
scratch for the B=1 T16 split — the round's structural bet — landed at 136.3 µs against its own
pre-registered refutation criterion ("if it stays ≥130 the model is wrong"); the model is
refuted by its author's own test.

---

## 3. Adversarial audit

### 3.1 Nothing failed, nothing is missing — verified against the raw records

I cross-checked the leaderboard against the raw JSON programmatically rather than reading it:

* **216 correctness records; 0 with `ok != true`; 0 with `rel_l2 > 1e-13`.** Range 1.34e-16 …
  8.42e-16 against `numpy.fft` at tol 1e-12. Worst is `baseline_matrix` at L=17 (8.4e-16,
  expected for an O(L⁴) matvec); best panel entry is `L8_radix8` at B=1 (1.4e-16). `check.log`
  contains no line other than `PASS`.
* **Record count reconciles exactly:** 6 libraries × 24 cells = 144, plus `baseline_matrix` at
  15 cells (skipped at each geometry's largest cell, and at L=64 only B=1), plus **19** panel
  entries × 3 cells = 57. 144 + 15 + 57 = **216**. mt_r2 had 213 because `L13_rader` was
  absent; the three recovered records are exactly `L13_rader` at L=13 B∈{1, 512, 8192}.
* **Every one of the 216 leaderboard rows has a matching `c_*.json`.** Zero orphans in either
  direction.
* `build_errors.txt` exists and is **0 bytes**. `failures.txt` **does not exist** — nothing
  crashed or hung. `agents/exits.txt` shows **`exit=0` for all nineteen** implementers.
* Every implementation declares and delivers exactly its own three cells; the 1,197 lines in
  `timing.err` are all benign `"<entry>: does not support L=<other>"` from the cross-product
  sweep, which is the harness working as designed.
* Every timing record has all three processes. No truncated runs.

**mt_r2's build casualty is repaired.** `L13_rader` defined the five symbols it was missing
(`l13r_pool_new/del/nap`, `l13r_dwell`, `l13r_ntcopy`), builds clean, scores three cells, and
carries a full `## Round mt_r3` section in its strategy record — including the lesson, written
in its own words: *"run tryout.sh after the last edit, not just the promising middle ones."*
That is the correct response to an instructive failure and it is worth saying so.

### 3.2 The two pick lotteries that decide a headline — both are `L13_rader`, and both are self-inflicted

This is where I have to be adversarial about a number that reads like a win.

**`L13_rader` at L=13 B=512 is listed first in the leaderboard at 154.577 µs and 1.19× over
`mkl2026_dfti`. It must not be quoted.** The three process minima are **154.6 / 230.9 / 231.3**
(a 1.50× flip), each with within-process `sd` ≤ 0.2% — so this is not sampling noise, it is two
processes installing a different plan. The description strings name the flip exactly:

| process | pick | `ntb` | scored µs/call | its own arena's rows (ns/vol) |
|---|---|---|---|---|
| r1 | `pw!` | 32/32 | **154.6** | `pw!:465` `n24:475` `pp:461` `i:481` |
| r2 | `n24` | 24/32 | 230.9 | `pw!:576` `n24:480` `pp:545` `i:567` |
| r3 | `n24` | 24/32 | 231.3 | `pw!:549` `n24:475` `pp:588` `i:571` |

The arena prices `n24` at 475–480 ns/vol in **all three** processes, and the driver runs it at
452 ns/vol — the arena is *accurate* for `n24`. It prices `pw!` at 465–576 while the driver runs
it at **302** — the arena over-prices `pw!` by up to 1.9×, so whenever noise lifts `pw!`'s
reading above `n24`'s, the tuner ships a config that is 1.50× slower. `n24` (the 24-thread /
3:2 socket-weighted split) is one of this round's two new partitioning modes and it is the
losing pick in 2 of 3 processes. **The representative number is 231.6 µs, which loses to
`mkl2026_dfti`'s 184.4.** `L13_direct` runs the cell at 156.1 / 156.4 / 157.1 — stable, and 1.18×
over MKL. `L13_direct` is the honest L=13 B=512 winner.

**`L13_rader` at B=1 is worse, because its own arena told it not to.** Minima 8.5 / 6.1 / 8.5;
scored 6.065. The picks are `p2` / `i` / `p2`, and the arena rows are:

| process | pick | `ab[B1]` incumbent `i` | `ab[B1]` `p2` | scored |
|---|---|---|---|---|
| r1 | **`p2`** | 7761 | 10417 (**34% worse**) | 8.5 µs |
| r2 | `i` | 6880 | 11744 | **6.1 µs** |
| r3 | **`p2`** | 11108 | 10911 | 8.5 µs |

In process r1 the entry installed a config its own instrument priced **34% worse than the
incumbent it displaced**. That is this round's new "adopt-unless-vetoed" race semantics (change
5 in its record: pool knobs are adopted unless the incumbent beats them by >3% *in both trial
blocks*) failing to veto on an unambiguous 34% reading. The representative number is 8.5 µs,
which loses to `mkl2026_dfti`'s 7.6 and to `L13_direct`'s 5.87. The rule was introduced to fix
a wallaby-specific pathology (napped pool workers losing cores on a shared login node) and it
imported a node-specific failure; it needs the veto to be evaluated per block, not jointly.

Two smaller lotteries, neither load-bearing:

* **`L64_blocked` at L=64 B=128:** 9445 / 11686 / 9435, a 1.24× flip on `G=8` vs `G=16`
  (r2's version of this was 1.50× on `G=2`/`dyn=0` vs `G=16`/`dyn=1`; removing `dyn` halved it).
  It wins the cell at either number.
* **`L36_mixedradix` at L=36 B=1:** 23.0 / 23.2 / 25.9, 1.13%. Wins at any of the three.

**Being adversarial in the other direction: the statistic still flatters `fftw3_patient` at two
of the four lost cells,** and by more than in mt_r2.

* **L=6 B=65536.** FFTW's minima are 4327 / 4431 / 4350, but its medians are **8031 / 7996 /
  8042** and its maxima reach 12563 — it alternates between a ~4.3 ms mode and a ~8.0 ms mode
  and is scored on the fast one (within-process `sd` 14.5–21.8%). `L6_pfa` returns 5178 at the
  median with `sd` 0.2%. On any typical-case statistic the panel wins that cell **1.55×**. The
  0.84× loss stands as scored; both facts belong in the record.
* **L=8 B=32768.** FFTW 5231 / 5231 / 5225 on minima, **9663 / 9687 / 9685 on medians**
  (`sd` 26–28%). `L8_fusedaxes` medians 6975 / 6981 / 7002. Panel wins **1.39×** typically.
  (Note `L8_fusedaxes`' median is deliberately inflated: its cyclic governor spends 12 of every
  20 calls on the wide config precisely to measure it, and publishes the cost.)
* **L=13 B=8192.** FFTW 10382 / **4939** / 10359 — it reaches its fast mode in only one of three
  processes this round (r2: two of three). Medians 10395 / 6461 / 10376. Panel wins **1.19×**
  typically, loses **0.61×** as scored.
* **L=64 B=8.** `mkl_dfti` 578.9 / 581.2 / 579.7 at `sd` 0.2% against `L64_blocked`'s 617.0 /
  618.0 / 621.7 at `sd` 1.1%. **Fully real, no statistical escape.** 6.6% and unchanged from r2.
  This is the panel's only genuine library loss.

### 3.3 The round's largest regression: `L6_unrolled` at B=65536, 2.77×, with no mechanism in its record

This is the item the next round should open with, so the evidence is laid out in full.

| | mt_r2 | mt_r3 |
|---|---|---|
| shipped variant | `3pass_nt_pf` | `3pass_nt_pf` (identical) |
| shipped team | `nthr=32` | `nthr=32` (identical) |
| shipped dispatch | `disp=omp` | `disp=omp` (identical) |
| driver, 3 processes | 2277.3 / 2274.6 / **2269.3** µs | 6290.6 / **6290.2** / 6293.6 µs |
| bandwidth | **199.6 GB/s** | **72.0 GB/s** |
| its own dispatch A/B `od=` (pool, omp) | 36.6, 36.5 ns/vol | 53.2, 53.1 ns/vol |
| its own team curve `tm=` T=1…32 | 645/381/195/105/88/52/**39** | 635/392/200/108/82/58/**53** |
| its own `fr=` scan (calls 1 / 49) | (not instrumented) | **`fr=0/0, nb=1`** |
| setup | 2.72 s | 2.58 s |

1. **It is not the machine.** Libraries at that cell are unchanged: `fftw3_patient` 4357 → 4327,
   `mkl_dfti` 8012 → 8022, `mkl2026` and `ducc0` within 1%. Across all 24 cells MKL's median
   r3/r2 is 0.9994.
2. **It is not the pick.** The variant, the team width, and the dispatch shape are the same
   strings in both rounds.
3. **It is not the scored execute path.** `l6_mt_call` — the function that runs the timed loop —
   is **byte-identical** between `impl_2/L6_unrolled.c` and `impl_3/L6_unrolled.c`, and so is
   the per-thread scratch first-touch region (the OMP `num_threads(pool)` block, which still
   precedes any pool creation).
4. **It is not page migration.** The entry shipped the very instrument mt_r2 asked for and read
   `fr=0` on call 1 *and* on call 49, under a 32-thread team, at the round's most
   DRAM-bound cell. Nothing migrates. So r2's 200 GB/s cannot have been "the spread regime."
5. **Its own instrument saw 1.45× of it and the entry shipped anyway.** `od` went 36.5 → 53.1
   ns/vol at the shipped variant, and `tm` at T=32 went 39 → 53 ns — a self-measured 36%
   slowdown at exactly the widest team, with T=1…T=16 unchanged within a few percent. The
   pre-registered expectation in its record was "~34–35 ns again."
6. **The arena also lost fidelity:** 53.1 ns raced against 96.0 ns delivered (1.81× off), where
   r2 was 36.5 against 34.6 (0.95× — essentially exact). So the round both slowed the code down
   and blinded the instrument that would have caught it.

**Candidate causes, explicitly labelled as hypotheses I did not confirm.** The round's changes
to this entry were: (a) the tournament and team race now dispatch through a create-time spin
pool (`L6_RACE_MT`), with an `rpool` of 31 spinning workers alive across the multi-second
tournament and destroyed before the OMP timing; (b) 64-byte done-flag padding and a join-scan
prefetch in the pool; (c) the widest-team-incumbent pick rule at streaming cells; (d) the `fr`
scan; (e) deletion of four candidate kernels (`fused_nt`, `fused_zp_nt_pf`, `fused_nt_pfnta`,
the `sp2` twins). Since the pick, the execute path and the placement are all unchanged, the two
survivors are **a pool that outlives the tournament and contends with the shipped OMP team**,
and **code layout/alignment of `l6_run_3pass_nt_pf` shifted by the four-kernel prune**. The
first is far more plausible at this magnitude, and it connects to two independent observations
elsewhere in this round: `L36_pfa` added nap-after-1 ms to its idle workers this round
specifically because "a locked s16 no longer drags the all-core clock," and `L17_matrixsimd`'s
own probe reads **`clk512=2.29 GHz`** in its spin-pool process against `L17_winograd`'s
**2.89 GHz** in its OMP process on the same node (§4.4). Idle or oversubscribed spinners
costing all-core AVX-512 clock is the one mechanism this round has independent evidence for.

**Why it matters beyond L=6:** recovering 2269 µs turns L=6 B=65536 from a 0.84× loss into a
**1.91× win** and restores the project's only >150 GB/s L=6 number. `impl_2/L6_unrolled.c`
versus `impl_3/L6_unrolled.c` is a clean, two-run A/B on this node, and the r2 binary's number
is already in hand.

### 3.4 The other stable regression: `L64_radix8` at B=128, 2.05×, on an identical pick

| | mt_r2 | mt_r3 |
|---|---|---|
| pick string, all processes | `gang-T32-g8-fused-nt+slabpf1+sc0+p10` (r2 also flipped to `g4` once) | `gang-T32-g8-fused-nt+slabpf1+sc0+p10`, **3/3** |
| its own arena `pick=` | 81 / **70** (g4) / 81 µs/vol | 84 / 85 / 82 µs/vol |
| driver | 10518 / **8895** / 10480 µs | 18272 / 18412 / **18217** µs |
| driver µs/vol | 82.2 / 69.5 / 81.9 | 142.8 / 143.8 / **142.3** |
| bandwidth | 102 / **121** / 103 GB/s | 58.8 / 58.3 / **58.9** GB/s |

The pick is the same, its own arena's price is the same (81 → 82–85), and the driver is 1.74×
slower. In r2 the arena was accurate to 1%; in r3 it is **1.70× optimistic at the same
configuration**. That rules out the tuner's *ranking* and points at the execute path or the
process's memory state. Its record claims "arithmetic, layout, codelets, passes byte-identical
to mt_r2", but change 4 — *"legacy gang drops the trailing barrier of each gang's last
volume"* — modifies the shipped path at B=128, and change 2 grows the `WREG` allocation from 33
to 34 × 6 MiB (≈ +6 MiB of scratch, ≈ 204 MB total, against 1024 MiB of caller buffers at that
cell). Its wallaby A/B for the round shows the same shape at 51.0 µs/vol, so the
wallaby→node transfer factor at a fixed pick blew out from **1.30× (r2) to 2.79× (r3)** — that
is not a machine difference, it is a change in what the node's code does.

Its two structural bets both failed against their own pre-registered criteria: `scs16` (B=1
"drops from 136 to ~110–125; if it stays ≥130 the model is wrong") measured **136.3**, and the
B=128 prediction ("hold with legacy g4/g8-nt+slabpf1, 65–75 µs/vol") measured **142.3**. The
`p1pf=2` A/B that showed −27% on wallaby at `g16` took no node pick.

The geometry is not harmed — `L64_blocked` improved 1.23× and holds the cell at 2.42× over the
best library — but a 2.05× stable regression on an unchanged pick is the second-cheapest
bisect on the board.

### 3.5 Regressions that are real but small

None of these change a cell's ownership; listed for completeness, all stable across processes.

1. `L13_direct` B=1: 5.734 → 5.868 (2.3%). Its execute-time governor measured `t2g2` 10% faster
   than `t1g1` on the real buffers and locked it in **all three** processes; the cell got
   slower. The instrument that has read `t2g2` high for three rounds is now the instrument that
   cost 2.3% when acted on.
2. `L23_matrixsimd` B=1: 11.628 → 11.956 (2.8%). Loses the cell to `L23_rader` by 0.8%, a tie.
3. `L45_mixedradix` B=1: 57.5 → 58.8 (2.2%). `L36_mixedradix` B=32: 166.8 → 171.6 (2.9%),
   loses the cell to `L36_pfa`. `L64_blocked` B=1: 127.0 → 128.7 (1.3%). `L6_unrolled` B=4096:
   38.34 → 38.57 (0.6%). `L17_matrixsimd` B=4096: 11705 → 11924 (1.9%).

---

## 4. Claimed versus measured

Every implementer develops on **wallaby** (Xeon Gold 6448Y, Sapphire Rapids, 32 close-bound
threads on one socket, 2 MiB L2/core, ~60 MiB L3, two 512-bit FMA units, DDR5) and is scored on
**Cascade Lake** (2×16 cores, 1 MiB L2, 22 MiB L3/socket, one 512-bit FMA unit, DDR4-2666,
32 threads across two sockets). MKL alone spans 2.9× between them, so a 1.3–2.9×
wallaby→node factor is the expected background and is not a criticism. Several wallaby sessions
this round were explicitly flagged noisy (`sd` up to 5%) by their own records, which is honest
reporting and is reflected below.

### 4.1 Attributable to the machine — no criticism implied

| entry | cell | claimed (wallaby) | measured (node) | factor |
|---|---|---|---|---|
| L6_unrolled | B=4096 | 8.1–8.6 ns/vol | 9.42 | **1.12× — best transfer of the round** |
| L17_rader | B=4096 | 0.964–1.052 µs/t | 1.289 | 1.29× |
| L17_winograd | B=4096 | 0.84–0.93 µs/t | 1.219 | 1.36× |
| L36_mixedradix | B=512 | 7.25 µs/vol | 9.896 | 1.37× |
| L6_pfa | B=4096 | 6.1–6.5 ns/vol | 9.22 | 1.45× |
| L8_batchsimd | B=2048 | 0.0158–0.0163 µs/t | 0.0267 | 1.67× |
| L17_winograd | B=256 | 0.46–0.48 µs/t | 0.823 | 1.75× |
| L13_rader | B=512 | 0.162 µs/vol | 0.302 (fast process) | 1.86× |
| L64_blocked | B=1 | 61.3 µs | 128.7 | 2.10× |
| L8_batchsimd | B=32768 | 0.0760–0.0762 µs/t | 0.1746 | 2.30× (DDR5 vs DDR4, both at their wall) |
| L23_rader | B=2048 | 2.16 µs/t | 5.810 | 2.69× |
| L13_rader | B=8192 | 0.350 µs/vol | 0.983 | 2.81× |

### 4.2 Predictions that landed — the behaviour to reward

* **`L17_winograd` is the round's best-calibrated record.** It predicted, in its own words,
  "the B=4096 pick that was a 1.73× lottery on the node is now the same configuration in every
  process." Measured: 4991.7 / 5013.5 / 4997.0, `sd` 0.4%, identical `var=h4 dyn=0` in all
  three. It also predicted the B=256 repair and delivered 1.31×. Both from a wallaby session it
  itself labelled "worse than usual."
* **`L36_mixedradix`** predicted that pinning `v1-vol32-sntp` at deep-streaming cells would
  kill the lottery. Measured: 150.5 / 149.5 / 150.9 GB/s, three for three, where r2 was
  77.5 / 77.5 / 149.5.
* **`L17_matrixsimd`** predicted its working-set gate would make the B=4096 lottery "dead by
  construction." It is: 11934 / 12098 / 11924, `sd` 1.5%, `ws=614MB aggl3=44MB` identical in
  all three. The prediction landed; the number did not (§4.4).
* **`L45_pfa`** found a real bug by inspection that no timing on its dev machine could reveal
  (a 16-mod-64-aligned `Pt` making ~26k cache-line-splitting plane-scratch accesses per volume,
  free on Sapphire Rapids, expensive on Cascade Lake) and recovered 1.70× at B=256 — past its
  own mt_r1 number. Finding a node-only defect by reading your own allocator is the most
  transferable skill demonstrated this round.
* **`L6_pfa`** half-landed: it predicted the B=65536 spread would collapse "back to a few
  percent" and it collapsed to 0.2%; it predicted "~39 ns/vol or better" and measured 78.8.

### 4.3 Gaps that are **not** explained by the machine

1. **`L6_unrolled`, B=65536: wallaby 30.7 ns/vol, node 96.0 — a 3.13× transfer factor where the
   *same entry at the same cell* transferred at 1.11× one round ago.** Full treatment in §3.3.
2. **`L64_radix8`, B=128: wallaby 51.0 µs/vol, node 142.3 — 2.79×, against 1.30× in r2 at the
   same pick.** §3.4.
3. **`L17_matrixsimd`, B=4096: wallaby 1.097 µs/t, node 2.911 — but `L17_winograd` runs *the
   same engine* on the same node at 1.219.** §4.4.
4. **`L36_pfa`, B=512: wallaby 7.15 µs/vol, node 19.322 (2.70×) — but `L36_mixedradix` runs the
   same nominal shape (per-thread L2 volume scratch, NT stream to `out`, full team) at 9.896 on
   the same node.** So ~2× of the gap is schedule, not machine. Its own execute-time governor is
   *accurate* about what it is running (`s16=20.05` in-arena against 19.32 delivered) — it is
   simply racing three shapes none of which is the winning one. §6.
5. **`L45_mixedradix`, B=256: 12208 against its sibling's 6851 on the same node and geometry
   (1.78×, 61.1 vs 109.0 GB/s).** Not a machine effect; the sibling's own record names the two
   changes that bought it.
6. **`L13_rader`, B=512 and B=1:** wallaby 0.162 µs/vol and 3.236 µs, node 0.452 and 8.5 on
   representative numbers. Part machine, but the 1.50× and 1.39× *intra-process* spreads are
   its own tuner (§3.2), and `L13_direct` runs both cells stably on the same node.

### 4.4 One gap deserves its own entry: the same engine, 2.39× apart

`L17_matrixsimd` did exactly what mt_r2's §6 ordered ("merge, don't diverge"): it copied
~1240 lines of `L17_winograd`'s rotating-pass engine verbatim, installed it behind a
deterministic working-set gate rather than a race, and ran it above the aggregate-cache
threshold. The gate fired correctly in all three processes (`ws=614MB > aggl3=44MB`). And the
result is **11924 µs against `L17_winograd`'s 4991.7 at the same cell on the same node — the
same arithmetic, 2.39× apart, both stable.**

The two entries' own telemetry names the difference, and it is not the kernel:

| | L17_matrixsimd | L17_winograd |
|---|---|---|
| dispatch | its own create-time **spin pool**, one static block of volumes per pool thread | **OMP static** contiguous split |
| `clk512` (own probe) | **2.29 GHz** | **2.89 GHz** |
| `clk256` (own probe) | 2.79 GHz | 2.89 GHz |
| `sbw[rd/wr/cp]` (own probe, same node) | 5.55 / 8.52 / 14.84 | 6.45 / 11.66 / 18.67 |
| setup | 0.13 s | 2.33 s |
| delivered | 54.0 GB/s | 129.0 GB/s |

`L17_matrixsimd`'s process measures its own 512-bit clock at **2.29 GHz — exactly the part's
base frequency — while `L17_winograd`'s measures 2.89 GHz**, a 26% deficit, and measures the
node's memory system 14–26% slower on read/write/copy. Same node, same job, same hour. A
26% clock deficit does not explain 2.39×, but it is a *symptom* of a different process state,
and the one thing that differs structurally is 31 pinned spin-pool workers. `L17_rader` moved
from pool to OMP dispatch at streaming cells this round on the reasoning that "OMP is the only
dispatch shape with node evidence at 193 GB/s" and gained 1.71×. `L36_pfa` added
nap-after-1 ms to its idle workers this round because they "drag the all-core clock." Three
independent entries pointing at the same thing is a lead, not a coincidence.

---

## 5. Which `docs/LITERATURE.md` §4 open question moved, and how

### Primary: **§4.3, "Is axis fusion worth 3× or 3%?" — the re-opened L2↔DRAM clause is now built and measured**

LITERATURE.md §4.3's amendment names one construction and calls it "the largest untried
structural move on the board": *tile the batch so a tile fits L2, then run all three axes inside
the tile*, at the L2↔DRAM boundary where the bandwidth gap is 7× rather than the 2.6× of the
L1↔L2 experiments the earlier panel ran.

**mt_r3 built it, at L=36, and it is worth 3.13×.** `L36_pencilfused` deleted both of its r2
mechanisms and shipped "one 729 KB volume through a 1 MiB-L2-resident scratch tile, all three
axes inside it, in + NT-out compulsory traffic only (1.5 MB/vol)". Node result at L=36 B=512:
**17430 → 5561 µs, 43.9 → 137.5 GB/s, three processes within 1.2%.** `L36_mixedradix`'s pinned
`vol32-sntp` — the same construction reached independently — delivers **150.9 GB/s in all three
processes**, against 77.5 typical in r2. Two entries, one geometry, both above 137 GB/s where
the geometry's representative number one round ago was 77.

The same construction is behind three of the round's other four large improvements:
`L45_pfa` 64.2 → 109.0 GB/s at B=256, `L17_rader` 71.5 → 122.0 at B=4096, `L17_winograd`
holding 129.0 with its L2-resident mini-buffer and *plain* stores. **Note that NT stores are
not the discriminator** — `L17_winograd` reaches 129 GB/s with `nt=0`, and `L17_rader` reached
122 by *deleting* its near-tie NT pick. Tolmachev's rule survives, and the amendment's reading
of it is now the one with CPU numbers attached: the payoff is the avoided passes times the
bandwidth gap **of the two levels the tile actually spans**, and getting a private L2 tile per
thread is what buys the 7× gap. Store discipline is a second-order knob on top.

### And the answer to mt_r2's §5, which is a refutation

mt_r2's verdict named one mechanism as the round's central finding and one experiment as the
way to settle it: *"nobody has yet read `fr` under a 32-thread team at a streaming cell."* Six
entries ran it. Here is every reading:

| entry | cell | team | `fr` reading | GB/s |
|---|---|---|---|---|
| L6_unrolled | 6 / 65536 | T=32 | `fr=0/0` (calls 1 and 49) | 72.0 |
| L8_fusedaxes | 8 / 32768 | T=32 wide blocks | `fr0=0, fr=0, frmax=0` (trajectory, every 4th wide call) | 95.1 |
| L8_batchsimd | 8 / 32768 | T=32 dwell | `fr0=0, fr=0` | 93.8 |
| L8_radix8 | 8 / 32768 | T=32, 40-call dwell | `fr0=0, fr1=0` | 93.9 |
| L13_direct | 13 / 8192 | T=32 | `fr=0/0, frl=0` | 71.4 |
| L45_mixedradix | 45 / 256 | thr=32 | `fr0=0, fr=0` | 61.1 |
| **L36_mixedradix** | **36 / 512** | **T=32** | **`fr=7` / `fr=23` / `fr=0`** | **150.5 / 149.5 / 150.9** |

**The mechanism is dead, on two independent grounds.**

1. **Pages do not migrate.** `numa_balancing` is on (`nb=1` everywhere), the runs are seconds
   long, and `fr` is 0 at first scan and last scan in six of seven readings, including
   trajectories sampled every 4th call over 161 wide calls (`L8_fusedaxes`) and after a
   deliberate 0.9 s migration-settling warmup of full correct wide passes (`mw=120/0.84s`).
2. **Where it does migrate, it changes nothing.** `L36_mixedradix`'s three processes read
   `fr` = 7%, 23% and 0% of sampled caller pages remote and deliver **150.5, 149.5 and
   150.9 GB/s** — a 1% spread across a 23-point spread in remote fraction.

**And the wide team is not being foreclosed — it is genuinely slower.** Four entries raced team
width *on the caller's real buffers*, in the scored regime, and all four preferred the narrow
team:

| entry | cell | wide (T=32) | narrow (T=16) | dwell before deciding |
|---|---|---|---|---|
| L8_fusedaxes | 8 / 32768 | 0.2038 µs/t | **0.1719** | 161 wide calls, never locks, republishes both |
| L8_batchsimd | 8 / 32768 | 0.2329 | **0.1731** | ≥16 self-extending calls, cap 44 |
| L8_radix8 | 8 / 32768 | 0.233 | **0.175** | 40-call full-team dwell then paired revisits |
| L36_pfa | 36 / 512 | 24.44 µs/vol (`s32`), 25.39 (`d32`) | **20.05** (`s16`) | round-robin on real buffers, relock every 16 |

That is mt_r2's §6 order for L=8 — *"use the whole machine … force T=32 with a socket-aware
chunk split at that cell and re-measure"* — carried out three independent ways and answered in
the negative, with the extra socket costing 19–35%.

**Finally, the 200 GB/s datum the whole mechanism rested on is not reproducible.**
`L6_unrolled` delivered 199.6 GB/s in r2 at `3pass_nt_pf / T=32 / disp=omp`; the identical
strings in r3 deliver **72.0 GB/s with `fr=0/0`**, on a node whose libraries moved <2%
(§3.3). `L6_pfa`'s r2 "T=32 = 200, T=16 = 85 on the same binary" — the single measurement the
two-regime story was built on — becomes, in r3, **T=32 and T=16 within 0.2% of each other at
87.7 GB/s** on its own real-buffer re-race. Whatever produced 200 GB/s in r2 was a property of
that binary, not of the machine's page state, and finding it is §3.3's bisect.

**Net effect.** §4.3 moves substantially and in the direction the corpus predicted: the L2-tile
construction is real and large (3.13× at L=36, 137–151 GB/s where 77 was typical). §4.6's
precondition from mt_r2 ("a search is only as good as its arena's fidelity") is refined into
something stronger and cheaper: **at DRAM-bound cells, install from the working set and don't
race at all.** Every entry that replaced a race with a deterministic working-set gate this round
killed its lottery (`L17_winograd`, `L17_matrixsimd`, `L36_mixedradix`, `L36_pencilfused`,
`L17_rader`); the two remaining lotteries in the round are both in the one entry that made its
race *more* permissive (§3.2).

### Secondary: **§4.2, "L=17: dense-symmetric, Rader, or a hand-derived Winograd module?" — now closed**

| cell | dense conj-symmetric | Rader-17 | hand-derived Winograd module |
|---|---|---|---|
| B=1 | **5.976** | 6.955 | 7.563 |
| B=256 | **0.756** | 0.796 | 0.823 |
| B=4096 | 2.911 (Winograd engine on a pool) | 1.289 | **1.219** |

**Rader-17 is second at every cell and first at none, in all three rounds, at one thread and at
32.** §02 §7's "Rader is not the lever at L=17" is settled as far as this project can settle it.
The regime split is confirmed for a third round — dense below the aggregate-cache threshold,
the Winograd module above it — and this round it was made *deterministic* rather than raced, on
both sides, by a working-set gate. §4.2(c) (the exact op count for a full 17-point Winograd
module) remains open and still needs journal access.

### **Still not moved: §4.5 (L=8 padding, 4 KiB aliasing, `ld_blocks_partial.address_alias`)**

Third consecutive round blocked. `perf_event_paranoid` on the node still forbids reading the
counter. `L8_radix8` reasoned about it correctly — the phase-B out-store alias count is set by
`(scr − out) mod 4096`, an allocation lottery only the real buffers can price — and built a
MID governor to race the trio on them, gaining ~0% at B=2048. With team width now eliminated as
the explanation for the L=8 loss, §4.5 is the *only* remaining named hypothesis for it, and it
is unmeasurable on this node as configured. **This is now the highest-value administrative ask
in the project** (§6).

---

## 6. The single highest-value thing the next round should attack

**Before the per-geometry items, one thing that dominates them all.** The panel has now spent
two rounds instrumenting page placement and team width, and this round proved both are dead
ends: `fr = 0` everywhere, and the wide team measurably slower in four independent on-buffer
races. **Stop building placement instruments.** The thing that actually moved bandwidth 2–3×
this round is §4.3's L2-tile construction, and it has been applied at exactly one geometry.
Port it. Concretely: `L36_pencilfused`'s and `L36_mixedradix`'s shape — one volume at a time
through a per-thread scratch tile sized to fit one core's 1 MiB L2, all three axes inside the
tile, `in` read once and `out` written once — is currently unbuilt at L=13, L=23, L=45 (in the
sibling), and L=64, and those four geometries hold the four lowest streaming bandwidths in the
round (58.9–71.7 GB/s against L=36's 150.9).

Second, one administrative ask: **get `perf_event_paranoid` lowered on `p55n3`.**
`ld_blocks_partial.address_alias` has been the named next measurement for L=8 for three rounds
and is now the *only* surviving hypothesis for the project's oldest library loss. One request to
whoever administers the node unblocks it.

Per geometry, in priority order:

* **L=6 — bisect `L6_unrolled` `impl_2` vs `impl_3` at B=65536. This is the round's cheapest
  large win and it is worth a whole cell.** The r2 source delivers 2269 µs on this node; the r3
  source delivers 6290 with a byte-identical `l6_mt_call`, the same shipped pick, and `fr=0/0`
  (§3.3). Recovering it turns a 0.84× loss into a 1.91× win. Test the two hypotheses in order:
  (1) build `impl_3` with the create-time `rpool` never created (or destroyed before the
  tournament rather than after) and re-measure — if the number returns, idle spinners
  contending with the shipped OMP team is the mechanism, and it also explains §4.4;
  (2) if not, restore the four pruned kernels and re-measure for a code-layout effect. Report
  the `clk512` probe alongside. Secondary: `L6_pfa`'s re-race now shows T=16 and T=32 tied at
  87.7 GB/s, so the team-width question at L=6 is closed — delete the shrink race there.
* **L=8 — the cell is 7% away and team width is not the route; §4.5 is.** Four measurements say
  the wide team costs 19–35% (§5), so mt_r2's L=8 order is answered and should not be
  re-attempted. The remaining 7% against `fftw3_patient`'s minimum is an allocation-level
  question: the L=8 volume stride is exactly 8192 B = 2 × 4096, so with two page-aligned
  buffers every load from `in` falsely aliases a recent store to `out` in the low 12 bits
  (LITERATURE §4.5's second hazard, §08 §1.8). Two moves that do not need the counter:
  (a) offset the *scratch* base by an odd number of cache lines and sweep the offset — ducc0's
  `if ((dstride & 256) == 0) dstride += 16;` guard, which §4.5 says to copy rather than reason
  about; (b) measure by construction — run one forced configuration at several `(scr − out) mod
  4096` values and see whether the 7% moves. Also worth noting for the record: `L8_fusedaxes`'s
  no-lock cyclic governor is the right *reporting* design even though its answer was negative —
  it publishes both configs' real-buffer costs every run instead of hiding a lock, and that is
  why this round could settle the question at all.
* **L=17 — close the 2.39× gap between two copies of the same engine (§4.4).** `L17_matrixsimd`
  runs `L17_winograd`'s engine, verbatim, at 11924 against winograd's 4991.7, with its own
  probes reading `clk512` 2.29 GHz vs 2.89 and node memory bandwidth 14–26% lower. The engine is
  not the variable; the dispatch and the process's thread picture are. Run `L17_matrixsimd`'s
  engine under an OMP static split (the only difference winograd's record and `L17_rader`'s
  1.71× gain both point at) and publish `clk512` on both sides. This is 2.4× on a cell the
  panel already wins 3.36×, and it is the same experiment as L=6's bisect — if both come back
  positive, "never leave a spin pool alive at a streaming cell" is a panel-wide rule and worth
  three cells. Also: `L17_rader` has now been second at all three cells in three consecutive
  rounds; §4.2(a) is closed and the entry should either merge into one of the winners or take a
  genuinely different swing.
* **L=36 — find the difference between 9.9 and 20.0 µs/vol, then push past 151 GB/s.** Three
  entries at the same cell describe the same nominal shape (per-thread L2-resident volume
  scratch, NT stream to `out`, full team) and deliver 150.9, 137.5 and 77.3 GB/s.
  `L36_pfa`'s execute-time governor is *honest* about what it runs — `s16=20.05` in-arena
  against 19.32 delivered — so it is not a tuner-fidelity problem: none of its three raced
  shapes is the winning shape. Diff `L36_pfa`'s `nt1` phase-1/phase-2 against
  `L36_mixedradix`'s `sntp` and `L36_pencilfused`'s mode-2 paced read cursor; the answer is
  worth 2.0× at L=36 and, transferred, probably more elsewhere. Then: 151 GB/s is roughly 65%
  of two sockets' DDR4-2666 peak and is now the project's best streaming number, so L=36 is the
  right place to ask what the last 35% costs.
* **L=13 — remove `L13_rader`'s two bad picks and build the L2 tile.** Its
  adopt-unless-vetoed rule installed a config its own arena priced 34% worse (B=1, 2 of 3
  processes) and its new `n24` mode cost 1.50× (B=512, 2 of 3); both are one-line tuner fixes
  with the numbers already in its description strings (§3.2). With those gone it is a credible
  competitor at 302 ns/vol. Then the real work: both L=13 entries sit at 69–72 GB/s where
  `L17_winograd` gets 129 on the same node, and neither runs a per-thread L2 tile. The B=8192
  loss to FFTW is 0.61× as scored and 1.19× *in the panel's favour* on medians, so the honest
  target is the bandwidth, not the statistic.
* **L=23 — the lowest streaming bandwidth in the round (65–67 GB/s) and no L2 tile.** Both
  entries are volume-parallel row–column at 380 KiB/volume compulsory. L=23's volume is 190 KiB,
  which fits a 1 MiB L2 five times over — this is the geometry where §4.3's construction should
  be easiest and it is unbuilt. `L23_rader`'s two-level socket-tree barrier and weighted
  near/far split were built this round and neither is what is binding. Secondary: B=1's
  23-planes-over-32-threads imbalance is named in both records and unaddressed for three rounds.
* **L=45 — run the `L45_pfa`-vs-`L45_mixedradix` A/B and port the winner.** `L45_pfa` recovered
  1.70× and now runs the same geometry 1.78× faster than its sibling (109.0 vs 61.1 GB/s) on
  the same node. It names two causes: the 16-mod-64 `Pt` misalignment and the reinstated OpenMP
  two-sweep with the pool napped. Check `L45_mixedradix` for the same alignment bug first — it
  is a five-minute read and its sibling's experience says Sapphire Rapids hides it completely.
* **L=64 — bisect `L64_radix8` (§3.4), then take B=8 from MKL.** The 2.05× regression sits on an
  identical pick with an arena that was accurate one round ago; the two named suspects are the
  dropped trailing gang barrier and the 34th `WREG` slice, and `impl_2` vs `impl_3` is a clean
  A/B. B=8 remains the panel's only genuine library loss (0.94×, `sd` 0.2% on both sides, 6.6%,
  unchanged) and `L64_blocked`'s named residual — the gang-internal all-to-all where each lane
  reads 3/4 of its second-pass input from sibling lanes — is still the route. Do not re-open
  `scs16`: its author's own pre-registered criterion refuted it (§3.4).

---

## 7. Curation decision

Applying `docs/CURATION.md`'s four grounds in order. All 19 strategy records stay in
`strategies/` regardless; `exemplars/mt_r3/` is the reading list. Ties within 1.5% are treated
as ties, not wins, since that is the round's typical cross-process spread. Rule 4 ("anything
that beat a library baseline") is used as a tiebreaker rather than a blanket — every one of the
19 entries beat the best library at at least one cell this round, so read literally it would
promote everything and stop being curation; the near-duplicate prohibition governs.

**Promoted (16):**

| entry | ground |
|---|---|
| `L6_pfa` | rule 1 — fastest L=6 at all three cells (1.76× / 2.23× over MKL); and its real-buffer re-race closed mt_r2's headline claim by measuring T=16 and T=32 **tied** at 87.7 GB/s where r2 read 85 vs 200 |
| `L6_unrolled` | rule 3 — the round's largest regression, 2.77× at B=65536, stable in all three processes, with a byte-identical execute path, an unchanged pick, `fr=0/0`, and its own arena moving 36.5 → 53.1 ns/vol. `impl_2` is a working 2269 µs on this node; the record documents both numbers and the bisect |
| `L8_fusedaxes` | rule 1 — fastest L=8 at all three cells; and it carries the no-lock cyclic governor that publishes wide (0.2038) and safe (0.1719) real-buffer costs every run with a disclosed 0.9 s settling warmup — the primary instrument behind §5 |
| `L8_radix8` | rule 2 — structurally distinct (`1f` half-team) and within 1.2% at B=32768; its 40-call deep dwell with `fr1` is the cleanest single form of the experiment mt_r2 ordered, and its MID-governor analysis of `(scr − out) mod 4096` is where §4.5 restarts |
| `L13_direct` | rule 1 — fastest L=13 at all three cells on representative numbers; the only L=13 entry with no pick lottery (`sd` ≤ 0.2% at every cell) |
| `L13_rader` | rule 3 — two self-inflicted pick failures with the killing numbers in its own telemetry (B=1: installed `p2` against an arena reading it 34% worse than the incumbent, 8.5 vs 6.1 µs; B=512: `n24` in 2 of 3 processes, 231 vs 155 µs), plus the repair of mt_r2's build failure and the lesson written down. Structurally the only Rader-13 |
| `L17_matrixsimd` | rule 1 — fastest L=17 at B=1 (3.87× MKL) and B=256 (5.16×); and rule 3 at B=4096, where it runs `L17_winograd`'s engine verbatim at 2.39× the original's time with `clk512` 2.29 vs 2.89 GHz in its own probe — §4.4, the round's most actionable single gap |
| `L17_winograd` | rule 1 at cell + rule 2 — fastest L=17 at B=4096 (129.0 GB/s, 3.36× the best library), and the round's best-calibrated record: it predicted the 1.73× lottery would become one configuration in every process and delivered 0.4% spread |
| `L23_matrixsimd` | rule 1 — fastest L=23 at B=128 (5.49× over the best library, the round's largest ratio); stable at all three cells for three rounds |
| `L23_rader` | rule 1 at cell + rule 2 — fastest L=23 at B=1 and B=2048 (both ties, ≤1.4%), structurally distinct from the dense form, and 1.23× better at B=2048 on the aggregate-cache arena it borrowed |
| `L36_mixedradix` | rule 1 — fastest L=36 at B=1 and B=512; **150.9 GB/s in all three processes**, the round's best result, converting r2's 1-in-3 into 3-of-3 by installing from the working set instead of racing |
| `L36_pencilfused` | rule 2 + the §4.3 result — the only entry that built the L2↔DRAM tile construction LITERATURE.md names as the largest untried structural move, 3.13× at B=512 (43.9 → 137.5 GB/s), after measuring its own r2 mechanism dead and deleting it |
| `L36_pfa` | rule 1 at cell (fastest L=36 at B=32) + rule 3 — the only on-buffer team-width race at L=36 (`s16` 20.05 / `s32` 24.44 / `d32` 25.39 µs/vol, locked `s16`, delivered 19.32), an honest instrument whose contradiction with its sibling's 9.9 at T=32 defines the next round's L=36 question |
| `L45_pfa` | rule 1 — fastest L=45 at all three cells; recovered mt_r2's 1.69× regression and beat its own mt_r1 number (6851 vs 6885), 64.2 → 109.0 GB/s, by finding a 16-mod-64 scratch misalignment by inspection that no measurement on its dev machine could reveal |
| `L64_blocked` | rule 1 — fastest L=64 at B=1 and B=128; 1.23× better at B=128 at 113.8 GB/s, and it halved its own r2 pick lottery (1.50× → 1.24×) by dropping `dyn` |
| `L64_radix8` | rule 3 — 2.05× regression at B=128 on an **identical pick string** whose own arena price moved only 81 → 84, so the tuner's ranking is exonerated and the execute path is convicted; plus two structural bets refuted by its own pre-registered criteria (`scs16` "≥130 means the model is wrong" → 136.3; B=128 "65–75 µs/vol" → 142.3) |

**Not promoted (3):**

* **`L8_batchsimd`** — near-duplicate of `L8_radix8` (identical arithmetic: 1248 vector FP + 896
  shuffles), and fastest at no cell: 0.9% behind at B=1, 2.2% at B=2048, 1.3% at B=32768. Its
  contributions are recorded here rather than in an exemplar: it delivered the round's third
  independent confirmation of §5 (`gov{T16/nt-s0=0.1731, T32/nt-s0=0.2329, lock=cfg0}` in all
  three processes), and it reverted its own mt_r2 tree collect on the pre-registered criterion
  it had written down ("if `poolrt{32}` is not well below ~2.5, revert" — it read 5.2–5.5), for
  1.04× at B=2048. Reverting on your own stated criterion is the behaviour to reward and it does
  not need an exemplar to be remembered.
* **`L17_rader`** — second at all three L=17 cells for the **third consecutive round**, first at
  none, and its own description still states the kernel came from `L17_winograd`, so it is a
  near-duplicate of a promoted entry. It improved 1.71× at B=4096 (to 122.0 GB/s) by pinning the
  wide team and refusing near-tie NT picks — a lesson `L17_winograd` already carries. Its
  contribution is the negative that closes §4.2(a), and one hypothesis this round refuted: its
  setup-time-as-migration-dwell model (fast entries have 2.7–12.8 s creates, slow ones 0.7 s)
  does not survive the r3 data — `L8_fusedaxes` reaches 95 GB/s on a 0.43 s create,
  `L45_mixedradix` 61 GB/s on 2.03 s, `L64_radix8` 59 GB/s on 2.17 s, and `L6_pfa` 88 GB/s on
  2.8 s where r2's 12.8 s create bought it 175.
* **`L45_mixedradix`** — 1.78× behind `L45_pfa` at B=256 (61.1 vs 109.0 GB/s) on the same
  geometry, the same node and structurally the same PFA 9×5, so it is a near-duplicate of a
  promoted entry and no longer close. Its mt_r2 exemplar value — "the worked example of how to
  fix an arena-fidelity failure" — has been superseded by `L45_pfa` doing it better and 1.78×
  faster at the same cell. Its remaining worth is precisely as the negative control for
  `L45_pfa`'s two fixes, and that A/B is fully recorded here and in both strategy records; the
  first thing to check in it is whether it carries the same 16-mod-64 scratch misalignment. Its
  mt_r2 exemplar stands.

PROMOTE: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pencilfused L36_pfa L45_pfa L64_blocked L64_radix8
