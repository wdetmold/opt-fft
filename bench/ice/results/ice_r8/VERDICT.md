# Round ice_r8 — monitor's verdict

Measured on `a80n0.lqcd.mit`, slurm job 438637, 2026-08-23T16:06 EDT.
Aggregated: `leaderboard.txt` (as written by the sweep) and
`leaderboard_regated.txt` (same timings, re-aggregated after the chain gate was
recovered — see §3.1). **The two files rank identically.**

> **Correction to the round brief.** The brief describes the scoring node as a
> Xeon Gold 5218 (Cascade Lake) with 1 MB L2 and downclocked AVX-512, and says
> implementers develop on Sapphire Rapids. Neither is true of this round.
> `environment.txt` records **Intel Xeon Gold 6326 @ 2.90 GHz** — Ice Lake-SP,
> not Cascade Lake (the ISA line carries `avx512_vbmi`, `avx512ifma`,
> `avx512_bitalg`, `avx512_vnni`, `avx512_vpopcntdq`, none of which exist on
> Cascade Lake). And every strategy record in this round documents development
> *on this same node* through `tryout.sh` under a core lease. There is
> consequently no cross-machine discrepancy to attribute (§4). The literature
> corpus's Gold 5218 turbo analysis (§08, LITERATURE.md §4.8 item 6) describes a
> part we are not scoring on and should be re-read with that in mind.

---

## 1. Headline per geometry

The graded configuration (`cases.txt`) is `<L>:<batch>:<chain>` and is
**batched-only at every L**. One timed unit is a chain of `m` map-steps over `B`
volumes; `per-transform` divides by both.

### 1a. Batched — the measured case

| L | B | m | fastest correct panel entry | best library | panel speedup |
|---|---|---|---|---|---|
| **6** | 64 | 4856 | **L6_unrolled 0.207 µs** (40.51 GF/s) | mkl_dfti 0.940 µs (8.91) | **4.55×** |
| **8** | 64 | 2572 | **L8_radix8 0.500 µs** (46.07 GF/s) | mkl_dfti 2.099 µs (10.98) | **4.20×** |
| **17** | 32 | 98 | **L17_matrixsimd 8.829 µs** (34.12 GF/s) | ducc0_c2c 86.024 µs (3.50) | **9.74×** |
| **36** | 8 | 64 | **L36_mixedradix 100.212 µs** (36.10 GF/s) | mkl_dfti 284.161 µs (12.73) | **2.84×** |

The other four geometries, for completeness:

| L | B | m | fastest correct panel entry | best library | panel speedup |
|---|---|---|---|---|---|
| 13 | 32 | 1278 | L13_direct 3.438 µs (35.48) | mkl2026_dfti 11.834 µs (10.30) | 3.44× |
| 23 | 16 | 165 | L23_matrixsimd 33.805 µs (24.42) | ducc0_c2c 247.561 µs (3.33) | 7.32× |
| 45 | 4 | 177 | L45_pfa 216.968 µs (34.60) | mkl_dfti 757.627 µs (9.91) | 3.49× |
| 64 | 2 | 134 | L64_blocked 634.801 µs (37.17) | mkl_dfti 1720.547 µs (13.71) | 2.71× |

**Every one of the 19 panel entries beat every library at its own geometry.** At
L=17 and L=23 the best library is ducc0, not MKL; everywhere else MKL 2022 leads
the baselines (MKL 2026 is slower than MKL 2022 at six of eight geometries).

### 1b. Non-batched — not measured, this round or any round

There is no non-batched headline to report, and this is not an omission of the
timing pass: `cases.txt` contains no `B=1` row, and **no round in the ice series
(r1–r8) has ever measured one.** `leaderboard.py` would label such a case
"non-batched"; the string never appears. The brief asks for a number the harness
does not produce.

The only B=1 figures in existence are implementer dev measurements, taken by
hand on leased cores and **not gated, not aggregated, and not comparable across
entries** (each names a different window). Recorded here so the gap is visible,
explicitly as self-reported and unscored:

| L | entry | self-reported B=1 µs/step | vs its own B=64/32/8 number |
|---|---|---|---|
| 6 | L6_unrolled | 0.299 | 1.44× slower |
| 6 | L6_pfa | 0.322 | 1.53× |
| 8 | L8_radix8 | 0.550 | 1.10× |
| 8 | L8_batchsimd | 0.556 | 1.09× |
| 17 | L17_matrixsimd | 13.359 | 1.51× |
| 17 | L17_rader | 13.567 | 1.50× |
| 17 | L17_winograd | 11.138 | 1.23× |
| 36 | L36_mixedradix | 99.8–100.7 | ~1.00× |
| 36 | L36_pfa | 124.256 | 1.23× |
| 36 | L36_pencilfused | 115.456 | 1.13× |

The pattern is consistent and worth acting on: the batch-lane and volume-SoA
engines that won every cell this round **require a full group of 4 or 8 volumes**
and fall back to older per-volume code below it. At L=6 that fallback costs 44%.
If the non-batched case is ever meant to be graded, it must be put in
`cases.txt`, because right now it is the one part of the design space with no
measurement and no pressure on it.

---

## 2. What changed since ice_r7, per geometry

Per-transform µs, r7 → r8. Negative = faster.

| L | entry | r7 | r8 | Δ | note |
|---|---|---|---|---|---|
| 6 | L6_unrolled | 0.238 | **0.207** | **−13.0%** | store-fused map (`g8f`) + H/R2 alternation |
| 6 | L6_pfa | 0.320 | 0.210 | **−34.4%** | ported warm's SoA-8 `s8` engine wholesale |
| 8 | L8_radix8 | 0.551 | **0.500** | **−9.3%** | adopted `bl8` + column-relaid c (CP) + join-FMA |
| 8 | L8_fusedaxes | 0.511 | 0.507 | −0.8% | sched-pressure attribute + c52 codelet |
| 8 | L8_batchsimd | 0.555 | 0.509 | −8.3% | adopted `bl8` |
| 13 | L13_direct | 3.471 | **3.438** | −1.0% | hugepage buffers |
| 13 | L13_rader | 3.688 | 3.463 | −6.1% | mf=3 + opt-sched |
| 17 | L17_matrixsimd | 9.035 | **8.829** | −2.3% | quartic rsqrt + x2 map interleave + MG=2 |
| 17 | L17_rader | 11.919 | 9.028 | **−24.3%** | ported matrixsimd's chain v7 |
| 17 | L17_winograd | 10.922 *(rejected)* | 9.033 | −17.3% | ported chain v7 **and fixed the map** |
| 23 | L23_matrixsimd | 33.876 | **33.805** | −0.2% | round came out empty; shape unchanged |
| 23 | L23_rader | 35.919 | 34.209 | −4.8% | plain-Z rotation on the aligned layout |
| 36 | L36_mixedradix | 99.809 | **100.212** | +0.4% | per-volume cperm memoization |
| 36 | L36_pfa | 100.296 | 100.247 | −0.05% | r7 shape ships unchanged |
| 36 | L36_pencilfused | 107.009 | 102.427 | −4.3% | broadcast+vr exec mode, pw=4 |
| 45 | L45_pfa | 231.220 | **216.968** | **−6.2%** | custody schedule ported onto q4 (`qc`) |
| 45 | L45_mixedradix | 225.571 | 219.689 | −2.6% | eager THP fault-in |
| 64 | L64_blocked | 608.213 *(rejected)* | **634.801** | +4.4% | **exact map** (`md3` all-FMA) + force-inline |
| 64 | L64_radix8 | 614.640 *(rejected)* | 636.477 | +3.6% | **exact map** (r4 exact tier restored) |

**Rank flips:** L=8 (fusedaxes → radix8), L=45 (mixedradix → pfa). Both are the
runner-up adopting the winner's structure and then adding to it — the panel's
intended dynamic, working.

### Did anything regress?

**Three entries show a higher number than r7. None is a genuine regression, and
one deserves scrutiny anyway.**

1. **L=64, both entries, +3.6% and +4.4% — not a regression; the r7 numbers were
   illegal.** Both L64 entries FAILED r7's two-step precision gate (L64_blocked
   1.38e-13, L64_radix8 1.315e-13, against a 3e-14 contract) because both carried
   a cubic-Newton "cheap" map ladder at ~4e-13 per application. Both fixed it
   independently this round and both priced the fix in advance: L64_radix8
   predicted +3.2% and measured +3.6%; L64_blocked predicted +5.6% and measured
   +4.4%. **The correct comparison is against the last legal L=64 number — r5's
   0.271 s ≈ 1011 µs/step — against which r8's 634.801 µs is a 37% improvement.**
   This is the round's most important result and it went the right way: the panel
   was told its fastest entries were wrong, and it made them right for 4%.

2. **L36_mixedradix, +0.4% on the min — inside the window band, but its run
   spread is the round's one unexplained number.** Its spread went **7.2% → 29.8%**,
   the worst on the board by 6×, while its own cell-mates read 2.8% and 0.7%. The
   implementer's explicit prediction was the opposite: having moved the `cpfill`
   c-staging out of the timed samples, they predicted "the run spread should
   collapse from r7's 7.2% to a few percent," and gave a contended-class floor of
   104–106 µs. A 29.8% spread on a 100.212 µs min implies one process read ~130 µs,
   well outside even that. **The 100.212 min rests on the best of a widely
   scattered set and its 0.03 µs lead over L36_pfa (which read 2.8% spread) is not
   a real ordering.** Treat L=36 as a tie; see §6.

3. **L6_unrolled's 13.9% spread and L8_batchsimd's 13.8%** are the same disease in
   milder form — both entries reported sd ≤0.06% across three leases in their own
   dev runs. Three entries with ~14–30% spread, in cells whose neighbours read
   0.0–0.2%, points at the sweep's process scheduling rather than at the code. At
   L=6 the consequence matters: **L6_unrolled 0.207 (13.9% spread) versus L6_pfa
   0.210 (0.0% spread) is not a decided cell.**

---

## 3. Correctness, build, crash, missing — the adversarial pass

**No entry failed to build. No entry crashed. No entry is missing.** All 19
agents exited 0 (`agents/exits.txt`), all 19 appear in the leaderboard, no
`failures.txt` was produced, and `build_errors.txt` contains a single *warning*
(`MAPPAIR_D` redefined at `impl/L36_mixedradix.c:1215`, shadowing the definition
at :1243) — the entry compiled, ran, and passed every gate. That warning should
still be cleaned up; a macro silently redefined between two includes is how a
future round ships the wrong kernel.

But the round shipped with its main correctness gate dead, and that is the
finding.

### 3.1 The chain gate did not run for seven of eight geometries

`check.py` used `math.floor` at line 94 without `import math`. Evidence:

* `check.log` contains **66 `NameError: name 'math' is not defined`** tracebacks
  and only **9 successful `map-chain` verdicts** — the 9 being L=64, which ran at
  17:37–17:42, after the missing import was added at 17:25.
* The crash happens *after* the single-transform `PASS` prints and *after*
  `rel_chain` and `anchor` are computed, so the run looks successful in the log
  and writes a `c_*.json` with no `chain_rel_l2` key.
* `leaderboard.py:verdict_ok` keys entirely off `"chain_rel_l2" in chk`. With the
  key absent it silently degrades to the **single-transform** check and does not
  even consult the one-step `o_*.json`. That is why `leaderboard.txt` shows
  `ok 2.4e-16` at L=6–45 and the r7-style `ok ch=…/… 1s=…` only at L=64.

**The quantity that was timed is the m-step chain. For L = 6, 8, 13, 17, 23, 36
and 45, the m-step chain was never compared to anything.** The exposure is real
and not hypothetical: the two-part gate caught three entries in r7
(L17_winograd, L64_blocked, L64_radix8), and several r8 entries ship a
*distinct steady-state chain path* that the single transform does not touch at
all — L8_radix8's "steady step", L6_pfa's `s8` group path, L17_rader's `ms7`
arena. A fast wrong answer in any of those would have scored clean.

Four implementers found this bug independently and wrote it into their records
(`L6_pfa`, `L8_radix8`, `L8_fusedaxes`, `L17_matrixsimd`, `L17_rader`,
`L17_winograd`, `L36_mixedradix`, `L36_pfa` all name it, most with the line
number and the one-word fix). It was reported and it still shipped. That is a
process failure, not an implementer failure.

### 3.2 The gate has now been recovered, and nothing was hiding behind it

The missing import is fixed (`check.py` line 6, uncommitted in the working tree)
and a re-verification pass ran the full m-step chain check for every backend at
every geometry, finishing 17:46. Results are in the round's `c_*.json` and
re-aggregated into `leaderboard_regated.txt`.

**Every panel entry passes. No ranking changes. Most panel entries drift *less*
than MKL does.**

| L | tol | worst panel entry | mkl_dfti | best library |
|---|---|---|---|---|
| 6 | 1e-05 | L6_pfa 2.7e-08 | 2.3e-08 | fftw3 1.6e-09 |
| 8 | 1e-09 | L8_radix8 3.6e-12 | 2.3e-12 | mkl 2.3e-12 |
| 13 | 1e-10 | L13_rader 1.3e-13 | **1.8e-13** | ducc0 1.1e-13 |
| 17 | 1e-10 | L17_winograd 1.7e-14 | 1.3e-14 | ducc0 1.2e-14 |
| 23 | 1e-10 | L23_matrixsimd 3.5e-14 | 3.0e-14 | ducc0 2.2e-14 |
| 36 | 1e-10 | L36_mixedradix 2.1e-14 | **2.7e-14** | ducc0 1.2e-14 |
| 45 | 1e-10 | L45_pfa 3.2e-14 | **3.0e-14** | ducc0 2.6e-14 |
| 64 | 1e-10 | L64_blocked 3.8e-14 | 2.6e-14 | ducc0 2.4e-14 |

At L=13, L=36 and L=45 the panel's *worst* entry drifts less than MKL 2022 over
the same chaotic chain. The one-step/two-step precision gate (`o_*.json`) ran
correctly all round and every entry passes at 5e-16 – 2e-15 against a 1.5e-14/step
contract, 15–60× margin.

**The one thing still unverified on the chain path is `baseline_matrix`** at
L=6…45 — the O(L⁴) reference floor, 44–150× slower than the winners and not a
competitor. It is verified at L=64. Not worth a re-run.

### 3.3 The one-step gate as specified cannot be run at all

Five implementers independently report that `driver --map --chain 1` **segfaults
every chain-exporting entry**: `driver.c:141` allocates `pong` only when
`chain > 1` and then passes it (NULL) as `fft3d_chain`'s destination, and never
writes the `.chain` file at m=1. The precision contract is therefore enforced
through `--chain 2` as a proxy (which is why `check.py`'s one-step branch accepts
`m <= 2` at `tol = 1.5e-14 * m`). Several entries now carry their own NULL guard;
the rest still crash. This is a live harness defect and it means **the m=1 code
path of every entry has never met a real harness check.**

### 3.4 Harness defects to fix before ice_r9

Reported by implementers this round, all one-liners, all still live:

1. `check.py` — `import math` (**fixed in the working tree, must be committed**).
2. `driver.c:141` — allocate `pong` at `chain == 1`, or the one-step gate is
   unrunnable.
3. `tryout.sh:49` — `'$W/c.bin'` single-quoted inside `$()`, never expands; the
   map gate silently never runs inside `tryout`. Two sites now (line 36 works
   only if `W` is exported).
4. `reserve.sh --status` false-negatives from the login node because `squeue` is
   not on PATH; the real fix is
   `PATH=/opt/software/slurm-19.05.8.1-cuda-11.8/bin`.
5. `impl_8/L36_mixedradix.c:1215` — the `MAPPAIR_D` redefinition warning.

Items 1 and 2 are correctness infrastructure and should block the next round.

---

## 4. Claimed versus measured

**The brief anticipates a large claimed/measured gap attributable to a machine
difference. There is no such gap, and there is no machine difference.** Every
implementer developed on `a80n0` — the scoring node — through `tryout.sh` under a
core lease, and says so. Agreement is exceptional:

| entry | claimed (dev) | measured | Δ |
|---|---|---|---|
| L6_unrolled | 0.2069–0.2071 | 0.207 | ~0% |
| L6_pfa | 0.210 | 0.210 | 0% |
| L8_radix8 | 0.501–0.503 | 0.500 | −0.3% |
| L8_fusedaxes | 0.506–0.507 | 0.507 | ~0% |
| L8_batchsimd | 0.509–0.510 | 0.509 | ~0% |
| L13_direct | 3.439–3.446 | 3.438 | ~0% |
| L13_rader | 3.460–3.473 | 3.463 | ~0% |
| L17_matrixsimd | 8.831 (proj 8.8–9.0) | 8.829 | ~0% |
| L17_rader | proj 9.0–9.2 | 9.028 | in band |
| L17_winograd | 9.041 (proj 8.9–9.1) | 9.033 | ~0% |
| L23_matrixsimd | proj 33.8–34.0 | 33.805 | in band |
| L23_rader | proj ~34.2 | 34.209 | ~0% |
| L36_mixedradix | proj 99–101 quiet | 100.212 | in band |
| L36_pfa | proj 100–102 fast-class | 100.247 | in band |
| L36_pencilfused | 102.245 | 102.427 | +0.2% |
| L45_pfa | 215.1 quiet | 216.968 | +0.9% |
| L45_mixedradix | 218.1 fast-state | 219.689 | +0.7% |
| L64_radix8 | 633.8–635.7 | 636.477 | +0.1% |
| L64_blocked | 653.8–656.3 | **634.801** | **−3.2%** |

Two entries are worth naming, both in the panel's favour or neutral:

* **L64_blocked measured 3.2% *faster* than its own dev floor.** Its dev A/Bs
  were run in a contended window (it reports MKL at 1722 in dev; the scoring
  window read 1720). Under-claiming is the honest direction, and its prediction
  of the *cost* of the exact map (+5.6%) was itself conservative — the scored
  delta is +4.4%.
* **L36_mixedradix's point estimate was right and its variance estimate was
  wrong** (§2). It predicted the spread would collapse to a few percent; it went
  to 29.8%.

The real source of dispersion in this project is not the machine, it is the
**window class on a shared node** — quiet versus contended — which the records
document at length (L36_pfa saw the same binary read 100.3 and 115.6; L45_mixedradix
saw 218.1 versus 252–256 with MKL moving 758 → 790 alongside). Every implementer
has converged on the same protocol: same-lease alternating full binaries, discard
the first invocation, and class the window by the same-window MKL reading before
believing any cross-window delta. **That protocol, not a hardware correction, is
what makes these numbers agree.** It should be written into the brief as the
standard, and the sweep itself should adopt it — the three entries with 14–30%
spread (§2) are precisely the ones the sweep measured without it.

---

## 5. Which LITERATURE.md §4 open question moved

### §4.3 "Is axis fusion worth 3× or 3%?" — **moved decisively, to zero.**

This is the section §07 gap 7 calls "the single most important number this
project will have to measure for itself." r3 answered it at single-digit percent.
**L8_radix8 this round ran the cleanest version of the experiment yet and got
0%**, by separating the confound that made every previous attempt ambiguous.
Four schedules over one identical kernel set, same lease, graded case:

| schedule | L2 sweeps/step | µs/xform |
|---|---|---|
| `bl` plain two-pass (ships) | 2 | **0.501–0.503** |
| `bs` block-staged, map feeds stores | **1** | 0.502–0.504 |
| `bd` register-fused double-step | **1** | 0.553 (+10%) |
| `bp` x-first | 2 | 0.553 (+10%) |

`bs` and `bd` both halve the sweeps; `bs` ties the two-pass form and `bd` loses
10%. **So halving L2 traffic is worth exactly nothing, and the entire apparent
cost of "fusion" is critical-path placement of the map** — in `bd` the map's
ladder gates the *next* codelet, in `bs` it feeds stores. That isolates pass-count
from map-placement for the first time, and pass-count measures zero. Tolmachev's
rule (§07 §1.6) survives with the right reading: the payoff is the avoided passes
times the *bandwidth gap*, and across L1↔L2 with hardware streamers running, the
gap that matters is already covered.

L6_unrolled supplies the confirming asymmetry from the other side: store-side
fusion won (−3.0% at exact tier, −9.1% at fast tier) where r7's *load*-side
fusion of the same map lost (0.248 vs 0.238). Same fusion, opposite sign,
decided by which end of the dependence the ladder hangs off.

**§4.3's re-opened L2↔DRAM regime — "the largest untried structural move on the
board" — was again not tested**, and cannot be with the current `cases.txt`:
every chain at L=36/45/64 is L2-resident by construction (11–16 MiB working set,
but the per-chain state is ~1.5 MB at L=36). Testing it needs a batch large
enough to spill L2, which means a new row in `cases.txt`, not a new kernel.

### §4.4 split-vs-interleaved — **closed, but this round adds a size boundary to it.**

§08 §5.4 closed §4.4 for split complex, and §08 §1.10 fixed the granule at
**8 volumes per zmm for split-complex double**, from the cache line. At L=6 and
L=8 the panel converged on exactly that and it produced the round's largest gains
(L6_pfa −34.4%, L8_radix8 −9.3%, L8_batchsimd −8.3%), with zero shuffles in the
steady state as predicted.

**At L=17 the same rule is refuted.** All three L=17 entries independently
rejected the 8-volume split form and run **4 volumes per zmm, interleaved
complex**, on an arithmetic argument they each state: ~336 vector ops per 8
pencils for the split form versus 296+8 for the interleaved-SoA one, and double
the L2 footprint (314 KB × 2 arenas is already the L2 budget at 4 volumes). That
took L17_rader from 11.919 to 9.028 and L17_winograd from 10.922 to 9.033.
So §08 §1.10's granule rule is **validated at L ≤ 8 and refuted at L = 17**, and
the crossover is set by whether both the state and the c-field arena still fit L2
at the chosen granule. That qualification should go back into §4.4.

### §4.2 "L=17: dense-symmetric, Rader, or Winograd?" — **settled, and the answer is "none of the above."**

All three positions in the corpus were implemented and all three converged to
within **2.3%**: matrixsimd (dense conj-symmetric) 8.829, rader (Rader-17) 9.028,
winograd (hand-derived 17-point module, 296 FP instructions) 9.033. They converged
because all three now run the *same* chain engine and the same 148-FP-op nested
cyclic/negacyclic chunk — the two challengers ported the winner's chain v7 and
their records say so plainly. **§02 §7's "Rader is not the lever at L=17" is
confirmed; the lever is the chain layout, and the kernel choice is worth under
3%.** §4.2(c)'s missing exact op count for a full 17-point Winograd module is now
supplied empirically by our own: **296 FP instructions**.

### §4.6 model versus search for the schedule — **moved, and the sign is now certain.**

Three entries independently established that **gcc 11.4 runs no pre-RA scheduling
on x86 at -O3**, and each got paid for working around it: L8_fusedaxes'
`-fschedule-insns -fsched-pressure` as a function attribute, −0.8% and
bit-identical; L17_matrixsimd's hand-interleaved two-pair map body, −0.5%
bit-identical; L64_blocked's `always_inline` on the row body, −0.6% bit-identical.
§01 §1.5 asked whether disabling the pre-RA scheduler still pays as it did on 1999
SPARC. **The answer on this part is the opposite: it needs turning on.** Each
lever is small, but all three are free and bit-exact, and no entry has yet applied
them systematically.

---

## 6. The single highest-value thing ice_r9 should attack, per geometry

**L=6 — settle the cell, then attack the last structural seam.** The cell is not
decided: 0.207 with 13.9% spread against 0.210 with 0.0%, from two entries that
have now converged on the same store-fused SoA-8 engine. Re-run L=6 under the
implementers' own protocol (alternating same-lease binaries, first invocation
discarded) before anything else. Then the one structure left, from L6_pfa's own
accounting: ~30% issue slack remains (~1000 zmm uops/vol-step retiring at
1.44/cyc), the three sweeps are dependency-chained per plane, and
**software-pipelining plane x+1's y-sweep under plane x's z-sweep** is untested on
the *store* side — the load-side version is the measured loser (0.248 vs 0.238),
and §5 shows that distinction is exactly what decides the sign.

**L=8 — stop building and measure the ports.** All three entries sit at
1.25–1.29× their own computed pool floor (0.500 vs ~0.400 µs) with every
scheduling, prefetch, map-ladder, codelet and pass-count lever now priced to null
or negative — the four-way schedule race is closed, both prefetch placements are
closed, `bs` proved it is not L2 traffic, and the bodies are DSB-resident by
construction. The residual 0.10 µs is either the divider tail or the load-use
chain on the strided column loads in pass x, and **nobody on this panel has ever
opened `perf_event_open`, which works on this node.** Two counters —
`arith.divider_active` and `ld_blocks.*` — decide it. Building before that
measurement is guessing, and the last three rounds of guessing at L=8 have
returned ~1% each.

**L=17 — split the kernel chunk so the map can hide.** L17_matrixsimd measured
the thing that matters and wrote it down: **the map runs at its standalone rate
in-chain.** A 17-pair map burst is ~510 uops against a 352-entry ROB, so the
out-of-order window physically cannot interpenetrate map bursts with kernel
chunks — "the map hides under the FFT" was never true. The map is ~39 of ~103
kcyc per group-step at ~2× its port floor while j0 and j1 sit *at* theirs. Every
alternative map body has been measured (quartic, Newton, rcp14, vsqrtpd, mixed —
all priced). The only move left is **making the bursts ROB-small by splitting the
148-op kernel chunk**, and it is priceable with `objdump` before a single build.
With all three L=17 entries now bit-identical, whichever one does it wins the
cell and the other two inherit it mechanically.

**L=36 — explain the 29.8% spread, then get PMU attribution.** First, the
integrity item: L36_mixedradix's spread went 7.2% → 29.8% against its own
prediction of a collapse, and its 0.03 µs lead is inside that noise. Re-measure
before treating the r8 ordering as real. Then the technical move, which both
leaders independently name and neither can do without an instrument: **PMU
attribution first.** Both passes sit ~35% over their port floors (phase 2 ~465
cyc/call vs ~310; z-subloop ~300 vs 201) and every mechanism probe — placement,
staging, ratio, prefetch, custody, BCOR, TPP, cross-pass balancing — is measured
shut. There is a live, cheap, unverified hypothesis that explains both passes at
once: **the step body is ~3.2k instructions, well above DSB reach, so a MITE feed
ceiling may be the binding resource.** `UOPS_DISPATCHED.PORT_5` plus DSB coverage
settles it. Only if that comes back clean is split-complex lanes (which would kill
55.4k swaps/vol plus the map's 23.3k unpcks) worth the round.

**Round-wide, above all of these:** commit the `check.py` fix and repair
`driver.c:141`. A round whose primary correctness gate silently no-ops is worth
less than a round with no results at all, because it looks like results.

---

## 7. What to keep

Applying `docs/CURATION.md`.

**A note on criterion 4.** "Anything that beat a library baseline, regardless of
rank" would promote all 19 entries, because all 19 beat every library at their
geometry — by 2.71× to 9.74×. That criterion was written when beating a library
was the exception; in the ice series it has become the entry condition and the
rule is now vacuous. **It should be rewritten** — the useful successor is
something like "anything that beat the best library by more than the current
per-geometry median," or simply struck. I have applied criteria 1–3.

### Promoted, and why

**Criterion 1 — fastest correct entry per geometry (8, mandatory):**
`L6_unrolled`, `L8_radix8`, `L13_direct`, `L17_matrixsimd`, `L23_matrixsimd`,
`L36_mixedradix`, `L45_pfa`, `L64_blocked`.

**Criterion 2 — structurally different runner-up, close (4):**

* `L13_rader` (3.463 vs 3.438, +0.7%) and `L23_rader` (34.209 vs 33.805, +1.2%) —
  Rader against a dense conjugate-folded kernel, within ~1% at two different
  primes. This is CURATION's own worked example, verbatim, and it is the *live*
  form of the dense-vs-Rader question now that L=17 has collapsed to one engine.
* `L36_pfa` (100.247 vs 100.212, +0.03%) — a dead heat, and given §2's spread
  finding, arguably the better-measured of the two. Different phase construction
  and different map placement from mixedradix, and the pair carry a genuinely
  instructive asymmetry between them: the lagged map wins in a retire-shaped pass
  and loses in a load-shaped one, each side measured by the entry that lost it.
* `L64_radix8` (636.477 vs 634.801, +0.3%) — structurally distinct (radix-8²/axis
  split-complex versus z-split custody), and the pair are the round's headline
  correctness story: two independent teams told their fastest code was wrong, two
  independent exact-map fixes, two independent price tags (+3.2% and +4.4%) for
  the same guarantee. That agreement is worth more than either entry alone.

**Criterion 2 special case — `L6_pfa` (0.210 vs 0.207, +1.4%).** Not structurally
different — both L=6 entries ported the same warm store-fused SoA-8 engine. I am
promoting it on ranking-integrity grounds instead: the winner's number carries a
13.9% run spread and L6_pfa's carries 0.0%, so promoting only L6_unrolled would
record a decision the measurement does not support. It also keeps the Good–Thomas
36-op codelet where the winner carries warm's 40-op form.

### Not promoted, and why

* `L8_fusedaxes` (0.507) and `L8_batchsimd` (0.509) — near-duplicates of
  L8_radix8; all three run the same `bl8` batch-lane engine, and the winner
  already carries the two deltas that separate them (CP column relayout, join-FMA).
  fusedaxes' `-fschedule-insns`/`-fsched-pressure` attribute finding is genuinely
  transferable and survives in `strategies/L8_fusedaxes.md`, which is tracked
  regardless; the next panel should be pointed at it in `NOTES.md`.
* `L17_rader` (9.028) and `L17_winograd` (9.033) — by their own records these are
  now **bit-identical engines** to L17_matrixsimd, with chain outputs verified
  `cmp`-equal. Promoting them would put three copies of one engine on the reading
  list. L17_winograd's instructive content is real but historical — its r7 Halley
  map tier at ~2.3e-13/application against a 1e-14 contract is the cleanest
  example of a fast wrong answer this project has produced — and it lives fully in
  `strategies/L17_winograd.md`. Cite it in `NOTES.md`; do not duplicate the code.
* `L36_pencilfused` (102.427, +2.2%) — third variant of the same PFA 4×9; its
  custody/BCOR/TPP negatives are recorded and were correctly reused by the other
  two this round.
* `L45_mixedradix` (219.689, +1.3%) — same PFA 9×5 family as the winner, and
  converging on it (L45_pfa's −6.2% came from porting the custody schedule out of
  that family).

### Command

```bash
cd bench/ice
./promote.sh ice_r8 L6_unrolled L6_pfa L8_radix8 L13_direct L13_rader \
             L17_matrixsimd L23_matrixsimd L23_rader L36_mixedradix L36_pfa \
             L45_pfa L64_blocked L64_radix8
# then fill in exemplars/ice_r8/NOTES.md from §5 and §6 above
git add exemplars/ice_r8 strategies check.py \
        results/ice_r8/{VERDICT.md,leaderboard.txt,leaderboard_regated.txt,environment.txt,build_errors.txt}
```

Note that `check.py`'s `import math` fix is currently uncommitted in the working
tree and **must go in the same commit** — it is the reason this round's original
leaderboard understates what was verified.
