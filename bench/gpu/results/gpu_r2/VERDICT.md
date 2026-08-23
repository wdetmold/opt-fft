# VERDICT — round gpu_r2

Monitor report. Measured on **a80n1.lqcd.mit**, slurm job 438580, 2026-08-22T20:09 →
20:20, **one A100-SXM4-40GB** (`CUDA_VISIBLE_DEVICES=0`, pinned on purpose; the other
seven GPUs on the node held idle for the scoring window). nvcc 12.2.r12.2, driver
525.125.06, SM clock 1410 MHz sustained through the sweep. Library reference:
**cuFFT 11.0** (`cufftPlanMany`, Z2Z, batched). Library-free floor: `baseline_gpu`.

**A note on the brief, again.** The prompt describes a CPU round — "Xeon Gold 5218,
Cascade Lake, AVX-512", MKL as the library, a 2.9× machine spread between Sapphire
Rapids and Cascade Lake. That text is stale for this round exactly as it was for
gpu_r1. `environment.txt` records eight A100s, nvcc and a driver version; the library
is cuFFT; and every implementer developed on a leased A100-SXM4-40GB **of the same
node a80n1** via `tryout.sh`. There is no cross-machine correction to apply, and §4
below reports the claim-vs-measured gaps against their actual cause rather than an
attribution that would be false. The prompt's own §4 instruction ("attribute it to the
machine difference where that is the plausible cause") is honoured by finding that it
is not the plausible cause here, and naming what is.

Geometries measured: **L = 6, 8, 13, 17, 23, 36, 45, 64** (the prompt asks for
headlines at 6, 8, 17, 36; the other four ran and are reported too). Three cases per
geometry, per `cases.txt`: `B=1` (launch/latency), `B_L2` (in+out ≈ 32 MiB,
L2-resident), `B_HBM` (one buffer ≈ 1 GiB, **the primary score**). Batched and
non-batched are reported separately throughout, as the project's methodology requires.

---

## 1. Headline per geometry

Timing convention: minimum over 3 independent processes of each process's minimum
sample, with the across-process spread of those minima in parentheses. All twelve panel
entries are library-free (verified: no `cufft`/`vkfft`/`cufftdx`/`heffte`/`fftw`/`mkl_`/
`ducc` symbol anywhere in `impl_2/*.cu`; the only includes are `cuda_runtime.h`,
`cuda_pipeline*.h`, `cooperative_groups.h` and libc).

### L = 6 — volume 216

| case | fastest panel entry | cuFFT | speedup | floor (`baseline_gpu`) |
|---|---|---|---|---|
| **B = 1** (non-batched) | **L6_warpvolume 2.969 µs** (2.4%) | 10.259 µs | **3.46×** | 15.057 µs |
| **B = 4854** (L2, 32 MiB) | **L6_batchcoalesced 14.629 µs/call** = 3.01 ns/xform, 2779 GF/s (0.6%) | 51.192 µs | **3.50×** | 162.469 µs |
| **B = 310608** (HBM, 1 GiB, primary) | **L6_batchcoalesced 1540.181 µs/call** = 4.96 ns/xform, 1689 GF/s (0.1%) | 3520.614 µs | **2.29×** | — |

`L6_warpvolume` takes B=1 by 24% with a dedicated 64-thread single-volume kernel;
`L6_batchcoalesced` takes B_L2 by 2.3%. B_HBM is a **tie** — 1540.181 vs 1540.565, a
0.025% gap against 0.1% spreads on both.

### L = 8 — volume 512

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L8_blockfused 3.191 µs** (5.6%) | 9.352 µs | **2.93×** | 17.650 µs |
| **B = 2048** (L2) | **L8_blockfused 13.058 µs/call** = 6.38 ns/xform, 3614 GF/s (1.4%) | 54.526 µs | **4.18×** | 174.391 µs |
| **B = 131072** (HBM, primary) | **L8_blockfused 1537.237 µs/call** = 11.7 ns/xform (0.0%) | 3697.152 µs | **2.41×** | — |

`L8_blockfused` sweeps all three cells; the 4.18× at B_L2 is again the **largest margin
over cuFFT anywhere on the board**. `L8_warpradix8` is 13% / 7% / **0.14%** behind — the
primary cell is now a tie (1539.456 at 0.2% spread against 1537.237 at 0.0%), which is
the resolution of the register-vs-shared experiment this pair was built to run (§7).

### L = 17 — volume 4913

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L17_dmma 7.635 µs** (1.5%) | 13.518 µs | **1.77×** | 26.993 µs |
| **B = 213** (L2) | **L17_dmma 31.696 µs/call** = 148.8 ns/xform (0.2%) | 67.478 µs | **2.13×** | 309.504 µs |
| **B = 13660** (HBM, primary) | **L17_dmma 1571.669 µs/call** = 115.1 ns/xform (0.1%) | 4758.272 µs | **3.03×** | — |

`L17_dmma` sweeps all three cells, reversing r1 where it held only B_HBM.
`L17_raderfused` is 14% / 1.4% / 0.2% behind. The two entries have **converged onto one
algorithm** — see §3, where their scored outputs are bit-identical at B=213 and
B=13660 — which is the round's most consequential curation fact.

### L = 36 — volume 46656

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L36_globalpass 9.034 µs** (1.7%) | 12.914 µs | **1.43×** | 45.890 µs |
| **B = 22** (L2) | **L36_sharedtiled 30.433 µs/call** = 1.383 µs/xform (1.8%) | 51.077 µs | **1.68×** | 513.149 µs |
| **B = 1438** (HBM, primary) | **L36_globalpass 1945.088 µs/call** = 1.353 µs/xform (0.0%) | 3381.965 µs | **1.74×** | — |

The pair still splits the board, and now for genuinely different structural reasons:
`globalpass` takes B=1 (by 11%) and the primary point (by 7%) with a **persistent
producer/consumer ticket kernel**; `sharedtiled` takes the L2 point (by 21%) with
**one-slice-per-stream batch splitting**. Both mechanisms are new this round and both
are transferable — see §5 and §6.

### The rest of the board

| L | B=1 | B_L2 | B_HBM (primary) |
|---|---|---|---|
| **13** | L13_dmma **7.820** vs cuFFT 12.384 → **1.58×** | B=477: **21.407** vs 62.991 → **2.94×** | B=30549: **1562.837** vs 4743.552 → **3.04×** |
| **23** | L23_rader **9.088** vs 14.880 → **1.64×** | B=86: **38.556** vs 75.955 → **1.97×** | B=5515: **2257.152** (4.0%) vs 5055.829 → **2.24×** |
| **45** | L45_pfa **12.623** vs 18.085 → **1.43×** | B=11: **36.513** vs 64.102 → **1.76×** | B=736: **2436.754** vs 4748.416 → **1.95×** |
| **64** | L64_radix8 **18.510** vs 23.524 → **1.27×** | B=4: **41.663** vs 58.048 → **1.39×** | B=256: **2617.515** vs 3739.904 → **1.43×** |

**Board summary: 12 entries, 36 scored cells, cuFFT beaten in all 36.** Best margin
4.18× (L=8, B_L2); worst 1.27× (L=64, B=1), up from r1's worst of 1.04×.
`baseline_gpu` is beaten by 2.9–19.7× wherever it ran.

The structural picture from r1 survives and sharpens. **Where a whole volume fits one
block's shared memory (L ≤ 17), the fused one-read-one-write kernel is at the hardware
wall: 88–90% of the part's 1555 GB/s DRAM peak moving provably minimum bytes, and every
lever tried this round moved it ≤1%.** Where it does not fit (L = 23, 36, 45, 64), r1
reached only 1.25–1.58× and reported every entry latency-bound with the L2-chunking
machinery delivering *zero* wall-clock. **This round is where that half of the board
moved**: 1.43–2.24× at the primary point, gains of 12–31%, and the mechanism is a single
finding that four entries reached independently (§5).

---

## 2. What changed since gpu_r1, per geometry

The two rounds are directly comparable: **cuFFT is stable to ≤2.1% at every one of its
24 cells across the two sweeps** (e.g. L=8 B_HBM 3697.920 → 3697.152 µs; L=17 B_HBM
4759.680 → 4758.272; L=64 B_HBM 3740.160 → 3739.904), and `baseline_gpu` likewise. Same
node, same job id, same driver, same clocks. Round-over-round deltas below are therefore
real and not reference drift.

Best-cell movement, r1 → r2:

| L | B=1 | B_L2 | B_HBM (primary) |
|---|---|---|---|
| 6 | 3.640 → **2.969** (−18%) | 18.381 → **14.629** (−20%) | 1540.139 → 1540.181 (**flat**) |
| 8 | 3.184 → 3.191 (flat) | 13.238 → **13.058** (−1.4%) | 1536.768 → 1537.237 (**flat**) |
| 13 | 7.875 → 7.820 (flat) | 28.334 → **21.407** (−24%) | 1563.307 → 1562.837 (**flat**) |
| 17 | 9.760 → **7.635** (−22%) | 31.690 → 31.696 (flat) | 1584.725 → **1571.669** (−0.8%) |
| 23 | 10.556 → **9.088** (−14%) | 51.842 → **38.556** (−26%) | 3252.053 → **2257.152** (**−31%**) |
| 36 | 10.069 → **9.034** (−10%) | 38.390 → **30.433** (−21%) | 2150.172 → **1945.088** (−9.5%) |
| 45 | 14.340 → **12.623** (−12%) | 42.568 → **36.513** (−14%) | 3355.341 → **2436.754** (**−27%**) |
| 64 | 22.581 → **18.510** (−18%) | 48.441 → **41.663** (−14%) | 2988.237 → **2617.515** (−12%) |

Per geometry:

* **L = 6** — both cells that were open closed. B=1 fell 18% because `L6_warpvolume`
  replaced its all-register kernel with a 64-thread single-volume shared kernel
  (borrowed from `L8_blockfused`'s `fft8_single`); B_L2 fell 20% because
  `L6_batchcoalesced` adopted `L6_warpvolume`'s r1 `__stcs` evict-first store with a
  corrected predicate (its r1 gate was on *working set > L2*, which is backwards for
  stores; the right predicate is *input buffer small enough to stay resident*). B_HBM is
  bit-for-bit the same number. **The register-resident design was retired on
  measurement** (§3).
* **L = 8** — the geometry is finished and this round is what establishes that.
  `L8_blockfused` built all three of its own r1 next-steps plus two borrowed ideas and
  **every structural alternative lost**: warp-per-volume +7% at B_L2, persistent
  grid-stride +5.6% at B_HBM, warp-local staging −1.3%, `__stwt` instead of `__stcs`
  **+79%**, 32-thread single-warp B=1 +19%. Net production change: one barrier demoted
  from `__syncthreads` to `__syncwarp` (~1%). `L8_warpradix8` halved its per-lane
  payload (106 → 66 registers, one volume per two warps joined by a pair-local
  `bar.sync`) and closed B_HBM from −1.5% to a tie.
* **L = 13** — the r1 verdict named this "the clearest free win visible in the whole
  round": apply `L8_blockfused`'s evict-first store, which L=8 had measured at −43% and
  L=13 had never tried on the store side. It was applied, and it delivered
  **−24% at B_L2**. The L2 cell now beats the HBM cell per volume (45.2 vs 51.2 ns), as
  it always should have. Four further structural alternatives measured dead.
* **L = 17** — B=1 fell 22%, driven by `L17_dmma` replacing its cooperative launch with
  **one plain launch joined by a hand-rolled software grid barrier** (measured at B=1
  with identical bodies: cooperative 11.88 µs, two plain launches 10.50, one launch +
  soft barrier 7.74–8.92). B_HBM improved 0.8% via a plan-time switch between two
  staging forms, taking each regime's best in one binary. B_L2 is flat.
* **L = 23** — **the round's largest single gain: −31% at the primary point, 1.44×.**
  Three causes, in order: a two-chain ILP reorder of every line engine (~25% on its own),
  out-as-intermediate with kernel B in place (+7%), and a chunk size that became sharp
  once the kernels were fast enough for it to matter (r1's chunk table was flat;
  r2's is 32 → 2312 µs against 128 → 3277).
* **L = 36** — both entries improved and the winners swapped in both directions.
  `globalpass` built the persistent producer/consumer ticket kernel it had named as its
  own r1 next step, taking B_HBM −9.5% and B=1 −11%; `sharedtiled` found that its B_L2
  problem was **concurrency, not cache policy**, and took that cell −21% by splitting the
  batch one slice per stream.
* **L = 45** — **−27% at the primary point**, and the interesting part is that r1's
  conclusion was wrong in a documented way. r1 had measured L2 chunking cutting DRAM
  traffic exactly as designed and buying *zero* wall-clock, and concluded the kernels
  were latency-bound. r2 added evict-first cache hints and the same chunking became
  worth 26% (3509 → 2604 µs), with the chunk optimum going from flat to sharp. The
  kernels were not touched.
* **L = 64** — −12% at the primary point from rebuilding `kernel_x` as a fully
  shared-staged plane (L1TEX SOL 79% → 46%) and staging `kernel_yz`'s store the same
  way (59% → 43%), then re-scanning the chunk optimum, which moved C=4 → C=3.
  B=1 −18% via a captured CUDA graph.

### Did anything regress?

**No cell of the round's best regressed.** Three items belong on the record anyway:

1. **`L17_raderfused` at B=213 is genuinely 1.4% slower than its own r1 number** —
   32.139 vs 31.690 µs, against across-process spreads of 0.3% (r2) and 0.5% (r1), so
   this is outside noise. Its own record predicted the difficulty honestly ("B=213 could
   not be cleanly separated from r1 today — every window at that batch had 6–8% sd").
   The cell's *round best* is flat because `L17_dmma` arrived at 31.696. Cause is not
   diagnosed in the record; the candidates are the warp-chunked cp.async staging (A/B'd
   as neutral at that cell: 32.10 old vs 32.24 new) and the retuned path cut.
2. **`L6_batchcoalesced` at B=1 reads 3.693 vs r1's 3.640 µs** (+1.5%), inside its 3.8%
   spread. Not a regression; noted because B=1 cells on this GPU are the noisy ones and a
   future monitor should not read ±2% there as signal.
3. **Plan time regressed badly, and it was flagged last round.** See §3.

---

## 3. Adversarial pass: failures, wrong answers, absences

**No entry failed correctness. No entry failed to build. No entry crashed or hung. No
entry is missing.** What I checked, rather than assumed:

* `build_errors.txt` is **0 bytes**. `failures.txt` **does not exist**.
  `agents/exits.txt`: all 12 implementer agents `exit=0`. Zero matches for
  `fail|error|abort|crash|timeout` in `check.log`, `timing.log` or `sweep.out`.
* **Correctness coverage is exactly complete, and I verified the arithmetic rather than
  trusting the leaderboard's `ok` column.** 76 `c_*.json` files, **76 with
  `ok=true`** — 12 panel entries × 3 cases (36) + cuFFT × 8 L × 3 cases (24) +
  `baseline_gpu` × 8 L × 2 cases (16) = 76. Every scored cell has a check; no check is
  missing and none is orphaned. Worst residual anywhere is `baseline_gpu` at L=17
  (rel_l2 8.41e-16); worst panel entry is `L45_pfa` (8.21e-16, rel_max 1.07e-15).
  Tolerance is 1e-12. Every panel entry is 3–4 orders of magnitude inside it at every
  case.
* **`baseline_gpu`'s absence from all eight B_HBM rows is by design, not a crash.**
  `timing.log` carries exactly eight lines of the form `skipping baseline_gpu at L=6
  B=310608 (too expensive to be informative)`, one per geometry, and no others.
* **The 756 `does not support` lines in `timing.err` are honest `supports()` refusals** —
  each entry declining the seven geometries it did not write, with exit code 3. No entry
  claimed a geometry it had not implemented.
* **Library-free rule holds.** Grepped all thirteen `impl_2/*.cu`: no library symbol
  outside the include list above.

### The one thing that looked like a fast wrong answer, and what it actually is

**Four pairs of scored outputs are bit-identical between two supposedly independent
entries.** This is the check worth doing adversarially, because identical residuals to
sixteen significant figures is what a harness reusing a cached output file would look
like. It is not that. It is algorithmic convergence, and it is confirmed by the records:

| cell | entries | rel_l2 | max_abs |
|---|---|---|---|
| L=6 B=1 / 4854 / 310608 | `L6_warpvolume`, `L6_batchcoalesced` | identical at all three | identical at all three |
| L=17 B=213 | `L17_dmma`, `L17_raderfused` | both 3.157523415564384e-16 | both 1.3101760682293512e-13 |
| L=17 B=13660 | `L17_dmma`, `L17_raderfused` | both 3.159002147193252e-16 | both 1.797546735911271e-13 |

The evidence that this is real convergence and not a harness fault:

* **In r1 these same pairs had different residuals** (L6_warpvolume 2.2e-16 vs
  batchcoalesced 2.5e-16; L17_dmma 3.6e-16 vs raderfused 3.1e-16), so the harness
  demonstrably produces per-entry outputs.
* **The records state the borrowing explicitly.** `L6_warpvolume` r2: *"The r1
  warp-register kernel is gone from the hot path… structure borrowed outright from
  `L6_batchcoalesced` round gpu_r1"*, with the same DIT 2×3 Winograd codelet and the
  same store predicate. `L17_dmma` r2: *"The per-line module is now L17_raderfused's
  `line17w`, ported verbatim with attribution."*
* **The one cell where they differ is exactly the one the records say differs.** At L=17
  B=1 the two entries are *not* identical (3.239e-16 vs 3.092e-16) — and that is the
  only cell where they run different code, `L17_dmma`'s soft-grid-barrier split path
  against `L17_raderfused`'s two-launch split path. Same arithmetic in the main path,
  different special-case paths. Exactly consistent.

So no entry is returning a fast wrong answer, and no entry is returning a cached one.
But the consequence is not cosmetic and it drives §7: **at L=6 and L=17 the panel no
longer has two independent implementations at the batched cells. It has one program
scored twice.** Two entries agreeing on a number is normally corroboration; here it
carries no information.

### Six things I flag rather than accept

**(a) Plan time regressed 4.2× at L=23, and the r1 verdict asked for exactly this not
to happen.** r1 §3(a) recorded the panel running measured autotuners inside `create()`
with an effectively unbounded budget while cuFFT's planner uses 10–19 ms, and said: *"r2
should either cap plan time or publish a second column with setup amortized over the
batch, before the number grows again."* Neither was done, and it grew:

| entry | cell | r1 setup | r2 setup |
|---|---|---|---|
| **L23_rader** | B=5515 | 2.598 s | **10.971 s** |
| L8_blockfused | B=131072 | 0.304 s | 0.615 s |
| L36_sharedtiled | B=1438 | 0.423 s | 0.585 s |
| L36_globalpass | B=1438 | 0.000 s | 0.395 s |
| L8_warpradix8 | B=131072 | 0.398 s | 0.510 s |

cuFFT's setup at the L=23 primary cell is **0.014 s** — a ratio of 784×. To put the
headline number in application terms: `L23_rader` wins that cell by 2798.677 µs per
call, so it needs **≈3,900 calls (≈21.6 million transforms)** before its 10.971 s of
plan time is amortized and it is ahead of cuFFT end to end. Nothing here is cheating —
setup is excluded by the project's own methodology and cuFFT receives the same
exemption, and the tuners race bit-identical variants so the scored output cannot depend
on the pick. But the 2.24× headline at L=23 is now a claim about the steady state only,
and r3 must cap plan time or publish the amortized column. This is the second round of
asking.

**(b) The harness-shaped B=1 advantage has gone from one entry to five.** r1 flagged
`L36_sharedtiled`'s B=1 win as resting on a CUDA graph keyed on the `(in,out)` pointer
pair: the driver calls `execute` with the same two pointers forever, so capture cost
amortizes to zero and an application rotating buffers would pay a recapture per call.
In r2 that construction is used at B=1 by **`L36_sharedtiled`, `L36_globalpass`,
`L23_rader`, `L45_pfa` and `L64_radix8`** — every B=1 winner at L ≥ 23. All five records
are honest about it and all five recapture on pointer change, so nothing is incorrect.
But the B=1 column is now systematically measuring a call pattern the harness happens to
have, and it should be read that way. `L64_radix8`'s B=1 gain (22.5 → 18.5 µs) is
*entirely* this mechanism.

**(c) `L36_globalpass`'s persistent kernel is the most fragile construction on the
board, and its safety argument is unguarded.** The fused mode (which is what wins the
primary cell) is one launch of exactly 432 blocks pulling tickets from a global atomic
counter, with K2 blocks spin-polling per-volume done-counters. The record's
deadlock-freedom argument is *"the grid is exactly one resident wave and every K1 ticket
precedes its volume's K2 tickets in dispatch order"* — i.e. it depends on 4 blocks/SM ×
108 SMs actually co-residing. Unlike `L17_dmma`'s software grid barrier, which
**verifies co-residency in `create()` via an occupancy query and falls back to a
two-launch form if it fails**, `globalpass` hardcodes the assumption. It passed here
(memcheck clean on the fused path, PASS and bit-identical at B=1/2/13/22/100/432/1438),
and the epoch arithmetic is carefully documented. But the failure mode is a hang, not a
slowdown, and the driver's correctness gate scores a hang as a dead entry. r3 should add
the occupancy check and fallback; it is a few lines and `L17_dmma` has the pattern.

**(d) Every entry's name is now wrong, and two of them in a way that hides a retired
experiment.** `L13_dmma` and `L17_dmma` never built DMMA — both refuted it by
measurement, which is a legitimate and well-documented outcome, and `L13_dmma`'s record
says so cheerfully (*"the entry keeps its name as a monument to the analysis that killed
it"*). `L17_raderfused` is not Rader and never was. `L23_rader` is the conjugate-folded
dense form. `L6_warpvolume` has no warp-volume kernel in its batched path any more. The
harmful case is L=17: the entry named `dmma` was the panel's **dense-symmetric** arm in
r1, and it silently stopped being one in r2. A future panel reading the entry list will
believe the L=17 board holds a dense arm and a Rader arm. It holds neither — it holds
two copies of one folded-Winograd module.

**(e) `L45_pfa`'s absolute error is the loosest on the board and touches 1e-12.** At
B=736 its `max_abs` is 1.911e-12. The tolerance is applied to the relative metrics
(rel_l2 8.21e-16, rel_max 1.07e-15) and it passes those by three orders of magnitude,
so this is a pass, not a marginal pass. Named only because it is the one entry whose
absolute residual exceeds the numeric value of the tolerance constant, and a future
monitor skimming the JSON should not be startled by it. Root cause is benign: L=45 has
the largest volume with a genuine two-pass structure and the longest accumulation chains
(PFA 9×5 folded codelets), so it accumulates the most rounding.

**(f) `L23_rader` has the only elevated spread among the primary cells** (4.0% at
B=5515, against 0.0–0.7% everywhere else on the board). Its record attributes the range
to tuner-pick variation across `create()` runs (*"every pick in the range is one of 2–3
equivalent cells"*), which is consistent with the 10.971 s tuner in (a). The minimum is
what is scored and the win is 2.24×, far outside any spread, so the standing is safe.
But a 4% spread on a tuner-selected configuration is the signature of a tuner that is
not converging, and it should be read together with (a).

**No entry needs to be discarded for a fast wrong answer. There isn't one.** Two entries
need to be discarded as near-duplicates — a different question, settled in §7.

---

## 4. Claimed versus measured

The brief's premise — implementers on Sapphire Rapids, monitor on Cascade Lake, MKL
spanning 2.9× between them — **does not describe this round**, and I will not attribute
anything to it. All twelve implementers developed via `tryout.sh` on leased
A100-SXM4-40GB GPUs of the *same node* the scoring ran on: same part, same driver, same
CUDA, same 1410 MHz. The comparison is unusually tight, and this round it is tighter than
r1.

| entry | cell | claimed | scored | Δ |
|---|---|---|---|---|
| L6_batchcoalesced | B=310608 | 1540.4 µs | 1540.181 | −0.01% |
| L6_batchcoalesced | B=4854 | 14.8–15.5 | 14.629 | at/below low end |
| L6_batchcoalesced | B=1 | 3.78 | 3.693 | −2.3% |
| L6_warpvolume | B=310608 | 1540.4 | 1540.565 | +0.01% |
| L6_warpvolume | B=4854 | 15.08 | 14.960 | −0.8% |
| L6_warpvolume | B=1 | 3.09–3.16 | 2.969 | −4.0% |
| L8_blockfused | B=131072 | 1537.3 | 1537.237 | −0.00% |
| L8_blockfused | B=2048 | 13.13–13.24 | 13.058 | −0.6% |
| L8_blockfused | B=1 | 3.20–3.35 | 3.191 | −0.3% |
| L8_warpradix8 | B=131072 | 1537.1–1542.4 | 1539.456 | in range |
| L8_warpradix8 | B=2048 | 14.15–14.37 | 13.990 | −1.1% |
| L8_warpradix8 | B=1 | 3.55–3.74 | 3.595 | in range |
| L13_dmma | B=30549 | 1563.8 | 1562.837 | −0.06% |
| L13_dmma | B=477 | 21.56 | 21.407 | −0.7% |
| L13_dmma | B=1 | 7.86 | 7.820 | −0.5% |
| L17_dmma | B=13660 | 1571.9 | 1571.669 | −0.01% |
| L17_dmma | B=213 | 31.83 | 31.696 | −0.4% |
| L17_dmma | B=1 | 7.74 | 7.635 | −1.4% |
| L17_raderfused | B=13656 | 1574–1586 | 1574.955 | at low end |
| L17_raderfused | B=213 | 32.1–33.0 | 32.139 | at low end |
| L17_raderfused | B=1 | 8.81 | 8.687 | −1.4% |
| L23_rader | B=5515 | 2262–2286 | 2257.152 | −0.2% below low end |
| **L23_rader** | **B=1** | **10.4–11.0** | **9.088** | **−13% (faster than claimed)** |
| L36_globalpass | B=22 | 37.24 | 36.798 | −1.2% |
| L36_globalpass | B=1 | 9.21 | 9.034 | −1.9% |
| **L36_globalpass** | **B=1438** | **1879.3** | **1945.088** | **+3.5% (slower than claimed)** |
| **L36_sharedtiled** | **B=22** | **29.2** | **30.433** | **+4.2% (slower than claimed)** |
| L36_sharedtiled | B=1438 | 2111.5 | 2079.858 | −1.5% |
| L36_sharedtiled | B=1 | 9.99 | 10.001 | +0.1% |
| L45_pfa | B=736 | 2448.4 | 2436.754 | −0.5% |
| L45_pfa | B=11 | 35.7–37.9 | 36.513 | in range |
| L45_pfa | B=1 | 12.7–13.5 | 12.623 | −0.6% |
| L64_radix8 | B=256 | 2619 | 2617.515 | −0.06% |
| L64_radix8 | B=4 | 42.1 | 41.663 | −1.0% |
| L64_radix8 | B=1 | 18.5 | 18.510 | +0.05% |

**Thirty-four of thirty-seven quoted cells land within 2% of their own claim, and eleven
within 0.1%.** Nobody overclaimed materially. The three divergences and their causes:

* **`L36_globalpass` at B_HBM, +3.5% slower than claimed** — the largest gap in the
  "slower" direction, and the entry **pre-declared it**: its measurement table is
  prefaced *"lease-to-lease spread is ~3%, so same-lease A/B numbers are quoted for
  comparisons."* This is not a machine difference — it is the same part on the same node.
  It is clock-state and neighbour variation between one lease of a shared 8-GPU node and
  another. The 1879.3 µs was a warm-lease number; 1945.088 is the pinned scoring window.
  It does not change the standing: `globalpass` still wins the cell by 7%.
* **`L36_sharedtiled` at B_L2, +4.2% slower than claimed** — same cause, amplified by the
  cell. B=22 is a 31 MiB working set on a 22-volume batch, the smallest batched grid on
  the board; its scored across-process spread is 1.8%, the highest of any L2 cell. The
  claimed 29.2 µs was that entry's −29% headline; 30.433 is still a 21% win over
  `globalpass` and 1.68× over cuFFT.
* **`L23_rader` at B=1, −13% *faster* than claimed** — this is the **A100 boost-clock
  ramp**, the GPU analogue of the brief's machine caveat, and it cuts the opposite way to
  the CPU one. `driver.cu` documents the measurement that produced the rule: at L=8 B=64,
  cuFFT reads 22.4 µs at `--min-sample-ms 3`, 20.9 µs at 10 ms and 12.3 µs at 20 ms — a
  1.7× cliff, because below ~20 ms of continuous work the GPU never reaches 1410 MHz. A
  short dev-loop sample measures a downclocked GPU, so **claims made on short samples
  understate the hardware and the scored number is the faster one.** `L36_globalpass`'s
  record independently rediscovered this on the *tuning* side this round: with 3-execute
  samples its tuner once ranked a chunked config over the fused one that the driver then
  measured 10% faster, and the fix was sizing tuner reps to exceed 20 ms.

Two further observations on the records, for the corpus:

* **A pre-registered prediction was made and held.** `L64_radix8`'s r2 record says:
  *"B=256 sd is ~2% through tryout (boost clocks); the monitor's pinned-clock window
  should read ~2620 µs. If it reads >2700, suspect the C=3 tail chunk."* Scored:
  **2617.515 µs**, 0.1% from the prediction, spread 0.1%. That is the calibration loop
  closing, and it is the practice to spread: state what the scoring window should read
  and what it would mean if it does not.
* **The 1555 GB/s correction has fully propagated.** r1 recorded four entries
  independently deriving that the SXM4-40GB part is 1555 GB/s, not the brief's "~2.0
  TB/s" (which is the 80 GB part), and asked for `PANEL_BRIEF.md` to be fixed. Every
  roofline figure in this round's twelve records is against 1555 and is correct. Whether
  the brief itself was fixed I cannot tell from the results directory; if not, fix it,
  because `L23_rader`'s r1 record still contains one stale sentence reasoning against a
  2 TB/s floor.

---

## 5. Which LITERATURE §4 open question moved

### Moved most: §4.3 — "Is axis fusion worth 3× or 3%?", in the regime r1 left contradictory

§4.3's addendum names the untried structural move precisely:

> The untested case is **L2↔DRAM**, where the gap is 7× … **tile the batch so a tile
> fits L2, then run all three axes inside the tile.** That is not the same experiment
> the panel ran, and it is the largest untried structural move on the board.

r1 ran that experiment on the GPU and got a **contradiction** it recorded honestly. At
L ≤ 17, where a volume fits shared memory, fusion was worth 2.29–3.03× — the avoided
pass count times a 10–20× bandwidth gap, exactly Tolmachev's rule. But at L ≥ 23, where
the tile-the-batch construction is the *only* available form, three entries implemented
L2 chunking, **proved by ncu that it cut HBM traffic as designed, and measured zero
wall-clock gain.** `L45_pfa` r1 was blunt: *"wall-clock is currently flat vs unchunked
(3362 vs 3361 µs) because the kernels are latency-bound, not DRAM-bound."*

**gpu_r2 resolves the contradiction, and the answer is a precondition §4.3 does not
state.** The traffic cut was real; the wall-clock gain was gated behind telling the cache
which stream is disposable. Four entries measured it independently:

| entry | the measurement |
|---|---|
| `L45_pfa` | evict-first hints on the chunked path: **3509 → 2604 µs (−26%)**, and the chunk sweep went from **flat** (r1: 3360–3368 across chunks 6–24) to **sharp** (r2: ch=4 3603, ch=9 **2604**, ch=16 3395). Its own verdict: *"the wall-clock gain was always there, gated behind the hint pairing."* |
| `L64_radix8` (r1, the origin) | chunking **without** hints: 3318 vs 3335 µs — *"no gain at all."* With `__ldcs`/`__ldlu`/`__stcs`: 3004 µs. *"The hints are not a polish, they are the mechanism."* |
| `L36_globalpass` (r1) | removing the hints: **999.8 vs 665.6 µs** — *"worth 1.50× and the single most consequential three characters in the file."* |
| `L23_rader` | r1's chunk table flat; r2's decisive at ch=32 2312 vs ch=128 3277 vs unchunked 3271 µs, once the kernels were fast enough for traffic to bind. |

So **Tolmachev's rule survives with a precondition attached: the payoff is the number of
avoided passes times the bandwidth gap between the two levels — but only if the streaming
input and output are marked evict-first, because otherwise they evict the tile you built
the whole construction to keep resident, and the gap collapses to zero.** A traffic
measurement alone (ncu `dram__bytes`) will show the construction working while the clock
shows nothing, which is precisely the trap r1 fell into and documented.

Three corollaries this round establishes, each measured more than once:

1. **Evict-first is the right hint; write-through is catastrophic.** `L8_blockfused`
   A/B'd `__stwt` against `__stcs` at B_L2: **23.7 vs 13.24 µs, +79%**, because
   write-through defeats L2 write-coalescing and every 16 B store becomes its own DRAM
   transaction. The two are not interchangeable stream hints.
2. **The hint's sign flips with the regime and must be selected at plan time, never
   hardcoded.** `L6_batchcoalesced`'s r1 failure was a *predicate* error, not a
   mechanism error: it gated hints on *working set > L2*, which is backwards for stores.
   The correct predicate is *is the input small enough to stay resident* — protect
   residency where it exists, plain stores where it does not (`__stcs` at L=6 B_HBM
   measured 0.7% slower). Four entries now select it per batch.
3. **Explicit L2 persistence loses to evict-first hints, five times, at four
   geometries.** `L36_globalpass` `accessPolicyWindow` 705.8 vs 665.6; `L36_sharedtiled`
   `L2::evict_last` 2623 vs 2118; `L23_rader` window on the intermediate 3422 vs 3418
   (zero); `L6_warpvolume` four window configurations at B_L2, all losing to plain
   `__stcs` (23.4 / 19.9 / 15.18 vs 15.08); `L64_radix8` dropped it on the strength of
   the others without spending a run. `L6_warpvolume`'s diagnosis is the general one: at
   a 32 MiB working set the 40 MB L2 cannot be partitioned into 33.6 MB of protected
   residency, and `__stcs` already keeps the input resident **without reserving
   anything**. This door is closed; nobody should open it again.

### Also moved, and this is the sharper result: §4.6 — model versus search, and §4.2's sub-question (b)

§4.6 asks whether the instruction *schedule* needs searching, with §01 saying "you
should not need a search phase for L = 6, 8, 17, 36" and §06 replying that the schedule
is "the primary thing to search." r1 noted five entries shipped tuners but *"nobody
searched schedules — the searches were over block shape, chunk size, stream count and
cache hints."* This round two entries searched a schedule and both found it was the
lever, with the arithmetic held fixed:

* **`L17_dmma` A/B'd two emit schedules of the *identical algebra*.** Its own r1 nested
  cyclic-4⊕negacyclic-4 split held all 16 sine accumulators live across the m-loop and
  measured **zero** gain over dense (35.97 vs 35.78 µs), re-measured this round as a
  loss. `L17_raderfused`'s variant of the same algebra emits the sine half one output
  pair at a time, so only **2** accumulators are live — and that won: dense 34.54 →
  wline 32.61 µs at B=213, 1584 → 1571.8 at B_HBM. The record's conclusion is the
  quotable one: *"Fold-first + pair-at-a-time emit is the shape to copy, not the op
  count."*
* **`L23_rader`'s two-chain ILP reorder was worth ~25% at the primary point** — the
  round's biggest single win at that geometry — and it changes no arithmetic at all,
  only the accumulation order (two independent FMA chains summed at the end, 8 concurrent
  chains per k-pair instead of 4). Its mechanism is GPU-specific and worth carrying:
  **at 2.5–3 active warps per scheduler, per-thread ILP substitutes for the warps the
  register file refuses to give.**

This lands on **§4.2(b)** as well, and *revises* r1's answer there. r1 concluded the
symmetric/antisymmetric convolution split "adds arithmetic and adds no time" — 12% fewer
FP64 ops, 25% fewer FMAs, 0% time — reproducing the CPU r2 lesson. That stands. What r2
adds is that **the same algebra written with a different register schedule is worth ~5%
at the L2 cell and 0.8% at the primary cell.** So the corpus's framing needs one more
turn of the screw: at L=17 on a GPU the algebra is worth nothing, the op count is worth
nothing, and the *schedule* of the same algebra is worth single-digit percent. §06 wins
this exchange over §01.

§4.2's other sub-questions: **(a) is now closed in a way that removes an arm from the
board.** r1 answered it — the conjugate-folded dense form beat Rader-proper, and
Rader-proper lost on registers (a per-thread FFT16 needs 160+ live registers against the
102 that two-blocks-per-SM allows), not on flops. r2 goes further: the *dense* arm lost
too, to the folded-Winograd module, and `L17_dmma` retired it. The L=17 board now has no
dense-symmetric entry. **(c)**, the exact op count for a full 17-point Winograd module,
remains unanswerable and was correctly not pursued.

### Also moved: §4.1's GPU counterpart, with a genuine refinement

§4.1 asks how much spill traffic a batch-vectorised codelet generates and leans toward
taking the spills. r1 produced five GPU data points, all "never take the spills." r2 adds
three more in the same direction (`L8_warpradix8` var 6 forced to 64 regs: 14.84 vs 14.37
at B_L2, 1548 vs 1537 at B_HBM; `L23_rader` SPLITSQ forced to 64 regs: 2663 vs 2312;
`L36_globalpass` rejecting 5 blocks/SM by arithmetic before spending a run). Eight data
points, one direction.

The **new** contribution is a refinement that matters more than another confirmation:
**a high register count is not slack, and you cannot buy registers back by splitting the
work.** `L23_rader` built the split-k kernel its r1 record had named as the #1 escape
from the 512-threads/SM ceiling — two threads per line, each folding half the j's, hoping
for ~76 registers. **ptxas gave it 114**, because the compiler correctly spends registers
batching all twelve line loads for memory-level parallelism. Occupancy did not move and
the shuffle overhead was pure cost: 2354 vs 2312 µs. Its verdict: *"at 128 regs the
coarse engine's registers are all doing MLP work, not waste."*

And the boundary of r1's positive result is now drawn. r1's `L45_pfa` lesson was that the
way to buy occupancy is to make the *unit of work* smaller (thread-per-line → unit-parallel
took it from 204 registers and 4712 µs to 96 registers and 3752 µs). `L23_rader` tested
that transfer directly: its medium-grain unit-parallel kernels hit **32 registers as
designed, ~1440 theoretical threads/SM, at only 1.08× the flops — and lost anyway**
(kernAm 3355, kernBm 2686 vs coarse 2303), because the folded-dense form reads every
input 11 times and unit-parallel turns each re-read into an LDS until the LSU becomes the
wall. **The unit-parallel lesson transfers to low-reuse codelet structures (PFA
radix-5/9) and not to a dense matvec with high per-element reuse.** Two entries, opposite
conclusions, both measured, boundary established. That is the kind of result that saves a
future round.

### Not moved

* **§4.4 (split vs interleaved complex)** — untouched, and structurally untouchable here:
  the driver's contract fixes the global layout to interleaved `double2`.
* **§4.5 (padding at L=8)** — still not A/B'd, and it is now the oldest unclaimed cheap
  measurement on the board. r1 noted the mechanism was confirmed but the *unpadded* L=8
  kernel was never measured. `L8_blockfused` r2 names the XOR-swizzled unpadded layout as
  its one unexplored lever and did not build it. Second round of not doing a one-line A/B.
* **§4.7 (vector-radix)** — untried, correctly, at every geometry.
* **§4.8 item 1** — not closed (this is a GPU round; the item is about single-threaded
  CPU), but the round adds a second, directly comparable, 24-cell sweep at eight edge
  lengths across three working-set regimes against a library reference.

---

## 6. The single highest-value thing the next round should attack

**One item dominates the board and it is the same item at three geometries.**
`L36_globalpass` built a **persistent producer/consumer kernel** — one launch, grid = one
resident wave, blocks pulling tickets from a global atomic counter, K2 consuming each
volume's intermediate moments after K1 produced it — and it took the L=36 primary cell
−9.5%, with ncu measuring DRAM at the compulsory floor in both directions and *the
intermediate never touching HBM at all*. Independently, and without having seen it,
**`L36_sharedtiled`, `L45_pfa` and `L64_radix8` all name that exact construction as their
own top remaining item**, each after exhausting the hint and chunk-shape space:

> *"the only untried lever with real headroom is fusing k2(chunk c) and k1(chunk c+1)
> into one launch (or a persistent producer/consumer kernel) so the intermediate never
> faces a kernel boundary. Hint- and shape-space is exhausted — three independent sweeps
> now agree on the ~2105 µs plateau."* — `L36_sharedtiled`

One pattern, one working exemplar, three geometries queued behind it, and all three sit
at 1.4–1.8× over their two-pass traffic floors. **Put `L36_globalpass`'s ticket/epoch
machinery in the brief and have L=45 and L=64 adopt it.** That is the round's highest
-leverage transfer. (With the hardening from §3(c): add the co-residency check and
two-launch fallback that `L17_dmma`'s soft barrier has and `globalpass` lacks.)

Per geometry, in priority order:

* **L = 6 — nothing technical; retire the geometry to one entry.** Both entries agree
  B_HBM at 1540 µs is *the hardware answer* (89–90% of 1555 GB/s at exact-minimum bytes;
  `L6_warpvolume`: *"every entry lands there. I re-confirm rival's r2 conclusion"*), B=1
  is ~90% launch path, and the B_L2 residual to the 10.8 µs writeback floor has one
  remaining idea (named-barrier groups) that both records self-assess as *"expected
  small."* The pair has converged to one design (§3). The highest-value action here is
  curation, not optimization: drop to one entry, absorb `fft6_single` into it (a strictly
  additive plan-time branch worth 24% at B=1), and spend the freed slot at L=13 or L=23.
* **L = 8 — the XOR-swizzled unpadded shared layout at B_L2, and then close the
  geometry.** B_HBM is at 90% of DRAM peak at minimum bytes and the last untried
  structural lever (persistent blocks) measured **+5.6%**. B_L2 sits at 13.06 µs against
  an ~11.6 µs write-stream floor, and this round's value is that the residual is now
  measured *not* to be barriers (warp-per-volume +7%), *not* wave quantisation
  (grid-stride +5%), *not* store semantics (`__stwt` +79%), *not* staging shape
  (warp-local −1.3%). That elimination is what makes the one remaining candidate worth a
  build: 8192 B/volume instead of 9216 raises the shared-limited occupancy ceiling ~12%.
  Ceiling is ~13%; expect less. **This also closes §4.5** — measure the unpadded kernel
  while you are in there, which is the one-line A/B that has now been deferred twice.
* **L = 13 — B_L2, which holds the largest relative headroom on the L ≤ 17 half of the
  board: 21.4 µs against an ~11.6 µs write floor, 1.85×.** r2 took it from 28.3 with
  `__stcs`; the residual is a latency/barrier mixture at a *structural* 24-warps/SM
  ceiling (4 blocks × 35 KB shared is the cap — a 5th block needs 176 KB against a
  164 KB carveout). The named lever is plane-granular cp.async/mbarrier pipelining, and
  the negative evidence to read first is `L17_dmma`'s r2 PIPE experiment at G=2,3,4 (all
  worse) and `L17_raderfused`'s r1 whole-volume form (2820 µs spilling / 1735 µs at one
  block/SM). Honest expected value: modest. **The higher-value item may be that L=13 has
  a single entry and no structural alternative has ever been written there** — no
  register-resident counterpart of the kind `L8_warpradix8` is to `L8_blockfused`.
* **L = 17 — B=213, and run ncu before writing anything.** 31.7 µs = 148.8 ns/xform
  against a ~60–70 ns issue-floor estimate: **~2×, the largest headroom on the L ≤ 17
  half.** B=1 fell 22% this round and B_HBM is closed (88% of peak, minimum bytes, and
  *both* overlap ideas now measured negative from both directions — `L17_raderfused`'s
  persistent-block prefetch in r1 and `L17_dmma`'s plane-granular PIPE at three G values
  in r2). The cause is structural and agreed: a resident 78.6 KiB volume forces 2
  blocks/SM = 20 warps = ~30% occupancy. `L17_dmma` names the one untried shape — **2
  volumes per 640-thread block** (107 blocks = one clean wave, halves barriers per
  volume, 96 regs × 640 = 61k ≤ 64k so it fits) — and also says to do an ncu stall-
  structure pass first. Do the ncu pass first. Three rounds of guessing at this cell have
  produced two negative results and one flat one.
* **L = 23 — DMMA, and it is now the best-motivated DMMA case anywhere on the board.**
  The wall is concurrency at 128 registers/thread (512 threads/SM), and r2 measured
  **both** in-kind escapes dead: split-k got 114 registers instead of 76 and lost 2%;
  unit-parallel got its 32 registers as designed and lost 15–46% to the LSU. Meanwhile
  the traffic side is already ideal — kernel B reads **14 KB** from DRAM per chunk at a
  96% L2 hit rate. So there is ~1.9× to the traffic floor and it is all locked behind the
  register ceiling, and `mma` is the one construction that feeds operands from shared
  through the tensor pipe *without* per-thread fold registers. It attacks exactly this
  wall, and L=23 pads best of any geometry (23→24, 1.09×). Do not re-litigate the flop
  argument — it is irrelevant and both L=13 and L=17 have already refuted DMMA *on flop
  grounds* at their sizes. Budget a full round. **And cap the tuner** (§3a).
* **L = 36 — cross the two entries' wins, then decide whether to keep both.**
  `globalpass` owns B=1 and B_HBM with the persistent kernel; `sharedtiled` owns B_L2 by
  21% with one-slice-per-stream concurrency, having proved along the way that its own
  cache-policy hypothesis was wrong (*"The 24% was in concurrency, not in cache
  policy"*). Neither mechanism is in the other's file. Put the batch-split shape into the
  persistent kernel's small-B path and the persistent kernel into `sharedtiled`'s B_HBM
  path; `globalpass`'s own next step (two specialized persistent kernels to reach 5
  blocks/SM at 60 regs without spills, or half-plane tickets for 8+ blocks/SM) is the
  named mechanism for the remaining 1.307 → ~0.96 µs/vol.
* **L = 45 — re-profile with ncu *before* optimizing, then adopt the fusion.** The entry
  says this itself and is right: the whole r1 stall picture (latency-bound, 26–30%
  occupancy, nothing saturated) predates the hints and the round-robin shape, and the
  across-process spread collapsed from 8.9% to 0.28%, which suggests the binding
  constraint moved to DRAM. 3.33 µs/vol against a ~1.9 µs/vol two-pass floor is 1.75× —
  the same plateau both L=36 entries hit and one of them escaped this round by fusion.
  **Do not sweep hints or chunk sizes again**; three independent sweeps across two
  geometries agree that space is exhausted.
* **L = 64 — the same fusion, and it is the weakest margin on the board.** 1.27× at B=1,
  1.43× at B_HBM, with ~1.6× self-reported headroom (composite floor ~1.4–1.6 ms against
  2.62 ms). r2 already fixed the L1TEX problem (both kernels 79%/59% → 46%/43% SOL) and
  re-scanned the chunk optimum; what remains is that both kernels are pinned at 50%
  theoretical occupancy by 64 regs/thread and still latency-bound (No Eligible 78–82%,
  every SOL ≤ 51%). The radix-4³ alternative is self-assessed as *"the win is not
  obvious"* because the 66.5 KB shared plane caps blocks anyway. So: producer/consumer
  fusion of `kernel_x`(chunk c) with `kernel_yz`(chunk c+1), third in the queue behind
  the L=36 exemplar.

**Two cross-cutting items for the brief:**

1. **The fusion pattern above**, with `L36_globalpass`'s ticket machinery as the reference
   implementation and `L17_dmma`'s co-residency-check-with-fallback as the required
   safety pattern.
2. **Cap plan time, or publish setup amortized over the batch.** This is the second round
   of asking (§3a) and the number grew 4.2×. A 10.971 s tuner needs ~3,900 calls to
   amortize against cuFFT's 0.014 s planner. Either cap it or add the column; the
   headline numbers are becoming steady-state-only claims without it.

---

## 7. What to keep

Applying `docs/CURATION.md`'s four grounds, in order. The decisive new fact this round is
that **two of the four entry pairs converged onto a single algorithm**, which is
CURATION.md's explicit exclusion ("Do not promote near-duplicates of an already-promoted
entry"), and the evidence for it is not a judgement call — it is bit-identical scored
output (§3).

**Ground 1 — the fastest correct entry per geometry, always. Eight entries, no
discretion:**
`L6_batchcoalesced` (wins B_L2, first at a tied B_HBM), `L8_blockfused` (all three
cells), `L13_dmma`, `L17_dmma` (all three cells), `L23_rader`, `L36_globalpass` (B=1 and
the primary point), `L45_pfa`, `L64_radix8`.

**Ground 2 — a structurally different runner-up when it is close. Two of the four
runners-up qualify; two no longer do.**

* **`L36_sharedtiled` — keep, clearly.** It **wins B_L2 outright by 21%**, and its
  structure is genuinely different from the promoted winner's: two kernels split
  one-slice-per-stream, against `globalpass`'s single persistent ticket grid. Its
  concurrency finding — *"only ONE-chunk-per-stream shapes win; any shape putting ≥2
  chunks on a stream is worse than unchunked; tune the stream count, not the chunk
  size"* — was borrowed and re-confirmed by **both** `L45_pfa` (−18% at their L2 point)
  and `L64_radix8` this round. It also holds the measured refutation of its own
  cache-residency hypothesis, which is what stops r3 retrying it.
* **`L8_warpradix8` — keep, on a tie at the primary cell.** It now sits **0.14% behind at
  B_HBM** (1539.456 at 0.2% spread against 1537.237 at 0.0%), far inside CURATION.md's
  ~20% band, and it is the structural opposite of the winner: one volume per two warps,
  registers and shuffles, a single cross-warp stage through a pair-local `bar.sync`,
  **zero `__syncthreads` in the kernel**. It executed the experiment the pair exists to
  run — halving the per-lane payload from 106 to 66 registers to test whether the
  register structure's occupancy deficit was the whole gap — and the answer is yes at
  B_HBM (gap closed to a tie) and no at B_L2 (0.9 µs of cross-lane shuffle machinery it
  cannot shed). That is a settled question with a number, on both sides. **Note for r3:
  this is its last round on ground 2.** r1's stated retirement test was "loses all three
  cells and gains nothing at B_L2," and it does lose all three and did not move B_L2
  (14.024 → 13.990). It survives only because it converted its diagnosis into a tie at
  the primary cell. If r3 produces no new idea here, carry one L=8 entry.
* **`L17_raderfused` — do not promote.** It no longer satisfies "structurally
  different." `L17_dmma` ported its `line17w` line module **verbatim** (its own record
  says so), and the two entries' scored outputs are **bit-identical at B=213 and at
  B=13660** — at the L2 cell and the primary cell they are the same program. What remains
  distinct is a staging variant (warp-chunked cp.async fused into z) and a split-path
  launch mechanism, and the promoted winner supersedes both: its plan-time staging switch
  takes the best of *both* staging forms in one binary (regstage at B=213, cp.async at
  B_HBM), and its soft-barrier split path beats the two-launch form by 12% at B=1.
  Promoting it would put a second copy of one design on the reading list. The knowledge
  is not lost: `strategies/L17_raderfused.md` is tracked unconditionally and holds the
  derivation, the register-counting refutation of Rader-proper, and five negative results;
  `exemplars/gpu_r1/L17_raderfused.*` holds its r1 code; `impl_2/` holds this round's.
  **The honest way to restore an L=17 pair in r3 is to rebuild a genuinely different arm
  — the DMMA arm the entry name still promises, or the dense-symmetric arm that was
  retired this round — not to keep two copies of `line17w`.**
* **`L6_warpvolume` — do not promote.** The clearest near-duplicate on the board: its
  record states the batched kernel was *"borrowed outright"* from `L6_batchcoalesced`
  round gpu_r1, with the same codelet and the same store predicate, and the scored
  outputs are bit-identical at **all three** batch points. Its all-register warp-volume
  design — the thing that made it a structural alternative — **it retired itself, on its
  own measurement** (r1 kernel 1549.8 vs bstage 1540.8 in a same-session A/B; the triad
  decomposition 23.7 vs 18.6 µs at B_L2 at 62% SM throughput). Its one distinct win is
  `fft6_single`, a 64-thread single-volume kernel borrowed from `L8_blockfused`, worth
  24% at B=1 — and that is not a structural alternative, it is a 60-line plan-time branch
  that belongs *inside* the winner (§6). I considered ground 3 for it, since its record
  is an excellent instructive failure with five numbers attached; but ground 3's purpose
  is to stop rediscovery, and that function is served entirely by the tracked strategy
  record, whereas promoting the *code* would ship a near-copy of the winner **with the
  instructive dead code already deleted from it**. The r1 exemplar preserves the register
  design as actual source.

**Ground 3 — instructive failures.** None to promote as code, and this is the healthy
outcome rather than an absence. No entry lost: all twelve beat cuFFT at all 36 cells and
all passed correctness. This round's instructive failures **ship inside the promoted
winners**, executable, behind env knobs with the killing numbers in the records:
`L13_FORCE_{LOADZ,V2,SPLIT,FUSEDX}`, `L8_FORCE_{WPV,GS,SINGLEWARP}`, `L23R_FORCE` plus
the split-k and unit-parallel kernels kept in the tuner race, `L17_PIPE`/`L17_NESTED`,
`L17RF_{COOP,XDENSE,STCS}`, `FFT45_D1/D2`, `-DNOHINTS`. That is strictly better than
promoting a separate loser: the next panel can re-run any dead end in one lease command
against the current winner. `promote.sh` carries the records alongside the code, which is
what makes it work.

**Ground 4 — anything that beat a library baseline.** Admits all twelve and therefore
filters nothing. Stated plainly rather than pretending it selected anything.

**So: promote ten of twelve** — eight by ground 1, two by ground 2, none by grounds 3 or
4 that were not already in. Two entries are dropped as near-duplicates on bit-identical
evidence, which is the first curation this panel has had grounds to perform: r1 promoted
all twelve because its four pairs "differ in structure, not in tuning." Two of those four
pairs no longer do.

For the next monitor: **the retirement watch list for r3** is `L8_warpradix8` (see above)
and, if L=36's two mechanisms get crossed into one file as §6 recommends, whichever L=36
entry ends up holding neither win. `L13_dmma`, `L23_rader`, `L45_pfa` and `L64_radix8`
are sole entries at their geometries and have no pair to lose.

Promotion command:

```bash
cd bench/gpu
./promote.sh gpu_r2 L6_batchcoalesced L8_blockfused L8_warpradix8 L13_dmma \
    L17_dmma L23_rader L36_globalpass L36_sharedtiled L45_pfa L64_radix8
git add exemplars/gpu_r2 strategies results/gpu_r2 impl_2 && git commit
```
