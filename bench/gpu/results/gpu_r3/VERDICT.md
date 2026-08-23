# Round gpu_r3 — monitor's verdict

Measured on `a80n1.lqcd.mit`, slurm job 438580, 2026-08-22T21:23 — **one NVIDIA
A100-SXM4-40GB** (device 0 of 8, pinned on purpose), nvcc 12.2, driver 525.125.06,
SM clock 1410 MHz / memory 1215 MHz, clocks unlocked (the driver refuses `-lgc` for
this user, so the harness defends against the boost ramp with 20 ms minimum samples).
Library baseline: **cuFFT 11.0**, `cufftPlanMany` Z2Z. Harness floor: `baseline_gpu`
(row-column dense, no factorization).

> **A note on my own brief.** The monitor prompt for this round describes a CPU pass
> (Xeon Gold 5218, Cascade Lake, downclocked AVX-512, 1 MB L2) and asks me to attribute
> claim/measurement gaps to a Sapphire-Rapids-vs-Cascade-Lake machine difference with
> MKL spanning 2.9×. That text is inherited from the CPU phases and does not describe
> `gpu_r3`: the scoring pass ran on an A100 against cuFFT, and the implementers
> developed on **leased slices of the same A100 model on the same node** via
> `tryout.sh`. There is no cross-machine correction to apply here. §4 below answers the
> question that actually exists in this round — claimed-vs-measured under different
> *lease clock and contention* state — and one case where the gap is not a clock effect
> at all.

Nine of twelve entries were revised this round; three implementer agents died of tooling
faults (§3). **All twelve entries pass correctness at all three of their batch points**
(rel L2 1.6e-16 … 8.2e-16 against a 1e-12 gate), and all twelve beat cuFFT at all 36
cells. Nothing was disqualified for a wrong answer. One entry's *headline* is
disqualified as a measurement — see §3.1, which is the substance of this round.

---

## 1. Headlines per geometry

Three cells per geometry: **non-batched** (B=1), **batched-L2** (working set ≈ 32 MiB,
in+out resident in the 40 MB L2), **batched-HBM** (working set ≈ 2 GiB, DRAM-bound).
"×" is against cuFFT in the same cell. Ruled numbers (§3.1) are marked ⚠.

### L = 6 (216 points)

| cell | fastest correct panel entry | cuFFT | speedup |
|---|---|---|---|
| non-batched | **L6_batchcoalesced 2.832 µs** (2.96 GF/s) | 10.317 µs | **3.64×** |
| batched B=4854 (L2) | **L6_batchcoalesced 14.599 µs/call**, 3.01 ns/transform, 2784.8 GF/s | 51.632 µs | **3.54×** |
| batched B=310608 (HBM) | **L6_warpvolume 1538.475 µs/call**, 4.95 ns/transform, 1690.9 GF/s | 3519.181 µs | **2.29×** |

L6_batchcoalesced is 1540.395 µs at the HBM point — 0.12% behind, a dead tie at the
DRAM wall (both ≈ 1394 GB/s = 90% of this part's 1555 GB/s peak).

### L = 8 (512 points)

| cell | fastest correct panel entry | cuFFT | speedup |
|---|---|---|---|
| non-batched | **L8_warpradix8 3.055 µs** (7.54 GF/s) | 9.075 µs | **2.97×** |
| batched B=2048 (L2) | **L8_blockfused 13.165 µs/call**, 6.43 ns/transform, 3584.1 GF/s | 54.568 µs | **4.14×** |
| batched B=131072 (HBM) | **L8_blockfused 1537.195 µs/call**, 11.73 ns/transform, 1964.6 GF/s | 3696.640 µs | **2.40×** |

Both batched cells are ties, not wins: L8_warpradix8 is 13.176 µs (+0.08%) and
1538.645 µs (+0.09%). The two entries are structurally opposite (block-in-shared vs
volume-in-registers-per-warp) and land within measurement noise of each other at both
batched points. 4.14× is the largest margin over cuFFT anywhere on the board.

### L = 17 (4913 points) — ⚠ the contested geometry

| cell | fastest correct panel entry | cuFFT | speedup |
|---|---|---|---|
| non-batched | ⚠ **L17_raderfused 7.781 µs** (38.7 GF/s) — ruled; L17_dmma's measured 1.743 µs is void as a single-transform number | 13.464 µs | **1.73×** |
| batched B=213 (L2) | ⚠ **L17_raderfused 31.185 µs/call**, 146.4 ns/transform, 2057.5 GF/s — ruled; L17_dmma's measured 19.738 µs is void | 67.354 µs | **2.16×** |
| batched B=13660 (HBM) | **L17_dmma 1551.872 µs/call**, 113.6 ns/transform, 2651.5 GF/s | 4759.424 µs | **3.07×** |

The HBM cell stands as measured: the ruled-out mechanism is worth only 1.3% there
(L17_dmma ring-free measures 1571.7 µs in its own record, still 0.4% ahead of
L17_raderfused's 1577.899 µs), so the ordering does not depend on the ruling. See §3.1.

### L = 36 (46656 points)

| cell | fastest correct panel entry | cuFFT | speedup |
|---|---|---|---|
| non-batched | **L36_sharedtiled 9.062 µs** (399.3 GF/s) | 12.908 µs | **1.42×** |
| batched B=22 (L2) | **L36_globalpass 28.711 µs/call**, 1.305 µs/transform, 2772.5 GF/s | 50.984 µs | **1.78×** |
| batched B=1438 (HBM) | **L36_sharedtiled 1931.520 µs/call**, 1.343 µs/transform, 2693.7 GF/s | 3397.530 µs | **1.76×** |

All three cells are effective ties between the two L=36 entries (9.062 vs 9.129;
28.711 vs 28.738 — 0.09%; 1931.5 vs 1942.1 — 0.55%). They are no longer structurally
distinct: L36_sharedtiled ported L36_globalpass's persistent ticket kernel wholesale
this round and says so.

### The other four geometries (measured, not requested in §1, recorded for completeness)

| geometry | non-batched | batched-L2 | batched-HBM |
|---|---|---|---|
| L=13 | L13_dmma 6.819 µs vs 12.091 → **1.77×** | 21.511 µs (45.1 ns/t) vs 62.812 → **2.92×** | 1563.051 µs (51.2 ns/t) vs 4740.352 → **3.03×** |
| L=23 | L23_rader 9.101 µs vs 14.673 → **1.61×** | 39.033 µs vs 75.783 → **1.94×** | 2275.072 µs vs 5059.413 → **2.22×** |
| L=45 | L45_pfa 12.626 µs vs 18.013 → **1.43×** | 35.922 µs vs 64.472 → **1.79×** | 2110.272 µs vs 4740.864 → **2.25×** |
| L=64 | L64_radix8 18.501 µs vs 23.444 → **1.27×** | 39.616 µs (B=4) vs 58.283 → **1.47×** | 2614.357 µs (B=256) vs 3740.416 → **1.43×** |

---

## 2. What changed since gpu_r2, and did anything regress

Sources: 9 of 12 files changed (1514 insertions / 383 deletions against the r2
sources); `L23_rader.cu`, `L8_blockfused.cu` and `baseline_gpu.cu` are byte-identical
to r2, so their r3 numbers are a re-measurement of r2 code and function as this round's
noise ruler.

**Three cells changed hands, and the two entries that lost them lost by ~1%:**

| geometry / cell | gpu_r2 best | gpu_r3 best | Δ on the cell | who moved |
|---|---|---|---|---|
| L=6 non-batched | L6_warpvolume 2.969 | **L6_batchcoalesced 2.832** | −4.6% | batchcoalesced −23.3% on its own r2 number (new dedicated 36-thread B=1 kernel with the z-pass fused into the global load) |
| L=8 non-batched | L8_blockfused 3.191 | **L8_warpradix8 3.055** | −4.3% | warpradix8 −15.0% (new 4-warps-per-volume `fft8_quad` + CUDA-graph replay on every path) |
| L=36 non-batched | L36_globalpass 9.034 | **L36_sharedtiled 9.062** | +0.3% (flat) | sharedtiled −9.4% on its own r2 number (line-engine micro-passes) |
| L=13 non-batched | 7.820 | **6.819** | −12.8% | soft-barrier single-launch plane split at B≤8, adopted from L17_dmma r2 |
| L=17 non-batched (ruled basis) | 8.687 (raderfused) | **7.781** | −10.4% | raderfused adopted L17_dmma's soft grid barrier + folded thread-per-output x; hot spin instead of `__nanosleep(20)` |
| L=36 B=22 | 30.433 | **28.711** | −5.7% | globalpass −22% on its own number (4-stream chunked config) |
| L=36 B=1438 | 1945.088 | **1931.520** | −0.7% | sharedtiled −7.1% on its own number (ticket kernel port + the carveout 100→50 fix) |
| L=45 B=736 | 2436.754 | **2110.272** | **−13.4%** | the round's largest genuine gain — persistent producer/consumer ticket kernel + direct-output forms |
| L=64 B=4 | 41.663 | **39.616** | −4.9% | tail-chunk balance in the chunked path |

**Nothing regressed from a code change.** Every cell that got slower did so by
0.1–3.2%, in every case on a file whose relevant path was unchanged, i.e. within the
round-to-round clock/boost band that the three byte-identical entries independently
calibrate:

* byte-identical rulers: L8_blockfused non-batched 3.191 → 3.245 (+1.7%), B=2048
  13.058 → 13.165 (+0.8%), B=131072 flat; L23_rader 9.088 → 9.101 (+0.1%),
  38.556 → 39.033 (+1.2%), 2257.152 → 2275.072 (+0.8%).
* L6_warpvolume non-batched 2.969 → 3.064 (+3.2%) — kernel bit-identical to r2 by its
  own record (six variants built, all lost, r2 config kept). It lost the cell to a
  rival that improved, not by getting slower.
* L36_globalpass non-batched 9.034 → 9.129 (+1.1%) — same story.
* L13_dmma B=477 21.407 → 21.511 (+0.5%); L17_raderfused HBM 1574.955 → 1577.899
  (+0.2%); L64_radix8 B=256 2617.5 → 2614.4 (flat, and its record predicted exactly
  this: "if the monitor's number moves from 2617, suspect nothing").

**The stability outlier is L23_rader**, which needs naming even though it did not
regress: 4.3% and 5.5% run spread at its two batched cells (every other entry is
≤3.3%), 11.0 s of create()-time autotuning at B=5515, and a create()-time pick that
came out completely different from r2's on byte-identical source (r2: `A(P=4,T=96) …
chunk=48/4 g0`, "0.60 µs/xform in-plan, nv=86"; r3: `Af(P=1,T=32) … chunk=1/4 g1`,
"16.38 µs/xform in-plan, nv=1"). The tuner is sampling inside the boost-ramp zone that
L36_globalpass's r2 record and L6_warpvolume's r3 record both warn about. It still
beats cuFFT 1.6–2.2×, so this is a robustness note, not a demotion.

---

## 3. Adversarial section: failures, ruled-out numbers, and missing work

### 3.1 ⚠ L17_dmma's 1.743 µs at B=1 (and 19.738 µs at B=213) do not measure a transform

**The claim.** `L17_dmma` leads the L=17 non-batched cell at 1.743 µs / **172.86 GF/s**,
4.47× ahead of the structurally near-identical `L17_raderfused` at 7.781 µs, having
been 7.635 µs itself last round with the same arithmetic. The record states that "zero
arithmetic was touched" this round.

**Why it is not a single-transform time — the arithmetic kills it before any code
reading.** The kernel is one 320-thread block per volume, so at B=1 the work sits on
**one SM**. An A100 SM has 32 FP64 cores at 1410 MHz = **90.2 GFLOP/s** of FP64 peak.
The leaderboard's 172.86 GF/s is 1.9× that; against the entry's own honest operation
count (87.5 real flop/point × 4913 = 429.9 kflop) the implied rate is **246.6 GFLOP/s,
or 2.7× single-SM FP64 peak**. No scheduling trick makes one SM exceed its own issue
rate. The number therefore cannot be the execution time of one volume, and the 38.5%
run spread — the largest on the board by a factor of five — is the tell.

**The mechanism, which the implementer documents plainly and does not hide.**
`impl_3/L17_dmma.cu:906-913`: `execute()` launches the fused kernel on the next of 8
`cudaStreamNonBlocking` streams **with no event fencing between calls**, and returns.
`driver.cu:192-200` times a sample as `for (i<inner) execute(); cudaDeviceSynchronize();`
and divides by `inner`. So `inner` back-to-back calls pipeline on the GPU, and the
reported "per execute" is the amortized cost of one call in an unbounded stream of
concurrent, identical, mutually-overlapping calls. The strategy record says so in
terms: *"at B=1 the measured 2.7 µs is pipelined throughput of repeated transforms, not
isolated call latency — the contract allows it, the record should not hide it"*, and it
asks the monitor for a ruling. This is exemplary conduct and I am ruling on the number,
not on the implementer.

**The ruling: void at the non-batched and L2 cells, allowed at HBM.**

1. The contract's letter permits it (`fft3d_gpu_api.h:76`: "Asynchronous work is fine:
   the driver synchronizes before stopping the clock"), and correctness passes for a
   real reason: overlapping calls write *identical bytes* to the same `out` from the
   same `in`, so any interleaving yields the same image. I am not calling this cheating.
2. But it is not a transform time. The whole point of the non-batched cell is
   single-transform latency; the ring converts that cell into a measurement of 8-way
   *batch* concurrency, which is precisely what the B=213 and B=13660 cells exist to
   measure. A caller of this API cannot collect the gain: to read `out` it must
   synchronize, which serializes the ring back to ~7.6 µs, and a caller that *does*
   have 8 independent volumes expresses that as B=8, not as 8 overlapping B=1 calls.
   The speedup exists only for a client that issues the same transform repeatedly and
   never looks at the result.
3. It is also not a like-for-like comparison. cuFFT's `cufftExecZ2Z` is asynchronous
   too, but on the NULL stream, which self-serializes; so does every other panel entry
   at its scored configs. Scoring one entry with 8 streams of cross-call concurrency
   against eleven entries and a library with one is a harness artifact, not a result.
4. **Ruled headline for L=17**: `L17_raderfused` 7.781 µs (non-batched) and 31.185 µs
   (B=213), both honestly measured, both still ahead of cuFFT by 1.73× and 2.16×.
   L17_dmma's own ring-free numbers (7.66 µs split-path at B=1; 31.7–31.9 µs at B=213,
   per its r2 and r3 records) are a statistical tie with those, so the ruling costs the
   panel nothing real — it removes a 4.5× phantom, not a 4.5× advance.
5. **The HBM cell is unaffected**: the ring is worth 1.3% there (1571.7 → 1551.9), and
   L17_dmma leads on either basis. 3.07× over cuFFT stands.

**Scope check — is anyone else exposed?** I grepped all twelve files. L17_dmma is the
only entry that removes cross-call ordering outright. Three others have a weaker form —
`L36_globalpass` mode 1 and `L45_pfa`'s `rr`/sliced paths map chunk *n* to stream
*n mod ns* on every call, so chunk *n* is ordered against itself across calls but can
overlap chunk *m≠n* of the previous call. That can only fill a launch-drain tail (the
work all still executes, and a saturated call has no tail worth filling), and the
measurements bear it out: L45_pfa's exposed cell (B=11) moved 36.513 → 35.922, 1.6%.
The persistent ticket kernels in L36_sharedtiled, L36_globalpass, L45_pfa and
L64_radix8 all launch on a single stream with monotonic epoch counters and are fully
ordered. So the exposure is confined to one entry and one mechanism.

**One caveat I will not paper over.** `L36_globalpass`'s −22% at B=22 came from moving
to a 4-stream chunked config on that same weakly-ordered path, and it has **no gpu_r3
strategy record** (§3.2) to tell me how much of it was intra-call parallelism (fully
legitimate: more blocks in flight within one call) and how much was cross-call
tail-fill. I have not ruled it out, because the L=36 B=22 headline does not depend on
it: `L36_sharedtiled` reaches 28.738 µs — 0.09% behind — through an intra-call
fork/join that is fenced with events on both sides (`impl_3/L36_sharedtiled.cu:414-426`).

**Harness recommendation, so the next monitor is not asked to make this call again.**
Either (a) the driver orders the timed loop — record an event after each `execute()` and
have the next iteration wait on it, or sync per call and accept that the B=1 cell then
includes one launch (which is what that cell is for), or (b) the contract adds one
sentence: *work enqueued by one `execute()` must complete before work enqueued by the
next `execute()` on the same plan begins*. I recommend (b) plus a `--calls-ordered`
mode in the driver, because (a) alone destroys the launch-amortization the 20 ms sample
was designed for.

### 3.2 Three implementer agents died; two entries are r2 code with r3 numbers

From `results/gpu_r3/agents/exits.txt` and the per-agent logs — **all three are tooling
faults, not implementation faults**, and I record them so the standings are read
correctly rather than as a judgement on the entries:

| entry | exit | evidence | consequence |
|---|---|---|---|
| `L23_rader` | 1 | `agents/L23_rader.log` is one line: `API Error: Out of memory` | source byte-identical to impl_2; **no gpu_r3 section in its strategy record**; its r3 numbers are a re-measurement of r2 code |
| `L8_blockfused` | 139 | Bun SIGSEGV at `0xBBADBEEF` after **870 ms** ("INSTANT" in exits.txt) — crashed before doing any work | source byte-identical to impl_2; **no gpu_r3 record section**; still holds both L=8 batched cells on r2 code (by 0.08%) |
| `L36_globalpass` | 139 | Bun SIGSEGV at `0xBBADBEEF` after **601846 ms** (10 min of work) | source **did** change (135 diff lines, a new `pf` prefetch knob, mtime 20:38 vs the 20:31 seed) and it **took the B=22 cell**, but the agent died before writing its record: **no gpu_r3 record section** |

`L36_globalpass` is the serious one: it holds a cell with code no one documented. Under
`docs/CURATION.md` ("do not promote … entries whose strategy record is missing — the
record is what makes the code useful later") it cannot be promoted this round, and its
B=22 number cannot be audited (§3.1). The r3 source survives in `impl_3/` either way,
which is exactly why per-round source directories exist.

### 3.3 Nothing failed correctness, built, crashed, or is missing

* `build_errors.txt` contains **no errors** — one `#128-D` "loop is not reachable"
  warning in `L45_pfa.cu:358` (a `for (k = 0; k < 2025/128; ++k)` body in the
  `DIRECT=0` instantiation) plus nvcc's suppression remark. Cosmetic; worth a cleanup
  so a real diagnostic is not lost in it.
* **No `failures.txt`** — nothing crashed or hung during the timed pass.
* `check.log` is `PASS` on every line; the leaderboard's correctness column reads `ok`
  at 1.6e-16 … 8.2e-16 for all 12 entries × 3 batch points, against a 1e-12 gate. The
  weakest is `baseline_gpu` at 8.4e-16 — the harness floor, dense O(L) per output, as
  expected.
* All 12 entries appear at all three of their batch points. No entry is missing.
* Two entries' records disclose that their largest batch point's *independent* numpy
  check could not be run this round (login-node commit limit pinned by other users):
  `L64_radix8` at B=256 and `L45_pfa` at B=736 rest on PASS at smaller batches on the
  identical code path plus bit-identical output across configurations. The scoring
  pass's own `check.py` passed at those points, so this is a record-keeping note, not
  an open correctness question — but it is the second round running that the login node
  has blocked implementer-side verification, and it is worth fixing at the
  infrastructure level.

---

## 4. Claimed versus measured

The premise in my brief — a cross-machine correction — does not apply (see the note at
the top). Implementers measured on **leased slices of the same A100 model on the same
node**, sharing it with other agents; I scored on the same device pinned exclusively,
with clocks unlocked but 20 ms samples. The residual variable is **boost-clock state and
lease contention**, which the records themselves quantify at ~2–3% lease-to-lease and
up to 13% within a noisy window. Agreement is correspondingly excellent: ten of twelve
entries land within ~1% of their claims at every scored cell.

| entry / cell | claimed | measured | gap | attribution |
|---|---|---|---|---|
| L13_dmma B=1 / 477 / 30549 | 6.78–6.95 / 21.70 / 1563.7 | 6.819 / 21.511 / 1563.051 | ≤0.9% | — |
| L17_raderfused B=1 / 213 / 13660 | 7.80–7.94 / 31.21–31.35 / 1580.3–1583.2 | 7.781 / 31.185 / 1577.899 | ≤0.3% | — |
| L6_batchcoalesced B=1 / 4854 | 2.87 / 14.72–14.88 | 2.832 / 14.599 | ≤1.3% | — |
| L6_warpvolume B=1 / 4854 / HBM | 3.09–3.13 / 14.92–15.11 / 1540.4 | 3.064 / 14.876 / 1538.475 | ≤0.9% | — |
| L8_warpradix8 B=1 / 2048 / HBM | 3.061 / 13.189 / 1536.5–1539.2 | 3.055 / 13.176 / 1538.645 | ≤0.2% | — |
| L64_radix8 B=1 / 4 / 256 | 18.56 / 41.9 / 2617.5 | 18.501 / 39.616 / 2614.357 | ≤5.5% (B=4 only) | B=4's 41.9 was "sd 0.25%" on a contended lease; the exclusive pass is faster |
| L45_pfa B=736 | 2112–2117 | 2110.272 | 0.1% | their record staked a ±3% prediction on this and won it |
| **L45_pfa B=1** | **14.6** | **12.626** | **−13.5%** | **lease clock state.** Their own record calls it: "12.7 on r2's warm clocks, path unchanged … the wobble is mostly clock state". The exclusive pass reproduces the warm-clock number exactly. Not a code effect — the path is bit-identical to r2's. |
| **L36_sharedtiled B=22** | **27.59** | **28.738** | **+4.2%** | just outside their stated ~3% lease-to-lease spread. Both L=36 entries' B=22 configs are picked by a create()-time autotuner, so part of this is which config the tuner picked on the scoring run, not a fixed kernel time. |
| L36_sharedtiled B=1 / 1438 | 9.125 / 1932.1 | 9.062 / 1931.520 | ≤0.7% | — |
| **L17_dmma B=1** | **2.76 (best window 2.67)** | **1.743** | **−37%, i.e. measured far *better* than claimed** | **not a machine effect.** This is the one gap in the round that a clock story cannot cover, and it is diagnostic: with the stream ring, the number is a function of how many calls the harness pipelines (30 samples × a large auto-calibrated `inner` on an exclusive device) rather than of the kernel. Its B=213 (19.76 → 19.738) and HBM (1552 → 1551.872) claims match to 0.1%, because those cells are closer to saturated. A number that improves 37% when you give the harness a quieter machine and more consecutive calls is measuring the harness. See §3.1. |
| L23_rader, L8_blockfused | no r3 claim | — | — | agents died (§3.2) |

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

**§4.3 — "Is axis fusion worth 3× or 3%?", specifically the L2↔DRAM regime that §4.3's
own addendum names "the largest untried structural move on the board."**

§4.3 stood as: settled in part by `panel_r3` for the CPU **L1↔L2** boundary (single-digit
percent, sometimes negative), and explicitly *re-opened* for **L2↔DRAM**, where the
bandwidth gap is ~7× and where Intel's manual, Alappat et al. and the L3-Fusion result
all recommend the same construction — *tile the batch so a tile fits the near level, then
run all three axes inside the tile*. This round is the first to run that experiment at
scale on the GPU, where the analogous construction is the persistent producer/consumer
ticket kernel: one launch per execute, a grid of exactly one resident wave, blocks
pulling per-plane tickets off a global atomic, consumers spinning on per-volume done
counters, so the intermediate lives in L2 for ~`lead` volumes and never round-trips to
HBM at a kernel boundary. Four entries measured it at the HBM point, and the answer has
**both signs, with the mechanism identified in each case**:

* **L45_pfa (L=45, PFA 9×5): fusion wins, −13.4%** (2436.8 → 2110.3 µs at B=736;
  −13.7% in their own leases). Its r2 record had predicted −13% from L36_globalpass's
  r2 result and landed within 0.3 points of it.
* **L36_sharedtiled (L=36): fusion wins, −7.1%** (2079.9 → 1931.5 µs at B=1438), by
  porting L36_globalpass's r2 kernel — and the port turned up a second-order finding
  worth more than the structure argument: the **shared-memory carveout** had been wrong
  at 100 for three rounds (634.6 → 587.5 µs at B=432 when set to 50), because
  all-streaming kernels never noticed the starved L1 but a ticket kernel whose
  intermediate reads hit L2 does.
* **L64_radix8 (L=64): fusion LOSES, +5.6%** (2617.5 chunked vs 2765 fused at B=256;
  713 vs 662 µs at B=64) — the round's most valuable single result, because it bounds
  the construction. Two mechanisms, both measured: (i) the register-union of two kernel
  bodies under a 64-register cap spilled 168 B/thread (fixed to 132 B by chained
  on-the-fly twiddles, still a loss), and (ii) **ncu shows the fused kernel writing
  493 MB against a 268 MB compulsory floor — the intermediate leaked to HBM anyway**, so
  there was no round trip left to save and nothing to pay the ticket loop's 5 syncs +
  fence + atomic + spin per 64 KB plane. `L45_pfa` independently found the register half
  of the same trap and fixed it with `__device__ __noinline__` per body (416 → 223 µs at
  B=64) — so the two entries triangulate the failure mode.

**What §4.3 gains, in one sentence:** Tolmachev's rule (payoff = avoided passes ×
bandwidth gap between the two levels) survives with GPU numbers attached, but the panel
can now state the precondition the literature leaves implicit — *the fused form must
actually keep the intermediate in the near level*; when the working tile is too large
for that (L=64's 64 KB planes at 2 blocks/SM), fusion collects the barrier and
register-pressure cost of the construction with none of its traffic benefit and comes
out **negative**. The measured L2↔DRAM range across four entries is **−13.7% to +5.6%**,
which is comfortably inside "single-digit-to-low-teens percent" and nowhere near 3×.

Secondary movement, worth recording but not new this round: **§4.2(c) / the DMMA
question at L=17 stays closed by measurement** — both L=17 entries now ship the same
double-folded (u/v) dense line module at 87.5 flop/point, the HBM cell runs at 89% of
DRAM peak with the FP64 pipe under half busy, and this round's L=17 work touched
*zero* arithmetic. §4.7 (vector-radix) and §4.5 (padding at L=8) were not moved.

---

## 6. The single highest-value thing the next round should attack

**L = 6 — build the small-block shape, or stop spending rounds here.** B_L2 (14.6 µs
against an 11.6 µs write-stream floor) is now negatively closed from every direction two
entries could measure: not barriers (named-barrier narrowing, both entries), not wave
quantization, not store semantics, not staging shape, not occupancy, not bank conflicts
(ncu: 37.6 K conflicts against millions of shared ops). Exactly one shape is named and
unbuilt: a **V=1–2 volume-major block of 36–72 threads with PSLOT padding** —
L8_blockfused's winning shape scaled down — which trades batch-major bank-conflict
freedom for a much shorter dependent chain. Build it or declare L=6 finished; the other
two cells are at the launch floor and the DRAM wall respectively.

**L = 8 — give L8_blockfused a live agent and have it take the two things it lacks.** It
held both batched cells this round on r2 code with an agent that died in 870 ms, and it
is the only entry still without **CUDA-graph replay**, which L8_warpradix8 measured at
0.2–0.6 µs/call, bit-identical, on every path. Beyond that, both entries have bracketed
the B_L2 residual from opposite sides (occupancy up: quad, worse; barriers deleted:
worse) and agree the remainder is the per-thread instruction stream — so the only
honest lever left is arithmetic: three genuinely fused cross-lane radix-8 stages, one
cmul per axis. Expected ≤0.5 µs; do it after the graph.

**L = 17 — settle the async question, then attack the L2 cell's in-kernel floor.**
Re-measure both entries under whatever ordering rule the panel adopts (§3.1), with the
ring off, so the standings reflect kernels. Then the cell both records independently
name: **B=213 at ~146 ns/transform against a ~60–70 ns issue-floor estimate**, where
L17_dmma's B=108 control measurement is the key datum (15.75 µs for one lone block per
SM vs 31.86 for two co-resident — lock-step pairs buy *zero* overlap at the single-wave
point). The named, unbuilt lever is **2 volumes per 640-thread block** (107 blocks = one
clean wave, halves the barrier count per volume, 96 regs × 640 ≤ 64 K so it fits) —
which is the *in-kernel* way to get the de-phasing the stream ring was faking. Precede
it with the ncu stall pass both records prescribe.

**L = 36 — build the two specialized co-resident persistent kernels.** This is the
highest-value item on the entire board, not just at L=36: **four separate records name
it** (L36_globalpass r2 next-item 1, L36_sharedtiled r3 next-item 1, L45_pfa r3
next-item 1, L64_radix8 r3 next-item 1) and **nobody has built it**. K1-only and K2-only
persistent grids, co-resident on two streams, sharing one ticket queue and the
done-counter handshake, grids ratio-balanced for the work asymmetry (~1.5:1 at L=36).
The case is made by measurement, not taste: every fused single-kernel port this round
paid a register-union tax (L45_pfa 752 B of spills before `__noinline__`; L64_radix8
168 B and a lost round; L36_sharedtiled pinned at 4 blocks/SM by a 72-reg union with
32 B of spills), and two specialized kernels each compile at solo register pressure.
L36_sharedtiled measures 0.4 µs/volume of latency headroom at 43.5% occupancy waiting
for it. Also, whoever draws L=36 must **write the gpu_r3 record L36_globalpass owes**
(§3.2) before trusting its B=22 config.

*(For the geometries outside my brief's list: L=13 — B_L2 at 21.7 µs vs an 11.6 µs floor
is declared structurally finished at 24 warps/SM by its own record, so the next move is
the graph replay from L=8, worth ~0.5 µs at B=1. L=23 — fix the autotuner: 11 s of
setup, 4–5% run spread and a wholly different pick from byte-identical source
(§2) mean its numbers are not yet reproducible; sample past the 20 ms boost cliff.
L=45 — 2.87 µs/volume against a ~1.9 µs two-pass HBM floor with L2 at 78%: the same
two-specialized-kernels idea as L=36, and it is the entry most likely to get there
first. L=64 — the radix-4³ codelet (≈40 regs → 3 blocks/SM) is the last untried
occupancy lever against a ~1.4–1.6 ms composite floor at B=256 vs today's 2.61 ms.)*

---

## 7. Curation: what to keep, and why

Applying `docs/CURATION.md`'s four grounds in order. Ground 4 ("anything that beat a
library") admits all twelve again and therefore filters nothing — stated so it is not
mistaken for a selection. `gpu_r2` promoted ten and dropped `L6_warpvolume` and
`L17_raderfused` as bit-identical near-duplicates; two of those judgements change this
round.

**Ground 1 — fastest correct entry per geometry (eight cells' worth, eight entries).**

* `L6_batchcoalesced` — non-batched and B_L2 at L=6, ties HBM. New this round: the
  dedicated 36-thread B=1 kernel with the z-pass fused into the global load.
* `L8_warpradix8` — takes the L=8 non-batched cell and ties both batched cells with a
  structurally opposite design (volume in registers per warp/pair/quad, no shared).
  **It comes off the r2 monitor's retirement watch list**: r2 dropped it as a
  near-duplicate; r3 it shipped a genuinely new kernel (`fft8_quad`, one quad-local
  barrier via redundant recompute) and universal graph replay, and won a cell.
* `L13_dmma` — sole entry, all three cells, −12.8% at B=1 from the soft-barrier split.
* `L17_raderfused` — **the fastest honestly-measured L=17 entry at both small cells
  under §3.1's ruling.** Also returns from r2's drop list: this round it ported the soft
  grid barrier, replaced the dense x pass with a folded thread-per-output form through
  scratch (11.27 → 8.20 µs at B=1) and found the hot-spin-vs-`__nanosleep` cost.
* `L17_dmma` — holds the L=17 HBM cell on either basis (1551.9 with the ring, 1571.7
  without, vs 1577.9). Promoted on ground 1 for that cell and on ground 3 below.
* `L36_sharedtiled` — non-batched and HBM at L=36, ties B=22 within 0.09%. Largest
  source change of the round (553 diff lines) and the carveout finding.
* `L45_pfa` — sole entry, and the round's largest genuine gain (−13.4% at B=736).
* `L64_radix8` — sole entry, all three cells.

**Ground 2 — structurally different runner-up when it is close.** `L8_blockfused` is the
textbook case on the numbers (0.08% behind at both batched cells, opposite structure) —
but its file is **byte-identical to the `exemplars/gpu_r2` copy** because its agent died
in 870 ms, so promoting it would add a duplicate file and a record with no r3 section.
The r2 exemplar already *is* that code. Same reasoning retires `L23_rader` from this
round's promotion (byte-identical, no r3 record section) and `L6_warpvolume` (kernel
bit-identical to r2 by its own account: six variants built, all lost). All three remain
fully available in `exemplars/gpu_r2` and `impl_3/`.

`L36_globalpass` is excluded on the explicit rule: **its gpu_r3 strategy record does not
exist** (agent SIGSEGV after 10 minutes, §3.2), it is no longer structurally distinct
from `L36_sharedtiled` (which ported its ticket kernel wholesale and says so), and the
one cell it holds is the one I could not audit (§3.1). Its r3 source is preserved in
`impl_3/`, which is what per-round directories are for.

**Ground 3 — instructive failures.** Two, and both ship *inside* promoted entries with
the killing numbers in their records, so no separate loser needs promoting:

* `L17_dmma` — **the round's most instructive record**, and the reason I promote it
  beyond its HBM cell. It contains the stream ring behind `-DL17_NSTREAM=0/8`, the B=108
  control measurement that motivated it (15.75 vs 31.86 µs — lock-step co-resident pairs
  buy zero overlap), the honest disclosure that B=1 became a throughput number, and now
  this verdict's ruling against it. The next panel needs the executable artifact, not a
  paraphrase, so it can re-run both sides under whatever ordering rule §3.1 produces.
* `L64_radix8` — the fused persistent kernel is kept behind `L64_MODE=1` with its full
  negative case: +5.6% at B=256, the lead sweep (2→713, 6→984 µs), the 168 B spill from
  the register union, the ticket-prefetch disaster (713→799, and 962 µs before the spin),
  and the ncu finding that the intermediate leaked to HBM anyway. This is what stops the
  next panel spending a round re-porting L36's win to a geometry where it cannot pay.

Other dead ends likewise ship executable inside the promoted winners with numbers
attached: `L17_PIPE`/`L17_NESTED`, `L17RF_SPIN_NS`, `L13_FORCE_WCP`, `NAMED_BAR`,
`-DMINB=n`, `L8WR_GRAPH=0`, `FFT45_PERS/PD1/PD2/LEAD`, `L64_LEANTW`, and
L36_sharedtiled's carveout and MINB knobs.

**Eight promoted, four dropped** — three of the four on bit-identical-code grounds
(their r2 exemplars stand in for them) and one on the missing-record rule.

```bash
cd bench/gpu
./promote.sh gpu_r3 L6_batchcoalesced L8_warpradix8 L13_dmma L17_raderfused \
    L17_dmma L36_sharedtiled L45_pfa L64_radix8
git add exemplars/gpu_r3 strategies results/gpu_r3 impl_3 && git commit
```

**Retirement watch list for gpu_r4:** the two L=36 entries — they now run the same
persistent ticket kernel and are separated by 0.09–0.55%; if r4 does not differentiate
them (the two-specialized-kernels idea in §6 is the obvious way), one should be
retired. And `L17_dmma`/`L17_raderfused` are converging the same way: both now ship the
same `line17w` module, the same soft grid barrier and the same batch-selected staging.
Once §3.1's ordering rule removes the stream ring from the comparison, r4 should check
whether two L=17 files are still buying the panel anything.
