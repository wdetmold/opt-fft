# Round gpu_r4 — monitor's verdict

Measured on `a80n1.lqcd.mit`, slurm job 438580, 2026-08-22T22:28:49-04:00, one NVIDIA
A100-SXM4-40GB (device 0 pinned; SM clock 1410 MHz, persistence off, clocks not lockable
on this cluster), CUDA 12.2, driver 525.125.06. 3 independent processes × 12 samples per
cell, inner repeat count auto-calibrated to ≥20 ms per sample.

**Two notes on the brief before the numbers.** (1) The brief describes a CPU pass — Xeon
Gold 5218, AVX-512, MKL, and geometries L = 6/8/17/36. This round is the *GPU* round in
`bench/gpu`: eight geometries (L = 6, 8, 13, 17, 23, 36, 45, 64), three batch points each,
and the only library baseline is cuFFT 11.0. I score what was actually measured, headline
the four L the brief names, and give the other four compactly. (2) The single library is
cuFFT; `baseline_gpu` is our own library-free row-column reference, i.e. the harness floor,
not a baseline to beat. The brief's Sapphire-Rapids-vs-Cascade-Lake caveat does **not**
transfer: implementers develop on leased A100s on this same node (`tryout.sh` ssh's to the
reserved node), so claimed and scored numbers come from the same part — see §4.

---

## 1. Headline per geometry

Batch points are defined by working set, per `cases.txt`: B=1 is the launch/latency case,
B_L2 puts in+out ≈ 32 MiB inside the 40 MiB L2, B_HBM puts one buffer ≈ 1 GiB and is the
primary score. Times are per call (best of 3 runs' min sample, as `leaderboard.py` scores).

### The four the brief names

| L | case | fastest correct panel entry | cuFFT | speedup |
|---|---|---|---|---|
| **6** | B=1 | **L6_warpvolume 1.723 µs** | 10.423 µs | **6.05×** |
| | B_L2 = 4854 | **L6_warpvolume 10.311 µs** (3943 GF/s) | 51.433 µs | **4.99×** |
| | B_HBM = 310608 | **L6_warpvolume 1535.351 µs** (1694 GF/s) | 3513.344 µs | **2.29×** |
| **8** | B=1 | **L8_warpradix8 2.192 µs** | 9.170 µs | **4.18×** |
| | B_L2 = 2048 | **L8_warpradix8 8.929 µs** (5285 GF/s) | 54.539 µs | **6.11×** |
| | B_HBM = 131072 | **L8_warpradix8 1534.379 µs** (blockfused 1536.640, a tie) | 3692.851 µs | **2.41×** |
| **17** | B=1 | **L17_dmma 2.297 µs** | 13.535 µs | **5.89×** |
| | B_L2 = 213 | **L17_dmma 15.716 µs** (4082 GF/s) | 67.268 µs | **4.28×** |
| | B_HBM = 13660 | **L17_raderfused 1550.763 / L17_dmma 1551.957 µs** — a 0.08% tie inside a 0.8% spread | 4759.168 µs | **3.07×** |
| **36** | B=1 | **L36_sharedtiled 7.967 µs** (globalpass 8.111) | 12.944 µs | **1.62×** |
| | B_L2 = 22 | L36_globalpass 26.061 µs — **not robust, see §3(e)**; honest read is a tie at ≈28.1–28.8 µs | 50.966 µs | 1.96× / 1.81× |
| | B_HBM = 1438 | **L36_sharedtiled 1752.343 µs** (globalpass 1762.583) | 3382.272 µs | **1.93×** |

### The other four

| L | B=1 | B_L2 | B_HBM (primary) |
|---|---|---|---|
| 13 | L13_dmma 1.601 µs vs cuFFT 12.335 (**7.70×**) | 14.084 vs 62.893 (**4.47×**) | 1554.708 vs 4741.120 (**3.05×**) |
| 23 | L23_rader 9.104 vs 14.856 (**1.63×**) | 39.623 vs 76.065 (**1.92×**) | 1989.581 vs 5061.291 (**2.54×**) |
| 45 | L45_pfa 3.723 vs 17.968 (**4.83×**) | 28.629 vs 64.535 (**2.25×**) | 2111.104 vs 4746.496 (**2.25×**) |
| 64 | L64_radix8 18.182 vs 23.609 (**1.30×**) | 41.061 vs 58.085 (**1.41×**) | 2561.792 vs 3736.704 (**1.46×**) |

**Every panel entry beats cuFFT at every point of its own geometry**, by 1.30× to 7.70×.
That makes CURATION criterion 4 ("anything that beat a library") non-discriminating this
round; it cannot carry a promotion on its own and I do not let it.

At B_HBM the four smallest geometries are all pinned to the same wall: 1534–1555 µs is
88–90% of the part's 1555 GB/s peak while moving the compulsory minimum bytes. Three
records independently converge on that number and say so.

---

## 2. What changed since gpu_r3, per geometry

**The round has one story: the asynchronous stream ring propagated.** L17_dmma invented it
in r3 (`execute()` launches on the next of N plan-owned streams, no fencing, so the driver's
back-to-back calls pipeline). In r4 it was adopted by L13_dmma, L6_warpvolume, L8_warpradix8,
L17_raderfused and L45_pfa. Every large B=1 and B_L2 gain below is that mechanism; almost no
arithmetic was touched anywhere in the round. Read §3(b) before reading the B=1 column.

| L | cell | r3 | r4 | change |
|---|---|---|---|---|
| 6 | B=1 | 2.832 (batchcoalesced won) | **1.723** (warpvolume) | winner flipped; ring + per-stream graph |
| 6 | B_L2 | 14.599 | **10.311** | **−29%**, the round's cleanest real gain (sd 0.1%, all 3 runs 10.31–10.32) |
| 6 | B_HBM | 1538.5 | 1535.4 | −0.2%, at the DRAM wall |
| 8 | B=1 | 3.055 | **2.192** | −28% (ring); blockfused 3.245 → 3.016 (graph only) |
| 8 | B_L2 | 13.176 | **8.929** | **−32%** (ring); blockfused 13.165 → 12.513 |
| 8 | B_HBM | 1538.6 | 1534.4 | −0.3%, wall |
| 13 | B=1 | 6.819 | **1.601** scored / ≈2.6 robust | −62% scored, **−61% honest**; planes path retired as ring-unsafe |
| 13 | B_L2 | 21.511 | **14.084** | **−35%** |
| 13 | B_HBM | 1563.1 | 1554.7 | −0.5% |
| 17 | B=1 | 1.743 | 2.297 | **+32% on the board — see below** |
| 17 | B_L2 | 19.738 | **15.716** | **−20%** (`__stcs` evict-first stores, borrowed from L8_blockfused r1 via L13_dmma r2) |
| 17 | B_HBM | 1551.9 | 1552.0 | flat; raderfused 1577.9 → 1550.8 (−1.7%) |
| 23 | B=1 | 9.101 | 9.104 | flat (no ring at this geometry) |
| 23 | B_L2 | 39.033 | 39.623 | +1.5%, inside the 3.2–4.3% run spread |
| 23 | B_HBM | 2275.1 | **1989.6** | **−12.5%** (persistent producer/consumer ticket kernel, ported from L36_globalpass r2 via L45_pfa r3) |
| 36 | B=1 | 9.062 | **7.967** | −12% (direct-form K1/K2, borrowed from L45_pfa r3) |
| 36 | B_L2 | 28.711 | 28.113 / 26.061* | ≈flat, −2%; *the 26.061 is a single-run artifact |
| 36 | B_HBM | 1931.5 | **1752.3** | **−9.3%** |
| 45 | B=1 | 12.626 | **3.723** | **−70%** (ring, made race-safe with a scratch intermediate) |
| 45 | B_L2 | 35.922 | **28.629** | −20% scored, ≈−11% on medians |
| 45 | B_HBM | 2110.3 | 2111.1 | flat — the ring was measured and *rejected* here, see §5 |
| 64 | B=1 | 18.501 | 18.182 | −1.7% |
| 64 | B=4 | 39.616 | 41.061 | **+3.6% on the board** |
| 64 | B_HBM | 2614.4 | **2561.8** | **−2.0%** (SHFL → dead-slot XOR-swizzled shared exchange; ncu showed MIO throttle) |

### Did anything regress?

**No entry genuinely regressed. Two cells look worse on the board and neither is real:**

* **L=17, B=1: L17_dmma 1.743 → 2.297 µs (+32%).** This is the board becoming honest, not
  the code becoming slower. r3's 1.743 carried a **38.5% run spread** — it was the lucky tail
  of a bimodal distribution. r4's three runs are 2.297 / 2.298 / 2.478 with per-run medians
  2.300 / 2.301 / 2.482, i.e. a 7.9% spread, because the entry added CUDA-graph replay
  specifically to collapse launch-path noise (their record: "the B=1 run-to-run sd collapsed
  from 38.5% … to ~1–2%, so the scored number should stop bouncing"). It did. Note the
  side-effect: L17_raderfused sized its ring depth to 16 *in order to* beat "dmma's ring-8
  drain floor of 13.5/8 = 1.7 µs — exactly their scored 1.743". They tuned against a noise
  artifact, and the tuning did not pay at scoring time (2.336 scored, 2.60–2.65 on medians).
* **L=64, B=4: 39.616 → 41.061 µs (+3.6%).** The r3 number carried a 2.9% spread and the
  entry's own r4 lease measurements are 41.0–41.7 vs r3's 41.9 — i.e. flat-to-slightly-better
  on their instrument. The kernel change this round is bit-identical arithmetic. Noise.

**One real cost increase, unscored:** L23_rader's plan setup grew from 0.043 → 0.967 s at
B=1, 0.267 → 5.130 s at B=86 and 10.98 → 15.90 s at B=5515, because its autotuner now sizes
final samples past the 20 ms boost-clock cliff. Setup is excluded from the timed window by
the contract and the change bought a correct pick (their classic candidate re-times 2280 →
2313 µs under honest sample lengths), so this is a good trade — but 16 s of plan time per
call to `create()` is now the largest setup cost on the board by two orders of magnitude and
should not grow again.

---

## 3. Adversarial pass: correctness, builds, crashes, gaps

**Nothing failed and nothing is missing.** Verified rather than assumed:

* `agents/exits.txt`: all 12 implementers exit 0.
* 76/76 `c_*.json` correctness files report `ok: true`. Worst relative-L2 error on the board
  is **8.2e-16** (L45_pfa) against a **1e-12** gate — four orders of margin. Range across the
  panel: 1.6e-16 … 8.2e-16.
* 108/108 own-geometry timing files present (12 entries × 3 batch points × 3 runs). No entry
  claims support for a geometry that is not its own; the cross-geometry `t_*.json` files all
  read `supported: false`, which is the expected shape.
* `failures.txt` does not exist (nothing crashed or hung).
* `build_errors.txt` contains **no error** — one nvcc warning, `#128-D "loop is not
  reachable"` in `impl/L45_pfa.cu:383`, in a `for (k = 0; k < 2025/128; ++k)` whose bound is
  a compile-time constant in one template instantiation. Same warning as r3 (then at line
  358). Cosmetic; worth a `if constexpr` guard so a real warning is not lost in the noise.
* The driver's own anti-memoization check (`driver.cu:230-253` — poke the input, require the
  output to change) passed for every entry, so no one is returning a cached answer.

That said, four things about *how* this round's numbers were produced need to be on the
record. None of them is a failed entry; all of them change how a cell should be read.

### (a) No fast wrong answers

I went looking specifically for one. There is none: every entry's error is at double-precision
round-off, every record reports bit-identical re-runs across sample counts (i.e. across
different ring phases), and `compute-sanitizer memcheck` is reported clean on the new paths in
all twelve records. The checked output comes from the same code path as the timed one
(`driver.cu:203-206`).

### (b) Every B=1 cell that improved this round is pipelined throughput, not latency

The driver times `inner` back-to-back `execute()` calls and synchronizes once per sample
(`driver.cu:192-200`). Six entries — **L13_dmma, L17_dmma, L17_raderfused, L6_warpvolume,
L8_warpradix8, L45_pfa** — launch each call on the next of 4–16 plan-owned streams with no
fencing, so calls *n* and *n+1* overlap on the GPU. The scored "B=1" number is therefore the
per-call cost of a deep pipeline of identical transforms, not the latency of one isolated
transform.

**This is legal and it was disclosed.** `fft3d_gpu_api.h:75-77` says outright "Asynchronous
work is fine: the driver synchronizes before stopping the clock", and line 24 says the driver
"synchronizes once per timed sample". More to the point, every one of the six wrote the
distinction into its own record unprompted, in nearly the same words ("this is pipelined
throughput of repeated transforms, not isolated call latency"), and every one left a
one-flag rollback. That is exactly the conduct the panel wants, and I am not penalising it.

But the cell must be read correctly. Both readings, from the records' own synchronous
fallbacks:

| entry | B=1 scored (pipelined) | isolated single-call latency | flag |
|---|---|---|---|
| L13_dmma | 1.601 µs | 6.78 µs | `L13_NSTREAM=0` |
| L17_dmma | 2.297 µs | ≈2.7 µs (r3 pre-graph) | `L17_NSTREAM=0` |
| L17_raderfused | 2.336 µs | 7.9 µs | `-DL17RF_NSTREAM=0` |
| L6_warpvolume | 1.723 µs | 2.86–2.96 µs | `L6_NSTREAM=0` |
| L8_warpradix8 | 2.192 µs | ≈3.06 µs | `L8WR_NSTREAM=0` |
| L45_pfa | 3.723 µs | 12.81 µs | `FFT45_RING=0` |

Even under the isolated reading every entry still beats cuFFT at B=1 (cuFFT is 9.2–18.0 µs
across these geometries), so no ranking against the library changes. What changes is the
panel's internal ranking at L=6 and L=8, where the r4 rivalry is now "who adopted the ring",
not "whose kernel is better" — see §6.

**Recommendation for the harness, not for the entries:** the B=1 row should be reported as
two cells, pipelined and synchronous. Both are meaningful workloads (a solver issuing many
independent small FFTs really does get the pipelined number), and the panel is currently
spending rounds optimizing a cell whose definition is ambiguous.

### (c) L45_pfa's ring is correct only because the driver replays one input

`impl_4/L45_pfa.cu:745` allocates **one** scratch intermediate `scr` for the whole plan, and
`execute()` (lines 862-869, 884-893) hands the same `scr` to every ring slot. The per-slot
privacy the entry built is on the *ticket counters* (`p->ctr + slot*(1+B)`), not on the data
buffer. The code states its own safety argument at line 861: "safe: the intermediate is scr,
all overlapped writes byte-identical" — which is true exactly while every in-flight call
transforms the *same* input. A caller that issues `execute(in1, out1)` then `execute(in2,
out2)` on one plan with no intervening sync — which the async contract invites — gets call
2's K1 output feeding call 1's K2. The single-kernel ring entries do not have this property:
they touch only `in` (read) and `out` (written), so distinct buffers make overlapped calls
independent.

This does not violate the contract as written (`fft3d_gpu_api.h:41`: "same plan, same input,
same answer") and it did not affect any scored number. It is a real limitation of the fastest
L=45 entry and the fix is cheap — one scratch per ring slot is ≤ 8 × 14 volumes × 1.4 MB at
the batches where the ring is on. **L45_pfa should do this in r5 and say so in the record.**

### (d) A latent race in five entries that L17_raderfused already found and fixed

`driver.cu:204` and `:239` issue `cudaMemset(d_out, …)` on the **legacy/NULL stream**
immediately before the checked `execute()` and the anti-memoization `execute()`. Kernels
launched on a `cudaStreamNonBlocking` stream are not ordered against NULL-stream work.
**L13_dmma, L17_dmma, L6_warpvolume, L8_warpradix8 and L45_pfa all use
`cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)`**, so their checked output is
produced by a kernel that is formally racing that memset. **L17_raderfused diagnosed this and
switched to blocking streams** (`impl_4/L17_raderfused.cu:43-47`, `cudaStreamCreate`), which
order against the NULL stream while still not serializing against each other, at zero
measured cost.

The failure mode is a zeroed output, i.e. a loud correctness FAIL, not a silent wrong answer —
which is why the round is clean. But it is a coin-flip that has come up heads 15 times, and it
will eventually flake a scoring run. **Every ring entry should take raderfused's blocking-stream
form in r5.** It is free and it was already paid for by someone else.

### (e) Three scored cells are decided by a single lucky run

`leaderboard.py:81` scores `min` over runs of the per-run `min` sample. Under ring pipelining
the B=1 and small-batch distributions are bimodal, so best-of-3-of-12 can land far off the
typical value. Comparing the scored number against per-run medians:

| cell | scored | per-run mins | per-run medians | verdict |
|---|---|---|---|---|
| **L13_dmma B=1** | **1.601** | 1.601 / 2.052 / 2.605 | 1.608 / 2.642 / 2.747 | one run out of three; honest ≈2.6 µs — and the implementer's own claim is 2.64–2.72 |
| **L6_warpvolume B=1** | **1.723** | 1.723 / 2.242 / 2.281 | **2.246 / 2.283 / 2.333** | one *sample* out of 36; that run's own median is 2.333. Honest ≈2.28 µs |
| **L36_globalpass B_L2=22** | **26.061** | 26.061 / 28.747 / 28.776 | 26.280 / 28.804 / 28.806 | **the L=36 B=22 "win" is not real** — sharedtiled's medians are 28.21/28.80/28.93. It is a tie |
| L45_pfa B_L2=11 | 28.629 | 28.629 / 30.313 / 30.937 | 32.367 / 32.497 / 32.718 | scored 13% below typical |
| L17_raderfused B=1 | 2.336 | 2.336 / 2.589 / 2.600 | 2.603 / 2.619 / 2.654 | scored 11% below typical |

Consequences I apply below: L6_warpvolume still wins L=6 at B=1 (2.28 vs batchcoalesced's
2.65 on medians) but by ~15%, not the 1.53× the board prints; L13_dmma's real r3→r4 B=1 gain
is ≈61%, not 76%; and **L36_globalpass does not hold a cell**. Everything else on the board
survives the median test unchanged — 55 of 60 cells have median-of-medians within 5% of the
scored min.

---

## 4. Claimed versus measured

The brief asks me to attribute large claim/measurement gaps to the machine difference. **That
mechanism does not exist in this round.** `tryout.sh` builds on the shared filesystem and runs
on the *same reserved A100 node* the sweep scores on, so implementers and monitor share the
part. The residual variance is lease-window boost state and host contention (eight agents
share the node's CPUs), plus the best-of-N selection in §3(e) — not architecture.

And the calibration is correspondingly good: **10 of 12 entries land within a few percent of
their own claims at all three points.** Spot check of the primary (B_HBM) cell:

| entry | claimed | measured | Δ |
|---|---|---|---|
| L45_pfa | 2110–2113 µs | 2111.104 | +0.0% |
| L36_globalpass | 1762.4 | 1762.583 | +0.0% |
| L17_dmma | 1552.3 | 1551.957 | −0.0% |
| L23_rader | 1963–2001 | 1989.581 | inside |
| L64_radix8 | 2568.2 | 2561.792 | −0.2% |
| L8_blockfused | 1536.4 | 1536.640 | +0.0% |
| L36_sharedtiled | 1713.6 | 1752.343 | **+2.3%** |

The gaps worth naming:

1. **L17_raderfused, B=1: claimed 1.804–1.817 µs "in quiet windows", measured 2.336 (medians
   2.60–2.65) — 1.4× optimistic, the largest gap on the board.** Cause is in their own record:
   "B=1 tryout is bimodal with host contention (8 agents share the node's CPUs)" and the cell
   is host-enqueue-bound. They then assumed the scoring window "resembles the quiet mode". It
   did not. This is the one place a claim materially oversold, and it is a *host-side*
   contention artifact, not a device difference. Their ring-depth-16 choice rests on the same
   quiet-window measurement and should be re-swept under scoring conditions.
2. **L36_sharedtiled, B_HBM: claimed 1713.6 µs, measured 1752.3 (+2.3%).** Within the ±3%
   lease clock variance L45_pfa's r3 record warned about for exactly this situation. Their
   B=1 claim (8.066) measured *faster* (7.967), which is the signature of window noise rather
   than a systematic error.
3. **Two claims were beaten by the board for the wrong reason.** L13_dmma claimed 2.64–2.72 µs
   at B=1 and scored **1.601**; L6_warpvolume claimed 2.30 and scored **1.723**. Here the
   *implementers were the honest ones* and the leaderboard flattered them, via the best-of-3
   selection in §3(e). When a scored number beats its author's own claim by 1.3–1.7×, the
   scoring rule is the thing to suspect.
4. L8_warpradix8 quoted B_L2 as "8.85–10.45 µs by window" — an 18% self-reported spread — and
   scored 8.929, the fast end. The claim is honest but the cell is not yet a stable
   measurement.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### §4.3 — "Is axis fusion worth 3× or 3%?" — **moved substantially, on the GPU side**

§07 argues from Hong–Kung that fusing all three axes is the only optimal strategy when the
volume fits the fast level; §07's own gap 7 counters that TurboFNO measured 3–5% and calls
that "the better prior". This round separates those two claims by making the volume-fits
condition the independent variable:

* **Volume fits the fast level (shared memory).** L=17 (4913 points, 78 KB): one fused
  3-axis kernel vs the same entry's two-pass plane-split, same lease — **2.69 vs 7.66 µs at
  B=1**. L=13 (2197 points): **2.65 vs 8.87 µs at B=8**. That is **2.8–3.3×**, and both
  entries retired their split paths on the strength of it. The mechanism is exactly the one
  §07 predicts: the split form pays a global round trip through an intermediate that the
  fused form never materializes.
* **Volume does not fit.** L=36 (46656 points, 746 KB against 163 KB of shared memory) —
  fusion is not slow, it is **unavailable**. Both L=36 entries independently reached the same
  terminal statement: four L2 accesses per point are compulsory for a two-pass form, and
  ncu now shows **L2 at 89.0% (globalpass) and 91.6% (sharedtiled) speed-of-light** with
  DRAM at 66–68%. L36_sharedtiled's record puts it plainly: "the only lever that changes the
  constraint is fewer L2 accesses, i.e. a genuinely fused 3-axis pass, which needs the whole
  746 KB volume visible to one block — impossible in 163 KB shared."

**So the answer is neither 3× nor 3%: it is 3× where it is possible and 0 where it is not,
and the boundary is the capacity of the fast level, exactly where Hong–Kung says it should
be.** The 3–5% GPU-literature prior appears to be measuring fusion *on top of* an already
L2-resident pipeline, which is the L=36 regime, not the L=13/17 regime. Caveat: §07 gap 7
asks specifically about *CPU L1-resident* cubes and three stride-parameterised passes; this
is adjacent evidence from a different memory hierarchy, so it informs that question without
closing it. The named CPU experiment (§05 §10.5) is still unrun.

### §4.2 — "L=17: dense-symmetric, Rader, or Winograd?" — **question (a) is effectively settled, and the answer is "the formulation stopped mattering"**

The two L=17 entries have **converged**. Both now report the identical operation count —
"line17w, 496 real flops/line, 867 lines/volume = 87.5 flop/point, one global read + one
global write" — in their own records, word for word. The entry still named `L17_raderfused`
runs conj-folded cyclic/negacyclic 17-point lines, i.e. §02's dense conjugate-symmetric form,
not Rader; the name is now a fossil. Neither entry touched arithmetic this round.

The entire measured spread between them came from memory and launch mechanics: **15.716 vs
19.491 µs at B_L2 (24%, and it is `__stcs` evict-first stores)**, 2.297 vs 2.336 at B=1
(launch path), and a 0.08% tie at the primary point. **On this hardware, batch-vectorised, the
answer to §4.2(a) is that dense-symmetric wins by default because Rader converged onto it,
and the 20–24% that is actually available at L=17 lives in cache-hint and launch decisions,
not in the flop count.** §4.2(b) (the symmetric/antisymmetric convolution split) and (c) (the
Winograd op count) remain untouched.

### Also touched, secondary

* **§4.6, model versus search.** Two independent results this round say ncu's stall
  attribution is a symptom, not a cost model. L23_rader cut ticket barriers 45 → 29 where ncu
  showed CTA-barrier waits at 35.7% of stall cycles, and got **1%** (2026.7 → 2023.4 µs);
  L36_globalpass's r2 postmortem had predicted exactly that. Meanwhile L64_radix8's *only*
  ncu-guided change that paid was a pipe-level one (MIO throttle → move the 8×8 exchange off
  SHFL into dead shared slots, 2617 → 2568 µs). Reading: profile counters localize the pipe,
  they do not price the fix.
* **§4.1 (register liveness), §4.5 (padding), §4.7 (vector-radix): not moved.** No entry
  changed arithmetic this round — every record says so explicitly. That is worth noticing:
  four consecutive rounds of gains have now come from scheduling and memory, none from
  reformulating the transform.

---

## 6. The single highest-value thing the next round should attack

**Across the board first:** this round's cross-entry comparisons are confounded. At L=6, L=8
and L=17 the rivalry is currently "who adopted the stream ring", not "whose kernel is
better" — L8_blockfused loses B=1 and B_L2 by 1.38–1.40× while tying at the primary point,
and the entire gap is launch path. **r5 should normalize the launch path first, then compare
kernels.** L8_warpradix8's own record makes the offer: "for any entry not yet on the ring
(L6 pair, L8_blockfused, L23, L36, L64): plan-owned streams plus one modulo counter,
bit-identical by construction". Two traps are already documented and paid for: one graph exec
*per ring slot* (L17_dmma r4), and re-tuning under the ring because synchronous-world
rankings do not survive it (L17_dmma r3, reproduced by L8_warpradix8 at B=2048).

Per geometry:

* **L = 6 — stop optimizing the cells that are at the wall; settle the B=1 definition.**
  B_L2 is 10.31 µs at ~3.26 TB/s effective through L2 and B_HBM is 90% of DRAM peak at
  minimum bytes; both records agree there is nothing structural left, and I agree. The only
  cell not against hardware is the *isolated* B=1 latency (2.86–2.96 µs synchronous), which
  no one has attacked because the pipelined number hides it. Either score it separately (§3b)
  or retire one of the two L=6 entries and give the slot to a geometry with headroom — L=6 is
  currently spending two implementer slots to re-measure the DRAM wall.
* **L = 8 — put L8_blockfused on the ring, then re-run the shared-vs-register question.**
  This is the one geometry where the panel holds two genuinely different kernel structures
  (thread-per-line in padded shared vs volume-per-warp in registers) and they tie at the
  primary point (1536.6 vs 1534.4, 0.15%). Right now nobody can say which is better, because
  one of them is measured through a slower launch path. Equalize it and the answer falls out
  in one round — and it is the answer §4.1 and §4.5 actually want.
* **L = 17 — the B_L2 residual, with ncu, and a decision about the second entry.**
  L17_dmma's own next-step is the right one: 15.8 µs against a ~10.8 µs pure-write floor
  (213 volumes × 78,608 B at 1555 GB/s, reads now L2-resident). One ncu pass says whether the
  remaining 5 µs is issue mixture or genuinely write-bound; if write-bound, L=17 is closed
  and should be said so. Separately, the two entries have converged (§5) — the panel should
  either re-diversify L17_raderfused onto a structurally different formulation (§4.2(b), the
  symmetric/antisymmetric convolution split, is the corpus's own untried suggestion) or free
  the slot.
* **L = 36 — land the ring at B=1 and B=22, where it is the only untried lever left.**
  This is the highest-value single experiment in the round's residue. L=36 is the only
  geometry the ring never reached, and its two unattacked cells (B=1 at 7.97 µs, B_L2 at
  ≈28.1 µs, both flat for two rounds) are exactly the cells the ring moved by 25–45%
  everywhere else. The obstacle is known and priced: L45_pfa measured that a **two-kernel**
  execute cannot ring in place — overlapping calls race intermediate bytes against final
  bytes on `out` — and that buying safety with a scratch intermediate cost **+27% (2086 →
  2650 µs) at an HBM batch**, because the scratch's dirty lines get evicted to HBM as an
  extra ~1 GB writeback stream. **But that tax is HBM-specific.** At L=36's B=1 (1.42 MiB)
  and B_L2=22 (31.3 MiB) the scratch stays inside the 40 MiB L2 and never generates the
  writeback stream that killed it at L=45 B=736 — which is precisely the regime where
  L45_pfa's own ring *won* (B=1 −70%, B=11 −20%). L45_pfa even wrote the invitation:
  "L36_globalpass/L36_sharedtiled's fused ticket kernels write the final image once and could
  ring for free — worth their checking." Do that, and keep B_HBM=1438 unringed and in place.
  Expected: 20–40% at two cells, from an already-debugged mechanism. Everything else at L=36
  is at the L2 roof (89–92% SOL) and should be left alone.

Briefly, the other four: **L=13** — B_L2 at 29.5 ns/transform is the only open cell; the
r2/r3 structural dead ends (V=2, wcp, loadz) were all measured *without* the ring and one may
flip sign in the de-phased regime, which is one lease command each. **L=23** — the only
geometry with no ring and no DMMA formulation; its own r5 agenda (two co-resident specialized
persistent kernels) is speculative, whereas the ring is proven, so ring it first. **L=45** —
fix the shared scratch (§3c), then the cyclic-scratch window with per-slot k2done counters
that its own record specifies, which is the only path to a ring at B_HBM. **L=64** — the
occupancy wall is the whole story (40% achieved, No-Eligible 78–83%, every pipe ≤35% SOL);
the radix-4³ + half-plane rewrite is a full round for an estimated 1.4–1.6 ms against today's
2.57 ms, and it is the only idea on the board with that much upside left.

---

## 7. Curation decision

Applying `docs/CURATION.md` in order. Criterion 4 (beat a library) is satisfied by all twelve
entries and therefore discriminates nothing; criterion 1 does the work, criteria 2 and 3
admit two more.

**1. Fastest correct entry per geometry — one per L, always:**

`L6_warpvolume` (wins all three cells, and still wins B=1 by ~15% on medians),
`L8_warpradix8` (wins B=1 and B_L2, ties B_HBM), `L13_dmma` (sole entry),
`L17_dmma` (wins B=1 and B_L2 by 24%; the 0.08% B_HBM deficit is inside a 0.8% spread),
`L23_rader` (sole entry), `L36_sharedtiled` (wins B=1 and the primary B_HBM cell;
globalpass's B=22 lead does not survive the median test, §3e), `L45_pfa` (sole entry),
`L64_radix8` (sole entry).

**2. A structurally different runner-up when it is close — `L8_blockfused`, and only it.**
I applied one uniform test: *tie at the primary (B_HBM) cell **and** a genuinely different
kernel structure.* L8_blockfused passes both — 1536.640 vs 1534.379 µs is a 0.15% tie at the
primary point, and a thread-per-line padded-shared block-fused kernel is a real alternative to
volume-per-warp cross-lane registers. Its B=1/B_L2 deficits are launch path, not kernel, which
is exactly why the next panel needs the code (§6, L=8).

The three other runners-up fail the second half of the test:
* `L6_batchcoalesced` ties at B_HBM but is the same design as the winner (both are 8-volumes-
  per-block batch-major swizzled shared with a DIT 2×3 codelet); it differs only in not having
  adopted the ring. Near-duplicate — CURATION forbids it. Its instructive failure is preserved
  in `strategies/` (see below).
* `L17_raderfused` nominally leads B_HBM by 0.08% but has **converged onto its rival**: both
  entries now publish the identical 496-flops-per-line / 87.5-flop-per-point kernel (§5), and
  it lost B_L2 by 24%. Promoting it would put two copies of the same formulation in front of
  the next panel under different names. **However — its blocking-stream fix (§3d) is the one
  piece of code in this round that five other entries need**, and its record is tracked in
  `strategies/L17_raderfused.md` regardless of promotion. r5's brief should point at it.
* `L36_globalpass` is within 1–2% everywhere, but it and sharedtiled now share a chassis
  (sharedtiled's own record: globalpass's "r2 ticket design remains the chassis everything
  else hangs off"), both took the same L45_pfa direct-form borrow this round, and its one
  scored win is a single-run artifact. Near-duplicate. `impl_4/` preserves its source in full,
  which is what per-round source directories are for.

**3. Instructive failures — none promoted as a separate entry, deliberately.** This round's
three best-documented negatives all live *inside* entries already being promoted or inside
`strategies/`, which is tracked: L45_pfa's scratch-intermediate writeback tax (2650 vs 2086 µs
at B=736 — the number that scopes the L=36 recommendation in §6), L64_radix8's ticket
coarsening (G=1 709 → G=8 1000 µs), L8_blockfused's output-parity-split B=1 kernels (3.365 /
4.248 vs 3.065 µs, which delimits exactly when the register form beats the shared form), and
L36's 5-blocks-per-SM attempt now refuted from three directions (595.7 vs 540.9 µs). None
needs a promotion slot of its own; all are named here so the r5 brief can cite them.

**Round note for `exemplars/gpu_r4/NOTES.md`:** the round established that the asynchronous
stream ring transplants across every single-kernel geometry (−25% to −70% at B=1, −29% to
−35% at B_L2, ~0 at B_HBM), that it is *not* free for a two-kernel execute (+27% at an HBM
batch through the scratch's writeback stream), that axis fusion is worth ~3× exactly while the
volume fits shared memory and nothing once it does not, and that the L=17 dense-vs-Rader
question dissolved because the two formulations converged. It also established that four
consecutive rounds of gains have come from scheduling, not arithmetic.

PROMOTE: L6_warpvolume L8_warpradix8 L8_blockfused L13_dmma L17_dmma L23_rader L36_sharedtiled L45_pfa L64_radix8
