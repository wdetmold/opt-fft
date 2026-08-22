# VERDICT — round gpu_r1

Monitor report. Measured on **a80n1.lqcd.mit**, slurm job 438580, 2026-08-22T18:36 →
18:46, **one A100-SXM4-40GB** (`CUDA_VISIBLE_DEVICES=0`, pinned on purpose; the other
seven GPUs on the node held idle by the scoring window). nvcc 12.2.r12.2, driver
525.125.06, SM clock 1410 MHz sustained through the sweep. Library reference:
**cuFFT 11.0** (`cufftPlanMany`, Z2Z, batched). Library-free floor: `baseline_gpu`.

**A note on the brief this monitor was handed.** The prompt describes a CPU round —
"Xeon Gold 5218, Cascade Lake, AVX-512", with MKL as the library and a 2.9× machine
spread between Sapphire Rapids and Cascade Lake. That text is stale for this round.
gpu_r1 was measured on an A100, the library is cuFFT, and — decisively for item 4 below —
**every implementer developed on a leased A100-SXM4-40GB of the same node a80n1** via
`tryout.sh`, not on different hardware. So there is no cross-machine correction to apply
here; the claim-vs-measured gaps have a different and better-understood cause, and I
report it as such rather than attributing it to a machine difference that does not exist
in this round.

Geometries measured: **L = 6, 8, 13, 17, 23, 36, 45, 64** (the prompt asks for headlines
at 6, 8, 17, 36; the other four ran and are reported too). Three cases per geometry, per
`cases.txt`: `B=1` (launch/latency), `B_L2` (in+out ≈ 32 MiB, L2-resident), `B_HBM`
(one buffer ≈ 1 GiB, **the primary score**). Batched and non-batched are reported
separately throughout, as the project's methodology requires.

---

## 1. Headline per geometry

Timing convention: minimum over 3 independent processes of each process's minimum sample,
with the across-process spread of those minima in parentheses. All twelve panel entries
are library-free (verified: no `cufft`/`vkfft`/`fftw`/`mkl`/`ducc` symbol anywhere in
`impl_1/*.cu` outside `baseline_gpu`).

### L = 6 — volume 216

| case | fastest panel entry | cuFFT | speedup | floor (`baseline_gpu`) |
|---|---|---|---|---|
| **B = 1** (non-batched) | **L6_batchcoalesced 3.640 µs** (1.5%) | 10.077 µs | **2.77×** | 14.904 µs |
| **B = 4854** (L2, 32 MiB) | **L6_warpvolume 18.381 µs/call** = 3.79 ns/xform, 2212 GF/s (0.6%) | 51.424 µs | **2.80×** | 162.282 µs |
| **B = 310608** (HBM, 1 GiB, primary) | **L6_batchcoalesced 1540.14 µs/call** = 4.96 ns/xform, 1689 GF/s (0.1%) | 3523.48 µs | **2.29×** | — |

The two L=6 entries split the board: `L6_batchcoalesced` (shared-memory, batch-major
swizzle) takes B=1 by 47% and the primary point by 0.5%; `L6_warpvolume` (all-register,
zero shared memory, zero barriers) takes the L2 point by **30%**. At B_HBM the 0.5% gap
is inside `warpvolume`'s own 1.1% spread — call that cell a tie.

### L = 8 — volume 512

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L8_blockfused 3.184 µs** (10.5%) | 9.038 µs | **2.84×** | 17.469 µs |
| **B = 2048** (L2) | **L8_blockfused 13.238 µs/call** = 6.46 ns/xform, 3564 GF/s (0.6%) | 54.205 µs | **4.09×** | 174.668 µs |
| **B = 131072** (HBM, primary) | **L8_blockfused 1536.77 µs/call** = 11.7 ns/xform (0.1%) | 3697.92 µs | **2.41×** | — |

`L8_blockfused` sweeps all three cells; `L8_warpradix8` is 14% / 6% / 1.5% behind. The
4.09× at B_L2 is the **largest margin over cuFFT anywhere on the board**. See §3 for why
the B=1 ordering here is not actually resolved.

### L = 17 — volume 4913

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L17_raderfused 9.760 µs** (2.4%) | 13.474 µs | **1.38×** | 27.062 µs |
| **B = 213** (L2) | **L17_raderfused 31.690 µs/call** = 148.8 ns/xform (0.5%) | 67.314 µs | **2.12×** | 308.810 µs |
| **B = 13660** (HBM, primary) | **L17_dmma 1584.73 µs/call** = 116.0 ns/xform (0.1%) | 4759.68 µs | **3.00×** | — |

The ranking flips at the primary point: `L17_raderfused` wins B=1 (1.16×) and B_L2
(1.13×), `L17_dmma` wins B_HBM by 1.6% (1584.73 vs 1610.66). Both spreads there are
≤0.1%, so the flip is real, but 1.6% is a small thing to hang a preference on.

### L = 36 — volume 46656

| case | fastest panel entry | cuFFT | speedup | floor |
|---|---|---|---|---|
| **B = 1** | **L36_sharedtiled 10.069 µs** (0.1%) | 12.944 µs (5.5%) | **1.29×** | 45.837 µs |
| **B = 22** (L2) | **L36_globalpass 38.390 µs/call** = 1.745 µs/xform (0.5%) | 50.768 µs | **1.32×** | 513.162 µs |
| **B = 1438** (HBM, primary) | **L36_sharedtiled 2150.17 µs/call** = 1.495 µs/xform (0.2%) | 3399.04 µs | **1.58×** | — |

Also splits: `sharedtiled` takes B=1 (by 4%) and the primary point (by 2.6%),
`globalpass` takes the L2 point (by 6%).

### The rest of the board

| L | B=1 | B_L2 | B_HBM (primary) |
|---|---|---|---|
| **13** | L13_dmma **7.875** vs cuFFT 12.105 → **1.54×** | B=477: **28.334** vs 62.941 → **2.22×** | B=30549: **1563.31** vs 4742.78 → **3.03×** |
| **23** | L23_rader **10.556** vs 14.865 → **1.41×** | B=86: **51.842** vs 75.611 → **1.46×** | B=5515: **3252.05** (4.4%) vs 5066.75 → **1.56×** |
| **45** | L45_pfa **14.340** vs 18.010 → **1.26×** | B=11: **42.568** vs 64.153 → **1.51×** | B=736: **3355.34** vs 4748.54 → **1.42×** |
| **64** | L64_radix8 **22.581** vs 23.487 → **1.04×** | B=4: **48.441** vs 57.937 → **1.20×** | B=256: **2988.24** vs 3740.16 → **1.25×** |

**Board summary: 12 entries, 36 scored cells, cuFFT beaten in all 36.** Best margin
4.09× (L=8, B_L2); worst 1.04× (L=64, B=1). `baseline_gpu` is beaten by 2.8–17×
wherever it ran.

The clean structural signal across the whole board: **where a whole volume fits one
block's shared memory (L ≤ 17), the fused one-read-one-write kernel wins 2.2–3.0× at the
primary point and sits at 86–90% of DRAM peak moving exactly the compulsory bytes. Where
it does not (L = 23, 36, 45, 64), the same idea reaches only 1.25–1.58×, and every
entry there reports itself latency-bound with DRAM at 26–50%.** That is the round's
headline finding and it is what §5 and §6 turn on.

---

## 2. What changed since the previous round

**Nothing regressed, because there was nothing to regress from.** The previous GPU round
is `results/gpu_validate/` (2026-08-21, same node, same GPU): a two-cell pipeline
validation carrying **only** `cufft` and `baseline_gpu`, at L=8 B=64 and L=17 B=8. All
twelve `impl/` files were stubs. `results/local_smoke/` is a login-node run on `wombat`
and is not a measurement at all (CURATION.md is explicit that login-node timings do not
count).

Per geometry, then, the change is the same change: **first panel number on the board.**

* **L = 6** — first entries. Two structures, both correct, both ≥2.29× cuFFT everywhere.
* **L = 8** — first entries. The register-vs-shared A/B the pair was created to run
  completed; shared wins narrowly at every cell (§3 qualifies B=1).
* **L = 17** — first entries. The dense-vs-Rader question from LITERATURE §4.2 was
  settled here (§5).
* **L = 36** — first entries. Two two-pass structures with two *different* L2-residency
  mechanisms, which disagree (§6).
* **L = 13, 23, 45, 64** — first entries, one each, all ahead of cuFFT at all three cells.

`gpu_validate`'s two cells (B=64, B=8) are not in `cases.txt`, so even cuFFT is not
directly comparable across the two rounds; nothing about the reference point can be said
to have moved.

Three things that are *not* regressions but should be on the record as
**structure-dependent instability**, because a future round could easily read them as
noise and stop measuring the cases that expose them:

1. At **L = 6** and **L = 36** the winner changes with the case (`warpvolume` +30% at
   B_L2; `globalpass` +6% at B_L2). Neither pair has a uniformly better member.
2. At **L = 17** the winner changes at the *primary* point specifically.
3. At **L = 13** the L2-resident cell is **slower per transform than the DRAM-bound
   cell** — 59.5 ns at B=477 against 51.2 ns at B=30549, on data that is entirely
   L2-resident with a 7.2 TB/s roof available. `L8_blockfused`'s record names the same
   pathology at its own L2 point and diagnoses it (mixed read+write traffic through L2
   sustains only ~1.4 TB/s, i.e. the same effective rate as HBM) and fixes it with
   evict-first stores, worth −43% there. `L13_dmma` has not applied that fix. This is
   the clearest free win visible in the whole round.

---

## 3. Adversarial pass: failures, wrong answers, absences

**No entry failed correctness. No entry failed to build. No entry crashed or hung. No
entry is missing.** The evidence, and what I checked rather than assumed:

* `build_errors.txt` is **0 bytes**. `failures.txt` **does not exist**.
  `agents/exits.txt`: all 12 implementer agents `exit=0`.
* **76 scored cells × 3 independent processes = 228 timed runs**, and **76 of 76 cells
  have a correctness JSON with `ok=true`**. I re-read every `c_*.json`: zero failures,
  zero missing. Worst residual anywhere is `baseline_gpu` at L=17 (rel_l2 **8.41e-16**);
  worst panel entry is `L45_pfa` (**8.21e-16**); tolerance is 1e-12. Every panel entry
  is 4–6 orders of magnitude inside tolerance at every case.
* **The harness guards that matter were actually engaged**, not merely present:
  * `driver.cu`'s **anti-memoization poke test** — it perturbs the device input after
    timing and requires the whole output buffer to change, so a cached answer fails.
    Every entry passed.
  * `cudaGetLastError()` after warmup *and* after the checked run — a faulting kernel
    fails the entry rather than returning a fast zero buffer.
  * `cudaDeviceSynchronize()` before the stop event. This one is load-bearing this
    round: `L23_rader`, `L36_globalpass`, `L36_sharedtiled`, `L45_pfa` and
    `L64_radix8` all launch on `cudaStreamNonBlocking` streams, which do **not**
    synchronize with the NULL stream. Without that sync their work would have been
    timed as free.
  * The checked output comes from the **same plan and the same `execute`** as the timed
    loop, so a fast path and a correct path cannot diverge.
* **Every entry's `supports()` is honest.** `timing.err` contains 33 `does not support
  L=64` lines — that is the other eleven entries declining L=64 with exit code 3, by
  design. No entry claimed a geometry it had not written.
* **Provenance holds.** `impl` → `impl_1`; the newest scored source is
  `L8_warpradix8.cu` at 18:34, the scored binaries in `build/a80n1/bin` date from 18:37,
  and the sweep ran 18:36–18:46. Every scored binary was built from the sources now on
  disk.
* **Library-free rule holds.** Grepped all thirteen `impl_1/*.cu` for cuFFT/VkFFT/
  cuFFTDx/heFFTe/FFTW/MKL/ducc references: **none**. The only includes are
  `cuda_runtime.h`, `cuda_pipeline.h`, `cooperative_groups.h` and libc.

### `baseline_gpu` is absent from all eight B_HBM rows

Named so nobody reads the absence as a crash. `sweep.sh:66` skips `baseline_gpu` when
`L³·B > 40e6` ("too expensive to be informative"), logged in `timing.log`. At L=6
B=310608 the row would have taken ~1.6 s per execute against 1.54 ms for the winner. By
design, correctly.

### Three things I flag rather than accept

**(a) Setup is unscored, and the panel is spending real time in it.** Plan time is
excluded from the measured region and reported separately — that is the project's rule
and it is right. But three entries now run *measured autotuners* inside `create()`:
`L23_rader` **2.598 s** at B=5515, `L36_sharedtiled` 0.423 s at B=1438, `L8_blockfused`
0.304 s at B=131072, `L36_sharedtiled` 0.102 s at B=22. cuFFT's planner gets the same
exemption but uses 10–19 ms of it. The panel therefore has, in effect, an unbounded
plan-time budget while the reference uses ~1% of what the panel uses. Nothing here is
cheating — the tuners race configurations on scratch buffers, and `L8_warpradix8`'s
record documents restricting its tune space to bit-identical variants precisely so the
scored output cannot depend on the pick. But r2 should either cap plan time or publish a
second column with setup amortized over the batch, before the number grows again.

**(b) `L36_sharedtiled`'s B=1 win rests on a CUDA graph keyed on the `(in,out)` pointer
pair.** `create()` races two plain launches against a cooperative fused kernel against a
captured graph and picks the graph (12.8 vs 14.5 vs 15.9 µs in its own probe). The driver
calls `execute` with the same two pointers forever, so the capture cost is amortized to
zero and the replay is measured at 10.069 µs with a 0.1% spread. The record is honest
(it recaptures if the pointers change) and the win is real for this call pattern — but it
is **harness-shaped**: an application rotating buffers pays a recapture per call.
`L36_globalpass`'s 10.490 µs is the launch-per-call number, 4% behind. Read the L=36 B=1
headline with that attached.

**(c) The L = 8 B=1 ranking is not resolved and should not be quoted as settled.**
`L8_blockfused` 3.184 µs at **10.5%** across-process spread against `L8_warpradix8`
3.632 µs at 6.6%. The 14% gap is inside the combined spread of the process minima; note
also that cuFFT itself shows 5.8% there and L=36's cuFFT shows 5.5%, so B=1 cells on this
un-clock-locked GPU are simply the noisy ones. Both entries beat cuFFT by >2.4×, so the
headline stands as written, but "the shared-memory structure is faster at B=1" is not a
claim gpu_r1 supports. It is a claim r2 can settle with more samples, and it matters,
because that cell is the one place the register-vs-shared experiment is not yet decided.

**No entry needs to be discarded for a fast wrong answer.** There isn't one.

---

## 4. Claimed versus measured

The brief's premise — implementers on Sapphire Rapids, monitor on Cascade Lake, MKL
spanning 2.9× between them — **does not apply to this round**, and I will not attribute
anything to it. All twelve implementers developed via `tryout.sh` on leased
A100-SXM4-40GB GPUs of the *same* node the scoring ran on. Same part, same driver, same
CUDA. The comparison is therefore unusually tight, and where it is not tight the cause is
identifiable.

| entry | cell | claimed | scored | Δ |
|---|---|---|---|---|
| L6_batchcoalesced | B=310608 | 1540.5 µs | 1540.14 | −0.02% |
| L6_batchcoalesced | B=1 | 3.64 µs | 3.640 | 0.0% |
| L6_warpvolume | B=310608 | 1563.9 µs | 1548.25 | −1.0% |
| L6_warpvolume | B=4854 | 18.59 µs | 18.381 | −1.1% |
| L8_blockfused | B=131072 | 1534.0 µs | 1536.77 | +0.2% |
| L8_blockfused | B=2048 | 13.44 µs | 13.238 | −1.5% |
| **L8_blockfused** | **B=1** | **4.26 µs** | **3.184** | **−25% (1.34× faster than claimed)** |
| L8_warpradix8 | B=131072 | 1558.5 µs | 1560.45 | +0.1% |
| L8_warpradix8 | B=1 | 3.74–3.90 µs | 3.632 | −3% |
| L13_dmma | B=30549 | 1564.0 µs | 1563.31 | −0.04% |
| L17_raderfused | B≈13660 | 1611 µs | 1610.66 | −0.02% |
| L17_dmma | B=13660 | 1582.4 µs | 1584.73 | +0.15% |
| L23_rader | B=5515 | 3348 µs | 3252.05 | −2.9% |
| L36_sharedtiled | B=1438 | 2116.6 µs | 2150.17 | +1.6% |
| L36_globalpass | B≈1404→1438 | 1.550 µs/vol | 1.534 µs/vol | −1.0% |
| L45_pfa | B=736 | 3360.9 µs | 3355.34 | −0.17% |
| L64_radix8 | B=256 | 2995 µs | 2988.24 | −0.23% |

**Eleven of twelve entries land within 3% of their own claim at the primary point, and
six of them within 0.25%.** For a first round with no prior calibration, that is the
methodology working: the driver's `--min-sample-ms 20` default, the 3-process
minimum-of-minima, and the implementers' own habit of quoting min-of-samples all agree.

**The one real divergence is `L8_blockfused` at B=1, and it is 34% faster than claimed,
not slower.** Its own record explains it without being asked: *"3.41 µs vs 3.26 µs for
the staging-free single kernel (min under ramping clocks; tryout's pinned-clock number is
4.26 µs)"*. This is the **A100 boost-clock ramp**, documented at the top of `driver.cu`
with the measurement that produced the rule: at L=8 B=64, cuFFT reads 22.4 µs at
`--min-sample-ms 3`, 20.9 µs at 10 ms, and 12.3 µs at 20 ms — a 1.7× cliff, not a
gradual ramp, because below ~20 ms of continuous work the GPU never reaches 1410 MHz.
A short dev-loop sample measures a downclocked GPU. So the GPU analogue of the brief's
machine-difference caveat exists and **cuts the opposite way to the CPU one: claims made
on short samples understate the hardware, and the scored number is the faster one.** The
same effect is visible in the smaller negative deltas (`L23_rader` −2.9%, `L8_warpradix8`
−3%, `L6_warpvolume` −1%) and in the B=1 spreads in §3(c). No entry overclaimed.

Two secondary observations on the records themselves:

* **The `PANEL_BRIEF` HBM figure is wrong and four entries independently corrected it.**
  `L13_dmma` derived it first (1215 MHz memory clock × 5120 bit × 2 / 8 = **1555 GB/s**,
  cross-checked against ncu's percent-of-peak), and `L8_blockfused`, `L8_warpradix8` and
  `L6_batchcoalesced` adopted it with attribution. The brief's "~2.0 TB/s" is the 80 GB
  part. Every roofline percentage in this round's records (86–90% of peak) is against the
  corrected figure and is right. **Fix `PANEL_BRIEF.md` before r2** so a fifth entry does
  not re-derive it.
* One inconsistency to note: `L23_rader`'s arithmetic section still reasons against a
  2 TB/s floor ("194 ns/volume at peak FP64 issue against a 195 ns single-pass HBM floor
  at 2 TB/s — exactly balanced"). Against the corrected 1555 GB/s the floor is ~251 ns,
  so folded-dense-on-CUDA-cores has more headroom than that sentence claims, not less.
  It does not change the entry's standing or its (correct) decision to skip DMMA.

---

## 5. Which LITERATURE §4 open question moved

### Moved most: §4.3 — "Is axis fusion worth 3× or 3%?"

§4.3 records the CPU panel's partial answer (single-digit percent, sometimes negative)
and then says exactly what was left untested:

> every panel experiment fused across an **L1↔L2** boundary, where the measured bandwidth
> gap is 2.6×. The untested case is **L2↔DRAM**, where the gap is 7× … **tile the batch
> so a tile fits L2, then run all three axes inside the tile.** That is not the same
> experiment the panel ran, and it is the largest untried structural move on the board.

**gpu_r1 ran that experiment**, with shared memory as the fast level and HBM as the slow
one — a bandwidth gap not of 2.6× but of roughly 10–20×. The answer is unambiguous and it
is "3×", not "3%":

* At **L ≤ 17**, where a volume fits one block's shared memory, the fused
  one-read-one-write kernel wins **2.29× (L=6), 2.41× (L=8), 3.03× (L=13), 3.00× (L=17)**
  over cuFFT at the primary point — and ncu confirms the mechanism rather than merely
  the outcome: `dram__bytes` equals the compulsory read-once/write-once minimum
  (2.13 GB) in *four independent entries*, at **89% (L6_batchcoalesced), 90.2%
  (L8_blockfused), 88% (L13_dmma), 87% (L17_dmma)** of the corrected 1555 GB/s DRAM
  peak. cuFFT sits at 450–650 GB/s on the same cells, i.e. it is spending ~3 global
  passes where these kernels spend 1. The 3× *is* the avoided pass count.
* At **L ≥ 23**, where a volume does not fit (190 KiB at L=23 against a 163 KiB shared
  ceiling; 1.42 MB at L=45), the same construction degrades to two passes plus an
  L2-resident intermediate and reaches only **1.25–1.58×**. And critically, all four of
  those entries report themselves **latency-bound, not traffic-bound**: L23_rader
  combined DRAM ~50%, L45_pfa DRAM 26–36% with every pipe under 56%, L64_radix8 ~32
  warps/SM issue-bound. Three of them (L23, L45, L64) implemented L2 chunking, *proved
  by ncu that it cuts HBM traffic as designed*, and measured **zero or near-zero
  wall-clock gain** — L45_pfa is blunt about it: "wall-clock is currently flat vs
  unchunked (3362 vs 3361 µs) because the kernels are latency-bound, not DRAM-bound".

**So Tolmachev's rule as §4.3 states it survives, now with GPU numbers: the payoff is the
number of avoided passes times the bandwidth gap between the two levels involved —
nothing more.** And §07 gap 7's prior ("TurboFNO's 3–5% is probably the better prior") is
correct for the CPU L1↔L2 regime and wrong by two orders of magnitude for this one. The
discriminator is not the machine and not the algorithm: it is whether the level you fuse
into is a real bandwidth step below the level you avoided. §4.3's own size-dependence
prediction — small at L=6/8 where everything is L1-resident, large at L=36 with a large
batch where it saves DRAM traffic — is confirmed with the axis relabelled: on a GPU it is
**capacity-dependent**, and the threshold is exactly "does the volume fit shared memory".
That threshold sits between L=17 (78.6 KiB, fits) and L=23 (190 KiB, does not), and it is
visible in the leaderboard as a cliff from 3.0× to 1.56×.

A corollary worth carrying forward, because three entries measured it independently and
it contradicts the naive reading of the fusion argument: **do not fuse the global load
into the first axis pass.** Fusing z into the load saves a shared round trip and a
barrier, and it lost every time — L17_raderfused **2358 vs 1657 µs (+42%)**, L13_dmma
**+8%**, L8_blockfused **1612 vs 1534 (+5%)**, L6_batchcoalesced **1582 vs 1540
(+2.5%)**, and on the store side L8_warpradix8 **2279 vs 1557 (+47%)**. The reason is
consistent across all five: a per-thread-contiguous strided global access uses half of
each 32-byte sector and relies on L1 to merge the halves, and with the shared carveout
maxed there is only ~28 KB of L1 left. **Flat instruction-level coalescing beats
barrier-count and pass-count both.** Five measurements, one direction; nobody should
re-derive this.

### Moved decisively: §4.2 — "L = 17: dense-symmetric, Rader, or Winograd?"

§4.2 asks (a) which of dense-symmetric and Rader-17 wins on this hardware, and (b)
whether the symmetric/antisymmetric convolution split adds anything on top. Both are now
answered on GPU, by two entries that reached the same place from different directions.

**(a) The conjugate-folded dense form wins, and Rader-proper lost on a resource the
corpus was not counting.** `L17_raderfused` — the entry *named* for Rader — analysed it
and dropped it without building: two FFT16s plus 16 pointwise complex multiplies is
**428 flops/line against its own module's 496**, so Rader is genuinely cheaper in flops,
and it was still rejected because a per-thread FFT16 needs **160+ live registers against
the 102 that two-blocks-per-SM allows**. §02's verdict ("Rader is not the lever at L=17")
survives; §01's lower flop counts were real and irrelevant. The corpus's framing —
multiplications are not the scarce resource — holds with the scarce resource renamed:
on a GPU it is **registers and shared-memory motion**.

**(b) The symmetric/antisymmetric split adds arithmetic and adds no time.** This is the
sharper result, because it was measured twice. `L17_raderfused` *ships* the cyclic-4 ⊕
negacyclic-4 split of the cosine circulant plus the sign-decorated negacyclic-8 sine
treatment (ported verbatim from the CPU `L17_winograd`/`L17_matrixsimd` records), and it
is what takes its line module from 608 to **496 flops/line**. `L17_dmma` then ported the
*same* nested split into its own kernel and A/B'd it: **−12% FP64 ops, −25% FMAs, and
exactly 0% time** (35.97 vs 35.78 µs at B=213; 1586 vs 1587 µs at B=13660). And the two
entries finish **within 1.6% of each other** at every scored cell. So the split is
correct, is cheaper on paper, and is invisible — the same lesson the CPU r2 round learned
(12% ops → 1.4% time), reproduced on entirely different hardware.

§4.2's open sub-question (c) — the exact op count for a full 17-point Winograd module —
remains unanswerable and was correctly not pursued.

### Also moved: the GPU counterpart of §4.1 — registers versus spill

§4.1 asks how much spill traffic a batch-vectorised codelet generates and whether it
costs more than the shuffles it avoids, and leans toward taking the spills ("almost
certainly worth taking, but untested"). The CPU question is untouched by this round. Its
**GPU counterpart got five independent answers, all in the same direction, and the
direction is "never take the spills"**:

| entry | what was forced | result |
|---|---|---|
| L6_batchcoalesced | `launch_bounds(288,6)` → 32 regs | 60 B spill, 1799 vs 1543 µs (**+17%**) |
| L6_warpvolume | `launch_bounds(64,8)` → 128 regs | 40 B spill, 1970 vs 1554 µs (**+27%**) |
| L17_dmma | MINB=3 → ≤68 regs | 816 B spill, 60.0 vs 35.8 µs (**+68%**) |
| L23_rader | `launch_bounds(256,3)` → 96 regs | 644 B spill, 4985 vs 3546 µs (**+41%**) |
| L45_pfa | T=128→160 → 81-reg cap | 3600 vs 3364 µs (**+7%**) |

**Five entries, five geometries, one conclusion: at these sizes never trade spills for
occupancy.** The corresponding positive result is `L45_pfa`'s: the way to buy occupancy is
to make the *unit of work* smaller, not to cap the register file. Its thread-per-line PFA
compiled to >204 registers → 4 blocks/SM → 9.9% occupancy → 4712 µs; going
**unit-parallel** (one thread = one DFT5 or one DFT9, not one 45-point line) dropped it to
96 regs and **3752 µs immediately**, the single largest structural win in that entry.

### Not moved

* **§4.4 (split vs interleaved complex)** — untouched. Every entry used the driver's
  interleaved `double2` layout; nobody built a split-complex variant. Note `L17_dmma`
  did A/B SoA-vs-AoS *in shared memory* and found a 2.6% wash (SoA 2377 vs AoS 2440 µs)
  while AoS halves the LSU instruction count — suggestive, but that is a shared-memory
  layout question, not §4.4's global-layout question, and the driver's contract fixes the
  global layout anyway.
* **§4.5 (padding at L=8)** — the *mechanism* is confirmed but not A/B'd. `L8_blockfused`
  padded the z-row stride 8→9 exactly as prescribed and measured 1.5% residual conflicts;
  `L36_*` padded 36→37; `L13/L17/L23/L45` exploited the odd-stride gift and padded
  nothing. Nobody measured the *unpadded* L=8 kernel, so the size of the avoided
  catastrophe is still unquantified on this hardware. A one-line A/B for r2.
* **§4.6 (model versus search)** — partially. Five entries shipped plan-time measured
  autotuners and three found the pick genuinely mattered, which supports §06's "search
  it" over §01's "you should not need a search phase". But nobody searched *schedules* —
  the searches were over block shape, chunk size, stream count and cache hints.
* **§4.7 (vector-radix)** — untried, correctly. `L36_globalpass` notes DMMA at L=36 is
  pre-refuted by the corpus and did not spend time rediscovering it; the same discipline
  applied to vector-radix.
* A note against **§4.8 item 1** (no published single-node benchmark of small fixed 3D
  cubes): this round does not close it — it is a GPU round, and the item is about
  single-threaded CPU — but it produces 24 such cells at eight edge lengths across three
  working-set regimes with a library reference, which is adjacent and citable.

### The round's own new question, for the record

**Why is the L2-resident cell so often the worst one?** L=13 is *slower per transform*
at B=477 than at B=30549. L=8 was too until evict-first stores fixed it (23.5 → 13.24 µs,
−43%). L=17's B=213 sits at 149.6 ns against a ~60–70 ns issue floor. The emerging answer
from `L8_blockfused`'s record — **mixed read+write traffic through L2 sustains only
~1.4 TB/s, the same effective rate as HBM, so an "L2-resident" problem is not actually
faster unless you get the write stream out of L2** — is the most transferable finding of
the round and it is currently written down in exactly one place. Three entries'
cache-hint results (L8's `__stcs` +43%, L36_sharedtiled's `__stcs` +2.5%, L13's `__ldcs`
−5%) are all consistent with it. Promote it to a corpus-level note.

---

## 6. The single highest-value thing the next round should attack

One per geometry. The B_HBM cells at L ≤ 17 are **done** — four entries at 86–90% of DRAM
peak moving provably minimum bytes, with every lever tried moving them ≤1%. Do not spend
r2 there; three separate records say so in their own words and they are right.

* **L = 6 — the L2 point, and only the L2 point.** B_HBM is at ~89% of peak with zero
  wasted traffic (≤6% left, latency-shaped). B=1 is pure launch path (~3.6 µs for 4.4 ns
  of work). At **B=4854 nothing is saturated** — DRAM 46%, L2 sectors 29%, SM 26%, warps
  active 57%, and the L2 roof is 5× away; `warpvolume`'s own writeback floor estimate is
  ~13 µs against 18.38 measured, so **~1.4× is on the table there and nowhere else at
  this geometry**. Two named mechanisms, both about resident warps: `warpvolume`'s triad
  decomposition (27 lanes/volume, 8 points/lane, ~64 regs → ~2.5× occupancy at 12 shuffle
  instr/point instead of 6), or `batchcoalesced`'s warp-autonomous variant that removes
  all three block barriers. Build one, not both.

* **L = 8 — resolve B=1, then take the 13% at B_L2.** B_HBM is at 90.2% of DRAM peak at
  exact-minimum bytes; that cell is closed and its own record forbids another round on
  it. B_L2 sits at 13.238 µs against an ~11.6 µs write-stream floor — a bounded ~13% via
  the warp-per-volume variant `L8_blockfused` already scopes. But the *higher-value* item
  is measurement, not optimization: **B=1 is 14% apart at 10.5% spread**, and that single
  cell is the only place the register-vs-shared experiment this pair was built to run
  remains undecided. More samples, or a longer minimum sample, settles it cheaply and
  tells r2 whether to keep carrying two L=8 structures.

* **L = 17 — the B=213 cell, via in-block load/compute overlap.** Both entries are ~87%
  of DRAM peak at B_HBM and within 1.6% of each other everywhere, and **both records
  independently name B=213 as the improvable cell**: 148.8/161.8 ns against a ~60–70 ns
  issue-floor estimate, i.e. **~2×, the largest single headroom on the L ≤ 17 half of the
  board**. The cause is agreed and structural: a resident 78.6 KiB volume forces 2
  blocks/SM (20 warps, ~30% achieved occupancy), so the z and y passes are DRAM-idle and
  the only overlap is two co-resident blocks drifting out of phase. The fix must overlap
  the load stream with compute *inside* a block **while keeping both blocks** — and
  `L17_dmma`/`L17_raderfused` have already killed the naive versions (whole-volume
  cp.async double-buffer with persistent blocks: 2820 µs spilling at 2 blocks/SM, 1735 µs
  at 1 block/SM against 1611 plain). The remaining candidate is `L17_raderfused`'s
  **plane-granular mbarrier pipeline**: z and y are plane-local, so plane *p*'s compute
  can start when plane *p*'s cp.async group lands. Nontrivial (a plane's 17 lines straddle
  warp boundaries) and it is the round's job at this geometry.

* **L = 36 — settle the L2-residency mechanism, with one A/B.** Both kernels of both
  entries run at 84–87% of DRAM peak, and `L36_globalpass` proved the arithmetic is
  entirely free (deleting all three `fft36_lines` calls: 669 vs 665.6 µs). So the kernels
  are done and **all the remaining loss is the intermediate's traffic**: `sharedtiled`
  measures effective HBM traffic of ~1.9 MB/volume against a 1.49 MB chunked ideal
  (1.472 vs ~1.15 µs/vol — **~28%**), and `globalpass` measures its chunked pipeline
  running DRAM at ~1.0 TB/s while its own unchunked kernels sustain ~1.35 TB/s. The two
  records **disagree about the mechanism**, which is exactly why this is the highest-value
  L=36 item: `globalpass` got **1.50×** from plain `__ldcs`/`__stcs` ("the single most
  consequential three characters in the file") and *lost* with `accessPolicyWindow`
  (705.8 vs 665.6 µs); `sharedtiled` lost with an `L2::evict_last` PTX hint (2623 vs
  2118 µs) and names a **real persisting-L2 carve** (`cudaLimitPersistingL2CacheSize` plus
  a sliding `accessPolicyWindow` over the chunk intermediate) as the untried fix that the
  `evict_last` experiment was missing. Run that one A/B against the evict-first hints, at
  both entries' chunk sizes. Worth ~28% here — and it is the same question L=45 and L=64
  are stuck on, so the answer pays three times.

* **L = 13 — apply `L8_blockfused`'s evict-first-store finding.** The L2 cell is *slower
  per transform than the HBM cell* (59.5 vs 51.2 ns). L=8 had the identical pathology and
  fixed it for −43% with `__stcs` on the final store. `L13_dmma` tried `__ldcs` on the
  *loads* (−5%, correctly rejected) and never tried the store side. Cheapest win visible
  in the round; do it before anything structural.

* **L = 23 — the concurrency wall, nothing else.** 128 regs/thread caps 512 threads/SM,
  both kernels idle ~90% of issue slots at 2.5–3 active warps/scheduler, combined DRAM
  ~50%, and the tuner's chunk sweep is **flat** because nothing is bandwidth-bound. The
  entry is 1.7× off the plain two-pass DRAM floor and 2.6× off the L2-resident floor.
  The one named escape is **split-k kernel B**: two threads per line, each folding half
  the *j*'s (44 regs), combining P/Q partials with 4 `__shfl_xor` per k-pair — ~64 regs →
  1024 threads/SM at +16% instructions. Forcing fewer registers directly is already
  measured dead (+41%).

* **L = 45 — occupancy first; the chunking is already correct and waiting.** Latency-bound
  at 26–30% achieved occupancy with every pipe under 56% and ~1.8× headroom to the traffic
  roofs. ncu already proves the L2 chunking works (20+22 MB DRAM per 12-volume chunk
  against 35 MB compulsory) and it buys **zero** wall-clock. So do not touch the traffic
  structure: attack warp supply with the **3-kernel split** (z rows-tiled, y column-tiled,
  x) so every kernel gets 6–7 blocks/SM, at the cost of +2 L2 accesses/point that the
  chunking absorbs. The chunking starts paying the moment occupancy moves.

* **L = 64 — kernel_x.** Weakest margin on the board (1.04× at B=1, 1.25× at B=256) and
  the largest self-reported headroom (~2×: composite floor ~1.4–1.6 ms against 3.0 ms
  measured). The record localizes the loss precisely: **kernel_x is 12.6% compute, 31%
  DRAM, ~59% memory-pipe SOL**, losing L1 wavefronts to scattered 16-byte accesses.
  Two candidates already scoped: a per-warp 8×9-padded shared scratch replacing the
  24-SHFL.32-pair shuffle transpose, or a fully shared-tiled kernel_x with 128-byte
  coalescing on both sides. Both perturb SM co-residency with kernel_yz, so measure.

**Cross-cutting, one item:** the round produced two findings that every geometry needs and
only one record each currently holds — *mixed r/w L2 traffic sustains only ~1.4 TB/s, so
evict-first stores are what make an L2-resident batch actually fast* (L8_blockfused), and
*flat instruction-level coalescing beats both barrier count and pass count, measured five
times* (§5). Fold both into the brief and the corpus, and fix the brief's 2.0 TB/s HBM
figure to 1555 GB/s while there.

---

## 7. What to keep

Applying `docs/CURATION.md`'s four grounds, in order.

**Ground 1 — the fastest correct entry per geometry, always. Eight entries, no
discretion:** `L6_batchcoalesced` (B=1 and the primary point), `L8_blockfused` (all
three cells), `L13_dmma`, `L17_dmma` (the primary point), `L23_rader`,
`L36_sharedtiled` (B=1 and the primary point), `L45_pfa`, `L64_radix8`.

**Ground 2 — a structurally different runner-up when it is close.** Four geometries ran
two entries each, and in every case the pair is a *designed* structural A/B whose loser
is inside CURATION.md's ~20% band or wins a cell outright:

* **`L6_warpvolume`** — all-register, zero shared memory, zero barriers, the exact
  opposite of the promoted winner's staged-shared structure, and it **wins the L2 cell by
  30%** while tying the primary point within its own spread. Also carries the round's
  best-documented negative result on index-map choice (Good–Thomas per axis lost at
  1117 GB/s against 1373 for DIF-z/DIT-yx: *on a GPU, choose the factorization variant
  per axis by its memory pattern, not its flop count*).
* **`L8_warpradix8`** — the register/shuffle half of an explicitly constructed
  register-vs-shared experiment, 1.5% behind at the primary point and undecided at B=1
  (§3c). Its record contains the head-to-head table and the reason the register structure
  does not convert (occupancy: ~40-reg threads allow 32 warps/SM against 19 at 106 regs),
  plus the autotune-repeatability lesson that any future entry with a tuner needs:
  *only tune over knobs that cannot change a single bit of the output.* This is precisely
  CURATION.md's case — "the next panel needs to see the alternative actually written
  down, not described."
* **`L17_raderfused`** — wins two of three cells including B=1 by 16%, differs from the
  promoted winner in its line module (the cyclic-4 ⊕ negacyclic-4 / negacyclic-8 split)
  and in its small-batch path, and together with `L17_dmma` constitutes the whole answer
  to LITERATURE §4.2. Keeping only one of the pair would throw away half the settled
  question.
* **`L36_globalpass`** — **wins the L2 cell by 6%**, and its L2-residency mechanism
  *disagrees* with the promoted winner's (evict-first hints worth 1.50× and
  `accessPolicyWindow` losing, against `sharedtiled`'s `evict_last` loss and its
  persisting-carve proposal). That disagreement is the r2 agenda item at L=36 (§6); both
  sides of it must be on the shelf. It also holds the round's cleanest
  arithmetic-is-free measurement (the copy-only test: 669 vs 665.6 µs).

**Ground 3 — instructive failures.** There are none to promote, because **no entry
lost**: all twelve beat cuFFT at all 36 of their cells and all passed correctness. The
instructive failures of this round are *inside* the records — the negative-results lists,
with the number that killed each one (§5's five-way spill result and five-way
load-fusion result are both assembled from them). CURATION.md's point that "the record is
what makes the code useful later" is what makes those survive: they are preserved by
promoting the entries together with their strategy records, which `promote.sh` does.

**Ground 4 — anything that beat a library baseline.** This admits all twelve and
therefore filters nothing. Stating it plainly rather than pretending it selected
anything.

**So: promote all twelve** — eight by ground 1, four by ground 2, none by grounds 3 or 4
that were not already in. That is not a monitor declining to judge; it is what a first
round on new hardware where nothing failed and every geometry's second entry won a cell
or held the opposite structure within 2% actually looks like. There is no third entry
anywhere to reject and no near-duplicate to cut: the four pairs differ in structure, not
in tuning, and their negative-result lists barely overlap.

For the record, since the next monitor will need it: **the first candidates for retirement
in r2 are whichever member of each pair loses all three cells.** Right now that is
nobody. `L8_warpradix8` is closest — it loses all three, but by 14% / 6% / 1.5% with the
14% inside the spread — so if r2 resolves L=8 B=1 in `L8_blockfused`'s favour and the
warp structure gains nothing at B_L2, that is the one to drop.

Promotion command:

```bash
cd bench/gpu
./promote.sh gpu_r1 L6_batchcoalesced L6_warpvolume L8_blockfused L8_warpradix8 \
    L13_dmma L17_dmma L17_raderfused L23_rader L36_sharedtiled L36_globalpass \
    L45_pfa L64_radix8
git add exemplars/gpu_r1 strategies results/gpu_r1 impl_1 && git commit
```
