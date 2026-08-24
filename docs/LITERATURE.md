# LITERATURE.md — corpus index and cross-cutting synthesis

**Audience.** The implementer panel about to write self-contained C for 3D complex-double
forward FFTs on cubes **L = 6, 8, 17, 36**, many volumes batched, single-threaded x86-64,
AVX2 baseline with AVX-512 on some nodes, no library calls inside the transform.

**What this file is.** An index to the eight literature sections in
`docs/literature/`, and — more usefully — the synthesis across them: what they agree
on, where they contradict each other, and what to build first. Every number here is
traceable to a section; the sections carry the URLs and the fetch status.

**Read this file, then read §08 §0 and §3.6 below, then §04 and §01 in full before writing a
line of code.** §04 and §01 carry the layout decision and the register budget, which between them
determine everything else. §08 carries the hardware you are actually scored on, and it corrects
two things the earlier sections got wrong for this specific part: the AVX-512 frequency penalty
(there is none at one active core on a Gold 5218) and the value of L3 blocking (almost none).

---

## 1. Map of the corpus

| § | File | Covers |
|---|---|---|
| **01** | `01-small-n-codelets-and-codegen.md` | Straight-line codelets and the generators that emit them: FFTW `genfft` (Frigo PLDI'99), SPIRAL, Blake's SFFT/FFTS, VkFFT's radix generator. Published op counts for small-*n* modules (Burrus Tables 7.1/7.2/9.1/10.1). **Measured peak register liveness of every FFTW codelet.** The definitive register budget. |
| **02** | `02-prime-and-awkward-lengths.md` | The L=17 problem: Rader, Winograd/WFTA, Bluestein, and Good–Thomas/PFA for L=6 and L=36. Exact op counts from FFTW's own plan accounting. **Measured FFTW and MKL timings on this node for L = 6, 8, 16, 17, 18, 36** — the only real performance data in the corpus. |
| **03** | `03-multidimensional-and-vector-radix.md` | Genuinely multidimensional algorithms: vector-radix, Nussbaumer–Quandalle polynomial transforms, Bailey four/six-step. Duhamel & Vetterli's op-count tables at small N. Concludes: don't. Establishes PFA-per-axis as the one multidimensional idea that pays. |
| **04** | `04-simd-layout-and-registers.md` | Interleaved vs split complex, which dimension to vectorise over, shuffle-port economics, FMA op counts and error bounds, alignment, cache-set conflicts and padding, AVX2-vs-AVX-512 port schemes and frequency licences. Intel Optimization Manual + Agner Fog throughout. |
| **05** | `05-memory-hierarchy-and-blocking.md` | I/O complexity (Hong–Kung, cache-oblivious), four/six-step crossovers with Bailey's own measured numbers, transposes, working-set arithmetic for all four sizes at all batch widths, TLB/huge pages, non-temporal stores. |
| **06** | `06-autotuning-and-search.md` | FFTW's planner, SPIRAL's formula search, ATLAS and the model-driven critique, learned autotuning, and — the load-bearing part — **how to benchmark defensibly** (benchFFT protocol, Intel RDTSC method, minimum-vs-mean statistics). |
| **07** | `07-register-level-fusion-and-accelerators.md` | What the accelerator world knows: cuFFTDx as a device function, VkFFT's runtime codegen, fused FFT pipelines and their honest accounting. Register-file-vs-L1 arithmetic. Twiddle precision strategy. Complex-multiply error bounds. |
| **08** | `08-recent-advances.md` | **Recent work, 2015–2026, and the hardware we are actually scored on.** Published single-core bandwidth, latency and cache behaviour of Cascade Lake-SP; Intel's own AVX-512 turbo table for the Xeon Gold 5218; non-temporal stores and Intel's Skylake-Server warning about them; measured line-fill-buffer count and prefetch-instruction choice; 4 KiB store-to-load aliasing; cache-blocking across the batch into L2; the one published measurement of small odd complex-double 3D cubes on one core (2–3× over MKL and FFTW, won by *layout*, not by the kernel); the μ-mode/batched-GEMM formulation; regime-dependent kernel selection mechanisms from the small-GEMM literature; Exo 2, FFTc/MLIR, ducc0's and VkFFT's design decisions read from the source; the 3-FMA lifting rotation. **Corrects §04 §8.2 and closes §4.4 below.** |
| **09** | `09-gpu-small-batched-a100.md` | **The GPU phase: batched complex-double `L^3` FFTs on one A100 (sm_80).** Measured device budgets and the capacity arithmetic for `L = 6, 8, 13, 17, 23, 36, 45, 64` against registers/shared/L2/HBM; the roofline (every geometry is bandwidth-bound by 2.2–5.2×) and the achieved-bandwidth evidence; cuFFT and VkFFT as baselines, read from NVIDIA's docs and VkFFT's shipped source; cuFFTDx as design evidence; **FP64 tensor cores (DMMA) for a dense per-axis DFT — supported for `L ≤ 23`, refuted above**; coalescing, 16-byte shared-memory bank rules, `cp.async`, occupancy and launch overhead; a recommended GPU benchmark contract and a per-geometry opening strategy. **Corrects §07 §1.2, §1.1, §4.1 and §08 §5.7.** |
| **10** | `10-icelake-under-glass.md` | **Forensics from the grading tier itself** (seven independent graded attempts on the grading tier's KVM/Firecracker Ice Lake-SP slice, main-v4, 2026-08-22). No PMU; two 512-bit FMA pipes but a virtualized feed cap of ~2.1 vector uops/cycle (memory-operand FMA collapses to ~1-1.25/cyc); register-resident phase-split codelets vs embedded-broadcast as the two winning responses; denormal-assist traps; 4K-aliasing epidemic; GCC 13.2 spill cures; the z-split L=64 layout (7x working-set compression); interleaved best-of-N as the only trustworthy timing protocol there. Full sources of all seven graded attempts are in `ext/reference/fft_v4_solutions/`. **Where it conflicts with 08/09 on Ice Lake behaviour, this section wins: it was measured on the grading machine.** |

Supporting material already in the tree: `docs/SURVEY.md` (installed libraries and hardware),
`docs/TEXTBOOK_FFT.md` (the Python reference implementation to validate against),
`docs/literature/tools/fftw_codelet_liveness.py` (the liveness measurement in §01).

---

## 2. The five things all seven sections agree on

Before the per-size table, the consensus. These are not contested anywhere in the corpus.

**§08 confirms all five and adds numbers to three of them.** #2 (vectorise across the batch) now has
a measured x86 figure: split `DFT_n ⊗ I_ν` kernels 1.3–2× faster than FFTW's in double precision,
plus a permute-versus-gather measurement of **IPC 0.13 → 2.59, 9.25× speed** (§08 §5.4) — which
between them close §4.4 below. #3 (instructions, not flops) gains a 2026 confirmation on Sapphire
Rapids: radix-4 beats radix-8 at every size despite radix-8 needing 20 % fewer flops (§08 §2.1). #5
(compile-time twiddles) gains a primary source for the 10–15 % positive-constant result and a proof
that the FMA complex multiply's 2u error bound is *sharp* and cannot be improved by compensation
(§08 §6.1, §6.7). **§08 also proposes a sixth consensus item that the earlier sections could not
have reached: on this hardware, the schedule around the codelets is worth more than the codelets.**
The one published measurement of our exact regime had 1D kernels slower than MKL's and a 3D kernel
2–3× faster, attributed entirely to layout and loop merging (§08 §2.2).

1. **All four line lengths are codelet territory. No recursion, no planner, no blocking
   inside the line transform.** 6, 8, 17, 36 are all ≤ 64, which is FFTW's own codelet
   ceiling (§01 §1.6, §06 §6.1), inside SPIRAL's 4ν–32ν basic-block rule (§05 §4), and three
   to nine binary orders of magnitude below Bailey's four-step crossover of n ≈ 4096
   (§03 §4.4, §05 §2.2 — where four-step is measured *25–28% slower* below n ≈ 2048).
2. **Vectorise across the batch, not within the transform.** `DFT_n ⊗ I_ν` is a vector
   terminal: every scalar operation becomes exactly one vector instruction with zero data
   reorganisation, for *any* algorithm you choose inside (§04 §3.1, Franchetti & Püschel).
   This is why L=17 can be fixed at all (§02 §6.1), and it is what FFTW itself does at every
   size where it has a vector codelet.
3. **Instructions, not flops, are the currency.** On Haswell-and-later a 256-bit FMA and a
   256-bit add have the same throughput on the same ports, so any transformation trading one
   multiply for two adds is a net loss (§02 §2.7, §03 §3.3, §04 §5.1, §06 §1). This kills
   the entire minimum-multiplication programme — WFTA, Winograd nesting, polynomial
   transforms — and FFTW says so in print twice.
4. **Arithmetic cleverness is worth ~25%; implementation quality is worth 5–40×.** Forty
   years of theory moved the power-of-two count from 5n log₂n to 34/9 n log₂n (§01 §2.3,
   §03 §6.2, §06 §1). Meanwhile SPIRAL measured a **2× runtime spread among 10 000
   arithmetically-equivalent formulas on a transform small enough that "no cache problems
   arise"**, with under 10% arithmetic spread — and "the formulas with lowest arithmetic cost
   yield both slowest and fastest runtimes" (§06 §3.2).
5. **Twiddles: compile-time constants, computed offline in extended precision, canonicalised
   positive.** The total twiddle data for all four sizes including pre-splatted forms is
   under 3 KiB (§07 §5.3), so the precomputed-vs-recurrence tradeoff simply does not exist.
   Never use a trig recurrence (O(√N)–O(N²) error growth on top of the FFT's own O(log N) —
   §04 §5.3, §07 §5.1). Making all constants positive was worth **10–15%** to Frigo
   (§01 §1.2, §06 §6.1).

---

## 3. Per-size strategy

### 3.1 At a glance

| | **L = 6** = 2·3 | **L = 8** = 2³ | **L = 17** prime | **L = 36** = 2²·3² |
|---|---|---|---|---|
| points / volume | 216 | 512 | 4 913 | 46 656 |
| bytes / volume | 3.375 KiB | 8 KiB | 76.8 KiB | 729 KiB |
| **primary decomposition** | **PFA 2×3** (no twiddles) §02 §5.3 | **radix-8 / split-radix**, all trivial twiddles written out §01 §2.3 | **contested — build two** (see 3.4) | **PFA 4×9** (no twiddles) §02 §5.4 |
| fully unroll? | yes, whole line | yes, whole line | yes, whole module | yes, but as **two stages** (4 then 9), never one 36-pt kernel §01 §8 |
| best published 1D count | 48 flops / 36 instrs — **provably optimal, stop** §02 §5.3 | 60 flops / 52 instrs (4 mul + 52 add) — **settled, stop** §01 §2.3 | 468/388 (Rader) or 592/**336** (dense-symmetric) §02 §2.4, §2.5 | 688 flops / **464** instrs (PFA, FFTW modules) §02 §5.4 |
| FFTW ships a codelet? | yes (`n1_6`,`t1_6`,`n1fv_6`,`q1_6`) | yes (`n1_8`,`t1_8`,`n1fv_8`,`q1_8`,`t2_8`) | **no** | **no** |
| measured FFTW today | 3.11 ns/pt | 2.95 ns/pt | **20.98 ns/pt** | 7.59 ns/pt |
| AVX2 speedup FFTW gets | 2.41× | 2.13× | **1.00×** | 2.07× |
| registers, batch-vectorised | 12 data (+temps) of 16 ymm | **16 data of 16 ymm — will spill on AVX2** | 34 — exceeds even AVX-512 | 72 — must decompose |
| cache regime | whole volume L1-resident ×9 | whole volume L1-resident ×4 | L2; block by plane (4.6 KiB) | **L3/DRAM; must tile** |
| padding | optional (Nx=Ny=7, +36%) | **contested — see §4.5** | none needed (17 odd) | **Nx=Ny=37, +5.6% — cheapest and most robust** §04 §7.3 |
| huge pages | >~1 000 volumes | >~500 volumes | >~100 volumes | **always** §05 §7 |
| where the win is | FMA + full-width lanes (FFTW reaches only ~19% of FMA peak using a *non-FMA* codelet) §02 §5.3 | AVX-512 register residency; L1 conflict avoidance | **vectorisation, ~2.4× over FFTW and MKL** §02 §6.2 | PFA + pass-count reduction + TLB |
| **measured on the node, r3, B=1** | 0.220 µs, 1.68× MKL | 0.570 µs, 1.14× MKL | 16.39 µs, 4.99× FFTW | 118.6 µs, 1.37× MKL |
| **margin over its FP-port floor at the *correct* 2.9 GHz clock** §08 §4.1 | **1.31×** | **1.32×** | **1.32×** | **1.43×** |
| **arithmetic intensity, flop/B** §08 §7 item 2 | 1.21 | 1.41 | 1.92 | 2.42 |
| **regime at large batch** (node balance: 4.03 flop/B at DRAM, 2.55 at L3, 0.53 at L2) §08 §7 item 2 | DRAM-bound, but partly L3-served | DRAM-bound, but partly L3-served | **L2-resident per volume — compute-bound** | DRAM-bound at every B ≥ 2 |
| **SIMD width on the scoring node** §08 §4.1–4.4 | **512-bit** (no frequency penalty at 1 core; 2L=12 of 32 zmm) | **512-bit** (2L=16 spills on 16 ymm, fits 32 zmm) | **512-bit** (register file + 1.7× L2 bandwidth) | **512-bit** throughout (do not mix widths) |
| **batch tile for the 1 MiB L2** §08 §1.9 | NB ≈ 32 → 216 KiB | NB ≈ 24 → 384 KiB | NB = 2 → 307 KiB | impossible — tile *inside* the volume |
| **interleave granule** (whole cache lines) §08 §1.10 | 8 volumes | 8 volumes | 8 volumes | 8 volumes |
| **4 KiB alias exposure** §08 §1.8 | exposed; volume stride repeats a page offset every **32** | exposed; **every volume** (stride 8192 = 2×4096) | exposed; repeats every **256** | exposed; repeats every **4** (stride ≡ 1024) |
| **round-4 first move** §3.6 | measure the clock, then 512-bit | batch-tile into L2 + both structures as candidates | scan B=1…32 for the real crossover | NT + concurrency, and instrument first |

### 3.2 L = 6 — priority order

1. **PFA 2×3, single fully-unrolled straight-line codelet, batch in the lanes.**
   §02 §5.3 shows FFTW's `n1_6` already attains the Good–Thomas count *exactly* in both
   metrics (48 flops = 3·DFT2 + 2·DFT3; 36 instrs likewise). **There is no arithmetic left to
   win at L=6.** Do not spend a day on it.
2. **FMA and full-width lanes.** FFTW picks the *non-FMA* `n1fv_6_avx` codelet and reaches
   ~19% of the 2.5 GHz AVX2 FMA peak (7.73 of 40 flops/ns, §02 §5.3). Four volumes per ymm
   (eight per zmm), split re/im, FMA throughout. This is the entire L=6 opportunity.
3. **Fuse all three axes inside L1.** 4 volumes batched = 13.5 KiB, 8 volumes = 27 KiB
   (§05 §5.3, §07 §4.3). Load once, do x/y/z, store once. On AVX-512 you can additionally
   fuse x and y at *plane* granularity in registers (a 6×6 plane is 9 zmm, 23 spare —
   §07 §4.2). On AVX2 the honest unit is the pencil.
4. **Watch the register count.** §04's data-only figure is 12 of 16 ymm; §01's *measured*
   liveness for the scalar-shaped codelet including temporaries is **17** — over 16. Expect a
   few spills to L1 on AVX2 and check the assembly (§4.1 below).
5. **Optional, worth one experiment:** the per-axis PFA factorisation
   `DFT_{6³} ≅ DFT_{2³} ⊗ DFT_{3³}` — a 27-fold batch of 8-point 3D DFTs (pure additions)
   tensored with an 8-fold batch of 27-point 3D DFTs, joined by a pure index permutation
   (§03 §5.2). The prettiest decomposition any of the four sizes admits. Benchmark against
   the flat codelet; §03 §9.4 flags it as the single most promising structural idea for L=6.

> **Corrections from §08.** (a) Item 4's register worry is an AVX2 worry only: on the scoring node
> 2L = 12 of **32** zmm, and §08 §4.1 shows there is **no AVX-512 frequency penalty at one active
> core on a Gold 5218** (3.9 GHz non-AVX / 2.9 AVX2 / **2.9 AVX-512**, and AVX2 = AVX-512 up to 8
> cores). `L6_pfa`'s round-1 decision not to write a 512-bit path was reasoned from the opposite
> premise and should be revisited. (b) The r3 verdict's "L=6 is finished, 1.04× above its port
> floor" used the 2.30 GHz base clock; at 2.9 GHz the margin is **1.31×** (§08 §4.1). Measure the
> clock before believing L=6 is done. (c) L=6 is the geometry where §08 §1.3's victim-L3 result
> matters most: at B=4096 (27 MiB, 1.2× L3) a meaningful share of the traffic is L3-served, which
> is why NT stores keep losing here (§08 §1.5).

### 3.3 L = 8 — priority order

1. **Radix-8 or split-radix, every trivial twiddle written out.** 4 real multiplies + 52
   real adds; radix-8-with-butterflies-unrolled, split-radix, and the Winograd length-8
   module all land on *exactly* this count (§01 §2.3 — Burrus T7.1 = T9.1 radix-8 = T9.1
   split-radix = FFTW `n1_8`). **Nothing left to find.** The only nontrivial constant is
   1/√2; every other twiddle is ±1 or ±i.
2. **Never let a generic complex-multiply routine near this codelet.** Burrus Table 9.1's
   M₁→M₅ columns show radix-2 at N=8 going from 48 multiplies to 4 purely from special-casing
   ω = 1, ±i, ±(1±i)/√2 — a **12× swing** (§01 §2.3). This is the single biggest lever in any
   small-n codelet.
3. **Solve the register problem explicitly.** Batch-vectorised, 2L = 16 = the entire AVX2
   register file with zero spare (§04 §1.3, §07 §4.3), and §01's measured liveness with
   temporaries is **19**. Three documented options, in preference order: (a) run it on
   AVX-512, where 16 of 32 zmm makes a fully unrolled radix-8 genuinely register-resident —
   this is a real reason to prefer the AVX-512 nodes for L=8 (§07 §4.3); (b) split into
   radix-4 × radix-2 so ~10 vectors are live at a time, reloading from L1 between halves
   (L1 is 4-cycle, 2×32 B/cycle — far cheaper than a spill/reload pair) (§04 §9); (c) drop to
   2-wide batching on AVX2 (8 of 16 ymm) and accept half the SIMD width (§07 §7.2).
4. **In split layout ±i is free** — a register rename plus a sign folded into the next
   add/sub or fmadd/fnmadd (§04 §2.4). Radix-8 is *full* of ×(±i), so this is where split
   batch-minor layout pulls furthest ahead of interleaved. Interleaved would spend a
   port-5 `FLIP_RI` on every one.
5. **Cache-set geometry is the L=8 hazard and it is layout-dependent.** See §4.5 — the two
   sections disagree, and the disagreement matters.

> **Corrections from §08.** (a) Item 3 is settled in favour of option (a): **512-bit, because there
> is no frequency penalty at one core on this SKU** (§08 §4.1) and 2L = 16 fits 32 zmm with 16
> spare where it fills 16 ymm exactly. Options (b) and (c) exist as fallbacks, not as the plan.
> (b) Two page-aligned buffers mean `in` and `out` collide in the memory order buffer's low 12 bits
> at **every element**, and L=8's volume stride of exactly 8192 B = 2 × 4096 makes the pattern
> maximally degenerate — every volume starts at the same page offset. L=8 is the most exposed
> geometry on the board. This is the panel's most likely unexamined stall (§08 §1.8) and one counter
> settles it.
> (c) The §4.5 padding question has a one-line production answer worth copying: ducc0 does
> `if ((dstride & 256) == 0) dstride += 16;` on every scratch stride (§08 §5.5).
> (d) **The largest single move available at L=8 is batch-tiling into L2** — NB ≈ 24 volumes is
> 384 KiB, 37 % of the 1 MiB L2, all three axes inside the tile, one stream in and one out
> (§08 §1.9).

### 3.4 L = 17 — priority order

**This is where the money is.** §02 §6.2 measures FFTW at 20.98 ns/point and MKL at
20.86 ns/point on this node — agreeing to within 1%, i.e. *neither library has a good 17*.
That is 5.3× the per-point cost of 16³. With SIMD disabled the penalty collapses to 1.65×,
so **~70% of the L=17 penalty is missing vectorisation, not arithmetic** (§02 §1). FFTW's
AVX2 speedup at 17³ is **1.00×** against 2.0–3.2× at every other size, because there is no
vector codelet for 17 and `dft/generic.c` is plain scalar C.

1. **Vectorise across the batch. This is the 2–3×, and it is mechanical.** Every candidate
   kernel below is straight-line code with *identical* index patterns in every lane: the
   primitive-root gather is the same index list in all lanes, the constants are lane-invariant
   broadcasts, and there is not a single cross-lane operation anywhere (§02 §8.7).
   Conservative target from §02 §6.2: **~8–9 ns/point, ≈2.4× faster than FFTW and MKL.**
2. **Build the dense conjugate-symmetric kernel first.** §02 §2.5: FFTW's own `dft-generic-17`
   exploits `g^{q+8} ≡ −g^q (mod 17)` so outputs pair up conjugately — 592 flops but only
   **336 FP instructions**, of which 256 are FMAs. No buffers, no permutation, no convolution
   table. It is what FFTW's measuring planner selects under `FFTW_EXHAUSTIVE` after 1.3 s of
   search, and it is where VkFFT's `fixMinRaderPrimeMult` default of exactly 17 puts p=17
   (§02 §2.6).
3. **Also build Rader-17, and measure.** 468 flops / 388 instructions (§02 §2.4). It wins on
   flops by 21% and *loses* on instructions by 13% — this is the metric trap of §02 §2.7 in
   its purest form. It also has a much shorter dependency chain, which may matter more than
   either count. 17 is the friendliest possible prime: p−1 = 16 = 2⁴, so the cyclic
   convolution is a power-of-two length — no zero-padding, no recursive Rader, no Bluestein.
   Constants (verified numerically to 4.4e-15 in §02 §2.2): g = 3, g⁻¹ = 6, the 16-entry
   gather `a[q] = x[3^q mod 17]`, and `B = DFT₁₆(b)/16` folded into a compile-time table so
   only **two** length-16 transforms run per 17-point DFT, not three. Add x[0] into the DC bin
   *before* the inverse transform, saving 16 complex adds.
4. **Do not use Bluestein.** 1932 flops / 1524 instructions — 4.1× Rader, 4.5× the dense form
   in instructions (§02 §4.2). FFTW never selects it for n=17 at any rigor level.
5. **Do not build a WFTA / minimum-multiplication 17-point module first.** §01 §2.2 quotes
   Burrus Table 7.1 (70 mul + 314 add) and Table 7.2 (82 mul + 274 add) as lower *flop*
   targets — but §02 §3.3 and §06 §6.4b both quote FFTW's flat statement that this tradeoff
   "is not beneficial on current processors", and §02 §8.6 estimates a cyclotomic length-16
   convolution would beat two length-16 FFTs by ≤10% on flops while losing on regularity and
   dependency depth. The canonical hand-derived module (Johnson & Burrus, TR 8105, Rice 1981)
   **is not available online in any fetchable form** (§02 §9, gap 1). See §4.2 below.
6. **Register-block whatever you build.** 2L = 34 exceeds even AVX-512's 32 zmm (§04 §9); the
   16-point convolution inside Rader needs 2·16 = 32, exactly the file with no temporaries.
   Stage it (4×4, or output-blocked 8+9 for the dense form) with the intermediate in L1.
7. **Blocking: plane granularity.** 76.8 KiB blows L1 by 2.4× but is 30% of L2. A 17×17 plane
   is 4.6 KiB; batched 4-wide it is 18.5 KiB = 56% of L1 (§05 §5.3, §07 §4.3). Fuse x and y
   over an L1-resident slab, z in a second pass. **No padding needed** — 17 is odd, so with a
   64-byte batch granule every stride is an odd number of cache lines and all 64 L1 sets and
   1024 L2 sets are used (§04 §7.3). L=17 is the one size where cache geometry is free.
8. **Build the 16-entry Rader kernel table in extended precision.** VkFFT reports a measurable
   accuracy deficit specifically "in Bluestein's and Rader's algorithms" from low-precision
   twiddles (§07 §5.2); that table's error feeds every output.

> **Corrections and additions from §08.** (a) §4.2's framing — "which of these does the literature
> favour?" — is wrong: **the field does not agree.** ducc0 uses *neither* Rader nor Bluestein at 17
> (its dispatch is `if (ip < 110) generic_pass else bluestein`, so 17 takes the generic pass);
> VkFFT's `fixMinRaderPrimeMult` **and** `fixMinRaderPrimeFFT` defaults are **both exactly 17**,
> with a documented hedge that direct multiplication can win "for small ones, like 17-23"; cuFFT
> uses direct-matrix Rader up to prime 127 and switches to Bluestein after 17 in FP32
> (§08 §5.5, §5.7). Three defensible answers at our one prime — this is the panel's clearest
> opportunity, not a gap in our reading. (b) McFarlin et al.'s "direct computation preferable up to
> n ≈ 20", which the corpus cites for the dense kernel, is a **Larrabee** result whose stated
> mechanism is *broadcast hardware plus FMA* — i.e. it transfers **better to AVX-512 embedded
> broadcast than to AVX2**, which is a positive for the dense conjugate-symmetric kernel on this
> node (§08 §3.1). (c) **The batch crossover should be scanned at B = 1, 2, 4, 8, 16, 32.** If the
> ranking inverts near B = ν = 8 the cause is vectorisability and the fix is a batch-blocked
> schedule, not a choice of kernel; if it inverts near 256 the cause is capacity. The panel's grid
> (1, 8, 256, 2048) cannot tell these apart (§08 §3.1). (d) L=17 is the one geometry that is
> **L2-resident per volume** (157 KiB of 1 MiB) and therefore *compute*-bound: AI 1.92 against an
> L2 balance of 0.53. Optimise it for instructions, at 512 bits, and use the 1.7× L2 read
> bandwidth 512-bit loads give you (§08 §1.2, §7.2). (e) The 3-FMA lifting/half-angle rotation
> (§08 §6.3) is a 25 % instruction cut on every non-trivial twiddle multiply and L=17 is where the
> twiddles are.

### 3.5 L = 36 — priority order

1. **PFA 4×9. Not Cooley–Tukey.** 4 and 9 are coprime, so Good–Thomas applies and the entire
   twiddle stage vanishes — no `t1_*` codelets, no twiddle table loads, no extra dependency
   chain, and none of the `dft-indirect-before` copy pass that FFTW's own plan drags in
   (§02 §5.4). Measured against FFTW's mixed-radix 3/6/12 tree: **−9% flops, −16% FP
   instructions**, plus only two kernel shapes instead of three. §01 §8 computes the CT
   alternatives for comparison: CT 4×9 ≈ 704 flops, CT 6×6 ≈ 678, PFA 4×9 = 560 (Burrus
   modules) — PFA beats both on multiplies by >2×.
2. **Bake in the index permutation; it is free here.** The Ruritanian/CRT tables are in
   §02 §5.2, verified to 4.5e-15. Each 1D pass along a 3D axis is *already* a strided
   gather/scatter, so composing a constant permutation changes the index arithmetic, not the
   number of memory operations (§02 §5.5). `static const int idx_in[36], idx_out[36]` as byte
   offsets and the loop looks identical.
3. **Two register-resident stages, never one 36-point kernel.** §01 §7.2 measures n=32 at 40
   vector registers; a fused 36-point kernel would need ~45 and spill even on AVX-512.
   Build a size-4 stage (6 live vectors) and a size-9 stage (15 live vectors + 14–19 distinct
   constants — keep those as broadcast-from-memory FMA operands, not registers, and it fits
   AVX2 with one to spare). **PFA is a register-pressure optimisation here as much as a
   twiddle-count one** (§07 §4.2). If the 9-point stage spills on AVX2, fall back to four thin
   stages (2,2,3,3 — 2,2,5,5 live vectors) at the cost of more passes, and balance the choice
   the way VkFFT's `isGoodSequence` test does: reject any factorisation whose fattest stage
   exceeds 16 registers or exceeds 2× the thinnest (§01 §5).
4. **Invest in the 9-point module.** `n1_9` at 136 flops / 80 instructions is ~79% of PFA-36's
   arithmetic (§02 §5.4). 9 = 3² is a prime power so Good–Thomas cannot help inside it. This
   is the one place at L=36 where a better hand-derived module could pay — but re-read §02 §2.7
   and §03 §3.3 first: on AVX2+FMA, trading multiplies for adds loses.
5. **Pad Nx = Ny = 37.** +5.6% memory, 37 is prime, and it makes every stride (37, 37²,
   37³ lines) an odd number of cache lines — full set coverage at L1 and L2. The cheapest and
   most robust padding of the four sizes (§04 §7.3). Without it, the naive 36³ volume stride
   is 746 496 = 2¹⁰·3⁶ bytes → four L1 sets, Bailey efficiency 0.062 in L2. **36³ looks
   innocent and is not.**
6. **Huge pages, unconditionally.** 183 4-KiB pages per volume, and at L=36 *every element of
   a z-pencil sits on a different 4-KiB page* (z-stride 20 736 B; §05 §7). A batch of ~10
   volumes exhausts TLB reach. With 2 MiB pages an entire volume is 0.36 of one page and the
   issue evaporates. `madvise(MADV_HUGEPAGE)` is the low-friction route on this node
   (`HugePages_Total: 0` at time of writing, so `hugetlbfs` needs configuring first).
7. **Reduce the pass count over the grid.** This is the only one of the four sizes that is
   genuinely DRAM-bound at realistic LQCD batch sizes: a 30 MiB L3 holds ~14 volumes at a 3×
   working set, and 768 volumes (64 time slices × 12 spin-colour) is 547 MiB (§05 §5.4).
   Three-pass arithmetic intensity is 0.81 flop/B against a machine balance near 3; one fused
   pass over tiles gets to 2.42 — still short, but 3× better. Tile the y and z passes
   (§05 §5.3 suggests ~4×4 pencil groups, ≈37 KiB at V=4; §07 §7.2 suggests a 36×8 tile at
   4.5 KiB scalar / 18 KiB batched-4 — tune from there). Liu et al.'s published two-pass 3D
   FFT is the prior art for pass-count reduction (§05 §3.4).
8. **NT stores on the final write-out only** — and never on a strided or transposing store,
   where Drepper measured them **25% slower** than ordinary stores (§05 §8.2). Inside the
   transform: never, every output element is re-read by the next axis pass.
9. **Do not tune to this node's 256 KiB L2.** L=36 is the size that flips L2 residency between
   Haswell (256 KiB) and Ice Lake/Sapphire Rapids (1–2 MiB). Block to L1 by planes, which is
   safe everywhere (§05 §10.6).

> **Corrections from §08 — item 9 is the one to change.** L1 blocking is *safe* but it now leaves
> the largest bandwidth step on the machine unused. On the scoring node a single core reads
> **L2 at 87.3 GB/s, L3 at 18.2 GB/s and DRAM at 11.5 GB/s** (§08 §1.2), and Intel's own
> optimisation manual recommends "blocking to L2 on Skylake Server microarchitecture if L2 can
> sustain the application's bandwidth requirements" while Alappat et al. recommend "switching to
> pure L2 blocking on SKX and CLX architectures" (§08 §1.4). **L3 residency is worth 1.6×; L2
> residency is worth 7.6×.** Every tile at L=36 should be sized against **1 MiB** (and be a
> compile-time constant the monitor can halve — wallaby's L2 is 2 MiB, so a tile tuned there is
> exactly twice too big).
>
> Also: (a) two page-aligned buffers mean L=36 is exposed to 4 KiB store-to-load aliasing at
> **every element** (load `in[i]` against a recent store to `out[i]`), and its volume stride of
> 746 496 B ≡ 1024 mod 4096 makes every fourth volume start at the same page offset — the
> second-most degenerate of the four geometries after L=8 (§08 §1.8). Check
> `ld_blocks_partial.address_alias` before anything else. (b) Item 7's pass-count reduction is
> right, and §08 §1.9 says why the panel's fusion experiments underdelivered: they tested fusion
> across an **L1↔L2** boundary (2.6× bandwidth gap, measured payoff ~5 %) rather than across
> **L2↔DRAM** (7× gap, untested). (c) Item 8's NT-store advice is right and its magnitude is
> **1.5×, not 4/3**, for a copy-shaped out-of-place kernel — but Intel documents a
> Skylake-Server-specific core-resource-occupancy regression that can cap per-core NT write
> bandwidth, which is a real mechanism for three rounds of tuner rejections (§08 §1.5). (d) The
> cross-volume prefetch schemes that all three L=36 entries built in r3 were aimed one whole
> volume (1.49 MB) ahead; Mowry's formula puts the distance at ≈**16 cache lines ≈ 1 KiB**, and
> Cascade Lake has **exactly 10 line fill buffers**, so `prefetcht1` is the only safe hint
> (§08 §1.6, §1.6b). (e) Read the ducc0 axis-ordering rule — "sort the extraneous dimensions in
> order of ascending output stride" — and the gemmi datum that a bad 3D axis order costs **9× over
> a straight copy** (§08 §5.5).

---

### 3.6 Round 4+ priorities: what the plateau suggests trying next

Three rounds converged on row–column with unrolled batch-vectorised codelets and stopped moving.
§08 §2 searched 2015–2026 for a measured CPU win over that structure and found none — so the
structure is right. But the plateau is *not* evidence that the problem is solved, for two reasons
that are both new information:

* **The B=1 floors were computed at the wrong clock.** Intel's turbo table for the Xeon Gold 5218
  gives **2.9 GHz** at one active core under AVX-512 (and the same 2.9 GHz under AVX2 — the licence
  levels coincide up to 8 cores). The r3 verdict's floors used the 2.30 GHz non-AVX base. At the
  right clock every geometry is **1.31–1.43×** above its floor, not 1.04–1.13× (§08 §4.1).
* **The one published measurement of our exact regime says the kernel is not the lever.** Popovici
  et al. (IPDPS 2015) built small odd complex-double 3D cubes on a single core, measured **2–3×
  over MKL and FFTW**, and were explicit that their own 1D kernels were sometimes *slower* than
  MKL's: "**This shows the impact of the data layout transformations and the loop merging we
  performed**" (§08 §2.2). The panel has spent three rounds on the half that paper lost on.

**Ranked, per geometry. Each item names the measurement that would confirm or kill it.**

#### L = 6 — measure the clock, then decide whether it is finished

1. **`perf stat -e cycles,ref-cycles` on B=1.** *Monitor action, one job.* Four entries across two
   geometries have asked for this in writing. It decides whether L=6 has ~4 % of headroom or ~31 %
   (§08 §4.1). Everything below is conditional on it.
2. **A 512-bit path, which does not currently exist.** 2L = 12 of 32 zmm; embedded-broadcast
   twiddle operands cost a load-port slot and no register on Skylake Server; and there is **no
   frequency penalty at one core**. `L6_pfa`'s analytic argument against 512-bit assumed the
   penalty (§08 §4.1, §4.2, §4.4).
3. **Batch-tile into L2 at NB ≈ 32 (216 KiB) and fuse all three axes inside the tile.** At B=4096
   the panel is *already beating* a pure-DRAM 12 GB/s model, which means a real share of traffic is
   L3-served (§08 §1.3, §1.12). Raising that share — not cutting DRAM traffic — is the lever, which
   also explains three rounds of NT-store rejections.
4. **`prefetcht1` along the batch axis at ~16 cache lines, not `prefetchw` one volume ahead**
   (§08 §1.6b). The r3 `prefetchw` result (1.41× on wallaby, 1.6 % here) is consistent with a hint
   issued far beyond the useful distance.
5. **If the clock measurement shows no headroom, move one of the two L=6 implementers to L=36**,
   as the r3 verdict already recommends. L=36 has the largest unclaimed margin on the board.

#### L = 8 — the batch tile, the alias check, and both structures as candidates

1. **Batch-tile into L2: NB ≈ 24 volumes = 384 KiB (37 % of 1 MiB), all three axes inside the
   tile, one long stream in and one out.** This is the single largest structural move available:
   it caps DRAM traffic at the compulsory 16 KiB/volume regardless of pass count, runs every
   intermediate pass at 52 B/cy instead of ~7, makes the output stream long enough for NT stores
   *and* for the hardware streamer to train on, and is the one regime where the corpus's
   pass-fusion argument has **not** been tested and found wanting (§08 §1.9).
2. **Check 4 KiB aliasing. L=8's volume stride is exactly 2 × 4096.** `perf stat -e
   ld_blocks_partial.address_alias`; if it fires, stage through a scratch buffer offset by an odd
   multiple of 64 B, or separate the load and store phases within an unrolled block (§08 §1.8).
3. **Ship `L8_batchsimd`'s r2 three-pass LANEX *and* its r3 two-pass LANEX2 as tuner candidates**
   — the r3 verdict's "single largest guaranteed gain on the board", because both numbers are
   already measured on this node. Add per-candidate licence warm-up (§08 §4.3) so the tournament
   is not measuring frequency transitions.
4. **512-bit, settled.** 2L = 16 fills 16 ymm exactly (measured liveness 19 → guaranteed spills)
   and fits 32 zmm with 16 spare, at no frequency cost (§08 §4.1, §4.4).
5. **Interleave granule of 8 volumes** so every vector access is a whole cache line — the
   condition NT stores and the L2 streamer both require (§08 §1.10).
6. **Run the free `-DL8R_SCRX=128` A/B** the r3 verdict asked for, and note ducc0's one-line
   production form of the same idea: `if ((dstride & 256) == 0) dstride += 16;` (§08 §5.5).

#### L = 17 — find the real crossover, then close the overlap gap

1. **Scan B = 1, 2, 4, 8, 16, 32.** The panel's grid (1, 8, 256, 2048) cannot distinguish
   "the ranking inverts at B ≈ ν = 8 because at B=1 there is no batch to vectorise over" from
   "it inverts at B ≈ 256 because of capacity". Those have different fixes: a batch-blocked
   schedule for the dense kernel versus tiling (§08 §3.1). One sweep, and it reframes the geometry.
2. **Software-pipeline across volumes** — `L17_matrixsimd`'s own named next step, and the r3
   verdict's 1.39× quantified gap (16.4 µs compute floor against 22.7 µs measured at B=2048).
   §08 §1.1 says to express it as **several concurrent demand streams**, not as prefetch hints:
   the ceiling is memory-level parallelism (~10–16 lines in flight), and hints are droppable.
3. **Treat L=17 as the compute-bound geometry it is.** One volume's input+output is 157 KiB — 15 %
   of L2 — so the binding bandwidth is 87.3 GB/s (512-bit) or 50.6 (256-bit), not 11.5. AI 1.92
   against an L2 balance of 0.53. **512-bit, and optimise instructions** (§08 §1.2, §7.2).
4. **The 3-FMA lifting/half-angle rotation** (`g = −tan(θ/2)`, `d = sin θ`, `|g| ≤ 0.414`): 3 FMAs
   instead of 4 instructions per constant rotation, no leftover scale, works natively in split
   layout, and L=17 is where the non-trivial twiddles live. Caveat: r2's 11.9 % op-count cut bought
   0.8 % of time here, so budget an hour, not a round (§08 §6.3).
5. **The one genuinely different structure worth a round anywhere: the three-batched-GEMM
   (μ-mode / Tucker) formulation.** At L=17 the per-axis shape is 17 × 289 × 17, i.e.
   `(MNK)^{1/3} ≈ 43.7` — inside libxsmm's documented `≤ 64` window and in the bin where libxsmm
   measured **427 GFlop/s against MKL's 215**. Nobody has published this comparison for a small
   cube on a CPU (§08 §2.3, §3.6). Hand-written straight-line code only; libxsmm and MKL stay
   external baselines.
6. **Do not spend another round on the 17-point op count.** Three rounds of evidence, and §08 §2.1
   adds a 2026 confirmation that instructions beat flops (radix-4 beating radix-8 on Sapphire
   Rapids despite 20 % more flops).

#### L = 36 — instrument, then attack traffic and concurrency

1. **Instrument before optimising** — the r3 verdict's demand, unchanged, plus §08's counter list:
   `cycles,ref-cycles` (the clock), `ld_blocks_partial.address_alias` (L=36 repeats a page offset
   every 4th volume — **exposed**), `L1D_PEND_MISS.PENDING` and `FB_FULL` (concurrency and fill-buffer
   pressure), `MEM_LOAD_RETIRED.L3_HIT` (how much the victim L3 is actually serving). And
   **`L36_pfa` must report its tuner pick** in `fft3d_description()`.
2. **Re-target every tile at 1 MiB, not 22 MiB.** L3 residency is worth 1.6× on this part; L2
   residency 7.6× (§08 §1.2, §1.4). One volume is 746 KiB, so the tile must live *inside* the
   volume — a 36×36×8 slab is 166 KiB, a 36×8 pencil group smaller still. This is the correction
   to §3.5 item 9.
3. **NT stores on the final unit-stride write-out only, as a tuner candidate, with the counters
   above.** Worth **1.5×** of DRAM traffic on a copy-shaped kernel (3:2, not 4:3) — the largest
   single number on the L=36 board — against a documented Skylake-Server occupancy risk
   (§08 §1.5). The measured ceiling at B=256 is 186 µs with write-allocate and **124 µs without**,
   versus 227.5 µs measured.
4. **Replace prefetch hints with concurrency.** Distance ≈16 cache lines, `prefetcht1` only, and
   express cross-volume overlap as *interleaved demand streams on distinct 4 KiB pages* rather than
   droppable hints (§08 §1.1, §1.6, §1.6b). Three entries' cross-volume prefetch schemes were
   rejected by the node's own tuners in r3; the distance arithmetic says why.
5. **Check the huge-page and NUMA-balancing state of the node.** A serial load-only stream on
   Cascade Lake measures a **2× difference** between `THP=always / NUMA balancing off` and the
   common Ubuntu default `THP=madvise / NUMA balancing on` (§08 §1.7). This is a monitor-side
   setting that caps what any L=36 kernel can do; `madvise(MADV_HUGEPAGE)` on scratch buffers is
   the part implementers control.
6. **Adopt ducc0's axis-ordering rule** — "sort the extraneous dimensions in order of ascending
   output stride" — and note that gemmi measured a bad 3D axis order costing **9× over a straight
   copy** at 256³ (§08 §5.5). `L36_pencilfused`'s mode-keyed pass-A variants are the right shape
   for this already.
7. **512-bit throughout, and do not mix widths within one `execute()`** (transition cost ~11 µs
   halt plus a ~680 µs relaxation window; §08 §4.3).

#### Cross-cutting, all four geometries

1. **Verify in the assembly that no address computation multiplies by a runtime value.** A codelet
   with a compile-time stride measured **1.9× faster than the same codelet with a runtime stride**
   at N=60 (§08 §2.4). Our `L` and our `L³` batch stride are both compile-time constants; confirm
   the compiler knows it.
2. **Store order, not traffic volume.** The single most valuable technique round 3 produced was a
   write-spreading reorder worth 10.8 % that changed no arithmetic and no traffic. Three
   independent sources arrive at the same rule (§08 §2.2, §5.5) — apply it at the other three
   geometries.
3. **Add candidates, never replace structures**, and give each candidate its own licence warm-up
   (≥56 000 cycles of its own width) and > 700 µs of timed run, so the tournament measures kernels
   rather than frequency transitions (§08 §4.3).
4. **Tune the search, not just the kernel.** For spaces of this size and with this failure rate,
   dual annealing at small budgets and first-improvement iterated local search at larger ones beat
   the alternatives, and Bayesian surrogates lose (§08 §5.8).
5. **Sweep the compiler as well as the flags.** Identical FFT source measured ~12 % faster under
   Clang than GCC; no compiler wins reliably (GCC fastest on 39 % of loops, ICX 40 %, Clang 21 %);
   `-mprefer-vector-width=512` must be passed explicitly because 256 is the default for
   `cascadelake`; and never add `-ffast-math` without a ±0 bit-exactness check — that exact bug
   broke FFTW 3.3.7 (§08 §5.6).
6. **Add `perf stat` output to `tryout.sh`.** In the one controlled study of this, adding profiler
   feedback on top of execution feedback moved kernel-generation success from 62 % to **72 %**
   (§08 §5.9). The counters are listed in item 1 of the L=36 list.

---

## 4. Disagreements and open questions

These are the places the corpus does not speak with one voice, or speaks with one voice and
admits it has no measurement. **Settle them empirically; do not paper over them.**

### 4.1 How many registers does a batch-vectorised codelet actually need?

§04 §1.3 and §07 §4.3 count `2L` — the L real plus L imaginary data vectors — giving 12 for
L=6 and 16 for L=8, and conclude L=6 "fits with 4 spare" on AVX2. §01 §7.2 *measured* peak
liveness on FFTW's shipped codelets under `genfft`'s own schedule, **including temporaries**,
and gets 17 for n=6 and 19 for n=8 in scalar-shaped form — both over 16. §01's own unsourced
note 3 states the consequence plainly: "for AVX2 the batch-major form of even n = 6 and n = 8
will spill a little."

`2L` is a data-only lower bound, not a budget. The trade §01 identifies — a few spill stores
to L1 versus zero shuffles and perfect streaming — is almost certainly worth taking, but it is
untested. **Open question: how much spill traffic do the AVX2 batch-vectorised L=6 and L=8
codelets actually generate, and does it cost more than the shuffles it avoids?** §07 §7.8
gives the cheap check: build them, then count stack traffic in the generated assembly
(`vmovupd` against `%rsp`/`%rbp`) before believing any timing. Frigo's UltraSPARC datum —
50–100% performance difference from lexical scoping alone, "entirely register spills and
reloads" (§01 §1.5, §07 §3) — is the price tag if you get it wrong.

### 4.2 L = 17: dense-symmetric, Rader, or a hand-derived Winograd module?

Four positions in the corpus:

- **§02 §7:** implement the dense conjugate-symmetric form (336 instructions) *and* Rader
  (388 instructions), measure both. Explicitly says Rader "is not the lever at L=17".
- **§01 §8:** derive a 17-point module the way `genfft` would — Rader with the
  symmetric/antisymmetric "alternate convolution" (`alternate_convolution = 17` in
  `magic.ml`), so the six length-16 sub-transforms are each real-input and half cost — and
  cites Burrus Tables 7.1/7.2 (356–384 flops) as targets *below* either of §02's numbers.
- **§02 §3.3, §06 §6.4b:** do not build a Winograd module at all; multiplications are not the
  scarce resource, and the WFTA "was tried and it lost".
- **§03 §6.4, §06 §6.4a:** benchmark a direct O(n²) matrix–vector product, because
  McFarlin et al. found direct computation preferable up to n ≈ 20 and FFTW keeps a generic
  Θ(n²) plan specifically "useful for n ≲ 100".

The reconciliation, insofar as one exists: §01's lower flop counts come from
minimum-multiplication modules, which the instruction-count argument says are the wrong
objective; but §01's *structural* suggestion (exploit the real/imaginary symmetry of the
symmetric and antisymmetric halves so each sub-transform is half cost) is orthogonal to the
multiply/add tradeoff and may be a genuine win on top of either kernel. Note also that §03/§06's
"direct O(n²)" should mean §02's conjugate-symmetric form (592 flops / 336 instrs), not the
naive 289-complex-multiply matvec, which §01 §8 puts at ~2278 flops — about 6× worse.

**Open questions.** (a) Which of dense-symmetric and Rader-17 wins on this hardware,
batch-vectorised? (b) Does the symmetric/antisymmetric convolution split add anything on top?
(c) Nobody in the corpus has the exact op count for a full 17-point Winograd module —
TR 8105 is not fetchable, and a search snippet claiming "27 multiplications and 57 additions"
is flagged as almost certainly covering only the Φ₁₆ sub-step and explicitly **must not be
used** (§02 §3.4). If someone has journal access, this is the highest-value gap to close.

### 4.3 Is axis fusion worth 3× or 3%?

§07 §2.1 argues from Hong–Kung that when the volume fits in the fast level, fusing all three
axes is *the only optimal strategy* and per-axis passes are "strictly worse by a constant
factor equal to the number of passes". §05 §5.4 quantifies the upside: three-pass arithmetic
intensity 0.40–0.81 flop/B versus one fused pass at 1.21–2.42.

But §07 §1.4 reports TurboFNO measuring fusion on top of an *already-optimised* FFT at only
**3–5%** for 2D and up to 10% for 1D, and degrading at large hidden dimensions. And §07's own
gap 7 is blunt: the GPU fusion literature measures against *global-memory* round trips, a
10–20× bandwidth gap; the L1↔L2 gap is far smaller, so those speedups do not transfer, and
"TurboFNO's 3–5% is probably the better prior."

**§07 gap 7 names this the single most important number this project will have to measure for
itself: there is no published CPU study comparing three L1-resident axis passes against one
fused pass for cubes in the 6³…36³ range.** §05 §10.5 gives the experiment: write the L=36
kernel twice — three stride-parameterised passes over the whole batch, versus one fused pass
over 4×4 pencil-tiles — and measure both with
`perf stat -e cache-misses,dTLB-load-misses,LLC-loads` at batch sizes 8, 64, 512. That single
sweep settles the pass-count question, the TLB question and the L3 spill point at once.

Note that the answer is probably size-dependent: at L=6/8 the whole batched volume is
L1-resident so fusion saves L1 traffic only (small); at L=36 with a large batch it saves DRAM
traffic (large). The corpus's own consensus, if it has one, is Tolmachev's rule from §07 §1.6:
the payoff is *the number of avoided passes times the bandwidth gap between the two levels
involved* — nothing more.

> **SETTLED IN PART by `panel_r3/VERDICT.md` §5, and re-opened in one regime by §08 §1.9.** The r3
> answer — single-digit percent, sometimes negative, and store *order* worth 18 % where pass count
> was worth 5 % — stands, and Tolmachev's rule survives with CPU numbers attached. But every panel
> experiment fused across an **L1↔L2** boundary, where §08 §1.9's measured bandwidth gap is 2.6×.
> The untested case is **L2↔DRAM**, where the gap is 7× (52 B/cy sustained L2 versus ~7 B/cy
> single-core DRAM) and where Intel's own manual, Alappat et al. and the L3-Fusion result all
> independently recommend the construction: **tile the batch so a tile fits L2, then run all three
> axes inside the tile.** That is not the same experiment the panel ran, and it is the largest
> untried structural move on the board.

### 4.4 Split vs interleaved complex: strongly motivated, unproven

§04 recommends split-complex batch-minor across the board and makes a strong case: ideal
vectorisation efficiency 4/4 versus interleaved's 2/4 on SPIRAL's own accounting; zero shuffle
instructions in the transform; free ×(±i); port 5 left entirely idle; 4× smaller twiddle
tables; and immunity to AVX-512's 512-bit port scheme, where FMAs and shuffles *share* port 5
(§04 §8.1) so a shuffle-heavy interleaved kernel that was FMA-bound on AVX2 can become
port-5-bound on AVX-512 and gain far less than 2×.

But §04's own "sources I could not get" section is candid: there is **no published head-to-head
of split versus interleaved complex for a batched small FFT on AVX2/AVX-512**. The closest
study (Sansone & Cococcioni 2023, on exactly this question) was paywalled, and its reported
speedups are marked UNVERIFIED. §04 says: "Treat §2's verdict as strongly motivated but
unproven, and settle it with a microbenchmark before committing all four sizes to one layout."

There is also a genuine counter-case in the corpus: FFTW's `DFT(A+iB) = DFT(A) + i·DFT(B)`
trick makes interleaved strictly *better* at complex vector length 1 (§04 §2.3), and FFTW's
4-way strategy is batch-major "applied at half strength" precisely because it must also work
with no batch. We have a batch of thousands, so the trick does not apply — but this is the
reason the world's most-benchmarked FFT ships interleaved, and it deserves one measurement
rather than a shrug.

> **CLOSED by §08 §5.4, in the corpus's favour, with three independent sources.** (1) Popovici,
> Franchetti & Low (IEEE HPEC 2017) measure split/block-interleaved kernels on x86 **in double
> precision**: `DFT_n ⊗ I_ν` kernels — literally our kernel shape — **1.3 to 2× faster than
> FFTW's**; 1D codelets 5–15 % better than FFTW and MKL; 2D 10–30 %. Their stated mechanism is
> §04's: "the split data layout **eliminates permutations**." (2) The FFTc 2.0 work measures what
> the interleaved layout costs when the compiler handles it — gather/scatter emitted "for each
> arithmetic step", and replacing it with in-register permutes moves **IPC from 0.13 to 2.59 for
> 9.25× the speed**. (3) ducc0 ships split (`Cmplx<Tsimd>` with separate `.r`/`.i`). §04's verdict
> was right; the open question is now only the *granule*, and §08 §1.10 answers that from the cache
> line: **8 volumes per granule for split complex double**, so every vector access is a whole
> 64-byte line.

### 4.5 Padding: does L = 8 need it, and where?

Three positions:

- **§04 §7.3:** with the recommended batch-minor layout at NB = 8 doubles (64 B granule), the
  naive L=8 z-stride is **exactly 4096 B = one L1 set** — Bailey's `E = 1/R = 1/64` worst case,
  verbatim. Pad Nx = Ny = 9 (+27% memory) to restore all 64 sets. Padding is "mandatory".
- **§05 §6:** analysing the naive single-volume interleaved layout, "nothing here is
  pathological — 2.25 elements per set against 8-way associativity is fine, and in any case at
  L=6 and 8 the *entire volume* fits in L1 so conflicts are moot." The real hazard §05
  identifies is elsewhere: the **volume-to-volume stride**, which at L=8 is exactly 8192 B =
  128 lines ≡ 0 mod 64 sets, so volume k and k+1 map to identical L1 sets at every offset.
- **§01 §8:** "Expect to need explicit padding for L = 8, and less so for L = 36."

These are not really contradictory — they analyse different layouts — but the practical
consequence is sharp and easy to miss: **the L=8 cache catastrophe §04 identifies is created by
the very layout §04 recommends.** A 64-byte batch granule makes every stride an integral number
of cache lines, which is exactly what turns a power-of-two extent into a single-set disaster.
§04's own rule is the fix and it is worth memorising: *with a one-cache-line element granule,
pad so every stride is an **odd** number of cache lines* — `gcd(odd, 64) = 1`, so all sets get
used. §05's alternative framing (use a genuinely batch-contiguous SoA layout so the volume
stride never appears in an inner loop) reaches the same place from the other side.

**Open question:** measure L=8 with (Nx,Ny) = (8,8) and (9,9) at NB = 8, and with NB = 4, and
check `perf stat -e L1-dcache-load-misses`. §04 §7.3 also notes NB = 4 makes odd strides a
*non-integral* number of lines, which is benign but "more fragile". The 27% memory cost at
L=8 is not trivial when the whole point is L1 residency.

> **§08 adds a production answer and a second, larger hazard.** ducc0 implements exactly §04's rule
> as a one-line guard on every scratch stride — `if ((dstride & 256) == 0) dstride += 16;` — and
> ships a public `make_noncritical()` helper it calls in its own benchmarks (§08 §5.5). Copy that
> rather than reasoning about it. **And the bigger hazard at L=8 is not cache sets, it is the memory
> order buffer:** L=8's volume stride is exactly 8192 B = 2 × 4096, so with two page-aligned buffers
> every load from `in` falsely aliases a recent store to `out` in the low 12 bits, which is a
> documented re-issue penalty (§08 §1.8). L=8's volume stride of 8192 B = 2 × 4096 makes that
> pattern maximally degenerate (every volume starts at the same page offset); L=36's stride is
> ≡ 1024 mod 4096, so every fourth volume does. One counter —
> `ld_blocks_partial.address_alias` — settles both, and neither has ever been checked.

### 4.6 Model versus search for the instruction schedule

§01 §3.6 and §07 §2.1 both quote FFTW's claim that the cache-oblivious √n-halving schedule
makes its machine-independent codelets "no slower than machine-specific codelets generated by
SPIRAL", and §01's implementer takeaway is: "you should not need a search phase for
L = 6, 8, 17, 36."

§06 §6.1 corrects this on a point of fact: genfft's spill-optimality proof holds only for sizes
that are a **power of 4** (4, 16, 64). **None of 6, 8, 17, 36 is a power of 4 — not even 8.**
Frigo himself says that for other sizes "neither the lower-bound nor the upper-bound analyses
hold — it merely works." Combined with SPIRAL's measured 2× runtime spread among equivalent
formulas on a cache-free small transform (§06 §3.2), §06 concludes the schedule is "the
primary thing to search" at every one of our sizes.

**Open question, and it is cheap to settle.** §06 §3.5 notes SPIRAL optimises sizes around 64
"within a few minutes", and §06 §6.2 records that Singer & Veloso did an *exhaustive* search
over all split trees of sizes 2²–2⁷ as unrolled straight-line code back in 2002. At our sizes a
near-exhaustive search over {schedule variant × unroll depth × batch-loop placement ×
copy-or-not × compiler flags} costs minutes. Do it. Note also §06 §3.4: >3× runtime spread
across 2000 random flag combinations, and `-O3` sometimes slower than `-O1`; plus §01 §1.5,
where disabling GCC's pre-RA scheduler was worth 50–100% on 1999 SPARC — old, and Blake
explicitly questions whether it still holds, but it costs one build to test
`-fno-schedule-insns`.

### 4.7 Vector-radix: 41.7% of the multiplications, or 4.2% of the work?

The corpus is unanimous on the verdict but the internal tension is worth stating because it is
the classic trap. §03 §2.2 derives from the textbook formula a nominal **41.7% reduction in
complex multiplications** for VR-2×2×2 in 3D. §03 §2.3 then computes from Duhamel & Vetterli's
published tables that at 8×8 the same technique saves **4.2% of total operations** (because
additions outnumber multiplications ~11:1 at that size) and at 4×4 saves *exactly zero*.
Duhamel & Vetterli's own verdict on radix-2 vector-radix is "it should not be considered".

So: **no vector-radix at any of our sizes**, and the reason is not that the multiplication
saving is fake but that multiplications are 8% of the work. §03 §9.1 adds the structural
argument: a vector-radix butterfly deliberately couples elements at strides 1, L and L² inside
one operation, so SIMD lanes must hold data from different dimensions and every butterfly needs
cross-lane work — destroying exactly the zero-shuffle property that makes batch-vectorised
row–column fast. §05 §10.3 concurs and adds that the fully unrolled 3D butterfly network would
spill on 16 ymm.

The only residual: §05 §10.3 says "if someone does try it, L = 6 is the size to try it on," and
§03 §8 marks L=8 as the sole size where classical VR-2×2×2 cleanly applies. Treat both as
last-resort, and only if profiling shows you are arithmetic-bound rather than load- or
shuffle-bound — which §03 §9.5 step 6 says the literature predicts you will not be.

### 4.8 Open questions the corpus cannot answer at all

Flagged so nobody wastes time searching:

1. ~~**No published single-node, single-threaded benchmark of small fixed 3D cubes on any
   algorithm.**~~ **PARTLY CLOSED by §08 §2.2.** Popovici, Russell, Wilkinson, Skylaris, Kelly &
   Franchetti (IPDPS 2015) measured **odd cubes of edge length 7–119, complex double, single core**,
   in both split and interleaved layouts, against **MKL 11.0.0 and FFTW 3.3.4**, and report
   "typically a factor of **2 to 3**" (abstract: averaging 3×). The specific cubes 6/8/17/36 remain
   unmeasured, but the *regime* is not virgin territory — and the paper's own diagnosis redirects
   our effort: their 1D kernels were sometimes **slower** than MKL's and the 3D kernel still won,
   because of "the data layout transformations and the loop merging we performed". See §3.6.
2. **No 3D analogue of Duhamel & Vetterli's Tables 4/5** (per-output-point op counts for
   complex data). Every 3D vector-radix figure in §03 is either the textbook formula or an
   extrapolation from the 2D 8×8 row, and is labelled as such.
3. **No exact op count for a full 17-point Winograd module** (§4.2 above).
4. **No primary source for MKL's prime-size algorithm** — §02 §6.3 reports the measurement
   (20.86 ns/pt at 17³) and nothing else.
5. **No quantified comparison of batch/vector-loop placement for batched small 3D transforms.**
   FFTW's vector-recursion results are the nearest evidence and they are for large 1D and 2D
   transforms (§06, sources searched for and not found). §06 §10.1 calls this "our largest
   untapped search axis, and the literature only gestures at it."
6. ~~**No AVX-512 measurement anywhere in this corpus.**~~ **CLOSED, and the conclusion is
   inverted — see §08 §1.2 and §4.1 before reading the rest of this item.** Intel's own turbo table
   for the **Xeon Gold 5218** (Specification Update 338848-028US, Figures 1–6) gives, at **one
   active core**: non-AVX **3.9 GHz**, AVX2 **2.9 GHz**, AVX-512 **2.9 GHz** — *identical* from 1 to
   8 active cores. The Gold 5120's 1.6 GHz figure quoted below is a **9+-core** number and does not
   describe a single-threaded run on an exclusive node. Moreover the Gold 5218 has **one** 512-bit
   FMA unit, so 512-bit and 256-bit code have *identical* peak FP throughput — meaning 512-bit is
   strictly preferable on this part (half the instructions, 32 registers instead of 16, 2× L1 and
   1.7× L2 load bandwidth, free embedded broadcast) at **zero** frequency cost. Published
   single-core bandwidths for this microarchitecture are in §08 §1.2. The historical text follows.
   Every SIMD number in §02 is an AVX2
   number on the Haswell login node. The AVX-512 nodes will shift the L=6/L=8 balance
   materially (§02 §9, gap 5) — and per §04 §8.1/§8.2 the shift is not simply "2× wider":
   the 512-bit port scheme moves FMAs onto ports 0 and 5, some Skylake SKUs have only one
   AVX-512 FMA unit, and licence-based downclocking on Skylake-SP is severe (Xeon Gold 5120:
   2.7 GHz normal → 2.3 AVX2 → **1.6 AVX-512** at 9+ active cores). Ice Lake client is benign;
   there is no primary measurement in the corpus for Ice Lake-SP or later *server* parts, which
   is the hardware most likely to be in a current LQCD cluster. **Measure it on the node.**
7. **Goedecker's actual radix-2/3/4/5 FMA operation counts** — abstract only (§04, G1). Directly
   relevant to L=6 (radix 2, 3) and L=36 (radix 4, 3).

---

## 5. Techniques ranked by expected payoff for this exact task

Batched small complex-double cubes, single-threaded x86-64, AVX2/AVX-512. Reasoning given for
each; the corpus's own measurements where they exist.

> **Read §08 §0 first.** That list is the same exercise done against the *measured* state of the
> board after three rounds and against published data for the exact scoring part, and where the two
> disagree §08 is newer. In particular: item 1 below is confirmed and now has a measured x86
> double-precision number (§08 §5.4); the AVX-512 caution threaded through Tiers 1–3 is wrong for
> this SKU at one active core (§08 §4.1); "padding to kill cache-set conflicts" (item 7) should be
> read alongside the 4 KiB-aliasing hazard, which is larger and unexamined (§08 §1.8); and the one
> genuinely new Tier-1-sized item is **cache-blocking across the batch into the 1 MiB L2**
> (§08 §1.9), which no section of the earlier corpus proposes.

**Tier 1 — do these first; they are worth multiples, not percentages.**

1. **Batch-minor split-complex layout: batch index innermost, separate real and imaginary
   arrays, NB a multiple of the vector width.** Everything else depends on this decision, and
   four sections converge on it independently (§04 §0/§3, §03 §9.1–9.2, §07 §4.3, §01 note 3,
   §06 §10.2). Payoff: the `A ⊗ I_ν` theorem makes every scalar operation exactly one vector
   instruction with **zero shuffles** for any algorithm you pick inside (§04 §3.1); FFTW's
   measured AVX2 speedups of 2.0–3.2× at the sizes where it *has* vector codelets (§02 §6.1)
   are the empirical floor. It also makes ×(±i) free, keeps port 5 idle, shrinks twiddle tables
   4×, and sidesteps AVX-512's shared-port-5 problem. §03 §9.2 puts it best: "Choose the layout
   first; it dominates the algorithm choice." Caveats in §4.1 and §4.4 above.
2. **Get L = 17 vectorised at all.** The single largest recoverable inefficiency in the target
   set, and it is recoverable by the most mechanical means available. FFTW and MKL agree to
   within 1% at 17³ and both are 5.3× per-point worse than 16³, purely because neither has a
   vector codelet for 17 (§02 §1, §6.1, §6.2). Conservative target ~2.4× over both libraries.
   Nothing else in this list has that large a certain payoff.
3. **Fully unrolled straight-line codelets with every trivial twiddle special-cased.** All four
   sizes are inside FFTW's codelet regime (§01 §1.6, §06 §6.1). The special-casing alone is the
   12× multiply swing at n=8 (§01 §2.3). Frigo's size-64 codelet was "about twice as fast as
   Digital's DXML library" (§01 §1.4). Non-negotiable baseline.

**Tier 2 — worth 10–30% each, and cheap.**

4. **PFA / Good–Thomas at L = 6 and L = 36.** Zero twiddle multiplications, zero twiddle table,
   fewer kernel shapes, and lower register pressure — measured −9% flops / −16% instructions at
   L=36 versus FFTW's own plan, plus the disappearance of FFTW's `dft-indirect-before` copy pass
   (§02 §5.4). At L=6 it is already what FFTW does and is exactly optimal, so the win there is
   only structural (§02 §5.3). Free because the permutation is compile-time constant and rides
   along with the stride arithmetic you were doing anyway (§02 §5.5, §03 §5.2).
5. **Compile-time twiddle constants, canonicalised positive, computed offline in extended
   precision.** 10–15% from positive constants alone (§01 §1.2, §06 §6.1); under 3 KiB total
   for all four sizes (§07 §5.3); removes the dominant accuracy risk (§04 §5.3, §07 §5.1–5.2).
6. **Write the FMAs explicitly as intrinsics, and target instruction count rather than flop
   count.** FMA improves the complex-multiply normwise error bound from √5·u to **2u** — the
   asymptotically optimal value — and costs nothing (§04 §5.3, §07 §6.1). Writing them as
   intrinsics rather than relying on `-ffp-contract` also pins reproducibility across your
   AVX2, AVX-512 and scalar-reference builds (§04 §10.6, §07 §7.5). Use the 4-multiply/2-add
   complex product as 2 mul + 2 FMA; **not** the 3-multiply Karatsuba form, which FFTW rejected
   on throughput grounds and which has no 2u guarantee (§07 §6.1).
7. **Padding to kill cache-set conflicts** — L=36 → 37 (+5.6%, unambiguous win); L=8 contested
   (§4.5); L=17 free; L=6 optional. When it matters it is the difference between using 64 L1
   sets and using 1 (§04 §7).
8. **Compiler flags and schedule variants as a searched dimension.** >3× spread across random
   flag sets, 8–15% from evolutionary flag search over `-O3`, `-O3` sometimes slower than `-O1`
   (§06 §3.4); GCC's pre-RA scheduler can undo your emission order (§01 §1.5). Costs one build
   each. Check the assembly for spill traffic before believing any timing (§07 §7.8).

**Tier 3 — matters only at L = 36, or only at large batch.**

9. **Cache tiling and pass-count reduction at L = 36.** The only configuration in the brief
   that is genuinely DRAM-bound at realistic LQCD batch sizes (§05 §5.4). Three-pass intensity
   0.81 flop/B against ~3 machine balance; one fused pass reaches 2.42. Contested magnitude
   (§4.3).
10. **Huge pages.** Unconditional at L=36 (183 4-KiB pages per volume, every z-pencil element on
    its own page), above a few hundred volumes elsewhere (§05 §7).
11. **Non-temporal stores on the final write-out**, when the batch exceeds L3 and the consumer
    will not touch the output immediately — worth up to the write-allocate share of the traffic
    (~1/3 of a read-1-write-1 stream). Never inside the transform, never on a strided store
    (§05 §8).
12. **Axis fusion.** Real but contested: 3× on arithmetic intensity by §05's accounting, 3–5%
    by TurboFNO's measurement on an already-optimised FFT (§4.3). Worth one experiment, not a
    week.

**Tier 4 — explicitly do not do these. The corpus says so with numbers.**

13. **Vector-radix** — 0% at 4×4, 4.2% total at 8×8, and it destroys the zero-shuffle property
    (§03 §2.3, §2.4, §9.1).
14. **Nussbaumer–Quandalle polynomial transforms** — the authors' own 7×7×7 number is 2.6% of
    the multiplications at an 11.8:1 add:mult ratio (§03 §3.2). Optimises for a machine that no
    longer exists.
15. **WFTA / minimum-multiplication modules anywhere** — §02 §3.3, §03 §3.3, §06 §6.4b, and
    Burrus's own verdict that the WFTA "has not proven as fast or as versatile as the PFA".
16. **Bluestein at L = 17** — 4.1× Rader in flops, 4.5× the dense form in instructions
    (§02 §4.2).
17. **Four-step / six-step for the line transform** — Bailey's own measurements show it 25–28%
    *slower* below n ≈ 2048 and 2× slower at n = 256 on the Y-MP (§03 §4.4, §05 §2.2). Our line
    lengths are 6 to 36. And a genuine 3D cube already *has* the unit-stride simultaneous-FFT
    structure four-step exists to manufacture, without any twiddle stage (§03 §4.5).
18. **Gather / scatter instructions** — Intel's own shuffle-based deinterleave is 4.9× the
    gather version on Skylake, and every access pattern Intel lists as "do not use gather for"
    describes an FFT (§04 §2.5, §8.4). If your layout needs them, change the layout.
19. **Trig recurrences for twiddles** (§5 above), **compensated summation** (no source justifies
    it, and an FFT butterfly has no long accumulation chain — §07 §6.4), and **exactly-rounded /
    error-free FFT machinery** (measured at **107–1315×** FFTW's runtime — §07 §6.3).
20. **Recursion, planners, or any runtime algorithm selection inside the transform.** All four
    sizes are leaves. A runtime planner at these sizes is pure overhead (§06 §9).

---

## 6. Citation health

**§08 (added after round 3).** Roughly 3 400 lines carrying **133 numbered source entries**, every
one fetched in the session that produced it, with a separate ledger of ~25 sources that could not be
fetched and the specific claims that therefore remain unverified (§08 §8.2). It is the only section
whose primary sources are predominantly **2015–2026** and the only one whose hardware claims are
about the Cascade Lake-SP part the panel is actually scored on. Its §8.3 tabulates, with both sides
cited, the ten places where it contradicts or supersedes an earlier section — the largest being
§04 §8.2's AVX-512 downclocking figure (wrong SKU and wrong core count for our workload), §05
§10.6's "block to L1, which is safe everywhere" (safe but now leaves 7.6× on the table), and §03 §7's
"genuinely unmeasured territory" (partly measured, and the measurement redirects our effort). It also
carries its own unsourced-notes section (§08 §7) keeping derived arithmetic separate from cited
material, in the style §02 §8 and §03 §9 established.

**Volume.** Roughly 7 600 lines across the original seven sections, carrying **116 distinct URLs**.
Counting bibliography entries: about **100 source-references marked fetched-and-read**
(with substantial and deliberate overlap — Frigo PLDI'99, Frigo & Johnson 2005, Johnson &
Frigo 2008, Burrus's *Fast Fourier Transforms*, Duhamel & Vetterli 1990, Bailey 1990,
McFarlin et al. 2011 and the VkFFT/FFTW source trees are each cited independently by three or
more sections, which is a strength — the cross-checks are real). Distinct primary sources
actually read: on the order of 70.

**Discipline.** Good, and unusually explicit. Every section carries a fetch-status ledger.
Marker counts: **31 `[UNVERIFIED — could not fetch]`** flags, **9 `[BIB-ONLY]`** (§02),
**12 `[F-via]`** (§07) — the latter two categories being classical papers (Rader 1968,
Winograd 1976/78/79, Good 1958, Thomas 1963, Hong & Kung 1981, Gentleman & Sande 1966,
Nussbaumer 1982) correctly cited *via* a fetched reference list rather than pretended to.
Several sections go further and refuse to quote numbers they could not verify: §02 §3.4
explicitly rejects a search-snippet claim of "27 multiplications and 57 additions" for a
17-point module; §05 §2.3 refuses to quote Takahashi's MFLOPS figures; §03 §7 refuses
Takahashi's cluster numbers; §04 flags the MKL storage-scheme page (I2) and the Sansone &
Cococcioni speedups (SA1) as unverified. That is the right instinct and it should be
preserved.

Cross-validation is strong where it exists. §01 §2.5 checks `genfft`'s shipped op counts
against Burrus's published tables and finds exact agreement at n = 3, 4, 6, 8, 12, 16, 32, 64;
§02 §2.4 validates its op-count model by reproducing FFTW's reported 3D plan flops exactly at
6³, 16³ and 36³; §02 §5.2 and §2.2 verify the permutation and primitive-root tables
numerically against `numpy.fft` to ~4e-15; §01 §2.3 confirms Yavne's closed form reproduces
Burrus Table 9.1. Several key claims (register counts, cache geometry) were verified directly
on the machine rather than from a document.

**Thin or weak areas.**

- **§05 §3.5 (transpose-free multidimensional / vector-radix)** is the thinnest passage in the
  corpus — every primary source failed to fetch, and §05 correctly asserts nothing. It is
  redundant with §03, which does have the numbers, so the gap is covered; but do not read
  §05 §3.5 as evidence of anything.
- **§05's TLB section (§7)** asserts no TLB entry count anywhere, because Agner Fog's manual
  did not yield Haswell numbers and wikichip refused the connection. The mechanism (Drepper) is
  well sourced; the capacity numbers must be read off the machine.
- **§04's central recommendation (split vs interleaved) rests on no direct published
  measurement** — §04 says so itself. It is assembled from a vectorisation-efficiency count, a
  format-conversion benchmark, two vendor preferences and the author's own port accounting. The
  strongest single argument for it ("multiplication by ±i is free in split layout") is flagged
  as unsourced.
- **§07's register counts (16 ymm / 32 zmm)** are unsourced — the Intel Optimization Manual
  exceeded the fetch limit for that section. Not controversial, and §01 §7.1 verified them
  directly by assembling `vaddpd %zmm31, %zmm30, %zmm29` and `vaddpd %ymm15, %ymm14, %ymm13`
  on this machine, and §04 fetched the manual successfully. Cross-covered.
- **§06 has the fewest "fetched" mentions** (5) relative to its 20 verified references, but its
  bibliography is fully enumerated with per-section usage notes; the low count is a stylistic
  difference, not a discipline problem. §06 is also honest about three numbers it could not
  extract because the source PDFs typeset them in math fonts it could not decode, and it
  declines to assert them — including correcting an earlier draft's error about genfft's
  power-of-4 condition.
- **Three named "closest thing that exists" papers are unread across the whole corpus** and
  each would materially change a recommendation if obtained: Johnson & Burrus TR 8105 (the
  canonical 17-point module), Blake/Witten/Cree 2013 (the only published CPU library built on
  runtime code specialisation — the nearest prior art to this entire project), and Sansone &
  Cococcioni 2023 (the only study of exactly §04's layout question under AVX-512). All three
  returned 403 from every route tried. **Two independent research passes retried all three while
  producing §08 and failed again.** Note however that Sansone & Cococcioni is no longer
  load-bearing: §08 §5.4 answers the layout question with three other fetched sources, one of them
  a direct x86 double-precision measurement of our exact kernel shape.

- **§08's own weakest points, stated so they are not over-read.** (a) Its central AVX-512 result
  rests on an Intel *specification* table, not a measurement, and it flags that the Gold 5218's
  2.9 GHz AVX turbo is anomalously low next to its sibling 5220's 3.8 GHz — one `perf stat -e
  cycles,ref-cycles` on the node would make all of §08 §4 measured rather than inferred, and that
  measurement has not been taken. (b) Its per-cycle FMA-throughput table is a *composition* of four
  verified documents, not a single source. (c) Several batched-transform numbers it relies on are
  fp32 and/or multi-threaded (Zlateski et al., Gelashvili et al.) and are flagged as such in place.
  (d) Two of the §6 sources are 2026 preprints, one of which appears to mis-attribute its central
  construction; §08 §7 note 9 says so and rests the recommendations on the peer-reviewed classics
  behind them instead.
- **The single largest evidential gap is not a citation problem at all:** there is no published
  measurement of a small fixed 3D complex-double cube on a single CPU core, by anyone, on any
  algorithm (§03 §7, §07 gap 7). The corpus is honest about this. It means the ranking in §5
  above is a literature-informed prior, not a result, and the first week of implementation work
  should be spent turning the three open questions in §4.1, §4.3 and §4.4 into measurements
  under §06 §8.5's protocol.

| 11 | post-2020 algorithmic ideas untested in performant software | literature/11-post2020-untested-ideas.md | 6-vein agent sweep 2026-08-24; tiered actionability; staging/ has full reports |
