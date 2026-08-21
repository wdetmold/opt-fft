# LITERATURE.md — corpus index and cross-cutting synthesis

**Audience.** The implementer panel about to write self-contained C for 3D complex-double
forward FFTs on cubes **L = 6, 8, 17, 36**, many volumes batched, single-threaded x86-64,
AVX2 baseline with AVX-512 on some nodes, no library calls inside the transform.

**What this file is.** An index to the seven literature sections in
`docs/literature/`, and — more usefully — the synthesis across them: what all seven agree
on, where they contradict each other, and what to build first. Every number here is
traceable to a section; the sections carry the URLs and the fetch status.

**Read this file, then read §04 and §01 in full before writing a line of code.** Those two
carry the layout decision and the register budget, which between them determine everything
else.

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

Supporting material already in the tree: `docs/SURVEY.md` (installed libraries and hardware),
`docs/TEXTBOOK_FFT.md` (the Python reference implementation to validate against),
`docs/literature/tools/fftw_codelet_liveness.py` (the liveness measurement in §01).

---

## 2. The five things all seven sections agree on

Before the per-size table, the consensus. These are not contested anywhere in the corpus.

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

1. **No published single-node, single-threaded benchmark of small fixed 3D cubes on any
   algorithm.** §03 §7 states this as the central gap: "there is no published measurement
   telling you what the fastest 6³/8³/17³/36³ complex-double kernel looks like. You are in
   genuinely unmeasured territory."
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
6. **No AVX-512 measurement anywhere in this corpus.** Every SIMD number in §02 is an AVX2
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

**Volume.** Roughly 7 600 lines across the seven sections, carrying **116 distinct URLs**.
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
  returned 403 from every route tried.
- **The single largest evidential gap is not a citation problem at all:** there is no published
  measurement of a small fixed 3D complex-double cube on a single CPU core, by anyone, on any
  algorithm (§03 §7, §07 gap 7). The corpus is honest about this. It means the ranking in §5
  above is a literature-informed prior, not a result, and the first week of implementation work
  should be spent turning the three open questions in §4.1, §4.3 and §4.4 into measurements
  under §06 §8.5's protocol.
