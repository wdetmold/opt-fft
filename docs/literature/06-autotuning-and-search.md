# 06 — Autotuning, Search, and Empirical Selection

**Scope.** What the literature actually establishes about *choosing* between
functionally-equivalent FFT implementations by measurement rather than by reasoning:
FFTW's planner, SPIRAL's search over formula space, ATLAS-style empirical tuning and its
model-driven critique, learned/ML plan selection, and — the case that matters here — what
is known about tuning when the transform size is **fixed and known at compile time**.
The last section covers the reliability of FFT benchmarking itself, because a monitor agent
will be timing competing implementations and must be able to defend the numbers.

**Citation policy in this file.** Every reference in §11 marked *(verified)* has a URL that
was fetched during the session that produced this document, and every number attributed to
it was read out of that fetched text. Two references are marked
`[UNVERIFIED — could not fetch]`; nothing load-bearing rests on them. Claims that are my own
reasoning rather than a source's are quarantined in §10.

**Our four sizes, as memory footprints** (complex double, one volume, 16 B/point):

| L | L³ points | one volume | 1-D line length | where it lives |
|---|---|---|---|---|
| 6 | 216 | 3.4 kB | 6 | fits L1 many times over |
| 8 | 512 | 8.2 kB | 8 | fits L1 many times over |
| 17 | 4 913 | 78.6 kB | 17 | L2 |
| 36 | 46 656 | 746 kB | 36 | L2/L3 boundary |

Keep this table in view: most of the published autotuning payoff is a *cache* payoff, and
two of our four sizes have no cache problem to solve. That changes which parts of the
literature apply. See §7.

---

## 0. Executive summary for implementers

1. **Do not select your algorithm by operation count.** This is the single most strongly
   supported empirical claim in the autotuning literature, and it is supported with numbers
   (§1). For L=17 in particular it is a direct argument against Winograd (§6.4).
2. **All four of our 1-D line lengths (6, 8, 17, 36) fall inside FFTW's straight-line
   "codelet" regime (n ≲ 64).** In FFTW's own architecture there is *no recursion and no
   planner decision* at these sizes — a hard-coded, fully unrolled, symbolically simplified
   block of code is used unconditionally (§6.1). The search space we actually face is not
   "which radix tree" but "which dag, which schedule, which loop order for the batch" (§7).
3. **No theoretical shortcut covers our sizes.** genfft's schedule is *provably*
   spill-optimal only for transform sizes that are a **power of 4** (4, 16, 64). None of
   6, 8, 17, 36 is a power of 4 — not even 8. At every one of our sizes the instruction
   schedule is a heuristic, so it is the primary thing to search (§6.1).
4. **The measured payoff of search is size-dependent and the FFTW authors quantified it:**
   ~20% for moderate n ≲ 2¹³, a factor of 2–3 for large n ≳ 2¹⁶ (§2.3). L=6, 8, 17 are all
   at or below 2¹³ total points; L=36 (46 656 ≈ 2¹⁵.⁵) sits in the transition. So expect
   search to buy the most at L=36 and the least at L=6/8 — but *not zero* at L=6/8, because
   SPIRAL measured a 2× runtime spread among equivalent formulas for a small transform with
   **no cache effects at all** (§3.2).
5. **Tune on the machine you will run on.** Cross-machine plan reuse cost FFTW ≥20% on at
   least a third of sizes benchmarked, up to 40% (§2.4); cross-generation reuse cost SPIRAL
   up to 50%, and cross-ISA reuse up to 320% (§3.3). We have both AVX2 and AVX-512 nodes;
   that is a cross-ISA boundary, not a detail.
6. **Compiler flags are part of the search space.** A random sample of 2000 flag
   combinations on SPIRAL-generated DCT code spread runtime by more than 3×, and evolutionary
   flag search bought a further 8–15% over `gcc -O3` on the larger sizes (§3.4).
7. **Search cost is real but bounded at our sizes.** SPIRAL: "Problem sizes around 64 are
   optimized within a few minutes." ATLAS's global search: 8 min to >8 h depending on machine
   (§4). At L=6/8/17/36 an exhaustive or near-exhaustive search is affordable — Singer &
   Veloso did exactly this for straight-line leaves of size 2²–2⁷ (§6.2).
8. **Time it properly or the whole exercise is theatre.** §8 gives a concrete protocol
   assembled from benchFFT, Intel's timing whitepaper, and the benchmarking-statistics
   literature, and explains the genuine disagreement in that literature about
   minimum-vs-mean.

---

## 1. The premise: arithmetic cost stopped predicting time

Every autotuning system in this file exists because of one empirical fact, and the FFT
literature states it unusually bluntly.

Frigo & Johnson, on why FFTW's `ESTIMATE` mode (which minimises a static cost function) is
inferior to measurement:

> "This penalty reinforces a conclusion of [3]: there is no longer any clear connection
> between operation counts and FFT speed, thanks to the complexity of modern computers."
> — Frigo & Johnson 2005, §V-B

The scale of the mismatch, from the same authors' book chapter:

- Forty years of theory moved the power-of-two arithmetic count from radix-2 Cooley–Tukey's
  ~5 n log₂ n down to the lowest known ~(34/9) n log₂ n — **"remains only about 25%."**
- Over the same period, implementation quality is worth **a factor of 5–40**: Figure 1 of
  that chapter plots FFTW 3.1.2 against the *Numerical Recipes in C* radix-2 routine on a
  3 GHz Intel Core Duo (Intel C compiler 9.1.043, single-precision complex), and "the former
  is faster by a factor of 5–40 (with a larger ratio as n grows)." The no-SSE line is
  included precisely so the comparison is at similar arithmetic cost.
  — Johnson & Frigo, *Implementing FFTs in Practice*, §1 and Fig. 1

SPIRAL measured the same thing directly, and more damningly, on a *small* transform where
cache cannot be blamed (details and platform in §3.2): across 10 000 randomly generated
equivalent formulas the **arithmetic-cost spread was under 10% while the runtime spread was
about 2×**, and

> "Surprisingly, the formulas with lowest arithmetic cost yield both slowest and fastest
> runtimes, which implies that arithmetic cost is not a predictor of runtime in this case."
> — Püschel et al. 2005, §VII (Performance spread)

**What this buys us.** It converts "which algorithm has fewest flops for L=17" from the
central question into a screening heuristic. **When it stops paying:** it does *not* license
ignoring arithmetic cost entirely — a 4× flop difference will still show up. It licenses
ignoring the last 10–25%, which is exactly the margin most hand-derived
minimum-multiplication modules are fighting over.

---

## 2. FFTW's planner

### 2.1 Mechanism

FFTW represents a transform as a *problem* and an implementation as a *plan* — a composition
of algorithmic steps, each of which may recursively spawn sub-problems. The planner's
principle is stated in one sentence:

> "the basic principle behind the FFTW planner is straightforward: construct a plan for each
> applicable algorithmic step, time the execution of these plans, and select the fastest one.
> Each algorithmic step may break the problem into subproblems, and the fastest plan for each
> subproblem is constructed in the same way."
> — Johnson & Frigo, *Implementing FFTs in Practice*, §4.3

Two structural points matter for us:

- **The plan space is the binding constraint, not the search.** "Because only problems that
  can be expressed can be solved, the representation of a problem determines an upper bound
  to the space of plans that the planner can explore, and therefore it ultimately constrains
  FFTW's performance." (Frigo & Johnson 2005, §IV). If a candidate implementation is not
  expressible, no amount of measurement will find it.
- **Dynamic programming plus memoisation, not exhaustive search.** "A direct implementation
  of this approach, however, faces an exponential explosion of the number of possible plans."
  DP optimises each sub-problem locally and re-uses it, at the cost of correctness: "Dynamic
  programming is not guaranteed to find the fastest plan, because the performance of plans is
  context-dependent on real machines (e.g., the contents of the cache depend on the preceding
  computations)." Memoisation is done on a **128-bit MD5 hash** of the problem plus a pointer
  to the solver that produced the plan, rather than storing problems and plans, to save
  memory; a hash collision is harmless because the solver is re-invoked and either produces a
  valid plan or fails. (Frigo & Johnson 2005, §IV-E)

The plan space includes categories that are easy to overlook and that turn out to matter:
rank-0/rank-1 plans, higher-rank and **vector-rank** plans (the batch loop!), *indirect*
plans (copy first, then transform in place), buffered plans, Rader plans, Bluestein plans,
and — relevant below — **generic Θ(n²) plans**.

### 2.2 The four rigor flags, verbatim

From the FFTW 3.3.10 manual (Planner Flags):

- `FFTW_ESTIMATE` — "specifies that, instead of actual measurements of different algorithms,
  a simple heuristic is used to pick a (probably sub-optimal) plan quickly."
- `FFTW_MEASURE` — "tells FFTW to find an optimized plan by actually computing several FFTs
  and measuring their execution time"; it is the default.
- `FFTW_PATIENT` — "is like FFTW_MEASURE, but considers a wider range of algorithms and often
  produces a 'more optimal' plan," at the cost of longer planning.
- `FFTW_EXHAUSTIVE` — "is like FFTW_PATIENT, but considers an even wider range of algorithms."
- `FFTW_WISDOM_ONLY` — "a special planning mode in which the plan is only created if wisdom
  is available for the given problem, and otherwise a NULL plan is returned."

`fftw_set_timelimit()` degrades gracefully: under a time limit with `FFTW_PATIENT`, planning
progresses through ESTIMATE, then MEASURE, then PATIENT within the budget.

The research paper's three internal modes map onto these and are described mechanically
(Frigo & Johnson 2005, §V-B):

- **patient**: "tries essentially all combinations of the possible plans, with dynamic
  programming."
- **impatient**: eliminates possibilities that "inordinately increase planner time relative
  to their observed benefits." Concretely: only one way to decompose multi-dimensional N or V
  is considered, **vector recursion is disabled**, and the time for a vector loop of ℓ
  transforms is approximated as ℓ × (time for one). "Altogether, impatient mode often
  requires a factor of 10 less time to produce a plan than the full planner."
- **estimate**: no measurements; minimises "the number of floating-point operations plus the
  number of 'extraneous' loads/stores (such as for copying to buffers)." "This can reduce the
  planner time by several orders of magnitude."

### 2.3 How much does MEASURE buy over ESTIMATE — the published numbers

**Measurement 1 (Frigo & Johnson 2005, Fig. 9 and §V-B).** Double-precision 1-D complex
DFTs, power-of-two sizes 2–2¹⁸, on a **2 GHz PowerPC 970 (G5)**:

- estimate mode: **median speed penalty 20%, maximum 72%**
- impatient mode: **maximum penalty 11%**
- but impatient's penalty is not uniformly small: **47% for a 1024 × 1024 2-D complex
  transform** on the same machine, "since vector recursion proves important there for the
  discontiguous (row) dimension of the transform."

**Measurement 2 (Johnson & Frigo, *Implementing FFTs in Practice*, §4.3)** — the more useful
form for us, because it is resolved by size:

> "a penalty of 20% is typical for moderate n ≲ 2¹³, whereas a factor of 2–3 can be suffered
> for large n ≳ 2¹⁶"

**Mapping to L=6/8/17/36.** Counting total points per volume: 216 (2⁷.⁸), 512 (2⁹),
4 913 (2¹².³), 46 656 (2¹⁵.⁵). L=6, 8 and 17 sit comfortably in the "≲ 2¹³ → ~20%" band.
L=36 sits between 2¹³ and 2¹⁶, i.e. entering the band where the FFTW authors saw the penalty
grow toward 2–3×. **Read this as: search is worth the most at L=36, and is worth roughly a
fifth of runtime at L=6/8/17.** A fifth is not noise.

The same passage flags something important for a fixed-size project:

> "Coming up with a better heuristic plan is an interesting open research question; one
> difficulty is that, because FFT algorithms depend on factorization, knowing a good plan for
> n does not immediately help one find a good plan for nearby n."

So there is no transfer between our four sizes. 6, 8, 17 and 36 are four independent tuning
problems. (Which is also an argument that four separate specialised codelets are the right
shape of deliverable.)

### 2.4 Wisdom, and why plans do not travel

`wisdom` is FFTW's mechanism for moving the search offline: "whenever you create a plan, the
FFTW planner accumulates wisdom, which is information sufficient to reconstruct the plan," it
is "cumulative, and is stored in a global, private data structure managed internally by
FFTW," and it can be exported/imported via `fftw_export_wisdom_to_filename` /
`fftw_import_wisdom_from_filename`, with `fftw_forget_wisdom()` to clear it.

The hard measurement about portability (Frigo & Johnson 2005, Fig. 10 and §V-B), 1-D complex
transforms on a 2 GHz PowerPC 970 (G5) and a 2.8 GHz Pentium IV, each running the other's
optimal plan:

> "In both cases, using the wrong machine's plan imposes a speed penalty of 20% or more for
> at least 1/3 of the cases benchmarked, up to a 40% or 34% penalty for the G5 or Pentium IV,
> respectively."

And the flat statement: "It is critical to create a new plan for each architecture."

Two practical hazards from the FFTW FAQ, both of which will bite a monitor agent:

- "planning with `FFTW_MEASURE` or `FFTW_PATIENT` overwrites the input/output arrays"
  (Q3.17) — initialise input data *after* plan creation.
- Planning can "take several seconds" (Q3.3), while `FFTW_ESTIMATE` "uses heuristics instead
  of runtime measurements and produces a good plan in a short time."

### 2.5 The planner finds things nobody predicted

Worth reading in full if you are tempted to hand-derive the composition. For an out-of-place
size-65536 DFT on the Pentium IV the chosen plan was DIT radices 32, then 8, then 16, then a
direct size-16 codelet — except that the first step used *buffered* DIT, its size-32 vector
loop was pushed down to the leaves by vector recursion, and the size-16 codelet was preceded
by an *indirect* plan that copies input to output so the codelet can run in-place on
contiguous data. The G5 plan for the same size was structurally different again. The authors'
own summary:

> "indirect plans for large out-of-place DFTs were initially a surprise (and often boosted
> speed by 20% or more)." … "While 'explanations' can usually be fabricated in hindsight, we
> do not really understand the planner's choices because we cannot predict what plans will be
> produced. Indeed, this is the whole point of implementing a planner."
> — Frigo & Johnson 2005, §V-C

They also record two specific preconceptions that measurement falsified: that transposes for
in-place DFTs would be grouped at an intermediate point with an explicit DIF step (in fact
they are "almost always used at the leaves with a direct codelet"), and that a low-stride
vector loop would always be pushed all the way to the leaves (in fact "the loop is only
pushed one or two levels down").

**Transferable lesson for us:** the two decisions the planner kept surprising its own authors
about — *where to put the vector/batch loop*, and *whether to copy to contiguous scratch
before transforming* — are precisely the two decisions a batched 3-D fixed-size transform has
to make. Do not settle them by argument. See §7.

---

## 3. SPIRAL: search over formula space

SPIRAL and FFTW are the two systems that define this area, and they differ in a way that
matters to us. Frigo & Johnson's own comparison:

> "SPIRAL searches at compile-time over a space of mathematically equivalent formulas
> expressed in a 'tensor-product' language, whereas FFTW searches at runtime over the
> formalism discussed in Section IV, which explicitly includes low-level details, such as
> strides and memory alignments, that are not as easily expressed using tensor products.
> SPIRAL generates machine-dependent code, whereas FFTW's codelets are machine-independent.
> FFTW's search uses dynamic programming, while the SPIRAL project has experimented with a
> wider range of search strategies including machine-learning techniques."
> — Frigo & Johnson 2005, §I

**Compile-time search over a fixed size is our situation exactly.** SPIRAL is therefore the
closer model.

### 3.1 The five search methods, and why DP wins by default

SPIRAL's search operates on *ruletrees* (a partially expanded derivation) rather than on
formulas, so that the search is transform-independent. The five methods (Püschel et al. 2005,
§VI-A), quoted:

- **Exhaustive** — "enumerates all formulas in the formula space and picks the best. Due to
  the large formula space, this is only feasible for very small transform sizes."
- **Random** — "enumerates a fixed number of random formulas and picks the best. Since fast
  formulas are usually rare, this method is not very successful."
- **Dynamic programming** — "For most problems, it is our method of choice." Default.
- **Evolutionary search** (cross-breeding + mutation on ruletrees) — "particularly useful in
  cases where dynamic programming fails."
- **Hill climbing** — "has proven to be inferior to the latter."

Why DP: for a DFT of size 2ⁿ using only Cooley–Tukey, the number of formulas grows like the
count of binary trees (Catalan-like, via Stirling's formula) while **DP visits only a
polynomial number of nodes**. (The exact exponents are typeset as math in the source and my
text extraction did not recover them; the structural claim — exponential space, polynomial DP
visit count — is stated in plain prose in §VI-A and is what matters.) The catch is the same
one FFTW documents: "The inherent assumption of DP is that the best code for a transform is
independent of the context in which it is called. This assumption holds for the arithmetic
cost … but not for the runtime of transform algorithms."

SPIRAL also runs DP over an *implementation* knob, not just the algorithm: "the selection of
implementation options, which, in the current version, is the degree of unrolling." Unroll
depth is a searched parameter, globally and per-node, with a default global unrolling
threshold (16 in the worked example in §IV, where "the generated code will be completely
unrolled").

### 3.2 The performance spread — the headline numbers

This is the most quantitatively useful passage in the autotuning literature for our purposes,
because the first example is a *small* transform.

**Example 1 — a small DCT-2, platform p4-3.0-lin, scalar code** (Püschel et al. 2005, §VII
and Fig. 8):

- SPIRAL reports **1 639 236 012 different formulas** for this single small transform.
- A random subset of **10 000** was generated and timed. Results:
  - **runtime spread ≈ 2×**
  - **assembly-instruction-count spread ≈ 1.5×**
  - **arithmetic-cost spread < 10%**
  - FMA-optimised arithmetic cost: lower count but **spread rises to about 25%** — "different
    formulas are differently well suited for FMA architectures"
  - accuracy with constants truncated to 8 bits: **spread about a factor of ten**, most
    formulas clustered within 2×
- And the crucial disclaimer: "for transforms of this size and on this platform **no cache
  problems arise**."
- The authors' reaction: "The large spread in runtime and assembly instruction counts is
  surprising given that each implementation is high-quality code that underwent SPL and C
  compiler optimizations."

**Example 2 — a large DFT of size 2ᵏ (exponent typeset as math, not recovered by extraction),
platform p4-3.0-win, Cooley–Tukey rule only, 20 000 random formulas** (Fig. 9):

- scalar and SSE vector code both show **runtime spread ≈ 5×**, most formulas within 3×
- **"The best 30% formulas are scarce."**
- the vector histogram "looks very much like a translation to the left" of the scalar one —
  vectorisation helped all 20 000 formulas including the slowest, but by differing amounts

**Conclusion the authors draw**, and it is the justification for this whole document:

> "Although different formulas for one transform have a similar operation count, their scalar
> or vector code implementations in SPIRAL have a significant spread in runtime. This makes a
> strong case for the need of tuning implementations to platforms, including proper algorithm
> selection."

**Why this matters for L=6 and L=8 specifically.** One could argue that at 216 and 512 points
the whole volume is L1-resident so there is nothing to tune. Example 1 refutes that: a
2× runtime spread among 10 000 equivalent formulas, on a transform small enough that "no
cache problems arise," with under 10% arithmetic spread. The spread at small sizes comes from
schedule, register pressure and instruction mix — not from cache. That is exactly the regime
L=6 and L=8 live in.

### 3.3 Tuning transfer: same ISA, different generation, different vector width

From the platform-tuning experiment (Püschel et al. 2005, §VII, Fig. 15), DFT of size 2¹⁰
tuned on one configuration and run on another, slowdown relative to code tuned in place:

- **Same platform, different data type:** the best algorithm found for `p4-2.53-win` **SSE2**,
  when implemented in **SSE**, "performs up to **320% slower**" than the SSE-tuned
  implementation. Reason given: "the different vector length of SSE2 and SSE (2 versus 4),
  which requires very different algorithm structures."
- **Same data type, different platform generation:** code tuned on `p3-1.0-win` SSE run on
  the binary-compatible `p4-2.53-win` SSE "performs up to **50% slower**" than SSE code tuned
  for the P4.

Also reported: "Most of the 12 ruletrees in this table are different from each other, meaning
that SPIRAL finds different trees when searching for the best tuned formula for a given
machine," and it found *balanced* trees for the P3 and Athlon XP but *unbalanced* trees for
the P4.

**Direct implication.** The 320% figure is a vector-width effect, and our project spans AVX2
(4 doubles) and AVX-512 (8 doubles). Treat AVX2 and AVX-512 as *different tuning targets with
possibly different algorithm structures*, not as the same code with a wider vector type.

### 3.4 Compiler flags are part of the search space

Also from Püschel et al. 2005, §VII, using ACOVEA (an evolutionary compiler-flag searcher) on
SPIRAL-generated DCT-2 code on `p4-3.0-lin`:

- Motivation, quantified: "the extreme case is gcc 3.3 with a total of more than 500 different
  documented flags, more than 60 of which are related to optimization," options interact
  non-trivially, and "the best options usually depend on the program being compiled."
- "ACOVEA gives an additional speedup ranging from **8% to 15%** for the relevant larger DCT
  sizes in this experiment" — over a baseline that included `gcc -O3`.
- Note also: "`gcc -O3` is always slower than `gcc -O1`" for some of these codes, "which means
  that some of the more advanced optimizations can make the code slower."
- Flag-space spread: an initial random population of **2000** flag combinations (each
  containing at least `-O1 -march=pentium4`) produced a runtime histogram with a **spread of
  more than a factor of three**.

**When this stops paying:** it is an 8–15% effect layered on already-good code, on gcc 3.3-era
compilers. But the "-O3 slower than -O1" and the >3× spread are still live hazards, and they
interact with genfft's warning below: some optimising compilers "will tend to greatly re-order
the code, destroying FFTW's optimal schedule. With GNU gcc, we circumvent this problem by
using compiler flags that explicitly disable certain stages of the optimizer." (Johnson &
Frigo, footnote 9). For fully-unrolled straight-line codelets — i.e. all four of our sizes —
**flags are a first-class search dimension, and more optimisation is not monotonically
better.**

### 3.5 Cost of the search

- "Problem sizes around 64 are optimized within a few minutes."
- "the generation of a scalar DFT library for two-powers up to 2ᴺ is done in 20–30 min on a
  Pentium 4, while the corresponding vector code generation takes on the order of hours."
- "SPIRAL requires only compile-time adaptation; thus, at runtime, no time is spent in further
  optimizing the code."
- Their comparison to FFTW: "in FFTW, real code generation (i.e., from scratch) is done only
  for small transform sizes and for unrolled code. These codelets … are pregenerated and
  distributed with the package. Further, the codelet generation is deterministic, i.e.,
  produces the same result independently of the machine."

At our sizes, search cost is minutes, not hours. That is a strong argument for near-exhaustive
search rather than DP.

---

## 4. ATLAS-style empirical tuning as the general pattern — and its strongest critique

ATLAS generalised the idea beyond transforms: rather than shipping code, ship a *generator*
plus a *search* and let the install-time search pick parameters. The Netlib project page
describes it as automating "the generation and optimization of numerical software for
processors with deep memory hierarchies and pipelined functional units," focused on BLAS and
initially DGEMM.

The most precise published account of ATLAS's actual search procedure that I could fetch is
Yotov et al.'s reimplementation study, which is useful because it enumerates the mechanism
rather than the marketing. ATLAS uses **orthogonal line search**: an n-dimensional
optimisation approximated by "solving a sequence of n 1-dimensional optimization problems,"
using reference values for parameters not yet optimised. "Orthogonal line search is heuristic
because it does not necessarily find the optimal value even for a convex function, but with
luck, it might come close." Its parameter order is:

> 1) Find best NB. 2) Find best MU and NU. 3) Find best KU. 4) Find best Ls.
> 5) Find best FF, IF, and NF. 6) Find best N_CNB: a non-copy version of NB.
> 7) Find best clean-up codes.

Concrete search-space bounds from the same paper: NB is restricted to multiples of 4 with
`16 ≤ NB ≤ min(80, √C1)`; for each NB, "ATLAS tries two extreme cases for KU — no unrolling
(KU = 1) and full unrolling (KU = NB)"; N_CNB is searched downward from NB in steps of 4 and
"ATLAS terminates the search when the performance falls by 20% or more from the best
performance it finds during this search"; K clean-up code generation stops "when the
performance of the general version falls within 1% of the performance of the current
specialized version."

Two ATLAS design choices are worth stealing verbatim:

- **Measure rather than model the instruction cache.** "Rather than estimate the size of the
  instruction cache directly by running a micro-benchmark and using that to determine the
  amount of unrolling, ATLAS generates a suite of mini-MMM kernels with different KU values,
  and selects the kernel that achieves best performance." For fully-unrolled FFT codelets,
  I-cache pressure is *the* scaling limit, and this says: don't compute it, measure it.
- **Specialise the fixed-shape case and unroll it fully.** ATLAS generates specialised
  clean-up kernels precisely because "Full unrolling is possible because the shapes of the
  operands are completely known." That is our situation for all four sizes, all the time.

### 4.1 The critique: is search actually necessary?

Yotov et al. replaced ATLAS's global search engine with an analytical model, holding the code
generator fixed so that "any differences in the performance of the code produced by the two
systems can come only from differences in optimization parameter values." Their finding:

> "Our experiments show that model-driven optimization can be surprisingly effective, and can
> generate code with performance comparable to that of code generated by ATLAS using global
> search."

With a cost comparison that should give any autotuner pause:

> "while ATLAS CGw/S spends considerable amount of time, ranging between 8 minutes on the DEC
> Alpha to more than 8 hours on the Intel Itanium 2, to find optimal values for optimization
> parameters, the model-based approach takes no measurable time."

It is not a clean win — on the AMD Athlon MP "the mini-MMM code generated by ATLAS Model runs
roughly 20% slower than the code generated by ATLAS CGw/S," traced to badly chosen register
tile values. And they close by noting "there is still a significant gap in performance between
the code generated by ATLAS CGw/S and the vendor BLAS routines."

**How to use this.** It is the honest counterweight to §2 and §3: search is not sacred, and a
good structural model gets most of the way for free. FFTW's own architecture agrees, from a
different direction — see §6.1, where a *model* (a cache-oblivious schedule) is reported to
match a *search* (SPIRAL's) at codelet level. **The defensible position for this project:
derive the codelet structure from a model, then search the small residual space (unroll depth,
schedule variant, loop order, flags) empirically, because at our sizes that search costs
minutes.**

---

## 5. Learned and ML approaches to plan selection

### 5.1 Singer & Veloso: learning to construct fast formulas

The canonical ML-for-transform-search work, done inside SPIRAL. Framing: "a single signal
processing algorithm can be represented by many mathematically equivalent formulas, but when
implemented in code and run on real machines, they have very different runtimes, which is
extremely difficult to model, and the space of formulas for real signal transforms is so large
that exhaustive search is impossible." They treat it "as a control learning problem."

Search-space sizes they report (§2.5):

- FFT split trees, with their restrictions: "at size 2¹⁴ there are still **2 449** different
  split trees and at size 2¹⁸ there are **70 376**."
- WHT(2ⁿ) has "on the order of θ((4+√8)ⁿ / n^{3/2}) different possible formulas"; binary
  split trees alone number "on the order of θ(5ⁿ / n^{3/2})". "For example, WHT(2⁸) has
  **16 768** different split trees."

Runtime spreads they measured:

- All DFT(2¹⁸) formulas: "about a factor of **4** difference in runtimes between the fastest
  and slowest formulas. Further there are relatively few formulas that have the fastest times."
- All binary WHT(2¹⁶) split trees with no size-2¹ leaves, on a Pentium III 450 MHz / Linux
  2.2.5-15: runtimes and L1 data cache misses "vary considerably differing by about a factor
  of **6** and **10** respectively," and "The formula with the fastest runtime also has the
  minimal number of cache misses."

Results of the learned generator (Püschel et al. 2005, §VI-B, reporting the same work):

- WHT: "In all cases, the fastest ruletree for a given WHT transform size was generated in the
  first **50** formulas produced." "Except for a few cases on the Sun, the very first ruletree
  generated by our method had a runtime **within 6%** of the fastest runtime," and "in all but
  one case, at least **40 of the 100** fastest ruletrees known to us were generated as one of
  the first 100."
- DFT on Pentium: "the fastest ruletree being generated usually within the first **20** and
  often as the very first ruletree. Further, the first ruletree to be generated had a runtime
  always **within 15%** of the runtime of the fastest formula."

### 5.2 Modern learned autotuning: what transfers

- **AutoTVM** (Chen et al., NeurIPS 2018) is the current template for learned cost models.
  Search space "on the order of billions of possible implementations for a single GPU
  operator." It uses a learned cost model (gradient-boosted trees over hand-designed context
  features, and a Context-Encoded TreeGRU over the loop AST) with a **rank loss** rather than
  regression, explores with **parallel simulated annealing** using the model as energy
  function, and measures a batch on real hardware each round. Results: "Both the GBT and
  TreeGRU models outperformed the black-box methods and found operators that were **2× faster
  than those found with random searches**" at equal trial budget (~800 hardware trials on an
  NVIDIA TITAN X); "using a rank-based objective was slightly better than using a
  regression-based one"; end-to-end "improvements ranging from **1.2× to 3.8×** over existing
  frameworks."
- **GenMAT** (Zhang et al., PEHC 2021) is directly about *ranking* candidate variants, which
  is what a monitor agent does. Tuning Halide schedules — "GenMAT needs to select the best
  schedule from thousands of candidate Halide schedules" — it reports Spearman rank
  correlation between predicted and true runtime of **ρ = 0.94** (Halide Blur) and
  **ρ = 0.96** (Halide FFT), with per-machine values 0.99/0.95/0.95 for Halide FFT on
  Xeon/i7/i5. Inference cost: "GenMAT can inference on 5000 candidate variants to select the
  best variant in 0.75 milliseconds, resulting in 0.15 microseconds per candidate." They also
  show **accelerated profiling** — train on small inputs (2¹⁰–2¹³), predict on large
  (2¹⁴–2¹⁵) — cutting profiling time 12.9× (Xeon) and 10.2× (i5) while ρ stays at 0.86/0.78.
- **OpenTuner** (Ansel et al., PACT 2014) is the generic harness. Its two ideas worth
  borrowing: (i) do not prune the space by hand — "for many problems excessive search space
  pruning will miss out on optimal solutions," and they operate "in massively large search
  spaces, exceeding 10³⁶⁰⁰ possible configurations"; (ii) run an **ensemble** of search
  techniques and allocate budget adaptively via "the multi-armed bandit with sliding window,
  area under the curve credit assignment (AUC Bandit)" — "techniques that perform well will
  dynamically be allocated a larger proportion of tests." Reported result: autotuners for 7
  projects / 16 benchmarks with "speedups over existing techniques of up to 2.8×", including
  "up to 2.8x speedup over -O3" for GCC flag selection.

**Honest assessment for this project.** A learned cost model is not worth building for four
fixed sizes: the whole point of a learned model is to amortise measurement across many
distinct problems, and we have four. What *does* transfer is (a) the rank-not-regress
objective — the monitor agent's job is to order candidates correctly, not to predict
nanoseconds; (b) the AUC-bandit idea of reallocating a fixed timing budget toward candidates
that are still winning; and (c) Singer & Veloso's finding that the fastest formula tends to
appear in the first tens of well-chosen candidates, which means a modest guided search is
likely sufficient.

---

## 6. Tuning for a single fixed size known at compile time

This is our situation. The literature on it is smaller than the general-autotuning literature
but it is unusually concrete, because it is mostly the literature of *codelet generation*.

### 6.1 All four of our line lengths are in the straight-line-code regime

FFTW's own size threshold, stated three ways:

- "When the DFT rank-1 problem is 'small enough' (usually, **n ≤ 64**), FFTW produces a
  *direct plan* … a codelet specialized to solve problems of one particular size."
- "A typical codelet in FFTW computes a DFT of a small, fixed size n (usually, **n ≤ 64**),
  possibly with the input or output multiplied by twiddle factors."
- Radices are drawn from the same set: "(r ≤ 64) produced by the codelet generator."
  — Johnson & Frigo, *Implementing FFTs in Practice*, §§4.2, 5

**6, 8, 17, 36 are all ≤ 64.** Every one of our 1-D line transforms is, in FFTW's own
architecture, a *leaf*: hard-coded, fully unrolled, no recursion, no planner decision.
This is the single most useful structural fact in this document.

What such a codelet looks like, with numbers:

- "a codelet for (e.g.) n = 64 is **~2000 lines** long, with hundreds of variables and **over
  1000 arithmetic operations** that can be executed in many orders, so what order should be
  chosen?"
- From the earlier genfft paper: "the codelet that performs a DFT of size 64 is used routinely
  by FFTW on the Alpha processor. The codelet is about **twice as fast as Digital's DXML
  library** on the same machine. The codelet consists of about **2400 lines of code, including
  912 additions and 248 multiplications**. Writing such a program by hand would be a
  formidable task for any programmer. At least for the DFT problem, these long sequences of
  straight-line code seem to be necessary in order to take full advantage of large CPU
  register sets and the scheduling capabilities of C compilers."
  — Frigo 1999, §1

The ordering problem, and the resolution that lets a *model* beat a *search*:

> "The key problem here is the efficient use of the CPU registers, which essentially form a
> nearly ideal, fully associative cache. Normally, one relies on the compiler for all code
> scheduling and register allocation, but the compiler needs help with such long blocks of
> code (indeed, the general register-allocation problem is NP-complete). In particular, FFTW's
> generator knows more about the code than the compiler — the generator knows it is an FFT,
> and therefore it can use an optimal cache-oblivious schedule … to order the code independent
> of the number of registers. The compiler is then used only for local 'cache-aware' tuning."
> — Johnson & Frigo, §3.3/§5

And the payoff claim, which is the sharpest "model beats search" datapoint in the FFT
literature:

> "As a practical matter, one consequence of this scheduler is that FFTW's machine-independent
> codelets are no slower than machine-specific codelets generated by SPIRAL [39, Figure 3]."

The theoretical backing, from the genfft paper, §6: the scheduler "produces a topological sort
of the dag in an attempt to maximize register usage. **For transforms whose size is a power of
4**, we prove that a schedule exists that is asymptotically optimal in this respect, even
though the schedule is independent of the number of registers. This fact is derived from the
red-blue pebbling game of Hong and Kung [HK81]." The lower bound cited is Aggarwal & Vitter's,
generalised from registers to block I/O, and Frigo notes "The same result holds for any
two-level memory, such as L1 cache vs. L2, or physical memory vs. disk." Crucially for us,
from the paper's own overview: **"For transforms of other sizes the scheduling strategy is no
longer provably good, but it still works well in practice."**

> **Mapping — and note this cuts against us.** The provable case is size = a *power of 4*
> (4, 16, 64). **None of 6, 8, 17, 36 is a power of 4.** So for all four of our line lengths
> the genfft schedule is a heuristic with no optimality guarantee, and the schedule is
> therefore a legitimate — indeed the *primary* — thing to search over at every one of our
> sizes. L=8 does not get a free pass here just because it is a power of two.
>
> (Honesty note: the paper's §1 overview renders this size condition in a math font that my
> PDF text extraction garbled. I read the condition off the plain-prose statement at the top
> of §6, quoted above, which is unambiguous. An earlier draft of this document wrongly said
> "size 2ᵏ" on the basis of the garbled glyph; that was wrong and is corrected here.)

Two more concrete, transferable codelet-level findings:

- **Eliminate negative constants.** "multiplicative constants in FFT algorithms often come in
  positive/negative pairs, but every C compiler we are aware of will generate separate load
  instructions for positive and negative versions of the same constants. We thus obtained a
  **10–15% speedup** by making all constants positive, which involves propagating minus signs
  to change additions into subtractions or vice versa elsewhere in the dag."
- **Prefer one big hard-coded transform over a loop of smaller ones.** "Is it better to
  implement a hard-coded FFT of size 64, for example, or an unrolled loop of four size-16
  FFTs, both of which operate on the same amount of data? The former should be more efficient
  because it performs more computations with the same amount of data, thanks to the log n
  factor in the FFT's n log n complexity." Singer & Veloso measured the same preference
  empirically from the other end: "We have found that the fastest formulas never have leaves
  of size 2¹ since it is beneficial to use unrolled code of larger sizes."

### 6.2 The one paper that searched exactly our problem

Singer & Veloso, building their leaf kernels (§2.4):

> "we constructed leaves (similar to those in the WHT package) that are highly optimized
> implementations of small sized FFTs. Specifically, we **performed an exhaustive search over
> all possible split trees of sizes 2² to 2⁷, implementing them in unrolled, straight-line
> code.**"

That is precisely the L=8 problem (and the shape of the L=6/17/36 problem): a *fixed, small,
fully-unrolled straight-line kernel, chosen by exhaustive search*. It was tractable in 2002
hardware and software. It is trivially tractable now.

They also record a timing hazard that will bite us directly (§2.4):

> "It can also use performance counters to keep track of the amount of runtime spent in
> computing different portions of the split tree. … Unfortunately due to overhead, **the
> timings become inaccurate for small sized nodes**, particularly those of size 2¹. To avoid
> this problem, we prevented the system from constructing split trees with nodes of size 2¹."

Their fix was to change the algorithm space. Ours must instead be to fix the *measurement* —
see §8. A single L=6 volume is 216 points; if the whole transform takes a few hundred
nanoseconds, it is within an order of magnitude of the timing overhead documented in §8.2.
**Batching is not just the application's requirement, it is a measurement necessity.**

### 6.3 Search space size, at fixed small size, with the caveat

Do not conclude from "the size is small" that "the space is small." SPIRAL reported
**1 639 236 012 formulas for a single small DCT-2** (§3.2). Singer & Veloso reported 2 449
split trees at size 2¹⁴ *after* restricting the rule set to Cooley–Tukey only and forbidding
size-2 leaves. The space is combinatorial in the derivation, not in the data.

The practical consequence is the one SPIRAL states: exhaustive search "is only feasible for
very small transform sizes," and what makes our case tractable is not the size alone but the
fact that we can *restrict the rule set on principled grounds* — which is exactly what Singer
& Veloso did to get from θ(5ⁿ/n^{3/2}) down to something searchable.

### 6.4 L=17 specifically: the prime case, with a sourced warning and a sourced surprise

Three findings from the FFTW authors bear directly on L=17.

**(a) A naive Θ(n²) DFT is a genuine contender at n = 17.** FFTW's plan space includes, for
prime sizes: "*Rader plans* implement the algorithm from Ref. 32 to compute one-dimensional
DFTs of prime size in Θ(n log n) time. *Bluestein plans* implement Bluestein's 'chirp-z'
algorithm, which can also handle prime n in Θ(n log n) time. **Generic plans implement a naive
Θ(n²) algorithm (useful for n ≲ 100).**" (§4.2.5). FFTW keeps the O(n²) kernel in the plan
space *specifically for sizes like 17*, and lets the planner choose. A 17×17 dense complex
matrix–vector product is 289 complex MACs with perfectly regular, contiguous, trivially
vectorisable access and zero permutation logic. **Benchmark it. On FFTW's own judgement it is
in the running.** (For the 3-D batched case it is even more attractive than the 1-D case,
because the 17×17 twiddle matrix stays resident and is reused across every line of every
volume.)

**(b) Winograd is explicitly advised against.** "There is also the Winograd FFT, which
minimizes the number of multiplications at the expense of a large number of additions; **this
trade-off is not beneficial on current processors that have specialized hardware
multipliers.**" (§2, *Implementing FFTs in Practice*). This is a 2008 statement about
processors far weaker than an AVX-512 machine with two FMA pipes; if anything the argument has
strengthened. Combined with §1 (arithmetic cost is not a runtime predictor) this is about as
clear a "do not spend your week on a minimum-multiplication 17-point module" as the literature
gives.

**(c) But do derive the module symbolically rather than transcribing a table.** genfft's
headline result was on a prime-size Rader codelet. Frigo, §1:

> "the generator employs an algorithm due to Rader, in the form presented by Tolimieri and
> others. In its most sophisticated variant, this algorithm performs **214 real
> (floating-point) additions and 76 real multiplications**. (See [TAL97, Page 161].) The
> generated code in FFTW for the same algorithm, however, contains only **176 real additions
> and 68 real multiplications**, because genfft found certain simplifications that the authors
> of [TAL97] did not notice."

The footnote gives the mechanism — patterns of the form `t1 = a+b; t2 = a-b; u = t1+t2`
collapsing to `u = 2a` when `t1, t2` are dead elsewhere — and adds "[SB96] reports an algorithm
with 188 additions and 40 multiplications, using a more involved DFT algorithm that I have not
implemented yet. To my knowledge, the program generated by genfft performs the lowest known
number of additions for this problem."

*Caveat, stated because citation discipline requires it:* the transform size in that passage is
typeset in a math font that my PDF text extraction rendered as garbage, so **I do not assert
which prime it was.** The transferable lesson does not depend on the size: mechanical
common-subexpression elimination and algebraic simplification over the dag beat a
carefully-published hand count by ~18% in additions and ~11% in multiplications. If you build
a 17-point module, generate and simplify it, then count; do not copy a table.

Also relevant: genfft "produces real variants of the Rader's algorithm mentioned above, which
to my knowledge do not appear anywhere in the literature," and FFTW3's real-input prime path
"uses an adaptation of Rader's algorithm that reduces the storage and time requirements roughly
by a factor of two with respect to the complex case … which to our knowledge has not been
published before" (it reduces the real DFT to a DHT, then runs a DHT variant of Rader). We are
doing complex forward transforms, so this is not directly usable — but it is evidence that the
prime-size design space is not exhausted by the textbook presentation.

### 6.5 Fixed-size specialisation as the general technique

The recurring pattern across FFTW, SPIRAL and ATLAS, when the size is known ahead of time:

- **Precompute the factorisation.** "eq. (2) entails a run-time factorization of n, which can
  be precomputed if n is known in advance." (Johnson & Frigo, §5)
- **Break the complex abstraction.** "eq. (2) operates on complex numbers, but breaking the
  complex-number abstraction into real and imaginary components turns out to expose certain
  non-obvious optimizations." (ibid.)
- **Drop multiplications by 1** and let CSE run over the whole dag; the simplifier is powerful
  enough that real-data specialisations fall out automatically "to match the lowest known
  operation count for a real-input FFT starting only from the complex-data algorithm." (ibid.)
- **Unroll fully and schedule deliberately**, because "the shapes of the operands are
  completely known" (Yotov et al. on ATLAS clean-up code) and because straight-line code is
  how you reach the register file (Frigo 1999, §1).
- **Move the search offline.** This is what FFTW wisdom is for, what SPIRAL does by
  construction ("SPIRAL requires only compile-time adaptation; thus, at runtime, no time is
  spent in further optimizing the code"), and what our project does by fiat.

[UNVERIFIED — could not fetch] Blake, Witten & Cree's FFTS ("The Fastest Fourier Transform in
the South," IEEE Trans. Signal Processing 61(19), 2013) takes the complementary approach of
*runtime* specialisation — generating machine code once the transform parameters are fixed,
with small leaves and no large codelet library and no machine-specific calibration. I could not
retrieve the paper text (403/404/410 on every mirror tried), so I am not citing any of its
numbers. Flagged only so a later reader knows this branch of the literature exists.

---

## 7. What the search space actually is for L=6/8/17/36

This section synthesises the sourced material above into the specific decisions our
implementers face. The mapping arguments are mine; the facts they rest on are cited above.

**Decisions the literature says are settled for us:**

- **No recursion.** All four line lengths are ≤ 64, i.e. inside FFTW's direct-codelet regime
  (§6.1). Fully unrolled straight-line code per line transform.
- **Not Winograd for 17** (§6.4b), and not "minimum multiplications" as an objective anywhere
  (§1).
- **Tune per ISA.** AVX2 vs AVX-512 is a vector-width change, and vector width was worth up to
  320% in SPIRAL's transfer experiment (§3.3).

**Decisions the literature says must be measured, with the source of the warning:**

| Decision | Why it must be measured | Source |
|---|---|---|
| Where the batch loop goes (outermost / innermost / pushed into the codelet) | FFTW's *vector recursion*; disabling it cost 47% on a 1024×1024 2-D transform, and the planner's actual choices ("pushed one or two levels down", not to the leaves) contradicted the authors' own predictions | §2.3, §2.5 |
| Copy-to-contiguous-scratch before transforming, vs strided in-place | FFTW's *indirect plans*; "initially a surprise (and often boosted speed by 20% or more)" | §2.5 |
| Schedule / instruction order within the unrolled codelet | genfft's schedule is provably spill-optimal only for sizes that are a power of 4 — so it is heuristic for **all** of 6, 8, 17, 36. 2× runtime spread among equivalent small-transform formulas with no cache effects | §6.1, §3.2 |
| Unroll depth and blocking for the 3 passes at L=36 | 46 656 points ≈ 2¹⁵.⁵ is in the band where ESTIMATE-vs-MEASURE grows from 20% toward 2–3× | §2.3 |
| Naive 17×17 matvec vs Rader vs Bluestein for L=17 | FFTW keeps a Θ(n²) generic plan "useful for n ≲ 100" and lets the planner pick | §6.4a |
| Compiler and flags, per codelet | >3× spread over 2000 random flag sets; `-O3` sometimes slower than `-O1`; aggressive reordering destroys a hand-computed schedule | §3.4, §6.1 |
| Sign/constant representation in the generated code | 10–15% from making all constants positive | §6.1 |

**Per-size character, in one line each:**

- **L=6 (216 pts, 3.4 kB).** Register/L1 problem only. The searchable space is schedule,
  unroll, constant handling, and batch-loop placement. Expect ~20%-class wins (§2.3), and note
  SPIRAL's evidence that a 2× spread is available even with no cache effects (§3.2). Timing
  overhead is the binding difficulty, not tuning (§6.2, §8).
- **L=8 (512 pts, 8.2 kB).** Same regime. Note that 8 is *not* a power of 4, so genfft's
  spill-optimality proof does not cover it either (§6.1) — the classic split-radix/radix-8
  literature gives a strong baseline, but the schedule still has to be earned by measurement.
- **L=17 (4 913 pts, 78.6 kB).** L2-resident. Genuine algorithm-selection question (§6.4a);
  everything else is schedule and codegen. Still under 2¹³ total points, so ~20%-class.
- **L=36 (46 656 pts, 746 kB).** The only size where cache blocking is a first-order concern,
  and the only one in the band where the FFTW authors measured search moving from 20% toward
  2–3× (§2.3). Also the size where *indirect*/buffered strategies and batch-loop placement have
  the most room to matter (§2.5). **If search time is rationed, spend it here.**

**Recommended search protocol shape** (structure borrowed from ATLAS's orthogonal line search
§4, budget allocation from OpenTuner's AUC bandit §5.2, and the "fastest appears early"
observation from Singer & Veloso §5.1): fix a small principled set of candidate dags per size;
cross them with a handful of schedule variants, unroll depths, batch-loop placements and
copy/no-copy choices; enumerate near-exhaustively (minutes, per §3.5); time under the protocol
in §8; then re-run the winner's neighbourhood, since orthogonal line search is only a heuristic
and re-running it with updated reference values is the standard refinement (§4).

---

## 8. Benchmarking FFTs defensibly

This section exists for the monitor agent. There is a real, substantive disagreement in this
literature about how to summarise timings; §8.4 resolves it for our specific case rather than
pretending it away.

### 8.1 The reference protocol: benchFFT

FFTW's own benchmark suite, which produced the numbers in §2 and §3, documents its method. Its
protocol, from the methodology page:

- Initialisation is timed separately, "as a rough indicator, without optimization attempts,"
  and excluded from the transform time.
- **"First, we compute enough repeated FFTs so that the total time is sufficient for accurate
  timing, and divide by the number of iterations to obtain the average time."**
- **"Second, we repeat this averaging process eight times, and report the minimum average time
  (to avoid fluctuations due to system interrupts, cache priming, etcetera)."**
- The input array is initialised to **zero**, to prevent values diverging under repeated
  transforms. No attempt is made to prime caches during the averaging.
- Timer: `gettimeofday`, with the minimum measurement time calibrated using `lmbench`.
- Metric: `mflops = 5 N log2(N) / (time for one FFT in microseconds)`, halved for real-data
  FFTs (`2.5 N log2(N)/t`), N being the total number of data points across all dimensions.
- In-place and out-of-place are benchmarked and reported **separately**.
- The explicit caveat: routines using different data formats and storage "are not strictly
  comparable," and results should be read "only as a rough description of the relative merits."

Note what the MFLOPS metric is and is not. Frigo & Johnson: it is based on "the asymptotic
number of operations for the radix-2 Cooley-Tukey algorithm … although the actual count is
lower for most DFT implementations. **The MFLOPS measure should thus be viewed as a convenient
scaling factor rather than as an absolute indicator of CPU performance.**"

**Recommendation:** adopt benchFFT's structure exactly — inner repeat loop to amortise timer
error, then minimum over 8 outer averages — because it is the protocol the comparison numbers
in this whole document were produced under, and because it happens to coincide with what the
statistics literature recommends for deterministic kernels (§8.3). Use cycles rather than
MFLOPS internally; report MFLOPS only for cross-library comparison, and label it as a scaling
factor.

### 8.2 Getting the clock right on x86-64

Intel's timing whitepaper (Paoloni, doc. 324264-001, September 2010) is the primary source for
the mechanics. Its recommended sequence, verbatim:

```c
asm volatile ("CPUID\n\t"
      "RDTSC\n\t"
      "mov %%edx, %0\n\t"
      "mov %%eax, %1\n\t": "=r" (cycles_high), "=r" (cycles_low)::
"%rax", "%rbx", "%rcx", "%rdx");
/***********************************/
/*call the function to measure here*/
/***********************************/
asm volatile("RDTSCP\n\t"
     "mov %%edx, %0\n\t"
     "mov %%eax, %1\n\t"
   "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
"%rax", "%rbx", "%rcx", "%rdx");
```

Why this shape:

- `CPUID` before `RDTSC` "implements a barrier to avoid out-of-order execution of the
  instructions above and below the RDTSC instruction. Nevertheless, this call does not affect
  the measurement since it comes before the RDTSC."
- `RDTSCP` is only *pseudo*-serialising — per the Intel manual it "waits until all previous
  instructions have been executed before reading the counter. However, subsequent instructions
  may begin execution before the read operation is performed." Hence the trailing `CPUID`,
  which prevents later instructions being hoisted into the measured region.
- Using `CPUID` *between* the two reads (the obvious approach) "works but there is a lot of
  variance (in terms of clock cycles) that is intrinsically associated with the CPUID
  instruction execution itself."

Reported quality of the improved method: "total variance = 2" cycles (standard error 1.414
cycles, down from 6.9), **"variance of variances = 0, variance of minimum values = 0"** —
"this improved benchmarking method is completely ergodic (between different ensembles the
maximum fluctuation of the variance is 1 clock cycle and the minimum value is perfectly
constant)." For the non-RDTSCP fallback method: total variance 1 cycle, variance of variances
2 cycles, and a measured **resolution of two assembly instructions** — "the min value … is
constant (44 cycles) between 0 and 5 measured assembly instructions and between 6 and 10
assembly instructions (48 clock cycles). Then it increases very regularly by four cycles every
two assembly instructions."

The prescribed workflow, quoted:

> "1. Run the tests … 2. Analyze the variance of the variances and the variance of the minimum
> values to validate the method on his platform. If the values that the user obtains are not
> satisfactory, he may have to change the BIOS settings or the BIOS itself. 3. Calculate the
> resolution that the method is able to guarantee. 4. Make the measurement and subtract the
> offset (additional cost of calling the measuring function itself)."

Two more details from the same document: the measured function should be declared `inline` "so
that from an assembly perspective there is no overhead in calling the function itself," and the
harness calls the function in a warm-up nest first "to 'warm up' the instruction cache to avoid
spurious measurements due to cache effects in the first iterations."

**Consequences for us.** A ~44–48 cycle measurement floor and a ~2-instruction resolution mean
a single L=6 volume transform is *close to* the noise floor if timed alone. Batch. Subtract the
calibrated offset. Report cycles. And validate the harness on each node (AVX2 and AVX-512 boxes
separately) by checking variance-of-variances before trusting any comparison.

### 8.3 The case for the minimum

Chen & Revels (IEEE HPEC 2016) build the statistical argument that benchFFT and Intel arrive
at empirically. Their model: a measurement is `T_i/n_i = t_P0 + E_i` where `t_P0` is the true
minimum single-execution time and `E_i` aggregates delay factors plus timer error. Then

> "t̂_P0 = min(T_1/1, … T_j/j) = t_P0 + min(E_1, … E_j). Thus, t̂_P0 is the estimate of t_P0
> which minimizes the error terms appearing in our sample."

Their justification for min over median/mean/trimmed mean:

> "the error terms E_i are sampled from a sum of scaled random variables following nonidentical
> Poisson binomial distributions. As such, these terms can and do exhibit multimodal behavior.
> While estimators like the median and trimmed mean are known to be robust to outliers, Fig. 3
> demonstrates that they still capture bimodality of the distributions … In contrast, the
> distribution of the minimum across all experimental trials is **unimodal in all cases we have
> observed**. Thus for our purposes, the minimum is a unimodal, robust estimator for the
> location parameter of a given benchmark's timing distribution."

And their rule for how many inner repetitions: let `j = τ_acc / τ_prec` (timer accuracy over
timer precision). "If each timing measurement consists of more than j repetitions, then the
contribution of timer inaccuracy to the total error is less than τ_acc/j = τ_prec, and so is
too small to measure. Thus, there is no reason to pick n > j." They note this calibration
"need only be applied once per benchmark, since the estimated n can be cached for use in
subsequent experiments on the same machine" — unlike `criterion`, "which re-determines n every
time a benchmark is run." Their Fig. 2 shows the point graphically: individual `T/n` vs `n`
curves are "wildly oscillatory" while "the minimum across all the curves at each n is much
smoother and asymptotically tends toward the same constant value."

### 8.4 The case against the minimum — and why it does not apply to us

The managed-language benchmarking literature reaches the *opposite* conclusion, and it is worth
understanding why before dismissing it.

**Georges, Buytaert & Eeckhout (OOPSLA 2007)** show a real case where "best of 30" and
"mean with 95% confidence interval" give contradictory answers. Comparing five garbage
collectors on SPECjvm98 `db` with a 120 MB heap:

> "Based on the best method, one would conclude that the performance for the CopyMS and
> GenCopy collectors is about the same. The statistically rigorous method though shows that
> GenCopy significantly outperforms CopyMS. Similarly, based on the best method, one would
> conclude that SemiSpace clearly outperforms GenCopy. The reality though is that the
> confidence intervals for both garbage collectors overlap … The 'best' method reports the
> really good run whereas a statistically rigorous approach reliably reports that the average
> scores for GenCopy and SemiSpace are very close to each other."

Their survey of 50 papers from OOPSLA, PLDI, VEE, ISMM and CGO: "about one third of the papers
(16 out of the 50 papers) does not specify the methodology used"; the most popular summaries
were "average and best — 8 and 10 papers out of the 50 … median, second best and worst are less
frequent, namely 4, 4 and 3 papers"; and "In only a small minority of the research papers
(4 out of 50), confidence intervals are reported." For sample sizes they use the standard
threshold: large-sample Gaussian treatment for "n ≥ 30", Student-t below that.

**Kalibera & Jones (ISMM 2013)** go further, treating an experiment as having *levels*
(iteration, execution, build/compilation) and asking where repetition is actually needed:
"repetition is most needed where most non-determinism occurs." Their machinery: run a one-off
*dimensioning experiment* (repetition counts "some arbitrary yet sufficient value; **20** may be
a good choice but use **30** if possible"), estimate variance per level and cost per level
(`c1` iteration duration, `c2` time to independent state, `c3` build time), then solve for
optimal repetition counts at all but the top level. Their recommendations, quoted:

> "Analysis of results should be statistically rigorous and in particular should quantify any
> variation. Report performance changes with **effect size confidence intervals**."
> "For each benchmark/VM/platform, conduct a dimensioning experiment to establish the optimal
> repetition counts for each but the top level of the real experiment. Re-dimension only if the
> benchmark/VM/platform changes."

Concrete counts, for SPEC CPU on their platform P3 with a layout-randomising gcc, to get 95%
CI half-widths of 0.5%/1%/1.5%/2% of the mean: most benchmarks need only 4–7 executions, but
`mcf` needs **42** executions for 0.5% and `xalancbmk` needs **48**. "Five of the benchmarks are
so stable that fewer than 5 executions already give a half-width of 0.5%. We would still run 5
executions of these, though, to get the confidence interval estimate." They also warn that
automatic steady-state detection is unreliable — "By choosing different iterations in different
runs, these heuristics can create an error of tens of percent" — and that reliable detection
"is an open problem."

**Mytkowicz et al. (ASPLOS 2009)** add the orthogonal hazard: *measurement bias* from factors
that look irrelevant. From the abstract:

> "measurement bias is significant and commonplace. By significant we mean that measurement
> bias can lead to an incorrect conclusion. By commonplace we mean that measurement bias occurs
> in all architectures that we tried (Pentium 4, Core 2, and m5 O3CPU), all compilers that we
> tried (gcc and Intel's C compiler), and all of the SPEC CPU2006 C programs. … Nevertheless,
> in a literature survey of 133 recent papers from ASPLOS, PACT, PLDI, and CGO, we determined
> that **none of the papers with experimental results adequately consider measurement bias**."

Their prescriptions are *causal analysis* (to detect) and *setup randomization* (to avoid).
Kalibera & Jones followed up on one such factor — code layout — by patching gcc to randomise
function order, and found "the effect of code layout to be small for reference sizes" of SPEC
CPU and "no significant impact on the performance of DaCapo benchmarks."

**Reconciliation.** The disagreement is about what the estimator is estimating.

- Georges/Kalibera measure systems whose *own behaviour* is random: JIT compilation plans,
  GC timing, heap layout. There, each run is a genuine sample from the population you care
  about, so the mean is the quantity of interest and the minimum is a biased pick of a lucky
  draw.
- benchFFT/Intel/Chen–Revels measure a *deterministic* kernel: fixed control flow, fixed
  instruction sequence, no runtime, no allocator, no JIT. The only variation is contamination
  (interrupts, migration, frequency transitions, cache pollution from other work), which is
  strictly additive. There the minimum is the *least contaminated* estimate of the machine's
  actual capability.

**Our transforms are in the second category.** A fixed-size unrolled C FFT on one thread has no
internal nondeterminism. So: minimum-of-repeated-averages is the right primary statistic, per
benchFFT and Chen & Revels.

But two obligations from the first camp survive and should be honoured:

1. **Report dispersion anyway.** Kalibera & Jones's recommendation to "quantify any variation"
   costs nothing and is the only way a reader can tell a 3% win from harness noise. Report the
   minimum *and* the spread across outer repetitions, and treat a difference smaller than that
   spread as unresolved.
2. **Randomise the setup and repeat at the level where the nondeterminism actually lives**
   (Mytkowicz; Kalibera & Jones). For us that level is *not* the iteration — it is the **build
   and the process**. Compile-and-link is where layout, alignment and code placement are
   decided, and those are exactly Mytkowicz's biasing factors. Concretely: rebuild and re-run
   each competing implementation several times in a randomised order, rather than timing one
   binary many times and moving on. A candidate that wins only in one link order has not won.

### 8.5 A concrete protocol for the monitor agent

Synthesising §8.1–§8.4 (the assembly is mine; each ingredient is sourced above):

1. **Pin and quiet.** Fix CPU affinity; check `constant_tsc`/`nonstop_tsc`; disable or record
   turbo and frequency governor state; validate the harness by the Intel variance-of-variances
   check (§8.2) on each distinct node. Record BIOS/microcode, compiler version and exact flags
   — a plan is only valid for the machine it was tuned on (§2.4).
2. **Calibrate the timer offset** once per machine (§8.2 step 4) and subtract it. Determine the
   inner repeat count `n` from `j = τ_acc/τ_prec` and cache it per machine (§8.3).
3. **Warm the I-cache and the twiddle tables** with untimed calls before the first measured
   iteration (§8.2); zero the input arrays so repeated transforms do not diverge (§8.1).
4. **Inner loop:** `n` repeated transforms over the batch, divide by `n`.
5. **Outer loop:** repeat the averaging **8** times; take the **minimum** average (§8.1).
6. **Report** minimum, plus spread across the 8 outer averages, plus cycles-per-point and
   (labelled as a scaling factor only) MFLOPS by the benchFFT formula (§8.1).
7. **Repeat across builds in randomised order** — at least 5 independent
   rebuild+relink+rerun cycles per candidate, interleaved (§8.4 obligation 2, and Kalibera &
   Jones's "5 executions … to get the confidence interval estimate").
8. **Never compare across nodes or across ISAs.** Rank within a node only (§2.4, §3.3).
9. **Verify correctness at every timing point.** Following FFTW's practice: compare against a
   slow high-precision reference DFT, and use the O(n log n) linearity/shift self-test. An
   implementation that is fast and wrong must be detectable by the harness, not by review.
10. **Publish the harness with the results.** Mytkowicz's finding that 0 of 133 papers
    adequately handled measurement bias is a statement about unreproducible harnesses as much
    as about statistics.

---

## 9. When each technique stops paying

| Technique | Buys | Stops paying when |
|---|---|---|
| Runtime planner (FFTW-style) | 20% (n ≲ 2¹³) to 2–3× (n ≳ 2¹⁶) over a static heuristic | The size is fixed at compile time — then the whole planner collapses to a build-time decision, and the runtime machinery is pure overhead |
| Dynamic programming over the derivation | Exponential → polynomial search; "method of choice" in SPIRAL | The context-independence assumption fails (cache state, register pressure from the caller). At ≤64-point straight-line codelets there is barely a tree to DP over |
| Exhaustive search | The actual optimum | Above "very small transform sizes" (SPIRAL). At 6/8/17/36 with a restricted rule set, it is affordable |
| Evolutionary / bandit search | Escapes DP's blind spots; OpenTuner reports up to 2.8× | Only worth the harness complexity when the space is genuinely huge; four fixed sizes do not qualify |
| Learned cost model | 2× better than random at equal trial budget (AutoTVM); ρ≈0.96 ranking (GenMAT) | Training cost cannot be amortised over four problems |
| Wisdom / offline plans | Removes planning from the runtime path | Across machines: ≥20% penalty on ≥1/3 of sizes, up to 40% |
| Compiler-flag search | 8–15% over `-O3`; >3× spread across flag sets | Diminishing, but the `-O3`-slower-than-`-O1` and schedule-destruction hazards never go away for unrolled codelets |
| Model-driven parameter choice | "comparable" to ATLAS's global search at "no measurable time"; genfft's scheduler matches SPIRAL's searched codelets | ~20% worse in the bad cases (Athlon MP register tiles). Use as baseline, not as final answer |

---

## 10. Unsourced engineering notes

Everything here is my own reasoning, attributed to nobody. It is separated out so it cannot be
mistaken for a citation.

1. **The batch dimension is our largest untapped search axis, and the literature only gestures
   at it.** LQCD momentum projection gives us many independent volumes. That means we can
   choose to run the batch loop outermost (best locality per volume), innermost (best SIMD
   utilisation — 4 or 8 volumes in flight per AVX register lane, with *zero* shuffle cost
   because the same element index of different volumes is contiguous if we lay the batch out
   as the fastest-varying index), or somewhere between. The innermost-batch layout is
   especially attractive at L=6 and L=17, where the line lengths (6, 17) are hostile to
   in-register SIMD along the transform direction. FFTW's vector-recursion results (§2.5) say
   this choice is unpredictable and worth 20–47%; nothing I found measures it for the
   batch-of-small-volumes case specifically. **Search it explicitly.**
2. **A batch-innermost ("struct of arrays over volumes") layout may make algorithm selection
   nearly irrelevant at L=6 and L=8.** If every arithmetic operation is a full-width vector op
   over independent volumes, the codelet becomes pure straight-line scalar-shaped code with no
   permutations, and the only remaining questions are instruction count, register pressure and
   scheduling. That would move L=6/8 from "algorithm selection" to "register allocation," which
   is precisely where genfft's model-driven scheduler is strongest (§6.1). Worth testing early,
   because if true it collapses a lot of the search space.
3. **L=36 = 4 × 9 is the only size where I would expect the classical cache literature to
   dominate the autotuning literature.** 746 kB per volume means a batch of even a few volumes
   exceeds L2 on many parts. The PFA split into coprime 4 and 9 is a *layout* decision as much
   as an algorithmic one, and the interaction with the batch layout is the thing to search.
4. **Timing a single L=6 transform is not possible to the precision we need.** With a ~44-cycle
   floor and 2-instruction resolution (§8.2), and 216 points, the transform itself is plausibly
   in the low hundreds of cycles. Any comparison must be over a batch large enough that the
   per-transform cost is ≫ the timer floor, and the batch size must be held identical across
   candidates or the comparison is meaningless.
5. **Verify before you rank, and rank before you optimise.** GenMAT's use of Spearman ρ (§5.2)
   is the right frame for the monitor agent: we do not need accurate absolute times, we need a
   correct ordering. That is a lower bar, and it means a noisy-but-unbiased harness with enough
   repetitions is fine, whereas a precise-but-biased one (e.g. always timing candidate A first,
   with a cold cache) is not.
6. **Expect the AVX2 and AVX-512 winners to be different codelets, not the same codelet
   recompiled.** This is an extrapolation from SPIRAL's 320% SSE/SSE2 transfer penalty (§3.3),
   which was caused by exactly this — a 2× vector-width change requiring "very different
   algorithm structures." Budget for two tuned variants per size.

---

## 11. References

### Verified — URL fetched during the session that produced this document

1. M. Frigo and S. G. Johnson, "The Design and Implementation of FFTW3," *Proceedings of the
   IEEE*, vol. 93, no. 2, pp. 216–231, 2005.
   <https://www.fftw.org/fftw-paper-ieee.pdf>
   *(Sections used: I, III, IV-A/C/D/E, V-A/B/C, VI, VII.)*
2. S. G. Johnson and M. Frigo, "Implementing FFTs in Practice," in *Fast Fourier Transforms*
   (C. S. Burrus, ed.), ch. 11, Rice Univ.: Connexions, 2008; posted as arXiv:2602.23525
   [math.NA], 26 Feb 2026. <https://arxiv.org/abs/2602.23525> ·
   <https://arxiv.org/pdf/2602.23525>
   *(Sections used: 1, 2, 3.3, 4.2, 4.3, 5, 5.1, 6, 7, Fig. 1, footnote 9.)*
3. M. Frigo, "A Fast Fourier Transform Compiler," *Proc. ACM SIGPLAN PLDI '99*, Atlanta, GA,
   May 1999. <https://www.fftw.org/pldi99.pdf>
   *(Sections used: 1, 6.)*
4. FFTW 3.3.10 manual, "Planner Flags."
   <https://www.fftw.org/fftw3_doc/Planner-Flags.html>
5. FFTW 3.3.10 manual, "Words of Wisdom — Saving Plans."
   <https://www.fftw.org/fftw3_doc/Words-of-Wisdom_002dSaving-Plans.html>
6. FFTW 3.3.10 manual, "Generating your own code."
   <https://www.fftw.org/fftw3_doc/Generating-your-own-code.html>
   *(Confirms only that the generated codelet set is defined by
   `{dft,rdft}/{codelets,simd}/*/Makefile.am`; it does not enumerate sizes.)*
7. FFTW FAQ, Section 3. <https://www.fftw.org/faq/section3.html>
8. benchFFT, "Benchmark Methodology." <https://www.fftw.org/speed/method.html>
   and benchFFT results index <https://www.fftw.org/speed/>
9. M. Püschel, J. M. F. Moura, J. R. Johnson, D. Padua, M. M. Veloso, B. W. Singer, J. Xiong,
   F. Franchetti, A. Gačić, Y. Voronenko, K. Chen, R. W. Johnson and N. Rizzolo, "SPIRAL: Code
   Generation for DSP Transforms," *Proceedings of the IEEE*, vol. 93, no. 2, pp. 232–275, Feb
   2005.
   <https://users.ece.cmu.edu/~moura/papers/ieeeproceedings-pueschelmouraetal-feb05-ieeexplore.pdf>
   *(Sections used: II, IV, V, VI-A/B, VII, Figs. 8, 9, 15, 16, Table 19.)*
10. B. Singer and M. Veloso, "Learning to Construct Fast Signal Processing Implementations,"
    *Journal of Machine Learning Research*, vol. 3, pp. 887–919, 2002 (special issue on
    ICML 2001). <https://www.jmlr.org/papers/volume3/singer02a/singer02a.pdf>
    Citation confirmed at <https://spiral.ece.cmu.edu/pub-spiral/abstract.jsp?id=37>
    *(Sections used: 1, 2.4, 2.5, 3.1.)*
11. K. Yotov, X. Li, G. Ren, M. Garzaran, D. Padua, K. Pingali and P. Stodghill, "Is Search
    Really Necessary to Generate High-Performance BLAS?," *Proceedings of the IEEE*, vol. 93,
    no. 2, 2005. <https://www.eecis.udel.edu/~xli/publications/ieee05.pdf>
    *(Sections used: Abstract, III-A/B/C, and the AMD Athlon MP and timing results.)*
12. ATLAS project page, Netlib. <https://netlib.org/utk/projects/atlas.0/>
13. G. Paoloni, "How to Benchmark Code Execution Times on Intel IA-32 and IA-64 Instruction Set
    Architectures," Intel white paper 324264-001, September 2010.
    <https://cis.temple.edu/~qzeng/cis3207-spring18/files/ia-32-ia-64-benchmark-code-execution-paper.pdf>
    *(Sections used: 2, 3.1, 3.2.1, 3.2.3, 4.)*
14. J. Chen and J. Revels, "Robust benchmarking in noisy environments," *2016 IEEE High
    Performance Extreme Computing Conference (HPEC)*; also arXiv:1608.04295.
    <https://math.mit.edu/~edelman/publications/robust_benchmarking.pdf>
    *(Sections used: II, III, IV-A/B/C, Figs. 2, 3.)*
15. A. Georges, D. Buytaert and L. Eeckhout, "Statistically Rigorous Java Performance
    Evaluation," *OOPSLA 2007*. <https://dri.es/files/oopsla07-georges.pdf>
    *(Sections used: 1, 2.1.1, 2.1.2, 3.2.1, Fig. 1.)*
16. T. Kalibera and R. E. Jones, "Rigorous Benchmarking in Reasonable Time," *ISMM 2013*,
    Seattle, WA. Fetched copy:
    <http://petertsehsun.github.io/soen691/current/papers/reasonable_benchmarking.pdf>
    Repository record: <https://kar.kent.ac.uk/33611/>
    *(Sections used: 1, 3, 5, 9, 9.1, 11, 12, Tables 6, 7.)*
17. T. Mytkowicz, A. Diwan, M. Hauswirth and P. F. Sweeney, "Producing Wrong Data Without Doing
    Anything Obviously Wrong!," *ASPLOS 2009*. Abstract fetched verbatim from the authors'
    group page: <https://sape.inf.usi.ch/publications/asplos09>
    *(Only the abstract was retrievable; only the abstract is quoted.)*
18. J. Ansel, S. Kamil, K. Veeramachaneni, J. Ragan-Kelley, J. Bosboom, U.-M. O'Reilly and
    S. Amarasinghe, "OpenTuner: An Extensible Framework for Program Autotuning," *PACT 2014*.
    <https://commit.csail.mit.edu/papers/2014/ansel-pact14-opentuner.pdf>
    *(Sections used: Abstract, 1, 2, 3.2.1.)*
19. T. Chen, L. Zheng, E. Yan, Z. Jiang, T. Moreau, L. Ceze, C. Guestrin and A. Krishnamurthy,
    "Learning to Optimize Tensor Programs," *NeurIPS 2018*.
    <https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2025_2026/papers/chen_NIPS_2018.pdf>
    *(Sections used: Abstract, 3, 4, 5, 6.1, Figs. 4, 5.)*
20. N. Zhang, A. Srivastava, R. Kannan and V. K. Prasanna, "GenMAT: A General-Purpose Machine
    Learning-Driven Auto-Tuner for Heterogeneous Platforms," *2021 IEEE/ACM Programming
    Environments for Heterogeneous Computing (PEHC)*, DOI 10.1109/PEHC54839.2021.00006.
    <https://naifeng.github.io/assets/pdf/PEHC_GenMAT_2021.pdf>
    *(Sections used: IV-B2, IV-B3, Tables II, III.)*

### Unverified

21. `[UNVERIFIED — could not fetch]` R. C. Whaley, A. Petitet and J. J. Dongarra, "Automated
    empirical optimizations of software and the ATLAS project," *Parallel Computing*, vol. 27,
    no. 1–2, pp. 3–35, 2001, DOI 10.1016/S0167-8191(00)00087-9. I could not retrieve the paper
    text. Everything this document says about ATLAS's mechanism is taken from Yotov et al.
    (ref. 11), which reimplements and describes it in detail, and from the Netlib project page
    (ref. 12). No number in this document is attributed to the Whaley paper.
22. `[UNVERIFIED — could not fetch]` A. M. Blake, I. H. Witten and M. J. Cree, "The Fastest
    Fourier Transform in the South," *IEEE Transactions on Signal Processing*, vol. 61, no. 19,
    pp. 4707–4716, Oct 2013. All mirrors tried returned 403/404/410. Mentioned in §6.5 as an
    existing branch of the literature (runtime code specialisation for fixed parameters); no
    numbers or claims from it are used.

### Sources searched for and not found

- A published measurement of `FFTW_PATIENT` vs `FFTW_EXHAUSTIVE` specifically. The research
  literature describes patient/impatient/estimate (§2.2–2.3); `FFTW_EXHAUSTIVE` is documented
  in the manual only qualitatively ("an even wider range of algorithms"), with no quantified
  benefit that I could locate.
- Any autotuning or plan-selection measurement for a **3-D** transform of a **non-power-of-two,
  non-prime composite** size (i.e. anything resembling L=36) in the FFTW or SPIRAL papers. The
  2-D 1024×1024 case (§2.3) is the closest published multi-dimensional datapoint.
- Any published operation count or timing for a 17-point DFT module. §6.4 gives FFTW's
  *structural* position on prime sizes near 17 (Rader / Bluestein / generic Θ(n²) for n ≲ 100)
  but I found no source with counts for n = 17 itself.
- A quantified comparison of batch/vector-loop placement strategies for batched *small* 3-D
  transforms. FFTW's vector-recursion results (§2.3, §2.5) are the nearest evidence and they
  are for large 1-D and 2-D transforms.
- The exact transform size in the genfft Rader operation-count result (§6.4c), and the exact
  exponents in SPIRAL's formula-count asymptotics (§3.1) and the size of the Fig. 8 DCT-2 and
  Fig. 9 DFT (§3.2) — all typeset in math fonts that my PDF text extraction could not decode.
  In each case the surrounding prose claim is quoted and the undecoded number is not asserted.
