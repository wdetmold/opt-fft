# 05 — Memory hierarchy, blocking, and transposes for 3D transforms

**Scope.** What the literature says about where the *data movement* goes in a 3D FFT,
which reformulations reduce it, and — crucially for this project — at what problem size
each technique stops paying. Targets: complex-double forward 3D FFTs on cubes
L = 6, 8, 17, 36, single-threaded x86-64 (AVX2, sometimes AVX-512), many volumes batched.

**Executive summary for implementers.** For all four of our sizes the classical
memory-hierarchy machinery (four-step / six-step / cache-oblivious recursion) is
*already past its useful range at the level of a single volume*. Frigo and Johnson state
outright that FFTW switches to the radix-√n four-step formulation only for
n ≳ 2²⁰ ≈ 10⁶ — three to four orders of magnitude above 36³ = 46 656
([Johnson & Frigo 2008, §3.3](https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf)).
Our per-volume working sets are 3.4 KiB / 8 KiB / 76.8 KiB / 729 KiB, i.e. L1- or
L2-resident. **The memory-hierarchy problem in this project is therefore not
"how do I block a big FFT" but "how do I stream a batch of small L1/L2-resident FFTs
at memory bandwidth without ever touching a transpose".** Sections 3, 5, 7, 8 and 9 are
the ones that matter; sections 1, 2 and 4 are there so you know why, and so you
recognise the point at which L=36 × large batch does start behaving like a
memory-bound streaming kernel.

---

## 1. The I/O-complexity results you should know

### 1.1 The models

The *ideal-cache model*: a cache holding **Z** data items (equivalently **M** items in
the FLPR notation), transferred in lines of **L** items (**B** bytes/items in FLPR),
with optimal replacement. Frigo, Leiserson, Prokop and Ramachandran require the
**tall-cache assumption** M = Ω(B²) for their transpose and FFT bounds
([Frigo, Leiserson, Prokop & Ramachandran, *Cache-Oblivious Algorithms*, ACM Trans.
Algorithms 8(1), art. 4, Jan 2012](https://g-trees.github.io/g_trees/assets/references/frigo1999cache.pdf),
abstract and §3). Johnson & Frigo restate the same requirement as Z = Ω(L²)
([Johnson & Frigo 2008](https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf), §3.1 fn. 5).

*Why you should care about tall-cache here*: on the Haswell node in this repo
(`lscpu -C`: L1d 32 KiB, 8-way, 64 sets, 64 B lines) B = 4 complex doubles, so
B² = 16 items = 256 B ≪ 2048 items of L1 capacity. Tall-cache holds by a wide margin at
every level for complex double. It is *not* a constraint for us.

### 1.2 The four bounds, verbatim

| Result | Statement | Source |
|---|---|---|
| Transpose (upper) | "The **Rec-Transpose** algorithm involves O(mn) work and incurs O(1 + mn/B) cache misses for an m × n matrix." | [FLPR 2012](https://g-trees.github.io/g_trees/assets/references/frigo1999cache.pdf), Lemma 3.1 |
| Transpose (optimality) | "The **Rec-Transpose** algorithm exhibits optimal cache complexity." | ibid., Theorem 3.2 |
| FFT (cache-oblivious) | Q(n) = O(1 + (n/B)(1 + log_M n)) — "which is optimal for a Cooley-Tukey algorithm, matching the lower bound by Hong and Kung […] when n is an exact power of 2." | ibid., §3 (FFT) |
| FFT (blocked, cache-aware) | Q_b(n; Z) = Θ(n log_Z n), and "this complexity is rigorously optimal for Cooley-Tukey FFT algorithms […], and immediately points us towards **large radices** (not radix 2!) to exploit caches effectively in FFTs." | [Johnson & Frigo 2008](https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf), eq. (3) |
| Depth-first radix-2 | Q₂(n; Z) = Θ(n log[n/Z]) — cache-oblivious but *not* optimal; "the textbook radix-2 algorithm […] is 'pessimal' cache-oblivious" when executed breadth-first (Θ(n log₂ n) misses, "no temporal locality at all is exploited!"). | ibid., eqs. (4)–(5), §3.1 |
| Four-step recursive | Q_o(n; Z) = Θ(n log_Z n) — "There exists a different recursive FFT that is *optimal* cache-oblivious, however, and that is the radix-√n 'four-step' Cooley–Tukey algorithm (again executed recursively, depth-first)." | ibid., eq. (6) |

**Reading of the three FFT numbers.** Breadth-first radix-2 → Θ(n log₂ n) misses.
Depth-first radix-2 → Θ(n log(n/Z)) misses: still not optimal, but it *stops paying
misses entirely* once a subtransform fits in cache — "once the recursion reaches a size
n ≤ Z, the subtransform fits into the cache and no further misses are incurred. The
algorithm does not 'know' this and continues subdividing the problem, of course, but all
of those further subdivisions are in-cache because they are performed in the same
depth-first branch of the tree" (ibid., §3.1). Recursive radix-√n → Θ(n log_Z n),
matching the blocked/optimal bound.

**When all of this stops paying: immediately, for us.** Every one of these Θ(·) results
degenerates when n ≤ Z, because then Q = Θ(n/B) — one compulsory pass — for *any* of the
algorithms. Since 6³, 8³, 17³ and 36³ are all ≤ Z for some real cache level (§5), the
asymptotic separation between "textbook" and "optimal cache-oblivious" is exactly zero
for a single volume. What replaces it is (a) instruction-level concerns inside the
codelet, and (b) *batch* streaming behaviour.

### 1.3 Provenance / unverified links in this subsection

- **Hong & Kung 1981** (red-blue pebble game; the FFT I/O lower bound
  Ω(n log n / log S) for O(S) fast memory) is the origin of the FFT lower bound quoted
  by both FLPR and Johnson & Frigo. I could not fetch the primary text:
  [DTIC ADA104739](https://apps.dtic.mil/sti/tr/pdf/ADA104739.pdf) —
  `[UNVERIFIED — could not fetch]`. Cite it *via* FLPR §3 and Johnson & Frigo eq. (3),
  both of which I read.
- **Aggarwal & Vitter, "The Input/Output Complexity of Sorting and Related Problems",
  Comm. ACM vol. 31 (1988), pp. 1116–1127** — the reference number and pagination are
  as printed in Bailey's reference list [2], which I read
  ([fftq.pdf](https://www.davidhbailey.com/dhbpapers/fftq.pdf)); I did not fetch the
  paper itself. `[UNVERIFIED — could not fetch primary]`

---

## 2. Four-step and six-step: the blocked formulations, with real crossover numbers

### 2.1 The algorithms as originally stated

Bailey's paper is the canonical statement and is worth reading directly; it is *also*
the source of the single most useful negative datapoint in this whole section.
**David H. Bailey, "FFTs in External or Hierarchical Memory", J. Supercomputing 4(1),
March 1990, pp. 23–35** ([PDF](https://www.davidhbailey.com/dhbpapers/fftq.pdf); the PDF
is the December 30, 1989 preprint carrying that journal reference on its title page).

Bailey's **four step** algorithm for n = n₁n₂ (column-major/Fortran storage), verbatim:

> 1. Perform n₁ simultaneous n₂-point FFTs on the input data considered as a n₁ × n₂ complex matrix.
> 2. Multiply the resulting data, considered as a n₁ × n₂ matrix A_jk, by e^(±2πijk/n). The ± sign is the sign of the transform.
> 3. Transpose the resulting n₁ × n₂ complex matrix into a n₂ × n₁ matrix.
> 4. Perform n₂ simultaneous n₁-point FFTs on the resulting n₂ × n₁ matrix.

and its properties, verbatim: "both of the simultaneous FFT steps can be performed using
exclusively unit stride data access"; "this algorithm produces an ordered transform […]
it is not necessary to perform a bit reversal permutation"; "only three passes through
the external data set are required to perform this algorithm — the second step can be
performed on a block of data after the first step, before it is returned to memory."
Bailey notes n₁, n₂ should be "as close as possible to √n", and credits the algorithm to
**Gentleman & Sande, "Fast Fourier Transforms — For Fun and Profit", AFIPS Proc. vol. 29
(1966), p. 563–578** (Bailey ref. [8], p. 569; pagination as printed in Bailey's
reference list, primary not fetched — `[UNVERIFIED — could not fetch primary]`).

The **six step** variant (Bailey, same paper) is: transpose → n₁ individual n₂-point
FFTs → twiddle multiply → transpose → n₂ individual n₁-point FFTs → transpose. Bailey's
own verdict, verbatim: it "is very well suited for distributed memory systems […] Its
main memory scratch requirement is only 4n₂ cells for step 2 and 4n₁ cells for step 5
(per processor)" *but* it "has the serious disadvantage of requiring an additional two
transpose steps, **which typically are the chief bottlenecks on any system with a
distributed or external memory.**" That sentence is the whole argument of §3 below.

The six-step form is what FLPR make cache-oblivious: "The basic algorithm is the
well-known 'six-step' variant [Bailey 1990; Vitter and Shriver 1994b] of the
Cooley-Tukey FFT algorithm. Using the cache-oblivious transposition algorithm, however,
the FFT becomes cache-oblivious" (FLPR §3, and identically in
[Prokop's MSc thesis, MIT, June 1999](https://ocw.mit.edu/courses/6-895-theory-of-parallel-systems-sma-5509-fall-2003/6dc7de52dcf13b53cebf2fe10ae6752a_cach_oblvs_thsis.pdf), §3).

### 2.2 Bailey's own numbers — and where four-step *loses*

Bailey Table 1, Cray-2 (4.1 ns clock, 268 Mword of 80 ns DRAM), forward 2^m FFT followed
by inverse, averaged over ten trials, MFLOPS computed from 10·m·2^m flops. Four-step
(Fortran driver calling the Cray assembly `CFFTMLT` for steps 1 and 4) vs. Cray's
assembly library `CFFT2`:

| m | n = 2^m | four-step MFLOPS | Cray library MFLOPS | four-step / library |
|---|---|---|---|---|
| 8 | 256 | 42.5 | 57.2 | **0.74** |
| 9 | 512 | 60.9 | 81.8 | **0.74** |
| 10 | 1 024 | 76.4 | 106.0 | **0.72** |
| 11 | 2 048 | 106.6 | 109.4 | **0.97** |
| 12 | 4 096 | 137.8 | 130.6 | 1.06 |
| 13 | 8 192 | 143.8 | 145.2 | 0.99 |
| 15 | 32 768 | 163.9 | 150.2 | 1.09 |
| 16 | 65 536 | 187.5 | 159.0 | 1.18 |
| 20 | 1 048 576 | 204.4 | 176.8 | 1.16 |

**The crossover on that machine is n ≈ 4096, and below n ≈ 2048 four-step is 25–28 %
slower than a plain library FFT.** Bailey is explicit about the mechanism: "the four step
FFT algorithm may actually require a slightly larger number of floating-point arithmetic
operations than conventional FFT algorithms," and his MFLOPS figures deliberately do not
count the extra work. He reports the same shape on the Cray Y-MP: at m = 8 the four-step
gets 68.68 MFLOPS against the library's 137.85 — a factor of two *worse*.

**Direct consequence for L = 6, 8, 17, 36.** The one-dimensional line transforms in a
row-column 3D FFT are of length 6, 8, 17, 36. Every one of them is three to nine binary
orders of magnitude below Bailey's crossover. **Do not use a four-step or six-step
decomposition for the 1D line transform at any of these four sizes.** Use a fully
unrolled straight-line codelet (see the codelet/genfft literature in the companion
sections). The four-step idea reappears in this project only as a way to organise the
*3D* transform over a batch, and even then only in the fused, transpose-free form of §3.

### 2.3 Blocked six-step / nine-step (cache-aware)

**D. Takahashi, "A blocking algorithm for FFT on cache-based processors", Proc. 9th
Int'l Conf. High-Performance Computing and Networking (HPCN Europe 2001), LNCS vol. 2110,
Springer, pp. 551–554, 2001** (DOI 10.1007/3-540-48228-8_58). This is the origin of the
"block nine-step FFT" that adds a further level of blocking to the six-step algorithm to
cut cache misses; it is FFTW3's reference [34], and I read that reference entry inside
the FFTW3 paper. **I could not fetch the paper itself** — Springer refused
([link](https://link.springer.com/chapter/10.1007/3-540-48228-8_58)) —
so: `[UNVERIFIED — could not fetch]`. Reported performance figures circulating for it
(≈108 MFLOPS on a 333 MHz UltraSPARC-IIi, ≈247 MFLOPS on a 1 GHz Pentium III for a 2²⁰
FFT) come only from secondary summaries and **should not be quoted as fact**. Relevance
to us: none directly (it is a large-1D-FFT technique), but it is the correct citation if
you need prior art for "block the six-step so each block is cache-resident".

---

## 3. Row-column 3D: where the transposes are, and how to not do them

### 3.1 The naive accounting

A row-column 3D FFT of L³ is 3L² one-dimensional transforms of length L. In a row-major
array `A[z][y][x]` the x-axis lines are contiguous (stride 1), the y-axis lines have
stride L, and the z-axis lines have stride L². Two of the three passes are strided.

The textbook fix is to transpose between passes so every pass is unit-stride — i.e. to
run each axis through Bailey's six-step shape. That is exactly what Bailey warns against
(§2.1). The cost is easy to bound (**my own arithmetic**, complex double, 16 B/point,
N = L³): one out-of-place full-volume transpose moves 2·16N bytes; two of them move
64N bytes, against 5N log₂N flops of useful arithmetic:

| L | N = L³ | volume (16N) | two transposes (64N) | 5N log₂N flops | transpose bytes per flop |
|---|---|---|---|---|---|
| 6 | 216 | 3 456 B | 13 824 B | 8 375 | 1.65 |
| 8 | 512 | 8 192 B | 32 768 B | 23 040 | 1.42 |
| 17 | 4 913 | 78 608 B | 314 432 B | 301 226 | 1.04 |
| 36 | 46 656 | 746 496 B | 2 985 984 B | 3 618 120 | 0.83 |

At ~0.8–1.7 bytes of pure permutation traffic per useful flop, and machine balance on the
order of 3 flop/byte for a single AVX2 FMA core against DRAM, **explicit full-volume
transposes are never affordable at these sizes.** (Own arithmetic. This is not an argument
against *in-cache* register/tile transposes — see §10.)

### 3.2 FFTW's answer: multi-dimensional transforms have no transposes at all

FFTW does not transpose between axes. Its higher-rank rule, verbatim from
**M. Frigo & S. G. Johnson, "The Design and Implementation of FFTW3", Proc. IEEE 93(2),
pp. 216–231, 2005** ([PDF](https://www.fftw.org/fftw-paper-ieee.pdf), §IV-C.4):

> Formally, to solve dft(N,V,I,O), where N = N₁ ∪ N₂, |N₁| ≥ 1 and |N₂| ≥ 1, FFTW
> generates a plan that first solves dft(N₁, V ∪ N₂, I, O), and then solves
> dft(copy-o(N₂), copy-o(V ∪ N₁), O, O).

Read it operationally: the *other* dimensions of the 3D problem are pushed into the
**vector rank** V — they become loop nests around a 1D transform with whatever stride
they happen to have. No permutation is materialised. FFTW's heuristic for splitting is
also stated: "A typical heuristic is to choose two sub-problems N₁ and N₂ of roughly
equal rank, where each input stride in N₁ is smaller than any input stride in N₂"
(ibid.) — i.e. do the contiguous axes first.

FFTW *does* own a cache-oblivious transposer, but it lives in rank-0 plans, i.e. it is
used for explicit permutations and for in-place 1D transforms, not between the axes of a
multi-dimensional DFT: "When |V| = 2, I = O, and the strides denote a matrix-transposition
problem, FFTW creates a plan that transposes the array in-place. FFTW implements the
square transposition dft({},{(n,ι,o),(n,o,ι)},I,O) by means of the 'cache-oblivious'
algorithm from [35], which is fast and, in theory, uses the cache optimally regardless of
the cache size" (ibid., §IV-C.2). Non-square cases use
**M. Dow, "Transposing a matrix on a vector computer", Parallel Computing 21(12),
pp. 1997–2005, 1995** and
**E. G. Cate & D. W. Twigg, "Algorithm 513: Analysis of in-situ transposition",
ACM TOMS 3(1), pp. 104–110, 1977** (FFTW3 refs [36], [37] — bibliographic data as printed
in the FFTW3 reference list I read; primaries not fetched,
`[UNVERIFIED — could not fetch primary]`).

**Takeaway for us: write the 3D transform as three stride-parameterised passes over the
same buffer. Do not permute. Fix the strided passes with tiling (§3.4), not with data
movement.**

### 3.3 Vector recursion: push the batch loop toward the leaves

This is the single most directly applicable FFTW technique for a batched-small-volume
workload. Verbatim from
[Johnson & Frigo 2008](https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf), §4.2.6
(same text in FFTW3 §IV-D.2):

> Another example of the effect of loop reordering is a style of plan that we sometimes
> call *vector recursion* (unrelated to "vector-radix" FFTs). The basic idea is that, if
> one has a loop (vector-rank 1) of transforms, where the vector stride is smaller than
> the transform size, it is advantageous to push the loop towards the leaves of the
> transform decomposition, while otherwise maintaining recursive depth-first ordering,
> rather than looping "outside" the transform; i.e., apply the usual FFT to "vectors"
> rather than numbers.

and the reason it is non-obvious: "Cooley–Tukey produces a unit *input*-stride vector loop
at the top-level DIT decomposition, but with a large *output* stride; this difference in
strides makes it non-obvious whether vector recursion is advantageous for the sub-problem,
but for large transforms we often observe the planner to choose this possibility."

Franchetti et al. derive the same thing as a formula identity — their eq. (29) — and say
explicitly what it looks like in code: "The formula manipulation leading to (29)
manifests itself as loop splitting and loop exchange in the equivalent code"
(**F. Franchetti, M. Püschel, Y. Voronenko, S. Chellappa, J. M. F. Moura, "Discrete
Fourier Transform on Multicores", IEEE Signal Processing Magazine, 2009, special issue on
Signal Processing on Platforms with Multiple Cores**;
[PDF](https://users.ece.cmu.edu/~franzf/papers/spmag09.pdf), §V-C). They also state the
alternative and why it is worse: "One solution is to give up on spatial locality: all
rightmost L's are fused and merged into the first loop as explained before. A better
solution is to translate it into a memory construct, which is indeed possible and done,
e.g., in FFTW."

**Mapping to L = 6/8/17/36:** in this project the "vector" is the batch of volumes
(time slices × spin-colour), and the vector stride is *by construction* whatever layout
you choose. Vector recursion says: choose the layout so the batch index is the innermost,
smallest-stride index, and then the batch loop *is* the SIMD lane index at every leaf.
This turns every strided 3D access into a full-width contiguous vector access and removes
the entire strided-access problem in one move. §5.3 gives the resulting working sets.

### 3.4 Fusing axes to cut the number of passes over the grid

The published single-node result that most closely matches "3D FFT, reduce passes,
avoid transposes" is
**Yi-Qun Liu, Yan Li, Yun-Quan Zhang, Xian-Yi Zhang, "Memory Efficient Two-Pass 3D FFT
Algorithm for Intel® Xeon Phi™ Coprocessor", Journal of Computer Science and Technology
29(6), pp. 989–1002, 2014**, DOI 10.1007/s11390-014-1484-z
([abstract/metadata verified](https://jcst.ict.ac.cn/EN/10.1007/s11390-014-1484-z)).
The technique, in the authors' words, is to "split one dimension into two sub-dimensions,
and then combine the transform along each sub-dimension with one of the rest dimensions
respectively" — turning the conventional **three** passes over the 3D grid into **two**,
specifically to reduce explicit main-memory↔cache transfers and non-contiguous accesses.
Supporting optimisations named in the paper: **memory padding, loop transformation and
vectorization**. Reported results: **136 Gflops with 240 threads in offload mode on an
Intel Xeon Phi 7110P, up to 2.22× faster than Intel MKL.** (Many-core coprocessor, not
our single AVX2 core — quote the *technique*, not the Gflops.)

**Applicability by size.** The two-pass trick pays when a full pass over the grid costs
real DRAM traffic, i.e. when the grid does not fit in cache. Per §5, a *single* volume
never triggers that for us; a *batch* does. For L = 36 with a batch of 64 volumes the
working set is 45.6 MiB (own arithmetic) — larger than the 30 MiB L3 on this node — so
pass-count reduction becomes a first-order concern there and only there. For L = 6 and 8,
where the whole volume is 3.4/8 KiB, "fusing axes" degenerates into "do all three axes
inside one L1-resident kernel invocation", which is the right answer anyway (§9).

### 3.5 Transpose-free multidimensional FFTs: vector-radix

The genuinely transpose-free family is the **vector-radix** multidimensional FFT, which
decimates all dimensions simultaneously rather than doing row-column passes.
`[UNVERIFIED — could not fetch]`: I could not retrieve a primary source in this session.
[IEEE Xplore doc 1170349](https://ieeexplore.ieee.org/document/1170349) (the ICASSP
"Vector radix fast Fourier transform" paper) failed to fetch, and the Springer chapter
"Multidimensional FFT Algorithms" (DOI 10.1007/978-981-13-9965-7_5) is paywalled.
Secondary summaries attribute a radix-2 2D direct FFT to G. E. Rivard (IEEE Trans.
ASSP, 1977) and claim ~25 % fewer multiplies than row-column, but **I have no verified
source for either the attribution or the 25 %, so treat both as unconfirmed.** Someone on
the panel with library access should nail this down; it is the one technique in this
section that could in principle remove the strided passes *algebraically* rather than by
layout tricks. See §10 for my own read on whether it is worth it at L = 6/8/17/36.

The GPU literature reaches the same destination by a different route: VkFFT's own
description of its "four-step FFT algorithm with no transpositions at all" is the
practical existence proof that four-step's twiddle-and-recombine structure can be
implemented purely with index arithmetic. `[UNVERIFIED — could not fetch]`: attempts at
the IEEE Access paper (D. Tolmachev, IEEE Access vol. 11, pp. 12039–12058, 2023) and its
mirrors were refused in this session.

---

## 4. Cache-oblivious recursion in practice, and its coarsening

The results in §1 are asymptotic. Every source that actually shipped code says the same
thing about turning them into a program: **coarsen the base case.**

- Johnson & Frigo, §3.2, verbatim: "Perhaps most importantly, one needs to perform an
  optimization that has almost nothing to do with the caches: the recursion must be
  'coarsened' to amortize the function-call overhead and to enable compiler optimization.
  For example, the simple pedagogical code of algorithm 1 recurses all the way down to
  n = 1, and hence there are ≈ 2n function calls in total, so that every data point
  incurs a two-function-call overhead on average. […] These problems can be effectively
  erased, however, simply by making the base cases larger, e.g. the recursion could stop
  when n = 32 is reached, at which point a highly optimized hard-coded FFT of that size
  would be executed."
- FLPR did the same for their transpose measurements: "the base cases were 'coarsened' by
  inlining the recursion near the leaves to increase their size and overcome the overhead
  of procedure calls."
- **FFTW's actual radix/base-case sizes**, verbatim (Johnson & Frigo §3.3): "for more
  moderate n, FFTW uses depth-first recursion with a bounded radix, similar in spirit to
  algorithm 1 but with much larger radices (**radix 32 is common**) and base cases
  (**size 32 or 64 is common**)". And the four-step threshold: "We currently find that the
  general radix-√n algorithm is beneficial only when n becomes very large, **on the order
  of 2²⁰ ≈ 10⁶**. In practice, this means that we use at most a single step of radix-√n
  (two steps would only be used for n ≳ 2⁴⁰)."
- **SPIRAL's base-case rule**, verbatim (Franchetti et al., §V-C, "Mapping to C code"):
  "On modern deeply pipelined superscalar processors, the recursive FFT has to be
  terminated with a basic block that is sufficiently large but does not cause instruction
  cache misses. Experiments show that a **DFT of a size between 4ν and 32ν** (ν is the
  SIMD vector length) is a good choice." They also flag the cost: "The downside is
  considerably increased code size. For example, FFTW requires several megabytes of C
  code to implement 1D FFTs."

**Mapping to L = 6/8/17/36.** FFTW's own numbers say a hard-coded straight-line codelet of
size 32–64 is where recursion should stop. **All four of our line lengths (6, 8, 17, 36)
are at or below that threshold, so all four should be single straight-line codelets with
no recursion at all.** SPIRAL's rule agrees: for AVX2 complex double, ν = 2 complex
numbers per 256-bit register, so 4ν–32ν = 8–64 — again bracketing all four sizes. For
AVX-512 complex double, ν = 4, giving 16–128, which brackets 17 and 36 and puts 6 and 8
*below* the recommended basic-block size — which is the technical statement of why
L = 6 and L = 8 want to be batched into the vector lanes rather than transformed one
volume at a time (§3.3, §5.3).

There is real measured benefit from recursion *over the grid*, but only when the grid
exceeds cache. Prokop's multipass-filter experiment (a stencil, but the closest thing in
that thesis to our "repeated passes over a 3D array" pattern) on a 167 MHz Sun UltraSPARC
with 16 kB L1 and 512 kB L2: "For arrays that do not fit in L2-cache, the recursive
implementation executes in less than 70 % of the time of the iterative version"
([Prokop 1999](https://ocw.mit.edu/courses/6-895-theory-of-parallel-systems-sma-5509-fall-2003/6dc7de52dcf13b53cebf2fe10ae6752a_cach_oblvs_thsis.pdf), §6.4).
FLPR report the same ratio for matrix transpose on a 450 MHz AMD K6III (32 kB 2-way L1,
64 kB 4-way L2, 1 MB L3, 32-byte lines): "For large matrices, the recursive algorithm
executes in less than 70 % of the time used by the iterative algorithm, even though the
transpose problem exhibits no temporal locality," and they attribute the iterative
version's erratic behaviour to conflict misses — "where limited cache associativity
interacts with the regular addressing of the matrix to cause systematic interference."
Note the sizes: FLPR's own plot runs to N ≈ 600, i.e. matrices up to ~1.4 MB. **Both 70 %
figures are conditioned on "does not fit in cache".** Neither applies to a single volume
of ours; both may apply to a large batch of L = 36.

Finally, the headline number for *why any of this matters*: Franchetti et al., Fig. 1,
DFT single precision on a 2.66 GHz Intel Core i7 (Nehalem) quadcore, sizes 16 to 1M —
"the performance difference between the best and worst is a factor of 12 to 35," broken
down as **memory hierarchy ≈ 5×, vector instructions ≈ 3×, multiple threads ≈ 3×**, with
the worst line being "the implementation from Numerical Recipes based on a standard
radix-2 iterative FFT." *5× is what memory-hierarchy-aware structure is worth on 1D FFTs
across that size range* — but note the range includes sizes far above ours, and the 5×
gap closes at the small end where the whole transform is L1-resident.

---

## 5. Working-set arithmetic for L = 6, 8, 17, 36

> **Everything in §5 is my own arithmetic**, computed from the sizes and cache
> parameters stated. It is not from any cited source. The one *rule* I borrow is
> Franchetti et al.'s cache-capacity heuristic, quoted in §5.2.

Hardware reference points. This repo's node, from `lscpu -C` / `lscpu`:
**Intel Xeon E5-2680 v3 (Haswell), AVX2 (no AVX-512), L1d 32 KiB 8-way 64 sets,
L2 256 KiB 8-way, L3 30 MiB 20-way, 64 B lines, 2 sockets × 12 cores,
`Hugepagesize: 2048 kB`, `pdpe1gb` present (1 GiB pages supported).** I also carry the
brief's "typical" figures (32 KiB L1d / 1 MiB L2 / 32 MiB L3) because newer server parts
(Ice Lake, Sapphire Rapids) have L2 in the 1–2 MiB range, and as §5.1 shows, L = 36
lands squarely on that difference.

### 5.1 Per-volume footprints (complex double, 16 B/point)

| L | N = L³ | pencil (L) | plane (L²) | volume (L³) | ≤ 32 KiB L1d? |
|---|---|---|---|---|---|
| 6 | 216 | 96 B | 576 B (0.56 KiB) | **3 456 B = 3.375 KiB** | volume **fits** (9.5×) |
| 8 | 512 | 128 B | 1 024 B (1.00 KiB) | **8 192 B = 8.000 KiB** | volume **fits** (4×) |
| 17 | 4 913 | 272 B | 4 624 B (4.52 KiB) | **78 608 B = 76.77 KiB** | volume no; **plane fits** (7×) |
| 36 | 46 656 | 576 B | 20 736 B (20.25 KiB) | **746 496 B = 729.0 KiB** | volume no; **plane fits** (1.58×) |

Residency verdicts, data only:

| capacity | complete volumes that fit |
|---|---|
| L1d 32 KiB | L=6: **9** · L=8: **4** · L=17: 0 · L=36: 0 |
| L2 256 KiB (Haswell) | L=6: 75 · L=8: 32 · L=17: **3** · L=36: 0 |
| L2 1 MiB (Ice Lake / SPR class) | L=6: 303 · L=8: 128 · L=17: 13 · L=36: **1** |
| L3 30 MiB (this node) | L=6: 9 102 · L=8: 3 840 · L=17: 400 · L=36: **42** |
| L3 32 MiB | L=6: 9 709 · L=8: 4 096 · L=17: 426 · L=36: 44 |

With a 3× working set (input + output + twiddle table, ≈48 B/point): L1d holds 3 volumes
at L=6, 1 at L=8, 0 at L=17 and L=36; a 1 MiB L2 holds 101 / 42 / 4 / **0**; a 30 MiB L3
holds 3 034 / 1 280 / 133 / **14**.

**The four verdicts:**
- **L = 6 (3.375 KiB)** — L1-resident with a factor of 9 to spare. Room for 2 volumes plus
  in/out/twiddles, or 8 volumes of pure data. Never memory-bound per volume; entirely
  register/issue-bound. Batching is mandatory to get any bandwidth utilisation at all.
- **L = 8 (8 KiB)** — L1-resident, exactly one quarter of L1d. In/out/twiddle for one
  volume is 24 KiB = 75 % of L1d: comfortable but tight. This is the classic
  "register-resident radix-8" regime.
- **L = 17 (76.77 KiB)** — **not** L1-resident (2.4× over) but very comfortably L2-resident
  on every current part. Its L² plane is 4.52 KiB, so plane-at-a-time blocking is
  L1-resident with 7× headroom — which matters, because Rader at L = 17 needs a 16-point
  cyclic-convolution scratch buffer per line, plus the 16-element permuted twiddle
  sequence; at 4.52 KiB per plane there is ample room for both in L1.
- **L = 36 (729 KiB)** — **the interesting one, and the only architecture-sensitive one.**
  On this Haswell node (256 KiB L2) a single volume does **not** fit in L2 and lives in
  L3. On a 1 MiB-L2 part it fits in L2 with 29 % to spare *for data only* — and does
  **not** fit once you add output and twiddles (2.13 MiB at 48 B/point). Its L² plane is
  20.25 KiB, which fits 32 KiB L1d with only 1.58× headroom — tight enough that you must
  be deliberate about twiddle storage, and a strong argument for Good–Thomas at L = 36
  (4 and 9 coprime ⇒ **no inter-factor twiddle array at all**, so the L1 budget is data
  plus a handful of 4- and 9-point constants).

### 5.2 Cross-check against a published capacity rule

Franchetti et al. give an explicit rule, verbatim: "For a given level, we call the
capacity N if it can hold the working set for the computation of y = Ax for an N × N
matrix A. This implies that the input vector x, the output vector y, and all necessary
temporary arrays and constants fit into that cache level. For instance, if we consider
double-precision, one (complex) value is 16 bytes. If A is a DFT, **N is the cache size
divided by 64** (assuming a factor of 4 space overhead)."

Applying that rule (my arithmetic, their rule): 32 KiB L1d → N_max = **512** points;
256 KiB L2 → **4 096**; 1 MiB L2 → **16 384**; 30 MiB L3 → **491 520**.

This is a startlingly clean match to our four sizes:

| L | N = L³ | verdict under the /64 rule |
|---|---|---|
| 6 | 216 | L1-resident (42 % of N_max = 512) |
| 8 | 512 | **exactly** N_max for 32 KiB L1d |
| 17 | 4 913 | just over the 256 KiB-L2 N_max of 4 096; comfortably inside a 1 MiB L2 |
| 36 | 46 656 | L3 only under the rule; ~9.5 % of the 30 MiB L3 N_max |

They also state their measured working-set convention, verbatim: "For the Spiral
generated libraries, the working set for input size n is **6n real numbers** or
**4 1/16 n real numbers if the twiddle factors are computed on the fly**." For complex
double that is 48 B/point vs. 32.5 B/point — a **32 % footprint reduction from on-the-fly
twiddles**, which is exactly the lever that puts a single L = 36 volume back inside a
1 MiB L2 (46 656 × 32.5 B = 1.45 MiB — still no; but a plane-blocked L = 36 with
on-the-fly twiddles is trivially L1-resident). Their headline residency observation on the
Core i7 (private 64 kB L1, 256 kB L2, 8 MB shared L3, 64 B lines): "in Fig. 5(a), an FFT
is L3 cache resident up to n = 2¹⁷", and "For sizes outside the shared L3 cache, the
performance drops as the computation becomes memory bound. At this point buffering, vector
recursion, and on-the-fly twiddle computation become crucial." Note 2¹⁷ = 131 072 — again
above 36³.

### 5.3 The batched picture: SIMD-over-batch footprints

This is the layout §3.3 argues for: interleave V volumes so the batch index is innermost
and each vector lane is a different volume. V = 4 for AVX2 (4 doubles per 256-bit
register, split-complex/SoA), V = 8 for AVX-512. **My own arithmetic:**

| L | V | pencil (V·L) | plane (V·L²) | volume (V·L³) |
|---|---|---|---|---|
| 6 | 4 | 384 B | 2 304 B (2.25 KiB) | 13 824 B = 13.5 KiB — **fits L1d** |
| 6 | 8 | 768 B | 4 608 B (4.50 KiB) | 27 648 B = 27.0 KiB — **fits L1d** |
| 8 | 4 | 512 B | 4 096 B (4.00 KiB) | 32 768 B = **exactly 32 KiB** — L1d with zero slack |
| 8 | 8 | 1 024 B | 8 192 B (8.00 KiB) | 65 536 B = 64 KiB — L2 |
| 17 | 4 | 1 088 B | 18 496 B (18.06 KiB) | 314 432 B = 307 KiB — 1 MiB L2 yes, 256 KiB L2 no |
| 17 | 8 | 2 176 B | 36 992 B (36.12 KiB) | 628 864 B = 614 KiB — 1 MiB L2 |
| 36 | 4 | 2 304 B | 82 944 B (81.00 KiB) | 2 985 984 B = 2.85 MiB — **L3** |
| 36 | 8 | 4 608 B | 165 888 B (162.0 KiB) | 5 971 968 B = 5.70 MiB — **L3** |

**What the table says.**
- **L = 6 is the sweet spot for SIMD-over-batch**: even 8 interleaved volumes (27 KiB) sit
  in L1d, so an AVX-512 kernel can hold an entire 8-volume 6³ transform in L1 and do all
  three axes without leaving it. On AVX2 (V = 4, 13.5 KiB) you can hold *two* such groups.
- **L = 8 at V = 4 hits 32 KiB exactly** — i.e. it will thrash a 32 KiB L1d as soon as you
  add twiddles or an output buffer. Either use V = 2 (16 KiB), or keep V = 4 and block by
  planes (4 KiB per V-plane), or work in place with on-the-fly twiddles.
- **L = 17 at V = 4 (307 KiB)** is L2 work on any modern part but overflows a 256 KiB
  Haswell L2. Block by V-planes (18.06 KiB) to stay L1-resident.
- **L = 36 must be blocked, period.** The V-volume never fits below L3, and at V = 8 even
  a single V-plane is 162 KiB — over L1d by 5×. For L = 36 the L1-resident tile is a
  V-pencil group, not a V-plane: a 36-long V-pencil is only 2.3 KiB (V=4) or 4.6 KiB
  (V=8), so you can hold a few hundred pencils' worth of a tile in L1. **Concretely: for
  L = 36, block the y and z passes into tiles of ~8 × 8 pencils** — 8·8·4.6 KiB ≈ 295 KiB
  is too much, so ~4 × 4 pencils (≈74 KiB at V=8, ≈37 KiB at V=4) is the L1-scale unit;
  tune from there.

### 5.4 When does the batch make the whole thing memory-bound?

**My own arithmetic.** Arithmetic intensity of a row-column 3D FFT if every pass streams
from memory (3 passes × read+write × 16 B/point = 96N bytes) against 5N log₂N flops:

| L | flops (5N log₂N) | 3-pass traffic | intensity (3 passes) | intensity (1 fused pass) |
|---|---|---|---|---|
| 6 | 8 375 | 20 736 B | 0.404 flop/B | 1.212 flop/B |
| 8 | 23 040 | 49 152 B | 0.469 flop/B | 1.406 flop/B |
| 17 | 301 226 | 471 648 B | 0.639 flop/B | 1.916 flop/B |
| 36 | 3 618 120 | 4 478 976 B | 0.808 flop/B | 2.423 flop/B |

A single AVX2 FMA core at 2.5 GHz has a double-precision peak near 40 Gflop/s, against a
realistic single-core DRAM read+write bandwidth on the order of 10–15 GB/s — a machine
balance around **3 flop/byte**. Every entry in that table is below 3, so **a batch large
enough to spill L3 is memory-bound at every one of our four sizes, and the only lever that
moves the needle is reducing the number of passes over the grid** (the two-pass idea of
§3.4, or better: one fused pass, right column — which is what "all three axes inside one
L1-resident tile" buys you, at 1.2–2.4 flop/B, still short of balance but 3× better).

L3 spill thresholds (30 MiB, data only, own arithmetic): L=6 at ~9 100 volumes;
L=8 at ~3 840; L=17 at ~400; **L=36 at ~42**. At 3× working set: 3 034 / 1 280 / 133 /
**14**. So an LQCD momentum projection over, say, 64 time slices × 12 spin-colour
components = 768 volumes is comfortably L3-resident at L = 6 and L = 8, marginal at
L = 17, and **firmly DRAM-streaming at L = 36** (768 × 729 KiB = 547 MiB).

---

## 6. Conflict misses, padding, and power-of-two strides

FFTW3's own statement of the problem, verbatim (§IV-D, and the same text in Johnson &
Frigo §3.3): "there are many other techniques that FFTW employs to supplement the basic
recursive strategy, mainly to address the fact that cache implementations strongly favor
accessing consecutive data—thanks to cache lines, limited associativity, and direct
mapping using low-order address bits (**accessing data at power-of-two intervals in
memory, which is distressingly common in FFTs, is thus especially prone to cache-line
conflicts**)." Bailey made the same point 15 years earlier from the vector-supercomputer
side: "many modern computers, particularly those with interleaved main memories, do very
poorly with data that is accessed with a memory stride that is a large power of two."

FFTW's countermeasures, verbatim: "the data for several butterflies at a time can be
copied to a small buffer before computing and then copied back, where the copies and
computations involve more consecutive access than doing the computation directly
in-place"; and buffering plans exist because "if the input/output arrays are noncontiguous
in memory, operating on a contiguous buffer might be faster because of better interaction
with caches and the rest of the memory subsystem" (FFTW3 §IV-C.7).

The systematic treatment is
**C. Hong, W. Bao, A. Cohen, S. Krishnamoorthy, L.-N. Pouchet, F. Rastello, J. Ramanujam,
P. Sadayappan, "Effective padding of multidimensional arrays to avoid cache conflict
misses", PLDI 2016 / ACM SIGPLAN Notices 51(6), pp. 129–144, 2016**,
DOI [10.1145/2908080.2908123](https://doi.org/10.1145/2908080.2908123)
([metadata + abstract verified](https://repository.lsu.edu/eecs_pubs/1567/)). From the
abstract, verbatim: "Caches are used to significantly improve performance. Even with high
degrees of set associativity, the number of accessed data elements mapping to the same set
in a cache can easily exceed the degree of associativity." They give "the first algorithms
for optimal padding of arrays for a set associative cache for arbitrary tile sizes, and
the first solution to padding for nested tiles and multi-level caches."
`[Full text not fetched — only title/authors/venue/pages/DOI and abstract verified. A
"padding by 8 elements improves MKL FFT by over 250 %" figure appears in secondary
summaries of this paper; I could not verify it and it should not be quoted.]`

Liu et al.'s two-pass 3D FFT lists **memory padding** among its three enabling
optimisations (§3.4) — independent confirmation that padding is a first-class technique
for 3D FFT specifically, not just a general array trick.

### Does any of our four sizes have a conflict problem?

**My own arithmetic**, for the 32 KiB / 8-way / 64-set / 64 B-line L1d on this node
(set index = ⌊addr/64⌋ mod 64):

| L | y-stride | z-stride | z-stride in lines | distinct L1 sets over one z-pencil | elements per set |
|---|---|---|---|---|---|
| 6 | 96 B | 576 B | 9 | 6 of 6 | 1.00 |
| 8 | 128 B | 1 024 B | 16 | 4 (of 8 elements) | **2.00** |
| 17 | 272 B | 4 624 B | 72.25 — *not* line-aligned | sets rotate | no systematic pattern |
| 36 | 576 B | 20 736 B | 324 (324 mod 64 = 4) | 16 (of 36 elements) | 2.25 |

Nothing here is pathological — 2.25 elements per set against 8-way associativity is fine,
and in any case at L = 6 and 8 the *entire volume* fits in L1 so conflicts are moot.
The reason our sizes escape the classic FFT conflict disaster is that **only L = 8 is a
pure power of two, and 8³ = 512 points = 8 KiB is small enough that the whole thing is
L1-resident.** The genuinely power-of-two-hostile case would be a large power-of-two grid
(256³ etc.), which we do not have.

**Where padding *does* become relevant for us: the batch stride.** If you lay out B
volumes contiguously at L = 8, the volume-to-volume distance is exactly 8 192 B = 128
lines = 2 pages. 128 mod 64 = 0, so **volume k and volume k+1 map to identical L1 sets at
every offset.** With 8-way associativity, gathering the same (x,y,z) from 9 or more
volumes — precisely what a SIMD-over-batch layout does if you use array-of-volumes with
`dist = 8192` — will conflict-miss on L1. **Recommendation (my own): pad the volume stride
at L = 8 by one cache line (`dist = 8192 + 64` B), or better, use a genuinely interleaved
SoA layout so the batch dimension is contiguous and the question never arises.** The same
arithmetic at L = 6 (3 456 B = 54 lines, 54 mod 64 = 54, coprime-ish with 64) and L = 17
(78 608 B, not line-aligned at all) is benign; at L = 36 the volume stride is 746 496 B =
11 664 lines, and 11 664 mod 64 = 16, so only 4 distinct set-groups — worth padding too.

The interface for expressing all of this is FFTW's advanced/guru layout, which is a useful
model even though we are not calling FFTW: `fftw_plan_many_dft(rank, n, howmany, in,
inembed, istride, idist, out, onembed, ostride, odist, sign, flags)`, where "the input of
the k-th transform is at location `in+k*idist`" and "the (j,k)-th element is at
`j*stride+k*dist`"
([FFTW 3.3.11 manual, Advanced Complex DFTs](https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html)).
The manual also confirms the batching payoff in principle: "Plans obtained in this way can
often be faster than calling FFTW multiple times for the individual transforms." **The
`idist`/`odist` parameters are exactly the knobs you want to pad.**

---

## 7. TLB and page size

Mechanism, verbatim from
**Ulrich Drepper, "What Every Programmer Should Know About Memory", Red Hat Inc.,
Version 1.0, November 21, 2007** ([PDF](https://akkadia.org/drepper/cpumemory.pdf)):

> The TLB cache is quite small since it has to be extremely fast. If more pages are
> accessed repeatedly than the TLB cache has entries for, the translation from virtual to
> physical address has to be constantly repeated. This is a very costly operation. With
> larger element sizes the cost of a TLB lookup is amortized over fewer elements.

and, crucially for strided 3D access: "Since the physical address has to be computed
before a cache line can be read for either L2 or main memory **the address translation
penalties are additive to the memory access times.**" His §3.3 experiment isolates the
effect: placing each list element on its own page produces "the huge spike starting when
the working set size reaches 2¹³ bytes. This is when the TLB cache overflows. With an
element size of 64 bytes we can compute that the TLB cache has 64 entries" (on his test
machine; the point is the shape, not the number).

Page sizes, verbatim (§4.3.2): "the x86/x86-64 processors have a normal page size of 4 kB
but they can also use 4MB and 2MB pages respectively," and the cost side: "On Linux it is
therefore necessary to allocate these big pages at system start time using the special
`hugetlbfs` filesystem. A fixed number of physical pages are reserved for exclusive use
as big virtual pages. This ties down resources which might not always be used. […] Still,
huge pages are the way to go in situations where performance is a premium, resources are
plenty, and cumbersome setup is not a big deterrent."

I could **not** verify per-microarchitecture TLB entry counts in this session. The Agner
Fog manual (**Agner Fog, "The microarchitecture of Intel, AMD and VIA CPUs", last updated
2026-05-23**, [PDF](https://www.agner.org/optimize/microarchitecture.pdf) — fetched and
searched) does not tabulate Haswell dTLB/STLB entry counts in a form I could extract, and
[wikichip](https://en.wikichip.org/wiki/intel/microarchitectures/haswell_(client))
refused the connection. **So: no TLB entry count is asserted anywhere in this document.**
`[UNVERIFIED — could not fetch]` Implementers should read the count off their own machine
(`cpuid` leaf 0x18, or `perf stat -e dTLB-load-misses`) rather than trusting a number from
memory.

### The arithmetic that actually matters (my own)

Pages touched, complex double, 4 KiB pages vs. 2 MiB pages:

| L | volume | 4 KiB pages / volume | fraction of a 2 MiB page |
|---|---|---|---|
| 6 | 3 456 B | 1 | 0.0017 |
| 8 | 8 192 B | 2 | 0.0039 |
| 17 | 78 608 B | 20 | 0.037 |
| 36 | 746 496 B | **183** | 0.356 |

Batched:

| L | batch | total | 4 KiB pages | 2 MiB pages |
|---|---|---|---|---|
| 6 | 1 024 | 3.38 MiB | 864 | **2** |
| 8 | 1 024 | 8.00 MiB | 2 048 | **4** |
| 8 | 4 096 | 32.00 MiB | 8 192 | **16** |
| 17 | 1 024 | 76.77 MiB | 19 652 | **39** |
| 36 | 64 | 45.56 MiB | 11 664 | **23** |
| 36 | 1 024 | 729.0 MiB | 186 624 | **365** |

**Reading.** Second-level TLBs on server x86 parts are in the low thousands of entries, so
the 4 KiB-page column crosses "no TLB reach" somewhere around a batch of a few hundred
volumes at L = 8/17 and around a batch of **~10 volumes** at L = 36. The 2 MiB column
never exceeds a few hundred entries even at the largest batches shown. **Concrete
recommendation (own analysis): allocate the batch arena with `MAP_HUGETLB` / 2 MiB
transparent huge pages for any batch above a few MiB, and unconditionally for L = 36.**
`Hugepagesize: 2048 kB` and `pdpe1gb` are both present on this node, so 2 MiB and 1 GiB
pages are both available; note `HugePages_Total: 0` at the time of writing, so the pool
needs configuring before `hugetlbfs` will work — transparent huge pages
(`madvise(MADV_HUGEPAGE)`) are the lower-friction route.

Stride-vs-page interaction, also my own arithmetic. Gathering one z-pencil touches:

| L | z-stride | z-stride mod 4096 | distinct 4 KiB pages touched by one z-pencil |
|---|---|---|---|
| 6 | 576 B | 576 | 1 |
| 8 | 1 024 B | 1 024 | 2 |
| 17 | 4 624 B | 528 | **17** (one per element) |
| 36 | 20 736 B | 256 | **36** (one per element) |

So at L = 17 and L = 36 **every element of a z-pencil is on a different 4 KiB page.** A
single z-pencil therefore consumes 17 or 36 dTLB entries; a tile of 16 pencils consumes
the same 36 pages (the pencils are adjacent in x, hence on the same pages) — which is
precisely why **tiling the strided passes is a TLB optimisation as much as a cache
optimisation**, and why the naive "loop over all z-pencils one at a time" ordering is the
worst possible choice at L = 36. With 2 MiB pages, an entire 36³ volume is 0.36 of one
page and the whole issue evaporates.

---

## 8. Streaming / non-temporal stores

### 8.1 Mechanism

Verbatim from Drepper §6.1: "These non-temporal write operations do not read a cache line
and then modify it; instead, the new content is directly written to memory. This might
sound expensive but it does not have to be. The processor will try to use write-combining
to fill entire cache lines. If this succeeds no memory read operation is needed at all."
The relevant intrinsics he lists include `_mm_stream_pd(double *p, __m128d a)` and
`_mm_stream_si128`; the AVX/AVX2 equivalents are `_mm256_stream_pd` / `_mm512_stream_pd`.

The traffic being avoided is the write-allocate / read-for-ownership transfer, described
verbatim in
**J. Laukemann, T. Gruber, G. Hager, D. Oryspayev, G. Wellein, "CloverLeaf on Intel
Multi-Core CPUs: A Case Study in Write-Allocate Evasion", arXiv:2311.04797**
([PDF](https://arxiv.org/pdf/2311.04797)):

> A transfer triggered by a write miss is called a write-allocate (WA) or
> read-for-ownership (RFO). […] Of course, if a core overwrites the whole cache line
> anyway, the WA transfer is basically unnecessary. Many CPU architectures provide
> mechanisms to avoid these transfers; one option are special store instructions with
> non-temporal hints. The x86 and ARM architectures offer non-temporal (NT) store
> instructions (a.k.a. streaming stores), which bypass the normal cache hierarchy and
> write directly to memory.

with the footnote that "NT stores actually write to a small write-combine buffer which is
later flushed to memory." That paper also documents Intel's *automatic* WA evasion
("SpecI2M", Ice Lake SP onward) and reports that it "is generally less effective in WA
evasion than non-temporal store instructions; the combination of both together with
minimal code changes resulted in best performance and lowest code balance on the full
node." Their measured gain from adding NT stores plus loop rearrangement to CloverLeaf on
an Ice Lake SP server: **"on average 5.8 % lower code balance with a maximum of 23.2 %"**
across the hotspot loops (full node, 72 cores). They also report SpecI2M "only avoids
roughly 50 % of the write-allocates in the best case," and that on Sapphire Rapids 8480+
they "could even observe deterioration."

### 8.2 When NT stores *hurt* — the two hard numbers

This is the part implementers get wrong, and there are two clean measured datapoints.

**Drepper Table 6.1** (initialising a 3000 × 3000 matrix, IA-32, "non-temporal hint"
stores vs. ordinary stores):

| inner-loop increment | Normal | Non-Temporal |
|---|---|---|
| Row (sequential) | 0.048 s | **0.048 s** |
| Column (strided) | 0.127 s | **0.160 s** |

His reading, verbatim: sequential NT is as fast as cached "because the processor is
performing write-combining"; but for the column case "no write combining is possible and
each memory cell must be addressed individually. This requires constantly selecting new
rows in the RAM chips with all the associated delays. **The result is a 25 % worse result
than the cached run.**"

**Georg Hager, "A case for the non-temporal store", 4 September 2008**
([blog post](https://blogs.fau.de/hager/archives/2103) — a blog, but by one of the authors
of the CloverLeaf paper above, and the clearest statement of the rule) makes the
complementary point verbatim: "Imagine you have a code whose working set fits into the
outer-level cache and which produces data to be stored in memory. Using NT stores for such
a code will probably slow it down." He notes `movntpd` "writes the contents of a full
16-byte SSE register […] to memory" and that "the memory address has to be a multiple of
16, else an exception will be generated."

### 8.3 Verdict for L = 6, 8, 17, 36

**My own analysis, from §5 + the two rules above.** NT stores pay only when (a) the store
stream is contiguous and full-cache-line, and (b) the destination will not be re-read soon.

- **Inside the transform of any single volume: never.** All four sizes are L1- or
  L2-resident (§5.1), and every output element is re-read by the next axis pass. Using NT
  stores here is precisely Hager's "working set fits into the outer-level cache" case —
  it will slow you down.
- **On the final write-out of a batched result: yes, conditionally.** If the batch exceeds
  L3 (§5.4: ~9 100 volumes at L = 6, ~3 840 at L = 8, ~400 at L = 17, **~42 at L = 36** on
  a 30 MiB L3) and the consumer will not touch the output immediately, the last pass's
  stores are a pure streaming write and NT stores remove the RFO traffic — worth roughly
  the write-allocate share of the traffic, i.e. up to 1/3 of a read-1-write-1 stream.
- **Never on a strided/transposing store.** Drepper's column-wise row of the table is
  exactly the "NT store into a transposed layout" case, and it was **25 % slower** than
  ordinary stores. If you are tempted to use NT stores to write the output of a
  transposing pass, don't. Fix the layout instead (§3.3).
- **Alignment.** NT stores require natural alignment (16 B for SSE; 32/64 B for
  AVX/AVX-512 variants). Combined with §6's padding recommendation, pad the volume stride
  by a multiple of the vector width, not by an odd number of complex doubles.

---

## 9. Per-size synthesis

| | **L = 6 (216 pts, 3.375 KiB)** | **L = 8 (512 pts, 8 KiB)** | **L = 17 (4 913 pts, 76.77 KiB)** | **L = 36 (46 656 pts, 729 KiB)** |
|---|---|---|---|---|
| Residency (1 volume, data) | L1d ×9.5 | L1d ×4 | L2 (not L1) | L3 on Haswell; L2 on 1 MiB-L2 parts |
| Blocking unit | whole volume, ×2–8 volumes | whole volume | **plane** (4.52 KiB) | **plane** (20.25 KiB) or pencil-tile |
| Four/six-step for the 1D line? | **no** (Bailey crossover ≈ n 4096) | **no** | **no** | **no** |
| Recursion? | **no** — straight-line codelet (FFTW: base cases 32–64) | **no** | **no** — 17 is at the top of FFTW's base-case range | **no** for the line; yes for tiling the grid |
| Transposes | none; fuse all 3 axes in L1 | none; fuse all 3 axes in L1 | none; plane-blocked row-column | none; tile-blocked row-column |
| Twiddles | Good–Thomas 2×3 ⇒ **no inter-factor twiddles** | radix-8: small constant set | Rader: 16-pt convolution buffer + permuted seq., 4.5 KiB plane leaves room in L1 | Good–Thomas 4×9 ⇒ **no inter-factor twiddles** — worth ~1/3 of footprint (Franchetti's 6n → 4 1/16 n) |
| SIMD-over-batch V | V = 8 keeps 27 KiB in L1d — ideal | **V = 4 hits 32 KiB exactly — pad or drop to V = 2** | V = 4 → 307 KiB (L2); block by V-plane 18 KiB | V-volume never below L3; block by pencil-tile |
| Conflict-miss risk | low (54-line stride) | **volume stride 8 192 B = 128 lines ≡ 0 mod 64 sets → pad `dist`** | low (unaligned stride) | volume stride ≡ 16 mod 64 sets → pad `dist` |
| 4 KiB pages / volume | 1 | 2 | 20 | **183**; every z-pencil element on its own page |
| Huge pages | above ~1 000 volumes | above ~500 volumes | above ~100 volumes | **always** |
| L3 spill (30 MiB, 3× WS) | ~3 000 volumes | ~1 280 | ~133 | **~14** |
| NT stores | final write-out only, large batch | final write-out only, large batch | final write-out only | **yes on final write-out** — the only size where a modest batch is already DRAM-streaming |
| Memory-bound? | only at huge batch | only at huge batch | at moderate batch | **immediately, at realistic LQCD batch sizes** |

---

## 10. Unsourced engineering notes

*Nothing in this section is attributed to any source. It is my own reasoning from the
material above, offered because the panel asked for usable judgement, not because a paper
says it.*

1. **The right mental model for this project is a strided batched kernel, not an FFT.**
   Once the batch index is in the SIMD lanes, all three axes of the L³ transform become
   *identical* strided loops over vector registers, differing only in stride (V·1, V·L,
   V·L²). There is then no "hard axis" and no reason to transpose. The whole
   four-step/six-step/cache-oblivious literature exists to solve a problem
   (n ≫ cache) that we do not have; what we have is a layout problem.

2. **In-register transposes are a completely different animal from array transposes.**
   §3.1's argument rules out materialising a permuted copy of the volume. It says nothing
   against `_mm256_unpacklo_pd`-style 4×4 register transposes to convert between
   interleaved-complex and split-complex, or to feed a codelet that wants a different lane
   order. Those cost shuffle throughput, not memory traffic, and are essentially free
   relative to the loads they replace.

3. **Vector-radix (§3.5) is probably not worth chasing at these sizes**, despite being the
   "correct" transpose-free algorithm. Its advantage is multiply count, and at L = 6/8 the
   whole volume is L1-resident so multiply count is already the binding constraint — but
   the fully unrolled 3D codelet you would write to exploit it is 216 or 512 points of
   straight-line code with the register pressure of a 3D butterfly network, which will
   spill on 16 architectural YMM registers. Row-column with the batch in the lanes keeps
   register pressure at O(L) instead of O(L³). If someone does try it, L = 6 is the size
   to try it on.

4. **The one place I would spend real effort on blocking is L = 36 with a large batch**,
   because that is the only configuration in the brief that is genuinely DRAM-bound at
   realistic problem sizes (§5.4: 42 volumes fills a 30 MiB L3). There, the priority order
   is: (i) 2 MiB pages, (ii) fuse all three axes into a single pass over pencil-tiles so
   the grid is traversed once rather than three times (the intensity goes from 0.81 to
   2.42 flop/B — §5.4), (iii) Good–Thomas 4×9 so there is no twiddle array competing for
   L1, (iv) NT stores on the final write, (v) pad the volume stride.

5. **A cheap experiment that will settle most of the above in an afternoon**: write the
   L = 36 kernel twice — once with three separate stride-parameterised passes over the
   whole batch, once with a single fused pass over 4×4 pencil-tiles — and measure both with
   `perf stat -e cache-misses,dTLB-load-misses,LLC-loads` at batch sizes 8, 64, 512. The
   pass-count reduction, the TLB effect and the L3 spill point will all be visible in one
   sweep, and the answer will tell you whether any of §§2–4 is worth implementing at all.

6. **Don't tune to this node's 256 KiB L2.** L = 36 is the size that flips residency
   between a Haswell-class 256 KiB L2 and an Ice Lake/Sapphire Rapids 1–2 MiB L2 (§5.1).
   If the code will run on both, block to the *smaller* L2 (or better, to L1 by planes,
   which is safe everywhere) rather than betting on the larger.

---

## 11. Bibliography with verification status

**Fetched and read in this session (primary sources):**

1. M. Frigo, C. E. Leiserson, H. Prokop, S. Ramachandran, *Cache-Oblivious Algorithms*,
   ACM Transactions on Algorithms **8**(1), Article 4, January 2012, 21 pp. (originally
   FOCS '99). <https://g-trees.github.io/g_trees/assets/references/frigo1999cache.pdf> ✔
2. H. Prokop, *Cache-Oblivious Algorithms*, M.Sc. thesis, Dept. of EECS, MIT, June 1999
   (supervised by C. E. Leiserson).
   <https://ocw.mit.edu/courses/6-895-theory-of-parallel-systems-sma-5509-fall-2003/6dc7de52dcf13b53cebf2fe10ae6752a_cach_oblvs_thsis.pdf> ✔
3. D. H. Bailey, *FFTs in External or Hierarchical Memory*, Journal of Supercomputing
   **4**(1), March 1990, pp. 23–35 (PDF is the 30 Dec 1989 preprint).
   <https://www.davidhbailey.com/dhbpapers/fftq.pdf> ✔
4. M. Frigo, S. G. Johnson, *The Design and Implementation of FFTW3*, Proceedings of the
   IEEE **93**(2), 2005, pp. 216–231. <https://www.fftw.org/fftw-paper-ieee.pdf> ✔
5. S. G. Johnson, M. Frigo, *Implementing FFTs in Practice*, ch. 11 in *Fast Fourier
   Transforms*, C. S. Burrus (ed.), Rice Univ.: Connexions, 2008.
   <https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf> ✔
   (also mirrored as arXiv:2602.23525, submitted 26 Feb 2026 —
   <https://arxiv.org/abs/2602.23525> ✔ metadata verified)
6. F. Franchetti, M. Püschel, Y. Voronenko, S. Chellappa, J. M. F. Moura, *Discrete
   Fourier Transform on Multicores*, IEEE Signal Processing Magazine, 2009, special issue
   on "Signal Processing on Platforms with Multiple Cores".
   <https://users.ece.cmu.edu/~franzf/papers/spmag09.pdf> ✔
   (volume/issue/pages not printed on the fetched PDF; secondary sources say vol. 26,
   no. 6 — treat the vol/issue as unconfirmed)
7. U. Drepper, *What Every Programmer Should Know About Memory*, Red Hat, Inc.,
   Version 1.0, 21 November 2007. <https://akkadia.org/drepper/cpumemory.pdf> ✔
8. J. Laukemann, T. Gruber, G. Hager, D. Oryspayev, G. Wellein, *CloverLeaf on Intel
   Multi-Core CPUs: A Case Study in Write-Allocate Evasion*, arXiv:2311.04797.
   <https://arxiv.org/pdf/2311.04797> ✔
9. G. Hager, *A case for the non-temporal store*, blog post, 4 September 2008.
   <https://blogs.fau.de/hager/archives/2103> ✔ (blog, not peer-reviewed)
10. FFTW 3.3.11 manual, *Advanced Complex DFTs*.
    <https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html> ✔
11. A. Fog, *The microarchitecture of Intel, AMD and VIA CPUs*, last updated 2026-05-23.
    <https://www.agner.org/optimize/microarchitecture.pdf> ✔ (fetched; did **not** yield
    the Haswell TLB numbers I was looking for)

**Metadata/abstract verified, full text not fetched:**

12. Y.-Q. Liu, Y. Li, Y.-Q. Zhang, X.-Y. Zhang, *Memory Efficient Two-Pass 3D FFT
    Algorithm for Intel® Xeon Phi™ Coprocessor*, Journal of Computer Science and
    Technology **29**(6), 2014, pp. 989–1002, DOI 10.1007/s11390-014-1484-z.
    <https://jcst.ict.ac.cn/EN/10.1007/s11390-014-1484-z> ✔ (abstract + performance
    figures verified from the journal's own landing page)
13. C. Hong, W. Bao, A. Cohen, S. Krishnamoorthy, L.-N. Pouchet, F. Rastello,
    J. Ramanujam, P. Sadayappan, *Effective padding of multidimensional arrays to avoid
    cache conflict misses*, PLDI 2016 / ACM SIGPLAN Notices **51**(6), pp. 129–144,
    DOI 10.1145/2908080.2908123. <https://repository.lsu.edu/eecs_pubs/1567/> ✔
    (metadata + abstract only; PLDI PDF not fetched)

**`[UNVERIFIED — could not fetch]` — cited only via a source I did read, or not at all:**

14. J.-W. Hong, H. T. Kung, *I/O complexity: the red-blue pebble game*, Proc. 13th STOC,
    1981. Attempted <https://apps.dtic.mil/sti/tr/pdf/ADA104739.pdf> (no output) and
    <https://scispace.com/pdf/i-o-complexity-the-red-blue-pebble-game-3auurd44wt.pdf>
    (HTTP 403). The FFT lower bound is quoted here **via** refs 1 and 5.
15. A. Aggarwal, J. S. Vitter, *The Input/Output Complexity of Sorting and Related
    Problems*, Comm. ACM **31** (1988), pp. 1116–1127. Bibliographic data as printed in
    Bailey's reference list [2] (ref. 3 above); primary not fetched.
16. W. M. Gentleman, G. Sande, *Fast Fourier Transforms — For Fun and Profit*, AFIPS Proc.
    **29** (1966), pp. 563–578. As printed in Bailey's reference list [8]; primary not
    fetched.
17. D. Takahashi, *A blocking algorithm for FFT on cache-based processors*, HPCN Europe
    2001, LNCS **2110**, Springer, pp. 551–554, DOI 10.1007/3-540-48228-8_58. As printed
    in FFTW3's reference list [34] (ref. 4 above); Springer refused the PDF. **The MFLOPS
    figures attributed to this paper in secondary summaries are not verified and should
    not be quoted.**
18. M. Dow, *Transposing a matrix on a vector computer*, Parallel Computing **21**(12),
    1995, pp. 1997–2005; and E. G. Cate, D. W. Twigg, *Algorithm 513: Analysis of in-situ
    transposition*, ACM TOMS **3**(1), 1977, pp. 104–110. As printed in FFTW3's reference
    list [36], [37]; primaries not fetched.
19. Vector-radix multidimensional FFT (G. E. Rivard, IEEE Trans. ASSP, 1977, and the
    ICASSP "Vector radix fast Fourier transform" paper). Attempted
    <https://ieeexplore.ieee.org/document/1170349> (no output) and
    <https://link.springer.com/chapter/10.1007/978-981-13-9965-7_5> (paywalled redirect).
    **Neither the attribution nor the "25 % fewer multiplies" claim is verified.**
20. D. Tolmachev, *VkFFT — A Performant, Cross-Platform and Open-Source GPU FFT Library*,
    IEEE Access **11**, 2023, pp. 12039–12058. Mirrors refused (403 / unparseable). The
    "four-step FFT with no transpositions" description is from a search summary, not a
    fetched source.

**Own arithmetic (not from any source):** all of §5, the transpose-cost table in §3.1, the
conflict-set and page-count tables in §6 and §7, the batch/L3-spill thresholds, and all of
§9 and §10. Hardware parameters for this node come from `lscpu`, `lscpu -C`,
`/proc/meminfo` and `/proc/cpuinfo` on the machine hosting the repository.
