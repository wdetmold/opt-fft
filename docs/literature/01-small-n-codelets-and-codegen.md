# 01 — Straight-line codelets for small *n*, and the code generators that produce them

**Scope.** This section covers the literature and the real source code behind *hard-coded,
loop-free DFT kernels* ("codelets") and the compilers that emit them: FFTW's `genfft`
(Frigo, PLDI 1999), SPIRAL (Püschel et al., Proc. IEEE 2005), Blake's SFFT/FFTS, and
VkFFT's radix-kernel generator. It also collects the published operation counts for
small-*n* DFT modules (Winograd/PFA/WFTA/split-radix lineage) and gives a *measured*
register-pressure budget for x86-64 AVX2 (16 vector registers) and AVX-512 (32).

**The headline for this project.** All four target sizes (L = 6, 8, 17, 36) are far below
every published codelet/recursion crossover in the literature. A 1D transform of length
6, 8, 17 or 36 is *entirely* codelet territory: there is no recursion to plan, no cache
blocking inside the line transform, and no reason to call a "planner". The whole
engineering problem collapses to (a) getting the straight-line arithmetic minimal,
(b) keeping the working set inside 16 (AVX2) or 32 (AVX-512) vector registers, and
(c) feeding the codelet from memory in long sequential batched streams. Sections 7 and 8
give the concrete numbers.

Every citation below carries a URL that was fetched during the preparation of this
document. Fetch status for each is recorded in §10. Anything not backed by a fetched
source is confined to §9 and attributed to nobody.

---

## 1. FFTW's `genfft` — what it actually does

**Primary source.** Matteo Frigo, "A Fast Fourier Transform Compiler," *Proceedings of the
1999 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI)*,
Atlanta, Georgia, May 1999. <https://www.fftw.org/pldi99.pdf> (fetched; text extracted
locally). Received the *Most Influential PLDI Paper* award in 2009.

**Companion source.** Matteo Frigo and Steven G. Johnson, "The Design and Implementation of
FFTW3," *Proc. IEEE* **93**(2), pp. 216–231 (2005). <https://www.fftw.org/fftw-paper-ieee.pdf>
(fetched). §VI is the `genfft` section.

**Textbook restatement by the same authors.** S. G. Johnson and M. Frigo, "Implementing FFTs
in Practice," ch. 11 of C. S. Burrus (ed.), *Fast Fourier Transforms*, Connexions/Rice
University, collection structure revised 18 Nov 2012 (`col10550`).
<https://repository.rice.edu/server/api/core/bitstreams/01e9e0a5-fa6f-453d-a1b5-8209fa0a565c/content>
(fetched). This chapter is the most quotable of the three on register pressure.

### 1.1 The four phases (Frigo 1999, §1; Frigo & Johnson 2005, §VI)

`genfft` is written in Objective Caml. Its input is *the integer n*; its output is C.

1. **Creation.** Build a directed acyclic graph (DAG) of the transform "according to some
   well-known algorithm for the DFT. The generator contains many such algorithms and it
   applies the most appropriate." FFTW3 §VI names them: Cooley–Tukey, prime-factor
   (Good–Thomas), split-radix, and Rader. Each is expressed in a "straightforward
   math-like notation, using complex numbers, with no attempt at optimization."
2. **Simplifier.** Local rewriting rules per node: algebraic transformation, CSE, plus
   DFT-specific transformations (see §1.2). Written in monadic style; the monad both
   memoizes (so the writer can pretend the DAG is a tree) and performs CSE.
3. **Scheduler.** A topological sort of the DAG that, for n a power of 2, "provably
   minimizes the asymptotic number of register spills, no matter how many registers the
   target machine has." (See §1.3.)
4. **Unparser.** Emit C. `genfft` deliberately contains *no* loop unrolling — the whole
   output is already straight-line.

### 1.2 What the simplifier actually buys — with numbers

Three results from Frigo (1999) that are worth copying directly:

* **Make all floating-point constants positive.** Otherwise a C compiler stores both `c`
  and `−c` in the program text and loads both at runtime. Forcing positive constants
  halves the number of constant loads and "speeds up the generated codelets by 10-15% on
  most machines." It also canonicalises subexpressions, which helps CSE. *(§5, Frigo
  1999.)*
* **DAG transposition.** A DAG computing a linear function can be reversed (transposed);
  in some cases the transposed DAG exposes simplifications the original does not. The
  simplifier therefore does *three* passes: simplify `A` → `B`; simplify `Bᵀ` → `Cᵀ`;
  simplify `C` → `D`. "Although one might imagine iterating this process, three passes
  seem to be sufficient in all cases." Transposition **reduces multiplications but never
  additions.** Frigo's Fig. 7 (complex-to-complex rows):

  | n  | adds (untransposed) | muls (untransposed) | adds (transposed) | muls (transposed) |
  |----|---------------------|---------------------|-------------------|-------------------|
  | 5  | 32                  | 16                  | 32                | **12**            |
  | 10 | 84                  | 32                  | 84                | **24**            |
  | 13 | 176                 | 88                  | 176               | **68**            |
  | 15 | 156                 | 68                  | 156               | **56**            |

  Complex-to-real rows in the same figure: n=16 → 58/22 becomes 58/**18**; n=32 → 156/62
  becomes 156/**54**; n=64 → 394/166 becomes 394/**146**. (Sizes for which transposition
  has no effect are not listed in Frigo's table.)

  Why it works: for `y = a·s + b·t`, collapsing to `y = a(s + (b/a)t)` destroys two common
  subexpressions and can *increase* the count. Transposed, the same network computes
  `s = a·y`, `t = b·y`, and the corresponding rewrite provably cannot increase the count.
  "In a sense, transposition provides a simple and elegant way to detect which dag nodes
  have more than one parent."
* **It finds algorithms people missed.** For n = 13, Rader's algorithm in the form
  presented by Tolimieri et al. does **214 real additions and 76 real multiplications**;
  `genfft`'s output for the same algorithm has **176 real additions and 68 real
  multiplications**, "because `genfft` found certain simplifications that the authors did
  not notice." (The specific case Frigo names in his footnote: `t = a + b`, `u = a − b`,
  `v = t + u` simplifies to `v = 2a` when `t` and `u` are dead.) Frigo also notes a
  published algorithm with **188 additions and 40 multiplications** using a more involved
  method, and claims his generated program has the lowest known *addition* count for n=13.

  *Cross-check performed here:* FFTW 3.3.10 still ships exactly that count. In
  `dft/scalar/codelets/n1_13.c` the non-FMA variant's header comment reads
  "*176 FP additions, 68 FP multiplications*". Frigo's 1999 number is still in the tree.

### 1.3 The scheduler and the Hong–Kung bound — the one piece of real theory

This is the part of `genfft` most worth stealing, because it tells you *what order to emit
the arithmetic in* without knowing the register count.

* **Lower bound.** Frigo cites Hong & Kung's red-blue pebbling game (Theorem 2.1): executing
  the FFT graph of size n = 2ᵏ on a machine with R registers (R < n) requires at least
  **Ω(n log n / log R)** register spills. Aggarwal & Vitter generalise to blocked I/O and
  give a *matching* schedule — but their schedule depends on R, and "a schedule for a given
  value of R does not work well for other values."
* **The cache-oblivious trick.** "It is perhaps surprising that a schedule exists that
  matches the asymptotic lower bound for *all* values of R." Take Cooley–Tukey with
  n₁ = n₂ = √n at every level of recursion. Then the transfer count T(n) satisfies
  `T(n) = 2√n · T(√n) + O(n)` for n > cR, and `T(n) = O(R)` otherwise, with solution
  `T(n) = O(n log n / log R)` — matching the bound, independent of R.
* **How that becomes a DAG schedule.** Splitting size n into √n problems of size √n is the
  same as cutting the DAG with a vertical line into two halves of roughly equal size;
  execute every node in the first half before any node in the second; each half consists of
  Θ(√n) connected components, scheduled recursively the same way.
* **The heuristic for non-powers-of-2** ("burning the candle at both ends"): colour inputs
  red, outputs blue, everything else black. Alternate phases — in a red phase any node whose
  predecessors are all red becomes red; in a blue phase any node whose successors are all
  blue becomes blue — until no black nodes remain. Red nodes are the first half, blue the
  second. For n = 2ᵏ this cuts the regular butterfly DAG straight down the middle.
  Frigo is explicit that for general n "neither the lower-bound nor the upper-bound analyses
  hold" — it merely works.

Johnson & Frigo restate the motivation more bluntly in the Burrus chapter: for a size-64
codelet, "~2000 lines long, with hundreds of variables and over 1000 arithmetic operations
that can be executed in many orders," the key problem is "the efficient use of the CPU
registers, which essentially form a nearly ideal, fully associative cache. Normally, one
relies on the compiler for all code scheduling and register allocation, but the compiler
needs help with such long blocks of code (indeed, the general register-allocation problem
is NP-complete)."

### 1.4 Why straight-line beats loops for small n

* Frigo (1999), §3: "At least for the DFT problem, these long sequences of straight-line
  code seem to be necessary in order to take full advantage of large CPU register sets and
  the scheduling capabilities of C compilers." The size-64 codelet is "about 2400 lines of
  code, including 912 additions and 248 multiplications" and is "about twice as fast as
  Digital's DXML library" on Alpha.
* Johnson & Frigo (Burrus ch. 11) give the crisp argument for *bigger* rather than *more*
  codelets: "Is it better to implement a hard-coded FFT of size 64, for example, or an
  unrolled loop of four size-16 FFTs, both of which operate on the same amount of data? **The
  former should be more efficient because it performs more computations with the same amount
  of data, thanks to the log n factor in the FFT's n log n complexity.**" This is directly
  relevant to L = 36: prefer one 36-point module over a loop of 6-point ones *if* it fits in
  registers (it does not — see §7).
* Same chapter, on why the recursion must be coarsened at all: a textbook recursion down to
  n = 1 incurs ≈ 2n function calls, "so that every data point incurs a two-function-call
  overhead on average. Moreover, the compiler cannot fully exploit the large register sets
  and instruction-level parallelism of modern processors with an n = 1 function body."

### 1.5 Real, copyable knobs from the `genfft` source

The generator source ships in the FFTW tarball. On this machine:
`/home/lqcd/wdetmold/fft/ext/src/fftw-3.3.10/genfft/`. The "magic parameters" in
`magic.ml` are an unusually direct statement of where the algorithmic thresholds are:

```ocaml
let karatsuba_min = ref 15
let circular_min = ref 64
let rader_min = ref 13          (* "-rader-min <n> : Use Rader's algorithm for prime sizes >= <n>" *)
let rader_list = ref [5]
let alternate_convolution = ref 17
let number_of_variables = ref 4
let pipeline_latency = ref 0
```

And the flags FFTW itself uses to build its shipped codelets
(`support/Makefile.codelets`):

```make
FLAGS_COMMON     = -compact -variables 4
DFT_FLAGS_COMMON = $(FLAGS_COMMON) -pipeline-latency 4
```

`-compact` mangles variable names to shrink the source; `-variables 4` controls the
lexical-scope nesting of temporaries — which matters, because Frigo (1999) §7 reports that
Digital's C compiler for Alpha produced measurably better code when temporaries were
declared in *private nested scopes* encompassing their lifetime rather than all at top
scope. On the SPARC side, GCC's first instruction-scheduling pass *destroyed* `genfft`'s
schedule; disabling it (`-fno-schedule-insns`) made the compiled code "between 50% and 100%
faster and about half the size" on a 167 MHz UltraSPARC I, and "inspection of the assembly
code reveals that the difference consists entirely of register spills and reloads."

> **Implementer takeaway.** If you hand a modern compiler a 500-line straight-line codelet,
> check the assembly for spills, and check whether the compiler's pre-RA scheduler is
> undoing your ordering. This is a documented, reproducible failure mode, not folklore.

Relevant to **L = 17**: `rader_min = 13` and `alternate_convolution = 17` together mean that
for p = 17 `genfft` takes the Rader path *and* uses its "alternate" convolution routine, whose
source comment (`genfft/fft.ml`) reads: "*alternate routine for convolution. Seems to work
better for small sizes. I have no idea why.*" That routine splits the length-16 sequences
`a` and `b` into symmetric and antisymmetric halves (`ap/am`, `bp/bm`), takes four forward
length-16 DFTs and two inverse ones, and recombines. The point of the split is that the DFT
of a symmetric real sequence is real and of an antisymmetric one is imaginary, so each of
those six length-16 transforms is a *half-cost* transform.

### 1.6 Where FFTW stops using codelets and starts recursing

* "The standard FFTW distribution contains a set of **about 150 pre-generated codelets**
  that cover the most common uses." (Frigo & Johnson 2005, §I.)
* "A typical codelet in FFTW computes a DFT of a small, fixed size n (**usually, n ≤ 64**)."
  (Johnson & Frigo, Burrus ch. 11 §11.6.)
* "the recursion could stop when n = 32 is reached, at which point a highly optimized
  hard-coded FFT of that size would be executed"; "FFTW uses depth-first recursion with a
  bounded radix … but with much larger radices (**radix 32 is common**) and base cases
  (**size 32 or 64 is common**)." (ibid.)
* Frigo (1999) §7: generating C for n = 64 (**"the biggest used in FFTW"**) took ~75 s on a
  200 MHz Pentium Pro and <3 MB; the whole system (~55,000 lines, 120 files) regenerated in
  ~15 min. "The sizes of these transforms in the standard FFTW distribution include **all
  integers up to 16 and all powers of two up to 64**." The largest program ever generated
  was n = 101, at ~2 h CPU and ~10 MB (and required replacing linked-list assoc tables with
  hashing; the naive generator "had not produced an answer after three days").

**What FFTW 3.3.10 actually ships** (read from
`/home/lqcd/wdetmold/fft/ext/src/fftw-3.3.10/dft/*/codelets*/Makefile.am` and the
directory listings):

| kind | meaning (verbatim from `Makefile.am`) | sizes shipped |
|---|---|---|
| `n1_<n>` | "a hard-coded FFT of size `<n>` (base cases of FFT recursion)" | 2–16, 20, 25, 32, 64 |
| `t1_<r>` | "a 'twiddle' FFT of size `<r>`, implementing a radix-r DIT step" | 2–10, 12, 15, 16, 20, 25, 32, 64 |
| `t2_<r>` | twiddle FFT that "partially generates the trig. values on the fly (this is faster for large sizes)" | 4, 5, 8, 10, 16, 20, 25, 32, 64 |
| `q1_<r>` | "`<r>` twiddle FFTs of size `<r>` (DIF step), where the output is transposed. This is used for in-place transposes in sizes that are divisible by `<r>²`" | 2, 3, 4, 5, 6, 8 |
| SIMD `n1fv_<n>` | vectorised direct codelets | 2–16, 20, 25, 32, 64, **128** |
| SIMD `t1fv_<r>` | vectorised twiddle codelets | 2–10, 12, 15, 16, 20, 25, 32, 64 |

Two things follow immediately for this project:

1. **There is no n = 17 codelet and no n = 36 codelet in FFTW.** For L = 17 and L = 36 you
   are outside FFTW's shipped set; nobody has published a tuned kernel you can crib. (This is
   also why FFTW's own performance at these sizes is beatable.)
2. The `q1_<r>` comment is a direct, primary-source statement of the register/code-size
   ceiling for the "transpose fused into the codelet" trick: *"These codelets have size ~
   `<r>²`, so you should probably not use `<r>` bigger than **8** or so."* FFTW ships
   `q1_6` (a 6×6 = 36-point transposing DIF stage) and `q1_8` (8×8 = 64-point), i.e. exactly
   the L = 6 and L = 8 cases, and stops there.

---

## 2. Published operation counts for small-*n* modules

### 2.1 Provenance warning about Nussbaumer and Winograd

The op-count tables below are quoted from **Burrus (ed.), *Fast Fourier Transforms***
(Connexions `col10550`, PDF generated 18 Nov 2012), which I fetched and read in this
session. Burrus's ch. 7 ("Winograd's Short DFT Algorithms") is the standard secondary
presentation of the Winograd/Nussbaumer short-DFT modules and is written by one of the
people who built the PFA/WFTA programs. **Nussbaumer's book itself
(*Fast Fourier Transform and Convolution Algorithms*, Springer) was not obtained in this
session; do not attribute any specific number below to Nussbaumer's tables.** Where I quote
a number, the citation is to the Burrus PDF and to its table number.

### 2.2 Winograd-style short DFT modules — Burrus Table 7.1

Burrus's framing: "*In Table 7.1 an operation count of several short DFT algorithms is
presented. … Most are optimized in having either the theoretical minimum number of
multiplications or the minimum number of multiplications without requiring a very large
number of additions.*" Two multiplication columns are given: actual floating-point
multiplies, and the total including multiplies by unity (the latter is what you need for
programming the WFTA).

**Table 7.1 — Real multiplications and additions for a length-N DFT of *complex* data:**

| N | Mult | Non-one Mult | Total Adds |
|---:|---:|---:|---:|
| 2 | 0 | 4 | 4 |
| 3 | 4 | 6 | 12 |
| 4 | 0 | 8 | 16 |
| 5 | 10 | 12 | 34 |
| 7 | 16 | 18 | 72 |
| 8 | 4 | 16 | 52 |
| 9 | 20 | 22 | 84 |
| 11 | 40 | 42 | 168 |
| 13 | 40 | 42 | 188 |
| 16 | 20 | 36 | 148 |
| **17** | **70** | **72** | **314** |
| 19 | 76 | 78 | 372 |
| 25 | 132 | 134 | 420 |
| 32 | 68 | – | 388 |

Burrus adds, immediately after: "*Because of the structure of the short DFTs, the number of
real multiplications required for the DFT of real data is exactly half that required for
complex data. The number of real additions required is slightly less than half … because
(N−1) of the additions needed when N is prime add a real to an imaginary, and that is not
actually performed. When N = 2ᵐ, there are (N−2) of these pseudo additions.*"

**Table 7.2 — a different multiply/add trade-off for primes** (Burrus ch. 7, from the
automatic prime-length FFT program design work described in the same chapter; the chapter
notes "*We have designed prime length FFTs up to length 53 that are as good as the previous
designs that only went up to 19*" and "*the operation counts depend on the factorability of
P − 1*"):

| N | Mult | Adds |
|---:|---:|---:|
| 7 | 16 | 72 |
| 11 | 40 | 168 |
| 13 | 40 | 188 |
| **17** | **82** | **274** |
| 19 | 88 | 360 |
| 23 | 174 | 672 |
| 29 | 190 | 766 |
| 31 | 160 | 984 |
| 37 | 220 | 920 |
| 41 | 282 | 1140 |
| 43 | 304 | 1416 |
| 47 | 640 | 2088 |
| 53 | 556 | 2038 |

> **For L = 17 this is the whole design space in two rows.** Table 7.1's module: 70 real
> multiplies + 314 real adds = 384 flops. Table 7.2's module: 82 + 274 = 356 flops — 8%
> fewer *total* flops but 17% more multiplies. On Haswell/Skylake with two FMA ports, the
> Table 7.2 balance (mult:add ≈ 1:3.3) is the more attractive of the two, and an
> FMA-targeted derivation should beat both. Note that 17 is the *only* one of these primes
> where P − 1 = 16 is a pure power of two, which is why 17 is unusually cheap relative to
> its neighbours 19 and 23 (Burrus: primes of the form 1 + 2P₁ "*mak[e] the design of
> efficient FFTs for these lengths more difficult*").

### 2.3 Cooley–Tukey single-radix and split-radix — Burrus Table 9.1

Burrus's Table 9.1 gives real multiplications `Mᵢ` and additions `Aᵢ` for complex
single-radix FFTs, where `i` is the number of separately written butterflies (i.e. how many
special-cased twiddle butterflies are unrolled out). Blocks are radix-2, radix-4, radix-8,
radix-16, split-radix. Extracted rows relevant here:

| algorithm | N | M₁ | M₂ | M₃ | M₅ | A₁ | A₂ | A₃ | A₅ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| radix-2 | 8 | 48 | 20 | 8 | 4 | 72 | 58 | 52 | 52 |
| radix-2 | 16 | 128 | 68 | 40 | 28 | 192 | 162 | 148 | 148 |
| radix-4 | 16 | 96 | 36 | 28 | 24 | 176 | 146 | 144 | 144 |
| radix-4 | 64 | 576 | 324 | 284 | 264 | 1056 | 930 | 920 | 920 |
| radix-8 | 8 | 32 | 4 | 4 | 4 | 66 | 52 | 52 | 52 |
| radix-8 | 64 | 512 | 260 | 252 | 248 | 1056 | 930 | 928 | 928 |
| radix-16 | 16 | 80 | 20 | 20 | 20 | 178 | 148 | 148 | 148 |
| split-radix | 8 | 24 | 8 | 4 | 4 | 60 | 52 | 52 | 52 |
| split-radix | 16 | 72 | 32 | 28 | 24 | 164 | 144 | 144 | 144 |
| split-radix | 32 | 184 | 104 | 92 | 84 | 412 | 372 | 372 | 372 |
| split-radix | 64 | 456 | 288 | 268 | 248 | 996 | 912 | 912 | 912 |

Two observations, both load-bearing for **L = 8**:

* **Radix-8 with its butterflies written out, split-radix, and the Winograd length-8 module
  all land on exactly the same count: 4 real multiplies + 52 real adds.** (Table 9.1 radix-8
  row N=8 M₂..M₅ = 4, A = 52; Table 9.1 split-radix N=8 M₃/M₅ = 4, A = 52; Table 7.1 N=8
  row = 4 mult / 52 adds.) There is no cleverness left to find at n = 8 — 56 flops is the
  number, and the only question is scheduling and register allocation.
* The M₁ → M₅ columns show how much you lose by *not* special-casing the trivial twiddles:
  radix-2 at N = 8 goes from 48 multiplies (one generic butterfly) to 4 (fully special-cased).
  A 12× swing purely from writing out the ω = 1, ±i, ±(1±i)/√2 cases. This is the single
  biggest lever in a small-n codelet and is exactly what `genfft`'s simplifier automates.

**Asymptotics, for orientation** (Johnson & Frigo, Burrus ch. 11): the 1965 radix-2
Cooley–Tukey algorithm costs ~5 n log₂ n and "*the currently lowest-known arithmetic count
(~ (34/9) n log₂ n) remains only about 25%*" better. The 34/9 constant is from Johnson &
Frigo, "A Modified Split-Radix FFT With Fewer Arithmetic Operations," *IEEE Trans. Signal
Processing* **55**(1), pp. 111–119, January 2007
(<https://math.mit.edu/~stevenj/papers/JohnsonFr07.pdf>, fetched; also
<https://www.fftw.org/newsplit.pdf>, fetched). That paper states the improvement over
Yavne's split-radix as "asymptotically about 6% fewer operations", elsewhere quantified as
"a further ~5.6% (1/18)", and notes Yavne's split radix was itself "an improvement by 20%
over the classic radix-2 algorithm presented by Cooley and Tukey". **Savings start at N = 64**
and are "purely in the number of real multiplications" under the 4-mult/2-add complex
multiply convention. The same paper records the known realisable Θ(n) lower bound on
*irrational real multiplications* following Winograd, which "match[es] split radix as well as
our algorithm up to [small n] but is achieved only at the price of many more additions and
thus has limited utility on CPUs with hardware multipliers."

*(Consistency check performed here: Yavne's closed form 4N log₂N − 6N + 8 reproduces Burrus
Table 9.1's split-radix totals exactly — N=8: 96−48+8 = 56 = 4+52; N=16: 256−96+8 = 168 =
24+144; N=64: 1536−384+8 = 1160 = 248+912.)*

### 2.4 PFA / WFTA composites — Burrus Table 10.1, and L = 36

Burrus ch. 10 gives the PFA addition/multiplication recursion for a four-factor
N = N₁N₂N₃N₄ (his eq. 10.18):

```
T(N) = N₁N₂N₃·T(N₄) + N₂N₃N₄·T(N₁) + N₃N₄N₁·T(N₂) + N₄N₁N₂·T(N₃)
```

and states that "*the count of multiplies and adds in Table 10.1 are calculated from [this]
with the counts of the factors taken from … Table 7.1*". The module set used is
"length 2, 3, 4, 5, 7, 8, 9 and 16", and "*a maximum of four relatively prime lengths can be
used from this group giving 59 different lengths over the range from 2 to 5040. The radix-2
or split-radix FFT allows 12 different lengths over the same range.*"

**Table 10.1 rows bracketing our sizes** (real ops, complex data):

| N | PFA Mults | PFA Adds | WFTA Mults | WFTA RMults | WFTA Adds |
|---:|---:|---:|---:|---:|---:|
| 12 | 16 | 96 | 24 | 16 | 96 |
| 15 | 50 | 162 | 36 | 34 | 162 |
| 18 | 40 | 204 | 44 | 40 | 208 |
| 20 | 40 | 216 | 48 | 40 | 216 |
| 24 | 44 | 252 | 48 | 36 | 252 |
| 30 | 100 | 384 | 72 | 68 | 384 |
| **36** | **80** | **480** | **88** | **80** | **488** |
| 40 | 100 | 532 | 96 | 84 | 532 |
| 48 | 124 | 636 | 108 | 92 | 660 |
| 60 | 200 | 888 | 144 | 136 | 888 |
| 72 | 196 | 1140 | 176 | 164 | 1156 |

Burrus's evaluation of the two (ch. 10 §10.4): "*compared to the PFA or any of the
Cooley-Tukey FFTs, the WFTA has significantly fewer multiplications. For the shorter
lengths, the WFTA and the PFA have approximately the same number of additions; however for
longer lengths, the PFA has fewer and the Cooley-Tukey FFTs always have the fewest. If the
total arithmetic, the number of multiplications plus the number of additions, is compared,
the split-radix FFT, PFA and WFTA all have about the same count.*" And, decisively for us:
"*The size of the Cooley-Tukey program is the smallest, the PFA next and the WFTA largest.
The PFA requires the smallest number of stored constants … For a DFT of approximately 1000,
the PFA stores 28 constants, the FFT 2048 and the WFTA 3564. Both the FFT and PFA can be
calculated in-place and the WFTA cannot. The PFA can be calculated in-order without an
unscrambler.*"

The same section states the register argument in one sentence: "*The shorter modules in the
PFA and WFTA and the butterflies in the radix 2 and 4 FFTs are more efficient than the
longer ones because intermediate calculations can be kept in cpu registers rather general
memory. However, the shorter modules and radices require more passes through the data for a
given approximate length.*"

### 2.5 What FFTW 3.3.10's generated codelets actually cost

Harvested directly from the header comments of the shipped codelets in
`/home/lqcd/wdetmold/fft/ext/src/fftw-3.3.10/dft/scalar/codelets/`. Each codelet ships in
two variants selected by `ARCH_PREFERS_FMA`; both are reported. "Stack vars" is `genfft`'s
own count of declared C locals; "peak live" is measured (see §7.2).

| n | non-FMA adds | non-FMA muls | FMA adds | FMA muls | FMA fmas | stack vars | consts | peak live (real) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 4 | 0 | 4 | 0 | 0 | 5 | 0 | 2 |
| 3 | 12 | 4 | 6 | 0 | 6 | 15 | 2 | 8 |
| 4 | 16 | 0 | 16 | 0 | 0 | 13 | 0 | 10 |
| 5 | 32 | 12 | 14 | 0 | 18 | 21 | 4 | 13 |
| **6** | **36** | **8** | **24** | **0** | **12** | 23 | 2 | **17** |
| 7 | 60 | 36 | 18 | 0 | 42 | 41 | 6 | 16 / 27 (FMA) |
| **8** | **52** | **4** | **44** | **0** | **8** | 28 | 1 | **19** |
| 9 | 80 | 40 | 24 | 0 | 56 | 41 | 8–10 | 21 |
| 12 | 96 | 16 | 72 | 0 | 24 | 43 | 2 | 26 |
| 13 | 176 | 68 | 62 | 0 | 114 | 76 | 20–25 | 34 |
| 16 | 144 | 24 | 104 | 0 | 40 | 50 | 3 | 36 |
| 32 | 372 | 84 | 236 | 0 | 136 | 100 | 7 | 74 |
| 64 | 912 | 248 | 520 | 0 | 392 | 172 | 15 | 138 |

**Cross-validation against the published tables** — this is the most reassuring result in
this section:

* n = 3: 12 adds / 4 muls — **exactly** Burrus Table 7.1.
* n = 4: 16 / 0 — **exactly** Table 7.1.
* n = 6: 36 / 8 — **exactly** the PFA recursion applied to Table 7.1: adds =
  3·T_add(2) + 2·T_add(3) = 3·4 + 2·12 = 36; muls = 3·0 + 2·4 = 8. `genfft` independently
  found the Good–Thomas-optimal count for 6 = 2·3.
* n = 8: 52 / 4 — **exactly** Table 7.1 *and* Table 9.1 radix-8 *and* Table 9.1 split-radix.
* n = 12: 96 / 16 — **exactly** Burrus Table 10.1's PFA row for N = 12.
* n = 16, 32, 64: 144/24, 372/84, 912/248 — **exactly** Table 9.1's split-radix M₅/A₅ column.
* n = 13: 176 / 68 — exactly Frigo's 1999 result (§1.2).
* n = 5, 7, 9: `genfft` deliberately picks a *different* trade-off than Table 7.1 —
  fewer adds, more multiplies (n=5: 32/12 vs 34/10; n=7: 60/36 vs 72/16; n=9: 80/40 vs
  84/20). Total flops: 44 vs 44, 96 vs 88, 120 vs 104. Winograd wins on total flops at
  n = 7 and 9, `genfft` wins on the mult:add balance. On a machine with two FMA units and
  no multiply penalty, `genfft`'s balance is the better one, which is presumably why it
  ships that way.

**Twiddle-codelet counts** (a radix-r DIT stage: r-point DFT plus the twiddle multiplies),
same directory, from `t1_*.c`:

(FMA variants; the header comment gives an abstract add/mul count and then the actual
instruction mix, which is what the last three columns report.)

| r | abstract adds | abstract muls | actual adds | actual muls | actual FMAs | stack vars |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 22 | 12 | 16 | 6 | 6 | 15 |
| 6 | 46 | 32 | 24 | 10 | 22 | 31 |
| 8 | 66 | 36 | 44 | 14 | 22 | 34 |
| 9 | 96 | 88 | 24 | 16 | 72 | 55 |

**The `q1_r` transposing codelets** — these are the interesting ones for L = 6 and L = 8,
because they fuse an r-point radix stage, the twiddles, *and* an r×r transpose:

| codelet | what it computes | adds | muls | fmas | stack vars | memory accesses | peak live (real) |
|---|---|---:|---:|---:|---:|---:|---:|
| `q1_6` | 6 twiddled size-6 DFTs, output transposed (36 complex in/out) | 276 | 168 (non-FMA) / 60+132 fma | | 109 | 144 | 76 (97 FMA) |
| `q1_8` | 8 twiddled size-8 DFTs, output transposed (64 complex in/out) | 528 | 256 / 112+176 fma | | 152 | 256 | 132 (142 FMA) |

*(Internal consistency check: `q1_6`'s 276 adds / 168 muls = 6 no-twiddle DFT-6 kernels
(6 × 36 = 216 adds, 6 × 8 = 48 muls) plus 30 non-trivial complex twiddle multiplies at
4 mul + 2 add each (120 muls, 60 adds). 216 + 60 = 276 ✓, 48 + 120 = 168 ✓.)*

---

## 3. SPIRAL — formula rewriting plus search

**Primary source.** M. Püschel, J. M. F. Moura, J. R. Johnson, D. Padua, M. M. Veloso,
B. W. Singer, J. Xiong, F. Franchetti, A. Gačić, Y. Voronenko, K. Chen, R. W. Johnson,
N. Rizzolo, "SPIRAL: Code Generation for DSP Transforms," *Proc. IEEE* **93**(2),
pp. 232–273, February 2005.
<https://users.ece.cmu.edu/~moura/papers/ieeeproceedings-pueschelmouraetal-feb05-ieeexplore.pdf>
(fetched; text extracted locally).

### 3.1 The model

Transforms are written as **SPL formulas** (structured matrix factorisations). *Breakdown
rules* rewrite a transform into smaller ones; recursive application yields a **ruletree**.
"*When a formula is recursively generated, these choices lead to a combinatorial explosion
and, in most cases, to an exponentially growing number of formulas for a given transform.
The different formulas for one transform all have similar arithmetic cost (number of
additions and multiplications) equal or close to the best known … but differ in dataflow,
which in turn leads to a usually large spread in runtime. Finding the best formula is the
challenge.*"

### 3.2 The unrolling threshold — SPIRAL's version of "how big a codelet"

"*In the current version of SPIRAL, the most important implementation choice considered is
the degree of unrolling, which can be controlled either globally or locally. The global
unrolling strategy is determined by an integer threshold that specifies the smallest size of
(the matrix corresponding to) a subformula to be translated into loop code. This threshold
may be overridden by local tags… Experiments have shown that a global setting is sufficient
in most cases.*" And, concretely: "*the default global unrolling threshold **16***." Below
16, the subformula becomes fully unrolled straight-line code; at or above, it becomes a loop.

"*Loops marked for unrolling are fully unrolled; currently, the SPL compiler does not support
partial unrolling. A reasonably large degree of unrolling is usually very beneficial, as it
creates many opportunities for optimizations*" — specifically, inlining of the constant
tables (so the twiddles become literals, and multiplies by 0/±1 vanish) and inlining of
transcendental-function "intrinsics" whose arguments are all statically known.

> Note the direction: SPIRAL's *default* boundary between "unroll into a codelet" and "emit a
> loop" is at size **16**, and FFTW's typical codelet ceiling is n ≤ 64. Both bracket all four
> of our target sizes.

### 3.3 Why search, with numbers

* For a single small transform, DCT-2 of size 32, "*SPIRAL reports **1 639 236 012** different
  formulas*". Over a random sample of 10 000: "*The spread of runtimes is approximately a
  factor of two, and the spread of the number of instructions is about 1.5, whereas the spread
  in arithmetic cost is less than 10%.*" And: "*the formulas with lowest arithmetic cost yield
  both slowest and fastest runtimes, which implies that arithmetic cost is not a predictor of
  runtime in this case.*" Converting to FMA code "*reduces the operations count … but increases
  the spread to about 25%. This means that different formulas are differently well suited for
  FMA architectures.*"
* For a large power-of-two DFT, 20 000 random Cooley–Tukey formulas: "*The spread of runtimes
  in both cases [scalar and SSE] is about a factor of five, with most formulas within a factor
  of three. The best 30% formulas are scarce.*"
* Dynamic programming as search: for a size-2ᵏ DFT with only the Cooley–Tukey rule, the number
  of formulas grows like the number of binary trees, "*whereas DP visits only*" O(k) — with the
  caveat that DP assumes "*the best code for a transform is independent of the context in which
  it is called. This assumption holds for the arithmetic cost … but not for the runtime.*"

### 3.4 SPIRAL's measured DFT performance, and what it says about small sizes

Platform `p4-3.0-win` (Pentium 4, 3.0 GHz; scalar peak 3 Gflop/s, vector peak 12 Gflop/s
single / 6 Gflop/s double), versus MKL 6.1, IPP 4.0 and FFTW 3.0.1:

* GNU/FFTPACK-derived scalar code is the slowest line; the gap from it to the best is "*an
  order of magnitude (e.g. a factor of ten … between the GNU library and the IPP/FFTW/SPIRAL
  code)*".
* FFTW 3.0.1 scalar and SPIRAL scalar C are "*about equal in performance*".
* Compiler auto-vectorisation (with SPIRAL's search picking formulas the compiler handles
  well) gives "*about 50%*". Hand/generator-directed vector code gives "*about a factor of two
  faster [than that], or a factor of three over the best scalar code*". Also: "*we note that
  FFTW cannot be compiler vectorized due to its complex infrastructure.*"
* **"For small sizes, within L1 cache, SPIRAL code is best by a margin, most likely due to the
  combination of algorithm search, code level optimizations, and the simplest code
  structure."** Outside L1 but inside L2, the Intel libraries win; for large sizes FFTW wins.
* Working-set rule of thumb worth keeping: "*for single precision, approximately **32 B per
  complex vector entry** are needed (input vector, output vector, constants and spill space)
  while for double precision **64 B** are needed.*"
* Code-generation cost: "*the generation of a scalar DFT library for two-powers up to 2ᵏ is
  done in 20–30 min on a Pentium 4, while the corresponding vector code generation takes on
  the order of hours. Problem sizes around 64 are optimized within a few minutes.*"

### 3.5 SPIRAL's own characterisation of FFTW's plan space

Useful because it tells you what FFTW's planner *cannot* consider: "*for a one-dimensional
DFT of composite size and in SPIRAL lingo, these recursion strategies are all the right-most
ruletrees based on the Cooley–Tukey breakdown rule, where the left leaf is a codelet.
Restricting the DFT computation to this restricted class of algorithms is a decision based
on the experience of the FFTW developers.*"

### 3.6 The FFTW-vs-SPIRAL result that matters for scheduling

Frigo & Johnson (2005) §VI, and again in Burrus ch. 11: "*As a practical matter, one
consequence of this scheduler is that FFTW's machine-independent codelets are **no slower
than machine-specific codelets generated by SPIRAL** [43, Figure 3].*"

> **Implementer takeaway.** The √n-recursive ("cut the DAG in half") schedule of §1.3 is
> reported by FFTW's authors to be as good as an *automated per-machine search* over codelet
> schedules. If you emit the arithmetic in that order, you should not need a search phase for
> L = 6, 8, 17, 36.

---

## 4. FFTS / SFFT — Blake: don't use big codelets, use big *streams*

**Primary source (fetched).** Anthony Martin Blake, *Computing the fast Fourier transform on
SIMD microprocessors*, PhD thesis, University of Waikato, Hamilton, New Zealand, 2012
(supervisors Witten, Cree, Perrone). Handle <https://hdl.handle.net/10289/6417>; PDF fetched
from <https://researchcommons.waikato.ac.nz/bitstreams/8209dc72-3ba2-4ba0-a79e-e4d77bad0736/download>.
Library source: <https://github.com/anthonix/sfft> (cited in the thesis).

The journal version — A. M. Blake, I. H. Witten, M. J. Cree, "The Fastest Fourier Transform
in the South," *IEEE Trans. Signal Processing* **61**(19), pp. 4707–4716, October 2013 —
`[UNVERIFIED — could not fetch]` (publisher and mirror both returned 403). Everything below
comes from the thesis, which I did read.

### 4.1 The size ceiling on fully hard-coded transforms

"*The fully hard-coded FFTs described in this section are generally only practical for
smaller sizes of transforms, typically where **N ⩽ 128**, however these techniques are
expanded in later sections to scale the performance to larger sizes.*" And from the
measurements (Fig. 7, MacBook Air 4,2, sizes 4–1024, versus FFTW 3.3 in estimate and patient
modes): "*A variety of vector lengths are represented, and the hard-coded FFTs have good
performance while **N/VL ⩽ 128**. After this point, performance drops off and other
techniques should be used.*"

### 4.2 Why the leaves, not the codelet size, are the bottleneck

Blake's diagnosis is the opposite of FFTW's: the problem with a deep recursion is not
arithmetic but the *memory access pattern at the leaves*. On a size-64 split-radix FFT:

"*the leaves of the computation are rather inefficient, because there are large numbers of
straight line blocks of code performing scalar memory accesses, and no loops of more than a
few iterations (i.e. the leaves of the computation are not taking advantage of the machine's
SIMD capability). … Because the addresses of memory operations at the leaves are a function
of variables passed on the stack, it is very difficult for a hardware prefetch unit to keep
these leaves supplied with data, and thus memory latency becomes an issue. In later chapters,
it is shown that **increasing the size of the base cases at the leaves improves
performance**.*"

He also reports that a vectorised *radix-2* FFT beat split-radix up to size 4096 for exactly
this reason: split-radix "*spends more time computing the leaves … so despite the split-radix
algorithms being more efficient in the inner loops of SIMD computation, the performance has
been held back by higher proportion of very small straight line blocks of code (corresponding
to sub-transforms smaller than size 4) performing scalar memory accesses at the leaves*."

### 4.3 The rest of SFFT's design, in brief

* Base cases in the DAG elaboration are **size 2 and size 4** only, with a special case that
  handles two size-2 leaves in parallel "*ensur[ing] that larger transforms are composed of
  nodes that are homogeneous in size — this is of little utility when emitting VL = 1 code,
  but it is exploited [when] the topological ordering of nodes is vectorized*".
* Conjugate-pair split radix, chosen because it "*requires only half the number of twiddle
  factor loads*". Measured: "*For smaller sizes of transform, the ordinary split-radix
  algorithm is faster, but above a certain size (**4096** in this case), the conjugate-pair
  algorithm is faster.*"
* Per-size sequentially-accessed twiddle LUTs instead of one strided LUT: "*the speed improves
  markedly, **by over 30%** in many cases*", even though it duplicates data and grows the
  memory footprint. Blake's conclusion: "*accessing data in sequential streams provides big
  performance gains, even in the somewhat counter-intuitive case where data is duplicated and
  more memory is required.*"
* Split (deinterleaved) vs interleaved complex format: split format lets the real/imaginary
  swap happen at the instruction level rather than by shuffling, and Blake's Listing 2
  "*computes complex multiplication for vectors twice as long while using one less SSE
  instruction*". But: "*The only disadvantage to the split format approach is that **twice as
  many registers are needed** to compute a given operation — this might preclude the use of a
  larger radix or force register paging for some kernels of computation.*"
* Benchmarked on **sixteen x86 machines and two ARM NEON machines** against FFTW, SPIRAL,
  Intel IPP and Apple Accelerate; SFFT "*was found to be as fast as, or faster than FFTW*"
  "*but without extensive machine calibration*". A regression model predicting performance
  from machine + compiler characteristics achieved **74.8% precision** under 10-fold CV.
* Blake explicitly challenges FFTW's premise: he notes that FFTW's claim that "*the compiler
  needs help with such long blocks of code*" rests on a 1999 comparison using *Sun WorkShop
  Compilers 4.2 (30 Oct 1996)*, and that "*there is no mention of re-testing the aforementioned
  hypothesis with more advanced compilers*."

> **Implementer takeaway for a *batched* workload.** Blake's finding is the most directly
> applicable one in this whole section. Our transforms are tiny (L ≤ 36) and enormously
> batched — which is precisely the regime where Blake says the win comes from turning the leaf
> memory traffic into long unit-stride SIMD streams, not from shaving flops. Batch-major
> layout (vectorise across volumes / spin-colour components, keep the codelet's loads and
> stores unit-stride and aligned) should dominate every arithmetic optimisation in §2.

---

## 5. VkFFT — a second, independent code generator with an explicit register budget

Source read locally: `/home/lqcd/wdetmold/fft/ext/src/VkFFT/`. Public repository:
<https://github.com/DTolm/VkFFT>. Relevant files:
`vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h`,
`.../vkFFT_RaderKernels.h`, `.../vkFFT_RegisterBoost.h`, and
`vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h`.

Although VkFFT targets GPUs, it is a useful second data point because (a) it is a *string-
emitting code generator* like `genfft`, (b) it ships a hand-written straight-line kernel per
small radix, and (c) it reasons about the register budget explicitly, in source, with numbers.

* **Radix kernel set** (`switch` cases in `vkFFT_RadixKernels.h`): **2, 3, 4, 5, 6, 7, 8, 9,
  10, 11, 12, 13, 14, 15, 16, 32**. Note that 6, 8, 9 and 12 are all present as
  *first-class* straight-line kernels — you do not need to build 6 out of 2 and 3 at runtime.
  Primes beyond 13 go through a separate Rader path (`vkFFT_RaderKernels.h`).
* **Register accounting.** `VkFFTGetRegistersPerThread()` fills
  `registers_per_thread_per_radix[r]` — the number of *complex* registers a radix-r stage
  needs. The values that appear in the source, per radix: radix 2 → {4, 6, 8, 10, 12, 14, 16};
  radix 3 → {3, 6, 9, 12, 15}; radix 5 → {5, 10, 15}; radix 7 → {7, 14}; radix 11 → {11};
  radix 13 → {13}.
* **The threshold, verbatim:**
  ```c
  if ((registers_per_thread[0] > 16) || (registers_per_thread[0] >= 2 * min_registers_per_thread[0]))
      isGoodSequence[0] = 0;
  else isGoodSequence[0] = 1;
  ```
  i.e. VkFFT declares a factorisation *not a good sequence* when the largest radix needs more
  than **16** registers, or when the largest radix needs ≥ 2× the smallest (an imbalance
  penalty — mixing a very fat radix with a very thin one wastes registers in the thin stages).

> A production FFT generator's own source says: **16 vector registers is the working ceiling
> for a single radix stage, and stages should be balanced in register demand.** That is
> exactly the AVX2 constraint. See §7.

---

## 6. One more generator worth knowing about

The heFFTe "stock" backend already present in this tree
(`/home/lqcd/wdetmold/fft/ext/install/avx512/include/stock_fft/heffte_stock_algos.h`) is a
much simpler C++-template FFT with radix-2, radix-4 and generic radix-N₁ Cooley–Tukey
helpers. It is *not* a codelet generator and is not competitive with FFTW at these sizes;
it is listed only so nobody mistakes it for one.

---

## 7. Register pressure on x86-64: how big can an unrolled codelet usefully be?

### 7.1 The architectural facts

* **AVX2 (VEX encoding): 16 named vector registers**, `ymm0`–`ymm15`, 256 bits each = 4 double
  or 2 complex double. **AVX-512 (EVEX encoding): 32**, `zmm0`–`zmm31`, 512 bits = 8 double or
  4 complex double. Verified locally in this session by assembling
  `vaddpd %zmm31, %zmm30, %zmm29` (accepted with `-mavx512f`) and `vaddpd %ymm15, %ymm14, %ymm13`.
  Blake's thesis gives the history: SSE introduced eight 128-bit XMM registers; "*The AMD 64
  architecture doubled the number of XMM registers to 16, and Intel followed by implementing 16
  XMM registers in the Intel 64 architecture*"; AVX widened them to 256-bit YMM.
* Note the machine mix in this project: the login node here is an **Intel Xeon E5-2680 v3
  (Haswell)** — `avx2` and `fma`, **no AVX-512** (`grep -o 'avx512[a-z]*' /proc/cpuinfo`
  returns nothing). AVX-512 is only on `axxxl`/`a100l`/`a100r` per the project README. So the
  **16-register** budget is the one to design against, with an AVX-512 variant as an upside.
* Physical vs architectural registers: Johnson & Frigo note in a footnote (Burrus ch. 11) that
  "*on current x86 processors, the user-visible instruction set (with a small number of
  floating-point registers) is internally translated at runtime to RISC-like µ-ops with a much
  larger number of physical rename registers that are allocated automatically*". Renaming
  removes *false* dependences; it does **not** give you more than 16 (or 32) simultaneously
  addressable values. Exceeding the architectural count means real spill stores/loads.

### 7.2 Measured peak liveness of FFTW's own codelets

To turn this into a budget I measured the **peak number of simultaneously live values** under
`genfft`'s emitted schedule, directly from the shipped codelet source. Method: single-static-
assignment last-use analysis over the statement list (each `Tn = …` is a definition, each
appearance in an RHS or in an `ST(...)` store is a use); a value is live from its definition
through its last use. Script preserved in this repo at
`/home/lqcd/wdetmold/fft/docs/literature/tools/fftw_codelet_liveness.py` — run as
`python3 fftw_codelet_liveness.py <codelet.c> …`. Validated by hand against `n1fv_4.c`
(measured 6, hand-counted 6).

**FFTW SIMD codelets (`dft/simd/common/n1fv_*.c`). These count *vector* registers, and the
count is independent of vector width because the same source compiles for SSE2/AVX/AVX-512:**

| n | peak live vectors (non-FMA) | peak live vectors (FMA) | distinct constants | verdict on AVX2 (16) | verdict on AVX-512 (32) |
|---:|---:|---:|---:|---|---|
| 2 | 2 | 2 | 0 | fits trivially | fits |
| 3 | 5 | 5 | 2 | fits | fits |
| 4 | 6 | 6 | 0 | fits | fits |
| 5 | 9 | 9 | 4 | fits | fits |
| **6** | **9** | **9** | **2** | **fits (11 with consts)** | fits |
| 7 | 9 | 14 | 6 | fits non-FMA; tight with FMA | fits |
| **8** | **12** | **12** | **1** | **fits (13 with consts)** | fits |
| 9 | 15 | 17 | 14–19 | **over** once constants are resident | fits |
| 12 | 16 | 16 | 2 | exactly at the ceiling | fits |
| 16 | 19 | 19 | 3 | **over** | fits |
| 32 | 40 | 40 | 7 | far over | **over** |
| 64 | 72 | 74 | 15 | far over | far over |

Twiddle codelets (`t1fv_*`, an r-point DFT plus its twiddle multiplies): r = 4 → 6, r = 6 →
8, r = 8 → 10, r = 9 → 15 (18 with FMA).

Scalar codelets (`dft/scalar/codelets/n1_*.c`) come out at almost exactly 2× the SIMD
numbers, as expected since they split real and imaginary into separate values: n = 6 → 17,
n = 8 → 19, n = 9 → 21, n = 16 → 36, n = 32 → 74, n = 64 → 138. Frigo's remark that the
size-64 codelet has "hundreds of variables" is borne out: 912 defined values, 138 of them
simultaneously live, against 16 architectural registers.

### 7.3 The budget, stated as a rule

Reading §7.1, §7.2, VkFFT's `> 16` test (§5), SPIRAL's default unrolling threshold of 16
(§3.2), and FFTW's `q1_r` comment "*you should probably not use `<r>` bigger than 8 or so*"
(§1.6) together, they all agree:

* **A fully unrolled complex codelet of size n needs roughly 1.2 n to 1.7 n vector
  registers** — measured 9 for n=6 (1.5 n), 12 for n=8 (1.5 n), 15 for n=9 (1.7 n), 16 for
  n=12 (1.3 n), 19 for n=16 (1.2 n), 40 for n=32 (1.25 n) — **plus one per resident twiddle
  constant.** The ratio falls with n because larger power-of-two kernels have more internal
  structure to exploit; the small odd sizes are the register-hungry ones per point.
* **AVX2 (16 regs): unrolled codelets are register-resident up to about n = 12 (which lands
  exactly on 16). n = 16 already needs 19 and spills.**
* **AVX-512 (32 regs): register-resident up to about n = 24 (interpolating 19 at n=16 and 40
  at n=32). n = 32 spills.**
* Above that, don't fight it: split into two register-resident stages and let the intermediate
  live in L1, which is what every library in this section does.
* Also budget for the *other* live things: 4–6 general-purpose registers for the input/output
  pointers, strides and the batch loop counter (FFTW's codelets take `ri, ii, ro, io, is, os,
  v, ivs, ovs`), and, if you keep twiddles in registers rather than as memory operands, one
  vector register per distinct constant. Memory operands are usually the right answer for
  constants on x86-64 — a broadcast-from-memory FMA operand costs no register.

---

## 8. Mapping onto L = 6, 8, 17, 36

Everything here follows from §2 and §7; the size-specific arithmetic marked *(derived)* is my
own and is flagged again in §9.

### L = 6 = 2·3 (216 points per volume)

* **Straight-line, one codelet, no question.** 6-point complex DFT: **36 real adds + 8 real
  multiplies** (Burrus Table 7.1 via the PFA recursion; FFTW's `n1_6` hits exactly this).
  With FMA: FFTW's FMA variant is 24 adds + 12 FMAs. Peak liveness **9 vector registers** + 2
  constants — comfortably inside AVX2's 16.
* 2 and 3 are coprime, so **Good–Thomas applies and there are no twiddle factors at all** —
  and `genfft` independently rediscovers the PFA count, which is a strong signal that PFA is
  the right derivation to hand-write.
* Because 6 ≤ 8, FFTW's own `q1_6` codelet exists: a 6×6 transposing DIF stage, 36 complex in
  and out, 276 adds + 168 muls, 76 live reals (38 vector regs, so *not* register-resident).
  Worth reading as a template for fusing the transpose into the kernel, but the 3D transform
  should almost certainly fuse the transpose into the batch loop instead of into a 36-point
  block.
* *(derived)* Full 3D: 3·L² = 108 line transforms of length 6 per volume, 108·44 = 4752 flops
  per L³ volume, i.e. ~22 flops per point. 216 complex doubles = 3456 B per volume — an entire
  volume fits in L1 with room to spare, so the only memory question is how many volumes to
  keep in flight.

### L = 8 = 2³ (512 points per volume)

* **The arithmetic is settled: 4 real multiplies + 52 real adds (56 flops).** Radix-8 with all
  butterflies written out, split-radix, and the Winograd length-8 module all give exactly this
  (§2.3). FFTW's `n1_8` gives exactly this. With FMA, FFTW's variant is 44 adds + 8 FMAs.
  There is no arithmetic win left to find.
* Peak liveness **12 vector registers** + 1 constant (1/√2) → fits AVX2 with 3 registers spare.
  This is the most comfortable of the four sizes.
* The only multiplicative constant is 1/√2. Every other twiddle in a radix-8 butterfly is
  ±1 or ±i, i.e. free (sign flip / real–imaginary swap). Special-casing all of them is where
  the 12× swing from M₁ = 48 to M₅ = 4 in Burrus Table 9.1 comes from — do not let a generic
  complex-multiply routine anywhere near this codelet.
* FFTW's `q1_8` (8×8 = 64-point transposing stage, 528 adds + 256 muls, 132 live reals) is the
  documented upper limit of the fused-transpose trick ("not bigger than 8 or so").
* *(derived)* 512 complex doubles = 8192 B per volume; 3·64 = 192 line transforms of length 8
  per volume, 192·56 = 10 752 flops, ~21 flops/point.

### L = 17 (prime; 4913 points per volume)

* **No shipped codelet anywhere.** FFTW ships primes only up to 13 (`n1_11`, `n1_13`); VkFFT's
  hand-written radix set stops at 16 with a generic Rader path above. This is the size where
  writing your own module actually buys something nobody else has.
* **Two published targets, from Burrus:** Table 7.1 gives **70 real multiplies + 314 real adds
  (384 flops)**; Table 7.2 gives **82 multiplies + 274 adds (356 flops)** — 7% fewer total
  flops at the cost of 17% more multiplies. Pick according to your FMA balance; on a
  2-FMA-port core the second is the better starting point, and neither has been optimised for
  FMA at all.
* For reference, a naive 17×17 complex matrix–vector product costs 17² = 289 complex
  multiplies (1156 real mults + 578 real adds) plus 17·16 = 272 complex accumulations
  (544 real adds) ≈ **2278 flops** before exploiting any symmetry. So a 356–384-flop
  Winograd/Rader module is a **~6× arithmetic win** over dense — a far larger relative win
  than at L = 6 or 8, where dense is only ~4–5× off. *(derived)*
* **Rader structure.** 17 is prime and 17 − 1 = 16 is a pure power of two, which makes 17 the
  friendliest prime in its neighbourhood: Rader's permutation turns the 17-point DFT into a
  **16-point cyclic convolution**, and a 16-point cyclic convolution can be done with
  power-of-two FFTs whose counts are known exactly (Burrus Table 9.1: split-radix 16 = 24
  mult + 144 adds). Burrus's ch. 7 makes the general point: "*the operation counts depend on
  the factorability of P − 1. The primes 11, 23, and 47 are all of the form 1 + 2P₁ making the
  design of efficient FFTs for these lengths more difficult.*" 17 is the opposite case.
* **What `genfft` would do, from its source** (§1.5): `rader_min = 13` → Rader; and
  `alternate_convolution = 17` → the *symmetric/antisymmetric* convolution variant, which
  splits the length-16 sequences into even and odd parts so that the six length-16 transforms
  involved are each real-input (half cost) rather than three full complex ones. The source
  comment says only "*Seems to work better for small sizes. I have no idea why*", but the
  reason is visible in the code: real/imaginary symmetry halves each sub-transform. **If you
  hand-derive a 17-point module, derive it this way** — it is the variant FFTW's generator
  would pick at exactly p = 17.
* **Register pressure is the real constraint here.** Extrapolating §7.2 (n = 16 needs 19
  vector registers, n = 32 needs 40), a straight-line Rader-17 kernel will need roughly
  25–40 vector registers plus a substantial constant set (the 16 pre-transformed
  ω-sequence values). Expect spills on AVX2 and a tight fit on AVX-512. The mitigation is
  the same as everywhere else: split the 16-point convolution into two register-resident
  halves with the intermediate in L1, following §1.3's √n cut.

### L = 36 = 2²·3² (46 656 points per volume)

* **4 and 9 are coprime → Good–Thomas/PFA, and it wins on *both* multiplies and adds.**
  From Burrus Table 10.1, N = 36 PFA = **80 real multiplies + 480 real adds (560 flops)**;
  WFTA = 88 mult (80 non-unity) + 488 adds. *(derived, from Burrus eq. 10.18 and Table 7.1:
  9·T(4) + 4·T(9) = 9·(0,16) + 4·(20,84) = (80, 480) — reproducing Table 10.1 exactly, which
  validates the recursion for the other factorisations below.)*
* *(derived)* Alternatives, computed with the same table so they are directly comparable:
  * **Cooley–Tukey 4×9**: 9 × DFT-4 (144 adds, 0 mult) + twiddle stage + 4 × DFT-9 (336 adds,
    80 mult). The twiddle stage has 36 − 12 = 24 non-trivial complex multiplies at 4 mult +
    2 add ⇒ 96 mult + 48 add. Total ≈ **176 mult + 528 adds (704 flops)**.
  * **Cooley–Tukey 6×6**: 6 × DFT-6 (216 adds, 48 mult) twice + ~25 non-trivial complex
    twiddles (100 mult + 50 add) ⇒ ≈ **196 mult + 482 adds (678 flops)**.
  * **PFA 4×9: 80 mult + 480 adds (560 flops).** PFA beats both CT variants on multiplies by
    more than 2× and on total flops by ~20%, because the coprime index map kills the twiddle
    stage entirely.
* **Do not build one 36-point straight-line codelet.** §7.2 says n = 32 already needs 40 vector
  registers; a 36-point kernel would need ~45 and spill even on AVX-512. Build it as **two
  register-resident stages** — a size-4 stage (6 live vectors) and a size-9 stage (15 live
  vectors, plus 14–19 distinct constants — keep those as broadcast-from-memory FMA operands
  rather than in registers and it fits AVX2 with one register spare, comfortably on
  AVX-512) — with the intermediate 36-complex
  array in L1. This is exactly the "shorter modules … can be kept in cpu registers rather
  general memory. However, the shorter modules … require more passes through the data"
  trade-off Burrus states in ch. 10 §10.4.
* If AVX2's 16 registers make the size-9 stage spill, note the alternative factorisation
  36 = 4·9 = (2·2)·(3·3), i.e. four register-cheap stages (radix 2, 2, 3, 3: 2, 2, 5, 5 live
  vectors) at the cost of more passes. Balance this the way VkFFT's `isGoodSequence` test does
  (§5): reject factorisations whose fattest stage exceeds 16 registers *or* exceeds 2× the
  thinnest.
* **Cache blocking starts here, and only here.** *(derived)* 36³ = 46 656 complex doubles =
  **746 kB (729 KiB) per volume**, which does not fit L2 on most parts and certainly not L1. A single
  36-point line is 576 B — 9 cache lines — so a batch of 36-point lines is the natural blocking
  unit. Burrus/Johnson–Frigo's warning applies directly here and not at all at L = 6/8:
  "*cache implementations strongly favor accessing consecutive data … (accessing data at
  power-of-two intervals in memory, which is distressingly common in FFTs, is thus especially
  prone to cache-line conflicts)*". 36 is *not* a power of two, which is a genuine advantage:
  the natural strides 36, 36² = 1296 (complex doubles → 576 B, 20 736 B) are less
  conflict-prone than the L = 8 strides 8 and 64 (128 B, 1024 B), which are exact
  power-of-two multiples of the cache line and will alias hard in a direct-mapped or
  low-associativity L1. Expect to need explicit padding for **L = 8**, and less so for L = 36.

### Summary table

| | L = 6 | L = 8 | L = 17 | L = 36 |
|---|---|---|---|---|
| factorisation | 2·3 (coprime) | 2³ | prime | 2²·3² (4,9 coprime) |
| best published 1D count (real mult + add) | 8 + 36 | 4 + 52 | 70+314 or 82+274 | 80 + 480 (PFA) |
| source | Burrus T7.1 + eq. 10.18; = FFTW `n1_6` | Burrus T7.1 = T9.1 radix-8 = T9.1 split-radix = FFTW `n1_8` | Burrus T7.1 / T7.2 | Burrus T10.1 |
| shipped FFTW codelet? | yes (`n1_6`, `t1_6`, `n1fv_6`, `q1_6`) | yes (`n1_8`, `t1_8`, `n1fv_8`, `q1_8`) | **no** | **no** (built from `t1_4`+`t1_9` etc.) |
| recommended derivation | PFA 2×3, single codelet | radix-8/split-radix, single codelet, all trivial twiddles special-cased | Rader → 16-pt cyclic convolution, symmetric/antisymmetric variant | PFA 4×9, two stages |
| peak live vector regs (measured or est.) | 9 (+2 const) | 12 (+1 const) | ~25–40 (est.) | 6 then 15 (two stages); ~45 if fused (don't) |
| fits AVX2 (16)? | yes | yes | no | yes, per stage (size-9 stage marginal) |
| fits AVX-512 (32)? | yes | yes | marginal | yes |
| cache blocking needed? | no (3.4 kB/volume) | no (8.2 kB/volume) | no (79 kB/volume) | **yes** (746 kB/volume) |
| power-of-two stride aliasing risk | low | **high** — pad | low | low |

---

## 9. Unsourced engineering notes

Everything in this section is my own reasoning or my own measurement on this machine. It is
attributed to nobody and cited by no one. Treat it as a hypothesis to be benchmarked, not as
literature.

1. **The liveness numbers in §7.2 are measurements, not citations.** They come from a
   last-use analysis I wrote (`tools/fftw_codelet_liveness.py`, see §7.2) applied to FFTW 3.3.10's shipped
   codelet source. They assume `genfft`'s emitted statement order, count one register per live
   value, ignore the possibility of rematerialisation, and do not model two-operand
   destructive instructions. They are validated only against a hand count for `n1fv_4`.
   Treat them as ±20%.
2. **All op counts marked *(derived)*** in §8 — the CT 4×9 and CT 6×6 estimates for L = 36,
   the flops-per-point figures, the dense-versus-Rader comparison at L = 17, the per-volume
   byte counts — are arithmetic I did from Burrus's Table 7.1 counts and his recursion 10.18.
   The PFA 4×9 derivation is self-validating (it reproduces Table 10.1's published row for
   N = 36 exactly, and the same recursion reproduces FFTW's shipped `n1_6` and `n1_12` counts
   exactly), which is why I trust it. The CT estimates depend on my count of "non-trivial
   twiddles" and are the weakest numbers in this document.
3. **For this batched workload I expect the vectorisation axis to be the batch, not the
   transform.** FFTW's SIMD scheme (Frigo & Johnson §IX) extracts 2-way parallelism *within* a
   single complex DFT by treating it as a pair of real DFTs, `DFT(A + iB) = DFT(A) + i·DFT(B)`,
   and 4-way by packing two complex numbers per vector. That is the right answer when you have
   one transform. We have thousands. Laying the batch out as the innermost (unit-stride)
   dimension — "batch-major", `[x][y][z][batch]` with batch a multiple of 4 (AVX2 double) or 8
   (AVX-512) — makes every load and store in the codelet a contiguous aligned vector load, makes
   every twiddle a broadcast, removes all shuffles, and removes the real/imaginary interleave
   problem entirely. The codelet then looks exactly like the scalar codelet with `double`
   replaced by `__m256d`, and the register budget is the *scalar* liveness count from §7.2
   (17 for n=6, 19 for n=8) rather than the SIMD one. That is over 16 — so for AVX2 the
   batch-major form of even n = 6 and n = 8 will spill a little. The trade is: a few spill
   stores to L1 versus zero shuffles and perfect streaming. Given Blake's finding (§4.2) that
   leaf *memory* behaviour dominates, I would bet on batch-major and measure.
4. **Consider generating the codelets rather than writing them.** Every source in this section
   that produced fast small-n code produced it with a generator, and the reason is the one
   Frigo gives (§1.1, "Rapid turnaround"): the interaction between emission order and the C
   compiler's register allocator is not predictable, so you want to be able to regenerate and
   re-time in minutes. A ~300-line Python emitter that (a) builds the DAG for the chosen
   derivation, (b) canonicalises constants positive, (c) does CSE, (d) schedules by the
   recursive-halving rule of §1.3, and (e) prints C with `__m256d`/`__m512d` intrinsics, would
   reproduce most of `genfft`'s value for four fixed sizes. The three highest-value pieces, in
   order: positive constants (10–15% per Frigo), special-casing every trivial twiddle (12× on
   multiply count at n = 8 per Burrus Table 9.1), and the recursive-halving schedule (reported
   as good as SPIRAL's per-machine search).
5. **Check the compiler's pre-RA scheduler.** Frigo's `-fno-schedule-insns` result (50–100%
   on 1999 SPARC) is old, and Blake explicitly questions whether it still holds. But it costs
   one build to test `-fno-schedule-insns`, `-fno-schedule-insns2`, and
   `-fsched-pressure` / `--param=max-sched-region-insns` on the L = 8 and L = 36 codelets, and
   to count `vmovupd` spill traffic in the emitted assembly. Do that before believing any
   arithmetic count.
6. **A cheap, high-information first experiment.** Emit the L = 8 codelet three ways —
   (i) batch-major scalar-shaped with AVX2 intrinsics, (ii) FFTW-style `n1fv_8`-shaped, and
   (iii) a naive nested-loop radix-2 — and measure all three at a batch size that keeps the
   working set in L1. The published counts say (i) and (ii) do identical arithmetic and (iii)
   does ~12× the multiplies; if the measured spread is smaller than that, you are
   memory-bound and §4 is the section to reread, not §2.

---

## 10. Bibliography, with fetch status

Fetched and read in this session:

1. **M. Frigo**, "A Fast Fourier Transform Compiler," *Proc. ACM SIGPLAN 1999 Conf. on
   Programming Language Design and Implementation (PLDI)*, Atlanta, GA, May 1999.
   <https://www.fftw.org/pldi99.pdf> — **fetched** (PDF, text extracted locally).
   ACM record: <https://dl.acm.org/doi/abs/10.1145/301618.301661> (not fetched).
2. **M. Frigo and S. G. Johnson**, "The Design and Implementation of FFTW3," *Proc. IEEE*
   **93**(2), pp. 216–231, 2005. <https://www.fftw.org/fftw-paper-ieee.pdf> — **fetched**.
3. **M. Püschel, J. M. F. Moura, J. R. Johnson, D. Padua, M. M. Veloso, B. W. Singer, J. Xiong,
   F. Franchetti, A. Gačić, Y. Voronenko, K. Chen, R. W. Johnson, N. Rizzolo**, "SPIRAL: Code
   Generation for DSP Transforms," *Proc. IEEE* **93**(2), pp. 232–273, Feb. 2005.
   <https://users.ece.cmu.edu/~moura/papers/ieeeproceedings-pueschelmouraetal-feb05-ieeexplore.pdf>
   — **fetched**.
4. **C. S. Burrus (ed.)**, *Fast Fourier Transforms*, Connexions collection `col10550`
   (authors: C. S. Burrus, M. Frigo, S. G. Johnson, M. Püschel, I. Selesnick; collection
   structure revised 18 Nov 2012). Rice University repository PDF:
   <https://repository.rice.edu/server/api/core/bitstreams/01e9e0a5-fa6f-453d-a1b5-8209fa0a565c/content>
   — **fetched**. Cited here for ch. 7 (Winograd's short DFT algorithms, Tables 7.1 and 7.2),
   ch. 9 (Cooley–Tukey / split-radix, Table 9.1), ch. 10 (PFA and WFTA, Table 10.1 and §10.4),
   and ch. 11 ("Implementing FFTs in Practice", by S. G. Johnson and M. Frigo).
5. **S. G. Johnson and M. Frigo**, "A Modified Split-Radix FFT With Fewer Arithmetic
   Operations," *IEEE Trans. Signal Processing* **55**(1), pp. 111–119, Jan. 2007.
   <https://math.mit.edu/~stevenj/papers/JohnsonFr07.pdf> — **fetched**. Also
   <https://www.fftw.org/newsplit.pdf> — **fetched**. *Caveat: this paper's Tables I–III are
   rendered as images and did not extract; only its prose numbers are quoted here.*
6. **A. M. Blake**, *Computing the fast Fourier transform on SIMD microprocessors*, PhD thesis,
   University of Waikato, 2012. Handle <https://hdl.handle.net/10289/6417>; record page
   <https://researchcommons.waikato.ac.nz/entities/publication/74489df7-8dc5-4c1f-b29f-4ea992667b85>
   — **fetched**; PDF
   <https://researchcommons.waikato.ac.nz/bitstreams/8209dc72-3ba2-4ba0-a79e-e4d77bad0736/download>
   — **fetched**.
7. **FFTW manual**, "Generating your own code."
   <https://www.fftw.org/fftw3_doc/Generating-your-own-code.html> — **fetched**. Content is
   thin: it says the codelet set is specified by `{dft,rdft}/{codelets,simd}/*/Makefile.am`
   and regenerated via `sh bootstrap.sh` then `make`, and that the generator is "rather
   sophisticated" and not for "casual users". It does **not** list default sizes; the size
   lists in §1.6 come from the `Makefile.am` files themselves.

Local source trees read directly (paths on this machine):

8. **FFTW 3.3.10 source**, `/home/lqcd/wdetmold/fft/ext/src/fftw-3.3.10/`. Specifically
   `genfft/magic.ml`, `genfft/fft.ml`, `support/Makefile.codelets`,
   `dft/scalar/codelets/{Makefile.am,n1_*.c,t1_*.c,q1_*.c}`,
   `dft/simd/common/{n1fv_*.c,t1fv_*.c}`. Upstream: <https://www.fftw.org/>.
9. **VkFFT source**, `/home/lqcd/wdetmold/fft/ext/src/VkFFT/`. Specifically
   `vkFFT/vkFFT/vkFFT_CodeGen/vkFFT_KernelsLevel1/vkFFT_RadixKernels.h` and
   `vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_HostFunctions/vkFFT_Scheduler.h`.
   Upstream: <https://github.com/DTolm/VkFFT>.
10. **heFFTe "stock" backend**,
    `/home/lqcd/wdetmold/fft/ext/install/avx512/include/stock_fft/heffte_stock_algos.h`
    (mentioned only to note it is not a codelet generator).

Cited but **not** fetched — do not treat any specific number as attributed to these:

11. **A. M. Blake, I. H. Witten, M. J. Cree**, "The Fastest Fourier Transform in the South,"
    *IEEE Trans. Signal Processing* **61**(19), pp. 4707–4716, Oct. 2013 —
    `[UNVERIFIED — could not fetch]` (publisher paywall and mirror both returned HTTP 403).
    The bibliographic details here come from search-result metadata, not from the document.
    Everything attributed to Blake in §4 comes from the 2012 thesis (item 6), which I did read.
12. **H. J. Nussbaumer**, *Fast Fourier Transform and Convolution Algorithms*, Springer —
    `[UNVERIFIED — not obtained]`. Referenced only to say that the Winograd short-DFT module
    tables in §2.2 are quoted from Burrus (item 4), **not** from Nussbaumer, and that no number
    in this document should be attributed to Nussbaumer's tables.
13. **Intel**, "Intel AVX-512 Instructions" —
    `[UNVERIFIED — could not fetch]` (<https://www.intel.com/content/www/us/en/developer/articles/technical/intel-avx-512-instructions.html>
    returned HTTP 403). The 16-vs-32 vector-register counts in §7.1 were instead verified
    directly on this machine by assembling `vaddpd %zmm31, %zmm30, %zmm29` and
    `vaddpd %ymm15, %ymm14, %ymm13`.
14. **S. Kral et al.** (the assembly-emitting `genfft` variant), **R. Tolimieri et al.**
    (the n=13 Rader form Frigo compares against), **R. Yavne** (1968, split radix),
    **Hong & Kung** (1981, red-blue pebbling), **Aggarwal & Vitter** (1988) — all cited
    here only *as Frigo/Johnson and Burrus cite them*. None was fetched; do not attribute
    page numbers or exact counts to them from this document.
