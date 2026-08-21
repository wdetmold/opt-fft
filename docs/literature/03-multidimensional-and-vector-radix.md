# 03 — Genuinely Multidimensional Algorithms: Vector-Radix, Polynomial Transforms, and the Four/Six-Step Formulation

**Scope.** Beyond row–column. What the literature says about computing a 3D DFT as a
*single* multidimensional algorithm rather than as three passes of 1D transforms, and
whether any of it is worth implementing for complex-double cubes of L = 6, 8, 17, 36 on
single-threaded x86-64 with AVX2/AVX-512.

---

## 0. Bottom line up front

**For L = 6, 8, 17, 36 the literature says: do not implement vector-radix, and do not
implement Nussbaumer–Quandalle polynomial transforms.** Use separable row–column, and
spend your effort on SIMD-across-the-orthogonal-dimensions, batch vectorization, and
keeping the working set resident. The reasons are quantitative, not hand-waving:

1. Vector-radix's arithmetic saving **shrinks to nothing as the transform gets small**.
   In Duhamel & Vetterli's published 2D tables, at 4×4 vector-radix saves *exactly zero*
   operations versus row–column; at 8×8 it saves 25% of the multiplications but only
   **4.2% of total operations**, because additions outnumber multiplications ~11:1 at
   that size ([D&V90], Tables 4 and 5 — numbers in §2.3).
2. Duhamel & Vetterli's own verdict on radix-2 vector-radix is: *"it should not be
   considered"* — and they say so precisely *because* good compact 1D FFTs exist ([D&V90] §9.5).
3. Nussbaumer–Quandalle polynomial transforms buy, for the one 3D cube they publish
   (7×7×7), **1002 real multiplications versus 1029 for Winograd — a 2.6% saving —
   at a cost of 11 820 real additions** ([N&Q78], Table 5). Trading multiplies for adds
   is backwards on FMA hardware; FFTW says so in as many words (§3.3).
4. The whole minimal-arithmetic program is worth ~25% on power-of-two sizes, while
   implementation quality is worth a factor of **5–40** ([JF08] §1). You are optimizing
   the wrong term if you chase op counts.

The genuinely multidimensional idea that *does* pay for two of our four sizes is
**Good–Thomas / PFA used per axis**, which turns 6³ and 36³ into higher-rank DFTs with
*no twiddle factors at all* (§5). That is a real structural win, and it is free.

---

## 1. The taxonomy: four classes, and who invented what

Duhamel & Vetterli's invited review is the canonical taxonomy. Verbatim ([D&V90] §9,
p. 289):

> "The methods for computing this transform are distributed in four classes: row-column
> algorithms, vector-radix algorithms, nested algorithms and polynomial transform
> algorithms. Among them, only the vector-radix and the polynomial transform were
> specifically designed for the 2-D case."

And from their introduction (§2.4, p. 265):

> "The two most interesting approaches are certainly the vector radix FFT (a direct
> approach to the multi-dimensional problem in a Cooley-Tukey mood) proposed in 1975 by
> Rivard [91] and the polynomial transform solution of Nussbaumer and Quandalle in 1978
> [87, 88]. Both algorithms substantially reduce the complexity over traditional
> row-column computational schemes."

Note the tension: the *introduction* says both "substantially reduce the complexity",
while §9.5 (the discussion, after the tables) says VR2 "should not be considered". Read
§9.5, not the introduction — and read the tables, not the prose.

Primary attributions, as printed in the reference list of [D&V90] (bibliographic data
verified by reading that list; the individual papers were **not** fetched):

- **[91]** G. E. Rivard, "Algorithm for direct fast Fourier transform of bivariant
  functions," 1975 Annual Meeting of the Optical Society of America, Boston, MA, Oct. 1975.
- **[92]** G. E. Rivard, "Direct fast Fourier transform of bivariant functions,"
  *IEEE Trans. ASSP*, vol. 25, no. 3, June 1977, pp. 250–252.
- **[85]** R. M. Mersereau and T. C. Speake, "A unified treatment of Cooley-Tukey
  algorithms for the evaluation of the multidimensional DFT," *IEEE Trans. ASSP*,
  vol. 22, no. 5, Oct. 1981, pp. 320–325.  *(Volume/year as printed in [D&V90]; the
  volume number looks like a typo in the original — treat with care.)*
- **[86]** Z. J. Mou and P. Duhamel, "In-place butterfly-style FFT of 2-D real
  sequences," *IEEE Trans. ASSP*, vol. ASSP-36, no. 10, Oct. 1988, pp. 1642–1650.
- **[87]** H. J. Nussbaumer and P. Quandalle, *IBM J. Res. Develop.*, vol. 22, 1978,
  pp. 134–144. **(This one I fetched and read in full — see §3.)**
- **[103]** B. Lhomme, J. Morgenstern and P. Quandalle, "Implantation de transformées de
  Fourier de dimension 2," *Techniques et Science Informatiques*, vol. 4, no. 2, 1985,
  pp. 324–328.

The Wikipedia article on vector-radix additionally credits, in its reference list
(metadata read from the fetched article; papers not fetched):

- **Harris, McClellan, Chan & Schuessler**, "Vector radix fast Fourier transform,"
  *ICASSP '77*, vol. 2, pp. 548–551 (extension to rectangular arrays and arbitrary radices).
- **Chan & Ho**, "Split vector-radix fast Fourier transform," *IEEE Trans. Signal
  Processing*, vol. 40, no. 8, 1992, pp. 2029–2039.
- **Pei & Wu**, "Split vector radix 2D fast Fourier transform," *ICASSP '87*, vol. 12,
  pp. 1987–1990.
- **Wu & Paoloni**, "On the two-dimensional vector split-radix FFT algorithm,"
  *IEEE Trans. ASSP*, vol. 37, no. 8, 1989, pp. 1302–1304.
- **Dudgeon & Mersereau**, *Multidimensional Digital Signal Processing*, Prentice Hall, 1983.

---

## 2. Vector-radix FFT

### 2.1 What it is

Vector-radix (VR) is Cooley–Tukey applied to **all indices simultaneously**. For the 2D
radix-2 DIT case, [D&V90] eq. (72) decimates both indices by 2 at once, splitting the
input into the four sub-arrays x(2i,2j), x(2i+1,2j), x(2i,2j+1), x(2i+1,2j+1), and then
exploits

> "the redundancy in the computation of X(k,r), X(k+N/2,r), X(k,r+N/2) and
> X(k+N/2,r+N/2) [which] leads to simplifications which allow to reduce the arithmetic
> complexity." — [D&V90] §9.2

In 3D the same construction decimates all three indices by 2, giving eight sub-cubes and
a **2×2×2 butterfly**: one 8-point 3D DFT (which is pure additions — a 3-fold tensor of
2-point butterflies) followed by 7 non-trivial twiddle multiplications per group of 8
outputs, iterated log₂L times. Higher radices exist; [D&V90] warns about the cost:

> "the complexity of these butterflies increases very quickly with the radix: a radix-2
> butterfly involves 4 inputs (it is a 2x2 DFT followed by some 'twiddle factors'), while
> VR4 and VSR butterflies involve 16 inputs."

In 3D those figures become 8 inputs for VR-2×2×2 and **64 inputs for VR-4×4×4** — a
64-point 3D butterfly. That is a large, irregular straight-line block, and it is the
reason VR4 in 3D is essentially never implemented.

[D&V90] also flags the practical restriction:

> "Note also that the only VR algorithms that have seriously been considered all apply to
> lengths that are powers of 2, although other radices are of course feasible."

**This alone eliminates VR for L = 6, 17, and 36.** Only L = 8 is a power of two.

### 2.2 The textbook arithmetic-saving claim, and what it actually means

The Wikipedia article on vector-radix gives the standard counts for an N^M array with
radix-2 VR:

- Vector-radix complex multiplications: `((2^M − 1)/2^M) · N^M · log₂N`
- Row–column complex multiplications: `(M · N^M/2) · log₂N`

Substituting M = 3 (**my arithmetic on their formula**, not a quoted result):

| | complex mults for L³ | ratio |
|---|---|---|
| Row–column (radix-2) | `1.500 · L³ log₂L` | 1.000 |
| VR 2×2×2 | `0.875 · L³ log₂L` | 0.583 |

i.e. a nominal **41.7% reduction in complex multiplications** in 3D, versus 25% in 2D.
That 41.7% is the number people quote when they say vector-radix is worth it in 3D.

**Three reasons it is a mirage at our sizes:**

1. Both sides of that formula count *radix-2* algorithms and count *every* twiddle as a
   full complex multiply. A real row–column implementation uses split-radix or a
   hand-optimized codelet per axis, which is much cheaper than radix-2 — and has **no
   inter-dimension twiddles at all**, because row–column is exactly separable.
2. It counts only multiplications. Additions dominate at small N.
3. It says nothing about data movement, which is what actually costs time (§6).

### 2.3 The reality check: Duhamel & Vetterli's Tables 4 and 5

These are the most useful published numbers in the entire multidimensional literature for
our purposes, because they include the small sizes. Both are **2D, real data,
per-output-point** counts. Columns are `R.C. | VR2 | VR4 | VSR | WFTA | P.T.`

**Table 4 — non-trivial real multiplications per output point** ([D&V90] p. 292):

| N×N | R.C. | VR2 | VR4 | VSR | WFTA | P.T. |
|---|---|---|---|---|---|---|
| 2×2 | 0 | 0 | — | 0 | — | 0 |
| 4×4 | 0 | 0 | 0 | 0 | — | 0 |
| 8×8 | 0.5 | 0.375 | — | 0.375 | — | 0.375 |
| 16×16 | 1.25 | 1.25 | 0.844 | 0.844 | — | 0.844 |
| 32×32 (30×30 WFTA) | 2.125 | 2.062 | 1.43 | — | 1.435 | 1.336 |
| 64×64 | 3.0625 | 3.094 | 2.109 | 2.02 | — | 1.834 |
| 256×256 (240×240) | 5.015 | 5.273 | 3.48 | 3.28 | 1.82 | 2.833 |
| 1024×1024 (1008×1008) | 7.004 | 7.506 | 4.878 | 4.56 | 3.12 | 3.83 |

**Table 5 — real additions per output point** ([D&V90] p. 293):

| N×N | R.C. | VR2 | VR4 | VSR | WFTA | P.T. |
|---|---|---|---|---|---|---|
| 2×2 | 2 | 2 | — | 2 | — | 2 |
| 4×4 | 3.25 | 3.25 | 3.25 | 3.25 | — | 3.25 |
| 8×8 | 5.56 | 5.43 | — | 5.43 | — | 5.43 |
| 16×16 | 8.26 | 8.14 | 7.86 | 7.86 | — | 7.86 |
| 32×32 (30×30) | 11.13 | 11.06 | 10.43 | — | 12.98 | 10.34 |
| 64×64 | 14.06 | 14.09 | 13.11 | 13.02 | — | 12.83 |
| 256×256 (240×240) | 20.01 | 20.27 | 18.48 | 17.67 | 22.79 | 17.83 |
| 1024×1024 (1008×1008) | 26.00 | 26.5 | 23.88 | 23.56 | 45.30 | 22.83 |

> **Caveat on the tables.** Rows 2×2 through 16×16 and 64×64 print only four or five
> values across six columns, so which specific alternative column a value belongs to is
> ambiguous in the scan. What is *unambiguous* — and is all my argument needs — is that
> the **first** column is R.C. and that all remaining entries in those rows are equal to
> the value I quote. My column assignment in the tables above for the sparse rows is a
> best-effort reading, marked with `—` where a column is blank. Do not quote the sparse
> rows' column labels; do quote the R.C. value and the best-alternative value.

**Now do the small-size arithmetic (my computation from their numbers):**

| N×N | R.C. total ops/pt | best alternative | total saving | mults as % of R.C. ops |
|---|---|---|---|---|
| 4×4 | 3.25 | 3.25 | **0.0%** | 0% |
| 8×8 | 6.06 | 5.805 | **4.2%** | 8.3% |
| 16×16 | 9.51 | 8.704 | 8.5% | 13% |
| 64×64 | 17.12 | 14.664 | 14.3% | 18% |
| 1024×1024 | 33.00 | 26.66 | 19.2% | 21% |

The pattern is unmistakable and it is the single most important fact in this document:
**the multidimensional-algorithm advantage is monotonically increasing in N and is
essentially zero at the sizes we care about.** At 8×8 vector-radix does deliver the
promised 25% multiplication saving (0.5 → 0.375) — but multiplications are only 8% of the
work, so the total drops 4.2%. At 4×4 there is literally nothing to win.

### 2.4 Duhamel & Vetterli's verdict, verbatim

From §9.5 "Discussion" ([D&V90] p. 293):

> "VR2 is more complicated to implement than row-column algorithms, and requires more
> operations for lengths ⩾32. Therefore, it should not be considered. Note that this
> result holds only because efficient and compact 1-D FFTs, such as SRFFT, have been
> developed."

> "The row-column algorithm is the one allowing the easiest implementation, while having a
> reasonable arithmetic complexity. Furthermore, it is easily parallelized, and
> simplifications can be found for the reorderings (bit reversal, and matrix transposition
> [66]), allowing one of them to be free in nearly any kind of implementation."

> "VSR is difficult to implement, and will certainly seldom defeat VR4, except in very
> special cases (huge memory available and N very large)."

> "VR4 is a good compromise between structural and arithmetic complexity. When row-column
> algorithms are not fast enough, we think it is the next choice to be considered."

Read the last one carefully: VR4 is the *fallback* for when row–column is not fast enough,
and the escape hatch for VSR is explicitly "N very large". Our N is 6 to 36.

### 2.5 Mapping to L = 6 / 8 / 17 / 36

- **L = 6.** VR needs radix-2^k per axis on a power-of-two length. 6 is not. A
  VR-2×2×2 first stage is *formally* possible (6 = 2·3, so one radix-2 stage then a
  3×3×3 base case), but you would be writing a 2×2×2 butterfly plus twiddles to save a
  fraction of the multiplications on a 216-point transform whose entire working set is
  3.4 KB. **No.** Use PFA (§5).
- **L = 8.** The only size where classical VR-2×2×2 cleanly applies: three stages of
  64 butterflies each. By direct analogy with the 8×8 row of Tables 4/5, expect a
  ~25% multiplication saving and a **~4–6% total-operation saving** — and pay for it
  with a butterfly that touches 8 points strided by 1, L, and L² simultaneously, which
  destroys the unit-stride SIMD structure that makes row–column fast (§6, §8). **No.**
- **L = 17.** Prime. No vector-radix of any radix exists. **Not applicable.**
- **L = 36.** 36 = 4·9. A VR-2×2×2 or VR-4×4×4 decomposition on the factor-4 part is
  formally possible, leaving a 9×9×9 base case. The VR-4×4×4 butterfly has **64 inputs**.
  Given that D&V put VR4's break-even against row–column well above N = 32 in 2D, and
  that 3D makes the butterfly 4× larger in inputs, this is a large amount of code for a
  saving that the tables say is under ~10%. **No.** Use PFA (§5).

---

## 3. Nussbaumer–Quandalle polynomial transforms

### 3.1 The primary source and what it actually claims

**[N&Q78]** H. J. Nussbaumer and P. Quandalle, "Computation of Convolutions and Discrete
Fourier Transforms by Polynomial Transforms," *IBM Journal of Research and Development*,
vol. 22, no. 2, March 1978, pp. 134–144. Received June 6, 1977; revised October 18, 1977.
Nussbaumer at Compagnie IBM France, La Gaude; Quandalle at the University of Nice.
*(Fetched and read in full.)*

Abstract, verbatim:

> "Discrete transforms are introduced and are defined in a ring of polynomials. These
> polynomial transforms are shown to have the convolution property and can be computed in
> ordinary arithmetic, without multiplications. Polynomial transforms are particularly
> well suited for computing discrete two-dimensional convolutions with a minimum number of
> operations. Efficient algorithms for computing one-dimensional convolutions and Discrete
> Fourier Transforms are then derived from polynomial transforms."

The mechanism for DFTs (their §"Computation of DFTs by polynomial transforms", p. 141) is
**Rader composed with multidimensional convolution**:

> "When N = N₁, with N₁ prime, this transform can be computed as a correlation by using
> Rader's algorithm [17] ... Thus, if N₁ and N₂ are primes, the DFT of dimension N₁N₂ can
> be calculated with one transform of dimension N₁, one correlation of dimension N₁ − 1
> and one two-dimensional correlation of dimension (N₁ − 1) × (N₂ − 1). We propose here to
> compute the transform of dimension N₂ and the correlation of dimension N₂ − 1 by
> Winograd's algorithms and to calculate the correlation of dimension (N₁ − 1) × (N₂ − 1)
> by polynomial transforms."

And, crucially for L = 17 — which is prime, so per-axis Rader gives a 16-point cyclic
convolution, and three axes give a 16×16×16 one:

> "This method can of course be extended recursively to DFTs of length N = N₁N₂, ..., Nᵢ
> provided the various factors Nᵢ are primes. In this case the computation is performed
> with multidimensional polynomial transforms."

That is *exactly* the L = 17³ structure. Nussbaumer–Quandalle is the right theoretical
tool for a prime 3D cube. Which makes their own published numbers all the more damning.

### 3.2 The numbers — [N&Q78] Table 5 and Table 4

**Table 5, "Number of real operations for DFTs computed by polynomial transforms"**
(p. 141). Figures in parentheses exclude "simple" multiplications by ±1, ±j:

| DFT size | P.T. mults | P.T. adds | Winograd mults | Winograd adds |
|---|---|---|---|---|
| 63 | 174 (168) | 1408 | 234 | 1440 |
| 504 | 1392 (1356) | 14540 | 1872 | 14796 |
| 1008 | 3132 (3084) | 34668 | 4212 | 35244 |
| 2520 | 8352 (8316) | 96364 | 11232 | 102348 |
| **9×9** | **242 (224)** | **1742** | **338** | **1936** |
| **7×7** | **138 (136)** | **1156** | **162** | **1152** |
| **7×7×7** | **1002 (998)** | **11820** | **1458** | **13896** |

And immediately after the table, in their own words:

> "The savings are much smaller, however, when the figures are compared to the Winograd
> method applied to larger fields. In the case of a DFT of 7 × 7 × 7 points, for example,
> this newer Winograd method [12] requires 1029 real multiplications against 1002
> multiplications for the polynomial transform approach."

**7×7×7 is the only 3D cube in the paper, and it is the closest published analogue to our
17³.** The verdict from the authors of the method:

- Polynomial transform: **1002 real mults, 11 820 real adds** for 343 points.
- Winograd (the 1977 method, ref [12] = S. Winograd, "A New Method for Computing DFT,"
  *ICASSP 1977*, Hartford, p. 366): **1029 real mults**. A **2.6% multiplication saving.**
- Total real operations: **12 822 for 343 points = 37.4 real ops per point**, with an
  **add:mult ratio of 11.8:1**.

Also relevant, **Table 4, "Number of operations for short multidimensional convolutions
computed by polynomial transforms"** (p. 140) — these are the convolution kernels that a
Rader-based prime cube would need:

| Convolution size | Total mults | Total adds |
|---|---|---|
| 3×3×3 | 40 | 325 |
| 3×3×3×3 | 121 | 1324 |
| **6×6×6** | **320** | **3896** |
| 6×6×6×6 | 1936 | 31552 |

The 6×6×6 line has a 12.2:1 add:mult ratio. Note this is a *cyclic convolution* of size
6³, not a DFT of size 6³ — do not confuse the two.

Their own closing hedge on where polynomial transforms win, verbatim:

> "It is also better than FFT for convolution sizes up to around 100 × 100."

Our largest axis is 36. We are far below their own stated crossover in every dimension.

### 3.3 Why the multiply/add trade is backwards on our hardware

This is the decisive practical point, and FFTW states it twice, in print, unambiguously.

[FJ05] §II, verbatim:

> "FFTW does not employ the Winograd FFT [30], which minimizes the number of
> multiplications at the expense of a large number of additions. (This tradeoff is not
> beneficial on current processors that have specialized hardware multipliers.)"

[JF08] §2, verbatim:

> "There is also the Winograd FFT [2,4–6], which minimizes the number of multiplications
> at the expense of a large number of additions; this trade-off is not beneficial on
> current processors that have specialized hardware multipliers."

And on the split-radix lineage generally, [FJ05] §II:

> "(Unfortunately, as we argue in this paper, minimal-arithmetic, fixed-factorization
> implementations tend to no longer be optimal on recent computer architectures.)"

On an AVX2 Haswell-or-later core with two FMA units, a multiply and an add cost the same
issue slot, and an FMA folds one of each into a single slot. An algorithm that removes
27 multiplications (7³: 1029 → 1002) while carrying 11 820 additions is strictly worse
than one that balances them, because the balanced one can use FMA. **Polynomial
transforms and Winograd optimize for a machine that no longer exists.**

### 3.4 Mapping to L = 6 / 8 / 17 / 36

- **L = 17.** The theoretically correct home for this technique (prime cube → 16³ cyclic
  convolution → multidimensional polynomial transform), and by the authors' own 7³ number
  it buys ~2.6% of the multiplications at 12:1 add:mult. **No.** Use per-axis Rader with a
  16-point convolution done as a 1D transform, or — see §6.4 — consider a direct
  matrix-vector 17-point DFT.
- **L = 6.** 6³ decomposes to 2³ ⊗ 3³ by PFA (§5); the N&Q machinery is not needed and
  their 6×6×6 *convolution* line (320 mults / 3896 adds) is not a 6³ DFT count. **No.**
- **L = 8, 36.** Powers of small primes; the N&Q construction is built on Rader for prime
  factors and does not apply cleanly. **Not applicable.**

---

## 4. The four-step / six-step (Bailey) formulation

### 4.1 Primary source, verified

**[Bai90]** David H. Bailey, "FFTs in External or Hierarchical Memory," *Journal of
Supercomputing*, vol. 4, no. 1, March 1990, pp. 23–35. Preprint dated December 30, 1989.
*(Fetched and read in full.)* Also appeared at Supercomputing '89.

Bailey is explicit that he is *not* the inventor, and the provenance matters:

> "However, as it turns out, this algorithm was actually first presented over twenty years
> ago in a paper by Gentleman and Sande [8, p. 569]. This early paper even described the
> application of this algorithm to a system with hierarchical memory. Unfortunately, this
> algorithm appears to have been largely forgotten in the interim, as a number of more
> recent papers have suggested much less efficient methods."

He also credits recent variants to "Agarwal and Cooley [1, p. 150], Ashworth and Lyne
[4, p. 219], and Swarztrauber [10, p. 202–203]."

### 4.2 The four steps, verbatim

For n = n₁n₂, matrices column-major (Fortran):

> "1. Perform n₁ simultaneous n₂-point FFTs on the input data considered as a n₁ × n₂
>    complex matrix.
> 2. Multiply the resulting data, considered as a n₁ × n₂ matrix A_jk, by e^(±2πijk/n).
>    The sign is the sign of the transform.
> 3. Transpose the resulting n₁ × n₂ complex matrix into a n₂ × n₁ matrix.
> 4. Perform n₂ simultaneous n₁-point FFTs on the resulting n₂ × n₁ matrix."

And his statement of why it wins:

> "note that both of the simultaneous FFT steps can be performed using exclusively unit
> stride data access, which is optimal on virtually any computer system. Secondly, this
> algorithm produces an ordered transform (provided the simultaneous FFTs are ordered) —
> it is not necessary to perform a bit reversal permutation, which is inefficient on many
> advanced computer systems."

On sizing: "On many systems, the implementation of this algorithm is most efficient when
n₁ and n₂ are as close as possible to √n."

### 4.3 The six steps, verbatim

> "1. Transpose the input data set, considered as a n₁ × n₂ complex matrix, into a n₂ × n₁ matrix.
> 2. Perform n₁ individual n₂-point one dimensional FFTs on the resulting n₂ × n₁ matrix.
> 3. Multiply the resulting n₂ × n₁ complex matrix A_ij by e^(±2πijk/n).
> 4. Transpose the resulting n₂ × n₁ matrix into a n₁ × n₂ matrix.
> 5. Perform n₂ individual n₁-point one dimensional FFTs on the resulting n₁ × n₂ matrix.
> 6. Transpose the resulting n₁ × n₂ complex matrix into a n₂ × n₁ matrix."

Bailey's own caution: the six-step "has the serious disadvantage of requiring an
additional two transpose steps, which typically are the chief bottlenecks on any system
with a distributed or external memory."

### 4.4 The performance numbers, and the honest admission

Bailey's key concession, verbatim:

> "Depending on implementation, the four step FFT algorithm may actually require a
> slightly larger number of floating-point arithmetic operations than conventional FFT
> algorithms. In spite of this slight handicap, it is remarkably efficient even for a
> single processor"

Measured, single-processor, versus Cray's tuned library routine (n = 2^m):

| m | Cray-2 four-step | Cray-2 library | Cray Y-MP four-step | Cray Y-MP library |
|---|---|---|---|---|
| 8 | 42.5 MFLOPS | 57.2 | 68.68 | 137.85 |
| 12 | 137.8 | 130.6 | 201.90 | 173.51 |
| 16 | 187.5 | 159.0 | 236.78 | 188.26 |
| 20 | 204.4 | 176.8 | 244.23 | 197.20 |

**Read the m = 8 row.** At n = 256 the four-step is *1.35× slower* on the Cray-2 and
**2.0× slower** on the Y-MP than the straightforward library FFT. The four-step only
crosses over around m = 11–12 (n = 2048–4096) and its win at m = 20 is 16% (Cray-2) to
24% (Y-MP). Eight-processor Y-MP speedups reach 7.891× at m = 20 (Table 3).

**This is a size-dependent technique and our sizes are on the wrong side of the
crossover.** 36³ = 46 656 points total is comparable to m ≈ 15.5, but the *individual
axis* transforms are length 36 — the four-step would be applied to n = 36 = 6·6, i.e.
√n = 6, which is far below any crossover Bailey measures.

### 4.5 How it maps to 3D — and the important observation

There is a structural identity worth being explicit about, because it is the reason the
four-step feels familiar:

> **The four-step FFT of a 1D transform of size n = n₁n₂ is a 2D row–column FFT of an
> n₁ × n₂ array with a twiddle stage in between. The six-step is the same with the
> transposes made explicit.**

Consequently, for a *genuine* 3D cube L³ the four/six-step formulation adds nothing new
at the top level: three passes of L² independent length-L transforms is already exactly
the "simultaneous FFTs with unit stride" structure Bailey is trying to manufacture, and
we get it **without any twiddle stage**, because a separable multidimensional DFT has no
inter-dimension twiddles. Bailey's algorithm exists to *make* a 1D problem look like our
3D problem. We already have the good structure for free.

Where four-step *is* relevant to us is one level down, **inside** a single axis:

- **L = 36:** the length-36 axis transform can itself be organized as a four-step with
  n₁ × n₂ = 6 × 6 (Cooley–Tukey, needs a 6×6 twiddle table) or — better — as a *pure*
  Good–Thomas 4 × 9 with **no twiddles at all** (§5). Prefer Good–Thomas.
- **L = 6:** as 2 × 3, coprime → Good–Thomas, no twiddles.
- **L = 8, 17:** no useful coprime split; four-step offers nothing.

FFTW's own use of the four-step confirms it is a large-size tool. [FJ05] §II, verbatim:

> "(On the other end of the scale, a 'radix' of roughly √n has been called a four-step FFT
> [18], and we have found that one step of such a radix can be useful for large sizes in
> FFTW; see Section IV-D1.)"

"Useful for large sizes." Not ours.

---

## 5. Good–Thomas / PFA: the multidimensional idea that *does* pay for L = 6 and L = 36

This is the one genuinely multidimensional technique in this document that I recommend.

### 5.1 The verified facts

From the Wikipedia article on the prime-factor FFT algorithm (fetched; a secondary source,
but the underlying mathematics is textbook and it cites the primary works correctly):

- Attribution: **I. J. Good**, "The interaction algorithm and practical Fourier analysis,"
  *J. Royal Statistical Society, Series B*, 20(2): 361–372, 1958; and **L. H. Thomas**,
  "Using a computer to solve problems in physics," in *Applications of Digital Computers*,
  Ginn, Boston, 1963. Follow-up: **I. J. Good**, "The relationship between two fast Fourier
  transforms," *IEEE Trans. Computers*, 100(3): 310–317, 1971.
  *(The 1971 Good paper is also cited as ref [18] in [N&Q78], independently corroborating it.)*
- Requirement: n = n₁n₂ with n₁, n₂ **relatively prime**.
- The index map is the **Chinese Remainder Theorem** map `m ↦ (m mod n_d)`, with the
  inverse reconstructing indices via idempotents `e_d` satisfying `Σ e_d = 1 (mod n)`.
- The payoff, quoted: PFA operates without the *"extra multiplications by roots of unity
  called twiddle factors"* that Cooley–Tukey requires; the price is *"more complicated
  re-indexing of the data based on additive group isomorphisms."*
- It re-expresses a size-n DFT as *"a two-dimensional N₁ × N₂ DFT"* — formally the tensor
  product `⊗_d DFT_{ω_{n_d}}`.

### 5.2 The consequence for L = 6 and L = 36 (my analysis, from the PFA property above)

Apply Good–Thomas **independently along each of the three axes**. Because a separable 3D
DFT is a tensor product of 1D DFTs, and each 1D DFT factors as a tensor product of coprime
sub-DFTs under a pure index permutation, the whole 3D transform factors:

**L = 6 (216 points):**
```
DFT_{6×6×6}  ≅  DFT_{2×2×2}  ⊗  DFT_{3×3×3}
                 (8 points)      (27 points)
```
The 216-point 3D transform is a **27-fold batch of 8-point 3D DFTs followed by an 8-fold
batch of 27-point 3D DFTs** (or the reverse), connected by nothing but an index
permutation — **zero twiddle multiplications between the two halves.** The 2×2×2 block is
pure additions. The 3×3×3 block is 27 points with only the ω₃ constants involved.

**L = 36 (46 656 points):**
```
DFT_{36×36×36}  ≅  DFT_{4×4×4}  ⊗  DFT_{9×9×9}
                    (64 points)     (729 points)
```
A **729-fold batch of 64-point 3D DFTs** and a **64-fold batch of 729-point 3D DFTs**,
again with no twiddles between them. Both blocks are individually well-conditioned for
straight-line code, and — note — 4 and 9 being coprime is exactly the condition the
project brief already identified.

Two things to like about this beyond the twiddle saving:
- **Batch structure falls out for free.** Both halves are large batches of small
  identical transforms, which is precisely the shape that vectorizes across the batch
  index with zero shuffles (§6.3, §8).
- **[D&V90] notes PFA's table advantage:** *"this drawback of PFA is compensated by the
  fact that only a few coefficients have to be stored. On the contrary, classical FFTs
  must store a large table of sine and cosine values, calculate them as needed, or update
  them with resulting roundoff errors."* For a fixed-size codelet, few constants means
  they can live in registers or a tiny read-only table.

The counterweight, also from [D&V90] §10.1, and it is real:

> "This is why the PFA and WFTA do not meet the performance expected from their
> computational complexity only."

— because of the irregular CRT index permutation. For a *fixed, known* L = 6 or L = 36
the permutation is a compile-time constant, so it can be baked into the addressing of a
generated codelet rather than computed at run time. That is the whole reason this project
is generating size-specific code, and it neutralizes D&V's objection.

### 5.3 L = 8 and L = 17

Neither has a coprime factorization (8 = 2³, 17 prime). PFA does not apply. L = 8 is
classic radix-8/split-radix; L = 17 is Rader/Winograd/hand-derived. Both are covered by
other sections of this corpus.

---

## 6. Is separable row–column actually optimal in practice? The memory-and-SIMD argument

This is the question the assignment asks to answer with sources. The answer is yes for our
sizes, and there are four independent lines of published evidence.

### 6.1 Load/store dominates arithmetic — [D&V90] §10.1

Verbatim:

> "When monitoring the amount of time spent in various elementary floating point
> operations, it is interesting to note that more time is spent in load/store operations
> than in actual arithmetic computations [30, 107, 109] (this is due to the fact that
> memory access times are comparable to ALU cycle times on current machines). Therefore,
> the locality of the algorithm is of paramount importance. This is why the PFA and WFTA
> do not meet the performance expected from their computational complexity only."

And, on the general futility of op-count-chasing:

> "Note that very often, the difference in computational complexity between algorithms is
> not large enough so as to differentiate between the efficiency of the algorithm and the
> quality of the implementation."

They also report that polynomial transforms — the lowest-arithmetic method in their
tables — **were measured slower than VR2**: *"It was even reported to be slower than VR2
[103]."* (ref [103] = Lhomme, Morgenstern & Quandalle, 1985.) That is the lowest-op-count
multidimensional algorithm losing in wall-clock to a middling one.

### 6.2 The 25%-versus-40× argument — [JF08] §1

The most quotable number in the whole implementation literature. Verbatim:

> "the difference in the number of arithmetic operations, for power-of-two sizes n, between
> the 1965 radix-2 Cooley–Tukey algorithm (∼5n log₂n [1]) and the currently lowest-known
> arithmetic count (∼(34/9) n log₂ n [11,12]) remains only about 25%."

> "And yet there is a vast gap between this basic mathematical theory and the actual
> practice—highly optimized FFT packages are often an order of magnitude faster than the
> textbook subroutines, and the internal structure to achieve this performance is radically
> different from the typical textbook presentation of the 'same' Cooley–Tukey algorithm."

> "fig. 1 plots the ratio of benchmark speeds between a highly optimized FFT [16,17] and a
> typical textbook radix-2 implementation [18], and the former is faster by a factor of
> 5–40 (with a larger ratio as n grows)."

**25% total available from all arithmetic cleverness ever discovered, versus 5–40× from
implementation.** Set against the 0–8.5% that vector-radix offers at our sizes (§2.3),
the priority ordering is not close.

### 6.3 FFTW explicitly declines vector-radix and says why

FFTW is the most-benchmarked FFT implementation in existence — [FJ05] §III reports
"extensive benchmarks of FFTW's performance, along with that of over 50 other FFT
implementations, on most modern general-purpose processors, comparing complex and
real-data transforms in one to three [dimensions]". It does not implement vector-radix.
The one time the term appears, it is in a parenthetical *disclaiming* it — [FJ05] §IV-D2
and [JF08] §4.2.2, verbatim:

> "Another example of the effect of loop reordering is a style of plan that we sometimes
> call vector recursion (unrelated to 'vector-radix' FFTs [16])."

(ref [16] there is [D&V90] itself.)

What FFTW does instead for rank > 1 — [FJ05] §IV-C4:

> "These plans reduce a multidimensional DFT problem to problems of lower rank, which are
> then solved recursively."

and the crucial performance rationale, [FJ05] §IV-D:

> "A depth-first style is also used for the multidimensional plans of Section IV-C4, where
> in this case the planner can (and often does) choose the optimal cache-oblivious
> algorithm: it breaks the transform into subproblems of roughly equal rank. In contrast,
> an iterative, 'breadth-first' approach might perform all of the 1-D transforms for the
> first dimension, then all of the 1-D transforms for the second dimension, and so on,
> which has extremely poor cache performance compared to grouping the dimensions into
> smaller multidimensional transforms."

**This is the single most actionable sentence in this document for L = 36.** Note what it
does and does not say. It does *not* say "use vector-radix". It says: naive row–column
where you sweep the whole array once per axis is bad; instead **group dimensions into
smaller multidimensional sub-blocks and recurse**. That is cache blocking of a separable
algorithm — you keep the separable arithmetic and fix only the traversal order. It is the
correct middle ground between naive row–column and vector-radix, and it costs no extra
arithmetic.

FFTW's other relevant machinery: the *vector loop* (rank-0 / higher-vector-rank plans,
[JF08] §4.2.3), which wraps a loop of identical transforms around a codelet — exactly the
batch structure a 3D cube hands you — and cache-oblivious square transposition ([JF08]
§4.2.1): *"FFTW implements the square transposition ... by means of the 'cache-oblivious'
algorithm from [35], which is fast and, in theory, uses the cache optimally regardless of
the cache size."*

### 6.4 SIMD: shuffles are the budget, and small-N direct DFT can win

**[MAF11]** Daniel S. McFarlin, Volodymyr Arbatov, Franz Franchetti (Carnegie Mellon),
"Automatic SIMD Vectorization of Fast Fourier Transforms for the Larrabee and AVX
Instruction Sets," *Proc. International Conference on Supercomputing (ICS)*, 2011.
*(Fetched and read.)*

Verified numbers and statements:

- Headline: *"speed-up of 5.5–6.5 for 8-way AVX and 10–12.5 for 16-way LRB"* (measured
  using runtime or instruction counts).
- Vectorization efficiency: *"Ideally, the vectorization efficiency should approach the
  architecture's vector width. However, due to the required shuffles, this is not
  achievable. Figure 11 shows that across vector architectures and lengths, we achieve an
  efficiency of up to about 80% of the vector length."* — i.e. **~20% of the theoretical
  SIMD width is lost to data reorganization even with a superoptimizer generating the
  shuffle sequences.**
- On AVX specifically: *"AVX operates most efficiently ... Cross-lane operations are
  limited and expensive."* (AVX registers are two 128-bit lanes.) This is exactly the cost
  a vector-radix butterfly would incur, because it must combine data from different
  dimensions — i.e. different strides — inside one register.
- Baselines used: Cooley–Tukey *"requires 5n log₂(n) operations"*; split-radix
  *"requires 4n log₂(n) − 6n + 8 many operations."*
- Machine for the AVX runtime comparison: *"a 3.3 GHz Intel Core i5-2500"*, against Intel
  IPP 7.0; *"Spiral's AVX performance compares well with IPP 7.0 on the full range of DFT
  sizes. Note, that this early platform implementing the AVX ISA does not feature support
  for FMAs."*

**And the finding that matters most for L = 6, 8, 17** — verbatim:

> "Next we investigate the trade-off between fast O(n log₂ n) algorithms and direct O(n²)
> computations for small kernel sizes. For these sizes the shuffles required by the fast
> algorithms can become prohibitive while the regular, FMA-friendly structure of the
> matrix-vector product allows for high efficiency. Figure 12 shows that indeed up to a
> size of about n = 20, the direct computation is preferable, even though the mathematical
> operations count (counting only additions and multiplications) is inferior. The reason is
> LRB's dedicated replicate HW, which enables efficient scalar broadcasts and FMA
> instructions which are well-suited for a direct computation."

**Read the caveat carefully before acting on this.** Figure 12's caption is *"Comparison
of vector operation counts for FFTs and 'DFTs by definition' on LRB"* — this is a
**vector-instruction-count** comparison on **Larrabee**, whose "replicate HW" (embedded
broadcast) is the stated reason. It is **not** a measured AVX result. However, AVX-512
does have embedded broadcast (`{1toN}` / `vbroadcastsd`) and FMA, so LRB is the closer
architectural analogue of AVX-512 than of AVX1. Treat "direct O(n²) DFT wins below
n ≈ 20" as a **strong hypothesis worth benchmarking on AVX-512 for L = 6, 8, and 17
axis transforms** — all three are below 20 — and as unproven for AVX2 (4-way double,
no embedded broadcast). L = 36 is above the threshold.

### 6.5 What production multidimensional libraries actually do

Two data points on real code, both fetched:

- **VkFFT** (github.com/DTolm/VkFFT) advertises *"1D/2D/3D/ND systems — specify
  VKFFT_MAX_FFT_DIMENSIONS for arbitrary number of dimensions"* and, for kernels, only
  **1-D radices**: *"Radix-2/3/4/5/7/8/11/13 FFT. Sequences using radix 3, 5, 7, 11 and 13
  have comparable performance to that of powers of 2."* For primes: *"Rader's FFT algorithm
  for primes from 17 up to max shared memory length (~10000). Inlined and done without
  additional memory transfers."* No vector-radix kernel is documented.
  (Note the coincidence: VkFFT's Rader path starts *exactly at 17*, and radices 3 and 8
  cover our L = 6 and L = 8 axes.)
- **cuFFTDx** (NVIDIA) ships small-fixed-3D-cube examples, described verbatim as
  *"Small 3D (equal dimensions) FP32 FFT that fits into a single block"* and
  *"In fft_3d_box_single_block and fft_3d_cube_single_block samples cuFFTDx is used on a
  thread-level to executed small 3D FFTs in a single block."* The documentation I could
  reach describes the FFT operator only in terms of 1-D sizes (e.g. "8-point FFT",
  "16-point FP32/FP64 R2C FFT") and does not document a native 3D butterfly — but I could
  **not** find an explicit statement of the operator's dimensionality, so I am not
  asserting that cuFFTDx composes 1-D passes; I only note that no multidimensional
  butterfly is documented.

So: the two modern libraries with first-class small/fixed-size multidimensional support
both expose **1-D radix kernels plus Rader**, and neither documents a vector-radix
butterfly. That is weak evidence, but it points the same way as everything else.

---

## 7. Work specifically on *small* 3D cubes (as opposed to large 3D grids)

**This is the biggest gap in the literature, and implementers should know it.** I searched
for it deliberately and found essentially nothing directly on point:

- The classical multidimensional papers ([D&V90] §9, Nussbaumer–Quandalle, the
  vector-radix line) treat 2D and are motivated by **image processing**, with worked
  examples at 512×512 and 1024×1024. [D&V90] explicitly motivates the memory discussion
  with *"a length 1024 × 1024 DFT requires 10⁶ words of storage, and the matrix is
  therefore stored in mass memory."* That is the opposite regime from ours.
- The 3D FFT performance literature is overwhelmingly about **large grids on parallel
  machines** — global transposes, MPI all-to-all, distributed six-step. Not applicable
  single-node.
- Bailey's four/six-step ([Bai90]) is explicitly an **external/hierarchical memory**
  technique and its own measurements show it *losing* at small n (§4.4).
- The one clearly on-target artifact is **cuFFTDx's small-single-block 3D cube examples**
  (§6.5), which establish that "small fixed 3D cube, fully resident, batched" is a
  recognized real workload — but it is GPU, and the docs do not give the algorithm or any
  performance numbers.
- Takahashi has a line of work on a *"block three-dimensional FFT algorithm"* for cache
  reuse and on SSE2-vectorized 3D FFT (e.g. "An Implementation of Parallel 3-D FFT Using
  Short Vector SIMD Instructions on Clusters of PCs", Springer LNCS, doi
  10.1007/11558958_139; and "A Blocking Algorithm for FFT on Cache-Based Processors",
  doi 10.1007/3-540-48228-8_58). **[UNVERIFIED — could not fetch; link.springer.com
  redirects to an authentication endpoint.]** Search snippets attributed ">1.3 GFLOPS on an
  8-node dual Pentium III 1 GHz cluster" and ">5 GFLOPS on a 16-node dual Xeon 2.8 GHz
  cluster" to this work, but **I did not verify those numbers against the papers and they
  should not be quoted.** In any case they are multi-node and large-grid.

**Practical consequence:** there is no published measurement telling you what the fastest
6³/8³/17³/36³ complex-double kernel looks like. You are in genuinely unmeasured territory,
which means (a) do not trust any op-count argument without benchmarking, and (b) the
generic evidence in §6 — locality and SIMD dominate, arithmetic savings are ≤ 8% at these
sizes — is the best guidance available and it points clearly at separable + heavily
vectorized + resident.

---

## 8. Per-size verdicts

| | L = 6 (2·3), 216 pts | L = 8 (2³), 512 pts | L = 17 (prime), 4913 pts | L = 36 (2²·3²), 46 656 pts |
|---|---|---|---|---|
| **Vector-radix 2×2×2** | Not applicable (6 not 2^k); formally one stage possible, pointless | Only clean case. ~25% of mults, **~4–6% of total ops** by analogy with [D&V90] 8×8 row. Destroys unit-stride SIMD. **Don't** | Not applicable (prime) | Formally possible on the factor-4 part; VR4 butterfly = 64 inputs. **Don't** |
| **Nussbaumer–Quandalle polynomial transform** | Not the natural fit | Not applicable | *Theoretically* the right tool (Rader → 16³ cyclic convolution). Authors' own 7³ number: 2.6% of mults, 11.8:1 add:mult. **Don't** | Not applicable |
| **Four-step / six-step (Bailey)** | Superseded by PFA (no twiddles) on the 2×3 axis split | Nothing to offer | Nothing to offer | Superseded by PFA (4×9, no twiddles) on the axis split |
| **Good–Thomas / PFA per axis** | **Yes.** 6³ ≅ 2³ ⊗ 3³, zero twiddles between halves | N/A (8 = 2³) | N/A (prime) | **Yes.** 36³ ≅ 4³ ⊗ 9³, zero twiddles between halves |
| **Separable row–column + batch SIMD** | **Yes** — and whole cube is 3.4 KiB, L1-resident | **Yes** — 8 KiB, L1-resident | **Yes** — 76.8 KiB, L2-resident | **Yes, with FFTW-style dimension grouping** ([FJ05] §IV-D) — 729 KiB, L2-resident |
| **Direct O(n²) matrix–vector for the axis transform** | Worth benchmarking (n=6 ≪ 20) | Worth benchmarking (n=8 ≪ 20) | Worth benchmarking (n=17 < 20) | No (n=36 > 20) |

Working-set figures are mine: L³ × 16 bytes for complex double (1 KiB = 1024 B).

---

## 9. Unsourced engineering notes

Everything in this section is **my own analysis**, attributable to nobody. It is
consistent with the sourced material above but is not itself sourced.

**9.1 The real reason row–column wins on a 3D cube: free SIMD with zero shuffles.**
A separable 3D transform of L³ gives you, at each of three passes, **L² independent
length-L transforms**. That is 36 independent transforms for L = 6, 64 for L = 8, 289 for
L = 17, 1296 for L = 36. Vectorize *across* those independent transforms, not within one:
load 4 (AVX2) or 8 (AVX-512) neighbouring transforms' element-k into one register, run the
scalar butterfly code verbatim on vectors, and you have **a perfectly vectorized FFT with
not a single shuffle instruction.** For the two axes whose elements are strided by L and
L², the neighbours in the *contiguous* direction are exactly what you want in the lanes —
so those two passes are unit-stride vector loads. Only the innermost (unit-stride) axis
needs a transpose to get into that form, and for fixed small L that transpose is a
compile-time-known register shuffle block done once, amortized over all L² transforms.
Given [MAF11]'s finding that shuffles cost ~20% of SIMD width even when superoptimized,
an algorithm that needs *zero* shuffles in two of three passes is worth far more than
8.5% of the additions.

Vector-radix does the opposite: its butterfly deliberately couples elements at strides
1, L, and L² inside one operation, so the SIMD lanes must hold data from different
dimensions. Every butterfly then needs cross-lane work, which on AVX2 is the expensive
kind.

**9.2 The batch dimension makes this even more lopsided.** The stated workload is LQCD
momentum projection over time slices and spin–colour components — i.e. **many volumes
batched**. That batch index is a fourth, fully independent, unit-stride-if-you-lay-it-out-
that-way dimension. With a batch of ≥ 8 volumes interleaved (SoA over the batch), *every*
pass of *every* axis vectorizes across the batch with zero shuffles, for all four L, and
the twiddle constants are shared across all lanes. In that layout the arithmetic-op-count
question becomes almost irrelevant: you are issuing dense FMA on full vectors regardless
of which separable factorization you chose. **Choose the layout first; it dominates the
algorithm choice.**

**9.3 Cache placement of the four sizes** (L³ × 16 bytes, one volume):

| L | points | bytes | fits |
|---|---|---|---|
| 6 | 216 | 3.4 KiB | L1d (32 KiB) comfortably; ~9 volumes fit in L1 |
| 8 | 512 | 8 KiB | L1d; exactly 4 volumes fit in L1 |
| 17 | 4 913 | 76.8 KiB | exceeds L1; comfortably L2 |
| 36 | 46 656 | 729 KiB | exceeds L1; fits a 1 MiB+ L2, tight on a 512 KiB L2 |

So L = 6 and L = 8 are **register/L1 problems** — cache blocking is irrelevant, and the
right move is fully unrolled straight-line code over the whole cube with the batch in the
lanes. L = 17 and especially L = 36 are the only two where [FJ05]'s "group the dimensions
into smaller multidimensional transforms" advice bites: for 36³, do not sweep the full
729 KiB array three times; tile it into sub-blocks (e.g. process a 36 × T × T slab so the
working set is a few tens of KB) and do two or three axes' worth of work per residency.

**9.4 On the 6³ ≅ 2³ ⊗ 3³ factorization being unusually pretty.** Under per-axis
Good–Thomas the full 216-point 3D transform becomes an 8-point 3D DFT (all additions, no
multiplications at all beyond sign flips) tensored with a 27-point 3D DFT, joined by a
pure index permutation. For a fixed size with a code generator, that permutation is free —
it is just the addressing in the generated code. This is about as clean a decomposition as
any of the four sizes admits, and it is the single most promising structural idea for
L = 6. Worth writing and benchmarking against a flat 216-point fully-unrolled codelet.

**9.5 Suggested priority order for implementers.**
1. Fix the memory layout for the batch (SoA over volumes) — this decides everything else.
2. Separable row–column with per-axis codelets, vectorized across the batch.
3. For L = 6 and L = 36, try the per-axis Good–Thomas factorization (§5.2) against a
   flat codelet.
4. For L = 6, 8, 17, benchmark a direct O(n²) matrix–vector axis transform against the
   fast one, per [MAF11]'s n ≈ 20 finding — but note that finding is Larrabee
   vector-op-counts, so verify on your actual AVX-512 hardware.
5. For L = 36, add FFTW-style dimension grouping / tiling.
6. Only if all of the above is exhausted and profiling shows you are arithmetic-bound
   rather than shuffle- or load-bound, reconsider vector-radix for L = 8. The literature
   says you will not get there.

---

## 10. Gaps — what I looked for and could not find

- **No published operation counts for 3D vector-radix at small L.** [D&V90]'s Tables 4/5
  are 2D and real-data only. Every 3D figure in this document is either the textbook
  formula (§2.2) or my extrapolation from the 2D 8×8 row, and both are labelled as such.
- **No 3D analogue of [D&V90] Tables 4/5.** Nobody appears to have published the
  equivalent per-output-point comparison for 3D complex data. This would be the single
  most useful missing table.
- **Nussbaumer's book, chapter 7** ("Computation of Discrete Fourier Transforms by
  Polynomial Transforms", in *Fast Fourier Transform and Convolution Algorithms*,
  Springer) is behind Springer authentication and I could not fetch it. It likely contains
  more multidimensional DFT op-count tables than the 1978 IBM paper. **[UNVERIFIED]**
- **Takahashi's block-3D-FFT papers** — Springer paywall; numbers in search snippets are
  explicitly **not** verified and must not be quoted (§7).
- **Harris/McClellan/Chan/Schuessler ICASSP 1977** and **Rivard 1977** — the primary
  vector-radix papers. I have their bibliographic details from two independent fetched
  reference lists but could not fetch either paper, so I cannot confirm their internal
  op counts.
- **ResearchGate** returned HTTP 403 for the multidimensional split vector-radix DIF paper;
  I therefore cite no numbers from it.
- **cuFFTDx's actual 3D algorithm** — the docs do not state whether the small-3D-cube
  example composes 1-D passes or uses a native 3D butterfly. I did not assert either.
- **No single-node, single-threaded, small-fixed-3D-cube benchmark anywhere**, on any
  algorithm. This is the central gap (§7).

---

## 11. References

### Fetched and read in this session (verified)

1. **[N&Q78]** H. J. Nussbaumer and P. Quandalle, "Computation of Convolutions and
   Discrete Fourier Transforms by Polynomial Transforms," *IBM Journal of Research and
   Development*, vol. 22, no. 2, March 1978, pp. 134–144.
   <https://mirrors.meulie.net/bitsavers.org/pdf/ibm/IBM_Journal_of_Research_and_Development/222/ibmrd2202D.pdf>
2. **[D&V90]** P. Duhamel and M. Vetterli, "Fast Fourier Transforms: A Tutorial Review and
   a State of the Art," *Signal Processing*, vol. 19, no. 4, April 1990, pp. 259–299
   (Invited Paper, Elsevier).
   <https://norbertwiener.umd.edu/Research/Duhamel_Vetterli_FFT_90.pdf>
3. **[Bai90]** D. H. Bailey, "FFTs in External or Hierarchical Memory," *Journal of
   Supercomputing*, vol. 4, no. 1, March 1990, pp. 23–35 (preprint dated 30 Dec 1989).
   <https://www.davidhbailey.com/dhbpapers/fftq.pdf>
4. **[FJ05]** M. Frigo and S. G. Johnson, "The Design and Implementation of FFTW3,"
   *Proceedings of the IEEE*, vol. 93, no. 2, 2005, pp. 216–231 (Invited Paper).
   <https://math.mit.edu/~stevenj/papers/FrigoJo05.pdf> and
   <https://www.fftw.org/fftw-paper-ieee.pdf>
5. **[JF08]** S. G. Johnson and M. Frigo, "Implementing FFTs in Practice," ch. 11 in
   *Fast Fourier Transforms*, C. S. Burrus (ed.), Rice Univ.: Connexions, 2008.
   <https://math.mit.edu/~stevenj/papers/JohnsonFr08-burrus.pdf>
6. **[MAF11]** D. S. McFarlin, V. Arbatov, F. Franchetti, "Automatic SIMD Vectorization of
   Fast Fourier Transforms for the Larrabee and AVX Instruction Sets," *Proc. Int. Conf. on
   Supercomputing (ICS)*, 2011.
   <https://users.ece.cmu.edu/~franzf/papers/ics2011.pdf>
7. **[FFTW97]** M. Frigo and S. G. Johnson, "The Fastest Fourier Transform in the West,"
   MIT-LCS-TR-728, MIT, 11 September 1997. (Fetched; contains no 3D content.)
   <https://www.fftw.org/fftw-paper.pdf>
8. Wikipedia, "Vector-radix FFT algorithm" (fetched incl. `Special:Export` wikitext for
   the exact operation-count formulas and reference list). Secondary source; used only for
   the textbook op-count formula and for bibliographic metadata.
   <https://en.wikipedia.org/wiki/Vector-radix_FFT_algorithm>
9. Wikipedia, "Prime-factor FFT algorithm" (fetched). Secondary source; used for the CRT
   index map, the no-twiddle property, and the Good/Thomas citations.
   <https://en.wikipedia.org/wiki/Prime-factor_FFT_algorithm>
10. VkFFT README (fetched) — documented radices, ND support, Rader-from-17.
    <https://github.com/DTolm/VkFFT>
11. NVIDIA cuFFTDx 1.2.0 examples page (fetched) — `fft_3d_cube_single_block`.
    <https://docs.nvidia.com/cuda/cufftdx/1.2.0/examples.html>
12. NVIDIA cuFFTDx 1.2.0 API methods page (fetched; inconclusive on dimensionality).
    <https://docs.nvidia.com/cuda/cufftdx/1.2.0/api/methods.html>

### Bibliographic metadata read from a fetched reference list; papers themselves NOT fetched

13. G. E. Rivard, "Direct fast Fourier transform of bivariant functions," *IEEE Trans.
    ASSP*, vol. 25, no. 3, June 1977, pp. 250–252. *(via [D&V90] ref [92] and the fetched
    Wikipedia reference list.)*
14. G. E. Rivard, "Algorithm for direct fast Fourier transform of bivariant functions,"
    1975 Annual Meeting of the Optical Society of America, Boston, Oct. 1975.
    *(via [D&V90] ref [91].)*
15. Harris, McClellan, Chan & Schuessler, "Vector radix fast Fourier transform,"
    *ICASSP '77*, vol. 2, pp. 548–551. *(via fetched Wikipedia reference list.)*
16. Z. J. Mou and P. Duhamel, "In-place butterfly-style FFT of 2-D real sequences," *IEEE
    Trans. ASSP*, vol. ASSP-36, no. 10, Oct. 1988, pp. 1642–1650. *(via [D&V90] ref [86].)*
17. R. M. Mersereau and T. C. Speake, "A unified treatment of Cooley-Tukey algorithms for
    the evaluation of the multidimensional DFT," *IEEE Trans. ASSP*, vol. 22, no. 5, Oct.
    1981, pp. 320–325. *(via [D&V90] ref [85]; volume number appears mis-set in the
    original.)*
18. B. Lhomme, J. Morgenstern, P. Quandalle, "Implantation de transformées de Fourier de
    dimension 2," *Techniques et Science Informatiques*, vol. 4, no. 2, 1985, pp. 324–328.
    *(via [D&V90] ref [103] — the source of the "polynomial transforms slower than VR2"
    report.)*
19. S. Winograd, "A New Method for Computing DFT," *Proc. 1977 Int. Conf. Acoust., Speech
    and Signal Processing*, Hartford, p. 366. *(via [N&Q78] ref [12] — the 1029-mult 7×7×7
    figure.)*
20. C. M. Rader, "Discrete Fourier Transforms When the Number of Data Samples is Prime,"
    *Proc. IEEE*, vol. 56, 1968, p. 1107. *(via [N&Q78] ref [17].)*
21. I. J. Good, "The Relationship Between Two Fast Fourier Transforms," *IEEE Trans.
    Computers*, vol. 20, 1971, p. 310. *(via [N&Q78] ref [18]; independently corroborated
    by the fetched Wikipedia PFA article.)*
22. I. J. Good, "The interaction algorithm and practical Fourier analysis," *J. Royal
    Statistical Society B*, 20(2): 361–372, 1958; L. H. Thomas, "Using a computer to solve
    problems in physics," in *Applications of Digital Computers*, Ginn, Boston, 1963.
    *(via fetched Wikipedia PFA article.)*
23. W. M. Gentleman and G. Sande (1966), cited by [Bai90] as ref [8], p. 569 — the true
    origin of the four-step algorithm. *(Full bibliographic details not recovered.)*
24. Chan & Ho, *IEEE Trans. Signal Processing* 40(8):2029–2039, 1992; Pei & Wu,
    *ICASSP '87* 12:1987–1990; Wu & Paoloni, *IEEE Trans. ASSP* 37(8):1302–1304, 1989;
    Dudgeon & Mersereau, *Multidimensional Digital Signal Processing*, Prentice Hall, 1983.
    *(via fetched Wikipedia reference list — split vector-radix line.)*

### Could not fetch — do not rely on

25. H. J. Nussbaumer, *Fast Fourier Transform and Convolution Algorithms*, Springer, ch. 7.
    **[UNVERIFIED — Springer authentication redirect.]**
26. D. Takahashi, "An Implementation of Parallel 3-D FFT Using Short Vector SIMD
    Instructions on Clusters of PCs," Springer LNCS, doi 10.1007/11558958_139; and
    "A Blocking Algorithm for FFT on Cache-Based Processors," doi 10.1007/3-540-48228-8_58.
    **[UNVERIFIED — Springer authentication redirect. Performance numbers seen only in
    search snippets and deliberately not quoted.]**
27. "Design of a multidimensional split vector-radix decimation-in-frequency FFT
    algorithm." **[UNVERIFIED — ResearchGate returned HTTP 403.]**
