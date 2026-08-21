# 02 — Prime and awkward lengths: the L=17 problem, and the coprime structure of L=6 and L=36

**Scope.** Rader (1968), Winograd/WFTA and nested small-n modules, Bluestein/chirp-z, and
the Good–Thomas prime-factor algorithm (PFA), read specifically against the four target
geometries `L = 6, 8, 17, 36` for a batched forward complex-double 3D FFT on AVX2/AVX-512.

**Citation policy used here.** Every URL in §9 was fetched in this session. Bibliographic
details for papers I could *not* fetch (paywalled IEEE/AMS/JRSS originals) were transcribed
from the reference lists of sources I *did* fetch, and are flagged `[BIB-ONLY]`. Anything
that is my own derivation or measurement is marked `[mine]` and collected in §8.

---

## 1. TL;DR for the implementers

**The single most important finding, and it is a measurement, not a citation.**
On this node, with FFTW 3.3.10 `FFTW_PATIENT`, complex double, out-of-place, 1 thread:

| L³ | AVX2 | scalar (`-onosimd`) | SIMD speedup |
|---|---|---|---|
| 6³ | 671 ns | 1.62 µs | **2.41×** |
| 8³ | 1.51 µs | 3.22 µs | **2.13×** |
| 16³ | 16.11 µs | 51.69 µs | **3.21×** |
| **17³** | **103.06 µs** | **102.30 µs** | **1.00×** |
| 18³ | 45.52 µs | 89.04 µs | **1.96×** |
| 36³ | 353.97 µs | 734.31 µs | **2.07×** |

L=17 gets **zero** benefit from AVX2, because FFTW has no vector codelet for 17 and falls
back to scalar C (§6.1). Consequently:

* With SIMD on, 17³ costs **5.3× per point** what 16³ costs (20.98 vs 3.93 ns/pt).
* With SIMD off, that penalty collapses to **1.65×** (20.82 vs 12.62 ns/pt).

**So ~70% of the L=17 penalty is missing vectorization, not arithmetic.** Rader's algorithm
buys you 21% of the flops (§2.4) but *costs* you 15% of the FP instructions (§2.7); the
vectorization buys you 2–3×. Get the priorities in that order.

| L | factorization | the structural fact that matters | what to build |
|---|---|---|---|
| 6 | 2·3 | Coprime → Good–Thomas applies, **zero twiddles**. FFTW's own 6-point codelet already attains the PFA count *exactly*, in both flops and instructions (§5.3). | No arithmetic left to win. FFTW picks a non-FMA `n1fv_6_avx` codelet and reaches only ~19% of FMA peak — go wider and use FMA. |
| 8 | 2³ | Prime power → PFA does **not** apply ("useless for power-of-two sizes" [Wiki-PFA]). Split-radix territory; covered elsewhere. | — |
| 17 | prime | p−1 = 16 = 2⁴. The **friendliest possible large prime**: Rader's cyclic convolution has power-of-two length → no zero-padding, no recursive Rader, no Bluestein. | SIMD-across-the-batch implementation of the dense conjugate-symmetric kernel (§2.5, best instruction count) or Rader-17 (§2.4, best flop count). Benchmark both; the SIMD is the win. |
| 36 | 2²·3² | 4 and 9 coprime → Good–Thomas applies. **FFTW does not use it**; it plans a messy mixed-radix 3/6/12 tree with an extra copy pass. | PFA 4×9: −9% flops, −16% FP instructions, and the whole twiddle stage disappears (§5.4). Fold the index permutation into the 3D gather (§5.5). |

---

## 2. Rader's algorithm and the 17-point DFT

### 2.1 The construction

For prime *p* the non-zero residues mod *p* form a cyclic multiplicative group, so a
primitive root *g* exists and every non-zero index is `g^q mod p` for a unique
`q ∈ {0,…,p−2}`. Substituting `n = g^u`, `k = g^{−v}` turns the (p−1)×(p−1) non-trivial
block of the DFT matrix into a **cyclic correlation of length p−1**:

> "the set of numbers lower than a prime p admits some primitive elements g such that the
> successive powers of g modulo p generate all the elements of the set … and the length-p
> DFT turns out to be a length (p − 1) cyclic correlation" — [DV90] §5.1.2

Duhamel & Vetterli work p=5 out explicitly (eqs. 40–44), including the row/column
permutation that exposes the circulant. Rader's original two-page note is the source
[Rader68, BIB-ONLY]. The DC output is not covered by the convolution and is computed
separately as `X[0] = Σ_n x[n]` [Wiki-Rader].

### 2.2 Why p = 17 is the lucky prime

`p − 1 = 16 = 2⁴`. Everything that normally goes wrong with Rader does not go wrong here:

* The convolution length is a **power of two** → two length-16 FFTs, the cheapest and most
  register-friendly kernels in existence.
* **No zero-padding.** The standard fallback for awkward primes is to zero-pad the
  length-(p−1) convolution "to a length of at least 2(N–1)–1, say to a power of two"
  [Wiki-Rader]. At p=17 you skip it.
* **No recursive Rader.** The pathological case is `p−1 = 2·p₂` with `p₂` prime, and chains
  thereof (Cunningham chains / Sophie Germain primes), which force Rader to recurse
  [Wiki-Rader]. p=17 is the opposite extreme.
* Among nearby primes only 17 gives a pure power of two: 11→10=2·5, 13→12=4·3,
  19→18=2·9, 23→22=2·11 (11 prime — genuinely awkward).

Constants you need `[mine, verified numerically against numpy.fft to 4.4e-15]`:

```
primitive roots mod 17 : 3, 5, 6, 7, 10, 11, 12, 14        (use g = 3)
g = 3,  g^{-1} mod 17 = 6
input gather   a[q] = x[ g^q mod 17 ],  q = 0..15 :
     1, 3, 9, 10, 13, 5, 15, 11, 16, 14, 8, 7, 4, 12, 2, 6
kernel         b[q] = exp(-2*pi*i * (g^{-q} mod 17) / 17)   -- precomputed, DFT'd once
reconstruction X[ g^{-q} mod 17 ] = x[0] + (a (*) b)[q];   X[0] = x[0] + A[0]
```

(`A = DFT₁₆(a)`; its DC bin is already `Σ_{n≠0} x[n]`, so `X[0]` is 2 flops, not 32.)

### 2.3 The two tricks that separate a good Rader from a naive one

1. **Fold the kernel transform and the 1/N into a constant.** `B = DFT₁₆(b)/16` is a
   compile-time table, so only **two** length-16 transforms run per 17-point DFT, not
   three. FFTW does exactly this — `dft/rader.c` carries a separate `cld_omega` plan used
   only at plan/awake time to build `omega` [FFTW-src].
2. **Add `x[0]` into the DC bin of the convolution before the inverse transform**, rather
   than into all p−1 outputs after it: the constant "can be added to all the outputs by
   adding it to the DC term of the convolution prior to the inverse FFT" [Wiki-Rader].
   Saves p−1 complex additions.

### 2.4 Exact operation count for Rader-17 — from FFTW's own accounting

`dft/rader.c` lines 276–279, verbatim:

```c
X(ops_add)(&cld1->ops, &cld2->ops, &pln->super.super.ops);
pln->super.super.ops.other += (n - 1) * (4 * 2 + 6) + 6;
pln->super.super.ops.add   += (n - 1) * 2 + 4;
pln->super.super.ops.mul   += (n - 1) * 4;
```

`cld1`/`cld2` are the forward and inverse length-(n−1) DFTs. FFTW's scalar size-16 codelet
header states its own cost [FFTW-src `dft/scalar/codelets/n1_16.c`]:

> "This function contains 144 FP additions, 40 FP multiplications, (or, 104 additions,
> 0 multiplications, 40 fused multiply/add), 50 stack variables, 3 constants, and 64 memory
> accesses"

Assembling for n = 17 `[mine, on FFTW's numbers]`:

| term | adds | muls | FMAs | flops | FP instrs |
|---|---|---|---|---|---|
| 2 × length-16 DFT | 2×104 | 0 | 2×40 | 368 | 288 |
| pointwise `A·B` + DC fix | 16·2+4 = 36 | 16·4 = 64 | 0 | 100 | 100 |
| **Rader-17 total** | **244** | **64** | **80** | **468** | **388** |

**468 real flops / 388 FP instructions for a 17-point complex DFT** = 27.5 flops/point.

Reference table, all from FFTW's own plan accounting via
`bench -opatient -onosimd -v2 -s ocf<n>` (so "what actually compiles and runs", not a
header I might have misread; `flops = add + mul + 2·fma`, `instrs = add + mul + fma`):

| n | add | mul | fma | flops | flops/pt | instrs | instrs/pt |
|---|---|---|---|---|---|---|---|
| 2 | 4 | 0 | 0 | 4 | 2.00 | 4 | 2.00 |
| 3 | 6 | 0 | 6 | 18 | 6.00 | 12 | 4.00 |
| 4 | 16 | 0 | 0 | 16 | 4.00 | 16 | 4.00 |
| 6 | 24 | 0 | 12 | 48 | 8.00 | 36 | 6.00 |
| 8 | 44 | 0 | 8 | 60 | 7.50 | 52 | 6.50 |
| 9 | 24 | 0 | 56 | 136 | 15.11 | 80 | 8.89 |
| 12 | 72 | 0 | 24 | 120 | 10.00 | 96 | 8.00 |
| 16 | 104 | 0 | 40 | 184 | 11.50 | 144 | 9.00 |
| **17** (Rader, derived) | 244 | 64 | 80 | **468** | **27.5** | **388** | **22.8** |
| **17** (FFTW's actual choice) | 80 | 0 | 256 | **592** | **34.8** | **336** | **19.8** |
| 18 | 108 | 24 | 96 | 324 | 18.00 | 228 | 12.67 |
| 32 | 236 | 0 | 136 | 508 | 15.88 | 372 | 11.63 |
| 36 | 288 | 60 | 204 | 756 | 21.00 | 552 | 15.33 |

Sanity check on this accounting model: 3D 16³ scalar, FFTW reports
`79872 add, 0 mul, 30720 fma`; the row-column model predicts `3 · 16² · (104,0,40)` =
`79872 add, 30720 fma`. **Exact.** Same for 6³ (`3·36·(24,0,12)` = `2592 add, 1296 fma`,
FFTW reports exactly that) and 36³ (`3·36²·(288,60,204)`, FFTW reports
`1119744 add, 233280 mul, 793152 fma` — exact). The model is trustworthy.

### 2.5 The dense alternative: exploit the conjugate symmetry instead of FFT-ing

There is a second, quite different way to do the Rader convolution: evaluate it densely but
halve the work using the fact that `g^{q+8} ≡ −g^q (mod 17)`, so
`W^{g^{-(q+8)}} = conj(W^{g^{-q}})` and outputs pair up.

That is exactly what FFTW's `dft/generic.c` does. `hartley()` forms the sum/difference
pairs `x[i] ± x[n−i]`; `cdot()` then emits two conjugate-paired outputs at once, 4 FMAs per
inner step [FFTW-src `dft/generic.c` lines 33–89]:

```c
for (i = 1; i + i < n; ++i) {
     rr += x[0] * w[0];   ir += x[1] * w[0];
     ri += x[2] * w[1];   ii += x[3] * w[1];
     x += 4; w += 2;
}
*or0 = rr + ii;  *oi0 = ir - ri;
*or1 = rr - ii;  *oi1 = ir + ri;
```

FFTW's own reported cost at n=17:

```
$ ./bench -oexhaustive -v2 -s ocf17
planner time: 1.3118 s
(dft-generic-17)
flops: 80 add, 0 mul, 256 fma
estimated cost: 592.000000
```

The count checks out exactly against the code `[mine]`: 8 conjugate pairs × 8 inner steps ×
4 FMA = **256 FMA**; `hartley` gives 8·4 pairwise + 16 accumulate = 48 adds, `cdot`'s four
final adds × 8 calls = 32, total **80 adds**.

### 2.6 Where each variant stops paying

* **Rader-via-FFT** stops paying when `p−1` has a large prime factor: pad to
  `≥ 2(p−1)−1` and round up to a smooth length [Wiki-Rader], or switch to Bluestein (§4).
  **Irrelevant at p=17.**
* **The dense form** is Θ(p²) and stops paying as p grows. VkFFT's default ceiling for
  direct-multiplication Rader is p = 17 on non-NVIDIA/AMD hardware and p = 89 on
  NVIDIA/AMD [VkFFT-src `vkFFT_InitializeApp.h` lines 1261–1272]. **At p=17 you are inside
  it.**

VkFFT documents the boundary explicitly, and puts it at exactly 17
[VkFFT-src `vkFFT/vkFFT_Structs/vkFFT_Structs.h` lines 254–258, verbatim]:

> `fixMinRaderPrimeMult` — "start direct multiplication Rader's algorithm for radix primes
> from this number. This means that VkFFT will inline custom Rader kernels if sequence is
> divisible by these primes. **Default is 17, as VkFFT has kernels for 2-13.**"
>
> `fixMinRaderPrimeFFT` — "start FFT convolution version of Rader for radix primes from
> this number. **Better than direct multiplication version for almost all primes (except
> small ones, like 17-23 on some GPUs).** Must be bigger or equal to fixMinRaderPrimeMult.
> Default 29 on AMD and 17 on other GPUs."

### 2.7 The metric trap: flops are the wrong currency on FMA hardware

This is the most useful thing in this section, so it gets its own heading.

| 17-point kernel | flops | FP instructions |
|---|---|---|
| Rader-17 (two length-16 FFTs) | **468** | 388 |
| dense conjugate-symmetric (`generic-17`) | 592 | **336** |

Rader wins on flops by 21%; the dense form wins on **instruction count by 13%**. On Haswell
(and every later x86) a 256-bit FMA and a 256-bit add have the same throughput and issue on
the same ports, so *instructions*, not flops, set the floor. That is why FFTW's
measurement-driven planner selects `dft-generic-17` over its own Rader plan for n=17 — and
keeps selecting it under `FFTW_EXHAUSTIVE` after 1.3 s of measured search.

Notice that FFTW's codelet headers report both bases for exactly this reason — `n1_16` is
"144 additions, 40 multiplications" *or* "104 additions, 0 multiplications, 40 fused
multiply/add". The second form is what executes.

Corollary for §3: any Winograd-style transformation that removes one multiplication at the
cost of two additions is a **net loss** on this machine.

Escalating to 3D, row-column `[mine, from FFTW's own per-kernel counts]`:

| 17³ kernel choice | flops/point | instrs/point |
|---|---|---|
| `generic-17` (what FFTW does) | 104.5 | 59.3 |
| Rader-17 | 82.6 (−21%) | 68.5 (**+15%**) |
| for calibration, 16³ | 34.5 | 27.0 |

So **Rader is not the lever at L=17.** It is worth having, and worth benchmarking, but the
2–3× is in the vectorization.

---

## 3. Winograd's FFT (WFTA) and nested small-n modules

### 3.1 What it is

Winograd's construction [Winograd76, Winograd78, BIB-ONLY] applies the CRT twice: once in
the integer domain (Good's mapping, §5) to turn a length `N₁N₂` DFT into a 2-D `N₁ × N₂`
DFT, and once in the polynomial domain (cyclotomic factorization of `z^{p−1} − 1`) to do the
Rader cyclic convolution with the minimum number of multiplications. Every short-n module
factors as

```
X = C · D · B · x       with D diagonal (all multiplications), B and C addition-only
```

and the diagonals of two coprime modules are then *nested* [DV90 §5.3, DSPH-7 §7.5.3]:

```
M_{N1·N2} = M_{N1} · M_{N2}                        (multiplications,  eq. 58)
A_{N1·N2} = N1·A_{N2} + M_{N2}·A'_{N1}             (additions,        eq. 59)
```

Duhamel & Vetterli prove `M_WFTA < M_PFA` whenever optimal short modules are used, and note
the structural cost: `A_WFTA ≥ A_PFA` always, "since `M₂ ≥ N₂`". They also record the useful
upper bound for CRT-derived short modules:

> "These methods always provide a number of multiplications lower than twice the length of
> the DFT: `M_N < 2N`." — [DV90] §5.1.4

### 3.2 The quantified trade-off (real numbers, not "much faster")

[DV90] Tables 1 and 2, "number of non-trivial real multiplications / real additions for
various FFTs on complex data":

| N | PFA mults | Winograd mults | PFA adds | Winograd adds |
|---|---|---|---|---|
| 30 | 100 | 68 | 384 | 384 |
| 60 | 200 | 136 | 888 | 888 |
| 120 | 460 | 276 | 2076 | 2076 |
| 240 | 1100 | 632 | 4812 | 5016 |
| 504 | 2524 | 1572 | 13388 | 14540 |
| 1008 | 5804 | 3548 | 29548 | 34668 |
| 2520 | 17660 | 9492 | 84076 | 99628 |

At N=30 the WFTA is a free −32% on multiplications with identical additions. By N=1008 it is
−39% mults for +17% adds; by N=2520, +18% adds. The 2-D tables are blunter — for 1024×1024,
45.30 additions per output point for the WFTA versus 23.88 for row-column split-radix, i.e.
"a huge number of additions (twice the number required for the other algorithms for N =
1024)" [DSPH-7 §7.9.5, Tables 7.4/7.5].

### 3.3 Why you should not build a WFTA here

1. **Multiplications are not the scarce resource.** Frigo & Johnson say it outright:
   > "FFTW does not employ the Winograd FFT [30], which minimizes the number of
   > multiplications at the expense of a large number of additions. (This tradeoff is not
   > beneficial on current processors that have specialized hardware multipliers.)"
   > — [FJ05] §II

   With FMA the argument is stronger still — see §2.7.

2. **Memory.** The nesting is what makes it expensive:
   > "WFTA is not performed in-place, and since all multiplications are nested, it requires
   > the availability of a number of memory locations equal to the number of multiplications
   > involved in the algorithms." — [DSPH-7] §7.9.3

   For a register-resident 17- or 36-point kernel, working storage proportional to the
   *multiplication count* is exactly the wrong shape.

3. **It was tried and it lost.** Burrus, surveying the field:
   > "A program that implements the nested Winograd Fourier transform algorithm (WFTA) is
   > given in [8] but it has not proven as fast or as versatile as the PFA."
   > — [Burrus-notes]

**What to steal from Winograd anyway:** the short modules themselves. The cyclotomic
factorization `z^{16} − 1 = Φ₁Φ₂Φ₄Φ₈Φ₁₆` (degrees 1,1,2,4,8) is the basis of a
minimum-multiplication length-16 cyclic convolution, i.e. of the best possible Rader-17
inner loop. Important structural note: 16 = 2⁴ is a **prime power**, so Agarwal–Cooley
nesting of coprime convolutions is *unavailable*. The routes are (a) FFT-based, (b) Winograd
cyclotomic, (c) Nussbaumer split-nesting, which

> "combines the structures of the Winograd and Agarwal-Cooley methods … for medium lengths,
> split-nesting can be more efficient than the Winograd convolution algorithm, even though
> it does not achieve the minimum number of multiplications" — [SB99-conv] §8.4.5

and which Duhamel & Vetterli credit with "the least known number of operations
(multiplications + additions)" [DSPH-7 §7.5.3]. Nussbaumer's book [Nussbaumer82, BIB-ONLY]
holds the standard tables. Selesnick & Burrus also delimit the useful range:

> "For the cyclic convolution of short sequences (n ≤ 10) and medium length sequences
> (n ≤ 100), special algorithms are available. For short lengths, algorithms that require
> the minimum number of multiplications possible have been developed by Winograd. However,
> for longer lengths Winograd's algorithms, based on his theory of multiplicative
> complexity, require a large number of additions and become cumbersome to implement."
> — [SB99-conv] §8.4

n=16 sits comfortably inside the "short" regime, so a minimum-multiplication length-16
convolution does exist in the literature. Whether it beats two length-16 FFTs on this
hardware is a different question — see §8 note 6.

### 3.4 Published hand-derived 17-point modules — what exists, and the gap

The canonical source for a hand-derived large prime module:

> "An efficient and practical development of Winograd's ideas has given a design method that
> does not require the rather difficult Chinese remainder theorem [18, 59] for short prime
> length FFT's. **These ideas have been used to design modules of length 11, 13, 17, 19, and
> 25** [60]." — [Burrus-notes]

with `[60] = H. W. Johnson and C. S. Burrus, "Large DFT modules: N = 11, 13, 17, 19, and
25", Tech. Rep. 8105, Dept. of Electrical Engineering, Rice University, Houston, TX,
December 1981` — the identical entry appears as ref 37 of [DSPH-7]. The follow-on line is
Selesnick & Burrus's automatic generator:

> "A method which automatically generates near-optimal prime length Winograd based programs
> has been given in [59, 86, 87, 88, 89]. This gives the same efficiency for shorter lengths
> (i.e. N ≤ 19) and new algorithms for much longer lengths and with well-structured
> algorithms." — [Burrus-notes]

**Gap, stated plainly: I could not obtain the exact multiplication/addition counts for a
full 17-point Winograd module.** TR 8105 is not online in fetchable form; the IEEE and AMS
originals returned 403; the LibreTexts mirror of Burrus's *Fast Fourier Transforms* (which
does contain the short-DFT tables) was unreachable from this session. A web search surfaced
a snippet claiming "the 17-point DFT … (with polynomial product modulo P₁₆(Z) = Z⁸+1) is
computed using 27 multiplications and 57 additions", attributed to LibreTexts §6.5. **I
could not fetch the page, and the phrasing strongly suggests the figure covers only the
`Φ₁₆` polynomial-product sub-step rather than the whole DFT. Do not use that number.**
`[UNVERIFIED — could not fetch]`

What you *can* rely on: the 468-flop/388-instruction Rader-17 count (§2.4) and the
592-flop/336-instruction dense count (§2.5), both derived from FFTW's own op accounting and
both re-checkable against the source sitting in `ext/src/fftw-3.3.10/`.

### 3.5 One lower bound worth knowing

For power-of-two lengths, Winograd's *realizable* multiplicative-complexity bound is
[DV90 §6.1, eq. 62]:

```
mu[ DFT_{2^n} ] = 2^{n+1} - 2n^2 + 4n - 8      (non-trivial complex multiplications)
```

> "This means that there will never exist any algorithm computing a length 2ⁿ DFT with a
> lower number of non-trivial complex multiplications than the one in (62) … Unfortunately,
> it is of no practical use for lengths greater than 64 (it involves much too many
> additions)." — [DV90]

Evaluated: N=8 → 2 complex mults; **N=16 → 8 complex mults** `[mine, arithmetic on eq. 62]`.
FFTW's `n1_16` uses 40 real multiplications ≈ 10 complex-mult-equivalents, within ~25% of
the bound. There is very little multiplication left to squeeze out of the length-16 kernel
inside Rader-17.

---

## 4. Bluestein / chirp-z — and why it is the wrong tool at p = 17

### 4.1 The method

Bluestein's identity `nk = [n² + k² − (n−k)²]/2` turns *any*-length DFT into a **linear**
convolution, embedded in a cyclic convolution of any convenient length ≥ 2N−1 and done by
FFTs [Bluestein68/70, RSR69, BIB-ONLY]. Unlike Rader it does not care that N is prime; it
cares only that you can pick a smooth padded length. Cost is (up to) three transforms of the
padded length plus O(N) chirp multiplications.

FFTW implements it and the source pins the padding rule down exactly
[FFTW-src `dft/bluestein.c`]:

```c
static INT choose_transform_size(INT minsz)
{
     while (!X(factors_into_small_primes)(minsz))
          ++minsz;
     return minsz;
}
...
nb = choose_transform_size(2 * n - 1);
```

with `factors_into_small_primes` allowing only `{2, 3, 5}` [FFTW-src `kernel/primes.c`:210].

### 4.2 What that costs at n = 17 — with a nice coincidence

`2·17 − 1 = 33`. 33 = 3·11 (11 not allowed), 34 = 2·17 (no), 35 = 5·7 (no),
**36 = 2²·3² ✓**. So FFTW's Bluestein for a 17-point DFT would run on a **length-36
convolution** — one of the very sizes in this project.

FFTW's cost accounting [`dft/bluestein.c` lines 226–229]:

```c
X(ops_add)(&cldf->ops, &cldf->ops, &pln->super.super.ops);   /* 2 x size-nb DFT */
pln->super.super.ops.add += 4 * n + 2 * nb;
pln->super.super.ops.mul += 8 * n + 4 * nb;
```

FFTW's own scalar count for a 1-D size-36 DFT is `288 add, 60 mul, 204 fma` = **756 flops /
552 instructions** (stable over 5 planner runs). Therefore `[mine, on FFTW's numbers]`:

| term | flops | FP instrs |
|---|---|---|
| 2 × length-36 DFT | 1512 | 1104 |
| chirp adds `4·17 + 2·36 = 140` | 140 | 140 |
| chirp muls `8·17 + 4·36 = 280` | 280 | 280 |
| **Bluestein-17 total** | **1932** (114 flops/pt) | **1524** (89.6 instrs/pt) |

**Bluestein-17 is 4.1× Rader-17 (468) and 3.3× the dense form (592) in flops, 4.5× the dense
form in instructions.** It is the wrong algorithm here by a wide, quantified margin, and
FFTW agrees at plan time: it never selects a Bluestein plan for n=17 under `FFTW_PATIENT` or
`FFTW_EXHAUSTIVE`.

### 4.3 When Bluestein *does* win

* When `p − 1` has a large prime factor, so the Rader convolution is itself awkward and the
  recursion nests (Sophie Germain / Cunningham-chain primes) [Wiki-Rader].
* When p is large enough that a hand-tuned Rader kernel will not fit in fast storage. VkFFT
  quantifies this: "switch to Bluestein's algorithm for radix primes from this number.
  **Default is 16384**, which is bigger than most current GPU's shared memory"
  [VkFFT-src `vkFFT_Structs.h`:258]. Its README states the same policy in prose: "Rader's
  FFT algorithm for primes from 17 up to max shared memory length (~10000) … Bluestein's
  FFT algorithm for all other sequences."

For L=17: every well-tuned library uses Rader (or a dense equivalent), not Bluestein. Do
the same.

---

## 5. Good–Thomas / prime-factor algorithm — the L=6 and L=36 story

### 5.1 The construction and the one thing it buys

Good's 1958 "interaction algorithm" and Thomas's 1963 reformulation [Good58, Thomas63,
BIB-ONLY] re-index a length `N = N₁N₂` DFT as a genuine 2-D `N₁ × N₂` DFT **iff
`gcd(N₁,N₂) = 1`**:

> "The prime-factor algorithm (PFA), also called the Good–Thomas algorithm (1958/1963), is a
> fast Fourier transform (FFT) algorithm that re-expresses the discrete Fourier transform
> (DFT) of a size N = N₁N₂ as a two-dimensional N₁ × N₂ DFT, but *only* for the case where
> N₁ and N₂ are relatively prime." — [Wiki-PFA]

The payoff is the whole point:

> mixed-radix Cooley–Tukey "requires extra multiplications by roots of unity called twiddle
> factors, in addition to the smaller transforms", whereas PFA achieves the decomposition
> without them; the re-indexing is "pure reindexing without actual arithmetic" — [Wiki-PFA]

Costs are then **purely additive per output point** [DV90 §5.2, eqs. 55–56]:

```
M_{N1 N2} = N1·M_{N2} + N2·M_{N1}                             (total multiplications)
A_{N1 N2} = N1·A_{N2} + N2·A_{N1}                             (total additions)
m_{N1 N2 N3 N4} = m_{N1} + m_{N2} + m_{N3} + m_{N4}           (per output point)
```

The price is the index permutation — "more complicated re-indexing of the data" [Wiki-PFA].
Two standard maps exist, the CRT map and the "Ruritanian" map, and [Wiki-PFA] notes both are
in use for n = 6.

### 5.2 The index tables you need

Using the Ruritanian input map `n = (N₂·n₁ + N₁·n₂) mod N` and the CRT output map
`k = (N₂·[N₂⁻¹]_{N₁}·k₁ + N₁·[N₁⁻¹]_{N₂}·k₂) mod N`
`[mine; verified numerically against numpy.fft to 4.5e-15]`:

**L = 6 = 2 × 3** (`[3⁻¹]₂ = 1`, `[2⁻¹]₃ = 2`)

```
input  n(n1,n2):   n1=0: 0 2 4        output k(k1,k2):   k1=0: 0 4 2
                   n1=1: 3 5 1                           k1=1: 3 1 5
```

**L = 36 = 4 × 9** (`[9⁻¹]₄ = 1`, `[4⁻¹]₉ = 7`)

```
input  n(n1,n2):                        output k(k1,k2):
  n1=0:  0  4  8 12 16 20 24 28 32        k1=0:  0 28 20 12  4 32 24 16  8
  n1=1:  9 13 17 21 25 29 33  1  5        k1=1:  9  1 29 21 13  5 33 25 17
  n1=2: 18 22 26 30 34  2  6 10 14        k1=2: 18 10  2 30 22 14  6 34 26
  n1=3: 27 31 35  3  7 11 15 19 23        k1=3: 27 19 11  3 31 23 15  7 35
```

Both are constants — bake them in as `static const int[]`.

### 5.3 L = 6: PFA is already what FFTW does, and it is exactly optimal

Apply eq. 55 with FFTW's own per-size counts: 3 transforms of length 2 and 2 of length 3.

| basis | PFA-6 = 3·(n=2) + 2·(n=3) | FFTW's `n1_6` codelet |
|---|---|---|
| flops | 3·4 + 2·18 = **48** | **48** |
| FP instrs | 3·4 + 2·12 = **36** | **36** |

**Exact match in both metrics.** This is not luck: `genfft`'s DAG builder knows the PFA —

> "The dag is produced according to well-known DFT algorithms: Cooley-Tukey (Eq. (2)),
> prime-factor [27, page 619], split-radix [16], and Rader [28]." — [FJ05] §VI

— and 6 = 2·3 is coprime, so FFTW's 6-point codelet *is* (equivalent to) the Good–Thomas
2×3. **Conclusion: there are no flops left to win at L=6.**

Where the L=6 headroom actually is `[mine]`. FFTW's 3D 6³ scalar count is
`2592 add, 1296 fma` = 5184 flops for 216 points (24.0 flops/pt, 18.0 instrs/pt), and the
AVX2 plan runs in 671 ns:

```
$ ./bench -opatient -v2 -s ocf6x6x6
(dft-rank>=2/0
  (dft-direct-6-x36 "n1fv_6_avx")
  (dft-rank>=2/1 (dft-vrank>=1-x6/-1 (dft-direct-6-x6 "n1fv_6_avx"))
                 (dft-direct-6-x36 "n1fv_6_avx")))
```

5184 flops / 671 ns = **7.73 flops/ns**, against a 2.5 GHz AVX2 FMA peak of 16 flops/cycle
= 40 flops/ns — i.e. **~19% of peak**. Note the codelet FFTW chose is `n1fv_6_avx`, the
**non-FMA** 256-bit variant. So the win at L=6 is: batch 4 (AVX2) or 8 (AVX-512) volumes into
full-width lanes with a struct-of-arrays layout, and use FMA. 216 complex doubles = 3.4 kB, so
several volumes plus scratch stay L1-resident (32 kB L1d on this part).

### 5.4 L = 36: FFTW does not use Good–Thomas, and the plan it picks is ugly

FFTW has PFA only inside `genfft`, for *hard-coded* codelet sizes — [FJ05] §II: "FFTW
implements the first two in its codelet generator for hard-coded n … and the latter two for
general prime n" — and its shipped codelet set is

* scalar `n1_*`: `{2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,20,25,32,64}`
* AVX2 `n1fv_*`: `{2,…,16,20,25,32,64,128}`

**No 36, and no 17** [FFTW-src `dft/scalar/codelets/codlist.c`, `dft/simd/*/`]. So 36 is
planned as mixed-radix Cooley–Tukey with twiddles, and in 3D the plan is genuinely messy —
three different radices and an extra copy pass:

```
$ ./bench -opatient -v2 -s ocf36x36x36
(dft-rank>=2/1
  (dft-vrank>=1-x36/1
    (dft-rank>=2/1
      (dft-ct-dit/3 (dftw-direct-3/8-x36 "t1fv_3_avx2")
                    (dft-vrank>=1-x3/-1 (dft-direct-12-x36 "n1fv_12_avx2")))
      (dft-ct-dit/6 (dftw-direct-6/20-x36 "t1fuv_6_avx2")
                    (dft-indirect-before ... ))))
  ...)
```

The 1-D scalar cost FFTW charges itself is `288 add, 60 mul, 204 fma` = **756 flops / 552
instructions** (21.0 flops/pt, 15.3 instrs/pt), stable across 5 planner runs. Compare
Good–Thomas `[mine, eq. 55 on FFTW's own per-size counts]`:

| decomposition of 36 | arithmetic | flops | instrs |
|---|---|---|---|
| **PFA 4×9**, no twiddles | 9 × (n=4: 16 fl / 16 in) + 4 × (n=9: 136 fl / 80 in) | **688** | **464** |
| FFTW's mixed-radix CT | — (FFTW's own accounting) | 756 | 552 |
| **saving** | | **−9.0%** | **−15.9%** |

Modest on flops, better on instructions — and the qualitative wins are larger than the
numbers:

* **The entire twiddle stage disappears.** No `t1fv_*` codelets, no twiddle table loads, no
  extra dependency chain, and none of the `dft-indirect-before` copy that FFTW's plan drags
  in.
* **Only two kernel shapes**, 4-point and 9-point, both tiny and fully unrollable, instead
  of the 3/6/12 mix FFTW ends up with.
* In 3D: `3·36²·688 = 2,674,944` flops (57.3/pt) vs FFTW's `3·36²·756 = 2,939,328` (63.0/pt).

Where the remaining cost sits: `n1_9` at 136 flops / 80 instructions (15.1 flops/pt) is ~79%
of PFA-36's flops. 9 = 3² is a **prime power**, so Good–Thomas cannot help inside it;
`genfft` builds it as Cooley–Tukey 3×3. If you want more, a hand-derived 9-point module is
where to look ([Johnson-Burrus81], [Temperton88]) — but re-read §2.7 and §3.3 first: on
AVX2+FMA, trading multiplies for adds loses.

### 5.5 Why the PFA permutation is free in *this* problem

The standard objection to PFA is the index shuffle. In a batched 3D row-column transform it
costs essentially nothing:

1. Each 1-D pass along a 3D axis is **already** a strided gather/scatter (stride 1, L, or L²
   depending on the axis). Composing the constant Good–Thomas permutation with that stride
   changes the *index arithmetic*, not the number of memory operations. Precompute
   `static const int idx_in[36], idx_out[36]` as byte offsets and the loop looks identical.
2. Duhamel & Vetterli make the same point about the parent structure: a 2-D/3-D DFT is
   "performed in two steps, as was explained for the PFA … Row-column algorithms are very
   easily implemented and only require efficient 1-D FFTs … together with a matrix
   transposition algorithm" [DSPH-7 §7.9.1]. The permutation rides along with the
   transposition you were doing anyway.
3. Going further, treat the whole 36³ volume as a 6-dimensional `4×9×4×9×4×9` DFT: the
   permutations become pure address remapping at the outermost loop level, and every inner
   kernel is a clean 4-point or 9-point module. This is the "full multi-dimensional
   factorization when N is highly composite" that Good's mapping provides [DV90 §5.1.1].

### 5.6 When PFA stops paying

* **Prime powers.** `gcd` must be 1. PFA is "useless for power-of-two sizes" [Wiki-PFA] — so
  nothing for L=8, and nothing *inside* the 4 and 9 of L=36.
* **Both coprime factors need good modules.** [DV90 §5.2] notes the constraint "implies the
  availability of a whole set of efficient small DFTs (Nᵢ = 2, 3, 4, 5, 7, 8, 16 is already
  sufficient to provide a dense set of feasible lengths)". You need 4 and 9; 9 is the one to
  invest in.
* **At larger N, cache blocking dominates the 9% flop saving.** 36³ = 46656 points = 746 kB
  of complex double — far outside the 32 kB L1 and the 256 kB L2 per core on this Haswell.
  How many volumes you keep resident and whether you fuse the y/z passes will matter at
  least as much as the arithmetic. That belongs to the cache-blocking section, not this one.

---

## 6. How real libraries actually handle 17 — and the measured penalty

### 6.1 FFTW: no 17-codelet, no SIMD, scalar Θ(n²)

The manual is explicit about which sizes are fast, and 17 is not among them:

> "FFTW is best at handling sizes of the form 2^a 3^b 5^c 7^d 11^e 13^f, where e+f is either
> 0 or 1, and the other exponents are arbitrary. Other sizes are computed by means of a
> slow, general-purpose algorithm (which nevertheless retains O(n log n) performance even
> for prime sizes)."
> — [FFTW-src `doc/reference.texi` lines 376–387; the paragraph appears three times]

Note the mismatch between doc and behaviour: the manual promises *O(n log n)* for primes
(the Rader/Bluestein path), but the measuring planner actually selects the **Θ(n²)**
`dft-generic-17`, because on this machine it is faster (§2.7). The paper describes both
plan classes:

> "Generic plans implement a naive Θ(n²) algorithm to solve one-dimensional DFTs. Similarly,
> Rader plans implement the algorithm from [28] to compute one-dimensional DFTs of prime
> size in O(n log n) time" — [FJ05] §IV-B

The 3D plan for 17³ is generic in all three dimensions:

```
$ ./bench -opatient -v2 -s ocf17x17x17
(dft-rank>=2/0
  (dft-vrank>=1-x289/1 (dft-generic-17))
  (dft-rank>=2/1
    (dft-vrank>=1-x17/-1 (dft-vrank>=1-x17/1 (dft-generic-17)))
    (dft-vrank>=1-x289/1 (dft-generic-17))))
flops: 69360 add, 0 mul, 221952 fma
```

`69360 + 2·221952 = 513,264` flops = **104.5 flops/point**, exactly `3 · 17² · 592`, i.e.
pure row-column with the generic-17 kernel. For 16³ the same model gives
`3 · 16² · 184 = 141,312` = **34.5 flops/point**. So 17³ burns **3.03× the flops per point**
of 16³ (2.20× the instructions).

**The decisive fact: there is no SIMD codelet for 17.** FFTW's AVX2 notw-codelet set stops
at 16 (plus 20, 25, 32, 64, 128), and `generic.c` is plain scalar C — and the reported flop
count for the 17³ plan is *identical* with and without `-onosimd`, proving no vector codelet
is involved. Measured, three runs each:

| L³ | AVX2 | scalar | SIMD speedup |
|---|---|---|---|
| 6³ | 671 / 675 / 674 ns | 1.62 µs ×3 | 2.41× |
| 8³ | 1.51 µs ×3 | 4.08 / 4.06 / 3.22 µs | 2.13–2.70× |
| 16³ | 19.88 / 16.13 / 16.11 µs | 51.80 / 51.69 / 51.73 µs | **3.21×** |
| **17³** | 134.14 / 134.77 / 103.06 µs | 102.30 / 111.66 / 103.09 µs | **1.00×** |
| 18³ | 45.52 / 48.20 / 46.21 µs | 89.04 / 90.76 / 94.25 µs | 1.96× |
| 36³ | 374.53 / 364.44 / 353.97 µs | 737.06 / 734.31 / 737.00 µs | 2.07× |

L=17 gets **no** AVX2 benefit whatsoever. Everything else gets 2–3.2×. That gap is the
single largest recoverable inefficiency among the four target sizes, and it is recoverable by
the most mechanical means available: **vectorize across the batch** (spin–colour components ×
time slices), which is exactly the workload here.

### 6.2 Measured penalty, FFTW and MKL

**Setup.** Intel Xeon E5-2680 v3 @ 2.50 GHz (Haswell; CPU flags `fma avx avx2`, **no
AVX-512**), 32 kB L1d + 256 kB L2 per core, 30 MB L3 per socket, 1 thread.
FFTW 3.3.10 double built from `ext/src/fftw-3.3.10`, `FFTW_PATIENT`, out-of-place, timed with
the tree's own `tests/bench`. MKL 2022.0.2 `libmkl_sequential`, `DFTI_NOT_INPLACE`, 3-D
`DFTI_COMPLEX/DFTI_DOUBLE`. gcc 11.4.0 `-O2 -march=native`. Benchmark sources I wrote:
`<scratchpad>/mklbench.c`, `<scratchpad>/batch.c`.

Best of repeated runs, converted to ns per point:

| L | points | FFTW single volume | FFTW batched (~32 MB/buffer) | MKL single volume |
|---|---|---|---|---|
| 6 | 216 | **3.11** | 4.94 | 2.08 |
| 8 | 512 | **2.95** | 4.76 | 2.12 |
| 16 | 4096 | **3.93** | 6.22 | 3.86 |
| **17** | **4913** | **20.98** | **22.32** | **20.86** |
| 18 | 5832 | **7.81** | 11.97 | 7.17 |
| 32 | 32768 | 8.20 | — | 5.21 |
| 36 | 46656 | **7.59** | 9.36 | 8.23 |

Derived penalties:

* **17³ vs 16³:** FFTW **5.34×** per point (single volume), 3.59× (batched). MKL **5.40×**.
* **17³ vs 18³** (nearest smooth size, 2·3²): FFTW **2.69×**, MKL **2.91×**.
* **17³ vs 16³ with SIMD disabled: only 1.65×** (20.82 vs 12.62 ns/pt) — the arithmetic
  penalty alone.
* **MKL and FFTW agree to within ~1% at 17³.** Neither library has a good 17. This is an
  algorithmic/implementation gap in the ecosystem, not one library's bug.
* For calibration, Wikipedia's folklore figure for Rader — "typically takes 3–10 times as
  long" as nearby composite sizes [Wiki-Rader] — brackets the measured 3.6–5.4×.

**Realistic headroom at L=17** `[mine]`. FFTW's generic-17 path needs 59.3 FP
instructions/point versus 27.0 for 16³, a ratio of 2.20×. If a hand-written kernel achieved
the *same instruction throughput* that FFTW's vectorized 16³ path achieves (3.93 ns/pt at
27.0 instrs/pt), L=17 would land at ≈ **8.6 ns/point** — i.e. **~2.4× faster than both FFTW
and MKL**. That is the conservative target. Pushing further means beating FFTW's own
throughput on the 16³ path, which is a separate fight.

### 6.3 MKL on non-smooth sizes

MKL's DFTI accepted 17³ without complaint and returned correct results, at 20.86 ns/point
versus 3.86 for 16³ (§6.2). Intel does not document the algorithm used, and I found no
fetchable Intel statement about prime-size handling, so I report only the measurement.
`[gap: no primary source for MKL's internal prime-size strategy]`

### 6.4 VkFFT: the clearest published policy

VkFFT v1.3.4 (commit `066a17c`, in `ext/src/VkFFT`) is the most explicit modern source on the
Rader/Bluestein decision, and worth reading even for a CPU target because the constraints
(register / fast-storage budget, no library calls in the kernel) are analogous. From the
README, verbatim:

> "Radix-2/3/4/5/7/8/11/13 FFT. Sequences using radix 3, 5, 7, 11 and 13 have comparable
> performance to that of powers of 2.
> **Rader's FFT algorithm for primes from 17 up to max shared memory length (~10000).
> Inlined and done without additional memory transfers.**
> Bluestein's FFT algorithm for all other sequences."

Combined with the struct documentation quoted in §2.6 and §4.3, VkFFT's policy is:
direct-multiplication Rader from p = 17 upward, FFT-convolution Rader from p = 17
(29 on AMD double precision) upward, Bluestein only from p = 16384. Also worth heeding:

> "With FP32 twiddle factors VkFFT is slightly less precise in Bluestein's and Rader's
> algorithms. If needed, this can be solved with FP64 precomputation." — [VkFFT README]

Rader's kernel `b[q]` and Bluestein's chirp both need accurate trig. You are already in
double; consider building the 16-entry `b[]` table in extended precision at setup, since its
error feeds every output.

---

## 7. Per-size recommendation summary

**L = 6 (216 points).** Fully unrolled straight-line codelet. Use PFA 2×3 (or equivalently
FFTW's 6-point codelet structure) — it is exactly optimal in both flops (48) and FP
instructions (36), so stop optimizing arithmetic. FFTW reaches only ~19% of FMA peak here
using a *non-FMA* `n1fv_6_avx` codelet, so the win is: 4 volumes per AVX2 vector (8 per
AVX-512), struct-of-arrays so every lane is a different volume, zero cross-lane shuffles
anywhere, FMA throughout. 3.4 kB/volume keeps everything L1-resident.

**L = 8 (512 points).** Not this section's subject; PFA does not apply (2³ is a prime power).
For calibration, Winograd's realizable lower bound is 2 non-trivial complex multiplications
at N=8 [DV90 eq. 62]; FFTW's codelet is 60 flops / 52 FP instructions.

**L = 17 (4913 points).** Priority order matters here, and it is the opposite of what the
literature emphasis suggests:

1. **First: vectorize across the batch.** This is 2–3× and it is the entire reason FFTW and
   MKL both lose 5.3× to L=16. Both candidate kernels below are pure straight-line code with
   *identical* index patterns in every lane, so SIMD-across-volumes needs no shuffles at all.
2. **Kernel choice second.** Implement the dense conjugate-symmetric form (§2.5): 592 flops
   but only **336 FP instructions**, no buffers, no permutation, 256 of the 336 are FMAs. It
   is what FFTW's measuring planner picks and where VkFFT's thresholds put p=17.
3. **Also implement Rader-17** (§2.4: 468 flops, 388 instructions, `g = 3`, folded `B/16`,
   DC-fold trick). It wins on flops and loses on instructions; measure, don't assume. It also
   has a much shorter dependency chain, which may matter more than either count.
4. **Do not use Bluestein** (1932 flops / 1524 instructions, §4.2).
5. 78.6 kB per volume means one volume does not fit L1. Plan for L2 residency (256 kB → ~3
   volumes) and vectorize *across* volumes, not within.
6. Conservative target: **~8–9 ns/point, ≈2.4× faster than FFTW and MKL** (§6.2).

**L = 36 (46656 points).** Use Good–Thomas 4×9 instead of the mixed-radix 3/6/12 tree FFTW
picks: −9% flops, −16% FP instructions, and the twiddle stage plus FFTW's
`dft-indirect-before` copy pass both vanish (§5.4). Bake in the constant index tables of §5.2
and fold them into the 3D gather/scatter so the permutation is free (§5.5). Invest in a good
9-point module — it is ~79% of the arithmetic. At 746 kB/volume, cache blocking across the
three passes matters here in a way it does not for L=6 or L=8.

---

## 8. Unsourced engineering notes (my own analysis, attributed to nobody)

These are *not* from the literature. Derived or measured here, listed so you can tell them
apart from cited results.

1. **Rader-17 = 468 flops / 388 instructions and Bluestein-17 = 1932 / 1524** are my
   arithmetic applied to FFTW's own op-count accounting (`dft/rader.c`, `dft/bluestein.c`)
   and its reported per-size plan costs. The inputs are primary-source; the sums are mine.
2. **PFA-36 = 688 flops / 464 instructions** is my application of [DV90] eq. 55 to FFTW's own
   authoritative scalar counts for n=4 and n=9. Cross-check: the same method applied to L=6
   reproduces FFTW's 6-point codelet *exactly* in both metrics (48 = 48, 36 = 36), and
   applied to 3D reproduces FFTW's reported 3D plan flops exactly for 6³, 16³ and 36³. The
   method is validated.
3. **Primitive-root and Good–Thomas permutation tables** (§2.2, §5.2) are mine, verified
   numerically against `numpy.fft` — max error 4.4e-15 (Rader-17), 3.1e-16 (GT 2×3),
   4.5e-15 (GT 4×9).
4. **The flops-vs-instructions argument (§2.7)** is mine. Haswell issues up to two 256-bit
   FMAs per cycle on the same ports the FP adds use, so a fused multiply-add and a bare add
   cost the same. Any transformation that removes one multiplication at the price of two
   additions therefore *increases* the instruction count. This is why I distrust every
   multiplication-count-only comparison in the 1970s–80s literature for this hardware, and
   why §3.3's Frigo–Johnson quote understates the case.
5. **Measurement caveats.** All timings are mine, on this node, single-threaded, in this
   session. FFTW's `--report-mflops` uses the `5n log₂ n` convention and is **not** a real
   flop rate — I used time-per-point throughout instead. FFTW's `bench -v2` flop counts for
   **SIMD** plans are reported per vector operation and are **not** comparable to scalar
   counts; every arithmetic comparison in this document therefore uses `-onosimd` or the
   scalar codelet headers. FFTW's planner is measurement-driven and its choices vary run to
   run: I saw the 1-D-36 scalar plan report 720, 756 and 904 flops on different invocations
   (756 in 5/5 of the final runs, which is the figure used). Treat all timings as ±5%; the
   17³ AVX2 figure was the noisiest (103–135 µs).
6. **On the length-16 cyclic convolution inside Rader-17.** Because 16 = 2⁴ is a prime power,
   Agarwal–Cooley nesting of coprime convolutions is structurally unavailable; the routes are
   FFT-based, Winograd-cyclotomic (`Φ₁Φ₂Φ₄Φ₈Φ₁₆`), or split-nesting. Given §3.5 (FFTW's 40
   real multiplications is within ~25% of Winograd's 8-complex-mult bound for N=16) and §2.7
   (instructions, not multiplications, are the cost), I would not expect a cyclotomic
   length-16 convolution to beat two length-16 FFTs by more than ~10% on flops, and it will
   lose on regularity and dependency depth. **Recommendation: do not build it first.**
7. **The batching structure specific to this workload.** All three axes of L=17 need the same
   17-point kernel, and the batch dimension (spin–colour × time slices) is large. A kernel
   that takes 4 (AVX2) or 8 (AVX-512) *independent* 17-point transforms in SIMD lanes has no
   cross-lane data movement at all — the primitive-root gather is the same index list in
   every lane, and the twiddle/kernel constants are lane-invariant broadcasts. This is the
   cleanest possible vectorization and is exactly what neither FFTW nor MKL does at 17. It is
   also, notably, precisely what FFTW *does* do at every other size — its 3D plans use the
   `n1fv_*` "vector-of-transforms" codelets (`dft-direct-6-x36 "n1fv_6_avx"`,
   `dft-direct-16-x256 "n1fv_16_avx"`). Copy that strategy for 17.

---

## 9. References

### Verified — URL fetched in this session

* **[FJ05]** M. Frigo and S. G. Johnson, "The Design and Implementation of FFTW3",
  *Proc. IEEE* **93**(2), 216–231 (2005). <https://www.fftw.org/fftw-paper-ieee.pdf>
  — fetched; text extracted locally. Used for: §II (PFA/Rader/Bluestein, and the explicit
  rejection of the WFTA), §IV-B (generic Θ(n²) vs Rader plans), §VI (genfft's algorithm set),
  §VII (the DHT-Rader real-data variant), and the reference list behind several `[BIB-ONLY]`
  entries below.
* **[DV90]** P. Duhamel and M. Vetterli, "Fast Fourier transforms: a tutorial review and a
  state of the art", *Signal Processing* **19**(4), 259–299 (1990).
  <https://norbertwiener.umd.edu/Research/Duhamel_Vetterli_FFT_90.pdf> — fetched.
  Used for: Rader derivation (§5.1.2), `M_N < 2N` (§5.1.4), PFA cost formulas eqs. 55–56,
  WFTA nesting eqs. 58–59 and the `M_WFTA < M_PFA` / `A_WFTA ≥ A_PFA` proof, Tables 1–3,
  Winograd's power-of-two lower bound eq. 62.
* **[DSPH-7]** P. Duhamel and M. Vetterli, "Fast Fourier Transforms: A Tutorial Review and a
  State of the Art", ch. 7 in *Digital Signal Processing Handbook*, ed. V. K. Madisetti and
  D. B. Williams, CRC Press, Boca Raton, 1999. <https://dsp-book.narod.ru/DSPMW/07.PDF>
  — fetched. The book version of [DV90] plus a multi-dimensional section: §7.9.1 row-column,
  §7.9.3 nested algorithms and the WFTA memory blow-up, §7.9.5 discussion, Tables 7.4/7.5
  (2-D op counts per output point). Its reference list is the source of several `[BIB-ONLY]`
  entries.
* **[SB99-conv]** I. W. Selesnick and C. S. Burrus, "Fast Convolution and Filtering", ch. 8
  in *Digital Signal Processing Handbook*, CRC Press, 1999.
  <https://dsp-book.narod.ru/DSPMW/08.PDF> — fetched. Used for: §8.4 (Winograd
  minimum-multiplication convolution for n ≤ 10 / n ≤ 100 and its addition penalty), §8.4.4
  Agarwal–Cooley nesting and its permutation `k → ⟨k⟩_{n1} + n1⟨k⟩_{n2}`, §8.4.5
  split-nesting.
* **[Burrus-notes]** C. S. Burrus, "Notes on the FFT", hosted at fftw.org.
  <https://www.fftw.org/burrus-notes.html> — fetched. Used for: PFA/Good–Thomas history, the
  "modules of length 11, 13, 17, 19, and 25" statement pointing at [Johnson-Burrus81], the
  WFTA-vs-PFA verdict, and full bibliographic entries for its refs 60, 61, 82, 83.
* **[Wiki-Rader]** "Rader's FFT algorithm", Wikipedia.
  <https://en.wikipedia.org/wiki/Rader%27s_FFT_algorithm> — fetched. Used for: the
  zero-padding variant, the Cunningham-chain pathological case, the DC-fold trick, the
  "3–10 times as long" rule of thumb. Tertiary source; used only where cross-checkable or
  clearly folklore.
* **[Wiki-PFA]** "Prime-factor FFT algorithm", Wikipedia.
  <https://en.wikipedia.org/wiki/Prime-factor_FFT_algorithm> — fetched. Used for: the
  coprimality requirement, CRT vs Ruritanian mappings, "pure reindexing without actual
  arithmetic", "useless for power-of-two sizes", and the Good 1958 / Thomas 1963 entries.
* **[JOS]** J. O. Smith III, *Mathematics of the Discrete Fourier Transform (DFT), with Audio
  Applications*, 2nd ed., W3K Publishing, 2007.
  <https://ccrma.stanford.edu/~jos/st/Rader_s_FFT_Algorithm_Prime.html> — fetched, but the
  page is a stub pointing at Wikipedia. Listed for completeness; nothing here rests on it.

### Verified — primary source code read on disk

* **[FFTW-src]** FFTW 3.3.10 source tree, `/home/lqcd/wdetmold/fft/ext/src/fftw-3.3.10`
  (tarball `fftw-3.3.10.tar.gz` from <https://www.fftw.org>). Files cited:
  `dft/rader.c` (Rader plan + op accounting, lines 276–279);
  `dft/generic.c` (`hartley`/`cdot`, the Θ(n²) conjugate-pair kernel, lines 33–89);
  `dft/bluestein.c` (`choose_transform_size`, `nb = 2n−1` rounding, op accounting lines
  226–229); `kernel/primes.c`:210 (`factors_into_small_primes` = {2,3,5});
  `dft/scalar/codelets/n1_*.c`, `t1_*.c` (per-codelet op counts in the headers — note each
  file contains **two** generated variants, `-fma` and plain, with different counts);
  `dft/scalar/codelets/codlist.c` and `dft/simd/*/` (which sizes have codelets at all);
  `doc/reference.texi` lines 376–387 (the `2^a 3^b 5^c 7^d 11^e 13^f` statement).
  All op counts quoted in §2.4/§4.2/§5.3/§5.4 were taken from the tree's own `tests/bench`
  harness (`-opatient -onosimd -v2`) rather than from the headers, to avoid variant confusion.
* **[VkFFT-src]** VkFFT v1.3.4, commit `066a17c`, `/home/lqcd/wdetmold/fft/ext/src/VkFFT`,
  origin <https://github.com/DTolm/VkFFT>. Files cited: `README.md` (supported radices,
  Rader/Bluestein ranges, FP32 precision note);
  `vkFFT/vkFFT/vkFFT_Structs/vkFFT_Structs.h` lines 254–258 (the `fixMin/MaxRaderPrimeMult`
  and `fixMin/MaxRaderPrimeFFT` documentation);
  `vkFFT/vkFFT/vkFFT_AppManagement/vkFFT_InitializeApp.h` lines 1257–1292 (the actual
  vendor-dependent thresholds).
* MKL 2022.0.2 (`$MKLROOT` on this node) — behaviour measured, not source-read.

### `[BIB-ONLY]` — details transcribed from the reference lists of the fetched sources above; **the primary texts were not fetched**

* **[Rader68]** C. M. Rader, "Discrete Fourier transforms when the number of data samples is
  prime", *Proc. IEEE* **56**, 1107–1108, June 1968. (Identical entry in [FJ05] ref 28 and
  [DSPH-7] ref 43. DOI `10.1109/PROC.1968.6477` per search results — DOI page not fetched.)
* **[Bluestein68]** L. I. Bluestein, "A linear filtering approach to the computation of the
  discrete Fourier transform", *Northeast Electronics Research and Eng. Meeting Record*
  **10**, 218–219, 1968. ([FJ05] ref 29.) The commonly cited journal version is *IEEE Trans.
  Audio Electroacoust.* **18**(4), 451–455, Dec. 1970 `[UNVERIFIED — could not fetch]`.
* **[RSR69]** L. R. Rabiner, R. W. Schafer and C. M. Rader, "The chirp z-transform
  algorithm", *IEEE Trans. Audio Electroacoust.* **AU-17**, 86–92, June 1969.
  ([Burrus-notes] ref 82.)
* **[Winograd76]** S. Winograd, "On computing the discrete Fourier transform", *Proc. Nat.
  Acad. Sci. USA* **73**, 1005–1006, April 1976. ([DSPH-7] ref 54.)
* **[Winograd78]** S. Winograd, "On computing the DFT", *Math. Comp.* **32**(1), 175–199,
  Jan. 1978. ([DSPH-7] ref 56; [FJ05] ref 30 gives the same volume/pages.)
* **[Winograd79]** S. Winograd, "On the multiplicative complexity of the discrete Fourier
  transform", *Adv. in Math.* **32**(2), 83–117, May 1979. ([DSPH-7] ref 57.)
* **[Good58]** I. J. Good, "The interaction algorithm and practical Fourier analysis",
  *J. Roy. Statist. Soc. Ser. B* **20**(2), 361–372, 1958; addendum **22**, 372–375, 1960.
  ([DSPH-7] ref 32; [Wiki-PFA] gives DOI `10.1111/j.2517-6161.1958.tb00300.x`.)
* **[Thomas63]** L. H. Thomas, "Using a computer to solve problems in physics", in
  *Applications of Digital Computers*, Ginn, Boston, 1963. ([Wiki-PFA].)
* **[Nussbaumer82]** H. J. Nussbaumer, *Fast Fourier Transform and Convolution Algorithms*,
  Springer, Berlin, 1982. ([DSPH-7] ref 12 — source of the split-nesting scheme and the
  standard short-convolution tables.)
* **[Johnson-Burrus81]** H. W. Johnson and C. S. Burrus, "Large DFT modules: N = 11, 13, 17,
  19, and 25", Tech. Rep. 8105, Dept. of Electrical Engineering, Rice University, Houston,
  TX, December 1981. ([Burrus-notes] ref 60; [DSPH-7] ref 37.) **This is the canonical
  hand-derived 17-point module. Not available online in any form I could fetch.**
* **[Johnson-Burrus83]** H. W. Johnson and C. S. Burrus, "The design of optimal DFT
  algorithms using dynamic programming", *IEEE Trans. Acoust. Speech Signal Process.*
  **ASSP-31**(2), 378–387, 1983. ([DSPH-7] ref 38.)
* **[Selesnick-Burrus96]** I. W. Selesnick and C. S. Burrus, "Automatic generation of prime
  length FFT programs", *IEEE Trans. Signal Processing* **44**(1), 14–24, 1996, DOI
  `10.1109/78.482008`. (Details from search results; paper not fetchable. [Burrus-notes]
  describes the method and says it "gives the same efficiency for shorter lengths
  (i.e. N ≤ 19)".)
* **[Temperton88]** C. Temperton, "A new set of minimum-add small-n rotated DFT modules",
  *J. Comput. Phys.* **75**, 190–198, 1988. ([Burrus-notes] ref 61.)
* **[Kolba-Parks77]** D. P. Kolba and T. W. Parks, "A prime factor algorithm using high-speed
  convolution", *IEEE Trans. Acoust. Speech Signal Process.* **ASSP-25**, 281–294, Aug. 1977.
  ([DSPH-7] ref 39.)
* **[Agarwal-Burrus74]** R. C. Agarwal and C. S. Burrus, "Fast one-dimensional digital
  convolution by multidimensional techniques", *IEEE Trans. Acoust. Speech Signal Process.*
  **ASSP-22**(1), 1–10, Feb. 1974. ([DSPH-7] ref 20.)
* **[Tolmachev23]** D. Tolmachev, VkFFT white paper, *IEEE Access* (2023),
  <https://ieeexplore.ieee.org/document/10036080> — **could not fetch**
  `[UNVERIFIED — could not fetch]`. The VkFFT source itself (above) is verified and is what
  this document actually cites.

### Explicit gaps

1. **No published exact operation count for a full 17-point DFT module.** TR 8105
   [Johnson-Burrus81] is the right source and is not online. The LibreTexts mirror of
   Burrus's *Fast Fourier Transforms* (which contains the short-DFT and prime-length tables)
   was unreachable — repeated `WebFetch` and `curl` attempts to `eng.libretexts.org` returned
   empty or timed out. The Rice institutional-repository copy returned HTTP 403. If someone
   has journal access, extract (a) the 17-point module's real multiply/add counts and
   (b) Nussbaumer's cyclic-convolution table entry for length 16.
2. **No primary source for MKL's prime-size algorithm.** Measured only (§6.3).
3. **Johnson & Frigo, "A modified split-radix FFT with fewer arithmetic operations"**
   (<https://www.fftw.org/newsplit.pdf>) was fetched but its PDF text would not extract, so
   its flop-count tables are unread here. Relevant to L=8, i.e. the power-of-two section.
4. **Nussbaumer's polynomial-transform approach to multi-dimensional DFTs** ([DSPH-7] refs
   74, 87, 88) has the lowest published total operation count for 2-D transforms
   (Table 7.5: 22.83 adds/output point for 1024², vs 23.88 for row-column split-radix). I did
   not pursue it: the published tables are 2-D and power-of-two, and it is a poor fit for a
   batched small-volume workload. It is nonetheless the one genuinely unexplored direction
   for 36³.
5. **No SIMD width above AVX2 was measurable here.** This node is Haswell (`fma avx avx2`,
   no `avx512f`). Every SIMD-speedup number in §6.1 is an AVX2 number; the AVX-512 nodes
   (`axxxl`, `a100l`, `a100r` per the project README) will shift the L=6/L=8 balance and
   should be re-measured there.
