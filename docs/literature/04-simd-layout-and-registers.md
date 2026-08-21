# 04 — SIMD, complex data layout, and register-level technique on x86-64

**Scope.** Everything between "I have an FFT algorithm" and "I have fast machine code":
how complex numbers sit in memory and in registers, which dimension you vectorise over,
what shuffles cost, how FMA changes both the op count and the error bound, and how
alignment and cache-set geometry bite at L = 8 and L = 36. Targets: L = 6, 8, 17, 36,
`L^3` complex double, forward, many volumes batched, single thread, AVX2 baseline with
AVX-512 on some nodes.

**Citation status.** Every reference in §11 carries a URL that was fetched during the
research session for this document, except where explicitly marked. Where a claim is my
own reasoning rather than a source's, it is either labelled inline as *(analysis)* or
lives in §10 "Unsourced engineering notes".

---

## 0. The one-paragraph verdict

For batched small transforms on x86-64 the winning layout is **split-complex with the
batch index innermost** — separate real and imaginary arrays, with the batch running
fastest in memory, so that one SIMD register holds the same transform element from `VL`
different volumes.

> *Terminology.* I call this **batch-minor** below, "minor" in the Fortran sense of
> fastest-varying. It is the same object the project brief calls a "batch-major layout" —
> the batch is the dimension you vectorise over. The defining property, whatever you call
> it, is: **consecutive addresses differ by batch index, not by transform element.**

In that layout the 3-D transform proper contains **zero shuffle instructions**, every
scalar operation of your chosen algorithm becomes exactly one vector instruction, all
three axes are the same code with a different stride, and multiplication by ±i is free
(a register rename plus a sign folded into the next add/sub). Three prices:

1. **You convert at the boundary.** Use loads + permutes, never gather. Intel measures the
   shuffle-based complex de-interleave at **4.9×** a scalar loop on Skylake (vs **2.7×**
   for hardware `VPGATHERD`) [I1, Table 15‑7], and the AVX-512 `vpermi2d` version at
   **4.8×** the `vpgatherdd` version [I1, §18.22.1].
2. **A straight-line `L`-point codelet needs `2L` live vector registers.** That is exactly
   the whole AVX2 register file at `L = 8`, and it overflows AVX-512's 32 `zmm` at
   `L = 17` — so those two sizes need explicit blocking (§1.3, §9).
3. **You must pad.** With a 64-byte batch granule the naive `L = 8` z-stride is 4096 B,
   which maps every element of a z-line onto **one** L1 cache set (Bailey efficiency
   `1/64`); `L = 36` is nearly as bad because `36^3·16 = 2^10·3^6`. Padding the two inner
   spatial extents to odd numbers (9, 9 for `L = 8`; 37, 37 for `L = 36`) fixes it for
   27% and 5.6% memory respectively. `L = 17` needs no padding at all (§7).

Interleaved storage is the right answer only when the *complex* vector length is 1
(SSE2 / AVX-128 with doubles), where FFTW's `DFT(A+iB)` trick makes it strictly better
than split [F1] — and AVX2/AVX-512 doubles are 2 and 4 complex per register, well past
that point.

---

## 1. The machine you are actually programming

Everything in this document follows from a small number of hardware asymmetries. Get
these in your head before writing a butterfly.

### 1.1 There is one shuffle port and two FP-arithmetic ports

Intel's own execution-unit table for Skylake client lists the unit multiplicities
directly [I1, Table 2‑12]:

| Execution unit | # of units | Representative instructions |
|---|---|---|
| Vec ALU | 3 | `(v)pand`, `(v)por`, `(v)pxor`, `(v)andp*`, `(v)orp*`, `(v)blendv*`, `(v)blendp*` |
| Vec Add | 2 | `(v)addp*`, `(v)cmpp*`, `(v)max*`, `(v)min*` |
| Vec Mul | 2 | `(v)mul*`, `(v)pmul*`, `(v)pmadd*` |
| **Shuffle** | **1** | `(v)shufp*`, `vperm*`, `(v)pack*`, `(v)unpck*`, `(v)pshuf*`, `(v)alignr`, `vbroadcast*` |

Agner Fog's measured tables for Skylake [A1] put numbers on it (columns are
µops-fused / µops-unfused / ports / latency / reciprocal throughput):

| Instruction | µops | Ports | Latency | Recip. TP |
|---|---|---|---|---|
| `ADDPS/D`, `SUBPS/D`, `MULPS/D`, all `VFMADD*` (`v,v,v`) | 1 | p01 | 4 | **0.5** |
| `ADDSUBPS/D` (`v,v,v`) | 1 | p01 | 4 | **0.5** |
| `SHUFPS/D`, `VPERMILPS/PD` (`v,v,i` and `v,v,v`) | 1 | **p5 only** | 1 | **1** |
| `VPERM2F128`, `VPERMPD y,y,i` (lane-crossing) | 1 | **p5 only** | **3** | 1 |
| `VBROADCASTSD y,m64` | 1 | **p23 (load ports)** | 3 | **0.5** |
| `VBROADCASTSD y,x` (register source) | 1 | p5 | 3 | 1 |
| `VEXTRACTF128 x,y,i` / `VINSERTF128 y,y,x,i` | 1 | p5 | 3 | 1 |
| `VINSERTF128 y,y,m128,i` | 2 | **p015 + p23** | 5 | 0.5 |
| `HADDPS/D` (`v,v,v`) | 3 | p01 + 2×p5 | 6 | 2 |

Three immediate consequences:

1. **Arithmetic is twice as wide as data movement.** A butterfly that needs one shuffle
   per arithmetic op is port-5 bound at half the machine's FLOP rate. Intel says this
   about its own 8×8 transpose example in as many words: *"In both cases, the bottleneck
   is Port 5 pressure."* [I1, §15.11.1]
2. **`ADDSUB`/`FMADDSUB` are free relative to `ADD`/`FMA`.** The interleaved
   complex-multiply idiom costs nothing extra in the arithmetic ports; its whole cost is
   the shuffles.
3. **Broadcast from *memory* runs on the load ports, broadcast from a *register* runs on
   port 5.** Intel states this explicitly for Skylake Server: *"a broadcast instruction
   with a memory operand of 32 bits or above is executed on the load ports; it is not
   executed on port 5 as other shuffles are."* [I1, §18.9.2] Never load a twiddle and
   then splat it; splat it out of memory, or use AVX-512 embedded broadcast `{1toN}`.

`HADDPD` is listed at 3 µops / 2-cycle throughput with two port-5 µops. There is no
horizontal reduction in a well-laid-out FFT; if you find yourself wanting one, your
layout is wrong.

### 1.2 Shuffle→FMA has no bypass penalty; ADD→FMA does

Skylake's producer/consumer bypass table [I1, Table 2‑13] gives, in cycles:

| producer ↓ / consumer → | `SIMD/0,1/1` | `FMA/0,1/4` | `SIMD/5/1,3` | `SHUF/5/1,3` |
|---|---|---|---|---|
| `SIMD/0,1/1` | 0 | **1** | 0 | 0 |
| `FMA/0,1/4` | **1** | 0 | 0 | 0 |
| `SIMD/5/1,3` | 0 | **1** | 0 | 0 |
| `SHUF/5/1,3` | 0 | **0** | 0 | 0 |

So feeding a shuffle result straight into an FMA is free, but feeding a plain
`vaddpd`/`vxorpd` result into an FMA costs a cycle. In a butterfly chain built purely
out of FMAs (§5) the chain stays in the FMA domain and pays nothing; a chain that
alternates `vaddpd` and `vfmadd` pays one extra cycle per transition.

### 1.3 Register files, and why `2L` is the number that matters

- AVX2: **16** `ymm` registers, 4 doubles each.
- AVX-512: **32** `zmm` registers, 8 doubles each, plus 8 mask registers and two-source
  cross-lane permutes — Intel's own feature list for AVX-512 Foundation reads
  "512-bit vector width / 32 512-bit long vector registers / … / 8 new 64-bit long mask
  registers / Two source cross-lane permute instructions / Scatter instructions /
  Embedded broadcast/rounding" [I1, Ch. 18 opening].

*(analysis)* In split batch-minor layout an `L`-point straight-line codelet holds `L`
real vectors and `L` imaginary vectors, so its minimum live set is **`2L` vector
registers**, independent of the vector width. That single number decides the shape of
each of our four sizes:

| L | `2L` | AVX2 (16 ymm) | AVX-512 (32 zmm) |
|---|---|---|---|
| 6 | 12 | fits, 4 spare | fits, 20 spare |
| 8 | 16 | **exactly full — expect spills** | fits, 16 spare |
| 17 | 34 | no | **overflows by 2** |
| 36 | 72 | no | no — must decompose |

Interleaved layout has the opposite property: an `L`-point transform needs only
`ceil(L/2)` `ymm` registers (2 complex each), i.e. 4 registers for `L = 8`. **Interleaved
trades registers for port 5; split trades port 5 for registers.** For `L = 6` and
`L = 8` on AVX-512 the split trade is obviously right. For `L = 17` and `L = 36` you must
decompose anyway (Rader/PFA — other sections), and each factor's `2·factor` fits easily:
PFA 4×9 for `L = 36` needs 8 and 18 registers respectively.

### 1.4 Cache geometry (the numbers used in §7)

- Skylake client L1D: **32 KB / 8-way**, 64 B lines → **64 sets**; fastest latency 4
  cycles; 96 B/cycle peak (2×32 B load + 1×32 B store) [I1, Table 2‑14].
- Skylake client L2: **256 KB / 4-way**, 64 B lines → **1024 sets**, 12-cycle latency
  [I1, Table 2‑14]. (Note "L2 associativity changed from 8 ways to 4 ways" vs Haswell.)
- Ice Lake client L1D is listed as **48 KB / 8**, 64 B lines, 5-cycle latency, 2×64 B
  loads + 1×64 B or 2×32 B stores per cycle [I1, Table 2‑7].

---

## 2. Interleaved vs split complex

### 2.1 The two formats, and who supports which

FFTW's manual defines them plainly: the interleaved format is "the same one used by the
basic and advanced interfaces" — you pass a pointer to the real part and the imaginary
part lives at the next memory location — while "the split format allows separate pointers
to the real and imaginary parts of a complex array." FFTW notes that "the interleaved
format is redundant, because you can always express an interleaved array in terms of a
split array with appropriate pointers and strides," and that it supports interleaved as a
special case only because it "is simpler to use, and it is common in practice" [F2].
Split is reachable through the guru interface (`fftw_plan_guru_split_dft`).

Two vendors state a performance preference for split:

- **Apple's vDSP programming guide** [AP1]: "With split complex vectors … the real parts
  of the elements in the vector are stored in one array, and the imaginary parts are
  stored in a separate array. Thus, as with real arrays, an address stride of 1 addresses
  every element. **Most functions use split complex vectors.**" And on performance: "For
  some functions, address strides of 1 for real vectors result in superior performance
  over non-unit strides because longer strides require those functions to fall back to
  using scalar-mode CPU instructions. **The use of split complex yields similar
  performance benefits for complex vectors.**"
- **Intel MKL** exposes split complex as `DFTI_COMPLEX_STORAGE = DFTI_REAL_REAL` (two
  real arrays `ZRe`, `ZIm`) alongside the default `DFTI_COMPLEX_COMPLEX` [I2]. Note that
  the newer oneMKL DPC++ interface documents only `DFTI_COMPLEX_COMPLEX` as supported on
  CPU and GPU — so if you are benchmarking against oneMKL, you are benchmarking
  interleaved.

### 2.2 The measured cost of interleaved storage: SPIRAL's "vectorisation efficiency"

McFarlin, Arbatov, Franchetti and Püschel define vectorisation efficiency as
"the ratio of total floating point operations in the scalar kernel to the total number of
vector instructions in the vectorized kernel", and work the elementwise complex
multiply of four interleaved complex numbers on 4-way SSE [S1, §3]:

> "On a traditional scalar processor, this kernel requires 24 floating point operations:
> 4 multiplications, 1 addition and 1 subtraction per complex product. … The six
> vectorized arithmetic instructions … are straightforward but the de- and re-interleaving
> of real and imaginary elements is less obvious and **requires six shuffle instructions
> as overhead**. … In Fig. 1 the efficiency is 24/(6 + 6) = 2. An ideal vectorization (not
> possible in this case) would yield 24/6 = 4 = vector length as efficiency."

That is the whole argument in one line: **interleaved complex storage throws away half
your SIMD width on de/re-interleaving.** The same paper reports 4 shuffles instead of 6
for Larrabee's swizzle-plus-writemask hardware (efficiency 2.4), and with LRB's FMA the
kernel reaches efficiency 3 out of 4 [S1, §4.2]. In split, batch-minor layout the same
kernel is 6 arithmetic instructions and 0 shuffles → efficiency 4, the ideal.

The same paper's headline: their superoptimised shuffle sequences plugged into SPIRAL give
1-D complex FFTs with "a vectorization speed-up of **5.5–6.5 for 8-way AVX** and
**10–12.5 for 16-way LRB**", and "across vector architectures and lengths, we achieve an
efficiency of up to about **80% of the vector length**" [S1, abstract and §6]. 80% is what
an expert program generator with a superoptimiser achieves *while still using interleaved
complex format*. Batch-minor split gets 100% for free.

### 2.3 The one case where interleaved wins: complex vector length 1

FFTW's 2-way SIMD scheme is not a batch scheme at all. Frigo and Johnson [F1, §IX]:

> "We view a complex DFT as a pair of real DFTs: `DFT(A + i·B) = DFT(A) + i·DFT(B)`,
> where A and B are two real arrays. Our algorithm computes the two real DFTs in parallel
> using SIMD instructions, and then it combines the two outputs according to Eq. (6).
> This SIMD algorithm has two important properties. First, **if the data is stored as an
> array of complex numbers, as opposed to two separate real and imaginary arrays, the
> SIMD loads and stores always operate on correctly-aligned contiguous locations, even if
> the complex numbers themselves have a non-unit stride.** Second, because the algorithm
> finds two-way parallelism in the real and imaginary parts of a single DFT (as opposed to
> performing two DFTs in parallel), we can completely parallelize DFTs of any size, not
> just even sizes or powers of 2."

This is genuinely elegant and it is why FFTW's SSE2 double path prefers interleaved. It
does not extend: for 4-wide vectors FFTW switches strategy entirely (§3.1), and Frigo and
Johnson are blunt about the difficulty: *"Four-way SIMD instructions are problematic,
because the input or the output are not generally stride-1, and arbitrary-stride SIMD
memory operations are more expensive than stride-1 operations."* [F1, §IX]

**Takeaway for us:** AVX2 and AVX-512 with doubles are 4-wide and 8-wide. We are firmly
in the regime where FFTW itself abandons the interleaved trick.

### 2.4 Port accounting for the complex multiply, four ways *(analysis, from [A1]+[F3])*

AVX2, double precision, `__m256d` = 4 doubles. "cmul" = one complex multiplication.

**(A) Interleaved, twiddle stored once as `(c, s)`.** This is FFTW's `simd-avx2.h`
`VZMUL` verbatim [F3]:

```c
/* fftw3/simd-support/simd-avx2.h */
#define VDUPL(x)   _mm256_movedup_pd(x)              /* [a0,a0,a2,a2] */
#define VDUPH(x)   _mm256_permute_pd(x, SHUFVALD(1,1))
#define FLIP_RI(x) _mm256_shuffle_pd(x, x, SHUFVALD(1,0))
static inline V VZMUL(V tx, V sr) {
     return _mm256_fmaddsub_pd(sr, VDUPL(tx), VMUL(FLIP_RI(sr), VDUPH(tx)));
}
```
Cost per 2 cmul: `movedup` + `permilpd` + `shufpd` = **3 p5 µops**, `mulpd` +
`fmaddsub` = 2 p01 µops. Port 5 is the bottleneck: **1.5 cycles/cmul**. *(analysis)* If `tx` comes straight from memory, `vmovddup ymm, m256` should fold `VDUPL`
into the load port, giving 2 p5 → 1.0 cycles/cmul — by analogy with Intel's explicit
statement for the single-precision forms, "moving some shuffles (`vmovsldup`/`vmovshdup`)
from Port 5 to the load ports improves performance significantly" [I1, §15.11.3]. I could
**not** extract a `MOVDDUP` row from [A1] to confirm the port assignment for the `pd`
form; verify with `perf stat -e uops_dispatched_port.port_5` before relying on it.

**(B) Interleaved with pre-splatted twiddles.** FFTW's "twiddle storage #2", commented in
the source as *"twice the space, faster (when in cache)"* [F3]: store
`tr = [c,c,c,c]` and `ti = [-s,s,-s,s]` as two whole vectors.

```c
static inline V BYTW2(const R *t, V sr) {
     const V *twp = (const V *)t;
     V si = FLIP_RI(sr);
     V tr = twp[0], ti = twp[1];
     return VFMA(tr, sr, VMUL(ti, si));    /* VFMA == _mm256_fmadd_pd */
}
```
Cost per 2 cmul: **1 p5** (`FLIP_RI`) + 2 p01 → **0.5 cycles/cmul**, balanced across
p5 and p01. Twiddle storage: 64 B per twiddle instead of 16 B.

**(C) Split, batch-minor, twiddle broadcast from memory.** Four transforms at once:

```c
/* xr,xi hold element k of four different volumes; wr,wi are broadcasts */
V yr = _mm256_fnmadd_pd(xi, wi, _mm256_mul_pd(xr, wr));   /* xr*wr - xi*wi */
V yi = _mm256_fmadd_pd (xi, wr, _mm256_mul_pd(xr, wi));   /* xr*wi + xi*wr */
```
Cost per 4 cmul: **4 p01 µops, 0 p5 µops** → **0.5 cycles/cmul**, and port 5 is
completely idle. Twiddle storage 16 B; the two broadcasts run on the load ports [I1,
§18.9.2]. On AVX-512 replace the broadcasts with embedded broadcast (`vmulpd zmm,
zmm, [rax]{1to8}`), which Intel notes "benefits from both executing the broadcast on the
load ports and micro fusion" [I1, §18.9.2].

**(D) Split, batch-minor, twiddle also vectorised** (different twiddle per batch lane —
not our case, but it is what you need if you fuse the batch and transform indices).
Same 4 p01, 0 p5, twiddle read as a full vector.

**Verdict.** (B) and (C) tie on arithmetic throughput. (C) wins because:
- Port 5 is left entirely free — and in an FFT there is other work that wants it in (B)
  but not in (C) (see below).
- Twiddle table is 4× smaller, which matters at `L = 36` where the tables get real.
- **Multiplication by ±i is free.** In (B), FFTW must spend `VBYI(x) = FLIP_RI(VCONJ(x))`
  = 1 p5 + 1 p015 for every `×i` in the butterfly network — and radix-4, radix-8 and
  split-radix are *full* of them. In (C) `×i` means "use the imaginary vector where the
  real one was, and flip a sign", and the sign flip folds into the next `add`→`sub` or
  `fmadd`→`fnmadd`. Zero instructions. This, not the complex multiply, is where split
  batch-minor actually pulls ahead.
- Every `sqrt(2)/2`-style constant multiply is likewise a single `vmulpd` with a
  broadcast, not a shuffle-multiply-shuffle sandwich.

### 2.5 What the conversion at the boundary costs

If the caller hands you interleaved data you must deinterleave. Do **not** use gather.
Intel's Optimization Manual makes the comparison with the complex-array AOS→SOA
transformation as its worked example — i.e. exactly our problem:

> "This section compares using the hardware GATHER instruction versus alternative
> implementations of handling Array of Structures (AOS) to Structure of Arrays (SOA)
> transformation. **The code separates the real and imaginary elements in a complex array
> into two separate arrays.**" [I1, §15.16.4.1]

| Microarch. | Scalar | `VPGATHERD` | AVX `VINSRTF128`/`VSHUFPS` |
|---|---|---|---|
| Broadwell | 1× | 1.7× | **4.8×** |
| Skylake | 1× | 2.7× | **4.9×** |

*(Table 15‑7 of [I1].)* And in the AVX-512 chapter, the same deinterleave done with
two-source permutes:

| Task | Baseline | Optimised | Speedup |
|---|---|---|---|
| strided load, complex → (re, im) | `vpgatherdd` | `vpermi2d` / `vpermt2d` | **4.8×** [I1, §18.22.1] |
| strided store, (re, im) → complex | `vscatterdps` | `vpermi2d` / `vpermt2d` | **4.4×** [I1, §18.22.2] |
| indexed adjacent load | `vgatherdpd` | load + masked broadcast | **2.2×** [I1, §18.22.3] |

Intel's recommendation is unambiguous: *"For best performance, replace strided loads
where the stride is short, with a sequence of loads and permutes."* [I1, §18.22.1]
(Caveat on the numbers: Chapter 18 measurements are "based on Data Cache Unit (DCU)
resident data measurements on the Skylake Server System with Intel Turbo-Boost technology
disabled, Intel SpeedStep Technology disabled, core and uncore frequency set to 1.8 GHz"
[I1, Ch. 18 opening].)

The double-precision recipe is in Intel's Example 15‑47 [I1, §15.16.4.2] — four 16-byte
loads, two `vinsertf128`, then `vunpcklpd`/`vunpckhpd`:

```asm
vmovupd     xmm0, [r9+r10]            ; one complex double
vinsertf128 ymm2, ymm0, [r9+r11], 1   ; ymm2 = {r0,i0,r1,i1}
vmovupd     xmm1, [r9+r10']
vinsertf128 ymm3, ymm1, [r9+r10''], 1 ; ymm3 = {r2,i2,r3,i3}
vunpcklpd   ymm4, ymm2, ymm3          ; {r0,r2,r1,r3}   <- reals
vunpckhpd   ymm5, ymm2, ymm3          ; {i0,i2,i1,i3}   <- imags
```
That is **2 shuffle-unit µops per 4 complex numbers** — half a shuffle per complex number,
one time, at the boundary. (Agner measures `VINSERTF128 y,y,m128,i` at 2 µops on
**p015 + p23**, so it can dispatch to port 0 or 1 instead of port 5 [A1]; Intel makes the
same point — "Using `VINSERTF128` from memory is executed in the load ports and on port 0
or 5" [I1, §15.11.2].) The lane order comes out permuted `{0,2,1,3}`; that is
harmless as long as you fix the batch ordering consistently and undo it on the way out.

---

## 3. Vectorising across the batch vs within the transform

### 3.1 The formal statement: `A ⊗ I_ν` is a free lunch

Franchetti and Püschel's short-vector Cooley–Tukey derivation gives the cleanest
statement of why batch-major works [S2, §3.1]:

> "The simplest construct that can be naturally mapped to vector code is any tensor
> product of the form `A ⊗ I_ν`, `A ∈ R^{m×n}`. The corresponding code is obtained by
> replacing every scalar operation in a program for the formula `A` by a ν-way vector
> operation. `A` is subsequently called **vector terminal**, since the construct solves
> the vectorization problem independent of `A`. In particular, `DFT_n ⊗ I_ν` can be
> **completely vectorized, no matter how `DFT_n` is further expanded**."

`DFT_L ⊗ I_ν` *is* the batched transform: `ν` independent `L`-point DFTs with the batch
index innermost. The theorem says: pick any algorithm you like — Rader, PFA, split-radix,
a hand-derived Winograd module, or the naive `O(L^2)` matrix–vector product — and its
scalar operation count maps 1:1 onto vector instructions with **no data reorganisation at
all**. That is the entire argument for batch-minor layout, and it is why you can choose
the algorithm for `L = 17` and `L = 36` on real-op count alone.

The contrast is the *within-transform* route. The same paper's short-vector Cooley–Tukey
recursion for `DFT_{mn}` carries the side condition **"we assume that `ν | m` and
`ν | n`"** [S2, §3.2]. For `ν = 4` (AVX2 double) that excludes `L = 6` and `L = 17`
outright, and forces awkward factorisations at `L = 36`. Their permutation set is also
restricted: only `P ⊗ Q` with `Q ∈ {I, L^{2ν}_2, L^{2ν}_ν, L^{ν²}_ν}` is efficiently
implementable [S2, §3.1] — i.e. within-transform vectorisation buys you a fixed menu of
in-register transposes, each costing port-5 µops (§4).

Reported result for that route, to calibrate: SPIRAL's generated short-vector DFT code was
"competitive with or faster than the hand-tuned Intel vendor library MKL 5.1, and yields a
speed-up compared to the best available C code (from FFTW 2.1.3 or SPIRAL) of **up to a
factor of 3.3 for SSE** (4-way single), and **up to a factor of 1.8 for SSE2**
(2-way double)" [S2, §1].

### 3.2 What FFTW actually does at 4-wide, and why

Frigo and Johnson [F1, §IX]:

> "On machines that support vectors of length 4, we view SIMD data as vectors of two
> complex numbers, and **each codelet executes two iterations of its loop in parallel**.
> … The source of this 2-way parallelism is the codelet loop, which can arise from the
> Cooley-Tukey decomposition of a single 1d DFT, the decomposition of a
> multi-dimensional DFT, or **a user-specified vector loop**."

"A user-specified vector loop" is `howmany` — the batch. FFTW's 4-way strategy *is*
batch-major, applied at half strength (2 complex per vector rather than 1 complex per
lane) because it must also work when there is no batch. We have a batch of thousands
(spin ⊗ colour ⊗ time-slice), so we can go all the way.

Note also what FFTW's headers reveal about its aspirations here. `simd-avx2.h` sets
`SIMD_VSTRIDE_OKA(x) ((x) == 2)` — the aligned vector load/store path is only used when
the *real* stride is exactly 2, i.e. unit complex stride [F3]. And `simd-avx512.h` falls
back to **hardware gather/scatter** for any other stride:

```c
/* fftw3/simd-support/simd-avx512.h */
static inline V LDu(const R *x, INT ivs, const R *aligned_like) {
  __m256i index = _mm256_set_epi32(3*ivs+1, 3*ivs, 2*ivs+1, 2*ivs,
                                   1*ivs+1, 1*ivs, 0*ivs+1, 0*ivs);
  return _mm512_i32gather_pd(index, x, 8);
}
```
See §8.4 for what that costs. A fixed-geometry code should never need it.

### 3.3 The concrete layout

FFTW's batched interface parameterises exactly the choice we are making [F4]:

```c
fftw_plan fftw_plan_many_dft(int rank, const int *n, int howmany,
                             fftw_complex *in,  const int *inembed, int istride, int idist,
                             fftw_complex *out, const int *onembed, int ostride, int odist,
                             int sign, unsigned flags);
```
with "the (j,k)-th element … at `j*stride + k*dist`". **Interleaved batch-major** means
`dist = 1`, `stride = howmany`; **batch-minor with a vector block** means `dist = 1`,
`stride = NB` with `NB` a multiple of the vector width. (FFTW notes only that "plans
obtained in this way can often be faster than calling FFTW multiple times for the
individual transforms" — it does not recommend a layout.)

*(analysis)* The layout I recommend for this project, written as C:

```c
/* NB = batch block, a multiple of the SIMD width in doubles (use 8) */
/* Nx, Ny = padded spatial extents (see section 7) */
double *Are;  /* Are[((z*Ny + y)*Nx + x)*NB + b] */
double *Aim;  /* same indexing */
```

Properties:
- **x-axis transform:** load `NB` doubles at `...+x*NB`, stride `NB*8 = 64 B` between
  successive `x`. Contiguous, aligned, no shuffles.
- **y-axis transform:** identical code, stride `Nx*NB*8`.
- **z-axis transform:** identical code, stride `Nx*Ny*NB*8`.
- One codelet body, three call sites, three strides. No transposes anywhere in the 3-D
  transform.
- Every load is a full aligned 64-byte vector if `NB*8 = 64` and the base is 64-B aligned.
- The batch dimension being innermost means **the per-volume stride never appears in an
  inner loop**, which removes the single worst cache-conflict hazard (§7.3).

Use `NB = 8` doubles even on AVX2: two `ymm` per plane, which both keeps everything
64-byte and cache-line aligned and gives you a free 2× unroll for latency hiding.

---

## 4. In-register transposes and shuffle networks

You need these at the format boundary, and throughout the transform if you choose the
within-transform route. Know the price.

### 4.1 The `ν log₂ν` rule, and FFTW's 4×4 double transpose

*(analysis)* A `ν × ν` transpose held in `ν` vector registers, built from two-source
shuffles, is a butterfly network of `log₂ν` stages × `ν` registers = **`ν log₂ν`
two-source shuffle instructions**. This is corroborated exactly by FFTW's own AVX 4×4
double transpose, `STN4` in `simd-avx.h` [F3] — 4 `unpck` + 4 `perm2f128` = **8 = 4·log₂4**:

```c
/* fftw3/simd-support/simd-avx.h : 4x4 double transpose + scatter */
#define STN4(x, v0, v1, v2, v3, ovs)                                   \
{                                                                      \
     V xxx0 = _mm256_unpacklo_pd(v0, v1);                              \
     V xxx1 = _mm256_unpackhi_pd(v0, v1);                              \
     V xxx2 = _mm256_unpacklo_pd(v2, v3);                              \
     V xxx3 = _mm256_unpackhi_pd(v2, v3);                              \
     STA(x,           _mm256_permute2f128_pd(xxx0, xxx2, 0x20), 0, 0); \
     STA(x +     ovs, _mm256_permute2f128_pd(xxx1, xxx3, 0x20), 0, 0); \
     STA(x + 2 * ovs, _mm256_permute2f128_pd(xxx0, xxx2, 0x31), 0, 0); \
     STA(x + 3 * ovs, _mm256_permute2f128_pd(xxx1, xxx3, 0x31), 0, 0); \
}
```

Cost: Intel lists `(v)unpck*` and `vperm*` under the single **Shuffle** unit [I1, Table
2‑12], and Agner measures `VPERM2F128 y,y,y,i` at 1 µop, **p5 only**, latency **3**,
reciprocal throughput 1 [A1]. So an AVX2 4×4 double transpose is **8 shuffle-unit µops ≈ 8 cycles of port-5
occupancy**, and the `vperm2f128` half carries latency 3 rather than 1. The
AVX-512 8×8 double transpose is `8·3 = 24` two-source permutes (`vpermt2pd` /
`vshuff64x2`), i.e. **24 port-5 µops** — and in the 512-bit port scheme the FMAs are
competing for that same port 5 (§8.1).

For scale: an 8-point split-radix complex FFT is about `4·8·log₂8 − 6·8 + 8 = 56` real
operations. Twenty-four shuffles to transpose an 8×8 block is *half the arithmetic of the
whole transform*. This is why in-register transposes must be at the boundary only, never
in the inner loop.

### 4.2 Intel's three tricks for relieving port-5 pressure

All from [I1, §15.11], whose worked example is an 8×8 single-precision transpose whose
"bottleneck is Port 5 pressure":

1. **Replace shuffles with blends.** "Alternative 1 uses 12 `vshufps` instructions that
   are executed only on port 5. Alternative 2 replaces eight of the `vshufps`
   instructions with the `vblendps` instruction which can be executed on Port 0. …
   replacing `VSHUFPS` with `VBLENDPS` relieved port 5 pressure and can gain **almost 40%
   speedup**." Codified as *Assembly/Compiler Coding Rule 66. (M impact, M generality) Use
   Blend instructions in lieu of shuffle instruction in AVX whenever possible.* Blends
   run on the Vec ALU (3 units) per [I1, Table 2‑12]. **Applies whenever the permutation
   you need is a *selection* between two vectors at fixed positions rather than a
   movement** — which covers a large fraction of FFT data motion.
2. **Design the algorithm with fewer shuffles.** "Using `VINSERTF128` from memory is
   executed in the load ports and on port 0 or 5. The original method required loads that
   are performed on the load ports and `VPERM2F128` that is only performed on port 5.
   Therefore redesigning the algorithm to use `VINSERTF128` reduces port 5 pressure and
   improves performance." [I1, §15.11.2]
3. **Perform basic shuffles on the load ports.** "Some shuffles can be executed in the
   load ports (ports 2, 3) if the source is from memory. The following example shows how
   moving some shuffles (`vmovsldup`/`vmovshdup`) from Port 5 to the load ports improves
   performance significantly." [I1, §15.11.3] — and Intel's example for this is
   *the interleaved complex multiply*, comparing `vmovaps` + register shuffle against
   `vmovsldup ymm2, [rbx+8*rcx]` directly. The double-precision analogue is
   `vmovddup ymm, m256` (see §2.4 case A).

### 4.3 Two-source permutes on AVX-512

AVX-512 Foundation adds "Two source cross-lane permute instructions" [I1, Ch. 18], and
`VPERMT2Q`/`VPERMT2D` measure at **1 µop, port 5, latency 3, throughput 1** on Skylake-X
[A1] — a full 512-bit two-source arbitrary permute for the price of one 256-bit shuffle.
This roughly halves the shuffle count of any permutation network relative to AVX2's
one-source-plus-blend idioms, and is what Intel uses in its 4.8×/4.4× gather-to-shuffle
examples [I1, §18.22]. If you do end up needing a boundary transpose on AVX-512, build it
out of `vpermi2pd`/`vpermt2pd` with the index vectors hoisted out of the loop.

The whole superoptimisation programme in [S1] exists because these sequences are hard to
find by hand: "Our superoptimizer evaluated **14 billion instruction sequences in about
2 hours** to find an efficient **6-instruction** implementation of the core data
reorganization." Their motivation for considering only two shuffle families is worth
quoting, because it tells you which two networks to bother implementing: "Two important
examples are 1) the interleaving/deinterleaving of two vectors of complex numbers
into/from one vector of real parts and one vector of imaginary parts, and 2) the
in-register transposition of a square matrix whose number of rows is the vector length.
… the motivation for considering these permutations is from [Franchetti & Püschel, CC
2008], which shows that **these are the only in-register shuffles needed to implement
FFTs**." [S1, §5]

---

## 5. FMA in butterflies: operation count and accuracy

### 5.1 Restructuring kernels for multiply–add balance

Goedecker's SIAM paper is the primary reference for FMA-shaped small radices: "We present
a new formulation of fast Fourier transformation (FFT) kernels for radix 2, 3, 4, and 5,
which have a **perfect balance of multiplies and adds**", demonstrated on IBM and SGI
workstations [G1]. (I fetched the abstract but not the full text — I am *not* quoting any
operation counts from it. Treat it as a pointer to the technique, not a source of numbers.)
Directly relevant: radix 2 and 3 for `L = 6`, radix 2/4 for `L = 8`, radix 4 and… no
radix-9 (`L = 36` PFA wants a 9-point module, for which Goedecker gives radix 3 to compose
or another section's hand-derived module).

The reason FMA balance matters on x86-64 is *port* balance, not instruction count: `mulpd`
and `fmadd` both issue 1 µop to p0/p1 at 0.5 cycles reciprocal throughput [A1], so an FMA
does two flops in the slot where a bare multiply did one. A kernel with `M` multiplies and
`A` adds costs `max(M, A)` FMA-slots if perfectly paired and `M + A` if not.

### 5.2 What FFTW's macro layer tells you to write

FFTW abstracts exactly the set of fused forms a butterfly needs [F3]. On AVX (no FMA
hardware) they degrade to add-of-multiply; on AVX2 they map to real instructions:

```c
/* simd-avx.h — no FMA available */          /* simd-avx2.h — FMA available */
#define VFMA(a,b,c)  VADD(c, VMUL(a,b))      #define VFMA    _mm256_fmadd_pd
#define VFNMS(a,b,c) VSUB(c, VMUL(a,b))      #define VFNMS   _mm256_fnmadd_pd
#define VFMS(a,b,c)  VSUB(VMUL(a,b), c)      #define VFMS    _mm256_fmsub_pd
#define VFMAI(b,c)   VADD(c, VBYI(b))        #define VFMAI(b,c) _mm256_addsub_pd(c, FLIP_RI(b))
```
Notice `VFMAI` — "add `i·b` to `c`" — is a *single* `addsub` in the interleaved AVX2 path.
And `VCONJ` is an XOR with a sign mask, not a negation, with a warning worth heeding:

```c
/* simd-avx.h */
static inline V VCONJ(V x) {
     /* We really want to write: V pmpm = VLIT(-0.0, 0.0);
        but historically some compilers have ignored the distinction between +0 and -0.
        It looks like 'gcc-8 -fast-math' treats -0 as 0 too. */
     static const union uvec pmpm = { { 0,0,0,0x80000000, 0,0,0,0x80000000 } };
     return VXOR(pmpm.v, x);
}
```
In split batch-minor layout none of this machinery is needed: `VBYI` is a rename, `VCONJ`
is a sign folded into the consumer, and every butterfly line is a plain
`vaddpd`/`vsubpd`/`vfmadd`/`vfnmadd`.

### 5.3 The accuracy consequence of FMA: it is *better*, and by exactly how much

The complete answer is Jeannerod, Kornerup, Louvet and Muller [J1]. Setting
`u = ½ β^{1−p}` (unit roundoff; `u = 2^{−53}` for binary64):

- **Without FMA**, evaluating `z = (ac − bd) + i(ad + bc)` as
  `A0 : RN(RN(ac) − RN(bd)) + i·RN(RN(ad) + RN(bc))`, Brent, Percival and Zimmermann's
  bound is `|ẑ₀/z − 1| < √5 u`, and it is sharp: "in the cases p = 24 and p = 53 the
  largest possible errors have the form `√4.9999899864…u` and
  `√4.9999999999999893…u`, respectively" [J1, §1].
- **With FMA**, the naive form
  `A1 : RN(ac − RN(bd)) + i·RN(ad + RN(bc))`
  satisfies `|ẑ₁ − z| ≤ 2u|z|`, and "for **any of these four conventional FMA-based
  algorithms**" (i.e. it does not matter which of the two products in each component you
  put inside the FMA) [J1, eq. 1.1]. The bound is asymptotically optimal.
- Compensated algorithms buy nothing normwise: "although highly accurate in the
  componentwise sense, these two compensated algorithms bring **no improvement to the
  normwise accuracy 2u** already achieved using the FMA naively" [J1, abstract].

So `√5 u ≈ 2.236 u` → `2 u`. **Using FMA in the butterfly improves the normwise error
bound of the twiddle multiplication by about 11%, and costs nothing.** There is no accuracy
argument against FMA in an FFT butterfly. The commonly-voiced worry — that FMA's single
rounding makes `ac − bd` behave unpredictably under cancellation — is a *componentwise*
concern; [J1] shows the normwise bound, which is what an FFT error analysis needs,
strictly improves.

Two genuine hazards remain:

1. **Reproducibility across code paths, not accuracy.** Whether the compiler contracts
   `a*b + c` into an FMA is a `-ffp-contract` / `#pragma STDC FP_CONTRACT` decision, so an
   AVX2 build, an AVX-512 build and a scalar reference build of the *same source* will
   differ in the last bits. If your LQCD workflow diffs against a reference, pin the
   contraction explicitly (write the intrinsics, or `-ffp-contract=off` for the reference
   path) rather than discovering it later. *(analysis; the mechanism is standard C, not a
   cited measurement.)*
2. **Twiddle table accuracy, which dominates everything above.** The FFTW project's
   accuracy notes [F5] are the cleanest statement:

   > "Most FFT implementations in the benchmark are based on the Cooley-Tukey algorithm,
   > whose floating-point error grows as **O(log N)** in the worst case (Gentleman & Sande,
   > 1966) and as **O(√log N)** on average (Schatzman, 1996), for a 1d transform of size N.
   > For these bounds to hold, certain trigonometric constants used by the algorithm (the
   > twiddle factors) must be computed accurately. **Inaccurate twiddle factors are the
   > most likely reason for the inaccuracy of an FFT routine.** … Unfortunately, most such
   > recurrence formulas accumulate errors as **O(√N), O(N), or even O(N²)**, much faster
   > than the FFT itself (Tasche & Zeuner, 2002)."

   Schatzman's `O(√log N)` average result is [SC1]. The practical rule for us: our
   transforms are tiny (`L ≤ 36`), the twiddle tables are tiny, so **compute them once
   with `sin`/`cos` (or in higher precision) into a static table and never use a
   recurrence.** For `L = 36` the whole PFA twiddle set is a few kilobytes.

   FFTW's page also carries an x86-specific warning that still matters when you compare
   against old codes: "If the floating-point unit is set in extended-precision mode (which
   is the default in GNU/Linux), variables that are register-allocated by the compiler
   become extended-precision even if they were declared `float` or `double`. Consequently,
   even double-precision routines that use inaccurate recurrences may become accurate on
   x86 machines (but they are still inaccurate on other processors)." [F5]

---

## 6. Alignment and cache-line splits

### 6.1 The rules and the measured penalties

Intel [I1, §15.6.1]:

> "Aligning data to vector length is recommended. When using 16-byte SIMD instructions,
> loaded data should be aligned to 16 bytes. Similarly, for best results when using Intel
> AVX instructions with 32-byte registers align the data to 32-bytes. **When using Intel
> AVX with unaligned 32-byte vectors, every second load is a cache-line split, since the
> cache-line is 64 bytes.** This doubles the cache line split rate compared to Intel SSE
> code that uses 16-byte vectors."

and the coding rule: *"Assembly/Compiler Coding Rule 64. (H impact, M generality) Align
data to 32-byte boundary when possible. **Prefer store alignment over load
alignment.**"* For a memory-bound SAXPY kernel Intel reports: "If only one of the three
address is not aligned to 32-byte boundary, the performance may be halved. If all three
addresses are mis-aligned relative to 32 byte, the performance degrades further."
[I1, §15.6.1]

The AVX-512 chapter quantifies it [I1, Table 18‑9] (SAXPY, 64-byte lines):

| Case | Relative performance |
|---|---|
| both sources and destination 64-B aligned | **1.00** (baseline) |
| sources aligned, destination +4 B off | **0.66×** |
| both sources and destination +4 B off | **0.59×** |
| one source +4 B off, other source and destination aligned | **0.77×** |

There is also a useful note that Skylake changed the calculus: "Beginning with Skylake
microarchitecture, this optimization [using 16-byte accesses for unaligned data] is not
necessary. The only case where 16-byte loads may be more efficient is when the data is
16-byte aligned but not 32-byte aligned. In this case 16-byte loads might be preferable as
no cache line split memory accesses are issued." [I1, §15.6.2]

### 6.2 What FFTW requires, for contrast

FFTW does **not** require 32- or 64-byte alignment on x86. `simd-common.h` sets
`ALIGNMENT = ALIGNMENTA = 16` for all of SSE2/AVX/AVX2/AVX-512 in double precision, and
the AVX/AVX2/AVX-512 `LDA`/`STA` macros are `_mm256_loadu_pd` / `_mm512_loadu` —
*unaligned* instructions [F3]. The manual only promises "typically 16-byte aligned" and
recommends `fftw_malloc` [F6]. This is a deliberate portability trade FFTW makes and that
we should *not* copy: a fixed-geometry code owns its buffers and should be 64-byte aligned
throughout, with `vmovapd`, so that misalignment bugs are loud rather than slow.

FFTW's AVX header also carries the AVX/SSE transition warning, citing the Intel manual:

```c
/* Use VZEROUPPER to avoid the penalty of switching from AVX to SSE.
   See Intel Optimization Manual (April 2011, version 248966), Section 11.3 */
#define VLEAVE _mm256_zeroupper
```
Current guidance is [I1, §18.19]: emit `VZEROUPPER` "after group B instructions were
executed and before any function call that might lead to an Intel SSE instruction
execution", and "at the end of any function that uses group B instructions", where group B
is any instruction that modifies bits 128–511 of `ymm0`–`ymm15`/`zmm0`–`zmm15`.

---

## 7. Cache-set conflicts and padding — the `L = 8` and `L = 36` problem

### 7.1 Bailey's formula

Bailey [B1] models a cache with `R = 2^r` sets, `C` ways, `W = 2^w` words per line, and a
walk of `L` elements at word-stride `S`:

```
D = min_{0 < a,b < R} | b·S − a·R·W |          (nearness of S to a simple fraction of RW)
G = (1/C)·max(C − D, 0)                        (replacement frequency)
E = ( L − G·(L − b·C) ) / L                    (fraction of lines still resident)
```

with the boundary case spelled out explicitly:

> "An obvious example of an inefficient stride is a large power of two. Then all cache
> lines will be fetched into the same location of the cache, and the other `R − 1`
> locations will be completely unutilized. In other words, at most `C` lines of this data
> can be stored in the cache. **The resulting efficiency is only `1/R`.** Clearly if an
> application program has arrays whose dimensions are large powers of two, these arrays
> should be 'padded,' such as by declaring their leading dimensions … to be slightly
> larger than a power of two." [B1, §1]

And the counter-intuitive part, which is why you cannot just avoid powers of two: for
`R = 32, C = 4, W = 16` (an RS/6000 L1), stride 72 gives efficiency **1.0** while
stride 73 gives **0.414** [B1, §2] — because `7 × 73 = 511 ≈ 512 = RW`. Near-rational
strides are as bad as power-of-two strides.

### 7.2 The four sizes, computed

*(analysis — Bailey's formula [B1] applied to Skylake-client geometry from [I1, Table
2‑14]. Reproducer:*

```python
import math
def bailey(S_words, R, C, W):            # R sets, C ways, W doubles per line
    RW, best, bb = R*W, None, None
    for b in range(1, R):
        for a in (math.floor(b*S_words/RW), math.ceil(b*S_words/RW)):
            if 0 < a < R:
                d = abs(b*S_words - a*RW)
                if best is None or d < best: best, bb = d, b
    G = max(C-best, 0)/C
    L = R*C
    return best, G, (L - G*(L - bb*C))/L          # D, G, E
def sets(stride_bytes, nsets=64, line=64):
    if stride_bytes % line: return None           # non-integral -> rotating, benign
    step = (stride_bytes//line) % nsets
    return nsets // math.gcd(step or nsets, nsets)
# Skylake client L1D: bailey(S_bytes//8, R=64, C=8, W=8);  L2: R=1024, C=4, W=8
```
*)* Naive layout
`A[z][y][x]` of `L^3` interleaved complex doubles (16 B/element), Skylake L1D = 64 sets ×
8 ways × 64 B:

| L | stride | bytes | lines | distinct L1 sets | max resident lines | Bailey E |
|---|---|---|---|---|---|---|
| 6 | y (L) | 96 | 1.50 | — (non-integral) | — | 0.836 |
| 6 | z (L²) | 576 | 9.00 | 64 | 512 | 1.000 |
| 6 | whole volume | 3456 | 54.00 | 32 | 256 | 0.500 |
| **8** | y (L) | 128 | 2.00 | 32 | 256 | **0.500** |
| **8** | z (L²) | 1024 | 16.00 | **4** | 32 | **0.062** |
| **8** | whole volume | 8192 | 128.00 | **1** | **8** | **0.016 = 1/64** |
| 17 | y (L) | 272 | 4.25 | — (non-integral) | — | 0.426 |
| 17 | z (L²) | 4624 | 72.25 | — (non-integral) | — | 0.613 |
| 17 | whole volume | 78608 | 1228.25 | — (non-integral) | — | 1.000 |
| **36** | y (L) | 576 | 9.00 | 64 | 512 | 1.000 |
| **36** | z (L²) | 20736 | 324.00 | **16** | 128 | 1.000 (L1) / **0.250 (L2)** |
| **36** | whole volume | 746496 | 11664.00 | **4** | 32 | **0.062 (L2)** |

Read the disasters off the table:

- **`L = 8`, whole-volume stride 8192 B = 128 lines.** `128 mod 64 = 0`: *every* volume
  maps to the **same single L1 set**. An 8-wide AVX-512 batch gather that touches element
  `(x,y,z)` of 8 consecutive volumes fills exactly all 8 ways of one set, and any other
  access evicts it. This is Bailey's `E = 1/R = 1/64` case verbatim.
- **`L = 8`, z-stride 1024 B = 16 lines.** Only 4 of 64 sets; 32 lines resident.
- **`L = 36`, whole-volume stride 746496 B.** `746496 = 2^10 · 3^6`, so `746496/64 =
  2^4 · 3^6 = 11664` and `11664 mod 64 = 16` — four L1 sets, and in L2 Bailey's `E =
  0.062`. `36^3` is not a power of two but it carries `2^10` as a factor, which is all
  that matters.
- **`L = 17` is free.** `17^3 · 16 = 78608 B` is not a whole number of cache lines, so
  the set index *and* the byte offset both rotate; no fixed conflict pattern exists. The
  flip side: 3 out of every 4 volumes start at a 16-B-but-not-64-B boundary, so you must
  pad to a multiple of 64 B if you want aligned loads (§6).

### 7.3 The fix: batch-minor plus odd-line strides

*(analysis)* Batch-minor layout removes the whole-volume stride from every inner loop
(§3.3), so the only strides that matter are y and z. With `A[z][y][x][NB]` and
`NB·8 = 64 B`, the y-stride is `Nx` lines and the z-stride is `Nx·Ny` lines. Since
`gcd(k, 64) = 1` iff `k` is odd, the criterion collapses to:

> **Pad the two inner spatial extents `Nx` and `Ny` to odd numbers.** Then every stride is
> an odd number of cache lines, every one of the 64 L1 sets (and 1024 L2 sets) is used,
> and Bailey's `E = 1`.

Computed for our sizes (`NB = 8` doubles = 64 B per element-vector):

| L | Nx = Ny | y-stride | z-stride | L1 sets (y / z) | L2 sets (y / z) | memory cost |
|---|---|---|---|---|---|---|
| 6 | 6 (naive) | 384 B | 2304 B | 32 / 16 | 512 / 256 | 1.00× |
| **6** | **7** | 448 B | 3136 B | **64 / 64** | **1024 / 1024** | 1.36× |
| 8 | 8 (naive) | 512 B | 4096 B | **8 / 1** | 128 / 16 | 1.00× |
| **8** | **9** | 576 B | 5184 B | **64 / 64** | **1024 / 1024** | 1.27× |
| **17** | **17** | 1088 B | 18496 B | **64 / 64** | **1024 / 1024** | 1.00× (free) |
| 36 | 36 (naive) | 2304 B | 82944 B | 16 / 4 | 256 / 64 | 1.00× |
| **36** | **37** | 2368 B | 87616 B | **64 / 64** | **1024 / 1024** | 1.06× |

Notes:
- `L = 8` unpadded with `NB = 8` gives a z-stride of exactly 4096 B — **one** L1 set. The
  single worst possible number. Padding `Nx = Ny = 9` costs 27% memory and fixes both axes.
- `L = 36 → 37` is nearly free (5.6%) and 37 is prime, so `Nx`, `Nx·Ny` and `Nx·Ny·Nz` are
  all odd. This is the padding to use.
- `L = 6 → 7` costs 36%; at `6^3 · NB` the whole padded volume is `7·7·6·64 = 18.4 KB`,
  which still fits in a 32 KB L1, so you may prefer to leave it unpadded and rely on the
  volume being resident. Measure.
- With `NB = 4` (AVX2 minimum), odd `Nx` makes strides a *non-integral* number of lines,
  which is also benign — but 32-byte-granular offsets are more fragile. Prefer `NB = 8`.
- The same reasoning applies to the 4 KB page/TLB level; Intel's general advice is to
  "organize the data so consecutive accesses can usually be found in the same 4-KByte
  page" [I1, §3] and to eliminate 64 KB-aliased accesses [I1, §11.6.3], which the
  odd-line rule also handles.

---

## 8. AVX2 vs AVX-512

### 8.1 The port scheme changes under you — this is the big one for FFTs

[I1, §18.20], verbatim:

> "Skylake microarchitecture has two port schemes, one for using 256-bit or less
> registers, and another for using 512-bit registers. **When using registers up to or
> including 256 bits, FMA operations dispatch to ports 0 and 1 and SIMD operations
> dispatch to ports 0, 1 and 5. When using 512-bit register operations, both FMA and SIMD
> operations dispatch to ports 0 and 5.** The maximum register width in the reservation
> station (RS) determines the 256 or 512 port scheme. Notice that when using AVX-512
> encoded instructions with YMM registers, the instructions are considered to be 256-bit
> wide. … The result of the 512-bit port scheme is that XMM or YMM code dispatches to two
> ports (0 and 5) instead of three ports (0, 1, and 5) and may have lower throughput and
> longer latency compared to the 256-bit port scheme."

And with the worked example: "In the 256-bit and 512-bit mixed code example, the broadcast
is 512 bits wide; therefore, the processor uses the 512-bit port scheme where the FMAs
dispatch to ports 0 and 5 and `permd` to port 5, **thus increasing the pressure on port
5**." [I1, §18.20]

**Why this is specifically dangerous for FFT code:** an FFT butterfly is exactly a mix of
FMA and shuffle. In the 256-bit scheme, FMAs own p0/p1 and shuffles own p5 — perfect
separation. In the 512-bit scheme, FMAs and shuffles *share* p5. A shuffle-heavy
interleaved kernel that was FMA-bound on AVX2 can become port-5-bound on AVX-512 and gain
far less than 2×. Batch-minor split layout, which has **no shuffles at all**, does not
have this problem — another argument for it, and the reason it is the layout I recommend
for the AVX-512 nodes in particular.

Two more AVX-512 microarchitectural facts to plan around:

- **FMA latency is port-dependent** [I1, §18.18, Table 18‑8]: "When executing in 512-bit
  register port scheme, Port 0 FMA has a latency of four cycles, and Port 5 FMA has a
  latency of six cycles." With "fast bypass" (all sources come from the FMA unit) group A
  (`vadd*`, `vfmadd*`, `vmul*`, …) is 4 cycles on both ports; without it, 4 on port 0 and
  **6 on port 5**. Chains of FMAs that stay inside the FMA domain are worth building
  deliberately (cf. §1.2).
- **Not every server has two FMA units** [I1, §18.21]: "Some processors based on Skylake
  microarchitecture have two Intel AVX-512 FMA units, on ports 0 and 5, while other
  processors based on Skylake microarchitecture have a single Intel AVX-512 FMA unit, which
  is located on port 0. Code that is optimized to run on a processor with two FMA units
  might not be optimal when run on a processor with one FMA unit." Intel gives a runtime
  detection routine (Example 18‑30) whose test is: run FMA-only and FMA+shuffle loops and
  declare one FMA unit if their throughput ratio is `< 1.5`. Xeon Bronze/Silver parts are
  the single-FMA ones — which is exactly the Xeon Silver 4116 in §8.2.

### 8.2 Frequency licences: the published numbers, and they disagree by generation

Intel's own description [I1, §2.5.3, Table 2‑10]:

| Level | Category | Max frequency | Instruction types |
|---|---|---|---|
| 0 | AVX2 light | highest (`P0n`) | scalar, AVX128, SSE, AVX2 **without** FP or INT MUL/FMA |
| 1 | AVX2 heavy + AVX-512 light | `P0n-AVX2` | AVX2 FP + INT MUL/FMA; AVX-512 **without** FP or INT MUL/FMA |
| 2 | AVX-512 heavy | `P0n-AVX-512` | AVX-512 FP + INT MUL/FMA |

Note where an FFT lands: **AVX2 FMA is already License 1.** AVX-512 *shuffles* are
License 1. AVX-512 *FMA* is License 2. Also: "When the core requests a higher license
level than its current one, **it takes the PCU up to 500 microseconds to grant the new
license. Until then the core operates at a lower peak capability.** … A timer of
approximately **2 ms** is applied before going back to a higher frequency level."
[I1, §2.5.3]

Skylake-SP server, measured:

- **Cloudflare / Krasnov** [C1], Xeon Silver 4116 (a single-FMA-unit part), dual socket:
  "We use the Xeon Silver 4116 CPUs, with a base frequency 2.1 GHz … **Running AVX-512
  even just on one core on this CPU will reduce the base frequency to 1.8 GHz. Running
  AVX-512 on all cores will reduce it to just 1.4 GHz.**" Their measured application-level
  cost of a 9%-slower cipher was "the server that uses OpenSSL serves 10% fewer requests
  per second", and throughput dropped "by over 7% when 20% of requests" used it. Their
  recommendation: "If you do not require AVX-512 for some specific high performance tasks,
  I suggest you disable AVX-512 execution on your server or desktop, to avoid accidental
  AVX-512 throttling."
- **Lemire (with Downs)** [L1], Xeon Gold 5120:

  | Mode | 1 active core | 9+ active cores |
  |---|---|---|
  | normal | 3.2 GHz | 2.7 GHz |
  | AVX2 | 3.1 GHz | 2.3 GHz |
  | AVX-512 | 2.9 GHz | **1.6 GHz** |

  With the decision rule stated as: use heavy 512-bit instructions only if they are "more
  than 2× faster on a per cycle basis"; light AVX-512 is fine if it gives "greater than
  15% gain for your overall application". Their cautionary tale: OpenSSL got "only 30%
  per-cycle gains using heavy AVX-512 instructions" and then "disabled this optimization".
  Also, importantly, "downclocking is per core and for a short time (approximately 2
  milliseconds)".
- **Travis Downs, Skylake-X W-2104** [D1]: measured licence frequencies L0 = 3.2 GHz,
  L1 = 2.8 GHz, L2 = 2.4 GHz. He also documents the *transition* cost, which is easy to
  overlook: the voltage-only transition lasts "~8 to 20 μs" during which "the payload
  executes *much* more slowly, with an IPC of ~0.25" — a 4× slowdown that "affects … all
  ALU instructions", SIMD or not; a full frequency transition halts the core for "~11 μs";
  and the core dwells at the low frequency for "~650 μs" after the last wide instruction.
  For a batched FFT kernel that runs for milliseconds this amortises; for one called in
  short bursts between scalar LQCD code, it does not.
- **Ice Lake client is a different world** [D2]. On an i5-1035G4: light and heavy 256-bit
  are identical to scalar at every core count, and "Only for a single active core count is
  there any decrease with wider instructions, and it is a paltry only 100 MHz: from 3,700
  MHz to 3,600 MHz when any 512-bit instructions are used." His conclusion: "on ICL and
  RKL client, you don't have to fear the downclock."

**Practical policy.** Write the kernel at 256-bit width with 64-byte-aligned data and a
`NB = 8` batch block (so it is two `ymm` per plane on AVX2 and one `zmm` on AVX-512),
runtime-dispatch a 512-bit variant, and **measure the 512-bit variant on the actual node
with the actual surrounding LQCD code in the loop, not in a microbenchmark.** The AVX-512
answer is generation-dependent (Skylake-SP: hostile; Ice Lake and later: benign) and
SKU-dependent (one vs two FMA units).

### 8.3 Mask registers and the tail

AVX-512's 8 mask registers make the batch tail (`NB_total mod 8`) a masked epilogue rather
than a scalar loop, at no shuffle cost. FFTW instead uses an "extra_iter hack" with a
delicate store-ordering requirement, visible in the comments of `simd-avx.h` [F3]:

```c
/* WARNING: the extra_iter hack depends upon the store of the low
   part occurring after the store of the high part */
```
Since we choose our own batch blocking, just round the batch up to a multiple of 8 with
zero padding; it is cheaper than either mechanism.

### 8.4 Gather and scatter: don't

Measured reciprocal throughputs from [A1] (double-precision FP gathers/scatters, all
values are reciprocal throughput in cycles; scatter also shows µop counts):

| Instruction | Haswell | Broadwell | Skylake | Skylake-X | Ice Lake |
|---|---|---|---|---|---|
| `VGATHERQPD xmm` (2 elts) | 7 | 5 | 2 | 2 | 1 |
| `VGATHERQPD ymm` (4 elts) | 9 | 6 | 4 | 4 | 3 |
| `VGATHERQPD zmm{k}` (8 elts) | — | — | — | **5** | **5** |
| `VSCATTERQPD zmm{k}` (8 elts) | — | — | — | **12** (27 µops) | **8** (20 µops) |
| `VSCATTERDPS zmm{k}` (16 elts) | — | — | — | **17–20** (43 µops) | **12** (36 µops) |

For comparison AMD Zen 4 measures `VGATHERQPD zmm{k}` at 48 µops / 10 cycles and
`VSCATTERQPD zmm{k}` at 48 µops / 12 cycles [A1]. Intel's Gracemont E-cores are worse
still: "The VGATHER instructions are implemented as micro-coded flow. **Latency is ~50
cycles.**" [I1, §4.1.8.6]

Intel's own list of access patterns that should *not* use gather is a description of the
FFT [I1, Example 15‑43]:

| Access pattern | Recommended instead |
|---|---|
| sequential elements | regular SIMD loads |
| fewer than 4 elements | regular SIMD load + horizontal data movement |
| **small strides** | "load all nearby elements + shuffle/permute to collected strided elements" |
| **transpositions** | "regular SIMD loads + shuffle/permute/blend to transpose to columns" |

with "If data supply locality is from memory, software sequences are likely to perform
better than the hardware GATHER instruction" [I1, §15.16.4]. Combined with the 4.8×/4.4×
figures in §2.5: **there is no place for gather or scatter in a fixed-geometry batched
FFT.** If your layout needs them, change the layout.

---

## 9. Size-by-size

All recommendations assume the batch-minor split layout of §3.3, `NB = 8` doubles,
64-byte alignment, odd `Nx`/`Ny` (§7.3), and precomputed twiddle tables (§5.3).

### L = 6 = 2·3, 216 points

- `2L = 12` registers: fits on AVX2 with 4 to spare, on AVX-512 with 20 to spare.
  This is the one size that is fully register-resident, in split layout, on both ISAs.
- Fully unrolled straight-line 6-point codelet, no loop. With Good-Thomas / PFA (2 and 3
  coprime) the 6-point module is a 2-point and a 3-point with an index permutation and
  **no twiddle factors at all** — which in batch-minor layout means the codelet is pure
  `vaddpd`/`vsubpd` plus two multiplies by `cos(2π/3)` and `sin(2π/3)`, both broadcasts.
  Zero shuffles, zero gathers, no twiddle table.
- The `×i` freedom (§2.4) matters most here: at `L = 6` twiddle multiplication is nearly
  all there is, and interleaved layout would spend a `FLIP_RI` on port 5 for essentially
  every operation.
- Padding: `Nx = Ny = 7` costs 36% memory. A padded `6^3` volume with `NB = 8` is
  `7·7·6·64 = 18.4 KB` — still L1-resident, so also consider leaving it unpadded and
  measuring; the naive z-stride (2304 B, 16 sets, 128 lines resident) is fine for a walk
  of length 6.

### L = 8 = 2³, 512 points

- **`2L = 16` = the entire AVX2 register file.** This is the single sharpest register
  constraint of the four sizes. Options, in order of preference:
  1. AVX-512: 16 of 32 `zmm`, 16 free for twiddles and temporaries. Comfortable.
  2. AVX2 with the 8-point split into two radix-4 halves (or radix-2 × radix-4), so the
     live set is ~10 vectors at any moment. Reload from L1 between halves; L1 is 4-cycle
     latency and 2×32 B/cycle [I1, Table 2‑14], so a handful of extra loads is far cheaper
     than a spill/reload pair on the same address.
  3. AVX2 with the twiddle-free structure of radix-8: the classic radix-8 butterfly needs
     only `1`, `±i`, and `(1±i)/√2`. In split layout `±i` is free and `(1±i)/√2` is one
     broadcast multiply; so the register pressure is genuinely `2L` with no twiddle
     registers at all, which makes option 3 viable if you also keep the two output
     accumulators in the same registers as the inputs.
- Split-radix's `4n log₂n − 6n + 8 = 56` real operations for `n = 8` becomes 56 vector
  instructions covering 8 transforms — ~28 cycles at 2 arithmetic µops/cycle, with port 5
  idle throughout.
- **Padding is mandatory.** The naive `NB = 8` z-stride is 4096 B = exactly one L1 set
  (§7.3). `Nx = Ny = 9` costs 27% memory and restores all 64 sets. If memory matters, pad
  only enough to make both extents odd — 9 is the minimum.
- The naive interleaved whole-volume stride of 8192 B (Bailey `E = 1/64`) is the exact
  scenario Bailey warns about; if any part of your pipeline strides across volumes at
  `8192 B`, fix it.

### L = 17 (prime), 4913 points

- **Batch-minor split layout is worth the most here**, because the `A ⊗ I_ν` theorem [S2]
  says the vector cost equals the scalar real-operation count *for any algorithm*. So the
  choice among Rader, Winograd, a hand-derived 17-point module, and the brute-force
  17×17 matrix–vector product reduces to a single scalar op-count comparison, decided in
  the other sections, with no SIMD-friendliness weighting at all.
- The brute-force option deserves a serious look, and there is published support for it.
  McFarlin et al. found, for Larrabee's 16-way single precision: "indeed **up to a size of
  about n = 20, the direct computation is preferable, even though the mathematical
  operations count … is inferior. The reason is in LRB's dedicated replicate HW, which
  enables efficient scalar broadcasts and FMA instructions which are well-suited for a
  direct computation.**" [S1, §6] AVX-512 has exactly the two features they credit:
  broadcast on the load ports and embedded broadcast [I1, §18.9.2], and 2×512-bit FMA.
  **Caveat:** Larrabee was a 16-way in-order single-precision machine with a different
  cost model; treat this as a strong hint to *measure*, not as a result that transfers.
- *(analysis)* The register arithmetic: `2L = 34 > 32 zmm`. A direct 17-point matvec needs
  34 accumulators; block the outputs into two halves (8 + 9) and re-read the 17 inputs
  twice from L1. A Rader implementation needs the 16-point cyclic convolution, whose
  `2·16 = 32` vectors also exactly fill the file — same problem, same fix.
- **No padding needed.** 17 is odd, so with `NB = 8` the y-stride (1088 B = 17 lines) and
  z-stride (18496 B = 289 lines) are both odd multiples of the line size → all 64 L1 sets
  and all 1024 L2 sets. `L = 17` is the one size where the cache geometry is free.
- If you ever handle `17^3` in *interleaved* form, note the volume is `78608 B`, which is
  `1228` lines **+ 16 bytes** — so consecutive volumes are 16-byte but not 64-byte aligned.
  Pad to `78656 B` (1229 lines, odd) to keep both alignment and set coverage.

### L = 36 = 2²·3², 46656 points

- `2L = 72` registers: must decompose. PFA on 4 and 9 (coprime) gives modules needing
  `2·4 = 8` and `2·9 = 18` registers — both comfortable on AVX2, luxurious on AVX-512, and
  PFA introduces **no twiddle factors between the two stages**, only an index permutation.
  In batch-minor layout that permutation is a change of address arithmetic, not a shuffle.
- **This is the size where cache blocking starts to matter.** A padded `37·37·36` volume
  with `NB = 8` is `37·37·36·8·8 B ≈ 3.0 MB` per real component, 6.1 MB for both — far
  beyond L2 (256 KB–1 MB) and into L3. Blocking strategy is another section's job, but two
  layout facts constrain it:
  - The naive `36^3` interleaved volume stride is `746496 = 2^10·3^6` bytes: four L1 sets,
    Bailey `E = 0.062` in L2. `36^3` looks innocent and is not.
  - Padding `Nx = Ny = 37` costs only **5.6%** and makes every stride (`37`, `37² = 1369`,
    `37³ = 50653` lines) an odd number of cache lines. 37 is prime; this is the cheapest
    and most robust padding of the four sizes. Do it.
- The twiddle table for the x/y/z passes is `O(L²)` complex numbers if you precompute per
  `(k, j)` pair — for `L = 36` that is `1296 × 16 B = 20.7 KB` per axis, which does *not*
  want to live in L1 alongside the data. Store twiddles as scalars (16 B) and broadcast
  from memory (§1.1) rather than as pre-splatted vectors (FFTW's `TWVL2` scheme, 64 B
  each, is "faster **when in cache**" [F3] — at `L = 36` it is not).

---

## 10. Unsourced engineering notes

Everything here is my own reasoning, attributed to nobody. It follows from the cited
hardware numbers but is not a claim anyone in the literature has made.

1. **The transpose-count rule.** A `ν × ν` in-register transpose built from two-source
   shuffles costs `ν log₂ν` shuffle instructions: 8 for AVX2 doubles (`ν = 4`), 24 for
   AVX-512 doubles (`ν = 8`). Corroborated for `ν = 4` by FFTW's `STN4` [F3] being exactly
   8 instructions, but the general rule is mine.
2. **The "odd number of cache lines" padding criterion.** With any layout whose element
   granule is exactly one cache line, `gcd(stride_in_lines, n_sets) = 1` iff the stride is
   odd (since `n_sets` is a power of two), so odd strides guarantee full set coverage.
   This is a corollary of Bailey [B1] plus a power-of-two set count; it is not stated in
   Bailey.
3. **The `2L` register criterion** for batch-minor split codelets, and the resulting
   AVX2/AVX-512 feasibility table in §1.3.
4. **The four-way port accounting in §2.4.** The instruction sequences are FFTW's [F3];
   the port and cycle attributions are mine, read off Agner's tables [A1].
5. **"Multiplication by ±i is free in split layout."** Obvious once stated, but I found no
   source that says it in those words, and it is the strongest single argument for the
   layout.
6. **`-ffp-contract` reproducibility warning** in §5.3.
7. **`NB = 8` even on AVX2.** Rationale: 64-byte granularity makes every stride an integral
   number of lines so the §7.3 criterion applies cleanly, and it gives a free 2× unroll.
8. **The direct-DFT arithmetic for L = 17.** A direct 17-point complex matvec in split
   batch-minor form costs about `(17·17 − 33)·4 + 66 ≈ 1090` vector FMA/mul µops per block
   of 8 transforms (exploiting only the `w = 1` row and column), i.e. ≈ 545 cycles at
   2 µops/cycle, ≈ 68 cycles per 17-point transform. A Rader implementation via three
   16-point FFTs plus 16 complex multiplies is roughly 650 real ops per transform, ≈ 41
   cycles. So Rader is ahead by ~1.6× on µop count, but the direct method has a single
   perfectly-pipelined dependency structure, no permutation, no convolution kernel table,
   and can stream inputs from L1. Given the ICS 2011 finding for `n ≤ 20` [S1], both should
   be built and timed. **These numbers are my arithmetic, not a measurement.**
9. **Deinterleave lane order.** Intel's `vunpcklpd`/`vunpckhpd` recipe (§2.5) produces
   batch order `{0,2,1,3}` per 256-bit register, not `{0,1,2,3}`. Since the batch index is
   semantically unordered (it is a set of independent transforms), you can simply define
   the permuted order to be the canonical one internally and invert it on output — no
   extra shuffles. Do document it, because it will otherwise show up as a "bug" in a
   correctness test that compares against a reference in the caller's ordering.

---

## 11. References

Verification status: **[fetched]** = I retrieved the document (or the specific page/API
record) during this session and the quotes above are taken from that retrieval.

| Tag | Reference | URL | Status |
|---|---|---|---|
| A1 | Agner Fog, *Instruction tables: Lists of instruction latencies, throughputs and micro-operation breakdowns for Intel, AMD, VIA and other CPUs*. Sections used: Intel Haswell, Broadwell, Skylake, Skylake-X, Ice Lake/Tiger Lake, AMD Zen 4/5. | https://www.agner.org/optimize/instruction_tables.pdf | **[fetched]** |
| AP1 | Apple, *About the vDSP API* (vDSP Programming Guide), "Data Formats" / performance notes. | https://developer.apple.com/library/archive/documentation/Performance/Conceptual/vDSP_Programming_Guide/About_vDSP/About_vDSP.html | **[fetched]** |
| B1 | D. H. Bailey, "Unfavorable Strides in Cache Memory Systems (RNR Technical Report RNR-92-015)", *Scientific Programming* **4**(2):53–58, 1995. DOI 10.1155/1995/937016. | https://www.davidhbailey.com/dhbpapers/cache.pdf (full text); https://api.crossref.org/works/10.1155/1995/937016 (metadata) | **[fetched]** (both) |
| C1 | V. Krasnov, "On the dangers of Intel's frequency scaling", Cloudflare blog, 2017. Xeon Silver 4116. | https://blog.cloudflare.com/on-the-dangers-of-intels-frequency-scaling/ | **[fetched]** |
| D1 | T. Downs, "Gathering Intel on Intel AVX-512 Transitions", *Performance Matters*, 17 Jan 2020. Skylake-X W-2104 licence frequencies and transition costs. | https://travisdowns.github.io/blog/2020/01/17/avxfreq1.html | **[fetched]** |
| D2 | T. Downs, "Ice Lake AVX-512 Downclocking", *Performance Matters*, 19 Aug 2020. i5-1035G4. | https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html | **[fetched]** |
| F1 | M. Frigo and S. G. Johnson, "The Design and Implementation of FFTW3", *Proc. IEEE* **93**(2):216–231, 2005. DOI 10.1109/JPROC.2004.840301. §IX "How FFTW3 uses SIMD". | https://www.fftw.org/fftw-paper-ieee.pdf ; https://api.crossref.org/works/10.1109/JPROC.2004.840301 | **[fetched]** (both) |
| F2 | FFTW 3.3.x manual, "Interleaved and split arrays". | https://www.fftw.org/doc/Interleaved-and-split-arrays.html | **[fetched]** |
| F3 | FFTW 3 source, `simd-support/simd-avx.h`, `simd-avx2.h`, `simd-avx512.h`, `simd-common.h` (git master). | https://raw.githubusercontent.com/FFTW/fftw3/master/simd-support/simd-avx.h (and siblings) | **[fetched]** |
| F4 | FFTW 3.3.x manual, "Advanced Complex DFTs" (`fftw_plan_many_dft`, `howmany`/`stride`/`dist`). | https://www.fftw.org/fftw3_doc/Advanced-Complex-DFTs.html | **[fetched]** |
| F5 | benchFFT / FFTW, "FFT Accuracy Benchmark Comments". | https://www.fftw.org/accuracy/comments.html | **[fetched]** |
| F6 | FFTW 3.3.x manual, "SIMD alignment and fftw_malloc". | https://www.fftw.org/fftw3_doc/SIMD-alignment-and-fftw_005fmalloc.html | **[fetched]** |
| G1 | S. Goedecker, "Fast Radix 2, 3, 4, and 5 Kernels for Fast Fourier Transformations on Computers with Overlapping Multiply–Add Instructions", *SIAM J. Sci. Comput.* **18**(6):1605–1611, 1997. DOI 10.1137/S1064827595281940. | https://epubs.siam.org/doi/10.1137/S1064827595281940 (abstract); https://api.crossref.org/works/10.1137/S1064827595281940 (metadata) | **[fetched — abstract and metadata only; full text NOT retrieved.** No operation counts are quoted from it above.] |
| I1 | Intel, *Intel® 64 and IA-32 Architectures Optimization Reference Manual*, Volume 1, document 248966-049US. Sections used: 2.5.3, 2.6.3, 4.1.8.6, 11.6.3, 15.6, 15.7, 15.11, 15.16.4, 18.1, 18.9.2, 18.18–18.24; Tables 2-7, 2-10, 2-12, 2-13, 2-14, 15-7, 18-8, 18-9; Examples 15-19, 15-21, 15-43, 15-46, 15-47, 18-18, 18-31, 18-32, 18-33. | https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf | **[fetched]** |
| I2 | Intel MKL / oneMKL DFTI documentation, `DFTI_COMPLEX_STORAGE` = `DFTI_COMPLEX_COMPLEX` / `DFTI_REAL_REAL` storage schemes. | https://www.smcm.iqfr.csic.es/docs/intel/mkl/mkl_manual/fft/fft_StorageSchemes.htm (mirror of the MKL manual page) | **[UNVERIFIED — I did not fetch this page directly; the parameter names and semantics above come from search-result summaries. Verify against the current oneMKL reference before relying on it.]** |
| J1 | C.-P. Jeannerod, P. Kornerup, N. Louvet and J.-M. Muller, "Error bounds on complex floating-point multiplication with an FMA", *Mathematics of Computation* **86**(304):881–898, 2017. DOI 10.1090/mcom/3123. Extends R. P. Brent, C. Percival and P. Zimmermann, *Math. Comp.* **76**:1469–1481, 2007. | https://www.ams.org/journals/mcom/2017-86-304/S0025-5718-2016-03123-3/S0025-5718-2016-03123-3.pdf ; https://api.crossref.org/works/10.1090/mcom/3123 | **[fetched]** (both) |
| L1 | D. Lemire (with T. Downs), "AVX-512: when and how to use these new instructions", *Daniel Lemire's blog*, 7 Sep 2018. Xeon Gold 5120 frequency table. | https://lemire.me/blog/2018/09/07/avx-512-when-and-how-to-use-these-new-instructions/ | **[fetched]** |
| S1 | D. S. McFarlin, V. Arbatov, F. Franchetti and M. Püschel, "Automatic SIMD vectorization of fast Fourier transforms for the Larrabee and AVX instruction sets", *Proc. Int'l Conf. on Supercomputing (ICS'11)*, pp. 265–274, 2011. DOI 10.1145/1995896.1995938. | https://users.ece.cmu.edu/~franzf/papers/ics2011.pdf ; https://api.crossref.org/works/10.1145/1995896.1995938 | **[fetched]** (both) |
| S2 | F. Franchetti and M. Püschel, "Short Vector Code Generation for the Discrete Fourier Transform", *Proc. IEEE Int'l Parallel and Distributed Processing Symposium (IPDPS)*, pp. 58–67, 2003. | https://users.ece.cmu.edu/~franzf/papers/ipdps03.pdf | **[fetched]** |
| SC1 | J. C. Schatzman, "Accuracy of the Discrete Fourier Transform and the Fast Fourier Transform", *SIAM J. Sci. Comput.* **17**(5):1150–1166, 1996. DOI 10.1137/S1064827593247023. Cited here only via [F5] for the `O(√log N)` average error result. | https://api.crossref.org/works/10.1137/S1064827593247023 (metadata) | **[fetched — metadata only; full text NOT retrieved.]** |
| SA1 | G. Sansone and M. Cococcioni, "Experiments on Speeding Up the Recursive Fast Fourier Transform by Using AVX-512 SIMD Instructions", in *Applications in Electronics Pervading Industry, Environment and Society*, Lecture Notes in Electrical Engineering, pp. 255–263, 2023. DOI 10.1007/978-3-031-30333-3_34. Studies exactly our question — "block interleaving" vs "complex interleaving" storage of complex vectors under AVX-512. | https://api.crossref.org/works/10.1007/978-3-031-30333-3_34 (metadata); https://link.springer.com/chapter/10.1007/978-3-031-30333-3_34 (paywalled) | **[fetched — metadata only; full text NOT retrieved (Springer 303/paywall). Its reported speedups (10.65% automatic vectorisation, 33.78% hand vectorisation) come from search-result summaries and are UNVERIFIED. Worth obtaining: it is the closest published study to our layout question.]** |

### Sources I looked for and could not get

- **uops.info** (`https://www.uops.info/html-instr/VGATHERQPD_YMM_VSIB_YMM_YMM.html` and
  the shuffle pages) was unreachable — `ECONNREFUSED` on 134.96.226.181 — so all
  instruction-level timings above come from Agner Fog [A1] rather than from independent
  microbenchmark corroboration. If uops.info comes back, cross-check the port assignments
  in §1.1; a second opinion on "shuffles are p5-only" would be worth having.
- **Franchetti & Püschel, "Generating SIMD Vectorized Permutations", CC 2008** — cited by
  [S1] as the source of the claim that complex de/interleave and the `ν × ν` in-register
  transpose "are the only in-register shuffles needed to implement FFTs". I could only
  retrieve a 3-page fragment from spiral.ece.cmu.edu, not the full 116–131 paper. The
  claim above is quoted *from [S1]*, not from CC 2008 itself. Getting the original would
  give a rigorous basis for "you only ever need these two shuffle networks".
- **A published head-to-head of split vs interleaved complex for a batched small FFT on
  AVX2/AVX-512.** [SA1] is the closest thing that exists and I could not read it. The
  argument in §2 is therefore assembled from a vectorisation-efficiency count [S1], a
  format-conversion benchmark [I1], two vendor preferences [AP1, I2], and my own port
  accounting — **not** from a direct published measurement of the thing we are about to
  build. Treat §2's verdict as strongly motivated but unproven, and settle it with a
  microbenchmark before committing all four sizes to one layout.
- **Goedecker's actual radix-2/3/4/5 FMA operation counts** [G1] — abstract only. Someone
  with SIAM access should extract the tables; they are directly relevant to `L = 6`
  (radix 2, radix 3) and `L = 36` (radix 4, radix 3).
- **Ice Lake / Sapphire Rapids server AVX-512 frequency data.** [D2] covers Ice Lake
  *client*; [C1] and [L1] cover Skylake *server*. I found no primary measurement of
  licence-based downclocking on Ice Lake-SP or later server parts, which is precisely the
  hardware most likely to be in an LQCD cluster today. This is a real gap — measure it on
  the target node.
