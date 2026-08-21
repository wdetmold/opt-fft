# 07 — Register/Shared-Memory-Level Fusion, and What the Accelerator World Knows That We Can Borrow

**Scope.** This section is about *where the data lives while you transform it*, not about
operation counts. Three questions:

1. What does the accelerator world actually do to keep a small transform on-chip, and what
   does it measurably buy? (cuFFTDx, VkFFT, shared-memory FFT kernels, fused FFT pipelines.)
2. Can you fuse all three axes of an `L^3` complex-double transform with no round trip to
   memory, and what is the exact capacity limit for that on a given register/L1 budget?
3. Twiddle handling and precision strategy — where the published error analysis actually
   constrains your choices at `L = 6, 8, 17, 36`.

**How to read the citations.** Every reference is tagged:

* `[F]` — I fetched this URL in this session and the quoted content came out of the fetch.
* `[F-via]` — I did *not* fetch this work, but its full bibliographic record (and, where
  quoted, the claim) came out of a `[F]` source's reference list or body. Treat the
  bibliographic data as good, the paraphrase as second-hand.
* `[UNVERIFIED — could not fetch]` — I could not retrieve it. Do not rely on it.

Anything with no citation tag at all lives in
[§7 Unsourced engineering notes](#7-unsourced-engineering-notes) and is my own arithmetic.

**Target hardware, measured on this machine** (not a citation — `lscpu` / sysfs on the node
this was written on):

```
Intel(R) Xeon(R) CPU E5-2680 v3 @ 2.50GHz   (Haswell-EP, 2 sockets x 12 cores)
ISA:  avx2, fma          <-- NO avx512 on this node
L1d:  32 KiB per core, 8-way, 64-byte lines
L2:   256 KiB per core
L3:   30 MiB per socket
```

So the "AVX-512 on some nodes" case is a *different* machine. The section gives both budgets
because, as §4 shows, the AVX2 and AVX-512 register files land on opposite sides of the
threshold for several of these sizes.

---

## 1. The accelerator design premise: a transform is a *device function*, not a library call

### 1.1 cuFFTDx — the FFT as a fragment you inline into your own kernel

NVIDIA's cuFFT Device Extensions exist for exactly one reason, stated plainly in the docs:

> "unlike cuFFT, cuFFTDx does not require moving data back to global memory after executing a
> FFT operation. This can be a major performance advantage as FFT calculations can be fused
> together with custom pre- and post-processing operations."
> — cuFFTDx, *First FFT Using cuFFTDx* `[F]` <https://docs.nvidia.com/cuda/cufftdx/introduction1.html>

The data model is explicit and is the part worth stealing:

* **Thread-level execution**: the whole transform lives in *one thread's registers*. The
  docs: "the input array should be in registers", sized by the `FFT::storage_size` trait.
  `[F]` <https://docs.nvidia.com/cuda/cufftdx/api/methods.html>
* **Block-level execution**: elements are distributed across threads (thread *n* holds
  elements `n + FFT::stride*i`), and `FFT::shared_memory_size` bytes of shared memory are
  used as the inter-thread exchange buffer. `[F]` (same page)
* `ElementsPerThread` "determines the number of registers required per thread and the exact
  implementation", and its default "is based on a heuristic which depends on the FFT itself
  (size, precision, GPU architecture etc.)"
  `[F]` <https://docs.nvidia.com/cuda/cufftdx/1.2.1/introduction2.html>

**The size limits are the interesting numbers**, because they are NVIDIA's own published
statement of "how big a transform fits in one thread's registers" vs "in one block's shared
memory". From the cuFFTDx requirements table `[F]`
<https://docs.nvidia.com/cuda/cufftdx/1.2.1/requirements_func.html>:

| execution scope | half/float | **double** |
| --- | --- | --- |
| **thread** (one thread's registers) | `[2; 64]` | **`[2; 32]`** |
| **block**, SM 75 | `[2; 4096]` | `[2; 2048]` |
| **block**, SM 70/72/86/89 | `[2; 16384]` | `[2; 8192]` |
| **block**, SM 80/87/90 | `[2; 32768]` | `[2; 16384]` |

Read the double-precision thread row again: **32 complex doubles is the largest transform
NVIDIA will put in a single thread's register file.** 32 complex doubles is 512 bytes. That
number recurs in §4 — it is almost exactly the AVX2 register file, and one quarter of the
AVX-512 register file.

Also note the shared-memory escape hatch: "Large FFTs may require more than 48 KB of shared
memory per CUDA block, and kernels with such FFTs must use dynamic shared memory rather than
statically sized shared memory arrays." `[F]` (per the requirements/methods docs above)

**Published performance claim.** cuFFTDx's own `convolution_performance` example compares a
single fused kernel against a 3-kernel cuFFT path and a 2-kernel cuFFT-callback path:

> "Depending on the device, the precision and the size of a given FFT the improvements from
> using cuFFTDx range from 45% to up to 3x speed-ups."
> `[F]` <https://docs.nvidia.com/cuda/cufftdx/1.2.1/examples.html>

Caveat the implementers should hold onto: that is a *forward FFT + pointwise + inverse FFT*
pipeline, i.e. the fusion removes two full global-memory round trips out of three passes.
It is not a claim about a bare forward transform.

**Fusing all three axes is a shipped example, and only for small cubes.** cuFFTDx ships
`fft_3d_cube_single_block` — "Small 3D (equal dimensions) FP32 FFT that fits into a single
block" — and `fft_3d_box_single_block` — "Small 3D FP32 FFT that fits into a single block,
each dimension is different". The docs describe both as using *thread-level* execution:
"cuFFTDx is used on a thread-level to executed small 3D FFTs in a single block".
`[F]` <https://docs.nvidia.com/cuda/cufftdx/1.2.0/examples.html>

That is the accelerator world's answer to "can you fuse `x`, `y`, `z`?": *yes, if the cube
fits in one block's on-chip storage, and NVIDIA only ships it for FP32 and only for small
cubes.* No published size ceiling is given for the 3D examples, but the thread-level limit
(32 double / 64 float per 1D transform) bounds what the per-axis inner call can be.

### 1.2 VkFFT — runtime kernel generation, and the memory-pass arithmetic stated as a factor

Tolmachev's VkFFT is the closest thing in the accelerator world to what this project is
doing by hand: generate exactly the code for exactly the size, at runtime, for exactly the
device. The SC22 poster is the primary source I could fetch in full `[F]`
<https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf>
(D. Tolmachev, EPM Group, ETH Zürich, *VkFFT — Vulkan/CUDA/HIP/OpenCL/Level Zero GPU FFT
library*, SC22 technical poster).

The single most quotable sentence for this project:

> "FFT is an extremely global memory bandwidth-limited algorithm, which means having two
> global memory data round trips instead of one often decreases performance by a factor of
> two." `[F]`

That is the whole justification for fusion, stated as a hard factor rather than a vibe. Its
CPU translation (§4) is: *your figure of merit is the number of times the volume crosses a
given cache boundary, and each avoided crossing is worth close to the full ratio of the
bandwidths involved.*

Concrete design points from the same poster `[F]`:

* "All FFT sequences that can fit in shared memory are done as a single upload from global
  memory. Including all advanced FFT algorithms, like Rader's/Bluestein's algorithms."
* Bigger sequences use "the four-step FFT algorithm — the sequence is split into two (three
  for even bigger sizes) uploads with an inlined twiddle multiplication."
* "VkFFT does multiple small FFTs at once – increases thread block size and allows to access
  continuous blocks of memory." — i.e. **batching is the vectorization strategy**, exactly as
  it must be for LQCD momentum projection.
* **VkFFT does not fuse the 3D axes.** "VkFFT does not have a transposition for the strided
  FFT axes – they are done by coalescing the neighboring sequences. Strided FFTs are done in
  a single upload/download, just like non-strided FFTs." So the best-in-class small-transform
  GPU library still does a 3D transform as *three* memory passes, one per axis, each pass
  individually on-chip-resident. Fusing axes is the cuFFTDx-style special case, not the
  production default.
* Zero padding: "it is possible to get up to two times the speed increase in the 2D case and
  up to 3x increase in the 3D case."
* Shared-memory capacity is the binding constraint and it is architecture-specific: "Due to
  the smaller size of available shared memory compared to A100 (64KB vs 192KB), VkFFT has to
  switch to double-upload scheme for Bluestein's algorithm after 2048 in FP64 and 4096 in
  FP32, **reducing effective bandwidth up to 2x**."

**The Rader threshold, and why `L = 17` is the canonical case.** From VkFFT's README `[F]`
<https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md>:

* "Radix-2/3/4/5/7/8/11/13 algorithms"
* "**Rader's FFT algorithm for primes from 17 up to max shared memory length (~10000)**"
* "Bluestein's FFT algorithm for all other sequences"
* "Single, double, half and quad (double-double) precision support." and "Double and quad
  precision uses CPU-generated LUT tables."

17 is *literally* the first prime for which VkFFT stops having an explicit radix kernel and
switches to Rader. And the SC22 poster's comparison against cuFFT says the same boundary
exists in NVIDIA's library, in a worse place `[F]`:

> "cuFFT only uses Rader's algorithm for primes up to 127 and implements it as a direct matrix
> multiplication." … "cuFFT does not use Rader's algorithm in FP32 and switches to Bluestein's
> algorithm for primes after 17. Rader's algorithm implementation in VkFFT works just as well
> in FP32 as in FP64." … "VkFFT uses FFT version of it for sequences decomposable as a
> multiplication of primes up to 4096 (if P-1 FFT can be done with the first algorithm)."

For `L = 17`, `P-1 = 16 = 2^4`, which is the friendliest possible case for Rader: the cyclic
convolution is a length-16 power-of-two transform. Both VkFFT and cuFFT reach for Rader here
rather than a hand-derived 17-point module, and VkFFT reports no FP32/FP64 accuracy penalty
for doing so.

**`registerBoost` — the GPU trick that does NOT transfer to CPU.** From VkFFT's own source
`[F]`
<https://raw.githubusercontent.com/DTolm/VkFFT/68b707955953aa50d997a47233b2d9726466cc30/lib/vkFFT.h>:

```c
uint32_t registerBoost = 1; //specify if register file size is bigger than shared memory (on Nvidia 256KB register file can be used instead of 32KB of shared memory, set this constant to 4)
```

On an NVIDIA SM the register file is **4× larger** than the usable shared memory, so the
optimizer's move is to push data *out of* shared memory *into* registers. §4 shows the CPU
ratio runs the other way by more than an order of magnitude. This is the single most
important asymmetry in this whole section.

### 1.3 Shared-memory-only FFT as a callable device function (Adámek et al.)

Adámek, Dimoudi, Giles & Armour, *GPU fast convolution via the overlap-and-save method in
shared memory*, ACM TACO **17**(3), Article 18 (2020) `[F]`
<https://arxiv.org/abs/1910.01972>, full text `[F]` <https://arxiv.org/pdf/1910.01972>.

This is the most honest published account of what it costs to keep an FFT on-chip, because
they had to write the FFT themselves for the same reason this project does:

> "The novelty of this work and its focus is to enable fast convolution by exploiting the
> fastest areas of GPU memory, registers and shared memory. To do this we needed to write FFT
> codes that will operate directly on data stored in shared memory (NVIDIA library functions
> do not do this)." `[F]`

Hard numbers `[F]`:

* Their FFT "deals only with short FFT lengths due to size limitation of the shared memory
  (currently N<=4096) and where N is a power of two."
* The fused kernel "works well only for small filter sizes M ≲ 3300 (for Titan V GPU). This
  limitation is imposed by the size of the GPU shared memory."
* GPUs used: P100, P4, Titan V. Table 1 lists "Max. sh. memory per thread-block: 48kB / 48kB
  / 48-96kB" and shared-memory bandwidths of 9121 / 2657 / 14550 GB/s against global-memory
  bandwidths of 720 / 192 / 652 GB/s — i.e. an on-chip:off-chip bandwidth ratio of roughly
  **12×–22×**.
* Speedups over the cuFFT-based overlap-and-save: "up to 2.5x faster for filter length 257
  samples for complex-to-complex (C2C) and up to 4x for real-to-real (R2R) convolution."
* **The cost of register residency**: "The occupancy … is only 50 %. This is a consequence of
  high register count used by the convolution kernel." And: "Our implementation of convolution
  uses registers to store the values of the signal segment and current filter value. **Further
  register utilization would lead to code slowdown.**"
* The fused kernel became *shared-memory-bandwidth* bound: "For R2R version the kernel
  utilizes around 75% of the shared memory bandwidth."
* On cuFFT's own callback/shared-memory route: "The cuFFT library also allows the user to use
  some shared memory. The amount is however limited to 16kB which can accommodate only 2048
  FFT elements".

The transferable lesson: **fusion converts a memory-bandwidth problem into a register-pressure
problem, and you can overshoot.** Their kernel stopped getting faster once register use cut
occupancy; the CPU analogue is spill traffic and reduced out-of-order lookahead.

### 1.4 Fusion accounted for honestly: TurboFNO

Wu, Zhai, Dai, Zhao, Zhu, Hu & Chen, *TurboFNO: High-Performance Fourier Neural Operator with
Fused FFT-GEMM-iFFT on GPU*, arXiv:2504.11681 (16 Apr 2025) `[F]`
<https://arxiv.org/pdf/2504.11681>.

Headline: "TurboFNO outperforms PyTorch, cuBLAS, and cuFFT by up to 150%" on an NVIDIA A100.
But the paper decomposes that number, and the decomposition is the useful part `[F]`:

* The FFT-side optimizations alone (pruning, truncation, zero-padding) give "up to 100%
  speedup over PyTorch, with an average speedup of 50%", from "our FFT pruning strategy, which
  reduces computation by 25%–67.5%".
* **Fusion on top of an already-optimized FFT is worth almost nothing**: "for 2D FNO, kernel
  fusion built on top of FFT optimization yields only a **3%–5%** additional speedup. In
  contrast, for 1D FNO where only the second FFT stage is present, TurboFNO's fused
  FFT-CGEMM-iFFT achieves up to a **10%** performance improvement".
* And it can go negative: "increasing the hidden dimension K from 32 to 128 leads to a gradual
  decline in the benefits of kernel fusion. For large hidden dimensions (K ≥ 128), fusion may
  even de[grade performance]".
* The mechanism they describe is exactly the one this project wants: "FFT outputs can be
  written directly into shared memory, forming the input tile for CGEMM. After CGEMM, the
  blocked output matrix C can remain in shared memory and be used immediately as input to the
  inverse FFT, **entirely bypassing global memory between stages**."

**Read this as the reality check on §2.** Fusion pays when the *unfused* baseline is
genuinely round-tripping through slow memory. If your unfused per-axis passes already hit in
L1/L2, fusing them buys single-digit percentages and costs you register pressure and code
size. Measure the baseline before believing the fused version will be 2× faster.

### 1.5 The maximal-fusion data point, with its caveat

Bergach, *From 8 Seconds to 370 ms: Kernel-Fused SAR Imaging on Apple Silicon via
Single-Dispatch FFT Pipelines*, arXiv:2604.03585v1 [cs.PF], 4 Apr 2026 `[F]`
<https://arxiv.org/html/2604.03585v1>.

Fuses "FFT → matched-filter multiply → IFFT" into one Metal dispatch on an Apple M1, keeping
intermediates on-chip. Numbers `[F]`: unfused baseline 8.16 s, fused 370 ms — **22.3×** — on
a 4096×4096 complex-float32 scene; the baseline used "three separate kernel launches with six
device-memory transfers per line (three reads, three writes)", the fused version two (one
read, one write). Threadgroup memory: "exactly 32 KiB (4096 × 8 bytes)" for N=4096, split
real/imaginary at "16 KiB" per component buffer; register file quoted as "208 KiB per SIMD
group". Their radix-8 Stockham kernel hit 138 GFLOPS scalar; an MMA/tensor-style
Cooley–Tukey variant reached "128 GFLOPS (93% of the scalar baseline)" — i.e. matrix units did
*not* help.

Caveat: 22.3× against a 3-launch/6-transfer baseline is a statement about the baseline's
memory traffic (6 transfers → 2 is 3×; the remaining 7× is launch overhead and per-kernel
inefficiency). Do not carry the number over; carry the *transfer count* accounting.

### 1.6 Synthesis of §1

The accelerator world's actual position, stripped of marketing:

1. Fusion's payoff is **the ratio of avoided memory passes times the bandwidth gap between the
   two memory levels involved** — nothing more (Tolmachev's "factor of two" for one avoided
   global round trip; Bergach's 6→2 transfers; Adámek's 12–22× on-chip:off-chip ratio).
2. **The capacity of the fast level sets a hard size ceiling** and everybody publishes it:
   N≤4096 shared-memory FFT (Adámek), 2048 FP64 before double-upload on 64 KB shared memory
   (VkFFT/MI250), 32 complex doubles per thread (cuFFTDx).
3. **You can overshoot into register pressure** (Adámek's 50% occupancy; TurboFNO's K≥128
   regression).
4. Runtime/compile-time specialization to *the exact size* is the enabling technique, and it is
   equally available on CPU — that is what FFTW's `genfft` and the FFTS library do (§3, §4).

---

## 2. Fusing all three axes: the capacity limit, stated properly

### 2.1 The theory is a register/cache lower bound, and it is tight

Frigo, *A Fast Fourier Transform Compiler*, PLDI '99 `[F]` <https://www.fftw.org/pldi99.pdf>
(Most Influential PLDI Paper award, 2009) states the bound this whole question turns on:

> "A lower bound on the number of register spills incurred by any execution of the FFT graph
> was first proved by Hong and Kung [HK81] in the context of the so-called 'red-blue pebbling
> game'. Paraphrased in compiler terminology, Theorem 2.1 from [HK81] states that the
> execution of the FFT graph of size n = 2^k on a machine with R registers (where R < n)
> requires at least Ω(n log n / log R) register spills." `[F]`
> (footnote: "The same result holds for any two-level memory, such as L1 cache vs. L2, or
> physical memory vs. disk.")

`[HK81]` = Jia-Wei Hong and H. T. Kung, "I/O complexity: the red-blue pebbling game", *Proc.
13th Annual ACM Symp. Theory of Computing*, pp. 326–333, Milwaukee, 1981. `[F-via]` (from the
pldi99 reference list, which I fetched).

And genfft attains it *without knowing R*:

> "It is perhaps surprising that a schedule exists that matches the asymptotic lower bound for
> all values of R. In other words, a single sequential order of execution of an FFT dag exists
> that, for all R, requires O(n log n / log R) register spills on a machine with R registers.
> We say that such a schedule is **cache-oblivious**." `[F]`

The FFTW3 paper restates the design consequence:

> "The goal of this phase is to find a schedule such that a C compiler can subsequently perform
> a good register allocation. The scheduling algorithm used by genfft offers certain
> theoretical guarantees because it has its foundations in the theory of cache-oblivious
> algorithms [35] (here, **the registers are viewed as a form of cache**). As a practical
> matter, one consequence of this scheduler is that FFTW's machine-independent codelets are no
> slower than machine-specific codelets generated by SPIRAL [43, Figure 3]."
> — Frigo & Johnson, *The Design and Implementation of FFTW3*, Proc. IEEE **93**(2), 216–231
> (2005) `[F]` <https://www.fftw.org/fftw-paper-ieee.pdf>

`[35]` = Frigo, Leiserson, Prokop & Ramachandran, "Cache-oblivious algorithms", *Proc. 40th
Ann. Symp. Foundations of Computer Science (FOCS '99)*, New York, Oct. 1999. `[F-via]`
`[43]` = Xiong, Padua & Johnson, "SPL: a language and compiler for DSP algorithms", PLDI 2001,
pp. 298–308. `[F-via]`

And the blunt reality about how far registers get you:

> "Even after simplification, a codelet dag of a large transform still contains hundreds or
> even thousands of nodes, and **there is no way to execute it fully within the register set of
> any existing processor.**" `[F]` (PLDI '99)

**What this gives the implementers.** The relevant regime split is:

* **If the whole volume `n = L^3` fits in the fast level (`n ≤ R`)**, the Hong–Kung bound
  degenerates: the optimal traffic is `n` loads plus `n` stores, full stop. Fusing all three
  axes is not merely a good idea, it is *the only optimal strategy* — any per-axis pass
  structure that re-reads the volume is strictly worse than optimal by a constant factor equal
  to the number of passes.
* **If `n > R`**, you are in the `Ω(n log n / log R)` regime and the question becomes tiling,
  not fusion. The right move is to fuse whatever *sub-tile* does fit, and Frigo's result says
  a single recursive order gets you the asymptotically right answer for every `R`
  simultaneously — so you do not need to tune the tile size to L1 vs L2 separately.

### 2.2 Applying the split to L = 6, 8, 17, 36

Complex double = 16 bytes. Volumes:

| L | points `L^3` | volume bytes | vs 32 KiB L1d (this node) | vs 256 KiB L2 | fully-fused 3-axis pass? |
| --- | --- | --- | --- | --- | --- |
| **6** | 216 | 3,456 B = **3.375 KiB** | 10.5% | 1.3% | **Yes, trivially. L1-resident with 29 KiB to spare.** |
| **8** | 512 | 8,192 B = **8 KiB** | 25% | 3.1% | **Yes. L1-resident.** |
| **17** | 4,913 | 78,608 B = **76.8 KiB** | **2.40×** — does not fit | 30% | No in L1; **yes in L2.** Fuse at plane granularity in L1. |
| **36** | 46,656 | 746,496 B = **729 KiB** | 23.3× | **2.85×** — does not fit | No in L1 or L2; L3-resident. Must tile. |

Out-of-place doubles every figure; in-place (or a single scratch plane) is worth real money at
L=17 and L=36.

Per-axis working sets (a single "pencil" and a single plane):

| L | 1 line (`L` cplx) | 1 plane (`L^2` cplx) |
| --- | --- | --- |
| 6 | 96 B | 576 B |
| 8 | 128 B | 1,024 B |
| 17 | 272 B | 4,624 B |
| 36 | 576 B | 20,736 B (63% of L1d) |

**Conclusions that fall straight out:**

* **`L = 6` and `L = 8` are the "cuFFTDx single-block cube" case on CPU.** The entire volume is
  3.4 KiB / 8 KiB. Load once, do `x`, `y`, `z` without letting it leave L1, store once. Three
  separate axis passes over an L1-resident array is *not* catastrophic (L1 is fast), but it is
  provably three times the optimal traffic and it forces three sets of loads/stores that the
  fused version can keep in registers across the axis boundary at the plane level.
* **`L = 17` is the interesting middle.** 76.8 KiB blows L1 by 2.4× but is only 30% of L2. So:
  do not try to fuse the whole cube. Fuse *two* axes over an L1-resident slab. A 17×17 plane is
  4,624 B; sixteen of them (17 planes = the whole volume) is the 76.8 KiB. A natural blocking
  is: bring in a `17 × 17 × k` slab, transform `x` and `y` fully inside L1, write back, then a
  second pass for `z` with a `k × 17` tile. Two passes instead of three, and both L1-resident.
* **`L = 36` needs real cache blocking and the PFA factorization helps twice.** 729 KiB is
  2.85× L2 on this node. Three naive axis passes = three L2↔L3 streams of 729 KiB each. Tile so
  that the `y` and `z` passes are fused over an L1-resident tile: a `36 × 8` tile of complex
  doubles is 4.5 KiB, a `36 × 36` plane is 20.7 KiB (63% of L1) — both viable. Because
  `36 = 4 · 9` with `gcd(4,9) = 1`, Good–Thomas/PFA removes the inter-factor twiddle
  multiplications entirely, which also removes the twiddle *table* from the L1 budget (§5).

### 2.3 What "fused" must actually mean on a CPU

On a GPU, "fused" means "one kernel launch, intermediates in shared memory". There is no CPU
equivalent of a kernel launch, so the CPU meaning is narrower and more precise:

1. **No store/reload of the volume between axes** — the `y` pass reads the `x` pass's output
   from cache (or from registers, at plane granularity), never from a level below.
2. **No separate transpose pass.** This is the one VkFFT calls out explicitly: "VkFFT does not
   have a transposition for the strided FFT axes – they are done by coalescing the neighboring
   sequences." `[F]` The CPU version of "coalescing neighbouring sequences" is: gather the
   strided pencil with SIMD loads across the *batch* dimension so that every load is
   contiguous, i.e. make the batch index the vector lane (§4.3). Then the strided axes cost
   the same as the contiguous one and you never materialize a transpose.
3. **Register-level fusion across axes only happens at plane granularity or smaller**, because
   (§4) no `L^3` from this set fits in any x86 register file.

---

## 3. Runtime/compile-time specialization: the CPU analogue already exists

VkFFT's contribution is not an algorithm, it is a *pipeline*: "Application → Plan → Code. This
allows to make code optimizations for the target device architecture at runtime." `[F]`
(SC22 poster; the poster's §4 describes the Application manager / Plan manager / Code manager
and Level 0/1/2 kernel layering, and notes: "it will be possible to inline VkFFT in the user's
kernels, reducing memory transfers even more.")

The CPU world got there first, twice:

* **FFTW's `genfft`** — an FFT compiler in OCaml that emits straight-line C. "The base cases of
  FFTW's recursive plans are its 'codelets' … They consist of long blocks of highly optimized,
  **straight-line code**". `[F]` (FFTW3, Proc. IEEE 2005)
* **FFTS** — Blake, Witten & Cree, "The Fastest Fourier Transform in the South", *IEEE Trans.
  Signal Processing* **61**(19), 4707–4716, Oct. 2013, DOI 10.1109/TSP.2013.2273199.
  `[UNVERIFIED — could not fetch]` (every route I tried returned 403). Reported to use runtime
  generation of specialized machine code after fixing sign/direction at plan time, and to beat
  FFTW, Intel IPP and Apple vDSP on x86 and ARM. **Do not cite the performance claim without
  reading the paper.**

**What the four target sizes get from this, concretely.** FFTW ships hard-coded no-twiddle
codelets for exactly these sizes — from FFTW3's own build file `[F]`
<https://raw.githubusercontent.com/FFTW/fftw3/master/dft/scalar/codelets/Makefile.am>:

```
N1  (no-twiddle):  n1_2 n1_3 n1_4 n1_5 n1_6 n1_7 n1_8 n1_9 n1_10 n1_11 n1_12
                   n1_13 n1_14 n1_15 n1_16 n1_32 n1_64 n1_20 n1_25
T1  (twiddle):     t1_2 t1_3 t1_4 t1_5 t1_6 t1_7 t1_8 t1_9 t1_10 t1_12 t1_15
                   t1_16 t1_32 t1_64 t1_20 t1_25
T2  (twiddle, precomputed):  t2_4 t2_8 t2_16 t2_32 t2_64 t2_5 t2_10 t2_20 t2_25
```

So:

* **`L = 6`** and **`L = 8`**: both a no-twiddle (`n1_6`, `n1_8`) *and* a twiddle
  (`t1_6`, `t1_8`) codelet exist. These are exactly the sizes the FFTW authors judged worth
  hard-coding as straight-line code. `L=8` also gets a `t2_8` precomputed-twiddle variant.
* **`L = 17`**: **no codelet.** The list stops at 16 (plus 20, 25, 32, 64). FFTW must reach a
  Rader or generic plan. The FFTW3 paper confirms the plan space: "FFTW contains one such
  codelet for each r ∈ {2, …, 16, 32, 64}" for twiddle codelets, and "Generic plans implement a
  naive Θ(n²) algorithm … Similarly, Rader plans implement" the Rader route. `[F]`
* **`L = 36`**: no monolithic codelet; would be built as e.g. 4×9 or 6×6 from the codelet set.

**How big does a straight-line codelet get?** Frigo's own datum `[F]` (PLDI '99):

> "the codelet that performs a DFT of size 64 is used routinely by FFTW on the Alpha processor.
> The codelet is about **twice as fast as Digital's DXML library** on the same machine. The
> codelet consists of about **2400 lines of code, including 912 additions and 248
> multiplications.** … At least for the DFT problem, these long sequences of straight-line code
> seem to be necessary in order to take full advantage of large CPU register sets and the
> scheduling capabilities of C compilers."

Size 64 complex, 2400 lines, 1160 flops, ~2× a vendor library. For reference, `L=36` sits below
that and `L^2 = 36` per plane is well inside it; a fully unrolled 36-point straight-line kernel
is squarely in the demonstrated-feasible range.

**And the compiler-interaction warning that costs the most performance in practice** `[F]`
(PLDI '99):

> "on a 167 MHz UltraSPARC I, the compiled code is **between 50% and 100% faster and about half
> the size** when this option is used. Inspection of the assembly code produced … reveals that
> **the difference consists entirely of register spills and reloads.**"

The option in question was declaring temporaries in the innermost lexical scope that spans
their lifetime rather than at the top of the function. Same generated arithmetic; up to 2×
performance difference from register allocation alone. Frigo also reports that gcc's *first*
instruction-scheduling pass "has the unfortunate effect of destroying genfft's schedule", and
that it must be disabled. **Both hazards apply verbatim to hand-written straight-line codelets
for L=6/8/17/36.**

**genfft's own arithmetic-strength choice, relevant to the FMA discussion in §6** `[F]`
(PLDI '99, footnote 6):

> "a complex multiplication by a constant can be implemented with either 4 real multiplications
> and 2 real additions, or 3 real multiplications and 3 real additions [Knuth, Exercise
> 4.6.4-41]. The current generator uses the former algorithm, since the operation count of the
> dag is generally dominated by additions. On most CPUs, it is advantageous to move work from
> the floating-point adder to the multiplier."

And a note on what specialization can find that humans miss `[F]`: for a complex transform of a
prime size, genfft used Rader's algorithm "in the form presented by Tolimieri and others
[TAL97]. In its most sophisticated variant, this algorithm performs **214 real (floating-point)
additions and 76 real multiplications** (See [TAL97, Page 161].) The generated code in FFTW for
the same algorithm, however, contains only **176 real additions and 68 real multiplications**",
while "[SB96] reports an algorithm with **188 additions and 40 multiplications**".
(*The exact prime is not reliably recoverable from the PDF's font encoding; it is a prime,
since genfft chose Rader.*)
`[TAL97]` = Richard Tolimieri, Myoung An, Chao Lu, *Algorithms for Discrete Fourier Transform
and Convolution*, Springer-Verlag, 1997. `[F-via]`
`[SB96]` = I. Selesnick and C. S. Burrus, "Automatic generation of prime length FFT programs",
*IEEE Transactions on Signal Processing*, pp. 14–24, January 1996. `[F-via]`
**Directly relevant to `L = 17`**: hand-derived Rader modules from the literature have been
beaten by ~18% in additions by a mechanical simplifier, and prime-length generators
(Selesnick–Burrus) exist specifically for this problem.

---

## 4. The CPU analogue: how much of a 6³ or 8³ complex-double transform fits where

### 4.1 The register-file arithmetic

Verified ISA facts: an AVX-512 `zmm` register is 512 bits and holds **8 double-precision
elements** — Intel's instruction reference for `VFMADD…PD` lists `(KL, VL) = (2,128), (4,256),
(8,512)` `[F]` <https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd> (a
mirror of the Intel SDM instruction reference). Register *counts* (16 `ymm` in AVX2, 32 `zmm`
in AVX-512) I could not pull from a primary Intel document in this session — the Intel
Optimization Reference Manual PDF exceeded the fetch size limit
(<https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf>,
`maxContentLength exceeded`). The counts below are therefore in the unsourced-notes category,
though they are not in dispute.

| | vector regs | bytes/reg | **total register file** | as complex doubles |
| --- | --- | --- | --- | --- |
| AVX2 (this node) | 16 `ymm` | 32 | **512 B** | **32** |
| AVX-512 | 32 `zmm` | 64 | **2,048 B** | **128** |

Compare against §1's landmarks:

* cuFFTDx's double-precision **thread-level** ceiling is 32 complex points = 512 B — *exactly
  the AVX2 register file*. NVIDIA's own judgement of "biggest transform worth keeping in one
  thread's registers" and the AVX2 register file coincide to the byte.
* VkFFT's `registerBoost` comment: NVIDIA register file 256 KB **vs** 32 KB shared memory, a
  ratio of **4:1 in favour of registers**.
* This CPU: 512 B register file vs 32,768 B L1d, a ratio of **1:64 against registers**
  (AVX-512: 1:16).

**This is the headline result of the section.** On a GPU, registers are the *large* fast
resource and shared memory is the small one, so the GPU optimization is "move data out of
shared memory into registers" (`registerBoost = 4`). On x86, registers are 16–64× *smaller*
than L1. **Therefore: the CPU unit of fusion is L1, not the register file.** Register-level
fusion on CPU means "fuse a pencil or a small tile", and L1-level fusion means "fuse the axes".
Any design that tries to be cuFFTDx — hold the transform in registers across all three axes —
is arithmetically impossible for every one of `L = 6, 8, 17, 36`.

### 4.2 Scalar-layout residency (one transform at a time, complex interleaved)

Data-only footprints against the register file (no temporaries counted):

| object | bytes | AVX2 (512 B file) | AVX-512 (2048 B file) |
| --- | --- | --- | --- |
| 6-point line | 96 | 3 of 16 `ymm` | 1.5 of 32 `zmm` |
| **6×6 plane** | 576 | **18 `ymm` — does not fit (16 exist)** | **9 of 32 `zmm` — fits, 23 spare** |
| 6³ volume | 3,456 | 6.75× the file | 1.69× the file — **does not fit** |
| 8-point line | 128 | 4 of 16 | 2 of 32 |
| **8×8 plane** | 1,024 | **32 `ymm` = 2× the file** | **16 of 32 `zmm` — fits, 16 spare** |
| 8³ volume | 8,192 | 16× | 4× — does not fit |
| 17-point line | 272 | 8.5 of 16 | 4.25 of 32 |
| 17×17 plane | 4,624 | 9× | 2.26× — does not fit |
| 36-point line | 576 | **18 `ymm` — does not fit** | 9 of 32 |
| 36×36 plane | 20,736 | 40× | 10× |

**Read-off for the implementers:**

* **On AVX-512, two of the three axes of a 6³ or an 8³ can be fused in registers.** A 6×6 plane
  is 9 `zmm` (23 free for temporaries — luxurious); an 8×8 plane is 16 `zmm` (16 free — enough
  for a radix-8 butterfly's temporaries). This is the real, defensible version of "fuse the
  axes in registers": *load a plane, do `x` and `y` entirely in registers, store the plane;
  then a `z` pass.* Two passes over an L1-resident volume, with the inner two axes never
  touching even L1 between them.
* **On AVX2 (this node), not even a 6×6 plane fits** (18 `ymm` needed, 16 exist), and a
  36-point *line* does not fit either. On AVX2 the honest unit is the **pencil**: one
  `L`-point line in registers, all three axes done as three pencil-wise passes over an
  L1-resident volume. Attempting plane-level register fusion on AVX2 will spill, and Frigo's
  UltraSPARC datum (50–100% slower, purely from spills) is the price tag.
* **`L = 36` on AVX-512 is saved by PFA.** A monolithic 36-point line is 9 `zmm` of data but a
  radix-36 straight-line kernel needs far more live temporaries. With Good–Thomas `36 = 4 · 9`,
  the largest simultaneously-live module is the 9-point one: 9 complex = 4.5 `zmm` of data,
  and a 9-point Winograd/PFA module's temporaries fit comfortably. The 4-point module needs
  2 `zmm`. **PFA is a register-pressure optimization here, not only a twiddle-count
  optimization.** Same argument for `6 = 2 · 3`: the 3-point module is 1.5 `zmm`.

### 4.3 Batched layout — the layout you actually want for LQCD, and what it costs

The workload is "many volumes batched … over time slices / spin-colour components". The
FFTW authors' own SIMD strategy tells you how to use that `[F]` (FFTW3, Proc. IEEE 2005,
§IX "How FFTW3 uses SIMD"):

> "consider first a machine with length-2 vectors, such as the Pentium IV using the SSE2
> instruction set … We view a complex DFT as a pair of real DFTs: `DFT(A + i·B) = DFT(A) +
> i·DFT(B)` … Our algorithm computes the two real DFTs in parallel using SIMD instructions"
>
> "**On machines that support vectors of length 4, we view SIMD data as vectors of two complex
> numbers, and each codelet executes two iterations of its loop in parallel.** … The source of
> this 2-way parallelism is the codelet loop, which can arise from the Cooley-Tukey
> decomposition of a single 1d DFT, the decomposition of a multi-dimensional DFT, or a
> user-specified vector loop."
>
> "Four-way SIMD instructions are problematic, because the input or the output are not
> generally stride-1, and arbitrary-stride SIMD memory operations are more expensive than
> stride-1 operations."
>
> "we are not aware of any [auto-vectorizing compiler] that is effective at vectorizing FFTW,
> nor indeed of any automatically vectorized code that is competitive on these 2-way and 4-way
> SIMD architectures."

Translated to AVX2/AVX-512 doubles (4-way / 8-way): the vectorization source should be **the
batch loop**, exactly as VkFFT does it on GPU ("VkFFT does multiple small FFTs at once" `[F]`).
Put the batch index in the vector lane, in split real/imaginary form, and every load is
stride-1 and every axis (including the strided ones) becomes contiguous — which is precisely
VkFFT's "coalescing the neighboring sequences" trick with no transpose.

**The cost: one complex point now occupies two full vector registers.**

| | 1 complex point | 6-line | 8-line | 17-line | 36-line | Rader's 16-pt inner conv (L=17) |
| --- | --- | --- | --- | --- | --- | --- |
| **AVX2**, 4 batches/lane | 2 `ymm` | 12 of 16 | **16 of 16 (zero spare)** | 34 (2.1×) | 72 (4.5×) | 32 (2×) |
| **AVX-512**, 8 batches/lane | 2 `zmm` | 12 of 32 | 16 of 32 | 34 (1.06×) | 72 (2.25×) | 32 of 32 (zero spare) |

**Read-off:**

* **`L = 6`, batch-vectorized: comfortable everywhere.** 12 of 16 `ymm` on AVX2 leaves 4
  scratch registers — tight for a full radix-6 but fine for PFA 2×3 (the 3-point module needs
  6 data + ~4 temporaries). On AVX-512, 12 of 32 is roomy: you can hold a whole 6-line and
  still have 20 `zmm` for a fully unrolled straight-line radix-6.
* **`L = 8`, batch-vectorized on AVX2: exactly fills the register file, zero temporaries.** A
  fully unrolled radix-8 straight-line codelet *will* spill on AVX2 with 4-wide batching. Two
  ways out: (a) batch 2-wide instead of 4-wide (8 `ymm` for data, 8 spare) and take the halved
  SIMD width; (b) split into radix-4 then radix-2 (or two radix-2 stages then radix-2) so that
  only a subset of the 8 points is live at once, accepting an intermediate store to L1. On
  AVX-512, 16 of 32 makes a fully unrolled radix-8 with 8-wide batching genuinely
  register-resident — this is `L=8`'s best case and it is a real reason to prefer the AVX-512
  nodes for `L=8`.
* **`L = 17`, batch-vectorized: does not fit either register file as a whole line.** But
  Rader's route decomposes it: the length-16 cyclic convolution needs 16 complex = 32 registers,
  which is exactly the AVX-512 file (no temporaries) and 2× the AVX2 file. So the 16-point
  convolution must itself be staged (e.g. as radix-4 × radix-4, holding 4 complex = 8 registers
  live per stage) with the intermediate in L1. The `x+y` plane (289 complex) is L1-resident at
  any batch width ≤ 7 on this node; batch-4 gives 18.5 KiB per plane, a comfortable 56% of L1.
* **`L = 36`, batch-vectorized: 72 registers for a line — 2.25× even AVX-512.** PFA 4×9 is
  essentially mandatory: the 9-point stage needs 18 registers (of 32 on AVX-512, of 16 on
  AVX2 — so AVX2 needs the 9-point module further decomposed or a narrower batch). The 4-point
  stage needs 8.

**Whole-volume L1 residency under batching**, against this node's 32 KiB L1d:

| L | ×1 | ×2 | ×4 (AVX2 lane width) | ×8 (AVX-512 lane width) |
| --- | --- | --- | --- | --- |
| 6 | 3.4 KiB | 6.8 KiB | **13.5 KiB (42%)** | **27 KiB (84%, tight)** |
| 8 | 8 KiB | 16 KiB | **32 KiB — exactly L1, will thrash** | 64 KiB (L2) |
| 17 | 76.8 KiB | 154 KiB | 307 KiB (past L2) | 614 KiB (L3) |
| 36 | 729 KiB | 1.46 MiB | 2.9 MiB | 5.8 MiB |

So: **`6³` batched 4-wide is a fully L1-resident, fully-fusable 3D transform on this node** —
the single sweetest configuration in the whole target set. `8³` batched 4-wide sits exactly at
L1 capacity, which in an 8-way 64-set cache with any other live data means conflict misses;
batch 2-wide (16 KiB) or accept L2 for the outer axis. `17³` and `36³` must be tiled regardless
of batch width; batch at *plane* or *slab* granularity, not at volume granularity.

### 4.4 What SIMD vectorization of small transforms actually delivers

Franchetti & Püschel, *Short Vector Code Generation and Adaptation for DSP Algorithms* `[F]`
<https://users.ece.cmu.edu/~franzf/papers/icassp03.pdf> — SPIRAL-generated SSE/SSE2 code
(venue: ICASSP 2003 per the author's file naming and secondary listings; **the PDF itself
carries no venue line**, so treat the venue as inferred).

> "SPIRAL generated vector code achieves excellent speed-ups over the fastest scalar codes on
> all platforms for all considered transforms. For real four-way extensions (SSE on Pentium III
> and Pentium 4) we achieve **up to a factor of 3.3**, and for two-way extensions (including
> SSE on the Athlon XP, which is implemented on top of 3DNow!) we achieve **up to a factor of
> 1.8**." `[F]`

Also `[F]`: "Using C compiler vectorization in tandem with SPIRAL C code generation (SPIRAL C
vect) in general improves performance, but is far from being optimal. (As an aside, **compiler
vectorization of FFTW does not improve its performance.**)" And on specialization: cross-timing
the best algorithm found for one platform on another gave slowdown factors up to ~4×, and even
between two SSE-capable x86 generations "the SSE optimized formulas, found for Pentium III or
Athlon XP, performed up to 1.6 times slower than the Pentium 4 SSE adapted formulas."

**Calibration for this project:** 4-way SIMD returned 3.3/4 = 83% of theoretical, 2-way
returned 1.8/2 = 90%. Scaling that efficiency curve down: expect AVX2 4-way double to land near
3.2–3.4× scalar, and AVX-512 8-way somewhere in 5.5–6.5× scalar *if* register pressure stays
under control (which §4.3 says it will for `L=6` and `L=8` and will not for a naive `L=36`).
Do not expect the compiler to find this: FFTW's authors report auto-vectorization did not help
their code at all.

---

## 5. Twiddle factors: precomputed tables vs recurrence vs on-the-fly trig

### 5.1 The published error analysis is unambiguous

FFTW's accuracy documentation is the most compact primary-adjacent statement `[F]`
<https://www.fftw.org/accuracy/comments.html>:

> "**Inaccurate twiddle factors are the most likely reason for the inaccuracy of an FFT
> routine.**"
>
> "A trigonometric recurrence is a trick that many codes use to avoid the time and memory
> overhead of precomputing and storing an array of accurate twiddle factors." … "most such
> recurrence formulas accumulate errors as **O(√N), O(N), or even O(N²)**, much faster than the
> FFT itself" (attributed there to Tasche & Zeuner, 2002).
>
> Cooley–Tukey floating-point error grows as "**O(log N) in the worst case** (Gentleman & Sande,
> 1966) and as **O(√log N) on average** (Schatzman, 1996)."
>
> "Even inaccurate recurrence formulas can produce accurate results if they are implemented in
> higher precision" (Kahan, 2001).
>
> A Buneman-style variant achieves "**O(√log N) mean accuracy (like the FFT itself) via a
> precomputed table of O(log N) twiddle factors**."

Full records, taken from that page's reference list `[F]` (so bibliographic data is verified,
the papers themselves were not read):

* W. M. Gentleman & G. Sande, "Fast Fourier transforms—for fun and profit", *Proc. AFIPS*
  **29**, 563–578 (1966). `[F-via]`
* O. Buneman, "Stable online creation of sines or cosines of successive angles", *Proc. IEEE*
  **75**(10), 1434–1435 (1987). `[F-via]`
* M. Tasche & H. Zeuner, "Improved roundoff error analysis for precomputed twiddle factors",
  *J. Computational Analysis and Applications* **4**(1), 1–18 (2002). `[F-via]`
* W. Kahan, "How Java's floating-point hurts everyone everywhere" (2001),
  `http://www.cs.berkeley.edu/~wkahan/JAVAhurt.pdf`. `[F-via]`

And the primary source for the recurrence warning, whose abstract I did fetch — J. C.
Schatzman, "Accuracy of the Discrete Fourier Transform and the Fast Fourier Transform", *SIAM
J. Sci. Comput.* **17**(5), 1150–1166 (1996), DOI 10.1137/S1064827593247023 `[F]`
<https://epubs.siam.org/doi/10.1137/S1064827593247023>:

> "However, these results depend critically on the accuracy of the FFT software employed, which
> should generally be considered suspect. **Popular recursions for fast computation of the
> sine/cosine table (or twiddle factors) are inaccurate due to inherent instability. Some
> analyses of these recursions that have appeared heretofore in print, suggesting stability,
> are incorrect.** Even in higher dimensions, the FFT is remarkably stable."

The FFTW3 paper confirms the target error scaling for *multi-dimensional* transforms too:
"All of our methods are observed to achieve the same **O(√log n) L² error** as the Cooley-Tukey
FFT [59]" `[F]`, where `[59]` is Schatzman 1996. FFTW's accuracy benchmark defines the measure
`[F]` <https://www.fftw.org/accuracy/method.html>: forward error `compare(FFT(x), exactFFT(x))`
with `compare(a,b) = ||a−b||ₙ / ||b||ₙ`, against "an arbitrary-precision arithmetic FFT (with
> 40 decimal places of accuracy)", on "uniform pseudorandom numbers in [-0.5,0.5)".

### 5.2 What VkFFT actually does about it, and what it measured

From the SC22 poster's precision-verification panel `[F]`:

> "VkFFT precision is verified by comparing its results with FP128 version of FFTW. We test all
> FFT lengths from the [2, 100000] range. … For both precisions, all tested libraries exhibit
> logarithmic error scaling. **The main source of error is imprecise twiddle factor computation
> – sines and cosines used by FFT algorithms.** For FP64 they are calculated on the CPU either
> in FP128 or in FP64 and stored in the lookup tables. **With FP128 precomputation VkFFT is
> more precise than cuFFT and rocFFT.** For FP32, twiddle factors can be calculated on-the-fly
> in FP32 or precomputed in FP64/FP32. With FP32 twiddle factors VkFFT is slightly less precise
> in Bluestein's and Rader's algorithms. If needed, this can be solved with FP64
> precomputation."

And from the README `[F]`: "Double and quad precision uses CPU-generated LUT tables."

The whole design, stated as a rule: **compute the twiddles once, in higher precision than you
will use them, on the host, and store them in a table.** VkFFT beats both vendor libraries in
FP64 accuracy purely by precomputing in FP128. FFTW's `genfft` does the identical thing at
compile time: "The structure Number maintains floating-point constants with arbitrarily high
precision (currently, **50 decimal digits**), in case the user wants to use the quadruple
precision floating-point unit" `[F]` (PLDI '99).

### 5.3 For L = 6, 8, 17, 36 the tradeoff does not exist — tables are free

The reason recurrences were ever tempting is that a length-`N` Cooley–Tukey twiddle table is
`O(N)`. At `N = 2^20` that is 16 MB and you would rather recompute. **At these sizes the tables
are so small that "precomputed vs recurrence" is not a real decision.** Per-axis distinct
twiddle constants (my arithmetic — see §7):

| L | factorization | twiddle scheme | distinct nontrivial constants | table bytes | splatted ×8 for AVX-512 |
| --- | --- | --- | --- | --- | --- |
| 6 | 2·3, coprime | **PFA — no inter-factor twiddles at all** | cos/sin(2π/3) → 2 reals | ~16 B | ~128 B |
| 8 | 2³ | radix-8 | 1/√2 only | ~8 B | ~64 B |
| 17 | prime | Rader: length-16 cyclic convolution | DFT of the permuted twiddle seq: 16 complex | 256 B | 2 KiB |
| 36 | 4·9, coprime | **PFA — no inter-factor twiddles**; 9-pt module | cos/sin(2πk/9), k=1..4 → 8 reals | ~64 B | 512 B |

**Total across all four sizes, including the ×8 splatted forms: under 3 KiB.** A cube reuses the
same length-`L` table for all three axes, so there is no 3D multiplier.

**Therefore, for this project:**

1. **Use compile-time constants.** Not a runtime table, not a recurrence, not `sincos`. Emit
   the constants as `static const double` (or pre-splatted `static const double[8]`
   `__attribute__((aligned(64)))`) computed offline to ≥30 significant digits and rounded once.
   This is FFTW's `genfft` approach and VkFFT's FP128-LUT approach, collapsed to the case where
   the table is small enough to be immediate data.
2. **Never use a trigonometric recurrence.** It saves nothing (the table is bytes) and costs you
   `O(√N)`–`O(N²)` error growth on top of the transform's own `O(log N)` (Tasche & Zeuner via
   `[F]` FFTW accuracy page; Schatzman `[F]`).
3. **Never call `sin`/`cos` inside the transform.** Besides being a library call (which the brief
   forbids anyway), an on-the-fly `sincos` in FP64 is both slower than a load from L1 and no more
   accurate than a correctly-rounded stored constant.
4. **PFA earns its keep twice at `L=6` and `L=36`**: it removes the twiddle multiplications
   (arithmetic) *and* it removes the twiddle constants from the register/L1 budget, freeing
   registers for the butterfly temporaries that §4.3 shows are the binding constraint.
5. **`L=17` Rader is the only size with a real table** (16 complex = 256 B). That table is the
   DFT of a permuted twiddle sequence, so it must be precomputed in extended precision and
   rounded once — this is exactly the case where VkFFT reports a measurable FP32 accuracy
   deficit "in Bluestein's and Rader's algorithms" from low-precision twiddles `[F]`. Compute it
   offline in `long double`/quad or symbolically.

---

## 6. Precision strategy

### 6.1 Complex multiplication: the error bounds are known exactly, and FMA wins

**Without FMA.** Brent, Percival & Zimmermann, "Error bounds on complex floating-point
multiplication", *Mathematics of Computation* **76**(259), July 2007, pp. 1469–1481,
S 0025-5718(07)01931-X `[F]` <https://maths-people.anu.edu.au/~brent/pd/rpb221.pdf>. Abstract,
verbatim:

> "Given floating-point arithmetic with t-digit base-β significands in which all arithmetic
> operations are performed as if calculated to infinite precision and rounded to a nearest
> representable value, we prove that the product of complex values z₀ and z₁ can be computed
> with maximum absolute error `|z₀||z₁| ½ β^{1−t} √5`. In particular, this provides relative
> error bounds of `2^{−24}√5` and `2^{−53}√5` for IEEE 754 single and double precision
> arithmetic respectively, provided that overflow, underflow, and denormals do not occur."

Theorem 1 is stated for the textbook form
`z₂ = ((a₀⊗a₁) ⊖ (b₀⊗b₁)) + ((a₀⊗b₁) ⊕ (b₀⊗a₁))i`, i.e. **4 multiplications + 2 additions**,
with the bound `ε√5` where `ε = ½β^{1−t}`. The paper also notes `[F]`: "Since the bound of
`ε√8` which is commonly used [1] is suboptimal, we present here a corrected proof of the tighter
bound", and that the `√5` bound is "effectively optimal" (they exhibit worst-case inputs).

**With FMA.** Jeannerod, Kornerup, Louvet & Muller, "Error bounds on complex floating-point
multiplication with an FMA", *Mathematics of Computation* **86**(304), pp. 881–898 (2017), DOI
10.1090/mcom/3123 `[F]`
<https://portal.findresearcher.sdu.dk/en/publications/error-bounds-on-complex-floating-point-multiplication-with-an-fma/>
(bibliographic record and abstract fetched; the full paper was blocked at HAL and AMS). The
abstract's result:

* The Brent–Percival–Zimmermann bound of "√5 u on the normwise relative error" reduces to
  "**2u** when using the FMA in the most naive way".
* That `2u` bound is **asymptotically optimal** across three algorithms — the naive FMA form and
  two compensated variants that use FMA for rounding-error compensation.
* Crucially: the compensated algorithms achieve high *componentwise* accuracy but give **no
  normwise improvement** over the naive `2u`.

**FMA semantics.** Intel's instruction reference for `VFMADD…PD` `[F]`
<https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd> describes the multiply-add
as computed with "infinite precision inputs and outputs (no rounding)" internally, with a single
final rounding via `RoundFPControl_MXCSR()` — **the intermediate product is not rounded**. That
is exactly the property that buys `√5u → 2u`.

**Actionable conclusions:**

1. Use the **4-multiply / 2-add** complex product, expressed as 2 multiplies + 2 FMAs:
   `t = a*c; re = fma(-b, d, t); u = a*d; im = fma(b, c, u)`. Error bound `2u`, optimal, and it
   maps onto 4 FMA-class instructions.
2. **Do not use the 3-multiply Karatsuba/Knuth complex product** (Knuth, *TAOCP* vol. 2,
   Exercise 4.6.4-41, cited by FFTW `[F]`). FFTW's own reasoning already rejected it on
   throughput grounds — "it is advantageous to move work from the floating-point adder to the
   multiplier" `[F]` (PLDI '99) — and on FMA hardware it is worse again: it trades 1 multiply
   for 3 additions that cannot be folded into FMAs, and it has no `2u` guarantee.
3. **Do not bother with compensated complex multiplication.** Jeannerod et al. say explicitly it
   buys nothing normwise `[F]`. Compensation is only worth considering if you need
   *componentwise* accuracy, which momentum projection does not.
4. **Be deliberate about FMA contraction rather than leaving it to the compiler.** Whether
   `a*b + c` becomes an FMA is a compiler-flag question (`-ffp-contract`), it changes results
   bit-for-bit, and it changes them *differently* on different compilers and different
   optimization levels. Since the whole point of this project is hand-written straight-line
   codelets, write the FMAs as intrinsics (`_mm256_fmadd_pd` / `_mm512_fmadd_pd`) so the
   generated arithmetic is fixed and the `2u` bound is guaranteed rather than hoped for.
   (Engineering judgement — see §7.)

### 6.2 Error growth at these sizes: you have enormous headroom

Combining §5.1's growth laws with the actual sizes (my arithmetic, §7): a 3D `L^3` transform is
three independent 1D passes, so the total relative error is bounded by roughly 3× the 1D bound.

| L | log₂L | worst-case per axis, `O(log N)` | 3 axes, worst case | average, `O(√log N)` × 3 |
| --- | --- | --- | --- | --- |
| 6 | 2.58 | ~2.6 u | ~8 u ≈ 9 × 10⁻¹⁶ | ~4.8 u ≈ 5 × 10⁻¹⁶ |
| 8 | 3 | ~3 u | ~9 u ≈ 1 × 10⁻¹⁵ | ~5.2 u ≈ 6 × 10⁻¹⁶ |
| 17 | 4.09 | ~4.1 u | ~12 u ≈ 1.4 × 10⁻¹⁵ | ~6.1 u ≈ 7 × 10⁻¹⁶ |
| 36 | 5.17 | ~5.2 u | ~16 u ≈ 1.7 × 10⁻¹⁵ | ~6.8 u ≈ 8 × 10⁻¹⁶ |

(`u = 2⁻⁵³ ≈ 1.11 × 10⁻¹⁶`. Constants in the `O(·)` are not pinned down by the sources, so treat
these as order-of-magnitude.)

**The transforms are so short that accuracy is essentially a non-issue in double precision** —
provided the twiddle constants are exact-to-rounding (§5). All the accuracy risk at these sizes
lives in the constants, not in the butterfly network. This is the direct payoff of Schatzman's
finding that "the FFT is remarkably stable" but "the accuracy of the FFT software … should
generally be considered suspect" `[F]`.

The same table also says: **single precision (`u ≈ 6 × 10⁻⁸`) would give ~10⁻⁶–10⁻⁷ relative
error** for all four sizes, and buy exactly 2× the SIMD lane count (4→8 on AVX2, 8→16 on
AVX-512) and half the cache footprint — which would move `8³` batched-4 from 32 KiB (exactly L1,
thrashing, §4.3) to 16 KiB, and `17³` from 76.8 KiB to 38.4 KiB (still over L1 but much
friendlier). **If the physics tolerates 10⁻⁶, the single-precision variant is worth building**;
the brief specifies double, so this is a note for the mixed-precision discussion, not a
recommendation.

### 6.3 Do not try to be more accurate than double

Kawakami & Takahashi, *Computing FFTs at Target Precision Using Lower-Precision FFTs*,
arXiv:2603.29129 (31 Mar 2026) `[F]` <https://arxiv.org/abs/2603.29129>. They apply the Ozaki
scheme to the cyclic convolution inside a Bluestein FFT, computing the split-component
convolutions exactly with number-theoretic transforms plus CRT. From the abstract, verbatim:

> "we implement a double-precision FFT using 32-bit NTTs and confirm reduced relative error
> compared with those for FFTs based on FFTW and Triple-Single precision arithmetic, with stable
> error across FFT lengths, at most 96 NTT calls, or 64 NTT calls with NTT-domain accumulation.
> On an Intel Xeon Platinum 8468 for lengths n=2^10-2^18, the execution time is approximately
> **107–1315× that of FFTW's double-precision FFT**, with NTTs accounting for approximately 80%
> of the total time."

**The number to remember is 107–1315×.** Exactly-rounded / error-free FFT machinery is a
research tool, not a performance option. Given §6.2 (double-precision error already ~10⁻¹⁵ at
these sizes), there is no case for it here at all.

### 6.4 Compensated summation

I found **no** primary source in this session giving an error analysis of compensated
(Kahan/Neumaier) summation *applied to an FFT butterfly network*. What the sources do say:

* Compensation applied to the complex *product* gives no normwise gain over naive FMA
  (Jeannerod et al. `[F]`).
* Recurrence-generated twiddles can be rescued by higher-precision evaluation (Kahan 2001, via
  FFTW `[F]`) — but §5.3 says do not use recurrences in the first place.

An FFT butterfly is not a long accumulation — the longest dependent addition chain in a radix-`r`
codelet is `O(log r)` deep — so the mechanism compensated summation exists to fix (long
sequential accumulation) is largely absent. Treating this as an open question rather than a
recommendation: **I would not spend anything on compensated summation for `L ≤ 36`**, and I have
no source that would justify doing so. (Engineering judgement — §7.)

---

## 7. Unsourced engineering notes

Everything in this section is my own arithmetic or judgement, attributed to nobody. It is
separated out deliberately.

**7.1 The register/cache arithmetic.** Every byte count in §2.2, §4.1, §4.2, §4.3, §5.3 and
§6.2 is elementary arithmetic from (a) complex double = 16 bytes, (b) the measured cache sizes
at the top of this file, (c) 16×32-byte `ymm` / 32×64-byte `zmm` register files. The `zmm`
width and its 8-double capacity are sourced `[F]`; the *counts* (16 / 32) are not — I could not
fetch a primary Intel document within the fetch size limit. They are not controversial, but
verify against the Intel Optimization Reference Manual before quoting them in a paper.

**7.2 Recommended residency plan per size.** Synthesized, not sourced:

* **`L = 6`**: batch 4-wide (AVX2) or 8-wide (AVX-512) in split re/im, batch index in the lane.
  Whole batched volume L1-resident (13.5 / 27 KiB). PFA 2×3. On AVX-512 fuse `x` and `y` at
  plane granularity in registers (9 `zmm` of data); on AVX2 fuse at pencil granularity
  (12 of 16 `ymm`). One load of the volume, one store.
* **`L = 8`**: AVX-512 preferred — 8-wide batching puts a whole 8-line in 16 of 32 `zmm`, and an
  8×8 plane in 16 `zmm` scalar-layout. On AVX2, drop to 2-wide batching (16 KiB volume, 8 of 16
  `ymm` per line) or split radix-8 into 4×2. Radix-8's only nontrivial constant is `1/√2`.
* **`L = 17`**: Rader with the length-16 cyclic convolution staged as 4×4 so no stage needs more
  than ~8–16 vector registers. Precompute the 16-complex convolution kernel offline in extended
  precision. Block the volume as `17 × 17 × k` slabs so that the `x`+`y` fusion happens inside
  L1 (a 17×17 plane batched 4-wide is 18.5 KiB = 56% of L1); `z` in a second pass.
* **`L = 36`**: PFA 4×9 (coprime, no inter-factor twiddles, and the 9-point module is the
  register-pressure bound at 18 vector registers under 8-wide batching). Volume is 729 KiB —
  2.85× L2 on this node — so three-level blocking: L3-resident volume, L2-resident plane slab,
  L1-resident `36 × 8` tile (4.5 KiB scalar / 18 KiB batched-4). Fuse `y` and `z` over the L1
  tile; `x` is contiguous and cheap.

**7.3 Twiddle constant counts** in §5.3's table are derived from the factorizations, not quoted
from a source. Sanity-check them against whichever section of this corpus does the operation
counts.

**7.4 SIMD-efficiency extrapolation** (§4.4: "expect 3.2–3.4× on AVX2, 5.5–6.5× on AVX-512") is
my extrapolation of Franchetti & Püschel's measured 83%/90% of theoretical on 4-way/2-way SSE
onto 4-way/8-way AVX. It is a plausibility estimate, not a measurement, and it assumes register
pressure stays under control.

**7.5 FMA intrinsics vs `-ffp-contract`.** Whether the compiler contracts `a*b+c` into an FMA
depends on flags, language mode and optimization level, and it changes results bit-for-bit. For
straight-line hand-written codelets, write the intrinsics. Unsourced.

**7.6 Compensated summation** — see §6.4. My judgement: skip it. Unsourced.

**7.7 Haswell FMA throughput.** This node has two FMA-capable ports and 256-bit vectors, i.e.
16 double-precision FLOP/cycle peak. AVX-512 parts with two FMA units reach 32/cycle. I did not
fetch a primary source for either; do not quote these.

**7.8 Register spill detection.** Frigo's UltraSPARC datum (§3: 50–100% faster from lexical
scoping alone, "entirely register spills and reloads") is sourced `[F]`; the *practice* that
follows is not: build the codelets, then check the generated assembly for stack traffic
(`grep -c 'rsp' `, or count `vmovupd` against `%rsp`/`%rbp`) before believing any timing. For
these sizes the spill count is a better early proxy for quality than a flop count.

---

## 8. Gaps — what I looked for and could not find

1. **The VkFFT journal paper itself.** D. Tolmachev, "VkFFT — A Performant, Cross-Platform and
   Open-Source GPU FFT Library", *IEEE Access* **11**, 12039–12058 (2023), DOI
   10.1109/ACCESS.2023.3242240. `[UNVERIFIED — could not fetch]`: IEEE Xplore, DOAJ, SciSpace
   and Semantic Scholar all returned 403/empty. Everything I attribute to Tolmachev here comes
   from the **SC22 poster** and the **GitHub README/source**, both of which I did fetch. The
   journal paper reportedly contains fuller precision and performance data; someone with
   institutional access should read it. Same for "VkFFT and beyond — a platform for runtime GPU
   code generation", *Proc. 2023 Int'l Workshop on OpenCL*, DOI 10.1145/3585341.3585357
   (ACM returned 403).
2. **FFTS (Blake, Witten & Cree 2013).** The one published CPU library built on runtime code
   generation — the closest existing thing to what this project is doing. Could not fetch
   (403 everywhere). Its register-allocation and base-case-sizing choices would be the single
   most directly applicable prior art in the whole section.
3. **A primary Intel source for register counts and per-microarchitecture L1D sizes.** The
   Optimization Reference Manual PDF exceeds the fetch size limit. §4.1's register counts are
   therefore unsourced.
4. **`cuFFTDx` 3D size ceilings.** The docs describe `fft_3d_cube_single_block` but publish no
   maximum cube edge, and the examples are FP32 only. The double-precision 3D single-block limit
   — the number that would most directly bound the "fuse all three axes" question on GPU — is not
   documented anywhere I could find.
5. **Published per-thread register limits for CUDA compute capabilities.** The CUDA Programming
   Guide's technical-specifications table would not render through WebFetch (only the ToC came
   back). VkFFT's source comment (256 KB register file vs 32 KB shared memory) is the only
   register-budget figure here that I could verify.
6. **Error analysis of compensated summation inside an FFT butterfly network.** No primary source
   found (see §6.4).
7. **A measured CPU study of axis-fusion for small 3D FFTs.** I found nothing that measures
   "three separate L1-resident axis passes" against "one fused pass" for cubes in the
   `6³ … 36³` range on x86. The GPU literature (§1) all measures against *global-memory*
   round trips, which is a 10–20× bandwidth gap; the L1↔L2 gap is far smaller, so the GPU
   speedups do not transfer and TurboFNO's 3–5% is probably the better prior. **This is the
   single most important number this project will have to measure for itself.**
8. **`L = 17`-specific register-level work.** Nothing found that discusses register allocation for
   a Rader-17 or Winograd-17 module specifically. The closest sourced anchors are FFTW's codelet
   list (no 17) `[F]`, VkFFT's "Rader for primes from 17" `[F]`, and Selesnick–Burrus's
   prime-length generator `[F-via]`.

---

## Citation ledger

**Fetched in this session (`[F]`) — 23:**

1. cuFFTDx, *First FFT Using cuFFTDx* — <https://docs.nvidia.com/cuda/cufftdx/introduction1.html>
2. cuFFTDx 1.2.1, *Your Next Custom FFT Kernels* — <https://docs.nvidia.com/cuda/cufftdx/1.2.1/introduction2.html>
3. cuFFTDx 1.2.1, *Requirements and Functionality* — <https://docs.nvidia.com/cuda/cufftdx/1.2.1/requirements_func.html>
4. cuFFTDx 1.2.1, *Examples* — <https://docs.nvidia.com/cuda/cufftdx/1.2.1/examples.html>
5. cuFFTDx 1.2.0, *Examples* — <https://docs.nvidia.com/cuda/cufftdx/1.2.0/examples.html>
6. cuFFTDx, *Execution Methods* — <https://docs.nvidia.com/cuda/cufftdx/api/methods.html>
7. D. Tolmachev, *VkFFT* SC22 technical poster (PDF) — <https://sc22.supercomputing.org/proceedings/tech_poster/poster_files/rpost143s3-file2.pdf>
8. VkFFT README — <https://raw.githubusercontent.com/DTolm/VkFFT/master/README.md>
9. VkFFT source, `lib/vkFFT.h` (`registerBoost`) — <https://raw.githubusercontent.com/DTolm/VkFFT/68b707955953aa50d997a47233b2d9726466cc30/lib/vkFFT.h>
10. M. Frigo, *A Fast Fourier Transform Compiler*, PLDI '99 — <https://www.fftw.org/pldi99.pdf>
11. M. Frigo & S. G. Johnson, *The Design and Implementation of FFTW3*, Proc. IEEE **93**(2), 216–231 (2005) — <https://www.fftw.org/fftw-paper-ieee.pdf>
12. FFTW, *FFT Accuracy Benchmark Comments* — <https://www.fftw.org/accuracy/comments.html>
13. FFTW, *FFT Accuracy Benchmark Methods* — <https://www.fftw.org/accuracy/method.html>
14. FFTW3, `dft/scalar/codelets/Makefile.am` — <https://raw.githubusercontent.com/FFTW/fftw3/master/dft/scalar/codelets/Makefile.am>
15. J. C. Schatzman, *SIAM J. Sci. Comput.* **17**(5), 1150–1166 (1996) — <https://epubs.siam.org/doi/10.1137/S1064827593247023>
16. R. Brent, C. Percival & P. Zimmermann, *Math. Comp.* **76**(259), 1469–1481 (2007) — <https://maths-people.anu.edu.au/~brent/pd/rpb221.pdf>
17. C.-P. Jeannerod, P. Kornerup, N. Louvet & J.-M. Muller, *Math. Comp.* **86**(304), 881–898 (2017), DOI 10.1090/mcom/3123 — record + abstract via <https://portal.findresearcher.sdu.dk/en/publications/error-bounds-on-complex-floating-point-multiplication-with-an-fma/>
18. K. Adámek, S. Dimoudi, M. Giles & W. Armour, ACM TACO **17**(3):18 (2020) — <https://arxiv.org/abs/1910.01972> and <https://arxiv.org/pdf/1910.01972>
19. S. Wu et al., *TurboFNO*, arXiv:2504.11681 — <https://arxiv.org/pdf/2504.11681>
20. S. Kawakami & D. Takahashi, arXiv:2603.29129 — <https://arxiv.org/abs/2603.29129>
21. M. A. Bergach, arXiv:2604.03585v1 — <https://arxiv.org/html/2604.03585v1>
22. F. Franchetti & M. Püschel, *Short Vector Code Generation and Adaptation for DSP Algorithms* — <https://users.ece.cmu.edu/~franzf/papers/icassp03.pdf> (venue inferred)
23. Intel SDM instruction reference for `VFMADD132PD/213PD/231PD` (mirror) — <https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd>

**Bibliographic record verified via a fetched source, paper not read (`[F-via]`) — 9:**
Hong & Kung 1981 (STOC); Frigo, Leiserson, Prokop & Ramachandran 1999 (FOCS); Xiong, Padua &
Johnson 2001 (PLDI); Gentleman & Sande 1966 (AFIPS); Buneman 1987 (Proc. IEEE); Tasche &
Zeuner 2002 (JCAA); Kahan 2001; Tolimieri, An & Lu 1997 (Springer); Selesnick & Burrus 1996
(IEEE TSP).

**Could not fetch (`[UNVERIFIED]`) — 3:**
Tolmachev, IEEE Access **11**, 12039–12058 (2023); Tolmachev, IWOCL 2023
(10.1145/3585341.3585357); Blake, Witten & Cree, IEEE TSP **61**(19), 4707–4716 (2013).
Intel Optimization Reference Manual (exceeded fetch size limit) is also not verified.
