# Vein 6: hardware-motivated algorithm reformulations (agent report, 2026-08-24)

## Highest leverage (agent's ranking)
1. **Constant Multiplier FFT (Garrido & Malagon, IEEE TCAS-I 2021)**: route data through the
   flow graph sorted BY TWIDDLE ANGLE so every multiplicative site sees a single constant.
   CPU reading: reorder batch/lane assignment so each SIMD lane-slot always multiplies the same
   twiddle -> twiddles become compiled-in broadcast constants (no loads/gathers, constant-folding,
   3-mult complex forms precomputed per site). Full angle-sorted schedule computable OFFLINE per
   fixed L. Never connected to SIMD software. (MCM shift-add itself does NOT transfer to fp64;
   the routing theorem does.)
2. **OpenFFT-SME (IPDPS 2024)**: Cooley-Tukey regrouped as OUTER-PRODUCT stages (rank-1 updates)
   for ARM SME; 3.6-4.1x over FFTW fp64. On AVX-512: broadcast-FMA rank-1 accumulation across
   batch lanes replacing shuffle butterflies. No x86 port. + **tcFFT (CLUSTER 2021)**: merge
   STAGES as matrices with twiddles folded into the stage operator — per-stage constant 8x8
   matrices in registers applied by broadcast-FMA.
3. **Dense DFT-GEMM crossover**: Ascend NPU study (Appl.Sci. 2024, MM-2DFT vs FFT crossover) +
   POWER10 MMA (arXiv:2104.03142 + IBM patents: "batches of M 32-point complex FFTs with separate
   real/imag parts using matrix multiplication", fp64 xvf64ger) — the only fp64-native CPU matrix
   engine with a published DFT-by-GEMM intent, and NO public library uses it. For L<=16 the whole
   DFT factor fits in zmm registers.
4. **FlashFFTConv order-p cost model**: p matmul passes; higher p = fewer FLOPs, more I/O — an
   explicit dial to set per size against measured flop:byte; block-skipping for pruned transforms
   free in matrix form. No CPU fp64 Monarch-FFT exists.
5. **REFFT (MEMSYS 2024)**: FFT plan space as tree rotations + codelet permutations — a complete
   ordered enumeration of CT split orders, memory-access order tunable INDEPENDENTLY of arithmetic
   order (what FFTW's planner conflates). Research library, large-1D only. Principled search space
   for our race layer.
6. **Radix-2^k MSC (IEEE Access 2023)**: (a) rotation thinning — schedule so only halves needing
   rotation meet a multiplier (halves twiddle-FMA per stage); (b) bit-dimension permutations as
   canonical algebra for in-register shuffles (bit-exchange <-> vshufpd/vpermt2pd/stride classes)
   — principled enumeration of minimal shuffle schedules.
7. **Register-resident Stockham counterexample (arXiv:2603.27569)**: simd-shuffle radix-32 LOST
   by 56% to register-resident radix-8 with rarer sequential exchanges — measured evidence against
   pure shuffle-count minimization; prefer register-resident columns + contiguous stores.

## PIM vein (low transferability, recorded)
FourierPIM (arXiv:2304.02336): O(log n) bulk row-parallel butterflies — the pure limit of
batch-lane; nothing AVX-512 doesn't exploit. Pimacolaba (arXiv:2308.03973): stage-to-tier
assignment by locality class + bytes-per-stage-per-tier accounting — useful analysis template for
pencil-pass fusion depth at L=64..128.

## Negative space (verified absences)
- Intel AMX <-> DFT: NOTHING exists, not even int8/bf16, despite AMX-COMPLEX; an Ozaki-split
  int8-AMX small-N DFT is unpublished territory (AMX has no fp64).
- Vector-length-agnostic FFT factorization: open hole (FFTW PR#279 concedes fixed-width is
  fundamental to its codegen).
- No CPU Monarch FFT; no public POWER10 MMA FFT despite IBM's own patents.
- MCM/ILP adder graphs (arXiv:2210.02742) never connected to floating-point SIMD twiddles.
