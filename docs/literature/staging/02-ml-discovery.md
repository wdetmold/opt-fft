# Vein 2: automated algorithm discovery & program synthesis (agent report, 2026-08-24)

## 1. SLOTHY — constraint-solving superoptimization of real FFT kernels [most actionable]
Abdulrahman, Becker, Kannwischer, Klein, "Fast and Clean: Auditable high-performance
assembly via constraint solving," TCHES 2024 (ePrint 2022/1303); github.com/slothy-optimizer/slothy.
Joint instruction scheduling + register allocation + software pipelining via CP-SAT under a
uarch model; validated on radix-4 complex FFT (fixed & floating point) on Cortex-M55/M85 and
PQC NTTs, "match or beat prior art in all cases". Mature OSS. NO x86/AVX-512 backend — the
open lane is writing a Golden-Cove/Ice-Lake model and running it over our existing kernels.
Schedules only, not new algorithms.

## 2. Exact butterfly factorization identification (Zheng/Riccietti/Gribonval)
SIMODS 2023 (arXiv:2110.01230) + ICASSP 2022: any N=2^J matrix with butterfly structure has
an essentially UNIQUE factorization, recoverable in O(N^2) by hierarchical factorization —
exact recovery incl. DFT/Hadamard, with identifiability guarantees (not gradient fitting).
Python research code only; no vectorized kernels, no FFTW/MKL benchmarks. Relevance: recover
exact sparse factorizations of fused operators (DFT-diag-DFT chains) for fixed geometries.
Power-of-two only (8..128, not 6/12/24).

## 3. Butterfly identification without known partitioning
Zheng et al., GRETSI 2023 (arXiv:2307.00820, French): identifies the block partitioning
algebraically, no analytical assumption on entries. Removes the "which butterfly" assumption.
Research-grade; unverified detail. See also "Butterfly Factorization with Error Guarantees"
(SIMAX, DOI 10.1137/24M1708796, unfetched).

## 4. Monarch matrices (Dao et al., ICML 2022, arXiv:2204.00595)
M = P1 L P2^T R products of two block-diagonals; provably contains DFT/DCT/Hadamard; exact
closed-form projection onto the class (blockwise SVD). GPU/training-precision only. Algebraic
takeaway: FFT = permuted product of two block-diagonal GEMM stages (four-step/Bailey rebrand)
— an FMA-dense schedule alternative to shuffle-heavy butterflies, unevaluated for fp64 CPU.

## 5. FlashFFTConv (Fu et al., ICLR 2024, arXiv:2311.05908)
Monarch-decomposed FFT on tensor cores with kernel fusion of the pointwise map between
forward/inverse FFT; "speeds up exact FFT convolutions by up to 7.93x over PyTorch". Real
optimized GPU code, half/mixed precision, long 1D sequences. The strongest evidence that
matmul-FFT + map fusion wins on matmul-dense hardware; fp64/AVX-512/small-3D analogue open.

## 6. Equality saturation for DSP kernel vectorization
Diospyros (ASPLOS 2021) and Isaria (ASPLOS 2024 best paper): e-graph rewriting (egg) for
small fixed-size kernels; Isaria auto-generates rewrite rules from ISA semantics, beats
Tensilica SDK by up to 6.9x. FFT not among evaluated kernels; no x86 target. Open project:
DFT algebra as an egg rule set over AVX-512 fp64. Enabler: MCTS-GEB (arXiv:2303.04651).

## 7. AlphaTensor/AlphaDev/AlphaEvolve — engines exist, FFT application does NOT
Nature 2022/2023; AlphaEvolve 2025 (arXiv:2506.13131) found 4x4 complex matmul in 48 mults
(first sub-Strassen-recursion since 1969) — complex-arithmetic savings are Fourier-adjacent.
VERIFIED NEGATIVE: no published RL/agentic attack on DFT factorizations or FFT op counts as
of Aug 2026. The correct analogue targets: cyclic-convolution tensor (bilinear, Winograd) or
sparse-factorization op-count minimization. Largest open target in the vein.
See also "Complex to Rational Fast Matrix Multiplication" (arXiv:2602.13171, unverified).

## 8. Flip graphs & automated bilinear complexity
Kauers & Moosbauer ISSAC 2023 (arXiv:2212.01175) + open-source framework (arXiv:2603.02398)
+ Wang's automated LOWER bounds with machine-checkable certificates explicitly covering
cyclic/negacyclic/truncated convolution over F2/F3 (arXiv:2603.07280). Nobody has pointed
flip graphs at cyclic-convolution tensors over R/C for our small L — concrete cheap
experiment; Wang's bounds cap the available headroom. Caveat: mult counts rarely survive
fp64 SIMD reality (adds/shuffles dominate).

## 9. SPIRAL successors (FFTX, FFTc/MLIR, semantics lifting) — infrastructure, not discovery
FFTX (DOE ECP): SPIRAL-generated kernels + cross-FFT fusion of FFT+linear chains — the one
production system built for our workload SHAPE; worth baselining for chain fusion. FFTc 2.0
(IEEE Cluster 2024): SPIRAL-style rewriting in MLIR, perf vs FFTW unverified. Rules still
human-written everywhere — the discovery layer above SPIRAL remains unbuilt.

## Negative space (checked)
No RL discovery on DFT; no equality-saturation paper on FFT dataflow; no fp64/CPU validation
in the butterfly/Monarch line; no AVX-512 SLOTHY backend; small non-2^J sizes served by none
of the above.

## Bottom line
Highest-leverage imports: (1) SLOTHY-style CP-SAT scheduling with an AVX-512 uarch model over
our existing kernels; (2) flip-graph/evolutionary search over small cyclic-convolution/DFT
tensors at our exact L values. Both unpublished as applied — genuinely open.
