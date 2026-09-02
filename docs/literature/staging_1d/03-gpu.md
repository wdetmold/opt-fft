# 1D vein: GPU FFT (agent, 2026-09-02)

## Baselines & best code to study (ranked)
- **VkFFT** (Tolmachev, IEEE Access/SC22; github.com/DTolm/VkFFT): runtime codegen, radix
  2/3/4/5/7/8/11/13, INLINE RADER for primes 17->~10000 with no extra memory transfers,
  Bluestein+merged-convolution above; four-step with NO transpose uploads (stride by
  grouping nearby FFTs — key for batched); precision single/double/HALF/QUAD(double-double,
  LUT-based). Competitive-or-better than cuFFT on A100, lengths 2-4096. THE codebase to
  study. Shared-mem padding costs ~6.25%.
- **TurboFFT** (Wu et al., PPoPP 2025, arXiv:2412.05824): PADDING-FREE shared memory via
  register-swizzled access (100% util, zero bank conflict — beats VkFFT's 6.25%); template
  codegen per size 2^3..2^29; fp32+fp64; reports ~20% over VkFFT on small FFTs (unverified).
- **rocFFT** (AMD): clean Stockham autosort + explicit plan-TREE of kernels (readable
  reference for FFT+transpose+pointwise staging). hipFFT dispatches to rocFFT or cuFFT.
- **cuFFT** (closed): CT radix<=127 then BLUESTEIN; pow2 16384 path very strong. Weak at
  (a) large primes (Rader only <=127 -> 100003 hits Bluestein), (b) single/small-batch
  (underfills device).

## fp64 on matrix units
- DMMA (A100/H100 native fp64 tensor MMA) is the credible fp64-FFT-as-matmul route.
- tcFFT (arXiv:2104.11471, FP16, 1.1-3.2x cuFFT) + Sorna 2018 (fp16-split accuracy recovery
  recipe) = mixed-precision datapoints, not fp64 paths; read for the error model.
- Kawakami-Takahashi NTT-exact Bluestein = fp64 accuracy but 107-1315x FFTW (cautionary).

## FFT+pointwise chain regime (ours)
- **FlashFFTConv** (ICLR 2024, arXiv:2311.05908): Monarch decomposition -> whole
  fwd-FFT->pointwise->inv-FFT fuses into ONE matmul kernel, SRAM-resident, kills the I/O
  "fusion barrier". fp16/bf16, convolution-tuned. THE model for our chain regime; re-derive
  fp64 story. No fp64/CPU precedent = the gap.
- **cuFFT LTO callbacks** (cufftXtSetJITCallback): inline pointwise at the transform
  BOUNDARY only (not between stages); the pragmatic fusion baseline our fused kernel beats.

## Batched-thousands (ours)
- fbfft (ICLR 2015) established: ONE-BLOCK-PER-TRANSFORM in registers/shared, thousands
  launched as one grid, beats cuFFT "more dramatically for smaller batch". Single-transform
  small-N is the GPU's worst case (can't fill device) — expect CPU to win there.

## Large primes on GPU (our 100003)
- Every current GPU library sends 100003 to BLUESTEIN (cuFFT Rader<=127, VkFFT Rader<=~10000).
  100003-1 = 2*3*7*2381 (2381 prime) -> even Rader needs a length-100002 FFT with an awkward
  factor, so padded Bluestein is likely simplest. NOBODY does large-prime Rader on GPU =
  open ground. Measure Bluestein's fp64 accuracy penalty.

## For our (eventual) GPU 1D phase
Study order: VkFFT > TurboFFT > rocFFT > FlashFFTConv. fp64 via register Stockham or DMMA.
Batched = one-block-per-transform. Chain = FlashFFTConv fusion, fp64 re-derived. Primes =
plan for Bluestein, consider being first to do large-prime Rader.
