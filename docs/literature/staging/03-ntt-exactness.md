# Vein 3: NTT & exact-arithmetic techniques (agent report, 2026-08-24)

## Directly actionable
- **Good-Thomas/Rader/Bruun as VECTORIZATION-first design (TCHES 2021 eprint 2020/1397;
  eprint 2023/1580; survey 2023/1962)**: post-2020 crypto kernels treat decomposition choice as
  an optimization variable driven by SIMD permutation cost (GT index maps kill twiddles AND make
  strides permutation-free; Rader convolutions written as vector kernels, not scalar fallbacks);
  measured 3-6x wins. FFTW/MKL still treat Rader/PFA as fallbacks. Transfer = methodology (no
  modular arithmetic involved); accuracy may IMPROVE (fewer twiddle roundings). Our gen campaign
  partially does this already — the untried part is global GT-first re-derivation of the 3D
  mixed-radix structure + vector Rader for 7/11/13 at composite L.
- **Compensated FFT with precomputed twiddle splits — DOES NOT EXIST [verified negative]**:
  no compensated Cooley-Tukey artifact post-2020 or ever. Key insight: twiddles are compile-time
  constants so their Dekker splits are FREE; a TwoProd butterfly then costs ~1 extra FMA per
  product (~1.3-1.7x, not 2x+). Cheapest unclaimed accuracy win under a 1e-14 gate. Pair with
  correctly-rounded sinpi/cospi tables (RLIBM, PLDI 2021).
- **VkFFT double-double mode (GPU, shipping)**: dd at ~2x cost for radix kernels, intended for
  "FP64 FFT with all on-chip calculations in double-double". NO CPU/AVX-512 equivalent exists.
  Selective dd (only accumulation-critical stages / twiddle products) = 1-2 extra digits of
  headroom, likely well under 2x with precomputed splits.
- **FLINT fft_small (Schultz)**: <=50-bit exact NTT butterflies HOSTED IN fp64 FMA pipes (AVX2/
  NEON, production). Kawakami-Takahashi's 107-1315x exactness cost is NOT the floor — a 50-bit
  fp64-FMA NTT on AVX-512 as a cheap exactness oracle beside the fast FFT is unmeasured.

## Analysis skeletons worth importing
- **Klemsa (eprint 2021/480)**: fp64 FFT certified error-free for integer convolution under norm
  bounds — the per-stage worst-case vs rare-error bookkeeping is the right machinery for
  certifying our 1e-14/step gate offline per L.
- **Ozaki-line slice-exactness accounting (arXiv 2203.03341, 2508.00441)**: proof skeleton for a
  "selectively compensated" FFT — 2-term splits with cross terms only where accounting says they
  matter (~2-4x in GEMM, vs 100x for full exactness).
- **Plantard lazy-reduction discipline (TCHES 2022)**: proven-headroom deferred normalization
  across butterfly layers; fp analog = deferred renormalization of (hi,lo) pairs across k stages
  with a growth bound. No fp FFT paper does this.
- **Probabilistic rounding (Connolly-Higham-Mary line)**: martingale sqrt(T) growth across chain
  steps rigorously justifies per-step gating — and a Higham-Mary-style bound for chained
  FFT+pointwise iterations is UNWRITTEN (publishing gap; our measured chain data speaks to it).
- **DGT over GF(p^2)** (Gaussian-integer transforms, FHE GPUs): finite-field mirror of the
  complex-arithmetic/half-length trade — its wide-vector verdicts map onto conjugate-pair packing
  choices where twiddle-load bandwidth is first-order.

## Negative space (verified)
No compensated-EFT FFT ever; no CPU dd FFT; no post-2020 probabilistic FFT error analysis (stops
at Calvetti 1991); transfer is one-way FFT->NTT (Franchetti PACT 2024, Takahashi CASC 2022 import
FFT vectorization into NTT — nothing flows back); no post-2020 Rader/GT ACCURACY quantification.
