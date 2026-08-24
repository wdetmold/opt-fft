# Vein 5: alternative transform formulations (agent report, 2026-08-24)

## Actionable
- **Caprioli, "flap" counts (IEEE 2025, hpkfft.com PDF)**: count FMA=1 ("flap") instead of
  flops=2 and the factorization ranking FLIPS: **2,8-split-radix needs the fewest flaps** — the
  classical 34/9 conjugate-split-radix "fewest flops" ranking is NOT FMA-optimal. Directly
  adoptable in our cost model / race candidate ranking. [partially verified]
- **Bergach dual-select FMA butterfly (arXiv:2604.00567)** [verified, also found by vein 1]:
  per-twiddle selection between tangent/cotangent Linzer-Feig factorizations; "worst-case ratio
  drops from 163 to exactly 1.0... zero computational overhead — only the precomputed twiddle
  table changes". Precision-agnostic; cheap experiment under our gates.
- **Stasinski nested Rader/Winograd classes (arXiv:2303.02647)** [verified; also vein 1]:
  size families incl. 24/48/120/240 (plausible 1D line sizes for composite L); zero fp stability
  analysis; high-risk, test only with a stability gate.

## Know-why-they-don't-apply (recorded so nobody rescans)
- **QFT has small entanglement (PRX Quantum 2023)**: DFT as tiny-bond-dimension MPO — real
  classical speedups ONLY for compressible signals; dense generic input has exponential bond
  dimension. Not exact, not applicable.
- **T-count-optimized AQFT (arXiv:2203.07739 + 2025)**: T-count has no correspondence to
  classical mult/add cost; the trick classically = truncating twiddles (approximate). Dry vein
  for exact classical arithmetic — verified no post-2020 QFT paper yields a new exact classical
  factorization.
- **Kedlaya-Umans lineage (FOCS22/JACM24 multipoint eval)**: finite fields, 1+o(1) exponents
  hide huge polylogs; "will not touch L<=128 this decade". The DFT-rigidity spinoff is
  conceptually notable only.
- **van der Hoeven-Lecerf amortized multipoint eval**: for tensor-product point sets the FFT IS
  the amortized-optimal special case; no win for dense cubes.
- **Sparse FT**: only deterministic+exact-capable post-2020 item is Plonka-von Wulffen
  (arXiv:2004.11097), degenerates to full FFT at M=N. Kapralov lineage randomized/approximate.
- **Hartley**: 2022 Springer book is hardware-niche; 1987 verdict stands (same mults, more adds).
- **Butterfly-factorization uniqueness (Zheng et al. SIMODS 2023)**: within the butterfly family
  there is NO undiscovered cheaper exact factorization of the DFT — fewer-ops hopes must come
  from different algebra, not factorization search.

## Open implementation gap
- **Coxon in-place TFT (JSC 2022, arXiv:2101.09005)**: only post-2020 motion in the truncated-FT
  lineage; an exact fp64 SIMD 3D TFT for padded convolutions is unbuilt. Applies ONLY if we ever
  serve convolutions (transform-pointwise-inverse) rather than spectra: removes zero-padding
  waste for non-2^k L.

## Negative space
No post-2020 improvement on 34/9 flop record (motion is FMA-aware recounting); no new small-prime
modules with stability analysis; no resultant/Chebyshev DFT reformulations.
