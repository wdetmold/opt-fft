# 11. Post-2020 algorithmic ideas untested in performant software

Six-vein parallel literature sweep, 2026-08-24. Screen: algorithmically new since 2020 AND never
validated in optimized code — the theory/practice gap this project's panels can arbitrage. Full
per-vein agent reports with quotes, links and verification levels: `staging/01..07-*.md`.
Sizes referenced: our gen acceptance suite (10..100) and graded cubes (6..64).

## Tier 1 — cheap, high-confidence, adoptable this campaign

| idea | source | what it is | cost to try |
|---|---|---|---|
| Dual-select FMA butterfly | Bergach, arXiv:2604.00567 (found independently by 3 veins) | per-twiddle choice between Linzer-Feig factorizations keeps every stored ratio <=1: worst case 163 -> 1.0, "235x tighter error bound", zero runtime cost — a TABLE-GENERATION policy | hours; 2^k codelets first |
| Flap-count ranking | Caprioli, IEEE 2025 (hpkfft) | counting FMA=1 flips the factorization ranking: 2,8-split-radix is FMA-optimal, not conjugate split-radix (34/9) | update race cost model |
| Constant-per-site twiddle routing | Garrido & Malagon, IEEE TCAS-I 2021 | route data sorted BY ANGLE so each multiplicative site sees ONE compile-time constant; for fixed L the schedule computes offline; twiddles become compiled-in broadcasts (no loads) | days per codelet family; never done in SIMD |
| GT/Rader as vectorization-first design | TCHES 2021/2023 crypto lineage | decomposition chosen by SIMD permutation cost (GT kills twiddles AND strides; Rader convolutions as vector kernels, not fallbacks); 3-6x measured in NTT world; accuracy improves (fewer twiddle roundings) | methodology transfer; partially ours already |

## Tier 2 — structural experiments (the large-L cells, our weakest at 1.6-2.0x, are the targets)

- **Stage-as-matrix / outer-product CT** (OpenFFT-SME IPDPS 2024: 3.6-4.1x over FFTW fp64 on ARM
  SME; tcFFT; IBM MMA patents' fp64 split-complex DFT-by-GEMM): fold twiddles into per-stage
  constant matrices held in registers, applied by broadcast-FMA rank-1 updates. No x86 port of
  any of it. Dense DFT-GEMM crossover at small L (Ascend study) says L<=16 may prefer pure GEMM.
- **Few-pass 3D + fusion** (FlashFFTConv's order-p FLOP<->I/O dial; TurboFNO's "choose the FFT
  dataflow so the CHAIN fuses"; MDFFT transpose-free column-order, ARM-only; vector-radix lineage
  dead post-2015): combine (a) map fused into edge butterfly stages, (b) two axes per DRAM pass
  with L2-resident tiles, (c) transpose-free ordering — validated separately, never combined on CPU.
- **Plan-space enumeration** (REFFT, MEMSYS 2024): tree rotations + codelet permutations as a
  complete ordered CT plan space with memory order tuned independently of arithmetic order —
  a principled search lattice for gen_race.
- **Rotation thinning + bit-exchange shuffle algebra** (radix-2^k MSC, IEEE Access 2023): halve
  twiddle-FMA per stage; enumerate minimal shuffle schedules as bit exchanges.
- **Stasinski composite-N constructions** (EUSIPCO 2022 + arXiv:2303.02647): coprime multi-base
  split-radix and nested Rader-Winograd for 24/48/120/240-family sizes; zero stability analysis;
  high-risk, gate carefully.
- Warning from measurement (arXiv:2603.27569): register-resident radix-8 with rare sequential
  exchanges BEAT a lower-shuffle-count scattered schedule by 56% — don't over-minimize shuffles.

## Tier 3 — accuracy & verification arsenal

- **Compensated FFT with precomputed twiddle Dekker splits: does not exist anywhere.** Twiddles
  are compile-time constants so their splits are free; TwoProd butterflies then cost ~1.3-1.7x,
  not 2x+. Pair with correctly-rounded sinpi/cospi tables. Cheapest unclaimed accuracy win.
- **Selective double-double**: VkFFT ships GPU dd at ~2x for radix kernels; no CPU/AVX-512
  equivalent. dd only at accumulation-critical stages = 1-2 extra digits of gate headroom.
- **Certified checking**: FLINT `acb_dft` (production) = ball-arithmetic reference chains whose
  radius growth MEASURES chaotic amplification (certificates instead of our empirical anchors);
  NTT-exact convolution fingerprints (Kawakami-Takahashi core) = bitwise-deterministic checks;
  TurboFFT-style linear checksums = 7-15%-cost per-step sanity tier. FLINT fft_small proves
  50-bit exact NTT butterflies run in fp64 FMA pipes — the exactness-oracle floor is far below
  the quoted 107-1315x.
- **Error-ledger constants**: Brisebarre et al. (TOMS 2020) provably-optimal per-op bounds =
  the backbone for budgeting deliberately cheaper stages under 1.5e-14/step.
- **Chain theory**: martingale sqrt(T) accumulation (Connolly-Higham-Mary lineage) licenses
  per-step gating; Croci-Giles warns the RN failure mode is bias-driven drift — the thing to
  monitor if any precision is dropped mid-chain; Klower et al.: past the divergence horizon the
  defensible gate is distributional, not pointwise (matches our two-part gate design).

## Publishing gaps our own measurements already touch
1. A modern probabilistic rounding theorem for Cooley-Tukey and its chain composition (unwritten).
2. Roundoff analysis of fused FFT+pointwise-nonlinear pipelines (our chain-divergence data IS the
   missing experiment).
3. An I/O lower bound for the composed FFT->map->FFT chain (strictly below sum of per-kernel
   floors; hourglass-pattern machinery exists, unapplied).

## Dead veins (recorded so nobody rescans)
Alman-Rao 3.75 constant (crossover ~2^16); QFT/T-count tricks (no classical arithmetic meaning);
Kedlaya-Umans multipoint eval (finite fields, giant polylogs); sparse FT (randomized/approximate,
degenerates at M=N); Hartley (1987 verdict stands); cache-oblivious (nothing since 1999); twiddle
recurrences (no advance); VLA FFT factorizations (confirmed hole, nothing to steal); butterfly-
factorization search (uniqueness theorem: no cheaper exact factorization hides in that family);
RL/AlphaTensor on DFT (verified: does not exist yet — and the flip-graph engines that would do it
are open-source and unaimed: the biggest open target found by the whole sweep).
