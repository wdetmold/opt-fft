# Vein 7: numerical analysis enabling shortcuts (agent report, 2026-08-24)

## Sharper bounds that justify cheaper arithmetic
- Brisebarre/Joldes/Muller et al. (TOMS 2020): TIGHT, provably optimal per-op bounds for
  radix-2 butterfly + twiddle multiply, with/without FMA, INCLUDING tabulated-twiddle error —
  the constants for an exact per-stage error ledger per fixed L: stages far under budget can
  absorb deliberately cheaper arithmetic, with proof. [constants unverified today: ACM/HAL 403]
- Connolly-Higham-Mary (SISC 2021) + El Arar lineage (SISC 2023/2025): probabilistic/martingale
  bounds ~sqrt(n)u; over a chain of T steps aggregate scales sqrt(T) not T — formal license for
  per-step gating without per-step tightening. Limited-precision SR theory quantifies how few
  random bits suffice if SR is ever injected.
- Bhola-Duraisamy (arXiv 2411.18747, 2404.12556): mixed-precision FMA/tensor-unit bounds;
  probabilistic ~10x tighter than deterministic for matmul; BIAS-AWARE version warns when
  cheap tricks are NOT safe (biased accumulation beats sqrt(n)). No FFT treatment (verified).

## Twiddles
- Bergach dual-select (2604.00567) [third vein to converge on it]: table-prep policy, ratio<=1,
  zero runtime cost. Adopt at plan time for FMA-fused butterflies.
- NEGATIVE: no post-2020 twiddle-recurrence advance; precomputed correctly-rounded tables remain
  right for fixed small L. Open gap: compensated/FMA recurrence with certified O(u) error.

## Chaotic chains and precision reduction
- Klower et al. (Sci Rep 2023): finite-precision chaotic orbits are eventually periodic; SR
  prevents periodicity; the meaningful criterion past the divergence horizon is DISTRIBUTIONAL
  (attractor statistics), not pointwise. Reframes our observed chain divergence as expected.
- Paxton et al. (J.Climate 2022) + Croci-Giles (IMA JNA 2022): chaotic SPECTRAL (FFT+map!) model
  runs fine at 10-12 significand bits judged statistically; RN's failure mode in iterated maps is
  BIAS-DRIVEN DRIFT/STAGNATION (O(u/dt) growth) — the thing per-step probabilistic models miss.
- NEGATIVE: no shadowing theorem for finite-precision FFT chains post-2020; statistical gating is
  the defensible frontier, not trajectory shadowing (Boghosian 2019 warns some statistics converge
  WRONG independent of precision).

## Certified / cheaper chain checking (our checker could be rebuilt on these)
- **FLINT acb_dft** (production, maintained): certified ball-arithmetic DFT with reusable per-size
  precomp — a checker whose output is a mathematical ENCLOSURE; ball-radius growth along the chain
  MEASURES chaotic amplification (replaces our empirical anchors with certificates). Ready today.
- Kawakami-Takahashi NTT-exact convolution as a bitwise-deterministic, machine-independent checker
  core (too slow to compute with; fine to check with). + integer scale-and-round NTT fingerprints.
- TurboFFT (2405.02520): linear checksums fused into FFT at 7-15% cost (built for soft errors; the
  mechanism gives a per-step numerical sanity tier between "no check" and "full reference chain").
- de Angelis (2205.13978): exact interval DFT amplitude/phase bounds in poly time (dependency
  problem solved) — per-step tool; intervals inflate at Lyapunov rate through the map.
- Schmid et al. (2504.10136): moment propagation through an FFT factor graph — cheap non-certified
  error-distribution tracking along chains.

## Publishing gaps our measured data already touches
1. Modern probabilistic (martingale) rounding theorem for Cooley-Tukey + chain composition.
2. Roundoff analysis of fused FFT+pointwise-nonlinear pipelines (our chain data = the experiment).
3. (From vein 4) I/O lower bound for the composed FFT->map->FFT chain.
