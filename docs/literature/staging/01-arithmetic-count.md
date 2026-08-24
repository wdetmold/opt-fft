# Vein 1: arithmetic-complexity advances for exact DFTs (agent report, 2026-08-24)

## 1. Alman & Rao (STOC 2023, arXiv:2211.06459) — leading constant 15/4 = 3.75 [fetched]
First improvement since Van Buskirk (34/9 ~ 3.7778); exact count (15/4)N logN - (223/108)N + o(N^0.8),
via WHT non-rigidity (low-rank + sparse) and an FFT->WHT(N/8) reduction. Reference code is a
correctness checker only; no performance implementation, no stability analysis. Agent's crossover
math: beats split-radix only near N~2^16; beats tangent-FFT near N~2^91. PROVABLY UNHELPFUL at
N<=128 — corpus headline + the WHT-splitting trick, not an implementable win.

## 2. Stasinski, "Split Multiple Radix FFT" (EUSIPCO 2022, IEEE 9909790) [abstract only]
Split-radix generalized to multiple mutually-prime auxiliary bases; for coprime-product sizes some
variants have smaller arithmetic/multiplicative complexity than split-radix. NO code, NO benchmarks.
Directly targets our 36=4x9, 45=9x5, 50=2x25, 100=4x25. Paywall—worth obtaining full text.

## 3. Stasinski, sub-O(N logN) multiplicative complexity (arXiv:2303.02647) [fetched]
Nested Rader–Winograd: map DFT to multidimensional, extract Rader convolutions of q_i-point DFTs,
combine into multidimensional convolutions; mult complexity O(N log^c logN), c<=1. N=240: 596 mults.
Flags radix-6 FFT with exceptionally few mults (Stasinski 1994 / Martens 1984) — relevant L=6/36.
Theoretical only. Caveat for AVX-512: FMA makes mults ~free when fused; the STRUCTURAL idea (batch
small Rader convolutions across factors into one multidim convolution over L^2 pencils) is the
unmined part.

## 4. Bergach, dual-select 6-FMA butterfly (arXiv:2604.00567) [MOST ACTIONABLE, abstract]
Keeps Linzer–Feig minimal 6-FMA radix-2 butterfly but selects per-twiddle whichever factorization
has |ratio|<=1: "eliminates all singularities, requires no epsilon clamping, bounds the precomputed
ratio to unity". Zero runtime cost — only twiddle TABLES change; 235x tighter error bound reported
(FP16 N=1024). No code, no benchmark. Drop-in table-generation change for our 2^k codelets
(8/32/64/128) and radix-2 stages inside composites: accuracy for free under our 1.5e-14/step gate.

## 5. Queiroz et al., square-index coefficients mult-free (arXiv:2407.00182)
X_{k sqrt(N)} via folding + sqrt(N)-point DFT, zero mults. Niche: only if a consumer ever needs a
sparse frequency subset; our transforms need all outputs. Filed.

## 6. NOVA (Lohia, arXiv:2512.18453) — ES search for well-conditioned exact Winograd transforms
Evolution-strategy over FRACTIONAL Vandermonde points + rationalization + symbolic verification;
F(8,3) conditioning improved 415x, transforms stay exact. ML-convolution context; nothing for DFT.
Transferable recipe: the conditioning of Winograd/Rader small-prime modules (13/17/23/31) is the
historical reason fp libraries abandoned them — searching OUR modules' point choices is unexploited.
Only post-2020 hit in "automated search for exact transforms".

## Negative space (searched, empty post-2020)
SAT/superoptimization of DFT circuits: nothing since Haynal & Haynal 2011. AlphaTensor-style RL on
DFT/convolution tensors: none. New fp Rader/Bluestein variants: none (all NTT/FHE hardware).
Hartley/real-factor revival: dormant. Leading constant below 3.75: none. New exact small-prime
modules: none (Portella–Bayer–Cintra 2025 multiplierless PFA-DFTs are APPROXIMATE — excluded).

## Bottom line
Exploitable: #4 (free accuracy, no competing implementation), #2/#3 (coprime-composite structure
for 36/45/50/100, never implemented), #6 (conditioning-search recipe for our prime modules).
