# 1D vein: accuracy & fused FFT+pointwise chains (agent, 2026-09-02)

## Error bounds
- Worst-case O(log N)u, average O(sqrt(log N))u (Gentleman-Sande; Schatzman 1996;
  Tasche-Zeuner 2001). N=16384: ~1.5e-15 worst / 4e-16 typical per FFT. Size costs only
  a factor of a few small->16384 IF twiddles accurate.
- Brisebarre/Joldes/Muller TOMS 2020: tight-constant CT bounds, FMA-accounted, twiddle-mul
  as the accuracy-critical op = rigorous justification for FMA butterflies + a CERTIFIED
  per-step forward bound to compose over chain steps.
- OPEN: Higham-Mary-Connolly probabilistic (sqrt(n log n)u, mean-zero) NEVER specialized to
  the FFT (FFT's sqrt(log N) is separate/older, Calvetti 1991). Porting HMM to the butterfly
  DAG -> high-prob per-FFT constant composing over 10^4 steps = tightest envelope. Our
  measured chain data is the missing experiment.

## Twiddle/chirp accuracy = THE dominant lever (converges with CPU vein)
- "Inaccurate twiddles are the leading cause of FFT inaccuracy" (FFTW/Schatzman). Naive
  recurrences grow O(sqrt N)..O(N^2). USE precomputed table from correctly-rounded/dd sincos
  (<=0.5-1 ulp), generated in the planner/wisdom stage (free at runtime). For Bluestein the
  chirp W^{k^2} has huge index args -> highest-leverage single accuracy decision; reduce
  k^2 mod 2N in INTEGERS first (matches CPU vein's Bluestein gotcha).
- Bergach dual-select 6-FMA butterfly (arXiv:2604.00567): bounded-ratio twiddle tables,
  fewer rounding ops -> lower per-step constant. A/B before adopting. [2026 preprint]

## Algorithm accuracy ordering: native/PFA codelet > Rader > Bluestein
Bluestein most twiddle-sensitive (chirp), least accurate -> spend extra precision on the
chirp table when unavoidable (10007/100003). Precision-aware planner rule.

## Higher-accuracy arithmetic (mostly ORACLES not production)
- Double-double: VkFFT quad mode (GPU, "main error source is twiddle precision", FP128 LUT);
  libeft CPU building blocks. Uses: (a) dd reference oracle for the chain; (b) selectively
  promote ONLY the twiddle multiply to dd (most accuracy at fraction of full-dd ~3-4x cost).
- Kawakami-Takahashi Ozaki-NTT FFT: fp64-exact via NTT+CRT, provable split count = ideal
  gate ORACLE for large primes, but 107-1315x FFTW (checkpoint verifier, not production).
- Block-floating-point (range not precision, arXiv:2605.28451): fp16 binding constraint is
  the 5-bit exponent; our chain self-regulates range (FFT grows sqrt(N)/pass, +c biases,
  x/(1+|x|) contracts back). Non-issue in fp64; clarifies WHERE chain error enters.

## The chain: contraction vs our measured chaos (RECONCILES the 3D finding)
g(x)=x/(1+|x|) has |dg/dx|=1/(1+|x|)^2 < 1 -> MAGNITUDE contraction keeps the chain BOUNDED
(no overflow, unlike a raw FFT chain). This is WHY the chain is gate-stable. Our 3D-campaign
measurement that two correct fp64 impls DIVERGE is the sensitive-dependence (chaos) part
living INSIDE the bounded set -- both true. Action (matches our CLAUDE.md discipline):
measure fp64-vs-dd ||.|| vs step on a LOG axis, fit the slope; <=0 means fixed-tolerance
gating is safe for 10^4 steps (our two-part gate already self-calibrates this per chain).

## Tiered correctness gate (cheapest -> most rigorous)
1. benchFFT-style rel-error vs dd reference (routine QA; MKL ~310 dB ~15.5 digits) -- our
   benchFFT wiring already does this at B=1.
2. NTT-exact convolution on rational test vectors (bit-exact, zero roundoff).
3. Arb acb_dft ball arithmetic (provable enclosure, auto-picks radix/Bluestein/factor per
   length -- covers small primes, 16384, large primes uniformly) or Ozaki-NTT at checkpoints.
Our chaotic-chain 300x-anchor gate already handles the routine tier; Arb acb_dft is the
upgrade path for a PROVABLE checkpoint.
