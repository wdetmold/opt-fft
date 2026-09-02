# Transverse vein: real vs complex FFT (agent, 2026-09-02)
# NOTE: survey prompt carried an incorrect "LQCD fields are real" framing (entry 11/bottom
# line). QCD position-space fields are COMPLEX (SU(3) links, complex propagators) -- see
# memory lqcd-fields-are-complex.md. Algorithmic findings stand; the QCD-real-field
# relevance claim is REJECTED and not carried into EXTENSIONS.md.

## Bottom line: ~2x flops + exactly 2x memory theoretical; realized regime-dependent.
No post-2015 algorithmic breakthrough (Sorensen 1987 / Johnson-Frigo 2007 op-counts remain
reference); modern action is SIMD/GPU packing only.

## Methods & verdicts (payoff-ranked)
1. Two-real-in-one (NR twofft): EASY WIN, most actionable. Batched regime: pair B reals into
   B/2 complex, run our batch-lane engine UNCHANGED; recombine index-reversal is along the
   transform axis -> vectorizes across the batch. ~full 2x, negligible overhead. START HERE.
2. One-real via length-N/2 complex + split (NR realft): B=1; split on critical path,
   reversed-index gather (SIMD-hostile) -> ~1.3-1.6x, -> parity at small N.
3. Real split-radix (Sorensen ASSP-35:849): ~0.47x complex flops but SIMD-messy (Hermitian
   half odd) -> realize IMPLICITLY via #1/#2 into our complex kernels, not bespoke butterflies.
4. genfft real codelets (FFTW r2c): prune complex DAG for Hermitian; our fixed-geometry
   N-exact pruning is the analog. Owning the code = the point.
5. DHT: DEAD since 1987 (>=2 more adds than real DFT; multi-D not simply separable). Skip.

## Win & what eats it
Flops ~0.47-0.5x; MEMORY exactly 1/2 (never eaten). Eaten by O(N) split/recombine (serial,
shuffle-heavy, reversed-index), N/2+1 odd non-pow2 output, DC/Nyquist special-casing ->
realized 1.3-1.7x. Storage (FFTW halfcomplex / MKL CCE-only-for-3D / cuFFT N/2+1) corrodes
our SoA lane packing + 8x8 transposes. DESIGN RULE: compute FULL-COMPLEX, pack to
half-complex only at the API boundary; never thread N/2+1 storage through the SIMD kernels.

## Accuracy / primes / GPU / multi-D
- Accuracy: r2c as good or slightly better than c2c; recombine twiddles fold into our dd
  table; gate unchanged. NON-BLOCKER (caveat: two-real subtraction cancels if scales differ).
- Large primes: TRAP -- Rader re-mixes symmetry, Bluestein chirp complexifies real input;
  libraries do complex-then-halve = MEMORY only, ~1x flops. Skip for d1 large primes.
- GPU: cuFFT D2Z/Z2D fp64 (out-of-place, x mult of 4 -> alignment cliffs); VkFFT R2C/C2R
  + R2R DCT I-IV, best open reference.
- Multi-D r2c (3D/4D): ROBUST win -- one real axis pass (-> N/2+1, halves data) + complex on
  the rest over halved volume -> ~1.7-1.9x flops + HARD 2x memory; one new kernel into our
  pencil/transpose arch. Memory halving matters most in the memory-bound d=4 regime for a
  REAL-INPUT workload (NOT QCD, whose fields are complex).

## Opening
FFTW/MKL tune real LESS than complex (benchFFT r2c ~1.4-1.7x c2c not 2x); MKL format zoo is a
user-error source. Our relative edge on real plausibly matches/exceeds our ~3.5x c2c -- measure.
