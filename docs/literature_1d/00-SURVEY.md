# 1D FFT optimization — literature survey (2026-09-02)

Five-vein parallel survey for the standalone d1 campaign (NOT the 3D corpus; separate because
1D is compute-bound, L1-resident, and its edges are different). Full per-vein reports with
citations and verification tags in staging_1d/ (01 CPU single-core, 02 batched/manycore,
03 GPU, 04 prime-length, 05 accuracy/chains). This file is the synthesis: what MULTIPLE veins
agreed on, ranked by leverage for our sizes (13-128, 1024/4096/16384, primes 1021/10007/65537/100003)
and four call regimes (batch x chain).

## Cross-vein convergences (independent agents, same conclusion = high confidence)

1. **Across-batch split-complex vectorization is the top under-exploited lever.**
   (CPU-single #16, batched #1-2 both name it first.) Lane j = transform j, split (SoA) so
   complex-mul needs no in-register swizzle, ZERO shuffles, near-perfect 8-lane fp64 — needs
   >=8 transforms. This is the 3D campaign's batch-lane trick, and it makes the small primes
   (7-31) trivially vectorize. OUR BATCHED REGIME'S DEFAULT. Formal name: DFT_n (x) I_v
   (SPIRAL); published fp64 justification: Popovici/Franchetti/Low HPEC 2017 (up to 2x FFTW).

2. **Correctly-rounded / double-double twiddle+chirp tables, generated in the planner stage.**
   (CPU-single #16, accuracy #4, prime #11 all converge.) "Inaccurate twiddles are the
   leading cause of FFT inaccuracy" — never in-loop recurrences (grow O(sqrt N)..O(N^2)).
   Free at runtime (our wisdom/plan stage already exists). Bluestein chirp: reduce k^2 mod 2N
   in INTEGERS before the trig call (fp64 trap at k~10^5).

3. **Large-prime batched Rader/Bluestein is genuinely open ground.**
   (batched #12, GPU #11, prime #7-8 converge.) No published wide-SIMD batch-vectorized
   Rader/Bluestein at ~10^5 primes on ANY target; GPU libs are Bluestein-only above ~10^4.
   The win is applying the batch-lane trick to the INNER convolution FFTs + pointwise stage,
   not a cleverer prime algorithm.

## The four primes — the campaign's sharpest edge (prime vein, corroborated)
Libraries default to Bluestein even where Rader is clearly better; the reframe is that
padded Rader / Bluestein cost is INDEPENDENT of N-1's factorization.
- 1021  (N-1=2^2*3*5*17 pairwise coprime): Rader + Good-Thomas TWIDDLE-FREE 1020-conv; direct
  O(N^2) DFT as accuracy oracle. Rader wins speed AND accuracy; libs may Bluestein.
- 65537 (N-1=2^16 Fermat): UNPADDED Rader — conv IS 65536=2^16, reuse pow2/NTT butterflies;
  crushes any Bluestein-only library (which pads to 131072). The dream case.
- 10007 (N-1=2*5003): Bluestein pad 20480/32768 baseline; A/B one-level NESTED Rader
  (5003-1=2*41*61 smooth). Exactly where FFTW's planner bails to Bluestein = our opening.
- 100003 (N-1=2*3*7*2381): padded Rader/Bluestein pad 200000=2^6*5^5; nested Rader viable
  (2381-1=2^2*5*7*17 smooth). Same opening.
Number to beat: FFTW's measured prime-vs-pow2 slowdown near 16384 is ~8.7-10.1x
(Harvey-vdHoeven CTAC 2020); asymptotic 1+eps Gaussian-resampling is theory-only.

## Per-size technique map (small->large pow2)
- 7-31 primes: minimum-OPERATION (not min-multiply) Winograd/dense small-N codelets; WFTA is
  dead (ill-conditioned + FMA inverted the multiply/add trade). Trivially vectorize across batch.
- small composites (15/21/35): Good-Thomas PFA (coprime, twiddle-free).
- 1024/4096: Stockham autosort (no bit-reversal); CONJUGATE-PAIR split-radix (same flops,
  ~half the twiddle-table traffic — better lever than the 34/9 modified split-radix, which is
  ~invisible when memory/FMA-bound).
- 16384 (256KB, spills L2): four-step/six-step (Bailey) factor 128x128 -> L1-resident sub-FFTs.
- Fixed-geometry specialization everywhere: hard-select the plan (no measurement), N-exact
  tables, drop codelet dispatch — FFTS (Blake 2013) is the closest prior art to our thesis.

## Multicore is SIMPLE for us (batched vein — a de-scoping result)
Batch = the parallel axis: embarrassingly parallel, NUMA-local first-touch + thread pinning,
thread-allocation by task size (OpenFFT). We do NOT need single-transform threading machinery
(SPIRAL-SMP etc). Downstream pull is real: heFFTe's local kernel is a batched-1D call we could
drop-replace.

## GPU (for an eventual d1 GPU phase)
Study VkFFT (inline Rader to ~10000, double-double, no-transpose four-step) > TurboFFT
(padding-free register swizzle, per-size codegen) > rocFFT (clean Stockham plan-tree) >
FlashFFTConv (Monarch fusion of the whole FFT+pointwise+iFFT chain = our chain regime's model).
Batched-thousands: one-block-per-transform in registers (fbfft->VkFFT). Single small-N: the
GPU's worst case (can't fill device) — expect CPU to win there. Large-prime Rader >~10000 is
open on GPU too.

## Accuracy / chain gating (accuracy vein)
Our map x/(1+|x|) is a magnitude CONTRACTION (|dg/dx|=1/(1+|x|)^2<1) -> keeps the chain BOUNDED
(why it's gate-stable); the divergence we measured in 3D is sensitive-dependence inside that
bounded set (both true). Tiered gate: (1) benchFFT-style vs dd reference (routine, our B=1
wiring), (2) NTT-exact on rational vectors (bit-exact), (3) Arb acb_dft ball arithmetic =
PROVABLE checkpoint enclosure (auto-covers small primes/16384/large primes) — the upgrade
path from our current 300x-anchor gate. Open research: Higham-Mary-Connolly probabilistic
bounds never specialized to the FFT; our chain data is the missing experiment.
