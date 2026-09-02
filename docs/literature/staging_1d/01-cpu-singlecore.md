# 1D vein: CPU single-core algorithm & technique (agent, 2026-09-02)

## Arithmetic backbone (small-N leaves / pow2)
- Modified split-radix (Johnson-Frigo 2007): 34/9 ~ 3.78 N log2 N, current "simple" pow2
  record. BUT the 6% flop win is ~invisible once cache/FMA-bound — chase last. Your 1K-16K.
- **Conjugate-pair split-radix** (depth-first iterative, IEEE TSP 2021): same flops, ~HALVES
  twiddle-table traffic (one twiddle load/radix-4 butterfly). Better lever than 34/9 at
  1024-16384 where you're twiddle-bandwidth bound. Under-exploited outside FFTS.
- FMA-aware butterflies (Goedecker 1997, Linzer-Feig 1993): every op an FMA — critical for
  AVX-512. Caveat: twiddle rescaling needs a divide -> suits PRECOMPUTED tables not on-line.
- Winograd small-N / minimum-ADD modules (Winograd 1978; 1988 min-add variant): your 7-31
  primes. On AVX-512 prefer minimum-OPERATION not minimum-multiply (mults cheap, adds/regs
  dominate).

## Cache/dataflow (large pow2, no bit-reversal)
- Good-Thomas PFA (coprime N1*N2, twiddle-FREE 2D DFT): small composites 15=3*5, 21, 35.
- Stockham autosort: self-sorting, unit-stride every stage, no bit-reversal pass — strong
  default for 1024-16384 (out-of-place buffer cheap there).
- Four-step/six-step Bailey (1990) + blocked (Takahashi 2001): 16384 fp64 = 256KB spills L2;
  factor 128x128 -> sub-FFTs L1-resident. Main lever at the top of the pow2 range.
- Cache-oblivious recursive (Frigo et al. 1999): near-optimal across L1/L2/L3 with no tuning
  = attractive for fixed sizes with no planner. Recurse above L1, iterate in the base case.

## What we're up against (baselines)
- FFTW3: cache-oblivious recursive CT + genfft codelets + AVX-512 SIMD + Rader AND Bluestein
  + planner. Where we beat it: FIXED geometry -> hard-select plan (no measurement), N-exact
  twiddle tables, drop codelet dispatch. FFTW does NOT default to modified-split-radix or
  conjugate-pair — ours to add. genfft's register-pressure-aware SCHEDULER is the load-
  bearing non-obvious part (ad-hoc hand kernels lose on scheduling, not flops).
- FFTS (Blake 2013): closest to our thesis — conjugate-pair + cache-oblivious + specialize-
  at-init SIMD codegen, NO offline calibration. Study its "specialize then run" model.
- SPIRAL (ICS 2011): synthesizes small-stride permutations from vector load+shuffle — THE
  reference for the in-register-transpose problem of across-transform vectorization.

## Prime N (size-by-size, N-1 factorizations re-verified by the agent)
- Rader cost = factorization of N-1: 65537 N-1=2^16 RADER-IDEAL (one clean radix-2 FFT);
  1021 N-1=2^2*3*5*17 smooth (efficient); 10007 N-1=2*5003 (5003 prime -> pad or Bluestein);
  100003 N-1=2*3*7*2381 (2381 prime -> Bluestein). Bake in: Rader if N-1 smooth, else Bluestein.
- Bluestein fp64 GOTCHA: reduce k^2 mod 2N in INTEGERS before *pi/N (k~10^5 -> k^2~10^10
  corrupts phase in fp64). Precompute chirp-FFT once per N (constant across calls) = big win
  in our batched/repeated regime. M=2^18=262144 for 100003; M=2^15 for 10007.

## The decision that matters most (AVX-512, cross-cutting)
Two orthogonal choices:
1. SPLIT (SoA) vs interleaved: no HW complex-mul, so split avoids per-mul swizzle -> preferred;
   need in-register 8x8 AoS<->SoA transpose (vpermi2pd/vshuff64x2) at boundaries only.
2. across-TRANSFORM (8 elts of 1 FFT/zmm, needs shuffle-synth permutations) vs
   **across-BATCH (lane j = transform j, ZERO shuffles, needs >=8 transforms)**.
Maps to our regimes: batched -> across-batch split (near-perfect 8-lane, THE highest-leverage
under-used card, makes 7-31 primes trivially vectorize); single 1024-16384 -> across-transform
Stockham/six-step; CHAINS -> split end-to-end so the map sees SoA, never repay transposes.

## Under-exploited openings (ranked)
1. Fixed-geometry specialization (no planner/dispatch, N-exact tables) — FFTS-style.
2. Across-batch AVX-512 for the batched regime — perfect 8-lane, zero shuffle.
3. Conjugate-pair twiddle-load reduction at 1024-16384.
4. Rader-into-pure-pow2 for 65537 (Fermat structure; generic libs won't).
5. Precomputed chirp-FFT for repeated Bluestein at 10007/100003.
6. Split end-to-end for FFT+map chains.
7. Modified split-radix 34/9 — real but LOW priority (invisible when memory/FMA-bound).
