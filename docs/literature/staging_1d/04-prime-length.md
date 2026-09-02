# 1D vein: prime-length & arbitrary-length FFT (agent, 2026-09-02)

## THE reframe (planner hinges on this)
UNPADDED/cyclic Rader cost is set by N-1's factorization. PADDED Rader and Bluestein cost
is INDEPENDENT of N-1 (you pick a smooth padding length m >= 2(N-1)-1). So "Rader is bad
when N-1 has a big prime factor" is ONLY true for the unpadded/recursive variant.

## Per-prime verdicts (primitive roots + factorizations sympy-verified)
| N | N-1 | prim root | pick |
|---|---|---|---|
| 1021 | 2^2*3*5*17 (smooth, PAIRWISE COPRIME) | 10 | Rader w/ Good-Thomas TWIDDLE-FREE 1020=4*3*5*17 conv; or pad 2048 |
| 10007 | 2*5003 (5003 prime) | 5 | Bluestein pad 20480/32768; A/B one-level nested Rader (5003-1=2*41*61 SMOOTH) |
| 65537 | 2^16 (Fermat) | 3 | Rader UNPADDED — ideal, conv IS 65536=2^16, reuse pow2/NTT butterflies |
| 100003 | 2*3*7*2381 (2381 prime) | 2 | padded Rader/Bluestein pad 200000=2^6*5^5 or 262144; nested Rader viable (2381-1=2^2*5*7*17 smooth) |

## THE two openings (measurable wins over the libraries)
1. **Libraries default to Bluestein for 65537 and often 1021 despite Rader being clearly
   better** (many GPU libs don't implement Rader at all; FFTW may Bluestein 1021). Rader-
   into-pure-pow2 at 65537 (conv=65536) beats a 2x-padded Bluestein (131072) outright.
2. **10007/100003: hand-tuned ONE-LEVEL nested Rader** targets exactly where FFTW's planner
   bails to Bluestein (its own source comment: when p-1 not nicely factorable "Bluestein
   should take care of"). Both inner primes have smooth p-1, so nested Rader is well-
   conditioned AND avoids Bluestein's 2x padding + accuracy penalty.

## Algorithm/accuracy facts
- Bluestein chirp fp64 TRAP: compute angle from (k^2 mod 2N), never k^2 directly (k~10^5 ->
  k^2~10^10 corrupts phase). Bluestein is the LEAST accurate of the three (chirp + long conv).
- Accuracy order: Cooley-Tukey pow2 (O(sqrt log N), best-conditioned) > Rader (small const
  above, well-conditioned, degrades per nesting level) > Bluestein > Winograd (worst).
- WFTA is DEAD in fp libs: Vandermonde modules ill-conditioned (error exp in module size)
  AND multiply/add trade inverted once HW multipliers got cheap (FFTW's explicit rejection
  quote). Only small stable modules (2/3/4/5/7/8) survive as codelets.
- 1021: keep a direct O(N^2) DFT (~10^6 ops, feasible) as an ACCURACY ORACLE.

## What we're up against / transferable
- FFTW: measuring planner; Rader only when p-1 smooth + above a threshold, else Bluestein;
  novel real-data Rader via DHT. Jagged prime curve. cuFFT Rader<=127, VkFFT Rader<=~10000,
  rocFFT dedicated Bluestein. GPU libs largely Bluestein-only for big primes = wide opening.
- Harvey-van der Hoeven Gaussian resampling: prime-length slowdown 1+eps asymptotically
  (vs classical >=4); FFTW-measured slowdown near 16384 is ~8.7-10.1x prime-vs-pow2 = the
  NUMBER TO BEAT. Theory-only, no NTT analogue (it's analytic/Gaussian).
- Crypto NTT SIMD (Seiler AVX2 2018; Dilithium AVX-512 +23% 2024; Intel HEXL): butterfly-
  level SIMD (Montgomery/Barrett, lazy reduction, layer merging, register blocking) transfers
  to the pow2 convolution FFTs INSIDE Rader/Bluestein — most useful at 65537 (2^16) and
  1021->2048 (clean pow2). NTT accuracy story irrelevant (exact arithmetic).
- PFA/Good-Thomas: applies INSIDE Rader to the length-(N-1) transform — 1021's 1020=4*3*5*17
  pairwise coprime -> pure twiddle-free Good-Thomas, the elegant route.

## Bottom line
1021 -> Rader+Good-Thomas (twiddle-free), oracle=direct DFT. 65537 -> unpadded Rader (ideal).
10007/100003 -> Bluestein baseline + A/B one-level nested Rader (the FFTW-bails gap).
