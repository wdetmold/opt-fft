# benchFFT (Frigo & Johnson, benchfft-3.1) on a80n0 — community-standard harness run

Their calibrated min-timing, their 5N log2 N / time "mflops" convention, pinned core,
quiet reserved node. Our benchee wraps the final trunk (impl_12 gen_race) with plan-time
racing (setup excluded from timing, per their methodology). All 24 problems passed
benchFFT's independent --verify (impulse/linearity) checks.

## B=1 single-transform curve (the community-comparable numbers), "mflops"

| L^3 | lqcd-gen | fftw3 (measure) | ratio |
|---|---|---|---|
| 10 | 13,440 | 23,842 | fftw 1.77x |
| 12 | 17,112 | 29,724 | fftw 1.74x |
| 15 | 25,261 | 21,104 | ours 1.20x |
| 16 | 32,719 | 23,510 | ours 1.39x |
| 20 | 38,698 | 27,377 | ours 1.41x |
| 25 | 35,568 | 14,491 | ours 2.45x |
| 27 | 39,255 | 8,258 | ours 4.75x |
| 31 | 29,544 | 2,877 | ours 10.3x |
| 32 | 31,886 | 24,163 | ours 1.32x |
| 40 | 39,145 | 15,104 | ours 2.59x |
| 50 | 30,285 | 12,905 | ours 2.35x |
| 64 | 21,406 | 18,487 | ours 1.16x |
| 100 | 16,664 | 12,897 | ours 1.29x |
| 128 | 13,031 | 12,604 | ours 1.03x |

Findings this harness adds beyond our own:
1. **A real weakness found: B=1 at tiny L (10, 12).** Our small-L engines are batch-lane
   designs; single-volume fallbacks lose to FFTW's superb small codelets by ~1.75x.
   Never visible in our batch-centric suite (B=64 there). A concrete work item.
2. Primes stay dominant even at B=1 (10.3x at L=31 — FFTW's Rader collapses to 2.9
   "gflops" while our dense/Rader hybrid holds 29.5).
3. Unscored sizes hold up: 16/64/128 were never in any campaign suite; the trunk's
   raced plans win 16 and 64 and hold 128 to 3% — generality evidence in an
   independent harness.

## Caveats
- benchFFT --accuracy is rank-1 only (its arbitrary-precision reference is 1D):
  accuracy certification remains with our two-part gate machinery.
- VECTOR (batched) problems: our side runs honestly (verify passes; numbers in
  speed.txt), but the 2003 fftw3 benchee's vector path reports physically impossible
  rates (e.g. 8x32^3 at ~385 GFLOPS on one core, >2x machine peak) — its vrank
  handling does not do the full work. Cross-benchee comparisons here are B=1 ONLY;
  batched comparisons live in our own harness (bench/gen leaderboards).
