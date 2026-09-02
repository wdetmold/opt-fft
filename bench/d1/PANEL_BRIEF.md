# 1D FFT competition — beat the libraries across all four call regimes

Separate campaign from the 3D generalize work (this is NOT an arbitrary-d library — it is a
focused 1D contest). Deliverable: a 1D complex-double FFT that beats the best stock library
(FFTW estimate/measure/patient, MKL DFTI) at each of SIX sizes, in EACH OF FOUR call regimes.

## The metric: four regimes, scored SEPARATELY (never averaged across regimes)
Per size L, four cells (a 2x2 of batch x chain), each scored as ratio vs the best correct
library at THAT cell:
  1. NON-BATCHED, NON-CHAINED  (B=1,   m=1)   -- single small transform; latency-bound;
       FFTW/MKL's home turf, the hardest cell to win. benchFFT's regime.
  2. BATCHED,     NON-CHAINED  (B=512, m=1)   -- throughput; the batch-lane advantage.
  3. NON-BATCHED, CHAINED      (B=1,   m=big) -- fused FFT+map chain; latency amortized,
       the fused-map advantage (own fft1d_chain).
  4. BATCHED,     CHAINED      (B=512, m=big) -- both advantages; the "production" regime.
These are DIFFERENT contests (like the 3D campaign's B=1-vs-batched rule). Report the 2x2
ratio grid per size; aggregate as a geometric mean of ratios WITHIN each regime across
sizes (four aggregate numbers), plus per-size grids. A single number is forbidden.
Also report benchFFT mflops (5 L log2 L / t_us) for community comparability at the
non-chained cells.

## Why 1D is HARD (be honest with yourself)
1D is where the libraries are most obsessively tuned, everything is L1-resident (compute-
bound, not memory-bound), and our 3D structural edge -- a prime axis-DFT amortized L^2
times per volume -- DOES NOT EXIST here (in 1D the only amortization is across the batch).
So: the non-batched non-chained cell is genuinely hard and may not win at pow2. Your
leverage is (a) small primes/composites where libraries fall back to Rader/Bluestein,
(b) the batched regime (fill zmm lanes across the batch: 8 vectors per register,
zero-shuffle split-complex), (c) the fused chain (own fft1d_chain, no per-step round-trip
or map barrier). Win where you can win; do not fake the cells you cannot.

## Classes (own a class, cross-class entries welcome; the race arbitrates)
d1_pow2 (16/32/64/128/256 radix-2^k) | d1_prime (7/11/13/17/31 dense/Winograd) |
d1_rader (13/31/127 Rader convolution) | d1_composite (12/24/36/60 PFA/mixed) |
d1_bluestein (any L fallback) | d1_batchlane (SoA 8-vec/zmm across the batch).
Library layers (adoption-scored): d1_planner, d1_race, d1_twiddle.

## Gates, timing, dev (identical machinery to the 3D campaign)
Two-part chaotic-chain gate for chained cells (self-calibrating anchor); single-call rel L2
< 1e-12 for non-chained. Timing after compile+warmup, min over runs, spread reported.
Develop on wallaby / leased node cores; never submit slurm jobs. Static analyzers
(llvm-mca icelake-server) + PMU (/tmp/perf, tools/pmu.sh) live -- 1D is compute-bound so
port_0/port_5 dispatch and instruction density are your dashboards, not memory counters.
Maintain your strategies/<name>.md each round.

## SIZE RANGE EXTENDED (added mid-r1): large application 1D sizes
Real 1D FFT applications live at 1K-64K (audio frames, spectroscopy, correlations), pure
powers of two -- FFTW's and MKL's most-tuned home turf. Added: 1024, 4096, 16384 (all four
regimes each). REGIME SHIFT to know: at these L the BATCHED cells go memory-bound (L=16384
x B=64 = 16 MB, beyond L2), so the r11 3D-campaign traffic lessons apply -- l1d.replacement
and L2/L3 traffic become the dashboard, not just port dispatch. Expect TIGHTER margins here
than at small L: pow2 at 4K/16K is exactly where FFTW's decades of tuning are strongest.
Winning even by 1.05-1.2x at 4096/16384 non-batched is a real result; the batched and
chained columns are where our lane-fill and fused-map edges should still open daylight.

## LARGE PRIMES (added before the fresh r1): the real large-size contest
Dense O(L^2) is impossible at 10^3..10^6, so EVERYONE (us, FFTW, MKL) uses Rader or
Bluestein -- and this is where FFTW is WEAKEST relative to its pow2 peak, i.e. our best
large-L shot. The decisive fact: Rader turns a prime-N DFT into an (N-1)-POINT CONVOLUTION,
so the factorization of N-1 sets the cost. The four primes are chosen to contrast:
  1021   N-1 = 2^2*3*5*17   smooth      -> Rader's (N-1)-conv is itself easy
  10007  N-1 = 2*5003       awkward     -> 5003 prime; Rader-conv is hard, Bluestein may win
  65537  N-1 = 2^16         RADER-IDEAL -> (N-1)-conv is a clean pow2 FFT; Rader should crush
  100003 N-1 = 2*3*7*2381   awkward     -> 2381 prime; awkward, Bluestein territory
d1_rader and d1_bluestein owners: this is your headline. Pick Rader where N-1 is smooth/
pow2 (65537, 1021), Bluestein (chirp-z, pad to a convenient pow2) where N-1 is awkward
(10007, 100003) -- and MEASURE the crossover, do not assume it. Compare against FFTW/MKL,
which make this choice internally and often badly. Batched large primes go memory-bound
(100003 x B=8 = 12.8 MB); the r11 traffic lessons apply.

## LITERATURE (docs/literature_1d/00-SURVEY.md) — read before r2
A 5-vein survey of 1D FFT optimization (CPU/batched/GPU/prime/accuracy). Actionable now,
by class:
- ALL: across-batch split-complex vectorization (lane j = transform j, ZERO shuffles, >=8
  transforms) is the top under-used lever for the batched regime; twiddle/chirp tables from
  correctly-rounded/dd sincos in the plan stage, never in-loop recurrences.
- d1_rader/d1_bluestein (the headline): libraries DEFAULT TO BLUESTEIN even where Rader wins.
  65537 -> UNPADDED Rader (conv = 65536 = 2^16, reuse pow2 butterflies) crushes Bluestein-only
  libs. 1021 -> Rader + twiddle-free Good-Thomas (1020=4*3*5*17). 10007/100003 -> Bluestein
  baseline BUT A/B a one-level nested Rader (both inner primes have smooth p-1) — that's
  exactly where FFTW's planner bails to Bluestein. Chirp: k^2 mod 2N in integers first.
  Number to beat: ~8-10x prime-vs-pow2 slowdown (FFTW, Harvey-vdHoeven).
- d1_pow2: Stockham (no bit-reversal) + conjugate-pair split-radix (half the twiddle traffic,
  better than 34/9); 16384 -> four-step 128x128 (L1-resident sub-FFTs).
- d1_composite: Good-Thomas PFA (coprime, twiddle-free).
- accuracy: our map is a contraction (bounded chain); gate tiers = dd reference / NTT-exact /
  Arb acb_dft provable checkpoint.
Study FFTS (Blake 2013) for the fixed-geometry specialize-then-run model closest to ours.
