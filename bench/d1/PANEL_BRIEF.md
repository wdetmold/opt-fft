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
