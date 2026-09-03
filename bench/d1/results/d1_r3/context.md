# What previous generations produced (round d1_r3 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r2/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 235 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 271 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 219 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 197 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 227 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 220 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 234 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 280 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 239 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round d1_r2 ===
# round d1_r2
host: a80n0.lqcd.mit   date: 2026-09-03T02:33:00-04:00   slurm_job: 440424
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient                0.0219 us     0.022 us    11.00      20.9%    0.001s  ok 2.0e-16       1.00x
   d1_batchlane                  0.0229 us     0.023 us    10.50      51.0%    0.000s  ok 1.4e-16       1.05x
   d1_race                       0.0230 us     0.023 us    10.45       0.3%   11.406s  ok 1.4e-16       1.05x
   mkl1d_dfti                    0.0259 us     0.026 us     9.30       0.6%    0.014s  ok 1.8e-16       1.18x
   fftw1d_estimate               0.0264 us     0.026 us     9.12       0.4%    0.001s  ok 2.0e-16       1.21x
   fftw1d_measure                0.0264 us     0.026 us     9.11       4.7%    0.001s  ok 2.0e-16       1.21x
   d1_prime                      0.0286 us     0.029 us     8.42       1.8%    0.000s  ok 1.9e-16       1.31x
   d1_rader                      0.0562 us     0.056 us     4.28       0.1%    0.000s  ok 3.0e-16       2.57x
   fftw1d_custom                 0.0567 us     0.057 us     4.24       2.8%    0.000s  ok 2.0e-16       2.59x
   d1_planner                    0.0655 us     0.066 us     3.67      16.3%    0.000s  ok 1.7e-16       3.00x
   d1_bluestein                  0.0962 us     0.096 us     2.50      18.7%    0.000s  ok 1.7e-16       4.40x
   baseline_dft                  0.3533 us     0.353 us     0.68       0.0%    0.000s  ok 4.5e-16       16.16x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0394 us  7880.022 us     6.10      13.9%    0.009s  ok ch=1.3e-15/1e-10 1s=4e-16 1.00x
   d1_prime                      0.0395 us  7893.427 us     6.09       0.0%    0.000s  ok ch=1.3e-15/1e-10 1s=4e-16 1.00x
   fftw1d_measure                0.0674 us 13484.005 us     3.57       0.0%    0.001s  ok ch=7.9e-16/1e-10 1s=4e-16 1.71x
   fftw1d_patient                0.0674 us 13484.720 us     3.57      20.7%    0.001s  ok ch=7.9e-16/1e-10 1s=4e-16 1.71x
   d1_batchlane                  0.0720 us 14404.016 us     3.34       0.1%    0.000s  ok ch=3.4e-16/1e-10 1s=2e-16 1.83x
   fftw1d_custom                 0.0754 us 15072.170 us     3.19       0.3%    0.000s  ok ch=3.9e-16/1e-10 1s=3e-16 1.91x
   d1_rader                      0.0801 us 16022.565 us     3.00       0.0%    0.000s  ok ch=1.7e-15/1e-10 1s=7e-16 2.03x
   mkl1d_dfti                    0.0803 us 16057.432 us     3.00       0.1%    0.014s  ok ch=1.0e-15/1e-10 1s=4e-16 2.04x
   fftw1d_estimate               0.0814 us 16277.469 us     2.96       0.0%    0.001s  ok ch=7.9e-16/1e-10 1s=4e-16 2.07x
   d1_planner                    0.0885 us 17709.787 us     2.72       1.0%    0.000s  ok ch=6.7e-16/1e-10 1s=4e-16 2.25x
   d1_bluestein                  0.1271 us 25415.046 us     1.89       1.9%    0.000s  ok ch=7.6e-16/1e-10 1s=4e-16 3.23x
   baseline_dft                  0.3930 us 78596.442 us     0.61       0.0%    0.000s  ok ch=9.2e-15/1e-10 1s=1e-15 9.97x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_measure                0.0140 us     7.144 us    17.24       2.8%    0.003s  ok 1.8e-16       1.00x
   fftw1d_patient                0.0140 us     7.178 us    17.16       1.1%    0.003s  ok 1.8e-16       1.00x
   mkl1d_dfti                    0.0192 us     9.849 us    12.50       2.0%    0.050s  ok 1.7e-16       1.38x
   d1_prime                      0.0206 us    10.551 us    11.67       3.2%    0.000s  ok 1.8e-16       1.48x
   d1_race                       0.0207 us    10.615 us    11.60       2.8%    2.280s  ok 1.8e-16       1.49x
   d1_batchlane                  0.0230 us    11.799 us    10.44       1.3%    0.000s  ok 1.4e-16       1.65x
   fftw1d_custom_soa             0.0367 us    18.771 us     6.56       1.6%    0.000s  ok 1.8e-16       2.63x
   fftw1d_estimate               0.0399 us    20.422 us     6.03       0.6%    0.001s  ok 1.8e-16       2.86x
   fftw1d_custom                 0.0488 us    24.989 us     4.93       1.1%    0.000s  ok 1.8e-16       3.50x
   d1_rader                      0.0511 us    26.159 us     4.71       0.4%    0.000s  ok 2.8e-16       3.66x
   d1_planner                    0.0644 us    32.980 us     3.73       0.4%    0.000s  ok 1.4e-16       4.62x
   d1_bluestein                  0.1073 us    54.923 us     2.24       2.2%    0.000s  ok 1.8e-16       7.69x
   baseline_dft                  0.3499 us   179.137 us     0.69       0.0%    0.000s  ok 4.1e-16       25.08x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0154 us 15762.826 us    15.63      13.8%    0.000s  ok ch=1.2e-14/1e-10 1s=4e-16 1.00x
   d1_race                       0.0154 us 15793.904 us    15.59      13.7%    0.007s  ok ch=1.2e-14/1e-10 1s=4e-16 1.00x
   d1_batchlane                  0.0186 us 19079.561 us    12.91      18.5%    0.000s  ok ch=1.7e-15/1e-10 1s=3e-16 1.21x
   fftw1d_custom_soa             0.0262 us 26846.950 us     9.17       0.8%    0.000s  ok ch=7.3e-15/1e-10 1s=4e-16 1.70x
   fftw1d_custom                 0.0473 us 48467.552 us     5.08       0.1%    0.000s  ok ch=7.3e-15/1e-10 1s=4e-16 3.07x
   fftw1d_patient                0.0513 us 52498.496 us     4.69      16.7%    0.003s  ok ch=7.4e-15/1e-10 1s=3e-16 3.33x
   fftw1d_measure                0.0514 us 52683.015 us     4.68      15.9%    0.002s  ok ch=7.4e-15/1e-10 1s=3e-16 3.34x
   mkl1d_dfti                    0.0530 us 54288.174 us     4.54      20.8%    0.051s  ok ch=1.2e-14/1e-10 1s=3e-16 3.44x
   d1_planner                    0.0574 us 58729.049 us     4.19      20.6%    0.000s  ok ch=1.5e-15/1e-10 1s=3e-16 3.73x
   fftw1d_estimate               0.0746 us 76373.487 us     3.22      14.2%    0.001s  ok ch=7.4e-15/1e-10 1s=3e-16 4.85x
   d1_rader                      0.0801 us 82070.820 us     3.00      20.7%    0.000s  ok ch=1.2e-14/1e-10 1s=5e-16 5.21x
   d1_bluestein                  0.1258 us 128781.538 us     1.91       1.4%    0.000s  ok ch=9.7e-16/1e-10 1s=3e-16 8.17x
   baseline_dft                  0.3270 us 334878.393 us     0.74       0.4%    0.000s  ok ch=7.1e-14/1e-10 1s=7e-16 21.24x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0643 us     0.064 us    11.95       3.0%    0.000s  ok 3.4e-16       1.00x
   d1_race                       0.0651 us     0.065 us    11.79       8.4%    0.084s  ok 3.4e-16       1.01x
   d1_batchlane                  0.0685 us     0.068 us    11.21       4.3%    0.000s  ok 2.7e-16       1.07x
   d1_rader                      0.2103 us     0.210 us     3.65      11.9%    0.000s  ok 3.4e-16       3.27x
   fftw1d_custom                 0.2152 us     0.215 us     3.57       1.1%    0.000s  ok 2.0e-16       3.35x
   d1_planner                    0.2330 us     0.233 us     3.30       0.5%    0.000s  ok 2.4e-16       3.62x
   mkl1d_dfti                    0.2795 us     0.279 us     2.75       0.1%    0.035s  ok 2.3e-16       4.35x
   fftw1d_estimate               0.3133 us     0.313 us     2.45       4.6%    0.001s  ok 2.5e-16       4.87x
   fftw1d_patient                0.3134 us     0.313 us     2.45       1.5%    0.121s  ok 2.5e-16       4.87x
   fftw1d_measure                0.3144 us     0.314 us     2.44       1.0%    0.043s  ok 2.5e-16       4.89x
   d1_bluestein                  0.3177 us     0.318 us     2.42       2.4%    0.001s  ok 2.9e-16       4.94x
   baseline_dft                  1.9980 us     1.998 us     0.38       0.0%    0.000s  ok 3.7e-16       31.08x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0582 us  5819.590 us    13.20       0.0%    0.000s  ok ch=6.1e-15/1e-10 1s=8e-16 1.00x
   d1_race                       0.0582 us  5819.706 us    13.19       0.0%    0.005s  ok ch=6.1e-15/1e-10 1s=8e-16 1.00x
   d1_batchlane                  0.0873 us  8730.751 us     8.80       0.2%    0.000s  ok ch=2.0e-15/1e-10 1s=5e-16 1.50x
   fftw1d_custom                 0.2089 us 20890.362 us     3.68      20.6%    0.000s  ok ch=2.0e-15/1e-10 1s=5e-16 3.59x
   d1_rader                      0.2504 us 25039.591 us     3.07       0.0%    0.000s  ok ch=1.2e-14/1e-10 1s=7e-16 4.30x
   d1_bluestein                  0.3155 us 31550.700 us     2.43       7.1%    0.001s  ok ch=6.8e-15/1e-10 1s=7e-16 5.42x
   mkl1d_dfti                    0.3194 us 31941.180 us     2.40      20.7%    0.052s  ok ch=3.2e-15/1e-10 1s=4e-16 5.49x
   d1_planner                    0.3321 us 33206.941 us     2.31       0.5%    0.000s  ok ch=2.8e-15/1e-10 1s=3e-16 5.71x
   fftw1d_measure                0.3475 us 34749.825 us     2.21       0.0%    0.042s  ok ch=2.9e-15/1e-10 1s=3e-16 5.97x
   fftw1d_patient                0.3481 us 34811.614 us     2.21       0.5%    0.118s  ok ch=2.9e-15/1e-10 1s=3e-16 5.98x
   fftw1d_estimate               0.3498 us 34982.192 us     2.20       1.0%    0.001s  ok ch=2.9e-15/1e-10 1s=3e-16 6.01x
   baseline_dft                  1.7389 us 173894.010 us     0.44       0.0%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 29.88x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0539 us    27.619 us    14.24      13.6%    1.180s  ok 2.9e-16       1.00x
   d1_prime                      0.0611 us    31.265 us    12.58       1.2%    0.000s  ok 2.9e-16       1.13x
   d1_batchlane                  0.0767 us    39.247 us    10.02       2.6%    0.000s  ok 2.1e-16       1.42x
   fftw1d_custom_soa             0.1050 us    53.743 us     7.32      11.6%    0.000s  ok 2.6e-16       1.95x
   d1_rader                      0.1750 us    89.626 us     4.39      21.6%    0.000s  ok 3.6e-16       3.25x
   fftw1d_custom                 0.1948 us    99.742 us     3.94       4.5%    0.000s  ok 2.6e-16       3.61x
   d1_planner                    0.2478 us   126.899 us     3.10       0.4%    0.000s  ok 2.1e-16       4.59x
   fftw1d_estimate               0.2634 us   134.866 us     2.92      20.6%    0.001s  ok 2.1e-16       4.88x
   fftw1d_measure                0.2637 us   134.995 us     2.91      20.5%    0.043s  ok 2.1e-16       4.89x
   mkl1d_dfti                    0.2735 us   140.022 us     2.81       0.1%    0.050s  ok 2.3e-16       5.07x
   d1_bluestein                  0.3140 us   160.747 us     2.45      12.6%    0.001s  ok 3.3e-16       5.82x
   fftw1d_patient                0.3177 us   162.674 us     2.42       0.1%    0.119s  ok 2.1e-16       5.89x
   baseline_dft                  1.9939 us  1020.861 us     0.39       0.0%    0.000s  ok 4.2e-16       36.96x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0483 us 29697.616 us    15.89       1.7%    0.000s  ok ch=3.9e-12/1e-10 1s=7e-16 1.00x
   d1_race                       0.0485 us 29820.888 us    15.82       0.0%    0.005s  ok ch=3.9e-12/1e-10 1s=7e-16 1.00x
   d1_batchlane                  0.0615 us 37810.455 us    12.48      14.6%    0.000s  ok ch=5.0e-13/1e-10 1s=4e-16 1.27x
   fftw1d_custom_soa             0.0777 us 47721.906 us     9.89      13.5%    0.000s  ok ch=5.0e-13/1e-10 1s=5e-16 1.61x
   fftw1d_custom                 0.1772 us 108857.196 us     4.33       0.3%    0.000s  ok ch=5.0e-13/1e-10 1s=5e-16 3.67x
   d1_planner                    0.2184 us 134213.443 us     3.52       0.8%    0.000s  ok ch=4.5e-13/1e-10 1s=4e-16 4.52x
   d1_rader                      0.2507 us 154018.324 us     3.06       0.0%    0.000s  ok ch=4.4e-12/1e-10 1s=7e-16 5.19x
   mkl1d_dfti                    0.3153 us 193725.434 us     2.44       0.1%    0.049s  ok ch=1.5e-12/1e-10 1s=5e-16 6.52x
   d1_bluestein                  0.3157 us 193983.971 us     2.43       2.3%    0.000s  ok ch=1.3e-12/1e-10 1s=6e-16 6.53x
   fftw1d_measure                0.3534 us 217101.883 us     2.17       0.0%    0.042s  ok ch=6.0e-13/1e-10 1s=4e-16 7.31x
   fftw1d_estimate               0.3534 us 217123.260 us     2.17       0.1%    0.001s  ok ch=6.0e-13/1e-10 1s=4e-16 7.31x
   fftw1d_patient                0.3536 us 217224.777 us     2.17       0.2%    0.118s  ok ch=6.0e-13/1e-10 1s=4e-16 7.31x
   baseline_dft                  1.7449 us 1072045.040 us     0.44       0.0%    0.000s  ok ch=2.2e-12/1e-10 1s=8e-16 36.10x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0205 us     0.020 us    39.03      11.8%    0.019s  ok 1.2e-16       1.00x
   d1_pow2                       0.0207 us     0.021 us    38.69       0.9%    0.000s  ok 1.2e-16       1.01x
   mkl1d_dfti                    0.0254 us     0.025 us    31.49       1.2%    0.002s  ok 1.6e-16       1.24x
   d1_twiddle                    0.0340 us     0.034 us    23.50      13.0%    0.000s  ok 9.3e-17       1.66x
   fftw1d_patient                0.0372 us     0.037 us    21.48       8.6%    0.015s  ok 1.7e-16       1.82x
   d1_batchlane                  0.0381 us     0.038 us    21.00      10.5%    0.000s  ok 9.9e-17       1.86x
   fftw1d_measure                0.0386 us     0.039 us    20.75      11.9%    0.008s  ok 1.7e-16       1.88x
   fftw1d_estimate               0.0484 us     0.048 us    16.52       4.2%    0.001s  ok 1.8e-16       2.36x
   d1_planner                    0.0620 us     0.062 us    12.90       1.1%    0.000s  ok 1.1e-16       3.03x
   d1_bluestein                  0.1005 us     0.100 us     7.96       6.1%    0.000s  ok 1.1e-16       4.90x
   fftw1d_custom                 0.1083 us     0.108 us     7.39       8.0%    0.000s  ok 1.2e-16       5.28x
   baseline_dft                  2.1284 us     2.128 us     0.38       0.0%    0.000s  ok 2.9e-16       103.84x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_batchlane                  0.0791 us  7908.789 us    10.12       0.1%    0.000s  ok ch=1.6e-15/1e-10 1s=3e-16 1.00x
   d1_race                       0.0792 us  7917.303 us    10.10       0.1%    0.007s  ok ch=1.6e-15/1e-10 1s=3e-16 1.00x
   d1_twiddle                    0.0829 us  8287.701 us     9.65      13.8%    0.000s  ok ch=1.1e-15/1e-10 1s=3e-16 1.05x
   d1_pow2                       0.0976 us  9760.650 us     8.20       0.0%    0.000s  ok ch=1.5e-15/1e-10 1s=3e-16 1.23x
   d1_planner                    0.1215 us 12152.103 us     6.58       3.5%    0.000s  ok ch=1.6e-15/1e-10 1s=3e-16 1.54x
   fftw1d_custom                 0.1216 us 12159.573 us     6.58       0.0%    0.000s  ok ch=1.9e-15/1e-10 1s=3e-16 1.54x
   mkl1d_dfti                    0.1310 us 13098.877 us     6.11      13.9%    0.002s  ok ch=1.4e-15/1e-10 1s=3e-16 1.66x
   fftw1d_patient                0.1359 us 13593.717 us     5.89      13.9%    0.015s  ok ch=2.8e-15/1e-10 1s=4e-16 1.72x
   fftw1d_measure                0.1361 us 13606.904 us     5.88      11.5%    0.007s  ok ch=2.8e-15/1e-10 1s=3e-16 1.72x
   fftw1d_estimate               0.1452 us 14522.726 us     5.51      15.6%    0.001s  ok ch=2.9e-15/1e-10 1s=4e-16 1.84x
   d1_bluestein                  0.2083 us 20828.183 us     3.84       1.9%    0.000s  ok ch=1.6e-15/1e-10 1s=3e-16 2.63x
   baseline_dft                  1.8469 us 184694.049 us     0.43       0.0%    0.000s  ok ch=2.3e-14/1e-10 1s=5e-16 23.35x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0153 us     7.827 us    52.33       0.0%    0.001s  ok 1.4e-16       1.00x
   d1_race                       0.0154 us     7.898 us    51.86      13.8%    0.662s  ok 1.4e-16       1.01x
   d1_pow2                       0.0176 us     8.987 us    45.58       0.0%    0.000s  ok 1.4e-16       1.15x
   fftw1d_measure                0.0251 us    12.828 us    31.93       4.7%    0.010s  ok 1.6e-16       1.64x
   fftw1d_patient                0.0253 us    12.929 us    31.68      13.4%    0.061s  ok 1.6e-16       1.65x
   d1_twiddle                    0.0335 us    17.157 us    23.87       6.5%    0.000s  ok 1.4e-16       2.19x
   d1_batchlane                  0.0359 us    18.377 us    22.29       1.7%    0.000s  ok 1.1e-16       2.35x
   d1_planner                    0.0531 us    27.183 us    15.07       3.2%    0.000s  ok 1.4e-16       3.47x
   fftw1d_custom_soa             0.0760 us    38.910 us    10.53      25.8%    0.000s  ok 1.4e-16       4.97x
   d1_bluestein                  0.0931 us    47.689 us     8.59       7.4%    0.000s  ok 1.4e-16       6.09x
   fftw1d_estimate               0.1005 us    51.439 us     7.96       1.3%    0.001s  ok 1.5e-16       6.57x
   fftw1d_custom                 0.1050 us    53.771 us     7.62       6.7%    0.000s  ok 1.4e-16       6.87x
   baseline_dft                  2.1240 us  1087.503 us     0.38       0.0%    0.000s  ok 3.4e-16       138.95x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0384 us 19657.084 us    20.84      13.7%    0.007s  ok ch=1.0e-13/1e-10 1s=3e-16 1.00x
   d1_batchlane                  0.0439 us 22491.546 us    18.21       0.6%    0.000s  ok ch=1.0e-13/1e-10 1s=3e-16 1.14x
   d1_pow2                       0.0461 us 23616.179 us    17.34       0.2%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-16 1.20x
   fftw1d_custom_soa             0.0665 us 34061.437 us    12.03       0.5%    0.000s  ok ch=9.4e-14/1e-10 1s=3e-16 1.73x
   d1_twiddle                    0.0830 us 42500.173 us     9.64       1.5%    0.000s  ok ch=8.7e-14/1e-10 1s=3e-16 2.16x
   d1_planner                    0.1004 us 51417.643 us     7.97      21.0%    0.000s  ok ch=1.0e-13/1e-10 1s=3e-16 2.62x
   fftw1d_custom                 0.1040 us 53224.118 us     7.70       0.3%    0.000s  ok ch=9.4e-14/1e-10 1s=3e-16 2.71x
   mkl1d_dfti                    0.1122 us 57471.551 us     7.13      14.3%    0.001s  ok ch=1.3e-13/1e-10 1s=3e-16 2.92x
   fftw1d_measure                0.1222 us 62549.456 us     6.55       1.1%    0.009s  ok ch=1.0e-13/1e-10 1s=4e-16 3.18x
   fftw1d_patient                0.1223 us 62622.169 us     6.54      15.4%    0.061s  ok ch=1.0e-13/1e-10 1s=4e-16 3.19x
   d1_bluestein                  0.1685 us 86262.148 us     4.75       4.8%    0.000s  ok ch=1.0e-13/1e-10 1s=3e-16 4.39x
   fftw1d_estimate               0.1858 us 95136.741 us     4.31       0.2%    0.001s  ok ch=1.3e-13/1e-10 1s=3e-16 4.84x
   baseline_dft                  1.9509 us 998846.568 us     0.41       0.1%    0.000s  ok ch=4.0e-13/1e-10 1s=6e-16 50.81x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_composite                  0.0534 us     0.053 us    33.16       4.1%    0.000s  ok 2.2e-16       1.00x
   d1_race                       0.0542 us     0.054 us    32.69       3.9%    0.012s  ok 2.2e-16       1.01x
   mkl1d_dfti                    0.0622 us     0.062 us    28.49       4.6%    0.002s  ok 2.6e-16       1.16x
   fftw1d_patient                0.0702 us     0.070 us    25.26      24.6%    0.128s  ok 2.1e-16       1.31x
