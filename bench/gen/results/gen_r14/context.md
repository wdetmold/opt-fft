# What previous generations produced (round gen_r14 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r10/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r11/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r12/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r13/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r4/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r5/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r6/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r7/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r8/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r9/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/xarch_clx_r6/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/xarch_spr_r5/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_batchlane.md 1995 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_bluestein.md 1565 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_dense_prime.md 2131 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_layout.md 2067 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_large.md 2019 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_small.md 1878 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_planner.md 1968 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pow2.md 1881 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_powp.md 2257 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_race.md 1931 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_rader.md 2030 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_twiddle.md 2162 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gen_r13 ===
# round gen_r13
host: a80n0.lqcd.mit   date: 2026-09-01T18:20:17-04:00   slurm_job: 439820
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (non-batched, chain m=16384), volume 1000, working set 0.03 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       1.943 us 31828.182 us    25.65      14.0%    0.218s  ok ch=8.9e-07/3e-03 1s=1e-15 1.00x
   gen_pfa_small                  1.943 us 31832.797 us    25.65      14.0%    0.000s  ok ch=8.9e-07/3e-03 1s=1e-15 1.00x
   gen_batchlane                  2.370 us 38823.000 us    21.03       0.1%    0.001s  ok ch=4.3e-06/3e-03 1s=1e-15 1.22x
   gen_planner                    2.549 us 41757.670 us    19.55      13.4%    0.004s  ok ch=4.8e-06/3e-03 1s=8e-16 1.31x
   gen_dense_prime                2.982 us 48850.567 us    16.71      17.6%    0.000s  ok ch=2.8e-06/3e-03 1s=1e-15 1.53x
   gen_twiddle                    4.294 us 70357.879 us    11.60      13.8%    0.004s  ok ch=3.9e-06/3e-03 1s=9e-16 2.21x
   mkl_dfti                       4.331 us 70963.939 us    11.50       0.7%    0.002s  ok ch=8.5e-07/3e-03 1s=1e-15 2.23x
   mkl2026_dfti                   4.471 us 73249.186 us    11.15       1.6%    0.002s  ok ch=3.4e-06/3e-03 1s=1e-15 2.30x
   fftw3_patient                  4.637 us 75979.536 us    10.74       8.2%    0.021s  ok ch=9.3e-06/3e-03 1s=8e-16 2.39x
   gen_layout                     5.008 us 82052.216 us     9.95       2.7%    0.000s  ok ch=1.2e-05/3e-03 1s=1e-15 2.58x
   fftw3_measure                  5.119 us 83874.162 us     9.73       0.9%    0.013s  ok ch=3.6e-06/3e-03 1s=9e-16 2.64x
   fftw3_custom                   5.464 us 89519.635 us     9.12       1.0%    0.000s  ok ch=6.6e-06/3e-03 1s=9e-16 2.81x
   fftw3_guru                     6.336 us 103804.013 us     7.86       0.6%    0.011s  ok ch=1.1e-06/3e-03 1s=8e-16 3.26x
   fftw3_estimate                 7.244 us 118683.263 us     6.88       1.7%    0.001s  ok ch=9.3e-06/3e-03 1s=8e-16 3.73x
   gen_bluestein                  8.623 us 141279.819 us     5.78       2.7%    0.000s  ok ch=1.1e-05/3e-03 1s=1e-15 4.44x
   ducc0_c2c                     11.088 us 181661.617 us     4.49       1.1%    0.000s  ok ch=1.0e-06/3e-03 1s=8e-16 5.71x
   baseline_matrix               55.005 us 901200.876 us     0.91       0.0%    0.000s  ok ch=1.1e-06/3e-03 1s=1e-15 28.31x

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.147 us 73391.091 us    43.45      11.1%    0.001s  ok ch=1.2e-13/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.147 us 73403.828 us    43.45       0.1%    0.000s  ok ch=1.2e-13/1e-10 1s=9e-16 1.00x
   gen_race                       1.149 us 73511.472 us    43.38      11.3%    0.010s  ok ch=1.2e-13/1e-10 1s=9e-16 1.00x
   gen_planner                    1.362 us 87189.420 us    36.58      10.6%    0.005s  ok ch=1.6e-13/1e-10 1s=1e-15 1.19x
   gen_dense_prime                3.055 us 195494.503 us    16.31       1.5%    0.000s  ok ch=1.2e-13/1e-10 1s=9e-16 2.66x
   gen_twiddle                    4.381 us 280406.536 us    11.37       1.1%    0.003s  ok ch=1.4e-13/1e-10 1s=9e-16 3.82x
   fftw3_custom_soa               4.481 us 286779.483 us    11.12       2.8%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 3.91x
   mkl_dfti                       4.652 us 297739.711 us    10.71       1.0%    0.002s  ok ch=1.4e-13/1e-10 1s=1e-15 4.06x
   mkl2026_dfti                   4.762 us 304756.704 us    10.46       0.4%    0.002s  ok ch=1.3e-13/1e-10 1s=1e-15 4.15x
   gen_layout                     5.133 us 328540.870 us     9.71       2.7%    0.000s  ok ch=1.2e-13/1e-10 1s=9e-16 4.48x
   fftw3_patient                  5.192 us 332283.945 us     9.60       3.2%    0.022s  ok ch=1.1e-13/1e-10 1s=8e-16 4.53x
   fftw3_measure                  5.201 us 332832.664 us     9.58       0.5%    0.012s  ok ch=9.5e-14/1e-10 1s=8e-16 4.54x
   fftw3_custom                   6.077 us 388907.943 us     8.20       0.8%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 5.30x
   fftw3_guru                     6.520 us 417301.848 us     7.64       0.3%    0.011s  ok ch=1.3e-13/1e-10 1s=9e-16 5.69x
   fftw3_estimate                 7.311 us 467914.831 us     6.82       1.8%    0.001s  ok ch=9.5e-14/1e-10 1s=8e-16 6.38x
   gen_bluestein                  8.826 us 564866.562 us     5.65       2.0%    0.000s  ok ch=1.9e-13/1e-10 1s=1e-15 7.70x
   ducc0_c2c                      9.917 us 634669.777 us     5.02       0.1%    0.000s  ok ch=1.1e-13/1e-10 1s=7e-16 8.65x
   baseline_matrix               56.299 us 3603145.640 us     0.89       0.2%    0.000s  ok ch=3.5e-13/1e-10 1s=1e-15 49.10x

-- L=12 (non-batched, chain m=12288), volume 1728, working set 0.05 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  2.999 us 36855.196 us    30.98      13.8%    0.000s  ok ch=1.7e-08/1e-05 1s=9e-16 1.00x
   gen_race                       3.001 us 36880.938 us    30.96      13.8%    0.239s  ok ch=1.7e-08/1e-05 1s=9e-16 1.00x
   gen_planner                    3.471 us 42655.228 us    26.77      16.1%    0.004s  ok ch=1.7e-08/1e-05 1s=9e-16 1.16x
   gen_batchlane                  3.648 us 44823.171 us    25.47       0.6%    0.001s  ok ch=1.3e-08/1e-05 1s=9e-16 1.22x
   gen_dense_prime                4.742 us 58269.392 us    19.60      12.7%    0.000s  ok ch=2.5e-08/1e-05 1s=1e-15 1.58x
   gen_twiddle                    5.344 us 65665.598 us    17.39      16.2%    0.004s  ok ch=1.5e-08/1e-05 1s=9e-16 1.78x
   mkl_dfti                       7.319 us 89933.411 us    12.70      13.8%    0.001s  ok ch=1.3e-08/1e-05 1s=9e-16 2.44x
   mkl2026_dfti                   7.385 us 90749.510 us    12.58      13.3%    0.003s  ok ch=2.1e-08/1e-05 1s=9e-16 2.46x
   fftw3_patient                  8.176 us 100468.851 us    11.36       4.0%    0.023s  ok ch=2.5e-08/1e-05 1s=9e-16 2.73x
   fftw3_measure                  8.214 us 100927.760 us    11.31       8.8%    0.015s  ok ch=1.5e-08/1e-05 1s=9e-16 2.74x
   gen_layout                     8.412 us 103363.965 us    11.05       1.1%    0.000s  ok ch=1.2e-08/1e-05 1s=9e-16 2.80x
   fftw3_custom                   8.627 us 106006.461 us    10.77       0.1%    0.000s  ok ch=1.3e-08/1e-05 1s=9e-16 2.88x
   fftw3_guru                    10.122 us 124383.404 us     9.18       2.3%    0.012s  ok ch=2.1e-08/1e-05 1s=9e-16 3.37x
   gen_bluestein                 12.172 us 149566.954 us     7.63       5.3%    0.000s  ok ch=5.4e-08/1e-05 1s=2e-15 4.06x
   ducc0_c2c                     17.152 us 210764.781 us     5.42       1.9%    0.000s  ok ch=1.5e-08/1e-05 1s=6e-16 5.72x
   fftw3_estimate                19.078 us 234430.760 us     4.87       0.1%    0.001s  ok ch=6.8e-09/1e-05 1s=9e-16 6.36x
   baseline_matrix              112.904 us 1387365.430 us     0.82       0.0%    0.000s  ok ch=6.0e-08/1e-05 1s=2e-15 37.64x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       1.947 us 74780.703 us    47.72      12.7%    0.014s  ok ch=4.6e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.952 us 74944.238 us    47.61       0.8%    0.000s  ok ch=4.6e-14/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.955 us 75063.531 us    47.54       2.6%    0.000s  ok ch=4.6e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.398 us 92099.816 us    38.74      11.0%    0.008s  ok ch=4.6e-14/1e-10 1s=9e-16 1.23x
   gen_dense_prime                4.812 us 184782.754 us    19.31       1.2%    0.000s  ok ch=4.6e-14/1e-10 1s=9e-16 2.47x
   gen_twiddle                    5.471 us 210088.705 us    16.98       1.6%    0.004s  ok ch=4.6e-14/1e-10 1s=9e-16 2.81x
   mkl_dfti                       7.912 us 303805.834 us    11.75       0.6%    0.003s  ok ch=4.2e-14/1e-10 1s=9e-16 4.06x
   mkl2026_dfti                   7.944 us 305038.939 us    11.70       0.2%    0.002s  ok ch=4.3e-14/1e-10 1s=9e-16 4.08x
   fftw3_custom_soa               8.178 us 314043.171 us    11.36       5.9%    0.000s  ok ch=4.2e-14/1e-10 1s=9e-16 4.20x
   gen_layout                     8.426 us 323560.612 us    11.03       3.5%    0.000s  ok ch=4.6e-14/1e-10 1s=1e-15 4.33x
   fftw3_patient                  8.632 us 331472.899 us    10.76       4.3%    0.024s  ok ch=4.8e-14/1e-10 1s=9e-16 4.43x
   fftw3_measure                  8.809 us 338278.219 us    10.55       1.5%    0.014s  ok ch=4.5e-14/1e-10 1s=9e-16 4.52x
   fftw3_guru                    10.386 us 398828.124 us     8.95       0.9%    0.010s  ok ch=4.3e-14/1e-10 1s=9e-16 5.33x
   fftw3_custom                  10.821 us 415535.980 us     8.59       2.9%    0.000s  ok ch=4.2e-14/1e-10 1s=9e-16 5.56x
   gen_bluestein                 13.654 us 524309.595 us     6.81       2.1%    0.000s  ok ch=1.1e-13/1e-10 1s=2e-15 7.01x
   ducc0_c2c                     16.314 us 626466.593 us     5.70       1.0%    0.000s  ok ch=3.1e-14/1e-10 1s=7e-16 8.38x
   fftw3_estimate                19.303 us 741237.244 us     4.81       0.3%    0.001s  ok ch=4.4e-14/1e-10 1s=9e-16 9.91x
   baseline_matrix              114.728 us 4405540.940 us     0.81       0.1%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 58.91x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  4.409 us 84654.049 us    44.86       3.5%    0.000s  ok ch=5.4e-14/1e-10 1s=1e-15 1.00x
   gen_race                       4.415 us 84766.889 us    44.80      12.6%    0.233s  ok ch=5.4e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                  4.420 us 84873.370 us    44.74       0.6%    0.000s  ok ch=5.4e-14/1e-10 1s=1e-15 1.00x
   gen_planner                    5.615 us 107798.814 us    35.23       1.7%    0.005s  ok ch=4.6e-14/1e-10 1s=1e-15 1.27x
   gen_dense_prime               10.888 us 209049.275 us    18.17       1.7%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 2.47x
   gen_twiddle                   14.716 us 282545.560 us    13.44       1.5%    0.003s  ok ch=5.1e-14/1e-10 1s=1e-15 3.34x
   fftw3_custom_soa              15.939 us 306034.785 us    12.41       1.7%    0.000s  ok ch=4.9e-14/1e-10 1s=1e-15 3.62x
   mkl_dfti                      16.844 us 323395.785 us    11.74       0.2%    0.002s  ok ch=5.6e-14/1e-10 1s=1e-15 3.82x
   mkl2026_dfti                  16.972 us 325854.071 us    11.65       1.4%    0.003s  ok ch=5.7e-14/1e-10 1s=1e-15 3.85x
   fftw3_patient                 19.541 us 375196.728 us    10.12       1.3%    0.020s  ok ch=5.5e-14/1e-10 1s=1e-15 4.43x
   fftw3_measure                 19.631 us 376917.400 us    10.08       6.3%    0.011s  ok ch=5.5e-14/1e-10 1s=1e-15 4.45x
   gen_layout                    20.077 us 385477.438 us     9.85       3.3%    0.001s  ok ch=5.2e-14/1e-10 1s=1e-15 4.55x
   fftw3_estimate                21.111 us 405331.238 us     9.37       0.6%    0.001s  ok ch=5.5e-14/1e-10 1s=1e-15 4.79x
   fftw3_custom                  25.831 us 495948.694 us     7.66       1.7%    0.000s  ok ch=4.9e-14/1e-10 1s=1e-15 5.86x
   fftw3_guru                    26.757 us 513735.666 us     7.39       1.2%    0.010s  ok ch=4.9e-14/1e-10 1s=1e-15 6.07x
   ducc0_c2c                     32.487 us 623747.746 us     6.09       1.4%    0.000s  ok ch=4.1e-14/1e-10 1s=1e-15 7.37x
   gen_bluestein                 33.725 us 647527.363 us     5.86       0.6%    0.000s  ok ch=8.1e-14/1e-10 1s=2e-15 7.65x
   baseline_matrix              278.560 us 5348343.110 us     0.71       0.1%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 63.18x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      12.846 us 105230.606 us    40.37       2.4%    0.014s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                 12.849 us 105255.753 us    40.36       5.0%    0.001s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                 12.849 us 105258.574 us    40.36       1.5%    0.001s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_planner                   17.543 us 143710.015 us    29.56       2.0%    0.018s  ok ch=3.5e-14/1e-10 1s=1e-15 1.37x
   gen_twiddle                   26.155 us 214261.693 us    19.83       2.7%    0.004s  ok ch=3.0e-14/1e-10 1s=1e-15 2.04x
   gen_dense_prime               35.075 us 287331.383 us    14.79       0.6%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 2.73x
   gen_layout                    35.279 us 289008.918 us    14.70       3.1%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 2.75x
   fftw3_custom_soa              42.805 us 350659.587 us    12.12       6.1%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 3.33x
   fftw3_patient                 44.871 us 367584.206 us    11.56       1.2%    0.299s  ok ch=3.1e-14/1e-10 1s=1e-15 3.49x
   fftw3_measure                 45.046 us 369017.643 us    11.51       0.3%    0.085s  ok ch=2.5e-14/1e-10 1s=1e-15 3.51x
   mkl2026_dfti                  58.610 us 480134.623 us     8.85       3.1%    0.054s  ok ch=3.6e-14/1e-10 1s=1e-15 4.56x
   mkl_dfti                      59.188 us 484868.840 us     8.76       0.6%    0.032s  ok ch=3.2e-14/1e-10 1s=1e-15 4.61x
   fftw3_guru                    61.317 us 502305.890 us     8.46       2.9%    0.078s  ok ch=2.8e-14/1e-10 1s=1e-15 4.77x
   fftw3_custom                  64.127 us 525326.023 us     8.09       4.7%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 4.99x
   gen_bluestein                 65.389 us 535666.460 us     7.93       3.6%    0.000s  ok ch=4.6e-14/1e-10 1s=2e-15 5.09x
   ducc0_c2c                     72.454 us 593546.069 us     7.16       3.4%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 5.64x
   fftw3_estimate                91.998 us 753646.915 us     5.64       0.3%    0.001s  ok ch=2.5e-14/1e-10 1s=1e-15 7.16x
   baseline_matrix              868.176 us 7112096.590 us     0.60       0.3%    0.000s  ok ch=9.6e-14/1e-10 1s=2e-15 67.59x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      31.369 us 128487.366 us    34.70       2.4%    0.406s  ok ch=3.6e-14/1e-10 1s=2e-15 1.00x
   gen_race                      31.378 us 128525.758 us    34.69       2.2%    0.014s  ok ch=3.6e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   40.777 us 167022.838 us    26.69       1.8%    0.028s  ok ch=4.1e-14/1e-10 1s=2e-15 1.30x
   gen_twiddle                   75.473 us 309135.380 us    14.42       1.0%    0.003s  ok ch=4.7e-14/1e-10 1s=2e-15 2.41x
   fftw3_custom_soa              76.278 us 312434.724 us    14.27       4.3%    0.000s  ok ch=4.4e-14/1e-10 1s=1e-15 2.43x
   gen_layout                    97.360 us 398785.471 us    11.18       0.3%    0.000s  ok ch=4.1e-14/1e-10 1s=2e-15 3.10x
   fftw3_patient                109.537 us 448664.161 us     9.94       3.6%    0.103s  ok ch=3.6e-14/1e-10 1s=2e-15 3.49x
   fftw3_measure                111.466 us 456563.035 us     9.76       5.3%    0.029s  ok ch=3.6e-14/1e-10 1s=2e-15 3.55x
   mkl_dfti                     123.064 us 504072.014 us     8.84       2.1%    0.031s  ok ch=3.8e-14/1e-10 1s=2e-15 3.92x
   mkl2026_dfti                 125.565 us 514313.986 us     8.67       1.8%    0.049s  ok ch=3.8e-14/1e-10 1s=2e-15 4.00x
   fftw3_estimate               132.829 us 544067.961 us     8.19       5.2%    0.001s  ok ch=4.1e-14/1e-10 1s=2e-15 4.23x
   ducc0_c2c                    145.227 us 594850.644 us     7.49       3.1%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 4.63x
   fftw3_custom                 163.622 us 670196.938 us     6.65       2.3%    0.000s  ok ch=4.4e-14/1e-10 1s=1e-15 5.22x
   gen_bluestein                166.823 us 683309.016 us     6.52       2.2%    0.000s  ok ch=5.6e-14/1e-10 1s=2e-15 5.32x
   fftw3_guru                   178.716 us 732022.752 us     6.09       7.1%    0.033s  ok ch=3.4e-14/1e-10 1s=2e-15 5.70x
   baseline_matrix             2101.963 us 8609638.510 us     0.52       0.2%    0.000s  ok ch=1.6e-13/1e-10 1s=3e-15 67.01x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      43.893 us 140456.411 us    31.98       0.6%    0.522s  ok ch=3.5e-14/1e-10 1s=2e-15 1.00x
   gen_race                      44.106 us 141139.813 us    31.83       1.6%    0.016s  ok ch=3.5e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   61.300 us 196158.866 us    22.90       1.4%    0.036s  ok ch=3.9e-14/1e-10 1s=2e-15 1.40x
   fftw3_custom_soa              97.955 us 313455.326 us    14.33       0.3%    0.000s  ok ch=8.0e-14/1e-10 1s=2e-15 2.23x
   gen_twiddle                  123.691 us 395812.508 us    11.35       1.1%    0.005s  ok ch=4.9e-14/1e-10 1s=2e-15 2.82x
   gen_layout                   125.951 us 403042.799 us    11.15       2.2%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 2.87x
   mkl_dfti                     146.844 us 469900.628 us     9.56       0.0%    0.034s  ok ch=6.8e-14/1e-10 1s=2e-15 3.35x
   mkl2026_dfti                 150.630 us 482016.329 us     9.32       0.4%    0.048s  ok ch=6.8e-14/1e-10 1s=2e-15 3.43x
   ducc0_c2c                    189.798 us 607352.864 us     7.40       0.9%    0.000s  ok ch=8.1e-14/1e-10 1s=1e-15 4.32x
   gen_bluestein                198.379 us 634812.295 us     7.08       5.5%    0.000s  ok ch=4.8e-14/1e-10 1s=2e-15 4.52x
   fftw3_patient                201.926 us 646164.703 us     6.95       2.6%    0.150s  ok ch=4.3e-14/1e-10 1s=2e-15 4.60x
   fftw3_custom                 203.543 us 651337.328 us     6.90       1.7%    0.000s  ok ch=8.0e-14/1e-10 1s=2e-15 4.64x
   fftw3_measure                227.595 us 728303.062 us     6.17       1.8%    0.034s  ok ch=7.0e-14/1e-10 1s=2e-15 5.19x
   fftw3_estimate               257.869 us 825181.524 us     5.44       1.3%    0.003s  ok ch=5.8e-14/1e-10 1s=2e-15 5.88x
   fftw3_guru                   264.924 us 847755.279 us     5.30       1.0%    0.044s  ok ch=5.1e-14/1e-10 1s=2e-15 6.04x
   baseline_matrix             2852.948 us 9129434.290 us     0.49       0.1%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 65.00x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     86.176 us 193033.231 us    25.69       0.8%    0.009s  ok ch=4.0e-14/1e-10 1s=2e-15 1.00x
   gen_race                      86.909 us 194675.395 us    25.47       0.5%    0.030s  ok ch=4.0e-14/1e-10 1s=2e-15 1.01x
   gen_dense_prime              111.668 us 250135.557 us    19.83       0.8%    0.001s  ok ch=2.9e-14/1e-10 1s=2e-15 1.30x
   gen_planner                  141.964 us 317999.310 us    15.59       3.4%    0.046s  ok ch=3.3e-14/1e-10 1s=2e-15 1.65x
   gen_layout                   202.894 us 454483.470 us    10.91       1.1%    0.001s  ok ch=3.4e-14/1e-10 1s=2e-15 2.35x
   fftw3_custom_soa             209.645 us 469604.060 us    10.56       4.8%    0.000s  ok ch=3.9e-14/1e-10 1s=2e-15 2.43x
   gen_twiddle                  263.235 us 589646.719 us     8.41       1.3%    0.006s  ok ch=3.4e-14/1e-10 1s=2e-15 3.05x
   gen_bluestein                276.687 us 619779.400 us     8.00       1.4%    0.000s  ok ch=4.7e-14/1e-10 1s=2e-15 3.21x
   fftw3_custom                 516.866 us 1157778.880 us     4.28       0.6%    0.000s  ok ch=3.9e-14/1e-10 1s=2e-15 6.00x
   ducc0_c2c                    726.558 us 1627491.010 us     3.05       2.2%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 8.43x
   fftw3_guru                   849.112 us 1902010.130 us     2.61       0.5%    0.093s  ok ch=4.0e-14/1e-10 1s=2e-15 9.85x
   mkl_dfti                     850.061 us 1904135.960 us     2.60       0.2%    0.037s  ok ch=3.6e-14/1e-10 1s=2e-15 9.86x
   fftw3_estimate               873.522 us 1956689.370 us     2.53       0.3%    0.002s  ok ch=4.0e-14/1e-10 1s=2e-15 10.14x
   fftw3_patient                874.286 us 1958400.060 us     2.53       0.4%    0.241s  ok ch=4.0e-14/1e-10 1s=2e-15 10.15x
   fftw3_measure                874.980 us 1959955.220 us     2.53       0.3%    0.089s  ok ch=4.0e-14/1e-10 1s=2e-15 10.15x
   mkl2026_dfti                 884.311 us 1980857.680 us     2.50       0.0%    0.051s  ok ch=3.6e-14/1e-10 1s=2e-15 10.26x
   baseline_matrix             4931.723 us 11047059.900 us     0.45       0.1%    0.000s  ok ch=8.2e-14/1e-10 1s=3e-15 57.23x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      55.189 us 110377.931 us    44.53       0.8%    0.391s  ok ch=3.1e-14/1e-10 1s=1e-15 1.00x
   gen_pow2                      55.600 us 111199.499 us    44.20       0.8%    0.000s  ok ch=3.1e-14/1e-10 1s=1e-15 1.01x
   gen_planner                  109.558 us 219115.840 us    22.43       1.6%    0.004s  ok ch=4.1e-14/1e-10 1s=2e-15 1.99x
   gen_twiddle                  153.042 us 306084.271 us    16.06       1.0%    0.006s  ok ch=4.1e-14/1e-10 1s=2e-15 2.77x
   gen_layout                   155.728 us 311455.796 us    15.78       1.7%    0.001s  ok ch=3.8e-14/1e-10 1s=2e-15 2.82x
   mkl_dfti                     174.979 us 349958.481 us    14.05       2.0%    0.002s  ok ch=2.9e-14/1e-10 1s=1e-15 3.17x
   fftw3_custom_soa             176.191 us 352381.837 us    13.95       6.8%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 3.19x
   mkl2026_dfti                 187.684 us 375367.436 us    13.09       0.5%    0.003s  ok ch=2.9e-14/1e-10 1s=1e-15 3.40x
