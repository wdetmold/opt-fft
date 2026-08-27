# What previous generations produced (round gen_r12 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r10/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r11/leaderboard.txt
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
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_batchlane.md 1631 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_bluestein.md 1375 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_dense_prime.md 1793 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_layout.md 1758 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_large.md 1709 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_small.md 1549 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_planner.md 1572 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pow2.md 1596 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_powp.md 1893 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_race.md 1713 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_rader.md 1705 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_twiddle.md 1821 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gen_r11 ===
# round gen_r11
host: a80n0.lqcd.mit   date: 2026-08-26T12:13:40-04:00   slurm_job: 438881
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.121 us 71754.236 us    44.44       0.1%    0.000s  ok ch=1.7e-13/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.122 us 71838.362 us    44.39       0.1%    0.000s  ok ch=1.7e-13/1e-10 1s=9e-16 1.00x
   gen_race                       1.124 us 71911.756 us    44.35      13.9%   12.184s  ok ch=1.7e-13/1e-10 1s=9e-16 1.00x
   gen_planner                    1.323 us 84702.425 us    37.65       0.7%    0.002s  ok ch=1.6e-13/1e-10 1s=1e-15 1.18x
   gen_twiddle                    4.251 us 272046.626 us    11.72       2.3%    0.003s  ok ch=1.3e-13/1e-10 1s=9e-16 3.79x
   fftw3_custom_soa               4.451 us 284836.591 us    11.20       3.1%    0.000s  ok ch=1.4e-13/1e-10 1s=8e-16 3.97x
   mkl_dfti                       4.553 us 291408.051 us    10.94       2.5%    0.002s  ok ch=1.9e-13/1e-10 1s=1e-15 4.06x
   mkl2026_dfti                   4.669 us 298826.735 us    10.67       0.2%    0.001s  ok ch=1.6e-13/1e-10 1s=1e-15 4.16x
   gen_layout                     4.935 us 315834.854 us    10.10       2.1%    0.000s  ok ch=1.6e-13/1e-10 1s=9e-16 4.40x
   gen_dense_prime                5.025 us 321619.274 us     9.92       3.9%    0.000s  ok ch=1.2e-13/1e-10 1s=9e-16 4.48x
   fftw3_measure                  5.093 us 325975.815 us     9.78       2.5%    0.012s  ok ch=1.8e-13/1e-10 1s=8e-16 4.54x
   fftw3_patient                  5.170 us 330908.376 us     9.64       0.8%    0.022s  ok ch=1.5e-13/1e-10 1s=8e-16 4.61x
   fftw3_custom                   5.976 us 382476.437 us     8.34       3.6%    0.000s  ok ch=1.4e-13/1e-10 1s=8e-16 5.33x
   fftw3_guru                     6.361 us 407124.049 us     7.83       0.2%    0.011s  ok ch=1.5e-13/1e-10 1s=9e-16 5.67x
   fftw3_estimate                 7.357 us 470867.367 us     6.77       0.2%    0.001s  ok ch=1.8e-13/1e-10 1s=8e-16 6.56x
   ducc0_c2c                      9.660 us 618229.162 us     5.16       1.7%    0.000s  ok ch=1.4e-13/1e-10 1s=8e-16 8.62x
   gen_bluestein                 12.832 us 821249.324 us     3.88       0.8%    0.000s  ok ch=2.5e-13/1e-10 1s=1e-15 11.45x
   baseline_matrix               54.687 us 3499980.130 us     0.91       0.1%    0.000s  ok ch=3.6e-13/1e-10 1s=1e-15 48.78x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       1.915 us 73542.157 us    48.52      14.2%    0.182s  ok ch=6.8e-14/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.915 us 73551.517 us    48.51       0.3%    0.000s  ok ch=6.8e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.916 us 73565.244 us    48.50       0.2%    0.000s  ok ch=6.8e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.339 us 89828.912 us    39.72       0.2%    0.005s  ok ch=8.7e-14/1e-10 1s=9e-16 1.22x
   gen_dense_prime                7.659 us 294089.367 us    12.13       1.8%    0.000s  ok ch=6.6e-14/1e-10 1s=1e-15 4.00x
   mkl_dfti                       7.740 us 297216.519 us    12.01       2.1%    0.002s  ok ch=5.2e-14/1e-10 1s=9e-16 4.04x
   gen_twiddle                    7.771 us 298422.999 us    11.96       4.7%    0.003s  ok ch=6.0e-14/1e-10 1s=1e-15 4.06x
   mkl2026_dfti                   7.827 us 300542.720 us    11.87       1.4%    0.001s  ok ch=5.7e-14/1e-10 1s=9e-16 4.09x
   gen_layout                     8.154 us 313095.743 us    11.40       2.6%    0.000s  ok ch=6.4e-14/1e-10 1s=1e-15 4.26x
   fftw3_patient                  8.515 us 326981.582 us    10.91       5.2%    0.024s  ok ch=7.2e-14/1e-10 1s=9e-16 4.45x
   fftw3_custom_soa               8.559 us 328668.704 us    10.86       0.1%    0.000s  ok ch=5.2e-14/1e-10 1s=9e-16 4.47x
   fftw3_measure                  8.855 us 340025.465 us    10.49       0.8%    0.014s  ok ch=5.6e-14/1e-10 1s=9e-16 4.62x
   fftw3_guru                    10.170 us 390537.169 us     9.14       3.3%    0.011s  ok ch=5.7e-14/1e-10 1s=9e-16 5.31x
   fftw3_custom                  10.466 us 401900.326 us     8.88       1.8%    0.000s  ok ch=5.2e-14/1e-10 1s=9e-16 5.46x
   ducc0_c2c                     16.035 us 615760.040 us     5.79       2.1%    0.000s  ok ch=3.6e-14/1e-10 1s=7e-16 8.37x
   gen_bluestein                 18.954 us 727842.324 us     4.90       1.7%    0.000s  ok ch=1.0e-13/1e-10 1s=1e-15 9.90x
   fftw3_estimate                19.303 us 741252.149 us     4.81       0.0%    0.001s  ok ch=7.2e-14/1e-10 1s=9e-16 10.08x
   baseline_matrix              112.127 us 4305686.290 us     0.83       0.1%    0.000s  ok ch=2.9e-13/1e-10 1s=2e-15 58.55x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  4.340 us 83336.391 us    45.57       1.8%    0.000s  ok ch=5.8e-14/1e-10 1s=1e-15 1.00x
   gen_race                       4.346 us 83443.443 us    45.51      14.3%    0.016s  ok ch=5.8e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                  4.348 us 83490.266 us    45.48       0.4%    0.000s  ok ch=5.8e-14/1e-10 1s=1e-15 1.00x
   gen_planner                    5.622 us 107944.377 us    35.18       1.6%    0.011s  ok ch=5.7e-14/1e-10 1s=1e-15 1.30x
   gen_dense_prime               15.220 us 292233.389 us    12.99       3.9%    0.000s  ok ch=5.2e-14/1e-10 1s=1e-15 3.51x
   fftw3_custom_soa              15.414 us 295953.158 us    12.83       4.1%    0.000s  ok ch=5.1e-14/1e-10 1s=1e-15 3.55x
   mkl_dfti                      16.490 us 316605.586 us    11.99       1.5%    0.002s  ok ch=5.9e-14/1e-10 1s=1e-15 3.80x
   mkl2026_dfti                  16.664 us 319954.476 us    11.87       0.5%    0.003s  ok ch=6.8e-14/1e-10 1s=1e-15 3.84x
   gen_twiddle                   18.013 us 345858.688 us    10.98       3.0%    0.002s  ok ch=5.9e-14/1e-10 1s=1e-15 4.15x
   fftw3_patient                 19.593 us 376189.966 us    10.09       1.7%    0.019s  ok ch=5.1e-14/1e-10 1s=1e-15 4.51x
   fftw3_measure                 19.658 us 377425.811 us    10.06       3.3%    0.011s  ok ch=5.1e-14/1e-10 1s=1e-15 4.53x
   gen_layout                    19.727 us 378750.566 us    10.03       1.5%    0.000s  ok ch=5.3e-14/1e-10 1s=1e-15 4.54x
   fftw3_estimate                20.628 us 396049.294 us     9.59       0.8%    0.001s  ok ch=5.1e-14/1e-10 1s=1e-15 4.75x
   fftw3_custom                  25.179 us 483432.039 us     7.86       2.9%    0.000s  ok ch=5.1e-14/1e-10 1s=1e-15 5.80x
   fftw3_guru                    26.162 us 502312.404 us     7.56       1.7%    0.010s  ok ch=5.1e-14/1e-10 1s=1e-15 6.03x
   ducc0_c2c                     32.490 us 623801.527 us     6.09       1.9%    0.000s  ok ch=4.3e-14/1e-10 1s=1e-15 7.49x
   gen_bluestein                 32.974 us 633097.852 us     6.00       1.1%    0.000s  ok ch=8.6e-14/1e-10 1s=2e-15 7.60x
   baseline_matrix              272.445 us 5230944.120 us     0.73       0.0%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 62.77x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                 12.597 us 103190.895 us    41.17       5.8%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 1.00x
   gen_race                      12.694 us 103988.645 us    40.86       4.8%    0.261s  ok ch=3.1e-14/1e-10 1s=1e-15 1.01x
   gen_pfa_small                 13.275 us 108750.775 us    39.07       0.1%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 1.05x
   gen_planner                   18.134 us 148552.998 us    28.60       1.0%    0.007s  ok ch=3.3e-14/1e-10 1s=1e-15 1.44x
   gen_twiddle                   32.903 us 269540.926 us    15.76       3.3%    0.003s  ok ch=3.4e-14/1e-10 1s=1e-15 2.61x
   gen_layout                    34.483 us 282483.377 us    15.04       3.7%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 2.74x
   gen_dense_prime               38.123 us 312306.402 us    13.60       1.3%    0.000s  ok ch=3.5e-14/1e-10 1s=1e-15 3.03x
   fftw3_custom_soa              41.407 us 339204.402 us    12.53       2.0%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 3.29x
   fftw3_measure                 44.922 us 368004.720 us    11.55       1.8%    0.081s  ok ch=3.4e-14/1e-10 1s=1e-15 3.57x
   fftw3_patient                 45.145 us 369831.778 us    11.49       0.4%    0.316s  ok ch=3.6e-14/1e-10 1s=1e-15 3.58x
   mkl2026_dfti                  57.423 us 470411.385 us     9.03       4.1%    0.049s  ok ch=3.7e-14/1e-10 1s=1e-15 4.56x
   mkl_dfti                      57.878 us 474133.291 us     8.96       1.0%    0.030s  ok ch=4.0e-14/1e-10 1s=2e-15 4.59x
   fftw3_guru                    60.254 us 493601.599 us     8.61       2.5%    0.079s  ok ch=3.0e-14/1e-10 1s=1e-15 4.78x
   fftw3_custom                  63.810 us 522732.561 us     8.13       3.9%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 5.07x
   ducc0_c2c                     73.121 us 599005.365 us     7.09       1.4%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 5.80x
   gen_bluestein                 82.181 us 673225.219 us     6.31       0.2%    0.000s  ok ch=4.7e-14/1e-10 1s=2e-15 6.52x
   fftw3_estimate                91.881 us 752687.418 us     5.64       0.3%    0.001s  ok ch=3.4e-14/1e-10 1s=1e-15 7.29x
   baseline_matrix              850.387 us 6966367.640 us     0.61       0.0%    0.000s  ok ch=9.7e-14/1e-10 1s=2e-15 67.51x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      30.857 us 126388.707 us    35.27       2.2%    0.004s  ok ch=4.8e-14/1e-10 1s=2e-15 1.00x
   gen_race                      31.164 us 127648.176 us    34.92       1.8%    0.268s  ok ch=4.8e-14/1e-10 1s=2e-15 1.01x
   gen_planner                   39.642 us 162374.956 us    27.46       6.0%    0.013s  ok ch=4.5e-14/1e-10 1s=2e-15 1.28x
   gen_twiddle                   74.364 us 304594.324 us    14.64       3.5%    0.003s  ok ch=4.4e-14/1e-10 1s=2e-15 2.41x
   fftw3_custom_soa              75.145 us 307795.964 us    14.48       1.6%    0.000s  ok ch=4.2e-14/1e-10 1s=1e-15 2.44x
   gen_layout                    94.269 us 386124.696 us    11.55       0.7%    0.000s  ok ch=5.2e-14/1e-10 1s=2e-15 3.06x
   fftw3_measure                107.869 us 441830.313 us    10.09      10.7%    0.028s  ok ch=4.1e-14/1e-10 1s=2e-15 3.50x
   fftw3_patient                108.372 us 443891.669 us    10.04       3.3%    0.103s  ok ch=4.1e-14/1e-10 1s=2e-15 3.51x
   mkl_dfti                     120.773 us 494688.035 us     9.01       3.5%    0.050s  ok ch=4.4e-14/1e-10 1s=2e-15 3.91x
   mkl2026_dfti                 123.453 us 505665.366 us     8.82       0.3%    0.047s  ok ch=4.5e-14/1e-10 1s=2e-15 4.00x
   fftw3_estimate               132.391 us 542274.903 us     8.22       1.7%    0.001s  ok ch=4.1e-14/1e-10 1s=2e-15 4.29x
   ducc0_c2c                    145.184 us 594675.448 us     7.50       1.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 4.71x
   fftw3_custom                 160.280 us 656505.815 us     6.79       1.7%    0.000s  ok ch=4.2e-14/1e-10 1s=1e-15 5.19x
   gen_bluestein                169.373 us 693750.754 us     6.43       2.4%    0.000s  ok ch=5.2e-14/1e-10 1s=2e-15 5.49x
   fftw3_guru                   174.947 us 716581.888 us     6.22       6.9%    0.029s  ok ch=4.0e-14/1e-10 1s=2e-15 5.67x
   baseline_matrix             2061.151 us 8442474.470 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 66.80x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      43.484 us 139148.416 us    32.28       5.6%    0.005s  ok ch=5.5e-14/1e-10 1s=2e-15 1.00x
   gen_race                      43.545 us 139344.105 us    32.24       1.7%    0.308s  ok ch=5.5e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   60.129 us 192413.978 us    23.35       2.1%    0.020s  ok ch=3.8e-14/1e-10 1s=2e-15 1.38x
   fftw3_custom_soa              96.257 us 308021.479 us    14.58       8.3%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 2.21x
   gen_twiddle                  122.608 us 392346.296 us    11.45       0.5%    0.004s  ok ch=3.4e-14/1e-10 1s=2e-15 2.82x
   gen_layout                   124.665 us 398926.696 us    11.26       0.8%    0.001s  ok ch=3.6e-14/1e-10 1s=2e-15 2.87x
   mkl_dfti                     144.451 us 462243.773 us     9.72       0.1%    0.048s  ok ch=5.0e-14/1e-10 1s=2e-15 3.32x
   mkl2026_dfti                 147.939 us 473404.791 us     9.49       0.8%    0.048s  ok ch=5.0e-14/1e-10 1s=2e-15 3.40x
   ducc0_c2c                    190.013 us 608040.373 us     7.39       2.8%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 4.37x
   gen_bluestein                202.482 us 647943.260 us     6.93       1.2%    0.000s  ok ch=5.4e-14/1e-10 1s=2e-15 4.66x
   fftw3_custom                 203.479 us 651133.777 us     6.90       0.7%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 4.68x
   fftw3_patient                204.983 us 655945.363 us     6.85       1.8%    0.139s  ok ch=3.9e-14/1e-10 1s=2e-15 4.71x
   fftw3_measure                223.610 us 715553.007 us     6.28       2.7%    0.033s  ok ch=3.8e-14/1e-10 1s=2e-15 5.14x
   fftw3_estimate               253.912 us 812519.708 us     5.53       1.6%    0.003s  ok ch=4.4e-14/1e-10 1s=2e-15 5.84x
   fftw3_guru                   260.938 us 835002.690 us     5.38       4.1%    0.043s  ok ch=3.1e-14/1e-10 1s=2e-15 6.00x
   baseline_matrix             2799.377 us 8958006.520 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 64.38x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      84.549 us 189388.724 us    26.18       0.8%    0.725s  ok ch=3.4e-14/1e-10 1s=2e-15 1.00x
   gen_rader                     84.776 us 189898.324 us    26.11       1.0%    0.012s  ok ch=3.4e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              111.117 us 248902.176 us    19.92       0.6%    0.001s  ok ch=2.5e-14/1e-10 1s=2e-15 1.31x
   gen_planner                  139.154 us 311706.053 us    15.91       4.0%    0.024s  ok ch=2.6e-14/1e-10 1s=2e-15 1.65x
   gen_layout                   203.297 us 455385.771 us    10.89       0.9%    0.001s  ok ch=2.8e-14/1e-10 1s=2e-15 2.40x
   fftw3_custom_soa             213.361 us 477929.153 us    10.38       3.5%    0.000s  ok ch=3.3e-14/1e-10 1s=2e-15 2.52x
   gen_twiddle                  258.929 us 580001.767 us     8.55       1.6%    0.005s  ok ch=2.5e-14/1e-10 1s=2e-15 3.06x
   gen_bluestein                274.252 us 614324.520 us     8.07       1.1%    0.000s  ok ch=6.3e-14/1e-10 1s=3e-15 3.24x
   fftw3_custom                 499.155 us 1118107.180 us     4.44       1.1%    0.000s  ok ch=3.3e-14/1e-10 1s=2e-15 5.90x
   ducc0_c2c                    714.242 us 1599901.450 us     3.10       3.1%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 8.45x
   fftw3_guru                   830.118 us 1859463.390 us     2.67       1.3%    0.089s  ok ch=2.5e-14/1e-10 1s=2e-15 9.82x
   mkl_dfti                     848.750 us 1901199.900 us     2.61       0.0%    0.051s  ok ch=3.1e-14/1e-10 1s=2e-15 10.04x
   fftw3_patient                859.776 us 1925898.500 us     2.57       0.1%    0.241s  ok ch=2.5e-14/1e-10 1s=2e-15 10.17x
   fftw3_estimate               859.789 us 1925927.060 us     2.57       0.0%    0.002s  ok ch=2.5e-14/1e-10 1s=2e-15 10.17x
   fftw3_measure                860.190 us 1926825.760 us     2.57       0.3%    0.089s  ok ch=2.5e-14/1e-10 1s=2e-15 10.17x
   mkl2026_dfti                 882.853 us 1977591.130 us     2.51       0.1%    0.054s  ok ch=3.1e-14/1e-10 1s=2e-15 10.44x
   baseline_matrix             4852.279 us 10869104.200 us     0.46       0.0%    0.000s  ok ch=8.7e-14/1e-10 1s=3e-15 57.39x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      53.809 us 107617.703 us    45.67       1.8%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_race                      54.041 us 108082.556 us    45.48       3.4%    0.020s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_planner                  108.888 us 217775.614 us    22.57       0.9%    0.005s  ok ch=3.3e-14/1e-10 1s=1e-15 2.02x
   gen_layout                   150.060 us 300120.498 us    16.38       3.7%    0.001s  ok ch=3.3e-14/1e-10 1s=1e-15 2.79x
   gen_twiddle                  153.846 us 307691.593 us    15.97       2.7%    0.005s  ok ch=3.4e-14/1e-10 1s=1e-15 2.86x
   mkl_dfti                     170.663 us 341326.787 us    14.40       0.6%    0.003s  ok ch=2.7e-14/1e-10 1s=1e-15 3.17x
   mkl2026_dfti                 184.547 us 369093.174 us    13.32       2.0%    0.002s  ok ch=2.8e-14/1e-10 1s=1e-15 3.43x
   fftw3_custom_soa             186.525 us 373049.696 us    13.18       0.6%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 3.47x
   fftw3_patient                209.787 us 419574.439 us    11.71       0.5%    0.542s  ok ch=3.2e-14/1e-10 1s=1e-15 3.90x
   fftw3_measure                210.219 us 420438.034 us    11.69       0.5%    0.086s  ok ch=2.9e-14/1e-10 1s=1e-15 3.91x
   fftw3_guru                   292.804 us 585607.161 us     8.39      10.3%    0.071s  ok ch=3.0e-14/1e-10 1s=1e-15 5.44x
   gen_bluestein                299.705 us 599409.872 us     8.20       2.8%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 5.57x
   ducc0_c2c                    311.544 us 623088.179 us     7.89       2.1%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 5.79x
   fftw3_estimate               409.160 us 818319.975 us     6.01       0.4%    0.001s  ok ch=2.9e-14/1e-10 1s=1e-15 7.60x
   fftw3_custom                 412.540 us 825080.155 us     5.96       1.2%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 7.67x
   baseline_matrix             5751.504 us 11503008.700 us     0.43       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=2e-15 106.89x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                     159.534 us 163362.803 us    32.02       3.8%    0.752s  ok ch=2.1e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                160.033 us 163873.615 us    31.93       3.5%    1.309s  ok ch=2.1e-14/1e-10 1s=2e-15 1.00x
   gen_planner                  239.232 us 244973.949 us    21.36       1.3%    0.157s  ok ch=3.0e-14/1e-10 1s=2e-15 1.50x
   gen_twiddle                  289.075 us 296012.434 us    17.67       1.9%    0.006s  ok ch=2.6e-14/1e-10 1s=2e-15 1.81x
   gen_layout                   329.358 us 337262.404 us    15.51       2.1%    0.001s  ok ch=2.7e-14/1e-10 1s=2e-15 2.06x
   mkl_dfti                     404.658 us 414370.223 us    12.63       0.5%    0.002s  ok ch=3.0e-14/1e-10 1s=2e-15 2.54x
   mkl2026_dfti                 404.989 us 414708.507 us    12.62       0.5%    0.002s  ok ch=4.0e-14/1e-10 1s=2e-15 2.54x
   fftw3_custom_soa             405.058 us 414779.727 us    12.61       6.3%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 2.54x
   fftw3_patient                524.668 us 537259.670 us     9.74       5.0%    1.415s  ok ch=3.0e-14/1e-10 1s=2e-15 3.29x
   fftw3_measure                550.540 us 563753.040 us     9.28       3.2%    0.154s  ok ch=4.1e-14/1e-10 1s=2e-15 3.45x
   ducc0_c2c                    599.118 us 613496.363 us     8.53       2.0%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 3.76x
   gen_bluestein                603.859 us 618351.494 us     8.46       3.3%    0.000s  ok ch=3.9e-14/1e-10 1s=3e-15 3.79x
   fftw3_guru                   686.300 us 702771.356 us     7.44       0.9%    0.157s  ok ch=2.5e-14/1e-10 1s=2e-15 4.30x
   fftw3_custom                 780.318 us 799045.753 us     6.55       3.6%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 4.89x
   fftw3_estimate              1603.025 us 1641497.320 us     3.19       0.7%    0.002s  ok ch=2.3e-14/1e-10 1s=2e-15 10.05x
   baseline_matrix            13460.603 us 13783657.200 us     0.38       0.1%    0.000s  ok ch=6.5e-14/1e-10 1s=2e-15 84.37x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                     410.975 us 210419.433 us    25.75       5.4%    0.022s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_powp                     413.898 us 211915.611 us    25.57       5.5%    1.435s  ok ch=3.3e-14/1e-10 1s=2e-15 1.01x
   gen_pfa_large                417.960 us 213995.453 us    25.32       4.9%    2.097s  ok ch=3.3e-14/1e-10 1s=2e-15 1.02x
   gen_batchlane                475.524 us 243468.214 us    22.25       0.7%    0.001s  ok ch=3.1e-14/1e-10 1s=2e-15 1.16x
   gen_planner                  547.776 us 280461.265 us    19.32       1.4%    0.011s  ok ch=3.3e-14/1e-10 1s=2e-15 1.33x
   gen_twiddle                  629.904 us 322510.650 us    16.80       3.0%    0.012s  ok ch=3.3e-14/1e-10 1s=2e-15 1.53x
   gen_pfa_small                726.599 us 372018.465 us    14.56       1.4%    0.009s  ok ch=3.3e-14/1e-10 1s=3e-15 1.77x
   gen_layout                   932.821 us 477604.607 us    11.34       2.2%    0.001s  ok ch=3.2e-14/1e-10 1s=2e-15 2.27x
   mkl_dfti                     946.334 us 484523.099 us    11.18       0.4%    0.032s  ok ch=3.6e-14/1e-10 1s=3e-15 2.30x
   mkl2026_dfti                 961.820 us 492451.663 us    11.00       0.3%    0.050s  ok ch=3.2e-14/1e-10 1s=2e-15 2.34x
