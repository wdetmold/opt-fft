# What previous generations produced (round gen_r13 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r10/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r11/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r12/leaderboard.txt
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
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_batchlane.md 1848 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_bluestein.md 1375 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_dense_prime.md 1946 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_layout.md 1919 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_large.md 1893 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_small.md 1749 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_planner.md 1765 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pow2.md 1748 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_powp.md 2080 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_race.md 1931 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_rader.md 1880 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_twiddle.md 1990 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gen_r12 ===
# round gen_r12
host: a80n0.lqcd.mit   date: 2026-08-26T21:19:44-04:00   slurm_job: 438947
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.116 us 71435.711 us    44.64       0.1%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.121 us 71738.521 us    44.45       0.1%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 1.00x
   gen_planner                    1.321 us 84514.954 us    37.73       0.5%    0.003s  ok ch=1.9e-13/1e-10 1s=1e-15 1.18x
   gen_race                       1.329 us 85051.742 us    37.50       0.4%    0.007s  ok ch=1.9e-13/1e-10 1s=1e-15 1.19x
   gen_twiddle                    4.261 us 272689.779 us    11.69       0.7%    0.003s  ok ch=1.3e-13/1e-10 1s=9e-16 3.82x
   fftw3_custom_soa               4.545 us 290872.284 us    10.96       1.4%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 4.07x
   mkl_dfti                       4.566 us 292235.689 us    10.91       0.2%    0.001s  ok ch=1.7e-13/1e-10 1s=1e-15 4.09x
   mkl2026_dfti                   4.645 us 297302.730 us    10.73       2.0%    0.002s  ok ch=1.4e-13/1e-10 1s=1e-15 4.16x
   gen_layout                     4.929 us 315433.601 us    10.11       2.2%    0.000s  ok ch=1.4e-13/1e-10 1s=9e-16 4.42x
   gen_dense_prime                5.079 us 325050.702 us     9.81       6.5%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 4.55x
   fftw3_measure                  5.109 us 326963.276 us     9.75       5.0%    0.013s  ok ch=1.3e-13/1e-10 1s=9e-16 4.58x
   fftw3_patient                  5.185 us 331854.552 us     9.61       0.0%    0.021s  ok ch=1.8e-13/1e-10 1s=8e-16 4.65x
   fftw3_custom                   5.969 us 381995.573 us     8.35       3.7%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 5.35x
   fftw3_guru                     6.369 us 407624.314 us     7.82       2.7%    0.011s  ok ch=1.5e-13/1e-10 1s=8e-16 5.71x
   fftw3_estimate                 7.354 us 470667.775 us     6.78       0.1%    0.001s  ok ch=1.2e-13/1e-10 1s=9e-16 6.59x
   ducc0_c2c                      9.605 us 614689.153 us     5.19       0.5%    0.000s  ok ch=9.4e-14/1e-10 1s=8e-16 8.60x
   gen_bluestein                 12.646 us 809372.583 us     3.94       1.6%    0.000s  ok ch=1.8e-13/1e-10 1s=1e-15 11.33x
   baseline_matrix               54.687 us 3499983.840 us     0.91       0.1%    0.000s  ok ch=3.3e-13/1e-10 1s=1e-15 48.99x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.912 us 73428.982 us    48.59       0.1%    0.000s  ok ch=4.4e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.915 us 73549.814 us    48.51       0.1%    0.000s  ok ch=4.4e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.353 us 90337.736 us    39.50       2.0%    0.003s  ok ch=6.0e-14/1e-10 1s=9e-16 1.23x
   gen_race                       2.353 us 90369.532 us    39.48       0.7%    0.006s  ok ch=6.0e-14/1e-10 1s=9e-16 1.23x
   gen_dense_prime                7.673 us 294653.112 us    12.11       8.1%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 4.01x
   mkl_dfti                       7.737 us 297103.053 us    12.01       0.3%    0.002s  ok ch=4.6e-14/1e-10 1s=9e-16 4.05x
   mkl2026_dfti                   7.776 us 298594.234 us    11.95       2.1%    0.002s  ok ch=4.7e-14/1e-10 1s=9e-16 4.07x
   gen_twiddle                    7.902 us 303423.731 us    11.76       1.2%    0.003s  ok ch=7.1e-14/1e-10 1s=1e-15 4.13x
   fftw3_custom_soa               7.960 us 305663.820 us    11.67       7.1%    0.000s  ok ch=4.6e-14/1e-10 1s=9e-16 4.16x
   gen_layout                     8.174 us 313871.382 us    11.37       4.1%    0.000s  ok ch=5.1e-14/1e-10 1s=9e-16 4.27x
   fftw3_patient                  8.553 us 328433.365 us    10.86       3.8%    0.023s  ok ch=4.7e-14/1e-10 1s=9e-16 4.47x
   fftw3_measure                  8.761 us 336425.713 us    10.61       2.7%    0.013s  ok ch=5.1e-14/1e-10 1s=9e-16 4.58x
   fftw3_custom                  10.401 us 399382.052 us     8.93       5.7%    0.000s  ok ch=4.6e-14/1e-10 1s=9e-16 5.44x
   fftw3_guru                    10.503 us 403323.780 us     8.85       0.4%    0.010s  ok ch=4.7e-14/1e-10 1s=9e-16 5.49x
   ducc0_c2c                     15.977 us 613501.124 us     5.82       2.3%    0.000s  ok ch=3.9e-14/1e-10 1s=7e-16 8.36x
   gen_bluestein                 18.751 us 720055.168 us     4.96       2.1%    0.000s  ok ch=9.2e-14/1e-10 1s=1e-15 9.81x
   fftw3_estimate                19.282 us 740432.446 us     4.82       0.3%    0.001s  ok ch=4.7e-14/1e-10 1s=9e-16 10.08x
   baseline_matrix              112.125 us 4305594.680 us     0.83       0.0%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 58.64x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  4.324 us 83027.737 us    45.74      14.3%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.333 us 83191.294 us    45.65       0.2%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 1.00x
   gen_race                       5.241 us 100620.274 us    37.74       7.4%    0.008s  ok ch=5.1e-14/1e-10 1s=1e-15 1.21x
   gen_planner                    5.471 us 105042.509 us    36.15       2.5%    0.003s  ok ch=5.1e-14/1e-10 1s=1e-15 1.27x
   gen_dense_prime               14.760 us 283383.582 us    13.40       4.1%    0.000s  ok ch=5.5e-14/1e-10 1s=1e-15 3.41x
   fftw3_custom_soa              15.628 us 300054.614 us    12.66       7.2%    0.000s  ok ch=6.0e-14/1e-10 1s=1e-15 3.61x
   mkl_dfti                      16.450 us 315839.652 us    12.02       1.7%    0.002s  ok ch=9.2e-14/1e-10 1s=1e-15 3.80x
   mkl2026_dfti                  16.901 us 324497.822 us    11.70       0.0%    0.001s  ok ch=9.1e-14/1e-10 1s=1e-15 3.91x
   gen_twiddle                   17.823 us 342192.305 us    11.10       1.4%    0.004s  ok ch=5.1e-14/1e-10 1s=1e-15 4.12x
   gen_layout                    19.601 us 376332.140 us    10.09       3.6%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 4.53x
   fftw3_patient                 19.637 us 377035.697 us    10.07       1.7%    0.020s  ok ch=7.3e-14/1e-10 1s=1e-15 4.54x
   fftw3_measure                 19.788 us 379934.833 us    10.00       5.6%    0.011s  ok ch=7.3e-14/1e-10 1s=1e-15 4.58x
   fftw3_estimate                20.635 us 396200.957 us     9.58       2.3%    0.001s  ok ch=7.3e-14/1e-10 1s=1e-15 4.77x
   fftw3_custom                  25.060 us 481161.495 us     7.89       4.9%    0.000s  ok ch=6.0e-14/1e-10 1s=1e-15 5.80x
   fftw3_guru                    26.162 us 502315.900 us     7.56       2.0%    0.010s  ok ch=8.9e-14/1e-10 1s=1e-15 6.05x
   ducc0_c2c                     32.435 us 622750.267 us     6.10       0.4%    0.000s  ok ch=5.2e-14/1e-10 1s=1e-15 7.50x
   gen_bluestein                 33.313 us 639605.027 us     5.94       0.5%    0.000s  ok ch=1.1e-13/1e-10 1s=2e-15 7.70x
   baseline_matrix              272.409 us 5230261.690 us     0.73       0.0%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 62.99x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                 12.552 us 102829.567 us    41.32       5.9%    0.001s  ok ch=2.8e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                 12.729 us 104278.002 us    40.74       4.2%    0.001s  ok ch=2.8e-14/1e-10 1s=1e-15 1.01x
   gen_planner                   17.021 us 139432.541 us    30.47       2.3%    0.006s  ok ch=2.8e-14/1e-10 1s=1e-15 1.36x
   gen_race                      17.662 us 144684.980 us    29.36       0.6%    0.004s  ok ch=2.7e-14/1e-10 1s=1e-15 1.41x
   gen_twiddle                   26.181 us 214470.811 us    19.81       1.6%    0.003s  ok ch=2.8e-14/1e-10 1s=1e-15 2.09x
   gen_layout                    34.337 us 281285.421 us    15.10       2.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 2.74x
   gen_dense_prime               38.428 us 314804.523 us    13.50       1.9%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 3.06x
   fftw3_custom_soa              41.485 us 339845.142 us    12.50       3.0%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 3.30x
   fftw3_patient                 44.870 us 367575.766 us    11.56       3.0%    0.296s  ok ch=3.0e-14/1e-10 1s=1e-15 3.57x
   fftw3_measure                 45.248 us 370673.374 us    11.46       1.0%    0.081s  ok ch=3.0e-14/1e-10 1s=1e-15 3.60x
   mkl2026_dfti                  57.501 us 471047.915 us     9.02       3.1%    0.054s  ok ch=3.2e-14/1e-10 1s=1e-15 4.58x
   mkl_dfti                      58.063 us 475653.839 us     8.93       0.7%    0.050s  ok ch=3.3e-14/1e-10 1s=1e-15 4.63x
   fftw3_guru                    60.129 us 492573.480 us     8.63       3.7%    0.079s  ok ch=2.8e-14/1e-10 1s=1e-15 4.79x
   fftw3_custom                  63.281 us 518401.547 us     8.20       0.9%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 5.04x
   ducc0_c2c                     72.544 us 594284.458 us     7.15       1.5%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 5.78x
   gen_bluestein                 80.712 us 661191.109 us     6.43       1.8%    0.000s  ok ch=4.5e-14/1e-10 1s=2e-15 6.43x
   fftw3_estimate                91.890 us 752766.130 us     5.64       0.6%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 7.32x
   baseline_matrix              850.293 us 6965599.790 us     0.61       0.0%    0.000s  ok ch=9.8e-14/1e-10 1s=2e-15 67.74x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      31.698 us 129834.632 us    34.34       3.9%    0.003s  ok ch=3.7e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   39.318 us 161045.765 us    27.68      11.6%    0.015s  ok ch=3.4e-14/1e-10 1s=2e-15 1.24x
   gen_race                      40.961 us 167777.313 us    26.57       8.1%    0.005s  ok ch=3.4e-14/1e-10 1s=2e-15 1.29x
   fftw3_custom_soa              75.339 us 308586.641 us    14.45       2.5%    0.000s  ok ch=3.5e-14/1e-10 1s=1e-15 2.38x
   gen_twiddle                   76.507 us 313371.589 us    14.23       0.0%    0.004s  ok ch=3.9e-14/1e-10 1s=2e-15 2.41x
   gen_layout                    94.779 us 388214.258 us    11.48       0.9%    0.000s  ok ch=3.7e-14/1e-10 1s=2e-15 2.99x
   fftw3_patient                107.777 us 441453.010 us    10.10       0.7%    0.112s  ok ch=3.7e-14/1e-10 1s=2e-15 3.40x
   fftw3_measure                111.705 us 457542.807 us     9.74       1.4%    0.029s  ok ch=3.7e-14/1e-10 1s=2e-15 3.52x
   mkl_dfti                     120.222 us 492427.358 us     9.05       0.6%    0.051s  ok ch=3.8e-14/1e-10 1s=2e-15 3.79x
   mkl2026_dfti                 123.408 us 505480.835 us     8.82       0.6%    0.047s  ok ch=3.8e-14/1e-10 1s=2e-15 3.89x
   fftw3_estimate               129.805 us 531683.142 us     8.38       4.6%    0.001s  ok ch=3.7e-14/1e-10 1s=2e-15 4.10x
   ducc0_c2c                    146.213 us 598890.187 us     7.44       1.3%    0.000s  ok ch=3.1e-14/1e-10 1s=1e-15 4.61x
   fftw3_custom                 159.388 us 652852.422 us     6.83       0.8%    0.000s  ok ch=3.5e-14/1e-10 1s=1e-15 5.03x
   gen_bluestein                169.158 us 692873.056 us     6.43       1.8%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 5.34x
   fftw3_guru                   177.492 us 727008.569 us     6.13       6.8%    0.035s  ok ch=4.1e-14/1e-10 1s=2e-15 5.60x
   baseline_matrix             2061.289 us 8443038.810 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 65.03x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      44.031 us 140899.719 us    31.88       5.5%    0.004s  ok ch=3.2e-14/1e-10 1s=2e-15 1.00x
   gen_race                      60.041 us 192130.503 us    23.38       0.4%    0.008s  ok ch=3.4e-14/1e-10 1s=2e-15 1.36x
   gen_planner                   63.661 us 203713.665 us    22.05       0.3%    0.015s  ok ch=3.4e-14/1e-10 1s=2e-15 1.45x
   fftw3_custom_soa              96.416 us 308531.706 us    14.56       7.6%    0.000s  ok ch=3.4e-14/1e-10 1s=2e-15 2.19x
   gen_twiddle                  120.435 us 385390.731 us    11.66       0.8%    0.004s  ok ch=3.5e-14/1e-10 1s=2e-15 2.74x
   gen_layout                   123.287 us 394516.919 us    11.39       2.2%    0.000s  ok ch=3.3e-14/1e-10 1s=2e-15 2.80x
   mkl_dfti                     144.132 us 461222.184 us     9.74       0.4%    0.036s  ok ch=3.5e-14/1e-10 1s=2e-15 3.27x
   mkl2026_dfti                 147.975 us 473519.650 us     9.49       0.1%    0.051s  ok ch=3.5e-14/1e-10 1s=2e-15 3.36x
   ducc0_c2c                    188.960 us 604672.469 us     7.43       3.2%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 4.29x
   fftw3_custom                 201.033 us 643304.747 us     6.98       3.1%    0.000s  ok ch=3.4e-14/1e-10 1s=2e-15 4.57x
   gen_bluestein                202.528 us 648090.093 us     6.93       1.5%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 4.60x
   fftw3_patient                203.268 us 650456.371 us     6.91       1.3%    0.137s  ok ch=3.4e-14/1e-10 1s=2e-15 4.62x
   fftw3_measure                227.173 us 726954.035 us     6.18       0.8%    0.034s  ok ch=3.6e-14/1e-10 1s=2e-15 5.16x
   fftw3_estimate               255.431 us 817377.876 us     5.50       1.6%    0.002s  ok ch=3.2e-14/1e-10 1s=2e-15 5.80x
   fftw3_guru                   260.413 us 833321.383 us     5.39       4.8%    0.043s  ok ch=3.3e-14/1e-10 1s=2e-15 5.91x
   baseline_matrix             2799.025 us 8956878.950 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 63.57x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     84.753 us 189846.565 us    26.12       0.9%    0.009s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              111.333 us 249385.975 us    19.88       0.6%    0.001s  ok ch=2.6e-14/1e-10 1s=2e-15 1.31x
   gen_race                     138.751 us 310801.524 us    15.96       0.8%    0.011s  ok ch=2.5e-14/1e-10 1s=2e-15 1.64x
   gen_planner                  140.804 us 315401.837 us    15.72       2.9%    0.030s  ok ch=2.5e-14/1e-10 1s=2e-15 1.66x
   gen_layout                   203.238 us 455253.673 us    10.89       0.5%    0.000s  ok ch=2.6e-14/1e-10 1s=2e-15 2.40x
   fftw3_custom_soa             207.161 us 464041.113 us    10.69       7.3%    0.000s  ok ch=3.2e-14/1e-10 1s=2e-15 2.44x
   gen_twiddle                  259.049 us 580269.096 us     8.55       1.3%    0.005s  ok ch=2.5e-14/1e-10 1s=2e-15 3.06x
   gen_bluestein                273.991 us 613740.536 us     8.08       1.1%    0.000s  ok ch=4.2e-14/1e-10 1s=3e-15 3.23x
   fftw3_custom                 504.333 us 1129706.790 us     4.39       2.3%    0.000s  ok ch=3.2e-14/1e-10 1s=2e-15 5.95x
   ducc0_c2c                    715.666 us 1603091.990 us     3.09       2.6%    0.000s  ok ch=2.2e-14/1e-10 1s=1e-15 8.44x
   fftw3_guru                   832.993 us 1865904.120 us     2.66       0.4%    0.093s  ok ch=2.6e-14/1e-10 1s=2e-15 9.83x
   mkl_dfti                     848.517 us 1900678.270 us     2.61       0.0%    0.051s  ok ch=3.2e-14/1e-10 1s=2e-15 10.01x
   fftw3_estimate               859.504 us 1925289.770 us     2.58       0.3%    0.002s  ok ch=2.6e-14/1e-10 1s=2e-15 10.14x
   fftw3_measure                859.508 us 1925298.400 us     2.58       2.2%    0.089s  ok ch=2.6e-14/1e-10 1s=2e-15 10.14x
   fftw3_patient                859.553 us 1925397.840 us     2.58       0.9%    0.226s  ok ch=2.6e-14/1e-10 1s=2e-15 10.14x
   mkl2026_dfti                 883.009 us 1977939.180 us     2.51       0.3%    0.049s  ok ch=3.2e-14/1e-10 1s=2e-15 10.42x
   baseline_matrix             4853.226 us 10871226.000 us     0.46       0.0%    0.000s  ok ch=7.7e-14/1e-10 1s=3e-15 57.26x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      54.851 us 109702.047 us    44.80       0.8%    0.000s  ok ch=3.1e-14/1e-10 1s=1e-15 1.00x
   gen_planner                  108.288 us 216575.217 us    22.70       1.1%    0.003s  ok ch=5.3e-14/1e-10 1s=2e-15 1.97x
   gen_race                     108.994 us 217988.694 us    22.55       0.3%    0.006s  ok ch=5.3e-14/1e-10 1s=2e-15 1.99x
   gen_layout                   150.228 us 300455.195 us    16.36       2.4%    0.001s  ok ch=4.0e-14/1e-10 1s=1e-15 2.74x
   gen_twiddle                  154.706 us 309411.484 us    15.89       0.8%    0.004s  ok ch=4.5e-14/1e-10 1s=1e-15 2.82x
   mkl_dfti                     171.758 us 343515.836 us    14.31       0.9%    0.001s  ok ch=3.9e-14/1e-10 1s=1e-15 3.13x
   fftw3_custom_soa             178.905 us 357810.391 us    13.74       1.3%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 3.26x
   mkl2026_dfti                 187.508 us 375016.649 us    13.11       2.8%    0.001s  ok ch=3.5e-14/1e-10 1s=1e-15 3.42x
   fftw3_patient                208.383 us 416766.636 us    11.79       2.0%    0.493s  ok ch=3.0e-14/1e-10 1s=1e-15 3.80x
   fftw3_measure                210.044 us 420087.838 us    11.70       1.5%    0.086s  ok ch=5.1e-14/1e-10 1s=1e-15 3.83x
   fftw3_guru                   278.819 us 557637.437 us     8.81       3.3%    0.073s  ok ch=3.3e-14/1e-10 1s=1e-15 5.08x
   gen_bluestein                297.947 us 595894.620 us     8.25       0.4%    0.000s  ok ch=7.0e-14/1e-10 1s=2e-15 5.43x
   ducc0_c2c                    310.199 us 620398.433 us     7.92       1.8%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 5.66x
   fftw3_estimate               407.050 us 814099.701 us     6.04       0.8%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 7.42x
   fftw3_custom                 413.538 us 827076.540 us     5.94       1.2%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 7.54x
   baseline_matrix             5752.899 us 11505797.700 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=3e-15 104.88x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                159.253 us 163074.765 us    32.08       1.6%    0.006s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_race                     236.639 us 242318.403 us    21.59       0.9%    0.008s  ok ch=2.6e-14/1e-10 1s=2e-15 1.49x
   gen_planner                  239.380 us 245124.986 us    21.34       1.6%    0.005s  ok ch=2.8e-14/1e-10 1s=2e-15 1.50x
   gen_twiddle                  286.269 us 293139.556 us    17.85       1.0%    0.006s  ok ch=2.6e-14/1e-10 1s=2e-15 1.80x
   gen_layout                   348.766 us 357135.920 us    14.65       1.8%    0.000s  ok ch=2.6e-14/1e-10 1s=2e-15 2.19x
   mkl2026_dfti                 403.913 us 413606.582 us    12.65       0.6%    0.003s  ok ch=4.8e-14/1e-10 1s=2e-15 2.54x
   mkl_dfti                     405.766 us 415504.241 us    12.59       0.1%    0.003s  ok ch=2.7e-14/1e-10 1s=2e-15 2.55x
   fftw3_custom_soa             417.128 us 427138.798 us    12.25       1.4%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 2.62x
   fftw3_patient                532.367 us 545143.596 us     9.60       2.3%    1.443s  ok ch=2.3e-14/1e-10 1s=2e-15 3.34x
   fftw3_measure                548.999 us 562175.165 us     9.31       2.7%    0.148s  ok ch=2.7e-14/1e-10 1s=2e-15 3.45x
   ducc0_c2c                    597.620 us 611962.414 us     8.55       0.5%    0.000s  ok ch=2.0e-14/1e-10 1s=2e-15 3.75x
   gen_bluestein                606.535 us 621092.096 us     8.42       2.6%    0.000s  ok ch=4.0e-14/1e-10 1s=3e-15 3.81x
   fftw3_guru                   679.329 us 695633.182 us     7.52       7.3%    0.158s  ok ch=2.5e-14/1e-10 1s=2e-15 4.27x
   fftw3_custom                 784.970 us 803809.439 us     6.51       0.9%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 4.93x
   fftw3_estimate              1596.170 us 1634478.550 us     3.20       1.2%    0.002s  ok ch=2.7e-14/1e-10 1s=2e-15 10.02x
   baseline_matrix            13466.071 us 13789256.300 us     0.38       0.1%    0.000s  ok ch=5.9e-14/1e-10 1s=3e-15 84.56x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                413.440 us 211681.042 us    25.60       5.8%    0.002s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_powp                     480.616 us 246075.401 us    22.02       2.0%    0.002s  ok ch=2.8e-14/1e-10 1s=2e-15 1.16x
   gen_batchlane                481.273 us 246411.580 us    21.99       5.3%    0.001s  ok ch=6.7e-14/1e-10 1s=2e-15 1.16x
   gen_race                     545.065 us 279073.510 us    19.41       1.0%    0.007s  ok ch=3.2e-14/1e-10 1s=2e-15 1.32x
   gen_planner                  561.248 us 287359.068 us    18.85       2.4%    0.013s  ok ch=3.2e-14/1e-10 1s=2e-15 1.36x
   gen_pfa_small                568.531 us 291087.874 us    18.61       0.2%    0.009s  ok ch=7.9e-14/1e-10 1s=2e-15 1.38x
   gen_twiddle                  629.258 us 322180.100 us    16.82       1.3%    0.012s  ok ch=4.7e-14/1e-10 1s=2e-15 1.52x
   gen_layout                   945.191 us 483937.864 us    11.20       1.9%    0.001s  ok ch=3.8e-14/1e-10 1s=2e-15 2.29x
   mkl_dfti                     947.464 us 485101.361 us    11.17       0.3%    0.048s  ok ch=3.4e-14/1e-10 1s=2e-15 2.29x
   mkl2026_dfti                 962.789 us 492947.968 us    10.99       0.4%    0.049s  ok ch=3.5e-14/1e-10 1s=2e-15 2.33x
