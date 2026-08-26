# What previous generations produced (round gen_r9 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r4/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r5/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r6/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r7/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r8/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/xarch_clx_r6/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gen/results/xarch_spr_r5/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_batchlane.md 1083 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_bluestein.md 1016 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_dense_prime.md 1286 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_layout.md 1248 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_large.md 1232 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_small.md 1109 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_planner.md 1218 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pow2.md 1151 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_powp.md 1343 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_race.md 1324 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_rader.md 1186 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_twiddle.md 1295 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gen_r8 ===
# round gen_r8
host: a80n0.lqcd.mit   date: 2026-08-25T15:40:06-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       1.146 us 73361.771 us    43.47      13.8%    0.009s  ok ch=1.1e-13/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.148 us 73469.526 us    43.41       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.152 us 73703.337 us    43.27       0.0%    0.000s  ok ch=8.8e-14/1e-10 1s=8e-16 1.00x
   gen_planner                    1.324 us 84766.195 us    37.62       0.6%    0.004s  ok ch=1.0e-13/1e-10 1s=1e-15 1.16x
   mkl_dfti                       4.577 us 292939.600 us    10.89       0.4%    0.002s  ok ch=1.4e-13/1e-10 1s=1e-15 3.99x
   mkl2026_dfti                   4.651 us 297694.904 us    10.71       0.1%    0.002s  ok ch=1.3e-13/1e-10 1s=1e-15 4.06x
   gen_layout                     4.888 us 312822.333 us    10.19       3.3%    0.000s  ok ch=1.1e-13/1e-10 1s=9e-16 4.26x
   fftw3_measure                  4.976 us 318446.921 us    10.01       7.9%    0.013s  ok ch=1.1e-13/1e-10 1s=8e-16 4.34x
   gen_dense_prime                5.063 us 324031.596 us     9.84       9.2%    0.000s  ok ch=1.6e-13/1e-10 1s=9e-16 4.42x
   fftw3_patient                  5.340 us 341773.444 us     9.33       0.1%    0.022s  ok ch=1.1e-13/1e-10 1s=8e-16 4.66x
   fftw3_guru                     6.368 us 407556.456 us     7.82       0.1%    0.010s  ok ch=8.7e-14/1e-10 1s=8e-16 5.56x
   gen_twiddle                    6.460 us 413443.896 us     7.71       2.3%    0.002s  ok ch=1.9e-13/1e-10 1s=9e-16 5.64x
   fftw3_estimate                 7.317 us 468286.374 us     6.81       0.2%    0.001s  ok ch=1.1e-13/1e-10 1s=8e-16 6.38x
   ducc0_c2c                      9.772 us 625388.093 us     5.10       2.7%    0.000s  ok ch=8.7e-14/1e-10 1s=7e-16 8.52x
   gen_bluestein                 13.362 us 855144.060 us     3.73       0.9%    0.000s  ok ch=2.0e-13/1e-10 1s=1e-15 11.66x
   baseline_matrix               54.687 us 3499984.820 us     0.91       0.0%    0.000s  ok ch=3.6e-13/1e-10 1s=1e-15 47.71x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.911 us 73381.758 us    48.63       0.1%    0.000s  ok ch=4.9e-14/1e-10 1s=9e-16 1.00x
   gen_race                       1.912 us 73427.724 us    48.59       0.2%    0.015s  ok ch=4.9e-14/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.914 us 73507.301 us    48.54       0.2%    0.000s  ok ch=4.9e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.334 us 89615.556 us    39.82       0.9%    0.006s  ok ch=5.3e-14/1e-10 1s=9e-16 1.22x
   gen_dense_prime                7.715 us 296262.806 us    12.04       2.6%    0.000s  ok ch=5.1e-14/1e-10 1s=9e-16 4.04x
   mkl_dfti                       7.770 us 298381.070 us    11.96       0.4%    0.003s  ok ch=5.0e-14/1e-10 1s=9e-16 4.07x
   mkl2026_dfti                   7.792 us 299196.587 us    11.93       0.5%    0.002s  ok ch=4.9e-14/1e-10 1s=9e-16 4.08x
   gen_layout                     8.092 us 310732.869 us    11.48       3.1%    0.000s  ok ch=5.0e-14/1e-10 1s=9e-16 4.23x
   fftw3_patient                  8.643 us 331886.620 us    10.75       2.9%    0.023s  ok ch=4.4e-14/1e-10 1s=9e-16 4.52x
   fftw3_measure                  8.757 us 336276.626 us    10.61       5.9%    0.014s  ok ch=5.0e-14/1e-10 1s=9e-16 4.58x
   gen_twiddle                    9.125 us 350391.765 us    10.18       3.6%    0.003s  ok ch=5.2e-14/1e-10 1s=1e-15 4.77x
   fftw3_guru                    10.182 us 390992.049 us     9.13       0.4%    0.011s  ok ch=4.9e-14/1e-10 1s=9e-16 5.33x
   ducc0_c2c                     16.037 us 615836.525 us     5.79       3.5%    0.000s  ok ch=3.7e-14/1e-10 1s=7e-16 8.39x
   fftw3_estimate                19.314 us 741669.958 us     4.81       0.2%    0.001s  ok ch=4.4e-14/1e-10 1s=9e-16 10.11x
   gen_bluestein                 19.449 us 746852.925 us     4.78       1.5%    0.000s  ok ch=9.4e-14/1e-10 1s=1e-15 10.18x
   baseline_matrix              112.125 us 4305590.700 us     0.83       0.0%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 58.67x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       4.374 us 83987.384 us    45.22       0.4%    0.188s  ok ch=5.7e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.376 us 84012.274 us    45.20       0.1%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                  4.416 us 84779.495 us    44.79       0.2%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 1.01x
   gen_planner                    5.542 us 106405.981 us    35.69       3.4%    0.005s  ok ch=5.4e-14/1e-10 1s=1e-15 1.27x
   gen_dense_prime               15.218 us 292192.167 us    13.00       2.9%    0.000s  ok ch=5.3e-14/1e-10 1s=1e-15 3.48x
   mkl_dfti                      16.502 us 316834.955 us    11.99       0.2%    0.002s  ok ch=6.1e-14/1e-10 1s=1e-15 3.77x
   mkl2026_dfti                  16.674 us 320141.412 us    11.86       0.4%    0.002s  ok ch=6.1e-14/1e-10 1s=1e-15 3.81x
   gen_layout                    19.445 us 373336.938 us    10.17       1.8%    0.000s  ok ch=5.9e-14/1e-10 1s=1e-15 4.45x
   fftw3_measure                 19.689 us 378032.659 us    10.05       0.3%    0.010s  ok ch=5.5e-14/1e-10 1s=1e-15 4.50x
   gen_twiddle                   19.759 us 379382.046 us    10.01       1.9%    0.003s  ok ch=5.9e-14/1e-10 1s=1e-15 4.52x
   fftw3_patient                 19.859 us 381286.803 us     9.96       0.3%    0.019s  ok ch=5.5e-14/1e-10 1s=1e-15 4.54x
   fftw3_estimate                20.661 us 396698.438 us     9.57       1.6%    0.001s  ok ch=5.5e-14/1e-10 1s=1e-15 4.72x
   fftw3_guru                    26.193 us 502905.947 us     7.55       3.1%    0.010s  ok ch=5.5e-14/1e-10 1s=1e-15 5.99x
   ducc0_c2c                     32.459 us 623207.105 us     6.09       1.1%    0.000s  ok ch=4.4e-14/1e-10 1s=1e-15 7.42x
   gen_bluestein                 34.247 us 657545.968 us     5.78       0.3%    0.000s  ok ch=9.3e-14/1e-10 1s=2e-15 7.83x
   baseline_matrix              272.448 us 5231002.190 us     0.73       0.0%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 62.28x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                 12.770 us 104611.312 us    40.61       1.4%    0.001s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_race                      12.964 us 106201.696 us    40.01       1.0%    0.017s  ok ch=2.9e-14/1e-10 1s=1e-15 1.02x
   gen_pfa_small                 13.048 us 106888.172 us    39.75       1.3%    0.001s  ok ch=2.8e-14/1e-10 1s=1e-15 1.02x
   gen_planner                   17.340 us 142052.978 us    29.91       1.5%    0.016s  ok ch=2.9e-14/1e-10 1s=1e-15 1.36x
   gen_layout                    34.200 us 280164.509 us    15.16       4.2%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 2.68x
   gen_twiddle                   37.016 us 303236.070 us    14.01       2.9%    0.004s  ok ch=2.9e-14/1e-10 1s=1e-15 2.90x
   gen_dense_prime               38.849 us 318250.357 us    13.35       0.6%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 3.04x
   fftw3_measure                 44.947 us 368203.277 us    11.54       1.3%    0.083s  ok ch=3.1e-14/1e-10 1s=1e-15 3.52x
   fftw3_patient                 45.184 us 370149.727 us    11.48       0.4%    0.316s  ok ch=3.3e-14/1e-10 1s=1e-15 3.54x
   mkl2026_dfti                  57.608 us 471924.106 us     9.00       3.9%    0.051s  ok ch=3.3e-14/1e-10 1s=1e-15 4.51x
   mkl_dfti                      58.259 us 477258.186 us     8.90       0.3%    0.031s  ok ch=3.3e-14/1e-10 1s=1e-15 4.56x
   fftw3_guru                    60.118 us 492485.689 us     8.63       0.1%    0.075s  ok ch=3.1e-14/1e-10 1s=1e-15 4.71x
   ducc0_c2c                     72.677 us 595369.246 us     7.14       1.1%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 5.69x
   gen_bluestein                 82.140 us 672887.190 us     6.31       2.0%    0.000s  ok ch=5.3e-14/1e-10 1s=2e-15 6.43x
   fftw3_estimate                91.934 us 753126.858 us     5.64       0.1%    0.001s  ok ch=3.2e-14/1e-10 1s=1e-15 7.20x
   baseline_matrix              850.286 us 6965546.520 us     0.61       0.0%    0.000s  ok ch=9.7e-14/1e-10 1s=2e-15 66.59x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      40.083 us 164180.611 us    27.15      13.3%    0.009s  ok ch=3.5e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   40.192 us 164628.477 us    27.08       2.9%    0.029s  ok ch=3.5e-14/1e-10 1s=2e-15 1.00x
   gen_powp                      41.025 us 168036.616 us    26.53       0.0%    0.485s  ok ch=4.4e-14/1e-10 1s=1e-15 1.02x
   gen_twiddle                   80.971 us 331656.030 us    13.44       4.2%    0.004s  ok ch=4.3e-14/1e-10 1s=2e-15 2.02x
   gen_layout                    94.095 us 385414.302 us    11.57       1.8%    0.000s  ok ch=3.5e-14/1e-10 1s=2e-15 2.35x
   fftw3_measure                108.681 us 445158.537 us    10.01       3.2%    0.028s  ok ch=3.5e-14/1e-10 1s=2e-15 2.71x
   fftw3_patient                110.997 us 454644.264 us     9.81       2.0%    0.103s  ok ch=3.8e-14/1e-10 1s=2e-15 2.77x
   mkl_dfti                     120.680 us 494305.417 us     9.02       0.3%    0.044s  ok ch=4.7e-14/1e-10 1s=2e-15 3.01x
   mkl2026_dfti                 124.042 us 508077.570 us     8.77       0.5%    0.075s  ok ch=4.0e-14/1e-10 1s=2e-15 3.09x
   fftw3_estimate               131.911 us 540307.888 us     8.25       3.2%    0.001s  ok ch=3.9e-14/1e-10 1s=2e-15 3.29x
   ducc0_c2c                    144.659 us 592523.760 us     7.52       1.1%    0.000s  ok ch=3.7e-14/1e-10 1s=1e-15 3.61x
   fftw3_guru                   175.110 us 717250.996 us     6.22       0.2%    0.030s  ok ch=3.5e-14/1e-10 1s=2e-15 4.37x
   gen_bluestein                180.699 us 740143.536 us     6.02       6.4%    0.000s  ok ch=5.9e-14/1e-10 1s=2e-15 4.51x
   baseline_matrix             2061.011 us 8441902.320 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 51.42x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      43.357 us 138743.228 us    32.38       2.2%    0.515s  ok ch=3.9e-14/1e-10 1s=2e-15 1.00x
   gen_race                      43.689 us 139803.699 us    32.13       1.8%    0.015s  ok ch=3.9e-14/1e-10 1s=2e-15 1.01x
   gen_planner                   60.241 us 192772.160 us    23.30       0.5%    0.033s  ok ch=4.0e-14/1e-10 1s=2e-15 1.39x
   gen_layout                   123.839 us 396283.792 us    11.34       1.1%    0.001s  ok ch=3.4e-14/1e-10 1s=2e-15 2.86x
   gen_twiddle                  128.318 us 410616.201 us    10.94       0.8%    0.005s  ok ch=4.3e-14/1e-10 1s=2e-15 2.96x
   mkl_dfti                     144.242 us 461574.878 us     9.73       0.1%    0.030s  ok ch=3.4e-14/1e-10 1s=2e-15 3.33x
   mkl2026_dfti                 147.852 us 473125.647 us     9.50       0.2%    0.052s  ok ch=3.4e-14/1e-10 1s=2e-15 3.41x
   ducc0_c2c                    189.076 us 605044.270 us     7.42       3.7%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 4.36x
   fftw3_patient                205.068 us 656217.647 us     6.85       1.3%    0.139s  ok ch=3.2e-14/1e-10 1s=2e-15 4.73x
   gen_bluestein                214.323 us 685833.676 us     6.55       0.6%    0.000s  ok ch=5.4e-14/1e-10 1s=2e-15 4.94x
   fftw3_measure                220.846 us 706708.588 us     6.36       2.5%    0.034s  ok ch=3.4e-14/1e-10 1s=2e-15 5.09x
   fftw3_estimate               252.515 us 808048.571 us     5.56       2.2%    0.003s  ok ch=3.4e-14/1e-10 1s=2e-15 5.82x
   fftw3_guru                   258.940 us 828609.170 us     5.42       1.0%    0.043s  ok ch=4.0e-14/1e-10 1s=2e-15 5.97x
   baseline_matrix             2799.031 us 8956897.880 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 64.56x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     84.745 us 189828.272 us    26.12       1.2%    0.008s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_race                      84.858 us 190081.796 us    26.09       0.3%    0.020s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              113.507 us 254255.589 us    19.50       0.4%    0.001s  ok ch=2.5e-14/1e-10 1s=2e-15 1.34x
   gen_planner                  140.041 us 313691.199 us    15.81       0.3%    0.040s  ok ch=2.7e-14/1e-10 1s=2e-15 1.65x
   gen_layout                   199.976 us 447947.166 us    11.07       1.2%    0.001s  ok ch=3.2e-14/1e-10 1s=2e-15 2.36x
   gen_twiddle                  267.735 us 599725.820 us     8.27       1.8%    0.004s  ok ch=2.6e-14/1e-10 1s=2e-15 3.16x
   gen_bluestein                292.564 us 655343.337 us     7.57       0.7%    0.000s  ok ch=4.5e-14/1e-10 1s=3e-15 3.45x
   ducc0_c2c                    717.096 us 1606295.530 us     3.09       7.2%    0.000s  ok ch=2.3e-14/1e-10 1s=1e-15 8.46x
   fftw3_guru                   832.965 us 1865842.290 us     2.66       0.5%    0.088s  ok ch=2.6e-14/1e-10 1s=2e-15 9.83x
   mkl_dfti                     848.573 us 1900803.340 us     2.61       0.1%    0.037s  ok ch=3.4e-14/1e-10 1s=2e-15 10.01x
   fftw3_measure                859.077 us 1924332.690 us     2.58       0.5%    0.085s  ok ch=2.6e-14/1e-10 1s=2e-15 10.14x
   fftw3_estimate               859.121 us 1924430.820 us     2.58       0.1%    0.002s  ok ch=2.6e-14/1e-10 1s=2e-15 10.14x
   fftw3_patient                860.052 us 1926515.740 us     2.57       0.3%    0.241s  ok ch=2.6e-14/1e-10 1s=2e-15 10.15x
   mkl2026_dfti                 883.028 us 1977982.650 us     2.51       0.1%    0.049s  ok ch=3.4e-14/1e-10 1s=2e-15 10.42x
   baseline_matrix             4852.868 us 10870425.100 us     0.46       0.0%    0.000s  ok ch=8.2e-14/1e-10 1s=3e-15 57.26x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      55.326 us 110651.968 us    44.42      26.5%    0.011s  ok ch=3.3e-14/1e-10 1s=1e-15 1.00x
   gen_pow2                      56.455 us 112909.778 us    43.53       1.0%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 1.02x
   gen_planner                  105.565 us 211129.955 us    23.28       0.7%    0.049s  ok ch=3.7e-14/1e-10 1s=1e-15 1.91x
   gen_layout                   149.143 us 298286.334 us    16.48       1.6%    0.001s  ok ch=4.5e-14/1e-10 1s=2e-15 2.70x
   mkl_dfti                     173.351 us 346702.024 us    14.18       0.4%    0.002s  ok ch=3.1e-14/1e-10 1s=1e-15 3.13x
   mkl2026_dfti                 184.852 us 369704.098 us    13.29       1.7%    0.002s  ok ch=3.3e-14/1e-10 1s=1e-15 3.34x
   gen_twiddle                  196.411 us 392821.768 us    12.51       0.9%    0.005s  ok ch=3.9e-14/1e-10 1s=1e-15 3.55x
   fftw3_patient                207.515 us 415030.006 us    11.84       2.9%    0.541s  ok ch=3.3e-14/1e-10 1s=1e-15 3.75x
   fftw3_measure                208.781 us 417562.259 us    11.77       0.8%    0.082s  ok ch=4.1e-14/1e-10 1s=1e-15 3.77x
   fftw3_guru                   289.426 us 578852.958 us     8.49       4.9%    0.074s  ok ch=3.8e-14/1e-10 1s=1e-15 5.23x
   ducc0_c2c                    309.793 us 619585.487 us     7.93       0.7%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 5.60x
   gen_bluestein                313.265 us 626529.394 us     7.85       1.4%    0.000s  ok ch=5.5e-14/1e-10 1s=2e-15 5.66x
   fftw3_estimate               406.899 us 813798.764 us     6.04       0.2%    0.001s  ok ch=3.8e-14/1e-10 1s=1e-15 7.35x
   baseline_matrix             5752.818 us 11505636.400 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=2e-15 103.98x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                159.708 us 163540.737 us    31.99       0.2%    1.305s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_planner                  235.309 us 240956.595 us    21.71       1.4%    0.007s  ok ch=2.5e-14/1e-10 1s=2e-15 1.47x
   gen_race                     241.810 us 247613.880 us    21.13       0.4%   30.653s  ok ch=2.5e-14/1e-10 1s=2e-15 1.51x
   gen_layout                   333.492 us 341495.350 us    15.32       0.3%    0.001s  ok ch=3.1e-14/1e-10 1s=2e-15 2.09x
   gen_twiddle                  339.515 us 347663.308 us    15.05       1.0%    0.007s  ok ch=3.2e-14/1e-10 1s=2e-15 2.13x
   mkl_dfti                     404.384 us 414088.921 us    12.63       0.7%    0.002s  ok ch=3.4e-14/1e-10 1s=2e-15 2.53x
   mkl2026_dfti                 405.049 us 414769.930 us    12.61       0.7%    0.002s  ok ch=4.7e-14/1e-10 1s=2e-15 2.54x
   fftw3_patient                525.977 us 538600.950 us     9.71       2.1%    1.478s  ok ch=2.2e-14/1e-10 1s=2e-15 3.29x
   fftw3_measure                550.724 us 563941.266 us     9.28       0.5%    0.154s  ok ch=3.0e-14/1e-10 1s=2e-15 3.45x
   ducc0_c2c                    597.652 us 611995.554 us     8.55       0.9%    0.000s  ok ch=2.2e-14/1e-10 1s=2e-15 3.74x
   gen_bluestein                617.531 us 632351.838 us     8.27       0.7%    0.000s  ok ch=4.3e-14/1e-10 1s=3e-15 3.87x
   fftw3_guru                   669.117 us 685176.211 us     7.64       2.1%    0.152s  ok ch=3.1e-14/1e-10 1s=2e-15 4.19x
   fftw3_estimate              1611.004 us 1649668.370 us     3.17       0.2%    0.002s  ok ch=2.4e-14/1e-10 1s=2e-15 10.09x
   baseline_matrix            13454.964 us 13777882.700 us     0.38       0.2%    0.000s  ok ch=6.0e-14/1e-10 1s=3e-15 84.25x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                     410.896 us 210378.832 us    25.75       7.3%    0.016s  ok ch=3.1e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                416.640 us 213319.802 us    25.40       2.1%    0.003s  ok ch=3.1e-14/1e-10 1s=2e-15 1.01x
   gen_powp                     421.653 us 215886.148 us    25.10       2.8%    1.869s  ok ch=3.1e-14/1e-10 1s=2e-15 1.03x
   gen_planner                  556.441 us 284897.582 us    19.02       0.9%    0.041s  ok ch=3.1e-14/1e-10 1s=2e-15 1.35x
   gen_twiddle                  725.371 us 371389.722 us    14.59       0.2%    0.011s  ok ch=2.9e-14/1e-10 1s=2e-15 1.77x
   gen_layout                   943.965 us 483310.142 us    11.21       0.9%    0.001s  ok ch=2.9e-14/1e-10 1s=2e-15 2.30x
   mkl_dfti                     946.694 us 484707.325 us    11.18       0.3%    0.037s  ok ch=3.1e-14/1e-10 1s=2e-15 2.30x
   mkl2026_dfti                 961.411 us 492242.450 us    11.01       0.2%    0.048s  ok ch=3.1e-14/1e-10 1s=2e-15 2.34x
   fftw3_patient               1167.598 us 597810.199 us     9.06       2.0%    1.230s  ok ch=2.9e-14/1e-10 1s=2e-15 2.84x
   fftw3_measure               1197.525 us 613132.939 us     8.84       0.7%    0.091s  ok ch=2.9e-14/1e-10 1s=2e-15 2.91x
   ducc0_c2c                   1270.889 us 650695.282 us     8.33       0.5%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 3.09x
   gen_bluestein               1392.512 us 712966.102 us     7.60       0.2%    0.000s  ok ch=4.4e-14/1e-10 1s=3e-15 3.39x
   fftw3_estimate              1618.710 us 828779.509 us     6.54       1.3%    0.002s  ok ch=3.1e-14/1e-10 1s=2e-15 3.94x
   fftw3_guru                  1636.933 us 838109.639 us     6.46       5.6%    0.116s  ok ch=2.7e-14/1e-10 1s=2e-15 3.98x
   baseline_matrix            33961.212 us 17388140.500 us     0.31       2.1%    0.000s  ok ch=8.1e-14/1e-10 1s=4e-15 82.65x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                    4562.285 us 291986.266 us    21.84       1.7%    0.018s  ok ch=2.1e-14/1e-10 1s=3e-15 1.00x
   gen_pfa_large               4567.719 us 292334.015 us    21.82       2.0%    0.003s  ok ch=2.1e-14/1e-10 1s=3e-15 1.00x
   gen_powp                    4596.649 us 294185.551 us    21.68      17.2%    3.350s  ok ch=2.1e-14/1e-10 1s=3e-15 1.01x
   gen_planner                 4978.108 us 318598.910 us    20.02       0.4%    0.320s  ok ch=2.2e-14/1e-10 1s=3e-15 1.09x
   gen_twiddle                 7317.669 us 468330.793 us    13.62       2.4%    0.075s  ok ch=2.4e-14/1e-10 1s=3e-15 1.60x
   mkl_dfti                    7802.441 us 499356.239 us    12.77       0.3%    0.032s  ok ch=2.7e-14/1e-10 1s=3e-15 1.71x
   mkl2026_dfti                7839.055 us 501699.530 us    12.71       7.5%    0.051s  ok ch=2.6e-14/1e-10 1s=3e-15 1.72x
   gen_layout                  9307.864 us 595703.281 us    10.71       0.5%    0.004s  ok ch=2.1e-14/1e-10 1s=3e-15 2.04x
   fftw3_patient               9928.302 us 635411.345 us    10.04       2.9%   17.387s  ok ch=2.2e-14/1e-10 1s=3e-15 2.18x
   ducc0_c2c                  11831.326 us 757204.890 us     8.42       1.5%    0.000s  ok ch=1.8e-14/1e-10 1s=2e-15 2.59x
