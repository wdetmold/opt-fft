# What previous generations produced (round gen_r10 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gen/results/gen_r1/leaderboard.txt
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
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_batchlane.md 1253 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_bluestein.md 1016 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_dense_prime.md 1427 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_layout.md 1445 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_large.md 1383 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pfa_small.md 1253 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_planner.md 1218 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_pow2.md 1291 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_powp.md 1527 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_race.md 1530 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_rader.md 1352 lines
  /home/lqcd/wdetmold/fft/bench/gen/strategies/gen_twiddle.md 1461 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gen_r9 ===
# round gen_r9
host: a81n2.lqcd.mit   date: 2026-08-25T23:43:54-04:00   slurm_job: 438854
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.121 us 71772.266 us    44.43       0.2%    0.000s  ok ch=1.1e-13/1e-10 1s=9e-16 1.00x
   gen_race                       1.122 us 71816.906 us    44.41      14.0%    9.841s  ok ch=1.1e-13/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.142 us 73113.969 us    43.62      13.7%    0.000s  ok ch=1.2e-13/1e-10 1s=9e-16 1.02x
   gen_planner                    1.320 us 84497.708 us    37.74       5.3%    0.002s  ok ch=1.1e-13/1e-10 1s=1e-15 1.18x
   fftw3_custom_soa               4.535 us 290239.601 us    10.99       5.4%    0.000s  ok ch=8.8e-14/1e-10 1s=8e-16 4.04x
   mkl_dfti                       4.583 us 293311.556 us    10.87       0.1%    0.001s  ok ch=1.2e-13/1e-10 1s=1e-15 4.09x
   mkl2026_dfti                   4.644 us 297228.008 us    10.73       0.0%    0.001s  ok ch=1.4e-13/1e-10 1s=1e-15 4.14x
   gen_layout                     4.952 us 316940.662 us    10.06       5.0%    0.000s  ok ch=1.0e-13/1e-10 1s=9e-16 4.42x
   fftw3_measure                  5.124 us 327936.336 us     9.72       5.7%    0.013s  ok ch=8.7e-14/1e-10 1s=8e-16 4.57x
   gen_dense_prime                5.151 us 329648.829 us     9.67       4.2%    0.000s  ok ch=8.7e-14/1e-10 1s=9e-16 4.59x
   fftw3_patient                  5.180 us 331509.595 us     9.62       0.9%    0.022s  ok ch=8.7e-14/1e-10 1s=8e-16 4.62x
   gen_twiddle                    5.319 us 340435.628 us     9.37       3.7%    0.003s  ok ch=9.9e-14/1e-10 1s=9e-16 4.74x
   fftw3_custom                   5.964 us 381694.782 us     8.35       0.2%    0.000s  ok ch=8.8e-14/1e-10 1s=8e-16 5.32x
   fftw3_guru                     6.374 us 407920.277 us     7.82       0.3%    0.010s  ok ch=8.7e-14/1e-10 1s=8e-16 5.68x
   fftw3_estimate                 7.316 us 468198.050 us     6.81       0.2%    0.001s  ok ch=8.7e-14/1e-10 1s=8e-16 6.52x
   ducc0_c2c                      9.621 us 615775.731 us     5.18       0.6%    0.000s  ok ch=8.1e-14/1e-10 1s=8e-16 8.58x
   gen_bluestein                 12.825 us 820825.092 us     3.89       1.6%    0.000s  ok ch=1.6e-13/1e-10 1s=1e-15 11.44x
   baseline_matrix               54.697 us 3500615.720 us     0.91       0.0%    0.000s  ok ch=3.2e-13/1e-10 1s=1e-15 48.77x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       1.913 us 73471.806 us    48.57       0.1%    0.176s  ok ch=5.6e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.915 us 73546.487 us    48.52       0.4%    0.000s  ok ch=5.6e-14/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.917 us 73602.854 us    48.48      13.8%    0.001s  ok ch=5.6e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.337 us 89746.954 us    39.76       1.3%    0.003s  ok ch=6.3e-14/1e-10 1s=9e-16 1.22x
   gen_twiddle                    7.553 us 290051.929 us    12.30       5.3%    0.003s  ok ch=6.1e-14/1e-10 1s=1e-15 3.95x
   mkl_dfti                       7.755 us 297780.251 us    11.98       2.1%    0.002s  ok ch=5.9e-14/1e-10 1s=9e-16 4.05x
   mkl2026_dfti                   7.779 us 298703.591 us    11.95       2.2%    0.002s  ok ch=6.1e-14/1e-10 1s=9e-16 4.07x
   fftw3_custom_soa               8.002 us 307273.599 us    11.61       0.1%    0.000s  ok ch=5.9e-14/1e-10 1s=9e-16 4.18x
   gen_layout                     8.141 us 312595.958 us    11.41       1.9%    0.000s  ok ch=6.7e-14/1e-10 1s=9e-16 4.25x
   gen_dense_prime                8.147 us 312836.739 us    11.41       1.9%    0.000s  ok ch=5.3e-14/1e-10 1s=9e-16 4.26x
   fftw3_measure                  8.626 us 331236.636 us    10.77       2.8%    0.014s  ok ch=6.6e-14/1e-10 1s=9e-16 4.51x
   fftw3_patient                  8.651 us 332216.943 us    10.74       4.1%    0.023s  ok ch=6.5e-14/1e-10 1s=9e-16 4.52x
   fftw3_guru                    10.220 us 392449.172 us     9.09       1.0%    0.010s  ok ch=6.1e-14/1e-10 1s=9e-16 5.34x
   fftw3_custom                  10.498 us 403135.942 us     8.85       1.1%    0.000s  ok ch=5.9e-14/1e-10 1s=9e-16 5.49x
   ducc0_c2c                     16.008 us 614689.026 us     5.80       1.6%    0.000s  ok ch=4.0e-14/1e-10 1s=7e-16 8.37x
   gen_bluestein                 18.887 us 725251.329 us     4.92       1.2%    0.000s  ok ch=1.0e-13/1e-10 1s=1e-15 9.87x
   fftw3_estimate                19.297 us 740995.708 us     4.82       0.0%    0.001s  ok ch=6.5e-14/1e-10 1s=9e-16 10.09x
   baseline_matrix              112.116 us 4305262.740 us     0.83       0.1%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 58.60x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                       4.326 us 83057.657 us    45.72      14.1%    0.008s  ok ch=5.5e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.339 us 83316.368 us    45.58       0.1%    0.000s  ok ch=5.5e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                  4.388 us 84248.807 us    45.07       7.4%    0.000s  ok ch=5.1e-14/1e-10 1s=1e-15 1.01x
   gen_planner                    5.564 us 106824.626 us    35.55       1.2%    0.008s  ok ch=5.6e-14/1e-10 1s=1e-15 1.29x
   gen_dense_prime               15.300 us 293766.787 us    12.93       5.4%    0.000s  ok ch=4.8e-14/1e-10 1s=1e-15 3.54x
   fftw3_custom_soa              15.551 us 298573.844 us    12.72       2.0%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 3.59x
   mkl_dfti                      16.492 us 316646.508 us    11.99       1.5%    0.002s  ok ch=7.2e-14/1e-10 1s=1e-15 3.81x
   mkl2026_dfti                  16.713 us 320882.214 us    11.83       0.3%    0.002s  ok ch=6.7e-14/1e-10 1s=1e-15 3.86x
   gen_twiddle                   17.979 us 345187.908 us    11.00       3.7%    0.002s  ok ch=5.4e-14/1e-10 1s=1e-15 4.16x
   fftw3_measure                 19.605 us 376419.606 us    10.09       5.6%    0.011s  ok ch=5.3e-14/1e-10 1s=1e-15 4.53x
   gen_layout                    19.683 us 377917.930 us    10.05       2.6%    0.000s  ok ch=5.0e-14/1e-10 1s=1e-15 4.55x
   fftw3_patient                 19.697 us 378188.844 us    10.04       0.8%    0.020s  ok ch=5.2e-14/1e-10 1s=1e-15 4.55x
   fftw3_estimate                20.643 us 396343.435 us     9.58       0.2%    0.001s  ok ch=5.3e-14/1e-10 1s=1e-15 4.77x
   fftw3_custom                  25.152 us 482915.874 us     7.86       1.8%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 5.81x
   fftw3_guru                    26.170 us 502455.309 us     7.56       2.3%    0.012s  ok ch=5.6e-14/1e-10 1s=1e-15 6.05x
   ducc0_c2c                     32.434 us 622727.169 us     6.10       1.6%    0.000s  ok ch=4.3e-14/1e-10 1s=1e-15 7.50x
   gen_bluestein                 32.894 us 631573.077 us     6.01       1.4%    0.000s  ok ch=1.0e-13/1e-10 1s=2e-15 7.60x
   baseline_matrix              272.405 us 5230170.600 us     0.73       0.1%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 62.97x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      12.549 us 102802.172 us    41.33       2.3%    0.254s  ok ch=3.1e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                 12.671 us 103802.385 us    40.93       1.2%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 1.01x
   gen_pfa_small                 13.019 us 106648.924 us    39.84       3.9%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 1.04x
   gen_planner                   17.361 us 142225.242 us    29.87       1.1%    0.009s  ok ch=2.6e-14/1e-10 1s=1e-15 1.38x
   gen_twiddle                   32.839 us 269015.281 us    15.79       4.3%    0.003s  ok ch=2.7e-14/1e-10 1s=1e-15 2.62x
   gen_layout                    35.351 us 289598.982 us    14.67       3.7%    0.001s  ok ch=2.6e-14/1e-10 1s=1e-15 2.82x
   gen_dense_prime               38.172 us 312707.230 us    13.59       2.1%    0.000s  ok ch=2.7e-14/1e-10 1s=1e-15 3.04x
   fftw3_custom_soa              43.087 us 352966.363 us    12.04       3.3%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 3.43x
   fftw3_patient                 45.038 us 368951.507 us    11.52       1.3%    0.301s  ok ch=2.4e-14/1e-10 1s=1e-15 3.59x
   fftw3_measure                 45.065 us 369169.260 us    11.51       1.4%    0.084s  ok ch=2.4e-14/1e-10 1s=1e-15 3.59x
   mkl2026_dfti                  57.379 us 470049.158 us     9.04       2.3%    0.053s  ok ch=2.9e-14/1e-10 1s=1e-15 4.57x
   mkl_dfti                      58.973 us 483109.160 us     8.79       1.6%    0.058s  ok ch=2.8e-14/1e-10 1s=1e-15 4.70x
   fftw3_guru                    60.101 us 492343.574 us     8.63       3.3%    0.077s  ok ch=2.7e-14/1e-10 1s=1e-15 4.79x
   fftw3_custom                  64.475 us 528177.700 us     8.04       0.2%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 5.14x
   ducc0_c2c                     73.260 us 600144.358 us     7.08       2.4%    0.000s  ok ch=2.2e-14/1e-10 1s=1e-15 5.84x
   gen_bluestein                 80.818 us 662059.118 us     6.42       1.7%    0.000s  ok ch=4.1e-14/1e-10 1s=2e-15 6.44x
   fftw3_estimate                91.516 us 749696.693 us     5.67       0.5%    0.001s  ok ch=2.4e-14/1e-10 1s=1e-15 7.29x
   baseline_matrix              850.351 us 6966078.650 us     0.61       0.0%    0.000s  ok ch=9.6e-14/1e-10 1s=2e-15 67.76x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      31.054 us 127195.187 us    35.05       6.8%    0.003s  ok ch=3.9e-14/1e-10 1s=2e-15 1.00x
   gen_race                      31.486 us 128965.924 us    34.57       0.2%    0.227s  ok ch=3.9e-14/1e-10 1s=2e-15 1.01x
   gen_planner                   39.589 us 162158.010 us    27.49       2.9%    0.013s  ok ch=4.9e-14/1e-10 1s=2e-15 1.27x
   gen_twiddle                   74.963 us 307049.314 us    14.52       3.5%    0.004s  ok ch=4.4e-14/1e-10 1s=2e-15 2.41x
   fftw3_custom_soa              77.748 us 318455.137 us    14.00       5.2%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 2.50x
   gen_layout                    93.493 us 382947.519 us    11.64       2.0%    0.000s  ok ch=3.7e-14/1e-10 1s=2e-15 3.01x
   fftw3_measure                108.642 us 444997.612 us    10.02       4.1%    0.029s  ok ch=4.3e-14/1e-10 1s=2e-15 3.50x
   fftw3_patient                109.085 us 446812.605 us     9.98       4.1%    0.113s  ok ch=4.1e-14/1e-10 1s=2e-15 3.51x
   mkl_dfti                     120.934 us 495347.683 us     9.00       0.4%    0.048s  ok ch=3.6e-14/1e-10 1s=2e-15 3.89x
   mkl2026_dfti                 122.986 us 503752.459 us     8.85       0.6%    0.049s  ok ch=3.8e-14/1e-10 1s=2e-15 3.96x
   fftw3_estimate               133.022 us 544857.308 us     8.18       5.1%    0.001s  ok ch=4.3e-14/1e-10 1s=2e-15 4.28x
   ducc0_c2c                    146.765 us 601150.722 us     7.42       1.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 4.73x
   fftw3_custom                 162.526 us 665705.769 us     6.70       0.4%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 5.23x
   gen_bluestein                169.597 us 694668.018 us     6.42       1.6%    0.000s  ok ch=8.8e-14/1e-10 1s=2e-15 5.46x
   fftw3_guru                   175.804 us 720092.592 us     6.19       6.0%    0.032s  ok ch=4.3e-14/1e-10 1s=2e-15 5.66x
   baseline_matrix             2061.549 us 8444104.270 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 66.39x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      43.607 us 139543.375 us    32.19       7.7%    0.004s  ok ch=4.2e-14/1e-10 1s=2e-15 1.00x
   gen_race                      43.713 us 139881.865 us    32.12       0.5%    0.319s  ok ch=4.2e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   59.926 us 191763.731 us    23.43       6.6%    0.016s  ok ch=4.1e-14/1e-10 1s=2e-15 1.37x
   fftw3_custom_soa              97.315 us 311408.094 us    14.43       2.0%    0.000s  ok ch=3.7e-14/1e-10 1s=2e-15 2.23x
   gen_twiddle                  121.330 us 388255.355 us    11.57       0.9%    0.004s  ok ch=3.8e-14/1e-10 1s=2e-15 2.78x
   gen_layout                   124.930 us 399775.459 us    11.24       0.5%    0.000s  ok ch=4.1e-14/1e-10 1s=2e-15 2.86x
   mkl_dfti                     144.787 us 463318.254 us     9.70       0.4%    0.048s  ok ch=4.6e-14/1e-10 1s=2e-15 3.32x
   mkl2026_dfti                 148.358 us 474745.782 us     9.46       0.3%    0.051s  ok ch=4.6e-14/1e-10 1s=2e-15 3.40x
   ducc0_c2c                    190.829 us 610653.048 us     7.36       2.1%    0.000s  ok ch=3.3e-14/1e-10 1s=1e-15 4.38x
   gen_bluestein                202.511 us 648034.763 us     6.93       1.7%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 4.64x
   fftw3_patient                202.719 us 648701.995 us     6.93       0.5%    0.138s  ok ch=3.8e-14/1e-10 1s=2e-15 4.65x
   fftw3_custom                 203.621 us 651587.927 us     6.89       0.2%    0.000s  ok ch=3.7e-14/1e-10 1s=2e-15 4.67x
   fftw3_measure                223.922 us 716550.555 us     6.27       1.7%    0.034s  ok ch=5.7e-14/1e-10 1s=2e-15 5.13x
   fftw3_estimate               256.119 us 819581.115 us     5.48       1.8%    0.003s  ok ch=5.1e-14/1e-10 1s=2e-15 5.87x
   fftw3_guru                   260.679 us 834173.513 us     5.39       0.3%    0.043s  ok ch=4.2e-14/1e-10 1s=2e-15 5.98x
   baseline_matrix             2798.775 us 8956080.650 us     0.50       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=3e-15 64.18x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      84.694 us 189715.480 us    26.14       0.4%    0.499s  ok ch=2.7e-14/1e-10 1s=2e-15 1.00x
   gen_rader                     85.210 us 190869.840 us    25.98       0.1%    0.007s  ok ch=2.7e-14/1e-10 1s=2e-15 1.01x
   gen_dense_prime              109.955 us 246299.875 us    20.13       1.3%    0.001s  ok ch=2.3e-14/1e-10 1s=2e-15 1.30x
   gen_planner                  138.384 us 309980.172 us    16.00       1.4%    0.022s  ok ch=2.8e-14/1e-10 1s=2e-15 1.63x
   gen_layout                   200.300 us 448672.543 us    11.05       1.5%    0.001s  ok ch=2.8e-14/1e-10 1s=2e-15 2.36x
   fftw3_custom_soa             216.444 us 484834.489 us    10.23       0.8%    0.000s  ok ch=2.8e-14/1e-10 1s=2e-15 2.56x
   gen_twiddle                  263.529 us 590305.894 us     8.40       1.0%    0.005s  ok ch=2.9e-14/1e-10 1s=2e-15 3.11x
   gen_bluestein                272.594 us 610609.972 us     8.12       1.5%    0.000s  ok ch=4.4e-14/1e-10 1s=3e-15 3.22x
   fftw3_custom                 513.157 us 1149472.510 us     4.31       0.6%    0.000s  ok ch=2.8e-14/1e-10 1s=2e-15 6.06x
   ducc0_c2c                    714.168 us 1599735.470 us     3.10       2.6%    0.000s  ok ch=2.0e-14/1e-10 1s=1e-15 8.43x
   fftw3_guru                   830.334 us 1859949.120 us     2.67       0.7%    0.093s  ok ch=2.5e-14/1e-10 1s=2e-15 9.80x
   mkl_dfti                     848.751 us 1901203.040 us     2.61       0.1%    0.049s  ok ch=3.1e-14/1e-10 1s=2e-15 10.02x
   fftw3_estimate               859.365 us 1924977.560 us     2.58       0.2%    0.002s  ok ch=2.5e-14/1e-10 1s=2e-15 10.15x
   fftw3_measure                859.829 us 1926017.640 us     2.57       0.1%    0.085s  ok ch=2.5e-14/1e-10 1s=2e-15 10.15x
   fftw3_patient                859.997 us 1926392.770 us     2.57       0.3%    0.240s  ok ch=2.5e-14/1e-10 1s=2e-15 10.15x
   mkl2026_dfti                 883.341 us 1978684.480 us     2.51       0.1%    0.048s  ok ch=3.1e-14/1e-10 1s=2e-15 10.43x
   baseline_matrix             4852.591 us 10869804.300 us     0.46       0.0%    0.000s  ok ch=7.9e-14/1e-10 1s=3e-15 57.30x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                      53.999 us 107997.712 us    45.51       1.3%    0.006s  ok ch=3.0e-14/1e-10 1s=1e-15 1.00x
   gen_pow2                      55.161 us 110321.918 us    44.55       0.4%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 1.02x
   gen_planner                  105.865 us 211730.843 us    23.21       1.3%    0.053s  ok ch=4.0e-14/1e-10 1s=1e-15 1.96x
   gen_layout                   150.472 us 300944.296 us    16.33       1.9%    0.000s  ok ch=3.5e-14/1e-10 1s=1e-15 2.79x
   mkl_dfti                     172.105 us 344210.287 us    14.28       0.8%    0.002s  ok ch=2.8e-14/1e-10 1s=1e-15 3.19x
   fftw3_custom_soa             176.973 us 353946.103 us    13.89       5.1%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 3.28x
   mkl2026_dfti                 182.105 us 364209.977 us    13.50       4.1%    0.001s  ok ch=3.2e-14/1e-10 1s=1e-15 3.37x
   gen_twiddle                  196.497 us 392994.827 us    12.51       0.3%    0.006s  ok ch=3.3e-14/1e-10 1s=1e-15 3.64x
   fftw3_measure                208.904 us 417807.134 us    11.76       0.5%    0.083s  ok ch=3.6e-14/1e-10 1s=1e-15 3.87x
   fftw3_patient                209.120 us 418240.782 us    11.75       0.5%    0.544s  ok ch=3.6e-14/1e-10 1s=1e-15 3.87x
   gen_bluestein                297.242 us 594484.622 us     8.27       1.9%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 5.50x
   fftw3_guru                   302.841 us 605681.025 us     8.12       6.8%    0.074s  ok ch=3.2e-14/1e-10 1s=1e-15 5.61x
   ducc0_c2c                    308.816 us 617632.982 us     7.96       0.8%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 5.72x
   fftw3_estimate               407.472 us 814944.600 us     6.03       0.1%    0.001s  ok ch=3.3e-14/1e-10 1s=1e-15 7.55x
   fftw3_custom                 414.788 us 829575.588 us     5.92       0.5%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 7.68x
   baseline_matrix             5752.909 us 11505818.900 us     0.43       0.1%    0.000s  ok ch=1.2e-13/1e-10 1s=3e-15 106.54x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_race                     159.721 us 163554.283 us    31.99       5.0%    0.565s  ok ch=2.5e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                160.167 us 164010.654 us    31.90       0.4%    0.004s  ok ch=2.5e-14/1e-10 1s=2e-15 1.00x
   gen_planner                  240.053 us 245813.947 us    21.28       3.1%    0.047s  ok ch=2.1e-14/1e-10 1s=2e-15 1.50x
   gen_layout                   328.658 us 336545.566 us    15.55       6.4%    0.001s  ok ch=2.2e-14/1e-10 1s=2e-15 2.06x
   gen_twiddle                  336.718 us 344798.921 us    15.17       1.7%    0.007s  ok ch=2.3e-14/1e-10 1s=2e-15 2.11x
   mkl2026_dfti                 406.933 us 416699.697 us    12.56       0.4%    0.003s  ok ch=4.0e-14/1e-10 1s=2e-15 2.55x
   mkl_dfti                     407.116 us 416886.555 us    12.55       0.7%    0.002s  ok ch=2.7e-14/1e-10 1s=2e-15 2.55x
   fftw3_custom_soa             422.408 us 432546.164 us    12.10       1.1%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 2.64x
   fftw3_patient                525.575 us 538189.249 us     9.72       1.5%    1.443s  ok ch=2.3e-14/1e-10 1s=2e-15 3.29x
   fftw3_measure                553.123 us 566398.042 us     9.24       4.9%    0.147s  ok ch=2.1e-14/1e-10 1s=2e-15 3.46x
   ducc0_c2c                    596.282 us 610593.019 us     8.57       1.6%    0.000s  ok ch=2.0e-14/1e-10 1s=2e-15 3.73x
   gen_bluestein                605.286 us 619812.859 us     8.44       1.1%    0.000s  ok ch=3.9e-14/1e-10 1s=3e-15 3.79x
   fftw3_guru                   676.467 us 692702.357 us     7.55       1.8%    0.158s  ok ch=2.8e-14/1e-10 1s=2e-15 4.24x
   fftw3_custom                 776.236 us 794866.136 us     6.58       2.6%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 4.86x
   fftw3_estimate              1574.667 us 1612458.870 us     3.24       2.4%    0.002s  ok ch=2.7e-14/1e-10 1s=2e-15 9.86x
   baseline_matrix            13452.864 us 13775732.500 us     0.38       0.2%    0.000s  ok ch=5.9e-14/1e-10 1s=3e-15 84.23x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                     417.400 us 213708.872 us    25.35       2.3%    1.444s  ok ch=3.7e-14/1e-10 1s=2e-15 1.00x
   gen_race                     419.645 us 214858.167 us    25.22       1.7%    0.008s  ok ch=3.7e-14/1e-10 1s=2e-15 1.01x
   gen_pfa_large                437.059 us 223774.370 us    24.21       1.2%    1.924s  ok ch=3.7e-14/1e-10 1s=2e-15 1.05x
   gen_planner                  563.424 us 288473.106 us    18.78       0.7%    0.054s  ok ch=3.4e-14/1e-10 1s=2e-15 1.35x
   gen_twiddle                  716.529 us 366862.838 us    14.77       0.1%    0.011s  ok ch=3.5e-14/1e-10 1s=2e-15 1.72x
   gen_layout                   923.055 us 472603.945 us    11.46       5.1%    0.001s  ok ch=3.3e-14/1e-10 1s=2e-15 2.21x
   mkl_dfti                     951.936 us 487391.388 us    11.12       0.0%    0.030s  ok ch=3.1e-14/1e-10 1s=2e-15 2.28x
   mkl2026_dfti                 963.258 us 493188.196 us    10.99       0.4%    0.050s  ok ch=3.4e-14/1e-10 1s=2e-15 2.31x
   fftw3_patient               1171.567 us 599842.309 us     9.03       0.8%    1.316s  ok ch=3.3e-14/1e-10 1s=2e-15 2.81x
   fftw3_measure               1203.598 us 616242.401 us     8.79       1.6%    0.092s  ok ch=3.7e-14/1e-10 1s=2e-15 2.88x
