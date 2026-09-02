```
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
   fftw3_measure                209.212 us 418423.066 us    11.75       2.2%    0.083s  ok ch=3.2e-14/1e-10 1s=1e-15 3.79x
   fftw3_patient                210.453 us 420906.698 us    11.68       1.0%    0.542s  ok ch=3.5e-14/1e-10 1s=1e-15 3.81x
   fftw3_guru                   282.279 us 564557.782 us     8.71       9.2%    0.073s  ok ch=2.8e-14/1e-10 1s=1e-15 5.11x
   gen_bluestein                303.410 us 606820.707 us     8.10       2.2%    0.000s  ok ch=4.5e-14/1e-10 1s=2e-15 5.50x
   ducc0_c2c                    307.228 us 614455.057 us     8.00       3.5%    0.000s  ok ch=2.3e-14/1e-10 1s=1e-15 5.57x
   fftw3_estimate               407.346 us 814692.267 us     6.03       0.9%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 7.38x
   fftw3_custom                 420.176 us 840351.966 us     5.85       0.3%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 7.61x
   baseline_matrix             5834.739 us 11669477.200 us     0.42       0.1%    0.000s  ok ch=1.2e-13/1e-10 1s=3e-15 105.72x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                161.068 us 164933.385 us    31.72       1.7%    0.003s  ok ch=2.3e-14/1e-10 1s=2e-15 1.00x
   gen_race                     162.172 us 166064.125 us    31.50       1.9%    0.010s  ok ch=2.3e-14/1e-10 1s=2e-15 1.01x
   gen_planner                  230.535 us 236067.396 us    22.16       1.2%    0.143s  ok ch=2.3e-14/1e-10 1s=2e-15 1.43x
   gen_twiddle                  283.036 us 289828.837 us    18.05       1.2%    0.006s  ok ch=4.0e-14/1e-10 1s=2e-15 1.76x
   gen_layout                   339.751 us 347904.763 us    15.04       1.2%    0.001s  ok ch=2.2e-14/1e-10 1s=2e-15 2.11x
   mkl2026_dfti                 409.863 us 419699.330 us    12.47       0.6%    0.002s  ok ch=4.0e-14/1e-10 1s=2e-15 2.54x
   mkl_dfti                     410.525 us 420377.991 us    12.45       0.2%    0.003s  ok ch=2.8e-14/1e-10 1s=2e-15 2.55x
   fftw3_custom_soa             424.893 us 435090.072 us    12.02       5.0%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 2.64x
   fftw3_patient                530.371 us 543100.315 us     9.63       5.3%    1.495s  ok ch=2.5e-14/1e-10 1s=2e-15 3.29x
   fftw3_measure                559.662 us 573093.825 us     9.13       2.1%    0.153s  ok ch=2.1e-14/1e-10 1s=2e-15 3.47x
   ducc0_c2c                    599.634 us 614025.237 us     8.52       0.9%    0.000s  ok ch=1.9e-14/1e-10 1s=1e-15 3.72x
   gen_bluestein                612.947 us 627657.958 us     8.34       1.7%    0.000s  ok ch=3.6e-14/1e-10 1s=3e-15 3.81x
   fftw3_guru                   681.906 us 698272.111 us     7.49       8.7%    0.158s  ok ch=2.2e-14/1e-10 1s=2e-15 4.23x
   fftw3_custom                 790.951 us 809934.057 us     6.46       3.8%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 4.91x
   fftw3_estimate              1625.877 us 1664897.900 us     3.14       0.6%    0.002s  ok ch=2.2e-14/1e-10 1s=2e-15 10.09x
   baseline_matrix            13650.081 us 13977683.200 us     0.37       0.1%    0.000s  ok ch=6.6e-14/1e-10 1s=3e-15 84.75x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                418.260 us 214149.051 us    25.30       5.4%    0.003s  ok ch=3.5e-14/1e-10 1s=2e-15 1.00x
   gen_powp                     420.549 us 215321.080 us    25.16       2.1%    0.003s  ok ch=3.5e-14/1e-10 1s=2e-15 1.01x
   gen_race                     423.261 us 216709.507 us    25.00       4.2%    0.814s  ok ch=3.5e-14/1e-10 1s=2e-15 1.01x
   gen_batchlane                482.103 us 246836.663 us    21.95       4.4%    0.001s  ok ch=3.6e-14/1e-10 1s=2e-15 1.15x
   gen_pfa_small                551.401 us 282317.376 us    19.19       0.3%    0.010s  ok ch=3.2e-14/1e-10 1s=2e-15 1.32x
   gen_planner                  556.886 us 285125.435 us    19.00       3.2%    0.013s  ok ch=3.2e-14/1e-10 1s=2e-15 1.33x
   gen_twiddle                  624.582 us 319785.870 us    16.94       2.5%    0.012s  ok ch=3.0e-14/1e-10 1s=2e-15 1.49x
   gen_layout                   953.282 us 488080.537 us    11.10       1.1%    0.001s  ok ch=3.6e-14/1e-10 1s=2e-15 2.28x
   mkl_dfti                     956.976 us 489971.947 us    11.06       0.5%    0.052s  ok ch=3.8e-14/1e-10 1s=3e-15 2.29x
   mkl2026_dfti                 972.993 us 498172.432 us    10.88       0.6%    0.050s  ok ch=3.8e-14/1e-10 1s=3e-15 2.33x
   fftw3_patient               1160.268 us 594057.228 us     9.12       5.4%    1.345s  ok ch=3.3e-14/1e-10 1s=2e-15 2.77x
   fftw3_measure               1191.802 us 610202.594 us     8.88       2.8%    0.093s  ok ch=3.2e-14/1e-10 1s=3e-15 2.85x
   ducc0_c2c                   1274.495 us 652541.621 us     8.30       1.6%    0.000s  ok ch=2.8e-14/1e-10 1s=2e-15 3.05x
   gen_bluestein               1392.482 us 712950.763 us     7.60       0.9%    0.000s  ok ch=5.0e-14/1e-10 1s=3e-15 3.33x
   fftw3_estimate              1621.706 us 830313.557 us     6.53       0.6%    0.003s  ok ch=3.3e-14/1e-10 1s=2e-15 3.88x
   fftw3_guru                  1664.790 us 852372.434 us     6.36       2.2%    0.116s  ok ch=3.4e-14/1e-10 1s=2e-15 3.98x
   fftw3_custom                1708.697 us 874853.014 us     6.19       1.6%    0.000s  ok ch=3.1e-14/1e-10 1s=2e-15 4.09x
   baseline_matrix            34775.435 us 17805022.600 us     0.30       0.8%    0.000s  ok ch=8.1e-14/1e-10 1s=4e-15 83.14x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane               4089.562 us 261731.961 us    24.37       3.1%    0.007s  ok ch=2.3e-14/1e-10 1s=3e-15 1.00x
   gen_race                    4113.269 us 263249.203 us    24.23       2.1%    0.040s  ok ch=2.1e-14/1e-10 1s=3e-15 1.01x
   gen_powp                    4149.089 us 265541.687 us    24.02       3.4%    3.943s  ok ch=2.1e-14/1e-10 1s=3e-15 1.01x
   gen_pfa_large               4525.440 us 289628.169 us    22.02       1.0%    0.010s  ok ch=2.0e-14/1e-10 1s=3e-15 1.11x
   gen_planner                 4543.533 us 290786.081 us    21.93       0.7%    0.842s  ok ch=2.0e-14/1e-10 1s=3e-15 1.11x
   gen_pfa_small               5168.028 us 330753.780 us    19.28       1.7%    0.076s  ok ch=2.2e-14/1e-10 1s=3e-15 1.26x
   gen_twiddle                 6415.248 us 410575.877 us    15.53       1.4%    0.080s  ok ch=2.2e-14/1e-10 1s=3e-15 1.57x
   mkl_dfti                    7861.152 us 503113.737 us    12.68       0.6%    0.033s  ok ch=2.1e-14/1e-10 1s=3e-15 1.92x
   mkl2026_dfti                7914.538 us 506530.416 us    12.59       1.0%    0.048s  ok ch=2.2e-14/1e-10 1s=3e-15 1.94x
   gen_layout                  9168.396 us 586777.327 us    10.87       2.3%    0.010s  ok ch=2.1e-14/1e-10 1s=3e-15 2.24x
   fftw3_patient              10019.744 us 641263.629 us     9.95       2.3%   17.722s  ok ch=2.1e-14/1e-10 1s=3e-15 2.45x
   fftw3_measure              11315.759 us 724208.606 us     8.81       6.6%    0.319s  ok ch=2.1e-14/1e-10 1s=3e-15 2.77x
   ducc0_c2c                  11868.370 us 759575.661 us     8.40       0.7%    0.000s  ok ch=1.7e-14/1e-10 1s=2e-15 2.90x
   gen_bluestein              13710.427 us 877467.360 us     7.27       0.7%    0.000s  ok ch=2.8e-14/1e-10 1s=4e-15 3.35x
   fftw3_guru                 14510.111 us 928647.114 us     6.87       0.8%    0.371s  ok ch=2.0e-14/1e-10 1s=3e-15 3.55x
   fftw3_estimate             20819.001 us 1332416.050 us     4.79       0.2%    0.002s  ok ch=2.1e-14/1e-10 1s=3e-15 5.09x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_custom             genfft custom codelets (17/23/31), split arrays, scalar DAG + autovec
   fftw3_custom_soa         genfft custom codelets (17/23/31), SoA 8-volume batch-lane split-complex
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_guru               FFTW 3.3.10 guru split-array dft, FFTW_MEASURE, fused split chain
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA pencils (10=2x5,12=3x4,15=3x5,20=4x5; r6 adds 7-smooth 14/21/28/35 via a DFT7 module; r8 adds 11-smooth 22/33/44/55 via a DFT11 module; r7 LIFTED DFT5 v-pair -- sin72=phi*sin36 exact, 6 ops not 8, lit 08 6.3), register-explicit at 10/12 (2L ld + 2L st) and in the 15 zy-sweep (r7 hybrid), memory form elsewhere with r6 SAFE PLACEMENT (stage-1 store permutation makes every stage-2 group in-place on its own mod-P residue class), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at 12, vdivpd elsewhere), r9 FACTOR-SWAPPED map x-pencils at 10/15/20 (large factor stage 1 in place, small factor + map stage 2: kills the fused-map spills, -1..-2%), r10 extends the swap to the wide-module family 21/22/28/33/35/44/55 (-1.7..-4.7%; 14 keeps the unswapped order), r11 adds L=100 WITHIN-VOLUME SoA (lanes = 8 x-planes of one volume, B=1 native: shuffle-free zy sweeps fused per slab, trans8-bracketed x-pass, PFA 4x25 with DFT25=5x5 CT through an L1 scratch, 9 compiled-in w25 twiddle constants -- the file's first twiddles), r12 L=100 chain goes ONE-SWEEP FUSED (gen_pow2 r11's step-boundary x-split, CT 10x10 -- the involutive equal-radix case PFA provably cannot tile: z-in-lanes planes, vertical DFT10 stage-2+map head / twiddled stage-1 tail per 10-plane tile, parity-alternating tiling, tile-order c, 36 compiled-in w100 constants; DRAM crossings 2->1 per step, LLC loads -19%), r13 WITHIN-VOLUME B=1 engines at 10/12 (z-in-lanes rows, elementwise y/x passes, trans8-bracketed z-pass with fused map -- the r11 trick at small scale for the new B=1 cells; remainder groups route per-volume), sched-pressure on 10/12 only, THP arena (gen_layout), plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: {2,3,5,7}*2^k radix-4/16 DIF/DIT convolution (no bit-reversal; twiddle-free DFT-3/5/7 and PFA-6/10/14 fused middles cut M up to 37% below next_pow2; r13 sub-32 grid 20/24/28/40/56 via clamp-overlapped transpose blocks collapses L=9..14,17..20,25..28 to 3-pass chains), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized; masked contig pipeline for nv<8 tail groups), axis-2 transpose-scatter final layer as 256-bit half stores (gen_pow2 extract-to-memory, -8 p5/block), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody with the map fused into the axis-1 scatter reading a custody-ordered c (two aligned sequential streams; gen_pow2 GP2_CT), gen_twiddle exact tables
   gen_dense_prime          folded dense prime p<=31 (any prime in class supported): 4h^2-FMA conjugate-pair fold, z-pass fused into the x-contraction with the z-combine folded straight into U/V (no stack round-trip), fully in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, mask-free), register-tiled EXACT-TILE GEMM in 24-accumulator 6-column x 4-zmm groups (rotating broadcast register; drains every 360 FMAs, 0.417 loads/FMA, no spills) with exact 1..5-column k-tails and 1..3-zmm d-tails -- no wasted FMA slots at any L; L=31 z-phase GEMM likewise 24-acc (gen_r9: 3 mirrored pairs per group, split C/S loops, 0.417 loads/FMA); cross-plane software-pipelined y-pass raced and REJECTED in gen_r12 (+7%, port uops identical, l1d.replacement +33%: the y-pass hot set sits at the 48 KB L1D boundary and cross-plane concurrency thrashes it; default OFF behind -DGDP_YPIPE / -DGDP_YPIPE_S, cross-arch knobs); vectorized any-L z-pass, zmm z-row fold, LAZY map fused into the next step's z-loads at L=31 (only the last step's map materializes); PAIRED divides in the standalone map pass (gen_r10: one vdivpd per 16 points via rp=1/(dA*dB); the fused-site form lost and sits behind -DGDP_MAPZ_PAIR); 6-row 24-acc generic z-kernel selected per L at plan time (odd L >= 17, the ICX race verdicts); generic custody/lazy-map generalizations raced and rejected in gen_r6 (cross-arch knobs); gen_r13: the small scored composites (10/12/15/20) run COMPILE-TIME instantiations -- literal-shape fold cores, a row-quad z kernel with immediate masks, staggered joint scratch, plan-owned +2048-phase chain volumes at 12/20 -- bit-identical outputs, -38..-42% at the B=1 and batched 10/12 cells
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement + stream audit & measured pitch picker, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, r3: packed cross-plane axis-1 lanes + trailing axis-2 through a 4-plane collision-picked window; r4: fold-load software prefetch (~L row streams beat the L2 streamer), NT full-line stores on DRAM-resident volumes, fused transpose+interleave scatter; r5: graded map fused into the axis-2 exit (gl_map8/gl_map16 in-register map primitives), chain in place, zt volume deleted; r6: even-L second-level fold (k-parity split over j<->L/2-j, halves the kernel j-sweep at every even L); r7: third-level k-fold at 4|L (rows k and L/2-k share one j-sweep via the column sign (-1)^j, parity-sorted columns; halves the sweep again) + exit-map packing for kcnt 4/2 tail chunks; r8: insert-load 8x8 transpose in the axis-2 staging (VINSERTF64X4 from memory is a load-port uop: 16 shuffles vs 24, port-5/FMA relief), kcnt=1 exit-map packing (8 rows one ladder), unrolled constant-index full-chunk exit (spill diet); r9: 4-lane (ymm) SoA library family gl_deint4/int4/tr4x4/pack4/unpack4 + gl_map4 (PMU-audit avenue 4 / gen_planner G=4 ask: unlocks batch-lane layout at B=4 and B%8 remainder lanes; ymm FP dispatches on the otherwise-idle port 1), dogfooded in the kcnt=2 exit-map tail (zero-shuffle ymm ladders, bit-identical to gl_map8; plan-gated m4t = L<=GL_M4T_MAX, r10 default 0: the ICX verdict is the zmm packing wins on the scoring host -- SPR builds want -DGL_M4T_MAX=16); r10: counter-directed audit round -- PMU dashboard measured (25/32 run AT the node's ~2.1 uops/cycle dispatch cap, loads the largest port class; 50/100 traffic-bound), m4t ICX A/B banked as a deterministic plan gate + race-flippable knob; r11 (all hands on L=100): gl_thp_bytes smaps THP verification (the brief's layout ask; finding: THP=madvise on the scoring node leaves the driver's 32 MB buffers 4K-backed, kernel 5.15 has no MADV_COLLAPSE) + zero-copy chain-state re-home into the THP arena (last step exits to final_out directly; c staged once per call), gated by the measured smaps verdict per buffer pair; r12: T1 fold-source prefetch on DRAM-resident sources (LFB vs L2-superqueue; -1.3..2.2% at L=100) + differential-counter protocol; r13 (B=1 small-L round): pencil-lane kit for adopters -- gl_tr8x8_ld promoted (insert-load 8x8 transpose, load-port VINSERTF64X4), gl_pack8_ld (-33% port-5 entry pack, stride=L gives 8 z-pencils of one volume per lane group), gl_map8s/gl_map4s (graded map in split lane form, zero shuffles, bit-identical to gl_map8: a B=1 chain stays lane-resident for all m steps, packing once and unpacking once)
   gen_pfa_large            GT-PFA 25x4 two-sweep (exact-tw modules) + owned chain (deferred NR map); pick: l100-ipa1 (wisdom) (B=1)
   gen_pfa_small            PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path; r3: generic runtime-table coprime P*Q engine (modules 2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63; r4: register-explicit 10/12 pencils, in-place generic pencils where Q==1 mod P (14,18,21,36,56); r5: per-size map ladder BODY+tail (bl hs-form + rcp at 12, bl + div at 15/20, legacy + div at 10), raced same-core; r6: modules widened to {2,3,4,5,7,8,9,11,13,15,16,17,19,21,23,25,27,29,31} + nested-PFA composite odd modules (21,33,35,39,45,51,55,57,63) -- 53 new sizes, all coprime P*Q in 14..127 except 50/80/100 (pfa_large/powp cells); r7: B=1/B%8 chain rebuilt -- map fused into the stride-1 pass (map_span pass gone), half-turn z-pass (transpose-in only, swapped stores, parity-alternating c) vs dense stage-matrix broadcast-FMA pencil (lit 11 Tier 2) raced per size via -DSPLITZ<L>; r8: the rotation fused-map split chain ported to the GENERIC engine (site-buffer gather + the existing in-place gpen codelets, L >= 8) -- B%8 remainder volumes r <= GSPLIT_RMAX no longer lane-replicate; r9: phi-lifted DFT5 v-pair (6 ops vs 8; borrowed gen_batchlane r7 / lit 08 6.3) in D5CORE + M_DFT5 -- pencil FP 84/156/208 at 10/15/20 batched and in the split pencils; -DLIFT5=0 races back; r10: factor-swapped fused-map x-pencils at 10/15/20 (small factor in stage 2 where the map fuses; borrowed gen_batchlane r9 incl. their 12-loses verdict), -DSWAP<L> races back; r11 (all hands on L=100): CLAIMS 50=2x25 and 100=4x25 -- module 25 as twiddled 5x5 CT (borrowed gen_pfa_large r1's DFT25 shape; long-double tables; -DG25CT=0 races the fold) and a slab-FUSED split step for L>=GSLAB_MIN (passes 2+3 per L2-resident x-slab, natural-order map-fused stores, no rotation: one volume round trip per step gone vs r8); r12: WITHIN-VOLUME SoA chain for EVERY generic size at small batch (borrowed gen_batchlane r11, approach #4): lanes = 8 x-planes of one volume in the batched engine's site format -- shuffle-free zy sweeps per L2-resident slab, x-pass via tr8-bracketed 16 KB scratch pencil with the map fused at its stage-2 stores, c prepacked in consumption order, exact-zero pads; beats the r8 rotation form at every measured size (-16..-25%); at 50/100 the pencils are FORMULA-baked (pragma-unrolled GT/CRT index literals, no table loads) and 100 runs role-SWAPPED (gdft25 stage 1 map-free, map at the DFT4 stage-2 -- the r10 swap verdict; 50 keeps the map on the DFT25 side per gen_batchlane r10); -DGWVS=0 races the r11 slab form back, -DGWVS_MIN/-DGWVSSW/-DGWVSSW50/-DGWVSPF race per host; r13 (the new B=1 cells 10:1/12:1): tuned split chain rebuilt -- register-explicit fused pass 3 (tr8 feeds named DFT registers, map8c at the rotated stores; the r7 array form kept its lane buffers in MEMORY) + the whole m-loop in one function (-13.5%/-10.6% at 10/12 B=1, bit-identical) + p3tail partial transposes (2/4 columns by pair loads, not a full tr8); pass-3/pass-1 fusion built and REJECTED (+12/+13%: 2L live site registers across L map ladders spill); -DSPLITZ10/12 in {0,1,2,3,4} races all forms
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor (in-place, fused twiddles, register-resident fused CT/GT codelets, hard leaves 2/3/4/5/7/8) + volume-resident fused chain, any 2<=L<=128; alternating-layout one-transpose THP-re-homed chain at L>80 AND at L<=12 with a whole-volume exactly-paired map span (gen_r13, the B=1 small-L cells), raced against a within-volume 8-x-plane split-lane chain (slab-fused z+y sweeps, trans8-bracketed x-pass, map fused in the split stores; gen_r12, from gen_batchlane r11; gen_r13: small-L fused-codelet mode with a register-resident x-pass group); noise-gated create() race with banked per-host picks (@f custody playoff); adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64/128: TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies, dual-select FMA-folded twiddles; r8 adds the G=16 DRAM-regime engine; r9 re-forms z-codelet output slots via 256-bit extract-to-memory stores; r11 fuses the L=128 chain step into ONE tile-resident sweep (x-stage-2+map+z+y+x-stage-1, tile-order c): -61% DRAM reads, -14% wall; r12 fuses the next step's z-rows into stage-2's L1-hot row completion (-8% wall at 128) and flips the fused sweep ON at 64 (-11%); r13 fuses the custody conversions into the EXECUTE path's z-loads and x-stores (GP2_XFE: the benchFFT B=1 single-call lever, two custody-volume round trips deleted per call)), other 2^k in 2..8 generic radix-2
   gen_powp                 powp CT 3x9(3x3) exact tw, SoA-8 lane chain (DIF/DIT in place); pick: l27-soa (B=16)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race (interleaved sample-major since r4; NEW r9 NOISE GATE: an upset must clear max(jitter, 6%% floor) or win a fresh-evidence confirmation, reverted upsets never stored -- banked, deterministic picks per PMU audit avenue 1) + per-host wisdom cache incl. string wisdom + drop_prefix (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo = the ASSEMBLED trunk: pln_enumerate trees + split-group arms (@s1/2/3/4) + tile + fm (all THREE pv forms) + cf custody + p4 races, r11: the ALT one-transpose chain (planner r11, L>80) own-gated and raced per host, r12: planner's NEW within-volume split-lane chain (pln_wv, L>80 -- gen_batchlane's r11 L=100 design generalized) own-gated and raced vs the incumbent form, then the r8 CROSS-CLASS stage: class entries compiled at plan time as .so (cached per source-hash; r9 prefetches ALL on the first cold create so heavyweight compiles land before their cells), dlopened, gated, raced as whole-graded-chain arms vs self, winner ships by vtable forwarding (r11: gen_batchlane's within-volume engine routed at 50/100; r12: their BL_FUSE100 knob raced as a variant arm at 100, their record's ask; r13: the eng race chain length now tracks the graded amortization at tiny volumes -- m cap 64 -> 4096 -- so the new B=1 small-L cells route to the class B=1 engines on honest terms); salts chain13/tile13/chaingate13/fm13/cf13/p413/alt13/wv13/eng13
   gen_rader                Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); 43/67/79/103 via OUTER-C3 Rader (the same Winograd-C3-over-dense-blocks at runtime tables, 8m^2 vs 18m^2 conv FMA); even-h primes (p=1 mod 4, 13..113) via the E-side C2 CRT split + O-side negacyclic Karatsuba (5m^2 vs 8m^2); any other prime via the generic folded half-system engine; generic chain on a padded gl_map_huge arena for p<=61 (alias-free pitch, tail-free x/y), flat above; r7: blocked convs pinned rolled (front-end fix, -7..-21% at 61..113), dead map arms compiled out, x-pass stream prefetch at p>=107; r8: paired-column (2-wide) chunks share broadcast constants (dense engine -25.5% at 127; 2-wide+stord at m>=25), rolled fold/combine on the 1-wide kernels at m=15..24 (-1..-10% at 61/73/89); r9/r10: 2-wide RP3_WINO at p=103 CLOSED -- loses on BOTH graded architectures (SPR +35%, ICL +13%; rp3's ~1:2 glue:conv ratio); r10: rp2 pairing boundary fully raced, 2-wide+stord extended to m=24 (p=97, -4..6% on ICL); r11: unchanged (.text-identical) -- flat-chain memcpy deletion and gen_layout-r11 zero-copy THP re-home both raced and REJECTED at 127 (+0.6% / +4.1% at m=2: in-place chains pay no RFO, NT memcpy is the optimal prologue, TLB prize measured 0.6% of cycles); r12: 4-wide dense chunks with an E/O phase split at p>=59 (table walk once per 16 columns at the 2-wide 512b load ratio; -12.2% at 127, -9.6% at 107, -6..9% at 59..83 -- broadcast-uop deletion, port_2_3 -20%/step); r13: COMPILE-TIME tiny-prime execute engines (the benchFFT B=1 small-L round, applied to this class: MKL beat the runtime-table engines at every p <= 13 at B=1) -- p=13 rp2-m3 arithmetic with hardcoded Rader index tables (rp13_*, create()-verified vs rp2_build), p=3/5/7/11 via rp_chunk instantiated at literal p/h (rpd_*); strides, masks and trip counts all compile-time: -40..-57% (13: 8.4 -> 4.2 us B=1, 1.44x over MKL; 5/7 now ahead, 11 at MKL-4%); outputs bit-identical to r12 at p >= 17 and 31; self-check gated; s6 map from gen_dense_prime, arena from gen_layout
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp) + NEW dual-select FMA form tw_cis_ds (lit 11 Tier 1: every stored ratio <=1, first performant validation), consumption-order CT/DFT/Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo (gen_r5: conjugate-fold prime butterflies; gen_r6: register-resident whole-level codelets for r=2/3/4/5, -30..-44% bit-identical; gen_r7: fused fold-combine codelet for combine radices 7/11/13; gen_r8: shuffle-free split-group handoff axes 1->2, axis-2 gather deleted, bit-identical; gen_r9: axis-1 pencil groups PACKED across plane seams via two-pointer masked gather/scatter + masked w<8 tails, per-plane scalar z-tails deleted, bit-identical; gen_r10: radix-8 levels -- one whole combine pass deleted at 8|L -- + transpose last stage as extract-to-memory stores, -8 p5/tr8x8, from gen_pow2 r9; gen_r11: radix-10 levels via PFA 2x5 DFT10 codelets, zero internal twiddles -- depth-minimizing factorizer deletes one whole level pass at 10/30/50/70/90/100/110, tw muls 155->90 per group at 100; gen_r12: PFA 4x5 leaf20 + gated axis-0 L2 prefetch; gen_r13: PFA 4x3 leaf12 and 3x5 leaf15 -- whole combine level deleted at the new B=1 cells 12 [4,3]->[12] and at 15/36/45/48/75/84/96/105/120), self-audited at create(), owned in-place fused-map chain
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
