```
=== round panel_r9 ===
# round panel_r9
host: p55n3   date: 2026-08-22T04:36:43-04:00   slurm_job: 438526
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
cores: 64   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.207 us     0.207 us    40.50       0.0%    0.441s  ok 2.4e-16       1.00x
   L6_unrolled                    0.217 us     0.217 us    38.53       6.1%    0.575s  ok 2.4e-16       1.05x
   mkl_dfti                       0.373 us     0.373 us    22.44       1.4%    0.004s  ok 2.3e-16       1.81x
   mkl2026_dfti                   0.410 us     0.410 us    20.43       5.5%    0.005s  ok 2.5e-16       1.98x
   fftw3_patient                  0.514 us     0.514 us    16.29       3.0%    0.029s  ok 1.8e-16       2.49x
   fftw3_measure                  0.546 us     0.546 us    15.34       2.7%    0.017s  ok 1.8e-16       2.64x
   fftw3_estimate                 1.177 us     1.177 us     7.11       1.3%    0.003s  ok 1.8e-16       5.69x
   ducc0_c2c                      5.037 us     5.037 us     1.66       2.8%    0.000s  ok 1.7e-16       24.36x
   baseline_matrix                8.388 us     8.388 us     1.00       0.0%    0.000s  ok 5.8e-16       40.56x

-- L=6 (batched B=64), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.214 us    13.710 us    39.10       4.0%    0.583s  ok 2.4e-16       1.00x
   L6_pfa                         0.215 us    13.763 us    38.95       2.9%    0.452s  ok 2.4e-16       1.00x
   mkl_dfti                       0.391 us    25.017 us    21.43       1.6%    0.004s  ok 2.4e-16       1.82x
   mkl2026_dfti                   0.429 us    27.444 us    19.53       1.9%    0.004s  ok 2.5e-16       2.00x
   fftw3_patient                  0.542 us    34.682 us    15.46       1.0%    0.028s  ok 2.0e-16       2.53x
   fftw3_measure                  0.566 us    36.230 us    14.79       1.4%    0.016s  ok 2.0e-16       2.64x
   fftw3_estimate                 1.214 us    77.664 us     6.90       1.2%    0.002s  ok 2.0e-16       5.66x
   ducc0_c2c                      2.779 us   177.830 us     3.01       0.4%    0.000s  ok 1.8e-16       12.97x
   baseline_matrix                8.380 us   536.306 us     1.00       0.0%    0.000s  ok 6.0e-16       39.12x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.395 us  1619.551 us    21.18       2.6%    0.585s  ok 2.4e-16       1.00x
   L6_unrolled                    0.396 us  1620.920 us    21.16       1.6%    0.681s  ok 2.4e-16       1.00x
   mkl_dfti                       0.560 us  2294.746 us    14.95       0.1%    0.002s  ok 2.4e-16       1.42x
   mkl2026_dfti                   0.580 us  2375.044 us    14.44       1.2%    0.002s  ok 2.5e-16       1.47x
   fftw3_measure                  0.661 us  2706.733 us    12.67       1.0%    0.016s  ok 2.0e-16       1.67x
   fftw3_patient                  0.723 us  2960.722 us    11.59       2.1%    0.026s  ok 2.0e-16       1.83x
   fftw3_estimate                 1.277 us  5229.584 us     6.56       0.3%    0.001s  ok 2.0e-16       3.23x
   ducc0_c2c                      2.833 us 11604.398 us     2.96       0.9%    0.000s  ok 1.8e-16       7.17x
   baseline_matrix                8.530 us 34940.583 us     0.98       0.2%    0.000s  ok 6.0e-16       21.57x

-- L=6 (batched B=32768), volume 216, working set 216.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.566 us 18533.122 us    14.81       0.8%    1.988s  ok 2.4e-16       1.00x
   L6_pfa                         0.573 us 18786.085 us    14.61       0.1%    2.238s  ok 2.4e-16       1.01x
   mkl_dfti                       0.700 us 22940.560 us    11.96       0.8%    0.002s  ok 2.4e-16       1.24x
   mkl2026_dfti                   0.713 us 23367.648 us    11.74       0.8%    0.002s  ok 2.5e-16       1.26x
   fftw3_measure                  0.795 us 26062.806 us    10.53       0.8%    0.015s  ok 2.0e-16       1.41x
   fftw3_patient                  0.913 us 29929.446 us     9.17       0.9%    0.025s  ok 2.0e-16       1.61x
   fftw3_estimate                 1.357 us 44449.983 us     6.17       0.4%    0.001s  ok 2.0e-16       2.40x
   ducc0_c2c                      3.269 us 107126.751 us     2.56       0.3%    0.000s  ok 1.8e-16       5.78x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.553 us     0.553 us    41.69       1.1%    0.000s  ok 2.1e-16       1.00x
   L8_fusedaxes                   0.553 us     0.553 us    41.67       0.9%    0.920s  ok 2.1e-16       1.00x
   L8_radix8                      0.578 us     0.578 us    39.84       0.6%    0.044s  ok 1.3e-16       1.05x
   mkl_dfti                       0.651 us     0.651 us    35.39       4.1%    0.004s  ok 1.5e-16       1.18x
   mkl2026_dfti                   0.701 us     0.701 us    32.85       6.4%    0.004s  ok 1.5e-16       1.27x
   fftw3_patient                  1.148 us     1.148 us    20.08      13.8%    0.031s  ok 1.6e-16       2.08x
   fftw3_measure                  1.225 us     1.225 us    18.80       3.2%    0.019s  ok 1.6e-16       2.22x
   fftw3_estimate                 5.291 us     5.291 us     4.35       0.2%    0.003s  ok 1.6e-16       9.57x
   ducc0_c2c                      6.315 us     6.315 us     3.65       1.6%    0.000s  ok 1.3e-16       11.43x
   baseline_matrix               26.403 us    26.403 us     0.87       0.0%    0.000s  ok 3.8e-16       47.77x

-- L=8 (batched B=64), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.580 us    37.151 us    39.69       2.6%    0.938s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.594 us    38.003 us    38.80       8.1%    0.027s  ok 2.3e-16       1.02x
   L8_radix8                      0.597 us    38.214 us    38.59      11.3%    0.086s  ok 2.3e-16       1.03x
   mkl_dfti                       0.720 us    46.053 us    32.02      11.5%    0.005s  ok 1.6e-16       1.24x
   mkl2026_dfti                   0.784 us    50.162 us    29.40       1.5%    0.004s  ok 1.6e-16       1.35x
   fftw3_patient                  1.243 us    79.579 us    18.53       3.4%    0.028s  ok 1.8e-16       2.14x
   fftw3_measure                  1.276 us    81.680 us    18.05       1.1%    0.018s  ok 1.8e-16       2.20x
   ducc0_c2c                      3.994 us   255.643 us     5.77       0.6%    0.000s  ok 1.3e-16       6.88x
   fftw3_estimate                 5.427 us   347.357 us     4.25       0.1%    0.003s  ok 1.7e-16       9.35x
   baseline_matrix               26.423 us  1691.048 us     0.87       0.1%    0.000s  ok 3.9e-16       45.52x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.947 us  1940.151 us    24.32       1.0%    1.068s  ok 2.3e-16       1.00x
   L8_radix8                      0.977 us  2001.678 us    23.57       2.3%    0.192s  ok 1.9e-16       1.03x
   L8_batchsimd                   0.978 us  2003.415 us    23.55       1.7%    0.239s  ok 1.9e-16       1.03x
   mkl2026_dfti                   1.342 us  2749.440 us    17.16       0.6%    0.002s  ok 1.6e-16       1.42x
   mkl_dfti                       1.362 us  2788.490 us    16.92       0.6%    0.002s  ok 1.6e-16       1.44x
   fftw3_measure                  1.701 us  3483.902 us    13.54       2.7%    0.017s  ok 1.8e-16       1.80x
   fftw3_patient                  1.825 us  3737.467 us    12.63       1.8%    0.027s  ok 1.8e-16       1.93x
   ducc0_c2c                      4.539 us  9296.410 us     5.08       7.2%    0.000s  ok 1.3e-16       4.79x
   fftw3_estimate                 5.621 us 11512.807 us     4.10       0.2%    0.001s  ok 1.7e-16       5.93x
   baseline_matrix               26.746 us 54776.302 us     0.86       0.1%    0.000s  ok 3.9e-16       28.23x

-- L=8 (batched B=16384), volume 512, working set 256.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   1.251 us 20496.808 us    18.42       0.1%    1.397s  ok 2.3e-16       1.00x
   L8_batchsimd                   1.260 us 20642.389 us    18.29       0.3%    0.761s  ok 2.3e-16       1.01x
   L8_radix8                      1.269 us 20788.953 us    18.16       0.7%    0.636s  ok 2.3e-16       1.01x
   mkl2026_dfti                   1.791 us 29351.880 us    12.86       0.5%    0.002s  ok 1.6e-16       1.43x
   mkl_dfti                       1.844 us 30211.854 us    12.49       0.4%    0.002s  ok 1.6e-16       1.47x
   fftw3_measure                  2.046 us 33518.799 us    11.26       0.8%    0.016s  ok 1.8e-16       1.64x
   fftw3_patient                  2.172 us 35588.052 us    10.61       1.6%    0.027s  ok 1.8e-16       1.74x
   ducc0_c2c                      5.742 us 94082.023 us     4.01       1.3%    0.000s  ok 1.3e-16       4.59x
   fftw3_estimate                 5.791 us 94882.287 us     3.98       0.1%    0.001s  ok 1.7e-16       4.63x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.739 us     5.739 us    21.25       0.3%    0.000s  ok 2.8e-16       1.00x
   L13_rader                      6.030 us     6.030 us    20.22       2.0%    0.000s  ok 4.0e-16       1.05x
   mkl2026_dfti                   7.595 us     7.595 us    16.06       0.8%    0.004s  ok 3.2e-16       1.32x
   mkl_dfti                       7.767 us     7.767 us    15.70       1.9%    0.004s  ok 3.2e-16       1.35x
   fftw3_patient                  8.358 us     8.358 us    14.59       9.9%    0.032s  ok 3.2e-16       1.46x
   fftw3_measure                  8.659 us     8.659 us    14.08       2.3%    0.020s  ok 3.2e-16       1.51x
   fftw3_estimate                 8.969 us     8.969 us    13.60       2.3%    0.002s  ok 3.2e-16       1.56x
   ducc0_c2c                     43.974 us    43.974 us     2.77       5.2%    0.000s  ok 2.5e-16       7.66x
   baseline_matrix              184.488 us   184.488 us     0.66       0.0%    0.000s  ok 8.0e-16       32.15x

-- L=13 (batched B=16), volume 2197, working set 1.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.957 us    95.309 us    20.47       1.5%    0.000s  ok 2.9e-16       1.00x
   L13_rader                      6.963 us   111.408 us    17.51       1.2%    0.000s  ok 4.0e-16       1.17x
   mkl2026_dfti                   7.651 us   122.418 us    15.94       1.1%    0.004s  ok 3.3e-16       1.28x
   mkl_dfti                       7.712 us   123.394 us    15.81       3.6%    0.004s  ok 3.2e-16       1.29x
   fftw3_patient                  8.591 us   137.452 us    14.20       4.2%    0.032s  ok 3.2e-16       1.44x
   fftw3_measure                  8.848 us   141.562 us    13.78       4.6%    0.020s  ok 3.2e-16       1.49x
   fftw3_estimate                 9.004 us   144.059 us    13.54       1.9%    0.002s  ok 3.2e-16       1.51x
   ducc0_c2c                     41.500 us   663.997 us     2.94       0.7%    0.000s  ok 2.5e-16       6.97x
   baseline_matrix              184.634 us  2954.146 us     0.66       0.1%    0.000s  ok 7.9e-16       31.00x

-- L=13 (batched B=512), volume 2197, working set 34.33 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     7.965 us  4077.904 us    15.31       2.5%    0.000s  ok 2.9e-16       1.00x
   L13_rader                      9.055 us  4635.947 us    13.47       0.6%    0.000s  ok 4.0e-16       1.14x
   mkl2026_dfti                   9.107 us  4662.880 us    13.39       1.3%    0.002s  ok 3.2e-16       1.14x
   mkl_dfti                       9.362 us  4793.154 us    13.03       0.8%    0.002s  ok 3.2e-16       1.18x
   fftw3_estimate                10.505 us  5378.632 us    11.61       0.4%    0.001s  ok 3.2e-16       1.32x
   fftw3_measure                 10.563 us  5408.098 us    11.55       0.8%    0.016s  ok 3.2e-16       1.33x
   fftw3_patient                 10.610 us  5432.485 us    11.49       2.1%    0.028s  ok 3.2e-16       1.33x
   ducc0_c2c                     43.731 us 22390.204 us     2.79       2.1%    0.000s  ok 2.5e-16       5.49x
   baseline_matrix              186.258 us 95364.278 us     0.65       0.1%    0.000s  ok 7.9e-16       23.39x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                14.866 us    14.866 us    20.26       1.2%    0.201s  ok 3.2e-16       1.00x
   L17_winograd                  16.460 us    16.460 us    18.30       1.3%    0.489s  ok 3.2e-16       1.11x
   L17_rader                     16.543 us    16.543 us    18.21       0.3%    0.303s  ok 3.2e-16       1.11x
   fftw3_patient                 81.695 us    81.695 us     3.69       9.7%    0.031s  ok 3.1e-16       5.50x
   fftw3_estimate                81.741 us    81.741 us     3.69       5.9%    0.006s  ok 3.1e-16       5.50x
   fftw3_measure                 81.770 us    81.770 us     3.68       5.9%    0.018s  ok 3.1e-16       5.50x
   mkl_dfti                      98.790 us    98.790 us     3.05       7.0%    0.044s  ok 3.1e-16       6.65x
   mkl2026_dfti                 100.727 us   100.727 us     2.99       7.3%    0.039s  ok 3.1e-16       6.78x
   ducc0_c2c                    104.033 us   104.033 us     2.90       1.6%    0.000s  ok 2.6e-16       7.00x
   baseline_matrix              538.264 us   538.264 us     0.56       0.0%    0.000s  ok 8.4e-16       36.21x

-- L=17 (batched B=8), volume 4913, working set 1.20 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                16.512 us   132.099 us    18.24       2.8%    0.225s  ok 3.2e-16       1.00x
   L17_winograd                  17.508 us   140.067 us    17.20       2.6%    0.561s  ok 3.2e-16       1.06x
   L17_rader                     18.063 us   144.500 us    16.68       0.1%    0.317s  ok 3.1e-16       1.09x
   fftw3_measure                 81.949 us   655.592 us     3.68       6.0%    0.017s  ok 3.0e-16       4.96x
   fftw3_estimate                81.964 us   655.712 us     3.68       5.9%    0.004s  ok 3.0e-16       4.96x
   fftw3_patient                 81.982 us   655.858 us     3.67       8.8%    0.031s  ok 3.0e-16       4.96x
   mkl_dfti                     100.096 us   800.766 us     3.01       6.7%    0.037s  ok 3.1e-16       6.06x
   mkl2026_dfti                 100.866 us   806.931 us     2.99       7.0%    0.019s  ok 3.1e-16       6.11x
   ducc0_c2c                    101.073 us   808.583 us     2.98       3.1%    0.000s  ok 2.6e-16       6.12x
   baseline_matrix              538.565 us  4308.522 us     0.56       0.1%    0.000s  ok 8.4e-16       32.62x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                21.073 us  5394.620 us    14.29       0.7%    1.382s  ok 3.3e-16       1.00x
   L17_winograd                  21.512 us  5506.995 us    14.00       1.1%    3.234s  ok 3.3e-16       1.02x
   L17_rader                     24.610 us  6300.168 us    12.24       0.7%    1.837s  ok 3.2e-16       1.17x
   fftw3_estimate                83.447 us 21362.435 us     3.61       6.3%    0.002s  ok 3.0e-16       3.96x
   fftw3_measure                 83.453 us 21364.077 us     3.61       9.9%    0.009s  ok 3.0e-16       3.96x
   fftw3_patient                 83.504 us 21377.085 us     3.61       6.2%    0.020s  ok 3.0e-16       3.96x
   mkl_dfti                     102.798 us 26316.308 us     2.93       6.7%    0.060s  ok 3.1e-16       4.88x
   mkl2026_dfti                 103.066 us 26384.995 us     2.92       0.6%    0.016s  ok 3.1e-16       4.89x
   ducc0_c2c                    106.364 us 27229.171 us     2.83       5.2%    0.000s  ok 2.6e-16       5.05x
   baseline_matrix              540.464 us 138358.718 us     0.56       0.0%    0.000s  ok 8.4e-16       25.65x

-- L=17 (batched B=2048), volume 4913, working set 307.06 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                  21.716 us 44474.346 us    13.87       0.7%    4.757s  ok 3.3e-16       1.00x
   L17_matrixsimd                21.718 us 44478.058 us    13.87       0.9%    2.138s  ok 3.3e-16       1.00x
   L17_rader                     24.983 us 51164.540 us    12.06       1.0%    2.798s  ok 3.2e-16       1.15x
   fftw3_estimate                84.071 us 172177.914 us     3.58       6.1%    0.002s  ok 3.0e-16       3.87x
   fftw3_measure                 89.179 us 182639.497 us     3.38       0.1%    0.009s  ok 3.0e-16       4.11x
   fftw3_patient                 89.197 us 182675.324 us     3.38       1.1%    0.020s  ok 3.0e-16       4.11x
   mkl_dfti                     101.875 us 208640.747 us     2.96       6.4%    0.057s  ok 3.1e-16       4.69x
   mkl2026_dfti                 102.218 us 209342.616 us     2.95       6.9%    0.056s  ok 3.1e-16       4.71x
   ducc0_c2c                    115.026 us 235572.957 us     2.62       1.7%    0.000s  ok 2.6e-16       5.30x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     47.795 us    47.795 us    17.27       0.6%    0.563s  ok 3.8e-16       1.00x
   L23_matrixsimd                47.945 us    47.945 us    17.22       0.7%    0.468s  ok 3.8e-16       1.00x
   fftw3_estimate               261.097 us   261.097 us     3.16       7.1%    0.004s  ok 3.7e-16       5.46x
   fftw3_patient                261.118 us   261.118 us     3.16       7.1%    0.034s  ok 3.7e-16       5.46x
   fftw3_measure                268.901 us   268.901 us     3.07       4.1%    0.021s  ok 3.7e-16       5.63x
   ducc0_c2c                    291.671 us   291.671 us     2.83       2.0%    0.000s  ok 2.8e-16       6.10x
   mkl_dfti                     313.731 us   313.731 us     2.63       0.5%    0.042s  ok 4.2e-16       6.56x
   mkl2026_dfti                 342.013 us   342.013 us     2.41       0.1%    0.041s  ok 4.2e-16       7.16x
   baseline_matrix             1801.250 us  1801.250 us     0.46       0.0%    0.000s  ok 7.4e-16       37.69x

-- L=23 (batched B=4), volume 12167, working set 1.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     49.524 us   198.096 us    16.67       4.7%    0.590s  ok 3.8e-16       1.00x
   L23_matrixsimd                49.632 us   198.527 us    16.63       1.4%    0.485s  ok 3.8e-16       1.00x
   fftw3_estimate               261.549 us  1046.197 us     3.16       7.3%    0.006s  ok 3.7e-16       5.28x
   fftw3_patient                261.565 us  1046.260 us     3.16       7.2%    0.045s  ok 3.7e-16       5.28x
   fftw3_measure                268.522 us  1074.088 us     3.07       7.3%    0.023s  ok 3.7e-16       5.42x
   ducc0_c2c                    294.602 us  1178.409 us     2.80       1.1%    0.000s  ok 2.8e-16       5.95x
   mkl_dfti                     315.281 us  1261.122 us     2.62       3.5%    0.039s  ok 4.2e-16       6.37x
   mkl2026_dfti                 343.683 us  1374.730 us     2.40       0.0%    0.019s  ok 4.2e-16       6.94x
   baseline_matrix             1801.947 us  7207.789 us     0.46       0.0%    0.000s  ok 7.4e-16       36.39x

-- L=23 (batched B=128), volume 12167, working set 47.53 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     64.882 us  8304.887 us    12.72       0.1%    2.458s  ok 3.8e-16       1.00x
   L23_matrixsimd                65.112 us  8334.309 us    12.68       0.1%    2.087s  ok 3.8e-16       1.00x
   fftw3_patient                264.931 us 33911.110 us     3.12       7.4%    0.032s  ok 3.7e-16       4.08x
   fftw3_estimate               265.110 us 33934.070 us     3.11       7.2%    0.002s  ok 3.7e-16       4.09x
   fftw3_measure                265.395 us 33970.596 us     3.11       7.2%    0.012s  ok 3.7e-16       4.09x
   ducc0_c2c                    314.509 us 40257.110 us     2.62       7.1%    0.000s  ok 2.8e-16       4.85x
   mkl_dfti                     321.701 us 41177.676 us     2.57       3.5%    0.057s  ok 4.2e-16       4.96x
   mkl2026_dfti                 348.610 us 44622.061 us     2.37       0.3%    0.055s  ok 4.2e-16       5.37x
   baseline_matrix             1807.136 us 231313.356 us     0.46       0.0%    0.000s  ok 7.4e-16       27.85x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               120.478 us   120.478 us    30.03       1.9%    0.315s  ok 4.0e-16       1.00x
   L36_pfa                      122.576 us   122.576 us    29.52       6.9%    0.308s  ok 3.6e-16       1.02x
   L36_pencilfused              123.657 us   123.657 us    29.26       2.5%    0.180s  ok 3.8e-16       1.03x
   mkl_dfti                     160.994 us   160.994 us    22.47       1.0%    0.040s  ok 3.9e-16       1.34x
   mkl2026_dfti                 170.515 us   170.515 us    21.22       1.0%    0.062s  ok 4.0e-16       1.42x
   fftw3_patient                298.676 us   298.676 us    12.11       2.1%    1.140s  ok 3.9e-16       2.48x
   fftw3_measure                348.241 us   348.241 us    10.39       8.4%    0.154s  ok 3.5e-16       2.89x
   ducc0_c2c                    397.625 us   397.625 us     9.10       0.7%    0.000s  ok 3.0e-16       3.30x
   fftw3_estimate               450.712 us   450.712 us     8.03       2.2%    0.007s  ok 3.5e-16       3.74x
   baseline_matrix            10842.099 us 10842.099 us     0.33       0.0%    0.000s  ok 8.0e-16       89.99x

-- L=36 (batched B=4), volume 46656, working set 5.70 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               128.957 us   515.826 us    28.06       2.4%    0.296s  ok 3.9e-16       1.00x
   L36_pencilfused              130.993 us   523.972 us    27.62       0.8%    0.237s  ok 3.8e-16       1.02x
   L36_pfa                      131.953 us   527.811 us    27.42      39.5%    0.567s  ok 3.6e-16       1.02x
   mkl_dfti                     175.247 us   700.987 us    20.65       0.3%    0.039s  ok 3.9e-16       1.36x
   mkl2026_dfti                 183.749 us   734.998 us    19.69       1.8%    0.057s  ok 4.0e-16       1.42x
   fftw3_patient                310.223 us  1240.891 us    11.66       3.2%    1.147s  ok 3.6e-16       2.41x
   fftw3_measure                367.207 us  1468.829 us     9.85       2.0%    0.157s  ok 3.8e-16       2.85x
   ducc0_c2c                    430.080 us  1720.321 us     8.41       2.8%    0.000s  ok 3.1e-16       3.34x
   fftw3_estimate               452.935 us  1811.739 us     7.99       1.8%    0.010s  ok 3.5e-16       3.51x
   baseline_matrix            10867.422 us 43469.690 us     0.33       0.1%    0.000s  ok 8.0e-16       84.27x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      168.253 us  5384.089 us    21.50       0.8%    3.300s  ok 3.7e-16       1.00x
   L36_mixedradix               168.357 us  5387.424 us    21.49       1.0%    0.431s  ok 4.0e-16       1.00x
   L36_pencilfused              168.959 us  5406.678 us    21.41       2.0%    1.176s  ok 3.7e-16       1.00x
   mkl_dfti                     261.493 us  8367.785 us    13.84       1.6%    0.058s  ok 3.9e-16       1.55x
   mkl2026_dfti                 269.966 us  8638.922 us    13.40       0.6%    0.042s  ok 4.0e-16       1.60x
   fftw3_patient                384.910 us 12317.115 us     9.40       1.1%    1.128s  ok 3.9e-16       2.29x
   fftw3_measure                398.240 us 12743.664 us     9.09       1.4%    0.157s  ok 3.8e-16       2.37x
   fftw3_estimate               486.945 us 15582.247 us     7.43       1.6%    0.004s  ok 3.5e-16       2.89x
   ducc0_c2c                    586.695 us 18774.241 us     6.17       0.9%    0.000s  ok 3.0e-16       3.49x
   baseline_matrix            11081.244 us 354599.823 us     0.33       0.1%    0.000s  ok 8.0e-16       65.86x

-- L=36 (batched B=256), volume 46656, working set 364.50 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               184.140 us 47139.864 us    19.65       1.6%    0.539s  ok 4.0e-16       1.00x
   L36_pfa                      185.818 us 47569.368 us    19.47       0.2%    6.918s  ok 3.7e-16       1.01x
   L36_pencilfused              189.323 us 48466.762 us    19.11       0.4%    2.433s  ok 3.7e-16       1.03x
   mkl_dfti                     309.424 us 79212.554 us    11.69       0.6%    0.056s  ok 3.9e-16       1.68x
   mkl2026_dfti                 318.226 us 81465.862 us    11.37       1.2%    0.054s  ok 4.0e-16       1.73x
   fftw3_patient                387.920 us 99307.525 us     9.33       6.0%    1.124s  ok 3.9e-16       2.11x
   fftw3_measure                417.094 us 106775.951 us     8.67       1.8%    0.154s  ok 3.5e-16       2.27x
   fftw3_estimate               513.044 us 131339.359 us     7.05       0.7%    0.004s  ok 3.5e-16       2.79x
   ducc0_c2c                    719.131 us 184097.663 us     5.03       1.1%    0.000s  ok 3.0e-16       3.91x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      315.898 us   315.898 us    23.76       0.4%    0.399s  ok 4.0e-16       1.00x
   L45_mixedradix               317.518 us   317.518 us    23.64       1.4%    0.274s  ok 4.1e-16       1.01x
   mkl_dfti                     606.174 us   606.174 us    12.38       0.1%    0.038s  ok 4.4e-16       1.92x
   mkl2026_dfti                 623.702 us   623.702 us    12.04       0.6%    0.058s  ok 4.4e-16       1.97x
   ducc0_c2c                    867.616 us   867.616 us     8.65       1.0%    0.000s  ok 3.7e-16       2.75x
   fftw3_patient                867.806 us   867.806 us     8.65       8.3%    0.736s  ok 4.2e-16       2.75x
   fftw3_estimate               953.690 us   953.690 us     7.87       3.4%    0.006s  ok 4.2e-16       3.02x
   fftw3_measure               1117.986 us  1117.986 us     6.71       3.0%    0.081s  ok 4.2e-16       3.54x
   baseline_matrix            28828.356 us 28828.356 us     0.26      10.7%    0.000s  ok 8.0e-16       91.26x

-- L=45 (batched B=2), volume 91125, working set 5.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      324.441 us   648.882 us    23.14       1.7%    0.408s  ok 4.0e-16       1.00x
   L45_mixedradix               325.971 us   651.942 us    23.03       2.5%    0.497s  ok 4.1e-16       1.00x
   mkl_dfti                     607.217 us  1214.435 us    12.36       0.7%    0.042s  ok 4.4e-16       1.87x
   mkl2026_dfti                 623.851 us  1247.703 us    12.03       0.4%    0.038s  ok 4.5e-16       1.92x
   fftw3_patient                853.664 us  1707.328 us     8.79      13.5%    0.715s  ok 4.3e-16       2.63x
   ducc0_c2c                    898.129 us  1796.257 us     8.36       1.4%    0.000s  ok 3.7e-16       2.77x
   fftw3_estimate               952.198 us  1904.396 us     7.88       2.8%    0.006s  ok 4.2e-16       2.93x
   fftw3_measure               1149.459 us  2298.918 us     6.53       2.4%    0.082s  ok 4.2e-16       3.54x
   baseline_matrix            28503.719 us 57007.439 us     0.26      11.8%    0.000s  ok 8.0e-16       87.85x

-- L=45 (batched B=16), volume 91125, working set 44.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      402.154 us  6434.457 us    18.67       3.1%    0.919s  ok 4.0e-16       1.00x
   L45_mixedradix               417.913 us  6686.613 us    17.96       1.3%    0.283s  ok 4.1e-16       1.04x
   mkl_dfti                     753.336 us 12053.370 us     9.96       0.3%    0.035s  ok 4.4e-16       1.87x
   mkl2026_dfti                 764.856 us 12237.702 us     9.81       0.9%    0.038s  ok 4.5e-16       1.90x
   fftw3_patient                990.911 us 15854.571 us     7.58       3.1%    0.729s  ok 4.3e-16       2.46x
   fftw3_estimate              1028.343 us 16453.495 us     7.30       3.2%    0.002s  ok 4.2e-16       2.56x
   fftw3_measure               1176.399 us 18822.388 us     6.38      11.4%    0.078s  ok 4.2e-16       2.93x
   ducc0_c2c                   1211.991 us 19391.862 us     6.19       1.2%    0.000s  ok 3.7e-16       3.01x
   baseline_matrix            33041.040 us 528656.633 us     0.23       0.6%    0.000s  ok 8.0e-16       82.16x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                   952.944 us   952.944 us    24.76       0.8%    0.185s  ok 4.5e-16       1.00x
   L64_blocked                 1098.376 us  1098.376 us    21.48       1.3%    0.488s  ok 4.2e-16       1.15x
   mkl_dfti                    1192.045 us  1192.045 us    19.79       0.6%    0.038s  ok 3.4e-16       1.25x
   mkl2026_dfti                1331.722 us  1331.722 us    17.72       0.5%    0.037s  ok 3.4e-16       1.40x
   fftw3_measure               1804.134 us  1804.134 us    13.08       3.4%    0.308s  ok 3.5e-16       1.89x
   fftw3_patient               1811.661 us  1811.661 us    13.02      12.8%    4.776s  ok 3.5e-16       1.90x
   ducc0_c2c                   2349.993 us  2349.993 us    10.04       1.4%    0.000s  ok 3.0e-16       2.47x
   fftw3_estimate              3330.390 us  3330.390 us     7.08       0.5%    0.002s  ok 3.5e-16       3.49x
   baseline_matrix           123572.400 us 123572.400 us     0.19       1.1%    0.000s  ok 7.8e-16       129.67x

-- L=64 (batched B=2), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                  1021.050 us  2042.099 us    23.11       2.0%    0.304s  ok 4.5e-16       1.00x
   L64_blocked                 1179.955 us  2359.910 us    19.99       2.5%    0.645s  ok 4.2e-16       1.16x
   mkl_dfti                    1265.228 us  2530.457 us    18.65       2.2%    0.037s  ok 3.4e-16       1.24x
   mkl2026_dfti                1434.934 us  2869.869 us    16.44       0.5%    0.056s  ok 3.4e-16       1.41x
   fftw3_measure               1792.796 us  3585.592 us    13.16       2.2%    0.305s  ok 3.5e-16       1.76x
   fftw3_patient               1869.559 us  3739.117 us    12.62       0.9%    4.779s  ok 3.6e-16       1.83x
   ducc0_c2c                   2413.487 us  4826.974 us     9.78       0.8%    0.000s  ok 3.0e-16       2.36x
   fftw3_estimate              3324.076 us  6648.151 us     7.10       0.3%    0.001s  ok 3.5e-16       3.26x
   baseline_matrix           125818.530 us 251637.059 us     0.19      11.7%    0.000s  ok 7.8e-16       123.22x

-- L=64 (batched B=8), volume 262144, working set 64.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                  1249.923 us  9999.387 us    18.88       1.4%    0.658s  ok 4.5e-16       1.00x
   L64_blocked                 1304.242 us 10433.936 us    18.09       1.7%    2.551s  ok 4.2e-16       1.04x
   fftw3_patient               2405.747 us 19245.980 us     9.81      11.3%    5.365s  ok 3.5e-16       1.92x
   mkl_dfti                    2516.135 us 20129.078 us     9.38       0.4%    0.057s  ok 3.4e-16       2.01x
   mkl2026_dfti                2667.558 us 21340.460 us     8.84       0.6%    0.016s  ok 3.4e-16       2.13x
   fftw3_measure               2882.394 us 23059.154 us     8.19       3.0%    0.304s  ok 3.6e-16       2.31x
   ducc0_c2c                   3481.242 us 27849.932 us     6.78       1.8%    0.000s  ok 3.0e-16       2.79x
   fftw3_estimate              4395.831 us 35166.645 us     5.37       0.1%    0.001s  ok 3.5e-16       3.52x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned+ymm-tail X-last
   L13_rader                Rader-13 as split cyclic/negacyclic correlations (186 FP/pt), X-first, global-row z-kernel port-fused with y lane-blocks, direct+prefetchw out, 512-bit
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, deferred-Z, addr-safe t1, pf=0, pw=0, pt=0, b1dec[yz/kyz/x/kx]=10.98/10.33/4.07/4.08, clk512/256=2.89/3.88 GHz, d256=3.89
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t dy, pf=0, pfw=0, clk256=3.89 clk512=2.89, probe ph/xp/fu=16.37/6.86/23.74 us/vol
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, clk256=2.89GHz, clk512=2.89GHz, p1=6.17 f23=10.26 fu=16.50
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pf=0, pw=0, clk512/256=2.89/3.89 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, tuner pick=49.69 inc=49.69 us/t nv=4
   L36_mixedradix           PFA 4x9 2-sweep, lanes=lines; pick=v1-cached-pf1-pfin-pfw (B=32, arena=32 vol, stream=1, 6 cand, pinD=-1)
   L36_pencilfused          L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes; tuner picked pw=4 mode=istream+pfw (B=32)
   L36_pfa                  GT-PFA 4x9 two-sweep; tuner pick: pw=4 mode=inplace pf=2 (B=256, nv=64); probe us p1=93.0 p2w=34.4 fu=123.5 fug=123.9; fe=na
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-pf0 (B=2, arena=2, stream=0, 4 cand); nv1 us fu=322.8 p1=246.3 p2w=80.8; fe=na
   L45_pfa                  GT-PFA 9x5 two-sweep; tuner pick: pw4-ip-pf3 (B=16, nv=16)
   L64_blocked              L64 8x8 two-stage, hugepage odd-line-padded scratch; tuner pick: pw=4 mode=cached pf=2 st=0(3-sweep) (B=8, nv=8)
   L64_radix8               radix-8^2 per axis, split-complex AVX-512, padded scratch; tuner pick[B=2]=fused-pfw+slabpf1+pro0+p10
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, plan-raced; variant=fused_pfw_xa clkS256=3.89 clkD256=2.89 clkS512=2.89 kclk=2.89GHz bf=223.0 bsp=221.3 bx=64.5 byz=156.8ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), ymm raced, zmm forced-only; variant=fused_pf clk256=3.89,3.89,3.89 clk512=2.89,2.89 kclk=2.89GHz ab1=f221.4,f3228.5,zf251.9ns f3d=-1.7%
   L8_batchsimd             radix-8 split; pick[B=16384]: mode=FUSED nt=0 pf=s0w
   L8_fusedaxes             8^3 fused/seq3/AA c52; B=2048 pick=fused+pfs+pfw (tuned) clk256s/p=3.26/2.42 clk512s/p=2.43/2.42 pmc=na
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p shapes; pick[B=1]=avx512-1f (tuned) arena{2p=0.603 1f=0.572 3p=0.730} scr@0x540
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
