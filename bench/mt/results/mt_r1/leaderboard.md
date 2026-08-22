```
=== round mt_r1 ===
# round mt_r1
host: p51n1   date: 2026-08-22T10:15:32-04:00   slurm_job: 438534
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
threads: 32 of 32 (PROC_BIND=close)   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.210 us     0.210 us    39.88       4.5%    0.507s  ok 2.3e-16       1.00x
   L6_unrolled                    0.221 us     0.221 us    37.87       0.0%    0.300s  ok 2.3e-16       1.05x
   mkl_dfti                       0.371 us     0.371 us    22.56       1.2%    0.037s  ok 2.3e-16       1.77x
   mkl2026_dfti                   0.412 us     0.412 us    20.33       8.9%    0.035s  ok 2.6e-16       1.96x
   fftw3_patient                  0.507 us     0.507 us    16.53       2.5%    0.150s  ok 1.8e-16       2.41x
   ducc0_c2c                      4.794 us     4.794 us     1.75       7.5%    0.000s  ok 1.8e-16       22.83x
   baseline_matrix                8.387 us     8.387 us     1.00       0.0%    0.000s  ok 5.9e-16       39.94x
   fftw3_estimate                 9.829 us     9.829 us     0.85       4.1%    0.005s  ok 1.8e-16       46.80x
   fftw3_measure                 22.380 us    22.380 us     0.37       1.9%    0.038s  ok 1.8e-16       106.57x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.009 us    38.593 us   888.89      52.9%    1.213s  ok 2.4e-16       1.00x
   L6_unrolled                    0.012 us    47.674 us   719.58       3.6%    0.455s  ok 2.4e-16       1.24x
   mkl_dfti                       0.021 us    86.445 us   396.84       1.5%    0.032s  ok 2.4e-16       2.24x
   mkl2026_dfti                   0.022 us    91.839 us   373.54       1.9%    0.004s  ok 2.5e-16       2.38x
   fftw3_patient                  0.025 us   102.931 us   333.28       0.7%    0.199s  ok 2.0e-16       2.67x
   fftw3_measure                  0.025 us   103.458 us   331.59       3.0%    0.017s  ok 2.0e-16       2.68x
   fftw3_estimate                 0.055 us   226.945 us   151.16       0.1%    0.003s  ok 2.0e-16       5.88x
   ducc0_c2c                      0.275 us  1125.440 us    30.48       1.5%    0.000s  ok 1.8e-16       29.16x
   baseline_matrix                8.500 us 34814.762 us     0.99       0.2%    0.000s  ok 6.0e-16       902.09x

-- L=6 (batched B=65536), volume 216, working set 432.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.039 us  2586.159 us   212.24      19.9%   12.617s  ok 2.4e-16       1.00x
   L6_unrolled                    0.072 us  4724.361 us   116.18       0.2%    0.774s  ok 2.4e-16       1.83x
   fftw3_estimate                 0.121 us  7950.799 us    69.03       0.5%    0.002s  ok 2.0e-16       3.07x
   fftw3_patient                  0.122 us  8006.269 us    68.56       0.5%    0.880s  ok 2.0e-16       3.10x
   mkl_dfti                       0.122 us  8017.062 us    68.46       0.3%    0.033s  ok 2.4e-16       3.10x
   fftw3_measure                  0.122 us  8018.085 us    68.46       0.1%    0.017s  ok 2.0e-16       3.10x
   mkl2026_dfti                   0.122 us  8025.717 us    68.39       0.1%    0.022s  ok 2.5e-16       3.10x
   ducc0_c2c                      0.271 us 17742.045 us    30.94       1.3%    0.000s  ok 1.8e-16       6.86x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.554 us     0.554 us    41.62       0.8%    0.077s  ok 2.2e-16       1.00x
   L8_batchsimd                   0.554 us     0.554 us    41.61       1.8%    0.086s  ok 2.2e-16       1.00x
   L8_radix8                      0.581 us     0.581 us    39.63       0.0%    0.056s  ok 1.2e-16       1.05x
   mkl_dfti                       0.658 us     0.658 us    35.01       1.2%    0.037s  ok 1.6e-16       1.19x
   mkl2026_dfti                   0.739 us     0.739 us    31.18       0.6%    0.025s  ok 1.6e-16       1.33x
   fftw3_patient                  1.217 us     1.217 us    18.93       4.6%    0.151s  ok 1.8e-16       2.20x
   ducc0_c2c                      6.241 us     6.241 us     3.69       0.4%    0.000s  ok 1.2e-16       11.27x
   fftw3_estimate                14.135 us    14.135 us     1.63       0.8%    0.004s  ok 1.7e-16       25.54x
   fftw3_measure                 20.403 us    20.403 us     1.13       6.6%    0.041s  ok 1.7e-16       36.86x
   baseline_matrix               26.401 us    26.401 us     0.87       0.0%    0.000s  ok 3.9e-16       47.69x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.026 us    53.721 us   878.35       0.7%    0.176s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.028 us    56.504 us   835.09       0.4%    0.119s  ok 2.3e-16       1.05x
   L8_radix8                      0.035 us    70.975 us   664.82       1.0%    0.141s  ok 1.9e-16       1.32x
   mkl_dfti                       0.037 us    75.205 us   627.43       1.0%    0.032s  ok 1.6e-16       1.40x
   mkl2026_dfti                   0.040 us    81.467 us   579.20       2.3%    0.004s  ok 1.6e-16       1.52x
   fftw3_patient                  0.056 us   114.812 us   410.98       1.4%    0.214s  ok 1.7e-16       2.14x
   fftw3_measure                  0.056 us   115.297 us   409.25       1.0%    0.019s  ok 1.8e-16       2.15x
   fftw3_estimate                 0.223 us   456.551 us   103.35       0.1%    0.003s  ok 1.7e-16       8.50x
   ducc0_c2c                      0.376 us   769.051 us    61.36       2.2%    0.000s  ok 1.3e-16       14.32x
   baseline_matrix               26.707 us 54695.814 us     0.86       0.2%    0.000s  ok 3.9e-16       1018.14x

-- L=8 (batched B=32768), volume 512, working set 512.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.161 us  5260.574 us   143.52       0.3%    1.030s  ok 1.7e-16       1.00x
   L8_radix8                      0.173 us  5654.237 us   133.52      71.1%    0.242s  ok 1.9e-16       1.07x
   L8_fusedaxes                   0.176 us  5779.072 us   130.64       0.3%    0.437s  ok 2.3e-16       1.10x
   fftw3_estimate                 0.293 us  9595.821 us    78.68       0.2%    0.002s  ok 1.7e-16       1.82x
   fftw3_measure                  0.295 us  9663.651 us    78.13       0.1%    0.018s  ok 1.8e-16       1.84x
   mkl2026_dfti                   0.295 us  9668.523 us    78.09       0.2%    0.023s  ok 1.6e-16       1.84x
   L8_batchsimd                   0.295 us  9674.299 us    78.04       0.3%    0.206s  ok 2.3e-16       1.84x
   mkl_dfti                       0.295 us  9674.983 us    78.03       0.4%    0.035s  ok 1.6e-16       1.84x
   ducc0_c2c                      0.546 us 17897.965 us    42.18       2.5%    0.000s  ok 1.3e-16       3.40x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.670 us     5.670 us    21.51       1.2%    0.052s  ok 2.8e-16       1.00x
   L13_rader                      5.784 us     5.784 us    21.08      12.4%    0.016s  ok 4.0e-16       1.02x
   mkl2026_dfti                   7.632 us     7.632 us    15.98       0.3%    0.028s  ok 3.2e-16       1.35x
   mkl_dfti                       7.828 us     7.828 us    15.58       0.6%    0.017s  ok 3.2e-16       1.38x
   fftw3_patient                  8.289 us     8.289 us    14.71       3.8%    0.262s  ok 3.2e-16       1.46x
   fftw3_measure                 33.544 us    33.544 us     3.64      17.8%    0.059s  ok 3.2e-16       5.92x
   fftw3_estimate                36.699 us    36.699 us     3.32       3.3%    0.007s  ok 3.2e-16       6.47x
   ducc0_c2c                     44.060 us    44.060 us     2.77       0.6%    0.000s  ok 2.5e-16       7.77x
   baseline_matrix              184.480 us   184.480 us     0.66       0.0%    0.000s  ok 7.8e-16       32.53x

-- L=13 (batched B=512), volume 2197, working set 34.33 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     0.308 us   157.602 us   396.17       1.1%    0.056s  ok 2.9e-16       1.00x
   L13_rader                      0.317 us   162.545 us   384.12       4.0%    0.048s  ok 4.0e-16       1.03x
   mkl2026_dfti                   0.360 us   184.146 us   339.06       0.6%    0.004s  ok 3.2e-16       1.17x
   fftw3_patient                  0.368 us   188.521 us   331.20       0.9%    0.335s  ok 3.2e-16       1.20x
   mkl_dfti                       0.370 us   189.460 us   329.55       1.0%    0.034s  ok 3.2e-16       1.20x
   fftw3_measure                  0.375 us   192.155 us   324.93       1.0%    0.018s  ok 3.2e-16       1.22x
   fftw3_estimate                 0.426 us   218.109 us   286.27      18.5%    0.003s  ok 3.2e-16       1.38x
   ducc0_c2c                      3.396 us  1738.719 us    35.91       4.4%    0.000s  ok 2.5e-16       11.03x
   baseline_matrix              186.402 us 95437.743 us     0.65       0.2%    0.000s  ok 7.9e-16       605.56x

-- L=13 (batched B=8192), volume 2197, working set 549.25 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.787 us  6450.267 us   154.88       0.7%    1.424s  ok 3.2e-16       1.00x
   L13_rader                      0.976 us  7997.508 us   124.91       0.1%    0.705s  ok 4.0e-16       1.24x
   mkl_dfti                       1.264 us 10353.892 us    96.49       0.3%    0.034s  ok 3.2e-16       1.61x
   mkl2026_dfti                   1.264 us 10354.714 us    96.48       0.4%    0.026s  ok 3.2e-16       1.61x
   fftw3_estimate                 1.267 us 10377.581 us    96.27       1.5%    0.002s  ok 3.2e-16       1.61x
   fftw3_measure                  1.268 us 10391.194 us    96.14       0.2%    0.017s  ok 3.2e-16       1.61x
   L13_direct                     1.275 us 10443.760 us    95.66       0.4%    0.057s  ok 2.9e-16       1.62x
   ducc0_c2c                      3.365 us 27565.747 us    36.24       0.5%    0.000s  ok 2.5e-16       4.27x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 7.112 us     7.112 us    42.36       3.6%    0.347s  ok 3.3e-16       1.00x
   L17_winograd                   8.998 us     8.998 us    33.48       1.7%    0.591s  ok 3.3e-16       1.27x
   L17_rader                      9.134 us     9.134 us    32.98      38.3%    0.486s  ok 3.2e-16       1.28x
   mkl_dfti                      22.820 us    22.820 us    13.20       1.4%    0.019s  ok 3.1e-16       3.21x
   mkl2026_dfti                  23.682 us    23.682 us    12.72       1.8%    0.003s  ok 3.1e-16       3.33x
   fftw3_patient                 30.112 us    30.112 us    10.00       4.4%    0.218s  ok 3.0e-16       4.23x
   fftw3_estimate                38.763 us    38.763 us     7.77      30.9%    0.009s  ok 3.0e-16       5.45x
   fftw3_measure                 43.195 us    43.195 us     6.97      10.6%    0.052s  ok 3.0e-16       6.07x
   ducc0_c2c                    103.688 us   103.688 us     2.91       0.2%    0.000s  ok 2.6e-16       14.58x
   baseline_matrix              538.314 us   538.314 us     0.56       0.0%    0.000s  ok 8.4e-16       75.69x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 0.726 us   185.851 us   414.92      12.2%    0.373s  ok 3.3e-16       1.00x
   L17_rader                      0.795 us   203.466 us   379.00       1.4%    0.283s  ok 3.2e-16       1.09x
   L17_winograd                   0.824 us   210.994 us   365.48       0.9%    2.190s  ok 3.3e-16       1.14x
   fftw3_measure                  3.893 us   996.597 us    77.38       0.5%    0.012s  ok 3.0e-16       5.36x
   fftw3_estimate                 3.899 us   998.148 us    77.26       0.4%    0.004s  ok 3.0e-16       5.37x
   fftw3_patient                  3.901 us   998.775 us    77.21       0.7%    0.323s  ok 3.0e-16       5.37x
   mkl2026_dfti                   4.836 us  1237.889 us    62.29       3.6%    0.027s  ok 3.1e-16       6.66x
   mkl_dfti                       4.854 us  1242.593 us    62.06       1.1%    0.035s  ok 3.1e-16       6.69x
   ducc0_c2c                      8.181 us  2094.372 us    36.82       6.0%    0.000s  ok 2.6e-16       11.27x
   baseline_matrix              539.906 us 138215.820 us     0.56       0.1%    0.000s  ok 8.4e-16       743.69x

-- L=17 (batched B=4096), volume 4913, working set 614.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                   1.222 us  5004.651 us   246.53       0.3%    2.901s  ok 3.3e-16       1.00x
   L17_matrixsimd                 2.106 us  8627.951 us   143.00       9.8%    0.745s  ok 3.3e-16       1.72x
   L17_rader                      2.904 us 11893.028 us   103.74       1.9%    0.507s  ok 3.2e-16       2.38x
   fftw3_patient                  3.925 us 16078.405 us    76.74      76.3%    1.851s  ok 3.0e-16       3.21x
   fftw3_measure                  4.051 us 16594.160 us    74.35       1.1%    0.011s  ok 3.0e-16       3.32x
   fftw3_estimate                 4.089 us 16749.195 us    73.66      11.0%    0.004s  ok 3.0e-16       3.35x
   mkl_dfti                       4.942 us 20240.430 us    60.96       1.1%    0.036s  ok 3.1e-16       4.04x
   mkl2026_dfti                   4.961 us 20321.926 us    60.71       1.4%    0.025s  ok 3.1e-16       4.06x
   ducc0_c2c                      7.853 us 32167.932 us    38.36       1.7%    0.000s  ok 2.6e-16       6.43x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                11.835 us    11.835 us    69.76       0.8%    0.185s  ok 3.8e-16       1.00x
   L23_rader                     14.150 us    14.150 us    58.34       0.8%    0.107s  ok 3.8e-16       1.20x
   fftw3_patient                 39.332 us    39.332 us    20.99       1.1%    0.243s  ok 3.7e-16       3.32x
   mkl_dfti                      42.277 us    42.277 us    19.53       3.3%    0.035s  ok 4.2e-16       3.57x
   mkl2026_dfti                  43.621 us    43.621 us    18.93       1.1%    0.009s  ok 4.2e-16       3.69x
   fftw3_estimate                46.779 us    46.779 us    17.65      42.8%    0.010s  ok 3.7e-16       3.95x
   fftw3_measure                 48.169 us    48.169 us    17.14      62.4%    0.051s  ok 3.7e-16       4.07x
   ducc0_c2c                    293.525 us   293.525 us     2.81       2.1%    0.000s  ok 2.8e-16       24.80x
   baseline_matrix             1801.218 us  1801.218 us     0.46       0.0%    0.000s  ok 7.3e-16       152.20x

-- L=23 (batched B=128), volume 12167, working set 47.53 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                 2.287 us   292.697 us   361.03       6.7%    0.108s  ok 3.8e-16       1.00x
   L23_rader                      2.350 us   300.810 us   351.30       6.8%    0.187s  ok 3.8e-16       1.03x
   fftw3_patient                 12.437 us  1591.954 us    66.38       1.5%    0.536s  ok 3.7e-16       5.44x
   fftw3_estimate                12.439 us  1592.231 us    66.37       0.1%    0.004s  ok 3.7e-16       5.44x
   fftw3_measure                 12.757 us  1632.923 us    64.71      51.1%    0.013s  ok 3.7e-16       5.58x
   mkl_dfti                      14.571 us  1865.127 us    56.66       0.8%    0.016s  ok 4.2e-16       6.37x
   mkl2026_dfti                  15.765 us  2017.888 us    52.37       1.0%    0.023s  ok 4.2e-16       6.89x
   ducc0_c2c                     23.548 us  3014.171 us    35.06       1.3%    0.000s  ok 2.8e-16       10.30x
   baseline_matrix             1806.331 us 231210.357 us     0.46       0.0%    0.000s  ok 7.4e-16       789.93x

-- L=23 (batched B=2048), volume 12167, working set 760.44 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      6.052 us 12393.893 us   136.42      18.8%    0.210s  ok 3.8e-16       1.00x
   L23_matrixsimd                 7.235 us 14817.455 us   114.11       0.1%    0.123s  ok 3.8e-16       1.20x
   fftw3_patient                 12.466 us 25531.063 us    66.22       0.6%    1.428s  ok 3.7e-16       2.06x
   fftw3_measure                 12.637 us 25879.559 us    65.33       1.2%    0.012s  ok 3.7e-16       2.09x
   fftw3_estimate                12.729 us 26069.206 us    64.86       0.2%    0.004s  ok 3.7e-16       2.10x
   mkl_dfti                      14.718 us 30141.641 us    56.09       0.8%    0.034s  ok 4.2e-16       2.43x
   mkl2026_dfti                  15.905 us 32573.054 us    51.91       0.5%    0.026s  ok 4.2e-16       2.63x
   ducc0_c2c                     24.541 us 50260.790 us    33.64       0.8%    0.000s  ok 2.8e-16       4.06x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                       25.659 us    25.659 us   141.01       0.6%    0.067s  ok 3.6e-16       1.00x
   L36_pencilfused               28.507 us    28.507 us   126.92       3.4%    0.218s  ok 3.6e-16       1.11x
   L36_mixedradix                28.947 us    28.947 us   124.99       1.2%    0.234s  ok 3.6e-16       1.13x
   mkl_dfti                      38.156 us    38.156 us    94.82       0.7%    0.014s  ok 3.9e-16       1.49x
   mkl2026_dfti                  38.287 us    38.287 us    94.50       0.5%    0.026s  ok 4.0e-16       1.49x
   fftw3_measure                 65.680 us    65.680 us    55.09      57.9%    0.277s  ok 3.8e-16       2.56x
   fftw3_patient                 70.811 us    70.811 us    51.10      31.4%   10.054s  ok 3.6e-16       2.76x
   fftw3_estimate                89.964 us    89.964 us    40.22       8.0%    0.025s  ok 3.5e-16       3.51x
   ducc0_c2c                    200.096 us   200.096 us    18.08      13.9%    0.000s  ok 3.0e-16       7.80x
   baseline_matrix            10848.195 us 10848.195 us     0.33       0.0%    0.000s  ok 8.0e-16       422.78x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                 5.589 us   178.857 us   647.33       0.6%    0.610s  ok 3.6e-16       1.00x
   L36_pencilfused                5.644 us   180.610 us   641.05       2.2%    0.146s  ok 3.6e-16       1.01x
   L36_pfa                        5.912 us   189.180 us   612.01       0.6%    0.370s  ok 3.6e-16       1.06x
   mkl_dfti                       7.510 us   240.319 us   481.78       0.8%    0.033s  ok 3.9e-16       1.34x
   mkl2026_dfti                   7.853 us   251.298 us   460.73       0.3%    0.024s  ok 4.0e-16       1.41x
   fftw3_patient                 27.578 us   882.484 us   131.20      55.7%   11.131s  ok 3.8e-16       4.93x
   ducc0_c2c                     38.893 us  1244.563 us    93.03       2.5%    0.000s  ok 3.0e-16       6.96x
   fftw3_measure                 50.476 us  1615.245 us    71.68      52.1%    0.161s  ok 3.5e-16       9.03x
   fftw3_estimate                58.753 us  1880.082 us    61.58       2.4%    0.008s  ok 3.5e-16       10.51x
   baseline_matrix            11076.992 us 354463.751 us     0.33       0.1%    0.000s  ok 8.0e-16       1981.82x

-- L=36 (batched B=512), volume 46656, working set 729.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                14.170 us  7254.849 us   255.34      62.2%    3.015s  ok 3.6e-16       1.00x
   L36_pfa                       19.465 us  9966.283 us   185.87       0.9%    1.398s  ok 3.6e-16       1.37x
   fftw3_patient                 19.969 us 10224.318 us   181.18      50.2%   11.702s  ok 3.9e-16       1.41x
   L36_pencilfused               24.544 us 12566.775 us   147.41      40.0%    0.518s  ok 3.6e-16       1.73x
   mkl2026_dfti                  33.417 us 17109.366 us   108.27       0.5%    0.025s  ok 4.0e-16       2.36x
   mkl_dfti                      33.563 us 17184.146 us   107.80       0.5%    0.032s  ok 3.9e-16       2.37x
   fftw3_estimate                42.114 us 21562.290 us    85.91      70.8%    0.008s  ok 3.5e-16       2.97x
   fftw3_measure                 47.946 us 24548.381 us    75.46       5.2%    0.159s  ok 3.5e-16       3.38x
   ducc0_c2c                     58.269 us 29833.654 us    62.09       0.7%    0.000s  ok 3.0e-16       4.11x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_mixedradix                58.317 us    58.317 us   128.72       1.4%    0.170s  ok 4.1e-16       1.00x
   L45_pfa                       60.645 us    60.645 us   123.78       0.9%    0.796s  ok 4.0e-16       1.04x
   mkl2026_dfti                  83.911 us    83.911 us    89.46       0.6%    0.030s  ok 4.5e-16       1.44x
   mkl_dfti                      86.582 us    86.582 us    86.70       1.4%    0.017s  ok 4.4e-16       1.48x
   fftw3_estimate               104.906 us   104.906 us    71.56       9.8%    0.022s  ok 4.3e-16       1.80x
   fftw3_patient                137.889 us   137.889 us    54.44      37.9%    4.505s  ok 4.3e-16       2.36x
   fftw3_measure                147.817 us   147.817 us    50.78      61.3%    0.260s  ok 4.3e-16       2.53x
   ducc0_c2c                    332.683 us   332.683 us    22.56       6.5%    0.000s  ok 3.7e-16       5.70x
   baseline_matrix            28302.644 us 28302.644 us     0.27      11.9%    0.000s  ok 8.0e-16       485.33x

-- L=45 (batched B=16), volume 91125, working set 44.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_mixedradix                16.873 us   269.968 us   444.89       0.2%    0.214s  ok 4.1e-16       1.00x
   L45_pfa                       21.196 us   339.143 us   354.15       1.0%    0.656s  ok 4.0e-16       1.26x
   mkl2026_dfti                  37.056 us   592.901 us   202.57       2.5%    0.025s  ok 4.5e-16       2.20x
   mkl_dfti                      37.684 us   602.940 us   199.20       0.3%    0.015s  ok 4.4e-16       2.23x
   ducc0_c2c                     81.193 us  1299.083 us    92.46      16.5%    0.000s  ok 3.7e-16       4.81x
   fftw3_patient                 81.720 us  1307.516 us    91.86      14.3%    6.553s  ok 4.3e-16       4.84x
   fftw3_estimate                99.855 us  1597.675 us    75.18       0.6%    0.006s  ok 4.2e-16       5.92x
   fftw3_measure                106.108 us  1697.724 us    70.75       6.9%    0.155s  ok 4.2e-16       6.29x
   baseline_matrix            31706.881 us 507310.101 us     0.24       1.1%    0.000s  ok 8.0e-16       1879.15x

-- L=45 (batched B=256), volume 91125, working set 711.91 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       26.897 us  6885.705 us   279.09       4.4%    4.789s  ok 4.0e-16       1.00x
   fftw3_patient                 57.708 us 14773.323 us   130.08      21.1%    5.731s  ok 4.3e-16       2.15x
   L45_mixedradix                79.067 us 20241.037 us    94.94       0.2%    0.442s  ok 4.1e-16       2.94x
   fftw3_estimate                84.449 us 21619.038 us    88.89       5.0%    0.004s  ok 4.2e-16       3.14x
   fftw3_measure                 87.459 us 22389.413 us    85.83      60.3%    0.081s  ok 4.3e-16       3.25x
   mkl2026_dfti                  91.946 us 23538.295 us    81.64       0.3%    0.023s  ok 4.5e-16       3.42x
   mkl_dfti                      92.067 us 23569.250 us    81.53       0.1%    0.033s  ok 4.4e-16       3.42x
   ducc0_c2c                    116.302 us 29773.242 us    64.54       0.2%    0.000s  ok 3.7e-16       4.32x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  127.995 us   127.995 us   184.33       1.0%    0.405s  ok 4.5e-16       1.00x
   L64_radix8                   135.673 us   135.673 us   173.90       2.9%    0.057s  ok 4.5e-16       1.06x
   mkl_dfti                     153.235 us   153.235 us   153.97       0.4%    0.017s  ok 3.4e-16       1.20x
   mkl2026_dfti                 156.992 us   156.992 us   150.28       0.3%    0.006s  ok 3.4e-16       1.23x
   fftw3_measure                191.127 us   191.127 us   123.44      56.7%    0.464s  ok 3.4e-16       1.49x
   fftw3_estimate               267.589 us   267.589 us    88.17       0.1%    0.005s  ok 3.5e-16       2.09x
   fftw3_patient                305.282 us   305.282 us    77.28       0.8%   15.219s  ok 3.5e-16       2.39x
   ducc0_c2c                    405.036 us   405.036 us    58.25       3.9%    0.000s  ok 3.0e-16       3.16x
   baseline_matrix           122417.409 us 122417.409 us     0.19       1.4%    0.000s  ok 7.8e-16       956.42x

-- L=64 (batched B=8), volume 262144, working set 64.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl_dfti                      73.742 us   589.934 us   319.94       0.3%    0.034s  ok 3.4e-16       1.00x
   mkl2026_dfti                  81.079 us   648.633 us   290.99       1.1%    0.005s  ok 3.4e-16       1.10x
   L64_radix8                    91.521 us   732.166 us   257.79       0.2%    0.362s  ok 4.5e-16       1.24x
   L64_blocked                  111.725 us   893.802 us   211.17       0.2%    1.453s  ok 4.5e-16       1.52x
   ducc0_c2c                    243.841 us  1950.726 us    96.76       0.1%    0.000s  ok 3.0e-16       3.31x
   fftw3_patient                288.453 us  2307.620 us    81.79      14.8%   29.271s  ok 3.5e-16       3.91x
   fftw3_measure                391.648 us  3133.180 us    60.24      16.3%    0.415s  ok 3.6e-16       5.31x
   fftw3_estimate               683.771 us  5470.171 us    34.50       0.2%    0.002s  ok 3.5e-16       9.27x

-- L=64 (batched B=128), volume 262144, working set 1024.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                   95.686 us 12247.865 us   246.57      28.5%    5.000s  ok 4.5e-16       1.00x
   L64_radix8                   146.939 us 18808.231 us   160.56       2.3%    1.607s  ok 4.5e-16       1.54x
   fftw3_patient                184.357 us 23597.662 us   127.97       3.5%   22.262s  ok 3.5e-16       1.93x
   fftw3_measure                289.113 us 37006.429 us    81.60      14.8%    0.310s  ok 3.5e-16       3.02x
   mkl_dfti                     289.897 us 37106.774 us    81.38       0.6%    0.034s  ok 3.4e-16       3.03x
   mkl2026_dfti                 291.962 us 37371.166 us    80.81       0.9%    0.022s  ok 3.4e-16       3.05x
   fftw3_estimate               307.319 us 39336.851 us    76.77       3.0%    0.002s  ok 3.5e-16       3.21x
   ducc0_c2c                    369.137 us 47249.570 us    63.91       0.9%    0.000s  ok 3.0e-16       3.86x

backends:
   L13_direct               conj-folded dense 13x13 per axis, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+pf; mt t32 g1; ab[B64]=t1g1:7642,t32g1:6842,t16g1:1157,t8g1:2018,t4g1:2497
   L13_rader                Rader-13 split cyc/nega (186 FP/pt), X-first, 512-bit, batch-parallel ntb=32/32 NUMA-local scratch, B1 team t1=1; fuse=0 um=1 ys=0 pf=1 pw=0 nts=0 pace=1 znb=22 ab[B512]=i:446,pw!:428,pf!:426,nt!:1177,n16:761 pick=pw!
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 256-bit, pinned, X-first, pipelined, mt[vol nt=32 dyn=0 pf=0 pw=0 pt=0], sbw[rd/wr/cp/s17]=4.37/6.26/15.12/4.09, b1dec[yz/kyz/x/kx]=15.34/14.70/5.99/5.62, clk512/256=2.29/2.79 GHz, d256=2.79
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; mt mode=1 nt=32, xl 512t dy, pf=0, pfw=0, clk256=3.89 clk512=2.89, par=0.793 us/t
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h8, pf=0, pfw=0, cw=0, mt[n=32 bt=32 t1=0], clk256=2.89GHz, clk512=2.89GHz, p1=6.19 f23=10.36 fu=17.20 fu4=18.23, sbw[rd/wr/cp/s17]=3.14/8.34/12.56/3.72
   L23_matrixsimd           dense 23x23/axis conj-folded, pinned consts, X-first, spin-pool vol-parallel 512-bit T=32, nt=0 pf=0 pw=0, tune[pick=2.28(nt0 pf0 pw0) inc=4.63 us/t nv=128], clk512/256=2.29/2.79 GHz
   L23_rader                rader23 MT batch w4 team=32 pf=0 pw=0, tuner pick=3.40 inc=5.51 us/t nv=148
   L36_mixedradix           MT PFA 4x9 n1_9; pick=v1-split8-pf0 (B=1 m=32 arena=1 stream=0 18c) us/vol ser=123.9 pick=35.0 eff=0.44
   L36_pencilfused          L=36 MT plane-fused y+z then strided x, PFA4x9; pick volpar team=32 pw=2 scratch+nt (B=512); probe us/vol t32=15.71 t16=24.01 is0=18.93 nt=15.71
   L36_pfa                  GT-PFA 4x9 (n1_9) spin-pool mt; pick: vols T=32 pw=4 inplace pf=2 (B=32 nv=32 nc=24 Tpool=32); us/vol ip32=6.27 nt32=21.64 ip16=19.93 nt16=18.69
   L45_mixedradix           MT PFA9x5 pool; pick=vol32-v2-pfpp (B=256 nt=32 str=1 thr=32 nc=10); vol32-v2-pp=28.8 vol32-v2-pfpp=25.4 vol32-v2-cpin=39.9 vol32-v2-cpinpf=34.0 vol32-v2-cpy=41.4 vol32-v2-m0=33.4 grp16x2-v2-pp=29.9 grp8x4-v2-pp=34.3 vol16-v2-pp=67.8 vol32-v1-pp=31.2
   L45_pfa                  GT-PFA 9x5 two-sweep; tuner pick: pw4-mtf-blk (B=16, nv=16)
   L64_blocked              L64 8x8 split-sc two-stage, hugepage scratch; mt pick: eng=1(S:slab) G=1 nth=16 dyn=0 mode=cached pf=0 st=3 pro=0 (B=1 nv=1)
   L64_radix8               radix-8^2/axis AVX-512 MT; pick[B=1]=split-T16-g0-fused-pfw+slabpf0+sc0+p11; ser=1259us/vol pick=144 eff=0.55
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm; mt batch-split, per-thread NUMA scratch, 2D-raced (7 kernels x T<=32); variant=fused_pf_nt_xa_d2 T=32 disp=omp fork=13.63us raceT1=0.5699 raceBest=0.0558 omp=0.1138 pool=0.1135us/vol
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), radix-2-first VD6, ymm; variant=fused_zp nthr=1 kclk=2.89GHz ab1=f221.8,fx219.9ns xod=-0.9%
   L8_batchsimd             radix-8 split; pick[B=2048]: mode=FUSED nt=0 pf=s0w alloc=r8(a64,si512) mt{T=32 run=s0 dyn=0} arena{T32/s0w=0.0297,T32/s0=0.0274,T32/none=0.0287,T32/nt-s0=0.1807,T32/nt=0.1809,T32/nt-s0/d8=0.1779,T32/s0w/d8=0.0981,T16/nt-s0=0.1132}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 mt=vol/pool; B=2048 pick=fused+pfs/nt32 (mt-tuned) arena{fused+pfs/32=0.027,fused+pfs+pfw/32=0.028,fused-nt+pfs/32=0.180,seq3-nt+pfs/32=0.173,fused+pfs/24=0.040,fused+pfs/16=0.095,fused-nt+pfs/24=0.156,fused-nt+pfs/16=0.115}
   L8_radix8                radix-8 52-instr codelet; B=1 serial, B>1 batch-parallel 3p; pick[B=1,T=32]=avx512-2p (fixed) arena{2p=0.572 1f520*=0.577 1f*=0.578 3p*=0.608}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 32 threads
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, threaded, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, threaded, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, threaded, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, threaded, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, GNU OpenMP threading, batched
```
