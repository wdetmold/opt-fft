```
=== round mt_r2 ===
# round mt_r2
host: p55n3   date: 2026-08-22T12:04:58-04:00   slurm_job: 438551
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
threads: 32 of 32 (PROC_BIND=close)   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.220 us     0.220 us    38.15       0.5%    0.508s  ok 2.5e-16       1.00x
   L6_unrolled                    0.222 us     0.222 us    37.75       1.4%    0.365s  ok 2.5e-16       1.01x
   mkl_dfti                       0.371 us     0.371 us    22.57       2.4%    0.106s  ok 2.4e-16       1.69x
   mkl2026_dfti                   0.414 us     0.414 us    20.24       0.4%    0.008s  ok 2.5e-16       1.88x
   fftw3_patient                  0.514 us     0.514 us    16.28      10.1%    0.146s  ok 1.9e-16       2.34x
   ducc0_c2c                      5.146 us     5.146 us     1.63       0.7%    0.000s  ok 1.7e-16       23.44x
   baseline_matrix                8.387 us     8.387 us     1.00       0.0%    0.000s  ok 6.0e-16       38.20x
   fftw3_estimate                 9.737 us     9.737 us     0.86       1.0%    0.005s  ok 1.9e-16       44.35x
   fftw3_measure                 20.113 us    20.113 us     0.42      20.0%    0.033s  ok 1.9e-16       91.61x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.009 us    38.340 us   894.76       1.0%    0.463s  ok 2.4e-16       1.00x
   L6_pfa                         0.009 us    38.368 us   894.10       1.9%    1.430s  ok 2.4e-16       1.00x
   mkl_dfti                       0.021 us    85.826 us   399.71       2.2%    0.033s  ok 2.4e-16       2.24x
   mkl2026_dfti                   0.023 us    92.398 us   371.28       0.4%    0.024s  ok 2.5e-16       2.41x
   fftw3_patient                  0.025 us   102.368 us   335.12       0.2%    0.191s  ok 2.0e-16       2.67x
   fftw3_measure                  0.025 us   104.338 us   328.79       1.3%    0.017s  ok 2.0e-16       2.72x
   fftw3_estimate                 0.055 us   226.241 us   151.63       0.4%    0.003s  ok 2.0e-16       5.90x
   ducc0_c2c                      0.270 us  1104.110 us    31.07       2.3%    0.000s  ok 1.8e-16       28.80x
   baseline_matrix                8.502 us 34825.372 us     0.99       0.5%    0.000s  ok 6.0e-16       908.33x

-- L=6 (batched B=65536), volume 216, working set 432.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.035 us  2269.271 us   241.88       0.4%    2.725s  ok 2.4e-16       1.00x
   L6_pfa                         0.039 us  2581.532 us   212.62     106.5%   12.807s  ok 2.4e-16       1.14x
   fftw3_patient                  0.066 us  4357.059 us   125.98      84.2%    0.876s  ok 2.0e-16       1.92x
   fftw3_estimate                 0.121 us  7953.022 us    69.02       0.4%    0.002s  ok 2.0e-16       3.50x
   mkl_dfti                       0.122 us  8008.902 us    68.53       0.1%    0.035s  ok 2.4e-16       3.53x
   mkl2026_dfti                   0.122 us  8013.570 us    68.49       0.2%    0.027s  ok 2.5e-16       3.53x
   fftw3_measure                  0.122 us  8017.210 us    68.46       0.7%    0.017s  ok 2.0e-16       3.53x
   ducc0_c2c                      0.274 us 17939.133 us    30.60       0.5%    0.000s  ok 1.8e-16       7.91x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.557 us     0.557 us    41.34       0.5%    0.139s  ok 2.2e-16       1.00x
   L8_fusedaxes                   0.558 us     0.558 us    41.27       0.5%    0.078s  ok 2.2e-16       1.00x
   L8_radix8                      0.581 us     0.581 us    39.63       0.6%    0.081s  ok 1.3e-16       1.04x
   mkl_dfti                       0.652 us     0.652 us    35.36       2.1%    0.036s  ok 1.6e-16       1.17x
   mkl2026_dfti                   0.737 us     0.737 us    31.24       0.8%    0.007s  ok 1.6e-16       1.32x
   fftw3_patient                  1.208 us     1.208 us    19.08       1.1%    0.151s  ok 1.7e-16       2.17x
   ducc0_c2c                      6.241 us     6.241 us     3.69       1.7%    0.000s  ok 1.3e-16       11.20x
   fftw3_estimate                13.227 us    13.227 us     1.74       6.1%    0.004s  ok 1.7e-16       23.74x
   fftw3_measure                 19.574 us    19.574 us     1.18       9.3%    0.041s  ok 1.7e-16       35.12x
   baseline_matrix               26.401 us    26.401 us     0.87       0.0%    0.000s  ok 4.0e-16       47.38x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.026 us    53.868 us   875.96       1.3%    0.175s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.028 us    56.940 us   828.69      10.2%    0.114s  ok 2.3e-16       1.06x
   L8_radix8                      0.028 us    57.192 us   825.04       0.4%    0.139s  ok 2.3e-16       1.06x
   mkl_dfti                       0.036 us    74.645 us   632.14       1.0%    0.014s  ok 1.6e-16       1.39x
   mkl2026_dfti                   0.040 us    81.592 us   578.31       1.2%    0.025s  ok 1.6e-16       1.51x
   fftw3_patient                  0.055 us   111.779 us   422.14       1.9%    0.203s  ok 1.8e-16       2.08x
   fftw3_measure                  0.056 us   114.051 us   413.72       1.7%    0.019s  ok 1.8e-16       2.12x
   fftw3_estimate                 0.222 us   455.203 us   103.66       0.4%    0.003s  ok 1.7e-16       8.45x
   ducc0_c2c                      0.366 us   749.007 us    63.00       4.2%    0.000s  ok 1.3e-16       13.90x
   baseline_matrix               26.711 us 54704.827 us     0.86       0.2%    0.000s  ok 3.9e-16       1015.54x

-- L=8 (batched B=32768), volume 512, working set 512.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.159 us  5214.935 us   144.77       0.6%    1.027s  ok 1.8e-16       1.00x
   L8_batchsimd                   0.174 us  5690.846 us   132.66       0.3%    0.479s  ok 2.3e-16       1.09x
   L8_fusedaxes                   0.174 us  5690.987 us   132.66       0.3%    0.430s  ok 2.3e-16       1.09x
   L8_radix8                      0.174 us  5697.777 us   132.50       0.6%    0.654s  ok 2.3e-16       1.09x
   fftw3_estimate                 0.292 us  9571.593 us    78.88       0.4%    0.002s  ok 1.7e-16       1.84x
   fftw3_measure                  0.295 us  9664.536 us    78.12       0.3%    0.018s  ok 1.8e-16       1.85x
   mkl2026_dfti                   0.295 us  9676.568 us    78.02       0.3%    0.025s  ok 1.6e-16       1.86x
   mkl_dfti                       0.296 us  9686.260 us    77.94       0.3%    0.031s  ok 1.6e-16       1.86x
   ducc0_c2c                      0.559 us 18329.617 us    41.19       0.0%    0.000s  ok 1.3e-16       3.51x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.734 us     5.734 us    21.27       0.2%    0.043s  ok 2.9e-16       1.00x
   mkl2026_dfti                   7.637 us     7.637 us    15.97       0.5%    0.028s  ok 3.3e-16       1.33x
   mkl_dfti                       7.833 us     7.833 us    15.57       0.4%    0.017s  ok 3.2e-16       1.37x
   fftw3_patient                  8.425 us     8.425 us    14.47       1.8%    0.258s  ok 3.2e-16       1.47x
   fftw3_estimate                34.983 us    34.983 us     3.49      11.6%    0.002s  ok 3.2e-16       6.10x
   fftw3_measure                 36.309 us    36.309 us     3.36       9.1%    0.051s  ok 3.2e-16       6.33x
   ducc0_c2c                     43.931 us    43.931 us     2.78       3.5%    0.000s  ok 2.4e-16       7.66x
   baseline_matrix              184.485 us   184.485 us     0.66       0.0%    0.000s  ok 7.8e-16       32.17x

-- L=13 (batched B=512), volume 2197, working set 34.33 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     0.309 us   158.046 us   395.06       1.9%    0.057s  ok 2.9e-16       1.00x
   mkl2026_dfti                   0.360 us   184.292 us   338.80       0.7%    0.030s  ok 3.2e-16       1.17x
   fftw3_patient                  0.366 us   187.542 us   332.93       1.5%    0.336s  ok 3.2e-16       1.19x
   mkl_dfti                       0.369 us   188.830 us   330.65       0.4%    0.034s  ok 3.2e-16       1.19x
   fftw3_measure                  0.374 us   191.299 us   326.39       1.5%    0.017s  ok 3.2e-16       1.21x
   fftw3_estimate                 0.433 us   221.476 us   281.91      16.0%    0.003s  ok 3.2e-16       1.40x
   ducc0_c2c                      3.397 us  1739.029 us    35.90       1.3%    0.000s  ok 2.5e-16       11.00x
   baseline_matrix              186.639 us 95559.061 us     0.65       0.1%    0.000s  ok 7.9e-16       604.63x

-- L=13 (batched B=8192), volume 2197, working set 549.25 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.603 us  4942.232 us   202.13     110.3%    1.348s  ok 3.2e-16       1.00x
   L13_direct                     0.984 us  8061.910 us   123.92       0.1%    0.072s  ok 2.9e-16       1.63x
   mkl_dfti                       1.267 us 10379.470 us    96.25       0.4%    0.034s  ok 3.2e-16       2.10x
   fftw3_measure                  1.268 us 10385.870 us    96.19       0.4%    0.017s  ok 3.2e-16       2.10x
   mkl2026_dfti                   1.269 us 10391.939 us    96.13       0.5%    0.024s  ok 3.2e-16       2.10x
   fftw3_estimate                 1.270 us 10407.074 us    95.99       1.3%    0.002s  ok 3.2e-16       2.11x
   ducc0_c2c                      3.356 us 27490.813 us    36.34       0.7%    0.000s  ok 2.5e-16       5.56x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 6.163 us     6.163 us    48.87       1.8%    0.472s  ok 3.2e-16       1.00x
   L17_rader                      6.776 us     6.776 us    44.46       4.8%    0.481s  ok 3.1e-16       1.10x
   L17_winograd                   7.604 us     7.604 us    39.62       1.1%    0.659s  ok 3.3e-16       1.23x
   mkl_dfti                      22.632 us    22.632 us    13.31       2.8%    0.014s  ok 3.1e-16       3.67x
   mkl2026_dfti                  23.553 us    23.553 us    12.79       2.9%    0.008s  ok 3.1e-16       3.82x
   fftw3_patient                 29.649 us    29.649 us    10.16      15.4%    0.214s  ok 3.0e-16       4.81x
   fftw3_measure                 43.230 us    43.230 us     6.97      14.1%    0.062s  ok 3.0e-16       7.01x
   fftw3_estimate                44.328 us    44.328 us     6.80       9.1%    0.020s  ok 3.0e-16       7.19x
   ducc0_c2c                    104.208 us   104.208 us     2.89       2.5%    0.000s  ok 2.6e-16       16.91x
   baseline_matrix              538.260 us   538.260 us     0.56       0.0%    0.000s  ok 8.4e-16       87.33x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 0.755 us   193.320 us   398.89       1.9%    0.602s  ok 3.3e-16       1.00x
   L17_rader                      0.792 us   202.669 us   380.49       1.0%    0.295s  ok 3.2e-16       1.05x
   L17_winograd                   1.078 us   275.953 us   279.45       7.2%    3.137s  ok 3.3e-16       1.43x
   fftw3_estimate                 3.906 us   999.893 us    77.12       0.2%    0.004s  ok 3.0e-16       5.17x
   fftw3_measure                  3.907 us  1000.085 us    77.11       3.4%    0.013s  ok 3.0e-16       5.17x
   fftw3_patient                  3.911 us  1001.225 us    77.02       0.4%    0.320s  ok 3.0e-16       5.18x
   mkl_dfti                       4.851 us  1241.782 us    62.10       1.7%    0.032s  ok 3.1e-16       6.42x
   mkl2026_dfti                   4.917 us  1258.790 us    61.26       1.3%    0.003s  ok 3.1e-16       6.51x
   ducc0_c2c                      8.088 us  2070.510 us    37.24       5.8%    0.000s  ok 2.6e-16       10.71x
   baseline_matrix              540.439 us 138352.316 us     0.56       0.1%    0.000s  ok 8.4e-16       715.66x

-- L=17 (batched B=4096), volume 4913, working set 614.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                   1.220 us  4998.128 us   246.86      73.3%    4.168s  ok 3.3e-16       1.00x
   L17_rader                      2.200 us  9010.270 us   136.93       0.2%    0.680s  ok 3.2e-16       1.80x
   L17_matrixsimd                 2.858 us 11705.497 us   105.41       2.0%    0.876s  ok 3.3e-16       2.34x
   fftw3_patient                  4.066 us 16653.193 us    74.09       2.6%    2.706s  ok 3.0e-16       3.33x
   fftw3_estimate                 4.094 us 16770.859 us    73.57       0.2%    0.004s  ok 3.0e-16       3.36x
   fftw3_measure                  4.107 us 16824.193 us    73.34       0.4%    0.011s  ok 3.0e-16       3.37x
   mkl_dfti                       4.924 us 20167.866 us    61.18       1.3%    0.032s  ok 3.1e-16       4.04x
   mkl2026_dfti                   4.969 us 20353.264 us    60.62       4.6%    0.024s  ok 3.1e-16       4.07x
   ducc0_c2c                      8.000 us 32768.948 us    37.65       0.9%    0.000s  ok 2.6e-16       6.56x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                11.628 us    11.628 us    71.00       2.0%    0.340s  ok 3.8e-16       1.00x
   L23_rader                     11.854 us    11.854 us    69.65       0.2%    0.079s  ok 3.8e-16       1.02x
   fftw3_patient                 39.272 us    39.272 us    21.02       1.0%    0.233s  ok 3.7e-16       3.38x
   mkl_dfti                      42.524 us    42.524 us    19.41       2.3%    0.017s  ok 4.2e-16       3.66x
   mkl2026_dfti                  43.674 us    43.674 us    18.90       0.6%    0.004s  ok 4.2e-16       3.76x
   fftw3_estimate                45.461 us    45.461 us    18.16      17.6%    0.019s  ok 3.7e-16       3.91x
   fftw3_measure                 50.938 us    50.938 us    16.21       1.3%    0.062s  ok 3.7e-16       4.38x
   ducc0_c2c                    290.439 us   290.439 us     2.84      10.8%    0.000s  ok 2.9e-16       24.98x
   baseline_matrix             1801.292 us  1801.292 us     0.46       0.0%    0.000s  ok 7.3e-16       154.91x

-- L=23 (batched B=128), volume 12167, working set 47.53 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      2.242 us   286.940 us   368.28       8.5%    0.192s  ok 3.8e-16       1.00x
   L23_matrixsimd                 2.284 us   292.357 us   361.45       5.1%    0.123s  ok 3.8e-16       1.02x
   fftw3_patient                 12.461 us  1595.008 us    66.25       0.1%    0.317s  ok 3.7e-16       5.56x
   fftw3_estimate                12.462 us  1595.120 us    66.25       0.1%    0.004s  ok 3.7e-16       5.56x
   fftw3_measure                 12.782 us  1636.037 us    64.59      53.6%    0.014s  ok 3.7e-16       5.70x
   mkl_dfti                      14.567 us  1864.590 us    56.67       0.9%    0.033s  ok 4.2e-16       6.50x
   mkl2026_dfti                  15.773 us  2018.928 us    52.34       1.5%    0.003s  ok 4.2e-16       7.04x
   ducc0_c2c                     23.523 us  3010.939 us    35.10       1.2%    0.000s  ok 2.8e-16       10.49x
   baseline_matrix             1806.361 us 231214.190 us     0.46       0.1%    0.000s  ok 7.4e-16       805.79x

-- L=23 (batched B=2048), volume 12167, working set 760.44 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                 5.934 us 12153.180 us   139.12       0.8%    0.471s  ok 3.8e-16       1.00x
   L23_rader                      7.167 us 14677.723 us   115.19       0.9%    0.221s  ok 3.8e-16       1.21x
   fftw3_patient                 12.707 us 26024.677 us    64.97       4.0%    1.384s  ok 3.7e-16       2.14x
   fftw3_measure                 12.853 us 26323.958 us    64.23       0.7%    0.013s  ok 3.7e-16       2.17x
   fftw3_estimate                12.854 us 26325.830 us    64.22       0.8%    0.004s  ok 3.7e-16       2.17x
   mkl_dfti                      14.739 us 30184.743 us    56.01       0.2%    0.034s  ok 4.2e-16       2.48x
   mkl2026_dfti                  15.921 us 32606.974 us    51.85       0.1%    0.023s  ok 4.2e-16       2.68x
   ducc0_c2c                     24.548 us 50275.248 us    33.63       0.5%    0.000s  ok 2.8e-16       4.14x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                23.012 us    23.012 us   157.23       1.3%    0.221s  ok 3.6e-16       1.00x
   L36_pfa                       25.818 us    25.818 us   140.14       0.8%    0.074s  ok 3.6e-16       1.12x
   L36_pencilfused               25.866 us    25.866 us   139.88       3.0%    0.232s  ok 3.6e-16       1.12x
   mkl2026_dfti                  38.297 us    38.297 us    94.47       0.8%    0.025s  ok 4.1e-16       1.66x
   mkl_dfti                      38.463 us    38.463 us    94.07       0.3%    0.035s  ok 3.9e-16       1.67x
   fftw3_measure                 64.814 us    64.814 us    55.82      75.7%    0.265s  ok 3.8e-16       2.82x
   fftw3_patient                 69.047 us    69.047 us    52.40       5.7%   10.499s  ok 3.6e-16       3.00x
   fftw3_estimate                75.359 us    75.359 us    48.01      29.4%    0.016s  ok 3.5e-16       3.27x
   ducc0_c2c                    210.786 us   210.786 us    17.16       2.9%    0.000s  ok 3.1e-16       9.16x
   baseline_matrix            10847.293 us 10847.293 us     0.33       0.2%    0.000s  ok 8.0e-16       471.37x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                 5.213 us   166.822 us   694.03      24.5%    0.934s  ok 3.6e-16       1.00x
   L36_pfa                        5.236 us   167.537 us   691.07      13.6%    0.231s  ok 3.6e-16       1.00x
   L36_pencilfused                5.515 us   176.482 us   656.04      13.2%    2.404s  ok 3.6e-16       1.06x
   mkl_dfti                       7.465 us   238.875 us   484.69       0.8%    0.033s  ok 3.9e-16       1.43x
   mkl2026_dfti                   7.823 us   250.332 us   462.51       0.9%    0.004s  ok 4.0e-16       1.50x
   fftw3_measure                 16.764 us   536.442 us   215.83     348.8%    0.161s  ok 3.5e-16       3.22x
   fftw3_patient                 17.862 us   571.580 us   202.56      56.7%   10.717s  ok 3.6e-16       3.43x
   ducc0_c2c                     39.026 us  1248.832 us    92.71       2.4%    0.000s  ok 3.0e-16       7.49x
   fftw3_estimate                42.639 us  1364.446 us    84.85      80.3%    0.009s  ok 3.5e-16       8.18x
   baseline_matrix            11084.954 us 354718.513 us     0.33       0.1%    0.000s  ok 8.0e-16       2126.33x

-- L=36 (batched B=512), volume 46656, working set 729.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                 9.989 us  5114.502 us   362.20      92.9%    4.727s  ok 3.6e-16       1.00x
   fftw3_patient                 19.677 us 10074.379 us   183.88      42.0%   11.488s  ok 3.6e-16       1.97x
   L36_pfa                       31.569 us 16163.259 us   114.61       0.5%    0.919s  ok 3.6e-16       3.16x
   mkl2026_dfti                  32.916 us 16853.165 us   109.92       0.3%    0.024s  ok 4.0e-16       3.30x
   mkl_dfti                      33.065 us 16929.219 us   109.42       0.3%    0.035s  ok 3.9e-16       3.31x
   fftw3_measure                 33.617 us 17211.963 us   107.63      33.8%    0.160s  ok 3.5e-16       3.37x
   L36_pencilfused               34.043 us 17430.230 us   106.28       0.3%    2.413s  ok 3.6e-16       3.41x
   fftw3_estimate                34.295 us 17559.128 us   105.50      31.9%    0.008s  ok 3.5e-16       3.43x
   ducc0_c2c                     58.523 us 29963.546 us    61.82       0.6%    0.000s  ok 3.0e-16       5.86x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       57.197 us    57.197 us   131.24       0.3%    0.230s  ok 4.0e-16       1.00x
   L45_mixedradix                57.549 us    57.549 us   130.44       2.3%    0.200s  ok 4.1e-16       1.01x
   mkl2026_dfti                  84.225 us    84.225 us    89.13       0.5%    0.027s  ok 4.5e-16       1.47x
   mkl_dfti                      86.365 us    86.365 us    86.92       0.9%    0.017s  ok 4.4e-16       1.51x
   fftw3_estimate               107.349 us   107.349 us    69.93       1.2%    0.013s  ok 4.3e-16       1.88x
   fftw3_measure                126.320 us   126.320 us    59.43      58.9%    0.255s  ok 4.3e-16       2.21x
   fftw3_patient                138.191 us   138.191 us    54.32       3.9%    4.385s  ok 4.3e-16       2.42x
   ducc0_c2c                    316.981 us   316.981 us    23.68      13.1%    0.000s  ok 3.7e-16       5.54x
   baseline_matrix            28702.895 us 28702.895 us     0.26      10.0%    0.000s  ok 8.0e-16       501.83x

-- L=45 (batched B=16), volume 91125, working set 44.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       15.751 us   252.023 us   476.57       2.5%    0.581s  ok 4.0e-16       1.00x
   L45_mixedradix                16.822 us   269.149 us   446.25       2.8%    0.336s  ok 4.1e-16       1.07x
   mkl_dfti                      37.669 us   602.705 us   199.28       1.5%    0.016s  ok 4.4e-16       2.39x
   mkl2026_dfti                  37.885 us   606.158 us   198.14       0.1%    0.003s  ok 4.5e-16       2.41x
   ducc0_c2c                     81.640 us  1306.237 us    91.95      17.1%    0.000s  ok 3.7e-16       5.18x
   fftw3_patient                 83.945 us  1343.115 us    89.42       7.1%    6.563s  ok 4.3e-16       5.33x
   fftw3_estimate                98.446 us  1575.139 us    76.25       1.4%    0.006s  ok 4.2e-16       6.25x
   fftw3_measure                109.434 us  1750.936 us    68.60       7.1%    0.152s  ok 4.2e-16       6.95x
   baseline_matrix            32868.896 us 525902.338 us     0.23       0.5%    0.000s  ok 8.0e-16       2086.73x

-- L=45 (batched B=256), volume 91125, working set 711.91 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       45.446 us 11634.136 us   165.18       6.2%    4.336s  ok 4.0e-16       1.00x
   L45_mixedradix                50.216 us 12855.207 us   149.49       1.9%    3.037s  ok 4.1e-16       1.10x
   fftw3_patient                 56.952 us 14579.672 us   131.81       6.2%    5.820s  ok 4.3e-16       1.25x
   fftw3_estimate                81.467 us 20855.475 us    92.14       1.2%    0.004s  ok 4.2e-16       1.79x
   mkl_dfti                      91.889 us 23523.498 us    81.69       0.2%    0.033s  ok 4.4e-16       2.02x
   mkl2026_dfti                  91.949 us 23539.006 us    81.64       0.4%    0.026s  ok 4.5e-16       2.02x
   fftw3_measure                102.962 us 26358.184 us    72.91      37.2%    0.081s  ok 4.3e-16       2.27x
   ducc0_c2c                    116.819 us 29905.677 us    64.26       1.1%    0.000s  ok 3.7e-16       2.57x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  126.988 us   126.988 us   185.79       0.5%    0.408s  ok 4.5e-16       1.00x
   L64_radix8                   136.183 us   136.183 us   173.25       2.0%    0.070s  ok 4.5e-16       1.07x
   mkl_dfti                     153.300 us   153.300 us   153.90       0.7%    0.016s  ok 3.4e-16       1.21x
   mkl2026_dfti                 156.942 us   156.942 us   150.33       0.1%    0.005s  ok 3.4e-16       1.24x
   fftw3_estimate               266.529 us   266.529 us    88.52       0.5%    0.006s  ok 3.5e-16       2.10x
   fftw3_measure                300.767 us   300.767 us    78.44       7.8%    0.478s  ok 3.5e-16       2.37x
   fftw3_patient                306.518 us   306.518 us    76.97       2.2%   14.925s  ok 3.5e-16       2.41x
   ducc0_c2c                    401.285 us   401.285 us    58.79      13.3%    0.000s  ok 3.0e-16       3.16x
   baseline_matrix           123504.305 us 123504.305 us     0.19       1.0%    0.000s  ok 7.8e-16       972.57x

-- L=64 (batched B=8), volume 262144, working set 64.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl_dfti                      72.988 us   583.903 us   323.25       1.4%    0.033s  ok 3.4e-16       1.00x
   L64_blocked                   76.722 us   613.773 us   307.51       8.8%    2.008s  ok 4.5e-16       1.05x
   mkl2026_dfti                  78.488 us   627.900 us   300.59       0.3%    0.003s  ok 3.4e-16       1.08x
   L64_radix8                    90.578 us   724.621 us   260.47       3.6%    0.450s  ok 4.5e-16       1.24x
   ducc0_c2c                    242.602 us  1940.818 us    97.25       0.3%    0.000s  ok 3.0e-16       3.32x
   fftw3_patient                283.853 us  2270.827 us    83.12      15.7%   30.054s  ok 3.5e-16       3.89x
   fftw3_measure                386.573 us  3092.586 us    61.03      18.4%    0.414s  ok 3.4e-16       5.30x
   fftw3_estimate               684.395 us  5475.161 us    34.47       0.1%    0.002s  ok 3.5e-16       9.38x

-- L=64 (batched B=128), volume 262144, working set 1024.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    69.494 us  8895.202 us   339.50      18.2%    2.049s  ok 4.5e-16       1.00x
   L64_blocked                   90.483 us 11581.865 us   260.74      49.8%    8.185s  ok 4.5e-16       1.30x
   fftw3_patient                164.390 us 21041.964 us   143.52      17.0%   21.995s  ok 3.5e-16       2.37x
   fftw3_measure                293.381 us 37552.773 us    80.42      27.6%    0.307s  ok 3.5e-16       4.22x
   mkl_dfti                     296.092 us 37899.763 us    79.68       1.1%    0.034s  ok 3.4e-16       4.26x
   mkl2026_dfti                 296.228 us 37917.232 us    79.64       1.4%    0.025s  ok 3.4e-16       4.26x
   fftw3_estimate               306.793 us 39269.442 us    76.90       3.0%    0.002s  ok 3.5e-16       4.41x
   ducc0_c2c                    362.930 us 46455.036 us    65.01       3.9%    0.000s  ok 3.0e-16       5.22x

backends:
   L13_direct               conj-folded dense 13x13 per axis, pinned sines; 512b zsolid staged-NT X-first+pfin; mt t32 g1; ab[B64]=t1g1:11937,t32g1:6869,t16g1:1382,t8g1:2228,t4g1:3399
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, addr-safe t1, pt=0, mt[intra nt=16 nxr=16 xpf=0], b1dec[yz/kyz/x/kx]=15.22/14.40/5.94/5.59, clk512/256=2.29/2.79 GHz, d256=2.79, anb=1
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; mt mode=1 nt=32, xl 512t stpnt dy, pf=0, pfw=0, clk256=3.89 clk512=2.89, par=1.926 us/t cut=eq spr=1.18x
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, cw=0, nt=0, mt[n=32 bt=32 dyn=1 t1=0 s2=0], clk256=2.89GHz, clk512=2.89GHz, p1=6.16 f23=10.71 fu=18.13 fu4=17.50, sbw[rd/wr/cp/s17]=3.69/10.37/16.56/4.10
   L23_matrixsimd           dense 23x23/axis conj-folded, pinned consts, X-first, spin-pool vol-parallel 512-bit T=32 nsock=2, nt=1 pf=0 pw=0 pt=0, tune[pick=5.35(nt1 pf0 pw0) inc=5.35 us/t nv=640], clk512/256=2.29/2.79 GHz
   L23_rader                rader23 pool batch w4 team=32 pf=2 pw=1, tuner pick=2.74 inc=2.91 us/t nv=148
   L36_mixedradix           MT PFA 4x9 n1_9 spin-pool; pick=v0-vol32-pfin (B=32 m=32 arena=32 stream=1 30c) us/vol ser=260.0 pick=7.3 eff=1.12 dsp=2.03
   L36_pencilfused          L=36 MT plane-fused y+z then strided x, PFA4x9; pick volpar-pool team=32 pw=4 istream+pfw (B=32); probe us/vol t32=6.34 t16=15.83 is0=6.51 nt=9.53 pl=6.34
   L36_pfa                  GT-PFA 4x9 (n1_9) spin-pool mt; pick: vols T=32 pw=4 inplace pf=7 (B=512 nv=128 nc=14 Tpool=32); us/vol@32 ip0=27.51 ip2=28.99 ip7=24.97 nt0=26.26 nt1=26.40 sc=30.77
   L45_mixedradix           MT PFA9x5 pool; pick=vnt32-v2-pfpk (B=256 nt=128 str=1 thr=32 nc=15); vnt32-v2-pf=57.0 vnt32-v2-pfpk=53.3 vnt32-v2-m0=53.5 vol32-v2-pfpp=79.6 vol32-v2-pp=81.0 vnt16-v2-pf=60.0 vol32-v2-pf=81.3 grp16x2-v2-pp=68.6 grp8x4-v2-pp=59.0 vol32-v2-cpin=74.1 vol32-v2-m0=79.4 vnt32-v1-pf=57.1 vol16-v2-pp=67.9 vol32-v2-cpinpf=71.4 vol32-v2-cpy=74.0
   L45_pfa                  GT-PFA 9x5 pool T=32; pick=pw4-mtn-pfi (B=256 nv=128) mtv-pf0=61.1 mtv-pf3=61.9 mtn-pf0=44.4 mtn-pfi=42.4 mts-pf0=44.8 mts-pfi=44.1 g2-pf0=49.0 g2-pfw=51.4 g2n-pf0=45.0 g2n-pfi=43.2 mtf-rr=76.9 mtf-blk=65.3 mtf-bpf=63.7 ip-pf0=715.6 ip-pf3=603.2 mtv-pf0=61.3 mtf-blk=68.7 ip-pf0=747.7
   L64_blocked              L64 8x8 split-sc two-stage, hugepage scratch; mt pick: eng=1(S:slab) G=1 nth=16 dyn=0 sb=0 mode=cached pf=0 st=3 pro=0 (B=1 nv=1)
   L64_radix8               radix-8^2/axis AVX-512 MT; pick[B=128]=gang-T32-g4-fused-nt+slabpf1+sc0+p10; ser=1441us/vol pick=70 eff=0.64
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm; mt batch-split, per-thread NUMA scratch, 2D-raced@pool (7 kernels x T<=32); variant=fused_pf_d2 T=32 disp=pool fork=14.17us raceT1=0.4080 raceBest=0.0092 omp=0.0117 pool=0.0092us/vol
   L6_unrolled              L=6: unrolled PFA 2x3 codelet ymm, batch-parallel contiguous chunks, per-thread NUMA-local scratch; variant=3pass_nt_pf nthr=32 disp=omp od=36.6,36.5ns tm=1:645,2:381,4:195,8:105,16:88,24:52,32:39ns
   L8_batchsimd             radix-8 split; pick[B=32768]: mode=FUSED nt=0 pf=s0w alloc=r8(a64,si512) mt{T=16 run=nt-s0 dyn=0} arena{T32/nt-s0=0.2105,T32/s0=0.2670,T32/s0w=0.2624,T32/none=0.2634,T32/nt=0.2076,T32/nt-s0/d128=0.2136,T32/s0/d128=0.2628,T24/nt-s0=0.1961,T16/nt-s0=0.1850}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 mt=vol/pool; B=2048 pick=fused+pfs/nt32 (mt-tuned) arena{fused+pfs/32=0.027,fused+pfs+pfw/32=0.028,fused-nt+pfs/32=0.181,seq3-nt+pfs/32=0.175,fused+pfs/24=0.039,fused+pfs/16=0.089,fused-nt+pfs/24=0.160,fused-nt+pfs/16=0.113}
   L8_radix8                radix-8 52-instr codelet; B=1 serial, B>1 pool-parallel 1f (3p probes); pick[B=1,T=32]=avx512-2p (fixed) arena{2p=0.571 1f520*=0.577 1f*=0.578 3p*=0.616}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 32 threads
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, threaded, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, threaded, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, threaded, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, threaded, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, GNU OpenMP threading, batched
```
