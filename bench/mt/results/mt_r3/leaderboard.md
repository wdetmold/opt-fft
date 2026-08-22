```
=== round mt_r3 ===
# round mt_r3
host: p55n3   date: 2026-08-22T13:55:03-04:00   slurm_job: 438566
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
threads: 32 of 32 (PROC_BIND=close)   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.211 us     0.211 us    39.67       4.0%    0.448s  ok 2.5e-16       1.00x
   L6_unrolled                    0.221 us     0.221 us    37.83       0.6%    0.330s  ok 2.5e-16       1.05x
   mkl_dfti                       0.372 us     0.372 us    22.50       1.2%    0.017s  ok 2.5e-16       1.76x
   mkl2026_dfti                   0.405 us     0.405 us    20.70       2.0%    0.008s  ok 2.5e-16       1.92x
   fftw3_patient                  0.509 us     0.509 us    16.44       1.5%    0.147s  ok 2.1e-16       2.41x
   ducc0_c2c                      5.103 us     5.103 us     1.64       0.9%    0.000s  ok 2.0e-16       24.17x
   baseline_matrix                8.387 us     8.387 us     1.00       0.0%    0.000s  ok 5.9e-16       39.72x
   fftw3_estimate                 9.833 us     9.833 us     0.85       1.2%    0.005s  ok 2.1e-16       46.57x
   fftw3_measure                 19.879 us    19.879 us     0.42      11.3%    0.038s  ok 2.1e-16       94.15x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.009 us    37.746 us   908.85       1.4%    1.689s  ok 2.4e-16       1.00x
   L6_unrolled                    0.009 us    38.567 us   889.50       1.6%    0.436s  ok 2.4e-16       1.02x
   mkl_dfti                       0.021 us    84.093 us   407.94       3.9%    0.034s  ok 2.4e-16       2.23x
   mkl2026_dfti                   0.023 us    92.489 us   370.91       0.8%    0.004s  ok 2.5e-16       2.45x
   fftw3_patient                  0.024 us    99.617 us   344.37       2.0%    0.196s  ok 2.0e-16       2.64x
   fftw3_measure                  0.026 us   104.792 us   327.36      12.1%    0.018s  ok 2.0e-16       2.78x
   fftw3_estimate                 0.055 us   226.578 us   151.41       0.2%    0.003s  ok 2.0e-16       6.00x
   ducc0_c2c                      0.266 us  1089.826 us    31.48       1.6%    0.000s  ok 1.8e-16       28.87x
   baseline_matrix                8.516 us 34881.136 us     0.98       0.2%    0.000s  ok 6.0e-16       924.11x

-- L=6 (batched B=65536), volume 216, working set 432.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.066 us  4326.681 us   126.86       2.4%    0.867s  ok 2.0e-16       1.00x
   L6_pfa                         0.079 us  5161.366 us   106.34       0.2%    2.557s  ok 2.4e-16       1.19x
   L6_unrolled                    0.096 us  6290.172 us    87.26       0.1%    2.589s  ok 2.4e-16       1.45x
   fftw3_estimate                 0.121 us  7946.963 us    69.07       0.2%    0.002s  ok 2.0e-16       1.84x
   mkl2026_dfti                   0.122 us  8000.971 us    68.60       0.2%    0.022s  ok 2.5e-16       1.85x
   mkl_dfti                       0.122 us  8007.355 us    68.55       0.2%    0.033s  ok 2.4e-16       1.85x
   fftw3_measure                  0.122 us  8010.434 us    68.52       0.5%    0.017s  ok 2.0e-16       1.85x
   ducc0_c2c                      0.271 us 17791.744 us    30.85       1.1%    0.000s  ok 1.8e-16       4.11x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.551 us     0.551 us    41.84       1.3%    0.084s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.559 us     0.559 us    41.24       0.2%    0.110s  ok 2.3e-16       1.01x
   L8_radix8                      0.576 us     0.576 us    40.01       1.4%    0.084s  ok 1.4e-16       1.05x
   mkl_dfti                       0.657 us     0.657 us    35.05       2.2%    0.033s  ok 1.6e-16       1.19x
   mkl2026_dfti                   0.743 us     0.743 us    31.00       1.2%    0.008s  ok 1.7e-16       1.35x
   fftw3_patient                  1.247 us     1.247 us    18.48       8.3%    0.152s  ok 1.8e-16       2.26x
   ducc0_c2c                      6.202 us     6.202 us     3.71       4.8%    0.000s  ok 1.4e-16       11.26x
   fftw3_estimate                13.692 us    13.692 us     1.68       4.1%    0.002s  ok 1.8e-16       24.87x
   fftw3_measure                 19.280 us    19.280 us     1.20      11.8%    0.041s  ok 1.8e-16       35.02x
   baseline_matrix               26.404 us    26.404 us     0.87       0.0%    0.000s  ok 3.9e-16       47.95x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.026 us    53.496 us   882.05       1.3%    0.171s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.027 us    54.683 us   862.89       1.1%    0.094s  ok 2.3e-16       1.02x
   L8_radix8                      0.028 us    56.894 us   829.36       1.2%    0.147s  ok 2.3e-16       1.06x
   mkl_dfti                       0.036 us    73.980 us   637.82       1.3%    0.036s  ok 1.6e-16       1.38x
   mkl2026_dfti                   0.040 us    81.454 us   579.29       1.3%    0.003s  ok 1.6e-16       1.52x
   fftw3_patient                  0.055 us   112.909 us   417.91       6.5%    0.211s  ok 1.8e-16       2.11x
   fftw3_measure                  0.056 us   113.833 us   414.52       1.6%    0.018s  ok 1.8e-16       2.13x
   fftw3_estimate                 0.223 us   455.785 us   103.53       0.1%    0.003s  ok 1.7e-16       8.52x
   ducc0_c2c                      0.376 us   770.895 us    61.21       3.1%    0.000s  ok 1.3e-16       14.41x
   baseline_matrix               26.725 us 54732.749 us     0.86       0.2%    0.000s  ok 3.9e-16       1023.12x

-- L=8 (batched B=32768), volume 512, working set 512.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.159 us  5224.497 us   144.51       0.1%    1.025s  ok 1.7e-16       1.00x
   L8_fusedaxes                   0.172 us  5646.393 us   133.71       0.4%    0.430s  ok 2.3e-16       1.08x
   L8_radix8                      0.174 us  5715.018 us   132.10       0.4%    0.656s  ok 2.3e-16       1.09x
   L8_batchsimd                   0.175 us  5721.440 us   131.96       0.2%    0.349s  ok 2.3e-16       1.10x
   fftw3_estimate                 0.292 us  9556.919 us    79.00       0.2%    0.002s  ok 1.7e-16       1.83x
   fftw3_measure                  0.294 us  9627.214 us    78.42       0.7%    0.018s  ok 1.8e-16       1.84x
   mkl2026_dfti                   0.295 us  9668.135 us    78.09       0.1%    0.024s  ok 1.6e-16       1.85x
   mkl_dfti                       0.296 us  9683.092 us    77.97       0.2%    0.033s  ok 1.6e-16       1.85x
   ducc0_c2c                      0.547 us 17914.282 us    42.14       2.1%    0.000s  ok 1.3e-16       3.43x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.868 us     5.868 us    20.78       3.3%    0.048s  ok 2.9e-16       1.00x
   L13_rader                      6.065 us     6.065 us    20.11      40.1%    0.021s  ok 4.0e-16       1.03x
   mkl2026_dfti                   7.642 us     7.642 us    15.96       0.2%    0.027s  ok 3.2e-16       1.30x
   mkl_dfti                       7.851 us     7.851 us    15.53       0.4%    0.018s  ok 3.2e-16       1.34x
   fftw3_patient                  8.506 us     8.506 us    14.34       0.7%    0.257s  ok 3.2e-16       1.45x
   fftw3_measure                 32.778 us    32.778 us     3.72      19.6%    0.059s  ok 3.2e-16       5.59x
   fftw3_estimate                36.600 us    36.600 us     3.33      26.0%    0.007s  ok 3.2e-16       6.24x
   ducc0_c2c                     44.025 us    44.025 us     2.77       0.5%    0.000s  ok 2.5e-16       7.50x
   baseline_matrix              184.378 us   184.378 us     0.66       0.1%    0.000s  ok 7.9e-16       31.42x

-- L=13 (batched B=512), volume 2197, working set 34.33 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_rader                      0.302 us   154.577 us   403.92      49.6%    0.063s  ok 4.0e-16       1.00x
   L13_direct                     0.305 us   156.053 us   400.10       0.7%    0.057s  ok 2.9e-16       1.01x
   mkl2026_dfti                   0.360 us   184.397 us   338.60       0.8%    0.005s  ok 3.2e-16       1.19x
   mkl_dfti                       0.369 us   189.071 us   330.23       0.6%    0.015s  ok 3.2e-16       1.22x
   fftw3_patient                  0.371 us   189.892 us   328.80       0.7%    0.322s  ok 3.2e-16       1.23x
   fftw3_measure                  0.375 us   192.060 us   325.09       0.6%    0.018s  ok 3.2e-16       1.24x
   fftw3_estimate                 0.427 us   218.384 us   285.91       0.2%    0.003s  ok 3.2e-16       1.41x
   ducc0_c2c                      3.405 us  1743.461 us    35.81       2.7%    0.000s  ok 2.5e-16       11.28x
   baseline_matrix              186.727 us 95604.231 us     0.65       0.0%    0.000s  ok 7.9e-16       618.49x

-- L=13 (batched B=8192), volume 2197, working set 549.25 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw3_patient                  0.603 us  4938.706 us   202.28     110.2%    1.351s  ok 3.2e-16       1.00x
   L13_direct                     0.980 us  8030.915 us   124.39       0.7%    0.073s  ok 2.9e-16       1.63x
   L13_rader                      0.983 us  8056.362 us   124.00       3.1%    0.663s  ok 4.0e-16       1.63x
   mkl2026_dfti                   1.263 us 10344.101 us    96.58       0.5%    0.024s  ok 3.2e-16       2.09x
   fftw3_measure                  1.266 us 10371.175 us    96.32       0.3%    0.017s  ok 3.2e-16       2.10x
   mkl_dfti                       1.267 us 10381.927 us    96.22       0.1%    0.035s  ok 3.2e-16       2.10x
   fftw3_estimate                 1.268 us 10389.498 us    96.15       1.5%    0.002s  ok 3.2e-16       2.10x
   ducc0_c2c                      3.344 us 27396.646 us    36.46       1.2%    0.000s  ok 2.5e-16       5.55x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 5.976 us     5.976 us    50.41       1.8%    0.474s  ok 3.3e-16       1.00x
   L17_rader                      6.955 us     6.955 us    43.31      11.1%    0.582s  ok 3.2e-16       1.16x
   L17_winograd                   7.563 us     7.563 us    39.83       0.8%    0.609s  ok 3.2e-16       1.27x
   mkl_dfti                      23.148 us    23.148 us    13.01       1.4%    0.015s  ok 3.2e-16       3.87x
   mkl2026_dfti                  23.755 us    23.755 us    12.68       0.8%    0.008s  ok 3.2e-16       3.98x
   fftw3_patient                 29.583 us    29.583 us    10.18       3.6%    0.217s  ok 3.0e-16       4.95x
   fftw3_estimate                39.947 us    39.947 us     7.54      12.0%    0.008s  ok 3.0e-16       6.68x
   fftw3_measure                 46.040 us    46.040 us     6.54       7.8%    0.053s  ok 3.0e-16       7.70x
   ducc0_c2c                    103.334 us   103.334 us     2.92       0.5%    0.000s  ok 2.6e-16       17.29x
   baseline_matrix              538.233 us   538.233 us     0.56       0.0%    0.000s  ok 8.4e-16       90.07x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                 0.756 us   193.459 us   398.61       1.2%    0.621s  ok 3.3e-16       1.00x
   L17_rader                      0.796 us   203.754 us   378.46       2.0%    0.295s  ok 3.2e-16       1.05x
   L17_winograd                   0.823 us   210.797 us   365.82       0.7%    1.224s  ok 3.3e-16       1.09x
   fftw3_patient                  3.900 us   998.504 us    77.23       0.7%    0.328s  ok 3.0e-16       5.16x
   fftw3_estimate                 3.905 us   999.659 us    77.14       0.3%    0.004s  ok 3.0e-16       5.17x
   fftw3_measure                  3.972 us  1016.860 us    75.84       1.8%    0.013s  ok 3.0e-16       5.26x
   mkl_dfti                       4.842 us  1239.435 us    62.22       2.9%    0.032s  ok 3.1e-16       6.41x
   mkl2026_dfti                   4.924 us  1260.467 us    61.18       2.0%    0.023s  ok 3.1e-16       6.52x
   ducc0_c2c                      8.096 us  2072.462 us    37.21       5.4%    0.000s  ok 2.6e-16       10.71x
   baseline_matrix              540.582 us 138388.915 us     0.56       0.1%    0.000s  ok 8.4e-16       715.34x

-- L=17 (batched B=4096), volume 4913, working set 614.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                   1.219 us  4991.721 us   247.17       0.4%    2.339s  ok 3.3e-16       1.00x
   L17_rader                      1.289 us  5280.306 us   233.66       0.2%    5.522s  ok 3.2e-16       1.06x
   L17_matrixsimd                 2.911 us 11924.409 us   103.47       1.5%    0.132s  ok 3.3e-16       2.39x
   fftw3_estimate                 4.096 us 16777.912 us    73.54       0.3%    0.004s  ok 3.0e-16       3.36x
   fftw3_patient                  4.100 us 16793.076 us    73.47       1.5%    1.765s  ok 3.0e-16       3.36x
   fftw3_measure                  4.101 us 16796.711 us    73.46       0.2%    0.011s  ok 3.0e-16       3.36x
   mkl2026_dfti                   4.964 us 20333.641 us    60.68       1.3%    0.022s  ok 3.1e-16       4.07x
   mkl_dfti                       4.964 us 20333.863 us    60.68       0.7%    0.034s  ok 3.1e-16       4.07x
   ducc0_c2c                      7.988 us 32719.634 us    37.71       0.6%    0.000s  ok 2.6e-16       6.55x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     11.865 us    11.865 us    69.58       0.5%    0.084s  ok 3.8e-16       1.00x
   L23_matrixsimd                11.956 us    11.956 us    69.05       0.7%    0.335s  ok 3.8e-16       1.01x
   fftw3_patient                 38.890 us    38.890 us    21.23       1.9%    0.240s  ok 3.7e-16       3.28x
   mkl_dfti                      41.543 us    41.543 us    19.87       4.5%    0.013s  ok 4.1e-16       3.50x
   mkl2026_dfti                  43.706 us    43.706 us    18.89       1.7%    0.004s  ok 4.1e-16       3.68x
   fftw3_estimate                47.914 us    47.914 us    17.23      42.7%    0.008s  ok 3.7e-16       4.04x
   fftw3_measure                 52.516 us    52.516 us    15.72       2.1%    0.049s  ok 3.7e-16       4.43x
   ducc0_c2c                    293.004 us   293.004 us     2.82       0.9%    0.000s  ok 2.8e-16       24.69x
   baseline_matrix             1801.168 us  1801.168 us     0.46       0.0%    0.000s  ok 7.3e-16       151.80x

-- L=23 (batched B=128), volume 12167, working set 47.53 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                 2.268 us   290.246 us   364.08       0.6%    0.142s  ok 3.8e-16       1.00x
   L23_rader                      2.300 us   294.439 us   358.90       5.2%    0.113s  ok 3.8e-16       1.01x
   fftw3_estimate                12.446 us  1593.055 us    66.33      21.6%    0.004s  ok 3.7e-16       5.49x
   fftw3_patient                 12.456 us  1594.326 us    66.28       1.5%    0.317s  ok 3.7e-16       5.49x
   fftw3_measure                 13.149 us  1683.095 us    62.79      44.2%    0.013s  ok 3.7e-16       5.80x
   mkl_dfti                      14.604 us  1869.313 us    56.53       0.5%    0.016s  ok 4.2e-16       6.44x
   mkl2026_dfti                  15.733 us  2013.770 us    52.48       0.7%    0.023s  ok 4.2e-16       6.94x
   ducc0_c2c                     23.458 us  3002.614 us    35.19       1.1%    0.000s  ok 2.8e-16       10.35x
   baseline_matrix             1807.193 us 231320.693 us     0.46       0.0%    0.000s  ok 7.4e-16       796.98x

-- L=23 (batched B=2048), volume 12167, working set 760.44 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      5.810 us 11898.030 us   142.11       2.3%    0.482s  ok 3.8e-16       1.00x
   L23_matrixsimd                 5.893 us 12069.239 us   140.09       1.1%    0.626s  ok 3.8e-16       1.01x
   fftw3_patient                 12.529 us 25658.392 us    65.90       2.8%    1.391s  ok 3.7e-16       2.16x
   fftw3_measure                 12.810 us 26234.732 us    64.45       0.4%    0.012s  ok 3.7e-16       2.20x
   fftw3_estimate                12.855 us 26327.472 us    64.22      10.9%    0.004s  ok 3.7e-16       2.21x
   mkl_dfti                      14.700 us 30104.652 us    56.16       1.0%    0.036s  ok 4.2e-16       2.53x
   mkl2026_dfti                  15.715 us 32185.304 us    52.53       1.2%    0.025s  ok 4.2e-16       2.71x
   ducc0_c2c                     24.673 us 50529.691 us    33.46       0.8%    0.000s  ok 2.8e-16       4.25x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                23.027 us    23.027 us   157.13      12.3%    0.125s  ok 3.6e-16       1.00x
   L36_pfa                       25.720 us    25.720 us   140.67       0.8%    0.067s  ok 3.6e-16       1.12x
   L36_pencilfused               25.817 us    25.817 us   140.14       1.0%    0.249s  ok 3.6e-16       1.12x
   mkl_dfti                      38.107 us    38.107 us    94.95       0.5%    0.036s  ok 3.9e-16       1.65x
   mkl2026_dfti                  38.177 us    38.177 us    94.77       0.7%    0.031s  ok 4.1e-16       1.66x
   fftw3_measure                 64.270 us    64.270 us    56.30      19.6%    0.281s  ok 3.8e-16       2.79x
   fftw3_patient                 67.596 us    67.596 us    53.53       6.0%   10.100s  ok 3.9e-16       2.94x
   fftw3_estimate                82.857 us    82.857 us    43.67      19.0%    0.029s  ok 3.5e-16       3.60x
   ducc0_c2c                    213.611 us   213.611 us    16.94       3.2%    0.000s  ok 3.0e-16       9.28x
   baseline_matrix            10827.757 us 10827.757 us     0.33       0.4%    0.000s  ok 8.1e-16       470.22x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                        5.212 us   166.785 us   694.19       2.1%    0.103s  ok 3.6e-16       1.00x
   L36_mixedradix                 5.362 us   171.590 us   674.75       0.8%    0.376s  ok 3.6e-16       1.03x
   L36_pencilfused                5.493 us   175.765 us   658.72       2.5%    0.242s  ok 3.6e-16       1.05x
   mkl_dfti                       7.411 us   237.148 us   488.22       0.4%    0.015s  ok 3.9e-16       1.42x
   mkl2026_dfti                   7.833 us   250.656 us   461.91       0.6%    0.024s  ok 4.0e-16       1.50x
   fftw3_patient                 14.712 us   470.772 us   245.94      32.0%   11.077s  ok 3.9e-16       2.82x
   ducc0_c2c                     39.158 us  1253.046 us    92.40       0.9%    0.000s  ok 3.0e-16       7.51x
   fftw3_measure                 60.342 us  1930.941 us    59.96      24.6%    0.161s  ok 3.5e-16       11.58x
   fftw3_estimate                64.533 us  2065.058 us    56.07       6.6%    0.008s  ok 3.5e-16       12.38x
   baseline_matrix            11073.631 us 354356.196 us     0.33       0.1%    0.000s  ok 8.0e-16       2124.63x

-- L=36 (batched B=512), volume 46656, working set 729.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix                 9.896 us  5066.539 us   365.63       0.9%    3.017s  ok 3.6e-16       1.00x
   L36_pencilfused               10.861 us  5560.935 us   333.12       1.2%    4.013s  ok 3.6e-16       1.10x
   L36_pfa                       19.322 us  9892.690 us   187.26       0.7%    0.695s  ok 3.6e-16       1.95x
   fftw3_patient                 19.397 us  9931.020 us   186.53       4.8%   11.807s  ok 3.9e-16       1.96x
   mkl2026_dfti                  32.888 us 16838.715 us   110.01       0.3%    0.023s  ok 4.0e-16       3.32x
   mkl_dfti                      33.055 us 16924.020 us   109.46       0.3%    0.034s  ok 3.9e-16       3.34x
   fftw3_estimate                34.289 us 17555.771 us   105.52      47.2%    0.008s  ok 3.5e-16       3.47x
   fftw3_measure                 38.645 us 19786.050 us    93.63      40.4%    0.159s  ok 3.5e-16       3.91x
   ducc0_c2c                     58.404 us 29902.950 us    61.95       0.1%    0.000s  ok 3.0e-16       5.90x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       56.569 us    56.569 us   132.70       3.5%    0.225s  ok 4.0e-16       1.00x
   L45_mixedradix                58.798 us    58.798 us   127.67       1.3%    0.199s  ok 4.0e-16       1.04x
   mkl2026_dfti                  83.718 us    83.718 us    89.67       1.3%    0.028s  ok 4.4e-16       1.48x
   mkl_dfti                      86.296 us    86.296 us    86.99       0.5%    0.018s  ok 4.4e-16       1.53x
   fftw3_estimate               104.356 us   104.356 us    71.93       8.9%    0.023s  ok 4.2e-16       1.84x
   fftw3_measure                126.370 us   126.370 us    59.40      41.3%    0.243s  ok 4.2e-16       2.23x
   fftw3_patient                135.809 us   135.809 us    55.27      41.4%    4.402s  ok 4.3e-16       2.40x
   ducc0_c2c                    335.226 us   335.226 us    22.39       6.1%    0.000s  ok 3.7e-16       5.93x
   baseline_matrix            28742.192 us 28742.192 us     0.26      12.5%    0.000s  ok 8.0e-16       508.09x

-- L=45 (batched B=16), volume 91125, working set 44.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       14.742 us   235.872 us   509.20       3.0%    0.576s  ok 4.0e-16       1.00x
   L45_mixedradix                15.305 us   244.885 us   490.46       0.8%    0.253s  ok 4.1e-16       1.04x
   mkl2026_dfti                  37.526 us   600.422 us   200.04       1.4%    0.025s  ok 4.5e-16       2.55x
   mkl_dfti                      37.532 us   600.514 us   200.01       1.0%    0.036s  ok 4.4e-16       2.55x
   ducc0_c2c                     81.154 us  1298.457 us    92.50       1.6%    0.000s  ok 3.7e-16       5.50x
   fftw3_patient                 82.039 us  1312.624 us    91.50      42.6%    6.460s  ok 4.3e-16       5.56x
   fftw3_measure                 98.670 us  1578.713 us    76.08      16.0%    0.146s  ok 4.2e-16       6.69x
   fftw3_estimate                99.242 us  1587.869 us    75.64       0.4%    0.007s  ok 4.2e-16       6.73x
   baseline_matrix            32695.591 us 523129.459 us     0.23       2.1%    0.000s  ok 8.0e-16       2217.85x

-- L=45 (batched B=256), volume 91125, working set 711.91 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       26.763 us  6851.343 us   280.49       1.1%    3.890s  ok 4.0e-16       1.00x
   L45_mixedradix                47.687 us 12207.815 us   157.42       0.6%    2.035s  ok 4.1e-16       1.78x
   fftw3_patient                 59.326 us 15187.517 us   126.53       3.4%    5.805s  ok 4.3e-16       2.22x
   fftw3_measure                 80.758 us 20674.043 us    92.95      39.7%    0.082s  ok 4.2e-16       3.02x
   fftw3_estimate                84.491 us 21629.702 us    88.85       4.1%    0.004s  ok 4.2e-16       3.16x
   mkl_dfti                      91.962 us 23542.289 us    81.63       0.2%    0.034s  ok 4.4e-16       3.44x
   mkl2026_dfti                  92.018 us 23556.540 us    81.58       0.0%    0.024s  ok 4.5e-16       3.44x
   ducc0_c2c                    115.586 us 29589.959 us    64.94       2.6%    0.000s  ok 3.7e-16       4.32x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  128.691 us   128.691 us   183.33       1.2%    0.549s  ok 4.5e-16       1.00x
   L64_radix8                   136.342 us   136.342 us   173.04       0.6%    0.071s  ok 4.5e-16       1.06x
   mkl_dfti                     152.760 us   152.760 us   154.44       0.6%    0.016s  ok 3.4e-16       1.19x
   mkl2026_dfti                 156.364 us   156.364 us   150.88       1.4%    0.006s  ok 3.4e-16       1.22x
   fftw3_measure                190.751 us   190.751 us   123.68      16.2%    0.463s  ok 3.5e-16       1.48x
   fftw3_estimate               265.702 us   265.702 us    88.79       1.0%    0.004s  ok 3.5e-16       2.06x
   fftw3_patient                309.680 us   309.680 us    76.18       0.7%   15.397s  ok 3.5e-16       2.41x
   ducc0_c2c                    408.599 us   408.599 us    57.74      15.1%    0.000s  ok 3.0e-16       3.18x
   baseline_matrix           123000.585 us 123000.585 us     0.19       0.8%    0.000s  ok 7.8e-16       955.78x

-- L=64 (batched B=8), volume 262144, working set 64.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl_dfti                      72.366 us   578.930 us   326.02       0.4%    0.034s  ok 3.4e-16       1.00x
   L64_blocked                   77.127 us   617.013 us   305.90       0.8%    2.329s  ok 4.5e-16       1.07x
   mkl2026_dfti                  77.930 us   623.438 us   302.75       0.5%    0.004s  ok 3.4e-16       1.08x
   L64_radix8                    91.053 us   728.428 us   259.11       0.8%    0.485s  ok 4.5e-16       1.26x
   ducc0_c2c                    241.509 us  1932.069 us    97.69       9.0%    0.000s  ok 3.0e-16       3.34x
   fftw3_patient                288.771 us  2310.171 us    81.70      13.5%   29.168s  ok 3.5e-16       3.99x
   fftw3_measure                390.392 us  3123.136 us    60.43      17.0%    0.429s  ok 3.4e-16       5.39x
   fftw3_estimate               683.972 us  5471.774 us    34.49       0.2%    0.002s  ok 3.5e-16       9.45x

-- L=64 (batched B=128), volume 262144, working set 1024.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                   73.715 us  9435.503 us   320.06      23.8%    6.286s  ok 4.5e-16       1.00x
   L64_radix8                   142.323 us 18217.381 us   165.77       1.1%    2.155s  ok 4.5e-16       1.93x
   fftw3_patient                178.077 us 22793.795 us   132.49       7.4%   22.528s  ok 3.5e-16       2.42x
   mkl_dfti                     282.508 us 36160.997 us    83.51       1.6%    0.034s  ok 3.4e-16       3.83x
   mkl2026_dfti                 283.318 us 36264.680 us    83.27       0.3%    0.025s  ok 3.4e-16       3.84x
   fftw3_measure                286.637 us 36689.491 us    82.31      11.0%    0.306s  ok 3.5e-16       3.89x
   fftw3_estimate               305.782 us 39140.074 us    77.16       0.0%    0.002s  ok 3.5e-16       4.15x
   ducc0_c2c                    368.723 us 47196.591 us    63.99       0.4%    0.000s  ok 3.0e-16       5.00x

backends:
   L13_direct               conj-folded dense 13x13 per axis, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first; mt t1 g1; ab[B1]=t1g1:9576,t2g2:11008,t4g4:15404,t8g8:21767,t16g16:40449,t32g32:106539; gov{t1g1:11647,t2g2:10491,lock=t2g2}
   L13_rader                Rader-13 split cyc/nega (186 FP/pt), X-first, 512-bit, batch ntb=32/32 dsp=omp as=0, B1 pt=0 t1=1; fuse=1 um=7 ys=0 pf=0 pw=0 nts=0 znb=22 ab[B1]=i:6880,f0:6914,t2:12743,t4:12478,t8:11918,p2:11744,p4:14370,p8:13811 pick=i
   L17_matrixsimd           17-pt rotating-pass fused engine (adopted from L17_winograd, i4), mt[wg nt=32 static ws=614MB aggl3=44MB], sbw[rd/wr/cp/s17]=5.55/8.52/14.84/4.41, b1dec[yz/kyz/x/kx]=15.40/14.48/5.88/5.62, clk512/256=2.29/2.79 GHz, d256=2.79, anb=1
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; mt mode=1 nt=32, xl 512t dy, pf=0, pfw=0, clk256=3.89 clk512=2.89, par=1.119 us/t cut=eq spr=1.04x dsp=omp str=1 tn=2 tr=1 bw=213GB/s
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, cw=0, nt=0, mt[n=32 bt=32 dyn=0 t1=0 s2=0 ns=2 strm=1 ar=ot], clk256=2.89GHz, clk512=2.89GHz, p1=6.15 f23=10.86 fu=16.43 fu4=17.39, sbw[rd/wr/cp/s17]=6.44/11.39/18.41/6.06
   L23_matrixsimd           dense 23x23/axis conj-folded, pinned consts, X-first, spin-pool fused 512-bit T=16 nsock=2, nt=0 pf=0 pw=0 pt=0 dy=0, tune[pick=11.92(512-bit fused T=16) inc=12.94 us/t nv=1], clk512/256=2.29/2.79 GHz
   L23_rader                rader23 pool batchNT w4 team=32 pf=0 pw=0 bar=0 wt=0 ns=2, tuner pick=5.43 inc=6.33 us/t nv=640
   L36_mixedradix           MT PFA 4x9 n1_9 spin-pool; pick=v1-vol32-pfin (B=32 m=32 arena=32 stream=1 deep=0 pin=0 nb=1 7c) us/vol ser=255.8 pick=5.4 eff=1.47 dsp=2.13 gov{fr0=0,fr=0,nb=1,sc=9}
   L36_pencilfused          L=36 MT plane-fused y+z then strided x, PFA4x9; pick volpar-PIN team=32 pw=4 scratch+nt (B=512); probe us/vol t32=14.58 t16=23.69 is0=16.10 i1=-1.00 nt=14.58 pl=-1.00 gov{nb=1 fi=28 fo=50 nc=16}
   L36_pfa                  GT-PFA 4x9 (n1_9) spin-pool mt; pick: vols T=32 pw=4 inplace pf=0 (B=32 nv=32 nc=3 Tpool=32); us/vol@32 ip0=10.21 ip7=11.92
   L45_mixedradix           MT PFA9x5 pool; pick=vns32-v1-nx (B=256 nt=128 str=1 thr=32 nc=10); vns32-v2-nx=51.6 vns32-v2-m0=51.2 vns32-v2-pfnx=51.9 vnt32-v2-m0=52.7 vns32-v1-nx=50.5 grp16x2-v2-ppnx=68.3 grp16x2-v2-pp=67.6 grp16x2-v1-ppnx=67.9 grp16x2-v1-pp=68.2 vnt32-v2-pfpk=58.9 gov{nb=1,fr0=0,fr=0,sc=2,n=24}
   L45_pfa                  GT-PFA 9x5 pool T=32; pick=pw4-omtn-pfi (B=256 nv=128) mtv-pf0=39.2 mtv-pf3=34.4 omtn-pf0=29.8 omtn-pfi=27.8 mtn-pf0=30.5 mtn-pfi=27.8 g2-pf0=30.4 g2-pfw=29.4 mtf-rr=65.8 mtf-blk=38.6 mtf-bpf=35.4 ip-pf0=605.9 ip-pf3=471.0 mtv-pf0=42.9 mtf-blk=42.9 ip-pf0=652.0 gov{nb=1,c=48,i=16/16,o=16/16}
   L64_blocked              L64 8x8 split-sc two-stage, hugepage scratch; mt pick: eng=1(S:slab) G=1 nth=16 dyn=0 sb=0 mode=cached pf=0 st=3 pro=0 (B=1 nv=1)
   L64_radix8               radix-8^2/axis AVX-512 MT; pick[B=8]=gang-T32-g8-fused-pfw+slabpf1+sc0+p10; ser=1355us/vol pick=88 eff=0.48
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm; mt batch-split, per-thread NUMA scratch, 2D-raced@pool (8 kernels x T<=32); variant=fused_pf_d2 T=32 disp=pool fork=13.21us raceT1=0.4113 raceBest=0.0092 omp=0.0117 pool=0.0092us/vol exre=fused_pf_xa_d2,T=32 0.0157us/vol (plan fused_pf_d2,T=32)
   L6_unrolled              L=6: unrolled PFA 2x3 codelet ymm, batch-parallel contiguous chunks, per-thread NUMA-local scratch, pool-raced; variant=3pass_nt_pf nthr=32 disp=omp rd=pool od=53.1,53.2ns tm=1:635,2:392,4:200,8:108,16:82,24:58,32:53ns fr=0/0,nb=1
   L8_batchsimd             radix-8 split; pick[B=32768]: mode=FUSED nt=0 pf=s0w alloc=r8(a64,si512) mt{T=16 run=nt-s0 dyn=0} arena{T32/nt-s0=0.2094,T32/s0=0.2664,T32/none=0.2590,T32/nt=0.2077,T24/nt-s0=0.1912,T16/nt-s0=0.1770} gov{fr0=0,fr=0,nb=1,T16/nt-s0=0.1731,T32/nt-s0=0.2329,lock=cfg0}
   L8_fusedaxes             8^3 fused/AA/AA2 c52 mt=vol/pool; B=32768 pick=seq3-nt+pfs/nt16 (mt-tuned) arena{fused-nt+pfs/32=0.212,seq3-nt+pfs/32=0.202,seq3AA-nt+pfs/32=0.201,fused+pfs+pfw/32=0.265,fused-nt+pfs/24=0.189,fused-nt+pfs/16=0.177,seq3-nt+pfs/16=0.170,seq3AA-nt+pfs/16=0.170,seq3AA-nt+pfs/24=0.184} gov{fr0=0,fr=0,frmax=0,nb=1,nfar=16,fs=25,fcur=42,mw=120/0.84s,safe=seq3-nt+pfs/16:0.1721/16,wide=seq3AA-nt+pfs/32w:0.2050/161}
   L8_radix8                radix-8 52-instr codelet; B=1 serial, B>1 pool-parallel 1f (3p probes); pick[B=2048,T=32]=mt-1f-pfs (default,pool) arena{mt-1f-pfs=0.028 mt-1f=0.029 mt-1f-pfw=0.031 mt-1f520-pfs=0.028 mth-1f-pfs=0.085 mt-1f-nt-pfs=0.183 mt-3p-pfs*=0.029} govm{mt-1f-pfs=0.0278,mt-1f520-pfs=0.0279,mt-1f=0.0291,lock=mt-1f-pfs}
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 32 threads
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, threaded, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, threaded, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, threaded, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, threaded, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, GNU OpenMP threading, batched
```
