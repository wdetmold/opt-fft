```
=== round panel_r2 ===
# round panel_r2
host: p55n3   date: 2026-08-21T18:46:15-04:00   slurm_job: 438476
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
cores: 64   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.218 us     0.218 us    38.35       4.1%    0.289s  ok 2.5e-16       1.00x
   L6_pfa                         0.221 us     0.221 us    37.95       1.9%    0.252s  ok 2.4e-16       1.01x
   mkl_dfti                       0.369 us     0.369 us    22.67       1.1%    0.005s  ok 2.4e-16       1.69x
   mkl2026_dfti                   0.406 us     0.406 us    20.62       0.9%    0.005s  ok 2.5e-16       1.86x
   fftw3_patient                  0.508 us     0.508 us    16.50       3.3%    0.029s  ok 1.9e-16       2.32x
   fftw3_measure                  0.550 us     0.550 us    15.24       5.8%    0.019s  ok 1.9e-16       2.52x
   fftw3_estimate                 1.176 us     1.176 us     7.12       2.3%    0.001s  ok 1.9e-16       5.39x
   ducc0_c2c                      3.818 us     3.818 us     2.19      35.0%    0.000s  ok 1.7e-16       17.48x
   baseline_matrix                8.387 us     8.387 us     1.00       0.0%    0.000s  ok 6.0e-16       38.41x

-- L=6 (batched B=64), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.214 us    13.677 us    39.19       3.6%    0.311s  ok 2.4e-16       1.00x
   L6_pfa                         0.223 us    14.300 us    37.49       2.7%    0.079s  ok 2.4e-16       1.05x
   mkl_dfti                       0.392 us    25.062 us    21.39       0.1%    0.004s  ok 2.4e-16       1.83x
   mkl2026_dfti                   0.440 us    28.130 us    19.06       0.5%    0.004s  ok 2.5e-16       2.06x
   fftw3_patient                  0.543 us    34.757 us    15.42       0.6%    0.029s  ok 2.0e-16       2.54x
   fftw3_measure                  0.559 us    35.786 us    14.98       2.4%    0.017s  ok 2.0e-16       2.62x
   fftw3_estimate                 1.212 us    77.539 us     6.91       1.0%    0.002s  ok 2.0e-16       5.67x
   ducc0_c2c                      2.585 us   165.415 us     3.24       7.4%    0.000s  ok 1.8e-16       12.09x
   baseline_matrix                8.379 us   536.282 us     1.00       0.0%    0.000s  ok 5.9e-16       39.21x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.384 us  1574.027 us    21.79       0.7%    0.237s  ok 2.4e-16       1.00x
   L6_pfa                         0.393 us  1609.500 us    21.31       0.5%    0.336s  ok 2.4e-16       1.02x
   mkl_dfti                       0.548 us  2243.670 us    15.29       0.6%    0.003s  ok 2.4e-16       1.43x
   mkl2026_dfti                   0.572 us  2344.638 us    14.63       0.3%    0.002s  ok 2.5e-16       1.49x
   fftw3_measure                  0.654 us  2677.580 us    12.81       8.4%    0.015s  ok 2.0e-16       1.70x
   fftw3_patient                  0.724 us  2963.467 us    11.58       0.4%    0.026s  ok 2.0e-16       1.88x
   fftw3_estimate                 1.273 us  5214.430 us     6.58       0.8%    0.002s  ok 2.0e-16       3.31x
   ducc0_c2c                      2.816 us 11535.540 us     2.97       2.2%    0.000s  ok 1.8e-16       7.33x
   baseline_matrix                8.530 us 34939.109 us     0.98       0.2%    0.000s  ok 6.0e-16       22.20x

-- L=6 (batched B=32768), volume 216, working set 216.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.572 us 18734.050 us    14.65       0.4%    1.129s  ok 2.4e-16       1.00x
   L6_pfa                         0.616 us 20180.771 us    13.60       1.1%    0.316s  ok 2.4e-16       1.08x
   mkl_dfti                       0.684 us 22413.260 us    12.24       0.3%    0.002s  ok 2.4e-16       1.20x
   mkl2026_dfti                   0.699 us 22890.347 us    11.99       1.5%    0.002s  ok 2.5e-16       1.22x
   fftw3_measure                  0.774 us 25349.571 us    10.83       2.0%    0.015s  ok 2.0e-16       1.35x
   fftw3_patient                  0.902 us 29560.857 us     9.28       0.6%    0.025s  ok 2.0e-16       1.58x
   fftw3_estimate                 1.346 us 44121.404 us     6.22       0.3%    0.001s  ok 2.0e-16       2.36x
   ducc0_c2c                      3.203 us 104962.408 us     2.61       1.7%    0.000s  ok 1.8e-16       5.60x
   baseline_matrix                8.693 us 284844.657 us     0.96       0.2%    0.000s  ok 6.0e-16       15.20x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.573 us     0.573 us    40.22       1.2%    0.000s  ok 2.3e-16       1.00x
   L8_radix8                      0.583 us     0.583 us    39.54       1.3%    0.163s  ok 1.3e-16       1.02x
   L8_batchsimd                   0.598 us     0.598 us    38.53       0.8%    0.033s  ok 2.0e-16       1.04x
   mkl_dfti                       0.651 us     0.651 us    35.39       1.3%    0.004s  ok 1.6e-16       1.14x
   mkl2026_dfti                   0.707 us     0.707 us    32.59       5.2%    0.003s  ok 1.6e-16       1.23x
   fftw3_patient                  1.179 us     1.179 us    19.54       2.2%    0.027s  ok 1.7e-16       2.06x
   fftw3_measure                  1.266 us     1.266 us    18.20       1.5%    0.019s  ok 1.7e-16       2.21x
   fftw3_estimate                 5.298 us     5.298 us     4.35       0.0%    0.002s  ok 1.7e-16       9.25x
   ducc0_c2c                      6.289 us     6.289 us     3.66       3.9%    0.000s  ok 1.3e-16       10.98x
   baseline_matrix               26.404 us    26.404 us     0.87       0.0%    0.000s  ok 4.0e-16       46.09x

-- L=8 (batched B=64), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.636 us    40.724 us    36.21       2.8%    0.046s  ok 1.9e-16       1.00x
   L8_fusedaxes                   0.638 us    40.826 us    36.12       5.8%    0.000s  ok 2.3e-16       1.00x
   mkl_dfti                       0.687 us    43.995 us    33.52       5.4%    0.004s  ok 1.6e-16       1.08x
   L8_radix8                      0.706 us    45.169 us    32.65       3.4%    0.175s  ok 1.3e-16       1.11x
   mkl2026_dfti                   0.784 us    50.155 us    29.40       4.5%    0.002s  ok 1.6e-16       1.23x
   fftw3_patient                  1.216 us    77.812 us    18.95       6.1%    0.028s  ok 1.8e-16       1.91x
   fftw3_measure                  1.277 us    81.755 us    18.04       1.6%    0.019s  ok 1.8e-16       2.01x
   ducc0_c2c                      4.011 us   256.681 us     5.74       0.4%    0.000s  ok 1.3e-16       6.30x
   fftw3_estimate                 5.437 us   347.958 us     4.24       0.2%    0.003s  ok 1.7e-16       8.54x
   baseline_matrix               26.425 us  1691.188 us     0.87       0.1%    0.000s  ok 3.9e-16       41.53x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   1.205 us  2468.771 us    19.11       1.4%    0.136s  ok 1.9e-16       1.00x
   mkl_dfti                       1.325 us  2713.282 us    17.39       0.6%    0.002s  ok 1.6e-16       1.10x
   mkl2026_dfti                   1.328 us  2720.711 us    17.34       0.5%    0.002s  ok 1.6e-16       1.10x
   L8_fusedaxes                   1.503 us  3078.524 us    15.33       0.2%    0.000s  ok 2.3e-16       1.25x
   L8_radix8                      1.526 us  3124.987 us    15.10       7.8%    0.271s  ok 1.3e-16       1.27x
   fftw3_measure                  1.694 us  3469.587 us    13.60       1.2%    0.017s  ok 1.8e-16       1.41x
   fftw3_patient                  1.782 us  3649.190 us    12.93       1.8%    0.028s  ok 1.8e-16       1.48x
   ducc0_c2c                      4.550 us  9317.947 us     5.06       0.5%    0.000s  ok 1.3e-16       3.77x
   fftw3_estimate                 5.623 us 11516.007 us     4.10       0.2%    0.001s  ok 1.7e-16       4.66x
   baseline_matrix               26.726 us 54735.270 us     0.86       0.1%    0.000s  ok 3.9e-16       22.17x

-- L=8 (batched B=16384), volume 512, working set 256.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   1.557 us 25514.526 us    14.79       0.3%    0.128s  ok 1.9e-16       1.00x
   L8_fusedaxes                   1.614 us 26442.034 us    14.28       0.4%    0.000s  ok 2.3e-16       1.04x
   L8_radix8                      1.778 us 29123.422 us    12.96       0.4%    0.571s  ok 1.3e-16       1.14x
   mkl2026_dfti                   1.784 us 29226.547 us    12.92       0.2%    0.002s  ok 1.6e-16       1.15x
   mkl_dfti                       1.826 us 29913.673 us    12.62       0.7%    0.003s  ok 1.6e-16       1.17x
   fftw3_measure                  1.964 us 32181.687 us    11.73       3.5%    0.016s  ok 1.8e-16       1.26x
   fftw3_patient                  2.173 us 35601.326 us    10.60       0.8%    0.027s  ok 1.8e-16       1.40x
   ducc0_c2c                      5.617 us 92021.199 us     4.10       0.6%    0.000s  ok 1.3e-16       3.61x
   fftw3_estimate                 5.780 us 94695.759 us     3.99       0.0%    0.001s  ok 1.7e-16       3.71x
   baseline_matrix               27.074 us 443585.233 us     0.85       0.3%    0.000s  ok 3.9e-16       17.39x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                16.751 us    16.751 us    17.98       2.0%    0.063s  ok 3.3e-16       1.00x
   L17_winograd                  18.247 us    18.247 us    16.51       0.7%    0.114s  ok 3.3e-16       1.09x
   L17_rader                     19.212 us    19.212 us    15.68       1.0%    0.000s  ok 3.1e-16       1.15x
   fftw3_measure                 81.761 us    81.761 us     3.68       5.9%    0.017s  ok 3.0e-16       4.88x
   fftw3_estimate                81.764 us    81.764 us     3.68       5.9%    0.006s  ok 3.0e-16       4.88x
   fftw3_patient                 81.832 us    81.832 us     3.68       5.9%    0.027s  ok 3.0e-16       4.89x
   mkl_dfti                      98.808 us    98.808 us     3.05       0.1%    0.043s  ok 3.1e-16       5.90x
   mkl2026_dfti                 100.572 us   100.572 us     3.00       4.3%    0.019s  ok 3.1e-16       6.00x
   ducc0_c2c                    104.058 us   104.058 us     2.89       1.3%    0.000s  ok 2.6e-16       6.21x
   baseline_matrix              538.267 us   538.267 us     0.56       0.0%    0.000s  ok 8.4e-16       32.13x

-- L=17 (batched B=8), volume 4913, working set 1.20 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                18.661 us   149.287 us    16.14       2.5%    0.070s  ok 3.3e-16       1.00x
   L17_winograd                  20.211 us   161.688 us    14.90       0.4%    0.117s  ok 3.3e-16       1.08x
   L17_rader                     20.365 us   162.923 us    14.79       0.7%    0.000s  ok 3.1e-16       1.09x
   fftw3_measure                 81.951 us   655.608 us     3.68      13.4%    0.016s  ok 3.0e-16       4.39x
   fftw3_estimate                82.027 us   656.216 us     3.67       5.8%    0.004s  ok 3.0e-16       4.40x
   fftw3_patient                 82.237 us   657.893 us     3.66       5.5%    0.031s  ok 3.0e-16       4.41x
   mkl_dfti                     100.123 us   800.986 us     3.01       6.9%    0.039s  ok 3.1e-16       5.37x
   ducc0_c2c                    100.425 us   803.400 us     3.00       1.0%    0.000s  ok 2.6e-16       5.38x
   mkl2026_dfti                 106.110 us   848.878 us     2.84       1.6%    0.020s  ok 3.1e-16       5.69x
   baseline_matrix              538.551 us  4308.409 us     0.56       0.0%    0.000s  ok 8.4e-16       28.86x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                  24.031 us  6151.932 us    12.53       0.7%    0.829s  ok 3.3e-16       1.00x
   L17_rader                     24.394 us  6244.919 us    12.35       1.8%    0.000s  ok 3.2e-16       1.02x
   L17_matrixsimd                26.067 us  6673.084 us    11.56       0.6%    0.144s  ok 3.3e-16       1.08x
   fftw3_measure                 83.412 us 21353.455 us     3.61       5.7%    0.010s  ok 3.0e-16       3.47x
   fftw3_estimate                83.462 us 21366.158 us     3.61       5.8%    0.002s  ok 3.0e-16       3.47x
   fftw3_patient                 83.691 us 21424.819 us     3.60       5.6%    0.020s  ok 3.0e-16       3.48x
   mkl_dfti                     103.293 us 26443.039 us     2.92       6.4%    0.039s  ok 3.1e-16       4.30x
   mkl2026_dfti                 103.682 us 26542.560 us     2.91       0.2%    0.016s  ok 3.1e-16       4.31x
   ducc0_c2c                    106.853 us 27354.458 us     2.82       0.5%    0.000s  ok 2.6e-16       4.45x
   baseline_matrix              540.358 us 138331.625 us     0.56       0.0%    0.000s  ok 8.4e-16       22.49x

-- L=17 (batched B=2048), volume 4913, working set 307.06 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                  24.603 us 50386.824 us    12.24       1.6%    1.234s  ok 3.3e-16       1.00x
   L17_rader                     26.635 us 54549.254 us    11.31       0.4%    0.000s  ok 3.2e-16       1.08x
   L17_matrixsimd                27.266 us 55840.962 us    11.05       0.5%    0.182s  ok 3.3e-16       1.11x
   fftw3_patient                 84.077 us 172190.117 us     3.58       5.5%    0.019s  ok 3.0e-16       3.42x
   fftw3_estimate                88.544 us 181338.636 us     3.40       0.5%    0.002s  ok 3.0e-16       3.60x
   fftw3_measure                 88.642 us 181538.375 us     3.40       0.2%    0.009s  ok 3.0e-16       3.60x
   mkl_dfti                     101.753 us 208390.940 us     2.96       6.5%    0.054s  ok 3.1e-16       4.14x
   mkl2026_dfti                 102.124 us 209150.070 us     2.95       6.8%    0.056s  ok 3.1e-16       4.15x
   ducc0_c2c                    114.844 us 235201.172 us     2.62       2.0%    0.000s  ok 2.6e-16       4.67x
   baseline_matrix              543.959 us 1114027.270 us     0.55       0.0%    0.000s  ok 8.4e-16       22.11x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      119.266 us   119.266 us    30.34       3.9%    0.042s  ok 3.7e-16       1.00x
   L36_mixedradix               120.322 us   120.322 us    30.07       1.3%    0.071s  ok 4.0e-16       1.01x
   L36_pencilfused              125.066 us   125.066 us    28.93       5.1%    0.063s  ok 3.9e-16       1.05x
   mkl_dfti                     162.396 us   162.396 us    22.28       1.1%    0.019s  ok 3.9e-16       1.36x
   mkl2026_dfti                 171.236 us   171.236 us    21.13       0.7%    0.056s  ok 4.1e-16       1.44x
   fftw3_patient                306.492 us   306.492 us    11.80       0.7%    1.153s  ok 3.8e-16       2.57x
   fftw3_measure                360.060 us   360.060 us    10.05       7.6%    0.160s  ok 3.8e-16       3.02x
   ducc0_c2c                    395.135 us   395.135 us     9.16       1.0%    0.000s  ok 3.1e-16       3.31x
   fftw3_estimate               450.149 us   450.149 us     8.04       1.2%    0.012s  ok 3.5e-16       3.77x
   baseline_matrix            10857.469 us 10857.469 us     0.33       0.3%    0.000s  ok 8.0e-16       91.04x

-- L=36 (batched B=4), volume 46656, working set 5.70 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      128.460 us   513.839 us    28.17       2.6%    0.064s  ok 3.7e-16       1.00x
   L36_mixedradix               129.745 us   518.980 us    27.89       2.2%    0.108s  ok 4.0e-16       1.01x
   L36_pencilfused              151.519 us   606.075 us    23.88       0.2%    0.084s  ok 3.8e-16       1.18x
   mkl_dfti                     174.138 us   696.551 us    20.78       1.1%    0.036s  ok 3.9e-16       1.36x
   mkl2026_dfti                 183.705 us   734.819 us    19.70       0.7%    0.055s  ok 4.0e-16       1.43x
   fftw3_patient                311.906 us  1247.624 us    11.60       2.7%    1.127s  ok 3.9e-16       2.43x
   fftw3_measure                367.372 us  1469.488 us     9.85       6.7%    0.159s  ok 3.8e-16       2.86x
   ducc0_c2c                    431.881 us  1727.524 us     8.38       0.1%    0.000s  ok 3.0e-16       3.36x
   fftw3_estimate               453.681 us  1814.725 us     7.98       1.0%    0.012s  ok 3.5e-16       3.53x
   baseline_matrix            10864.506 us 43458.023 us     0.33       0.1%    0.000s  ok 8.0e-16       84.58x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      202.746 us  6487.875 us    17.85       2.2%    0.271s  ok 3.7e-16       1.00x
   L36_mixedradix               231.373 us  7403.943 us    15.64       0.8%    0.116s  ok 4.0e-16       1.14x
   L36_pencilfused              241.425 us  7725.610 us    14.99       1.5%    0.244s  ok 3.8e-16       1.19x
   mkl_dfti                     259.771 us  8312.681 us    13.93       0.3%    0.016s  ok 3.9e-16       1.28x
   mkl2026_dfti                 268.035 us  8577.116 us    13.50       0.8%    0.058s  ok 4.0e-16       1.32x
   fftw3_patient                376.857 us 12059.420 us     9.60       1.7%    1.136s  ok 3.9e-16       1.86x
   fftw3_measure                399.719 us 12791.006 us     9.05       0.9%    0.157s  ok 3.5e-16       1.97x
   fftw3_estimate               483.075 us 15458.386 us     7.49       1.0%    0.004s  ok 3.5e-16       2.38x
   ducc0_c2c                    574.743 us 18391.772 us     6.30       2.2%    0.000s  ok 3.0e-16       2.83x
   baseline_matrix            11076.330 us 354442.557 us     0.33       0.0%    0.000s  ok 8.0e-16       54.63x

-- L=36 (batched B=256), volume 46656, working set 364.50 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      238.796 us 61131.693 us    15.15       2.0%    0.553s  ok 3.7e-16       1.00x
   L36_mixedradix               261.463 us 66934.423 us    13.84       0.6%    0.109s  ok 4.0e-16       1.09x
   L36_pencilfused              283.910 us 72680.991 us    12.74       1.3%    0.239s  ok 3.8e-16       1.19x
   mkl_dfti                     307.771 us 78789.366 us    11.76       0.7%    0.057s  ok 3.9e-16       1.29x
   mkl2026_dfti                 318.020 us 81413.064 us    11.38       0.1%    0.056s  ok 4.0e-16       1.33x
   fftw3_patient                396.054 us 101389.743 us     9.14       0.3%    1.127s  ok 3.6e-16       1.66x
   fftw3_measure                417.088 us 106774.463 us     8.67       3.6%    0.154s  ok 3.6e-16       1.75x
   fftw3_estimate               504.644 us 129188.745 us     7.17       0.3%    0.004s  ok 3.5e-16       2.11x
   ducc0_c2c                    707.323 us 181074.635 us     5.12       0.5%    0.000s  ok 3.0e-16       2.96x
   baseline_matrix            11099.202 us 2841395.670 us     0.33       0.0%    0.000s  ok 8.0e-16       46.48x

backends:
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit
   L17_rader                Rader-17 in cyclic/negacyclic form (kernel adopted from L17_winograd): 296 FP instr per 17-pt DFT, split re/im, L1-resident plane fusion, 4 L1 crossings per volume
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=c8, pf=0
   L36_mixedradix           row-column PFA 4x9 line codelet, batch-vectorised over lines, L1-blocked planes, 2 passes over the volume, AVX2/AVX-512 autotuned
   L36_pencilfused          L=36: two-pass pencil-fused row-column, PFA 4x9 line kernel, interleaved-complex SIMD over the spectator axis, register transposes, in-place/NT modes self-tuned
   L36_pfa                  Good-Thomas PFA 4x9, interleaved-complex lanes, two sweeps (z+y fused per x-plane, x in place or via reused scratch + NT stores), variant autotuned in create()
   L6_pfa                   Good-Thomas PFA 2x3 per axis, unrolled, 2 complex/ymm interleaved, no twiddles, self-tuned pass fusion; variant=6
   L6_unrolled              L=6: unrolled straight-line PFA 2x3 six-point codelet (48 flops/36 instr, no twiddles) on all three axes, AVX2/FMA 2-complex lanes, in-register z-pencil transposes; variant=fused_pf
   L8_batchsimd             split-complex radix-8 (DFT8 (x) I_8), non-destructive vshuff64x2 8x8 repack, lane-per-x default, NT stores at large batch
   L8_fusedaxes             8^3 fused x/y/z in L1, split-complex, spatial axis in the SIMD lanes
   L8_radix8                radix-8 codelet, 52 instrs (44 add + 8 FMA, no twiddle table), split-complex 8 lines per vector; copy-free immediate-controlled 8x8 transposes (AVX-512); next-volume prefetch + NT stores in the streaming regime; backend and store type chosen by self-timing at setup
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
