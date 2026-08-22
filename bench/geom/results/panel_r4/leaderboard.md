```
=== round panel_r4 ===
# round panel_r4
host: p55n3   date: 2026-08-21T20:51:52-04:00   slurm_job: 438480
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
cores: 64   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.219 us     0.219 us    38.32       2.9%    0.468s  ok 2.4e-16       1.00x
   L6_pfa                         0.219 us     0.219 us    38.26       5.0%    0.158s  ok 2.5e-16       1.00x
   mkl_dfti                       0.369 us     0.369 us    22.71       2.0%    0.004s  ok 2.4e-16       1.69x
   mkl2026_dfti                   0.402 us     0.402 us    20.85       1.2%    0.005s  ok 2.5e-16       1.84x
   fftw3_patient                  0.511 us     0.511 us    16.40       8.5%    0.029s  ok 2.0e-16       2.34x
   fftw3_measure                  0.557 us     0.557 us    15.03       9.8%    0.018s  ok 2.0e-16       2.55x
   fftw3_estimate                 1.179 us     1.179 us     7.10       0.4%    0.002s  ok 2.0e-16       5.39x
   ducc0_c2c                      5.135 us     5.135 us     1.63       0.5%    0.000s  ok 1.9e-16       23.50x
   baseline_matrix                8.387 us     8.387 us     1.00       0.0%    0.000s  ok 6.0e-16       38.38x

-- L=6 (batched B=64), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.214 us    13.684 us    39.17       3.5%    0.463s  ok 2.4e-16       1.00x
   L6_pfa                         0.222 us    14.215 us    37.71       2.8%    0.128s  ok 2.4e-16       1.04x
   mkl_dfti                       0.392 us    25.069 us    21.38       2.7%    0.004s  ok 2.4e-16       1.83x
   mkl2026_dfti                   0.429 us    27.436 us    19.54       2.5%    0.002s  ok 2.5e-16       2.00x
   fftw3_patient                  0.536 us    34.283 us    15.64       1.6%    0.030s  ok 2.0e-16       2.51x
   fftw3_measure                  0.575 us    36.789 us    14.57       0.5%    0.019s  ok 2.0e-16       2.69x
   fftw3_estimate                 1.212 us    77.595 us     6.91       0.3%    0.002s  ok 2.0e-16       5.67x
   ducc0_c2c                      2.773 us   177.485 us     3.02       1.7%    0.000s  ok 1.8e-16       12.97x
   baseline_matrix                8.380 us   536.305 us     1.00       0.0%    0.000s  ok 6.0e-16       39.19x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.387 us  1584.252 us    21.65       2.3%    0.378s  ok 2.4e-16       1.00x
   L6_unrolled                    0.397 us  1626.062 us    21.10       1.4%    0.419s  ok 2.4e-16       1.03x
   mkl_dfti                       0.557 us  2279.514 us    15.05       0.9%    0.002s  ok 2.4e-16       1.44x
   mkl2026_dfti                   0.585 us  2397.749 us    14.31       0.8%    0.003s  ok 2.5e-16       1.51x
   fftw3_measure                  0.656 us  2687.227 us    12.77       9.5%    0.015s  ok 2.0e-16       1.70x
   fftw3_patient                  0.720 us  2948.405 us    11.64      11.1%    0.026s  ok 2.0e-16       1.86x
   fftw3_estimate                 1.278 us  5236.726 us     6.55       1.0%    0.001s  ok 2.0e-16       3.31x
   ducc0_c2c                      2.830 us 11590.969 us     2.96       0.5%    0.000s  ok 1.8e-16       7.32x
   baseline_matrix                8.511 us 34859.379 us     0.98       0.3%    0.000s  ok 6.0e-16       22.00x

-- L=6 (batched B=32768), volume 216, working set 216.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.566 us 18550.399 us    14.79       1.6%    2.135s  ok 2.4e-16       1.00x
   L6_pfa                         0.570 us 18690.137 us    14.68       2.1%    1.956s  ok 2.4e-16       1.01x
   mkl_dfti                       0.707 us 23156.635 us    11.85       0.5%    0.002s  ok 2.4e-16       1.25x
   mkl2026_dfti                   0.724 us 23730.767 us    11.56       1.5%    0.002s  ok 2.5e-16       1.28x
   fftw3_measure                  0.808 us 26481.066 us    10.36       1.1%    0.015s  ok 2.0e-16       1.43x
   fftw3_patient                  0.906 us 29678.280 us     9.25       1.5%    0.025s  ok 2.0e-16       1.60x
   fftw3_estimate                 1.338 us 43858.245 us     6.26       0.4%    0.001s  ok 2.0e-16       2.36x
   ducc0_c2c                      3.298 us 108052.554 us     2.54       3.7%    0.000s  ok 1.8e-16       5.82x
   baseline_matrix                8.725 us 285893.984 us     0.96       0.4%    0.000s  ok 6.0e-16       15.41x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.570 us     0.570 us    40.42       2.0%    0.010s  ok 1.4e-16       1.00x
   L8_radix8                      0.570 us     0.570 us    40.41       6.6%    0.032s  ok 1.4e-16       1.00x
   L8_fusedaxes                   0.579 us     0.579 us    39.77       0.3%    0.000s  ok 2.2e-16       1.02x
   mkl_dfti                       0.654 us     0.654 us    35.24       1.2%    0.004s  ok 1.7e-16       1.15x
   mkl2026_dfti                   0.708 us     0.708 us    32.53       3.4%    0.004s  ok 1.7e-16       1.24x
   fftw3_patient                  1.175 us     1.175 us    19.61       2.6%    0.031s  ok 2.0e-16       2.06x
   fftw3_measure                  1.228 us     1.228 us    18.77       4.7%    0.019s  ok 2.0e-16       2.15x
   fftw3_estimate                 5.299 us     5.299 us     4.35       0.5%    0.001s  ok 1.8e-16       9.30x
   ducc0_c2c                      6.222 us     6.222 us     3.70       0.8%    0.000s  ok 1.4e-16       10.92x
   baseline_matrix               26.402 us    26.402 us     0.87       0.0%    0.000s  ok 3.9e-16       46.32x

-- L=8 (batched B=64), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.623 us    39.888 us    36.97       3.3%    0.000s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.665 us    42.529 us    34.67       2.6%    0.035s  ok 1.9e-16       1.07x
   L8_radix8                      0.680 us    43.511 us    33.89       1.6%    0.034s  ok 1.9e-16       1.09x
   mkl_dfti                       0.712 us    45.573 us    32.36       4.6%    0.003s  ok 1.6e-16       1.14x
   mkl2026_dfti                   0.772 us    49.403 us    29.85       3.0%    0.004s  ok 1.6e-16       1.24x
   fftw3_patient                  1.216 us    77.833 us    18.95       6.1%    0.027s  ok 1.8e-16       1.95x
   fftw3_measure                  1.279 us    81.860 us    18.01       0.9%    0.017s  ok 1.8e-16       2.05x
   ducc0_c2c                      4.003 us   256.189 us     5.76       5.9%    0.000s  ok 1.3e-16       6.42x
   fftw3_estimate                 5.433 us   347.694 us     4.24       0.3%    0.002s  ok 1.7e-16       8.72x
   baseline_matrix               26.425 us  1691.176 us     0.87       0.1%    0.000s  ok 3.9e-16       42.40x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_radix8                      1.136 us  2325.552 us    20.29       1.0%    0.255s  ok 1.9e-16       1.00x
   L8_batchsimd                   1.215 us  2489.278 us    18.96       1.9%    0.538s  ok 1.9e-16       1.07x
   L8_fusedaxes                   1.313 us  2689.280 us    17.55       2.7%    0.418s  ok 2.3e-16       1.16x
   mkl2026_dfti                   1.324 us  2712.403 us    17.40       2.6%    0.003s  ok 1.6e-16       1.17x
   mkl_dfti                       1.344 us  2753.135 us    17.14       0.1%    0.002s  ok 1.6e-16       1.18x
   fftw3_measure                  1.735 us  3553.081 us    13.28       0.7%    0.017s  ok 1.8e-16       1.53x
   fftw3_patient                  1.807 us  3699.900 us    12.75       2.2%    0.028s  ok 1.8e-16       1.59x
   ducc0_c2c                      4.559 us  9337.272 us     5.05       1.5%    0.000s  ok 1.3e-16       4.02x
   fftw3_estimate                 5.623 us 11515.671 us     4.10       0.1%    0.001s  ok 1.7e-16       4.95x
   baseline_matrix               26.722 us 54727.213 us     0.86       0.1%    0.000s  ok 3.9e-16       23.53x

-- L=8 (batched B=16384), volume 512, working set 256.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_radix8                      1.418 us 23229.808 us    16.25       3.8%    0.796s  ok 1.9e-16       1.00x
   L8_fusedaxes                   1.585 us 25972.456 us    14.53       0.9%    1.286s  ok 2.3e-16       1.12x
   L8_batchsimd                   1.642 us 26898.475 us    14.03       1.5%    1.192s  ok 1.9e-16       1.16x
   mkl2026_dfti                   1.804 us 29555.623 us    12.77       0.3%    0.002s  ok 1.6e-16       1.27x
   mkl_dfti                       1.842 us 30172.000 us    12.51       1.2%    0.002s  ok 1.6e-16       1.30x
   fftw3_measure                  2.043 us 33479.940 us    11.28       1.7%    0.016s  ok 1.8e-16       1.44x
   fftw3_patient                  2.193 us 35931.086 us    10.51       0.5%    0.026s  ok 1.8e-16       1.55x
   ducc0_c2c                      5.710 us 93547.161 us     4.04       1.6%    0.000s  ok 1.3e-16       4.03x
   fftw3_estimate                 5.766 us 94475.783 us     4.00       0.2%    0.001s  ok 1.7e-16       4.07x
   baseline_matrix               27.036 us 442961.409 us     0.85       0.4%    0.000s  ok 3.9e-16       19.07x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                16.431 us    16.431 us    18.33       0.9%    0.141s  ok 3.3e-16       1.00x
   L17_rader                     17.742 us    17.742 us    16.98       0.6%    0.026s  ok 3.2e-16       1.08x
   L17_winograd                  18.325 us    18.325 us    16.44       1.2%    0.140s  ok 3.2e-16       1.12x
   fftw3_estimate                81.705 us    81.705 us     3.69       6.0%    0.004s  ok 3.0e-16       4.97x
   fftw3_patient                 81.713 us    81.713 us     3.69       6.3%    0.030s  ok 3.0e-16       4.97x
   fftw3_measure                 86.600 us    86.600 us     3.48       0.1%    0.012s  ok 3.0e-16       5.27x
   mkl_dfti                      98.829 us    98.829 us     3.05       6.7%    0.019s  ok 3.1e-16       6.01x
   mkl2026_dfti                 100.551 us   100.551 us     3.00       0.4%    0.062s  ok 3.1e-16       6.12x
   ducc0_c2c                    103.860 us   103.860 us     2.90       1.8%    0.000s  ok 2.6e-16       6.32x
   baseline_matrix              538.266 us   538.266 us     0.56       0.0%    0.000s  ok 8.4e-16       32.76x

-- L=17 (batched B=8), volume 4913, working set 1.20 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                18.008 us   144.061 us    16.73       2.2%    0.158s  ok 3.3e-16       1.00x
   L17_rader                     19.293 us   154.346 us    15.61       3.3%    0.036s  ok 3.2e-16       1.07x
   L17_winograd                  19.628 us   157.025 us    15.35       0.2%    0.173s  ok 3.3e-16       1.09x
   fftw3_patient                 81.957 us   655.654 us     3.68       5.8%    0.020s  ok 3.0e-16       4.55x
   fftw3_measure                 82.027 us   656.215 us     3.67       9.4%    0.014s  ok 3.0e-16       4.56x
   fftw3_estimate                82.030 us   656.242 us     3.67       6.7%    0.006s  ok 3.0e-16       4.56x
   mkl_dfti                      99.968 us   799.743 us     3.01       0.1%    0.020s  ok 3.1e-16       5.55x
   mkl2026_dfti                 100.621 us   804.966 us     2.99       0.1%    0.056s  ok 3.1e-16       5.59x
   ducc0_c2c                    101.331 us   810.651 us     2.97       2.9%    0.000s  ok 2.6e-16       5.63x
   baseline_matrix              538.745 us  4309.963 us     0.56       0.0%    0.000s  ok 8.4e-16       29.92x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                21.626 us  5536.377 us    13.93       0.8%    0.916s  ok 3.3e-16       1.00x
   L17_winograd                  24.032 us  6152.090 us    12.53       0.9%    1.220s  ok 3.3e-16       1.11x
   L17_rader                     25.202 us  6451.740 us    11.95       3.7%    0.296s  ok 3.2e-16       1.17x
   fftw3_estimate                83.497 us 21375.218 us     3.61       5.7%    0.002s  ok 3.0e-16       3.86x
   fftw3_patient                 83.910 us 21480.844 us     3.59       6.2%    0.019s  ok 3.0e-16       3.88x
   fftw3_measure                 88.211 us 22582.003 us     3.41       3.3%    0.010s  ok 3.0e-16       4.08x
   mkl_dfti                     101.412 us 25961.421 us     2.97       0.6%    0.036s  ok 3.1e-16       4.69x
   mkl2026_dfti                 101.689 us 26032.509 us     2.96       0.2%    0.035s  ok 3.1e-16       4.70x
   ducc0_c2c                    107.370 us 27486.821 us     2.81       1.7%    0.000s  ok 2.6e-16       4.96x
   baseline_matrix              540.403 us 138343.210 us     0.56       0.0%    0.000s  ok 8.4e-16       24.99x

-- L=17 (batched B=2048), volume 4913, working set 307.06 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                22.290 us 45649.611 us    13.51       1.7%    1.417s  ok 3.3e-16       1.00x
   L17_winograd                  24.221 us 49604.267 us    12.44       0.9%    1.822s  ok 3.3e-16       1.09x
   L17_rader                     25.704 us 52642.740 us    11.72       1.0%    0.439s  ok 3.2e-16       1.15x
   fftw3_patient                 84.054 us 172142.492 us     3.58       5.5%    0.019s  ok 3.0e-16       3.77x
   fftw3_measure                 84.124 us 172286.580 us     3.58       4.7%    0.009s  ok 3.0e-16       3.77x
   fftw3_estimate                88.649 us 181552.713 us     3.40       0.1%    0.002s  ok 3.0e-16       3.98x
   mkl_dfti                     101.795 us 208476.269 us     2.96       6.4%    0.054s  ok 3.1e-16       4.57x
   mkl2026_dfti                 102.102 us 209104.829 us     2.95       6.8%    0.056s  ok 3.1e-16       4.58x
   ducc0_c2c                    113.028 us 231480.916 us     2.67       1.2%    0.000s  ok 2.6e-16       5.07x
   baseline_matrix              543.528 us 1113146.250 us     0.55       0.0%    0.000s  ok 8.4e-16       24.38x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               119.021 us   119.021 us    30.40       1.5%    0.141s  ok 4.0e-16       1.00x
   L36_pfa                      121.866 us   121.866 us    29.69       2.7%    0.134s  ok 3.7e-16       1.02x
   L36_pencilfused              122.478 us   122.478 us    29.54       2.7%    0.068s  ok 3.8e-16       1.03x
   mkl_dfti                     162.171 us   162.171 us    22.31       1.8%    0.040s  ok 3.9e-16       1.36x
   mkl2026_dfti                 170.066 us   170.066 us    21.27       0.9%    0.040s  ok 4.0e-16       1.43x
   fftw3_patient                298.114 us   298.114 us    12.14       4.0%    1.137s  ok 3.8e-16       2.50x
   fftw3_measure                367.667 us   367.667 us     9.84       1.6%    0.160s  ok 3.8e-16       3.09x
   ducc0_c2c                    396.130 us   396.130 us     9.13       4.6%    0.000s  ok 3.0e-16       3.33x
   fftw3_estimate               453.254 us   453.254 us     7.98       0.6%    0.013s  ok 3.5e-16       3.81x
   baseline_matrix            10858.583 us 10858.583 us     0.33       0.1%    0.000s  ok 8.0e-16       91.23x

-- L=36 (batched B=4), volume 46656, working set 5.70 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               129.921 us   519.685 us    27.85       0.3%    0.181s  ok 4.0e-16       1.00x
   L36_pfa                      132.347 us   529.387 us    27.34       0.5%    0.231s  ok 3.7e-16       1.02x
   L36_pencilfused              132.360 us   529.439 us    27.34       5.5%    0.081s  ok 3.8e-16       1.02x
   mkl_dfti                     175.427 us   701.706 us    20.62       0.1%    0.039s  ok 3.9e-16       1.35x
   mkl2026_dfti                 184.238 us   736.951 us    19.64       0.2%    0.019s  ok 4.1e-16       1.42x
   fftw3_patient                310.371 us  1241.486 us    11.66       3.0%    1.144s  ok 3.9e-16       2.39x
   fftw3_measure                360.044 us  1440.177 us    10.05       5.3%    0.158s  ok 3.8e-16       2.77x
   ducc0_c2c                    431.399 us  1725.596 us     8.39       2.3%    0.000s  ok 3.0e-16       3.32x
   fftw3_estimate               454.589 us  1818.356 us     7.96       1.7%    0.011s  ok 3.5e-16       3.50x
   baseline_matrix            10846.198 us 43384.790 us     0.33       0.0%    0.000s  ok 8.0e-16       83.48x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      174.226 us  5575.246 us    20.77       2.9%    0.923s  ok 3.7e-16       1.00x
   L36_pencilfused              219.626 us  7028.041 us    16.47       3.3%    0.486s  ok 3.7e-16       1.26x
   L36_mixedradix               221.602 us  7091.249 us    16.33       1.7%    1.051s  ok 4.0e-16       1.27x
   mkl_dfti                     221.910 us  7101.124 us    16.30       0.2%    0.038s  ok 3.9e-16       1.27x
   mkl2026_dfti                 230.079 us  7362.529 us    15.73       0.1%    0.016s  ok 4.0e-16       1.32x
   fftw3_patient                361.456 us 11566.591 us    10.01       7.6%    1.133s  ok 3.6e-16       2.07x
   fftw3_measure                403.452 us 12910.468 us     8.97       4.2%    0.155s  ok 3.5e-16       2.32x
   fftw3_estimate               490.233 us 15687.465 us     7.38       0.4%    0.004s  ok 3.5e-16       2.81x
   ducc0_c2c                    586.106 us 18755.407 us     6.17       0.7%    0.000s  ok 3.0e-16       3.36x
   baseline_matrix            11064.150 us 354052.790 us     0.33       0.1%    0.000s  ok 8.0e-16       63.50x

-- L=36 (batched B=256), volume 46656, working set 364.50 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      218.899 us 56038.118 us    16.53       0.2%    1.905s  ok 3.7e-16       1.00x
   L36_mixedradix               228.743 us 58558.238 us    15.82       1.7%    1.285s  ok 4.0e-16       1.04x
   L36_pencilfused              242.231 us 62011.174 us    14.94       0.7%    1.001s  ok 3.7e-16       1.11x
   mkl_dfti                     248.867 us 63709.921 us    14.54       0.1%    0.054s  ok 3.9e-16       1.14x
   mkl2026_dfti                 256.835 us 65749.722 us    14.09       0.0%    0.054s  ok 4.0e-16       1.17x
   fftw3_patient                380.533 us 97416.575 us     9.51       8.1%    1.125s  ok 3.6e-16       1.74x
   fftw3_measure                415.561 us 106383.731 us     8.71       1.4%    0.153s  ok 3.5e-16       1.90x
   fftw3_estimate               510.707 us 130740.944 us     7.08       1.4%    0.004s  ok 3.5e-16       2.33x
   ducc0_c2c                    729.222 us 186680.816 us     4.96       0.6%    0.000s  ok 3.0e-16       3.33x
   baseline_matrix            11116.819 us 2845905.640 us     0.33       0.0%    0.000s  ok 8.0e-16       50.79x

backends:
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit, pinned, X-first, pf=0
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t, pf=0
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=a8, pf=0
   L36_mixedradix           PFA 4x9 2-sweep, lanes=lines; pick=v1-nt-pf1-pfin (B=256, arena=39 vol, ntpolicy=1, 12 cand)
   L36_pencilfused          L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes; tuner picked pw=4 mode=inplace (B=1)
   L36_pfa                  GT-PFA 4x9 two-sweep; tuner pick: pw=4 mode=inplace pf=1 (B=32, nv=32)
   L6_pfa                   Good-Thomas PFA 2x3 per axis, unrolled, 2 complex/ymm interleaved, no twiddles, plan-time raced kernels; variant=fused
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), AVX2 2-complex lanes; variant=fused_pf clk=3.89GHz
   L8_batchsimd             radix-8 split, 3 structures as candidates; pick[B=64]: mode=LANEX2S nt=0 pf=t0
   L8_fusedaxes             8^3 fused x/y/z in L1, lanes=z; B=1 pick=plain (rule)
   L8_radix8                radix-8 split-complex 52-instr codelet, copy-free transposes; 3-pass writes the volume sequentially; tuner pick[B=64]=avx512-3p (default)
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
