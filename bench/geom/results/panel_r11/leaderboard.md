```
=== round panel_r11 ===
# round panel_r11
host: p55n3   date: 2026-08-22T08:03:31-04:00   slurm_job: 438529
cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
isa: avx2 avx512_vnni avx512bw avx512cd avx512dq avx512f avx512vl fma 
cores: 64   governor: powersave
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.208 us     0.208 us    40.35      10.1%    0.440s  ok 2.4e-16       1.00x
   L6_unrolled                    0.209 us     0.209 us    40.04       4.2%    0.516s  ok 2.4e-16       1.01x
   mkl_dfti                       0.371 us     0.371 us    22.57       7.0%    0.004s  ok 2.4e-16       1.79x
   mkl2026_dfti                   0.397 us     0.397 us    21.08       3.5%    0.004s  ok 2.5e-16       1.91x
   fftw3_patient                  0.525 us     0.525 us    15.97       3.5%    0.029s  ok 1.9e-16       2.53x
   fftw3_measure                  0.553 us     0.553 us    15.14      10.6%    0.017s  ok 1.9e-16       2.67x
   fftw3_estimate                 1.176 us     1.176 us     7.12       0.2%    0.002s  ok 1.9e-16       5.67x
   ducc0_c2c                      4.846 us     4.846 us     1.73       6.6%    0.000s  ok 1.7e-16       23.35x
   baseline_matrix                8.388 us     8.388 us     1.00       0.2%    0.000s  ok 6.2e-16       40.41x

-- L=6 (batched B=64), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_pfa                         0.220 us    14.079 us    38.07       0.2%    0.451s  ok 2.4e-16       1.00x
   L6_unrolled                    0.221 us    14.150 us    37.88       0.2%    0.526s  ok 2.4e-16       1.01x
   mkl_dfti                       0.401 us    25.641 us    20.91       0.7%    0.004s  ok 2.4e-16       1.82x
   mkl2026_dfti                   0.437 us    27.981 us    19.16       0.3%    0.005s  ok 2.5e-16       1.99x
   fftw3_patient                  0.544 us    34.825 us    15.39       9.1%    0.027s  ok 2.0e-16       2.47x
   fftw3_measure                  0.570 us    36.473 us    14.70       1.2%    0.018s  ok 2.0e-16       2.59x
   fftw3_estimate                 1.215 us    77.753 us     6.89       0.2%    0.003s  ok 2.0e-16       5.52x
   ducc0_c2c                      2.779 us   177.826 us     3.01       0.3%    0.000s  ok 1.8e-16       12.63x
   baseline_matrix                8.380 us   536.302 us     1.00       0.0%    0.000s  ok 6.0e-16       38.09x

-- L=6 (batched B=4096), volume 216, working set 27.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.394 us  1612.474 us    21.27       2.7%    0.620s  ok 2.4e-16       1.00x
   L6_pfa                         0.394 us  1612.946 us    21.27       3.5%    0.584s  ok 2.4e-16       1.00x
   mkl_dfti                       0.562 us  2301.035 us    14.91       1.5%    0.002s  ok 2.4e-16       1.43x
   mkl2026_dfti                   0.582 us  2382.598 us    14.40       1.3%    0.002s  ok 2.5e-16       1.48x
   fftw3_measure                  0.659 us  2700.474 us    12.70       8.0%    0.015s  ok 2.0e-16       1.67x
   fftw3_patient                  0.716 us  2932.621 us    11.70       1.2%    0.026s  ok 2.0e-16       1.82x
   fftw3_estimate                 1.278 us  5235.023 us     6.55       0.1%    0.001s  ok 2.0e-16       3.25x
   ducc0_c2c                      2.822 us 11557.628 us     2.97       2.5%    0.000s  ok 1.8e-16       7.17x
   baseline_matrix                8.505 us 34838.473 us     0.98       0.2%    0.000s  ok 6.0e-16       21.61x

-- L=6 (batched B=32768), volume 216, working set 216.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.563 us 18434.118 us    14.89       0.6%    1.924s  ok 2.4e-16       1.00x
   L6_pfa                         0.565 us 18528.782 us    14.81       1.0%    2.240s  ok 2.4e-16       1.01x
   mkl_dfti                       0.707 us 23162.700 us    11.85       2.7%    0.002s  ok 2.4e-16       1.26x
   mkl2026_dfti                   0.722 us 23670.494 us    11.59       0.8%    0.002s  ok 2.5e-16       1.28x
   fftw3_measure                  0.806 us 26398.194 us    10.40       1.0%    0.015s  ok 2.0e-16       1.43x
   fftw3_patient                  0.912 us 29877.901 us     9.19       0.9%    0.025s  ok 2.0e-16       1.62x
   fftw3_estimate                 1.343 us 43995.407 us     6.24       0.6%    0.001s  ok 2.0e-16       2.39x
   ducc0_c2c                      3.255 us 106671.933 us     2.57       1.3%    0.000s  ok 1.8e-16       5.79x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.551 us     0.551 us    41.84       1.4%    0.007s  ok 2.3e-16       1.00x
   L8_fusedaxes                   0.555 us     0.555 us    41.50       0.1%    0.078s  ok 2.3e-16       1.01x
   L8_radix8                      0.578 us     0.578 us    39.87       0.6%    0.053s  ok 1.2e-16       1.05x
   mkl_dfti                       0.652 us     0.652 us    35.34       2.0%    0.003s  ok 1.5e-16       1.18x
   mkl2026_dfti                   0.707 us     0.707 us    32.57       3.9%    0.004s  ok 1.5e-16       1.28x
   fftw3_patient                  1.174 us     1.174 us    19.63       5.0%    0.030s  ok 1.7e-16       2.13x
   fftw3_measure                  1.242 us     1.242 us    18.55       0.6%    0.019s  ok 1.8e-16       2.26x
   fftw3_estimate                 5.288 us     5.288 us     4.36       0.2%    0.002s  ok 1.7e-16       9.60x
   ducc0_c2c                      6.271 us     6.271 us     3.67       0.9%    0.000s  ok 1.2e-16       11.39x
   baseline_matrix               26.403 us    26.403 us     0.87       0.0%    0.000s  ok 4.1e-16       47.95x

-- L=8 (batched B=64), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.578 us    37.001 us    39.85       4.1%    0.073s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.587 us    37.564 us    39.26       2.0%    0.023s  ok 2.3e-16       1.02x
   L8_radix8                      0.610 us    39.058 us    37.75       1.2%    0.050s  ok 2.3e-16       1.06x
   mkl_dfti                       0.715 us    45.759 us    32.22       1.4%    0.004s  ok 1.6e-16       1.24x
   mkl2026_dfti                   0.756 us    48.398 us    30.47       5.3%    0.004s  ok 1.6e-16       1.31x
   fftw3_patient                  1.259 us    80.590 us    18.30       7.7%    0.028s  ok 1.8e-16       2.18x
   fftw3_measure                  1.271 us    81.372 us    18.12       5.9%    0.017s  ok 1.8e-16       2.20x
   ducc0_c2c                      4.001 us   256.068 us     5.76       0.3%    0.000s  ok 1.3e-16       6.92x
   fftw3_estimate                 5.446 us   348.542 us     4.23       0.1%    0.002s  ok 1.7e-16       9.42x
   baseline_matrix               26.434 us  1691.744 us     0.87       0.0%    0.000s  ok 3.9e-16       45.72x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_fusedaxes                   0.922 us  1887.525 us    25.00       2.9%    0.245s  ok 2.3e-16       1.00x
   L8_batchsimd                   0.959 us  1964.667 us    24.02       2.7%    0.257s  ok 2.3e-16       1.04x
   L8_radix8                      0.960 us  1966.195 us    24.00       4.2%    0.155s  ok 1.9e-16       1.04x
   mkl2026_dfti                   1.332 us  2727.593 us    17.30       2.3%    0.003s  ok 1.6e-16       1.45x
   mkl_dfti                       1.354 us  2773.883 us    17.01       1.6%    0.002s  ok 1.6e-16       1.47x
   fftw3_measure                  1.724 us  3529.958 us    13.37       1.8%    0.017s  ok 1.8e-16       1.87x
   fftw3_patient                  1.827 us  3741.086 us    12.61       0.7%    0.027s  ok 1.8e-16       1.98x
   ducc0_c2c                      4.587 us  9393.942 us     5.02       3.6%    0.000s  ok 1.3e-16       4.98x
   fftw3_estimate                 5.621 us 11512.458 us     4.10       0.1%    0.001s  ok 1.7e-16       6.10x
   baseline_matrix               26.725 us 54733.114 us     0.86       0.0%    0.000s  ok 3.9e-16       29.00x

-- L=8 (batched B=16384), volume 512, working set 256.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   1.236 us 20245.490 us    18.65       2.5%    0.755s  ok 2.3e-16       1.00x
   L8_fusedaxes                   1.251 us 20501.582 us    18.41       1.1%    0.585s  ok 2.3e-16       1.01x
   L8_radix8                      1.272 us 20838.316 us    18.12       1.0%    0.506s  ok 1.9e-16       1.03x
   mkl2026_dfti                   1.795 us 29411.667 us    12.83       1.2%    0.002s  ok 1.6e-16       1.45x
   mkl_dfti                       1.855 us 30397.168 us    12.42       0.3%    0.002s  ok 1.6e-16       1.50x
   fftw3_measure                  2.087 us 34190.015 us    11.04       1.0%    0.016s  ok 1.8e-16       1.69x
   fftw3_patient                  2.184 us 35790.703 us    10.55       5.2%    0.026s  ok 1.8e-16       1.77x
   fftw3_estimate                 5.763 us 94428.328 us     4.00       0.9%    0.001s  ok 1.7e-16       4.66x
   ducc0_c2c                      5.784 us 94768.896 us     3.98       0.8%    0.000s  ok 1.3e-16       4.68x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.733 us     5.733 us    21.27       1.1%    0.017s  ok 2.8e-16       1.00x
   L13_rader                      6.031 us     6.031 us    20.22       1.7%    0.003s  ok 4.0e-16       1.05x
   mkl2026_dfti                   7.530 us     7.530 us    16.20       1.6%    0.004s  ok 3.3e-16       1.31x
   mkl_dfti                       7.775 us     7.775 us    15.69       2.1%    0.004s  ok 3.3e-16       1.36x
   fftw3_patient                  8.459 us     8.459 us    14.42       1.2%    0.029s  ok 3.2e-16       1.48x
   fftw3_measure                  8.605 us     8.605 us    14.17       3.7%    0.015s  ok 3.2e-16       1.50x
   fftw3_estimate                 8.919 us     8.919 us    13.67       1.8%    0.002s  ok 3.2e-16       1.56x
   ducc0_c2c                     43.949 us    43.949 us     2.77       0.1%    0.000s  ok 2.5e-16       7.67x
   baseline_matrix              184.490 us   184.490 us     0.66       0.0%    0.000s  ok 7.8e-16       32.18x

-- L=13 (batched B=16), volume 2197, working set 1.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     5.992 us    95.880 us    20.35       0.9%    0.022s  ok 2.9e-16       1.00x
   L13_rader                      6.398 us   102.371 us    19.06       1.7%    0.008s  ok 4.0e-16       1.07x
   mkl2026_dfti                   7.607 us   121.706 us    16.03       1.9%    0.002s  ok 3.2e-16       1.27x
   mkl_dfti                       7.832 us   125.314 us    15.57       2.0%    0.003s  ok 3.2e-16       1.31x
   fftw3_patient                  8.640 us   138.242 us    14.11       1.4%    0.027s  ok 3.2e-16       1.44x
   fftw3_measure                  8.855 us   141.672 us    13.77       0.3%    0.020s  ok 3.2e-16       1.48x
   fftw3_estimate                 9.001 us   144.021 us    13.55       3.6%    0.002s  ok 3.2e-16       1.50x
   ducc0_c2c                     41.499 us   663.980 us     2.94       2.7%    0.000s  ok 2.5e-16       6.93x
   baseline_matrix              184.569 us  2953.102 us     0.66       0.2%    0.000s  ok 7.9e-16       30.80x

-- L=13 (batched B=512), volume 2197, working set 34.33 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     8.075 us  4134.606 us    15.10       0.7%    0.013s  ok 2.9e-16       1.00x
   L13_rader                      9.292 us  4757.396 us    13.12       0.4%    0.178s  ok 4.0e-16       1.15x
   mkl2026_dfti                   9.336 us  4780.190 us    13.06       1.1%    0.002s  ok 3.2e-16       1.16x
   mkl_dfti                       9.418 us  4821.949 us    12.95       2.0%    0.002s  ok 3.2e-16       1.17x
   fftw3_estimate                10.638 us  5446.777 us    11.46       1.0%    0.001s  ok 3.2e-16       1.32x
   fftw3_measure                 10.674 us  5464.992 us    11.42       1.1%    0.016s  ok 3.2e-16       1.32x
   fftw3_patient                 10.679 us  5467.565 us    11.42       2.5%    0.028s  ok 3.2e-16       1.32x
   ducc0_c2c                     43.411 us 22226.665 us     2.81       3.0%    0.000s  ok 2.5e-16       5.38x
   baseline_matrix              186.340 us 95406.091 us     0.65       0.3%    0.000s  ok 7.9e-16       23.08x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                15.066 us    15.066 us    19.99       1.1%    0.226s  ok 3.3e-16       1.00x
   L17_winograd                  16.458 us    16.458 us    18.30       0.2%    0.521s  ok 3.3e-16       1.09x
   L17_rader                     16.525 us    16.525 us    18.23       1.1%    0.342s  ok 3.1e-16       1.10x
   fftw3_patient                 81.725 us    81.725 us     3.69       6.0%    0.032s  ok 3.0e-16       5.42x
   fftw3_estimate                81.726 us    81.726 us     3.69       6.0%    0.006s  ok 3.0e-16       5.42x
   fftw3_measure                 86.732 us    86.732 us     3.47       7.0%    0.017s  ok 3.0e-16       5.76x
   mkl_dfti                      98.790 us    98.790 us     3.05       0.6%    0.038s  ok 3.1e-16       6.56x
   mkl2026_dfti                 100.692 us   100.692 us     2.99       0.2%    0.020s  ok 3.1e-16       6.68x
   ducc0_c2c                    104.050 us   104.050 us     2.90       1.8%    0.000s  ok 2.6e-16       6.91x
   baseline_matrix              538.260 us   538.260 us     0.56       0.0%    0.000s  ok 8.4e-16       35.73x

-- L=17 (batched B=8), volume 4913, working set 1.20 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                16.438 us   131.507 us    18.32       1.9%    0.258s  ok 3.2e-16       1.00x
   L17_winograd                  17.667 us   141.335 us    17.05       2.6%    0.608s  ok 3.2e-16       1.07x
   L17_rader                     17.795 us   142.363 us    16.93       1.2%    0.359s  ok 3.2e-16       1.08x
   fftw3_patient                 81.953 us   655.623 us     3.68       5.9%    0.030s  ok 3.0e-16       4.99x
   fftw3_estimate                81.993 us   655.943 us     3.67       5.8%    0.003s  ok 3.0e-16       4.99x
   fftw3_measure                 82.026 us   656.207 us     3.67       5.7%    0.013s  ok 3.0e-16       4.99x
   mkl_dfti                      99.980 us   799.841 us     3.01       0.2%    0.042s  ok 3.1e-16       6.08x
   mkl2026_dfti                 100.647 us   805.178 us     2.99       0.2%    0.037s  ok 3.1e-16       6.12x
   ducc0_c2c                    101.246 us   809.969 us     2.98       0.7%    0.000s  ok 2.6e-16       6.16x
   baseline_matrix              538.513 us  4308.107 us     0.56       0.0%    0.000s  ok 8.4e-16       32.76x

-- L=17 (batched B=256), volume 4913, working set 38.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                20.903 us  5351.136 us    14.41       3.5%    1.517s  ok 3.3e-16       1.00x
   L17_winograd                  21.562 us  5519.769 us    13.97       0.4%    3.757s  ok 3.3e-16       1.03x
   L17_rader                     24.348 us  6233.136 us    12.37       0.5%    2.345s  ok 3.2e-16       1.16x
   fftw3_estimate                83.436 us 21359.514 us     3.61       8.0%    0.002s  ok 3.0e-16       3.99x
   fftw3_measure                 83.452 us 21363.789 us     3.61       7.0%    0.009s  ok 3.0e-16       3.99x
   fftw3_patient                 83.480 us 21370.982 us     3.61       6.4%    0.020s  ok 3.0e-16       3.99x
   mkl_dfti                     101.295 us 25931.439 us     2.97       0.2%    0.056s  ok 3.1e-16       4.85x
   mkl2026_dfti                 101.763 us 26051.354 us     2.96       7.0%    0.061s  ok 3.1e-16       4.87x
   ducc0_c2c                    107.925 us 27628.872 us     2.79       4.8%    0.000s  ok 2.6e-16       5.16x
   baseline_matrix              540.501 us 138368.181 us     0.56       0.0%    0.000s  ok 8.4e-16       25.86x

-- L=17 (batched B=2048), volume 4913, working set 307.06 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                  21.721 us 44483.649 us    13.87       0.9%    5.567s  ok 3.3e-16       1.00x
   L17_matrixsimd                21.881 us 44812.456 us    13.77       0.3%    2.363s  ok 3.3e-16       1.01x
   L17_rader                     24.980 us 51159.406 us    12.06       0.6%    3.576s  ok 3.2e-16       1.15x
   fftw3_estimate                84.098 us 172231.952 us     3.58       6.0%    0.002s  ok 3.0e-16       3.87x
   fftw3_patient                 84.132 us 172301.477 us     3.58       0.0%    0.019s  ok 3.0e-16       3.87x
   fftw3_measure                 89.290 us 182866.707 us     3.37       0.1%    0.008s  ok 3.0e-16       4.11x
   mkl_dfti                     101.613 us 208102.977 us     2.96       6.1%    0.057s  ok 3.1e-16       4.68x
   mkl2026_dfti                 101.912 us 208715.703 us     2.96       7.0%    0.058s  ok 3.1e-16       4.69x
   ducc0_c2c                    114.254 us 233991.559 us     2.64       1.4%    0.000s  ok 2.6e-16       5.26x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     47.614 us    47.614 us    17.34       0.8%    0.561s  ok 3.8e-16       1.00x
   L23_matrixsimd                47.648 us    47.648 us    17.33       0.8%    0.459s  ok 3.8e-16       1.00x
   fftw3_patient                260.810 us   260.810 us     3.17       8.9%    0.047s  ok 3.6e-16       5.48x
   fftw3_estimate               261.104 us   261.104 us     3.16       7.1%    0.006s  ok 3.6e-16       5.48x
   fftw3_measure                261.133 us   261.133 us     3.16       9.9%    0.016s  ok 3.6e-16       5.48x
   ducc0_c2c                    292.067 us   292.067 us     2.83       2.4%    0.000s  ok 2.8e-16       6.13x
   mkl_dfti                     314.336 us   314.336 us     2.63       3.4%    0.041s  ok 4.2e-16       6.60x
   mkl2026_dfti                 341.802 us   341.802 us     2.42       3.3%    0.044s  ok 4.2e-16       7.18x
   baseline_matrix             1801.304 us  1801.304 us     0.46       0.0%    0.000s  ok 7.4e-16       37.83x

-- L=23 (batched B=4), volume 12167, working set 1.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     49.117 us   196.466 us    16.81       1.3%    0.581s  ok 3.8e-16       1.00x
   L23_matrixsimd                49.729 us   198.914 us    16.60       0.9%    0.470s  ok 3.8e-16       1.01x
   fftw3_patient                261.378 us  1045.510 us     3.16       9.8%    0.045s  ok 3.7e-16       5.32x
   fftw3_measure                261.540 us  1046.158 us     3.16       7.2%    0.017s  ok 3.7e-16       5.32x
   fftw3_estimate               261.784 us  1047.134 us     3.15       7.2%    0.006s  ok 3.7e-16       5.33x
   ducc0_c2c                    293.236 us  1172.945 us     2.82       2.5%    0.000s  ok 2.8e-16       5.97x
   mkl_dfti                     315.157 us  1260.629 us     2.62       0.2%    0.038s  ok 4.2e-16       6.42x
   mkl2026_dfti                 342.979 us  1371.917 us     2.41       3.3%    0.040s  ok 4.2e-16       6.98x
   baseline_matrix             1801.833 us  7207.332 us     0.46       0.0%    0.000s  ok 7.4e-16       36.68x

-- L=23 (batched B=128), volume 12167, working set 47.53 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_matrixsimd                64.157 us  8212.077 us    12.87       1.1%    2.392s  ok 3.8e-16       1.00x
   L23_rader                     64.536 us  8260.650 us    12.79       0.3%    2.437s  ok 3.8e-16       1.01x
   fftw3_patient                265.149 us 33939.099 us     3.11       8.4%    0.033s  ok 3.7e-16       4.13x
   fftw3_measure                265.351 us 33964.907 us     3.11       9.9%    0.011s  ok 3.7e-16       4.14x
   fftw3_estimate               266.219 us 34076.083 us     3.10       7.0%    0.002s  ok 3.7e-16       4.15x
   ducc0_c2c                    315.435 us 40375.635 us     2.62       2.9%    0.000s  ok 2.8e-16       4.92x
   mkl_dfti                     317.915 us 40693.176 us     2.60       0.4%    0.054s  ok 4.2e-16       4.96x
   mkl2026_dfti                 345.723 us 44252.507 us     2.39       1.0%    0.061s  ok 4.2e-16       5.39x
   baseline_matrix             1806.090 us 231179.522 us     0.46       0.0%    0.000s  ok 7.4e-16       28.15x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_mixedradix               114.561 us   114.561 us    31.58       1.0%    0.304s  ok 3.6e-16       1.00x
   L36_pfa                      117.861 us   117.861 us    30.70       2.9%    0.186s  ok 3.6e-16       1.03x
   L36_pencilfused              118.986 us   118.986 us    30.41       3.1%    0.191s  ok 3.6e-16       1.04x
   mkl_dfti                     163.118 us   163.118 us    22.18       1.2%    0.036s  ok 3.9e-16       1.42x
   mkl2026_dfti                 170.475 us   170.475 us    21.22       0.6%    0.057s  ok 4.0e-16       1.49x
   fftw3_patient                302.949 us   302.949 us    11.94       1.2%    1.141s  ok 3.6e-16       2.64x
   fftw3_measure                364.644 us   364.644 us     9.92       2.9%    0.159s  ok 3.8e-16       3.18x
   ducc0_c2c                    395.881 us   395.881 us     9.14       3.4%    0.000s  ok 3.0e-16       3.46x
   fftw3_estimate               451.137 us   451.137 us     8.02       0.7%    0.012s  ok 3.5e-16       3.94x
   baseline_matrix            10850.663 us 10850.663 us     0.33       0.4%    0.000s  ok 8.0e-16       94.72x

-- L=36 (batched B=4), volume 46656, working set 5.70 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pencilfused              124.701 us   498.805 us    29.01       2.8%    0.262s  ok 3.8e-16       1.00x
   L36_mixedradix               125.358 us   501.432 us    28.86       1.7%    0.296s  ok 3.6e-16       1.01x
   L36_pfa                      127.694 us   510.776 us    28.33       1.2%    0.355s  ok 3.6e-16       1.02x
   mkl_dfti                     175.303 us   701.212 us    20.64       0.2%    0.036s  ok 3.9e-16       1.41x
   mkl2026_dfti                 183.447 us   733.789 us    19.72       0.8%    0.061s  ok 4.1e-16       1.47x
   fftw3_patient                305.787 us  1223.147 us    11.83       4.2%    1.147s  ok 3.9e-16       2.45x
   fftw3_measure                363.497 us  1453.986 us     9.95       2.1%    0.156s  ok 3.8e-16       2.91x
   ducc0_c2c                    426.788 us  1707.151 us     8.48       2.7%    0.000s  ok 3.0e-16       3.42x
   fftw3_estimate               452.735 us  1810.940 us     7.99       0.7%    0.012s  ok 3.5e-16       3.63x
   baseline_matrix            10844.845 us 43379.382 us     0.33       0.2%    0.000s  ok 8.0e-16       86.97x

-- L=36 (batched B=32), volume 46656, working set 45.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      161.788 us  5177.220 us    22.36       1.0%    3.847s  ok 3.6e-16       1.00x
   L36_mixedradix               163.380 us  5228.167 us    22.15       0.7%    0.422s  ok 3.6e-16       1.01x
   L36_pencilfused              164.790 us  5273.278 us    21.96       2.5%    1.166s  ok 3.6e-16       1.02x
   mkl_dfti                     222.813 us  7130.011 us    16.24       0.3%    0.040s  ok 3.9e-16       1.38x
   mkl2026_dfti                 230.723 us  7383.140 us    15.68       0.5%    0.016s  ok 4.0e-16       1.43x
   fftw3_patient                380.961 us 12190.742 us     9.50       2.0%    1.121s  ok 3.9e-16       2.35x
   fftw3_measure                399.781 us 12792.992 us     9.05       1.4%    0.155s  ok 3.8e-16       2.47x
   fftw3_estimate               494.396 us 15820.686 us     7.32       0.7%    0.005s  ok 3.5e-16       3.06x
   ducc0_c2c                    583.343 us 18666.980 us     6.20       1.3%    0.000s  ok 3.0e-16       3.61x
   baseline_matrix            11081.231 us 354599.387 us     0.33       0.0%    0.000s  ok 8.0e-16       68.49x

-- L=36 (batched B=256), volume 46656, working set 364.50 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      174.268 us 44612.658 us    20.76       0.1%    8.281s  ok 3.6e-16       1.00x
   L36_mixedradix               179.886 us 46050.928 us    20.11       0.9%    0.532s  ok 3.6e-16       1.03x
   L36_pencilfused              185.071 us 47378.066 us    19.55       1.2%    2.406s  ok 3.6e-16       1.06x
   mkl_dfti                     248.954 us 63732.154 us    14.53       0.3%    0.058s  ok 3.9e-16       1.43x
   mkl2026_dfti                 257.457 us 65908.895 us    14.05       0.3%    0.058s  ok 4.0e-16       1.48x
   fftw3_patient                404.844 us 103639.979 us     8.94       9.1%    1.125s  ok 3.6e-16       2.32x
   fftw3_measure                412.632 us 105633.801 us     8.77       1.8%    0.154s  ok 3.5e-16       2.37x
   fftw3_estimate               518.205 us 132660.534 us     6.98       0.1%    0.004s  ok 3.5e-16       2.97x
   ducc0_c2c                    726.448 us 185970.763 us     4.98       3.6%    0.000s  ok 3.0e-16       4.17x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      304.194 us   304.194 us    24.68       1.3%    0.441s  ok 4.0e-16       1.00x
   L45_mixedradix               312.948 us   312.948 us    23.99       1.1%    0.264s  ok 4.1e-16       1.03x
   mkl_dfti                     605.981 us   605.981 us    12.39       0.1%    0.036s  ok 4.4e-16       1.99x
   mkl2026_dfti                 615.838 us   615.838 us    12.19       1.3%    0.061s  ok 4.5e-16       2.02x
   fftw3_patient                857.246 us   857.246 us     8.76       2.1%    0.732s  ok 4.3e-16       2.82x
   ducc0_c2c                    864.015 us   864.015 us     8.69       0.7%    0.000s  ok 3.7e-16       2.84x
   fftw3_estimate               935.834 us   935.834 us     8.02       5.2%    0.004s  ok 4.2e-16       3.08x
   fftw3_measure               1095.507 us  1095.507 us     6.85       6.8%    0.081s  ok 4.2e-16       3.60x
   baseline_matrix            28825.669 us 28825.669 us     0.26       0.3%    0.000s  ok 8.0e-16       94.76x

-- L=45 (batched B=2), volume 91125, working set 5.56 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      308.874 us   617.748 us    24.30       1.3%    0.451s  ok 4.0e-16       1.00x
   L45_mixedradix               310.275 us   620.550 us    24.19       1.2%    0.502s  ok 4.1e-16       1.00x
   mkl_dfti                     598.233 us  1196.466 us    12.55       1.5%    0.036s  ok 4.4e-16       1.94x
   mkl2026_dfti                 623.739 us  1247.478 us    12.03       1.8%    0.039s  ok 4.5e-16       2.02x
   fftw3_patient                864.964 us  1729.929 us     8.68      13.0%    0.740s  ok 4.3e-16       2.80x
   ducc0_c2c                    889.599 us  1779.198 us     8.44       1.9%    0.000s  ok 3.7e-16       2.88x
   fftw3_estimate               952.688 us  1905.375 us     7.88       2.8%    0.004s  ok 4.3e-16       3.08x
   fftw3_measure               1091.426 us  2182.851 us     6.88       8.9%    0.083s  ok 4.3e-16       3.53x
   baseline_matrix            28809.061 us 57618.123 us     0.26       0.2%    0.000s  ok 8.0e-16       93.27x

-- L=45 (batched B=16), volume 91125, working set 44.49 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                      391.951 us  6271.214 us    19.15       0.8%    1.022s  ok 4.0e-16       1.00x
   L45_mixedradix               405.188 us  6483.011 us    18.53       2.9%    0.721s  ok 4.1e-16       1.03x
   mkl_dfti                     683.443 us 10935.084 us    10.98       1.3%    0.041s  ok 4.4e-16       1.74x
   mkl2026_dfti                 699.571 us 11193.128 us    10.73       0.5%    0.035s  ok 4.5e-16       1.78x
   fftw3_patient               1020.161 us 16322.580 us     7.36      13.3%    0.714s  ok 4.3e-16       2.60x
   fftw3_estimate              1020.944 us 16335.112 us     7.35       2.5%    0.002s  ok 4.2e-16       2.60x
   ducc0_c2c                   1216.924 us 19470.782 us     6.17       0.9%    0.000s  ok 3.7e-16       3.10x
   fftw3_measure               1227.528 us 19640.446 us     6.12       7.8%    0.077s  ok 4.3e-16       3.13x
   baseline_matrix            32638.863 us 522221.810 us     0.23       0.1%    0.000s  ok 8.0e-16       83.27x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_blocked                  952.743 us   952.743 us    24.76       1.4%    0.260s  ok 4.5e-16       1.00x
   L64_radix8                   955.095 us   955.095 us    24.70       1.6%    0.245s  ok 4.5e-16       1.00x
   mkl_dfti                    1193.813 us  1193.813 us    19.76       0.9%    0.041s  ok 3.4e-16       1.25x
   mkl2026_dfti                1340.402 us  1340.402 us    17.60       0.1%    0.038s  ok 3.4e-16       1.41x
   fftw3_measure               1757.128 us  1757.128 us    13.43       2.6%    0.310s  ok 3.6e-16       1.84x
   fftw3_patient               1789.351 us  1789.351 us    13.19       1.7%    4.716s  ok 3.5e-16       1.88x
   ducc0_c2c                   2342.583 us  2342.583 us    10.07       1.1%    0.000s  ok 3.0e-16       2.46x
   fftw3_estimate              3332.616 us  3332.616 us     7.08       0.2%    0.001s  ok 3.5e-16       3.50x
   baseline_matrix           122333.481 us 122333.481 us     0.19       3.8%    0.000s  ok 7.8e-16       128.40x

-- L=64 (batched B=2), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                  1021.689 us  2043.378 us    23.09       4.2%    0.445s  ok 4.5e-16       1.00x
   L64_blocked                 1041.882 us  2083.765 us    22.64       3.7%    0.453s  ok 4.5e-16       1.02x
   mkl_dfti                    1236.183 us  2472.367 us    19.09       1.0%    0.038s  ok 3.4e-16       1.21x
   mkl2026_dfti                1369.188 us  2738.376 us    17.23       0.3%    0.056s  ok 3.4e-16       1.34x
   fftw3_measure               1814.972 us  3629.943 us    13.00       2.6%    0.304s  ok 3.5e-16       1.78x
   fftw3_patient               1840.183 us  3680.367 us    12.82       4.2%    4.705s  ok 3.5e-16       1.80x
   ducc0_c2c                   2394.223 us  4788.446 us     9.85       1.6%    0.000s  ok 3.0e-16       2.34x
   fftw3_estimate              3319.150 us  6638.299 us     7.11       0.4%    0.002s  ok 3.5e-16       3.25x
   baseline_matrix           124667.421 us 249334.842 us     0.19       0.8%    0.000s  ok 7.8e-16       122.02x

-- L=64 (batched B=8), volume 262144, working set 64.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                  1247.663 us  9981.300 us    18.91       0.9%    2.056s  ok 4.5e-16       1.00x
   L64_blocked                 1324.474 us 10595.789 us    17.81       0.7%    1.188s  ok 4.5e-16       1.06x
   mkl_dfti                    1971.810 us 15774.482 us    11.97       0.3%    0.053s  ok 3.4e-16       1.58x
   mkl2026_dfti                2087.450 us 16699.598 us    11.30       0.5%    0.038s  ok 3.4e-16       1.67x
   fftw3_patient               2398.681 us 19189.451 us     9.84       9.7%    5.321s  ok 3.5e-16       1.92x
   fftw3_measure               2958.629 us 23669.028 us     7.97       2.7%    0.305s  ok 3.5e-16       2.37x
   ducc0_c2c                   3516.127 us 28129.012 us     6.71       0.8%    0.000s  ok 3.0e-16       2.82x
   fftw3_estimate              4419.899 us 35359.189 us     5.34       0.5%    0.001s  ok 3.5e-16       3.54x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first+pf; ab[B16]=y26034,zs6090,xt6009,xl6513 ns/vol
   L13_rader                Rader-13 split cyc/nega (186 FP/pt), X-first, 512-bit; fuse=0 um=1 ys=0 pf=1 pw=1 pace=1 znb=22 ab[B512]=i:8637,pw!:9877,pf!:8780 pick=i
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, addr-safe t1, pf=0, pw=0, pt=0, sbw[rd/wr/cp/s17]=3.28/5.41/11.12/3.15, b1dec[yz/kyz/x/kx]=10.99/10.52/4.52/4.03, clk512/256=2.89/3.89 GHz, d256=3.89
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512t dy, pf=0, pfw=0, clk256=3.89 clk512=2.89, xrace xl/xfs=23.25/26.44, probe ph/xp/fu=16.50/6.76/23.50 us/vol
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, cw=0, clk256=2.89GHz, clk512=2.89GHz, p1=6.34 f23=10.28 fu=16.72 fu4=17.46, sbw[rd/wr/cp/s17]=3.79/9.33/13.58/3.53
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, pf=0, pw=0, tune[pick=47.74 inc=47.74 us/t nv=1], clk512/256=2.89/3.89 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, tuner pick=48.09 inc=48.09 us/t nv=1
   L36_mixedradix           PFA 4x9 2-sweep, lanes=lines, n1_9 DFT9; pick=v1-cached-pf1-pfin-pfw (B=32, arena=32 vol, stream=1, 6 cand, pinD=-1)
   L36_pencilfused          L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes; tuner picked pw=4 mode=istream+pfw (B=256)
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep; tuner pick: pw=4 mode=inplace pf=0 (B=4, nv=4, nc=14); probe us p1=91.3 p1z=41.6 p1y=26.3 p2w=30.3 fu=117.9
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-pf0 (B=1, arena=1, stream=0, 4 cand); nv1 us fu=315.2 p1=238.8 p2w=77.3; fe=na
   L45_pfa                  GT-PFA 9x5 two-sweep; tuner pick: pw4-ip-pf0 (B=2, nv=2)
   L64_blocked              L64 8x8 two-stage, hugepage odd-line-padded scratch; tuner pick: pw=4 mode=cached pf=0 st=3(split-sc) pro=0 (B=1, nv=1)
   L64_radix8               radix-8^2 per axis, split-complex AVX-512, padded scratch; tuner pick[B=2]=fused-pfw+slabpf1+pro0+p10+sc0+xb0+fo0
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, plan-raced; variant=fused_pfw_xa_d2 clkS256=3.89 clkD256=2.89 clkS512=2.89 kclk=2.89GHz bf=217.7 bsp=217.3 bx=62.8 byz=153.6ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), radix-2-first VD6, ymm; variant=fused_zp_pf kclk=2.89GHz ab1=f220.1,fx218.3ns abL=f524.0,f3529.6ns xod=+0.7%
   L8_batchsimd             radix-8 split; pick[B=16384]: mode=FUSED nt=0 pf=s0w alloc=r8(a64,si512) arena{FUSED/s0w=1.121,FUSED/s0=1.487,LANEX3/s0w=1.134,LANEX3/s0=1.326,FUSED-nt/s0=1.516,LANEX3/none=1.727}
   L8_fusedaxes             8^3 fused/AA/AA2 c52; B=1 pick=fused (tuned) arena{fused=0.555,fusedAA=0.564,fusedAA2=0.566}
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p shapes; pick[B=2048]=avx512-3p-pfs-pfw (default) arena{3p-pfs-pfw=0.903 1f-pfs-pfw*=0.928 3p-pfs=1.037 3p-pf=1.197} scr@0x4c0
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
