# What previous generations produced (round mt_r2 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r1/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_direct.md 138 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_rader.md 116 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_matrixsimd.md 182 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_rader.md 174 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_winograd.md 119 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_matrixsimd.md 200 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_rader.md 168 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_mixedradix.md 144 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pencilfused.md 210 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pfa.md 156 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_mixedradix.md 192 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_pfa.md 177 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_blocked.md 171 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_radix8.md 163 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_pfa.md 156 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_unrolled.md 151 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_batchsimd.md 140 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_fusedaxes.md 178 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_radix8.md 167 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/mt/exemplars/mt_r1/
      # Round mt_r1 — what it established
      
      Promoted: L6_pfa L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_pfa L36_mixedradix L36_pencilfused L45_pfa L45_mixedradix L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
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
