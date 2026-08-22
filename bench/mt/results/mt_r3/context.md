# What previous generations produced (round mt_r3 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r2/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_direct.md 286 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_rader.md 116 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_matrixsimd.md 325 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_rader.md 310 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_winograd.md 218 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_matrixsimd.md 372 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_rader.md 308 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_mixedradix.md 281 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pencilfused.md 409 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pfa.md 297 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_mixedradix.md 333 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_pfa.md 369 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_blocked.md 314 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_radix8.md 324 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_pfa.md 257 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_unrolled.md 284 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_batchsimd.md 278 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_fusedaxes.md 352 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_radix8.md 333 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/mt/exemplars/mt_r1/
      # Round mt_r1 — what it established
      
      Promoted: L6_pfa L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_pfa L36_mixedradix L36_pencilfused L45_pfa L45_mixedradix L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/mt/exemplars/mt_r2/
      # Round mt_r2 — what it established
      
      Promoted: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L13_direct L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pfa L36_pencilfused L45_pfa L45_mixedradix L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
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
