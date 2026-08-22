# What previous generations produced (round mt_r4 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/mt/results/mt_r3/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_direct.md 477 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L13_rader.md 251 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_matrixsimd.md 487 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_rader.md 464 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L17_winograd.md 364 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_matrixsimd.md 533 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L23_rader.md 457 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_mixedradix.md 448 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pencilfused.md 582 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L36_pfa.md 486 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_mixedradix.md 503 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L45_pfa.md 541 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_blocked.md 452 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L64_radix8.md 475 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_pfa.md 379 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L6_unrolled.md 424 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_batchsimd.md 451 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_fusedaxes.md 530 lines
  /home/lqcd/wdetmold/fft/bench/mt/strategies/L8_radix8.md 509 lines

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
      
  /home/lqcd/wdetmold/fft/bench/mt/exemplars/mt_r3/
      # Round mt_r3 — what it established
      
      Promoted: L6_pfa L6_unrolled L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_matrixsimd L23_rader L36_mixedradix L36_pencilfused L36_pfa L45_pfa L64_blocked L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
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
