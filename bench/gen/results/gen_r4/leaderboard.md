```
=== round gen_r4 ===
# round gen_r4
host: a80n0.lqcd.mit   date: 2026-08-24T16:19:33-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.153 us 73787.811 us    43.22       0.1%    0.000s  ok ch=1.3e-13/1e-10 1s=8e-16 1.00x
   gen_batchlane                  1.154 us 73878.699 us    43.17       7.9%    0.000s  ok ch=1.5e-13/1e-10 1s=8e-16 1.00x
   gen_planner                    3.460 us 221451.497 us    14.40       1.5%    0.002s  ok ch=1.3e-13/1e-10 1s=9e-16 3.00x
   gen_race                       3.463 us 221628.820 us    14.39       0.2%    0.004s  ok ch=1.3e-13/1e-10 1s=9e-16 3.00x
   mkl_dfti                       4.553 us 291410.307 us    10.94       0.4%    0.003s  ok ch=1.5e-13/1e-10 1s=1e-15 3.95x
   mkl2026_dfti                   4.632 us 296441.729 us    10.76       0.2%    0.002s  ok ch=1.7e-13/1e-10 1s=1e-15 4.02x
   gen_layout                     4.904 us 313877.627 us    10.16       4.5%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 4.25x
   fftw3_measure                  5.085 us 325449.525 us     9.80       3.3%    0.013s  ok ch=1.2e-13/1e-10 1s=8e-16 4.41x
   fftw3_patient                  5.211 us 333531.178 us     9.56       0.4%    0.022s  ok ch=1.2e-13/1e-10 1s=8e-16 4.52x
   gen_dense_prime                5.282 us 338067.463 us     9.43       5.1%    0.000s  ok ch=1.1e-13/1e-10 1s=9e-16 4.58x
   fftw3_guru                     6.365 us 407369.396 us     7.83       0.0%    0.010s  ok ch=9.8e-14/1e-10 1s=9e-16 5.52x
   fftw3_estimate                 7.314 us 468119.419 us     6.81       0.7%    0.001s  ok ch=1.2e-13/1e-10 1s=8e-16 6.34x
   ducc0_c2c                      9.680 us 619491.214 us     5.15       0.3%    0.000s  ok ch=9.4e-14/1e-10 1s=7e-16 8.40x
   gen_twiddle                   10.935 us 699844.314 us     4.56       0.7%    0.003s  ok ch=1.2e-13/1e-10 1s=9e-16 9.48x
   gen_bluestein                 13.281 us 850007.701 us     3.75       0.8%    0.000s  ok ch=1.8e-13/1e-10 1s=1e-15 11.52x
   baseline_matrix               54.650 us 3497607.340 us     0.91       0.1%    0.000s  ok ch=3.4e-13/1e-10 1s=1e-15 47.40x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.912 us 73424.689 us    48.60       0.2%    0.000s  ok ch=8.7e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.969 us 75620.311 us    47.19       0.8%    0.000s  ok ch=1.1e-13/1e-10 1s=9e-16 1.03x
   gen_race                       5.107 us 196097.143 us    18.20       0.8%    0.004s  ok ch=6.3e-14/1e-10 1s=9e-16 2.67x
   gen_planner                    5.112 us 196291.515 us    18.18       0.5%    0.003s  ok ch=6.3e-14/1e-10 1s=9e-16 2.67x
   mkl_dfti                       7.737 us 297098.439 us    12.01       0.0%    0.002s  ok ch=7.8e-14/1e-10 1s=9e-16 4.05x
   mkl2026_dfti                   7.743 us 297331.312 us    12.00       0.3%    0.002s  ok ch=7.3e-14/1e-10 1s=9e-16 4.05x
   gen_layout                     8.075 us 310090.647 us    11.51       2.5%    0.000s  ok ch=5.3e-14/1e-10 1s=9e-16 4.22x
   gen_dense_prime                8.272 us 317654.336 us    11.23      10.1%    0.000s  ok ch=6.1e-14/1e-10 1s=9e-16 4.33x
   fftw3_patient                  8.469 us 325213.681 us    10.97       6.3%    0.023s  ok ch=1.1e-13/1e-10 1s=9e-16 4.43x
   fftw3_measure                  8.617 us 330909.541 us    10.78       7.0%    0.014s  ok ch=9.7e-14/1e-10 1s=9e-16 4.51x
   fftw3_guru                    10.178 us 390817.847 us     9.13       1.0%    0.011s  ok ch=7.3e-14/1e-10 1s=9e-16 5.32x
   gen_twiddle                   14.967 us 574727.491 us     6.21       1.7%    0.002s  ok ch=7.2e-14/1e-10 1s=1e-15 7.83x
   ducc0_c2c                     16.004 us 614570.585 us     5.81       0.3%    0.000s  ok ch=3.7e-14/1e-10 1s=7e-16 8.37x
   gen_bluestein                 19.262 us 739663.423 us     4.82       0.4%    0.000s  ok ch=9.4e-14/1e-10 1s=1e-15 10.07x
   fftw3_estimate                19.285 us 740527.173 us     4.82       0.1%    0.001s  ok ch=1.1e-13/1e-10 1s=9e-16 10.09x
   baseline_matrix              112.110 us 4305025.000 us     0.83       0.0%    0.000s  ok ch=2.9e-13/1e-10 1s=2e-15 58.63x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  4.429 us 85043.223 us    44.65       0.2%    0.000s  ok ch=5.5e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.566 us 87659.138 us    43.32       0.2%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 1.03x
   gen_planner                   11.634 us 223370.331 us    17.00       0.8%    0.014s  ok ch=5.3e-14/1e-10 1s=1e-15 2.63x
   gen_race                      11.638 us 223455.912 us    16.99       0.1%    0.030s  ok ch=5.3e-14/1e-10 1s=1e-15 2.63x
   gen_dense_prime               14.264 us 273866.004 us    13.87       2.9%    0.000s  ok ch=5.4e-14/1e-10 1s=1e-15 3.22x
   mkl_dfti                      16.559 us 317932.691 us    11.94       1.0%    0.002s  ok ch=6.1e-14/1e-10 1s=1e-15 3.74x
   mkl2026_dfti                  16.729 us 321203.664 us    11.82       0.1%    0.003s  ok ch=6.4e-14/1e-10 1s=1e-15 3.78x
   gen_layout                    18.404 us 353364.674 us    10.75       2.1%    0.000s  ok ch=4.9e-14/1e-10 1s=1e-15 4.16x
   fftw3_patient                 19.586 us 376043.819 us    10.10       0.4%    0.021s  ok ch=5.4e-14/1e-10 1s=1e-15 4.42x
   fftw3_measure                 19.836 us 380855.124 us     9.97       5.2%    0.012s  ok ch=5.4e-14/1e-10 1s=1e-15 4.48x
   fftw3_estimate                20.624 us 395972.224 us     9.59       0.7%    0.001s  ok ch=5.4e-14/1e-10 1s=1e-15 4.66x
   fftw3_guru                    26.188 us 502814.781 us     7.55       2.5%    0.010s  ok ch=5.1e-14/1e-10 1s=1e-15 5.91x
   gen_twiddle                   29.672 us 569700.242 us     6.67       2.6%    0.002s  ok ch=5.6e-14/1e-10 1s=1e-15 6.70x
   ducc0_c2c                     32.303 us 620212.707 us     6.12       0.9%    0.000s  ok ch=4.2e-14/1e-10 1s=1e-15 7.29x
   gen_bluestein                 33.768 us 648344.459 us     5.86       0.3%    0.000s  ok ch=8.5e-14/1e-10 1s=2e-15 7.62x
   baseline_matrix              272.400 us 5230087.360 us     0.73       0.0%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 61.50x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                 13.059 us 106980.301 us    39.71       1.7%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                 13.145 us 107687.715 us    39.45       1.3%    0.001s  ok ch=2.7e-14/1e-10 1s=1e-15 1.01x
   gen_race                      24.032 us 196870.645 us    21.58       0.6%    0.005s  ok ch=3.2e-14/1e-10 1s=1e-15 1.84x
   gen_planner                   24.685 us 202218.226 us    21.01       0.1%    0.003s  ok ch=3.4e-14/1e-10 1s=1e-15 1.89x
   gen_layout                    39.905 us 326905.249 us    13.00       1.5%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 3.06x
   gen_dense_prime               44.159 us 361752.488 us    11.74       1.7%    0.000s  ok ch=3.7e-14/1e-10 1s=1e-15 3.38x
   fftw3_measure                 44.889 us 367727.422 us    11.55       1.5%    0.085s  ok ch=2.7e-14/1e-10 1s=1e-15 3.44x
   fftw3_patient                 45.185 us 370157.449 us    11.48       4.0%    0.315s  ok ch=2.7e-14/1e-10 1s=1e-15 3.46x
   mkl2026_dfti                  57.543 us 471389.172 us     9.01       1.9%    0.048s  ok ch=3.8e-14/1e-10 1s=1e-15 4.41x
   gen_twiddle                   58.209 us 476844.419 us     8.91       1.7%    0.003s  ok ch=3.0e-14/1e-10 1s=1e-15 4.46x
   mkl_dfti                      58.211 us 476861.357 us     8.91       0.9%    0.031s  ok ch=3.9e-14/1e-10 1s=1e-15 4.46x
   fftw3_guru                    60.037 us 491823.986 us     8.64       1.6%    0.078s  ok ch=2.5e-14/1e-10 1s=1e-15 4.60x
   ducc0_c2c                     72.899 us 597190.035 us     7.11       2.6%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 5.58x
   fftw3_estimate                91.722 us 751388.363 us     5.65       0.1%    0.001s  ok ch=2.7e-14/1e-10 1s=1e-15 7.02x
   gen_bluestein                102.568 us 840234.068 us     5.06       1.2%    0.000s  ok ch=5.7e-14/1e-10 1s=2e-15 7.85x
   baseline_matrix              850.261 us 6965335.530 us     0.61       0.1%    0.000s  ok ch=1.0e-13/1e-10 1s=2e-15 65.11x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      32.100 us 131480.338 us    33.91       5.8%    0.509s  ok ch=4.0e-14/1e-10 1s=1e-15 1.00x
   gen_planner                   60.130 us 246292.487 us    18.10       0.7%    0.004s  ok ch=3.6e-14/1e-10 1s=2e-15 1.87x
   gen_race                      60.130 us 246294.375 us    18.10       0.6%    0.005s  ok ch=3.6e-14/1e-10 1s=2e-15 1.87x
   gen_layout                    93.358 us 382392.716 us    11.66       2.5%    0.000s  ok ch=3.5e-14/1e-10 1s=2e-15 2.91x
   fftw3_patient                108.437 us 444159.314 us    10.04       3.6%    0.110s  ok ch=3.7e-14/1e-10 1s=2e-15 3.38x
   fftw3_measure                108.597 us 444813.564 us    10.02      10.1%    0.028s  ok ch=3.7e-14/1e-10 1s=2e-15 3.38x
   mkl_dfti                     120.545 us 493754.225 us     9.03       0.3%    0.033s  ok ch=4.0e-14/1e-10 1s=2e-15 3.76x
   gen_twiddle                  122.093 us 500091.967 us     8.91       2.8%    0.003s  ok ch=4.4e-14/1e-10 1s=2e-15 3.80x
   mkl2026_dfti                 123.579 us 506180.252 us     8.81       0.6%    0.055s  ok ch=3.7e-14/1e-10 1s=2e-15 3.85x
   fftw3_estimate               130.899 us 536163.875 us     8.31       6.4%    0.001s  ok ch=3.7e-14/1e-10 1s=2e-15 4.08x
   ducc0_c2c                    145.223 us 594834.008 us     7.49       0.1%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 4.52x
   fftw3_guru                   175.106 us 717232.997 us     6.22       0.3%    0.029s  ok ch=3.7e-14/1e-10 1s=2e-15 5.46x
   gen_bluestein                179.150 us 733798.042 us     6.08       1.0%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 5.58x
   baseline_matrix             2061.293 us 8443056.040 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 64.22x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      44.281 us 141699.109 us    31.70       6.8%    0.437s  ok ch=5.3e-14/1e-10 1s=2e-15 1.00x
   gen_race                      85.902 us 274887.606 us    16.34       0.8%    0.009s  ok ch=5.6e-14/1e-10 1s=2e-15 1.94x
   gen_planner                   86.358 us 276344.466 us    16.26       0.6%    0.004s  ok ch=5.6e-14/1e-10 1s=2e-15 1.95x
   gen_layout                   123.836 us 396274.308 us    11.34       1.7%    0.000s  ok ch=5.9e-14/1e-10 1s=2e-15 2.80x
   mkl_dfti                     144.393 us 462057.352 us     9.72       0.1%    0.029s  ok ch=4.1e-14/1e-10 1s=2e-15 3.26x
   mkl2026_dfti                 148.070 us 473822.446 us     9.48       0.1%    0.049s  ok ch=4.1e-14/1e-10 1s=2e-15 3.34x
   ducc0_c2c                    190.238 us 608761.361 us     7.38       2.5%    0.000s  ok ch=4.8e-14/1e-10 1s=1e-15 4.30x
   fftw3_patient                204.059 us 652990.327 us     6.88       4.2%    0.141s  ok ch=4.3e-14/1e-10 1s=2e-15 4.61x
   gen_bluestein                212.987 us 681559.007 us     6.59       1.3%    0.000s  ok ch=8.1e-14/1e-10 1s=2e-15 4.81x
   fftw3_measure                225.417 us 721333.668 us     6.23       2.2%    0.034s  ok ch=3.7e-14/1e-10 1s=2e-15 5.09x
   gen_twiddle                  235.300 us 752958.463 us     5.97       3.3%    0.003s  ok ch=4.9e-14/1e-10 1s=2e-15 5.31x
   fftw3_estimate               253.226 us 810321.933 us     5.54       3.3%    0.002s  ok ch=3.4e-14/1e-10 1s=2e-15 5.72x
   fftw3_guru                   260.491 us 833570.303 us     5.39       1.8%    0.044s  ok ch=3.3e-14/1e-10 1s=2e-15 5.88x
   baseline_matrix             2798.714 us 8955885.900 us     0.50       0.0%    0.000s  ok ch=1.4e-13/1e-10 1s=3e-15 63.20x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     84.603 us 189510.156 us    26.17       1.2%    0.010s  ok ch=3.8e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              120.490 us 269898.167 us    18.37       0.1%    0.001s  ok ch=2.5e-14/1e-10 1s=2e-15 1.42x
   gen_race                     195.606 us 438156.667 us    11.32       0.7%    0.010s  ok ch=4.2e-14/1e-10 1s=2e-15 2.31x
   gen_planner                  202.978 us 454671.160 us    10.91       0.9%    0.008s  ok ch=4.2e-14/1e-10 1s=2e-15 2.40x
   gen_layout                   223.461 us 500552.569 us     9.91       1.7%    0.001s  ok ch=3.6e-14/1e-10 1s=2e-15 2.64x
   gen_bluestein                299.371 us 670591.444 us     7.40       1.7%    0.000s  ok ch=5.6e-14/1e-10 1s=3e-15 3.54x
   gen_twiddle                  463.249 us 1037676.830 us     4.78       1.3%    0.006s  ok ch=4.1e-14/1e-10 1s=2e-15 5.48x
   ducc0_c2c                    720.156 us 1613149.750 us     3.07       2.0%    0.000s  ok ch=4.4e-14/1e-10 1s=1e-15 8.51x
   fftw3_guru                   830.449 us 1860204.940 us     2.67       0.4%    0.093s  ok ch=3.5e-14/1e-10 1s=2e-15 9.82x
   mkl_dfti                     848.956 us 1901660.900 us     2.61       0.0%    0.034s  ok ch=4.4e-14/1e-10 1s=2e-15 10.03x
   fftw3_estimate               859.354 us 1924952.960 us     2.58       0.3%    0.002s  ok ch=3.5e-14/1e-10 1s=2e-15 10.16x
   fftw3_measure                859.444 us 1925154.560 us     2.58       0.1%    0.089s  ok ch=3.5e-14/1e-10 1s=2e-15 10.16x
   fftw3_patient                859.876 us 1926121.850 us     2.57       0.2%    0.240s  ok ch=3.5e-14/1e-10 1s=2e-15 10.16x
   mkl2026_dfti                 883.044 us 1978017.840 us     2.51       0.0%    0.051s  ok ch=4.4e-14/1e-10 1s=2e-15 10.44x
   baseline_matrix             4852.568 us 10869751.500 us     0.46       0.0%    0.000s  ok ch=8.1e-14/1e-10 1s=3e-15 57.36x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      55.746 us 111492.803 us    44.09       1.0%    0.000s  ok ch=3.8e-14/1e-10 1s=1e-15 1.00x
   gen_planner                  129.806 us 259611.650 us    18.93       2.7%    0.007s  ok ch=3.5e-14/1e-10 1s=2e-15 2.33x
   gen_race                     131.527 us 263053.439 us    18.69       1.4%    0.006s  ok ch=3.5e-14/1e-10 1s=2e-15 2.36x
   mkl_dfti                     171.200 us 342400.049 us    14.36       0.7%    0.002s  ok ch=4.2e-14/1e-10 1s=1e-15 3.07x
   mkl2026_dfti                 182.103 us 364206.421 us    13.50       2.6%    0.002s  ok ch=3.7e-14/1e-10 1s=1e-15 3.27x
   fftw3_patient                208.829 us 417658.048 us    11.77       1.7%    0.544s  ok ch=4.0e-14/1e-10 1s=1e-15 3.75x
   fftw3_measure                213.132 us 426263.754 us    11.53       0.2%    0.082s  ok ch=3.7e-14/1e-10 1s=1e-15 3.82x
   gen_layout                   227.712 us 455423.165 us    10.79       1.2%    0.001s  ok ch=4.4e-14/1e-10 1s=2e-15 4.08x
   fftw3_guru                   280.523 us 561046.663 us     8.76      14.4%    0.073s  ok ch=3.7e-14/1e-10 1s=1e-15 5.03x
   ducc0_c2c                    307.184 us 614367.752 us     8.00       1.5%    0.000s  ok ch=3.1e-14/1e-10 1s=1e-15 5.51x
   gen_bluestein                318.287 us 636573.588 us     7.72       2.4%    0.000s  ok ch=8.6e-14/1e-10 1s=2e-15 5.71x
   gen_twiddle                  360.710 us 721420.902 us     6.81       1.9%    0.004s  ok ch=4.1e-14/1e-10 1s=1e-15 6.47x
   fftw3_estimate               407.035 us 814070.977 us     6.04       0.7%    0.001s  ok ch=3.3e-14/1e-10 1s=1e-15 7.30x
   baseline_matrix             5752.234 us 11504467.000 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=2e-15 103.19x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                188.718 us 193247.679 us    27.07       0.4%    0.004s  ok ch=2.3e-14/1e-10 1s=2e-15 1.00x
   gen_planner                  282.620 us 289402.867 us    18.08       0.5%    0.019s  ok ch=2.4e-14/1e-10 1s=2e-15 1.50x
   gen_race                     284.364 us 291188.901 us    17.97       0.9%    0.059s  ok ch=2.2e-14/1e-10 1s=2e-15 1.51x
   mkl2026_dfti                 404.827 us 414542.721 us    12.62       1.1%    0.002s  ok ch=3.9e-14/1e-10 1s=2e-15 2.15x
   mkl_dfti                     405.174 us 414897.865 us    12.61       0.4%    0.002s  ok ch=2.6e-14/1e-10 1s=2e-15 2.15x
   gen_layout                   507.702 us 519887.003 us    10.06       0.9%    0.001s  ok ch=2.5e-14/1e-10 1s=2e-15 2.69x
   fftw3_patient                524.953 us 537551.773 us     9.73       2.7%    1.495s  ok ch=2.1e-14/1e-10 1s=2e-15 2.78x
   fftw3_measure                552.593 us 565855.725 us     9.25       2.1%    0.154s  ok ch=2.3e-14/1e-10 1s=2e-15 2.93x
   ducc0_c2c                    599.941 us 614339.941 us     8.52       1.6%    0.000s  ok ch=1.9e-14/1e-10 1s=1e-15 3.18x
   gen_twiddle                  647.058 us 662587.524 us     7.90       1.5%    0.008s  ok ch=2.2e-14/1e-10 1s=2e-15 3.43x
   fftw3_guru                   675.882 us 692103.561 us     7.56       1.0%    0.152s  ok ch=2.3e-14/1e-10 1s=2e-15 3.58x
   gen_bluestein               1063.781 us 1089311.400 us     4.80       3.4%    0.000s  ok ch=3.3e-14/1e-10 1s=3e-15 5.64x
   fftw3_estimate              1607.957 us 1646548.060 us     3.18       0.6%    0.002s  ok ch=2.1e-14/1e-10 1s=2e-15 8.52x
   baseline_matrix            13437.375 us 13759872.400 us     0.38       0.1%    0.000s  ok ch=5.9e-14/1e-10 1s=2e-15 71.20x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                466.040 us 238612.391 us    22.71       0.7%    1.586s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_powp                     472.873 us 242111.142 us    22.38       1.0%    1.012s  ok ch=3.3e-14/1e-10 1s=2e-15 1.01x
   gen_race                     640.209 us 327786.916 us    16.53       1.5%    0.009s  ok ch=3.0e-14/1e-10 1s=2e-15 1.37x
   gen_planner                  641.142 us 328264.716 us    16.51       1.1%    0.012s  ok ch=3.0e-14/1e-10 1s=2e-15 1.38x
   mkl_dfti                     945.688 us 484192.347 us    11.19       0.5%    0.036s  ok ch=3.3e-14/1e-10 1s=2e-15 2.03x
   mkl2026_dfti                 963.655 us 493391.475 us    10.98       0.2%    0.049s  ok ch=5.4e-14/1e-10 1s=2e-15 2.07x
   fftw3_patient               1155.776 us 591757.209 us     9.16       1.8%    1.317s  ok ch=3.4e-14/1e-10 1s=2e-15 2.48x
   fftw3_measure               1186.667 us 607573.660 us     8.92       0.6%    0.092s  ok ch=3.4e-14/1e-10 1s=2e-15 2.55x
   gen_layout                  1204.407 us 616656.137 us     8.79       0.8%    0.001s  ok ch=4.9e-14/1e-10 1s=2e-15 2.58x
   ducc0_c2c                   1270.471 us 650481.397 us     8.33       1.1%    0.000s  ok ch=2.7e-14/1e-10 1s=2e-15 2.73x
   gen_twiddle                 1367.703 us 700264.059 us     7.74       0.4%    0.012s  ok ch=3.4e-14/1e-10 1s=2e-15 2.93x
   fftw3_estimate              1547.853 us 792500.855 us     6.84       4.5%    0.003s  ok ch=3.7e-14/1e-10 1s=2e-15 3.32x
   fftw3_guru                  1663.903 us 851918.533 us     6.36       3.4%    0.107s  ok ch=3.5e-14/1e-10 1s=2e-15 3.57x
   gen_bluestein               1787.229 us 915061.441 us     5.92       1.1%    0.000s  ok ch=4.8e-14/1e-10 1s=3e-15 3.83x
   baseline_matrix            34346.785 us 17585554.000 us     0.31       0.3%    0.000s  ok ch=9.2e-14/1e-10 1s=4e-15 73.70x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large               4827.566 us 308964.255 us    20.64       0.8%    0.006s  ok ch=2.3e-14/1e-10 1s=3e-15 1.00x
   gen_powp                    4828.216 us 309005.823 us    20.64      14.1%    2.625s  ok ch=2.3e-14/1e-10 1s=3e-15 1.00x
   gen_planner                 5466.892 us 349881.107 us    18.23       1.1%    0.086s  ok ch=2.4e-14/1e-10 1s=3e-15 1.13x
   gen_race                    5640.202 us 360972.903 us    17.67       3.6%    0.005s  ok ch=2.4e-14/1e-10 1s=3e-15 1.17x
   mkl_dfti                    7813.913 us 500090.448 us    12.75       0.3%    0.031s  ok ch=2.7e-14/1e-10 1s=3e-15 1.62x
   mkl2026_dfti                7822.415 us 500634.536 us    12.74       0.3%    0.055s  ok ch=2.6e-14/1e-10 1s=3e-15 1.62x
   fftw3_patient               9908.409 us 634138.161 us    10.06       0.3%   17.369s  ok ch=2.3e-14/1e-10 1s=3e-15 2.05x
   fftw3_measure              11303.697 us 723436.586 us     8.82       7.2%    0.308s  ok ch=2.2e-14/1e-10 1s=3e-15 2.34x
   gen_twiddle                11375.761 us 728048.730 us     8.76       0.2%    0.100s  ok ch=2.5e-14/1e-10 1s=3e-15 2.36x
   ducc0_c2c                  11883.841 us 760565.797 us     8.39       0.8%    0.000s  ok ch=1.9e-14/1e-10 1s=3e-15 2.46x
   fftw3_guru                 14314.409 us 916122.195 us     6.96      10.7%    0.387s  ok ch=2.2e-14/1e-10 1s=3e-15 2.97x
   gen_layout                 15083.150 us 965321.577 us     6.61       2.0%    0.007s  ok ch=2.8e-14/1e-10 1s=4e-15 3.12x
   gen_bluestein              15621.150 us 999753.610 us     6.38       0.6%    0.000s  ok ch=3.0e-14/1e-10 1s=4e-15 3.24x
   fftw3_estimate             20593.164 us 1317962.510 us     4.84       0.7%    0.002s  ok ch=2.2e-14/1e-10 1s=3e-15 4.27x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_guru               FFTW 3.3.10 guru split-array dft, FFTW_MEASURE, fused split chain
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA register-explicit pencils (10=2x5,12=3x4,15=3x5,20=4x5; stage1 mem->regs, stage2 regs->mem, 2L ld + 2L st per pencil), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at 10/12/15, vdivpd at 20), sched-pressure on 10/12 only (r4 same-core re-race), THP arena (gen_layout), plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: pow2 radix-4/16 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody (axes 2+1 + sequential map sweep per L2-hot block), gen_twiddle exact tables
   gen_dense_prime          folded dense prime p<=31 (any prime in class supported): 4h^2-FMA conjugate-pair fold, z-pass fused into the x-contraction with the z-combine folded straight into U/V (no stack round-trip), fully in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, mask-free), register-tiled GEMM, vectorized any-L z-pass, zmm z-row fold, LAZY map fused into the next step's z-loads (s6 arithmetic, one vdivpd/8pts; only the last step's map materializes)
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement + stream audit & measured pitch picker, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, r3: packed cross-plane axis-1 lanes + trailing axis-2 through a 4-plane collision-picked window; r4: fold-load software prefetch (~L row streams beat the L2 streamer), NT full-line stores on DRAM-resident volumes, fused transpose+interleave scatter
   gen_pfa_large            GT-PFA 8x5 two-sweep (DFT25=5x5 CT exact tw) + owned chain (NR map); pick: l40-ip0 (wisdom) (B=8)
   gen_pfa_small            PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path; r3: generic runtime-table coprime P*Q engine (modules 2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63; r4: register-explicit 10/12 pencils, in-place generic pencils where Q==1 mod P (14,18,21,36,56)
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor (in-place, fused twiddles) + volume-resident fused chain, any 2<=L<=128; adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64: TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies), other 2^k in 2..128 generic radix-2
   gen_powp                 powp CT 3x9(3x3) exact tw, SoA-8 lane chain (DIF/DIT in place); pick: l27-soa (B=16)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race (interleaved sample-major since r4: core-state-drift immune) + per-host wisdom cache incl. string wisdom (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo = round-6 trunk: pln_enumerate trees (12, incl. sub-tree diversity) + tile width raced on the graded chain step by gr_pick, persisted, fused chain
   gen_rader                Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); any other prime via a generic folded half-system engine (runtime k-quad chunk kernel + transpose z-quads), in-place chain, self-check gated; s6 map adopted from gen_dense_prime
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp), consumption-order CT/DFT/Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo, self-audited at create(), owned in-place fused-map chain
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
