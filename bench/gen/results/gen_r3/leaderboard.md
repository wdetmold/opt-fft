```
=== round gen_r3 ===
# round gen_r3
host: a80n0.lqcd.mit   date: 2026-08-24T11:24:41-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.156 us 74001.455 us    43.09       0.1%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 1.00x
   gen_batchlane                  1.157 us 74033.496 us    43.08       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 1.00x
   mkl_dfti                       4.553 us 291362.641 us    10.95       0.2%    0.002s  ok ch=1.5e-13/1e-10 1s=9e-16 3.94x
   mkl2026_dfti                   4.634 us 296602.244 us    10.75       0.7%    0.001s  ok ch=1.8e-13/1e-10 1s=1e-15 4.01x
   gen_race                       4.718 us 301982.861 us    10.56       0.4%    0.002s  ok ch=1.5e-13/1e-10 1s=9e-16 4.08x
   gen_planner                    4.741 us 303414.719 us    10.51       1.4%    0.002s  ok ch=1.5e-13/1e-10 1s=9e-16 4.10x
   gen_layout                     4.904 us 313840.693 us    10.16       6.1%    0.000s  ok ch=1.5e-13/1e-10 1s=9e-16 4.24x
   fftw3_patient                  5.172 us 330978.031 us     9.64       4.4%    0.022s  ok ch=1.0e-13/1e-10 1s=8e-16 4.47x
   fftw3_measure                  5.188 us 332029.328 us     9.60       0.4%    0.013s  ok ch=1.1e-13/1e-10 1s=8e-16 4.49x
   gen_dense_prime                5.414 us 346521.278 us     9.20       9.1%    0.000s  ok ch=1.4e-13/1e-10 1s=9e-16 4.68x
   fftw3_estimate                 7.309 us 467763.551 us     6.82       0.7%    0.001s  ok ch=1.1e-13/1e-10 1s=8e-16 6.32x
   ducc0_c2c                      9.701 us 620888.827 us     5.14       1.4%    0.000s  ok ch=1.1e-13/1e-10 1s=7e-16 8.39x
   gen_twiddle                   10.709 us 685381.790 us     4.65       1.8%    0.002s  ok ch=1.4e-13/1e-10 1s=9e-16 9.26x
   gen_bluestein                 13.314 us 852118.685 us     3.74       0.3%    0.000s  ok ch=2.1e-13/1e-10 1s=1e-15 11.51x
   baseline_matrix               54.687 us 3499988.780 us     0.91       0.1%    0.000s  ok ch=3.4e-13/1e-10 1s=1e-15 47.30x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.915 us 73546.605 us    48.52       0.0%    0.000s  ok ch=5.3e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.970 us 75641.598 us    47.17      12.4%    0.000s  ok ch=5.0e-14/1e-10 1s=9e-16 1.03x
   gen_race                       6.634 us 254745.654 us    14.01       1.6%    0.006s  ok ch=5.0e-14/1e-10 1s=1e-15 3.46x
   gen_planner                    6.692 us 256958.854 us    13.89       1.6%    0.002s  ok ch=5.0e-14/1e-10 1s=1e-15 3.49x
   mkl_dfti                       7.747 us 297501.905 us    11.99       0.1%    0.002s  ok ch=4.9e-14/1e-10 1s=9e-16 4.05x
   mkl2026_dfti                   7.773 us 298497.649 us    11.95       2.1%    0.001s  ok ch=4.8e-14/1e-10 1s=9e-16 4.06x
   gen_layout                     8.009 us 307559.822 us    11.60       8.3%    0.000s  ok ch=4.6e-14/1e-10 1s=1e-15 4.18x
   gen_dense_prime                8.368 us 321337.125 us    11.10       5.1%    0.000s  ok ch=4.9e-14/1e-10 1s=9e-16 4.37x
   fftw3_measure                  8.697 us 333978.431 us    10.68       2.0%    0.014s  ok ch=4.7e-14/1e-10 1s=9e-16 4.54x
   fftw3_patient                  8.772 us 336862.895 us    10.59       2.1%    0.022s  ok ch=4.7e-14/1e-10 1s=9e-16 4.58x
   gen_twiddle                   14.958 us 574398.778 us     6.21       1.2%    0.002s  ok ch=4.7e-14/1e-10 1s=1e-15 7.81x
   ducc0_c2c                     15.967 us 613148.339 us     5.82       0.8%    0.000s  ok ch=3.2e-14/1e-10 1s=7e-16 8.34x
   fftw3_estimate                19.289 us 740705.400 us     4.82       0.2%    0.001s  ok ch=4.6e-14/1e-10 1s=9e-16 10.07x
   gen_bluestein                 19.361 us 743456.947 us     4.80       2.3%    0.000s  ok ch=9.0e-14/1e-10 1s=1e-15 10.11x
   baseline_matrix              112.110 us 4305004.850 us     0.83       0.0%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 58.53x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  4.464 us 85703.473 us    44.31       0.3%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.771 us 91598.781 us    41.46       2.3%    0.000s  ok ch=5.2e-14/1e-10 1s=1e-15 1.07x
   gen_race                      14.785 us 283865.284 us    13.38       1.1%    0.022s  ok ch=6.2e-14/1e-10 1s=1e-15 3.31x
   gen_planner                   14.874 us 285576.283 us    13.30       0.5%    0.004s  ok ch=6.2e-14/1e-10 1s=1e-15 3.33x
   gen_dense_prime               16.328 us 313492.155 us    12.11       7.2%    0.000s  ok ch=5.1e-14/1e-10 1s=1e-15 3.66x
   mkl_dfti                      16.464 us 316104.493 us    12.01       1.8%    0.002s  ok ch=5.7e-14/1e-10 1s=1e-15 3.69x
   mkl2026_dfti                  16.646 us 319598.770 us    11.88       0.2%    0.002s  ok ch=5.9e-14/1e-10 1s=1e-15 3.73x
   gen_layout                    18.621 us 357522.170 us    10.62       3.2%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 4.17x
   fftw3_measure                 19.730 us 378815.413 us    10.02       1.5%    0.010s  ok ch=5.1e-14/1e-10 1s=1e-15 4.42x
   fftw3_patient                 19.849 us 381109.331 us     9.96       0.2%    0.019s  ok ch=5.5e-14/1e-10 1s=1e-15 4.45x
   fftw3_estimate                21.035 us 403869.184 us     9.40       0.1%    0.001s  ok ch=5.1e-14/1e-10 1s=1e-15 4.71x
   gen_twiddle                   29.567 us 567685.395 us     6.69       0.3%    0.002s  ok ch=5.8e-14/1e-10 1s=1e-15 6.62x
   ducc0_c2c                     32.376 us 621616.711 us     6.11       4.3%    0.000s  ok ch=4.8e-14/1e-10 1s=1e-15 7.25x
   gen_bluestein                 33.634 us 645769.622 us     5.88       0.6%    0.000s  ok ch=1.0e-13/1e-10 1s=2e-15 7.53x
   baseline_matrix              272.499 us 5231976.180 us     0.73       0.0%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 61.05x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                 13.011 us 106582.374 us    39.86       3.9%    0.001s  ok ch=2.7e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                 13.257 us 108599.942 us    39.12       1.4%    0.001s  ok ch=2.6e-14/1e-10 1s=1e-15 1.02x
   gen_planner                   27.627 us 226323.320 us    18.77       0.5%    0.003s  ok ch=3.1e-14/1e-10 1s=1e-15 2.12x
   gen_race                      27.826 us 227953.522 us    18.64       0.4%    0.007s  ok ch=3.1e-14/1e-10 1s=1e-15 2.14x
   gen_layout                    40.346 us 330513.733 us    12.85       3.0%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 3.10x
   gen_dense_prime               43.509 us 356426.127 us    11.92       1.5%    0.000s  ok ch=3.1e-14/1e-10 1s=1e-15 3.34x
   fftw3_patient                 44.869 us 367565.657 us    11.56       2.3%    0.316s  ok ch=2.9e-14/1e-10 1s=1e-15 3.45x
   fftw3_measure                 44.899 us 367810.771 us    11.55       0.3%    0.084s  ok ch=2.8e-14/1e-10 1s=1e-15 3.45x
   gen_twiddle                   56.878 us 465944.253 us     9.12       4.2%    0.003s  ok ch=2.6e-14/1e-10 1s=1e-15 4.37x
   mkl2026_dfti                  57.440 us 470547.754 us     9.03       1.9%    0.049s  ok ch=3.3e-14/1e-10 1s=1e-15 4.41x
   mkl_dfti                      57.950 us 474724.360 us     8.95       0.4%    0.029s  ok ch=3.2e-14/1e-10 1s=1e-15 4.45x
   ducc0_c2c                     73.332 us 600732.559 us     7.07       1.2%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 5.64x
   fftw3_estimate                91.865 us 752554.155 us     5.65       0.1%    0.001s  ok ch=2.7e-14/1e-10 1s=1e-15 7.06x
   gen_bluestein                103.798 us 850313.972 us     5.00       0.7%    0.000s  ok ch=4.4e-14/1e-10 1s=2e-15 7.98x
   baseline_matrix              850.297 us 6965632.970 us     0.61       0.0%    0.000s  ok ch=9.9e-14/1e-10 1s=2e-15 65.35x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      32.208 us 131922.106 us    33.79       6.0%    0.136s  ok ch=3.1e-14/1e-10 1s=1e-15 1.00x
   gen_race                      69.317 us 283922.741 us    15.70       0.6%    0.005s  ok ch=3.4e-14/1e-10 1s=1e-15 2.15x
   gen_planner                   69.408 us 284296.590 us    15.68       0.2%    0.004s  ok ch=3.4e-14/1e-10 1s=1e-15 2.16x
   gen_layout                    92.999 us 380924.679 us    11.70       2.6%    0.000s  ok ch=3.4e-14/1e-10 1s=2e-15 2.89x
   fftw3_patient                109.921 us 450236.451 us     9.90       2.3%    0.111s  ok ch=3.6e-14/1e-10 1s=2e-15 3.41x
   fftw3_measure                111.284 us 455820.298 us     9.78       6.3%    0.028s  ok ch=3.6e-14/1e-10 1s=2e-15 3.46x
   mkl_dfti                     120.772 us 494680.154 us     9.01       0.1%    0.029s  ok ch=3.6e-14/1e-10 1s=2e-15 3.75x
   gen_twiddle                  120.961 us 495454.673 us     9.00       3.3%    0.003s  ok ch=3.5e-14/1e-10 1s=2e-15 3.76x
   mkl2026_dfti                 123.073 us 504108.687 us     8.84       0.9%    0.032s  ok ch=3.6e-14/1e-10 1s=2e-15 3.82x
   fftw3_estimate               130.592 us 534903.983 us     8.33       5.6%    0.001s  ok ch=3.6e-14/1e-10 1s=2e-15 4.05x
   ducc0_c2c                    145.508 us 595999.278 us     7.48       0.9%    0.000s  ok ch=2.6e-14/1e-10 1s=1e-15 4.52x
   gen_bluestein                178.965 us 733040.777 us     6.08       1.9%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 5.56x
   baseline_matrix             2061.229 us 8442795.900 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 64.00x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      44.811 us 143394.634 us    31.33       6.0%    0.201s  ok ch=2.9e-14/1e-10 1s=2e-15 1.00x
   gen_race                     104.537 us 334518.523 us    13.43       1.2%    0.005s  ok ch=2.7e-14/1e-10 1s=2e-15 2.33x
   gen_planner                  104.687 us 334997.812 us    13.41       0.8%    0.006s  ok ch=2.7e-14/1e-10 1s=2e-15 2.34x
   gen_layout                   124.173 us 397354.781 us    11.31       1.8%    0.001s  ok ch=3.0e-14/1e-10 1s=2e-15 2.77x
   mkl_dfti                     144.315 us 461808.440 us     9.73       0.2%    0.033s  ok ch=3.1e-14/1e-10 1s=2e-15 3.22x
   mkl2026_dfti                 148.253 us 474408.542 us     9.47       0.1%    0.047s  ok ch=3.1e-14/1e-10 1s=2e-15 3.31x
   ducc0_c2c                    188.561 us 603394.154 us     7.45       2.3%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 4.21x
   fftw3_patient                202.515 us 648049.030 us     6.93       4.2%    0.139s  ok ch=3.0e-14/1e-10 1s=2e-15 4.52x
   gen_bluestein                213.442 us 683015.418 us     6.58       1.0%    0.000s  ok ch=4.3e-14/1e-10 1s=2e-15 4.76x
   fftw3_measure                224.149 us 717276.563 us     6.26       3.8%    0.033s  ok ch=3.1e-14/1e-10 1s=2e-15 5.00x
   gen_twiddle                  235.358 us 753144.944 us     5.96       1.7%    0.004s  ok ch=3.0e-14/1e-10 1s=2e-15 5.25x
   fftw3_estimate               255.072 us 816230.088 us     5.50       1.6%    0.002s  ok ch=3.2e-14/1e-10 1s=2e-15 5.69x
   baseline_matrix             2799.109 us 8957149.830 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 62.47x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     85.088 us 190596.034 us    26.02       0.4%    0.008s  ok ch=3.2e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              123.828 us 277374.107 us    17.88       0.4%    0.001s  ok ch=2.9e-14/1e-10 1s=2e-15 1.46x
   gen_race                     195.633 us 438218.698 us    11.32       1.0%    0.007s  ok ch=2.8e-14/1e-10 1s=2e-15 2.30x
   gen_planner                  203.733 us 456362.278 us    10.87       0.1%    0.005s  ok ch=2.8e-14/1e-10 1s=2e-15 2.39x
   gen_layout                   226.722 us 507857.604 us     9.76       1.5%    0.001s  ok ch=2.9e-14/1e-10 1s=2e-15 2.66x
   gen_bluestein                301.272 us 674849.183 us     7.35       0.6%    0.000s  ok ch=4.7e-14/1e-10 1s=2e-15 3.54x
   gen_twiddle                  462.532 us 1036070.750 us     4.79       1.6%    0.006s  ok ch=3.0e-14/1e-10 1s=2e-15 5.44x
   ducc0_c2c                    715.949 us 1603724.760 us     3.09       0.8%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 8.41x
   mkl_dfti                     848.597 us 1900856.820 us     2.61       0.1%    0.030s  ok ch=3.3e-14/1e-10 1s=2e-15 9.97x
   fftw3_estimate               859.565 us 1925424.530 us     2.58       0.1%    0.002s  ok ch=2.7e-14/1e-10 1s=2e-15 10.10x
   fftw3_measure                859.984 us 1926365.140 us     2.57       0.3%    0.089s  ok ch=2.7e-14/1e-10 1s=2e-15 10.11x
   fftw3_patient                859.986 us 1926369.550 us     2.57       0.1%    0.225s  ok ch=2.7e-14/1e-10 1s=2e-15 10.11x
   mkl2026_dfti                 883.339 us 1978680.400 us     2.51       0.0%    0.054s  ok ch=3.3e-14/1e-10 1s=2e-15 10.38x
   baseline_matrix             4852.189 us 10868903.200 us     0.46       0.0%    0.000s  ok ch=8.6e-14/1e-10 1s=3e-15 57.03x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      57.322 us 114643.071 us    42.87       2.2%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 1.00x
   gen_race                     127.314 us 254627.757 us    19.30       1.8%    0.006s  ok ch=3.2e-14/1e-10 1s=1e-15 2.22x
   gen_planner                  128.262 us 256523.230 us    19.16       0.7%    0.005s  ok ch=3.2e-14/1e-10 1s=1e-15 2.24x
   mkl_dfti                     176.100 us 352199.962 us    13.96       4.2%    0.003s  ok ch=2.7e-14/1e-10 1s=1e-15 3.07x
   mkl2026_dfti                 187.189 us 374377.731 us    13.13       0.3%    0.002s  ok ch=2.7e-14/1e-10 1s=1e-15 3.27x
   fftw3_patient                208.409 us 416817.491 us    11.79       2.2%    0.497s  ok ch=2.7e-14/1e-10 1s=1e-15 3.64x
   fftw3_measure                210.502 us 421003.123 us    11.67       0.8%    0.082s  ok ch=3.0e-14/1e-10 1s=1e-15 3.67x
   gen_layout                   226.091 us 452181.674 us    10.87       2.9%    0.001s  ok ch=3.2e-14/1e-10 1s=2e-15 3.94x
   ducc0_c2c                    308.990 us 617980.310 us     7.95       2.0%    0.000s  ok ch=2.2e-14/1e-10 1s=1e-15 5.39x
   gen_bluestein                314.375 us 628749.295 us     7.82       1.0%    0.000s  ok ch=4.5e-14/1e-10 1s=2e-15 5.48x
   gen_twiddle                  360.667 us 721334.715 us     6.81       0.2%    0.006s  ok ch=3.0e-14/1e-10 1s=1e-15 6.29x
   fftw3_estimate               407.648 us 815295.421 us     6.03       0.5%    0.001s  ok ch=2.8e-14/1e-10 1s=1e-15 7.11x
   baseline_matrix             5752.451 us 11504902.600 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=2e-15 100.35x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                202.382 us 207238.946 us    25.24       0.8%    0.003s  ok ch=2.4e-14/1e-10 1s=2e-15 1.00x
   gen_race                     269.776 us 276250.757 us    18.94       0.0%    0.054s  ok ch=2.1e-14/1e-10 1s=2e-15 1.33x
   gen_planner                  277.752 us 284418.223 us    18.39       0.9%    0.018s  ok ch=2.1e-14/1e-10 1s=2e-15 1.37x
   mkl2026_dfti                 404.881 us 414598.031 us    12.62       0.1%    0.003s  ok ch=4.3e-14/1e-10 1s=2e-15 2.00x
   mkl_dfti                     405.851 us 415591.828 us    12.59       0.4%    0.002s  ok ch=2.5e-14/1e-10 1s=2e-15 2.01x
   gen_layout                   486.922 us 498607.936 us    10.49       1.4%    0.001s  ok ch=2.2e-14/1e-10 1s=2e-15 2.41x
   fftw3_patient                520.222 us 532707.551 us     9.82       2.9%    1.293s  ok ch=2.2e-14/1e-10 1s=2e-15 2.57x
   fftw3_measure                552.616 us 565878.841 us     9.25       1.1%    0.154s  ok ch=2.3e-14/1e-10 1s=2e-15 2.73x
   ducc0_c2c                    597.449 us 611787.376 us     8.55       2.6%    0.000s  ok ch=1.8e-14/1e-10 1s=1e-15 2.95x
   gen_twiddle                  653.413 us 669094.977 us     7.82       2.1%    0.008s  ok ch=2.3e-14/1e-10 1s=2e-15 3.23x
   gen_bluestein               1150.441 us 1178051.660 us     4.44       1.2%    0.000s  ok ch=3.2e-14/1e-10 1s=2e-15 5.68x
   fftw3_estimate              1607.317 us 1645892.950 us     3.18       0.3%    0.002s  ok ch=2.3e-14/1e-10 1s=2e-15 7.94x
   baseline_matrix            13458.264 us 13781262.000 us     0.38       0.1%    0.000s  ok ch=5.9e-14/1e-10 1s=3e-15 66.50x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                     473.678 us 242523.120 us    22.34       3.6%    0.849s  ok ch=4.0e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                473.988 us 242682.002 us    22.33       0.5%    0.899s  ok ch=4.0e-14/1e-10 1s=2e-15 1.00x
   gen_planner                  772.454 us 395496.679 us    13.70       1.0%    0.014s  ok ch=4.8e-14/1e-10 1s=2e-15 1.63x
   gen_race                     772.611 us 395576.708 us    13.70       1.5%    0.003s  ok ch=4.8e-14/1e-10 1s=2e-15 1.63x
   mkl_dfti                     947.159 us 484945.448 us    11.17       0.1%    0.032s  ok ch=4.9e-14/1e-10 1s=2e-15 2.00x
   mkl2026_dfti                 963.973 us 493554.426 us    10.98       0.1%    0.048s  ok ch=4.6e-14/1e-10 1s=2e-15 2.04x
   fftw3_patient               1171.259 us 599684.784 us     9.03       3.7%    1.325s  ok ch=4.4e-14/1e-10 1s=2e-15 2.47x
   fftw3_measure               1184.331 us 606377.265 us     8.94       5.4%    0.092s  ok ch=4.2e-14/1e-10 1s=2e-15 2.50x
   gen_layout                  1197.177 us 612954.870 us     8.84       1.1%    0.001s  ok ch=5.6e-14/1e-10 1s=3e-15 2.53x
   ducc0_c2c                   1269.447 us 649957.076 us     8.34       0.4%    0.000s  ok ch=4.4e-14/1e-10 1s=2e-15 2.68x
   gen_twiddle                 1382.591 us 707886.502 us     7.65       0.2%    0.012s  ok ch=4.4e-14/1e-10 1s=2e-15 2.92x
   fftw3_estimate              1626.369 us 832700.780 us     6.51       0.2%    0.003s  ok ch=4.3e-14/1e-10 1s=2e-15 3.43x
   gen_bluestein               1931.920 us 989143.079 us     5.48       2.2%    0.000s  ok ch=6.1e-14/1e-10 1s=3e-15 4.08x
   baseline_matrix            34089.938 us 17454048.200 us     0.31       2.1%    0.000s  ok ch=1.1e-13/1e-10 1s=4e-15 71.97x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                    5021.039 us 321346.474 us    19.85       0.8%    5.145s  ok ch=3.1e-14/1e-10 1s=3e-15 1.00x
   gen_pfa_large               5089.048 us 325699.073 us    19.58       1.9%    0.003s  ok ch=3.1e-14/1e-10 1s=3e-15 1.01x
   gen_race                    6198.286 us 396690.278 us    16.08       2.1%    0.006s  ok ch=3.1e-14/1e-10 1s=3e-15 1.23x
   gen_planner                 6219.999 us 398079.937 us    16.02       1.0%    0.083s  ok ch=2.6e-14/1e-10 1s=3e-15 1.24x
   mkl_dfti                    7784.041 us 498178.653 us    12.80       0.6%    0.013s  ok ch=3.0e-14/1e-10 1s=3e-15 1.55x
   mkl2026_dfti                7875.033 us 504002.124 us    12.65       3.1%    0.048s  ok ch=3.0e-14/1e-10 1s=3e-15 1.57x
   fftw3_patient               9955.561 us 637155.934 us    10.01       1.1%   17.439s  ok ch=2.9e-14/1e-10 1s=3e-15 1.98x
   fftw3_measure              11058.705 us 707757.119 us     9.01       0.8%    0.278s  ok ch=3.8e-14/1e-10 1s=3e-15 2.20x
   gen_twiddle                11346.713 us 726189.654 us     8.78       0.7%    0.105s  ok ch=2.9e-14/1e-10 1s=3e-15 2.26x
   ducc0_c2c                  11827.305 us 756947.527 us     8.43       0.8%    0.000s  ok ch=2.3e-14/1e-10 1s=2e-15 2.36x
   gen_bluestein              15741.831 us 1007477.210 us     6.33       0.3%    0.000s  ok ch=3.7e-14/1e-10 1s=4e-15 3.14x
   gen_layout                 18602.660 us 1190570.220 us     5.36       1.0%    0.007s  ok ch=3.1e-14/1e-10 1s=3e-15 3.70x
   fftw3_estimate             20607.121 us 1318855.740 us     4.84       0.7%    0.002s  ok ch=2.9e-14/1e-10 1s=3e-15 4.10x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA register-explicit pencils (10=2x5,12=3x4,15=3x5,20=4x5; stage1 mem->regs, stage2 regs->mem, 2L ld + 2L st per pencil), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at 10/12/15, vdivpd at 20), per-size sched-pressure, THP arena (gen_layout), plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: pow2 radix-4/16 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain with the rsqrt14/rcp14 ladder fused into the axis-0 scatter, gen_twiddle exact tables
   gen_dense_prime          folded dense prime p<=31 (any prime in class supported): 4h^2-FMA conjugate-pair fold, z-pass fused into the x-contraction with the z-combine folded straight into U/V (no stack round-trip), fully in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, mask-free), register-tiled GEMM, vectorized any-L z-pass, separate s6 map (one vdivpd/8pts)
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement + stream audit & measured pitch picker, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, r3: packed cross-plane axis-1 lanes + trailing axis-2 through a 4-plane collision-picked window
   gen_pfa_large            GT-PFA 25x4 two-sweep (DFT25=5x5 CT exact tw) + owned chain (NR map); pick: l100-ip1 (wisdom) (B=1)
   gen_pfa_small            PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path; r3: generic runtime-table coprime P*Q engine (modules 2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor (in-place, fused twiddles) + volume-resident fused chain, any 2<=L<=128; adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: L32 custody split-complex chain (4x8 z-pair TR8, lazy exact map, DSB-resident loop bodies), any other 2^k in 2..128 generic radix-2
   gen_powp                 powp CT GT 25x4(5x5) exact tw, two-sweep + owned chain (NR map); pick: l100-ip0 (B=1)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race + per-host wisdom cache incl. string wisdom (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo = round-6 trunk: pln_enumerate trees + tile width raced on the graded chain step by gr_pick, winner persisted, fused chain
   gen_rader                Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); any other prime via a generic folded half-system engine (runtime k-quad chunk kernel + transpose z-quads), in-place chain, self-check gated; s6 map adopted from gen_dense_prime
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp), consumption-order CT/DFT/Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo, self-audited at create(), owned in-place fused-map chain
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
