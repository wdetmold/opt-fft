```
=== round gen_r6 ===
# round gen_r6
host: a80n0.lqcd.mit   date: 2026-08-25T03:21:23-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.155 us 73939.219 us    43.13      14.0%    0.000s  ok ch=1.7e-13/1e-10 1s=8e-16 1.00x
   gen_pfa_small                  1.156 us 73987.295 us    43.10       0.0%    0.000s  ok ch=1.4e-13/1e-10 1s=9e-16 1.00x
   gen_planner                    1.418 us 90765.153 us    35.14       0.8%    0.005s  ok ch=2.4e-13/1e-10 1s=9e-16 1.23x
   gen_race                       1.424 us 91161.136 us    34.98      13.8%    0.046s  ok ch=2.4e-13/1e-10 1s=9e-16 1.23x
   mkl_dfti                       4.569 us 292445.444 us    10.90       0.2%    0.002s  ok ch=1.5e-13/1e-10 1s=1e-15 3.96x
   mkl2026_dfti                   4.646 us 297333.983 us    10.73       0.2%    0.002s  ok ch=1.4e-13/1e-10 1s=1e-15 4.02x
   fftw3_measure                  5.136 us 328691.305 us     9.70       1.6%    0.013s  ok ch=1.2e-13/1e-10 1s=9e-16 4.45x
   fftw3_patient                  5.185 us 331830.879 us     9.61       0.5%    0.022s  ok ch=1.2e-13/1e-10 1s=8e-16 4.49x
   gen_layout                     5.200 us 332801.039 us     9.58       1.3%    0.000s  ok ch=1.6e-13/1e-10 1s=9e-16 4.50x
   gen_dense_prime                5.431 us 347565.700 us     9.18       3.0%    0.000s  ok ch=1.7e-13/1e-10 1s=9e-16 4.70x
   gen_twiddle                    6.249 us 399944.005 us     7.97       1.0%    0.003s  ok ch=1.9e-13/1e-10 1s=9e-16 5.41x
   fftw3_guru                     6.366 us 407415.412 us     7.83       0.1%    0.011s  ok ch=1.3e-13/1e-10 1s=9e-16 5.51x
   fftw3_estimate                 7.316 us 468204.599 us     6.81       0.3%    0.001s  ok ch=1.2e-13/1e-10 1s=9e-16 6.33x
   ducc0_c2c                      9.675 us 619195.489 us     5.15       3.4%    0.000s  ok ch=9.3e-14/1e-10 1s=7e-16 8.37x
   gen_bluestein                 13.410 us 858246.560 us     3.72       0.6%    0.000s  ok ch=2.3e-13/1e-10 1s=1e-15 11.61x
   baseline_matrix               54.687 us 3499985.700 us     0.91       0.0%    0.000s  ok ch=3.6e-13/1e-10 1s=1e-15 47.34x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.911 us 73387.330 us    48.62       0.1%    0.000s  ok ch=5.0e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  1.917 us 73603.719 us    48.48       0.1%    0.000s  ok ch=5.0e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.467 us 94721.981 us    37.67       3.2%    0.006s  ok ch=8.9e-14/1e-10 1s=1e-15 1.29x
   gen_race                       2.471 us 94873.573 us    37.61       5.1%    0.056s  ok ch=5.9e-14/1e-10 1s=9e-16 1.29x
   mkl_dfti                       7.765 us 298178.925 us    11.97       0.1%    0.002s  ok ch=5.6e-14/1e-10 1s=9e-16 4.06x
   mkl2026_dfti                   7.792 us 299206.629 us    11.93       0.5%    0.002s  ok ch=5.4e-14/1e-10 1s=9e-16 4.08x
   gen_dense_prime                7.958 us 305582.360 us    11.68       4.3%    0.000s  ok ch=1.0e-13/1e-10 1s=9e-16 4.16x
   gen_layout                     8.244 us 316554.278 us    11.27       0.6%    0.000s  ok ch=5.5e-14/1e-10 1s=9e-16 4.31x
   fftw3_patient                  8.517 us 327069.008 us    10.91       2.6%    0.023s  ok ch=7.8e-14/1e-10 1s=9e-16 4.46x
   fftw3_measure                  8.591 us 329912.399 us    10.82       1.0%    0.014s  ok ch=7.8e-14/1e-10 1s=9e-16 4.50x
   gen_twiddle                    8.714 us 334621.779 us    10.66       6.2%    0.003s  ok ch=9.5e-14/1e-10 1s=9e-16 4.56x
   fftw3_guru                    10.219 us 392392.757 us     9.09       1.0%    0.010s  ok ch=5.4e-14/1e-10 1s=9e-16 5.35x
   ducc0_c2c                     16.040 us 615924.681 us     5.79       0.6%    0.000s  ok ch=6.5e-14/1e-10 1s=7e-16 8.39x
   fftw3_estimate                19.295 us 740939.236 us     4.82       0.1%    0.001s  ok ch=7.8e-14/1e-10 1s=9e-16 10.10x
   gen_bluestein                 19.390 us 744573.292 us     4.79       1.2%    0.000s  ok ch=1.0e-13/1e-10 1s=1e-15 10.15x
   baseline_matrix              112.155 us 4306751.430 us     0.83       0.0%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 58.69x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  4.406 us 84590.169 us    44.89       0.3%    0.000s  ok ch=5.2e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.412 us 84703.081 us    44.83       0.2%    0.000s  ok ch=5.2e-14/1e-10 1s=1e-15 1.00x
   gen_planner                    5.518 us 105938.780 us    35.85       1.9%    0.004s  ok ch=5.3e-14/1e-10 1s=1e-15 1.25x
   gen_race                       5.591 us 107344.788 us    35.38       3.5%    0.007s  ok ch=5.3e-14/1e-10 1s=1e-15 1.27x
   gen_dense_prime               14.846 us 285045.552 us    13.32       7.2%    0.000s  ok ch=4.8e-14/1e-10 1s=1e-15 3.37x
   mkl_dfti                      16.557 us 317894.705 us    11.95       0.6%    0.003s  ok ch=5.9e-14/1e-10 1s=1e-15 3.76x
   mkl2026_dfti                  16.695 us 320552.742 us    11.85       0.0%    0.001s  ok ch=6.1e-14/1e-10 1s=1e-15 3.79x
   gen_layout                    18.810 us 361152.734 us    10.51       2.7%    0.000s  ok ch=4.6e-14/1e-10 1s=1e-15 4.27x
   gen_twiddle                   19.613 us 376561.471 us    10.08       1.8%    0.003s  ok ch=5.5e-14/1e-10 1s=1e-15 4.45x
   fftw3_patient                 19.678 us 377809.366 us    10.05       0.1%    0.019s  ok ch=5.0e-14/1e-10 1s=1e-15 4.47x
   fftw3_measure                 19.929 us 382646.005 us     9.92       1.6%    0.011s  ok ch=5.0e-14/1e-10 1s=1e-15 4.52x
   fftw3_estimate                20.713 us 397682.795 us     9.55       1.5%    0.001s  ok ch=5.0e-14/1e-10 1s=1e-15 4.70x
   fftw3_guru                    26.182 us 502702.976 us     7.55       0.8%    0.010s  ok ch=5.4e-14/1e-10 1s=1e-15 5.94x
   ducc0_c2c                     32.427 us 622605.795 us     6.10       2.0%    0.000s  ok ch=4.2e-14/1e-10 1s=1e-15 7.36x
   gen_bluestein                 34.064 us 654035.495 us     5.81       0.5%    0.000s  ok ch=8.6e-14/1e-10 1s=2e-15 7.73x
   baseline_matrix              272.298 us 5228129.190 us     0.73       0.1%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 61.81x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                 12.866 us 105398.203 us    40.31       1.5%    0.001s  ok ch=3.3e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                 12.867 us 105408.788 us    40.31       1.5%    0.001s  ok ch=3.3e-14/1e-10 1s=1e-15 1.00x
   gen_planner                   18.111 us 148363.542 us    28.64       5.7%    0.017s  ok ch=4.0e-14/1e-10 1s=1e-15 1.41x
   gen_race                      19.145 us 156831.907 us    27.09       2.5%    0.064s  ok ch=3.5e-14/1e-10 1s=1e-15 1.49x
   gen_twiddle                   35.991 us 294837.950 us    14.41       3.8%    0.004s  ok ch=4.0e-14/1e-10 1s=1e-15 2.80x
   gen_layout                    38.117 us 312253.908 us    13.61       2.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 2.96x
   gen_dense_prime               38.872 us 318439.140 us    13.34       3.2%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 3.02x
   fftw3_measure                 44.810 us 367084.258 us    11.57       1.0%    0.082s  ok ch=3.1e-14/1e-10 1s=1e-15 3.48x
   fftw3_patient                 45.383 us 371779.294 us    11.43       1.2%    0.299s  ok ch=3.4e-14/1e-10 1s=1e-15 3.53x
   mkl2026_dfti                  57.439 us 470540.834 us     9.03       2.5%    0.055s  ok ch=3.7e-14/1e-10 1s=1e-15 4.46x
   mkl_dfti                      58.055 us 475585.084 us     8.93       1.1%    0.049s  ok ch=3.3e-14/1e-10 1s=1e-15 4.51x
   fftw3_guru                    60.104 us 492371.077 us     8.63       1.5%    0.077s  ok ch=3.5e-14/1e-10 1s=1e-15 4.67x
   ducc0_c2c                     72.856 us 596838.605 us     7.12       1.8%    0.000s  ok ch=2.7e-14/1e-10 1s=1e-15 5.66x
   gen_bluestein                 82.906 us 679169.045 us     6.26       0.6%    0.000s  ok ch=5.3e-14/1e-10 1s=2e-15 6.44x
   fftw3_estimate                91.909 us 752919.969 us     5.64       0.4%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 7.14x
   baseline_matrix              850.302 us 6965677.070 us     0.61       0.0%    0.000s  ok ch=1.0e-13/1e-10 1s=2e-15 66.09x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      31.971 us 130952.481 us    34.04       0.1%    0.005s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   39.681 us 162532.868 us    27.43       2.2%    0.026s  ok ch=3.5e-14/1e-10 1s=2e-15 1.24x
   gen_race                      41.248 us 168950.904 us    26.39       1.1%    0.054s  ok ch=3.5e-14/1e-10 1s=2e-15 1.29x
   gen_twiddle                   78.837 us 322916.708 us    13.81       2.6%    0.003s  ok ch=3.3e-14/1e-10 1s=2e-15 2.47x
   gen_layout                    96.057 us 393448.593 us    11.33       0.9%    0.001s  ok ch=3.8e-14/1e-10 1s=2e-15 3.00x
   fftw3_measure                108.193 us 443160.503 us    10.06       2.1%    0.028s  ok ch=3.5e-14/1e-10 1s=2e-15 3.38x
   fftw3_patient                108.286 us 443540.900 us    10.05       3.0%    0.110s  ok ch=3.7e-14/1e-10 1s=2e-15 3.39x
   mkl_dfti                     120.379 us 493073.041 us     9.04       0.6%    0.048s  ok ch=3.4e-14/1e-10 1s=2e-15 3.77x
   mkl2026_dfti                 123.899 us 507488.379 us     8.78       0.1%    0.048s  ok ch=3.3e-14/1e-10 1s=2e-15 3.88x
   fftw3_estimate               129.952 us 532284.512 us     8.38       3.9%    0.001s  ok ch=3.7e-14/1e-10 1s=2e-15 4.06x
   ducc0_c2c                    145.730 us 596910.540 us     7.47       3.4%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 4.56x
   fftw3_guru                   174.932 us 716521.757 us     6.22       7.5%    0.029s  ok ch=3.8e-14/1e-10 1s=2e-15 5.47x
   gen_bluestein                180.573 us 739627.290 us     6.03       1.4%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 5.65x
   baseline_matrix             2061.229 us 8442792.350 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 64.47x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      44.544 us 142541.386 us    31.52       2.2%    0.005s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_race                      59.016 us 188849.842 us    23.79       2.2%    0.083s  ok ch=3.0e-14/1e-10 1s=2e-15 1.32x
   gen_planner                   60.373 us 193193.682 us    23.25       0.3%    0.032s  ok ch=3.0e-14/1e-10 1s=2e-15 1.36x
   gen_layout                   124.086 us 397073.966 us    11.31       0.8%    0.000s  ok ch=3.5e-14/1e-10 1s=2e-15 2.79x
   gen_twiddle                  130.063 us 416202.094 us    10.79       2.2%    0.005s  ok ch=3.0e-14/1e-10 1s=2e-15 2.92x
   mkl_dfti                     144.394 us 462060.409 us     9.72       0.1%    0.049s  ok ch=3.1e-14/1e-10 1s=2e-15 3.24x
   mkl2026_dfti                 148.351 us 474722.260 us     9.46       0.3%    0.048s  ok ch=3.1e-14/1e-10 1s=2e-15 3.33x
   ducc0_c2c                    189.893 us 607658.934 us     7.39       1.1%    0.000s  ok ch=2.7e-14/1e-10 1s=1e-15 4.26x
   fftw3_patient                205.023 us 656074.485 us     6.85       1.9%    0.139s  ok ch=3.1e-14/1e-10 1s=2e-15 4.60x
   gen_bluestein                214.669 us 686939.689 us     6.54       1.5%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 4.82x
   fftw3_measure                223.404 us 714894.278 us     6.28       5.5%    0.034s  ok ch=3.1e-14/1e-10 1s=2e-15 5.02x
   fftw3_estimate               256.038 us 819320.105 us     5.48       0.2%    0.003s  ok ch=3.2e-14/1e-10 1s=2e-15 5.75x
   fftw3_guru                   258.240 us 826369.551 us     5.44       1.8%    0.044s  ok ch=2.9e-14/1e-10 1s=1e-15 5.80x
   baseline_matrix             2799.172 us 8957350.720 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 62.84x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     84.801 us 189954.931 us    26.11       0.9%    0.007s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              120.708 us 270385.519 us    18.34       0.7%    0.001s  ok ch=4.2e-14/1e-10 1s=2e-15 1.42x
   gen_planner                  138.210 us 309590.038 us    16.02       2.0%    0.044s  ok ch=3.0e-14/1e-10 1s=2e-15 1.63x
   gen_race                     138.928 us 311199.154 us    15.94       1.0%    0.105s  ok ch=3.0e-14/1e-10 1s=2e-15 1.64x
   gen_layout                   200.777 us 449740.353 us    11.03       0.1%    0.001s  ok ch=5.3e-14/1e-10 1s=2e-15 2.37x
   gen_twiddle                  268.617 us 601702.607 us     8.24       2.8%    0.006s  ok ch=3.9e-14/1e-10 1s=2e-15 3.17x
   gen_bluestein                288.579 us 646416.433 us     7.67       1.7%    0.000s  ok ch=7.8e-14/1e-10 1s=3e-15 3.40x
   ducc0_c2c                    719.293 us 1611215.450 us     3.08       1.7%    0.000s  ok ch=4.3e-14/1e-10 1s=1e-15 8.48x
   fftw3_guru                   833.660 us 1867397.750 us     2.66       0.3%    0.088s  ok ch=3.3e-14/1e-10 1s=2e-15 9.83x
   mkl_dfti                     848.537 us 1900722.670 us     2.61       0.1%    0.051s  ok ch=8.2e-14/1e-10 1s=2e-15 10.01x
   fftw3_estimate               859.080 us 1924340.100 us     2.58       0.1%    0.002s  ok ch=3.3e-14/1e-10 1s=2e-15 10.13x
   fftw3_measure                859.434 us 1925132.400 us     2.58       0.1%    0.089s  ok ch=3.3e-14/1e-10 1s=2e-15 10.13x
   fftw3_patient                859.593 us 1925488.770 us     2.58       0.4%    0.240s  ok ch=3.3e-14/1e-10 1s=2e-15 10.14x
   mkl2026_dfti                 883.079 us 1978097.050 us     2.51       0.0%    0.048s  ok ch=8.2e-14/1e-10 1s=2e-15 10.41x
   baseline_matrix             4852.701 us 10870050.700 us     0.46       0.0%    0.000s  ok ch=8.3e-14/1e-10 1s=3e-15 57.22x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      56.524 us 113048.793 us    43.48       0.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_race                     105.534 us 211068.679 us    23.29       5.6%    0.152s  ok ch=3.4e-14/1e-10 1s=2e-15 1.87x
   gen_planner                  105.717 us 211434.297 us    23.25       3.5%    0.006s  ok ch=3.4e-14/1e-10 1s=2e-15 1.87x
   mkl_dfti                     172.086 us 344171.182 us    14.28       0.5%    0.002s  ok ch=3.2e-14/1e-10 1s=1e-15 3.04x
   gen_layout                   172.732 us 345463.641 us    14.23       1.8%    0.001s  ok ch=3.0e-14/1e-10 1s=2e-15 3.06x
   mkl2026_dfti                 187.565 us 375130.277 us    13.10       3.8%    0.003s  ok ch=3.1e-14/1e-10 1s=1e-15 3.32x
   gen_twiddle                  195.573 us 391145.104 us    12.57       1.4%    0.006s  ok ch=3.7e-14/1e-10 1s=2e-15 3.46x
   fftw3_patient                208.633 us 417266.291 us    11.78       1.4%    0.492s  ok ch=3.1e-14/1e-10 1s=1e-15 3.69x
   fftw3_measure                210.143 us 420286.830 us    11.69       0.9%    0.086s  ok ch=3.1e-14/1e-10 1s=1e-15 3.72x
   fftw3_guru                   298.974 us 597948.644 us     8.22       1.9%    0.072s  ok ch=3.2e-14/1e-10 1s=1e-15 5.29x
   ducc0_c2c                    310.307 us 620613.135 us     7.92       0.4%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 5.49x
   gen_bluestein                314.695 us 629389.513 us     7.81       0.8%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 5.57x
   fftw3_estimate               407.759 us 815517.600 us     6.03       0.3%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 7.21x
   baseline_matrix             5752.387 us 11504773.500 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=3e-15 101.77x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                160.560 us 164413.164 us    31.82       1.2%    1.176s  ok ch=2.1e-14/1e-10 1s=2e-15 1.00x
   gen_race                     236.997 us 242684.587 us    21.56       1.8%    0.010s  ok ch=2.1e-14/1e-10 1s=2e-15 1.48x
   gen_planner                  239.911 us 245668.606 us    21.30       1.4%    0.104s  ok ch=2.4e-14/1e-10 1s=2e-15 1.49x
   gen_twiddle                  339.232 us 347373.937 us    15.06       1.8%    0.006s  ok ch=2.3e-14/1e-10 1s=2e-15 2.11x
   gen_layout                   391.754 us 401156.005 us    13.04       2.1%    0.001s  ok ch=2.1e-14/1e-10 1s=2e-15 2.44x
   mkl2026_dfti                 404.415 us 414120.502 us    12.63       0.6%    0.002s  ok ch=4.0e-14/1e-10 1s=2e-15 2.52x
   mkl_dfti                     405.615 us 415349.375 us    12.60       0.4%    0.003s  ok ch=2.5e-14/1e-10 1s=2e-15 2.53x
   fftw3_patient                520.007 us 532487.415 us     9.82       0.5%    1.474s  ok ch=2.1e-14/1e-10 1s=2e-15 3.24x
   fftw3_measure                559.300 us 572723.006 us     9.13       1.0%    0.154s  ok ch=2.3e-14/1e-10 1s=2e-15 3.48x
   ducc0_c2c                    599.803 us 614198.771 us     8.52       1.0%    0.000s  ok ch=1.8e-14/1e-10 1s=1e-15 3.74x
   gen_bluestein                620.182 us 635066.805 us     8.24       0.6%    0.000s  ok ch=3.5e-14/1e-10 1s=3e-15 3.86x
   fftw3_guru                   680.223 us 696547.918 us     7.51       3.3%    0.157s  ok ch=2.1e-14/1e-10 1s=2e-15 4.24x
   fftw3_estimate              1608.970 us 1647584.790 us     3.18       0.2%    0.002s  ok ch=2.1e-14/1e-10 1s=2e-15 10.02x
   baseline_matrix            13461.066 us 13784131.800 us     0.38       0.1%    0.000s  ok ch=5.7e-14/1e-10 1s=2e-15 83.84x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                     415.524 us 212748.364 us    25.47       3.4%    0.004s  ok ch=3.3e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                420.018 us 215049.307 us    25.19       1.5%    0.003s  ok ch=3.3e-14/1e-10 1s=2e-15 1.01x
   gen_planner                  566.668 us 290133.940 us    18.67       0.5%    0.015s  ok ch=3.1e-14/1e-10 1s=2e-15 1.36x
   gen_race                     567.466 us 290542.504 us    18.65       0.2%    0.078s  ok ch=3.1e-14/1e-10 1s=2e-15 1.37x
   gen_twiddle                  747.660 us 382801.997 us    14.15       1.3%    0.012s  ok ch=3.0e-14/1e-10 1s=2e-15 1.80x
   gen_layout                   941.267 us 481928.926 us    11.24       1.6%    0.001s  ok ch=3.0e-14/1e-10 1s=2e-15 2.27x
   mkl_dfti                     943.923 us 483288.567 us    11.21       0.4%    0.050s  ok ch=3.2e-14/1e-10 1s=2e-15 2.27x
   mkl2026_dfti                 960.449 us 491749.941 us    11.02       0.5%    0.052s  ok ch=3.0e-14/1e-10 1s=2e-15 2.31x
   fftw3_patient               1160.001 us 593920.295 us     9.12       4.4%    1.236s  ok ch=3.1e-14/1e-10 1s=2e-15 2.79x
   fftw3_measure               1195.204 us 611944.499 us     8.85       2.2%    0.088s  ok ch=3.2e-14/1e-10 1s=2e-15 2.88x
   ducc0_c2c                   1283.135 us 656965.047 us     8.25       1.4%    0.000s  ok ch=2.5e-14/1e-10 1s=2e-15 3.09x
   fftw3_estimate              1612.430 us 825564.304 us     6.56       1.5%    0.002s  ok ch=3.2e-14/1e-10 1s=2e-15 3.88x
   fftw3_guru                  1661.764 us 850823.118 us     6.37       0.7%    0.108s  ok ch=3.1e-14/1e-10 1s=2e-15 4.00x
   gen_bluestein               1771.651 us 907085.343 us     5.97       1.1%    0.000s  ok ch=4.0e-14/1e-10 1s=3e-15 4.26x
   baseline_matrix            34344.207 us 17584233.800 us     0.31       1.4%    0.000s  ok ch=8.4e-14/1e-10 1s=4e-15 82.65x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large               4570.267 us 292497.072 us    21.81       1.0%    0.004s  ok ch=2.0e-14/1e-10 1s=3e-15 1.00x
   gen_powp                    4617.510 us 295520.636 us    21.58       8.3%    0.003s  ok ch=2.0e-14/1e-10 1s=3e-15 1.01x
   gen_planner                 5058.392 us 323737.116 us    19.70       2.4%    0.315s  ok ch=2.2e-14/1e-10 1s=3e-15 1.11x
   gen_race                    5200.083 us 332805.329 us    19.16       1.4%    0.708s  ok ch=2.2e-14/1e-10 1s=3e-15 1.14x
   gen_twiddle                 7734.618 us 495015.531 us    12.88       4.1%    0.086s  ok ch=2.2e-14/1e-10 1s=3e-15 1.69x
   mkl_dfti                    7797.120 us 499015.659 us    12.78       0.3%    0.049s  ok ch=2.4e-14/1e-10 1s=3e-15 1.71x
   mkl2026_dfti                7800.217 us 499213.892 us    12.78       0.6%    0.055s  ok ch=2.5e-14/1e-10 1s=3e-15 1.71x
   fftw3_patient               9961.371 us 637527.742 us    10.00       1.1%   17.506s  ok ch=1.9e-14/1e-10 1s=3e-15 2.18x
   fftw3_measure              11148.960 us 713533.416 us     8.94       8.6%    0.306s  ok ch=2.0e-14/1e-10 1s=3e-15 2.44x
   ducc0_c2c                  11910.594 us 762277.994 us     8.37       0.4%    0.000s  ok ch=1.7e-14/1e-10 1s=2e-15 2.61x
   gen_layout                 12099.883 us 774392.531 us     8.24       0.9%    0.004s  ok ch=2.3e-14/1e-10 1s=3e-15 2.65x
   fftw3_guru                 14348.435 us 918299.828 us     6.95      10.5%    0.384s  ok ch=2.2e-14/1e-10 1s=3e-15 3.14x
   gen_bluestein              15562.968 us 996029.924 us     6.40       2.5%    0.000s  ok ch=2.8e-14/1e-10 1s=4e-15 3.41x
   fftw3_estimate             20588.506 us 1317664.410 us     4.84       0.4%    0.002s  ok ch=2.0e-14/1e-10 1s=3e-15 4.50x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_guru               FFTW 3.3.10 guru split-array dft, FFTW_MEASURE, fused split chain
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA pencils (10=2x5,12=3x4,15=3x5,20=4x5; r6 adds 7-smooth 14/21/28/35 via a DFT7 module), register-explicit at 10/12 (2L ld + 2L st), memory form elsewhere with r6 SAFE PLACEMENT (stage-1 store permutation makes every stage-2 group in-place on its own mod-P residue class -- no fused-pair hazards at any coprime split), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at 10/12, vdivpd elsewhere), sched-pressure on 10/12 only, THP arena (gen_layout), plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: {2,3,5}*2^k radix-4/16 DIF/DIT convolution (no bit-reversal; twiddle-free DFT-3/5 and PFA-6/10 fused middles cut M up to 33% below next_pow2), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody with the map fused into the axis-1 scatter reading a custody-ordered c (two aligned sequential streams; gen_pow2 GP2_CT), gen_twiddle exact tables
   gen_dense_prime          folded dense prime p<=31 (any prime in class supported): 4h^2-FMA conjugate-pair fold, z-pass fused into the x-contraction with the z-combine folded straight into U/V (no stack round-trip), fully in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, mask-free), register-tiled EXACT-TILE GEMM (full 4-zmm d-chunks + one exact 1..3-zmm tail, exact k-tails -- no wasted FMA slots at any L), vectorized any-L z-pass, zmm z-row fold, LAZY map fused into the next step's z-loads at L=31 (only the last step's map materializes); generic custody/lazy-map generalizations raced and rejected in gen_r6 (kept as cross-arch knobs)
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement + stream audit & measured pitch picker, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, r3: packed cross-plane axis-1 lanes + trailing axis-2 through a 4-plane collision-picked window; r4: fold-load software prefetch (~L row streams beat the L2 streamer), NT full-line stores on DRAM-resident volumes, fused transpose+interleave scatter; r5: graded map fused into the axis-2 exit (gl_map8/gl_map16 in-register map primitives), chain in place, zt volume deleted; r6: even-L second-level fold (k-parity split over j<->L/2-j, halves the kernel j-sweep at every even L)
   gen_pfa_large            GT-PFA 25x4 two-sweep (exact-tw modules) + owned chain (deferred NR map); pick: l100-ipp1 (wisdom) (B=1)
   gen_pfa_small            PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path; r3: generic runtime-table coprime P*Q engine (modules 2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63; r4: register-explicit 10/12 pencils, in-place generic pencils where Q==1 mod P (14,18,21,36,56); r5: per-size map ladder BODY+tail (bl hs-form + rcp at 12, bl + div at 15/20, legacy + div at 10), raced same-core; r6: modules widened to {2,3,4,5,7,8,9,11,13,15,16,17,19,21,23,25,27,29,31} + nested-PFA composite odd modules (21,33,35,39,45,51,55,57,63) -- 53 new sizes, all coprime P*Q in 14..127 except 50/80/100 (pfa_large/powp cells)
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor (in-place, fused twiddles) + volume-resident fused chain, any 2<=L<=128; adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64: TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies, dual-select FMA-folded twiddles), other 2^k in 2..128 generic radix-2
   gen_powp                 powp CT 3x9(3x3) exact tw, SoA-8 lane chain (DIF/DIT in place); pick: l27-soa (B=16)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race (interleaved sample-major since r4: core-state-drift immune) + per-host wisdom cache incl. string wisdom + round-end drop_prefix (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo = round-6 trunk: pln_enumerate trees + gen_planner split-group batch-lane arms (@s1/2/3 + r6 staged @s4, batch>=8, 6-arm cap) + tile width + NEW fm race (both engines' runtime fused-map boundary, an ICX-tuned constant, raced per host in place), all raced on the graded chain step by gr_pick, persisted, fused chain
   gen_rader                Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); 43/67/79/103 via OUTER-C3 Rader (the same Winograd-C3-over-dense-blocks at runtime tables, 8m^2 vs 18m^2 conv FMA); even-h primes (p=1 mod 4, 13..113) via the E-side C2 CRT split + O-side negacyclic Karatsuba (5m^2 vs 8m^2); any other prime via the generic folded half-system engine; generic chain on a padded gl_map_huge arena for p<=61 (alias-free pitch, tail-free x/y), flat above; self-check gated; s6 map from gen_dense_prime, arena from gen_layout
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp) + NEW dual-select FMA form tw_cis_ds (lit 11 Tier 1: every stored ratio <=1, first performant validation), consumption-order CT/DFT/Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo (gen_r5: conjugate-fold prime butterflies; gen_r6: register-resident whole-level codelets for r=2/3/4/5, -30..-44% bit-identical), self-audited at create(), owned in-place fused-map chain
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
