```
=== round gen_r1 ===
# round gen_r1
host: a80n0.lqcd.mit   date: 2026-08-23T23:06:36-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.174 us 75146.847 us    42.44       0.2%    0.000s  ok ch=9.4e-14/1e-10 1s=8e-16 1.00x
   gen_pfa_small                  1.426 us 91237.137 us    34.95       0.2%    0.000s  ok ch=1.0e-13/1e-10 1s=9e-16 1.21x
   mkl_dfti                       4.548 us 291049.037 us    10.96       0.3%    0.001s  ok ch=1.2e-13/1e-10 1s=1e-15 3.87x
   mkl2026_dfti                   4.632 us 296457.928 us    10.76       0.6%    0.002s  ok ch=1.2e-13/1e-10 1s=1e-15 3.95x
   fftw3_patient                  5.184 us 331796.233 us     9.61       3.2%    0.022s  ok ch=9.8e-14/1e-10 1s=8e-16 4.42x
   fftw3_measure                  5.191 us 332211.386 us     9.60       1.2%    0.013s  ok ch=9.8e-14/1e-10 1s=8e-16 4.42x
   fftw3_estimate                 7.312 us 467955.116 us     6.81       1.3%    0.001s  ok ch=9.8e-14/1e-10 1s=8e-16 6.23x
   gen_dense_prime                8.291 us 530651.104 us     6.01       4.0%    0.000s  ok ch=9.8e-14/1e-10 1s=9e-16 7.06x
   gen_planner                    8.594 us 549986.386 us     5.80       1.5%    0.000s  ok ch=9.1e-14/1e-10 1s=9e-16 7.32x
   gen_layout                     9.721 us 622151.219 us     5.13       1.2%    0.000s  ok ch=9.7e-14/1e-10 1s=9e-16 8.28x
   ducc0_c2c                      9.813 us 628019.235 us     5.08       0.1%    0.000s  ok ch=8.3e-14/1e-10 1s=7e-16 8.36x
   gen_bluestein                 15.648 us 1001493.230 us     3.18       1.4%    0.000s  ok ch=1.8e-13/1e-10 1s=1e-15 13.33x
   gen_twiddle                   18.134 us 1160562.630 us     2.75       3.5%    0.002s  ok ch=1.0e-13/1e-10 1s=9e-16 15.44x
   gen_race                      48.578 us 3109006.540 us     1.03       0.7%    0.077s  ok ch=9.6e-14/1e-10 1s=9e-16 41.37x
   baseline_matrix               54.687 us 3499983.800 us     0.91       0.1%    0.000s  ok ch=3.4e-13/1e-10 1s=1e-15 46.58x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  1.995 us 76617.747 us    46.57       0.1%    0.000s  ok ch=5.5e-14/1e-10 1s=9e-16 1.00x
   gen_pfa_small                  2.500 us 96014.556 us    37.16       0.5%    0.000s  ok ch=6.0e-14/1e-10 1s=1e-15 1.25x
   mkl_dfti                       7.744 us 297371.032 us    12.00       0.3%    0.002s  ok ch=5.9e-14/1e-10 1s=9e-16 3.88x
   mkl2026_dfti                   7.761 us 298040.366 us    11.97       0.1%    0.002s  ok ch=5.2e-14/1e-10 1s=9e-16 3.89x
   fftw3_patient                  8.685 us 333512.748 us    10.70       4.1%    0.024s  ok ch=7.0e-14/1e-10 1s=9e-16 4.35x
   fftw3_measure                  8.851 us 339886.138 us    10.50       0.2%    0.015s  ok ch=6.5e-14/1e-10 1s=9e-16 4.44x
   gen_planner                   14.145 us 543176.630 us     6.57       0.6%    0.000s  ok ch=5.1e-14/1e-10 1s=1e-15 7.09x
   gen_dense_prime               14.468 us 555556.013 us     6.42       2.7%    0.000s  ok ch=6.5e-14/1e-10 1s=9e-16 7.25x
   gen_layout                    15.947 us 612382.836 us     5.83       0.9%    0.000s  ok ch=6.2e-14/1e-10 1s=1e-15 7.99x
   ducc0_c2c                     16.026 us 615414.209 us     5.80       1.8%    0.000s  ok ch=3.8e-14/1e-10 1s=7e-16 8.03x
   fftw3_estimate                19.293 us 740846.414 us     4.82       0.0%    0.001s  ok ch=7.4e-14/1e-10 1s=9e-16 9.67x
   gen_bluestein                 23.461 us 900897.590 us     3.96       2.2%    0.000s  ok ch=1.0e-13/1e-10 1s=1e-15 11.76x
   gen_twiddle                   27.067 us 1039363.350 us     3.43       1.6%    0.002s  ok ch=6.1e-14/1e-10 1s=9e-16 13.57x
   gen_race                      93.249 us 3580775.050 us     1.00       0.5%    0.001s  ok ch=8.7e-14/1e-10 1s=1e-15 46.74x
   baseline_matrix              112.140 us 4306163.990 us     0.83       0.1%    0.000s  ok ch=2.9e-13/1e-10 1s=2e-15 56.20x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_batchlane                  4.686 us 89972.498 us    42.21       0.8%    0.000s  ok ch=5.6e-14/1e-10 1s=1e-15 1.00x
   gen_pfa_small                  6.178 us 118619.374 us    32.01       2.3%    0.000s  ok ch=5.4e-14/1e-10 1s=1e-15 1.32x
   mkl_dfti                      16.445 us 315744.746 us    12.03       0.1%    0.002s  ok ch=6.0e-14/1e-10 1s=1e-15 3.51x
   mkl2026_dfti                  16.621 us 319121.826 us    11.90       0.2%    0.002s  ok ch=6.1e-14/1e-10 1s=1e-15 3.55x
   fftw3_patient                 19.633 us 376951.618 us    10.07       0.4%    0.020s  ok ch=6.6e-14/1e-10 1s=1e-15 4.19x
   fftw3_measure                 19.889 us 381871.873 us     9.94       1.2%    0.010s  ok ch=6.6e-14/1e-10 1s=1e-15 4.24x
   fftw3_estimate                20.606 us 395633.743 us     9.60       0.2%    0.001s  ok ch=6.6e-14/1e-10 1s=1e-15 4.40x
   gen_planner                   28.937 us 555587.116 us     6.84       0.2%    0.000s  ok ch=5.7e-14/1e-10 1s=1e-15 6.18x
   gen_dense_prime               31.055 us 596249.530 us     6.37       0.2%    0.000s  ok ch=6.1e-14/1e-10 1s=1e-15 6.63x
   ducc0_c2c                     32.512 us 624231.871 us     6.08       0.8%    0.000s  ok ch=4.2e-14/1e-10 1s=1e-15 6.94x
   gen_layout                    33.629 us 645672.883 us     5.88       0.8%    0.000s  ok ch=6.3e-14/1e-10 1s=1e-15 7.18x
   gen_bluestein                 43.527 us 835719.548 us     4.54       0.2%    0.000s  ok ch=1.1e-13/1e-10 1s=2e-15 9.29x
   gen_twiddle                   55.370 us 1063105.600 us     3.57       1.1%    0.002s  ok ch=6.9e-14/1e-10 1s=1e-15 11.82x
   gen_race                     223.060 us 4282749.260 us     0.89       0.1%    0.002s  ok ch=5.9e-14/1e-10 1s=1e-15 47.60x
   baseline_matrix              272.439 us 5230828.840 us     0.73       0.0%    0.000s  ok ch=2.8e-13/1e-10 1s=2e-15 58.14x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                 16.920 us 138607.680 us    30.65       4.3%    0.001s  ok ch=4.2e-14/1e-10 1s=1e-15 1.00x
   fftw3_measure                 44.893 us 367765.439 us    11.55       2.1%    0.081s  ok ch=4.9e-14/1e-10 1s=1e-15 2.65x
   fftw3_patient                 45.699 us 374363.945 us    11.35       0.4%    0.297s  ok ch=3.7e-14/1e-10 1s=1e-15 2.70x
   mkl2026_dfti                  57.518 us 471189.247 us     9.02       2.9%    0.044s  ok ch=7.0e-14/1e-10 1s=1e-15 3.40x
   mkl_dfti                      57.963 us 474830.925 us     8.95       0.5%    0.049s  ok ch=4.2e-14/1e-10 1s=1e-15 3.43x
   gen_planner                   61.658 us 505105.494 us     8.41       0.9%    0.000s  ok ch=5.5e-14/1e-10 1s=1e-15 3.64x
   ducc0_c2c                     73.438 us 601601.201 us     7.06       1.3%    0.000s  ok ch=5.0e-14/1e-10 1s=1e-15 4.34x
   gen_dense_prime               79.768 us 653457.594 us     6.50       0.9%    0.000s  ok ch=3.9e-14/1e-10 1s=1e-15 4.71x
   fftw3_estimate                91.839 us 752347.271 us     5.65       0.3%    0.001s  ok ch=4.9e-14/1e-10 1s=1e-15 5.43x
   gen_layout                    93.156 us 763132.735 us     5.57       0.7%    0.000s  ok ch=6.1e-14/1e-10 1s=1e-15 5.51x
   gen_twiddle                  121.935 us 998890.869 us     4.25       1.6%    0.002s  ok ch=3.7e-14/1e-10 1s=1e-15 7.21x
   gen_bluestein                127.139 us 1041525.920 us     4.08       2.9%    0.000s  ok ch=4.6e-14/1e-10 1s=2e-15 7.51x
   gen_race                     633.053 us 5185969.390 us     0.82       0.2%    0.456s  ok ch=3.5e-14/1e-10 1s=1e-15 37.41x
   baseline_matrix              850.286 us 6965541.440 us     0.61       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=2e-15 50.25x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      48.778 us 199796.047 us    22.31       2.8%    0.059s  ok ch=3.4e-14/1e-10 1s=2e-15 1.00x
   fftw3_patient                108.664 us 445088.714 us    10.02       3.7%    0.112s  ok ch=3.9e-14/1e-10 1s=2e-15 2.23x
   fftw3_measure                116.313 us 476419.031 us     9.36       0.8%    0.028s  ok ch=3.9e-14/1e-10 1s=2e-15 2.38x
   mkl_dfti                     120.849 us 494996.394 us     9.01       0.2%    0.049s  ok ch=3.9e-14/1e-10 1s=2e-15 2.48x
   mkl2026_dfti                 123.533 us 505991.716 us     8.81       0.1%    0.051s  ok ch=3.9e-14/1e-10 1s=2e-15 2.53x
   fftw3_estimate               130.568 us 534807.941 us     8.34       2.3%    0.001s  ok ch=3.9e-14/1e-10 1s=2e-15 2.68x
   gen_planner                  137.546 us 563386.550 us     7.91       0.7%    0.000s  ok ch=3.5e-14/1e-10 1s=2e-15 2.82x
   ducc0_c2c                    145.092 us 594296.015 us     7.50       2.8%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 2.97x
   gen_layout                   229.259 us 939045.345 us     4.75       0.6%    0.001s  ok ch=4.0e-14/1e-10 1s=2e-15 4.70x
   gen_bluestein                229.310 us 939255.794 us     4.75       3.6%    0.000s  ok ch=5.0e-14/1e-10 1s=2e-15 4.70x
   gen_twiddle                  260.651 us 1067626.210 us     4.18       0.6%    0.003s  ok ch=3.6e-14/1e-10 1s=2e-15 5.34x
   gen_race                    1569.553 us 6428888.740 us     0.69       0.5%    0.666s  ok ch=3.8e-14/1e-10 1s=2e-15 32.18x
   baseline_matrix             2061.155 us 8442490.850 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 42.26x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      66.078 us 211448.961 us    21.25       2.8%    0.078s  ok ch=3.2e-14/1e-10 1s=2e-15 1.00x
   mkl_dfti                     144.425 us 462159.920 us     9.72       0.1%    0.049s  ok ch=3.3e-14/1e-10 1s=2e-15 2.19x
   mkl2026_dfti                 148.049 us 473756.171 us     9.48       0.2%    0.052s  ok ch=3.3e-14/1e-10 1s=2e-15 2.24x
   ducc0_c2c                    187.483 us 599946.011 us     7.49       2.2%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 2.84x
   fftw3_patient                203.932 us 652582.743 us     6.88       8.0%    0.139s  ok ch=3.4e-14/1e-10 1s=2e-15 3.09x
   gen_planner                  216.159 us 691707.309 us     6.49       0.5%    0.000s  ok ch=3.6e-14/1e-10 1s=2e-15 3.27x
   fftw3_measure                225.301 us 720963.329 us     6.23       3.2%    0.034s  ok ch=3.4e-14/1e-10 1s=2e-15 3.41x
   fftw3_estimate               256.118 us 819577.653 us     5.48       0.9%    0.003s  ok ch=3.4e-14/1e-10 1s=2e-15 3.88x
   gen_bluestein                280.274 us 896876.693 us     5.01       0.1%    0.000s  ok ch=5.1e-14/1e-10 1s=2e-15 4.24x
   gen_layout                   291.970 us 934302.989 us     4.81       0.4%    0.001s  ok ch=3.4e-14/1e-10 1s=2e-15 4.42x
   gen_twiddle                  427.814 us 1369005.310 us     3.28       1.1%    0.002s  ok ch=3.3e-14/1e-10 1s=2e-15 6.47x
   gen_race                    2094.428 us 6702168.340 us     0.67       0.0%    0.902s  ok ch=4.4e-14/1e-10 1s=2e-15 31.70x
   baseline_matrix             2799.078 us 8957049.740 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 42.36x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     94.170 us 210941.671 us    23.51       2.6%    0.008s  ok ch=2.8e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              175.591 us 393324.846 us    12.61       3.1%    0.000s  ok ch=2.6e-14/1e-10 1s=2e-15 1.86x
   gen_bluestein                405.309 us 907892.373 us     5.46       0.5%    0.000s  ok ch=4.1e-14/1e-10 1s=3e-15 4.30x
   gen_layout                   466.299 us 1044510.130 us     4.75       0.4%    0.001s  ok ch=3.2e-14/1e-10 1s=2e-15 4.95x
   gen_planner                  606.815 us 1359266.580 us     3.65       0.3%    0.000s  ok ch=3.6e-14/1e-10 1s=2e-15 6.44x
   ducc0_c2c                    725.494 us 1625106.760 us     3.05       1.9%    0.000s  ok ch=2.1e-14/1e-10 1s=1e-15 7.70x
   mkl_dfti                     848.435 us 1900495.220 us     2.61       0.1%    0.055s  ok ch=3.3e-14/1e-10 1s=2e-15 9.01x
   fftw3_estimate               859.150 us 1924496.800 us     2.58       0.1%    0.002s  ok ch=2.8e-14/1e-10 1s=2e-15 9.12x
   fftw3_measure                859.483 us 1925242.160 us     2.58       0.1%    0.091s  ok ch=2.8e-14/1e-10 1s=2e-15 9.13x
   fftw3_patient                859.795 us 1925940.930 us     2.57       0.5%    0.241s  ok ch=2.8e-14/1e-10 1s=2e-15 9.13x
   mkl2026_dfti                 883.107 us 1978159.770 us     2.51       0.0%    0.050s  ok ch=3.3e-14/1e-10 1s=2e-15 9.38x
   gen_twiddle                 1120.143 us 2509119.240 us     1.98       0.1%    0.003s  ok ch=3.2e-14/1e-10 1s=2e-15 11.89x
   gen_race                    3545.389 us 7941670.960 us     0.62       0.1%    1.457s  ok ch=3.2e-14/1e-10 1s=2e-15 37.65x
   baseline_matrix             4852.055 us 10868603.800 us     0.46       0.0%    0.000s  ok ch=8.6e-14/1e-10 1s=3e-15 51.52x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      63.683 us 127365.908 us    38.59       0.4%    0.000s  ok ch=3.2e-14/1e-10 1s=1e-15 1.00x
   mkl_dfti                     171.437 us 342873.363 us    14.34       0.6%    0.002s  ok ch=3.0e-14/1e-10 1s=1e-15 2.69x
   mkl2026_dfti                 186.259 us 372518.192 us    13.19       3.3%    0.002s  ok ch=3.0e-14/1e-10 1s=1e-15 2.92x
   fftw3_measure                208.078 us 416156.337 us    11.81       1.6%    0.087s  ok ch=3.0e-14/1e-10 1s=1e-15 3.27x
   fftw3_patient                209.307 us 418614.095 us    11.74       1.3%    0.541s  ok ch=2.9e-14/1e-10 1s=1e-15 3.29x
   gen_planner                  306.017 us 612034.479 us     8.03       0.7%    0.000s  ok ch=3.3e-14/1e-10 1s=2e-15 4.81x
   ducc0_c2c                    308.857 us 617714.963 us     7.96       2.1%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 4.85x
   fftw3_estimate               407.493 us 814986.096 us     6.03       0.6%    0.001s  ok ch=2.9e-14/1e-10 1s=1e-15 6.40x
   gen_bluestein                411.408 us 822816.390 us     5.97       0.9%    0.000s  ok ch=4.5e-14/1e-10 1s=2e-15 6.46x
   gen_layout                   493.852 us 987704.026 us     4.98       1.2%    0.001s  ok ch=4.0e-14/1e-10 1s=2e-15 7.75x
   gen_twiddle                  641.583 us 1283166.470 us     3.83       1.7%    0.002s  ok ch=3.0e-14/1e-10 1s=1e-15 10.07x
   gen_race                    3888.056 us 7776112.620 us     0.63       1.4%    0.002s  ok ch=3.8e-14/1e-10 1s=2e-15 61.05x
   baseline_matrix             5753.337 us 11506673.400 us     0.43       0.0%    0.000s  ok ch=1.2e-13/1e-10 1s=2e-15 90.34x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                200.515 us 205327.402 us    25.48       0.5%    0.245s  ok ch=2.6e-14/1e-10 1s=2e-15 1.00x
   mkl2026_dfti                 405.425 us 415155.694 us    12.60       0.1%    0.002s  ok ch=4.2e-14/1e-10 1s=2e-15 2.02x
   mkl_dfti                     406.343 us 416095.091 us    12.57       0.3%    0.003s  ok ch=2.8e-14/1e-10 1s=2e-15 2.03x
   fftw3_patient                527.853 us 540521.249 us     9.68       1.6%    1.482s  ok ch=2.5e-14/1e-10 1s=2e-15 2.63x
   fftw3_measure                552.732 us 565998.032 us     9.24       4.3%    0.154s  ok ch=2.2e-14/1e-10 1s=2e-15 2.76x
   ducc0_c2c                    596.769 us 611091.674 us     8.56       1.8%    0.000s  ok ch=2.1e-14/1e-10 1s=2e-15 2.98x
   gen_planner                  666.323 us 682314.501 us     7.67       1.5%    0.000s  ok ch=2.4e-14/1e-10 1s=2e-15 3.32x
   gen_layout                  1114.321 us 1141064.430 us     4.58       0.2%    0.001s  ok ch=3.0e-14/1e-10 1s=2e-15 5.56x
   gen_twiddle                 1287.204 us 1318096.720 us     3.97       0.4%    0.003s  ok ch=2.3e-14/1e-10 1s=2e-15 6.42x
   gen_bluestein               1300.442 us 1331653.100 us     3.93       0.9%    0.000s  ok ch=3.7e-14/1e-10 1s=3e-15 6.49x
   fftw3_estimate              1608.435 us 1647036.960 us     3.18       0.2%    0.002s  ok ch=2.2e-14/1e-10 1s=2e-15 8.02x
   gen_race                    9325.926 us 9549748.230 us     0.55       0.5%    1.954s  ok ch=3.5e-14/1e-10 1s=2e-15 46.51x
   baseline_matrix            13439.233 us 13761774.300 us     0.38       0.4%    0.000s  ok ch=5.9e-14/1e-10 1s=3e-15 67.02x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                     473.484 us 242424.003 us    22.35       1.8%    0.689s  ok ch=4.8e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                481.372 us 246462.702 us    21.98       1.3%    0.627s  ok ch=4.8e-14/1e-10 1s=2e-15 1.02x
   mkl_dfti                     948.870 us 485821.363 us    11.15       0.2%    0.029s  ok ch=6.1e-14/1e-10 1s=2e-15 2.00x
   mkl2026_dfti                 961.769 us 492425.475 us    11.00       0.1%    0.051s  ok ch=3.2e-14/1e-10 1s=2e-15 2.03x
   fftw3_patient               1158.041 us 592917.153 us     9.14       2.3%    1.222s  ok ch=3.6e-14/1e-10 1s=2e-15 2.45x
   fftw3_measure               1198.350 us 613555.146 us     8.83       4.1%    0.088s  ok ch=3.8e-14/1e-10 1s=2e-15 2.53x
   ducc0_c2c                   1268.892 us 649672.726 us     8.34       1.1%    0.000s  ok ch=3.2e-14/1e-10 1s=2e-15 2.68x
   gen_planner                 1535.440 us 786145.026 us     6.89       2.3%    0.000s  ok ch=3.8e-14/1e-10 1s=2e-15 3.24x
   fftw3_estimate              1613.937 us 826335.737 us     6.56       0.6%    0.003s  ok ch=4.5e-14/1e-10 1s=2e-15 3.41x
   gen_bluestein               2266.638 us 1160518.740 us     4.67       0.3%    0.000s  ok ch=7.9e-14/1e-10 1s=3e-15 4.79x
   gen_twiddle                 2783.520 us 1425162.180 us     3.80       1.4%    0.003s  ok ch=7.1e-14/1e-10 1s=2e-15 5.88x
   gen_layout                  2918.735 us 1494392.530 us     3.63       1.0%    0.002s  ok ch=7.5e-14/1e-10 1s=3e-15 6.16x
   gen_race                   23750.849 us 12160434.900 us     0.45       0.7%    0.002s  ok ch=7.1e-14/1e-10 1s=3e-15 50.16x
   baseline_matrix            33903.762 us 17358725.900 us     0.31       1.6%    0.000s  ok ch=8.3e-14/1e-10 1s=4e-15 71.60x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                    5026.755 us 321712.343 us    19.83       2.0%    4.163s  ok ch=2.0e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large               5086.365 us 325527.349 us    19.59       1.0%    3.347s  ok ch=2.0e-14/1e-10 1s=2e-15 1.01x
   mkl_dfti                    7817.717 us 500333.875 us    12.75       0.1%    0.048s  ok ch=3.7e-14/1e-10 1s=3e-15 1.56x
   mkl2026_dfti                7839.003 us 501696.182 us    12.71       0.5%    0.048s  ok ch=4.0e-14/1e-10 1s=3e-15 1.56x
   fftw3_patient               9863.450 us 631260.769 us    10.10       3.4%   17.575s  ok ch=2.1e-14/1e-10 1s=3e-15 1.96x
   fftw3_measure              11087.118 us 709575.542 us     8.99       7.9%    0.277s  ok ch=2.0e-14/1e-10 1s=3e-15 2.21x
   ducc0_c2c                  11879.252 us 760272.144 us     8.39       0.6%    0.000s  ok ch=2.1e-14/1e-10 1s=2e-15 2.36x
   gen_planner                15044.641 us 962857.005 us     6.62       0.6%    0.000s  ok ch=1.9e-14/1e-10 1s=3e-15 2.99x
   fftw3_estimate             20512.491 us 1312799.400 us     4.86       0.9%    0.003s  ok ch=2.0e-14/1e-10 1s=3e-15 4.08x
   gen_bluestein              20705.685 us 1325163.860 us     4.81       0.2%    0.000s  ok ch=2.6e-14/1e-10 1s=3e-15 4.12x
   gen_twiddle                26639.857 us 1704950.840 us     3.74       0.1%    0.006s  ok ch=2.2e-14/1e-10 1s=3e-15 5.30x
   gen_layout                 42345.885 us 2710136.630 us     2.35       0.2%    0.007s  ok ch=3.3e-14/1e-10 1s=4e-15 8.42x
   gen_race                  391379.555 us 25048291.500 us     0.25       0.6%    0.004s  ok ch=2.9e-14/1e-10 1s=4e-15 77.86x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA pencils (10=2x5,12=3x4,15=3x5), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14-ladder map, plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: pow2 radix-4 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, exact long-double chirp tables
   gen_dense_prime          folded dense prime p<=31: 4h^2-FMA conjugate-pair fold, transpose-free z-row/x/y passes (AVX-512 row-GEMM at L=31), volume-resident fused chain (rsqrt14+2NR, one vdivpd per 8 pts)
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L dense matrixsimd demo of the layer
   gen_pfa_large            GT-PFA 25x2 two-sweep (DFT25=5x5 CT exact tw) + owned chain (NR map); pick: l50-ip1 (B=4)
   gen_pfa_small            PFA coprime (10=2x5,12=4x3,15=3x5,20=4x5), no twiddles; split-complex SoA 8 vols/zmm, 3 in-place passes, chain owns state w/ map fused in z-pass; B%8 via per-volume split path
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor, any 2<=L<=128; adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: L32 custody split-complex chain (4x8 z-pair TR8, lazy exact map), 16/64/128 generic radix-2
   gen_powp                 powp CT GT 25x4(5x5) exact tw, two-sweep + owned chain (NR map); pick: l100-ip1 (B=1)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race + per-host wisdom cache, results/wisdom_<host>.json (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo races 3 dense any-L variants at create()
   gen_rader                Rader-31: conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist to cyclic), each via Winograd-C3 x dense-C5 (100 FMA + 65 add), all-real constants on interleaved zmm; dense z-row pass + volume-resident fused chain (s6 map, one vdivpd/8pts) adopted from gen_dense_prime
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp), consumption-order CT/DFT/chirp fillers + ulp audits + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix demo, self-audited at create()
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
