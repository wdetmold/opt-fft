```
=== round d1_libbase2 ===
# round d1_libbase2
host: a81n2.lqcd.mit   date: 2026-09-02T13:56:09-04:00   slurm_job: 440335
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_estimate               0.0220 us     0.022 us    10.93      34.3%    0.001s  ok 1.5e-16       1.00x
   fftw1d_measure                0.0220 us     0.022 us    10.91      19.7%    0.001s  ok 1.5e-16       1.00x
   mkl1d_dfti                    0.0259 us     0.026 us     9.29       7.9%    0.037s  ok 1.7e-16       1.18x
   fftw1d_patient                0.0264 us     0.026 us     9.11       1.1%    0.001s  ok 1.5e-16       1.20x
   fftw1d_custom                 0.0567 us     0.057 us     4.24       0.4%    0.000s  ok 1.1e-16       2.58x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom                 0.0626 us 12516.396 us     3.84      20.5%    0.000s  ok ch=1.4e-15/1e-10 1s=3e-16 1.00x
   fftw1d_measure                0.0674 us 13480.514 us     3.57      20.8%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 1.08x
   fftw1d_patient                0.0674 us 13484.953 us     3.57      20.7%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 1.08x
   mkl1d_dfti                    0.0803 us 16059.012 us     3.00       0.1%    0.048s  ok ch=1.6e-15/1e-10 1s=2e-16 1.28x
   fftw1d_estimate               0.0814 us 16273.953 us     2.96       0.0%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 1.30x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient                0.0139 us     7.140 us    17.25       2.6%    0.003s  ok 1.8e-16       1.00x
   fftw1d_measure                0.0141 us     7.199 us    17.11       3.3%    0.002s  ok 1.8e-16       1.01x
   mkl1d_dfti                    0.0193 us     9.861 us    12.49       1.5%    0.034s  ok 1.7e-16       1.38x
   fftw1d_custom_soa             0.0366 us    18.744 us     6.57       1.7%    0.000s  ok 1.8e-16       2.63x
   fftw1d_estimate               0.0400 us    20.487 us     6.01       0.4%    0.001s  ok 1.8e-16       2.87x
   fftw1d_custom                 0.0487 us    24.955 us     4.93      14.5%    0.000s  ok 1.8e-16       3.50x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.0262 us 26822.345 us     9.18       1.4%    0.000s  ok ch=1.5e-14/1e-10 1s=3e-16 1.00x
   fftw1d_custom                 0.0473 us 48445.492 us     5.08       1.0%    0.000s  ok ch=1.5e-14/1e-10 1s=3e-16 1.81x
   fftw1d_patient                0.0512 us 52465.548 us     4.69      15.0%    0.002s  ok ch=8.3e-15/1e-10 1s=3e-16 1.96x
   fftw1d_measure                0.0515 us 52691.847 us     4.67      13.3%    0.002s  ok ch=8.3e-15/1e-10 1s=3e-16 1.96x
   mkl1d_dfti                    0.0530 us 54259.082 us     4.54       2.4%    0.044s  ok ch=7.2e-14/1e-10 1s=3e-16 2.02x
   fftw1d_estimate               0.0746 us 76356.018 us     3.23       0.1%    0.001s  ok ch=8.3e-15/1e-10 1s=3e-16 2.85x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom                 0.2173 us     0.217 us     3.53       0.6%    0.000s  ok 2.4e-16       1.00x
   fftw1d_estimate               0.2599 us     0.260 us     2.95      20.8%    0.001s  ok 1.7e-16       1.20x
   fftw1d_patient                0.2615 us     0.262 us     2.94      28.1%    0.120s  ok 1.7e-16       1.20x
   mkl1d_dfti                    0.2794 us     0.279 us     2.75       0.0%    0.032s  ok 2.2e-16       1.29x
   fftw1d_measure                0.3177 us     0.318 us     2.42       3.1%    0.043s  ok 1.7e-16       1.46x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom                 0.2519 us 25193.089 us     3.05       0.4%    0.000s  ok ch=1.4e-15/1e-10 1s=6e-16 1.00x
   fftw1d_measure                0.3479 us 34787.517 us     2.21      21.3%    0.043s  ok ch=5.2e-15/1e-10 1s=4e-16 1.38x
   fftw1d_patient                0.3485 us 34845.908 us     2.20       0.7%    0.118s  ok ch=5.2e-15/1e-10 1s=4e-16 1.38x
   fftw1d_estimate               0.3658 us 36577.540 us     2.10      16.2%    0.001s  ok ch=5.2e-15/1e-10 1s=4e-16 1.45x
   mkl1d_dfti                    0.3855 us 38550.880 us     1.99       0.1%    0.034s  ok ch=7.9e-15/1e-10 1s=4e-16 1.53x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.0899 us    46.040 us     8.54      24.1%    0.000s  ok 2.6e-16       1.00x
   fftw1d_custom                 0.1620 us    82.925 us     4.74      20.1%    0.000s  ok 2.6e-16       1.80x
   fftw1d_patient                0.2633 us   134.833 us     2.92       0.0%    0.119s  ok 2.1e-16       2.93x
   fftw1d_measure                0.2634 us   134.883 us     2.91      20.7%    0.042s  ok 2.1e-16       2.93x
   mkl1d_dfti                    0.2736 us   140.072 us     2.81       0.1%    0.038s  ok 2.3e-16       3.04x
   fftw1d_estimate               0.3180 us   162.794 us     2.42      21.0%    0.001s  ok 2.1e-16       3.54x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.0771 us 47387.083 us     9.96       1.4%    0.000s  ok ch=8.1e-13/1e-10 1s=5e-16 1.00x
   fftw1d_custom                 0.1770 us 108773.222 us     4.34       0.2%    0.000s  ok ch=8.1e-13/1e-10 1s=5e-16 2.30x
   mkl1d_dfti                    0.3152 us 193673.078 us     2.44       0.1%    0.052s  ok ch=7.1e-13/1e-10 1s=5e-16 4.09x
   fftw1d_estimate               0.3534 us 217111.937 us     2.17       0.1%    0.001s  ok ch=1.1e-12/1e-10 1s=4e-16 4.58x
   fftw1d_measure                0.3534 us 217135.351 us     2.17       0.1%    0.042s  ok ch=1.1e-12/1e-10 1s=4e-16 4.58x
   fftw1d_patient                0.3536 us 217248.502 us     2.17       0.0%    0.119s  ok ch=1.1e-12/1e-10 1s=4e-16 4.58x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0254 us     0.025 us    31.44       2.0%    0.002s  ok 1.5e-16       1.00x
   fftw1d_measure                0.0370 us     0.037 us    21.64       4.5%    0.008s  ok 1.6e-16       1.45x
   fftw1d_patient                0.0376 us     0.038 us    21.29      18.6%    0.015s  ok 1.6e-16       1.48x
   fftw1d_estimate               0.0472 us     0.047 us    16.94       0.4%    0.001s  ok 1.6e-16       1.86x
   fftw1d_custom                 0.1083 us     0.108 us     7.39       1.9%    0.000s  ok 1.4e-16       4.25x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.1309 us 13092.527 us     6.11       0.1%    0.002s  ok ch=3.8e-15/1e-10 1s=3e-16 1.00x
   fftw1d_measure                0.1360 us 13602.497 us     5.88      13.8%    0.008s  ok ch=4.5e-15/1e-10 1s=3e-16 1.04x
   fftw1d_patient                0.1363 us 13630.443 us     5.87       0.0%    0.014s  ok ch=6.4e-15/1e-10 1s=3e-16 1.04x
   fftw1d_estimate               0.1452 us 14521.556 us     5.51      13.8%    0.001s  ok ch=4.4e-15/1e-10 1s=3e-16 1.11x
   fftw1d_custom                 0.1465 us 14647.492 us     5.46       2.0%    0.000s  ok ch=5.5e-15/1e-10 1s=2e-16 1.12x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0153 us     7.830 us    52.31      13.8%    0.002s  ok 1.3e-16       1.00x
   fftw1d_patient                0.0253 us    12.978 us    31.56      13.9%    0.061s  ok 1.6e-16       1.66x
   fftw1d_measure                0.0287 us    14.682 us    27.90       1.3%    0.010s  ok 1.5e-16       1.88x
   fftw1d_custom                 0.0875 us    44.794 us     9.14      27.8%    0.000s  ok 1.4e-16       5.72x
   fftw1d_custom_soa             0.0916 us    46.908 us     8.73       8.2%    0.000s  ok 1.4e-16       5.99x
   fftw1d_estimate               0.1004 us    51.396 us     7.97       0.2%    0.001s  ok 1.5e-16       6.56x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.0665 us 34043.698 us    12.03       0.4%    0.000s  ok ch=8.0e-13/1e-10 1s=3e-16 1.00x
   fftw1d_custom                 0.1040 us 53227.399 us     7.70      20.7%    0.000s  ok ch=8.0e-13/1e-10 1s=3e-16 1.56x
   mkl1d_dfti                    0.1123 us 57472.724 us     7.13       0.0%    0.001s  ok ch=6.2e-13/1e-10 1s=3e-16 1.69x
   fftw1d_measure                0.1221 us 62500.892 us     6.55       1.3%    0.010s  ok ch=1.3e-12/1e-10 1s=3e-16 1.84x
   fftw1d_patient                0.1222 us 62547.274 us     6.55       0.3%    0.057s  ok ch=1.3e-12/1e-10 1s=3e-16 1.84x
   fftw1d_estimate               0.1860 us 95228.594 us     4.30       0.3%    0.001s  ok ch=9.9e-13/1e-10 1s=3e-16 2.80x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0533 us     0.053 us    33.25      14.0%    0.002s  ok 2.4e-16       1.00x
   fftw1d_measure                0.0682 us     0.068 us    25.99      23.0%    0.046s  ok 1.9e-16       1.28x
   fftw1d_patient                0.0701 us     0.070 us    25.26      12.5%    0.128s  ok 1.9e-16       1.32x
   fftw1d_estimate               0.0830 us     0.083 us    21.34      14.8%    0.001s  ok 2.5e-16       1.56x
   fftw1d_custom                 0.2289 us     0.229 us     7.74      21.7%    0.000s  ok 2.4e-16       4.30x

-- L=60 (non-batched, chain m=60000), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_measure                0.2314 us 13881.752 us     7.66       9.5%    0.046s  ok ch=2.4e-15/1e-10 1s=5e-16 1.00x
   mkl1d_dfti                    0.2370 us 14222.240 us     7.48      13.8%    0.002s  ok ch=1.1e-15/1e-10 1s=7e-16 1.02x
   fftw1d_patient                0.2415 us 14488.064 us     7.34      14.4%    0.128s  ok ch=1.6e-15/1e-10 1s=5e-16 1.04x
   fftw1d_custom                 0.2716 us 16293.054 us     6.53       0.3%    0.000s  ok ch=1.5e-15/1e-10 1s=6e-16 1.17x
   fftw1d_estimate               0.2859 us 17154.401 us     6.20       0.3%    0.001s  ok ch=1.1e-15/1e-10 1s=6e-16 1.24x

-- L=60 (batched B=512, single call), working set 0.938 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0432 us    22.110 us    41.04      16.1%    0.002s  ok 2.5e-16       1.00x
   fftw1d_patient                0.0660 us    33.768 us    26.87       0.7%    0.435s  ok 2.0e-16       1.53x
   fftw1d_measure                0.0695 us    35.586 us    25.50      45.2%    0.046s  ok 2.3e-16       1.61x
   fftw1d_estimate               0.0861 us    44.079 us    20.58       0.6%    0.001s  ok 2.3e-16       1.99x
   fftw1d_custom_soa             0.1877 us    96.085 us     9.44      12.7%    0.000s  ok 2.2e-16       4.35x
   fftw1d_custom                 0.2344 us   120.008 us     7.56      23.4%    0.000s  ok 2.2e-16       5.43x

-- L=60 (batched B=512, chain m=600), working set 0.938 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.1347 us 41392.416 us    13.15      28.9%    0.000s  ok ch=2.5e-13/1e-10 1s=5e-16 1.00x
   mkl1d_dfti                    0.2273 us 69817.068 us     7.80       0.6%    0.001s  ok ch=1.7e-13/1e-10 1s=5e-16 1.69x
   fftw1d_custom                 0.2311 us 71005.142 us     7.67       0.2%    0.000s  ok ch=2.5e-13/1e-10 1s=5e-16 1.72x
   fftw1d_patient                0.2499 us 76758.931 us     7.09       0.9%    0.434s  ok ch=1.5e-13/1e-10 1s=5e-16 1.85x
   fftw1d_estimate               0.2697 us 82865.785 us     6.57       0.1%    0.001s  ok ch=2.2e-13/1e-10 1s=5e-16 2.00x
   fftw1d_measure                0.2729 us 83828.623 us     6.49       0.9%    0.047s  ok ch=2.6e-13/1e-10 1s=5e-16 2.03x

-- L=64 (non-batched, single call), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0453 us     0.045 us    42.43       8.7%    0.002s  ok 2.0e-16       1.00x
   fftw1d_patient                0.0568 us     0.057 us    33.80      14.0%    0.053s  ok 1.7e-16       1.26x
   fftw1d_measure                0.0590 us     0.059 us    32.54      19.3%    0.022s  ok 1.8e-16       1.30x
   fftw1d_estimate               0.0786 us     0.079 us    24.42       6.8%    0.001s  ok 1.9e-16       1.74x
   fftw1d_custom                 0.2255 us     0.226 us     8.51      22.0%    0.000s  ok 1.8e-16       4.98x

-- L=64 (non-batched, chain m=60000), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.2369 us 14211.978 us     8.11      14.0%    0.002s  ok ch=1.8e-15/1e-10 1s=4e-16 1.00x
   fftw1d_measure                0.2436 us 14613.434 us     7.88       1.7%    0.021s  ok ch=2.6e-15/1e-10 1s=4e-16 1.03x
   fftw1d_patient                0.2448 us 14689.567 us     7.84      14.3%    0.054s  ok ch=2.6e-15/1e-10 1s=4e-16 1.03x
   fftw1d_estimate               0.2690 us 16142.396 us     7.14      13.4%    0.001s  ok ch=2.4e-15/1e-10 1s=5e-16 1.14x
   fftw1d_custom                 0.2987 us 17924.444 us     6.43      20.5%    0.000s  ok ch=1.9e-15/1e-10 1s=4e-16 1.26x

-- L=64 (batched B=512, single call), working set 1.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0380 us    19.438 us    50.57      18.1%    0.002s  ok 1.7e-16       1.00x
   fftw1d_patient                0.0666 us    34.095 us    28.83       4.2%    0.212s  ok 1.9e-16       1.75x
   fftw1d_measure                0.0679 us    34.767 us    28.28      19.2%    0.026s  ok 1.9e-16       1.79x
   fftw1d_estimate               0.1943 us    99.500 us     9.88      14.3%    0.001s  ok 1.9e-16       5.12x
   fftw1d_custom_soa             0.2119 us   108.498 us     9.06       1.3%    0.000s  ok 1.8e-16       5.58x
   fftw1d_custom                 0.2614 us   133.813 us     7.35      21.7%    0.000s  ok 1.8e-16       6.88x

-- L=64 (batched B=512, chain m=500), working set 1.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.1488 us 38096.258 us    12.90       3.4%    0.000s  ok ch=2.7e-13/1e-10 1s=4e-16 1.00x
   mkl1d_dfti                    0.2374 us 60768.734 us     8.09       0.9%    0.002s  ok ch=2.9e-13/1e-10 1s=4e-16 1.60x
   fftw1d_custom                 0.2504 us 64092.924 us     7.67       0.1%    0.000s  ok ch=2.7e-13/1e-10 1s=4e-16 1.68x
   fftw1d_patient                0.2678 us 68550.460 us     7.17       1.6%    0.219s  ok ch=3.2e-13/1e-10 1s=4e-16 1.80x
   fftw1d_measure                0.2685 us 68744.951 us     7.15       0.4%    0.029s  ok ch=2.5e-13/1e-10 1s=4e-16 1.80x
   fftw1d_estimate               0.3947 us 101038.536 us     4.86       0.6%    0.000s  ok ch=3.2e-13/1e-10 1s=4e-16 2.65x

-- L=128 (non-batched, single call), working set 0.004 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.1048 us     0.105 us    42.76       1.8%    0.002s  ok 2.1e-16       1.00x
   fftw1d_patient                0.1177 us     0.118 us    38.05       3.0%    0.139s  ok 2.3e-16       1.12x
   fftw1d_measure                0.1243 us     0.124 us    36.06       6.0%    0.040s  ok 2.3e-16       1.19x
   fftw1d_estimate               0.1703 us     0.170 us    26.31       1.2%    0.001s  ok 2.2e-16       1.63x
   fftw1d_custom                 0.7721 us     0.772 us     5.80       0.0%    0.000s  ok 2.2e-16       7.37x

-- L=128 (non-batched, chain m=30000), working set 0.004 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.4756 us 14267.598 us     9.42       0.1%    0.002s  ok ch=7.3e-15/1e-10 1s=5e-16 1.00x
   fftw1d_patient                0.4850 us 14551.179 us     9.24      14.9%    0.135s  ok ch=8.8e-15/1e-10 1s=5e-16 1.02x
   fftw1d_measure                0.4866 us 14598.223 us     9.21       1.5%    0.039s  ok ch=9.0e-15/1e-10 1s=5e-16 1.02x
   fftw1d_estimate               0.5296 us 15888.532 us     8.46       0.1%    0.001s  ok ch=1.0e-14/1e-10 1s=6e-16 1.11x
   fftw1d_custom                 0.7450 us 22349.191 us     6.01      20.8%    0.000s  ok ch=1.0e-14/1e-10 1s=5e-16 1.57x

-- L=128 (batched B=512, single call), working set 2.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.1462 us    74.832 us    30.65      14.8%    0.002s  ok 2.1e-16       1.00x
   fftw1d_patient                0.1869 us    95.715 us    23.96      41.3%    0.706s  ok 2.2e-16       1.28x
   fftw1d_measure                0.2052 us   105.070 us    21.83       1.7%    0.050s  ok 2.2e-16       1.40x
   fftw1d_custom_soa             0.5021 us   257.099 us     8.92      16.7%    0.000s  ok 2.1e-16       3.44x
   fftw1d_estimate               0.6525 us   334.076 us     6.87       5.7%    0.001s  ok 2.1e-16       4.46x
   fftw1d_custom                 0.7356 us   376.613 us     6.09      19.5%    0.000s  ok 2.1e-16       5.03x

-- L=128 (batched B=512, chain m=250), working set 2.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_custom_soa             0.3924 us 50228.713 us    11.42       0.7%    0.000s  ok ch=3.0e-14/1e-10 1s=5e-16 1.00x
   mkl1d_dfti                    0.5385 us 68932.311 us     8.32      12.1%    0.002s  ok ch=2.5e-14/1e-10 1s=5e-16 1.37x
   fftw1d_measure                0.5750 us 73604.151 us     7.79       1.2%    0.050s  ok ch=4.0e-14/1e-10 1s=5e-16 1.47x
   fftw1d_patient                0.5787 us 74068.386 us     7.74       0.3%    0.806s  ok ch=4.6e-14/1e-10 1s=6e-16 1.47x
   fftw1d_custom                 0.6671 us 85391.306 us     6.72       0.4%    0.000s  ok ch=3.0e-14/1e-10 1s=5e-16 1.70x
   fftw1d_estimate               0.8487 us 108633.359 us     5.28       0.3%    0.001s  ok ch=3.3e-14/1e-10 1s=5e-16 2.16x

-- L=1021 (non-batched, single call), working set 0.031 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    9.4092 us     9.409 us     5.42       0.3%    0.036s  ok 5.6e-16       1.00x
   fftw1d_measure               11.8605 us    11.861 us     4.30       4.7%    0.285s  ok 5.7e-16       1.26x
   fftw1d_patient               11.8723 us    11.872 us     4.30       1.2%    2.954s  ok 5.7e-16       1.26x
   fftw1d_estimate              12.7705 us    12.771 us     4.00       0.2%    0.001s  ok 5.7e-16       1.36x

-- L=1021 (non-batched, chain m=2000), working set 0.031 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                   12.9544 us 25908.847 us     3.94       0.1%    0.049s  ok ch=9.4e-13/1e-09 1s=2e-15 1.00x
   fftw1d_patient               14.9698 us 29939.675 us     3.41       0.7%    2.988s  ok ch=1.6e-12/1e-09 1s=2e-15 1.16x
   fftw1d_measure               15.4171 us 30834.171 us     3.31       9.7%    0.286s  ok ch=1.6e-12/1e-09 1s=2e-15 1.19x
   fftw1d_estimate              18.0468 us 36093.634 us     2.83       1.9%    0.001s  ok ch=2.5e-12/1e-09 1s=2e-15 1.39x

-- L=1021 (batched B=256, single call), working set 7.977 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                   10.0832 us  2581.287 us     5.06       5.5%    0.038s  ok 5.7e-16       1.00x
   fftw1d_patient               12.0181 us  3076.640 us     4.25       2.5%    3.005s  ok 5.7e-16       1.19x
   fftw1d_measure               12.1186 us  3102.373 us     4.21      21.5%    0.287s  ok 5.7e-16       1.20x
   fftw1d_estimate              13.0269 us  3334.893 us     3.92      14.1%    0.001s  ok 5.8e-16       1.29x

-- L=1021 (batched B=256, chain m=400), working set 7.977 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                   12.3817 us 1267888.600 us     4.12       0.2%    0.051s  ok ch=9.4e-14/1e-10 1s=2e-15 1.00x
   fftw1d_patient               15.6180 us 1599284.560 us     3.27       1.4%    2.989s  ok ch=9.4e-14/1e-10 1s=2e-15 1.26x
   fftw1d_measure               15.6409 us 1601631.540 us     3.26       2.0%    0.287s  ok ch=9.4e-14/1e-10 1s=2e-15 1.26x
   fftw1d_estimate              16.4602 us 1685525.890 us     3.10       0.1%    0.001s  ok ch=9.2e-14/1e-10 1s=2e-15 1.33x

-- L=1024 (non-batched, single call), working set 0.031 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    1.0860 us     1.086 us    47.15       1.5%    0.002s  ok 2.7e-16       1.00x
   fftw1d_measure                1.1536 us     1.154 us    44.38       8.1%    0.115s  ok 2.7e-16       1.06x
   fftw1d_patient                1.1584 us     1.158 us    44.20       0.4%    0.794s  ok 2.6e-16       1.07x
   fftw1d_estimate               1.3380 us     1.338 us    38.27       0.1%    0.000s  ok 2.8e-16       1.23x

-- L=1024 (non-batched, chain m=4000), working set 0.031 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    4.1846 us 16738.349 us    12.24      13.5%    0.002s  ok ch=8.1e-12/1e-10 1s=1e-15 1.00x
   fftw1d_patient                4.2918 us 17167.247 us    11.93      13.1%    0.812s  ok ch=1.4e-11/1e-10 1s=9e-16 1.03x
   fftw1d_measure                4.8233 us 19293.139 us    10.62       0.5%    0.117s  ok ch=1.4e-11/1e-10 1s=9e-16 1.15x
   fftw1d_estimate               5.0432 us 20172.966 us    10.15      15.3%    0.001s  ok ch=6.4e-12/1e-10 1s=1e-15 1.21x

-- L=1024 (batched B=512, single call), working set 16.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    1.6765 us   858.370 us    30.54      22.9%    0.003s  ok 2.8e-16       1.00x
   fftw1d_measure                1.8277 us   935.787 us    28.01      55.7%    0.117s  ok 2.7e-16       1.09x
   fftw1d_patient                1.8319 us   937.917 us    27.95       7.9%   11.553s  ok 2.7e-16       1.09x
   fftw1d_estimate               2.7824 us  1424.613 us    18.40      49.8%    0.001s  ok 2.8e-16       1.66x

-- L=1024 (batched B=512, chain m=2000), working set 16.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    4.9080 us 5025806.250 us    10.43       0.1%    0.003s  ok ch=7.1e-13/1e-10 1s=9e-16 1.00x
   fftw1d_measure                5.0521 us 5173395.000 us    10.13       1.9%    0.117s  ok ch=6.8e-13/1e-10 1s=9e-16 1.03x
   fftw1d_patient                5.0706 us 5192311.180 us    10.10       2.6%   11.364s  ok ch=6.8e-13/1e-10 1s=9e-16 1.03x
   fftw1d_estimate               5.6898 us 5826324.630 us     9.00       0.8%    0.001s  ok ch=7.4e-13/1e-10 1s=9e-16 1.16x

-- L=4096 (non-batched, single call), working set 0.125 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    6.0103 us     6.010 us    40.89      15.3%    0.002s  ok 3.1e-16       1.00x
   fftw1d_patient                6.7162 us     6.716 us    36.59       0.7%    1.937s  ok 3.1e-16       1.12x
   fftw1d_measure                8.4280 us     8.428 us    29.16       2.0%    0.234s  ok 3.1e-16       1.40x
   fftw1d_estimate              15.7503 us    15.750 us    15.60      14.7%    0.001s  ok 3.2e-16       2.62x

-- L=4096 (non-batched, chain m=1000), working set 0.125 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               19.1146 us 19114.559 us    12.86       0.3%    1.930s  ok ch=1.1e-13/1e-10 1s=1e-15 1.00x
   fftw1d_measure               19.8053 us 19805.327 us    12.41      13.4%    0.220s  ok ch=9.4e-14/1e-10 1s=1e-15 1.04x
   mkl1d_dfti                   20.8905 us 20890.463 us    11.76       0.4%    0.002s  ok ch=1.0e-13/1e-10 1s=1e-15 1.09x
   fftw1d_estimate              31.7342 us 31734.234 us     7.74       2.8%    0.001s  ok ch=8.9e-14/1e-10 1s=1e-15 1.66x

-- L=4096 (batched B=256, single call), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               10.9046 us  2791.571 us    22.54       1.7%   45.491s  ok 3.1e-16       1.00x
   mkl1d_dfti                   12.0018 us  3072.469 us    20.48      36.9%    0.003s  ok 3.1e-16       1.10x
   fftw1d_measure               14.9693 us  3832.145 us    16.42      75.3%    0.234s  ok 3.0e-16       1.37x
   fftw1d_estimate              21.6753 us  5548.874 us    11.34      46.7%    0.001s  ok 3.1e-16       1.99x

-- L=4096 (batched B=256, chain m=400), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                   22.8833 us 2343249.400 us    10.74       0.1%    0.002s  ok ch=4.6e-14/1e-10 1s=1e-15 1.00x
   fftw1d_patient               23.9202 us 2449427.040 us    10.27       0.8%   43.996s  ok ch=4.1e-14/1e-10 1s=1e-15 1.05x
   fftw1d_measure               28.6837 us 2937211.710 us     8.57       2.0%    0.233s  ok ch=4.1e-14/1e-10 1s=1e-15 1.25x
   fftw1d_estimate              35.3168 us 3616440.320 us     6.96       0.4%    0.001s  ok ch=4.1e-14/1e-10 1s=1e-15 1.54x

-- L=10007 (non-batched, single call), working set 0.305 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient              198.1671 us   198.167 us     3.36       3.6%   19.760s  ok 8.0e-16       1.00x
   mkl1d_dfti                  266.8753 us   266.875 us     2.49      32.0%    0.004s  ok 6.8e-16       1.35x
   fftw1d_measure              322.5148 us   322.515 us     2.06       1.8%    1.142s  ok 8.0e-16       1.63x
   fftw1d_estimate             562.7622 us   562.762 us     1.18       3.7%    0.002s  ok 8.5e-16       2.84x

-- L=10007 (non-batched, chain m=400), working set 0.305 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient              230.5323 us 92212.914 us     2.88       2.8%   19.797s  ok ch=3.3e-13/1e-10 1s=3e-15 1.00x
   mkl1d_dfti                  295.7338 us 118293.511 us     2.25       0.9%    0.002s  ok ch=2.7e-13/1e-10 1s=3e-15 1.28x
   fftw1d_measure              354.5291 us 141811.647 us     1.88       1.4%    1.138s  ok ch=3.9e-13/1e-10 1s=3e-15 1.54x
   fftw1d_estimate             492.9776 us 197191.052 us     1.35       0.3%    0.002s  ok ch=1.9e-13/1e-10 1s=3e-15 2.14x

-- L=10007 (batched B=64, single call), working set 19.545 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient              205.1253 us 13128.016 us     3.24       0.9%   19.731s  ok 8.2e-16       1.00x
   fftw1d_measure              216.1994 us 13836.764 us     3.08      25.9%    1.141s  ok 8.0e-16       1.05x
   mkl1d_dfti                  356.6877 us 22828.012 us     1.86       0.8%    0.057s  ok 6.7e-16       1.74x
   fftw1d_estimate             360.6601 us 23082.247 us     1.84      25.0%    0.002s  ok 8.5e-16       1.76x

-- L=10007 (batched B=64, chain m=80), working set 19.545 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient              238.6910 us 1222097.880 us     2.79       0.5%   19.746s  ok ch=3.4e-14/1e-10 1s=3e-15 1.00x
   fftw1d_measure              248.4479 us 1272053.180 us     2.68       1.5%    1.223s  ok ch=3.2e-14/1e-10 1s=3e-15 1.04x
   mkl1d_dfti                  304.5805 us 1559452.000 us     2.18       0.2%    0.054s  ok ch=3.3e-14/1e-10 1s=3e-15 1.28x
   fftw1d_estimate             390.2377 us 1998017.010 us     1.70       0.4%    0.002s  ok ch=3.7e-14/1e-10 1s=3e-15 1.63x

-- L=16384 (non-batched, single call), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               32.4932 us    32.493 us    35.30       3.5%    6.252s  ok 3.5e-16       1.00x
   mkl1d_dfti                   33.4950 us    33.495 us    34.24      13.1%    0.002s  ok 3.4e-16       1.03x
   fftw1d_measure               35.4718 us    35.472 us    32.33       1.0%    0.568s  ok 3.4e-16       1.09x
   fftw1d_estimate              69.6182 us    69.618 us    16.47      14.4%    0.001s  ok 3.5e-16       2.14x

-- L=16384 (non-batched, chain m=250), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               82.1565 us 20539.124 us    13.96       1.1%    6.386s  ok ch=3.0e-14/1e-10 1s=2e-15 1.00x
   mkl1d_dfti                   84.3586 us 21089.653 us    13.60      23.6%    0.003s  ok ch=3.3e-14/1e-10 1s=1e-15 1.03x
   fftw1d_measure               85.3707 us 21342.668 us    13.43      17.8%    0.600s  ok ch=3.6e-14/1e-10 1s=2e-15 1.04x
   fftw1d_estimate             135.3172 us 33829.305 us     8.48       0.2%    0.001s  ok ch=3.6e-14/1e-10 1s=1e-15 1.65x

-- L=16384 (batched B=64, single call), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               44.6950 us  2860.478 us    25.66       6.9%   67.455s  ok 3.5e-16       1.00x
   mkl1d_dfti                   59.6996 us  3820.777 us    19.21      24.8%    0.004s  ok 3.4e-16       1.34x
   fftw1d_measure               61.3832 us  3928.524 us    18.68      12.7%    0.568s  ok 3.4e-16       1.37x
   fftw1d_estimate              91.5904 us  5861.787 us    12.52      18.8%    0.001s  ok 3.5e-16       2.05x

-- L=16384 (batched B=64, chain m=150), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient               97.1211 us 932362.356 us    11.81       8.7%   66.682s  ok ch=2.8e-14/1e-10 1s=1e-15 1.00x
   mkl1d_dfti                   99.8905 us 958948.693 us    11.48       2.8%    0.003s  ok ch=2.8e-14/1e-10 1s=1e-15 1.03x
   fftw1d_measure              120.0816 us 1152783.230 us     9.55       0.2%    0.618s  ok ch=2.9e-14/1e-10 1s=1e-15 1.24x
   fftw1d_estimate             146.1310 us 1402857.650 us     7.85       1.0%    0.001s  ok ch=2.6e-14/1e-10 1s=1e-15 1.50x

-- L=65537 (non-batched, single call), working set 2.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             1473.3206 us  1473.321 us     3.56       1.2%   58.419s  ok 1.2e-15       1.00x
   fftw1d_measure             1579.0995 us  1579.100 us     3.32       3.1%    4.230s  ok 1.2e-15       1.07x
   fftw1d_estimate            1726.2492 us  1726.249 us     3.04      13.4%    0.011s  ok 1.2e-15       1.17x
   mkl1d_dfti                 3265.2067 us  3265.207 us     1.61      17.6%    0.014s  ok 1.0e-15       2.22x

-- L=65537 (non-batched, chain m=60), working set 2.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             1626.9905 us 97619.431 us     3.22       0.8%   58.008s  ok ch=6.4e-14/1e-10 1s=4e-15 1.00x
   fftw1d_measure             1735.6237 us 104137.424 us     3.02       0.8%    4.227s  ok ch=6.7e-14/1e-10 1s=4e-15 1.07x
   fftw1d_estimate            1893.2821 us 113596.925 us     2.77       0.6%    0.008s  ok ch=3.9e-14/1e-10 1s=4e-15 1.16x
   mkl1d_dfti                 3440.2214 us 206413.284 us     1.52       1.4%    0.013s  ok ch=3.5e-14/1e-10 1s=4e-15 2.11x

-- L=65537 (batched B=16, single call), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             1539.8404 us 24637.447 us     3.40       0.6%   58.289s  ok 1.2e-15       1.00x
   fftw1d_measure             1652.8237 us 26445.179 us     3.17       0.8%    4.429s  ok 1.2e-15       1.07x
   fftw1d_estimate            2420.6416 us 38730.266 us     2.17       1.6%    0.010s  ok 1.2e-15       1.57x
   mkl1d_dfti                 3992.6880 us 63883.008 us     1.31       3.3%    0.069s  ok 1.0e-15       2.59x

-- L=65537 (batched B=16, chain m=20), working set 32.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             1756.4603 us 562067.282 us     2.98       1.4%   57.866s  ok ch=2.0e-14/1e-10 1s=4e-15 1.00x
   fftw1d_measure             1893.2865 us 605851.673 us     2.77       0.1%    4.356s  ok ch=1.7e-14/1e-10 1s=4e-15 1.08x
   fftw1d_estimate            2087.6427 us 668045.667 us     2.51       0.9%    0.009s  ok ch=2.0e-14/1e-10 1s=4e-15 1.19x
   mkl1d_dfti                 3596.3939 us 1150846.050 us     1.46       0.1%    0.063s  ok ch=1.6e-14/1e-10 1s=4e-15 2.05x

-- L=100003 (non-batched, single call), working set 3.052 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             2689.0142 us  2689.014 us     3.09       1.4%   62.464s  ok 9.0e-16       1.00x
   fftw1d_measure             2745.0406 us  2745.041 us     3.03      10.9%    4.733s  ok 9.0e-16       1.02x
   mkl1d_dfti                 3893.2962 us  3893.296 us     2.13       9.7%    0.016s  ok 8.8e-16       1.45x
   fftw1d_estimate            5555.8663 us  5555.866 us     1.49      23.8%    0.016s  ok 9.1e-16       2.07x

-- L=100003 (non-batched, chain m=40), working set 3.052 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             3105.6192 us 124224.770 us     2.67       0.8%   62.773s  ok ch=2.5e-14/1e-10 1s=5e-15 1.00x
   fftw1d_measure             3211.1740 us 128446.959 us     2.59       2.1%    4.643s  ok ch=2.5e-14/1e-10 1s=5e-15 1.03x
   mkl1d_dfti                 3632.5981 us 145303.925 us     2.29       0.3%    0.013s  ok ch=2.4e-14/1e-10 1s=4e-15 1.17x
   fftw1d_estimate            5190.6837 us 207627.347 us     1.60       0.9%    0.014s  ok ch=2.6e-14/1e-10 1s=5e-15 1.67x

-- L=100003 (batched B=8, single call), working set 24.415 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             2810.0967 us 22480.774 us     2.96       1.6%   62.703s  ok 9.0e-16       1.00x
   fftw1d_measure             3033.3725 us 24266.980 us     2.74       1.7%    4.654s  ok 9.0e-16       1.08x
   mkl1d_dfti                 3490.4845 us 27923.876 us     2.38      17.5%    0.064s  ok 8.8e-16       1.24x
   fftw1d_estimate            4975.4987 us 39803.989 us     1.67      26.7%    0.013s  ok 9.1e-16       1.77x

-- L=100003 (batched B=8, chain m=15), working set 24.415 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_patient             3184.8066 us 382176.792 us     2.61       6.2%   62.542s  ok ch=1.5e-14/1e-10 1s=4e-15 1.00x
   fftw1d_measure             3253.5748 us 390428.976 us     2.55       3.9%    4.695s  ok ch=1.5e-14/1e-10 1s=4e-15 1.02x
   mkl1d_dfti                 3860.7573 us 463290.870 us     2.15       0.7%    0.070s  ok ch=1.5e-14/1e-10 1s=4e-15 1.21x
   fftw1d_estimate            5345.5598 us 641467.170 us     1.55       1.8%    0.012s  ok ch=1.5e-14/1e-10 1s=4e-15 1.68x

backends:
   fftw1d_custom            genfft monolithic codelet, split arrays, scalar DAG + autovec
   fftw1d_custom_soa        genfft monolithic codelet, SoA 8-transform batch-lane split-complex
   fftw1d_estimate          FFTW 3.3.10 plan_many_dft 1D, fftw1d_estimate
   fftw1d_measure           FFTW 3.3.10 plan_many_dft 1D, fftw1d_measure
   fftw1d_patient           FFTW 3.3.10 plan_many_dft 1D, fftw1d_patient
   mkl1d_dfti               oneMKL DFTI 1D, sequential, batched
```
