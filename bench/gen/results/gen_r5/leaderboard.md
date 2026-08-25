```
=== round gen_r5 ===
# round gen_r5
host: a80n0.lqcd.mit   date: 2026-08-24T21:32:29-04:00   slurm_job: 438682
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=10 (batched B=64, chain m=1000), volume 1000, working set 1.95 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.152 us 73733.529 us    43.25      13.8%    0.000s  ok ch=1.1e-13/1e-10 1s=8e-16 1.00x
   gen_batchlane                  1.158 us 74118.940 us    43.03      11.5%    0.000s  ok ch=1.0e-13/1e-10 1s=8e-16 1.01x
   gen_planner                    1.408 us 90092.907 us    35.40       0.7%    0.004s  ok ch=1.2e-13/1e-10 1s=9e-16 1.22x
   gen_race                       1.408 us 90112.110 us    35.39       0.8%    0.003s  ok ch=1.2e-13/1e-10 1s=9e-16 1.22x
   mkl_dfti                       4.570 us 292497.848 us    10.90       0.7%    0.002s  ok ch=1.5e-13/1e-10 1s=9e-16 3.97x
   mkl2026_dfti                   4.668 us 298779.391 us    10.67       0.2%    0.003s  ok ch=1.5e-13/1e-10 1s=9e-16 4.05x
   fftw3_patient                  5.028 us 321798.453 us     9.91       6.7%    0.022s  ok ch=1.0e-13/1e-10 1s=8e-16 4.36x
   fftw3_measure                  5.181 us 331604.727 us     9.62       4.4%    0.013s  ok ch=1.1e-13/1e-10 1s=8e-16 4.50x
   gen_layout                     5.259 us 336598.957 us     9.47       2.3%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 4.57x
   gen_dense_prime                5.583 us 357326.806 us     8.92       2.3%    0.000s  ok ch=1.3e-13/1e-10 1s=9e-16 4.85x
   fftw3_guru                     6.367 us 407479.985 us     7.83       0.5%    0.010s  ok ch=1.2e-13/1e-10 1s=8e-16 5.53x
   fftw3_estimate                 7.321 us 468562.590 us     6.81       0.2%    0.001s  ok ch=1.0e-13/1e-10 1s=8e-16 6.35x
   ducc0_c2c                      9.611 us 615089.911 us     5.18       1.2%    0.000s  ok ch=9.2e-14/1e-10 1s=7e-16 8.34x
   gen_twiddle                   10.359 us 662964.655 us     4.81       1.3%    0.002s  ok ch=1.1e-13/1e-10 1s=9e-16 8.99x
   gen_bluestein                 13.275 us 849610.005 us     3.75       3.0%    0.000s  ok ch=2.0e-13/1e-10 1s=1e-15 11.52x
   baseline_matrix               54.652 us 3497698.690 us     0.91       0.1%    0.000s  ok ch=3.4e-13/1e-10 1s=1e-15 47.44x

-- L=12 (batched B=64, chain m=600), volume 1728, working set 3.38 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  1.914 us 73513.586 us    48.54      13.8%    0.000s  ok ch=4.1e-14/1e-10 1s=9e-16 1.00x
   gen_batchlane                  1.915 us 73526.438 us    48.53       0.7%    0.000s  ok ch=4.1e-14/1e-10 1s=9e-16 1.00x
   gen_planner                    2.463 us 94591.047 us    37.72       0.4%    0.005s  ok ch=4.6e-14/1e-10 1s=9e-16 1.29x
   gen_race                       2.474 us 95019.830 us    37.55       2.3%    0.003s  ok ch=4.6e-14/1e-10 1s=9e-16 1.29x
   mkl_dfti                       7.761 us 298030.496 us    11.97       0.6%    0.002s  ok ch=4.0e-14/1e-10 1s=9e-16 4.05x
   mkl2026_dfti                   7.800 us 299516.041 us    11.91       0.3%    0.002s  ok ch=4.8e-14/1e-10 1s=9e-16 4.07x
   gen_dense_prime                8.091 us 310701.932 us    11.48       6.5%    0.000s  ok ch=4.2e-14/1e-10 1s=9e-16 4.23x
   gen_layout                     8.410 us 322927.807 us    11.05       5.3%    0.000s  ok ch=4.4e-14/1e-10 1s=1e-15 4.39x
   fftw3_measure                  8.888 us 341293.414 us    10.45       0.6%    0.014s  ok ch=3.9e-14/1e-10 1s=9e-16 4.64x
   fftw3_patient                  8.993 us 345329.291 us    10.33       3.9%    0.023s  ok ch=4.0e-14/1e-10 1s=9e-16 4.70x
   fftw3_guru                    10.173 us 390628.334 us     9.13       1.5%    0.011s  ok ch=4.8e-14/1e-10 1s=9e-16 5.31x
   gen_twiddle                   13.979 us 536774.552 us     6.65       4.1%    0.003s  ok ch=4.4e-14/1e-10 1s=1e-15 7.30x
   ducc0_c2c                     16.009 us 614740.034 us     5.80       2.4%    0.000s  ok ch=3.0e-14/1e-10 1s=7e-16 8.36x
   gen_bluestein                 19.294 us 740871.313 us     4.82       0.5%    0.000s  ok ch=8.3e-14/1e-10 1s=1e-15 10.08x
   fftw3_estimate                19.301 us 741177.133 us     4.81       0.1%    0.001s  ok ch=4.0e-14/1e-10 1s=9e-16 10.08x
   baseline_matrix              112.134 us 4305933.680 us     0.83       0.1%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 58.57x

-- L=15 (batched B=32, chain m=600), volume 3375, working set 3.30 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                  4.406 us 84603.596 us    44.89       0.1%    0.000s  ok ch=5.3e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                  4.411 us 84688.402 us    44.84       0.1%    0.000s  ok ch=5.3e-14/1e-10 1s=1e-15 1.00x
   gen_planner                    5.648 us 108449.426 us    35.02       2.6%    0.004s  ok ch=5.4e-14/1e-10 1s=1e-15 1.28x
   gen_race                       5.734 us 110091.448 us    34.49       0.2%    0.030s  ok ch=5.4e-14/1e-10 1s=1e-15 1.30x
   gen_dense_prime               14.875 us 285603.055 us    13.30       2.2%    0.000s  ok ch=5.3e-14/1e-10 1s=1e-15 3.38x
   mkl_dfti                      16.456 us 315963.750 us    12.02       0.6%    0.002s  ok ch=5.6e-14/1e-10 1s=1e-15 3.73x
   mkl2026_dfti                  16.638 us 319450.703 us    11.89       1.6%    0.002s  ok ch=6.3e-14/1e-10 1s=1e-15 3.78x
   gen_layout                    18.710 us 359230.318 us    10.57       2.0%    0.000s  ok ch=5.0e-14/1e-10 1s=1e-15 4.25x
   fftw3_patient                 19.700 us 378242.167 us    10.04       0.4%    0.020s  ok ch=5.7e-14/1e-10 1s=1e-15 4.47x
   fftw3_measure                 19.908 us 382238.641 us     9.93       0.5%    0.011s  ok ch=5.7e-14/1e-10 1s=1e-15 4.52x
   fftw3_estimate                20.637 us 396227.676 us     9.58       4.2%    0.001s  ok ch=5.7e-14/1e-10 1s=1e-15 4.68x
   fftw3_guru                    26.156 us 502192.152 us     7.56       2.4%    0.011s  ok ch=5.3e-14/1e-10 1s=1e-15 5.94x
   gen_twiddle                   28.146 us 540408.260 us     7.03       3.0%    0.003s  ok ch=5.6e-14/1e-10 1s=1e-15 6.39x
   ducc0_c2c                     32.448 us 623010.999 us     6.10       1.4%    0.000s  ok ch=4.0e-14/1e-10 1s=1e-15 7.36x
   gen_bluestein                 33.949 us 651826.353 us     5.83       0.3%    0.000s  ok ch=8.5e-14/1e-10 1s=2e-15 7.70x
   baseline_matrix              272.395 us 5229977.190 us     0.73       0.1%    0.000s  ok ch=2.7e-13/1e-10 1s=2e-15 61.82x

-- L=20 (batched B=32, chain m=256), volume 8000, working set 7.81 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_small                 13.072 us 107089.652 us    39.67       0.1%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 1.00x
   gen_batchlane                 13.268 us 108690.165 us    39.09       0.2%    0.001s  ok ch=3.0e-14/1e-10 1s=1e-15 1.01x
   gen_planner                   18.349 us 150318.246 us    28.26       7.1%    0.013s  ok ch=3.0e-14/1e-10 1s=1e-15 1.40x
   gen_race                      18.405 us 150777.636 us    28.18       8.0%    0.002s  ok ch=3.1e-14/1e-10 1s=1e-15 1.41x
   gen_dense_prime               38.949 us 319072.010 us    13.32       2.8%    0.000s  ok ch=3.8e-14/1e-10 1s=1e-15 2.98x
   gen_layout                    41.215 us 337631.402 us    12.58       3.8%    0.000s  ok ch=3.0e-14/1e-10 1s=1e-15 3.15x
   fftw3_measure                 44.944 us 368182.192 us    11.54       1.7%    0.083s  ok ch=3.1e-14/1e-10 1s=1e-15 3.44x
   fftw3_patient                 45.265 us 370808.725 us    11.46       1.1%    0.316s  ok ch=3.1e-14/1e-10 1s=1e-15 3.46x
   gen_twiddle                   55.357 us 453483.838 us     9.37       0.5%    0.004s  ok ch=2.7e-14/1e-10 1s=1e-15 4.23x
   mkl2026_dfti                  57.862 us 474004.548 us     8.96       3.3%    0.049s  ok ch=3.6e-14/1e-10 1s=1e-15 4.43x
   mkl_dfti                      58.330 us 477840.519 us     8.89       1.0%    0.032s  ok ch=3.3e-14/1e-10 1s=1e-15 4.46x
   fftw3_guru                    60.334 us 494257.293 us     8.60       1.2%    0.078s  ok ch=3.5e-14/1e-10 1s=1e-15 4.62x
   ducc0_c2c                     73.749 us 604148.993 us     7.03       0.1%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 5.64x
   fftw3_estimate                91.992 us 753594.922 us     5.64       0.2%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 7.04x
   gen_bluestein                103.410 us 847135.629 us     5.02       0.9%    0.000s  ok ch=4.1e-14/1e-10 1s=2e-15 7.91x
   baseline_matrix              850.280 us 6965491.900 us     0.61       0.0%    0.000s  ok ch=1.0e-13/1e-10 1s=2e-15 65.04x

-- L=25 (batched B=16, chain m=256), volume 15625, working set 7.63 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      31.352 us 128418.121 us    34.72       2.8%    0.298s  ok ch=3.8e-14/1e-10 1s=1e-15 1.00x
   gen_race                      40.223 us 164752.605 us    27.06       5.5%    0.002s  ok ch=4.4e-14/1e-10 1s=2e-15 1.28x
   gen_planner                   40.398 us 165469.143 us    26.94       7.7%    0.025s  ok ch=4.4e-14/1e-10 1s=2e-15 1.29x
   gen_layout                    95.488 us 391117.762 us    11.40       1.7%    0.000s  ok ch=3.9e-14/1e-10 1s=2e-15 3.05x
   fftw3_measure                109.191 us 447244.898 us     9.97       9.2%    0.029s  ok ch=4.1e-14/1e-10 1s=2e-15 3.48x
   fftw3_patient                109.226 us 447388.803 us     9.96       2.0%    0.112s  ok ch=4.1e-14/1e-10 1s=2e-15 3.48x
   gen_twiddle                  115.600 us 473496.240 us     9.42       2.6%    0.003s  ok ch=4.4e-14/1e-10 1s=2e-15 3.69x
   mkl_dfti                     120.558 us 493806.227 us     9.03       0.4%    0.030s  ok ch=5.2e-14/1e-10 1s=2e-15 3.85x
   mkl2026_dfti                 123.559 us 506096.000 us     8.81       0.2%    0.044s  ok ch=4.3e-14/1e-10 1s=2e-15 3.94x
   fftw3_estimate               132.094 us 541056.709 us     8.24       2.5%    0.001s  ok ch=4.1e-14/1e-10 1s=2e-15 4.21x
   ducc0_c2c                    145.979 us 597929.466 us     7.46       1.9%    0.000s  ok ch=2.8e-14/1e-10 1s=1e-15 4.66x
   gen_bluestein                179.949 us 737072.569 us     6.05       0.8%    0.000s  ok ch=5.4e-14/1e-10 1s=2e-15 5.74x
   fftw3_guru                   186.694 us 764698.816 us     5.83       2.9%    0.032s  ok ch=4.3e-14/1e-10 1s=2e-15 5.95x
   baseline_matrix             2061.305 us 8443107.220 us     0.53       0.0%    0.000s  ok ch=1.3e-13/1e-10 1s=3e-15 65.75x

-- L=27 (batched B=16, chain m=200), volume 19683, working set 9.61 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                      51.444 us 164621.397 us    27.29       0.1%    0.483s  ok ch=3.9e-14/1e-10 1s=2e-15 1.00x
   gen_planner                   78.446 us 251027.868 us    17.90       4.4%    0.031s  ok ch=3.5e-14/1e-10 1s=2e-15 1.52x
   gen_race                      80.145 us 256462.602 us    17.52       0.5%    0.005s  ok ch=3.5e-14/1e-10 1s=2e-15 1.56x
   gen_layout                   122.481 us 391940.279 us    11.46       2.2%    0.000s  ok ch=3.3e-14/1e-10 1s=2e-15 2.38x
   mkl_dfti                     144.359 us 461948.271 us     9.72       0.3%    0.036s  ok ch=4.0e-14/1e-10 1s=2e-15 2.81x
   mkl2026_dfti                 148.165 us 474128.936 us     9.47       0.3%    0.046s  ok ch=4.0e-14/1e-10 1s=2e-15 2.88x
   ducc0_c2c                    188.698 us 603832.968 us     7.44       0.9%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 3.67x
   fftw3_patient                202.751 us 648803.672 us     6.92       2.3%    0.138s  ok ch=2.8e-14/1e-10 1s=2e-15 3.94x
   gen_bluestein                214.250 us 685599.107 us     6.55       1.2%    0.000s  ok ch=5.3e-14/1e-10 1s=2e-15 4.16x
   gen_twiddle                  218.155 us 698094.580 us     6.44       2.2%    0.004s  ok ch=3.4e-14/1e-10 1s=2e-15 4.24x
   fftw3_measure                225.599 us 721915.595 us     6.22       2.2%    0.033s  ok ch=3.3e-14/1e-10 1s=2e-15 4.39x
   fftw3_estimate               253.690 us 811808.550 us     5.53       2.0%    0.003s  ok ch=3.3e-14/1e-10 1s=2e-15 4.93x
   fftw3_guru                   259.707 us 831063.540 us     5.41       1.2%    0.043s  ok ch=2.9e-14/1e-10 1s=2e-15 5.05x
   baseline_matrix             2799.054 us 8956974.220 us     0.50       0.0%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 54.41x

-- L=31 (batched B=16, chain m=140), volume 29791, working set 14.55 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_rader                     84.668 us 189655.982 us    26.15       1.1%    0.008s  ok ch=4.4e-14/1e-10 1s=2e-15 1.00x
   gen_dense_prime              120.287 us 269442.069 us    18.40       1.3%    0.001s  ok ch=3.0e-14/1e-10 1s=2e-15 1.42x
   gen_race                     139.795 us 313139.812 us    15.84       1.4%    0.003s  ok ch=3.0e-14/1e-10 1s=2e-15 1.65x
   gen_planner                  139.998 us 313594.893 us    15.81       3.8%    0.036s  ok ch=3.0e-14/1e-10 1s=2e-15 1.65x
   gen_layout                   195.931 us 438884.477 us    11.30       2.2%    0.000s  ok ch=3.4e-14/1e-10 1s=2e-15 2.31x
   gen_twiddle                  267.265 us 598673.747 us     8.28       1.9%    0.006s  ok ch=3.0e-14/1e-10 1s=2e-15 3.16x
   gen_bluestein                289.292 us 648015.136 us     7.65       1.2%    0.000s  ok ch=4.9e-14/1e-10 1s=3e-15 3.42x
   ducc0_c2c                    716.947 us 1605961.360 us     3.09       2.6%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 8.47x
   fftw3_guru                   832.937 us 1865779.450 us     2.66       0.5%    0.089s  ok ch=2.5e-14/1e-10 1s=2e-15 9.84x
   mkl_dfti                     848.862 us 1901451.440 us     2.61       0.0%    0.037s  ok ch=3.2e-14/1e-10 1s=2e-15 10.03x
   fftw3_estimate               859.739 us 1925814.400 us     2.58       0.0%    0.002s  ok ch=2.5e-14/1e-10 1s=2e-15 10.15x
   fftw3_measure                860.380 us 1927251.950 us     2.57       0.3%    0.086s  ok ch=2.5e-14/1e-10 1s=2e-15 10.16x
   fftw3_patient                860.404 us 1927305.630 us     2.57       0.4%    0.225s  ok ch=2.5e-14/1e-10 1s=2e-15 10.16x
   mkl2026_dfti                 882.914 us 1977727.670 us     2.51       0.0%    0.048s  ok ch=3.2e-14/1e-10 1s=2e-15 10.43x
   baseline_matrix             4852.190 us 10868906.000 us     0.46       0.1%    0.000s  ok ch=8.4e-14/1e-10 1s=3e-15 57.31x

-- L=32 (batched B=8, chain m=250), volume 32768, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pow2                      56.472 us 112943.572 us    43.52       1.7%    0.000s  ok ch=2.9e-14/1e-10 1s=1e-15 1.00x
   gen_planner                  128.479 us 256957.062 us    19.13       3.7%    0.003s  ok ch=3.5e-14/1e-10 1s=1e-15 2.28x
   gen_race                     130.728 us 261455.675 us    18.80       1.4%    0.125s  ok ch=3.4e-14/1e-10 1s=1e-15 2.31x
   mkl_dfti                     172.480 us 344960.578 us    14.25       0.9%    0.002s  ok ch=2.7e-14/1e-10 1s=1e-15 3.05x
   mkl2026_dfti                 188.015 us 376029.679 us    13.07       1.3%    0.002s  ok ch=2.8e-14/1e-10 1s=1e-15 3.33x
   gen_layout                   199.953 us 399905.821 us    12.29       1.2%    0.000s  ok ch=3.6e-14/1e-10 1s=2e-15 3.54x
   fftw3_patient                209.515 us 419030.415 us    11.73       0.7%    0.542s  ok ch=3.1e-14/1e-10 1s=1e-15 3.71x
   fftw3_measure                209.880 us 419760.327 us    11.71       0.6%    0.083s  ok ch=3.1e-14/1e-10 1s=1e-15 3.72x
   fftw3_guru                   277.591 us 555182.540 us     8.85      15.9%    0.071s  ok ch=3.2e-14/1e-10 1s=1e-15 4.92x
   ducc0_c2c                    308.586 us 617171.901 us     7.96       2.8%    0.000s  ok ch=2.5e-14/1e-10 1s=1e-15 5.46x
   gen_bluestein                316.623 us 633245.609 us     7.76       0.9%    0.000s  ok ch=4.9e-14/1e-10 1s=2e-15 5.61x
   gen_twiddle                  337.387 us 674773.494 us     7.28       0.8%    0.005s  ok ch=3.5e-14/1e-10 1s=1e-15 5.97x
   fftw3_estimate               409.093 us 818186.394 us     6.01       0.1%    0.001s  ok ch=3.1e-14/1e-10 1s=1e-15 7.24x
   baseline_matrix             5752.735 us 11505470.500 us     0.43       0.1%    0.000s  ok ch=1.2e-13/1e-10 1s=2e-15 101.87x

-- L=40 (batched B=8, chain m=128), volume 64000, working set 15.62 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large                160.240 us 164085.490 us    31.88       1.4%    1.140s  ok ch=5.0e-14/1e-10 1s=2e-15 1.00x
   gen_race                     281.470 us 288225.492 us    18.15       1.5%    0.002s  ok ch=6.7e-14/1e-10 1s=2e-15 1.76x
   gen_planner                  282.009 us 288777.337 us    18.12       1.0%    0.099s  ok ch=3.5e-14/1e-10 1s=2e-15 1.76x
   mkl2026_dfti                 405.524 us 415256.618 us    12.60       0.4%    0.002s  ok ch=4.5e-14/1e-10 1s=2e-15 2.53x
   mkl_dfti                     406.419 us 416173.550 us    12.57       0.3%    0.003s  ok ch=5.9e-14/1e-10 1s=2e-15 2.54x
   gen_layout                   456.698 us 467658.590 us    11.19       2.4%    0.001s  ok ch=3.2e-14/1e-10 1s=2e-15 2.85x
   fftw3_patient                517.111 us 529521.928 us     9.88       4.5%    1.287s  ok ch=2.7e-14/1e-10 1s=2e-15 3.23x
   fftw3_measure                548.939 us 562113.795 us     9.31       3.7%    0.148s  ok ch=2.4e-14/1e-10 1s=2e-15 3.43x
   ducc0_c2c                    593.594 us 607840.362 us     8.61       2.0%    0.000s  ok ch=4.6e-14/1e-10 1s=2e-15 3.70x
   gen_twiddle                  611.968 us 626655.100 us     8.35       4.5%    0.008s  ok ch=3.3e-14/1e-10 1s=2e-15 3.82x
   fftw3_guru                   678.810 us 695101.789 us     7.53       2.7%    0.153s  ok ch=3.2e-14/1e-10 1s=2e-15 4.24x
   gen_bluestein               1062.958 us 1088468.830 us     4.81       1.3%    0.000s  ok ch=8.1e-14/1e-10 1s=3e-15 6.63x
   fftw3_estimate              1604.176 us 1642676.060 us     3.18       0.7%    0.002s  ok ch=3.1e-14/1e-10 1s=2e-15 10.01x
   baseline_matrix            13454.469 us 13777376.300 us     0.38       0.3%    0.000s  ok ch=1.1e-13/1e-10 1s=3e-15 83.96x

-- L=50 (batched B=4, chain m=128), volume 125000, working set 15.26 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_powp                     415.637 us 212806.070 us    25.46       1.9%    0.001s  ok ch=3.1e-14/1e-10 1s=2e-15 1.00x
   gen_pfa_large                420.957 us 215529.769 us    25.14       0.4%    0.004s  ok ch=3.1e-14/1e-10 1s=2e-15 1.01x
   gen_planner                  641.679 us 328539.432 us    16.49       0.6%    0.011s  ok ch=3.0e-14/1e-10 1s=2e-15 1.54x
   gen_race                     644.310 us 329886.619 us    16.42       1.3%    0.072s  ok ch=3.0e-14/1e-10 1s=2e-15 1.55x
   mkl_dfti                     947.620 us 485181.443 us    11.17       0.2%    0.047s  ok ch=3.4e-14/1e-10 1s=3e-15 2.28x
   mkl2026_dfti                 962.413 us 492755.672 us    11.00       0.3%    0.045s  ok ch=3.6e-14/1e-10 1s=3e-15 2.32x
   gen_layout                  1104.910 us 565713.896 us     9.58       1.1%    0.001s  ok ch=3.4e-14/1e-10 1s=3e-15 2.66x
   fftw3_patient               1157.616 us 592699.300 us     9.14       1.1%    1.112s  ok ch=3.4e-14/1e-10 1s=2e-15 2.79x
   fftw3_measure               1188.189 us 608353.022 us     8.91       5.3%    0.088s  ok ch=3.2e-14/1e-10 1s=2e-15 2.86x
   ducc0_c2c                   1269.370 us 649917.681 us     8.34       2.4%    0.000s  ok ch=2.9e-14/1e-10 1s=2e-15 3.05x
   gen_twiddle                 1272.112 us 651321.229 us     8.32       1.8%    0.015s  ok ch=3.4e-14/1e-10 1s=2e-15 3.06x
   fftw3_estimate              1622.867 us 830907.997 us     6.52       1.1%    0.003s  ok ch=3.6e-14/1e-10 1s=2e-15 3.90x
   fftw3_guru                  1658.668 us 849237.896 us     6.38       0.2%    0.117s  ok ch=3.1e-14/1e-10 1s=2e-15 3.99x
   gen_bluestein               1767.065 us 904737.117 us     5.99       1.2%    0.000s  ok ch=4.5e-14/1e-10 1s=3e-15 4.25x
   baseline_matrix            34068.812 us 17443231.800 us     0.31       0.3%    0.000s  ok ch=7.9e-14/1e-10 1s=4e-15 81.97x

-- L=100 (non-batched, chain m=64), volume 1000000, working set 30.52 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   gen_pfa_large               4531.445 us 290012.474 us    21.99       2.1%    3.813s  ok ch=2.5e-14/1e-10 1s=3e-15 1.00x
   gen_powp                    4618.710 us 295597.454 us    21.58       9.3%    3.240s  ok ch=2.5e-14/1e-10 1s=3e-15 1.02x
   gen_race                    5491.319 us 351444.428 us    18.15       5.1%    0.006s  ok ch=2.4e-14/1e-10 1s=3e-15 1.21x
   gen_planner                 5519.543 us 353250.761 us    18.06       0.7%    0.329s  ok ch=2.4e-14/1e-10 1s=3e-15 1.22x
   mkl_dfti                    7790.545 us 498594.882 us    12.79       0.6%    0.031s  ok ch=2.9e-14/1e-10 1s=3e-15 1.72x
   mkl2026_dfti                7819.960 us 500477.448 us    12.74       0.6%    0.052s  ok ch=2.6e-14/1e-10 1s=3e-15 1.73x
   fftw3_patient               9932.734 us 635694.960 us    10.03       1.2%   17.207s  ok ch=2.5e-14/1e-10 1s=3e-15 2.19x
   gen_twiddle                10725.350 us 686422.415 us     9.29       0.6%    0.102s  ok ch=2.7e-14/1e-10 1s=3e-15 2.37x
   fftw3_measure              10938.892 us 700089.060 us     9.11      10.1%    0.277s  ok ch=2.6e-14/1e-10 1s=3e-15 2.41x
   ducc0_c2c                  11767.334 us 753109.361 us     8.47       1.7%    0.000s  ok ch=2.0e-14/1e-10 1s=2e-15 2.60x
   fftw3_guru                 14324.155 us 916745.924 us     6.96       9.4%    0.359s  ok ch=2.3e-14/1e-10 1s=3e-15 3.16x
   gen_layout                 14529.744 us 929903.644 us     6.86       0.5%    0.004s  ok ch=3.8e-14/1e-10 1s=3e-15 3.21x
   gen_bluestein              15518.928 us 993211.396 us     6.42       2.0%    0.000s  ok ch=3.2e-14/1e-10 1s=4e-15 3.42x
   fftw3_estimate             20614.873 us 1319351.900 us     4.83       0.6%    0.002s  ok ch=2.3e-14/1e-10 1s=3e-15 4.55x

backends:
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_guru               FFTW 3.3.10 guru split-array dft, FFTW_MEASURE, fused split chain
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   gen_batchlane            SoA 8-vol/zmm batch-lane (bl8 lineage): twiddle-free 2-stage PFA pencils (10=2x5,12=3x4,15=3x5,20=4x5), register-explicit at 10/12 (2L ld + 2L st), memory form at 15/20 (r5: reg spills past 24 live sites; 15 has the fused DFT5X2 equal-slot pair), L1 zy-sweep + x-pass, fused chain in SoA with eager rsqrt14 map (rcp14 ladder at 10/12, vdivpd at 15/20 -- r5 same-core re-race), sched-pressure on 10/12 only, THP arena (gen_layout), plane stride 256 mod 4096
   gen_bluestein            Bluestein chirp-Z for ANY L: pow2 radix-4/16 DIF/DIT convolution (no bit-reversal), 8-row SoA lanes, gather/scatter fused into the pruned end stages (masked dual-run loads keep seam groups vectorized), owned in-place map chain -- map fused into the axis-0 scatter while state+c fit LLC, else axis-0-first k-plane-blocked custody with the map fused into the axis-1 scatter reading a custody-ordered c (two aligned sequential streams; gen_pow2 GP2_CT), gen_twiddle exact tables
   gen_dense_prime          folded dense prime p<=31 (any prime in class supported): 4h^2-FMA conjugate-pair fold, z-pass fused into the x-contraction with the z-combine folded straight into U/V (no stack round-trip), fully in-place L2-resident chain on a padded 31x31x32 state (64B-aligned, mask-free), register-tiled EXACT-TILE GEMM (full 4-zmm d-chunks + one exact 1..3-zmm tail, exact k-tails -- no wasted FMA slots at any L), vectorized any-L z-pass, zmm z-row fold, LAZY map fused into the next step's z-loads (only the last step's map materializes)
   gen_layout               LIBRARY LAYER (scored by adoption): THP arenas, 4K stagger/collision-model placement + stream audit & measured pitch picker, pencil SoA pack (adopt: #define GEN_LAYOUT_LIB_ONLY + #include gen_layout.c); entry=any-L conjugate-pair-folded dense matrixsimd demo of the layer, r3: packed cross-plane axis-1 lanes + trailing axis-2 through a 4-plane collision-picked window; r4: fold-load software prefetch (~L row streams beat the L2 streamer), NT full-line stores on DRAM-resident volumes, fused transpose+interleave scatter; r5: graded map fused into the axis-2 exit (gl_map8/gl_map16 in-register map primitives), chain in place, zt volume deleted
   gen_pfa_large            GT-PFA 25x4 two-sweep (DFT25=5x5 CT exact tw) + owned chain (deferred NR map); pick: l100-ipp1 (B=1)
   gen_pfa_small            PFA coprime (10=2x5,12=3x4,15=3x5,20=4x5), no twiddles; interleaved site SoA 8 vols/zmm, padded planes 256 mod 4096, in-place slot codelets, zy sweep + x-pass w/ in-register fused map; B%8 split path; r3: generic runtime-table coprime P*Q engine (modules 2,3,4,5,7,8,9) for 6,14,18,21,24,28,35,36,45,56,63; r4: register-explicit 10/12 pencils, in-place generic pencils where Q==1 mod P (14,18,21,36,56); r5: per-size map ladder BODY+tail (bl hs-form + rcp at 12, bl + div at 15/20, legacy + div at 10), raced same-core
   gen_planner              planner layer: L -> {ct,gt-pfa,rader,bluestein,dense} candidate trees + generic strided-row executor (in-place, fused twiddles) + volume-resident fused chain, any 2<=L<=128; adopt via GEN_PLANNER_LIB include
   gen_pow2                 2^k axes: custody split-complex chain engine over G=L/8 (L=16/32/64: TR8 z-codelets, x-fastest c, lazy exact map, DSB-resident bodies, dual-select FMA-folded twiddles), other 2^k in 2..128 generic radix-2
   gen_powp                 powp CT 5x5 exact tw, SoA-8 lane chain (DIF/DIT in place); pick: l25-soa (B=16)
   gen_race                 LIBRARY LAYER (scored by adoption): plan-time candidate race (interleaved sample-major since r4: core-state-drift immune) + per-host wisdom cache incl. string wisdom + round-end drop_prefix (adopt: #define GEN_RACE_LIB_ONLY + #include gen_race.c); demo = round-6 trunk: pln_enumerate trees + gen_planner r5 split-group batch-lane arms (@s1/2/3, batch>=8) + tile width, all raced on the graded chain step by gr_pick, persisted, fused chain
   gen_rader                Rader-class primes 3..127: at 31, conjugate fold -> cyclic-15 (cos) + negacyclic-15 (sin; odd-N sign-twist), Winograd-C3 x dense-C5 on a fully padded huge-page arena (64B-aligned, anti-4K pitch, c mirror phase-split); 43/67/79/103 via OUTER-C3 Rader (the same Winograd-C3-over-dense-blocks at runtime tables, 8m^2 vs 18m^2 conv FMA); any other prime via the generic folded half-system engine; generic chain on a padded gl_map_huge arena for p<=61 (alias-free pitch, tail-free x/y), flat above; self-check gated; s6 map from gen_dense_prime, arena from gen_layout
   gen_twiddle              LIBRARY LAYER (scored by adoption): octant-folded exact twiddles <=0.51 ulp (tw_cis/tw_chirp) + NEW dual-select FMA form tw_cis_ds (lit 11 Tier 1: every stored ratio <=1, first performant validation), consumption-order CT/DFT/Rader(+folded)/chirp/fold-half/SIMD-dense fillers + ulp audits + primitive roots + long-double DFT oracle (adopt: #define GEN_TWIDDLE_LIB_ONLY + #include gen_twiddle.c); entry = any-L mixed-radix zmm-lane demo (gen_r5: conjugate-fold prime butterflies, pair-packed map ladder), self-audited at create(), owned in-place fused-map chain
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
