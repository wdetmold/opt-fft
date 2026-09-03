# What previous generations produced (round d1_r4 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r3/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 370 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 401 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 342 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 321 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 337 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 320 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 350 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 404 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 360 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/d1/exemplars/d1_r3/
      # Round d1_r3 — what it established
      
      Promoted: d1_prime d1_rader d1_pow2 d1_batchlane d1_composite d1_bluestein d1_twiddle d1_race
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round d1_r3 ===
# round d1_r3
host: a80n0.lqcd.mit   date: 2026-09-03T03:54:53-04:00   slurm_job: 440424
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0166 us     0.017 us    14.48      16.5%    0.019s  ok 1.2e-16       1.00x
   d1_prime                      0.0172 us     0.017 us    14.02      37.3%    0.000s  ok 1.2e-16       1.03x
   fftw1d_estimate               0.0218 us     0.022 us    11.02      20.9%    0.001s  ok 1.3e-16       1.31x
   fftw1d_measure                0.0219 us     0.022 us    11.00      22.0%    0.001s  ok 1.3e-16       1.32x
   fftw1d_patient                0.0219 us     0.022 us    10.96      20.4%    0.001s  ok 1.3e-16       1.32x
   d1_batchlane                  0.0242 us     0.024 us     9.92      56.4%    0.000s  ok 1.1e-16       1.46x
   mkl1d_dfti                    0.0259 us     0.026 us     9.29       0.1%    0.014s  ok 1.1e-16       1.56x
   fftw1d_custom                 0.0567 us     0.057 us     4.24       0.3%    0.000s  ok 1.4e-16       3.41x
   d1_rader                      0.0568 us     0.057 us     4.23       0.1%    0.000s  ok 2.8e-16       3.42x
   d1_planner                    0.0897 us     0.090 us     2.68      15.4%    0.000s  ok 1.5e-16       5.40x
   d1_bluestein                  0.1074 us     0.107 us     2.24       2.2%    0.000s  ok 1.3e-16       6.47x
   baseline_dft                  0.3533 us     0.353 us     0.68       0.3%    0.000s  ok 3.9e-16       21.27x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_batchlane                  0.0392 us  7835.864 us     6.14      13.8%    0.000s  ok ch=3.4e-16/1e-10 1s=4e-16 1.00x
   d1_race                       0.0393 us  7867.331 us     6.11      13.8%    0.012s  ok ch=3.4e-16/1e-10 1s=4e-16 1.00x
   d1_prime                      0.0395 us  7898.147 us     6.09       0.1%    0.000s  ok ch=1.1e-15/1e-10 1s=4e-16 1.01x
   fftw1d_custom                 0.0625 us 12501.935 us     3.85      20.7%    0.000s  ok ch=6.2e-16/1e-10 1s=3e-16 1.60x
   fftw1d_measure                0.0674 us 13483.898 us     3.57      20.7%    0.001s  ok ch=4.9e-16/1e-10 1s=3e-16 1.72x
   fftw1d_patient                0.0674 us 13484.011 us     3.57       0.0%    0.001s  ok ch=4.9e-16/1e-10 1s=3e-16 1.72x
   fftw1d_estimate               0.0674 us 13485.901 us     3.57       0.0%    0.001s  ok ch=4.9e-16/1e-10 1s=3e-16 1.72x
   d1_rader                      0.0801 us 16021.457 us     3.00       0.0%    0.000s  ok ch=3.2e-15/1e-10 1s=3e-16 2.04x
   mkl1d_dfti                    0.0803 us 16052.928 us     3.00       0.2%    0.048s  ok ch=1.3e-15/1e-10 1s=4e-16 2.05x
   d1_planner                    0.0863 us 17252.029 us     2.79      20.8%    0.000s  ok ch=4.4e-16/1e-10 1s=3e-16 2.20x
   d1_bluestein                  0.1458 us 29154.937 us     1.65       0.4%    0.000s  ok ch=4.5e-16/1e-10 1s=2e-16 3.72x
   baseline_dft                  0.3678 us 73550.614 us     0.65       7.7%    0.000s  ok ch=7.4e-15/1e-10 1s=6e-16 9.39x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0107 us     5.455 us    22.58       0.1%    0.000s  ok 1.7e-16       1.00x
   d1_race                       0.0107 us     5.467 us    22.52       1.7%    0.021s  ok 1.7e-16       1.00x
   fftw1d_measure                0.0123 us     6.317 us    19.50      14.4%    0.002s  ok 1.9e-16       1.16x
   fftw1d_patient                0.0140 us     7.166 us    17.19       4.5%    0.003s  ok 1.9e-16       1.31x
   d1_rader                      0.0186 us     9.544 us    12.90       0.5%    0.000s  ok 2.8e-16       1.75x
   d1_batchlane                  0.0189 us     9.654 us    12.76      14.7%    0.000s  ok 1.4e-16       1.77x
   mkl1d_dfti                    0.0193 us     9.873 us    12.47       0.2%    0.054s  ok 1.8e-16       1.81x
   fftw1d_custom_soa             0.0327 us    16.717 us     7.37      13.9%    0.000s  ok 1.8e-16       3.06x
   fftw1d_estimate               0.0352 us    18.022 us     6.83      13.9%    0.001s  ok 1.9e-16       3.30x
   fftw1d_custom                 0.0486 us    24.894 us     4.95       0.3%    0.000s  ok 1.8e-16       4.56x
   d1_planner                    0.0533 us    27.280 us     4.51      22.1%    0.000s  ok 1.4e-16       5.00x
   d1_bluestein                  0.1037 us    53.083 us     2.32       4.3%    0.000s  ok 1.8e-16       9.73x
   baseline_dft                  0.2899 us   148.423 us     0.83      20.7%    0.000s  ok 4.1e-16       27.21x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0146 us 14944.936 us    16.48      13.8%    0.010s  ok ch=1.9e-14/1e-10 1s=5e-16 1.00x
   d1_rader                      0.0146 us 14946.567 us    16.48      13.8%    0.000s  ok ch=1.9e-14/1e-10 1s=5e-16 1.00x
   d1_prime                      0.0154 us 15777.631 us    15.61      13.7%    0.000s  ok ch=2.0e-13/1e-10 1s=4e-16 1.06x
   d1_batchlane                  0.0206 us 21138.711 us    11.65       7.0%    0.000s  ok ch=2.1e-14/1e-10 1s=3e-16 1.41x
   fftw1d_custom_soa             0.0262 us 26844.917 us     9.17      13.9%    0.000s  ok ch=3.3e-14/1e-10 1s=3e-16 1.80x
   fftw1d_custom                 0.0475 us 48653.895 us     5.06      21.6%    0.000s  ok ch=3.3e-14/1e-10 1s=3e-16 3.26x
   fftw1d_patient                0.0513 us 52498.714 us     4.69       0.5%    0.002s  ok ch=9.1e-14/1e-10 1s=3e-16 3.51x
   mkl1d_dfti                    0.0530 us 54290.372 us     4.54      20.7%    0.049s  ok ch=2.1e-13/1e-10 1s=3e-16 3.63x
   fftw1d_measure                0.0584 us 59810.774 us     4.12       2.0%    0.002s  ok ch=9.1e-14/1e-10 1s=3e-16 4.00x
   d1_planner                    0.0646 us 66109.563 us     3.73       9.4%    0.000s  ok ch=1.1e-14/1e-10 1s=3e-16 4.42x
   fftw1d_estimate               0.0747 us 76472.549 us     3.22       0.1%    0.001s  ok ch=9.1e-14/1e-10 1s=3e-16 5.12x
   d1_bluestein                  0.1229 us 125898.060 us     1.96       1.9%    0.000s  ok ch=6.2e-15/1e-10 1s=3e-16 8.42x
   baseline_dft                  0.3270 us 334883.619 us     0.74       0.0%    0.000s  ok ch=4.9e-13/1e-10 1s=7e-16 22.41x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0530 us     0.053 us    14.49       2.5%    0.000s  ok 3.6e-16       1.00x
   d1_race                       0.0538 us     0.054 us    14.27       5.5%    0.015s  ok 3.6e-16       1.02x
   d1_batchlane                  0.0666 us     0.067 us    11.54      14.9%    0.000s  ok 2.3e-16       1.26x
   d1_rader                      0.2113 us     0.211 us     3.63       2.6%    0.000s  ok 3.5e-16       3.99x
   fftw1d_custom                 0.2148 us     0.215 us     3.57       1.0%    0.000s  ok 2.6e-16       4.05x
   mkl1d_dfti                    0.2794 us     0.279 us     2.75       0.4%    0.031s  ok 2.9e-16       5.27x
   d1_bluestein                  0.3071 us     0.307 us     2.50       6.9%    0.002s  ok 3.1e-16       5.80x
   fftw1d_measure                0.3133 us     0.313 us     2.45       1.5%    0.043s  ok 1.9e-16       5.91x
   fftw1d_estimate               0.3135 us     0.313 us     2.45       4.0%    0.001s  ok 1.9e-16       5.92x
   fftw1d_patient                0.3137 us     0.314 us     2.45       0.5%    0.121s  ok 1.9e-16       5.92x
   d1_planner                    0.3143 us     0.314 us     2.44      15.1%    0.000s  ok 2.0e-16       5.93x
   baseline_dft                  1.9980 us     1.998 us     0.38       0.0%    0.000s  ok 4.7e-16       37.71x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_batchlane                  0.0546 us  5458.660 us    14.07      13.9%    0.000s  ok ch=2.2e-15/1e-10 1s=4e-16 1.00x
   d1_race                       0.0619 us  6190.379 us    12.40       0.3%    0.007s  ok ch=2.2e-15/1e-10 1s=4e-16 1.13x
   d1_prime                      0.0662 us  6622.649 us    11.60       0.0%    0.000s  ok ch=2.2e-15/1e-10 1s=7e-16 1.21x
   d1_rader                      0.2348 us 23483.766 us     3.27       1.1%    0.000s  ok ch=1.9e-14/1e-10 1s=9e-16 4.30x
   fftw1d_custom                 0.2520 us 25195.004 us     3.05       0.8%    0.000s  ok ch=1.6e-15/1e-10 1s=5e-16 4.62x
   d1_bluestein                  0.3045 us 30446.086 us     2.52      15.8%    0.001s  ok ch=7.6e-15/1e-10 1s=5e-16 5.58x
   d1_planner                    0.3328 us 33275.716 us     2.31      20.9%    0.000s  ok ch=1.8e-15/1e-10 1s=3e-16 6.10x
   fftw1d_measure                0.3475 us 34751.550 us     2.21       1.0%    0.041s  ok ch=9.7e-16/1e-10 1s=3e-16 6.37x
   fftw1d_estimate               0.3498 us 34980.723 us     2.20      21.9%    0.001s  ok ch=9.7e-16/1e-10 1s=3e-16 6.41x
   fftw1d_patient                0.3509 us 35089.920 us     2.19      19.6%    0.119s  ok ch=9.7e-16/1e-10 1s=3e-16 6.43x
   mkl1d_dfti                    0.3848 us 38479.449 us     2.00       0.2%    0.013s  ok ch=2.1e-15/1e-10 1s=5e-16 7.05x
   baseline_dft                  1.7390 us 173903.671 us     0.44       0.0%    0.000s  ok ch=3.3e-14/1e-10 1s=8e-16 31.86x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0441 us    22.569 us    17.42      13.9%    0.010s  ok 2.9e-16       1.00x
   d1_prime                      0.0501 us    25.664 us    15.32       0.0%    0.000s  ok 2.9e-16       1.14x
   d1_rader                      0.0630 us    32.245 us    12.19      14.4%    0.000s  ok 3.6e-16       1.43x
   d1_batchlane                  0.0664 us    34.004 us    11.56       1.3%    0.000s  ok 2.1e-16       1.51x
   fftw1d_custom_soa             0.0913 us    46.752 us     8.41      15.3%    0.000s  ok 2.6e-16       2.07x
   fftw1d_custom                 0.1956 us   100.131 us     3.93       5.2%    0.000s  ok 2.6e-16       4.44x
   mkl1d_dfti                    0.2271 us   116.291 us     3.38      20.6%    0.053s  ok 2.3e-16       5.15x
   d1_planner                    0.2487 us   127.319 us     3.09       5.5%    0.000s  ok 2.1e-16       5.64x
   fftw1d_estimate               0.2633 us   134.830 us     2.92      20.7%    0.001s  ok 2.1e-16       5.97x
   fftw1d_measure                0.2633 us   134.831 us     2.92      20.7%    0.041s  ok 2.1e-16       5.97x
   d1_bluestein                  0.3042 us   155.771 us     2.52      10.1%    0.002s  ok 3.2e-16       6.90x
   fftw1d_patient                0.3177 us   162.668 us     2.42       0.1%    0.121s  ok 2.1e-16       7.21x
   baseline_dft                  1.9939 us  1020.876 us     0.39       0.0%    0.000s  ok 4.3e-16       45.23x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0477 us 29318.475 us    16.09       1.7%    0.006s  ok ch=2.5e-12/3e-10 1s=7e-16 1.00x
   d1_prime                      0.0485 us 29805.736 us    15.83       0.1%    0.000s  ok ch=2.5e-12/3e-10 1s=7e-16 1.02x
   d1_rader                      0.0552 us 33944.937 us    13.90       2.5%    0.000s  ok ch=5.9e-12/3e-10 1s=7e-16 1.16x
   d1_batchlane                  0.0628 us 38564.613 us    12.23      12.1%    0.000s  ok ch=5.7e-13/3e-10 1s=4e-16 1.32x
   fftw1d_custom_soa             0.0773 us 47477.600 us     9.94       1.6%    0.000s  ok ch=1.5e-12/3e-10 1s=5e-16 1.62x
   fftw1d_custom                 0.1771 us 108788.875 us     4.34       0.4%    0.000s  ok ch=1.5e-12/3e-10 1s=5e-16 3.71x
   d1_planner                    0.2680 us 164651.301 us     2.87       1.0%    0.000s  ok ch=4.4e-13/3e-10 1s=4e-16 5.62x
   d1_bluestein                  0.3056 us 187736.505 us     2.51       6.6%    0.001s  ok ch=2.7e-12/3e-10 1s=7e-16 6.40x
   mkl1d_dfti                    0.3152 us 193670.328 us     2.44       0.2%    0.050s  ok ch=1.2e-12/3e-10 1s=5e-16 6.61x
   fftw1d_measure                0.3534 us 217151.762 us     2.17       0.1%    0.042s  ok ch=9.8e-13/3e-10 1s=4e-16 7.41x
   fftw1d_estimate               0.3535 us 217216.444 us     2.17       0.0%    0.001s  ok ch=9.8e-13/3e-10 1s=4e-16 7.41x
   fftw1d_patient                0.3537 us 217329.291 us     2.17       0.0%    0.119s  ok ch=9.8e-13/3e-10 1s=4e-16 7.41x
   baseline_dft                  1.7450 us 1072137.030 us     0.44       0.0%    0.000s  ok ch=5.6e-12/3e-10 1s=8e-16 36.57x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0206 us     0.021 us    38.76       9.6%    0.067s  ok 1.3e-16       1.00x
   d1_pow2                       0.0207 us     0.021 us    38.64      13.9%    0.000s  ok 1.3e-16       1.00x
   mkl1d_dfti                    0.0225 us     0.022 us    35.59      42.0%    0.002s  ok 1.2e-16       1.09x
   d1_batchlane                  0.0311 us     0.031 us    25.72      20.7%    0.000s  ok 8.6e-17       1.51x
   fftw1d_measure                0.0335 us     0.033 us    23.89      16.3%    0.007s  ok 1.2e-16       1.62x
   d1_twiddle                    0.0348 us     0.035 us    22.97      17.5%    0.000s  ok 1.2e-16       1.69x
   fftw1d_patient                0.0372 us     0.037 us    21.51       3.0%    0.015s  ok 1.2e-16       1.80x
   fftw1d_estimate               0.0416 us     0.042 us    19.21      14.4%    0.001s  ok 1.1e-16       2.02x
   d1_planner                    0.0668 us     0.067 us    11.97       5.6%    0.000s  ok 1.1e-16       3.24x
   fftw1d_custom                 0.0913 us     0.091 us     8.76      20.2%    0.000s  ok 1.3e-16       4.42x
   d1_bluestein                  0.0964 us     0.096 us     8.30      12.5%    0.001s  ok 1.1e-16       4.67x
   baseline_dft                  1.7634 us     1.763 us     0.45      20.7%    0.000s  ok 3.4e-16       85.45x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_batchlane                  0.0591 us  5907.456 us    13.54      18.2%    0.000s  ok ch=1.8e-15/1e-10 1s=3e-16 1.00x
   d1_race                       0.0598 us  5979.310 us    13.38      16.9%    0.009s  ok ch=1.8e-15/1e-10 1s=3e-16 1.01x
   d1_pow2                       0.0706 us  7060.490 us    11.33       0.0%    0.000s  ok ch=3.2e-15/1e-10 1s=3e-16 1.20x
   d1_twiddle                    0.0830 us  8303.403 us     9.63      13.7%    0.000s  ok ch=2.0e-15/1e-10 1s=3e-16 1.41x
   d1_planner                    0.1043 us 10431.256 us     7.67      10.9%    0.000s  ok ch=2.8e-15/1e-10 1s=3e-16 1.77x
   fftw1d_custom                 0.1223 us 12231.126 us     6.54      20.0%    0.000s  ok ch=2.8e-15/1e-10 1s=3e-16 2.07x
   mkl1d_dfti                    0.1310 us 13095.806 us     6.11       0.1%    0.001s  ok ch=2.5e-15/1e-10 1s=3e-16 2.22x
   fftw1d_measure                0.1359 us 13594.115 us     5.88      13.9%    0.007s  ok ch=4.3e-15/1e-10 1s=3e-16 2.30x
   fftw1d_patient                0.1360 us 13604.075 us     5.88      13.8%    0.015s  ok ch=3.5e-15/1e-10 1s=3e-16 2.30x
   fftw1d_estimate               0.1453 us 14530.983 us     5.51      16.2%    0.001s  ok ch=4.3e-15/1e-10 1s=3e-16 2.46x
   d1_bluestein                  0.1847 us 18465.987 us     4.33       3.6%    0.001s  ok ch=3.5e-15/1e-10 1s=3e-16 3.13x
   baseline_dft                  1.8470 us 184695.979 us     0.43       0.0%    0.000s  ok ch=3.8e-14/1e-10 1s=5e-16 31.26x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0153 us     7.827 us    52.33      13.8%    0.002s  ok 1.3e-16       1.00x
   d1_pow2                       0.0154 us     7.881 us    51.97      13.9%    0.000s  ok 1.4e-16       1.01x
   d1_race                       0.0176 us     8.996 us    45.53       0.1%    0.014s  ok 1.4e-16       1.15x
   fftw1d_measure                0.0250 us    12.780 us    32.05      13.3%    0.009s  ok 1.6e-16       1.63x
   fftw1d_patient                0.0252 us    12.877 us    31.81      14.7%    0.055s  ok 1.6e-16       1.65x
   d1_twiddle                    0.0292 us    14.960 us    27.38      24.6%    0.000s  ok 1.4e-16       1.91x
   d1_batchlane                  0.0315 us    16.110 us    25.43      14.4%    0.000s  ok 1.1e-16       2.06x
   d1_planner                    0.0594 us    30.425 us    13.46       3.4%    0.000s  ok 1.4e-16       3.89x
   fftw1d_estimate               0.0888 us    45.486 us     9.00      13.5%    0.001s  ok 1.5e-16       5.81x
   fftw1d_custom_soa             0.0907 us    46.435 us     8.82       4.3%    0.000s  ok 1.4e-16       5.93x
   d1_bluestein                  0.0913 us    46.756 us     8.76      17.1%    0.001s  ok 1.4e-16       5.97x
   fftw1d_custom                 0.1049 us    53.728 us     7.62      10.2%    0.000s  ok 1.4e-16       6.86x
   baseline_dft                  2.1241 us  1087.538 us     0.38       0.0%    0.000s  ok 3.4e-16       138.94x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0382 us 19560.198 us    20.94       1.2%    0.010s  ok ch=2.9e-13/1e-10 1s=4e-16 1.00x
   d1_pow2                       0.0384 us 19641.436 us    20.85       1.2%    0.000s  ok ch=2.9e-13/1e-10 1s=4e-16 1.00x
   d1_batchlane                  0.0384 us 19651.824 us    20.84      14.6%    0.000s  ok ch=3.0e-13/1e-10 1s=3e-16 1.00x
   d1_twiddle                    0.0413 us 21134.120 us    19.38       0.8%    0.000s  ok ch=3.2e-13/1e-10 1s=3e-16 1.08x
   fftw1d_custom_soa             0.0665 us 34054.904 us    12.03       0.4%    0.000s  ok ch=3.7e-13/1e-10 1s=3e-16 1.74x
   fftw1d_custom                 0.1040 us 53267.371 us     7.69      21.2%    0.000s  ok ch=3.7e-13/1e-10 1s=3e-16 2.72x
   d1_planner                    0.1076 us 55091.647 us     7.43       8.2%    0.000s  ok ch=5.0e-13/1e-10 1s=3e-16 2.82x
   mkl1d_dfti                    0.1123 us 57475.336 us     7.13      13.8%    0.002s  ok ch=3.1e-13/1e-10 1s=3e-16 2.94x
   fftw1d_patient                0.1218 us 62349.594 us     6.57      14.2%    0.062s  ok ch=3.0e-13/1e-10 1s=4e-16 3.19x
   fftw1d_measure                0.1228 us 62887.790 us     6.51       0.6%    0.009s  ok ch=3.0e-13/1e-10 1s=4e-16 3.22x
   d1_bluestein                  0.1774 us 90853.770 us     4.51       0.9%    0.001s  ok ch=2.9e-13/1e-10 1s=3e-16 4.64x
   fftw1d_estimate               0.1859 us 95157.773 us     4.30       0.2%    0.000s  ok ch=3.0e-13/1e-10 1s=4e-16 4.86x
   baseline_dft                  1.9513 us 999047.599 us     0.41       0.1%    0.000s  ok ch=6.3e-13/1e-10 1s=6e-16 51.08x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0473 us     0.047 us    37.50       0.0%    0.103s  ok 2.1e-16       1.00x
   d1_composite                  0.0489 us     0.049 us    36.25      14.0%    0.000s  ok 2.1e-16       1.03x
   mkl1d_dfti                    0.0608 us     0.061 us    29.15       7.5%    0.002s  ok 2.6e-16       1.29x
   fftw1d_patient                0.0676 us     0.068 us    26.23       9.7%    0.127s  ok 2.2e-16       1.43x
