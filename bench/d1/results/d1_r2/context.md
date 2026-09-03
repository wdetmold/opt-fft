# What previous generations produced (round d1_r2 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 121 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 138 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 132 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 99 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 108 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 107 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 131 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 149 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 122 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round d1_r1 ===
# round d1_r1
host: a80n0.lqcd.mit   date: 2026-09-02T18:30:52-04:00   slurm_job: 440371
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0216 us     0.022 us    11.12      15.9%    0.000s  ok 1.9e-16       1.00x
   fftw1d_estimate               0.0218 us     0.022 us    11.01       0.1%    0.000s  ok 1.5e-16       1.01x
   fftw1d_measure                0.0219 us     0.022 us    11.00      40.2%    0.001s  ok 1.5e-16       1.01x
   d1_race                       0.0222 us     0.022 us    10.83      24.2%    0.004s  ok 1.9e-16       1.03x
   mkl1d_dfti                    0.0259 us     0.026 us     9.29      11.4%    0.051s  ok 1.7e-16       1.20x
   fftw1d_patient                0.0264 us     0.026 us     9.11      11.9%    0.001s  ok 1.5e-16       1.22x
   d1_batchlane                  0.0383 us     0.038 us     6.29       4.9%    0.000s  ok 3.4e-16       1.77x
   fftw1d_custom                 0.0470 us     0.047 us     5.12      20.8%    0.000s  ok 1.1e-16       2.17x
   d1_planner                    0.0896 us     0.090 us     2.69       0.3%    0.000s  ok 4.1e-16       4.14x
   d1_bluestein                  0.0939 us     0.094 us     2.56      20.2%    0.000s  ok 4.5e-16       4.34x
   d1_rader                      0.1128 us     0.113 us     2.13      17.2%    0.000s  ok 3.4e-16       5.21x
   baseline_dft                  0.2927 us     0.293 us     0.82      20.7%    0.000s  ok 4.5e-16       13.53x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0406 us  8122.921 us     5.92      13.8%    0.005s  ok ch=1.5e-15/1e-10 1s=3e-16 1.00x
   d1_prime                      0.0406 us  8125.488 us     5.92      13.8%    0.000s  ok ch=1.5e-15/1e-10 1s=3e-16 1.00x
   mkl1d_dfti                    0.0665 us 13305.846 us     3.62      21.1%    0.043s  ok ch=1.6e-15/1e-10 1s=2e-16 1.64x
   d1_batchlane                  0.0665 us 13308.529 us     3.61      13.7%    0.000s  ok ch=6.1e-15/1e-10 1s=4e-16 1.64x
   fftw1d_patient                0.0674 us 13482.760 us     3.57      20.7%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 1.66x
   fftw1d_measure                0.0674 us 13485.186 us     3.57      18.1%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 1.66x
   fftw1d_custom                 0.0754 us 15072.497 us     3.19       0.5%    0.000s  ok ch=1.4e-15/1e-10 1s=3e-16 1.86x
   fftw1d_estimate               0.0814 us 16273.226 us     2.96       0.0%    0.001s  ok ch=1.5e-15/1e-10 1s=3e-16 2.00x
   d1_planner                    0.1235 us 24707.080 us     1.95      20.4%    0.000s  ok ch=5.5e-15/1e-10 1s=6e-16 3.04x
   d1_bluestein                  0.1276 us 25513.027 us     1.89       1.4%    0.000s  ok ch=4.1e-15/1e-10 1s=8e-16 3.14x
   d1_rader                      0.1367 us 27347.202 us     1.76       4.0%    0.000s  ok ch=2.8e-15/1e-10 1s=3e-16 3.37x
   baseline_dft                  0.3256 us 65116.067 us     0.74      21.0%    0.000s  ok ch=4.1e-15/1e-10 1s=8e-16 8.02x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   fftw1d_measure                0.0123 us     6.311 us    19.51      18.1%    0.002s  ok 1.8e-16       1.00x
   fftw1d_patient                0.0140 us     7.180 us    17.15       2.3%    0.002s  ok 1.8e-16       1.14x
   d1_prime                      0.0146 us     7.477 us    16.47      14.7%    0.000s  ok 1.7e-16       1.18x
   d1_race                       0.0147 us     7.537 us    16.34      14.0%    0.007s  ok 1.7e-16       1.19x
   d1_batchlane                  0.0189 us     9.655 us    12.76       1.5%    0.000s  ok 3.4e-16       1.53x
   mkl1d_dfti                    0.0192 us     9.843 us    12.51       0.6%    0.034s  ok 1.7e-16       1.56x
   fftw1d_estimate               0.0352 us    17.998 us     6.84      14.0%    0.001s  ok 1.8e-16       2.85x
   fftw1d_custom_soa             0.0368 us    18.820 us     6.54       0.5%    0.000s  ok 1.8e-16       2.98x
   fftw1d_custom                 0.0407 us    20.813 us     5.92      19.8%    0.000s  ok 1.8e-16       3.30x
   d1_bluestein                  0.0881 us    45.097 us     2.73      27.9%    0.000s  ok 4.1e-16       7.15x
   d1_planner                    0.1100 us    56.308 us     2.19       1.0%    0.000s  ok 4.1e-16       8.92x
   d1_rader                      0.1151 us    58.916 us     2.09      13.7%    0.000s  ok 3.3e-16       9.33x
   baseline_dft                  0.2899 us   148.415 us     0.83       0.0%    0.000s  ok 4.1e-16       23.52x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0154 us 15768.727 us    15.62      13.8%    0.000s  ok ch=6.3e-14/1e-10 1s=4e-16 1.00x
   d1_race                       0.0155 us 15821.642 us    15.57      13.7%    0.006s  ok ch=6.3e-14/1e-10 1s=4e-16 1.00x
   d1_batchlane                  0.0243 us 24846.490 us     9.91      13.8%    0.000s  ok ch=2.5e-13/1e-10 1s=6e-16 1.58x
   fftw1d_custom_soa             0.0261 us 26762.336 us     9.20       0.5%    0.000s  ok ch=1.5e-14/1e-10 1s=3e-16 1.70x
   fftw1d_custom                 0.0474 us 48529.091 us     5.08       0.2%    0.000s  ok ch=1.5e-14/1e-10 1s=3e-16 3.08x
   fftw1d_measure                0.0510 us 52270.791 us     4.71       3.3%    0.002s  ok ch=8.3e-15/1e-10 1s=3e-16 3.31x
   fftw1d_patient                0.0512 us 52471.060 us     4.69       0.1%    0.002s  ok ch=8.3e-15/1e-10 1s=3e-16 3.33x
   mkl1d_dfti                    0.0530 us 54261.012 us     4.54       0.1%    0.054s  ok ch=7.2e-14/1e-10 1s=3e-16 3.44x
   fftw1d_estimate               0.0746 us 76418.889 us     3.22       1.6%    0.001s  ok ch=8.3e-15/1e-10 1s=3e-16 4.85x
   d1_rader                      0.1176 us 120449.082 us     2.04       1.7%    0.000s  ok ch=1.4e-13/1e-10 1s=5e-16 7.64x
   d1_bluestein                  0.1249 us 127935.661 us     1.93       0.0%    0.000s  ok ch=2.3e-13/1e-10 1s=7e-16 8.11x
   d1_planner                    0.1472 us 150730.004 us     1.63       0.1%    0.000s  ok ch=2.3e-13/1e-10 1s=7e-16 9.56x
   baseline_dft                  0.3270 us 334879.795 us     0.74       0.0%    0.000s  ok ch=2.3e-13/1e-10 1s=7e-16 21.24x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0527 us     0.053 us    14.57      22.1%    0.000s  ok 3.0e-16       1.00x
   d1_race                       0.0589 us     0.059 us    13.03       5.8%    0.004s  ok 3.0e-16       1.12x
   d1_batchlane                  0.0786 us     0.079 us     9.77      13.8%    0.000s  ok 3.4e-16       1.49x
   fftw1d_custom                 0.1781 us     0.178 us     4.31       0.3%    0.000s  ok 2.4e-16       3.38x
   fftw1d_estimate               0.2600 us     0.260 us     2.95      20.8%    0.001s  ok 1.7e-16       4.94x
   fftw1d_patient                0.2614 us     0.261 us     2.94       3.2%    0.117s  ok 1.7e-16       4.96x
   d1_bluestein                  0.2658 us     0.266 us     2.89      23.2%    0.000s  ok 5.1e-16       5.05x
   mkl1d_dfti                    0.2793 us     0.279 us     2.75       0.2%    0.049s  ok 2.2e-16       5.30x
   fftw1d_measure                0.3136 us     0.314 us     2.45       3.7%    0.042s  ok 1.7e-16       5.95x
   d1_rader                      0.3861 us     0.386 us     1.99       0.4%    0.000s  ok 4.0e-16       7.33x
   d1_planner                    0.5317 us     0.532 us     1.44       6.5%    0.000s  ok 3.7e-16       10.09x
   baseline_dft                  1.6553 us     1.655 us     0.46      20.7%    0.000s  ok 4.2e-16       31.42x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0582 us  5819.696 us    13.19       0.0%    0.006s  ok ch=5.7e-15/1e-10 1s=8e-16 1.00x
   d1_prime                      0.0582 us  5819.742 us    13.19      13.8%    0.000s  ok ch=5.7e-15/1e-10 1s=8e-16 1.00x
   d1_batchlane                  0.1374 us 13735.997 us     5.59       0.0%    0.000s  ok ch=2.0e-14/1e-10 1s=8e-16 2.36x
   fftw1d_custom                 0.2093 us 20928.427 us     3.67      20.3%    0.000s  ok ch=1.4e-15/1e-10 1s=6e-16 3.60x
   fftw1d_measure                0.3475 us 34749.442 us     2.21       3.9%    0.042s  ok ch=5.2e-15/1e-10 1s=4e-16 5.97x
   fftw1d_patient                0.3475 us 34751.026 us     2.21       0.6%    0.119s  ok ch=5.2e-15/1e-10 1s=4e-16 5.97x
   fftw1d_estimate               0.3499 us 34985.502 us     2.19      21.1%    0.001s  ok ch=5.2e-15/1e-10 1s=4e-16 6.01x
   d1_rader                      0.3501 us 35009.579 us     2.19       1.6%    0.000s  ok ch=2.4e-14/1e-10 1s=1e-15 6.02x
   d1_bluestein                  0.3855 us 38550.300 us     1.99       1.3%    0.000s  ok ch=1.5e-14/1e-10 1s=8e-16 6.62x
   mkl1d_dfti                    0.3856 us 38555.197 us     1.99       0.1%    0.031s  ok ch=7.9e-15/1e-10 1s=4e-16 6.62x
   d1_planner                    0.5121 us 51208.643 us     1.50      20.9%    0.000s  ok ch=1.1e-14/1e-10 1s=8e-16 8.80x
   baseline_dft                  1.7391 us 173908.115 us     0.44       0.0%    0.000s  ok ch=2.3e-14/1e-10 1s=8e-16 29.88x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_prime                      0.0552 us    28.287 us    13.90       4.1%    0.000s  ok 2.8e-16       1.00x
   d1_race                       0.0557 us    28.524 us    13.78      16.1%    0.007s  ok 2.8e-16       1.01x
   d1_batchlane                  0.0897 us    45.947 us     8.56      14.5%    0.000s  ok 3.7e-16       1.62x
   fftw1d_custom_soa             0.0904 us    46.297 us     8.49       3.1%    0.000s  ok 2.6e-16       1.64x
   fftw1d_custom                 0.1624 us    83.125 us     4.73       0.9%    0.000s  ok 2.6e-16       2.94x
   fftw1d_patient                0.2631 us   134.730 us     2.92       0.1%    0.118s  ok 2.1e-16       4.76x
   fftw1d_measure                0.2632 us   134.749 us     2.92       0.2%    0.043s  ok 2.1e-16       4.76x
   fftw1d_estimate               0.2633 us   134.804 us     2.92       0.3%    0.001s  ok 2.1e-16       4.77x
   d1_bluestein                  0.2724 us   139.491 us     2.82       0.7%    0.000s  ok 5.2e-16       4.93x
   mkl1d_dfti                    0.2736 us   140.082 us     2.81       0.2%    0.037s  ok 2.3e-16       4.95x
   d1_rader                      0.3907 us   200.062 us     1.97      11.3%    0.000s  ok 4.6e-16       7.07x
   d1_planner                    0.5483 us   280.730 us     1.40       0.1%    0.000s  ok 4.2e-16       9.92x
   baseline_dft                  1.6520 us   845.822 us     0.46      20.7%    0.000s  ok 4.2e-16       29.90x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0546 us 33555.621 us    14.06       1.2%    0.006s  ok ch=2.8e-12/1e-10 1s=7e-16 1.00x
   d1_prime                      0.0550 us 33789.476 us    13.96       0.5%    0.000s  ok ch=2.8e-12/1e-10 1s=7e-16 1.01x
   fftw1d_custom_soa             0.0771 us 47378.492 us     9.96      15.3%    0.000s  ok ch=8.1e-13/1e-10 1s=5e-16 1.41x
   d1_batchlane                  0.0995 us 61113.692 us     7.72      13.8%    0.000s  ok ch=1.1e-12/1e-10 1s=8e-16 1.82x
   fftw1d_custom                 0.1768 us 108639.202 us     4.34       0.1%    0.000s  ok ch=8.1e-13/1e-10 1s=5e-16 3.24x
   d1_bluestein                  0.3106 us 190855.025 us     2.47       1.5%    0.000s  ok ch=5.3e-12/1e-10 1s=1e-15 5.69x
   mkl1d_dfti                    0.3152 us 193686.249 us     2.44       0.1%    0.050s  ok ch=7.1e-13/1e-10 1s=5e-16 5.77x
   d1_rader                      0.3498 us 214928.898 us     2.20       5.2%    0.000s  ok ch=6.4e-12/1e-10 1s=1e-15 6.41x
   fftw1d_estimate               0.3533 us 217069.526 us     2.17       0.1%    0.001s  ok ch=1.1e-12/1e-10 1s=4e-16 6.47x
   fftw1d_measure                0.3534 us 217104.567 us     2.17       0.1%    0.042s  ok ch=1.1e-12/1e-10 1s=4e-16 6.47x
   fftw1d_patient                0.3535 us 217190.117 us     2.17       0.0%    0.118s  ok ch=1.1e-12/1e-10 1s=4e-16 6.47x
   d1_planner                    0.6347 us 389962.654 us     1.21       0.5%    0.000s  ok ch=7.1e-12/1e-10 1s=8e-16 11.62x
   baseline_dft                  1.7449 us 1072078.330 us     0.44       0.0%    0.000s  ok ch=7.1e-12/1e-10 1s=8e-16 31.95x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0182 us     0.018 us    44.04      16.7%    0.006s  ok 1.2e-16       1.00x
   d1_pow2                       0.0183 us     0.018 us    43.76      14.9%    0.000s  ok 1.2e-16       1.01x
   mkl1d_dfti                    0.0224 us     0.022 us    35.70      13.2%    0.001s  ok 1.5e-16       1.23x
   fftw1d_patient                0.0348 us     0.035 us    22.98      12.9%    0.015s  ok 1.4e-16       1.92x
   fftw1d_measure                0.0364 us     0.036 us    22.00       9.3%    0.008s  ok 1.6e-16       2.00x
   fftw1d_estimate               0.0442 us     0.044 us    18.10       2.8%    0.001s  ok 1.6e-16       2.43x
   d1_batchlane                  0.0662 us     0.066 us    12.08       0.0%    0.000s  ok 1.9e-16       3.65x
   d1_twiddle                    0.0765 us     0.076 us    10.46      21.8%    0.000s  ok 8.7e-17       4.21x
   d1_bluestein                  0.0826 us     0.083 us     9.69      22.5%    0.000s  ok 1.5e-16       4.55x
   fftw1d_custom                 0.0918 us     0.092 us     8.72      22.9%    0.000s  ok 1.4e-16       5.05x
   d1_planner                    0.1505 us     0.150 us     5.32       1.9%    0.000s  ok 1.4e-16       8.29x
   baseline_dft                  1.7634 us     1.763 us     0.45      20.8%    0.000s  ok 3.4e-16       97.09x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_pow2                       0.0852 us  8517.628 us     9.39      13.8%    0.000s  ok ch=6.1e-15/1e-10 1s=2e-16 1.00x
   d1_race                       0.0852 us  8517.681 us     9.39      13.8%    0.003s  ok ch=6.1e-15/1e-10 1s=2e-16 1.00x
   d1_batchlane                  0.1020 us 10203.860 us     7.84      14.0%    0.000s  ok ch=8.8e-15/1e-10 1s=3e-16 1.20x
   fftw1d_custom                 0.1214 us 12141.648 us     6.59       0.4%    0.000s  ok ch=5.5e-15/1e-10 1s=2e-16 1.43x
   mkl1d_dfti                    0.1309 us 13089.907 us     6.11       0.1%    0.001s  ok ch=3.8e-15/1e-10 1s=3e-16 1.54x
   fftw1d_patient                0.1331 us 13313.009 us     6.01      16.5%    0.015s  ok ch=6.4e-15/1e-10 1s=3e-16 1.56x
   fftw1d_measure                0.1361 us 13607.338 us     5.88      13.8%    0.008s  ok ch=4.2e-15/1e-10 1s=3e-16 1.60x
   fftw1d_estimate               0.1454 us 14544.707 us     5.50      14.2%    0.001s  ok ch=4.4e-15/1e-10 1s=3e-16 1.71x
   d1_twiddle                    0.1587 us 15866.101 us     5.04      21.6%    0.000s  ok ch=1.5e-15/1e-10 1s=2e-16 1.86x
   d1_bluestein                  0.1701 us 17007.402 us     4.70      27.3%    0.000s  ok ch=8.3e-15/1e-10 1s=3e-16 2.00x
   d1_planner                    0.2133 us 21330.492 us     3.75       0.9%    0.000s  ok ch=1.2e-14/1e-10 1s=4e-16 2.50x
   baseline_dft                  1.8469 us 184691.963 us     0.43       0.0%    0.000s  ok ch=2.8e-14/1e-10 1s=6e-16 21.68x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0153 us     7.830 us    52.31      13.8%    0.001s  ok 1.3e-16       1.00x
   d1_pow2                       0.0154 us     7.877 us    52.00      13.8%    0.000s  ok 1.4e-16       1.01x
   d1_race                       0.0154 us     7.887 us    51.93      13.9%    0.008s  ok 1.4e-16       1.01x
   fftw1d_patient                0.0250 us    12.825 us    31.94      18.9%    0.060s  ok 1.6e-16       1.64x
   fftw1d_measure                0.0251 us    12.830 us    31.92      14.3%    0.011s  ok 1.6e-16       1.64x
   d1_batchlane                  0.0316 us    16.187 us    25.31       0.4%    0.000s  ok 1.5e-16       2.07x
   d1_planner                    0.0740 us    37.874 us    10.81       1.1%    0.000s  ok 1.8e-16       4.84x
   d1_bluestein                  0.0756 us    38.722 us    10.58       0.6%    0.000s  ok 1.7e-16       4.95x
   fftw1d_custom_soa             0.0759 us    38.882 us    10.53      18.8%    0.000s  ok 1.4e-16       4.97x
   d1_twiddle                    0.0781 us    39.969 us    10.25      20.7%    0.000s  ok 9.7e-17       5.10x
   fftw1d_estimate               0.0888 us    45.476 us     9.01      14.9%    0.001s  ok 1.5e-16       5.81x
   fftw1d_custom                 0.0888 us    45.476 us     9.01      21.6%    0.000s  ok 1.4e-16       5.81x
   baseline_dft                  1.7597 us   900.979 us     0.45      20.7%    0.000s  ok 3.4e-16       115.07x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   d1_race                       0.0606 us 31052.562 us    13.19      18.4%    0.010s  ok ch=1.2e-12/1e-10 1s=4e-16 1.00x
   d1_batchlane                  0.0607 us 31070.955 us    13.18      13.8%    0.000s  ok ch=1.2e-12/1e-10 1s=4e-16 1.00x
   fftw1d_custom_soa             0.0665 us 34070.728 us    12.02       0.3%    0.000s  ok ch=8.0e-13/1e-10 1s=3e-16 1.10x
   d1_pow2                       0.0852 us 43606.472 us     9.39       0.0%    0.000s  ok ch=7.2e-13/1e-10 1s=3e-16 1.40x
   fftw1d_custom                 0.1039 us 53186.478 us     7.70       0.2%    0.000s  ok ch=8.0e-13/1e-10 1s=3e-16 1.71x
   mkl1d_dfti                    0.1122 us 57469.812 us     7.13       0.0%    0.001s  ok ch=6.2e-13/1e-10 1s=3e-16 1.85x
   fftw1d_patient                0.1219 us 62427.514 us     6.56      14.0%    0.061s  ok ch=1.3e-12/1e-10 1s=3e-16 2.01x
   fftw1d_measure                0.1220 us 62466.900 us     6.56      14.3%    0.010s  ok ch=1.3e-12/1e-10 1s=4e-16 2.01x
   d1_twiddle                    0.1577 us 80754.365 us     5.07       8.7%    0.000s  ok ch=5.3e-13/1e-10 1s=3e-16 2.60x
   d1_bluestein                  0.1662 us 85091.047 us     4.81       3.2%    0.000s  ok ch=1.5e-12/1e-10 1s=4e-16 2.74x
   d1_planner                    0.1707 us 87373.992 us     4.69       0.2%    0.000s  ok ch=2.0e-12/1e-10 1s=4e-16 2.81x
   fftw1d_estimate               0.1860 us 95257.271 us     4.30       0.3%    0.000s  ok ch=9.9e-13/1e-10 1s=3e-16 3.07x
   baseline_dft                  1.9531 us 999992.090 us     0.41       0.0%    0.000s  ok ch=3.1e-12/1e-10 1s=6e-16 32.20x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl1d_dfti                    0.0612 us     0.061 us    28.95       7.2%    0.002s  ok 2.4e-16       1.00x
   fftw1d_measure                0.0616 us     0.062 us    28.78      20.0%    0.046s  ok 2.4e-16       1.01x
   fftw1d_patient                0.0658 us     0.066 us    26.95      39.2%    0.128s  ok 1.9e-16       1.07x
   d1_composite                  0.0690 us     0.069 us    25.69       0.3%    0.000s  ok 2.4e-16       1.13x
