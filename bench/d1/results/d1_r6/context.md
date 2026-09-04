# What previous generations produced (round d1_r6 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r4/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r5/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 616 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 658 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 579 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 446 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 583 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 543 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 630 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 629 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 606 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/d1/exemplars/d1_r3/
      # Round d1_r3 — what it established
      
      Promoted: d1_prime d1_rader d1_pow2 d1_batchlane d1_composite d1_bluestein d1_twiddle d1_race
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/d1/exemplars/d1_r4/
      # Round d1_r4 — what it established
      
      Promoted: d1_prime d1_pow2 d1_rader d1_bluestein d1_composite d1_batchlane d1_twiddle d1_race
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/d1/exemplars/d1_r5/
      # Round d1_r5 — what it established
      
      Promoted: d1_prime d1_composite d1_pow2 d1_rader d1_bluestein d1_twiddle d1_batchlane d1_planner d1_race
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round d1_r5 ===
# round d1_r5
host: a80n0.lqcd.mit   date: 2026-09-03T13:49:50-04:00   slurm_job: 440424
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_planner                    0.0169 us      0.0148 us    14.26     31.9%   9r   0.000s  ok 1.3e-16       1.00x?
   d1_batchlane                  0.0180 us      0.0140 us    13.35     38.2%   9r   0.000s  ok 1.3e-16       1.07x?
   d1_race                       0.0189 us      0.0161 us    12.74     18.7%   9r   0.008s  ok 1.9e-16       1.12x
   d1_prime                      0.0202 us      0.0177 us    11.92     43.9%   9r   0.000s  ok 1.9e-16       1.20x
   mkl1d_dfti                    0.0259 us      0.0219 us     9.29     26.6%   9r   0.014s  ok 2.0e-16       1.53x
   fftw1d_measure                0.0264 us      0.0264 us     9.11      9.4%   9r   0.001s  ok 2.1e-16       1.57x
   fftw1d_estimate               0.0264 us      0.0263 us     9.10      1.3%   9r   0.001s  ok 2.1e-16       1.57x
   fftw1d_patient                0.0267 us      0.0264 us     9.02      7.3%   9r   0.001s  ok 2.1e-16       1.58x
   d1_rader                      0.0563 us      0.0562 us     4.27      4.0%   9r   0.000s  ok 2.7e-16       3.34x
   fftw1d_custom                 0.0571 us      0.0567 us     4.21      5.5%   9r   0.000s  ok 2.5e-16       3.39x
   d1_bluestein                  0.1107 us      0.0917 us     2.17     32.5%   9r   0.000s  ok 2.3e-16       6.56x
   baseline_dft                  0.2927 us      0.2927 us     0.82     20.7%   9r   0.000s  ok 4.5e-16       17.36x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0338 us      0.0338 us     7.11      0.0%   3r   0.007s  ok ch=1.1e-15/1e-10 1s=5e-16 1.00x?
   d1_prime                      0.0385 us      0.0385 us     6.25      0.0%   3r   0.000s  ok ch=1.1e-15/1e-10 1s=5e-16 1.14x
   d1_batchlane                  0.0392 us      0.0392 us     6.14     13.8%   3r   0.000s  ok ch=4.6e-16/1e-10 1s=3e-16 1.16x
   d1_planner                    0.0448 us      0.0448 us     5.37      0.1%   3r   0.000s  ok ch=5.4e-16/1e-10 1s=3e-16 1.33x
   fftw1d_custom                 0.0626 us      0.0624 us     3.84     21.1%   3r   0.000s  ok ch=1.2e-15/1e-10 1s=7e-16 1.85x
   mkl1d_dfti                    0.0666 us      0.0666 us     3.61      0.1%   3r   0.051s  ok ch=1.5e-15/1e-10 1s=5e-16 1.97x
   fftw1d_measure                0.0674 us      0.0674 us     3.57      0.0%   3r   0.001s  ok ch=8.8e-16/1e-10 1s=8e-16 1.99x
   fftw1d_estimate               0.0674 us      0.0674 us     3.57     20.7%   3r   0.000s  ok ch=8.8e-16/1e-10 1s=8e-16 1.99x
   fftw1d_patient                0.0814 us      0.0814 us     2.96      0.0%   3r   0.001s  ok ch=8.8e-16/1e-10 1s=8e-16 2.41x
   d1_rader                      0.0967 us      0.0801 us     2.49     20.7%   3r   0.000s  ok ch=1.2e-15/1e-10 1s=7e-16 2.86x
   d1_bluestein                  0.1217 us      0.1209 us     1.98     21.3%   3r   0.000s  ok ch=8.8e-16/1e-10 1s=4e-16 3.60x
   baseline_dft                  0.3930 us      0.3256 us     0.61     20.7%   3r   0.000s  ok ch=1.2e-14/1e-10 1s=7e-16 11.62x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0093 us      0.0093 us    25.93     14.3%   9r   0.000s  ok 1.7e-16       1.00x?
   d1_planner                    0.0094 us      0.0094 us    25.52     14.0%   9r   0.000s  ok 1.4e-16       1.02x?
   d1_batchlane                  0.0108 us      0.0095 us    22.31     14.5%   9r   0.000s  ok 1.4e-16       1.16x
   d1_race                       0.0109 us      0.0095 us    22.16     15.8%   9r   0.011s  ok 1.4e-16       1.17x
   fftw1d_patient                0.0141 us      0.0123 us    17.08     16.6%   9r   0.002s  ok 1.8e-16       1.52x
   fftw1d_measure                0.0141 us      0.0124 us    17.07     21.7%   9r   0.002s  ok 1.8e-16       1.52x
   mkl1d_dfti                    0.0160 us      0.0159 us    15.06     23.4%   9r   0.037s  ok 1.7e-16       1.72x
   d1_rader                      0.0165 us      0.0163 us    14.62     18.0%   9r   0.000s  ok 2.8e-16       1.77x
   fftw1d_custom_soa             0.0367 us      0.0324 us     6.56     13.7%   9r   0.000s  ok 1.8e-16       3.95x
   fftw1d_estimate               0.0401 us      0.0352 us     6.00     13.9%   9r   0.001s  ok 1.8e-16       4.32x
   fftw1d_custom                 0.0490 us      0.0405 us     4.91     22.0%   9r   0.000s  ok 1.8e-16       5.28x
   d1_bluestein                  0.0970 us      0.0865 us     2.48     24.7%   9r   0.000s  ok 1.8e-16       10.45x
   baseline_dft                  0.3499 us      0.3498 us     0.69      0.0%   9r   0.000s  ok 4.1e-16       37.71x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_planner                    0.0155 us      0.0155 us    15.55      0.2%   3r   0.000s  ok ch=2.2e-15/1e-10 1s=3e-16 1.00x?
   d1_prime                      0.0157 us      0.0139 us    15.35     13.1%   3r   0.000s  ok ch=2.2e-14/1e-10 1s=4e-16 1.01x?
   d1_race                       0.0157 us      0.0157 us    15.29      1.3%   3r   0.011s  ok ch=2.2e-14/1e-10 1s=4e-16 1.02x
   d1_rader                      0.0166 us      0.0166 us    14.47      0.0%   3r   0.000s  ok ch=3.5e-14/1e-10 1s=5e-16 1.07x
   d1_batchlane                  0.0175 us      0.0155 us    13.76     13.8%   3r   0.000s  ok ch=7.5e-15/1e-10 1s=3e-16 1.13x
   fftw1d_custom_soa             0.0267 us      0.0262 us     9.02     14.3%   3r   0.000s  ok ch=1.0e-14/1e-10 1s=3e-16 1.72x
   fftw1d_patient                0.0512 us      0.0512 us     4.69      0.8%   3r   0.002s  ok ch=1.1e-14/1e-10 1s=3e-16 3.31x
   fftw1d_measure                0.0514 us      0.0513 us     4.68     14.7%   3r   0.002s  ok ch=1.1e-14/1e-10 1s=3e-16 3.32x
   mkl1d_dfti                    0.0531 us      0.0530 us     4.53     23.4%   3r   0.052s  ok ch=1.9e-14/1e-10 1s=3e-16 3.43x
   fftw1d_custom                 0.0571 us      0.0477 us     4.21     21.0%   3r   0.000s  ok ch=1.0e-14/1e-10 1s=3e-16 3.69x
   fftw1d_estimate               0.0849 us      0.0746 us     2.83     14.2%   3r   0.001s  ok ch=1.1e-14/1e-10 1s=3e-16 5.49x
   d1_bluestein                  0.1238 us      0.1236 us     1.94      0.3%   3r   0.000s  ok ch=6.8e-15/1e-10 1s=3e-16 8.00x
   baseline_dft                  0.3270 us      0.3270 us     0.74      0.0%   3r   0.000s  ok ch=9.3e-14/1e-10 1s=8e-16 21.14x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0536 us      0.0465 us    14.34     26.6%   9r   0.009s  ok 2.9e-16       1.00x?
   d1_batchlane                  0.0540 us      0.0530 us    14.21      9.3%   9r   0.000s  ok 1.9e-16       1.01x?
   d1_prime                      0.0544 us      0.0471 us    14.12     22.8%   9r   0.000s  ok 2.9e-16       1.02x?
   d1_planner                    0.0588 us      0.0575 us    13.06      7.6%   9r   0.000s  ok 1.9e-16       1.10x
   d1_bluestein                  0.2099 us      0.2094 us     3.66     10.8%   9r   0.001s  ok 3.5e-16       3.92x
   d1_rader                      0.2112 us      0.2099 us     3.64     11.7%   9r   0.000s  ok 3.1e-16       3.94x
   fftw1d_custom                 0.2149 us      0.1777 us     3.57     22.3%   9r   0.000s  ok 3.1e-16       4.01x
   mkl1d_dfti                    0.2316 us      0.2315 us     3.32     20.7%   9r   0.013s  ok 2.2e-16       4.33x
   fftw1d_estimate               0.3141 us      0.3135 us     2.44     22.8%   9r   0.001s  ok 1.9e-16       5.87x
   fftw1d_measure                0.3153 us      0.2601 us     2.44     27.0%   9r   0.042s  ok 1.9e-16       5.89x
   fftw1d_patient                0.3172 us      0.2619 us     2.42     47.0%   9r   0.119s  ok 1.9e-16       5.92x
   baseline_dft                  1.9979 us      1.9978 us     0.38      0.0%   9r   0.000s  ok 3.8e-16       37.30x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0510 us      0.0510 us    15.05      0.2%   3r   0.005s  ok ch=1.7e-15/1e-10 1s=6e-16 1.00x?
   d1_prime                      0.0511 us      0.0510 us    15.02     13.8%   3r   0.000s  ok ch=1.7e-15/1e-10 1s=6e-16 1.00x?
   d1_planner                    0.0592 us      0.0582 us    12.96      3.6%   3r   0.000s  ok ch=9.4e-16/1e-10 1s=4e-16 1.16x
   d1_batchlane                  0.0621 us      0.0619 us    12.36      0.4%   3r   0.000s  ok ch=1.1e-15/1e-10 1s=4e-16 1.22x
   fftw1d_custom                 0.2090 us      0.2088 us     3.67     20.8%   3r   0.000s  ok ch=1.5e-15/1e-10 1s=5e-16 4.10x
   d1_rader                      0.2352 us      0.2087 us     3.27     12.7%   3r   0.000s  ok ch=6.9e-15/1e-10 1s=5e-16 4.61x
   d1_bluestein                  0.2403 us      0.2151 us     3.20     15.2%   3r   0.002s  ok ch=4.5e-15/1e-10 1s=6e-16 4.71x
   fftw1d_measure                0.3483 us      0.3475 us     2.20     21.8%   3r   0.043s  ok ch=9.6e-16/1e-10 1s=3e-16 6.83x
   fftw1d_patient                0.3575 us      0.3476 us     2.15     20.7%   3r   0.120s  ok ch=9.6e-16/1e-10 1s=3e-16 7.01x
   fftw1d_estimate               0.3586 us      0.3499 us     2.14     20.7%   3r   0.001s  ok ch=9.6e-16/1e-10 1s=3e-16 7.03x
   mkl1d_dfti                    0.3856 us      0.3841 us     1.99      0.4%   3r   0.036s  ok ch=7.9e-16/1e-10 1s=4e-16 7.56x
   baseline_dft                  1.7391 us      1.7390 us     0.44      0.0%   3r   0.000s  ok ch=1.1e-14/1e-10 1s=7e-16 34.09x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0484 us      0.0419 us    15.88     16.6%   9r   0.000s  ok 2.9e-16       1.00x?
   d1_race                       0.0484 us      0.0422 us    15.87     16.0%   9r   0.009s  ok 2.9e-16       1.00x?
   d1_planner                    0.0502 us      0.0502 us    15.29      0.0%   9r   0.000s  ok 2.1e-16       1.04x
   d1_batchlane                  0.0502 us      0.0502 us    15.29      0.1%   9r   0.000s  ok 2.1e-16       1.04x
   d1_rader                      0.0634 us      0.0626 us    12.12     15.1%   9r   0.000s  ok 3.6e-16       1.31x
   fftw1d_custom_soa             0.1014 us      0.0893 us     7.57     36.8%   9r   0.000s  ok 2.6e-16       2.10x
   fftw1d_custom                 0.1976 us      0.1948 us     3.89      3.8%   9r   0.000s  ok 2.6e-16       4.09x
   d1_bluestein                  0.2088 us      0.2083 us     3.68      3.6%   9r   0.001s  ok 3.2e-16       4.32x
   mkl1d_dfti                    0.2268 us      0.2264 us     3.39     21.0%   9r   0.033s  ok 2.3e-16       4.69x
   fftw1d_estimate               0.3177 us      0.2632 us     2.42     21.9%   9r   0.001s  ok 2.1e-16       6.57x
   fftw1d_patient                0.3180 us      0.2653 us     2.42     20.2%   9r   0.119s  ok 2.1e-16       6.57x
   fftw1d_measure                0.3180 us      0.3176 us     2.41     22.0%   9r   0.043s  ok 2.1e-16       6.58x
   baseline_dft                  1.9936 us      1.9936 us     0.39      0.0%   9r   0.000s  ok 4.2e-16       41.22x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0459 us      0.0452 us    16.73      1.9%   3r   0.006s  ok ch=1.3e-12/1e-10 1s=7e-16 1.00x?
   d1_prime                      0.0511 us      0.0509 us    15.02      0.6%   3r   0.000s  ok ch=1.3e-12/1e-10 1s=7e-16 1.11x
   d1_planner                    0.0549 us      0.0485 us    13.99     13.8%   3r   0.000s  ok ch=5.6e-13/1e-10 1s=5e-16 1.20x
   d1_batchlane                  0.0553 us      0.0552 us    13.90      0.7%   3r   0.000s  ok ch=5.6e-13/1e-10 1s=4e-16 1.20x
   d1_rader                      0.0553 us      0.0553 us    13.88      0.1%   3r   0.000s  ok ch=3.2e-12/1e-10 1s=8e-16 1.21x
   fftw1d_custom_soa             0.0776 us      0.0776 us     9.89      0.0%   3r   0.000s  ok ch=1.3e-12/1e-10 1s=5e-16 1.69x
   fftw1d_custom                 0.1771 us      0.1767 us     4.34      0.4%   3r   0.000s  ok ch=1.3e-12/1e-10 1s=5e-16 3.86x
   d1_bluestein                  0.2168 us      0.2130 us     3.54      2.9%   3r   0.001s  ok ch=2.4e-12/1e-10 1s=7e-16 4.72x
   mkl1d_dfti                    0.3153 us      0.3153 us     2.44      0.0%   3r   0.050s  ok ch=8.4e-13/1e-10 1s=5e-16 6.87x
   fftw1d_measure                0.3535 us      0.3535 us     2.17      0.0%   3r   0.041s  ok ch=2.4e-13/1e-10 1s=4e-16 7.70x
   fftw1d_estimate               0.3536 us      0.3534 us     2.17      1.0%   3r   0.001s  ok ch=2.4e-13/1e-10 1s=4e-16 7.70x
   fftw1d_patient                0.3536 us      0.3536 us     2.17      0.1%   3r   0.119s  ok ch=2.4e-13/1e-10 1s=4e-16 7.70x
   baseline_dft                  1.7449 us      1.7449 us     0.44      0.0%   3r   0.000s  ok ch=3.8e-12/1e-10 1s=8e-16 38.02x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0192 us      0.0172 us    41.57     32.8%   9r   0.015s  ok 1.7e-16       1.00x?
   d1_twiddle                    0.0194 us      0.0173 us    41.22     18.7%   9r   0.000s  ok 1.7e-16       1.01x?
   d1_planner                    0.0209 us      0.0186 us    38.27     43.6%   9r   0.000s  ok 1.3e-16       1.09x?
   d1_pow2                       0.0211 us      0.0185 us    37.94     45.2%   9r   0.001s  ok 1.3e-16       1.10x?
   d1_batchlane                  0.0222 us      0.0214 us    36.05     22.2%   9r   0.000s  ok 1.3e-16       1.15x
   mkl1d_dfti                    0.0255 us      0.0223 us    31.33     62.4%   9r   0.001s  ok 1.3e-16       1.33x
   fftw1d_patient                0.0371 us      0.0346 us    21.59     34.5%   9r   0.014s  ok 1.7e-16       1.93x
   fftw1d_measure                0.0380 us      0.0325 us    21.05     27.5%   9r   0.008s  ok 1.5e-16       1.97x
   fftw1d_estimate               0.0474 us      0.0415 us    16.89     29.4%   9r   0.001s  ok 1.5e-16       2.46x
   d1_bluestein                  0.1006 us      0.0972 us     7.95     14.8%   9r   0.001s  ok 1.4e-16       5.23x
   fftw1d_custom                 0.1092 us      0.0912 us     7.33     22.2%   9r   0.000s  ok 1.7e-16       5.67x
   baseline_dft                  2.1283 us      2.1283 us     0.38      0.1%   9r   0.000s  ok 3.1e-16       110.59x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_batchlane                  0.0599 us      0.0599 us    13.36      0.4%   3r   0.000s  ok ch=7.7e-15/1e-10 1s=3e-16 1.00x?
   d1_race                       0.0700 us      0.0699 us    11.43      0.1%   3r   0.009s  ok ch=7.7e-15/1e-10 1s=3e-16 1.17x
   d1_planner                    0.0791 us      0.0791 us    10.11      0.0%   3r   0.000s  ok ch=9.7e-15/1e-10 1s=4e-16 1.32x
   d1_pow2                       0.0804 us      0.0706 us     9.95     13.8%   3r   0.001s  ok ch=9.7e-15/1e-10 1s=4e-16 1.34x
   d1_twiddle                    0.0804 us      0.0804 us     9.95      0.0%   3r   0.000s  ok ch=9.2e-16/1e-10 1s=4e-16 1.34x
   mkl1d_dfti                    0.1310 us      0.1310 us     6.11      0.0%   3r   0.002s  ok ch=8.6e-15/1e-10 1s=3e-16 2.19x
   fftw1d_patient                0.1412 us      0.1360 us     5.67     13.7%   3r   0.016s  ok ch=1.7e-14/1e-10 1s=4e-16 2.36x
   fftw1d_estimate               0.1459 us      0.1454 us     5.48     13.7%   3r   0.001s  ok ch=1.7e-14/1e-10 1s=3e-16 2.44x
   fftw1d_custom                 0.1467 us      0.1466 us     5.45      0.1%   3r   0.000s  ok ch=7.6e-15/1e-10 1s=4e-16 2.45x
   fftw1d_measure                0.1548 us      0.1360 us     5.17     13.8%   3r   0.007s  ok ch=1.7e-14/1e-10 1s=4e-16 2.58x
   d1_bluestein                  0.1911 us      0.1886 us     4.19      1.5%   3r   0.001s  ok ch=9.5e-15/1e-10 1s=3e-16 3.19x
   baseline_dft                  1.8469 us      1.8469 us     0.43      0.0%   3r   0.000s  ok ch=7.1e-14/1e-10 1s=7e-16 30.84x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_planner                    0.0154 us      0.0154 us    52.04     13.9%   9r   0.000s  ok 1.4e-16       1.00x?
   d1_batchlane                  0.0155 us      0.0155 us    51.67     13.9%   9r   0.000s  ok 1.4e-16       1.01x?
   d1_twiddle                    0.0173 us      0.0152 us    46.30     14.6%   9r   0.000s  ok 1.4e-16       1.12x
   mkl1d_dfti                    0.0174 us      0.0174 us    45.97      0.1%   9r   0.002s  ok 1.3e-16       1.13x
   d1_pow2                       0.0176 us      0.0176 us    45.57      0.1%   9r   0.001s  ok 1.4e-16       1.14x
   d1_race                       0.0176 us      0.0154 us    45.53     13.8%   9r   0.014s  ok 1.4e-16       1.14x
   fftw1d_patient                0.0268 us      0.0250 us    29.82     19.7%   9r   0.064s  ok 1.6e-16       1.75x
   fftw1d_measure                0.0290 us      0.0257 us    27.57     17.0%   9r   0.011s  ok 1.6e-16       1.89x
   fftw1d_custom_soa             0.0879 us      0.0770 us     9.11     24.4%   9r   0.000s  ok 1.4e-16       5.72x
   d1_bluestein                  0.0931 us      0.0835 us     8.60     25.8%   9r   0.001s  ok 1.4e-16       6.05x
   fftw1d_estimate               0.1008 us      0.0883 us     7.94     15.3%   9r   0.001s  ok 1.5e-16       6.56x
   fftw1d_custom                 0.1079 us      0.0889 us     7.42     27.7%   9r   0.000s  ok 1.4e-16       7.02x
   baseline_dft                  2.1240 us      1.7598 us     0.38     20.7%   9r   0.000s  ok 3.4e-16       138.17x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_twiddle                    0.0368 us      0.0366 us    21.74     14.0%   3r   0.000s  ok ch=2.7e-13/1e-10 1s=4e-16 1.00x?
   d1_pow2                       0.0387 us      0.0387 us    20.68      0.5%   3r   0.001s  ok ch=3.8e-13/1e-10 1s=4e-16 1.05x?
   d1_batchlane                  0.0391 us      0.0389 us    20.44      1.1%   3r   0.000s  ok ch=3.0e-13/1e-10 1s=3e-16 1.06x?
   d1_race                       0.0392 us      0.0391 us    20.42      0.2%   3r   0.008s  ok ch=3.0e-13/1e-10 1s=3e-16 1.07x?
   fftw1d_custom_soa             0.0665 us      0.0664 us    12.03      0.2%   3r   0.000s  ok ch=3.0e-13/1e-10 1s=3e-16 1.81x
   d1_planner                    0.0775 us      0.0774 us    10.32      0.2%   3r   0.000s  ok ch=4.1e-13/1e-10 1s=3e-16 2.11x
   fftw1d_custom                 0.1040 us      0.1039 us     7.69      0.1%   3r   0.000s  ok ch=3.0e-13/1e-10 1s=3e-16 2.83x
   mkl1d_dfti                    0.1123 us      0.1122 us     7.13     15.2%   3r   0.001s  ok ch=2.0e-13/1e-10 1s=3e-16 3.05x
   fftw1d_measure                0.1225 us      0.1223 us     6.53     14.4%   3r   0.010s  ok ch=3.6e-13/1e-10 1s=3e-16 3.33x
   fftw1d_patient                0.1239 us      0.1224 us     6.46     13.1%   3r   0.061s  ok ch=3.4e-13/1e-10 1s=4e-16 3.37x
   d1_bluestein                  0.1803 us      0.1791 us     4.44      0.7%   3r   0.001s  ok ch=3.9e-13/1e-10 1s=3e-16 4.90x
   fftw1d_estimate               0.1861 us      0.1858 us     4.30      0.4%   3r   0.001s  ok ch=3.6e-13/1e-10 1s=3e-16 5.06x
   baseline_dft                  1.9513 us      1.9509 us     0.41      0.0%   3r   0.000s  ok ch=1.1e-12/1e-10 1s=6e-16 53.03x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_composite                  0.0429 us      0.0413 us    41.27     19.4%   9r   0.000s  ok 2.1e-16       1.00x?
   d1_race                       0.0433 us      0.0415 us    40.91     13.8%   9r   0.006s  ok 2.1e-16       1.01x?
   d1_planner                    0.0473 us      0.0446 us    37.46     18.5%   9r   0.000s  ok 2.1e-16       1.10x
   d1_batchlane                  0.0511 us      0.0449 us    34.68     22.2%   9r   0.000s  ok 2.1e-16       1.19x
