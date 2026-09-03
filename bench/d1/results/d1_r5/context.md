# What previous generations produced (round d1_r5 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r4/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 496 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 531 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 467 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 321 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 468 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 448 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 480 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 510 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 476 lines

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
      

## Current standings (most recent leaderboard)
=== round d1_r4 ===
# round d1_r4
host: a80n0.lqcd.mit   date: 2026-09-03T08:56:37-04:00   slurm_job: 440424
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512_vbmi avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0148 us      0.0145 us    16.26      2.5%   3r  14.337s  ok 1.6e-16       1.00x?
   d1_planner                    0.0155 us      0.0148 us    15.52    133.7%   3r   0.000s  ok 1.4e-16       1.05x?
   d1_prime                      0.0181 us      0.0181 us    13.28      0.4%   3r   0.000s  ok 1.6e-16       1.22x
   d1_batchlane                  0.0192 us      0.0148 us    12.52     38.0%   3r   0.000s  ok 1.4e-16       1.30x
   mkl1d_dfti                    0.0260 us      0.0259 us     9.27      2.6%   3r   0.032s  ok 1.7e-16       1.75x
   fftw1d_measure                0.0264 us      0.0221 us     9.12     26.3%   3r   0.001s  ok 2.1e-16       1.78x
   fftw1d_estimate               0.0266 us      0.0266 us     9.03      1.3%   3r   0.001s  ok 2.1e-16       1.80x
   fftw1d_patient                0.0266 us      0.0264 us     9.03      5.5%   3r   0.001s  ok 2.1e-16       1.80x
   fftw1d_custom                 0.0470 us      0.0470 us     5.11      0.2%   3r   0.000s  ok 2.3e-16       3.18x
   d1_rader                      0.0471 us      0.0471 us     5.11     20.7%   3r   0.000s  ok 3.2e-16       3.18x
   d1_bluestein                  0.0890 us      0.0888 us     2.70     14.3%   3r   0.000s  ok 1.4e-16       6.02x
   baseline_dft                  0.3533 us      0.2927 us     0.68     20.7%   3r   0.000s  ok 4.0e-16       23.88x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_planner                    0.0394 us      0.0393 us     6.10     13.9%   3r   0.000s  ok ch=2.3e-15/1e-10 1s=5e-16 1.00x?
   d1_prime                      0.0422 us      0.0421 us     5.70      0.5%   3r   0.000s  ok ch=1.6e-15/1e-10 1s=5e-16 1.07x
   d1_race                       0.0425 us      0.0423 us     5.66      0.4%   3r   0.005s  ok ch=1.6e-15/1e-10 1s=5e-16 1.08x
   d1_batchlane                  0.0451 us      0.0451 us     5.33      0.1%   3r   0.000s  ok ch=2.5e-15/1e-10 1s=5e-16 1.14x
   fftw1d_measure                0.0674 us      0.0674 us     3.57      0.0%   3r   0.001s  ok ch=2.3e-15/1e-10 1s=8e-16 1.71x
   fftw1d_estimate               0.0674 us      0.0674 us     3.57      0.0%   3r   0.001s  ok ch=2.3e-15/1e-10 1s=8e-16 1.71x
   fftw1d_custom                 0.0756 us      0.0756 us     3.18      0.8%   3r   0.000s  ok ch=1.7e-15/1e-10 1s=9e-16 1.92x
   d1_rader                      0.0801 us      0.0801 us     3.00     20.7%   3r   0.000s  ok ch=3.4e-15/1e-10 1s=1e-15 2.03x
   mkl1d_dfti                    0.0803 us      0.0803 us     2.99      0.1%   3r   0.034s  ok ch=1.8e-15/1e-10 1s=4e-16 2.04x
   fftw1d_patient                0.0814 us      0.0814 us     2.96      0.0%   3r   0.001s  ok ch=2.3e-15/1e-10 1s=8e-16 2.06x
   d1_bluestein                  0.1205 us      0.1201 us     2.00     21.9%   3r   0.000s  ok ch=2.1e-15/1e-10 1s=5e-16 3.06x
   baseline_dft                  0.3930 us      0.3256 us     0.61     20.7%   3r   0.000s  ok ch=8.9e-15/1e-10 1s=1e-15 9.97x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0106 us      0.0095 us    22.62     12.1%   3r   0.000s  ok 1.7e-16       1.00x?
   d1_planner                    0.0107 us      0.0107 us    22.54      0.1%   3r   0.000s  ok 1.4e-16       1.00x?
   d1_batchlane                  0.0107 us      0.0107 us    22.38      1.8%   3r   0.000s  ok 1.4e-16       1.01x?
   d1_race                       0.0108 us      0.0107 us    22.29      0.8%   3r  12.198s  ok 1.4e-16       1.01x?
   fftw1d_patient                0.0140 us      0.0139 us    17.14      2.2%   3r   0.003s  ok 1.8e-16       1.32x
   fftw1d_measure                0.0141 us      0.0140 us    17.02      4.0%   3r   0.002s  ok 1.8e-16       1.33x
   d1_rader                      0.0188 us      0.0179 us    12.82      5.5%   3r   0.000s  ok 2.8e-16       1.76x
   mkl1d_dfti                    0.0193 us      0.0193 us    12.46      0.2%   3r   0.032s  ok 1.7e-16       1.82x
   fftw1d_custom_soa             0.0369 us      0.0368 us     6.52      0.4%   3r   0.000s  ok 1.8e-16       3.47x
   fftw1d_estimate               0.0400 us      0.0398 us     6.02      0.6%   3r   0.001s  ok 1.8e-16       3.76x
   fftw1d_custom                 0.0489 us      0.0488 us     4.92      0.3%   3r   0.000s  ok 1.8e-16       4.60x
   d1_bluestein                  0.1038 us      0.0860 us     2.32     38.8%   3r   0.000s  ok 1.8e-16       9.76x
   baseline_dft                  0.3499 us      0.3499 us     0.69      0.0%   3r   0.000s  ok 4.1e-16       32.90x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0146 us      0.0146 us    16.46      0.0%   3r   0.006s  ok ch=4.1e-13/1e-10 1s=5e-16 1.00x?
   d1_rader                      0.0146 us      0.0146 us    16.43      0.1%   3r   0.000s  ok ch=4.1e-13/1e-10 1s=5e-16 1.00x
   d1_planner                    0.0155 us      0.0155 us    15.56     13.9%   3r   0.000s  ok ch=1.6e-14/1e-10 1s=3e-16 1.06x?
   d1_prime                      0.0171 us      0.0171 us    14.04      1.5%   3r   0.000s  ok ch=1.5e-13/1e-10 1s=4e-16 1.17x
   d1_batchlane                  0.0176 us      0.0176 us    13.64      0.2%   3r   0.000s  ok ch=1.3e-14/1e-10 1s=3e-16 1.21x
   fftw1d_custom_soa             0.0299 us      0.0298 us     8.05      1.2%   3r   0.000s  ok ch=4.2e-14/1e-10 1s=3e-16 2.05x
   fftw1d_custom                 0.0476 us      0.0474 us     5.05     21.1%   3r   0.000s  ok ch=4.2e-14/1e-10 1s=3e-16 3.26x
   fftw1d_measure                0.0514 us      0.0511 us     4.68      1.3%   3r   0.002s  ok ch=3.4e-14/1e-10 1s=3e-16 3.51x
   fftw1d_patient                0.0515 us      0.0513 us     4.67     14.3%   3r   0.003s  ok ch=3.4e-14/1e-10 1s=3e-16 3.52x
   mkl1d_dfti                    0.0530 us      0.0530 us     4.54      0.1%   3r   0.050s  ok ch=1.4e-13/1e-10 1s=3e-16 3.63x
   fftw1d_estimate               0.0746 us      0.0745 us     3.22     11.3%   3r   0.001s  ok ch=3.4e-14/1e-10 1s=3e-16 5.11x
   d1_bluestein                  0.1231 us      0.1229 us     1.95      1.3%   3r   0.000s  ok ch=1.2e-14/1e-10 1s=3e-16 8.42x
   baseline_dft                  0.3270 us      0.3270 us     0.74      0.0%   3r   0.000s  ok ch=2.6e-13/1e-10 1s=8e-16 22.38x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0471 us      0.0470 us    16.29     22.3%   3r   0.000s  ok 3.3e-16       1.00x?
   d1_race                       0.0525 us      0.0473 us    14.62     13.7%   3r   0.065s  ok 2.5e-16       1.11x
   d1_planner                    0.0528 us      0.0502 us    14.55     13.8%   3r   0.000s  ok 2.5e-16       1.12x
   d1_batchlane                  0.0530 us      0.0464 us    14.50     16.4%   3r   0.000s  ok 2.5e-16       1.12x
   d1_bluestein                  0.2067 us      0.1824 us     3.72     14.2%   3r   0.001s  ok 3.4e-16       4.38x
   d1_rader                      0.2112 us      0.1750 us     3.64     22.7%   3r   0.000s  ok 3.2e-16       4.48x
   fftw1d_custom                 0.2158 us      0.1803 us     3.56     20.2%   3r   0.000s  ok 3.2e-16       4.58x
   fftw1d_measure                0.2680 us      0.2649 us     2.87     26.5%   3r   0.042s  ok 2.0e-16       5.69x
   mkl1d_dfti                    0.2794 us      0.2793 us     2.75      0.1%   3r   0.031s  ok 2.1e-16       5.93x
   fftw1d_estimate               0.3133 us      0.3131 us     2.45      0.3%   3r   0.001s  ok 2.0e-16       6.65x
   fftw1d_patient                0.3144 us      0.3144 us     2.44      0.5%   3r   0.120s  ok 2.0e-16       6.67x
   baseline_dft                  1.9979 us      1.9979 us     0.38      0.0%   3r   0.000s  ok 3.6e-16       42.39x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0534 us      0.0533 us    14.39     13.9%   3r   0.000s  ok ch=4.2e-15/1e-10 1s=8e-16 1.00x?
   d1_batchlane                  0.0549 us      0.0548 us    14.00      0.1%   3r   0.000s  ok ch=2.9e-15/1e-10 1s=5e-16 1.03x?
   d1_planner                    0.0582 us      0.0582 us    13.19      0.0%   3r   0.000s  ok ch=2.4e-15/1e-10 1s=5e-16 1.09x
   d1_race                       0.0604 us      0.0534 us    12.70     13.4%   3r   0.008s  ok ch=4.2e-15/1e-10 1s=8e-16 1.13x
   d1_rader                      0.2064 us      0.2063 us     3.72      0.1%   3r   0.000s  ok ch=1.1e-14/1e-10 1s=7e-16 3.87x
   fftw1d_custom                 0.2095 us      0.2084 us     3.67      0.6%   3r   0.000s  ok ch=3.5e-15/1e-10 1s=5e-16 3.93x
   d1_bluestein                  0.2149 us      0.2082 us     3.57      4.7%   3r   0.001s  ok ch=2.6e-15/1e-10 1s=8e-16 4.03x
   fftw1d_patient                0.3489 us      0.3477 us     2.20      0.5%   3r   0.119s  ok ch=8.1e-16/1e-10 1s=4e-16 6.54x
   fftw1d_estimate               0.3498 us      0.3497 us     2.20      0.1%   3r   0.001s  ok ch=8.1e-16/1e-10 1s=4e-16 6.55x
   mkl1d_dfti                    0.3855 us      0.3848 us     1.99      0.2%   3r   0.049s  ok ch=1.8e-15/1e-10 1s=5e-16 7.22x
   fftw1d_measure                0.4132 us      0.3497 us     1.86     20.2%   3r   0.043s  ok ch=8.1e-16/1e-10 1s=4e-16 7.74x
   baseline_dft                  1.7390 us      1.7390 us     0.44      0.0%   3r   0.000s  ok ch=1.8e-14/1e-10 1s=7e-16 32.59x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0425 us      0.0420 us    18.07      1.7%   3r   1.026s  ok 2.9e-16       1.00x?
   d1_prime                      0.0484 us      0.0484 us    15.86      0.3%   3r   0.000s  ok 2.9e-16       1.14x
   d1_planner                    0.0502 us      0.0502 us    15.29      0.0%   3r   0.000s  ok 2.1e-16       1.18x
   d1_batchlane                  0.0502 us      0.0502 us    15.29      0.0%   3r   0.000s  ok 2.1e-16       1.18x
   d1_rader                      0.0719 us      0.0717 us    10.68      0.4%   3r   0.000s  ok 3.6e-16       1.69x
   fftw1d_custom_soa             0.1054 us      0.0915 us     7.29     21.4%   3r   0.000s  ok 2.6e-16       2.48x
   d1_bluestein                  0.1840 us      0.1826 us     4.17     15.3%   3r   0.001s  ok 3.3e-16       4.33x
   fftw1d_custom                 0.1989 us      0.1989 us     3.86      0.5%   3r   0.000s  ok 2.6e-16       4.68x
   fftw1d_measure                0.2633 us      0.2633 us     2.92     20.6%   3r   0.041s  ok 2.1e-16       6.20x
   fftw1d_estimate               0.2641 us      0.2641 us     2.91     20.7%   3r   0.001s  ok 2.1e-16       6.21x
   mkl1d_dfti                    0.2735 us      0.2271 us     2.81     20.6%   3r   0.033s  ok 2.3e-16       6.44x
   fftw1d_patient                0.3178 us      0.2635 us     2.42     21.0%   3r   0.119s  ok 2.1e-16       7.48x
   baseline_dft                  1.6518 us      1.6518 us     0.46      0.0%   3r   0.000s  ok 4.2e-16       38.87x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0465 us      0.0462 us    16.50      4.7%   3r   0.006s  ok ch=5.1e-12/3e-10 1s=7e-16 1.00x?
   d1_prime                      0.0476 us      0.0467 us    16.15      3.8%   3r   0.000s  ok ch=5.1e-12/3e-10 1s=7e-16 1.02x?
   d1_planner                    0.0552 us      0.0547 us    13.90      1.2%   3r   0.000s  ok ch=1.7e-12/3e-10 1s=5e-16 1.19x
   d1_rader                      0.0555 us      0.0553 us    13.83     16.0%   3r   0.000s  ok ch=2.7e-12/3e-10 1s=8e-16 1.19x
   d1_batchlane                  0.0560 us      0.0553 us    13.72      1.7%   3r   0.000s  ok ch=1.3e-12/3e-10 1s=4e-16 1.20x
   fftw1d_custom_soa             0.0780 us      0.0772 us     9.85      1.1%   3r   0.000s  ok ch=1.1e-12/3e-10 1s=5e-16 1.68x
   fftw1d_custom                 0.1770 us      0.1769 us     4.34      0.2%   3r   0.000s  ok ch=1.1e-12/3e-10 1s=5e-16 3.80x
   d1_bluestein                  0.2266 us      0.2101 us     3.39      8.4%   3r   0.001s  ok ch=4.0e-12/3e-10 1s=7e-16 4.87x
   mkl1d_dfti                    0.3153 us      0.3153 us     2.44      0.0%   3r   0.049s  ok ch=3.9e-12/3e-10 1s=5e-16 6.78x
   fftw1d_patient                0.3536 us      0.3534 us     2.17      0.1%   3r   0.121s  ok ch=2.6e-13/3e-10 1s=4e-16 7.60x
   fftw1d_measure                0.3536 us      0.3535 us     2.17      0.0%   3r   0.043s  ok ch=2.6e-13/3e-10 1s=4e-16 7.60x
   fftw1d_estimate               0.3537 us      0.3533 us     2.17      0.1%   3r   0.001s  ok ch=2.6e-13/3e-10 1s=4e-16 7.60x
   baseline_dft                  1.7450 us      1.7449 us     0.44      0.0%   3r   0.000s  ok ch=2.1e-12/3e-10 1s=8e-16 37.50x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_twiddle                    0.0192 us      0.0192 us    41.64     13.4%   3r   0.000s  ok 1.3e-16       1.00x?
   d1_race                       0.0207 us      0.0191 us    38.65     16.2%   3r   0.011s  ok 1.3e-16       1.08x?
   d1_pow2                       0.0208 us      0.0208 us    38.42      0.3%   3r   0.000s  ok 1.3e-16       1.08x
   d1_batchlane                  0.0218 us      0.0218 us    36.68     28.4%   3r   0.000s  ok 1.3e-16       1.14x?
   mkl1d_dfti                    0.0257 us      0.0254 us    31.15      7.7%   3r   0.003s  ok 8.6e-17       1.34x
   d1_planner                    0.0271 us      0.0211 us    29.48     30.2%   3r   0.000s  ok 1.3e-16       1.41x
   fftw1d_patient                0.0391 us      0.0372 us    20.45      5.7%   3r   0.015s  ok 1.5e-16       2.04x
   fftw1d_measure                0.0405 us      0.0363 us    19.77     19.9%   3r   0.008s  ok 1.5e-16       2.11x
   fftw1d_estimate               0.0480 us      0.0471 us    16.67      7.1%   3r   0.001s  ok 1.5e-16       2.50x
   d1_bluestein                  0.0960 us      0.0959 us     8.33      0.3%   3r   0.001s  ok 1.2e-16       5.00x
   fftw1d_custom                 0.1096 us      0.1086 us     7.30      3.5%   3r   0.000s  ok 1.1e-16       5.71x
   baseline_dft                  2.1283 us      2.1283 us     0.38      0.0%   3r   0.000s  ok 3.8e-16       110.78x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0600 us      0.0596 us    13.34     17.3%   3r   0.006s  ok ch=2.3e-15/1e-10 1s=3e-16 1.00x?
   d1_planner                    0.0694 us      0.0694 us    11.52      0.0%   3r   0.000s  ok ch=2.6e-15/1e-10 1s=4e-16 1.16x
   d1_batchlane                  0.0699 us      0.0698 us    11.45      0.1%   3r   0.000s  ok ch=2.3e-15/1e-10 1s=3e-16 1.17x
   d1_pow2                       0.0706 us      0.0706 us    11.33      0.0%   3r   0.000s  ok ch=2.6e-15/1e-10 1s=4e-16 1.18x
   d1_twiddle                    0.0706 us      0.0706 us    11.33      0.0%   3r   0.000s  ok ch=1.2e-15/1e-10 1s=4e-16 1.18x
   fftw1d_custom                 0.1217 us      0.1216 us     6.57      1.0%   3r   0.000s  ok ch=2.5e-15/1e-10 1s=4e-16 2.03x
   mkl1d_dfti                    0.1310 us      0.1309 us     6.11      0.1%   3r   0.001s  ok ch=2.2e-15/1e-10 1s=3e-16 2.19x
   fftw1d_patient                0.1360 us      0.1343 us     5.88      1.3%   3r   0.015s  ok ch=3.3e-15/1e-10 1s=3e-16 2.27x
   fftw1d_measure                0.1548 us      0.1548 us     5.17      0.0%   3r   0.008s  ok ch=3.3e-15/1e-10 1s=3e-16 2.58x
   fftw1d_estimate               0.1653 us      0.1651 us     4.84      0.2%   3r   0.001s  ok ch=3.3e-15/1e-10 1s=3e-16 2.76x
   d1_bluestein                  0.1819 us      0.1814 us     4.40      3.9%   3r   0.001s  ok ch=2.4e-15/1e-10 1s=4e-16 3.03x
   baseline_dft                  1.8469 us      1.8469 us     0.43      0.0%   3r   0.000s  ok ch=1.5e-14/1e-10 1s=8e-16 30.81x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   mkl1d_dfti                    0.0153 us      0.0153 us    52.32     13.9%   3r   0.001s  ok 1.4e-16       1.00x?
   d1_twiddle                    0.0174 us      0.0173 us    46.08      0.4%   3r   0.000s  ok 1.4e-16       1.14x
   d1_race                       0.0176 us      0.0154 us    45.54     13.8%   3r   0.953s  ok 1.4e-16       1.15x
   d1_planner                    0.0176 us      0.0176 us    45.46      0.0%   3r   0.000s  ok 1.4e-16       1.15x
   d1_batchlane                  0.0176 us      0.0176 us    45.43      0.1%   3r   0.000s  ok 1.4e-16       1.15x
   d1_pow2                       0.0176 us      0.0176 us    45.42      0.0%   3r   0.000s  ok 1.4e-16       1.15x
   fftw1d_patient                0.0254 us      0.0253 us    31.45     12.3%   3r   0.057s  ok 1.6e-16       1.66x
   fftw1d_measure                0.0290 us      0.0259 us    27.56     12.4%   3r   0.009s  ok 1.5e-16       1.90x
   fftw1d_custom                 0.0897 us      0.0892 us     8.91     19.3%   3r   0.000s  ok 1.4e-16       5.87x
   d1_bluestein                  0.0935 us      0.0920 us     8.56     17.9%   3r   0.001s  ok 1.4e-16       6.12x
   fftw1d_custom_soa             0.0947 us      0.0884 us     8.44      9.9%   3r   0.000s  ok 1.4e-16       6.20x
   fftw1d_estimate               0.1010 us      0.1006 us     7.92      1.4%   3r   0.001s  ok 1.5e-16       6.61x
   baseline_dft                  2.1243 us      2.1240 us     0.38      0.0%   3r   0.000s  ok 3.4e-16       138.94x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_batchlane                  0.0340 us      0.0340 us    23.54     13.8%   3r   0.000s  ok ch=6.5e-12/1e-10 1s=3e-16 1.00x?
   d1_race                       0.0387 us      0.0340 us    20.69     13.8%   3r   0.007s  ok ch=6.5e-12/1e-10 1s=3e-16 1.14x
   d1_twiddle                    0.0415 us      0.0369 us    19.26     12.6%   3r   0.000s  ok ch=2.8e-12/1e-10 1s=4e-16 1.22x
   d1_pow2                       0.0436 us      0.0387 us    18.37     13.6%   3r   0.000s  ok ch=1.2e-11/1e-10 1s=4e-16 1.28x
   fftw1d_custom_soa             0.0667 us      0.0666 us    12.00      0.2%   3r   0.000s  ok ch=7.2e-12/1e-10 1s=3e-16 1.96x
   d1_planner                    0.0774 us      0.0773 us    10.34      0.1%   3r   0.000s  ok ch=1.1e-11/1e-10 1s=3e-16 2.28x
   fftw1d_custom                 0.1039 us      0.1039 us     7.70      0.3%   3r   0.000s  ok ch=7.2e-12/1e-10 1s=3e-16 3.06x
   mkl1d_dfti                    0.1123 us      0.1122 us     7.13     13.8%   3r   0.002s  ok ch=7.3e-12/1e-10 1s=3e-16 3.30x
   fftw1d_patient                0.1222 us      0.1220 us     6.55      0.5%   3r   0.057s  ok ch=9.2e-12/1e-10 1s=4e-16 3.60x
   fftw1d_measure                0.1238 us      0.1228 us     6.46     13.1%   3r   0.010s  ok ch=7.3e-12/1e-10 1s=3e-16 3.64x
   d1_bluestein                  0.1832 us      0.1787 us     4.37      5.2%   3r   0.001s  ok ch=1.0e-11/1e-10 1s=3e-16 5.39x
   fftw1d_estimate               0.1858 us      0.1858 us     4.30      0.1%   3r   0.001s  ok ch=7.3e-12/1e-10 1s=3e-16 5.47x
   baseline_dft                  1.9513 us      1.9509 us     0.41      0.1%   3r   0.000s  ok ch=1.4e-11/1e-10 1s=6e-16 57.42x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_composite                  0.0451 us      0.0430 us    39.25     15.1%   3r   0.000s  ok 1.9e-16       1.00x?
   d1_race                       0.0490 us      0.0415 us    36.20     19.3%   3r   0.007s  ok 1.9e-16       1.08x?
   mkl1d_dfti                    0.0616 us      0.0612 us    28.78      3.5%   3r   0.002s  ok 2.4e-16       1.36x
   d1_batchlane                  0.0687 us      0.0676 us    25.79     19.6%   3r   0.000s  ok 2.3e-16       1.52x
