# What previous generations produced (round d1_r7 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_libbase2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r4/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r5/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/d1/results/d1_r6/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_batchlane.md 730 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_bluestein.md 814 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_composite.md 695 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_planner.md 554 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_pow2.md 707 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_prime.md 661 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_race.md 790 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_rader.md 770 lines
  /home/lqcd/wdetmold/fft/bench/d1/strategies/d1_twiddle.md 724 lines

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
      
  /home/lqcd/wdetmold/fft/bench/d1/exemplars/d1_r6/
      # Round d1_r6 — what it established
      
      Promoted: d1_prime d1_rader d1_pow2 d1_composite d1_batchlane d1_bluestein d1_twiddle d1_planner d1_race
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round d1_r6 ===

-- L=13 (non-batched, single call), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_rader                      0.0163 us      0.0143 us    14.77     42.0%   9r   0.000s  ok 1.6e-16       1.00x?
   d1_planner                    0.0169 us      0.0148 us    14.26     34.1%   9r   0.000s  ok 9.0e-17       1.04x?
   d1_batchlane                  0.0173 us      0.0144 us    13.93     40.1%   9r   0.000s  ok 9.0e-17       1.06x?
   d1_race                       0.0175 us      0.0140 us    13.77     32.2%   9r   0.009s  ok 1.6e-16       1.07x?
   d1_prime                      0.0187 us      0.0150 us    12.83     46.4%   9r   0.000s  ok 1.6e-16       1.15x
   mkl1d_dfti                    0.0224 us      0.0214 us    10.75     25.8%   9r   0.034s  ok 2.0e-16       1.37x
   fftw1d_measure                0.0264 us      0.0219 us     9.12     31.5%   9r   0.001s  ok 2.1e-16       1.62x
   fftw1d_estimate               0.0264 us      0.0224 us     9.12     19.9%   9r   0.001s  ok 2.1e-16       1.62x
   fftw1d_patient                0.0264 us      0.0218 us     9.11     27.1%   9r   0.001s  ok 2.1e-16       1.62x
   fftw1d_custom                 0.0567 us      0.0470 us     4.24     36.8%   9r   0.000s  ok 1.8e-16       3.48x
   d1_bluestein                  0.0927 us      0.0911 us     2.59     23.1%   9r   0.000s  ok 1.4e-16       5.69x
   baseline_dft                  0.2927 us      0.2927 us     0.82     20.7%   9r   0.000s  ok 3.9e-16       17.98x

-- L=13 (non-batched, chain m=200000), working set 0.000 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_rader                      0.0337 us      0.0337 us     7.13     13.8%   3r   0.000s  ok ch=7.8e-16/1e-10 1s=4e-16 1.00x?
   d1_prime                      0.0340 us      0.0340 us     7.07      0.0%   3r   0.000s  ok ch=7.8e-16/1e-10 1s=4e-16 1.01x?
   d1_batchlane                  0.0357 us      0.0357 us     6.74      0.0%   3r   0.000s  ok ch=4.7e-16/1e-10 1s=4e-16 1.06x?
   d1_race                       0.0387 us      0.0340 us     6.21     13.8%   3r   0.005s  ok ch=7.8e-16/1e-10 1s=4e-16 1.15x?
   d1_planner                    0.0394 us      0.0393 us     6.10     13.3%   3r   0.000s  ok ch=6.7e-16/1e-10 1s=3e-16 1.17x
   fftw1d_custom                 0.0631 us      0.0625 us     3.81     21.0%   3r   0.000s  ok ch=8.1e-16/1e-10 1s=4e-16 1.87x
   fftw1d_measure                0.0674 us      0.0674 us     3.57      0.0%   3r   0.001s  ok ch=5.7e-16/1e-10 1s=5e-16 2.00x
   fftw1d_estimate               0.0674 us      0.0674 us     3.57      0.0%   3r   0.000s  ok ch=5.7e-16/1e-10 1s=5e-16 2.00x
   mkl1d_dfti                    0.0803 us      0.0803 us     2.99      0.3%   3r   0.054s  ok ch=9.1e-16/1e-10 1s=5e-16 2.38x
   fftw1d_patient                0.0814 us      0.0814 us     2.96      0.0%   3r   0.001s  ok ch=5.7e-16/1e-10 1s=5e-16 2.41x
   d1_bluestein                  0.1221 us      0.1212 us     1.97     21.9%   3r   0.000s  ok ch=4.9e-16/1e-10 1s=2e-16 3.62x
   baseline_dft                  0.3256 us      0.3256 us     0.74     20.7%   3r   0.000s  ok ch=1.2e-14/1e-10 1s=1e-15 9.65x

-- L=13 (batched B=512, single call), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_rader                      0.0095 us      0.0094 us    25.37     15.9%   9r   0.000s  ok 1.7e-16       1.00x?
   d1_batchlane                  0.0095 us      0.0095 us    25.29     16.7%   9r   0.000s  ok 1.4e-16       1.00x?
   d1_prime                      0.0096 us      0.0094 us    25.18     18.0%   9r   0.000s  ok 1.7e-16       1.01x?
   d1_race                       0.0098 us      0.0095 us    24.60     13.7%   9r   0.008s  ok 1.4e-16       1.03x?
   d1_planner                    0.0108 us      0.0095 us    22.28     14.0%   9r   0.000s  ok 1.4e-16       1.14x
   fftw1d_measure                0.0139 us      0.0122 us    17.28     19.6%   9r   0.002s  ok 1.8e-16       1.47x
   fftw1d_patient                0.0143 us      0.0141 us    16.86      3.0%   9r   0.003s  ok 1.8e-16       1.50x
   mkl1d_dfti                    0.0173 us      0.0159 us    13.89     21.5%   9r   0.033s  ok 1.7e-16       1.83x
   fftw1d_custom_soa             0.0324 us      0.0322 us     7.41     14.8%   9r   0.000s  ok 1.8e-16       3.42x
   fftw1d_estimate               0.0400 us      0.0350 us     6.02     14.6%   9r   0.001s  ok 1.8e-16       4.21x
   fftw1d_custom                 0.0488 us      0.0404 us     4.93     21.5%   9r   0.000s  ok 1.8e-16       5.15x
   d1_bluestein                  0.0864 us      0.0862 us     2.78     24.3%   9r   0.000s  ok 1.8e-16       9.12x
   baseline_dft                  0.3499 us      0.2899 us     0.69     20.7%   9r   0.000s  ok 4.1e-16       36.90x

-- L=13 (batched B=512, chain m=2000), working set 0.203 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0137 us      0.0137 us    17.52     13.7%   3r   0.000s  ok ch=4.3e-14/1e-10 1s=4e-16 1.00x?
   d1_race                       0.0141 us      0.0141 us    17.02      1.1%   3r   1.396s  ok ch=3.3e-14/1e-10 1s=3e-16 1.03x?
   d1_rader                      0.0146 us      0.0146 us    16.44     13.8%   3r   0.000s  ok ch=5.8e-13/1e-10 1s=5e-16 1.07x?
   d1_planner                    0.0155 us      0.0154 us    15.54      0.4%   3r   0.000s  ok ch=2.6e-14/1e-10 1s=3e-16 1.13x
   d1_batchlane                  0.0161 us      0.0160 us    14.90      0.9%   3r   0.000s  ok ch=3.3e-14/1e-10 1s=3e-16 1.18x
   fftw1d_custom_soa             0.0263 us      0.0262 us     9.14     14.8%   3r   0.000s  ok ch=6.2e-14/1e-10 1s=3e-16 1.92x
   fftw1d_measure                0.0513 us      0.0512 us     4.69     15.1%   3r   0.002s  ok ch=8.2e-14/1e-10 1s=3e-16 3.73x
   mkl1d_dfti                    0.0531 us      0.0530 us     4.53      0.0%   3r   0.051s  ok ch=1.2e-13/1e-10 1s=3e-16 3.87x
   fftw1d_custom                 0.0573 us      0.0478 us     4.20     20.2%   3r   0.000s  ok ch=6.2e-14/1e-10 1s=3e-16 4.17x
   fftw1d_patient                0.0586 us      0.0582 us     4.10      1.0%   3r   0.003s  ok ch=8.2e-14/1e-10 1s=3e-16 4.27x
   fftw1d_estimate               0.0746 us      0.0746 us     3.22      0.2%   3r   0.001s  ok ch=8.2e-14/1e-10 1s=3e-16 5.43x
   d1_bluestein                  0.1232 us      0.1230 us     1.95      1.0%   3r   0.000s  ok ch=5.5e-14/1e-10 1s=3e-16 8.98x
   baseline_dft                  0.3283 us      0.3270 us     0.73      0.4%   3r   0.000s  ok ch=9.3e-13/1e-10 1s=7e-16 23.91x

-- L=31 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_batchlane                  0.0478 us      0.0464 us    16.06     19.0%   9r   0.000s  ok 2.1e-16       1.00x?
   d1_prime                      0.0485 us      0.0472 us    15.83     17.1%   9r   0.000s  ok 2.5e-16       1.01x?
   d1_planner                    0.0504 us      0.0497 us    15.24     13.9%   9r   0.000s  ok 2.1e-16       1.05x?
   d1_race                       0.0533 us      0.0470 us    14.40     24.7%   9r   0.006s  ok 2.1e-16       1.12x
   d1_rader                      0.0552 us      0.0543 us    13.90      3.5%   9r   0.000s  ok 2.5e-16       1.16x
   d1_bluestein                  0.1882 us      0.1843 us     4.08     20.8%   9r   0.001s  ok 3.4e-16       3.94x
   fftw1d_custom                 0.2150 us      0.1785 us     3.57     22.8%   9r   0.000s  ok 2.8e-16       4.50x
   fftw1d_measure                0.2633 us      0.2591 us     2.92     21.7%   9r   0.043s  ok 2.1e-16       5.51x
   mkl1d_dfti                    0.2794 us      0.2304 us     2.75     21.6%   9r   0.037s  ok 2.2e-16       5.84x
   fftw1d_estimate               0.3134 us      0.2605 us     2.45     25.2%   9r   0.001s  ok 2.1e-16       6.56x
   fftw1d_patient                0.3143 us      0.2595 us     2.44     23.8%   9r   0.118s  ok 2.1e-16       6.58x
   baseline_dft                  1.9979 us      1.6553 us     0.38     20.7%   9r   0.000s  ok 4.9e-16       41.79x

-- L=31 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_batchlane                  0.0520 us      0.0520 us    14.76      0.2%   3r   0.000s  ok ch=3.4e-15/1e-10 1s=6e-16 1.00x?
   d1_race                       0.0541 us      0.0509 us    14.20     13.6%   3r   0.009s  ok ch=1.2e-14/1e-10 1s=6e-16 1.04x?
   d1_rader                      0.0579 us      0.0579 us    13.26      0.1%   3r   0.000s  ok ch=1.2e-14/1e-10 1s=6e-16 1.11x
   d1_prime                      0.0580 us      0.0578 us    13.25      0.6%   3r   0.000s  ok ch=1.2e-14/1e-10 1s=6e-16 1.11x
   d1_planner                    0.0614 us      0.0583 us    12.50     13.7%   3r   0.000s  ok ch=4.1e-15/1e-10 1s=5e-16 1.18x
   d1_bluestein                  0.2444 us      0.2344 us     3.14      4.9%   3r   0.001s  ok ch=4.1e-15/1e-10 1s=6e-16 4.70x
   fftw1d_custom                 0.2528 us      0.2518 us     3.04      0.4%   3r   0.000s  ok ch=1.2e-15/1e-10 1s=7e-16 4.86x
   fftw1d_measure                0.3492 us      0.3472 us     2.20     20.8%   3r   0.042s  ok ch=4.2e-15/1e-10 1s=4e-16 6.71x
   fftw1d_patient                0.3595 us      0.3522 us     2.14      2.5%   3r   0.118s  ok ch=4.2e-15/1e-10 1s=4e-16 6.91x
   mkl1d_dfti                    0.3857 us      0.3856 us     1.99      0.0%   3r   0.032s  ok ch=9.1e-15/1e-10 1s=4e-16 7.41x
   fftw1d_estimate               0.4223 us      0.3631 us     1.82     16.3%   3r   0.001s  ok ch=4.2e-15/1e-10 1s=4e-16 8.11x
   baseline_dft                  1.7390 us      1.7390 us     0.44      0.0%   3r   0.000s  ok ch=1.9e-14/1e-10 1s=9e-16 33.42x

-- L=31 (batched B=512, single call), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_prime                      0.0480 us      0.0423 us    16.01     14.6%   9r   0.000s  ok 2.9e-16       1.00x?
   d1_race                       0.0489 us      0.0427 us    15.69     15.5%   9r   0.008s  ok 2.9e-16       1.02x?
   d1_planner                    0.0502 us      0.0441 us    15.29     13.9%   9r   0.000s  ok 2.1e-16       1.05x?
   d1_batchlane                  0.0502 us      0.0441 us    15.29     13.9%   9r   0.000s  ok 2.1e-16       1.05x?
   d1_rader                      0.0504 us      0.0496 us    15.25      2.5%   9r   0.000s  ok 2.9e-16       1.05x
   fftw1d_custom_soa             0.1080 us      0.0931 us     7.11     26.1%   9r   0.000s  ok 2.6e-16       2.25x
   d1_bluestein                  0.1978 us      0.1835 us     3.88     15.9%   9r   0.001s  ok 3.2e-16       4.12x
   fftw1d_custom                 0.1985 us      0.1632 us     3.87     24.5%   9r   0.000s  ok 2.6e-16       4.14x
   mkl1d_dfti                    0.2267 us      0.2266 us     3.39     21.0%   9r   0.014s  ok 2.3e-16       4.73x
   fftw1d_estimate               0.2641 us      0.2632 us     2.91     21.1%   9r   0.001s  ok 2.1e-16       5.51x
   fftw1d_measure                0.2643 us      0.2631 us     2.91     20.8%   9r   0.043s  ok 2.1e-16       5.51x
   fftw1d_patient                0.3179 us      0.2633 us     2.42     46.1%   9r   0.122s  ok 2.1e-16       6.63x
   baseline_dft                  1.9936 us      1.6517 us     0.39     20.7%   9r   0.000s  ok 4.2e-16       41.55x

-- L=31 (batched B=512, chain m=1200), working set 0.484 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0453 us      0.0451 us    16.93      1.0%   3r   2.255s  ok ch=1.1e-11/1e-10 1s=7e-16 1.00x?
   d1_prime                      0.0454 us      0.0451 us    16.92     12.5%   3r   0.000s  ok ch=1.1e-11/1e-10 1s=7e-16 1.00x?
   d1_batchlane                  0.0550 us      0.0536 us    13.95      2.9%   3r   0.000s  ok ch=2.2e-12/1e-10 1s=4e-16 1.21x
   d1_planner                    0.0553 us      0.0494 us    13.90     12.2%   3r   0.000s  ok ch=3.1e-12/1e-10 1s=4e-16 1.22x
   d1_rader                      0.0634 us      0.0562 us    12.11     13.0%   3r   0.000s  ok ch=4.0e-12/1e-10 1s=7e-16 1.40x
   fftw1d_custom_soa             0.0781 us      0.0770 us     9.83     15.1%   3r   0.000s  ok ch=1.3e-12/1e-10 1s=5e-16 1.72x
   fftw1d_custom                 0.1770 us      0.1768 us     4.34      0.2%   3r   0.000s  ok ch=1.3e-12/1e-10 1s=5e-16 3.90x
   d1_bluestein                  0.2119 us      0.2103 us     3.62     11.0%   3r   0.001s  ok ch=1.2e-12/1e-10 1s=7e-16 4.67x
   mkl1d_dfti                    0.3153 us      0.3153 us     2.44      0.0%   3r   0.048s  ok ch=5.9e-12/1e-10 1s=5e-16 6.95x
   fftw1d_measure                0.3535 us      0.3534 us     2.17      0.0%   3r   0.043s  ok ch=1.4e-12/1e-10 1s=4e-16 7.79x
   fftw1d_estimate               0.3536 us      0.3534 us     2.17      0.1%   3r   0.001s  ok ch=1.4e-12/1e-10 1s=4e-16 7.80x
   fftw1d_patient                0.3538 us      0.3536 us     2.17      0.1%   3r   0.118s  ok ch=1.4e-12/1e-10 1s=4e-16 7.80x
   baseline_dft                  1.7449 us      1.7449 us     0.44      0.0%   3r   0.000s  ok ch=1.1e-12/1e-10 1s=8e-16 38.48x

-- L=32 (non-batched, single call), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_twiddle                    0.0192 us      0.0171 us    41.59     23.1%   9r   0.000s  ok 2.0e-16       1.00x?
   d1_batchlane                  0.0201 us      0.0190 us    39.76     26.7%   9r   0.000s  ok 1.9e-16       1.05x?
   d1_planner                    0.0210 us      0.0209 us    38.17     25.2%   9r   0.000s  ok 1.9e-16       1.09x?
   d1_pow2                       0.0210 us      0.0184 us    38.02     47.1%   9r   0.000s  ok 1.9e-16       1.09x?
   d1_race                       0.0212 us      0.0211 us    37.82     26.8%   9r   0.059s  ok 1.9e-16       1.10x?
   mkl1d_dfti                    0.0263 us      0.0225 us    30.47     43.8%   9r   0.001s  ok 1.6e-16       1.37x
   fftw1d_patient                0.0387 us      0.0346 us    20.67     26.8%   9r   0.016s  ok 1.6e-16       2.01x
   fftw1d_measure                0.0392 us      0.0323 us    20.39     44.8%   9r   0.008s  ok 1.6e-16       2.04x
   fftw1d_estimate               0.0445 us      0.0415 us    17.99     16.7%   9r   0.001s  ok 1.6e-16       2.31x
   d1_bluestein                  0.0881 us      0.0874 us     9.08     27.3%   9r   0.000s  ok 1.7e-16       4.58x
   fftw1d_custom                 0.1101 us      0.0894 us     7.27     24.1%   9r   0.000s  ok 1.5e-16       5.72x
   baseline_dft                  2.1283 us      1.7633 us     0.38     20.7%   9r   0.000s  ok 3.2e-16       110.65x

-- L=32 (non-batched, chain m=100000), working set 0.001 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0573 us      0.0572 us    13.97      0.2%   3r   0.006s  ok ch=2.5e-15/1e-10 1s=3e-16 1.00x?
   d1_twiddle                    0.0647 us      0.0647 us    12.36      0.0%   3r   0.000s  ok ch=1.1e-15/1e-10 1s=4e-16 1.13x
   d1_batchlane                  0.0674 us      0.0572 us    11.88     17.7%   3r   0.000s  ok ch=2.5e-15/1e-10 1s=3e-16 1.18x
   d1_planner                    0.0791 us      0.0791 us    10.11      0.0%   3r   0.000s  ok ch=1.9e-15/1e-10 1s=4e-16 1.38x
   d1_pow2                       0.0804 us      0.0706 us     9.96     13.8%   3r   0.000s  ok ch=2.4e-15/1e-10 1s=3e-16 1.40x
   fftw1d_custom                 0.1217 us      0.1216 us     6.57      1.0%   3r   0.000s  ok ch=2.4e-15/1e-10 1s=3e-16 2.12x
   mkl1d_dfti                    0.1311 us      0.1309 us     6.10     14.0%   3r   0.002s  ok ch=2.2e-15/1e-10 1s=3e-16 2.29x
   fftw1d_measure                0.1359 us      0.1341 us     5.89      1.4%   3r   0.007s  ok ch=4.7e-15/1e-10 1s=3e-16 2.37x
   fftw1d_patient                0.1361 us      0.1357 us     5.88      0.4%   3r   0.015s  ok ch=3.0e-15/1e-10 1s=3e-16 2.38x
   fftw1d_estimate               0.1454 us      0.1452 us     5.50      0.6%   3r   0.001s  ok ch=4.3e-15/1e-10 1s=3e-16 2.54x
   d1_bluestein                  0.2099 us      0.1941 us     3.81     14.0%   3r   0.001s  ok ch=2.5e-15/1e-10 1s=3e-16 3.67x
   baseline_dft                  1.8470 us      1.8469 us     0.43      0.0%   3r   0.000s  ok ch=2.8e-14/1e-10 1s=6e-16 32.25x

-- L=32 (batched B=512, single call), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   mkl1d_dfti                    0.0153 us      0.0153 us    52.31     13.9%   9r   0.002s  ok 1.4e-16       1.00x?
   d1_twiddle                    0.0174 us      0.0173 us    46.09      0.5%   9r   0.000s  ok 1.4e-16       1.13x
   d1_planner                    0.0175 us      0.0153 us    45.79     13.9%   9r   0.000s  ok 1.4e-16       1.14x
   d1_pow2                       0.0176 us      0.0154 us    45.52     13.9%   9r   0.000s  ok 1.4e-16       1.15x
   d1_race                       0.0176 us      0.0176 us    45.49      0.1%   9r   0.011s  ok 1.4e-16       1.15x
   d1_batchlane                  0.0176 us      0.0176 us    45.36      0.1%   9r   0.000s  ok 1.4e-16       1.15x
   fftw1d_measure                0.0256 us      0.0248 us    31.24     21.0%   9r   0.010s  ok 1.6e-16       1.67x
   fftw1d_patient                0.0256 us      0.0248 us    31.23     14.4%   9r   0.055s  ok 1.6e-16       1.67x
   fftw1d_custom_soa             0.0883 us      0.0777 us     9.06     33.6%   9r   0.000s  ok 1.4e-16       5.77x
   fftw1d_custom                 0.0887 us      0.0882 us     9.02     30.8%   9r   0.000s  ok 1.4e-16       5.80x
   fftw1d_estimate               0.0892 us      0.0884 us     8.97     14.7%   9r   0.001s  ok 1.5e-16       5.83x
   d1_bluestein                  0.0996 us      0.0936 us     8.03     19.7%   9r   0.001s  ok 1.4e-16       6.51x
   baseline_dft                  2.1240 us      2.1239 us     0.38      0.0%   9r   0.000s  ok 3.4e-16       138.87x

-- L=32 (batched B=512, chain m=1000), working set 0.500 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_race                       0.0331 us      0.0328 us    24.15     14.5%   3r   0.008s  ok ch=1.8e-13/1e-10 1s=3e-16 1.00x?
   d1_twiddle                    0.0341 us      0.0341 us    23.47     13.6%   3r   0.000s  ok ch=2.6e-13/1e-10 1s=4e-16 1.03x?
   d1_batchlane                  0.0378 us      0.0339 us    21.15     12.3%   3r   0.000s  ok ch=1.8e-13/1e-10 1s=3e-16 1.14x?
   d1_pow2                       0.0389 us      0.0384 us    20.56     13.2%   3r   0.001s  ok ch=1.6e-13/1e-10 1s=4e-16 1.17x
   fftw1d_custom_soa             0.0666 us      0.0666 us    12.01     13.8%   3r   0.000s  ok ch=2.3e-13/1e-10 1s=3e-16 2.01x
   d1_planner                    0.0680 us      0.0680 us    11.76      1.5%   3r   0.000s  ok ch=1.3e-13/1e-10 1s=3e-16 2.05x
   fftw1d_custom                 0.1042 us      0.1039 us     7.68     20.9%   3r   0.000s  ok ch=2.3e-13/1e-10 1s=3e-16 3.15x
   mkl1d_dfti                    0.1122 us      0.1122 us     7.13      0.0%   3r   0.001s  ok ch=1.6e-13/1e-10 1s=3e-16 3.39x
   fftw1d_patient                0.1224 us      0.1221 us     6.54      0.3%   3r   0.056s  ok ch=2.7e-13/1e-10 1s=3e-16 3.69x
   fftw1d_measure                0.1232 us      0.1232 us     6.49     13.9%   3r   0.010s  ok ch=2.8e-13/1e-10 1s=4e-16 3.72x
   fftw1d_estimate               0.1857 us      0.1857 us     4.31      0.3%   3r   0.001s  ok ch=2.8e-13/1e-10 1s=3e-16 5.61x
   d1_bluestein                  0.1906 us      0.1802 us     4.20      6.1%   3r   0.001s  ok ch=1.3e-13/1e-10 1s=3e-16 5.75x
   baseline_dft                  1.9513 us      1.9509 us     0.41      0.1%   3r   0.000s  ok ch=1.5e-12/1e-10 1s=6e-16 58.90x

-- L=60 (non-batched, single call), working set 0.002 MiB --
   backend                    median us/xf   best us/xf     GF/s    spread runs    setup  correctness   (? = gap inside the noise)
   d1_batchlane                  0.0422 us      0.0422 us    41.97     26.2%   9r   0.000s  ok 2.4e-16       1.00x?
   d1_composite                  0.0487 us      0.0467 us    36.41      5.6%   9r   0.000s  ok 2.4e-16       1.15x
   d1_race                       0.0489 us      0.0470 us    36.25     12.0%   9r   0.009s  ok 2.4e-16       1.16x
   d1_planner                    0.0507 us      0.0488 us    34.98      9.4%   9r   0.000s  ok 2.4e-16       1.20x
   mkl1d_dfti                    0.0620 us      0.0606 us    28.59      9.3%   9r   0.002s  ok 2.8e-16       1.47x
   fftw1d_patient                0.0742 us      0.0623 us    23.90     28.4%   9r   0.129s  ok 2.2e-16       1.76x
   d1_twiddle                    0.0801 us      0.0764 us    22.12     15.3%   9r   0.000s  ok 1.9e-16       1.90x
   fftw1d_measure                0.0863 us      0.0657 us    20.55     38.7%   9r   0.046s  ok 2.6e-16       2.04x
   fftw1d_estimate               0.0948 us      0.0945 us    18.70      1.8%   9r   0.001s  ok 2.2e-16       2.24x
   d1_bluestein                  0.2221 us      0.2173 us     7.98     13.2%   9r   0.001s  ok 1.8e-16       5.26x
