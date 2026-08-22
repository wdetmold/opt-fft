# What previous generations produced (round ice_r1 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/ice/results/ice_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  (none yet)

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round ice_smoke ===
# round ice_smoke
host: a80n0.lqcd.mit   date: 2026-08-22T15:17:31-04:00   slurm_job: 438572
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                   22.575 us  1444.779 us     0.37       0.0%    0.445s  ok 2.4e-16       1.00x
   L6_pfa                        24.969 us  1598.039 us     0.34       0.0%    0.430s  ok 2.4e-16       1.11x
   mkl_dfti                      38.641 us  2473.012 us     0.22       0.0%    0.033s  ok 2.4e-16       1.71x
   mkl2026_dfti                  40.430 us  2587.519 us     0.21       0.0%    0.002s  ok 2.5e-16       1.79x
   fftw3_measure                 51.042 us  3266.689 us     0.16       0.0%    0.012s  ok 2.0e-16       2.26x
   fftw3_patient                 51.227 us  3278.518 us     0.16       0.0%    0.020s  ok 2.0e-16       2.27x
   fftw3_estimate                95.905 us  6137.909 us     0.09       0.0%    0.001s  ok 2.0e-16       4.25x
   ducc0_c2c                    226.559 us 14499.768 us     0.04       0.0%    0.000s  ok 1.8e-16       10.04x
   baseline_matrix              718.145 us 45961.291 us     0.01       0.0%    0.000s  ok 6.0e-16       31.81x

-- L=17 (batched B=32), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_winograd                1586.374 us 50763.962 us     0.19       0.0%    0.741s  ok 3.3e-16       1.00x
   L17_matrixsimd              2061.465 us 65966.889 us     0.15       0.0%    0.227s  ok 3.2e-16       1.30x
   L17_rader                   2325.005 us 74400.153 us     0.13       0.0%    0.388s  ok 3.2e-16       1.47x
   ducc0_c2c                   7378.352 us 236107.271 us     0.04       0.0%    0.000s  ok 2.6e-16       4.65x
   mkl2026_dfti                7452.526 us 238480.821 us     0.04       0.0%    0.050s  ok 3.1e-16       4.70x
   mkl_dfti                    7476.644 us 239252.601 us     0.04       0.0%    0.051s  ok 3.1e-16       4.71x
   fftw3_estimate              7848.411 us 251149.145 us     0.04       0.0%    0.002s  ok 3.0e-16       4.95x
   fftw3_patient               7850.893 us 251228.566 us     0.04       0.0%    0.017s  ok 3.0e-16       4.95x
   fftw3_measure               7854.156 us 251332.979 us     0.04       0.0%    0.008s  ok 3.0e-16       4.95x
   baseline_matrix            42802.293 us 1369673.380 us     0.01       0.0%    0.000s  ok 8.4e-16       26.98x

backends:
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit, pinned sines, b1dec[yz/kyz/x/kx]=8.54/8.01/3.80/3.65, clk512/256=2.90/2.90 GHz, d256=2.90
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 256, pf=0, pfw=0, clk256=2.90 clk512=2.90
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h4, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.19 f23=7.59 fu=12.81 fu4=13.43
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, plan-raced; variant=fused_pf_d2 clkS256=2.90 clkD256=2.90 clkS512=2.90 kclk=2.90GHz bf=195.5 bsp=194.6 bx=56.3 byz=137.0ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), radix-2-first VD6, ymm; variant=fused_zp kclk=3.30GHz ab1=f164.1,fx163.5ns abL=f438.1,f3445.6ns xod=-0.3%
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
