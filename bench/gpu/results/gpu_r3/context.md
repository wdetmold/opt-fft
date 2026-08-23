# What previous generations produced (round gpu_r3 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_validate/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/local_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L13_dmma.md 230 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_dmma.md 289 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_raderfused.md 275 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L23_rader.md 293 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_globalpass.md 221 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_sharedtiled.md 199 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L45_pfa.md 275 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L64_radix8.md 219 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_batchcoalesced.md 174 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_warpvolume.md 203 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_blockfused.md 239 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_warpradix8.md 274 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/gpu/exemplars/gpu_r1/
      # Round gpu_r1 — what it established
      
      Promoted: L6_batchcoalesced L6_warpvolume L8_blockfused L8_warpradix8 L13_dmma L17_dmma L17_raderfused L23_rader L36_sharedtiled L36_globalpass L45_pfa L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      
  /home/lqcd/wdetmold/fft/bench/gpu/exemplars/gpu_r2/
      # Round gpu_r2 — what it established
      
      Promoted: L6_batchcoalesced L8_blockfused L8_warpradix8 L13_dmma L17_dmma L23_rader L36_globalpass L36_sharedtiled L45_pfa L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round gpu_r2 ===
# round gpu_r2
host: a80n1.lqcd.mit   date: 2026-08-22T20:09:36-04:00   slurm_job: 438580
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
gpu: NVIDIA A100-SXM4-40GB, 40960 MiB, 1410 MHz, 1215 MHz, Disabled
visible devices: 0  (pinned to one A100 on purpose)
all gpus on node: 0, NVIDIA A100-SXM4-40GB
all gpus on node: 1, NVIDIA A100-SXM4-40GB
all gpus on node: 2, NVIDIA A100-SXM4-40GB
all gpus on node: 3, NVIDIA A100-SXM4-40GB
all gpus on node: 4, NVIDIA A100-SXM4-40GB
all gpus on node: 5, NVIDIA A100-SXM4-40GB
all gpus on node: 6, NVIDIA A100-SXM4-40GB
all gpus on node: 7, NVIDIA A100-SXM4-40GB
nvcc: Build cuda_12.2.r12.2/compiler.33053471_0
driver: 525.125.06
clocks after sweep: 210 MHz, 1410 MHz, 28, 56.20 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 55.34 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 25, 58.82 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 56.36 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 54.95 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 25, 55.76 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 54.68 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 26, 51.85 W, 0x0000000000000001

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_warpvolume                  2.969 us     2.969 us     2.82       2.4%    0.000s  ok 2.5e-16       1.00x
   L6_batchcoalesced              3.693 us     3.693 us     2.27       3.8%    0.000s  ok 2.5e-16       1.24x
   cufft                         10.259 us    10.259 us     0.82       3.0%    0.014s  ok 2.5e-16       3.46x
   baseline_gpu                  15.057 us    15.057 us     0.56       3.7%    0.000s  ok 6.0e-16       5.07x

-- L=6 (batched B=4854), volume 216, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              0.003 us    14.629 us  2779.00       0.6%    0.000s  ok 2.5e-16       1.00x
   L6_warpvolume                  0.003 us    14.960 us  2717.47       1.1%    0.000s  ok 2.5e-16       1.02x
   cufft                          0.011 us    51.192 us   794.14       0.5%    0.017s  ok 2.5e-16       3.50x
   baseline_gpu                   0.033 us   162.469 us   250.22       0.4%    0.000s  ok 6.0e-16       11.11x

-- L=6 (batched B=310608), volume 216, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              0.005 us  1540.181 us  1689.04       0.1%    0.000s  ok 2.5e-16       1.00x
   L6_warpvolume                  0.005 us  1540.565 us  1688.62       0.1%    0.000s  ok 2.5e-16       1.00x
   cufft                          0.011 us  3520.614 us   738.91       0.2%    0.015s  ok 2.5e-16       2.29x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  3.191 us     3.191 us     7.22       5.6%    0.000s  ok 1.6e-16       1.00x
   L8_warpradix8                  3.595 us     3.595 us     6.41       4.0%    0.000s  ok 2.2e-16       1.13x
   cufft                          9.352 us     9.352 us     2.46       0.7%    0.009s  ok 2.4e-16       2.93x
   baseline_gpu                  17.650 us    17.650 us     1.31       3.0%    0.000s  ok 4.0e-16       5.53x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.006 us    13.058 us  3613.70       1.4%    0.114s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.007 us    13.990 us  3372.77       2.1%    0.109s  ok 2.3e-16       1.07x
   cufft                          0.027 us    54.526 us   865.38       0.1%    0.018s  ok 2.4e-16       4.18x
   baseline_gpu                   0.085 us   174.391 us   270.58       0.5%    0.000s  ok 3.9e-16       13.36x

-- L=8 (batched B=131072), volume 512, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.012 us  1537.237 us  1964.50       0.0%    0.615s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.012 us  1539.456 us  1961.67       0.2%    0.510s  ok 2.3e-16       1.00x
   cufft                          0.028 us  3697.152 us   816.82       0.1%    0.010s  ok 2.4e-16       2.41x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       7.820 us     7.820 us    15.59       3.3%    0.000s  ok 3.2e-16       1.00x
   cufft                         12.384 us    12.384 us     9.85       1.6%    0.009s  ok 2.9e-16       1.58x
   baseline_gpu                  22.638 us    22.638 us     5.39       2.8%    0.000s  ok 7.8e-16       2.89x

-- L=13 (batched B=477), volume 2197, working set 31.98 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.045 us    21.407 us  2717.25       1.6%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.132 us    62.991 us   923.45       0.6%    0.016s  ok 2.9e-16       2.94x
   baseline_gpu                   0.535 us   255.290 us   227.86       0.3%    0.000s  ok 7.9e-16       11.93x

-- L=13 (batched B=30549), volume 2197, working set 2048.22 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.051 us  1562.837 us  2383.73       0.1%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.155 us  4743.552 us   785.36       0.0%    0.010s  ok 2.9e-16       3.04x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       7.635 us     7.635 us    39.45       1.5%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 8.687 us     8.687 us    34.68       3.6%    0.000s  ok 3.1e-16       1.14x
   cufft                         13.518 us    13.518 us    22.28       4.3%    0.016s  ok 3.2e-16       1.77x
   baseline_gpu                  26.993 us    26.993 us    11.16       1.5%    0.000s  ok 8.4e-16       3.54x

-- L=17 (batched B=213), volume 4913, working set 31.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       0.149 us    31.696 us  2024.28       0.2%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 0.151 us    32.139 us  1996.38       0.3%    0.000s  ok 3.2e-16       1.01x
   cufft                          0.317 us    67.478 us   950.85       0.4%    0.017s  ok 3.2e-16       2.13x
   baseline_gpu                   1.453 us   309.504 us   207.30       0.4%    0.000s  ok 8.4e-16       9.76x

-- L=17 (batched B=13660), volume 4913, working set 2048.08 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       0.115 us  1571.669 us  2618.07       0.1%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 0.115 us  1574.955 us  2612.61       0.7%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.348 us  4758.272 us   864.76       0.1%    0.010s  ok 3.2e-16       3.03x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      9.088 us     9.088 us    90.84       0.5%    0.050s  ok 3.5e-16       1.00x
   cufft                         14.880 us    14.880 us    55.48       2.7%    0.016s  ok 3.8e-16       1.64x
   baseline_gpu                  32.250 us    32.250 us    25.60       1.0%    0.000s  ok 7.3e-16       3.55x

-- L=23 (batched B=86), volume 12167, working set 31.93 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.448 us    38.556 us  1841.47       0.3%    0.266s  ok 3.5e-16       1.00x
   cufft                          0.883 us    75.955 us   934.76       0.4%    0.018s  ok 3.8e-16       1.97x
   baseline_gpu                   4.542 us   390.604 us   181.77       0.2%    0.000s  ok 7.4e-16       10.13x

-- L=23 (batched B=5515), volume 12167, working set 2047.76 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.409 us  2257.152 us  2017.16       4.0%   10.971s  ok 3.5e-16       1.00x
   cufft                          0.917 us  5055.829 us   900.55       0.3%    0.014s  ok 3.8e-16       2.24x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_globalpass                 9.034 us     9.034 us   400.51       1.7%    0.359s  ok 4.8e-16       1.00x
   L36_sharedtiled               10.001 us    10.001 us   361.77       1.7%    0.056s  ok 4.8e-16       1.11x
   cufft                         12.914 us    12.914 us   280.18       3.6%    0.016s  ok 4.3e-16       1.43x
   baseline_gpu                  45.890 us    45.890 us    78.84       0.8%    0.000s  ok 8.0e-16       5.08x

-- L=36 (batched B=22), volume 46656, working set 31.32 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_sharedtiled                1.383 us    30.433 us  2615.53       1.8%    0.409s  ok 4.8e-16       1.00x
   L36_globalpass                 1.673 us    36.798 us  2163.10       4.4%    0.575s  ok 4.8e-16       1.21x
   cufft                          2.322 us    51.077 us  1558.40       0.5%    0.016s  ok 4.3e-16       1.68x
   baseline_gpu                  23.325 us   513.149 us   155.12       0.7%    0.000s  ok 8.0e-16       16.86x

-- L=36 (batched B=1438), volume 46656, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_globalpass                 1.353 us  1945.088 us  2674.87       0.0%    0.395s  ok 4.8e-16       1.00x
   L36_sharedtiled                1.446 us  2079.858 us  2501.54       0.3%    0.585s  ok 4.8e-16       1.07x
   cufft                          2.352 us  3381.965 us  1538.41       0.6%    0.011s  ok 4.3e-16       1.74x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       12.623 us    12.623 us   594.67       0.7%    0.000s  ok 8.2e-16       1.00x
   cufft                         18.085 us    18.085 us   415.08       2.2%    0.010s  ok 4.4e-16       1.43x
   baseline_gpu                  72.129 us    72.129 us   104.07       1.5%    0.000s  ok 8.0e-16       5.71x

-- L=45 (batched B=11), volume 91125, working set 30.59 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        3.319 us    36.513 us  2261.47       0.3%    0.000s  ok 8.2e-16       1.00x
   cufft                          5.827 us    64.102 us  1288.16       0.6%    0.016s  ok 4.4e-16       1.76x
   baseline_gpu                  55.797 us   613.772 us   134.53       0.4%    0.000s  ok 8.0e-16       16.81x

-- L=45 (batched B=736), volume 91125, working set 2046.75 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        3.311 us  2436.754 us  2267.33       0.4%    0.000s  ok 8.2e-16       1.00x
   cufft                          6.452 us  4748.416 us  1163.53       0.0%    0.012s  ok 4.4e-16       1.95x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    18.510 us    18.510 us  1274.61       0.1%    0.000s  ok 4.2e-16       1.00x
   cufft                         23.524 us    23.524 us  1002.92       1.0%    0.018s  ok 4.0e-16       1.27x
   baseline_gpu                 207.411 us   207.411 us   113.75       0.5%    0.000s  ok 7.8e-16       11.21x

-- L=64 (batched B=4), volume 262144, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    10.416 us    41.663 us  2265.11       0.5%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.512 us    58.048 us  1625.76       0.4%    0.016s  ok 4.0e-16       1.39x
   baseline_gpu                 205.451 us   821.805 us   114.83       0.4%    0.000s  ok 7.8e-16       19.72x

-- L=64 (batched B=256), volume 262144, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    10.225 us  2617.515 us  2307.46       0.1%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.609 us  3739.904 us  1614.96       0.1%    0.011s  ok 4.0e-16       1.43x

backends:
   L13_dmma                 fused block-per-volume in shared, conj-folded dense-13 lines; evict-first (__stcs) store when L2-resident
   L17_dmma                 one volume/block in shared, fused 3-axis cyclic/negacyclic 17-pt lines, batch-selected staging (regs vs cp.async), 1 global read + 1 write; single-launch soft-barrier plane-split at batch<=16
   L17_raderfused           17^3 volume per block in shared, 3 axes fused, 1 global read + 1 write; conj-folded cyclic/negacyclic 17-pt lines; warp-chunked cp.async z-overlap; plane+line split path below B=14
   L23_rader                L23_rader: folded-dense 23pt, 2-pass z+y|x; tuner A(P=4,T=96) u0 B0 TB=32 ns1 chunk=48/4 g0 (0.60 us/xform in-plan, nv=86)
   L36_globalpass           two-pass radix-6^2 shared s37; plain pair hints=0 chunk=0 ns=1 lead=0 graph=1
   L36_sharedtiled          two-pass plane-per-block, 6x6 CT lines, padded shared; smallB split chunk=0 ns=1 pol=stream launch=graph
   L45_pfa                  two-pass PFA 9x5: unit-parallel zy-plane + x-tile kernels; evict-first-hinted L2 chunks (9) round-robined on 2 streams; slice-per-stream + direct stores at L2-resident B; B=1 graph
   L64_radix8               two-pass plane-per-block, register radix-8^2 lines, shuffle transpose
   L6_batchcoalesced        L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 codelet, fused single kernel, last axis direct to global, __stcs when in fits L2
   L6_warpvolume            L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 lines, x-pass stores direct; stcs stores while input is L2-resident; 64-thread B=1 kernel
   L8_blockfused            fused block of V volumes in shared (row pad 8->9), min-op radix-8 per thread-line, warp-per-volume variant, autotuned
   L8_warpradix8            one volume per warp in registers, radix-8 x3, butterfly shuffle transposes, no shared/barriers
   baseline_gpu             row-column dense O(L) per output, no factorization
   cufft                    cuFFT 11.0 cufftPlanMany Z2Z, batched
