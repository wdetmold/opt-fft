# What previous generations produced (round gpu_r2 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_validate/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/local_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L13_dmma.md 127 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_dmma.md 174 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_raderfused.md 157 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L23_rader.md 158 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_globalpass.md 107 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_sharedtiled.md 110 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L45_pfa.md 159 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L64_radix8.md 118 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_batchcoalesced.md 108 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_warpvolume.md 97 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_blockfused.md 137 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_warpradix8.md 152 lines

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  /home/lqcd/wdetmold/fft/bench/gpu/exemplars/gpu_r1/
      # Round gpu_r1 — what it established
      
      Promoted: L6_batchcoalesced L6_warpvolume L8_blockfused L8_warpradix8 L13_dmma L17_dmma L17_raderfused L23_rader L36_sharedtiled L36_globalpass L45_pfa L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round gpu_r1 ===
# round gpu_r1
host: a80n1.lqcd.mit   date: 2026-08-22T18:36:13-04:00   slurm_job: 438580
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
clocks after sweep: 210 MHz, 1410 MHz, 29, 56.14 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 55.28 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 25, 58.88 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 56.09 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 55.00 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 55.22 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 54.68 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 26, 51.90 W, 0x0000000000000001

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              3.640 us     3.640 us     2.30       1.5%    0.000s  ok 2.5e-16       1.00x
   L6_warpvolume                  5.365 us     5.365 us     1.56       0.7%    0.000s  ok 2.2e-16       1.47x
   cufft                         10.077 us    10.077 us     0.83       4.5%    0.019s  ok 2.3e-16       2.77x
   baseline_gpu                  14.904 us    14.904 us     0.56       3.6%    0.000s  ok 5.9e-16       4.09x

-- L=6 (batched B=4854), volume 216, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_warpvolume                  0.004 us    18.381 us  2211.77       0.6%    0.000s  ok 2.2e-16       1.00x
   L6_batchcoalesced              0.005 us    23.881 us  1702.34       0.4%    0.000s  ok 2.5e-16       1.30x
   cufft                          0.011 us    51.424 us   790.56       0.5%    0.015s  ok 2.5e-16       2.80x
   baseline_gpu                   0.033 us   162.282 us   250.51       0.4%    0.000s  ok 6.0e-16       8.83x

-- L=6 (batched B=310608), volume 216, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              0.005 us  1540.139 us  1689.09       0.1%    0.000s  ok 2.5e-16       1.00x
   L6_warpvolume                  0.005 us  1548.245 us  1680.24       1.1%    0.000s  ok 2.2e-16       1.01x
   cufft                          0.011 us  3523.482 us   738.31       0.1%    0.010s  ok 2.5e-16       2.29x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  3.184 us     3.184 us     7.24      10.5%    0.000s  ok 1.5e-16       1.00x
   L8_warpradix8                  3.632 us     3.632 us     6.34       6.6%    0.000s  ok 2.3e-16       1.14x
   cufft                          9.038 us     9.038 us     2.55       5.8%    0.010s  ok 2.4e-16       2.84x
   baseline_gpu                  17.469 us    17.469 us     1.32       3.0%    0.000s  ok 3.9e-16       5.49x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.006 us    13.238 us  3564.31       0.6%    0.057s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.007 us    14.024 us  3364.74       0.6%    0.086s  ok 2.3e-16       1.06x
   cufft                          0.026 us    54.205 us   870.50       0.4%    0.009s  ok 2.4e-16       4.09x
   baseline_gpu                   0.085 us   174.668 us   270.15       0.3%    0.000s  ok 3.9e-16       13.19x

-- L=8 (batched B=131072), volume 512, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.012 us  1536.768 us  1965.10       0.1%    0.304s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.012 us  1560.448 us  1935.28       0.0%    0.398s  ok 2.3e-16       1.02x
   cufft                          0.028 us  3697.920 us   816.65       0.1%    0.010s  ok 2.4e-16       2.41x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       7.875 us     7.875 us    15.49       1.9%    0.000s  ok 3.2e-16       1.00x
   cufft                         12.105 us    12.105 us    10.07       3.3%    0.009s  ok 2.8e-16       1.54x
   baseline_gpu                  22.634 us    22.634 us     5.39       2.7%    0.000s  ok 7.8e-16       2.87x

-- L=13 (batched B=477), volume 2197, working set 31.98 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.059 us    28.334 us  2052.99       0.4%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.132 us    62.941 us   924.19       0.7%    0.017s  ok 2.9e-16       2.22x
   baseline_gpu                   0.537 us   256.304 us   226.95       0.0%    0.000s  ok 7.9e-16       9.05x

-- L=13 (batched B=30549), volume 2197, working set 2048.22 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.051 us  1563.307 us  2383.02       0.0%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.155 us  4742.784 us   785.49       0.1%    0.010s  ok 2.9e-16       3.03x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_raderfused                 9.760 us     9.760 us    30.86       2.4%    0.000s  ok 3.1e-16       1.00x
   L17_dmma                      11.357 us    11.357 us    26.52       0.3%    0.000s  ok 3.6e-16       1.16x
   cufft                         13.474 us    13.474 us    22.36       3.6%    0.010s  ok 3.2e-16       1.38x
   baseline_gpu                  27.062 us    27.062 us    11.13       1.6%    0.000s  ok 8.4e-16       2.77x

-- L=17 (batched B=213), volume 4913, working set 31.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_raderfused                 0.149 us    31.690 us  2024.64       0.5%    0.000s  ok 3.2e-16       1.00x
   L17_dmma                       0.168 us    35.815 us  1791.48       0.3%    0.000s  ok 3.1e-16       1.13x
   cufft                          0.316 us    67.314 us   953.16       0.5%    0.015s  ok 3.2e-16       2.12x
   baseline_gpu                   1.450 us   308.810 us   207.77       0.3%    0.000s  ok 8.4e-16       9.74x

-- L=17 (batched B=13660), volume 4913, working set 2048.08 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       0.116 us  1584.725 us  2596.50       0.1%    0.000s  ok 3.1e-16       1.00x
   L17_raderfused                 0.118 us  1610.659 us  2554.69       0.0%    0.000s  ok 3.2e-16       1.02x
   cufft                          0.348 us  4759.680 us   864.50       0.0%    0.015s  ok 3.2e-16       3.00x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     10.556 us    10.556 us    78.21       1.0%    0.007s  ok 3.6e-16       1.00x
   cufft                         14.865 us    14.865 us    55.54       1.6%    0.016s  ok 3.8e-16       1.41x
   baseline_gpu                  32.268 us    32.268 us    25.59       1.0%    0.000s  ok 7.4e-16       3.06x

-- L=23 (batched B=86), volume 12167, working set 31.93 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.603 us    51.842 us  1369.53       0.6%    0.059s  ok 3.7e-16       1.00x
   cufft                          0.879 us    75.611 us   939.01       1.1%    0.016s  ok 3.8e-16       1.46x
   baseline_gpu                   4.548 us   391.095 us   181.54       0.2%    0.000s  ok 7.4e-16       7.54x

-- L=23 (batched B=5515), volume 12167, working set 2047.76 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.590 us  3252.053 us  1400.05       4.4%    2.598s  ok 3.6e-16       1.00x
   cufft                          0.919 us  5066.752 us   898.61       0.1%    0.013s  ok 3.8e-16       1.56x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_sharedtiled               10.069 us    10.069 us   359.34       0.1%    0.042s  ok 4.8e-16       1.00x
   L36_globalpass                10.490 us    10.490 us   344.90       1.0%    0.000s  ok 4.8e-16       1.04x
   cufft                         12.944 us    12.944 us   279.53       5.5%    0.017s  ok 4.3e-16       1.29x
   baseline_gpu                  45.837 us    45.837 us    78.93       1.3%    0.000s  ok 8.0e-16       4.55x

-- L=36 (batched B=22), volume 46656, working set 31.32 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_globalpass                 1.745 us    38.390 us  2073.41       0.5%    0.000s  ok 4.8e-16       1.00x
   L36_sharedtiled                1.853 us    40.755 us  1953.09       0.1%    0.102s  ok 4.8e-16       1.06x
   cufft                          2.308 us    50.768 us  1567.88       0.8%    0.018s  ok 4.3e-16       1.32x
   baseline_gpu                  23.326 us   513.162 us   155.11       0.8%    0.000s  ok 8.0e-16       13.37x

-- L=36 (batched B=1438), volume 46656, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_sharedtiled                1.495 us  2150.172 us  2419.74       0.2%    0.423s  ok 4.8e-16       1.00x
   L36_globalpass                 1.534 us  2205.403 us  2359.14       0.5%    0.000s  ok 4.8e-16       1.03x
   cufft                          2.364 us  3399.040 us  1530.68       0.1%    0.015s  ok 4.3e-16       1.58x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       14.340 us    14.340 us   523.49       1.7%    0.000s  ok 8.2e-16       1.00x
   cufft                         18.010 us    18.010 us   416.81       1.1%    0.016s  ok 4.4e-16       1.26x
   baseline_gpu                  72.458 us    72.458 us   103.60       0.6%    0.000s  ok 8.0e-16       5.05x

-- L=45 (batched B=11), volume 91125, working set 30.59 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        3.870 us    42.568 us  1939.78       1.5%    0.000s  ok 8.2e-16       1.00x
   cufft                          5.832 us    64.153 us  1287.13       0.6%    0.017s  ok 4.4e-16       1.51x
   baseline_gpu                  55.846 us   614.301 us   134.42       0.3%    0.000s  ok 8.0e-16       14.43x

-- L=45 (batched B=736), volume 91125, working set 2046.75 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        4.559 us  3355.341 us  1646.60       0.1%    0.000s  ok 8.2e-16       1.00x
   cufft                          6.452 us  4748.544 us  1163.50       0.0%    0.014s  ok 4.4e-16       1.42x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    22.581 us    22.581 us  1044.82       0.8%    0.000s  ok 4.2e-16       1.00x
   cufft                         23.487 us    23.487 us  1004.51       1.0%    0.016s  ok 4.0e-16       1.04x
   baseline_gpu                 207.223 us   207.223 us   113.85       0.9%    0.000s  ok 7.8e-16       9.18x

-- L=64 (batched B=4), volume 262144, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    12.110 us    48.441 us  1948.17       0.5%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.484 us    57.937 us  1628.87       0.8%    0.018s  ok 4.0e-16       1.20x
   baseline_gpu                 205.724 us   822.895 us   114.68       0.2%    0.000s  ok 7.8e-16       16.99x

-- L=64 (batched B=256), volume 262144, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    11.673 us  2988.237 us  2021.19       0.1%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.610 us  3740.160 us  1614.85       0.1%    0.010s  ok 4.0e-16       1.25x

backends:
   L13_dmma                 fused block-per-volume in shared, conj-symmetric folded dense-13 per thread-line
   L17_dmma                 one volume/block in shared, fused 3-axis conj-folded dense 17-pt DFT, cp.async staged, 1 global read + 1 write; coop plane-split at batch<=4
   L17_raderfused           17^3 volume per block in shared, 3 axes fused, 1 global read + 1 write; conj-folded cyclic/negacyclic 17-pt lines; plane+line split path below B=12
   L23_rader                L23_rader: folded-dense 23pt, 2-pass z+y|x; tuner coarse A(P=1,T=32) TB=32 chunk=86 ul=0 (0.71 us/xform in-plan, nv=86)
   L36_globalpass           two-pass plane-per-block radix-6^2, padded shared stride 37, L2-chunked batch
   L36_sharedtiled          two-pass plane-per-block, 6x6 CT lines, padded shared; L2 chunk=12 streams=3 scratch=0 graph=1
   L45_pfa                  two-pass PFA 9x5: unit-parallel zy-plane + x-tile kernels, L2-chunked dual-stream pipeline, direct-store variants for L2-resident batches
   L64_radix8               two-pass plane-per-block, register radix-8^2 lines, shuffle transpose
   L6_batchcoalesced        L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 codelet, fused single kernel, last axis writes direct to global
   L6_warpvolume            L6: 4 volumes/warp all-register, DIF-z/DIT 2x3 parity split, shuffle butterflies, zero shared memory, one kernel
   L8_blockfused            fused block of V volumes in shared (row pad 8->9), min-op radix-8 per thread-line, V autotuned
   L8_warpradix8            one volume per warp in registers, radix-8 x3, butterfly shuffle transposes, no shared/barriers
   baseline_gpu             row-column dense O(L) per output, no factorization
   cufft                    cuFFT 11.0 cufftPlanMany Z2Z, batched
