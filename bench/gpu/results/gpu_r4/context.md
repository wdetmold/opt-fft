# What previous generations produced (round gpu_r4 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r1/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r2/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_r3/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_validate/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/local_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L13_dmma.md 344 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_dmma.md 405 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L17_raderfused.md 396 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L23_rader.md 293 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_globalpass.md 221 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L36_sharedtiled.md 314 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L45_pfa.md 406 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L64_radix8.md 356 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_batchcoalesced.md 273 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L6_warpvolume.md 307 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_blockfused.md 239 lines
  /home/lqcd/wdetmold/fft/bench/gpu/strategies/L8_warpradix8.md 392 lines

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
      
  /home/lqcd/wdetmold/fft/bench/gpu/exemplars/gpu_r3/
      # Round gpu_r3 — what it established
      
      Promoted: L6_batchcoalesced L8_warpradix8 L13_dmma L17_raderfused L17_dmma L36_sharedtiled L45_pfa L64_radix8
      
      ## Result
      
      (Fill in: the leaderboard headline per geometry, panel best vs best library.)
      
      ## What this round settled
      
      (Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)
      

## Current standings (most recent leaderboard)
=== round gpu_r3 ===
# round gpu_r3
host: a80n1.lqcd.mit   date: 2026-08-22T21:23:00-04:00   slurm_job: 438580
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
clocks after sweep: 210 MHz, 1410 MHz, 25, 58.82 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 56.03 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 27, 54.68 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 55.17 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 24, 54.68 W, 0x0000000000000001
clocks after sweep: 210 MHz, 1410 MHz, 26, 51.90 W, 0x0000000000000001

-- L=6 (non-batched), volume 216, working set 0.01 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              2.832 us     2.832 us     2.96       7.5%    0.000s  ok 2.6e-16       1.00x
   L6_warpvolume                  3.064 us     3.064 us     2.73       4.8%    0.000s  ok 2.6e-16       1.08x
   cufft                         10.317 us    10.317 us     0.81       1.2%    0.013s  ok 2.6e-16       3.64x
   baseline_gpu                  14.839 us    14.839 us     0.56       2.0%    0.000s  ok 5.9e-16       5.24x

-- L=6 (batched B=4854), volume 216, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_batchcoalesced              0.003 us    14.599 us  2784.76       0.8%    0.000s  ok 2.5e-16       1.00x
   L6_warpvolume                  0.003 us    14.876 us  2732.88       0.5%    0.000s  ok 2.5e-16       1.02x
   cufft                          0.011 us    51.632 us   787.38       0.1%    0.014s  ok 2.5e-16       3.54x
   baseline_gpu                   0.034 us   163.301 us   248.95       0.1%    0.000s  ok 6.0e-16       11.19x

-- L=6 (batched B=310608), volume 216, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_warpvolume                  0.005 us  1538.475 us  1690.91       0.1%    0.000s  ok 2.5e-16       1.00x
   L6_batchcoalesced              0.005 us  1540.395 us  1688.81       0.1%    0.000s  ok 2.5e-16       1.00x
   cufft                          0.011 us  3519.181 us   739.21       0.2%    0.010s  ok 2.5e-16       2.29x

-- L=8 (non-batched), volume 512, working set 0.02 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_warpradix8                  3.055 us     3.055 us     7.54       7.7%    0.009s  ok 2.3e-16       1.00x
   L8_blockfused                  3.245 us     3.245 us     7.10       2.8%    0.000s  ok 1.8e-16       1.06x
   cufft                          9.075 us     9.075 us     2.54       3.3%    0.014s  ok 2.5e-16       2.97x
   baseline_gpu                  18.015 us    18.015 us     1.28       0.7%    0.000s  ok 3.9e-16       5.90x

-- L=8 (batched B=2048), volume 512, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.006 us    13.165 us  3584.07       0.4%    0.115s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.006 us    13.176 us  3581.12       0.5%    0.148s  ok 2.3e-16       1.00x
   cufft                          0.027 us    54.568 us   864.72       0.6%    0.018s  ok 2.4e-16       4.14x
   baseline_gpu                   0.085 us   175.071 us   269.53       0.1%    0.000s  ok 3.9e-16       13.30x

-- L=8 (batched B=131072), volume 512, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_blockfused                  0.012 us  1537.195 us  1964.55       0.1%    0.615s  ok 1.6e-16       1.00x
   L8_warpradix8                  0.012 us  1538.645 us  1962.70       0.0%    1.215s  ok 2.3e-16       1.00x
   cufft                          0.028 us  3696.640 us   816.93       0.1%    0.015s  ok 2.4e-16       2.40x

-- L=13 (non-batched), volume 2197, working set 0.07 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       6.819 us     6.819 us    17.88       1.5%    0.000s  ok 4.0e-16       1.00x
   cufft                         12.091 us    12.091 us    10.09       2.9%    0.012s  ok 2.9e-16       1.77x
   baseline_gpu                  22.637 us    22.637 us     5.39       0.9%    0.000s  ok 7.9e-16       3.32x

-- L=13 (batched B=477), volume 2197, working set 31.98 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.045 us    21.511 us  2704.12       0.8%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.132 us    62.812 us   926.08       0.3%    0.019s  ok 2.9e-16       2.92x
   baseline_gpu                   0.537 us   255.917 us   227.30       0.2%    0.000s  ok 7.9e-16       11.90x

-- L=13 (batched B=30549), volume 2197, working set 2048.22 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_dmma                       0.051 us  1563.051 us  2383.41       0.0%    0.000s  ok 3.2e-16       1.00x
   cufft                          0.155 us  4740.352 us   785.89       0.0%    0.014s  ok 2.9e-16       3.03x

-- L=17 (non-batched), volume 4913, working set 0.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       1.743 us     1.743 us   172.86      38.5%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 7.781 us     7.781 us    38.71       2.5%    0.000s  ok 3.1e-16       4.47x
   cufft                         13.464 us    13.464 us    22.37       1.8%    0.010s  ok 3.3e-16       7.73x
   baseline_gpu                  26.862 us    26.862 us    11.21       2.1%    0.000s  ok 8.4e-16       15.41x

-- L=17 (batched B=213), volume 4913, working set 31.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       0.093 us    19.738 us  3250.58       0.1%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 0.146 us    31.185 us  2057.45       0.4%    0.000s  ok 3.2e-16       1.58x
   cufft                          0.316 us    67.354 us   952.59       0.6%    0.016s  ok 3.2e-16       3.41x
   baseline_gpu                   1.452 us   309.256 us   207.47       0.2%    0.000s  ok 8.4e-16       15.67x

-- L=17 (batched B=13660), volume 4913, working set 2048.08 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_dmma                       0.114 us  1551.872 us  2651.47       0.8%    0.000s  ok 3.2e-16       1.00x
   L17_raderfused                 0.116 us  1577.899 us  2607.73       0.1%    0.000s  ok 3.2e-16       1.02x
   cufft                          0.348 us  4759.424 us   864.55       0.0%    0.014s  ok 3.2e-16       3.07x

-- L=23 (non-batched), volume 12167, working set 0.37 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      9.101 us     9.101 us    90.71       1.3%    0.043s  ok 3.5e-16       1.00x
   cufft                         14.673 us    14.673 us    56.26       2.8%    0.010s  ok 3.8e-16       1.61x
   baseline_gpu                  32.248 us    32.248 us    25.60       0.8%    0.000s  ok 7.3e-16       3.54x

-- L=23 (batched B=86), volume 12167, working set 31.93 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.454 us    39.033 us  1818.95       4.3%    0.267s  ok 3.5e-16       1.00x
   cufft                          0.881 us    75.783 us   936.88       0.8%    0.017s  ok 3.8e-16       1.94x
   baseline_gpu                   4.551 us   391.346 us   181.42       0.1%    0.000s  ok 7.4e-16       10.03x

-- L=23 (batched B=5515), volume 12167, working set 2047.76 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                      0.413 us  2275.072 us  2001.27       5.5%   10.983s  ok 3.5e-16       1.00x
   cufft                          0.917 us  5059.413 us   899.91       0.2%    0.010s  ok 3.8e-16       2.22x

-- L=36 (non-batched), volume 46656, working set 1.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_sharedtiled                9.062 us     9.062 us   399.28       0.5%    0.456s  ok 4.8e-16       1.00x
   L36_globalpass                 9.129 us     9.129 us   396.31       0.9%    0.358s  ok 4.8e-16       1.01x
   cufft                         12.908 us    12.908 us   280.29       5.2%    0.009s  ok 4.3e-16       1.42x
   baseline_gpu                  45.907 us    45.907 us    78.81       1.1%    0.000s  ok 8.1e-16       5.07x

-- L=36 (batched B=22), volume 46656, working set 31.32 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_globalpass                 1.305 us    28.711 us  2772.45       2.1%    0.988s  ok 4.8e-16       1.00x
   L36_sharedtiled                1.306 us    28.738 us  2769.84       3.3%    0.911s  ok 4.8e-16       1.00x
   cufft                          2.317 us    50.984 us  1561.24       0.7%    0.016s  ok 4.3e-16       1.78x
   baseline_gpu                  23.387 us   514.506 us   154.71       0.5%    0.000s  ok 8.0e-16       17.92x

-- L=36 (batched B=1438), volume 46656, working set 2047.46 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_sharedtiled                1.343 us  1931.520 us  2693.66       0.4%    0.672s  ok 4.8e-16       1.00x
   L36_globalpass                 1.351 us  1942.118 us  2678.96       0.0%    0.456s  ok 4.8e-16       1.01x
   cufft                          2.363 us  3397.530 us  1531.36       0.2%    0.016s  ok 4.3e-16       1.76x

-- L=45 (non-batched), volume 91125, working set 2.78 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                       12.626 us    12.626 us   594.54       0.4%    0.000s  ok 8.2e-16       1.00x
   cufft                         18.013 us    18.013 us   416.74       1.4%    0.009s  ok 4.4e-16       1.43x
   baseline_gpu                  72.551 us    72.551 us   103.47       1.2%    0.000s  ok 8.0e-16       5.75x

-- L=45 (batched B=11), volume 91125, working set 30.59 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        3.266 us    35.922 us  2298.68       1.9%    0.000s  ok 8.2e-16       1.00x
   cufft                          5.861 us    64.472 us  1280.77       0.1%    0.018s  ok 4.4e-16       1.79x
   baseline_gpu                  55.886 us   614.747 us   134.32       0.6%    0.000s  ok 8.0e-16       17.11x

-- L=45 (batched B=736), volume 91125, working set 2046.75 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_pfa                        2.867 us  2110.272 us  2618.11       0.3%    0.000s  ok 8.2e-16       1.00x
   cufft                          6.441 us  4740.864 us  1165.38       0.2%    0.010s  ok 4.4e-16       2.25x

-- L=64 (non-batched), volume 262144, working set 8.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    18.501 us    18.501 us  1275.20       0.2%    0.000s  ok 4.2e-16       1.00x
   cufft                         23.444 us    23.444 us  1006.36       1.2%    0.013s  ok 4.0e-16       1.27x
   baseline_gpu                 207.540 us   207.540 us   113.68       0.8%    0.000s  ok 7.8e-16       11.22x

-- L=64 (batched B=4), volume 262144, working set 32.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                     9.904 us    39.616 us  2382.18       2.9%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.571 us    58.283 us  1619.21       0.0%    0.009s  ok 4.0e-16       1.47x
   baseline_gpu                 205.379 us   821.515 us   114.88       0.1%    0.000s  ok 7.8e-16       20.74x

-- L=64 (batched B=256), volume 262144, working set 2048.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L64_radix8                    10.212 us  2614.357 us  2310.24       0.3%    0.000s  ok 4.2e-16       1.00x
   cufft                         14.611 us  3740.416 us  1614.74       0.2%    0.010s  ok 4.0e-16       1.43x

backends:
   L13_dmma                 fused block-per-volume in shared, conj-folded dense-13 lines; __stcs store when L2-resident; soft-barrier single-launch plane-split at batch<=8
   L17_dmma                 one volume/block in shared, fused 3-axis cyclic/negacyclic 17-pt lines, 1 global read + 1 write; async execute round-robined over 8 streams so back-to-back calls pipeline (driver syncs per sample)
   L17_raderfused           17^3 volume per block in shared, 3 axes fused, 1 global read + 1 write; conj-folded cyclic/negacyclic 17-pt lines; batch-picked staging (warp reg vs cp.async); soft-barrier single-launch zy-plane + folded-x split below B=13
   L23_rader                L23_rader: folded-dense 23pt, 2-pass z+y|x; tuner Af(P=1,T=32) u0 B3 TB=256 ns0 chunk=1/4 g1 (16.38 us/xform in-plan, nv=1)
   L36_globalpass           two-pass radix-6^2 shared s37; split/chunked rr streams hints=1 chunk=6 ns=4 lead=0 graph=0 pf=1
   L36_sharedtiled          6x6 CT lines padded shared; persistent ticket lead=14 chunk=0 ns=1 pol=2 launch=plain
   L45_pfa                  two-pass PFA 9x5: persistent producer/consumer ticket kernel at HBM B (45 planes + 64 x-tiles/vol, lead-10 runway, direct-out bodies, evict-first hints); slice-per-stream + direct stores at L2-resident B; B=1 graph
   L64_radix8               two-pass plane-per-block, register radix-8^2 lines, shuffle transpose
   L6_batchcoalesced        L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 codelet, fused single kernel, last axis direct to global, __stcs when in fits L2; fused-z 36-thread single-volume kernel at B=1
   L6_warpvolume            L6: 8 volumes/block batch-major swizzled shared, DIT 2x3 lines, x-pass stores direct; stcs stores while input is L2-resident; 64-thread B=1 kernel; r3: barrier/load variants measured, all lost, r2 config kept
   L8_blockfused            fused block of V volumes in shared (row pad 8->9), min-op radix-8 per thread-line, warp-per-volume variant, autotuned
   L8_warpradix8            volume per warp/pair/quad in registers, cross-lane DIF radix-8, measured autotune, graph-replayed launch
   baseline_gpu             row-column dense O(L) per output, no factorization
   cufft                    cuFFT 11.0 cufftPlanMany Z2Z, batched
