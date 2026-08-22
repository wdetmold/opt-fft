# What previous generations produced (round gpu_r1 is the current one)

## Leaderboards from earlier rounds
  /home/lqcd/wdetmold/fft/bench/gpu/results/gpu_validate/leaderboard.txt
  /home/lqcd/wdetmold/fft/bench/gpu/results/local_smoke/leaderboard.txt

## Strategy records -- every implementation's own account of what it tried
  (none yet)

## Promoted exemplars -- code kept from earlier rounds because it was worth keeping
  (none yet)

## Current standings (most recent leaderboard)
=== round gpu_validate ===
# round gpu_validate
host: a80n1.lqcd.mit   date: 2026-08-21T22:39:34-04:00   slurm_job: 438495
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

-- L=8 (batched B=64), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   cufft                          0.193 us    12.367 us   119.23       0.0%    0.396s  ok 2.4e-16       1.00x
   baseline_gpu                   0.312 us    19.962 us    73.87       0.0%    0.000s  ok 3.9e-16       1.61x

-- L=17 (batched B=8), volume 4913, working set 1.20 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   cufft                          1.899 us    15.194 us   158.60       0.0%    0.156s  ok 3.2e-16       1.00x
   baseline_gpu                   3.742 us    29.937 us    80.50       0.0%    0.000s  ok 8.4e-16       1.97x

backends:
   baseline_gpu             row-column dense O(L) per output, no factorization
   cufft                    cuFFT 11.0 cufftPlanMany Z2Z, batched
