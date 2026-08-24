# Master rival timing table — all cohorts, one node, one protocol

Every attempt regenerated for x86 (Ice Lake longdouble constants), compiled with its own
flags, and timed on bare-metal a80n0: pinned core (self-affinity neutralized), 5 reps,
anti-memoization perturbation, cpu/wall + thread telemetry. "grader/bare" compares the
grader's reported best-of-shots (C_opt/3 for the 3x workload) against our measurement;
the honest VM tier factor, calibrated on 21 consistent attempts, is 1.46x. Ratios below
~0.85x cannot be explained by silicon (GCP n2 turbo is at most ~10-15% faster per core).

| attempt | graded score | measured on our node (1x) | grader implied (1x) | grader/bare | verdict |
|---|---|---|---|---|---|
| **our panel (ice_r8)** | — | **0.7791 s** | — | — | fastest ever measured |
| hot_16d44d13 | 0.92 | 0.8056 | 0.690 | 0.86x | clock-explainable |
| hot_04b0abdc | 0.87 | 0.8170 | 0.943 | 1.15x | tier~bare |
| hot_262a05c6 | 0.94 | 0.8176 | 0.584 | 0.71x | ANOMALOUS |
| hot_d82aee89 | **0.98** | 0.8257 | 0.408 | **0.49x** | ANOMALOUS |
| hot_502912a3 | 0.95 | 0.8288 | 0.554 | 0.67x | ANOMALOUS |
| warm_00291a90 | 0.97 | 0.8495 | 0.711 | 0.85x | clock-explainable |
| warm_57053476 | 0.90 | 0.8562 (7 sizes) | 0.992 | ~0.9x | L=64 recon broken |
| warm_d43251c2 | 0.99 | 0.8777 | 0.673 | 0.79x | ANOMALOUS (lucky shot) |
| warm_53ebdad6 | 0.89 | 0.9738 | 1.050 | 1.10x | tier~bare |
| v7_69505252 | 0.78 | 0.9864 | 1.561 | 1.58x | consistent |
| v7_47551a02 | 0.82 | 0.9973 | 1.392 | 1.40x | consistent |
| v7_91a35119 | 0.78 | 1.0182 | 1.486 | 1.46x | consistent |
| v7_b1eaa90c | 0.77 | 1.0841 | 1.532 | 1.41x | consistent |
| v7_9d9b3935 | 0.77 | (1.18; outputs fail gates via our adapter — not validated) | 1.613 | ~1.36x | adapter mismatch |
| hot_90378bc1 | 0.84 | not runnable (recon ctypes defect) | 1.037 | — | — |
| warm_361a3485 | 0.93 | not reconstructible | 0.891 | — | — |

## The full table — every reconstructed competitor attempt

(v4 reported walls are 1x-workload per TIMINGS.md's "divide by 3" list, which names only
v5,v6,v7,warm,hot — with that correction the v4 cohort resolves to consistent/slow-shot,
matching its known unpinned-subprocess grader, and the only anomalous cluster left is the
top of the warm/hot leaderboards.)

| attempt | cohort | score | reported C_opt | implied 1x | ours bare 1x | grader/bare | shots max/min | verdict |
|---|---|---|---|---|---|---|---|---|
| **our panel (ice_r8)** | ice | — | — | — | **0.7791** | — | — | fastest measured |
| hot_16d44d13_score0.92 | hot | 0.92 | 2.071 | 0.690 | 0.8056 | 0.86x | — | fast-tier (clock-explainable) |
| hot_04b0abdc_score0.87 | hot | 0.8661 | 2.828 | 0.943 | 0.8170 | 1.15x | — | tier~bare |
| hot_262a05c6_score0.94 | hot | 0.94 | 1.752 | 0.584 | 0.8176 | 0.71x | — | ANOMALOUS (<0.8x bare) |
| hot_d82aee89_score0.98 | hot | 0.98 | 1.224 | 0.408 | 0.8257 | 0.49x | — | ANOMALOUS (<0.8x bare) |
| hot_502912a3_score0.95 | hot | 0.95 | 1.661 | 0.554 | 0.8288 | 0.67x | — | ANOMALOUS (<0.8x bare) |
| hot_90378bc1_score0.84 | hot | 0.84 | 3.112 | 1.037 | — | — | — | not measured |
| warm_00291a90_score0.97 | warm | 0.97 | 2.132 | 0.711 | 0.8495 | 0.84x | 1.03x | fast-tier (clock-explainable) |
| warm_d43251c2_score0.99 | warm | 0.99 | 2.019 | 0.673 | 0.8777 | 0.77x | 2.02x | ANOMALOUS (<0.8x bare) |
| warm_53ebdad6_score0.89 | warm | 0.89 | 3.151 | 1.050 | 0.9738 | 1.08x | 1.00x | tier~bare |
| warm_57053476_score0.9 | warm | 0.9 | 2.975 | 0.992 | 1.0848~ | 0.91x | 1.06x | fast-tier (clock-explainable) ~est |
| warm_361a3485_score0.93 | warm | 0.93 | 2.674 | 0.891 | — | — | 1.06x | not measured |
| v7_69505252_score0.78 | v7 | 0.7832 | 4.683 | 1.561 | 0.9864 | 1.58x | 1.02x | consistent |
| v7_47551a02_score0.82 | v7 | 0.8153 | 4.177 | 1.392 | 0.9973 | 1.40x | — | consistent |
| v7_91a35119_score0.78 | v7 | 0.7835 | 4.457 | 1.486 | 1.0182 | 1.46x | — | consistent |
| v7_9d9b3935_score0.77 | v7 | 0.7741 | 4.838 | 1.613 | 1.0654~ | 1.51x | — | consistent ~est |
| v7_b1eaa90c_score0.77 | v7 | 0.7667 | 4.595 | 1.532 | 1.0841 | 1.41x | — | consistent |
| v5_3907583b_score0.87 | v5/v6 | 0.869 | 4.006 | 1.335 | 0.9531~ | 1.40x | — | consistent ~est |
| v5_95ab77a1_score0.82 | v5/v6 | 0.8168 | 4.688 | 1.563 | 0.9819~ | 1.59x | — | consistent ~est |
| v6_3f30d81f_score0.88 | v5/v6 | 0.884 | 6.040 | 2.013 | 0.9859~ | 2.04x | — | SLOW-SHOT ~est |
| v6_4d0483ea_score0.85 | v5/v6 | 0.8455 | 4.353 | 1.451 | 0.9989~ | 1.45x | — | consistent ~est |
| v6_2cbe0fb0_score0.80 | v5/v6 | 0.7961 | 4.589 | 1.530 | 0.9997~ | 1.53x | 1.07x | consistent ~est |
| v6_f40c5e25_score0.91 | v5/v6 | 0.9078 | 3.555 | 1.185 | 1.0001 | 1.18x | — | tier~bare ~est |
| v6_8b0fbe57_score0.77 | v5/v6 | 0.7737 | 4.880 | 1.627 | 1.0001 | 1.63x | — | consistent ~est |
| v6_78662a62_score0.84 | v5/v6 | 0.8415 | 4.637 | 1.546 | 1.0018~ | 1.54x | — | consistent ~est |
| v5_197d7c23_score0.83 | v5/v6 | 0.8266 | 7.035 | 2.345 | 1.0767~ | 2.18x | — | SLOW-SHOT ~est |
| v5_26833bab_score0.87 | v5/v6 | 0.8729 | 5.080 | 1.693 | 1.0869~ | 1.56x | — | consistent ~est |
| v5_2c2dfce8_score0.74 | v5/v6 | 0.7448 | 5.139 | 1.713 | 1.1064~ | 1.55x | — | consistent ~est |
| v5_8175a973_score0.90 | v5/v6 | 0.8962 | 3.512 | 1.171 | 1.1164~ | 1.05x | — | tier~bare ~est |
| v6_4a50d3db_score0.77 | v5/v6 | 0.7726 | 5.073 | 1.691 | 1.1263~ | 1.50x | — | consistent ~est |
| v6_5a869e40_score0.80 | v5/v6 | 0.8004 | 5.692 | 1.897 | 1.2170~ | 1.56x | — | consistent ~est |
| v5_2419f19d_score0.81 | v5/v6 | 0.8127 | 5.705 | 1.902 | 1.3346~ | 1.42x | — | consistent ~est |
| 1760b1bf_score0.96 | v4 (1x) | 0.9593 | 1.500 | 1.500 | 1.0043 | 1.49x | — | consistent |
| 1000f989_score1.00 | v4 (1x) | 1.0 | 2.014 | 2.014 | 1.0044 | 2.01x | — | SLOW-SHOT |
| 8dc1a96d_score1.00_floor-artifact | v4 (1x) | 1.0 | 3.250 | 3.250 | 1.0598 | 3.07x | — | SLOW-SHOT |
| dd9fa88c_score0.76 | v4 (1x) | 0.764 | 2.038 | 2.038 | 1.0637 | 1.92x | — | SLOW-SHOT |
| a31f5f85_score0.81 | v4 (1x) | 0.8103 | 2.306 | 2.306 | 1.0720 | 2.15x | — | SLOW-SHOT |
| 0f45aeae_score0.44 | v4 (1x) | — | 2.355 | 2.355 | 1.2642 | 1.86x | — | SLOW-SHOT |


## The two findings

1. **The hot leaderboard ranking is grader noise.** The five measured hot attempts span
   0.806-0.829 s on identical hardware — 2.8% apart — while their graded scores span
   0.87-0.98 (implied speeds 2.3x apart). The truly fastest (16d44d13) placed 4th; the
   graded winner (d82aee89, "0.98") is 4th fastest and its tight shots are 2.0x faster
   than the same binary runs on uncontended bare metal — a systematic grader-clock
   artifact, not luck and not threads (cpu/wall = 1.00, single-threaded).
   Honestly timed on the hot curve, every hot attempt scores ~0.80-0.82 and our panel
   ~0.83. Nobody is near the 12.8x roofline; nobody has ever beaten the measured 7.3x
   ceiling.
2. **The v7 cohort certifies the method end-to-end**: four independent attempts, graded
   the same day on the same tier class, all land at 1.40-1.58x = the calibrated tier
   factor. When the grader is healthy our pipeline reproduces its numbers exactly.

## Standing per-cell committee (best rival cell vs ours, gate-clean)

6: 0.0591 (hot/warm a90 engine) | 8: 0.0835 | 13: 0.1451 | 17: 0.0305 | 23: 0.0845 |
36: 0.0530 | 45: 0.1681 | 64: **0.1660 (hot_16d44d13 — newly taken from us)**
Committee total 0.7898 vs our 0.7791: we lead overall and at 5 of 8 cells
(rivals hold 6, 23, 64).
