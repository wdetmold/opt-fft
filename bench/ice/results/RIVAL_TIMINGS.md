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
