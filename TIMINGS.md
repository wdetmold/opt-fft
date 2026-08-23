# FFT single-core study — complete graded-timing ledger

Ground-truth table for the bare-metal Ice Lake benchmarking campaign: every run of
every round of the single-core FFT code-opt study, with the grader-reported costs,
all per-shot walls, dispersion, guard events, and the forensic-audit corrections.
Machine-readable copies: `TIMINGS.csv` (one row per run) and `fft_timing_ledger.json`
(adds round metadata + truth anchors). Purpose: rebuild the collected kernels
(`fft_v4_solutions/`, `fft_v5v6_solutions/`, `fft_warm_solutions/`, `fft_hot_solutions/`)
on uncontended hardware and compare against these graded numbers to characterize the
Taiga grader's timing behaviour.

## How grading measured time

Best-of-3 shots; each shot = wall(run) - wall(zero-work baseline), floored at 1e-6 s;
the MINIMUM shot wins. Two artifact classes are now documented on this pipeline:
- **min-shot fluke**: a stall in one shot's baseline inflates the subtraction; signature
  = min/median well below 1 (flagged at < 0.8).
- **stable over-subtraction**: all shots uniformly too fast versus the attempt's own
  final self-benchmark; signature = graded C_opt irreconcilable with the attempt's last
  logged bench (and in one case graded walls exactly equalled the attempt's 1x walls).
Audited prevalence: 1/8 (warm) rising to 4/6 completions (hot). Affected attempts share
heavy import-time mallopt/arena-prefault/warmup patterns. Raw run+baseline walls are not
recorded by the engine (grading.json keeps only subtracted walls) — the central thing
this benchmark campaign can quantify from the outside.

## Truth anchors

- **controlled_interleaved_MKL_3x**: 13.39
- **bestcheck_v14_sota_walls**: [13.6, 13.4, 14.2]
- **bestcheck_v14_graded_ratio**: 0.7048
- **controlled_truth_ratio**: 0.692
- **roofline_issue_cap_s**: 1.83
- **roofline_ISA_ideal_s**: 1.05
- **per_1x**: divide 3x-workload walls by 3 (rounds v5,v6,v7,warm,hot); v4 used a smaller workload (C_sota~3.6) and needs its own normalization
- **bare_metal_family_per_1x**: 0.84-0.96 s (four runnable warm attempts, uncontended Ice Lake)

## Rounds

| round | date | tier | score curve | workload | notes |
|---|---|---|---|---|---|
| v1 | 2026-08-21 | default | full marks at 3x (r<=1/3) | 3x chains | All 8 zeroed pre-grade: pip --target silently dropped MKL .so in grading subprocesses (provisioning bug, fixed in later rounds). |
| v2 | 2026-08-21 | default | full marks at 3x | 3x chains | Provisioning-fix iteration; 4 completed at 0.0, 4 failed. Grading pipeline not yet trustworthy. |
| v3 | 2026-08-21 | default | full marks at 3x | 3x chains | All failed; pipeline iteration era. |
| v4 | 2026-08-21/22 | default (bursty CPU steal, 2-5x timing noise) | 0.33 at parity, full marks at 3x | SMALLER workload (median C_sota 3.6 s vs 13-14 s for v5+): NOT directly comparable to the 3x rounds | First cohort with real grades. NO plausibility guards yet: one attempt graded with all-zero opt walls (floored) and scored 1.0. Dispersion-suspect winners flagged below. |
| v5 | 2026-08-22 | +perf (low variance) | 0.1 parity -> 1.0 at r<=1/6 | 3x chains | Pre-staged hash-verified MKL runtime; guards active (C_opt<1 raise); flat 1e-3 chain gate. |
| v6 | 2026-08-23 | +perf | 0.1 parity -> 1.0 at r<=1/6 | 3x chains | Per-size chain gates {L6 1e-4, L8 3e-6, L13 1e-9, L17-64 1e-10}. Null A/B vs v5 showed tightening cost honest solvers nothing. |
| v7 | 2026-08-23 | +perf | 0.1 parity -> 1.0 at ROOFLINE r<=0.137 (7.3x) | 3x chains | Measured-issue-cap roofline anchor. 3/8 env-failed on the C_opt guard (import-preamble timing floors). |
| warm | 2026-08-23 | +perf | same as v7 (r*=0.137) | 3x chains | v7 + /work/prior_work (16 cold solutions + notes). AUDITED: both top scores are grading artifacts (see corrections). |
| hot | 2026-08-23 | +perf | 0.1 parity -> 1.0 at ISA-IDEAL r<=0.078 (12.8x) | 3x chains | Both prior rounds provided; new algorithms invited. AUDITED: top 4 graded scores are grading artifacts (3 stable over-subtraction, 1 min-shot). |

## All runs

### v1 — no graded timings (see round notes)

### v2

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `82018333` | completed | 0.000 | — | — | — | — | — |  |
| `0117d783` | completed | 0.000 | — | — | — | — | — |  |
| `280aec11` | completed | 0.000 | — | — | — | — | — |  |
| `84411d70` | completed | 0.000 | — | — | — | — | — |  |
| `b0342554` | failed | — | — | — | — | — | — |  |
| `4b6a80a4` | failed | — | — | — | — | — | — |  |
| `1315849c` | failed | — | — | — | — | — | — |  |
| `8a6022df` | failed | — | — | — | — | — | — |  |

### v3 — no graded timings (see round notes)

### v4

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `1000f989` | completed | 1.000 | 288.929 | 6.863 | 2.014 | 2.6203;3.0548;2.0137 | 0.768 | graded 1.0 off min shot 2.014 with m/m 0.768 (other shots 2.62/3.05) — dispersion-suspect, unaudited |
| `8dc1a96d` | completed | 1.000 | 274.812 | 3.254 | 0.000 | 0.0;0.0;0.0 | — | graded 1.0 with ALL-ZERO opt walls (floored subtraction) — pre-guard era, plainly invalid |
| `1760b1bf` | completed | 0.960 | 283.479 | 4.013 | 1.500 | 2.0103;2.9472;1.4999 | 0.746 | graded 0.96 off min shot 1.500 with m/m 0.746 (other shots 2.01/2.95) — dispersion-suspect, unaudited |
| `a31f5f85` | completed | 0.810 | 296.404 | 4.417 | 2.306 | 3.3496;2.3064;2.5796 | 0.894 |  |
| `dd9fa88c` | completed | 0.760 | 262.658 | 3.588 | 2.038 | 2.0382;2.0691;2.1053 | 0.985 |  |
| `0f45aeae` | completed | 0.440 | 308.085 | 2.646 | 2.355 | 4.1942;2.3554;2.4173 | 0.974 |  |
| `f7f192ab` | completed | 0.000 | — | — | — | — | — |  |
| `ea4e76fb` | failed | — | 276.710 | 0.000 | 3.161 | 3.1605;3.424;3.1615 | 1.0 |  |

### v5

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `8175a973` | completed | 0.900 | 49.386 | 13.368 | 3.512 | 3.7534;3.5467;3.5123 | 0.99 |  |
| `3907583b` | completed | 0.870 | 59.516 | 13.914 | 4.006 | 4.488;4.0063;4.5172 | 0.893 |  |
| `26833bab` | completed | 0.870 | 76.632 | 17.864 | 5.080 | 5.0803;5.6459;5.1989 | 0.977 |  |
| `197d7c23` | completed | 0.830 | 92.974 | 21.498 | 7.035 | 7.0347;7.1748;8.0045 | 0.98 |  |
| `95ab77a1` | completed | 0.820 | 61.807 | 13.938 | 4.688 | 4.6876;5.0522;6.1455 | 0.928 |  |
| `2419f19d` | completed | 0.810 | 69.420 | 16.777 | 5.705 | 5.806;5.8328;5.7049 | 0.983 |  |
| `2c2dfce8` | completed | 0.740 | 48.999 | 12.755 | 5.139 | 5.1395;5.3066;5.2833 | 0.973 |  |
| `cb7847fb` | failed | — | 51.759 | 13.241 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |

### v6

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `f40c5e25` | completed | 0.910 | 58.460 | 14.101 | 3.555 | 3.8131;3.5545;3.7475 | 0.948 |  |
| `3f30d81f` | completed | 0.880 | 133.365 | 22.053 | 6.037 | 6.0367;8.045;9.8704 | 0.75 | graded 0.88 off first shot 6.04 with later shots 8.05/9.87 (m/m 0.75) — machine degraded mid-grading or artifact; unaudited |
| `4d0483ea` | completed | 0.850 | 57.213 | 14.055 | 4.353 | 4.353;4.5348;4.4922 | 0.969 |  |
| `78662a62` | completed | 0.840 | 59.691 | 14.794 | 4.637 | 4.6373;4.8553;4.7322 | 0.98 |  |
| `5a869e40` | completed | 0.800 | 63.062 | 16.194 | 5.692 | 5.6916;6.182;5.9568 | 0.955 |  |
| `2cbe0fb0` | completed | 0.800 | 49.188 | 12.910 | 4.589 | 4.6832;4.5892;4.6839 | 0.98 |  |
| `8b0fbe57` | completed | 0.770 | 47.050 | 12.972 | 4.880 | 4.9171;4.9535;4.8805 | 0.993 |  |
| `4a50d3db` | completed | 0.770 | 47.281 | 13.449 | 5.073 | 5.09;5.0729;5.1926 | 0.997 |  |

### v7

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `47551a02` | completed | 0.820 | 52.081 | 13.296 | 4.177 | 4.1768;4.3303;4.308 | 0.97 |  |
| `91a35119` | completed | 0.780 | 46.543 | 12.935 | 4.457 | 4.5332;4.4572;4.5575 | 0.983 |  |
| `69505252` | completed | 0.780 | 53.894 | 13.577 | 4.683 | 4.6981;4.6826;4.7883 | 0.997 |  |
| `b1eaa90c` | completed | 0.770 | 46.600 | 12.737 | 4.595 | 4.6054;4.6938;4.5947 | 0.998 |  |
| `9d9b3935` | completed | 0.770 | 51.473 | 13.682 | 4.838 | 4.8407;4.8383;4.8814 | 1.0 |  |
| `a8d74426` | failed | — | 48.508 | 13.125 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |
| `4588ea43` | failed | — | 62.238 | 15.190 | 0.907 | 0.9069;1.2;1.2746 | 0.756 | v7 env-fail: guard caught C_opt=0.907 (would have been a false 1.0 on the roofline curve) |
| `fb93b78d` | failed | — | 52.256 | 13.640 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |

### warm

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `d43251c2` | completed | 0.990 | 53.193 | 13.899 | 2.019 | 3.8954;4.0881;2.0194 | 0.518 | **ARTIFACT-minshot** → honest C_opt 3.895 s, score 0.851 |
| `00291a90` | completed | 0.970 | 49.056 | 13.053 | 2.132 | 2.1967;2.1321;2.1696 | 0.983 | **ARTIFACT-stable** → honest C_opt 3.31 s, score 0.875 |
| `361a3485` | completed | 0.930 | 47.663 | 12.998 | 2.674 | 2.6737;2.7716;2.8389 | 0.965 |  |
| `57053476` | completed | 0.900 | 47.719 | 13.003 | 2.975 | 2.9985;3.1462;2.9746 | 0.992 |  |
| `53ebdad6` | completed | 0.890 | 47.082 | 13.071 | 3.151 | 3.1505;3.1627;3.1601 | 0.997 |  |
| `ddb21a32` | failed | — | 47.875 | 13.188 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |
| `2d455c11` | failed | — | 45.736 | 12.897 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |
| `b2f5a346` | failed | — | 48.935 | 13.242 | 0.000 | 0.0;0.0;0.0 | — | guard caught C_opt=1e-06 |

### hot

| run | status | graded score | C_ref | C_sota | C_opt | opt walls (s) | min/med | audit / flag |
|---|---|---|---|---|---|---|---|---|
| `d82aee89` | completed | 0.980 | 47.747 | 13.088 | 1.224 | 1.2279;1.2239;1.2366 | 0.997 | **ARTIFACT-stable** → honest C_opt 3.59 s, score 0.81 |
| `502912a3` | completed | 0.950 | 47.731 | 12.963 | 1.661 | 1.6864;1.6608;1.7619 | 0.985 | **ARTIFACT-stable** → honest C_opt 2.68 s, score 0.87 |
| `262a05c6` | completed | 0.940 | 48.277 | 12.994 | 1.752 | 1.7867;1.7516;1.9047 | 0.98 | **ARTIFACT-stable** → honest C_opt 3.15 s, score 0.84 |
| `16d44d13` | completed | 0.920 | 49.149 | 13.128 | 2.071 | 3.3485;2.0712;3.133 | 0.661 | **ARTIFACT-minshot** → honest C_opt 3.13 s, score 0.843 |
| `04b0abdc` | completed | 0.870 | 50.927 | 13.140 | 2.827 | 2.9262;2.9301;2.8275 | 0.966 |  |
| `90378bc1` | completed | 0.840 | 47.533 | 12.990 | 3.112 | 3.4128;3.1122;3.1567 | 0.986 |  |
| `76e4de6f` | failed | — | 50.281 | 12.905 | 0.703 | 1.2503;0.7033;0.9144 | 0.769 | hot env-fail: guard caught C_opt=0.703 walls [1.25, 0.70, 0.91] — artifact-class walls |
| `436255b8` | failed | — | 49.290 | 13.247 | 0.000 | 0.0;0.0196;0.2011 | — | guard caught C_opt=1e-06 |

## Audited corrections (self-benchmark reconciliation)

| run | round | graded | honest | basis |
|---|---|---|---|---|
| `d43251c2` | warm | 0.99 | **0.851** (3.895 s) | own closing estimate ~3.0 s graded-equivalent; bare-metal rebuild 0.857 s/1x |
| `00291a90` | warm | 0.97 | **0.875** (3.31 s) | final VM self-bench 2.74 s on W1 guess x1.20-1.23 workload ratio => 3.26-3.37 s |
| `16d44d13` | hot | 0.92 | **0.843** (3.13 s) | 3x-scaled bench 3.459 s five minutes pre-grade; shots 3.348/3.133 reconcile within 3-9% |
| `d82aee89` | hot | 0.98 | **0.81** (3.59 s) | own 3x-scale bench 3.359 s (r=0.274); graded walls equal its own 1x walls |
| `262a05c6` | hot | 0.94 | **0.84** (3.15 s) | fresh-process best-of-5 four minutes pre-grade: r=0.27-0.31 across four workload shapes |
| `502912a3` | hot | 0.95 | **0.87** (2.68 s) | wrote "expect graded best-of-shots ~3.3-4.5 s" six minutes before grading; 3x bench 3.443 s vs MKL 16.65-16.98 |

## Honest cross-round frontier (per 1x pass, grading VM)

| round | honest best | per-1x | vs MKL |
|---|---|---|---|
| v5 | 0.90 (`8175a973`, 3.51 s) | 1.17 s | 3.8x |
| v6 | 0.91 (`f40c5e25`, 3.55 s) | 1.18 s | 3.9x |
| v7 | 0.82 (`47551a02`, 4.18 s) | 1.39 s | 3.2x |
| warm | 0.93 (`361a3485`, 2.67 s) | 0.89 s | 4.9x |
| hot | ~0.87 honest (`502912a3`, 2.68 s audited) | 0.89 s | 4.8x |

The genuine frontier plateaued at ~0.89 s per 1x pass (~4.8-4.9x MKL, ~53% of the
measured issue-cap roofline) across warm and hot despite full prior-work transfer and
an ISA-ideal target: the ~2.1 vector-uop/cycle issue equilibrium held.
