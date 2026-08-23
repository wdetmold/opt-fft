# Solution notes — iterated batched 3D complex FFTs (8 fixed cube sizes)

## What this is

`implementation.c` is a single self-contained C file of hand-written AVX-512
DFT kernels (no FFT library anywhere): per-size engines for
L = 6, 8, 13, 17, 23, 36, 45, 64 computing m steps of
`z = FFT3(x) + c; x <- z/(1+|z|)` entirely in IEEE double precision.
`solution.py` is the required wrapper skeleton (verbatim except the marked
fill-ins: gcc flags, ctypes bindings, pooled output buffers, import-time
warmup). `implementation.so` is prebuilt from that exact source so the graded
import does not compile.

## Composition (warm-start rules allow building on the supplied prior work)

This artifact ROUTES per size across the best prior engines, selected by
extensive interleaved A/B benchmarking on this machine, plus my own glue:

- L = 6, 8, 13, 17, 23, 36: the engines of `warm_00291a90`'s recovered
  `implementation_final.c` (batch-lane SoA with hand-asm prime codelets,
  PFA/radix codelets, stage-interleaved float-seeded map, per-volume tail
  drivers), compiled under their preferred flags via
  `#pragma GCC optimize("O3","unroll-loops","schedule-insns","sched-pressure")`.
- L = 64: `v6_3f30d81f`'s engine (within-volume, lanes = low 3 bits of x,
  radix-8^2 vertical z/y kernels, four-step x pass, chain-resident per volume),
  under its native flags (no schedule-insns).
- L = 45: `v5_3907583b`'s engine (via d43251c2's `impl_3907.c`) for B >= 8,
  w00's engine for B < 8 (measured crossover).
- Symbol isolation by mechanical renaming; one C file; entry points
  `run_{L}(x0, c, out1, outm, B, m)`.

## Own measurements that drove the routing (this machine, interleaved best-of-N)

ns per element-step (lower is better), at L3-warm batch shapes:

| L | 6 | 8 | 13 | 17 | 23 | 36 | 45 | 64 |
|---|---|---|----|----|----|----|----|----|
| w00 engine  | 1.13 | 1.27 | 2.33 | 2.73 | 3.46 | 3.12 | 3.45-3.50 | 3.94 |
| 3f30 engine | - | - | - | - | - | - | - | **3.35-3.38** |
| 3907 engine | 1.29 | 1.47 | 2.39 | 2.95 | - | 3.17 | **3.14-3.33** | - |
| shipped     | 1.13 | 1.27 | 2.33 | 2.73 | 3.46 | 3.12 | 3.33 | 3.37 |

My own from-scratch engines (IR-generated intrinsics and hand-asm prime
pencils with register-resident folded twiddles; within-volume L2-resident
designs for 23/36) were built, verified correct, and benchmarked at
13: 2.39-2.44, 17: 3.4, 23: 4.9, 36: 3.52 - they did not beat the incumbents,
so they are not in the graded path. Machine findings from that work (re-verified
here, sometimes contradicting the provided notes):
- vdivpd/vsqrtpd zmm are ~16/~24 cyc throughput on REAL data (constant-input
  microbenchmarks show 4 cyc - data-dependent early exit; do not trust them).
- vrsqrt14pd/vrcp14pd/vrsqrtps are all ~4 cyc throughput here; the
  rsqrt14+2-Newton + rcp14+2-Newton map (~19 uops/8 elems) is the floor, and
  alternating divider/Newton or float-packed seeds do NOT help in context.
- 512-bit loads cap at ~0.92/cyc (64B/cyc L1); reg-reg FMA 2/cyc @ ~3.1 GHz;
  broadcast-FMA ~1.9/cyc; in-place L3 sweeps run ~2x faster than
  out-of-place (50 vs 27 GB/s counted).
- -fschedule-insns/-fsched-pressure HALVED the speed of my IR-generated
  codelets while helping w00's (hence per-section optimize pragmas).

## Self-benchmark (final, this VM, fresh processes, best of 5-7 shots)

Workload shapes with ~45e6 element-steps per size (m = 400..40 resp. 200..18):
- shape W1 (chain-heavy): solution 1.186 s | MKL-DFTI sequential 5.274 s (r = 0.225)
- shape W2 (batch-heavy): solution 1.433 s | MKL-DFTI sequential 4.674 s (r = 0.307)
  (W2's base.py/MKL ratio = 3.70, matching the graded C_ref/C_sota ratios ~3.65;
   the difference between shapes is numpy input-generation cost and MKL's
   batch amortization - our compute is shape-insensitive at ~0.96 s per
   360e6 element-steps.)

Correctness: one-step blocks 3.5e-16..1.0e-15 relative L2 per size (gate 1e-14);
m-step blocks pass with >=1e4 margin at every size; batteries over
B in {1,2,3,5,7,8,9,15,16,17,21,33,37,61,102,...}, m in {1..7, 11..151} and the
shapes above, all verified against base.py. Deterministic (bit-identical repeat
calls). Single-threaded; links only libm/libc.

## Final validation (logged before submission)

- Randomized fuzz: 14 random (seed, B, m) workloads (B up to 40/30/20/14/10/7/5/4,
  m in 1..29) checked against base.py: 14/14 PASS, worst one-step block error
  1.05e-15 (gate 1e-14); all per-size m-step gates passed.
- Fixed batteries: B in {0,1,2,...}, m in {1,...,151}, B=0 blocks: PASS.
- Determinism: repeat transform() calls bit-identical.
- Final W1 fresh-process walls (solution): best 1.154-1.186 s across sessions;
  MKL-DFTI sequential same-shape: best 4.95-5.27 s. 3x-scale single-shape:
  3.443 s vs 16.65 s.
- Rebuild check: gcc on the shipped implementation.c reproduces the shipped
  implementation.so byte-identically (md5 e2b0f7ee7f4185920402ac28000b1ee0).
