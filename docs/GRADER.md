# The graded call: specification and rationale

The grade is one number per implementation, over a fixed set of (L, B, m) points — B
volumes of L³, transformed in a chain of m steps. This document specifies the semantics
precisely, because two details decide whether the grade measures FFTs at all.

## The fixed points

| L | B | m | B·L³ | in+out working set |
|---|---|---|---|---|
| 6 | 64 | 4856 | 13,824 | 0.42 MiB |
| 8 | 64 | 2572 | 32,768 | 1.00 MiB |
| 13 | 32 | 1278 | 70,304 | 2.15 MiB |
| 17 | 32 | 98 | 157,216 | 4.80 MiB |
| 23 | 16 | 165 | 194,672 | 5.94 MiB |
| 36 | 8 | 64 | 373,248 | 11.39 MiB |
| 45 | 4 | 177 | 364,500 | 11.12 MiB |
| 64 | 2 | 134 | 524,288 | 16.00 MiB |

Every point is cache-resident (L2 for 6 and 8, L3 for the rest, on any of the cluster's
nodes). That is the regime the m·B weighting implies — a long chain over a working set that
stays put — and it is the regime the real workload lives in. Note this is *not* the regime
of large-batch benchmarking (hundreds of MiB, memory-bound), so numbers from those sweeps
do not predict the grade.

## Chain semantics: each step must be unitary-normalized

**A raw forward chain is not runnable.** Feeding an unnormalized forward transform its own
output grows magnitudes by ~√V per step; measured with numpy at these exact points, it
overflows to ±inf at 5 of the 8 sizes — L=6 at step 264 of 4856, L=64 at step 114 of 134.
A grader that ran it literally would be timing inf/NaN propagation at most sizes, and
inf/NaN arithmetic timing is hardware-dependent, so even the *timing* would be corrupt.

So the graded step is:

    state ← FFT(state) · V^(−1/2)

with the scaling applied by the **driver**, identically for every contender (one
bandwidth-bound read+write, small against a 3D transform, and realistic — production
chains always have pointwise physics between transforms). Buffers ping-pong, so each step
genuinely consumes the previous step's output: this preserves the property that a
memoizing or input-caching implementation gains nothing.

## Correctness: the whole chain, in closed form, for free

The normalized chain has an exact closed form: FFT² = V·(index reversal), so after m steps

| m mod 4 | end state |
|---|---|
| 0 | x |
| 1 | FFT(x)/√V |
| 2 | x[−j mod L] |
| 3 | FFT(x[−j])/√V |

The checker verifies the **end of the chain** against this (at most one reference FFT,
however long the chain), with tolerance `1e-12·√m` for accumulated roundoff. This is a
strictly stronger test than checking one transform: an error anywhere in m steps compounds
into the end state. Measured: a correct implementation lands at ~3e-13 after 4,856 steps,
right on the √m·ε curve. A single-transform check (rel L2 < 1e-12 against numpy) is kept
alongside as the fast familiar gate.

## Gating a chaotic chain (the graded map workload)

The graded map step `z = FFT3(x)+c; x <- z/(1+|z|)` is weakly chaotic: two correct fp64
implementations differing only in rounding order diverge roughly exponentially with step
count. Measured at L=6, m=4856, the honest drift of MKL/FFTW/ducc0 against the numpy
reference ranged 2e-10 to 1.1e-8 across three input seeds. A fixed chain-end tolerance is
therefore a coin flip: our original max(1e-12, 1e-13·m) gate failed all six libraries on
two of three seeds and falsely rejected panel entries that sat closer to the reference
than MKL did. (The original calibration propagated a single 1-ulp perturbation — that
measures a transient, not the divergence under continuous per-step rounding injection.)

The corrected gate is two-part, so no single number has to both forgive chaos and catch
cheats:

1. **Two-step precision gate** — the fused chain path is run for m=2 and must match the
   numpy reference within 3e-14 (1.5e-14 per step). Chaos cannot amplify anything in two
   steps; careful fp64 lands ~1e-15, an fp32-seeded map lands ~5e-12, and in-chain
   shortcuts are exposed even when the single-call path is exact (observed: entries with
   4e-16 single-call error and 1.4e-13 two-step error from an approximate in-chain map).
   The check exercises the chain path itself, not the single-call path, because that is
   where shortcuts live.
2. **Chain-end gate** — rel L2 within 300x the honest divergence measured on the SAME
   chain (the worst library drift and a numpy two-path anchor), floored at 1e-10 and
   rounded up onto a {1,3}x10^n grid. The 300x is 30x for a solver legally at the
   per-step ceiling (1.5e-14 against the references' ~3e-16 per-step difference,
   amplified linearly-in-seed) times a decade of run-to-run slop. This still sits 4+
   orders below an fp32-interior chain and O(1) bugs; its only job is gross cheats and
   memoization, and it can no longer punish honest rounding.

The single-transform check (rel L2 < 1e-12 vs numpy) remains alongside as before.

## Timing

Per the standing methodology: compilation and plan setup excluded, warmup discarded, the
timed unit is **one full chain** (m transforms), several samples, several independent
processes, minimum reported with spread alongside. Reported as **µs per transform**
(chain time / m / B is *not* used — per transform means chain time / m, with B volumes per
transform — stated to avoid ambiguity).

## The score

Per size: `r_L = t_lib(L) / t_ours(L)` using each side's best correct entry at that point
(the libraries get their native batched interfaces: MKL DFTI with
`DFTI_NUMBER_OF_TRANSFORMS`, FFTW `plan_many_dft`, both planned outside the timed region).

Aggregate, reported as three numbers, never one:

1. **time-weighted**: Σ t_lib / Σ t_ours over the whole 8-point workload — what a user
   running exactly this mix experiences. Note the m·B weights make the large-L points
   dominate: L=45 and L=64 together are ~45% of total library time.
2. **geometric mean** of the r_L — scale-free, every size counts equally.
3. **worst case**: min r_L, with the size named — the honest "where do we still lose".

A single weighted average is required to be accompanied by the per-size table: the earlier
1.65× Ice Lake result was an aggregate over a different regime (memory-bound large
batches) with the B=1 kernels selected at every batch size, and the per-size data shows
our kernels *ahead* at the same sizes where the aggregate said behind.

## Measurement protocol: paired, interleaved, ratio-of-windows

This section exists because of a real incident. The grader once reported MKL 1.65x ahead;
a pinned probe on the same machine then showed our suite 1.46x FASTER end-to-end (3.02-3.14 s
against 4.42-4.55 s, reproducible across three protocol executions, ahead at every size in
both regimes). The forensics: the grader timed each code in separate unpinned subprocesses
minutes apart on a steal-bursty VM and kept each side's best shot. MKL's shots were 2.42 s
and 4.51 s -- the 4.51 matches its true time, the 2.42 was a lucky quiet window that
best-of then kept -- while our shots (4.0/4.9 s) carried ~0.7 s of in-call input generation
plus an unlucky window. Cross-code timing noise of ~2x, at a hard parity cliff in the score,
inverted the verdict.

Three rules, each of which alone would have prevented it:

1. **Never compare absolute times taken in different windows.** Shots are PAIRED and
   INTERLEAVED -- A,B,A,B (and B,A,B,A on alternate repetitions, so ordering effects cancel)
   within the same window -- and the reported quantity is the **median of per-window ratios**
   A_i/B_i. Environmental drift then hits both sides of every ratio equally. Best-of-shots
   taken independently is forbidden: it selects each side's luckiest window, and on shared
   hardware that is a coin with ~2x sides.
2. **Nothing but the transform inside the timed call.** Input generation, file IO and
   checksumming happen outside it (the harness driver already reads pre-generated input; a
   graded call that generates data in-call adds a constant that dilutes one side only).
3. **Record the environment with every shot, and reject dirty windows.** /proc/stat steal
   time and involuntary context switches before and after each window; a window with
   nonzero steal or a load spike is discarded, not averaged in. Pin with taskset where the
   platform allows it. The two sides always run under identical pinning.

And one rule about the score itself: **no hard cliff at parity.** If the score steps at
A/B = 1.0, measurement noise near parity flips whole grades; use a smooth function of the
median ratio (the current one maps our 0.68 ratio to ~0.65) and publish the per-size table
next to it, so a single aggregate can never silently invert a verdict again.

## Environment

Measured on an exclusive node, `-march=native` **built on that node** (several kernels
select 512-bit paths by probing at plan time; a generic build silently benchmarks their
fallback against MKL's native dispatch). Machine, ISA and clock state recorded next to
every result.

## Implementation

All of this exists and is verified: `driver.c --chain M --unitary` (the scale is
driver-side), `check.py --chain-check M`, and the graded case list in
`bench/geom/cases_graded.txt`. Verified end to end at the L=6 point: 4,856-step chain,
closed-form check passing at 2.9e-13, MKL 0.716 µs/xform vs `L6_pfa` 0.446.
