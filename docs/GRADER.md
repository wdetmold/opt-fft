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
