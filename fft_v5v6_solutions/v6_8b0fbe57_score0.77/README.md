# Attempt 8b0fbe57 — fft3d-fixed-geometry-opt (v6, per-size gates) — score 0.7737

Reconstructed final graded `/workdir` sources for Taiga attempt `8b0fbe57` on
`fft3d-fixed-geometry-opt-20260821` (the **v6** spec: per-size final-block
tolerance gates, `L: 6->1e-4, 8->3e-6, 13->1e-9, 17..64->1e-10`; one-step
blocks gated at 1e-14).

## Grading result

- score = **0.773672** (graded 2026-08-23T05:29 UTC)
- C_ref = 47.05 s, C_sota = 12.97 s, **C_opt = 4.88 s**
  (opt walls: 4.917 / 4.954 / 4.880; sota walls: 12.972 / 13.120 / 12.994)
- checks: format / constraint / content all true
- ~9.6x faster than the graded reference, ~2.66x faster than the hidden SOTA.

## Files

- `solution.py` — the graded wrapper (problem skeleton verbatim + ctypes
  bindings). Compiles `implementation.c` at import time if
  `implementation.so` is absent.
- `implementation.c` — the single self-contained C file (2428 lines), the
  exact final graded state (last deploy at 05:24:45, before the
  `grade_problem` call). Byte-identical between `/tmp/dev` and `/workdir`
  at grading time.
- `dev_generators/` — the 12 development part files the agent maintained in
  `/tmp/dev`; the graded `implementation.c` is exactly their concatenation
  (`rebuild.sh` reproduces it byte-for-byte).
- **Not reconstructable:** `/workdir/implementation.so` (prebuilt binary was
  in the graded snapshot). Harmless: `solution.py` rebuilds it at import,
  and import/compile time is excluded from the measured cost.

## Compile command (what solution.py runs on the grading host)

    gcc -O2 -march=native -fschedule-insns -fsched-pressure -shared -fPIC \
        implementation.c -o implementation.so -lm

(Ice Lake x86-64 host with AVX-512; do not attempt to compile on ARM.)

## Algorithm summary

Hand-written AVX-512 DFT kernels per size (split re/im SoA, single thread,
no FFT libraries): PFA 2x3 for L=6, radix-2^3 for L=8, folded symmetric
matrix-DFT with k-pair ILP unrolling for the primes 13/17/23, twiddle-free
PFA 4x9 / 9x5 for 36/45, and radix-8^2 with exact long-double twiddles for
64. Structural layer tuned to the VM's weak memory system (~4 MB effective
LLC): (i) a volume-lane SIMD path for L in {6,8,13,17,23} when B >= 8
(remainder groups down to per-size thresholds), putting eight volumes in
the eight vector lanes so all three axis passes are unmasked streaming
kernels; (ii) fused two-stage streaming pipelines for 36/45/64 that split
the outer-axis DFT across successive iterations through a parity slot-reused
transpose buffer held L2-hot, with y/z passes on L2-resident slabs
(zslab64 with hoisted twiddle registers, spass_gen 8x8 register-transpose
sandwich for 36/45); (iii) the nonlinear map z/(1+|z|) fused into the last
DFT pass using rsqrt14 + 2 Newton steps for |z| and one hardware divide
overlapped with FMA work (small-size MAPST) or the same NR-sqrt + divide in
the pipeline map; (iv) out-of-place ping-pong on small sizes to avoid
store-to-load-forwarding stalls, pre-scaled PFA index tables, and
non-temporal stores on the final export. All precomputation (twiddle/fold
tables) happens in `init_tables()` at import.

## Reconstruction provenance

Rebuilt from the session log
(`fft_codeopt/v6_audit/attempt_8b0fbe57_score0.77.log`) by replaying, in a
stubbed sandbox (gcc/taskset/objdump/etc. stubbed, real python3/sed), the
complete post-restart bash action sequence (162 commands; the container was
restarted at 04:07 UTC after API rate limits and the harness replayed the
full prefix, so this sequence is exactly what built the graded state).
Verification: every in-session patch script's `assert old in src` passed and
printed its success marker (including reproducing the one expected
AssertionError at the Tbuf patch and the one bash syntax error on
`echo p45 written (36 next)` that the model then worked around);
`diff /tmp/dev/implementation.c /workdir/implementation.c` clean; part-file
concatenation reproduces `implementation.c` byte-for-byte;
`python3 -m py_compile solution.py` passes; C brace/paren balance = 0/0
outside strings/comments; no FFT-library references. No md5sums or full
file listings were printed in the log, so byte-level identity to the graded
container cannot be checked directly; identity rests on the deterministic
replay of every file-mutating command.
