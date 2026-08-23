# Attempt 4a50d3db — fft3d-fixed-geometry-opt (v6, per-size gates) — score 0.7726

Reconstructed final graded `/workdir` sources for Taiga attempt `4a50d3db` on
`fft3d-fixed-geometry-opt-20260821` (the **v6** spec: per-size final-block
tolerance gates, `L: 6->1e-4, 8->3e-6, 13->1e-9, 17..64->1e-10`; one-step
blocks gated at 1e-14).

## Grading result

- score = **0.772625** (graded 2026-08-23T05:59 UTC)
- C_ref = 47.28 s, C_sota = 13.45 s, **C_opt = 5.073 s**
  (opt walls: 5.090 / 5.073 / 5.193; sota walls: 13.461 / 13.449 / 13.491)
- checks: format / constraint / content all true
- ~9.3x faster than the graded reference, ~2.65x faster than the hidden SOTA.

## Files

- `solution.py` — the graded wrapper (problem skeleton verbatim + ctypes
  bindings + per-(L,B) cached output buffers). Compiles `implementation.c`
  at import time if `implementation.so` is absent.
- `implementation.c` — the single self-contained generated C file
  (1,330,357 bytes, 39,426 lines), regenerated from the final generators
  (see caveat below). In the graded container the agent verified by md5
  that the final `implementation.c` was identical to its `snap4` snapshot
  after reverting a last experimental normalize variant; this
  reconstruction reproduces that exact endpoint ("identical to snap4"
  reproduced in replay).
- `dev_generators/gen_fft.py`, `dev_generators/gen_c.py` — the code
  generator (mini-genfft: expression IR with hash consing + C emitter).
  In the graded `/workdir` these two files sat at top level alongside
  `solution.py`; regenerate with `cd dev_generators && python3 gen_c.py`
  (imports `gen_fft`, writes `implementation.c` to the cwd; pure Python +
  numpy, a second or two).
- **Not reconstructable:** `/workdir/implementation.so` (prebuilt x86-64
  binary in the graded snapshot). Harmless: `solution.py` rebuilds it at
  import, and import/compile time is excluded from the measured cost.
- **Not included:** `/workdir/base.py` (problem-provided reference, not
  authored by the agent).

## Compile command (what solution.py runs on the grading host)

    gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm

(Ice Lake x86-64 host with AVX-512; do not attempt to compile on ARM.)

## Algorithm summary

A Python generator emits straight-line AVX-512 DFT codelets (split re/im,
8-wide `VD` vectors, all constants folded at generation time from
long-double tables) for each size, with per-size factorization plans chosen
by in-situ benchmarking: 6 = PFA(2x3), 8 = radix-2 CT, 13 = Rader(12 = PFA
4x3), 17 = Rader(16 = CT4), 23 = Rader(22 = PFA 2x11), 36 = CT(6x6),
45 = PFA(5x9), 64 = CT(4x16). Two execution paths, both single-threaded:
a T-path for any batch size (per volume: in-place y-DFT per x-plane, then
per y-plane a strided x-DFT into an L1 scratch, z-DFT via register-transposed
row codelets with `TP8`, and a fused `+c` / `z/(1+|z|)` normalize streamed
back, software-prefetch pipelined across stages and planes), and a B-path
that packs 8 volumes into the SIMD lanes (no transposes; used when B >= 8
for L in {6,8,13,17,23,36,45} — `useB = {1,1,1,1,1,1,1,0}` — plus
padded-lane tails for leftover volumes above per-size thresholds
`BTHR = {6:5, 8:9, 13:6, 17:7, 23:9, 36:9, 45:7}`). The normalize uses
`rsqrt14`/`rcp14` with Newton refinement (`EXACT_NORM`/`PRECISE_NORM`
compile-time variants exist; the graded build used the default branch).
All buffers come from one `mmap` arena with `MADV_HUGEPAGE`; `impl_init`
runs at import time.

## Reconstruction provenance and caveat

Rebuilt by replaying all 150 actions of the attempt's final container epoch
(the session was twice requeued on API rate limits; the harness replayed the
full action history into a fresh container each time, so the last epoch
contains the complete effective history) from the session log
(`attempt_4a50d3db_score0.77.log`), executing every write/patch to
`gen_fft.py` / `gen_c.py` / `solution.py` in order (heredocs, in-place
Python patch scripts, seds, plan sweeps with their restores), then running
the final `gen_c.py`. Every patch script's internal `assert old in s`
passed, and the attempt's own final md5 self-check ("identical to snap4"
after reverting the quartic-reciprocal experiment) reproduced in replay.
Verified: `python3 -m py_compile` on all three Python files; C brace and
paren balance 0; clean-room regeneration from the shipped generators is
byte-identical to the shipped `implementation.c`
(md5 `0f5d1b0bbcf7a7fbb63e49c88db74906`).

Caveat: the generators derive twiddle/Rader constants via `np.longdouble`
(80-bit extended on the x86 grading container, only 64-bit double on the
ARM Mac used for this reconstruction), so a handful of emitted float64
constants could differ from the container-generated file in the last ulp.
The reconstruction is exact with respect to every logged edit; only this
platform difference in constant rounding is not certifiable byte-for-byte.
