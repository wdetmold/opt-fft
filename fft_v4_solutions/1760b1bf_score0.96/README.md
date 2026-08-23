# Attempt 1760b1bf — score 0.9593 (fft3d-fixed-geometry-opt-20260821)

Reconstructed final graded source files, replayed from the session log
(`attempt_1760b1bf_score0.96.log`) by re-executing, in chronological order, the
original heredoc writes plus all 30+ in-place patch scripts and sed edits that
produced the state present in `/workdir` at the pre-grade capture.

## Grade (from the log's grade_problem response)

- score: **0.9593228080023828**  (checks: format/constraint/content all True)
- C_ref  = 283.479 s   (walls 314.58 / 283.48 / 303.41)
- C_sota = 4.0125 s    (walls 6.073 / 4.825 / 4.013)
- C_opt  = **1.4999 s** (walls 2.010 / 2.947 / 1.500)
- i.e. ~189x over the trusted base, ~2.7x faster than the held-out SOTA.

## Files graded in /workdir (reconstructed here)

| file | role |
|---|---|
| `solution.py` | ctypes wrapper; verbatim-skeleton `transform()`; per-size dispatch tables `_MODE = {6:D, 8:B, 13:A, 17:A, 23:A, 36:B, 45:E, 64:E}` (per-volume path) and `_GMODE = {6:2, 8:3, 13:3, 17:1, 23:3}` (4-volume batched path, remainder volumes fall back to `_MODE`); atomic compile-to-temp then `os.replace` |
| `implementation.c` | single self-contained generated C file (~1.4 MB), regenerated here from the final generator (`python3 dev_generators/gen.py implementation.c`) |
| `generator.py` | provenance copy of the generator made by the agent at 16:00 — one patch older than `dev_generators/gen.py` (lacks the final pack/unpack zmm-copy patch), exactly as it was graded |
| `dev_generators/gen.py` | the FINAL `/workdir/dev/gen.py`, i.e. the exact generator state that produced the graded `implementation.c` |

Also present at grading time but not reconstructible from the log:
`implementation.so` (prebuilt binary; the log verified it bit-identical to a
fresh build of `implementation.c`), `base.py` (grader-provided), and the
`/workdir/dev/` scratch tree (benchmarks, probes, test scripts — only
`gen.py` is load-bearing and is included here).

## Compile command used by the wrapper

```
gcc -O3 -march=native -ffp-contract=fast -DPW_STYLE=2 -shared -fPIC \
    implementation.c -o implementation.so.tmp.<pid> -lm   # then os.replace()
```

Do not try to compile on ARM — the code is AVX-512 (Ice Lake-SP target).

## Algorithm summary

Single-threaded AVX-512 (one zmm = 4 complex128), fully generated straight-line
codelets per cube size. Factorizations: PFA (twiddle-free) for 6=2x3, 36=4x9,
45=9x5; two-stage Cooley-Tukey with an explicit stack buffer between stages
(so each stage's live set fits the 32 zmm registers) for 8 and 64=4x16; and a
phase-structured symmetric half-DFT for the primes 13/17/23 that splits the
cos-sweep and sin-sweep into register-resident phases (only 2h distinct
constants, kept in registers — discovered because memory-operand/broadcast
FMAs run at ~half throughput on this core). Two layout engines, tuned per
size: a padded per-volume layout (conflict-free strides S2/S1, 2MB-aligned
hugepage buffers, 4x4 in-register complex transposes for the contiguous axis,
fused-transpose A2 codelet for L=64) and a 4-volume-batched layout (one zmm =
the same element of 4 volumes; no transposes or pad lanes), used for L<=23
with per-volume fallback for B mod 4. The nonlinear map z/(1+|z|) uses a
float-seeded (`vrsqrtps`/`vrcpps`) Newton pipeline — PW_STYLE=2 pairs an
all-FMA rsqrt-Newton sqrt with the hardware divider for the reciprocal; the
full-precision (~1 ulp) variant is applied only where graded outputs are
produced (pack step / first iteration), a cheaper ~2^-46 variant inside the
iteration. Fusion point of `+c` and the map is a per-size mode (A: at next
pass's loads; B/3: at axis-0 store; D/E/2: separate linear or per-slab sweep),
selected empirically per size. Per-function gcc `optimize` attributes give
8/36 (and prime a2 flavors) `-fschedule-insns -fsched-pressure
-frename-registers` while keeping the rest plain. All m iterations run
cache-resident per volume (or per 4-volume group); c is read once per
iteration.

## Reconstruction verification

- Every patch script was replayed with its original internal `assert old in
  src` guards intact — all passed, which cross-checks each intermediate gen.py
  state against the log to the byte.
- Two patch scripts in the log failed historically (silent no-ops): the second
  "attach OA" script (its target text had already been rewritten by the first)
  and the PW_STYLE=3/4 patch (over-escaped backslashes in its `old` block).
  The replay reproduced both failures identically.
- Regenerated `implementation.c` is 1,406,353 bytes, matching the agent's own
  "~1.4 MB" description; contains `setup`, all `run{L}_{A,B,D,E}`, `rung{L}`,
  `fft64_a2f`, PW_STYLE conditionals.
- `solution.py`, `generator.py`, `dev_generators/gen.py` all pass
  `python3 -m py_compile`; `transform()` matches the required skeleton
  verbatim (same check the agent ran before grading, and the log's final
  in-container check printed True).
- `generator.py` vs `dev_generators/gen.py` differ exactly by the final
  pack/unpack zmm-copy patch, as the log's timeline dictates.
- Grep for mkl/fftw/pocketfft/libfft in the deliverables: zero hits (matches
  the log's compliance check).

Caveat: `implementation.c` was regenerated on macOS/ARM python3. The generator
is deterministic, but non-snapped twiddle constants go through `math.cos/sin`;
a libm that rounds differently in the last ulp could alter some hex-float
constant strings relative to the container-generated file (no semantic
effect). The container's own md5s were not printed to the log, so byte
identity with the graded file cannot be proven, only structural identity.
