# Attempt 2cbe0fb0 — fft3d-fixed-geometry-opt-20260821 (v6) — score 0.7961

Reconstructed final graded `/workdir` state, recovered from the session log
(`v6_audit/attempt_2cbe0fb0_score0.8.log`).

## Attempt facts

| | |
|---|---|
| Attempt id | `2cbe0fb0` |
| Problem | `fft3d-fixed-geometry-opt-20260821` — **v6** (per-size m-step gates: tol 1e-4 / 3e-6 / 1e-9 / 1e-10 x5 for L = 6...64; v5 had flat gates) |
| Score | **0.7961** |
| Graded costs | C_ref = 49.19 s, C_sota = 12.91 s, **C_opt = 4.589 s** (best of 3; walls: ref [52.87, 49.19, 49.56], sota [12.91, 13.29, 13.58], opt [4.683, 4.589, 4.684]) |
| Checks | format/constraint/content all passed |

## Files

- `solution.py` (50 lines) — the required ctypes wrapper. Skeleton kept verbatim;
  filled in: compile flags, `ns_init`/`ns_run` bindings, `_run` call.
- `implementation.c` (1056 lines) — single self-contained C file, all transform
  arithmetic hand-written (no FFT libraries).
- The graded `/workdir` also shipped a **prebuilt `implementation.so`** (built in
  the container at ~05:12 with the same flags below). The binary cannot be
  reconstructed from the log; `solution.py` rebuilds it automatically when the
  `.so` is absent, so these two source files are the complete deliverable.

## Compile command

```
gcc -O3 -march=native -ffp-contract=fast -fno-math-errno -fno-trapping-math \
    -funroll-loops -falign-loops=32 -shared -fPIC implementation.c -o implementation.so -lm
```

(x86-64 AVX-512 target; do not attempt to compile on ARM.)

## Algorithm summary

AVX-512 split re/im implementation with per-size FFT codelets operating on
vectors of 8 doubles: L=6 Cooley-Tukey 2x3; L=8 radix-2/4 DFT8; L=13/17/23
even/odd-folded symmetric half-matmul DFTs (long-double-accurate cos/sin tables,
pure FMA accumulation); L=36 CT 6x6 with twiddles; L=45 twiddle-free prime-factor
9x5 (DFT9 = 3x3 CT, DFT5 codelet); L=64 CT 8x8 with twiddles plus a fused z-axis
codelet doing one in-register 8x8 transpose between the two radix-8 stages. Two
runtime paths: a batch-interleaved path (8 volumes in SIMD lanes, zero padding
waste, zero shuffles) for L<=36, including partial final groups (nv >= RMIN)
run with duplicated lanes and masked outputs; and a per-volume path (padded
rows, 8x8-transposed z-tiles) otherwise, with a 72-double row stride at L=64 to
break 512-byte L1 set aliasing. Each volume/group stays cache-resident across
all m iterations; the pass order (Y, X, Z) lets the contiguous z-pass complete
each step's FFT, at which point the nonlinear map x = z/(1+|z|) is applied
plane-locally using `rsqrt14`/`rcp14` plus Newton iterations (full double
accuracy, no divider ops), with the constant `c` deinterleaved on the fly from
the caller's array and the step-1/step-m outputs captured inside the fused
passes. Hugepage-backed cached buffers, tuned software prefetch,
single-threaded.

## Reconstruction provenance

- The run hit an API rate-limit AUTO-RETRY at 04:01: the attempt restarted in a
  fresh container restored from the post-setup snapshot, and the earlier
  transcript (44 actions) was replayed into it. Only the post-restart command
  history (135 bash actions, log lines 3647-11327) determines the graded state;
  it was replayed here in a sandbox with path rewriting and shims (gcc/objdump/
  taskset stubbed, GNU sed, named benchmark scripts skipped, inline python
  patch scripts executed for real).
- `implementation.c` was written by heredocs and evolved via ~30 inline
  `python3 - <<'PYEOF'` text-patch scripts (no generator scripts were used in
  this attempt, so there is no `dev_generators/` directory). Every exact-match
  `src.replace`/`src.index` patch found its target during the replay, which
  self-checks byte fidelity of the heredoc decoding.
- Final-state cross-checks: `python3 -m py_compile solution.py` passes;
  brace/paren/bracket counts balance in the C; the file contains the late-stage
  features the agent's closing summary describes (documentation header, RS=72
  row stride and `zline64_pwc_cap` for L=64, RMIN partial-group interleaved
  dispatch, `-funroll-loops -falign-loops=32` flag patch in solution.py); the
  final `/tmp/dev` copy and `/workdir` copy agree, matching the session's last
  A/B step. The log records commands only (no tool outputs), so the md5sums the
  agent printed at 05:17 are not available for comparison.
