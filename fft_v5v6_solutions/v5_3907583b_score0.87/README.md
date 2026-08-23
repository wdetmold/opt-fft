# Attempt 3907583b — fft3d-fixed-geometry-opt-20260821 (v5, flat gates) — score 0.869

Reconstruction of the FINAL graded `/workdir` state of attempt `3907583b`,
rebuilt from its session log
(`fft_codeopt/v5_audit/attempt_3907583b_score0.87.log`).

## Grade (from the log's grade_problem response, 03:12:30Z)

| quantity | value |
|---|---|
| score | **0.8690** |
| C_ref (base) | 59.52 s (best of 61.17 / 59.52 / 60.38) |
| C_sota | 13.91 s (best of 14.76 / 13.91 / 14.95) |
| C_opt (this attempt) | **4.006 s** (best of 4.488 / 4.006 / 4.517) |
| vs base | 14.9x |
| vs SOTA (MKL) | 3.47x faster |
| checks | format ✓ constraint ✓ content ✓ |

Problem version: **v5** — flat accuracy gates (one-step blocks < 1e-14
relative L2 per block, chain blocks < 1e-3, uniform across all eight sizes).

## Files

| file | provenance |
|---|---|
| `solution.py` | Verbatim from the log (str_replace_editor `create`, 02:39:20Z; final version — no edits afterward). ctypes wrapper: pins core 0, compiles `implementation.c` if the .so is absent, sets glibc `mallopt` M_MMAP_THRESHOLD / M_TRIM_THRESHOLD to 1 GiB (keeps numpy's large per-call buffers on the reusable heap — avoids page-fault storms), pools the two output arrays per size, calls `run(lid,B,m,x0,c,o1,om)`. |
| `implementation.c` | Regenerated (912,677 bytes; 25,005 generated lines + 21-line header, matching the attempt's own "~25k lines" record). Never printed in full in the log; produced by replaying the generator (below). |
| `gen_impl.py` | The generator exactly as shipped in the graded `/workdir` (copy of the dev `gen2.py` at 03:03:02Z). Rebuilt by replaying 10 heredoc writes/appends + 27 in-place patch scripts from the log; every original `assert old in src` in those patch scripts passed, which pins the intermediate states. |
| `dev_generators/gen2.py` | Same bytes as `gen_impl.py` (the `/tmp/exp` dev copy). |
| `dev_generators/regen_implementation.py` | The final generation call as recorded in the log (RADER_SET={13,17}; `generate_v6(gsizes=(6,8,13,17,23,36,45), gmap={L:'ALT' ...})` + the carried-forward header comment). Running it reproduces `implementation.c` **byte-identically** (md5 `cbbe31c14470b8ebcb07d50696e92d74`). |

Not reconstructed:

- `implementation.so` — prebuilt x86-64 AVX-512 binary; cannot be rebuilt on
  this ARM host. Harmless: `solution.py` recompiles it at import time
  (~8 s in-container, excluded from grading time). Note the graded .so was
  itself a fresh recompile (the attempt moved it away at 03:05Z for a
  cold-compile test and let import rebuild it).
- `base.py` — provided by the environment, not authored by the agent.

## Compile command

```
gcc -O3 -march=native -shared -fPIC -ffp-contract=fast -fno-math-errno \
    -funroll-loops implementation.c -o implementation.so -lm
```

(Same flags are embedded in `solution.py`. Do NOT compile on ARM — the code
is AVX-512 intrinsics/vector-extension C for Ice Lake SP.)

## Algorithm (one paragraph)

A Python generator (`gen_impl.py`) emits ~25k lines of specialized AVX-512
split-complex C (8 doubles/vector) for the eight cube sizes: prime-factor
(Good–Thomas) and Cooley–Tukey factorizations (6=2·3, 36=4·9, 45=5·9, 64=8·8
buffered two-phase), Rader prime transforms for 13 (→FFT-12) and 17 (→FFT-16),
and symmetric half-matrix direct prime transforms for 3/5/11/23 (23's k-loop
table-driven, unrolled by 3), with all twiddle/convolution constants computed
in 120-bit mpmath and emitted as exact hex doubles. For L ≤ 45 the data
layout is batch-lane groups of 8 volumes (SIMD across the batch: zero
transposes, one c copy, plane stride padded by `gSPL()` to de-alias L1 sets —
a 2.3x win on L=36's x-pass); L=64 and batch remainders use per-volume padded
split-complex planes with in-register 8×8 transposes and a digit-swap y-pass.
Each iteration is an x-pass over the volume followed by fused per-plane y/z
passes with the elementwise map x ← z/(1+|z|) folded into the final stores,
the map alternating per store row between the hardware divider path
(`vsqrtpd` + `rcp14`+Newton, MAPW_H) and a divider-free `rsqrt14`+Newton path
(MAPW_R2) so the divide unit and FMA ports run concurrently. State buffers
live on madvised huge pages; everything is single-threaded, pinned to core 0.

## Reconstruction verification

- All 27 replayed patch scripts ran with their original asserts intact — any
  divergence from the session's true gen2.py would have failed an assert.
- `python3 -m py_compile` passes on `solution.py` and `gen_impl.py`.
- `implementation.c`: braces/parens/brackets balance exactly; dispatcher has
  all 8 `case` arms; expected symbols present (fA_64/fB_64/fC_64, gA/gB/gC
  for 6–45, SYMC_23/SYMS_23 tables, rsqrt_nr2, MAPW_H/MAPW_R2 alternation);
  correctly absent: SYMC_13/17 (Rader replaces them), any `__builtin_prefetch`
  (all prefetch variants were measured and dropped).
- Codelet math check: the generator's straight-line DFT emissions for
  N ∈ {2,...,64} (including the Rader 13/17 paths) were interpreted
  numerically in Python and match `numpy.fft.fft` to ≲3e-16 relative L2.
- Generated line count 25,005 matches the attempt's final summary ("~25k
  lines"); total reconstructed size ≈ 0.98 MB vs the recorded 1.4 MB
  `/workdir` (difference ≈ the .so + base.py).

## Session notes

- The session hit API rate limits at 01:45Z; the container was rebuilt from
  snapshot and the transcript replayed, rolling back two experimental gen2.py
  patches (A-major c layout, prefetch-distance tunables) that were never part
  of the final code.
- Config decisions recorded in the log: Rader adopted for {13,17} only (loses
  for 23); group path adopted for L ≤ 45, per-volume digit-swap path for 64;
  all software prefetching dropped (VM noise, A/B-tested); `-funroll-loops`
  added after flag sweep; deferred-map and split-accumulator variants tested
  and rejected.
