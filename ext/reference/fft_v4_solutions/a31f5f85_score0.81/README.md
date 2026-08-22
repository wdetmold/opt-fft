# Attempt a31f5f85 — score 0.8103 (fft3d-fixed-geometry-opt-20260821)

Reconstructed FINAL graded source files, replayed from the session log
`attempt_a31f5f85_score0.81.log` (grade call at log end, 2026-08-22T16:08:59Z).

## Grading result

- score = **0.8102680603306167** (checks: format/constraint/content all true)
- C_ref  = 296.404 s   (walls 323.06 / 296.40 / 297.77)
- C_sota =   4.4174 s  (walls 11.475 / 4.4174 / 5.4109)
- C_opt  =   **2.3064 s**  (walls 3.3496 / 2.3064 / 2.5796)
- i.e. ~1.92x faster than the held-out SOTA (MKL-class) reference.

## Files in /workdir at grading time

Graded deliverables (reconstructed here):
- `solution.py` — ctypes wrapper (the problem's skeleton verbatim, with
  bindings for `plan()` and `run(L,B,m,x,c,one,fin)` filled in at import time,
  plus reusable output buffers). Created once, never modified.
- `implementation.c` — single self-contained C file, all optimization lives
  here. 932 lines / 37970 bytes after 34 in-place modifications.
- `implementation.so` — prebuilt shared object shipped alongside (binary, not
  reconstructible from the log). If absent, solution.py rebuilds it with the
  compile command below.

Also present in /workdir (dev files, not part of the graded import path):
`implementation.c.bak` (identical copy of final implementation.c), `base.py`
(provided), and test/bench scripts: test_fft3.py, test_full.py, test_full2.py,
test_base.py, test_edge.py, test_longm.py, test_final_gates.py,
test_stress.py, bench.py, bench1.py, bench2.py, bench3.py, grader_bench.py,
mockref_full.py, longm.log.

## Compile command (as used by the wrapper and throughout the session)

    gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm

(Target was a virtualized Ice Lake Server, AVX-512, gcc 13.2. Do not try to
build on ARM.)

## Algorithm summary

Hand-written AVX-512 codelets per size, split re/im planes (pure vertical
SIMD, broadcast twiddles, trig tables built in `long double` at `plan()`
time): PFA 2x3 for L=6; radix-8; symmetric half-matrix kernels for the primes
13/17/23 (cos/sin constants held in zmm registers, compile-time index/sign
folding, register-pressure-tuned k-range splits — final CORE17 uses a 4-wide
k-split); twiddle-free Good–Thomas PFA 4x9 (L=36) and 9x5 (L=45); 8x8
Cooley–Tukey for L=64. Two runtime modes: a "grouped" mode (USE_GROUPED on
for 6/8/13/17/23) that carries 8 volumes in the SIMD lanes with contiguous
loads and no transposes (tail groups padded via `tovolsg_n`), and a
per-volume mode with a slab-fused z+y pass (8x8 register transposes, L2-hot
slabs) plus an x-pass fused with the `+c` and `z/(1+|z|)` elementwise map
(hardware sqrt + rcp14 with two Newton steps). Volumes are iterated one at a
time so all m steps run cache-resident; allocation staggering avoids
cache-set aliasing on L=64; snapshot ("one-step") outputs use streaming
stores.

## Reconstruction provenance / verification

- Baseline: the third full `create` of /workdir/implementation.c (log line
  2627) and the single create of /workdir/solution.py (log line 613).
- Replayed, in order, all 34 subsequent in-place modifications (28 python
  heredoc patch scripts + 6 `sed -i` commands, log lines 3227–7031).
- Every patch script's own `assert old in src` passed against the replayed
  state — a chained exact-content check over the whole edit history. The one
  exception (log line ~5785, dual-output-interleave patch) failed its first
  assert *in the session as well* ("Patch didn't apply (context drift) …
  skipping"); it wrote nothing there and nothing here.
- Intermediate state check: after the mod at log line 6522 the replayed file
  has exactly 918 lines, matching the session's `wc -l` driven paging
  (`sed -n '760,918p'`) at that point.
- `python3 -m py_compile solution.py` passes; final C has balanced
  braces/parens and contains all late-session markers (CORE17 4-split,
  USE_GROUPED[8]=1, SLBR[45*45*8+64], tovolsg_n, grun_64 stub).
- The log stores parameter text after a lossy unescape (literal `\n` inside
  string literals rendered as backslash+newline). Only one patch (log ~3771)
  was affected; its three artifacts were restored to `\n` before replay
  (verified by the later patches' asserts continuing to pass).
- The session's own md5sums/wc outputs were NOT captured in the log (tool
  results are not logged), so no byte-level checksum comparison is possible;
  the chained asserts + line-count match are the verification basis.
