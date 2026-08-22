# Attempt 1000f989 — score 1.00 (fft3d-fixed-geometry-opt-20260821)

Reconstructed FINAL graded state of `/workdir`, replayed mechanically from the
session log (`attempt_1000f989_score1.0.log`): the v4 full rewrite of
`implementation.c` (heredoc, log lines 3208–3741) plus all 21 subsequent in-place
python patch scripts executed in log order, plus the final
`sed 's/#define PXZ 577/#define PXZ 583/'`. `solution.py` is the verbatim final
heredoc (log lines 6548–6596).

## Graded result

- score = **1.0**; checks: format/constraint/content all true
- C_ref = 288.93 s, C_sota = 6.863 s, **C_opt = 2.014 s** (best of walls
  [2.620, 3.055, 2.014]; sota walls [11.09, 6.86, 8.56])
- i.e. ~143x over the trusted reference, ~3.4x faster than the held-out SOTA.

## Files

- `solution.py` — graded ctypes wrapper (problem skeleton verbatim; compiles
  `implementation.c` at import only if `implementation.so` is absent, then
  `init_tables()` once; adds C-contiguity guards in `_run`).
- `implementation.c` — the single self-contained C file (1045 lines), state
  after the last pre-grade modification (PXZ=583).
- The graded `/workdir` also contained a prebuilt `implementation.so` (x86-64
  binary, not reconstructible from the log) and the problem-provided pristine
  `base.py`. `solution.py` rebuilds the .so automatically when it is missing.

## Compile command used by the wrapper

    gcc -O3 -march=native -mprefer-vector-width=512 -ffp-contract=fast \
        -fno-math-errno -shared -fPIC implementation.c -o implementation.so -lm

(Do not compile on this Mac — the code is AVX-512 x86; it was verified here with
`clang -arch x86_64 -mavx512f -mavx512vl -fsyntax-only` only.)

## Algorithm summary

Volumes are lane-interleaved into SoA slots `{vec re, vec im}` with 8/4/2
volumes per AVX-512 zmm/ymm/xmm lane (width-templated by self-`#include` with
`VW=8/4/2`), so all three FFT axis passes are pure vertical SIMD — no shuffles
or gathers; PFA/index permutations compile to constant offsets. Per-size
kernels: L=6 PFA 3x2, L=8 radix-2 DIT, L=36 PFA 9x4, L=45 PFA 9x5, L=64 radix
8x8 with w64 twiddles; primes 13/17/23 use a symmetric-pair direct DFT in
j-outer matvec form with k-indexed register accumulators and transposed
`long double`-generated cos/sin tables. Each iteration is exactly two
cache-blocked sweeps: y-planes (lazy elementwise map + FFT_z + FFT_x) and
x-planes (FFT_y); the buffer holds the RAW FFT3 between iterations and the map
`z/(1+|z|)` (z = X + c) is applied lazily at the start of the next iteration
using `vrsqrt14pd` + 2 Newton steps for sqrt plus one exact `vdivpd` (divider
latency hidden under FMA work), row-pipelined with software prefetch for the
L3-sized cases. L=64 has a special within-volume z-split path (`run64_zsplit`,
zmm lanes = z-octants, PXZ/PYZ strides): cross-lane DIF/DIT DFT-8 butterfly
networks (`xl_dif8`/`xl_dit8`) alternate the buffer between natural and
bit-reversed z forms each iteration, with the form-B constants derived on the
fly by in-register 8x8 transposes (`tr8x8_brev`/`map_row_trB`), shrinking the
working set from 68 MB to ~9 MB per volume for any batch size. IO uses
vectorized 8x8 (4x4/2x2) transposes; `m==1` output is memcpy'd; plane/row
strides are empirically tuned odd paddings (PX6=37, PX8=70, PX13=174,
PX17=290, PX23=531, PX36=1299, PX45=2027, PX64=4161, PXZ=583) against cache-set
conflicts; buffers are 2 MB-aligned with `MADV_HUGEPAGE`.

## Reconstruction verification

- Every one of the 21 patch scripts is an exact-string `str.replace` (or
  `index`-based splice) — each reported a real change on replay, so every
  target string matched the evolving file exactly; a single divergence would
  have surfaced as a silent no-op, and none occurred.
- Final source contains all expected end-state markers (PXZ 583, PYZ 9, tuned
  PX paddings, FW[64]=1 zsplit policy, m<1 clamp, scheme-B lite prefetch,
  DFT36P/DFT45P wrappers, iterA experiment code removed) and balanced
  braces/parens; 1045 lines, consistent with the agent's own "~1050 lines".
- `python3 -m py_compile solution.py` passes; `clang -arch x86_64 -mavx512f
  -mavx512vl -fsyntax-only implementation.c` (with `-DMADV_HUGEPAGE=14` shim)
  passes with zero diagnostics.
- The log records `md5sum implementation.c implementation.so solution.py` and
  `wc -l implementation.c` being run pre-grade, but tool outputs are not stored
  in this log format, so no reference checksums were available to compare.

Log escaping note: the log stores literal `\n` (2-char) sequences as a
backslash followed by a real newline. The C heredoc contains no string
literals, so it was copied verbatim; inside python patch scripts, lines ending
in an odd run of backslashes were re-joined as `\n` (even runs are real C-macro
continuations written as `\\` in triple-quoted strings). No printf/format
strings exist in the final C file, so no ambiguity remained.
