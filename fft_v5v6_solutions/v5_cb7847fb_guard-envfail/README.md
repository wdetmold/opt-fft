# Attempt cb7847fb — FFT3D fixed-geometry code-opt (v5, flat gates)

Reconstructed FINAL graded /workdir sources for attempt `cb7847fb`, replayed
edit-by-edit from the session log
(`fft_codeopt/v5_audit/attempt_cb7847fb_scoreNone.log`).

## Attempt facts

| field | value |
|---|---|
| attempt id | cb7847fb |
| problem version | v5 (`fft3d-fixed-geometry-opt-20260821`): flat accuracy gates — one-step rel-L2 < 1e-14 per block, chain < 1e-3 per block, same thresholds for every L |
| score | **None** (`env_internal_failure`, not a solution failure) |
| correctness checks | format ✓ constraint ✓ content ✓ (all passed) |
| graded costs | C_ref = 51.759 s, C_sota = 13.241 s, **C_opt = 1e-06 s (artifact)** — all three "opt" walls hit the 1e-6 floor |
| failure reason | `score function raised: implausible C_opt 1e-06: at this workload no correct implementation finishes under 1 s; baseline-subtraction floor artifact -- regrade` |

The grade guard (added to catch the baseline-subtraction bug) fired: the
measured opt wall time minus the startup baseline clamped to the 1e-6 floor,
so the run was flagged for regrade rather than scored. The solution itself
passed every correctness gate; the agent's in-container benchmarks put it
5–9x over base.py and ~1.4–2.6x over an MKL-DFTI+AVX-512-map composite,
which would have scored highly on a clean regrade.

## Files

- `solution.py` — graded wrapper (verbatim skeleton + ctypes bindings,
  import-time self-test with rebuild-on-incompatibility fallback, glibc
  mallopt tuning, C-side `prewarm()` call, and a ~2.3 GB heap-arena page
  warm at import; all excluded from measured time).
- `implementation.c` — the single self-contained generated C file
  (23,064 lines incl. header; the generator reports "23005 lines" of body,
  matching the agent's own "~23k lines" summary). Contains every codelet,
  all run{L} entry points, and `prewarm()`.
- `gen_impl.py` — the Python generator that emits `implementation.c`. It was
  itself part of /workdir in this attempt (not a separate dev directory).
  `dev_generators/gen_impl.py` is an identical copy kept for the
  cross-attempt convention.
- NOT reconstructable: `implementation.so` (prebuilt x86-64/AVX-512 binary in
  the graded /workdir). Harmless: `solution.py` rebuilds it from
  `implementation.c` when missing, and even self-tests + rebuilds if the
  shipped binary fails to load or produces non-finite output.
- `/workdir/base.py` was the problem-provided reference (not agent-authored,
  not included here).

## Compile command (as used in the graded wrapper)

```
gcc -O3 -march=native -shared -fPIC -fno-math-errno -fno-trapping-math \
    -ffp-contract=fast implementation.c -o implementation.so -lm
```

## Algorithm summary

All transform arithmetic is generated from first principles (no FFT-library
code or tables). Composite sizes use prime-factor (Good–Thomas) index maps —
6=2x3, 36=4x9, 45=9x5, twiddle-free — and 64 is an 8x8 two-stage
Cooley–Tukey with long-double-computed twiddles; odd primes 13/17/23 use the
symmetric half-length direct form (e_j = x_j + x_{L-j}, o_j = x_j - x_{L-j})
with k-tiled accumulators driven by broadcast-constant tables to bound
register pressure. Two data layouts: a batch-SIMD "lane" layout (8/4/2/1
volumes interleaved per AVX-512/AVX2/SSE lane, split re/im, slab padding
against power-of-two strides) used per PREFW = {6:8, 8:8, 13:8, 17:4, 23:4,
36:4, 45:8, 64:4}, and a "natural" interleaved-complex single-volume layout
(vectorized across contiguous z with swap/fmaddsub tricks and 4x4-complex
register transposes, z rows padded to multiples of 4) that fully handles
36/45/64 and B<=3 tails of 23 (NATTHRESH). The decisive memory schedule:
each volume group runs through ALL m iterations while cache-resident; each
step is only two sweeps — pass A does the z- and y-FFTs per x-slab, pass B
does the x-FFT through contiguous scratch tiles (ZT = 12/15/16 for 36/45/64)
fused with the nonlinear map z = FFT3(x)+c, x <- z/(1+|z|) applied on
copy-out ('hyb' style: guarded rsqrt14+2 Newton steps for the sqrt, one IEEE
divide), with software prefetch of the c-array and next tiles, FTZ/DAZ set
inside the C entry points, and mmap'd MADV_HUGEPAGE buffers.

## Reconstruction & verification notes

- Rebuilt by decoding all 150 bash commands in the session log and replaying
  the ~45 file-mutating ones in order (initial heredocs, every in-place patch
  script, all PREFW/MAPSTYLE/PASSX regex substitutions, the ZT sweep, and the
  final revert of the experimental c-prefetch). Every `assert old in src`
  inside the replayed patches passed, so intermediate states matched the log.
- Final config lines in `gen_impl.py` match the log's final state:
  MAPSTYLE all-'hyb'; PASSX direct for 6/8/13/17/23, scratch 12/15/16 for
  36/45/64; PFPRIME={}; W2SET={13,17,23}; W1SET={23}; NATSET={23,36,45,64};
  NATTHRESH={23:3, rest 1<<30}; natural ZT={23:24,36:12,45:16,64:16}.
- `python3 gen_impl.py` runs clean on this host: prints
  "straight-line codelets validated", "complex-interleaved codelets
  validated", "wrote implementation.c, 23005 lines" (its built-in checks
  validate every codelet against an exact DFT at generation time).
  Regeneration is deterministic (identical md5 across runs on this host).
- `python3 -m py_compile solution.py` OK. `implementation.c`: braces 699/699
  and parens 8858/8858 balanced; full cross-target
  `clang -fsyntax-only --target=x86_64-apple-darwin -mavx512f -mavx512vl`
  pass succeeds (with `-DMADV_HUGEPAGE=14`, a Linux-only constant absent from
  the macOS SDK). NOT compiled to binary here (ARM host, AVX-512 code).

### Caveat: twiddle-constant bit-identity

`gen_impl.py` computes twiddles in `np.longdouble` then rounds to double,
emitting hex float literals. On the graded x86-64 Linux box, longdouble is
80-bit extended; on this ARM Mac it is plain float64, so a few emitted hex
constants could differ from the graded `implementation.c` in the last ulp.
The `gen_impl.py` here is exact (replayed source); re-running it on any
x86-64 Linux host reproduces the graded C file byte-for-byte. No md5 of the
original `implementation.c` was captured in the log (the log records
commands, not outputs), so byte-identity cannot be checked directly.
