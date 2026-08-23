# Attempt 95ab77a1 — fft3d-fixed-geometry-opt-20260821 (v5, flat gates)

Reconstructed FINAL graded /workdir state, rebuilt by mechanically replaying every
file-writing action in the session log
(`fft_codeopt/v5_audit/attempt_95ab77a1_score0.82.log`).

## Grade

| field  | value |
|--------|-------|
| score  | **0.8168** |
| C_ref (numpy base)   | 61.81 s (walls 61.81 / 62.97 / 75.93) |
| C_sota (held-out)    | 13.94 s (walls 13.94 / 15.79 / 15.09) |
| C_opt (this attempt) | 4.688 s (walls 4.69 / 5.05 / 6.15) |
| speedup vs base | 13.2x |
| speedup vs SOTA | 2.97x |

Version: **v5** — flat correctness gates (one-step blocks < 1e-14 relative L2,
m-step chain blocks < 1e-3, same thresholds for every L; not the v6 per-size gates).

## Files

- `solution.py` — graded wrapper (mandated skeleton verbatim); ctypes bindings,
  persistent hugepage arena with a +1088 B offset on the c-arena (anti 4K-aliasing),
  cached 64B-aligned output buffers, batch-lane vs per-volume path selection,
  OOM guard. Rebuilds `implementation.so` at import if absent.
- `implementation.c` — 612,080 bytes (1186-byte audit header + 610,894-byte
  generated body, 10,921 lines) of AVX-512 C emitted by `codegen.py`.
- `codegen.py`, `dftv.h` — the generator, which the agent itself shipped in
  /workdir for provenance. `dev_generators/` holds identical copies plus
  `regen.sh`; running `KB=2 python3 codegen.py` (needs mpmath) deterministically
  reproduces the implementation.c body byte-for-byte (md5 43979316e7b7de114835cbaac9e9f458,
  stable across runs; the shipped file adds the audit comment header on top).
- `implementation.so` — **not reconstructable** (prebuilt x86-64 AVX-512 binary,
  existed only in the grading container). Harmless: `solution.py` recompiles it
  with the skeleton's exact command when missing.
- `base.py` — environment-provided reference, not agent-authored, not included.

## Compile command

```
gcc -O3 -march=native -shared -fPIC implementation.c -o implementation.so -lm
```

(EWV/PFD macros default to 2 / 64; the graded build passed no -D flags.)

## Algorithm summary

Split re/im storage, 8 doubles per zmm lane. Per-size DFT networks with exact
mod-L twiddles rounded from 120-bit mpmath: PFA(2x3) for 6, CT(2x4) for 8,
symmetric half-matrix DFT (cos/sin even-odd split, (L-1)^2 FMAs, table-driven
k-blocks of 2 with the u/v fold fused into the first block) for primes 13/17/23,
PFA(4x9) for 36, PFA(9x5) for 45, CT(8x8) for 64. z-axis via in-register 8x8
transposes feeding the kernels directly (a four-step single-transpose kernel
`z64f` for L=64), y-pass interleaved per slab for L2 residency, x-pass fused
with the +c and the chaotic map z/(1+|z|) (alternating vsqrtpd and
rsqrt14+2-Newton per store to balance the divider and FMA ports; reciprocal via
rcp14+2 Newton). For L in {6,8,13,17} with B>=8, eight volumes become SIMD
lanes (`runb*`/`convball*`: shuffle-free batch path; remainder volumes fall back
to the per-volume path). Memory engineering: padded row/slab geometry (rows
padded to multiples of 8 for 6/13/17/23/36/45; 64 uses row stride 72, slab
64*72+8) against 4K aliasing and cache-set conflicts, 2 MB-aligned hugepage
arenas (`madvise(MADV_HUGEPAGE)`), volume-outer/iteration-inner loops,
non-temporal interleaved output stores, +64 B software prefetch in the strided
x-pass kernels, and a dual-destination emit for the m==1 case.

## Reconstruction / verification notes

- The session hit API rate limits twice; each requeue rolled the transcript back,
  discarding pass-1 work after 01:28 (a Rader-16/12 implementation and a z/y
  fusion rewrite) and all pass-2 work. The kept trajectory is: log lines
  231-2416 (35 actions), then 9583 to the end. All 191 kept action blocks'
  file mutations (creates, str_replace edits, python patch heredocs, seds) were
  replayed in order; 56 patch scripts ran, 0 failed, and the log's `\n`-mangled
  encoding was inverted exactly.
- The log records no tool outputs (no md5s/wc/compile logs to compare against);
  corroboration is indirect: the final in-log summary states "~600 KB of
  generated C" (we get 612 KB with header), and the shipped kernel inventory
  (z64f four-step, batch lanes for 6/8/13/17, mixed elementwise, vs_doubles/
  bps_doubles exports) all appear in the regenerated source.
- Checks run here: `python3 -m py_compile solution.py codegen.py` (OK);
  implementation.c brace balance 438/438 and paren balance 16351/16351 (OK);
  dftv.h braces 12/12; all expected symbols present (run6..run64, runb6/8/13/17,
  convert_all*/convball*, vs_doubles, bps_doubles, ensure_arena, z64f,
  ew_store_alt, emit_inter2). The C was deliberately NOT compiled on this ARM
  host (AVX-512 intrinsics).
