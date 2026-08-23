# warm attempt d43251c2 — score 0.99

Reconstructed final `/workdir` state of one completed attempt on the
**warm-start** FFT code-opt problem (`fft3d-fixed-geometry-opt-warm`):
iterated batched 3D complex FFTs, single-core, eight fixed cube sizes
L = 6, 8, 13, 17, 23, 36, 45, 64, per-size chain tolerances, SOTA = MKL DFTI,
roofline anchored at r* = 0.137. This attempt was seeded with the 16 prior
cold-start reconstructions + `IMPLEMENTATION_NOTES.md` as background.

Run id `d43251c2-83ee-4e3b-b2af-69dccf41c69f`.

## (1) Grading result

| quantity | value |
|---|---|
| **score** | **0.9913445689** |
| C_ref (base.py, pocketfft) | 53.19256883 s |
| C_sota (held-out MKL DFTI) | 13.89850890 s |
| C_opt (this attempt, best-of-shots) | 2.01944774 s |
| ratio to SOTA (C_opt / C_sota) | 0.1453  (≈ 6.88× faster than MKL) |
| ratio to base (C_opt / C_ref) | 0.0380  (≈ 26.3× faster than base) |
| checks | format ✓ constraint ✓ content ✓ |

Wall arrays from the grader (best-of-three each):
- `ref`:  [53.193, 56.437, 56.701]
- `sota`: [13.920, 13.989, 13.899]
- `opt`:  [3.895, 4.088, **2.019**]  → C_opt = min = 2.019 s

The scored ratio to SOTA (0.145) sits essentially at the machine roofline
(r* = 0.137), which is why the score lands at ~0.99.

## (2) Files

Graded `/workdir` files (last-write-wins from the transcript):

- `solution.py` — the ctypes wrapper (the marked-fill-in of the required
  skeleton). Builds and loads **four** self-contained hand-written-DFT C
  libraries and routes each `(L, B)` to the fastest engine; adds batch
  tail-splitting for 13/17/23. Reconstructed by replaying, in order, the 6
  applied `/workdir/solution.py` write/edit operations from the log
  (blocks 49→128→169→174→178→247→263→265); every `assert old in src` in the
  replayed edits passed, validating the reconstruction. One superseded
  attempt (a syntactically-broken f-string variant of the routing edit) was a
  no-op in the agent's run and is correctly excluded.
- `impl_mine.c` — the novel prime engine (13/17/23, plus unused 6/8/36/45
  codelets), **generator-emitted**. Regenerated here by running the
  reconstructed `dev_generators/gen.py` with default environment (the exact
  invocation the transcript's byte-identical-regeneration check used). 616 KB,
  brace-balanced, exports `run_6/8/13/17/23/36/45`; `solution.py` binds only
  13/17/23/36/45.
- `impl_3907.c` — engine adopted verbatim from prior work
  `v5_3907583b_score0.87/implementation.c` (AoSoA 8-volume lanes, PFA/radix
  codelets). Used for L = 6, 8 and as the small-B fallback. Copied from the
  local prior-work reconstruction (identical to the bundle the attempt was
  given). Exports `init_all` + `run`.
- `impl_s81.c` — engine adopted verbatim from `v5_8175a973_score0.90`
  (PFA batched groups). Used for L = 36, 45 and the 23 small-B fallback.
  Exports `run_size`.
- `impl_3f30.c` — engine adopted verbatim from `v6_3f30d81f_score0.88`
  (per-volume chain-resident, lanes = low x-bits). Used for L = 64.
  Exports `init_tables` + `run64`.
- `impl_f40.c` — engine copied into `/workdir` early (from
  `v6_f40c5e25_score0.91`) but **not referenced** by the final `solution.py`
  (routing never returns 'A'); kept for fidelity, unused in the graded path.
- `dev_generators/gen.py` — the generator that emits `impl_mine.c` (the
  `/workdir/dev/gen.py` of the graded state). Reconstructed by replaying its
  edit chain from the log; standalone `sys.path` fix applied (blocks 241/242).
  Frozen at the block-257 state — i.e. **excludes** the later block-260 ZPERM
  edit, which was applied only to the agent's scratch copy and is inert at
  default env (verified: it changes only `#if 0/#else` scaffolding, not the
  active emitted code).
- `dev_generators/dftgen.py` — DFT-codelet emitter imported by `gen.py`
  (`Emitter`, `emit_dft`, `emit_dft8`). Written once (block 101), never edited.
- `dev_generators/genasm.py` — inline-asm helper imported by `gen.py`. The ASM
  path was explored and abandoned ("dead end"); imported at module load but not
  used in the default emitted engine.
- `attempt.log` — raw Taiga environment-log stream this reconstruction was
  built from.

**Compile check (STEP 4):** the host running this reconstruction is `arm64`
(Apple Silicon); all four engines use x86 AVX-512 intrinsics (`<immintrin.h>`,
`__m512d`), so `gcc -O3 -march=native -c impl_mine.c` fails with
`"This header is only meant to be used on x86 and x64 architecture"`. This is a
host-architecture mismatch, not a code defect — `gen.py` ran cleanly on this
same host and emitted complete, brace-balanced C, and the attempt itself
compiled and scored 0.99 on the x86 grading machine.

## (3) Approach

Started from the prior-work bundle and kept it as a per-`(L, B)` engine router
rather than a single kernel. The three "adopted" engines are reused verbatim
from the provided reconstructions (explicitly permitted): `impl_3907` for
L = 6/8 and small-B fallbacks, `impl_s81` for the L = 36/45 PFA composites, and
`impl_3f30` for L = 64 (discovered to be ~20% faster than the previous best
64-path at small B). The new contribution is `impl_mine.c` for the prime sizes
13/17/23: batched 8-volume SoA groups with symmetric-folded direct prime DFTs
(≈4h² FMAs/pencil), phase-split into register-resident cos/sin halves with
k-blocking, fused per-plane z+y sweeps and an x-sweep with consumption-ordered
`c`, and the elementwise `z/(1+|z|)` map **deferred into the next iteration's
z-pass loads** using a hardware-divide / Newton-Raphson mix. Twiddle/permutation
tables are baked at generate time; the whole chain stays in IEEE double to clear
the per-size gates (one-step blocks ≈1.2e-15 vs the 1e-14 bar). Final tuning:
per-`(L, B)` route thresholds, 4K-alias-free padded strides, and batch
tail-splitting (13/17/23, B ≥ 8: sub-group remainders dispatched to the
per-volume engine instead of padding a half-dead 8-lane group).

**Reconstruction caveat:** `impl_mine.c` was regenerated off-host. Baked twiddle
constants are computed in `numpy.longdouble`, which is 80-bit extended on the
x86 grading machine but not on this arm64 host, so the low-order bits of the
hardcoded constants here may differ from the exact shipped bytes. The generator
is identical and the kernel structure/approach is faithful; this is the same
provenance caveat the problem statement and project notes flag for
generator-emitted C.
