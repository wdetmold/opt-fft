# Attempt 3f30d81f — fft3d-fixed-geometry-opt (v6, per-size gates) — score 0.884

Reconstructed final graded `/workdir` state of attempt `3f30d81f`, replayed
edit-by-edit from the session log
(`attempt_3f30d81f_score0.88.log`, grade at 06:33:23 on 2026-08-23).

## Grading record (from the log)

- Problem: `fft3d-fixed-geometry-opt-20260821` — **v6** (per-size final-block
  tolerances: L=6: 1e-4, 8: 3e-6, 13: 1e-9, 17/23/36/45/64: 1e-10; one-step
  blocks gated at 1e-14).
- Score: **0.8843692523652322**
- Costs (best of 3 walls): `C_ref = 133.37 s`, `C_sota = 22.05 s`,
  `C_opt = 6.04 s` (opt walls 6.04 / 8.04 / 9.87). That is ~22x over the
  numpy/pocketfft reference and ~3.7x faster than the held-out SOTA.

## Files

- `solution.py` — the required ctypes wrapper (verbatim problem skeleton plus
  the marked fill-ins: compile flags, `init_tables`/`run{L}` bindings, and a
  `_run` that coerces B/m to int and forces C-contiguity). At import it
  rebuilds `implementation.so` if absent.
- `implementation.c` — single self-contained C file, all transform code
  hand-written (links only libm). 1689 lines, md5
  `104ae8d0d0fd7e223bd88450e23ab856`.
- `dev_generators/` — the in-session generator scripts:
  - `genprime2.py` — final prime-kernel generator (sign-folded conjugate-pair
    DFTs; modes `both`/`cos` control which twiddle sets are preloaded into
    registers). The shipped kernels use config `13:2:both,17:3:cos,23:2:cos`.
  - `regenerate_kernels.py` — reruns genprime2 and splices the section into
    `implementation.c`; **verified byte-identical** to the shipped section.
  - `genprime.py` — earlier generator revision (historical; superseded).
  - `aosoa_new.c` — source block for the pipelined AoSoA driver macro that was
    spliced into `implementation.c` (already contained in it; kept for
    provenance).
- Not reconstructable: `implementation.so` (binary was never logged). It is
  irrelevant for grading provenance: `solution.py` rebuilds it from
  `implementation.c` at import time with the exact command below.

## Compile command

```
gcc -O3 -march=native -funroll-loops -fno-math-errno -fno-trapping-math \
    -shared -fPIC implementation.c -o implementation.so -lm
```

(AVX-512F required; the file `#error`s without it. Do not build on ARM.)

## Algorithm summary

Fully specialized, single-threaded AVX-512 implementation of the iterated
batched 3D DFT + `z/(1+|z|)` map for the eight fixed cube sizes, designed so
each batch block/volume stays cache-resident across its entire m-step chain
(RAM touched only for input conversion and the two snapshots). Data layout is
"paired split complex": logical slot q holds 8 real parts at vd index 2q and 8
imaginary parts at 2q+1, so all arithmetic is vertical SIMD with no shuffles in
the hot loops. Per size: L∈{6,8,13,17} use AoSoA with lanes = 8 batch volumes
(L=8 gets a padded 68-slot x-slab stride to kill 4K aliasing); L∈{23,36,45} are
in-volume with lanes = 8 consecutive z, the z-axis DFT done via in-register 8x8
tile transposes and the x-axis DFT run directly on aliasing-free strides;
L=64 puts lanes = the low 3 bits of x with padded row/slab strides (66/4232
slots), plain radix-8^2 vertical kernels for z/y, and a four-step x-pass with
one in-register transpose. Odd-prime DFTs (13/17/23) are generated code:
symmetric conjugate-pair kernels, k-blocked (2 or 3), with the (p-1)/2 distinct
twiddle magnitudes kept in registers and sign-folded FMAs — tuned to the
machine's measured ~1x512-bit memory-op/cycle limit. The elementwise map
(rsqrt14/rcp14 plus two Newton steps each) is fused into the x-pass stores, and
the next iteration's z-pass is pipelined behind each completed row group.
Composite kernels 36=6x6, 45=5x9(3x3), 64=8x8 are buffered Cooley–Tukey with
long-double twiddles generated at init with exact mod-L angle reduction.

## Reconstruction provenance

- The log records model actions only (no command outputs), so no in-log
  md5/wc/compile transcripts exist to check against; verification here was:
  full deterministic replay of all 30+ edits with change assertions (two
  in-session scripts that aborted before writing were replayed and confirmed
  to fail identically), `python3 -m py_compile solution.py`, code-level
  brace/paren/bracket balance of the C (balanced after stripping comments and
  strings), structural marker checks, and byte-exact regeneration of the
  generated kernel section from `genprime2.py`.
- The session survived three rate-limit container restarts; the harness
  replayed all prior actions into fresh containers each time, which the
  reconstruction mirrors (the replays are no-ops on file state).
- The C was NOT compiled here (ARM host; AVX-512-only source).
- solution.py md5: `d91be82f2299b34728e9b3a73ee0f852` (60 lines).
