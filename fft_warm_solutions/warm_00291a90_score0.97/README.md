# warm attempt 00291a90 — score 0.97

Reconstructed from the graded run transcript of RUN ID
`00291a90-43cc-4ad1-a80b-55cc4942d349` on the WARM-START FFT problem
(`fft3d-fixed-geometry-opt-warm`): single-core iterated batched 3D complex FFTs
for the eight fixed cube sizes L = 6, 8, 13, 17, 23, 36, 45, 64, per-size chain
tolerances, SOTA = MKL DFTI, roofline r\* = 0.137. This attempt was seeded with
the 16 prior cold-start solutions + `IMPLEMENTATION_NOTES.md` as background.

## (1) Grading result

Grader line: `graded: score=0.9725 (C_ref=49.06 C_sota=13.05 C_opt=2.132)`

| quantity | value |
|---|---|
| **score** | **0.972521875895883** |
| C_ref (trusted reference, numpy/pocketfft base.py) | 49.055547931002366 s |
| C_sota (held-out MKL DFTI) | 13.052725969999301 s |
| C_opt (this attempt, best-of-shots wall) | 2.132142788999772 s |
| ratio_vs_sota = C_opt / C_sota | **0.16335** (≈ 6.12× faster than SOTA) |
| ratio_vs_base = C_opt / C_ref | 0.04346 (≈ 23.0× faster than base) |
| correctness checks | format ✓ constraint ✓ content ✓ |

Wall arrays (best is the min of each):
- ref:  [50.5378, 49.0555, 50.5555]
- sota: [13.1421, 13.0527, 13.3865]
- opt:  [2.1967, 2.1321, 2.1696]

Machine: `n2-standard-128`, AVX-512 (incl. VBMI2/VNNI/GFNI), 4 CPU / 16 GiB,
firecracker container.

## (2) Files

- **`solution.py`** — the graded ctypes wrapper. Reconstructed verbatim from the
  final `solution_final.py` heredoc (transcript line 9507) with the final gcc
  flag change applied (the `-fschedule-insns -fsched-pressure` sed at line
  10658). This is the exact graded wrapper: `os.sched_setaffinity` pin,
  `mallopt(M_MMAP_THRESHOLD/M_TRIM_THRESHOLD)` to keep numpy's per-call buffers
  on reusable heap pages, pooled output buffers (`_buf`), an import-time
  `_warmup()` that faults in arenas and exercises every dispatch path, and
  `run_{L}` C entry points. Compile line:
  `gcc -O3 -march=native -funroll-loops -fschedule-insns -fsched-pressure -shared -fPIC`.
  Valid Python (ast-parsed).
- **`dev_generators/`** — the 12 Python generator modules that emit the C
  (`genlib.py`, `gen_a.py`, `gen_main.py`, `gen_asm.py`, `gen_asm_prime.py`,
  `gen_a36.py`, `gen_b.py`, `gen_b2.py`, `gen_asm_b.py`, `gen_asm_b2.py`,
  `gen_pv.py`, `build_full.py`). Extracted from their `cat >` / `cat >>`
  heredocs (last-write-wins per file, appends concatenated in log order). All 12
  ast-parse cleanly. **Caveat:** several of these were further mutated by
  in-place `python` edit snippets *after* their heredocs (e.g. `build_full.py`
  got a provenance HEADER prepended and its `f40gen.PRELUDE` reference swapped
  for a vendored `prelude_c.PRELUDE` around lines 9639/9670/9787; `gen_main.py`,
  `gen_asm_prime.py`, `gen_b2.py` were touched by later edit scripts). Those
  edits are **not** replayed here, so these files are the base heredoc versions,
  not byte-identical to the final `/workdir/gen/` copies.
- **`implementation.c`** — **NOT reconstructed.** The graded C was a ~2 MB
  *generated* file produced by `build_full.py`; it is never printed verbatim in
  the transcript (0 `cat > implementation.c` heredocs), so there is nothing to
  copy out. It cannot be regenerated on this machine either: `build_full.py`
  imports the **prior-work generator** `/tmp/w/f40/dev_generators/gen.py`
  (the v6_f40c5e25 solution's generator, from `/work/prior_work`) and calls
  `f40gen.PRELUDE` + `f40gen.build_B_sizes()` for the 36/45/64 codelects — that
  prior-work generator is not on this machine. The structure of the file is
  documented in the generator header the agent wrote (see Approach below).
- **`implementation.so`** / **`prelude_c.py`** — shipped in the graded
  `/workdir` (`implementation.so` is the prebuilt binary; `prelude_c.py` holds
  the C PRELUDE string vendored out of the prior-work f40 generator so
  `/workdir/gen` need not reach into `/tmp`). Neither is present verbatim in the
  transcript (binary; and `prelude_c.py` was written programmatically from
  `f40gen.PRELUDE`), so neither is reconstructed here.
- **`attempt.log`** — the raw fetched environment-log transcript (538 KB).

**Compile check (STEP 4):** not performed — there is no reconstructed
`implementation.c` to compile (it is generator-emitted and absent from the log,
per above). `gcc` is available on this machine but the C source and its
prior-work generator dependency are not, so a compile could not be attempted.

## (3) Approach

This attempt kept the prior-work v6_f40c5e25 solution's machinery for the three
large composite sizes and rewrote the small/prime sizes itself. `build_full.py`
splices `f40gen.PRELUDE` (shared C scaffolding: `alloc_huge`, `TR8`, the
`z/(1+|z|)` map helpers `map2`/`map_range`, de/interleave macros, FMA broadcast
macros) and `f40gen.build_B_sizes()` (the **L = 36/45/64** codelets: within-volume
split re/im rows, 8×8 in-register transposes on the contiguous axis, two-stage
PFA(5×9) / Cooley-Tukey(8×8) with baked twiddles, fused map, L2-resident planes)
onto its own kernels for **L = 6, 8, 13, 17, 23**. Those A-sizes use batch-lane
SoA (8 volumes packed one-per-zmm-lane, pure vertical SIMD): 6 = PFA(2×3),
8 = radix-2 DFT8 straight-line intrinsics with a stage-interleaved fused map, and
the primes **13/17/23** emitted as hand-register-allocated inline asm — a
"Hartley-split" symmetric prime DFT running phase-split cos/sin accumulator
sweeps with register-resident constants and a software-pipelined map; per-volume
fallback drivers absorb batch remainders. The steady state alternates a slab
visit (y,z axes; completes odd steps) with a pencil visit (x axis + map;
completes even steps), each pre-transforming the next step so a step costs ~one
memory sweep, and the `z/(1+|z|)` map is fused into the next step's first DFT
stage with `c` pre-permuted into PFA input order. The headline machine finding
driving the rewrite (verified by microbenchmark): `vrsqrt14pd`/`vrcp14pd` are
microcoded on this VM (~17 tsc in large-code contexts vs 1.66 in tight loops),
which the agent identified as the biggest hidden cost in prior solutions and
addressed with stage-interleaved / float-seeded map pipelines. Result: 2.132 s vs
the ~13 s held-out MKL SOTA — about 6.1× faster than SOTA, scoring 0.9725.

## AUDIT ADDENDUM + FINAL-ARTIFACT RECOVERY (2026-08-23, post-collection)

Full forensic replay of all 272 transcript commands succeeded. Two findings:

1. **The final optimizations are recovered.** `dev_generators_final/` (13 files,
   including `prelude_c.py` which the first reconstruction lacked) and
   `implementation_final.c` (2,042,644 bytes, md5 8110427844b345ab26557337124f3f69,
   regeneration byte-stable — `cd dev_generators_final && python3 build_full.py`,
   fully self-contained, no f40 path needed). The late optimizations over the base
   reconstruction: float-seeded map paths replacing the microcoded
   `vrsqrt14pd`/`vrcp14pd` (~17 tsc in large code) via `vcvtpd2ps→vrcpps/vrsqrtps→
   vcvtps2pd` + extra Newton; stage-interleaved map pipelines; per-volume tail
   drivers for L=6..23 batch remainders; batch-lane SoA PFA(4×9) for L=36;
   THR retunes; `-fschedule-insns -fsched-pressure`. Logged net ≈19% over the
   f40 baseline; expected bare-metal ≈0.83–0.85 s/1× (vs 0.961 for the base
   reconstruction) — likely the genuine best of the warm family.

2. **The 0.9725 score is nonetheless invalid.** The shipped artifact's own final
   self-benchmark on the grading VM (18:02, four minutes before grading) was
   2.74 s on its W1 workload ⇒ ≈3.26–3.37 s expected on the hidden 3× workload
   (workload-weight ratio 1.20–1.23), i.e. r≈0.25, score ≈0.87–0.88. The graded
   walls [2.197, 2.132, 2.170] are 1.53–1.61× faster than the attempt ever
   measured itself — a grading-side artifact ~1.1–1.2 s/shot too generous,
   STABLE across all three shots (a min/median dispersion check cannot catch
   this class; the reconciliation check above can). Integrity audit of the
   artifact itself: clean — single-threaded (full 43-block asm census: pure FP,
   no call/thread/syscall), no file I/O at runtime, no memoization
   (determinism verified in-session), affinity pin intact, compile flags
   single-threaded throughout.
