# warm attempt 57053476 — score 0.9

Reconstruction of the graded `/workdir` for one completed attempt on
`fft3d-fixed-geometry-opt-warm` (warm-start: given the 16 prior cold-start
solutions + `IMPLEMENTATION_NOTES.md` as background). Run
`57053476-6f0c-4b27-beba-8deddc53c074`.

## (1) Grading result

Source: `attempt.log` grading section (`code_opt.grading: graded: score=0.9043`),
best-of-shots wall per implementation.

| quantity | value |
|---|---|
| **score** | **0.9042944352379831** |
| C_ref (base.py, held-out reference) | 47.71887459100071 s |
| C_sota (held-out MKL DFTI) | 13.002710044002015 s |
| C_opt (this attempt, best shot) | 2.9746430140003213 s |
| ratio_vs_sota = C_opt / C_sota | 0.22877 (4.37× faster than SOTA) |
| ratio_vs_base = C_opt / C_ref | 0.06234 (16.04× faster than reference) |
| roofline anchor r\* | 0.137 (score 1.0) |

Correctness gates all passed: `format=true, constraint=true, content=true`.

Per-implementation wall arrays (grader took the min of each):
- ref:  `[47.7189, 48.1395, 47.9913]`
- sota: `[13.0999, 13.3904, 13.0027]`
- opt:  `[2.9985, 3.1462, 2.9746]`

## (2) Files

- **`solution.py`** — the graded ctypes wrapper. **Exact, verbatim** from the
  transcript (single `cat > /workdir/solution.py` heredoc, never re-written).
  Compiles `implementation.c` with `gcc -O3 -march=native -funroll-loops
  -shared -fPIC ... -lm`; binds `init_all()` + `run_size(L,B,m, x0,c, one,final)`;
  reuses per-`L` output buffers from a pool; runs a `_warmup()` over all sizes
  at import (excluded from the timed call).
- **`implementation.c`** (363,655 B, 11,108 lines) — **regenerated here** by
  `python3 dev_generators/gen.py implementation.c`. See the fidelity note
  below: this is the *latest verbatim generator snapshot in the log*, an
  **intermediate-session SoA-8 build**, **not the exact ~855 KB graded C**
  (which is never emitted verbatim in the transcript). It covers all 8 sizes
  and defines `init_all` + `run_size` with `case` arms for 6/8/13/17/23/36/45/64.
  Compile check (STEP 4): **fails on this host** — the machine is arm64
  (Apple clang) and the kernel is x86-only (`immintrin.h` / AVX-512
  `__m512d`, `_mm512_*`). This is a platform mismatch, not a code defect; the
  file targets the x86 grading machine.
- **`dev_generators/`** — the Python code-generator modules. Each is the
  **last complete `cat >` heredoc snapshot** of that module found in the log
  (verbatim). `gen.py`, `netlib.py`, `primelib2.py` are the set that actually
  produces the included `implementation.c` (`gen.py` imports only those two).
  `genvol.py`, `gen64z.py`, `maplib.py`, `pipelib.py`, `prime23.py` are the
  later volume-era modules the agent also carried into `/workdir/dev_generators`;
  they are **not wired into the `gen.py` snapshot included here** (see note).
  - `AGENT_PROVENANCE_NOTE.md` — the agent's own `dev_generators/README.md`,
    verbatim, describing the final module roles.
- **`attempt.log`** — full raw environment-logs transcript.

## Fidelity note (important — read before trusting `implementation.c`)

The graded `implementation.c` is **generator-emitted and never dumped verbatim**
into the transcript (the log contains only the generator's incremental edits,
grep snippets, and benchmark output). The agent evolved `gen.py` and its helper
modules through ~70 in-place programmatic edits (`open('gen.py','w').write(g)`
after `g.replace(...)`/`g.index(...)` splices) plus `sed` edits and two `cat`
re-creations. Those edits use positional splicing that hard-fails unless applied
to the exact prior intermediate state, and those intermediate states are also
not dumped — so the edit chain **cannot be replayed** to recover the final
sources byte-for-byte (a strict replay diverges with cascading `assert old in g`
failures; a tolerant replay silently drops dependent edits).

What is therefore recoverable and included: `solution.py` (exact), the grading
numbers (exact), the approach (below), and the **latest verbatim generator
snapshot**, which produces a correct, self-consistent **363 KB SoA-8** build.
The agent's own note and the final `run_size`/README indicate the graded build
was a larger **~855 KB** hybrid that additionally wired in the per-volume engine
(`genvol.py`/`gen64z.py`) for L=64 and batch remainders. The included
`implementation.c` represents the **SoA-8 core** of that approach, not the full
final hybrid. Nothing has been fabricated to fill the gap.

## (3) Approach

Warm start: the agent first read `IMPLEMENTATION_NOTES.md` and benchmarked
several prior solutions (notably `v5_3907583b` @0.87 and `v5_8175a973` @0.90),
then built its **own Python code-generator** rather than shipping a prior file,
emitting a single self-contained C file (links only `libm`; Rule 1 clean).

Kernel strategy (per the agent's provenance note + generated C):
- **SoA-8 batch-lane engine** (`gen.py`): 8 batch volumes are packed across the
  8 doubles of a 512-bit `zmm` lane and transformed together — an x-pass then
  per-plane y/z passes, with the elementwise map `z/(1+|z|)` **fused into the
  x-pass stores** for most sizes. Split re/im (structure-of-arrays) throughout.
- **Per-size factorization / map** (`plan(L)`): PFA (prime-factor / Good–Thomas,
  CRT index maps, twiddle-free) for L=6 (2×3), 36 (4×9), 45 (9×5); a direct
  straight-line codelet for L=8; symmetric **component-phased prime DFTs**
  (`primelib2.py`, register-budgeted k-splits with j-outer accumulation) for the
  primes 13/17/23; **Cooley–Tukey 8×8** for L=64.
- **Twiddles**: baked at generation time as exact mod-N constants computed in
  numpy `longdouble` (x87 80-bit on the x86 host) and emitted as hex doubles —
  full fp64 carried through the whole chaotic iteration to clear the per-size
  gates (1e-4 … 1e-10). No FFT library is used.
- **Map arithmetic**: hardware `sqrt` combined with `rcp14`/`rsqrt14`
  Newton–Raphson refinement for the reciprocal in the elementwise step.
- The graded build extended this with `genvol.py`'s **per-volume engine**
  (padded split re/im rows, vertical y/x passes, z-pass via in-register 8×8
  transposes; `gen64z.py` radix-8 z-codelet) for L=64 and batch remainders — the
  part not present in the SoA-8 snapshot reconstructed here.

Result: 4.37× faster than the held-out MKL SOTA and 16.0× faster than the
pocketfft reference, for a score of 0.904 against the r\*=0.137 roofline.
