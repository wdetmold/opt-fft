# warm attempt 361a3485 — score 0.93

Warm-start attempt on `fft3d-fixed-geometry-opt-warm` (single-core, 8 fixed cube
sizes L = 6, 8, 13, 17, 23, 36, 45, 64; per-size chain tolerances; SOTA = MKL
DFTI; roofline anchored at r\* = 0.137). This attempt was given the 16 prior
cold-start solutions (`v5_*`/`v6_*`) + `IMPLEMENTATION_NOTES.md` as background.

Run id: `361a3485-b6ca-48b9-900e-e880491b5fda` (short `361a3485`).
Reconstructed from `attempt.log` (the Taiga environment-log stream).

## (1) Grading result

| quantity | value |
|---|---|
| score | **0.9283526713796518** |
| C_ref (trusted base, pocketfft/numpy) | 47.6634775170005 |
| C_sota (held-out MKL DFTI) | 12.998159621000013 |
| C_opt (this attempt, best-of-shots) | 2.673745186000815 |
| ratio to SOTA (C_opt / C_sota) | **0.2057** |
| ratio to base (C_opt / C_ref) | 0.0561 |
| checks | format ✔ constraint ✔ content ✔ |

Best-of-3 walls from the grading section:
- `opt`:  [2.673745, 2.771638, 2.838885]  (C_opt = min = 2.6737)
- `sota`: [13.967361, 12.998160, 13.165407] (C_sota = min = 12.9982)
- `ref`:  [49.107512, 47.663478, 47.964952] (C_ref = min = 47.6635)

The score matches the linear-in-r curve `score = 0.1 + 0.9·(1 − r)/(1 − 0.137)`
with r = C_opt/C_sota = 0.2057 → 0.928, i.e. the scoring reference is the held-out
MKL SOTA and this attempt runs ~4.86× faster than MKL DFTI on the graded mix
(and ~17.8× faster than the pocketfft base).

## (2) Files

- **`solution.py`** — the graded ctypes wrapper. Recovered **verbatim** from the
  log (final `cat > solution.py`, finalized mid-session and never rewritten
  after). Skeleton-verbatim outside the marked regions; the marked regions add
  the gcc flags (`-funroll-loops -fno-math-errno -fno-trapping-math`), glibc
  `mallopt` heap tuning (raise `M_MMAP_THRESHOLD`/`M_TRIM_THRESHOLD` to 1 GiB so
  numpy's per-call buffers stay on the reusable sbrk heap), the ctypes bindings
  to `init_tables` + `run{L}`, an import-time `_warmup()` that faults arenas and
  exercises every dispatch path, and a pooled-output-buffer `_run`.
- **`implementation.c`** — **NOT reconstructed** (see truncation note below).
  The graded C file was generator-emitted and its final consolidated source is
  never printed in full in the log.
- **`dev_generators/`** — the generator pipeline sources that ARE present in the
  log, as starting-point/initial versions (several were later mutated in place
  by `sed`/Python-`.replace()` edits and `git checkout` reverts that are not
  consolidated anywhere in the transcript):
  - `assemble.py` — early version: reads the hand-maintained template C, inserts
    `pgen`-generated prime kernels (L=13/17/23) before `GEN_WRAPPERS`, rebinds
    the AoSoA drivers, writes `impl_gen.c`. (The final `/workdir/gen/assemble.py`
    reads `template.c` with a prepended HEADER and also wires in the L=36/45
    ports; that final form is not dumped in full.)
  - `pgen_initial.py` — initial prime-codelet generator (symmetric-folded direct
    prime DFTs, k-blocked accumulators). Later extended in-session to emit
    asm-broadcast register constants (`VCA`/`PT{p}` tables) and consumption-order
    `c` — those edits are in the log as `.replace()` patches, not as a final dump.
  - `v2gen.py` — the batched two-stage PFA codelet generator (used for L=36/45
    and the small-size lane-major paths).
  - `redo_impl.py` — base-template transformer: patches the prior-work
    `v6_3f30d81f` `implementation.c` to swap in the AoSoA3 kernels (`GEN_RUN_AOSOA3`),
    the rewritten `k6`/`k8` codelets, `KSTORE3`, and consumption-order conv-in.
  - `aosoa3.txt`, `k6.txt`, `k8.txt`, `convmap.txt` — the C fragments
    `redo_impl.py` splices into the template.
  - `port36.inc` / `port45.inc` are referenced by the final `assemble.py` (the
    adapted L=36/45 two-stage codelets, ported from prior-work `v5_8175a973`) but
    are themselves produced programmatically in-session and never dumped in full,
    so they are **not** included here.
  - **Base template**: the pipeline started from the on-disk prior-work solution
    `../../fft_v5v6_solutions/v6_3f30d81f_score0.88/implementation.c`
    (1689 lines / 83 KB), which `redo_impl.py`/`assemble.py` transform. That file
    is the documented starting point; it is not copied here to avoid duplicating
    the sibling solution.
- **`attempt.log`** — the raw environment-log transcript this reconstruction was
  built from.

### Compile check (STEP 4)
N/A on two counts: (a) no byte-exact `implementation.c` was reconstructed, and
(b) this reconstruction host is arm64 with only Apple clang — the kernels are
hand-written AVX-512 (`_mm512_*`, `vbroadcastsd` asm), x86-64 only, so they would
not build here even if present. The attempt itself compiled and graded cleanly
in-container (`gcc -O3 -march=native -funroll-loops -fno-math-errno
-fno-trapping-math -shared -fPIC`), and its own `gen/assemble.py` re-run produced
an `impl_gen.c` byte-identical to the deployed `implementation.c` (`REGEN MATCHES`
in the log).

### Truncation note
`implementation.c` cannot be reconstructed byte-exact from this transcript. It is
generator-emitted: base = prior-work `v6_3f30d81f/implementation.c`, transformed
by `redo_impl.py`, then by `assemble.py` splicing in `pgen`/`v2gen` codelets plus
the `port36.inc`/`port45.inc` L=36/45 ports. Across the session the template C
(`implementation.c` / `template.c`), `pgen.py`, and the `.inc` ports were mutated
by ~12 `cat >>` appends, several in-place Python `.replace()`/`sed` edits, and
multiple `git checkout` reverts — and the final consolidated template.c / pgen.py
/ port*.inc are never printed in full in the log. Per the reconstruction rules,
missing code is not fabricated; the recoverable generator sources are saved above
and the method is documented from the transcript.

## (3) Approach

Started from the prior-work bundle and kept its overall architecture — one C
file of hand-written AVX-512 kernels, per-size specialized, chunk-resident over
the whole m-step chain, everything carried in IEEE fp64 — then rebuilt the
kernels through its own generator pipeline rather than shipping the prior code
unchanged.

- **Small composites L=6, 8**: lane-major SoA (8 volumes packed one-per-zmm-lane),
  PFA(2×3)/radix-2×4 butterflies; sweeps in Y then X with the elementwise
  `z = FFT3(x)+c`, the `z/(1+|z|)` map, and the *next* iteration's Z pass all
  fused into the trailing pass. Batch-remainder tail uses a within-volume variant
  (z in lanes, 8×8 in-register transposes).
- **Primes L=13, 17, 23**: lane-major, symmetric-folded **direct prime DFTs**
  ((p−1)/2 distinct cos/sin kept register-resident via `vbroadcastsd` inline-asm
  loads from aligned `PT{p}` tables — needed because gcc otherwise folded the
  "register constants" back into memory-operand FMAs), k-blocked accumulators
  (KB = 3/2/4), with `c` stored in consumption order. This was the main change vs
  prior work (the `pgen.py` asm-broadcast + consumption-order rewrite).
- **Large composites L=36, 45**: batched two-stage PFA(4×9)/(5×9) codelets,
  outlined (`noinline,noclone`), split re/im lanes, with the `z=FFT+c`-then-map
  pass pipelined one slab behind — adapted from the prior-work reconstruction
  `v5_8175a973` (`port36.inc`/`port45.inc`).
- **L=64**: within-volume, slot = (a,y,z) with lanes = low 3 bits of x; radix-8×8
  vertical kernels for z/y; four-step x pass with one in-register transpose;
  `c` in consumption order; map fused into the x pass; next iteration's z pass
  fused per y-row.
- **The map** `z/(1+|z|)`: `rsqrt14`+2 Newton for `|z|`; `1/(1+|z|)` via `rcp14`+2
  Newton or one hardware divide, alternating per output slot so the divider and
  FMA pipes overlap. All twiddles baked as hex fp64 from long-double generation.
  Arenas 2 MB-aligned THP-backed with per-arena skew; strides padded against
  4K/L1-set aliasing.

Self-reported (not graded): one-step blocks ≈ 3–9e-16 rel-L2 vs the long-double
matrix-DFT reference (gate 1e-14); per-size chain gates pass with ≥10³–10⁶×
margin; ~15–22 % faster than the best of the 16 prior attempts on every workload
shape tested. Composite sizes sit at their measured L3-bandwidth wall (~30–40
GB/s), primes at the ~2.1 uops/cycle issue ceiling.
