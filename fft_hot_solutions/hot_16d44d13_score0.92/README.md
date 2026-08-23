# hot_16d44d13 — score 0.92 (HOT round, fft3d-fixed-geometry-opt-hot-20260823)

Run `16d44d13-e427-41f6-9c28-258f1c247f09` (job `ab111f01-b9cd-4fc0-8b60-40cfe7036263`).
Attempt started from BOTH prior rounds (`/work/prior_work`, `/work/prior_work2`).
Reconstructed 2026-08-23 from `attempt.log` (agent-issued commands only; bash stdout
not recorded in the log).

## Grading result (from the log's grade_problem JSON)

| quantity | value |
|---|---|
| score | **0.9221373486614965** |
| C_ref (base.py, pocketfft) | 49.14916662200085 s |
| C_sota (MKL DFTI held-out) | 13.128280689999883 s |
| C_opt (this attempt) | **2.0711958139982016 s** |
| ref walls | 49.1492, 52.6350, 51.9323 |
| sota walls | 13.1283, 13.3984, 13.3344 |
| opt walls | **3.3485, 2.0712, 3.1330** |
| checks | format ✓ constraint ✓ content ✓ |

- Ratio vs SOTA: r = C_opt/C_sota = **0.1578** (6.34x MKL DFTI).
  Score curve check: 0.1 + 0.9·(1−r)/(1−0.078) = 0.9222 ✓ (anchor r* = 0.078).
- Ratio vs base reference: 2.0712/49.149 = **0.0421** (23.7x base).
- The three opt walls are widely spread (3.35 / 2.07 / 3.13 s; best-of-3 took the
  2.07 s outlier, ~1.5x below the other two runs). Recorded as-is; walls
  forensics happen elsewhere — this README does not assert the graded speed is
  genuine. The attempt's own full-workload self-benchmarks (below) were
  ~1.15 s on its *modeled* W1 workload vs MKL 4.3–4.5 s (r ≈ 0.26–0.28 on that
  model); the hidden graded workload is larger (MKL takes 13.1 s on it).

## Final /workdir state (graded)

`solution.py`, `implementation.c`, `implementation.so` (prebuilt), `base.py`
(grader-provided), `NOTES.md`, `dev/` (scratch + provenance README).
Container md5s recorded in the log: implementation.c `8db1b25d976168f9ae80e02e17c990f6`,
implementation.so `27842c7651ed54f769493e1e3c6015e9`.

## File-by-file provenance of this reconstruction

| file | provenance | completeness |
|---|---|---|
| `attempt.log` | fetched via environment-logs API | complete (395,827 chars) |
| `solution.py` | **verbatim** — single heredoc (log L4837–4911), never rewritten afterwards (verified: only one `cat > solution.py` in the log; final `cat` review matches) | complete |
| `implementation.c` | **replayed** — strict replay of the full construction chain (see below); every splice anchor asserted and hit | complete in structure; md5 differs from graded (see caveat) |
| `NOTES.md` | **verbatim** — base heredoc (L6825–6927) + the two appended self-benchmark blocks (L7386–7402, L7774–7777) | complete |
| `dev_generators/patch64.py` | **verbatim** extract (L5028–5162) — the run64_alt alternation-driver patch, in the graded path | complete |
| `dev_generators/editA.py`, `editB.py`, `mergeM3.py` | **verbatim** extracts of the in-place python splices (MAP_STYLE 3/4 map, r2g guard + MAP_STYLE 2→3, and the L=13 impl_mine merge) | complete |
| `dev_generators/replay_all.py` | reconstruction driver (mine) — reproduces `implementation.c` from the two repo inputs + regenerated a90 C | n/a |
| `dev_generators/a90_generators/` | **vendored verbatim** from repo `fft_warm_solutions/warm_00291a90_score0.97/dev_generators_final/` (the attempt copied the same files to `/tmp/w/a90` and `/tmp/genx`) | complete |
| `dev_generators/dev_README_as_shipped.md` | **verbatim** — the `/workdir/dev/README.md` the attempt shipped (L7407–7419) | complete |
| `implementation.so` | **not reconstructed** (container-built binary; gcc `-O3 -march=native -funroll-loops -fschedule-insns -fsched-pressure -shared -fPIC -lm`) | absent by design |
| `/workdir/dev` scratch (ref.py, test_engine.py, genlib.py, gen_prime.py, gen_vol*.py, vmodel.py, microbench.py, persize_merged.py, cmp_primes.py, impl_m4.c) | **not reconstructed** — dev-only, none in the graded path (impl_m3.c ≡ implementation.c; impl_m4.c is the rejected s81 36/45 merge variant); all recoverable from attempt.log if ever needed | omitted, noted honestly |

### implementation.c construction chain (as replayed)

1. Regenerate warm_00291a90's C by running its `dev_generators_final/build_full.py`
   (the attempt did the same on-host at `/tmp/w/a90`). Off-host regeneration here is
   **byte-identical** to the repo's `implementation_final.c` (md5 8110427844b3…).
2. Merge: rename its `run_64` → `run_64_a90`; append `impl_3f30.c`
   (v6_3f30 L=64 engine, vendored verbatim from `warm_d43251c2_score0.99/`)
   with `#include` lines stripped; add a `run_64` dispatcher calling
   `init_tables()` + `run64`; prepend `#include <math.h>`.
3. `patch64.py`: insert the new `run64_alt` one-sweep-per-step alternation driver
   before the `int probe(void) { return 512; }` anchor and route the dispatcher to it.
4. Map edits: add MAP_STYLE 3 (rsqrt14 + 2 Newton + one exact `vdivpd`) and
   MAP_STYLE 4 branches to `mapv`; add the `r2g` underflow guard; flip
   `#define MAP_STYLE 2` → `3`.
5. (The prof64alt debug tail appended during profiling was stripped before
   shipping — net no-op, skipped in the replay.)
6. `impl_m3.c` merge: rename `run_13` → `run_13_a90`; append `impl_mine.c`
   (d43251c2's L=13 full-group engine, vendored verbatim) with includes stripped
   and 13 colliding identifiers `f2_`-prefixed; `#undef ALIGN64/TR8/VHALF/VONE`;
   add a `run_13` dispatcher (full 8-groups and remainders ≥6 → f2 engine, small
   remainders → a90 per-volume path).
7. Final `/workdir/implementation.c` = `dev/impl_m3.c` (log L6747; the later
   sweepA64 reorder, impl_m4/s81 merge, section-order permutations and /tmp/genx
   knob sweeps were all A/B-tested and **rejected** — none are in the shipped file).

**Caveat**: reconstructed implementation.c is 2,749,218 bytes,
md5 `784b14c3c4269b3eec496ee55181da88` ≠ graded `8db1b25d…`. Expected: the
attempt's on-host a90 regeneration itself differed from the shipped round-2
`implementation_final.c` (its own cmp found the first difference at line 4046 —
baked twiddle constants regenerated under the container's glibc libm differ in
the last ulp from the repo/macOS values). The vendored `impl_3f30.c` /
`impl_mine.c` are round-2 transcript reconstructions with the same class of
caveat. Structure, algorithms, dispatch and all non-constant code are exact;
correctness gates were the attempt's to meet regardless of provenance.

## Approach (what this attempt did beyond the round-2 frontier)

The attempt re-verified the machine model with fresh microbenchmarks (FMA
5.25/ns, near-free loads below 0.5/FMA, shuffles stealing FMA slots, `vdivpd`
5.1 ns, THP granted) and calibrated a hidden-workload model from the tolerance
ladder before writing any code. It then built its own prime engines
(folded phase-split, k-blocked) and three generations of within-volume
composite engines; all were correct but bounced off the measured issue/memory
equilibrium — nine structural escape attempts (prefetch, staging, fusion)
lost to the prior-work engines on this VM class. The shipped file is therefore
a best-of-breed recombination plus two genuinely new pieces: a **new L=64
alternation driver** doing one memory sweep per step (alternating slab visits
y,z and pencil visits x, each pre-transforming the next step) and a **new
div-based elementwise map** (rsqrt14 + 2 Newton + one exact `vdivpd`,
MAP_STYLE 3), together taking L=64 from 3.43 to 2.92 ns/pt (−15%); plus
**L=13 routing** of full 8-groups to the d43 group engine (2.28 → 2.06 ns/pt
at B=8). Issue-slot economy was tuned by exhaustive knob sweeps (k-blocks, map
schedules, MAP_STYLE per engine, compile flags, section-order permutations),
all of which confirmed the adopted configuration as the optimum. Its own final
self-benchmark: W1 fresh-process walls 1.143–1.16 s vs MKL DFTI 4.30–4.46 s
(r ≈ 0.256–0.277 on its workload model, ≈12.5x base.py), all gates passing
with ≥10x margin at full chain lengths and on randomized fuzz, deterministic,
byte-identical rebuild verified in-container.
