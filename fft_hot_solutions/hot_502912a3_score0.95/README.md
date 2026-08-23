# hot_502912a3 — score 0.95 (HOT round, fft3d-fixed-geometry-opt-hot-20260823)

Run id: `502912a3-ce68-4238-b904-42d47d478283`
(job `ab111f01-b9cd-4fc0-8b60-40cfe7036263`, graded 2026-08-23T23:04Z)

## Grading result (verbatim from the grader)

| quantity | value |
|---|---|
| score | **0.9510764179390147** |
| C_ref (base.py, pocketfft) | 47.731 s |
| C_sota (MKL DFTI, held out) | 12.963 s |
| C_opt (this attempt) | **1.6608 s** |
| ref walls | 47.731, 48.875, 49.025 |
| sota walls | 12.963, 13.243, 13.544 |
| opt walls | 1.6864, **1.6608**, 1.7619 |
| checks | format ✓ constraint ✓ content ✓ |

Ratios: C_opt/C_sota = **0.1281** (7.80× SOTA); C_opt/C_ref = 0.0348
(28.7× the pocketfft reference). On the round-3 curve (0.1 at r = 1 vs
SOTA, 1.0 at r ≤ 0.078) r = 0.1281 gives exactly the recorded 0.9511.
The graded speed is recorded as-is; walls forensics happen elsewhere.

## Graded /workdir contents and provenance

| file | provenance | completeness |
|---|---|---|
| `solution.py` | **verbatim** — extracted from the single heredoc that wrote it (log 22:20:20Z); never rewritten afterwards | complete |
| `implementation.c` (~3.04 MB) | **regenerated** — final generator state replayed, then run against the repo engine copies (see below). The generator does pure string surgery on the three engine files (no floating-point is computed at generation time), so the output should be byte-identical to the graded file provided the repo engine copies match what was mounted at `/work` in the container | complete (regenerated) |
| `implementation.so` | **not reconstructed** — prebuilt binary; the attempt verified in-session that gcc on the shipped `.c` reproduces it byte-identically (md5 `e2b0f7ee7f4185920402ac28000b1ee0`, in NOTES.md). `solution.py` rebuilds it when absent | absent by design |
| `NOTES.md` | **verbatim** — base heredoc (22:48Z) plus the "Final validation" append (23:00Z), both extracted exactly | complete |
| `dev_gen/assemble.py` | **replayed** — base heredoc (22:07Z full rewrite) + the complete chain of 11 in-place python edit snippets (log write-sites 4620, 4666, 4725, 5430, 5446, 5481, 5811, 5950, 5970, 6019, 6356). Strict replay: every `assert` passed, including reproducing the one silent in-session `str.replace` miss (5430) and its fix (5481). Saved here as `dev_generators/assemble.py` | complete |
| `dev_gen/build.sh` | **verbatim** — final rewrite heredoc (22:44Z): `gcc -O3 -march=native -funroll-loops -shared -fPIC`. (The build that produced the shipped `.c`/`.so` at 22:36Z used the earlier, flag-identical version) | complete |

`dev_generators/` here additionally holds `_base_assemble.py` + `_edit_*.py`
(the extracted edit chain, provenance for the replay), `assemble_local.py`
(paths rewired to the local repo copies — this is what emitted the
`implementation.c` in this directory), and `my_kernels.c` (a single newline:
the attempt shipped with its own kernels **disabled**, exactly as in the final
in-container build, which it md5-verified against `/workdir` at 22:53Z).

### Engine inputs consumed by the generator (vendored prior work, baked into the .c)

- `W00` → `fft_warm_solutions/warm_00291a90_score0.97/implementation_final.c` — L = 6, 8, 13, 17, 23, 36 engines, compiled under `#pragma GCC optimize("O3","unroll-loops","schedule-insns","sched-pressure")`
- `F30` → `fft_v5v6_solutions/v6_3f30d81f_score0.88/implementation.c` — L = 64 chain-resident engine, under `("O3","unroll-loops","no-math-errno","no-trapping-math")`
- `S39` → `fft_warm_solutions/warm_d43251c2_score0.99/impl_3907.c` (v5_3907583b's engine as vendored by d43251c2) — L = 45 for B ≥ 8; `w00_run_45` handles B < 8 (measured crossover baked into the shim)
- `DM` → `.../impl_mine.c` is still assigned in the final generator but **unused** (the L = 13 graft it fed was A/B-tested and reverted at 22:26Z)

In-container these were read from `/work/prior_work{,2}` and `/tmp/cand/d43/`
(the latter copied verbatim from `warm_d43251c2_score0.99/impl_*.c`), i.e.
all inputs trace to the two prior-round repo directories. Verified: the local
W00 copy md5-matches the value recorded in its own README
(`8110427844b345ab26557337124f3f69`). Note the prior-round repo copies are
themselves transcript reconstructions (their READMEs carry the caveats), but
they are the very bundles mounted to the attempt as `/work/prior_work{,2}`,
so the regeneration input equals the graded input.

## Approach (what this attempt did beyond the round-2 frontier)

The attempt spent most of its budget trying to *beat* the round-2 engines
rather than just recombine them: it built a numerically self-verifying
IR-to-intrinsics generator plus a hand-asm emitter (register-resident folded
twiddles, sign-folded FMAs) and produced correct from-scratch engines for
L = 13/17/23/36 in three architectures (batch-lane SoA, within-volume
L2-resident, asm pencils) — all of which converged to, but never past, the
measured ~2-uop/cycle issue equilibrium the incumbents already sit at, so
none shipped. Along the way it re-measured and partially *refuted* the
inherited machine notes: `vdivpd`/`vsqrtpd` are ~16/~24 cyc on real data
(the 4-cyc figures were constant-input early-exit artifacts), the
rsqrt14+Newton map (~19 uops/8 elems) is the floor, and
`-fschedule-insns`/`-fsched-pressure` halves its generated codelets while
helping w00's — which is why the final single file compiles each engine
under per-section `optimize` pragmas matching its native flags. The shipped
artifact is therefore a measured per-size optimum over all 21 prior engines
plus glue: w00 for 6/8/13/17/23/36, 3f30 for 64 (3.94 → 3.35–3.38 ns/el-step),
3907 for 45 with a B < 8 fallback to w00 (3.45–3.50 → 3.14–3.33), plus
wrapper-level `mallopt` heap tuning and an import-time warmup that pre-faults
all arenas. Its own final self-benchmarks: 1.154–1.186 s vs MKL-DFTI
4.95–5.27 s on a 360M element-step chain-heavy shape (r ≈ 0.22–0.24),
1.433 s vs 4.674 s batch-heavy (r ≈ 0.307), 3.443 s vs 16.65 s at 3× scale
(r ≈ 0.207); one-step blocks at 3.5e-16–1.0e-15, all per-size chain gates
passed with ≥ 1e4 margin, 14/14 randomized fuzz workloads, deterministic.
Note the graded r = 0.128 is substantially better than every self-measured
ratio (grader walls: opt 1.66 s vs sota 12.96 s) — recorded without
interpretation, as walls forensics are out of scope here.
