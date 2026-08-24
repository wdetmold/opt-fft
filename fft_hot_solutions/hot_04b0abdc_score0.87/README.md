# HOT round (round 3) attempt 04b0abdc — score 0.8661

Problem: `fft3d-fixed-geometry-opt-hot-20260823` (single core, 8 fixed cubes
L=6,8,13,17,23,36,45,64; per-size chain tolerances; score 1.0 anchored at the
ISA-ideal r* = 0.078 vs the held-out SOTA = MKL DFTI). Attempt started from
both prior rounds' solutions (`/work/prior_work`, `/work/prior_work2`).
Run id: `04b0abdc-5678-43b7-94a6-2a03f279b9f0` (job ab111f01), graded
2026-08-23T22:45:52Z.

## Grading result (from the log's grading JSON)

| quantity | value |
|---|---|
| **score** | **0.8660970601748491** |
| C_ref (base.py, pocketfft) | 50.927 s (walls 52.197 / 50.927 / 51.306) |
| C_sota (held-out MKL DFTI) | 13.140 s (walls 13.206 / 13.361 / 13.140) |
| C_opt (this attempt) | 2.8275 s (walls 2.926 / 2.930 / 2.827) |
| checks | format ✓ constraint ✓ content ✓ |
| ratio vs SOTA r = C_opt/C_sota | 0.2152 (4.65× SOTA) |
| ratio vs base = C_opt/C_ref | 0.0555 (18.0× base) |

Score curve check: 0.1 + 0.9·(1−r)/(1−0.078) with r = 0.2152 → 0.866 ✓
(the anchor r* = 0.078 = 12.8× SOTA was not reached; this attempt sits at
4.65× SOTA, just past the best genuine round-2 ratios of ~4–4.9×).

## Files and provenance

| file | provenance | completeness |
|---|---|---|
| `attempt.log` | full environment log fetched from Taiga | complete |
| `solution.py` | **verbatim** from the log's `str_replace create` of `/workdir/solution.py` (log line 5341; last write wins — no later edits) | complete |
| `implementation.c` | **regenerated** by running the reconstructed final `dev_generators/build_combined.py`; 2,131,343 bytes, matching the attempt's own "~2.1 MB". Assembled from (a) the prior warm-round artifact `fft_warm_solutions/warm_00291a90_score0.97/implementation_final.c` (vendored verbatim from this repo — permitted prior work; its own README notes possible last-ulp constant drift vs the graded original), with its `run_36/45/64` renamed to `prior_run_*`; (b) `e64v8.c` (new L=64 engine, `MY_`-namespaced, MAPV=1); (c) `pfa.c`+`pfa_impl.h` (new L=45/36 PFA engines, suffixes `_n45`/`_n36`) plus the `run_45`/`run_36` routing wrappers. Syntax-checked cleanly (clang, x86-64 target; only Linux-specific `MADV_HUGEPAGE` unresolvable on macOS headers). | regenerated |
| `implementation.so` | **not reconstructed** (prebuilt ELF binary; not recoverable from the log). Rebuild on the target machine with the flags recorded in `solution.py`: `gcc -O3 -march=native -funroll-loops -fschedule-insns -fsched-pressure -ffp-contract=fast -fno-math-errno -shared -fPIC implementation.c -o implementation.so -lm` (solution.py auto-rebuilds when the .so is absent). | absent by nature |
| `NOTES.md` | replayed: heredoc (log 5924) + both later in-place edits (6274, 6793) | complete |
| `base.py` | grader-provided file, not written by the agent; not reproduced here | n/a |
| `dev_generators/` | final states of the generation chain, reconstructed by executing the attempt's own edit scripts in order (strict replay; every snippet ran, no tolerant substitutions). `build_combined.py` is the final generator (original `/tmp/w/...` paths preserved; execution used a path-rewired copy pointed at `fft_warm_solutions/warm_00291a90_score0.97/implementation_final.c`). `e64.c` → `e64_nopf.c` (prefetch-stripped) → `e64v6.c` (RS64 136/PS64 8728, profiling stripped) → `e64v8.c` (zcol_pipe added) is the L=64 lineage; `pfa.c`/`pfa_impl.h` the 45/36 lineage. Dev-only exploration files (primes.c/primes2.c/genprime.py, stall.c harness, e64v2/v3/v4/v5/v7/diag dead branches) were not part of the graded artifact and were not reconstructed. | complete for the graded chain |

### Reconstruction caveat worth recording (source-level fact, not speculation)

The attempt's final "software-pipelined fused stage" integration (`zcol_pipe`,
adopted as "v8, −9% on the hot loop") **never actually took effect in the
shipped C**: the two call-site `.replace()` edits at log line 6551 target the
single-argument call forms (`zcol_s2a_map_s1(CS64 + ...);` /
`col_s2_map_s1(CP64 + ...);`), but those call sites had been rewritten to the
3-argument prefetch forms at log line 2578 and never reverted — so both
replaces silently no-opped, exactly as replayed here. `implementation.c`
therefore contains `zcol_pipe` as a never-called static function, and the
L=64 engine actually runs the v6 fused path. The attempt itself observed
"Dead code is cold (never called) — harmless" when objdumping the final
binary. (The same no-op affected the diag edit at 6104.) The NOTES.md claim
that the pipeline is integrated "worth ~9%" is the attempt's belief, not the
shipped reality; the graded 2.83 s was produced by the v6-path binary.

## Approach (what this attempt did beyond the round-2 frontier)

Starting from the strongest round-2 artifact (warm_00291a90) as an installed
safety net, the attempt re-derived the machine envelope with its own
microbenchmarks and found it looser than the prior notes' 2.1-vuop/cycle
equilibrium: 2 FMA/cyc sustains with up to ~2 loads + 1 store per 8 FMAs, and
1024-byte strides suffer heavy 4K-alias stalls. It then wrote new
within-volume engines for the three big sizes — L=64 as a two-stage CT(8×8)
with all three axes stored digit-swapped so every stage reads/writes
contiguous slot groups (rows padded to 136 doubles for the alias class), 8×8
register-tile transposes fused into the z-stages, and the z/(1+|z|) map
(rsqrt14+NR sqrt, rcp14+NR reciprocal) fused between step t's completing
stage and step t+1's opening stage; L=45/36 as twiddle-free PFA(5×9)/PFA(4×9)
in PFA-input-order storage with CRT output tables. Primes 13/17/23 (and
batch-multiple-of-8 L=36) stayed routed to the prior engine after the new
direct and register-resident-constant prime codelets measured slower than the
prior hand-scheduled asm. The wrapper was rewritten to write engine outputs
in place into the final concatenated array and to warm all dispatch paths at
import. Self-benchmark on its W3 proxy (B=4, m=25000…300): best 2.199 s vs
local MKL DFTI 13.3 s and base 22.5–23.6 s — about 6× MKL locally; the
graded run measured C_opt = 2.83 s vs C_sota = 13.14 s (4.65×). Walls
forensics are out of scope here; the graded numbers are recorded above as-is.
