# hot_262a05c6 — score 0.94 (fft3d-fixed-geometry-opt-hot-20260823, round 3 / HOT)

Reconstructed final graded /workdir state of run `262a05c6-6846-4093-bf4c-1cf0808a06a4`
(job ab111f01-b9cd-4fc0-8b60-40cfe7036263). The attempt started from BOTH prior rounds'
solutions (`/work/prior_work` = 16 round-1 solutions, `/work/prior_work2` = 5 warm-round
solutions) and scored **0.9446**.

## Grading result (from the transcript's grade_problem JSON)

| quantity | value |
|---|---|
| score            | 0.9445594293870975 |
| C_ref (pocketfft base) | 48.27662905899979 s |
| C_sota (MKL DFTI, held out) | 12.994219000999692 s |
| C_opt (this attempt) | 1.7515659450000385 s |
| ref walls        | 49.840133753999, 48.27662905899979, 48.327200239000376 |
| sota walls       | 12.994219000999692, 13.049934870999095, 13.38914333899993 |
| opt walls        | 1.7867347120009072, 1.7515659450000385, 1.9047148990002825 |
| checks           | format ✓, constraint ✓, content ✓ |

Ratios: **C_opt/C_sota = 0.1348** (7.42x SOTA), C_opt/C_ref = 0.0363 (27.6x the pocketfft
reference). Score curve check: 0.1 + 0.9·(1−0.1348)/(1−0.078) = 0.9446 ✓ (r is measured
against SOTA; full marks at r ≤ 0.078). Per standing policy the graded speed is recorded,
not asserted genuine — walls forensics happen elsewhere. The attempt's own NOTES.md flags:
any graded wall materially below ~0.9x of its self-benchmarks should be treated as a
measurement artifact; here C_opt 1.75 s sits inside the self-measured W1/W2/W4 band
(1.36–1.80 s per 1x mix), i.e. no red flag on its face.

## Directory contents / provenance

| file | provenance | completeness |
|---|---|---|
| `solution.py` | replayed: last full heredoc write + two subsequent in-place python edits (padded-8 remainder path added; `goff`→`eoff` revision). Compiles clean. | complete |
| `NOTES.md` | replayed: final heredoc + end-of-session python append (last self-benchmark block) | complete |
| `engines/impl_a.c` | regenerated off-host by running `warm_00291a90_score0.97/dev_generators_final/build_full.py` from the repo copy (exactly what the attempt did on-host from `/work/prior_work2`). Our output is byte-identical to the repo's `implementation_final.c`; the on-host graded copy differed from that in last-ulp baked twiddles only (attempt verified this itself). | reconstructed, last-ulp caveat |
| `engines/impl_b64.c` | vendored verbatim = `fft_warm_solutions/warm_d43251c2_score0.99/impl_3f30.c` (diff-verified identical) | complete |
| `engines/impl_s81.c` | vendored verbatim = `fft_warm_solutions/warm_d43251c2_score0.99/impl_s81.c` (diff-verified identical) | complete |
| `engines/impl_v8p.c` | regenerated: full ordered replay of this attempt's own generator pipeline (genlib/genprime2/gen23wv + inline assembly scripts, 7 successive rewrites). 316,399 bytes, includes the dz_17/dz_23 experiments that were built but routed away. | reconstructed, last-ulp caveat (libm-baked constants) |
| `engines/impl_v8b.c` | regenerated: final `genbdrv.py` state replayed (incl. the plane-prefetch add **and its revert** — the shipped engine has row-level prefetches only), then `python3 genbdrv.py` re-run. One `sed -i` edit (L=64 row-stride pad removal, "packed rows") failed silently under BSD sed during automated replay and was re-applied by hand before the final regeneration — tolerant replay, disclosed. | reconstructed, last-ulp caveat |
| `gen_v8/` | the six generators the attempt itself shipped to /workdir (genlib, genb, genbdrv, genprime2, gendrive, gen23wv + README.txt), in final replayed state | complete |
| `bench/` | shipped benchmark mixes (wl…wl4.py, e2e.py + README.txt) | complete |
| `dev_generators/` | full final /tmp/v8 generator suite incl. the not-shipped experiments (genprime.py v1, genb2/genb3 pipelining studies, gen68/gen68b 6-and-8 engines, gen64x) | complete as of session end |
| `attempt.log` | full environment transcript (466 KB) | complete |

Not reconstructable / intentionally absent: `base.py` (grader-provided pristine file),
`engines/*.so` and `__pycache__` (x86 binaries built on-host; `solution.py` rebuilds every
.so from source when missing, so the state is functionally complete), and the MKL DFTI
surrogate (calibration only, deliberately kept out of /workdir by the attempt).

Reconstruction method: all 203 bash commands were extracted from the transcript and
replayed in order in a sandbox (last-write-wins plus in-place edit replay). Heredoc-aware
`&&`→`;` tolerance let file writes land even where compiles/benchmarks could not run on
this machine; the single silent divergence found (BSD `sed -i` no-op on genbdrv.py) was
corrected manually and the downstream generator re-run. Sandbox paths were reverse-mapped
out of the four text files that mention /tmp//workdir. All engine-content markers audited
against the transcript (col_dft_pipe, init_pcol_36, prefetch counts, dz_17 presence,
padded-8 routing in solution.py).

## Approach (what this attempt did beyond the round-2 frontier)

The attempt first rebuilt and gate-verified the best recombined prior artifact (warm
00291a90's SoA-8 asm engine A for 6/8/13/17/23/45-tails, d43251c2's vendored impl_3f30
for 64 and impl_s81 PFA(4x9)/(5x9) for 36/45 8-groups), then built an MKL DFTI surrogate
purely for calibration. Its genuinely new work is two engines: **impl_v8p**, a
phase-split symmetric-folded direct prime DFT (fold to sum/difference halves, cos-phase
accumulation with register-resident constants, spill, sin-phase, combine) with ping-pong
arenas so every load — including the x-pass — is contiguous and all strided traffic is
stores, plus a consumption-ordered c array and a 17-op rsqrt14/rcp14+Newton map, which
beat the prior best at L=13 by ~6–9% (its 17/23 variants tied/lost and were routed back
to the warm asm engine); and **impl_v8b**, a within-volume L2-resident two-pass engine
(split-[8re|8im] 128-byte vector points, two-stage PFA columns through L1 scratch,
software-pipelined column pairs, TR8 plane transposes, parity-alternating orientation
with two pre-permuted c copies) which wins L=36 at B<8 and covers 36-remainders. On top
of the engine work it tightened issue-slot economy at the routing level: per-(L,B)
dispatch across four .so's, plus a padded-8 trick that pads 13/17 batch remainders ≥5–6
up to a full 8-group so the SoA-8 fast path handles ragged tails. A measured key insight
overturning round-2 lore: vrsqrt14pd/vrcp14pd are fast when the loop body stays DSB
(uop-cache) resident — the "microcode" penalty only bites MITE-resident giant bodies —
which the map exploits; conversely 2-column pair codelets and 1.6 MB pipelined variants
LOST once bodies exceeded ~4K uops. Rejected experiments (all transcript-evidenced):
dedicated 6/8 engines (gen68/gen68b), a 64-specific gen64x rewrite, zipped 13/17
(v8prime2z), whole-plane prefetch in v8b pass1 (built, measured, reverted), and an
impl_a map upgrade (hand-allocated asm judged too risky). Final self-benchmarks: per-size
1.11/1.25/2.14/2.65/3.53/2.84/3.35/3.30 ns per point-step (L=6…64); four workload mixes
1.44–1.80 s vs the MKL surrogate's 4.73–5.98 s (ratio 0.27–0.31, pulling toward ~0.2 on
prime-heavy mixes where MKL is weakest).
