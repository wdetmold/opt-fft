# Warm-start FFT solutions (`fft3d-fixed-geometry-opt-warm`, 2026-08-23)

Reconstructed `/workdir` states of the five completed attempts on the **warm-start**
variant of the single-core FFT code-opt problem. This variant is identical to the
main v7 problem (eight fixed cube sizes L = 6, 8, 13, 17, 23, 36, 45, 64; iterated
map `z = FFT3(x) + c ; x ← z/(1+|z|)`; per-size chain tolerances; held-out SOTA =
MKL DFTI; score anchored at the measured-machine roofline **r\* = 0.137 ≈ 7.3× MKL**),
except that each attempt was **given the 16 prior cold-start solutions plus
`IMPLEMENTATION_NOTES.md` as `/work/prior_work`** and told to read them as background,
verify what it needed, and make its own strategy choices.

`IMPLEMENTATION_NOTES.md` in this directory is the exact notes bundle those attempts
started from (machine model, prior strategies, the roofline, and caveats).

## What warm-start bought

Same task, gates, and roofline curve as the cold-start cohorts:

| cohort | completed scores | best r = C_opt/C_sota | env-failed (C_opt guard) |
|---|---|---|---|
| cold v5/v6 | 0.74 – 0.91 | ~0.28 (≈3.6× MKL) | — |
| cold v7 | 0.77, 0.78, 0.79, 0.81, 0.82 | ~0.24 | 3/8 |
| **warm** | **0.89, 0.90, 0.93, 0.97, 0.99** | **0.145 (6.88× MKL)** | 3/8 |

Handing agents the prior work lifted the completed band from ~0.80 to ~0.93. The headline
**0.99 was audited and is a measurement artifact**: its opt walls were [3.90, 4.09, 2.02] —
min/median 0.49 — and the 2.02 s shot is *below the physical floor of that binary* (its
bare-metal rebuild runs 0.857 s per 1× pass, so the 3× graded workload needs ≥ 2.57 s even
at bare-metal speed, and the VM's own honest shots imply ~1.3 s/1×). The mechanism is a
CPU-steal burst landing in that shot's zero-work *baseline*, inflating the subtraction;
best-of-shots then latched the under-measurement, which sat above the 1.0 s plausibility
floor. Honest score from the 3.90 s shot: **≈ 0.85** (r = 0.28). The audit found the
solution itself integrity-clean (no FFT library, no threads, no memoization). The 3/8 env-failures are the recurring import-preamble timing
artifact (heavy variable import work floors best-of-shots → `C_opt = 1e-6` → the score_fn
plausibility guard converts it to an env-failure rather than a false 1.0); the rate is the
same as cold v7 and independent of warm-start.

## Provenance and completeness (read before benchmarking)

These are reconstructed **from the run transcripts**, which capture the agent's issued
write/edit commands but **not bash stdout** and **not large generator-emitted files that
were never printed verbatim**. The only source of the exact graded bytes is each run's
`out_dir` tar in GCS (not fetched here). Completeness therefore varies per attempt:

| attempt | score | C_opt | ratio vs MKL | state | runnable as-is? |
|---|---|---|---|---|---|
| `d43251c2` | 0.99 → **honest ≈ 0.85** (audited) | 2.02 s (artifact; honest 3.90 s) | 0.280 (3.6×) | **complete** — wrapper + regenerated `impl_mine.c` + 3 vendored prior-work engines | **yes** (x86) |
| `00291a90` | 0.97 | 2.13 s | 0.163 (6.1×) | wrapper + 12 generators; `implementation.c` (~2 MB, generator-emitted) **not recoverable** from transcript | no — needs regeneration |
| `361a3485` | 0.93 | 2.67 s | 0.206 (4.9×) | wrapper + generators; `implementation.c` generator-emitted, **not recovered** | no — needs regeneration |
| `57053476` | 0.90 | 2.97 s | 0.229 (4.4×) | wrapper + **best-effort** reconstructed `implementation.c` (tolerant edit-replay; **may diverge** from graded source) | build-and-check |
| `53ebdad6` | 0.89 | 3.15 s | 0.241 (4.2×) | wrapper + `implementation.c` regenerated via a traceable patch chain on the `v6_f40c5e25` base | build-and-check |

Costs: base (numpy/pocketfft) C_ref ≈ 47–53 s; held-out MKL DFTI C_sota ≈ 13 s (best-of-3).

**Vendoring note (all permitted):** the warm-start rule allows building on the supplied
prior solutions. `d43251c2` (0.99) reuses three prior cold-start engines verbatim
(`impl_3907.c` = v5_3907583b for L=6/8; `impl_s81.c` = v5_8175a973 for L=36/45 + a
23-fallback; `impl_3f30.c` = v6_3f30d81f for L=64) and wrote its own generator-emitted
prime engine (`impl_mine.c`) for the rest; `impl_f40.c` is present but unused in the graded
routing. `00291a90` and `53ebdad6` similarly build the large composite sizes from the
prior `v6_f40c5e25` generator. So the fastest attempts are genuinely *composed on top of*
the prior kernels, as intended. No FFT-library binary appears in any graded path (audit
clean).

## Benchmarking on bare-metal Ice Lake

- All kernels are x86 AVX-512; they were tuned against the **grading VM's measured
  ceilings** (issue cap ~2.1 vector-uops/cyc, ~64 B/cyc L1 — see `../ROOFLINE.md`), which
  are tighter than bare-metal Ice Lake-SP. Expect faster absolute times on real hardware
  and shifted MKL-relative ratios; the roofline itself moves down roughly by the
  issue-cap/port ratio off the VM.
- Generator-emitted C bakes twiddle constants computed in x86 80-bit `long double`; the
  reconstructions were regenerated on an arm64 host, so the low-order bits of some baked
  constants may differ from the shipped bytes. Regenerate the generators on the x86 target
  (or rely on the per-size 1e-14 one-step gate) if you need exact constants.
- Each `solution.py` compiles its C at import (`gcc -O3 -march=native …`) and binds via
  ctypes; the per-attempt README lists the exact compile line and engine→size routing.

If you want the **exact graded bytes** for the two generator-only attempts (0.97, 0.93)
and the best-effort one (0.90), those live in each run's GCS `out_dir` tar — recoverable
with bucket access, which this reconstruction did not have.
