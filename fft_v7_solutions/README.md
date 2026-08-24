# v7 (roofline-round) FFT solutions — `fft3d-fixed-geometry-opt-20260821` v7

Five completed cold-start attempts on the v7 curve (0.1 at MKL parity → 1.0 at the
measured-issue-cap roofline r ≤ 0.137 = 7.3×). No prior work provided. Scores 0.77–0.82,
honest C_opt 4.18–4.68 s (r ≈ 0.31–0.36, ≈2.8–3.2× MKL) — all walls stable (min/median
≥ 0.97), no artifacts, so these graded numbers are trustworthy as-is. This is the
genuine cold-start frontier the warm/hot rounds were seeded to beat.

Reconstructed from transcripts by full command replay. Two attempts are hand-written C
(complete, no generator); three are generator-emitted C regenerated from their recovered
generators — byte-stable in-session, with the standard x86-longdouble twiddle caveat
(regenerate on x86-64 Linux for exact graded bytes). Per-attempt READMEs carry grading
tables, provenance, approach, and self-benchmarks. See `../TIMINGS.md` for the
cross-round ledger.
