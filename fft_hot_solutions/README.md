# Hot-round (round 3) FFT solutions — `fft3d-fixed-geometry-opt-hot-20260823`

Six completed attempts, reconstructed from transcripts with the round-2 replay
methodology (in-place generator edits replayed; prior-round engines wired to the repo
copies). ISA-ideal score curve (1.0 at r ≤ 0.078 = 12.8×; both prior rounds provided).

**All four top graded scores are grading artifacts** (see `../TIMINGS.md` for the full
audit table): 0.98→honest 0.81, 0.95→0.87, 0.94→0.84, 0.92→0.84. The solutions
themselves are integrity-clean and sit at the genuine frontier (best honest r ≈ 0.207,
`hot_502912a3`). Genuinely new pieces this round: a ping-pong SoA-8 phase-split prime
engine for L=13, pad-to-8 ragged-batch routing, a new L=64 one-sweep driver (−15%),
per-region `#pragma GCC optimize` scheduler-flag fixes, and several honest refutations
of inherited machine-note claims. Per-attempt READMEs carry grading tables, provenance
(verbatim / replayed / regenerated / vendored), and each attempt's own final
self-benchmark — the reconciliation reference for bare-metal comparison.
