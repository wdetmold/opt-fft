# Warm cohort measured on our Ice Lake (a80n0, bare metal, pinned, best of 6)

All C regenerated ON THIS NODE before timing (the shipped reconstructions were
generated on an arm64 laptop; numpy.longdouble there is not x86 80-bit extended,
so baked constants differed — d43251c2's impl_mine.c and 57053476's
implementation.c both changed on regeneration; timing unchanged within noise).

| attempt (score) | total chain s (1x m) | notes |
|---|---|---|
| warm_d43251c2 (0.99) | **0.857** | exact everywhere (two-step 1e-15/16); grader shots [3.9, 4.1, 2.0] — the scored 2.0 s was a lucky VM window; true pace is 0.85-class |
| warm_00291a90 (0.97) | 0.961 | REBUILD from pre-final generators (three lost edits reconstructed: register-pool merge guard, prime x_mns emit, composite mns arity); chain-end exact at all 8 sizes; step-1 snapshot layout at L=6/8 scrambled (lost final edit) — chain timing unaffected. Graded artifact was likely faster than this rebuild. |
| warm_57053476 (0.90) | 0.844 (7 sizes) | L=64 engine returns garbage (reconstruction gap, unchanged by regeneration) |
| warm_53ebdad6 (0.89) | 0.958 | complete, exact |
| warm_361a3485 (0.93) | not runnable | final assemble.py inputs (port36/45.inc) never dumped in the transcript |

Per-size bests across the cohort (chain seconds, gate-clean):
6: 0.0651 (d43) | 8: 0.0843 (57) | 13: 0.1451 (57) | 17: 0.0305 (d43) |
23: 0.0845 (00291a90) | 36: 0.0530 (d43) | 45: 0.1866 (d43) | 64: 0.1748 (d43)

## Update: 00291a90 FINAL artifact (dev_generators_final + prelude_c.py, x86-regenerated)

The true final generator state (pushed later) regenerates the graded 2.04 MB C on our
node (the shipped implementation_final.c was ARM-generated: ~97k bytes of low-order
constant digits differ). Measured: **0.832 s total**, exact one-steps everywhere, the
step-1 snapshot bug of the pre-final rebuild gone. Its L=6 = 0.0592 s is the new
cohort-best cell. Residual gap to its graded 2.13 s shots (= 0.71 s at 1x) is
consistent with the grading tier's higher clock (GCP n2 Ice Lake turbo ~3.4 GHz vs
our Gold 6326 ~3.0), not measurement luck.

Cohort bests, final: 6: 0.0592 | 8: 0.0835 | 13: 0.1451 | 17: 0.0305 | 23: 0.0845 |
36: 0.0530 | 45: 0.1681 | 64: 0.1748  (committee total 0.799 s)
