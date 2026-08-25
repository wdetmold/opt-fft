# Cross-arch advisory: gen_r6 (Ice Lake, scored) vs xarch_clx_r6 (CLX, advisory)

| L | ICX winner | ICX vs-lib | CLX winner | CLX vs-lib | finding |
|---|---|---|---|---|---|
| 10 | gen_batchlane | 3.96x | gen_pfa_small | 2.74x | winner changed gen_batchlane->gen_pfa_small; ratio degraded 3.96x->2.74x |
| 12 | gen_batchlane | 4.06x | gen_batchlane | 2.77x | ratio degraded 4.06x->2.77x |
| 15 | gen_pfa_small | 3.76x | gen_pfa_small | 2.62x | ratio degraded 3.76x->2.62x |
| 20 | gen_batchlane | 3.48x | gen_batchlane | 2.48x | ratio degraded 3.48x->2.48x |
| 25 | gen_powp | 3.38x | gen_powp | 2.25x | ratio degraded 3.38x->2.25x |
| 27 | gen_powp | 3.24x | gen_powp | 2.23x | ratio degraded 3.24x->2.23x |
| 31 | gen_rader | 8.48x | gen_rader | 6.15x | ratio degraded 8.48x->6.15x |
| 32 | gen_pow2 | 3.04x | gen_pow2 | 2.19x | ratio degraded 3.04x->2.19x |
| 40 | gen_pfa_large | 2.52x | gen_pfa_large | 2.20x | ok |
| 50 | gen_powp | 2.27x | gen_pfa_large | 1.95x | winner changed gen_powp->gen_pfa_large |
| 100 | gen_pfa_large | 1.71x | gen_pfa_large | 1.66x | ok |

Flags for the next round:
- L=10: winner changed gen_batchlane->gen_pfa_small; ratio degraded 3.96x->2.74x
- L=12: ratio degraded 4.06x->2.77x
- L=15: ratio degraded 3.76x->2.62x
- L=20: ratio degraded 3.48x->2.48x
- L=25: ratio degraded 3.38x->2.25x
- L=27: ratio degraded 3.24x->2.23x
- L=31: ratio degraded 8.48x->6.15x
- L=32: ratio degraded 3.04x->2.19x
- L=50: winner changed gen_powp->gen_pfa_large

## Interpretation (for rounds 7-8)
All 11 cells still WIN on Cascade Lake (1.66-6.15x). The uniform ~30% ratio compression at the
compute-bound cells (L<=32) is the CLX 512-bit license downclock hitting our zmm-dense kernels
harder than MKL's native-dispatch paths — clock physics, not kernel bugs; the memory-bound cells
(40, 100) barely move, as expected. The load-bearing finding is the TWO WINNER CHANGES (L=10:
batchlane->pfa_small; L=50: powp->pfa_large): the plan-time race picked different variants on
different silicon, which is exactly the mechanism the library ships. Actionable if anyone wants
the CLX ratios back: race 256-bit (ymm) variants of the small-L kernels — on CLX they may clear
the downclock; the race will keep zmm on ICX automatically.
