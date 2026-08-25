# Cross-arch advisory: gen_r5 (Ice Lake, scored) vs xarch_spr_r5 (SPR, advisory)

| L | ICX winner | ICX vs-lib | SPR winner | SPR vs-lib | finding |
|---|---|---|---|---|---|
| 10 | gen_pfa_small | 3.97x | gen_batchlane | 4.02x | winner changed gen_pfa_small->gen_batchlane |
| 12 | gen_pfa_small | 4.05x | gen_pfa_small | 4.16x | ok |
| 15 | gen_pfa_small | 3.73x | gen_batchlane | 3.78x | winner changed gen_pfa_small->gen_batchlane |
| 20 | gen_pfa_small | 3.44x | gen_pfa_small | 4.02x | ok |
| 25 | gen_powp | 3.48x | gen_powp | 3.31x | ok |
| 27 | gen_powp | 2.81x | gen_powp | 2.88x | ok |
| 31 | gen_rader | 8.47x | gen_rader | 7.39x | ok |
| 32 | gen_pow2 | 3.05x | gen_pow2 | 3.35x | ok |
| 40 | gen_pfa_large | 2.53x | gen_pfa_large | 2.39x | ok |
| 50 | gen_powp | 2.28x | gen_powp | 2.21x | ok |
| 100 | gen_pfa_large | 1.72x | gen_powp | 1.84x | winner changed gen_pfa_large->gen_powp |

Flags for the next round:
- L=10: winner changed gen_pfa_small->gen_batchlane
- L=15: winner changed gen_pfa_small->gen_batchlane
- L=100: winner changed gen_pfa_large->gen_powp
