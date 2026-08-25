# GENERALIZE campaign — final accounting (8 rounds, closed 2026-08-25)

All numbers: chain seconds on bare-metal Ice Lake a80n0, pinned, two-part gate
(single-call 1e-12, two-step 3e-14, chain-end 300x measured honest anchor).

## Best-of-campaign composite (per-cell best gate-clean round)

| L | winner | chain s | round | stock best-lib | vs stock | fftw3_custom_soa* |
|---|---|---|---|---|---|---|
| 10 | gen_race | 0.0734 | r8 | 0.2914 | 3.97x | — |
| 12 | gen_pfa_small | 0.0734 | r8 | 0.2969 | 4.05x | — |
| 15 | gen_race | 0.0840 | r8 | 0.3160 | 3.76x | — |
| 20 | gen_batchlane | 0.1046 | r8 | 0.3671 | 3.51x | — |
| 25 | gen_powp | 0.1265 | r7 | 0.4432 | 3.50x | — |
| 27 | gen_powp | 0.1387 | r8 | 0.4616 | 3.33x | — |
| 31 | gen_rader | 0.1894 | r7 | 1.6005 | 8.45x | 0.4946 |
| 32 | gen_race | 0.1107 | r8 | 0.3429 | 3.10x | — |
| 40 | gen_pfa_large | 0.1635 | r8 | 0.4130 | 2.53x | — |
| 50 | gen_race | 0.2104 | r8 | 0.4833 | 2.30x | — |
| 100 | gen_pfa_large | 0.2864 | r7 | 0.4982 | 1.74x | — |
| **total** | | **1.5609** | | 5.514 | **3.53x** | |

*fftw3_custom columns are SEPARATE by design (Will's rule): genfft-generated custom
codelets are a configuration no shipped FFTW provides. Measured at the ice graded
points 17/23 and gen point 31 (chain s, same node/protocol):

| L (B x m) | fftw3_custom (scalar-split) | fftw3_custom_soa (8-vol batch-lane) | stock MKL same-window | our best |
|---|---|---|---|---|
| 17 (32x98) | 0.1867 | **0.0781** | 0.2786 | 0.0277 |
| 23 (16x165) | 0.6270 | **0.2047** | 0.6922 | 0.0892 |
| 31 (16x140) | 1.2339 | **0.4946** | 1.9034 | 0.1894 |

**The SoA verdict (Will's question, answered by measurement):** retyping the SAME
genfft straight-line DAG over an 8-double vector element (gcc vector extensions,
scalar twiddles broadcast) speeds it up 2.4-3.1x — genfft output DOES enjoy the full
batch-lane split-complex benefit, contrary to the guess. Consequences:
(a) FFTW's own generator output, in the right layout, beats FFTW's shipped library
    3-4x at these primes — the interleaved-API framing, not codelet quality, is the
    library's binding constraint;
(b) hand-tuning still matters: our panel kernels beat SoA-custom by another 2.3-2.8x
    (schedule, fusion, and memory choreography that genfft does not model);
(c) a 20-line generator retype delivers most of the layout win — the cheapest
    "first performant implementation" from the whole literature sweep.

## Round-8 note
r8 consolidated (+0.5-2.6% at seven cells) but REGRESSED L=25 by 28% (0.1310->0.1681,
verified on a quiet node — an honest code regression in gen_powp's r8 edit, not a gate
catch and not wisdom-cache poisoning). r6/r7's kernel is preserved in impl_6/impl_7;
the composite above uses it. Lesson for the trunk: per-cell winners should be pinned
by measured provenance, not only by latest-round code.

## Validation record
- Surprise sizes (never briefed): L=21 1.99x, L=44 1.29x, L=96 1.56x over best library,
  cold plan <=0.6 s. The library generalizes.
- Cascade Lake advisory: all 11 cells win (1.66-6.15x); race picks different winners
  per host as designed.
- SPR spot-check: recorded in results/xarch_spr_r5.
