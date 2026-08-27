# GENERALIZE campaign — final accounting (8 rounds, closed 2026-08-25)

All numbers: chain seconds on bare-metal Ice Lake a80n0, pinned, two-part gate
(single-call 1e-12, two-step 3e-14, chain-end 300x measured honest anchor).

## Best-of-campaign composite (per-cell best gate-clean round)

| L | winner | chain s | round | stock best-lib | vs stock | fftw3_custom_soa* | soa vs stock |
|---|---|---|---|---|---|---|---|
| 10 | gen_race | 0.0734 | r8 | 0.2914 | 3.97x | 0.2884 | 1.01x |
| 12 | gen_pfa_small | 0.0734 | r8 | 0.2969 | 4.05x | 0.3287 | 0.90x |
| 15 | gen_race | 0.0840 | r8 | 0.3160 | 3.76x | 0.3208 | 0.98x |
| 20 | gen_batchlane | 0.1046 | r8 | 0.3671 | 3.51x | 0.3666 | 1.00x |
| 25 | gen_powp | 0.1265 | r7 | 0.4432 | 3.50x | 0.3289 | 1.35x |
| 27 | gen_powp | 0.1387 | r8 | 0.4616 | 3.33x | 0.3312 | 1.39x |
| 31 | gen_rader | 0.1894 | r7 | 1.6005 | 8.45x | 0.4837 | 3.31x |
| 32 | gen_race | 0.1107 | r8 | 0.3429 | 3.10x | 0.3730 | 0.92x |
| 40 | gen_pfa_large | 0.1635 | r8 | 0.4130 | 2.53x | 0.4422 | 0.93x |
| 50 | gen_race | 0.2104 | r8 | 0.4833 | 2.30x | n/a (B=4) | — |
| 100 | gen_pfa_large | 0.2864 | r7 | 0.4982 | 1.74x | n/a (B=1) | — |
| **total** | | **1.5609** | | 5.514 | **3.53x** | | |

### Notes on the fftw3_custom_soa column

*What it is.* FFTW's own code generator (genfft, driven by an OCaml toolchain we built for
the purpose) asked for single monolithic straight-line codelets at each exact size — a
configuration NO shipped FFTW provides — then the identical generated C compiled with the
element type retyped to an 8-double vector, turning it into an SoA batch-lane split-complex
kernel (8 volumes per lane-slot, scalar twiddle constants broadcast by the compiler). It is
kept as a SEPARATE column by project rule: it answers "how good could library-generated code
be", not "what a library user gets".

*The layout effect is real and large*: the SoA retype beats the same codelet in scalar-split
form by 1.3-2.4x (scalar column in results/fftw_custom/), confirming that genfft output DOES
enjoy the batch-lane benefit — a ~20-line compile-time change worth up to 2.4x.

*But the algorithm still decides where it matters.* Against the best stock library the SoA
custom wins ONLY at the prime sizes (25: 1.35x, 27: 1.39x, 31: 3.31x; and at the ice points
17/23 it was 3.5x+) — exactly where libraries fall back to Rader/Bluestein. At CT-friendly
sizes (10, 12, 20, 32, 40) a monolithic direct codelet has a worse op count than the
libraries' staged Cooley-Tukey, and the layout gain only brings it back to parity or a
slight loss (0.90-1.01x). Layout fixes the arithmetic-throughput problem; it cannot fix an
op-count problem.

*The B>=8 requirement is itself a finding*: batch-lane SoA needs 8 volumes to fill a zmm,
so the B=4 and B=1 cells (50, 100) cannot use it at all — the layout's advantage is
conditional on batch shape, which is why the library's plan-time race (which picks
per-(L,B) layouts) is the right architecture rather than a fixed layout choice.

*Hand-tuning still wins everywhere*: our panel kernels beat the SoA custom by 2.4-4.5x
across the board — schedule, axis fusion, and memory choreography that a straight-line
codelet in a generic 3-pass driver does not capture.

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

## Extension rounds 9-10 (counter-directed, closed 2026-08-26 on a81n2)

Avenue 1 (bank the picks) DELIVERED: L=25 restored to 0.1264-0.1272 across both rounds
(r8's instability gone); the noise-gate behaved exactly as designed per the strategy
records ("gate correctly reverted + refused storage" under a hot window, stored in a
quiet one); gen_race now wins or co-wins ~half the cells. Avenues 2-4 did not move the
board: totals r9 1.5619, r10 1.5596 (vs 1.5609 best-of-campaign) — converged. On avenue
2 there is an open, documented disagreement: the PMU audit reads L=100 at 0.82/cycle
p0+p5 dispatch with ~100M L1 line-fills per chain (headroom), while gen_pfa_large's own
r7 accounting declares the engine uop-saturated on this host and closed the avenue. A
same-node counter session on the r10 binary reproduces the traffic number (unchanged per
chain), so the fusion question remains genuinely unresolved — the right shape for it is
a dedicated effort with PMU from day one, not an overnight round.

Final standing: 11/11 cells, total 1.5596 s = 3.54x best stock library; arbitrary-L
trunk validated on unseen sizes, two Ice Lake nodes, and Cascade Lake.

## Rounds 11-12: the all-hands counter-directed rounds (closed 2026-08-27)

Design: every implementer on the large-size bottleneck, mandatory PMU protocol
(baseline counters -> change -> counters), six-approach menu, race-arbitrated
cross-class entries.

**L=100 broke open in r11: 0.2899 -> 0.2606 (+10.1%), confirmed in r12 (0.2614).**
The winner was a CROSS-CLASS challenger: gen_batchlane (small-L specialist) built a
within-volume engine that software-pipelines the x-column loads through the memory-stall
band it had itself measured (stalls_mem_any = 27% of cycles at 1.4/2.1 dispatch). The
incumbent's "closed from three directions" was phase-true for its own engine and
cell-false in general — the round's central lesson: counters settle arguments between
measurements of DIFFERENT engines only via the race. r12 spread the technique (powp
adopted the pipelining; race armed multi-way arms at 100) and confirmed convergence.

FINAL best-of-campaign composite (per-cell best gate-clean round, r1-r12):
10: 0.0714 | 12: 0.0734 | 15: 0.0830 | 20: 0.1028 | 25: 0.1264 | 27: 0.1371 |
31: 0.1893 | 32: 0.1076 | 40: 0.1631 | 50: 0.2104 | 100: 0.2606
**TOTAL 1.5251 s = 3.62x the best stock library**, range 8.45x (L=31) to 1.91x (L=100).

Standing open item for any successor effort: two-axes-per-pass fusion at 40/50/100 —
named by multiple records as "the one remaining structural lever", never attempted; it
should COMPOUND with the r11 pipelining, not replace it.
