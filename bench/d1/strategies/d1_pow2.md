# d1_pow2 — strategy record

## Round d1_r1 (2026-09-02, fresh restart)

Starting point was the dense O(L^2) stub (nothing survived the lost rounds). The Ice Lake
dev reservation was dead when this round started and I was instructed not to submit slurm
jobs, so all development and timing was done on wallaby (Xeon Gold 6448Y, Sapphire Rapids,
AVX-512), pinned + nice'd on one core, using the same gcc flags as tryout.sh. Wallaby was
intermittently loaded by other agents' jobs (load 20–31); numbers below are min-over-runs
from quiet windows and carry maybe ±15% machine skew vs the a80n0 scoring node. Supported
sizes were extended from {16..256} to all pow2 16..65536 — the stub was silently forfeiting
the graded 1024/4096/16384 cells.

### What the implementation is now

1. **Stockham autosort DIF** (no bit reversal, ping-pong out-of-place, per-transform loop
   so each transform's working set stays cache-resident). Interleaved complex, one zmm =
   4 complexes. Complex multiply = 1 vpermilpd + mul + fma using twiddles stored as
   (broadcastable re) + (-im,+im) pair — no sign-mask xors (survey's split-format trick).
2. **Stage schedule** (per-size, measured): first stage is radix-4 at s=1 (vectorized
   across the twiddle index with per-lane tables + a 4x4 complex-lane in-register
   transpose); then, when log2(L/4) % 3 != 0, one radix-4 stage at s=4 (a radix-4 stage is
   owed anyway; spending it at s=4 keeps every radix-8 stage at s>=16 where its 14 twiddle
   broadcasts amortize over >=4 vector iterations); then radix-8 stages; then a
   twiddle-free radix-8/4 final codelet. Pass counts: 32→2, 64/128/256→3, 1024→4,
   4096/16384→5. A/B (randomized order, min-of-mins over 8 reps): mixed schedule beats
   pure radix-4 at 4096 (6.77 vs 8.09 us) and beats greedy radix-8 at 1024 (0.896 vs 1.334).
3. **In-register codelets for L=32 and L=64**: whole transform in 8/16 zmm, no
   intermediate stores, full precomputed w/w²/w³ first-stage tables. The on-the-fly
   w²=w1², w³=w1w2 variant was SLOWER (L=32 B=1: 0.027 vs 0.018 us — serial FMA chain) and
   is also a rounding-bias source (below); full tables won on both counts.
4. **fft1d_chain is owned** — the two structural wins of the round:
   - the chain is separable per transform (batched FFT is independent per transform, map
     is pointwise), so each transform runs ALL m steps back-to-back while resident
     (~4L·16 B working set). Libraries through the driver fallback stream the whole
     B·L batch three times per step. This is what flips the batched chained cells.
   - the map z/(1+|z|) is fused into the final butterfly stage (no separate map pass) and
     computed with rsqrt14/rcp14 + Newton, finishing each quantity (sqrt, reciprocal, and
     the quotient itself) with an exact-residual FMA refinement. For L=32/64 the entire
     m-step chain state lives in registers (plus 8 c-field vectors for L=32).

### The accuracy fight (do not rediscover this)

At pow2 sizes the chain gate's two numpy reference paths agree bitwise → anchor = 0 →
tolerance floors at 1e-10 FIXED, and per-step deviation from numpy accumulates through a
weakly chaotic chain (amplification is seed- and L-dependent; L=128 with the standard
seeds is the nastiest of the graded cells). Three successive fixes, each measured at
L=128 m=30000 B=8 (a deliberately unlucky non-graded config):
   - 2-Newton rsqrt/rcp map (~2-3 ulp): 6.7e-10 — FAILS the 1e-10 floor.
   - + exact-residual refinements in the map (~0.5-1 ulp each op): 6.1e-10 — map was NOT
     the dominant error; the FFT itself was.
   - + long-double twiddle generation (M_PI's rounding is a BIASED ~2e-16 phase error —
     cosl/sinl with 80-bit pi gives correctly-rounded-double tables; this is the survey's
     "twiddle tables from correctly-rounded sincos" point, empirically confirmed) and
     full precomputed first-stage w²/w³ tables (squaring in-loop = correlated bias):
     1.56e-10 at the unlucky config, and the ACTUAL graded cells all pass with >=100x
     margin. Single-call rel_l2 also dropped ~30% (L=16384: 4.6e-16 → 3.4e-16).
All 12 graded chained cells verified at graded (L,B,m): worst is 1024:1:4000 at 5.0e-12;
one-step m=2 gates all ~1e-15 (tol 3e-14); single-call rel_l2 ≤ 3.6e-16 at every
supported size 16..65536; output bitwise repeatable across runs.

### What did not work, with the number that killed it

- **Non-temporal final-stage stores** for the big batched m=1 cells: 3x SLOWER
  (1024 B=512: 5.95 vs 2.1 us). Every graded batched cell's in+out (<=32 MB) fits L3
  (60 MB wallaby / 24 MB a80n0), so regular stores hit L3 while NT forces DRAM. Code kept
  behind plan->nt = 0.
- **On-the-fly w², w³ in stride-1 stages**: slower AND biased (numbers above).
- **Greedy radix-8 at s=4** (1024: 1.334 vs 0.896 us) — twiddle-broadcast overhead at one
  q-iteration per p.
- Driver edge case found: `--chain 1 --map` segfaults for ANY backend (driver never
  allocates `pong` at chain=1 but passes it); no graded cell hits it.

### Best wallaby numbers (min us/transform; "lib" = best library on a80n0 from
results/d1_libbase + BASELINE.md, so cross-machine — the monitor arbitrates)

| L     | B=1 m=1 | lib   | batched m=1 | lib   | B=1 chained | lib   | batched chained | lib   |
|-------|---------|-------|-------------|-------|-------------|-------|-----------------|-------|
| 32    | 0.012   | 0.025 | 0.014 (512) | 0.015 | 0.057       | 0.131 | 0.052 (512)     | 0.112 |
| 64    | 0.029   | 0.045 | 0.028 (512) | 0.038 | 0.088       | 0.237 | 0.089 (512)     | 0.238 |
| 128   | 0.078   | 0.091 | 0.106 (512) | 0.148 | 0.208       | 0.485 | 0.190 (512)     | 0.535 |
| 1024  | 0.896   | 1.084 | 1.512 (512) | 1.684 | 1.933       | 4.180 | 1.924 (512)     | 4.915 |
| 4096  | 6.63    | 6.00  | 10.33 (256) | 11.12 | 9.08        | 19.09 | 9.80 (256)      | 22.74 |
| 16384 | 27.5    | 32.11 | 43.8 (64)   | 45.66 | 43.3        | 82.19 | 42.2 (64)       | 99.24 |

Chained cells win ~2x everywhere. Non-chained: everything at or ahead of the libraries on
wallaby numbers except 4096 B=1 (~0.9x of FFTW patient) and thin margins on the
memory-bound batched large cells.

### Borrowed / sources

Everything here is from docs/literature_1d/00-SURVEY.md (Stockham + conjugate-pair advice,
correctly-rounded plan-time twiddles, FFTS fixed-geometry specialize-then-run model — the
L=32/64 codelets are that idea); no other implementer had produced anything this round
(context.md was empty — post-restart round 1).

### Next round

- 4096/16384 B=1: the one remaining library edge. Try radix-16 stages (4 passes at 16384)
  or Bailey four-step with an explicit L1-blocked transpose; also llvm-mca/PMU the
  radix-8 stage on the scoring node — port-5 shuffle pressure (4x vpermilpd + transpose
  ops) is the suspect ceiling.
- L=128 codelet (32 zmm; needs a 2-block structure) for the same treatment as 32/64.
- Batched large-L m=1 cells are pure L3/DRAM bandwidth; consider interleaving two
  transforms to overlap load/store streams, or software prefetch of the next transform.
- If a chained cell ever fails on the scoring node's seeds: the fallback is exact
  vsqrtpd/vdivpd in map_vec (matches driver-map quality; costs ~30% of chained-cell time
  at small L).
