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

## Round d1_r2 (2026-09-02)

No Ice Lake reservation again (job 440371 gone, tryout.sh refuses), so all numbers are
wallaby (SPR 6448Y), pinned nice'd core, min-over-runs, A/B against binaries built in the
same session. r1's scoring-node lesson drove the round: my AoS code degraded 1.54x from
wallaby to a80n0 while MKL stayed put and batchlane's shuffle-free kernels degraded only
1.32x — on Ice Lake every 512-bit shuffle lands on port 5, which is also one of the two
FMA ports. So the round's theme was removing shuffles from every hot loop.

### Change 1 — across-batch SoA chain path (TAKEN FROM d1_batchlane, r1 record)

For fft1d_chain with batch >= 8 and L <= 2048: groups of 8 transforms, zmm lane = batch
index, split-complex planes, one 8x8 double transpose in/out per group per CHAIN (not per
step), broadcast scalar twiddles, pure radix-4 ladder, map fused in split form (cheaper
than AoS: |z|^2 needs no pair-swap dup). This is exactly batchlane's design; they beat me
with it at 32/64 batched chained in r1 (0.0607 vs my 0.0852 on a80n0).
  - 32 B=512 m=1000: 0.064 -> 0.032 us   (fixes the one cell where a LIBRARY beat me:
    fftw1d_custom_soa 0.0665 on a80n0)
  - 64 B=512 m=500: 0.116 -> 0.062       128 B=512 m=250: 0.233 -> 0.146
  - GATED at L <= 2048: group working set is 3 x 16L doubles; at 16384 it MEASURED 2x
    SLOWER than the per-transform path (98.0 vs 48.8 us) — streams L3 every step while
    per-transform blocking stays L2-resident. At 4096 it fit wallaby's 2 MB L2 (4% win)
    but not the scoring node's 1.25 MB; gated out.

### Change 2 — blocked split-complex radix-8 engine for L >= 128 (execute + chain)

Replaced the interleaved-AoS pipeline above the codelets. Three design points, each
measured the hard way:
  1. FORMAT: blocks of 8 complexes as [8 re | 8 im], NOT two big planes. Plane format
     lost 42% at 4096 B=1 (9.6 vs 6.7 us): each radix-8 stage runs 16 read + 16 write
     streams whose strides are 4K multiples, exhausting L1 sets/fill buffers (a 64B
     im-plane pad did NOT fix it — it is stream count, not set conflicts). Blocks keep
     the AoS engine's 8+8 streams; the split kernels are unchanged (im = re + 8 doubles).
  2. FREE CONVERSION: the stride-1 first stage writes blocked-split directly — its store
     transpose is 8 permutes per 16 complexes in EITHER output format, so AoS->split
     costs zero. (The first attempt, a separate AoS->split stage at s=4 with deint loads
     + ymm-half stores, was the bottleneck: 36% total loss at 1024, shrinking as more
     clean stages diluted it.) One paired-p radix-8/4 stage covers s == 4 with vector
     twiddles and pair-merged full-zmm stores; every later stage is pure vertical FMA,
     ZERO shuffles, broadcast twiddles; the twiddle-free final re-interleaves on store.
  3. BUFFERS: keep using the caller's out buffer as one ping-pong intermediate. A second
     scratch instead MEASURED +20% at 1024 B=1 and +10% at 16384 (L1 working set 48 ->
     64 KB at 1024). This one is subtle and worth remembering: buffer COUNT, not just
     traffic, is a first-order effect at L1-boundary sizes.
  Schedules (first stage radix-4 s1s, then paired s=4 stage radix 8 or 4 so the rest
  factors as 8^a * (8|4), then SS8s, then SF8/SF4): 128 -> 3 passes, 1024 -> 4,
  4096/16384 -> 5. Same pass counts as the AoS engine.

Result on wallaby: parity to +3% on non-chained cells (even at 512-16384, -3% at 128 B=1,
~+5% at the DRAM-bound 1024 B=512 — accepted), and the split-form fused map turned into
real chained wins:
  - 128 B=1 m=30000: 0.231 -> 0.173      1024 B=1 m=4000: 2.169 -> 1.716
  - 4096 B=1 m=1000: 10.82 -> 9.16 (chain 256/400: 10.89 -> 9.23)
  - 16384 B=1 m=250: 49.7 -> 41.7 (64/150: 48.8 -> 42.6), 16384 B=1 m=1: 29.3 -> 27.0
The wallaby-parity bet is that the scoring node pays ~2/3 fewer port-5 shuffle uops in
every middle stage; r1's cross-machine degradation numbers say that is where my 1.3-1.7x
non-chained losses at 128-16384 came from.

### Tooling note (record so nobody re-trusts it blindly)

llvm-mca (/opt/software/llvm-22.1.8, -mcpu=icelake-server) models ONE 512-bit FMA unit:
it piles all 512-bit FP on port 0 (aos radix-8 loop: p0=42, p5=21 per iter) and rates my
AoS and split radix-8 stages IDENTICAL per complex. The scoring node's Gold 6326 has TWO
FMA units, so rebalancing by hand gives split ~0.88 vs AoS ~0.98 cycles/complex. Use mca
for uop counts and dependency chains, not for p0/p5 balance on this machine.

### What did NOT work, with the number that killed it

- Split-complex as two L-sized planes: 9.6 vs 6.7 us at 4096 B=1 (stream-count blowup,
  see above). 64B im-plane pad: no change (9.56).
- Dedicated AoS->split conversion stage at s=4 (deint loads, per-lane vector twiddles,
  ymm-half stores): 1.24 vs 1.01 us at 1024 B=1 even after the block format fixed the
  streams. Conversion must be fused where a transpose already exists.
- Two private scratch buffers in run_stages: 1.19 vs 1.01 at 1024 B=1, 30.6-32.3 vs 28.1
  at 16384 (this one masqueraded as "split kernels are slow" for an hour — the honest
  A/B was rebuilding the AoS engine against the SAME buffer scheme).
- SoA chain at 4096/16384 (see gate above).

### Correctness

All 24 graded pow2 cells PASS (rel_l2 <= 3.4e-16 single-call; chain gates worst
4.5e-12 at 1024:1:4000 vs 1e-10 floor). Remainder paths (batch % 8 through SoA+tail,
B=3/5/9/11/12), L=16 legacy path, and two-process repeatability all verified. The
non-graded unlucky config 128 B=8 m=30000 reads 1.19e-10 vs r1's 1.56e-10 — same
known-hot config, slightly better, still above the floor, still not a graded cell.

### Best wallaby numbers this round (min us/transform)

| L     | B=1 m=1 | batched m=1 | B=1 chained | batched chained |
|-------|---------|-------------|-------------|-----------------|
| 32    | 0.012   | 0.012 (512) | 0.064       | 0.032 (512)     |
| 64    | 0.029   | 0.028 (512) | 0.116       | 0.062 (512)     |
| 128   | 0.073   | 0.089 (512) | 0.173       | 0.146 (512)     |
| 1024  | 1.014   | 1.49 (512)  | 1.716       | 2.093 (512)     |
| 4096  | 6.68    | 8.91 (256)  | 9.16        | 9.23 (256)      |
| 16384 | 27.0    | 37.1 (64)   | 41.7        | 42.6 (64)       |

### Borrowed

- d1_batchlane (r1 record): the entire SoA-groups-of-8 fused-chain design, including the
  L1-residency argument and the "transpose once per chain" accounting. Named plainly:
  change 1 is their idea, re-implemented and extended with my refined rsqrt/rcp map and
  the L <= 2048 gate.
- gen/tools/TOOLS.md llvm-mca recipe (and its "trust relative, not absolute" warning,
  which this round sharpened: not even relative is safe when the unit count is wrong).

### Next round

- The 4096/16384 B=1 gap to MKL on the scoring node (1.57-1.70x in r1) is bigger than
  shuffle relief alone can close if wallaby parity translates to ~1.3x on a80n0. The
  remaining lever is four-step/six-step 64x64 and 128x128 with L1-resident sub-FFTs and
  fused twiddle multiply — now much cheaper to build because the split kernels already
  exist. Do that FIRST next round, measured per stage from the start.
- 1024 B=512 m=1 (+5% this round): try the AoS schedule for large-batch execute only, or
  software prefetch of the next transform's input inside stage_s1s.
- If the monitor's leaderboard shows the split engine did NOT close the ICX gap: get a
  reservation and PMU the port-5 dispatch directly (tools/pmu.sh, /tmp/perf) before
  touching anything else. The whole round is a bet on that counter.

## Round d1_r3 (2026-09-03)

The Ice Lake reservation was dead again (job 440424 not running, tryout.sh refuses), so
all numbers are wallaby (SPR 6448Y), pinned to a core whose HYPERTHREAD SIBLING was also
verified idle (a plain idle-core check let a busy sibling inflate spreads to 45% early in
the session), nice'd, min over interleaved reps. The r2 leaderboard set the targets: my
worst ratios were the large non-chained cells (4096 B=1 1.52x, 16384 B=1 1.43x,
16384 B=64 1.48x, 4096 B=256 1.30x, 1024 B=1 1.27x) and the small chained B=1 cells
(32 at 1.23x behind d1_batchlane).

### Change 1 — fused middle stage-pair at 4096 (ST_SS64: radix-64 through an L1 tile)

Two consecutive split radix-8 stages = two full read+write passes of the array. Fused:
for each second-level group p2, run the first level for its 8 feeder groups
p = p2 + j*(n/64) into a 64-block tile (64*s complexes, 16 KB at 4096), then the second
level straight from the tile. Same butterflies, same twiddles, same operation order —
output bitwise identical (verified by cmp against the r2 binary); the array is read and
written once instead of twice. 4096 B=1: 6.67 -> 6.29 us (-5.7%).

**But the same fusion LOST 5% at 16384** (27.0 -> 28.3, quiet-core min-of-6): the fused
traversal reads 64 interleaved 512 B bursts 4 KB apart — too short for the hardware
prefetcher to lock on — where the unfused stages each ran 8 long sequential streams.
Explicit T0 prefetch of the next feeder group did not recover it (28.7). This is the
r2 plane-format lesson again, in read form: STREAM COUNT AND STREAM LENGTH, not just
total traffic. Gated to L == 4096 (where the whole array is only 128 KB so the burst
pattern still lives in L2 comfortably).

### Change 2 — NT final stores above the scoring node's L3, and the r1 NT verdict was confounded

r1 recorded "NT stores 3x slower" and turned them off forever. Re-examined: with an odd
stage count, run_stages ping-pongs intermediates through the CALLER'S out buffer, so the
final NT streams targeted lines the same call had just dirtied — measured 1.5x slower
than no NT at all (16384 B=256, 128 MB working set: 133 vs 85 us). Routing intermediates
through two private scratches when NT is on (dst untouched until the final stream) flips
it: 71.9 -> 50.7 us at the same cell (-30%). The r2 "second scratch costs 20%" lesson
does not apply here: that was L1-resident B=1; these cells stream DRAM.

Policy: NT when in+out = 32*L*B >= 25 MB, i.e. above the scoring node's 24 MB L3 —
catches the graded 4096xB=256 and 16384xB=64 (33.5 MB each), leaves 1024xB=512 (16.8 MB,
L3-resident there) alone; measured NT at 1024x512 anyway: 1.54 vs 1.51, correctly
excluded. Even on wallaby's 60 MB L3 the two graded cells improved (16384 B=64:
36.9 -> 35.4; with fusion 4096 B=256: 9.4 -> 8.5), because NT kills the RFO reads
regardless and the shared login-node L3 never really holds 33 MB for you.

### Change 3 — fast map by default (drop the exact-residual refinements)

r1's own numbers, re-read: the 2NR-only map scored 6.7e-10 at the unlucky config and
+refinements only moved it to 6.1e-10 — the real fix was long-double twiddles (1.56e-10).
With exact twiddles the refinements are noise: dropping them (Heron on sqrt + final
quotient residual; the rcp's two NRs stay) moved the graded gates by <25% — worst cell
1024:1:4000 went 4.54e-12 -> 5.68e-12 vs the 1e-10 floor (17.6x margin), 128:1:30000
went 7.277e-13 -> 7.286e-13, m=2 short-chain gates all ~1e-15 vs 3e-14 — and bought
10-27% on EVERY chained cell. d1_batchlane's r2 record said exactly this ("drop the
map's residual corrections last; the 2-NR-only variant is what composite ships") — this
is their/d1_composite's call, adopted after verifying the margins. Escape hatch:
-DD1_FASTMAP_MAX_L=0 restores the precise map everywhere.

### Best wallaby numbers this round (min us/transform; r2 -> r3 where changed)

| L     | B=1 m=1        | batched m=1     | B=1 chained     | batched chained  |
|-------|----------------|-----------------|-----------------|------------------|
| 32    | 0.012          | 0.012 (512)     | 0.064 -> 0.053  | 0.032 -> 0.026   |
| 64    | 0.029          | 0.028 (512)     | 0.116 -> 0.095  | 0.062 -> 0.052   |
| 128   | 0.074          | 0.090 (512)     | 0.173 -> 0.153  | 0.146 -> 0.121   |
| 1024  | 1.021          | 1.51 (512)      | 1.716 -> 1.485  | 2.093 -> 1.945   |
| 4096  | 6.68 -> 6.31   | 8.91 -> 8.10 (256) | 9.16 -> 7.67 | 9.23 -> 7.75 (256) |
| 16384 | 27.25          | 37.1 -> 34.8 (64)  | 41.7 -> 36.9 | 42.6 -> 37.9 (64)  |

### Correctness

All 24 graded pow2 cells PASS at graded (L,B,m): single-call rel_l2 <= 3.4e-16 at every
size; worst chain gate 5.68e-12 (1024:1:4000) vs 1e-10. Remainder/tail paths (B=3,5,9,
11,12), non-graded sizes 16/256/512/2048/8192/32768/65536, tail-path chains, m=2 gates,
and two-run bitwise repeatability all verified on the final binary. Fusion output
verified bitwise identical to r2 code at 4096 and 16384 (before gating 16384 off).

### What did NOT work, with the number that killed it

- SS64 fusion at 16384: 27.0 -> 28.3 us (+5%); +T0 prefetch of the next feeder group:
  28.7. Read-burst length beats pass count when the array streams from L2. Gated off.
- NT stores with the final stage streaming into the ping-pong dst: 133 vs 85 us at
  16384 B=256 — worse than no NT. NT requires an output buffer with no dirty lines;
  r1's blanket "NT is 3x slower" was THIS bug plus a fits-in-L3 test case.
- Software prefetch inside ss8_group generally: never helped (also tried during the
  fusion experiments), the linear-stream stages don't need it.

### Borrowed, explicitly

- d1_batchlane r2 / d1_composite: the 2NR-only fast map as the shipping default
  (change 3), adopted with my own gate-margin measurements.
- No structural code taken this round; changes 1-2 are traffic restructurings of my own
  r2 engine, prompted by re-reading r1's confounded NT experiment.

### Next round

1. 16384 B=1 (and 1024 B=1) remain the non-chained deficits (1.43x/1.27x on a80n0 in
   r2). Wallaby now says 16384 is NOT pass-count-bound (the fusion loss), so the
   remaining candidates are (a) true Bailey four-step 128x128 with L1-resident sub-FFTs,
   fused W^{n1k2} twiddles and an 8x8 blocked output transpose — more compute, much less
   L2/L3 traffic; only worth shipping with ICX PMU evidence, it may well LOSE on wallaby
   — or (b) an ICX-side port-5 diagnosis of the split stages. GET A RESERVATION FIRST;
   two rounds of blind cross-machine betting is enough.
2. 1024 B=1 needs a 3-pass schedule to move: radix-16 middle stages spill registers in
   split form (32 live zmm for data alone); the workable variant is a radix-16 stage as
   two in-register radix-4 levels over a 16-vector strip — unattempted.
3. If the monitor's seeds ever change and a chain gate fails: -DD1_FASTMAP_MAX_L=0.
4. 32/64/128 B=1 m=1 sit 1.11-1.30x behind MKL on ICX but at/ahead on wallaby — that gap
   is machine-specific shuffle pressure; the L=128 two-block in-register codelet (32 zmm)
   is the only untried structural idea there.
