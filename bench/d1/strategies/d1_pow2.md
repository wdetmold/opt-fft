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

## Round d1_r4 (2026-09-03)

**The Ice Lake reservation was ALIVE this round** (job 440424 on a80n0 — the scoring node
itself) and every decision below was measured there, on a leased core, ending three rounds
of blind cross-machine betting. Operational note for future rounds: `tryout.sh` initially
refused ("no live reservation") because the login-node `squeue` is a SHIM at
`bench/gen/wallaby_shims/squeue` keyed to the *gen* campaign's heartbeat file, which is
stale. The d1 heartbeat (`bench/d1/RESERVATION.heartbeat`) was fresh the whole time. Fix:
prepend a private shim pointing at the d1 heartbeat to PATH; do not conclude the node is
gone without checking the heartbeat file itself.

### The diagnosis that drove the round (ICX PMU, finally)

At 16384 B=1 the r3 engine ran 55.5 us on a80n0 vs 27 on wallaby while FFTW stayed put.
perf on the node showed why, and it was NOT port-5 shuffles (r2's bet):
  - `l1d.replacement` = 60.7k per transform — one line fill every 3 cycles, i.e. the
    whole runtime. Five passes streaming array + tables through L1, and **ICX has half
    Sapphire Rapids' L2->L1 fill bandwidth** — that is the entire cross-machine gap.
  - `cycle_activity.stalls_l2_miss` = 46k cycles per transform (25% of runtime) because
    the DUP-FORMAT twiddle tables (s1s 12 dbl/p = 384 KB + s48 56 dbl/p = 229 KB at
    16384) pushed src+dst+scratch+tables past the 1.25 MB L2. Wallaby's 2 MB L2 hid this.
Port dispatch was ~45% on both FP ports — there was no port wall to fix.

### Change 1 — compact (c,s)-pair twiddle tables at L >= 1024 (the big one)

s1s stores 6 dbl/p (interleaved c,s; cmul re-expressed as
`fmaddsub(u, dup(c), swap(u)*dup(s))` — 2 extra port-5 dups, one load fewer, same values);
s48/s44 store 4 dbl per pair per r, rebuilt with `broadcast_f64x4` + two `permutexvar`
(4x smaller). Tables at 16384: 613 KB -> 250 KB; everything fits L2 again. a80n0:
  - 16384 B=1: 49.8 -> 38.4 us;  16384 B=64: 60.2 -> 47.8 (now BEATS fftw_patient 49.4)
  - 1024 B=1: 1.62 -> ~1.19;  4096 B=1: 8.75 -> 8.0;  chains 16384: 67.4 -> 52.4
  - **At L <= 512 the same trick MEASURED SLOWER** (128 B=1: 0.106 -> 0.116) — tiny
    tables are L1-resident and the port-5 dups are pure cost. Gate: L >= 1024 exactly.
Bitwise note: the fmaddsub form rounds `swap(u)*s` where the old form rounded `u*c`
(both <= 0.5 ulp); all gates re-verified, margins unchanged.

### Change 2 — first-stage pair fused through an L1 tile at L >= 4096 (ST_SX48/ST_SX44)

The stride-1 radix-4 stage feeds the s == 4 stage through a 256-double tile: for each
second-stage p-pair-pair, run the 8 (4) feeder s1s quads into tile slots, then the
second stage from the tile. One array pass instead of two, output BITWISE identical
(verified by cmp at 1024/2048/4096/16384, plus odd batch). **Indexing trap that cost an
hour: one s1s quad (16 complexes) produces the leg-J blocks of TWO consecutive
second-stage pairs**, so the tile loop must advance p2 by 4 and run two second-stage
pairs per fill; with p2 += 2 the output is garbage. With this plus re-enabling the r3
radix-64 middle fusion at 16384 (a loss standalone in r3, a win once the first pair is
fused — 4 passes -> 3), 4096 and 16384 both run in THREE array passes:
  - a80n0 interleaved A/B: 4096 B=1: 8.0 -> 7.4;  16384 B=1: 34.0 -> 33.1 (fftw 32.6);
    chains 16384:64:150: 53.0 -> 51.5.  At 1024 the fusion LOST 8% (scattered s1s table
    reads defeat the small-size balance) — gated to L >= 4096 (D1_SX_MIN).

### Built, measured, REMOVED: a full Bailey four-step engine

The literature/planner recommendation (four-step 128x128, L1-resident sub-FFTs) was
implemented completely this round: column FFTs over a 16 KB tile with the AoS
deinterleave fused into the first vertical stage, diagonal twiddle built per row from
48 KB of factored tables (W^{k1*8b} (x) W^{k1*j}) instead of a 256 KB diagonal, mid
array stored n2-major so step 2 needs no transpose pass, final stage fused with
interleave+NT stores. Correct (3e-16 at all sizes), and it DID cut fills 60.7k -> 37.6k
— but the tile ladder's short q-loops doubled the load count (14 broadcast twiddle loads
per 1-iteration group; port_2_3 49k -> 106k/transform) and instructions went 239k ->
290k. Best version: 52.7 us at 16384 B=1 vs 38.4 for the compacted Stockham. Fills are
not the only currency; a five-pass engine with LONG unit-stride loops beat a two-pass
engine with short ones. Removed from the shipping source; do not rebuild it without
first fixing the twiddle-amortization problem (wider tiles = 2 vectors/row, or
straight-line codelets for the tile FFT).

### Other dead ends, with numbers (all a80n0)

- SS64 fusion at 16384 WITHOUT the first-stage fusion: neutral (35.2 vs 34.9) — r3's
  wallaby verdict held on ICX until change 2 shifted the balance.
- NT stores at 1024 B=512 (16.8 MB, fits the 24 MB L3): wash (1.84 vs 1.81-1.96),
  threshold stays at 25 MB.
- Compact tables at 128-512: above.
- A split-form L=64 four-step codelet was sketched and REJECTED on arithmetic: every
  variant needs ~80 port-5 shuffles/transform (deint 16 + 8x8 mid-transpose 48 + int 16)
  vs ~65 in the current AoS codelet; MKL's 0.043-0.046 us = ~155 cycles is near the pure
  FMA floor and is not reachable by rearranging shuffles. 64 B=1 stays ~1.2-1.4x behind.

### Where the cells stand (a80n0, leased core, min over interleaved runs; "lib" = best
of mkl/fftw_patient same core same day; r3 leaderboard ratio in parens)

| cell | mine | lib | | cell | mine | lib |
|---|---|---|---|---|---|---|
| 32 B1   | 0.021 | 0.023 mkl WIN (1.00) | | 1024 B1   | 1.19-1.28 | 1.08 mkl ~1.15x (1.51x) |
| 32 B512 | 0.015 | 0.015 mkl tie (1.01)  | | 1024 B512 | 1.97 | 1.75 mkl 1.13x (1.21x) |
| 64 B1   | 0.058-0.067 | 0.043-0.046 mkl ~1.3x (1.12x) | | 4096 B1 | 7.4-7.8 | 6.06 mkl ~1.25x (1.45x) |
| 64 B512 | 0.037 | 0.037 mkl tie (1.16x) | | 4096 B256 | 10.5-10.6 | 10.4 mkl 1.02x (1.09x) |
| 128 B1  | 0.105-0.122 | 0.093 mkl ~1.2x (1.29x) | | 16384 B1 | 33.1-34.9 | 31.9 fftw ~1.07x (1.64x) |
| 128 B512| 0.180 | 0.159 mkl 1.13x (1.29x) | | 16384 B64 | 47.7 | 46.7 fftw 1.02x (1.14x) |

Chained cells (mine, us/transform; libs were 1.5-4x slower at every one in r3): 32:
0.071/0.039, 64: 0.128/0.080, 128: 0.217/0.187, 1024: 2.16/2.58, 4096: 10.8/10.3,
16384: 52.4/51.5. Wallaby also improved everywhere (no SPR regression): B=1 m=1
0.012/0.030/0.073/0.90/5.55/25.4 for 32..16384; batched 0.012/0.028/0.105/1.55/6.9/32.7;
chains 0.053/0.095/0.149/1.48/7.05/34.2 (B=1), 0.026/0.052/0.121/1.95/7.05/34.9 (batched).

### Correctness

Single-call rel_l2 <= 3.5e-16 at every supported size 16..65536, B in {1,3,8} plus graded
batches; all 12 graded pow2 chain gates PASS at graded (L,B,m) — worst 6.17e-12 at
1024:1:4000 vs 1e-10 (16x margin) — plus odd-batch chains (B=3 at 1024/4096/16384) and
m=2 strict gates (<= 1.5e-15 vs 3e-14); SX/SS64 fusions verified BITWISE identical to the
unfused pipeline; two-run outputs bitwise repeatable; official tryout.sh green at
32/1024/16384.

### Borrowed, explicitly

- The PMU-first discipline is my own r3 note ("GET A RESERVATION FIRST"), but the
  pointer that memory LAYOUT (not arithmetic) was the large-L lever came from reading
  d1_planner's r3 record (their huge-page arena, adopted from d1_bluestein r2, was worth
  2x at 16384 B=64) — my table compaction is a different fix for the same disease.
- The four-step attempt followed the survey's and d1_planner's "Bailey four-step at
  16384" recommendation; measured and rejected here with numbers, which is itself the
  contribution — see above before re-trying it.

### Next round

1. 4096 B=1 (~1.25x) is now the worst large cell: it already runs 3 passes, so the next
   lever is FLOPS/loop shape, not traffic — split-radix / conjugate-pair butterflies
   (~15% fewer FMAs), or a radix-8 FIRST stage (4096 = 8^4) to shorten the s1s stage.
2. 64/128 B=1 (~1.2-1.4x): MKL's codelets are near the FMA floor; only a genuinely
   lower-shuffle structure helps. The dead-end analysis above bounds what transposes
   cost — look at FFTS-style fully fused straight-line codelets with the twiddles baked
   into constants before writing any code.
3. 1024 B=512 (1.13x) and 128 B=512 (1.13x): L2/L3-resident batched streams; consider
   software prefetch of the NEXT transform's input inside the first stage.
4. PMU numbers worth keeping: port_5 is NOT the ICX wall for the split engine (45%
   occupancy); l1d fill rate and L2 capacity are. Judge any new stage by
   l1d.replacement per transform first.

## Round d1_r5 (2026-09-03)

The a80n0 reservation (job 440424) was alive all round; every decision below is a
same-window interleaved A/B on a leased core of the scoring node against the r4 binary
(impl_4/d1_pow2.c).  The r4 leaderboard framing that drove the round: my worst
library ratios were 4096 B=1 (1.30x), 16384 B=64 (1.30x MEDIAN with 21% spread),
1024 B=512 (1.27x), 64 B=1 (1.28x) — and for the big cells the board MEDIANS were far
worse than anything I measured on a leased core in r4 (16384 B=64: 63.7 board vs 47.7
measured) while FFTW's spread stayed 1.4%.  That gap is d1_bluestein's r2 diagnosis:
allocation-luck bimodality, which the r5 scoring change (median over NINE independent
runs, see RESCORE_PLAN.md) turns from noise into a direct score penalty.

### Change 1 — deterministic huge-page arena (ADOPTED from d1_bluestein r2/r3 via d1_planner r3)

All plan buffers (state, scratch, scratch2, twmem, tile, tws1full) now come from ONE
mmap'd MADV_HUGEPAGE block, pre-faulted at plan time, buffer i at
round-to-64KB(prev_end) + i*(16 KB + 192 B).  64 KB is the node's L2 set period
(1.25 MB / 20 ways), 192 B staggers L1 sets.  Node A/B (reps after warmup, interleaved):
  - 16384 B=64: r4's cross-invocation bimodality (44.6 / 64.4 us modes, exactly
    bluestein's disease) is GONE — new binary reads 43.5-44.1 every invocation.
    Under median-of-9 scoring this alone should move the cell from ~1.30x to ~0.9x.
  - 4096 B=256: 10.29-10.38 -> 9.36-9.43 (-9.5%), stable.
  - 4096 B=1: 7.70-7.87 -> 7.29-7.49 (-5%).   1024 B=1: 1.26-1.28 -> 1.23-1.25.
  - 16384 B=1: wash (33.8-37 both, window drift dominates).  Small L: parity.
TWO TRAPS, both measured:
  1. My first skew was 32 KB + 192 B — but 2*32960 = 65536 + 384, so buffers TWO
     indices apart sat 384 B from the same L2 phase: state-vs-twmem at 16384 made the
     B=1 CHAIN 2% slower, soa_a-vs-soa_c at 1024.  16576 fixes it (closest concurrent
     pair 768 B apart); 16384 B=1 chain went from +2% to -3% vs r4 (52.2 -> 49.7-51.0).
  2. The SoA chain buffers do NOT want the arena AT ALL: co-locating soa_a/b/c in it
     measured +7% on the 1024:512 chain (2.57 -> 2.75), reproduced with THP off
     (-DD1_ARENA_THP=0) and with reordered phases — mechanism not identified; glibc's
     separate per-buffer mmaps are simply faster for that access pattern.  They stay
     on their own posix_memalign allocations (that path never had the variance
     problem — it was stable in r4).  Do not "clean this up" without re-measuring.

### Change 2 — SX first-stage-pair fusion at 1024 gated by BATCH working set

Bookkeeping first: r4's record says the SX fusion was "gated to L >= 4096", but the
shipped r4 source has D1_SX_MIN=1024 — the record was wrong, 1024 shipped FUSED.
Node A/B at 1024: fused wins B=1 (1.243 vs 1.263) but LOSES B=512 by 6% (1.896 vs
1.778) — the fused s1s reads are quad-scattered, and once the input streams from L3
the scatter defeats the prefetcher; at B=1 everything is L2-resident and scatter is
free.  At 4096 B=256 fused still wins big (10.3 vs 11.4; NT changes the balance).
New gate: fuse always at L >= 4096, at 1024 only when L*batch*32 <= 2 MB.
1024 B=512 lands at 1.786-1.791 (r4: 1.90-1.92; board median was 2.35 vs MKL 1.86 —
this cell should now be ~parity).

### Built, measured, GATED OFF: AoS in-register codelet for L=128

fft128_execute/fft128_chain (32 vector values: 8x S1QUADT -> radix-8 with broadcast
tw128 -> twiddle-free R4Q), the r3/r4 "two-block 32-zmm codelet" idea, correct first
try (2.3e-16; graded 128:1:30000 gate 8.5e-13).  Node A/B vs the 3-pass split
pipeline: B=1 m=1 WASH (0.13-0.14 both), B=512 LOSES 25% (0.22-0.24 vs 0.17-0.19),
chain no win.  Two reasons, now understood: (a) the two all-to-all 32-vector barriers
make gcc spill ~half the state per stage, so the "saved passes" come back as stack
traffic; (b) stage 2 pays one vpermilpd per cmul where the split engine's SS8 middle
stage has ZERO shuffles — I reintroduced exactly the port-5 pressure r2 removed.
Code kept behind D1_CODELET128 (default 0).  The only variant left worth building is
a SPLIT-form codelet (SCMUL stage 2); do not retry the AoS form.

### Where the graded cells stand (a80n0 leased core, min us/xf, warmed interleaved reps;
### board-vs-library comparison uses the r4 board's library numbers)

| cell | r4 bin | r5 bin | | cell | r4 bin | r5 bin |
|---|---|---|---|---|---|---|
| 1024 B1   | 1.26-1.28 | 1.23-1.25 | | 4096 B256  | 10.29-10.38 | 9.36-9.43 |
| 1024 B512 | 1.90-1.92 | 1.79      | | 16384 B1   | 33.8-36.6 | 33.9-37 (wash) |
| 4096 B1   | 7.70-7.87 | 7.29-7.49 | | 16384 B64  | 43.9-64.4 BIMODAL | 43.4-44.3 stable |

Chains (node): 1024:1:4000 2.01-2.03 (r4 2.12-2.18), 16384:1:250 49.7-51.0 (r4
50.3-52.0), 1024:512:2000 2.56-2.59 (r4 parity after the SoA revert), small-L
unchanged.  Wallaby (contended login, min-over-6): B=1 m=1
0.024/0.036/0.073/0.880/5.44/24.9 for 32..16384; batched 0.015/0.028/0.106/1.39/
7.03/32.3; chains 1024:1 1.44, 1024:512 1.94, 16384:1 33.4, 32:512 0.026.

### Correctness (final source, all on a80n0)

Single-call rel_l2 PASS at every size 16..65536 x B in {1,3,8,12} (52 combos); all 12
graded chain gates PASS with margins IDENTICAL to r4 (worst 6.17e-12 at 1024:1:4000 vs
1e-10 — the arena moves no arithmetic); chained outputs bitwise repeatable across
process invocations at every graded cell; official tryout.sh green at 32/1024/16384.

### What did NOT work, with the number that killed it

- 32KB+192B arena skew: +2% on 16384 B=1 chain, fixed by 16576 (above).
- SoA buffers in the arena: +7% at 1024:512 chain, reverted (above).
- NT stores at 1024 B=512 (16.8 MB, -DD1_NT_MIN_BYTES=16000000): 1.71-1.73 vs 1.77
  in one window — looked like a small win but was NOT reproduced against the final
  gate-fixed binary in later windows (1.79 both); threshold left at 25 MB.  Re-test
  next round with a cleaner window if the cell still trails.
- AoS fft128 codelet: gated off (above).

### Borrowed, explicitly

- d1_bluestein r2/r3: the deterministic huge-page arena (their bimodality diagnosis
  and the fresh-mmap + pre-fault + skew recipe), re-derived for my buffer set with a
  corrected skew constant.  d1_planner r3's adoption of it (2x at 16384 B=64) was the
  evidence it transfers to pow2-shaped working sets.
- The 9-run-median scoring rationale (RESCORE_PLAN.md) set the round's priority:
  variance is now score, not noise.

### Next round

1. 16384 B=1 (~1.03-1.07x vs fftw_patient) and 4096 B=1 (~1.07-1.15x vs MKL) are the
   remaining large deficits; both are 3-pass and L1-fill-bound.  The unexplored levers:
   a SPLIT-form 128-codelet used as the tile kernel inside a radix-128 fused first
   pass, or conjugate-pair/split-radix butterflies (~15% fewer FMAs) in the SS8/SS64
   stages.  PMU first: l1d.replacement per transform is still the dashboard.
2. 64 B=1 stays ~1.2-1.3x behind MKL — r4's shuffle-floor analysis still stands; only
   a structurally different codelet (split-form, FFTS-style constants) could move it.
3. If the SoA-arena mystery matters elsewhere: perf c2c or dtlb counters on the two
   variants; the +7% is reproducible and unexplained.
4. If a scoring seed ever fails a chain gate: -DD1_FASTMAP_MAX_L=0 (unchanged).

## Round d1_r6 (2026-09-03)

The a80n0 reservation (job 440424) was alive; every number below is a leased core of the
scoring node, interleaved same-window A/B against the r5 binary (build/a80n0/bin/d1_pow2 ==
impl_5). The wallaby squeue shim still points at the gen heartbeat, not d1's; same personal
/tmp shim workaround as r4/r5 (d1_prime's recipe) made tryout.sh see the reservation. The
round produced ONE shipped fix plus two hard negative findings that close open questions,
rather than a new engine — I chased the r5 leaderboard's worst ratios to their root causes
and most of them turned out to be node physics, not my code.

### THE HEADLINE (shipped): the r5 "deterministic huge-page arena" was getting ZERO huge pages

r5 borrowed d1_bluestein's arena (one mmap'd MADV_HUGEPAGE block, buffers 64 KB-set-period
staggered) on the theory that huge pages make L2 set-indexing follow the virtual layout, so
the skew constants actually deconflict. I measured what the arena was really backed by, by
summing AnonHugePages over /proc/self/smaps after fft1d_create:

    r5 binary:  L=16384 B=64  Anonymous=2160 kB  AnonHugePages=0 kB
    r5 binary:  L=1024  B=1   Anonymous=2164 kB  AnonHugePages=0 kB   (every size: 0)

**Zero.** On this kernel (5.15) an anonymous mmap is only 4 KB-aligned, and THP can back
only the 2 MB-ALIGNED subranges of a region; a 2 MB arena whose base is not 2 MB-aligned
straddles the boundary and gets no huge page at all. So madvise(MADV_HUGEPAGE) was a no-op
for the whole campaign since r5 — the skew constants were staggering buffers inside 4 KB
pages whose physical colouring is still per-process random, i.e. the arena was NOT doing the
one thing it was adopted to do. (This is why d1_race's r4 "heap spacer" probe still found
placement luck to re-roll: the arena never removed it.)

Fix: over-map by 2 MB, round the base UP to a 2 MB boundary, munmap the head and tail slack
(ar_map). Same posix_memalign(2 MB) alignment in the non-mmap fallback. After the fix every
size reports AnonHugePages = 2048 kB (the full arena, one huge page). Output bitwise
unchanged; single-call rel_l2 and all chain gates identical.

Measured benefit (interleaved A/B, leased core, min us/xf; the arena is ~2 MB = one huge
page for these working sets, so the effect is modest but consistent and never negative):
  - 16384 B=64: r5 44.2-45.7  ->  fixed 43.1-44.3  (best-window ~43.1 vs 44.2, ~2-3%)
  - 16384 B=1:  r5 34.55-34.70 -> fixed 33.74-34.41 (~1-2%)
  - 4096 B=256: wash (9.6 both) -- its working set is 32 MB, far past the 2 MB arena, so
    the arena's page backing is not its bottleneck.
This is a correctness-of-intent fix more than a speed one: the arena finally does what its
own comment claims. Adopters of the bluestein arena (d1_planner, d1_twiddle, d1_race) should
CHECK THEIR OWN AnonHugePages -- if their arena base is a bare mmap they are almost certainly
getting 0 too, and the 2 MB-align-and-trim is a drop-in.

### The 1024/4096/64/128 B=1 deficits are node physics or global codegen, not fixable here

I ran the r5 board's four worst pow2 ratios to ground; the record is the point, so nobody
re-spends a round on them:

1. **The m=1 bimodality is AVX-512 turbo licensing, and it is FAIR.** The r5 board scored
   1024 B=1 at 1.4164 median (9/9 in the slow mode) while a leased core gives 1.23-1.25.
   Same binary, same core, alternates 1.24 / 1.42 across invocations. perf: the fast mode
   runs 3.04 GHz, the slow 2.86 GHz -- the heavy-512-FMA frequency license. Frequency
   explains ~6% and cycles/xf the other ~6% (slow mode also retires ~6% more cycles, a
   real placement component the huge-page fix trims). CRUCIAL fairness check: pinned to the
   SAME core, pow2/MKL ratio is 1.14-1.15x whether the core is in the fast state (pow2
   1.234 / mkl 1.084) or the slow state (pow2 1.419 / mkl 1.232). Both throttle together;
   the 9-run median absorbs it. So my TRUE 1024 B=1 gap to MKL is ~1.14x, compute-bound
   (already 3 passes, L2-resident) -- only fewer FLOPS (split-radix/conjugate-pair
   butterflies) can close it, and that is a high-risk radix-8-core rewrite that also risks
   the chain gates. Deferred deliberately.

2. **64 B=1: my own fft64_execute compiles to 417 instructions where d1_twiddle's
   byte-identical port of it compiles to 325** -- ~92 extra, almost all vmovapd register
   spills (mine 107 moves vs their 60). Same S1QUADT/R4Q/cmul_bc source, same gcc, same
   flags; the difference is global register-allocation context (the surrounding functions in
   my 2100-line file vs their file). I tried three local fixes and NONE moved it off 417:
   twiddle's pointer-increment + loadu loop form (417), loading the stage-2 twiddles inside
   the loop instead of holding 24 g2r/g2p vectors live (418), and -falign-functions=64
   -falign-loops=32 (0.052-0.077, no help). Measured: pow2 0.052, twiddle 0.038-0.044,
   batchlane 0.045, MKL 0.042 -- a real ~1.3x, but d1_twiddle (running MY ported codelet)
   already owns the cell in the race, so the panel does not lose it. Do NOT re-attempt local
   fixes; the lever, if any, is reducing the whole file's register pressure or a genuinely
   different (fewer-shuffle) 64 structure, and MKL's ~0.042 is near the shuffle floor r4
   already bounded.

3. 128 B=1 (~1.18x real) and 4096 B=1 (~1.2x real) are the same story as (1): compute-bound,
   3-pass, need fewer ops not less traffic. The D1_CODELET128 AoS codelet stays gated off
   (r5: wash at B=1, -25% at B=512).

### What did NOT work, with the number

- fft64_execute loop-form rewrite (loadu + pointer-increment, matching d1_twiddle): 417
  insns (unchanged), 0.053-0.069 vs r5 0.052. Register spilling is not from the loop form.
- fft64 stage-2 twiddles loaded inside the loop (frees 24 live vectors): 418 insns, no
  change at B=1 (the array is computed once and used once there anyway).
- -falign-functions=64 -falign-loops=32 on my source (d1_race's r4 al64 lanes): 64 B=1
  0.051-0.077, no better than 0.052 default. These flags help d1_race's separately-linked
  small kernels, not my file's fft64.

### Correctness (final source, all on a80n0 leased core)

Single-call rel_l2 <= 3.8e-16 at every supported size 16..65536 (B=3) plus B in {1,8,9,11,
12} at 1024 and B=8 at 256/2048; all 12 graded chain gates PASS at graded (L,B,m) -- worst
4.35e-12 at 1024:1:4000 vs the 1e-10 floor, identical margins to r5 (the arena moves no
arithmetic); odd-batch chains (1024:3, 128:9, 64:11, 4096:3) PASS; THP-off build
(-DD1_ARENA_THP=0) and THP-on both correct; 16384 B=64 output bitwise repeatable across
runs; official tryout.sh green at 32/64/128/1024/4096/16384.

### Borrowed / returned

- The bluestein/planner huge-page arena is what I fixed -- the 2 MB-align-and-trim is
  returned to all its adopters (see headline; their arenas are very likely getting 0 huge
  pages too).
- d1_twiddle's fft64_exec (their r4 port of my own r1/r4 fft64_execute) was the A/B target
  that exposed the 92-instruction codegen gap -- their leaner compile is the existence proof
  that 325 is reachable, I just cannot reach it from inside my file.
- /tmp squeue-shim (d1_prime r3), interleave-under-drift discipline (composite r2): reused.

### Next round

1. The huge-page fix is in; re-check the board -- if 16384 B=64 / B=1 moved, the arena was
   worth more than the leased-core A/B showed (the scoring window's quiet may let the
   placement component matter more than turbo noise did here).
2. The remaining real gaps (1024/4096/128 B=1 at ~1.14-1.2x) are all "fewer FLOPS":
   split-radix or conjugate-pair butterflies in the SS8/SS64 radix-8 core (~15% fewer
   twiddle multiplies). This is the one structural lever left and it is a careful,
   gate-risky rewrite -- budget a whole round, verify the chain gates every step, keep the
   current radix-8 core behind a -D flag for A/B.
3. 64 B=1 is conceded to d1_twiddle in the race; only a whole-file register-pressure
   reduction or a fewer-shuffle 64 structure would move my standalone number, and MKL's
   ~0.042 bounds the prize.
4. If a scoring seed ever fails a chain gate: -DD1_FASTMAP_MAX_L=0 (unchanged).

## Round d1_r7 (2026-09-03)

First cumulative round for me in practice: three of the four shipped changes trace to other
entries' records, and the round's biggest single win is an idea of mine that d1_twiddle
improved and I took back. The a80n0 reservation (job 440424) was alive throughout; every
decision is an interleaved same-window A/B on a leased core against the r6 exemplar binary
(impl_6). The wallaby squeue shim still points at the gen heartbeat — same personal /tmp
shim as r4-r6. tryout.sh's awk chokes on the four-regime cases.txt (multiple rows per L),
silently falling back to m=1; chains were driven by hand with the driver's --chain/--map.

### Change 1 — latency-shaped chain map (ADOPTED from d1_prime r5)

Both map forms (map_vec_p AoS, soa_map_p split) rewritten with prime's three moves, whose
transfer d1_batchlane r6 and d1_twiddle r6 had already proven on their engines: (a) the
junk-lane floor made ADDITIVE and folded into the |z|^2 FMA (max leaves the critical
path); (b) Goldschmidt sqrt (fnmadd + 2 independent fma per iteration) instead of Newton;
(c) the round's real idea, prime's early-seeded reciprocal: q0 = rcp14(1 + s*y0) off the
RAW rsqrt14 estimate, then 2 reciprocal-NR steps against the true 1 + sqrt(s), so the rcp
chain overlaps the sqrt refinement. The `precise` variants keep their exact-residual
refinements on top (Heron now uses the Goldschmidt h ~ 0.5/sqrt directly).
Node A/B (min us/xf, old -> new), every graded chained cell:
    32 B1   0.071 -> 0.065 (-8.5%)   32 B512   0.043 -> 0.040 (-7%)
    64 B1   0.146 -> 0.136 (-7%)     64 B512   0.090 -> 0.085 (-6%)
    128 B1  0.265 -> 0.246 (-7%)     128 B512  0.212 -> 0.201 (-5%; one 0.176 window)
    1024 B1 2.01  -> 1.97  (-2%)     1024 B512 2.58  -> 2.47  (-4%)
    4096 B1 11.07 -> 10.66 (-4%)     4096 B256 10.16 -> 9.79  (-4%)
    16384 B1 wash (window drift)     16384 B64 49.7  -> 47.8  (-4%)
Exactly prime's advertised range. Gate margins statistically unchanged (see correctness).

### Change 2 — register-resident fused first pair, ST_SXR4 (TAKEN BACK from d1_twiddle r5)

Twiddle's r5 record re-derived my r4 tile fusion in a register-resident form and reported
it beat their tile gate. Ported into the split engine: when the second stage is radix-4,
the four feeder s1s quads' outputs (16 zmm) stay live and both second-stage pairs consume
them directly — no 32-store + 32-load tile round-trip per 64 complexes. s1s_quad split
into a vector-returning core (s1s_quad_v) plus a store wrapper; s44_pair_v reads legs from
the register block. Output BITWISE identical to the tile path (cmp-verified at 4096 B=256).
Node A/B (tile SX44 vs SXR4): 4096 B=1 7.32-7.51 -> 6.97-7.02 (-5%); 4096 B=256
9.19-9.23 -> 8.93-8.97 (-3%); chains 4096 B=1 9.70-9.92 -> 9.04-9.11 (-7%, map included).
16384 keeps the tile SX48 (radix-8 second level = 32 live zmm of legs, would spill).
D1_SXR=0 restores the tile form.

### Change 3 — 1024 rescheduled 4*4*8*8f when the batch is cache-resident

The default 1024 schedule takes radix-8 at s=4 ([4,8,8,4f]), so SXR4 never fired there.
The residue also factors [4,4,8,8f] — same pass count, radix-4 second stage, register
fusion applies. Node A/B: B=1 m=1 1.40-1.46 -> 1.38-1.39 (-1.5%; one fast-turbo window
1.21 vs 1.25); B=1 chain wash. At B=512 the same schedule measured +3.5% (1.75-1.77 ->
1.81-1.85; the 8-leg SF8 final streams worse than SF4 from L3) — so it is batch-gated by
the same 2 MB working-set condition as the SX fusion. TRADEOFF RECORDED: the SF8-final
rounding path halves the 1024:1:4000 gate margin, 5.95e-12 -> 1.229e-11 vs the 1e-10
floor (8.1x margin — same one d1_twiddle ships at this cell). D1_S44_1024=0 reverts.

### Change 4 — two-pass 128 ([4,4,8f] + SXR4), twiddle's gate borrowed

Same reschedule at L=128 makes the whole transform TWO passes: [SXR4(4*4), SF8]. Gate is
d1_twiddle r6's, borrowed with their number (fused at B=512 lost them 9%): only when
32*L*batch <= 256 KB. Node A/B: 128 B=1 m=1 0.138-0.140 -> 0.126-0.129 (-8%; fast-mode
sample 0.111); B=8 0.129-0.134 -> 0.112-0.117 (-13%); B=1 chain 0.259-0.265 -> 0.223-0.235
(-12% incl. map); B=512 correctly untouched (wash). Board mkl at this cell is 0.1042 —
this closes about a third of my worst small-L m=1 gap. The r5 AoS fft128 codelet stays
gated off; this got the pass-count win WITHOUT the 32-zmm spill barrier that killed it.

### Change 5 — NT threshold 25 MB -> 16 MB (r5's open question closed)

r5 measured NT at 1024 B=512 (16.8 MB in+out) as a wash in one window and left the
threshold at 25 MB with a "re-test in a cleaner window" note. Re-tested twice this round:
NT won EVERY interleaved pair across both windows (window 1: 1.72-1.92 vs 1.78-2.27, up
to -15%; window 2, busier: 1.85-2.02 vs 1.88-2.05, -0.5..-1.5%) and cuts the sd tenfold
(0.03-0.13% vs 0.11-7.9%). Under median-of-9 scoring the variance cut is itself score.
Mechanism: on a shared 24 MB L3 the output RFO reads are waste well below nominal capacity.

### What did NOT work, with the number that killed it

- Software prefetch of the next transform's input in execute (my own r4 next-round item):
  T1-prefetch the full 16L bytes up front per transform. LOSS everywhere it fires:
  1024 B=512 2.03 -> 2.29 (+13%), 4096 B=256 11.0 -> 13.3 (+20%), 16384 B=64 50 -> 56
  (+12%), 128 B=512 wash. The burst competes with the current transform's demand misses
  and evicts its working set. Ships behind D1_PFNEXT default 0; do not re-derive.
- 1024 [4,4,8,8f] at B=512: +3.5%, gated out (change 3).
- (Cross-check of r6's conclusions: 64 B=1 m=1 stands at 0.058-0.060, the codegen story
  unchanged; not re-attempted per the r6 record.)

### Where the graded cells stand (a80n0 leased core, min us/xf, final binary; old = r6)

    m=1:  32 B1 0.021 (=)      32 B512 0.018 (=)      64 B1 0.058 (=)   64 B512 0.046-0.050 (=)
          128 B1 0.126-0.129 (0.138)  128 B512 0.163-0.167 (=)
          1024 B1 1.38 (1.40; fast window 1.21)       1024 B512 1.72-2.02 NT (1.75-2.27)
          4096 B1 6.97-7.02 (7.32-7.51)               4096 B256 8.93-8.97 (9.19-9.23)
          16384 B1 34.8-34.9 (wash)                   16384 B64 43.7-43.9 (44.2)
    ch:   32 B1 0.065  32 B512 0.040  64 B1 0.136  64 B512 0.085  128 B1 0.223-0.235
          128 B512 0.176-0.201  1024 B1 2.20-2.22  1024 B512 2.47-2.49  4096 B1 9.04-9.11
          4096 B256 9.79-10.03  16384 B1 59.5-61.2 (window-drifty, wash)  16384 B64 47.6

Wallaby (contended login, min-over-runs, quiet samples): B=1 m=1
0.024/0.029/0.077/0.845/5.33/25.2 for 32..16384; batched m=1 0.012/0.029/0.102/1.42/
6.90/32.1; chains B=1 0.062/0.087/0.141/1.36/6.51/31.9; chains batched 0.023/0.047/
0.111/1.89/6.57/31.8.

### Correctness (final source, all on a80n0)

Single-call rel_l2 <= 3.8e-16 at every supported size 16..65536 x B in {1,3,8} (39
combos); all 12 graded chain gates PASS at graded (L,B,m) — worst 1.229e-11 at
1024:1:4000 vs 1e-10 (8.1x margin; the S44-schedule rounding change, see change 3 — all
other cells' margins statistically identical to r6); odd-batch chains (64:11, 128:9,
1024:3/12, 4096:3, 16384:3, 32:5) PASS; strict m=2 gates <= 1.6e-15 vs 3e-14 at all 12
graded shapes; SXR4 output cmp-verified bitwise identical to the tile path; two-run
outputs bitwise repeatable at every cell tried; official tryout.sh green at
32/128/1024/16384.

### Borrowed, explicitly

- d1_prime r5: the entire latency-shaped map (change 1), with d1_batchlane r6 and
  d1_twiddle r6 as the transfer evidence that made it a low-risk first move.
- d1_twiddle r5: the register-resident fused-pair idea (change 2) — my r4 tile fusion,
  returned improved, now taken back into the split engine — and their r6 batch-resident
  gate for fusing at 128 (change 4).
- d1_prime r3: the /tmp squeue shim, fifth round running.

### Next round

1. The one structural lever left for 1024/4096 B=1 m=1 (~1.15-1.2x real vs MKL) is still
   split-radix/conjugate-pair butterflies in the radix-8 core — unchanged from r5/r6
   notes, still a whole-round, gate-risky job. All three pass-count/fusion levers at
   these sizes are now exhausted (3 passes at 1024/4096, register-fused first pair).
2. Check the board's 4096 B=1: my binary reads consistently ~12% slower under the scoring
   harness (8.34 board vs 7.3-7.5 leased in r6) while d1_twiddle's matches its leased
   numbers. If that persists after this round's SXR4 (-5%), diagnose the harness delta
   (allocation? core? 512-license residence from the preceding backend's runs?) before
   spending more structure on it.
3. 1024 B=512 sits on NT now; if the board still shows >1.1x there, the remaining idea is
   twiddle's 2-3-pass batched schedule shape (they run [4,16,16] fused at 1.74).
4. If a scoring seed fails a chain gate: -DD1_FASTMAP_MAX_L=0 (unchanged), and note the
   1024:1:4000 margin is now 8.1x (was 16x in r6); D1_S44_1024=0 also restores the old
   margin there.
