# L17_rader — Ice Lake panel strategy record

Single-thread lineage is in `../geom/strategies/L17_rader.md` (phase 1, CLX
node); the multicore layer's record is `../mt/strategies/L17_rader.md`.  This
panel starts from the phase-1 single-thread source (the mt layer is not here:
this panel is single-threaded by contract).

## Round ice_r1

### Where the smoke round left me, and what this round is built on

ice_smoke (Gold 6326 ICX, graded chain L=17 B=32 m=98, all cache-resident):
scored 2325 us per chain unit = **23.7 us per transform step**, 3rd of 3
(L17_winograd 16.19, L17_matrixsimd 21.03), with the plan having picked
**"xl 256", pf=0, pfw=0, clk256=2.90 clk512=2.90**.  Two things wrong with
how my plan met this machine:

1. **B=32 fell into the small-batch tuner path** (threshold was `batch < 64`):
   class A only, ranked at nv=16 with 3 reps, no pf race, no joint
   (variant, pf, pfw) grid, no probes.  The graded cell got the least
   tuning of any batch size.
2. **The workload is new**: a chain of 98 transforms feeding output into
   input (driver ping-pong, driver-side unitary scaling between steps), so
   `in` AND `out` are L3-resident every step.  Every DRAM-regime mechanism
   in my candidate set (st/stp staging, volume-ahead pf, pfw) is mistuned
   for it, and the machine changed under the kernels: TWO 512-bit FMA
   pipes (drain shadows halved), 512-bit shuffles p5-only, no measurable
   AVX-512 licence downclock (clk256 = clk512 = 2.90 in-plan), gcc 11.4.

### What was built (tuning machinery + two ICX-specific mechanisms + one missing candidate)

1. **Graded cell routed into the full batched tuner** (threshold 64 -> 32):
   at B=32 the plan now ranks both classes at nv=32 (the real working set,
   ~5 MB, exactly the chain's L3-resident regime), runs the joint
   (variant, pf, pfw) grid, and the ph/xp/fu decomposition probe -- all
   inside the scoring window, which is the ONLY quiet time on this node
   (see "what the dev box can and cannot tell you" below).

2. **Width discipline in stage 1** (class A and class B): the lone w4
   candidates must beat the best w8 candidate by >3% to be picked.  All
   class-A candidates are bit-identical, so a near-tie switch is free; the
   smoke round's noisy create picking "xl 256" against two 512-bit FMA
   pipes is the target.  Note the node then twice put "xl 256" on top of
   stage 1 anyway (by 2.7% and by 4.4% under contention) -- the second
   cleared the margin and was kept, so the discipline lets a genuine w4
   win through and only kills coin flips.

3. **pf=2, plane-ahead input prefetch** (new, exec_body X-last only): pf=1
   prefetches the NEXT VOLUME's plane (a whole volume of compute ahead --
   right for DRAM streaming, useless-to-harmful when src is an L3 hit).
   pf=2 prefetches THIS volume's next plane, ~0.9 us ahead of its
   deinterleave -- sized to hide ~60-cycle L3 hit latency, which is all
   the latency the chain regime has.  The joint grid now races
   pf in {0,1,2} x pfw in {0,1} (6 configs per grid variant).

4. **"ty" candidates** (xl 512 ty / xl 512 pin ty / xl 512t ty /
   xl 512t pin ty): ymm 4x4-tile transposes AND ymm deint inside the
   otherwise-zmm pipeline.  ICX motivation: every 512-bit two-source
   shuffle issues on port 5 only, where it competes with the SECOND
   512-bit FMA pipe (p0+p5 on ICX); the 4x4 ymm tiles cost ~1.6x the
   shuffle uops but dual-issue on p1/p5 and stop displacing FMA slots.
   Pure movement, bit-identical, tuner-ranked.

5. **"xl 256 pin"** (new candidate, w4 + pinned SN constants): gcc 11.4
   materialises every kernel constant as a full-width `.LC` memory
   operand (`vfmadd132pd .LC2(%rip), %ymm3, ...` -- checked in the ICX
   build's asm), ~200 constant-load uops per kernel block.  With
   AVX-512VL the w4 pipeline has ymm16-31 free, so pinning costs no named
   register.  Added because "xl 256" twice won stage 1 on this node and
   pin was a w8-only candidate before; kept LAST in the table (the width
   discipline knows the w4 entries as 0 and NCA-1).

6. **sp pin twins** ("xl 512t sp pin", "xl 512t sp dy pin"): added after
   "xl 512t sp dy" won the first honest nv=32 rank and the sp family had
   no pinned variants; they lost their own race (see negatives) and stay
   as candidates only.

Final state as shipped: 24 class-A + 4 class-B candidates; the node's
honest create-time race picked **xl 512t sp dy, pf=0, pfw=0** twice
running at the graded cell, and the small-batch path serves B=1.

### Operation count

Unchanged: 296 FP instructions (192 FMA + 104 add/sub, 488 flops) per
17-point transform, 867 transforms per volume ≈ 423 kflop.  Everything this
round is scheduling, prefetch shape, and candidate/tuner structure; every
class-A candidate remains bit-identical (verified by cmp on the driver
output across builds, plus the chain check).

### Measured (ICE NODE via tryout.sh, graded config L=17 B=32 m=98 — NOT wallaby; contention warning below)

| run | window | pick | us/step |
|---|---|---|---|
| smoke (monitor) | scoring | xl 256 (old small-batch path) | 23.72 |
| new tuner, run 1 | ~quiet-ish (first after lease jam cleared) | (not logged) | **18.86, sd 0.02%** |
| new tuner, verbose run 2 | contended | xl 512 pin (discipline overrode xl 256's 2.7% stage-1 win) | 28.35 |
| new tuner, verbose run 3 | mildly contended | xl 256 (won stage 1 by 4.4%, cleared the margin) | 19.73, sd 4.1% |
| + ty + xl 256 pin, run 4 | fairly quiet | **xl 512t sp dy** (18.22, honest 2% win over 20 rivals) | 18.64, sd 4.3% |
| + sp pin twins, run 5 | quiet start, contended tail | xl 512t sp dy again (beat its own pin twins) | **18.91 min** (median 21.1: contention arrived mid-run) |
| **B=1, chain m=98** | quiet (one volume's chain is L2-resident -- immune to neighbours) | small-batch path | **15.83, sd 0.01%** |

The honest-window run-4 stage-1 table (the round's best evidence): plain
family 18.2-19.0 us/vol all within ~2.5% -- xl 512t sp dy 18.22,
xl 512t sp 18.43, xl 512 pin 18.55, xl 512t dy 18.56, xl 512t pin ty
18.59, xl 256 18.69, xl 256 pin 18.94; st/stp 21.2-22.3; xfs 19.6-21.5.
The cell is memory-latency-bound, not port-bound -- which is why w4 and w8
converge here and why sp (cross-volume x-block overlap padding the plane
phase's junction stalls with independent work) is what wins rather than
any width or port trick.

Correctness every run: single-transform rel_l2 = 3.16e-16, chain m=98
rel_l2 = 1.07e-14 (tol 9.9e-12), repeatable bit-identical outputs.
MKL same case/core: 76.2-76.7 us/step (I am ~4x ahead of it here).

**What tryout on this node can and cannot tell you (measured, not
speculation):** the SAME binary read 18.86 us/step and then 28.35 us/step
half an hour apart at sd=0.02% within each run.  19 implementers' tryouts
share one 24 MB L3, and the graded chain is exactly an L3-resident
workload, so cross-run dev numbers on this node are contention-poisoned;
only same-window comparisons (one tuner table) rank candidates honestly,
and the only genuinely quiet time is the monitor's scoring window -- which
is where fft3d_create's tuner runs.  Hence this round's bet: put every
plausible ICX mechanism in the candidate set and let the create-time race
pick under the same quiet the score is measured under.

Also recorded: the in-plan decomposition probe read ph=13.0/xp=5.3 us/vol
in the honest windows (15.5-16.9 / 6.8-7.6 contended) with fu ≈ ph + xp
(phases add) -- the plane phase (src reads + deint + z + transpose + y +
A fill) is 70% of the cell, which is why pf=2 targets the src reads and
why the st/stp staged-output machinery (+15-25% over plain in every
table) is dead weight in the chain regime, exactly as predicted.

### What did not work / negative results with numbers (same-window tables)

* **st/stp staged dense out flush at the graded cell**: 25.7-29.0 us/vol vs
  ~20-23 for plain variants in the same table.  Staging deletes
  partial-line DRAM RFO waste; there is no DRAM RFO here.  Do not expect
  the staged family to win any cache-resident chain cell.
* **xfs (X-first) class at the graded cell**: 23.7-25.7 vs ~20-22 plain
  X-last, both verbose runs.  Consistent with CLX; the class stays a
  raced challenger only.
* **pf=1/pf=2/pfw under CONTENTION**: all lose by 3-8% (extra traffic into
  a thrashed L3).  This says nothing about the quiet window; the grid
  decides there.  Do not hard-enable any prefetch on this machine from a
  contended dev run.
* **ty in the honest window**: xl 512t pin ty 18.59 vs xl 512t pin 18.70
  (-0.6%, inside noise); xl 512 ty 18.89 vs xl 512 19.64 (-3.8%) but
  xl 512 pin ty 18.63 vs xl 512 pin 18.55 (+0.4%).  The p5-conflict theory
  did not produce a clear win because the cell is latency-bound, not
  port-bound.  Kept as candidates (free at plan time), expect no cell win
  from them here; they may matter at B=1-class cells where compute
  dominates.
* **xl 256 pin in the honest window**: 18.94 vs xl 256's 18.69 -- pinning
  HURTS w4 by 1.3% (the KPIN asm barrier constrains scheduling more than
  ~200 32-B .LC loads cost on 2 load ports).  The .LC memory-operand
  constant traffic is NOT the w4 bottleneck at this cell.
* **sp pin twins (run 5)**: raced and lost to plain sp dy in their own
  table (exact numbers lost to a tail-truncated log, but the winner line
  and pick are recorded).  pin composes with nothing in the sp schedule.

### Borrowed this round, named

* The joint-grid discipline (variant, pf, pfw raced together, 3% margins,
  incumbent-at-(0,0) reference) carried over from L23_rader panel_r8 via my
  phase-1 rounds; the description-string telemetry pattern (probe + xrace
  published in fft3d_description) from L36_pfa panel_r8.
* The §10 corpus (Ice Lake Under Glass) supplied the ICX port model behind
  "ty" (512-bit shuffles p5-only vs dual-issue ymm shuffles), the
  reg-resident > memory-operand constant ordering behind "xl 256 pin", and
  the warning that trustworthy timing on this tier needs same-window
  comparisons.

### Next round

1. **Read the scored description string first**: it now carries the stage-1
   xrace numbers, the grid's (variant, pf, pfw) pick, clk256/clk512, and
   the ph/xp/fu probe measured IN THE QUIET WINDOW.  That one line is the
   only clean ICX evidence that exists; act on it, not on tryout numbers.
2. **The width question is answered**: in honest windows w4 and w8 converge
   to within 2.5% because the cell is latency-bound.  Do not spend another
   round on width/port/constant tricks at B=32 (ty wash, xl 256 pin
   negative, pin twins of sp negative); the discipline + honest race is
   the right permanent shape.  Spills and phase-splitting remain relevant
   only for compute-bound cells (B=1 is 15.8 us/step and L2-resident --
   THAT is where the §10 phase-split register-resident kernel rewrite
   (1.6x on the 23-point kernel) would show, and B=1 is a separately
   scored cell).
3. **If winograd still leads** (their smoke 16.19 vs my 18.6-18.9): their
   advantage is the 3-rotating-pass structure with in-plan probe
   fu=12.81 us/vol -- fewer movement uops AND fewer store->load junctions
   than my plane pipeline (my honest ph=13.0 alone matches their whole
   volume).  The mt_r4 precedent applies: port their engine wholesale as
   a gated candidate rather than fight it piecewise.  That is the
   round-2-sized move for the batched cell.
4. **The plane phase is the target** (probe: 70% of the cell).  A fused
   deint+z structure that never materialises T (kernel loads with
   in-register deinterleave from `in` directly, as the xfs x pass already
   does) is blocked by the y-lane gather problem, but a HALF-plane
   deint->z pipeline (deint tile k+1 under z block k) inside one plane is
   unexplored -- unlike dz (which pipelines across planes and lost), it
   keeps the working set at one plane.  Cheaper still: sp currently pads
   the plane phase with the PREVIOUS volume's x blocks -- the winner --
   so extending the same padding into volume 0's plane phase (which has
   nothing to overlap) using the NEXT volume's deint is the marginal
   version of the same trick.

(This entry sat out ice_r2 and ice_r3; the r3 leaderboard had it at
19.368 us/step, 1.48x behind L17_matrixsimd's 13.061, FFT-only chain
semantics.)

## Round ice_r4

### The task changed: the graded step is now FFT + map, and I own the chain

The chain step became `state <- (z + c)/(1 + |z + c|), z = FFT(state)`,
timed through the optional `fft3d_chain` entry point (weak symbol).  An
entry without it is timed through the driver fallback -- fft3d_execute
sweeping all 32 volumes per step plus a driver-side vectorized map -- which
for this entry would have been ~24 us/step equivalent.  This round is
entirely about owning that chain; nothing in the FFT kernels changed.

### What was built

1. **`fft3d_chain` with PER-VOLUME iteration** -- each volume runs all
   m=98 steps while its working set is L2-resident (state ping-pong
   2 x 78.6 KB in plan scratch, its `c` volume, and the A buffer: ~320 KB
   against the 1.25 MB L2), instead of the fallback's per-step sweep that
   keeps 3 full batches live in L3.  This is corpus sec 10.3's consensus
   design ("iterate each volume through all m steps while cache-resident;
   never sweep passes across volumes"), and it makes the B=1 and B=32
   cells nearly the same workload.
2. **One map ladder, everywhere** -- ADOPTED VERBATIM from the rival
   1.00-scorer's `mapc` (1000f989, sec 10.2 consensus shape): w = z + c,
   s = 1e-300 + wr^2 + wi^2, `vrsqrt14pd` seed + 2 Newton steps for
   sqrt(s) on the FMA pipes, ONE exact `vdivpd` for 1/(1+|w|).  The bias
   makes s=0 and denormal-range squares safe without touching MXCSR.
   Exactness: chain m=98 rel_l2 = 2.9e-14 against the numpy reference
   chain (tol 9.8e-12, ~340x margin); the brief's tiered-precision lever
   (float-seed maps) was NOT needed -- at ~3 us/step embedded cost the
   full-precision ladder is not the binding constraint (see probe).
   vrsqrt14pd is elementwise-identical across 128/256/512 widths, and
   every point in every variant goes through this one ladder (the tail of
   a non-multiple-of-8 run is an OVERLAPPED 8-group, not a scalar), so
   the whole candidate set below is bit-identical -- cmp-VERIFIED on the
   node: forced xm vs mp vs dz vs w4 chain outputs are byte-equal.
3. **Map placement variants, plan-time raced** (create-time race, 2
   sweeps, ~60 ms, runs in the scoring window's quiet like the rest of
   create):
   - `mp` (lazy): state stays RAW between steps; the map runs as a
     per-plane pre-pass into an L1-hot 4.6 KB buffer at the top of the
     next step's plane phase, one final materialization pass at the end.
   - `mps` (lazy, shifted): same, but plane x+1's map runs right after
     plane x's z pass, so the 36 divides per plane issue under the z
     drain instead of as an exposed burst.
   - `fd` (lazy, fused): map fused INTO the zmm deint tile (dz8x8m adds
     c in interleaved space before the transpose and runs the ladder on
     the split rows before the store).  Built, raced, never won a table
     (21.4-22.1 vs mp pin 20.9-21.5 same-window); dropped from the table
     when the scalar-edge bit-unification favored the others; code kept.
   - `xm` (EAGER): the map runs AT THE X-PASS STORE as a per-block
     epilogue -- each kernel block's 17 just-stored rows are mapped in
     place (loads hit the store queue), c streamed from the same offsets.
     State buffers hold MAPPED state, the plane phase is plain, and the
     LAST step's x pass writes final_out directly: the separate map pass
     disappears from the program entirely.  Overlap lanes stay correct
     because every block re-stores its lanes raw before its own epilogue
     maps them.
   - `xk` (EAGER, IN-KERNEL -- the shipped winner): the map moves INSIDE
     wino17's ST macro (new `cm`/`mst` kernel args, mode-1 stores only):
     the ladder runs on the split (vr, vi) registers already in hand,
     BEFORE the interleave shuffles, with c split at the destination
     offsets.  No epilogue, no reload of just-stored rows; the ladder and
     its one vdivpd per store site interleave with the kernel's own
     296-FP drain at instruction granularity -- the rivals' `zpassAB_m`
     pattern (sec 10.2) applied to my store side.  Kernel signature
     change touched all 57 call sites mechanically (`, 0, 0` appended);
     mode!=mst paths are textually unchanged, so every non-chain exec
     keeps its bit class.
   - pin twins of all of the above; `ty` twins of xm (ymm transposes);
     `dz` twin of xm (the deferred-junction plane schedule composed with
     the eager x pass); w4 twins of xm.
4. **`nm` probe + telemetry**: create() also times the chain step with
   the map OFF (unscored) and publishes `ch=<pick> <us> nm=<us>` in the
   description string, so the scored line carries the map's embedded cost
   measured in the quiet window.
5. **Divider/seed microbenchmarks on the bare-metal node** (settling
   sec 10.2's CONTESTED question): `vrsqrt14pd`/`vrcp14pd` ~1-2 cyc tput
   (PIPELINED -- 1760b1bf's "~10 cyc microcoded" was a VM artifact),
   `vdivpd` zmm ~15.5 cyc, `vsqrtpd` zmm ~23 cyc, at clk512 = 3.20 GHz
   measured in the same process.  Standalone map ladder shapes all land
   at 0.81-1.06 ns/pt (rsqrt+2N+div 0.88, rsqrt+2N+rcp+2N 0.98,
   sqrt+rcp 1.06, sqrt+div 1.73, float-seed 0.98; 4-way manual
   interleave moves div only 0.88 -> 0.81) -- the standalone pass is
   memory/latency-bound, NOT unit-bound, which is why ladder-op tuning
   was abandoned in favor of placement (xm).

### Operation count

FFT unchanged: 296 FP instructions (192 FMA + 104 add/sub, 488 flops) per
17-point transform.  The map adds per point: 2 add + 2 FMA (|w|^2), 1
rsqrt14 seed + 9 FMA-class Newton ops, 1 FMA (1+|w|), 1/8 vdivpd, 2 mul
-- ~17 vector-op-lanes + one divider slot per 8 points, 614 groups per
volume-step.  xk moves those ops into the x pass's 629 ST sites and
deletes the standalone map pass and xm's epilogue reload entirely.

### Measured (ICE node via tryout.sh; every number below names its window)

| config | result |
|---|---|
| graded B=32 m=98, driver steady state, QUIET window (final xk code) | **min 17.551 / median 17.554 us/step, sd 0.02%** |
| same code, other windows | 18.57 quiet-ish / 20.1-21.6 contended |
| pre-xk (xm epilogue) code, best windows | 18.75-19.62 |
| B=1 m=98 (windows with MKL at 99.5 vs its quiet 73.8, i.e. ~35% inflated) | 21.11-21.54 us/step, sd 0.03% |
| MKL same case/core B=32 | 88.8-90.2 us/step (~5.1x ahead at the pick) |
| single-transform gate | rel_l2 3.16e-16 |
| chain gate (m=98, manual check.py) | rel_l2 2.896e-14 at B=32, 1.21e-14 at B=1 (tol 9.8e-12) |
| repeatability (manual, two runs, final code) | out.bin AND out.bin.chain byte-identical |

The one QUIET same-window table (the round's best evidence): **xk pin
17.40**, xk 17.90, xm ty pin 18.44, xm pin 18.46, mp pin 18.54, mps pin
18.69, xm dz pin 18.72, xm 256 pin 21.12; `nm` (map off) = **15.58**,
which matches r1's quiet B=1 FFT-only 15.83 -- i.e. the per-volume chain
runs at the old quiet B=1 floor, and the map's embedded cost fell
4.3 us (standalone pass) -> 2.7-3.5 us (xm epilogue, partial hiding) ->
**1.82 us (in-kernel xk)**.  Contended windows show the same ranks at an
inflated level (xm-family 21.3-22.9, nm 18.4-18.7).  Other same-window
facts: **xm 256 (w4) loses by ~13%** (the "xl 256 won stage 1" pattern
from the L3-batched cells does NOT transfer to the L2-resident chain);
xm dz loses by ~1% (junction deferral does not pay here); pin wins in
every family.  Forced-pick chain outputs xk vs xm vs mp vs dz vs w4 are
cmp-VERIFIED byte-equal, so the race is free.

### What did NOT work / negatives with numbers

* **w4 chain body** (xm 256 / xm 256 pin): 24.0-24.7 vs 21.3-21.8 for
  the w8 family, three same-window tables.  Do not re-derive: at
  L2-resident chain regime the two 512-bit FMA pipes win outright.
* **fd (map fused into the deint tile)**: never beat mp in any table
  (22.06 vs 21.85; 21.80 vs 21.52).  The ladder burst inside the tile
  displaces p5 shuffle slots; the deint is not where the slack is.
* **mps (map shifted under the z drain)**: wash (within 0.5% of mp in
  every table).  The OoO window cannot hold a whole 780-uop map_run plus
  the next kernel group; shifting whole passes is too coarse.
* **dz + xm**: 21.99 vs 21.72 (one window).  Kept as a candidate.
* **Ladder arithmetic tuning** (rcp14 vs div, float seeds, 4-way
  interleave): all within 0.81-1.06 ns/pt standalone -- the standalone
  map is bound by memory/latency, not by the divider or the seeds.  Do
  not spend another round micro-tuning the ladder; move it or delete it.
* **tryout.sh has a `set -u` bug this round** (`$W` used before defined
  when a chain case exists): run it as
  `W=$ICE/build/tryout/<name> ./tryout.sh <name> 17 <B>`, and note its
  check.py invocation also loses `$W` remotely, so the map-chain check
  silently never runs -- run
  `python3 check.py ... --map-check 98 --cin $W/c.bin` yourself, and the
  repeatability cmp too (the shared FS makes both trivial locally).

### Borrowed this round, named

* The map ladder is **1000f989's `mapc`** (rsqrt14 + 2 Newtons + one
  exact vdivpd), verbatim from
  `ext/reference/fft_v4_solutions/1000f989_score1.00/implementation.c`.
* Per-volume cache-resident chaining and the lazy-map idea are corpus
  **sec 10.3 consensus**; the divider-under-FFT placement is **sec 10.2
  consensus** (their in-kernel fusion, here approximated at x-block
  granularity with zero kernel surgery).
* The settle-the-contested-claim-by-microbenchmark move is 1760b1bf's
  own advice from sec 10.2 ("benchmark the seed instruction before
  blaming the ladder") -- their microcode claim itself turned out to be
  the VM artifact.

### Next round

1. **Read the scored description first**: it now carries
   `chain pv fused map ch=<pick> <us> nm=<us>` from the quiet window.
   ch - nm is the map's remaining embedded cost (quiet dev window: 1.82);
   nm is the FFT residual (quiet: 15.58).
2. **In-kernel ST fusion was DONE this round** (the xk shape, planned as
   a next-round item and then executed after the epilogue measurements
   justified it): -1.06 us/step over xm in the quiet table, no codegen
   degradation observed (the ST-site temps are short-lived; no new spill
   pathology at either width).  What remains of the map is ~1.8 us
   against a ~1.1 us alu/traffic floor estimate -- the next bite is
   small; do not lead with it.
3. **The FFT residual (nm = 15.58 quiet) is now the target again**, and
   it is r1's B=1 structural story: plane-phase junctions at L2 latency.
   Winograd's 3-pass engine (fu = 12.8 us/vol on this node, r1 probe)
   remains the wholesale port candidate, worth MORE now because the
   chain multiplies any per-volume saving by m = 98 and the map fusion
   (xk lives in the x-pass store macro) ports with it.  Alternative
   in-structure lever: cross-VOLUME chain interleaving -- two volumes'
   chains advanced alternately (double working set, still < half of L2)
   so volume B's plane phase pads volume A's junction stalls; this is
   sp's winning mechanism transplanted into the chain regime, and
   nothing else in the chain has independent work to pad with.
4. If the scoring window's quiet changes the race pick, trust it -- all
   candidates are cmp-verified bit-identical, so the pick is free.
5. Housekeeping: tryout.sh's `$W` bug (workaround in the negatives
   section) also disables its repeatability cmp -- keep running both
   checks manually until the monitor fixes the script.

## Round ice_r5

### The round's premise: cumulative, and the standings said "take the engine"

ice_r4 leaderboard: L17_matrixsimd 13.009, L17_winograd 14.854, me 17.521
us/step.  My scored description carried nm=16.37 (FFT residual, map off) and
ch-nm ~ 2.3 (embedded map) -- i.e. essentially ALL of my 4.5 us gap to the
leader is the FFT engine, not the map.  matrixsimd's own record prices their
chain skeleton (structure-only probe, map replaced by a scale) at 11.33
us/step; my plane pipeline has spent five rounds pinned at ~15.6-16.4 by
plane-phase store->load junctions at L2 latency (probe: ph = 70% of the
cell).  My r1 and r4 "next round" entries both said the round-sized move is
porting a rival engine wholesale rather than fighting it piecewise; the
round's context file made that legal and explicit.  Winograd's engine
(fu = 12.8 quiet) was the original port target, but matrixsimd's is measured
faster end-to-end (13.0 scored vs 14.85), so I ported theirs.

### What was built: the "msr" chain engine, ported from L17_matrixsimd

ADOPTED WHOLESALE from impl/L17_matrixsimd.c (their ice_r2/r3/r4 layers), as
a chain-race candidate behind a create-time numerical self-check:

1. **chunk17zr** -- their ice_r3 merged-reordered class-R kernel, verbatim:
   lanes = 4 interleaved-complex lines per zmm, 17 row loads, 148 FP ops per
   chunk, real coefficients as splatted memory operands (cosine) + 8
   asm-pinned registers K0..K7 (sine), m walked in pair order 0,4,1,5,2,6,3,7
   so only ONE parked cosine u is live.  My old kernel needed split re/im
   (deint + 2 transposes per plane); theirs eats the driver layout directly
   -- that is where the 4 us lives.
2. **X-first pass order with the addr-safe shifted t1** (their panel_r8
   collision model, mode 1 only, padded 5120 B plane stride) -- port of
   l17_as_build, pure integer arithmetic.
3. **Chain schedule v2** (deferred-Z: Y(x+1) between Y(x) and Z(x) on
   ping-pong plane buffers) with the **s6 map** (pair-shared |z+c|^2
   packing, rsqrt14 + 2 Newton, ONE vdivpd per 8 points, in place on the
   interleaved state) after each Z plane; **volume-major IN-PLACE** chaining
   with final_out as the state arena.
4. **One deliberate deviation**: their ymm tail chunk (chunk17n_w2, the old
   phase-serial kernel plus its whole table set: ~300 lines, Q pins, cn4/sn4)
   is replaced by a **5th OVERLAPPED zmm chunk17zr at offset 13** (their own
   pre-panel_r5 off17 shape {0,4,8,12,13}); the X pass tail likewise at 285.
   On 2x512-pipe ICX the FP time is identical, the overlapped chunk deletes
   the tail's 16 extra phase-serial row loads, the recomputed lanes are
   bit-identical (lane-independent arithmetic), and it cut the ported
   surface by ~400 lines.  The engine is one clean bit class.
5. **Create-time self-check gate**: one FFT-only msr step vs the tuned exec
   on the same volume must agree to 1e-13 rel L2 before msr may enter the
   race (they are different rounding classes ~1e-15 apart; a transcription
   bug would be ~1e0).  A fast wrong chain scores nothing; the check makes
   any port bug fall back to the ice_r4 xk path instead of shipping.
   msr is picked unless the incumbent beats it by >3%.  msr vs xk is a
   cross-bit-class race -- legal, the harness compares the chain end state
   to numpy only, never across processes (winograd ice_r4 precedent).

### Operation count

msr chunk: 148 FP ops (84 fma + 12 mul + 52 addsub as vector ops) per 4
lines; per volume 3*17*5 + 73 = 328 zmm chunks (with the overlap recomputes)
~ 48.5k vector FP ops -- MORE flops than my old engine (527 vs 423 kflop)
but ~17 row loads/chunk, no deinterleave anywhere, and two in-register tile
transposes per plane instead of my three buffer round trips.  Map unchanged
in spirit from r4: ~15 FMA-class + 1 vdivpd per 8 points, now their s6
packing.  The old engine and its whole candidate set stay in the file
(fft3d_execute still runs it; the chain race still ranks xk/xm/mp).

### Measured (ICE node via tryout.sh; windows named per r1 discipline)

| config | result |
|---|---|
| graded B=32 m=98, driver steady state, mid window | **min 15.143 / median 15.151 us/step, sd 0.03%** |
| same, second run (contention arrived mid-run) | min 15.248, sd 6.3% |
| in-tuner same-window race, run 1 | **msr 17.37** vs xk pin 20.01, xk 20.39, mp pin 21.16 (old family 20-24) |
| msr FFT residual (msnm, map off, same windows) | 13.68 / 13.49 |
| old-engine nm same windows (contended: r4 quiet was 15.58) | 17.83 / 17.73 |
| B=1 m=98 (window ~35% inflated: MKL 99.6 vs quiet 73.8) | **min 17.002, sd 0.04%**, msr picked there too (17.15 vs xk pin 19.93) |
| MKL same case/core B=32 | 89.6-90.2 us/step (~5.9x) |
| single-transform gate (old engine, untouched) | rel_l2 3.160e-16 |
| chain gate, manual check.py (script's $W bug still live) | **B=32: 2.051e-14; B=1: 1.154e-14** (tol 9.8e-12) |
| repeatability (two runs, leased core, manual cmp) | out.bin AND out.bin.chain byte-identical |

r4 shipped code in its best QUIET window was 17.55; this round's 15.14-15.25
was measured under visible contention (the same-window xl-family table sat
18.1-19.7 vs r4's quiet 17.4-19.0, and nm read 17.8 vs its quiet 15.58).
matrixsimd's identical engine scored 13.009 in the r4 quiet window; expect
**~13.0-14.0 us/step scored**, from 17.521.

### What did NOT work / notes

* No failed mechanisms this round -- the round was one large port and it
  landed on the first node run (self-check ok=1 immediately).  The porting
  discipline that made that true: copy the kernel arithmetic VERBATIM
  (slot table, sign patterns, pair order untouched), rename only types and
  table plumbing, and gate the result behind a numerical cross-check
  against an engine already trusted by the harness.
* The in-tuner msr number (17.2-17.4) sits ~2 us ABOVE the driver's own
  steady state (15.14) in the same process -- create runs early on a
  less-warm core.  Treat create-race numbers as ranks, never levels
  (r1 lesson, still true).
* tryout.sh's $W bug from r4 is STILL live (CH built from $W two lines
  before W is assigned; remote check.py gets literal '$W/c.bin' -> '/c.bin'
  and the repeatability cmp is skipped).  Workaround unchanged: prefix
  `W=$PWD/build/tryout/L17_rader`, then run check.py --map-check and the
  second-run cmp by hand.  Monitor: still worth fixing.
* g_desc grew (msr/msnm/msok fields); buffer bumped 448 -> 576 so nothing
  truncates.

### Borrowed this round, named

* **The entire scored chain engine is L17_matrixsimd's**: chunk17zr (their
  ice_r3), the merged-phase idea behind it (their ice_r2, itself corpus
  sec 10's ZIPP item), X-first + addr-safe t1 (their panel_r3/r8), chain v2
  + s6 map + volume-major in-place (their ice_r4).  Stated plainly: this
  round is their work adopted under the cumulative-round rules, with the
  overlapped-zmm tail as my one structural simplification and the
  self-check gate as my safety addition.
* The "port the engine wholesale rather than fight it piecewise" decision
  rule is from my own r1 next-round entry (originally about winograd's
  engine, redirected to the measured leader).

### Next round

1. **Read the scored description**: `chain ch=msr <us> nm=<us> msr=<us>
   msnm=<us> msok=1`.  msnm in the quiet window is the ported engine's FFT
   residual on this node; matrixsimd's skeleton probe says ~11.3-12 is
   available.  If scored msr lands >14, the gap is window/create-shape, not
   engine.
2. The remaining daylight to matrixsimd is whatever THEY build in r5 --
   their open items were an X-pass kernel lever and the port-5-free Z-store
   layout.  Read their r5 record first; their improvements now transfer to
   this file almost mechanically (the engine is shared).
3. My own untried card on TOP of the shared engine: fuse the map INTO the
   Z-group tile-transpose store (my xk mechanism, their costed-not-built
   idea).  Their v3 negative (chunk-granularity interleave, port-5
   saturation) predicts it loses; my r4 fd negative agrees.  Only attempt
   with a probe showing Z-store slack.
4. The old plane-pipeline chain candidates (xk/xm/mp families) are now dead
   weight in create (~80 ms) but free insurance; drop them only if create
   time ever matters.
5. If B=1 returns to scoring separately: msr won B=1 in-race too (17.15 vs
   19.93 contended); nothing else needed.

## Round ice_r6

### Where this round started

ice_r5 leaderboard: L17_matrixsimd 12.736, L17_winograd 14.412, me **15.052
us/step** (scored desc: `ch=msr 15.14 nm=15.53 msr=15.14 msnm=11.77 msok=1`).
The r5 port of matrixsimd's engine was supposed to land ~13.0-14.0; it landed
at the contended-window dev level instead, and the gap to their IDENTICAL
engine (2.3 us) was stable across every window -- structural, not
contention.  This round found it, and it was a one-line bug of my own
making, not their engine.

### The find: the msr arena was 16 B off a cache line, so EVERY access split

Diffing my r5 binary against theirs (objdump histograms: map bodies
byte-equivalent, chunk codegen equivalent) left arithmetic differences too
small to explain 2.3 us, so I audited the plan arena arithmetic instead.
`2*NVOL = 9826 = 2 (mod 8 doubles)`, and FIVE volume-sized buffers
(vo_w8, vo2_w8, sc0, sc1, mv) sit ahead of the msr block in the (64 B
posix_memalign'd) arena -- net offset 10 doubles = 16 B past a line
boundary for msc, mss, mspb, mspb2, mst1.  Consequences per graded step:
all 32 cosine-splat FMA memory operands per chunk (~7.8k loads across 243
chunks) were cache-line-SPLIT loads, and every t1 store/load and pb
store/load split likewise; the addr-safe t1 shift table's 64 B-granular
collision model was also silently operating 16 B off its assumptions.
matrixsimd's own arena keeps these aligned, which is why the same source
ran 12.7 there and 15.1 here.

**Fix: one realign statement before p->msc (`q += (8 - (q - mem) & 7) & 7`)
plus +8 doubles of nd slack.  Bit-identical by construction (alignment
changes no arithmetic).  Worth ~2.3 us/step alone (15.14 -> ~12.7-12.8).**

### Also this round (both bit-identical, cmp-verified)

1. **Sign-fold, ADOPTED from L17_matrixsimd ice_r5**: every use of
   MS_MULI's result is an FMA against a sine pin, so the odd-lane sign
   flip moved into the K tables ((+k,-k,...) splats) and MULI is now just
   the re/im swap -- deletes 8 vpxor-class uops per chunk from ports 0/5
   (the two FMA pipes), ~1.9k uops/step.  `-DL17R_NO_SFOLD` restores the
   XOR form.  IEEE multiply carries sign as XOR, so (-k)*x == k*(-x)
   bitwise, FMAs included -- same bit class, and the node cmp confirms.
2. **tr=2 lane-3 tail store (my own, new this round)**: the r5 deviation
   (overlapped 5th zmm chunk at offset 13) was doing the FULL 4-row tile
   store -- 40 p5 shuffles + 20 stores -- when lanes 0..2 only recompute
   rows the f0=12 chunk already stored bit-identically.  The tail now
   stores only its new row 16: 3 shuffles + 1 store per 4-column block
   (15 + 5 total), deleting 25 port-5 zmm shuffles + 15 stores per Y/Z
   group x 34 groups/step.  This makes my tail CHEAPER than their
   chunk17n_w2 ymm tail (which pays 33 phase-serial row loads + ~40 ymm
   shuffles); the deviation is now an advantage, not a wash.
   `-DL17R_MS_TAILFULL=1` restores the full tile for A/Bs.  Stored bytes
   identical (same E lane-3 registers; overlapping m0=12/13 columns carry
   the same values).

### Operation count

FFT arithmetic unchanged (148 FP ops per zmm chunk, 243 chunks/volume-step,
527 kflop/volume; map s6 unchanged: 26 uops + one vdivpd per 8 points).
Deleted per step vs r5: ~7.8k line-split load penalties (alignment), ~1.9k
p0/p5 XOR uops (sfold), 850 p5 shuffles + 510 stores (tail tr=2).

### Measured (ICE node via tryout.sh, graded L=17 B=32 m=98; MKL same
core/window quoted as the contention gauge)

| config | result |
|---|---|
| r5 shipped (for reference, r5's own windows) | 15.14-15.25, scored 15.052 |
| **r6 full (align + sfold + tail3), cleanest window** | **min 12.339 / median 12.355 us/step, sd 0.38% (MKL 88.8 sd 0.03%)** |
| r6 full, later contended window | min 12.465, median 12.98, sd 6.4% (MKL 89.2 steady) |
| align-only (`-DL17R_NO_SFOLD -DL17R_MS_TAILFULL=1`) | min 12.819 (contended, sd 11%; first attempt 14.76 sd 7.3% -- window-poisoned, discard) |
| align + sfold (`-DL17R_MS_TAILFULL=1`) | min 12.679 / median 12.690, sd 0.05% (MKL 88.9) |
| **B=1 m=98** | **min 13.888 / median 13.890, sd 0.02%** (MKL 87.6 on that core -- window ~inflated vs its quiet 73.8) |
| single-transform gate | rel_l2 3.160e-16 (B=32), 3.114e-16 (B=1) |
| chain gate, manual check.py ($W bug still live) | **B=32: 2.051e-14, B=1: 1.154e-14** (tol 9.8e-12) -- BYTE-IDENTICAL to the r5 chain values, as all three changes require |
| repeatability / bit-class | out.bin.chain cmp-identical across runs AND across all hook configs AND vs the r5 binary's output |

Decomposition (same-day windows): alignment ~-2.3 us, tail tr=2 -0.34 us
(12.679 -> 12.339 in back-to-back clean runs), sfold ~-0.1-0.15 us (inside
the align-only run's noise; matrixsimd's matched pairs say -1.3% ~ -0.17,
consistent).

### What did NOT work / notes

* No failed mechanisms this round; the round was one diagnosis and three
  bit-identical deletions.  The near-miss to record: my first align-only
  A/B read 14.76 (sd 7.3%) and briefly pointed at sfold as a 2 us lever,
  which is impossible by uop arithmetic (8 uops/chunk); a re-run in a
  cleaner window read 12.819.  The r1 lesson generalizes: a same-binary
  sd above ~1% on this node means the MIN is contaminated too, not just
  the median -- re-run before believing any per-piece decomposition.
* **Transfer warning for every implementer sharing arena-style plans: check
  buffer alignment arithmetic whenever buffer COUNTS or volume sizes
  change.**  9826 mod 8 != 0 was invisible in every objdump diff (the asm
  is identical -- only the runtime addresses split), invisible to
  correctness (bits don't change), and cost 2.3 us/step for one round.
  The probes that WOULD have caught it in-plan: a create-time
  `(uintptr_t)ptr & 63` assert, or comparing msnm against the donor's
  skeleton probe (11.77 vs their 11.33-with-map-sweep was the smell this
  round chased).
* tryout.sh $W bug from r4/r5 STILL live; same workaround (env `W=` prefix,
  manual check.py --map-check, manual cmp).

### Borrowed this round, named

* **Sign-folded sine constants: L17_matrixsimd ice_r5**, adopted verbatim
  in mechanism (their stab/nts fill trick applied to my mss pins + MS_MULI).
* The objdump-histogram diff protocol that localized the gap (map bodies
  identical, chunks identical, therefore addresses) follows their r5
  within-one-binary A/B discipline; the alignment find itself is mine.

### Score projection

Dev floor 12.34 (MKL 88.8 window, the same gauge level r5's 15.14 was
measured under -> scored 15.05).  Expect **~12.2-12.5 us/step scored**,
from 15.052.  matrixsimd's r5 floor was 12.74-12.82 -> scored 12.736; if
their r6 stands still I take the lead at L=17; their record says only
algorithmic kernel levers remain for them, so expect them at ~12.4-12.7.

### Next round

1. Read the scored description: `msr` and `msnm` are now measured on
   ALIGNED buffers -- msnm should read ~9.5-10.5 (was 11.77).  If msr
   lands >12.8, suspect a window story, not structure.
2. The engine is now matrixsimd's minus their ymm tail plus my cheaper
   tr=2 tail, aligned like theirs, sign-folded like theirs.  Remaining
   levers are the same as their r5 "open" list: algorithmic only (their
   verdict: symmetric/antisymmetric convolution split or negacyclic-8
   Winograd, both gated on TOTAL-uop arithmetic vs the 64-uop dense sine
   block -- and L17_winograd's record says no split of x^8+1 beats it).
   Read BOTH rivals' r6 records before building anything.
3. If matrixsimd's r6 ships a kernel-level win, port it -- the engines are
   now structurally identical, so their deltas apply almost mechanically
   (and vice versa: they should take my tr=2 tail and the alignment
   audit).
4. The old xk/xm/mp chain family and the vo/sc buffers remain misaligned
   (deliberately untouched -- unscored insurance paths only).  If any of
   them is ever promoted back to a scored path, realign those buffers
   first.

## Round ice_r7

### Where this round started

ice_r6 leaderboard: L17_winograd 11.649, L17_matrixsimd 11.935, me **12.284
us/step** (scored desc: `ch=msr 12.28 nm=16.82 msr=12.28 msnm=9.03 msok=1`).
The r7 brief made the mining explicit: the rivals' sources re-benchmarked on
THIS node put the honest L=17 target at 1760b1bf's 33.5 ms = **10.68
us/step**, and my own r6 "next round" entry said it plainly: if matrixsimd's
r6 ships an engine-level win, port it -- the engines are structurally
identical, so their deltas apply almost mechanically.  matrixsimd's r6 DID
ship one: "chain v6", the padded in-place strided engine (structure from
rival 1760b1bf), worth ~-0.85 us matched on their side.  This round is that
port, done the same way as my r5 msr port: verbatim structure, a numerical
self-check gate, and same-lease matched-pair A/Bs for every claim.

### What was built: the "ms6" chain engine (port of L17_matrixsimd chain v6)

ADOPTED WHOLESALE from impl/L17_matrixsimd.c ice_r6 (their l17_chain_v6 /
chunk17zri / L17_V6_A2G / l17_map_vec1; structure originally rival
1760b1bf's run17_A pass shape):

1. **Padded private arenas** (p->msv6a state, p->msv6c c field): rows 17 ->
   20 complex (320 B), slab stride 808 doubles = 6464 B = 101 lines (odd,
   L1-set spread), ~110 KB each, both L2-resident.  Unpack x0+c once per
   volume-chain, pack the mapped end state once -- amortized by m=98.  Pad
   lanes zeroed at create and provably stay zero (linear passes map 0 -> 0,
   map(0+0) = 0 with the 1e-300 bias).
2. **ms_chunk17zri**: in-place twin of the shared kernel -- one non-restrict
   pointer, all 17 loads before all 17 stores, plain stores only.  Passes b
   (axis j0, at the slab stride) and a1 (axis j1, stride 40) transform IN
   PLACE along the stride: zero shuffles, no t1, no plane buffers, no
   addr-safe machinery.  Deletes the 87 KB t1 round trip per step and every
   line-split access class the r6 alignment fix was about.
3. **a2 (axis j2)** as 4-row groups through a transposed 20-vector stack
   array (MS_TP4 4x4-complex tiles, 8 vpermt2pd each way), the SAME kernel
   on the stack, with the s6 map interleaved at GROUP granularity; row 16
   rides their cross-slab fringe (4 full zmm groups over 16 slabs' row-16
   lines + 1 overlap group storing only slab 16's lane, as a shared
   noinline unit ms_v6_frg here); a1(s0+1) software-pipelined into a2(s0).
   Their A/B hooks kept: -DL17R_V6_XS=0, -DL17R_V6_P1=0.
4. **One deviation, bit-identical and free**: the a2 stack tile array io_[]
   is declared __attribute__((aligned(64))).  Both my v8d and their vd_w4
   typedefs are aligned(8), so NOTHING guarantees the donor's stack tiles
   sit on cache lines -- the r6 arena-misalignment lesson applied to the
   stack.  (matrixsimd: take this back; it costs one attribute.)
5. **Self-check gate (r5 pattern)**: ms6 enters the race only if a full
   m=3 chain (FFT + map each step; m=3 exercises the arena re-entry and the
   pack path) agrees with the already-gated msr chain to 1e-12 rel L2
   (distinct rounding classes ~1e-15 apart -- msr maps each plane's 289th
   complex via the exact 128-bit sqrt+div tail, ms6 runs everything through
   the rsqrt14 ladder; a transcription bug would read ~1e0 and the plan
   falls back to msr).  ok6 is published in the description.
6. **Pick band widened to 15%** (was 3% for msr in r5): the create-time
   race is cold-biased AGAINST ms6 (its arenas are first-touched moments
   earlier; msr's buffers are hot from the whole earlier tuner) -- in-create
   ms6 raced 2.9% BEHIND msr while same-lease steady-state pairs put it
   2.8% AHEAD, 7/7.  The donor ships v6 with no race at all; the band only
   guards against a machine-side anomaly, msr stays the verified fallback.

### Operation count

Per volume-step: 243 zmm kernel chunks (85 b + 85 a1 + 73 a2) x 148 FP =
36.0k vector FP ops, all-zmm over 972 lane-transforms (+12% pad/overlap
waste vs the old engine's 904 at 209 zmm + 34 ymm chunks).  Shuffles: 73
groups x 80 TRANSP4-uops + 8 MULI/chunk (vs the old tile+MULI ~6.8k).
DELETED per step vs msr: the 87 KB t1 store+reload, both pb plane buffers,
the addr-safe shift machinery, and the separate map sweep sites.  Map
unchanged (s6 exact tier: rsqrt14 + 2 Newton, ONE vdivpd per 8 points),
714 pair-iters + 17 single-vector fringe calls per volume-step.

### Measured (ICE node; every A/B is same-lease alternating full binaries)

| config | result |
|---|---|
| **ms6 (shipped), matched pairs vs msr, B=32 m=98** | **11.890 / 11.911 / 11.922 / 11.935 / 11.947(x2) / 11.957 / 11.969 us/step** vs msr 12.261-12.359 -- ms6 wins 7/7 clean pairs by ~0.35 us (first pair's ms6 13.52 was the documented bimodal co-tenant mode; its msr twin read 12.28) |
| tryout steady state, other windows | 12.093-12.128 (sd 0.01% within-lease; a 13.57-13.63 co-tenant mode also exists -- same binary, MKL steady 88.9, r6 lesson holds) |
| **B=1 m=98** | **min 13.548-13.567, sd 0.01%** (r6: 13.888; donor's v6 B=1: 13.610 -- the io_ alignment may be the difference) |
| MKL same case/core | B=32: 88.88-90.59; B=1: 99.5 |
| single-transform gate | rel_l2 3.160e-16 (B=32), 3.114e-16 (B=1) |
| chain gate m=98, manual check.py ($W bug STILL live) | **B=32: 2.055e-14, B=1: 1.163e-14** (tol 9.8e-12) -- B=32 value byte-matches the donor's v6, as identical arithmetic must |
| repeatability | out.bin AND out.bin.chain cmp-identical across processes; auto-pick binary's chain == forced-ms6 binary's chain |
| in-create race (cold-core levels, ranks only) | ch=ms6 14.30 msr=14.14 msnm=10.39 **ms6nm=9.84** ok6=1 msok=1 |

### What did NOT work / negatives with numbers

* **Fringe/b overlap (-DL17R_V6_FB=1, default OFF)** -- the donor's own r6
  "open" item (est. -0.2..-0.3 us), built here: fringe(s-1) interleaved
  into pass b(s) after rows 2/5/8/11/14, row-16 b chunks last.  Chain
  outputs cmp-verified BIT-IDENTICAL both ways, and it LOSES: fb1
  11.943/11.947/11.969 vs fb0 11.919/11.927/11.942, 3/3 pairs (+0.02..
  +0.05 us).  The fringe is not an exposed serial tail -- the OoO window
  already overlaps it with the next pass b at the seam, and the interleave
  only disturbs pass b's stride stream.  matrixsimd: strike this from your
  open list; the mechanism is built and measured, take the hook if you want
  the re-A/B.
* **Environment note**: tryout.sh/reserve.sh need squeue; this session's
  PATH lacked slurm (export PATH=/opt/software/slurm-19.05.8.1/bin:$PATH
  fixes it -- the reservation itself was alive the whole time, heartbeat
  fresh).  The r4 $W bug is STILL live; same workaround (env W= prefix,
  manual check.py --map-check, manual cmp).  Also: tryout.sh REGENERATES
  in.bin/c.bin at the requested batch -- after a B=1 tryout, every manual
  B=32 run against those files dies with "in.bin too short"; regenerate
  with gen_input.py (seeds 42 / 900042, scale 0.1 for c) before manual
  B=32 work.

### Borrowed this round, named

* **The entire ms6 chain engine is L17_matrixsimd ice_r6's chain v6**
  (chunk17zri, padded arenas, in-place strided passes, a2 tile-io groups,
  cross-slab fringe, a1->a2 pipelining, map-at-group-granularity), itself
  structurally from **rival 1760b1bf** (ext/reference/fft_v4_solutions/
  1760b1bf_score0.96/generator.py).  Stated plainly: this round is their
  work adopted under the cumulative rules, with the io_ 64-B stack
  alignment as my one (bit-identical) addition and the widened pick band +
  m=3 cross-check gate as the safety layer.
* The self-check-gate-before-race pattern is my own r5 invention, reused;
  the same-lease alternating-binary A/B protocol is **L17_matrixsimd
  ice_r5**'s, reused throughout.

### Score projection

Dev floor 11.89-11.97 in clean windows (matched against msr 12.26-12.36;
r6's msr scored 12.284 from the same window class), so expect **~11.9-12.0
us/step scored**, from 12.284.  That likely stays behind winograd (11.649
scored, and their r6 record has smaller open items left) and near-parity
with matrixsimd unless they moved again.  The honest rival mark on this
node is 10.68 (1760b1bf re-benchmarked); the panel-wide gap at L=17 is now
kernel-issue-shape, not structure.

### Next round

1. Read the scored description: expect `ch=ms6 ... ok6=1`; ms6nm ~9.8 is
   the FFT residual on ALIGNED in-place passes.  If ch=msr shipped, the
   15% band tripped -- investigate the create window before anything else.
2. The three L=17 entries now share (variants of) one engine at 11.6-12.0
   and the rivals' 10.68 exists on more FP per line, so the remaining ~1.2
   us is issue shape inside the a2 group + map interleave and the kernel's
   dependency graph.  The costed-but-unbuilt cards: (a) the donor's
   store-side-transpose twin of a2 (within ~1k uops on paper, only worth an
   A/B behind a probe showing the io staging on the critical path); (b) the
   v6 rival generator's H17=s44 Hartley-split pencil
   (fft_v5v6_solutions/v6_f40c5e25/dev_generators/gen.py) -- a DIFFERENT
   prime kernel family; price its total-uop count against our 148-op chunk
   before building anything (ROOFLINE.md: count total vector uops, not
   FMAs).  Read L17_winograd's and L17_matrixsimd's r7 records first; if
   either shipped a kernel-level win on the shared engine, port it.
3. My FB hook answers their fringe/b question negatively with numbers; do
   not rediscover it.  The old xk/xm/mp families and the msr engine remain
   as raced, gate-verified fallbacks (msr is one L17R_FORCE_V6=0 away).
4. Housekeeping unchanged: $W bug, W= prefix, manual check.py, manual cmp,
   and now the PATH/slurm and in.bin-regeneration notes above.
