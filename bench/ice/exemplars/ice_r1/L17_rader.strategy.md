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
