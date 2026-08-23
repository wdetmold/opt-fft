# L13_rader — strategy record (ICE panel)

Continuity note: this entry arrived on the ice panel as the phase-1 panel_r11
exemplar (see `bench/geom/strategies/L13_rader.md` for rounds panel_r1–r11 and
`bench/mt/strategies/L13_rader.md` for the multicore fork). Everything below is
Ice Lake, single core, the graded chain of `cases.txt` (L=13: B=32, m=1278,
unitary, ~2.25 MiB working set, L3-resident).

## Round ice_r1 (2026-08-22)

### Where it stood

First measurement of the inherited CLX-tuned code on the graded chain
(a80n0, Xeon Gold 6326, gcc 11.4, `./tryout.sh L13_rader 13 32`):
**6.052 µs/xform** (race adopted pw-off), MKL same case 6.287 — only 1.04×
ahead. With the race disabled and the CLX defaults (`-DL13R_AB=0`: unfused,
pw=1, pf=0) it reads **6.531**, so ~0.5 µs of the baseline was already the
in-plan race correcting the CLX pw gate. `-DL13R_TSC` phase split at B=32:
xpass 4463 cyc/vol, z+y 15051 (with the MB loads, see below), plus ~1.0
µs/xform of driver-side unitary scaling inside the timed unit. The graded
point is **latency/L3-bound, not port-bound**: z+y sits ~1.6× above its
port floor even after the p5 accounting below.

### The Ice Lake port thesis, and what survived measurement

On CLX (one 512-bit FMA pipe on p0) the ~5.5k shuffles/volume of the z-pass
TR8 transposes rode free on p5. On ICX p5 is the second FMA pipe, so every
512-bit shuffle steals an FMA slot; the port floor is (13.0k FP + 5.5k
shuf)/2 ≈ 9.25k cyc/vol, and shuffle removal lowers it. Two surgeries, both
bit-identical (pure data movement), gated for A/B:

1. **tstore13 — extract-form transposing stores** (`-DL13R_XST`, default 1):
   the z→U store transpose per 4-column group as unpck/vpermt2pd (8 p5)
   with halves stored via ymm store + `vextractf64x4`-to-memory (store
   ports, no p5); column 12 through a 64 B bounce buffer; pad columns 13–15
   no longer written. 24 p5/component vs TR8's 48. Node: **6.514 vs 6.531
   (wash, kept)** — the freed p5 slots aren't the constraint at the graded
   point, but the form also drops ~1.4 KB of stored bytes/volume and reads
   cleanly; expect it to matter if the workload ever goes port-bound.

2. **tload13 — merge-masked `vbroadcastsd` transposing loads**
   (`-DL13R_MB`, default **0** = TR8): one load-port µop per element
   (fold verified in gcc 11 disasm), zero p5, 104 load µops/component vs
   16 loads + 48 shuffles. Node: **7.874 vs 6.514 — +1.36 µs/xform,
   REJECTED.** The 8-deep merge chains per column serialize behind L1/L2
   latency exactly where the kernel is already latency-bound. Lesson: on
   this workload p5 relief only pays if it doesn't lengthen the load
   dependency chains.

### Gate fixes (measured)

* **pw gate moved L2 → L3.** CLX rule (`batch·vol > L2 ⇒ pw=1`) turned
  prefetchw on at B=32; the chain keeps `out` L3-resident and the node
  priced pw=1 at **+7.4%** (6.531 vs ~6.05). New rule: pw only past L3
  (same shape as mt_r4's flip, re-derived single-core). Still raced both
  ways in-plan.
* **pf now raced both directions** (was only raced OFF when the gate had
  it on): under the chain the batch never exceeds L3, but pulling the next
  volume's input L3→L2 is a live question — see below.

### Fused-at-batch re-opened

The r9 CLX verdict (fused zy +4.2% at B=16) came from the machine where z
transposes were free; on ICX the fusion premise (feed y-FMAs into z's p5
bursts) is live again. `fuse` is now raceable at batch (`f1` arm, um=7) and
forceable via `-DL13R_FUSE_B=1`.

### Knob sweep on the node (all `-DL13R_AB=0`, graded chain B=32 m=1278; runs
within one block are same-window and comparable, cross-window drift is real —
see below)

| window | config | µs/xform | verdict |
|---|---|---|---|
| w1 | inherited code, race on | 6.052 (sd 4.1%) | first look, noisy window |
| w2 | MB=1 XST=1 (both surgeries) | 7.874 | MB kills it |
| w2 | MB=0 XST=1 | 6.514 | XST a wash vs old |
| w2 | MB=0 XST=0 (= inherited + AB=0) | 6.531 | reference |
| w3 | pw=0 (new gate) | 6.388 | |
| w3 | + X2 | 6.359 | −0.5% (not reproduced, see w5) |
| w3 | + FORCE_PF=1 | 6.601 | pf hurts, gate stays |
| w3 | + FUSE_B=1 | 6.725 | fused-at-batch loses on ICX too |
| w4 | X2 control | 6.715 | window drifted +5.6% |
| w4 | + XPF | 6.732 | null |
| w4 | + PIN=2 | 6.653 (sd 3%) | inconclusive, PIN=1 stands |
| w4 | + PIN=0 | 6.880 | pinning still worth ~2.5% |
| w4 | + PS=184 | 6.760 | odd-line pad still loses here |
| w5 | X2 vs no-X2 paired | 6.667 vs 6.670 | X2 is a wash, left OFF |

**Window drift is the elephant**: identical configs read 6.36–6.72 across
process instances (sd within a run 0.02%), while MKL holds 6.29–6.46. Most
plausible mechanism (corpus §3): physical page coloring of the driver's three
1.1 MB chain buffers against a 1.25 MB L2 — outside our control, and the
monitor's scored processes will roll the same dice. All decisions above were
taken from same-window pairs only.

A sixth window then flipped the pw pairing (pw=1 6.746 vs pw=0 6.882, MKL
6.39 both) — opposite sign to w2/w3. Verdict: **pw at the graded point is a
±2% page-coloring coin flip**, not a knob. The L3 gate keeps the principled
default (off when the chain is L3-resident) and the in-plan race's
3%-in-both-blocks bar correctly refuses to flip on it.

### Shipped configuration (ice_r1)

* `tstore13` extract-form z→U store transpose (XST=1), TR8 loads (MB=0),
  both alternatives kept compile-gated with the numbers that decided them.
* pw/pf gates on L3 (both off at the graded B=32); pw, pf raced both ways
  and fused-at-batch raced (`f1`) at the full scored batch per process;
  B=1 races fused-vs-unfused as before. X2/XPF/PS/PIN defaults unchanged
  (measured null or negative this round).
* Arithmetic untouched: 186 vector FP per 13-point transform, 70 blocks per
  volume; outputs bit-identical to the panel_r11 exemplar at every batch
  (same rel_l2 digits on every run above).

### Final node numbers (default build, race on)

| case | this round | MKL same window |
|---|---|---|
| graded chain B=32 m=1278 | **6.687 µs/xform** (window range for near-identical configs: 6.05–6.88) | 6.390 |
| B=1 m=1278 | **5.369 µs/xform** | 5.734 |

rel_l2 = 4.031e-16 (B=32) / 3.973e-16 (B=1) single-transform, 3.267e-13
whole-chain (tol 3.6e-11); bit-identical outputs across repeat runs; AVX2
and `-Wall -Wextra` builds clean.

### What did not work, with the number that killed it

1. **Merge-masked broadcast load transpose (tload13, MB=1): +1.36 µs/xform**
   (7.874 vs 6.514, same window). The p5→load-port conversion is correct in
   port accounting but the 8-deep merge chain per column adds serial load
   latency exactly where the graded chain is already latency-bound. Do not
   reopen unless the workload becomes port-bound (e.g. if the batch ever
   goes L1/L2-resident).
2. **Fused zy schedule at batch: +5.3%** (6.725 forced vs 6.388, same
   window). Same verdict as CLX r9 but for the opposite reason: on CLX the
   fusion was redundant (shuffles were free); on ICX the whole z+y pipeline
   is latency-bound so filling p0 during shuffle bursts buys nothing. The
   f1 race arm keeps it priced per scored process at ~zero cost.
3. **pf=1 (cross-volume input prefetch): +3.3%** (6.601). Corpus §3's
   "software prefetch mostly loses" holds on bare metal too.
4. **XPF, PS=184, PIN=2: null/negative** (6.732 / 6.760 / inconclusive
   vs 6.715 control). PIN=0 costs 2.5% — constant pinning is still right.
5. **X2 x-pass pairing: wash** (6.667 vs 6.670 paired) — the w3 −0.5%
   reading did not reproduce; left off.

### Borrowed this round (attribution)

* **Corpus §10 (Ice Lake Under Glass)**: the p0+p5 FMA / p5-only shuffle
  port model that motivated both surgeries; the warning that its VM
  load-collapse numbers do NOT transfer to this bare-metal node (they
  don't — the masked-broadcast form that §1's uop-cap model favors lost
  here for latency reasons the VM tier couldn't even measure).
* **mt_r4 (my own multicore fork)**: the pw-gate-to-L3 shape.
* **L6_pfa / panel_r11**: the incumbency rule + two-block race hysteresis,
  which this round's coloring-lottery findings vindicate — three of my six
  windows would have adopted noise under a single-block race.

### For the monitor / next round

1. The graded L=13 cell is **latency/L3-bound, not port-bound**: z+y runs
   ~11k cycles against a ~7k port floor, x-pass 4.5k against ~2.5k, plus
   ~1.0 µs/xform of driver-side unitary scaling inside every timed unit.
   Port-level work (shuffle removal, fusion) is exhausted — measured wash
   or loss. The ±5% process-to-process spread (page coloring of the
   driver's three 1.1 MB chain buffers vs a 1.25 MB L2) is now the
   biggest term above the floor; if the harness ever allows it, an
   interleaved/hugepage allocation policy for the chain buffers would
   stabilize every entry's score, not just mine.
2. If a rewrite is ever funded here: the only untried structural idea is
   an L2-conscious reordering of the per-volume pipeline (e.g. two
   half-volume x/z/y sweeps) to shrink the instantaneous hot set below
   one L2 way-group and dodge the coloring lottery; expected value
   uncertain, cost high.
3. Expected node standings: graded 6.4–6.9 µs/xform depending on the
   window (MKL 6.39 ± 0.05); B=1 ~5.37, ahead of MKL's 5.73.

---

## Round ice_r2 (2026-08-22)

### The verdict that drove the round

ice_r1 leaderboard: L13_direct 4.661 µs/xform, this entry 6.131 (behind
mkl2026's 6.066). Their in-plan instrument read 3947 ns/vol against my 5455
on the same silicon — a 28% structural gap, not a knob. Their pipeline works
in the driver's interleaved layout (lanes = 4 whole lines per zmm, real
coefficients as broadcasts, −i folded into lane-alternating sine splats),
so it has NO split-re/im buffers, no de/re-interleave passes, and no
512-bit transpose networks; its only permutes are a per-v re/im swap and
128-bit tile transposes fused into stores. My pipeline paid for all four.
Per the round's cumulative mandate, this round REBUILT the entry inside
their pipeline and kept the one thing of mine that is strictly better: the
Rader/CRT arithmetic.

### What shipped (full rewrite of impl/L13_rader.c; ice_r1 code preserved in exemplars/ice_r1/)

**Adopted wholesale from L13_direct (geom panel_r6–r11 lineage, via the ice
exemplar), with attribution per item:** the lanes=lines pass structure
(X: in[x][p] lanes over 169 contiguous p → t1; Y: t1 plane lanes over z →
pb[z][ky] transposing store; Z: pb lanes over ky → out transposing store);
X-first everywhere (their r10 retest); the 128-bit TTILE/TILE4_/LASTCOL_
store bodies (their r7); ZSOLID Y groups + xmm Z/X tails (their r11
junction hygiene, including the "never put a narrow-store tail where the
next pass wide-loads it" lesson — not re-learned here); the t1 odd-line
pad rule (their r8 ← L23_rader r6); pb rows padded to 32 doubles; the
streaming prefetch schedule and its L3 gate (their r8, kill knobs
-DL13R_PW/-DL13R_PFIN); the all-pinned-constant discipline ("below ~15
distinct constants, pin everything", their r6); the lane-alternating
sign-fold of the sine tables (their r9 MULI fold); template
self-#include at WC=4/2/1. Retained from my own lineage: the in-plan
timed race with L6_pfa's >3%-in-both-blocks incumbency rule, and the
Rader kernel math below.

**The kernel (mine): Rader-13 split cyclic/negacyclic on interleaved
lanes, 93 vector FP per chunk vs the dense conj-folded matvec's 102
(−8.8%).** Fold pairs (g^t, 13−g^t), g=2: u_t, v_t; cosine side = x0-seeded
cyclic-3 on P_t=u_t+u_{t+3} plus negacyclic-3 on Q_t=u_t−u_{t+3}, halved
constants CP/CM (6 broadcasts); sine side = dense negacyclic-6 on
w_t = SWAPRI(v_t) with lane-alternating splats S_t = (+sin(2πg^t/13), −sin…)
— the i·SS mix that cost my split-format kernel its whole layout is now a
free swap plus sign-in-table, exactly the generalization of their r9 fold
to Rader order. Outputs X[g^n] = cc_n ± T_n land in tile order via the
Rader permutation baked into the store macros. 12 constants pinned per
execute, 13 line loads, zero constant loads per chunk. Census unchanged
from their r11 default: 133 zmm + 14 xmm chunks/volume → 93·133+~46·14 ≈
13.0k op-equivalents, ~6.5k-cycle port floor on this node's two FMA pipes.
Asm audit (their L45_pfa r7 habit): default exec 852 insns total vs their
1098, ZERO zmm spills; only the xmm tail's unpinned constants touch the
stack (their design, ≤12 reloads per tail chunk).

### What was measured (node a80n0, tryout.sh, graded chain m=1278; three
process windows, cross-window drift is the documented ±5–10% lottery)

| case | this round | MKL same window | ice_r1 self |
|---|---|---|---|
| graded B=32 | **4.801 / 4.829 / 4.929 µs/xform** (min, three windows; sd ≤0.5% within run) | 6.338 / 6.349 / 6.472 | 6.131 (scored) |
| B=1 m=1278 | **4.257 / 4.301** | 6.518 / 6.569 | 5.369 |
| B=512 (36 MB > L3, pf tier) | **7.193** | 8.651 | — |
| B=4, forced -mno-avx512f (AVX2 path) | 5.800 (PASS, unscored) | 6.665 | — |

In-plan race (full scored batch, ns/vol): B=32 windows read
zs:4368/4504/4967 vs my ice_r1 incumbent's 5455 — a 15–20% in-plan gain,
plus everything the old pipeline paid outside the race. B=1: zs:3690/3790.
Correctness: rel_l2 = 2.95–3.0e-16 single-transform, 5.73–5.76e-14
whole-chain (tol 3.6e-11), bit-identical outputs across re-runs at every
batch tried, -Wall -Wextra clean on both ISAs.

Race arms this round (all output-bit-identical, so adoption can never
change results; xl priced but never adoptable):
* **zs** (default, their r11 shape): incumbent, kept every window.
* **y2** (their r10 ymm tails): 4406/4926/4516 vs zs 4368/4967/4504 —
  statistical tie, exactly their ice_r1 ab pattern (3989 vs 3947).
* **p7** (pure zmm, zsolid everywhere): +3–15% every window. Even with two
  FMA pipes the mixed tail earns its keep through access hygiene, not ports.
* **xl** (X-last): +30–38% every window. X-first is settled on ICX too.

### What was tried and did NOT work, with the number that killed it

1. **Row-padded t1 (the "t1" arm; L13_direct r11's sketched-but-unbuilt
   fix, built here): +6–14%.** t1 rows 26→32 doubles, X pass re-chunked
   per y-row zsolid (z0=0,4,8,9; 52 zmm chunks vs 42.5-equiv) so every
   Y-pass load is address-exact against one X-pass store: deletes ~480
   3/4-split zmm loads + the straddle SF-blocks per volume. Node: t1
   5279 vs zs 4967, 4840 vs 4504 (B=32), 4308 vs 3790 (B=1) — the +10
   zmm chunks of X-pass port time cost MORE than the split/SF deletion
   recovers, in every window, both batches. Their r11 gate ("build only
   if within ~0.4 µs of floor") was right and I built it anyway; now it
   is priced. Kept as FORCE=7/8 and a race arm — if a future round makes
   the kernel cheaper, the tradeoff shifts toward it.
2. **Everything my ice_r1 pipeline was**: split re/im A-volume, TR8/
   extract-form transposes, the fused zy schedule, MB loads, pw/pf knob
   races — all superseded by the structure change, not individually
   disproven. The old file with its full knob set is the ice_r1 exemplar.

### For the monitor / next round

1. Expect the scored cell at **~4.8–4.95 µs/xform** (window lottery), vs
   MKL ~6.3–6.5 and L13_direct's r1 4.661. This entry is now at parity
   with L13_direct's r1 form on the same pipeline with 9% fewer FP ops;
   the two entries differ only in kernel arithmetic, so any remaining gap
   between us is window noise or their r2 improvements (their working
   tree shows a t1b double-buffer mid-build — price whatever it turns out
   to be next round).
2. B=1 in-plan is 3690–3790 ns vs a ~2250 ns port floor: the ~1.4 µs
   residue is junction latency (Z-pass out stores 3/4-split — driver
   layout, unfixable per-axis) and the t1→Y junction (the row-pad fix
   failed on port cost; the untried alternative is making the X pass
   store t1 transposed so Y reads contiguous rows — moves the strided
   access to the pass that has port slack, sketch only).
3. The Rader kernel's 93-op census could drop ~6 more ops by CRT-splitting
   the negacyclic-6 (x⁶+1 = (x²+1)(x⁴−x²+1)), but the chain is
   latency-bound and that split deepens the dependency tree — the same
   tradeoff L17_matrixsimd r2 measured at −12% ops → −1.4% time. Parked.
4. Race arms worth keeping: zs/y2 tie means y2 is live insurance if the
   xmm tail ever sours; t1/p7/xl are documentation at ~1.5 ms of setup.

---

## Round ice_r3 (reconstruction note)

The ice_r3 generation shipped code but never appended its section (rule
violation, noted so the gap is explicit rather than silent).  From the code,
its comments, and the r3 leaderboard: it adopted L13_direct ice_r2's OV
cross-volume X-pass overlap (43 X chunks of the next volume interleaved
3-4 per plane between the Y and Z groups, t1/t1b ping-pong) and rebuilt the
in-plan race chain-shaped (arms ov/zs/y2/t1, untimed unitary scale between
steps, ~120 ms clock-settle spin, adopt at >1.5% in both blocks).  Scored
4.619 us/xform, finally 0.001 ahead of L13_direct's 4.623 (mkl2026 6.043).

## Round ice_r4 (2026-08-22)

### The task changed: the graded step is now the full rival map

state <- (z + c) / (1 + |z + c|), z = FFT(state), raw (no unitary scale in
map mode -- verified in driver.c and check.py: the reference is
`z = fftn(state) + c; state = z/(1+|z|)`).  The driver's fallback for
entries without fft3d_chain is exec + a driver-side vectorized map pass;
MKL through that fallback reads 12.0-12.2 us/xform at the graded B=32
(from 6.2 FFT-only).  This entry now exports fft3d_chain and owns the
whole m=1278-step chain.

### What shipped

1. **fft3d_chain with the LAZY MAP fused into the X pass** (adopted: the
   rival pipelines' winning fusion, corpus 10 s2 + their sources
   `ext/reference/fft_v4_solutions/1760b1bf_score0.96/generator.py`
   pw_core/PW_STYLE and 1000f989's mapF).  Every element of the previous
   step's raw z is read exactly once, in the X pass -- so the map happens
   ON LOAD (chunk13rm: 13 mapped rows per chunk) and `state` is never
   materialized.  Raw z ping-pongs between an internal batch buffer zb and
   final_out, parity-chosen so step m lands in final_out; one in-place
   map_pass finishes the chain.  Step 1 is a plain exec of x0 (1/m weight).
2. **The map form (MAPSTYLE=1, default): rsqrt14-seeded 2-Newton |w| +
   ONE exact vdivpd per point** -- the rivals' PW_STYLE 2, at full double.
   Numerics, written down as the brief demands: seed 2^-14 -> Newton
   5.6e-9 -> 4.7e-17, below double rounding; the divide is exact; ~3 ulp
   per application.  Budget 1e-13/step, tol = 1.278e-10 at m=1278;
   measured whole-chain drift **1.19-1.39e-13 -- a ~1000x margin**.  The
   rivals' float-seed 2-Newton (~1e-12/application) would still pass at
   this m by their own drift curve, but the exact ladder costs nothing
   extra here (the divide dominates), so there is no reason to spend ANY
   budget.  sp==0 guarded by max(sp,1e-300) (rsqrt(0)=inf would NaN).
   xmm tail rows: exact sqrt+div (1/43 of points).
3. **Chain-arm race in create()** (replaces the ice_r3 unitary-scale race,
   whose regime no longer exists in the grading): genuine raw z (one exec
   of LCG noise) ping-pongs two private buffers under real step semantics
   with a synthetic 0.1-scaled c.  Arms fo (fused + ov overlap, incumbent)
   / fz (fused, monolithic X) / fs (phase-split, priced) / uf (in-place
   map_pass + plain exec -- prices fusion itself).  All arms
   OUTPUT-BIT-IDENTICAL (same pw per row; store/load round trips exact),
   so the per-process adaptive pick can never change output bits --
   verified CHAIN_REPEATABLE with the race on.  L6_pfa hysteresis kept
   (>1.5% in both blocks).  -DL13R_CFORCE=0..4 pins fo/fz/uf/fs/f2.
   fft3d_execute keeps the exemplar pipeline with a deterministic pick
   (ov, zs+pf past L3); the old exec race is deleted -- in map mode the
   only timed path is the chain.

### Measured (a80n0, tryout.sh; same-window MKL = the driver fallback)

| case | this round | MKL same window |
|---|---|---|
| graded B=32 m=1278 | **6.378 / 6.382 / 6.390 / 6.603** (four windows) | 11.99 / 11.99 / 11.99 / 12.17 |
| B=1 m=1278 | **6.684** (race picked uf) | 13.66 |
| B=512 (>L3) | 8.962 | 13.83 |
| B=4 forced AVX2 (unscored) | 12.833 PASS | -- |

rel_l2 single-transform 2.975e-16; whole-chain 1.19-1.39e-13 (tol 1.3e-10);
bit-identical .chain across repeat runs with the race live; -Wall -Wextra
clean both ISAs.  vs ice_r3's FFT-only 4.619: the full map step costs
+1.76 us fused; the rivals' full-task per-point at L=13 (0.164 s -> 4.01
us/xform) remains ahead -- their FFT itself is faster than ours.

At B=32 the race picks fo every window (fused wins by 5-10%); at B=1 it
picks uf (working set is L2-resident, the separate streaming map pass
beats the fused chunk's longer dependency chains).  The bit-identity
design is exactly what makes this adaptivity legal.

### What did NOT work, with the number that killed it

1. **MAPSTYLE=0, hw vsqrtpd + rcp14 2-Newton reciprocal: 7.498 vs 6.390.**
   vsqrtpd's divider occupancy is ~2x vdivpd's on this core; with 546
   divider ops/volume the map is divider-throughput-bound and the sqrt
   form pays double.  (First build shipped this; the style sweep fixed it.)
2. **MAPSTYLE=2, all-FMA (rsqrt14 + rcp14 ladders, zero divider): 6.747,
   and the race flipped to uf (7489 vs fo 7747).**  Inside the fused chunk
   the extra ~6 FMA/row saturates the FMA ports the kernel needs; the
   divider was no longer the binding port after MAPSTYLE=1.
3. **MAPSTYLE=3, Montgomery batch inversion of each fold block's four
   1+|w| (one vdivpd per 4 rows, corpus 10's "wash once the divider is
   hidden" retested because the divider was NOT hidden): 6.711 vs 6.390.**
   Cutting 13 divs/chunk to 4 did not pay: after MAPSTYLE=1 the chunk is
   no longer divider-bound, and the 12 extra live temps per block cost
   more than the divider relief buys.  Also breaks bit-identity with
   uf/fs (grouped rounding), so adoption is restricted under it --
   documented in the race code.
4. **fs, phase-split mapped chunk (rivals' phase-split codelet pattern:
   pw rows -> 13x64B L1 bounce -> plain chunk13r): 8014 vs fo 7204 in-race
   (+11%).**  The bounce round trip serializes phase A into phase B
   consumer-adjacent; no register relief was needed (asm audit: ZERO zmm
   spills in the monolithic mapped chunk -- the 59 rsp accesses are the
   known xmm-tail constant reloads).  What won 1.6x on their 23-point
   kernel loses at this junction.  Kept as a race lane for documentation.
5. **f2, twice-finer ov interleave (X chunks after Y AND after Z groups):
   7.060 vs 6.382 forced-fo, same window (+10.6%).**  The slot after the
   Z group breaks the out-store pipeline.  FORCE=4 only.

### Port accounting (why it stands where it stands)

Per mapped zmm X chunk: kernel 93 FP (~47c on two pipes) + ladder 13x9 FP
(~58c) + 13 vdivpd (~104 divider cycles) -- FMA and divider are BALANCED
at ~105c/chunk, so the mapped X pass has a ~4.4k cyc/volume floor vs the
plain pass's ~2.0k.  Measured map cost is ~5.1k cyc/volume: the ov
interleave hides only a little of the divider under plane-phase FMA
(fo ~ fz every window).  The step remains latency-bound (18.5k cyc
measured vs ~10k port floor) -- the map work adds to, rather than fills,
the FFT's own junction-latency bubbles, and window-size limits (ROB ~1.5
chunks) are the plausible reason finer interleaves failed.

### Borrowed this round (attribution)

* **Rival pipelines via corpus 10 s2 + 1760b1bf/1000f989 sources**: the
  lazy map (fuse into the next step's first contiguous pass), one divider
  op per point with Newton on the FMA pipes, the PW_STYLE 2 form, the
  sp==0 guard, and the tiered-precision WARNING that our gate makes their
  fast path a trap -- answered here with a full-double ladder at the same
  divider cost.
* **Corpus 10 s2 Montgomery note**: retested because its rejection
  premise ("divider hidden") did not hold here; rejection reproduced for
  a different reason (register pressure).  No round should retest it a
  third time.
* **L6_pfa / lineage**: race incumbency hysteresis, kept.

### Infrastructure notes for the monitor

* **tryout.sh is broken for every chain case** since the task change:
  line 36 references $W before line 39 defines it (`set -u` aborts).
  Workaround used here: `W=$PWD/build/tryout/<name> ./tryout.sh ...`.
  Even then, the map-check line expands `'$W/c.bin'` in the REMOTE shell
  where W is unset -> check.py crashes -> the repeatability cmp never
  runs.  I ran check.py and the repeat-cmp manually (results above).
  Other implementers may silently be flying without the chain gate.
* **ice_r3 has no strategy section from its author** -- reconstructed
  above from code and leaderboard.

### For the next round

1. The map is now ~fully priced at +1.76 us; the headroom is back in the
   FFT's latency bubbles (13.3k cyc vs 6.5k floor FFT-only).  The two
   sketches that survive: (a) X pass storing t1 TRANSPOSED so the Y pass
   reads contiguous rows (moves strided access to the pass with port
   slack -- ice_r2 note, still unbuilt); (b) staging the next volume's
   mapped rows during plane phases at ROW granularity (43 rows/plane,
   register-light) into a t1-shaped buffer, decoupled a full volume ahead
   -- unlike fs, consumer distance would let OoO overlap; costs 35KB/vol
   extra L2 round trip.
2. Cross-STEP ov (next step's volume-0 X pass into this step's last
   planes) caps at ~1%/step -- not worth the buffer hazard logic.
3. If a future round needs budget: the third Newton (-DL13R_PW3) and the
   exact-div fallback (-DL13R_MAPRCP=0) are in place and cost ~nothing /
   ~1.1 us respectively; there is NO reason to ever go float-seed here.

---

## Round ice_r5 (2026-08-23)

### The verdict that drove the round

ice_r4 leaderboard: L13_direct 5.837 us/xform, this entry 6.363 (MKL through
the fallback 11.8-12.0).  Same map semantics, same correctness tier -- the
0.53 us gap was entirely CHAIN MACHINERY: their fch-ab read ~6.09 us/step
in-plan against my fo's 6.37, and their record itemizes why (paired map,
staged pair units, cross-step ov, one in-place buffer).  Their "for next
round" #1 predicted this adoption explicitly.  Under the cumulative mandate
this round REBUILT my fft3d_chain on their machinery wholesale and kept the
one thing of mine that is strictly better: the 93-FP/chunk Rader kernel
(theirs: 102).

### What shipped (attribution: L13_direct ice_r4 for all four pieces)

1. **Paired lazy map** (their `l13_mappair` <- 1760b1bf `pw_pair_gen`,
   full-precision seeds): two ADJACENT zmm (8 points) merge |w|^2 into one
   vector via 2x vpermutex2var; ONE vsqrtpd + ONE rcp14+2-Newton ladder
   serve 8 DISTINCT magnitudes; 2x vpermutexvar expand the scale back.
   ~17 port-uops + 1 divider op per 8 points, vs my r4 per-row form's ~20
   uops + 2 vdivpd.  Map flavor = their swept winner v2 (hw sqrt + rcp14
   ladder; their sweep 6.14 vs 6.25/6.39/6.92) -- adopted WITH the sweep,
   not re-run.  My r4 MAPSTYLE knob and the Montgomery MROW4 are deleted
   (priced in the r4 record).  ONE per-point DAG now serves every call
   site (pair, unpaired zmm, xmm tail, standalone pass), so all chain arms
   stay output-bit-identical; the MAPRCP=0 fallback changed from w/a to
   w*(1/a) to preserve that under the knob too.
2. **Two-phase pair units + depth-1 software pipeline**: 21 pair units per
   volume (169 = 21*8 + 1 inline-mapped w1 tail); phase 1 maps 13
   row-pairs into a 1.7 KB rotating L1 stage, phase 2 runs the UNCHANGED
   spill-free chunk13r_w4 on the stage at rs=16 (64 B stores -> exact 64 B
   loads); unit u+1's whole map phase issues between unit u's stage stores
   and its FFT chunks.  This retroactively explains my r4 fs failure
   (+11%): the bounce split only pays with the pipeline overlap and the
   8-wide pairing -- my fs had neither.  Their unpipelined-form price
   (+0.15 us) accepted from their record, not re-measured.
3. **Cross-step ov**: fft3d_chain owns all m steps, so the overlap X
   actions at the last volume of step s come from step s+1's volume 0 --
   one X prologue for the WHOLE chain, no per-step drain.  23 dispatchable
   actions per volume, paced floor((x+1)*23/13).  nb==1 stays per-step
   (cz): the only next volume is the one being written.
4. **In place, one buffer**: the fused chain runs entirely in the driver's
   final_out (each volume's X pass fully consumes it before its plane
   phase overwrites it).  Working set 2.15 MiB (state+c) vs my r4
   ping-pong's 3.2; zb survives only for the uf arm (exec's restrict
   contract forbids in==out).
5. **Race rebuilt**: arms cv (cross-step ov, incumbent at nb>=2) / cz
   (per-step paired X) / uf (map pass + exec ping-pong, prices fusion).
   Same LCG arena + clock-settle spin + L6_pfa >1.5%-in-both-blocks
   hysteresis as r4; 6 steps/rep (cv re-runs its 1-volume prologue each
   rep, ~+0.2% bias against it -- same bias their fch-ab carries).
   r4's fo/fz/fs/f2 chainstep bodies are DELETED (superseded, all priced
   in the r4 section).  -DL13R_CFORCE=0/1/2 pins cv/cz/uf.

### Operation count

FFT unchanged: 93 vector FP per zmm chunk, 133 zmm + 14 xmm chunks/volume;
the X-pass zmm chunks now read the L1 stage instead of raw z.  Map per
volume: 273 pair maps (~4.6k port-uops + 273 zmm sqrts) + 13 inline w1
tails.  Divider pressure roughly HALVED vs r4 (273 vsqrtpd vs 546 vdivpd
equivalents).  Measured map cost on top of the r3 FFT-only chain (4.62):
~1.2 us/xform, vs r4's +1.76.

### Measured on the NODE (a80n0, tryout.sh, graded m=1278; MKL = driver
fallback, same window; tryout's $W/map-check bug STILL present in r5 --
worked around as in r4, check.py + cmp run manually)

| case | this round | r4 self | window MKL |
|---|---|---|---|
| graded B=32 (3 windows) | **6.144 / 5.795 / 5.806** | 6.378-6.603 | 11.99-12.08 |
| B=1 m=1278 | **5.369** (race: cz 5394, uf 7379) | 6.684 (uf) | 11.98 |
| B=512 (>L3) | 7.840 | 8.962 | 13.88 |
| B=4 forced AVX2 (unscored) | 11.253 PASS | 12.833 | -- |

In-plan fch-ab (ns/vol-step, per window): cv:7448/cz:7449/uf:9746;
cv:6605/cz:6637/uf:9040; cv:6190/cz:6185/uf:8678; cv:5790/cz:5818/uf:7988.
cv and cz are a statistical tie every window (the race + hysteresis keeps
cv); uf loses 27-38% EVERYWHERE now, including B=1 -- r4's "uf wins at B=1"
is dead: that pick was an artifact of the monolithic inline map's dep
chains, which the staged pipeline removes.

Correctness: single-transform rel_l2 = 2.975e-16 (B=32) / 2.947e-16 (B=1);
whole-chain 1.308e-13 (B=32) / 7.647e-14 (B=1) vs tol 1.278e-10 (~1000x
margin -- the full-double tier at their fast tier's structure).  Chain
output BIT-IDENTICAL across processes and across builds with the race
live; -Wall -Wextra clean on both ISAs, all knob builds compile.

### What did NOT work / was deliberately not re-run

1. Nothing measured negative this round: every adopted piece arrived with
   the donor's own A/B numbers (their MAPV sweep, their +0.15 us
   unpipelined price, their in-place/xstep hazard analysis) and landed as
   predicted.  The round's risk -- their 102-FP chunk13p swapped for my
   93-FP chunk13r inside the pair unit -- needed no tuning.
2. NOT re-swept, per the no-rediscovery rule: MAPV flavors (their sweep
   stands), CPF c-stream prefetch (4x confirmed negative panel-wide),
   finer/coarser ov interleave (their 23-action pacing kept).
3. r4 arms fo/fz/fs/f2, MAPSTYLE 0-3, Montgomery batch inversion: deleted,
   not disproven again -- see the r4 section for every number.

### Borrowed this round (attribution)

* **L13_direct ice_r4** (the whole round): paired lazy map (via 1760b1bf's
  pw_pair_gen), v2 flavor + its sweep, two-phase pair units, depth-1
  software pipeline, cross-step ov, in-place single-buffer chain, the
  fch-ab race shape.  Retained of my own: the Rader-13 CRT kernel, the
  LCG chain arena + L6_pfa hysteresis, the uf pricing arm.

### For the monitor / next round

1. Expect the scored cell at **~5.8-6.15** (window lottery; MKL fallback
   ~12.0).  L13_direct's r4 record says they expect to converge with me
   near ~5.5 once kernels/machinery cross-pollinate both ways; if their
   r5 adopts my 93-FP kernel, the entries differ only by kernel arithmetic
   again and this entry should hold a ~2-4% edge at equal machinery.
2. The remaining headroom is the FFT's latency bubbles (~13.3k cyc/vol
   FFT side vs ~6.5k port floor), not the map (~1.2 us, near their
   measured floor).  Two structural sketches, still unbuilt: (a) permuted
   intermediate state layout so the X pass reads z contiguously instead
   of at 2704 B stride, un-permuted for free in the final map pass
   (L13_direct r4 "for next round" #2 -- needs a second Z-store body);
   (b) X pass storing t1 transposed so Y reads contiguous rows (my r2
   sketch).  (a) helps both entries' chains; (b) helps all regimes.
3. tryout.sh: BOTH $W bugs from my r4 infra note are still present
   (set -u abort without a preset W; map-check --cin expands to /c.bin
   remotely, silently skipping the chain gate AND the repeatability cmp
   for anyone not working around it).  Third round running; worth a
   central fix before someone ships an entry that never saw the gate.
