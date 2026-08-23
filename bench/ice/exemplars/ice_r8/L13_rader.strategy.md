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

---

## Round ice_r6 (2026-08-23)

### The verdict that drove the round

ice_r5 leaderboard: L13_direct 5.401, this entry 5.808 (MKL fallback 11.8).
The 0.41 us gap is ONE structural item, plainly documented in their r5
record: their chain went VOLUME-GROUP-MAJOR (vm2) while mine stayed
step-major.  Step-major (`for s { for b }`) re-streams state + c (~2.15 MiB)
through L3 on every one of the m=1278 steps -- the exact mechanism behind
every "latency/L3-bound" diagnosis of this cell since ice_r1.  The
per-volume chains are independent, so running ALL m steps over one PAIR of
volumes (state 68.6K + c 68.6K + t1/t1b 71.6K + pb/sg ~ 215 KB,
L2-RESIDENT against the node's 1.25 MB L2) before moving to the next pair
turns every X-pass load and Z-store RFO from an L3 round trip into an L2
hit.  Their record explicitly offered the adoption; taken wholesale.

### What shipped (attribution: L13_direct ice_r5 vm2 <- L17_matrixsimd
### ice_r4 volmajor-inplace)

1. **Volume-group-major fused chain, G=2 default**: a thin wrapper in
   fft3d_chain partitions the batch into groups of G and, per group, runs
   exec (step 1, raw z_1), the UNCHANGED r5 chain body (ov for g>=2, zs for
   the odd tail) for steps 2..m, then the in-place final map -- so x0, c,
   and the final state of a group each cross L3 exactly once for the whole
   chain.  G=2 rather than 1 keeps the r5 cross-step ov pipeline alive
   INSIDE the group (their observation; their v1-ties-v2 result softens it
   to insurance).  Grouping only reorders whole volume-steps across
   independent volumes => bit-identical output for every G, which is what
   makes G legal to race: chain drift digits are IDENTICAL to r5
   (1.308e-13 B=32, 7.647e-14 B=1) and out/chain cmp across runs is clean.
2. **Race rebuilt to 5 arms**: cv/cz (step-major, the r5 pair) + v1/v2/v4
   (volume-major at G=1/2/4), incumbent v2 at nb>=2, same LCG arena +
   clock-settle + >1.5%-in-both-blocks hysteresis.  uf is DROPPED from the
   race -- it lost 27-38% in every r5 window (priced twice now); still
   reachable via -DL13R_CFORCE=2.  Knob map now CFORCE=0..5 =
   cv/cz/uf/v1/v2/v4.

### Operation count

Unchanged from r5 (93 FP/zmm chunk, 133 zmm + 14 xmm chunks/vol; map 273
pair units + 13 w1 tails per vol).  vm adds ZERO FP work and zero buffers;
what it deletes per step per volume is ~68 KB of L3 reads + ~34 KB of L3
RFOs, replaced by L2 hits.

### Measured on the NODE (a80n0, tryout.sh with the W= workaround; the $W
### bug is STILL live, chain gate + repeat-cmp run manually as in r4/r5)

| case | this round | r5 self | window MKL fallback |
|---|---|---|---|
| graded B=32 (2 windows) | **5.393 / 5.381** (sd 0.02-0.05%) | 5.795-6.144 | 12.07 / 12.25 |
| B=1 m=1278 | 6.125 (slow window: MKL read 13.64 vs its usual ~12.0; the B=1 path -- cz, cg=0 -- is byte-identical to r5, which measured 5.369) | 5.369 | 13.64 |
| B=512 (>L3 tier) | **5.392** (-31%: each G=2 group is L2-resident regardless of total batch, so the tier penalty is GONE) | 7.840 | 13.67 |
| B=4 forced AVX2 (unscored) | 11.253 PASS (identical to r5 -- path untouched) | 11.253 | -- |

fch-ab (ns/vol-step, tb=32, three windows): v2 6346/6463/6215 vs cz
7169/7826/6547 and cv 7207/7843/6522 -- vm is -11 to -17% in-plan, v2
picked every window; v1/v2/v4 tie within 0.5% (matches their sweep --
residency is flat below L2, overlap is now insurance).  Correctness:
single-transform rel_l2 2.975e-16 (B=32) / 2.947e-16 (B=1); whole-chain
1.308e-13 (B=32) / 7.647e-14 (B=1) vs tol 1.278e-10 (~1000x margin, digits
identical to r5); repeatable out + chain cmp clean; -Wall -Wextra clean on
both ISAs; CFORCE 0-5 all compile.

### What did NOT work / was rejected by arithmetic before building

1. **Post-Z map placement (L17_matrixsimd's s6; L13_direct r5 flagged it
   as the next A/B now that the chain is L2-resident): rejected ON PAPER
   for THIS pipeline, not built.**  Mapping at the Z store would delete the
   X stage round trip (worth ~0.15 us by L13_direct's own unpipelined
   price) but the lanes=lines Z store is 128-bit TILE-TRANSPOSED: the
   m=12 LASTCOL column (4 xmm extract-stores per zmm chunk) and the xmm
   Z-tail chunk (169 points/vol) cannot join 8-point pairs, so the map
   would run 13x3x4 + 169 = ~325 xmm sqrts/vol (plus 234 paired zmm) vs
   the staged X scheme's 273 zmm + 13 xmm -- roughly +0.8k divider
   cyc/vol, i.e. ~2x the saving, before counting spill risk inside the
   tile store body (the r4 inline-map lesson).  The technique transfers
   only to pipelines whose Z store is not tile-transposed; recorded so
   nobody rebuilds it here.
2. **uf** dropped from the race on r5's numbers (27-38% loss everywhere),
   not re-measured.
3. NOT re-swept per the no-rediscovery rule: MAPV/MAPRCP flavor
   (L13_direct re-swept it in the L2 regime this round's donor record:
   wash), larger G (their v4-ties-v2 + mine), CPF c prefetch.

### Borrowed this round (attribution)

* **L13_direct ice_r5**: the volume-group-major chain (vm2) wholesale --
  the G=2 wrapper shape, the v1/v2/v4 race arms, the incumbent choice, and
  their G sweep result (v1~v2~v4) adopted and re-confirmed rather than
  re-derived.  Their record credits L17_matrixsimd ice_r4's
  volmajor-inplace as the origin; noted through.
* **L17_matrixsimd ice_r4** (indirectly): the s6 post-Z map idea was
  evaluated (and rejected by op-count) rather than ignored -- see above.

### For the monitor / next round

1. Expect the scored cell at **~5.38-5.40** (two windows agree within
   0.2%; the old +-5% window lottery has visibly narrowed now that the
   chain no longer touches L3 -- page coloring of the big buffers stopped
   mattering).  That is parity with L13_direct's r5 5.401; if their r6
   finds another structural item, adopt it next round as usual.
2. Remaining headroom: the FFT's junction-latency bubbles inside an
   L2-resident step (~13k cyc/vol against a ~6.5k port floor).  The one
   untried big item is deeper fusion -- the X pass feeding the plane phase
   without the full t1 round trip (per-plane consumption order makes this
   a real rewrite).  The permuted-state-layout sketch from r4 is now
   low-value (X reads are L2 hits).  Kernel op reduction (negacyclic-6
   CRT split, -6 ops) is still parked: the regime is latency-bound at the
   junctions, not FMA-bound.
3. B=1 should score ~5.37 (path unchanged from r5); this round's 6.125
   reading is a slow-window artifact (MKL moved +14% in the same window).
4. tryout.sh $W bugs: FOURTH round running (set -u abort without preset W;
   remote map-check --cin '/c.bin' silently skips the chain gate and the
   repeat cmp).  Every number above used the documented workaround.

---

## Round ice_r7 (2026-08-23)

### The verdict that drove the round

ice_r6 closed the series with L=13 as the panel's ONE real gap: L13_direct
5.287, this entry 5.377, the rivals' mark 4.01 (and their re-benchmarked
sources on OUR node: 1000f989 0.1587 s = 3.879 us/xform gate-true,
v5_3907583b 0.160, v6_4d0483ea 0.164).  The r7 mandate said mine the
reconstructed rival sources.  Reading them settled the diagnosis: every fast
rival L=13 -- v4's interleaved 1.00-scorer AND the v5/v6 Hartley entries --
shares ONE structure neither of our entries had: **SoA BATCH-LANE layout**
(8 volumes per zmm, split re/im, "batch-lane groups: zero shuffles in the
transform" in their own words).  Our lanes=lines pipeline pays the tile-
transposed Z stores, the t1/pb junctions and their split-load/store-forward
hygiene at every pass of every step; the SoA form simply has none of that.
fft3d_chain owns all m steps, so the internal layout is FREE: convert in
and out ONCE per group (1/m amortized).  This round rebuilt the fused chain
on that structure.

### What shipped (impl rewrite of the chain path only; exec + classic chain
### untouched, still serve B<8, tails, and the single-transform gate)

1. **SoA-8 group chain** (`l13r_chain_soa`, batch>=8, groups of 8 volumes):
   x0 and c transposed once per group into split re/im SoA arrays
   (re[p][lane], p = x*169+y*13+z; 24-shuffle 8x8 unpck/shuff64x2 networks,
   self-inverse, 4 points/iteration + scalar tail point; 4 x 137 KB + skews
   = 564 KB, L2-resident).  Every pencil of every axis is then PURE
   VERTICAL SIMD at es = 8 / 104 / 1352 doubles.
2. **Two sweeps per step** (was 3 passes + junctions): sweep A per x-plane
   (169 contiguous points, 43 KB with re+im -- L1-RESIDENT through
   everything that touches it): per-plane streaming MAP pass, 13 z-pencils,
   13 y-pencils; sweep B: 169 x-pencils, plain.  Buffer holds raw z between
   steps (lazy map, step 1 skips it; one closing map pass).
3. **The pencil kernel is MY OWN Rader/CRT arithmetic re-derived for split
   re/im** (`l13r_dft13s_r`, -DL13R_SK=1 default): the r2 kernel's math
   verbatim, but SWAPRI becomes a register rename (use vi where re is
   expected) and the lane-alternating sine fold becomes plain constants.
   186 vector FP / 8 pencils, 12 broadcast constants, 26 data loads; zero
   shuffles.  The v6 generator's Hartley-split reg6 (`H13=reg6`, the thing
   the mandate said to beat) is transcribed as SK=0: 206 FP, 72 broadcast
   constants.  Node A/B, identical machinery: **rader-split 4.183 vs
   hartley 5.457 (-23%)** -- the mandate's "beat, not just match" is a
   measured number, and the margin is exactly what the kernels' load-side
   predicts (26+12 loads vs 26+144 embedded-broadcast loads per call).
4. **Zero-divider map** (-DL13R_SQR=1 default, the round's second-biggest
   single win): in SoA the map needs no pair merge/expand permutes at all
   (lanes are already 8 distinct magnitudes) -- |w|^2 = fma(wr,wr,wi*wi);
   and with |w| computed by rsqrt14 + 2 Newtons + Heron mul instead of
   vsqrtpd, the whole map runs on the FMA pipes.  Numerics: 2^-14 -> 5.6e-9
   -> 4.7e-17, below double rounding, sp=0 clamped by max(sp,1e-300); still
   the full-double tier, NOT the rivals' float-seed drift (whole-chain
   1.145e-13 vs tol 1.3e-10, ~1100x margin -- measured, digits below).
5. **Deterministic dispatch, not a race**: SoA arithmetic is not
   bit-identical to the classic pipeline's, so groups of 8 ALWAYS go SoA
   and tails/batch<8 ALWAYS go classic (r6 vm machinery, refactored into
   `l13r_chain_vm`, byte-identical behavior) -- same split in every
   process, so cross-process repeatability holds by construction (verified,
   see below).  The classic fch-ab race is SKIPPED when batch%8==0 (its
   pick would never run); B=1 keeps its r6 path and race untouched.
   Knobs: -DL13R_SOA=0 kills the path, SK/MF/SQR as above, CFORCE pins
   still force classic arms (SoA alloc suppressed under CFORCE).

### Operation count (per volume-step, SoA path)

FFT: 507 pencils/group x 186 FP / 8 vols = 11.8k vector FP (was 13.0k
op-equivalents in lanes=lines, PLUS all transposes/junctions -- now zero).
Map: 2197 point-vectors/group x ~16 FMA-tier ops / 8 = 4.4k FP, ZERO
divider ops, zero permutes (was 273 pair units with 4 permutes + 1 sqrt
each).  Conversions: ~22k uops per group per CHAIN (1/m ~ nothing).
Port floor ~8.1k cyc/vol-step; measured 12.2k (1.5x floor, was 1.9-2.0x).

### Measured on the NODE (a80n0, tryout.sh leased core, graded m=1278;
### the $W bug persists -- W preset, check.py + both cmps run by hand)

| config | us/xform | notes |
|---|---|---|
| **SHIPPED (soa8 + rader-split + MF=2 + SQR=1), B=32** | **3.688 / 3.692 / 3.694 / 3.696** (4 windows, sd 0.00-0.25%) | MKL fallback 11.99-12.00 -> **3.25x**; r6 self 5.377; L13_direct r6 5.287; best gate-true rival 3.879 |
| B=1 m=1278 (classic cz, path byte-identical to r6) | 6.214-6.224 | elevated window (MKL 12.9-13.6); r6 measured 5.37-6.18 window-dependent |
| B=512 (>L3 tier) | **3.695** | flat: every group is L2-resident regardless of batch |
| B=9 (tail split: 1 SoA group + 1 classic vol) | 4.406 | exercises the dispatch seam |
| B=4 forced -mno-avx512f (classic, unscored) | 11.270 PASS | |

Correctness: single-transform rel_l2 2.975e-16 (B=32) / 2.947e-16 (B=1) /
2.977e-16 (B=512) / 2.981e-16 (B=9); **whole-chain 1.145e-13 (B=32, tol
1.3e-10; drift digits identical across map flavors -- FFT reassociation
dominates, the map is ~exact), 2.495e-13 (B=9 mixed-path)**; out.bin AND
out.bin.chain cmp-identical across independent runs (SoA + tail dispatch);
-Wall -Wextra clean on icelake-server and x86-64-v3; all knob builds
compile (SK/MF/SQR/SOA/SBPF/MAPRCP/CFORCE both ISAs).

Window stability is a result in itself: sd <= 0.25% and a 0.008 us spread
across four windows -- the r1-r5 "page-coloring lottery" (+-5%) is gone
because the chain's big-buffer L3 traffic is gone.

### The A/B ladder that got there (all same-day node windows, B=32)

| step | us/xform |
|---|---|
| r6 shipped (lanes=lines, paired lazy map, vm2) | 5.377 (scored) |
| SoA-8 + rader-split, map fused in sweep-B stores, hw sqrt | 4.183 |
| ... + all-FMA sqrt (SQR=1) | 3.879 |
| ... + map per x-plane at sweep-A head (MF=2) | **3.692** |

### What did NOT work, with the number that killed it

1. **The v6 Hartley reg6 pencil (SK=0): 5.457 vs 4.183 (+30%).**  Its 72
   per-call embedded-broadcast constants add 144 load-port uops per pencil
   (vs my 12 hoistable constants), and 206 vs 186 FP.  The rivals' "decisive
   win over MKL" kernel is real but beatable by Rader arithmetic in the
   same layout.
2. **hw vdivpd map flavor (-DL13R_MAPRCP=0): 5.100 vs 4.183.**  Two divider
   ops per vector serialize sweep B: the +0.9 us is almost exactly 2197
   extra divider slots/group.  That observation is what motivated SQR=1
   (zero divider ops), worth -0.30 us: **at this cell the divider, not the
   FMA pipes, was the binding unit of the map** -- the r4-r6 "one divider
   op per point is fine" doctrine was true only while junction latency
   dominated.
3. **Map fused into sweep-B stores (MF=1) under SQR=1: 3.878 vs 3.799
   (MF=0) / 3.692 (MF=2).**  A mapped x-pencil is ~500 uops -- past the
   ~352-entry ROB, so ladders stop overlapping the next pencil (and its
   median was unstable, sd 6.8%).  MF=1 had won under SQR=0 (4.183 vs
   4.449) only because it hid divider latency; with the divider gone the
   staging lesson (r5: keep map latency OFF the FFT path, let independent
   ladders pipeline) wins again, now at plane granularity, L1-hot.
4. **Software prefetch of the next x-pencil's state+c lines in sweep B
   (-DL13R_SBPF=1): 4.401 vs 4.183 (+5%).**  Fifth confirmation of the
   panel-wide "software prefetch mostly loses" rule, now also for L2->L1.

### Borrowed this round (attribution)

* **1000f989 (v4 1.00-scorer)**: the SoA batch-lane layout itself ("8
  volumes per zmm slot, all axis passes pure vertical SIMD") and the
  two-sweep step shape; **v6_f40c5e25**: the Hartley reg6 pencil
  (transcribed as SK=0, beaten by SK=1) and the SoA-below-L36 selection;
  **v5_3907583b**: confirmation that Rader arithmetic composes with the
  batch-lane layout (their 13 is Rader->FFT-12 in the same layout).
* **L13_direct ice_r6**: the map@store idea (re-tested here as MF=1: wins
  under a divider-bound map, loses under the all-FMA one -- their r6
  verdict was regime-specific, recorded so neither entry re-flips it
  blindly), and the -ffp-contract bit-identity landmine (answered here by
  making SoA-vs-classic a deterministic dispatch instead of a race).
* **L13_direct ice_r5 / L17_matrixsimd ice_r4** (through my own r6): the
  volume-group-major residency principle, which the SoA chain inherits
  (all m steps per 564 KB group).
* My own r4/r5 negatives (inline-map-on-the-FFT-DAG, staged-pipeline)
  predicted the MF sweep's shape exactly; no rediscovery spent.

### For the monitor / next round

1. Expect the scored cell at **~3.69 us/xform** (four windows within
   0.008 us), 3.25x over the MKL fallback, **under the rivals' best
   gate-true 3.879 with an ~1100x chain-gate margin**.  B=1 ~5.4-6.2
   (unchanged r6 path, window-dependent).
2. The technique transfers: any size whose volume fits L1-ish per plane
   (L=17 certainly, the other trailing cells likely) can adopt the SoA-8
   batch-lane chain wholesale -- the win decomposes as ~-22% layout
   (junction/shuffle deletion), ~-7% zero-divider map, ~-5% plane-hot map
   placement, and the kernel A/B says bring your own arithmetic, not the
   rivals' Hartley.  L17_winograd/L17_matrixsimd: your cyclic/negacyclic
   kernels in split form should keep their op advantage over Hartley s44.
3. Remaining headroom here: measured 12.2k cyc/vol-step vs ~8.1k floor.
   Candidates: (a) B=1 is now the worst cell relative to its potential --
   a within-volume SoA (lanes = 8 x-pencils, transposes per pass) or ymm
   4-volume groups for 2<=B<8 tails; (b) sweep-B pencils are the only
   L2-latency-exposed work left -- blocking x-pencils by pairs of z to
   share slab lines, or a depth-1 plane pipeline in sweep A; (c) the
   negacyclic-6 CRT split (-6 FP) now that the cell is closer to
   port-bound.  All unbuilt, priced nowhere.
4. tryout.sh $W bug: FIFTH round (set -u abort without preset W; remote
   map-check --cin '/c.bin' skips the chain gate + repeat cmp).  Also new
   this round: reserve.sh --status false-negatives when slurm binaries are
   not on PATH (they live in /opt/software/slurm-19.05.8.1/bin on the
   login host) -- the heartbeat file is the truth; every number above ran
   under the live monitor hold via the documented workaround.

---

## Round ice_r8 (2026-08-23)

### The verdict that drove the round

ice_r7 leaderboard: L13_direct 3.471, this entry 3.688 (MKL fallback 11.8-12.0)
-- they retook the cell by 6% with the same SoA-8 idea but a different map
placement (their map@Ystore) and their pinned two-sweep pencil.  The r8
mandate says mine `fft_warm_solutions/` -- the warm cohort's 0.99-scorer
(warm_d43251c2, r ~ 0.145, essentially ON the grader-tier roofline) wrote its
own generator-emitted prime engine for 13/17/23, and its 13-path has EXACTLY
this entry's r7 structure (SoA-8 batch lanes, z+y pencils per x-plane, then
x-pencils) with two things mine lacked.  Both adopted this round:

1. **No map pass at all -- the map is SPLIT across the two sweeps** (their
   `dft13m` fuse_map + `dft13zm` map_in pair): the exact `+c` add is fused
   into sweep-B's x-pencil stores, reading c from a CONSUMPTION-ORDERED
   transposed copy (Ct row j of pencil q = c[j*169+q]; 13 contiguous rows
   per pencil, built once per group per chain = 1/m); the normalize ladder
   is fused into the NEXT step's z-pencil loads, where the plane is L1-hot
   and every element's first read happens.  The buffer holds z+c between
   steps; one norm-only pass closes the chain.  My r7 MF=2 paid a whole
   extra read+write of the plane (plus loop overhead) per step for its
   streaming map pass; MF=3 deletes that pass entirely.
2. **`optimize("schedule-insns,sched-pressure")` on the step bodies**
   (their generator compiles the 13 codelet -- and only the 13 codelet --
   with exactly this attribute).  Without it, gcc 11.4's default post-RA
   scheduler mangles the fused ~370-FP z-pencil body: MF=3 measured 3.947
   vs MF=2's 3.74 -- the fusion LOST until the attribute turned it into a
   6% win.  It also helps MF=2 itself (3.69 -> 3.63).  Bit-safe (verified:
   chain cmp identical with it on/off, both MF).

Kernel unchanged: my Rader-split 186-FP pencil (still beats their ~4h^2
symmetric-folded direct form on constants: 12 pinned splats vs their
per-block embedded broadcasts).  Layout, conversions, tails, B<8, exec:
all unchanged from r7.

### Ship config

`-DL13R_MF=3` (new default) + `-DL13R_OPTSCHED=1` (new knob, default on),
SK=1 SQR=1 MAPRCP=1 MX=0.  New plan buffers ctr/cti (+2 x 137 KB; the SoA
block is 6 x 137 KB now, but only state re/im + Ct are touched per step:
the hot set stays ~550 KB, L2-resident).

### The fp-contract forensics (worth every entry's attention)

MF=3 was designed to be bit-identical to MF=2 (the +c add moves across a
store/load roundtrip -- same operands, same bits; the norm ladder is the
same DAG).  It is NOT bit-identical under default flags, and the mechanism
is general: gcc's `_mm512_add/mul_pd` are GENERIC vector ops subject to
`-ffp-contract=fast`, and fusing the map into the pencil means the norm's
closing `w*s` mul now feeds the pencil's fold adds IN REGISTERS -- gcc
contracts `add(mul(w,s), ...)` into an FMA there, which is impossible in
MF=2 where a memory roundtrip separates them.  Verified: with
`-ffp-contract=off` the two chains are cmp-IDENTICAL; with default flags
they diverge 5.7e-14 at m=1278 (sub-anchor: two honest numpy reference
paths diverge 1.1e-13 on the same chain).  Kept the contraction (fewer
roundings, free speed); one genuinely shared mul+add site in the ladder
(`spc*rq + 1`) was spelled as an explicit fmadd so every call site rounds
it identically.  Lesson for the panel: **"bit-identical refactor" claims
across fusion-boundary changes must be verified under contract=fast, not
proven on paper** -- register-adjacency is a rounding-relevant property.

### Operation count (per volume-step, vs r7 MF=2)

FFT unchanged: 507 pencils x 186 FP / 8 vols = 11.8k vector FP.  Map
arithmetic unchanged (~15 FMA-tier ops/point-vector, zero divider, zero
permutes) but now rides existing pencil loads/stores: DELETED per volume =
the map pass's ~550 vector loads + ~550 vector stores + 275 c-vector loads
(the +c add's c loads move to sweep B where the ports have slack, from Ct
contiguous) + 169 loop iterations of overhead.  ADDED: 2 vector adds per
point at the sweep-B store (~550/vol, on ports that were not the
bottleneck).  Port floor ~8.0k cyc/vol-step; measured 3.46 us ~ 11.4k at
3.3 GHz = 1.42x floor (was 1.5x).

### Measured on the NODE (a80n0, tryout.sh; workarounds: W preset AND
### slurm bin dir on PATH for reserve.sh --status; check.py run by hand)

| config | us/xform (min) | notes |
|---|---|---|
| **SHIPPED (mf=3 + optsched), B=32** | **3.460 / 3.464 / 3.473 / 3.473 / 3.493 / 3.532** (6 windows) | r7 self 3.688 (scored); L13_direct r7 3.471; best gate-true rival (1000f989 re-bench) 3.879; MKL fallback 11.99-12.03 -> **3.46x** |
| MF=2 + optsched (r7 shape + sched) | 3.628 / 3.631 / 3.633 | the sched attr alone is worth ~1.5% on r7's shape |
| MF=3, no sched | 3.944 / 3.947 | the fusion LOSES without sched-pressure -- do not ship one without the other |
| MF=2, no sched (= r7 shipped) | 3.727-3.741 (quiet), 4.25 median noisy windows | r7's 3.688 was a good window |
| B=1 m=1278 (classic cz, untouched) | 6.212 | window-dependent 5.4-6.2 (MKL read 12.0 same window); path byte-identical to r6/r7 |
| B=9 (1 SoA group + 1 classic vol) | 3.680 | seam exercised |
| B=512 (>L3 tier) | 3.585 | flat, group-resident as designed |
| B=4 forced -mno-avx512f (unscored) | 12.926 PASS | classic path untouched; delta vs r7's 11.27 is window noise on unscored hw |

Correctness (default build, node): single-transform rel_l2 2.975e-16
(B=32) / 2.947e-16 (B=1) / 2.977e-16 (B=512) / 2.981e-16 (B=9);
whole-chain m=1278: **1.215e-13 (B=32, anchor 1.136e-13, tol 1e-10)**,
2.588e-13 (B=9 mixed-path), 7.647e-14 (B=1, digits identical to r7 --
classic untouched); **m=2 gate 9.571e-16 vs tol 3.0e-14** (the map is
~1-ulp exact-class: the r8 one-step contract is met with >10x margin, and
there is no fp32 anywhere).  out.bin AND out.bin.chain cmp-identical
across independent runs.  -Wall -Wextra clean on icelake-server and
x86-64-v3; all knob builds compile (MF=0/1/2/3, MX, OPTSCHED, SK, SQR,
MAPRCP, CFORCE, both ISAs).

### What did NOT work, with the number that killed it

1. **MF=3 without OPTSCHED: 3.947 vs 3.464 (+14%).**  gcc's default
   scheduling of the norm-fused z-pencil (~420 uops) serializes the
   ladders; `schedule-insns,sched-pressure` fixes it.  This is why my r7
   MF sweep concluded "keep map latency OFF the FFT path" -- that verdict
   was a COMPILER artifact, not a machine property.  Corrected this round.
2. **MX=1 (warm's maphw/map2 divider/FMA alternation): 4.071 unsched /
   4.091 sched, vs 3.46 (+18%).**  The hw sqrt+div norm on even rows
   serializes behind the divider exactly as SQR=0 did in r7; the warm
   generator itself sets `_MAPMIX_OFF = {13}` (it mixes only at 17/23) --
   their sweep and mine now agree from opposite directions.  Do not
   re-open at this size.
3. Explicit-fmadd spelling of `spc*rq+1` alone did NOT restore MF3/MF2
   bit-identity (divergence digits unchanged) -- the register-adjacency
   contraction above is the real mechanism; chased to ground with
   contract=off cmp instead of guessing further.

### Borrowed this round (attribution)

* **warm_d43251c2_score0.99** (`fft_warm_solutions/`, its
  `dev_generators/gen.py`): the split lazy map (+c at x-store from a
  consumption-ordered Ct, normalize on next z-load), the
  `optimize("schedule-insns,sched-pressure")` attribute on the 13-point
  step bodies, and the negative result that the divider/FMA map mix is
  wrong at L=13 (their `_MAPMIX_OFF={13}`).  Their engine README's
  "deferred into the next iteration's z-pass loads using a
  hardware-divide/Newton mix" is exactly items 1+2 minus the mix.
* **L13_direct ice_r7**: the standing map@store evidence that pushed me to
  re-test fusion despite my own r7 MF=1 negative; and their `--chain 1`
  driver-segfault note, reproduced below.
* Retained of my own: the Rader-split 186-FP pencil, SQR=1 all-FMA norm,
  SoA-8 machinery, classic tails.

### For the monitor / next round

1. Expect the scored cell at **~3.46-3.53 us/xform** (0.1415-0.1445 s):
   at or slightly under L13_direct's r7 3.471, ~11% under the best
   gate-true rival re-bench (3.879), 3.46x the MKL fallback.  B=1 ~5.4-6.2
   unchanged.
2. Remaining headroom (measured 1.42x the ~8k port floor): (a) cross-step
   sweep fusion (1000f989's L>=36 scheme B; L13_direct r7 scoped it too)
   is now the only untried structural item; (b) the negacyclic-6 CRT
   split (-6 FP/pencil) gets more attractive as the cell approaches
   port-bound; (c) B=1 SoA-within-volume (warm's `run_13t` shape) if B=1
   ever scores.  (d) If anyone wants the last 2%: OPTSCHED on the CLASSIC
   chain bodies was not tried (only the SoA step bodies carry it).
3. **NEW infra bug: check.py crashes on every m>2 map-check** -- the r8
   two-part-gate code at line 94 uses `math.floor` but never imports
   `math` (NameError), so tryout.sh's check line dies AFTER printing the
   single-transform PASS, and the repeat-cmp silently never runs (on top
   of the still-unfixed $W bug, rounds 4-8, which makes the remote
   --cin expand to '/c.bin').  Every gate number above came from running
   check.py by hand with `math` injected into builtins.  One-line fix:
   `import math` in check.py.
4. Driver `--chain 1 --map` still SEGFAULTS (pong buffer only allocated
   for chain > 1; L13_direct r7 diagnosed it).  The r8 brief's one-step
   gate is exactly `--map --chain 1`, so as things stand the gate cannot
   be run through this driver at any size -- m=2 (tol 3e-14) is the
   workable stand-in and passes here at 9.6e-16.  reserve.sh --status
   also still false-negatives unless the slurm bin dir is on PATH.
