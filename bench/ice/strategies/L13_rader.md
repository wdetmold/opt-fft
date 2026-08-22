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
