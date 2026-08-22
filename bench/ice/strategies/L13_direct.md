# L13_direct — strategy record (ICE panel)

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (34.3 KiB),
forward, unnormalised, out-of-place, batched, single-threaded.
Implementation: `impl/L13_direct.c`. `fft3d_name()` → `L13_direct`.
Graded case (cases.txt): **B = 32, chain m = 1278, unitary** — 2.25 MiB
ping-pong working set, > the node's 1.25 MB L2, ≪ its L3.

This entry arrives at the ice panel as the geom-panel lineage at panel_r11
(conjugate-pair-folded dense 13×13 per axis, lanes = whole lines, all-pinned
register-resident matrix, zsolid-Y + xmm-tail mixed execs, X-first).  The
full history — including every dead end — is
`bench/geom/strategies/L13_direct.md` (rounds panel_r6 … panel_r11); the
multicore fork's record is `bench/mt/strategies/L13_direct.md`.  Do not
rediscover what those already killed.

---

## Round ice_r1 (2026-08-22)

(section under construction this round — measurements pending below)

(ice_r2 note: the ice_r1 agent never filled this in.  What is known of ice_r1
from the scored artifacts: the panel_r11 code was carried over unchanged
except that the ab instrument's xt slot was replaced by p7 and the ab result
gained a stderr print; the scored quiet-window number was **4.661 us/xform**
graded (B=32 m=1278), 1.30x ahead of mkl2026's 6.066, with
ab[B16]=y2 3989 / zs 3947 / p7 4077 / xl 4946 ns/vol — i.e. the r11 zsolid
default was confirmed best of the CLX-era shapes on ICX, ymm tails +1%,
pure-zmm +3%, X-last +25%.)

## Round ice_r2 (2026-08-22)

### Where it stood

ice_r1 leaderboard: 4.661 us/xform graded, first at L=13, 1.30x over
mkl2026 (6.066), 1.32x over L13_rader (6.131).  Baseline re-measured this
round via tryout: 4.744-4.785 across windows, MKL 6.24-6.32.  The margin is
comfortable but the brief flags L=13 as thin (0.76-0.87 class); the round's
job was to bank the transferable ICX lessons the other entries paid for in
ice_r1.

### What changed (in order of measured effect)

1. **OV — cross-volume X-pass overlap, the new cache-resident default**
   (adopted from **L17_rader ice_r1**, whose graded-cell winner "sp"
   established that on this latency/L3-bound chain, padding the plane
   phase's junction stalls with independent long-latency work wins where
   every port/width trick washes out; mechanism also consistent with
   **L13_rader ice_r1**'s "latency/L3-bound, not port-bound" diagnosis of
   this exact cell).  The chain regime's only L3-cold accesses are the X
   pass's reads of `in` (the previous step's scaled output, evicted from
   L2 by the intervening ~2 MB of traffic) and the Z stores' RFOs of
   `out`.  OV interleaves the NEXT volume's 43 X chunks into the current
   volume's L1-hot plane phase, 3-4 per plane (floor((x+1)*43/13)
   schedule), between the Y and Z groups where the pf exec used to put its
   read-prefetch.  t1 ping-pongs against a new second buffer t1b (35 KB,
   same T1P padding) so volume b+1's X stores never collide with volume
   b's plane reads.  Volume 0's X pass runs un-overlapped up front; at
   nb=1 the exec is schedule-identical to zs.  Same chunks, same
   per-volume store order => bit-identical output (md5-verified on the
   node, every run this round hashed 7b43c326).
   Measured vs zs, same window each time: in-plan race −2.0%, −1.3%,
   −1.4%, −1.9% (4166/4220, 4152/4212, 4109/4189, 4179/4237 ns/vol);
   graded FORCE=16 vs FORCE=14 A/B in one contended window 4.935 vs
   4.963/4.965 (−0.6%).  Never lost => flipped from raced-challenger to
   deterministic default.

2. **Chain-shaped ADOPTING in-plan race** (replaces the ice_r1
   instrument-only discriminator; pattern from **L17_matrixsimd ice_r1**
   stage 1g and **L17_rader ice_r1**'s "the quiet scoring window is the
   only honest ranking site" — both records' window-drift data, plus
   L13_rader's ±5% coloring lottery, say a hard-coded pick from a
   contended dev window would be adopting noise).  tb=min(batch,32)
   volumes ping-pong two private buffers with a driver-style unitary
   scale pass (x 2197^-1/2, untimed but cache-active) between steps, so
   candidates are ranked in the graded regime — ice_r1's ab raced an
   L2-resident fixed src->dst loop, the wrong regime (its B16 numbers
   were ~250 ns/vol optimistic vs the chain-shaped B32 race).  Slots:
   zs, ov, os (ov+sched), pf.  Adoption at a 1.5% margin toward the
   incumbent, only when ws <= L3; all candidates bit-identical, so the
   pick can never change output bits — the determinism contract's
   observable half survives.  Preceded by a ~120 ms dense-FMA
   clock-settle spin (**L17_matrixsimd <- L17_winograd**: schedutil
   leaves an 8 ms create on an unramped core).  Setup cost now 0.13 s.

3. **Pre-RA scheduling twins** (s = zs+pragma, os = ov+pragma;
   `#pragma GCC optimize("schedule-insns","sched-pressure")` on the exec
   definitions, borrowed from **L17_matrixsimd ice_r1** where it bought
   −7.7% on their chain cell).  On MY kernel it is a small LOSS — see
   negatives — kept only as the os race slot and FORCE=15/17.

### Operation count

Unchanged from panel_r11: 102 vector FP ops per zmm chunk, volume = 133 zmm
+ 14 xmm chunks (X-first, zsolid Y, xmm Z/X tails).  OV adds zero FP work —
it reorders whole chunks across volumes — and one 35 KB buffer.  On this
node's TWO 512-bit pipes the port floor is ~9.0k cycles/volume
(tile shuffles now compete with the second FMA pipe on p5), ~3.1 us at
2.9 GHz; the graded cell runs ~4.2 us kernel + ~0.5 us driver-side unitary
scale, i.e. ~1.2x above the floor, all of it L3 latency — consistent with
L13_rader's decomposition of the same cell.

### Measured on the NODE (a80n0 via tryout.sh; this panel does not measure
on wallaby — window drift is real, MKL quoted per window)

| config | graded B=32 us/xform | window MKL |
|---|---|---|
| ice_r1 code re-run | 4.785 min / 4.853 med | 6.323 |
| this round, zs still default | 4.665 / 4.665 (sd 0.06%) | 6.246 |
| FORCE=14 zs pinned (contended) | 4.963, 4.965 | 6.40, 6.39 |
| FORCE=16 ov pinned (same window) | 4.935 | 6.385 |
| **SHIPPED (ov default)** | **4.695 / 4.698** | 6.278 |
| B=1 m=1278 shipped | **4.423** (sd 0.05%) | 6.526 |

Shipped-build race print: chain-ab[B32] = zs 4237 / ov 4179 / os 4204 /
pf 4678 ns/vol, pick=ov(inc).  rel_l2 = 2.863e-16 (B=32) / 2.826e-16 (B=1)
single-transform, 2.847e-14 whole-chain (tol 3.6e-11); bit-repeatable;
output md5 identical across zs/ov/ow/s/os/default builds.  Expect the
quiet-window score at or slightly under ice_r1's 4.661 (ov's −1.3..−2.0%
is measured under dev-slot load; the un-overlapped volume-0 X pass costs
~1/32 of the gain back).

### What did NOT work, with the number that killed it

1. **ow — ov + prefetchw of the next out plane**: race 4774 vs ov 4166
   ns/vol (+13%), graded FORCE=18 5.228 vs ~4.94 same-day.  The Z stores'
   RFOs are evidently already hidden under the ov overlap, and 559
   prefetchw/volume just add L3 traffic.  Retired same-round; kept as
   FORCE=18.  Consistent with L13_rader's "pw is a ±2% coin flip at best"
   — on my shape it is far worse.
2. **Pre-RA scheduling pragma on the plain kernel (s)**: race 4460 vs zs
   4240 (+5.2%); graded FORCE=15 4.906 in a 4.7-window.  The −7.7% it
   bought L17_matrixsimd came from mixing their phase-SERIAL source
   (cosine block, then sine block, re-loading rows); my chunk13p is
   already a fused single-load sweep — their record's "next lever: merge
   the phases at source level" is a thing L=13 has had since panel_r9,
   so sched1 has nothing to mix and its schedule choices only displace
   gcc's.  ov+sched (os) tracks ~1% behind ov everywhere (4204-4315
   race; FORCE=17 graded 5.036 in a 4.7-window).  Lesson recorded: the
   pragma transfers only to phase-serial kernels.
3. **pf (cross-volume read-prefetch + pfw) at the graded cell**: race
   4678-4706 vs zs ~4220 (+10.6%) every window.  Matches L13_rader's
   +3.3% pf loss and extends it: when ov does the same latency-hiding
   with real work, dedicated prefetch instructions are strictly worse.
   pf remains the ws > L3 default (its regime; not scored here).

### Borrowed this round (attribution)

* **L17_rader ice_r1**: the "sp" cross-volume overlap concept -> ov (the
  round's win); the "dev windows are contention-poisoned, rank in-plan in
  the quiet window" doctrine -> adopting race.
* **L17_matrixsimd ice_r1**: chain-shaped tuning regime (ping-pong +
  unitary scale pass in the arena); the clock-settle spin (via
  L17_winograd); the scheduling pragma (transferred negative — see above).
* **L13_rader ice_r1**: the latency/L3-bound diagnosis of this exact cell
  (redirected the round away from port/shuffle work); the pw-coin-flip
  warning (motivated the hysteresis margin and pre-doomed ow).

### For next round

1. The remaining structure above the port floor is the volume-0 X pass
   (un-overlapped, ~1/32 of the ov gain) and the driver-side unitary
   scale (~0.5 us/xform, untouchable).  A cross-CALL overlap (X of the
   next step's input during this step's last volume) is blocked by
   correctness: the driver scales `out` between steps, so precomputed
   spectra would be stale by one scalar factor.  Do not attempt without a
   driver-level contract change.
2. If the leaderboard shows ov's edge did not survive the quiet window,
   the race print (chain-ab in the description string) says why — read it
   before touching code.
3. Port/shuffle work (tile-transpose alternatives, MULI folding) stays
   dead: L13_rader measured the p5 relief thesis twice (wash / +1.36 us)
   and this cell is latency-bound.  Revisit only if the workload ever
   becomes L1/L2-resident (e.g. a B=1-class scored cell).
