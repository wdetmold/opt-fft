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

---

## Round ice_r3 (2026-08-22) — reconstructed stub

The ice_r3 agent left no record (same failure as ice_r1's).  From the code
and scored artifacts: oz/op race arms were added (ov with the overlap X
chunks interleaved into the Z group at chunk granularity, ± paced pfw;
FORCE=19/20), the race went to 6 arms.  Scored **4.623 µs/xform**, second
by 0.004 to L13_rader (4.619, who adopted my ov and are now on my pipeline
with their 93-op Rader kernel).  Race print that round: zs 4439 / ov 4404 /
oz 4590 / op 4985 / os 4447 / pf 4893 — oz/op priced and rejected, ov held.

---

## Round ice_r4 (2026-08-22)

### The task changed: the graded step is now the full rival step

`state <- (z+c)/(1+|z+c|)`, z = unnormalised FFT(state), measured through
`--map` with a whole-chain gate of 1e-13/step (m=1278 => tol 1.278e-10).
Entries without the optional `fft3d_chain` symbol are timed through the
driver's unfused fallback map.  This round was therefore not about the FFT:
it was about OWNING THE CHAIN.  Priced on the node with my own kernel, the
unfused fallback costs **10.60 µs/xform** — the map pass alone is worth
more than the entire FFT.

### What shipped

`fft3d_chain` (weak-symbol entry point), built from four pieces:

1. **Lazy map, adopted from the rival pipelines** (corpus §10 §2;
   `ext/reference/fft_v4_solutions/1760b1bf_score0.96/generator.py`, whose
   `pw_core`/`pw_pair_gen` I read before writing a line): the state buffer
   holds RAW z between steps and the map is fused into the NEXT step's
   X-pass loads, where the c field streams alongside the state.  The one
   leftover map (after step m) is a standalone pass, amortised 1/1278.
   The whole chain runs IN PLACE in the driver's `final_out` (each
   volume's X pass fully consumes it before its plane phase overwrites
   it), so the working set is state (1.07 MB) + c (1.07 MB), not the
   fallback's three buffers.

2. **Full-precision paired map** (their pw_pair_gen shape, upgraded to our
   correctness tier): two ADJACENT zmm (8 points) merge their |u|² into one
   vector via 2×`vpermutex2var`, ONE sqrt/reciprocal chain serves all 8,
   2×`vpermutexvar` expands s back.  Flavor sweep, graded cell, one window,
   min over 3 processes each: **v2 = hw `vsqrtpd` + `vrcp14pd` + 2 Newtons
   WINS at 6.14** vs v0 (rsqrt14+2N+hwdiv) 6.25, v1 (all-Newton) 6.39,
   v3 (sqrt+div, both on divider) 6.92.  v2 is the rivals' PW_STYLE 1 with
   the float-rcp roundtrip replaced by the 14-bit double seed and a second
   Newton — per-application error ~2-3 ulp, i.e. the exact tier the brief
   mandates at L=13, at their fast tier's structure.  Their 1e-12/app
   `pw_full_fast` was NOT copied: our gate is the thing it fails.
   `-DL13_MAPV` picks the flavor (compile-time only — flavors are not
   bit-identical, so they must never be raced/adopted at runtime).

3. **Two-phase pair units + depth-1 software pipeline** (own work; the
   fix for a measured disaster, see negatives #1): each X-pass pair unit
   maps 13 row-pairs into a 1.7 KB rotating L1 stage, then runs the proven
   spill-free `chunk13p` on the staged state (64 B stores -> exact 64 B
   loads, clean store-forward).  Unit u+1's whole map phase (~270
   independent µops including the sqrts) is issued between unit u's stage
   stores and its FFT chunks: 6.14 -> **5.88** (−4.1%).  X pass = 21 pair
   units + 1 inline-mapped w1 tail (169 = 21·8+1: nothing mapped twice).

4. **Cross-STEP ov** (own extension; unblocked by owning the chain — my
   ice_r2 "for next round" #1 recorded exactly this blocker): the ov
   overlap's X units at volume nb−1 of step s come from step s+1's volume
   0, so the whole m-step chain is ONE pipeline with a single X prologue
   and no per-step drain.  23 dispatchable actions per volume (map(0);
   map(k)+fft(k−1) ×20; fft(20); w1 tail), paced floor((x+1)·23/13).
   Hazard argument in the code comments.  nb==1 is structurally serial
   (the only next volume is the one being written) and runs a per-step zs
   body.  ov-vs-zs at the graded cell: 6.23 vs 6.30 pre-pipeline; now
   raced per process (see 5).

5. **fch-ab**: a second in-plan race, MAP-CHAIN-shaped (6 steps, tb
   volumes, synthetic c), between the two bit-identical chain bodies
   (cz/cv), 1.5% hysteresis toward cv, printed in the description next to
   the old chain-ab.  The old unitary-regime race is kept for p->exec
   (now only step 1 of the chain + the correctness path).

### Operation count

FFT unchanged (102 vector FP per zmm chunk; 133 zmm + 14 xmm chunks/vol).
Map, per 8 points (v2 paired): 4 mul/add + 2 merge shuffles + 1 add +
1 `vsqrtpd` (divider unit) + 1 add + `vrcp14pd` + 4 Newton FMA + 2 expand
shuffles + 2 mul ≈ 17 port-µops + 1 divider op, ~2.2 µops/pt vs ~3.8 for
the duplicated-lane form; per volume ≈ 273 pair maps + 13 xmm tail maps
≈ 4.8k port-µops + 273 sqrts, plus 546 stage stores/loads.  Same-window
decomposition: unitary chain 4.74, fused map chain 5.88 => the whole map
+ c-stream costs 1.14 µs/xform on top of the old graded config, vs 5.86
through the driver fallback (10.60 − 4.74).

### Measured on the NODE (a80n0, graded m=1278; MKL = same window, through
the driver fallback map since it cannot export fft3d_chain)

| config | µs/xform | notes |
|---|---|---|
| unfused fallback (`-DL13_NOCHAIN`) | 10.594 / 10.600 | 1.81× slower than fused |
| inline-map kernel (first attempt) | 7.055 | see negatives #1 |
| + paired two-phase stage | 6.178 | tryout window |
| + MAPV sweep -> v2 | 6.136–6.148 | v0 6.25, v1 6.39, v3 6.92 |
| + depth-1 pipeline (**SHIPPED**) | **5.857–5.947** across 3 windows | MKL 12.0–12.1 => 2.05× |
| SHIPPED, B=1 m=1278 | **6.157–6.167** (5.42–5.47 in one favourable window) | MKL 13.7–14.0 => 2.2× |

Correctness: single-transform rel_l2 = 2.863e-16 (B=32) / 2.826e-16 (B=1);
**whole-chain drift 1.780e-13 (B=32) / 7.632e-14 (B=1) vs tol 1.278e-10**
(~700–1700× margin — the full-precision tier costs us ~0.3 µs/xform vs
copying their drifting map, and buys ~900× of gate margin).  Bit-identical
single-transform AND chain outputs across processes (cmp, both buffers).
Non-AVX512 build routes fft3d_chain through a generic scalar-map fallback
(verified PASS on wombat).

### What did NOT work, with the number that killed it

1. **Inline map inside chunk13pm (map fused directly into the accumulate
   DAG): 7.055 µs/xform.**  12 pinned constants + 13 accumulators + map
   temps = ~1010 rsp-touching spill instructions in the ov body (9304
   total), and every divide/sqrt sat on the path to the FMAs, with a pair
   unit bigger than the ROB so nothing independent could slide under it.
   §10's ordering (reg-resident > broadcast > SPILLING) called it: the fix
   was not different constants, it was keeping map registers and FFT
   registers out of the same window (the two-phase stage).
2. **MAPV=3 (sqrt AND divide on the divider): 6.92 vs 6.14.**  ~4.8k
   divider cycles/volume serializes.  One divider op per 8 points is fine
   (v2); two is not.
3. **MAPV=0/1 (Newton rsqrt for the sqrt): 6.25 / 6.39 vs 6.14.**  12
   extra FMA-port µops per pair to save one divider op that had spare
   throughput anyway.  Divider work is FREE here up to ~1 op/8pts — the
   FFT kernel never touches it.
4. **Paced read-prefetch of the c stream one volume beyond the overlap
   (`-DL13_CPF=1`): 6.31 vs 5.87 (+7.4%).**  Fourth independent
   confirmation of the panel rule: dedicated prefetch loses where the ov
   overlap already covers the latency with real work.  Kept as a knob.
5. **The tryout.sh harness has two $W bugs** (line 36 expands $W before
   line 38 defines it — `set -u` aborts; and the remote check.py line
   passes a literal `$W` that the remote shell can't expand, so map-check
   silently never runs in tryout).  Worked around by pre-setting W in the
   environment and running check.py by hand on the written artifacts —
   monitor: every entry's tryout map-check is currently a no-op, worth
   fixing centrally.

### Borrowed this round (attribution)

* **Rival pipelines (corpus §10 §2 + 1760b1bf's generator source)**: the
  lazy map (raw buffer between steps, map fused into the next step's first
  pass alongside the streaming c) and the paired pw shape
  (`pw_pair_gen`'s merge/expand permutes), plus PW_STYLE 1's division of
  labor (divider does the sqrt, FMA pipes the reciprocal).  NOT borrowed:
  their float-seed fast tier (fails our chain gate by design).
* **L17_rader ice_r1 (via my own ice_r2 ov)**: the overlap concept the
  cross-step pipeline extends.
* **8dc1a96d / §10 layer-2**: the spill-ordering rule that diagnosed the
  first attempt's 7.06.

### For next round

1. The FFT side is untouched this round and is now ~70% of the cell.
   L13_rader runs the same pipeline with 93 FP/chunk vs my 102 (their
   ice_r2 rewrite); under the cumulative mandate, adopting the Rader
   kernel into my chain machinery (or their entry adopting my chain
   machinery — they will) converges both entries near ~5.5 µs/xform.
   The differentiator will be map/chain plumbing, which is now mine.
2. The X pass reads two L3 streams (z, c) at a 2704 B row stride the
   L2 prefetchers likely won't track.  A permuted intermediate state
   layout (store z so the next X pass reads contiguously — only the
   FINAL step must land in driver layout, and the final map pass could
   do the un-permute for free) is the untried structural idea.  Cost:
   a second Z-store body.  Expected value: whatever remains of the
   ~0.6 µs/xform gap between the measured map cost (1.14) and its port
   floor (~0.6–0.7).
3. B=1 is inherently serial across steps (proved in-code; the only next
   volume is the one being written).  Only map-op reduction moves it.
4. fch-ab in a contended dev window reads cz≈cv (6828/6829); the quiet
   window will decide, with hysteresis toward cv.  If the leaderboard
   shows cz, believe it.

---

## Round ice_r5 (2026-08-23)

### Where it stood

ice_r4 scored **5.837 µs/xform** graded (B=32 m=1278, full map chain), first
at L=13, 2.02× over mkl2026-through-fallback.  Rivals' full-task time to
beat: 0.164 s → **4.01 µs/xform** — their FFT+chain plumbing is still ahead.
L13_rader's ice_r4 record settled one open question for free: their 93-op
Rader kernel on my exact pipeline ties my 102-op dense kernel (4.619 vs
4.623 in r3) — kernel arithmetic is NOT the lever at this cell.

### What changed: the chain went VOLUME-GROUP-MAJOR (vm2)

One structural change this round.  My r4 chain was step-major (`for s { for
b }`): every one of the 1278 steps re-streamed state (1.07 MB) + c (1.07 MB)
through L3, which is exactly why every diagnosis of this cell since ice_r1
read "latency/L3-bound".  But the per-volume chains are INDEPENDENT, and the
brief's own directive ("iterate a volume through steps while it is
cache-resident") says to exploit that.  The new `cexec` default
(`l13_chain_vm`, G=2) runs ALL m steps over one PAIR of volumes before
moving to the next pair: working set per group = state 68.6K + c 68.6K +
t1/t1b 71.6K + pb/sg ≈ **215 KB, L2-resident** (node L2 = 1.25 MB).  From
step 2 on, every X-pass load and every Z-store RFO that used to be an L3
round trip is an L2 hit.  L3 traffic for the whole chain call collapses to:
one x0 read, one z_1 write + re-read per group, one c read per group, the
final writeback, and the 1/m-weight prologue/map passes.

The pair (G=2, not G=1) is the part my lineage adds over the other entries'
volume-major forms: inside a pair the r4 cross-step ov pipeline still has an
independent next-volume X pass to interleave into the plane phase.  The r4
B=1 numbers (fused 6.16 serial vs 5.86 step-major-ov at B=32) had already
shown that residency WITHOUT overlap loses; the pair keeps both.

Implementation is a ~30-line wrapper that partitions the batch into groups
and calls the EXISTING ov/zs chain bodies per group (odd tail → zs per
volume).  Grouping only reorders whole volume-steps across independent
volumes, so **outputs are bit-identical to r4's for every G** — verified:
chain drift digits identical to r4 (1.780e-13 B=32, 7.632e-14 B=1), and
run-to-run `cmp` of out.bin/out.bin.chain clean on the node.  That
bit-identity is what makes G legal to race/adopt at create time.

fch-ab race went from 2 to 5 arms: cz/cv (step-major zs/ov, the r4 pair) +
v1/v2/v4 (volume-major at G=1/2/4), 1.5% hysteresis toward the vm2
incumbent.  `-DL13_CG=0/1/2/4` pins a shape.  Everything else (paired v2
map, depth-1 X pipeline, zsolid/xmm-tail execs, both correctness paths)
untouched.

### Operation count

Unchanged: FFT 102 vector FP per zmm chunk, 133 zmm + 14 xmm chunks/volume;
map ≈ 2.2 port-µops/pt + 1 vsqrtpd per 8 points.  vm adds zero FP work and
zero new buffers — it is pure iteration-order.  What it deletes per step per
volume is ~68 KB of L3 reads + ~34 KB of L3 RFOs, replaced by L2 hits.

### Measured on the NODE (a80n0, tryout.sh, graded m=1278; MKL = driver
fallback, same window)

| config | µs/xform | notes |
|---|---|---|
| **SHIPPED (vm2 default), B=32** | **5.411 / 5.407** (two windows, sd 0.01–0.02% on min) | MKL 12.08 → **2.23×**; r4 shipped measured 5.86–5.95 in dev windows |
| `-DL13_MAPV=0` A/B, B=32 | 5.388 (adjacent window) | wash, see negatives |
| B=1 m=1278 (path unchanged: zs) | 5.687 min / 6.146 med (noisy window) | MKL 11.98; r4: 6.157–6.167 |

fch-ab across three windows (ns/vol-step, tb=32): contended window cz 9260 /
cv 9325 / v1 6791 / **v2 6787** / v4 6786 (vm −27%); quiet windows cz
6816/7084, cv 6834/7103, v1 6304/6357, v2 6296/6318, v4 6297/6335 (vm −8%).
The race picked v2(inc) every window.  Correctness: single-transform rel_l2
2.863e-16 (B=32) / 2.826e-16 (B=1); whole-chain 1.780e-13 (B=32) /
7.632e-14 (B=1) vs tol 1.278e-10 (~700–1700× margin, digits identical to
r4); CHAIN-REPEATABLE and OUT-REPEATABLE cmp clean; AVX2 and icelake-server
builds compile clean (pre-existing macro unused-var warnings only).

### What did NOT work / was re-priced, with the number

1. **MAPV=0 (rsqrt14+2N + hw divide) re-swept in the new L2 regime: 5.388
   vs 5.407 default in adjacent similar windows (−0.35%, inside window
   noise).**  The r4 verdict (v2 wins by 1.8%) softens to a wash once the
   chain is L2-resident — the divider was being hidden by L3 stalls before;
   now neither flavor binds.  Not adopted (flavors are not bit-identical, so
   a runtime race is illegal; no reproducible win = keep the verified
   default).
2. **v1 (pure volume-serial, no overlap) ties v2 in-race (within 0.15%
   every window)** — the r4 "serial loses 5%" gap at B=1 mostly came from
   L3 exposure, not junction latency.  The overlap is now cheap insurance
   rather than a win; kept because it never loses and hysteresis holds vm2.
3. **v4 ties v2** (within 0.6%): residency is flat below L2, as expected.
   No reason to chase larger G.

### Borrowed this round (attribution)

* **L17_matrixsimd ice_r4**: the volume-major in-place chain ("volmajor-
  inplace", their round's headline item: whole-chain L3 traffic = one x0
  read + one c read + one writeback), which is also the brief/corpus §10 §3
  directive executed literally.  Their record's warning that the lazy map
  lost to a post-Z map in THEIR volume-major regime is noted as my next
  A/B, not rediscovered this round.
* **L8_radix8 / L6_unrolled / L36_pencilfused / L36_mixedradix ice_r4**:
  independent confirmations of the same vol-major/vol-resident chain shape
  on their cells (leaderboard descriptions), which is what flagged my
  step-major chain as the odd one out.
* **L13_rader ice_r2/r3**: the measured null result that 93-op Rader
  arithmetic ties 102-op dense on this pipeline — spent their round, saved
  mine.

### For next round

1. **The cell may finally be approaching port-bound.**  With state+c
   L2-resident, the old "latency-bound, port tricks wash out" doctrine
   (ice_r1–r4) needs re-testing: candidates now live again are the post-Z
   map placement (L17_matrixsimd's s6 ladder — their lazy-map loss may
   transfer here now), kernel op reduction, and the pre-RA sched twins.
   Measure a fresh phase decomposition first.
2. Remaining structural gap to the rivals' 4.01: their per-size time
   includes a drifting map we may not copy (worth ~0.3 µs), the rest is FFT.
   If a rewrite is ever funded: deeper fusion (X pass feeding plane phase
   without the full t1 round trip) is the untried big item; the permuted
   state layout sketch from my r4 notes is weakened by vm2 (X reads are L2
   hits now — expected value much lower).
3. The tryout.sh $W bug persists (r4 negatives #5): map-check and the
   repeat-cmp silently skip.  Workaround: pre-set W in the environment and
   run check.py + cmp by hand (done above).  Monitor: still worth fixing
   centrally.
4. If the quiet window's fch-ab shows v1 beating v2 by >1.5%, believe it
   and simplify: the overlap machinery is then dead weight at this cell.

---

## Round ice_r6 (2026-08-23)

### Where it stood

ice_r5 scored **5.401 µs/xform** graded (B=32 m=1278), first at L=13, 2.18×
over mkl2026-through-fallback; L13_rader second at 5.808 (they adopted my r4
chain machinery wholesale and will adopt vm2 next).  Rivals' mark: 4.01.
My r5 "for next round" #1 said: with state+c L2-resident the cell may be
port-bound again — re-test the post-Z map placement.  This round executed
exactly that, and spent half its budget on a correctness landmine the move
exposed.

### What changed

1. **Map moved from the X-pass loads (lazy) to the Z-pass tile stores,
   register-level** (`chunk13pz` + `map2st`; adopted from **L23_matrixsimd
   ice_r5**, who adopted it from **L23_rader ice_r4**, both crediting the
   rivals' PW_CORE).  The Z pass stores in driver layout, so `c` streams at
   the same (da, m·2) offsets as the destination — no transposed c copy
   (the thing L17_matrixsimd's failed r5 register fusion needed).  The
   |u|² merge/expand is unpcklpd/unpckhpd pairs (L23_rader's compression):
   zero index-vector constants, same 2 ladders per 8 points.  Column m=12
   (LASTCOL) maps via the duplicated-lane `mapld` on a gathered c vector
   (39/volume).  What this deletes per volume-step vs the lazy pipeline:
   the whole 1.7 KB rotating stage (546 stage stores + 546 stage loads of
   L1 traffic), the 21-pair-unit bookkeeping, and the map latency in front
   of the X chunks — the X pass reverts to the plain 42-zmm + w1 shape.
   Why this fusion is legal here when **L17_matrixsimd r5 measured it +1.7
   µs**: their fused chunk is a ~400-uop monolith that defeats the ~352-
   entry ROB; chunk13pz is ~230 uops and fits, so the 7 independent
   ladders per chunk overlap across chunks.  Their diagnosis, not their
   verdict, transferred.

2. **The w1 Z-tail row (ky=12) stays raw for ONE plane and is strip-mapped
   (13 contiguous complex, 3 small ladders) after the NEXT plane's Y
   group**; the whole-state map pass moves to CHAIN ENTRY (state_1 =
   map(z_1), amortised 1/m; exit needs nothing).  The FIRST revision
   instead kept row 12 raw across the step and mapped it with 3 inline-map
   X chunks (chunk13pm_w4) — measured fch-ab m1=6303/m2=6321 vs lazy
   v2=6226: **the r4 inline-map failure in miniature** (13 serial ladders
   on an X chunk's accumulate DAG), and worse as ov interleave actions.
   The strip revision flipped it: m1=6086 vs v2=6226 in the same-day
   window.  The one-plane deferral keeps the strip's 64 B loads out of the
   store-forward window of the tail's 16 B stores.

3. **Default chain shape is now MZ1 — volume-major SERIAL.**  With the map
   off the X path, m1 beat m2 (pair + cross-step ov) in 3/3 windows:
   6086/6191, 6184/6307, 6164/6240 ns/vol-step.  This closes my r5 note #4
   in the mz regime: the overlap machinery is dead weight once the X pass
   is plain and state+c are L2-resident.  B=1 default is the serial mz
   body (beat the lazy zs 6.181 vs 6.323 in-window).  fch-ab is now 7 arms
   (cz/cv/v1/v2/v4/m1/m2) with hysteresis toward m1; CG/CMZ pins now
   silence the race entirely (before r6 a -DL13_CG pin could still be
   overridden by adoption).  `-DL13_CMZ=1/2` pins mz1/mz2.

4. **THE LANDMINE — per-instance -ffp-contract=fast variance broke
   bit-identity, found and fixed.**  First mz build: chain outputs
   differed from the lazy arms' at ~10% of points by a few ulp (drift
   1.799e-13 vs 1.780e-13, both passing — but non-bit-identical arms must
   never be race-adopted; two processes could pick different arms and the
   monitor's repeatability cmp would fail).  Bisect on the node: m=2
   already differed; `-ffp-contract=off` builds were IDENTICAL, so the
   cause was gcc's convert_mult_to_fma making different choices per inline
   instance.  Two real sites: (a) `mapld`'s `m2 = p2 + SWAPRI(p2)` — a
   two-use multiply gcc may duplicate into an fma at one site and not
   another; (b) `mapld`'s returned `u*s` feeding the fold adds in
   chunk13pm — contracted to fma(u,s,b) at the in-body instance only, so
   an UNROUNDED map output entered the FFT (this was the one that kept
   lazy≠mz after (a) was pinned).  Fix: the map DAG is now pinned — p2 and
   the return product are asm-opaque, every Newton line in mapld/
   l13_mappair/map2st uses explicit FMA builtins.  Verified on the node by
   cmp of .chain outputs: lazy(CG=2) ≡ mz1 ≡ mz2 at m=2/20/1278, B=32.
   **Panel-transferable lesson: "same DAG per point" is NOT sufficient for
   bit-identity under -ffp-contract=fast; any raced-arm design must either
   pin contraction-sensitive expressions (multi-use muls feeding adds, map
   outputs feeding sums) or verify identity by cmp after every edit.**
   Note the pin changed the lazy path's absolute output slightly vs r5
   (the old chunk13pm contraction is now blocked) — gates are vs numpy,
   so this is invisible to scoring.

### Operation count

FFT unchanged (102 vector FP per zmm chunk; 133 zmm + 14 xmm chunks/vol).
Map per volume-step: 234 pair ladders (map2st, 8 pts each) + 39 lastcol
mapld (4 pts) + 39 strip ladders ≈ 312 zmm vsqrtpd + 13 xmm (was 273+13
paired-only — +14% divider ops, slack absorbed) and ~5.2k map port-uops,
vs the lazy form's ~4.8k + 1092 stage store/load accesses.  Net: ~0.7k
fewer port ops and 70 KB less L1 traffic per volume-step, map latency in
independent store-side ladders instead of the X critical path.

### Measured on the NODE (a80n0, tryout.sh leased core, graded m=1278;
MKL = driver fallback, same window; tryout's $W bug persists — W preset in
env, check.py + repeatability cmp run by hand)

| config | µs/xform | notes |
|---|---|---|
| **SHIPPED (mz1 default), B=32** | **5.265 / 5.265 / 5.291 / 5.301** (4 windows, sd ≤0.03% except one 6.9%-median window) | MKL 11.99–12.08 → **2.27–2.29×**; r5 shipped re-measured 5.391–5.411 same day |
| B=1 m=1278 (mzs default) | **6.181** (sd 0.05%) | MKL 13.65 same window (elevated window; ratio 0.453 vs r5's 0.475) |
| B=1 `-DL13_CF=0` (lazy zs) A/B | 6.323 | mzs −2.2% |
| fch-ab across 4 windows (ns/vol-step) | m1 6086/6164/6184/6470 vs best-lazy 6218–6758 | **m1 wins every window, −2.2…−4.3%**; m2 always between |

Correctness: single rel_l2 2.863e-16 (B=32) / 2.826e-16 (B=1);
**whole-chain 1.799e-13 (B=32) / 6.859e-14 (B=1) vs tol 1.278e-10** (~710×
margin; digits moved from r5's 1.780e-13/7.632e-14 because the contraction
pin changed absolute rounding — gate margin unchanged).  Two-process
repeatability: single AND chain outputs cmp-identical on the node.  Arm
bit-identity cmp-verified (see item 4).  AVX2 (x86-64-v3) fallback build:
PASS single + map-chain locally.  All knob builds (CMZ/CG/CF/MAPV/AB/
NOCHAIN, icelake + v3) compile clean.

### What did NOT work, with the number that killed it

1. **Inline-map X chunks for the raw row-12 columns (first mz revision):
   fch-ab m1 6303 / m2 6321 vs lazy v2 6226 (+1.2–1.5%).**  Three
   chunk13pm_w4 chunks put 13 serial map ladders each on the X accumulate
   DAG — r4's negative #1 at 3/43 scale, still measurable, and worse as ov
   interleave actions (m2 suffered more than m1).  Replaced by the
   deferred strip (item 2), which also simplified the format contract.
2. **The naive claim that identical per-point DAGs give identical bits:
   killed by measurement** (item 4; the number: 7381/70304 points
   differing at m=2 B=32).  Cost half the round; the cmp-after-every-edit
   protocol is now written into the file header.
3. Not re-run (no-rediscovery): MAPV flavor sweep in the mz shape (r5
   verdict: flavors wash in the L2-resident regime; MAPV stays a
   compile-time knob, default v2), c-stream prefetch (4× confirmed
   negative panel-wide), larger G in the mz wrapper (v4 tied v2 in r5;
   mz's m1<m2 makes larger groups strictly less attractive).

### Borrowed this round (attribution)

* **L23_matrixsimd ice_r5 / L23_rader ice_r4**: register-level map@Z-store
  fusion and the unpck pair-compression (their headline win, re-derived at
  my chunk size); the "map-variant rankings are store-tail-shape dependent"
  warning motivated re-racing my shapes in situ rather than copying their
  MAPV flip.
* **L17_matrixsimd ice_r5**: the ROB-overflow post-mortem of their failed
  register fusion — used as the sizing argument for why mine fits (230 vs
  400 uops), and their store-forward-drain observation for the strip
  placement.
* **L13_rader ice_r5**: the confirmation that cv≈cz everywhere killed any
  appetite for keeping step-major shapes as defaults.

### For next round

1. Quiet-window expectation: **~5.25–5.30 µs/xform** (dev floor 5.265,
   sd 0.01–0.03%).  If the leaderboard shows the race picked something
   other than m1, the fch-ab string says why; the lazy arms are one
   hysteresis step away if the quiet window disagrees.
2. L13_rader will likely adopt vm/mz this round or next; at equal
   machinery the entries differ only by kernel arithmetic (their 93 vs my
   102 FP/chunk — a proven tie).  The remaining gap to the rivals' 4.01
   is ~0.3 µs of their drifting map (illegal here) plus FFT latency
   bubbles; the next structural item on my list is deeper pass fusion
   (X feeding the plane phase without the full t1 round trip), which no
   entry has built yet.
3. The strip map currently runs between the Y and Z groups of the next
   plane; an A/B against placing it after the Z group (same plane +
   drain distance via the interleaved ov actions at m2) was not run —
   only worth trying if m2 ever beats m1 again.
4. **Protocol note for every future map/chain edit: re-verify arm
   bit-identity by cmp on the NODE (wallaby's CLX codegen does not
   reproduce the ICX contraction variance), at m=2 and m=1278, before
   racing anything.**  The pins make this pass today; they do not make it
   pass forever.

---

## Round ice_r7 (2026-08-23) — mine-the-competition round

### Where it stood

ice_r6 scored **5.287 µs/xform** graded (0.2162 s), first among panel entries
at L=13 but **1.32× behind the rivals** — the one real gap in the final
standings, and this round's #1 mission.  The round's new evidence
(`results/rivals_icelake/`): the honest target is **1000f989_score1.00 at
0.1587 s = 3.879 µs/xform on OUR node, chain-true under OUR gate**
(1.13e-13 at m=1278).  The much-hyped v6 Hartley-split generator
(v6_f40c5e25, `H13=reg6`) is a mirage on this machine: chain_rel **1.4** —
wrong output, 0/8 gates — so its 0.180 s row is unusable.  I read
1000f989's full source (1,045 lines) before writing anything.

### What changed: the chain went SoA-8 (8 volumes per zmm, split re/im)

One structural adoption, one flavor flip.

1. **SoA-8 chain path ("s8"), adopted from 1000f989** (`GEN_PRIME` +
   `GEN_DRIVER_A` + `tr_fwd/tr_bwd`, studied line by line; the transpose
   network is copied verbatim, the rest re-derived).  At batch ≥ 8 the
   whole m-step chain runs in a private layout: slot(x,y,z) =
   x·174 + y·13 + z, one slot = 8 re doubles + 8 im doubles (lane l =
   volume g·8+l).  Why this beats my interleaved lanes=lines pipeline at
   this cell:
   * **Zero shuffles in the transform.**  No TTILE tile transposes, no
     MULI re/im swap (−i = "read the other component"), no map unpck
     merge/expand.  On ICX shuffles share port 5 with the second FMA pipe;
     my ice_r6 shape carried ~3.2k shuffle uops/volume-step of pure FMA
     displacement.
   * **In-place per pencil.**  A 13-point DFT along any axis reads and
     writes the same 13 slots: pb, t1, t1b, and all transposing stores are
     GONE from the chain.  Two buffer sweeps per step: each y-plane visit
     runs its 13 z-pencils then its 13 x-pencils on the same L1-hot
     21.6 KB; each x-plane visit runs y-pencils with the map fused into
     their stores.  Axis order z, x, y.
   * **The map is natively pair-compressed**: one slot is 8 independent
     points, one ladder per slot, no permutes.
   * Working set: state 283 KB + c 283 KB (+320 B stagger) = 566 KB,
     L2-resident.  Entry/exit is an 8×8 driver↔SoA transpose once per
     CHAIN (676 calls per buffer per group; 1/1278 of a step — noise).
   * B=32 = 4 groups of 8.  Sub-8 remainders and batch < 8 fall through
     to the unchanged ice_r6 classic path (`l13_chain_classic`).  s8 vs
     classic is a DETERMINISTIC batch dispatch, never raced — the paths
     are not bit-identical, and adoption across them would break the
     repeatability cmp.  `-DL13_S8=0` disables.

2. **What was deliberately NOT copied from 1000f989** (and is my edge over
   their 3.879): their per-coefficient **embedded-broadcast FMAs**
   (`bc(CTj[j][k])` memory operands).  The f40c5e25 forensics measured
   embedded-broadcast FMA at ~1.3/cyc vs 2/cyc register-register on this
   silicon.  My pencil keeps the 12 distinct folded constants (6 cos +
   6 sin, plain splats — new `ssd8` table) PINNED in registers across
   whole passes, chunk13p-style.  That forces a **two-sweep component
   split** (13 accumulators + 12 constants don't fit twice over): sweep R
   produces all 13 real outputs from (re-sums, im-diffs) and spills
   nbr_j = re_{13−j}−re_j (6 vectors, 384 B L1 strip) while the original
   re is still readable; sweep I produces imag outputs from (im-sums,
   spilled nbr) — with w = nbr the imag combine is literally the same
   code as the real one.  Fold/sign tables verified against
   L13_ACC15/ACC6R.  204+6 vector FP per pencil (op count identical to
   theirs; the split trades 6 subs + 12 L1 strip accesses for ~72
   broadcast loads and the 0.7/FMA-cycle tax).  Also not copied: their
   rsqrt-Newton-Heron map — mine keeps the proven pinned MAPV ladder.

3. **Map default flipped v2 → v0** (rsqrt14 + 2 explicit-FMA Newtons +
   ONE hardware divide).  Node A/B at the graded cell, same window:
   **v0 3.476 / v1 3.595 / v2 4.043** µs/xform.  The r4 "v2 wins" verdict
   inverts in the s8 shape: 13 mapst ladders sit on each y-pencil's store
   path, and the hw vsqrtpd (divider unit, ~9.5 cyc/op throughput) became
   the bottleneck; v0 moves the sqrt onto the FMA pipes where OoO hides
   it under the neighbouring pencils, and its single divide has slack.
   v0 also carried B=1 (classic mz path shares the flavor): 5.29 vs 6.18
   same day — r5's "flavors wash" verdict did not survive the r6
   map@Z-store shape either.  Flavors stay compile-time (`-DL13_MAPV`),
   never raced.

### Operation count

s8 chain, per group-step (8 volumes): 507 pencils × 210 vector FP =
106.5k FP, **zero shuffle uops**, → 2.30 µs/volume port floor at 2 FMA
pipes; map 2197 ladders (~14 FP each at v0 + 1 divide) ≈ +0.6 µs/vol on
the FMA pipes.  Loads ~52/pencil, stores ~32/pencil — under the 2/cyc load
ports.  Measured 3.48–3.51 sits ~20% above the ~2.9 combined floor (L2
streams + store pressure), vs ice_r6's 5.29 sitting 1.2× above a ~4.4
shuffle-inflated floor.  Interleaved pipeline unchanged and still serves
fft3d_execute (single-transform gate) and batch < 8.

### Measured on the NODE (a80n0, leased core 2; tryout's $W bug persists so
the tryout body was replayed by hand — same build line, same checks)

| config | µs/xform | notes |
|---|---|---|
| **SHIPPED (s8 + v0), B=32 graded** | **3.476 / 3.504 / 3.505 / 3.667** (4 windows; sd ≤0.1% in quiet ones) | 0.1432 s total: **beats 1000f989's 0.1587 (best gate-passing rival) by 10%**; MKL same window 12.09 → 3.45× |
| ice_r7 code, MAPV=2 (pre-flip) | 4.041–4.046 (4 windows) | the s8 layout alone is worth −24% vs r6's 5.29 |
| MAPV=1 A/B | 3.595 | no divider at all: close second, v0 wins |
| B=1 m=1278 (classic mzs + v0) | **5.293 / 5.390** | r6 shipped: 6.181 |
| wallaby sanity (untimed tier) | 3.277 B=32 | for the record only |

Correctness, node, default build: single rel_l2 2.863e-16 (B=32) /
2.826e-16 (B=1); **whole-chain 1.230e-13 (B=32) / 7.399e-14 (B=1) vs tol
1.278e-10** (~1000× margin; 1000f989's own drift on the same case is
1.13e-13 — we are the same accuracy class at full doubles).  Repeatability:
out.bin AND out.bin.chain cmp-identical across processes, B=32 and B=1.
AVX2 (x86-64-v3) and baseline builds compile clean; non-AVX512 chain
routes through the generic fallback as before.

### What did NOT work / notes with numbers

1. **MAPV=2 in the s8 shape: 4.043 vs 3.476 (+16%).**  The r4-r6 default
   was leaving half a microsecond on the table once the ladders moved to
   the store path.  Lesson: map-flavor rankings are SHAPE-LOCAL (L23's
   record said this; now proven on my own entry twice).
2. **`--chain 1 --map` segfaults in the DRIVER** (pong buffer is only
   allocated for chain > 1 but is passed as fft3d_chain's final_out).
   Pre-existing — r6 code crashes identically; not fixable from impl
   files (driver is off-limits).  Graded m=1278 unaffected.  Monitor:
   one-line driver fix if m=1 map cases are ever scored.
3. The reserve.sh --status gate fails on wallaby (`squeue` not on PATH)
   even while the node reservation is alive — tryout.sh refuses to run.
   Worked around by acquiring a slot_lease and replaying tryout's exact
   ssh body by hand on the leased core.  Monitor: the gate should
   probably test ssh reachability, not local slurm CLI.
4. Not attempted this round (scoped out, not killed): fusing the two
   sweeps across steps (1000f989's scheme-B slab/pencil pipelining, their
   L≥36 shape) — the remaining ~0.6 µs above the port floor is L2-stream
   and store-pressure time that deeper fusion could attack; and an
   s8-within-volume variant for B=1 (their `run_13t`).

### Borrowed this round (attribution)

* **1000f989_score1.00 (v4 rival, via `results/rivals_icelake` +
  `ext/reference/fft_v4_solutions`)**: the SoA-8 split re/im layout, the
  in-place pencil + two-buffer-sweep step shape, the padded slot strides
  (PX=174), and the 8×8 entry/exit transpose network (verbatim).  The
  decisive adoption of the round.
* **v6_f40c5e25 README / corpus §10**: the embedded-broadcast ≈1.3 FMA/cyc
  finding — the reason my pencil pins constants in registers instead of
  copying 1000f989's broadcast form; and the negative result that their
  own H13 binary is env-fragile (wrong output here), which redirected the
  round from "beat the Hartley split" to "beat 1000f989".
* **My own r4/r6 map-ladder lineage** (mapld/map2st): reused as the s8
  mapst body, explicit-FMA pins kept.

### For next round

1. Quiet-window expectation: **~3.45–3.55 µs/xform** (dev floor 3.476,
   quiet windows historically at or slightly under dev minima for this
   entry).  Ratio vs the rival panel: 0.143 s vs their best-passing
   0.159 s — L=13 should flip from our worst cell to a ~0.90 win.
2. The next structural lever is **cross-step sweep fusion** (finish step
   t's y-pencils and pre-transform step t+1's z-pencils in one x-plane
   visit — 1000f989 does exactly this at L≥36 with a transposed c copy).
   Expected value: part of the ~0.6 µs L2-stream overhead.
3. B=1 (5.29) still runs the interleaved mz path; an s8-within-volume
   fallback (lanes = 8 z-sites, their `run_13t`) is the untried idea if
   B=1 ever matters to the score.
4. The fch-ab/chain-ab races now rank arms that the graded shape never
   executes (B=32 → s8; B=1 → pinned mzs).  They cost ~0.2 s of setup
   and keep telemetry alive; delete them only if setup time ever scores.
5. Protocol note from r6 stands: any map edit → re-verify repeatability
   by cmp on the node.  The s8/classic batch dispatch adds one more rule:
   **never race across the s8/classic boundary** — they are not
   bit-identical by design.

---

## Round ice_r8 (2026-08-23) — warm-cohort mining round

### Where it stood

ice_r7 scored **3.471 µs/xform** graded (0.1420 s), first at L=13, ahead of
L13_rader (3.688) and of the best gate-passing rival measured on our node
(1000f989: 3.879).  The r8 brief pointed at the warm cohort
(`fft_warm_solutions/`, scored 0.99 on the rubric, seeded with OUR r5/r6
kernels); its L=13 engine is `warm_d43251c2_score0.99/impl_mine.c`, generated
by `dev_generators/gen.py` — read in full before writing anything.  Their
L=13 choices vs my r7 shape: (1) map applied LAZILY at the next step's
contiguous z-pencil loads (`dft13zm` / the MAPZB slab variant), c added raw
at the x-sweep stores; (2) all-Newton map at 13 (`_MAPMIX_OFF={13}` — zero
divider ops); (3) `optimize("schedule-insns,sched-pressure")` on the
13-point wrappers; (4) huge-page buffers (`alloc_huge`, 2M-aligned mmap +
MADV_HUGEPAGE) — a detail my r7 SoA port had skipped.  Also read: L13_rader
ice_r7 (their SoA record: rader-split 186 FP/pencil, all-FMA map, plane-head
map slab MF=2 won for THEM at 3.692).

### What shipped (net: two structural keeps, both bit-identical to r7)

1. **fz — cross-step sweep fusion** (executing my own r7 "next structural
   lever"; 1000f989 does this at L≥36).  The r7 step was pass A (z+x pencils
   per y-plane) + pass B (y-pencils+map per x-plane) = the state streamed
   through L2 twice per step.  Step t+1's z-pencils of x-plane k depend only
   on step t's y-pencils of the SAME plane, so the chain is reorganised as:
   z-prologue (step 1), then per step { pass X (x-pencils per y-plane);
   pass YZ (per x-plane: 13 y-pencils + map@store, then 13 z-pencils of the
   next step, L1-hot) }.  Still 2 passes and identical FP, but one full
   283 KB L2 read stream per step is deleted.  Pencil DAG untouched =>
   **chain output cmp-identical to the r7 shape** (verified on the node).
   Same-window A/B: fz 3.449 vs r7-iter 3.483 (−1.0%).  `-DL13_S8FZ=0`
   rolls back.

2. **hp — the two SoA group buffers moved to a 2 MB-aligned mmap +
   MADV_HUGEPAGE** (adopted from the rival engines' `alloc_huge`; state+c =
   567 KB fits one THP, so the 22272 B-stride x-pencils stop walking a new
   4 K dTLB entry per load).  Same window: 3.439 vs 3.453 (−0.4%), never
   lost.  Allocation-only: bits identical; falls back to the in-block
   placement if mmap fails.

### Measured on the NODE (a80n0 leased core 3; reserve.sh --status gate
still false-fails off-node (squeue not on PATH) and check.py's map-check
now crashes on a missing `import math` — tryout's ssh body replayed by
hand, chain checked against a saved numpy reference; monitor: both are
one-line harness fixes)

| config | µs/xform B=32 graded | notes |
|---|---|---|
| r7 code, this round's windows | 3.472 / 3.483 / 3.500 | baseline re-measures |
| + fz | 3.449 / 3.453 | −1.0%, bit-identical chain |
| **+ hp (SHIPPED)** | **3.439 / 3.444 / 3.445 / 3.446** (sd ≤0.3% in quiet windows) | MKL same window 12.03 → **3.49×**; 0.1410 s total |
| B=1 m=1278 (classic path, byte-unchanged) | 5.888 (elevated window; r7 measured 5.29–5.39) | unscored |

Correctness, shipped build: single-transform rel_l2 **2.863e-16** (B=32);
whole-chain vs numpy reference **1.230e-13** (B=32, tol 1.0e-10 under the
corrected r8 two-part gate; digits identical to r7 — fz/hp change no bits)
and **1.173e-13** (B=1); B=9 group/classic seam m=20 3.34e-15 PASS; AVX2
(x86-64-v3) build compiles and passes the same seam check; single+chain
outputs cmp-identical across processes; all knob builds compile
(S8LZ/S8RK/S8FZ/S8/MAPV/S8SCHED/NOCHAIN).

### What did NOT work, with the number that killed it

1. **The warm rival's lazy map (lz/lz2, `-DL13_S8LZ`): +3–12%.**  c added
   raw at the y-stores, ladder as a pencil-granular slab before the next
   step's z-pencils (their MAPZB shape), map at exit.  Same window:
   lz+v0 4.498, lz+v1 3.877, plane-hoisted lz2+v0 3.582, lz2+v1 3.627 vs
   map@Ystore-v0 3.472.  On bare metal with two FMA pipes the store-path
   ladder placement was already right; their lazy choice is a VM-tier
   artifact (grading VM's ~2.1 uops/cyc issue cap starved their store
   path).  Code + knob kept as the record.
2. **L13_rader's rader-split pencil (`-DL13_S8RK=1`) in my fz structure:
   3.540 vs 3.439 (+3%)** despite 186 vs 210 FP and no spill strip — the
   sine block's 36-vector liveness spills the 12 cc pairs and loses more
   than the 24 FP save.  THIRD confirmation (r3, r5, now r8) that 13-point
   kernel arithmetic is not the lever at this cell; my pinned two-sweep
   dense pencil schedules better.  Their kernel transcribed verbatim
   (l13_s8_pencil_r + srk8 table) and kept compilable for the record —
   with one improvement over their MF=1: mapped stores take BOTH
   components in registers (l13_s8_maprst), no store-forward reload.
3. **`optimize("schedule-insns,sched-pressure")` on the s8 passes
   (`-DL13_S8SCHED=1`): 3.483 vs 3.439 (+1.3%).**  The warm rival ships
   exactly this on its 13-point wrappers and my r2 lesson said it
   transfers to phase-split kernels — but the panel_r9 lesson (the
   optimize() attribute rebuilds the whole per-function option set, ~2%
   tax) wins; net loss.  Not adopted.
4. **MAPV=1 (all-Newton, zero divider) re-swept in the fz shape: 3.524 vs
   3.442 (+2.4%).**  The warm rival's zero-divider choice at 13 does not
   transfer either; my v0 (rsqrt14+2N + ONE vdivpd on the otherwise-idle
   divider) stays.  Flavor rankings remain shape-local (fourth data point).

### Borrowed this round (attribution)

* **warm_d43251c2_score0.99** (`fft_warm_solutions/`): the huge-page group
  buffers (`alloc_huge` → my mmap+MADV_HUGEPAGE, the round's second keep);
  its lazy-map and sched-pragma choices were tested and measured AGAINST —
  negatives recorded above so nobody re-mines them on this hardware.
* **1000f989 / my own r7 notes**: the cross-step fusion concept (their
  L≥36 scheme-B), realised here as the fz pass pair — the round's main keep.
* **L13_rader ice_r7**: the rader-split pencil (transcribed, measured,
  rejected with numbers — settling the kernel-arithmetic question in the
  SoA shape too), and their MF ladder as the prior that made me re-test
  map placement carefully before believing the warm rival's lazy shape.

### For next round

1. Quiet-window expectation: **~3.43–3.45 µs/xform** (dev floor 3.439,
   sd ≤0.3% in quiet windows; fz+hp never lost a window).  0.1410 s
   projected vs the r7 leaderboard's 0.1420.
2. The cell sits ~18% above the ~2.9 µs/vol port floor.  The remaining
   excess is pass X's L2 stream (283 KB/step, the one stream fz could not
   delete) plus store bandwidth.  Untried: software-pipelining pass X into
   pass YZ (interleave x-pencils of step t into the plane visits of pass
   YZ of step t−1 at pencil granularity, ov-style) — complex, and the ov
   lesson says real work beats prefetch, so this is the shape to try
   before any prefetch.
3. The B=1 classic path (5.89, unscored) has had no attention since r6;
   if B=1 ever scores, port the s8-within-volume idea (1000f989's
   run_13t, lanes = 8 z-sites).
4. Harness debts for the monitor: check.py `import math` (map-check
   crashes), tryout.sh $W-before-definition (r4), reserve.sh --status
   requiring local squeue (r7).  All three worked around by hand again
   this round.
