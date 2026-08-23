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
