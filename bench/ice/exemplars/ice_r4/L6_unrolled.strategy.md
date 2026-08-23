# L6_unrolled — strategy record (Ice Lake panel)

Lineage: this entry's full history (rounds panel_r1…panel_r11, Cascade Lake node,
wallaby/Sapphire Rapids dev machine) lives at `bench/geom/strategies/L6_unrolled.md`,
and the multicore phase at `bench/mt/strategies/L6_unrolled.md`.  This file starts at
the Ice Lake panel.  There is no "Round ice_r1" section because the ice_r1 implementer
agent was killed by the worker-launch crash storm 15 seconds into the round (ice_r1
VERDICT §3); ice_r1's L6_unrolled numbers are a re-measurement of the unmodified
geom/impl_11 binary (0.219 µs/xform, B=64 chain m=4856, spread 3.2%, variant=fused_zp,
1.00× slot winner, 1.56× over MKL).

---

## Round ice_r2 (dev machine = the reserved Ice Lake node itself, via tryout.sh)

### Where ice_r1 left me

0.219 µs/xform on the graded chain (B=64, m=4856, working set 0.42 MiB), first of the
two L=6 entries (L6_pfa 0.220 with 13.5% run spread), 1.56× ahead of mkl_dfti (0.341).
No ice-round work had ever landed (r1 agent crashed).  The ice_r1 VERDICT's L=6
instructions: kill the plan race, then settle LITERATURE §4.1 — the premise that
blocked AVX-512 ("will spill a little") is dead on a 32-register machine, so try the
wider batch granule.  The VERDICT also established the two facts that reopen the width
question my geom record had closed 0-for-20: this node has **two** 512-bit FMA pipes,
and **clk512 = clk256** (2.90 loaded / 3.30–3.50 ramped, two independent in-plan
probes) — every zmm rejection in my record is a one-pipe Cascade Lake reading.

### What I changed

1. **AVX-512 reinstated as raced candidates** (my own r7 code, recovered from git
   `987d061` — the panel_r9 tree — rather than rederived): `zff` (zmm x-pass + fully
   fused zmm/ymm y+z per plane, 9 aligned zmm plane loads + 3 valignq, no t2 round
   trip, −25% uops vs `fused`), `zxf` (zmm x-pass + the node-proven ymm fused y+z,
   −17% uops), plus pf/pfw x-pass hook twins (`zff_pf`, `zff_pfw`, `zxf_pf`,
   `zxf_pf2`).  All trail the ymm incumbents at the full 2.5% margin (pf twins of zxf
   at 1.0%, bit-identical class), so a machine where width does not pay keeps the ymm
   pick exactly as before.
2. **The race is now chain-shaped** — ADOPTED FROM L17_matrixsimd's ice_r1 tuner fix,
   the one change that round that produced a measurable result.  The graded unit is
   not a bare transform: driver.c runs m chained steps, each `execute(src→dst)` +
   a driver-side unitary scale of the whole destination + a ping-pong of dst between
   two buffers.  My tournament used to time bare back-to-back `run(ain→aout)` calls;
   every race rep is now one driver-shaped chain step (transform, fence if NT,
   `*= 1/√216` over the destination, swap between aout/apong).  This matters: see
   the forced-variant table below — the bare race's preference order is wrong in the
   chain regime by more than the takeover margins.
3. **Deleted the answered-question apparatus**: the VD63 codelet, the probe-only
   `fused3_pfw`, and `l6_abL` (113 MiB, ~0.25 s of create()).  The r10 VERDICT's DRAM
   codelet A/B shipped on every ice_r1 leaderboard line (`abL=f434.3,f3440.3`,
   f3/f = +1.4%, inside the allocation-draw band).  The r5 lesson (dead code moved
   B=1 by 3.5%) says this deletion also pays for the zmm section's added text.
4. **Instruments retargeted to the width question**: `ab1` now times `fused_pf` (the
   chain-winning ymm shape) vs `zxf` (the chain-winning zmm shape) at nvol=1,
   licence-fair, published as `ab1=y…,z…`; the race table publishes `zwd` = best-zmm
   vs best-ymm delta at the graded batch (positive = 512-bit slower).  `xod` stays.
   Also added `fused_pf2`/`zxf_pf2` (distance-2 T0 hooks, pruned in r6 on CLX data,
   re-raced once here because the chain keeps all three buffers L2-resident), and a
   `-DL6_VERBOSE_DEFAULT` compile switch because tryout.sh cannot pass env vars over
   ssh.

### Operation count

Unchanged for the ymm kernels (PFA 2×3, 48 flops / 36 instr per line, no twiddles;
972 vector FP uops + 108 shuffles per volume).  `zxf` halves the x-pass: 9 zmm groups
× (16 arith + 2 shuffles) + 54 zmm loads + 54 zmm stores, then the identical ymm fused
y+z — ~1512 total uops/volume vs `fused`'s 1728, with x-pass loads/stores dropping
from 216×32B to 108×64B.  `zff` is ~1296 uops/volume but concentrates its z-transpose
in 512-bit `vpermt2pd`/`valignq`, which are port-5-only on ICX — the same port the
second 512-bit FMA pipe lives on (the r1 VERDICT's L=17 mechanism).

### What was measured (all on the node via tryout.sh, one leased pinned core,
graded chain L=6 B=64 m=4856 unless stated; rel_l2 = 2.428e-16 and chain check PASS
at 2.904e-13 in every run; output bit-identical across runs)

**Forced-variant chain table** (compile-time `L6_FORCE_DEFAULT`, sd ≤ 0.16% per run
unless noted; taken across ~30 min, so neighbor-lease drift of a few % between rows
is possible — the *within-run* discipline is what's tight):

| variant | µs/xform | note |
|---|---|---|
| fused_pf | **0.220** | best ymm |
| fused | 0.224 | sd 6.1% |
| fused_zp | 0.226 | the ice_r1 pick |
| 3pass, zff, zff_pf | 0.239 | |
| fused_pfw | 0.241 | prefetchw +9% vs fused_pf |
| zxf | 0.243 | see below — contention-insensitive |
| 3pass_pf | 0.248 | |
| fused_zp_pf | 0.250 | |
| zff_pfw | 0.268 | |
| 3pass_nt_pf | 0.765 | NT stores 3.5× — chain keeps out L2-resident |

**Chain-shaped race tables** (two `-DL6_VERBOSE_DEFAULT` runs ~15 min apart,
row-for-row reproducible to ±0.4%, under visible neighbor load): zxf **0.2433 —
chosen** (cleared the 2.5% margin over fused_zp_pf 0.2497 / fused_pf 0.2502 by a
hair); zff 0.272; zxf_pf 0.2434, zxf_pf2 0.2443 (hooks buy nothing on zxf);
fused_pf2 0.2509 (distance-2 no better than distance-1); pfw family 0.274–0.286;
NT 0.77–0.80.

**Auto (tournament) graded-chain results across five invocations**: 0.213 (median
0.232, busy), 0.213/0.213 sd 0.03%, 0.240/0.258 (one busy window), 0.214/0.214 twice.
Best-vs-ice_r1: **0.219 → 0.213 min (−2.7%), median 0.232-class → 0.213-class, spread
3.2% → 0.03% in quiet windows**.  MKL same case, same core: 0.341–0.347 → **1.60×**.
B=1 chain: 0.188 µs/xform (sd 0.06%); B=4096 chain: 0.556 min (DRAM regime, spread
7.4% under neighbor load).

**The finding the round should keep — the tuner regime, with the number that proves
it**: the bare (pre-ice_r2) race prefers fused_zp; the chain regime measures fused_zp
at 0.226 vs fused_pf 0.220 (+2.7%, twice the pf-family margin) and the ice_r1
leaderboard indeed shows `variant=fused_zp` scoring 0.219.  A race that does not
reproduce the scale-pass + ping-pong cache state mis-picks by more than its own
hysteresis.  Second finding: **zxf's chain time is contention-insensitive** —
0.243 forced in a quiet-ish window, 0.2433 raced under load, 0.214 measured when the
sample window went quiet — while the ymm kernels swing 0.220→0.250 with neighbor
load.  The halved load/store instruction count is the plausible mechanism.  The
drained scoring window will adjudicate via `zwd` on the leaderboard line.

### What was tried and did NOT work (with the number that killed it)

1. **zff — the "more 512-bit is better" shape**: 0.239–0.272 in every regime, never
   within 6% of zxf's race time.  The y+z half's 512-bit transposes are port-5-only
   and displace the second FMA pipe — the r1 VERDICT's L=17 port-5 mechanism
   reproduced at L=6.  Width pays exactly where it does NOT add p5 pressure (the
   x-pass, stride-72 aligned columns: zero extra shuffles).
2. **prefetchw in the chain**: fused_pfw +9% over fused_pf, zff_pfw +12% over zff_pf.
   Confirms L13_rader's ice_r1 gate finding (`pw` pays only past L3; the chain keeps
   `out` L2-resident) — on this workload it is not merely useless but a 9% tax.
   Kept raced; the chain-shaped race now sees exactly this and will never pick it.
3. **Distance-2 T0 prefetch** (fused_pf2, zxf_pf2): 0.2509 vs 0.2502 / 0.2443 vs
   0.2433 — one volume of lead is already enough on this node; two is marginally
   worse.  The r6 CLX prune stands on ICX.
4. **NT stores**: 0.765 vs 0.220 (3.5×) — the third microarchitecture and second
   panel to reject them in a cache-resident regime; they survive only as the raced
   canary.

### Borrowed / lent

Borrowed: **the chain-shaped tuner stage from L17_matrixsimd** (ice_r1, attributed
above — their fix for exactly this mis-pick class); the port-5 mechanism used to
explain zff's loss is **L17_matrixsimd's + L13_rader's ice_r1 finding**; the
prefetchw-past-L3-only gate check is **L13_rader's ice_r1 measurement** (this round
adds the L=6 number: +9%).  Recovered rather than borrowed: my own r7 zmm kernels
from git history — read them out of `git show 987d061:bench/geom/impl_9/L6_unrolled.c`
instead of rederiving; the panel_r9 tree is the last one with the full zmm section.
Lendable: (a) the chain-shaped race pattern is one screenful and applies to EVERY
entry on this panel — any tuner still racing bare transforms is optimizing the wrong
workload by up to 3% (my measured mis-pick) and possibly more at larger L where the
scale pass is a bigger fraction; (b) the zmm x-pass/ymm y-z split (width only where
it adds no p5 pressure) should port directly to L6_pfa's PASS_X and is worth a look
at L=8 (L8_batchsimd's x-pass has the same aligned-column structure).

### Node predictions (falsifiable via the description string and cell times)

* The scored window is drained, so the race will read the quiet regime.  If the
  quiet-regime order matches my quiet forced table, the pick is `fused_pf` and the
  cell lands 0.213–0.220 with `zwd` ≈ +8..+10%.  If zxf's contention insensitivity
  was actually the whole story and quiet zxf ≈ 0.214 (the two quiet auto windows),
  the pick is `zxf` and `zwd` reads ≤ −2.5%.  Either way the description carries the
  width answer; I bet 55/45 on `zxf` being picked (both 0.213–0.214 quiet readings
  came from invocations whose race chose zxf).
* Spread: the chain-shaped pick should hold the ice_r1 3.2% run spread at ≤1% in the
  drained window (the quiet dev runs read 0.03–0.06%).
* kclk ≈ 3.3 (ramped single core, no licence cliff); ab1 y vs z within ±5% of each
  other at nvol=1 (zxf's zmm x-pass is latency-neutral at one volume).
* Correctness: rel_l2 2.4e-16, chain 2.9e-13, unchanged fingerprints.

### Next

1. Read `zwd`/`ab1` from the scored line.  If zxf is picked and wins the cell, flip
   zxf to incumbent in r3 (the r10/r11 incumbency-flip playbook) and delete the zff
   family (three kernels, keep the one-line finding); if fused_pf is picked, delete
   zxf_pf/zxf_pf2 (hooks proven useless on zxf) and keep zxf as the standing width
   canary.
2. Port the chain-shaped race to whatever else I still own if the panel does not do
   it wholesale — and say so in context for L6_pfa, whose 13.5% ice_r1 spread is the
   same disease this round cured here.
3. The unexplained residual: quiet-window measured 0.213–0.214 vs forced-quiet
   fused_pf 0.220 — the tournament's picks reach numbers forced runs of ANY single
   variant did not reproduce in their own windows.  Most likely pure window luck
   (neighbor leases), but if the scored cell lands ≤0.210 it is worth one round of
   create()-state archaeology (does the 2 s of racing pre-warm L2/TLB in a way the
   0.056 s forced setup does not?).

---

## Round ice_r4 (dev = the reserved node via tryout.sh; 2026-08-23)

### The task changed: this round is the fused chain

The graded step is now the rivals' full step, `state <- (z+c)/(1+|z+c|)`, `z` = RAW
FFT(state) (no unitary scale anywhere in map mode — verified in driver.c RUN_UNIT and
check.py), and the driver detects an exported `fft3d_chain` (weak symbol) that owns the
whole m=4856-step chain.  Where ice_r3 left me: 0.213 µs/xform FFT-only, variant=zxf,
first at L=6.  MKL through the new driver-side fallback map reads 0.942 µs/xform on the
full step — the map costs 3× the FFT if unfused.  Rivals' L=6 target: 0.102 s / 4856 / 64
= 0.328 µs/xform (achieved with the ILLEGAL float-seed map, per the brief).

### What shipped

1. **`fft3d_chain`, VOLUME-MAJOR, IN-PLACE** (ADOPTED from L17_matrixsimd's ice_r4 §1,
   executed at the panel's best-case geometry): each volume runs all m=4856 steps while
   its state (3.4 KB), c slice (3.4 KB), t1 (3.4 KB) and map staging (3.4 KB) sit in the
   48 KB L1 with 3× headroom.  `final_out` is the state arena; step 1 is a plain zxf of
   x0 (const, read once); L2/DRAM traffic for a volume's whole 4856-step chain is one x0
   read, one c read, one writeback.  In-place is legal because the x-pass drains the
   volume into t1 before the first y+z store.
2. **The map, pair-shared, exact tier** (ladder from L17_matrixsimd's s6 + L13_rader's
   MAPSTYLE=1, extended): per 8 points — deinterleave (2 vunpck), s = im²+(re²+1e-300)
   (2 FMA, guard folds L13's sp==0 trap into an addend), w ≈ 1/√s via vrsqrt14pd + 2
   Newton, **FMA-Heron residual step r ← r + (s−r²)·(w/2)** (see below), d = 1+r, **ONE
   vdivpd per 8 points** (y = 1/d, exact), two mul-outs, reinterleave.  22 vector uops +
   1 divide per 8 points.
3. **Style race, decided on the node — default is bdiv (PHASE-SPLIT), not the lazy map**:
   the map runs as its own 27-pair zmm pass into a staging volume, then the untouched zxf
   step.  Forced-style table (graded chain B=64, tryout):
   | style | shape | µs/xform |
   |---|---|---|
   | **bdiv (default)** | phase-split map pass + zxf, shared vdivpd | **0.331–0.332** |
   | adiv | map fused into the zmm x-pass loads (lazy) | 0.360–0.408 |
   | bfma | phase-split, rcp14+Newton (no divider) | 0.374 |
   | afma | fused, rcp14+Newton | 0.396 |
   | bdiv −Heron | 2-Newton only (biased, see below) | 0.299 |
   The lazy map — L13_rader's ice_r4 winner and the rivals' fusion — LOSES at L=6 by
   9–23%: the ~100-cycle ladder+divide chain sits in front of every x-group's codelet
   (VD6Z needs all 6 vectors, so each group waits for its slowest pair), and the OoO
   window does not bridge nine of those in a 3.4 KB volume.  This is L17_matrixsimd's
   ice_r4 latency-exposure mechanism reproduced at the smallest geometry; their rule
   ("the map needs a following pass to hide in") holds here with the map AS the
   preceding pass instead.
4. **Chain scratch: closed-form 4K placement + a per-volume c copy.**  B=1 measured 13%
   slower than B=64 (0.375 vs 0.332) with identical code — the only difference is the
   driver's allocation residues.  c is read-only, so each volume's slice is copied once
   (3.4 KB per 4856 steps, ~0.02%) into a plan arena where t1 / staging / c-copy sit at
   residues sp+1024 / +2048 / +3072 (mod 4096): every controllable store→load delta is
   pinned at ≥1024 B, no search.  B=1 went 0.375 → 0.332, now equal to B=64.

### Operation count (per volume per chain step, default bdiv)

Map pass: 27 pairs × (4 ld + 2 add + 4 unpck + 12 ladder/Heron + 1 fmadd + 2 mul +
2 st) + 27 vdivpd ≈ 675 vector uops + 27 divides (~432 divider cycles, hidden under the
FFT's port work).  FFT: unchanged zxf, ~1512 uops.  Per-step ≈ 2200 uops + 108 extra
staging ld/st ≈ 0.332 µs measured (~960 cyc @2.9 GHz).  Whole-chain traffic per volume:
x0 + c read once, state written back once — everything else L1.

### Measured (node, graded chain L=6 B=64 m=4856 unless stated)

- **SHIPPED (bdiv): min/median 0.331–0.332 µs/xform** (sd 0.01–3% by window); **B=1
  0.332** (sd 0.01%).  MKL same case, same core, through the driver fallback: 0.942 →
  **2.84×**.  vs rivals' full-task 0.328 µs/xform: parity, with a full-double map they
  did not have.  vs my own unfused ice_r3 config: MKL's fallback ratio implies ~0.9+
  µs/xform unfused — the fusion is worth ~2.8×.
- Single-transform rel_l2 2.428e-16 (B=64), 2.342e-16 (B=1), PASS.
- Bit-repeatable: two independent runs, .chain and single outputs cmp-identical.
- Map ladder accuracy (node microbench, 16M points, vs long double): ladder+div max
  5.36e-16, ladder+rcp 6.55e-16, hw sqrt+div 4.04e-16 — all ~2-3 ulp class.
- Whole-chain drift of the shipped binary vs numpy: m=320: 3.6e-14 · m=1278: 5.0e-13 ·
  m=2572: 2.5e-11 · m=3600: 2.9e-10 · m=4856: **1.40e-9 vs tol 4.86e-10 — FAIL** (see
  next section: so does everything else, including the harness's own reference).

### THE ROUND'S CENTRAL FINDING: the L=6 map-chain gate is below the task's noise floor

The m=4856 gate (tol = 1e-13·m = 4.86e-10) cannot be passed by ANY independent
implementation.  Evidence, all reproducible from build/tryout/L6_unrolled/:

1. **The harness's own reference fails it**: mkl_dfti through the driver's own fallback
   map → rel_l2 = 1.759e-9 (3.6× over).  My exact-scalar style (libm sqrt, same formula
   as driver MAP_STEP) → 1.955e-9.  My shipped ladder → 1.403e-9 (the best of the
   three).
2. **Pure-numpy pricing (no C anywhere)**: with numpy's OWN fftn, replacing only
   `np.abs(z)` by `sqrt(z.real²+z.imag²)` drifts 1.12e-9; an 80-bit CORRECTLY-ROUNDED
   |z| (more accurate than numpy's glibc hypot) drifts 1.56e-9; random per-step z-noise
   at FFT-reassociation scale (2.4e-16) drifts 1.58e-9.  Reciprocal-multiply vs true
   division: 0.0 (numpy's complex-by-real divide IS reciprocal-multiply).  So the gate
   anchors to pocketfft's and glibc-hypot's specific bit patterns, not to accuracy:
   being MORE accurate than numpy also fails.
3. **Mechanism — weak chaos, not accumulation**: drift grows ~4e-16·m to m≈2500, then
   exponentially with a ~450-step e-fold (2.5e-11 → 2.9e-10 → 1.4e-9 over
   2572→3600→4856).  e^(4856/450) ≈ 4.8e4 — exactly the brief's "1-ulp ends at 4.8e-12"
   conditioning number; the linear 1e-13/step budget and that transient measurement
   both miss the late chaotic regime, where end-state noise ≈ ulp × 450·e^(m/450) ≈
   2e-9.  (A single 1-ulp perturbation of one input point ended BIT-IDENTICAL after
   m=4856 in numpy — per-volume dynamics are a mix of contracting and chaotic under
   this c field, so single-point transients can die while distributed per-step noise
   always excites the chaotic volumes.)
4. **Consequences elsewhere**: at L=8's m=2572 the same physics gives my binary
   2.5e-11 vs tol 2.6e-10 — an exact-tier L=8 entry should still pass with ~10×
   margin, but the margin is chaos-limited, not ladder-limited.  L=13 m=1278 (5.0e-13
   here) matches L13_rader's measured 1.19e-13 — the regime is pre-chaotic there.
   MONITOR ACTION NEEDED at L=6: either shorten the graded chain to m ≲ 3000, or set
   the tolerance from the measured floor (reference-fallback 1.76e-9 → e.g. 5e-9), or
   gate on a mid-chain state (any m ≤ 2572 checkpoint separates exact from
   float-seeded maps by 4+ orders: my 2.5e-11 vs their ~1e-12/step × amplification).
   The float-seed cheat the gate exists to catch is still caught at ANY of these: its
   1e-12/application bias explodes to 1.2e-8+ by the same mechanism.

### The Heron step (kept) and the bias mechanism

First build used rsqrt14 + 2 plain Newtons: chain drift 2.29e-9.  Newton converges from
below, so w carries a ~5e-17 SYSTEMATIC underestimate — unbiased rounding noise excites
the chaos incoherently, but a bias is a persistent forcing and costs a measured extra
~0.9e-9.  The FMA-Heron residual step (r ← r + (s−r²)·(w/2), the exact residual via
fnmadd) makes sqrt bias-free (~1e-32) at +3 ops/pair: drift 2.29e-9 → 1.40e-9.  It
costs 10% (0.299 → 0.331) — kept anyway, because 1.40e-9 is BELOW the harness's own
reference floor (1.76e-9) and 2.29e-9 is above it: whatever recalibration the monitor
picks, an entry that beats the reference backend's drift is defensible.
-DL6_MAP_NOHERON is the one-flag speed lever if the recalibrated gate is generous.

### What did NOT work, with the number that killed it

1. **The lazy map (rivals' + L13_rader's fusion), at L=6**: adiv 0.360–0.408 vs bdiv
   0.331.  Latency exposure (above).  Do not re-fuse the map into the x-pass here
   without software-pipelining it across groups (est. ceiling ~0.30, not attempted —
   two groups in flight ≈ 40 live zmm → spills eat the margin).
2. **All-FMA reciprocal (no divider)**: bfma 0.374, afma 0.396.  27 divides/volume
   (~432 divider cycles) hide under ~960 cycles of port work; +4 uops/pair does not.
   Same verdict as L17's s6-vs-s4 and L13's MAPSTYLE=2, at a third geometry.
3. **Naive 4K handling at B=1**: 13% tax (0.375 vs 0.332) from driver-allocation
   residues — fixed by the per-volume c copy + closed-form residue pinning, not by a
   search.
4. **vsqrtpd anywhere**: not even tried this round — L17 (23.76 vs 13.0) and L13
   (7.498 vs 6.390) both measured the divider-occupancy catastrophe; corpus verdict
   accepted.

### Borrowed this round (attribution)

- **L17_matrixsimd ice_r4**: volume-major in-place chain with final_out as state arena;
  pair-shared denominator with one vdivpd per 8 points (s6); vrsqrt14pd is ~2.3 cyc on
  bare-metal ICX (their falsification of corpus §10's "microcoded" claim); the
  latency-exposure warning about lazy maps, confirmed here; the tryout.sh `$W`
  workaround (W=... env prefix + manual check.py + manual repeatability cmp).
- **L13_rader ice_r4**: the rsqrt14+2-Newton |w| + exact-divide exact-tier recipe and
  its budget arithmetic; the sp==0 → rsqrt(0)=inf NaN guard; the lazy-map shape itself
  (raced, lost here — geometry-dependent, not wrong).
- **Rival pipelines via corpus §10 §2**: burn the divider once, Newton on the FMA pipes;
  their float-seed trap is re-confirmed from the other side (even EXACT maps fail the
  current gate — their 1.28e-8 is the same chaos with a 1e-12 forcing).

### Next round

1. If the gate is recalibrated: -DL6_MAP_NOHERON buys 10% (0.331 → 0.299) the moment
   the tolerance clears ~2.5e-9; re-check drift first.
2. The remaining speed levers, in expected order: software-pipelined adiv (map ladder
   of group g+1 issued ahead of codelet g; est. −5-8% if spills stay bounded); a
   fully split-complex (deinterleaved) chain state — kills all 4 map unpcks AND the
   codelet vpermilpds, ~200 p5 uops/volume, but is a full kernel rewrite; check
   L8_radix8's split-cplx experience first.
3. The FFT half is untouched since ice_r3 — if anyone lands a faster L=6 FFT shape,
   the chain inherits it through l6_run_zxf unchanged (one call site).
4. B=1 and B=64 are now identical per volume (0.332): the chain has no batch regime
   left to tune; only per-step work remains.
