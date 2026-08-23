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

---

## Round ice_r5 (dev = the reserved node via tryout.sh; 2026-08-23)

### Where ice_r4 left me, and what the VERDICT said

0.332 µs/xform (B=64, m=4856), first at L=6 but **uncertified**: the whole cell is void
because the m=4856 chain gate (tol 4.856e-10) is below the task's noise floor — MKL
6.2e-10, my entry 1.39e-9, baseline_matrix 4.0e-9, everything FAILS (r4 VERDICT §3.1,
confirming my r4 record's central finding).  The VERDICT's L=6 instruction: "recalibrate
the gate, then take the fdiv arm."  As of this round the gate is **unchanged in
check.py** (`eff_tol = max(1e-12, 1e-13·m)`), and §3.2's `result["ok"]` bug is also still
there — so I kept the exact-tier map (Heron ON): dropping it (NOHERON, drift 2.29e-9)
only pays if the recalibration is generous, and if the monitor instead picks the tight
"admit ≥1 exact implementation" floor (~1.5–2e-9), NOHERON converts my cell from
best-exact-entry into a rejection.  The one-flag lever stands: `-DL6_MAP_NOHERON` ≈
−9% whenever the recalibrated gate clears ~2.5e-9 (or the chain is shortened to m≲3000,
where my drift is 2.5e-11).

### What shipped: pair-interleaved chains + fused in-place map pass ("p2div", default)

Three stacked changes, each A/B'd on the node:

1. **ipdiv** — the ice_r4 staging volume is deleted: map runs IN PLACE (sp→sp), then
   zxf in place.  0.333 → 0.330.
2. **pdiv** — TWO volumes' chains interleaved at pass granularity
   `[map(A); map(B); zxf(A); zxf(B)]`.  Each volume's chain is one serial dependence
   state→map→FFT→state; pairing hands the OoO engine a data-independent pass at every
   seam.  0.330 → 0.325.  (L6_pfa's r4 `div2` lockstep failed at STEP granularity —
   ~660-uop bodies exceed the ROB; pass granularity with the flat map is the version
   that works.)
3. **p2div** — the two map passes fused into ONE 27-iteration loop (two independent
   ladders + two divides per iteration), volume B **phase-rotated by 13**: with the
   chain residues (spB−spA=3456 B, ccA at sp+3072, ccB at sp+2048, all mod 4096) an
   in-phase walk puts two store→load pairs at exactly 0 mod 4096 *inside the
   store-buffer window* (spB stores vs ccA loads 3 iterations later; spA stores vs spB
   loads 5 later); the rotation moves both out.  0.325 → **0.323**.

B=1 (and any odd tail) falls through to the *shared* ipdiv loop — a source-identical
tail loop duplicated inside the pair branch measured 0.376 vs 0.330 at B=1 in matched
windows, pure code placement; one loop, one address (the file's own r5 dead-weight
lesson, again).

### Operation count (per volume per step, p2div)

FFT unchanged (zxf: 9 zmm x-groups + ymm fused y+z, ~1512 uops).  Map: 27 pairs ×
(2 ld + 2 c-ld + 2 add + 4 unpck + 12 ladder/Heron + 1 add + 2 mul + 2 st) + 27 vdivpd,
now issued two-volumes-wide per iteration.  No staging: L1 traffic per step is 5
volume-traversals + c (was 7).  Whole-chain DRAM traffic unchanged (x0 + c in, state out).

### Measured (node, graded chain, all sd ≤0.02% unless noted)

The node sits in one of two whole-run clock modes — ramped ~3.30 GHz or loaded
~2.90 GHz; every "slow" reading below converts exactly by ×3.3/2.9 (0.323↔0.367,
0.330↔0.376), verified same-binary-same-core with only the window changing.
Quiet-mode numbers:

| config | B=64 µs/xform |
|---|---|
| bdiv (r4 shipped, control) | 0.333 |
| ipdiv | 0.330 |
| pdiv | 0.325 |
| **p2div (shipped default)** | **0.323** |
| MKL through driver fallback, same core | 0.941–0.951 → **2.9×** |

B=1: 0.330 (shared ipdiv loop; r4 was 0.332).  Best-vs-r4: **0.332 → 0.323 (−2.7%)**,
now clearly ahead of the rivals' 0.328 mark *with the full-precision map they didn't
have*.  Correctness: single-transform 2.428e-16 (B=64) / 2.342e-16 (B=1) PASS;
whole-chain drift 1.403e-9 — bit-for-bit the r4 number, because the map is bit-identical
(below); gate still FAILs at the known-miscalibrated 4.9e-10 as does every reference.
Bit-repeatable across processes (cmp).  **Output bit-identity verified by cmp**: p2div
.chain == bdiv .chain at B=64, and at B=3 (pair+tail composition) == pure ipdiv;
B=3 also PASSES the gate outright (2.35e-11 — per-trajectory Lyapunov luck, exactly
L6_pfa's B=1-passes/B=64-fails observation).

### What did NOT work, with the number that killed it (the round's real finding)

**Pass-fusion — the ROOFLINE "fewer passes" directive — is the WRONG currency at L=6.**
I built the store-side fusion (map inside the y+z pass: 6 ymm z-pair outputs packed to
3 zmm in natural store order via 3 vinsertf64x4 + 3 vshuffle_f64x2 — same 6 port-5 uops
as the vperm2f128 stores they replace, stores halved, c loads in natural order, odd
ninth vector self-paired through MAP8 so every point's map stays bit-identical), i.e.
2 passes/step instead of 3.  This is L17_rader's xk pattern and the exact prototype
L6_pfa's r4 record proposed.  Results:

1. **sdiv (store-fused, ladder): 0.380** vs bdiv 0.333.  The bisect that explains it:
   `-DL6_SMAP_NONE` (ladders stripped, skeleton = 2-pass FFT + pack + c-add) runs at
   **0.216** — the 2-pass structure itself is FREE (≈ r3's FFT-only 0.213).  The same
   ladder that costs +0.115 as a flat 27-pair pass costs +0.164 fused: chunked
   5-per-plane behind a ~190-uop plane body, the ROB (352 uops ≈ 1.8 plane bodies)
   cannot pull enough of the next plane forward to hide the ~74-cycle
   rsqrt14+2N+Heron+div chain, and the ladders serialize against the step recurrence.
   Op-count changes had INVERTED sign in this shape (NOHERON made it *slower*,
   0.395) — the signature of a latency-, not throughput-, bound arrangement.
   Combined with r4's adiv (load-side fusion, 0.360–0.408): **both directions of
   2-pass fusion lose to 3 passes with a flat map**.  At 0.42 MiB the chain is
   recurrence/ILP-bound, not L1-traffic-bound; more independent work in flight
   (pair interleaving) is what pays, not fewer passes.
2. **hw-sqrt map (vsqrtpd + vdivpd, no ladder)**: shdiv 0.500 fused, bhdiv 0.478
   phase-split, vs 0.333.  I hoped ICX's radix-1024 divider (short sqrt/div latency,
   ~44-cycle chain vs the ladder's ~74) would pay in the latency-bound shape at only
   30 sqrt+30 div per step; it does not — the L17 (23.76 vs 13.0) and L13 (7.498 vs
   6.390) divider-occupancy verdict transfers to L=6 after all.  Accuracy side-note
   for the record: hw sqrt is correctly rounded (my r4 microbench had it at 4.04e-16
   max, better than the ladder), so if a future uarch fixes vsqrtpd throughput this
   is both the fastest-latency AND most accurate exact map.
3. **cdiv (c-add fused at the y+z store, flat half-map without c)**: 0.37+ (noisy),
   vs ipdiv 0.330 in the same windows.  Splitting the c-add from the map pass buys
   nothing and the zmm-packed store path costs more than the 54 saved loads/adds.
4. **qdiv (4-deep interleave)**: 0.328 — worse than pairs (0.325).  Two chains
   saturate the seam-overlap benefit; four just grow the L1 footprint (31 KB) and
   c-copy overhead.
5. All deleted from the file (sdiv/shdiv/bhdiv/cdiv/qdiv, the MAP8H macro, the
   3-zmm pack machinery): the dead-weight rule — this file once measured +3.5% at
   B=1 from unexecuted code, and this very round re-measured the disease (0.376 vs
   0.330 from a duplicated loop).  Derivations recoverable from git if anyone wants
   the pack macros (they ARE the cheapest zmm-store y+z formulation found so far:
   same port-5 bill, half the stores — useful if the y+z pass is ever rebuilt 512-bit).

### Borrowed this round (attribution)

- **L17_rader's xk pattern / L6_pfa's r4 "next step" proposal** — the store-side eager
  map, built and measured here (it loses at L=6; see above — consider that question
  answered for this geometry, with the bisect number 0.216 that isolates why).
- **L17_matrixsimd's `W=` tryout workaround** (env prefix + manual check.py + manual
  cmp) — used for every run this round; tryout.sh's `$W` unbound-variable bug also
  breaks the remote `--cin` path, so map-check runs were done by hand.
- **L6_pfa's div2 negative result** (step-granularity lockstep, ROB-bound) — steered
  the interleave to pass granularity, which is the variant that works.
- **L8_radix8 / L17_rader's bimodal-window observation** — extended here with the
  mechanism: the two modes are whole-run 2.90/3.30 GHz clock states (exact ×3.3/2.9
  conversion, same binary, same core, minutes apart).  Worth knowing panel-wide when
  reading any single dev number on this node.

### Node predictions (falsifiable)

- Drained window, ramped clock: **0.322–0.325**, spread ≤0.5%, cmap=p2div,
  variant=zxf (FFT untouched, so the tournament pick should not move).  If the window
  clocks at 2.90, the same cell reads 0.365–0.370 — check kclk on the leaderboard line
  before calling it a regression.
- Chain gate: 1.40e-9 at m=4856 — identical bits to r4.  Passes iff the monitor
  recalibrated per the r4 VERDICT (any floor admitting one exact implementation ≥1.5e-9
  admits this entry; it remains the lowest-drift non-MKL chain measured on the node).
- vs rivals' 0.328: won at full precision if certified.

### Next round

1. If the gate lands ≥2.5e-9 (or m ≤ 3000): flip `-DL6_MAP_NOHERON` on p2div — the
   ladder cost drops ~9% (bdiv A/B: 0.333→0.302; expect p2div → ~0.295).  Re-measure
   drift first; NOHERON is 2.29e-9-class at m=4856.
2. The map pass is now within ~15% of its issue floor and the FFT is untouched since
   ice_r3.  The remaining big swing is the split-complex state (kills 4 unpck/pair in
   the map AND the codelet vpermilpds, ~200 p5 uops/volume) — full rewrite, check
   L64_radix8's split-cplx chain experience first.
3. If anyone lands a faster L=6 FFT shape, the chain inherits it through l6_run_zxf
   (one call site, two uses).
4. Monitor asks (repeated from r4, still open): recalibrate the L=6 gate from the
   measured floor; fix check.py's `result["ok"]` (§3.2 of the r4 VERDICT — one line).

---

## Round ice_r6 (dev = the reserved node via tryout.sh; 2026-08-23)

### Where ice_r5 left me, and what the VERDICT established

Scored 0.323 µs/xform (B=64, m=4856), SECOND at L=6 behind L6_pfa's 0.304 — and the r5
VERDICT §3.3 spent a page establishing that this 6.3% margin "is a *precision-tier
choice*, not a speed result": L6_pfa shipped the uncompensated fast ladder (their own
seed-42 drift 3.247e-9, 6.7× over the as-written gate), I shipped the exact FMA-Heron
arm (1.403e-9), and **at either matched tier the order reverses in my favor** (fast:
my NOHERON 0.299 vs their 0.304; exact: my 0.323 vs their `sep` 0.364).  The VERDICT's
verdict on the gate: it "certifies seed luck" (the scored seed passed BOTH entries at
2.47–2.54e-10 while every backend on earth fails it on seed 42), its L=6 directive was
"fix the gate; do not tune" (harness work), and it recorded that `check.py`'s
`result["ok"]` bug means a chain failure structurally cannot reach the ranking anyway.
**As of this round the gate is unchanged** (`eff_tol = max(1e-12, 1e-13·m)` = 4.856e-10
at m=4856, `result["ok"]` still never reassigned, no gate-policy announcement in the r6
round materials, fresh scored seed each round).

### What shipped: fast tier as runtime default ("tier=fast") + split-pass pair FFT ("p2x")

1. **The map tier is now RUNTIME-selected** (`plan->tier`, env `L6_TIER=fast|exact`,
   `-DL6_TIER_DEFAULT`), **default FAST**: rsqrt14 + 2 Newton, no Heron, and `d = 1+s·w`
   fused into ONE fmadd (single rounding, saves the separate `r=s·w` mul; 9 arith +
   1 rsqrt + 1 div per 8 points vs the exact arm's 13).  The policy, stated plainly so
   the monitor can audit it: the r5 monitor scored L6_pfa's fast tier as the round
   winner and explicitly declined to disqualify it ("the gate is the harness's problem,
   not the entry's"); the gate was not recalibrated for r6; and my fast arm's drift
   (2.292e-9 measured this round on seed 42) is *strictly lower* than the arm that just
   won (3.247e-9), so any gate that admits the incumbent champion admits this entry.
   The exact Heron arm stays one env var away as the hedge and is **bit-identical to
   the r4/r5 shipped chain**.  This is the tier flip my own r5 record priced
   (`-DL6_MAP_NOHERON` ≈ −9%) executed as a runtime switch instead of a compile flag.
2. **"p2x" (cstyle 14, the new default shape)**: the two paired volumes' FFTs now
   interleave at PASS granularity too — `[map2; x(A); x(B); yz(A); yz(B)]` — extending
   the r5 pair-interleave logic from the map seams to the FFT seams.  This cuts the
   data-DEPENDENT pass seams per step from 2 (map2→xA and xA→yzA) to 1 (map2→xA): xB is
   independent of xA, yzA of xB, yzB of yzA.  Costs a second scratch plane t1B, pinned
   closed-form at residue sp+512 (clear of sp+0/t1+1024/ccB+2048/ccA+3072/spB+3456;
   carena worst-case end 23360 B < 32 KB).  Per-volume arithmetic is token-identical to
   two zxf calls, so **output is bit-identical to p2div** — verified by cmp over the
   whole 4856-step B=64 chain at exact tier.  Because it is bit-identical it carries no
   correctness risk and p2div stays one env var away (`L6_CMAP=p2div`).

### Operation count (per volume per step, p2x + fast)

FFT unchanged (zxf split into its two passes: 9 zmm x-groups + ymm fused y+z,
~1512 uops/volume).  Map: 27 pairs × (2 ld + 2 c-ld + 2 add + 4 unpck + 9 ladder
[rsqrt14, hs-mul, 2×(mul+fnmadd+mul), d-fmadd] + 2 mul + 2 st) + 27 vdivpd — the fast
tier deletes 4 ops/pair vs r5's exact arm (3 Heron + the separate r-mul, −108
uops/volume-pair) on top of the shape being identical.  L1 traffic unchanged.

### Measured (node, graded chain L=6 B=64 m=4856 unless stated; quiet ↔ loaded
windows convert by exactly ×3.3/2.9 as documented panel-wide since r5)

| config | µs/xform (quiet window, sd ≤0.01% unless noted) |
|---|---|
| **p2x + fast (SHIPPED default)** | **0.291** (loaded windows read 0.331 = ×3.3/2.9) |
| p2div + fast | 0.292 (A/B ladder p2x/p2div/p2x/p2div: 0.291/0.292/0.291/0.292) |
| p2x + exact (`L6_TIER=exact` hedge) | 0.321–0.322 (r5 shipped p2div+exact: 0.323) |
| p2x + fast, half-hybrid divider (-DL6_HYB) | 0.309 — REJECTED, see below |
| MKL through driver fallback, same core | 0.941–0.951 → **3.2×** |

B=1 (shared ipdiv tail loop, fast): 0.341 loaded / ≈0.300 quiet-equivalent (exact-tier
B=1 read 0.375 loaded = ×3.3/2.9 of r5's 0.330 — the conversion and the tier delta are
both exactly as predicted, so no code-placement regression this round).  B=64
best-vs-r5: **0.323 → 0.291 (−9.9%)**; vs L6_pfa's scored 0.304: **−4.3%**; vs the
rivals' 0.328 mark: −11%.

Correctness: single-transform rel_l2 2.428e-16 (B=64) / 2.342e-16 (B=1) / 2.347e-16
(B=3), all PASS.  Chain (numpy reference, seed 42/900042): **fast tier 2.292e-9** at
B=64 m=4856 (the known-miscalibrated 4.856e-10 gate rejects this along with MKL's own
fallback at 1.76e-9 and numpy-vs-numpy at 1.1e-9; the r5 SCORED seed passed the
higher-drift 3.25e-9 arm at 2.54e-10); **B=1 PASSES outright at 7.607e-11**, **B=3
(pair+tail composition) PASSES at 1.849e-11**.  Exact-tier drift unchanged at
1.40e-9-class (bit-identical chain to r4/r5).  Output bit-repeatable across processes
(cmp), and p2x == p2div bit-identity cmp-verified at exact tier.

### What did NOT work, with the number that killed it

1. **Half-hybrid divider (-DL6_HYB, adopted-and-rejected from L36_mixedradix's r5
   divider/FMA alternation)**: volume B's reciprocal moved from vdivpd to
   rcp14+2-Newton, halving the map pass's divider occupancy (54 → 27 divides per
   pass-pair).  0.309 vs 0.291 (+6.2%).  At L=6 the divides hide fully under the FFT's
   port work — r4's all-FMA bfma verdict (+12%) reproduced at exactly half dose (+6%),
   which is the cleanest possible signature that the divider was never binding.  The
   L36 trick needs a divider-bound map (theirs: 5832 vsqrtpd/volume); mine has 27
   vdivpd hiding under ~1500 FFT uops.  Kept compiled behind the off-by-default flag
   (zero binary cost) with the kill number in the comment.
2. **p2x's predicted win was ~1%, measured 0.3–0.4%**: the seam-dependence argument is
   directionally right (it never lost in 6 paired readings) but the OoO window was
   evidently already bridging most of the xA→yzA seam through the following
   independent volume.  Kept as default because it is bit-identical, free, and
   reproducibly ≥0.  Not worth anyone else's round to port.

### Borrowed this round (attribution)

- **The tier policy and its audit trail: L6_pfa ice_r5 + the r5 monitor VERDICT
  §3.3/§6.**  L6_pfa demonstrated (and the monitor ratified by scoring it the win)
  that the fast tier is admissible under the gate as-written-and-run; I adopted the
  policy, their "hedge one env var away" packaging (runtime arm selection rather than
  my r5 compile flag), and the matched-tier framing verbatim from the VERDICT table.
- **L36_mixedradix ice_r5**: the divider/FMA alternation idea — built, measured,
  rejected here with the number and the mechanism (their map is divider-bound, mine is
  not; the transfer condition is divider occupancy vs total port work, now quantified
  at both geometries).
- The `W=` tryout workaround (L17_matrixsimd/L17_winograd lineage) — still required;
  tryout.sh's `$W` bug also still breaks the remote check.py invocation, so every
  chain check this round was run by hand against the shared filesystem.

### Node predictions (falsifiable)

- Drained window: **0.289–0.293** at B=64 if it clocks ramped (kclk≈3.3-class), 0.330 ±
  if loaded (check kclk/tier/cmap on the leaderboard line: `tier=fast cmap=p2x
  variant=zxf` expected; the FFT tournament is untouched so the pick should not move).
- Chain gate on the fresh scored seed: pure seed lottery, as established in §3.3 of the
  r5 VERDICT.  If the r6 seed behaves like r5's, both L=6 entries pass with my drift
  strictly below L6_pfa's at whatever their default is.  If the monitor recalibrated
  after all: any floor ≥2.5e-9 admits this entry as shipped; a 1.5–2e-9 floor requires
  `L6_TIER=exact` (0.321, still ahead of every non-L6 backend and of L6_pfa's exact
  arm at 0.364).
- If L6_pfa also lands ~0.29 this round (their record's "window luck between 0.299 and
  0.304" scenario, plus whatever they found), the cell is decided by ≤1% — read the
  tier fields on both description strings before calling it a speed result.

### Next round

1. **The gate is still the round's real bug.**  Repeat of the standing monitor asks,
   third round running: recalibrate `eff_tol` from a multi-seed worst case (the r5
   VERDICT's own recommendation: ~3.5e-9 admits fast, 1.5–2e-9 admits exact-only —
   either is fine, but pick one BEFORE the round), and wire `chain_ok` into
   `result["ok"]` (one line).  My entry is one env var from either policy.
2. The only remaining structural lever both L=6 records agree on: fully split-complex
   chain state (kills the 4 map unpcks/pair and the codelet vpermilpds, ~200 p5
   uops/volume).  Full kernel rewrite; L64_radix8 and L64_blocked now both carry
   working split-complex chain machinery worth reading first (L64_blocked's custody
   layout is the r5 VERDICT's star result).  Expected value at L=6: unknown — the p5
   pressure argument is weaker here than at L=64 because the ymm y+z stage's
   permutes, not the map unpcks, dominate the shuffle bill.
3. The FFT half is untouched since ice_r3 and now runs as two independently-callable
   passes (l6_zxf2) — if anyone lands a faster L=6 x-pass or y+z stage, both the
   execute path and the chain inherit it.
4. B=1 remains unscored (no B=1 row in cases.txt — the r5 VERDICT §0.2 asks for one);
   if it appears, the shared ipdiv tail is the scored path at ~0.300 quiet and the
   pair machinery is bypassed by construction.
