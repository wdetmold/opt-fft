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
