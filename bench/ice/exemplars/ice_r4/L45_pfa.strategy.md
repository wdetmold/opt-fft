# L45_pfa — strategy record (ICE panel)

Lineage note: this entry's rounds panel_r6–r11 are recorded in
`bench/geom/strategies/L45_pfa.md` (and the mt panel's file); those records are
the authoritative history of the Good-Thomas 9x5 structure, the n1_9 DAG
transcription, the PW=1 xmm tails, and the exec-variant tuner.  In **ice_r1
this entry had no agent** — it died in the launch crash-storm (VERDICT ice_r1
§3) — so the 295.054 µs / 12% spread on that leaderboard is a re-measurement of
the CLX-tuned geom/impl_11 binary, plan re-raced on every run.  ice_r2 below is
the first ice-panel round with a live implementer.

## Round ice_r2

### Where it started

ice_r1 standings: L45_mixedradix 284.202 µs/xform (2.6% spread) vs L45_pfa
295.054 (12.0% spread), best library mkl_dfti 521.591.  Machine truths that
reframe everything CLX-era: two 512-bit FMA pipes on p0/p5 (no licence
downclock), 1.25 MB L2, 48 KB L1d, graded chain (B=4, m=177, unitary) is
L3-resident end to end.  The two-pipe port floor of this kernel is ~124 µs/vol;
ice_r1 measured ~295.

### What the new create()-time probes established (the round's main diagnostic)

LOUD/verbose-only probes added: p1/p2 phase split, p1i (stores forced L1-hot),
p1h (reads forced hot), p1c (both hot).  Quiet-window numbers, B=4 arena:

    p1 = 213–227 µs/vol   p2 = 65–68   (phase 1 is 76–78% of the volume)
    p1i = 180–187         p1h = 195–205        p1c = 164–168

So: store-side stall ≈ 30 µs, read-side ≈ 14 µs, and **the kernel itself runs
at ~1.9× its two-pipe port floor with all memory hot** (p1c 166 vs ~87 µs
floor).  The binding constraint is latency/scheduling inside the z-site
(transpose shuffle bursts serialize against codelet FMA bursts; the ~350-entry
ROB cannot span a ~550-uop codelet to overlap them), NOT L3 bandwidth and NOT
the front end: gcc 11 keeps the outer group loops rolled (forced unroll-1 is
byte-identical), so each loop body (~1 k uops) replays from the DSB and the
DSB-vs-MITE hypothesis from the r10 record is dead.  perf_event_open is
**denied** on the node (`perf_event_paranoid=4`) — the brief's "PMU exposed"
does not hold at implementer privilege; the pmu probe degrades gracefully.

### What shipped

1. **il — cross-volume software pipelining (mine, untried in either L=45
   lineage).**  Between phase-1 planes of volume b, run a proportional band
   (~11 tiles) of phase-2 tiles of volume b−1: phase 2's 45-stream read
   latency (the L2 streamer tracks ~16 streams) hides under phase 1's compute.
   Correctness is order-trivial (volume b−1's phase 1 completes before its
   phase 2 starts; disjoint buffers); nvol=1 degenerates to sequential.
   **Won every dev window it appeared in (6/6)**: quiet 288.3–288.5 vs
   291.3–293.2 for ip0 (−1.5%); contended 321–349 vs 354–374 (−5…−9%, the
   advantage grows when L3 is slow — consistent with latency hiding).  Ranked
   first in the hysteresis tie-break, else its sub-3% win would be suppressed.
2. **hz — mixed-width hybrid (mine).**  pw2 (ymm) z-subpass + pw4 (zmm)
   y-subpass through the width-agnostic pl layout; rationale: pw2 measures
   EQUAL to pw4 whole-volume despite a 63% worse FMA-slot floor (ymm FMAs
   issue on p0/p1, freeing p5 for the transpose shuffles), so take ymm at the
   shuffle-dense site.  Dev windows: within noise of ip0 (327–374), never beat
   il0.  Left tuner-gated for the drained window to price.  `phase1_plane` was
   split into `zsub`/`ysub` to compose this.
3. **Clock-settle spin** (~150 ms dense FMA before the tournament) + nv=4
   timing rounds 6→8.  Borrowed from L17_matrixsimd ice_r1 (transitively
   L17_winograd).  This kills the ice_r1 12%-spread mechanism (schedutil ramp
   edge inside the tournament).  Post-change chain runs read sd 0.02–0.8%.
4. **Pool pruned for this node**: pw2-ip-pf3 / pw2-sp-* deleted (every 256-bit
   variant lost every ice_r1 tournament — VERDICT §5); pf3/pf3a kept but
   ranked last (L13_rader's rule: prefetchw +7.4% when the store target is
   L3-resident, which the drained chain guarantees).  zal (pf0a) re-ranked
   ahead of ip0 so its sub-3% win isn't tie-broken away.
5. **Leaderboard-legible description**: the desc string now carries the
   drained-window head-to-head (`ip0=… il0=… hz=…`) for the next round.

### What was tried and killed, with the number that killed it

* **cpy (ERMS plane-image copy, borrowed from L45_mixedradix r8): +11%.**
  cp0 = 314.8–320.6 vs ip0 = 282.5–286.0 µs/vol in quiet windows.  At 32 KB
  per plane the rep-movsb neither skips the RFO reliably nor beats the stall
  it removes, and in this uop-starved kernel the copy's own time (~0.7 µs/
  plane) is pure loss.  Matches their own mt-era note ("cpy lost every cell");
  now also measured in the L3-resident regime it was designed for.  Kept in
  the pool as a priced negative.
* **Overlap-recompute tails (r7–r9 form) on ICX: p1c 173.4 vs 167.7 (+3.4%).**
  The r10 xmm tail lines survive on a two-pipe machine too;
  `-DFFT45_OVERLAP_TAIL` control run, contention-immune metric.
* **Outer-loop unroll pinning as a *fix*: no-op.**  Byte-identical binaries
  with/without `GCC unroll 1` on the group loops — gcc 11 never unrolled them.
  Kept as a contract, documented as such (not as a win).
* **The whole prefetch ladder is negative in quiet windows**: pf1 +2–8, pf2
  +4, pf3 +10 µs/vol vs pf0.  Every added uop hurts (kernel at IPC ~1.6).
  BUT contended windows flip this — pf3/pf3a won several noisy tournaments —
  so the ladder stays tuner-gated rather than deleted: the tuner adapts to
  the environment it actually runs in.
* **zal (aligned z-loads, r11): real but small and window-dependent** —
  −1…−2.5% in two quiet windows (278.4–280.0 vs 282.5–286.0), ±1% later.
  Stays in the pool at rank 1.

### Measured, end state (tryout.sh = graded chain on the reserved node, leased core)

* B=4 (graded case, m=177): best quiet window **296.5 µs/xform** min
  (sd 0.02%, MKL same window 519.3 → 1.75×); contended windows 327–358 while
  MKL read 552–612.  Correctness rel_l2 = 4.0e-16, chain m=177 check 2.1e-14,
  repeatable output.
* B=1 (m=177): **279.8 µs/xform** (MKL 515.0 → 1.84×).
* B=8: 466.4 (working set ~2× L3 half; MKL 555.4 → 1.19×).
* Op count unchanged since r10: 1497 zmm calls × (344 FMA-port + 78 shuffles)
  + 90 xmm tail lines per volume; two-pipe port floor ~124 µs/vol.
* setup (unscored) grew to ~1.0–1.1 s (bigger pool + spin + probes).

### Attribution

il and hz are mine.  cpy/plane_copy: L45_mixedradix r8.  Clock-settle spin:
L17_matrixsimd ice_r1 (transitively L17_winograd).  prefetchw-only-past-L3
rule: L13_rader ice_r1.  256-bit prune and the drift-floor discipline:
VERDICT ice_r1.

### Next round

1. **The strategic prize is p1c: 166 vs ~87 µs floor.**  The z-site
   serializes ~176 cycles of p5-only transpose bursts against ~190-cycle FMA
   bursts per call.  Two candidate attacks, in order: (a) manual software
   pipelining — interleave group g+1's load-transposes into group g's codelet
   text so both port groups stay fed (macro surgery on PFA45R's call site);
   (b) the split-complex PW=8 rewrite (688 scalar ops/line on 8 lanes = 86
   port-slots/line vs the current 105.5, zero codelet shuffles) — big, risky,
   needs an 8-row transpose family and a 5-row tail scheme.
2. il granularity: the band is currently ~11 tiles per plane; try 2×/0.5×,
   and an il+zal recheck in drained windows (il0a lost to il0 by ~2% in dev
   windows — another added-uop casualty, but the drained window may differ).
3. If the panel adds B=1 grading: il degenerates there; intra-volume overlap
   is structurally impossible (x-pass needs all planes), so B=1 gains must
   come from item 1.
4. Read the leaderboard desc: it prints the drained-window ip0/il0/hz
   tournament — that's the only clean measurement environment we get.

## Round ice_r4

(ice_r3 note: this entry had no live changes that round — the file still
carried the ice_r2 header; the 293.720 µs on the ice_r3 board is the ice_r2
binary re-measured.)

### The task changed: this round is fft3d_chain

The graded step is now `state <- (z+c)/(1+|z+c|)`, z the RAW (unnormalized)
FFT — no unitary scale in map mode (verified in driver.c).  The driver times
an exported `fft3d_chain(plan, x0, c, final_out, m)` weak symbol as the whole
m=177-step unit; entries without it pay fft3d_execute + a driver-side
vectorized map pass (MKL through that fallback: **760–851 µs/xform** on this
round's windows, from 521 FFT-only — the unfused map at L=45 costs more than
half our whole FFT).  FFT kernels, tuner, and fft3d_execute are untouched.

### What shipped

1. **fft3d_chain with final_out as the in-place state arena.**  Phase 1 is
   in-place-safe by construction (zsub drains plane x through the plane
   scratch before ysub's first store — checked, and the `restrict` on
   phase1_plane's in/mid had to be REMOVED to make in==mid legal), phase 2
   always was.  So the chain runs in final_out end to end: x0 read once, no
   ping-pong buffer, no per-step scale pass.  This alone is why the full map
   chain now measures at/below ice_r3's FFT-only number: the old graded
   config paid a driver unitary pass (r/w 11 MiB per step) plus two-buffer
   ping-pong traffic that are simply gone.
2. **Lazy map fused into zsub's loads** (BORROWED: the rivals' winning
   fusion, corpus §10 §2, taken concretely from L13_rader ice_r4).  Between
   steps the buffer holds raw z; the next step's z-pass loads do
   `w = z + c` (both stream sequentially) and map in-register (`mp` flag on
   zsub).  The odd z-column maps on its GCOL gather (map45_one); the odd
   45th row pre-maps into a 720 B stack line via map45_span before
   dft45_line1.  One map45_span pass finishes step m.  The map's dep chains
   land in exactly the pass with latency slack (zsub runs at ~1.9× its port
   floor, ice_r2 probes) — measured marginal cost of the whole map: ~0–8 µs
   per transform.  L17_matrixsimd's warning that lazy loses when the mapped
   pass is the first pass did NOT transfer here: their X pass is
   compute-dense; our z-subpass stalls on transpose/FMA serialization and
   swallows the ladder.
3. **Map arithmetic** (BORROWED: the L13_rader/L17_matrixsimd/L23_matrixsimd
   ice_r4 consensus): `s = re²+im²` with a 1e-300 guard folded into the fma
   (rsqrt14(0)=inf would NaN); vrsqrt14pd seed + 2 Newton
   (2⁻¹⁴ → 5.6e-9 → 4.7e-17, below double rounding); `d = fma(s,y,1)`;
   **ONE exact vdivpd per 8 points** — two interleaved vectors' s pack into
   one zmm (even lanes), one ladder + one divide, unpack, two mul-outs
   (L17's s6 shape: ~19 pipe uops + 1 divide per 8 pts).  ~2–3 ulp per
   application; budget spent: none.  `-DFFT45_MS=1` swaps the divide for an
   rcp14+2-Newton ladder (different bit class → compile-time only, per
   L23_matrixsimd).
4. **Five chain-structure arms, gated + raced in create()**, legal because
   every arm is bit-identical (the map is per-point deterministic under any
   grouping/width — unit-tested pair vs one vs xmm over 100k random
   vectors — and the FFT arithmetic per line never changes):
   vm-zs volume-major (corpus §10 §3 via L17/L23: one volume through all m
   steps; at L=45 state+c = 2.8 MiB so it's a locality play, not full L2
   residency); vp-zs volume pairs with the ice_r2 il interleave running
   ACROSS STEPS (p2 of volume a's step s hides under p1 of volume b);
   bm-zs batch-major il; vm-pp plane-premap (map45_span into the P image,
   zsub untouched — hedges zsub pressure); uf unfused control.  The gate
   checks each arm against fft3d_execute + an exact scalar map chain
   (rel 1e-13) AND memcmp's arms against each other — a bit-diverging arm is
   disqualified, so the per-process adaptive pick can never break the
   repeatability cmp.  FFT45_CH forces an arm; FFT45_MSG sets gate steps.
   Race results: quiet window vm-zs 305.2 < vp 316.4 ≈ bm 316.7 µs/step/vol;
   a contended window flipped to vm-pp 370.0 < vm-zs 381.5 — the race adapts
   per process, which bit-identity makes safe.

### The round's most transferable finding: -ffp-contract=fast broke bit-identity

First build: uf and vm-pp failed the cross-arm memcmp by 1 ulp while the
map forms unit-tested identical.  Mechanism, isolated by chain-length
bisection (m=1 agreed, m=2 diverged): the map's final `o = w*q` multiply is
a generic vector MULT in GIMPLE, and when the map inlines into zsub it flows
in registers straight into the codelet's first adds (`x1_+x4_`) — **gcc's
default -ffp-contract=fast contracts the pair into vfmadd**, so the fused
path's mapped state differs by 1 ulp from the stored form (map45_span is
noinline; a store/load blocks contraction).  Fix: a zero-uop asm register
barrier (`__asm__("" : "+v"(r))`) on the map outputs in all three forms.
ANY entry that computes its map both fused-in-register and through a stored
pass will hit this; intrinsic FMAs are safe, bare vector multiplies feeding
adds are not.

### Operation count

FFT unchanged (1497 zmm calls × 344 FMA-port + 78 shuffles, + 90 xmm tail
lines; two-pipe port floor ~124 µs/vol).  Map per volume: 10,890 row-segment
pairs + 495 GCOL singles + 45 tail-row spans ≈ 220k pipe uops (~38 µs
two-pipe floor) + **11,700 vdivpd ≈ 64 µs of divider occupancy that runs in
parallel with the pipes** — measured marginal cost ~0–8 µs/transform, i.e.
the ladder hides in zsub's existing stalls and the divider was idle.

### Measured (a80n0, tryout.sh graded chain, m=177, --map; MKL = fallback)

| case | this round | MKL same window |
|---|---|---|
| graded B=4 | **287.9 / 293.0 / 295.2 / 300.3** µs/xform (four windows, sd 0.02–0.06%) | 761–806 → **2.6×** |
| B=1 | **295.3** | 760 → 2.6× |
| B=8 | **286.4** (ice_r2 measured 466 at B=8 — the in-place volume-major chain deleted the >L3 ping-pong) | 843 → 2.9× |

Correctness: single rel_l2 4.002e-16; **whole-chain m=177: 4.5e-14 (B=4),
2.6e-14 (B=1) vs tol 1.8e-11 — ~400× margin, exact tier by construction.**
Two independent processes: single output AND .chain bitwise identical with
the race live.  -Wall -Wextra clean.  vs ice_r3's 293.7 FFT-only: the full
graded step now costs the same or less than the bare FFT did.

### What did NOT work / negatives with numbers

* **MS=1 all-FMA map (rcp14+2NR, no divider): 303.2 vs 300.0/293.0 same
  window (+1–3%).**  Matches L13/L17: the divider is otherwise idle here, so
  burning it once per 8 points is free while the extra 5 uops/pair compete
  with the kernel.  Default stays MS=0.
* **The first-build contraction bug** (above) — cost half the round to
  diagnose; the asm barrier is the permanent cure.
* Tooling: tryout.sh line 36 uses $W before defining it (set -u abort) —
  work around with `W=$PWD/build/tryout/<name> ./tryout.sh ...`; and its
  check.py invocation loses $W inside the $() (chain check runs with
  --cin /c.bin and dies) — run check.py --map-check manually; the fs is
  shared so it works from the login node.

### Attribution

Lazy-map fusion + one-divide-per-point consensus: L13_rader ice_r4 (their
MAPSTYLE sweep saved me the vsqrtpd trap: 2× divider occupancy), transitively
the rival generator's mapF/PW_STYLE.  Pair-shared ladder + "rsqrt14 is 2.3
cyc, not microcoded" + divider-throughput numbers: L17_matrixsimd ice_r4
(s6).  Volume-resident chain order + guard-folded fma + "mv is compile-time
because bit classes differ": L23_matrixsimd ice_r4.  Bit-identity as the
license for plan-time adaptivity: L13_rader ice_r4.  In-place arena and il
structure: mine (ice_r2/r4).  The contraction diagnosis and asm-barrier fix:
mine, this round.

### Next round

1. **The FFT itself is now the whole gap to the rivals' 284 µs/xform**
   (0.201 s / 708 transforms).  The p1c latency attack from the ice_r2
   record (166 vs ~87 µs floor: manual software-pipelining of group g+1's
   load-transposes into group g's codelet, or the split-complex PW=8
   rewrite) is unchanged as the strategic prize.
2. Eager store-side map (fold the map into phase 2's stores of the SAME
   step, L23's placement) was NOT tried — the lazy form measured ~free, so
   there was nothing to win this round; revisit only if zsub becomes the
   binding site after item 1.
3. Watch the leaderboard desc: it now carries the chain-arm head-to-head
   (`ch=<pick> (vm/vp/bm/pp/uf=...)`) from the scored drained window — if
   vp/bm win there, the il-across-steps idea deserves a deeper look (finer
   bands, pp+il composition).
