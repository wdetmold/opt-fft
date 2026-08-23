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

## Round ice_r5 — transpose-free phase 1 (adopted), row-group map pipeline
## (adopted), state-phase control (adopted): 283.3 → ~266 µs/xform

### Where the round started

r4 scored 283.339 µs/xform, first at L=45 by 0.2% over L45_mixedradix
(283.800) — but with a 4.2% run spread against their 1.2%, and dead level
with the rivals' 284 µs.  This is the cumulative round; almost everything
below is taken, with names, from other entries' records.  Nothing in the
map arithmetic or the codelet DAG changed: the round moved *where* work
issues, not what is computed, so every output bit is unchanged.

### What shipped

1. **Transpose-free phase 1 ("tr") — BORROWED from L36_pfa ice_r2,
   transfer to L=45 already proven by L45_mixedradix ice_r2 (their
   −9.6…−12%).**  On this bare-metal ICX a masked `vbroadcastf64x2` with a
   memory operand is ONE load-port uop and zero port 5, so:
   * `zsubt`: every codelet input vector = 4 masked 16 B broadcasts
     straight from the group's 4 y-rows (lanes=y); codelet output is stored
     UNtransposed as full-width 64 B aligned stores to a slot-major scratch
     (slot = ygroup*45 + kz at P + slot*8 doubles; the odd 45th row's xmm
     line stores lane 0 of slots 495–539 at a 64 B stride).
   * `ysubt`: re-gathers lanes=kz — element (y,kz) sits at slot
     (y>>2)*45+kz, granule y&3, ONE formula that also covers y=44, so the
     y-side loads have no tail case; the odd kz=44 column becomes
     `dft45_line1t`, an xmm twin whose loads address the slot layout
     directly (their offsets fold at compile time).
   Both TRNC passes, GCOL/SCOL, and the Wv register array die: ~90k port-5
   shuffles/volume deleted; the FFT's p5-only term is now the 116,766
   codelet swaps alone, two-pipe floor 124.5 → 108.9 µs/vol at 2.9 GHz
   (identical arithmetic to mixedradix's tr — same kernel, same trick).
   The corner turn is repaid as ~360 16 B load uops per group on load ports
   that were near idle.  Codegen audited on the .o before touching the node
   (their discipline): 810 memory-operand `vbroadcastf64x2`, kmovs hoisted
   (3–6/function), tr plane body 2146–2207 instructions vs 3005 classic,
   shuffle-class 156 vs 452, zmm stack spills 48 vs 94.
   **Bit-identity by construction**: the gathers deliver exactly the values
   TRNC delivered, the z-site keeps PFA45R order and the y/tail sites keep
   PFA45, so tr and classic arms coexist in one racing pool and two
   processes with different picks still produce identical bytes (verified:
   out.bin AND out.bin.chain bitwise identical across processes).
2. **Row-group-pipelined lazy map — BORROWED from L36_pencilfused ice_r5
   (`maprows`).**  The mapped `zsubt` stages map(z+c) of row group yg+1
   (4 contiguous rows = 2.88 KB, one `map45_span` call, arithmetic and
   per-point bits unchanged from r4) into a 2×2.88 KB L1 ping-pong
   immediately BEFORE the FFT of group yg; the gathers then read the mapped
   buffer.  The map's rsqrt/divide chains issue under a full codelet
   (~500 uops) of independent work, and the map stores sit a whole codelet
   ahead of their own 16 B reloads.  Their two priced traps are exactly why
   this shape: map fused INTO broadcast load sites spilled +57% for them
   (bcstm), and a standalone staged map starves the divider.  `map45_span`
   gained `noclone` (gcc's constprop clone for the new constant-n callsites
   tripped a spurious -Waggressive-loop-optimizations warning).
3. **Chain state arena with page phase derived from c — BORROWED from
   L45_mixedradix ice_r4 (transitively L23_rader).**  The chain now runs in
   a plan-owned 4096-aligned arena placed at byte phase
   ((phase(c)+2048)&4095): maximally far from c for every volume (both
   stride VDBL).  Their measurement of the state-vs-c 4K false-dependence
   lottery — 294.1 vs 333.0 µs across runs of ONE binary, in-run sd 0.05% —
   is the leading suspect for this entry's 4.2% r4 spread.  final_out now
   takes exactly one write (the last step's map span); x0 is read once;
   W==final_out remains the alloc-failure fallback.
4. **Chain arms rebuilt: {vt-pf, vt, vm-zs, uf}.**  vt = volume-major tr
   chain; vt-pf adds the PF45 one-line-ahead T0 poke on phase 2's 45 RMW
   streams (BORROWED: L45_mixedradix ice_r4 measured the same-shape poke
   worth ~11 µs under the chain; L36_pencilfused ice_r5 re-proved the
   36-stream analog load-bearing, "the one prefetch that survives on
   cache-resident chains" — my ice_r2 "pf1 is a tax" number was the unfused
   exec and does not transfer).  vp/bm/pp DELETED per their r4 drained
   pricing (315.9/317.3/310.3 vs vm 301.8).  FFT45_CH indices are now
   0=vt, 1=vt-pf, 2=vm-zs, 3=uf.  Exec tournament gains `pw4-tr0` (rank 0)
   and il0's rank-0 bet is REVERSED (the r4 drained window read ip0 259.8 <
   il0 262.2 — the hysteresis had been installing the slower plan).
5. **Clock estimate in the description** (`clk=…`, from the settle spin's
   iteration count; the loop is latency-bound at 4 cyc/iter) — BORROWED
   from L36_pfa ice_r5's decode that a 3.3-vs-2.9 GHz window class explains
   every >10% swing; now the scored desc will say which clock we raced at.

### Operation count (per volume-step, PW=4, vt arms)

FFT FMA-class unchanged: 1497 zmm calls × 344 + 90 xmm tail lines =
514,968 zmm FMA-port ops.  Port-5-only: 116,766 codelet swaps (the ~90k
transpose/gather-insert shuffles are GONE); two-pipe floor
(514,968+116,766)/2 = 315,867 cyc = 108.9 µs/vol at 2.9 GHz (was 124.5).
Loads: +~178k 16 B gather uops/volume against 2×64 B/cyc ports.  Map
unchanged from r4 (~220k pipe uops + 11,700 vdivpd/vol, pair ladder, one
divide per 8 points), now issued at row-group granularity.

### Measured (tryout.sh = leased core on a80n0, graded chain m=177 --map;
### MKL same core/case as window anchor; same-window contrasts only)

* Exec tournament, in-arena: **tr0 270.3 vs ip0 305.4 µs/vol at B=4
  (−11.5%)**; 243.6 vs 278.7 at B=1 (−12.6%).  Every candidate passes the
  1e-13 gate; tr0 chosen at B=1/4/8.
* Chain arms, in-arena µs/step/vol (B=4 / B=1 / B=8 windows):
  vt-pf 369.5 / 312.3 / 326.0;  vt 372.2 / 317.2 / 330.0;
  vm-zs 397.0 / 335.6 / 347.2;  uf 400.6 / 333.7 / 363.4.
  vt-pf < vt in ALL three windows (0.7–1.5%, inside the 3% band) → its
  rank was set to 0 mid-round so hysteresis stops suppressing it.
* Driver minima, final build: **B=4 (graded) 265.909 µs/xform (sd 0.15%,
  MKL 786.2 → 2.96×); B=1 265.078 (sd 0.04%, MKL 753.1); B=8 268.6
  (MKL 785.5).**  r4 equivalents: scored 283.3; dev windows 287.9–300.3.
* Correctness: single rel_l2 4.002e-16 (B=4) / 3.998e-16 (B=1) —
  fingerprint unchanged from r4, as bit-identity demands; **whole-chain
  m=177: 4.470e-14 vs tol 1.8e-11 (~400× margin), byte-for-byte equal to
  r4's chain output**; out.bin and out.bin.chain bitwise identical across
  independent processes with the race live.  Setup 0.8–2.0 s (unscored).

### What did not work / traps hit, with numbers

* **My own rank ordering, again**: vt at rank 0 suppressed vt-pf for the
  first three windows exactly the way il0's rank 0 had been suppressing
  ip0 since ice_r2 (r4 drained window: il0 picked at 262.2 with ip0 at
  259.8).  Both reversed this round.  Rule for the file: when the drained
  window contradicts the dev-window bet that set a rank, the rank moves
  the same round — hysteresis is for protecting SIMPLER plans, not pet
  mechanisms.
* gcc 11 emitted a `.constprop` clone of map45_span for the new constant-n
  callsites whose object-size loop bound fired a spurious
  -Waggressive-loop-optimizations warning → `noclone` (zero measured cost;
  the shared copy is also what the store-to-load-distance argument wants).
* Not re-tried, on other entries' priced negatives: map fused into the
  broadcast LD sites (L36_pencilfused r4: +57%), per-plane map staging
  (their r5: the L1-eviction mechanism), c-stream pacing (L45_mixedradix
  r4: +12–20 µs), vsqrtpd in a staged map (pencilfused r5: +40%),
  cpy/ERMS (twice dead here), NTA anything (pencilfused r3).

### Borrowed this round (attributions)

* tr broadcast-gather phase 1 + codegen-audit-before-node discipline:
  **L36_pfa ice_r2**, via **L45_mixedradix ice_r2** (the L=45 transfer
  proof and the slot-layout tail-twin idea).
* Row-group-interleaved lazy map with the one-group pipeline:
  **L36_pencilfused ice_r5** (plus both of its priced traps).
* State-arena page-phase derived from c: **L45_mixedradix ice_r4**,
  transitively **L23_rader ice_r4**.
* Phase-2 45-stream T0 poke under the chain: **L45_mixedradix ice_r4**,
  re-confirmed by **L36_pencilfused ice_r5**.
* Clock-class decode → clk in the desc: **L36_pfa ice_r5** (transitively
  L17_matrixsimd's clk probes).
* The slot-major untransposed intermediate and "broadcast-on-load strictly
  beats extract-on-store": **L36_pfa ice_r2** (their on-paper store-port
  arithmetic, reused unexamined).

### Next round

1. **The codelet swaps (116.8k/vol ≈ 40 µs of p5 floor) are now the entire
   FFT shuffle term.**  The one structural door left is split-complex
   (SoA) lanes — kills every SWAP and VPAIR sign constant.  Every record
   that looked at it (L36_pfa r5 §1, L36_pencilfused, my ice_r2 PW=8
   sketch) says the boundary/load-doubling costs are the risk; L64_radix8
   runs whole-pipeline split-complex at a size with longer passes.  Only
   worth a dedicated round.
2. **The y-subpass is now the slow subpass** (mixedradix measured the
   corner turn at +12% there; my p1 probes still measure the classic path
   only — add p1zt/p1yt tr probes first).  Candidate: a 4-plane-batched
   ysubt that reads each slot vector ONCE full-width and scatters to 4 mid
   columns, trading gathers for stores (store port has slack: 45 stores vs
   211-cyc floor per group).
3. The in-arena chain race consistently reads 20–40% above the driver's
   steady number (create's arena is colder and its self-warm shorter) —
   it ranks correctly but do NOT quote its absolute numbers.
4. If the scored spread stays >2% with the phase fix in, read clk= in the
   desc before touching code (L36_pfa's 3.3/2.9 lesson).
5. vt-pf vs vt on the drained window: the desc carries both; if vt wins
   there by >1%, put its rank back.
