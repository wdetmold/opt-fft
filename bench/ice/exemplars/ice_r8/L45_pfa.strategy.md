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

## Round ice_r6 — the split-complex (SoA) bet built and PRICED OUT; the
## round's actual win is the x-first map placement (xf family, −1.7%)

### Where the round started

r5 board: L45_mixedradix 262.959 (spread 3.5%), me 263.807 (spread 0.5%)
— second by 0.3%.  Round-start re-measure, quiet window: **B=4 264.341
min, sd 0.08%, MKL 758.1**.  Both L=45 records named split-complex (SoA)
state as the one structural lever left; this cumulative round was the
"dedicated round" it needed, so I built it — with the bit-identity gate
as the safety net — and let the node price it.  It lost, for reasons the
probes made quantitative (below), and the diagnosis pointed at a
different, cheaper lever that DID pay.

### 1. Split-complex chain arms (vs / vs2): built, bit-exact, REJECTED

Design (licence: L17_winograd ice_r5 "the chain owns the state format";
precedent: L64_radix8): state between steps SPLIT — per plane a 45x48 re
array then a 45x48 im array (y-major, rows padded to 48 dbl = 6 whole
lines, pads zero forever).  Split codelets PFA45S/PFA45RS transcribe the
interleaved DAGs per component (every SWAP+VPAIR crossing becomes one
FMA on the other component, sign folded; z-site keeps PFA45R's factor
order for rows 40-43 + a scalar PFA45 line for row 44, y/x sites keep
PFA45) so per-point bits are IDENTICAL to the vt arms — the cross-arm
memcmp gate passed on the FIRST build, which is what made a rewrite this
size safe to attempt at all.  Deleted: all 116,766 codelet swaps/vol
(entire FFT p5 term) + the map's pack permutes; two-pipe floor 108.9 →
96.1 us/vol.  Codegen audit (.o, before the node): ZERO shuffle-class
instructions in all six codelet blocks, all gathers memory-folded
vbroadcastsd, spills 65-68 vs 48-94 interleaved.

**Measured (in-arena, same window as vt): vs 452.7 → after probes 395-419
vs vt-pf 289-337.  +30-38%, consistently.**  The decomposition
(new split probes, all same-window contrasts):

    p1s (split z+y, no map)   203-228 us/vol   vs p1t (tr, interleaved) 153
    p1sm (with staged map)    266-300          vs p1tm 243.9
    p2s (split x pass)        66.9-76.5        vs p2 65-66  (90 streams OK)
    p2sm (map at x stores)    215.3            (!!)
    conv (interleave<->split) 83-97 one-time   (2/177 amortized, fine)

Two mechanisms, both now written down so nobody rebuilds them:

* **Split p1 is LOAD-PORT-bound**: the corner-turn gathers become 8 B
  scalar broadcasts (16/vec-pair vs G4B's 4/vec) → ~540k load-class
  uops/vol ≈ 93 us of load-port floor, vs the tr path's 60 us FMA floor.
  Both run ~2.4-2.5x their binding floor, so the split arm loses by
  exactly its extra load uops.  The fix would be 16 B granules, which
  requires a y-PAIR-interleaved SOURCE layout — possible for the map
  ping-pong (2 permutex2var per vec at span time), IMPOSSIBLE for the
  state itself: the state's writer (ysubs, lanes=kz at fixed ky) can
  never emit y-pairs full-width.  Conservation: the y<->z corner turn
  must cross the volume somewhere, and 8 B lanes double its uop cost.
* **p2sm's +139 us is the c loads**: at the x-pass store sites, c is
  needed at (x=0..44, ky, kz) = 90 loads scattered over 45 PAGES per
  call — unprefetchable, every one an exposed L3 latency.
  L36_mixedradix ships map-at-p2-stores at L=36 only because their c is
  PERMUTED to match store order ("cperm").  A CS2 permuted-c layout
  (per-tile 45x16 dbl, fully sequential per step) is designed and
  documented here, NOT built — it would only bring vs2 back to ~even
  with its own p1 problem still unsolved.

Both arms stay in the pool as gate-verified priced controls (vs-pf
465.0 vs vs 452.7 → deleted; vs2-pf 510.0 vs vs2 497.9 → deleted).

### 2. The probe that changed the round: the r5 STAGED map costs +90 us/vol

Same-window twins: **p1tm − p1t = 243.9 − 153.4 = +90 us/vol** for the
vt family's own row-group staged map (and +72 for the split twin).  The
r4 "map is ~0-8 us" number was the r4 REGISTER-FUSED form; the r5
maprows staging is NOT free: a ~430-cyc span (48 vecs, 48 divides) is
too big for the ROB to overlap with the neighboring codelet, so it runs
nearly serial — L36_pencilfused's "standalone staged map starves the
divider" trap, now with a number on this entry.  That made map PLACEMENT
the round's real lever, worth more than the whole SoA delta.

### 3. What ships: the xf family — x-first step, map at the y-subpass
### stores (BORROWED: L45_mixedradix ice_r4 CXF + their ice_r5 MPAIR)

Per step: x pass FIRST (in place, on the mapped state), then per plane
zsubt (plain) → **ysubtm**: PFA45 with the map fused at the STORE sites —
consecutive outputs pair through a one-register stash (pair grouping is
value-transparent, so bits are safe), one rsqrt14+2NR ladder + ONE
vdivpd per 8 points, c loads PLANE-LOCAL (their CPFY negative says OoO
covers them).  45 is odd → each call flushes one single through
map45_one; the odd kz column runs the raw line into a stack row, gathers
its c column, and maps through one paired span (6 divides for 45 points
— raced vs 45 scalar ladders: dead tie 270.22/270.23, kept for the
divider count).  Step m stores raw and one span writes final_out; x0 is
memcpy'd in once per volume (0.34 us/xform).

**The axis order (x,z,y vs z,y,x) is a different bit class**, so the two
families are compile-time selected (-DFFT45_XF, default 1), never raced
adaptively: arm[0] of the pool sets the gate's bit reference, the other
family auto-DQs, and cross-process repeatability survives.  A
timing-based pick across families would flip output bits between
processes — the L13_rader bit-identity licence read in reverse.

**Measured, same-window A/B (the mixedradix drift-trap protocol:
rebuild-and-rerun the incumbent, MKL anchors flat 760-765):**

| window | xf-pf (XF=1) | vt-pf (XF=0 control) |
|---|---|---|
| 1 | **270.234** (sd 0.09%) | 275.246 (sd 0.06%) |
| 2 | **270.217** (sd 0.25%) | 274.932 (sd 0.02%) |

−1.7/−1.8%, 2/2.  B=1: **259.755** (sd 0.12%, MKL 752.0) vs r5's 265.1.
In-arena race (XF=1): xf-pf 324.9-353.6 < xf 329.9-356.8 (poke worth
~1%, rank 0).  NOTE the session drift: byte-equivalent z-first builds
read 264.3 → 275.2 across the day's windows while MKL stayed 758-765 —
mixedradix's r5 "MKL does not track the drift" trap reproduced; every
verdict above is same-window.

### Operation count (xf arms, per volume-step)

FFT unchanged from r5 tr: 1497 zmm calls x (344 FMA-port + 78 swaps) +
90 xmm tail lines; two-pipe floor ~108.9 us/vol.  Map: 495 mapped
y-calls x (22 pair ladders + 1 single) + 45 column spans ≈ 11,700
divides + ~210k pipe uops — same arithmetic as r4/r5, moved to the
store sites.  The split arms' count for the record: 810 zmm calls x 688
FMA-port ops + 45 scalar lines, ZERO codelet shuffles, floor 96.1
us/vol — the better floor loses to its own 540k load uops.

### Correctness (all on the node)

Single transform rel_l2 4.002e-16 (B=4) / 3.998e-16 (B=1).  Whole-chain
m=177 vs the numpy reference chain: **6.325e-14 (B=4), 3.280e-14 (B=1)
vs tol 1.77e-11** (~280x / ~540x margin; class change from r5's 4.47e-14
is the axis-order bit class, still exact-tier).  check.py run by hand
both batches (tryout's /c.bin bug STILL live in r6 — fifth round — and
its && chain silently skips the repeatability cmp when check.py dies).
Two-process cmp of the B=4 chain output: bitwise identical.  -Wall
-Wextra clean (3 pre-existing pw2 unused-function warnings unchanged).

### What did NOT work, with numbers (this round's negatives)

* **Split-complex p1**: +50-75 us/vol vs tr p1, load-port arithmetic
  above.  Do not rebuild without a 16 B-granule corner-turn design.
* **Split staged map**: +72; **interleaved staged map (r5 SHIPPED
  form!)**: +90 — the r5 maprows adoption was a mis-buy this entry never
  priced in isolation until now.
* **Map at x-pass stores without permuted c (vs2)**: +139 us/vol.
  With permuted c: designed, unbuilt, documented above.
* **vs-pf / vs2-pf pokes**: +2.7% / +2.4% — pokes tax arms that are
  already issue- or memory-starved.
* Paired vs scalar odd-column map tail: dead tie (kept paired).

### Borrowed this round (attributions)

* x-first step + map at y-subpass stores + plane-local c: **L45_mixedradix
  ice_r4** (their CXF=1, measured 287.1 vs 347.0 then); MPAIR store
  pairing via stash: **L45_mixedradix ice_r5**, transitively **L23_rader
  ice_r4** and the rivals' PW_CORE.
* "Map placement: store sites hang off the END of the dependency graph;
  z/load sites feed the critical path": **L36_mixedradix ice_r6** (their
  split-protocol post-mortem, read before building xf).
* Split-complex licence and precedent: **L17_winograd ice_r5**,
  **L64_radix8**; the staged-map starvation trap: **L36_pencilfused
  ice_r5** (requantified here).
* Same-window incumbent-rebuild protocol: **L45_mixedradix ice_r5**;
  "node B=1 state is currently elevated, same-window A/B only":
  **L23_rader ice_r6** / **L23_matrixsimd ice_r6** (my B=1 read clean
  anyway).

### Predictions for the scoring window

* Pick: ch=xf-pf (xf within 1% behind).  Scored B=4 ~262-271 depending
  on which drift class the monitor's window lands in (my quiet-class
  equivalent: ~265 − 4.7 ≈ **~260-265**); the desc now prints the whole
  arm table + xf flag, and OUT entries are the other bit family, by
  design, not failures.
* Fingerprints: single 4.002e-16 / chain 6.325e-14 (B=4), 3.280e-14
  (B=1).  Setup ~1.1-1.6 s.

### Next round

1. **If xf holds, the staged-map deletion is worth porting further**:
   the +90 us staged-map tax likely afflicts other entries that adopted
   maprows-style staging — check their records before they re-buy it.
2. Split-complex round 2, only with a 16 B corner turn: pair-interleaved
   PONG (2 permutex2var/vec at span time) fixes the mapped z-gather; the
   state-side gather stays 8 B unless someone finds a pair-emitting
   y-store.  Expected recovery ~30-40 us of the 75; probably still short
   of tr — read the arithmetic here first.
3. vs2 + CS2 permuted c: designed above, ~1 day of work, caps at ~even.
4. The xf map's remaining exposure: ysubtm runs ~+35-50 us over plain
   ysubt; if the scored number lands >272, try poking c rows one CALL
   ahead (not one line) before touching structure — but mixedradix's
   CPFY negative says expect a loss.
5. tryout.sh: FIFTH round of the $W//c.bin bug + the silent cmp skip.
   Central fix overdue.

## Round ice_r7 — quad-volume lanes (q4, from the rivals' own sources):
## 264 → ~233 µs/xform, −11.8% same-window, the round's mission executed

### Where the round started

r6 board: L45_pfa 264.111 (spread 3.2%), first by 0.3% over L45_mixedradix
264.977.  Round-start re-measure, quiet window: **262.445 µs/xform (sd
0.12%, MKL 756.4)**.  This is the "mine the competition" round; the newly
published `results/rivals_icelake/` table shows the best GATE-PASSING
rival at L=45 is 1760b1bf at 0.208 s (≈294 µs), but the fastest raw L=45
time across all 23 rival attempts is **v6_5a869e40 at 0.167 s (≈236 µs)**
— env-gate-failed on this node (its campaign's (B,m) assumptions), so the
number is not fully trustworthy, but its README names the structure: for
L ≤ 45 a "4-volume interleaved layout (SIMD lanes = volumes, no tail
lanes, no transposes)" dispatched as `run4_`, plus 2 MB-hugepage arenas.
Our graded case IS B=4.  That structure is exactly the lane-mapping this
entry never tried (ice_r2's PW=8 sketch and r6's split-complex both
attacked the corner turn, neither deleted it), so this round built it.

### What shipped: the q4 chain family (BORROWED: v6_5a869e40 run4_ idea;
### all DAGs, map, and step order are this entry's own, for bit-identity)

State for a group of 4 volumes lives POINT-MAJOR in a plan-owned arena Q:
point (x,y,z) of volumes 0..3 is ONE zmm at Q + ((x*45+y)*45+z)*8 doubles.
c is permuted into C4 once per group per chain call (C4 = Q + Q4VD + 344:
64B-aligned AND at +2048 B page phase from Q — the L23_rader/L45_mixedradix
4K rule applied arena-internally).  Per step, x-first exactly like xf:

    x pass:  2025 pencils, 45 streams at 129600 B stride (2624 mod 4096,
             no fixed page-offset relation), in-place RMW, PFA45.
    z pass:  per x-plane, 45 CONTIGUOUS 2880 B pencils, in place;
             y-rows 0..43 PFA45R + row 44 PFA45 (zsubt's exact bit split).
    y pass:  per x-plane, 45 pencils at 2880 B stride, PFA45, with the map
             fused at the store sites (steps 1..m-1): ysubtm's stash
             pairing verbatim, one rsqrt14+2NR ladder + ONE vdivpd per 8
             points, map45_one flush for the odd 45th ST.  Step m stores
             raw; the final map is FUSED INTO the deinterleave (q4_unpackm:
             map45_pair per two points, 16 B scatters to the 4 volumes —
             deletes the separate 11.7 MB span round-trip).

Every codelet access in every pass is one full-width 64B-ALIGNED zmm
load/store.  Deleted from the steady loop vs xf: ~178k 16 B corner-turn
broadcast uops/vol, the slot-major scratch round-trip (~32 KB/plane
written + regathered), ALL tail machinery (90 xmm lines, GCOL/SCOL, odd
z-column map paths).  FFT op count per point unchanged: 6075 zmm calls
per group × (344 FMA-port + 78 swaps) = 1518.75 calls/vol-equivalent,
two-pipe floor ~110 µs/vol — the win is all in the feeding, not the count.
Paid: the group working set (Q + C4 = 11.7 MB) is L3-resident (the xf
arms' partial-L2 volume locality is gone), plus ~1.6 ms/group/call of
pack+permute+unpack (2.3 µs/xform at m=177).

**Bit-identity with xf BY CONSTRUCTION, and it held on the FIRST build**:
same step order, same DAG per site (incl. the row-44 PFA45 quirk), same
map arithmetic (pair grouping is value-transparent), pack/unpack are pure
moves.  So q4 sits in the same racing pool, the cross-arm memcmp gate
proves the transcription, arm[0] = xf-pf still sets the bit reference,
and B=1/B%4 remainders run the xf body — every batch produces the same
bytes as r6's shipped code (B=4 chain fingerprint 6.325e-14, B=1
3.280e-14, both UNCHANGED from r6).

Support changes: W arena grown to hold Q/C4 behind the split region
(layout of the old arms untouched); **madvise(MADV_HUGEPAGE) on the q4
region** (BORROWED: v6_5a869e40's hugepage arenas — priced below, −1.4%);
chain-race msT 5 → 12, NRC 6 → 5 (at msT=5 the once-per-call pack cost is
a ~13% false penalty on q4 in the race — enough to flip a pick in a noisy
window; at msT=12 it is ~5% against a measured ~12-24% true margin);
ranks now q4=0, q4-pf=1, xf-pf=2, xf=3 (same-round rank move per the r5
rule, on 3 same-window driver A/Bs); q4 pass probes + a probe-only
store-direct z-DAG twin added to the LOUD block.

### Measured (a80n0, leased cores, graded chain B=4 m=177; every verdict
### same-window, MKL anchored; the day drifted 756 → 798 as the node
### filled with other implementers)

* Round-start baseline: 262.445 (sd 0.12%, MKL 756.4).
* First q4 build, adaptive: **232.052 (sd 0.02%)**; race q4 335.4 vs
  xf-pf 373.1 µs/step/vol (q4 wins by 10% DESPITE the msT=5 pack bias).
* Same-window forced A/B (sd ≤ 0.07%): **q4 232.5, q4-pf 232.7 (dead
  tie), THP=0 235.7 (hugepages worth −1.4%), xf-pf control 263.7** —
  q4 −11.8%; all four outputs bitwise identical.
* Contended windows (MKL 776-798): q4 240.9-248.6 vs xf-pf 274.6 (−9.8%).
* Final shipped build, adaptive: **233.678 (sd 0.05%)**, picks q4, chain
  PASS 6.325e-14 (tol 1.8e-11, ~280× margin), cross-process outputs
  bitwise identical.  **B=1 264.9-265.1** (xf fallback, 3.280e-14);
  **B=8 237.1** (two q4 groups, 3.332e-14).  Setup 0.7-1.3 s.
* Cleanest end-of-day anchor, QUIET window (MKL 758.6, sd 0.05%):
  **B=4 232.838 µs/xform, sd 0.02% → 3.26× MKL** — vs the same-class
  round-start 262.4 with MKL 756.4, i.e. −11.3% at matched window class.
* Pass decomposition (LOUD probe, µs/GROUP of 4 vols, quiet-ish window):
  pack2=895 x=218 z=245 y=237 ym=554 unpkm=705.  Per-vol: x 55, z 61,
  y 59, ym 138 — the mapped y-pass carries the whole map cost (+79/vol
  over plain y); the FFT passes each run ~2× their ~27 µs/vol two-pipe
  slice, i.e. the q4 kernel feeds BOTH pipes about as well as phase 2
  ever did, at every site.

### What did NOT work, with numbers (all same-window)

* **q4-cz (c-poke in the z-pass to pre-warm the y-map's C4 loads):
  287.6 vs 248.6 (+16%).**  2025 poke uops/plane starve the z-pass's own
  loads; the y-map's c loads were already OoO-covered.  Third confirmation
  of the CPFY/poke-tax rule in this lineage (r6 item 4 predicted it).
  Removed from the pool; the function stays as the priced negative.
* **MS=1 (rcp14+2NR instead of the one vdivpd): 257.9 vs 241.0 (+7%,
  sd 4.3%).**  The divider is still not binding even with q4's denser map
  pass.  Fourth confirmation; default stays MS=0.
* **Store-direct PFA45 at the q4 z site (probe zsd): 268.3 vs 245.1
  µs/group (+9%).**  The r11 per-site rule holds even with memory STs at
  a contiguous site — PFA45R stays, and the "switch the whole family's
  z-DAG" idea is dead without ever risking a bit class.
* The v6_5a869e40 0.167 s number itself: treat as structure evidence
  only, not a target — 10 of 16 v5/v6 attempts return rel~1 garbage at
  exactly L=36/45 on this node (their campaign's (B,m) table baked in),
  so their L=45 walls may under-count work.

### Attribution

Quad-volume lane mapping + hugepage arena: **rival v6_5a869e40**
(`fft_v5v6_solutions/v6_5a869e40_score0.80/README.md`, `run4_` dispatch),
via this round's context directive to mine the rival sources.  Everything
that made it SAFE is the panel's accumulated machinery: bit-identity
racing licence (L13_rader ice_r4), chain-owns-the-format licence
(L17_winograd ice_r5), map placement at store sites (L45_mixedradix
CXF/MPAIR via my r6), arena page-phase control (L45_mixedradix ice_r4,
transitively L23_rader), same-window incumbent-rebuild protocol
(L45_mixedradix ice_r5).  The race-bias diagnosis (msT amortization) and
the fused map+deinterleave are mine, this round.

### Prediction for the scoring window

Pick: ch=q4 (q4-pf within noise behind).  Scored B=4 ≈ **230-240**
depending on drift class (quiet-class ≈ 233), chain ≈ 0.163-0.170 s vs
the rivals' best gate-passing 0.208 s.  Fingerprints unchanged from r6:
single 4.002e-16, chain 6.325e-14 (B=4) / 3.280e-14 (B=1).  Setup ~1.3 s.

### Next round

1. **The mapped y-pass is now the fat target: ym−y = +79 µs/vol** while
   the map's twelve-hundred-odd divides/plane are provably not
   divider-bound (MS=1 lost).  The ladder uops compete with the y-pass's
   FMA stream.  Candidates: cross-CALL pairing of the 45 odd singles
   (saves ~22 ladders/plane, small); issuing the map of plane x's
   columns interleaved into the x-PASS of the same step (the x-pass has
   latency slack and reads different addresses — but this is
   staging-adjacent, and staging is a priced trap: read r5/r6 first).
2. pack2+unpkm ≈ 1.6 ms/group/call (2.3 µs/xform).  NT stores for the
   final_out scatter and/or the C4 pack would skip ~11 MB of RFO per
   call; only worth it if someone finds the scored config calls chains
   with small m.
3. B=1 (264.9) still runs the r6 xf path untouched — if a B=1-graded
   panel ever appears, a 2-volume (ymm-lane) or within-volume 4-plane
   variant of q4 is the obvious sketch, but at B=1 there is no second
   volume: it would have to be 4 x-planes as lanes with a transpose at
   the x-pass — the corner turn returns.  Probably not worth it.
4. The q4 idea should transfer to other sizes whose graded B ≥ 4 and
   whose volumes are small enough to keep the group cache-resident
   (L=36 at B=8: 8 volumes × 46656 × 16 B = 6 MB/group of 8 — L36's
   entries should read v6_5a869e40's README before their next round).
5. tryout.sh $W bug: SIXTH round.  Worked around again by exporting W
   before the call (and this round the reservation check also went
   stale — no squeue on any reachable host — so the whole tryout body
   ran as a manual ssh replica under a normal slot lease).

## Round ice_r8 — the custody schedule ported onto q4 ("qc"): 231.2 scored
## → 215.1 µs/xform quiet-window, −7% same-window, first at L=45 again

### Where the round started

r7 scored 231.220 (spread 3.1%) — SECOND at L=45, behind L45_mixedradix's
225.571 (they took −15% in one round with vg4-custody).  The warm-cohort
target for this cell (0.1866 s = 263.6 µs/xform) was already beaten by
both of us; the round's real race is intra-panel.  Round-start re-measure,
quiet window: **214.9 µs/xform... after the change; the r7 code re-read
231.2-231.5 in every window tonight** (three interleaved control runs,
below).  The r7 record's "next round" list named the mapped y-pass and
pack overheads; the actual win came from reading L45_mixedradix's r7
record instead, exactly as the cumulative-round brief intends.

### What shipped: the qc chain family (FFT45_QC=1 default)

**BORROWED: L45_mixedradix ice_r7's vg4-custody schedule (transitively
L64_radix8 ice_r6 ckind=2), rebuilt on my r7 q4 point-major group layout
(their layout half I already had — my q4 IS their VG, independently
derived from rival v6_5a869e40).**  m+1 alternating-orientation sweeps
replace 2m volume sweeps:

    sweep 0      (P, fixed-x planes): z(1), y(1)               [carries x(1)]
    sweep s odd  (Q, fixed-y slabs):  x(s)+map, z(s+1), x(s+1) [carries y]
    sweep s even (P, fixed-x planes): y(s)+map, z(s+1), y(s+1) [carries x]
    sweep m: carried axis of step m RAW; q4_unpackm applies map(m).

The map always lands at the LAST axis of its step (the carried axis) with
the r5/r6 MPAIR stash + one vdivpd per 8 points, unchanged arithmetic.
Each 129.6 KB plane/slab (+ its 129.6 KB c image) gets its three passes
while L2-resident: the 5.8 MB group state crosses L3 once per step
instead of twice (~29 → ~17.5 MB of L3 traffic per group-step).  A fixed-y
slab is 45 chunks of 2880 B at 129,600 B stride (2624 mod 4096) — the
same full-width pencils as q4, so PFA45/PFA45R run verbatim; the z-site
keeps the DAG-by-y rule (PFA45R rows 0-43, PFA45 row 44) in both
orientations.

Supporting pieces:

1. **CQ pencil-order c for the Q sweeps' mapped x pass** (BORROWED:
   L45_mixedradix ice_r7 VGC, applied only where the addressing needs it).
   My own r6 vs2 negative — point-major c at x-store sites = loads
   scattered over 45 pages, +139 µs/vol — is exactly what CQ avoids:
   CQ[((y·45+z)·45+x)] makes every mapped pencil's c contiguous.  Built
   from C4 by a slab-local 45×45 zmm transpose once per group per call:
   **354 µs/group = 0.5 µs/xform amortized** (probe).  The P sweeps keep
   reading plane-local point-major C4 (see qc2 below for why that is
   enough).
2. **Custody is a NEW BIT CLASS** (step axis order alternates
   (z,y,x)/(z,x,y) with sweep parity — deterministic per (m,B), the
   L23_matrixsimd m-parity discipline), so FFT45_QC selects the family at
   COMPILE TIME exactly the way FFT45_XF already does: qc leads the arm
   table, sets the gate's bit reference, and every x-first/z-first arm
   rel-passes but bit-DQs (verified in the gate log: q4/xf/vt/vs all
   "BIT FAIL first at [0]", 1-ulp class differences).  qc and qc2 are
   bit-identical to each other and race in one pool.
3. **The create() gate now checks every arm at BOTH m and m+1**
   (reference chain extended one step; the L45_mixedradix m=2/m=3 gate
   discipline): all five sweep types and both final-sweep parities are
   exercised before any arm may run.  Cost: milliseconds, in setup.
4. **Rank swap in the x-first family (QC=0 builds): q4-pf now rank 0** —
   the r7 SCORED drained window read q4-pf 253.5 < q4 255.5 while my dev
   windows had a dead tie; per the r5 rule the rank moves the same round.
5. Arena grown to 4 regions (Q, C4, CQ, CP), every c image at page phase
   (phase(Q)+2048)&4095; hugepage madvise covers the whole span.

### Operation count (per volume-step, qc)

FFT unchanged from r7 q4: 1518.75 zmm calls/vol-equivalent × (344
FMA-port + 78 swaps), two-pipe floor ~110 µs/vol; map unchanged (~11.7k
vdivpd + ~210k pipe uops/vol, one divide per 8 points).  What changed is
only WHERE passes issue: 3 passes per plane per sweep, 1 sweep per step.
Per-call overhead now pack2 900.7 + packQ 353.6 + unpkm 636.6 ≈ 1.9
ms/group ≈ 2.7 µs/xform at m=177.

### Measured (a80n0 leased cores; tryout's $W bug is STILL live — seventh
### round — plus TWO NEW tooling breaks, see below; every verdict
### same-window, forced-arm A/B via FFT45_CH)

* Quiet-window driver minima, shipped build: **B=4 (graded) 215.079
  µs/xform (sd 0.02%, MKL 758.1 same window → 3.53×)**; first run of the
  round read 214.933 (MKL 758.9).  vs r7's scored 231.220 and its 232.8
  quiet-window equivalent: **−7.6%**, and −4.7% vs L45_mixedradix's r7
  scored 225.571.
* Same-window forced A/B, 3/3 interleaved pairs on one core:
  **qc 214.6 / 215.6 / 217.7 vs q4 231.2 / 231.4 / 231.5 (−6.0…−7.2%)**.
* **B=8 214.654** (two custody groups, sd 0.04%).  **B=1 276.9** (sd
  6.3%, unchanged xf fallback in the elevated node B=1 cell — r7 read
  264.9-265.1 in calmer windows; the path is byte-identical to r7).
* Probe decomposition (LOUD, µs/group of 4): sweepP=860.0 sweepQ=835.2
  packQ=353.6 | q4 steady twins in the same window: x=216.8 z=243.6
  ym=540.7 (sum 1001) — custody's one sweep replaces the three passes'
  worth of L3 exposure at ~85% of their cost.
* Correctness, shipped build: single 4.002e-16 (B=4) / 3.998e-16 (B=1) /
  4.001e-16 (B=8); **two-step gate (this round's new scoring gate)
  2.045e-15 vs tol 3e-14 (B=4), 2.055e-15 (B=8)**; whole-chain m=177
  **4.701e-14 vs tol 1e-10, at 1.05× the honest-library anchor
  (4.465e-14)** — the chain-class change from r7's 6.325e-14 is the new
  axis-order class, exact-tier by construction.  out.bin AND
  out.bin.chain bitwise identical across runs AND across different cores
  (cross-core cmp added this round).

### What did NOT work, with numbers (all forced-arm same-window A/Bs)

* **qc2 — pencil-order c for the P sweeps too (the other half of VGC):
  +0.8-0.9%, 3/3 (217.0-217.3 vs 215.2-215.5).**  The P sweeps'
  point-major C4 reads are plane-local 2880 B strides and were already
  OoO-covered; qc2 pays packP (~0.5 µs/xform) for nothing.  Mixedradix's
  −6% VGC number is therefore mostly the Q-orientation page-scatter
  class, which qc avoids by construction.  Kept in the pool as the raced
  bit-identical control.
* **qc-pf — next-plane/slab T2 pokes issued from the custody z pass:
  +20% (258.2/258.8 vs 215.9/216.9, 2/2 clean pairs).**  I re-tested the
  poke family because the custody z pass's OWN loads are L2-hit (the
  q4-cz starvation mechanism does not apply) — mixedradix's r7 record
  explicitly flagged the re-test as legitimate.  Verdict: the poke tax is
  ISSUE pressure, not cache pollution — 90 poke uops per ~500-cycle call
  in a pass already at ~2× its port floor.  Fifth poke-family negative in
  this lineage; removed from the pool, function kept as the priced
  negative.
* Tooling, for the next agent: (a) `reserve.sh --status` fails on wallaby
  (no slurm client) — L23_matrixsimd r7's `~/bin_shim/squeue` heartbeat
  shim (answers R while RESERVATION.heartbeat < 180 s old) makes tryout.sh
  work unmodified; I installed it this round.  (b) **check.py's m>2
  map-check path crashes with NameError: `math` is not imported** — the
  m≤2 branch works, so the two-step gate runs through check.py, but the
  m=177 chain check must be replicated by hand (numpy fftn vs
  V·conj(ifftn(conj))) — snippet in this round's transcript.  (c) tryout
  regenerates in.bin/c.bin per batch: rerunning a binary after a
  different-B tryout needs the inputs regenerated first.

### Borrowed this round (attributions)

* **L45_mixedradix ice_r7**: the custody schedule itself (their −7.4%
  layout+custody move), the m/m+1 create-gate discipline, and the VGC
  pencil-order c idea (adopted for the Q orientation, priced out for P).
  Transitively **L64_radix8 ice_r6** (ckind=2 custody),
  **L17_matrixsimd ice_r7** (volume-SoA lanes), **L23_matrixsimd ice_r7**
  (m-parity one-bit-class rule).
* **L13_rader ice_r7**: deterministic dispatch across bit-class
  boundaries (never race them) — realized here as the FFT45_QC
  compile-time family switch with the arm[0] bit-reference gate.
* The re-test licence for pokes under a changed shape:
  **L45_mixedradix ice_r7** next-round item 2 (verdict: still negative).
* Same-window incumbent-rebuild A/B protocol: **L45_mixedradix ice_r5**;
  squeue shim: **L23_matrixsimd ice_r7**.

### Prediction for the scoring window

Pick: ch=qc (qc2 within 1% behind, both printed in the desc; every other
arm OUT by design — they are the other bit classes).  Scored B=4 ≈
**213-219** if the window is quiet (my quiet-class 214.7-215.1); the desc
prints ch=... ms0 xf1 qc1 plus the arm table.  Fingerprints: single
4.002e-16, chain 4.701e-14 (B=4).  Setup 1.1-1.5 s (unscored).

### Next round

1. **The mapped pass is still the fat target** (sweepP 860 vs its ~475
   µs/group compute+map floor).  The one untried structural idea: issue
   the carried-axis map's ladder/divide chains INTERLEAVED into the
   following z pass's codelet text (true software pipelining, not pokes,
   not staging) — big macro surgery on PFA45's call site; read the r5/r6
   staged-map traps first (+90 µs/vol) — the difference here would be
   register-level interleave, not a separate span.
2. pack2 (901) is now the largest per-call overhead; a lane-packing
   variant that reads x0/c through 64 B loads + 2 permutex2var per point
   pair (instead of 4×16 B broadcasts) is a ~30-line experiment worth one
   A/B (~1.3 µs/xform at stake).
3. If the scored spread stays >2%, read clk= in the desc before touching
   code (L36_pfa's rule) — tonight's windows drifted 757-797 on MKL while
   in-run sd stayed ≤0.1%.
4. B=1 still runs the r6/r7 xf path (276.9 in tonight's elevated cell).
   Unchanged advice: only worth touching if a B=1-graded panel appears.
5. check.py's missing `import math` (m>2 map-check) needs a central fix —
   flag it to the monitor rather than patching per-implementer.
