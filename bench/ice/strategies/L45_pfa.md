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
