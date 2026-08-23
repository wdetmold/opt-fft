# L45_mixedradix — strategy record (ICE LAKE panel)

Lineage: the serial PFA 9x5 kernel was developed over geom rounds panel_r6..r11 on
Cascade Lake (`bench/geom/strategies/L45_mixedradix.md`) and left arithmetically
untouched through the multicore phase (`bench/mt/strategies/L45_mixedradix.md`).
Do not re-derive that history; read it.

## Round ice_r1 (no agent work — scored the geom r11 code as-is)

The unmodified r11 build took first at L=45 on the new node: **284.202 us/transform**
(graded chain B=4 m=177), 1.04x ahead of L45_pfa (295.054), 1.84x ahead of MKL
(521.591). Tuner picked v1-pf0 from the 4-candidate cached pool; the in-plan probes
put the B=1 split at fu=273.5, p1=208.1, p2w=67.9 us — phase 1 is three quarters of
the time. fe=na on the node (see r2: the reason turned out to be EACCES).

## Round ice_r2 — transpose-free phase 1 (tr) + overlap-mechanism stack

### The lever (borrowed, with attribution)

**BORROWED from L36_pfa ice_r2's tr=1** (their strategy record, written this
round): on this node (Xeon Gold 6326, bare-metal ICX-SP) the second 512-bit FMA
pipe lives on PORT 5 — the same and only port that executes 512-bit shuffles —
and, unlike the VM grading tier of corpus §10, `vbroadcastf64x2 (mem), zmm{k}`
folds into ONE load-port uop.  My phase 1 carried 90,090 port-5 transpose
shuffles per volume (495 z-groups x 182: TRANSP-in 88+3, TRANSP-out 88+3) on
top of the 116,766 codelet swaps.  tr=1 rebuilds phase 1 as:

* z subloop: every lane-transposed input vector = 4 masked 128-bit broadcast
  loads straight from the 4 y-rows (lanes = y, zero port-5); codelet output
  stored UNtransposed as full-width 64B stores to a slot-major plane
  (slot s = ygroup*45 + kz holds one PW-lane vector).  Both TRANSP passes AND
  the Xv/Yv spill arrays vanish; the odd 45th column needs no special form at
  all (every element goes through the same gather).
* y subloop: re-gathers element (y=i, kz=z0+j) from slot ((i/PW)*45 + z0+j)
  lane i%PW, again as 4 masked broadcasts per vector; output stores unchanged.
* xmm tails (y=44 z-line, kz=44 y-line) get tr twins addressing the slot
  layout (dft45_tz1t8/ty1t8, PW=2 twins *t4).  Phase 2 untouched.
* PW=2 variants get an LDCOL-based tr path for correctness/pricing only
  (no masked-broadcast win at 2 lanes; v0 is plain AVX2 anyway).

Port arithmetic per volume at PW=4: p5-only shuffles 206,856 -> 116,766 (the
codelet swaps are now the entire term); two-port floor (514,968+116,766)/2 =
315,867 cyc = 108.9 us at 2.90 GHz (was 124.5).  Load uops rise ~132/z-call
and ~135/y-call (broadcast granules vs full-width loads), well under the
2x64B/cyc bare-metal load ports.  Codegen verified on the .o before touching
the node: 360 vbroadcastf64x2 in exec_1_14, ALL memory-operand, and exactly
4x78 = 312 remaining shuffle-class instructions = the codelet swaps.

Integration: tr is mechanism axis 14, twins 15-21 stack it with the existing
memory mechanisms; FFT45_TR=0|1 env override; -DFFT45_DEBUG (new) forces the
tuner table + probe line to stderr so tryout.sh (which cannot pass env over
ssh) can read them.

### Node measurements (tryout.sh = leased core on a80n0, graded B=4 m=177
chain --unitary; same-process arena tables are the trustworthy ranking,
absolute levels drift ~8% between windows/cores)

Window 2 (quiet, MKL sd 0.02%), us/vol in-arena:

    v1-tr-pkw-pfw    252.1      v1-tr-pf1       259.6
    v1-tr-pf1-pfw    253.5      v1-tr           264.3
    v1-tr-pfw        257.0      v1-pf1          275.1
    v1-tr-pf1-pfin-pfw 273.2    v1-pfw          278.8
    v1-pf0 (r1 pick) 280.3      v2-pf0          301.1

**tr-pf1-pfw / tr-pkw-pfw are tied at -9.6% vs the r1 incumbent** (they
differ by 0.6%, inside noise; pf1-pfw heads the pool, pkw-pfw is slot 2).
Same ranking in two earlier windows (tr-pfw -10.3% and -9.5% vs pf0 before
the stacks existed).  Driver on the graded chain, same quiet window:
**min 275.1 us/xform** (old code measured 284.9 in the equivalent earlier
window; r1's scored number was 284.2).  Correctness: rel_l2 = 4.058e-16,
chain m=177 closed-form 2.130e-14 (tol 1.3e-11), bit-repeatable, all
candidates pass the create()-gate at 1e-13.

B=1 (m=177 chain, core 3): pick v1-tr-pf1-pfw, in-arena 247.1 vs pf0 280.7
(**-12.0%**); driver min 244.5 us/xform (median noisy, 9.8% — that core was
toggling; min is the honest number per this entry's long-standing protocol).
rel_l2 4.065e-16, chain 2.132e-14, repeatable.  MKL same window: 503.9.

In-plan nv=1 probes (new p1z/p1y/p1t/p1zt/p1yt subpass splits, shape from
L36_pfa): fu=248.4  p1=204.0  p1t=190.5  p1z=99.7->p1zt=74.3 (z subloop
-25%)  p1y=94.0->p1yt=105.4 (y subloop +12%: the corner turn moved into the
y-gather)  p2w=68.1.  The tr win is entirely on the z side; the ~13 us
y-side regression is the once-per-plane corner-turn cost, which lands
cheaper as z-side load uops than as z-side port-5 shuffles.

### Why the overlap mechanisms suddenly matter (pfw +7%, pokes +1-3%)

With the shuffles gone, every pass still sits ~1.8x above its port floor
(z 74 vs ~41, y 105, p2w 68 vs ~37) and the *uniformity* across three
differently-shaped passes plus the fact that only latency-overlap mechanisms
(pfw on the cold-out RFO, pf1/pkw pokes on phase 2's 45 L3 streams) move the
needle says the B=1 chain is substantially L2<->L3 transfer/latency bound:
the 2.78 MiB per-volume working set does not fit the 1.25 MiB L2 (unlike
L=36), and ~5 MB/volume cycles through L3 every step.  This is also why
L36_pfa's "prefetch is pure uop tax on cache-resident cells" does NOT
transfer to L=45 unmodified: their volume fits L2, mine does not.  pfw was
in r1's pool but only ever offered without tr; under tr it is worth -2.8%
(264.3 -> 257.0) and the pf1 poke another -1.4%.

### What did NOT work / negative results with numbers

* **tr-cpy (ERMS no-RFO plane copy): 317.2 vs tr 287.2 us/vol** (same
  window) — re-retired.  pfw's win said the out-RFO stalls; cpy deletes the
  RFO but pays a full extra 65 KB/plane L1->out copy, and on this
  L3-resident chain that costs 3x what it saves.  Second retirement (r9 was
  the first); do not re-propose without a DRAM-streaming cell.
* **tr-pf1-pfin-pfw: 273.2 vs tr-pf1-pfw 253.5** — the paced T1 input
  prefetch is a pure uop tax here (the in-read is one linear
  HW-prefetchable stream), consistent with pfin never winning a node cell
  since panel_r7.
* **perf_event_open fails with EACCES (errno 13) on the node too** — the
  fe=na mystery of four rounds is settled: perf_event_paranoid blocks
  unprivileged counters on a80n0 despite the brief's PMU note.  The port
  counters I added this round (UOPS_DISPATCHED.PORT_0/5, 2_3, 4_9) are in
  the code, gated the same way, in case the monitor runs privileged.
* Extract-store z-output (STCOL, considered on paper): 135 p5 extracts +
  180 store uops per group vs tr's 0 p5 + 45 full stores — L36_pfa's
  "broadcast-on-load strictly better than extract-on-store" holds at L=45;
  not fielded.
* Instruction-bytes audit (disp32 broadcasts): 1673 of the tr build's
  broadcasts encode at 10-11 B (row offsets > disp8*16 range).  Not
  attacked: corpus §10 layer 3 refuted the decode-bound hypothesis on the
  VM tier, the z body (~6 KB) fits the DSB per subloop, and without PMU
  access (EACCES) there is no way to convict MITE.  Noted for a future
  round if counters ever work.

### Operation count (per volume, PW=4, tr path)

Unchanged FMA-class work: 1497 full-width calls x 344 zmm FMA-port ops +
90 xmm tail lines (514,968 zmm ops).  Port-5-only: 116,766 codelet swaps
(was 206,856 with transposes).  Loads: +132 per z-call / +135 per y-call
(16 B broadcast granules, same cache lines).  Stores: z-subloop halved
(90 vs ~184: Xv/Yv spills gone).  Plane scratch: slot-major, 540 slots x
64 B = 34,560 B, fits the existing allocation; PPITCH is irrelevant on the
tr path.

### Borrowed this round (attributions)

* **L36_pfa ice_r2**: the entire tr concept (masked vbroadcastf64x2 gather,
  untransposed slot-major intermediate, broadcast re-gather on the second
  subloop), the codegen-verification-before-node discipline, and the
  p1z/p1y probe split.  Adapted for L=45: odd tail lines (their L=36 has
  none), 9x5 PFA index maps in the gather addressing, and tr as a
  stackable mechanism axis rather than a plan dimension.
* **Corpus §10**: the bare-metal "broadcasts fold into the load uop"
  proviso underlying tr, and the port-5 shuffle/FMA conflict.
* **L23_matrixsimd ice_r2**: the "within-window contrasts only" measurement
  discipline (their dev-window offset caution), and the negative result on
  -fschedule-insns for fully-unrolled straight-line kernels (mine is the
  same shape; not attempted, corpus also says it hurts L=45).

### Next round

1. **Split-complex (SoA) codelet lanes** — the 116,766 codelet swaps are
   now the whole p5-only term (40 us of p5 floor).  L36_pfa's next-round
   item too; L64_radix8 already runs split-complex on this panel.  Full
   layout rework of both phases; only worth it if the L3-transfer wall
   (item 2) is not the binding constraint.
2. **The ~1.8x-over-floor residue is memory-system, not ports**: probes
   and mechanism sensitivities point at L2<->L3 traffic (5+ MB/volume).
   Structural options: none found this round that beat the pokes (the
   corner turn must be paid once; phase 2 cannot tile below 45 streams
   without an extra pass).  If the monitor can run ONE privileged
   perf-stat of the picked exec, the port/DSB/L2-miss split in the
   description decides between items 1 and 2 for free.
3. Streaming pool: node-raced once at B=16 (46.7 MB, stream=1) as a sanity
   check — **v1-tr-pf1-pfw won there too**, 346.7 vs the four-round
   incumbent v2-pf1-pfin-pfw-oc's 380.0 us/vol (-8.8%), driver 382.7
   us/xform, PASS/repeatable.  The tr rows are now proven streaming
   challengers; if a streaming cell is ever graded, consider promoting a
   tr-pfin twin raced fine-paced (pfin's coarse pacing is the one untested
   variable there).

### Shipped state

Pick on the graded cell (B=4 m=177): **v1-tr-pf1-pfw**.  Node driver
minima this round: B=4 275.1 us/xform (quiet window; r1's same-window
equivalent was 284.9, scored 284.2 — expect ~255-263 scored if the -9.6%
in-arena delta carries), B=1 244.5, B=16 382.7.  rel_l2 4.056-4.065e-16
everywhere, chain checks 2.13e-14, bit-repeatable, setup 0.4-0.8 s.

## Round ice_r3 (no agent section was written)

The r2 code ran unchanged and scored **259.287 us/xform** (graded B=4
chain), first at L=45, 1.13x ahead of L45_pfa, 2.01x ahead of MKL.  Noted
here so the gap in the record is explicit.

## Round ice_r4 — the task became the full rival step: fft3d_chain, map fused

The graded step is now `state <- (z+c)/(1+|z+c|)`, z = raw FFT(state), and
the driver times an exported `fft3d_chain(plan, x0, c, final_out, m)` for
the whole m=177-step chain (fallback = fft3d_execute + a driver-side map
pass; MKL through that fallback read 751-820 us/xform across this round's
windows).  The rivals' full-task time at L=45 is 0.201 s = **283.9
us/xform**.  Everything this round is that entry point; the fft3d_execute
tournament path is untouched (it still runs the single-transform gate).

### What shipped

1. **fft3d_chain, per-volume order** (corpus 10 s3 consensus; every ice_r4
   entry adopted it): volume b runs all 177 steps before b+1.  The state
   lives IN ONE private volume: pass A transforms it in place, pass B
   rewrites it plane by plane (below), so the per-volume working set is
   state 1.46 + c 1.46 + scratch ~0.04 = **2.9 MiB, L3-resident for the
   whole chain** (a ping-pong pair would make it 4.4).  x0 is memcpy'd in
   once per volume (1/177 of a step); the last step's final pass writes
   final_out directly, so the driver's buffer takes one RFO per volume.
2. **x-first pass order, map fused into the y subloop's stores (CXF=1).**
   The FFT is separable, so the step runs x-lines FIRST (the old phase 2,
   in place, unfused), then per x-plane: tr z subloop -> slot-major
   scratch -> **mapped** y subloop back onto the same plane.  The map's
   ~13 vector-uops-per-4-points ride the y pass's gather-latency slack
   (p1yt probe: ~105 us against a ~41 us port floor = ~65 us of slack; the
   old phase 2 had only ~31), and c is read at plane-matching offsets.
   Node A/B, same knobs otherwise: **x-first 287.1 vs fused-phase-2 347.0
   us/xform** — the SAME map arithmetic costs ~60 us less in the pass with
   issue slack.  The one odd y-line per plane (2025 of 91125 points) maps
   through an exact 128-bit sqrt+div tail twin.
3. **Map arithmetic (CMS=0): s = |w|^2 + 1e-300, vrsqrt14pd double seed +
   2 Newton, d = fma(s,y,1), ONE exact vdivpd.**  Budget arithmetic as the
   brief demands: seed 2^-14 -> 5.6e-9 -> 4.7e-17 = sub-ulp, so the map is
   ~2-3 ulp per application against a 1e-13/step budget (tol 1.77e-11 at
   m=177).  Measured whole-chain drift: **4.795e-14 (B=4) / 2.794e-14
   (B=1) — ~370x margin.**  The additive 1e-300 bias replaces both a max()
   clamp and the rsqrt14(0)=inf NaN trap (L17_winograd's form).  The
   rivals' float-seed tier is legal at this (L,m) but saves nothing: the
   14-bit double seed costs the same two Newtons.
4. **State page-phase derived from c at run time.**  The driver's buffers
   are only 64B-aligned, so their mod-4096 phase is heap luck; when c's
   phase landed near the state volume's, pass B's c loads false-depended
   on the plane stores at equal page offsets: **294.1 vs 333.0 us/xform
   across two runs of the SAME binary at B=1**, in-run sd 0.05-0.07% both
   times (L23_rader ice_r4's pathology, reproduced).  fft3d_chain now
   places the state at ((c & 4095) + 2048) & 4095 into its 4096-aligned
   block — maximally far from c for every batch's volume phases (volume
   stride 1458000 = 3920 mod 4096 spreads them only +-528 B).  Post-fix
   B=1: 292.6 / 282.4 across runs.  The plane scratch allocation was
   bumped to 4096-alignment so its phase is deterministic too.
5. A create()-time gate runs one fused step against exec_0_0 + the exact
   scalar map (1e-13) and falls back to the exact-map classic step on
   mismatch; all chain knobs are compile-time so chain output is
   bit-identical across runs and processes (verified: two-run cmp of both
   out.bin and out.bin.chain identical).

### Operation count (map add-on, per volume, PW=4)

495 mapped y calls x 45 stores = 22,275 zmm map sites x (~12 FMA-class ops
+ 1 vpermilpd + 1 c load + 1 vdivpd) ~= 290k vector uops + 22.3k divides;
plus 2025 xmm tail sites x ~7 ops exact form.  FFT vector work unchanged
(514,968 zmm FMA-port ops + tails).  Measured full-step cost over the
unfused FFT: ~+25-40 us depending on window — against a ~+50 us pure issue
floor if the map had ridden zero slack, and +88 us measured when it sat in
phase 2.

### Measured on the node (a80n0 leased cores; same-window MKL-fallback
### minimum quoted as the window anchor; windows drifted 751-820 tonight)

| config (graded B=4 m=177) | us/xform | MKL anchor |
|---|---|---|
| classic fused-p2, CMS0 | 347.0 | 757.7 |
| classic fused-p2, CMS1 (rcp ladder) | 387.8 | 765.1 |
| x-first + c-plane T1 pacing + prefetchw pokes | 319.7 | 809.8 |
| x-first, no c pacing, prefetchw pokes | 299.5 | 806.2 |
| x-first, no c pacing, T0 pokes (= shipped) | 287.1 | 757.6 |
| **shipped defaults, confirm run** | **283.99** (sd 0.07%) | 758.5 |
| B=1 shipped (post-phase-fix, two runs) | 292.6 / **282.4** | 762.2 / 751.7 |

Correctness: single transform 4.058e-16 (B=4) / 4.065e-16 (B=1); map-chain
m=177 4.795e-14 (B=4) / 2.794e-14 (B=1) vs tol 1.8e-11; bit-repeatable.
Setup 0.36-0.50 s.  **At the rivals' 283.9 us pace at full double
precision** (their fastest drifts to 1.28e-8 on long chains; ours cannot).

### What did NOT work, with the number that killed it

* **Map fused into phase 2's in-place stores (the classic order): 347.0 vs
  287.1.**  Phase 2 has ~31 us of slack against its port floor; the map
  needs ~50.  Placement, not arithmetic — same ops, -60 us, by moving them.
* **CMS=1, rcp14+2NR instead of the exact vdivpd: +25 us in the y pass
  (312.2 vs 287.1), +40 us in phase 2 (387.8 vs 347.0).**  The step is
  issue-bound, not divider-bound: L23_matrixsimd's mv=0 verdict transfers
  to L=45, L23_rader's mp=2 verdict does not.  One hidden divide beats six
  extra FMA-pipe uops.  (CMS=2, hw vsqrtpd: not fielded — two entries
  measured it at +18-19%.)
* **Pacing a T1 prefetch of the next c plane over the z subloop (CPFN):
  +12-20 us** (299.5 -> 287.1 in adjacent windows; 319.7 with it plus
  prefetchw).  pfin's "pure uop tax" history extends to the c stream.
* **prefetchw instead of T0 for pass A's 45 RMW-stream pokes (CPKW):
  ~+12 us** (299.5 vs 287.1, adjacent windows).  Surprising — pkw tied
  tr-pf1-pfw in r2 — but pass A's lines now come back MODIFIED from pass
  B's rewrite two passes ago, so the RFO the prefetchw would save is
  often not there.
* **T0-poking the y subloop's 45 c rows one call ahead (CPFY): 298.2 vs
  291.5** back-to-back.  The c loads are 45-way parallel and OoO covers
  their L3 latency; the poke is 22k uops/volume of tax.
* Poke distance 3 (298.6, contended) and no pass-A poke at all (298.3 vs
  287.1): distance-1 T0 stands.

### Borrowed this round (attributions)

* **The whole round-shape** — export fft3d_chain, per-volume chaining,
  fuse the map where values are in registers, rsqrt14 double seed + 2NR +
  one exact divide, the 1e-300 bias, do-the-budget-arithmetic-first, and
  the "all knobs compile-time for bit-identical outputs" discipline:
  L13_rader / L17_winograd / L23_matrixsimd / L23_rader ice_r4 records,
  themselves from corpus 10 s2 and ext/reference pw_full.  Their records
  also flagged both tryout.sh chain bugs (W unbound under set -u; check.py
  gets an unexpanded '$W/c.bin' so the map check and the repeatability cmp
  silently die) — I used their `W=... ./tryout.sh` workaround and ran
  check.py + the two-run cmp by hand every time.
* **The mod-4096 state-phase pathology and its diagnosis**: L23_rader
  ice_r4.  Extended here: derive the phase from c at chain time instead of
  hoping a fixed skew stays lucky (their STOFF knob is a probe; this is
  the "plan-time self-tuned base phase" they proposed, computable in
  closed form because only st-vs-c matters in this step shape).
* **Map placement follows slack, not pass identity**: L17_winograd's
  "+0.44 us because the fused pass is store-bound and the map rides idle
  pipes" — the same reasoning picked the y subloop here, and it was worth
  17% of the whole step.

### Next round

1. **The map now costs ~+25-40 us against a ~+15 us ideal** (pure c-read
   traffic + divider occupancy if everything else hid).  The remaining
   lever is pairing: compress two output vectors' |w|^2 into one zmm
   (2 two-source shuffles + 1 add), run ONE rsqrt ladder per 8 points,
   expand with movddup/permute — halves the ladder+divide count for ~4
   port-5 shuffles per pair.  Needs the y-subloop SDST macro restructured
   to emit stores in pairs; L23_rader's PW_CORE extension is the model.
2. The B=4 confirm run hit 284.0 with in-run sd 0.07%, but adjacent
   windows wobbled 287-294: if the scored number lands >290, suspect the
   remaining phase interactions (in/out driver buffers vs the state; only
   c is controlled today) before suspecting the code.
3. fft3d_execute's tournament still prices tr-pf1-pfw for a regime that is
   no longer scored; if a future round frees time, either retire the
   unused mechanisms or re-shape the tournament to race chain steps.
4. Streaming (B=16) was not re-raced this round — the graded cell is
   B=4 and the chain is per-volume, so batch size only changes c-phase
   dilution.  B=1 and B=4 land within 2 us post-fix.

## Round ice_r5 — pair-compressed map: one ladder + ONE vdivpd per 8 points

r4 scored 283.800, a dead heat with L45_pfa's 283.339 and exactly at the
rivals' 283.9 us pace.  This round executed r4's own "next round" item 1
with the shape the whole panel converged on, and it was worth far more
than the uop arithmetic predicted.

### What shipped

**FFT45_MPAIR=1 (default): the mapped y subloop's stores leave in pairs.**
ST2GMAP = the n1_9 DAG with the 9 outputs of each block flushed as 4 pairs
-- (k1=0,3), (1,4), (6,7), (2,5) -- plus the k1=8 single, so 40 of every 45
points share one map ladder.  MPAIR compresses two output vectors' |w|^2
lane-pair sums into ONE zmm (vunpcklpd + vunpckhpd + vaddpd: s_ =
{sa0,sb0,sa1,sb1,...}), runs ONE rsqrt14+2NR ladder and ONE exact
vdivpd(1,d) for all 8 points, and fans the scale back out with 2
vpermilpd-imm (VDUPE/VDUPO) + 2 muls.  o6 is deferred one sub-block to meet
o7; every other pair is adjacent in the DAG, so no long live ranges.  The
1e-300 bias lands twice per point (once in each unpack summand),
reproducing the r4 mapv bias exactly.  Bit class changes (w*(1/d) instead
of w/d, ~1 extra ulp); measured chain drift IMPROVED: 4.214e-14 (B=4) /
2.804e-14 (B=1) vs r4's 4.795e-14 / 2.794e-14, tol 1.77e-11 -- ~420x
margin.  Codegen verified on the .o before the node (r2 discipline):
exec_1_cstepx carries exactly 25 vdivpd + 25 vrsqrt14pd + 20 unpack pairs
per y-body, stack traffic unchanged (190 rsp-refs), 139 FEWER instructions
than r4's body.

### Operation count (map, per volume, PW=4)

495 mapped y calls x (20 pairs + 5 singles): vdivpd 22,275 -> 12,375
(-44%), rsqrt ladders likewise; per pair 16 FMA-class + 4 p5 + 1 divide
vs the r4 form's 24 FMA-class + 2 p5 + 2 divides per 8 points => ~79k
FMA-class uops deleted, +19.8k p5 (unpack/expand), 9.9k divides deleted.
xmm tail sites (2025/volume) unchanged: exact per-point sqrt+div.  FFT
work untouched (514,968 zmm FMA-port ops + tails).

### Measured on the node (tryout.sh leased cores, graded B=4 m=177 chain;
### within-window contrasts ONLY -- see the drift trap below)

| config | us/xform | window anchor (MKL fallback) |
|---|---|---|
| r4 map (MPAIR=0), fast window | 283.556 | 758.1 |
| **paired map, same window** | **263.064** (sd 0.08%) | 760.7 |
| r4 map (MPAIR=0), drifted window | 292.935 | 760.5 |
| paired map, drifted window | 272.220-272.960 (sd 0.06%) | 760.4-761.8 |

Within-window contrast: **-7.2% / -7.1%** in both machine states.  B=1:
272.604 min (core 2) / 273.268 min (core 3, noisy median) -- tracks B=4 as
in r4; one core-3 window read a flat 312.2 (sd 0.06%), see next-round
item 2.  Single-transform rel_l2 4.058e-16 (B=4) / 4.065e-16 (B=1); chain
checks above; out.bin AND out.bin.chain bit-identical across two
processes (check.py run by hand both batches -- tryout's $W bug is STILL
live in r5, same workaround as r4).

### The measurement trap this round documented: MKL does not track the drift

Byte-identical .text measured 263.064 and then 272.2-274.5 in later
windows while the MKL anchor stayed flat (758-762) and in-run sd stayed
0.06-0.29%.  So a ~3.5% machine-state drift exists that the same-window
MKL number does NOT expose -- an A/B whose sides straddle it gets a
confident wrong answer with clean error bars.  It cost me one wrong
verdict for ~40 minutes (tail pairing read as +9 us; the within-window
rerun read a dead tie).  Rule sharpened from L23_matrixsimd r2's
"within-window contrasts only": REBUILD AND RERUN THE INCUMBENT in the
same window as the challenger; never trust the anchor to certify the
window.

### What did NOT work, with the number that killed it

* **FFT45_CMS=2 paired (hw vsqrtpd + vdivpd instead of the rsqrt ladder):
  303.2 vs 263.1/272.2 surrounding windows (+11-15%).**  The sqrt->div
  divider chain serializes right before the stores -- L23_rader's r4
  finding, reproduced at pair granularity.  L23_matrixsimd ice_r5's
  "ladder + one exact divide" ranking transfers to this store tail; their
  in-situ-race discipline is why it was raced and not assumed.
* **FFT45_TPAIR (pairing the 2025 xmm tail sites): dead tie, 272.248 vs
  272.220 same window.**  Too few sites; default OFF keeps the simpler
  exact bit class.  (First reading of +8-9 us was the drift trap above.)
* **FFT45_MPAIR8 (pair the five k1=8 leftovers across K2 parity through a
  stash register): 273.202/273.477 vs 272.220 (+0.4%, 2/2).**  o8s's live
  range across a whole ST2GMAP expansion costs more than two saved
  ladders+divides per line buy.  Knob kept, default OFF.

### Borrowed this round (attributions)

* **L23_rader ice_r4**: the L23R_MAP2 pair-compression shape verbatim
  (unpacklo/unpackhi+add compress, one ladder per 8, 2 expand shuffles +
  2 muls), itself extending the rivals' PW_CORE (1760b1bf generator).
* **L23_matrixsimd ice_r5**: "map-variant rankings are store-tail-shape
  dependent -- pin-race them in situ" (drove the CMS=2 A/B), and the
  confirmation that rsqrt-ladder + ONE exact vdivpd wins in a mapped
  store tail.
* **L45_pfa ice_r4**: corroboration that one divide per 8 points is the
  panel-consensus shape (their lazy z-load map), and the reassurance that
  w*(1/d) vs w/d bit-class changes are chain-gate-safe at ~500x margin.

### Next round

1. **Split-complex (SoA) chain state between steps** -- L17_winograd
   ice_r5 proved the chain OWNS the state format (only step 1 reads and
   step m writes interleaved).  For L=45 that would delete the 116,766
   codelet VSWAPs/volume (the ENTIRE p5-only term, ~40 us of p5 floor at
   2.9 GHz) at the price of a full two-phase kernel rework.  This is now
   the only structural lever of that size left on the table.
2. **The flat-312 B=1 window (core 3)**: either a core-state artifact or
   a surviving phase interaction (driver in/out vs state -- only c is
   controlled today).  If the scored B=4 number lands >285, suspect this
   before the code; a plan-time raced state-base phase (L23_rader's
   self-tune idea) is the structural fix.
3. The map is now ~12.4k divides + ~210k pipe uops/volume; the remaining
   map fat is the 5 per-call singles and the 2025 exact tail sites --
   both measured at <=0.4% each this round.  Diminishing; item 1 is where
   the next 5%+ lives.
