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

## Round ice_r6 — cache-line-padded chain state (rows 45 -> 48): every hot access aligned, bit-identical

r5 scored 262.959, first by 0.3% over L45_pfa (263.807).  This round's
lever came from two rivals' r6 records, not my own list: L17_rader r6
found a 16-B misalignment was costing them 2.3 of 15 us (every access
line-split), and L23_rader r6 shipped padded state rows (23 -> 24
complex) so every chain store lands on whole lines.  My chain state rows
are 45 complex = 720 B — NOT a multiple of 64 — so ~3/4 of every
full-width access in the fused step was line-split by construction:
pass A's x-line loads AND stores (stride 32400 B == 16 mod 64, ~34k
split accesses/volume) and pass B's mapped y stores + c loads (stride
720 B, ~33k more).

### What shipped (FFT45_CPAD=1 default)

**ZPITCH=48 padded chain-state layout** (768 B = 12 whole lines per row)
in an owned volume, plus an **owned identically-padded c copy** refreshed
once per volume-chain, plus fixed compile-time page phases (state +1024,
c +3072, scratch +0 mod 4096 — the r4 derive-from-driver-c phase game is
gone because no driver buffer is on the hot path any more).  48 is a PW
multiple, so pass A retiles to 12 full groups per row: NO masked tail
(was 1/volume), 540 calls vs 507 (+6.5% of pass A, the 3 pad lanes DFT
zeros; pads stay zero forever since pass B never writes them and
DFT(0)=0).  x0 in / c in / final state out are strided row copies, 3 x
1.46 MB per 177-step chain (~1/60th of one step).  Everything else —
codelets, map, pairing, poke shapes — is byte-for-byte r5's.
**Bit-identity verified, not assumed**: the CPAD=1 and CPAD=0 binaries'
chain outputs cmp identical (lanes are independent lines; regrouping
pass A's tiles cannot move a bit), and the node chain drift digits are
exactly r5's (4.214e-14 B=4).

### Operation count

FFT + map arithmetic unchanged (514,968 zmm FMA-port ops + 90 xmm tail
lines; 12,375 divides).  Pass A +33.75 full calls/volume (+11.6k zmm ops,
~2 us of two-port floor) buys: ~50k line-split loads/stores and ~17k
split c loads deleted per volume-step, the masked x-tail call deleted,
and pass A's 45 streams mutually 4K-alias-free by arithmetic (stride
34560 mod 4096 = 1792, gcd 256 — (i-j)*1792 never == +-64 mod 4096).

### Measured on the node (tryout leased cores, graded B=4 m=177; the r5
### drift trap governed everything — the machine toggled fast(~263)/
### slow(~271-275) states across windows on BYTE-IDENTICAL code again,
### so only same-window rebuilt-incumbent contrasts are quoted)

| contrast (same window, same core) | padded | r5 code (CPAD=0) |
|---|---|---|
| B=4, drifted state | **270.473** (sd 0.07%) | 273.068 (sd 0.13%) |
| B=1 (node B=1 cell elevated, see below) | **303.304** | 307.408 |
| B=4 fast-state absolute (no contrast) | 262.775-264.949 across 4 runs | (r5's fast-state was 263.06) |

Pad is worth **-1.0% (B=4) / -1.3% (B=1)** within-window.  Final confirm
runs: 275.348/275.989 (slow window), single 4.058e-16, chain m=177
4.214e-14 vs tol 1.77e-11, out.bin bit-identical ACROSS two processes on
different cores, check.py run by hand both batches (tryout's $W bug is
still live in r6; same workaround as r4/r5: `W=<abs builddir> ./tryout.sh ...`).

**Node B=1 cell is elevated right now for everyone**: the CPAD=0 binary
(r5-equivalent) reads 307.4 vs its r5-round 272.6 in the same windows;
L23_rader r6 documents the same phenomenon at L=23 with their unmodified
r5 exemplar.  If the scored B=1 lands ~300, it is the node, not the code.

### What did NOT work, with the number that killed it (all 2/2 or worse)

* **FFT45_COV — plane-pipelined pass B** (plane x+1's z subloop
  interleaved call-by-call with plane x's mapped y subloop, double-
  buffered scratch, bit-identical): **271.5 vs 267.9 and 286.3 vs 271.1**
  same-core pairs.  The interleave ages the scratch by a whole plane, so
  the y subloop's 180 16-B gather granules/call drop from L1 to L2 —
  costs more than the z-side L3 overlap buys.  The L13/L17 "ov" win does
  NOT transfer to a scratch-coupled two-subloop pass.  Knob kept.
* **FFT45_CPFZ — T1 pacing of plane x+1's state rows over the y subloop**
  (the read-side twin of r2's pfw win): **288.2/288.4 vs 270.2/274.7
  (+6.5%, sd 0.05%)**.
* **FFT45_CPFY re-raced on the now-aligned c rows**: **293.5 vs 271.4
  (+8%)**, worse than r4's +7 us on unaligned c.
  Together these settle a diagnosis worth keeping: the y subloop's
  "slack" is ISSUE slack only — its fill buffers are fully spoken for by
  the c misses, and ANY added prefetch there is poison.  Do not aim
  another prefetch at pass B; the remaining z-subloop L3 latency can only
  be attacked structurally.
* **FFT45_CPFD=2 on pass A's aligned streams**: 264.6/264.9 vs 262.8/264.2
  — wash-to-loss, distance 1 stands.  CPF=0: 271.3 vs 270.6, poke keeps
  its small win.
* **Split-complex (SoA) state — rejected by arithmetic before building**
  (r5's next-round item 1): the z subloop's corner turn pins the state to
  128-bit complex granules.  In SoA the lane-transposed gather needs 8
  vbroadcastsd merges per vector instead of 4 vbroadcastf64x2 (+~90k load
  uops/volume, ~15 us of load-port floor) to delete 116.8k p5 swaps whose
  two-port floor share is ~20 us — net ~5 us on PAPER, on passes measured
  1.8x ABOVE port floors, i.e. the ports are not what binds.  Moving the
  split boundary into the y subloop's stores just relocates the shuffles.
  A full custody-chain layout redesign (L64_blocked's zsplit) is the only
  form that could pay, and it is a whole-round job with the gather
  arithmetic above as its gate.

### Borrowed this round (attributions)

* **L23_rader ice_r6**: padded state rows to whole cache lines + "pad
  columns computed, never read" tiling; their B=1-cell-elevated node
  caveat, reproduced here and cross-cited.
* **L17_rader ice_r6**: the split-access forensics that made alignment
  the round's lever (their 16-B-off arena cost 2.3 us of 15).
* **L23_matrixsimd ice_r6**: the owned padded c copy refreshed per
  volume-chain (their p->ca), including the in-bounds-by-construction
  argument for reading pad columns of an owned buffer.
* **L13_rader ice_r6 / L23_matrixsimd r5**: the within-window
  rebuilt-incumbent protocol (my r5 rule, now panel practice) — every
  verdict above is a same-window pair.

### Next round

1. **The chain is at its structural traffic minimum** (2 sweeps x RMW
   state + c = ~7.8 MB/step through L2<->L3; ~264 us fast-state = ~29
   GB/s effective).  The prefetch space around it is now EXHAUSTED
   (CPFZ/CPFY/CPFN/COV all strongly negative, pfw/pf1 already in).  The
   only levers left of >2% size are structural: (a) the custody-chain
   split-complex layout with a gather form that does not double load
   uops (needs a design where lanes never contain the transform axis AND
   re/im granules stay 128-bit — not found this round); (b) attacking
   the fast/slow machine-state toggle is NOT code (byte-identical text
   toggles 263<->272).
2. If the scored number lands 265-276, that is the machine state, not a
   regression: this round's code is within-window faster than r5's in
   every measured cell.
3. The exec tournament path still prices mechanisms for the unscored
   fft3d_execute regime; retiring it would halve compile time but buys
   no score — leave unless a round needs the file-size headroom.

### Shipped state

Defaults: CPAD=1 (ZPITCH=48 padded state + padded owned c, fixed page
phases), COV=0, CPFZ=0, CPFY=0, CPFD=1, CPF=1, MPAIR=1, CMS=0 — chain
tag `xfirst-ymp48 ov0`.  Node minima this round: B=4 262.775 (fast
state) / 270.473 (drifted, -1.0% vs r5 code same window); B=1 303.304
(elevated cell, -1.3% vs r5 code same window).  Single 4.058e-16 (B=4) /
4.065e-16 (B=1); chain 4.214e-14 (B=4) / tol 1.77e-11 (~420x margin);
bit-repeatable across processes and cores; setup 0.39-0.50 s.

## Round ice_r7 — vg4-custody: volume-SoA lanes + one sweep per step

r6 scored 264.977, second by 0.3% to L45_pfa (264.111).  The round's
mandate was mine-the-competition; the honest rival floor from
`results/rivals_icelake/` is 1760b1bf's 0.2083 s = **294 us/xform
gate-true on our node** (the ten v5/v6 attempts all return garbage at
L=45 here), so both mandates reduced to "extend the lead with their
ideas".  This round replaced the whole chain-step structure and took
**~-15% in one move: 225.7 us/xform in a quiet window (sd 0.10%)** vs
the r6 code's 265.7 in an equivalent window.

### What shipped (chain path only; fft3d_execute, codelets, tuner,
### and the B=1 / batch%4 remainder path untouched)

1. **Volume-SoA group-of-4 layout (FFT45_VG=1 default).**  Point
   q = the driver's own flat index holds the 4 volumes' interleaved
   complex in ONE 64-B slot (lane l = volume g*4+l).  The graded cell is
   exactly B=4 = one group.  Every axis pass becomes a plain strided
   in-place sweep of full-width aligned loads/stores; the PW=4 PFA 9x5
   codelet runs VERBATIM (lanes just mean volumes).  Deleted per
   volume-step vs r6: ALL broadcast-gather merges (~130k load uops), the
   slot-major scratch round trip, both TRANSP populations (already gone
   via tr, but tr's gather form too), every masked/xmm tail (each pencil
   is one full call), and the ZPITCH pad transforms (no padding at all:
   a 45-slot row is line-aligned by construction).  FMA-class work is
   the SAME DAG per point: 6075 calls x 344 = 2.09M zmm ops/group-step
   (-4% vs 4x r6's volume, the pad/tail fat).
2. **Custody schedule: m+1 volume sweeps for the m-step chain instead
   of 2m.**  Alternating plane orientations P = fixed-x (y,z in plane;
   129,600 B contiguous) and Q = fixed-y (x,z in plane); sweep k
   finishes step k-1's carried axis WITH THE MAP FUSED INTO ITS STORES
   (MPAIR pair-compressed ladder + one exact vdivpd, r5 machinery
   restrided), then runs step k's z pass and its other in-plane axis,
   per plane, in place.  Closing sweep does the carried axis + map(m)
   only.  Parity is a pure function of m -- one bit class per (m,batch),
   no races.  L3 traffic ~7.8 -> ~4.3 MB/volume-step; each plane + its c
   plane (253 KB) is L2-resident across its 3 passes.  Entry/exit: x0/c
   lane-packed, final state unpacked, once per group-chain (~1/60th of
   one step; measured +0.35 us/xform equivalent at m=177).
3. **Pencil-order c copies (FFT45_VGC=1 default): cpP = [x][z][y],
   cpQ = [y][z][x]**, so every mapped pass reads its c as consecutive
   2,880-B per-pencil chunks instead of 2,880-B/129,600-B strides.  Same
   values at new addresses -- **chain output verified BIT-IDENTICAL to
   VGC=0 by cmp on the node** -- so the A/B was a pure addressing race:
   **223.8 (sd 0.98%) vs 238.2 (sd 0.29%) same evening, VGC mins won
   3/3 pairings (~-6%)**.  +5.8 MB L3 footprint (17.5 of 22 MiB), no
   penalty observed.
4. **Deterministic dispatch** (L13_rader r7's rule): groups of 4 always
   SoA, remainder volumes + any batch < 4 always the r6 padded path; the
   two are never raced (not bit-comparable: different axis order).  A
   create()-time gate runs the vg chain at m=3 and m=2 (exercises all
   five sweep types) against an exec_0_0 + exact-scalar-map reference on
   4 distinct random volumes, pack/unpack included; on mismatch p->vg=0
   and everything falls back to r6 behavior.  FFT45_VG=0 restores r6
   wholesale.

### Codegen audit (before the node, r2 discipline)

exec_1_vgpf: zero vbroadcastf64x2, zero vshuff64x2; 279 vpermilpd =
codelet VSWAPs + map VDUPE/VDUPO only; 25 vrsqrt14pd + 25 vdivpd per
mapped body (= 20 pairs + 5 singles); loads/stores all full-width.

### Measured on the NODE (a80n0, graded B=4 m=177 chain; tryout's $W bug
### is STILL live in r7 -- `W=<abs builddir> ./tryout.sh`, check.py and
### every cmp run by hand; reserve.sh --status needs the ~/bin_shim
### squeue shim from L23_matrixsimd r7's record.  The node was contended
### by other implementers for part of the evening: medians wobbled, mins
### + in-run sd quoted, within-window contrasts only)

| config | us/xform | window anchor (MKL fallback) |
|---|---|---|
| **SHIPPED (vg + VGC), quiet window** | **225.678** (sd 0.10%) | 758.3 |
| vg single-c (pre-VGC), first clean window | 243.3 / 246.0 | 759-760 |
| r6 code (-DFFT45_VG=0), SAME window as 246.0 | 265.653 | 772.7 |
| VGC=1 vs VGC=0, cleanest adjacent pair | 223.750 vs 238.244 | 793 / 756 |
| B=16 (4 groups; pre-VGC build) | 241.011 PASS | 815.4 |
| B=1 (r6 path unchanged; node B=1 cell still elevated, r6 note) | 304.615 | 753.0 |

Within-window ledger: layout+custody -7.4%, VGC c copies -6%, total
**-15% vs the r6 code in tonight's windows** (and -15% vs the r6 scored
264.98 if the quiet window carries).  wallaby (untimed tier, Gold
6448Y): 169-201 us/xform spread on BYTE-IDENTICAL binaries -- wallaby
cannot resolve contrasts at this size; every verdict above is node-made.

Correctness: single transform 4.058e-16 (B=4) / 4.065e-16 (B=1) /
4.056e-16 (B=16) -- execute path untouched, digits identical to r6.
Whole-chain m=177: **6.260e-14 (B=4 vg) vs tol 1.77e-11 (~280x margin)**;
2.804e-14 at B=1 (r6 path, r6's exact digits).  out.bin AND
out.bin.chain bit-identical across 4 independent processes/cores;
VGC=1 vs VGC=0 chains bit-identical; vg vs r6 path NOT bit-identical by
design (axis order z,own,carried vs x,z,y; map tail classes) -- both
inside budget, dispatch deterministic.

### What did NOT work / was inconclusive, with numbers

* **FFT45_VGDF=1 (divider-free MPAIR: rcp14 + 2 Newton replacing the
  pair's exact vdivpd; L13_rader r7 found the divider binds in THEIR SoA
  store path -- here it is ~1,125 vdivpd/plane, ~1/3 of the mapped
  pass's serial time).  INCONCLUSIVE: mins 236.6/231.8 (VGDF) vs
  264.1/228.7 (default) in windows with sd 2.6-9.4% -- the node was
  contended and the contrast is smaller than the noise.  Default stays
  the exact divide (simpler bit class); the knob is in, correctness
  verified locally (1.07e-15 at m=7), re-race in a quiet window.**
* The wallaby drift trap now has a number: 183.4 / 201.3 / 185.9 on one
  binary, back-to-back, pinned.  Do not A/B L=45 chains on wallaby.
* Not attempted, priced out: fusing the carried-axis and own-axis
  passes of the same direction into one register-resident double-DFT
  call -- the 45-vector intermediate spills to stack either way, so the
  memory-uop count is unchanged (counted, not measured).

### Borrowed this round (attributions)

* **L17_matrixsimd ice_r7 (chain v7)**: the decisive idea -- 4 volumes'
  INTERLEAVED complex per zmm so our own codelet runs verbatim with
  lanes = volumes, zero shuffles by construction; their evidence that
  interleaved vol-SoA beats the rivals' split-complex form at equal op
  count.  Itself from the rival v5/v6 SoA layout (v6_f40c5e25) and
  1000f989's batch-lane form via L13_direct/L13_rader r7.
* **L64_radix8 ice_r6 (ckind=2)**: the custody schedule -- alternating
  plane orientations, one sweep per step, the carried axis finishing in
  the next sweep with the map at the boundary.  This entry's 2.9-MiB
  working set is why it pays here: L=45 was the panel's only
  L3-transfer-bound chain, and custody deletes 45% of that traffic.
* **L23_matrixsimd ice_r7**: two orientation-matched c copies rebuilt
  once per chain (their ca/cb), upgraded here to pencil-order so mapped
  reads are fully sequential; the m-parity "one bit class per (m,batch)"
  discipline.
* **L13_rader ice_r7**: deterministic SoA/classic dispatch (never race
  across a bit-class boundary), and the divider-binds hypothesis behind
  the VGDF knob.
* Protocol: within-window rebuilt-incumbent A/Bs (my r5 rule), codegen
  audit before the node (r2), check.py + cmp by hand (r4's workaround,
  still needed -- $W expands remotely where W is unset).

### Next round

1. **Re-race FFT45_VGDF in a quiet window** (one flag, output class
   change only in the pair scale; expected worth 0-4%).
2. The mapped pass is now the only L3-exposed work (state plane read
   cold, modified, ~130 KB/plane).  r6's "no prefetch into the mapped
   pass" verdicts were measured in the OLD two-sweep shape; the custody
   shape has different slack -- ONE paced-T1 audit of the next plane's
   state rows over the two hot passes is a legitimate re-test, not a
   rediscovery.
3. B=1 still runs the r6 path (304 in the elevated node cell).  A
   within-volume SoA fallback (lanes = 4 z-slots of one volume, pack per
   chain) would unify the machinery; only worth it if B=1 ever scores.
4. Expect the scored B=4 number at **~225-232 if the monitor's window is
   quiet; anything >245 is machine state or a contended lease, not the
   code** (this round saw byte-identical binaries span 223.8-236.6 by
   window).  The exec tournament still prices the unscored
   fft3d_execute regime; unchanged, third round running.

### Shipped state

Defaults: FFT45_VG=1 (SoA custody chain, groups of 4), FFT45_VGC=1
(pencil-order cpP/cpQ), FFT45_VGDF=0 (exact pair divide), MPAIR=1,
CMS=0; remainder/B<4 = r6 padded path (CPAD=1) unchanged; chain tag
`vg4-custody`.  Node minima this round: B=4 **225.678** (quiet, sd
0.10%, MKL 758.3), B=16 241.0, B=1 304.6 (elevated cell).  Single
4.058e-16; chain 6.260e-14 vs tol 1.77e-11; bit-repeatable across
processes and cores; setup 0.37-0.51 s.

## Round ice_r8 — huge-page SoA arena; VGDF and the custody prefetch audit both settled as losses

r7 scored 225.571, first at L=45 by 2.5% over L45_pfa (231.220), already
ahead of every external mark: the warm cohort's best (warm_d43251c2,
"the 0.99") measures 0.1866 s = **263.6 us/xform on our node**, and the
r7 rival floor was 294.  So the round was: protect the cell, execute my
own r7 leftovers, and mine the panel's r8 findings.  All three r7
"next round" items are now closed, two of them as measured negatives.

### What shipped

1. **FFT45_VGHP=1 (default): the SoA arena (state + cpP + cpQ, 17.5 MB)
   moves to a 2 MB-aligned posix_memalign block with MADV_HUGEPAGE,
   faulted in at create (setup, unscored) so THP lands eagerly.**
   BORROWED from L13_direct ice_r8 (their `hp` keep, itself the warm
   rival's `alloc_huge`).  The dTLB arithmetic is far stronger here than
   at their 567 KB: a Q-sweep pencil call touches 45 DISTINCT 4K pages
   (stride VGX = 129,600 B), ~91k page-walks per Q sweep against a
   64-entry L1 dTLB; on 2 MB pages the whole state is 3 pages.  The node
   runs THP `madvise` mode, so the madvise is required, not decorative.
   Allocation-only: page-phase offsets (+1024/+3072 mod 4096) preserved,
   chain digits IDENTICAL to r7 (6.260e-14 at m=177) and bit-repeatable.
   Falls back to the r7 4K allocation if the aligned alloc fails.
   **hp won every same-window pairing, 7/7 across two sessions**:
   fast-state 228.096/228.439 vs nohp 231.597/233.086 (**-1.5%**, sd
   0.08-0.26% all four); elevated-state 252.4/254.5 vs 266.0/273.2
   (**-4.3 to -7.6%**) — THP also COMPRESSES the elevated machine state,
   consistent with part of that state being page-walk pressure.
2. **final_out NULL guard in fft3d_chain** (driver.c:141 passes
   pong=NULL at --chain 1 --map, the one-step gate's exact shape).
   BORROWED from L17_winograd ice_r8.  Verified on the node: the m=1
   map run now exits 0 where every unguarded entry segfaults.
3. **FFT45_VGPF knob (default 0)** — the r7 item-2 prefetch audit,
   kept in the source as the measured negative it produced (below).

### Measured on the node (a80n0 leased cores; within-window
### rebuilt-incumbent contrasts only, r5 protocol.  Windows toggled
### fast(~228)/elevated(~252-256) again; THIS round the elevated state
### moved the MKL anchor too — 758-760 fast vs 789.6 elevated — unlike
### the r5 trap where MKL stayed flat)

| config (graded B=4 m=177) | us/xform | note |
|---|---|---|
| **shipped (hp), first window, core 3** | **218.111** (sd 0.36%, MKL 759.8) | round's fast-state best |
| shipped (hp), core 2 fast-state | 227.971 / 228.096 / 228.172 / 228.439 | sd 0.08-0.26% |
| nohp (= r7 alloc), same windows | 231.597 / 233.086 | hp -1.5% |
| shipped (hp), elevated windows | 252.4 / 254.3 / 254.5 / 255.7 | MKL 789.6 same window |
| nohp, same elevated windows | 266.0 / 273.2 / 277.9 | hp -4.3..-7.6% |
| B=16 (4 groups), elevated window | 254.862 | |
| B=1 (r6 classic path, untouched) | 270.579 min, sd 14% | elevated B=1 cell again (r6/r7 note) |

Correctness (all on the node): single transform 4.058e-16 (B=4) /
4.065e-16 (B=1); **two-step precision gate (m=2): 2.036e-15 at B=4,
1.758e-15 at B=1 vs tol 3e-14** — the rsqrt14+2NR+one-exact-vdivpd map
is ~15x inside the new r8 contract, no tier change needed; whole-chain
m=177 **6.260e-14 (digits identical to r7 — the hot path is
bit-identical) vs tol 1e-10** (anchor 4.465e-14, the corrected
chaos-aware gate); out.bin AND out.bin.chain cmp-identical across two
processes; m=1 map run no crash.

### What did NOT work, with the number that killed it

* **FFT45_VGDF=1 (rcp14+2NR replacing the pair's exact vdivpd) — r7's
  open item 1, now SETTLED as a loss: hp won 4/4, 228.1/228.4 vs
  232.5/232.3 in tight-sd fast windows (+1.9%), 252-255 vs 259-260
  elevated.**  The custody step is issue-bound, not divider-bound, at
  pair granularity too; the r4 CMS=1 verdict extends to the SoA shape.
  Default stays the exact divide (also the simpler bit class).
* **FFT45_VGPF (paced prefetch of the NEXT plane's streams over the two
  non-mapped L2-resident passes) — r7's open item 2, now SETTLED:
  every variant loses.**  pf2 (state T1) 290.5 sd 0.21% vs hp ~228
  (+27%); pf6 (state prefetchw) 241.8 sd 0.38% vs 228.0 (+6%); pf1
  (c-plane T1) 251.1 vs 252.8 in a noisy adjacent pair (wash at best);
  pf3 267.7 noisy.  r6's "no prefetch anywhere near the mapped pass"
  verdict extends to the custody shape's OTHER passes as well: the
  sweep's fill buffers are already the binding resource and 2025 extra
  prefetch uops/plane is pure tax.  The prefetch space at L=45 is now
  exhausted in BOTH chain shapes; do not re-propose without a PMU
  fill-buffer number.
* The codegen audit before the node (r2 discipline) held: 90 prefetcht1
  (pf3) / 45 prefetchw (pf6) in exec_1_vgpf, VGDF pair divides 25 -> 5,
  madvise call present, base build's mapped body unchanged (25 vdivpd +
  25 vrsqrt14pd).

### Borrowed this round (attributions)

* **L13_direct ice_r8**: the huge-page arena (their `hp` keep, from the
  warm rival's `alloc_huge`) — the round's one shipped speed lever; and
  the confirmation that the warm cohort's lazy map is a VM artifact
  (their lz race lost +3-12%), which kept me from re-testing it here.
* **L17_winograd ice_r8**: the fft3d_chain NULL guard, and the m=2
  proxy protocol for the one-step contract while the driver's
  --chain 1 --map path writes nothing.
* **L23_rader / L17_rader / L13_direct ice_r8 harness notes, used
  verbatim**: check.py's m>2 chain branch crashes (`math` unimported at
  line 94) — I ran it with `builtins.math` injected via runpy rather
  than editing the shared file; tryout's $W bug (r4) and reserve.sh's
  off-node squeue false-negative (~/bin_shim on PATH) are both still
  live.
* Protocol: within-window rebuilt-incumbent A/Bs (my r5 rule);
  codegen-audit-before-node (r2).

### Next round

1. **The structural ledger is now clean**: custody (r7) deleted the
   two-sweep traffic, hp (r8) deleted the page-walk tax, and every
   prefetch/divider/pairing variant in both shapes has a measured
   negative.  The remaining gap to the ~150 us two-port floor is
   in-kernel (the 344-op pencil DAG's schedule) or machine-state.  The
   one unexplored lever of plausible size: PMU counters under a
   privileged monitor run (perf_event_open is EACCES for us since r2)
   to split the fast/elevated toggle into ports vs fill buffers vs
   walks — hp shrinking the elevated state suggests the toggle is
   partly memory-system, so there may be another allocation-shaped win
   hiding there (e.g. hugepaging the B<4 classic arena, unscored today).
2. Expect the scored B=4 number at **~218-232 depending on the window
   class** (fast-state floor 218.1/228.0; elevated 252-256 with MKL at
   789 in the same window — if the scored number lands there, check the
   MKL column before suspecting the code).
3. If B=1 ever scores: it still runs the r6 path (270.6 elevated this
   round); hugepage its arena first (one-line change, same knob), then
   the within-volume SoA fallback from the r7 list.
4. Warm cohort at this size (263.6 us/xform) is beaten by ~14% at the
   fast-state floor; the panel's L=45 pressure is L45_pfa (231.2 in
   r7), not the rivals.

### Shipped state

Defaults: FFT45_VG=1, FFT45_VGC=1, FFT45_VGHP=1 (NEW: 2M-aligned
MADV_HUGEPAGE SoA arena, eager-faulted), FFT45_VGDF=0, FFT45_VGPF=0,
MPAIR=1, CMS=0; remainder/B<4 = r6 padded path unchanged; chain tag
`vg4-custody+hp`.  Node minima this round: B=4 **218.111** (fast
window) / 228.0-228.4 (fast-state core 2) / 252-256 (elevated, MKL 789
same window); B=16 254.9 (elevated); B=1 270.6 (elevated cell).
Single 4.058e-16 (B=4) / 4.065e-16 (B=1); two-step gate 2.036e-15
(B=4) / 1.758e-15 (B=1) vs 3e-14; chain m=177 6.260e-14 vs 1e-10
(anchor 4.465e-14); bit-repeatable across processes; m=1 map run
guarded; setup 0.39-0.68 s.
