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
