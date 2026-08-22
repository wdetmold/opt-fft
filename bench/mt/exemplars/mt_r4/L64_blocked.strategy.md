# L64_blocked — multicore strategy record

Phase-1 history (rounds panel_r6..r11, single-thread) is in
`../../geom/strategies/L64_blocked.md`.  This file starts at the multicore
phase.  The kernel underneath is phase-1's final form: st=3 split-complex
fused 2-sweep (pass 1 = y-FFT per x-plane into the odd-line-padded split
scratch SC; fused pass 2+3 = in-place x-FFT per ky slab + z-lines straight to
out), one bit class across every tuner row.

## Round mt_r1

### What changed

An OpenMP layer over the UNCHANGED st=3 kernel — no arithmetic, no kernel
loop, and no padding was touched.  The mt layer only reassigns whole loop
iterations (planes, ky slabs, volumes), so **every engine at every thread
count is bit-identical to the single-thread kernel** and the r11
one-bit-class invariant survives the phase by construction (verified: PASS
fingerprints at B=1/8/128 are exactly the phase-1 values 4.462/4.460e-16).

Three engines, raced at create time per {B, host}:

* **eng=1 "S" (slab-split)** — all nth threads on ONE volume at a time:
  `omp for` over the 64 pass-1 x-planes, implicit barrier, `omp for nowait`
  over the 64 ky slabs of the fused pass 2+3.  Both unit types are whole
  odd-line-count blocks (SCXS/SCKS strides, 1-KB out rows), so no false
  sharing exists anywhere.  At nvol>1 the SC is double-buffered (p->S and
  pool slot 0 by volume parity), which is what makes the pass-2+3 `nowait`
  legal: a straggling reader of volume b is fenced from pass1(b+2)'s
  overwrite by the b+1 barrier it must pass first.  One barrier per volume.
* **eng=2 "G" (grouped)** — G groups of gsz=32/G ADJACENT threads (every
  raced gsz divides 16, so bind=close never splits a group across the node's
  two 16-core sockets).  Group g owns volumes b = g, g+G, ... and two
  hugepage slots (double-buffered by parity), with ONE per-group epoch spin
  barrier per volume between pass 1 and pass 2+3; the pass23→next-pass1
  hazard is ordered by the following volume's barrier, so fast threads flow
  into the next cold in-read while stragglers finish z-lines.  Groups never
  synchronize with each other.
* **eng=3 "V" (volume-parallel)** — one volume per thread, per-thread slot,
  zero synchronization, `schedule(static,1)` vs `dynamic,1` raced.
* **eng=0** — the single-thread kernel, kept as anchor rows (it lost every
  cell, as it must; the anchor proves it rather than assumes it).

Infrastructure: a pool of 32 scratch slots, 3 × 2-MB hugepages each
(SLOTDBL = 786432 doubles ≥ SC and ≥ mid+OB), mmap+MADV_HUGEPAGE,
**first-touched by the owning thread inside fft3d_create** so eng=2/3
scratch is NUMA-local; the shared p->S is first-touched distributed in
eng=1's static plane order.  The create-time parallel touch also warms the
GOMP pool — execute creates no threads.  Tuner rebuilt around
`struct mcand {eng,G,nth,dyn,mode,pf,st,pw}`: arena widened to
nv=min(B,32) volumes (so batched engines tune with every thread loaded),
reps raised (parallel runs are tens of µs), correctness interlock unchanged
(every row is L2-checked against the independent single-thread st=0 kernel —
it now guards the whole mt layer: barriers, slot aliasing, decomposition).
New env axes for the monitor: `FFT64B_ENG/G/NTH/DYN` on top of the phase-1
`FFT64B_MODE/PF/ST/PW/VERBOSE/PRO`.

### Operation count

Unchanged from phase 1: ~1.6M FMA-port vector ops + 328K shuffles per
volume; the mt layer adds zero arithmetic.  Per volume the mandatory DRAM
traffic is in+out = 8.4 MB (NT stores, no RFO); SC (4.46 MB/volume-in-
flight) is the L3-residency lever — G groups keep G×4.46 MB live, which is
why G=4–8 fits wallaby's 60-MB L3 and eng=3's 32×4.46 = 143 MB does not.

### Measured (wallaby, Xeon 6448Y, 32 threads = one 32-core socket)

Wallaby has multi-hour placement windows (phase-1 record, sixth instance):
this round's early window gave B=1 61.5 µs, the later window 68.0 µs for the
SAME pick; cross-window deltas below 10% are not evidence.  Driver-level
minima, per transform:

| B | time/vol | pick | vs phase-1 1-thread (wallaby) | par. eff. |
|---|---|---|---|---|
| 1   | **61.5–68.0 µs** | eng=1 S nth=32 nt pf0 | 522.9 µs → 7.7–8.5× | 24–27% |
| 8   | **41.4–44.0 µs** (331–352/call) | eng=2 G=8 nt pf0 | 568.0 µs → 13.7× | 43% |
| 128 | **52.6 µs** (6735/call, sd ~10% window) | eng=2 G=8 nt pf0 | 683.3 µs (B=64) → 13.0× | 41% |

In-arena orderings (same window, usable within-table): B=8: G=8 nt 42.1,
G=4 nt 44.5, G=2 nt 46.6, S nt 54.8, cached versions +2–8; B=128: G=8 nt
50.5, G=4 nt 50.9, G=16 nt 55.0, G=2 66.1, S nt 70.4, **V nt 87.6, V cached
110** — the eng=3 "obvious" batch axis loses by 1.7× to grouped because 32
concurrent SCs spill L3 (the create-comment analysis, confirmed by
measurement).  B=1: S-nt 69.5–71.4, S nth=16 107.7, nth=8 145.2, nth=4
236.5 — scaling continues to 32 threads, so sync does NOT cap B=1 here
(unlike the small-L entries); NT beats cached by ~25% at every engine.

Batched bandwidth accounting: 8.4 MB/vol ÷ 41.4 µs = 203 GB/s ≈ 2/3 of the
socket's DDR5 peak — the batched cells are at the memory wall, which is why
G=8-vs-G=4 and pf differences are ~1-µs effects.  B=1 accounting: ideal
split 523/32 ≈ 16 µs; measured 62–68.  The gap is the all-to-all (pass 2+3
reads the whole 4.46-MB SC written by all 32 cores — the transpose cannot be
deleted, only hidden; L17_matrixsimd mt_r1 reached the same conclusion at
their size) plus ~2× omp region entry+barrier.  B=1 parallelises to ~8×,
recorded as a measurement with the decomposition raced down to nth=4.

### What was tried and did NOT work

* **eng=2 with G=1** (my 32-wide central spin barrier replacing GOMP's tree
  barrier for the S split): 89.6 vs 69.5 µs at B=1-nt — the central atomic
  is WORSE at 32 waiters; rows deleted from the pool.  A tree barrier might
  close it, but GOMP already has one.
* **eng=3 V at B=128**: 87.6 nt / 110 cached vs G=8's 50.5 — L3 spill, see
  above.  Kept in the pool because the node's L3 is 22 MB/socket and its
  DRAM slower: the balance point differs, the node decides.
* **S-dyn (schedule(dynamic))**: 63.6 vs 54.8 at B=8-nt, 83.5 vs 71.4 at
  B=1-nt — 64 lock-xadds per loop cost more than the straggler absorption
  buys on an idle one-socket machine.  Kept raced for the node's cross-
  socket jitter.
* **pf=6 slabpf at B=1/S**: 76.3 vs 69.5 (nt) — cross-core dirty-line
  prefetch doesn't pay; pf=1 p1pf on eng=2 is +0.1–1.5 µs (within noise
  here, kept raced for the node where prefetch history differs).
* **nowait A/B at B=1**: no effect (68.0 nowait vs 68.5 barrier, same
  window) — the apparent 61.5→68 "regression" mid-round was a window flip,
  reconfirming the standing rule: only same-window comparisons are usable.

### Borrowed from other entries

* **L23_matrixsimd mt_r1**: the decomposition-by-batch frame
  (volume-parallel at large B / slab-split inside a volume at small B), the
  node facts I planned against (Gold 5218 = 2×16 cores so 32 threads span
  BOTH sockets; the driver freads `in` and touches `out` on thread 0 so all
  caller pages sit on socket 0), and the GOMP-overhead warning (their
  measured 6.2–8.2 µs fork+barrier+join) — my volumes are ~500 µs so I kept
  GOMP instead of their pthread pool, and the measurement above confirms
  that was right at this size (B=1 still scales to 32 threads).
* **L17_matrixsimd mt_r1**: the honest B=1 accounting style (ideal-split
  floor + barrier + all-to-all coherence term) and the evidence that
  xrange-style probes mismeasure cross-core dirty-line traffic — I raced
  engines on the full job only.
* Phase-1 carriers unchanged: hugepage scratch + odd-line padding
  (L64_radix8 r6 lineage), NT-vs-cached 2% bar, one-bit-class pool
  discipline, self-warming interleaved tuner (L36_pencilfused r5).

### Node predictions (Gold 5218, 2×16 cores, 22-MB L3/socket, to be scored)

* **B=1**: eng=1 S nth=32 spans both sockets — the barrier and the SC
  all-to-all cross UPI.  Expect the pick at S nth=16 (one socket) or nth=32,
  60–140 µs.  If nth=16 wins, half the machine is idle at B=1 and the
  honest next step is socket-local pipelining, not more prefetch.
* **B=8**: G=8 → gsz=4, 4 groups/socket, socket-1 groups read in/out over
  UPI.  Expect G=8 or G=4, cached to be closer to NT than on wallaby (the
  node has never loved NT at L=64), 60–110 µs/vol.
* **B=128**: G=8/G=16 nt or cached; eng=3 V could win if 22-MB L3 makes
  grouped SC-sharing moot and DRAM is the only wall.  Watch whether
  socket-1's UPI-remote in/out shows as a ~2× spread between the fastest
  and slowest group — that is the signal for next round's NUMA staging.
* Monitor asks, in cost order: (1) `FFT64B_VERBOSE=1` once per B — the full
  engine table answers every axis; (2) if B=1 picks nth=16, one forced
  `FFT64B_NTH=32` run quantifies the UPI barrier tax; (3) standing controls
  unchanged (`FFT64B_ST=0/1/2` single-thread fingerprints, FFT64B_NOHP).

### Next

1. **Node-driven**: read the picks.  The three engines bracket the design
   space; the interesting node question is UPI (barrier cost and remote
   in/out), which wallaby cannot exhibit at 32 threads.
2. **Dynamic volume scheduling for eng=2** (an atomic next-volume counter
   with a race-free per-group handoff — the naive handoff has a leader-
   overwrite race, worked out but not shipped this round): matters only if
   the node shows socket-imbalanced groups at B=128.
3. **B=1 ceiling**: the remaining ~45 µs over the ideal split is coherence-
   bound.  Candidate: assign each pass-2+3 thread the ky range whose slab
   lines its own pass-1 planes contributed most to — impossible exactly (the
   transpose is all-to-all) but a 2-phase hierarchical variant (socket-local
   partial transpose into a per-socket staging buffer, then cross-socket
   merge) is the only structural idea left; cost it against the perf-stat
   ask first.
4. Standing dead list (mt): 32-wide central spin barrier, eng=3 at B≥128 on
   one socket (L3 spill), schedule(dynamic) on 64-unit loops at idle
   machines, cross-window wallaby comparisons (reconfirmed this round).

## Round mt_r2

### Where the node left me (mt_r1 scored)

B=1 **WON**: 128.0 µs vs L64_radix8 135.7, mkl 153.2 — pick `eng=1 S nth=16
cached pf0`, i.e. one socket, as predicted (the UPI barrier + all-to-all tax
exceeds 16 more cores' worth of compute).  B=128 **WON**: 95.7 µs vs rival
146.9, fftw3_patient 184.4 — but run spread 28.5%.  B=8 **LOST**: 111.7 vs
mkl 73.7 (1.52×), rival 91.5.  The B=8 accounting that drives this round:
the driver first-touches `in` and `out` on socket 0, so ALL of a call's
compulsory traffic (8.4 MB/vol × 8 = 67 MB) crosses socket 0's memory
controllers NO MATTER which cores compute — socket-1 groups add UPI latency
without adding a byte of bandwidth, and mt_r1's static b=g,g+G map convoys
the fast socket behind the slow one.  mkl's 73.7 µs/vol = 114 GB/s through
one socket's DDR4 controllers is near the practical wall; catching it means
eliminating everything that is not compulsory traffic.

### What changed (schedule only; kernel, arithmetic, padding untouched —
### every row is still the one bit class, verified below)

Three new eng=2 axes, all raced by the create-time tuner:

* **dyn=1 — dynamic volume pull** (mt_r1 "Next" item 2, now shipped).  One
  atomic next-volume counter shared by all groups; the group LEADER
  fetch-adds the group's next volume BEFORE arriving at the volume's spin
  barrier and publishes it in a per-group, round-parity mailbox pair;
  members read the slot right after the barrier exit, so the handoff rides
  the existing sync — zero extra barriers.  The mt_r1 leader-overwrite race
  is closed by the two parity slots: the leader runs at most one barrier
  ahead, so its round-(r+2) overwrite of slot rp is separated from any
  straggler's round-r read by the round-(r+1) barrier (read sequenced before
  the straggler's arrival RMW; write after the leader's exit, which
  acquire-synchronises with every arrival through the count RMW chain).  On
  the node this lets socket-0 groups absorb the volumes that UPI-throttled
  socket-1 groups would convoy on — the B=128-spread and B=8-imbalance fix.
* **sb=1 — single-buffer SC**: one SC per group instead of the volume-parity
  double buffer, at the price of a second spin barrier per volume (the
  pass23 → next pass1 reuse hazard needs it).  Halves live scratch: at
  B=128/G=8 that is 8×4.46 = 36 MB live instead of 71 MB — the difference
  between fitting and thrashing wallaby's 60-MB L3, and on the node the only
  way 4 groups/socket fit a 22-MB L3.
* **nth=16 — socket-0-only grouped rows at batch** (bind=close puts threads
  0..15 on the node's socket 0): 16 local threads can saturate the socket-0
  controllers that carry all the traffic anyway, without paying UPI at all.
  Plus an eng=1 S nth=16 row at batch>2 (same idea, volume-serial).  These
  rows MUST lose on wallaby's single socket; they exist for the node.
* pf=6 slabpf joins the batched eng=2 race — L64_radix8's node B=128 pick
  carries slabpf1 under gangs, and their 32-thread B=1 slabpf loss does not
  transfer to gsz≤8 groups prefetching their own next slab.

New env `FFT64B_SB`; `FFT64B_DYN` now also selects the eng=2 pull rows.
Pool grows to ≤~56 rows at B=128 (mc[72]); tuner arena/interlock unchanged.

### Operation count

Unchanged: ~1.6M FMA-port vector ops + 328K shuffles per volume; dyn adds
one relaxed fetch_add + two relaxed mailbox ops per group-volume, sb adds
one spin barrier per volume.  Zero added FP work, zero added traffic; sb
*removes* 4.46 MB of live scratch per group.

### Measured (wallaby, Xeon 6448Y, 32 threads = one socket; one window,
### same-window comparisons only)

Driver-level minima: **B=1 67.1 µs** (window band; mt_r1 saw 61.5–68.0 for
the same pick), **B=8 372.0/8 = 46.5 µs/vol** (pick eng2 G=8 static cached
pf0; hysteresis took cached over nt pf1 42.2 within the bars), **B=128
7013.5/128 = 54.8 µs/vol (pick eng2 G=8 nt sb=1)**.  PASS rel_l2 =
4.460–4.464e-16 at B=1/2/8/128, repeatable (bit-identical) everywhere.

* **sb=1 is the round's measured win — same-window, in-arena at B=128:
  G=8 nt sb1 48.3 vs sb0 58.2 (−17%); G=16 nt sb1 52.9 vs sb0 84.4 (−37%);
  cached G=8 sb1 69.1 vs sb0 83.6.**  The L3-residency model is confirmed:
  halving live SC beats double-buffering's barrier avoidance as soon as the
  double buffer spills L3.  At B=8 with G=8 each group runs one volume so
  sb is a no-op there (rows correctly not generated).
* pf=6 slabpf under gangs, B=128: G=8 dyn sb1 nt 49.3 vs pf0 51.7; G=16
  52.4 vs 53.5; cached 69.4 vs 70.3 — a consistent ~1–4% and never a loss,
  vindicating the rival's node pick; raced, the node decides.
* dyn on wallaby: no benefit, as expected with nothing to balance on one
  socket — G=8 sb1 nt dyn 51.7 vs static 48.3 at B=128; B=8 G=2 dyn 48.7
  vs static 48.9.  The ~3 µs/vol dyn cost at B=128/G=8 is the changed
  volume→group locality plus the mailbox, and is exactly what the static
  rows are still in the pool to beat on a symmetric machine.
* nth=16 on wallaby loses everywhere at batch, as it must on one socket:
  B=8 best nth=16 row 54.6 (G=8 nt) vs 42.2 at nth=32; B=128 G=8 nth16 dyn
  nt 75.4.  Forced end-to-end (`FFT64B_ENG=2 FFT64B_NTH=16`): PASS, and
  bit-identical to the dyn+sb forced run (`cmp` exact) — the one-bit-class
  invariant holds through every new axis.
* Parallel efficiency vs phase-1 wallaby single-thread: B=1 522.9/67.1 =
  7.8× (24%); B=8 568.0/46.5 = 12.2× (38%); B=128 683.3/54.8 = 12.5× (39%).

### What did NOT work / negatives with numbers

* dyn at B=128 on wallaby: 51.7 vs 48.3 (see above) — kept for the node,
  where the 28.5% spread and the UPI asymmetry are the case it was built
  for; on a symmetric socket static wins and the tuner will keep saying so.
* nth=16 at batch on wallaby: 54.6 vs 42.2 at B=8 — the expected one-socket
  loss; not evidence about the node.
* Nothing failed correctness at any point this round.

### Borrowed from other entries

* **L64_radix8 mt_r1**: slabpf-under-gangs (their node B=128 pick string
  `gang-g4-nt+slabpf1`) → my pf=6 eng=2 rows; and their measured warning
  that 32-thread B=1 slabpf inverts (135.5 vs 123.4) → pf=6 stays out of
  the B=1/S rows.
* **L23_matrixsimd mt_r1 (via L45_mixedradix's record)**: the socket-0
  pages accounting — driver first-touches in/out on thread 0, so socket-0
  controllers carry everything — which is the entire argument for the
  nth=16 rows and the dyn balance; L45's sketched "NUMA-asymmetric split
  (socket-0 threads own the stores)" is the same idea, here realised as
  raced socket-0-only rows rather than an asymmetric split.
* My own mt_r1 "Next" item 2 (the dynamic handoff design), now shipped with
  the parity-mailbox fix.

### Node predictions (Gold 5218, 2×16, 22-MB L3/socket, DDR4)

* **B=8, the cell that matters**: if the socket-0-controller model is
  right, `eng=2 nth=16 G=2/4` (or S nth=16) lands near 67 MB / ~100 GB/s ≈
  84 µs/vol, beating my 111.7; dyn nth=32 G=4 should land between.  Beating
  mkl's 73.7 needs the controllers saturated with prefetch — watch whether
  nt wins (no RFO).  If nth=32 static still gets picked, the UPI convoy
  model is wrong and the residual is elsewhere (ask for FFT64B_VERBOSE=1).
* **B=128**: dyn sb1 G=8/16 nt should cut the 28.5% spread (pull replaces
  the convoy) and sb fits 4 groups/socket in 22 MB; expect 70–95 µs/vol.
* **B=1**: unchanged rows; expect the S nth=16 cached pick to repeat at
  ~128 µs.
* Monitor asks, in cost order: (1) FFT64B_VERBOSE=1 once at B=8 — the
  engine table answers the socket-0 question outright; (2) if the B=8 pick
  is nth=16, a forced FFT64B_NTH=32 FFT64B_DYN=1 run prices the UPI convoy;
  (3) standing controls unchanged.

### Next

1. Node-driven: read the B=8 pick.  If nth=16 wins there, the honest next
   step is overlapping socket-1 compute with socket-0 streaming (socket-1
   threads doing pass-2+3 x-FFT slab work on SC only, never touching
   in/out) — a producer/consumer split, one round of work.
2. If dyn wins B=128 but the spread stays >10%, the residue is the window/
   governor, not the schedule; stop chasing it.
3. Standing dead list unchanged, plus: dyn on symmetric hosts, nth=16 on
   wallaby (both by measurement this round).

## Round mt_r3

### Where the node left me (mt_r2 scored)

B=1 **WON**, 127.0 µs vs rival 136.2, mkl 153.3 — same pick (`S nth=16
cached pf0`) in all three processes; stable, untouched this round.  B=8
**LOST to mkl by 5%**: 76.7 vs 73.0 µs/vol (a 1.46× improvement over mt_r1;
picks `G=4/G=2 nt nth=32`, so mt_r2's socket-0-only nth=16 theory is DEAD —
the node rejected those rows 3/3).  B=128 **LOST 1.30× to L64_radix8**
(90.5 vs 69.5) with the round's worst pick lottery: 2/3 processes picked
`G=16 dyn=1 sb=1` and read 135.6/128.6 µs/vol against the 1/3 that picked
`G=2 dyn=0 sb=0` and read 90.5.  The verdict's mechanism (§5): the
create-time arena races in a different placement regime than the scored
loop and mis-prices team shapes at streaming cells; its explicit
instruction: **dyn was the losing pick in every process that chose it,
panel-wide — remove it from the tuner surface**.  Meanwhile the rival won
B=128 with `gang-T32-g4-nt+slabpf1` — in my vocabulary **eng=2 G=8 nth=32
STATIC sb=1 nt pf=6**, a row combination my mt_r2 pool never generated
(static sb=1 existed only at pf=0; slabpf-under-groups only under dyn).

### What changed (tuner policy only; kernel, arithmetic, engines, padding
### untouched — every row is still the one bit class, verified below)

1. **dyn rows and nth=16 batched rows leave the default pool** (both
   rejected 3/3 by the node).  They are still *generated* when
   `FFT64B_DYN` / `FFT64B_NTH=16` ask, so every monitor control keeps
   working — forced-dyn output verified bit-identical to the default pick.
2. **Static sb=1 rows at every G < batch, pf ∈ {0,6}** — this adds the
   rival's node-winning B=128 shape to my pool.  Also pf=8 (slabpf+p1pf)
   rows at nt, G ∈ {4,8}, both sb=0 and sb=1: the node's B=8 winner
   carried p1pf and the rival's batch picks carry slabpf; the pair had
   never been raced together here.
3. **Node-prior incumbent with a 5% beat-margin at batch ≥ 32** (where
   nv=min(B,32) < B means the arena cannot even be size-faithful): the
   incumbent is `eng2 G=8 nth=32 static sb=1 nt pf=6` — the shape the node
   itself scored fastest at this geometry (rival's 69.5 µs/vol at 18%
   spread, vs my lottery's 90.5–135.6) — and a challenger ships only if it
   beats it by >5% in-arena.  This is the verdict §6 recipe ("make the
   proven team the incumbent and require a margin") with the incumbent
   taken from the strongest node evidence available.  Env forcing bypasses
   it naturally (a filtered pool lacking the incumbent row skips the
   override).
4. Timing rounds 4 → 6 at nv ≤ 8 (B=8 had 3 picks in 3 node processes;
   more best-of samples per row cost ~0.1 s of unscored setup).
   Candidate arrays widened mc[72] → mc[96].

### Operation count

Unchanged: ~1.6M FMA-port vector ops + 328K shuffles per volume.  Zero new
code in any execute path — this round is entirely candidate-generation and
pick policy inside `fft3d_create`.

### Measured (wallaby, Xeon 6448Y, 32 threads = one socket; one window,
### several runs, quoted as bands)

| B | this round | pre-change baseline (same window) | pick (wallaby) |
|---|---|---|---|
| 1   | **65.5–68.8 µs** | 59.5–68.8 (window band) | S nth=32 nt pf0 (unchanged family) |
| 8   | **338.5–364.5/8 = 42.3–45.6 µs/vol** | 346.2 (43.3) | G=8 cached pf1 (nt within the 2% bar) |
| 128 | **6906–7360/128 = 53.9–57.5 µs/vol** | 7040 (55.0) | **incumbent: G=8 sb=1 nt pf=6** |

PASS rel_l2 = 4.460–4.462e-16 at B=1/8/128, repeatable (bit-identical)
everywhere; forced `FFT64B_DYN=1` at B=8 is bit-identical to the default
pick's output (cmp over all shared volumes); `FFT64B_NTH=16` and
`FFT64B_ST=0` controls run.  In-arena at B=128 the margin rule did exactly
what it was built to do: arena best was G=8 sb1 nt **pf8** 51.6 with
G=4 nt pf1 51.7 and G=8 sb1 nt pf0 51.8 within 0.4%, incumbent pf6 53.5
within 5% → incumbent shipped.  That 3.5% is wallaby's price for the node
prior; on the node the incumbent's shape is the measured 69.5-µs winner,
and the three rows it displaces are exactly the ones the node priced at
90.5–135.6.  Parallel efficiency vs phase-1 wallaby single-thread: B=1
522.9/65.5 = 8.0×; B=8 568.0/42.3 = 13.4× (42%); B=128 683.3/53.9 =
12.7× (40%).

### What was tried and did NOT work

* Nothing failed correctness or regressed on wallaby this round.  The one
  measured cost is the incumbent's pf6 vs the arena's pf8/pf0 best at
  B=128 on wallaby (53.5 vs 51.6 in-arena, ~3.5%) — accepted deliberately;
  the node, not wallaby, is where the 90.5→69.5 µs/vol gap lives.  If the
  node's arena picks pf8 by >5% the margin rule lets it through.

### Borrowed from other entries

* **L64_radix8 mt_r2** (the round's model): the entire B=128 incumbent
  shape — 8 static gangs of 4 adjacent threads, ONE SC per gang (their
  legacy two-barrier gang = my sb=1), NT stores + slabpf — taken from
  their stable node win at my geometry (69.5 µs/vol).  Credit both ways:
  their legacy gang and my eng=2 co-evolved (they adopted my mt_r1
  pipeline as their mt=5; I adopt their node-proven parameter point).
* **mt_r2 VERDICT §5/§6**: the arena-regime-infidelity mechanism, the
  "dyn loses everywhere it is picked" panel-wide instruction (executed:
  rows env-gated out of the default pool), and the
  incumbent-with-margin recipe (executed at batch ≥ 32).

### Node predictions (Gold 5218, 2×16, 22-MB L3/socket, DDR4)

* **B=128**: the incumbent (G=8 static sb=1 nt pf=6) should ship in ≥2/3
  processes — live SC 8×4.46 = 35.7 MB = 17.8 MB/socket, fits 22 MB — and
  land near the rival's 69.5 (my kernel's z-line tail differs from theirs;
  band 66–80 µs/vol).  The lottery rows that cost mt_r2 its 49.8% spread
  cannot recur: dyn and G=16-sb1 are out of the pool, and anything else
  must beat the incumbent by 5%.  If a process still ships something else,
  read its verbose table — that row beat 69.5-equivalent by >5% in-arena
  and is worth knowing about.
* **B=8**: pool is the node's own winners (G=4/G=2, sb0) plus new sb=1 and
  pf=8 variants; expect G=4 nt pf1 or pf8 at 70–78 µs/vol.  Beating mkl's
  73.0 needs either slabpf+p1pf to stack (~2–4% each on the rival's node
  evidence) or the sb=1 L3-residency effect (live SC halves to 8.9 MB at
  G=4, leaving ~13 MB/socket of L3 for `in` re-reads across the timed
  loop's iterations — at B=8 the 33.5-MB `in` is marginally cacheable and
  every resident line is a socket-0-controller byte saved).  That theory
  is unfalsifiable on wallaby's 60-MB L3; the node decides.
* **B=1**: unchanged rows, expect the S nth=16 cached repeat at ~127 µs.
* Monitor asks, in cost order: (1) `FFT64B_VERBOSE=1` once at B=128 — the
  table shows whether the incumbent shipped and what the margin was;
  (2) if B=8 still loses to mkl, one forced `FFT64B_G=2 FFT64B_SB=1` run
  prices the L3-residency theory directly; (3) standing controls
  unchanged (`FFT64B_ST=0` fingerprint, `FFT64B_DYN=1`, `FFT64B_NTH=16`,
  `FFT64B_NOHP`).

### Next

1. Read whether the incumbent shipped at B=128 and its driver number.  If
   it lands at 66–80, the remaining gap to the rival (if any) is kernel
   tail, not schedule — profile the z-line store path before touching
   anything.
2. B=8: if mkl still leads after the enriched pool, the honest remaining
   move is the structural one both L64 records have deferred: reduce the
   group all-to-all by making pass-1 threads write the ky-slabs their own
   pass-23 range reads (lane-blocked SC), which at gsz=4 turns 3/4 remote
   slab reads into 1/4.  One round of work, kernel-adjacent, keep the bit
   class by permuting only the SC layout, never the arithmetic.
3. Standing dead list (mt): 32-wide central spin barrier, eng=3 at B≥128,
   schedule(dynamic) on 64-unit loops, nth=16 at batched cells (node 3/3),
   dyn in the default pool (node 3/3 + verdict instruction), cross-window
   wallaby comparisons.

## Round mt_r4

### Where the node left me (mt_r3 scored)

B=1 **WON**, 128.7 µs vs rival 136.3, mkl 152.8 (pick `S nth=16 cached pf0`
again; the 127.0 → 128.7 drift is noise).  B=128 **WON**, 73.7 µs/vol vs
rival 142.3, fftw3_patient 178.1 — the mt_r3 incumbent policy repaid itself
in one round (90.5 → 73.7, a 1.23× improvement, 113.8 GB/s), though the cell
still carried a 23.8% spread: 2/3 processes shipped the incumbent (9435/9445
µs/call) and 1/3 shipped a `G=16 sb1 nt pf6` arena upset at 11686 — G=16 has
now shipped three node processes across mt_r2/r3 (135.6, 128.6, 91.3 µs/vol)
and lost to the G=8 shape every time.  B=8 **LOST to mkl**, 77.1 vs 72.4
µs/vol (0.94×, sd 0.2% both sides, "fully real, no statistical escape") —
the panel's only genuine library loss, pick `G=4 nth=32 sb=1 cached pf=6` in
all three processes.  The verdict names my route verbatim: *"the
gang-internal all-to-all where each lane reads 3/4 of its second-pass input
from sibling lanes — is still the route"* — which is my own mt_r3 "Next"
item 2, now executed.

### What changed (schedule/layout of scratch traffic only; arithmetic,
### twiddles, padding, engines untouched — one bit class, verified below)

1. **kb=1 — compact slab buffer, the named all-to-all fix.**  The fused
   pass-2+3 x-FFT now (optionally) writes its result into a per-thread
   68-KB odd-line-padded buffer KB (row = one z-line, stride 17 lines, in
   the tail of the thread's own NUMA-local slot) instead of IN PLACE back
   into the group-shared SC, and the z-line stage reads KB back
   sequentially.  What this buys: **SC becomes read-only in pass 2+3**, so
   the (gsz−1)/gsz of slab lines each lane reads from sibling lanes'
   pass-1 writes need no RFO/ownership transfer and are never re-dirtied;
   SC is swept twice per volume instead of four times (the 4.46-MB
   writeback sweep and its re-read move into a 68-KB L2-resident buffer);
   and the z-line loads walk 1088-B-sequential KB rows instead of
   revisiting the slab at the 69.7-KB SCXS stride.  The stored/loaded
   values are identical, so kb stays inside the one bit class (verified:
   `cmp` bit-identical kb0-vs-kb1 at B=8 and eng1-kb1-vs-eng0-kb0 at B=1).
   kb is NOT a pool row — the pick-lottery surface does not grow.  It is
   decided by a create-time A/B **on the picked candidate** (the pro
   protocol), gated on its own interlock check against the independent
   st=0 reference and a >1% win, in-place on ties; `FFT64B_KB=0|1` forces
   (still correctness-gated).
2. **G=16 leaves the default pool at batch ≥ 32** (three node processes,
   three losses, see above; the arena keeps over-pricing 2-thread groups
   at streaming cells).  Still generated when `FFT64B_G=16` asks.
3. **pf=6 joins eng=2 at sb=0.**  The node's B=8 pick carries slabpf, but
   G=8 at B=8 (one volume per group, so sb rows are correctly not
   generated) had only ever raced pf0/pf1/pf8 — the G=8+slabpf combination
   never existed.  +6 rows at B=8, +6 at B=128.
4. **The sb=1 trailing barrier of a group's last volume is skipped** (no
   next pass-1 to guard; the parallel-region join orders everything; bn is
   group-uniform so all members take the same branch).  1 of 4 barriers
   gone at B=8/G=4.  Adopted from L64_radix8 mt_r3 change 4.

### Operation count

Arithmetic unchanged: ~1.6M FMA-port vector ops + 328K shuffles per volume;
kb adds ZERO FP work and zero instructions in the arithmetic path (same
FFT64S macro, different store target).  Traffic: kb=1 removes one full
4.46-MB SC write sweep + one 4.46-MB strided SC re-read per volume from the
L3/cross-core domain, replacing both with a 68-KB thread-local buffer that
recirculates in L2 (write + read per ky slab, 64×/volume).  Sync: −1 spin
barrier per group per call at sb=1.

### Measured (wallaby, Xeon 6448Y, 32 threads = one socket; one window,
### same-window comparisons only)

Driver-level minima: **B=1 59.1–61.9 µs** (pick `S nth=32 nt pf0 kb=1`),
**B=8 326.0–338.1/8 = 40.8–42.3 µs/vol** (pick `G=8 sb0 cached pf1 kb=1`;
best number this kernel has ever posted on wallaby, prior best 41.4 mt_r1),
**B=128 6650–6774/128 = 52.0–52.9 µs/vol** (incumbent `G=8 sb1 nt pf6` +
kb=1; mt_r3 band was 53.9–57.5).  PASS rel_l2 = 4.460–4.462e-16 at
B=1/8/128, repeatable (bit-identical) everywhere.

The kb A/B, in-arena, same window — the round's result:

| cell / shape | in-place | kb=1 | delta |
|---|---|---|---|
| B=1, eng=1 S nth=32 (all-to-all = 31/32 sibling lines) | 71.6 | **61.2** | **−14.5%** |
| B=128, incumbent G=8 sb1 nt pf6 | 53.0 | **46.3** | **−12.6%** |
| B=8, forced node shape G=4 sb1 cached pf6 | 53.1 | **47.6** | **−10.4%** |
| B=8, wallaby pick G=8 sb0 cached pf1 | 43.0 | **42.5** | −1.2% |

The gradient is exactly the all-to-all model's prediction: the win grows
with lanes-per-volume (gsz 32 → 8 → 4 → 4-with-own-volume), because that is
the fraction of the slab a lane pulls from siblings.  The node's B=8 shape
(gsz=8, all traffic through socket-0 controllers) sits in the steep part —
if even half the in-arena −10.4% transfers to the driver, the 6.6% mkl gap
closes.  Parallel efficiency vs phase-1 wallaby single-thread: B=1
522.9/59.1 = 8.8× (28%); B=8 568.0/40.8 = 13.9× (44%); B=128 683.3/52.0 =
13.1× (41%).

### What was tried and did NOT work

* Nothing failed correctness, and nothing regressed on wallaby this round.
* pf=6 (slabpf) still loses on wallaby everywhere it is raced (B=8 G=8:
  45.4 pf6 vs 41.9 pf1 cached; G=4 nt: 53.4 vs 46.0 pf0) — as in every
  round since mt_r1: SPR's prefetchers cover the slab walk.  Not evidence
  about the node, whose picks carry pf6 at every batched cell; the rows
  exist for the node.
* Noted risk, not a measured loss: kb=1 adds ~68 KB/thread of L2 footprint
  (competing with the pass-2+3 line working set).  On wallaby's 2-MB-L2
  cores it is free; the node's Cascade Lake L2 is 1 MB.  The A/B runs on
  the node's own arena, so if the footprint inverts the sign there, the
  in-place path ships — that is what the 1%-bar A/B is for.

### Borrowed from other entries

* **mt_r3 VERDICT §6**: the named route itself ("the gang-internal
  all-to-all ... is still the route") — kb=1 is that instruction executed
  as a layout change rather than a decomposition change (the transpose's
  data motion is irreducible; what kb removes is the ownership churn and
  the two redundant SC sweeps, which are not).
* **L64_radix8 mt_r3, change 4**: dropping the trailing gang barrier of
  each gang's last volume (their legacy-gang form) → my sb=1 last-volume
  barrier skip.
* **L64_radix8 mt_r2 (standing)**: the B=128 incumbent shape stays their
  node-proven `gang-T32-g4-nt+slabpf1` = my `G=8 static sb=1 nt pf=6`; the
  kb A/B rides on top of whatever the incumbent rule ships.
* Their mt_r3 negative (p1pf costs −27% when forced in loaded 16-lane
  gangs on wallaby) is noted; my pf axis already races p1 on/off per cell,
  and my node picks at batch have been pf6-without-p1 for two rounds.

### Node predictions (Gold 5218, 2×16, 22-MB L3/socket, DDR4)

* **B=8, the cell this round is about**: the pick should repeat `G=4 sb1
  cached pf6` (its pool is unchanged apart from strictly-additive rows)
  with kb decided by the on-node A/B.  The in-arena kb win at that exact
  shape is −10.4% on wallaby; on the node the all-to-all crosses 1-MB L2s
  instead of 2-MB, which raises the sibling-pull cost kb removes — but the
  68-KB KB footprint also bites harder.  If kb transfers at even half
  strength, 77.1 → ~73 and the mkl gap (72.4) is a coin flip; band
  70–78 µs/vol.  If the node A/B says kb=0, the B=8 story is unchanged and
  the record gains the important negative: the all-to-all residual is NOT
  ownership churn, and next round goes at the decomposition itself.
* **B=128**: the incumbent + kb should ship in ≥2/3 processes; the G=16
  lottery cannot recur (rows gone).  Incumbent kb A/B was −12.6% on
  wallaby; band 62–74 µs/vol if it transfers, 70–76 if not.
* **B=1**: pick repeats `S nth=16 cached pf0`; kb A/B runs on it (S
  nth=32's wallaby A/B was −14.5%; nth=16 halves the sibling fraction to
  15/16 — similar regime).  Band 110–129 µs.
* Monitor asks, in cost order: (1) `FFT64B_VERBOSE=1` once per B — the
  table now prints the kb A/B line (`kb A/B inplace X / kb Y`) next to the
  pick; (2) if B=8 still loses to mkl WITH kb=1 shipped, one forced
  `FFT64B_KB=0` run isolates kb's driver-level effect at the scored cell;
  (3) standing controls unchanged (`FFT64B_ST=0` fingerprint, `FFT64B_DYN`,
  `FFT64B_NTH=16`, `FFT64B_G=16`, `FFT64B_NOHP`).

### Next

1. Read the node's kb A/B lines at all three cells — this round's changes
   are one experiment with three replicates at different all-to-all
   fractions, and the A/B line makes the mechanism legible per cell.
2. If B=8 closes to within ~2% of mkl but does not flip: the remaining
   lever is pass-1 store placement — with kb=1, SC is written once and read
   (gsz−1)/gsz remotely; writing each ky-slab COLUMN to the slot of the
   thread that will pass-23 it (a permuted SC layout, same values, same bit
   class) would make the pass-23 reads mostly slot-local.  That is one
   round of work and only pays if the node shows the residual is still
   line transport, not controller bandwidth.
3. If B=128's spread is now <5% with the incumbent shipping 3/3, consider
   retiring the incumbent override in favour of the (then-proven) arena —
   only with node evidence, not before.
4. Standing dead list (mt): 32-wide central spin barrier, eng=3 at B≥128,
   schedule(dynamic) on 64-unit loops, nth=16 at batched cells (node 3/3),
   dyn in the default pool (node 3/3 + verdict), G=16 at batch ≥ 32 (node
   3/3 across two rounds, removed this round), cross-window wallaby
   comparisons.
