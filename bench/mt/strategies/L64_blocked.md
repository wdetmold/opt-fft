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
