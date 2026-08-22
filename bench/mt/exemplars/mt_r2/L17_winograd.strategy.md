# L17_winograd — multicore strategy record

Phase-1 history (how the serial kernel got to its current form, 19 tuner
variants, kernel H, the whole story) lives in
`../../geom/strategies/L17_winograd.md`.  This file starts at the multicore
phase.

## Round mt_r1

First multicore round; no other generation's mt records existed yet
(context.md was empty), so nothing could be borrowed from rivals this time.
The serial kernel is UNCHANGED — same 296-FP-instruction 17-point module,
same 3*289*296 = 256,632 FP instructions / 423,096 flops per volume.  What
was added is purely a parallel harness around it, plus tuners that pick team
shapes by measurement.

### What was built

1. **Batched path (B >= 2): volumes across threads, NUMA-owned scratch.**
   Each thread gets its own 241 KB scratch (6 x 17*SP doubles), allocated
   AND first-touched by that thread inside an `omp parallel` region in
   `fft3d_create()` — under `OMP_PROC_BIND=close, OMP_PLACES=cores` thread t
   sits on core t in every later region, so the scratch stays socket-local
   and page-aligned per thread (no cross-thread scratch line, ever).
   `fft3d_execute()` splits the batch into contiguous static chunks (thread
   t gets volumes [t*B/T, (t+1)*B/T)) and runs the untouched serial
   `run_vols` on a plan clone whose `buf` points at the thread's scratch.
   All 19 phase-1 variants (including the software-pipelined p/q) work
   per-thread unmodified, and the result is bit-identical to serial.

2. **Threaded stage-2 tuner with a team-size axis.**  The phase-1 streaming
   tuner (arena = 2.5x L3, blocked per candidate, 3 reps, min) now times the
   THREADED path over (variant, pf-mask, T) with T in {32, 16, 8}.  The T
   axis exists because the driver freads `in` and memsets `out` from its
   main thread, so on a two-socket node every caller page is first-touched
   on one socket, and the far half of the team may pay UPI for nothing —
   whether it does is measured in the scored regime, not guessed.  On
   wallaby (single-socket 32-thread run) it picked var=h8, pf=1, T=32 at
   B=256.

3. **B=1 path: within-volume decomposition on a persistent spin pool.**
   Three data-parallel phases over the fused w=8 schedule (kernel C),
   separated by two barriers plus one completion barrier:
     * phase 1: 36 `p1g8` vector groups + scalar tail = 37 units -> A;
     * phase 2: 2 kx-blocks x 17 z-groups + the kx=16 tail block = 35 units
       -> per-block mini-buffers.  Unlike serial `fused23_f8`, which reuses
       one [17][136] mini-buffer for both blocks sequentially, the blocks
       live at br+0 / br+2312 and the 17x24 tail buffer at br+4624
       (2*2312+408 = 5032 = 17*SP exactly), so both blocks' pass-2 units run
       concurrently.  Block rows are 17 cache lines each, so units write
       disjoint lines — no false sharing;
     * phase 3: 34 (blk,h) groups + tail block = 35 units -> out.
   Same instructions, same values, same order as serial f8 per output =>
   bit-identical; the barriers only order disjoint writes.
   The team is RAW PTHREADS created once in `fft3d_create()`, parked on an
   epoch-counter spin, bound to the exact cores OMP's close/cores map would
   use (read from `omp_get_place_proc_ids`), synchronised by a flat
   arrival-flag/release barrier: each arriver writes its OWN padded line,
   thread 0 scans them (independent lines, misses overlap) and publishes
   one release word whose value is the global barrier sequence 3*epoch-2..
   3*epoch — derived from the dispatch epoch so threads that sat out a
   tuning dispatch can never be out of phase.  After tuning, the pool is
   shrunk to the picked team size so unused cores are given back.
   Team size raced in create() over {2,4,8,16,32} against the serial
   stage-1 pick; serial remains the fallback (t1=0) if it wins.

### Measured (wallaby, Sapphire Rapids, 32 threads close/cores; dev numbers)

| cell | this round | single-thread same kernel | speedup / par. eff. |
|---|---|---|---|
| B=1    | **4.60–4.90 us**  | 7.97 us (h8 serial)       | 1.73x / 5.4%  |
| B=256  | **137.2 us** = 0.536 us/vol | 2531 us = 9.89 us/vol | 18.2x / 57%  |
| B=4096 | **4430 us** = 1.082 us/vol  | 64720 us = 15.8 us/vol | 14.5x / 45%  |

rel L2 3.3e-16 in all cells; repeatable (bit-identical across runs).
B=4096 at 1.08 us/vol is ~145 GB/s of in+out traffic before RFOs — this
cell is DRAM-bound on wallaby, which is why efficiency drops to 45%; more
threads cannot help, only less write traffic can (see "next").

### What did NOT work, with the numbers that killed it

* **One `omp parallel` region per B=1 execute.**  First implementation of
  the within-volume path.  T=2 ran 10.3 us against 7.9 serial with only
  ~5 us of compute per thread; T=8 7.28 us, T=16 7.45 us — libgomp's
  fork/join + barriers cost ~5 us per call on wallaby and ate the entire
  prize.  The tuner correctly returned t1=0 (stay serial).  Do not retry
  OMP regions for anything this short; the pthread pool exists for a
  reason.  (The batched path keeps OMP regions: at 137+ us per call the
  ~2-5 us region cost is noise.)
* **Central atomic-counter sense barrier in the pool.**  ~1.2 us per
  barrier at T=16: sixteen serialized RFOs on one counter line.  With it,
  best B=1 was 5.16 us at t1=16.  The flat arrival-flag/release barrier
  (per-thread padded lines) cut this to ~0.3-0.4 us per barrier and moved
  the optimum to t1=32 at 4.60 us.
* **Trap for future rounds: `omp_get_max_threads()` without the harness
  env.**  On a raw wallaby shell it returns 128 (2 sockets x 64 HT) and the
  first build happily took all of them — a rules violation waiting to
  happen.  Now hard-capped at 32.

### Next

* **The node will re-tune everything.**  Cross-socket barriers (CLX UPI)
  will likely push the B=1 pick down to t1=8..16, and the single-socket
  first-touch of the caller's buffers may make T=16 win the batched cells;
  both axes are in the tuner, so the node decides.
* **B=4096 write traffic.**  The cell is bandwidth-bound and 1/3 of the
  traffic is the `out` RFO.  NT stores can't be dropped in directly —
  `tsto8`'s output runs are 272 B and not 64-B aligned — but a line-aligned
  staging + `movntdq`-per-line epilogue, or wider use of the existing
  paced-CLWB column, is the obvious ~20-30% prize.  Phase 1 measured NT
  stores as a serial LOSS four rounds running; at 32-core saturation that
  conclusion must be re-measured, not assumed.
* **B=1 floor.**  ~0.7 us of the 4.6 is compute; the rest is 3 barriers,
  wake latency, and phase imbalance (37/35/35 units over 32 threads is a
  2:1 critical path).  A w=4 unit split (73/70/70 units) and/or a
  dissemination barrier are the next levers; also worth trying kernel H in
  the phase-2/3 units (the fused loops are exactly where H pays serially).
* **B=2..31 are unscored and currently naive** (one volume per OMP thread);
  if the sweep ever scores them, give the pool a batch mode.

## Round mt_r2

Standing after mt_r1 on the node: B=1 8.998 us (2nd, matrixsimd 7.112),
B=256 0.824 us/vol (3rd, matrixsimd 0.726), B=4096 1.222 us/vol (1st).
The serial kernel is again UNCHANGED: 3*289*296 = 256,632 FP instructions /
423,096 flops per volume.  Everything below is scheduling, stores, and sync.

### What was changed

1. **nt (pf bit 3): NT-store staging for the batched cells.**  BORROWED FROM
   L17_matrixsimd, whose mt_r1 node pick at B=4096 was "pipelined + NT store"
   and whose record shows NT flipping from serial loss to 32-thread win.
   Each volume is computed into two new per-thread scratch rows (buf rows
   6-7, 80.5 KB, first-touched with the rest) and then streamed to the
   caller's out with `_mm512_stream_pd` (16B `movntpd` peel/tail -- volumes
   are 78608 B = 16 mod 64, so volume bases are only 16B-aligned).  This
   deletes the out RFO, a third of the streaming cells' DRAM traffic, for
   the price of an L2-resident 78.6 KB round trip.  No FP change; the copy
   stores exact copies, bit-identical.  Offered to the tuner at h4/h8/q4/q8/
   i4, never assumed (phase 1 measured NT as a serial loss four rounds
   running).
2. **Dynamic batched schedule (dynb volumes per atomic grab)** raced against
   the static split -- also from L17_matrixsimd's record.  One padded
   counter line, `fetch_add(dynb)` per grab; assignment-only, bit-identical.
3. **Stage 2b of the batched tuner**: the grid still races (variant, pf, T)
   under the static schedule; a new stage then races the grid's TOP-3
   (variant, pf) combos plus the three nt candidates x dyn {0,1,2,4} x T,
   jointly.  Two lessons forced the joint/top-3 shape (numbers below).
4. **B=1**: phase-2/3 unit bodies switched from the f8 (kernel C) groups to
   the h8 ones (k17_h2_8 / k17_h3_8 / k17_e2_8) -- h8 is what the node picked
   serially and at B=1 in mt_r1, and the h groups drop the explicit
   load/store arrays.  Result is now bit-identical to serial h8 (was: f8;
   both are cmp-identical to each other, so the output bytes are unchanged).
   The phase-2->3 global barrier can now be replaced by three per-block
   ready counters (s2mode=1): a phase-3 unit (blk,h) needs only its own
   block's 17 z-units, so block-0 consumers overlap block-1 producers.  The
   mode is RACED against the mt_r1 barrier (s2mode=0), jointly with T, whose
   grid is refined to {2,4,8,12,16,20,24,32} around the node's t1=16 pick.

### Measured (wallaby, SPR, 32 threads close/cores; dev numbers, quiet window)

| cell | mt_r1 | this round | node-relevant pick |
|---|---|---|---|
| B=1    | 4.60-4.90 us | **4.36 us** | h8 units, t1=12, s2=0 (barrier) |
| B=256  | 137.2 us = 0.536/vol | **113.4 us = 0.443/vol** | q8+pfw, dyn=1 |
| B=4096 | 4430 us = 1.082/vol | **3395 us = 0.829/vol (-23%)** | q8+NT, dyn=1 |

rel L2 3.26-3.27e-16 everywhere, repeatable (bit-identical across runs).
Caveat: wallaby's shared L3 moved the B=256 cell 114 -> 145 us between
sessions (52 users; within-run sd stayed 0.04-1%), so treat the batched
numbers as relative.  B=4096 stayed 3395-3887 across the same windows and
q8+NT+dyn was picked every time.

### What did NOT work, with the numbers that killed it

* **Per-block counters as the only phase-2/3 sync.**  First build replaced
  the barrier outright; interleaved A/B against the mt_r1 binary at B=1:
  counters 5.0-6.0 us vs barrier 4.4-5.1 us, and the T race sagged to
  t1=12/16 from 32.  The 34 `lock add`s on two shared lines plus the
  phase-3 spin loads are a milder edition of the central-counter barrier
  that mt_r1 already measured at 1.2 us/crossing -- reintroduced by me from
  the overlap argument, killed by the same mechanism.  Kept only as the
  raced s2mode=1; wallaby picks s2=0.  If the node also picks s2=0, delete
  the counters next round.
* **NT raced only under the static schedule.**  The main grid (static) never
  picked nt: at B=4096 the grid winner was g8+pf+pfw at 4399 us while FORCED
  h8+nt ran 3840 vs h8-plain 4527.  NT's fate flips with the schedule
  (dyn=1 + NT is the winning combination), so a pf-only race can never see
  it -- hence stage 2b racing (candidate x dyn x T) jointly.  Same shape as
  my r4 "never race interacting knobs sequentially" lesson, now with the
  schedule as the interacting knob.
* **Racing dyn on the single grid winner only.**  The h8/q8 grid ranking is
  within wallaby's session noise, and q8+pfw went 145 -> 114 us the moment
  dyn was on -- a win stranded whenever h8 happened to top the grid.  Hence
  top-3 combos into stage 2b, not top-1.

### Borrowed from other entries

* NT staged stores at full-socket bandwidth, and the dynamic-grab schedule:
  L17_matrixsimd mt_r1 (its exec20_w4 NT pipelined pick and its dyn axis).
* The "re-race under the real regime, jointly" discipline: my own phase-1
  r4 record, extended to the schedule knob.

### Next

* **Node predictions to check against the next leaderboard**: B=4096 should
  drop from 1.222 toward ~0.9 us/vol if the node's tuner also lands on
  q8/h4+NT+dyn (its 2-socket UPI imbalance is exactly what dyn fixes; its
  22 MB L3 makes B=256 a streaming cell too, so NT may be picked there as
  well, unlike wallaby).  B=1 should improve from the h units + refined T
  grid; whether s2=1 ever wins is the node's call.
* **B=1 floor**: still 3 sync points (wake, bar1, completion) + spin wake.
  Untried levers: prefetchw of each thread's phase-3 out lines during phase
  2 (pfw never offered inside ovp_body), and a w=4 unit split (73/70/70
  units) for T>16 balance.
* **B=256 on the node**: if NT+dyn is not enough to pass 0.726, try
  socket-local T=16 with the caller's pages all on socket 0 -- the T axis
  already races 16, but only at the grid's static stage; 2b now covers it.
