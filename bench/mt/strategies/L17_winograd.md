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

## Round mt_r3

Standing after mt_r2 on the node: B=1 7.604 us (3rd, stable; matrixsimd
6.163), B=256 1.078 us/vol (3rd, a stable 1.31x REGRESSION from r1's 0.824),
B=4096 1.220 us/vol (held 1st at 3.3x the best library) but scored on a
1.73x pick lottery: 8662 / 5029 / 4998 us across the three processes, where
r1 had sd 0.2%.  The serial kernel and EVERY execute path are byte-for-byte
unchanged this round (same 3*289*296 = 256,632 FP instructions / 423,096
flops per volume).  mt_r3 is entirely a tuner-fidelity round, because the
node data says both of my r2 losses were self-inflicted by the tuner:

* The slow B=4096 process picked `g4 dyn=2`; the two fast ones `i4/h4 dyn=0`.
* ALL THREE B=256 processes picked `dyn=1` (r1's winner was `h8` static).
* `nt=0` in all six batched process picks, although wallaby's honest
  streaming race had NT winning by 15% and the RFO it deletes is a third of
  the streaming cell's compulsory DRAM traffic.

The diagnosis for the third bullet: my arena was sized from
`sysconf(_SC_LEVEL3_CACHE_SIZE)` = ONE socket's L3.  On the node that gave
384 volumes = 57.6 MB in+out against 44 MB of AGGREGATE L3 (2 x 22), so the
"streaming" race was running mostly in cache -- precisely the regime where
NT stores rightly lose (L17_rader mt_r1 measured +55% for NT at a
cache-resident cell).  The mt_r2 VERDICT's mechanism (section 5) explains
the first two: the arena races in the pre-migration transient (every caller
page on socket 0), where dynamic grabbing genuinely helps; the scored
statistic is the min over a multi-second loop during which AutoNUMA migrates
pages toward the thread that keeps faulting them -- a steady state only a
STATIC split can reach, because only a static split gives each page one
consistent claimant.  My own winning process at B=4096 ran 193 GB/s, above
one socket's DRAM, so the fast regime is real and reachable; dyn=2 threw it
away and scored 8662.

### What was changed (all tuner; zero arithmetic, zero execute-path bytes)

1. **dyn REMOVED from the tuner surface** (VERDICT section 6, verbatim
   instruction: "it should be removed from the tuner surface, not
   re-raced").  Stage 2b is deleted outright; the batched schedule is always
   the static contiguous split.  `p->dynb` stays as a field, always 0.
2. **Arena sized to 2.5x AGGREGATE L3.**  The create()-time scratch region
   already runs one thread per core; each thread now records
   `sched_getcpu()`, and the plan reads each cpu's
   `topology/physical_package_id` to count the packages the team spans.
   cap = 2.5 * (L3 * nsock) / 157 KB, clamped [384, 1024].  Node: 733
   volumes = 115 MB, which actually streams; wallaby: 1002, unchanged.
3. **Owner-touched arena** (ADOPTED FROM L17_matrixsimd mt_r2): bin/bout are
   first-touched in parallel, volume v by the thread that owns v under the
   full-team static split, THEN filled with values serially.  This is the
   AutoNUMA steady state the min-statistic scores.  A serially-touched arena
   prices the transient instead -- that transient is where all three B=256
   processes found dyn=1 "winning".
4. **Explicit streaming regime from the WORKING SET, not from an arena
   measurement** (VERDICT section 6): strm = (batch * 157216 B > 2 * aggregate
   L3).  Node: B=256 40 MB < 92 MB -> resident; B=4096 644 MB -> streaming.
   At streaming cells only, two incumbency rules act on the final pick, both
   among bit-identical candidates: prefer the full team within a 5% median
   margin (L=6 measured T=32 at 200 GB/s vs T=16 at 85 on one binary), and
   prefer the best NT candidate within a 5% median margin (a partly-resident
   arena structurally UNDER-prices NT, never over-prices it).
5. **Median-of-3 pick statistic** (ADOPTED FROM L17_matrixsimd mt_r2, per
   VERDICT 3.2): the grid keeps all three rep times per candidate and ranks
   by median; min-of-reps is what let one lucky g4+dyn=2 rep take a scored
   cell to 8662 us.
6. **Grid pruned** so the honestly-sized (2x larger on the node) arena keeps
   setup sane: variants a/b/c/d/e (0-5, 10, 11) dropped from the batched grid
   (no batched win on any machine since the fused variants landed in phase
   1), quarter-team column dropped (never within 2x of winning in two
   rounds), and the B=1 s2 knob fixed at 0 (s2=0 won 6/6 node processes plus
   wallaby; my r2 record pre-committed to deleting it on exactly this
   evidence).  Setup at B=4096 went 4.2 s (node, r2) -> 1.6 s (wallaby, r3)
   despite the bigger arena.

### Operation count

Unchanged: 3*289*296 = 256,632 FP instructions / 423,096 flops per volume.
The round adds zero arithmetic and zero new execute code -- it only changes
which pre-existing bit-identical configuration the plan picks.

### Measured (wallaby, SPR, 32 threads close/cores; shared login node, treat
### as relative -- session noise this week is worse than usual)

| cell | mt_r2 (same host) | this round | pick, 3 independent processes |
|---|---|---|---|
| B=1    | 4.36 us | 5.41-5.8 us (noisy session; sd 5%) | h8 units, t1 raced (32 this session) |
| B=256  | 113-145 us/call | **118.1-123.5 us = 0.46-0.48 us/vol** | h8 static T=32, 3/3 (pfw flips 0/1, a +-1% knob) |
| B=4096 | 3395-3887 us/call | **3437-3800 us = 0.84-0.93 us/vol** | **h8+nt+pf T=32 static, IDENTICAL 3/3** |

rel L2 3.25-3.27e-16 at B=1/8/33/256/4096, bit-identical across runs at
every batch (tryout cmp).  The headline is the right-hand column: the
B=4096 pick that was a 1.73x lottery on the node is now the same
configuration in every process, and it is the NT configuration that the
node's undersized arena could never see.  ns/strm/ar=ot telemetry now rides
the description string so the node run itself will show whether nsock=2 and
strm were detected as designed.

### What did NOT work / what was deliberately not done

* No new mechanism was attempted this round -- the r2 evidence convicted the
  tuner, not the kernel, and every change above is either a VERDICT
  instruction, a proven borrow, or a deletion.  Nothing measured worse than
  its predecessor on wallaby.
* B=1 stays structurally unchanged (7.6 us stable on the node, 3rd).
  matrixsimd owns B=1 with fewer vector ops and one fewer sync phase; the
  levers left in my file (w=4 unit split, phase-3 prefetchw inside the pool
  units) attack at most ~0.5 us of skew and none of the ~2 us all-to-all,
  so I spent the round's risk budget on the two batched cells instead.

### Borrowed from other entries, named

* Owner-touched (parallel-first-touch) tuner arena -- L17_matrixsimd mt_r2.
* Median-over-min for create-time races -- L17_matrixsimd mt_r2 / VERDICT 3.2.
* Working-set regime threshold + wide-team/NT incumbency-with-margin --
  the mt_r2 VERDICT sections 5 and 6 (L8_fusedaxes's fr=0 governor data and
  L6_pfa's T=32/T=16 bracketing are the measurements behind it).

### Pre-registered node expectations

* **B=4096**: pick should be h8+nt (or q8+nt) T=32 static in all three
  processes.  If the 193 GB/s regime holds, NT's traffic ratio (157/236 KB
  per volume) predicts **0.85-1.0 us/vol**; anything at ~1.22 with nt=1
  means the fast regime is NT-hostile in a way wallaby cannot show, and
  anything with nt=0 means the 733-volume arena is STILL not streaming
  honestly on CLX -- check strm=1 and ns=2 in the description first.
* **B=256**: static h8 restores r1's 0.824; the owner-touched arena and
  median may shave a little more (wallaby reads 0.46-0.48).  I do not
  expect to pass matrixsimd's 0.755 -- that gap is compute, not schedule.
* **B=1**: unchanged code, expect 7.5-7.7 us and boring.
* Run spread at B=4096 should collapse from 73.3% to single digits; that,
  not the headline, is this round's claim to check first.

### Next round

1. If nt=1 lands and B=4096 sits at ~0.85: the next lever there is the
   staging copy itself (78.6 KB L2 round trip per volume) -- fold the NT
   stream into the fused pass-3 store epilogue per kx-block (write the
   block's 17-line runs to a line-aligned bounce row, stream full lines,
   handle the 16-mod-64 volume offset with the existing peel) to cut the
   staging traffic ~17x.
2. If B=256 merely restores 0.824: the remaining 9% to matrixsimd is
   arithmetic density; the only in-family answer is the interleaved-complex
   pass-1 rewrite my phase-1 record gated on p1/fu, which the node probe
   still prices at ~37% (p1=6.16 of fu=17.4) -- re-derive before spending.
3. B=1: if the panel merges regimes per the VERDICT's L=17 note, the useful
   contribution from this file is the fused h-kernel units + flat barrier;
   the dense X-first schedule should come from matrixsimd's exemplar, not be
   reinvented here.

## Round mt_r4

Standing after mt_r3 on the node: B=1 7.563 us (3rd; matrixsimd 5.976),
B=256 0.823 us/vol (3rd; matrixsimd 0.756 -- the r2 regression repaired as
predicted), B=4096 1.219 us/vol (1st, 3.36x the best library) with the run
spread collapsed from 73.3% to 0.4% -- the r3 tuner-fidelity claims all
landed (the VERDICT calls the record "the round's best-calibrated").  But
the B=4096 telemetry shows nt=0 shipped despite strm=1/ns=2 detected as
designed: even under the honest aggregate-L3 owner-touched arena, the
mt_r2 full-volume NT staging lost by more than the 5% margin on the node.
The cell runs 129 GB/s of in+out with a 193 GB/s DRAM demand (the RFO is
the difference), and L17_rader sits 5.7% behind, so the RFO deletion is
both the offensive and the defensive move this round.

### What was changed: ntb, the RFO deletion without the staging round trip

One mechanism, pre-registered as item 1 of my r3 "next round" list, and
matching the shape behind every >130 GB/s streaming result in mt_r3
(L36_mixedradix 150.9 GB/s, L36_pencilfused 137.5, both "per-thread L2
tile + one NT write of out" -- the VERDICT section 4.3 headline):

* **ntb (pf bit 4).**  In the fused h4/h8/q4/q8 variants (i4 inherits via
  fused23_h4), pass 3 builds each kx-block's out span -- which is
  CONTIGUOUS and line-aligned relative to the volume base: 4624 doubles
  per w=8 block, 2312 per w=4 block, 578 for the kx=16 tail -- in one hot
  scratch row (buf row 6, reused block after block, so it never leaves
  cache) and then streams it once to the caller's out with NT stores
  (nt_copy_span: 16B movntpd peel to the first line boundary, zmm
  movntpd body, peel again at the tail; volumes are 78608 B = 16 mod 64
  so block bases cycle 0/16/32/48 mod 64).  Same DRAM traffic as mt_r2's
  nt (in + NT out, no RFO), but the 78.6 KB per-volume staging write+read
  round trip becomes a ~37 KB row that stays resident.  A new botailL
  table (botail rebased by -2*16*289) lets the kx=16 tail build in the
  bounce row; the tail's scalar (16,16) pencil lands at local offset 544.
  Exact copies of identically-computed values: bit-identical, cmp-checked
  at every batch tried.
* **Tuner surface**: pf columns extended {0,1,2,3,4,8,9} -> +{16 = ntb,
  17 = ntb + input prefetch}, offered at h4/h8/q4/q8/i4 exactly like nt
  (q takes 16 only -- its interleaved real loads replace the input pf).
  nt stays in the grid so the node arbitrates full-volume vs per-block
  staging by measurement, not by my belief.  The existing streaming
  margin rule ("best NT candidate within 5% of the pick") already covers
  ntb since it keys on pf >= 8.  Everything else -- static split, arena,
  median-of-3, B=1 path -- is byte-identical to mt_r3.

### Operation count

Unchanged: 3*289*296 = 256,632 FP instructions / 423,096 flops per volume.
ntb adds zero arithmetic; it adds 2*4913 doubles of load+NT-store copy per
volume (non-FP), against the RFO's 78.6 KB of deleted DRAM traffic.

### Measured (wallaby, SPR, 32 threads close/cores; shared login node,
### relative numbers -- forced A/Bs interleaved 3 rounds to control drift)

Forced A/B at B=4096, T=32, same binary set, min over samples:

| config | round 1 | round 2 | round 3 |
|---|---|---|---|
| h8 plain   | 4627 us | 4619 | 4624 |
| h8+nt (r2) | 3871 | 3929 | 3864 |
| **h8+ntb** | **3726** | **3704** | **3722** |
| h4 plain / h4+ntb | 5518 / 4133 | 5506 / 4141 | 5549 / 4072 |
| q8 plain / q8+ntb | 4576 / 3593 | 4520 / 3719 | 4483 / 3531 |

ntb beats nt at every variant and plain by 20-26%; q8+ntb touched 3531 us
= 0.862 us/vol.  Full tuner, three independent processes at B=4096:
**h8+ntb picked 3/3, 3423 / 3458 / 3461 us = 0.836-0.845 us/vol** (r3 same
host: 3437-3800 with nt=1-or-flip).  One earlier, noisier process picked
h8+nt at 3911 -- the nt-vs-ntb arena gap is ~4-5%, inside session noise,
and both delete the RFO, so that flip costs at most ~5% and never regresses
past r3.  B=256, three processes: **h8 static, nt=0 ntb=0, 3/3** at
115.9-126.3 us = 0.45-0.49 us/vol (forced nt +47%, forced ntb +32% at the
resident cell -- the honest race rejects both, exactly as it should).
B=1 4.893 us (path untouched), B=33 27.9 us.  rel L2 3.25-3.27e-16 at
B = 1 / 33 / 256 / 4096, bit-identical outputs across configs and reruns.
Parallel efficiency vs the phase-1 serial kernel (r1 record's same-host
numbers): B=4096 15.8/0.836 = 18.9x / 59%; B=256 9.89/0.46 = 21.5x / 67%.

### What did NOT work / what was deliberately not done

* Nothing measured worse than its predecessor this round; the one
  mechanism built is a pre-registered borrow with three independent node
  proofs behind its shape.  The known residual risks are recorded
  instead: (i) nt-vs-ntb can flip in the arena within ~5%, which is
  benign; (ii) if Cascade Lake NT-store throughput caps the cell near
  L=36's 150 GB/s, ntb lands at ~1.0-1.05 us/vol, not the 0.81 the
  193 GB/s demand number promises -- still a win over 1.219.
* B=256's 9% gap to matrixsimd is compute density (r3 verdict and my own
  probe agree); the interleaved-complex pass-1 rewrite stays gated on the
  node's p1/fu ~ 37% probe and was not attempted with the round's risk
  budget on the streaming cell.
* B=1 stays structurally unchanged for the second round (7.5-7.6 stable,
  3rd).  The VERDICT's L=17 item is matrixsimd's experiment (its pool vs
  my OMP dispatch, the 2.29 vs 2.89 GHz clk512 gap); duplicating it from
  my side would spend the round rediscovering their answer.

### Borrowed from other entries, named

* The L2-tile + NT-out store shape, applied per kx-block:
  L36_mixedradix mt_r3 (vol32-sntp, 150.9 GB/s 3/3) and L36_pencilfused
  mt_r3 (mode-2 tile, 3.13x), via the mt_r3 VERDICT section 4.3; the
  full-volume ancestor is L17_matrixsimd mt_r1's "pipelined + NT store".
* The discipline of keeping the displaced mechanism (nt) in the race
  rather than deleting it on belief: my own r2/r3 lesson.

### Pre-registered node expectations

* **B=4096**: pick h8+ntb (h4+ntb / q8+ntb equally acceptable; nt=1 an
  acceptable near-tie flip) in all three processes, strm=1 ns=2 in the
  description.  If the 193 GB/s no-RFO regime holds: **0.81-0.90 us/vol**;
  if NT stores cap near L=36's 150 GB/s: **1.0-1.1**.  Anything at ~1.22
  with ntb=1 means per-block NT staging is no better than the RFO path on
  CLX and the r3 nt=0 verdict extends to ntb; anything with ntb=0 AND
  nt=0 at >5% margin would mean CLX NT store throughput under this access
  pattern is the binding constraint -- check the forced numbers in the
  next round's arena before believing either.
* **B=256**: nt=0 ntb=0 static h8, ~0.82 us/vol, unchanged.
* **B=1**: unchanged code, expect 7.5-7.7 us and boring.

### Next round

1. If ntb lands at ~0.85: the residual staging cost is the bounce row's
   ~74 KB/vol of L2 read+write.  Folding the NT stream directly into
   tsto8's shuffle epilogue would delete it, but the lane runs are 272 B
   and 16-mod-64 -- per-run movnt needs a peel that fragments every run
   into <=4 partial-line WC flushes; only worth building if the node
   shows ntb's copy loop as exposed time (compare ntb's per-vol gain to
   the 74 KB/L2-bandwidth bound first).
2. If the panel merges L=17 regimes: contribute the fused h kernels +
   ntb epilogue; take the dense X-first B=1 schedule from matrixsimd's
   exemplar rather than reinventing it.
3. B=256: the remaining 9% is arithmetic density; re-derive the p1/fu
   probe on the node before spending anything on the interleaved-complex
   pass-1 rewrite.
