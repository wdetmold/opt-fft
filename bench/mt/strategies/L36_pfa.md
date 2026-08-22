# L36_pfa — strategy record (multicore phase)

Phase-1 history (how the serial kernel got its shape, panel_r1..r11) lives in
`../../geom/strategies/L36_pfa.md`.  This file starts at the 32-core phase.
Phase-1 node baselines used for efficiency below: B=1 117.9 us, B=4 127.7,
B=32 161.8, B=256 174.3 us/vol (panel_r11 scored).  Scored mt cells:
B = 1, 32, 512.

## Round mt_r1

### What was built

The serial kernel — GT-PFA 4x9 codelet, n1_9 DFT9 DAG, two-sweep plane-fused
passes, all the pf machinery — is untouched.  Everything new is a threading
layer plus a regime-aware tuner:

**1. Persistent pinned pthread spin pool** (create()-time; execute() never
creates a thread).  T−1 workers pinned to the CPUs the harness's
`close/cores` OMP mapping uses (read back from one throwaway
`omp parallel num_threads(T)` via `sched_getcpu()`), parked on an epoch spin
over one release word.  Barriers are flag-array: each arriver release-stores
its OWN padded cache line, participant 0 (the driver thread) scans them and
publishes the next phase epoch on the release word.  Main never dispatches
again until every worker — idle ones included — posted its final flag, so
the job descriptor is never written while anyone might read it (the torn-read
hazard is closed by protocol, not by locks).  After the tuner installs its
pick the pool is SHRUNK to the picked team; unpicked spinning workers drag
the all-core clock (measured: the serial B=1 candidate runs 77 us with 31
spinners vs 51.4/56.6 us without, wallaby).

**2. Batched path (K_VOLS)**: thread t owns the contiguous volume block
[nvol·t/T, nvol·(t+1)/T) and runs the untouched serial `run_vols` on it with
its OWN scratch volume, allocated and first-touched by the owning thread in
create() (NUMA-local on the two-socket node).  Zero synchronisation inside
the job; one dispatch + one join per execute.

**3. B=1 paths (within-volume on the pool)**:
* **K_FUSED2** — phase 1 as 36 x-plane units (the existing `phase1_plane`,
  pf0-specialized by constant args), barrier, phase 2 as 324 (y, zb-granule)
  tile units (`mt_xpass`).  2 barriers/execute.
* **K_FUSED3** — phase 1's two subloops split into separate passes over a
  746 KB plane arena: z pass = 324 4-y-row units, y pass = 324 zb-granule
  units, then the same 324 x-tiles.  3 barriers, near-perfect balance
  (36 units over 32 threads is a 2-wave span at 56% efficiency; 324 units is
  98%), pass A and B use the IDENTICAL x-major static partition so each
  thread reads back the arena planes it wrote.  Unit granules are sized so
  every unit writes whole 64-byte lines at both PW — no false sharing at any
  partition cut, by construction.
* Serial remains a candidate (and the shipped fallback) — B=1 must never
  lose to 117.9 us by parallelising.

**4. Tuner**: same discipline as phase 1 (deterministic serially-filled
arena, correctness gate vs a serial reference at 1e-13, self-warming, min of
7/3 rounds, 3% hysteresis toward the simpler candidate).  New axes: kind,
team size T ∈ {32,24,16,8} at B=1 / {32,16} batched (the T=16 row is the
two-socket question: the driver first-touches all caller pages on one
socket, so the far 16 threads may pay UPI for nothing — the node decides).
The arena is deliberately filled single-threaded so create()-time racing
sees the same NUMA placement the scored run does (L23's note).  Rank
tie-break prefers pw4 (the node's standing L=36 pick) — a noisy pw2 win
observed once on a loaded wallaby cannot install (r10 pick-lottery lesson).
Forcing envs for the monitor: `FFT36_MT` (serial/vols/fused2/fused3),
`FFT36_T`, plus the phase-1 `FFT36_PW/MODE/PF`.  `FFT36_PROBE=1` prints a
phase-level timing split of the fused shapes (barrier-only, p1-only,
p2-only) at B=1.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores, shared login
node — quiet-window minima; node numbers will differ, esp. cross-socket)

| cell | 1 thread (same binary) | mt_r1 | pick | speedup / par. eff. |
|---|---|---|---|---|
| B=1   | 56.6 us | **13.50 us** | fused2 T=32 pw4 inplace pf0 | 4.2× / 13% |
| B=32  | 72.7 us/vol (2327 us/call) | **87.0 us/call = 2.72 us/vol** | vols T=32 pw4 inplace pf0 | 26.7× / 83% |
| B=512 | 103.8 us/vol (53.1 ms/call) | **3685 us/call = 7.20 us/vol** | vols T=32 pw4 scratch+nt pf1 | 14.4× / 45% |

rel L2 = 3.59e-16 in all cells (bit-identical to the serial kernel at the
same PW — parallelisation only reassigns whole units); repeatable
(bit-identical across runs); PASS at B = 1, 2, 4, 8, 31, 32, 33, 512;
AVX2-only build (wombat) compiles and passes.

Where the missing cores went, honestly:
* **B=512** is the DRAM wall: 1.5 MB/vol compulsory at 7.20 us/vol =
  ~207 GB/s aggregate on wallaby.  NT scratch beats in-place 5.83 vs 7.02
  us/vol in-arena (the r5–r9 "NT always loses" node rule INVERTS at 32
  threads, exactly as L23_matrixsimd measured at L=23), and nt+pf1 (paced T1
  read prefetch under NT) adds another −17%: 7.02 → 5.83.
* **B=32** sits at 83% efficiency, LLC-resident: in-place pf0 wins, NT loses
  (5.67 vs 2.79 — no DRAM to save, cache residency is the prize).
* **B=1 is coherence-bound, not compute- or barrier-bound.**  FFT36_PROBE
  split (T=32): barrier floor 1.40 us (2 barriers + dispatch ≈ 0.7 us each),
  phase 1 alone 5.40, phase 2 alone 3.18 — isolated sum 7.2 us, full execute
  14.0.  The ~7 us interaction term is cross-core dirty-line traffic: phase 2
  reads lines phase 1 just wrote remote-M (~723 KB of HITM transfers), and
  phase 1's stores invalidate the previous execute's phase-2 readers.  The
  all-to-all is information-theoretic (every x-line output needs all 36
  planes); the 2× coherence multiplier on it is the target.

### What did NOT work, with the numbers that killed it

1. **fused3 at T=32**: 17.1–18.6 vs fused2's 13.7 us.  The balance win
   (probe: pass A 2.54 us vs fused2-p1's 4.0) is real but the third barrier
   plus pushing the pl handoff from L1 registers to an L2 arena costs more.
   At T=8 the order flips (24.2 vs 26.9) — balance matters more when waves
   are long.  Kept as a candidate; the node's cross-socket barriers may
   reorder everything.
2. **prefetchw in fused phase 1 (pf=2)**: 19.28 vs 13.74 us at T=32.  The
   general phase-1 body (runtime flags, 108 live prefetch slots) plus the
   prefetchw uop tax swamps any invalidation-hiding.  Candidate kept for the
   node (CLX snoop behaviour differs) but expected dead.
3. **NT mid stores in fused phase 1 (pf=3, `mt_p1plane_nt`)**: 18.25 vs
   13.74.  On wallaby's single-socket mesh a HITM is ~fast; pushing 746 KB
   through DRAM to avoid it is a bad trade THERE.  This is a wallaby-can't-
   price-it bet (same epistemic class as phase-1 r6/r7): on the node a
   cross-socket HITM chain vs LLC/DRAM re-read is a different ratio, so the
   candidate ships gated and the node tournament judges it.
4. **Deeper phase-2 read prefetch (2 tiles out, +16 doubles)**: 13.98 →
   13.74 us — inside noise.  The HITM term is transfer-bound, not
   lead-time-bound.  Left in (free), but it answered the question.
5. **NTA in-read under NT scratch (nt+pf4, the r7 shape)**: 8.11 vs 5.83 on
   wallaby — expected there (2 MB L2 holds S regardless).  It exists FOR the
   node (S = 3/4 of the 1 MB L2); wallaby cannot reject it, only the node
   can (phase-1 r7 epistemics, unchanged).

### Borrowed, and from whom

* **Spin pool + flag-array barrier + pin-to-OMP-map**: L23_matrixsimd mt_r1
  (their 6.2–8.2 us GOMP-region and 5.0 us central-barrier measurements are
  why this was built first, not discovered independently) and L17_winograd
  mt_r1 (flag-array 0.3–0.4 us vs counter 1.2 us; pool-shrink-after-pick).
* **Volume-parallel contiguous blocks + owner-first-touch scratch**:
  L13_direct mt_r1 and L17_winograd mt_r1.
* **Serial arena fill so the tuner sees the driver's NUMA placement**:
  L23_matrixsimd mt_r1.
* **NT re-race at 32 threads**: L23_matrixsimd mt_r1's inversion result;
  confirmed here at B=512 (−17% vs best cached-store shape).
* From my own phase-1 lineage: pf=1/pf=2/pf=4 machinery reused verbatim as
  vols-mode candidates; the tuner discipline (hysteresis, self-warming,
  correctness interlock, forced-variant envs) carried forward.

### Next round

1. **B=1 coherence term (~7 us)** is the whole prize there.  Ideas in
   order: (a) socket-aware two-stage x-pass — stage the DFT4 leg of the
   x-line PFA so each socket combines its own 18 planes first and only the
   4×9 intermediate crosses the UPI (needs a per-socket reduction buffer;
   halves cross-socket bytes); (b) pair-split the 4 leftover planes of the
   36/32 partition (phase-1 span 2→1.5 waves, ~1.7 us); (c) a
   dissemination barrier if the node's flat-scan barrier costs >1 us
   cross-socket.
2. **Node NUMA at B=512**: read the ip32/nt32/ip16/nt16 A/B off the
   description line.  If T=16 wins, the fix is not fewer threads but far-
   socket threads staging `in` through their local scratch (L13's next-round
   sketch) — only worth building on node evidence.
3. If the node's B=1 fused pick survives, wire `FFT36_PROBE` numbers into
   the description string so the leaderboard carries the barrier/HITM split
   without a manual run.

## Round mt_r2

### Where mt_r1 landed on the node (the input to this round)

B=1 **25.659 us, 1st** (fused2 T=16 pw4 inplace pf0; sd 0.09% — the verdict
calls it the round's most robust B=1 result; 1.49x MKL).  B=32 **5.912, 3rd**
(vols T=32 pw4 inplace pf2; mixedradix 5.589 with vol32+pfw).  B=512
**19.465, 2nd** (vols T=16 pw4 scratch+nt pf1, dead stable; mixedradix vol32
in-place 14.170 in 2-of-3 processes, 22.98 in the third).  Parallel
efficiency vs my phase-1 node baselines: B=1 117.9/25.66 = 4.6x, B=32
161.8/5.91 = 27.4x (108% of the 25.4x clock-adjusted ceiling), B=512
174.3/19.47 = 9.0x (35%) — B=512 is the broken cell.

### The diagnosis that drives this round

My B=512 pick was an **arena artifact**.  19.47 us/vol at 1.5 MB/vol
compulsory = 77 GB/s = one socket.  The tuner's race is milliseconds long on
caller-surrogate pages that are ALL on socket 0 (serial arena fill — by
design, to mimic the driver), so T=32 candidates pay full UPI and lose
in-arena (nt16=20.1 vs nt32=23.7, every process), and nt16 installs.  But
the driver's timing loop runs for seconds; the verdict's §4 finding is that
the far socket demonstrably ends up contributing memory bandwidth (L=6
sustains 175 GB/s > one socket's ~100), with AutoNUMA migration the leading
explanation for the cross-process bimodality — and min-of-samples scores the
good end.  A T=16 team can never benefit: its threads never touch pages from
socket 1, so nothing migrates and it stays stably slow.  Cross-entry node
evidence at my own cell: mixedradix vol32 14.17 (2-of-3), pencilfused's probe
t32=15.71 vs t16=24.01.  Even without migration, 16 remote threads add UPI
read throughput on top of the local socket's controllers (mixedradix's
prediction, ~135 GB/s aggregate → ~16 us/vol in-place, matching their 14–16).

### What changed (tuner only; codelets, passes, pool untouched)

1. **Batched cells (batch >= 32) race T=32 only.**  T=16 rows are generated
   only when `FFT36_T` is set (monitor forced controls keep working).  This
   deliberately OVERRIDES the arena, on the node evidence above; the priced
   risk is stated in the header: if mt_r2's harness pins all memory to
   socket 0, T=32 costs ~15% vs the old nt16 — accepted against the observed
   ~27% loss (19.47 vs 14.17) the other way.
2. **inplace pf=7 joins the batched list** (write-intent-only prefetch, the
   r11 mechanism).  At B=32 the working set (46 MB) sits in the 76 MB
   aggregate cache, so pf=2's paced T1 read cursor is pure uop tax;
   mixedradix's store-side-only pfw wins that cell by 5.8%.
3. **fused2 T=18 joins the B=1 pool** (idea from L36_mixedradix mt_r1, raced
   there on wallaby only): 36 plane units = EXACTLY 2 waves (T=16 spans 3,
   with 12 threads idle while 4 planes finish), 324 tile units = exactly 18
   waves, and only 2 threads sit on the far socket — 4x less UPI exposure
   than the T=24 that already lost in r1, so the 2-wave saving (~2.4 us of
   phase-1 span) has a chance to survive the coherence cost.  Enters below
   T=16 in the hysteresis (higher tuse = more complex): must beat it by >3%.
4. **fused2 pf=3 (NT mid stores) at T=16 and T=18.**  r1 built NT-mid only
   at T=32/24; the node then picked T=16, where the shape was never raced.
   At a one-socket team the HITM it avoids is same-socket (~cheaper), so
   expected dead — but it is the one unpriced cell of that mechanism.
5. Description string now publishes the full batched mode A/B at T=32
   (`ip0/ip2/ip7/nt0/nt1/sc` us/vol) and, at B=1, `t18=` and `ntm<T>=`
   next to the existing ser/f2/f3 — so the next verdict can read the
   migration and team questions off the leaderboard whatever the pick is.

Operation count: unchanged (232 FMA-port vector ops + ~57 swaps per 36-line
over PW lanes; 225 504 port-0 ops/volume at PW=4).  Output remains
bit-identical to the serial kernel at the same PW — parallelisation still
only reassigns whole units.

### Measured (wallaby, Gold 6448Y SPR, 32 threads on one socket, shared
### login node — quiet-window minima; wallaby CANNOT price any of the
### two-socket questions this round bets on, see below)

| cell | mt_r1 wallaby | mt_r2 wallaby | pick (wallaby) |
|---|---|---|---|
| B=1   | 13.50 us | **13.70 us** | fused2 T=32 pw4 inplace pf0 |
| B=32  | 2.72 us/vol | **2.79 us/vol** (2.82 in-arena; ip2=2.94 ip7=2.93) | vols T=32 pw4 inplace pf0 |
| B=512 | 7.20 us/vol | **7.21 us/vol** | vols T=32 pw4 scratch+nt pf1 |

rel L2 = 3.59e-16 everywhere (B=1 3.591e-16, B=4 3.577e-16, B=8 wombat
3.586e-16, B=32 3.586e-16, B=33 3.588e-16, B=512 3.587e-16), all PASS,
repeatable (bit-identical across runs), AVX2-only wombat build passes.
Wallaby is flat vs r1 — expected: its 32 close-bound threads land on one
socket, so removing T=16 from the batched list changes nothing there (T=32
already won), T=18 just looks like fewer threads (r1's mixedradix A/B:
T=18 +12% over T=32 on wallaby), and NT-mid keeps losing on a single-socket
mesh (30.97 vs 31.46 on a loaded run — within noise there).  Every change
this round is a bet only the node tournament can settle; wallaby's job was
correctness and no-regression, both confirmed.

### What did NOT work / was considered and rejected, with numbers

1. **Keeping nt16 in the streaming list with a rank penalty**: the arena gap
   is 15% (20.1 vs 23.7), far beyond any honest hysteresis band, so it would
   always re-install; the only correct move given the arena cannot see
   migration is exclusion (env-restorable).  This is a deliberate
   arena-override and is flagged as such here and in the header.
2. **Tail-hybrid for T=16 phase 1** (fused2 for 32 planes + fused3-style
   split of the last 4): saves ~0.8 us of span at the cost of 2 extra
   barriers (~0.7 us) and 83 KB of arena traffic — break-even at best on the
   model, dominated by the free T=18 candidate.  Not built.
3. **Locality-aware phase-2 unit order** (the verdict's L=17 suggestion): at
   L=36 fused2 every phase-2 tile reads one line from ALL 36 x-planes, so
   each tile's inputs are 1/T local regardless of order — there is no
   ordering gain to have.  Checked by inspection, not built.
4. **pfwd sweep at batched cells**: r11's serial question, but at 32 threads
   the streaming cells are bandwidth-bound, not RFO-latency-bound, and the
   list is already 14 wide; skipped to keep the pick lottery small.

### Borrowed, and from whom

* **T=32-at-streaming over the arena's T=16**: the diagnosis rests on
  L36_mixedradix mt_r1's scored vol32 numbers + their UPI-throughput
  prediction, L36_pencilfused's t32/t16 probe pair, and the mt_r1 VERDICT
  §3.2/§4 AutoNUMA analysis.
* **fused2 T=18**: L36_mixedradix mt_r1's split pool ("T=18 divides both
  36-unit phases exactly").
* **pf=7 batched**: my own r11 mechanism, promoted to the mt list on the
  evidence of mixedradix's pfw win at B=32.

### Predictions for the node (so mt_r2's verdict can grade them)

* B=512: pick nt32-pf1 or ip32-pf2 at **10–16 us/vol** (nt floor 1.5 MB/vol
  at 135–175 GB/s aggregate = 8.6–11.1; in-place at the same rates = 13–16).
  If it lands >18, the far socket contributed nothing and the T=16 exclusion
  was wrong — the ip0/nt0 A/B on the desc line will say which.
* B=32: ip0 or ip7 at **5.5–6.0 us/vol** (the pf=2 read-cursor tax was worth
  0.1–0.3 us if mixedradix's 5.8% gap is mostly that).
* B=1: t16 pf0 again at ~25.7, OR t18 at **23–25** if 2 far threads cost
  less than one wave of planes (~2.4 us).  t18/ntm on the desc line carry
  the A/B either way.

### Next round

1. Read the B=512 ip0/ip2/ip7/nt0/nt1/sc A/B off the node desc line.  If nt
   wins but sits >2 us/vol above the 1.5 MB floor at the achieved rate, the
   next lever is explicit socket-local staging (far-socket threads walk
   their block through NUMA-local scratch — L13_direct's sketch), which is
   only worth building on that specific evidence.
2. If t18 wins B=1, try t=20/t=22 (still 2-wave, more workers for phase 2's
   324 tiles); if t16 holds, B=1 is coherence-saturated at one socket and
   the remaining ~10 us is the all-to-all itself — the honest end state.
3. If the harness fixed page placement this round (verdict §6 asked for it),
   re-admit T=16 rows: the arena becomes trustworthy again and the override
   comment in fft3d_create() should be retired.

## Round mt_r3

### Where mt_r2 landed on the node (the input to this round)

B=1 **25.818, 2nd** (fused2 T=16 pw4 pf0; mixedradix split12 23.012).  B=32
**5.236 us/vol, tie for 1st on min** but a 3-pick lottery (190.3 / 167.5 /
169.8 us/call: one process arena-picked pf2 and paid 14% on the driver).
B=512 **31.569, a 1.62x STABLE regression** (r1: 19.465) — the verdict's
SS3.4 item 1 is me: I deleted my own r1-winning T=16 scratch+nt row on the
AutoNUMA-migration theory and shipped T=32 inplace pf7, whose own arena
number (24.1–25.0) was already worse than the row I deleted (20.1), and
whose driver number (31.57) was worse still.  Pre-registered band 10–16,
missed by 2x.  Verdict SS5 (the round's central finding, measured by
L8_fusedaxes' governor): the machine has TWO placement regimes — sometimes
the caller's pages are effectively socket-0-resident for the whole timed
loop (fr=0 measured, nb=1), sometimes the far socket's controllers end up
contributing (L=6 sustains 200 GB/s; mixedradix's B=512 hit 9.99 us/vol in
the 1-of-3 process that picked vol32-sntp, 19.27 in the two that picked
vol16) — and **no create-time arena can tell which regime the driver's
buffers are in**, because the regime is a property of those buffers, per
process.

### What changed (tuner + threading layer; codelets and passes untouched)

1. **B>=128: mode fixed to scratch+nt pf=1** (nt1: phase 1 writes the
   per-thread L2-resident volume scratch, phase 2 reads it back and
   NT-streams to out; 1.5 MB/vol compulsory).  Node evidence from both
   regimes points at this one shape: it is my r1 winner at T=16 (19.47)
   and the mechanism behind mixedradix's 9.99 at T=32 (their sntp).  The
   T=16 row is RESTORED (the verdict's first L=36 order).  ip/sc/nt0/nt4
   rows remain raced at T=32 for the description A/B and as fallbacks
   under forcing envs.
2. **K_VOLS dyn=1, a work-stealing twin**: threads pull 1–2-volume chunks
   off one atomic cursor instead of owning a static block.  Why: in the
   socket-0 regime the static T=32 split is bounded by its REMOTE half —
   16 far threads streaming their entire 384 MB share over UPI while the
   16 near threads finish early and spin; 1.5 MB/vol over ~20-25 GB/s of
   UPI read throughput is ~30-37 us/vol, which IS the measured 31.57.
   Stealing lets the near socket take the work its own controllers can
   serve, so T=32-bad-regime should land near (local + UPI) aggregate
   (~15-17 us/vol modeled) instead of min-of-halves.  In the spread
   regime it degenerates to the static split (measured cost of the
   cursor + chunk-tail prefetch loss: dy=6.02 vs nt1=5.87 us/vol
   in-arena on wallaby, ~2.5%).
3. **Execute-time TEAM GOVERNOR at batch>=128** (the round's main move):
   the first 6 executes round-robin {dyn32-nt1, s16-nt1, s32-nt1} on the
   CALLER's real buffers — the only surrogate-free race possible — with
   each shape's first call as its unscored warm; lock the fastest; then
   re-probe the losers round-robin every 16th execute (>3% relocks, so a
   late migration is followed).  The driver's warmup (5) plus inner
   calibration (~3–15 executes) absorb the probation; min-of-samples
   ignores the probes that land in the timed region.  All three shapes
   are correctness-gated in create() before the governor may run them.
   Any forcing env (FFT36_MT/T/PW/MODE/PF/DYN, or FFT36_GOV=0) disarms
   the governor so the monitor's controls measure exactly what they force.
   The pool is not shrunk when the governor is armed; instead idle
   workers now spin ~1 ms then nap in 50 us sleeps, so a locked s16
   no longer drags the all-core clock (join tail <=50 us on a ~10 ms
   execute).
4. **Placement instrument**: L8_fusedaxes' get_mempolicy page-home scan
   ported verbatim; fr (percent of ~256 sampled in/out pages off the
   main thread's node) is read at execute 0 and execute 40 and published
   with the governor state:
   `gov{fr0 fr40 nb d32 s16 s32 lock sw}` — the diagnostic the verdict
   ordered for L=36 B=512, readable off the leaderboard whatever happens.
5. **B=32 lottery closed**: candidate list cut to {ip0, ip7} (+pw2 ip0).
   pf2's paced read cursor — the 1-in-3 losing pick — is gone from this
   cell; ip0 is the rank incumbent, ip7 must beat it by >3%.
6. **B=1: fused2 T=12 joins the pool** (from L36_mixedradix's
   node-winning split12: exactly 3 planes/thread, zero phase-1 imbalance,
   324 tiles = exactly 27 waves, 12 < 16 threads in the phase-2
   all-to-all, one socket; my dispatch is 0.7 us vs the 2.0 theirs paid,
   so their 23.0 should be beatable).  The pf=3 NT-mid rows are dropped
   (ntm16=33.0 on the node, dead at every team, twice).  t12= rides the
   B=1 description string next to t18=.

Operation count: unchanged (232 FMA-port vector ops + ~57 swaps per
36-line over PW lanes; 225,504 port-0 vector ops/volume at PW=4).  Output
remains bit-identical to the serial kernel at the same PW: the dyn twin
and the governor only reassign whole volumes to threads.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores on one
### socket, shared login node — quiet-window minima; wallaby cannot price
### any two-socket question, its job is correctness + no-regression)

| cell | mt_r2 wallaby | mt_r3 wallaby | notes |
|---|---|---|---|
| B=1   | 13.70 us | **13.85 us** | pick fused2 T=32 pw4 pf0 (unchanged); t12 reads 36.9 on one socket — expected there, exists for the node |
| B=32  | 2.79 us/vol | **2.56 us/vol** (82.0 us/call) | pick ip0 or ip7, both present, no other rows to lotteried into |
| B=512 | 7.21 us/vol | **7.15 us/vol** (3662 us/call) | governor locked s32 (7.16) over d32 (7.39) and s16 (10.35) on the symmetric machine — exactly right there |

Governor A/B on wallaby's real driver buffers at B=512:
`gov{fr0=0 fr40=0 nb=1 d32=7.16-7.39 s16=10.35-10.38 s32=7.16 lock=s32|d32 sw=0}`
(d32 and s32 are within ~3% of each other on one socket, so either locks;
both are ~30% ahead of s16 there).  Forced controls verified:
FFT36_T=16 installs vols T=16 nt1 (nc=2, 10.2-10.4 us/vol on wallaby's
half-team), FFT36_GOV=0 installs the arena winner (nt1 T=32, 7.28),
FFT36_DYN=1 forces the stealing twin (7.37, nc=1).  In-arena A/B line at
B=512: `ip0=11.1 ip7=8.4-8.9 nt0=6.9-7.2 nt1=5.9-6.3 dy=6.0-6.3
n16=9.5-9.8 sc=9.3-9.4`.

Correctness: rel L2 vs numpy 3.575e-16 … 3.591e-16, PASS at
B = 1, 2, 8 (wombat AVX2-only build), 32, 33, 130, 512; repeatable
(bit-identical output across runs) at every cell including the governor
cells — the governor changes thread assignment, never arithmetic.
Setup 0.06 s (B=1) … 0.41 s (B=512).

### What did NOT work / was considered and rejected, with numbers

1. **Keeping the pf=2 row at B=32 with a rank penalty**: the r2 node data
   shows the arena cannot separate ip0/ip2/ip7 (5.95–6.46, overlapping
   across processes) while the driver separates them by 14% — no penalty
   fixes an instrument that cannot resolve the choice; the losing row had
   to go.  Same reasoning as r2's nt16 exclusion, applied to my own cell.
2. **Weighted static splits for the bad regime** (socket-0 threads take
   ~80% of the volumes): a static weight serves exactly one regime; the
   stealing cursor is the same idea made self-tuning, at a measured 2.5%
   single-socket cost.  Not built separately.
3. **chunk=4 stealing**: rejected on the tail model — a far-socket thread
   grabbing a 4-volume chunk near the end adds a ~2-5 ms UPI straggler
   tail at B=512; chunk<=2 caps the tail at ~1 ms.  (chunk=1 below
   nvol=256, chunk=2 above; the 512-grab cursor cost is ~26 us against a
   ~3.7 ms execute.)
4. **fr-threshold team selection** (pick T from the page scan instead of
   racing): the scan says where the pages are, not what the interconnect
   will deliver under 32 streaming threads; the empirical probation race
   answers the actual question and needs no threshold to calibrate.  fr
   is published as a diagnostic only.
5. **B=1 structural work** (socket-aware two-stage x-pass): still not
   built — the node's winning teams are single-socket (T=16, mixedradix's
   T=12), so there is no UPI in the B=1 all-to-all to stage around; the
   remaining ~10 us over the barrier floor is same-socket coherence, and
   T=12's smaller reader set is this round's cheap bite at it.

### Borrowed, and from whom (also credited inline in the source)

* **Execute-time governor (lock + periodic re-probe) and the
  get_mempolicy page-home scan**: L8_fusedaxes mt_r2, ported nearly
  verbatim (their gov_scan_remote and cfg lock/probe protocol).  The
  verdict SS6 asked for exactly this port to L=36.
* **scratch+nt as the streaming mode at T=32**: L36_mixedradix mt_r2's
  sntp (their 9.99 us/vol 1-of-3 process is the good-regime existence
  proof at my cell); transitively my own r1 nt machinery.
* **fused2 T=12 at B=1**: L36_mixedradix mt_r2's split12, the scored
  B=1 winner (23.012).
* **Idle-worker nap after a hot-spin budget**: L36_mixedradix mt_r2's
  pool refinement.
* **T=16 row restoration and "wide-team must be priced, not assumed"**:
  the mt_r2 VERDICT SS3.4/SS5/SS6, plus L6_pfa's T=32/T=16 200-vs-85 GB/s
  bracket as the cleanest statement of the stakes.

### Predictions for the node (so the verdict can grade them)

* **B=512**: governor probation on the driver's buffers reads all three
  shapes.  Bad regime (fr0=0, pages stay put): s16 ~19-20, s32 ~30-32,
  d32 ~15-18 (the stealing model above) — lock d32 if the UPI-throughput
  model holds, s16 if UPI contention eats it; either way the cell lands
  **15-20 us/vol**, never the 31.57 again.  Good regime: d32/s32 ~10-12,
  lock either, **10-12 us/vol**.  If a 1-in-3 process is good-regime, the
  scored min is the 10-12; the honest representative is whatever fr0
  says.  gov{...} carries the whole story either way.
* **B=32**: ip0 or ip7 at **5.2-5.6 us/vol**, and the three processes
  should now agree to ~2% (the lottery rows are gone).
* **B=1**: t12 wins at **22-24 us** if mixedradix's split12 result
  transfers to my cheaper barrier (their 23.0 minus ~1 us of dispatch
  difference), else t16 holds at ~25.8; t12= and t18= on the description
  line grade this either way.

### Next round

1. Read gov{fr0 fr40 lock d32 s16 s32} off the node's three processes.
   If lock=d32 in the bad regime with d32 ~15-17, the stealing model is
   confirmed and the remaining B=512 gap to the 1.5 MB floor is UPI read
   throughput — the next lever would be far-socket threads NT-staging
   their in-reads through local scratch to convert UPI demand reads into
   streamed ones (only worth building on that evidence).  If lock=s16
   everywhere, UPI contention is the binding fact and B=512 is honestly
   ~19 in the bad regime; say so and stop.
2. If t12 takes B=1, race t=9 (4 planes each, 36 tile waves) and a
   t12-with-pair-split-of-nothing is moot — instead try trimming the
   phase-2 tile partition to plane-major order so each thread's 27 tiles
   share source planes (reader-set locality, the only coherence knob
   left at one socket).
3. If the harness fixes page placement (verdict SS6 item 2), the governor
   becomes a no-op that locks one shape in probation and costs nothing;
   the create-time arena becomes trustworthy again and the T question can
   move back into it.  Leave the governor in either way — it is the only
   instrument that races in the scored regime by construction.

## Round mt_r4

### Where mt_r3 landed on the node (the input to this round)

B=1 **25.720, 2nd** (fused2 T=16 pw4 pf0; mixedradix split12 23.027).
B=32 **5.212 us/vol, 1st** (vols T=32 ip0; the r2 lottery is dead — 166.8 /
170.3 / 166.8 us/call, ≤2% spread).  B=512 **19.322, 3rd, 1.95x behind
L36_mixedradix's 9.896 running the SAME nominal shape** (per-thread
L2-resident volume scratch, NT stream to out, wide team).  The governor
did exactly what I built it to do — raced d32/s16/s32 on the caller's real
buffers, locked s16 (20.05 in-race, 19.32 delivered, an honest instrument)
— and that is precisely why it lost: **none of its readings can see the
steady state a SUSTAINED wide team produces.**  The verdict's L=36 order:
find the difference between 9.9 and 20.0.

### The diagnosis that drives this round

The governor measures T=32 in **one-execute islands** — six probation
executes, then one probe every 16th — sandwiched between s16 executes.  A
one-execute probe of s32 runs against socket-0-resident pages and reads
24.4; it never triggers, and never benefits from, the page spread that a
wide team running the WHOLE loop produces.  The direct evidence that
sustained-wide is a different regime: L36_pencilfused pinned team=32
scratch+nt for the whole run and its own scan read **fi=28 / fo=50 percent
of caller pages remote** in the scored process (10.9 us/vol); my probe
islands read fr0=0 in all three processes.  L36_mixedradix pinned
v1-vol32-sntp and delivered **9.9 us/vol 3-of-3** (150.9 GB/s — above one
socket's DDR4 controllers, so both sockets' memory demonstrably serve it).
So the choice "probe-race vs pin" is not instrumentation hygiene, it
decides which regime the scored loop runs in.  A probe long enough to
reach the migrated steady state would BE the scored run; there is nothing
left to race, and the r3 verdict's SS6 says stop building placement
instruments.  Pin it.

### What changed (tuner + policy; codelets, passes, pool untouched)

1. **B>=128 PIN** (policy from L36_mixedradix mt_r3, adopted whole): when
   /proc/sys/kernel/numa_balancing reads 1 and no forcing env is set,
   fft3d_create() installs **vols static T=32 pw4 scratch+nt pf=1**
   directly — the shape behind 9.9 (mixedradix) and 10.9 (pencilfused) —
   still correctness-gated at 1e-13 against the serial reference.  With
   nb=0 the never-migrate arena and the driver share one regime, so the
   create-time race installs as before (its r3 winner was n16 at an honest
   19.3–20; that row stays raced as the fallback).  FFT36_PIN=0 (and the
   old FFT36_GOV=0, kept as an alias) or any forcing env disarms the pin.
2. **~3 s create-time dwell at the pinned cell** (L36_pencilfused mt_r2's
   dwell via L36_mixedradix mt_r3, whose 9.9 processes carried 3.0 s
   setups against my 0.69): after installing the pin, create() runs the
   pinned shape on the arena until the plan is ~3 s old.  AutoNUMA's
   per-task scanning ramps over the first seconds; the driver's timed loop
   at B=512 is only ~0.3 s, so an aged process starts it with the balancer
   already active.  Setup is unscored.
3. **The execute-time governor is DELETED.**  What survives at streaming
   is read-only: the get_mempolicy fr scan at execute 0 and execute 20
   (r3's second scan at 40 never fired — the cell only runs ~29 executes),
   published as `pin{on nb fr0 fr20}`.  dyn rows are gone from the default
   list (0 scored wins panel-wide, r3 verdict); FFT36_DYN still generates
   the stealing twin for forced controls.  Streaming candidate list
   trimmed to {nt1@32, nt1@16, nt0@32, ip7@32, pw2-nt1@32} — gate rows +
   the description A/B.
4. **B=1: honest-clock team race.**  The tournament now times every
   candidate with the pool SHRUNK to that candidate's team, walking the
   ladder descending (32, 24, 18, 16, 12, 9, 8, then serial at pool=1),
   then stops and restarts the pool for the installed pick.  Why: idle
   pool workers spin hot between back-to-back tournament dispatches (they
   never reach the 1 ms nap), so the old full-pool race timed a T=12
   candidate under a 32-thread all-core AVX-512 clock — my r3 node arena
   read t12=30.4 while mixedradix's scored split12, the same decomposition
   on a 12-thread team, ran 23.0.  My own mt_r1 record measured the
   mechanism directly (serial 77 us with 31 spinners vs 51.4 without) and
   I still let it price the team ladder for two rounds.  **T=9 joins the
   ladder** (mixedradix r3: the other exact divisor — 4 planes and 36
   tiles per thread).  The description now carries t9= beside t18=/t12=.
5. B=32: untouched (the cell is won and stable; verdict: leave it alone).

Operation count: unchanged (232 FMA-port vector ops + ~57 swaps per
36-point line over PW lanes; 225,504 port-0 vector ops/volume at PW=4).
Output remains bit-identical to the serial kernel at the same PW — the pin
and the ladder only choose which thread runs which whole unit.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores on one
### socket, shared login node — sessions this round ranged quiet to busy
### (sd up to 17% on the streaming cell); wallaby still cannot price any
### two-socket question — its job is correctness + no-regression)

| cell | mt_r3 wallaby | mt_r4 wallaby | pick |
|---|---|---|---|
| B=1   | 13.85 us | **13.53 us** (quiet window) | fused2 T=32 pw4 pf0 (wide always wins on one socket; honest ladder changes nothing there by design) |
| B=32  | 2.56 us/vol | **2.80 us/vol** (busier window; path byte-identical to r3) | vols T=32 pw4 ip0 |
| B=512 | 7.15 us/vol | **7.33–7.41 us/vol**, setup 3.01 s (dwell fires) | vols-PIN T=32 pw4 scratch+nt pf1 |

In-arena A/B on wallaby at B=512: nt1=5.86 nt0=6.86 ip7=9.36 n16=9.67 —
same ordering as r3, and the pin installs exactly the arena's winner
there, so no wallaby regression by construction.  Forced controls all
verified: FFT36_T=16 → vols T=16 nt1 installs, pin{on=0}, 8.28 us/vol
(half-team correctly slower on one socket); FFT36_PIN=0 → race installs
nt1 T=32, 7.27; FFT36_DYN=1 → dyn twin runs, 7.34; FFT36_MT=serial → B=1
serial pf0 installs.  fr scan reads fr0=0 fr20=0 on wallaby's single
socket, as it must.

Correctness: rel L2 vs numpy 3.575e-16 … 3.591e-16, PASS at
B = 1, 2, 32, 33, 130 (pin path with a non-divisible batch), 512;
repeatable (bit-identical output across runs) at every cell; AVX2-only
host (no AVX-512) builds clean and passes at B=1 and B=130 (the pin
correctly targets the pw2 nt1 row when pw4 does not exist).  Setup
0.07 s (B=1) … 3.01 s (pinned cells, dominated by the deliberate dwell).

### What did NOT work / was considered and rejected, with numbers

1. **Keeping the governor with longer probes**: a probe long enough to
   reach the spread steady state is indistinguishable from just running
   the shape — the instrument's entire budget (6 probation + 1-in-16
   probes inside a ~29-execute cell) cannot contain it.  The r3 data is
   the refutation: s32 probed at 24.4 in all three processes while the
   same shape sustained scores 9.9.  No probe schedule fixes an
   instrument whose act of measuring changes the regime back.
2. **Pinning dyn32 instead of static32**: my own r3 governor read d32
   25.2–26.1 vs s32 24.4 (static ahead even pre-spread), mixedradix's
   scored 9.9 is static, and dyn is 0-for-every-scored-race panel-wide.
   The stealing cursor solves a problem (min-of-halves in the stuck
   regime) that the pin's sustained-wide regime does not have.
3. **A create-time dwell on the CALLER's buffers**: not possible — create
   never sees them; the dwell ages the process on the arena and the
   driver's own warmup executes do the caller-page part.
4. **B=1 structural work (socket-staged x-pass)**: still deferred — the
   node's B=1 winners remain single-socket sub-teams (mixedradix's 12),
   so the honest ladder + T=9 is this round's cheap bite; structure only
   if the repriced ladder says the cell is UPI-bound after all.

### Borrowed, and from whom (also credited inline in the source)

* **The pin policy, its nb=1 gate, and wide-team-sustained-at-streaming**:
  L36_mixedradix mt_r3 (their v1-vol32-sntp pin, 9.9 us/vol 3-of-3).
* **The create-time dwell**: L36_pencilfused mt_r2, via L36_mixedradix
  mt_r3's simpler use of it.
* **The sustained-wide-spreads-pages evidence**: L36_pencilfused mt_r3's
  scored fi=28/fo=50 scan under their pinned team=32.
* **T=9 in the B=1 ladder**: L36_mixedradix mt_r3.
* **The honest-clock diagnosis at B=1**: the r3 VERDICT SS4.4 (spin pools
  depressing clk512 panel-wide) plus my own mt_r1 spinner measurement,
  finally applied to my own tournament.

### Predictions for the node (pre-registered, so mt_r4's verdict can grade)

* **B=512**: all three processes install vols-PIN T=32 nt1 and land
  **9.5–11.5 us/vol** with pin{fr0 or fr20 > 0} in at least one process —
  the mixedradix/pencilfused number, now with my cheaper dispatch.  If it
  lands ~19–24 with fr0=fr20=0 in all three, the spread never happened for
  my process and the residual difference from mixedradix is inside the
  phase bodies, not the schedule — that A/B (their 9.9 vs my pinned same
  shape) would then be the next round's first diff.  Either way the cell
  stops being decided by a probe that cannot see the answer.
* **B=32**: unchanged path, **5.2–5.6 us/vol**, ≤2% spread again.
* **B=1**: the honest ladder reprices t12/t9 from 30.4/— to the low-to-mid
  20s; pick t12 or t9 at **22.5–24.5 us** if mixedradix's split12 result
  transfers to my cheaper barrier, else t16 holds at ~25.7 and the ladder
  numbers on the description line say the sub-team clock story was wrong.

### Next round

1. Read pin{fr0 fr20} and the B=512 number off all three processes first.
   If pinned-32 lands at ~10, L=36 B=512 is at the same 150 GB/s wall as
   mixedradix and the next lever is the verdict's "what does the last 35%
   of two-socket DDR4 cost" question — likely a socket-aware split where
   each socket's threads keep their NT streams on their own controllers
   (needs the driver's pages actually spread; the fr numbers will say).
2. If B=1's honest ladder installs t12/t9 and lands 22–24, the remaining
   gap to the ~15 us barrier+port floor is the same-socket all-to-all;
   the next structural idea is plane-major phase-2 tile ordering per
   thread (reader-set locality), sketched in r3 and still unbuilt.
3. If the pin misses (19–24 with fr=0), diff my nt1 phase-1/phase-2
   against mixedradix's sntp bodies line by line — same nominal traffic,
   2x apart, would then be a code fact, not a placement fact.
