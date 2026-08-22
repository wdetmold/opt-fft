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
