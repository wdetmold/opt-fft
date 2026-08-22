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
