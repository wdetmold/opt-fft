# L36_pencilfused — strategy record (multicore phase)

Phase-1 history (how the serial kernel reached its current form: PFA 4×9
interleaved-complex line kernel, plane-fused pass A + strided pass B, the
n1_9 DFT9 DAG, the mode zoo, eleven rounds of it) lives in
`../../geom/strategies/L36_pencilfused.md`. This file starts at the multicore
phase. Phase-1 wallaby baselines used below (same host, driver-level, fast
windows): B=1 51.8–53.3 µs, B=32 72.3 µs/vol, B=256 ~100 µs/vol.

## Round mt_r1 — first parallelization: volume chunks, a two-phase B=1 split, and a pthread spin pool

### What changed

The serial kernels are UNTOUCHED — same arithmetic, same operation count
(232 FMA-port vector ops + 57 port-5 shuffles per 36-point line over PW
lanes; 225 504 FMA-port ops/volume at PW=4), and the parallel layer does
zero extra arithmetic, so every strategy below is bit-identical to a serial
execution of the same pass-A order. What was added:

1. **Per-thread NUMA-owned scratch.** Each of up to 32 threads gets a mid
   slot (1.43 MB, holds the ping-pong pair) and a plane-buffer slot (24 KB),
   both whole-page-sized so no slot shares a page, allocated once and
   **first-touched by the owning thread** inside a parallel region in
   `fft3d_create()` — which also spins up the OpenMP pool, so no timed call
   creates a thread. (Pattern from L13_direct / L17_winograd mt_r1; the
   caller's `in`/`out` are driver-freaded on one socket and not ours to fix.)

2. **VP (volume-parallel), the batched path.** Contiguous static volume
   chunks over T threads (ranges computed from the team OpenMP actually
   delivers — L13_direct's rule), each thread running the serial per-volume
   pipeline on its own scratch. Per-thread mode raced at plan time over
   {inplace, istream0, istream+pfw, scratch+NT} × {pw2, pw4} × T {32, 16}.
   The T=16 candidate is the UPI question (in/out live on one socket; the
   far 16 threads may pay interconnect for nothing) — wallaby's one-socket
   32-thread run cannot answer it, the node's tuner will.
   The single-core mode zoo was PRUNED to those four: modes 3–6 existed to
   overlap one core's reads with its NT drain, which 32 threads do
   naturally (more threads = more outstanding misses); modes 9–11 target
   single-core node L2/DSB stories that the B=1 cell no longer runs into.

3. **TP (two-phase within-volume), the B=1 path.** Phase 1: all nvol×36
   pass-A planes over the team (a 20.25 KB plane is one unit; planes are
   64-B multiples, no false sharing). One barrier. Phase 2: all nvol×324
   pass-B units (a unit = one flat group at PW=4, a PAIR at PW=2 — either
   way it owns whole 64-B lines, so partitions never split a cache line).
   L=36 is the geometry where this pays: one volume is ~52 µs of single-core
   work against ~5 µs of OpenMP region cost, where L13/L17 (2.5–8 µs of
   work) measured within-volume splits as pure loss.

4. **A persistent pthread spin pool under the B=1 path (strat "2phase-pool").
   Adopted from L17_winograd mt_r1**, stated plainly — their design wholesale:
   raw pthreads created once in `fft3d_create()`, bound to the exact cores
   OMP's close/cores map uses (`omp_get_place_proc_ids`), parked on an
   epoch-counter spin, synchronised by a FLAT arrival-flag/release barrier
   (each arriver writes its OWN padded line; thread 0 scans them — misses
   overlap — and publishes one release word derived from the dispatch epoch).
   Their negative result is honored too: no central atomic-counter barrier
   (they measured ~1.2 µs/barrier at T=16 from serialized RFOs on one line).
   Added in this file: a **spin-then-park idle policy** (a worker parks on a
   condvar after a spin budget; the dispatcher takes the mutex only to wake
   parked workers, classic re-check-under-lock, no lost wakeups), so an
   idle or unpicked pool never competes with the OpenMP paths for cores.
   The pool is destroyed at the end of create() unless it won the race.

5. **Bit-class discipline kept from phase-1 r11, extended to strategies.**
   y-first pass A (fingerprint 3.748e-16) vs z-first (3.586e-16) may never
   be chosen by a timing coin flip or two processes produce bit-different
   output (my first build did exactly that at B=32 — see "did not work" #1).
   The class is deterministic: batch > 1 → z-first only (per-thread volume
   sets stream; y-first's plane read order is the documented 2× loss on
   non-warm input, phase-1 r3: 101 vs 58 µs/vol); batch == 1 → the phase-1
   L2 gate (L2 holds in+out → y-first, else z-first; wallaby 2 MB → y-first,
   node 1 MB → z-first, wombat 256 KB → z-first, all verified). Within one
   class every candidate — either width, any team, any strategy — is
   bit-identical (lane-wise vector arithmetic), so timing chooses freely.

6. **Tuner**: same 1e-11 correctness interlock against the serial PW=2
   INPLACE reference and interleaved-rounds-min protocol as phase 1, over
   the candidate lists above; pool candidates are timed in a separate wave
   so their spinning never contends with an OpenMP candidate mid-measurement.
   Simplest-wins 2% band: serial < VP < TP-OMP < TP-pool, then smaller team,
   then mode 0 < 12 < 8 < 2. Decomposition probes ride the description
   string onto the leaderboard (`pl32/pl16/tp32/ser` at B=1,
   `t32/t16/is0/nt` µs/vol at batch). New force flags for A/Bs:
   `-DFFT36PF_FORCE_STRAT/TEAM/MODE/PW/YF`.

### Operation count

Unchanged from panel_r11: 232 FMA-port + 57 shuffle vector ops per line,
225 504 FMA-port vector ops/volume at PW=4. The parallel layer adds per
execute: one OMP region (VP/TP) or one pool dispatch + one flat barrier +
one completion scan (~1 µs total at T=32), and zero arithmetic or extra
memory traffic. Compulsory DRAM traffic per volume is unchanged per mode
(scratch+NT: 1.5 MB compulsory; inplace/istream: ~2.2 MB with the RFO).

### What was measured (wallaby, Gold 6448Y, 32 threads close/cores, shared
login node — same-window serial comparisons where quoted; µs per transform,
driver min over ≥3 runs; rel_l2 and bit-identical re-runs on every run)

| cell | mt_r1 | pick | serial same host | speedup / par. eff. |
|---|---|---|---|---|
| B=1 | **13.0 µs** (12.9–13.3 across 4 processes, sd ≤0.7%) | pool tp32 y-first pw4 | 53.3 µs same window | **4.1× / 13%** |
| B=32 | **108.1 µs = 3.38 µs/vol** | vp32 istream+pfw pw4 | 72.3 µs/vol | **21.4× / 67%** |
| B=512 | **3638 µs = 7.11 µs/vol** | vp32 scratch+NT pw4 | ~100 µs/vol (B=256) | **~14× / 44%** |

Unscored cells for the record: B=2 13.7 µs/vol (TP-OMP), B=3 10.7, B=4 8.2,
B=16 6.6. Correctness: rel_l2 = 3.748e-16 at B=1 (y-first class on wallaby),
3.573–3.591e-16 at B=2..512 (z-first class); tolerance 1e-12; repeatable
(bit-identical across processes) at every batch listed — after fix #1 below.
AVX2-only path exercised end-to-end on wombat/Haswell (B=1 40.4 µs with the
z-first gate firing, B=2, B=32): PASS, repeatable. Clean compiles: bare -O2
without -fopenmp, -march=haswell.

Where the missing cores went, honestly:

* **B=512 is aggregate-bandwidth-bound**: 512 × ~1.5 MB compulsory traffic
  in 3.64 ms ≈ 210 GB/s against ~250–300 GB/s practical on wallaby's
  socket — ~85% of the roofline, so the 44% "efficiency" is the memory
  wall, not sync (sd 0.18%, zero barriers). NT is what got it there (#3).
* **B=1 is sync/straggler-bound on a SHARED box**: critical path is 2
  pass-A planes (~1.5 µs) + ~10 pass-B units (~0.8 µs) + ~1 µs pool sync
  ≈ 3.5–4 µs ideal; the measured 13.0 µs carries wallaby's multi-tenant
  scheduling noise through two barriers (any preempted worker stalls all
  32). The exclusive node should land much closer to the model; the pl/tp/
  ser probe in the description gives the node's own curve.
* **B=32 at 67%** with in-arena == scored regime: each thread does exactly
  one volume; the residual is the shared-L3/bandwidth term plus one region.

### What was tried and did NOT work — with the number that killed it

1. **Letting timing choose the pass-A bit class.** First build raced VP
   mode 0 (y-first) against modes 2/8/12 (z-first) at batch: two
   consecutive B=32 processes installed different classes and tryout
   flagged **NOT REPEATABLE** (bit-different, both correct). The phase-1
   r11 lesson replayed at MT scale within an hour. Fixed by the
   deterministic class rule in item 5; B=32/512 then repeatable across
   processes every run since.
2. **A 64-volume tuning arena under 32 threads.** At tb=64 (93 MB, half
   wallaby's L3) the tuner ranked istream+pfw 4.49 µs/vol AHEAD of
   scratch+NT 5.64 and picked it; the end-to-end B=512 run said pfw 10.3
   vs **NT 7.35** (−29%). The phase-1 arena rule ("an NT-vs-cached decision
   is only measurable on a working set that streams past every L3
   involved") recurs with the threshold scaled by the thread count: 32
   threads re-reference the arena 32× faster. Arena cap raised 64 → 128
   volumes (187 MB); the tuner then picks NT and matches the forced run.
3. **OpenMP regions for B=1** (kept as a raced fallback, but they lose):
   the TP-OMP team curve is fork/join-flat — in-arena 26.3 (t8) / 23.3
   (t16) / 19.7 µs (t32) for ~52 µs of work, i.e. adding 24 threads buys
   6.6 µs. The pool's same-window twins: 26.6 (t8) / 22.2 (t16) / **13.5
   (t32)** — the pool is −32% at t32 and its curve still slopes, so the
   region cost, not the algorithm, was the wall. This reproduces
   L17_winograd's finding at a geometry where the split itself pays
   (their B=1 could only reach 1.73×; this one is at 4.1× and
   node-bound, not structure-bound).
4. **istream0 (pf-less) per thread at streaming batch**: tb=128 in-arena
   11.25 vs istream+pfw 7.81 vs NT 6.02 µs/vol — under 32-way bandwidth
   contention the paced read cursor and the RFO-avoiding NT store BOTH
   still matter; "more threads = prefetch is free" is only true until the
   fill buffers are the shared resource. (istream0 stays in the pool: at
   B=32 on wallaby, where the batch fits L3, cached in-place istream+pfw
   beat NT 4.48 vs 12.52 — regime-dependent, the tuner prices it per cell.)

### Borrowed (attribution)

Pthread spin pool, flat arrival-flag/release barrier, epoch-derived release
values, OMP-place core binding, and the shrink-after-tuning discipline:
**L17_winograd mt_r1** (including their central-counter-barrier negative
result, not repeated here). Per-thread first-touched scratch replicas,
contiguous chunks, ranges-from-delivered-team: **L13_direct mt_r1** (same
pattern independently in **L13_rader mt_r1**). Re-measure-NT-under-threads
rationale and the per-thread-slice regime rebase: **L13_rader mt_r1**
(their −21% at B=8192). Bit-class discipline and the L2 gate: my own
phase-1 r11. Spin-then-park idle policy, the two-wave tuner separation, and
the deterministic strategy/class split by batch: this file, this round.

### Predictions for the node (stated so they can be scored)

* Picks: B=1 **pool tp32 z-first pw4** (the L2 gate fires: 1 MB < 1.46 MB;
  fingerprint 3.59e-16 is the marker), B=32 raced between istream+pfw and
  NT at T=32 (47 MB vs 22 MB socket-0 L3 says streaming, so NT — but in/out
  are all on socket 0, so T=16 could upset), B=512 **vp scratch+NT**, T
  decided by the UPI question the t16 probe publishes.
* B=1: **5–9 µs** if the node's exclusive cores let the pool run at the
  model (~3.5–4 µs) plus CLX's slower cores (node serial 113 µs = 2.2×
  wallaby's 52); ≥11 µs means the barrier scan is pricier on two sockets
  than one — try a 16-thread single-socket pool pick there (already raced).
* B=512: **13–18 µs/vol** if socket-0 DRAM (~100 GB/s) is the wall and NT
  holds; t16 winning the probe would confirm UPI-bound and point at the
  copy-in staging idea below.

### Next

1. **Read the node's probe strings** (`pl32/pl16/tp32/ser` at B=1;
   `t32/t16/is0/nt` at batch) before touching anything — they are the
   decomposition curve measured on the machine that matters.
2. **If t16 beats t32 at B=512** (UPI-bound): don't shrink the team — stage
   far-socket threads' input through their NUMA-local mid (one extra
   746 KB local read converts remote demand misses into one remote stream),
   L13_direct's copy-in idea; their record defers it on node evidence, mine
   would too.
3. **Pool for the batched cells**: B=32 pays one OMP region (~5 µs) on a
   108 µs call — ~5%; wire VP through the pool dispatch if the node's B=32
   is within 10% of a rival.
4. **B=1 imbalance**: at T=32 four threads carry 2 planes (2:1 pass-A
   critical path). A finer pass-A unit needs a per-plane sub-barrier
   (pp is written then read within one plane) — only worth it if the node's
   B=1 lands ≥11 µs with the sync probe showing the barrier is NOT the cost.
5. B=2..8 currently go through TP-OMP; if those cells are ever scored,
   extend the pool path to nvol > 1 (it already takes nvol as a parameter;
   only the tuner gate keeps it at B=1).
