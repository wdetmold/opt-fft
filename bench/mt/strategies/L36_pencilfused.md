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

## Round mt_r2 — NUMA balancing is the B=512 story; pool discipline for B=1; pool-VP for B=32

### Where mt_r1 landed on the node, and the diagnosis this round is built on

Node (Gold 5218, 2x16 CLX, caller buffers first-touched on socket 0):
B=1 **28.5** (2nd, pool tp16/18 z-first pw4; L36_pfa 25.7), B=32 **5.64**
(2nd by 1%, volpar istream0 t32; L36_mixedradix 5.59), B=512 **24.5** (3rd
by a lot, volpar scratch+NT t32; L36_mixedradix 14.2, L36_pfa 19.5).

The B=512 forensics, from the raw t_*.json of all three L=36 entries:

* My in-arena probe said nt=15.7 us/vol; the scored run said 24.5.  The
  arena is serially filled (socket-0 pages, same as the driver), so page
  placement does NOT explain the gap directly.
* L36_mixedradix's scored runs: r1 = 23.0, r2/r3 = **14.2-14.5** with
  in-arena picks of 22.2-23.8 — the END-TO-END run beat their own arena by
  40%, but only in runs whose process had lived ~3.5 s before sampling
  (their setup: 3.0 s).  My setup was 0.52 s and my runs never sped up.
* 2.2 MB/vol at 14.2 us/vol = ~155 GB/s — more than one socket's DRAM.

Only one mechanism fits all of it: **automatic NUMA balancing** (the node
runs Ubuntu 22.04; kernel.numa_balancing=1 on every machine I can check).
The balancer starts scanning ~1 s after process start and migrates a page
to the socket that DEMAND-faults it.  Cached in-place shapes (istream0/
istream+pfw) demand-read `in` and demand-RFO `out`, so with a static
volume partition both buffers converge to the accessing socket and the run
gets two sockets' bandwidth.  NT stores never fault, so under scratch+NT
`out` stays on socket 0 forever and the 16 remote threads push every line
through the UPI: that is my 24.5.  My arena's nt=15.7 was itself an
artifact: the arena's `out` had been migrated mid-tuning by the CACHED
sibling candidates racing in the same rounds, which is exactly the state
the scored NT run never reaches.  (My r1 pick istream+pfw scored 34.4:
a 1.1 s process samples DURING the migration transient, which is worse
than either steady state — setup length decides which regime the samples
see.)

### What changed (serial kernels untouched, arithmetic identical)

1. **NUMA topology awareness** (`/sys/devices/system/node/node*/cpulist`
   parsed once, thread slot -> node map via the create()-time discovery
   region; any parse failure degrades to mt_r1 behavior exactly).
2. **scratch+NT is pick-ineligible for any candidate whose team spans >1
   node** — still timed, still on the probe string, never installable.
   Single-node teams (all of wallaby-close; t16 on the node) keep NT fully
   raceable: the -24%-at-streaming-batch wallaby result stands there.
3. **create() dwells to ~2.4 s** (running the picked config on the arena)
   when the team spans >1 node and the batch streams past ~1.5x the
   combined LLC, so the scored process is past the balancer's scan delay
   when the driver's warmup starts and the migration transient is spent in
   warmup/calibration, not in the timed samples.  Setup is excluded from
   the score; rivals already sit at 1.6-5 s.
4. **nt-adapt** (execute-side, installed only for a multi-node-team VP pick
   with cached mode at batch >= ~6x combined LLC): each execute until the
   flip, query `move_pages(pid=0, nodes=NULL)` — placement QUERY only, no
   page is moved by me — on the mid-chunk `out` pages of up to 4
   remote-socket threads; once all report their owner's node, switch the
   final stores to scratch+NT, sticky.  Post-migration NT is all-local and
   is the lower-traffic shape (1.5 vs 2.2 MB/vol); pre-migration it is the
   UPI trap.  If the OS never migrates, the flag never flips and the
   cached pick simply runs — strictly no worse than mixedradix's shape.
   NT and the cached modes are bit-identical in output (same z-first
   arithmetic, stores differ only in temporal hint), so the flip cannot
   violate the repeatability contract.
5. **Pool discipline (B=1)**: barriers are now TEAM-scoped — a worker
   outside the job's team posts one flag and stands aside (idle-post-once,
   shape from L36_pfa mt_r1), the mid-barrier scan covers team members
   only, and after the pick the pool is **shrunk to the picked team**
   (destroy+recreate; L17_winograd/L36_pfa discipline).  mt_r1 kept 31
   workers in every barrier under the node's team-16/18 pick: two
   full-width cross-socket flag scans per execute plus 15 phantom
   spinners.  Team ladder extended {8,16,18,32} -> {8,16,18,20,24,32}.
6. **Pool-VP (strat 4, batch 2..64)**: the volume-parallel body can now run
   on the spin pool — one dispatch, zero internal barriers, one completion
   scan — deleting the OMP region cost from the B=32 cell.  Raced against
   the OMP twin, never assumed.
7. Probe string gains `pl=` (best pool-VP) at batch; desc says `+ntad`
   when nt-adapt is installed.  `FFT36PF_NODWELL` skips the dwell for dev
   loops; `FFT36PF_FORCE_MODE` bypasses the NT eligibility filter so the
   monitor can still force a remote-NT A/B.

### Operation count

Unchanged: 232 FMA-port + 57 shuffle vector ops per 36-point line, 225 504
FMA-port vector ops/volume at PW=4.  nt-adapt adds one move_pages query
syscall (~1-2 us) per execute until the flip, then zero; pool-VP replaces
one OMP region with one epoch store + one flag scan.  No arithmetic or
traffic change at fixed mode; the whole round is about WHICH mode runs and
what the pages under it are doing.

### What was measured

wallaby close/cores (single-node team — the mt_r1 regime, must not regress):

| cell | mt_r1 | mt_r2 | pick |
|---|---|---|---|
| B=1   | 13.0 us | **13.0** (15.7 in a busier window) | pool tp32 y-first pw4 (unchanged) |
| B=32  | 108.2 us/call | **101.9 us/call = 3.18 us/vol** | **volpar-POOL istream0 t32 pw4** — pool-VP 3.21 vs OMP 3.45 in-arena, the region cost is real |
| B=512 | 3638-3894 us/call | 3852-3894 (same window spread) | volpar scratch+NT t32 pw4 (unchanged — single-node team keeps NT) |

Pool-discipline A/B, wallaby close, forced strat=3 team=16 (the node's B=1
regime), interleaved 3 rounds, same window: mt_r1 pool **20.7 us**, mt_r2
pool **17.0-17.3 us** — the team-scoped barrier + shrink is worth **-18%**
whenever the pick is a sub-full team.  Forced team=32 twin: 12.9-13.5 both
versions (parity; at full team the changes are no-ops by construction).

wallaby with OMP_PROC_BIND=**spread** (32 threads over both sockets = the
node's close/cores geometry; numa_balancing=1 on wallaby too — used as the
only available two-socket testbed for the new machinery):

* B=512: tuner excludes NT from the pick (elig rule fires), installs
  **istream+pfw t32 +ntad**, dwell runs setup to 2.4 s.  Short run (4
  samples, pre-migration): 11.4 us/vol.  Full-length run (12 samples,
  warmup 5): **min 4.27 us/vol** — the probe flipped to NT after the
  balancer migrated the buffers, beating the same window's in-arena pfw
  (4.85) and the wallaby-close NT pick (7.5).  The full chain
  (migration -> probe -> flip -> NT-local) works end-to-end and is
  bit-repeatable across processes.
* B=32 spread: picks istream0 t32 (OMP twin won in-arena there, pl=3.15 vs
  t32=2.84 — under spread binding the pool's cross-socket completion scan
  eats its dispatch advantage; the race prices it per machine).
* B=1 spread: pool tp32, PASS, repeatable (spread halves are the wrong
  geometry for B=1; close is what the node runs).

Correctness: PASS rel_l2 = 3.748e-16 (B=1 wallaby, y-first class) /
3.575-3.591e-16 (B=2,8,32,512, z-first class), tol 1e-12; bit-identical
re-runs on every cell above including both nt-adapt runs; wombat
AVX2-only end-to-end B=1 (32.9 us) and B=32 PASS+repeatable; bare -O2
without -fopenmp and -march=haswell clean compiles.

### What did NOT work / negatives worth keeping

1. **Pool-VP under spread binding at B=32**: in-arena 3.15 vs the OMP
   region's 2.84 us/vol — the pool's flat completion scan pays ~31
   cross-socket line reads where libgomp's tree barrier pays log-depth.
   Under close binding the pool wins (3.21 vs 3.45).  Both stay raced; if
   the node's close/cores map spans sockets at T=32, the OMP twin may
   keep B=32 — the tuner decides, not me.
2. **Reading the mt_r1 arena numbers as placement-clean**: nt=15.7
   in-arena was produced by an arena whose `out` had already been migrated
   by cached sibling candidates in the same interleaved rounds.  A
   serially-filled arena is necessary but NOT sufficient once the process
   lives past the balancer's scan delay — every in-arena NT-vs-cached
   verdict on a >1 s-old multi-socket process is suspect.  This round's
   answer is to stop asking the arena that question (deterministic elig
   rule + runtime placement probe) rather than to try to build an
   unmigratable arena.
3. Not a kill but a caution: my r1 istream+pfw scored 34.4 on the node —
   sampling DURING the migration transient is worse than either steady
   state.  That is what the dwell is for; if the node's balancer is off,
   the dwell wastes 2 s of unscored setup and nothing else.

### Borrowed (attribution)

* **Pool shrink after pick** and idle-workers-post-once / team-scoped
  scans: **L36_pfa mt_r1** (their mt_pool_run scans `tuse` per phase and
  their J_QUIT shrink; my mt_r1 already credited the pool itself to
  L17_winograd).
* **The migration diagnosis** is built directly on **L36_mixedradix
  mt_r1**'s numbers (their 3 s setup + vol32-pfw runs are the natural
  experiment: 23.0 pre-migration vs 14.2-14.5 post) and their serial-
  arena-fill note; their unshipped move_pages(2) idea is related but
  distinct — I QUERY placement and let the OS move pages, they proposed
  moving pages outright and deferred it as spirit-questionable.
* **NT-loses-on-the-node-at-t32** corroborated by L36_pfa mt_r1 (nt16
  20.1-20.5 beating nt32 23.7-26.3 at B=512) and L13_rader (nt! 1177 vs
  pw! 428 at B=512).
* Team ladder middle points (20, 24): prompted by the node picking 16/18
  from a ladder with nothing between 18 and 32.

### Predictions for the node (stated so they can be scored)

* B=512: pick **istream+pfw or istream0 t32 pw4 +ntad**, setup ~2.5 s.
  If the node's balancer behaves like wallaby's: **10-14 us/vol** (14.2 is
  mixedradix's cached-only ceiling; the NT flip should cut below it).  If
  balancing is off on the node: ~18-22 (arena-like cached numbers), still
  better than mt_r1's 24.5.  A scored run at ~24 with desc showing +ntad
  and NO improvement over r1 would mean migration never happened AND
  cached t32 is UPI-bound — then nt16 (pfa's 19.5 shape) is the fallback
  to force next round.
* B=1: pool tp at some team in {16,18,20,24}; the -18% A/B at team 16
  says **~23-25 us** against mt_r1's 28.5.  If it lands >27, the barrier
  was not the cost and the next lever is pfa's socket-staged x-pass.
* B=32: volpar-pool istream0 t32 if the node's region cost matches
  wallaby-close (**~5.3-5.5 us/vol**), else the OMP twin at mt_r1's 5.64.

### Next

1. Read the node's B=512 desc first: `+ntad` present + t32/nt probe pair
   tells whether the flip fired; compare r1 vs r2/r3 per-run minima for
   the migration signature (buffers are per-process, so every run pays
   its own transient — the dwell should make all three uniform).
2. If B=1 lands >= 27: build pfa's two-stage x-pass (socket-local DFT4
   partials, only the 4x9 intermediate crosses the UPI) — the ~7 us
   coherence term is then the only thing left.
3. If pool-VP loses B=32 on the node the way it lost under spread: try a
   tree completion scan (thread 16 aggregates socket 1's flags) before
   giving up on the pool there.
4. B=2..8 still TP-OMP; unscored, unchanged.

## Round mt_r3 — the NT-eligibility rule was backwards; streaming cells are pinned, not raced; the fr instrument ships

### Where mt_r2 landed on the node, and the reversal this round is built on

Node (Gold 5218, p55n3): B=1 **25.87** (3rd by 0.05 behind L36_pfa; mixedradix
23.01), B=32 **5.52 µs/vol** (3rd; pick lottery — my three processes installed
istream0/pfw/pfw and read 176.5/199.4/199.8 µs/call), B=512 **34.04 µs/vol,
stable, a 1.39× regression on mt_r1** (pick `volpar t32 pw2 istream+pfw+ntad`,
probe t32=15.7 — the verdict names my 2.2× arena mis-pricing at that cell,
twice in the same direction, as "an instrument that cannot see the regime it
is supposed to price").

The reversal: mt_r2's central mechanism — "NT stores never fault, so under
scratch+NT `out` stays on the driver's socket forever" — is **wrong at the
kernel level**. Automatic NUMA balancing migrates on PROT_NONE *protection*
faults, which an NT store takes exactly like a cached store; the store's
non-temporal hint is a cache property, not a page-table one. The node data
that should have said so: L36_mixedradix's winning B=512 process ran
`vol32-sntp` — full-team scratch+NT+read-pf, the exact shape my eligibility
rule made pick-ineligible — at **9.99 µs/vol** (150 GB/s aggregate; their two
losing processes picked vol16 at 19.3, a pick lottery, not a shape failure).
Meanwhile every cached-t32 shape in the round (L36_pfa's ip7 31.6, my
pfw+ntad 34.0) sat at one socket's bandwidth: the migration transient is
longer than the timed run unless the process is old enough, and cached
in-place carries an RFO+writeback penalty on top. And this round I *measured*
the premise dead (below): under NT stores on a two-socket team, 50% of the
sampled `out` pages migrate off their first-touch node.

### What changed (serial kernels untouched; arithmetic identical)

1. **NT-eligibility rule and nt-adapt DELETED.** Both mt_r2 mechanisms
   removed outright — the machinery, the probe, the `+ntad` desc tag.
2. **Streaming cells are PINNED** (working set 2·B·729 KB > 4× team-aggregate
   LLC; node B=512 fires, B=32 does not; wallaby B=512 fires too, so the dev
   loop exercises the path): the pick is **full-team volume-parallel
   scratch+NT with the paced read cursor (mode 2)** and only the SIMD width
   is raced. The cached full-team row and a half-team NT row are still timed
   for the probe string, never installable. This is the verdict §6 rule
   ("wide-team incumbent at any cell whose working set exceeds aggregate
   L3"), plus mixedradix's winning shape adopted wholesale, plus the verdict's
   §4.3 construction in spirit: one 729 KB volume through a 1 MiB-L2-resident
   scratch tile, all three axes inside it, in + NT-out compulsory traffic
   only (1.5 MB/vol).
3. **Dwell 2.4 → 4.0 s.** Setup-length evidence across rounds: mixedradix's
   winning B=512 processes sat at 3.0 s (r1, 14.2) and 4.7 s (r2, 9.99); my
   0.52 s (r1, 24.5), my 2.41 s (r2, 34.0) and pfa's 0.92 s (r2, 31.6) all
   lost. The dwell also keeps the balancer's per-task scan machinery active
   (remote faults on the arena) when the driver's warmup begins.
4. **Placement governor** (ported from L8_fusedaxes mt_r2, attributed): at
   streaming cells execute() samples 32 pages of the caller's `in` and 32 of
   `out` with move_pages(2) (query only, nodes=NULL — the mt_r1-legal read)
   on call 1 and every 16th call, and publishes
   `gov{nb=<numa_balancing> fi=<%in-moved> fo=<%out-moved> nc=<calls>}` on
   the description string. This is the exact experiment the mt_r2 verdict
   names as the highest-value next measurement (fr under a wide team at a
   streaming cell). Instrument only; ~30 µs per 16×17 ms of calls.
5. **B=32 class: mode 7 (istream + paced read cursor, no prefetchw) joins
   the race; mode 8 (istream+pfw) leaves it.** Node driver evidence: my pfw
   pick scored 199.4/199.8 µs/call in two processes against istream0's 176.5
   in the third, while the arena priced them a tie (6.2 vs 6.3) — prefetchw
   on semi-cached, cross-socket lines is a tax the arena cannot see. Mode 7
   is the analog of mixedradix's `pf1`, which holds the cell record (166.8).
   Simplest-wins order now 0 < 12 < 7 < 2, so pfw cannot return by luck.
6. **B=1: leftover-plane pair-split (strat 5)**, adopted from L36_mixedradix
   mt_r2 (who built L36_pfa mt_r1's idea (b)): at T=32 (and 24), thread t
   owns plane t; each of the R = 36−T leftover planes is split between
   threads 2i/2i+1 — disjoint halves of pass-A subloop 1 (cut on a PW-group
   boundary, 5/4 groups at PW=4, 9/9 at PW=2, mixedradix's exact cuts), one
   2-thread flag handshake, disjoint halves of subloop 2. Pass-A span 2.0 →
   1.5 waves. Both bit classes implemented (y-first and z-first halves);
   output bit-identical to the plain split by construction. Raced on the
   pool, ranked most-complex-last in the hysteresis (must win by >2%).

### Operation count

Unchanged: 232 FMA-port + 57 shuffle vector ops per 36-point line over PW
lanes, 225 504 FMA-port vector ops/volume at PW=4. The governor adds one
64-page move_pages query per 16 executes (~2 µs amortized per 17 ms call);
pair-split adds one 2-thread flag handshake and ~20 KB of same-socket
cross-core staging per shared plane, zero arithmetic.

### What was measured (wallaby Gold 6448Y, 32 threads, shared login node;
### driver minima; rel_l2 vs numpy, tol 1e-12; all repeatable/bit-identical)

| cell | mt_r2 | mt_r3 | pick |
|---|---|---|---|
| B=1 | 13.0 µs | **13.07–13.5 µs** | 2phase-pool t32 y-first pw4 (unchanged) |
| B=2 | — | 29.2 µs/call | TP-OMP (unchanged path) |
| B=32 | 101.9 µs/call | **96.4 µs/call = 3.01 µs/vol** | volpar-pool istream0 t32 pw4 |
| B=512 close | 3852–3894 µs | **3712–3788 µs = 7.25 µs/vol** | volpar-**PIN** t32 pw4 scratch+nt |
| B=512 spread | 4.27 µs/vol (ntad flip) | **2406.6 µs/call = 4.70 µs/vol**, sd 0.66% | volpar-PIN t32 pw4 scratch+nt, setup 4.0 s |

* **The spread run is the round's proof.** 32 threads over both wallaby
  sockets (the node's geometry): the pinned NT shape sustains 319 GB/s
  aggregate (1.5 MB/vol · 512 / 2.41 ms) with `gov{nb=1 fi=15 fo=50 nc=64}`
  — **50% of the sampled NT-written `out` pages migrated off their
  first-touch node**. That is the mt_r2 premise falsified by direct
  measurement, on the same instrument the monitor asked for, and it beats
  mt_r2's nt-adapt spread result (4.27) without any adaptive machinery.
* Correctness: 3.748e-16 (B=1 wallaby y-first), 3.575–3.591e-16 (B=2/8/32/512
  and forced z-first B=1), all PASS, repeatable across processes. Wombat
  AVX2-only end-to-end: B=1 32.6 µs (z-first gate fires), B=32 483 µs, PASS,
  repeatable. Clean compiles: bare -O2 without -fopenmp, -march=haswell.
* Pair-split A/B (same process, wallaby, B=1 pw4): plain pool t32 13.41–13.37
  vs pair-split t32 **13.96–15.41 µs/vol in-arena (−4%, it LOSES on
  wallaby)** — the ~20 KB cross-core staging through the shared plane buffer
  eats the 0.5-wave saving when a plane costs only ~1.3 µs. On the node a
  plane costs ~2.2× more and the barrier regime differs (it picked t16/18,
  where span is what t32 must beat), so the rows stay raced; mixedradix's
  23.0 says the shape can pay there. FFT36PF_FORCE_YF now overrides the bit
  class at candidate generation so both halves' classes are testable.

### What did NOT work / bugs worth recording

1. **Staging the pair-split shared plane through the even thread's `pp`**:
   admission failed every strat-5 candidate — the partner may still be
   mid-own-plane in that same buffer, a write-after-read race my first build
   had. Fix: stage through the even thread's `tmid` slot, which is idle in
   every TP shape (no extra sync needed). If you split a plane between two
   threads, the shared buffer must be one neither thread's own work touches.
2. Pair-split on wallaby: −4% (numbers above) — kept raced for the node,
   behind a >2% hysteresis, not installed by default anywhere.
3. Not re-tried on the negatives already priced by others: fused3-style
   3-barrier splits (pfa r1: 17.1–18.6 vs 13.7), dynamic scheduling (dyn≠0
   lost in every mt_r2 process that picked it — verdict §6), and re-racing
   T=16 at streaming cells (the entire §3.4/§3.5 story).

### Borrowed, and from whom

* **Full-team scratch+NT pin at streaming cells**: L36_mixedradix mt_r2's
  `vol32-sntp` (9.99 µs/vol) + the mt_r2 VERDICT §5/§6 wide-team-incumbent
  rule. The pin (vs race-with-margin) follows L6_pfa's verdict item: stop
  racing team width at streaming cells.
* **fr placement scan**: L8_fusedaxes mt_r2's governor, reimplemented on my
  existing move_pages plumbing; their fr=0-under-T16 measurement is also why
  the pin is full-team.
* **Leftover-plane pair-split**: L36_mixedradix mt_r2 (design, cuts, and the
  share-into-partner's-scratch idea), transitively L36_pfa mt_r1 idea (b).
* **Longer dwell**: the setup-length natural experiment across
  L36_mixedradix mt_r1/mt_r2's winning processes.
* Mode 7's promotion rests on mixedradix's pf1 win at B=32.

### Predictions for the node (stated so they can be scored)

* B=512: pick is deterministic (`volpar-PIN team=32 pw=? scratch+nt`,
  setup ~4 s, gov on desc). If the balancer migrates as it did for
  mixedradix's winning process: **10–14 µs/vol** and `fo` well above 0. If
  placement never moves (fo≈0, the L8 regime): **20–25** (mt_r1's NT number,
  minus the pw2 tax) — still better than mt_r2's 34.0 in either regime, and
  the gov string finally tells the panel which regime the node is in at
  32 threads. All three processes now run the same shape, so the min and the
  median should agree for the first time at this cell.
* B=32: istream0 or istream(7) t32 on the pool, **5.2–5.6 µs/vol**; the
  199-µs pfw mode can no longer install, so the worst process should improve
  even if the best does not.
* B=1: pool tp z-first at T∈{16,18,20,24,32}, **~24–26 µs**; if the node
  prices pair-split above the plain t32 rows AND above t16/18, then
  23–24.5 (mixedradix's regime). ps= rides the probe string either way.

### Next

1. Read gov{fi,fo} off the node's B=512 desc first — it decides everything
   downstream. If fo≈0 AND the cell lands ~20+: the node pins memory harder
   than wallaby-spread, and the next lever is a socket-split NT schedule
   (socket-0 threads take the out-half that stays local, socket-1 threads
   take in-side work) rather than any tuner change.
2. If B=1 pair-split loses on the node too, delete strat 5 next round
   (r10 prune rule) and put the effort into pfa's socket-staged x-pass — the
   coherence term, which no split-geometry change addresses.
3. B=32's remaining 6% to mixedradix (166.8 vs 176.5) is likely their pf1
   read-cursor shape; mode 7 is now in the race to answer that. If the node
   picks 7 and lands ≤170 µs/call, promote it into the B≤8 lists too.

## Round mt_r4 — take the two structural deltas the r3 leaderboard names: mixedradix's ncw read-flow into my pinned NT drain, and their sub-socket exact-divisor teams into my B=1 ladder

### Where mt_r3 landed on the node, and what this round is built on

Node (Gold 5218, p55n3, min of 3 processes, all picks stable 3-of-3 for the
first time at every cell):

* B=1 **25.82** (3rd; mixedradix split12 **23.03**, pfa fused2-t16 25.72).
  My pick: 2phase-pool team=16 z-first pw4, probe pl32=32.4–33.1
  pl16=28.8–30.2 ps=29.6–29.8 ser=160.9–161.5.
* B=32 **5.49 µs/vol** (3rd; pfa ip0 **5.21**, mixedradix 5.36).  Pick
  volpar-pool istream(7) t32 in all three processes — the r3 lottery fix
  held; the residual gap is not a pick problem any more.
* B=512 **10.86 µs/vol** (2nd; mixedradix vol32-sntp **9.90**, pfa 19.32).
  Pick volpar-PIN t32 pw4 scratch+nt, setup 4.01 s, gov{nb=1 fi=25–28
  **fo=50** nc=16–80} — the r3 pin + dwell worked exactly as designed
  (34.0 → 10.86, 3-of-3, and the governor measured 50% of sampled NT-written
  out pages migrated, killing the r2 premise a second time).  The VERDICT
  (§4.3, §6) credits this entry with building the L2-tile construction and
  sets the L=36 order: "find the difference between 9.9 and 20.0 µs/vol."

The diagnosis this round acts on: at B=512 mixedradix and I now run the SAME
nominal shape (full-team, per-thread L2-resident volume scratch, NT stream to
out, ~1.5 MB/vol compulsory) and differ 9%.  Reading their exemplar source
(`exemplars/mt_r3/L36_mixedradix.c`, body()) against mine leaves exactly one
structural difference in the winning path: their phase 2 — the NT drain —
issues a small paced read flow into the NEXT volume's input (their `ncw`
block: 3 T1 line-prefetches per phase-2 group, 324 × 3 × 64 B = 62 KB of
in[b+1]), so the DRAM system never sits in a pure-write regime between
volumes.  My mode 2 issued NO reads at all during the drain.  (I checked
their anti-alias `pind` slide too, and did NOT take it: their own phase-1 r8
node data prices it 0 to −1.2% and it ships default-off.)  At B=1 the delta
is decomposition: their winning split12 gives every thread exactly 3 pass-A
planes and 27 pass-B units on ONE socket; my ladder {8,16,18,20,24,32} had no
sub-16 exact divisor of 36 at all — 16 splits 36 as 2.25 (span 3 with 12
threads idle a wave) and everything ≥18 crosses the node's socket boundary.

### What changed (serial kernel arithmetic untouched since panel_r11; both
### changes are prefetch shape / team decomposition, bit-identical by class)

1. **Mode 2 (the pinned streaming shape) gains light next-volume pre-coverage
   in the NT drain**: passB_nt takes a lines-per-unit knob; mode 2 passes 3
   (T1 hint, 62 KB of in[v+1] per volume — L36_mixedradix's ncw, adopted
   verbatim from their node-winning sntp body and attributed inline); mode 3
   keeps its old 36 (T2, the whole volume — the phase-1-rejected heavy XV,
   now reachable only by forcing).  Desc marker: `scratch+ntx` (vs
   `scratch+nt` when compiled with -DFFT36PF_NONXT, the A/B control).
   Prefetch-only: output bit-identical (verified: ntx and nonxt binaries
   produce cmp-identical out.bin).
2. **B=1 pool ladder gains 12 and 9** — the exact divisors of 36 planes and
   324 units that stay on one node socket.  From L36_mixedradix's split12
   (node 23.0) and their ladder note ("bracket 12 from below with the other
   exact divisor" — their T=9 = 4 planes + 36 units exact).  pl12= and pl9=
   ride the B=1 probe string so the node prices the sub-socket curve even if
   the pick goes elsewhere.  TP-OMP ladder gains 12 for symmetry.
3. **Pair-split rows (strat 5) pruned**, per this record's own r3 rule: on
   the node it beat plain t32 (29.7 vs 33.1 in-arena) but lost the pick to
   sub-socket t16 — the mechanism it fixes (leftover-plane imbalance at wide
   teams) is now covered by the exact-divisor teams with zero handshakes.
   Code path kept behind FFT36PF_KEEP_PS; ps= leaves the probe string.

### Operation count

Unchanged: 232 FMA-port + 57 shuffle vector ops per 36-point line over PW
lanes, 225 504 FMA-port vector ops/volume at PW=4.  The light ncw adds 972
prefetch uops per volume (324 units × 3) and zero arithmetic, zero extra
compulsory traffic (it touches only bytes pass A was about to read anyway).
Team-ladder and prune changes are plan-time only.

### What was measured (wallaby Gold 6448Y, 32 threads, shared login node;
### driver minima; rel_l2 vs numpy, tol 1e-12; all PASS and bit-repeatable)

| cell | mt_r3 | mt_r4 | pick |
|---|---|---|---|
| B=1 | 13.07–13.5 µs | **13.39 µs** | 2phase-pool t32 y-first pw4 (unchanged — wallaby is one socket, wide wins there) |
| B=32 | 96.4 µs/call = 3.01 µs/vol | **95.76 µs/call = 2.99 µs/vol** | volpar-pool istream t32 pw4 (unchanged) |
| B=512 close | 3712–3788 µs = 7.25 µs/vol | **3697.7 µs = 7.22 µs/vol** | volpar-PIN t32 pw4 scratch+**ntx** |
| B=512 spread (node geometry) | 2406.6 µs = 4.70 µs/vol | **2339.5–2420.1 µs = 4.57–4.73 µs/vol** | same pin, setup 4.0 s, dwell fires |

* **The ncw A/B is not priceable on wallaby** — stated plainly rather than
  oversold.  Same-window interleaved ntx-vs-nonxt at B=512 spread: 2392.6 /
  2417.1 vs 2356.7 / 2407.5 µs (a ~1% wash, plus one 4761-µs round poisoned
  by login-node interference); close binding 3697.7 vs 3712.6–3744.2
  (ntx ≤1% ahead, same noise band).  Wallaby's SPR memory system (DDR5, 2 MB
  L2, stronger streamers) is exactly where a 62 KB read-flow-during-drain
  should vanish; the node's DDR4-2666 with its costlier read/write turnaround
  is where mixedradix's 9.90 says it pays.  The node run is the experiment;
  scratch+ntx vs the r3 baseline 10.86 is the readout, and FFT36PF_NONXT is
  the control if the monitor wants it.
* B=1 in-arena pool team curve, wallaby pw4 (new rows in context): t8 23.15,
  **t9 19.87, t12 17.15**, t16 18.00, t18 15.23, t20 15.44, t24 14.32, t32
  13.56 — t12 beats t16 by 5% even on ONE socket (the 2.25-plane imbalance
  is real and machine-independent); the ladder's wide end still wins where
  there is no UPI, exactly as it should.
* Correctness: 3.748e-16 (B=1 wallaby y-first class), 3.586–3.591e-16
  (B=32/512 z-first class), repeatable (bit-identical across processes) at
  every cell; wombat AVX2-only end-to-end B=1 32.66 µs, B=32 487.4 µs, PASS,
  no regression vs r3 (32.6 / 483).  Clean compiles: bare -O2 without
  -fopenmp, -march=haswell.

### What did NOT work / negatives worth keeping

1. **Wallaby cannot price the ncw** (numbers above) — this round ships it on
   rival node evidence, not dev-box evidence.  If the node reads
   scratch+ntx ≈ 10.9 (no change), the mechanism is not the mixedradix
   delta and the remaining 9% is ENGINE: their node serial is ~6.5% faster
   than mine (ser=150.7–152.1 vs my 160.9–161.5 in the r3 B=1 descs) on a
   kernel that is FASTER than theirs on wallaby — a CLX-vs-SPR scheduling
   gap (single 512-bit FMA pipe, half-size DSB), not a memory one.
2. **Considered and rejected: racing mode 4 (PIPE ping-pong) under the pin**
   as the read-overlap vehicle — its live mid set is 2 volumes = 1.46 MB
   per thread, which blows the node's 1 MiB L2 and, at 16 threads/socket,
   the 22 MiB socket L3 too (16 × 1.46 = 23.4 MB), converting scratch
   traffic into DRAM traffic at exactly the cell that is DRAM-bound.  The
   62 KB ncw buys the same read-during-drain property for 972 prefetches.
3. **Considered and rejected: mixedradix's pind anti-alias slide** — their
   own record ships it default-off after phase-1 r8 priced always-on at 0
   to −1.2% at B=1.  Not rediscovered; noted so nobody else spends a round
   on it either.
4. Pair-split pruned on its pre-registered criterion (r3 Next item 2); the
   negative that kills it is r3's own node probe (ps lost the pick to a
   smaller plain team), not a new measurement.

### Borrowed, and from whom (also credited inline in the source)

* **Light next-volume pre-coverage in the NT drain (ncw, 3 lines/unit, T1)**:
  L36_mixedradix's node-winning B=512 sntp body, `exemplars/mt_r3/
  L36_mixedradix.c`, taken verbatim including the hint level and the 3-line
  weight.  Transitively their pfin lineage runs through L36_pfa's PFIN.
* **T=12 and T=9 exact-divisor sub-socket teams**: L36_mixedradix's split12
  (the standing B=1 cell winner) and their T=9 ladder rationale, quoted in
  the source comment.  L36_pfa's r3 t12 numbers (29.5–30.5 in-arena, did
  NOT transfer to their fused2 structure) are the caution that made me keep
  16/18 in the ladder rather than swap.
* The prune discipline is phase-1 r10's rule, applied to my own strat 5.

### Predictions for the node (pre-registered, so mt_r4's verdict can grade)

* B=512: pick is deterministic (`volpar-PIN team=32 pw=? scratch+ntx`,
  setup ~4 s, gov on desc).  If the ncw read-flow is the mixedradix delta:
  **9.9–10.4 µs/vol** (their 9.90 is the existence proof of the shape's
  ceiling on this node).  If it lands **10.6–11.0** (r3-flat), the delta is
  engine, not schedule — then next round stops touching the memory system
  at this cell and ports kernel scheduling instead (their DFT layering or
  the zy-style port-5 interleave, measured on the node's own serial row).
  Either way fo≈50 should reappear; fo≈0 with a slow run would mean the
  balancer regime changed under me.
* B=1: pool tp z-first at **t12** if mixedradix's decomposition transfers to
  my kernels (**~23.5–25**), else t16 again at ~25.8 with pl12/pl9 published
  for the record.  pl12 < pl16 on wallaby's single socket (17.15 vs 18.00)
  is weak supporting evidence; the node's sub-socket coherence regime is
  what actually decides it.
* B=32: unchanged path, **5.3–5.6 µs/vol**, 3-of-3 on one pick.  No change
  was made at this cell on purpose: the 5% to pfa's ip0 is the same engine
  gap as the serial 6.5%, and no tuner change addresses it.

### Next

1. Read the node's B=512 number against 10.86 first — it adjudicates
   schedule-vs-engine for the whole geometry (see prediction).  If ntx won,
   sweep the ncw weight (3 → 6 lines) before anything else; if it tied,
   open the engine file: CLX-side kernel scheduling (DSB-resident pass-A
   bodies, the mode-11 compact-twin trick applied to the z-first class, or
   mixedradix's zy interleave) is the only lever left at every cell.
2. Read pl12/pl9/pl16 off the node's B=1 desc.  If t12 wins and lands ≤24,
   the remaining ~1 µs to mixedradix is engine (same conclusion as item 1).
   If t16 still wins despite pl12 < pl16 in-arena, the arena is mispricing
   the B=1 cell too and the fix is an execute-time team race on the real
   buffers (L8_fusedaxes' protocol, already ported by pfa).
3. B=2..8 remain TP-OMP, unscored, unchanged.
