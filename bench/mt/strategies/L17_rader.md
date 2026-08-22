# L17_rader — multicore strategy record

Phase-1 history (how the kernel itself got to its current form) is in
`../geom/strategies/L17_rader.md`; this file records only the multicore
phase.  Phase-1 node baselines to measure parallel efficiency against
(single-thread, Cascade Lake, panel_r9 leaderboard): B=1 16.54 us,
B=8 18.06 us, B=256 24.61 us, B=2048 24.98 us per transform.

## Round mt_r1

### What was built

The phase-1 kernel (Rader-17 in cyclic/negacyclic form, 296 FP instr per
17-point transform, 867 transforms = ~423 kflop per volume — arithmetic
unchanged this round) got a multicore layer with two parallel modes plus the
single-thread fallback, all selected at plan time by timing the REAL
dispatch path on the tuner arena:

1. **mode 1 — volume-parallel (batch >= 2).**  The batch is cut into nt
   contiguous chunks; thread t runs the tuned single-thread exec on its
   chunk with a full per-thread shadow plan whose scratch was allocated and
   first-touched BY thread t inside fft3d_create's one OpenMP region
   (NUMA-local under OMP_PROC_BIND=close).  Contiguous chunks keep each
   thread's slice of `out` first-touched by its own warmup writes — the
   driver never touches `out` before warmup, so out pages land socket-local
   for free.  `in` is fread by the driver's main thread (all socket-0 on
   the node); nothing an implementation can do about that except weight
   the partition (below).

2. **mode 2 — plane-parallel (B=1 and small batches).**  (b,x) plane
   pipelines (deint→z→transpose→y into a shared per-volume A slot) are
   tasks; one team barrier; then (b,blk) x-pass blocks (37/volume) are
   tasks.  Plane regions of A are cache-line-aligned (PS8*8 = 2368 B = 37
   lines exactly), so plane tasks never share a store line; x blocks are
   handed out in contiguous runs so only chunk-boundary out lines are ever
   written by two threads.  Bit-identical to class A by construction (same
   kernel calls, same operands, disjoint writes).

3. **Custom spin-wait pthread pool instead of OpenMP at execute time.**
   Measured on wallaby (SPR, this round): one `#pragma omp parallel` +
   barrier costs **2.7 us at nt=2, 4.4 at nt=4, 8.3 at nt=32** — the entire
   B=1 budget.  The pool (created once in fft3d_create; workers pinned to
   the exact cores the harness's close binding chose, recorded via
   sched_getcpu inside the create-time OMP region) dispatches through one
   release-store + spin handshake and syncs mid-job with a centralized
   sense-reversing barrier.  Workers busy-poll ~20 ms after their last job
   then park on a condvar; a mode-0 plan tears the pool down entirely so a
   single-thread plan leaves 31 idle cores.  With the pool, B=1 went from
   6.9 us (OMP dispatch) to 5.7 us on a quiet wallaby.
   **Do not "optimize" the all-workers-ack completion protocol**: I tried
   having only team members ack (to skip 31-nt stores at small teams) and
   it is RACY — a slow non-team worker can read the NEXT job descriptor
   with a stale sequence number, run it twice, and corrupt the completion
   count for the job after.  All-ack restores the invariant that every
   worker is past its descriptor reads before the next dispatch overwrites
   it.  Reverted before it ever shipped; the fix would need a seqlock
   snapshot of the descriptor, parked for a future round.

4. **Non-temporal flush variants ("stnt", "stnt dy", "stpnt", "stpnt dy",
   L17R_FORCE 17–20).**  Phase 1 rejected NT stores four rounds running —
   single-threaded, prefetchw hid the RFO more cheaply.  With 32 cores
   sharing DRAM the batched cells are bandwidth-bound and the out RFO is a
   third of the traffic, which is exactly where NT pays.  The staged
   x-pass (vo) flush gets an `_mm512_stream_pd` path: short normal-store
   head to reach 64-B alignment (a volume's out base is only 16-B aligned:
   2*NVOL doubles = 78608 B is NOT a line multiple — the unaligned
   vmovntpd segfaulted before the head/tail peel was added), aligned NT
   full lines, normal tail, sfence before the worker's completion store.
   Bit-identical (same values, same places).  Wallaby measured, 32
   threads: at B=4096/nv=1024 (streaming) **stnt dy 0.914 vs plain
   xl 512t dy 1.062 us/t (-14%)**; at B=256 (L3-resident on wallaby's
   60 MB L3) NT rightly LOSES (0.73 vs 0.47 us/t) and the tuner keeps the
   plain variant.  The node's 22 MB L3 makes B=256 stream, so the NT twins
   may take that cell there — the plan-time race decides per machine,
   which is the point.

5. **Measured weighted partition (mode 1).**  Each thread times its own
   chunk every dispatch (two clock_gettime per thread per execute, ~50 ns,
   min across reps); if the equal-chunk spread exceeds 10%, the partition
   is re-cut proportional to measured per-thread rate (clamped to
   [0.5, 2]x mean) and adopted only if re-measurement beats equal chunks
   by >3%.  Motivation is the node's NUMA asymmetry: `in` is entirely
   socket-0 (driver fread), so the remote socket's 16 threads stream input
   over UPI and lag, and equal chunks make every execute as slow as the
   slowest thread.  On wallaby (all 32 threads on one socket) it still
   fired at B=4096 — measured spread 1.44x, arena 0.871 -> 0.749 us/t —
   though the driver-visible gain there was small; the real test is the
   node, where the two-socket imbalance is structural.  Bit-identical
   (changes who computes which volume, not the values).

6. **Tuning = racing the real dispatch.**  Stage 1 races 10 class-A
   variants (xl 512t, dy, st, st dy, stp, stp dy, stnt, stnt dy, stpnt,
   stpnt dy) under full-team contention on a streaming arena
   (nv = min(batch, 1024)); then the L23_rader-style joint (variant, pf,
   pfw) grid over winner + best-of-other-store-shape partner (3% margin to
   leave (0,0)); then a team-size race nt ∈ {32, 24, 16} (3% margin to
   shrink); then for batch < 32 a plane-parallel challenge on the full
   team.  For B=1 the phase-1 single-thread tuner runs first (workers
   parked so the incumbent sees idle-machine turbo — its real conditions
   if it wins and the pool is torn down), then mode 2 is raced at
   nt ∈ {2,4,8,16,17,24,32} with workers spinning (their real conditions).
   The tuner arena's tout is deliberately NOT memset in create any more:
   first touch happens in each candidate's warmup execs by the writing
   thread, matching what the driver's warmup does to `out`.

### Measured (wallaby, Xeon Gold 6448Y SPR, 32 threads on one socket of 64 cores; dev numbers, relative only)

| case | this round | phase-1 single-thread (same host class) | note |
|---|---|---|---|
| B=1 | 5.7–7.5 us (run spread; quiet best 5.72) | ~8.8–9.2 us | mode 2, nt=16–17 wins its race at 1.4–1.6x; sync-bound |
| B=8 | 14.0–14.2 us/batch = 1.77 us/t | ~11.6 us/t | mode 1 or vp per race |
| B=256 | 114–122 us/batch = 0.45–0.48 us/t | ~11.5 us/t | plain xl 512t (wallaby L3-resident; NT loses here) |
| B=4096 | 3933–4390 us/batch = 0.96–1.07 us/t | ~14.4 us/t | stnt dy + weighted cut, nt=32 |

Correctness: rel_l2 ~3.15e-16 at every case, repeatable bit-identical
across runs (tryout cmp).  Batched parallel efficiency on wallaby is
~24x/32 at B=256 and ~14x/32 at B=4096 against the same host's
single-thread times — better than arithmetic scaling at B=256 because the
single-thread kernel was MLP-limited (~10 fill buffers), exactly as the
brief's bandwidth note predicted; B=4096 saturates socket DRAM
(~236 KB/volume at 0.96 us/t ≈ 245 GB/s with the RFO deleted).
B=1 is sync-bound: one volume is 78.6 KB and the plane phase's 17 tasks
cost ~0.5 us each, so the pool handshake + barrier (~1.5–2.5 us total)
caps the speedup near 1.6x.  That is a measured answer to the brief's
"does B=1 parallelise at all": yes, modestly, and only with a sub-3-us
sync path — libgomp's 8 us region alone erases it.

### What did not work, with numbers

* **OpenMP parallel-for dispatch at execute time**: 2.7–8.3 us per
  region+barrier on wallaby (nt=2–32).  B=1 with OMP dispatch: 6.7–6.9 us
  vs 5.7 us with the spin pool.  Replaced, kept only in create.
* **Team-only completion acks in the pool**: correctness race (double-run
  of a job by a stale non-team worker), reverted same session — see item 3.
* **Unaligned NT stores**: segfault; vmovntpd demands 64-B alignment and a
  volume's out base is 16-B aligned for odd volume indices.  Head/tail
  peel fixed it.
* **NT flush at L3-resident batch** (B=256 on wallaby): +55% over plain
  (0.73 vs 0.47 us/t) — NT bypasses a cache that was doing useful work.
  Left to the per-machine race, never forced.
* **Shrinking the team at batch** (nt 24/16 vs 32): 0.42/0.58/0.90 us/t at
  B=256 — nt=32 wins decisively on one wallaby socket; re-race on the node
  where 32 spans two sockets.

### Borrowed

No cross-entry context existed this round (first mt round; context file
empty).  Within my own lineage: the joint (variant,pf,pfw) grid discipline
from L23_rader panel_r8, the L3-scaled arena from L17_matrixsimd/L36_mixedradix,
and the plane pipeline/kernels from phase 1 as-is.

### Next round

1. **Node evidence first**: the description string now reports
   `mt mode/nt`, the B=1 st-vs-vp race, and the batched winner — read the
   leaderboard's description column before touching anything.
2. **B=1 sync path**: a seqlock job descriptor to make team-only acks safe
   (saves ~0.5 us at nt=17), a tree barrier, or merging the x pass into
   plane-completion polling (each x block needs ALL planes, so per-plane
   done flags are equivalent to a barrier — but arrival-order polling could
   start blocks whose rows... no: every block needs every plane; only the
   barrier's implementation can improve, not its existence).
3. **Node NUMA**: check whether the weighted partition fires on the node
   and how big the spread is (expect ~1.5x from UPI-bound input reads on
   socket 1 at B=4096).  If it fires strongly, consider making the x-pass
   NT flush the default partner in the grid there.
4. **Mode 2 at intermediate batches** (8–31): plane tasks scale to
   batch*17, so the barrier amortizes; wallaby raced it against mode 1 per
   plan — verify the node agrees.
5. **If B=1 matters for the score**: the remaining lever is cutting the
   plane-task critical path (deint+z+transpose+y ≈ 0.5 us serial per
   plane); splitting z/y kernel groups across 2 threads per plane doubles
   sync for ~0.25 us of work — probably not worth it; measure before
   believing.

## Round mt_r2

### Where mt_r1 landed on the node, and the diagnosis

Node (2-socket CLX 5218): B=1 9.134 us with a 38.3% run spread (3rd,
matrixsimd 7.112), B=256 0.795 (2nd, matrixsimd 0.726), B=4096 **2.904
against L17_winograd's 1.222** -- 2.4x behind with the same arithmetic
(my kernel IS their 17-point module).  The description string showed the
plan-time arena measured 0.793 us/t for the very config that scored
2.904.  Three causes, all mine:

1. **The r1 arena raced under a NUMA placement the driver never
   provides.**  I built r1 on "the driver never touches `out` before
   warmup, so my threads' first touch distributes it" -- WRONG:
   driver.c:109 memsets `out` on the main thread BEFORE fft3d_create, so
   every caller page (in via fread, out via memset) starts on socket 0.
   My un-memset tout raced candidates with distributed out pages
   (0.793 us/t) and the pick did not transfer (2.904).  L17_winograd and
   L8_fusedaxes had this right all along; L23_matrixsimd mt_r1 stated the
   lesson verbatim ("the tuner arena is deliberately filled serially so
   create()-time racing sees the same placement the scored run does").
2. **The weighted partition froze noise into a skewed cut.**  Its trigger
   fired on SINGLE-SOCKET wallaby ("spread 1.44x" where no structural
   imbalance exists), i.e. per-thread min-tsec rates are transient; the
   adopted cut (chunks clamped to [0.5,2]x mean) then makes every scored
   execute as slow as the biggest chunk, up to ~2x equal-cut.  This is my
   prime suspect for 2.904 vs winograd's equal-cut 1.222.
3. **The pool handshake bounced shared lines through all 31 workers.**
   One shared `go` line invalidated in 31 caches + 31 serialized RFOs on
   one `done` line per dispatch, 15+ of them across UPI on the node at
   any team size -- the standing suspect for B=1's 9.13 us and its 38.3%
   spread (wallaby, single socket, never showed it).

### What was built (all three fixes, plus one new bug found and fixed)

1. **Arena matches the driver** (ADOPTED FROM L23_matrixsimd mt_r1, as
   restated by L8_fusedaxes): tout is now memset serially by the main
   thread in l17r_tune_alloc, tin was already filled serially.  Every
   plan-time race now sees socket-0 caller pages, exactly like the scored
   run.  On wallaby this changed no pick (single socket); the node's
   B=4096 race should now correctly price the NT twins (stnt dy won
   wallaby's honest streaming race at 0.780 vs plain 0.964).
2. **Weighted partition REMOVED; equal static contiguous cut always.**
   The per-thread chunk timing stays as pure telemetry: the description
   now carries "cut=eq spr=N.NNx" so the node itself reports whether a
   structural spread even exists before any future re-attempt.  (Wallaby
   telemetry this round: 1.38x at B=4096 on ONE socket -- confirming the
   r1 trigger was reading noise.)
3. **Per-worker release/done flags** (ADOPTED FROM L17_matrixsimd mt_r1:
   "only the ACTIVE team's flags are touched").  Dispatch release-stores
   only the team's rel flags (own cache line each) and collects only the
   team's fin flags; a never-released worker never reads the job
   descriptor, which closes r1's team-only-ack staleness race BY
   CONSTRUCTION (the r1 revert note stands for the shared-counter design;
   this is the safe per-worker version).  Small teams on the node now
   touch zero remote-socket lines per dispatch.
4. **Flat arrival-flag/release barrier** (ADOPTED FROM L17_winograd
   mt_r1): each mode-2 team member writes its OWN padded arrival line,
   rank 0 scans them (misses overlap) and publishes one release word;
   epoch = the dispatch generation, so threads that sat out a tuning
   dispatch can never be out of phase.  Replaces the central
   fetch_add sense barrier (winograd measured ~0.3-0.4 us vs ~1.2 us at
   nt=16 for the central shape; matrixsimd saw 3.5 us at nt=32).
5. **Bug found while measuring: thundering herd on parked workers.**
   With per-worker release, non-team workers park FOREVER (nothing ever
   bumps their flag), so the r1-style global "anyone parked -> broadcast"
   woke ~30 sleepers on EVERY small-team dispatch just to re-sleep:
   measured B=1 vp nt=2 at **80 us** against 7 us steady state.  Fix: a
   per-worker parked flag, written under the mutex with the same
   seq_cst Dekker pattern as r1's nparked; dispatch scans only the
   team's flags and broadcasts once if any is set.  After the fix the
   same race read nt=2 14.9 us and the B=1 pick landed at nt=17/5.71.

Operation count: unchanged (296 FP instr per 17-point transform, 867
transforms = ~423 kflop per volume; the round touched only dispatch,
placement, and tuning).  Bit-identical across all modes as before
(tryout cmp: repeatable at every batch tested).

### Measured (wallaby, SPR 6448Y, 32 threads, shared login node)

| case | mt_r2 | mt_r1 same host | note |
|---|---|---|---|
| B=1 | **4.98-5.19 us, sd 0.2-0.3%** | 5.7-7.5 (quiet best 5.72) | mode 2, race kept nt=17 (5.71 in-plan); ~9% faster than r1's best and far more stable |
| B=8 | 14.04 us/call = 1.75 us/t | 14.0-14.2 | unchanged |
| B=256 | 122.9 us/call = 0.48 us/t | 114-122 | plain xl 512t dy, nt=32 (NT rightly loses in wallaby's 60 MB L3) |
| B=4096 | 4209 us/call = 1.028 us/t | 3933-4390 | stnt dy nt=32 wins the honest race 0.780 vs plain 0.964 in-arena |

Correctness: rel_l2 3.11-3.18e-16 at every batch tested (1, 2, 8, 31,
64, 256, 4096), repeatable bit-identical across runs.  Wallaby cannot
show the NUMA fixes (its 32 threads sit on one socket); the node is the
real test, and every change this round is either node-targeted (1, 2, 3)
or measured better on wallaby too (4, 5).

### What did not work, with numbers

* **Global "anyone parked" wake check with per-worker release flags**:
  B=1 vp nt=2 80.2 us, nt=4 56.8, nt=8 40.1 -- every dispatch broadcast-
  woke ~30 permanently-parked non-team workers.  Per-worker parked flags
  fixed it same session (see item 5).
* Nothing else was tried and rejected this round; the round was
  deliberately three borrowed, already-proven mechanisms plus the
  removal of my own two mt_r1 mistakes.

### Borrowed

* Serial memset of the tuner arena's output buffer -- L23_matrixsimd
  mt_r1 (via L8_fusedaxes's restatement).
* Per-worker release/done handshake -- L17_matrixsimd mt_r1.
* Flat arrival-flag/release barrier with dispatch-epoch sequencing --
  L17_winograd mt_r1.

### Next round

1. **Read the node's new telemetry first**: "cut=eq spr=N.NNx" says
   whether a structural two-socket spread exists at B=4096 (if spr is
   reproducibly >1.3, a weighted cut is worth re-attempting -- but only
   adopted from repeated, separated measurements, never one race); the
   b1 st/vp line prices the new handshake on CLX.
2. **B=4096 expectation**: honest race + equal cut + NT should land near
   the node's bandwidth floor (winograd's 1.222 = ~193 GB/s with the
   RFO; stnt deletes the RFO, so ~0.9-1.1 us/t is the target).  If it
   still loses to winograd's OMP static split, the difference is the
   pool itself and I should race an OMP-region dispatch at batch as a
   candidate (region cost is noise at 5+ ms calls).
3. **B=1 floor**: the plane phase's 17 tasks x ~0.5 us over nt=17 leaves
   the x phase + 1 barrier + handshake; if the node's new B=1 is still
   >7 us while wallaby holds 5.0, the remaining gap is CLX's slower
   uncore and the next lever is merging the deint/z/transpose/y plane
   pipeline into fewer, larger tasks (fewer release/collect round trips),
   or matrixsimd's locality-aware x-chunk order (each thread starts on
   the columns of planes IT wrote).
4. **B=256 on the node**: the honest arena may flip the pick to st/stnt
   (44 MB combined L3 vs 77 MB working set streams more than wallaby's
   60 MB single-socket L3 suggested); if matrixsimd still leads, steal
   their staged-input mechanism next.

## Round mt_r3

### Where mt_r2 landed on the node, and the diagnosis this round is built on

B=1 6.776 (2nd, matrixsimd 6.163), B=256 0.792 (2nd, 0.755), B=4096
**2.200 against L17_winograd's 1.220** -- with MY OWN arena having raced
the shipped config honestly at 1.93 (par=1.926 in the description), so
mt_r2's arena-fidelity fix worked and was not enough.  The r3 diagnosis
comes from lining up EVERY streaming cell in the round across entries:

* Entries at ~193-200 GB/s: L17_winograd B=4096 (1.220 us/t = 193 GB/s
  with the RFO, `nt=0 dyn=0` OMP static split, **setup 4.2 s**),
  L6_unrolled B=65536 (199.6 GB/s, `disp=omp`, **setup 2.7 s**),
  L6_pfa's T=32 process (200 GB/s, **setup 12.8 s**).
* Entries at 66-82 GB/s: me (NT pool, **setup 0.68 s**, 71 GB/s),
  L17_matrixsimd (plain pool, **0.87 s**, 82), L23_matrixsimd (NT pool,
  **0.47 s**, 66).

Store shape does not separate the groups (winograd won PLAIN, L6_unrolled
won NT); dispatch and setup time do.  The mechanism I believe (extending
the mt_r2 VERDICT SS5): fft3d_create(L, batch) never sees the caller's
buffers -- the driver memsets `out` and freads `in` on its main thread
BEFORE create, all socket-0 -- and the kernel NUMA balancer needs WALL
TIME while the process runs to mark 1.2 GB of PTEs (scan delay ~1 s,
~256 MB per scan pass) before the warmup/timed executes can fault pages
over to the socket that uses them.  A 4 s create hands the timed loop
pre-marked pages that migrate during warmup; a 0.7 s create gets scored
in the unspread all-socket-0 regime at one socket's bandwidth.  My mt_r2
arena was faithful to the driver's INITIAL placement -- which is exactly
the regime the scored loop of a long-setup entry has already left.

### What was built (all schedule/dispatch/timing; zero FP change; bit classes untouched -- outputs cmp-identical to mt_r2 at B=1 and B=4096 on wallaby)

1. **Streaming-cell classification from the working set, not from a
   race**: streaming iff 2*batch*78608 B > 2x (per-socket L3 x nodes the
   TEAM spans).  Team span measured from the create-time OMP region's
   sched_getcpu list against /sys/devices/system/node (wallaby's 32
   close-bound threads span 1 of its 2 sockets; the node's span 2).
   Node: B=4096 streaming, B=256 not (38 < 88 MB).
2. **2 s migration settle before the streaming race** (was 0.15 s clock
   settle): full-team static-cut plain streaming over the arena walks the
   serially-memset (driver-faithful-initial) arena to the owner-local
   steady state the scored loop reaches, so the race prices THAT regime.
3. **Plain-incumbent discipline at streaming cells**: best-plain
   bandwidth is computed from stage 1 (235824 B/vol / us); on a
   multi-node team below 130 GB/s the arena provably never left the
   unspread regime, whose NT-vs-plain verdict did not transfer in either
   prior round -- then plain is pinned outright.  If the regime WAS
   reached (or single-node, e.g. wallaby), the race stands but a
   staged/NT winner must beat best-plain by 10% (was 3%), in stage 1 and
   in the (variant,pf,pfw) grid both.  The near-tie NT pick is the exact
   shape of the 2.904 (r1) and 2.200 (r2) losses.
4. **Team width hard-pinned wide at streaming cells** (shrink race
   deleted there, kept elsewhere) -- the mt_r2 VERDICT's prescription
   verbatim (its L6_pfa evidence: 85 vs 200 GB/s on team width alone).
5. **Dispatch-shape race, spin pool vs OMP parallel region** (same
   l17r_work_batch, same shadow plans, bit-identical), streaming cells
   only; on a multi-node team OMP WINS TIES since it is the only dispatch
   shape with node evidence at 193 GB/s; single-node keeps the pool on
   ties.  If OMP ships, the pool is destroyed (31 extra threads have no
   business next to the OMP team).  Wallaby raced pool 0.704 / omp 0.677
   -> omp shipped there too.
6. **5.5 s create() dwell at streaming cells**: after the pick, create
   keeps executing the final configuration on the arena until 5.5 s of
   wall time -- the same marked-PTE head start the round's fast entries
   had, bought with the exact scored code path.  Setup is "arbitrarily
   expensive" by contract; B=1/B=256 setups stay ~0.3-0.6 s.
7. **B=1**: team grid refined to {2,4,8,12,14,16,17,20,24,32} (node
   picked 16 in r2; winograd's t1=12 also in range), and an x-phase
   cross-core prefetch `xpf` (ADOPTED FROM L17_matrixsimd mt_r1) raced
   JOINTLY with team size: before each x block, pull the NEXT block's
   2x17 A lines (dirty in the plane writers' caches) under this block's
   ~300-cycle compute.  Wallaby: neutral at the winning nt=17
   (6.665/6.680) but -35% at nt=32 (11.14 -> 7.18) and -33% at nt=24 --
   exactly the cross-core-latency signature; the node's slower uncore and
   16-thread pick is where it may actually pay.

### Operation count

Unchanged: 296 FP instr per 17-point transform, 867 transforms = ~423
kflop per volume.  The round touched placement, margins, dispatch, and
one read-prefetch knob.

### Measured (wallaby, SPR 6448Y, 32 threads, shared login node, contended window: clk256 probe read 1.47-3.40 GHz across sessions)

| case | mt_r3 | mt_r2 binary SAME window | note |
|---|---|---|---|
| B=1 | 5.32-6.13 us | 5.40-6.15 | mode 2 nt=17 xpf=0; outputs identical |
| B=8 | 9.71 us/call = 1.21 us/t | (14.0 in r2's window) | mode-1 race |
| B=256 | 122.4 us/call = 0.478 us/t | 122.9 (r2 window) | plain, no dwell (setup 0.28 s) |
| B=4096 | 3947-4309 us/call = 0.964-1.052 us/t | 3969-4187 | stnt dy, dsp=omp, str=1 tn=1 tr=1 bw=294GB/s, setup 5.52 s; outputs identical |

rel_l2 3.11-3.15e-16 at every batch tried (1, 8, 256, 4096), repeatable
bit-identical across runs, and cmp-identical to the mt_r2 binary at B=1
and B=4096.  Wallaby CANNOT show the round's point (tn=1: no unspread
regime to escape); the honest single-socket race still picks NT there
(0.659 vs plain 0.802, past the 10% gate), which is correct for wallaby.

### What did not work / what to know, with numbers

* Nothing was tried and reverted this round; the round is three verdicts
  applied (working-set incumbency, hard-pin wide, arena regime) plus two
  borrowed mechanisms and one forensic conclusion.  The risk carried:
  wallaby A/B at B=4096 is a wash (3947-4309 vs 3969-4187 across two
  interleaved reps) -- every change is either node-conditional (tn>1) or
  tie-broken toward the node-proven shape, so a wallaby wash is the
  expected reading, not a warning sign.
* xpf=1 at nt<=17 on wallaby: within noise (see table in item 7).  Do
  not force it; it is raced.

### Borrowed this round, named

* **x-phase cross-core prefetch (xpf)** -- L17_matrixsimd mt_r1's xpf
  mechanism, applied one-block-ahead (their xpf=2 bulk variant lost on
  wallaby in their own r2 record, so only the lookahead form is raced).
* **OMP-region static dispatch at streaming cells** -- L17_winograd
  mt_r1/r2 (the `mt[n=32 dyn=0]` shape that scored 1.220 = 193 GB/s).
* **Working-set incumbency, hard-pinned wide teams, and the
  pre-spread-arena mechanism** -- the mt_r2 VERDICT (SS5, SS6 item 1);
  the setup-time forensics across L6/L17/L23 entries are this round's
  own addition.

### Pre-registered node expectations (read these against the mt_r3 leaderboard)

* **B=4096**: the whole round.  Dwell + warmup should put the scored
  loop in the spread regime: plain floor ~1.2-1.3 us/t (winograd
  parity).  Branches: (a) settled arena, NT wins >=10% -> expect
  0.85-1.1 us/t and a cell win (desc: tr=1 bw>=130, NT tag, dsp=omp
  likely); (b) arena unsettled -> plain pinned (desc: tr=0 bw<130),
  scored still ~1.2-1.3 if the dwell alone spreads the driver's pages;
  (c) scored ~2.2 again with tr=0 -> the dwell did NOT spread the
  driver's pages and the setup-time theory is wrong for the node --
  in that case next round must test whether numa_balancing is even
  active there (matrixsimd's anb=1 says it is) and consider that
  winograd's advantage is the OMP region itself (my dsp=omp branch
  covers that this round).
* **B=1**: 6.3-6.8 us; upside if the node picks xpf=1 at nt=14-20
  (cross-socket A pulls are its target).  Watch `b1 st/vp(nt,xpf)`.
* **B=256**: unchanged path, expect 0.78-0.80 (desc: str=0, no dwell).

### Next round

1. Read desc telemetry first: str/tn/tr/bw + dsp + par-vs-scored at
   B=4096 decides which branch above happened; act on that branch only.
2. If (c): race a create-time get_mempolicy page-home scan of the ARENA
   (L8_fusedaxes's governor, legal read-only) to measure directly what
   fraction migrated during settle/dwell, and report fr in the desc.
3. B=1 floor: merge the deint/z/transpose/y plane pipeline into fewer
   dispatch units at nt<=8 (fewer release/collect trips), or split the
   x phase's 37 blocks into 74 half-blocks for balance at nt=14-20
   (needs a half-width wino17 store path -- check cost first).
4. B=256: if matrixsimd still leads, their staged-input mechanism is the
   remaining unstolen piece.
