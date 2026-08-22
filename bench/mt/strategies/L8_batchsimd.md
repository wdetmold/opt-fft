# L8_batchsimd — multicore strategy record

Phase-1 history (how the serial kernel got its shape — the 52-op split-complex
radix-8 codelet, FUSED shuffle placement, the SI de-alias, the regime-gated
allocation, the retired B=1 tournament) lives in
`../../geom/strategies/L8_batchsimd.md`.  This file records the MULTICORE
rounds only.

## Round mt_r1 — volume-parallel over a persistent pinned spin pool; B=1 stays serial, with the measurement that says why

### What was built

The serial kernel is byte-for-byte the phase-1 result — every phase-1 runner
still compiles, the B=1 path and its fixed FUSED/SI520 pick are untouched, and
the phase-1 tuner remains as the fallback when no pool can be built.  On top:

1. **A persistent pthread spin pool, built in `fft3d_create()`** — execute
   creates no threads.  The design is borrowed nearly whole from
   **L17_matrixsimd mt_r1**, whose record killed the alternatives with
   numbers: gcc's execute-time `#pragma omp parallel` fork/join costs
   3.4/5.4/11.8/13.9 us at 4/8/16/32 threads on wallaby (more than my whole
   B=64 execute), the all-workers-acknowledge handshake and the futex/condvar
   pool both lost badly in their file.  So: per-worker padded release/done
   generation flags, pause-spin with a decay to a 100 us nanosleep poll,
   and affinity captured by running `sched_getaffinity` inside a create()-time
   OMP parallel region and copying mask t to pool worker t — reproducing
   `OMP_PROC_BIND=close, OMP_PLACES=cores` exactly, no topology guessing.
   Workers pin FIRST, then allocate and memset their private 2112-double
   arena, so all per-thread scratch is first-touched NUMA-locally (the brief's
   two-socket warning).  Never more than `omp_get_max_threads()` threads.
2. **batch > 1: volume-parallel.**  Rank t runs the UNCHANGED phase-1 FUSED
   runner over the contiguous slice [nvol·t/T, nvol·(t+1)/T) on its own
   scratch (a volume is 128 whole cache lines, so slice boundaries share
   nothing), or over dynamically grabbed blocks (one `fetch_add` per grab)
   when the tuner prefers it.  Each runner's last volume already executes with
   PF_NONE, so no prefetch reaches past a slice.  NT-store ranks `sfence`
   before their done-store (NT is weakly ordered; the done flag must not
   overtake the data).
3. **The MT tuner** races (team size × runner × schedule) under the REAL pool
   on a serially-filled surrogate (first touch matches the driver's `fread`),
   keeping the phase-1 arena discipline: interleaved min-of-7, one untimed
   state-setting pass per candidate per trial, 3% hysteresis toward the
   default.  Mid regime: T ∈ {32,16,8,1} × {s0,none} × {static,dyn2};
   streaming: T ∈ {32,16} × {s0w,s0,none,nt+s0,nt} × {static,dyn8}.
   NT and no-prefetch are BACK in the candidate sets although five phase-1
   rounds retired them — L17_matrixsimd's mt_r1 measured NT flipping from
   loser (single-thread) to winner (32 threads, the RFO is a third of the
   DRAM traffic) and all prefetch flags losing at 32 threads (no idle fill
   buffers).  Both reproduced here, numbers below.  If the tuner's verdict is
   T=1, the pool is torn down entirely so idle workers cannot even sleep-poll
   during the timed region.
4. **B=1 runs the phase-1 serial path with NO pool.**  `create()` still builds
   a pool, measures its empty-job round-trip (release+collect, zero work) at
   T=2/8/32, publishes it in the description as `poolrt{...}`, and destroys
   the pool — so the node's own numbers will document the claim every run.

### Why B=1 does not parallelise (the argument, with the measurement)

One volume is 0.551 us of total work on the node (0.328 on wallaby).  The
cheapest possible 2-way split (pass A halves, barrier, pass B halves) saves at
most half the compute, ~0.27 us, and pays the handshake twice.  Measured
empty-job round-trip on wallaby: **0.407 us at T=2, 0.620 at T=8, 1.684 at
T=32** (L17_matrixsimd measured 0.32/1.41 independently — same order).  The
handshake alone exceeds the theoretical maximum saving at every team size, so
the whole intra-volume design space is dead at L=8 before any of its own costs
(the mid-pass barrier, the split's lost locality) are even counted.  Not
rediscovering this per round: the poolrt line in every t_*.json is the check.

### Operation count

Unchanged per volume: 1248 vector FP + 896 shuffles + 256/256 loads/stores
(FUSED).  The MT layer adds zero arithmetic — only the release/collect flags
(31+31 line transfers per execute at T=32) and, under dyn, one fetch_add per
block.

### Measured on wallaby (SPR 6448Y, 32 threads = one socket, dev numbers, best-of-runs; login-node spread at 32T is a few %)

| case | serial (phase-1 path, wallaby) | this round | per-execute | speedup | pick |
|---|---|---|---|---|---|
| B=1     | 0.328 us | 0.328 us (unchanged) | — | 1.0x by design | serial FUSED/SI520 |
| B=64    | ~0.55 us/t | **0.036 us/t** | 2.29 us | ~15x | T=32, no prefetch, static |
| B=2048  | ~0.70 us/t | **0.021 us/t** | 42.9 us | ~33x (32 MiB fits wallaby's 60 MiB L3 when split — superlinear-looking, honest number is B=16384) | T=32, s0, static |
| B=16384 | ~1.10 us/t | **0.058 us/t** | 944 us | ~19x | T=32, **nt+s0**, static |

Node phase-1 baselines for the efficiency report the brief asks for: 0.551 /
0.587 / 0.959 / 1.236 us at B=1/64/2048/16384.  Wallaby-to-node caveat as
always: wallaby is single-socket for 32 threads (no NUMA, no UPI), 2-FMA SPR
vs the node's 1-FMA CLX, 60 vs 22 MiB L3 — so the node's B=2048 will tune in
the *streaming* set (32 MiB > 0.9×22 MiB) even though wallaby tunes it mid.

Arena tables from wallaby creates (us/transform, published in every JSON):
* B=64: `T32/s0=0.0604, T32/none=0.0352, T32/s0/d2=0.0888, T16/s0=0.0446,
  T16/none=0.0418, T8/s0=0.0683, T1/s0=0.4672` — **prefetch costs +72% at 32
  threads** (0.0604 vs 0.0352); dyn2 loses 2.5x (fetch_add every ~1 us of
  work × 32 threads on one cursor line); serial-vs-best = 13x in-arena.
* B=16384: `T32/s0w=0.0571, T32/s0=0.0504, T32/none=0.0485, T32/nt-s0=0.0448,
  T32/nt=0.0457, T32/nt-s0/d8=0.0468, T32/s0w/d8=0.0621, T16/nt-s0=0.0543` —
  the phase-1 verdict EXACTLY inverted: **s0w (prefetchw), phase-1's 6/6
  streaming winner, is now the worst runner**, and **NT stores, five rounds a
  loser, win** (−21% vs s0w).  Both directions are L17_matrixsimd's mt_r1
  finding, adopted; my numbers agree.

### What did not work / dead ends, with numbers

* **Dynamic scheduling on wallaby**: dyn2 at B=64 0.0888 vs static 0.0352;
  dyn8 at B=16384 0.0468 vs static 0.0448.  Kept in the candidate sets anyway
  — wallaby cannot see the node's two-socket imbalance (driver buffers are
  fread-filled by one thread, so socket 1 pays UPI for every line), and
  L17_matrixsimd measured dynamic WINNING at streaming even single-socket.
  The node decides.
* **Timing the first pool job for reps calibration**: a worker that decayed to
  the sleep poll pays up to 100 us wake latency on the next release; the
  calibration was underestimating reps until an untimed warm call was added.
* **Not tried, deliberately**: intra-volume split at B=1 (killed by the
  poolrt measurement above before implementation); OpenMP-region execute
  (killed by L17's fork/join numbers); LANEX/FUSED3/AA runners in the pool
  (zero batched node picks in phase-1 rounds 7–11).

### Borrowed, and from whom

* Persistent pinned spin pool, per-worker release/done flags, nanosleep
  decay, OMP-affinity capture: **L17_matrixsimd mt_r1** (their record also
  pre-killed the all-ack handshake and the futex pool).
* NT-at-32-threads and prefetch-loses-at-32-threads candidate reversals:
  **L17_matrixsimd mt_r1** (B=4096 stage-A table).
* Serially-filled surrogate so first touch matches the driver: same file.

### What I would do next

* **Node evidence first**: read the mt_r1 leaderboard's picks and arena tables
  (they publish in every t_*.json).  The open node-only questions: does dyn
  win at streaming with two sockets; is T=16 (one socket) better than T=32 at
  B=64 (the cross-socket collect at 1.4+ us vs 0.6 us of extra compute per
  rank is close); does nt-s0 hold at B=2048 in the node's streaming set.
* **Leaner collect for the mid regime**: the 31-flag flat release/collect is
  ~half the B=64 execute.  A tree release halves the caller's store chain;
  worth one candidate.  Also worth trying: releasing only T−1 = 15 workers on
  the caller's socket at B=64 (T=16 static already approximates this).
* **Slice-aligned dyn blocks** (dynb = slice size) to recover dyn's balance
  without its grab frequency.

## Round mt_r2 — fix the two things the mt_r1 node data convicted: the streaming surrogate and the flat collect

### Where mt_r1 landed on the node, and the diagnosis

* B=1: 0.554 us, tied for the cell win with L8_fusedaxes.  poolrt{2=0.406,
  8=0.829, 32=4.424} published — the serial-B=1 argument is now node-documented.
* B=2048: 0.028 us/t (56.5 us/call) vs fusedaxes 0.026 (53.7).  Pick T=32/s0,
  arena 0.0274 — the pick was right; the ~2.8 us/call gap is the handshake.
* **B=32768: 0.295 us/t — 1.84x off the cell, my worst result of the round.**
  The t_*.json descriptions show exactly why: the tuner's arena read
  `T32/none=0.145–0.157` and picked plain stores 3/3, and the driver then ran
  0.295.  The surrogate was 4×L3 capped at 8192 volumes = 5632 volumes =
  88 MiB on the node, so ~1/4 of a plain-store candidate's RFO traffic stayed
  L3-resident and plain looked 2× better than it is at the driver's 512 MiB.
  Meanwhile the same arena's `T32/nt-s0=0.168–0.171` matches the 0.176
  fusedaxes scored with NT.  The information to win the cell was in my own
  arena table; the surrogate regime threw it away.  L8_fusedaxes mt_r1 hit
  the identical failure on wallaby (their "what did NOT work" item 1, wrong
  pick by 27%) and published the fix; I did not rediscover it, I took it.

### What changed this round

1. **Streaming surrogate: cap 4×L3 → 8×L3, clamp [4096,8192] → [8192,32768]
   volumes** — borrowed verbatim from L8_fusedaxes mt_r1.  On the node the
   B=32768 arena now runs at 11264 volumes = 176 MiB (8× the 22 MiB L3), in
   the driver's residency regime.  Wallaby confirms the flip: the tuner now
   picks **T=32 nt-s0** with arena
   `{T32/nt-s0=0.0764, T32/s0=0.1126, T32/s0w=0.1094, T32/none=0.1133,
   T32/nt=0.0765, T32/nt-s0/d128=0.0762, T32/s0/d128=0.1125,
   T24/nt-s0=0.0781, T16/nt-s0=0.0805}` — NT wins by 32%, the exact margin
   the old 4×L3 surrogate inverted.
2. **Tree collect** (my own mt_r1 "leaner collect" queue item; no other
   entry had built it).  mt_r1's collect was rank 0 serially spin-reading 31
   remote done lines — on the two-socket node each is a cross-socket
   cache-to-cache transfer, and poolrt measured the empty round-trip at
   4.424 us at T=32, ~8% of the whole B=2048 execute and the entire
   53.7-vs-56.5 gap to fusedaxes (their dispatch+join ≈ 0.8 us).  Now worker
   t acquire-waits for children 2t+1/2t+2 (< T, hence always released) and
   only then release-stores its own done, so rank 0 collects TWO lines and
   the critical path is log2(T) ≈ 5 c2c hops instead of 31 serialized ones.
   The release stays FLAT: 31 independent store RFOs pipeline in the store
   buffer, and a tree release would serialize the wake path.  The
   rewrite-job-fields safety argument survives: done[1]/done[2] are
   release-stored only after the whole subtree was acquire-read, so the
   acquire chain to rank 0 still covers every released worker.
3. **Streaming candidate set rebuilt on the node's own arena tables.**
   Anchor rule adopted from fusedaxes: nt-s0 anchors when ws > 1.5×L3;
   s0 (the node's 3/3 winner at B=2048, which sits at 1.45×L3) anchors in
   the 0.9–1.5× band.  Dropped: nt-s0/d8 and s0w/d8 (lost 3/3 on the node,
   0.20–0.25 vs 0.15–0.19 — fetch_add every ~1.4 us of work × 32 threads on
   one cursor line).  Added: **coarse dynamic blocks dblk = nvol/256, clamp
   [8,512]** (one fetch_add per ~22 us of work at B=32768) so the two
   sockets can self-balance the UPI asymmetry that static equal slices
   cannot see — wallaby (single-socket) has it tying static at 0.0762 vs
   0.0764, and only the node can show the two-socket case; and **T=24
   nt-s0** (16 near + 8 far threads), probing between the node's near-equal
   T32 (0.168–0.171) and T16 (0.169–0.172) nt-s0 endpoints.

### Operation count

Unchanged per volume: 1248 vector FP + 896 shuffles + 256/256 loads/stores
(FUSED), 16 KiB DRAM traffic per volume with NT (8 read + 8 written, RFO
deleted), 24 KiB with plain stores.  The tree collect moves work between
threads but adds none: still exactly one release store and one done store
per worker per execute; internal workers add ≤2 acquire spins they pay while
they would otherwise idle.

### Measured on wallaby (SPR 6448Y, 32 threads = one socket, shared login
node — dev numbers, relative only)

| case | mt_r1 (this file) | mt_r2 | pick | note |
|---|---|---|---|---|
| B=1     | 0.328 us | 0.340 us min (load-dependent, phase-1 floor) | serial FUSED/SI520, unchanged path | rel L2 2.27e-16 |
| B=2048  | 42.9 us/call = 0.021 us/t | **34.4 us/call = 0.0168 us/t** (−20%) | T=32, s0, static (wallaby mid set: 32 MiB < 60 MiB L3) | tree collect is the only path change at this cell on wallaby |
| B=32768 | (not measured at this B; B=16384 was 0.058) | **2508–2531 us/call = 0.0766–0.0773 us/t** | **T=32, nt-s0, static** — NT now wins the arena by 32% | equals fusedaxes' mt_r1 wallaby 2466–2515 on the same host |

poolrt (empty-job round-trip, published every B=1 create): wallaby
{2=0.295–0.343, 8=0.673–0.880, 32=1.527–1.957} vs mt_r1's flat-collect
{2=0.407, 8=0.620, 32=1.684}.  On single-socket wallaby the tree is only
~0–10% at T=32 (intra-socket c2c is cheap; each tree hop pays a pause-loop
detection latency), and T=8 is slightly worse — the tree's target is the
node, where 31 serialized CROSS-SOCKET reads cost 4.424 us and depth-5
should land ~1.5–2 us.  The node's own B=1 poolrt line will price it.

Correctness: PASS (rel L2 2.20–2.28e-16, tol 1e-12) at B = 1, 2, 3, 5, 31,
33, 64, 100, 513, 2048, 32768; repeatable bit-identical across runs at every
size; builds warning-free (-Wall -Wextra) at native SPR, cascadelake,
haswell (AVX2), and -DL8_EMU8.

### What did not work / verdicts recorded, with numbers

* **Tree collect on ONE socket is nearly free but not a win**: wallaby T=32
  round-trip 1.527–1.957 vs flat 1.684, T=8 0.673–0.880 vs 0.620.  Shipped
  anyway on the node evidence (4.424 us flat at T=32 is 31 serialized
  cross-socket c2c transfers; the tree replaces them with ~5) — if the
  node's poolrt does not drop well below ~2.5 us next round, revert to flat
  and try prefetch-then-scan instead.
* **Fine dynamic (d8) is dead on the node**: 0.2037–0.2126 vs static nt-s0
  0.168–0.171 in all three B=32768 node arenas (and d2 lost 2.4× at B=2048
  on wallaby again this round, 0.0531 vs 0.0219).  Only the coarse
  nvol/256 block survives as a candidate.
* **Not rediscovered, per records**: intra-volume B=1 splitting stays dead —
  my own node poolrt (0.406 us at T=2 > the 0.27 us maximum saving) plus
  fusedaxes' independent 0.8 us join measurement both say so.

### Borrowed, and from whom

* **8×L3 / [8192,32768] streaming surrogate cap**: L8_fusedaxes mt_r1,
  verbatim, including the diagnosis pattern (their wallaby wrong-pick was my
  node wrong-pick).
* **NT anchor above 1.5×L3, plain anchor below**: L8_fusedaxes mt_r1's
  anchor rule.
* Tree collect and the coarse-dynamic-block idea are mine (both were queued
  in my mt_r1 "what I would do next"); T=24 was prompted by the node's flat
  T32≈T16 nt-s0 readings.

### What I would do next

* **Read the node's B=32768 pick and arena first.**  Expected: nt-s0 (or
  nt-s0/d128) at ~0.17 us/t, from 0.295 — that alone moves the cell from
  1.84× to ~1.05× vs fftw3_patient's 0.161.  If d128 wins, the socket
  self-balancing hypothesis is confirmed; consider dblk tuning (64/256).
* **Read the node's new poolrt{32}**: if it is not ≤ ~2 us, try
  prefetch-all-then-scan flat collect, or a two-level tree matched to the
  socket topology (workers 1 and 16 as socket roots).
* **B=2048**: with the handshake shaved, the remaining gap to fusedaxes
  (if any) is real kernel difference; their fused+pfs at T=32 and mine are
  the same shape, so parity is the expectation.  If the node still shows
  +1–2 us/call, measure the release path (31 store RFOs from rank 0) with a
  socket-rooted two-level release.
* If the node's B=32768 lands ~0.17 and fftw3_patient holds 0.161, the next
  lever is traffic, not scheduling: nothing below 16 KiB/vol exists for an
  out-of-place transform, so the question becomes whether fftw's edge is
  page placement (their planner first-touches with the team) — test a
  variant where each rank streams its slice through a small NUMA-local
  bounce buffer before the NT write.  Price it first; it may cost more than
  the UPI it saves.

## Round mt_r3 — the handshake the node convicted twice, and the VERDICT's L=8 order: use the whole machine

### Where mt_r2 landed on the node, and the diagnosis

* **B=1: 0.557 us — won the cell** (fusedaxes 0.558, a 0.2% tie).  But
  poolrt{2=0.276–0.445, 8=0.878–1.484, **32=5.205–5.544**}: my mt_r2 tree
  collect made the round-trip WORSE than mt_r1's flat 4.424 us.  Each of the
  log2(T) tree hops pays a cross-socket detection latency *serially*; the
  revert rule my own mt_r2 record pre-registered ("if poolrt{32} is not well
  below ~2.5, revert") fired.
* **B=2048: 56.9 us/call vs fusedaxes 53.9.**  The pick was right (T=32/s0,
  arena 0.029 — the driver ran 0.0278/t), so the ~3 us/call gap **is** the
  handshake: fusedaxes' epoch-word pool joins in ~0.8 us.  Also a 3-process
  pick lottery (s0w/s0/s0 → 62.7/56.9/58.3): s0w and s0 tie in-arena at
  0.029 and the hysteresis can't separate them.
* **B=32768: 0.174 us/t — mt_r1's 0.295 repaired (1.70x), three-way tie with
  the siblings at 0.13%, still 0.92x vs fftw3_patient's 0.159.**  The mt_r2
  VERDICT §5 measured why, using fusedaxes' governor: `gov{fr=0,nb=1}` —
  the driver's 512 MiB is 100% socket-0 — and all three L=8 entries (mine:
  mt{T=16 run=nt-s0}) chose single-socket teams and parked at 94 GB/s,
  while L=6 proves T=32 reaches 200 GB/s at a streaming cell on this node
  and fftw's winning mode is >102 GB/s.  The VERDICT's L=8 instruction is
  explicit: "use the whole machine … force T=32 with a socket-aware chunk
  split and re-measure", and its §6 names the settling experiment: "nobody
  has yet read fr under a 32-thread team at a streaming cell."

### What changed this round

1. **Epoch-word pool protocol** (borrowed whole from **L8_fusedaxes**, who
   took it from **L36_pfa mt_r1**; the all-ack invariant from **L17_rader
   mt_r1**).  Release is now ONE store to a single cache-line-aligned epoch
   word that all 31 workers spin-acquire on (they re-read it *shared*; my
   old per-worker release flags cost rank 0 31 separate store RFOs).
   Collect is a flat publish-then-scan over per-worker padded done lines,
   and EVERY worker acks every epoch — team members and idle ranks alike —
   which is what makes rewriting the job fields for the next dispatch safe
   with no tree, no fence gymnastics.  The tree collect is deleted.
   Workers with tid >= nthr ack without working, so the execute-time
   governor can switch team width call-by-call on one pool.
2. **Execute-time streaming governor** (armed when ws > 6x L3, batch >=
   4096, pool built, >= 8 threads; `L8_GOV=0` disables).  cfg0 = the create
   pick (node: T16/nt-s0, the known 0.174); cfg1 = the OTHER width on nt-s0
   (node: T=32).  Three baseline calls at cfg0, then a **self-extending
   full-width dwell** on the caller's real buffers — minimum 16 calls,
   extended while still improving >1.5%, hard cap 44 — with
   `get_mempolicy(MPOL_F_NODE|MPOL_F_ADDR)` scans of ~128 sampled pages of
   in/out every 8th call.  A sustained wide dwell is also exactly the
   remote-fault pattern AutoNUMA needs if it is ever going to spread
   socket-0 pages, and "still improving" is what migration looks like from
   inside.  Lock = argmin with 2% hysteresis to cfg0; the loser is revisited
   once every 24 calls (a late migration re-locks on a >3% win).  Publishes
   `gov{fr0,fr,nb,T…=…,T…=…,lock}` in the description, so this round's L=8
   JSONs finally carry **fr under a wide team** — the §6 experiment — from
   every process.  Legality is fusedaxes' mt_r2 argument, adopted verbatim:
   every probe call is a full, correct, bit-identical execute, and
   min-of-min ignores slow ones.
3. **Tuner surface pruned per the VERDICT**: dynamic scheduling REMOVED
   from all candidate sets (d2 2.4x worse, d8 lost 3/3, d128 tied-to-worse
   3/3 across two rounds of node tables; VERDICT: "it should be removed
   from the tuner surface, not re-raced"); s0w removed from the streaming
   set (zero node picks in mt_r1/mt_r2; the one process that picked it at
   B=2048 was the 62.7 outlier — this also shrinks that cell's lottery).

### Operation count

Unchanged per volume: 1248 vector FP + 896 shuffles + 256/256 loads/stores
(FUSED); 16 KiB DRAM traffic per volume with NT, 24 with plain.  The epoch
protocol is one release store + one done store per worker per execute (the
tree's extra acquire spins are gone).  The governor adds two clock_gettime
per call and, on scan calls only, ~256 get_mempolicy syscalls (~tens of us
against a 5.8 ms call).

### Measured on wallaby (SPR 6448Y, 32 threads = one socket, shared login
node — same-window comparisons only, quiet minima)

| case | mt_r2 | mt_r3 | note |
|---|---|---|---|
| B=1     | 0.340 | 0.329–0.333 us | serial path untouched; rel L2 2.27e-16 |
| B=2048  | 34.4 us/call | **32.3–33.3 us/call = 0.0158–0.0163 us/t** (−4–6%) | pick T=32/s0 static, stable 3/3 (one 43.2 outlier was a busy window — its own arena rows were inflated 35% too) |
| B=32768 | 2508–2531 us | **2490–2498 us = 0.0760–0.0762 us/t** | single-socket null test: `gov{fr0=0,fr=0,nb=1,T32/nt-s0=0.0748,T16/nt-s0=0.0777,lock=cfg0}` — governor correctly reads 0% remote, prices both widths on the real buffers, locks the incumbent |

poolrt (empty round-trip, all-ack pool): wallaby {2=1.73, 8=1.44,
**32=1.27**} vs mt_r2 tree {32=1.53–1.96} and mt_r1 flat {32=1.68}.  Note
the semantics changed: with all-ack the round-trip is ~team-independent
(the "T=2" figure now pays all 31 acks), so the one number that matters is
the full-pool one, and it dropped ~25% on a single socket.  The node —
where the flat/tree numbers were 4.4/5.4 us — is where the epoch word
should pay: fusedaxes measures ~0.8 us for the identical protocol.

Correctness: PASS (rel L2 1.3–2.3e-16, tol 1e-12) at B = 1, 2, 3, 5, 31,
33, 64, 100, 513, 2048, 4096, 32768; repeatable bit-identical across runs
at every size (including with the governor probing); builds warning-free
(-Wall -Wextra) at native SPR, cascadelake, haswell (AVX2), scalar, and
-DL8_EMU8.

### What did not work / what was deliberately not done, with reasons

* **Tree collect: reverted for cause.**  Node poolrt{32} 5.24–5.54 vs flat
  4.42 — my mt_r2 record's own revert threshold (2.5 us) was missed by 2x.
  Not patched (prefetch-then-scan etc.); replaced by the protocol a rival
  measured at 0.8 us.  Lesson recorded: on this node, serializing the
  collect path through worker-to-worker acquire chains adds latency per
  hop; one shared read-mostly line beats both of my designs.
* **Weighted per-thread re-cut (fusedaxes' cfg2/3 machinery): not ported.**
  Their node result shows it converging to nt16-equivalent when fr=0, i.e.
  it re-derives cfg0, and if pages spread under the wide dwell a static
  equal split is already balanced (stable contiguous ownership means each
  thread's pages migrate to its own socket).  Two static widths + the fr
  scan cover both regimes at a fraction of the machinery.
* **A T=24 governor config: not added.**  Node arena T24/nt-s0 =
  0.196–0.198 sits between T32 (0.209) and T16 (0.184) in the socket-0
  regime — it is never the argmin in either regime, and every extra config
  dilutes the dwell that makes cfg1 informative.  T24/nt-s0 stays in the
  create-time arena only.
* **B=1: no change.**  Cell won at the serial floor; poolrt keeps
  documenting why it stays serial (even at 1.27 us the empty handshake is
  ~4x the whole 0.33 us transform).

### Borrowed, and from whom

* **Epoch-word release + flat all-ack collect**: L8_fusedaxes (protocol via
  L36_pfa mt_r1; all-ack invariant via L17_rader mt_r1).
* **Execute-time governor pattern, its legality argument, gov_scan_remote
  (get_mempolicy fr scan, ported nearly verbatim), the numa_balancing
  read, and the gov{} publication format**: L8_fusedaxes mt_r2.
* **Dyn removal**: the mt_r2 VERDICT's panel-wide ruling plus my own three
  rounds of node tables.
* The self-extending wide dwell (probe budget that rides improvement, so a
  migration in progress extends its own experiment) is my construction,
  motivated by fftw3_patient's within-process bimodality at this cell
  (minima 5215/5222/5230 vs medians ~9640 — something improves DURING a
  process under a wide team, in all three of its processes).

### Node predictions (stated to be scored)

* **B=1: 0.55–0.56 us**, byte-identical serial path.  poolrt{32} should
  read ~0.8–2 us (from 5.2–5.5); if it does not drop below ~2.5 us the
  epoch protocol underdelivered cross-socket and the record should say so.
* **B=2048: 53–55 us/call** (0.026–0.027 us/t) — parity with fusedaxes ±1
  us, since the ~3 us handshake gap was the only stable difference and we
  now run the same protocol.  Pick T=32/s0 static.
* **B=32768, keyed on the governor's published fr under the T=32 dwell**:
  (a) fr rises (AutoNUMA spreads under the wide team): lock=cfg1 and
  0.09–0.14 us/t — takes the cell from fftw's 0.159 outright;
  (b) fr stays 0 and T32 never beats T16 on the real buffers: lock=cfg0,
  ~0.174 us/t (the probe calls cost only early, unscored samples) — the
  cell stays a narrow loss but the §6 experiment is finally on record in
  every process, with the wide-team driver-buffer number published beside
  it;  (c) fr stays 0 but T32 wins anyway (fftw's 103-GB/s mode is
  concurrency into one socket's controllers, not placement): lock=cfg1 at
  ~0.15–0.16 — a photo finish either way.
* Pick-lottery risk at B=2048 is reduced (s0w gone, dyn gone): expect one
  pick string across all three processes.

### What I would do next

* **Read gov{fr0,fr,…,lock} from all three B=32768 processes first.**  It
  decides which of (a)/(b)/(c) happened and whether the dwell budget (44)
  was the binding constraint — if fr was still climbing at lock time,
  lengthen the dwell or start cfg1 from call 0.
* If (b): the remaining 9% is single-socket stream efficiency; the next
  levers are fusedaxes' seq3AA alias-free store order inside MY pool
  runners (they measured seq3-nt > fused-nt in-arena at T=32), and the
  §4.5 counter if perf_event_paranoid ever drops.
* **B=2048**: if parity with fusedaxes is reached, the cell is closed —
  both entries at the same protocol, same shape, same pick.  If a ~1 us
  gap persists, diff the remaining per-call work (their pool shrinks to
  the picked team; mine keeps 31 ackers for the governor's sake — a
  shrink-when-not-armed is a 10-line change).
* If the node's poolrt{32} lands ~1 us, B=64-class mid batches (not scored
  at L=8, but tryout-visible) become handshake-light too; nothing to do
  unless a future round scores them.
