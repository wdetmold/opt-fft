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
