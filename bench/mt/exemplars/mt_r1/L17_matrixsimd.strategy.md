# L17_matrixsimd — multicore strategy record

Phase-1 history (how the kernel itself got its shape — nested cyclic/negacyclic
17-point modules, mixed zmm+ymm tails, X-first/X-last bit classes, address-safe
t1, staged input/output) lives in `../../geom/strategies/L17_matrixsimd.md`.
This file records the MULTICORE rounds only.

## Round mt_r1 — the parallel layer: per-thread children, a spin pool, and a two-phase intra-volume decomposition for B=1

### What was built

The single-thread kernel was left byte-for-byte intact (all 50 phase-1 exec
variants still compile and the phase-1 plan-time tuner still runs for the
`batch < 64` single-thread fallback).  On top of it, three parallel modes,
selected at plan time by timing the REAL parallel path on the plan's machine:

1. **mode 0 — single thread**: the phase-1 path, unchanged.  Kept as a raced
   candidate so "does not parallelise" is a measurable verdict, not a guess.
2. **mode 1 — volume-parallel** (`batch >= 64`, and raced at `2 <= batch < 64`):
   every thread runs the SAME plan-selected exec variant on its own volumes,
   either a contiguous static split or dynamically grabbed blocks of `dynb`
   volumes (one `fetch_add` per grab).  Each thread has a fully private child
   plan — own coefficient tables, own pb/sc/t1/t1b/so/t2 scratch — built INSIDE
   the team's parallel region so it is first-touched NUMA-locally.  No shared
   mutable state; the only shared cache lines anywhere are the 31
   volume-boundary lines of `out` (a volume is 78608 B = 1228.25 lines).
3. **mode 2 — intra-volume** (`batch < 64`, the B=1 case): the X-last class-B
   transform decomposes into `17*B` independent plane units (Y group + Z group
   on a thread-private plane buffer, `in -> t1g`), ONE barrier, then `73*B`
   independent X-chunk units (`t1g -> out`).  Every unit runs exactly the chunk
   sequence of `l17_execm_xla` (mixed zmm+ymm tail, pinned sines, r8 padded
   t1 stride 5120 B — so plane boundaries never share a cache line between
   threads), which is why the output is bit-identical to class B: same chunks,
   same operands, same per-value order, only the interleaving across
   independent units differs (the r4 pipelining argument, reused).

**The execute-time team is NOT an OpenMP region.**  Measured on wallaby, gcc's
fork/join for an execute-time `#pragma omp parallel` costs 3.4 us at 4 threads,
5.4 at 8, 11.8 at 16, 13.9 at 32 — more than the whole parallel B=1 transform.
So `fft3d_create()` builds a persistent pthread pool (the brief's "thread pools
belong in create()"), pinned to the SAME cores OpenMP was given: each OMP
thread's affinity mask is captured with `sched_getaffinity` inside the
child-building parallel region and copied to the pool worker of the same index,
so `OMP_PROC_BIND=close, OMP_PLACES=cores` is reproduced exactly without
guessing the machine's topology.  Release is one atomic generation store into a
per-worker padded flag (only the ACTIVE team's flags are touched — idle workers
never wake and cost nothing on the release path); collection is per-worker done
flags; workers busy-spin (`pause`) between back-to-back executes — the driver's
timing loop — and decay to a 0.1–1 ms `nanosleep` poll after ~4 ms idle so the
plan-time single-thread probes and other tenants are not perturbed.

Two dead ends inside the pool design, both with numbers:
* **All-workers-acknowledge handshake** (every worker acks every generation so
  job fields can never be read stale): the caller then reads 31 remote done
  lines per execute, and B=1 intra nt=2 measured 14.4 us against 12.7
  single-thread.  Replaced by per-worker release flags, which close the
  staleness race by construction (a worker only reads job fields when ITS flag
  was bumped, and the caller waits for that worker before mutating anything).
* **Futex sleep with cond-var wakeup**: inactive workers (never released) fell
  asleep permanently, so every execute paid mutex + broadcast + spurious wakes
  of up to 30 threads: the whole B=1 race degraded ~3x (intra nt=2 went 14.4 ->
  45.9 us).  Replaced by the nanosleep decay, which needs no wake handshake.

### Tuning (all in create(), all within one bit class per batch regime)

* `batch >= 64`: stage A races all 21 class-D variants under the full team
  (dyn=2 while racing, so the node's remote-socket imbalance cannot scramble
  the kernel ranking); stage B races team size {32,24,16,8} x schedule
  {static, dyn1, dyn2, dyn4} on the winner; stage C races (pf,pw,pt) jointly
  (the r4 lesson about non-factorizing knobs, kept).  Arena = min(batch, 1024)
  volumes, filled serially so first-touch matches the driver's own buffers.
  The phase-1 single-thread stage 1b/2 are skipped when a team exists — the
  single-thread ranking does not predict the 32-thread bandwidth-shared one
  (verified below: NT store LOSES single-threaded on wallaby and WINS at 32).
* `batch < 64`: phase-1 stage 1 still picks the single-thread winner within
  class B; then the MT race times mode 0 vs mode 1 (nt=min(batch,32)) vs mode 2
  (nt in {1,2,4,8,12,16,17,24,32}), then races the mode-2 X-phase team
  `nxr <= nt` and an X-phase cross-core prefetch `xpf` on the winner.
* Bit discipline unchanged from phase 1: every candidate raced for one batch
  size is in that regime's bit class (mode 1 and mode 2 cannot change a
  volume's bits — identical kernels, identical tables, address-only
  differences), so the wall-clock pick cannot change the output.  VERIFIED on
  wallaby by cmp on full outputs at B=8: mode0 == mode1(nt=8) == mode1(nt=3)
  == mode2(nt=16) == mode2(nt=17,nxr=4); and two independent processes at
  B=256 produced identical bytes.

### Operation count

Unchanged per volume: 40.8k vector FP ops (zmm-equivalents ~225.5 per volume
with the mixed tail).  The multicore layer adds zero arithmetic — only the
mode-2 path's extra t1g traffic (87 KiB per volume crossing cores once, the
unavoidable all-to-all of a transposed pass) and the handshake.

### Measured on wallaby (SPR 6448Y, 32 threads on one socket, dev numbers)

| case | phase-1 single-thread | this round | speedup |
|---|---|---|---|
| B=1    | 12.1 us/t (mode-0 race print) | **5.26 us/t** (intra, nt=16 typical pick) | 2.3x |
| B=8    | ~12.1 us/t | **1.62 us/t** (13.0 us/execute, mode-1 nt=8) | 7.5x |
| B=256  | ~13 us/t | **0.385 us/t** (98.6 us/execute; 0.53 on a noisy rerun) | ~34x |
| B=4096 | ~13 us/t | **0.84 us/t** (3.45 ms/execute) | ~15x |

B=256 sits in wallaby's 60 MB L3 (in+out = 40 MB), hence the superlinear-
looking number; B=4096 (644 MB) is the honest streaming regime.  At B=4096 the
stage-A table shows exactly the inversion the MT race exists for: the winner is
**exec20_w4 = 512-bit, C-parked, pinned, X-first, PIPELINED + NT STORE at
0.655 us/t** (then nt=32 dyn=1 -> 0.634), with plain X-first at 0.79–0.88 —
NT stores lost on this machine single-threaded in phase 1 (r3: 13.75 vs 12.58)
and win at 32 threads, because with bandwidth shared the RFO deletion is a
third of all DRAM traffic.  Team-size stage B: nt=32 0.634, nt=24 0.799,
nt=16 1.172, nt=8 2.298 — bandwidth keeps scaling to the full socket on SPR.
Dynamic beats static at B=4096 (0.634 vs 0.740) even on ONE socket — tail
volumes, not NUMA, wallaby is single-socket for 32 threads.  Prefetch flags all
lost at 32 threads (pf=0 pw=0 pt=0 picked): with every core's LFBs busy there
is nothing idle to prefetch into.  Sustained clocks with 32 cores active:
clk512/256 = 2.90/3.00 GHz against 3.9 single-threaded — a third of the
"missing" speedup at batch is just the all-core licence clock.

### B=1 decomposition (the honest account of where the other 29 cores go)

Dev knob `L17MT_SKIP` (bit0 planes / bit1 xrange / bit2 barrier, output wrong,
timing only) on wallaby, wall us per execute:

* empty job (release+collect): 0.32 @nt2, 0.56 @8, 1.01 @16, 1.41 @32
* planes only: 4.75 @2, 2.12 @8, **1.65 @17** (one plane per thread), 1.87 @32
  — scales to 17 threads, floored by plane granularity (a plane is ~0.5 us and
  Y->Z inside a plane is sequential by construction).
* xrange only (warm t1g): 4.23 @1, 3.21 @2, **2.08 @4**, then RISES: 2.81 @8,
  3.16 @16 — the X phase stops scaling at ~4 threads even on warm local data.
* barrier, threads arriving simultaneously: 0.77 @8, 1.56 @16, 3.53 @32
  (central atomic; real cost is lower because arrivals stagger).

Full-job B=1: 13.1 @nt1 (pool overhead over mode-0's 12.1), 7.65 @8, 6.41 @16,
8.1 @32; the winner picks nt~12–17 and lands at 5.3–6.4 us.  Two fixes tried:
* **xpf: prefetch chunk h+2's 17 rows in the X phase** (the r9 pt mechanism
  repointed at cross-core dirty lines): ~0.4–0.7 us win at nxr=4–8, ~nothing at
  nxr=nt; offered to the tuner, sometimes picked.
* **nxr < nt (smaller X-phase team)**: LOST on the full job (nt=16: nxr=2 8.36,
  nxr=4 7.99, nxr=16 6.39) although xrange-only said nxr=4 is best — with the
  planes phase actually run, the X phase's reads are dirty in 16 cores' caches
  and MORE readers pull them with more aggregate MLP.  The xrange-only probe
  (t1g clean/shared after rep 1) mismeasures exactly the traffic that matters.
  Kept as a raced knob since the node may answer differently.

So B=1 = 5.3 us against a ~1.3 us ideal-split floor: ~1 us handshake+barrier,
~0.5 us plane-granularity skew, and ~2 us of all-to-all coherence traffic
(87 KiB of t1g must cross cores between the phases; that is the transpose, it
cannot be deleted, only hidden).  B=1 does not parallelise past ~2.3x for this
kernel at this size — recorded as a measurement, not a surrender; round 2
ideas below.

### Borrowed from other entries

* The in-create() probe pattern (route machine answers out through
  `fft3d_description()`) — L36_pfa r8, carried over from phase 1 unchanged.
* The joint-knob tournament (never race interacting knobs sequentially) — my
  own r4 lesson, applied to the MT grid.
* No mt-phase rival records existed yet this round (context.md is empty).

### For the node (what I expect the scoring run to decide)

* B=256/4096: node is 2-socket CLX, driver first-touches in/out on socket 0;
  16 remote threads read everything over UPI.  Expect dyn=1/2 to beat static
  clearly (unlike wallaby), possibly nt=24 to beat nt=32, and NT+pipelined to
  win as it did here.  The sbw probe still rides the description string.
* B=1: close binding keeps nt<=16 on one socket; expect the pick at nt=8–16,
  ~2x over the phase-1 16.99 us.

### Next round

1. **B=1 all-to-all**: overlap the pull — have each X-phase thread prefetch its
   whole column range right at the barrier exit (bulk, not per-chunk), or
   restructure phase 2 so each thread starts on the columns of the planes IT
   wrote (locality-aware chunk order, zero extra traffic).
2. **Tree barrier / release in pairs** if the node's 2-socket barrier is worse
   than wallaby's (skip=3 numbers will not transfer).
3. **NUMA staging at batch**: if the node's remote-socket threads are starved
   (stage-B table will say), try having socket-1 threads stream their volumes
   through an L2 stage first-touched locally (the r10 staged-input mechanism,
   repointed at UPI).
4. If the leaderboard shows a rival's mt layer faster at batch, steal its
   schedule first — the kernel is already the phase-1 winner here.
