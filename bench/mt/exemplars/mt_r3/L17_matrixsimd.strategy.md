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

## Round mt_r2 — arena fidelity for the node's pages, winograd's flat barrier, bulk X-phase pull, t1g re-homing

### Where round 1 left me on the node

Won B=1 (7.112 us, 3.21x MKL) and B=256 (0.726 us/t, but a 1.12x cross-process
pick spread: 199.2/208.5/185.9); LOST B=4096 to L17_winograd by 1.72x (2.106 vs
1.222 us/t) with my own create-time race having picked NT+pipelined at nt=24/32
static in all three processes.  The r1 VERDICT diagnosed the round's shared
defect and it is exactly my B=4096 story: the create-time arena was filled
SERIALLY ("to match the driver's first touch"), but the driver's
serially-touched pages do not STAY on socket 0 through the multi-second scored
loop — AutoNUMA migrates them toward the threads that keep faulting them
(L=6 B=65536 sustains 175 GB/s, far above one socket; winograd's plain static
schedule reached 129 GB/s).  So my arena priced a transient (all pages remote
to half the team) and the scored run ran a different machine (pages
owner-local under a static split).  L17_rader's description string is the
smoking gun for the same trap: its arena read 1.319 us/t for a config that
scored 2.904.  Everything below follows from taking that diagnosis seriously.

### What changed (all schedule/address/timing-only; every bit class untouched)

1. **Owner-touched tuner arena** (`l17_tune_alloc_mt`): the batch>=64 mt arena
   is now first-touched IN PARALLEL, volume v touched by the thread that will
   process v under the static split, THEN filled with values serially (first
   touch is per page at first write; later writes move nothing).  This is the
   AutoNUMA-migrated steady state, and it is also what the harness will
   produce if the monitor adopts the VERDICT's parallel-first-touch fix.
   Stage A now races under static dyn=0 (what the node picked panel-wide in
   r1) instead of dyn=2, whose only purpose was hiding the imbalance the old
   arena faked.
2. **Median-of-5 race statistic** in `l17mt_time_cfg` (was min-of-3), against
   VERDICT §3.2's pick lottery (nine entries, mine included at B=256, scored
   on their luckiest process).  A median flips only when the distributions
   actually cross.
3. **Flat arrival-flag/release barrier** for the mode-2 phase boundary,
   ADOPTED FROM L17_winograd mt_r1 (its record: central atomic ~1.2 us at
   T=16 — serialized RFOs on one counter line — vs 0.3–0.4 us flat; my own r1
   skip-probe read 1.56 us at nt=16).  Each arriver writes its own padded
   line, thread 0 scans (misses overlap) and publishes one release word; the
   sequence is the pool generation, unique per execute, so threads that sat
   out smaller-team dispatches can never be out of phase (winograd's
   derive-from-the-epoch argument, reused verbatim).
4. **Bulk X-phase prefetch, xpf=2** (the r1 VERDICT's named L=17 fix and my
   own r1 next-round item 1): at barrier exit each X thread pulls its WHOLE
   column range back-to-back (capped at 24 chunks ~26 KiB) so the cross-core
   dirty-line pulls overlap each other, not just two chunks ahead.  Raced
   against xpf=0/1.
5. **t1g re-homed after the team pick** (`l17mt_t1g_map`, now mmap'd so a
   re-touch gets fresh pages): the race needs t1g to exist, so it is first
   touched by the full-team map, but when the pick is nt<maxt (node r1:
   nt=16) that map leaves planes 8..16's pages homed on socket 1 while every
   user of them runs on socket 0 — a permanent cross-socket exposure this
   entry shipped in r1.  After the mode race, t1g is munmap'd, re-mmap'd and
   first-touched by the PICKED team's plane map, before the nxr/xpf race.
6. **anb=** in the description string: /proc/sys/kernel/numa_balancing read
   at plan time — the cheap check the VERDICT asked for, carried back on the
   scoring node's own JSON.

### Operation count

Unchanged: 40.8k vector FP ops per volume (~225.5 zmm-equivalents with the
mixed tail); the round adds zero arithmetic — a barrier, prefetches, and
page placement.

### Measured on wallaby (SPR, 32 threads one socket — where the arena fix is
### expected to be nearly invisible, since there is no second socket to misprice)

| case | mt_r1 | this round | note |
|---|---|---|---|
| B=1    | 5.26 us | **4.27–4.36 us** (–19%) | pick moved to nt=17 nxr=17 xpf=0: with the flat barrier, one-plane-per-thread granularity now wins |
| B=8    | 1.62 us/t | **1.40 us/t** | mode-1 nt=8 |
| B=256  | 0.385–0.53 us/t | 0.479 us/t (sd 6% — login-node noise) | |
| B=4096 | 0.84 us/t | **0.84–0.88 us/t** | same pick as r1: 512-bit C-parked pipelined+NT, nt=32 dyn=1 (stage A: NT+pipe 0.729 vs plain X-first 0.814) |

rel_l2 3.2e-16 everywhere, bit-repeatable across processes at every batch
tried (1, 8, 33, 64, 100, 256, 4096).

### What did not work / what lost, with numbers

* **xpf=2 (bulk pull) loses on wallaby at every nxr**: nt=17 nxr=17: 5.340
  vs 4.823 (xpf=0); nxr=12: 5.347 vs 4.990 (xpf=1).  On a single socket the
  per-chunk lookahead already covers the c2c latency, and the bulk burst just
  adds ~100 in-flight requests of L1 pressure up front.  KEPT as a raced
  candidate because the node's cross-socket pulls are several times slower
  and staggered arrivals mean the X phase starts while remote lines are still
  dirty-far — the mechanism the cap-at-24-chunks version is aimed at.  The
  node decides; wallaby already rejected it for itself.
* Nothing else regressed; the flat barrier dominated the central one in
  every B=1 config raced (it is not reported separately because the barrier
  is not a raced knob — winograd's numbers plus my skip-probe made central
  strictly worse).

### Borrowed this round, named

* **Flat arrival-flag/release barrier** — L17_winograd mt_r1, including the
  epoch-derived sequence trick.
* **The AutoNUMA/page-migration reading of the node** and the
  parallel-first-touch prescription — the mt_r1 VERDICT (§4 last paragraph,
  §6 harness item); the owner-touch arena is that prescription applied to
  the only pages I own.
* **Median-over-min for create-time races** — VERDICT §3.2's diagnosis;
  L6_unrolled's stability-vs-speed contrast is the datum that motivated it.

### Note for the monitor on the probes

The sbw[rd/wr/cp/s17] numbers at batched cells are now measured on the
OWNER-TOUCHED arena by a single-threaded prober, so on a two-socket node
they will read higher than r1's (the prober sees ~half-remote pages).  They
are still comparable between each other within the round; do not read the
r2-vs-r1 sbw delta as a machine change.  b1dec is unaffected (volume-sized,
socket-local).

### Pre-registered node expectations

* **B=4096**: the whole point of the round.  If the owner-touched arena
  re-ranks stage A toward the plain/addr-safe X-first family, expect
  1.2–1.4 us/t (parity with winograd's 129 GB/s schedule); if NT+pipelined
  survives on faithful pages it should score BELOW 1.2 (RFO deletion is
  236->157 KB/vol of compulsory traffic).  Anything near 2.1 again means the
  arena is still unfaithful in a way I have not modeled — in that case next
  round should copy winograd's per-thread serial `run_vols` schedule outright.
* **B=256**: expect the win held with the cross-process spread collapsed
  (median statistic + owner-touched arena).
* **B=1**: expect 5.5–6.5 us (barrier ~-1 us of the 7.11, t1g re-home removes
  the nt=16 cross-socket exposure; node clocks are lower than wallaby's).
  If nt=17 is picked on the node, note its thread 16 sits on socket 1 under
  close binding — the race prices that UPI barrier hop honestly now.

### Next round

1. If B=4096 still loses: adopt winograd's schedule wholesale (per-thread
   serial volume loop over its static block with per-thread 241 KB scratch)
   as a 22nd class-D candidate and let the race arbitrate — the VERDICT says
   the 1.72x was schedule, and the schedule is sitting in a promoted exemplar.
2. B=1 all-to-all: the locality-aware chunk ORDER (each X thread starts on
   columns of planes it wrote) is still untried — the bulk prefetch attacks
   latency, not order; if xpf=2 loses on the node too, reorder instead of
   prefetch.
3. If the harness adopts parallel first-touch/interleave, delete the
   owner-touch code path comment and re-verify the arena still matches (it
   does by construction for parallel first-touch; interleave would want an
   interleaved arena fill instead).

## Round mt_r3 — merge, don't diverge: the winograd rotating-pass engine grafted in for the streaming regime, installed by a working-set gate instead of a raced pick

### Where round 2 left me on the node

Won B=1 (6.163 us, 3.67x MKL, inside my pre-registered 5.5–6.5 band) and
B=256 (0.755 us/t, 5.17x, spread collapsed as promised).  LOST B=4096 again,
and worse: 2.106 -> 2.858 us/t (a stable 1.36x self-regression, all three
processes agreeing), against L17_winograd's 1.220 — identical to ITS r1
number.  My r2 pre-registration said "anything near 2.1 again means the
arena is still unfaithful in a way I have not modeled — in that case next
round should copy winograd's schedule outright."  It scored 2.858.  This
round does exactly what the pre-registration and the mt_r2 VERDICT (L=17
action item: "merge, don't diverge ... the threshold set from the working
set rather than from an arena") ordered.

The decisive forensics, from the r2 JSONs and VERDICT:

* My node pick at B=4096 was plain 256-bit X-first, nt=32 STATIC, dyn=0 —
  the same schedule shape as winograd's winner — and it still ran 82.6 GB/s
  effective against winograd's ~193 (1.220 us/t at 236 KB/vol).  So the gap
  is NOT team shape, NOT dyn, NOT NT stores: it is the kernel's DRAM access
  pattern itself.  VERDICT §4 item 4 says it flatly: "~2x of each gap is
  schedule, not machine."
* Winograd's own r2 pick lottery is instructive: its dyn=2 process scored
  2.115 (my territory), its dyn=0 h4/i4 processes 1.220/1.228.  The engine
  only streams when static; dyn lost in every process that picked it,
  panel-wide (VERDICT §6: remove dyn from the tuner surface, do not re-race).
* My own create-time arena has now misranked the streaming cell twice
  (r1: NT+pipelined arena-picked, scored 2.106; r2: plain X-first
  arena-picked on the owner-touched arena, scored 2.858).  Wallaby cannot
  see the collapse either — my dense path measures 0.84–0.88 us/t there,
  BEATING winograd's engine (~1.08).  Both instruments are blind at this
  cell; only the node's scored numbers are trustworthy, and they say
  winograd's engine, static, plain stores, 1.220, twice in a row.

### What was built

1. **The streaming engine, adopted wholesale from L17_winograd** (named
   attribution, its exemplar and source): pass1_f4/pass1_i4 (72 4-wide
   groups deinterleaving `in` into split re/im kx-blocked scratch, kernel
   B), tail1_f4, and fused23_h4 (kernel H component-split fused pass 2+3,
   one kx-block at a time out of an L2-resident mini-buffer, kernel E tail,
   tsto4 transposed stores to `out`), plus kernels A/B/E/H, the tst4/tsto4/
   tst4f transposes and the store-base tables — ~1240 lines copied verbatim
   from impl/L17_winograd.c.  Two local changes, both marked in the source:
   fused23_h4's plan indirection replaced by explicit scratch/table
   arguments, and its pf/pfw/CLWB hooks dropped (the node's winning
   processes ran pf=0 pfw=0 cw=0; hints only).  The engine runs as a mode-1
   exec on MY existing spin pool: each pool thread serially processes its
   contiguous static block of volumes out of its OWN 157 KB scratch
   (4 x 17*296 doubles, page-aligned, first-touched by its own thread), so
   winograd's schedule inherits my cheaper dispatch (no OMP region per
   execute).
2. **The working-set gate** (deterministic, never raced): at batch >= 64,
   if in+out = batch * 157216 B exceeds 1.5x the aggregate LLC the team can
   reach, the engine is INSTALLED — no stage A/B/C races run at all.
   Aggregate LLC = sysconf per-package L3 x distinct physical_package_id
   over the team's captured affinity masks (a pure sysfs read), floored at
   96 MB if the machine will not answer.  Node: 22 MB x 2 -> threshold
   66 MB, so B=256 (40 MB) keeps the dense X-first path that WON that cell
   and B=4096 (614 MB) gets the engine; wallaby: 60 MB x 1 -> 90 MB, same
   split.  Same machine + same batch => same plan in every process: the
   pick lottery at this cell is dead by construction.
3. **i4 (split-free pass 1) as the shipped variant**: pass 1's row loads
   sit on the 16 B alignment classes of the 4624 B row stride, and at w=4,
   8 of 34 loads per group split a cache line; on the node's 2-load-port
   CLX a split on a COLD line holds fill resources for two fills.  i4
   re-issues exactly those 8 loads as xmm pairs (identical values —
   winograd cmp-verified i4 == h4).  Node evidence: winograd's i4 process
   1.228 vs h4 1.220 (a tie), but L17_rader's same mechanism ("dy") was
   node-PICKED at batch (-2.8%); wallaby A/B here: i4 5037 vs h4 5110 us
   back-to-back under ~4% login noise (a tie leaning i4).  Dev overrides
   L17MT_WG=0/1 and L17MT_WGI4=0/1 exist for A/Bs; the harness never sets
   them.
4. **Below the threshold, nothing changed**: B=1 (mode 2 intra-volume),
   B=256 (mode-1 races on the owner-touched arena) are byte-for-byte the
   r2 paths.

### Operation count

Below the gate: unchanged (40.8k vector FP ops per volume, ~225.5
zmm-equivalents).  Above the gate: the engine does MORE arithmetic —
3*289*296 = 256,632 FP instructions / 423,096 flops per volume (winograd's
count) against my dense form's ~301k flops — and wins anyway, because at
82-vs-193 GB/s the cell's price is traffic pattern, not flops.  Worth
saying plainly: above the aggregate-cache line, arithmetic is free and the
access pattern is everything.

### Measured on wallaby (SPR, 32 threads one socket; a NOISY session, sd up to 5% — treat batched numbers as relative)

| case | mt_r2 | this round | note |
|---|---|---|---|
| B=1    | 4.27–4.36 us | 3.87 us | unchanged path (intra nt=17); quiet-window luck |
| B=256  | 0.479 us/t | 0.376 us/t | unchanged path (vol nt=32 static) |
| B=1024 | — | 0.898 us/t | wg engine (ws=157MB > 90MB) |
| B=4096 | 0.84–0.88 us/t (dense) | **1.097 us/t (wg engine)** | a deliberate WALLABY regression: dense forced via L17MT_WG=0 in the same session read 1.84 us/t under load, and r2's 0.84 was a quiet window; the node data (2.858 dense vs 1.220 engine, both stable, two rounds) overrules wallaby here |

rel_l2 3.226e-16 (B=1), 3.258-3.259e-16 (batched); bit-repeatable across
processes at 1, 64, 256, 1024, 4096.  Setup at B=4096 dropped 0.88 s ->
0.19-0.24 s (no arena race above the gate).

### What did not work / what was rejected, with the numbers

* **Fixing my own dense kernel's streaming behaviour**: rejected for this
  round without an attempt, deliberately.  Both available instruments are
  blind — my arena priced plain X-first at rank 1 while the driver scored
  it 2.858, and wallaby says the dense path WINS the cell (0.84 vs 1.10).
  Any "fix" would be tuned against instruments that have been wrong twice.
  The engine's 1.220 x2 rounds is the only number at this cell that has
  ever reproduced.
* **Racing the engine against the dense class-D candidates**: rejected on
  VERDICT §4.6 ("racing a surrogate is not searching") and my own r1/r2
  arena history.  If the arena says dense 0.8 vs engine 1.1 (as wallaby
  does), and the node says dense 2.86 vs engine 1.22, a 20% incumbent
  margin would not have saved the pick.  Hard gate, no race.
* **h4 over i4**: wallaby cannot price cold splits (3 load ports, DDR5) and
  read a wash; kept i4 on the CLX fill-buffer mechanism + rader's node
  pick.  If the node disagrees it costs <=1% (its own h4/i4 processes were
  0.7% apart).

### Borrowed this round, named

* **The entire streaming engine** — L17_winograd (pass1_f4/i4, kernels
  B/E/H, fused23_h4, tst4/tsto4/tst4f, table layouts), from its mt_r2
  exemplar source.  The i4 pass 1 is winograd's port of **L17_rader's**
  "dy" split-free loads, so the mechanism is rader's by two hops.
* **The working-set-gate-not-arena-race prescription and the dyn removal**
  — the mt_r2 VERDICT (§6 L=17 item, §5).

### Pre-registered node expectations

* **B=4096**: 1.15–1.35 us/t, all three processes within ~2% (no raced
  knobs remain at this cell).  The engine is winograd's 1.220 schedule on a
  cheaper dispatch; anything above 1.4 means the pool's static split
  interacts with page placement differently than winograd's OMP region and
  I will A/B the dispatch next round.  Description will read
  "mt[wg nt=32 static ws=614MB aggl3=44MB]" — if aggl3 does not read 44 on
  the node, the sysfs path failed and the 96 MB floor carried the gate.
* **B=256**: 0.72–0.80 us/t, the unchanged r2 path (its pick was stable).
* **B=1**: 6.0–6.4 us, unchanged path.
* If the harness adopts the VERDICT's parallel-first-touch fix, the node's
  streaming regime changes and my dense NT+pipelined path may deserve a
  re-hearing at B=4096 — that is a NEW measurement question for the round
  after, not a reason to race now.

### Next round

1. If B=4096 lands in band: try shaving the engine further — the pool
   dispatch is ~1 us cheaper than an OMP region per call but the split is
   still 128 volumes/thread; a socket-aware split (touch order = fault
   order) or the cw paced-CLWB hook (dropped this round, winograd r11
   measured it mixed) are the remaining levers.  Also A/B NT staging inside
   the engine (winograd's nt rows) — its node procs picked nt=0 under
   static, but my pool + i4 may move the balance.
2. If B=256 ever destabilizes: the gate can take that cell too (drop the
   1.5x to 1.0x); on r2 evidence the dense path is 1.4x faster there, so
   only do this on a measured loss.
3. B=1 remains mine (6.163 vs rader 6.776): the locality-aware X-phase
   chunk order (each X thread starts on columns of planes it wrote) is
   still untried and is the only idea left that attacks the ~2 us
   all-to-all floor.
