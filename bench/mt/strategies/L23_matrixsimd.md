# L23_matrixsimd -- strategy record (multicore phase)

Phase-1 history (how the kernel itself got here, rounds panel_r6..r11) lives in
`../../geom/strategies/L23_matrixsimd.md`.  This file starts fresh for the
32-core phase, as the brief prescribes.

## Round mt_r1

### Starting point

Carried-over single-thread kernel: dense 23x23 DFT per axis, conjugate-pair
folded, PINNED-CONSTANT kernel (whole matrix from 11 cosines + 11 sines in
registers), X-first schedule, flat 1064-double padded t1 planes.  Node
(Gold 5218): B=1 47.7 us (1.13x the one-FMA-port floor, arithmetically
closed), B=128 64.9 us/vol.  Wallaby ST reference: B=1 21.9, B=128 24.7,
B=256 28.2 us/vol.  Scored mt cases: B=1, 128, 2048.

### What I built (all of it new this round)

**1. Persistent spin pool instead of per-call OpenMP regions.**  First
measurement of the round, and the one that shaped everything: on wallaby one
GOMP `parallel` + `for`-barrier + join costs **6.2-8.2 us at T=8..32** --
several times the entire 32-way-parallel compute of one 23^3 volume
(~1.3 us).  A first OpenMP version of the fused path measured 8-10 us at
B=1; the arithmetic under it was ~1.3 us.  So `fft3d_create()` spawns 31
pthreads once (thread creation is setup, per the brief), pins each to the
CPU the harness's `OMP_PROC_BIND=close / OMP_PLACES=cores` mapping gives
(read back from one throwaway OMP region via `sched_getcpu()`, so whatever
GOMP would do, the pool replicates -- including under cpusets), and
`execute()` publishes a job and release-stores one atomic generation
counter.  Workers never sleep (the driver's timing loop is back-to-back and
the 32 cores are ours).  OpenMP is never entered again after create().

**2. Flag-array barriers, not a central counter.**  The first pool used a
central fetch-add sense barrier: measured **5.0 us** for dispatch+2
barriers+join at T=32 (standalone microbenchmark) -- 32 serialized RMWs on
one line per episode.  Replaced with: per-thread padded arrival flags (one
uncontended store each; main scans them, the loads overlap in the fill
buffers) plus a single release word.  Epochs derive from the job generation
(2*gen for the mid barrier, 2*gen+1 for the join), so a varying participant
count stays correct.  The JOIN needs no release at all: workers post their
flag and go straight back to the generation spin, because main cannot
dispatch again until it has seen every flag.  B=1 went 6.7 -> 5.6 us.

**3. Two decompositions, chosen by batch.**
* **batch >= 32 -- volume-parallel**: thread t owns volumes
  [nb*t/T, nb*(t+1)/T) and runs the settled single-core schedule on its own
  scratch (allocated and first-touched by the owning worker AFTER pinning,
  so it is NUMA-local on the two-socket node).  One join barrier per
  execute.  Per-thread t1 (191 KiB) sits in the core's private L2 exactly
  as in phase 1.
* **batch < 32 -- fused**: X pass = 132 independent chunk items per volume
  (the two overlapping tail slots are ONE item, so the identical-value
  overlap lanes stay in one thread), ONE mid barrier (every t1 plane needs
  every X chunk of its volume), plane phase = 23 independent planes per
  volume (Y into the thread's own plane buffer, Z into the caller's out
  plane).  Item loops run over ALL volumes at once: 2 barriers per execute
  total, not per volume.  Fused t1 arena = batch volumes (<= 5.9 MiB),
  first-touched by the pool with the same static plane partition the plane
  loop uses.

**4. NT stores re-raced at batch >= 32 -- and they finally win.**  Phase 1
rejected NT four rounds running at ONE core (latency-bound cell).  At 32
threads the batch cells sit at the bandwidth wall and NT deletes the
out-RFO third of the traffic.  Wallaby streaming walk (nv=416, T=32):
**nt=1 1.51 us/t vs plain 1.89-2.42 us/t** -- a 20-25% win, exactly the
predicted inversion.  The single-core streaming winner (plain + pf=2 +
pw=1) is now a loser here (1.90); pw is pointless under NT (no RFO to
hide), pf=2 is a wash (1.511 vs 1.515).  Canonical head = {nt=1, pf=0,
pw=0}.
* NUMA note: the driver fread()s `in` and memset()s `out` on thread 0, so
  ALL caller pages live on socket 0 on the node; the tuner arena is
  deliberately filled serially so create()-time racing sees the same
  placement the scored run does.

**5. Tuner, phase-1 discipline kept**: canonical-order walk, two full
sweeps, per-candidate min, hysteresis (>3% fused / >4% streaming, the
phase-1 streaming-noise lesson), per-candidate 1.5 ms licence warmup,
deterministic arena, pick/inc telemetry in the description string.  Fused
walk: T in {32,24,16,12,8}, a vol-parallel row (1 < B), a SERIAL row (the
phase-1 floor -- B=1 must never lose to 47.7 us by parallelising), and a
256-bit row.  Streaming walk: (width, nt, pf, pw) combos, 8 rows.

### Bit class (ONE, cmp-verified on wallaby, not assumed)

Parallelisation only reassigns whole chunks to threads; every output element
is computed in the same chunk with the same accumulation order at any thread
count, either width, either decomposition, NT on or off.  Verified by cmp:
B=1 across forced modes {fused, serial, vol-parallel, fused-256, vol-256}
all bit-identical; B=128 across {nt=1} vs {nt=0,pf=2,pw=1} vs {256-bit} vs
{T=7, a deliberately ragged partition} all bit-identical.  Repeatable
(bit-identical across runs) at B=1, 4, 8, 31, 32, 128, 2048.

### Operation count

Unchanged: 594 real flop/line, 943 kflop/volume, 297 vector FP ops per
chunk, 409 zmm chunks/volume.  Parallel overhead added per execute: 1
dispatch store + T flag stores + 1 release word (fused) or flags only
(volume mode); zero extra FP work.

### Measured (wallaby, Xeon 6448Y, 32 threads on one 32-core socket)

| case | this round | phase-1 ST (wallaby) | speedup | parallel eff |
|---|---|---|---|---|
| B=1    | 5.33-5.60 us | 21.9 us      | 3.9-4.1x | 12-13% |
| B=8    | 2.77 us/vol  | ~22 us/vol   | ~8x      | 25% |
| B=31   | 1.32 us/vol  | ~22 us/vol   | ~17x     | 52% |
| B=32   | 1.29 us/vol  | ~22 us/vol   | ~17x     | 53% |
| B=128  | 1.09-1.36 us/vol across windows (139-174 total; best window 3/3 at 142.7-143.3) | 24.7 | 16-22x | 50-69% |
| B=2048 | 2.05 us/vol (4203 total) | 28.2 (ST B=256) | 13.8x | 43% |

Wallaby picks split by regime exactly as the L3 argument says they should:
B=128 (arena 47.5 MiB, FITS wallaby's 60 MiB L3) picks {nt=0, pf=2, pw=1}
3/3 -- the RFO hits L3 and plain stores win; B=2048 (nv=416, true
streaming) picks {nt=1, pf=0, pw=0} with the 20-25% margin quoted above.
On the node B=128's working set (47.5 MiB vs 22 MiB L3) is genuinely
streaming, so the node should pick nt=1 at BOTH batched cells -- if its
B=128 keeps {nt=0,pf=2,pw=1} instead, that is the wallaby-L3 pick leaking
through and the tuner arena needs to be forced past the node's L3.

rel L2 3.767e-16 - 3.808e-16 everywhere (identical to phase 1 at B=1 --
same arithmetic).  AVX2 host (wombat): PASS, repeatable, B=4 26.9 us/vol.

Where the other 28 threads go at B=1 (the honest accounting the brief asks
for): the whole volume is only 132 X items + 23 planes.  Critical path =
X (~0.4 us) + mid barrier + longest plane (~1.0 us, and 23 planes across 32
threads idles 9 of them) + join + dispatch ~= 5.3-5.6 us measured, of which
~2-3 us is sync+dispatch floor and ~1 us is plane-granularity imbalance.
B=1 is latency-bound, not work-bound; it parallelises 4x, not 32x, and the
serial row in the tuner guarantees it can never regress below phase 1.
B=2048 is bandwidth-bound (NT pick); 128 at 1.11 us/vol on wallaby is
partly L3-resident (47.5 MiB arena vs 60 MiB L3) -- the node's 22 MiB L3
makes its B=128 a true streaming cell, expect it nearer its B=2048.

### What did NOT work (numbers attached)

* **Per-call OpenMP regions**: 8-10 us at B=1 against ~1.3 us of work;
  GOMP fork+barrier+join microbenchmarked at 6.2-8.2 us (T=8..32).  Dead on
  arrival for the latency cell; fine for batch (amortized), but the pool
  serves both, so OMP is gone from execute entirely.
* **Central fetch-add barrier**: 5.0 us per dispatch+2 barriers at T=32
  standalone; B=1 exec 6.7 us.  Flag-array + collector scan: B=1 5.6 us.
* **Plane-phase t1 prefetch (pt knob)**: burst-prefetching the 133 lines of
  the t1 plane before the Y pass (theory: lines sit dirty in 32 remote L2s
  after the transposing X pass).  Measured 6.45-6.68 vs 5.59-5.73 us at
  B=1: **~0.9 us of pure uop tax**, the same prefetch-on-resident-lines
  failure phase 1 hit (L36_pfa's +13%).  Kept env-only (L23_PT), default
  off, never tuned.
* **Note for interpreting tuner tables**: the serial row reads 32-51 us
  in-pool vs 21.9 us in phase 1 -- 31 spinning workers drop the all-core
  clock, and one wallaby window was outright slow-state (51.5, sd 30% --
  the third round this trap appears in this entry's records; never read one
  window).

### Borrowed from other entries

Nothing yet -- mt_r1 has no prior multicore records (context file empty).
From my own phase-1 record: the pinned kernel, X-first schedule, t1
padding, tuner discipline (hysteresis, licence warmup, two sweeps, serial
arena fill), NT-copy staging code, env-override pattern (L23_*), telemetry-
in-description pattern.

### Env knobs for forced A/Bs (node sets none)

`L23_T` (active threads), `L23_NT`, `L23_PF`, `L23_PW`, `L23_PT`,
`L23_MODE` (1=fused, 2=vol-parallel, 3=serial, 4=fused-256, 5=vol-256),
`L23_VERBOSE` (tuner tables).

### Node predictions (Gold 5218, 2 sockets x 16 cores -- unlike wallaby,
32 threads SPAN BOTH SOCKETS there)

* B=1: cross-socket barrier is slower; the T walk may pick T=16 (one
  socket) over T=32.  Expect ~6-12 us; the pick and the T tell us the
  node's barrier economics.
* B=128/2048: all caller pages on socket 0, so socket 1's 16 threads
  read/write over UPI and socket 0's memory controllers carry everything:
  expect clearly worse than wallaby's 1.1-2.1 us/vol, and expect nt=1 to
  matter MORE (RFO deletion also halves UPI write traffic).  Compulsory
  traffic 570 KiB/vol plain, 380 NT.
* Efficiency to report next round from the node leaderboard: cell vs
  47.7/T.

### Next round

1. Read the node's picks (T, nt/pf/pw, tune[pick/inc]) off the leaderboard
   description strings and re-head the walks to the node's choices.
2. B=1 sync floor: a two-level (socket-tree) collector for the mid barrier
   if the node picks T=16 over T=32 (i.e. if the cross-socket scan is the
   cost); potentially split X items over socket 0 only and planes over all
   32.
3. B=1 plane imbalance: half-plane items (46 items, pair-local Y/Z split
   with a 2-thread mini-sync) if the node shows the plane phase dominating.
4. Streaming: if the node's B=2048 sits far above 380 KiB/vol / (socket-0
   bandwidth), try interleaving the fused decomposition at large batch so
   socket-1 threads never touch `out` (socket-0 threads do all Z stores,
   socket-1 threads do X/Y) -- a NUMA-asymmetric split the volume-parallel
   scheme cannot express.
5. Port whatever the panel's other entries find this round; this record's
   pool + flag-barrier numbers are the reusable part for THEM (the GOMP
   6-8 us fork cost applies to every geometry's latency cell).
