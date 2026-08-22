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

## Round mt_r2

### Where mt_r1 landed on the node (what this round is aimed at)

Node (Gold 5218, 2x16 cores, 22 MiB L3/socket): B=1 **11.835 us** (won,
pick = fused T=16 -- one socket; tuner pick 11.62-11.88 vs inc 12.7-13.0,
honest), B=128 **2.287 us/vol** (won, pick = plain vol-par T=32), B=2048
**7.235 us/vol -- LOST** to L23_rader's 6.052.  The raw t_*.json files and
the monitor's verdict show exactly why: my three processes all tuned at
nv=160 (59 MiB arena) and picked {nt=0,pf=2,pw=1}, with the nt=1 head
measured 4.6 vs 2.5 us/t -- but 59 MiB sits INSIDE the node's **76 MiB
aggregate cache** (32x1 MiB L2 + 2x22 MiB L3), so the arena priced out-RFOs
at L3 speed.  L23_rader's one process that happened to pick batchNT ran the
real 760 MiB batch at **6.05 us/vol vs 7.18-7.24 plain** (theirs AND mine).
The kernel was never the problem; the arena lied about the regime.  Same
mechanism the verdict documents for L8_batchsimd (stuck at exactly MKL's
cached-store roof) and L45_mixedradix (arena 25.4 vs driver 79.1 us/t).

### What changed (this round is tuner-surface and decomposition, zero new FP)

1. **Aggregate-cache tuner arena.**  `l23_tune_nv` now sizes off
   `nthr*L2 + nsockets*L3` (sysconf, x3.5, cap 640 volumes, multiple of
   32), not one socket's L3.  Node: nv 160 -> 512 (190 MiB arena = 2.5x
   the 76 MiB aggregate).  Wallaby: nv 416 -> 640.  BORROWED: L6_pfa's
   "single biggest lesson" and the mt_r1 verdict's §5 corollary verbatim.
2. **Socket map + asymmetric NT.**  Each pool thread's
   `physical_package_id` is read from sysfs for its pinned CPU at
   create(); `far[tid]` = not tid 0's package (the driver first-touches
   both caller buffers on thread 0, so tid 0's socket owns the pages).
   The streaming walk gains nt=2 (NT far socket only -- a far plain store
   pays RFO *and* dirty writeback across UPI; NT is one line transfer) and
   nt=3 (NT near only, the UPI-hates-NT counter-hypothesis).  Rows are
   skipped when nsock=1, so wallaby behavior is unchanged.  All nt codes
   write identical bits (cmp-verified) -- the class is intact.
3. **Streaming walk re-headed**: {nt1 pf0} head, then {nt1 pf2} (the combo
   L23_rader's lucky node process actually ran at 6.05 -- BORROWED), then
   the old {nt0 pf2 pw1} and plain rows, the two asymmetric rows, 256-bit
   NT sanity.  Dropped the four never-winning plain-prefetch permutations.
4. **B=1 plane-imbalance attack**, all as RACED candidates, nothing forced:
   * T=23 in the fused ladder (23 planes set the natural team --
     L23_rader's wallaby B=1 table, BORROWED);
   * the pt knob (burst-prefetch of the t1 plane before the Y pass) is
     finally in the walk at T=32/16.  It was +0.9 us of pure uop tax on
     wallaby's resident lines in mt_r1, but it was BUILT for cross-L2
     dirty-line handoff, which wallaby cannot exhibit and the node's
     fused T=16 pick says is the cost (same mechanism as
     L17_matrixsimd's xpf, which won 0.4-0.7 us on small teams);
   * **fus3, a split-plane fused mode**: X items, barrier, Y chunk items
     into a SHARED per-plane buffer arena (batch*23 x 1152 doubles,
     64-aligned planes, 384-B rows so Y items never share a line),
     barrier, Z chunk items into the caller's out planes.  5 Y + 5 Z
     items per plane (the two overlapping tail chunks stay one item, same
     discipline as the X pass) instead of one 12-chunk plane item: cuts
     the ceil(23/T) round imbalance -- at the node's T=16 the critical
     path is 2 whole planes today.  Price: one more barrier, and pb lines
     crossing L2s (a Z item reads 4 columns of every Y item's rows).
5. Plumbing: barrier epochs are now 4*gen+k (two mids + join) so fus3's
   second mid barrier fits the monotone-epoch scheme; `l23_time_cand`
   carries pt; env knobs gain `L23_MODE=6/7` (fus3 w4/w2) and `L23_NT`
   accepts 0..3.  Description string now prints nsock and pt.

### Operation count

Unchanged: 594 real flop/line, 943 kflop/volume, 409 zmm chunks.  fus3
reassigns the identical chunk calls (same operands, same accumulation
order) and adds one barrier episode; the asymmetric-NT rows change which
threads take the `l23_ntcopy` staging path, zero FP either way.

### Bit class (ONE, cmp-verified on wallaby this round, not assumed)

B=3 full-output cmp across forced {tuner pick, fus3, fus3-w2, fused,
fused-w2, serial, vol-parallel, fus3 T=5 (ragged), fus3 T=23}: all
bit-identical.  B=128 cmp across {nt0, nt1, nt2, nt3, nt1+pf2, pf2+pw1,
w2+nt1, T=7 ragged}: all bit-identical.  PASS rel_l2 3.77e-16..3.81e-16 at
B = 1, 3, 8, 31, 32, 128, 2048; repeatable (bit-identical across runs)
everywhere.

### Measured on wallaby (Xeon 6448Y, one socket, 32 threads; quiet windows)

| case | mt_r1 | mt_r2 | phase-1 ST | speedup | parallel eff |
|---|---|---|---|---|---|
| B=1    | 5.33-5.60 us | 5.31-5.61 us (fused T=32 kept) | 21.9 | 3.9-4.1x | 12-13% |
| B=8    | 2.77 us/vol  | 2.45 us/vol  | ~22 | 9.0x | 28% |
| B=31   | 1.32 us/vol  | 1.19 us/vol  | ~22 | 18x | 58% |
| B=32   | 1.29 us/vol  | 1.09 us/vol  | ~22 | 20x | 63% |
| B=128  | 1.09-1.36    | 1.08-1.12 us/vol | 24.7 | 22-23x | 71% |
| B=2048 | 2.05 us/vol  | 2.01-2.06 us/vol (nt1 pf2 kept) | 28.2 (ST B=256) | 14x | 44% |

Wallaby is one socket, so the two changes this round exists for
(asymmetric NT, honest two-socket arena) CANNOT show a win here -- parity
with mt_r1 plus the small B=8..32 gains is the expected wallaby result,
and the round's real test is the node's pick telemetry.

B=1 fused walk, one quiet window (us/t): fused T32 **5.31 <- kept**, T24
5.51, T23 5.92, T16 7.64, T12 7.90, T8 10.78; fused+pt T32 5.87, T16 7.87;
fus3 T32 6.68, T23 6.39, T16 8.12; serial 31.9 (in-pool clock, see mt_r1
trap note); 256-bit fused 5.57.
B=2048 streaming walk (nv=640, us/t): nt1 pf0 1.707, **nt1 pf2 1.613 <-
kept**, nt0 pf2 pw1 2.330, plain 2.880, w2 nt1 1.896.  Even on ONE socket
the honest arena moves the plain-vs-NT gap from mt_r1's "wash" (1.51 vs
1.90 at nv=416) to a clean 1.4-1.8x -- at nv=640 the L3 assist is gone and
pf2 now pays under NT (1.61 vs 1.71; mt_r1's nv=416 read it as a wash).

### What did NOT work on wallaby (numbers attached; node may disagree, raced there)

* **fus3 at B=1**: 6.39-6.68 vs 5.31 us for classic fused at T=32 -- the
  extra barrier plus the within-plane pb all-to-all (a Z item pulls 4
  columns from every Y item's rows, cross-L2) costs ~1.1-1.4 us and the
  imbalance saving at T=32 is only ~1 plane round.  Kept as node rows:
  at the node's T=16 the imbalance term is 2x bigger (2 rounds of ~1.4 us)
  and its cross-core costs are what the race is for.
* **pt at B=1 (wallaby)**: 5.87 vs 5.31 at T=32, 7.87 vs 7.64 at T=16 --
  same resident-line uop tax as mt_r1.  Unchanged conclusion HERE; the row
  exists for the node.
* **T=23 fused (wallaby)**: 5.92 vs 5.31 at T=32 -- unlike L23_rader's
  kernel, mine loses width faster than it gains balance on SPR.  Node
  re-races it against its T=16 pick.
* The raw-ssh trap (L23_rader's record documents it five times): hit it
  again this round -- remote one-offs need `cd` by absolute path (or the
  script's own dirname-cd) and `&&`-chains, and a `test -s` before any cmp.

### Borrowed this round (attributions)

* **Aggregate-cache arena sizing**: L6_pfa mt_r1 ("single biggest lesson",
  -21% recovered) + the monitor's verdict §5 corollary; confirmed against
  my own B=2048 loss mechanism in the raw node JSONs.
* **NT+pf2 as the streaming head candidate and the 6.05 us/vol anchor**:
  L23_rader mt_r1's lucky process (their batchNT pf2 cell).
* **T=23 fused row**: L23_rader mt_r1 wallaby table (t23 9.07 vs t32 9.83
  on their kernel).
* **Racing the cross-core prefetch instead of assuming it**: the xpf
  result in L17_matrixsimd mt_r1 (0.4-0.7 us win on small teams; their
  "xrange-only probe mismeasures the traffic that matters" warning is why
  fus3/pt are raced on the full job, never on a phase in isolation).

### Node predictions (pre-registered)

* **B=2048**: all three processes should now tune at nv=512 (190 MiB,
  truly past the 76 MiB aggregate) and pick nt1 (pf2 or pf0) ->
  **~6.0-6.3 us/vol reproducibly** (the 6.05 anchor), vs 7.235 in mt_r1.
  If nt=2 (far-only NT) wins instead, the near socket's L3 assist is real
  even at 760 MiB; I expect full nt1.
* **B=128**: arena = real batch (honest), plain should hold; asymmetric
  rows might shave a few % if far-RFO deletion pays inside the combined
  L3.  Expect 2.2-2.3 us/vol, pick {nt0} or {nt2}.
* **B=1**: the walk now asks the node three new questions (T=23, pt,
  fus3).  If the T=16 pick was cross-socket data movement, T=16 stands
  and pt/fus3-T16 are the candidates that can move it below ~11 us; if it
  was barrier economics, fus3 loses (extra barrier).  Honest band:
  10.5-12 us; anything below 11 means one of the new rows won.
* Read the picks off the description strings: nsock=2 confirms the socket
  map worked on the node; nt/pt/T tell the rest.

### Next round

1. If B=2048 lands ~6.0: the remaining gap to the machine (verdict: this
   cell ran 64 GB/s where L=6 reached 175) is read-side -- try staging
   far-socket volumes' `in` through local scratch (L23_rader's idea 2,
   L17_matrixsimd's r10 staged-input mechanism) to convert strided UPI
   reads into one sequential stream.
2. If the harness fixes caller-page placement (the verdict's #1 item),
   re-head everything: interleaved pages double the streaming ceiling,
   change the NT margin, and make my serial arena fill WRONG -- switch the
   arena fill to mimic whatever the driver then does, first thing.
3. If fused T=16 still holds B=1 with pt/fus3 losing, the floor is
   dispatch+2 barriers on one socket; next lever is a one-socket X phase
   with a 32-thread plane phase (socket-tree barrier so the far 16 join
   only where they help), or accepting B=1 as latency-closed and writing
   the measurement down.
4. Whoever wants the fus3 shape: it is only worth taking together with
   the shared-pb false-sharing layout notes in the code comments.
