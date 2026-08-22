# L13_rader — strategy record (multicore phase)

Phase-1 history (how the kernel itself got to 186 FP/pt Rader-13 with the fused
zy schedule and the in-plan race instrument) is in
`../../geom/strategies/L13_rader.md`. This file starts at the multicore phase.

## Round mt_r1 — first parallelization: batch split + NT stores, B=1 raced

### What changed

The phase-1 kernel (x/z/y passes, 186 FP per 13-point transform, 3×169
transforms per volume — arithmetic unchanged this round) got three multicore
mechanisms, all settled by measurement rather than by belief:

1. **Batch-parallel path (B>1).** Contiguous static split of the batch over
   up to 32 OpenMP threads (`ntb` of them; see 3), each thread running the
   phase-1 per-volume pipeline (extracted into `l13r_vol()`, scratch passed
   explicitly) on its **own scratch replica**: allocated, 64B-aligned and
   memset **inside create()'s parallel region**, so first touch puts every
   thread's A volume / U planes / staging on its own socket. Contiguous
   chunks (not round-robin) so each thread touches the same volumes every
   timed call and the cross-volume `pf` prefetch stays inside its own chunk.
   The caller's `in`/`out` are freaded/allocated by the driver's main thread
   → socket-0 resident; not ours to fix, same handicap for every backend.
   The pf/pw cache gates were rebased from whole-batch footprint to the
   **per-thread slice** (B=512 is 36 MB serial but 1.1 MB per thread — a
   completely different regime).

2. **NT store mode (`nts`) for the DRAM-streaming cells.** The y pass stages
   each volume into a per-thread L2-hot 35 KB buffer, then one
   `_mm512_stream_pd` burst copies it to `out` (scalar head/tail — a 13³
   volume is 549.25 cache lines, so volume bases are only 16B-aligned; one
   sfence per thread per execute). This deliberately **overturns the phase-1
   panel_r5 verdict** ("hide the RFO with prefetchw, don't NT-avoid it") —
   that verdict was the single-core latency/concurrency regime. At 32
   threads the streaming cells are aggregate-bandwidth bound and the RFO
   read of `out` is a third of the out-side traffic. Measured on wallaby at
   B=8192: **3731 → 2948 µs (−21 %)**; the in-plan race read nt-off at
   +29 % (463 vs 353 ns/vol). Gated on the batch streaming past both
   sockets' L3 (4× the sysconf socket value), and raced at batch either way.

3. **Everything node-undecidable is raced in-plan** (the panel_r11 race
   instrument, extended). New raced knobs this round:
   - `t1` — B=1 intra-volume team size ∈ {1(serial fused), serial unfused,
     2, 4, 8}. The threaded B=1 path is a 3-phase split (x lane-blocks / z
     row-blocks / whole y planes) with two barriers; the two clamped tail
     blocks overlap their neighbor in lanes, so partitions keep each tail
     with its neighbor block, and U went 16-deep (um=15) so all 13 planes'
     U stay live across the z→y barrier.
   - `nts` — toggled at batch (race bound raised from 2048 to 8192; the
     private race buffers are 1.2 GB at B=8192, acceptable for setup).
   - `ntb` — 16-vs-32 team at batch: on the node all of `in`/`out` sits on
     socket 0, so whether 16 cross-socket threads pay for the UPI hop is a
     node-only question wallaby (single 32-core socket) cannot answer.
   - `pf` is now toggled both ways at batch (was: only off-when-on).

### Measured (wallaby, Sapphire Rapids, OMP_NUM_THREADS=32, close/cores)

| cell | mt_r1 | phase-1 serial (same host, r8/r11 readings) | scaling |
|---|---|---|---|
| B=1 | **3.22 µs** (serial fused picked) | ~3.2 µs | 1× (see below) |
| B=512 | **104.4 µs = 0.204 µs/vol** | 1923 µs = 3.76 µs/vol | **18.4× (58 % of 32)** |
| B=8192 | **2948 µs = 0.360 µs/vol** | ~5.26 µs/vol (B=2048 r8) | **~14.6× (46 %)** |

rel_l2 = 4.0e-16 at all three cells, repeatable-bit-identical across runs.
B=512 race readings: i:206 pw!:204 pf!:211 nt!:273 → incumbent (pw within
noise of the 3 % bar, NT correctly ruinous when the batch is cache-resident).
B=8192 race: i:353 pw!:357 pf!:429 nt!:463 n16:591 → incumbent; the
cross-volume `pf` prefetch is worth **30 %** in the threaded streaming
regime (it was ~1–4 % serial) — more threads means more outstanding misses
to hide, exactly the brief's fill-buffer argument.

Missing 42 % at B=512: one parallel region fork/join per execute over 104 µs
of work, plus login-node contention (wallaby dev numbers are relative; the
sd on B=1 medians is 30 %+ since the harness now pins us to the busiest
cores — trust the min).

### What did not work, with the numbers that killed it

- **Threaded B=1 on wallaby**: t2 = 19.4 µs, t4 = 13.7 µs, t8 = 14.3 µs
  against 6.3 µs serial in the same race window. Parallel-region entry +
  two barriers dwarf ~6 µs of work on a contended login node. This is the
  brief's predicted outcome for B=1; the path stays in the binary and the
  **node races it at plan time** — if the idle node's t4 also loses, the
  race keeps serial and the B=1 cell ships the unchanged phase-1 kernel.
- **n16 (half team) at B=8192 on wallaby**: 591 vs 353 ns/vol — but wallaby's
  32 threads are one socket, so this measured nothing about UPI; kept in the
  race for the node.
- **NT at B=512**: 273 vs 206 ns/vol (+33 %) — the gate keeping NT off in
  cache-resident cells is load-bearing, do not widen it.

### Borrowed

- The in-plan timed race pattern is my own phase-1 lineage (L13_direct r10 ←
  L6_unrolled r9 ← L36_pfa r8, incumbency rule from L6_pfa) — extended here
  with team-size and NT variants. No other mt_r1 entry existed yet to borrow
  from (context file empty; I am apparently the first mt round writer).

### Next round

1. Read the node's `ab[...]` strings off the leaderboard description — they
   ship the raced readings. If t4 is within ~2× of serial at B=1 on the node,
   build the **one-barrier variant**: partition x and z by y-ranges (a z row
   (kx,y) needs only x lanes m ∈ [13y,13y+13), so per-thread x→z needs no
   barrier), and/or a create()-time **spin-pool with a sense-reversing
   barrier** to beat libgomp region entry; the brief explicitly blesses
   thread pools in create().
2. If n16 wins on the node at B=8192, try 24 and rebalance chunk sizes by
   socket (socket-0 threads get bigger chunks since their reads are local).
3. B=512 is the cell where the missing 42 % lives and the working set is
   per-thread L2-resident — profile whether it is fork/join or the x pass's
   strided reads; a persistent-team execute (single region, flag handoff)
   would amortize the fork.
4. If the monitor's node run shows nts losing at B=8192 (Cascade Lake WC
   buffers are weaker than SPR), flip with -DL13R_FORCE_NT=0 — but the
   in-plan race should already have caught it.

## Round mt_r3 — rebuild from the mt_r2 link failure; pool, two-phase B=1, carried NT appender

### Where mt_r2 actually left this entry (read this before trusting the code)

mt_r2's implementer wrote the plan-side scaffolding (pool race, pt/disp/asym/nv
knobs, an `l13r_ntcopy` call) but never defined five symbols —
`l13r_pool_new/del/nap`, `l13r_dwell`, `l13r_ntcopy` — and never wrote the
execute paths those knobs select. The entry FAILED TO LINK, scored nothing in
mt_r2 (VERDICT §3.1: the only build failure of nineteen; agent exit=134), and
left no mt_r2 section in this file. The B=8192 cell we won in mt_r1 (0.976
µs/vol) fell to L13_direct/fftw. Lesson encoded here for every future round:
**run tryout.sh after the last edit, not just the promising middle ones** —
an entry that does not build is worth exactly zero, and the round had real
ideas in it that died unmeasured.

### What changed (mt_r3)

1. **The five missing symbols now exist.** The persistent spin team is
   L17_winograd mt_r1's design taken whole: workers created in create(),
   parked on an epoch spin, flat arrival-flag/release barrier (each arriver
   writes its own padded line, thread 0 scans and publishes one release word;
   their record: ~0.3–0.4 µs/barrier at T=16 vs ~1.2 µs for a central atomic
   counter — did not re-derive). Workers pinned to the exact cores the
   harness's OMP close/cores binding reports (masks captured inside create()'s
   parallel regions). A `nap` mode (nanosleep park) keeps spinners from
   sitting under the serial variants' timings. Pool torn down in
   fft3d_destroy (the mt_r2 sketch leaked it).
2. **B=1 two-phase split (pt) actually runs**: thread t owns y-range [y0,y1),
   does its x lane-blocks AND its z row-blocks back to back with NO barrier
   between (a z row (kx,y) needs only x lanes m ∈ [13y,13y+13), thread-local
   by construction; boundary x blocks recomputed bit-identically by both
   neighbours; the tail z block clamps into the thread's OWN row list). One
   barrier, then whole y planes. Per-team block tables for T ∈ {2,4,8} built
   at plan time; raced behind a memcmp bit-identity gate.
3. **The batched NT path is now a carried-partial-line streaming appender**
   (adopted from L13_direct mt_r2, who measured their appender 5% under my
   mt_r1 per-volume NT copy on the same host): the y pass stages each plane
   into the L1-hot 2.7 KB sb buffer and appends it to ONE continuous NT
   stream over the thread's contiguous out range; partial cache lines are
   carried across plane and volume junctions, so the whole range pays 2
   partial-line WC flushes instead of 2 per volume, and the NT drain
   naturally interleaves with compute one plane at a time. This replaces
   both mt_r1's whole-volume copy and mt_r2's planned nts=2 overlap drain
   (deleted, with the 70 KB/thread sv staging buffer).
4. **Pool dispatch at batch (disp) actually runs** — same range body as the
   OMP path (bit-identical by construction), raced sequenced (OMP timed with
   no pool alive, then the pool after a dwell; L6_pfa mt_r1's pattern), now
   at every batch, not just ≤2048. **asym (3:2 socket-weighted split) and
   n24 are real partitioning modes**, raced at batch for the node.
5. **Race semantics flip for the pool knobs: adopt-unless-vetoed.** The old
   >3%-adoption bar measured on wallaby: arena read p8 at par with serial
   (6019 vs 6047 ns) while the DRIVER-level min said p8 −9% (3.93 vs 4.33
   µs); at B=512 the arena priced the pool +5% while the driver min said
   −26% (79.4 vs 106.9 µs). Cause: napped/parked workers lose their cores to
   other login-node jobs before the pool laps run, so the arena
   systematically overprices the pool under contention. Pool teams and pool
   dispatch are now adopted unless the incumbent beats them by >3% in both
   trial blocks; the bit-identity gate still guarantees a flip cannot change
   results, and the OMP t2/t4/t8 fallback teams stay raced alongside so a
   vetoed pool still leaves a threaded candidate priced. On the idle node
   (cell spread ~0.3%) the arena readings should be truthful either way.
   `-DL13R_FORCE_PT=n` / `-DL13R_FORCE_DISP=1` are the monitor's A/B knobs.

### Operation count

Arithmetic unchanged: 186 vector FP per 13-point transform, 3×169 transforms
per volume, zero extra FP in any threading mode (boundary-block recomputes at
B=1 add ≤2 x-blocks and ≤7 z-rows per team, ~2% of one volume's kernels,
B=1 only). NT tier per-volume traffic: 13 planes × 2.7 KB staged L1-hot and
re-read for the burst; compulsory DRAM stays 70 KB/volume (in 35 + NT out
35, no RFO).

### Measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, shared login node — driver-level min; medians carry up to 20% scheduler noise)

| cell | mt_r1 | mt_r3 | note |
|---|---|---|---|
| B=1 | 3.22 µs (serial) | **3.236 µs** (pool p8 adopted, sd 0.09%) / 4.05–4.33 when a noisy create vetoes the pool | forced p8: 3.925 µs sd 0.16% vs 4.33 serial in the same window |
| B=3 | — | 3.58 µs/call | |
| B=16 | 7.8 µs/call (r1 readings) | **4.10 µs/call min** | pool dispatch kills the GOMP fork |
| B=512 | 104.4 µs | **83.0 µs min** (0.162 µs/vol; medians ~103 under contention) | forced-disp A/B: 79.4 vs 106.9 OMP |
| B=8192 | 2948 µs | **2866 µs = 0.350 µs/vol** (sd 0.3%) | appender −2.5% vs the r1 whole-volume copy; L13_direct's wallaby number was 2788 |

rel_l2 = 4.0e-16 at every batch tried (1,3,16,33,512,2048,8192), tol 1e-12;
repeatable bit-identical across runs everywhere; builds clean AVX-512, AVX2,
and no-OpenMP.

Parallel efficiency (vs phase-1 serial on the same host): B=512
3.76/0.162 ≈ 23× (72% of 32); B=8192 ~5.26/0.350 ≈ 15× (47%, bandwidth
wall: ~200 GB/s aggregate with 70 KB/vol compulsory).

### What did not work / was declined, with numbers

- **The adopt-only-if-3%-faster bar for pool knobs** — see above; it is the
  thing that made mt_r1's numbers unimprovable on wallaby. Replaced by the
  veto form for POOL knobs only; pf/pw/nt/n24/a6 keep the old bar (their
  arena readings track the driver fine, e.g. nt0 +30% at B=8192, nt1 +180%
  at B=512 — both correct signs).
- **Pool teams at B=1 under login contention are a pick lottery**: two
  fresh creates minutes apart read p8:8964-vs-f0:6178 (veto) and p8-adopted
  → 3.236 µs scored. Node arena spread is ~0.3%, so the node's pick will be
  stable; do not tune this knob on wallaby numbers.
- **nts=2 (mt_r2's planned double-buffered volume drain)**: deleted unbuilt —
  the per-plane appender already overlaps the NT stream with compute at finer
  grain and with less scratch (2.7 KB vs 70 KB per thread).

### Borrowed, explicitly

- **Flat arrival-flag/release spin barrier + epoch dispatch + park/pinning:
  L17_winograd mt_r1** (their record documents the central-counter failure
  at 1.2 µs/barrier — not rediscovered).
- **Carried-partial-line NT appender: L13_direct mt_r2** (their §"the
  appender is the one part that is mine": partial line carried across plane
  and volume junctions; 2 WC flushes per thread range, not per volume).
- **Sequenced pool-vs-OMP dispatch race: L6_pfa mt_r1**; **per-thread sfence
  before the join: L8_radix8 mt_r1**; **"pw is pointless under NT":
  L23_matrixsimd mt_r1** (pw forced 0 in the NT path).

### Next round

1. Read the node's ab strings first: whether pt (and which team) and disp
   survived their vetoes, and what n24/a6 read on two real sockets. If the
   node vetoes the pool at B=1 while OMP t4 wins again, the two-phase body
   is fine and the pool wake path is the suspect (measure wake latency).
2. B=8192 remaining gap to fftw3_patient (node 0.603 with 110% spread —
   possibly not reproducible): the read side is now the only RFO-free lever
   left; try software-pipelining the next volume's x pass into the current
   volume's y/z (L13_direct's declined item 3a) or a 2×L3 NT gate on the
   node.
3. B=1: the pool two-phase floor on wallaby is ~3.2 µs ≈ compute 2.5 +
   barrier + wake; if the node adopts p4/p8 and lands ~4.5–5.0 µs the cell
   flips from L13_direct (5.67). If it wants more, fuse the mid-barrier away
   per-plane (per-plane U completion counters, L17_winograd mt_r2's s2mode —
   they measured counters LOSING at their size; only try if the barrier
   shows up in a node profile).

## Round mt_r4 — tuner repair: the two convicted pick lotteries, arena page-regime fidelity, n16 at the streaming cell

### Where mt_r3 left this entry (node, Gold 5218, VERDICT §3.2 — read it, both convictions are mine)

Scored: B=1 6.065 µs min but **8.5 µs representative** (2 of 3 processes
installed `p2` against my own arena pricing it +34%/−2% vs serial —
i:7761/p2:10417 in the very process that shipped p2); B=512 154.6 µs min but
**231 µs representative** (2 of 3 installed `n24`, driver-priced 452 vs the
pw! config's 302 ns/vol); B=8192 0.983 µs/vol, stable, pick=i in 3/3, cell
lost to fftw3_patient's 1-in-3 fast mode (0.603; on medians we win 1.19×).
The monitor's per-geometry order was explicit: remove the two bad picks,
then attack the bandwidth (both L=13 entries sit at 69–72 GB/s where
L36_mixedradix does 150.9 on the same node).

### What changed (all tuner/race mechanics; kernels, NT appender, pool, partitions untouched — arithmetic identical)

1. **B=1: strict incumbency restored.** The mt_r3 adopt-unless-vetoed rule
   for pool teams is deleted; a pool team is now adopted only if it beats
   the serial-group winner by >3% in BOTH trial blocks (L6_pfa's original
   rule). The veto form was built for a wallaby login-node pathology
   (napped workers losing cores under-prices the pool) that the idle node
   — arena spread ~0.3%, i:6880 vs p2:11744 — never shows. On the node
   this makes the pick `i` (serial fused) deterministically; wallaby dev
   runs may under-adopt the pool, which is the right trade since only the
   node scores. `-DL13R_FORCE_PT` stays as the monitor's A/B.
2. **Batched: dispatch race first, knob race second, under the settled
   dispatch.** mt_r3 raced knobs under the OMP fork and priced every
   variant 1.5–1.9× over the delivered pool config (node B=512: arena
   465–576 ns/vol, driver 302) — that inflation is what let n24's accurate
   475 read beat pw!'s noisy 549–576 in 2 of 3 processes. Wallaby
   confirmation of the fix, B=16: arena i:447 vs driver 437 ns/vol (2%);
   B=512: i:169 vs driver 159.
3. **Arena page-regime fidelity: the private race `out` is memset by the
   main thread right after allocation, exactly like the driver's** (the
   driver memsets `out` and freads `in` on the main thread → both
   socket-0-resident; L13_direct's on-buffer governor measured fr=0/0 at
   B=8192 — they never migrate). Until now my arena `ro` was first-touched
   by the NT appender's streaming stores in the first race lap, i.e.
   DISTRIBUTED across sockets per executing thread — every batched race ran
   in a NUMA regime the scored run never sees, which is why the arena read
   640 ns/vol at B=8192 where the driver delivered 983. Team-geometry
   racing was meaningless before this one-line fix.
4. **pw is deterministic-off when the batch fits both sockets' L3** (gate
   flipped from per-thread-slice-vs-L2 to aggregate 2×sysconf-L3,
   mirroring nts's 4× gate). Node B=512 (36 MB < 44 MB): incumbent becomes
   pw=0 — the config the driver ran at 302 ns/vol in all three processes.
   pw is still raced both ways at non-NT cells, so a machine where the
   gate is wrong can overturn it (wallaby did, honestly: pw=1 won its
   B=512 race −9% under the now-faithful arena and got adopted there).
5. **n16 replaces n24 in the streaming race (nts=1 cells, ntb=32 only).**
   n24 was +12% at B=8192 in 3/3 node processes and was the B=512 mispick
   — deleted both places; no team-geometry rows below the streaming gate
   at all. n16 is the single-socket team that won every fr=0 on-buffer
   width race in mt_r3 (L8×3: 19–35%; L36_pfa: s16 20.05 vs s32 24.44
   µs/vol; VERDICT §5). Mechanism I am betting on: with in/out
   socket-0-resident and immobile, the far socket's reads also pay Cascade
   Lake directory-update writes INTO socket 0's DRAM, so at the socket-0
   controller wall (my 71 GB/s) halving the team can raise USEFUL
   bandwidth. a6 (3:2 socket-weighted, keeps 32 threads) stays raced as
   the intermediate. If a narrower team is adopted, the pool is rebuilt at
   that width so the far socket's cores are EMPTY under the scored runs,
   not spinning (VERDICT §3.3/§4.4: three entries measured spinners
   dragging the all-core clock). `-DL13R_FORCE_NTB=n` added for the
   monitor.

### Operation count

Unchanged: 186 vector FP per 13-point transform, 3×169 transforms/volume,
zero extra FP in any mode; NT-tier compulsory DRAM stays 70 KB/volume.
Setup grows by one 549 MB memset at B=8192 (~0.15 s, excluded from score;
measured setup 0.46 s vs mt_r3's 0.66).

### Measured on wallaby (Gold 6448Y SPR, 32 threads close/cores, shared login node, driver-level min; loaded session, sd up to 39% on medians — treat as relative)

| cell | mt_r3 | mt_r4 | pick |
|---|---|---|---|
| B=1 | 3.236 µs | **3.704 µs min** (noisy window; arena p8:4205 vs i:5536) | p8 — clears the STRICT bar on wallaby, −24% |
| B=16 | 4.10 µs/call | **7.00 µs/call** (contended window; arena i:447 ≈ driver 437) | pool dispatch (dp:575 vs do:1187) |
| B=512 | 83.0 µs | **81.4 µs = 0.159 µs/vol** | pool (dp:168 vs do:254); pw! adopted honestly (154 vs 169) |
| B=8192 | 2866 µs | **2839 µs = 0.347 µs/vol** | i; n16:455 vs i:335 — single socket, n16 correctly loses 36% |
| B=512 FORCE_NTB=16 | — | 207 µs (T=16 on one socket = half the cores, as expected) | knob validated, pool shrinks to 15 workers |

rel_l2 = 4.0e-16 at B ∈ {1,16,512,8192}, tol 1e-12; bit-identical across
runs everywhere; AVX2-only and no-OpenMP builds pass (8.9 µs / 51.5 µs at
B=16). Wallaby cannot price any of this round's node questions (single
socket, no UPI, no socket-0 handicap) — the point of the round is that the
node's arena now runs in the node's own page regime.

### What did not work / was declined, with numbers

- **n16 on wallaby: 455 vs 335 ns/vol (+36%)** — as it must be on one
  socket (halved fill buffers, no UPI to avoid). Not evidence against the
  node bet; kept raced with strict adoption so a node where the directory
  story is wrong keeps ntb=32 at zero cost.
- **Improving B=1 serial itself: not attempted.** Node arena i readings
  (6880/7761/11108 ns across processes) say create-time clock state, not
  kernel, dominates that instrument; the scored 6.06 stands on eleven
  phase-1 rounds of tuning. The round's B=1 job was stability, which the
  strict rule buys.
- **The L2-tile port the VERDICT suggests for L=13** is a no-op here: the
  batched pipeline already streams one volume at a time through ~56 KB of
  per-thread L2-resident scratch (in read once, out written once, NT, 70
  KB/vol compulsory). Our 71 GB/s ceiling is the caller's socket-0 page
  residency (L13_direct's fr=0/0 + my arena's 640-vs-983 regime gap prove
  it), not the tile shape — which is why this round attacks team geometry
  instead.

### Borrowed, explicitly

- **fr=0/0 at L=13 B=8192 (caller buffers socket-0-resident, immobile):
  L13_direct mt_r3's governor measurement** — the fact my whole round is
  built on, not re-measured.
- **Single-socket team wins at fr=0 streaming cells: L8_fusedaxes /
  L8_batchsimd / L8_radix8 mt_r3 (19–35%) and L36_pfa mt_r3 (s16 20.05 vs
  s32 24.44)**, via VERDICT §5 — taken as the prior for n16 instead of
  re-deriving the width curve blind.
- **"Never leave a spin team idle under a scored run": VERDICT §3.3/§4.4**
  (L6_unrolled's 2.77× regression forensics, L17_matrixsimd's clk512 2.29
  vs 2.89 GHz) — implemented as the pool rebuild-at-adopted-width.
- **"Init race buffers to reproduce the driver's first touch": L6_pfa
  mt_r1 via L13_direct mt_r3's record** — the memset(ro) fix is exactly
  their lesson, applied to the out side where my NT appender had been
  silently changing the regime.

### Predictions for the node (pre-registered)

- **B=1: pick=i in 3/3 processes, 6.0–6.2 µs.** The two p2 processes are
  impossible under the strict rule (p2 read +34% and −2% — never >3%
  faster in both blocks). Loses to L13_direct's 5.87 honestly; beats
  mkl2026's 7.64.
- **B=512: pw=0 + pool in 3/3, 152–158 µs (0.30–0.31 µs/vol).** The knob
  race now runs at ~300 ns/vol scale where mt_r3's ran at ~470–580; n24
  no longer exists to sneak in. Should retake the honest cell win from
  L13_direct's 156.05.
- **B=8192: the informative cell.** If the directory/UPI mechanism is
  real, n16 prices well under i in the now-faithful arena and the cell
  lands **0.80–0.90 µs/vol** (socket-0 local ceiling ~85–90 GB/s); if the
  far socket's bandwidth contribution outweighs the directory cost, i
  stays at **0.96–1.00** and the negative datum (n16's honest two-socket
  price at fr=0) is published in the ab string either way. fftw's 0.603
  needs 116 GB/s through socket 0 — not reachable by any 32-thread
  schedule if fr=0 holds; on medians we already win.

### Next round

1. Read the node ab strings: (a) did B=1 ship i in 3/3; (b) B=512 scale of
   the knob readings (should be ~300 ns/vol — if still ~470, the pool
   dispatch was vetoed somewhere and that is its own datum); (c) n16 vs i
   vs a6 at B=8192 in the driver's page regime — this is the round's
   measurement.
2. If n16 wins B=8192: try n20/n24 asymmetric (16 socket-0 + 4–8 far
   threads doing READ-ONLY work, e.g. prefetch-into-L3 of socket-0 lines,
   never writing) — keep the directory quiet but borrow far-core LFBs.
3. If i holds and the cell stays ~0.98: the schedule is done; the residue
   is the harness's page placement, and the honest ask (already in two
   VERDICTs) is a parallel first-touch or interleave policy in the driver.
4. B=1 residue vs L13_direct (5.87 vs 6.06): their lead is the cheaper
   dense kernel at one thread, not threading; the phase-1 record says the
   next Rader FP reduction (CRT 6→3+3 on the SS side) was tried and lost
   to port pressure — do not reopen without a new idea.
