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
