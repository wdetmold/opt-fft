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
