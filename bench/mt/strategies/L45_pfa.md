# L45_pfa — multicore strategy record

Phase-1 history (how the serial kernel reached its 344-op Good-Thomas 9x5
form, six rounds of it) lives in `../../geom/strategies/L45_pfa.md`.  This
file starts at the multicore phase.  Phase-1 node baselines (CLX,
panel_r11 scored, single thread): B=1 304.2 µs, B=2 308.9 µs/vol,
B=16 392.0 µs/vol.  Wallaby serial reference measured this round via the
in-plan serial control rows: ~250 µs/vol at B=1, ~190-260 µs/vol streaming.

## Round mt_r1

### What changed

The serial kernel is UNTOUCHED — same codelet, same two-sweep plane-fused
schedule, same 514,968 zmm FMA-port ops + 90 xmm tail lines per volume, and
every parallel path is bit-identical to the serial exec (parallelisation
only reassigns whole planes/tiles/volumes; the per-line DAG and operand
order never change).  What was added is a threading layer with four exec
classes, all raced at plan time on the real batch:

1. **mtv — volume-parallel** (`m_v0/v3/v0a/v3a`): thread t owns the
   contiguous volume block [nvol·t/T, nvol·(t+1)/T) and runs the serial
   per-volume pipeline with its own plane scratch.  No sync except the
   join.  The phase-1 pf ladder (pf0/pf3) and the r11 zal aligned-z-load
   twins are re-raced inside it — single-core pf verdicts do not transfer
   to the bandwidth-shared regime.
2. **mtn — mtv + NT-staged output** (`m_n0/ni/nia`, and `m_ni16`, below):
   both phases run in the thread's PRIVATE mid volume M (1.46 MB — inside
   one SPR L2), then one linear `vmovntpd` burst flushes M → out (head
   peel to 64 B: out volume bases rotate 16 B per volume since
   VDBL·8 mod 64 = 16; scalar tail; one sfence per thread per execute).
   Direct NT on the transform's own stores is impossible at L=45 (every
   64 B store is 16B-aligned, rows are 720 B), so staging is the only way
   to delete the out-RFO third of the streaming traffic.
3. **mtf — fused two-sweep** (`m_f0/f0a` round-robin, `m_fb/fba` blocked):
   omp-for over ALL nvol·45 x-plane pipelines (z+y of one plane into
   mid=out, per-thread plane scratch), ONE implicit barrier (every phase-2
   line reads all 45 mid planes of its volume), then omp-for `nowait` over
   nvol·507 flat phase-2 tiles (506 full + the masked tail as its own
   unit).  The plane-sweep schedule is the rr/blk knob: static,1 for B=1
   balance (45 = 32+13; blocked static idles 9 threads there), blocked
   static for B>1 so a volume's planes stay on ~2 adjacent threads.
4. **mtf-nt** (`m_fn`): fused three-sweep over per-VOLUME M slots
   (volume b stages in Mt[b mod 32]; batches beyond 32 run in 32-volume
   blocks with an inter-block barrier so a slot is flushed before reuse).
   All threads work at any batch AND no out-RFO.  Lost everywhere on
   wallaby (see below); kept for the node's different cache economics.
5. **Per-thread scratch**: Pt[t] (37 KB plane scratch) and Mt[t] (1.46 MB
   mid volume) are allocated and FIRST-TOUCHED by thread t inside
   fft3d_create()'s parallel region (which also spins up the OMP pool);
   page-aligned, NUMA-local under close/cores.  SCRATCHP (padded S) and
   the serial pf1/pf2 rungs are DELETED — sp took zero node picks in five
   phase-1 rounds (its pre-registered exit).
6. **Tuner**: same gate + interleaved-min + 3% simplest-first hysteresis,
   two changes.  (a) The arena cap rose 32 → 128 volumes (2×178 MB): at
   nv=32 the whole walk sat inside wallaby's 60 MB L3 and priced mtn at
   9.6 µs/vol where the B=256 driver said 25.7 — the streaming class was
   mis-ranked.  At nv=128 the tuner's pick and the driver agree.
   (b) Serial rows (ip-pf0/ip-pf3, both widths) stay as controls so the
   "does B=1 parallelise" question is answered by measurement per machine.
   Env forcing: FFT45_MT (v/f/s/n) replaces FFT45_MODE; FFT45_PW and
   FFT45_PF as before (pf codes: 0,3 ladder; 4,5 zal; 1 blk; 6 fused-nt;
   6..9 mtn family).
7. **`m_ni16`, the node NUMA probe** (rank last, wins only outright): mtn
   at a 16-thread half team.  The driver freads `in`/allocates `out` on
   its main thread, so on the two-socket node every caller page is
   socket-0 resident and the far 16 threads stream over UPI; whether 16
   all-local threads beat 32 half-remote ones is unanswerable on wallaby's
   single 32-core socket.  Every other mt_r1 record flags the same
   question; this entry ships it as a raced candidate.

### Operation count

Per volume unchanged from panel_r11 (INPLACE): 1497 zmm codelet calls ×
344 = 514,968 zmm FMA-port ops + 90 xmm lines.  mtn/mtf-nt add one volume
copy (182,250 doubles: ~22.8k NT stores of 64 B, zero FP) in exchange for
deleting the out-RFO; mtf adds 1 barrier per execute (2 per 32-volume
block in mtf-nt); mtv adds only the fork/join.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores on one socket
of 64; shared login node, driver-level min; serial reference from the same
build's in-plan control rows)

| case | mt_r1 | pick | serial (same host) | speedup | par. eff. |
|---|---|---|---|---|---|
| B=1   | **36.5-39.9 µs** | mtf-blk(a) | ~250 µs | 6.3-6.9× | 20-22% |
| B=3   | 22.7 µs/vol | (mtf) | ~250 | 11× | — |
| B=16  | **14.1 µs/vol** (225.0 µs/call) | mtf-blk(a) | ~236 | 16.7× | 52% |
| B=33  | 13.4 µs/vol | mtf-blk | — | — | — |
| B=256 | **16.9 µs/vol** (4330.7 µs/call) | mtn-pfi | ~200 | ~12× | ~37% |

Correctness: PASS rel_l2 = 4.0e-16 at B = 1, 3, 4, 16, 33, 256 (identical
to phase 1 — same arithmetic); repeatable (bit-identical across runs)
everywhere; AVX2-only host (wombat) PASS B=4 at 55.8 µs/vol; compiles
clean without OpenMP (shimmed).

Honest accounting of the missing cores:
* **B=256 is at the bandwidth wall.**  mtn moves the compulsory 2.92 MB
  per volume (in read + NT out write, M held in L2) at 16.9 µs/vol ≈
  170 GB/s aggregate — the NT flush bought −26% over plain mtv-pf3
  exactly as the L13R/L17R/L23M inversion predicted, and further threads
  cannot help, only less traffic can (there is none left to delete).
* **B=1 is L3-and-sync-bound, not work-bound**: the driver's timing loop
  keeps the whole 2.9 MB in+out set L3-resident, so the ~40 µs decomposes
  as ~10 µs sweep-1 critical path (13 threads own 2 planes: 45 = 32+13),
  ~2 µs tile sweep, ~6-10 µs GOMP fork + 1 barrier + join (rivals'
  measured region cost), rest L3 latency + imbalance.  Parallelising B=1
  was still worth 6.3-6.9× — L=45 sits on the opposite side of the
  "work ≫ region cost" line from L=13/17/23, as their records predicted.

### What did NOT work, with the numbers that killed it

1. **mtf-nt everywhere on wallaby**: 55.4 vs 43.0 µs at B=1 (NT bypasses
   an L3 that was doing useful work — the same +55% failure L17_rader
   measured at L3-resident B=256), 30.4 vs 11.8 µs/vol at B=16 (three
   sweeps' worth of cross-thread M handoff), 32.0 vs 16.6 at nv=128
   (private-M mtn strictly better once volumes ≥ threads).  Kept compiled
   and raced for the node; expect it to stay a loser there too.
2. **Round-robin plane sweep at B>1**: mtf-pf0 (static,1) 25.6 vs mtf-blk
   11.8 µs/vol at nv=16 — scattering a volume's 45 planes over 32 threads
   makes every phase-2 tile read 45 remote-dirty mid planes.  Blocked
   static keeps a volume on ~2 adjacent threads: −54%.  This is
   L23_rader's fused-at-mid-batch lesson, solved by schedule choice
   instead of abandoning fused.
3. **mtv at B<32**: 17.3-22.5 µs/vol at nv=16 vs mtf-blk 11.8 — half the
   team idle is worse than the fused handoff once the handoff is blocked.
4. **The nv=32 tuning arena at B=256** (inherited from phase 1): priced
   the whole streaming class wrong — mtn-pfia read 9.6 µs/vol in-plan
   while the driver's B=256 said 25.7 for the then-picked mtv-pf3a.  The
   arena must exceed the machine's L3 by a wide margin under 32 threads;
   nv=128 (356 MB) fixed the agreement.  Do not shrink it back.
5. **Serial at B=1** (the brief's question): 235-255 µs vs 36.5 parallel
   — at L=45 "B=1 does not parallelise" is decisively false, settled by
   the in-plan control row.

### Borrowed (attributions)

* **L13_direct mt_r1**: the whole batched decomposition — contiguous
  volume blocks from the delivered team, per-thread slots first-touched by
  their owner inside a create()-time region.
* **L23_rader mt_r1**: the fused/batch regime split and the "intra-volume
  splitting pays iff work ≫ region cost" rule (which said L=45 must split
  at B=1); the `nowait` on the last sweep (their −11%); the
  serially-filled tuner arena note.
* **L13_rader / L17_rader / L23_matrixsimd mt_r1**: the NT-at-threaded-
  streaming inversion (their −13/−14/−25%), the stage-then-NT-flush
  pattern, and L17_rader's head/tail-peel segfault lesson (vmovntpd
  demands 64 B alignment; volume bases rotate 16 B here).
* **L23_matrixsimd mt_r1** (transitively L36_pfa phase 1): prefetch on
  cache-resident lines is a pure uop tax — mtn keeps PF45/prefetchw off M
  and only paces PFIN/PFNX over the DRAM `in` stream.
* From my own phase-1 lineage: the exec-variant const-propagation
  structure, tuner gate on first AND last arena volume, hysteresis,
  slow-window-hardened round counts, env forcing.

### Next round

1. **Read the node leaderboard first**: which class won each cell, whether
   mtn-t16 beat mtn-pfi at B=256 (if yes, the fix is far-socket input
   staging or a weighted partition à la L17_rader, not fewer threads),
   and whether zal (blka/pfia) finally pays on the node's split-load
   physics as r11 pre-registered.
2. **B=1 sweep-1 granularity**: the critical path is 2 planes (45 over
   32).  Splitting phase 1 into z-subpass and y-subpass sweeps over
   half-plane units (90-135 units, per-plane pl slots, one extra barrier)
   cuts the floor to ~1.2 plane-equivalents; worth ~4-5 µs of the 36.5 if
   the node confirms sweep 1 dominates.
3. **Spin pool** only on node evidence: rivals price GOMP fork+barrier at
   3-8 µs; a flag-array pool (L23_matrixsimd's design, ~0.3-0.4 µs/barrier)
   is worth ~5-8 µs at B=1 and nothing elsewhere.  L17_rader and
   L23_matrixsimd both ship working pools to copy — do not rebuild their
   central-counter mistake (5 µs at T=32) or the team-only-ack race.
4. **B=16 traffic**: mtf-blk still pays the out-RFO.  A pair-grained mtn
   (2 threads share one M volume with a 2-thread spin barrier) would
   combine full team + NT; only worth it if the node shows B=16
   bandwidth-bound (wallaby says it is not yet: 14.1 µs/vol ≈ 5 MB in
   14 µs ≈ far from the wall per volume-pair).
