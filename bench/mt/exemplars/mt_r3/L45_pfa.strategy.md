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

## Round mt_r2

### Where mt_r1 landed on the node (what this round attacks)

Node (Gold 5218, 2×16 cores): B=1 **60.645 µs, 2nd** (L45_mixedradix
58.317; my picks pw4-mtf-blk 2/3, pw4-mtf-pf0 1/3); B=16 **21.196 µs/vol,
2nd by 1.26×** (their grp16×2 + spin pool at 16.873); B=256 **26.897
µs/vol, WON** (mtn-pf0/pfi picks, 2.15× fftw3_patient, 2.94× over the
rival, run spread 1.04× — the round's cleanest streaming result).  Two
node facts killed r1 candidates: the zal twins took ZERO picks at any
cell (the split-load class loses on CLX under contention too), and the
VERDICT falsified the socket-0 page premise (T=32 won every batched cell
panel-wide; L=6 sustains 175 GB/s = both sockets' controllers; AutoNUMA
migration is the leading explanation), which kills mtn-t16.  The VERDICT
also directs L45_mixedradix to adopt my B=256 streaming construction, so
holding B=256 means improving it, not sitting on it.

### What changed (zero new FP; the serial kernel is untouched again)

1. **The OpenMP execute regions are GONE.**  fft3d_create() now builds a
   persistent pinned spin pool: 31 pthreads pinned to the CPUs one
   throwaway OMP region reports (the unbound-run guard included), each
   worker allocating and first-touching its own scratch block (M 1.46 MB
   + plane scratch 37 KB + tile buffer 45 KB, one page-aligned block)
   after pinning; execute() publishes a job with plain stores and
   release-stores one generation counter.  Barriers are flag-array:
   padded per-thread arrival flags + one release word per group, epochs
   derived from (generation, in-job volume index) so join, global barrier
   and pair barrier share the flag array monotonically.  BORROWED
   wholesale from L23_matrixsimd mt_r1 (their measured 6.2-8.2 µs GOMP
   fork+barrier+join at T=8..32, and their central-counter and
   flag-array numbers) via L45_mixedradix mt_r1's re-spelling; I copied
   the architecture rather than re-measuring a settled result.
2. **NEW grp2 class** (BORROWED from L45_mixedradix mt_r1, their B=16
   node winner): the adjacent-core pair (2g, 2g+1) owns a contiguous
   volume block; member m does x-planes [45m/2, 45(m+1)/2), ONE pair
   barrier (~0.1 µs), then phase-2 tiles [507m/2, 507(m+1)/2).  No global
   straggler coupling: a pair advances the moment its own volume's phase 1
   is done.  Raced plain (g2-pf0) and with pfin+prefetchw (g2-pfw = their
   winning v1 mechanism).
3. **NEW g2n = grp2 + NT staging** (my r1 "next round" item 4, affordable
   now that a pair barrier is ~0.1 µs): phase 1 into the pair's shared M,
   pair barrier, phase 2 via the staged-tile NT path below, each member
   NT-flushing exactly the tile range it computed.  Full team AND no
   out-RFO — the combination neither mtf nor mtn has, aimed at node B=16
   which is streaming there (89 MB in+out vs 76 MB aggregate cache).
   M is double-buffered on volume parity across the pair's two Mt slots
   (see the race under "did NOT work").
4. **NEW mts = staged-tile NT phase 2** (the B=256 refinement): phase 2
   reads M and writes each block of 16 flat tiles through a 45×128-double
   tile buffer (1 KiB rows, 64B-aligned), flushed as 45 short NT segments
   per block.  Against mtn (phase 2 in place in M + one linear ntcopy)
   this deletes one full write AND one full re-read of M per volume
   (~2.9 MB of L2/L3 traffic per volume).  M does not fit the node's
   1 MiB L2 (it does fit wallaby's 2 MiB — so this is a node-targeted
   candidate, raced not assumed).  Cost: segment head/tail peels are
   regular stores, ~6% of out pays RFO (the 16 B per-plane phase
   rotation, PLND·8 mod 64 = 16).
5. **mtf kept on the pool** (rr and blk plane schedules — the node's r1
   picks at B=1), plus mtf-bpf (blk + pfin/prefetchw/PF45 poke; my r1
   fused paths carried NO prefetch, while the rival's winning B=16 mech
   was exactly pfin+pfw) and mtf-t23 (T=23 team, raced only at B≤2;
   L45_mixedradix's wallaby B=1 pick was T=23 — 45 planes over 23 threads
   has the same 2-plane critical path as 32 but 9 fewer barrier
   participants and less licence-clock pressure).
6. **DELETED**: zal (pf0a/pf3a/blka/pfia — zero node picks, its
   pre-registered exit; the mechanism stays in the r11 exemplar), mtf-nt
   (lost every r1 cell everywhere), mtn-t16 (premise falsified).
   Candidate count 19, same tuner (nv = min(batch,128), serial fill, gate
   on first AND last arena volume, interleaved min, 3% simplest-first
   hysteresis).  Env: FFT45_MT is now v/n/g/f/s; FFT45_PF codes 0/3
   (plain/pf), 6/7 (mtn pf0/pfi), 8/9 (mts and g2n pf0/pfi), 1/2/5 (blk/
   bpf/t23).  The description string now publishes the whole in-arena
   table (the monitor used the rivals' tables in the r1 verdict; mine
   was opaque).

### Operation count

Per volume unchanged: 1497 zmm codelet calls × 344 = 514,968 zmm FMA-port
ops + 90 xmm tail lines.  mts/g2n write the same 182,250 doubles of out
non-temporally in 45×~32 segments per volume instead of one linear burst
(same NT store count, ~6% of out re-acquired as boundary-line RFOs) and
delete the 2.9 MB/vol M rewrite+reread that mtn pays.  Sync per execute:
one release-store dispatch + T arrival stores (join), plus one release
word per volume per pair (grp2) or one global flag barrier (mtf).  GOMP
fork/join (6-8 µs/execute in r1) is gone.

### Measured (wallaby, Gold 6448Y SPR, 32 threads close/cores; quiet
windows, driver-level min; in-pool serial reference ~274 µs)

| case | mt_r1 | mt_r2 | pick | par. eff. (vs 274 µs in-pool serial) |
|---|---|---|---|---|
| B=1   | 36.5-39.9 µs | **28.7-35.8 µs** | mtf-t23 | 24-30% at T=23 |
| B=3   | 22.7 µs/vol | 19.9 µs/vol | (mtf/g2) | — |
| B=16  | 14.1 µs/vol | **11.7 µs/vol** (187.8 µs/call) | g2-pfw | ~63% vs streaming serial ~236 |
| B=33  | 13.4 µs/vol | 13.3 µs/vol | g2/mtv | — |
| B=256 | 16.9 µs/vol | **16.8-16.9 µs/vol** (4302-4328 µs/call) | mtn-pfi | ~37% |

In-arena tables (quiet windows, µs/vol): B=1 — t23 **35.1**, blk 40.4,
rr 41.0, bpf 41.2, serial 274.8.  B=16 — g2-pfw **11.5**, bpf 11.9, g2-pf0
14.2, blk 15.3, g2n 19.1-20.1, mtv 23.0-23.9, mtn 27.1.  B=256 (nv=128) —
mtn-pfi **16.8**, mtn-pf0 18.1, mts-pfi 18.3, mts-pf0 19.8, g2n-pfi 20.3,
mtv-pf3 22.2, g2-pfw 23.5.  Correctness: PASS rel_l2 = 4.00e-16 at
B = 1, 3, 4, 16, 33, 256 (identical to phase 1 — same arithmetic),
repeatable (bit-identical across runs) everywhere; wombat (AVX2, pw2 path)
PASS B=4 repeatable; compiles clean without OpenMP; forced modes
(FFT45_MT=g/s etc.) verified working.

B=1 accounting: the pool + T=23 moves wallaby B=1 from 36.5-39.9 to
28.7-35.8 — roughly the 5-8 µs the GOMP fork/join cost plus the smaller
T=23 barrier.  The residue is still the 2-plane critical path (~11 µs)
plus L3 latency; the half-plane sweep idea (r1 next-round item 2) remains
the next lever there and was NOT built this round (pool + grp2 + mts was
the round's budget).

### What did NOT work, with the numbers that killed it

1. **g2n with a single shared M slot: a real data race.**  A member's
   phase 1 of volume b+1 overwrites the pair's M while its sibling is
   still READING M in volume b's phase 2.  Invisible at B=16 (every pair
   owns ONE volume — no reuse), it gated OUT nondeterministically at
   nv=128 (g2n-pf0 failed the last-arena-volume check in one window,
   passed in another).  Fix: double-buffer M on volume parity across the
   pair's two Mt slots — the single pair barrier then orders slot reuse
   (a member reaches phase 1 of b+2 only through the barrier of b+1,
   which its partner passes only after finishing phase 2 of b).  Zero
   extra sync.  The create()-time gate on the LAST arena volume is what
   caught this; a first-volume-only gate would have shipped it.
2. **mts on wallaby streaming: 18.3 vs mtn 16.8 µs/vol** — M fits
   wallaby's 2 MiB L2, so the M rewrite+reread mts deletes was nearly
   free there, and the 45-segment flush is costlier than one linear NT
   burst.  Raced for the node's 1 MiB L2, where the deleted traffic is
   real; if the node also keeps mtn, mts is dead and says so.
3. **g2n at wallaby B=16: 19.1-20.1 vs g2-pfw 11.5 µs/vol** — B=16 is
   L3-resident on wallaby (44.5 MiB of 60) and NT bypasses an L3 doing
   useful work, the same +55%-class failure as r1's mtf-nt.  Node B=16
   is past its aggregate cache; the race there is the point.
4. **Wallaby slow windows, three hits this round** (28 ms B=16 call with
   a garbage tuner table that picked pw2-mtf-blk; 28 ms B=3; sd 12-47%).
   Never read one window — every number above is from a quiet-window
   rerun.  The min-over-interleaved-rounds tuner survives mid-window
   dispatch, but a whole-window tuner run can still mis-pick; on the
   quiet node this has never been observed.
5. **mtf-rr at B>1 still loses** (25.8 vs blk 15.3 at B=16), as in r1 —
   kept only as the B=1 twin.

### Borrowed this round (attributions)

* **L23_matrixsimd mt_r1** (via **L45_mixedradix mt_r1**'s re-spelling):
  the persistent pinned spin pool, flag-array arrival/release barriers,
  the OMP thread→CPU read-back with unbound-run guard, worker-side
  first-touch after pinning, and the 6-8 µs GOMP measurement that
  justifies all of it.  Their records say "do not rebuild the
  central-counter mistake (5.0 µs at T=32)" — I did not.
* **L45_mixedradix mt_r1**: the grp2 pair decomposition (their node B=16
  winner) including the one-barrier-per-volume argument, the pfin+pfw
  (v1) mechanism as the in-place pair candidate, the T=23 B=1 team size,
  and the publish-the-tuner-table-in-the-description practice.
* **L17_matrixsimd mt_r1**: the cross-L2 read-poke rationale behind
  mtf-bpf's PF45 use at a fused sweep (their xpf won 0.4-0.7 µs on
  small teams).
* **mt_r1 VERDICT**: T=32 everywhere batched (kills mtn-t16), zal's exit,
  and the arena-fidelity lessons already baked into my nv=128 arena.

### Node predictions (pre-registered)

* **B=1**: the pool deletes the 6-8 µs GOMP cost from 60.6 → **52-57 µs**
  expected; t23-vs-blk is the open question the race answers (the node
  picked blk/T=32 in r1, but that was against GOMP pricing; rival's 58.3
  is the target).
* **B=16**: g2-pfw ≈ the rival's winning shape plus my kernel, g2n adds
  RFO deletion in a cell that streams there (compulsory 2.92 vs 4.4
  MB/vol) → **14-18 µs/vol**, beating 16.873 requires either to hold.
* **B=256**: mtn-pfi is the incumbent at 26.9; mts wins only if the
  1 MiB-L2 M-traffic argument is real on CLX → **25-27 µs/vol**, any
  mts pick is upside.  L45_mixedradix will arrive with my construction
  per the verdict's directive; the mts margin is my hedge.

### Next round

1. Read the node picks per cell (the description now carries the full
   in-arena table).  If mts/g2n took no picks, delete them per their
   pre-registered exits above.
2. B=1 half-plane phase-1 units (z and y subpasses as separate sweeps
   over per-plane scratch slots, one extra flag barrier ~0.3 µs) — cuts
   the 2-plane critical path to ~1.2 plane-equivalents, now cheap enough
   to pay at the pool's barrier price.
3. If node B=256 is still ~108 GB/s aggregate (well under the 175 the
   L=6 entry sustains), the remaining lever is read-side: far-socket
   input staging (L23_matrixsimd's next-round item 1) or a weighted
   volume partition by socket.

## Round mt_r3

### Where mt_r2 landed on the node (what this round attacks)

Node (p55n3, Gold 5218): B=1 **57.197 µs, WON** (mtf-rr picks 3/3; rival
57.549); B=16 **15.751 µs/vol, WON** (g2-pfw 3/3; rival 16.822); B=256
**45.446 µs/vol, WON at 1.25× fftw3_patient but REGRESSED 1.69×** from
r1's 26.897 — the round's largest stable regression, and the VERDICT
names impl_1-vs-impl_2 (OMP regions vs spin pool) as the A/B to run.
The raw records sharpen it: r1's three processes settled MID-RUN
(min 6.9-7.2 ms/call ≈ 105-108 GB/s, with max-samples of 9.7-10.8 ms —
the early samples were slow, then the run reached a faster regime);
r2's three processes NEVER settled (min≈median≈max ≈ 11.6-12.5 ms,
~64 GB/s, in every process).  And the create-time arena cannot separate
the two forms: r2's node arena read mtn-pfi at 33.6/45.8/42.4 against
drivers of 48.1/45.4/48.3 — the arena races in the pre-settling regime
(the VERDICT's §5 mechanism, measured by L8_fusedaxes's fr=0 governor).

### What changed (serial kernel untouched, fourth round running)

1. **A real r2 bug found by inspection: Pt was 16-mod-64 aligned.**  r2
   packed each worker's scratch as one page-aligned block [M | Pt | Tb];
   VDBL·8 = 1,458,000 ≡ 16 (mod 64), so EVERY 64 B plane-scratch access
   — z-pass stores, y-pass loads, ~26k per volume, all cache-resident —
   split a cache line.  SPR hides splits (wallaby read 16.8 vs 16.9
   across the r1→r2 change, which is why I never saw it); CLX does not
   (the whole r11 zal exercise existed because CLX splits cost), and the
   tax lands at every node cell in proportion to per-volume work.  Fix:
   M is padded to a page boundary, Pt is page-aligned again.  Footnote:
   r2's "64B-aligned tile buffer" claim was false too (same offset bug),
   so mts raced the node handicapped; its exit stands regardless since
   the winner it lost to carried the same handicapped Pt.
2. **The dispatch A/B (the verdict's named experiment): o-mtn.**
   omtn-pf0/omtn-pfi reinstate the r1 construction verbatim — vols_nt
   under one `#pragma omp parallel` per execute, same contiguous volume
   blocks, same (now aligned) per-thread scratch — next to the pool's
   mtn-pf0/pfi.  The pool gets a NAP mode (mutex+condvar; nap_on waits
   until every worker is actually blocked) entered whenever an sfn
   candidate runs, so a GOMP team never timeshares a core with a
   spinning worker; with an OMP pick installed the pool sleeps for the
   plan's whole life and the process presents the r1 thread picture to
   the kernel (whatever it was about that picture that let r1's runs
   settle into 108 GB/s).  Because the arena cannot separate the twins,
   o-mtn is RANKED AHEAD of pool mtn and the 3% hysteresis routes the
   tie to the r1 form; the driver's long run is the instrument.  Nap
   design after L13_rader's unbuilt r2 intent (their l13r_pool_nap /
   l13r_dwell symbols); pool architecture otherwise unchanged.
3. **Placement governor (BORROWED from L8_fusedaxes mt_r2)**: every 48th
   execute, a pure READ of the page homes of the caller's in/out
   (get_mempolicy MPOL_F_NODE|MPOL_F_ADDR, 32 sampled pages each),
   published in the description as gov{nb,c,i0/i1,o0/o1} with
   /proc/sys/kernel/numa_balancing.  The verdict names "fr under a wide
   team at a streaming cell" as the ONE experiment that settles §5;
   L=45 B=256 under T=32 is that cell, and my B=256 pick runs T=32.
   Nothing moves a page (the mt_r1 ruling).
4. **NEW mtf-hp (B≤2)** — my r1/r2 records' pre-registered B=1 lever:
   phase 1's z and y subpasses become separate sweeps over
   (plane, lane-group) units (12 per subpass per plane: 11 full PW=4
   groups + the PW=1 tail line) through 45 SHARED page-aligned per-plane
   scratch slots, one extra flag barrier between the subpasses.  45
   planes over 32 threads quantizes to a 2-plane critical path; 540
   units over 32 quantizes to 17/12 ≈ 1.42 plane-equivalents, and both
   sweeps partition the SAME contiguous unit index space, so a thread
   y-transforms the planes it z-transformed and slots stay in its own
   L2 except at partition boundaries.  Unit bodies are verbatim
   extractions of phase1_plane's two subloops — bit-identical output.
   Ranked LAST among the fused rows: the new mechanism must beat the
   incumbents by >3% to install.
5. **DELETED per pre-registered exits**: mts (node kept mtn 3/3:
   mts-pfi 44.1-46.3 vs mtn-pfi 42.4-45.8) and g2n (zero node picks;
   43 vs g2-pfw's 15.6 at node B=16), with phase2_nts, the tile buffer,
   and grp2's NT arm.  mtf-t23 kept one more round (lost node B=1 by
   only 4% and won wallaby B=1 windows again this round).
   Candidate count 18.  Env: FFT45_PF adds 10/11 (omtn pf0/pfi) and
   4 (mtf-hp); 8/9 retired.

### Operation count

Per volume unchanged: 1497 zmm codelet calls × 344 = 514,968 zmm
FMA-port ops + 90 xmm tail lines.  hp does the same phase-1 work in the
same order through shared slots (+1 global barrier per execute, and the
slot arena is 45·40960 B/volume vs 37 KB/thread — still L2/L3-resident).
o-mtn is arithmetically identical to mtn; it trades the pool's ~0.3 µs
dispatch for a GOMP region (6-8 µs, <0.1% of an 11.6 ms call).  The
governor adds ~64 syscalls every 48th call (~30-40 µs once per 48,
invisible to min/median).

### Measured (wallaby, Gold 6448Y SPR, 32 threads; windows this session
were mid-grade and toggling — every number is the best of 2-4 runs)

| case | mt_r2 | mt_r3 | pick |
|---|---|---|---|
| B=1   | 28.7-35.8 µs | **29.5-38.1 µs** (window-bound) | mtf-t23 mostly; mtf-hp in one window |
| B=2   | — | 39.3-47.8 µs/call | (forced hp 19.6 µs/vol) |
| B=16  | 11.7 µs/vol | **12.5-14.4 µs/vol** (200.7-230.1 µs/call) | g2-pfw (unchanged mechanism) |
| B=256 | 16.8-16.9 µs/vol | **16.65-16.8 µs/vol** (4263-4312 µs/call) | **omtn-pfi** |

In-arena (B=256, nv=128): omtn-pfi 16.9 vs mtn-pfi 16.7 — within 3%,
rank routes the pick to the r1 OMP form, and the driver confirms zero
wallaby cost for it.  In-arena (B=1, three windows): hp 35.4/31.8/35.7
vs t23 37.4/26.2/34.2 vs rr 43.3/31.8/37.5 — hp ties or beats rr (the
node's incumbent) in every window; t23 and hp trade wins on wallaby.
Governor on wallaby reads gov{nb=1,c=48,i=32/0,o=32/0} (single socket —
trivially all node 0, numa_balancing ON).  Correctness: PASS rel_l2 =
4.00e-16 at B = 1, 2, 3, 4, 16, 256, bit-identical repeatability
everywhere, all forced modes (omtn-pf0/pfi, mtn, g2, hp) PASS with
identical rel_l2; wombat (AVX2, pw2) PASS B=4 at 37.5 µs/vol;
compiles clean without OpenMP (o-mtn rows compile out).

### What did NOT work / what was not done, with numbers

1. **Nothing separated the dispatch twins on wallaby** — omtn-pfi 16.9
   vs mtn-pfi 16.7 in-arena, driver identical.  Expected: wallaby is
   one socket, the mechanism under test is a two-socket placement
   regime.  The A/B is decided on the node or not at all; that is the
   point of shipping both.
2. **hp did not dominate t23 on wallaby B=1** (26.2 vs 31.8 in the
   quietest window; hp won a slower window 35.4 vs 37.4).  Wallaby's
   t23 advantage (9 fewer barrier participants on fast SPR cores) is
   exactly what the node r2 numbers said does NOT transfer (t23 lost
   all 3 node processes to rr by ~4%); the node contest is hp vs rr,
   which hp never lost here.  If the node still picks rr and hp takes
   no cell, hp exits next round.
3. **Wallaby slow windows again** (B=1 driver spanned 29.5-48.8 µs
   across identical binaries; the r2 record's warning stands).  Nothing
   this round can be concluded from wallaby magnitudes, only from
   within-window candidate ordering.

### Borrowed this round (attributions)

* **L8_fusedaxes mt_r2**: the read-only get_mempolicy page-home governor
  including the 48-call re-scan cadence and the gov{} description
  publishing — the instrument the verdict called the round's most
  valuable diagnostic, ported nearly verbatim.
* **L13_rader mt_r2 (intent)**: the pool nap/dwell design its unbuilt r2
  symbols named (l13r_pool_nap, l13r_dwell); implemented fresh here as
  mutex+condvar with a counted quiesce.
* **mt_r2 VERDICT §5**: the pre-settling-arena mechanism, the "rank the
  known-good wide form as incumbent at streaming cells" remedy (my rank
  ordering of omtn ahead of mtn is exactly that), and the directive to
  run the impl_1/impl_2 A/B.

### Node predictions (pre-registered)

* **B=256**: the cell decomposes three ways and the gov line arbitrates.
  If the r2 regression was the spinning pool suppressing the settling,
  omtn-pfi returns to **27-30 µs/vol** and gov shows i/o migrating off
  node 0 mid-run.  If it was the Pt split tax, the fix alone lands
  **32-40** and gov stays i=32/0.  If both partial, between.  Any pick
  of omtn at ≤34 beats fftw3_patient by ≥1.7× and holds the cell; the
  in-arena table publishes both twins either way.
* **B=1**: Pt fix (~0.5-1 µs of split tax spread over 32 threads) plus
  hp-if-real (~3-4 µs off the 2-plane critical path) → **52-56 µs**;
  the win condition vs the rival's 57.5 is either lever alone.
* **B=16**: mechanism unchanged + Pt fix (each g2 member runs 23 planes'
  worth of now-split-free scratch) → **14.0-15.7 µs/vol**.

### Next round

1. Read the B=256 pick, its driver number, AND the gov{} line per
   process — this round was built to make that triple decisive.  If
   omtn ~27 with migration visible: the spin pool is convicted at
   streaming cells; consider retiring pool dispatch above the aggregate-
   cache threshold panel-wide.  If ~32-40 with i=32/0: placement never
   settles for anyone anymore, r1's 26.9 was a regime the kernel no
   longer grants, and the remaining lever is the weighted socket-0-heavy
   volume partition (raceable honestly, since the arena and driver would
   finally share a regime) — build it.
2. If hp took node B=1, extend it to B≤4 and retire t23; if rr held,
   delete hp per its exit above.
3. If g2-pfw holds B=16, leave it alone permanently.
