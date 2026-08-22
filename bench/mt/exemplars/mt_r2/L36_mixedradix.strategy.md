# L36_mixedradix — multicore phase strategy record

Phase-1 (single-core) history lives in `../../geom/strategies/L36_mixedradix.md`
(11 rounds; final form: row-column PFA 4x9, n1_9 DFT9, two sweeps, V0/V1
kernels, pf/pfin/pfw mechanisms, plan-time tournament). That kernel is kept
verbatim as the per-thread body; this file records the 32-core layer on top.

## Round mt_r1

### What changed

First multicore round. The serial kernel was untouched (same 972 codelet
calls, 232 FMA-port ops + 57 shuffles per 36-point line over PW lanes,
2,814,912 flops/volume); everything new is scheduling around it:

1. **Volume-parallel for B >= 2** (`mt_vol_run`): a contiguous block of
   volumes per thread, each thread running the tuned serial body (with its
   phase-1 pf/pfin/pfw mechanisms intact) on its own slab. Zero
   synchronization inside a call; the serial body's per-volume L2 blocking is
   exactly the per-thread working-set story the mt brief asks for (in+out =
   1.5 MB/thread live at a time). Contiguous blocks match
   `OMP_PROC_BIND=close`.
2. **Work-stealing twin** (`mt_vol_dyn_run`, streaming batch only,
   batch > T): `schedule(dynamic,1)` over volumes. Rationale: the driver
   first-touches both caller buffers on its main thread (fread/memset — I
   read driver.c), so on the two-socket node **all caller pages live on
   socket 0** and socket-1 threads run each volume slower through UPI; a
   static split then parks socket-0 threads at the join. Dynamic rebalances
   the speed asymmetry. Offered only where there is no cross-call cache reuse
   for static ownership to protect. Wallaby cannot price this (see below);
   the node tuner decides.
3. **Within-volume split for B=1 / small B** (`split_body`): phase 1
   parallel over the 36 independent x-planes (`schedule(static,1)`, each
   thread does z+y subloops of its planes on its own scratch; the loop's
   implicit barrier is the one phase1→phase2 sync the dataflow needs), phase
   2 parallel over (y,zb) tiles (V1, 324 units of 64 B stores) or whole
   y-rows (V0 — 32 B tile stores would false-share) with `nowait`, so at
   B>1 a finished thread runs ahead into volume b+1's phase 1. All unit
   boundaries are 64-byte aligned: no false sharing anywhere.
4. **Scratch**: NT_MAX=32 per-thread chunks, 48 KiB (page-multiple) apart,
   each **first-touched by its own pinned thread in `fft3d_create()`**
   (NUMA-local), where the OpenMP pool is also spun up once — execute never
   creates a thread beyond the (warm) team fork.
5. **Tournament extended** to {serial, split(T ∈ {32,18,16,8}),
   vol(T ∈ {32,16,batch}), dyn(T)} × {V0,V1} × mechanisms, serial first in
   hysteresis order (the phase-1-certified incumbent), V1 before V0, 3% bar
   kept. The streaming arena is now sized against the **combined** cache
   (2×L3 + 32×L2, ≈190 MB target) instead of one socket's L3 — the phase-1
   arena (2.5×22 MB) would sit comfortably inside the 32-thread combined
   cache and rank streaming candidates on a cached arena (L36_pfa's round-2
   lesson, upgraded for 32 threads). Arena fill stays serial on purpose: that
   reproduces the driver's single-socket page placement.
   T=18 is in the split pool because it divides both 36-unit phases exactly
   (2 planes + 2 rows per thread); T=16 is one full socket on the node.
   Env for monitor A/Bs: `FFT36_MT_T=<n>` (restrict team size),
   `FFT36_MT_MODE=ser|split|vol`; `FFT36_PFIN/PFW/PIND/ZY` keep their
   phase-1 meanings (zy default-out this round — it was a single-core port-5
   bet the phase boundary left unpriced; `FFT36_ZY=1` re-admits it).

### What was measured (wallaby, 2×32-core SPR — NOTE: with 32 threads and
### PLACES=cores the whole team lands on socket 0 there, so wallaby prices
### neither UPI nor the two-socket barrier; login node was busy, sd 5–16%)

All PASS rel_l2 = 3.586e-16…3.591e-16 vs numpy at B=1/8/32/512, bit-identical
re-runs everywhere (the parallel bodies run the same codelets on the same
data in a different order, so output is bit-identical to serial by
construction; the 1e-13 admission gate stayed as a safety net and passed for
all 18–24 candidates per cell).

| cell | pick | best us/transform | vs. in-plan serial twin |
|---|---|---|---|
| B=1   | v1-split32-pf0 | **17.76** | ser 74.8 (busy), 50.0 quiet r11 ⇒ ~2.8× |
| B=32  | v1-vol32-pfw   | **3.69**  | ser 94.7 ⇒ 26× , eff 0.80 |
| B=512 | v1-vol32-pfw   | **9.97**  | ser 166.0 ⇒ 25× in-arena, eff 0.80 |

* Team-size A/B at B=1 (FFT36_MT_T forced, same process style): T=32 17.6,
  T=18 19.8, T=16 21.2 µs — on wallaby's single socket wider is simply
  better. **Do not read this as the node's answer**: on 2×16 CLX the
  cross-socket barrier and UPI transfer enter at T=32, which is why T=16/18
  stay in the pool for the node tuner.
* B=1 parallel efficiency is honestly poor (~0.09–0.17 by wallaby windows)
  and the reason is structural, not barrier count: phase 2's transpose reads
  ~746 KB/call out of other cores' L2s, and next call's phase 1 pulls those
  lines back — ~1.5 MB of cross-core movement per 746 KB volume. Critical
  path arithmetic (2 planes ≈ 1.9 µs + tiles ≈ 0.6 µs) accounts for <3 µs
  of the 17.8; the rest is transfer + fork/join. That movement is the 3D
  transpose itself; fewer threads shrink it only by idling cores.
* B=32 at 3.69 µs/vol ≈ 982 GF/s nominal is cached-regime on wallaby
  (47.8 MB < its 60 MB L3 × 1.25). The node will stream this cell.
* setup 0.16 s (B=1) … 1.8 s (B=512, 128-volume arena ×3 buffers).

### What did NOT work / negatives worth keeping

1. **T=16 and T=18 at B=1 on wallaby**: +19% and +12% over T=32 (numbers
   above) — expected on one socket; kept in the pool strictly for the node.
2. No other mechanism was tried and killed this round; the phase-1 negatives
   (NT stores, sp2, nta, roll, ct9 — all node-closed) were not re-opened, per
   the brief's instruction not to rediscover closed results.

### Borrowed

* The tournament/hysteresis/arena machinery is my own phase-1 file; the
  "arena must actually stream on the tuning machine" rule is L36_pfa r2/r3,
  here extended to the combined 32-thread cache.
* pfw/pfin mechanisms ride along unchanged inside the per-thread body
  (L36_pfa r5 / L6_unrolled r3, attributed in the header since phase 1).
* No mt-phase rival records existed yet to borrow from (round 1).

### Predictions for the node (Gold 5218, 2×16 cores, all caller pages on socket 0)

* B=1: pick v1-split{16 or 32}-pf0; if the two-socket barrier + UPI transfer
  bite, T=16 wins and lands ~25–40 µs; if not, T=32 lands ~20–30 µs. Either
  way well under threaded MKL's small-volume threading.
* B=32: streaming there (95 MB). Static vol32 has a socket-asymmetry stall
  (16 remote threads each own 1 volume, all pages remote); dyn cannot help at
  batch == T (nothing to steal, hence not offered); **T=16 might genuinely
  win this cell** — one socket, all-local, 2 volumes/thread. Watch the pick.
* B=512: v1-vol32-pfw or v1-dyn32-pfw at roughly 15–25 µs/vol — the cell is
  capped by ONE socket's DRAM bandwidth (~3 crossings × 746 KB ≈ 2.2 MB/vol
  over ~100 GB/s ⇒ ~11 µs/vol floor if remote cores add UPI throughput on
  top, worse if they only contend).

### Next

1. **Read the node picks first** (they ride the description string:
   `pick=… ser=… pick=… eff=…`), especially T=16-vs-32 at B=32/512 and
   split-T at B=1 — they answer the NUMA questions wallaby cannot.
2. **Page migration question for the monitor**: since the driver first-touches
   both buffers on socket 0, a `move_pages(2)` pass over the caller's buffers
   during the first (warmup, discarded) execute would place each thread's
   slab on its own socket and roughly double streaming bandwidth. It is
   within the letter of the contract (contents untouched, repeatable,
   warmup excluded) but subverts the "same buffers for everyone" spirit, so I
   did NOT ship it; if the panel rules it legal, it is the single biggest
   lever on B=512 (~2×) and every entry should get it.
3. If the node's B=1 shows T=32 sync-bound: a persistent pthread gang with
   sense-reversing spin barriers (fork/join ≈ 0.3 µs vs libgomp's few µs)
   is the next structural step; also consider a 2-gang hybrid at B in
   [2,32): split the team into per-socket gangs, volumes round-robin between
   gangs, within-volume split inside a gang (kills both the imbalance at
   batch == T and the cross-socket transpose).
4. If B=32's static vol has the predicted remote-half stall and T=16 wins:
   try weighted static splits (socket-0 threads take ~60% of the volumes) as
   a cheap middle ground before the gang machinery.

## Round mt_r2

### What changed

The serial kernel is again untouched (232 FMA-port ops + 57 shuffles per
36-point line over PW lanes, 2,814,912 flops/volume; the snt body below adds
zero arithmetic — it only redirects stores).  Three changes, all in the
32-core layer, all taken from other entries' mt_r1 records as this round
invites:

1. **Persistent pinned pthread spin pool** replaces every OpenMP parallel
   region in execute().  BORROWED WHOLE from L36_pfa mt_r1 (who credit
   L23_matrixsimd's 6.2–8.2 µs GOMP-region measurement and L17_winograd's
   flag-array barrier).  Same protocol as theirs: workers spawned in
   create(), pinned to the close/cores CPU map read back via sched_getcpu(),
   parked on an epoch spin over one release word; flag-array barriers (each
   arriver release-stores its own padded line, participant 0 scans and
   publishes); every worker — idle ones included — posts a done flag per
   dispatch so the job descriptor is never rewritten while readable (torn
   reads closed by protocol); pool SHRUNK to the picked team after the
   tournament (their spinner-drag measurement).  One refinement of mine: a
   worker's idle spin goes to 200 µs naps after ~10 ms hot, so the long
   single-threaded stretches of create() (arena fill, L2 diffs) do not run
   against 31 hot spinners; back-to-back dispatch (the only case that is
   timed) stays on the hot path.  Measured dispatch+join at T=32 on wallaby:
   **dsp = 0.70 µs** (rides the description string now), vs the several-µs
   OMP fork the r1 shape paid per call.
2. **Pair-split of the leftover planes** in the B=1 within-volume split.
   36 planes over T=32 used to leave 4 threads with 2 planes (phase-1 span
   2.0 waves); now threads 2i, 2i+1 share leftover plane i: z-halves into
   the even partner's (otherwise idle) volume-scratch area, a 2-thread flag
   sync, disjoint y-halves.  Span 2.0 → 1.5.  This is L36_pfa mt_r1 "next
   round" idea (b), implemented; all half boundaries are 64-byte aligned in
   both scratch and out (V1 cuts: yb 5/4, zb 5/4; V0: yb 9/9, zb 10/8 so the
   cut lands at 320 B).  T=12 added to the split pool (3 planes/thread,
   exact balance, sub-socket on the node); pool is now T ∈
   {32,24,18,16,12,8}, pair-split active where 36 mod T ≠ 0 and 2R ≤ T.
3. **Scratch-volume + NT phase 2** (exec codes 7 "snt" / 8 "sntp" = +pfin,
   streaming candidates only): phase 1 writes a per-thread 746 KiB volume
   scratch (L2-warm after the first volume), phase 2 reads it and STREAMS to
   out (dft36_xnt: same codelet, NT stores).  DRAM traffic per volume drops
   read(in)+RFO(out)+wb(out) ≈ 2.2 MB → read(in)+NTwrite(out) ≈ 1.5 MB.
   Phase 1 rejected NT four rounds SINGLE-core; L23_matrixsimd mt_r1
   measured the verdict inverting at 32 threads and L36_pfa mt_r1 confirmed
   at B=512 (their nt 5.83 vs inplace 7.02 in-arena, wallaby) — both
   attributed.  Offered at vol T ∈ {32,16} and dyn; the tournament (which
   runs on whatever machine create() runs on, i.e. the node prices it
   itself) decides against pfw-in-place per cell.
   Scratch chunks grew to PT_STRIDE = 99840 doubles/thread (plane scratch +
   volume scratch, page-multiple, still first-touched by the owning pinned
   thread), 25.5 MiB total.

Env knobs added/kept: FFT36_SNT=0|1 (exclude / force-only snt),
FFT36_MT_T now accepts the wider split pool, everything else as r1.

### What was measured (wallaby, Gold 6448Y SPR, 32 threads close/cores,
### shared login node — quiet-window minima of 3 runs × 8 samples)

All PASS rel_l2 = 3.575e-16…3.591e-16 vs numpy at B = 1, 2, 8, 31, 32, 33,
512; repeatable (bit-identical across runs); AVX2-only path verified on
wombat (28.9 µs at B=1, PASS, repeatable).

| cell | mt_r1 (wallaby) | mt_r2 (wallaby) | pick | note |
|---|---|---|---|---|
| B=1   | 17.76 | **12.04** (−32%) | v1-split32-pf0 | dsp=0.71 µs; in-tuner 12.7–14.0 |
| B=32  | 3.69  | **2.66** (−28%)  | v1-vol32-pf1   | cached regime on wallaby |
| B=512 | 9.97  | **7.19** (−28%)  | v1-vol32-sntp  | in-arena 5.7–5.9 µs/vol |

* B=512's pick is the new snt body: in-arena 5.8 vs 7.0-ish for pfw shapes —
  the same −17..−25% L36_pfa saw, reproduced in my kernel.  Full-buffer 7.19
  µs/vol ≈ 1.5 MB/vol over ~208 GB/s aggregate.
* B=1 split T A/B (FFT36_MT_T forced, one noisy window): T=32 15.2,
  T=24 18.5, T=18 13.4, T=16 17.0, T=8 18.7 µs.  T=18 beat T=32 in that
  window but full-tournament runs kept picking split32 at 12.0–13.5 µs;
  wallaby login noise (sd up to 19%) is bigger than the T=18/32 gap, so the
  node tournament decides (both stay in the pool, now with T=12 too).
* CAUTION on the eff numbers now riding the description string: the serial
  denominator is timed with the pool's idle workers spinning (they only nap
  after ~10 ms), so ser=108 (B=32) / 199 (B=512) us/vol is inflated vs the
  true 1-thread numbers and eff > 1 appears.  Judge parallel efficiency
  against phase-1 node serials instead: B=1 113.4/(32×12) → the honest
  story is still "B=1 is coherence-bound" (r1's analysis stands).

### What did NOT work / negatives worth keeping

1. Nothing was killed outright this round — the three changes all installed
   on wallaby.  The r1 negatives stand (T=16/18 at B=1 on wallaby's single
   socket; the phase-1 node-closed mechanisms).  fused3-style separate z/y
   passes were NOT re-tried: L36_pfa r1 already priced 3-barrier fused3 at
   17.1–18.6 vs 13.7 µs at T=32 and I took that result instead of
   rediscovering it.
2. Wallaby cannot price: snt under a two-socket UPI (their node B=32 A/B
   read nt32 = 21.64 vs ip32 = 6.27 — NT from 16 remote threads is toxic
   when semi-cached), the cross-socket barrier at split32, T=16-vs-32
   everywhere.  All of these are candidate rows the node tournament ranks
   itself at create() time; none are hard-wired.

### Borrowed, and from whom (all also credited inline in the source)

* Spin pool, flag-array barrier, pin-to-OMP-map, all-post dispatch
  protocol, pool-shrink-after-pick: **L36_pfa mt_r1**, transitively
  L23_matrixsimd and L17_winograd mt_r1.
* Leftover-plane pair-split: **L36_pfa mt_r1** next-round idea (b) —
  they named it, I built it.
* NT-inverts-at-32-threads (snt bodies): **L23_matrixsimd mt_r1** via
  **L36_pfa mt_r1**'s B=512 confirmation.
* dsp probe through the description string: L36_pfa's r8/mt_r1 pattern.

### Predictions for the node (Gold 5218, 2×16, caller pages all socket 0)

* B=1: the OMP-fork tax the r1 split8 pick paid (~6+ µs of its 28.9) is
  gone and span is 1.5 waves; expect split{32 or 18} in the low 20s µs.
  Should close the 25.66-vs-28.95 gap to L36_pfa; whoever's barrier is
  cheaper wins the cell.
* B=32: vol32 with pool dispatch, ~5.3–5.5 µs/vol (pool saves ~0.15 µs/vol
  of OMP fork; snt should NOT be picked there — if the description says
  snt won the B=32 arena, suspect arena caching and re-check).
* B=512: if snt survives two-socket pricing, 14.2 → ~10–12 µs/vol
  (traffic −32% against a mixed local/UPI bandwidth); if the node instead
  picks dyn16-sntp (all-local NT from one socket), watch whether 16 local
  threads at 1.5 MB/vol beat 32 mixed threads at 2.2 MB/vol.

### Next

1. Read the node picks and dsp off the description strings, especially
   (a) split T at B=1, (b) whether any snt row survived on the node and at
   which T.
2. If B=1 is now barrier-floor-bound (2 × dsp ≈ 1.4 µs of ~20), the
   remaining lever is the coherence term: L36_pfa's socket-aware two-stage
   x-pass (per-socket DFT4 partials, only the 4×9 intermediate crosses UPI)
   is the only idea on the table that shrinks the all-to-all bytes; it
   needs a shared per-socket reduction buffer and one extra barrier —
   worth building only if the node shows split16 ≈ split32 (i.e. UPI-bound).
3. The move_pages question from r1 stands unresolved and un-shipped; it is
   still the single biggest B=512 lever (~2×) if the panel ever rules it
   legal.
