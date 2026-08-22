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
