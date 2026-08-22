# L6_pfa — multicore strategy record

Phase-1 history (11 rounds, single-core) lives in `../geom/strategies/L6_pfa.md`.
Phase-1 final node numbers, for efficiency accounting: B=1 0.2068 us, B=4096
0.394 us/vol, B=32768 0.565 us/vol (panel_r11 leaderboard).

## Round mt_r1 — batch split + NT-store twins + 2D (kernel x T) race + spin-pool dispatch

### What I changed

1. **Batch-axis parallelism, static contiguous split.** Thread t gets volumes
   `[t*nb/T ...)` (balanced, uneven remainder to the low threads), running the
   unmodified phase-1 kernel over its chunk. Volumes are 3456 B = 54 cache
   lines, so chunk boundaries never share a line. Each thread has its OWN
   two-volume scratch (separate 6976 B allocation), allocated and
   first-touched by that thread inside `fft3d_create()`'s pool-warm parallel
   region under the harness's close/cores binding — so scratch is NUMA-local
   to the core that will use it. The caller's `in`/`out` are serially
   first-touched by the driver (I checked `driver.c`: `fread` + `memset`), so
   all caller pages sit on socket 0 and nothing I do changes that; the race
   below decides empirically whether cross-socket threads still pay.

2. **B=1 stays single-threaded, with the measurement.** The whole transform is
   ~0.13 us on wallaby (~0.21 us on the node); one empty OMP fork/join at 32
   threads measures **5.4–7.9 us on wallaby** (the `fork=` probe in every
   batched description). Even a 2-thread handoff costs a cache-line round
   trip per direction (~0.1 us) against ~0.07 us of halved work. The B=1
   process never spawns a thread, so the phase-1 path (race, 4K placement,
   boundary/clock probes) is byte-identical and B=1 measured **0.134 us,
   sd 0.28%** on wallaby — same as phase 1.

3. **NT-store twins re-enter for the multicore race** (`fused_nt_xa_d2`,
   `fused_pf_nt_xa_d2`; `STNT = _mm256_stream_pd` on the fused stage's output
   stores only — the six 32B stores per z-chunk cover exactly three whole 64B
   lines ascending, so WC buffers see full-line writes and `out` pays **no
   read-for-ownership**). Phase 1 rejected NT on the node (single core is
   fill-buffer-concurrency-limited, not bandwidth-limited; see
   `docs/literature/08-*`), but 32 cores saturate the bus, and killing the RFO
   removes 1/3 of all traffic. **Measured, wallaby, nb=65536 (DRAM regime):
   NT+pf wins its column at every T — 0.0333 us/vol at T=24 vs 0.0438 for the
   best regular-store kernel (−24%).** At nb=4096 (fits aggregate L2s) NT is
   correctly *rejected* by the race (0.0218 vs 0.0088 at T=32 — streaming
   past the cache when the problem lives in cache is poison). Idea lineage:
   phase-1 L8_batchsimd kept an NT variant in its arena race; the regime
   argument is literature section 08's.

4. **Plan-time 2D tournament** over (7 kernel rows) x (T in {1,2,4,8,16,24,32},
   clipped to `omp_get_max_threads()` capped at 32 — never more than given).
   Timed with the *exact* split-and-region code `fft3d_execute` runs,
   round-robin over all cells with per-cell minimum, per-column calibrated
   reps (~1.2 ms slices), 0.3 ms untimed dwell per slice, 100 ms licence
   settle spin first (all phase-1 discipline). Column pick is
   smallest-T-first with a 2% takeover margin; rows keep phase-1 safest-first
   margins. Kernel rows: the four prefetch/x-order d2 shapes, sp2, and the
   two NT twins.

5. **Race-arena cap raised from 16384 to 65536 volumes for the MT arm only.**
   The phase-1 cap (113 MiB) was chosen to exceed any single L3, but 32
   threads see L3 + 32 L2s of aggregate cache (~124 MiB on wallaby, ~54 MiB
   on the node): **measured on wallaby, the capped race arena ran 0.0159
   us/vol while the real 452 MiB batch ran 0.0434** — the race was crowning a
   cache-regime kernel (pfw, T=32) for a DRAM-regime problem. With the 65536
   cap the race picks NT at T=24 and the real batch improved **2828 -> 2238 us
   (−21%)**. This is the single biggest lesson of the round: *the aggregate
   cache, not the largest single cache, sets the race-arena floor once you
   are threaded.* Single-thread arm keeps 16384.

6. **Spin-wait pthread dispatch pool, raced against OMP dispatch.** GOMP's
   fork/join at 32 threads is 5–8 us — 15% of a whole B=4096 execute. The
   pool: master release-stores an epoch, 31 workers spin on it (`_mm_pause`),
   run their chunk, release-store padded per-worker done flags the master
   scans; ~1–2 us round trip. Workers are pinned via `pthread_setaffinity_np`
   to the affinity masks the OMP threads *reported* inside the pool-warm
   region (`pthread_getaffinity_np`), so thread t sits on the same core — and
   same socket as its first-touched scratch — under either dispatch. Because
   pool workers never sleep, the dispatch race is **sequenced, not
   interleaved** (OMP timed first with no pool alive; then the pool, after a
   3 ms dwell so GOMP's spinners have futex-slept), and the pool is torn down
   before `create()` returns unless it beats OMP by >2% — a losing pool never
   leaves a spinning core in the scored run. Total threads incl. master =
   tmax, never more. **Measured, wallaby B=4096: omp 0.0090 -> pool 0.0080
   us/vol; full execute 36.2 -> 32.8 us (−9%).** At B=65536 the saving is
   noise (0.2%) and the 2% margin correctly keeps OMP there sometimes; either
   is fine.

7. **Threaded-split correctness gate at create time**: the final
   (kernel, T, dispatch) config runs an odd 61-volume batch (uneven chunks,
   idle threads exercised) against the scalar reference, rel L2 <= 1e-13 or
   the plan falls back to the gate-proven T=1. All 20 kernels still pass the
   phase-1 per-kernel gate before being raced at all.

### Operation count

Unchanged from phase 1: Good-Thomas 2x3 per axis, 44 real flops per 6-point
line, 4752 flops/volume, no twiddle stage, no index tables. This round is
pure orchestration: threads, stores, dispatch.

### Measured (wallaby, Xeon Gold 6448Y, OMP_NUM_THREADS=32 close/cores)

| case | time | vs phase-1 single-core (wallaby race T=1 col) | pick |
|---|---|---|---|
| B=1 | **0.134 us** (sd 0.28%) | 1.00x (by design; fork alone is 40x the transform) | phase-1 path, single-threaded |
| B=4096 | **32.8 us = 0.0080 us/vol** | 0.2079 -> 0.0080: **26x, 81% parallel efficiency** | `fused_sp2_pf_xa` T=32, pool dispatch |
| B=65536 | **2199 us = 0.0336 us/vol** | 0.3395 -> 0.0336: **10.1x** (bandwidth wall) | `fused_pf_nt_xa_d2` T=24, ~200 GB/s sustained |

rel L2 vs numpy: 2.3–2.4e-16 in all cases. Repeatable across runs at all
three scored sizes.

B=65536 efficiency reading: 453 MB moved per execute in 2.2 ms is ~200 GB/s
— near one SPR socket's achievable bandwidth, so the last 22 cores buy
concurrency, not arithmetic; T=24 beating T=32 in the NT row says the bus is
already saturated with 8 idle cores. On the node (CLX, all caller pages on
socket 0, ~110 GB/s local + UPI remote) I expect the race to land on T=16 or
T=24 and NT to matter even more; the plan-time race decides on the node, not
me.

### What did not work / gotchas, with numbers

* **NT stores at cache-resident sizes**: 0.0218 vs 0.0088 us/vol at
  B=4096/T=32 — 2.5x worse. The race's regime detection (point 5) is what
  makes NT safe to carry at all.
* **zp-outer x order collapses under threading at B=4096**: `fused_pf_d2`
  (phase-1 B=1 winner's order) scales 0.2332 -> 0.0215 at T=32 while
  ascending `fused_pf_xa_d2` goes 0.2197 -> 0.0099. With 32 streams the L2
  streamer wants strictly ascending walks. zp-outer stays competitive only in
  the DRAM regime (nb=65536 column: 0.0489 vs 0.0486, a wash).
* **The 16384-volume race cap** (see point 5): −21% left on the table until
  raised; the number that killed it is 0.0159 (raced) vs 0.0434 (real).
* **Cross-process pick flips at tiny unscored batches** (B=8, B=33): the 2D
  race is noisy at microsecond batch sizes, different processes pick
  different (bit-differently-rounded) kernels, and `tryout.sh`'s cross-run
  bit-compare flags NOT REPEATABLE. Within one plan the output is
  deterministic (fixed split, no reductions) — the contract's actual
  requirement — and all three *scored* sizes pass the stricter check. Left
  as-is; worth remembering before panicking at a dev-box red flag.

### Borrowed

* NT-variant-in-the-race pattern: phase-1 `L8_batchsimd` (its arena race
  carried `FUSED-nt`); regime argument from `docs/literature/08-*`.
* All single-core kernels, margins, race discipline: my own phase-1 record.
* Context file for mt_r1 was empty (first multicore round) — nothing to take
  from rivals yet.

### Next round

* **Dispatch pool join**: master scans 31 flags serially; a 2-level tree join
  (4 groups of 8) might shave another ~0.5 us at B=4096. Measure first.
* **Node NUMA**: if the node race lands on T=16 (socket 0 only), try a
  socket-0-only pool with the master doing socket-1 prefetch — or accept that
  remote-socket threads add bandwidth via UPI and move on.
* **B=4096 on the node** sits near the 22 MiB L3 + 32 MiB aggregate-L2
  boundary; the sp2-vs-plain-fused choice may differ from wallaby's. The race
  covers it; check which row wins in the leaderboard description.
* If a rival finds a cheaper wake than the epoch broadcast (e.g. sequenced
  per-socket doorbells), steal it.

## Round mt_r2 — race what you ship: the 2D tournament moves onto the spin pool

### Where mt_r1 landed on the node, and the diagnosis

Node (mt_r1 leaderboard): B=1 0.210 us (1st), B=4096 0.009 us/vol (1st, but
**52.9% run spread**), B=65536 0.039 us/vol (1st, 19.9% spread). The per-run
JSONs explain the B=4096 spread completely: runs r2/r3 picked `fused_pf_d2
T=32 disp=pool` and scored **9.4 ns/vol**; run r1's race picked
`fused_sp2_pf_xa T=24` (raceBest 0.0220 vs 0.0216 — a coin flip) and scored
**14.4 ns/vol**. Root cause: the 2D race timed every cell through the OMP
fork/join (`fork=` reads **13–18 us on the node**, vs 4.5–5.5 on wallaby),
but the scored run then used the ~1–2 us spin pool. At B=4096 a call is
~40 us, so the fork tax flattened and distorted the T-columns — the race was
ranking configurations under a dispatch the winner would never use. The
B=65536 spread (r3 47.3 vs r1/r2 39.5 ns/vol, identical pick and setup) is
node-side page-placement luck, not a pick problem.

### What I changed

1. **The 2D (kernel x T) tournament now runs through the spin pool.** The
   pool is created BEFORE the race; every T>1 cell is dispatched with
   `l6_run_pool` (idle workers spinning during T<32 cells, exactly as in a
   scored pool run); T=1 cells stay direct calls (what a T=1 pick would ship
   as). OMP dispatch is only the fallback if pool creation fails, and the
   description records the race dispatch as `2D-raced@pool|omp`.
2. **Dispatch race re-sequenced** (pool exists first now): pool timed first
   — GOMP's workers have futex-slept through the multi-second tournament —
   then the pool is DESTROYED (workers joined, cores genuinely idle) before
   OMP is timed, and recreated only if it won by >2%. Strictly cleaner
   isolation than mt_r1's order in both directions.
3. **Pool join false-sharing fix**: mt_r1's done-flag array was
   `{_Atomic long; char pad[48]}` = **56 B per element** — `aligned(64)` on
   the array does not align elements, so adjacent workers' done flags shared
   cache lines and the join's 31 release-stores ping-ponged. Now 64 B each.
4. **Join-scan prefetch**: the master prefetches all 31 done-flag lines
   before scanning them, so the cross-core misses (each flag Modified in its
   worker's cache) overlap in the fill buffers instead of serializing.

### Operation count

Unchanged (Good-Thomas 2x3, 44 real flops per 6-point line, 4752
flops/volume, no twiddles, no index tables). Zero new arithmetic; the raced
kernel set is byte-identical to mt_r1. This round is selection integrity
plus dispatch overhead.

### Measured (wallaby, Gold 6448Y, OMP_NUM_THREADS=32 close/cores)

| case | mt_r1 | mt_r2 | pick |
|---|---|---|---|
| B=1 | 0.134 us | **0.137 us** (sd 2%, session band; path untouched) | phase-1 single-thread path |
| B=4096 | 32.8 us = 0.0080 us/vol | **25.5–27.0 us = 0.0062–0.0066 us/vol (−18–22%)** | `fused_pf_d2` T=32, pool |
| B=65536 | 2199–2238 us | **2224 us = 0.0339 us/vol** (same band; bandwidth wall) | NT row band unchanged |

rel L2 vs numpy: 2.34–2.45e-16 at B=1/2/33/4096/65536, bit-repeatable
across runs at all scored sizes. Setup at B=65536 dropped 12.6 s -> 7.9 s
(pool-raced cells don't pay the fork per call).

The pool-raced table at B=4096 is now clean and monotone — T=32 columns:
fused_pf_d2 0.0065, fused_pf_xa_d2 0.0065, fused_sp2_pf_xa 0.0067,
NT rows correctly poison (0.0209) — so the sp2/T=24 flip cannot recur: the
margin now acts on real, shipped-dispatch differences. Dispatch race:
omp=0.0071 vs pool=0.0064 us/vol (pool by 10%, kept).

### What did not move / honest attribution

* **A/B of the two pool fixes alone** (pad 48->56 + join prefetch reverted,
  pool-raced tournament kept): 27.11 us vs 25.5/26.4/27.0/27.0 us with the
  fixes — inside the wallaby session band, so the fixes are worth at most
  ~1 us of the ~7 us gain. The payload of this round is the PICK
  correction (pf_d2 T=32 over sp2), which the node already scored at 9.4 vs
  14.4 ns/vol; the pool fixes are kept because the padding one is simply a
  bug and the prefetch is free.
* B=65536 unchanged (0.0339 us/vol): the case is at the memory wall; no
  dispatch or pick change can move it from inside the process.

### Borrowed

* The join-scan overlap argument (flag loads overlapping in the fill
  buffers) is L23_matrixsimd's mt_r1 flag-array-vs-counter reasoning; their
  measured 5.0 us central-counter number is also why I did NOT switch the
  join to a fetch-add counter.
* The r1 mis-pick diagnosis is from the monitor's per-run description
  strings (t_L6_pfa_L6_B4096_r*.json) — the instrument-through-description
  discipline paying for itself.

### Next round

* **Tree join** (4 leaders x ~7 members) if the flat prefetch scan still
  shows in a node profile; expected <1 us, so measure before building.
* **Interleaved chunk assignment** at cache-resident sizes only
  (L6_unrolled's r1 idea #3): spreads each thread's footprint over both
  sockets' L3 slices on the node; would enter as a raced dispatch twin.
* **B=65536 r3-style page-luck** (47.3 vs 39.5 ns/vol, same pick): the only
  in-process defense I can see is a first-execute `move_pages(2)` QUERY of
  the caller's buffers and a locality-aware volume assignment; first
  execute lands in the driver's warmup, so the query would be unscored.
  Complexity is real — only if the node shows the luck again.
* Watch whether the node's B=65536 T-column moves under pool racing
  (mt_r1 picked T=32 disp=omp there; fork is 0.5% of a 2.6 ms call, so no
  change expected).

## Round mt_r3 — re-race on the real buffers; 3pass NT row adopted; setup cut 8x

### Where mt_r2 landed on the node, and the diagnosis

B=1 0.220 us (1st). B=4096 38.37 us/call (2nd by 0.07% to L6_unrolled — a
tie). B=65536 **lost**: 0.039 vs L6_unrolled's 0.035 us/vol, with a 106.5%
run spread and 12.8 s setup. The per-run JSONs explain the spread exactly:
all three runs picked the same kernel (`fused_pf_nt_xa_d2`), but r1's race
picked **T=32** (scored 39.4 ns/vol) while r2/r3's races picked **T=16**
(scored 81.3 ns/vol — 2x). Deeper: even at the same (kernel, T=32) config
the plan race's own arena read 57.5–80.4 ns/vol across the three processes
while the scored run on the driver's buffers ran 39.4. **The race arena can
sit in a different NUMA/page regime than the caller's buffers**, and when it
does, the flattened T-columns let the smallest-T-first 2% rule ship one
socket for a two-socket problem. My mt_r1/mt_r2 "page-placement luck" notes
were this same effect, undiagnosed. Meanwhile L6_unrolled's winning B=65536
cell used a `3pass_nt_pf` shape at 34.7 ns/vol — 12% ahead of my fused NT —
with a clean monotone tm curve.

### What I changed

1. **First-execute re-race ON THE CALLER'S BUFFERS** (`l6_mt_rerace`). The
   plan-time 2D tournament still runs in its arena and produces a pick, but
   it now also records its top-2 kernel rows and every T column whose winner
   raced within 2.5x of the best cell. The FIRST `fft3d_execute` — inside
   the driver's discarded warmup, hence unscored — re-races those configs on
   the real `in`/`out`, through the SHIPPED dispatch (pool if the plan kept
   it — race what you ship, mt_r2's own rule), with the plan's pick as the
   incumbent under the same margins (rival row needs its own margin, columns
   smallest-T-first at 2%). Every trial computes the full correct transform
   over the whole batch, and a final run of the winner leaves `out` holding
   the shipped config's bits, so output is identical from execute 1 onward
   within a plan. Cost: ~30 ms at B=4096, ~200 ms at B=65536, all in warmup.
   Forced picks (`L6_FORCE`/`L6_FORCE_T`) skip the re-race — an A/B must
   stay forced. Result rides the description as
   `exre=<kernel>,T=<n> <us/vol> (plan <kernel>,T=<n>)`.
   This is the round's payload: on the node, a r2/r3-style flattened arena
   race can still hand over a T=16 pick, but the re-race then measures T=24
   and T=32 where they actually run (~39 ns, not 74+) and takes over.
2. **3pass NT row added** (`3pass_pf_nt_xa_d2` = ascending x + T0 input
   prefetch + NT z-pass stores + d2 codelet), ADOPTED FROM **L6_unrolled
   mt_r2**, whose `3pass_nt_pf` took the node B=65536 cell at 34.7 vs my
   fused NT's 39.4 ns/vol. Their VD6 codelet is my d2 graph, so this is
   their winner in my generator vocabulary (GEN_3P grew PX/CD parameters;
   k_3p's own codegen is unchanged). Entered as a trailing 1.5% challenger
   behind my proven fused NT rows.
3. **Plan-race trim at nb >= 16384**: T columns below 8 are dropped (no tm
   or tournament curve on either machine has ever put T<8 within 2x of the
   winner in the DRAM regime, and a single T=1 slice at nb=65536 is 41 ms)
   and rounds go 7 -> 5. raceT1 for the efficiency report is now one warm +
   one timed T=1 call. **Setup at B=65536: 12.8 s -> 1.5 s.** The trim is
   guarded (tmax >= 8), so a squeezed team still races everything it has.

### Operation count

Unchanged (Good-Thomas 2x3, 44 real flops per 6-point line, 4752 flops per
volume, no twiddles, no index tables). The 3pass NT row is a re-parameterized
existing shape; the re-race adds zero arithmetic to any scored execute.

### Measured (wallaby, Gold 6448Y, OMP_NUM_THREADS=32 close/cores)

| case | mt_r2 | mt_r3 | pick (wallaby) |
|---|---|---|---|
| B=1 | 0.137 us | **0.138 us** (sd 0.07%; path untouched) | phase-1 single-thread |
| B=4096 | 25.5–27.0 us | **25.1–26.7 us = 0.0061–0.0065 us/vol** | `fused_sp2_pf_xa` T=32 pool; re-race confirms T=32 |
| B=65536 | 2224 us | **2202–2216 us = 0.0336–0.0338 us/vol** | plan: `3pass_pf_nt_xa_d2` T=24; re-race ships `fused_pf_nt_xa_d2` T=24 (0.0340 vs 0.0355 in its column) |

rel L2 vs numpy: 2.38–2.45e-16 at B=2/8/33/4096/65536, 2.24e-16 at B=1.
B=65536 verbose re-race table (real buffers): 3pass 62.6/38.6/35.5/33.7 ns
at T=8/16/24/32; fused NT 52.0/36.6/34.0/33.7 — the two NT shapes are within
noise of each other on wallaby's SPR; the node (where L6_unrolled's 3pass
won by 12%) will resolve the pair, and now does so on the REAL buffers.
Parallel efficiency: B=65536 raceT1 0.538 -> 0.0336 us/vol shipped = 16x on
32 (bus-bound, both sockets' fill buffers engaged); B=4096 0.221 -> 0.0061 =
36x on 32 (superlinear vs the T=1 DRAM column because 32 L2s hold the 27 MiB
working set — same aggregate-cache mechanism as the mt_r1 arena lesson).

### What did not work / gotchas, with numbers

* Wallaby cannot reproduce the node's arena-vs-real distortion: arena raced
  0.0318 us/vol vs real-buffer 0.0337 at the same config (-6%, mild and
  rank-preserving), versus the node's r2/r3 +90% rank-flipping version. So
  the re-race's value is proven by mechanism (it re-measures the exact cell
  that mis-picked, where it mis-picked) rather than by a wallaby A/B; the
  node leaderboard is the real test. Pre-registered expectation: three
  near-identical B=65536 runs at ~39 ns/vol or better, spread back to a few
  percent, `exre=` showing T=24/32 shipped in every process.
* Cross-process pick flips at dev sizes still flag `NOT REPEATABLE` in
  tryout (B=1 flagged once, then repeatable on rerun; B=4096 flagged once):
  same mt_r1 artifact, now also reachable via the re-race when two configs
  are within noise. Within one plan the output is deterministic (fixed
  split, no reductions, re-race locks after execute 1) — the contract's
  requirement — and the node's scored history shows stable picks there.
* The 61-volume threaded gate tests the PLAN's config; the re-race can ship
  the runner-up row instead. Accepted deliberately: the gate's subject (the
  split/pool handoff protocol) is kernel-independent, both rows passed the
  4-volume per-kernel gate, and every re-race trial computes the full batch
  through the same protocol before anything is shipped.

### Borrowed

* `3pass_nt_pf` as a B=65536 NT shape: **L6_unrolled mt_r2** (their node
  win, 34.7 vs 39.4 ns/vol). Built here as `3pass_pf_nt_xa_d2`.
* "Race what you ship" applied to the re-race dispatch: my own mt_r2 rule.
* The unscored-first-execute idea is my mt_r2 "next round" note (the
  `move_pages` query proposal), simplified: instead of querying pages and
  reasoning about locality, measure the candidates where they will run.

### Next round

* Read the node's `exre=` strings first: if the re-race overturned the plan
  pick in any process, the arena distortion is confirmed on the node and the
  plan race can be slimmed further (it only needs to nominate, not decide).
* If 3pass vs fused NT resolves cleanly on the node's real buffers, prune
  the loser (r10 rule).
* B=4096 is now a three-way tie band with L6_unrolled (both ~9.4 ns node);
  the remaining daylight there is the ~1–2 us pool round trip — the 2-level
  tree join idea from mt_r1 remains unmeasured and is the only lever left.
* If a rival's record shows a cheaper way to bind the re-race to the warmup
  (or a driver change makes execute 1 scored), the re-race must grow a time
  budget guard; today it is ~200 ms worst case.

## Round mt_r4 — deep-streaming pin: wide NT installed, not raced; dwell, don't measure cold

### Where mt_r3 landed on the node, and the diagnosis

B=1 0.211 us (1st, takes the cell). B=4096 0.0092 us/vol (1st, 2.23x MKL, a
2% tie with L6_unrolled). B=65536 **lost 0.84x to fftw3_patient's min**
(0.0788 vs 0.0660 us/vol) — though the VERDICT notes the panel wins that
cell 1.55x on medians (FFTW alternates 4.3/8.0 ms modes within one process
and is scored on its lucky minimum; my 5178 us median has sd 0.2%).

The mt_r3 re-race did exactly what it was built to do — 5161/5173/5165 us,
0.2% spread, lottery dead — **and the fix was worthless because it measured
its way into the slow regime and locked it in.** The assembled evidence:

* My re-race read T=16 and T=32 **tied at ~84 ns/vol (87.7 GB/s)** on the
  real buffers in all three processes, and shipped T=16 in two of them.
* But mt_r2's r1 process had `fused_pf_nt_xa_d2 T=32 disp=omp` sustain
  **39.4 ns/vol (200 GB/s) with median == min for the whole process**, and
  L6_unrolled's mt_r2 sustained 34.6 ns 3/3 at `3pass_nt_pf T=32 omp`. Same
  node, same driver, same buffers-by-construction.
* mt_r3's panel-wide `fr` instruments read **fr=0 everywhere**: the caller's
  pages never migrate (VERDICT s5). So the fast regime is NOT page
  placement — it behaves like machine/process state that only materialises
  under *sustained* wide streaming, and fftw's within-process bimodality
  (4.3 ms mode = ~157 GB/s, 8.0 ms mode = the same ~85 GB/s wall I hit)
  says the fast mode exists dynamically even in the r3 node state.
* A cold ~5 ms re-race slice can never see a mode that needs seconds of
  sustained wide traffic to appear; a T=16 ship then forecloses it for the
  entire scored run. That is my r3: shipped T=16, scored 78.8 stable.
* The one entry that got >150 GB/s in ALL THREE processes
  (L36_mixedradix, 150.9 GB/s, the project's best streaming number) did
  it by **pinning the wide NT shape at deep cells without racing team
  width, plus a ~3 s create-time dwell in that config**. The VERDICT's
  refined rule: "at DRAM-bound cells, install from the working set and
  don't race at all" — every entry that did this killed its lottery.

### What I changed

1. **Deep-streaming pin.** When the caller working set (in + out =
   nb x 6912 B) exceeds 192 MiB — past aggregate cache on the node
   (~76 MiB) and wallaby (~124 MiB), so B=65536 yes, B=4096 no —
   `fft3d_create()` skips the 2D tournament entirely and installs
   **3pass_pf_nt_xa_d2 at T=tmax** (the shape holding the node's
   fast-regime record: L6_unrolled's 34.7 vs my fused NT's 39.4 ns/vol).
   In the slow regime everything ties at ~84-88 ns (my own r3 re-race
   table), so the pin costs nothing there; in the fast regime it is worth
   ~2x. Team-width racing at this cell is deleted, per the VERDICT's L=6
   order ("the team-width question at L=6 is closed — delete the shrink
   race there"). The dispatch race (omp vs pool) and the 61-volume
   threaded correctness gate still run on the pinned config.
2. **Create-time wide dwell, ~3 s** (ADOPTED FROM **L36_mixedradix mt_r3**,
   who adopted the dwell diagnosis from L36_pencilfused mt_r2): the
   steady-state tail at deep cells now dwells ~3.0 s (was 3 ms) running
   the exact shipped config on the race arena, so the driver's warmup
   begins in a process that has been streaming wide for seconds. Setup is
   unscored; setup_seconds goes ~2.6 -> ~3.4 s.
3. **Re-race reduced at deep cells: kernel shape only, never team width.**
   `l6_mt_rerace` still runs in the driver's discarded first execute, but
   at deep cells it races exactly two rows — 3pass NT (incumbent, margin
   0) vs fused NT (1.5% takeover) — at the single column T=tmax, **after a
   0.5 s wide dwell on the real buffers** so both rows are priced in the
   sustained-wide state the scored loop runs in. Non-deep cells keep the
   full mt_r3 re-race (top-2 rows x surviving T columns) unchanged.
4. Description gains `pin=<0|1>`. Everything else — B=1 single-thread arm,
   B=4096 tournament + re-race, pool, margins — is byte-identical.

### Operation count

Unchanged (Good-Thomas 2x3, 44 real flops per 6-point line, 4752 flops per
volume, no twiddles, no index tables). This round is pure selection policy
and process-state management; zero new arithmetic, zero new kernels.

### Measured (wallaby, Gold 6448Y, OMP_NUM_THREADS=32 close/cores)

| case | mt_r3 | mt_r4 | pick |
|---|---|---|---|
| B=1 | 0.138 us | **0.133 us** (sd 0.12%; path untouched, session band) | phase-1 single-thread |
| B=4096 | 25.1–26.7 us | **25.07 us = 0.0061 us/vol** (path untouched) | tournament as before |
| B=65536 | 2202–2216 us | **2208–2212 us = 0.0337 us/vol** | PIN `3pass_pf_nt_xa_d2` T=32; re-race shipped `fused_pf_nt_xa_d2` T=32 (0.0337 vs 0.0348 on the real buffers — the two NT twins stay a coin flip on SPR) |

rel L2 vs numpy: 2.34–2.42e-16 at B=1/33/4096/65536, bit-repeatable across
runs at every size tried. Wallaby's close binding is single-socket, so it
CANNOT price the wide-vs-narrow question the pin answers; the numbers above
only show no regression.

**The dev measurement that matters** (manual run of the tryout-built binary
with `OMP_PROC_BIND=spread`, 32 threads across wallaby's two sockets — the
node's topology, DEV ONLY, not the harness binding): B=65536 runs
**1029 us = 0.0157 us/vol (~440 GB/s on DDR5), 2.15x faster** than the same
binary close-bound, PASS 2.4e-16. The pinned wide path, the contiguous
split and the NT stores all hold up when the team genuinely spans two
sockets — which is what the harness gives the node run by default.

### What did not work / risks, stated honestly

* **Wallaby cannot falsify the pin.** Its harness binding is one socket, so
  the wide-vs-narrow and dwell effects are invisible there (0.0337 both
  ways). The bet is placed on node evidence only: mt_r2's sustained 39.4
  (mine, T=32) and 34.6 (L6_unrolled, 3/3), against the known worst case —
  L6_unrolled's r3 wide reading of 96 ns vs my narrow 84 (a -14% downside
  if the slow regime is real AND wide is genuinely worse this time, though
  the VERDICT suspects their 96 was self-inflicted by a leftover race
  pool, a mechanism I don't have: my pool is destroyed before OMP timing,
  and my r3 read wide == narrow, not worse). Asymmetric bet: lose <=14%
  worst case, gain ~2x if the fast mode is reachable, tie otherwise.
* **Pre-registered expectation for the node:** three B=65536 processes
  within a few percent of each other, `pin=1`, `exre=` showing a T=32 NT
  ship in every process; ~39 ns/vol or better if the fast mode
  materialises under the dwell, ~84 ns if the machine refuses. If it
  reads ~96 (worse than r3's 78.8), the pin is refuted and mt_r5 should
  restore the narrow column to the deep re-race.
* The B=1 min of one noisy session read 0.163 us (sd 36%) before a rerun
  gave 0.133 (sd 0.12%) — wallaby login noise, path untouched; noted so
  nobody bisects a phantom.

### Borrowed

* Pin-plus-dwell at deep-streaming cells: **L36_mixedradix mt_r3**
  (v1-vol32-sntp pinned + 3 s dwell -> 150.9 GB/s in 3/3 processes), whose
  dwell in turn credits L36_pencilfused mt_r2's diagnosis.
* "Install from the working set, don't race at all" + "delete the shrink
  race at L=6": the mt_r3 VERDICT (s5, s6), adopted as written.
* 3pass NT as the pinned shape: L6_unrolled mt_r2's node record (34.7),
  already in my generator vocabulary since mt_r3.

### Next round

* Read the node `exre=`/`pin=` strings first. Outcomes: (a) ~39 ns or
  better, stable -> the fast mode is dwell-reachable; keep, and try
  trimming the 3 s dwell to find the threshold. (b) ~84 ns stable -> the
  machine's fast mode was absent this round; the pin still killed the
  lottery, keep it and stop spending rounds here (the cell is then
  bandwidth-capped for everyone; FFTW's lucky min is a statistic problem,
  not a kernel problem). (c) worse than 84 -> pin refuted, restore the
  narrow column at deep cells.
* Watch L6_unrolled's impl_2-vs-impl_3 bisect (the VERDICT ordered it):
  if the mechanism is "never leave a spin pool alive at a streaming
  cell", audit my pool lifecycle again under their finding.
* B=4096 remains a 2% tie: the ~1-2 us pool round trip (2-level tree join,
  unmeasured since mt_r1) is still the only visible lever; only touch it
  with a same-round A/B, since the cell is currently won.
