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
