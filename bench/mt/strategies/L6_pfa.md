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
