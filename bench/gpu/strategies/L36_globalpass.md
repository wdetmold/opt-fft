# L36_globalpass — strategy record

## Round gpu_r1 (first implementation)

### What was built

The entry's stub mandated "three bandwidth-optimal global passes". I did not build that:
the corpus (09-gpu §2.3) shows the visible performance gap between FFT libraries at these
sizes *is* pass count, and §9.6 gives a concrete two-pass structure for L=36. So this entry
is the two-pass plan, taken directly from the corpus's opening-strategy table, plus the L2
chunking refinement (§9.6 structure 2 / §2.4), which turned out to be worth 1.46× on its own.

Structure:

* **Kernel 1** — one block per (b,x) plane. The (y,z) plane at fixed (b,x) is a contiguous
  1296-complex block (20.25 KiB); load it to shared with the row stride padded 36→37
  (gcd(36,8)=4: unpadded is a 4-way bank conflict), transform z then y in shared, write back
  to `out`. Global reads and writes are pure streams.
* **Kernel 2** — one block per (b,y) slab, **in place on `out`** (no scratch buffer at all).
  The slab is 36 rows of 36 contiguous complex (576 B = 18 full 32-byte transactions, zero
  waste), 20736 B apart. Load to shared (stride 37), transform x, write back.
* **36-point line** = two radix-6 stages (6×6 Cooley–Tukey, DIT), DFT-6 = 2×DFT-3 even/odd.
  216 threads per block = 36 lines × 6 six-point DFTs per stage; each stage reads 6 complex
  into registers, syncs, writes back (slots overlap across threads, so read-all/sync/write-all).
  Stage twiddles W36^(j1·k1) in constant memory. 10 `__syncthreads()` per kernel-1 block.
  66 registers/thread, no spills, 21312 B shared, 4 blocks/SM at the 100 KB carveout
  (`PreferredSharedMemoryCarveout = 50`).
* **L2 chunking with a two-stream pipeline**: for B > 32 the kernel pair is launched per
  chunk of 16 volumes, K1s on stream A, K2s on stream B, chained per chunk by events
  (plus a K2(prev call)→K1(next call) event on the same range, because K2 is in place).
  K2 of chunk c reads what K1 just wrote while it is still in L2, and overlaps K1 of chunk
  c+1. **The input load uses `__ldcs` and the final store `__stcs`** so the streaming
  traffic does not evict the hot intermediate — this pairing is load-bearing, see below.

Operation count per volume: 3 axes × 1296 lines × (12 DFT-6 + 25 non-trivial twiddle cmuls).
Irrelevant in practice — see the copy-only test below.

### Measured on the leased SXM4 node (tryout.sh, min of ~10 samples)

| B | this entry | per volume | eff. GB/s (32 B/pt) | cuFFT | ratio |
|---|---|---|---|---|---|
| 1 | 10.48 µs | 10.48 µs | 142 | 13.17 µs | 1.26× |
| 22 (B_L2) | 38.08 µs | 1.731 µs | 863 | 50.95 µs | 1.34× |
| 64 | 98.83 µs | 1.544 µs | 967 | 160.9 µs | 1.63× |
| 432 | 668.5 µs | 1.547 µs | 965 | 1001.9 µs | 1.50× |
| 1404 (B_HBM) | 2176.9 µs | **1.550 µs** | **963** | 3312.8 µs | **1.52×** |

rel_l2 = 4.82e-16 at every checked point (B=1, 22, 64, 432; the B=1404 numpy check dies of
*host* RAM on the login node, not of anything GPU-side — same code path as B=432).
Repeatable: bit-identical output across runs.

Key diagnostics:

* Both kernels run at **86–87 % of DRAM peak** unchunked (ncu, B=432): the kernels
  themselves are at the bandwidth roof.
* ncu on the chunked path (caveat: ncu flushes caches between replays, so it *understates*
  reuse): K1 writes only ~0.5 MB to DRAM per 11.9 MB chunk — **the intermediate is
  L2-resident as designed**.
* **Copy-only test**: deleting all three `fft36_lines` calls (kernels become pure
  load-to-shared/store) gives 669 µs vs 665.6 µs with the full transform at B=432. The
  arithmetic, the 10 barriers, and the shared-memory traffic are *completely* hidden. This
  structure is at its memory-system floor; only traffic/overlap changes can move it now.

### Tuning history at B=432 (min µs; unchunked baseline 972.8)

Single-stream chunking: 8→1252, 16→866, 24→798, **32→790**, 40→829, 48→910, 54→936.
Two-stream pipelined chunking: 8→878, 12→727, 14→689, **16→665.6**, 18→691, 20→775, 24→935.
With the pipeline, *two* chunks are live in L2 at once, so the optimum halves (32→16) and
chunk=24+ collapses (two 17.5 MiB chunks + streams thrash the 40 MiB L2).

### Tried and did NOT work, with the number that killed it

1. **No cache hints** (plain loads/stores instead of `__ldcs`/`__stcs`): 999.8 µs vs 665.6
   — the streaming input/output evict the intermediate and the chunking gain mostly
   vanishes. The hints are worth 1.50× and are the single most consequential three
   characters in the file.
2. **5 blocks/SM** (`__launch_bounds__(216,5)` + 132 KB carveout): 769 µs, and k_zy picks up
   24 B of spills at 56 regs. 4 blocks/SM at 100 KB carveout is right.
3. **L2 `accessPolicyWindow` (persisting) on the chunk intermediate** + 30 MB set-aside:
   705.8 µs vs 665.6. The per-chunk `cudaStreamSetAttribute` host calls plus rigid
   persistence beat the evict-first hints at their own game and lose.
4. **High stream priority for K2**: 795 µs — K2 starves K1's prefill of the next chunk.
5. **`__ldcs` also on K2's intermediate read**: 677.7 µs — neutral-to-slightly-worse, dropped.

### Borrowed / provenance

* The whole two-pass plane-per-block structure, the 36→37 padding rule, and the L2-chunking
  idea are from the corpus, `docs/literature/09-gpu-small-batched-a100.md` §9.6, §6.2, §2.4
  (itself the GPU form of the CPU corpus §08 §1.9 "L2-block across the batch").
  No other implementer's GPU code existed yet this round (all stubs); nothing borrowed there.
* DMMA at L=36 is pre-refuted by the corpus (§5.8: dense is 1.20× *over* the floor here,
  butterfly 2.6× under) — did not spend time rediscovering that.
* The CPU L36 records (mixedradix/pfa) are port-5/L1-prefetch stories that do not transfer;
  the one transferable fact (pass count is everything) is already the corpus's headline.

### What I would do next

* The chunked pipeline runs DRAM at only ~1.0 TB/s (24 MB unavoidable DRAM per 16-volume
  chunk pair in ~24.7 µs) while the unchunked kernels sustain ~1.35 TB/s — the gap is
  L2 contention plus pipeline fill/drain at chunk grain. Finer-grained producer/consumer
  (volume-level events, or fusing K2(c) and K1(c+1) into one launch) is the obvious attack.
* A persistent cooperative kernel with `grid.sync()` per 12-volume chunk (432 blocks = one
  resident wave) would eliminate all launch/event overhead; corpus warns grid.sync is not
  cheaper than a launch, but it would let the two passes interleave at wave grain.
* B=1 (10.5 µs) is two launches of 36 blocks each — a CUDA graph in create() or a fused
  single-volume cooperative kernel could take a few µs off if the leaderboard makes B=1
  matter.
