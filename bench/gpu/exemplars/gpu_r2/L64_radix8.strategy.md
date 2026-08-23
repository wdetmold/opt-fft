# L64_radix8 — strategy record

## Round gpu_r1 (first implementation — the file was a stub before this round)

### What was built

Two global passes, L2-blocked across the batch, no workspace buffer at all:

* **kernel_yz** — one block per (b,x) plane, 512 threads = 64 lines × 8 lanes.
  z-axis lines are FFT'd straight from global memory into registers (loads are
  128-byte coalesced because z is contiguous), then ONE padded shared-memory
  transpose (66.5 KB dynamic, row stride 65 double2), then the y-axis lines are
  FFT'd from the now-contiguous shared rows and written to `out` in standard
  layout (64-byte segments, zero sector waste).
* **kernel_x** — x-axis (stride L² = 64 KiB), **in place on `out`**: each lane
  reads and rewrites exactly its own 8 elements, so the pass is race-free without
  any intermediate buffer. No shared memory. Loads/stores are 4-consecutive-z per
  lane group → 64-byte segments, every 32-byte sector fully used.
* **The 64-point line FFT is register-resident across 8 lanes** (a 64-point line
  cannot live in one thread: 256 registers of data > the 255 ceiling). With
  j = 8j1+j2 (lane t = j2), k = k1+8k2:
  `X[k1+8k2] = DFT8_j2( W64^(j2·k1) · DFT8_j1( x[8j1+j2] ) )`.
  The j2↔k1 exchange is an 8×8 xor-butterfly **shuffle transpose** (3 stages of
  `__shfl_xor_sync`, register indices all compile-time), so a line FFT touches
  neither shared nor local memory. Per-lane twiddles W64^(t·k1) come from a
  64-entry device table computed in create() with exact binary angles
  (cos/sin of −π·j/32), loaded once per kernel into 7 registers.
* **Bank conflicts**: the shared tile uses stride 65 *double2* with forced
  16-byte accesses. For 16-byte shared accesses a quarter-warp (8 lanes) is one
  transaction; odd double2 stride makes both the strided stores (bank group
  4t mod 32, all distinct) and the contiguous reads (8 consecutive double2 =
  all 32 banks) conflict-free. ncu confirms: ~12K conflicts per million shared
  stores (~1%), zero on loads. NOTE: the textbook "pad 64→65 doubles" recipe is
  stated for 4-byte elements; for double2 you must either pad in double2 units
  and force 128-bit accesses (done here) or split re/im into separate double
  arrays. Padding 64→65 doubles with 2×8-byte accesses is still 8-way conflicted.
* **L2 blocking across the batch** (the literature's structure 3, and the round's
  biggest lever): the batch is processed in chunks of 4 volumes, alternating over
  two non-blocking streams with a fixed chunk→stream map (so back-to-back
  executes stay ordered). kernel_x's read+rewrite of the intermediate then hits
  L2 instead of HBM. This did NOTHING (3318 vs 3335 µs) until the streaming
  accesses got **evict-first cache hints**: `__ldcs` on the input read, `__ldlu`
  on kernel_x's read, `__stcs` on the final store. With hints: 3004 µs. ncu
  verification: kernel_x moves 17.5 MB of DRAM per chunk whose naive traffic is
  33.6 MB — the intermediate read is L2-resident as designed; whole-run effective
  HBM traffic ≈ 42% below unchunked.
* Small batches: B ≤ chunk uses a monolithic path with cached (`__ldg`) input
  loads and a half-tile kernel_x (256 threads, 2×blocks) so B=1's 128 blocks
  cover all 108 SMs; B=4 uses chunk=2 so the two passes overlap across streams.

### Operation count

Nominal 15·log2(64) = 90 flop/point (benchFFT convention). Per thread per line
FFT the SASS is 132 FP64 (88 DADD + 30 DFMA + 14 DMUL) + 24 SHFL.32-pairs
(48 SHFL) + 8–16 LDG/STG.128. Arithmetic is NOT the ceiling: the FP64-pipe floor
is ~2.7 µs/volume against the 2-pass HBM floor of ~8.4 µs/volume at 2 TB/s.

### Measured (leased SXM4 A100 on the reserved node, tryout.sh)

| case | this entry | cuFFT | ratio |
|---|---|---|---|
| B=1   | **22.5 µs** (median 24.1) | 23.5 µs | 1.04× |
| B=4   | **48.7 µs** (12.2 µs/vol) | 58.0 µs | 1.19× |
| B=256 | **2995 µs** (11.70 µs/vol, sd 0.03%) | 3740 µs | 1.25× |

rel_l2 = 4.2e-16 at every point, bit-identical across runs. Effective bandwidth
at B=256: 4.29 GB nominal traffic / 2.995 ms = 1.43 TB/s moved, of which ~58%
comes from HBM thanks to the L2 chunking.

### Tried and rejected, with the number that killed it

* **Shared-memory carveout 100% (164 KB) to host 2 yz blocks/SM**: 3335 → 3615 µs
  at B=256. Doubling occupancy is not worth shrinking L1 to 28 KB — the 64-byte
  segment stores and twiddle loads live in L1. Do not repeat this.
* **kernel_x as 256-thread half-tiles at large batch**: 2999 → 3134 µs. Kept only
  for the small-batch path where B=1 needs the block count.
* **Chunk-size scan** (with hints, B=256): C=2 3118, **C=4 2994**, C=6 3355,
  C=8 3435. Two chunks in flight × 4 volumes × (in+out) ≈ L2 capacity is the
  sweet spot; C=8 (64 MiB in flight) thrashes.
* **chunk=1 at B=4**: 67.0 µs vs 48.7 at chunk=2 — 64-block kernels are too small
  even with stream overlap.
* **Chunking without cache hints**: no gain at all (3318 µs). The hints are not a
  polish, they are the mechanism.
* **`__launch_bounds__(512,2)` on kernel_yz** (forces 84→64 regs, ~52 B spill):
  neutral monolithic, but in the chunked path it lets one yz block (32K regs)
  and one x block (32K regs) co-reside on an SM; final config runs at sd 0.03%.
  Same bound on the natural-64-reg kernel_x is free.
* **In-place kernel_x replacing a separate tmp buffer**: neutral at B=256, big
  enabler at B=4 (working set drops to exactly the 32 MiB that fits L2) and it
  is what makes the L2-chunk trick pay (the rewrite re-dirties the same lines
  instead of touching a third buffer).

### Borrowed

No other GPU strategy records existed this round (first GPU round, this entry
was a stub). Everything structural is from `docs/literature/09-gpu-small-batched-a100.md`
§9.8: structure 1 (plane-per-block, pad 64→65), structure 2 (radix-8² with the
8-lane register codelet), structure 3 (L2 chunking over ~4 volumes), plus the
§0.2 rules (128 B/thread loads, odd shared strides). DMMA was not attempted:
§9.9 rates it "no for the whole axis" at L=64 and the entry is nowhere near
FLOP-bound.

### What I would do next

1. **kernel_x is the weak kernel**: 12.6% compute, 31% DRAM, ~59% memory-pipe
   SOL — the scattered 16-byte accesses cost L1 wavefronts. Two candidates:
   (a) replace the shuffle transpose with a per-warp 8×9-padded shared scratch
   (16 LDS/STS.128 + 2 syncwarp vs 24 SHFL.32-pairs, shorter dependency chain);
   (b) a fully shared-tiled kernel_x with 128-byte coalescing on both sides.
   Both change SM co-residency with kernel_yz — measure, don't assume.
2. B=1 medians wobble ~10% through tryout while a direct single-process run is
   stable (22.7 min / 22.7 median); understand before trusting any B=1 delta.
3. `cudaAccessPolicyWindow` (persisting) on the intermediate chunk instead of /
   in addition to the evict-first hints; might allow C=6–8 and bigger grids.
4. A grid-synchronized cooperative single kernel for B=1 to drop one launch.
5. The two kernels are issue-latency-bound at ~32 warps/SM, not bandwidth-bound:
   the theoretical composite floor is ~1.4–1.6 ms at B=256 against our 3.0 ms,
   so there is still ~2× on the table if the stalls can be hidden.

## Round gpu_r2

### Standing at the start

Led all three L=64 points in gpu_r1 (B=1 1.04×, B=4 1.20×, B=256 1.25× over
cuFFT). This round executed r1's own "next" list items 1(b) and 4 (as a graph,
not a coop kernel), plus ideas from the other entries' r1/r2 records.

### What changed

1. **kernel_x rebuilt as a fully shared-staged plane** (r1 candidate 1(b)).
   ncu at B=64 chunked showed kernel_x at 79% L1TEX SOL (its roof) with 39% of
   warp cycles in long-scoreboard stalls, while DRAM sat at 31%: the direct
   register↔global accesses were 64B segments, costing 2× the L1 wavefronts.
   New form: block cooperatively loads the whole (x,z) plane at fixed (b,y)
   into padded shared (stride 65 double2, 66.5 KB — needs the
   MaxDynamicSharedMemorySize attribute now) with 128B-coalesced rows of
   consecutive z, lanes read their strided line elements from shared, same
   register radix-8² line FFT, write back to shared, cooperative 128B store.
   Key detail: between the line read and the line write-back NO barrier is
   needed — lane t reads and rewrites exactly the slot set {rows 8j+t, col g},
   disjoint across lanes — so the kernel has only 2 `__syncthreads()`.
   ncu after: L1TEX 79→46%, kernel 36.8→33.3 µs solo.
2. **kernel_yz's final store staged through the (already-allocated) shared
   plane** the same way: the r1 direct store was 64B segments and ncu flagged
   2.4-of-4 sectors per L2 store line; the plane in the standard layout is a
   contiguous 64 KB run, so a shared round-trip (+2 barriers) makes the global
   side perfectly coalesced. ncu: L1TEX 59→43%, kernel 42.2→39.9 µs solo.
   Together, 1+2 measured **3013 → 2765 µs at B=256** (−8%), 818→688 at B=64.
3. **Chunk re-scan after the balance shift** (the r1 sweet spot moved):
   C=2 2900, **C=3 2622–2642**, C=4 2749–2768, C=6 3144 µs at B=256.
   Adopted C=3: **2619 µs = 10.23 µs/vol**.
4. **B=1 runs as a captured CUDA graph** (borrowed from L36_sharedtiled r1,
   who measured graph > plain > fused-cooperative; I did not re-discover the
   cooperative failure). Graph keyed on (in,out), recaptured on pointer
   change so the driver's poke test stays safe. **22.5 → 18.5 µs**.
5. **B=2..8 split one-chunk-per-stream over 4 streams** (borrowed from
   L36_sharedtiled r2's batch-split lesson: never ≥2 chunks per stream at
   small B). B=4: 48.7 → **42.1 µs** (10.5 µs/vol) — most of the gain was
   from the kernel work (43.3 before the split, so the split itself is ~3%).

### Operation count

Unchanged: nominal 15·log2(64) = 90 flop/point; the line FFT SASS is the same
132 FP64 + 48 SHFL. Added per point: 2 shared round-trips (kernel_x) and 1
(kernel_yz), which the latency budget absorbs — compute stays ≤23% SOL.

### Measured (leased SXM4 A100, tryout.sh)

| case | gpu_r1 | gpu_r2 | per volume | cuFFT | ratio |
|---|---|---|---|---|---|
| B=1   | 22.5 µs | **18.5 µs** | 18.5 µs | 23.8 µs | 1.29× |
| B=4   | 48.7 µs | **42.1 µs** (sd 0.1%) | 10.5 µs | 58.3 µs | 1.38× |
| B=256 | 2995 µs | **2619 µs** (sd 2%) | 10.23 µs | 3741 µs | 1.43× |

rel_l2 = 4.16–4.17e-16 at B=1,2,3,4,5,8,9,10,64,256; bit-identical across
runs at every point; compute-sanitizer memcheck clean on all three execute
paths (graph B=1, split B=5, chunked B=10 incl. the C=3 tail chunk of 1).

### Tried and rejected, with the number that killed it

* **`__launch_bounds__(256,4)` on the half-tile kernel_x** (force 4 blocks/SM
  at 64 regs): B=4 42.1 → 45.2 µs, B=1 flat. The register squeeze costs more
  than the extra blocks buy; (256,2) is right.
* **Full-tile kernel_x<0> in the small-batch split path**: B=4 42.2 → 43.5 µs
  min with a much worse median. Half-tiles' 128-block grids win below ~8
  volumes; kept the HALF template split exactly as in r1.
* **C=4 kept against the new kernels**: 2749–2768 vs 2622–2642 µs — re-scan
  after any kernel-balance change; the r1 "C=4 sweet spot" did not survive.

### Borrowed

* **L36_sharedtiled r1**: CUDA-graph-per-(in,out) at B=1, and their measured
  graph-beats-cooperative result (saved me re-testing grid.sync; L36_globalpass
  and the PERKS note in the corpus agree).
* **L36_sharedtiled r2**: the one-chunk-per-stream batch split at small B,
  including the warning that ≥2 chunks/stream is worse than unchunked.
* **L36_globalpass r1 + L36_sharedtiled r2**: accessPolicyWindow persisting-L2
  loses to evict-first hints (705.8 vs 665.6 µs at their scale, reconfirmed by
  sharedtiled) — r1's "next" item 3 dropped without spending a run on it.
* **L45_pfa r1**: the >200-reg thread-per-line failure and unit-parallel
  lesson — used as the reason NOT to attempt a one-thread-per-line variant,
  and context for why 64 regs/thread is this entry's occupancy wall.

### What I would do next

1. **The RF wall is the frontier**: both kernels sit at 50% theoretical
   occupancy because 8 complex doubles + 7 twiddles = 64 regs/thread fills the
   256 KB register file at 1024 threads/SM, and both are still latency-bound
   (No Eligible 78–82%, everything ≤51% SOL). A radix-4³ codelet (16 lanes ×
   4 complex = ~40 regs) is the only way to more warps, but the shared plane
   (66.5 KB) then caps blocks anyway — the win is not obvious, measure.
2. Chained on-the-fly twiddles (w^k by iterated multiply, 4 regs instead of
   14) might drop below 56 regs; only pays if the compiler turns it into
   occupancy, which `__launch_bounds__` experiments this round say is doubtful.
3. B=256 sd is ~2% through tryout (boost clocks); the monitor's pinned-clock
   window should read ~2620 µs. If it reads >2700, suspect the C=3 tail chunk.
4. Producer/consumer fusion of kernel_x(chunk c) with kernel_yz(chunk c+1)
   (L36_sharedtiled r2's suggestion) is the remaining structural idea for the
   HBM point; the launch-grain L2 pipeline is otherwise tuned out.
