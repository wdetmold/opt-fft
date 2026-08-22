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
