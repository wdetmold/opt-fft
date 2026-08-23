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

## Round gpu_r3

### Standing at the start

Led all three L=64 points in gpu_r2 (B=1 1.29×, B=4 1.39×, B=256 1.43× over
cuFFT, 2617.5 µs on the monitor at B=256). This round executed the one
structural idea every large-L record pointed at — the fused persistent
producer/consumer kernel, proven by L36_globalpass r2 at −13% — and it LOST
here. The numbers below are the honest account; the round's net keep is a
tail-chunk fix and three negative results that close the fusion door for this
entry's current kernel shape.

### What was built: fused persistent kernel (kept, but not default)

`kernel_fused`, borrowed structurally from **L36_globalpass gpu_r2**: one
launch per execute, grid = one resident wave (occupancy-probed: 2 blocks/SM ×
108 = 216 blocks of 512 threads, 66.5 KB shared). Blocks loop pulling tickets
from a global atomic; ticket = one yz (b,x) plane or one x (b,y) plane;
schedule = yz(0..LEAD−1) runway, then alternating yz(v+LEAD)/x(v), then the x
tail. Per-volume done counters (yz releases with store→threadfence→sync→
atomicAdd, x polls with __nanosleep); host-side epoch bases in mod-2^32
arithmetic, no counter resets (also L36_globalpass's machinery). Deadlock-free
because the grid is one resident wave and every yz(v) ticket precedes every
x(v) ticket in dispatch order. Correct (4.44e-16), memcheck clean,
bit-identical re-runs. Runs under `L64_MODE=1`, lead under `L64_LEAD`.

**It measured slower than the r2 chunked launch pipeline at every batch:**

| case | chunked (default) | fused best (lead=2) |
|---|---|---|
| B=64 | 662 µs (min, sd 0.1% on-lease) | 713 µs (sd 0.03%) |
| B=256 | 2626–2631 µs min | 2765 µs min / 2775 median |

Lead sweep at B=64 (lean-twiddle build): **2 → 713**, 3 → 726, 4 → 801,
6 → 984 µs. Grid must be the full 216-block wave: 108 → 809, 432 → 891.
The fused kernel's ncu picture at B=64/lead=2: DRAM reads at the compulsory
floor (276 vs 268 MB) but **writes 493 MB vs a 268 MB floor** — the
intermediate leaks to HBM about as much as the chunked path's does (r1
measured ~50% leak there too), so fusion did NOT buy the traffic win here the
way it did at L=36. Top stalls: barrier 24.7%, long-scoreboard 12.7%, membar
4.6%. The deficit is the ticket loop itself: 5 syncs + 1 fence + 1 atomic +
spin per 64 KB plane, on kernels that were already latency-bound at 50%
occupancy — my planes are ~18× more work-starved per barrier than
L36_globalpass's 21 KB slabs with 3 syncs, and their fused kernel had
register headroom (72 regs at 216 threads) where mine is capped at 64.

### Chained on-the-fly twiddles: opposite verdicts in the two settings

`fft64_reg_lean`: w1 = W64^t from the table, W64^{t·k1} chained by log-depth
cmuls (w2=w1², w4=w2², ...) — ~6 extra cmuls per line FFT, 1 register pair of
live twiddle instead of 7 (my r2 "next" item 2, finally measured):

* **In kernel_fused it is a clear win**: 786 → 713 µs at B=64. The 64-reg
  union of both phase bodies spilled 168 B/thread with the table (the spill
  writebacks show up as DRAM write pressure); lean cut spills to ~132 B and
  bought 9%. Kept in the fused kernel.
* **In the launch kernels it is a clear LOSS**: B=64 chunked 743 vs 662 µs,
  B=256 2976 vs 2627 µs (B=1 neutral: 18.45 vs 18.56). The solo kernels only
  spill ~40 B; what lean adds there is a serial dependent-multiply chain on
  the critical path of latency-bound warps. Table twiddles stay the default
  (`-DL64_LEANTW` re-enables lean for A/B).
* Accuracy with lean: rel_l2 4.44e-16 vs 4.17e-16 — ~2 ulp on chained
  twiddles, three orders inside the gate, not a factor.

### Tried and rejected, with the number that killed it

* **Ticket prefetch** (grab ticket i+1 during ticket i, publish at an
  existing phase barrier, removing the dedicated top-of-loop sync): B=64
  lead=2 713 → 799, B=256 2765 → 3138. Moving the prefetch after the consumer
  spin (so a producer ticket is never queued behind a stalled consumer) was
  still worse; prefetch before the spin was catastrophic (962 µs / 3807 µs)
  for exactly that queuing reason. The dedicated grab barrier is cheaper than
  any schedule distortion. Do not revisit.
* **Fused with table twiddles** (the natural first form): 786 µs at B=64 —
  the 64-reg two-phase union spills 168 B/thread. Registers, not traffic,
  are why a fused kernel that worked at L=36 does not transplant to L=64.
* **1-block/SM fused grid** (better L1, half the warps): 809 vs 786/713.
* **Lean twiddles in the launch kernels**: see above, −12% at the HBM point.

### What actually shipped this round

1. **Tail-chunk balance in the chunked path**: B=256 at C=3 used to end
   85×3+1 — a 64-block, single-stream 1-volume tail. Now (C,1) tails split
   (C−1,2): 84×3+2+2. Measured 2617.5 µs min / 2623 median, sd 0.11% (r2 form:
   2619–2642 µs); output bit-identical to the unbalanced form, as it must be
   (chunking never changes per-volume arithmetic).
2. The fused kernel + counters stay in the file behind `L64_MODE=1` with the
   occupancy probe and epoch machinery, for whoever attacks the HBM point
   next round.

### Measured (leased SXM4 A100, tryout.sh / on_gpu.sh, final build)

| case | gpu_r2 | gpu_r3 | per volume | cuFFT (r2 board) | ratio |
|---|---|---|---|---|---|
| B=1   | 18.5 µs | **18.56 µs** (min; median boost-noisy) | 18.6 µs | 23.5 µs | 1.27× |
| B=4   | 42.1 µs | **41.9 µs** (sd 0.25%) | 10.5 µs | 58.0 µs | 1.39× |
| B=256 | 2619 µs | **2617.5 µs** (sd 0.11%) | 10.22 µs | 3739.9 µs | 1.43× |

rel_l2 = 4.16–4.17e-16 at B=1,4,5,10,64; bit-identical across runs at every
point; memcheck clean on the chunked (new tail) and fused paths. The B=256
numpy check cannot run this session — the login node's host commit limit is
pinned at 99% by other users' jobs (Committed_AS 130.7/131.9 GB), which also
required `OPENBLAS_NUM_THREADS=1` for every gen_input/check invocation.
B=256 correctness rests on B=10/64 PASS on the identical code path plus
bit-identical output across chunk configs.

### Borrowed, and from whom

* **L36_globalpass gpu_r2**: the whole persistent ticket/epoch design —
  ticket runway, done-counter release/poll idiom, no-reset epoch bases, and
  the one-resident-wave deadlock argument. Their win did not transplant;
  the negative result (and why: plane size per barrier, register ceiling,
  no traffic win to collect) is this round's main contribution to the pool.
* **L45_pfa r2 / L36_sharedtiled r2**: took "the chunk lever is real but only
  with the hint pairing" as settled and did not re-sweep hints.
* The chained-twiddle idea is my own r2 next-list item 2; the split verdict
  (win in a spilling kernel, loss in a latency-bound one) is new information
  for every entry with a register-capped codelet.

### What I would do next

1. **Two specialized persistent kernels instead of one fused union**: yz-only
   and x-only persistent grids of 108 blocks each, co-resident (identical
   footprints), sharing the done-counter handshake, one event between
   executes for the in-place hazard. Each compiles at solo register pressure
   (no union spills), each keeps its own hint policy, and the ticket loop
   sheds the branch. This was L36_globalpass's r2 next-item 1; my fused data
   says spills and barrier grain are exactly what it would fix.
2. **The real ceiling is unchanged since r2**: both launch kernels
   latency-bound at 50% occupancy (64 regs/thread), everything ≤51% SOL,
   composite floor ~1.4–1.6 ms at B=256 vs our 2.62. The radix-4³ codelet
   (16 lanes × 4 complex ≈ 40 regs → 3 blocks/SM if shared allows) is still
   the only untried occupancy lever. Note the shared plane then caps blocks:
   it needs the half-plane (33 KB) x-kernel form to get past 2 blocks/SM.
3. If the monitor's pinned-clock B=256 number moves from 2617, suspect
   nothing: the tail fix is bit-identical and the chunk config is unchanged.

## Round gpu_r4

### Standing at the start

Led all three L=64 points in gpu_r3 (B=1 1.27×, B=4 1.47×, B=256 1.43× over
cuFFT; 2614.4 µs at B=256 on the monitor). This round: two transplants from
L45_pfa's r3 record (one worked, one refuted harder here), one ncu-guided
codelet change that is the round's win, and a negative result that closes the
ticket-granularity question.

### The round's win: shuffle transpose → dead-slot shared exchange

ncu stall metrics on the launch kernels at B=64 (the numbers r2/r3 never
pulled): kernel_yz top stall was **lg_throttle 5.7** cycles/issue (global LSU
queue full), kernel_x **mio_throttle 8.9 + long_scoreboard 10.2** — the MIO
queue was choking, and *shuffles go through the MIO pipe*. Each line FFT
issued 24 SHFL pairs (48 SHFL.32) for the 8×8 j2↔k1 exchange. Replaced it
with a staging pass through the line's OWN 64 shared slots, which are dead at
exactly that moment (just read into registers, rewritten after), so it costs
ZERO extra shared memory — this is r1's untried candidate 1(a), but without
the 8×9-per-warp scratch that made it look unaffordable. Addresses are
XOR-swizzled, slot(a,b) = 8a + (a^b): both the k1-indexed writes (banks k1^t)
and the j-indexed reads (banks t^j) are conflict-free for any odd slot stride
(the strides here are 65, 33, and 1; bank group = (q·ST + base) mod 8 =
(q + base) mod 8). Three `__syncwarp()` around the exchange (sm_80 ITS does
not keep quarter-warps in lockstep; the last one lets the caller rewrite the
slots on return). Slot-privacy audit, kernel by kernel:

* kernel_x: the line IS column g of the tile — reads, exchange, and result
  write-back all stay in the lane-group-private column; no new barrier.
* kernel_yz z-phase: scratch = column g, which is precisely where the
  transposed z-results land anyway (group-private, tile otherwise idle).
* kernel_yz y-phase: scratch = row g (the line's own input slots); the
  cross-row result store stays behind the already-existing __syncthreads.

24 SHFL pairs → 8 STS.128 + 8 LDS.128 per line FFT, no barrier count change.
Measured (same lease, B=64 chunked): kernel_x only 660 → 652, plus kernel_yz
647; solo kernel times 30.4 → 28.7 (yz) and 25.4 → 24.9 µs (x). B=256:
2617–2627 → **2568–2582**. Arithmetic identical, so output is bit-identical
to r3 and rel_l2 is unchanged. Wired into the fused bodies too (<ST> template
on the twiddle-variant wrapper).

### Transplant 1, worked: __noinline__ ticket bodies (from L45_pfa gpu_r3)

r3's fused persistent kernel lost partly to 132–168 B/thread of spills from
the two-body union frame under the 64-reg cap. L45_pfa r3 hit the identical
failure (752 B spills) and fixed it by wrapping each pass body in a
`__device__ __noinline__` function. Ported: spills collapsed to 8 B in the
loop + 24/60 B inside the bodies (solo-kernel level), and table twiddles are
usable inside the fused kernel again (lean no longer needed: with noinline,
table beats the r3 lean form). Fused at B=64: 713 → 700; at B=256:
2765 → 2699 (with the exchange change too). **Still loses to the chunked
launch pipeline everywhere** (647 / 2568), so mode=0 stays the default — but
the fused path is now genuinely spill-free and anyone attacking it next round
starts ~2.5% closer, with the register excuse eliminated: what remains is the
ticket loop's barrier/fence grain on 64 KB planes, full stop.

### Transplant 2, refuted harder here: coarser tickets (L45_pfa's warning)

Templated the fused kernel on G = planes per ticket (1/2/4/8), amortizing the
grab sync + threadfence + release sync + done-atomic over G planes, one
inter-plane __syncthreads instead (also dropped the trailing sh-protection
sync — the top-of-loop grab sync already orders it). Sweep at B=64, lead=2:
**G=1 709, G=2 769, G=4 913, G=8 1000 µs.** Monotonically catastrophic.
L45_pfa's 45×45-tile result (2349 vs 2263, ~4%) said coarse tickets lose to
imbalance; at my plane sizes it is 8–41%. The grab overhead was never the
fused deficit — convoying is: coarser tickets make the x-pass poll wait on
4× longer yz tickets and bunch blocks onto the same volume. Template and
L64_TPG env kept so nobody re-implements this to re-learn it.

### Operation count

FP64 unchanged (nominal 90 flop/point; SASS 132 FP64 per 8-lane line FFT).
Per line FFT: −48 SHFL, +8 STS.128 +8 LDS.128 +3 __syncwarp. Global traffic
and barrier counts unchanged everywhere.

### Measured (leased SXM4 A100, tryout.sh, final build, default config)

| case | gpu_r3 | gpu_r4 | per volume | cuFFT same lease | ratio |
|---|---|---|---|---|---|
| B=1   | 18.56 µs | **18.2–19.1 µs** (min; boost-noisy as always) | ~18.7 µs | 23.5 µs | ~1.24× |
| B=4   | 41.9 µs | **41.0–41.7 µs** | 10.3 µs | 58.1 µs | 1.40× |
| B=64  | 662 µs | **646.9–647.5 µs** (sd 0.03%) | 10.11 µs | 915.3 µs | 1.41× |
| B=256 | 2617.5 µs | **2568.2 µs** (sd 0.05%) | **10.03 µs** | 3746.6 µs | 1.46× |

rel_l2 = 4.16–4.17e-16 at B=1,4,5,9,10,64,256 — **including a full numpy PASS
at B=256 this round** (login-node RAM allowed it for the first time);
bit-identical across runs at every point; compute-sanitizer memcheck clean on
chunked B=10 (rebalanced 3+3+2+2 tail), fused G=1, and fused G=4. Chunk
re-scan after the codelet change: C=2 2898, **C=3 2582**, C=4 2707 — C=3
survives.

### Tried and rejected, with the number that killed it

* **Ticket coarsening G>1**: see above. G=2 already loses 8%.
* **Fused persistent as default after the noinline fix**: 700 vs 647 (B=64),
  2699 vs 2568 (B=256). Two rounds of fixes (lean r3, noinline+exchange r4)
  moved it 713→700; the launch pipeline moved 662→647 in the same time. The
  structure itself is the deficit at this plane size, not the compilation.

### Borrowed, and from whom

* **L45_pfa gpu_r3**: the `__device__ __noinline__` per-body register
  isolation for persistent/fused kernels (worked exactly as their record
  promised — check `-Xptxas -v` before blaming a structure), and their
  coarse-ticket imbalance warning (confirmed, much stronger here).
* The dead-slot XOR-swizzled exchange is this round's own contribution
  (r1 candidate 1(a) made free by reusing the line's just-read slots; the
  swizzle is the standard xor-permutation trick). Transferable note for the
  other register-codelet entries (L8_warpradix8, L6_*): **SHFL competes with
  LDS/STS for the MIO pipe** — if ncu shows mio_throttle and there is a dead
  shared region, the exchange is better staged through it; if the kernel has
  no shared traffic at all, shuffles remain right.

### What I would do next

1. **The occupancy wall is now the whole story**: both launch kernels at
   40% achieved occupancy (theoretical 50%: 64 regs × 512 threads × 2 blocks
   = the full RF), No-Eligible 78–83%, every pipe ≤35% SOL, DRAM 28–33%.
   The only route to more warps is a smaller-register codelet AND a smaller
   tile: radix-4³ (16 lanes × 4 complex ≈ 40 regs) + half-plane yz tiles
   (33.25 KB, z-lines computed twice — compute is at 23%, affordable) would
   allow 3 × 512 threads/SM = 48 warps. It is a full-round rewrite; the
   estimated composite floor remains ~1.4–1.6 ms at B=256 vs our 2.57.
2. Two specialized co-resident persistent kernels (yz-grid + x-grid sharing
   the counters): still untried, but the fused data now says the barrier
   grain, not registers, is the fused deficit — a static SM partition also
   forfeits the work-stealing balance, so I rate it below idea 1.
3. If the monitor's B=256 number lands near 2568 the chunk config is doing
   its job; if it reads >2650, re-check C=3 on pinned clocks first.
