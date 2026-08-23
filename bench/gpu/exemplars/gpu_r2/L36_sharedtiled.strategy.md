# L36_sharedtiled — strategy record

## Round gpu_r1 (first implementation)

### What was built

Two-pass plane-per-block shared-memory kernel, following literature §9.6 structure 1,
plus the §9.6-structure-2 L2 chunking, which turned out to be worth 1.5× on the primary
score point.

* **Kernel 1** (`k36_zy`): one (y,z) plane per block, grid `B·36`. The plane is a
  contiguous 1296-element (20.25 KiB) run of the volume, so load and store are perfectly
  coalesced 16-byte accesses. Staged in shared memory with the row stride padded
  **36 → 37** complex doubles (gcd(36,8)=4 is a 4-way conflict unpadded; odd stride in
  complex-double units is conflict-free for 16-byte accesses because conflicts are
  evaluated per quarter-warp). Transforms the z-lines, then the y-lines, writes back.
* **Kernel 2** (`k36_x`): one (x,z) slab per block at fixed (v,y), rows read coalesced
  along z (576 B per x-row), same line engine along x, in place on `out`.
* **Line engine**: 36 = 6·6 Cooley–Tukey, negative exponent. 6 threads per line ×
  36 lines = 216 threads per block (1296 = 6·216, so the copy loops have no remainder).
  Stage 1: thread b does a DFT-6 over a of u[6a+b], writes to slot 6b+q. Stage 2: thread
  q multiplies by W36^{bq} (36-entry `__constant__` table) and does a DFT-6 over b,
  reading and writing only its own slots {6b+q} — so a full 36-point pass costs two
  `__syncthreads()`. DFT-6 is two DFT-3s plus three 2-point combines (~60 flops).
  Thread mapping tid → (line = tid%36, sixth = tid/36) makes all four shared access
  patterns conflict-free per quarter-warp (checked by hand: addresses mod 8 distinct).
* **Operation count**: 3·36² = 3888 lines per volume, ~900 flops per line ≈ 3.5 Mflop
  per volume — irrelevant, as the literature said: the kernel is bandwidth-bound
  (ncu: 84–86% DRAM speed-of-light, 43–66% occupancy, compute SOL 24–44%).
* **Cache hints**: `__ldcs` on the `in` read (never reused) and `__stcs` on the final
  `out` write (never re-read) — worth ~2.5% at B_HBM by keeping streams from evicting
  the chunk intermediate.
* **L2 chunking** (the big one): for `B > 32` the batch is processed in chunks of C
  volumes round-robined over ns non-blocking streams; kernel 2 of a chunk reads kernel
  1's intermediate out of L2 (measured with ncu: k36_x DRAM reads drop from 8.95 MB to
  0.13–1.3 MB per 12-volume chunk) and re-dirties the same lines, so the intermediate
  mostly never touches HBM. `out` itself is the intermediate — see failures below for
  why a separate scratch buffer is *worse*. (C, ns) is autotuned in `create()` on
  scratch buffers; the sweet spot on the node was **C=7–8, ns=4** (7·36=252…288 blocks
  per kernel ≈ within a wave at 4 blocks/SM, 4 streams overlapping; live footprint
  ~4·2·C·0.75 MB ≲ 40 MB L2). Chunk 12/2-streams is equivalent within noise; 24/2 or
  12/4 overflow L2 and collapse to unchunked speed.
* **Small-B launch path**: at B=1 the cost is CPU launch submission. `create()` measures
  three options — two plain launches, a fused cooperative kernel with `grid.sync()`, and
  a captured CUDA graph replayed per execute — and picks by time. The graph wins
  (12.8 vs 14.5 plain vs 15.9 fused µs in the create-time probe; 10.1 µs measured by the
  driver). The graph is keyed on the (in,out) pointers and recaptured if they change,
  so repeatability and the driver's poke test are safe.

### Measured on the leased SXM4 A100 (tryout.sh, reserved node)

| point | this entry | per volume | cuFFT | speedup |
|---|---|---|---|---|
| B=1 | **10.08 µs** | 10.08 µs | 13.3 µs | 1.32× |
| B_L2=22 | **40.98 µs** | 1.86 µs | 51.2 µs | 1.25× |
| B_HBM=1438 | **2116.6 µs** | **1.472 µs** | 3390 µs (2.36 µs/vol) | **1.60×** |

rel_l2 = 4.8e-16 at every checked batch (1, 4, 22, 64, 100); repeatable (bit-identical
across runs); compute-sanitizer memcheck clean on the chunked path including the
non-divisible-tail case (B=100, chunk 10) and the scratch variant.

Roofline context: ncu reports the part's DRAM peak as 1.55 TB/s and both kernels at
84–86% of it. Unchunked two-pass = 4·746 KB/volume → measured 2.22 µs/vol (1.35 TB/s
actual). Chunked ideal (intermediate entirely in L2) would be 1.49 MB/volume of HBM ≈
1.15 µs/vol; we sit at 1.472, i.e. the effective HBM traffic is ~1.9 MB/volume — the
residual ~28% is premature intermediate evictions under streaming pressure.

### Tried and did NOT work, with the number that killed it

1. **Separate per-stream scratch buffer as the intermediate** (fixed addresses reused
   every chunk, hoping they'd stay L2-hot): 2339 µs vs 2118 µs for out-as-intermediate
   at (6,4). Reason in hindsight: out-as-intermediate uses ONE L2 line per element
   (k1 write → k2 read → k2 rewrite of the same line), scratch needs that line PLUS a
   cold line for the final `out` write — strictly more footprint.
2. **`L2::evict_last` cache-hint store (createpolicy + st.global.L2::cache_hint PTX) on
   the chunk intermediate**: 2118 → 2623 µs at (6,4), 24% worse, high variance. Without
   a persisting-L2 carve the hint just distorts replacement. Left a comment in the code
   so nobody re-adds it.
3. **Fused cooperative kernel (one launch + grid.sync) at B=1**: 15.9 µs vs 14.5 plain —
   grid.sync + cooperative-launch overhead exceeds the saved launch, exactly as the
   literature's PERKS note warned. Kept in the code because the autotuner rejects it by
   measurement; it may win on a different driver.
4. **4 streams with chunk 12** (footprint 4·2·12·0.75 = 72 MB > 40 MB L2): 3169 µs —
   as slow as no chunking. The chunk×streams product must stay under L2.
5. **Graph replay at B_HBM**: neutral (2134 plain vs 2142 graph µs) — CPU submission of
   ~410 launches is not the bottleneck there. Autotuned per batch, so it costs nothing.

### Borrowed from

* Literature `09-gpu-small-batched-a100.md` §9.6: the whole opening structure (two-pass
  plane-per-block, 36→37 padding, L2 chunking across the batch), and §6 for the
  conflict-free odd-stride reasoning. First round, so no other entry records existed to
  borrow from; the CPU L36 records (`../geom/strategies/L36_*.md`) confirmed "two-sweep
  structure at the traffic floor, don't chase arithmetic", which is what this is.

### What I would do next

1. **Recover the last ~28% of HBM traffic**: a persisting-L2 carve
   (`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, …)` + per-stream
   `accessPolicyWindow` sliding over the chunk intermediate) is the mechanism the
   evict_last experiment was missing. Ideal is ~1.15 µs/vol.
2. **Kernel-2 row alignment**: x-rows are 576 B, so odd-y slabs start at a 64 B offset
   and read 5 sectors per 4.5-sector row (~11% amplification on one of four
   pass-halves). Processing two adjacent y-slabs per block (1152 B contiguous rows,
   2 planes of shared = 42.6 KB, still under 48 KB) would remove it.
3. **128 B/thread vectorized copies** (double4 pairs) in the load/store loops if ncu
   shows the copy phases below DRAM SOL.
4. B=1: the two kernels are only ~36 blocks each; a graph with the two kernels is
   already down to 10 µs, most of which is one graph launch + two kernel dispatches.
   Little left without a persistent kernel.

## Round gpu_r2

### Standing at the start

Led at B=1 (10.07 vs globalpass 10.49) and B_HBM=1438 (1.495 vs 1.534 µs/vol), but
LOST B_L2=22 to L36_globalpass (1.853 vs 1.745 µs/vol). This round was about that point.

### What changed

1. **Batch-split stream overlap at small B — the round's win.** The old small-batch
   path ran the two kernels back-to-back on one stream: k1's 792 blocks (1.83 waves at
   4 blocks/SM), a hard k1→k2 dependency, then k2's 792 blocks — two ragged tails and a
   dead barrier in the middle. New: split the batch into one chunk per stream
   (chunk=ceil(B/ns), ns∈{2,4,8}, MAXSTREAMS raised 4→8), each stream running its own
   k1;k2 pair, so pass 2 of one slice fills the tail waves of pass 1 of another.
   Measured at B=22: **40.94 → 29.2 µs (−29%)**, now 1.75× cuFFT (51.1) and well past
   globalpass (38.4). The critical shape fact, from a manual `L36_CHUNK` sweep: only
   ONE-chunk-per-stream shapes win ({11,2} 30.5, {6,4} 29.7); any shape putting ≥2
   chunks on a stream is *worse than unchunked* ({8,2} 43.0, {6,2} 44.0, {4,4} 42.6 vs
   40.9). This is pure concurrency, not L2 chunking — do not "tune the chunk size" here,
   tune the stream count. Autotuned in create() over ns∈{2,4,8}×{plain,stream hints},
   then launch mode (plain/fused/graph) on the best shape. B=4 also improved: 13.3 µs
   (split 2/2 + graph) vs cuFFT 17.9.
2. **Cache-hint policy became a 3-way template** (POL_PLAIN / POL_STREAM / POL_CHUNKED)
   and the choice at small B is measured, not assumed. The hypothesis that killing the
   streaming hints would unlock cross-execute L2 residency at B=22 (in+out = 32 MiB vs
   40 MiB L2) was WRONG: plain vs stream measured 50.9 vs 50.5 µs serialized and the
   tuner still picks stream in the split shapes — 32 MiB against a 40 MiB L2 is too
   marginal for residency to survive the churn of the intermediate pass. Keep the
   template (PLAIN does win at B=4-class sizes), but don't expect miracles from it.
3. **`__ldlu` (last-use) instead of `__ldcs` on the chunked intermediate read** —
   borrowed from L64_radix8's hint pairing. Neutral at the driver (2113 vs 2112 µs at
   B=1438; create-probe suggested ~2%). Kept: it is the semantically right hint for a
   read whose line is dead after the load.

### Operation count

Unchanged from gpu_r1: 3·36² lines/volume ≈ 3.5 Mflop/volume, bandwidth-bound.

### Measured on the leased SXM4 node (tryout.sh)

| point | gpu_r1 | gpu_r2 | per volume | cuFFT | speedup |
|---|---|---|---|---|---|
| B=1 | 10.08 µs | **9.99 µs** | 9.99 µs | 12.94 µs | 1.30× |
| B_L2=22 | 40.98 µs | **29.2 µs** | 1.33 µs | 51.1 µs | **1.75×** |
| B_HBM=1438 | 2116.6 µs | **2111.5 µs** | 1.469 µs | 3400 µs | 1.61× |

rel_l2 = 4.8e-16 at B=1, 4, 22, 30, 64, 1438; bit-identical across runs at every point;
compute-sanitizer memcheck clean on the new split path (B=22, incl. the non-divisible
8-stream tail 3×7+1).

### Tried and did NOT work, with the number that killed it

1. **PLAIN hints to hold B=22 resident in L2 across executes**: 50.9 vs 50.5 µs — see
   above. The 24% was in concurrency, not in cache policy.
2. **Wider-stream chunk shapes at B_HBM** (manual sweep with MAXSTREAMS=8): {7,4} 2105,
   {3,8} 2112, {4,8} 2105, {4,6} 2122, {6,6} 2180, {5,8} 2255 µs. The ~2105 µs plateau
   is robust to shape — the residual ~30% over the 2-HBM-pass floor is L2 hit-rate
   under streaming pressure, not launch geometry. globalpass already measured the
   persisting-L2 accessPolicyWindow fix losing (705.8 vs 665.6 at their scale), so that
   door is closed too; a real B_HBM gain needs a structurally different intermediate
   (or kernel fusion), not another hint.
3. **Graph replay of the split shape at B=22**: 35.0 vs 29.6 µs plain — multi-stream
   graph replay serializes enough to lose; the tuner rejects it by measurement.

### Borrowed from

* **L45_pfa (gpu_r1)**: the regime-split template idea (measure hint/shape policy per
  batch regime in create(), never assume one policy fits all points).
* **L64_radix8 (gpu_r1)**: the `__ldcs`/`__ldlu`/`__stcs` hint-pairing vocabulary;
  `__ldlu` on the dead intermediate read.
* **L36_globalpass (gpu_r1)**: the pipelined-streams direction at the L2 point (their
  K1/K2 event pipeline is why they held B_L2 in r1); this entry's split is the simpler
  form of the same overlap and measured past it. Their accessPolicyWindow failure was
  taken at face value and not rediscovered.

### What I would do next

1. **B_HBM is the open front (1.469 µs/vol vs ~1.05 ideal)**: the only untried lever
   with real headroom is fusing k2(chunk c) and k1(chunk c+1) into one launch (or a
   persistent producer/consumer kernel) so the intermediate never faces a kernel
   boundary. Hint- and shape-space is exhausted — three independent sweeps now agree
   on the ~2105 µs plateau.
2. B=1 (9.99 µs) is one graph launch + two 36-block kernels; only a persistent/fused
   single kernel could cut it further, and the cooperative version measured 16.0 µs —
   the launch mechanism, not the sync, would have to change.
3. If the monitor ever scores intermediate batches (64–512), the current autotuner
   already handles them (B=64: 1.48 µs/vol, 1.70× cuFFT).
