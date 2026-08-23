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

## Round gpu_r3

### Standing at the start

Held B_L2=22 (1.383 vs globalpass 1.673 µs/vol) but lost B=1 (10.00 vs 9.03) and — the
primary point — B_HBM=1438 (1.446 vs 1.353) to L36_globalpass, whose r2 built the
persistent producer/consumer kernel that my own r2 postmortem had named as the only
remaining lever. This round is explicitly cumulative: I took their kernel.

### What changed

1. **Persistent ticket producer/consumer kernel — ported from L36_globalpass (gpu_r2),
   stated plainly.** One launch per execute, grid = one resident wave (432 blocks at
   4/SM); blocks pull tickets (one ticket = one (b,x) K1 plane or (b,y) K2 slab) off a
   global atomic; K1 runs `lead` volumes ahead; K2 tickets spin on per-volume done
   counters (release: __threadfence + atomicAdd, acquire: poll + __nanosleep + fence);
   epoch bases (tbase/dbase in mod-2^32 arithmetic) make the counters valid across
   executes with no reset. Their deadlock argument (every K1(v) ticket precedes every
   K2(v) ticket in grab order, grid ≤ one wave) re-verified before porting. I kept my
   POL template vocabulary: CHUNKED = __ldcs input / plain intermediate store / __ldlu
   intermediate read / __stcs final store. Never graph-captured (args mutate per call).
2. **Shared-memory carveout 100 → 50 — the hidden 7.4%.** Same-structure A/B at B=432:
   carveout 100 gave 634.6 µs and the tuner "preferred" no cache hints; carveout 50
   gave 587.5 µs with hints correctly winning. My r1/r2 carveout=100 starved L1 to
   ~28 KB, which the all-streaming-hints kernels never noticed, but the ticket kernel's
   L2-hit intermediate reads do. Spotted only because globalpass's create() sets 50
   with a comment; this is why reading rivals' code beats reading their conclusions.
3. **Autotuner samples now exceed the 20 ms boost cliff** (lesson taken verbatim from
   globalpass's r2 record item 3): my r2 big-B tuner sampled 3 executes ≈ 6 ms, inside
   the clock-ramp artefact zone. Now reps = 22 ms / est_per_execute at every batch.
4. **Line-engine micro-passes** (mostly from globalpass's code): fma-form cmul, skip
   the identity twiddle W36^0 (stage-2 b=0), exact-6-trip unrolled copy loops instead
   of guarded strided loops, and the element stride of the line pass made a template
   constant (1 or 37). Worth ~0.9 µs at B=1 (10.0 → 9.13) where kernel latency is the
   whole story; also part of the B=22 improvement.
5. **cp.async staging as a tuner option** (16 B cp.async.cg global→shared, no register
   round-trip): measured NEUTRAL at B=432 within lease noise (581.973 vs 589.464 µs at
   lead 12, but 597.184 vs 578.349 at lead 13 — contradictory, i.e. noise). Kept as an
   autotune candidate since it costs nothing and occasionally measures best; not relied
   on. The L2 evict-first hint is lost under cp.async (plain .cg policy), which may be
   why it does not separate.
6. **Small-B candidate list extended** with ticket configs (leads min(4,B)/12/B, both
   hint policies) and the launch-mode probe (plain/coop/graph) restructured to run on
   the best non-ticket shape, then compare against the best ticket. At B=22 the split
   3-per-stream/8-stream shape still wins (ticket loses to it); at B=1 pair+graph wins.

### Operation count

Arithmetic essentially unchanged (3·36² lines/volume; one identity cmul per thread per
stage-2 removed, ~3.4 Mflop/volume) and still irrelevant: ncu on the ticket kernel at
B=432 shows DRAM bytes ≈ 652 MB against a 644 MB compulsory floor (2 passes), DRAM
throughput 59% of peak, occupancy 43.5% (4 blocks/SM, register+shared limited), 31
warp cycles/issued instruction — the kernel is latency/warp-supply-bound with its
traffic already minimal, exactly globalpass's r2 diagnosis.

### Measured on the leased SXM4 node (tryout.sh; lease-to-lease spread ~3%)

| point | gpu_r2 | gpu_r3 | per volume | cuFFT same lease | speedup |
|---|---|---|---|---|---|
| B=1 | 9.99 µs | **9.125 µs** | 9.125 µs | 13.19 µs | 1.45× |
| B=4 | 13.3 µs | **12.26 µs** | 3.06 µs | — | — |
| B_L2=22 | 29.2 µs | **27.59 µs** | 1.254 µs | 50.6 µs | 1.83× |
| B=432 | — | **573.6 µs** | 1.328 µs | 998 µs | 1.74× |
| B_HBM=1438 | 2111.5 µs | **1932.1 µs** | 1.344 µs | ~3400 µs (r2 lb) | ~1.76× |

Configs picked: B=1 pair+graph PLAIN; B=4 split 2/2 PLAIN; B=22 split 3/8 STREAM;
B=432/1438 ticket lead=12 CHUNKED (lead 12–14 is a plateau; 9 → 673 µs, 16 → 865 µs
at B=432, so the tuner's {9,12,14} brackets the cliff).

rel_l2 = 4.83–4.85e-16 at B = 1, 4, 13, 22, 64, 100, 432; bit-identical re-runs at
every point; compute-sanitizer memcheck clean on the ticket path (B=13 PLAIN+cp.async,
B=100 CHUNKED with and without cp.async — exercises decode prologue/alternation/tail).
B=1438 timed on a tiled input (login-node numpy could not allocate 1 GiB this round —
also: prefix everything with OPENBLAS_NUM_THREADS=1 on the login node or numpy dies of
per-thread buffer allocation; correctness at 1438 rides on B=432/100 passing the same
ticket code path).

### Tried and did NOT work, with the number that killed it

1. **Carveout 100 with the ticket kernel**: 634.6 vs 587.5 µs at B=432 (see above).
   The r1/r2 setting was wrong all along for anything that reads through L1.
2. **MINB=3 (launch_bounds 216,3: 80 regs, fewer spills, 3 blocks/SM)**: 607.9 vs
   585.2 µs same-lease at B=432. Occupancy beats spill removal; the 72-reg cap with
   32 B of spills is the right trade. Corroborates globalpass's r1 5-blocks failure
   from the other side: 4 blocks/SM is a local optimum from both directions.
3. **cp.async as a guaranteed win**: neutral within noise (numbers in item 5 above).
   Kept only as a measured candidate.
4. **Ticket kernel at the small points**: loses to split-streams at B=22 and to
   pair+graph at B=1 in every tuner run (globalpass's record said the same of their
   fused path at B=1; not rediscovered — just confirmed by the tuner's own numbers).

### Borrowed, and from whom

* **L36_globalpass (gpu_r2)**: the persistent ticket kernel wholesale (decode, epoch
  counters, lead concept, deadlock argument), lead=12 as the starting sweep center,
  the carveout=50 setting read out of their create(), the ≥20 ms tuner-sample rule
  from their record, and the fma cmul. This round's B_HBM gain is their design plus
  my hint template; the honest statement is that we converged on their structure.
* **L45_pfa / L64_radix8 (r1/r2 records, re-read)**: confirmed the regime-split and
  hint-pairing vocabulary already in this entry; nothing new taken this round.

### What I would do next

1. **B_HBM has ~0.4 µs/vol of latency headroom left** (1.344 vs ~0.96 floor at the
   measured 43.5% occupancy). The untried structural idea remains globalpass's "two
   specialized persistent kernels" (K1-only and K2-only co-resident on two streams,
   each compiling leaner than the 72-reg union, grids ratio-balanced ~1.5:1 for the
   work asymmetry) — neither entry has built it; it is the only idea on the table
   that could reach 5 blocks/SM without spills.
2. B=1 at 9.13 µs is one graph launch + two 36-block kernels; the remaining cost is
   dominated by launch/dispatch latency and one HBM round trip. A single-volume
   specialized kernel was tried by globalpass and lost; I see nothing cheap left.
3. If the monitor adds intermediate batch points, the tuner now covers them
   (B=64: 96.7 µs = 1.51 µs/vol, ticket lead=14; B=100: 143.9 µs = 1.44 µs/vol).

## Round gpu_r4

### Standing at the start

Led all three points, but by hair-thin margins: B=1 9.062 vs globalpass 9.129, B_L2=22
28.738 vs 28.711 (a statistical tie, formally lost), B_HBM=1438 1931.5 vs 1942.1. The
round's context file makes the panel cumulative; the obvious loot was in L45_pfa's r3
record, which had already measured the two ideas my own r3 postmortem was circling.

### What changed

1. **Direct global-output/input bodies — the round's win, borrowed from L45_pfa (gpu_r3),
   stated plainly.** Their r3 measured K1-direct-out −4.8% and K2-direct a further win
   *inside the persistent-kernel regime* (where their r2 had measured the same forms
   LOSING at launch-grain chunking). Ported to this entry's bodies:
   * **DIR1 — K1's y-pass stage-2 streams straight to global**: thread (l=z, j) writes
     X[6c+j] of y-line l to out[(6c+j)·36+l]; fixed (c,j) with consecutive l per warp, so
     every store is a coalesced 576 B row. Drops the y-pass's stage-2 shared write, the
     trailing barrier, and the whole 6-trip shared→global copy loop (2×1296 shared
     accesses + 1 sync per plane). Stores stay plain — the intermediate must land in L2.
   * **DIR2 — K2 reads global straight into registers and stores straight back**: stage 1
     reads u[6a+j] at src[(6a+j)·1296+z] (fixed (a,j), consecutive z → coalesced), dft6,
     and only the inter-stage exchange goes through shared (slot z+(6j+q)·37, conflict-free);
     stage 2 twiddles, dft6, stores X[6c+j] direct (__stcs when streaming). The staging
     copy loops and two of K2's barriers disappear; K2 falls from 5 to 1 internal syncs
     and from ~5200 to ~2600 shared accesses per slab. In-place safe: all global reads
     precede the one barrier, all writes follow it.
   Both forms are compile-time A/B-able (-DL36_NODIR1/-DL36_NODIR2). Attribution at
   B=432 same lease: both staged 577.1, DIR1 only 559.5, DIR2 only 549.6, both 524.5 µs
   — the two gains are roughly additive (−9.1% total). Register pressure collapsed as a
   side effect: k36_x 60→40 regs, k36_zy →60, pair kernels spill-free; the ticket union
   stays 72 regs / 20 B spills.
2. **Ticket lead re-swept under the new bodies** (the K1:K2 cost ratio changed): lead
   9→544, 10→539, 11→535.5, 12→536.3, 13→539.6, 14→541, 16→561 µs at B=432 (one lease
   set). Still a lead 11–12 plateau; the tuner's {9,12,14} bracket stands, no change.
3. Plumbing for the failed experiments below is left in the file, default-off:
   MINB-dependent ticket carveout, -DL36_TNI noinline wrappers, -DL36_PAIRCO.

### Operation count

Arithmetic unchanged (3·36² lines/volume ≈ 3.4 Mflop/volume). Global traffic unchanged
at the two-pass compulsory 4 accesses/point through L2, ~2 passes HBM. What changed is
shared traffic (K1 −2592, K2 −2592 accesses per ticket) and barriers (K1 7→6, K2 5→1).
ncu on the final ticket kernel (B=432, lead 12): **DRAM 68.5% (r3: 59%), L2 91.6%, SM
49.7%, occupancy 43.6%** (4 blocks/SM, unchanged), 31.1 warp cycles/issued instruction.
The L2 pipe is now the nearest roof — the same terminal state L45_pfa's r3 profile
reached. With all 4 L2 accesses/point compulsory in a two-pass structure, the headroom
left at B_HBM is small and mostly not reachable by latency tricks.

### Measured on the leased SXM4 node (tryout.sh; same-day baselines re-measured)

| point | r3 code (same day) | gpu_r4 | per volume | cuFFT same lease | speedup |
|---|---|---|---|---|---|
| B=1 | 9.142 µs | **8.066 µs** | 8.066 µs | 13.50 µs | 1.67× |
| B=4 | 12.26 µs (r3 rec) | **10.66 µs** | 2.67 µs | — | — |
| B_L2=22 | 29.056 µs | **28.458 µs** | 1.294 µs | 51.02 µs | 1.79× |
| B=64 | 96.7 µs (r3 rec) | **88.41 µs** | 1.381 µs | — | — |
| B=100 | 143.9 µs (r3 rec) | **131.9 µs** | 1.319 µs | — | — |
| B=432 | 577.1 µs | **524.5 µs** | 1.214 µs | 1000.6 µs | 1.91× |
| B_HBM=1438 | 1932 µs (r3 rec) | **1713.6 µs** | **1.192 µs** | 3398 µs (r3 lb) | ~1.98× |

Configs picked by the tuner: B=1 pair+graph STREAM; B=4 split 2/2 STREAM; B=22 split
6/4 STREAM; B≥64 ticket lead=12 CHUNKED cpa=0. rel_l2 = 4.83–4.85e-16 at B = 1, 4, 13,
22, 64, 100, 432, 1438 (the B=1438 numpy check ran fine on the login node this round);
bit-identical re-runs everywhere; compute-sanitizer memcheck clean on the ticket path
with both direct forms (B=100 CHUNKED, cpa=0 and cpa=1, and B=13 PLAIN lead=4 —
prologue/alternation/tail decode all exercised).

### Tried and did NOT work, with the number that killed it

1. **MINB=5 (5 blocks/SM ticket, 60-reg cap, carveout raised to 81→132 KB)**: 96 B of
   spills, tuner abandoned the ticket entirely (chunked path won at 607 µs vs 524.5).
2. **MINB=5 + L45_pfa's __noinline__ body isolation**: registers fixed exactly as their
   r3 record promised (72→56 regs, spills 96 B→~0) — and it STILL lost: 578.0 vs 524.5
   µs. At 5 blocks the carveout must go to 132 KB, leaving 32 KB L1 for a kernel whose
   intermediate reads live in L1/L2, plus ABI call overhead. With r3's MINB=3 result
   (607.9) and both rivals' r1/r2 failures, 4 blocks/SM is now confirmed the optimum
   from *three* directions; consider this door closed at L=36.
3. **__noinline__ bodies at MINB=4** (kills the 20 B spills): 523.1 vs 524.5 µs —
   neutral within noise. Kept available (-DL36_TNI), default off.
4. **Pair-kernel carveout 100** (direct k36_x at 40 regs could hold 7 blocks/SM):
   create()-probe said 29.2 µs at B=22 but the driver measured **43.7 vs 28.8** — a
   probe/driver mismatch worth remembering: the probe warms the same buffers every
   rep, the driver's sampled loop does not, and 164 KB shared leaves 0 KB L1. Reverted.
5. **lead ≠ 11–12** at B=432: see sweep in item 2; 16 is already +5%.

### Borrowed, and from whom

* **L45_pfa (gpu_r3)**: both direct forms (their pd1/pd2, including the observation
  that the direct/staged verdict flips between launch-grain and persistent regimes —
  taken at face value, and it reproduced here: −9.1% inside the ticket kernel where my
  own r1 had measured direct-ish stores losing at launch grain), and the __noinline__
  register-frame isolation trick (worked exactly as documented, 96 B spills → 0; the
  surrounding 5-blocks idea still lost — see above).
* **L36_globalpass**: nothing new this round; their r2 ticket design remains the
  chassis everything else hangs off.

### What I would do next

1. **B_HBM is near its structural limit for a two-pass kernel**: L2 at 91.6% SOL with
   all four L2 accesses/point compulsory. The only lever that changes the constraint
   is fewer L2 accesses, i.e. a genuinely fused 3-axis pass, which needs the whole
   746 KB volume visible to one block — impossible in 163 KB shared. A radix-6 z-step
   that keeps 6 partial planes in shared while streaming x-lines is a 3-pass-worth of
   traffic in disguise; I see no honest way below 4 accesses/point at L=36. Expect
   diminishing returns; defend the lead by porting whatever the rivals find.
2. B=1 at 8.07 µs is one graph launch + two 36-block kernels, both now barrier-light.
   The remaining cost is launch/dispatch plus one L2 round trip; nothing cheap left.
3. B_L2=22 moved least (−2%): the split path is concurrency-bound, not traffic-bound,
   and the direct forms only shave kernel-internal work. If the tie must be broken,
   the untested idea is a persistent ticket kernel with POL_STREAM hints sized to keep
   the 32 MiB working set resident across executes (the tuner's ticket candidates only
   try CHUNKED/PLAIN there today).
