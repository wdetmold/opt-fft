# L17_raderfused — strategy record

Geometry: **L = 17**, cube 17³ = 4913 complex doubles (78,608 B) per volume, forward,
unnormalised, out-of-place, batched, one A100.
Implementation: `impl/L17_raderfused.cu`. `fft3d_gpu_name()` → `L17_raderfused`.

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique (final form)

Two paths, selected in `fft3d_gpu_create()` by batch size:

**Fused path (batch ≥ 12, the scored L2 and HBM points).** One volume per block,
resident in 78,608 B of dynamic shared memory (`cudaFuncAttributeMaxDynamicSharedMemorySize`
opt-in; max shared carveout requested so **two blocks co-reside per SM**, 157 KB of 164).
320 threads. Structure per volume:

1. staging load, coalesced, **fully unrolled to 15 independent rounds** so one block keeps
   ~75 KB of loads in flight rather than chaining load→store at one latency each;
2. z pass: 289 lines, one per thread, base `tid*17`, stride 1, in place in shared;
3. y pass: base `(tid/17)*289 + tid%17`, stride 17, in place;
4. x pass: base `tid`, stride 289, **results written straight to global** — for each
   output k the warp writes 32 consecutive double2, perfectly coalesced, so there is no
   separate store loop and no fourth barrier.

One global read + one global write per volume — the corpus's central design fact for
L ≤ 17 (lit. §09 §1.5). All shared strides are odd multiples of 16 B, so every pass is
bank-conflict-free without padding (the 8-thread 128-bit phase covers all 32 banks exactly
once because 17 ≡ 1 mod 8 in double2 units). 96 registers, zero spills,
`__launch_bounds__(320, 2)`.

**Split path (batch < 12).** The fused kernel gives `batch` blocks, so B=1 uses 1 SM of
108 and is pure single-block latency (~15 µs). Instead: `fft17_planes` does the z and y
axes of one 17×17 x-plane per block (`B*17` blocks, **thread-per-output**: thread
(k = tid/17, line = tid%17) computes one X_k from 17 broadcast shared reads — ~30
registers, 40+ warps/SM), then `fft17_xlines` does the x axis thread-per-line, in place
on `out` (no scratch buffer; the two kernels run back to back on the same stream). Twice
the global traffic of the fused path, but at these batches everything is L2-resident and
latency is what is scored.

**Per-line module (`line17w`), both paths.** Not Rader, despite the entry name — see the
negative-results list. It is the CPU panel's winning algebra, ported: the conjugate-pair
fold to all-real coefficients (**adopted from `../geom/strategies/L17_matrixsimd.md`
round 1**), plus the primitive-root reindexing that turns the folded 8×8 cosine matrix
into a cyclic-8 correlation split into cyclic-4 ⊕ negacyclic-4 halves with pre-halved
kernels, the sine half a sign-decorated negacyclic-8 handled by a 15-entry table
(**adopted from `../geom/strategies/L17_winograd.md` via matrixsimd round 2** — their
derivation and index tables `f = {1,3,8,7,4,5,2,6}`, `sig = {+,+,−,−,−,+,−,−}` verbatim;
the sig factors are folded into operand order and output-pair assignment, both
compile-time). Coefficients (4+7+15 doubles) in `__constant__`, computed in `long double`
at plan time; all threads read the same coefficient at the same unrolled step, so every
access is a broadcast.

### Operation count

496 real flops per 17-point line (fold 32 + A/B reduction 16 + X0 16 + cyclic-4 and
negacyclic-4 cosine 64 FMA-flops×2 + P-combine 16 + negacyclic-8 sine 256 + output
combine 32), versus 608 for the plain folded dense form and 2312 naive. Per volume:
867 lines × 496 ≈ **430 kflop = 87.5 flop/point**, arithmetic intensity 2.7 flop/B —
under the A100's 6.24 flop/B FP64 balance, so at the HBM point the arithmetic rides
under the 32 B/point bandwidth floor. Global traffic: exactly 157,216 B per volume
(one read + one write) on the fused path.

### Measured — reserved-node A100-SXM4-40GB over the tryout lease, vs cuFFT same case

| case | this entry | per transform | cuFFT | speedup |
|---|---|---|---|---|
| B=1 (launch point) | **9.87 µs** | 9.87 µs | 13.58 µs | **1.38×** |
| B=213 (in+out = 32 MiB ≈ L2 point) | **31.86 µs** | **149.6 ns** | 67.5 µs / 317 ns | **2.12×** |
| B=13656 (1 GiB buffer ≈ HBM point) | **1611 µs** | **118.0 ns** | 4766 µs / 349 ns | **2.96×** |

Effective bandwidth at the HBM point: **1332 GB/s** (driver's GB/s column) against the
~2.0 TB/s SXM4 peak — 67%, cuFFT sits at 450 GB/s (i.e. cuFFT is spending ~3 global
passes where this kernel spends 1, exactly the corpus's diagnosis of its Rader path).
rel L2 vs numpy: 3.07–3.17e-16 at every batch tried (1, 2, 4, 8, 11, 12, 16, 32, 64,
100, 213, 2160, 13656); bit-identical across processes at every batch;
`compute-sanitizer --tool memcheck` clean on both paths. ncu at B=2160 (fused): DRAM
72%, SM 58%, achieved occupancy 29% (2 blocks/SM is the shared-memory ceiling; that IS
the design). Note: several measurement windows this day had 8–13% sd on the leased GPUs
(cuFFT too); min-of-samples is the statistic quoted, and clean windows (sd 0.02%)
reproduced the same minima.

Path crossover, measured (fused vs split µs/call): B=4 15.5/14.9, B=8 15.4/15.0,
B=16 15.6/16.3, B=64 15.8/34.1 → CUT = 12. The fused kernel is flat ~15–16 µs from B=4
to B=64 (one wave, single-volume latency); the split path scales with B.

### What was tried and did NOT work — with the numbers

1. **Fusing the global load into the z pass** (a z line is 272 contiguous bytes, read
   straight into registers). HBM point 2358 µs vs 1657 staged — **42% slower**. With the
   164 KB carveout only ~28 KB of L1 remains; the warp's 272 B-strided 16 B requests then
   genuinely fetch every 32 B sector twice (the paired half is evicted before use). The
   x-pass store fusion, whose per-k writes are warp-contiguous, is fine and stayed.
2. **Rader proper (the entry's name), analysed and dropped without building.** Two FFT16s
   + 16 pointwise complex multiplies is 428 flops/line vs line17w's 496 — but a per-thread
   FFT16 keeps ~16 complex + temporaries ≈ 160+ live registers, past the 102/thread that
   two-blocks-per-SM allows, and the flop saving buys nothing that is not already under
   the bandwidth floor. On a GPU the flop count is not the scarce resource; registers are.
3. **Preloading the 16 per-k coefficients into per-thread arrays in `fft17_planes`**
   under a 48-register occupancy budget: ptxas spills them, and the spill traffic made
   the kernel "Memory 81%, DRAM 6%" in ncu — 51 µs at B=64 for a kernel now at ~10 µs.
4. **Divergent in-loop `__ldg` coefficient reads in `fft17_planes`** (thread map
   k = tid%17: 17 distinct rows per warp): B=64 went 48 → 63 µs. Fixed by swapping the
   map to k = tid/17 (warp-uniform rows, 1–2 addresses per warp) at the cost of a mild
   2-way shared conflict: 63 → 34 µs for the split path at B=64.
5. **Persistent blocks with a cp.async pipeline** (x pass folds its line to registers,
   barrier, prefetch volume v+gridDim.x into the now-dead shared buffer, then emit +
   store). At 2 blocks/SM the fold state held across the barrier spills (364 B/thread):
   HBM point **2820 µs**. At 1 block/SM (168 regs, no spills) the pipeline works but the
   lost co-residency costs more than the hidden load buys: **1735 µs vs 1611 plain**.
   Two co-resident blocks drifting out of phase are a better load/compute overlap than
   an explicit single-block pipeline on 10 warps. Do not retry without a way to keep
   two blocks AND the prefetch (e.g. warp-specialised loaders (?) — unmeasured).
6. **`__constant__` for the 17-row tables of the planes kernel** — rejected on the
   documented serialization of divergent constant-cache access before being built; the
   `__device__`-global + `__ldg` form with warp-uniform rows measures fine.

### Borrowed this round (attribution)

* **`L17_matrixsimd` (CPU, round 1):** the conjugate-pair fold making every coefficient
  real — the whole reason the driver's interleaved complex layout is directly usable
  with broadcast coefficients and no cross-lane work.
* **`L17_winograd` (CPU, round 1, via matrixsimd round 2):** the cyclic-4 ⊕ negacyclic-4
  split of the cosine circulant and the negacyclic-8 sine treatment, constants and index
  tables verbatim; their CPU dead-ends (WFTA-17, further negacyclic splits) were treated
  as settled and not retried on the GPU.
* **Literature §09:** the one-read-one-write target, the 163 KB shared opt-in, the
  odd-stride no-padding fact, the 32 B-sector coalescing model that explained item 1,
  and the DMMA analysis — which I read and deliberately did NOT take: the m8n8k4 padding
  waste at L=17 is 1.66×, and this entry's CUDA-core arithmetic already hides under the
  bandwidth floor, so tensor cores could only help via fewer shared reads, which is the
  rival `L17_dmma` entry's question to answer.

### Next round, in order

1. **The HBM point sits at DRAM 72% / 1332 GB/s; the ~25% gap is phase serialization.**
   The z+y passes are DRAM-idle; the only overlap is two co-resident blocks de-phasing.
   The untried structural fix: plane-granular pipelining with `cuda::barrier` (mbarrier)
   per plane — z and y are exactly plane-local, so plane p's z/y compute can start when
   plane p's cp.async group lands, without waiting for the whole volume. That would
   overlap ~2/3 of the compute with the load stream inside one block and could plausibly
   reach ~1.6 TB/s ≈ 98 ns/t. Nontrivial: a plane's 17 lines straddle warp boundaries.
2. **B=213 is wave-quantized:** 213 blocks on 216 slots, per-call ≈ 2× single-volume
   latency because co-resident pairs share the FP64 pipe and LSU. Any cut in per-volume
   issue count (e.g. double2-vectorizing the two shared line reads a+b into fewer LDS)
   moves this point almost 1:1.
3. **The split path's `fft17_xlines` is latency-bound** (DRAM 22%, SM 12%, 14.8 µs of
   B=64's split time) — if the B=1 cell matters next round, fold the x pass into the
   planes kernel with a cooperative grid sync (17 blocks ≤ 108 SMs, so a cooperative
   launch is legal at B ≤ 6) and save a launch plus the L2 round trip.
4. **Compare with `L17_dmma`'s measured numbers** (same geometry, matrix formulation on
   tensor cores). If their shared-read savings beat this entry's 118 ns/t, the merge is
   their DMMA panels inside this entry's one-read-one-write volume-resident shell.
5. If a future round needs finer path selection, replace the compile-time CUT=12 with a
   plan-time measured A/B (`create()` may time both paths; setup is unscored).

---

## Round gpu_r2 (2026-08-22) — load/compute overlap, split-path latency cuts

Context at the start of the round (gpu_r1 leaderboard): won B=1 (9.76 vs L17_dmma's
11.36 µs) and the L2 point (149 vs 168 ns/t), **lost the primary HBM point to
L17_dmma by 2%** (118.0 vs 115.8 ns/t). Two facts taken from the rival records before
touching code: the SXM4-40GB peak is **1555 GB/s, not the brief's ~2.0 TB/s**
(hardware correction from `L13_dmma`/`L17_dmma` round 1 — 1215 MHz memory clock, ncu
percentages agree), so r1's 1332 GB/s was already 86% of peak; and cp.async staging
was worth −2.5% to L17_dmma.

### What changed

1. **Warp-chunked cp.async staging fused into the z pass** (fused kernel). The z map
   is line = tid, so warp w's 32 z lines cover exactly the contiguous elements
   [544w, 544w+544). Each warp now issues cp.async (16 B copies → `cp.async.cg`,
   global→shared direct, bypassing the ~28 KB of L1 left under the max carveout) for
   its **own** chunk only, `__pipeline_commit`/`wait_prior(0)`s on its own group,
   `__syncwarp()`s, and starts its z lines while later warps' loads are still in
   flight. Load/z overlap with zero cross-warp synchronization, on top of the
   two-co-resident-block de-phasing; also deletes the load→z `__syncthreads` and the
   15-register staging buffer. The idea of cp.async came from **`L17_dmma` round 1**;
   the per-warp chunking + early z start is new (theirs is a block-wide flat copy with
   a full barrier). HBM point 1611 → **1574–1586 µs** across windows (−2.3%); ncu at
   B=2160: DRAM 72 → **80.7%** of peak, FP64 pipe 58 → 65%. Neutral at B=213 (A/B in
   the same window: 32.10 old vs 32.24 µs new, sd ~7%). Old staging kept under
   `-DL17RF_STAGE_R1`.
2. **Split-path planes kernel: removed a no-op staging round-trip.** r1 staged the
   y-pass result through shared "so the global store is coalesced" — an indexing
   thinko: the y output of thread (k = tid/17, ln = tid%17) lands at plane offset
   k*17+ln **= tid**, so the direct global store was already coalesced. The staging
   store, its two `__syncthreads`, and the extra shared load are simply deleted.
3. **`fft17_xlines` block size 128 → 32.** The kernel is latency-bound (r1: DRAM 22%,
   SM 12%); at B=1, 289 lines in 128-thread blocks is 3 blocks = 3 SMs. 32-thread
   blocks put 10 SMs on it: 10.08 → 9.15 µs at B=1 (16/32/64 all measure 9.0–9.15;
   128 is the only bad choice). Combined with item 2: **B=1 9.87 → 8.81 µs**.
4. **Path cut retuned 12 → 14** (`L17RF_CUT`): the faster split path now wins B=12
   (11.15 vs 13.36 µs) and B=13 (13.09 vs ~13.4), tie at 14.

Operation count unchanged: 496 real flops/line, 867 lines/volume, one global read +
one global write per volume on the fused path.

### Measured — reserved-node A100-SXM4-40GB over the tryout lease (min of samples)

| case | gpu_r1 | gpu_r2 | per transform | cuFFT same window | speedup |
|---|---|---|---|---|---|
| B=1 | 9.87 µs | **8.81 µs** | 8.81 µs | 13.66 µs | **1.55×** |
| B=213 (L2) | 31.86 µs | **32.1–33.0 µs** (windows noisy, sd 7%) | ~150 ns | 67.1–68.1 µs | **2.1×** |
| B=13656 (HBM) | 1611 µs | **1574–1586 µs** | **115.3–116.1 ns** | 4767 µs | **3.0×** |

HBM effective bandwidth **1363 GB/s = 87.7% of the 1555 GB/s peak** (r1: 85.7%);
L17_dmma's r1 number on the same case was 1582–1587 µs, so the primary cell should
flip. rel L2 3.07–3.16e-16 at every batch tried (1, 2, 4, 8, 11, 12, 13, 14, 16, 24,
32, 48, 64, 213, 13656); bit-identical across runs at every batch;
`compute-sanitizer --tool memcheck` clean on both paths (B=8 split, B=64 fused).
B=213 could not be cleanly separated from r1 today — every window at that batch had
6–8% sd (cuFFT elevated identically); the minima match r1 within ~1%.

### What was tried and did NOT work — with the numbers

1. **Cooperative single-launch split path** (adopted from `L17_dmma` round 1, where it
   was worth 1.7 µs): planes body + `grid.sync()` + x lines in one
   `cudaLaunchCooperativeKernel`. **13.95 µs vs 10.08 two-launch at B=1** — the
   cooperative launch + grid.sync cost ~3.9 µs more than the second launch. It paid
   for dmma because their two kernels are heavy (dense per-output z+y, global
   scratch); against two already-lean launches it is pure overhead. Kept under
   `-DL17RF_COOP` so nobody re-measures it.
2. **Dense thread-per-output x pass** (adopted from `L17_dmma`'s `fft17_x_dense`:
   17B blocks × 289 threads, planes→scratch→out): **11.96 µs vs 10.08** at B=1. The
   17× SM coverage does not compensate for the scratch round trip and the 17×
   read amplification of the volume. Their x_dense is only right next to their
   heavier planes kernel. Kept under `-DL17RF_XDENSE`.
3. **Streaming stores (`st.global.cs`) on the fused x pass**: clean-window A/B at
   B=13656: **1599.6 µs vs 1575.3 plain** (+1.5%). Evict-first on the write stream
   does not help a kernel whose reads already stream through L2 once. Kept under
   `-DL17RF_STCS`.
4. **Warp-per-plane variant (544 threads, plane-owning warps, z AND y overlapped with
   load) — rejected by arithmetic before building.** With ncu showing the FP64 pipe
   at 65% after change 1, plane-granular warps run 17-lane-active line passes at 53%
   lane efficiency → z+y issue cost ×1.7 ≈ 94% pipe → compute-bound, and 544 threads
   ×2 blocks busts the 60-register budget (line17w needs ~80 live doubles at peak).
   A 2-planes-per-warp variant at 320 threads has 34 lines on 32 lanes: the 2
   leftover lines serialize a full second line17w issue stream per warp per pass.

### Borrowed this round (attribution)

* **`L17_dmma` round 1**: the cp.async staging direction (reworked into per-warp
  chunks + early z), the cooperative-launch and dense-x ideas (both measured, both
  rejected here — see above), and the co-residency occupancy check pattern in
  `create()`.
* **`L13_dmma` / `L17_dmma` round 1**: the 1555 GB/s SXM4-40GB bandwidth correction,
  which reframes this kernel as ~88% of peak rather than ~67% — i.e. near the wall,
  so round 3+ effort should go to the L2 and B=1 cells, not HBM heroics.

### Next round, in order

1. **The remaining HBM gap (~12% to the 1381 µs floor) is the y window**: z overlaps
   the load now, x overlaps the store; y is the one phase with zero memory traffic.
   The only structure that overlaps y without killing occupancy or lane efficiency is
   plane-granular mbarrier pipelining (`cuda::memcpy_async` + 17 shared barriers with
   17-thread waiter groups). Fiddly; expected ceiling ~1450–1500 µs. Measure the
   barrier-poll cost on a toy first.
2. **B=213 (~150 ns/t vs 115.3 at HBM) is still wave-quantized latency**: 213 blocks
   on 216 slots, per-call ≈ 2× single-volume latency because co-resident pairs share
   the FP64 pipe and LSU. Any cut in per-volume issue count moves this cell ~1:1.
   The two remaining `__syncthreads` (z→y, y→x) are candidates: a 2-volume block
   sharing barriers is NOT (barriers would couple 640 threads), but the z→y barrier
   could become 17 shared mbarriers with plane-local waiters, same machinery as
   item 1.
3. **B=1 residual (~8.8 µs)**: profile where it sits now — planes (17 blocks) vs
   xlines (10 blocks) vs two launch overheads. If launches dominate, the only lever
   left is a single non-cooperative kernel doing z+y+x for B≤2 via inter-block
   spin-flags on L2 (risky, and the driver's correctness gate punishes any hang —
   prototype off-line first).
4. If the monitor's scored B=213 window is as noisy as today's, consider a plan-time
   measured A/B for CUT (create() may time both paths; setup is unscored).
