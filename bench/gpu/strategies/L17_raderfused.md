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
