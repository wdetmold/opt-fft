# L13_dmma — strategy record

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (35,152 B), forward,
unnormalised, out-of-place, batched, one A100. Implementation: `impl/L13_dmma.cu`.
Scored cases (cases.txt): B = 1, 477 (L2-resident), 30549 (HBM, primary).

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique

Fused single kernel, **one volume per block in shared memory, one thread per 13-point
line, three axis passes, one global read + one global write**:

1. Stage the volume into shared with the classic interleaved-coalesced pattern:
   169 threads × 13 fully-unrolled iterations (169·13 = 2197 exactly, so no bounds
   check and 13 independent 16-byte loads in flight per thread).
2. z-pass (stride 1), y-pass (stride 13), x-pass (stride 169), separated by
   `__syncthreads()`; each of 169 threads owns one whole line in registers.
   The x-pass writes its results **directly to global** (for fixed output index k,
   consecutive threads hit consecutive complex doubles — coalesced), saving one full
   shared round trip.
3. The line transform is the **conjugate-symmetric folded dense matvec** — adopted
   wholesale from the CPU phase's `L13_direct` winner (geom panel_r6), which itself
   derives from `L17_matrixsimd`; also literature 09 §9.3 structures 1+3:

   ```
   u_j = x_j + x_{13-j},  v_j = x_j - x_{13-j}          (j = 1..6)
   P_k = x_0 + Σ_j cos(2π kj/13) u_j
   S_k =       Σ_j sin(2π kj/13) v_j                    (k = 1..6)
   X_0 = x_0 + Σ u_j,   X_k = P_k - i·S_k,   X_{13-k} = P_k + i·S_k
   ```

   All coefficients real; the 6×6 cos/sin tables live in `__constant__` (uniform
   access per fully-unrolled (k,j) step → broadcast, free). Folding on load keeps
   only u, v, x0 and four accumulators live.

Block shape: 192 threads (6 warps; 169 compute). **80 registers/thread, zero spills**,
static shared 35,152 B → **4 blocks/SM** (register- and shared-capped simultaneously;
5 blocks would need 176 KB > the 164 KB carveout). `__launch_bounds__(192, 4)` plus a
`PreferredSharedMemoryCarveout = 100` hint in create(). Stride 13 is odd, so every
pass is bank-conflict-free with no padding (lit 09 §6.2 — the odd-L gift).

### Operation count

Per 13-point line: 12 complex fold add/sub (24 real ops) + X0 sum (12) + 6×(24 FMA +
4 add/sub) = **~204 FP instructions ≈ 350 real flops per line**, i.e. ~27 flop/point
per axis, **~81 flop/point for the full 3D transform** — against 936 for the naive
dense complex matvec and 55.5 for a textbook butterfly. Arithmetic intensity
81/32 ≈ 2.5 flop/B against the 6.24 machine balance: comfortably under the HBM floor
on the **vanilla** FP64 pipe, which is why DMMA was not needed (below).

### Measured (tryout.sh → leased A100-SXM4-40GB of the reserved node)

| case | per transform | GB/s | cuFFT same case | speedup |
|---|---|---|---|---|
| B=1 | **7.88 µs**/call | — | 12.14 µs | 1.54× |
| B=477 (L2) | **59.5 ns** (28.39 µs/call) | 1181 | 63.5 µs/call | 2.24× |
| B=30549 (HBM, primary) | **51.2 ns** (1564.0 µs/call) | **1373** | 4731.5 µs | **3.03×** |

Correctness: rel L2 = 3.2e-16 at every batch (B=30549 checked against numpy on 400
sampled volumes; check.py itself cannot allocate 1 GiB on the login node). Bit-identical
across runs.

ncu on the HBM case (earlier build at 1585 µs): **dram__throughput 86.7% of peak,
dram__bytes 2.13 GB = exactly the read-once/write-once minimum**, FP64 pipe 60%,
L2 70%, L1 48%, achieved occupancy 36%. The final build (1564 µs) is ≈ **88% of DRAM
peak moving minimum bytes** — the kernel is at the bandwidth roof; VkFFT's published
best-in-class band is ~82–84%.

**Hardware note that changes the roofline math for everyone:** the brief's table says
the SXM4 node has "~2.0 TB/s". The leaderboard header shows 1215 MHz memory clock →
5120-bit × 2 × 1.215 GHz / 8 = **1555 GB/s**, and ncu agrees (1355 GB/s reported as
86.7% of peak ⇒ peak ≈ 1563). The **A100-SXM4-40GB** is a 1555 GB/s part like the
PCIe card; only the 80 GB SXM4 has 2.0 TB/s. The lit 09 floor table (45.2 ns per
volume at L=13) is therefore the right target here, and we sit at 1.13× of it.

### What did NOT work, with the numbers that killed it

* **Fusing the z-pass into the global load** (each thread reads its own contiguous
  208 B z-line, transforms, writes shared — saves one barrier and all staging):
  **1244 GB/s vs 1355** at B_HBM, 8% slower end-to-end. Per-instruction the 13-complex
  stride-208 pattern uses 16 B of every 32 B sector and relies on L1 to merge the
  halves; with the 164 KB shared carveout L1 is only 28 KB and it does not hold.
  Confirms lit 09 §6.1: keep the classic interleaved pattern for the global side and
  do all transposition in shared.
* **`cp.async` (`__pipeline_memcpy_async`) for the staging loop**: 1563.6 µs vs
  1562.5 — exactly neutral (nothing to overlap in a load-once kernel; lit 09 §6.3
  predicted this). Dropped for simplicity.
* **`__ldcs` (evict-first) on the staging loads**: 1645 µs vs 1562 — 5% *worse*.
* **A specialised B=1 kernel** (384 threads, each line's outputs split across two
  threads 192 apart so the halves live in different warps; results parked in
  registers across a compute/store barrier): **11.0 µs vs 7.9 µs** — the three extra
  barriers cost more than the six extra warps bought. The plain fused kernel stays
  the only kernel.

### Why the entry's namesake (DMMA) is deferred, deliberately

The measured limiter at the primary point is DRAM at ~88% of peak with **minimum
bytes moved** — there is at most ~12% left in the whole kernel and none of it is
arithmetic (FP64 pipe: 60%). The folded matvec is already at 0.42× of the flop
budget the bandwidth floor allows on the *vanilla* pipe, so moving the same flops to
tensor cores cannot move the time. DMMA would matter only if (a) the B_L2 regime
(where DRAM is 34% and the kernel is latency/barrier-bound — FP64 42%) can be
restructured so arithmetic becomes the limiter, or (b) a future variant needs the
shared-memory *read* traffic reduction that DMMA's operand reuse gives (lit 09 §5.5).
Neither is true of this round's kernel. This is a measured decision, not an evasion:
see the ncu numbers above.

### What I would do next

1. **B_L2 (477) is the soft spot**: 59.5 ns against 51.2 at B_HBM even though the
   data is L2-resident (7.2 TB/s available). ncu shows nothing saturated — stalls are
   long-scoreboard (7.7/warp) and barrier (6.1/warp), plus a 1.10-wave tail
   (477 blocks over 432 resident). Ideas: 2 volumes per block only for mid-size
   batches (halves barrier count per volume), or a persistent-block form.
2. Try padding the shared array to break the z-pass's *phase* pattern further — not
   expected to matter (no conflicts measured), but cheap to A/B under ncu's
   `l1tex__data_bank_conflicts` metric.
3. B=1 at 7.9 µs is ~60% kernel, ~40% launch; the kernel path is latency-bound with
   6 warps on one SM. The two-half split failed; a 4-way split with `__syncwarp`
   choreography inside one warp-pair might do better, but the point is scored as
   launch overhead and cuFFT is already 1.54× behind.
4. If anyone wants the DMMA experiment: do it at **B_L2**, as three back-to-back
   real GEMMs (9×13 cos / sin against 13×169·B panels), and compare against this
   entry's 59.5 ns — that is the number to beat, not the HBM one.

---

## Round gpu_r2 (2026-08-22) — evict-first store fixes the L2 point; four structural
## alternatives measured dead

### What changed in production

**One change: the final global store is issued with `__stcs` (evict-first) whenever
the working set is L2-resident** (`2·B·35152 B ≤ 40 MiB`, i.e. B ≤ 596), selected at
plan time. **Adopted from `L8_blockfused` round gpu_r1**, whose record diagnosed my
round-1 "B_L2 soft spot" exactly: at the L2 point the driver re-reads `in` every
timed call, and with plain stores the output stream evicts it, so mixed read+write
traffic through L2 sustains no more than DRAM rate. Evict-first stores send the
output to DRAM without polluting L2, dedicating L2 to the input stream. Their −43% /
−1% (L2/HBM) measurement is why it is per-batch selected, not hardcoded; at B_HBM
this plan picks the unchanged round-1 kernel.

The kernel is otherwise round 1's: one volume per 192-thread block, staged
interleaved-coalesced into 35,152 B shared, z/y/x passes with the conjugate-folded
dense-13 line (~81 flop/point over three axes, unchanged), x-pass writing straight
to global. 80 registers, zero spills, 4 blocks/SM.

### Measured (tryout.sh → leased A100-SXM4-40GB; min over samples)

| case | gpu_r1 | gpu_r2 | per transform | cuFFT | speedup |
|---|---|---|---|---|---|
| B=1 | 7.88 µs | **7.86 µs** (unchanged) | 7.86 µs | 12.24 µs | 1.56× |
| B=477 (L2) | 28.39 µs | **21.56 µs (−24%)** | **45.2 ns** | 63.1 µs | 2.93× |
| B=30549 (HBM, primary) | 1563.3 µs | **1563.8 µs** (unchanged) | **51.2 ns** | 4731 µs | 3.03× |

rel L2 = 3.2e-16 at B=1/477/1024 (B=1024 exercises the plain-store template path);
bit-identical across runs at every point. The B_L2 cell now beats the B_HBM cell per
volume, as it should. ncu at B=477 after the change: DRAM 33%, L2 50%, SM 46%,
occupancy 29%, warps 27.8 stall-cycles/instr, 1.10 waves — the residual 21.6 µs vs
the ~11.6 µs pure-write floor is a latency/barrier mixture at the structural
24-warps/SM ceiling (4 blocks × 35 KB shared is the cap; a 5th block needs 176 KB),
plus the 45-block second wave. Fixing that needs cp.async/mbarrier plane pipelining,
which L17_raderfused's record (item 5) already measured as a loss in two forms —
left alone.

### What was tried and did NOT work, with the numbers that killed it

* **Fused-z global load at the L2 point** (round 1 killed it at HBM only; retried
  here because L2-resident reads should absorb the half-used 32 B sectors):
  B=477 **25.0 µs vs 21.7** staged (+15%). L2 read-sector amplification is not free
  even at 50% L2 utilisation. Dead in both regimes now.
* **V=2 volumes per 384-thread block** (idea from `L8_blockfused`, whose L2-point
  autotune picked V=2): B=477 **27.6 µs vs 21.7** (+27%), B=30549 **1587.7 vs
  1563.8** (+1.5%). Same direction as their V=4/8 losses: my 6-warp block is already
  the right barrier granularity, and doubling it only coarsens the barriers. Their
  V=2 win was V=2×64=128 threads — still smaller than my V=1 block.
* **Two threads per line at B=1** (retry of round 1's failed split, redesigned with
  ping-pong shared buffers so there is no register parking and no extra barrier —
  warp-aligned halves, h = t/192): **7.90 vs 7.84 µs — exactly nothing**. The six
  k-iterations of a line are mutually independent, so the compiler already overlaps
  them; halving the outputs per thread does not shorten the critical path, which is
  shared-load latency plus launch (~3.4 µs of the 7.86). B=1 is done short of a
  cooperative plane-split, and the L17_dmma coop measurement (11.4 µs for a volume
  2.24× bigger, saving 1.7 µs over two launches) says the ceiling there is ≲1 µs.
* **Pass-reordered kernel (x-in-registers during the staging load)** — the round's
  one genuinely new structure: staging round k for thread t IS element k of x-line
  (y,z)=t, so the x-DFT can run in registers during the identical perfectly-coalesced
  load, saving a full shared round trip and a barrier (2R+2W shared/point instead of
  3R+3W, 2 barriers instead of 3). Cost: the final z-pass stores 13 consecutive
  double2 per thread (per-instruction scattered, per-warp full-coverage 6656 B span).
  Measured: B=477 **37.8 µs vs 21.8** (+74%), B=30549 **2447.6 vs 1563.8** (+57%,
  877 GB/s effective ≈ 2× write amplification). **A100's L2 does not merge
  half-sector stores from separate instructions** — the 16 B stores each cost a 32 B
  sector to DRAM. Together with round 1's loadz result this closes the question
  symmetrically: *both* global sides must be classic interleaved-coalesced; every
  transpose belongs in shared. Do not rediscover this on either side.

All four dead ends stay in the file behind `L13_FORCE_{LOADZ,V2,SPLIT,FUSEDX}` env
knobs (never selected by default), so the next generation can re-A/B in one lease
command: `on_gpu.sh -- "L13_FORCE_X=1 build/tryout/L13_dmma/bin ..."` — note env vars
must be set inside the ssh'd command string; a prefix on tryout.sh does not cross
`on_gpu.sh`'s ssh boundary (I lost one A/B cycle to that).

### Borrowed this round (attribution)

* **`L8_blockfused` round gpu_r1**: the `__stcs` evict-first store and its L2-vs-HBM
  selection logic — the entire production gain of this round — plus the warning that
  bigger V loses.
* **`L17_raderfused` / `L17_dmma` round gpu_r1**: read for the small-batch split and
  persistent-pipeline results; their measured failures (persistent blocks at 2
  blocks/SM spill; pipeline at 1 block/SM loses co-residency) are why I did not
  spend this round on a persistent B_L2 kernel.

### What I would do next

1. The primary cell is at **88% of DRAM peak moving minimum bytes** — structurally
   finished, same verdict as L8 (90%) and L6 (89%). Only a fundamentally better DRAM
   schedule (persistent read/write interleaving) could move it, expected ≤ 5%, and
   two entries' persistent-kernel attempts already lost.
2. B_L2 residual (21.6 vs ~11.6 µs write floor): the honest lever is plane-granular
   cp.async pipelining inside the block (load plane p+1 while transforming plane p)
   with mbarrier instead of __syncthreads — untried in this entry, failed in
   L17_raderfused's whole-volume form. Modest expected value; high effort.
3. B=1 (7.86 µs, ~3.4 µs launch): a cooperative 13-block plane-split is the only
   untried shape; L17's numbers cap the gain at ~1 µs. Bottom of the list.
4. DMMA remains unjustified by measurement: FP64 pipe ≤ 50% in every regime; the
   entry keeps its name as a monument to the analysis that killed it.

---

## Round gpu_r3 (2026-08-22) — soft-barrier single-launch plane split takes B=1;
## the warp-chunked cp.async idea measured dead at both batched points

### What changed in production

**One new path: at batch ≤ 8 the transform is a single plain launch of `fft13_planes`
— 13·B blocks of 192 threads joined mid-kernel by a software grid barrier.** Adopted
from **L17_dmma round gpu_r2** (their measured launch mechanics: a cooperative launch
costs ~1.4 µs over `<<<>>>`, a second plain launch ~1.6 µs; their soft barrier took
L=17's B=1 from 11.49 to 7.74 µs). My r2 record had capped this idea's value at ~1 µs
using their *r1* cooperative numbers — the r2 soft-barrier form beats that cap.

Structure per block (b, p):
1. Phase 1: stage the contiguous 169-element x-plane p into shared (one element per
   thread, coalesced), dense per-output z-DFT (thread = (y,kz), row-contiguous shared
   reads), dense per-output y-DFT (thread = (ky,kz), stride-13 shared reads — odd, so
   conflict-free), write the plane to a plan-owned global scratch, coalesced.
   Twiddles for both passes are lane-divergent (kz/ky vary across lanes), so they come
   from a `__device__` global table via `__ldg`, NOT constant memory — L17_dmma r1's
   finding #3 (divergent constant reads replay 32-way) respected, not rediscovered.
2. Software grid barrier, ported from L17_dmma r2: thread 0 does
   `__threadfence(); atomicAdd(ctr, 1)` then spins until the plan-owned MONOTONIC u64
   counter reaches `epoch · gridDim` (target passed as a kernel argument, so
   back-to-back calls on one stream cannot confuse epochs); `__syncthreads` on both
   sides, `__threadfence` after the spin for acquire. `create()` verifies the whole
   13·B-block grid co-resident (occupancy query × SM count) and falls back to the
   fused kernel otherwise; counter and scratch are cudaMalloc'd in create().
3. Phase 2: block (b, kx=p), thread t=(ky,kz) sums the 13 scratch planes at stride
   169 (coalesced across the warp at each x) against `__constant__` twiddles indexed
   by (kx·x) mod 13 — kx is block-uniform, so this IS the broadcast pattern constant
   memory is good at. Contiguous coalesced store of the output plane.

Dense per-output arithmetic is ~2× the folded matvec's flops and the scratch round
trip doubles global traffic — both irrelevant at B ≤ 8 where the whole problem is
launch- and latency-bound (~35 KB of data, 13 SMs busy).

The batched points are untouched: the r2 fused kernel (conj-folded lines, __stcs
when L2-resident) still carries B=477 and B=30549.

### Operation count

Fused path unchanged (~81 flop/point over three axes). Planes path: dense per-output
= 13 complex FMA per point per axis ≈ 26 real flop/point/axis plus twiddle loads —
~2× the folded matvec, chosen deliberately: it keeps all 169 threads of a plane
active per pass instead of 13 thread-per-line threads.

### Measured (tryout.sh → leased A100-SXM4-40GB; min over samples)

| case | gpu_r2 | gpu_r3 | cuFFT same window | speedup |
|---|---|---|---|---|
| B=1 | 7.86 µs | **6.78–6.95 µs (−13%)** | 12.39–12.52 µs | **1.8×** |
| B=477 (L2) | 21.56 µs | **21.70 µs** (unchanged path) | 62.9 µs | 2.9× |
| B=30549 (HBM, primary) | 1563.8 µs | **1563.7 µs** (unchanged path) | 4743 µs (r2 board) | 3.03× |

Crossover sweep, same-window A/B (planes vs fused, µs/call): B=2 7.12 vs 8.19,
B=4 7.25 vs 8.45, B=8 7.63 vs 8.41, B=10 9.17 vs 8.47, B=12 9.14 vs 8.24,
B=16 10.04 vs 8.27, B=24 12.53 vs 8.36 → cut at **B ≤ 8**. (The planes path's
per-call cost grows with B because 13·B blocks × 2× traffic; the fused kernel is
flat ~8.3 µs for one wave.)

rel L2: 3.9–4.0e-16 on the planes path (B=1, 4, 8 — dense per-output accumulates
slightly differently than the folded matvec's 3.2e-16; both far under the 1e-12
gate), 3.2e-16 on the fused path (B=477). Bit-identical across runs at every point;
`compute-sanitizer --tool memcheck` 0 errors on the planes path at B=8.

### What was tried and did NOT work, with the numbers that killed it

* **Warp-chunked cp.async staging fused into the z-pass** (adopted for trial from
  **L17_raderfused round gpu_r2**, where it bought −2.3% at their HBM point): warp
  w's 32 z-lines cover exactly the contiguous elements [416w, 416w+416), so each
  warp cp.asyncs its own chunk, waits its own group, `__syncwarp`s, and starts z
  while other warps' loads are in flight — deleting the load→z `__syncthreads`.
  Same-window A/B: B=30549 **1568.7 vs 1563.7 plain** (+0.3%, median +1%),
  B=477 **22.86 vs 21.70** (+5%). **L8_blockfused r2's warning wins the tug-of-war**
  ("do not perturb a global access pattern at 90% of DRAM peak" — their warp-local
  staging lost 1.3% the same way): at L=13's 4-blocks/SM residency the co-resident
  blocks already de-phase the load streams, and the per-warp-contiguous read order
  costs more than the deleted barrier saves. L17_raderfused's gain evidently needed
  their 2-blocks/SM regime, where in-block overlap is the only de-phasing available.
  Kept behind `L13_FORCE_WCP=1`.
* Planes path at B ≥ 10 (see crossover table) — the cut is a measured number, not a
  guess.

### Borrowed this round (attribution)

* **L17_dmma round gpu_r2**: the entire B=1 win — single plain launch + arrive-and-
  spin software grid barrier with a monotonic epoch counter, the co-residency check
  in create(), and the release/acquire fence discipline; plus their r1 finding that
  lane-divergent twiddles must come from global/`__ldg`, not constant memory (used
  in phase 1), while block-uniform ones belong in `__constant__` (phase 2).
* **L17_raderfused round gpu_r2**: the warp-chunked cp.async idea (measured, dead
  here — numbers above so nobody retries it at L=13).
* **L8_blockfused round gpu_r2**: the read-order warning that correctly predicted
  the wcp result.

### What I would do next

1. **B=1 residual (~6.8 µs ≈ 3.4 µs launch + ~3.4 µs kernel)**: phase 2's serial
   13-load global chain and the two dense passes are the kernel time. A folded
   per-output form would halve phase arithmetic but the loads dominate; expected
   ≲0.5 µs. Diminishing returns.
2. **B_L2 (21.7 µs vs ~11.6 µs write floor)**: every structural idea is now measured
   dead by me or a neighbour — persistent blocks (L8 r2: +5.6%), plane-granular
   cp.async pipelining (L17_dmma r2: loss at both points), V=2 (mine r2: +27%),
   warp-chunked overlap (this round: +5%), 2-thread lines (register-file math says
   no more blocks). I believe this cell is structurally finished at 24 warps/SM.
3. **B_HBM stays closed**: 88% of DRAM peak at minimum bytes, three entries with the
   same verdict, every overlap trick measured negative. Do not spend rounds here.
4. If a future geometry wants the planes path at larger B: the limiter is the 2×
   traffic through scratch, not the barrier — a fused-phase-2-into-phase-1-of-the-
   next-call shape does not exist under the per-call contract.
