# L8_blockfused — strategy record

Geometry: **L = 8**, cube 8³ = 512 complex doubles (8,192 B) per volume, forward,
unnormalised, out-of-place, batched, one A100-SXM4-40GB.
Implementation: `impl/L8_blockfused.cu`. Scored cases (`cases.txt`): B = 1, 2048 (L2),
131072 (HBM, primary).

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique

Fused single kernel, literature 09 §9.2 structure 1: **V volumes per block staged in
shared memory, one thread per 8-point line (64 threads per volume), three axis passes,
one global read + one global write per volume.** Structural skeleton adopted from
`L13_dmma` round gpu_r1 (staging pattern, direct-to-global final pass, full shared
carveout hint) — with attribution.

* **Shared layout**: element (x,y,z) at `(x*8+y)*9 + z` — the z-row stride padded
  **8 → 9**. gcd(8,8) = 8 makes the natural layout the worst-case 8-way bank conflict
  (lit 09 §6.2 called it "the single most likely silent performance bug at this
  geometry"). With the padded layout every pass's 8-lane phase has lane stride 9 (odd,
  conflict-free) or 1: z-pass lines start at multiples of 9; y- and x-pass phases run
  along contiguous z. ncu confirms: 6.4 M residual conflicts at B=131072 ≈ 1.5% of
  shared accesses — noise.
* **Staging**: flat interleaved-coalesced copy, `V*64` threads × 8 unrolled 16-byte
  loads (V·512 exactly, no bounds check in full blocks), consecutive lanes touch
  consecutive global addresses; tail blocks (B mod V ≠ 0) take a guarded loop.
* **Passes**: z (stride 1), y (stride 9) in place in shared; x (stride 72) reads shared
  and writes **directly to global** — for fixed output k, consecutive threads hit
  consecutive complex doubles (perfectly coalesced), saving one shared round trip.
* **Line codelet**: minimum-operation radix-8 DIF, natural order in and out — the only
  irrational constant is 1/√2, giving **4 real mults + 52 real adds per line** (the
  published minimum; same arithmetic as the CPU phase's `L8_radix8`). No twiddle
  table, no constant memory.
* **Autotuned in create()** on scratch buffers (3 round-robin cycles, min): V ∈
  {1,2,4,8} × streaming final store (`__stcs`) ∈ {off,on} × staging mode ∈ {plain,
  `cp.async`}. All variants are bit-identical in output (same per-line arithmetic), so
  the pick affects time only. `__launch_bounds__(V·64, 65536/(64·V·64))` pins every
  variant at 64 registers, zero spills.
* **B = 1**: dedicated one-block kernel, no staging — the z-pass reads each thread's
  contiguous 128 B global z-line directly, two barriers total.

### Operation count

56 real flops × 192 lines = 10,752 per volume = **21 flop/point for the full 3D
transform** (vs 936 dense, 3·48 = 144 for a dense per-axis matvec). Arithmetic
intensity 21/32 B = 0.66 flop/B against the 6.24 machine balance — arithmetic is
invisible by 10×; the whole game is memory, as the corpus said.

### Measured — leased A100-SXM4-40GB of the reserved node, tryout.sh

| case | this kernel | per volume | cuFFT | speedup | rel_l2 |
|---|---|---|---|---|---|
| B=1 | **4.26 µs** | 4.26 µs | 9.14 µs | **2.15×** | 1.58e-16 |
| B=2048 (L2, 32 MiB) | **13.44 µs** | **6.56 ns** | 55.6 µs | **4.14×** | 1.59e-16 |
| B=131072 (HBM, 1 GiB) | **1534.0 µs** | **11.70 ns** | 3702 µs | **2.41×** | 1.59e-16 |

Autotune picks: B=2048 → V=2 + `__stcs`, plain staging; B=131072 → V=1/2 (within
noise), no `__stcs`, plain staging. Repeatable (bit-identical across runs) at every
batch; all 24 forced variant × tail combinations PASS at B=63 with identical error;
`compute-sanitizer memcheck` clean (fused with tail, and single). The full B=131072
numpy check must run **on the reserved node** — the login node's strict overcommit
kills 512 MiB numpy allocations under agent contention (generate inputs there too:
`./on_gpu.sh -- python3 gen_input.py ...`).

ncu at B=131072 (V=1): **dram__bytes 2.13 GB = exactly the read-once/write-once
minimum; DRAM 90.2% of peak** (L13's best was ≈88%, VkFFT's published band 82–84%),
L2 69.7%, L1 41.9%, occupancy 48.6%, issue 11.7%. The primary point is at the DRAM
wall moving minimum bytes — nothing structural left there. (Peak = 1555 GB/s, adopting
L13_dmma/L17_dmma's correction of the brief's "~2 TB/s".)

### The finding of the round: the L2-resident case is a *mixed r/w L2 throughput*
### problem, and evict-first stores fix it

At B=2048 (in+out = 32 MiB ≤ 40 MiB L2) the plain kernel ran **23.5 µs — no faster
per call than DRAM would allow**, with ncu showing nothing saturated: DRAM 53%,
L2 72%, occupancy 39%, and the top stall **MIO throttle 16.8 cycles/instr**. Mixed
read+write traffic through L2 sustains only ~1.4 TB/s — the same effective rate as
HBM, which is why B_L2 wasn't beating B_HBM per volume (L13's record flags the same
"soft spot" at their B=477).

`__stcs` on the final store (evict-first) sends the output stream to DRAM without
polluting L2, dedicating L2 to the input stream: **23.5 → 13.44 µs (−43%)**. The case
is now bound by the DRAM *write* stream alone (16.8 MB / ~1.45 TB/s ≈ 11.6 µs floor;
we sit at 1.16× of it). The same hint is **1% slower at B=131072** (both streams must
go to DRAM anyway, and the hint only distorts replacement) — so it must be a per-batch
autotuned choice, not a hardcode. L36_sharedtiled's `__stcs` gain (+2.5%) and
L13_dmma's `__ldcs` loss (−5%) are both consistent with this reading.

### What was tried and did NOT work, with the number that killed it

1. **z-pass fused into the global load** (each thread reads its own contiguous 128 B
   z-line; saves one shared round trip — 6 → 4 shared accesses/point — and one
   barrier): B=131072 **1612 µs vs 1534** (−5%, exactly L13's result for the same
   structure); B=2048 forced **27.0 µs vs 23.5** plain / 13.6 best. The stride-128 B
   lane pattern doubles L1 wavefronts on the read and it never pays. Kept in the file
   (`L8_FORCE_LOADZ=1`) but excluded from the autotune space. Do not rediscover this.
2. **`__stcs` at B_HBM**: 1548–1551 µs vs 1532–1534 (−1%). Autotuner rejects it.
3. **`cp.async` staging**: B=131072 1542–1545 µs vs 1534 (−0.6%); L13 measured neutral,
   L17 +2.5% — at 8 KB volumes there is nothing to overlap. Left in the autotune space
   (it costs nothing and might matter at other batches).
4. **V = 4, 8 at the scored points**: HBM 1542/1545 vs 1534 (V=1/2); L2-with-stcs
   14.9/19.0 vs 13.6 (V=2). Bigger blocks mean coarser barriers (128–512 threads
   waiting instead of 64–128) with no coalescing gain — the staging span of V=1
   (8 KB contiguous per block) is already ≥ the 128 B granule by two orders.
5. **Staged fused kernel at B=1**: 3.41 µs vs 3.26 µs for the staging-free single
   kernel (min under ramping clocks; tryout's pinned-clock number is 4.26 µs). Kept
   the single kernel.

### Borrowed from

* `L13_dmma` (round gpu_r1): the whole fused-kernel skeleton — interleaved-coalesced
  staging (and its measurement that per-thread contiguous global z-lines lose), direct
  global write from the last pass, carveout hint, launch-bounds discipline, the
  1555 GB/s peak correction, and the warning that check.py/gen_input need the node.
* `L36_sharedtiled` (round gpu_r1): took `__stcs` seriously as a candidate.
* Literature 09 §6.2 (pad 8 → 9; conflict = gcd(S,8)), §9.2 structure 1 (thread↔line),
  §1.3 (volume-contiguous layout makes global coalescing free at L ≤ 17).
* CPU phase `../geom/strategies/L8_radix8.md`: the min-op radix-8 arithmetic (4 mults).

### What I would do next

1. **B_L2 residual (13.44 vs ≈11.6 µs write-stream floor)**: a warp-per-volume variant
   (32 threads, 2 lines/thread, `__syncwarp` only — no block barriers at all) attacks
   the remaining MIO/barrier mixture; capped at ~13% by the write stream, so measure
   cheaply before investing. V=2+stcs already sits at 1.16× of the floor.
2. **B=1 (4.26 µs, cuFFT 9.1)**: single kernel is 2 warps on one SM; a per-output
   dense split over 8+ warps, or a persistent-kernel/graph trick, might shave ~1 µs.
   Low value — the case is launch-dominated.
3. **B_HBM is done** (90.2% of DRAM peak at minimum bytes): do not spend another round
   on arithmetic, shared traffic, or hints there — every lever tried moved it ≤ 0 ±1%.
   The only thing that could move it is a fundamentally better DRAM schedule (e.g.
   read/write interleaving via persistent blocks), expected worth ≤ 5%.
4. The `L8_FORCE_*` env knobs skip the autotune for A/B work; autotune output is
   deterministic in results (all variants bit-identical), so scoring is unaffected.

---

## Round gpu_r2 (2026-08-22) — the structure was already right: five alternatives measured, all lost, ~1% net from a barrier demotion

Standing entering the round: led all three L=8 cells (3.18 µs / 13.24 µs / 1536.8 µs vs
rival L8_warpradix8's 3.63 / 14.02 / 1560.4). My r1 record listed three next steps; this
round built and measured all of them, plus two ideas from other entries' r2 records. The
honest headline: **every structural alternative lost**, and the round's value is that the
losses are now numbers rather than guesses.

### What changed in production

1. **z→y barrier demoted from `__syncthreads` to `__syncwarp`** in the fused kernel.
   Provably warp-local: warp w holds t ∈ [0,32) or [32,64) of ONE volume; the z-pass
   writes row t and the y-pass thread t reads rows 8·(t>>3)+0..7 — for t<32 those are
   rows 0..31 (same warp), for t≥32 rows 32..63 (same warp). Only staging→z (flat copy,
   any-to-any) and y→x (x spans both warps) need block barriers. Worth ~1% at B_L2,
   neutral at B_HBM; free and strictly correct, kept.
2. **B=1 pick hardcoded to the 64-thread single kernel** (was: implicit). A new
   32-thread single-warp variant (two lines/thread, zero block barriers) measured
   **3.82 vs 3.20 µs** — serializing two lines per thread lengthens the latency chain
   more than removing two barriers saves. Kept behind `L8_FORCE_SINGLEWARP`.
3. **Autotune space extended 16 → 32 candidates** (warp-per-volume and persistent
   grid-stride families added, both bit-identical in output to the fused kernel — same
   per-line codelet, so the measured pick can never change a bit). The tuner rejects
   both at every scored point; they stay as executable dead ends. `L8_DEBUG=1` now
   prints the plan pick.

### Measured — leased A100-SXM4 via tryout.sh, final file

| case | gpu_r1 | gpu_r2 | cuFFT | speedup | rel_l2 |
|---|---|---|---|---|---|
| B=1 | 3.18 µs | **3.20–3.35 µs** (launch-bound, sd 11–13%) | 9.4–12.3 µs | ~2.9× | 1.58e-16 |
| B=2048 (L2) | 13.24 µs | **13.13–13.24 µs** (autotune: fused V=1 + `__stcs`) | 54.9 µs | **4.2×** | 1.59e-16 |
| B=131072 (HBM, primary) | 1536.8 µs | **1537.3 µs** (fused V=1, plain; sd 0.01%) | 3700 µs | **2.41×** | 1.59e-16 |

Bit-identical across runs at every point; all forced variants (WPV tail warps at B=63,
GS at B=63, single-warp) PASS with identical error; `compute-sanitizer memcheck` 0 errors
on the new WPV and GS paths. Note the B_L2 autotune now picks **V=1** (r1 picked V=2;
they were within noise then and still are — 13.24 vs 13.63 this window).

### What was tried and did NOT work, with the number that killed it

1. **Warp-per-volume kernel** (my r1 next-step #1: one warp owns a volume, 2 lines per
   thread per pass, per-warp 512 B-contiguous staging, ZERO block barriers): B_L2
   **14.31 µs vs 13.32** best fused (+7%), B_HBM **1553–1566 vs 1537** (+1.3%). Shared
   per volume is unchanged (9216 B) so warps/SM halve (17 vs ~34); the occupancy loss
   costs more than every barrier in the kernel. Together with the rival's pure-register
   14.19 µs this closes the question from both sides: **at L=8 the B_L2 residual is NOT
   barrier or shared-traffic cost.** `L8_FORCE_WPV=n`.
2. **Warp-local staging** (warp stages its own half-volume so load→z is also
   `__syncwarp`; only one block barrier left): B_HBM **1555.9–1560.7 vs 1537.4** forced
   like-for-like (−1.3%), B_L2 neutral. The per-instruction sector shape is identical
   (512 B contiguous per warp either way) — what changed is the block-wide read ORDER,
   and at the DRAM wall the flat interleaved order wins. Reverted; the staging comment
   in the file records it. This is the same lesson as L13_dmma r2's fused-x failure,
   at 100× smaller magnitude: **do not perturb a global access pattern that is at 90%
   of DRAM peak.**
3. **Persistent grid-stride kernel** (my r1 next-step #3, "read/write interleaving via
   persistent blocks, expected ≤5%"): grid = one occupancy wave (1728 blocks at V=1),
   loop over volume groups. B_HBM **1623.6 vs 1537.4 (+5.6%)**, B_L2 **13.86 vs 13.24**
   (+5%). The hardware block scheduler's natural wave overlap beats a fixed persistent
   grid at both points; the expected ≤5% was real but with the opposite sign. The r1
   "B_HBM is done" verdict now has its last untried lever measured. `L8_FORCE_GS=n`.
4. **`__stwt` (write-through) instead of `__stcs` (evict-first)** on the final store at
   B_L2: **23.7 vs 13.24 µs** (+79%). Write-through defeats L2 write-coalescing — every
   16 B store becomes its own DRAM transaction. Evict-first is the right stream hint;
   write-through is a different (and here catastrophic) semantics. Not kept in the file.
5. **32-thread single-warp B=1 kernel**: 3.82 vs 3.20 µs (see above). `L8_FORCE_SINGLEWARP`.

### Borrowed from / cross-checked against other entries this round

* **L6_batchcoalesced r2** and **L8_warpradix8 r1**: their "B_L2 residual is
  latency-shaped" diagnosis motivated the WPV experiment; my measurement now puts a
  number on the barrier half of that hypothesis (it is not the barriers).
* **L13_dmma r2**: the practice of keeping dead ends behind `FORCE_*` env knobs with the
  numbers in the record, and the warning that env vars must be inside the `on_gpu.sh`
  ssh'd command string (saved me the same lost cycle).
* **L17_dmma r2** (software grid barrier at B=1): read and consciously NOT adopted — my
  B=1 is already a single 2-warp launch with nothing to join; their win requires a
  second launch to eliminate.
* **L36_sharedtiled r2**: their measured failure of cross-execute L2 residency at
  32 MiB working set closed my "keep both streams in L2 to beat the write floor" idea
  without spending lease time on it.

### Where this leaves L=8, and what I would do next

* **B_HBM (primary): closed.** 1537 µs = 1397 GB/s = 90% of the 1555 GB/s part peak at
  exact-minimum bytes, and the final untried structural lever (persistent blocks)
  measured −5.6%. Every entry on the board lands at 89–90%; this is the hardware answer.
* **B_L2: 13.13–13.24 µs vs the ≈11.6 µs pure-write-stream floor (1.14×).** Now measured
  NOT to be barriers (WPV), NOT wave quantisation (GS), NOT store semantics (__stwt),
  NOT staging shape (warp-local). What remains is the latency of the read+compute+write
  chain itself at ~34 warps/SM. The one unexplored idea: an XOR-swizzled unpadded shared
  layout (8192 B/volume instead of 9216, conflict-free by `addr mod 8 = z^y` argument)
  raises the shared-limited occupancy ceiling ~12%; achieved occupancy (39%) sits below
  the current theoretical (50%), so expect little — measure before believing.
* **B=1: launch-dominated at 3.2 µs.** Both split-work variants (mine, L13's) measured
  zero or negative; nothing to do under the per-call execute contract.
* The autotune's measured selection is doing real work across regimes (V=1+stcs at L2,
  V=1 plain at HBM) — keep every candidate bit-identical, per L8_warpradix8 r1's lesson.

---

## Round gpu_r4 (2026-08-22) — graph-replayed launch taken from the rival; the parity-split B=1 idea measured and rejected

(No r3 section exists above: the r3 sweep re-measured the unchanged r2 file. r3
leaderboard standing entering this round: B=1 **3.245 µs vs L8_warpradix8's 3.055 —
lost**, B_L2 13.165 vs 13.176 — statistical tie, B_HBM 1537.2 vs 1538.6 — tie at the
DRAM wall. The rival's r3 record says plainly where their B=1 win came from and ends
with "blockfused does not currently do it". This round takes it.)

### What changed in production

1. **Every execute now replays its single kernel launch through a lazily-captured
   CUDA graph**, keyed on the (in,out) pointers, recaptured if they change —
   **borrowed from L8_warpradix8 r3, who took it from L36_sharedtiled r1 / L45_pfa
   r2.** Every plan kind here is exactly one launch, so the graph is one node and the
   gain is pure launch path. Same kernel, same arguments → bit-identical output
   (cmp-verified at B=1 vs graph-off, and tryout's re-run check at 1/63/2048/131072).
   `L8_GRAPH=0` disables. Measured, same lease windows, graph off → on:
   * B=1: 3.18–3.25 → **2.99–3.10 µs**, and the sample spread collapses (sd 13% →
     0.01% in pinned windows) — the same noise-collapse the rival recorded.
   * B=2048: 13.36 → **12.71–12.97 µs** (−0.5 µs; plan pick unchanged, fused V=1+stcs).
   * B=131072: 1537.6 → **1536.4–1536.6 µs** (−1 µs ≈ noise, consistent sign). Enabled
     everywhere; the rival's "neutral-to-+0.6" caution did not reproduce as a regression.
2. **B=1 is now a measured create()-time pick** over three bit-identical kernels:
   the r2 64-thread single, and two new **output-parity-split** kernels (below). The
   pick landed on the 64-thread single in every window (3/3 runs + forced A/B); the
   split kernels stay in the pick space behind `L8_FORCE_SPLIT1={1,2}`.

### The round's designed experiment: output-parity-split B=1 kernels — LOST

Motivated by the rival's r3 quad win at B=1 (128 threads, ~quarter per-thread chain,
3.06 vs 3.19): rebuild the same idea in thread-per-line shared-memory form. `dft8`
splits cleanly by OUTPUT parity — even outputs k∈{0,4,2,6} need only the t-half
(sums, zero irrational twiddles), odd k∈{1,5,3,7} only the s-half (all 4 real mults) —
and the split expressions are verbatim subsets in the same operand order, so a pair
of halves is **bit-identical** to one full dft8 (cmp-verified against the shipped
kernel, both split kernels, graph on and off). `fft8_single_split` (128 threads:
pair (t, t+64) shares a line, each computes one half) and `fft8_single_split4`
(256 threads: quad shares a line, each computes one output pair). Both halves read
all 8 inputs (duplicate z-line global loads hit L1 on an 8 KB volume); passes
ping-pong between two shared buffers sa→sb so an in-place half never races its
cross-warp partner — same 2 barriers as fft8_single.

**Numbers that killed it (graph on, pinned 10-sample windows):** split128 **3.365 µs**,
split256 **4.248 µs**, vs the plain 64-thread single at **3.065 µs**. Monotonically
worse with more splitting, sd 0.00% on the splits (they are longer but perfectly
steady). Reading: at one 8 KB volume the critical path is global-load latency plus
two barriers, not per-thread arithmetic — halving the flop chain saves nothing while
the duplicated loads and wider barriers add. The rival's split won because their flat
load gives each lane DISTINCT data (4 complex/lane, no duplication); a shared-memory
line-split necessarily duplicates reads. The two structures are not equivalent at
B=1, and the register form is simply better there. Do not rediscover this.

### Operation count

Production arithmetic unchanged: min-op radix-8, 4 mults + 52 adds per line, 21
flop/point full 3D. The split kernels do redundant half-stage-1 work (each half
recomputes its 4-input combination), bit-identical by construction — moot since
they lost.

### Measured — leased A100-SXM4 via tryout.sh, final file

| case | r3 board | r4 (this file) | cuFFT | speedup | rel_l2 |
|---|---|---|---|---|---|
| B=1 | 3.245 µs | **3.000 µs** (repeats 2.99–3.10; sd 0.01%) | 9.68 µs | **3.2×** | 1.58e-16 |
| B=2048 (L2) | 13.165 µs | **12.97 µs** tryout, 12.71 repeat (fused V=1+stcs+graph) | 54.6 µs | **4.2×** | 1.59e-16 |
| B=131072 (HBM, primary) | 1537.2 µs | **1536.4 µs** (fused V=1 plain + graph; sd 0.03%) | 3694 µs | **2.40×** | 1.59e-16 |

PASS + bit-identical re-run at B = 1, 63 (tail path under graph), 2048, 131072;
`compute-sanitizer memcheck` 0 errors on the graph path and both split kernels;
all forced split × graph combinations PASS with identical output.

### Borrowed this round, with attribution

* **CUDA graph replay of the single launch: L8_warpradix8 r3** (their execute-side
  capture/recapture pattern copied nearly verbatim; originally L36_sharedtiled r1 /
  L45_pfa r2). Their record's closing line — "the single cheapest microsecond
  available to any entry whose execute is one launch" — was exactly right: it is
  worth 0.25 µs at B=1 and 0.5 µs at B_L2 here, for ~35 lines.
* The B=1 128/256-thread split idea: **L8_warpradix8 r3's quad result**, tested in
  shared-memory form and measured dead (above) — the technique does not transfer
  across the register/shared boundary.
* Env-in-the-ssh-command-string discipline for on_gpu.sh A/B runs: **L13_dmma r2**
  (hit it again this round: quoted `sh -c` strings are flattened by on_gpu.sh's `$*`).

### Where this leaves L=8, and what I would do next

* **B=1: 3.00 µs**, launch-path-bound with the graph; ahead of the rival's r3 3.055.
  What remains is the graph-launch interval plus one 2-warp kernel; nothing visible
  under the per-call contract.
* **B_L2: 12.7–13.0 µs vs the ≈11.6 µs write-stream floor (1.10×).** The graph took
  the launch slice; r2+r4 have now measured away barriers, occupancy (both
  directions), store semantics, staging shape, wave packing, and per-thread
  arithmetic. What's left is ramp/drain on a 13 µs one-wave kernel. The unmeasured
  XOR-swizzle unpadded-shared idea from r2 stays unmeasured: it only raises an
  occupancy ceiling that r2/r3 evidence (fused V=4/8 worse, rival quad worse at 40+
  warps/SM) says is not the binding constraint.
* **B_HBM: closed since r2** — 1536.4 µs = 90% of the 1555 GB/s part peak at exact
  minimum bytes; the graph is neutral there. Do not spend rounds on it.
* If a future round wants more: the only untried structural family at B_L2 is a
  multi-volume-per-thread kernel (2 volumes' lines per thread, halving blocks and
  barriers-per-volume at constant occupancy) — expected small, and the r2 WPV result
  argues against; measure for one lease window at most.
