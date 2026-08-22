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
