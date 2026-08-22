# L6_batchcoalesced — strategy record

Geometry: **L = 6**, cube 6³ = 216 complex doubles per volume (3,456 B), forward,
unnormalised, out-of-place, batched, one A100. Implementation: `impl/L6_batchcoalesced.cu`.
Scored cases (cases.txt): B = 1, 4854 (L2-resident), 310608 (HBM, primary).

---

## Round gpu_r1 (2026-08-22) — first implementation

### The one correction to the brief's premise

The stub's premise — "thread t handles volume t, so every load is a fully coalesced
transaction across the batch" — is wrong for the contract's layout. Volumes are
batch-OUTERMOST (`element (b,x,y,z) at ((b*6+x)*6+y)*6+z`), so at fixed point index,
consecutive volumes are 3,456 B apart: a volume-per-thread mapping is the *worst*
possible global access pattern, and a whole volume per thread is 864 registers anyway
(impossible; the 255-register ceiling gives 63 complex doubles). The honest form of
"batch-major" is: **global accesses follow the contract's contiguous layout; the batch
becomes the fast dimension only inside shared memory**, where the transpose costs bank
cycles instead of DRAM transactions (literature 09 §1.3: for L ≤ 17 the volume is
contiguous, so the 3D transpose is a shared-memory problem, full stop).

### Technique (as landed)

Fused single kernel, **8 consecutive volumes per 288-thread block**, staged through a
**batch-major swizzled shared array** `s[point][volume]`, slot
`i*8 + ((v+i) & 7)`, static 27,648 B:

1. **Load**: consecutive threads read consecutive global elements of the block's
   contiguous 8×216-element chunk — perfectly coalesced; fully unrolled to 6
   independent 16-byte loads per thread when the block is full. The batch-major
   shared write is a 4-way bank conflict (the additive swizzle beats the unpadded
   layout's 8-way), paid once.
2. **z-pass (stride 1) and y-pass (stride 6)**: thread = (volume v fast, line l slow).
   Because the 8 lanes of a line-group differ only in v, every shared access is
   8 consecutive complex doubles inside one **aligned 128-byte block** — bank-conflict-
   free for *any* axis stride. That is the entire point of batch-major: no axis is
   strided in a bank sense, no padding needed, and there is no transpose step at all.
3. **x-pass (stride 36)**: thread = (line l fast, volume v slow), so the six output
   stores go **directly to global, coalesced** (for a fixed output plane, consecutive
   threads write consecutive complex doubles). Skips the entire store phase and one
   barrier. Borrowed from **L13_dmma's round-gpu_r1 record** ("the x-pass writes its
   results directly to global"), which is also lit. 09 §9.1's structure.

Line codelet: DIT 6 = 2×3 — two DFT-3s + two twiddle rotations, all constants
compiled in, no twiddle table. Operation count is irrelevant here and was never
measured against alternatives: L = 6 is 5.15× bandwidth-bound (lit. 09 §2.2), and ncu
confirms 18% issue utilisation. Three `__syncthreads()` total. 40 registers, zero
spills, 5 blocks/SM (register-capped; needs the max shared carveout attribute set in
`create()`, otherwise the 100 KB default carveout caps at 3 blocks/SM).

`fft3d_gpu_execute` is **one kernel launch** for any B (grid = ⌈B/8⌉, tail block
guarded), which is what the B = 1 score measures.

### Measured (reserved-node SXM4 lease via tryout.sh, cuFFT same case in brackets)

| case | per-transform | bandwidth | vs cuFFT |
|---|---|---|---|
| B = 1 | **3.64 µs** [10.44 µs] | — | 2.87× |
| B = 4854 (L2) | **4.93 ns** (23.92 µs/call) | 1403 GB/s [649] | 2.16× |
| B = 310608 (HBM, primary) | **4.96 ns** (1540.5 µs/call) | **1394 GB/s** [608] | 2.29× |

rel L2 error 2.5e-16 at every batch point; repeatable (bit-identical across runs);
compute-sanitizer memcheck clean.

**Where that sits**: ncu on the HBM case reports `dram__bytes` = 2.13 GB against the
2.147 GB in+out floor (zero wasted traffic) and **89% of sustained DRAM peak**. Note
the scored node's HBM peak is ~1,555 GB/s (1215 MHz memory clock, confirmed by ncu's
pct-of-peak against the driver's GB/s), *not* the ~2 TB/s the brief claims for the
SXM4 — so 1394 GB/s is already VkFFT-class (their published A100 radix numbers are
~82–84% of peak) and the remaining headroom at this batch is ≤ ~6%, latency-shaped,
not traffic-shaped.

### Tried and rejected, with the numbers that killed them

* **`__launch_bounds__(288, 6)` to force 6 blocks/SM**: ptxas obeys by cutting to 32
  registers with 60 B of spills — 1799 µs at B_HBM, **17% worse**. Spill traffic hits
  DRAM exactly where there is no slack. 5 blocks at 40 regs is the right point.
* **V = 4 (144-thread blocks, better wave quantisation)**: 1543.8 µs at B_HBM (wash),
  **25.2 µs vs 23.9 at B_L2** — worse despite 11 blocks/SM, so the L2-point limiter is
  not block granularity.
* **Streaming hints `__ldcs`/`__stcs`, selected at plan time when working set > L2**:
  1545.0 vs 1543.9 µs — pure noise on A100; removed rather than carry dead complexity.
* **Fusing the z-pass into the load** (each thread reads its own z-line as six
  16-byte loads from a 96-byte run — sector pairs still fully covered — transforms in
  registers, writes shared once; swizzle `(v + i + i/6) & 7` keeps every phase ≤ 4-way):
  saves one barrier and a full shared round trip, but **1582 µs, 2.5% worse**. The
  flat perfectly-coalesced load phase is worth more than the saved barrier; DRAM
  sector-reuse-through-L1 is evidently not free. This closes lit. 09 §9.1's structure-1
  question for the staged variant: keep the load phase dumb and flat.

### What I would do next

* **B_L2 point is latency-bound, not bandwidth-bound**: ncu at B = 4854 shows nothing
  saturated (DRAM 46%, L2 sectors 29%, SM 26%, warps active 57%) — 607 blocks is 5.6
  waves of the 5-blocks/SM residency and the 3-barrier dependency chain shows. The L2
  roof (7.2 TB/s) is 5× away. Ideas in order: warp-autonomous processing (one warp
  owns its volumes end-to-end, `__syncwarp()` only — kills all block barriers, but the
  x-pass's direct coalesced store needs rethinking at warp width vs 36 lines), or
  splitting the block into independent barrier groups with named `bar.sync`.
* **B = 1 is a single ~3.6 µs launch** — that is the launch path, as lit. 09 §6.6
  predicts (4.4 ns of work). Only a persistent kernel or graph upload could move it,
  and the driver's execute-per-call shape gives no room for either. Not worth a round.
* If someone wants the last ~5% at B_HBM: the load-phase 4-way conflict is the only
  non-clean access left; a register-staged two-step transpose (coalesced global →
  registers → conflict-free shared via a second thread mapping) might shave it, but
  the fused-z failure above says be skeptical of anything that perturbs the load.
