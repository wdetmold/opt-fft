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

---

## Round gpu_r2 (2026-08-22) — streaming stores at the L2 point

### Standing after r1, and what this round targeted

r1 leaderboard: won B = 1 (3.64 vs rival L6_warpvolume's 5.37 µs) and B_HBM (1540.1 vs
1548.2 µs), **lost B_L2 by 1.30×** (23.88 vs their 18.38 µs). My own r1 record had
already diagnosed B_L2 as latency-bound and proposed warp-autonomy; the rival's record
showed the real cause was simpler and cheaper to fix.

### The change (one idea, borrowed)

**`__stcs` evict-first stores on the x-pass global writes, selected at plan time.**
Taken directly from **L6_warpvolume round gpu_r1** ("stores are `__stcs` unconditionally
... 28.0 → 18.6 µs, the single biggest win of the round") and independently confirmed by
**L8_warpradix8 round gpu_r1** (23.5 → 14.8 µs at their B_L2, *and* the observation that
the sign flips at B_HBM, so never hardcode it). Mechanism: at the L2-resident batch
point the driver's repeat loop re-reads the same `in` buffer every call; normal stores
allocate `out` in L2 and evict `in`, so every call re-fetches 16 MiB from DRAM.
Evict-first stores keep the write stream out of L2 and `in` stays resident.

My r1 experiment had missed this because I gated streaming hints on *working set >
L2* — exactly backwards for stores. The right predicate is **input buffer small enough
to stay resident** (`B·3456 ≤ 36 MiB` against the 40 MB L2): protect residency when it
exists, plain stores when it does not (at B_HBM `__stcs` measured ~0.7% slower:
1554.9 vs 1540.4 µs). Implemented as a `template <bool STREAM_ST>` on the kernel chosen
in `create()` — both instantiations execute identical arithmetic, so the choice cannot
change a bit of the output (repeatability lesson from L8_warpradix8's autotune bug).

### Measured (reserved-node SXM4 lease via tryout.sh)

| case | r1 | r2 | note |
|---|---|---|---|
| B = 1 | 3.64 µs | **3.78 µs** (median 3.79, sd ~12%) | launch path, unchanged within noise |
| B = 4854 (L2) | 23.88 µs | **14.8–15.5 µs min** (2,256 GB/s eff.) | **1.62×**, was the round's target |
| B = 310608 (HBM) | 1540.1 µs | **1540.4 µs**, 1393.7 GB/s | unchanged, `__stcs` correctly disabled |

rel L2 error 2.5e-16 at every point; PASS and bit-identical repeatability at B = 1,
4854, 10925 (tail block of 5 on the non-streaming path), 310608. Effective bandwidth at
B_L2 is now 2.26 TB/s — above DRAM peak, i.e. the reads are actually coming from L2,
which is the whole point. Remaining distance to the writeback floor (16.8 MiB of
mandatory DRAM writes ≈ 11 µs, plus L2 read time) is ~12% and latency-shaped.

### Tried and rejected this round

* **VPB = 4 retest at B_L2 with `__stcs`** (wave quantisation: 1214 blocks/11 per SM =
  1.02 waves vs VPB=8's 1.12): 16.55 vs 14.77 µs — still worse. r1's conclusion stands
  under the new store regime; block granularity is not the L2-point limiter.
* **Considered and declined without measuring**: fusing z+y per thread to drop a barrier
  needs a 36-point (144-register) plane per thread → 1–2 blocks/SM or a 48-thread block
  at ~12% occupancy — that is converging on the rival's all-register design, which
  measures *slower* than where this kernel now is (18.6 vs 14.8 µs). Their record's own
  triad-decomposition idea is the same territory.

### What I would do next

* The ~12% left at B_L2 is the 3-barrier dependency chain at 5.6 waves; the only
  untested idea that does not sacrifice occupancy is splitting the block into
  independent named-barrier groups (`bar.sync` with barrier id per 4-line group) so the
  z→y handoff synchronises 96 threads instead of 288. Small expected value.
* B = 1 remains pure launch path; nothing to do under the per-call execute contract.
* B_HBM is at 89–90% of the part's sustained DRAM peak with exact-minimum bytes;
  every entry that measured it (this one, L6_warpvolume, L8_*, L13_dmma) lands at the
  same ceiling. Treat 1540 µs as the hardware answer at this geometry.
