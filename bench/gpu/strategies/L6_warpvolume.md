# strategies/L6_warpvolume.md — L = 6 (6^3 = 216 complex doubles) on one A100

Implementation: `impl/L6_warpvolume.cu`   ·   `fft3d_gpu_name()` = `L6_warpvolume`

---

## Round gpu_r1 (first implementation; dev loop = tryout.sh on a leased A100-SXM4-40GB)

### Technique ended on

**Whole volume in warp registers, zero shared memory, one kernel.** 8 lanes per volume,
4 volumes per warp, no idle lanes. Lane `q = (ox,oy,oz)` owns the 27 points with
`x ≡ ox, y ≡ oy, z ≡ oz (mod 2)` — a 3×3×3 parity sub-lattice, 27 complex doubles =
108 registers of data (148 regs/thread total, no spills). Each axis is one radix-2×3
stage split across the lane pair that differs in that axis's parity bit: a local Winograd
DFT-3, a `w6^m` twiddle on the odd lane, and a cross-lane DFT-2 butterfly done with
2 × `__shfl_xor_sync` (8 B each) per complex. No `__syncthreads`, no bank conflicts,
no barriers. One coalesced global read, three all-register passes, one global write.

The **z axis runs DIF** (input halves `{0,1,2}/{3,4,5}`, output parity-interleaved) and
**y, x run DIT** (input parity-interleaved, output halves). That asymmetry is deliberate
and worth ~10% on its own: z is the only axis where 16-byte elements share 32-byte
sectors, and DIF-z makes the two z-parity lanes **write** the two halves of the same
32-byte sector in the same store instruction (perfect full-sector stores), while the
loads become 48-byte per-lane contiguous runs that the L1 heals across adjacent
instructions. Making the *loads* the perfect side instead (DIT-z) measured slower —
nothing heals store wavefronts before L2. All load/store offsets are compile-time
immediates off one base pointer.

Stores are `__stcs` (evict-first) unconditionally: `out` is never re-read, and keeping
the write stream out of L2 protects `in`'s residency at the L2-resident batch point
(28.0 → 18.6 µs, the single biggest win of the round). Loads additionally get `__ldcs`
only when `create()` sees a working set over 40 MiB (template parameter; neutral at
B_HBM, protects the L2 point). Block size 32 (one warp): with 148 regs/thread the
register file, not threads, limits residency, and small blocks avoid losing warps to
block granularity (256-thread blocks → 1 block/SM = 12.5% occupancy; 32/64-thread
blocks → 12–13 warps = ~19%).

### Operation count

Per lane per axis: 9 × (Winograd DFT-3 ≈ 18 flop + 2 twiddle cmul = 12 flop + 3 complex
butterfly adds = 6 flop) = 324 flop, and 54 shuffle instructions. Whole transform:
36 flop/point (vs the Good–Thomas optimum of 24 — the extra 12 are the DIT/DIF twiddles,
invisible at 5.15× bandwidth-bound; ncu shows 30% compute utilization) and 162 shuffles
per lane per 4 volumes = 6 shuffle instructions per point. Memory: exactly 16 B read +
16 B written per point.

### Measured (tryout.sh leased SXM4, sd ≤ 0.5% at batched points; scored points from cases.txt)

| point | this kernel | per transform | eff. BW | cuFFT same case | ratio |
|---|---|---|---|---|---|
| B=1 | **5.29 µs** (median 5.3–5.5, sd ~10%) | 5.29 µs | — | 10.6–11.7 µs | **2.0–2.2×** |
| B_L2 = 4854 | **18.59 µs** min | 3.83 ns | 1805 GB/s | 51.8 µs | **2.79×** |
| B_HBM = 310608 | **1563.9 µs** min (median 1565.4, sd 0.04%) | 5.03 ns | 1373 GB/s | 3524 µs | **2.25×** |

rel L2 error 2.1–2.6e-16 at every batch tested (1, 2, 3, 5, 7, 33, 63, 64, 100, 4854,
310608); bit-identical across runs; compute-sanitizer memcheck clean. Final ncu at
B_HBM: DRAM 89% of SoL, L2 74%, compute 30%, achieved occupancy 16.8%.

### What did not work, with the number that killed it

* **Good–Thomas per axis (tried first, borrowed from the CPU winner):** the CRT input
  map `j = (3j1+2j2) mod 6` hands each lane the z-order `{0,2,4}/{3,5,1}`, so every
  warp load touches 32 distinct 32-byte sectors and uses half of each. ncu: L1 healing
  at 67% hit rate but 2× wavefronts. **1117 GB/s** at B_HBM vs 1373 final. The
  twiddle-free property is worth nothing here (arithmetic is free); the sector geometry
  of the index map is everything. On a GPU, *choose the factorization variant (DIT vs
  DIF vs PFA) per axis by its memory pattern, not its flop count.*
* **256-thread blocks:** 1 block/SM by register granularity = 12.5% occupancy,
  **1922 µs** at B_HBM. Same kernel at 64-thread blocks: 1737 µs.
* **Forcing 128 regs with `__launch_bounds__(64,8)`** to reach 16 warps/SM: 40 B/thread
  of spill, **1970 µs** — worse than the occupancy it bought. Keep the natural 148.
* **All-DIT (perfect-sector loads, half-sector stores):** 1709 µs vs DIF-z's 1554 µs.
  Loads are L1-healed for free; stores are not.
* **`__ldcs`+`__stcs` on everything at B_HBM:** neutral (1565 vs 1567 µs median) —
  kept only where it has a reason (stores always, loads above the L2 threshold).

### Borrowed

* Winograd DFT-3 module and the 2×3 structure: `../geom/strategies/L6_unrolled.md`
  (CPU round 1) and rival `L6_batchcoalesced` (same module).
* Register-resident warp-volume structure: literature 09 §9.1 structure 1, §6.4, §6.5.
* The "last pass writes straight to global" idea from `L13_dmma` via `L6_batchcoalesced`
  is subsumed here: *every* pass is register-to-register, only the last writes out.

### What I would do next

1. **Triad decomposition** (27 active lanes/volume, 2×2×2 = 8 points/lane, radix-3
   cross-lane × radix-2 local): ~64 regs/thread → ~2.5× the occupancy, and 4× the warps
   per batch at B_L2 (which is latency/writeback-limited, not bandwidth-limited). Costs
   12 shuffle instr/point (vs 6) and idles 5/32 lanes. Worth one experiment: B_HBM is
   at 89% DRAM so the ceiling there is ~10%, but B_L2 has ~1.4× of headroom left
   (writeback floor ≈ 13 µs vs 18.6 measured).
2. B=1 is pure launch path (~5.3 µs for 4.4 ns of work); if the harness ever allows a
   CUDA graph or a pre-recorded launch in create(), that is where the 2× lives.
3. If a future round adds an inverse/normalized variant, the DIF-z/DIT-yx split and the
   parity maps transpose cleanly; keep the store side the full-sector one.

---

## Round gpu_r2 (2026-08-22) — the register-resident design retired on measurement

### Standing entering the round

r1 leaderboard: won B_L2 (18.38 vs 23.88), lost B=1 (5.37 vs 3.64), near-tie at B_HBM
(1548.2 vs 1540.1). But rival `L6_batchcoalesced`'s round-gpu_r2 record shows they had
already borrowed my `__stcs` trick and taken B_L2 to 14.8–15.5 µs — entering this round
I was behind at **all three points**.

### Technique ended on

**The r1 warp-register kernel is gone from the hot path.** This file now ships:

* **batched (all B > 1): `fft6_bstage`** — shared-staged batch-major kernel, structure
  **borrowed outright from `L6_batchcoalesced` round gpu_r1** (8 consecutive volumes
  per 288-thread block, batch-major swizzled shared `slot = i*8 + ((v+i)&7)`, flat
  coalesced load, z/y passes thread=(volume fast, line slow) so shared access is
  bank-conflict-free at any stride, x-pass thread=(line fast, volume slow) storing
  straight to global). My own DIT-6 = 2×3 codelet (two Winograd DFT-3s + w6 twiddles).
  40 registers, zero spills, 5 blocks/SM = 45 warps/SM. Their occupancy argument is
  simply correct at a 5.15× bandwidth-bound size, and my measurements below say the
  register-resident design cannot match it at any scored point.
* **stores: `__stcs` exactly while the *input* buffer ≤ 36 MiB** (predicate borrowed
  from their r2 record), plain stores above (at B_HBM `__stcs` measured 1543.6 vs
  plain **1540.8** — same sign they found; at B_L2 stcs is the 18.6 → 15.1 win).
* **B = 1: `fft6_single`** — one 64-thread block, 36 line-threads per pass, padded
  shared `slot = i + i/6`, 3 barriers of 2 warps, x-pass stores direct. Same idea as
  `L8_blockfused`'s `fft8_single`. **3.09–3.16 µs** vs 3.64 for the 288-thread kernel
  at B=1 and 5.29 for my r1 kernel: at B=1 the whole GPU is one block, so what matters
  is minimum serial depth per thread (one 6-point line per pass) and cheap barriers —
  my r1 kernel's 8 working lanes × 27 points of serial chain was ~1.9 µs of kernel on
  a ~3.4 µs launch floor.

### Measured (tryout.sh leased SXM4; min over runs, sd on the batched points ≤ 1%)

| case | r1 (this entry) | r2 | rival r2 | cuFFT |
|---|---|---|---|---|
| B = 1 | 5.29 µs | **3.09 µs** | 3.78 µs | 10.2 µs |
| B = 4854 (L2) | 18.59 µs | **15.08 µs** (2,224 GB/s eff) | 14.8–15.5 µs | 51.7 µs |
| B = 310608 (HBM) | 1563.9 µs | **1540.4 µs** (1393.8 GB/s) | 1540.4 µs | 3528 µs |

rel L2 error 2.4–2.5e-16 at B = 1, 3, 13, 100, 4854, 10925 (tail + stcs boundary),
310608; bit-identical across runs; compute-sanitizer memcheck 0 errors.

### Tried and rejected, with the numbers that killed them

* **Triad decomposition** (my r1 "next" idea #1: 27 lanes/volume, 2×2×2 points/lane,
  cross-lane DFT-3 via computed-source shuffles with the CT twiddles folded into
  per-lane coefficients, DIF-z for the least-bad ternary store pattern). Built,
  verified (2.4e-16), 96 regs no spills, 21 warps/SM. **B_L2 23.7 µs vs 18.6 —
  compute-bound: ncu 62% SM throughput.** One-output-per-lane DFT-3 costs 3 generic
  cmuls per output ≈ 2× Winograd's arithmetic, ×32/27 idle-lane tax = 72 flop/point
  against warpvolume's 36 and bstage's ~31. At B=1 it gave 4.23 µs (better than r1's
  5.29, worse than fft6_single's 3.09). The lesson the hard way: *occupancy bought
  with more arithmetic is a loss at a bandwidth-bound size; occupancy must come from
  thinner threads, not wider distribution of the same math.*
* **L2-persistence windows (`cudaAccessPolicyWindow`), 4 configurations at B = 4854.**
  The idea: in+out = 32 MiB fits the 40 MB L2, so persist buffers and stop paying the
  16.8 MiB/call DRAM writeback (~10.8 µs floor) entirely. All lost to plain `__stcs`
  (15.08): window-on-out + plain stores **23.4 µs** (reads pushed to DRAM — read
  latency is what the kernel actually feels); exact-fit 16.8 MB carve thrashes on
  associativity (out written back every call; ncu dram__bytes 16.8 MB); with 21 MB
  carve, writes drop to 2–4 MB but reads lose residency (**19.9 µs** window-on-in +
  plain stores; the 19 MB normal partition can't also hold out, and plain stores
  write-allocate against in); window-on-in + stcs stores **15.18 µs** = stcs within
  noise, because stcs already keeps in resident without reserving anything. Net: at
  this working set the 40 MB pie cannot be partitioned into 33.6 MB of protected
  residency; `__stcs` alone is the whole win. Removed from the shipped code.
* **`cp.async` (`__pipeline_memcpy_async`) load staging at B_HBM**: 1610.6 vs 1540.8
  µs, **4.5% worse**. The register/L1 round trip it removes was not costing anything;
  the direct global→shared path evidently serializes worse at 5 blocks/SM.
* **r1 warpvolume at B_HBM, same-session A/B**: 1549.8 vs bstage's 1540.8. The 0.6%
  is real and repeatable (sd 0.1%); perfect store sectors do not beat +32 warps/SM of
  latency hiding even at the DRAM-bound point.

### Borrowed, with attribution

* Whole batched-kernel structure, V=8/288-thread choice, and the store-policy
  predicate (`__stcs` while input ≤ 36 MiB, plain above): **L6_batchcoalesced rounds
  gpu_r1 and gpu_r2**. Their records also saved me re-testing V=4 (worse both rounds)
  and load-fused-z (1582 µs at B_HBM).
* B=1 single-volume shared kernel shape: **L8_blockfused's `fft8_single`** (via
  L8_warpradix8's record).
* "Only choose between kernels deterministically, never by measured autotune across
  arithmetic-different kernels": **L8_warpradix8 r1** repeatability bug.

### What I would do next

1. **B_L2 (15.08) still sits ~4 µs above the 10.8 µs writeback floor** and ncu shows
   nothing saturated; the remaining structure is the 3-barrier chain at 1.12 waves.
   The one untried idea with real headroom: narrow the z→y barrier to 96-thread
   named-barrier groups (`bar.sync id, 96` — the z→y data dependence is closed within
   48 threads, 96 is the warp-aligned cover), and check whether the y→x full barrier
   can be split per-volume-pair. Expected small; measure before believing.
2. B = 1 at 3.09 µs is ~90% launch path now; nothing left under the per-call execute
   contract.
3. B_HBM 1540 µs = 89–90% of the part's sustained 1555 GB/s with exact-minimum bytes;
   every entry lands there. I re-confirm rival's r2 conclusion: treat it as the
   hardware answer for this geometry.
4. The triad kernel's folded-coefficient cross-lane DFT-3 machinery (computed-source
   shuffles, twiddles folded receiver-side into per-lane constants) is sound and
   verified — it is the wrong tool at L=6 where arithmetic must stay minimal, but a
   prime-size entry distributing a Rader convolution across lanes could reuse it.

---

## Round gpu_r3 (2026-08-22) — barrier-narrowing round: six variants measured, all lost, shipped config unchanged; the "win" that inverted under rotated-order A/B

### Standing entering the round

r2 leaderboard: won B=1 (2.969 vs rival L6_batchcoalesced's 3.693 µs), lost B_L2 by 2.3%
(14.960 vs 14.629), dead tie at B_HBM (1540.6 vs 1540.2). Both entries now ship the same
structure (they wrote it, I adopted it in r2), so the only levers left were structural.
The round's one target was B_L2, where 14.9 sits ~1.3× above the ≈11.6 µs DRAM
write-stream floor and L8_blockfused reaches 13.2 µs at the identical byte count.

### What was built and measured

The r2 kernel's remaining structure is the 3× 288-thread `__syncthreads` chain. I proved
the load→z and z→y dependences close inside a 96-thread x-slab group (group g owns
planes x = 2g, 2g+1 of all 8 volumes: the z-pass line l=(x,y) and y-pass line l=(x,z)
share the l/6 = x grouping, and a group-local load — 8 contiguous 1152-byte runs per
group instead of one flat 27.6 KB block copy — makes staging group-local too). Those two
barriers become named `bar.sync id,96` over 3 warps; only y→x (x-pass lines cross all
six slabs) needs the full block. All variants are bit-identical in output (same dft6 on
same data) and live behind an `L6_MODE` env knob (dev only, unset in scored runs):
flat/group/fused-z load × narrow/full z→y × stcs/plain stores.

**First A/B session said the narrow+group variant won: 14.89–14.97 vs baseline
15.14–15.26 at B=4854, consistent over 3 reps each. It was an artefact.** That session
ran modes in a fixed order (baseline always first) inside one lease; the SXM4's boost
clock ramps across the first processes of a lease, so the first-run mode measures
~0.25 µs slow — exactly the "cliff" PANEL_BRIEF warns about, showing up at process
granularity inside a lease. Re-measured with the order **rotated** across reps and
across 4 separate leases (10+ independent processes per mode, all on the same physical
GPU): baseline **14.92–14.96 µs**, narrow+group **15.14–15.42**, in every rotation
position. The barrier chain is NOT the B_L2 residual — the same conclusion
L8_blockfused r2 reached by deleting barriers (their warp-per-volume test), now
confirmed at L=6 from the named-barrier side.

### Measured (tryout.sh leased SXM4; shipped config = r2 config)

| case | r2 scored | r3 tryout (min over final-lease reps) | note |
|---|---|---|---|
| B = 1 | 2.969 µs | **3.09–3.13 µs** (launch path, sd ~8%) | unchanged kernel |
| B = 4854 (L2) | 14.960 µs | **14.92–15.11 µs** | unchanged kernel |
| B = 310608 (HBM) | 1540.6 µs | **1540.4 µs** (1393.7 GB/s) | unchanged kernel |

rel L2 error 2.4–2.5e-16 at B = 1, 13, 100, 4854, 10925 (stcs boundary + tail), 310608
(numpy check run on the reserved node — the login node cannot allocate 1 GiB);
bit-identical across runs at every point; compute-sanitizer memcheck 0 errors on the
batched-tail and B=1 paths.

### Tried and rejected, with the numbers that killed them

* **96-thread named barriers + group-local load at B_L2** (mode 2): 15.14–15.42 vs
  14.92–14.96 rotated-order baseline. Each half alone is also worse: narrow z→y only
  15.33–15.37; group load with full z→y 15.52–15.68.
* **Fused-z at B_L2** (z-pass thread reads its own 96-byte contiguous global line, no
  staging phase, one barrier fewer, 1/3 less shared traffic): **18.47–18.60 vs 14.93 µs,
  −24%**. Rival measured the same structure −2.5% at B_HBM in r1; from L2 it is far
  worse, not better — a flat block-wide coalesced load beats per-thread 96-byte runs
  even when every sector is fully consumed and the data comes from L2. The read
  *pattern*, not read *bytes*, is what matters at both batch points.
* **Anything at B_HBM**: narrow z→y 1542.4–1543.2, group-local load 1550.3–1550.5, vs
  flat/full-barrier **1539.8–1540.6**. Third round in a row confirming: do not perturb
  a kernel at 90% of DRAM peak.
* **Register effects worth recording**: the `bar.sync` asm (volatile + memory clobber)
  bloats the kernel 40 → 50–53 registers, silently cutting 5 blocks/SM to 4 (45 → 36
  warps); the 53-reg and reg-capped-40 versions measure the same at B_L2, so the
  occupancy loss is not why the variants lose. Forcing `__launch_bounds__(288,5)` on
  everything puts a 4-byte spill into the *flat* kernels that were naturally at 40 —
  fixed by giving capped and uncapped launch shells to one shared `__forceinline__`
  body. Rewriting the group-load indexing from per-k division to an incremental
  (v,off) carry cut 53 → 40 regs but measured ~0.15 µs *slower* (the carried index
  serializes the 6 load-address computations; the divisions are independent and ILP
  wins over registers here).

### Borrowed, with attribution

* Env knob inside the on_gpu.sh ssh'd command string (`L6_MODE=n` in `$*`, not exported
  locally): **L13_dmma r2 / L8_blockfused r2** records.
* Keeping dead-end variants compiled behind a dev knob with numbers in the record:
  **L13_dmma r2** practice.
* The rotated-order/same-lease protocol is the GPU version of lit 10's "interleaved
  best-of-N is the only trustworthy timing protocol"; adopted after the fixed-order
  artefact above.
* CUDA-graph launch for B=1 was considered and consciously NOT adopted: L36/L45 use
  graphs to collapse *many* launches into one; my B=1 is already a single launch, so
  there is nothing to collapse.

### Where this leaves L=6, and what I would do next

1. **B_L2 is now negatively closed from every direction anyone has measured**: not
   barriers (this round, both narrowing and — via L8 — deletion), not wave quantization
   (rival's V=4, L8's grid-stride), not store semantics (r2), not staging shape (this
   round + L8 r2), not L2 persistence windows (r2), not occupancy (53-reg vs 40-reg
   tie). What remains is the intrinsic latency of read→3-pass→write at this block
   granularity. The only untried shape is a fundamentally smaller block (V=1–2
   volume-major, 36–72 threads, PSLOT padding — L8's winning shape scaled down), which
   costs the batch-major bank-conflict-freedom; I did not build it this round and rate
   it the only remaining candidate.
2. **B=1 (3.0–3.1 µs) and B_HBM (1540 µs = 89–90% of the 1555 GB/s part peak, minimum
   bytes)** are launch-floor and hardware-floor respectively. Leave them alone.
3. Methodology note for whoever measures next: within-lease fixed-order A/B on this
   cluster carries a ~0.25 µs first-process penalty at B_L2 scale. Rotate the order or
   discard the first process per lease.
