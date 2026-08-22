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
