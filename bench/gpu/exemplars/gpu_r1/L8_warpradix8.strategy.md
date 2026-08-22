# L8_warpradix8 — strategy record

Geometry: **L = 8**, cube 8³ = 512 complex doubles per volume (8,192 B), forward,
unnormalised, out-of-place, batched, one A100. Implementation: `impl/L8_warpradix8.cu`.
Scored cases (cases.txt): B = 1, 2048 (L2-resident), 131072 (HBM, primary).
Mandate: one volume per **warp**, registers + shuffles, no shared memory — the
register half of the register-vs-shared experiment against `L8_blockfused`
(lit. 09 §9.2 structure 2).

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique (as landed)

**One volume per warp, 16 complex doubles per lane (64 registers of data), one
global read + one global write, zero shared memory, zero barriers in the batched
path.** The production kernel (`fft8_warp_x`, "var 2") went through two designs in
this round; the second is the interesting one:

1. **Flat load**: `v[r] = in[vol*512 + r*32 + lane]` — 16 perfectly coalesced
   512-byte load instructions. The index algebra is a gift: with lane = 8h + w this
   puts **two complete x-lines in every lane's registers** (y ∈ {h, h+4}, z = w),
   so the first axis pass needs no data motion at all.
2. **Pass X**: two minimum-operation radix-8 DIF codelets in registers (4 real
   mults + 52 adds, natural order in and out) — **codelet arithmetic taken verbatim
   from L8_blockfused round gpu_r1**, which is the CPU phase's winning L8 line
   transform.
3. **Passes Y and Z as cross-lane DIF stages, not transposes.** y = 4a + h lives
   on one register bit and two lane bits; z = w on three lane bits. Each axis is
   done as radix-2 DIF stages directly across those bits: one local stage (both
   butterfly outputs stay in-lane) plus five `__shfl_xor_sync` stages. A cross-lane
   butterfly is 2 shuffles + 2 FMA (`pre = recv + s*mine`, s = ±1 by side) + one
   complex multiply by a **per-lane twiddle hoisted out of the stage** — the hot
   path contains no selects and no divergent branches.
4. **The DIF bit-reversal is absorbed into the store address.** DIF leaves ky/kz
   bit-reversed across lane bits. A warp's 32 store addresses are the same *set*
   regardless of which lane holds which element, so coalescing is
   permutation-invariant: the 16 store instructions stay 4×128 B sector-dense and
   the bit-reversal costs literally nothing.
5. `create()` runs a measured autotune (pattern **adopted from L8_blockfused
   round gpu_r1**) over warps/block {1,2,4,8,16} × {single, double-buffered pair} ×
   {normal, `__stcs`} on scratch buffers with the real batch. B = 1 uses a
   dedicated kernel (below).

### Operation count

Per lane per volume (÷32 for per-point): 16 ld + 16 st + 2×dft8 (112 FP) + local
y-stage (16 add + 8 cmul) + 5 cross-lane stages (160 shfl + 160 FMA + 3×64 cmul-FP)
≈ **~530 FP + 160 shuffle instructions**, ≈ 33 flop/point against the 200-flop/point
budget the 32 B/point HBM floor allows (lit. 09 §2.2: L=8 is 4.4× bandwidth-bound).
Arithmetic is free here; the design variable is global-access shape and latency.

### Measured (reserved-node SXM4 lease via tryout.sh; cuFFT same case in brackets)

| case | per-transform | bandwidth | vs cuFFT |
|---|---|---|---|
| B = 1 | **3.74–3.90 µs** [9.5 µs] | — | ~2.5× |
| B = 2048 (L2) | **6.93 ns** (14.19 µs/call) | 2365 GB/s [612] | 3.86× |
| B = 131072 (HBM, primary) | **11.89 ns** (1558.5 µs/call) | **1377.9 GB/s** [580] | 2.37× |

rel L2 error 2.3e-16 at every point (the 1 GiB case checked with check.py on the
reserved node — the login node cannot allocate it); bit-identical across runs;
compute-sanitizer memcheck clean; 106 registers, zero spills.

ncu at B_HBM: `dram__bytes` 2.13 GB = exactly the in+out minimum, **89.2% of
sustained DRAM peak**. Same hardware correction as L13_dmma's record: the SXM4-40GB
node is a **1555 GB/s** part (1215 MHz memory clock), not the brief's ~2 TB/s, so
this is VkFFT-class (their published radix band is 82–84% of peak).

### The head-to-head this pair exists to settle (my measurements, same session)

| case | L8_warpradix8 (registers/shuffles) | L8_blockfused (shared) |
|---|---|---|
| B = 1 | 3.74–3.90 µs | **3.48 µs** |
| B = 2048 | 14.19–14.26 µs | **13.45 µs** (median 17.2, sd 9.6%) |
| B = 131072 | 1558.5 µs, 1377.9 GB/s | **1537.6 µs, 1396.7 GB/s** |

**Verdict so far: on this part the two structures are within ~1.5% at the HBM
point and the shared-memory version holds a small edge everywhere.** Both move
minimum bytes at ~89% of DRAM peak; the residual differences are latency-shaped
(occupancy: their ~40-register threads allow 32 warps/SM against my 19 at 106
registers), not traffic-shaped. The register path's structural wins — no barriers,
no shared traffic, warp-granular tails — do not convert into time at L = 8 because
neither shared bandwidth nor barriers were the shared version's limiter.

### Tried and rejected, with the numbers that killed them

* **Butterfly shuffle transposes with per-lane line stores (var 0)** — the first
  design: 5 select-predicated exchange rounds (natural-order landing provable by
  induction on settled lane bits), then each lane stores its two contiguous
  128B-aligned z-lines. **2279 µs vs 1557 at B_HBM — 1.47× worse.** Per-lane-
  contiguous global access loses even when perfectly aligned; this is the store-
  side mirror of the load-side failures already recorded by L6_batchcoalesced and
  L13_dmma. Flat instruction-coalescing is the only shape that reaches the roof.
* **Same transposes + transpose-back for flat stores (var 1)**: correct and
  1556.7 µs at B_HBM — the extra 5 rounds are free under the bandwidth floor —
  but ~960 select instructions per lane per volume against var 2's zero, and var 2
  matches or beats it everywhere while being simpler. Superseded.
* **Double-buffered two-volumes-per-warp (var 3)**: load volume 2 while volume 1
  computes, 180 registers, no spills. **16.8 vs 14.1 µs at B_L2, 1566 vs 1550 µs at
  B_HBM** — the occupancy drop (19 → 11 warps/SM) costs more than the latency
  self-hiding buys. Kept in the tune space (bit-identical to var 2) but never wins.
* **Barrier-free B = 1 via redundant partner loads**: a DIF stage on z commutes
  with the x/y passes, so each thread can load both z-partners (2× reads on 8 KB —
  free) and apply the cross-warp butterfly at load time, eliminating the shared
  bounce entirely. Elegant, pure-warp — and **4.65 µs vs 3.90: the 8 extra load
  instructions on the critical path cost more than one barrier + an 8 KB bounce.**
* **Autotuning across arithmetic-different kernel families broke repeatability**:
  runs picked var 1 or var 2 nondeterministically and their outputs differ in the
  last ulp (different z-pass rounding), so tryout's bit-identity check failed at
  B = 131072. Fixed by restricting the tune space to {var 2, var 3} × WPB × STCS,
  which all share `l8_compute` exactly. **Lesson for any entry with a measured
  autotune: only tune over knobs that cannot change a single bit of the output.**
* **`__stcs` evict-first stores: autotuned per batch, and the sign flips.** At
  B_L2 it is a large win (23.5 → 14.8 µs — without it the repeat-loop working set
  thrashes); at B_HBM it is neutral-to-slightly-worse (1550 → 1569 µs). Contrast
  with L6/L13 who dropped streaming hints as noise — at L = 8 the L2 point
  genuinely needs it. Never hardcode it; measure at plan time with the real B.

### Borrowed, with attribution

* Min-op radix-8 DIF codelet arithmetic and the create()-time measured (V, stcs)
  autotune pattern: **L8_blockfused round gpu_r1**.
* Shared-staging idea for the B = 1 latency kernel (one 8 KB padded bounce, one
  barrier, two warps): **L8_blockfused's fft8_single**, restructured — my version
  keeps x/y/2-of-3-z-stages in registers and crosses warps only on the z2 bit.
* The 1555 GB/s hardware correction and "check the big case on the reserved node"
  practice: **L13_dmma round gpu_r1**.
* Confirmation that per-thread-contiguous global access loses to flat coalescing:
  **L6_batchcoalesced / L13_dmma round gpu_r1** (their load-side evidence; I
  measured the store-side counterpart before finding their records said so).

### What I would do next

1. **B_L2 (2048) is one wave and latency-bound**: ncu shows 12.5 active warps/SM,
   21 cycles per issued instruction, nothing saturated (L2 roof is 5× away). The
   only real lever left is occupancy: 16 complex/lane forces ~106 registers. A
   **half-volume-per-warp** layout (8/lane, ~60 regs, 2 warps cooperating through
   one shuffle... no — through one shared or global exchange) would double resident
   warps but abandons the pure-warp premise; the b1 kernel's structure generalised
   to the batch is the honest way to test it.
2. **B = 1 at 3.74–3.90 µs is ~85% launch path** (lit. 09 §6.6). A 4-warp version
   of fft8_b1 (radix-4 cross-warp gather, still one barrier) might shave ~0.2 µs of
   kernel; the launch itself is untouchable under the per-call execute contract.
3. If anyone wants the last ~1% at B_HBM: the gap to L8_blockfused (1377.9 vs
   1396.7 GB/s) tracks occupancy, same as (1). Both entries are at exact-minimum
   DRAM bytes, so nothing is left in access shape.
4. **For L64_radix8**: lit. 09 §9.8 says the L8 kernel is the L64 codelet. The
   cross-lane DIF machinery here (sign-FMA butterflies, hoisted per-lane twiddles,
   bit-reversal absorbed by addressing) transfers directly to a 64-point line
   split 2 or 4 lanes per line.
