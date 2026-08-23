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

---

## Round gpu_r2 (2026-08-22) — half the registers: one volume per TWO warps

### Standing entering the round, and the diagnosis

r1 leaderboard: lost all three cells to L8_blockfused — B=1 3.632 vs 3.184 µs,
B_L2 14.024 vs 13.238 µs, B_HBM 1560.4 vs 1536.8 µs. My own r1 record had already
diagnosed the cause: both entries move exact-minimum DRAM bytes, and the residual
is occupancy — var 2's 16 complex/lane costs 106 registers = 19 warps/SM against
their 40-register/32-warp threads. So this round executed r1's item (1): keep the
warp-register structure but halve the per-lane payload.

### What changed (production)

**var 4, `fft8_pair`: one volume per two warps, 8 complex doubles per lane,
66 registers, zero spills → 28–30 warps/SM.** Thread u = t&63 owns one full
x-line via the flat coalesced load `v[r] = in[vol*512 + r*64 + u]` (r = x, so
pass X needs no data motion — the same index gift as var 2, at half the payload).
The y axis splits as (warp bit, lane4, lane3), z = lane2..0: exactly ONE stage
crosses warps, done through a stride-9 padded shared bounce (4.6 KB/warp) with a
**pair-local `bar.sync p, 64`** (inline PTX named barrier), so the pairs of a
multi-volume block never couple — a tail pair exits without deadlocking anyone,
and no `__syncthreads` exists in the kernel. All other stages are the unchanged
cross-lane `xstage` machinery; the lane-bit meanings work out to the SAME twiddle
constants as var 2 (`l8_setup` is reused verbatim), and the bounce mirrors var 2's
local y stage slot-for-slot (low side `cadd` only, high side `cmul(csub(lo,hi),
W8^(y&3))`). **Output is bit-identical to var 2 — verified by `cmp` at B=63
(tail path, wpb 2 and 8) and B=2048 — so var 4 joins the measured autotune
without touching the r1 repeatability lesson.** Stores stay 4×128B sector-dense:
the address absorbs both DIF bit-reversals (ky = c.H + 8·warpbit, kz = c.Wp).

**B = 1 now runs `fft8_pair<2>` as one 64-thread block**: 3.55–3.59 µs vs
fft8_b1's 3.73–3.82 in the same windows. Same barrier count (one); the entire
difference is the load shape — b1's per-lane map `in[64x+8y+4j+zl]` touches half
of each 128B sector, the pair kernel's flat load is dense. fft8_b1 is kept
env-forceable (`L8WR_VAR=-1`).

### Operation count

Per lane per volume (two lanes now share what one did in r1): 8 ld + 8 st +
1 dft8 (56 FP) + 8 shared st + 8 shared ld + 1 bounce butterfly (8 cadd or
8 csub+cmul) + 5 xstage stages (80 shfl + 80 FMA + 3×32 cmul-FP) ≈ ~265 FP +
80 shuffle + 16 shared accesses; per point identical arithmetic to var 2 (~33
flop/point), plus 2 shared accesses/point that var 2 does not have — invisible
under the 32 B/point HBM floor, and cheaper than the 40 registers they free.

### Measured (leased SXM4 via tryout.sh; forced A/B in the same windows)

| case | r1 (var 2) | r2 best | config | vs blockfused r1 score |
|---|---|---|---|---|
| B = 1 | 3.74–3.90 µs | **3.55–3.74 µs** | fft8_pair<2>, one block | 3.184 (still theirs) |
| B = 2048 (L2) | 14.19–14.26 µs | **14.15–14.37 µs** (autotuned 14.15) | var 4 wpb 4 + `__stcs` ≈ var 2 wpb 4 | 13.238 (still theirs) |
| B = 131072 (HBM, primary) | 1549–1560 µs | **1537.1–1542.4 µs** | var 4 wpb 4, plain stores | 1536.8 (now a tie) |

rel L2 2.3e-16 at every point (B = 1, 3, 63, 64, 333, 2048, 131072 all PASS);
bit-identical across runs; compute-sanitizer memcheck 0 errors on the batched
tail path and the B=1 path. ncu at B_HBM: DRAM 88.8% of peak at 1.38 TB/s —
the same wall as everyone (L6/L8/L13 all land 89–90%); the 12 µs gained over r1
was the occupancy residual, as diagnosed.

### Tried and rejected this round, with the numbers

* **var 5, grid-stride persistent form of var 4** (kill the 1.35-wave tail that
  ncu showed at B_L2: achieved occupancy 13.8 of 28 theoretical warps): B=2048
  **15.52 vs 14.37**, B_HBM **1642 vs 1537**. The loop needs a second pair
  barrier per iteration (the partner may still be reading the bounce region when
  the next volume wants to overwrite it), and that serializes inter-volume
  memory-level parallelism inside each pair — independent one-shot blocks
  overlap far better. Kept env-forceable, excluded from the tune.
* **var 6, ptxas forced to 64 registers via `__launch_bounds__(128, 8)`**
  (32 warps/SM instead of 28, at 24 B/thread of spills): B=2048 **14.84 vs
  14.37**, B_HBM **1548 vs 1537**. Two more resident warps do not pay for spill
  traffic — the same verdict L6_warpvolume recorded for spills-vs-occupancy in
  r1. Natural 66 registers stands.
* **Odd warps-per-block for wave packing** (wpb 6 → 30 warps/SM at 1.26 waves,
  wpb 10 → 30 at 1.26, vs wpb 4's 28 at 1.35): B=2048 **15.38 / 16.06 vs
  14.36**. Wave quantisation is not the L2-point limiter; 128-thread blocks are
  simply the sweet spot (blockfused's V=2 winner is also 128 threads). Kept
  env-forceable, excluded from the tune.
* The B_L2 residual (14.15 vs their 13.24, floor ≈ 11.6 µs) survived every
  lever tried this round: more theoretical warps (var 6), tail removal (var 5),
  block shape (wpb sweep). ncu shows nothing saturated (DRAM 52%, L2 71%,
  SM 44%, 12 cy/instr) — it is ramp/drain time on a 14 µs kernel plus my ~400
  shuffle+FMA instructions per warp of cross-lane machinery, which var 2/4
  cannot shed. A thread-per-line shared-memory pass structure (= blockfused)
  is presumably what those 0.9 µs cost; that is the experiment this pair
  exists to keep separate.

### Borrowed this round, with attribution

* The occupancy framing that motivated the whole round: **L8_blockfused r1**
  (their 40-reg/32-warp point and my own r1 ncu numbers).
* Carveout-hint discipline for the shared-using kernels
  (`cudaFuncAttributePreferredSharedMemoryCarveout = 100`): **L13_dmma /
  L8_blockfused r1**.
* Confirmation to not chase spill-for-occupancy trades: **L6_warpvolume r1**
  (their `__launch_bounds__(64,8)` failure, reproduced here as var 6).
* The bounce structure itself is my own fft8_b1 generalised to the batch —
  with the loads fixed to flat-dense, which is the r1 lesson from
  **L6_batchcoalesced/L13_dmma** (per-thread-contiguous access loses) applied
  to my own B=1 kernel.

### What I would do next

1. **B_L2**: the remaining 0.9 µs to blockfused is structural (shuffle-stage
   instruction count + ramp on a one-wave kernel). If it must be closed, the
   honest experiment is a hybrid: pair-kernel loads and registers for x, but y/z
   as thread-per-line passes through the 4.6 KB of shared the bounce already
   allocates — that converges toward blockfused and should only be done if the
   panel wants the register-vs-shared question collapsed.
2. **B = 1 (3.55 vs their 3.18)**: both are one ~2.5–3 µs launch plus a short
   kernel; the remaining delta is kernel critical path. A 4-warp split (4
   complex/lane, two bounces) lengthens the chain and was reasoned dead; a
   graph/persistent trick is outside the per-call execute contract. Low value.
3. **B_HBM is at the wall** (88.8% DRAM, minimum bytes, tied with blockfused).
   Do not spend another round there.
4. For **L64_radix8**: the pair-kernel pattern (one cross-warp stage through a
   pair-local named barrier, everything else in registers) is exactly the shape
   for a 64-point line at 2 lanes/point-set; the bar.sync trick keeps their
   plane blocks decoupled too.
