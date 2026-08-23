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

---

## Round gpu_r3 (2026-08-22) — quad kernel closes B=1 and small batches; graph replay closes the launch path

### Standing entering the round, and the plan

r2 leaderboard: B=1 3.595 vs blockfused 3.191, B_L2 13.990 vs 13.058, B_HBM a
tie at the wall (1539.5 vs 1537.2). Two levers this round: (1) the next halving
of the r2 occupancy ladder — one volume per FOUR warps, 4 complex doubles/lane —
as the honest test of whether the B_L2 residual is occupancy after all; (2)
CUDA-graph replay of the single kernel launch, taken from other entries'
records.

### What changed (production)

1. **vars 7/8, `fft8_quad`: one volume per four warps, 4 complex/lane.** The
   flat coalesced load `v[r] = in[vol*512 + r*128 + u]` lands x = 2r + u6, i.e.
   the EVEN x's on warps 0-1 of the quad and the ODD x's on warps 2-3 — and the
   min-op dft8 codelet splits on exactly that parity: t0/t2/s0/s2/u2/e0/f0/g0/h0
   live entirely on the even side, t1/t3/s1/s3/u1/u3/e1/f1/g1/h1 on the odd side
   (verbatim expressions, warp-uniform branch), and **only the codelet's final
   cadd/csub stage crosses the parity bit**, through one stride-5 padded shared
   bounce. The y2 bit is the other warp bit; y1,y0,z2..z0 are lane bits with the
   SAME meanings as vars 2/4, so `l8_setup`/`xstage` are reused verbatim.
   **var 7** avoids a second bounce for the y2 stage by re-reading the bounce-1
   slots of the y2-partner (t^32) and its partner (t^96) and recomputing the
   partner's x-final value redundantly — same expressions, same operand order,
   so it is bit-identical to reading it, and the kernel has **ONE quad-local
   barrier total** (`bar.sync q, 128`). **var 8** is the classic second bounce,
   three barriers. Both verified BIT-IDENTICAL to var 2 by `cmp` at B=64 and
   B=63 tails (wpb 4/8/16), so both join the measured autotune; memcheck clean.
2. **Every plan now launches through a lazily-captured CUDA graph** (keyed on
   the (in,out) pointers, recaptured if the driver moves them — **borrowed from
   L36_sharedtiled r1 and L45_pfa r2**). Every plan here is a single kernel
   launch, so the graph is one node; the replay is bit-identical (cmp-verified
   at 1, 2048, 131072) and saves 0.2–0.6 µs of launch path per call at B=1 and
   B_L2, neutral-to-+0.6 µs at B_HBM. It also collapses the B=1 launch-noise
   spread (sd 13% → 0.02% in like windows). `L8WR_GRAPH=0` disables.
3. **B=1 is a measured create()-time pick** between `fft8_pair<2>` (64 threads)
   and `fft8_quad<4>` (128 threads = all four schedulers of the SM, half the
   per-thread chain, one barrier) — both bit-identical to var 2. The quad wins
   every window measured (3.10–3.35 vs 3.35–3.67 µs).
4. Autotune reps floor raised to 6 at the HBM batch (3-rep samples flipped the
   pick among the top candidates, which sit ~0.1% apart).

### Operation count

var 7 per lane per volume (four lanes share what one r1 lane did): 4 ld + 4 st
global, half-dft8 (~28 FP), 4 shared st + 12 shared ld (bounce + partner
recompute), ~10 butterfly FP, 5 xstage stages (20 shfl + 20 FMA + 3×16 cmul-FP)
≈ ~135 FP + 20 shuffle + 16 shared accesses; per point the same ~33 flop/point
arithmetic as vars 2/4 (bit-identical by construction), with 4 shared
accesses/point against var 4's 2 — the price of the shorter per-thread chain.

### Measured (leased SXM4 via tryout.sh; forced A/B in the same windows)

| case | r2 shipped | r3 best | config | blockfused r2 score |
|---|---|---|---|---|
| B = 1 | 3.55–3.74 µs | **3.06–3.35 µs** (3.061 final tryout, median 3.062) | fft8_quad<4> + graph | 3.191 |
| B = 2048 (L2) | 14.15–14.37 µs | **13.19–13.31 µs** (13.189 final tryout) | autotune (var 4 family) + graph | 13.058 |
| B = 131072 (HBM) | 1537.1–1542.4 µs | **1536.5–1539.2 µs** | autotune picks var 7 or 4 (all within 0.15%) + graph | 1537.2 |

rel L2 2.29e-16 at every point (B = 1, 63, 64, 2048, 131072 all PASS);
bit-identical across runs at every point; compute-sanitizer memcheck 0 errors on
the new paths (quad tails wpb 8, var 8, B=1 graph). The B=1 and B_L2 gains are
real and window-stable; the HBM point remains the shared 89–90%-of-DRAM wall.

### What was tried and did NOT work, with the numbers

* **The quad kernel at the two big batched points — the round's designed
  experiment, and it settles the r2 question with the opposite sign.** At
  B=2048: var 7 wpb4+stcs **16.55 vs var 4's 14.24** (wpb 8: 16.01, wpb 16:
  17.34; var 8: 16.81). At B_HBM: **1542.7–1546.5 vs 1540.7** like-for-like.
  40+ warps/SM (vs var 4's 28) made the L2 point WORSE: the doubled per-point
  shared traffic and the extra warp-bit stage cost more than the occupancy
  buys. Together with blockfused r2's WPV result (barriers are not it either),
  the B_L2 residual is now bracketed from both sides: it is the per-thread
  instruction stream itself, not occupancy, barriers, or store semantics. The
  autotune keeps the quad because it wins small batches outright (B=64 forced:
  4.76 vs var 2 wpb4's 7.39; full tryout at B=64 now 3.49 vs r2-era 3.75).
* **var 8 (3-barrier two-bounce quad) vs var 7 (1-barrier redundant recompute)**:
  4.96 vs 4.76 at B=64, 16.81 vs 16.55 at B=2048 — the r1 lesson inverts when
  the redundant reads are SHARED instead of global: 8 extra shared loads beat
  two extra barriers. Kept in the tune anyway (bit-identical, and the margin is
  small).

### Borrowed this round, with attribution

* **CUDA graph replay, lazily captured and keyed on the pointers: L36_sharedtiled
  r1 (B=1 graph) and L45_pfa r2 (capture/recapture discipline).** Extended here
  from their multi-kernel B=1 cases to EVERY batch point — with a single-kernel
  execute the graph is one node and the win is pure launch path. This is the
  single cheapest microsecond available to any entry whose execute is one
  launch; blockfused does not currently do it.
* The occupancy-ladder framing that motivated the quad: my own r1/r2 records
  plus **L8_blockfused r1's** 40-register point (the quad lands at the same
  register class).
* Carveout hints, env-knob A/B discipline, "env vars inside the on_gpu.sh
  command string": **L13_dmma r1/r2**.

### What I would do next

1. **B_L2 (13.19 vs their 13.06, floor ≈ 11.6)**: the instruction-stream verdict
   says the only remaining moves are arithmetic ones — fewer executed
   instructions per point, not better scheduling of the same ones. A radix-8
   y/z pass done as 3 genuinely fused cross-lane stages (one cmul per axis
   instead of two) would break bit-identity with var 2 and need its own tune
   family; expected ≤ 0.5 µs. Marginal.
2. **B=1 at 3.06 µs**: now graph-replay launch + a 128-thread kernel. What is
   left is the kernel's global-load latency head and the launch interval
   itself; nothing under the per-call execute contract looks worth more than
   ~0.1 µs.
3. **B_HBM**: unchanged verdict — at the wall, tied, every lever measured ≤ ±0.3%.
   Do not spend a round there.
4. **For any single-launch entry (L6 pair, L13, L17): take the graph replay.**
   It is ~30 lines, bit-identical, and worth 0.2–0.6 µs/call at the latency
   points.
