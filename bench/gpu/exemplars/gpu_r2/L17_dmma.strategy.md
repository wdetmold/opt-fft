# L17_dmma — strategy record

Geometry: **L = 17**, cube 17³ = 4913 complex doubles (78,608 B) per volume, forward,
unnormalised, out-of-place, batched, one A100-SXM4-40GB.
Implementation: `impl/L17_dmma.cu`. `fft3d_gpu_name()` → `L17_dmma`.

Assignment: dense matrix via FP64 tensor cores (DMMA). This round establishes the
strongest possible non-DMMA baseline of the same matrix formulation and answers the
DMMA question with that measurement — see the verdict section at the end.

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique

Single fused kernel, one volume per block, whole volume in shared memory, one global
read + one global write (corpus 09 regime C). Grid = batch, 320 threads (289 active
lines per pass), 78,608 B dynamic shared (needs the sm_80 opt-in via
`cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` — the brief's trap is real, the
kernel refuses to launch without it).

Pass structure:
1. **Staged load**: flat coalesced copy global → shared, 16 B `cp.async` per element
   (`__pipeline_memcpy_async`), thread-strided so every warp reads 512 consecutive B.
2. **z-pass** in place in shared (lines contiguous, stride 1),
3. **y-pass** in place (stride 17),
4. **x-pass** reads shared (stride 289) and writes straight to `out`, 32 consecutive
   double2 per warp per k — perfectly coalesced.

Per-line arithmetic is the CPU panel winner's conjugate-folded dense matvec
(**adopted from `../geom/strategies/L17_matrixsimd.md` round 1, with attribution**):
fold `u_j = x_j + x_{17-j}`, `v_j = x_j - x_{17-j}`, then
`P_k = x0 + Σ cos(2πkj/17)u_j`, `Q_k = -iΣ sin(2πkj/17)v_j`,
`X_k = P_k ± Q_k`. All coefficients real, broadcast from `__constant__` (all lanes
read the same entry per step — the one access pattern constant memory is good at).

**Shared layout is AoS double2, deliberately unpadded.** A 16-byte shared access is
serviced in phases of 8 consecutive lanes; the phase is conflict-free iff the
lane-to-lane stride in complex elements is odd (corpus 09 §6.2). All three passes'
line-start strides here step by odd amounts mod 8 (17, 289≡1 mod 16 with the z-wrap,
1), so L=17 needs no padding anywhere — the corpus's table said exactly this.

**Occupancy**: `__launch_bounds__(320, 2)` forces 96 registers (zero spills) so two
blocks (2 × 78.6 KB = 157 KB ≤ the 164 KB carveout) share an SM — 20 warps. That is
the structural ceiling: a resident 17³ volume costs 78.6 KB of shared (or ~40+
regs/thread of register file) per block, so no arrangement of this structure exceeds
2 blocks/SM. Theoretical occupancy 31%, achieved 30%.

**Small-batch path (batch ≤ 4)**: the fused kernel at B=1 is one block on one SM,
latency-bound at ~15 µs. Instead: one **cooperative launch** (grid sync, checked
supported + co-resident in `create()`, plain two-launch fallback kept) of a
plane-split kernel — 17·B blocks of 289 threads do a dense per-output z-DFT then
y-DFT of one 17×17 plane out of shared into a global scratch, `grid.sync()`, then the
same blocks do a dense per-output x-pass (block = (volume, kx), twiddle
constant-broadcast, reads/writes 32-consecutive per warp). Twice the traffic, denser
arithmetic, 17× the SM coverage — at B=1 the right trade by 1.35×.

### Operation count

Per line: 272 FMA + 64 add = 608 real flops (the CPU record's count, unchanged).
Per volume: 867 lines → 236 k FMA, 291 k FP64 pipe ops, i.e. **107 flop/point over
three axes**. Arithmetic intensity 107/32 B = 3.35 flop/B against the vanilla-pipe
balance of 6.24 — **0.54× of the HBM floor on the ordinary FP64 pipe**, no tensor
cores needed. (This is half the corpus §5.4 "conj-symmetric" count of 216 flop/point:
that column folds only the outputs; folding the inputs too — the u/v butterfly —
halves it again. The corpus's DMMA crossover table compares against the wrong scalar
baseline; see the verdict.)

### What was measured — reserved node (A100-SXM4-40GB, leased, sd ≤ 0.2% at large B)

Scored cases (`cases.txt`: B = 1, 213, 13660), min over samples, per transform:

| case | this kernel | per call | cuFFT | speedup | rel_l2 |
|---|---|---|---|---|---|
| B=1 (launch case) | **11.49 µs** | 11.49 µs | 13.71 µs | **1.19×** | 3.60e-16 |
| B=213 (L2, 32 MiB) | **161.8 ns** | 34.47 µs | 68.0 µs | **1.97×** | 3.08e-16 |
| B=13660 (HBM, 1 GiB) | **115.8 ns** | 1582.4 µs | 4765 µs | **3.01×** | 3.08e-16 |

Effective bandwidth at B=13660: **1357 GB/s**. Repeatable (bit-identical across
processes) at every batch; `compute-sanitizer` clean on both paths; B=13660
correctness verified with a chunked numpy check (login node ran out of host RAM for
the stock checker — transient contention, use chunks of 512 volumes).

**Hardware correction for whoever reads this next: the SXM4-40GB peak is 1555 GB/s,
not the brief's "~2.0 TB/s".** The node reports 1215 MHz memory clock; ncu reports our
719 GB/s as 46.25% of peak → 1555. (2.0 TB/s is the 80 GB part.) So 1357 GB/s is
**87% of peak**, above VkFFT's ~82–84% radix-path band — the batched kernel is at the
DRAM wall, and further arithmetic/issue work provably cannot move the primary score.

### The story of the round: three structural findings, each worth remembering

1. **Occupancy: registers, not shared, were the first limiter.** The natural build
   used 128 regs → 1 block/SM → B=13660 at 3380 µs. `__launch_bounds__(320,2)` gave
   96 regs with zero spills → 2377 µs (−30%). Forcing 3 blocks (MINB=3) spills 816 B
   → 60 µs at B=213 vs 34.5 — and shared caps at 2 blocks anyway. 2 is the answer.
2. **The z-pass must not read global memory per-line.** Reading each thread's
   contiguous 272 B line directly from global (via L1) was the single biggest cost in
   the file: ncu showed 46% DRAM, 43% FP64, warps stalled 37 cycles/instr spread over
   MIO-throttle/math/barrier/LG-throttle. Replacing it with a flat coalesced staging
   copy into shared dropped B=13660 from 2377 → 1629 µs (−31%) and the run-to-run sd
   from ~6% to 0.02%. `cp.async` on the copy gave another −2.5% (1629 → 1587).
3. **Constant memory serialized the small-batch kernel.** The plane kernel's twiddle
   table read `d_w17[t]` per-lane divergent — constant cache replays 32-way. Moving
   that one read to global memory (and using constant only where all lanes read the
   same entry) plus the dense per-output x-pass took the split path from 17.5 µs to
   13.1 µs; fusing the two launches into one cooperative launch took it to 11.4 µs.

### What was tried and did NOT work (with the number that killed it)

1. **SoA shared (separate re/im doubles).** Chosen first out of bank-conflict
   caution; the 8-lane-phase rule makes unpadded AoS conflict-free at L=17, and AoS
   halves the LSU instruction count. Measured (unstaged, B=13660): SoA 2377 vs AoS
   2440 µs — a 2.6% wash, kept AoS for the cp.async pairing. Not worth revisiting.
2. **MINB=3** (force ≤68 regs): spills 816 B/thread → B=213 60.0 µs vs 35.8. Dead.
3. **Two-launch split at B=1 before the fixes**: 17.5 µs vs 15.0 fused — launches are
   not free; the split only wins once the kernels themselves are latency-lean.
4. **Nested cyclic-4⊕negacyclic-4 split of the cosine half** (ported from
   L17_matrixsimd rounds panel_r2/r7, with attribution — tables, slot map and signs
   all verified, PASS 3.1e-16 first try, 96 regs no spills, kept in the file under
   `-DL17_NESTED=1`): −12% FP64 ops, −25% FMAs, and **zero measured gain**: B=213
   35.97 vs 35.78 µs, B=13660 1586 vs 1587 µs. Same lesson as the CPU r2 round (12%
   ops → 1.4% time): this kernel's batched cells are bytes-bound (HBM) or
   stall-mixture-bound (L2), not FMA-count-bound. Do not spend another round on
   arithmetic reduction of the fused kernel.
5. **Full-volume checker at B=13660 on the login node**: numpy OOMs at 1 GiB under
   agent contention; the earlier "missing PASS" lines in my tryouts were this, not a
   kernel fault. Chunked check (512 volumes at a time) works.

### The DMMA verdict this entry was created to deliver

**DMMA cannot pay at this geometry in the batched regimes, and the measurement is
now in hand.** The chain of evidence:

* The primary cell (B=13660) runs at **87% of DRAM peak with the FP64 pipe only ~43%
  busy** — the arithmetic of the double-folded dense matvec (107 flop/point, 0.54× of
  the memory floor on the *vanilla* pipe) is already invisible. A 2× arithmetic pipe
  cannot speed up a kernel that is at the memory wall; the corpus's own §5.4 framing
  ("the dense matrix is free at L≤16.7 *on DMMA*") is superseded by the double fold,
  which makes it free at L≤17 *without* DMMA. The corpus's crossover table compares
  DMMA against the single-fold (216 flop/point) baseline; the right baseline is 107.
* The L2 cell (B=213) is the one compute-window: measured stall profile there is
  math-pipe-throttle-led (6.2 of ~22.7 cycles/instr) with FP64 ~50% active. But
  `m8n8k4` pads M=17→24 (1.41× waste), cutting DMMA's 2× to ≤1.4× effective — and the
  nested-split experiment above showed a genuine 12% FP64-op cut moving the needle
  0%: the cell is not FMA-throughput-bound either, it is a latency/issue mixture at
  the 31% occupancy that a resident 78.6 KB volume structurally imposes. DMMA's
  fragment staging (fold results scattered into 4×8 B-fragments, D-fragments
  scattered back through shared for the ±combine) roughly triples shared-memory
  traffic per point, on the resource whose throttle already leads the profile — the
  FEM paper's finding ("shared memory data motion, not FLOPs, is the bottleneck")
  applies with force.
* B=1 is a launch/latency problem; tensor cores are irrelevant there.

What would change this verdict: a structure where the DMMA operand staging is free —
e.g. lines pre-transposed so u/v land in fragment order from the butterfly, and the
±combine fused into the next pass's load. If a future round wants to try, the target
is the **B=213 cell only** (34.5 µs vs a ~13 µs FP64-issue floor), the budget for
staging is < 1 shared round-trip per point, and the file's `dft17_line` lambda
factoring means only the line kernel needs replacing.

### What I would do next

1. **B=213 is the improvable cell** (161.8 ns vs a ~60–70 ns issue-floor estimate).
   The lever is not arithmetic (measured, item 4) but stall structure at 20 warps/SM:
   software-pipelining volume n+1's cp.async staging into volume n's x-pass
   (persistent blocks, double-buffered shared — costs the second block per SM, 10
   warps, so it must win back 2× in issue efficiency; measure, don't assume).
2. **B=1 residual**: 11.49 µs against cuFFT's 13.7. The cooperative kernel's
   grid.sync plus two dense passes leave maybe 2–3 µs; a folded (rather than dense)
   per-output form in the plane kernel would cut its FMAs ~2× if the u/v fold is
   shared across the 17 outputs of a line via shared memory. Diminishing returns.
3. **Do not re-measure**: MINB∈{1,3}, SoA, nested split, per-line global z-reads,
   divergent constant-memory reads. All numbered above.

---

## Round gpu_r2 (2026-08-22) — cumulative round: took the rival's line module, beat them at B=1 with a software grid barrier

Standings entering the round: L17_raderfused led B=1 (9.76 vs my 11.36 µs) and B=213
(31.69 vs 35.82); I led B=13660 (1584.7 vs 1610.7). This round closes both gaps and
keeps the HBM lead. Everything below measured on a leased SXM4 via tryout.sh;
min-of-samples quoted (several windows had 7–12% sd; clean windows reproduced the minima).

### What changed (three things)

1. **The per-line module is now L17_raderfused's `line17w`, ported verbatim with
   attribution** (`dft17_line_w`, default `L17_WLINE=1`). The decisive fact is NOT the
   flop count: my r1 nested split (same cyclic-4⊕negacyclic-4 algebra, 296 FP instr/line
   vs dense 336) measured **zero** gain, and re-measured this round it is still a loss
   (35.09 µs at B=213 with regstage). Their variant of the *same algebra* wins because of
   the **register schedule**: it emits the sine half one output pair at a time, so only 2
   sine accumulators are live, where my nested held all 16 `S[]` accumulators across the
   whole m-loop. Fold-first + pair-at-a-time emit is the shape to copy, not the op count.
   Isolated A/B at B=213 (staging fixed at cp.async): dense 34.54 → wline 32.61.
   At B=13660: dense 1584 → wline 1571.8.
2. **The staging copy is batch-selected at plan time** (`fft17_volume<REG>` template,
   both instantiations compiled, cut at `L17_REG_MAX_B=266` = in+out fits the 40 MB L2):
   - register-batched staging (15 unrolled ld→st rounds, taken from L17_raderfused r1):
     B=213 **31.75–31.84** µs — but 1610.5 at the HBM point;
   - cp.async staging (my r1 form): 32.61 at B=213 but **1571.8–1573.8** at HBM.
   Neither staging wins both regimes; the rival's file is stuck with one (they measure
   1611 at HBM), a plan-time switch takes both bests. This is the whole reason I now tie
   B=213 and keep the HBM lead in one binary.
3. **The small-batch split path (batch ≤ 16) is one PLAIN launch joined by a hand-rolled
   software grid barrier.** Measured launch mechanics at B=1 with identical bodies:
   cooperative launch 11.88 µs, two plain launches 10.50, **one plain launch + soft
   barrier 7.74–8.92** — `cudaLaunchCooperativeKernel` itself costs ~1.4 µs over `<<<>>>`,
   and the second launch ~1.6 µs more. The barrier is an arrive-and-spin on a plan-owned
   monotonic u64 counter (each call adds gridDim; target passed as an argument, so
   back-to-back calls on one stream cannot confuse epochs); release/acquire =
   __syncthreads + __threadfence around the atomics. Legal only because `create()`
   verifies the whole 17·B-block grid is co-resident (occupancy query × 108 SMs); if not,
   it falls back to the two-launch form automatically (exercised and PASSing at B=16–24
   in some windows). compute-sanitizer memcheck: 0 errors on both paths.
   The zy-plane body was also rebuilt along the rival's `fft17_planes` design (adopted,
   with attribution): no shared twiddle-table staging, folded per-output form with
   warp-uniform `__ldg` coefficient rows (`k = tid/17`), output transposed through shared
   so the global store is contiguous. The x pass stays my dense per-output form with
   broadcast `__constant__` twiddles (block-uniform kx). Split/fused crossover re-measured
   with the faster split path: split 14.0 µs at B=16 vs fused 15.6; fused wins by B=24
   (19.5 vs 15.6) → `L17_SPLIT_MAX_BATCH=16`.

### Operation count

Per line now 496 real flops (line17w), 867 lines/volume → **87.5 flop/point**, arithmetic
intensity 2.7 flop/B — still under the vanilla-pipe balance of 6.24, so the r1 DMMA
verdict stands unchanged (the arithmetic hides under the 32 B/point HBM floor without
tensor cores). Global traffic unchanged: one read + one write per volume on the fused path.

### Measured — final file, defaults, leased SXM4 (min over samples; per transform)

| case | this round | r1 | rival's r1 score | cuFFT | vs cuFFT |
|---|---|---|---|---|---|
| B=1 | **7.74 µs** (8.9 in a clean-sd window) | 11.49 | 9.76 | 13.95 | 1.8× |
| B=213 | **31.83 µs** = 149.4 ns/t | 34.54 | 31.69 | 67.9 | 2.13× |
| B=13660 | **1571.9 µs** = 115.07 ns/t, 1366 GB/s = 88% of the 1555 peak | 1582.4 | 1610.7 | 4758.7 | 3.03× |

rel L2 3.16–3.20e-16 at every batch tried (1, 4, 8, 16, 24, 64, 213, 300, 13660);
B=13660 verified with the chunked checker (512 volumes/chunk — the stock check.py still
OOMs at 1 GiB on the login node); bit-identical across runs at every point;
memcheck clean at B=1 and B=64.

### What was tried and did NOT work (with the number that killed it)

1. **Plane-granular cp.async pipelining of the staging load** (`L17_PIPE=G`, kept in the
   file): the 17 x-planes load in G commit groups; z and y are plane-local in x, so phase
   p runs y(group p−1) + z(group p) as soon as group p lands — the rival's r1 "next
   round" idea #1, built and measured. HBM point: G=2 1581.5, G=3 1579.8, G=4 1596.8 vs
   **1571.8 plain**; B=213: G=2 35.8, G=3 33.8 vs **31.75 regstage**. The overlap is real
   but the price (G+2 barriers instead of 3, and only ~45% of threads active in the
   overlap phases) exceeds the ~13%-of-DRAM-time it can hide; two co-resident blocks
   drifting out of phase already buy most of the overlap for free. Same conclusion as the
   rival's persistent-block failure, from the other direction. **The HBM point appears
   closed**: 88% of DRAM peak moving minimum bytes, and both structural overlap ideas
   (theirs r1, mine r2) measured negative.
2. **`__stcs` evict-first stores in the x pass** (HBM point): 1590.7 vs 1573.8. Matches
   L13_dmma's `__ldcs` result — sm_80 cache hints keep losing on this kernel family.
3. **Nested split re-test at B=213 with register staging**: 35.09 vs 33.95 (dense) and
   31.75 (wline). Confirms r1: the algebra's op count is not the lever; the emit
   schedule is.
4. **Plain `__ldg` staging loop**: 35.63 at B=213 — worse than cp.async (34.54) and
   regstage (33.95). The r1 finding that *some* explicit staging beats a naive loop holds.

### Borrowed this round (attribution)

* **L17_raderfused gpu_r1**: the `line17w` fold/emit module ported verbatim (their
  gpu_r1 port of the CPU `L17_winograd`-via-`matrixsimd` algebra — my nested tables are
  the same math; what I actually took is their emit ordering); the register-batched
  staging loop; the `fft17_planes` body design for the split path (no twiddle staging,
  warp-uniform `__ldg` rows, transposed store). Their record's negative results
  (persistent-block prefetch, fused z-load, spilled coefficient arrays) were treated as
  settled and not retried.
* **L13_dmma gpu_r1**: the 1555 GB/s SXM4 peak correction (also found independently in
  my r1), and the cache-hint skepticism their `__ldcs` number justified.

### What I would do next

1. **B=213 is still the open cell**: 149.4 ns/t against a ~60–70 ns issue-floor
   estimate, and mine/rival are now within noise of each other (31.7–31.9) with
   near-identical kernels. The unexplored levers: 2 volumes per 640-thread block (107
   blocks → one clean wave, halves barrier count per volume, 96 regs × 640 = 61k ≤ 64k
   so it fits), and an ncu stall-structure pass on the current kernel before guessing
   further.
2. **B=1 at 7.7 µs**: the two bodies are now lean; remaining time is one launch, one
   arrive-spin barrier, and two latency-bound passes. A folded (u/v) form of the zy body's
   per-output sums would cut its FMA chain ~2× but the loads dominate; low expected value.
3. **Do not re-measure**: everything in the r1 list, plus PIPE at any G, `__stcs`,
   nested-vs-wline, coop-vs-plain launch. All numbered above.
