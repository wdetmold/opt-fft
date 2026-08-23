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

---

## Round gpu_r3 (2026-08-22) — B=1 kernel borrowed back; two batched ideas measured dead

### Standing after r2, and what this round targeted

r2 leaderboard: won B_L2 (14.63 vs rival L6_warpvolume's 14.96) and tied B_HBM (1540.2
vs 1540.6, both at the ceiling), but **lost B=1 by 1.24×** (3.69 vs their 2.97 µs) —
the rival added a dedicated 64-thread single-volume kernel while my execute still
launched the 288-thread batched kernel for one volume. B=1 was the round's target; at
the batched points the plan was cheap measured experiments only, since L8_blockfused's
r2 record had already closed most of the latency hypotheses at the same working set.

### The change (one addition that stuck)

**`fft6_single`, a dedicated B=1 kernel selected in `execute()` when `batch == 1`:**
one 36-thread block (2 warps), the **z-pass fused into the global load** — thread
l = (x,y) reads its own contiguous 96 B z-line from global, transforms in registers,
writes shared once — then y-pass and x-pass over padded shared (`slot = i + i/6`),
x-pass storing straight to global. **Two barriers total** and one shared round trip
fewer than the staged shape. Borrowed twice over, with attribution: the single-volume
line-thread shape from **L6_warpvolume round gpu_r2** (`fft6_single`, 3.09 µs), the
fused-z-load-at-B=1 refinement from **L8_blockfused round gpu_r1** (their staging-free
B=1 kernel measured fused-z the winner at B=1 even though the same fusion loses 2.5%
batched — my own r1 record killed it batched at 1582 vs 1540 µs, and both records
agree the sign flips when latency, not sector economy, is what is being paid).
Arithmetic identical to the batched kernel (same `dft6`), deterministic, so
repeatability is structural.

### Measured (reserved-node SXM4 lease via tryout.sh / on_gpu.sh, final binary)

| case | r2 | r3 | note |
|---|---|---|---|
| B = 1 | 3.69 µs | **2.87 µs min, 2.88 median** (sd ~11%, launch-bound) | 1.29× on the lost cell; rival scored 2.97 |
| B = 4854 (L2) | 14.63 µs | **14.72–14.88 µs min** across windows | unchanged — batched kernel is bit-for-bit r2's |
| B = 310608 (HBM) | 1540.2 µs | **1540.2–1544.6 µs min** across windows | unchanged (cross-lease clock drift ~0.3%) |

rel L2 error 2.4–2.5e-16 at B = 1, 3, 4854, 10925 (tail block of 5, plain-store side
of the stcs boundary), 310608 — all PASS, all bit-identical across runs;
`compute-sanitizer memcheck` 0 errors on the new B=1 path and the tail path.

### Tried and rejected this round, with the numbers that killed them

* **z→y barrier demoted to a 96-thread named barrier** (`bar.sync id, 96`, ids 1–3;
  provably correct: a y-pass thread reads only its own volume's points, written by
  z-pass threads of the same aligned 48-tid group, and 96 is the warp-aligned cover —
  analogue of L8_blockfused r2's `__syncwarp` demotion). First A/B looked +0.8% at
  B_L2, but interleaved same-lease runs reversed it: **B_L2 14.97/14.99 vs 14.76/14.78
  µs plain (−1.4%), B_HBM 1543.3–1544.5 vs 1540.2–1541.1 (−0.2%), consistently**. The
  named-barrier path costs more than waking 6 fewer warps saves. Kept in the file
  behind `-DNAMED_BAR=1`, default off. Lesson restated: a one-window A/B at sd > 1%
  is not a measurement; interleave in one lease before believing a sign.
* **Single-wave residency at B_L2 via `__launch_bounds__(288, 6)`** (607 blocks ≤
  648 = 6/SM capacity, so the 1.12-wave tail becomes exactly one wave): **23.16 µs vs
  14.88** — catastrophic. ptxas cuts 40 → 32 registers with spills, and the spill
  traffic lands in L2 exactly where `__stcs` is protecting the input's residency.
  Extends r1's B_HBM-only measurement (+17%) to the L2 point (+56%); the knob stays
  in the file as `-DMINB=n`. Occupancy bought with spills is a loss everywhere here.
* **ncu at B=4854 (final kernel)** closed r1's leftover worry: shared bank conflicts
  are 37.6 K ld + 18.9 K st per launch against millions of shared ops — the load
  phase's "4-way conflict, paid once" is **not measurable**, so the register-staged
  two-step transpose idea from r1's "next" list is dead without being built.
  dram__bytes = 16.80 MB = exactly the write stream (reads fully L2-resident);
  achieved 30 warps/SM of 45 theoretical, nothing above L2's 67%. The residual
  ~1.25× over the 11.6 µs write floor is the read+compute+write latency chain at
  1.12 waves — same verdict L8_blockfused r2 reached from five structural
  experiments at the identical working set.

### Borrowed, with attribution

* B=1 single-volume kernel shape: **L6_warpvolume gpu_r2** (`fft6_single`), itself
  descended from **L8_blockfused gpu_r1**'s `fft8_single`.
* Fused-z global load at B=1 (two barriers): **L8_blockfused gpu_r1**.
* "Generate inputs and run check.py on the reserved node — the login node's strict
  overcommit kills ≥512 MiB numpy allocations under agent contention": **L8_blockfused
  gpu_r1**'s warning, hit verbatim this round (login node failed even small OpenBLAS
  allocations mid-session).
* Not re-tested on the strength of others' records: warp-per-volume / zero-barrier
  variants (L8_blockfused r2: +7% B_L2), persistent grid-stride (+5%), `__stwt`
  (+79%), cross-execute L2 persistence windows (L6_warpvolume r2: all ≥ +0.7%).

### Where this leaves L=6, and what I would do next

* **All three cells now lead or tie on tryout numbers**: 2.87 / 14.7 / 1540-class,
  vs rival's scored 2.97 / 14.96 / 1540.6. cuFFT is 3.5× / 3.5× / 2.29× behind.
* **B=1 (2.87 µs) is the launch floor**: the kernel is ~0.1 µs of work on a ~2.8 µs
  launch path; the remaining levers (graphs, persistent kernels) are outside the
  per-call execute contract, and single-kernel graph replay has no second launch to
  amortize. Done.
* **B_L2 (14.7 µs = 1.27× of the 11.6 µs write floor)**: now measured NOT to be bank
  conflicts (ncu), NOT barriers (named-barrier loss here; L8's WPV loss), NOT wave
  quantisation (MINB=6 loss; L8's grid-stride loss), NOT store semantics, NOT staging
  shape. What remains is pure memory latency at 30 achieved warps/SM. I see no
  remaining structural lever that does not sacrifice occupancy or bytes; treat ~14.7
  as this design's answer unless someone finds a fundamentally different schedule.
* **B_HBM: closed since r1** — 89–90% of the part's 1555 GB/s sustained peak at
  exact-minimum bytes, every entry lands there, and both of this round's experiments
  confirmed the sign of any perturbation is negative. Do not spend another round.

---

## Round gpu_r4 (2026-08-22) — CUDA-graph replay of the single launch, taken whole from L8_warpradix8

### Standing entering the round

r3 leaderboard: won B=1 (2.832 vs rival L6_warpvolume's 3.064) and B_L2 (14.599 vs
14.876); B_HBM effectively tied (1540.4 vs their 1538.5, 0.12% — noise-level, both at
the DRAM wall). My r3 record had declared all three cells closed under the per-call
execute contract. **L8_warpradix8's r3 record proved that wrong for the launch path**
and said so explicitly: "For any single-launch entry (L6 pair, L13, L17): take the
graph replay. It is ~30 lines, bit-identical, and worth 0.2–0.6 µs/call at the latency
points." The rival L6_warpvolume had considered graphs in r3 and declined *without
measuring* ("nothing to collapse" — true of node count, but the win is launch-path CPU
cost, not node collapsing). This round is that borrow, measured.

### The change (one idea, borrowed whole)

**Every execute now replays its single kernel launch through a lazily-captured CUDA
graph**, keyed on the (in, out) pointers and recaptured if they change — the exact
pattern (capture on a throwaway non-blocking stream, instantiate once, `cudaGraphLaunch`
onto stream 0, destroy in `destroy()`) from **L8_warpradix8 round gpu_r3**, which took
it from **L36_sharedtiled r1 / L45_pfa r2**. Same kernel, same arguments → bit-identical
output; the one-time capture cost lands in the driver's warmup. `L6BC_GRAPH=0/1` forces.

Rotated same-lease A/B (graph alternating with plain, 3–4 process pairs per point,
after the r3 lesson that fixed-order A/B carries a first-process clock penalty):

| point | plain launch | graph replay | delta |
|---|---|---|---|
| B = 1 | 2.822–2.986 µs | **2.618–2.632 µs** (one 2.893 outlier pair) | −0.34 µs, −11% |
| B = 4854 (L2) | 14.775–14.889 µs | **14.350–14.465 µs** | −0.42 µs, −2.9% |
| B = 310608 (HBM) | 1540.52 / 1540.69 / 1543.85 µs | **1540.35 / 1540.05 / 1543.77 µs** | wash, graph ahead in every pair |

**Unlike L8's finding (neutral-to-+0.6 µs at their HBM point), graph replay never
measured a loss here, so it is on at every batch** — not gated on `stream_st`. It also
collapses the B=1 launch-noise spread (sd ~10% → 0.02–0.4%), which makes the scored
minimum far more repeatable.

### Tried and rejected this round, with the number that killed it

* **`fft6_single216`, a 216-thread one-point-per-thread B=1 kernel** (idea from
  L8_warpradix8 r3's quad-at-B=1 win: use all four SM schedulers, shorten the
  per-thread chain): ping-pong shared (no RAW hazard), each pass a direct 6-point DFT
  for one output via a `__constant__` W6 table with incremental (j·k) mod 6 indexing,
  three barriers. Correct (2.5e-16) and clean, but **3.24–3.39 vs 2.63–2.87 µs in every
  rotated pair — 0.65 µs worse**. The serial 6-cmul accumulation chain per output plus
  the third barrier cost more than 5 extra warps of scheduler coverage buy; L8's quad
  win does not transfer to a 216-point volume where the 36-thread kernel's dft6 tree is
  already short. Kept env-gated (`L6BC_B1=1`), default off. Do not rediscover.

### Measured (reserved-node SXM4 lease via tryout.sh, final binary)

| case | r3 scored | r4 tryout | note |
|---|---|---|---|
| B = 1 | 2.832 µs | **2.618–2.632 µs min** (sd 0.01–0.4%) | graph replay |
| B = 4854 (L2) | 14.599 µs | **14.350–14.465 µs min** | graph replay |
| B = 310608 (HBM) | 1540.4 µs | **1539.6–1540.7 µs min** | graph on, wash-to-marginal gain |

rel L2 error 2.4–2.5e-16 at B = 1, 4854, 10925 (tail block of 5 + stcs boundary,
58.4 µs), 310608 — all PASS (the 1 GiB case checked with check.py on the reserved
node); bit-identical across runs at every point; `compute-sanitizer memcheck` 0 errors
on the graph-launch paths at B = 1 and B = 4854. Kernels themselves are untouched
since r2 (batched) and r3 (B=1) — every claim about their behaviour in earlier rounds
still holds.

### Borrowed, with attribution

* **The whole round is one borrow: single-kernel CUDA-graph replay, lazily captured and
  keyed on the pointers, from L8_warpradix8 gpu_r3** (originally L36_sharedtiled r1 /
  L45_pfa r2). Their record's closing advice named this entry; it was right.
* The rotated same-lease A/B protocol: **L6_warpvolume gpu_r3**'s fixed-order artefact
  lesson, followed throughout.
* The failed wide-B=1 experiment's motivation: **L8_warpradix8 gpu_r3**'s
  quad-at-B=1 result (measured not to transfer here — recorded above so nobody
  re-imports it).

### Where this leaves L=6, and what I would do next

* All three cells lead or tie on tryout numbers: **2.62 / 14.4 / 1540-class** vs the
  rival's r3-scored 3.06 / 14.88 / 1538.5. cuFFT is 3.9× / 3.6× / 2.29× behind.
* **B=1 (2.62 µs)**: now graph-launch floor + ~0.3 µs of kernel. The 216-thread
  widening is measured dead; the 36-thread kernel's two-barrier chain is short enough
  that only the launch interval itself remains. If the rival adopts the same graph
  (they should), expect them near ~2.7 (their kernel is 64-thread and slightly longer).
* **B_L2 (14.4 µs)**: the graph shaved the launch contribution; what remains is the
  same read→3-pass→write latency at 1.12 waves that r2/r3 closed from every structural
  direction. No remaining lever known.
* **B_HBM (1540 µs)**: unchanged hardware answer, 89–90% of the 1555 GB/s part peak at
  exact-minimum bytes. The 0.1% gap to the rival on the r3 board is lease noise; do not
  chase it.
