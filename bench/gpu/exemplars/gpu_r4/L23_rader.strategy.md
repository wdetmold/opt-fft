# L23_rader — GPU strategy record

First GPU implementation for this entry (the impl/ file was a stub; no prior GPU rounds).
The name is kept for continuity with the CPU panel; as there, the "Rader" entry ships the
conjugate-folded dense form, because Rader-proper provably loses at p = 23.

## Round gpu_r1 (2026-08-22)

### Where it landed (leased A100-SXM4, the same part the monitor scores on)

| case | this entry | cuFFT same lease | ratio | rel L2 |
|---|---|---|---|---|
| B=1 | **10.8–11.5 µs** | 15.3–16.0 µs | **~1.4×** | 3.638e-16 |
| B=86 (L2 point) | **52.1 µs/call = 0.606 µs/xform** | 75.8 µs | **1.45×** | 3.650e-16 |
| B=5515 (HBM point, primary) | **3348 µs = 0.607 µs/xform** | 5070 µs | **1.51×** | 3.649e-16 |

Bitwise repeatable at every batch tried (1, 4, 32, 86, 100, 512, 2048, 5515); every
verified case PASS (the B=5515 numpy check OOMs on the login node sometimes — it passed
twice when numpy got its 1 GiB; the kernel path is identical at all batches ≥ 33).
Run-to-run spread on a quiet lease 0.04–1.3% at the batch points; B=1 median wobbles
~±10% with the boost clock (min is stable).

### Arithmetic (imported from the CPU record, not re-derived)

The CPU panel settled this at panel_r6: p−1 = 22 = 2·11, the conv-11 pair has no
realization under 121 fused FMAs, so honest-optimal Rader-23 IS the conjugate-folded
dense form. Per 23-point line: u_j = x_j + x_{23−j}, d_j = x_j − x_{23−j};
P_k = x_0 + Σ cos(2πjk/23)·u_j, Q_k = Σ sin(2πjk/23)·d_j; X_k = P−iQ, X_{23−k} = P+iQ
with the ±i a free component swap. **594 real FP instr/line (484 FMA + 110 add/sub),
943k per volume.** On the A100 that is 194 ns/volume at peak FP64 issue (4.87 T instr/s)
against a 195 ns single-pass HBM floor at 2 TB/s — folded-dense on CUDA cores is exactly
balanced, which is why I did not write DMMA this round (the corpus §09 recommends a
12×23 DMMA matrix at 0.75× floor, but that figure assumes the unfolded 288 flop/point
form; folded scalar FMA needs no tensor cores to hide under the memory floor).
Coefficients: 121 cos + 121 sin, computed on host with `cosl/sinl` (80-bit) and rounded,
in `__constant__` memory — every thread reads the same coefficient at the same time, so
it broadcasts.

### Structure (the actual GPU work of the round)

190.1 KiB/volume does not fit one block's 163 KiB shared (PANEL_BRIEF's central table),
and I verified the corpus's register-resident single-pass (structure 1 in §09 9.5)
does not close either: holding one 23-point line is 92 registers = 74% of the file, so
during the axis exchange a thread cannot hold both its old u/d set (88 regs) and its
incoming line — every wave/pipeline variant I worked through (paired-plane waves,
value-waves with ping-pong pair buffers, shuffle transposes) either doubles register
residency, floods shared traffic ×12, or idles half the block. L=23 single-pass needs
register+shared combined and a transpose that never doubles the moving data; I could
not construct one. **Two-pass it is:**

* **Kernel A** (z then y): per x-plane, staged through shared. A 23×23 plane is 8.46 KiB;
  stride 23 is odd, so both row (stride-16B×23) and column (contiguous) shared patterns
  are bank-conflict-free with zero padding. Block = P planes (P ∈ {1,2,4,5,8} × block
  sizes up to 192, tuner-selected), cooperative coalesced load, one line per thread,
  z in place in shared, **y written straight to global** (runs of 23 double2 per warp) —
  dropping the separate store phase and one barrier was worth **3.4% at B_HBM**.
* **Kernel B** (x): one line per thread, mapped so consecutive threads = consecutive
  (ky,kz) — at each x the warp's 32×16B accesses are contiguous, fully coalesced.
  Outputs via `__stcs` (never re-read; +3% over plain stores).
* **L2 chunking + stream rotation**: execute() loops over batch chunks with NSTR=4
  rotating tmp buffers, chunk c on stream (c % 4) with tmp (c % 4). Every hazard is
  ordered by stream identity alone (A(c)→B(c) same stream; B(c)→A(c+4) same tmp =
  same residue = same stream) — **no events**. Neighbouring chunks' A and B overlap,
  which is what lets 32–64-volume chunks (small enough to be L2-friendly) still fill
  the machine. `in` is read `__ldcs` (evict-first) to keep L2 for the intermediate.
* **FINE mode for tiny batch**: at B=1 both coarse kernels are thread-starved (23 and 9
  blocks) and expose a 594-instruction FP chain per thread; cuFFT was beating me there
  (17.2 vs 15.3 µs). `dft23_pair` puts **one thread on one conjugate output pair**
  (12 threads/line, u/d recomputed per thread and consumed immediately, ~30 live
  registers, 1.86× total FP) — 12× the parallelism. Same accumulation order, so outputs
  are bit-identical to the coarse path. Auto-selected at B=1 by the tuner: **17.2 →
  10.8 µs, from losing to cuFFT to 1.4× ahead.** Only raced when batch ≤ 32.
* **Plan-time autotuner** (create() is free): times (A shape, unrolled-load, kernel-B
  block ∈ {32,64,256}, chunk ∈ {24,32,48,64,full-capped-128}, fine) with CUDA events +
  device sync, min of 3, warm-up run per cell. The pick is baked into
  `fft3d_gpu_description()` (CPU-panel telemetry discipline, from L36_pfa r8 via my CPU
  record). `L23R_FORCE="aidx,TB,chunk[,uload[,fine]]"` pins a cell (that is how every
  A/B below was run), `L23R_VERBOSE=1` prints the table.

### What was measured on the way (all forced-config A/Bs, same lease unless noted)

* **The naive cooperative load was the B=1 disaster**: `for(i=tid;...;i+=T) sh[i]=in[...]`
  is a chain of dependent load→store DRAM latencies — ncu measured kernel A at **14.1 µs
  for ONE plane-group at B=1** (~3× its FP time). Fully unrolling with all loads batched
  before stores fixed B=1 but lost **11% at B_HBM** (3684 vs 3325 µs — floods the
  LSU/MSHRs); a group-of-4 compromise lost at both ends (3636). Shipped: unrolled load
  as a tuner flag, raced only at batch ≤ 32.
* **Tuner event bug worth recording**: events recorded on the default stream do NOT
  bracket work on `cudaStreamNonBlocking` streams — the first streams version "tuned"
  nothing (timed launch overhead, picked garbage, cost 15% end-to-end until found).
  Sync the device before the stop event, exactly as driver.cu's comment warns.
* **Register experiments, both directions dead**: everything (both dft23 kernels) sits
  at exactly 128 regs, 0 spills. Forcing 96 regs (launch_bounds 256,3) to raise
  occupancy: 644 B spill stores → **4985 vs 3546 µs** — catastrophic. So 512 threads/SM
  is the ceiling for the coarse kernels and latency must be hidden another way (streams,
  fine mode), not with occupancy.
* **L2 persisting window (accessPolicyWindow on each stream's tmp): zero effect**
  (3422 vs 3418 µs at chunk=32). tmp is already effectively L2-resident; the stall
  profile says the wall is LSU throughput/latency (kernB: long_scoreboard 7.9 +
  lg_throttle 5.4 per issue), not where tmp lives. Left in the code (it is free and
  should matter more if chunk grows); `L23R_NOL2WIN=1` disables for A/Bs.
* **kernB direct z-reads from global** (skip the kernA staging, per-thread-contiguous):
  4948 vs 3301 µs — per-instruction uncoalesced loads are unusable even
  sector-complete-in-aggregate. Coalescing wins over barrier-count at every scale tried.
* **kernB block size at scale: 256 ≫ 96/192** (3384 / 4627 / 5073 µs) — the ragged-tail
  lane waste of TB=256 (last block 17/256 lanes) is irrelevant because empty warps
  retire instantly; more blocks of smaller size lose on scheduling.
* **The tuner table at B_HBM is FLAT across chunk sizes** (P=8/TB=256: 3407 at chunk=128,
  3451 at 32, 3498 at 48, 3540 at 64) — i.e. the L2-blocking premise buys little here
  yet, because the kernels are latency-bound at ~50% combined DRAM utilisation, not
  bandwidth-bound. The chunking+streams machinery is still the right skeleton — it just
  needs faster kernels to matter.

### Cost accounting, honestly

At B_HBM: 0.607 µs/xform = 2.6× the 2-pass-with-L2-resident-tmp floor (~0.23 µs at
1.7 TB/s effective) and 1.7× the plain 2-pass DRAM floor (0.46 µs at 4 traffics). ncu
(serialized): kernA SM 34–61%, kernB SM 20–30%, combined DRAM ~50% of peak — both
kernels are memory-LATENCY bound at 2.5–3 active warps/scheduler with 0.13–0.09 issued
per cycle. The FP64 pipe is never the wall; neither is DRAM bandwidth. The gap is
concurrency at 128 registers/thread.

### Borrowed (attributions)

* **CPU L23_rader r6** (my own lineage): the entire arithmetic — the folded dense form,
  the Rader-counting that kills the alternatives, the 594-op/line count. Nothing there
  needed re-deriving; this round spent zero time on op counts, exactly as that record
  instructs ("treat further L=23 work as scheduling-only").
* **CPU L23_rader r10 / L13_direct r9**: the sine-sign-fold lesson arrives here for free
  — the ±i is a component swap with signs absorbed into the P±Q combination, so no GPU
  equivalent of the vpxor population ever existed.
* **corpus §09 (9.5, §2.4, §6.2)**: the 163 KiB shared ceiling correction (L=23 does NOT
  fit — an earlier corpus version's 192 KB figure would have sunk this design), the
  odd-stride-needs-no-padding rule, the L2-blocking-across-the-batch construction, and
  the plane-per-block fallback structure this entry ships as kernel A.
* **L36_pfa r8 via my CPU record**: tuner pick telemetry in the description string.
* **L17_matrixsimd r3 via my CPU record**: plan-time tuner shape (warm-up per cell,
  min-of-N, forced-pin env override).

### What I would do next (r2 agenda, ranked)

1. **The concurrency wall is the whole game at the batch points.** 128 regs/thread caps
   512 threads/SM and both kernels idle ~90% of issue slots on memory. Two escapes worth
   building: (a) **split-k kernB** — two threads per line, each folding only half the j's
   (44 regs), combining P/Q partials with 4 `__shfl_xor` per k-pair: ~64 regs → 1024
   threads/SM at +16% instructions; (b) a **fine-grained kernA/B pair at MEDIUM batch**
   (the B=1 fine kernels at 1.86× FP but 12× threads already win at B=1; the crossover
   vs coarse is unmeasured between B=2 and B=86 — scan it).
2. **DMMA is still unexplored** and L=23 pads best of all geometries (23→24, 1.09×).
   The corpus's caveat stands (shared-memory motion, not FLOPs, was the published
   bottleneck) — but if the coarse kernels stay latency-bound, an m8n8k4 formulation
   changes the memory pattern entirely, not just the FLOP rate. Budget a real round.
3. **Persistent-kernel / cooperative single launch at B=1**: two launches are ~3-4 µs of
   the 10.8; a grid-sync fused kernel could halve B=1 again. Low risk, bounded win.
4. **Do not re-try**: full-unroll loads at scale (+11%), sub-128 register forcing
   (spills, +50%), direct uncoalesced axis reads (+50%), kernB blocks < 256 at scale
   (+37%), L2 persisting windows for the current kernels (0%), Rader-proper/Winograd
   conv-11 (CPU record kills it by counting; nothing on a GPU changes that arithmetic).

## Round gpu_r2 (2026-08-22)

### Where it landed (leased A100-SXM4, same part as the monitor's node)

| case | this entry | r1 | cuFFT same lease | ratio | rel L2 |
|---|---|---|---|---|---|
| B=1 | **10.4–11.0 µs** | 10.8–11.5 | 15.0–15.1 µs | **~1.4×** | 3.535e-16 |
| B=86 (L2 point) | **38.5–41.9 µs = 0.45–0.49 µs/xform** | 52.1 | 75.8–76.1 µs | **~1.9×** | 3.547e-16 |
| B=5515 (HBM, primary) | **2262–2286 µs = 0.411–0.415 µs/xform** | 3348 | 5020–5072 µs | **~2.2×** | 3.547e-16 |

**1.47× faster than r1 at the primary point.** PASS at every batch tried (1, 4, 33, 86,
100, 5515 — the B=5515 numpy check passed this round), bitwise repeatable across fresh
processes, compute-sanitizer memcheck clean on both in-place (noscratch) B engines.
Range on the batch points is tuner-pick noise across create() runs (~1%); every pick in
the range is one of 2–3 equivalent cells.

### What changed, in order of measured effect

1. **Two-chain ILP reorder of every line engine (~25% at B_HBM by itself).** Every P/Q
   accumulation is now two independent FMA chains (j=1..6 seeded with x0, j=7..11 seeded
   with 0) summed at the end — 8 concurrent chains per k-pair instead of 4. The r1 code's
   best scratch-mode cells went from ~3400 µs to ~2470 µs with no other change. This was
   originally done for bit-exactness bookkeeping (see 3) and turned out to be the round's
   biggest single win: at 2.5–3 active warps/scheduler (r1's diagnosed concurrency wall),
   per-thread ILP substitutes for the warps the register file refuses to give. The extra
   4 accumulators fit: all coarse kernels still 128 regs / 0 spills.
2. **noscratch: out-as-intermediate, kernel B in place (+7% best-vs-best, and it swept
   the board — every top-20 tuner cell at every batch point is ns1).** Borrowed directly
   from L36_sharedtiled r1 ("scratch is strictly more L2 footprint", their 2339 vs 2118)
   and L64_radix8 r1 (in-place kernel_x). Kernel A writes `out`, kernel B transforms it
   in place: one L2 line per element instead of two. Safe because every B engine
   completes its line loads (into registers, or provably before the first cross-lane
   shuffle) before its first store. The fine B engine (12 threads/line in different
   blocks) cannot do this and keeps the tmp path.
3. **All engines bit-identical under one canonical accumulation order.** The tuner may
   pick different engines in different processes (tryout re-runs the whole binary), so
   coarse/split/fine/medium must agree to the last bit. The two-chain order is the
   contract; the split-k halves ARE the two chains, one per thread. FP addition is
   commutative (not associative), so `mine + __shfl_xor(other)` is exact on both sides.
4. **Chunk size is now sharp, and (chunk, nstr) gets a stage-2 refinement grid.** r1's
   tuner table was flat across chunks; with the faster kernels it is decisive:
   chunk 32 = 2312, 64 = 3016, 128 = 3277, unchunked = 3271 µs (winner shape, B=5515).
   Best: chunk 32, nstr 4 (in-flight mid footprint 4×32×190 KiB ≈ 24 MiB vs 40 MiB L2).
   nstr 3 chunk 48 ties; nstr 5–8 (NSTR raised to 8) buy nothing; nstr 2 loses ~4%.
5. **CUDA graph replay at B=1** (borrowed from L36_sharedtiled r1): captured at first
   execute, keyed on the (in,out) pointers, raced against plain launches in create()
   (19.5 vs 20.5 µs in-tuner). Worth ~0.5–1 µs on the driver clock. Cooperative fused
   kernel NOT tried — L36 already measured it losing to the graph; took their word.
6. **__ldcs on the B-pass intermediate read** (L64's "hints are the mechanism" lesson;
   the read is the line's last use before overwrite). Neutral-to-slightly-positive here
   (~0.5%, within noise) — kept because the ncu traffic numbers say it is the right
   intent and it can only matter more under pressure.
7. **uload (batched kernel-A load) re-raced at all batches** after the reorder changed
   the instruction mix: now ~par at B_HBM (r1: −11%), still tuner-decided. Occasionally
   picked (u1 cells within noise of u0).

### Built, measured, and rejected — with the numbers that killed them

* **Split-k kernel B (kernB2: 2 threads/line, half the j-folds each, 4 __shfl_xor(16)
  per k-pair, +~15% instructions).** The r1 agenda's #1 item. It works, it is exact,
  and it is ~2% SLOWER than coarse B (2354 vs 2312 µs at B_HBM): ptxas gives it **114
  registers, not the hoped ~76** — the compiler (correctly) batches all 12 line loads
  into registers for memory-level parallelism, so occupancy stays 2 blocks × 256 = 512
  threads/SM, the same ceiling as coarse, now with shuffle overhead on top. Kept in the
  race (it ties; on a different driver it might win), but the premise is dead: at 128
  regs the coarse engine's registers are all doing MLP work, not waste.
* **SPLITSQ (kernB2 with __launch_bounds__(256,4) forcing 64 regs): 2663 vs 2312 µs**
  (256 B spill stores / 424 B loads). Exactly the r1 96-reg experiment's verdict, at a
  milder setting. Not raced anymore; kernel kept for FORCE experiments.
* **Medium-grain unit-parallel kernels (kernAm/kernBm), the L45_pfa r1 lesson: fold
  phase (one u/d fold per thread, in place in shared) + pair phase (one output pair per
  thread reading pre-folded u/d from shared broadcast), coefficients staged to shared
  (multi-k constant reads serialize; shared multicast does not).** Landed at the target
  **32 registers, 0 spills → ~1440 theoretical threads/SM at only 1.08× FP — and lost
  anyway: kernAm 3355 vs 2303, kernBm 2686 vs 2303 µs.** The folded-dense form reads
  every input 11 times (once per k); coarse keeps the re-reads in registers for free,
  unit-parallel turns each into an LDS, and the LSU pipe (shared by LDS+LDG) becomes
  the wall — ~45 shared accesses per ~50 FP per pair thread. **The L45 unit-parallel
  lesson does NOT transfer to a dense matvec at high per-element reuse; it transfers to
  low-reuse codelet structures (PFA radix-5/9). Do not rediscover this.** Both kernels
  kept in the race (they lose honestly, ~50 cells of setup time).
* **Fine engines at scale** (raced at every batch this round, r1 only raced ≤32):
  fine-B 3875, fine-A 3276 vs 2306 µs at B_HBM — the 1.86× FP and 12× read
  amplification lose exactly as r1 predicted. Fine A+B remains the B=1 winner.
* **nstr > 4**: flat to worse everywhere (refine grid nstr 3–8 × chunk 12–64 measured;
  nstr 4 chunk 32 ≈ nstr 3 chunk 48 ≈ best, everything else worse).

### Cost accounting, honestly (ncu, forced winner config, B=512)

The noscratch+chunk scheme is **traffic-ideal**: kernA reads exactly the compulsory
6.37 MB/chunk from DRAM, kernB reads **14 KB** (L2 hit 96%) — the intermediate never
touches HBM even before pipelining. So at 0.41 µs/xform = 942 GB/s nominal (~47% of
realistic DRAM peak) against a ~0.21 µs 2-traffic floor, the entry is **still
latency/occupancy-bound, not bandwidth-bound**. Serialized, one 32-volume chunk is
kernA 21 µs + kernB 14 µs at ~9% warps active (a lone chunk's grid is far too small to
fill 108 SMs); the 4-stream chunk pipeline is what fills the machine (13.2 µs/chunk
effective, ~2.6 chunks concurrently resident). Remaining headroom to the traffic floor
is ~1.9×, all locked behind the 512-threads/SM register ceiling, and this round
measured both escape routes (fewer regs via shuffle split, fewer regs via shared
staging) as losers. The wall is real.

### Borrowed (attributions)

* **L36_sharedtiled gpu_r1**: out-as-intermediate (their failure #1 is why tmp loses),
  CUDA-graph-at-B=1 including the pointer-keyed recapture, and the warning off
  cooperative fused kernels.
* **L64_radix8 gpu_r1**: in-place final pass as the L2-chunk enabler, and the
  cache-hints-are-the-mechanism framing (__ldcs on last-use reads).
* **L45_pfa gpu_r1**: the unit-parallel idea (tested here, does not transfer — see
  above) and the "measure occupancy vs spills, don't guess" discipline.
* **CPU L23_rader r6** (my lineage): the arithmetic, untouched — 594 real FP
  instructions per line remains the count for the coarse engines; medium 1.08×,
  fine 1.86×, split ~1.15×.

### What I would do next (r3 agenda)

1. **The register ceiling needs a structural move, not a scheduling move.** Both
   in-kind escapes measured dead this round. The one untried direction with a different
   memory pattern entirely: **DMMA** (corpus §09 recommends 12×23 for L=23, pad
   23→24). The FLOP argument is irrelevant (we are nowhere near FP-bound); the reason
   to try it is that mma feeds operands from shared through the tensor pipe without
   per-thread fold registers — it attacks exactly this wall. Budget a real round;
   L13_dmma/L17_dmma records are the starting point.
2. **Chunk-shaped grids**: a lone chunk fills 9% of the machine; instead of more
   streams (measured flat), try z+y and x passes for DIFFERENT chunks fused into one
   grid (kernel launched over 2 chunks' A-work + previous chunk's B-work) to cut the
   pipeline's reliance on stream co-scheduling. Speculative; measure first whether the
   scheduler actually leaves SMs idle between stream launches (nsys, not ncu).
3. B=1 is within 0.6 µs of the two-kernel launch floor; leave it unless the graph
   misbehaves on the monitor's driver.
4. **Do not re-try**: unit-parallel/shared-staged line engines at any granularity
   (LSU wall, numbers above), register squeezes below 128 on the coarse engines
   (three data points now: 96→catastrophic, 85-equiv→−15%, 64→−15%), fine engines
   above B≈32, scratch-tmp intermediates, nstr > 4, chunks outside 24–48 at B_HBM.

## Round gpu_r4 (2026-08-22) — persistent producer/consumer ticket kernel

(r3 note: the leaderboard shows my r3 numbers identical to r2 — the file carried
through unchanged and L23_rader was not in the r3 promote list, so this round builds
directly on the r2 code.)

### Where it landed (leased A100-SXM4, same part as the monitor's node)

| case | this entry | r2 | cuFFT same lease | ratio | rel L2 |
|---|---|---|---|---|---|
| B=1 | **9.14 µs** | 10.4–11.0 | 15.1 µs | **1.65×** | 3.535e-16 |
| B=86 (L2 point) | **37.5–40.5 µs = 0.44–0.47 µs/xform** | 38.5–41.9 | 76.2 µs | **~1.9×** | 3.547e-16 |
| B=5515 (HBM, primary) | **1963–2001 µs = 0.356–0.363 µs/xform** | 2262–2286 | 5025–5137 µs | **~2.55×** | 3.547e-16 |

**−13% at the primary point vs r2.**  PASS at every batch tried (1, 4, 33, 73, 86,
100, 517, 5515 — and the B=5515 numpy check finally got its 1 GiB and PASSED this
round, so the primary point is now directly verified, not verified-by-proxy).
Bit-identical re-runs everywhere; compute-sanitizer **memcheck, synccheck AND
racecheck clean** on the persistent path (racecheck matters: the ticket broadcast is
a shared-memory protocol; see the double-buffer note below).

### What changed

1. **Persistent producer/consumer ticket kernel for the batch points — ported from
   L36_globalpass gpu_r2 via L45_pfa gpu_r3, stated plainly.**  One launch per
   execute, grid = one resident wave (occupancy-probed per shape in create()).
   Blocks pull tickets off a global atomic: a K1 ticket is one x-plane-group (the
   kernA body: P planes staged in shared, z then y, plain stores to out so the
   intermediate lands in L2), a K2 ticket is one tile of T x-lines transformed in
   place (the kernB body: __ldlu line loads, __stcs stores).  Dispatch order: LEAD
   volumes of K1 runway, then K1(v+LEAD) and K2(v) interleaved nA:nB by Bresenham,
   then a K2-only tail; per-volume done counters with the release
   (__threadfence + atomicAdd) / acquire (poll + __nanosleep + fence) pattern.
   Deadlock argument re-verified before porting (one resident wave + every K1(v)
   ticket dispatched before v's first K2 ticket).  Their noinline lesson held
   exactly: both bodies wrapped __device__ __noinline__ compile to 128 regs / 0
   spills under __launch_bounds__(128,4); nothing spilled at any probed shape.
2. **My one structural deviation from the L36 design: no epoch bookkeeping.**
   execute() resets the counters with one cudaMemsetAsync of 4·(B+1) bytes on the
   same stream before the launch (~22 KB at B_HBM, invisible next to 2 ms).  This
   keeps the kernel arguments constant across calls — so the persistent path stays
   CUDA-graph-capturable — and makes the tuner's copy-by-value plan trials unable
   to desynchronize an epoch (my time_case copies the plan struct; host-side epoch
   state in the copy would have been a hang waiting to happen).
3. **Single-barrier ticket protocol (new here, as far as I can tell from the
   records).**  Thread 0 grabs the ticket, decodes it, does the K2 dependency poll
   itself BEFORE the block-wide barrier (the other threads would only have been
   parked at a barrier anyway), and publishes the ticket into a parity-indexed
   double-buffered shared slot.  The double buffer is what makes the loop-top
   barrier removable: a laggard thread can still be reading slot i&1 only until it
   arrives at barrier i+1, and slot i&1 is next overwritten only after thread 0
   passes barrier i+1 — barrier-ordered both ways (racecheck agrees: 0 hazards).
   A K2 ticket now costs exactly ONE __syncthreads(); a K1 ticket four.  Barriers
   per 6K1+5K2 ticket group: 45 → 29.  Measured effect: ~1% (forced same-config
   2026.7 → 2023.4 µs; in-plan best 1994 → 1963–1976).  ncu had CTA-barrier waits
   at 35.7% of stall cycles, so this looked like a 10%-class win and was not:
   L36_globalpass's r2 warp-local postmortem ("the barrier stall was a symptom of
   memory latency — threads parked at the barrier while stragglers waited on
   memory") is confirmed from a second direction.  Keeping it: 1% is 1% and the
   protocol is simpler.
4. **LEAD is a plateau at ~ one wave, with an L2 cliff just past it** (B=5515,
   pc=0 = P4/T128/4-blocks, wave 432, forced-config sweep): 4→2382, 8→2329,
   12→2283, 16→2238, 24→2162, 32→2104, 48→2023, **56–88 → 2015–2040 (plateau;
   in-plan best 1963–1993 at lead 64–80)**, 96→2088, 128→2586, 176→3281,
   256→3391, 384→3393.  432/6 tickets = 72 volumes = exactly one wave of K1
   runway — L36_globalpass's lead=12=one-wave rule lands here at 72; past ~96 the
   runway intermediate (>18 MiB + streams) falls out of L2 and the far side is a
   flat 3.3 ms = the everything-through-HBM regime.  Tuner lead list now
   {4,8,12,16,24,32,48,64,72,80,96}.
5. **Block-shape race: fine tickets win, coarse tickets lose** (B=5515, in-plan,
   same tuner run): P4/T128/4blk 1993, P4/T128/3blk 2010, P2/T64/8blk 2047,
   P8/T256/2blk 2204, P8/T192/2blk 2154.  The two winners are near-tied and both
   get picked across processes (outputs bit-identical, so no repeatability issue).
   Coarse plane-group tickets lose to imbalance exactly like L45_pfa's 45×45 K2
   tile did ("coarser tickets lose more to imbalance than amortized staging
   saves").  A sixth shape P6/T144/3blk (96% K1 lane utilization, 432 thr/SM) was
   added late; it verifies correct, is raced by the tuner, and did not displace
   pc0/pc1 in the runs since.
6. **Tuner finals now exceed the 20 ms boost cliff** (the tuner-sample form of the
   brief's warning; lesson verbatim from L36_globalpass r2 item 3, seconded by
   L36_sharedtiled r3).  My r1–r3 tuner sampled ONE execute per event pair — 2.3 ms
   at B_HBM, deep in the clock-ramp artefact zone.  Stage 1/2 still rank cheaply,
   then the classic winner is re-timed with nin executes per sample (nin sized to
   ≥22 ms) and the persistent cells race at the same length — the classic-vs-
   persistent playoff is apples-to-apples.  The classic winner re-times at 2313 µs
   under long samples (vs its own r2-era in-plan claims of ~2280 with the pick
   made *and* measured on short ones).
7. **__ldlu replaces __ldcs on the dead intermediate read** in every B engine
   (kernB, kernB2, kernBm, and the persistent K2 body): the line is loaded once
   and overwritten in place, which is the exact "last use" semantic.  From
   L36_sharedtiled r2 / L45_pfa r2's CHUNKED hint vocabulary.  Within noise on its
   own; bitwise identical output; kept as the honest intent.
8. **The small points keep their r2 paths**: at B=86 the classic chunked+streams
   pipeline still wins (this round's picks: A(P=5,T=128) B-coarse TB=32 ns1
   chunk=48/4, 37.5–40.5 µs; the persistent kernel loses there in every tuner run,
   exactly as L36_sharedtiled r3 found at their B_L2), and at B=1 the fine-grained
   A+B pair behind a CUDA graph is untouched (9.14 µs, boost-state dependent).

### Operation count

Arithmetic untouched since CPU r6: folded-dense, 594 real FP instructions per
23-point line (484 FMA + 110 add/sub), 943k per volume, two passes.  Compulsory
global traffic unchanged; what changed is that the whole batch now flows through
ONE kernel with the live intermediate bounded at ~LEAD volumes (~14 MiB) by the
ticket schedule, no kernel-boundary drains, no stream co-scheduling dependence.

### Cost accounting, honestly (ncu, forced pc0 lead=72, B=517, one launch)

Occupancy 24.45% achieved vs 25% theoretical (register AND shared capped at
4×128-thread blocks/SM — the r1/r2 512-threads/SM wall, unmoved); issue 0.20/cycle
per scheduler, No-Eligible 79.9%; warp cycles/issued 19.9 of which 7.1 barrier
(pre-restructure).  **L2 throughput 78.7% is now the nearest roof** (DRAM 59.5%,
SM 62.7%) — the same end-state L45_pfa r3 reached ("the L2 pipe is now the nearest
roof").  At 0.356 µs/xform the entry runs ~1.07 TB/s of compulsory HBM traffic
against the ~0.21 µs 2-traffic floor: ~1.7× headroom, locked jointly behind the
register-file occupancy ceiling and, increasingly, the L2 pipe (4 L2 accesses per
point is structural for a two-pass form).

### Built, measured, and rejected — with the numbers that killed it

* **LEAD past the L2 cliff** (96–384): numbers in item 4.  The far-side plateau
  (~3.39 ms) is the intermediate-through-HBM regime; do not revisit.
* **Coarse-ticket shapes** (P8/T256, P8/T192): −8 to −11% vs pc0 at every lead
  tried.  Same lesson as L45_pfa r3's coarse-tile rejection.
* **The barrier-count attack as a headline win**: 45→29 barriers bought ~1%, not
  the 35%-of-stalls share ncu advertises.  Symptom, not cost.  (Kept — see item 3.)
* **Persistent at B=86 / B=1**: loses to the r2 chunked and graph paths in every
  tuner run; the tuner keeps racing it so this stays measured, not assumed.

### Borrowed, and from whom (this round is deliberately mostly borrowing)

* **L36_globalpass gpu_r2**: the entire persistent ticket design — ticket atomic,
  runway + Bresenham interleave, per-volume done counters with the fence/poll
  pattern, the one-resident-wave deadlock argument, lead≈one-wave — plus the
  ≥20 ms tuner-sample rule.  Their −13% at their HBM point predicted mine (−13.2%)
  as precisely as it predicted L45_pfa's (−13.7%).
* **L45_pfa gpu_r3**: the __noinline__-per-body register isolation (their 416 µs
  spill disaster avoided here for free), the "check -Xptxas -v before concluding
  the structure lost" discipline, and the coarse-ticket warning.
* **L36_sharedtiled gpu_r2/r3**: __ldlu on the dead intermediate read; the
  confirmation that ticket kernels lose at the L2-resident point; the second vote
  on long tuner samples.  Their carveout-50 lesson was READ and rejected with
  reason: my winning shapes need 102–135 KB shared/SM, which forces the 132/164 KB
  carveout anyway — there is no L1 to give back without dropping a block.
* **The memset-instead-of-epochs reset and the single-barrier double-buffered
  ticket broadcast are new here**, as far as I can tell from the records; both are
  free to take.

### What I would do next (r5 agenda)

1. **Two specialized co-resident persistent kernels** (K1-only + K2-only sharing
   the ticket queue on two streams) — the idea both L36_globalpass and L45_pfa
   list and neither has built.  For me the payoff would be a K2-only kernel with
   ZERO shared memory: it could run at pure register occupancy (4×256 threads/SM)
   while a K1-only kernel keeps its 4×128+shared shape — the static partition
   cannot load-balance, which is why it must be measured, not assumed.
2. **The L2 pipe at 79% is the new suspect.** Before any more warp-supply work,
   ncu the L2 sectors: if K2's 529-stride line loads are fighting K1's plane
   stores for the same slices, reordering K2 tiles to chase K1's plane order
   (ticket idx permutation — free in decode_ticket) might cut conflict misses.
3. **DMMA remains unexplored at L=23** (r2/r3 agenda item, still true, still the
   only untried different-memory-pattern formulation).  L13_dmma/L17_dmma records
   are the starting point; 23→24 pads at 1.09×.
4. **Do not re-try**: everything in the r2 list, plus leads past the L2 cliff,
   coarse plane-group tickets, and barrier-count reduction as a primary lever.
