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
