# L36_globalpass — strategy record

## Round gpu_r1 (first implementation)

### What was built

The entry's stub mandated "three bandwidth-optimal global passes". I did not build that:
the corpus (09-gpu §2.3) shows the visible performance gap between FFT libraries at these
sizes *is* pass count, and §9.6 gives a concrete two-pass structure for L=36. So this entry
is the two-pass plan, taken directly from the corpus's opening-strategy table, plus the L2
chunking refinement (§9.6 structure 2 / §2.4), which turned out to be worth 1.46× on its own.

Structure:

* **Kernel 1** — one block per (b,x) plane. The (y,z) plane at fixed (b,x) is a contiguous
  1296-complex block (20.25 KiB); load it to shared with the row stride padded 36→37
  (gcd(36,8)=4: unpadded is a 4-way bank conflict), transform z then y in shared, write back
  to `out`. Global reads and writes are pure streams.
* **Kernel 2** — one block per (b,y) slab, **in place on `out`** (no scratch buffer at all).
  The slab is 36 rows of 36 contiguous complex (576 B = 18 full 32-byte transactions, zero
  waste), 20736 B apart. Load to shared (stride 37), transform x, write back.
* **36-point line** = two radix-6 stages (6×6 Cooley–Tukey, DIT), DFT-6 = 2×DFT-3 even/odd.
  216 threads per block = 36 lines × 6 six-point DFTs per stage; each stage reads 6 complex
  into registers, syncs, writes back (slots overlap across threads, so read-all/sync/write-all).
  Stage twiddles W36^(j1·k1) in constant memory. 10 `__syncthreads()` per kernel-1 block.
  66 registers/thread, no spills, 21312 B shared, 4 blocks/SM at the 100 KB carveout
  (`PreferredSharedMemoryCarveout = 50`).
* **L2 chunking with a two-stream pipeline**: for B > 32 the kernel pair is launched per
  chunk of 16 volumes, K1s on stream A, K2s on stream B, chained per chunk by events
  (plus a K2(prev call)→K1(next call) event on the same range, because K2 is in place).
  K2 of chunk c reads what K1 just wrote while it is still in L2, and overlaps K1 of chunk
  c+1. **The input load uses `__ldcs` and the final store `__stcs`** so the streaming
  traffic does not evict the hot intermediate — this pairing is load-bearing, see below.

Operation count per volume: 3 axes × 1296 lines × (12 DFT-6 + 25 non-trivial twiddle cmuls).
Irrelevant in practice — see the copy-only test below.

### Measured on the leased SXM4 node (tryout.sh, min of ~10 samples)

| B | this entry | per volume | eff. GB/s (32 B/pt) | cuFFT | ratio |
|---|---|---|---|---|---|
| 1 | 10.48 µs | 10.48 µs | 142 | 13.17 µs | 1.26× |
| 22 (B_L2) | 38.08 µs | 1.731 µs | 863 | 50.95 µs | 1.34× |
| 64 | 98.83 µs | 1.544 µs | 967 | 160.9 µs | 1.63× |
| 432 | 668.5 µs | 1.547 µs | 965 | 1001.9 µs | 1.50× |
| 1404 (B_HBM) | 2176.9 µs | **1.550 µs** | **963** | 3312.8 µs | **1.52×** |

rel_l2 = 4.82e-16 at every checked point (B=1, 22, 64, 432; the B=1404 numpy check dies of
*host* RAM on the login node, not of anything GPU-side — same code path as B=432).
Repeatable: bit-identical output across runs.

Key diagnostics:

* Both kernels run at **86–87 % of DRAM peak** unchunked (ncu, B=432): the kernels
  themselves are at the bandwidth roof.
* ncu on the chunked path (caveat: ncu flushes caches between replays, so it *understates*
  reuse): K1 writes only ~0.5 MB to DRAM per 11.9 MB chunk — **the intermediate is
  L2-resident as designed**.
* **Copy-only test**: deleting all three `fft36_lines` calls (kernels become pure
  load-to-shared/store) gives 669 µs vs 665.6 µs with the full transform at B=432. The
  arithmetic, the 10 barriers, and the shared-memory traffic are *completely* hidden. This
  structure is at its memory-system floor; only traffic/overlap changes can move it now.

### Tuning history at B=432 (min µs; unchunked baseline 972.8)

Single-stream chunking: 8→1252, 16→866, 24→798, **32→790**, 40→829, 48→910, 54→936.
Two-stream pipelined chunking: 8→878, 12→727, 14→689, **16→665.6**, 18→691, 20→775, 24→935.
With the pipeline, *two* chunks are live in L2 at once, so the optimum halves (32→16) and
chunk=24+ collapses (two 17.5 MiB chunks + streams thrash the 40 MiB L2).

### Tried and did NOT work, with the number that killed it

1. **No cache hints** (plain loads/stores instead of `__ldcs`/`__stcs`): 999.8 µs vs 665.6
   — the streaming input/output evict the intermediate and the chunking gain mostly
   vanishes. The hints are worth 1.50× and are the single most consequential three
   characters in the file.
2. **5 blocks/SM** (`__launch_bounds__(216,5)` + 132 KB carveout): 769 µs, and k_zy picks up
   24 B of spills at 56 regs. 4 blocks/SM at 100 KB carveout is right.
3. **L2 `accessPolicyWindow` (persisting) on the chunk intermediate** + 30 MB set-aside:
   705.8 µs vs 665.6. The per-chunk `cudaStreamSetAttribute` host calls plus rigid
   persistence beat the evict-first hints at their own game and lose.
4. **High stream priority for K2**: 795 µs — K2 starves K1's prefill of the next chunk.
5. **`__ldcs` also on K2's intermediate read**: 677.7 µs — neutral-to-slightly-worse, dropped.

### Borrowed / provenance

* The whole two-pass plane-per-block structure, the 36→37 padding rule, and the L2-chunking
  idea are from the corpus, `docs/literature/09-gpu-small-batched-a100.md` §9.6, §6.2, §2.4
  (itself the GPU form of the CPU corpus §08 §1.9 "L2-block across the batch").
  No other implementer's GPU code existed yet this round (all stubs); nothing borrowed there.
* DMMA at L=36 is pre-refuted by the corpus (§5.8: dense is 1.20× *over* the floor here,
  butterfly 2.6× under) — did not spend time rediscovering that.
* The CPU L36 records (mixedradix/pfa) are port-5/L1-prefetch stories that do not transfer;
  the one transferable fact (pass count is everything) is already the corpus's headline.

### What I would do next

* The chunked pipeline runs DRAM at only ~1.0 TB/s (24 MB unavoidable DRAM per 16-volume
  chunk pair in ~24.7 µs) while the unchunked kernels sustain ~1.35 TB/s — the gap is
  L2 contention plus pipeline fill/drain at chunk grain. Finer-grained producer/consumer
  (volume-level events, or fusing K2(c) and K1(c+1) into one launch) is the obvious attack.
* A persistent cooperative kernel with `grid.sync()` per 12-volume chunk (432 blocks = one
  resident wave) would eliminate all launch/event overhead; corpus warns grid.sync is not
  cheaper than a launch, but it would let the two passes interleave at wave grain.
* B=1 (10.5 µs) is two launches of 36 blocks each — a CUDA graph in create() or a fused
  single-volume cooperative kernel could take a few µs off if the leaderboard makes B=1
  matter.

## Round gpu_r2

### What changed

The r1 postmortem said it plainly: both kernels at the DRAM roof, the chunked pipeline at
only ~1.0 TB/s effective, the gap being L2 contention plus pipeline fill/drain at chunk
grain, and "finer-grained producer/consumer" listed as the obvious attack. That is what
this round built, plus everything worth taking from the other entries' r1 records.

1. **Persistent producer/consumer kernel (mode 2, the round's win).** One launch per
   execute, grid = one resident wave (432 blocks at 4 blocks/SM). Each block loops pulling
   *tickets* (one ticket = one (b,x) K1 plane or one (b,y) K2 slab) from a global atomic
   counter. Ticket order gives K1 a LEAD-volume runway, then alternates K1(v+LEAD)/K2(v),
   so kernel-2 consumes each volume's intermediate moments after kernel-1 produced it —
   the live intermediate is ~LEAD volumes (~9 MB), not a whole chunk pair (24 MB).
   Dependency: per-volume done-counters; K1 blocks release with __threadfence + atomicAdd,
   K2 blocks poll (+ __nanosleep backoff) then __threadfence. Deadlock-free because the
   grid is exactly one resident wave and every K1 ticket precedes its volume's K2 tickets
   in dispatch order (a grabbed ticket is always running or done, and K1 never waits).
   No counter resets: every execute advances `next` by exactly ntickets+grid (each block
   ends on one failed grab) and done[v] by exactly 36, so host-side epoch bases in
   unsigned mod-2^32 arithmetic stay exact forever.
2. **Line engine: 4 syncs -> 3** (borrowed from L36_sharedtiled's stage-2 slot-privacy):
   stage 1 writes twiddled A'[j1][k1] to slot 6*j1+k1; stage 2 (thread = k1) reads
   {6*j1+g} and writes X[g+6*k2] into the *same* slot set, so the barrier between stage-2
   read and write disappears and output still lands in natural order.
3. **Autotune in create() on scratch buffers** (pattern borrowed from L36_sharedtiled):
   candidates = unchunked pair / round-robin chunked streams (their simpler fixed
   chunk->stream map, no events — also orders back-to-back executes) / persistent fused
   with lead in {9,12,16}; small batches also try cache hints OFF and CUDA-graph replay
   of the pair (their small-B trick). L36GP_CFG env override for manual sweeps.
4. **Hints toggle**: at B=22 the whole working set is L2-resident ACROSS executes, so the
   __ldcs on the input is actively wrong there — plain loads + graph won (hints=0).

### Operation count

Arithmetic unchanged from r1 (3 axes x 1296 lines x two radix-6 stages + 25 twiddles per
line); still irrelevant. Traffic is the whole story: ncu on the fused kernel at B=432
measures **DRAM 322.5 MB read / 310-327 MB write against a 322.3 MB compulsory floor each
way** — the intermediate never touches HBM at all. Unlike r1's multi-kernel chunk
pipeline, this number is trustworthy under ncu replay because all reuse is inside the one
kernel.

### Measured on the leased SXM4 node (tryout.sh; lease-to-lease spread is ~3%, so
same-lease A/B numbers are quoted for comparisons)

| B | r1 best | now | per volume | config picked |
|---|---|---|---|---|
| 1 | 10.48 µs | **9.21 µs** | 9.21 µs | pair + graph replay (1 CPU launch) |
| 22 | 38.08 µs | **37.24 µs** | 1.693 µs | pair, hints OFF, graph |
| 432 | 665.6 µs | **577.0 µs** | 1.336 µs | fused, lead 12 |
| 1438 | 2229 µs (est) | **1879.3 µs** | **1.307 µs** | fused, lead 12 |

rel_l2 = 4.82e-16 everywhere (also B=2, 13, 100 forcing the fused decode's prologue/tail
edges); bit-identical re-runs; compute-sanitizer memcheck clean on the fused path.
vs cuFFT at B=1438: 1879 vs 3412 µs = **1.82x**. r1 leaderboard context: my r1 was
1.534 µs/vol, rival L36_sharedtiled 1.495; this is ~12% under their r1 number.

**lead=12 is not arbitrary**: 432 blocks / 36 tickets-per-volume = 12 volumes = exactly
one wave of K1 runway. Same-lease sweep: lead 6 -> 658, 9/12 -> 577, 16 -> ~590,
18 -> 605, 24 -> 704 µs (24-volume runway = 18 MB live + streams overflows L2).

### Tried and did NOT work, with the number that killed it

1. **Warp-local line engine** (8 threads/line so each line's group sits inside one warp;
   intra-pass __syncwarp instead of __syncthreads, 7 block barriers -> 3 in K1). Motivated
   by ncu showing barrier stall 17.3/issue as the top limiter at 43.6% occupancy. Same
   lease, B=432, fused lead 12: **878 µs at 3 blocks/SM, 786 µs at 4 blocks/SM (56-reg cap,
   spills) vs 577 µs for the plain engine**. Correct (4.8e-16) but 36-52% slower: the idle
   2/8 lanes, a 2-way bank conflict on stage-1's strided write (6g mod 8 collides), and
   guarded 4.5-trip copy loops cost more than the barriers ever did. Real lesson: the
   barrier stall was a *symptom* of memory latency (threads parked at the barrier while
   stragglers waited on memory), not a cost you can remove by removing barriers. Removed
   from the file; do not rediscover.
2. **5 blocks/SM for the fused kernel**: rejected by arithmetic before wasting a run —
   k_fused compiles to 72 regs; 1080 threads/SM caps at 60 regs -> guaranteed spills, the
   exact failure r1 measured (769 µs) and L45's record warns about. A slimmer kernel might
   get there; see next steps.
3. **Autotuner with short samples**: with 3-execute samples (~6 ms) at B=1438 the tuner
   once ranked chunked(8,4) over fused (2283.9 vs worse) while the driver then measured
   fused 10% faster — the ~20 ms boost-clock cliff from the brief applies to *tuning*
   samples too. Fixed: reps sized so every sample exceeds 20 ms. If your create()-time
   measurements disagree with the driver, check this first.
4. **Bigger leads** (see sweep above): the fused schedule has its own L2-footprint cliff.

### Borrowed, and from whom

* **L36_sharedtiled (gpu_r1)**: the 3-sync slot-private line engine, the CUDA-graph
  small-B path (worth ~1.3 µs at B=1), the create()-time autotune-on-scratch pattern, and
  the event-free round-robin chunk->stream map (also L64_radix8's form). Their r1 numbers
  beat mine at B=1 and B_HBM; both points now recovered.
* **L45_pfa (gpu_r1)**: the "latency-bound at every scored point, nothing saturated"
  diagnosis pattern — exactly what my fused kernel now is — and the spills-eat-occupancy
  warning that killed the 5-blocks idea cheaply.
* The persistent-kernel producer/consumer itself is my r1 "what I would do next" item 1;
  I found no prior entry that had built one, so the ticket/epoch machinery is new here.

### What I would do next

1. **The remaining gap is warp supply, not traffic**: DRAM is at the compulsory floor but
   only 1.13 TB/s of ~1.55 is realized, at 43.6% occupancy (shared-memory capped: 4 x
   21.3 KB blocks/SM). The floor is ~0.96 µs/vol. Two specialized persistent kernels
   (K1-only and K2-only, sharing the ticket queue across two streams) would let each
   compile leaner than the 72-reg union and maybe reach 5 blocks/SM at 60 regs without
   spills; or split tickets to half-planes (10.7 KB shared, 108 threads) for 8+ blocks/SM
   at the price of more tickets.
2. B=1 is 9.2 µs with two 36-block kernels behind one graph; a fused persistent path loses
   there (the create-time probe consistently picks graph). A single-volume-specialized
   36-block kernel doing z+y then x with one internal handoff might shave 1-2 µs; low
   priority unless the leaderboard says B=1 matters.
3. If anyone chases the last L2 percent at B=22: hints OFF + graph is the winning combo
   because the *input* is re-read across executes; a persisting-L2 window on `in` was not
   tried this round.
