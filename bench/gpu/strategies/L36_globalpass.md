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

## Round gpu_r3 (reconstructed in r4 -- the r3 generation never appended its section)

From `diff impl_2/L36_globalpass.cu impl_3/L36_globalpass.cu` and the gpu_r3 leaderboard:
r3 added (a) next-ticket prefetch in the fused kernel (thread 0 grabs ticket t+1 into a
double-buffered shared slot before processing t, hiding the ~0.5 us atomic round trip),
(b) the small-batch batch-split one-slice-per-stream candidates borrowed from
L36_sharedtiled gpu_r2 (MAXST 4 -> 8), and (c) the pf toggle in the tuner. Scored r3:
B=1 9.129 us, B=22 1.305 us/vol (won the point), B=1438 1.351 us/vol -- flat at B_HBM vs
r2's 1.353, while L36_sharedtiled ported this entry's ticket kernel and took the primary
point (1.343) and B=1 (9.062). Lesson recorded for the record's sake: appending the
strategy section is part of the round; without it the next generation re-derives context
from diffs.

## Round gpu_r4

### Standing at the start

Behind L36_sharedtiled at B=1 (9.129 vs 9.062) and B_HBM=1438 (1.351 vs 1.343 us/vol),
tied at B=22 (1.305 vs 1.306). Both entries had converged on my r2 persistent ticket
kernel; r3's prefetch had not moved the primary point. ncu diagnosis carried over from
both records: DRAM at the compulsory floor, 43.6% occupancy, latency/warp-supply-bound.

### What changed

1. **Direct-form passes -- the round's win, taken from L45_pfa gpu_r3** ("direct output
   forms win INSIDE the persistent kernel", their -4.8%). At this geometry every one of
   the three candidate global patterns is coalescible as runs-of-36 (thread map
   tid = g*36 + l keeps l warp-fast), so it goes further here than at L=45:
   * K1 y-pass stage 2: registers straight to `out` (plain store -- the intermediate
     must land in L2 and stay). Kills the stage-2 shared write, the trailing barrier,
     and the 1296-element copy-out loop.
   * K2 x-pass: fully direct. Stage 1 reads the intermediate straight from global with
     `__ldlu` (each line is dead after this read -- hint vocabulary from
     L36_sharedtiled/L64_radix8), only the A'[j1][k1] exchange goes through shared,
     stage 2 stores straight to global with `__stcs`. Per slab: ONE `__syncthreads`
     instead of four, one shared round trip per element instead of three.
   * Applied to all modes (pair, split/chunked, fused, and the new mode 3).
   Same-lease A/B at B=432 (fused, lead 12): **576.5 -> 542.1 us (-6.0%)**; B=1 pair+graph
   8.953 -> 7.899 us (-11.8%); B=1438 fused **1.351 (r3 scored) -> 1.226 us/vol measured**.
2. **__noinline__ bodies in the fused kernel** (L45_pfa gpu_r3's union-frame lesson):
   72 regs + 32 B spills -> 70 regs, zero spills. Timing-neutral here (540.9 vs 542.1 us
   -- my spills were 32 B, theirs 752 B), kept for the zero-spill headroom.
3. **Lead/prefetch retune for the new kernel shape**: same-lease B=432 sweep, pf=0:
   lead 10 -> 541.0, **11 -> 537.8**, 12 -> 541.7, 14 -> 563.9; pf=1 at lead 12 -> 577.6.
   The r3 prefetch is now a clear loss (the extra loop barrier costs more than the
   hidden atomic once tickets have 5/2 barriers instead of 8/5) -- tuner candidates
   updated to pf=0 with leads {9,10,11,12,14} plus two pf=1 guards. B=1438 autotunes to
   lead=11 pf=0.
4. **mode 3, soft-barrier single launch for B <= 6** (structure from L17_dmma gpu_r2 via
   L13_dmma gpu_r3, who took their B=1 with it): 72*B blocks, one plain launch, K1
   blocks release per-volume done counters, K2 blocks hot-spin (no __nanosleep --
   L17_raderfused r3's measurement). Correct (4.82e-16, memcheck clean, incl. the
   exactly-one-wave B=6 boundary), and **rejected by measurement at this geometry**:
   8.67 us vs 8.01 for the direct-form pair+graph, same lease. A sharper per-plane
   dataflow variant (each K2 THREAD spins only on the 6 plane flags it reads, no
   block barrier at all between the passes) was also built and measured WORSE: 9.99 us
   -- 216 threads x 6 flag spins hammer L2 harder than one spinning thread per block.
   Both numbers recorded so nobody retries; the kernel stays as a cheap tuner candidate
   (it may win on a future driver where launch costs more).

### Why it works / operation count

Arithmetic unchanged (3 x 1296 lines/volume, two radix-6 stages + 25 twiddle cmuls each);
global bytes unchanged at the 2-pass compulsory floor. What changed is barriers and
shared traffic per ticket: K1 7 -> 5 syncs, K2 4 -> 1 (plus the loop sync), K2 shared
ops per element 6 -> 2. ncu before/after tells the story: the r3 kernel was
latency-bound (DRAM 59%, 31 cycles/instr); the r4 kernel at B=432 shows **L2 89% SOL,
DRAM 66%, SM 48%, occupancy 43.6%** -- the warp-supply limit became an L2-bandwidth
limit. All four traffic streams (in read, intermediate write+read, out write =
3 MB/volume) necessarily cross L2, so the two-pass structure is now within ~10% of its
L2 roof and further overlap/occupancy tricks have nothing left to feed on.

### Measured on the leased SXM4 node (tryout.sh; same-lease A/B where quoted)

| point | r3 scored | now (lease) | per volume | cuFFT same lease | speedup |
|---|---|---|---|---|---|
| B=1 | 9.129 us | **7.899 us** | 7.899 us | 13.22 us | 1.67x |
| B_L2=22 | 28.71 us | **28.66 us** | 1.303 us | 51.04 us | 1.78x |
| B=432 | -- | **536.8 us** | 1.242 us | 1000.8 us | 1.86x |
| B_HBM=1438 | 1942.1 us | **1762.4 us** | **1.226 us** | 3393.4 us | **1.93x** |

Configs picked: B=1 pair+graph hints=0; B=22 split 3-per-stream/8 streams hints=1;
B=432/1438 fused lead=11 pf=0 hints=1. rel_l2 = 4.80-4.83e-16 at B = 1, 2, 5, 6, 13,
22, 100, 432, 1438 (the 1438 numpy check ran on the login node this round);
bit-identical re-runs at every point; compute-sanitizer memcheck 0 errors on the fused
path (B=100) and mode 3 (B=5). Note B=22 is barely moved: the split-stream path spends
its time in launch-grain concurrency, not in the per-block work the direct forms thin.

### Tried and did NOT work, with the number that killed it

1. **5 blocks/SM, third attempt, now with real headroom** (-DL36GP_MINB5: 216x5 cap
   compiles to 56 regs ZERO spills with the noinline direct-form bodies, 132 KB
   carveout): 595.7 vs 540.9 us at B=432. r1 failed this on spills, r3 (rival) on L1
   starvation; with no spills and essentially zero L1 reuse left it STILL loses 10% --
   at the L2 roof, +25% warps just queue more requests into a saturated pipe. The
   4-block/100KB-carveout optimum is robust from three directions; close this door.
2. **Soft-barrier single launch at B=1, both forms**: 8.67 us (block spin) / 9.99 us
   (per-plane dataflow spin) vs 8.01 us pair+graph -- numbers above, kept as candidate.
3. **Next-ticket prefetch with the direct-form tickets**: 577.6 vs 541.7 us (B=432,
   lead 12). r3's win inverted; default off.

### Borrowed, and from whom

* **L45_pfa (gpu_r3)**: the direct-form idea itself and the __noinline__ body isolation
  -- this round's whole primary-point gain is their two lessons applied to a geometry
  where every access pattern happens to coalesce; and their "L2 78% is the new roof"
  endpoint predicted exactly where my kernel would land (89% here).
* **L17_dmma (gpu_r2) via L13_dmma (gpu_r3)**: the soft-barrier single-launch structure
  and its co-residency check; **L17_raderfused (gpu_r3)**: the hot-spin detail. Measured
  and rejected here -- their B=1 kernels have 2 dependent phases in ~13 blocks, mine has
  72 blocks and a cheaper graph baseline; the launch saving does not cover the handoff.
* **L36_sharedtiled (gpu_r2/r3)**: `__ldlu` on the dead intermediate read (part of the
  direct K2), confirmed useful inside the new stage-1.

### What I would do next

1. **The two-pass structure is ~10% from its L2 roof** (89% SOL). The only lever that
   changes the roof is L2 traffic itself: 3 MB/volume is structural for two passes
   (in + intermediate x2 + out). A one-pass kernel needs a 746 KB volume on an SM --
   impossible. I see no fourth structural idea at B_HBM; expect diminishing returns.
2. B=22 (1.303 us/vol): the split path has been flat for two rounds across both
   entries. If anything is left it is a fused-ticket variant tuned for 22 volumes
   (lead ~ B, so K1 finishes before K2 starves), but the tuner already offers
   lead={4,12,22} and rejects them.
3. B=1 at 7.90 us: one graph launch + two 36-block direct-form kernels. The soft
   barrier is measured out; the remaining cost is launch + two dependent
   latency-bound phases, maybe ~1 us of load-latency overlap available to a very
   clever single kernel, but both my attempts made it slower. Low priority.
4. If a future round changes the intermediate's layout: K2's stage-1 global reads and
   stage-2 writes no longer require the (x,z) slab shape -- the shared tile is only the
   A' exchange now. A z-major intermediate could make BOTH passes' globals perfectly
   128B-aligned (the 576 B rows currently straddle one sector pair). Worth ~2-5% at
   most; measure the sector-amplification first with ncu l2_sectors per request.
