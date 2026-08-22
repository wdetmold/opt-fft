# L45_pfa — strategy record

Geometry: **L = 45**, cube 45³ = 91,125 complex doubles per volume (1.39 MiB), forward,
unnormalised, out-of-place, batched, one A100. Implementation: `impl/L45_pfa.cu`.
Scored cases (cases.txt): B = 1, 11 (L2-resident), 736 (HBM, primary).
A volume does NOT fit one block's shared memory (1.42 MB vs 163 KB) — this geometry is a
memory-traffic and latency-hiding problem, not a fusion problem.

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique (final form)

**Two-pass Good–Thomas 9×5, both kernels unit-parallel, out as the intermediate:**

1. **K1** (grid B·45, 128 threads, one 45×45 (y,z)-plane per block, 31.6 KiB static
   shared): stage the contiguous plane in coalesced (register-batched, compile-time
   trip counts so all ~16 loads per thread issue back to back), then z-pass and
   y-pass in shared, then a coalesced store that folds the PFA output scramble into
   a shared-side gather. Two axes for one global read + one global write.
2. **K2** (grid B·64, 128 threads, tile = 45(x) × 32(flat yz), shared pitch 33 = odd,
   23.2 KiB → up to 7 blocks/SM): the x-pass on 32 lines per block, in place on `out`.
   No scratch buffer anywhere: K1 writes `out`, K2 transforms `out` in place — at
   B=11 the whole inter-kernel round trip stays in L2 by construction.

**The 45-point line = PFA 9×5 done IN PLACE in shared with the scrambled-slot trick**
(Burrus' classic PFA in-place property, worked out here from the maps): input map
n=(5a+9b)%45, output map k=(10k1+36k2)%45 kill all inter-stage twiddles; each DFT5
(fixed a=g) reads slots {(5g+9j)%45} and writes its 5 outputs back to those same slots;
each DFT9 (fixed k2=c) then reads {(5g+9c)%45} and writes back to the same set. X[k]
lands at slot sig(k) = (5·(k%9)+9·(k%5))%45, a fixed permutation folded into whichever
consumer reads it next. **No inter-stage temp, no padding: stride 45 is odd, so every
shared pattern here is bank-conflict-free** (measured: 23% residual conflict rate on K1
loads from mixed-g warp phases, 8% on stores — minor). The mod is one conditional
subtract (5g+9j ≤ 76).

**Unit-parallel compute is the round's central lesson**: one thread = one DFT5 or one
DFT9 of one line (405 then 225 units per plane-pass, strip-mined over 128 threads with
compile-time trip counts). The obvious thread-per-line form compiled to **>204
registers/thread → 4 blocks/SM → 9.9% occupancy → 4712 µs at B=736**; unit-parallel
dropped it to 96 regs, 20 warps/SM theoretical, 3752 µs immediately.

**Codelets**: folded real-coefficient DFT5/DFT9 (u/v fold, X_k = P_k ∓ iS_k), the same
family as every CPU-phase winner; cos/sin tables in `__constant__`, accessed at
unroll-time-uniform addresses (broadcast). The output permutation sig45() is computed
inline — a `__constant__ int[45]` table read at warp-divergent addresses serializes in
the constant cache; arithmetic is free.

**Regime split** (template `<DIRECT>` on both kernels, picked in create(): DIRECT=1 for
B ≤ 14, overridable via `FFT45_DIRECT`):
* DIRECT=1 (L2-resident batches): final DFT9 stage of BOTH kernels streams straight to
  global from registers — for fixed (k1,c) consecutive threads write consecutive
  addresses (K1's y-stage-2 units are indexed by true output kz working on column
  sig(kz), which makes the global side coalesced and puts the permutation on the
  shared side). K2's first stage also reads global directly (fixed slot, consecutive
  f → coalesced). Fewer shared round trips and barriers. Worth **−8% at B=11**.
* DIRECT=0 (chunked HBM batch): fully staged forms — each direct piece measured
  1.5–2% SLOWER at B=736, consistently.

**Chunked dual-stream pipeline** (lit 09 §2.4 structure 3): execute() walks the batch
in chunks of 12 volumes; K1s on stream A, K2s on stream B, K2_n waits K1_n via event,
next execute's K1_n waits previous K2_n (out region reuse). ncu on the chunked run:
**dram bytes = 20+22 MB per 12-volume chunk against 35 MB compulsory** — the K1→K2
round trip genuinely never leaves L2, i.e. HBM traffic is ~1.2× the read-once/
write-once floor instead of 2×. Honest caveat: **wall-clock is currently flat vs
unchunked (3362 vs 3361 µs)** because the kernels are latency-bound, not DRAM-bound —
the chunking is kept because it is free, it caps DRAM pressure on a shared-clock node,
and it is the right structure the moment occupancy improves. Cross-execute stream
pipelining does show up at B=1 (min 14.4 µs vs 16.8 single-stream-era).

### Operation count

Per 45-point line: 9 DFT5 (folded, ~56 flops) + 5 DFT9 (folded, ~186 flops) ≈ 1434
real flops ≈ 32 flop/point/axis ≈ **96 flop/point for the 3D transform** — against a
two-pass bandwidth budget of ~400 flop/point (64 B/point at balance 6.24), so
arithmetic is structurally free; every measured limit is memory latency. Global
traffic: 64 B/point through L2, ~35-42 B/point reaching HBM (chunked, measured).

### Measured (tryout.sh → leased A100-SXM4-40GB; rel L2 vs numpy, bit-identical re-runs)

| case | per transform | vs cuFFT same case | correctness |
|---|---|---|---|
| B=1 | **14.4–17.8 µs** (boost-clock noisy; best min 14.37) | 18.4 µs → **~1.25×** | 8.24e-16 |
| B=11 (L2) | **43.4 µs**/call = 3.95 µs/vol | 64.7 µs → **1.49×** | 8.21e-16 |
| B=736 (HBM, primary) | **3360.9 µs**/call = **4.57 µs/vol** | 4749 µs → **1.41×** | 8.21e-16 |

Progression at B=736: 4712 (thread-per-line) → 3752 (unit-parallel) → 3606 (unrolled
staging + inline sig45) → 3364 (K2 45×32 tile, 7 blocks/SM) → **3360.9** (final).
ncu final-ish state: K1 ~35 µs/chunk-wave, DRAM 26%/L1TEX 56%/SM 43%, achieved
occupancy 26% of 20-warp theoretical; K2 ~31 µs, DRAM 36%, occupancy 30% of 24.
Nothing saturated — the entry is **latency-bound at every scored point**, roofline
headroom ≈ 1.8× remains (est. ~2.5 µs/vol if the pipes could be kept fed).

### What did NOT work, with the numbers that killed it

* **Thread-per-line PFA (whole 45-line owned by one thread, staged via shared)**:
  >204 regs → Block Limit Registers = 4 → 9.9% occupancy → **4712 µs** vs 3752
  unit-parallel. The 45-complex intermediate simply does not fit a thread.
* **T1=128→160 (5 blocks, 81-reg cap)**: 3600 vs 3364 — spills eat the extra warps.
  T=128 at 96 regs is the sweet spot; T=64 (10 warps/SM) was the v1 regime, slow.
* **Warp-owns-lines with __syncwarp stage boundaries** (replaces 2 of 5 block
  barriers): **3854 vs 3367, −14%** — the block-wide unit pool load-balances better
  than per-warp ownership; barrier cost was not the binding term. Do not revisit.
* **Paired two-line DFT5 units** (rows r, r+23 → 10 independent loads, 2 FMA chains
  per thread): 3401 vs 3363 at B=736, 50.3 vs 48.5 at B=11 — more ILP per thread did
  not beat more (smaller) units.
* **cp.async (`__pipeline_memcpy_async`) staging in both kernels**: **3711 vs 3362,
  −10%** — nothing overlaps it in a load-once kernel, and the async path has worse
  standalone latency. Same conclusion L13_dmma reached; at L=45 it is actively bad.
* **Direct-store / direct-load variants at the HBM batch**: K2 final-stage direct
  store 3432 vs 3362; K2 stage-1 direct global load 3411 vs 3362. Both consistently
  ~2% worse chunked (yet clearly better L2-resident — hence the template split).
* **Chunk-size sweep 6–24 and unchunked at B=736**: all 3360–3368, flat. See caveat
  above — traffic is provably cut (ncu dram bytes), wall-clock is latency-bound.
* **Dual-stream overlap at B=736**: flat within noise (kept for B=1 and structure).
  The "one-wave-per-chunk lockstep burst" theory of the flat result was tested and
  is wrong — grids feed continuously across chunks; it is per-warp latency, not
  supply.
* **Constant-memory sig[45] table**: replaced by inline arithmetic (divergent
  constant-cache reads serialize); part of the 3752→3606 step together with the
  compile-time-trip staging loops (batched independent loads fixed K2's
  long_scoreboard = 11.2 cycles/issue, the largest single stall).

### Borrowed, and from whom

* **L13_dmma (gpu_r1)**: the corrected HBM peak (1555 GB/s, not 2.0 TB/s), the
  classic interleaved staging pattern (and its warning against fused strided global
  loads), the cp.async and __ldcs null results (cp.async re-tested here — worse),
  carveout=100 hint, and the general "measure with ncu, don't guess" discipline.
* **CPU L45_pfa / L45_mixedradix (geom panel_r6–r10)**: the PFA 9×5 factorization
  with zero inter-stage twiddles as the settled arithmetic for L=45, and the
  two-sweep pass structure (z+y fused, then x). The GPU plane-per-block K1 is the
  direct analogue of their phase-1 plane fusion.
* **Lit 09 §9.7**: opening structures 1+2 adopted as-is (plane-per-block two-pass +
  PFA codelets; "stride 45 is odd, no padding" confirmed); structure 3 (L2 chunking)
  implemented and measured — traffic claim confirmed, wall-clock benefit not yet.
* The in-place scrambled-slot PFA (no inter-stage temp) is standard Burrus/Temperton
  PFA lore, re-derived here for the (5,9) maps; I did not find it in another entry.

### What I would do next

1. **The entry is latency-bound at 26–30% achieved occupancy with every pipe under
   56%** — the ~1.8× headroom to the traffic roofs is all in warp supply. The
   K1 plane (31.6 KiB) pins 5 blocks/SM at 4 warps each; ideas not yet tried:
   (a) a 3-kernel split (z rows-tiled, y column-tiled, x) where every kernel gets
   6–7 blocks/SM — costs +2 L2 accesses/point, chunking keeps HBM flat, and the
   z-kernel's tiles have no barrier at all between its two stages if sized per-warp;
   (b) K1 with 2 planes/block + 256 threads was NOT tried (register math says 16
   warps/SM, worse — but the barrier amortization was never measured).
2. **B=1 is boost-clock noisy (sd 10–12%, min 14.4 vs median 18)** — worth checking
   what the monitor's 3-process scoring reports before optimizing it further; the
   two-launch structure costs ~4 µs of the 14.4.
3. If anyone wants the last 2% at B=736: the composite (K1-direct + K2-staged) and
   (K1-staged + K2-direct) cross terms were never measured separately — FFT45_DIRECT
   toggles both kernels together.
4. **For other large-L entries (L=36, L=64)**: the unit-parallel register lesson and
   the >204-reg thread-per-line failure transfer directly; so does the regime-split
   template (direct stores win L2-resident, staged wins chunked-HBM).
