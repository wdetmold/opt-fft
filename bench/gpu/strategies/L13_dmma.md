# L13_dmma — strategy record

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (35,152 B), forward,
unnormalised, out-of-place, batched, one A100. Implementation: `impl/L13_dmma.cu`.
Scored cases (cases.txt): B = 1, 477 (L2-resident), 30549 (HBM, primary).

---

## Round gpu_r1 (2026-08-22) — first implementation

### Technique

Fused single kernel, **one volume per block in shared memory, one thread per 13-point
line, three axis passes, one global read + one global write**:

1. Stage the volume into shared with the classic interleaved-coalesced pattern:
   169 threads × 13 fully-unrolled iterations (169·13 = 2197 exactly, so no bounds
   check and 13 independent 16-byte loads in flight per thread).
2. z-pass (stride 1), y-pass (stride 13), x-pass (stride 169), separated by
   `__syncthreads()`; each of 169 threads owns one whole line in registers.
   The x-pass writes its results **directly to global** (for fixed output index k,
   consecutive threads hit consecutive complex doubles — coalesced), saving one full
   shared round trip.
3. The line transform is the **conjugate-symmetric folded dense matvec** — adopted
   wholesale from the CPU phase's `L13_direct` winner (geom panel_r6), which itself
   derives from `L17_matrixsimd`; also literature 09 §9.3 structures 1+3:

   ```
   u_j = x_j + x_{13-j},  v_j = x_j - x_{13-j}          (j = 1..6)
   P_k = x_0 + Σ_j cos(2π kj/13) u_j
   S_k =       Σ_j sin(2π kj/13) v_j                    (k = 1..6)
   X_0 = x_0 + Σ u_j,   X_k = P_k - i·S_k,   X_{13-k} = P_k + i·S_k
   ```

   All coefficients real; the 6×6 cos/sin tables live in `__constant__` (uniform
   access per fully-unrolled (k,j) step → broadcast, free). Folding on load keeps
   only u, v, x0 and four accumulators live.

Block shape: 192 threads (6 warps; 169 compute). **80 registers/thread, zero spills**,
static shared 35,152 B → **4 blocks/SM** (register- and shared-capped simultaneously;
5 blocks would need 176 KB > the 164 KB carveout). `__launch_bounds__(192, 4)` plus a
`PreferredSharedMemoryCarveout = 100` hint in create(). Stride 13 is odd, so every
pass is bank-conflict-free with no padding (lit 09 §6.2 — the odd-L gift).

### Operation count

Per 13-point line: 12 complex fold add/sub (24 real ops) + X0 sum (12) + 6×(24 FMA +
4 add/sub) = **~204 FP instructions ≈ 350 real flops per line**, i.e. ~27 flop/point
per axis, **~81 flop/point for the full 3D transform** — against 936 for the naive
dense complex matvec and 55.5 for a textbook butterfly. Arithmetic intensity
81/32 ≈ 2.5 flop/B against the 6.24 machine balance: comfortably under the HBM floor
on the **vanilla** FP64 pipe, which is why DMMA was not needed (below).

### Measured (tryout.sh → leased A100-SXM4-40GB of the reserved node)

| case | per transform | GB/s | cuFFT same case | speedup |
|---|---|---|---|---|
| B=1 | **7.88 µs**/call | — | 12.14 µs | 1.54× |
| B=477 (L2) | **59.5 ns** (28.39 µs/call) | 1181 | 63.5 µs/call | 2.24× |
| B=30549 (HBM, primary) | **51.2 ns** (1564.0 µs/call) | **1373** | 4731.5 µs | **3.03×** |

Correctness: rel L2 = 3.2e-16 at every batch (B=30549 checked against numpy on 400
sampled volumes; check.py itself cannot allocate 1 GiB on the login node). Bit-identical
across runs.

ncu on the HBM case (earlier build at 1585 µs): **dram__throughput 86.7% of peak,
dram__bytes 2.13 GB = exactly the read-once/write-once minimum**, FP64 pipe 60%,
L2 70%, L1 48%, achieved occupancy 36%. The final build (1564 µs) is ≈ **88% of DRAM
peak moving minimum bytes** — the kernel is at the bandwidth roof; VkFFT's published
best-in-class band is ~82–84%.

**Hardware note that changes the roofline math for everyone:** the brief's table says
the SXM4 node has "~2.0 TB/s". The leaderboard header shows 1215 MHz memory clock →
5120-bit × 2 × 1.215 GHz / 8 = **1555 GB/s**, and ncu agrees (1355 GB/s reported as
86.7% of peak ⇒ peak ≈ 1563). The **A100-SXM4-40GB** is a 1555 GB/s part like the
PCIe card; only the 80 GB SXM4 has 2.0 TB/s. The lit 09 floor table (45.2 ns per
volume at L=13) is therefore the right target here, and we sit at 1.13× of it.

### What did NOT work, with the numbers that killed it

* **Fusing the z-pass into the global load** (each thread reads its own contiguous
  208 B z-line, transforms, writes shared — saves one barrier and all staging):
  **1244 GB/s vs 1355** at B_HBM, 8% slower end-to-end. Per-instruction the 13-complex
  stride-208 pattern uses 16 B of every 32 B sector and relies on L1 to merge the
  halves; with the 164 KB shared carveout L1 is only 28 KB and it does not hold.
  Confirms lit 09 §6.1: keep the classic interleaved pattern for the global side and
  do all transposition in shared.
* **`cp.async` (`__pipeline_memcpy_async`) for the staging loop**: 1563.6 µs vs
  1562.5 — exactly neutral (nothing to overlap in a load-once kernel; lit 09 §6.3
  predicted this). Dropped for simplicity.
* **`__ldcs` (evict-first) on the staging loads**: 1645 µs vs 1562 — 5% *worse*.
* **A specialised B=1 kernel** (384 threads, each line's outputs split across two
  threads 192 apart so the halves live in different warps; results parked in
  registers across a compute/store barrier): **11.0 µs vs 7.9 µs** — the three extra
  barriers cost more than the six extra warps bought. The plain fused kernel stays
  the only kernel.

### Why the entry's namesake (DMMA) is deferred, deliberately

The measured limiter at the primary point is DRAM at ~88% of peak with **minimum
bytes moved** — there is at most ~12% left in the whole kernel and none of it is
arithmetic (FP64 pipe: 60%). The folded matvec is already at 0.42× of the flop
budget the bandwidth floor allows on the *vanilla* pipe, so moving the same flops to
tensor cores cannot move the time. DMMA would matter only if (a) the B_L2 regime
(where DRAM is 34% and the kernel is latency/barrier-bound — FP64 42%) can be
restructured so arithmetic becomes the limiter, or (b) a future variant needs the
shared-memory *read* traffic reduction that DMMA's operand reuse gives (lit 09 §5.5).
Neither is true of this round's kernel. This is a measured decision, not an evasion:
see the ncu numbers above.

### What I would do next

1. **B_L2 (477) is the soft spot**: 59.5 ns against 51.2 at B_HBM even though the
   data is L2-resident (7.2 TB/s available). ncu shows nothing saturated — stalls are
   long-scoreboard (7.7/warp) and barrier (6.1/warp), plus a 1.10-wave tail
   (477 blocks over 432 resident). Ideas: 2 volumes per block only for mid-size
   batches (halves barrier count per volume), or a persistent-block form.
2. Try padding the shared array to break the z-pass's *phase* pattern further — not
   expected to matter (no conflicts measured), but cheap to A/B under ncu's
   `l1tex__data_bank_conflicts` metric.
3. B=1 at 7.9 µs is ~60% kernel, ~40% launch; the kernel path is latency-bound with
   6 warps on one SM. The two-half split failed; a 4-way split with `__syncwarp`
   choreography inside one warp-pair might do better, but the point is scored as
   launch overhead and cuFFT is already 1.54× behind.
4. If anyone wants the DMMA experiment: do it at **B_L2**, as three back-to-back
   real GEMMs (9×13 cos / sin against 13×169·B panels), and compare against this
   entry's 59.5 ns — that is the number to beat, not the HBM one.
