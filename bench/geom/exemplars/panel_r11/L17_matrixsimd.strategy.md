# L17_matrixsimd — strategy record

Geometry: **L = 17**, cube 17³ = 4913 complex doubles per volume, forward, unnormalised,
out-of-place, batched, single-threaded.
Implementation: `impl/L17_matrixsimd.c`. `fft3d_name()` → `L17_matrixsimd`.

Assignment: the contrarian entry — *dense 17×17 matrix, maximally vectorised*. More
arithmetic than Rader, but perfect data flow: no permutations, no index tables, no
cross-lane operations, the matrix resident in L1. Report the honest comparison of
arithmetic count versus achieved time.

---

## Round 1 (2026-08-21)

### Technique

Row-column: one dense length-17 DFT matrix applied along each axis, three passes.
The matrix is **not** applied as 289 complex MACs. The `j ↔ 17−j` conjugate pair is
folded first (this is exactly FFTW's `dft-generic-17` form, §02 §2.5), which converts
one *complex* 17×17 matvec into two *real*-coefficient matrix products on complex data:

```
u_j = x_j + x_{17-j}        v_j = x_j - x_{17-j}              j = 1..8
P_k = x_0 + Σ_{j=1..8} cos(2π kj/17) · u_j                    k = 0..8
R_k =       Σ_{j=1..8} sin(2π kj/17) · (-i v_j)               k = 1..8
X_k = P_k + R_k        X_{17-k} = P_k - R_k        X_0 = P_0
```

Derivation (sign convention `w = e^{-2πi/17}`, so `X_k = Σ_j x_j w^{kj}`):

```
X_k = x_0 + Σ_{j=1..8} ( w^{kj} x_j + w^{-kj} x_{17-j} )
    = x_0 + Σ cosθ (x_j + x_{17-j}) - i Σ sinθ (x_j - x_{17-j}),   θ = 2π kj/17
X_{17-k}: θ → -θ, cos even, sin odd  ⇒  same P_k, opposite sign on the sine part.
```

**The one decision that makes the whole entry work:** after the fold every surviving
coefficient is *real*. A real scalar times an interleaved complex vector is a plain
vector FMA — the real and imaginary lanes want the *same* multiplier. So the driver's
interleaved layout is already the correct SIMD layout, there is no split/interleave
repacking anywhere, and the kernel contains **zero cross-lane operations in the
arithmetic**. (Contrast `L17_rader`, which needs split real/imag because Rader's
pointwise stage multiplies by *complex* constants.) The only permutes in the file are
(a) one re/im swap per butterfly, which implements `−i·v`, and (b) the in-register plane
transposes that the innermost axis needs.

`−i·v` is a swap plus a sign flip on the odd (imaginary) lanes. The flip is an **integer
XOR of the sign bit**, not a multiply by `{1,-1,…}`, on purpose: under the 512-bit port
scheme a `vmulpd` can only issue to the FMA port, which is the bottleneck, while `vpxorq`
can also go to port 5, which is idle. Worth 2% locally (below).

### Operation count

Per line (real arithmetic):

| form | flop/line | note |
|---|---|---|
| naive dense complex 17×17 matvec | **2312** | 289 complex MACs = 1156 real FMAs |
| this kernel (conj-pair folded) | **608** | 272 real FMAs + 16 cplx add + 16 cplx sub |
| FFTW `dft-generic-17` | 592 | same, with the all-ones `k=0` row done as adds |
| Rader-17 (`L17_rader`) | 468 | but 388 instructions vs 336, plus a gather |

So the conjugate fold is **3.8× cheaper than the naive dense matrix at identical
regularity** — the fold costs one butterfly and buys a factor 2 on the multiplies and
another factor ~2 by halving the output count. Per volume: 3·289 = 867 lines ×
608 = **527 kflop**. The driver's yardstick `5·N·log₂N` = 301 kflop, so a reported
"GF/s" of 17 corresponds to ~30 real Gflop/s.

I keep the `k = 0` row as FMAs with `cos = 1` (608 rather than 592 flop): on the target,
`vaddpd` and `vfmadd` have identical throughput on the same port, so special-casing it
would save flops on paper and nothing in cycles.

**What actually costs, though, is vector instructions, not flops.** One chunk transforms
`WC` lines at once (WC = 4 complex per zmm, 2 per ymm) and needs

```
136 vector FMAs                      (9 P accumulators × 8 j  +  8 R × 8 j)
 16 vector add/sub                   (butterfly u_j, v_j)
 16 vector add/sub                   (combine X_k, X_{17-k})
= 168 FP ops, + 8 integer XOR, 33 loads, 20 stores, 40 shuffles (tile transpose)
```

17 = 4·4+1, so a 17-long free index costs **5 chunks, not 4.25**: a volume is
85 (Y) + 85 (Z) + 73 (X) = **243 chunks** at WC=4 against an ideal 217 — a 12% lane tax
that no arrangement of a 17-wide index avoids (see "did not work" item 8).

⇒ **40.8k vector FP ops per volume.** On a Cascade Lake with *one* 512-bit FMA unit that
is ~41k cycles = 17.7 µs at 2.30 GHz. The 256-bit kernel does 451 chunks × 168 at
2 FMA/cycle = 37.9k cycles = 16.5 µs, so the two widths were *predicted* to tie, which is
why both are built and both are measured at plan time.

**The node says 512-bit, and the prediction for it was almost exact:** 16.99 µs measured
at B=1 = 39.1k cycles at 2.30 GHz = **1.045 FP ops per cycle**. The kernel saturates a
single 512-bit FMA unit with essentially no overhead, and there is no licence downclock
worth speaking of. The 256-bit kernel, with twice the paper FMA throughput, measures
20.9 µs = ~1.5 FP ops/cycle: it needs twice the instructions and cannot reach 2/cycle.
So *this* is where the arithmetic count becomes the whole story — see "next".

### Layout, passes, SIMD

Lanes always hold `WC` *different lines*, taken from a contiguous run of a free index, so
every load and store is a contiguous vector access and every coefficient is a
lane-invariant broadcast. Chunks that would run off the end of a 17-long index instead
**overlap the previous chunk** (offsets 0, 4, 8, 12, **13** for WC=4): the last chunk
recomputes 3 lines and stores bit-identical values over them. Cheaper than masking (same
instruction count), never reads out of range, portable (no AVX-512 mask registers), and
deterministic.

```
Y : in[x][y][z]  --lanes over z (17) --> pb[z][ky]         transposing store
Z : pb[z][ky]    --lanes over ky (17)--> t1[x][ky][kz]     transposing store
X : t1[x][p]     --lanes over p (289)--> out[kx][p]        plain store
```

* Y and Z run **plane by plane**: a 17×17 plane is 4.6 KiB, so it stays in L1 while both
  of its transforms run; only pass X walks the whole 78.6 KiB volume (8% of L2).
  §05 §5.3's plane granularity, and no padding is needed anywhere (17 is odd).
* **Exactly two plane transposes are unavoidable.** Both `in` and `out` are `[x][y][z]`;
  a row-column algorithm must make each axis the row index once and come back, so two of
  the three passes cannot have their free index contiguous in both source and
  destination. They are done as 4×4 (or 2×2) transposes of 128-bit blocks fused into the
  chunk's store — 8 `vshuff64x2` per tile, pure port-5 work that hides under the FMA
  stream.
* Pass X is **last** so the stores into the caller's `out` are the long sequential ones.
* Coefficients live in a **pre-splatted** table (each of the 136 constants stored `WC·2`
  times, 8.7 KiB for zmm / 4.4 KiB for ymm) so an FMA reads it as a plain full-width
  memory operand. This is not the obvious choice — see "did not work" item 2, it is the
  single biggest win in the file.
* Trig is computed in `long double` in `fft3d_create()`, so the table is good to ~1e-19
  and the 17-term sums land at 3.2e-16 relative.

### Ten kernels, chosen by measurement inside `fft3d_create()`

Stage 1 (always): `{512-bit, 256-bit} × {fused transpose, split transpose, k-blocked,
k-blocked + L1 butterfly}`. All eight compute the same thing; they differ only in vector
width and in how much register pressure the chunk kernel puts on the allocator.
`create()` times all eight *on the machine running the benchmark* (16 volumes, 5 reps,
min) and keeps the winner.

Stage 2 (only when `batch >= 64`): the winner versus the same kernel with the finished
volume staged in an aligned 78.6 KiB buffer and streamed to `out` with **non-temporal
stores**, which removes the read-for-ownership of the output — a third of all DRAM
traffic once the batch outruns L3. Timed on 384 volumes (60 MiB, past any L3) so the
decision is made in the regime it applies to.

**`fft3d_description()` reports the winner**, so the leaderboard JSON carries the answer
back. Setup is 70 ms (B=1) to ~300 ms (large batch) and is excluded from the score.

Why this is not laziness: (a) a Xeon Gold 5218 has **one** 512-bit FMA unit, so 512-bit
code has no peak-flop advantage over 256-bit code (which dispatches FMAs to ports 0 *and*
1) while it does pay the AVX-512 licence clock penalty — and with `-march=native` the
256-bit kernel is still EVEX-encoded, so it keeps 32 registers and embedded broadcasts;
(b) the difference between the kernel structures is entirely a register-allocation
question, and I could not measure the 32-register machine from an AVX2 development box.

The whole per-width kernel is a template instantiated twice by a **self-`#include`** of
`impl/L17_matrixsimd.c` (`__has_include`-guarded, with a fallback path), because the file
count is fixed at two and a 300-line backslash macro would be unreadable.

### What was measured — the benchmark node

`./probe_node.sh L17_matrixsimd`, **isolated Xeon Gold 5218 @ 2.30 GHz, `--exclusive`,
gcc 11.4, `-O3 -march=native -mtune=native -funroll-loops`**, 20 samples, min; MKL 2022
built on the same node from the same data in the same job. Per transform:

| B | L17_matrixsimd | MKL 2022 | speedup | vs FFTW patient (81.68 µs, sota_r1) |
|---|---|---|---|---|
| 1 | **16.99 µs** (17.7 GF/s nominal) | 98.79 µs | **5.8×** | **4.8×** |
| 8 | **19.37 µs** | 99.87 µs | 5.2× | — |
| 256 | **28.19 µs** | 101.2 µs | 3.6× | — |
| 2048 | **29.41 µs** | 103.8 µs | 3.5× | — |

Correctness on the node: rel_l2 = 3.19e-16 … 3.23e-16 at every batch size.

**The tuner's own table on the node** (per transform, `L17_VERBOSE=1`), which is the
AVX-512-vs-AVX2 measurement the corpus does not contain:

| kernel | B=1 | B=8 | B=2048 |
|---|---|---|---|
| 512-bit, fused transpose | 26.32 | 21.63 | 20.49 |
| 512-bit, split transpose | 33.22 | 30.81 | 23.94 |
| 512-bit, k-blocked | 20.76 | **19.29** | 19.92 |
| 512-bit, k-blocked + L1 butterfly | **17.07** | 19.42 | **19.63** |
| 256-bit, fused transpose | 19.53 | 21.84 | 22.24 |
| 256-bit, split transpose | 24.00 | 26.16 | 26.32 |
| 256-bit, k-blocked | 20.75 | 23.21 | 23.76 |
| 256-bit, k-blocked + L1 butterfly | 20.88 | 23.57 | 23.57 |
| NT-store output (stage 2, past L3) | — | — | 34.39 vs 27.99 plain |

Three things to carry forward:

1. **512-bit beats 256-bit by 16–18% on a Gold 5218**, for a kernel with no cross-lane
   arithmetic and one shuffle per four FMAs. The brief's warning about licence
   downclocking and single-FMA SKUs is real but does not bite here: the 512-bit kernel
   runs at ~1 FMA/cycle at ~2.3 GHz, and what it wins is *instruction count*, not flops.
   The corpus had no AVX-512 measurement at all; this is one.
2. **The register-pressure structure is worth up to 35%** (17.07 vs 26.32 at B=1 for the
   same arithmetic, same width, same everything except how many vectors are live).
3. **Non-temporal output stores lose on the isolated node** (34.4 vs 28.0 at 384 volumes)
   even though they win by 19% on a bandwidth-contended machine. A single Cascade Lake
   core is not RFO-bound at this arithmetic intensity. Measured, not assumed — the tuner
   rejects it on the node and takes it on the shared box.

### What was measured — the development machine

Development machine: **Haswell Xeon E5-2680 v3, 2.5 GHz, AVX2 only, 16 ymm registers**,
and *shared with 11 other agents while measuring*, so these numbers are pessimistic and
the absolute clock is not the target's. Compiled with the Makefile's exact flags
(`-O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops`).
Min of 3 processes × 20 samples, per transform:

| B | µs/transform | tuner picked |
|---|---|---|
| 1 | **25.7** | 256-bit, k-blocked + L1 butterfly |
| 8 | **27.1** | same |
| 64 | **27.2** | same |
| 256 | **31.7** | 256-bit, k-blocked + **NT store** (38.9 without it) |

Correctness (vs numpy, `check.py`): rel_l2 = **3.216e-16 … 3.316e-16**, rel_max ≤ 4.0e-16,
at B = 1, 8, 64, 256 — including the SSE-only build and the *emulated* 512-bit build.
Output is **byte-identical** between a single execute and the 30th execute on the same
plan (`cmp` of the two output files), so repeatability is exact, not just within
tolerance.

Static instruction accounting from the **AVX-512** build (`-march=skylake-avx512`), which
is the graded path and the only handle I have on it without the machine:

| exec variant | static insns | FMAs | stack refs |
|---|---|---|---|
| 512-bit fused transpose (monolithic) | 1149 | 260 | **342** |
| 512-bit split transpose (monolithic) | 1076 | 260 | **347** |
| 512-bit **k-blocked** | 709 | 104 | **63** |
| 512-bit k-blocked + L1 butterfly | 724 | 104 | 71 |
| 256-bit **k-blocked** | 660 | 104 | **60** |

The k-blocked kernel is the one to beat on the node: rolled `j` loops, ~311 dynamic
instructions per chunk for 200 FP ops (168 with the L1-butterfly variant), and no spills.

This static accounting predicted 14–21 µs on the node, and the node delivered 16.99. Two
predictions from it that the node confirmed: the k-blocked kernel wins (it does, by up to
35%), and the ~40.8k FP ops per volume are the whole cost (they are — 1.045 ops/cycle).
One that it got backwards: the 5218 does **not** need two FMA units for 512-bit to win.
It has one, the 512-bit kernel saturates it, and it wins anyway because it issues half as
many instructions as the 256-bit kernel for the same FMA throughput.

### What was tried and did NOT work

1. **Coefficient as a scalar + AVX-512 embedded broadcast** (`c[k] * u` with the compact
   1.1 KiB table). This *is* the textbook-optimal encoding — `vfmadd231pd zmm, zmm,
   m64{1to8}`, one instruction, 8 bytes of L1 — and gcc 11 turns it into a catastrophe:
   it hoists all 136 `vec_dup`s and, having nowhere to keep them, **materialises them into
   stack slots inside the chunk loop** — 136 `vbroadcastsd` + 136 `vmovapd` to `%rsp` per
   chunk, i.e. **272 wasted instructions on a chunk whose useful work is 168 FP ops**.
   Static count for `exec_w4`: 1981 instructions with 408 broadcasts. Switching to a
   pre-splatted table (the FMA reads a full-width memory operand, nothing to hoist)
   dropped it to 1250 with 0 broadcasts. *Cost of the wrong choice: ~60% of the
   instruction stream.* If a future round has a newer gcc or uses clang, re-measure this
   — the compact table is better if the compiler cooperates.
2. **The monolithic kernel: 9 P + 8 R accumulators live at once (25 vectors with the tile
   temporaries).** Elegant — one sweep over `j` per half, minimum arithmetic (168 FP ops,
   the butterfly computed once) — and it spills on *both* register files: 342–805 stack
   references per exec in the AVX-512 build, 917 in the AVX2 build (≈265 register moves
   per chunk against 176 FP ops). Locally 29.5 µs vs 28.1 for k-blocked. Kept as a tuner
   candidate only because the spill cost is a compiler property and might vanish.
3. **k-blocking the outputs** (block A: k = 0..4 → outputs m = 0..4, 13..16; block B:
   k = 5..8 → m = 5..12; tiles chosen so no tile straddles the blocks) is the fix: peak
   liveness 17 vectors, stack references 63. It costs one extra butterfly pass over `j`
   (+24 FP ops, +14%) unless the u_j/w_j are parked in a 16-vector L1 scratch, which is
   what the "+L1 butterfly" variant does (16 stores + 16 loads on ports 2/3/4 instead of
   24 FP ops on port 0): 220.4 µs vs 225.0 at B=8 locally.
4. **Staging the 17 accumulators through an L1 scratch** instead of registers (explicit
   store-then-reload, so the allocator has nothing to spill): **230 µs vs 226 µs** at
   B=8 — no help at all, because gcc's spill code is about as good as my staging and it
   also re-forwards some of the loads. Removed from the file entirely.
5. **Splitting the plane transposes out into a dedicated tight loop** (all three passes
   then store untransposed, and a separate 4×4-tile transpose loop moves the plane). The
   chunk kernel gets much cleaner — 266 instructions instead of 554 for the AVX2
   build — and the variant is still **38.9 µs vs 28.5 fused (+37%)** at B=8: the
   transposes cost about as many instructions as they save, and being in a separate loop
   they can no longer hide under the FMA stream. Kept as a candidate; it will not win.
6. **Compiler flags and laundering tricks to stop the hoisting**: `-fno-tree-loop-im`,
   `-fno-move-loop-invariants`, `__attribute__((noinline))` on the chunk kernel, and
   `asm("" : "+r"(ct))` to make the table pointer opaque. All four, and both pairs,
   landed inside the noise: 222–231 µs at B=8 where the baseline was 226. The
   pre-splatted table (item 1) is the fix; none of these are.
7. **Letting gcc fully unroll the 8-iteration `j` loops** in the k-blocked kernel. It
   does so by default and then spills: **815 stack references vs 60** with
   `#pragma GCC unroll 1`, in the AVX-512 build of the 256-bit kernel. The pragma is now
   in the file. Locally it is timing-neutral (the AVX2 path spills either way), so this
   one rests on the static count alone — but the count is not ambiguous.
8. **Eliminating the 12% lane tax of a 17-wide free index.** Masking the tail: identical
   instruction count (a masked chunk costs what a full one costs), so no gain, plus
   AVX-512-only code. A narrower vector for the tail: same instruction count again.
   Letting a vector straddle two rows of the plane: each straddling lane then computes a
   *cyclically shifted* line, whose transform differs by a twiddle `w^{-k·shift}` — a
   correction of 17 complex multiplies per chunk (68 FP ops) to save ~30. Dropped. The
   tax is structural: only the axis whose other two indices are adjacent in memory gets a
   289-wide free index, and that is one axis out of three.
9. **`vfmaddsub`/`vfmsubadd` for the `±i` combine.** Would fold the sign flip into the
   combine for free — but once the `×{1,-1}` had been replaced by an integer XOR (which
   issues to port 5, off the FMA port), there was nothing left to save, and it would have
   cost intrinsics and an `#ifdef` per width. Not done, deliberately.
10. **Non-temporal stores issued directly from the pass-X store.** `out`'s row stride is
    289 complex = 4624 B ≡ 16 (mod 64), so only rows with `kx ≡ 0 (mod 4)` — 5 of 17 —
    are 64-byte aligned, and one chunk writes all 17 rows from one register set, so the
    offsets cannot be shifted per row. That recovers only ~29% of the RFO. **What does
    work** is finishing the volume in an aligned staging buffer and streaming it out with
    one aligned NT copy (widest available: `_mm512_stream_pd` / `_mm256_stream_pd`, with
    16-byte NT pairs to reach alignment first, since a volume is 78 608 B ≡ 16 mod 64 and
    therefore only every fourth volume starts 64-byte aligned). Measured **B=256:
    31.7 µs with, 38.9 µs without (+19%)**; **B=64: 32.3 vs 28.2 (−15%)**, because at
    10 MiB the whole batch still fits L3 and the extra copy is pure cost. Hence it is a
    tuner candidate gated at `batch >= 64`, not a default.
11. **Tuning the NT decision by alternating the two candidates.** This *inverts the
    answer*: alternating reported plain 37.8 / NT 39.5 µs at B=256, blocked measurement
    of the same code reports plain 41.5 / NT 32.7, and the driver's own steady state
    agrees with the blocked one. A plain run leaves 20 MiB of dirty output lines in L3
    and the NT run immediately after pays to flush them, while the plain run after an NT
    run finds its output uncached. Any A/B of a cache-bypassing variant has to be done in
    consecutive blocks.
12. **Tuning candidates by interleaving them, one run each per round.** The obvious way
    to A/B ten kernels — and it mis-ranked them on the real node by **35%**. Interleaving
    512-bit and 256-bit kernels makes every single sample pay an AVX-512
    licence/frequency transition, and short samples never let the clock settle: the
    interleaved tuner reported the winning kernel at 23.52 µs where the driver then
    measured 16.99 µs with it, and it preferred a different candidate at B=8. Fixed by
    timing each candidate in a *block* of ≥ 64 consecutive volume transforms with two
    warmup runs; the tuner and the driver now agree to 0.5% (17.07 vs 16.99 at B=1,
    19.29 vs 19.37 at B=8). **Anyone tuning across ISA widths on this hardware must
    measure in blocks.**
13. **Aligned loads in the NT copy** (`_mm256_load_pd` on the staging buffer after
    aligning the *destination*). Segfault: aligning the destination to 32 B leaves the
    source at 16 B mod 32. The source load must be `loadu`; only the NT store needs
    alignment. Caught locally at B=64.

### Next

1. **Prefetch/NTA the input as well.** The NT-store variant fixed the output side of the
   streaming case (item 10 above). The input side still pulls 78.6 KiB/volume through L2
   and evicts part of `t1` doing it; `prefetchnta` on the next plane, or splitting pass Y
   so that `in` is touched once per plane rather than once per chunk, is the remaining
   traffic lever. Expect single-digit percent, only at large batch.
2. **Reduce the 136 FMAs.** This kernel is FMA-port-bound with no spills, so cycles now
   track the multiply count exactly. The 9×9 cosine and 8×8 sine matrices contain only
   8 distinct magnitudes; a Winograd/cyclotomic treatment of the same conjugate-folded
   form is the only remaining large lever (§01 §2.2's 70 mul + 314 add). It sacrifices
   the property this entry exists to test — perfect data flow — so it belongs in a
   different entry, and the honest comparison against *this* one is the experiment.
3. **Two chunks in flight — now known to be pointless.** The node measurement says the
   winning kernel runs at 1.045 FP ops per cycle on a one-FMA-unit part, i.e. it is
   already at the hardware limit; there is no latency left to hide. Do not spend a round
   on scheduling. Every remaining microsecond has to come from item 2 (fewer multiplies)
   or item 4 (fewer chunks).
4. **The 12% lane tax is now the second-largest lever** (243 chunks where 217 would do).
   Everything cheap was tried (item 8 below). The one untried idea: process the Y and Z
   passes over *two volumes at once* when `batch > 1`, pairing plane `x` of volume `b`
   with plane `x` of volume `b+1` so the free index is 34 long and costs 9 chunks instead
   of 10. Saves ~5% of all chunks at B>1, needs a second plane buffer and does nothing
   for B=1.
5. **Re-measure item 1 of "did not work" with a newer compiler.** The compact scalar
   coefficient table plus embedded broadcast is 8 bytes per constant instead of 64, i.e.
   1.1 KiB of L1 instead of 8.7 KiB, and it is one instruction per FMA. gcc 11 refuses;
   gcc 14+ / clang 18+ may not. This is a request for `gcc-14` or `clang` on the node
   rather than a code change.

### Honest summary of the contrarian premise

The premise was: 289 multiply-adds per line is more arithmetic than Rader's ~200, but
perfect data flow wins anyway. The measurement says the premise is *half* right, and the
interesting half is not the one advertised.

* Nobody should run the naive 289-MAC form. Conjugate-pair folding costs one butterfly,
  keeps the data flow exactly as perfect, and removes 74% of the arithmetic. The dense
  matrix that is actually competitive is a 608-flop/line kernel, not a 2312-flop one.
* Once folded, the dense form is within 30% of Rader on flops (608 vs 468) and *ahead*
  on instructions, and it needs no gather, no permutation table, no convolution
  constants, and no split/interleave repacking — the fold makes every coefficient real,
  which is what lets the driver's own interleaved layout be the SIMD layout.
* On the node the finished kernel runs at **1.045 FP ops per cycle on a part with one
  512-bit FMA unit** — the arithmetic is now the *only* thing left. That is the cleanest
  possible statement of the premise's outcome: perfect data flow was achievable, it was
  achieved, and having achieved it the flop count is exactly what you pay. Rader's 468
  flop/line against this kernel's 608 is therefore worth up to 23% if and only if Rader
  can also be made to run at 1 FMA/cycle — which is the experiment the panel should read
  off `L17_rader`'s record next to this one.
* What limited this entry was never the arithmetic. It was **the register allocator**
  (25 live vectors → 342–805 spills → up to 3× the instructions) and **one bad
  interaction with gcc's loop-invariant motion** (+272 instructions per chunk). Both were
  fixed by restructuring, not by reducing flops, and the fixes are worth more than the
  entire flop difference between dense and Rader. That is §04's and §06's point, arriving
  from a direction neither section predicted.

---

## Round panel_r2 (2026-08-21)

### Standing after round 1

Won B=1 on the node (16.99 µs, vs L17_winograd 18.26 and L17_rader 20.88), lost every
batched regime (B=8 to L17_winograd 20.53 vs 21.63; B=256/2048 to L17_rader 25.98/27.86
vs 27.36/29.28). Round 1's own conclusion: the kernel is FMA-port-bound at 1.045 FP
ops/cycle, so the only levers that matter are *fewer FP ops* and *fewer chunks*.

### What changed: the nested kernel (borrowed from L17_winograd, round 1)

**Adopted, with attribution: the cyclic → (cyclic-4 ⊕ negacyclic-4) halving of the
cosine half of the conjugate-folded matrix is taken directly from `L17_winograd`'s
round-1 derivation (its §1b–1c).** Their record proves the identities and counts every
alternative; what is new here is the observation that the trick drops into *this*
entry's layout for free, because it preserves the one property this entry is built on:
**every coefficient stays real**, so lanes-are-lines with interleaved complex data,
broadcast coefficients, and zero cross-lane arithmetic all survive unchanged.

Reindex j and k by powers of the primitive root 3: `3^m ≡ sig[m]·f[m] (mod 17)`,
`f = {1,3,8,7,4,5,2,6}`, `sig = {+,+,-,-,-,+,-,-}`. Then the folded 9×8 cosine matrix
becomes a length-8 *circulant* (`cos(2π f[m]f[n]/17) = c[(m+n) mod 8]`,
`c[r] = cos(2π·3^r/17)`) and splits by `x^8−1 = (x^4−1)(x^4+1)` with sign-only
reductions; the 8×8 sine matrix becomes a *negacyclic* correlation and stays dense
(L17_winograd's record counts all four splits of `x^8+1`; every one loses — not
retried). Both `sig` factors are free: `sig[m]` is the operand order of the `v`
subtraction (equivalently, load indices `IA[m] = 3^m mod 17`, `IB[m] = 17−IA[m]`),
`sig[n]` is which output of the `(f[n], 17−f[n])` pair takes the `+`.

In this layout the *permutation costs nothing*: it is compile-time constant load
offsets (a static index table in a rolled loop; the kernel is always_inline so the row
stride constant-folds at each call site) and the order in which coefficient rows are
written into the pre-splatted table at plan time. No data is ever physically permuted.
The identities and the 17-slot output map were verified against numpy in isolation
before writing a line of C (max abs err 5.2e-15 on random data).

### Operation count

Per chunk (WC lines at once), vector FP ops:

| | dense (round 1) | nested (this round) |
|---|---|---|
| butterflies u/v | 16 | 16 (8 U adds + 8 V subs) |
| P/Q reduction + X0 | — | 12 (8 add/sub + 4 X0 adds) |
| cosine matvec | 72 FMA | 32 FMA (16 cyclic-4 + 16 negacyclic-4) |
| sine matvec | 64 FMA | 64 FMA (dense negacyclic-8) |
| C = A±B | — | 8 |
| output combine | 16 | 16 |
| **total FP** | **168** | **148 (−11.9%)** |

Per line: 488 flops (matches L17_winograd's count exactly, as it must — same algebra).
Per volume: 243 chunks × 148 = 35 964 vector FP ops (was 40 824). Predicted node B=1
from round 1's measured 1.045 FP ops/cycle: **~15.0 µs** (from 16.99).

### Register pressure, handled the round-1 way

The nested kernel wants C_0..7 + X0 live across the whole 8-iteration sine loop
(8 S accumulators + w + C's ≈ 19 vectors). Round 1 says that shape spills, so both
resolutions are built and *measured*:

* `pc=0` (exec6): C stays in registers. Static (AVX-512 build): 934 insns, 94 stack refs.
* `pc=1` (exec7): C_0..7 + X0 parked in a 9-vector L1 scratch across the sine loop,
  reloaded as the combine's operands. 981 insns, 103 stack refs.

The tuner now has 12 stage-1 candidates ({512,256} × {fused, split, kb, kb+L1, nested,
nested-parked}) and the NT stage maps the winner to its matching NT variant.

### What was measured — wallaby (Sapphire Rapids, Gold 6448Y, shared, NOISY today)

**Warning for anyone reading this for calibration: wallaby was NOT the 0.04%-spread
machine the brief describes, this day.** Run-to-run swings of 2× on identical binaries
(MKL itself flipped 46 ↔ 90 µs between processes minutes apart; my sd column hit 35%
within single runs). Treat min-across-several-runs as the only meaningful statistic,
and expect the isolated node to be the cleaner measurement. Best observed, per
transform, `-DL17_FORCE` pinning one variant per process:

| variant | B=1 (min of 3+ runs) | B=256 (min of 2) |
|---|---|---|
| dense 512-bit k-blocked (round-1 winner) | 9.71 µs | 12.42 µs |
| **nested 512-bit, C parked (exec7)** | **8.90 µs** | **11.68 µs** |
| nested 512-bit, C in registers (exec6) | 10.51 µs — but bimodal, see below | — |

Unforced (full tuner) runs: B=1 9.33 µs, B=8 11.21 µs/t, B=256 13.20 µs/t,
B=2048 19.68 µs/t. Correctness on every run and every batch: rel_l2 = 3.23e-16 …
3.26e-16, bit-identical across repeated executes; the AVX2-emulated w4 path and the
real-SSE w2 path verified locally at B=64 (3.256e-16).

So the nested kernel is worth **~8%** over the round-1 winner on wallaby, against the
~12% op-count reduction — plausible since Sapphire Rapids has *two* 512-bit FMA units
and is therefore less FMA-port-bound than the scoring node, where round 1 measured the
dense kernel at 1.045 ops/cycle on one unit. If the node behaves as round 1 measured,
expect close to the full 12% there (16.99 → ~15.0 µs at B=1).

### What was tried / observed that did NOT work

1. **exec6 (C in registers) is bimodal on wallaby: 10.5 µs in some processes, 17.5–20.1
   in others, sd < 1% within each.** Same binary, so it is address-layout (ASLR) or
   core-placement dependent, presumably via the spill slots. exec7 (parked) never
   showed the slow mode. Both stay as tuner candidates — the tuner measures in-process
   at plan time, and the addresses it sees are the addresses the scored run uses, so a
   per-process pathology is exactly what plan-time tuning handles.
2. **Further splits, not retried, on the strength of other entries' records:** the
   negacyclic-8 (all four factorisations counted and rejected by L17_winograd §4.3),
   the second-level cyclic-4 split (exactly break-even, L17_winograd §4.4), and
   Winograd/WFTA-17 proper (30% more instructions, L17_winograd §4.1).
3. **The 12% lane tax via paired volumes (round-1 "next" item 4) was examined and
   dropped without code:** a chunk's lanes must load a *contiguous* run of the free
   index, and plane x of volume b is 4913 complex away from plane x of volume b+1, so
   a straddling chunk has no contiguous source. Making it contiguous costs a physical
   repack pass, which is the same class of cost the fused layout exists to avoid
   (L17_rader's record prices the equivalent slab scheme at ~3% for real complexity).
   The tax stays; it is structural.

### Next

1. **Get the node numbers.** The tuner's stage-1 table (L17_VERBOSE=1) on the node is
   the real deliverable of the variant zoo: whether nested-parked beats nested-live on
   32 clean registers, and whether the dense kernel retakes any batch size.
2. **The sine side is now 64 of 96 FMAs (2/3 of the multiplies).** The only remaining
   arithmetic idea with a chance: Karatsuba on the negacyclic-8 (3 negacyclic-4
   products) is an exact tie on ops (L17_winograd §4.3 last bullet) but converts 32
   FMAs into adds — worth testing **only if** the node shows the adds can issue off
   the FMA port (they cannot on CLX 512-bit: one fused port... but the 256-bit build
   has ports 0+1, where a tie on ops could become a win if port 5 takes vaddpd — it
   does not on CLX either. Skip unless the ISA changes).
3. **Batched regime is now a traffic problem, not an arithmetic one** (19.7 µs at
   B=2048 vs 8.9 at B=1 on wallaby). The input side still pulls 78.6 KiB/volume
   through L2 with no prefetch; `prefetchnta` on the next plane during pass Y is the
   remaining single-digit lever, unmeasured. L17_rader's record (its §4.7) measured
   prefetch as uniformly worse *for its pass structure*; mine touches `in` exactly
   once per plane, which is the case where NTA prefetch usually does pay. Worth one
   iteration.

---

## Round panel_r3 (2026-08-21)

### Standing after round panel_r2

Node: won B=1 (16.75 µs, nested-parked 512-bit) and B=8 (18.66); **lost both
batched regimes** (B=256: 26.07 vs L17_winograd 24.03; B=2048: 27.27 vs 24.60).
The sobering number: the nested kernel cut FP ops 12% but delivered only 1.4%
at B=1 on the node — measured throughput fell from 1.045 (dense, r1) to ~0.93
FP ops/cycle (nested, r2). The kernel is no longer purely FMA-port-bound, so
from this round on, *µop count and memory behaviour* are the levers, not just
FP ops. Also read off the r2 JSONs: L17_winograd's node tuner picked **256-bit**
variants in both batched regimes while 512-bit won B=1 — width is
regime-dependent on the node, which my round-2 tuner never got to see because
it tuned on an L3-resident 16-volume set even at B=2048.

### What changed (three things)

1. **Pinned sine constants** — *adopted from L17_winograd round 2, "variant C"
   (its register-residency fix), with attribution.* The negacyclic-8 sine
   matrix has only 8 distinct constants `s[r] = sin(2π·3^r/17)`. They are
   broadcast **once per execute** into 8 vector registers made opaque with an
   empty `asm("" : "+v"(K0), ...)` (so gcc can neither rematerialise them nor
   fold them back into memory operands), and the sine sweep is **fully
   unrolled** with the negacyclic sign `(-1)^⌊(m+n)/8⌋` baked in as a
   compile-time `vfnmadd`. Removes **64 of ~137 loads per chunk** (~20% of the
   chunk's µops); FP count unchanged at 148. Guarded: the pin is emitted only
   under `__AVX512F__` (w4) / `__AVX512VL__` (w2), i.e. only on 32-register
   files; elsewhere the constants decay back to memory operands and the
   variant degrades gracefully (verified correct on the AVX2 host).

2. **X-first pass reordering** (new exec schedule, same kernels). Old order:
   Y,Z per x-plane, then X last — reads `in` sequentially but writes the whole
   78.6 KiB `out` volume in one burst of 73 chunks at the end; at batch that
   burst (RFO + writeback) serialises against the next volume's input reads.
   New order: X from `in` into t1, then per-kx-plane Y,Z storing the finished
   4.6 KiB plane **straight into `out`** — output writes spread evenly across
   the volume's compute. The cost is that the volume-opening X pass reads `in`
   as 17 strided streams instead of one sequential one. The axes commute
   exactly; the rounding order differs (see item 3 below).
   Wallaby, forced, same window: **B=256 9.96 vs 12.03 µs/t (−17%), B=2048
   17.65 vs 20.53 (−14%); loses ~5% at B=1 (8.98 vs 8.53) and B=8 (11.41 vs
   10.56)** — exactly the tradeoff the two regimes want.

3. **Streaming-regime tuning + bit-determinism discipline.**
   * *Adopted from L17_winograd round 2 (with attribution):* at `batch >= 64`
     the tuner now times **every** candidate on 384 volumes (~60 MB, past any
     L3), blocked, never interleaved; round 2 tuned on 16 L3-resident volumes
     even at B=2048 and could not see any of the batched effects.
   * *Also adopted from L17_winograd:* a cross-volume input prefetch
     (`prefetcht1` of volume b+1's input, 73 lines per plane, during the
     plane phase whose own sources are L2-resident) as a tunable runtime flag,
     A/B'd blocked at plan time; `fft3d_description()` reports `pf=`.
   * **The repeatability trap (new, read this if you tune by wall clock):**
     `tryout.sh` re-runs the whole binary and `cmp`s outputs across the two
     *processes*. A measuring tuner is not deterministic across processes, and
     my new variants are **not bit-identical** to the old ones (X-first
     reorders the passes; the pinned path contracts differently — see below).
     First run after adding them: `!! NOT REPEATABLE`. Fix: outputs were
     classified into **cmp-verified bit-equivalence classes**, and the class
     is now a *pure function of the batch size* (`batch < 64`: X-last pinned;
     `batch >= 64`: X-first pinned); the tuner selects freely only *within*
     the class — width, C-parking, NT staging and prefetch were all verified
     by `cmp` to change **no bits**. Deterministic choice, machine-dependent
     tuning, repeatability preserved at every batch size.
   * **Do not assume FMA-sign algebra gives bit-identity.** On paper
     `fnmadd(k,w,acc)` rounds identically to `fmadd(−k,w,acc)`, so the pinned
     kernel "must" be bit-identical to the table kernel. Measured: it is NOT
     (cmp differs; gcc emits a handful of `vfmsub132pd`/`vfnmadd132pd` forms,
     i.e. it contracts some ± statements differently in the unrolled code).
     Both are correct to 3.2e-16; they are different fixed points. cmp, don't
     derive.

### Operation count

Unchanged from round 2: 148 vector FP ops per chunk, 243 chunks, 35 964 vector
FP ops per volume, 488 flop per line. This round moved **loads** (−64 per
chunk in the pinned kernels, ~340 → ~270 µops per chunk) and **DRAM write
scheduling** (X-first), not arithmetic.

### What was measured — wallaby (Gold 6448Y, shared, noisy: sd up to 35%, min-across-runs is the statistic)

Autotuned (the shipping configuration), per transform, vs MKL same case:

| case | this round | r2 same machine | MKL | picked |
|---|---|---|---|---|
| B=1 | **8.88 µs** (8.53 forced exec12) | 8.90 | 46.2 | X-last pinned parked, 512-bit |
| B=8 | **8.86 µs** | 11.21 | 46.3 | X-last pinned, 512-bit |
| B=256 | **9.95 µs** | 13.20 (11.68 forced) | 46.9 | X-first pinned, 512-bit, pf=0 |
| B=2048 | **17.19 µs** | 19.68 | — | X-first pinned, 512-bit, pf=0 |

Correctness every run and every batch: rel_l2 = 3.226e-16 … 3.259e-16,
bit-identical across re-runs (the repeatability check passes at B=1, 8, 256,
2048 after the class fix). AVX2 host (Haswell): PASS 3.258e-16, repeatable.

The tuner's own >L3 table (nv=384, one window, B=2048 plan) — the cleanest
single-window comparison of everything this entry has:

| kernel (512-bit unless noted) | µs/transform |
|---|---|
| dense k-blocked + L1 butterfly (r1 winner) | 17.17 |
| nested (r2 winner class) | 15.67–15.77 |
| nested, X-first | 14.70 |
| nested, pinned | 14.71 |
| **nested, pinned, X-first** | **12.77 <== kept** |
| nested, pinned, X-first, 256-bit | 15.22 |
| NT store on the winner | 13.75 (plain 12.58) → plain |
| prefetch on the winner | 13.42 (pf=0 12.81) → pf=0 |

The two new levers stack: −8% (X-first) and −7% (pinned) combine to −19% in
the same window. 512-bit wins every regime on wallaby (2 FMA units); on the
node the 256-bit X-first candidates stay selectable and the tuner decides.

### What was tried / observed that did NOT work

1. **NT stores still lose on wallaby with X-first** (13.75 vs 12.58 at nv=384)
   — consistent with r1's node result. Still offered, still machine-decided.
2. **Cross-volume input prefetch loses on wallaby** (13.42 vs 12.81 at
   nv=384), unlike in L17_winograd's pass structure where it won −4.4%.
   Plausible reason: after X-first, my plane phase already writes `out` to
   DRAM, so the "DRAM-idle compute phase" the prefetch wants does not exist
   here. Kept as an A/B because the node's memory system is different.
3. **X-first at B=1/B=8** loses ~5% (8.98 vs 8.53; 11.41 vs 10.56): with
   everything cache-resident the write-spreading buys nothing and the
   17-stream strided read of `in` costs a little. Hence the batch<64 /
   batch>=64 class split.
4. **Assuming bit-identity from FMA sign algebra** — see above. Cost one
   NOT-REPEATABLE flag and an afternoon of cmp; the class table
   (A: X-last unpinned = both widths = parked = NT; B: X-last pinned;
   C: X-first unpinned; D: X-first pinned; A≠B≠C≠D) is in this round's
   history for whoever adds a variant next.

### Expectations for the node

* B=1/B=8: pinned X-last should convert some of the 0.93→1.0+ ops/cycle gap
  into time; expect 16.75 → ~15.5–16.3 at B=1. If it does NOT move, the B=1
  wall is front-end/retire and the next lever is two-chunk software
  pipelining, which r1 already argued against — read the number first.
* Batched: if the wallaby −14…−17% transfers, B=256 lands ~21.5–22.5 and
  B=2048 ~23–24, i.e. ahead of L17_winograd's 24.03/24.60. Whether the node
  prefers 256-bit X-first (as winograd's batch winners suggest) and whether
  pf/NT flip there is exactly what the description strings will report back.

### Next

1. Read the node's `fft3d_description()` strings and the tuner deltas off the
   leaderboard JSONs; they decide everything below.
2. If batch still trails: software-pipeline across volumes (interleave volume
   b+1's X chunks into volume b's plane phase) — pure scheduling, stays inside
   bit-class D, overlaps the only remaining DRAM-read burst with compute.
3. Pin the 8 cosine constants too (cp/cm of the cyclic-4 ⊕ negacyclic-4):
   +8 registers, cosine-phase liveness ~29 of 32 — may spill; if it does, the
   c-lite trick from L17_winograd's round-2 "next" list (re-broadcast between
   the halves) is the fallback. Worth ~32 loads/chunk.
4. B=1 floor: 148 ops × 243 chunks at 1/cycle = 15.6 µs at 2.30 GHz. If the
   node shows pinned B=1 at ~15.6, stop optimizing B=1 — it is done.

---

## Round panel_r4 (2026-08-21)

### Standing after round panel_r3

Swept all four L=17 cells on the node (B=1 16.386, B=8 17.930, B=256 21.444,
B=2048 22.697 µs), 3.70–4.99× the best library — the largest margin on the
board. The node's picks, read off the description strings: B=1/B=8 = 512-bit
pinned sines, X-last, C **in registers** (exec14_w4); B=256/B=2048 = 512-bit
pinned, X-first, plain stores, **pf=0** (exec15_w4). NT and prefetch both
rejected by the node. The r3 VERDICT (§6) closed the arithmetic question —
B=1 is at 1.05× its own FMA-port floor (15.6 µs at 2.30 GHz) — and quantified
the remaining prize: **6.3 µs/volume of un-overlapped memory time at B=2048**
(22.697 measured vs a ~16.4 µs ceiling), with the named move being to
software-pipeline volume b+1's X pass into volume b's plane phase, inside bit
class D. That is what this round builds.

### What changed (three things, all batch >= 64; B=1/B=8 untouched)

1. **Cross-volume software pipelining** (exec18/19/20, both widths).
   *Adopted from the r3 VERDICT §6 (the monitor's named move), with
   attribution.* X-first still opens every volume with a serial 78.6 KiB
   DRAM read burst — the 73 X chunks — with nothing to hide the misses
   under. The pipelined variants double-buffer t1 (new t1b, +78.6 KiB) and
   execute volume b+1's X chunks *during* volume b's plane phase: NX/17 =
   4–5 chunks per plane, split across **two insertion points** (before the
   Y group and between Y and Z), so the read stream never bursts more than
   2–3 chunks between compute-bound plane chunks. Volume 0's X pass runs
   un-overlapped as a prologue (cost ~1/nb). Pure scheduling: every chunk
   computes the same values on the same operands in the same per-value
   order. exec18 = C-in-registers, exec19 = C-parked, exec20 = C-parked +
   NT staging with a **per-plane** ntcopy (578 doubles after each plane's Z
   group, spreading the NT writes the same way the plain stores are spread).
   pf is meaningless here (the X chunks *are* the prefetch, doing real work)
   and is ignored; the pf A/B is skipped when a pipelined variant wins.

2. **The NT variants moved INTO the batch>=64 selection class** (selD is now
   12 candidates: plain/pipelined/NT × both widths), and the old stage-2
   winner-then-A/B NT step is deleted. Reason, measured on wallaby at
   nv=1000: NT loses without pipelining in the L3-resident regime and the
   two decisions **do not factorize** — at B=2048 the full 2×2 was
   {plain 15.72, pipelined 15.33, NT 12.66, pipelined+NT 14.45} µs/t. A
   tuner that picks the plain winner first and then A/Bs NT on it can reach
   only two corners of that square. All 12 candidates were **cmp-verified
   bit-identical** (class D) on identical inputs at B=256, so the tuner may
   select freely among them without breaking cross-process repeatability.
   r3's warning stands and was re-checked after every structural edit: the
   two-insertion-point version was re-cmp'd (all 6 pipelined variants ==
   class D) — derive nothing, cmp everything.

3. **L3-scaled tuner arena.** *Adopted from L36_mixedradix's machine-relative
   tuner arena (its r2/r3 record), with attribution.* The fixed 384-volume
   (60 MB) streaming arena is exactly wallaby's L3 size, so on wallaby the
   "streaming" tuner was actually tuning L3-resident and **inverted the NT
   decision**: at nv=384 it kept plain (r3 table: 13.75 NT vs 12.58 plain)
   where the driver's steady state at B=2048 has NT 13% faster. Now
   nv = min(batch, clamp(2.5·L3 / 157 KB, 384, 1024)) via
   sysconf(_SC_LEVEL3_CACHE_SIZE): wallaby (60 MB L3) → 1000 volumes, the
   node (22 MB) → 384, i.e. **node behaviour is bit-for-bit identical to
   r3's tuner**; only machines with big L3 change. Plan time at B=2048 on
   wallaby grew 0.7 → 2.5 s (32 candidates × 4 execs × 1000 volumes); on
   the node it stays ~1.2 s as in r3. Setup is unscored but noted.

### Operation count

Unchanged: 148 vector FP ops per chunk, 243 chunks, 35 964 vector FP ops per
volume, 488 flop/line. Pipelining moves no arithmetic and adds none — the 73
X chunks per volume are relocated in time, not duplicated. The NT-pipelined
variant adds the same staging copy NT always cost (78.6 KiB L1/L2 read + NT
write per volume), now issued per plane instead of per volume.

### What was measured — wallaby (Gold 6448Y, shared; noisy again today, sd up
to 32% within runs; min-across-runs is the only statistic)

Forced, per transform, min of 3 runs (V numbers are -DL17_FORCE indices):

| variant | B=256 | B=2048 |
|---|---|---|
| exec15_w4 = r3 winner class, plain (V21) | 9.87 | 16.80 |
| exec18_w4 pipelined plain, C in regs (V32) | **9.36** (split interleave; 9.64 single) | 17.20 |
| exec19_w4 pipelined plain, C parked (V33) | 9.78 | 17.18 |
| exec20_w4 pipelined + NT (V34) | 14.09 | 14.50 (single-point; 15.67 split, noisy) |

Tuner's own >L3 table at nv=1000 (B=2048 plan), the round's cleanest window:
plain X-first 15.72 / pipelined 15.33 / **NT 12.66 <== kept** / pipelined+NT
14.45; the 256-bit column landed 14.75–16.52 and lost every cell to its
512-bit twin except plain X-first (14.75), which the class makes selectable
anyway. Autotuned end-to-end (the shipping config), per transform, all PASS
rel_l2 = 3.226e-16 … 3.259e-16 and bit-repeatable across processes:

| case | this round (best of runs) | r3 same machine | picked |
|---|---|---|---|
| B=1 | 9.76 µs | 8.88 | X-last pinned (unchanged path; delta is machine noise) |
| B=8 | 10.04 µs/t | 8.86 | same |
| B=256 | 9.66 µs/t (9.36 forced) | 9.95 | **512-bit pipelined plain** |
| B=2048 | **13.45 µs/t** | 17.19 | **512-bit X-first + NT** (via the nv=1000 arena) |

B=2048 is a −22% round-over-round move on this machine, but read it
honestly: most of it is the **arena fix letting NT be chosen**, not the
pipelining; pipelining is what won B=256. AVX2 host (Haswell): PASS
3.255e-16…3.258e-16, repeatable, B=8 and B=64.

### What was tried / observed that did NOT work

1. **Pipelining the plain-store path does not help wallaby's B=2048**
   (17.2 vs 16.8 non-pipelined) even though it helps B=256 (9.36 vs 9.87).
   At full streaming on this machine the bottleneck is the RFO+writeback
   write traffic, which pipelining does not touch and NT removes. Whether
   the node — where NT has lost twice and writes are cheaper relative to
   its slower DRAM reads — orders these the same way is exactly what the
   description strings will report back.
2. **NT + pipelining is slower than NT alone on wallaby** (14.45 vs 12.66 at
   nv=1000). Plausible: the interleaved X reads now compete with the NT
   write stream inside every plane, where the non-pipelined variant
   alternates read-heavy and write-heavy phases that each get the full
   memory pipe. Kept as a candidate anyway — the node's memory system is
   different and the variant costs nothing when not picked.
3. **A fixed 384-volume streaming arena silently tunes the wrong regime on
   a 60 MB-L3 machine** — it kept plain stores at B=2048 while the driver's
   steady state had NT 13% faster. Anyone with a tuner sized in absolute MB:
   scale it to the machine's L3 (2.5× here) or you are measuring cache
   behaviour, not streaming behaviour.
4. **Winner-then-A/B tuning cannot find a non-factorizing corner.** The NT
   decision flips sign depending on pipelining and on the arena size. If two
   knobs interact, they must be in the candidate set jointly; this is the
   generalisation of r1's blocked-measurement lesson and it cost r3 nothing
   only by luck (the node rejected NT in every reachable corner).
5. **A cmp scare that was an artifact, documented so nobody repeats the
   panic:** after the two-insertion-point edit, all six pipelined variants
   suddenly "differed" from class D — because tryout.sh regenerates
   $W/in.bin at the last batch size it was run with (64), so the B=256
   reference runs failed and cmp compared against garbage. Regenerating a
   dedicated in256.bin restored bit-identity for all six. If you cmp across
   tryout invocations, generate your own inputs with explicit names.
6. **Not retried, on the strength of earlier records:** every arithmetic
   split (r2/r3 dead-end lists), NT issued directly from pass-X stores (r1
   item 10, alignment), paired-volume lane-tax elimination (r2 item 3),
   cross-volume prefetcht1 as a substitute for real pipelined work (r3: pf=0
   won on both machines; this round's X-chunk interleave is that idea done
   with real work instead of hints).

### Expectations for the node

* B=1/B=8: bit-for-bit the r3 configuration; 16.386/17.930 should stand.
* B=256/B=2048: the class now offers the node three new levers (pipelined
  plain, pipelined NT, and — properly measured for the first time in a
  correctly-sized arena — plain NT, though at 22 MB L3 the r3 arena was
  already correct there, so the honest expectation is that NT still loses
  on the node and **pipelined plain 512-bit is the likely pick**). If the
  wallaby B=256 delta (−3 to −5%) transfers to the node's streaming cells,
  B=256 ~20.4–20.9 and B=2048 ~21.5–22.0 µs. The ceiling remains 16.4; I do
  not expect to reach it this round — plane-granularity interleave overlaps
  the read latency but not the write bandwidth.
* Prediction to score: node picks pipelined plain (exec18 or 19, 512-bit) at
  B=256 and B=2048; B=2048 lands 21.0–22.3 µs. If instead the node picks NT
  (plain or pipelined), that overturns two rounds of node NT rejections and
  the write path becomes next round's whole story.

### Next

1. Read the node's picks and deltas. Three-way fork: (a) pipelined plain won
   and gap narrowed → deepen the pipeline (run b+1's X chunks TWO planes
   ahead so each read has ~1.3 µs of slack, still class D); (b) NT won →
   attack the staging copy (fuse ntcopy into the Z-group stores via an
   L1-resident plane, cutting the extra 78.6 KiB round trip); (c) nothing
   moved → the remaining 6 µs is write-bandwidth-bound and the only lever
   left is getting `out`'s RFO off the critical path, i.e. (b).
2. The B=1 wall stands at 1.05× the port floor; leave it alone (r3 VERDICT).
3. If a future round changes the interleave or insertion points: it is
   scheduling-only, but **cmp anyway** (see item 5 above for how to do it
   without fooling yourself).

---

## Round panel_r5 (2026-08-21)

### Standing after round panel_r4

Swept all four L=17 cells again on the node (B=1 16.431, B=8 18.008, B=256
21.626, B=2048 22.290 µs), 3.77–4.97× the best library. The node's picks:
B=1/B=8 = 512-bit pinned sines X-last (pc flip-flopped between processes,
bit-identical); B=256 = X-first pf=0 in two processes, **pipelined in one**;
B=2048 = X-first pf=0 in all three. So of r4's three new levers the node took
pipelining once, NT never — the r4 VERDICT's headline for the panel was that
five cross-volume pipelining schemes were built and the node selected none in
any cell (mine was the closest to an exception). Net round-over-round: B=2048
−1.8%, everything else flat. Meanwhile **L17_rader's mixed-width ymm-tail
("512t") was picked by the node in all 12 process-cells and moved them −4.1%
(B=1) to −5.2% (B=2048)** — the first confirmed port-level lever at this
geometry, and it transfers directly to this entry's structure.

### What changed (two things)

1. **Mixed-width tail chunks** (`l17_execm_xl/xlp/xf/pipe`, FORCE 38–41,
   tuner candidates 32–35). *Adopted from L17_rader round panel_r4, with
   attribution.* The Gold 5218 has ONE fused 512-bit FMA unit but TWO 256-bit
   FMA ports, so a ymm chunk retires its 148 FP ops in ~74 cycles where a zmm
   chunk needs ~148. A 17-long free index therefore costs 4 zmm + 1 ymm tail
   (offsets 0,4,8,12 + 15; one line recomputed) = ~4.5 zmm-equivalents,
   instead of 5 zmm (0,4,8,12,13; three lines recomputed). The X pass gets
   72 zmm + 1 ymm at 287. Implementation: the mixed execs live in the main
   body after both template passes and call `chunk17n_w4` and `chunk17n_w2`
   from the same function — the zmm K set is asm-pinned as before, the ymm
   tail runs the same pinned code path with its own unpinned Q set (pinning
   both sets would hold 16 of 32 registers and make the zmm kernel spill; the
   Q set spilling to stack instead costs the tail chunk a few reloads, visible
   as 64 stack refs vs 27 in the pure exec). The zmm group loop is kept
   rolled with an **asm-opaque bound** (L17_rader r4: gcc 11 ignores
   `#pragma GCC unroll 1` around an always_inline callee). Four variants:
   X-last pc=0/pc=1 (class B), X-first pc=0 and X-first pipelined (class D).

2. **Sustained-clock probe in the description string** — the monitor's r4 ask
   for L=17. Four independent latency-4 FMA chains issue exactly 1 FMA/cycle:
   dense enough in heavy ops to hold the width's licence clock, and
   latency-bound rather than unit-bound on 1- and 2-FMA parts alike, so
   cycles = 4N and clk = 4N/elapsed (assumes FMA latency 4: true on
   SKX/CLX/ICX/SPR, 25% low on Haswell, which is not scored). Best of 3 reps
   of ~1 ms, first rep absorbs the licence transition; runs once at the end
   of create(). `fft3d_description()` now ends with
   `clk512/256=X.XX/Y.YY GHz`, so the r5 leaderboard JSONs will carry the
   node's actual 512-bit vs 256-bit clocks — the number the whole
   width-tuning story has been inferred around since r1. Wallaby measures
   4.10/4.10 GHz (Gold 6448Y: no 512-bit downclock, as the brief says).

### Operation count

Per chunk unchanged (148 vector FP ops). Per volume the chunk mix changes:
pure zmm = 243 chunks = 35 964 zmm-op-slots; mixed = **208 zmm + 35 ymm**
chunks. Node-cycle floor at 1 zmm-FMA/cycle, ymm at 2/cycle:
208×148 + 35×74 = **33 374 cycles = 14.51 µs** at 2.30 GHz, against the pure
512-bit floor of 35 964 cycles = 15.64 µs (−7.2%). Line-slot accounting: Y/Z
now waste 1 recomputed line per 17 instead of 3 (18 slots for 17 lines), X
wastes 1 per 289; the ideal no-waste floor is 32.1k cycles, so the mixed
kernel sits 4% above ideal — the lane tax that r1 called structural is now
almost entirely paid off *on the node's port structure* (on a 2-FMA-unit
machine the mix is exactly FP-neutral and only the instruction count grows).

### Bit-classes: verified, and how

All four mixed variants were cmp-verified on **full output files** on wallaby
before being added to the selection classes: F38, F39 == F20 (exec14_w4,
class B rep) at B=8; F40, F41 == F21 (exec15_w4, class D rep) at B=256.
Dedicated per-batch input files (in8/in256), not tryout's recycled in.bin
(r4 item 5). So the per-lane arithmetic of the w2 instantiation contracts
identically to w4 under this gcc — as it did for L17_rader — and the tuner
may select mixed vs pure freely within each class. Caveat for the record:
bit-identity is a codegen artifact verified on wallaby's -march=native
(SPR); the node compiles -march=native (CLX). r4's selD carried exactly the
same exposure and held on the node (the B=1 pick flipped between processes
with correctness JSONs identical to the last ULP), so the risk is accepted,
not ignored.

### What was measured — wallaby (Gold 6448Y, shared, noisy again: sd up to
28% within runs; min-across-runs is the statistic)

Forced, per transform, min of 3 same-window reps:

| case | pure (F20/F21) | mixed (F38/F40) | mixed pipelined (F41) |
|---|---|---|---|
| B=1 X-last | **8.96** | 9.26 (+3.3%) | — |
| B=256 X-first | 9.89 | 9.99 | **9.35** |

Exactly the expected sign: on TWO 512-bit FMA units the mix is FP-neutral
and pays ~750 extra static instructions (1906–1987 per exec vs 1163–1175
pure; pipe 3059, all within L1i), so wallaby prefers pure and **the mixed
variants are a pure node bet, shipped as tuner candidates** — same shape as
L17_rader's r4 bet, which the node then took everywhere.

Autotuned end-to-end (shipping config), per transform, best of runs, all
PASS rel_l2 = 3.226e-16 … 3.259e-16 and bit-repeatable across processes:

| case | this round | r4 same machine |
|---|---|---|
| B=1 | **8.64 µs** | 9.76 |
| B=8 | **8.96 µs/t** | 10.04 |
| B=256 | **9.14 µs/t** | 9.66 |
| B=2048 | **13.22 µs/t** | 13.45 |

(Deltas vs r4 are mostly machine-mood, not code: the B=1 path is unchanged
when the tuner picks pure. In one heavily contended window the B=1 tuner
table ranked everything ~2× slow and kept mixed C-parked at 16.22 — plan-time
in-process tuning correctly tracks whatever conditions the process actually
has.) AVX2 host (Haswell): PASS 3.255e-16, repeatable, 23.4 µs/t at B=8 —
the emulated-zmm + real-ymm mix compiles and runs correctly there too.

### What was tried / observed that did NOT work

1. **Mixed tail on wallaby at B=1: +3.3% (9.26 vs 8.96 forced).** Not a
   failure — the predicted sign on a 2-FMA-unit part — but recorded so
   nobody reads wallaby numbers and deletes the variants. Only the node can
   confirm the −7% prediction; L17_rader's node data says it will.
2. **Pinning both K (zmm) and Q (ymm) constant sets — rejected at design
   time, not measured:** 16 of 32 registers held across the whole execute
   would guarantee spills in the zmm kernel (r1 item 2: 25 live vectors →
   342–805 stack refs). The Q set spills ~37 extra stack refs instead, on
   1 chunk in 5.
3. **An xmm (WC=1) tail instead of ymm — rejected by arithmetic, not built:**
   a WC=1 chunk still costs 148 FP ops for ONE line = 74 cycles/line on the
   node, exactly what the ymm tail pays for two lines (one wasted). Equal
   cycles, one more kernel instantiation in L1i. The ymm tail is optimal at
   the chunk level; the remaining 4% over the no-waste floor is not
   addressable by width mixing.
4. **The `#pragma GCC unroll 1` trap re-confirmed second-hand:** the mixed
   group loop uses an asm-opaque bound from the start (L17_rader r4 item 2);
   the 4-copy unroll it prevents would have put ~1000 extra instructions
   into each exec.

### Expectations for the node

* B=1/B=8: if the tuner picks mixed (it should — L17_rader's kernel, with
  worse lane economics than mine, gained 4.1% from the same trick), the
  port-floor argument predicts 16.43 → **~15.3 µs** at B=1 (16.43 × 33374 /
  35964 = 15.25, if B=1 is still purely FMA-port-bound) and B=8 ~16.7.
  The pick string will say `512-bit+ymm tail`.
* B=256/B=2048: the compute floor drops ~1.1 µs; the memory-bound overhead
  on top is untouched, so expect ~20.5 / ~21.2 µs if the streaming picks go
  mixed (plain or pipelined).
* The clk512/256 numbers in the description strings are the round's real
  deliverable for the panel: they close LITERATURE.md §4.8 gap 6 with a
  measured licence-clock pair on the scoring part, and they price every
  future width decision at L=17 (and L=8/L=36, whose implementers should
  read them).

### Next

1. Read the node's picks and the clk512/256 pair. If mixed won B=1 at
   ~15.3, B=1 is within ~5% of the new 14.5 µs floor and the lane tax is
   closed; the only remaining B=1 lever would be fewer FP ops per line,
   which r2/r3 closed (the −12% nested split bought 1.4%). Declare B=1 done.
2. If mixed also won the batch cells, the gap to the ~15 µs batched floor is
   pure memory scheduling again: the next candidate is deepening the
   pipeline (X chunks of volume b+1 issued TWO planes ahead), which stays in
   class D and was the r4 VERDICT's named move before it de-prioritised
   pipelining panel-wide. Read the B=2048 delta first.
3. If the node's clk256 comes back well above clk512 (a real licence gap),
   re-examine the 256-bit X-first candidates at batch with the measured
   ratio in hand — the r3 observation that L17_winograd's node tuner picked
   256-bit in batched regimes has never been explained, and the clock pair
   either explains it or rules it out.

---

## Round panel_r6 (2026-08-21)

### Standing after round panel_r5

Swept all four L=17 cells on the node for the fourth round running (B=1
15.223, B=8 16.658, B=256 21.198, B=2048 21.983 µs), 3.94–5.37× the best
library. The node picked the mixed 512-bit+ymm-tail shape in all 12
process-cells, and the r5 port-floor prediction landed within 0.2% at B=1 —
the round's headline in the VERDICT. Two numbers now steer everything:

* **The clock probe came back: clk512 = 2.89 GHz (settled, 4/5 probes
  agree), clk256 = disputed (mine reads 3.89, L17_winograd's saturating
  design reads 2.89).** Re-derived at 2.89 GHz, B=1 is 44.0k cycles against
  my 33.4k-cycle mixed-shape floor — **1.32× the floor, ~44 cycles/chunk of
  non-FP time**, not the 1.05× the r3 record assumed. B=1 is NOT done; the
  prize the whole panel now hunts at L=17 is that 10.6k cycles/volume.
* **The panel-wide store lesson (VERDICT §4.5): hide the RFO with
  `prefetchw`, do not avoid it with NT stores.** pfw was selected 3/3 in
  every streaming cell at L=8 (−20% at B=2048) and L=36 (−16.6% at B=256);
  NT lost on the node for the fourth consecutive round, everywhere. My
  batched cells carry ~6.0–6.8 µs/volume of memory overhead over B=1 and
  have never had a write-side prefetch.

### What changed (three things)

1. **Write-intent prefetch (`pw`)** — *adopted from L8_fusedaxes round r5
   (`pfw`) and L36_pfa round r5 (`pf=2`), with attribution.* All X-first
   variants (nested exec13/15/17 both widths, mixed xf/xfd, mixed pipelined)
   now optionally issue `__builtin_prefetch(line, 1, 3)` → `prefetchw` on
   the 73 out-lines of the plane the Z group is about to store, in two
   half-bursts (37 before the Y group, 36 between Y and Z), i.e. ~0.25 µs
   ahead of the stores at node speed. Runtime flag `p->pw`, A/B'd at plan
   time; prefetches change no bits, so bit-class D is untouched (cmp'd
   anyway). The old stage-2 pf A/B became a **joint (pf, pw) 2×2 grid** on
   the stage-1b winner — r4's lesson that interacting knobs must be judged
   jointly, applied to the new knob.

2. **Deferred-Z plane schedule** (`l17_execm_xld` FORCE 42 → class B;
   `l17_execm_xfd` FORCE 43 → class D). The r5 VERDICT's ~44 cycles/chunk of
   non-FP time, and all three entries' suspicion of the serialized per-plane
   traffic, pointed at this entry's one true intra-volume dependency: the Z
   group's loads depend on its own Y group's stores to the same 5.4 KiB
   plane buffer (a store→load-forwarding junction once per plane, 17 per
   volume, plus group-tail drains with no independent work behind them).
   Fix at group granularity, in the spirit of L17_winograd's d8 (defer the
   dependent group one slot — theirs deferred the transposed store, mine
   defers the whole consuming group): double-buffer the plane buffer
   (pb/pb2, already allocated since r1) and run Y(x+1)→pbB between Y(x)→pbA
   and Z(x)←pbA. Order: Y0; Y1 Z0; Y2 Z1; … Y16 Z15; Z16. Every group
   junction now has a full independent group (≥ ~700 node cycles) between a
   store group and its dependent load group. Pure scheduling — same chunks,
   same operands, same per-value order — and **cmp-verified bit-identical**
   to the class representatives on full outputs (xld ≡ xl at B=1 and B=8;
   xfd ≡ xf at B=256, including with pf=1 pw=1 forced).

3. **Dense-256 clock probe** (`d256` in the description string) — the
   monitor's §5 ask: my sparse 4-chain probe (1 FMA/cycle) and a saturating
   8-chain probe (2 FMA/cycle, both 256-bit FMA ports busy) now run in the
   SAME process, same method, differing only in chain count. If the node
   reports `clk256=3.89, d256=2.89`, the licence discriminator is density
   and both r5 readings were right about different regimes; if
   `d256=3.89` too, L17_winograd's probe has some other systematic. Either
   way the clk256 dispute closes with one leaderboard string.

### Operation count

Unchanged from r5: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per
volume, node port floor 33 374 cycles = 11.55 µs at the now-measured
2.89 GHz. This round moved scheduling (deferred-Z), the write path (pw) and
a measurement (d256); it added zero arithmetic. Cost of pw when on: 73
prefetchw µops per plane (~0.5 µops/chunk-equivalent); of deferred-Z: one
extra L1-resident plane buffer, zero extra instructions.

### What was measured — wallaby (Gold 6448Y, shared; the 2.10↔4.10 GHz
clock lottery documented in r5 was visible again — same-window forced pairs
are the only statistic quoted)

Forced same-window A/B, per transform, min:

| case | incumbent (F38/F40) | deferred-Z (F42/F43) | delta |
|---|---|---|---|
| B=1, X-last | 8.915 | **8.651** | **−3.0%** |
| B=8, X-last (tight window, sd<0.7%) | 9.212 | **8.685** | **−5.7%** |
| B=8, X-last (noisy window) | 9.98 | 8.56 | −14% (min-of-noisy, read with care) |
| B=256, X-first | 10.15 | 10.21 | flat |

pw forced A/B on F40 (X-first mixed) at B=2048 (307 MB, streams even on
wallaby's 60 MB L3): pw=0 13.90 vs pw=1 13.96 µs/t — **flat-to-slightly-worse
on wallaby**, exactly as expected on a machine that prefers NT there and has
~2× the node's memory bandwidth per core. The in-tuner (pf,pw) grid at
nv=1000: 12.65 / 12.71 / 13.18 / 13.53 for 00/01/10/11 → wallaby keeps
pf=0 pw=0. **pw is a pure node bet, the same shape as r5's mixed-tail bet**
(wallaby +3%, node −7%), riding on the node's own r5 selections at L=8 and
L=36 rather than on any wallaby number.

Autotuned end-to-end (shipping config), per transform, all PASS rel_l2 =
3.226e-16 … 3.259e-16 and bit-repeatable across processes at B=1, 8, 256,
2048: B=1 **8.488**, B=8 10.33, B=256 10.24, B=2048 **13.73**. (B=8/256
landed in contended windows; the forced pairs above are the signal. The B=1
tuner table had deferred-Z 8.35 vs incumbent 8.32 — a tie on a 2-FMA-unit
machine whose junction cost is relatively larger per chunk but whose OoO
window is also 2.3× deeper; the node's one-FMA/224-ROB configuration is the
one the variant was built for.) AVX2 host (Haswell): PASS 3.255e-16,
repeatable, 24.3 µs/t at B=8 — emulated-zmm + real-ymm + both new execs all
correct on a 16-register machine.

Description string now ends `clk512/256=X.XX/Y.YY GHz, d256=Z.ZZ`; wallaby
windows read 2.10/2.10, d256=2.10 (contended) and 4.10/4.10, d256=4.10
(idle) — the clock lottery, and no width or density split on SPR, as the
brief says.

### What was tried / observed that did NOT work

1. **pw on wallaby: flat to −0.4% at B=2048, ~−10% when forced together
   with pf on xfd at B=256** (11.0 vs 9.99 µs/t). Not a failure of the
   mechanism — the grid rejects it here and offers it to the node — but
   recorded so nobody reads wallaby numbers and deletes the flag. Node r5
   data (two geometries, 3/3 picks) is the entire case for it.
2. **Deferred-Z does nothing for X-first at B=256 on wallaby** (10.21 vs
   10.15). Plausible: at streaming batch the junction stall is already
   hidden under DRAM misses; the variant is aimed at the cache-resident
   regimes. It stays in class D anyway — it costs nothing when not picked.
3. **Chunk-level Y/Z interleaving was considered and NOT built**: within a
   group, adjacent chunks are already independent, so the OoO scheduler
   gains nothing from manual interleave there; only the group-level
   dependency (Y(x)→Z(x)) needed breaking, and group-level deferral does
   that with zero extra instructions. (Also avoids doubling live-register
   pressure, which r1 items 2/7 price at up to 3× instructions.)
4. **Raw-ssh measurement trap, for whoever automates dev runs next:** a bare
   `ssh wallaby './tryout.sh …'` lands in $HOME, silently does nothing, and
   leaves STALE out.bin files in the scratch dir — my first "bit-identical"
   cmp compared a file against itself. Caught because the run logs contained
   `No such file or directory`. Always `cd` inside the remote command (or
   use a helper script) and check the logs actually contain a PASS line
   before believing a cmp.

### Expectations for the node

* **B=1/B=8**: the tuner should pick `…, deferred-Z` (tag suffix; F42
  shape). The junction cost per plane on the node is unknown — wallaby says
  −3.0%/−5.7% with 74-cycle chunks and a 512-entry ROB; the node's chunks
  are 2× longer (junction relatively smaller) but its ROB is 2.3× shallower
  (junction relatively larger). Honest range: **B=1 14.6–15.2, B=8
  15.8–16.6**. If B=1 lands ≤14.8, the junction was a real part of the 44
  cycles/chunk and the remaining gap to the 11.55 µs floor is chunk-boundary
  window capacity — the next lever is then µops/chunk (the ~32 cosine
  coefficient loads), not scheduling.
* **B=256/B=2048**: the bet is the node selects **pw=1** on an X-first
  mixed variant. If the RFO share of the 6.0–6.8 µs/volume overhead behaves
  as it did at L=8/L=36, expect **B=256 19.5–21.0, B=2048 20.0–21.5**. If
  the node keeps pw=0, then my spread-out per-plane stores already hide the
  RFO that L8/L36's burstier store patterns exposed, the overhead is
  read-side, and the next round should deepen the X-chunk pipeline instead.
* **d256**: predicted **2.89** next to clk256=3.89 in the same string,
  confirming density as the licence discriminator and closing VERDICT §5's
  open question in one process.

### Next

1. Read the node picks: deferred-Z at B=1/B=8? pw at batch? d256 = 2.89?
2. If B=1 moved but sits >1.15× floor: the cosine side still loads 32
   coefficient vectors per chunk. A per-chunk 8-register cosine residency
   (fully unrolled 4-iteration cyclic/negacyclic-4 with compile-time
   rotation, constants made asm-opaque per chunk, NOT pinned across the
   execute — pinning 16 of 32 registers is a documented dead end, r5 item 2)
   would cut ~24 loads ≈ 10% of chunk µops. Spill risk in the combine phase
   (25 live + tile temporaries); build it as a tuner candidate and let the
   allocator vote.
3. If pw won: try pacing variants (per-chunk instead of half-group bursts,
   one plane ahead instead of same-plane) — L36_pfa's record says pacing
   granularity mattered on their store stream.
4. If neither moved the batch cells: the overhead is read-side latency of
   the X pass; deepen the pipeline (X chunks of volume b+1 issued two planes
   ahead, still class D) — the r4 idea the node half-took once.

---

## Round panel_r7 (2026-08-22)

### Standing going in

**panel_r6 was abandoned between development and timing** (stale-runner
provenance issue; results/panel_r6_abandoned_no_timing/WHY.md), so there are
NO node numbers for anything r6 built: pw (write-intent prefetch), the
deferred-Z schedule, and the d256 dense-clock probe all ship unmeasured into
this round, and the r5 leaderboard still stands: swept all four L=17 cells
(B=1 15.223, B=8 16.658, B=256 21.198, B=2048 21.983 µs), 3.94–5.37× the
best library. The r6 expectations section therefore remains the live
prediction sheet, and this round's leaderboard will price r6 and r7 jointly.
The r6 "Next" list, acting blind: item 2 (cosine-side register residency,
~24 loads ≈ 10% of chunk µops, aimed at the 44 cycles/chunk of non-FP time
at B=1) was the only lever with its own wallaby-measurable signal. Items 3/4
(pw pacing, deeper pipeline) would be third variants of mechanisms with zero
node signal — exactly what the panel's r4/r6 discipline says not to build.

### What changed: cosine-resident kernel (execm_xlc/xldc/xfc/xfdc, FORCE 44–47)

The nested kernel's cosine phase still read its 32 coefficients per chunk as
memory-operand FMAs, though only 8 distinct constants exist: cp[0..3],
cm[0..3] = row m=0 of the table (where every negacyclic sign is +). The cr
variants load those 8 into registers once per chunk — asm-opaque
(`L17_CR_ASM`), so gcc neither refolds them into memory operands nor hoists
them out of the chunk loop into spill slots (r1 item 1) — and fully unroll
the 4 m-iterations with the rotation (m+n)%4 and the negacyclic signs
resolved at compile time as vfnmadd. Removes 24 of ~73 remaining loads per
zmm chunk plus the rolled m-loop's ~9 overhead instructions; FP count
unchanged (148/chunk). Same shape as r3's pinned sines, applied per chunk
because per-execute pinning of both sets (16 of 32 registers) is a
documented dead end (r5 item 2). The ymm tail keeps the table path (a
16-register file would spill 8 residents). Own idea from my r6 "Next",
lineage: L17_winograd's r2 register-residency trick, attributed since r3.

### The round's finding: NOT bit-identical, therefore NOT selectable

cmp on full outputs (dedicated inputs, PASS asserted per run): xlc vs xl and
xldc vs xld differ from byte 25 at B=8; xfc vs xf, xfdc vs xfd differ at
B=256. Both fixed points are correct (3.258e-16 vs 3.255e-16) — this is
r3's phenomenon, fourth documented instance: **gcc 11 contracts unrolled
straight-line ± code differently than the rolled-loop form, even when the
source expresses identical operations** (fnmadd(c,q,B) == fmadd(−c,q,B)
exactly, so the sign algebra is not the cause). The statics say how far the
restructuring goes: xlc carries 78 MORE FP instructions than xl (810 vs 732)
for the same arithmetic, with FEWER stack refs (193 vs 223) — gcc
rematerialises rather than spills in the unrolled form. Since the tuner may
only select within a bit-class and the classes are a pure function of batch
size, the cr variants are **measured in the verbose tables and forceable
(-DL17_FORCE=44..47) but in neither selection class**. Redefining the
classes to cr-only without a node number would bet my strongest cell on a
wallaby tie; not done. The class comment in the file says never to add them
to selB/selD without re-deriving the class structure.

### Operation count

Unchanged: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per volume,
node port floor 33 374 cycles = 11.55 µs at the measured 2.89 GHz. This
round moved per-chunk load µops (−24 zmm-chunk loads in the cr twins) and
loop overhead; zero arithmetic.

### What was measured — wallaby (Gold 6448Y, shared; the 2.1↔4.1 GHz clock
lottery again; in-process tuner tables are the only same-window statistic)

B=1 stage-1 tuner tables (blocked, one process per column pair), µs/t:

| pair | window 1 | window 2 |
|---|---|---|
| xl / xlc | 10.04 / 10.08 | 8.60 / 8.65 |
| xld / xldc | 9.88 / 9.97 | 8.50 / 8.59 |
| xf / xfc | 9.91 / 10.02 | 8.93 / 8.66 |
| xfd / xfdc | 9.76 / 9.83 | 8.49 / 8.80 |

B=256 streaming table (nv=1000, one window): xf 9.97 / xfc 10.04, xfd 9.95 /
xfdc 10.05, xl 11.28 / xlc 10.36, xld 10.50 / xldc 10.30. Verdict: **a tie
within wallaby noise (±2%), leaning slightly against cr** — the expected
sign on a 2-FMA-unit, 6-wide machine where load µops do not bind. Like the
r5 mixed tail (wallaby +3%, node −7%) this is a pure node bet: the node is
4-wide with one FMA unit and 44 cycles/chunk of unexplained non-FP time,
which is exactly what 24 fewer load µops + 9 fewer loop-overhead
instructions per chunk attack.

Shipping (unforced) config, all PASS rel_l2 = 3.226e-16 … 3.259e-16 and
bit-repeatable across processes: B=1 9.86 µs (noisy window; 8.71 in the
verbose window), B=8 10.44 µs/t (contended window), B=256 9.20 µs/t, B=2048
13.61 µs/t — indistinguishable from the r6 code, as it must be: the
selectable set is bit-for-bit r6's. AVX2 host (wombat): PASS 3.255e-16,
repeatable, 23.6 µs/t at B=8.

### What was tried / observed that did NOT work

1. **Assuming the cr path would land in the existing bit-class.** On paper
   the sign algebra rounds identically; in practice gcc restructured the
   unrolled block (+78 FP instructions, different fmadd132/231/fmsub mix).
   Cost: the round's lever is built and verified but cannot score until a
   node number justifies moving the class rule. cmp, don't derive — now
   with a fourth data point.
2. **The raw-ssh measurement trap (r6 item 4) re-triggered, three times,
   with a nastier disguise:** `ssh wallaby 'W=...; ./tryout.sh ... 2>&1 |
   grep PASS; cp $W/out.bin ...'` without a leading `cd` lands in $HOME,
   the shell's "No such file or directory" goes INTO the pipe and grep
   silently eats it, and the cp then copies the STALE out.bin from the
   previous legitimate run — so the cmp "verified" four copies of one file
   as bit-identical. My first two class-B "BIT-IDENTICAL" results were
   this artifact; the truth (DIFFERS) only appeared once every forced run
   asserted its own PASS line. Protocol now: any remote forced-cmp loop
   must (a) `cd` first, (b) print and check a PASS line per run, (c) only
   then cmp. Wallaby numbers without a PASS line adjacent are void.
3. **Not built, deliberately:** pw pacing variants and a two-plane-deep
   X pipeline (no node signal for the r6 versions yet — r6 was never
   timed), any class redefinition (see above), intrinsics-exact rewrite of
   the cr cosine to force bit-identity (gcc canonicalises intrinsic
   mul/add through the same contraction machinery, so it buys no
   guarantee; and three rounds of history say don't fight the allocator).

### Asks for the monitor (this is the round's real deliverable at L=17)

The r6+r7 code carries FOUR node-unmeasured mechanisms. The leaderboard
prices the selectable ones (pw, deferred-Z, d256 in the description
string). For the non-selectable cr bet, please force-measure same-window
pairs on the node, blocked, two warmups:
* B=1: `-DL17_FORCE=38` vs `-DL17_FORCE=44` (xl vs xlc), and
  `-DL17_FORCE=42` vs `-DL17_FORCE=45` (xld vs xldc).
* B=2048: `-DL17_FORCE=43 -DL17_FORCE_PW={0,1}` vs `-DL17_FORCE=47`
  same grid (xfd vs xfdc).
If cr wins ≥3% anywhere, next round moves the class rule (cr twins become
the class under a deterministic `__AVX512F__` gate, old members leave);
if it loses, the 44 cycles/chunk are not load-µop-bound and the remaining
B=1 suspects are chunk-boundary window drain (re-read r1's two-in-flight
argument at the true 2.89 GHz) or genuinely done.

### Next

1. Read the r7 leaderboard: deferred-Z at B=1/B=8? pw at batch? d256=2.89?
   cr forced pairs? Each answer settles a mechanism on the scoring machine.
2. If cr wins on the node: redefine the classes (deterministic ISA gate),
   and consider extending residency to the X-first pipelined exec.
3. If deferred-Z won B=1 and cr also helps: they compose (xldc exists).
4. If nothing moved B=1: it sits at 1.32× a floor that three rounds of
   µop/scheduling work could not close further; the honest next candidate
   is a different kernel shape entirely (e.g. L17_rader's 468-flop line at
   this entry's data flow), which is a rewrite, not a tweak.

---

## Round panel_r8 (2026-08-22)

### Standing going in

Swept all four L=17 cells on the node again in panel_r7 (B=1 15.227, B=8
16.715, B=256 21.437, B=2048 21.661 µs), 3.88–5.37× the best library, but
flat round-over-round and B=256 was a cell-level regression (+1.1%) in the
one cell where the r6 deferred-Z was selected.  The r7 VERDICT's synthesis
for L=17: three scheduling attacks declined by the node in two rounds; **the
only mechanism that has moved an L=17 cell in three rounds is µop deletion**
(L17_winograd's g8, −9% in four cells).  My r7 lever (cosine-resident `cr`)
shipped unmeasurable (outside the bit classes) with a standing ask for forced
node pairs.  B=1 sits at 44.0k cycles against the 33.4k mixed-shape port
floor — ~44 cycles/chunk of non-FP time, unexplained after deferred-Z was
declined at B=1/B=8.  Meanwhile the new L23_matrixsimd entry (my architecture
at L=23) reached **1.14× its floor** with a t1 stride padding the VERDICT
called the round's best mechanism, asking for `ld_blocks_partial.address_alias`
counters to explain it.

### Finding 1: the r7 cosine-resident variants are DEAD (withdrawn ask)

Before betting anything on `cr`, I counted its instructions properly.  No
perf/valgrind exists on wallaby, so: objdump of the shipping binary, count
instructions inside each ROLLED loop body (the group loops have asm-opaque
bounds, so a loop body × trip count IS the dynamic count):

| loop body | xl (table cosine) | xlc (cosine-resident) |
|---|---|---|
| zmm Y/Z group chunk (tr=1) | 228 instr | **254 instr (+26)** |
| zmm X chunk (tr=0) | 185 instr | **210 instr (+25)** |
| whole exec, FP instructions | 408 | 456 (+48) |
| whole exec, loads (reg column) | 271 | 249 (−22) |

gcc 11 does delete the 24 coefficient loads — and then **rematerializes the
butterfly adds instead of keeping them live** (+48 static FP, +26
instructions per chunk dynamically).  On the node's one-FMA-unit port 0,
rematerialized vaddpd lands ON the binding port to save loads that were
hidden on ports 2/3.  The r7 wallaby tie now has its explanation, and the
node prediction flips sign: **cr should LOSE on the node.  Dead end,
documented; the r7 ask for forced cr node pairs is withdrawn.**  Lesson for
every entry: "fewer loads" is not "fewer µops" — count the loop body after
the compiler is done, not the source.

### Finding 2: the heap layout is deterministic, and the X pass is an
address-collision machine

A dev-only diagnostic (`L17_ASDBG`, kept in the file) prints the page-offset
relations at execute time.  Across 24 processes on wallaby: `(in − t1) mod
4096 = 2624` and `(out − t1) mod 4096 = 3456` — **identical in every
process** (glibc mmap's the big allocations; relative offsets are not
ASLR'd).  So address pathologies here are not a lottery, they are a fixed
property of the build — the same on the node every run, invisible to
round-over-round comparison.

The X pass is maximally exposed: its load stream (t1) and store stream (out)
walk 289-complex rows in lockstep — same 16·p₀ offsets, same 4624 B row
stride.  Two consequences:

1. **Line splits**: 4624 ≡ 16 (mod 64), so consecutive rows cycle the four
   16-byte alignment classes and 3 of 4 zmm accesses split a cache line
   (r1 item 10 noted this for `out`, where it is contract-fixed; t1 is MINE
   to pad).
2. **4K aliasing**: with the measured residues, chunk c's loads at row j sit
   within 48 B (mod 4096) of chunk c−1's stores at row j+9 — ~8 false
   store→load dependencies per chunk, every chunk, and each one stalls the
   FMA its load feeds.  This is `ld_blocks_partial.address_alias`, the exact
   counter the r7 VERDICT asked about for L=23.

### What changed: address-safe twins (xla/xlda/xfa/xfda, FORCE 48–51)

*Attribution: t1 stride padding adopted from L23_matrixsimd r7 (its 1058→
1064 `L23_T1P`), upgraded after the collision model showed naive padding is
not enough; the 4K-alias framing follows the r7 VERDICT §on L=23.
L17_rader r7 tried an execute-time scratch-base selection that the node
declined — mine differs in being model-driven and per-volume, but that
precedent is acknowledged.*

* **Padded t1 plane stride, chosen by the collision model, not rounded up.**
  Over all start residues, the best reachable weighted collision count
  (17×17 row pairs, store-buffer lag 1–3 chunks, ±64 B window, both pass
  orders) is: dense 4624 B → 8; **4672 B (73 lines, the "natural" padding) →
  31 — WORSE than dense**; **5120 B (80 lines) → ≤ 4, usually 0.**  So the
  twins use 640 doubles/plane (t1 grows 78.6 → 87 KB, still 9% of node L2),
  which also makes every non-tail X-pass access 64-byte aligned (t1 side).
* **Per-volume de-aliased t1 base.**  `out` (and `in`) slide 78608 ≡ 784
  (mod 4096) per volume, so no single base fixes a batch — but t1 is fully
  rewritten every volume, so its base may legally differ per volume.
  `create()` builds `astab[2][256]`: for each 16-byte residue class of
  (t1 − ref) mod 4096, the 64-byte-step shift minimizing the model's
  collision count (mode 0 = X-last, ref = vout; mode 1 = X-first, ref =
  vin).  Pure integer arithmetic (~30 ms, unscored), machine-independent,
  deterministic.  At execute: one table lookup + pointer add per volume.
* **Bit-discipline**: addresses only, no value changes.  cmp-verified on
  full outputs with dedicated per-batch inputs and a PASS asserted per run
  (the r7 protocol): xla ≡ xlda ≡ xl ≡ xld at B=8 (class B) and xfa ≡ xfda ≡
  xf ≡ xfd at B=256 (class D).  selB grows 7→9, selD 15→17.  The pw/pf grid
  covers xfa/xfda.

### Operation count

Unchanged: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per volume,
node port floor 33 374 cycles = 11.55 µs at 2.89 GHz.  The twins add one
table lookup per volume and zero arithmetic; what they remove is not µops
but **stalls**: ~8 false store→load dependencies per X chunk and the split-
line penalty on 3/4 of X-pass t1 accesses.

### What was measured — wallaby (Gold 6448Y, shared; the process-placement
lottery documented below makes same-window alternation the only statistic)

Forced same-window alternating pairs (6 rounds at B=1/B=8, 3 at B=256),
per-transform minima:

| pair | legacy (per round) | addr-safe (per round) |
|---|---|---|
| B=1: xl vs xla | 10.14, 10.18, 10.14, 10.33, **8.86**, 10.18 | **9.03, 8.92**, 10.17, **8.88, 9.08, 8.94** |
| B=8: xld vs xlda (µs/t) | 10.58, 10.99, 11.11, **8.71**, 10.19, 10.81 | 10.36, **8.89, 8.89, 9.30**, 10.15, **8.88** |
| B=256: xf vs xfa (µs/t) | 10.85, 10.31, 10.11 | **8.72, 8.70, 8.60 (−14%, every round)** |
| B=256: xfd vs xfda (µs/t) | 10.09, 10.07, 10.12 | **8.69, 8.64, 8.64 (−14%, every round)** |

Reading it honestly: at B=256 the twins win **−14% consistently in every
same-window round** — that is a real streaming-regime effect (the X pass
stores t1 there: split stores cost store-buffer occupancy and the alias
model applies to the in→t1 pair).  At B=1/B=8 the picture is
mode-structured: the same fast mode (~8.9 µs) exists for both binaries, but
the addr-safe twin lands in it in 8 of 12 process draws against 2 of 12 for
the legacy twin.  I cannot fully separate wallaby's placement lottery from
the address effect at B=1 (see negatives below) — what I can say is the twin
never created a new slow mode, its best equals the legacy best, and its
typical draw is ~10% better.

Unforced end-to-end (shipping config), all PASS rel_l2 = 3.226e-16 …
3.259e-16, bit-repeatable across processes: **B=1 8.77–8.81, B=8 9.42/t,
B=256 8.70/t (best-ever for this entry on this machine; r7: 9.20), B=2048
13.41/t.**  AVX2 host (wombat): PASS 3.255e-16, repeatable, 24.0 µs/t at
B=8 — the twins run correctly on a 16-register machine (the table shift is
width-independent).

### What was tried and did NOT work — with the numbers

1. **Cosine-resident (r7's lever): dead by disassembly**, +26 instructions
   per chunk (254 vs 228, 210 vs 185); see Finding 1.  Do not revisit under
   gcc 11; a future clang/gcc-14 build could re-run the same objdump check
   in minutes.
2. **Stack-frame page pinning (built, tested, REMOVED).**  Hypothesis: with
   the heap deterministic, the per-process bimodality (9.05 vs 10.15 µs at
   B=1, sd 0.04% within each) must be stack ASLR aliasing spill slots.  A
   4096-aligned wrapper frame + `alloca(64·k)` shifted the exec's frame
   deterministically; sweeping k = 0..56 gave **bimodality WITHIN every k**
   (e.g. k=0: 10.08/9.23/10.09) — falsified.  Pinning cores next: core 51
   consistently 10.55–10.91, core 37 ranging 9.18–11.44 — the wallaby
   lottery is (at least partly) placement/SMT-neighbour luck, not addresses.
   The wrapper was removed; do not rebuild it.  Corollary: r2's "exec6
   bimodality" may have been placement noise too.
3. **A t1-offset env sweep (L17_T1OFF, also removed) looked structured but
   did not reproduce**: k=20 read 8.88 then 17.3 in adjacent windows.  Any
   single-process-per-point sweep on wallaby measures the lottery, not the
   knob.  Only same-window alternation with multiple rounds counts.
4. **Naive line-alignment padding (73 lines = 4672 B) makes 4K aliasing
   WORSE** (model: best reachable collision score 31 vs 8 for dense) —
   fixing line splits and fixing aliasing pull the stride in different
   directions unless the stride is chosen by the model (80 lines does both).
   Anyone porting L23's padding to another geometry: run the collision count
   first, the "round up to whole lines" instinct can backfire.
5. **The raw-ssh trap (r6 item 4, r7 item 2) triggered AGAIN** — a remote
   command without its leading `cd` silently ran gen_input.py from $HOME and
   nearly "verified" against a STALE in256.bin from r5.  Caught by the
   protocol (per-run PASS assertion + explicit regeneration).  Three rounds,
   three triggers: treat every remote one-liner as suspect until its PASS
   line prints.

### Asks for the monitor

1. *(replaces the withdrawn cr ask)* One counter run, B=1, same window:
   `perf stat -e ld_blocks_partial.address_alias,ld_blocks.store_forward,cycles,instructions`
   on `-DL17_FORCE=38` (legacy xl) vs `-DL17_FORCE=48` (addr-safe xla).
   This attributes the node's 44 cycles/chunk directly: if `address_alias`
   is ~8×73×(executes) for F38 and ~0 for F48, the mechanism is confirmed on
   the scoring machine and the r7 VERDICT's L=23 question gets its L=17
   answer for free.  If both read ~0, the node's fixed layout is lucky, the
   twins are a no-op there, and the remaining B=1 overhead is elsewhere.
2. The r7 deferred-Z-at-B=256 forced A/B the VERDICT suggested is still
   worth one run (`-DL17_FORCE=40` vs `43`), independent of this round.

### Expectations for the node

* The picks to read off the description strings: `…, addr-safe t1` in any
  cell means the in-process tuner saw the win on the node's own buffers.
* **B=256/B=2048**: the wallaby −14% was consistent, not mode-luck.  The
  node's DRAM is slower relative to compute, so the store-side share is
  smaller; honest range **B=256 19.5–21.4, B=2048 20.0–21.7** (i.e. −1 to
  −9%).  Winograd is 1.6% behind at B=256 — this is also the defensive move.
* **B=1/B=8**: everything depends on whether the node's fixed layout sits in
  a collision mode.  The counter (ask 1) decides it; blind range **B=1
  13.9–15.3** (−9% if the node is in the bad mode, ~flat if lucky).
* If the node takes the twins and B=1 lands ≤14.5, the 44-cycle mystery was
  substantially address stalls and the next round should extend the model to
  the plane phase (pb↔vin pair, ~2 unlucky planes per volume, X-first out
  planes).  If the counters read clean, B=1 is genuinely at its structural
  window-drain limit and the honest next candidate remains a kernel-shape
  change (Rader's 468-flop line at this data flow), which is a rewrite.

### Next

1. Read the node counters and picks; they decide whether address stalls or
   ROB drain owns the remaining 1.32× at B=1.
2. If the twins win: extend per-volume de-aliasing to the pipelined exec
   (t1/t1b both shifted) and model the plane-phase pairs.
3. If a compiler newer than gcc 11 ever lands on the node: re-run the
   Finding-1 objdump count for cr (and r1 item 1's embedded-broadcast table)
   before anything else — both are one-command checks now.

---

## Round panel_r9 (2026-08-22)

### Standing going in

Held B=1 (15.182), B=8 (16.662) and B=256 (21.311) on the node in panel_r8;
lost B=2048 to L17_winograd (21.645 vs 21.757), their first cell.  The r8
addr-safe twins were picked in all four cells and moved them −0.3/−0.3/−0.6/
+0.4% — wallaby's consistent −14% at B=256 did not transfer.  The r8 VERDICT's
L=17 synthesis is now the frame for everything: **four mechanism classes
falsified on the node** (rescheduling, address alignment, spill deletion,
movement-uop deletion — the last three all *selected* by the node's own tuners
and all ~0), B=1 flat at 15.18–15.23 for five rounds at **1.31× the 33.4k-cycle
mixed-shape port floor** (43.9k cycles at the measured 2.89 GHz), and the
instruction to this geometry: *"stop optimizing and measure, or rewrite"* —
with L36_pfa's in-plan probe named as the pattern the whole panel should adopt
("when the monitor cannot run your counter, build the discriminator into
create() and route it out through fft3d_description()").  The alias-counter
ask from my r8 record is still unrun (fifth outstanding counter ask on the
board).  So this round does exactly what the VERDICT says: build the
discriminating measurement into the plan, and pair it with the one mechanism
class nobody has falsified yet, so that whichever way the probe reads, the
round produces either a fix or a definitive close.

### What changed (two things)

1. **In-plan B=1 residual decomposition probe (`b1dec`)** — *adopted from
   L36_pfa round r8's create()-side probe, with attribution.*  After tuning,
   create() times four configurations, blocked (128 calls per sample, 3
   samples + discarded warmup rep, min), on the plan's own buffers, and
   appends `b1dec[yz/kyz/x/kx]=a/b/c/d` (µs per volume-equivalent) to the
   description string:
   * `yz`  = the plane phase (17× Y-group + Z-group) in situ: source walks a
     78.6 KiB volume (L2-resident), dest = t1 at the padded stride.
   * `kyz` = the IDENTICAL instruction stream with stride arguments 0: the
     same code re-reads one 4.6 KiB plane and re-writes one plane, all L1-hot.
   * `x`   = the X pass in situ: source = t1 (87 KiB, L2), dest = a volume.
   * `kx`  = the same X-pass code with source stride 34: rows overlap into a
     ~9 KiB L1-hot footprint (values junk, timing valid; nothing feeds the
     output path — t1 is fully rewritten by every execute).
   The stride is a runtime argument either way, so hot and in-situ runs retire
   the same instructions; the only variable is where the loads hit.  Port
   floors at 2.89 GHz for calibration: yz 7.83 µs, x 3.71 µs.  Decision table
   for the node's string (which every leaderboard JSON now carries):
   * `kyz ≈ 7.8 and kx ≈ 3.7` while `yz/x` sit high → the residual is
     **L2→L1 fill latency**, and pt (below) is the matching fix.
   * `kyz/kx` high **too** → the kernel is >1.3× its port floor even from
     L1 → front-end/dependency-bound → **B=1 is at its structural limit for
     this kernel family**, said plainly, and the only remaining move is the
     kernel-shape rewrite (or stopping).

2. **`pt`: in-pass prefetcht0 of the volume's own upcoming chunk sources**
   (runtime flag, A/B'd; `-DL17_FORCE_PT` for forced pairs).  The one stall
   class no L=17 mechanism has attacked: pf was cross-volume DRAM-side
   (prefetcht1 of volume b+1), pw is store-side (RFO), deferred-Z was the
   store→load junction.  Nothing ever touched the same-volume load side,
   and the suspect is specific: a chunk is ~228 instructions ≈ one full ROB,
   so the next chunk's ~17 L1-miss loads cannot issue until the current chunk
   drains, putting ~L2-latency serial stalls at every chunk head.  pt issues:
   in the X pass, 17 prefetcht0 (one per source row) inside chunk i for chunk
   i+2's lines (~300 cycles of lead); in the plane phase, the next plane's
   source lines (72–80) in one burst per plane, ~a full group ahead.  Built
   as a SEPARATE X-pass macro (`L17_MIX_XPASS_PT`) under `if (p->pt)`, so the
   pt=0 loop body is byte-for-byte the r8 code — the r8 VERDICT documents
   four regressions from refactors around an untouched hot path, so the
   incumbent path was not touched.  Tail chunks prefetch ≤48 B past the last
   volume of `in` in the X-first order: prefetch is a hint, never faults,
   noted in the macro comment.  Tuning: batch < 64 gets a new blocked pt=0/1
   A/B on the stage-1 winner (only the mixed execs carry the hook); batch >=
   64 extends the stage-2 grid to **(pf, pw, pt) = 8 cells jointly** (r4's
   non-factorizing-knobs lesson).  Description now reports `pt=`.

### Operation count

Unchanged: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per volume,
node port floor 33 374 cycles = 11.55 µs at 2.89 GHz.  pt adds ~17 prefetch
µops per X chunk and ~76 per plane when ON (ports 2/3, which run ~35 of 148
cycles occupied), zero when off; the probe adds ~15 ms of unscored plan time.
B=2048 plan time grew 2.5 → 3.7 s on wallaby (8-cell grid); node ~2 s expected.

### Bit-discipline: verified, and how

Prefetches change no bits, but per the standing protocol nothing was assumed:
on wallaby, full-output cmp with dedicated regenerated inputs and a PASS line
asserted per run: `xla pt=0 ≡ xla pt=1` at B=8, `xfa pt=0 ≡ xfa pt=1` at
B=256 (pf=pw=0 forced), and `xl ≡ xla+pt1` at B=8 (class B intact).  Three
unforced B=1 processes produced bit-identical outputs.  tryout.sh PASS +
repeatable at B = 1, 8, 256, 2048 (rel_l2 3.226e-16 … 3.259e-16) and on the
AVX2 host (wombat, 23.6 µs/t at B=8, PASS 3.255e-16, repeatable).

### What was measured — wallaby (Gold 6448Y, shared; the 2.10↔4.10 GHz clock
lottery and the placement bimodality both visible again; same-window
alternation is the only statistic)

Forced pt pairs, same-window alternation, per-transform minima:

| case | pt=0 (per round) | pt=1 (per round) |
|---|---|---|
| B=1, xla (F48), 6 rounds | **8.936**, 10.111, 8.966, 8.975, 10.084, 10.006 | 8.992, **8.818**, 10.150, **8.847, 8.890, 8.980** |
| B=256, xfa (F50, pf=pw=0), 4 rounds | **2226.0–2410.7 µs/call (best 8.70/t)** | 2297.8–2394.3 (best 8.98/t) |

Reading it honestly: at B=1 both binaries show the documented two-mode
placement lottery; pt=1 drew the fast mode 4/6 against 2/6 and its best
(8.818) edges pt=0's (8.936) — lean-positive, inside the lottery, exactly the
r5-mixed-tail / r6-pw shape of a **pure node bet** (wallaby's two FMA units
and 512-entry ROB are the configuration pt does NOT target; the node's
one-unit/224-ROB is).  At B=256 streaming **pt=1 loses ~3% on wallaby in 3 of
4 rounds** — expected: the X source there is DRAM and a ~150-cycle lead is
useless against it, so the extra µops are pure cost; the in-tuner grid
rejected it on wallaby (8.46 pt=0 vs 8.80 pt=1 at nv=256) and the node's grid
will make its own call.

The probe on wallaby (4.10 GHz windows, B=1 plans, three processes):
`b1dec[yz/kyz/x/kx] = 5.80–5.86 / 5.40–5.42 / 4.55 / 4.21–4.23` — stable to
1% across processes, and internally consistent: yz+x = 10.4 ≈ the mixed
X-last exec's 10.0 in the same tuner table (the probe uses the mixed kernel;
the pure-512 exec wallaby prefers is faster there, 8.6–8.9).  Wallaby's
in-situ/hot ratios are small (yz/kyz 1.07, x/kx 1.04–1.08) — on ITS memory
system the phases are nearly latency-clean, yet kyz sits ~2× wallaby's
2.76 µs two-unit port floor, i.e. wallaby's own residual is in the kernel,
not the caches.  Do NOT extrapolate that to the node: wallaby is 6-wide with
two FMA units; the node is 4-wide with one, which is precisely what the
node's own b1dec string will settle.  A 2.10 GHz window scales all four
numbers by the clock (11.32/10.55/8.86/8.28), as it must.

Shipping (unforced) end-to-end, all PASS and bit-repeatable: B=1 8.61 µs
(8.88 in a second window), B=8 11.8/t (contended window; the binary is
bit-for-bit r8's when the tuner picks pure/pt=0), B=256 **8.58/t (best-ever
for this entry on this machine**; r8: 8.70), B=2048 13.28/t.

### What was tried / observed that did NOT work — with the numbers

1. **pt at streaming batch on wallaby: +3–4%** (grid 8.46→8.80 at nv=256;
   forced 8.70→8.98 best-of-4) — recorded so nobody reads wallaby and deletes
   the flag; the grid rejects it per-machine and it costs nothing when off.
2. **The raw-ssh missing-`cd` trap fired for the FOURTH consecutive round**,
   with a new twist worth recording: I re-sent the same broken one-liner
   THREE times while writing "cd included" in my own run notes — the trap is
   not just forgetting the cd, it is that re-reading your own command after
   composing it is unreliable once you believe it is fixed.  Caught by the
   error line each time (numpy/gen_input missing from $HOME), zero cost.
   Protocol addition: paste the `cd` as the first token, then verify the
   command STRING contains it before running, not after it fails.
3. **The B<64 pt A/B never fires on wallaby** because stage 1 there picks the
   pure 512-bit exec (no pt hook) — so the B=1 pt decision is untested on
   wallaby's tuner path.  Deliberate: the hook set matches the node's actual
   picks (mixed family, all 12 process-cells since r5); if the node's stage 1
   surprises and picks pure, pt correctly defaults to 0.
4. **Not built, deliberately:** any new kernel/scheduling/address variant
   (four falsified classes — the VERDICT's list), a Rader-line rewrite
   (gated on the probe reading "kernel-bound", not before), pt for the
   pipelined exec (its interleaved X slots already fill the gaps pt would
   prefetch into).

### Asks for the monitor

None this round — that is the point.  The b1dec string in every L=17
matrixsimd JSON replaces the r8 alias-counter ask as the primary
discriminator (the counter run would still be welcome corroboration if ever
convenient: `-DL17_FORCE=38` vs `48` at B=1, ld_blocks_partial.address_alias).

### Expectations for the node (pre-registered)

* **The probe.**  Floors: yz 7.83, x 3.71 µs.  Branch (a): `kyz ≲ 8.5` and
  `kx ≲ 4.0` with `yz ≳ 9.5` and/or `x ≳ 5` → fill latency owns the
  residual; expect the B=1 tuner to select `pt=1` and **B=1 13.9–14.8 µs**
  (closing part of the 3.6 µs gap; the plane phase's share transfers only if
  yz also reads high).  Branch (b): `kyz ≳ 9.5` at ~2.89 GHz → the kernel is
  ~1.25×+ its port floor from L1 → pt buys ~nothing (expect `pt=0` or a
  <1% pt=1), **B=1 stays 15.0–15.3**, and the honest conclusion is that B=1
  is front-end/dependency-limited for this kernel family — the first
  positive identification of the 44-cycle/chunk residual in nine rounds, and
  the input the panel needs to decide rewrite-vs-stop.
* **Batched cells**: the grid now searches 8 corners on the node.  If the
  node keeps pf=pw=pt=0 (its r8 corner), expect B=256 ~21.3, B=2048 ~21.7
  (flat, MKL noise aside).  pt=1 at batch on the node would mean its DRAM
  latency is long enough that even a 300-cycle lead pays — possible on CLX
  (slower DRAM than SPR relative to core), not predicted.
* B=8 follows B=1's branch.  No prediction of retaking B=2048; nothing in
  this round targets winograd's 0.5% there specifically.

### Next

1. Read the node's b1dec four-tuple and the pt picks; they decide the fork:
   (a) fill-latency confirmed and pt selected → tune pt pacing (distance 1
   vs 2 chunks, split vs burst plane prefetch) and extend the hook to the
   pure-512 execs so the B<64 A/B also runs if stage 1 flips; (b)
   kernel-bound confirmed → write the finding plainly in the strategy
   record, propose the panel stop funding B=1 at L=17 or fund exactly one
   rewrite arm (Rader's 468-flop line in this entry's lanes-are-lines
   layout), and freeze this entry's B=1 path.
2. If pt was picked anywhere at batch, model where the lead should sit for
   DRAM (the 17-line burst may want to be 2 planes ahead, like L36_pfa's
   deep variant) before building anything.
3. The b1dec probe itself is portable: any entry with a phase structure can
   report in-situ vs L1-hot phase times through its description string.
   L17_winograd and L17_rader should run the same experiment — three
   independent four-tuples on the node would close the L=17 B=1 question
   panel-wide.

---

## Round panel_r10 (2026-08-22)

### Standing going in

Swept all four L=17 cells in panel_r9 for the first time on distributions, not
just minima (B=1 14.866, B=8 16.512, B=256 21.073, B=2048 21.718 — the last a
dead tie with L17_winograd's 21.716), 3.87–5.50× the best library.  The r9
b1dec probe settled B=1: kyz = 10.30 against the 7.83 µs port floor with the
whole working set in L1, phases additive, pt declined in all four cells — the
six-round 1.31× is **intra-kernel issue limitation**, positively identified.
The r9 VERDICT's instruction to this geometry: **stop funding B=1** (nothing
left that is not a rewrite, and winograd's own 42% gate measured 37.4% —
the rewrite is unfunded by measurement), and if one L=17 item is funded, fund
**the streaming cells**, where matrixsimd and winograd tie at ~21.7 µs =
~10.9 GB/s against ~236 KB/volume of compulsory DRAM traffic (in-read 78.6 KB
+ out-RFO 78.6 + out-writeback 78.6), with NT stores dead four rounds and
ERMS dead — "this needs a new idea or an honest closed."  This round supplies
one of each: the instrument that decides which, and the one traffic-shape
mechanism nobody has tried.

### The traffic ledger that frames the round

At B=2048 the cell time (21.72 µs) equals the compulsory traffic (236 KB)
divided by 10.9 GB/s.  If the node's own single-core copy speed on this
stream mix is ~11 GB/s, the cell IS the memory system: compute (14.9 µs) is
already fully hidden under the streams, no scheduling change can move it, and
the only lever that could — deleting the RFO — does not exist on CLX without
NT stores (no SpecI2M until ICX, no cldemote/movdir64b).  Two independent
implementations landing 2 ns apart is exactly what an external wall looks
like.  But if the machine can actually move this mix at 12.5+ GB/s, there is
1.5+ µs of recoverable inefficiency, and the prime suspect is specific: in
the X-first order the volume's entire DRAM read is the X pass, which walks
`in` as **17 interleaved streams of one row (4624 B = 1.13 pages) each** —
short-lived per page, so the L2 streamer pays lock-on repeatedly and coverage
is partial — issued by the same instructions that feed the FMAs.

### What changed (two things)

1. **`sbw` — in-plan streaming bandwidth decomposition** (the b1dec pattern
   from r9, pointed at the batch regime; instrument lineage L36_pfa r8, with
   attribution).  At batch >= 64, create() times four pure memory patterns on
   the >L3 tuner arena (blocked, min of 3, warmup rep discarded), µs per
   78.6 KiB volume-equivalent, appended to the description string as
   `sbw[rd/wr/cp/s17]`:
   * `rd`  = sequential zmm read of a volume (best-case read),
   * `wr`  = sequential zmm write, plain stores (RFO + writeback),
   * `cp`  = per-volume read burst then write burst — the X-first exec's own
     phase alternation with the compute deleted: **the streaming floor**,
   * `s17` = the X pass's actual read pattern (17 row streams, one 64 B line
     per row per step, rows 4624 B apart, split lines and all).
   Ledger, pre-registered: **cell ≈ cp ⇒ the streaming cells are
   bandwidth-closed on this machine and should be declared so**; cell >> cp
   with s17 >> rd ⇒ the read shape is the recoverable share and the staged
   twins (below) are the matching fix; cell >> cp with s17 ≈ rd ⇒ the
   headroom is write-side/overlap and staging will not be picked.

2. **Staged-input streaming twins** (`l17_execm_xfsa`/`xfdsa`, FORCE 52/53,
   candidates 46/47, selD grows 17 → 19).  During volume b's compute-rich
   plane phase, volume b+1's input is copied **sequentially** (the
   prefetch-friendliest stream that exists) into an L2-resident stage, in two
   half-plane chunks per plane slot (288+290 doubles, unaligned zmm moves);
   volume b+1's X pass then reads the stage at L2 latency.  DRAM traffic
   unchanged — the read is moved and reshaped, not duplicated — at the cost
   of a 78.6 KiB L2 store+reload and ~2.5k copy µops per volume in a phase
   that is not port-bound at batch.  Differs from the r4 pipelined execs
   (which move the X *compute* into the plane phase but keep the 17-stream
   read shape) exactly where `s17 − rd` prices the difference.  Details:
   * stage lives in t1b (the pipelined execs' second buffer, never
     co-selected), with a **per-volume base shift keeping (stage − vin) mod
     4096 ≈ 2048** — the copy is otherwise the textbook 4K-aliasing case
     (1:1 load/store at equal offsets from sliding bases);
   * the X pass's t1 shift reuses the r8 astab, mode 1, with the **stage as
     the load-side reference** (a stage is a dense volume image, rows 4624 B
     apart — exactly the geometry the mode-1 model scores);
   * volume 0's X pass reads `in` directly (same bits; never reads a stale
     stage, so repeated executes on one plan stay bit-identical);
   * hooks: pw only.  pf is meaningless (the copy IS the cross-volume input
     touch), pt is not offered (the X pass reads L2-hot).  Stage-2 grid
     conditions updated accordingly.

### Operation count

Unchanged: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per volume,
node port floor 33 374 cycles = 11.55 µs at the measured 2.89 GHz.  The
staged twins add zero arithmetic; per volume they add ~1229 unaligned zmm
loads + 1229 stores (the copy) plus 34 scalar tail moves, and convert the X
pass's ~1229 DRAM-demand loads into L2 hits.  The sbw probe adds ~130 ms of
unscored plan time at batch (the arena is re-filled once after the stage-2
free); B=2048 setup on wallaby grew 3.7 → 3.8 s.

### Bit-discipline: verified, and how

Copies preserve bits and every chunk runs in the same order on the same
values, so the twins must be class D — **verified, not assumed** (r3
protocol, per-run PASS asserted, dedicated same-seed inputs): on wallaby at
B=256, full-output cmp gives F52 ≡ F50 (xfsa ≡ xfa), F53 ≡ F51 (xfdsa ≡
xfda), and F50 ≡ F51 (class D internal consistency re-confirmed); F51 ≡ F53
re-verified on the final shipped binary after the last (fprintf-only) edit.
tryout.sh PASS + bit-repeatable across processes at B = 1, 8, 256, 2048
(rel_l2 3.226e-16 … 3.259e-16) and on the AVX2 host (wombat, B=64, PASS
3.258e-16, repeatable — the vd8u copy compiles to 2×ymm there and the sbw
probe runs correctly on a 16-register machine).

### What was measured — wallaby (Gold 6448Y, shared; same-window in-process
tuner tables are the statistic, as always)

Tuner >L3 tables (one window each), µs/transform:

| variant | nv=256 (B=256 plan) | nv=1000 (B=2048 plan) |
|---|---|---|
| X-first addr-safe (F50 class) | 8.61 | 14.97 |
| X-first deferred-Z addr-safe (F51, r9 node pick) | **8.52 <== kept** | 14.90 |
| **staged (F52)** | 9.64 (+13%) | 17.04 (+14%) |
| **staged deferred-Z (F53)** | 9.73 | 17.08 |
| 512-bit pipelined | 9.22 | — |
| 512-bit NT store | 12.29 | **12.69 <== kept** |

The staged twins **lose ~13–14% on wallaby in both regimes** — the expected
sign on a machine with two FMA units, DDR5, a 2 MB L2 and prefetchers strong
enough that the whole cell runs at 1.5× its own cp floor with NT stores
available.  Like the r5 mixed tail (wallaby +3%, node −7%) and r6 pw, this
is a **pure node bet**, and unlike those, this one ships WITH its own
discriminator: the node's `s17 − rd` says in advance whether the mechanism
has anything to collect.

The sbw probe on wallaby (the probe's own first data):

| arena | rd | wr | cp | s17 | s17/rd |
|---|---|---|---|---|---|
| nv=256 (40 MB, ~L3-resident there) | 1.94 | 2.23 | 5.50 | 2.54 | **1.31** |
| nv=1000 (157 MB, streaming) | 2.52 | 5.18 | 8.38 | 3.23 | **1.28** |

Two readings that matter: **the 17-stream X-pass read pattern is 28–31%
slower than a sequential read of the same bytes even on Sapphire Rapids'
prefetchers** — the mechanism the staged twins attack is real and measured,
not hypothesised; and wallaby's cell (12.6–13.3 µs unforced) sits at ~1.5×
its own cp (8.38), i.e. wallaby is NOT bandwidth-closed, consistent with NT
stores winning there.  Whether the node's cell sits ON its cp is precisely
what the leaderboard string will now say.

Unforced end-to-end (shipping config), all PASS and bit-repeatable:
B=1 10.00 µs (placement-lottery window, path bit-for-bit r9's; sd 0.01%),
B=8 12.02 µs/t (contended window), B=256 **8.54 µs/t (best-ever for this
entry on this machine; r9: 8.58)**, B=2048 13.25 µs/t (flat vs r9's 13.28;
NT-store pick as every round on wallaby).

### What was tried / observed that did NOT work — with the numbers

1. **Staged twins on wallaby: +13% (9.64 vs 8.52, nv=256) and +14% (17.04 vs
   14.90, nv=1000).**  Recorded so nobody reads wallaby and deletes them; on
   wallaby's economics (compute-bound cells, strong prefetchers, s17/rd only
   1.28) the copy µops are pure cost.  The node bet rides on the node's
   weaker streamer, DDR4, one FMA unit — and is priced by its own probe.
2. **The missing-`cd` ssh trap fired for the FIFTH consecutive round, in its
   nastiest form yet**: I re-sent the same broken one-liner THREE times,
   adding `pwd` on the third attempt to discover the `cd` I "knew" was there
   was genuinely absent from the executed string each time.  Protocol update
   that actually terminated the loop: **stop hand-composing remote cd
   one-liners entirely** — route every forced run through `./tryout.sh --on
   wallaby` (which does its own cd) and do file copies/cmps in separate ssh
   commands built from absolute paths only.  Zero failures after switching.
3. **Not built, deliberately**: anything touching B=1/B=8 (closed by r9's
   b1dec + VERDICT; the class-B paths are bit-for-bit r9's), NT-store
   revival on the node (dead four rounds), a full pipelined+staged hybrid
   (three interacting schedules in one exec with no node signal for either
   half — if the node picks staged AND the s17 gap is large, that is next
   round's one candidate).

### Asks for the monitor

None.  The sbw four-tuple in every batch-cell JSON is the deliverable; the
grid decides staged-vs-not on the node's own buffers.

### Expectations for the node (pre-registered)

* **The probe.**  The compulsory ledger says the B=2048 cell (21.7 µs) is
  bandwidth-closed iff cp reads ≈ 20.5–22.  Branch (a): `cp ≥ 20.5` → the
  streaming cells are **closed on this machine** — the transform already
  runs at the node's own memcpy-with-RFO speed, compute fully hidden; say so
  plainly and stop funding them (staged may still be picked at ±1%, it
  cannot matter).  Branch (b): `cp ≤ 18.5` AND `s17/rd ≥ 1.3` → 3+ µs of
  read-shape headroom exists; expect the grid to take a staged twin and the
  cell to land 19.5–21.0.  Branch (c): `cp ≤ 18.5` with `s17 ≈ rd` → the
  headroom is write-side/overlap, staged declines, and next round's item is
  the write stream (or pipelined+staged).  My honest prior, from the two-
  entry dead tie at 10.9 GB/s: **branch (a), with cp ≈ 21 ± 1.**
* **Cells**: B=1/B=8 bit-for-bit r9's configuration — 14.87/16.51 should
  stand to layout noise (±2%, which the panel now refuses to read as
  signal).  B=256/B=2048: flat under branch (a)/(c); 19.5–21.0 under (b).
* B=256's arena on the node is 40 MB vs 22 MB L3 → genuinely streaming, so
  its sbw string is valid there too (unlike wallaby's nv=256 reading, which
  is L3-resident and quoted above only for the s17/rd ratio).

### Next

1. Read the node's sbw four-tuple against the three branches; it decides
   between "closed" (write it in the record and the VERDICT can retire the
   geometry), "staged won" (extend staging to the pipelined exec as the one
   follow-up), and "write-side" (model the out-stream's interaction with the
   RFO stream before building anything).
2. If a future round ever gets NT stores that don't lose on the node (new
   compiler, new microcode, or an ICX+ node), the staged twins compose with
   NT trivially (the stage copy and the NT staging are the same shape on
   opposite sides) — but do not build that speculatively; NT is 0-for-4.
3. The sbw probe is portable to any entry with a phase structure
   (L17_winograd's ph/xp maps 1:1); a second independent four-tuple on the
   node would make the "closed" verdict panel-wide rather than mine.

---

## Round panel_r11 (2026-08-22)

### Standing going in

Took three of four cells in panel_r10 (B=1 14.995, B=8 16.735, B=256 21.001;
B=2048 21.621 on minima but L17_winograd's on disjoint medians, 21.723 vs
21.940 — the VERDICT reads it as "a tie that happens to be statistically
clean"), 3.89–5.45× the best library, still the panel's largest margin.  The
round's real result was the sbw probe's node reading, which **re-opened the
streaming cells the panel was ready to close**:

* `sbw[rd/wr/cp/s17] = 4.61–5.10 / 7.56–8.22 / 13.05–13.31 / 3.72–4.16` at
  B=2048 (three processes, tight).  So: the cell (21.6 µs) runs 8.4 µs above
  the machine's own read-burst-then-write-burst floor for its own traffic
  (cp) — **the cells are NOT bandwidth-closed**; `s17/rd = 0.82` — the
  17-stream X read is *faster* than sequential on the node (opposite sign to
  wallaby's 1.28), so the read shape has nothing to collect and the staged-
  input twins were correctly declined by the node's tuner; and `cp ≈ rd + wr`
  — **the node overlaps DRAM read bursts and write bursts not at all.**
* b1dec again read kyz = 10.2–10.5 vs the 7.83 floor from L1 (B=1 is
  kernel-issue-bound, closed since r9; pt=0 everywhere again).

The r10 VERDICT §6 makes this geometry's batched cells "the highest-value
item on the whole board" and names the build precisely: *"stage the output
densely and pace its flush under the next volume's compute"* — the write-side
analogue of my r10 staged-input twins.  My own r10 pre-registration called
this branch (c) ("headroom is write-side/overlap, staged declines, next
round's item is the write stream").  Both agree; this round builds exactly
that and nothing else.

### What changed: staged-output twins (l17_execm_xfso/xfdso, FORCE 54/55, candidates 48/49, selD 19 → 21)

*Attribution: the mechanism is the r10 VERDICT §6's named build.  L17_rader
built the same shape in r10 (`stp`) and the node declined it — the VERDICT
attributes that to their x pass's 17 concurrent misaligned output streams
being the wrong baseline, and explicitly prices the same construction from
this entry's dense-store structure as "worth exactly one candidate".  The
copy pacing and the (stage − ref) mod 4096 ≈ 2048 anti-alias rule are my own
r10 staged-input machinery, reused on the opposite side; the pw hook is the
r6 one (L8_fusedaxes/L36_pfa lineage), repointed.*

The diagnosis the build answers: in the X-first order the Z group stores
each finished 4.6 KiB plane straight to `out` — 73 RFO-missing lines in one
burst, more than the store buffer holds, issued by the same instruction
stream that computes.  When the RFO misses go to DRAM (batch), the store
buffer fills and stalls the FMA stream — that is the write-side non-overlap
the cp ≈ rd + wr reading exposes.  Fix:

* Volume b's Z groups store their planes into an **L2-resident stage**
  (so0/so1 ping-pong, new 91 KB buffers; stores hit L2, no stall).
* Volume b's stage is **flushed to out during volume b+1's plane phase**, in
  half-plane bursts (289 doubles = 36/37 lines — sized to the 56-entry store
  buffer) at the two group boundaries of each plane slot, so every burst
  issues and then drains under a full compute group (~640 ns on the node).
  The flush is a bit-exact unaligned-zmm copy (the r10 L17_STG_CPY8).
* **The flush never enters the X pass**: cp ≈ rd + wr says the node
  serializes DRAM reads against DRAM writes, so the X pass's read burst
  keeps the pipe to itself.  Last volume's stage flushes in an epilogue
  (~8 µs once per call: ≤0.13 µs/t at batch ≥ 64; nothing at B=1 where the
  twins are not selectable anyway).
* **Addressing**: each stage takes a per-volume base shift keeping
  (stage − vout_b) mod 4096 ≈ 2048 — the flush is a 1:1 copy at equal
  offsets from sliding bases, the textbook 4K-alias case (r10 rule).  The
  Z-group stores to stage b then sit 784 B (mod 4096, the out volume
  stride's residue) from the flush loads of stage b−1 — clear of the ±64 B
  window.  t1 keeps its r8 astab mode-1 shift (X pass unchanged).
* **Hooks**: pw now prefetchw's the *flush destination* lines one compute
  group ahead of each burst (the stream that actually RFO-misses; the Z
  stores no longer need it).  pf is NOT offered (it would push volume b+1's
  input reads into the write-paced plane phase — the exact mix cp says
  serializes); pt is not offered (X pass unchanged).  Stage-2 grid for these
  twins is therefore pw ∈ {0,1} only.

### Operation count

Unchanged: 148 vector FP ops per chunk, 208 zmm + 35 ymm chunks per volume,
node port floor 33 374 cycles = 11.55 µs at the measured 2.89 GHz.  The twins
add zero arithmetic; per volume they add ~1229 unaligned zmm load+store pairs
(the flush) plus a 78.6 KiB L2 store+reload round trip, and convert the
plane phase's ~1229 RFO-missing demand stores into L2 hits.  DRAM traffic is
byte-identical (the RFO+writeback moves to the paced flush).  Plan time at
B=2048 on wallaby grew 3.8 → 4.3 s (two more candidates × 1000-volume arena).

### Bit-discipline: verified, and how

The twins store the same bits at a staged address and copy them bit-exactly;
every chunk runs in the same order on the same operands as xfa/xfda → class
D.  Verified per the r3/r10 protocol (dedicated tryout runs, PASS asserted
per run, cmp on full outputs via absolute paths, all on wallaby):
**F54 ≡ F50** and **F55 ≡ F51** at B=256 (pf=pw=pt pinned 0), **F50 ≡ F51**
re-confirmed, rel_l2 = 3.258e-16 on every run, `repeatable: identical output
across runs` on every run including unforced B=1/B=256/B=2048 and the AVX2
host (wombat B=64: PASS 3.258e-16, repeatable — the copy compiles to 2×ymm
there).  Epilogue path exercised standalone: forced F54 at B=1 (PASS,
9.11 µs) and F55 at B=8 (PASS, 9.03 µs/t).

### What was measured — wallaby (Gold 6448Y, shared; same-window in-process
tuner tables are the statistic, as always)

Tuner >L3 table, nv=1000 (B=2048 plan), one window, µs/transform:

| variant | µs/t |
|---|---|
| X-first addr-safe (F50 class) | 15.78 |
| X-first deferred-Z addr-safe (F51, r9/r10 node pick) | 16.61 |
| staged input (F52/F53, r10) | 18.77 / 18.74 |
| **staged OUTPUT (F54/F55, this round)** | **17.75 / 17.77** |
| 512-bit pipelined | 16.18–16.91 |
| 512-bit NT store | **13.08–13.37 <== kept** |

So on wallaby the staged-output twins lose ~13% to the plain twin — the
expected sign for the fourth time (r5 mixed tail, r6 pw, r10 staged input all
lost here and the first two then won on the node): wallaby has two FMA units
(plane-phase compute is short, less room to hide the flush), DDR5 with
strong prefetchers, NT stores that win outright, and — its own sbw says —
the same read/write serialization but half the write cost (this round's
wallaby sbw: rd 3.30 / wr 6.64 / cp 9.87 / s17 3.01; note cp ≈ rd + wr here
too, and wallaby's NT cell 13.6 µs/t sits at 1.38× its own cp — wallaby is
not closed either, but NT is legal traffic-deletion there and the node has
no working NT).  Worth recording: staged-output beats staged-input by 1 µs
on wallaby even while losing — the write side is the cheaper stage even on
the wrong machine.

Unforced end-to-end (shipping config), all PASS and bit-repeatable: B=1
9.98 µs (placement-lottery window; path bit-for-bit r10's), B=8 8.64 µs/t,
B=256 13.78 µs/t (contended window, sd 9.6%; forced same-day F50/F51 windows
read 8.61/8.68), B=2048 13.62 µs/t (NT pick, as every round on wallaby).

### What was tried / observed that did NOT work — with the numbers

1. **Staged output on wallaby: +12–13% streaming** (17.75 vs 15.78 at
   nv=1000; forced 10.17 vs 8.61 µs/t at B=256, different processes).
   Recorded so nobody reads wallaby and deletes the twins: this is a pure
   node bet, and unlike r5/r6 it ships with the discriminator already
   measured on the node — the 8.4 µs cp gap and the wr = 7.6–8.2 µs write
   term are exactly what the mechanism attacks.
2. **The missing-`cd` ssh trap, SIXTH consecutive round, terminal form and
   the actual fix.**  I composed the remote one-liner four times, each time
   believing the `cd` was present; an `echo "$CMD"` before execution proved
   it absent every single time (the r9 observation — "re-reading your own
   command after composing it is unreliable once you believe it is fixed" —
   is apparently a law).  What terminated it for good: **never compose a
   remote `cd` at all — invoke tryout.sh by ABSOLUTE PATH**
   (`ssh wallaby '/home/.../tryout.sh …'`); it does `cd $(dirname $0)`
   itself.  A guard that `case`-matches the command string for `cd /home/*`
   before running is the belt to that suspenders.  Zero failures after.
3. **Not built, deliberately:** NT flush of the stage (NT is 0-for-4 on the
   node across four rounds; if the node ever gets working NT the stage copy
   composes with it trivially, r10 item 2); pf on the staged-output twins
   (cp ≈ rd + wr — do not mix the input read stream into the write-paced
   phase); flush bursts interleaved into the X pass (same reason);
   quarter-plane pacing (only two group boundaries exist per plane slot;
   36-line bursts already fit the store buffer); anything touching B=1/B=8
   (closed by r9 b1dec, kernel-issue-bound; the class-B paths are
   bit-for-bit r10's).

### Expectations for the node (pre-registered)

* **The mechanism's own ledger**: X phase (~4.6 µs compute, reads overlap
  it at s17 ≈ 4.0) + plane phase max(compute ~10.9, paced writes ~8.0) +
  copy overhead ≈ **16–18 µs at B=2048** if the flush hides as designed —
  against 21.6 today.  Honest range **17.5–20.5**, because the L2
  store+reload and the flush loads compete with t1/pb traffic in ways only
  the node can price.  Expect the grid to take **pw=1 with the twins** (the
  flush destination is the only RFO-missing stream left).  B=256 (40 MB,
  genuinely streaming on the node): same mechanism, expect 19.5–21.0 from
  21.0.
* **If the node declines the twins**: the write-side stall is not
  store-buffer/RFO-shaped, and since s17/rd already cleared the read side
  and b1dec cleared compute, the residual would be the DRAM scheduler
  itself (read/write turnaround on one channel set) — at which point the
  honest next step is to say the cell is closed at ~1.6× cp for a
  single-threaded plain-store machine and move the panel's attention
  elsewhere.  Either way the round produces a verdict, not a maybe.
* **B=1/B=8**: bit-for-bit r10's configuration; 14.99/16.74 should stand to
  layout noise (the ±2% band the panel refuses to read as signal).

### Next

1. Read the node's picks and the B=2048/B=256 deltas.  If a staged-output
   twin won: tune the pacing (burst size vs store-buffer occupancy, and
   whether the epilogue should be paced under the SAME volume's X pass at
   small batch) and extend the stage to the pipelined exec (pipelined X
   compute + staged output = both bursts gone; three interacting schedules,
   so only with a node signal for both halves — the r10 rule).
2. If pw=1 was picked with the twins, note it as the first node pw
   selection at L=17 and check whether pw alone (on F51) now also moves —
   the grid's 2×1 corner tells.
3. If the twins were declined, write the closure argument of expectation
   bullet 2 into the record and propose the monitor confirm with one
   counter run (`offcore_requests_outstanding` or store-buffer-full stalls
   on F51 vs F55, B=2048) before the panel retires the geometry.
