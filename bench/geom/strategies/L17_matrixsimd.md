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
