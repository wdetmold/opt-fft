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
