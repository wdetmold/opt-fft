# L6_pfa — Good–Thomas / prime-factor 6-point codelet, 3D row–column, AVX2

`impl/L6_pfa.c` · geometry `L = 6` (216 complex doubles/volume) · `fft3d_name() = "L6_pfa"`

---

## Round 1 (2026-08-21, first implementation)

### Technique

Row–column 3D transform: three axis passes, every 6-point line done by the Good–Thomas
(prime-factor) algorithm for `6 = 2·3`. Derivation, because the whole point is that it is
twiddle-free:

With the CRT index maps `n = (3n1 + 4n2) mod 6`, `k = (3k1 + 4k2) mod 6`
(`n1,k1 ∈ Z2`, `n2,k2 ∈ Z3` — note `3n1 ≡ n1 (2)`, `4n2 ≡ n2 (3)`),

```
W6^(nk) = W6^(3 n1 k) · W6^(4 n2 k) = (-1)^(n1 k1) · W3^(-n2 k2)     exactly
```

so

```
A[n1][k2] = Σ_{n2} x[(3n1+4n2) mod 6] · W3^(-n2 k2)      (2 conjugate DFT3s)
X[(3k1+4k2) mod 6] = Σ_{n1} (-1)^(n1 k1) · A[n1][k2]     (3 DFT2s)
```

**No twiddle factor appears anywhere between the two stages** — that is PFA's whole
selling point against Cooley–Tukey, which for `6 = 2·3` needs a `W6` diagonal between the
radix-3 and the radix-2 step.

**The permutation is free, and this is the single most useful fact about PFA at n = 6.**
Both index maps are the *same* map `p = [0,4,2,3,1,5]`, which is an involution that only
swaps 1 ↔ 4. Concretely the two radix-3 groups are `a = (x0,x4,x2)`, `b = (x3,x1,x5)` and
the outputs land as

```
X0 = A0+B0   X4 = A1+B1   X2 = A2+B2
X3 = A0-B0   X1 = A1-B1   X5 = A2-B2
```

In a fully unrolled codelet that costs **zero instructions**: it is only which register
goes into which slot. There is no index table in this implementation, in 1D or in 3D.

Per DFT3 (conjugate twiddle `W3^-1 = -1/2 + i·√3/2`), with `t1 = a1+a2`, `t2 = a1-a2`,
`m = a0 - t1/2`, `s = swap_re_im(t2)`:

```
A0 = a0 + t1                 add
m  = fnmadd(0.5, t1, a0)     fma           A1 = fmadd (cn, s, m)
                                           A2 = fnmadd(cn, s, m)     cn = (-C,+C), C = √3/2
```
= **6 FP instructions + 1 vpermilpd** per DFT3, i.e. the multiply by `i` is a lane swap
plus a sign folded into the constant. Folding `C·t2` into the two output FMAs instead of
computing `b = C·t2` first saves one instruction per DFT3 (6 instead of 7).

### Operation count

| unit | flops | FP instructions |
|---|---|---|
| DFT2 | 4 | 4 |
| DFT3 | 16 (4 mul + 12 add) | **6** (3 add, 3 fma) |
| **DFT6 = 2·DFT3 + 3·DFT2** | **44** (8 mul + 36 add) | **36** scalar-shaped |
| **6³ volume** = 3 axes · 36 lines | **4752** | **3888** |

Matches FFTW's own `n1_6` accounting exactly (`docs/literature/02` §5.3 quotes 48 flops /
36 instrs on the convention that charges DFT3 18 flops; on that same convention this file
is identical). **There is no arithmetic left to win at L = 6.**

Against the alternatives, per volume:

| decomposition | real mults | real flops | FP instrs | vectorised ymm ops |
|---|---|---|---|---|
| **PFA 2×3 (this file)** | **864** | **4752** | **3888** | **972** |
| Cooley–Tukey 2×3 (+2 nontrivial `W6` per line) | 1728 | 6048 | 4752 | 1188 |
| `python/fft3d.py` `line_cost(6)=30` cplx MACs | — | ~19440 | — | — |

So PFA **halves the multiplies** (864 vs 1728), removes 21.4% of the flops and 18.2% of
the instructions versus radix-2×radix-3 Cooley–Tukey, and removes the twiddle *table* (no
constant loads at all beyond two broadcast registers, `0.5` and `(-C,+C)`).

### Layout and SIMD decisions

**Interleaved complex, 2 complex per ymm, lanes = 2 independent lines. Never
deinterleaved.** This is the decision the whole file turns on, and it goes against §04's
general advice to use split re/im — because at L = 6 the split form's win (no `×i`
shuffle) is smaller than its cost (deinterleave/reinterleave of 432 doubles per volume).
Counting per pair of lines:

* split re/im, 4 lanes/ymm: DFT6 = 36 FP instrs per 4 lines = **9 FP/line**, 0 shuffles,
  **plus** ~4 shuffles/line amortised for the interleave↔split conversion of the volume.
* interleaved, 2 lanes/ymm: DFT6 = 18 FP + 2 vpermilpd per 2 lines = **9 FP/line +
  1 shuffle/line**, and **no conversion at all**.

Identical FP cost, 4× less shuffle traffic. And the slot efficiency is exact: 18
instructions × 4 doubles = 72 slots carrying 88 flops (8 of the 18 are FMA), i.e. **100%
lane utilisation with zero padding** — because the lane set for each pass is the other 36
indices, and 36 is even.

Which axis gets the transposes: with `in`/`out` in the driver's `[x][y][z]` layout, a
vector of 2 adjacent `z` is a single aligned 32-byte load, so the **x and y passes are
completely transpose-free** (strided vector loads, stride 72 and 12 doubles). Only the z
pass needs lane-axis changes: load 2 adjacent y-rows (6 loads), 6 × `vperm2f128` to get
lanes = (y, y+1), DFT6, 6 × `vperm2f128` back, 6 stores. 12 shuffles/set × 18 sets = 216.
**That is provably minimal**: the lane axis must differ from the transform axis in each
pass and must be `z` at both ends (input and output layout), so at least one pass has
axis = z and at least two lane-axis transitions are unavoidable; every ordering of the
three axes costs exactly 12 shuffles/set. Verified by construction for the (z,x,y) order
with an `[x][z][y]` intermediate — same 216.

Alignment: every vector access is 32-byte aligned, because 36, 6 and the z-pair offset are
all even numbers of complexes and the driver aligns both buffers to 64 B.

Per volume (variant 1): **972 ymm FP + 324 loads + 324 stores + 324 port-5 shuffles**
≈ 2050 uops. Bounds on the scored Cascade Lake part: FP 972/2 = **486 cycles**, front-end
2050/4 ≈ 512, stores 324, p5 324. So it is FP/front-end bound, which is why the fused
variant (below) exists.

### Why there is no 512-bit path (the AVX-512 question, answered analytically)

**Deliberate: the file contains no `__m512d` at all.** Reasons, in order of weight:

1. The scored node is a **Xeon Gold 5218 — a single-AVX-512-FMA-unit SKU** (Intel Opt.
   Manual §18.21; Gold 5xxx/Silver/Bronze have one, Gold 6xxx+/Platinum have two).
2. §18.20: with 512-bit registers in the RS, **FMA and SIMD both dispatch to ports 0 and
   5 only**, versus ports 0/1 for FP and port 5 for shuffles in the 256-bit scheme. With
   one FMA unit that means 512-bit FP = **1 op/cycle × 8 lanes = 8 double-ops/cycle**,
   exactly what 256-bit already gets from 2 ops/cycle × 4 lanes — **zero FP upside** — and
   our 324 shuffles/volume would then *contend with the FP* on port 5 instead of running
   free alongside it.
3. Licence 2 downclocking on top (§04 §8.2: Gold 5120 measured 2.9 GHz AVX-512 vs 3.1 GHz
   AVX2 at one active core), for no throughput gain.
4. Geometry: `6` is not a multiple of 4, so a 4-complex zmm does not tile a line's lane set
   without masking, and only the x pass has a naturally contiguous 4-complex lane set
   (36 = 9 zmm); y and z would need 2 loads + 1 insert per vector, i.e. **more** uops.

What *is* taken from AVX-512, for free: §18.20 also says "AVX-512 encoded instructions
with YMM registers are considered to be 256-bit", so on the target `-march=native` gives
these same kernels **32 evex ymm registers at licence 1**. Variant `fused=1` is built
specifically to spend them (26 live vectors).

### Variants and the plan-time tournament

Eight kernels from three compile-time knobs, `variant = 1 + fused + 2·nt + 4·pf`:

* `fused=0` — three passes through a 3.4 KB scratch; ~14 live ymm, unspilled on 16-register
  hosts.
* `fused=1` — x pass, then **y and z fused over one 6×6 (y,z) plane held entirely in
  registers** (18 live + ~8 temps): 216 loads + 216 stores instead of 324 + 324, i.e.
  ~1834 uops (front-end 459 cycles, under the 486-cycle FP floor). Spills on AVX2 hosts
  (20 stack refs in the Haswell asm), should not on the target's 32 evex ymm.
* `nt=1` — non-temporal stores in the z pass. Its six 32-byte stores per set cover exactly
  three whole 64-byte lines **in ascending order**, so write-combining is complete.
* `pf=1` — `prefetcht0` of the *next* volume, 3 lines per pass-1 set = all 54 lines, one
  volume (~800 cycles) of lead time.

`fft3d_create` (a) checks every candidate against the scalar reference path and drops any
that disagrees beyond rel-L2 1e-13, then (b) runs a **round-robin** timing tournament
(9 rounds × 8 candidates × ~0.6 ms, per-candidate minimum) on a dummy buffer of the real
batch size, and freezes the winner in the plan. Setup 0.05–0.20 s, excluded from the score.
`execute()` therefore stays deterministic and repeatable.

### What was measured — Haswell E5-2680 v3, ~3.19 GHz, shared login node

`gcc 11 -O3 -march=native -mtune=native`, per transform, min over 4 driver runs of 20
samples. **The dev host has no AVX-512 so the numbers below are the 256-bit path, which is
also what the target will run.**

| B | winner | µs/transform | note |
|---|---|---|---|
| 1 | v1 (`fused=0,nt=0,pf=0`) | **0.251** | ~800 cycles; whole volume L1-resident |
| 2 | v1 | 0.263 | |
| 8 | v1 / v5 | 0.269–0.272 | 54 KB working set, leaves L1 |
| 64 | v8 | 0.294 | |
| 128 | v8 (`fused=1,nt=1,pf=1`) | **0.301** | past L2 |
| 512 | v8 | 0.304 | |
| 2048 | v8 | 0.301–0.324 | 14 MB, L3-resident |
| 8192 | v8 | 0.357 | 54 MB, DRAM-bound |

Accuracy: `rel_l2 = 2.3–2.6e-16` at every batch size tested (B = 1,2,3,7,8,64,128,512,
2048,8192), `rel_max ≤ 4.5e-16`; all eight AVX2 variants and the scalar path are
bit-identical to each other. Repeatability is exercised by construction — the driver runs
tens of thousands of executes on one plan before the checked one.

Full variant × batch matrix (µs/transform, same host, one run each):

```
B        v1      v2      v3      v4      v5      v6      v7      v8
1     0.2530  0.2650  0.4230  0.4060  0.2520  0.2690  0.6010  0.6510
8     0.2804  0.2868  0.3544  0.3010  0.2707  0.2814  0.3584  0.3259
128   0.3743  0.3815  0.3862  0.3428  0.3549  0.3648  0.3473  0.3012
512   0.3852  0.3960  0.3949  0.3476  0.3524  0.3655  0.3497  0.3029
2048  0.3782  0.4035  0.3986  0.3539  0.3656  0.3750  0.3617  0.3082
```

**Context for the target node**: the standing SOTA at L=6 B=1 is MKL at 0.370 µs on the
*scored* machine. This implementation already does 0.251 µs on a *slower* host, and the
Haswell number is structurally pessimistic: **Haswell issues `vaddpd` on port 1 only**, so
the local floor is 648 cycles (the 648 adds/volume), while Skylake/Cascade Lake issues
FP add on ports 0 *and* 1, giving a 486-cycle floor. Measured 801 cycles = 1.24× the local
floor; the same 1.24× over the target's floor would be ~600 cycles ≈ **0.17–0.19 µs** at
3.2–3.5 GHz. Treat that as the prediction to check, not as a measurement.

### What was tried and did NOT work (with the number that killed it)

1. **Non-temporal stores at small batch** — v3 at B=1 is 0.423 µs vs v1's 0.253 µs, **67%
   worse**. Obvious in hindsight (the output is L1-resident and re-written every call), but
   it is what motivated putting `nt` in the tournament rather than behind a size heuristic.
2. **Software prefetch at B=1** — 0.257 vs 0.252 µs: prefetching a volume that does not
   exist costs 54 useless uops. Harmless but negative; the tournament rejects it.
3. **The fused (register-resident plane) variant on AVX2** — 0.265 vs 0.253 µs at B=1,
   because 26 live vectors spill on 16 architectural ymm (20 stack references in the
   generated code). Kept anyway *because* the target has 32 evex ymm and the fused form is
   the only one whose uop count (1834) drops below the FP floor; the runtime tournament
   will decide it there. **If a later round finds v2/v6/v8 losing at small B on the real
   node too, the fused path can be deleted outright.**
4. **A naive self-tuner** (each candidate timed to completion in sequence, 5 rounds, min)
   **mis-picked NT at B=8** and gave 0.329 µs instead of 0.271 µs — **21% worse than doing
   nothing**. Cause: background load on a shared node drifts between candidates. Fixed by
   round-robin interleaving of the candidates across rounds. *Any implementer doing plan-
   time self-tuning on this cluster needs this.*
5. **`malloc` for the tuner's scratch buffers segfaulted** — glibc `malloc` returns only
   16-byte alignment for small requests, and every vector access here is a 32-byte
   *aligned* access. Use `posix_memalign(…, 64, …)` for anything the kernels touch. (The
   driver's own buffers are 64 B aligned, so the shipped path was never affected.)
6. **The composed 3D Good–Thomas map (216 = 8 × 27), i.e. `DFT_{2³} ⊗ DFT_{3³}` with one
   permutation instead of six — not built, and the reason is worth recording.** It is
   *arithmetically identical*: the 8-point 3D Hadamard part is 1296 adds and the 27-point
   3D part is 8 × 3 × 9 × 16 = 3456 flops, total **4752 — exactly the row–column PFA
   count**, because it is the same tensor product with the factors reordered. So the only
   possible win is data movement, and there it *loses*: the row–column form's permutation
   is already free (the 1↔4 involution absorbed into register naming), whereas the composed
   form's natural lane sets are the 8 CRT parity classes (stride 3 in every axis) or the 27
   residue classes — neither is 2 adjacent complex in memory, so every 32-byte vector needs
   2 loads + 1 insert instead of 1 load: **324 loads → ~972 uops**, against a front-end
   budget that is already the binding constraint. §03 §9.4 flags this decomposition as the
   most promising structural idea for L=6; the counting above says it is a mirage *once the
   line codelet is unrolled and the permutation costs nothing*.
7. **Fusing the x and y passes** (to cut the 3.4 KB scratch round trip further) needs the
   full 6×6 of `(x,y)` for one z-pair live at once = **36 vectors**, over even AVX-512's 32.
   Only y+z (18 vectors) fits, which is exactly what `fused=1` does.
8. **Split re/im layout** was rejected on the count in "Layout" above: same FP instruction
   count per line, plus ~430 deinterleave/reinterleave shuffles per volume that the
   interleaved form does not pay. (Analytic, not measured — the interleaved form was
   written first and the shuffle budget left no motive.)

### Next

1. **Read the monitor's node numbers and check the variant the tournament chose** (build
   with `-DL6_VERBOSE` to have `fft3d_create` print it). The two open questions are both
   answered by that one run: does `fused=1` win at small B once 32 evex ymm are available,
   and is the B=1 time near the predicted 486–600 cycles.
2. **If B=1 lands well above ~600 cycles**, the suspect is the pass boundary: pass 2's
   first set needs the *last* stores of pass 1 (both need all six y). Pass 1 is already
   ordered `zp` outer / `y` inner for this reason (worth a measured 0.6%: 0.2520 vs 0.2535
   µs); the next step is a proper software pipeline that starts pass 2 for `zp = 0` while
   pass 1 runs `zp = 1,2`, which needs the passes interleaved by hand rather than by the
   out-of-order window.
3. **Batched regime is bandwidth-bound, not compute-bound** (0.301 vs 0.251 µs, with 6912
   B/volume of compulsory traffic = 23 GB/s at that rate). The only remaining lever is
   making the *input* side cheaper: 4 KB-aligned volumes are 3456 B apart, so consecutive
   volumes shift their page offset every time; huge pages (`madvise(MADV_HUGEPAGE)` cannot
   be applied to the driver's buffers from inside the plan) or a prefetch distance of two
   volumes are the two things left to try. Expect single-digit percent.
4. **Do not** revisit the arithmetic. 4752 flops / 3888 instructions per volume is the
   Good–Thomas optimum and matches FFTW's codelet exactly; the entire remaining headroom at
   L = 6 is scheduling and memory.
