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

---

## Round panel_r2 (2026-08-21, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round 1 landed (node, panel_r1)

Tied with `L6_unrolled` at B=1 (both **0.219 µs**, my spread 0.7% vs their 4.4%), 2% behind
at B=64 (0.223 vs 0.219), clearly ahead at large batch (B=4096: **0.392 vs 0.514**;
B=32768: **0.631 vs 0.760**; MKL 0.556/0.698). 0.219 µs × 2.3 GHz = **504 cycles against
the 486-cycle FP-port floor**, so the node runs near base clock and B=1 has ≤4% of
theoretical headroom. The arithmetic (4752 flops / 972 ymm FP per volume) stays closed —
this round is all scheduling, addresses, and information capture.

### What changed (three things, two of them borrowed)

1. **4K-aliasing scratch placement — borrowed from `L6_unrolled`** (their round-1 record
   measured **+22% at B=1** for an unlucky malloc: a store→load pair agreeing mod 4096
   replays the load). The scratch now lives in a 4 KiB-oversized arena; at first execute
   (when `in`/`out` are known) it is positioned, in 64 B steps, to maximise the minimum
   cyclic distance mod 4096 of the two cross-buffer store→load base deltas (pass 1 stores
   S / loads `in`; the last pass stores `out` / loads S). Within a pass the offsets differ
   by ≤~440 B inside the store-buffer window, so a base delta ≥1024 clears every pair; the
   two bad zones total <4096 B, so a good offset always exists. B=1 only in effect: at
   B≥32 the volume stride (3456 ≡ −640 mod 4096, gcd 128) walks all 32 residues anyway.
   Addresses only — output bit-identical, repeatability untouched.
2. **The chosen variant is spliced into `fft3d_description()`** — borrowed from
   `L6_unrolled` — so the next leaderboard tells us *which kernel actually won on the
   node*. Round 1's biggest open question (does `fused` win at B=1 once 32 evex ymm are
   available?) was unanswerable because I didn't report it.
3. **Prefetch distance 2 as a third `pf` state** → 12 kernels
   (`variant = 1 + fused + 2·nt + 4·pf`, `pf ∈ {0,1,2}`), same correctness gate and
   round-robin tournament. Rationale: at the node's DRAM-bound rate one volume of lead is
   ~600 ns; two volumes might cover better on a 1 MB-L2 machine. Also added a
   `-DL6_VERBOSE` dump of the full per-variant tournament table (µs/transform), which
   turned out to be the only trustworthy measurement mode on wallaby (see below).

### Operation count

Unchanged: DFT6 = 2·DFT3 + 3·DFT2 = 44 flops / 36 scalar-shaped FP instructions per line,
4752 flops / 972 ymm FP instructions per 6³ volume. Nothing arithmetic was touched.

### What was measured — wallaby, and a warning for every implementer

**wallaby's clock is a per-invocation lottery: ~2.1 GHz or ~4.1 GHz, bimodally, for the
whole run.** Identical binaries at B=1 returned min = 0.246 µs (sd 0.02%) and min = 0.126
µs (sd 20%) in back-to-back invocations — a 1.95× ratio, exactly base/turbo. B=4096 swung
1102–1390 µs across invocations (wallaby is also 2-socket, so NUMA placement of the
driver's buffers adds to it). **Cross-invocation absolute numbers on wallaby mean nothing;
the within-process round-robin tournament ranks are the signal.** Tournament tables
(µs/transform, one process each):

| variant | B=1 | B=4096 | B=32768 |
|---|---|---|---|
| v1 (3-pass) | **0.2472** | 0.6966 | 0.7200 |
| v2 (fused) | 0.2499 | 0.4265 | 0.4289 |
| v4 (fused+nt) | 0.3783 | 0.3715 | 0.3849 |
| v8 (fused+nt+pf1) | 0.3804 | **0.3187** | 0.3376 |
| v12 (fused+nt+pf2) | 0.3796 | 0.3196 | **0.3369** |

Chosen: B=1→v1, B=64→v2 (fused wins there), B=4096→v8, B=32768→v8 (v12 inside noise).
Steady-state (base-clock) driver numbers: **B=1 0.246 µs**, B=4096 ≈0.319, B=32768 ≈0.337
µs/transform (tournament-measured; turbo invocations show 0.126/0.27/—). Accuracy
everywhere: **rel L2 2.2–2.4e-16**, repeatable bit-identical across runs, B ∈
{1,3,8,64,4096,32768} all PASS.

**Spill check for the node (cross-compiled `-march=cascadelake`, grep of the asm):**
`kern_2` (fused) has **86 ymm16–31 references and 1 rsp reference** (the callee-save
restore) — spill-free with 32 evex ymm. So on the node's 4-wide front end, where v1's
~2050 uops/volume (512 cycles) sit *above* the 486-cycle FP floor and v2's ~1834 (459)
sit *below* it, **v2 should take B=1 on the node by ~4%**; wallaby can't see this (6-wide
alloc, front end never binds, v1 ≈ v2 ≈ 0.247 there). The tournament + the new
description reporting will answer it in the monitor's run.

### What was tried and did NOT work (with the number/reason that killed it)

1. **`madvise(MADV_HUGEPAGE)` on `in`/`out` at first execute** (`L6_unrolled`'s Next #3)
   — killed by inspection before coding: wallaby (and per the brief's OS, likely the
   node) has THP in **`madvise` mode with `khugepaged scan_sleep_millisecs = 10000`**,
   and the driver faults its buffers in (fills them with data) *before* the first
   execute. Post-fault `MADV_HUGEPAGE` only queues the range for khugepaged, which at a
   10 s scan cadence collapses nothing within a timing run; the synchronous
   `MADV_COLLAPSE` needs kernel ≥6.1 and this cluster runs 5.15. **Closes that item
   negatively for every entry — don't rebuild it.**
2. **Prefetch distance 2** — no gain on wallaby (v12 0.3369 vs v8 0.3376 at B=32768,
   inside noise; same story at B=4096). Kept anyway: it costs nothing (the tournament
   picks per-machine) and the node's 1 MB L2 + slower DRAM is exactly where it could
   still pay. The node's choice will appear in the description string.
3. **Chasing wallaby cross-invocation minima** — half a session lost to what looked like
   a 2× algorithmic effect (0.126 vs 0.246 µs) and was the turbo lottery. Recorded above
   so nobody else burns time on it.

### Next

1. **Read the `variant=` field off the panel_r2 leaderboard.** If v2 won B=1 on the node,
   the front-end analysis is confirmed and the remaining B=1 gap to 486 cycles is ~zero;
   if v1 won, the fused kernel is stalling somewhere the uop count doesn't show
   (store-forwarding at the pass-1→fused boundary is the suspect) and a software-pipelined
   x-pass/fused-pass interleave is the one idea left (~3% ceiling).
2. **Batched is at the compulsory-traffic floor** (3456 B in + 3456 B NT out per volume;
   0.631 µs = 11 GB/s single-core on the node). Only huge pages could move it and (1
   above) they are unreachable from inside the plan on this kernel. If the monitor can
   set `transparent_hugepage=always` or boot-time hugepages for a control experiment,
   that would quantify the TLB share; expect single-digit percent.
3. **Do not revisit the arithmetic** (still closed at 4752 flops) and do not build a
   512-bit path for the Gold 5218 (1 FMA unit — round-1 analysis stands).

---

## Round panel_r3 (2026-08-21, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r2 landed (node)

Second in all four cells behind `L6_unrolled`: B=1 0.221 vs 0.218, B=64 0.223 vs 0.214,
B=4096 0.393 vs 0.384, B=32768 **0.616 vs 0.572** µs. The description-string reporting
paid off: the node picked **v2 (fused) at B=1** — front-end analysis confirmed, B=1 is
within ~4% of the 486-cycle FP floor at base clock — and **v6 (fused+pf1, normal stores)
at B=64/4096/32768**. NT stores lost on the node at every batch size, for both L=6
entries independently (VERDICT §"the same tuner reporting"), despite NT winning wallaby's
DRAM regime by a wide margin. The B=32768 loss is a tournament artifact, not a kernel
gap: my grid *contains* the node's winning shape (v5 = 3pass+pf — `L6_unrolled` won that
cell with exactly `3pass_pf`), but my race truncated at 4096 volumes = 27 MiB, which is
L3-resident on wallaby and L3-marginal on the node, so the race never saw the DRAM
regime it was picking for.

### What changed (three things, one borrowed, one new)

1. **Race truncation cap 4096 → 16384 volumes (113 MiB)** — borrowed from
   `L6_unrolled`'s round-2 record, which raised its cap for exactly this failure and
   documented the rationale ("a tuning arena that fits the dev machine's L3 mis-picks
   the store policy for a streaming run"; the monitor's VERDICT lists my 27 MiB arena
   as one of the round's two reusable tuner-protocol failures). 113 MiB is unambiguously
   DRAM on wallaby (60 MiB L3) and the node (22 MiB). Setup grows to ~1.1 s at B=32768,
   still unscored.
2. **New pf=3 column: `prefetcht1` at distance 1** (vs pf=1/2 = `prefetcht0` at
   distance 1/2). Theory: T1 fills L2 but not L1, so streaming the next volume does not
   displace the L1-resident scratch; the x-pass then eats an L1-miss/L2-hit (~14 cy)
   per load, which the OoO window hides. Grid is now {fused}×{nt}×{pf∈0,1,2,3} = 16
   kernels, variant = 1 + fused + 2·nt + 4·pf.
3. **Tournament selection fix**: the reference time now tracks the true minimum even
   when the incumbent survives the 1% takeover margin (previously a chain of <1% steps
   could drift the pick off the fastest candidate).

Arithmetic untouched (still 4752 flops / 972 ymm FP per volume; closed). Asm re-check at
`-march=cascadelake`: all 16 kernels spill-free (3pass kernels 0 rsp refs, fused 1 =
callee-save restore); `prefetcht1` confirmed emitted in v13–v16.

### What was measured (wallaby; race-table numbers, which are the trustworthy statistic here — see r2's clock-lottery warning)

B=32768 tournament at the new 113 MiB race size (µs/transform):

| shape | pf=0 | pf=T0·1 | pf=T0·2 | pf=T1·1 |
|---|---|---|---|---|
| 3pass | 0.5340 | 0.5252 | 0.5213 | 0.5247 |
| fused | 0.4848 | 0.4885 | 0.4826 | 0.4909 |
| 3pass+nt | 0.2771 | 0.2310 | 0.2312 | **0.2142** |
| fused+nt | 0.2805 | 0.2742 | 0.2728 | 0.2674 |

Chosen v15 = 3pass+nt+pfT1. Driver: **B=32768 0.231 µs/vol vs 0.338 last round = 1.46×
on wallaby** (MKL same case: 0.708). The T1 hint beats T0 by 7% in the NT rows — first
positive datum for pf=3; on wallaby only, so far. B=4096 picked v16 (fused+nt+pfT1),
race 0.3171 vs old v8's 0.3187 — T1 marginally ahead there too. B=1: 0.126 µs (turbo
invocation) / 0.247 (base), unchanged, picks v5≈v1 within noise; B=64 picks v6, as
before. Accuracy everywhere rel L2 2.2–2.4e-16, bit-identical across re-runs, B ∈
{1,8,64,4096,32768} all PASS.

**Node prediction, explicitly falsifiable via the description string:** the node raced
at 113 MiB in panel_r2 (`L6_unrolled`) still rejected NT, so I expect the node to pick
**v5 (3pass+pf1, normal stores) or v13 (3pass+pfT1) at B=32768** and land near
`L6_unrolled`'s 0.572; if it picks an NT variant (v7/v11/v15) the node-NT story was a
race-arena artifact after all and the cell should drop below 0.55. B=4096 should stay
fused+pf (race size unchanged there, 27 MiB = faithful to the real working set). B=1
unchanged.

### What was tried and did NOT work (with the number)

1. Nothing new failed outright this round — the changes were narrow by design. The
   standing negative results that shaped it: NT-on-node (both L=6 entries, panel_r2,
   every batch size), `prefetchnta` (L6_unrolled r2: 0.53–0.65 vs 0.19 µs/vol,
   catastrophic — why the new column is T1, not NTA), `MADV_HUGEPAGE` (my r2 item 1,
   kernel 5.15 + pre-faulted buffers), and two-volumes-in-flight software pipelining
   across the pass boundary (L6_unrolled r1 item 3: inside noise — which is why I did
   not attempt the cross-volume x-pass/fused interleave this round either).
2. Not attempted, recorded as consciously skipped: an ascending-`g`-order x-pass twin
   (L6_unrolled's loop order). My r1 measured zp-outer worth +0.6% on Haswell; the
   residual same-shape gap to them at B=64/4096 on the node is ~2% with overlapping run
   spreads, wallaby cannot resolve node front-end effects, and doubling the grid to 32
   kernels for an unmeasurable-here fraction of a percent is bad economics. If the r3
   leaderboard still shows a consistent 2% at B=64 with identical variant strings, diff
   the two entries' generated asm on the node rather than guessing (their r2 Next #3
   says the same).

### Next

1. **Read the r3 leaderboard `variant=` strings against the prediction above.** If v5/v13
   wins B=32768 at ~0.57, the tournament story is closed and L=6 large-batch is at the
   compulsory-traffic floor with normal stores (10368 B/vol incl. RFO ≈ 18 GB/s at
   0.572); the only thing left there would be an answer to why NT loses on the node
   (LFB contention under `powersave`?) — a monitor-side `perf stat` question, not a code
   change.
2. **B=1 needs the node clock measured before any further work** (VERDICT asks for
   `perf stat -e cycles,ref-cycles` during an L=6 B=1 run). At base clock we are ≤4%
   from the FP-port floor and done; if the node actually turbos, there is real headroom
   and the pass-boundary schedule becomes worth attacking despite item 1's dead end.
3. If pf=T1 wins on the node too, propagate the idea to the other geometries' records —
   L17_winograd found pf=T0 *loses* on the node at L=17; T1 was never tried there.

---

## Round panel_r4 (2026-08-21, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r3 landed (node)

B=1 0.220 (first, dead heat with L6_unrolled at 0.220), B=64 0.223 vs their 0.214,
B=4096 0.398 vs 0.392, **B=32768 0.637 vs their 0.563 — a 13% loss and my own 3.4%
regression** (r2: 0.616). Node picks were stable across all three runs per cell
(v2/v6/v6/v5), so the regression was not tuner instability: my 3pass+pfT0 (v5) at 0.637
sits 11% behind the same nominal shape in their file (their r2 3pass_pf: 0.572), and
their round's new `prefetchw` variant (fused_pfw, 0.563) won the cell. The monitor's
VERDICT declared L=6 "finished" at the compulsory-traffic floor — but not for me: my
B=32768 cell was 13% off the demonstrated floor, which is exactly the round's job.

### What changed (mostly adoption; one new variant class)

1. **Write-intent prefetch of the next volume's output (`prefetchw`) — adopted from
   L6_unrolled's panel_r3 round, with the node's own 0.563 µs as the motivation.**
   The x-pass hook can now also touch the next volume's 54 output lines with
   `__builtin_prefetch(p,1,3)` (emits `prefetchw` on PRFCHW machines, which both
   wallaby and the node are). With NT stores rejected by the node tuner three rounds
   running, every output line pays a write-allocate RFO at store time; W issues that
   RFO one volume early, off the critical path.
2. **New variant class: W-only (`3pass_w`, `fused_w`) — no input prefetch.** My own
   inference from L6_unrolled's r1 record (input prefetch alone is neutral-to-negative;
   the L2 streamer already covers a perfectly sequential input): if the input prefetch
   uops are dead weight, W alone saves 54 uops/volume. Wallaby says the combination
   still wins in DRAM (see table) but W-only *dominates at B=64* — see measurements.
3. **The two-scratch 3pass (x→t1, y: t1→t2, z: t2→out) — adopted from L6_unrolled** as
   an additional shape beside my original in-place-y 3pass ("add candidates; do not
   replace structures", the r3 VERDICT's process lesson). Rationale: their two-scratch
   3pass ran 0.572 on the node where my in-place 3pass ran 0.637; wallaby (below) says
   the two are within noise of each other, so the node gap was probably not this — but
   it costs one extra scratch volume and nothing else to carry both.
4. **Candidate table restructured to named kernels** (à la L6_unrolled): the
   description string now reports `variant=fused_pfw` instead of `variant=13`, and the
   grid is hand-picked rather than a full product: 21 kernels =
   {3pass, 3pass_ip, fused} × {plain, pfT0} + {3pass, fused} × {pfT1, W, pfT0+W, pfT1+W}
   + {3pass, fused} × NT × {plain, pfT0, pfT1}. **Dropped: all distance-2 prefetch
   variants** — never separated from distance 1 in any measurement, mine or theirs,
   either machine, four rounds of data.
5. **Takeover margin 1% → 1.5%** (their value; the r3 VERDICT measured unstable tuners
   costing 3.9–6.7% elsewhere on the board). Round-robin, per-candidate minimum, and
   my r3 tracked-true-minimum fix all retained; race cap stays 16384 volumes = 113 MiB.
6. 4K-aliasing placement extended to the two-scratch shape (deltas t1−in, out−t1,
   out−t2; t2−t1 is fixed at 3456 ≡ 640-clear mod 4096), and the tournament now runs
   with realistically-placed scratch (place_scratch against the race arena, reset
   after).

Arithmetic untouched (4752 flops / 972 ymm FP per volume; closed since round 1). The
DFT6V codelet, pass structures, zp-outer x-pass order, and z-pass in-register
transposes are byte-identical to r3.

### What was measured (wallaby; race tables are the trustworthy statistic — same-process,
same clock; the r2 clock-lottery warning stands for driver minima)

B=32768 race (113 MiB, DRAM on both machines), µs/vol, base-clock invocation:

| shape | plain | pfT0 | pfT1 | W only | pfT0+W | pfT1+W | nt | nt+pfT0 | nt+pfT1 |
|---|---|---|---|---|---|---|---|---|---|
| 3pass | 0.5299 | 0.5085 | 0.4993 | 0.3425 | 0.3163 | 0.3114 | 0.3101 | 0.1897 | 0.1945 |
| 3pass_ip | 0.5232 | 0.5044 | — | — | 0.3153 | — | — | — | — |
| fused | 0.4612 | 0.4478 | 0.4470 | 0.3404 | **0.3114** | **0.3074** | 0.2253 | **0.1829** | 0.1898 |

Three headlines: (a) **my fused_pfw now measures 0.3114 — exactly L6_unrolled's r3
number for the same shape (0.3114)**, so the kernels are at parity and the r3 gap is
closed on this machine; (b) W is worth 1.6× in the normal-store rows (0.5085 → 0.3163),
and W-only captures most of it (0.3425) but the input prefetch still pays its way in
DRAM; (c) my in-place and the adopted two-scratch 3pass are within noise of each other
everywhere on wallaby (0.5232 vs 0.5299 plain), so the r3 node gap at B=32768 was
evidently W + shape choice, not the scratch discipline. Wallaby still picks NT in DRAM
(fused_nt_pf 0.1829; driver 0.258 µs/vol at B=32768 vs MKL 0.516).

**B=64 (0.42 MiB, L2-resident): W variants dominate by 1.7×** — fused_pf 0.2514 vs
3pass_pfw/ip_pfw/fused_pfw 0.1436/0.1431/0.1435, fused_w **0.1453** chosen (driver
0.146 µs/vol, vs my r3-era ~0.25 at base clock). Same process, same clock, so this is
real: even when the output is L2-resident, the 54 demand-RFOs per volume (L2-hit
latency, ~12 fill buffers) stall the store buffer, and prefetchw converts them to L1
hits a volume ahead. This was NOT visible in any earlier round because nobody had W
variants racing at cache-resident sizes — L6_unrolled's r3 record only measured W at
27/113 MiB. **If the same mechanism holds on the node's 1 MB L2, B=64 (and possibly
B=4096) should improve; their entry has the same variants, so expect both entries to
move together.**

B=4096 (27 MiB, L3-resident on wallaby): W loses in the normal rows (fused_pfw 0.4020
vs fused_pf 0.3366), consistent with L6_unrolled's r3 finding at that size on this
machine; wallaby picks fused_nt_pf (0.2856). On the node 27 MiB streams, so the node
race may resolve differently. B=1: picks 3pass (0.2462; 3pass_ip 0.2456 and fused
0.2494 inside the 1.5% margin), driver 0.252 base / 0.131 turbo — unchanged from r3,
as intended.

Correctness: rel L2 2.24–2.42e-16 at B ∈ {1, 3, 8, 64, 512, 4096, 32768}, all PASS,
bit-identical across re-runs everywhere. Cross-compile at `-march=cascadelake`:
`prefetchw` emitted (51 sites), all 21 vector kernels vector-spill-free (3 rsp refs =
callee-save frame; the three `3pass*w` kernels spill one integer loop bound, harmless).
Setup ≤1.2 s at B=32768 (21 candidates × 7 rounds), unscored.

### Node prediction (falsifiable via the description string)

* B=32768: node rejects NT again → picks `fused_pfw`/`fused_pft1w`/`3pass_pfw` and
  lands ≈0.56, closing the 13% gap to L6_unrolled (who should stay ≈0.563; we now
  carry the same winning shape). If the node picks an NT variant, the three-round NT
  story finally breaks.
* B=64: if the wallaby RFO mechanism transfers, the node picks a W variant and the
  cell drops below 0.214 for both L=6 entries; if the node's smaller L1 (32 KB vs
  48 KB) makes the prefetched output lines evict the working set, it stays fused_pf
  ≈0.223.
* B=4096: borderline (27 MiB > 22 MiB L3) — fused_pfw or fused_pf, 0.38–0.40.
* B=1: any of 3pass/3pass_ip/fused at ≈0.220 (all inside 1.5% on the node too,
  most likely).

### What was tried and did NOT work (with the number)

1. **Nothing failed outright this round** — by design it was an adoption round. The
   negative results that shaped the grid are all cited above: distance-2 prefetch
   (four rounds, never ≠ distance 1), NT-on-node (three rounds, both entries),
   `prefetchnta` (L6_unrolled r1: catastrophic), `MADV_HUGEPAGE` (my r2: kernel 5.15 +
   pre-faulted buffers).
2. **W at B=1** measured and correctly rejected by the race (0.2537–0.2653 vs 0.2462):
   prefetching a volume that does not exist costs 54–108 useless uops. The safest-first
   ordering keeps the plain kernels in front, so B=1 cannot be noise-stolen.
3. Consciously skipped again: B=1 structural work (pass-boundary software pipelining)
   — still blocked on the monitor's `perf stat -e cycles,ref-cycles` clock question
   (asked by four entries, two rounds running); at base clock B=1 has ≤4% headroom and
   two documented dead ends (L6_unrolled r1 item 3, my r1 Next 2) say the OoO window
   already covers the boundary.

### Next

1. **Read the r4 `variant=` strings against the predictions above.** The B=64 cell is
   the most informative: it is the first node test of write-intent prefetch at a
   cache-resident size, and the answer transfers to L=8's batched cells (L8_fusedaxes'
   B=64 cell is L2-resident too) and possibly L=36 B=4.
2. If W wins broadly on the node, the one knob left is **W distance and interleaving**
   (currently: 3 lines per x-pass group, 1 volume ahead; an interleave into the y/z
   passes would spread the RFO issue more evenly — worth trying only if the r4 numbers
   show the cell still above the 12.3 GB/s floor L6_unrolled demonstrated).
3. If B=32768 lands ≈0.563 for both entries, L=6 really is finished (both entries at
   the compulsory-traffic floor with the same kernel set); the panel should follow the
   r3 VERDICT and move one L=6 implementer to L=36.

---

## Round panel_r5 (2026-08-21, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r4 landed (node), and the fact that reopens B=1

B=1 0.219 (dead heat with L6_unrolled's 0.219), B=64 0.222 vs their 0.214 (the same
~4% same-shape gap, third round running), B=4096 **0.387 — took the cell**, B=32768
0.570 vs 0.566 (tie within spread). Node picks, stable ×3: `fused` / `fused_pf` /
`fused_pf` / `fused_pfw`. Two r4 predictions resolved negative and are now closed:
**W at B=64 did not transfer to the node** (wallaby's 1.7× RFO win at L2-resident
sizes; the node stayed `fused_pf`, so the demand-RFO mechanism is SPR-specific or the
node's smaller L1 eats the benefit), and NT lost on the node a fourth round. I was
**not promoted** in r4 — judged a near-duplicate of L6_unrolled (same codelet, same
variant taxonomy, a round spent adopting their prefetch).

The round-changing datum is L6_unrolled's clock probe: **the node runs 3.89 GHz, not
2.30**. So B=1's 0.219 µs is **852 cycles against the 486-cycle FP-port floor — 43%
of B=1 runtime is not FP-bound and unexplained**, and the VERDICT reopened the
geometry with exactly that assignment. Meanwhile wallaby (4.10 GHz, a 5% clock
difference) runs the same kernels at ~470-520 cycles: the 1.9× machine gap at L=6 is
per-cycle throughput, not clock.

### The hypothesis this round is built on (mine; stated so the node can falsify it)

What differs 1.7× per-cycle between CLX and SPR for an L1-resident straight-line
kernel? Not FP width, not ports (both 2×256-bit FMA effectively), not DSB fit (the
rolled kernels are ~2.5 KB/volume, comfortably DSB-resident on both). The largest
relevant deltas are the **out-of-order window: ROB 224 vs 512 uops, RS 97 vs 205**.
My fused kernel's plane body is ~195 uops with a serial spine (18 loads → 3 y-DFT6V
→ transpose → z-DFT6V → transpose → store, depth ≈45 cycles): on CLX barely one
plane fits the window, so plane x+1's independent work cannot overlap plane x's
z-tail, and the kernel degrades toward latency-bound; SPR holds ~2.6 planes and sits
at the FP floor. If that is right, the fix is not fewer uops but **more independent
work per window slot and earlier program-order entry of the next plane**.

### What changed (all candidate additions; arithmetic untouched, still 4752 flops / 972 ymm FP per volume)

1. **`_u2` kernels — every strided pass unrolled ×2** with two independent DFT6V
   chains per iteration, program order loads-loads-codelets-stores so the scheduler
   sees ~40 independent FP ops per iteration. Shapes: `3pass_u2`, `fused_u2` (u2
   x-pass + register-fused y/z), each × {plain, pfT0, pfT0+W}.
2. **`fused_sp` — software-pipelined fused stage** (the round's real idea): plane
   registers double-buffered (P for even x, Q for odd), and the NEXT plane's 18
   loads + 3 y-DFT6Vs interleaved by thirds into the CURRENT plane's z-chunks, so
   plane x+1 enters the ROB ~130 uops earlier than the sequential form allows.
   Register budget: CUR drains 18→12→6 as NXT fills 6→12→18; peak ~30 live ymm.
   Cross-compiled at `-march=cascadelake`: **0 stack references, 653 ymm16–31 uses —
   spill-free on the 32 evex registers.** × {plain, pfT0, pfT0+W}.
3. **`fused_sp2` — the same pipeline with the plane-pair loop kept rolled.**
   `fused_sp` fully unrolls to ~8.9 KB/volume ≈ 1700 instructions, which overflows
   the node's ~1.5K-uop DSB and would put it on 16 B/cycle legacy decode — a
   confound that could sink the window experiment for a front-end reason. sp2 rolls
   planes 0–3 into a 2-iteration loop (body ~450 insns, above gcc's complete-peel
   limit, so it stays rolled) → ~5.5–6.4 KB total, DSB-resident. **The sp-vs-sp2
   race on the node is a direct measurement of whether straight-line code footprint
   matters there — new information for every geometry, since all the big entries
   ship multi-KB unrolled kernels.** × {plain, pfT0, pfT0+W}.
4. **Grid pruned from 21 to 22 net** (10 kept + 12 new): dropped every variant the
   node never picked in three rounds of stable pick-reporting — `3pass_ip_pf/pfw`,
   both `_pft1` (non-W), both `_w`-only, `3pass_pft1w`, and 4 of 6 NT kernels
   (kept `3pass_nt`, `3pass_nt_pf` as the NT representatives; NT is 0-for-4 rounds
   on the node). Kept `fused_pft1w` as the only T1 representative (won my wallaby
   B=32768 race twice). Safest-first order and the 1.5% takeover margin unchanged.

### What was measured (wallaby; race tables = the trustworthy statistic, turbo invocations)

B=1 (all 22 candidates within 2%, as expected on a 512-ROB machine — the u2/sp
mechanisms target CLX, wallaby cannot confirm or deny them at B=1):
3pass 0.1262, 3pass_ip 0.1258, fused 0.1277, fused_sp 0.1258, fused_sp2 0.1270,
fused_u2 0.1279; chosen `3pass` (safest-first, everything inside margin). Driver
0.129 µs, sd 1.4%.

**B=64: `fused_sp` won outright — 0.1283 vs fused 0.1310 (−2.1%), fused_sp_pf
0.1275 best in table** — the first same-clock, same-process evidence that the plane
pipeline is a real steady-state gain even on the wide machine. Driver 8.174 µs/call
= 0.1277 µs/vol. This is the exact cell where the node has me 4% behind three
rounds running.

B=4096: 3pass_nt_pf chosen (wallaby DRAM regime; irrelevant to the node, which
rejects NT). B=32768: 3pass_nt_pf 0.2067 chosen; **in the normal-store rows the
node actually picks from: fused_pft1w 0.3109, fused_sp_pfw 0.3119, fused_sp2_pfw
0.3124, fused_pfw 0.3135** — sp/sp2 at the front but inside noise.

Correctness: rel L2 2.24–2.43e-16 at B ∈ {1, 3, 8, 64, 4096, 32768}, all PASS,
bit-identical across re-runs. All 22 candidates pass the plan-time gate. Setup
≤1.45 s at B=32768 (unscored).

### Node predictions (falsifiable via the `variant=` strings)

* **B=1: if the window hypothesis is right, `fused_sp` or `fused_sp2` takes the cell
  and it moves toward 700–750 cycles ≈ 0.18–0.19 µs.** If sp2 wins but sp loses,
  the DSB-footprint confound was real (record that for every geometry). If neither
  is picked and B=1 stays 0.219, the window theory is wrong at this size and the
  remaining suspects are load/store-buffer occupancy or something only `perf stat`
  can see — I then second the monitor's offer to counter-profile B=1.
* B=64: `fused_sp[_pf]`, and the cell finally closes toward 0.214 or below.
* B=4096: `fused_pf` or `fused_sp_pf`, ≈0.387.
* B=32768: `fused_pfw`-family incl. possibly `fused_sp_pfw`, ≈0.563–0.570 (the
  compulsory-traffic floor; no mechanism in this round should move it).

### What was tried and did NOT work (with the number)

1. **`_u2` on wallaby is neutral-to-negative in every cell** (B=64: 3pass_u2 0.1444
   vs 3pass 0.1550 — helps the 3pass shape, but fused_u2 0.1315 vs fused 0.1310 —
   nothing; B=32768 fused_u2_pfw 0.3144 vs fused_pfw 0.3135). Expected on a 512-ROB
   machine; kept because the mechanism targets CLX and costs only race time. If the
   node also rejects every `_u2`, drop the class next round.
2. **W-only and pfT1-without-W variants**: not re-raced — killed by three rounds of
   node pick-reporting (never selected), per the "don't rediscover documented dead
   ends" rule. Same for NT beyond two representatives (0-for-4 rounds on the node).
3. Consciously skipped: L6_unrolled's split-store `_s` shapes — their own r4 node
   data (12 invocations, never picked at any batch size, B=1 stayed `fused`) says
   the mechanism is SPR-specific (2 store ports vs CLX's 1); adding my own copies
   would rediscover their negative. Also skipped: any 512-bit path (round-1 analysis
   stands — 1 FMA unit on the Gold 5218) and MADV_HUGEPAGE (my r2, kernel 5.15).

### Borrowed / lent

Borrowed this round: the node-clock fact (3.89 GHz) and the DSB-size framing come
from L6_unrolled's r4 clock probe and the r4 VERDICT §5a. The `fused_sp` plane
pipeline, the `_u2` interleave, and the sp-vs-sp2 DSB experiment are mine. Lendable:
the double-buffered-plane pipeline applies to any register-resident fused stage
(L8_fusedaxes' single fused pass is the obvious candidate — their plane body is
larger than mine, so their window pressure on CLX is worse), and the sp-vs-sp2
comparison will say whether multi-KB unrolled kernels pay a decode tax on the node.

### Next

1. **Read the r5 `variant=` strings against the predictions above.** The B=1 cell is
   the experiment; the B=64 cell is the payoff. If `fused_sp*` wins either, propagate
   the pattern to L8_fusedaxes' record.
2. If sp wins B=64 but not B=1, the boundary overhead at B=1 is call/loop-entry
   effects, not the window — measure a B=1-specialized entry point (no batch loop,
   no indirect-call double dispatch) before concluding anything.
3. If the node rejects everything again and B=1 stays 852 cycles, stop guessing:
   ask the monitor for `perf stat -e uops_issued.any,cycle_activity.stalls_total,
   resource_stalls.rob,resource_stalls.rs` on a forced `fused` vs `fused_sp` B=1
   run. One measurement ends a three-round argument.
4. The B=32768 cell is at the compulsory-traffic floor for both entries; do not
   spend another round there.

---

## Round panel_r6 (2026-08-21, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r5 landed (node), and what this round is for

On the run distributions I now **own B=1 outright (0.2187 in all three runs, 0.0%
spread**, vs L6_unrolled's 0.2271/0.2271/0.2203) and B=4096 (0.3905–0.3944 vs their
0.3966–0.4068); they own B=64 (their 0.2146–0.2217 vs my 0.2173–0.2262) and B=32768
(0.5657–0.5723 vs my 0.5740–0.5768). Promoted for the first time (r5 VERDICT §7
reversed the r4 "near-duplicate" judgement). My r5 experiment resolved half-negative:
`fused_sp`/`fused_sp2` were **rejected at B=1, the cell they were built for** — the
OoO-window hypothesis is dead there — but `fused_sp2_pf` was **selected at B=64
(3/3), beating my own `fused_pf` by ≥1.5%**, and sp2-over-sp also settles the DSB
question (the rolled form won; code footprint matters on CLX). The `_u2` class: zero
picks anywhere, as my r5 record promised, so it dies this round.

The VERDICT's L=6 instruction is explicit: **settle `clk256`, then profile; stop
shipping speculative kernels.** Five r5 clock probes split 3.89/2.89/3.27 GHz on the
256-bit question, and the difference decides whether B=1's 0.219 µs is 852 cycles
(1.75× the 486-cycle FP floor, 43% unexplained) or 633 (1.30×, 23%). Everything
cycle-derived in five rounds of records hangs on it.

### What changed (one evidence-based kernel mechanism, one measurement, two pieces of housekeeping)

1. **`_xa` kernels — x-pass in strictly ascending address order — ADOPTED FROM
   L6_unrolled.** Reading their file for the B≥64 same-shape gap (their `fused_pf`
   0.215 vs mine ≈0.23 at B=64 three rounds running; their `fused_pfw` 0.566 vs my
   0.574 at B=32768) turned up exactly ONE structural difference in these nominally
   identical kernels: their `L6_PASS_X` walks the 18 groups at offsets `4g`,
   g = 0..17 — loads and stores in strictly ascending 32 B steps — where mine is
   zp-outer/y-inner (a +0.6% *Haswell B=1* measurement from my round 1 that I never
   re-examined). Ascending is friendlier to CLX's L2 streamer in the streaming
   regime; wallaby (SPR) measures parity between the orders, which is consistent
   with r4's observed cross-file parity at fused_pfw (0.3114 == 0.3114) while the
   node kept a gap. New twins: `fused_xa`, `fused_pf_xa`, `fused_pfw_xa`,
   `fused_sp2_pf_xa`. Ordering bet: the xa twin sits BEFORE its zp twin for the
   prefetch shapes (streaming regime, where their file's numbers say ascending
   wins), and AFTER for plain `fused` (B=1 is the cell I reproducibly hold with
   zp-outer — their ascending `fused` runs 0.227 there).
2. **Four back-to-back clock probes in one process — the r5 VERDICT's #1 ask —
   reported via the description string in every cell:**
   * `clkS256` = 1 serially dependent ymm FMA chain (4 cy/iter): L6_unrolled's
     sparse design, read 3.89 on the node.
   * `clkD256` = 12 independent ymm chains, throughput-bound 2/cy (6 cy/iter):
     L17_winograd's saturating design, read 2.89.
   * `clkS512` = serial zmm chain (settled: 2.89), for cross-checking.
   * **`kclk` — my addition, and the number the panel actually needs: dwell ~2 ms
     in the CHOSEN kernel, then immediately time a ~140 µs sparse ymm chain,
     alternate 9×, median.** CLX licence transitions persist >600 µs, so the short
     chain reads the licence the real kernel established: this is the clock the
     scored kernel runs at, measured directly rather than inferred from a synthetic
     chain's density. Probe order sparse-first with ~4–8 ms scalar-spin gaps so a
     heavier licence cannot leak backward (winograd's r5 rationale).
   Wallaby validation, both clock-lottery states: pinned session
   `clkS256/D256/S512/kclk = 2.10/1.92/2.10/2.10`; turbo session
   `4.10/3.74/4.10/4.10`. The machinery discriminates: dense-FMA pulls SPR below
   max turbo (3.74) while the real kernel does not (kclk = 4.10, FMA density ~50%)
   — precisely the distinction the node question turns on. If the node reports
   kclk = 2.89 the B=1 headroom story shrinks to 147 cycles and L=6 is nearly
   closed; if 3.89, the 366-cycle hunt stays open and rename/pass-boundary are the
   remaining suspects.
3. **`L6_FORCE=<name>` env switch — adopted from L6_unrolled's r5 round**: forced
   pick by name, skips the race (fast setup), still passes the correctness gate,
   reports `variant=<name>!`. For the monitor's `perf stat` A/Bs, which the r5
   VERDICT wants at B=1 (`fused` vs alternates). Verified on wallaby: forced and
   raced outputs bit-identical, description carries the bang.
4. **Pruned on node evidence**: all six `_u2` kernels (0 picks in r5; promised in
   my r5 record), `fused_sp`, `fused_sp_pf`, `fused_sp_pfw` (node chose sp2 over sp
   at ≥1.5%, 3/3 — the fully-unrolled ~7 KB pipeline loses to the DSB-resident
   rolled form). Grid: 22 → 17 candidates, setup shrinks accordingly.

### Operation count

Unchanged and closed since round 1: DFT6 = 2·DFT3 + 3·DFT2 = 44 flops / 36
scalar-shaped FP instructions per line; 4752 flops / 972 ymm FP instructions per 6³
volume. `_xa` is a loop-order change over independent lines — identical arithmetic,
bit-identical output (verified by the plan-time gate and tryout).

### What was measured (wallaby; same-process race tables are the trustworthy statistic)

Base-clock (2.10 GHz) session, B=64 race: fused_sp2_pf_xa 0.2508, fused_pf_xa
0.2531, fused_pf 0.2539, fused_sp2_pf 0.2518, fused 0.2566 (chosen — everything
inside the anti-drift margin chain) — xa ≈ zp within noise, as expected on SPR.
Turbo session, B=32768 race: fused_pfw_xa 0.3103 ≈ fused_pfw 0.3104 (parity, exactly
like r4's cross-file parity), 3pass_nt_pf 0.2076 chosen (wallaby DRAM is still NT
country; irrelevant to the node, 0-for-5 there). Driver numbers (clock lottery
caveat): B=1 0.129 µs (turbo), B=3 0.127 µs/vol, B=64 0.270 µs/vol (base-clock
invocation), B=4096 0.188 µs/vol (turbo), B=32768 0.256 µs/vol (turbo).
Correctness: rel L2 2.24–2.43e-16 at B ∈ {1, 3, 64, 4096, 32768}, all PASS,
bit-identical across re-runs. Cross-compile at `-march=cascadelake`: clean, all 17
kernels vector-spill-free (1 rsp ref = callee-save), prefetchw/prefetcht0 emitted.
Setup ≤1.2 s at B=32768 including ~0.1 s of probes (unscored).

### Node predictions (falsifiable via the description strings)

* **The probes are the round.** Prediction, betting on winograd's density theory
  plus Intel's turbo table: `clkS256=3.89 clkD256=2.89 clkS512=2.89`, and
  **kclk = 2.89** (the fused kernel's ~50% FMA density is enough to hold the AVX2
  heavy licence). If kclk = 2.89: B=1 = 633 cycles = 1.30× floor, the "366 missing
  cycles" halve by measurement alone, and the r4 reopening deflates. If kclk = 3.89:
  the kernel runs in the non-AVX licence band and the 852-cycle accounting stands —
  then the monitor's perf-stat rename/window numbers are the only path left.
  Either way every L=6 (and most L=8) cycle number in the corpus gets its
  denominator fixed.
* B=1: `fused` again at 0.219 (xa sits behind it and SPR-parity says it won't leap
  the 1.5% margin; their ascending `fused` reads 0.227 on the node, so zp-outer
  should hold this cell).
* B=64: `fused_sp2_pf_xa` or `fused_pf_xa`; if the x-order hypothesis is right the
  cell drops to ≈0.215 and finally closes the three-round gap. If it stays
  `fused_sp2_pf` at 0.226, x-order was not the difference and the remaining suspect
  is codegen/code-placement (diff the .s on the node next).
* B=4096: `fused_pf_xa` or `fused_pf`, ≈0.387–0.392.
* B=32768: `fused_pfw_xa` at ≈0.566 (matching L6_unrolled, whose kernel it now
  mirrors) or `fused_pfw` at 0.574 if x-order was irrelevant.

### What was tried and did NOT work (with the number)

1. Nothing failed outright — by design a measurement round plus one adopted
   mechanism. The r5 negatives that shaped it: `_u2` 0 picks in 4 cells (class
   deleted), `fused_sp` unrolled pipeline lost to rolled sp2 (deleted), NT 0-for-5
   rounds on the node (two representatives kept), W-at-B=64 non-transfer (r4),
   MADV_HUGEPAGE (r2, kernel 5.15).
2. **Not attempted, and recorded as consciously rejected: any new B=1 mechanism.**
   Both B=1 hypotheses (uop count via L6_unrolled's zmm; OoO window via my sp) are
   now node-falsified, and the honest next step is the clock + perf measurement,
   not a third guess. The kclk probe is this round's B=1 work.
3. Consciously skipped: xa twins for the 3pass shapes (node picks fused-family in
   all cells, 5 rounds) and for `fused_sp2_pfw` (B=32768 pick is plain fused_pfw;
   grid discipline).

### Borrowed / lent

Borrowed: the ascending x-pass order (L6_unrolled — named mechanism, their file's
one structural difference from mine at the shapes that beat me); the `L6_FORCE`
switch (L6_unrolled r5); the sparse-chain probe design (L6_unrolled r4) and the
saturating-chain design plus sparse-first ordering (L17_winograd r5). Mine and
lendable: **the kclk kernel-context licence probe** — any entry can paste it (it
needs only its own chosen kernel and ~30 lines) and it answers "what clock does MY
kernel run at" per geometry, which no synthetic chain can; L=17's and L=36's cycle
models want exactly this number.

### Next

1. **Read kclk off the r6 leaderboard** and re-derive B=1 cycles. If 2.89: L=6 B=1
   is ≈1.30× floor, declare the remaining ~147 cycles (call overhead + pass-boundary
   fill + the z-pass p5 tail) not worth a round and propose following the r4
   VERDICT's redeployment suggestion. If 3.89: request the monitor's
   `perf stat -e uops_issued.any,cycles,resource_stalls.rs` on `L6_FORCE=fused` at
   B=1 — the switch is now in place.
2. Read the B=64/B=32768 picks: xa selected → propagate the ascending-order lesson
   to the strategy records of L=8/L=36 (any strided pass raced on CLX); xa rejected
   → the same-shape gap is code placement, and the next step is diffing the two
   entries' node-compiled .s for the fused_pf kernel, not more variants.
3. If the node picks `fused_pf_xa` at B=4096 AND `fused_pfw_xa` at B=32768 and both
   cells match L6_unrolled's, the two L=6 entries are converged at every batch size
   and the geometry is done pending the B=1 clock verdict.

---

## Round panel_r7 (2026-08-22, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where things stand: panel_r6 was never timed

The r6 timing phase was abandoned (stale-runner incident; see
`results/panel_r6_abandoned_no_timing/WHY.md`), so everything my r6 record predicted —
the `_xa` ascending x-pass twins, and above all the **kclk kernel-context licence
probe** — is still unmeasured and will finally be scored THIS round together with the
r7 changes. The r6 predictions carry over verbatim. The newest node data remains
panel_r5: B=1 0.219 (mine, 0.0% spread ×3), B=64 0.217 vs L6_unrolled's 0.215,
B=4096 0.391 (mine), B=32768 0.574 vs their 0.566.

### The round's input: the brief's AVX-512 correction

The r7 brief (and LITERATURE §3.2's correction block) settles from Intel's own
Specification Update 338848-028US that on the Gold 5218 the **AVX2 and AVX-512 licence
clocks are identical — 2.9 GHz at 1–8 active cores** — and says explicitly that my
round-1 decision not to write a 512-bit path "was reasoned from the opposite premise
and should be revisited". It also removes the r5 VERDICT's explanation ("the cost is
the licence transition, not the width") for the node rejecting all eight of
L6_unrolled's r5 zmm candidates: if the ymm kernel is FMA-dense enough to sit in the
AVX2-heavy licence, there IS no transition when a zmm kernel replaces it. Two readings
survive of the r5 zmm rejection: (i) the ymm kernels actually run in the *light*
licence at 3.89 GHz (their ~50% FMA density stays under the threshold), so zmm costs a
real 26% of clock — then zmm is genuinely dead and kclk will read 3.89; or (ii) the
kernels run at 2.89 anyway and the r5 zmm shapes lost for *structural* reasons (all
three were 3-pass shapes; the node picks fused-family in every cell) or because the
race protocol itself penalised a minority-licence candidate. My kclk probe decides
(i) vs (ii) this round; the new kernels below bet a few race slots on (ii).

### What changed (one new kernel family, three race-protocol fixes)

1. **`fused_zx` family — zmm x-pass + the node-proven ymm FUSED_YZ stage** (+ `_pf`,
   `_pfw`), guarded by `__AVX512F__`, appended LAST in safest-first order so the ymm
   incumbents can never be displaced by less than the 1.5% margin. The x-pass at
   4 complex/zmm: lanes = 4 adjacent (y,z) sites, 9 groups in strictly ascending 64 B
   steps (the `_xa` lesson holds by construction), every access 64 B aligned, prefetch
   hooks re-granulated to 6 lines/group so the same 54 lines are covered in the same
   order. 9 × (6 ld + 6 st + 18 FP + 2 vpermilpd) = **288 uops vs the ymm pass's
   576**; whole-kernel ≈ 1546 uops vs fused's 1834 (−16%). FP-port floor UNCHANGED at
   486 cycles on the node (1 zmm FMA/cy = 2 ymm FMA/cy on this 1-FMA-unit SKU) — the
   target is uop count, not FLOPs. The zmm-x + register-fused-y/z combination is the
   one point in the space nobody has tried: L6_unrolled's rejected r5 set (z2p/z2s/z3t)
   were all 3-pass shapes. Structure adopted from their z2s x-pass; the pairing with
   FUSED_YZ is mine.
2. **Per-candidate licence dwell in the race**: each candidate now runs UNTIMED for
   ~0.5 ms immediately before each of its timed slices. CLX licence transitions
   persist >600 µs, so previously a zmm candidate was timed partly during its own
   down-transition AND poisoned the following ymm candidate's slice — the round-robin
   race was structurally biased against any minority-licence candidate, which is
   exactly the confound the r5 VERDICT's zmm reading could not exclude. Cost: ~0.1 s
   at B=1, ~0.6 s at B=32768, unscored.
3. **Settle spin before the race** (~100 ms dense ymm FMA) — adopted from L17_rader r5
   via L6_unrolled r6 (a fixed-order table on a ramping clock mis-ranked bit-identical
   work by 76% in their measurement; my round-robin blunts but does not eliminate it).
4. **Licence tail at the end of create()** — adopted from L6_unrolled's r6 lesson
   (their r5 B=1 regressed 0.219→0.227 because create() ENDED with a 512-bit probe and
   the driver's whole B=1 sample set fit inside the licence-recovery window): my
   create() now ends by dwelling ~3 ms in the CHOSEN kernel, so the driver gets a core
   in the scored kernel's own steady state, never a probe's. My kclk probe already ran
   last and mostly covered this; the explicit tail makes it unconditional (incl. forced
   picks).

Arithmetic untouched and closed since round 1: PFA 2×3, 44 flops / 36 scalar-shaped FP
instructions per line, 4752 flops per volume; 972 ymm FP for the ymm kernels, 162 zmm
+ 648 ymm FP for fused_zx. Grid 17 → 20 candidates.

### What was measured (wallaby; same-process race tables are the trustworthy statistic)

All 20 candidates pass the plan-time gate (rel L2 ≤ 1e-13 vs scalar). Driver: rel L2
2.24–2.43e-16 at B ∈ {1, 3, 64, 32768}, bit-identical re-runs, all PASS.

* B=1 (base-clock race session): everything inside the margin as always — 3pass 0.2460
  chosen; fused_zx 0.2539 vs fused 0.2495 (parity-ish). Driver min 0.129 µs (turbo).
* B=64 (base-clock race): fused_sp2_pf 0.2495 / fused_sp2_pf_xa 0.2503 lead the table,
  `fused` 0.2541 chosen through the anti-drift margin chain; fused_zx 0.2583 (−1.7% vs
  fused — SPR cannot show the CLX uop effect; its 6-wide front end never binds).
* B=32768 (turbo race, 113 MiB arena): normal-store rows fused_pft1w 0.3096,
  fused_sp2_pfw 0.3109, fused_pfw_xa 0.3114, fused_pfw 0.3116 — matches r6 exactly.
  **The zx surprise: in the non-W DRAM rows zx beats its ymm twin by 8–9% even on SPR**
  (fused_zx 0.3981 vs fused 0.4369; fused_zx_pf 0.3844 vs fused_pf 0.4201), though
  fused_zx_pfw 0.3199 stays −2.7% behind fused_pfw. Fewer load/store uops evidently
  help the streaming regime even on the wide machine. Wallaby still picks 3pass_nt_pf
  (0.2062) for its own DRAM — NT remains 0-for-5 on the node and is raced there only as
  a representative. Driver 0.274 µs/vol.
* Setup: 0.42 s (B=1) to 2.24 s (B=32768), unscored. Probes read
  clkS256/D256/S512/kclk = 2.10/2.09/2.10/2.10 (pinned session) and 4.10/4.09/4.10/4.10
  (turbo) — the machinery still discriminates both wallaby clock states cleanly.
* Cross-compile `-march=cascadelake`: all 20 kernels vector-spill-free (1 rsp ref =
  callee-save restore), zmm codelet emits vfmadd/vfnmadd zmm + vpermilpd zmm as
  intended, 21 prefetchw sites, single trailing vzeroupper.

### Node predictions (falsifiable via the description strings)

* **kclk is the round's headline number** (r6 prediction carries over): expected
  clkS256=3.89, clkD256=2.89, clkS512=2.89, and **kclk = 2.89** if the fused kernel's
  ~50% FMA density holds the AVX2-heavy licence. If kclk=2.89: B=1 = 633 cycles =
  1.30× the 486-cycle floor, and world (ii) is live — then `fused_zx` has a real shot
  at B=1/B=64 (−16% uops at zero clock cost). If kclk=3.89: the kernel lives in the
  light licence, zmm costs 26% of clock, the dwell-fair race will correctly reject the
  zx family, and the r5 zmm story stands *with the mechanism corrected* (light-licence
  ymm, not transition cost).
* B=1: `fused` at ≈0.219, or `fused_zx` at ≈0.19–0.21 if kclk=2.89 AND the B=1
  limiter tracks uop count (both prior uop/window hypotheses failed, so I rate this
  <50% but the race slot is free).
* B=64: `fused_sp2_pf_xa` or `fused_pf_xa` ≈0.215 (r6 prediction), `fused_zx_pf` a
  live outsider.
* B=4096: `fused_pf_xa`/`fused_pf` ≈0.387–0.392. B=32768: `fused_pfw_xa` ≈0.566; if
  the wallaby zx-in-DRAM effect transfers, `fused_zx_pfw` could take it — watch this
  cell.

### What was tried and did NOT work (with the number)

1. **fused_zx_pfw on wallaby DRAM: 0.3199 vs fused_pfw 0.3116 (−2.7%)** — the W hook
   at 6-lines-per-group (54 prefetchw in 9 bursts vs 18) evidently bunches the RFO
   issue too hard on SPR; not worth a re-granulated variant until the node says the zx
   family is alive at all.
2. Consciously skipped, per documented dead ends: a full-zmm fused y/z plane stage
   (L6_unrolled's z3t — the 4×6 transpose's vpermt2pd chain lost on BOTH machines,
   and 6 = 4+2 tiles the plane badly, my round-1 item 4); a zmm y-pass (only helps
   3-pass shapes, which the node has never picked); NT beyond the two standing
   representatives (0-for-5); MADV_HUGEPAGE (r2, kernel 5.15).
3. Not a failure but a cost worth recording: the licence dwell roughly doubles race
   time at the 113 MiB arena (setup 1.2 → 2.2 s at B=32768). Unscored, and the
   fairness it buys is the point.

### Borrowed / lent

Borrowed: the zmm x-pass layout from **L6_unrolled r5 (z2s)**; the settle spin from
**L17_rader r5** via L6_unrolled r6; the licence-tail fix from **L6_unrolled r6**; the
licence-equality fact from the **r7 brief / corpus §08**. Mine and lendable: the
**per-candidate licence dwell** — any entry racing mixed-width candidates (L=8 and
L=17 all do) has the same minority-licence bias in its tuner, and the fix is ~8 lines;
and the fused_zx pairing itself if it wins anywhere.

### Next

1. **Read kclk and the picks off the r7 leaderboard.** kclk=2.89 + any zx pick: the
   uop-count lever is real at equal clock — extend zmm into the y-stage of the fused
   kernels (zmm z0..3 + ymm z4..5 rows, saving another ~100 uops) before touching the
   transpose. kclk=2.89 + zx rejected: uops are simply not the B=1 limiter (third
   falsification); request the monitor's perf A/B (`L6_FORCE=fused` vs `fused_zx`,
   B=1: uops_issued vs cycles vs resource_stalls) — the forcing switch is shipped.
   kclk=3.89: close the zmm question at L=6 permanently and record the mechanism.
2. If `fused_zx_pfw` takes B=32768, re-granulate its W hook (interleave the 54
   prefetchw across the 9 groups in 3-line bursts) — the one knob item 1 of "did not
   work" points at.
3. The xa-vs-zp question (r6) finally resolves this round too; propagate whichever
   way it lands to the L=8/L=36 records as promised in my r6 Next.

---

## Round panel_r8 (2026-08-22, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r7 landed (node) — the round that settled the clock and killed the uop theory

**kclk = 2.89 GHz**, in every cell, from both entries' independent probes (clkS256=3.89,
clkD256=2.89, clkS512=2.89): the ymm kernels hold the AVX2-heavy licence, so **B=1's
0.219 µs is 632 cycles = 1.30× the 486-cycle FP-port floor — ~147 cycles of overhead, not
366**. And the zmm bet lost cleanly: my `fused_zx` family (−16% uops) and L6_unrolled's
`zxf`/`zff` (−17%/−25%) took **zero picks in eight cells across all three processes**, with
the licence bias removed by the per-candidate dwell and the clock measured equal. That is
the third independent falsification aimed at the same 147 cycles (uop count, after r4's
port-5 and r5's OoO-window). The r7 VERDICT's L=6 instruction: *stop shipping B=1
mechanisms; the one thing left is the monitor's perf-stat A/B* (its consolidated ask #3:
`uops_issued.any, cycles, resource_stalls.rob, cycle_activity.stalls_mem_any` on
`L6_FORCE=fused` at B=1 — the switch is shipped and waiting).

Cells: B=1 0.223 leaderboard / **0.2234 typical — I own the cell on the run distributions
third round running** (their 0.219 min is a 3%-low outlier vs their 0.2253 typical, VERDICT
§3b); B=64 0.226 vs their 0.214 (min; their typical ≈0.222 on a 3.6% spread — the honest
gap is ~2%); **B=4096 0.381, took the cell, my `fused_pf_xa` r6 bet finally scored
(−2.6%)**; B=32768 0.572 vs their 0.563. Node picks, stable: `fused` / `fused_sp2_pf_xa`
(2 of 3; `fused_pf_xa` once) / `fused_pf_xa` / `fused_pfw_xa`. So xa is node-confirmed at
B≥64 (as promised in my r6 Next: propagated — the ascending-order lesson is now proven
transferable to any strided pass raced on CLX; L=8/L=36 records take note).

### What changed (no new B=1 mechanism, per the VERDICT; two adopted fixes, one new batched mechanism, a hard prune)

1. **64-byte kernel-entry pinning (`__attribute__((aligned(64)))` on all 11 kernels) —
   ADOPTED FROM L6_unrolled r6.** Kernels run behind a function pointer; without pinning
   their placement moves whenever unrelated code changes, and that has twice produced
   same-pick-string swings on the node (their r5 B=1: 0.219→0.227 typical; the r5 VERDICT
   names the same disease at L36_mixedradix). My persistent same-shape B=64 deficit
   (0.226 vs their 0.214–0.222, three rounds, identical kernel structure since r6's xa)
   has exactly this fingerprint, and pinning is the one structural difference left
   between our files that I had not adopted. Verified in the cascadelake asm: all 11
   entries `.align 64`, spill-free (1 rsp ref = callee-save), prefetchw emitted.
2. **Grid pruned 20 → 11 on node evidence.** Deleted: the whole `fused_zx` zmm family
   (falsified r7, above — the ~150 lines of AVX-512 kernel code are gone from the file,
   which also serves the code-layout bet); all NT kernels (0-for-6 rounds on the node);
   `3pass_ip`, `3pass_pf`, `3pass_pfw`, `fused_sp2` plain, `fused_sp2_pf` (zp twin),
   `fused_sp2_pfw`, `fused_pft1w` (never picked). Every kernel the node picked in r4–r7
   survives. `fused` (zp-outer) now leads the safest-first order — it is the B=1 pick six
   rounds running, so it holds that cell as incumbent instead of having to re-beat
   `3pass` through the margin chain every plan.
3. **`_rot` kernels — per-volume scratch rotation, the round's one new mechanism (mine;
   inspired by L23_rader's r7 stride-ownership result).** My r2 record noted the 4K-alias
   placement is "B=1 only in effect: at B≥32 the volume stride (3456 ≡ −640 mod 4096)
   walks all 32 residues anyway" — including the replay-prone ones near 0. Nobody ever
   fixed that. The `_rot` twins advance t1 by the volume stride mod 4096 each volume
   (`t1_b = t1 + ((432·b) mod 512)` doubles — always a multiple of 128 B, so alignment
   is preserved), which makes BOTH cross-buffer store→load deltas (t1−in_b, out_b−t1)
   **constant in b and equal to the volume-0 optimum** place_scratch computes. Cost: one
   AND+ADD per volume; scratch window 3.4 → 7.5 KB (still comfortably L1). At b=0 a rot
   kernel is address-identical to its twin, so B=1 is untouched by construction. Twins:
   `fused_pf_xa_rot`, `fused_sp2_pf_xa_rot`, `fused_pfw_xa_rot`, each placed AFTER its
   parent — rotation is node-unmeasured, so it must earn the 1.5% takeover margin rather
   than inherit the incumbent slot (xa got the forward slot in r6 only because the node
   had already measured ascending winning).
4. Housekeeping kept intact: correctness gate, round-robin race with per-candidate
   licence dwell, settle spin, licence tail, clock probes incl. kclk, `L6_FORCE`,
   4K-alias placement. Arena grown 4096+864·8 → 4096+944·8 bytes for the rot window.

### Operation count

Unchanged and closed since round 1: DFT6 = 2·DFT3 + 3·DFT2 = 44 flops / 36 scalar-shaped
FP instructions per line; 4752 flops / 972 ymm FP instructions per 6³ volume. Rotation
adds one integer AND+ADD per volume (~0.02% of the 1834-uop fused body); pinning and the
prune add zero instructions. Output of every kernel is bit-identical to r7's (same
arithmetic, same order — verified by the plan gate and tryout's re-run check).

### What was measured (wallaby; same-process race tables are the trustworthy statistic — the clock lottery hit mid-session: kclk read 4.10, then 3.0/3.5 between invocations)

* B=1 (turbo session): all 11 inside the margin as always (3pass 0.1260 … fused 0.1277,
  chosen through safest-first); driver **0.132 µs min, sd 0.20%**. B=1 behavior is
  unchanged by design.
* B=64: race led by the sp2/pf family both sessions (session 1: fused_sp2_pf_xa 0.1282,
  rot twin 0.1291; session 2: fused_sp2_pf_xa_rot 0.1284, fused_pf 0.1289) — **rot =
  parity on wallaby, order flips between sessions**, exactly the SPR-blindness pattern xa
  showed in r6 (wallaby parity, node gap). Driver 8.836 µs/call = 0.138 µs/vol (busy
  machine, sd 11%).
* B=4096: `fused_pf_xa` chosen (0.1867; rot 0.1894, fused_pf 0.1879); driver 773.7 µs =
  0.189 µs/vol.
* B=32768 (base-ish session): `fused_pfw` 0.3134 chosen, fused_pfw_xa 0.3239,
  **fused_pfw_xa_rot 0.3230 — rot marginally ahead of its parent here**, first weak
  positive; driver 13278 µs = 0.405 µs/vol (that session's clock). NT rows no longer
  exist to win wallaby's DRAM regime, so the wallaby pick is now node-shaped.
* Correctness: **rel L2 2.24–2.43e-16 at B ∈ {1, 3, 64, 4096, 32768}, all PASS,
  bit-identical across re-runs.** All 11 candidates pass the plan gate. Setup 0.32 s
  (B=1) to 1.47 s (B=32768), unscored.

### Node predictions (falsifiable via the description strings)

* **B=1: `fused` at 0.219–0.223, unchanged or slightly better.** No mechanism was aimed
  here (per the VERDICT); the pinning + 9-KB-smaller .text can only move process-luck
  variance. If the cell moves >1%, code layout was a real part of the L=6 B=1 story.
* **B=64 is the round's cell to read.** The bet is that pinning (+ prune) closes the
  same-shape gap to L6_unrolled: prediction `fused_sp2_pf_xa` or a rot twin at
  **0.214–0.222**, down from my three-round 0.226 plateau. If it stays 0.226 with the
  same pick, code layout is falsified for this gap and the remaining suspect is their
  VD6 factorization (radix-2-first vs my radix-3-first — same counts, different
  dependency graph), which I would then A/B in r9.
* B=4096: `fused_pf_xa` (or rot) holds ≈0.381.
* B=32768: `fused_pfw_xa` ≈0.566–0.572, or **`fused_pfw_xa_rot` if the alias-replay
  mechanism is real on CLX** — at 32768 volumes all 32 residues are exercised ~1024
  times each, so this cell is rot's cleanest test. A rot pick here at ≤0.563 would also
  close the last gap to L6_unrolled.
* kclk: 2.89 again (regression check on the probe, nothing more).

### What was tried and did NOT work (with the number)

1. **Rotation on wallaby is a null: parity within ±0.5% in every cell** (best case
   0.3230 vs 0.3239 at B=32768, worst 0.1894 vs 0.1867 at B=4096). Expected — the r1
   +22% aliasing measurement was Haswell, the r7 L23 result was CLX, and SPR's deeper
   store buffer hides replays; wallaby cannot adjudicate this mechanism, the node race
   will. Recorded so the parity is not misread as a kill.
2. Consciously NOT done, and the reasons: any new B=1 kernel (VERDICT instruction;
   three falsified theories are enough — the perf-stat A/B is the only honest next
   step and it is monitor-side); re-adding NT or T1 hooks (0-for-6 and never-picked
   respectively); a zmm revival (falsified r7 with the cleanest experiment of the
   series); adopting L6_unrolled's radix-2-first VD6 (identical op count, no evidence
   it matters — it is the *named fallback suspect* if pinning fails to close B=64).
3. Cost worth recording: deleting NT means wallaby's DRAM race no longer has its local
   favourite; wallaby driver numbers at B=32768 look ~25% worse than r7's NT-picked
   ones. That is a wallaby-only cosmetic regression — the node rejected NT six rounds
   running, and the scored pick set is unchanged.

### Borrowed / lent

Borrowed this round: **64-byte kernel pinning from L6_unrolled r6** (named mechanism,
node-motivated); the stride-ownership framing behind rotation from **L23_rader r7**
(their t1 plane-stride pad; my per-volume phase-lock is the new part). Lendable: the
**per-volume scratch rotation** pattern applies to any entry whose per-volume scratch is
static while the driver's volume stride is not a multiple of 4096 — L=8 (8.2 KB volumes,
stride 8192 B ≡ 0 mod 4096: their deltas are ALREADY constant, so it does NOT apply
there — check your stride first), L=13/17/23/36/45 all have non-4K-multiple volume
strides and static scratch. One line of address arithmetic per volume buys a constant,
optimal alias phase for the whole batch.

### Next

1. **Read the B=64 pick and number.** Closed to ≤0.222 → pinning/layout was the gap;
   propagate "pin every function-pointer-reached kernel" as a standing rule to the
   other geometries' records. Still 0.226 → A/B the VD6 factorization (radix-2-first
   twin codelet, one extra candidate, zero risk).
2. **Read whether any `_rot` twin is picked.** Picked at B=32768 or B=64 → the alias
   mechanism is real on CLX; extend rotation to the 3pass family's t2 and tell the
   L=13/17/23/36 entries. Rejected everywhere → record that CLX demand-side replays at
   these strides are below 1.5%, and close the item.
3. **The B=1 perf-stat A/B remains the monitor's** (VERDICT ask #3, `L6_FORCE=fused`,
   B=1). If it lands and shows `cycle_activity.stalls_mem_any` dominating, the t1
   store→load joint theory finally gets its number and r9 has a real target; if it
   shows nothing, L=6 B=1 is done at 1.30× floor and I second the redeployment
   question the VERDICT raises.
4. B=32768 stays bandwidth-closed for non-rot shapes; do not spend kernels there
   beyond the rot experiment (fourth round this note survives).

---

## Round panel_r9 (2026-08-22, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r8 landed (node)

Leaderboard: B=1 0.220 vs L6_unrolled's 0.217, B=64 0.222 vs 0.214, B=4096 **0.389,
held the cell** (their 0.391), B=32768 0.572 vs 0.565. But the VERDICT's per-run
reading matters more than the minima: **on medians I own B=1 (0.2205 vs their 0.2256,
fourth round the leaderboard reads opposite to the distributions)** and B=4096; they
own B=64 (honest gap ~2.7%: 0.228 vs 0.222 medians) and B=32768. The r8 experiments
resolved: **my pinning bet FAILED its own pre-registered criterion** (B=64 stayed on
the 0.226–0.228 plateau, pick reverted to plain `fused_pf` — code layout is falsified
for the B=64 gap), and **the `_rot` scratch-rotation twins took zero picks in all
eight cells** — per-volume alias phase-lock is a null on CLX; demand-side 4K replays
at these strides sit below the 1.5% margin. Also mine: **B=4096 regressed 0.381 →
0.389 (+2.1%) with an identical pick string** (`fused_pf_xa`, 3/3) — the fourth
instance on the board of the refactor-around-an-untouched-hot-path disease; nothing
on that path changed but the deleted candidates around it.

Score sheet after r8: **six falsified L=6 B=1/B=64 mechanisms** (r4 port-5/uop-mix,
r5 OoO window, r7 width/uop count, r8 pinning, r8 rotation, plus their `restrict`).
The VERDICT's L=6 instruction: the mechanism list is empty — run the counter (its
consolidated ask 3), **or better, L36_pfa's trick: put the discriminating
measurement inside `create()` and route it out through `fft3d_description()`**.

### What changed (one adopted codelet A/B, one in-plan measurement, one prune)

1. **`_d2` kernels — radix-2-first DFT6 codelet, ADOPTED FROM L6_unrolled's VD6
   factorization.** This is the r8 VERDICT's (and my own r8 record's) *named standing
   suspect* for the three-round same-shape B=64 gap, now that layout is falsified. My
   `DFT6V` is radix-3-first: two conjugate DFT3s over the CRT groups (x0,x4,x2),
   (x3,x1,x5), then six add/sub joins at the END feeding the stores. Their `VD6` is
   radix-2-first: the three DFT2s (x0±x3, x4±x1, x2±x5) run FIRST (by linearity
   DFT3(a)±DFT3(b) = DFT3(a±b)), then two conjugate DFT3s whose FMA results feed the
   stores DIRECTLY. Identical count — 18 FP = 12 add/sub + 6 fma, + 2 vpermilpd,
   depth-4 critical path both — but a different dependency graph: the 6-way join
   moves from the bottom to the top, and the last op before every store is an FMA
   instead of an add. I derived my own `DFT6V2` from my scalar reference rather than
   transliterating theirs (same graph shape, my sign/constant conventions). Every
   pass macro now takes the codelet name as a parameter (preprocessed output for
   `DFT6V` is identical to the r8 file, so incumbent codegen is untouched); twins
   `fused_d2`, `fused_pf_d2`, `fused_pf_xa_d2`, `fused_sp2_pf_xa_d2`,
   `fused_pfw_xa_d2` cover all four cells, each placed AFTER its parent in
   safest-first order so the new graph must earn the 1.5% takeover margin.
2. **Boundary probes — the round's measurement, implementing the VERDICT's
   recommended route (mechanism from L36_pfa's r8 in-plan probe).** Four unraceable
   probe kernels timed at nb=1 in `create()` (round-robin, per-kernel min, under the
   same placed-scratch addresses the scored B=1 run sees), reported in the
   description string:
   * `bf` = the real plain `fused` kernel (x: in→t1, fused y/z: t1→out);
   * `bx` = x-pass alone; `byz` = fused y/z alone (t1 pre-filled, L1-resident);
   * `bsp` = **the dependence-broken twin**: x-pass in→t1, fused y/z reads t2
     (pre-filled with the same x-pass output). Identical instruction stream to
     `bf` — only the cross-pass store→load forwarding joint is severed.
   Readings: `bf − bsp` = the direct price of the t1 store→load joint at equal
   uops; `bf − bx − byz` = the fusion deficit vs perfectly-overlapped parts. This
   finally tests the ONLY surviving B=1 theory (the t1 joint, suspect since r3)
   without needing the monitor's perf run. Cost ~40 ms of unscored setup.
3. **Grid: deleted the 3 falsified `_rot` twins, added the 5 `_d2` twins → 13
   candidates.** Everything else (correctness gate, round-robin race with licence
   dwell, settle spin, licence tail, kclk + clock probes, `L6_FORCE`, 4K placement,
   64B pinning) carried unchanged.

### Operation count

Unchanged and closed since round 1: DFT6 = 2·DFT3 + 3·DFT2 = 44 flops / 36
scalar-shaped FP instructions per line; 4752 flops / 972 ymm FP + 108 vpermilpd per
6³ volume — for BOTH codelets (d2 is the same arithmetic reassociated; the plan gate
holds it to rel L2 ≤ 1e-13 of the scalar reference, and measured driver accuracy is
unchanged at 2.2–2.4e-16).

### What was measured (wallaby; same-process race tables are the trustworthy statistic)

* All 13 candidates pass the plan gate. Driver: rel L2 **2.24–2.43e-16** at
  B ∈ {1, 3, 64, 4096, 32768}, all PASS, bit-identical across re-runs.
* B=1 (base-clock race session): `fused` 0.2493 chosen; `fused_d2` 0.2519 (−1.0%),
  every d2 twin 0.4–1.0% behind its parent — **wallaby is codelet-blind, exactly as
  it was xa-blind in r6** (SPR's 6-wide front end and 512-uop ROB hide graph-shape
  effects that CLX exposes; the node decides this). Driver 0.130–0.131 µs (turbo).
* B=64: `fused` 0.2552 chosen through the margin chain; table led by `fused_pf`
  0.2515 / `fused_sp2_pf_xa` 0.2520 / `fused_pf_xa` 0.2527; d2 twins 0.4–0.8%
  behind parents except `fused_sp2_pf_xa_d2` at −3.3% (0.2602 vs 0.2520 — the d2
  codelet's top-heavy adds may fight sp2's interleave; if the node agrees, that one
  twin dies next round). Driver 8.876 µs/call = 0.139 µs/vol.
* B=4096: PASS, 781.7 µs = 0.191 µs/vol (turbo session). B=32768: PASS, 12590 µs =
  0.384 µs/vol, setup 1.6 s.
* **Boundary probes on wallaby (base-clock 2.1 GHz session): bf=249.5, bsp=249.3,
  bx=61.9, byz=188.0 ns → joint = +0.2 ns ≈ ZERO, and bx+byz = 249.9 ≈ bf.** On SPR
  the x→yz store→load joint is completely hidden and the two stages overlap
  perfectly — consistent with wallaby sitting at ~1.08× the FP floor (524 cycles at
  2.1 GHz). This is the control experiment: SPR shows the probe reads clean zero
  where zero is expected. The node number is the payload.
* Cross-compile `-march=cascadelake`: **all 13 raced kernels + 3 probe kernels
  vector-spill-free** (1 rsp ref = callee-save restore, including both sp2 forms),
  9 prefetchw sites, every raced entry `.align 64`.

### Node predictions (falsifiable via the description strings)

* **The boundary probe is the round's headline number.** If `bf − bsp` ≈ 45–55 ns
  (~130–160 cycles at kclk 2.89), the t1 joint IS the B=1 overhead, six falsified
  mechanisms get their survivor, and r10 has a concrete target: double-buffer t1
  between calls (severing the joint exactly as bsp does, at the cost of one more
  L1-resident volume) or interleave the first fused plane into the x-pass tail. If
  `bf − bsp` ≈ 0 but `bf − bx − byz` is large, the cost is fusion-scheduling, not
  forwarding. If both ≈ 0, B=1's ~145 cycles are OUTSIDE the kernel (call/dispatch/
  driver overhead) and L=6 B=1 is genuinely done at 1.30× floor.
* B=1: `fused` at 0.218–0.223 (d2 must beat it by 1.5% to take the cell; if
  `fused_d2` IS picked, the codelet graph matters even at B=1 and the gap story
  closes at both cells at once).
* **B=64 is the experiment cell: if the node picks `fused_pf_d2`/`fused_pf_xa_d2`/
  `fused_sp2_pf_xa_d2` and the cell drops to ≤0.222, the factorization was the
  three-round gap** and the lesson (store-feeding FMAs beat store-feeding adds on
  CLX's 1-store-port back end) propagates to every unrolled codelet on the board.
  If it stays `fused_pf`-family at 0.226–0.228, the d2 theory dies as the seventh
  falsification and the honest remainder is driver-side (their file vs mine differ
  only in things already individually falsified — then the B=64 gap is process
  luck on a 3.6% spread, per the r8 VERDICT's own median caveat, and I stop).
* B=4096: `fused_pf_xa` (or its d2 twin) at 0.381–0.389 — watching whether the r8
  regression reverts with the grid re-shuffle (same disease class, opposite sign).
* B=32768: `fused_pfw_xa` ≈ 0.565–0.572 (compulsory-traffic floor; d2 cannot move
  DRAM bandwidth).
* kclk: 2.89 (regression check).

### What was tried and did NOT work (with the number)

1. **`fused_sp2_pf_xa_d2` on wallaby: 0.2602 vs parent 0.2520 (−3.3%) at B=64** —
   the only d2 twin clearly behind its parent locally. Kept for the node race
   (wallaby is codelet-blind and CLX is the machine the mechanism targets), but
   flagged: if the node also rejects it while picking flat-fused d2 twins, the d2
   graph composes badly with the sp2 software pipeline specifically.
2. Consciously NOT done: any B=1 kernel mechanism beyond the probe (the VERDICT's
   standing instruction — measure first; six falsifications are enough); reviving
   `_rot` (falsified r8, 0-for-8) or zmm (falsified r7) or NT (0-for-6) or
   MADV_HUGEPAGE (r2, kernel 5.15); adopting their exact VD6 register naming rather
   than deriving DFT6V2 from my own conventions (bit-identical requirement to my
   scalar reference decides the signs; the graph shape is what is being tested).
3. Cost worth recording: the CD-parameter macro refactor touches every pass macro.
   The preprocessed expansion of every incumbent kernel is unchanged (verified by
   the identical wallaby race ordering and bit-identical outputs), but the file has
   twice seen same-pick regressions after exactly this kind of around-the-hot-path
   churn (r8 B=4096 +2.1%), so the B=4096 prediction above doubles as the
   regression's re-roll.

### Borrowed / lent

Borrowed this round: **the radix-2-first DFT6 factorization from L6_unrolled's VD6**
(the A/B is the round's kernel bet; graph adopted, derivation mine) and **the
in-plan measurement route from L36_pfa's r8 probe** (their fu−p1−p2w decomposition
settled a three-round question in one leaderboard line; my bf/bsp/bx/byz is the same
idea aimed at the t1 joint, with the dependence-broken twin as the new part).
Lendable: **the dependence-broken-twin probe pattern** — any entry with a
scratch-mediated pass boundary (all of L=8, L=36's pencil stages, L=17's convolution
staging) can sever its store→load joint at identical uops by double-filling a second
scratch and timing both twins in `create()`; it turns "the monitor should perf-stat
this" into one description-string field.

### Next

1. **Read `bf/bsp/bx/byz` off the r9 leaderboard** (any L=6 cell reports them; B=1's
   process is the cleanest). Joint ≈ 45–55 ns → build the t1 double-buffer kernel
   (`t1` alternating per CALL, not per volume — one extra L1 volume, zero extra
   uops) as r10's one B=1 mechanism. Joint ≈ 0 twice over → declare B=1 closed at
   1.30× floor and formally propose the panel consolidate the two L=6 slots (the r8
   VERDICT already asks; the medians say the entries split 2–2, so consolidation
   should keep whichever file wins this round's d2 adjudication).
2. **Read the B=64 pick.** d2 picked and ≤0.222 → propagate the store-feeding-FMA
   lesson to L=8 (their dft8s has the same join-at-the-bottom shape) and L=36's
   codelets. d2 rejected at 0.226+ → seventh falsification; stop attacking B=64 and
   accept the spread reading.
3. B=4096: if the +2.1% r8 regression reverts with an unchanged pick, record the
   refactor-churn disease as ±2% noise-floor at this cell and stop reading moves
   inside it.
4. B=32768 stays bandwidth-closed; no kernels spent there (fifth round this note
   survives).

---

## Round panel_r10 (2026-08-22, dev machine = wallaby, Sapphire Rapids Gold 6448Y)

### Where round panel_r9 landed (node) — the round the association order paid

The best round this file has had. **`fused_d2` took B=1 3/3: 0.2068 µs min (three
processes agreeing to 0.03%), median 0.2205 → 0.2152, −6.2%** — the first time the
cell was decided on non-overlapping distributions, and it is mine. **`fused_pf_xa_d2`
took B=64 3/3 (median 0.2281 → 0.2214, −3.0%)**, firing my pre-registered criterion
exactly: the radix-2-first codelet graph WAS the three-round same-shape gap.
L6_unrolled's mirror A/B agreed from the other side (their `f3d` = +3.3–6.2%, their
`ab1` f = 212–221 ns vs f3 = 225–229 ns). The r9 VERDICT's law: **on Cascade Lake,
store-feeding FMAs beat store-feeding adds by 3–6% on identical arithmetic.** On
distributions I now own B=1 (9.1%) and B=64 (0.4%), B=4096 is a tie, and L6_unrolled
owns only B=32768 (0.5656 vs my 0.5733). The VERDICT's slot ruling: keep L6_pfa,
redeploy the other L=6 slot; **spend nothing further on B=1** — the boundary probe
read `bf − bsp` = +0.1..+0.6 ns and `bx + byz` ≈ `bf` (both branches ZERO against a
pre-registered 45–55 ns criterion), which killed the seventh and last named B=1
mechanism. B=1 is 598 cycles at kclk 2.89 = **1.23× the 486-cycle FP-port floor**,
and the residual has no named suspect left in either entry's record.

### What changed (one move: the incumbency flip; zero new kernels, zero deletions)

**The safest-first order was re-derived from the r9 node evidence: every `_d2` twin
now sits BEFORE its radix-3-first parent** (except the sp2 pair, see below). The
mechanism this captures is the takeover margin's blind spot, and it is exactly where
my one remaining cell loss lives:

* At B=32768 the node picked my `fused_pfw_xa` (radix-3-first) 3/3 while
  L6_unrolled's same-shape radix-2-first kernel (`fused_pfw`, their VD6 + ascending
  x-pass + pfT0+W — structurally identical to my `fused_pfw_xa_d2`) won the cell by
  ~1.4%. A ~1.4% DRAM-regime delta cannot clear the 1.5% takeover margin from the
  trailing slot, so my own copy of the winning kernel was systematically locked out
  by its ordering. Flipping the pair makes parity-or-better keep d2.
* Same logic at B=4096 (cell is a tie; the node picked the parent `fused_pf_xa` 3/3
  with the d2 twin sub-margin behind it).
* At B=1/B=64 the flip just makes the proven 3/3 winners (`fused_d2`,
  `fused_pf_xa_d2`) incumbents instead of challengers — they no longer have to
  re-beat their parents through the margin chain every plan.

Exception kept deliberately: **`fused_sp2_pf_xa` stays ahead of its d2 twin** — r9
measured the d2 graph composing badly with the sp2 interleave (wallaby −3.3%, node
never picked it), so the twin still has to earn the slot.

Implementation detail that matters for the layout-disease history of this file: the
change is a **pure reorder of the `cand[]` table (data), plus a name-independent
lookup for the boundary-probe guard** (it assumed `k_fu` was `cand[0]`; it now finds
the entry). No kernel was added, deleted, or edited — **.text is byte-identical to
r9**, so the r8-style refactor-around-the-hot-path regression class cannot fire.
Everything else (correctness gate, round-robin race with licence dwell, settle spin,
licence tail, clock + kclk probes, boundary probes as a regression check, `L6_FORCE`,
4K placement, 64B pinning) carried unchanged.

### Operation count

Unchanged and closed since round 1: DFT6 = 2·DFT3 + 3·DFT2 = 44 flops / 36
scalar-shaped FP instructions per line; 4752 flops / 972 ymm FP + 108 vpermilpd per
6³ volume, both codelets (d2 is the same arithmetic reassociated). A table reorder
adds zero instructions anywhere.

### What was measured (wallaby; same-process race tables are the trustworthy statistic)

* All 13 candidates pass the plan gate. Driver: **rel L2 2.34–2.43e-16 at
  B ∈ {1, 64, 4096, 32768}, all PASS, bit-identical across re-runs.**
* B=1 (base-clock race session): `fused_d2` **chosen as incumbent** and holds
  (parent `fused` 0.2493 vs d2 0.2520 — wallaby codelet-blind as in r9, sub-margin,
  the flip decides it the node-proven way). Driver 0.132 µs min, sd 0.13% (turbo
  session).
* B=4096: **`fused_pf_xa_d2` chosen, 0.2034 vs parent 0.2044** — the exact
  sub-margin case the flip was built for, now resolving to the d2 graph. Driver
  801.8 µs = 0.196 µs/vol (turbo).
* B=32768: **`fused_pfw_xa_d2` chosen** (0.3065; parent 0.3059, plain pfw 0.3056 —
  parity within 0.3%, wallaby cannot see the CLX codelet effect; the node data says
  d2 wins there). Driver 13255.7 µs = 0.4045 µs/vol (that session's clock). B=64:
  8.857 µs/call = 0.138 µs/vol, pick `fused_pf_xa_d2` family as in r9.
* Boundary probes (regression check): bf=249.4 bsp=249.3 bx=61.9 byz=188.0 ns —
  joint 0.1 ns, identical to r9's wallaby reading. Clock probes healthy
  (2.09–2.10 GHz pinned session).

### Node predictions (falsifiable via the description strings)

* **B=32768 is the round's cell: pick flips to `fused_pfw_xa_d2` and the cell drops
  to ≈0.565–0.568, closing the last gap to L6_unrolled.** If the pick flips but the
  number stays 0.573, the codelet does NOT matter in DRAM and their 1.4% edge is
  something else (then diff the two files' pfw kernels' asm — but I checked the
  shapes: they are structurally identical). If the node picks the parent through
  some >1.5% reversal, the r9 d2 story has a regime boundary worth recording.
* B=4096: `fused_pf_xa_d2` at 0.389–0.396 (tie regime; parity or a hair better).
* B=1: `fused_d2` at 0.206–0.215, unchanged — nothing on that path moved.
* B=64: `fused_pf_xa_d2` at 0.215–0.222, unchanged.
* kclk 2.89, joint ≈ 0 (regression checks).

### What was tried and did NOT work (with the number)

1. Nothing failed this round — by design the narrowest round this file has shipped
   (the r9 VERDICT: "there is no kernel work left" at L=6; the one liberty taken is
   that the incumbency flip is selection-plumbing, not a kernel). The negative
   results that scoped it: seven falsified B=1 mechanisms (r4–r9, see r9 record),
   the boundary probe's double zero (r9, node), sp2+d2 composition (r9, −3.3%
   wallaby, 0 node picks — why the sp2 pair kept its order).
2. Consciously NOT done: a third association order / mixed-codelet FUSED_YZ (d2 for
   the store-feeding z-stage, radix-3-first for the register-feeding y-stage) — the
   r9 VERDICT directs the association-order search at OTHER geometries' codelets
   (L=8 dft8s, L=36 DFT4/DFT9, L=13 chunk13), wallaby is codelet-blind so I cannot
   pre-screen it locally, and it would break the .text-byte-identical property that
   protects this round from the layout noise floor. If the panel keeps funding L=6
   kernels after the slot consolidation, that experiment is the only unplayed card
   in the file.
3. Not done on instruction: anything aimed at B=1 (VERDICT, twice now).

### Borrowed / lent

Borrowed this round: nothing new — the round cashes in r9's adoption (L6_unrolled's
VD6 graph) at the two cells where the takeover margin had locked my copy of it out.
Lendable: **the incumbency-flip lesson generalizes** — any entry running a
safest-first tournament with a takeover margin should re-derive its candidate ORDER
from node pick strings each round, because a node-proven sub-margin winner in a
trailing slot is invisible to the tuner by construction. Check your own pick strings
for cells where a rival's structurally-identical kernel beats your chosen one by
less than your margin: that is not a kernel gap, it is an ordering bug.

### Next

1. **Read the B=32768 pick and number.** Flip + ≈0.566 → L=6 is converged in all
   four cells and genuinely closed (B=1 1.23× floor, B=64/4096 owned, B=32768 at the
   compulsory-traffic floor with the right codelet); second the VERDICT's
   consolidation and redeploy proposal. Flip + 0.573 → codelet is DRAM-irrelevant;
   record it and stop (the cell difference would then be inside the two entries'
   process spread).
2. If the panel consolidates L=6 to this file, the maintenance posture is: change
   nothing without node evidence, keep the probes as regression tripwires, and
   propagate the two exportable lessons (store-feeding-FMA association order;
   incumbency re-derivation) to the geometries still open.
3. B=1 stays closed per the VERDICT. Sixth round this note survives for B=32768's
   bandwidth story; the codelet flip is address-free and does not contradict it.
