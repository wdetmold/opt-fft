# L8_radix8 — strategy record

Geometry: **L = 8** (8³ = 512 complex doubles = 8 KiB per volume, L1-resident).
File: `impl/L8_radix8.c`.  `fft3d_name()` = `L8_radix8`.

---

## Round 1 (first implementer, this session)

### Technique

One fully unrolled **radix-8 codelet**, applied along all three axes, with the codelet
operating on **split-complex vectors whose lanes hold 8 independent lines**.  No twiddle
table exists at run time: every 8th root of unity is ±1, ±i or ±(1±i)/√2, so the only
multiplicative constant in the whole transform is `C = 1/√2`, broadcast once per kernel
call.  Multiplication by ±i costs nothing in split layout — it is a register rename plus a
sign folded into the following add/sub — and radix-8 is full of them (§04 §2.4).

**Index mapping.**  Input/output element (x,y,z) of a volume lives at double offset
`x*128 + y*16 + 2z` (`+1` for the imaginary part).  Per volume:

*Pass 1, once per x-plane (8 iterations, each plane = 64 complex = 128 doubles):*

1. 16 aligned 8-double loads: `A[y] = plane + y*16` (z = 0..3, interleaved),
   `B[y] = plane + y*16 + 8` (z = 4..7).
2. **8×8 double transpose of {A[y]} and of {B[y]}** (48 shuffles).  This single step both
   de-interleaves re/im and moves z into registers: transposing the 8×8 block of doubles
   `M[y][d] = A[y][d]` yields register `d`, and `d = 2z` is `re(z)` with lanes = y,
   `d = 2z+1` is `im(z)`.  A gives z = 0..3, B gives z = 4..7.  (Doing the de-interleave
   separately and then transposing costs 64 ops instead of 48 — the fusion is free.)
3. radix-8 along z (54 instrs) → registers = k2, lanes = y.
4. **8×8 transpose of the 8 re-registers and of the 8 im-registers** (48 shuffles) →
   registers = y, lanes = k2.
5. radix-8 along y (54 instrs) → registers = k1, lanes = k2.
6. 16 stores to an 8 KiB split-complex scratch at `scr + x*128 + k1*16` (re) and `+8` (im).

*Pass 2, once per k1 (8 iterations):*

7. Load `scr + x*128 + k1*16` (+8) for x = 0..7 → registers = x, lanes = k2.
8. radix-8 along x (54 instrs) → registers = k0, lanes = k2.
9. Interleave each (re,im) pair back with 2 two-source permutes and store the 128
   contiguous bytes of output row (k0,k1,·) at `out + k0*128 + k1*16`.

**Why exactly two transposes per plane, and why that is the floor.**  The lane axis is z on
entry (the contiguous axis) and must be z again on exit, but the z-butterflies need z in
*registers*.  So the z axis has to leave the lanes and come back: two transposes per
element, no matter how the three axis passes are ordered.  I enumerated the orderings; see
"what did not work" for the numbers.  Within that, 24 ops per 8×8 double transpose is
information-theoretically minimal for two-source permutes (each output register draws from
8 sources ⇒ ≥ log₂8 = 3 levels × 8 registers), and the final interleave at 1 op per output
register is likewise minimal (2 sources ⇒ 1 level).

### Derivation and operation count

Radix-8 = three radix-2 stages with the trivial twiddles written out (DIT; the bit reversal
is absorbed by choosing which register feeds which butterfly, so it is free):

```
b0,b1 = x0±x4   b2,b3 = x2±x6   b4,b5 = x1±x5   b6,b7 = x3±x7          16 real adds
c0,c2 = b0±b2   c1,c3 = b1∓i·b3   c4,c6 = b4±b6   c5,c7 = b5∓i·b7      16 real adds
X0,X4 = c0±c4                                                            4 real adds
X2,X6 = c2∓i·c6                                                          4 real adds
d = w8^1·c5 = (C(c5r+c5i), C(c5i−c5r))      X1,X5 = c1±d          2 mul + 2 add + 4 add
e = w8^3·c7 = (C(c7i−c7r), −C(c7i+c7r))     X3,X7 = c3±e          2 mul + 2 add + 4 add
```

= **4 real multiplies + 52 real adds = 56 flops**, which is the published optimum for n = 8
and is reached by three different routes at once (Burrus T7.1 n=8 row; T9.1 radix-8
M₂..M₅ = 4 / A = 52; T9.1 split-radix likewise; FFTW's `n1_8`), and equals Yavne's closed
form `4N log₂N − 6N + 8 = 56` (§01 §2.3).  **There is nothing left to find in the
arithmetic**; the record should say so to stop the next implementer looking.

I then fold each `(x±y)·C` pair into `1 mul + 2 FMA` (`p = C·a; C·b ± p`), which trades
2 flops for 2 instructions: **54 instructions / 58 flops per codelet**.  FMA also improves
the rounding of the only inexact operation in the kernel.

Per volume (192 lines = 3 axes × 64), vectorised 8 lines to a vector:

| | count | port (Skylake-SP 512-bit scheme) |
|---|---|---|
| arithmetic instructions | 24 codelets × 54 = **1296** | p0 (p0+p5 if the part has 2 FMA units) |
| shuffles | 768 transpose + 128 interleave = **896** | p5 |
| loads / stores | 256 / 256 | p2,p3 / p4 |
| total | **2704** instructions per 8³ transform | |

Classic flop accounting: 192 × 56 = **10752 real flops** per volume, i.e. 21 flops/point,
versus the driver's nominal `5N log₂N` = 23040 (the driver's GF/s therefore over-reads our
real rate by 2.1×, uniformly for every entry).  `python/fft3d.py`'s `line_cost(8) = 48`
complex multiply-adds = 288 real flops per line, **5.1× our 56**, because it charges a
generic complex twiddle multiply for every butterfly; that ratio is exactly the
"special-case ω = 1, ±i, ±(1±i)/√2" lever §01 §2.3 measures as a 12× swing on multiplies.

Cycle estimate on the grading node: the Gold 5218 is a **one-512-bit-FMA-unit** part
(§04 §8.1 quoting Intel §18.21), so 512-bit FP issues 1/cycle → 1296 cycles bound by p0,
with the 896 shuffles hiding under it on p5.  If it turns out to have two FMA units, FP and
shuffles share p0+p5 and the bound is (1296+896)/2 ≈ 1096 cycles.  Either way ≈ 1100–1300
cycles ≈ 0.48–0.57 µs at 2.3 GHz, against MKL's 0.653 µs.  **Note the consequence of the
single FMA unit: 512-bit and 256-bit code have identical FP throughput (8 doubles/cycle)
on this SKU.  AVX-512's entire advantage here is that it halves the shuffle, load and
store instruction counts, not the arithmetic.**

### Layout and SIMD decisions

* **Split complex, batch-minor-in-lanes-of-lines.**  A vector holds one component (re or
  im) of 8 different lines.  Every scalar operation of the codelet becomes exactly one
  vector instruction with zero cross-lane traffic inside the butterflies (§04 §3.1's
  `DFT_n ⊗ I_ν` vector terminal).  Interleaved layout was costed and rejected: the same
  complex add costs the same there (re and im share the register), but each ×(±i) needs a
  port-5 permute and each w8^{1,3} needs a permute + xor, giving 38 ops per 4 lines
  (9.5/line) against split's 54 per 8 lines (6.75/line) — 41% worse.
* **One 8-wide vector abstraction, three backends, one kernel body.**  `KERNEL_BODY` and
  `RADIX8` are object-like macros instantiated three times with different primitive
  macros: portable `struct {double d[8];}`, AVX2 `struct {__m256d lo, hi;}`, and AVX-512
  `__m512d`.  This is not cosmetic — the dev machine is Haswell, so **the algorithm is
  verified locally through the portable and AVX2 instantiations of the same text that the
  node compiles into zmm**, leaving only the ~8 AVX-512 primitives unexercised.  Those were
  checked by emulating the documented semantics of `vunpcklpd/vunpckhpd`,
  `vshuff64x2` and `vpermt2pd` and confirming that the 3-level network really transposes
  and that the permute index vectors really interleave (scratch program, 0 mismatches).
* **AVX-512 8×8 transpose = 24 ops:** 8 `vunpckl/hpd` (swaps i-bit0 with lane-bit0),
  8 `vpermt2pd` with the two index vectors `{0,1,8,9,4,5,12,13}` / `{2,3,10,11,6,7,14,15}`
  (swaps bit1; `vshuff64x2` *cannot* do this level — it can only produce A,A,B,B block
  patterns and this level needs A,B,A,B), 8 `vshuff64x2` with imm 0x44 / 0xEE (swaps bit2).
  Interleave = `vpermt2pd` with `{0,8,1,9,2,10,3,11}` / `{4,12,5,13,6,14,7,15}`.
* **Register budget.**  16 live data vectors + ~6 temps + 2 index constants + the C
  broadcast.  On AVX-512 that fits 32 zmm: **the generated kernel has zero stack traffic**
  (checked by grepping the assembly for `vmov*pd` against `%rsp`/`%rbp`: 0 stores, 0
  loads).  This is the concrete answer to §04 §4.1's open question at L=8 — the batch-style
  16-vector codelet spills nothing on AVX-512.  On AVX2 the same body needs 32 ymm on a
  16-register machine and spills hard: **192 stack stores + 179 stack reloads + 387
  register-register moves** in the static loop bodies.  §01's prediction ("for AVX2 the
  batch-major form of even n = 6 and n = 8 will spill a little") is an understatement; it
  spills a lot, and that is the whole reason to prefer the AVX-512 nodes at L=8 (§07 §4.3).
* **Two passes, both inside L1.**  in (8 KiB) + scratch (8 KiB) + out (8 KiB) = 24 KiB of
  32 KiB L1d.  Pass 1 cannot be fused with pass 2: the x-butterfly needs all 8 planes.
  No padding is used and none is needed — every access is a full 64-byte line at a
  64-byte-aligned address, and the working set is L1-resident, so §04 §7.3's
  "z-stride = one L1 set" pathology (which is created by a 64-byte *batch* granule, a
  layout I do not use) never arises.  Measured: no benefit was left on the table here,
  since the B=1 kernel is instruction-bound, not cache-bound (see below).
* **Non-temporal output stores for large batches.**  Every output row is 128 contiguous
  bytes written by exactly one `KILV` call at a 128-byte-aligned address, i.e. two (AVX-512:
  one each) *complete* cache lines, so `vmovntpd` needs no write-combining games and
  removes the read-for-ownership traffic: 24 KiB/volume of DRAM traffic becomes 16 KiB.
  One `sfence` per `fft3d_execute`, not per volume.  There is an alignment guard that falls
  back to ordinary stores if the buffers are ever not 64-byte aligned.
* **Self-tuning in `fft3d_create()`** over {AVX-512, AVX2, portable} × {normal, NT stores},
  timed at the *real batch size* (capped at 4096 volumes), warm-up ≈ 1 ms per candidate and
  7 interleaved rounds so an AVX-512 licence transition cannot bias whoever runs first;
  AVX-512 + normal stores wins ties (a candidate must be 2% faster to displace the
  preference).  This is the honest way to settle the "is AVX-512 a win on a downclocking
  Cascade Lake" question the brief raises: the node decides, not me.  Setup is 0.12–0.24 s.

### What was measured

**Dev machine: Haswell Xeon E5-2680 v3 @ 2.5 GHz, AVX2 only, gcc 11.4, `-O3 -march=native
-std=gnu11`.**  The backend the tuner picks locally is AVX2 (NT variant from B≈512 up).
Driver numbers (min of 20 samples), per transform:

| B | per transform | note |
|---|---|---|
| 1 | **1.279 µs** | rel_l2 = 1.25e-16 |
| 8 | 1.270 µs | rel_l2 = 1.40e-16 |
| 512 | 1.430 µs | NT stores chosen; rel_l2 = 1.37e-16 |
| 2048 | 1.376 µs | NT stores chosen; rel_l2 = 1.38e-16 |

Backend comparison on the same machine (own harness, min of 8 rounds, per transform):

| backend | B=1 | B=512 | B=2048 | B=8192 |
|---|---|---|---|---|
| AVX2, normal stores | 1.21 | 1.41 | 1.96 | 2.49 |
| AVX2, NT stores | 1.25 | 1.38 | **1.36** | **1.76** |
| portable C | 4.97 | 5.27 | 6.12 | 6.93 |

So **NT stores are worth 30–44% once the batch's footprint leaves the LLC** and nothing at
B=1, which is exactly what the create-time tuner is for.

Correctness: `check.py` passes at B = 1, 3, 8, 512, 2048 with rel_l2 = 1.25e-16 … 1.38e-16
and rel_max ≤ 2.8e-16 (tolerance 1e-12); the NT-store path is checked at B = 512 and 2048.
Separately verified: the input buffer is bit-identical after execute, and two executes on
one plan give bit-identical output, for B = 1,2,3,4,20,100,500.  The portable
(`-march=x86-64`) and `-march=sandybridge` builds are also correct (5.6 µs/transform).

**Node numbers (Xeon Gold 5218, AVX-512): not measured by me — the monitor's are the first.**
Predicted 1100–1300 cycles/volume at B=1 (0.48–0.57 µs), and at large B a floor set by
single-core read+NT-write bandwidth (16 KiB/volume).  My local memory-floor probe for the
same access pattern was 0.63 µs/volume at B=2048 and 0.92 µs at B=8192 (26 and 18 GB/s), so
on Haswell the kernel is **compute-bound even at B=2048** (1.38 µs against a 0.63 µs
memory floor) — the batched regression from 1.27 to 1.38 µs is only ~0.1 µs of memory
exposure, not a bandwidth wall.  On the node, where the compute half should fall to ~0.5 µs,
the batched case is likely to become genuinely bandwidth-bound.

### What was tried and did NOT work

* **Interleaved-complex codelet** (keep re/im in the same register, pay a permute per ×i):
  costed at 38 instructions per 4 lines = 9.5/line against split's 6.75/line, +41%.  Not
  built.  In split layout ±i is free, and radix-8 has five such sites per codelet.
* **Reordering the axis passes to save a transpose.**  Three orderings costed per plane
  (loads/stores excluded): (a) T-in → z → T-back → y = 48+54+48+54 = **204** ← chosen;
  (b) de-interleave → y → T → z = 16+54+48+54 = 172, but it leaves lanes = k1 with k2 in
  registers, so the output rows would need a stride-1024-byte scatter (8 elements/instr,
  ~8 cycles each) — dead; (c) y-DFT in the interleaved domain first, then T-in → z →
  T-back = 16+76+48+54+48 = 242.  The z axis must leave the lanes and come back; two
  transposes per element is the floor for any per-volume kernel in this layout.
* **Doing the z-DFT "vertically", inside the lanes, to avoid the transposes entirely.**
  Each of the 3 stages needs a permute to align partners, a lane-dependent sign xor, and a
  lane-dependent twiddle (blend of the swapped re/im), ≈ 14 ops per stage per line and a
  final bit-reversal permute: ~45 ops per line, against 19 ops/line for
  (2 transposes + codelet)/8 lines.  2.4× worse; not built.
* **Doing the first z-butterfly before the transpose** (it is free: z and z+4 sit at the
  same element position of the two loaded halves, so `A±B` needs no shuffle).  Correct but
  worth exactly zero — it is the same 16 add/sub either way, and the remaining DFT_4 still
  needs z in registers, so the transposes do not shrink.
* **Batch-in-lanes (8 volumes per vector, no transposes at all inside the transform).**
  Costed: the repack in/out is 8×8 transposes at 24 ops per 4 points per 8 volumes = 384
  shuffles per volume each way = 768 total, only 14% below the 896 this design pays, while
  the 8-volume split intermediate is 64 KiB and *blows L1 by 2×* between axis passes.
  Rejected on that arithmetic; not built.
* **Vector-radix / any minimum-multiplication module.**  Only 4 of the 56 flops per codelet
  are multiplies (7%); there is nothing for a multiply-saving transform to take, and it
  would add adds (§03 §3.3, §02 §2.7).
* **Compiler flag experiments** (they cannot be used — the graded build's flags are fixed —
  but they say where the AVX2 path hurts): `-fsched-pressure` 1.227 µs vs baseline
  1.288 µs (−5%, it is fighting the spills), `-fno-schedule-insns` 1.500 µs (+16%),
  `-funroll-loops` 1.470 µs (+14%), `-fno-schedule-insns2` neutral.  I did *not* pin these
  with `__attribute__((optimize(...)))`: the win is 5% on the fallback path only, and the
  attribute is known to drop command-line flags on some gcc versions, which could damage
  the AVX-512 path that actually gets graded.
* **Software prefetch of the next volume's input** was not built after the memory-floor
  probe showed only ~0.1 µs of memory exposure at B=2048 on this machine; the hardware
  streamer already covers a sequential 8 KiB-per-volume read stream.

### Next

1. **Read the monitor's node numbers first, and specifically which backend the tuner
   picked.**  That single line settles §04 §8's open question for L=8: if `avx512` wins,
   AVX-512 is worth its licence on a Gold 5218 for a shuffle-heavy L1-resident kernel; if
   `avx2` wins, the 256-bit port scheme beat it despite spilling 371 times, which would be
   a striking result and would make a **hand-written 4-lane AVX2 kernel** (8 re + 8 im ymm
   = exactly 16, half the current AVX2 pressure, at the same 0.5 shuffle-ops/double) the
   obvious next build.
2. **Cut the transpose bill with a 2-volume kernel at B≥2.**  Two volumes give 32 live
   vectors' worth of independent work; more usefully, the *pair* of planes could share one
   transpose network's index constants and give the scheduler twice the ILP against the
   4-cycle FP latency on a single-FMA part.  Cheap to try, ~5% expected.
3. **If the node's batched case turns out bandwidth-bound**, the only remaining lever is
   traffic, and it is already at the 16 KiB/volume floor (read in, NT-write out) — so the
   next thing to try is `madvise(MADV_HUGEPAGE)`-backed buffers, which §05 §7 predicts to
   matter above ~500 volumes at L=8.  We do not own the driver's buffers, so this can only
   be tested by measuring a hugepage-backed copy in a harness first.
4. **Do not** revisit the codelet arithmetic. 56 flops is proven optimal three ways, the
   instruction form is 54, and the kernel has no twiddle table to load.
