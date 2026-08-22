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

---

## Round panel_r2

### Where round 1 landed (node, panel_r1)

B=1 **0.576 µs** — a three-way tie with L8_fusedaxes (0.570) and L8_batchsimd (0.573),
all ahead of MKL (0.651).  Batched is where I lost: B=2048 **1.526 µs** vs batchsimd
1.432 and MKL 1.338; B=16384 **2.018 µs** vs batchsimd 1.782 and MKL 1.772.  The node
picked the AVX-512 backend (0.576 µs ≈ 1325 cycles at 2.3 GHz against my 1296-cycle p0
bound), which settles round 1's open question: AVX-512 is worth its licence at L=8 on the
Gold 5218.  B=1 was already at ~98% of my port-0 bound, so this round attacks the bound
itself and the batched regime.

### What changed (three edits, all borrowed — this is the cumulative round working as intended)

1. **Codelet: 54 → 52 instructions** (borrowed from **L8_batchsimd** round 1).  My round-1
   w8^{1,3} twiddle+butterfly was `p = C·c5r; d = fma/fms(C, c5i, ±p)` then 4 add/sub
   = 7 instructions per twiddle pair.  batchsimd's formulation is
   `s = c5r+c5i; t = c5i−c5r; Y1,Y5 = fma/fnma(C, s|t, c1)` = 6.  Two pairs per codelet
   ⇒ 52 = 44 add/sub + 8 FMA/FNMA.  Per volume: 24 × 52 = **1248 FP instructions**, the
   same p0 floor batchsimd quotes.  On a 1-FMA-unit node that is −48 p0 cycles/volume
   = −3.7% at B=1.  Verified the algebra term-by-term against the round-1 expressions
   before editing (d = C(1−i)c5 ⇒ dr = C(c5r+c5i), di = C(c5i−c5r); e = −C(1+i)c7 ⇒
   er = C(c7i−c7r), ei = −C(c7r+c7i)).  KMUL/KFMS primitives are gone; KFNMA added.

2. **AVX-512 transposes: copy-free, index-vector-free** (borrowed from **L8_fusedaxes**
   round 1, which measured the mechanism, via **L8_batchsimd**'s round-1 write-up of the
   same network).  My round-1 8×8 transpose used 8 `vpermt2pd` in its middle level;
   `vpermt2pd` is destructive and each source is used twice, so gcc emitted 4
   `vmovapd zmm,zmm` per transpose = **128 copies/volume**, plus 2 live index registers.
   The replacement network is `vshuff64x2 0x44/0xEE` (reg bit 2 ↔ lane bit 2), then
   `vshuff64x2 0x88/0xDD` (the 3-cycle r1→l2→l1→r1 — the trick, since a straight r1↔l1
   has no non-destructive encoding), then `vunpcklo/hi` (reg bit 0 ↔ lane bit 0):
   24 immediate-controlled non-destructive shuffles giving `o[k][l] = in[SW(l)][k]`,
   SW = (0,1,4,5,2,3,6,7), an involution.  I re-verified this against emulated intrinsic
   semantics before use (same discipline as round 1).
   **How the SW residue is absorbed in MY pass structure** (differs from both rivals):
   the first transpose (de-interleave) already yields natural register order, and the
   lane residue on y is invisible through the elementwise z-codelet; the second transpose
   writes network output k into slot SW(k) — a free renaming — so registers are natural
   again and the lanes carry k2 = SW(l) through the scratch into pass 2; the final
   interleave index vectors compose SW⁻¹ = SW: il = {0,8,1,9,4,12,5,13},
   ih = {2,10,3,11,6,14,7,15}.
   **A wrinkle worth recording:** I also killed the interleave's own copy.  Both output
   permutes read {re, im}, so whichever runs first must preserve the other's source — one
   copy per KILV (64/volume) — *unless* the two permutes destroy different sources:
   `permutex2var(re, il, im)` for the low line and `permutex2var(im, ih', re)` for the
   high line, with ih' rewritten for the swapped operand order ({10,2,11,3,14,6,15,7}).
   Assembly audit of the emitted cascadelake code: **0 `vmovapd zmm,zmm`, 0 stack
   spills/reloads, 0 `vpermt2pd` outside the 16 interleaves** — 192 copies/volume in
   round 1, 0 now.

3. **Next-volume software prefetch in the streaming regime** (borrowed from
   **L8_batchsimd** round 1).  The L2 streamer stops at 4 KiB page boundaries; an 8 KiB
   volume spans two pages, so at large B the read stream restarts twice per volume.
   Pass 1 of volume b now issues `prefetcht0` for the 128 lines of volume b+1 (16 per
   x-plane iteration), only when the tuner chose an NT-store kernel (`plan->nt`) — at
   cache-resident batch sizes the prefetches are pure overhead.  The create-time tuner
   times the NT candidates *with* the prefetch so the decision reflects what will run.
   Kernel signature gained a `pf` pointer (NULL = off; also NULL for the last volume).

### Operation count per volume (updated)

24 codelets × 52 = **1248 vector FP instructions** (p0), 896 shuffles (768 transpose +
128 interleave, p5), 512 loads/stores, **0 register copies, 0 spills** — down from
1296 FP + 896 shuffles + 192 copies in round 1.  Classic flop accounting unchanged
(10752 = 192 lines × 56).  New p0 bound: 1248 cycles ≈ **0.543 µs at 2.3 GHz**.

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, gcc 11.4, tryout.sh)

| case | round 1 | this round | delta |
|---|---|---|---|
| B=1 | 0.341 µs | **0.323 µs** | −5.3% (beats MKL's 0.330 on wallaby) |
| B=512 | 0.502 µs/vol | **0.469 µs/vol** | −6.6% |
| B=4096 | 0.507 µs/vol | **0.453 µs/vol** | −10.8% |
| B=64 | — | 0.347 µs/vol | |
| B=2048 | — | 0.885 µs/vol (sd 3.3%, straddles wallaby's L3) | |
| B=16384 | — | 0.623 µs/vol | |

Correctness: PASS at B = 1, 3, 8, 64, 512, 2048, 4096, 16384, rel_l2 = 1.31e-16 …
1.34e-16 (round 1: 1.38e-16 — the FNMA form is marginally *better* conditioned),
rel_max ≤ 2.9e-16, bit-identical re-runs everywhere.  The AVX2 path (wallaby,
`-mno-avx512f`: 0.616 µs/vol at B=8) and the portable path (`-mno-avx512f -mno-avx2`:
3.81 µs/vol) both PASS — the shared kernel body means the AVX-512 algorithm text is
exercised by every backend; only the ~10 z_* primitives are AVX-512-specific, and those
were re-verified by emulation.  Builds warning-free under `-Wall -Wextra` at
cascadelake / haswell / x86-64.

Caveat for the monitor's numbers: wallaby runs 512-bit code at full clock and has 2 MB
L2 / huge L3, so (a) the copy-removal win shows up smaller there than the model predicts
for a 1-FMA Cascade Lake (copies are rename-eliminated on real SPR hardware; on the node
they still cost front-end slots and, per llvm-mca, p0/p5 slots), and (b) the B=2048
number does not transfer at all (32 MiB vs different L3).  The prefetch win should be
*larger* on the node, whose DRAM latency is not hidden by a 2 MB L2.

### What was tried and did NOT work

* **Making the final interleave unpck-only** (2 `vunpck` instead of 2 `vpermt2pd` per
  row) by choosing the lane residue: needs lanes in perfect-shuffle order
  (0,4,1,5,2,6,3,7), but the reachable lane permutations of a 3-stage non-destructive
  network are only identity and SW = swap(l1,l2) (L8_fusedaxes round 1 proved this; I
  re-checked by enumeration while deriving the index vectors).  The perfect shuffle is
  unreachable, so 2 two-source permutes per output row is already minimal.  Not built;
  killed by the enumeration.
* **Nothing else was newly attempted this round** — the round-1 failure list (interleaved
  complex, pass reorderings, lane-axis DFT, batch-in-lanes, vector radix) still stands
  and none of the other entries' records contradict it.

### Attribution summary

52-instruction codelet: **L8_batchsimd**.  Non-destructive transpose network:
**L8_fusedaxes** (discovered), **L8_batchsimd** (also adopted it; my SW-absorption and
the copy-free interleave-source swap are new here).  Next-volume prefetch:
**L8_batchsimd**.

### Next

1. **Node numbers first.**  Predicted B=1 ≈ 0.545–0.555 µs (1248-cycle p0 bound + loads
   at the boundary).  If the node still shows ≥ 1300 cycles, the residue is front-end or
   load-port pressure — profile with llvm-mca markers per loop rather than guessing.
2. **B=16384 is the test of the prefetch.**  If it does not close most of the gap to
   batchsimd's 1.782 µs, the difference is their block structure (8 volumes in flight
   giving longer independent load streams), and the next move is a 2-volume software
   pipeline (pass 2 of volume b overlapped with pass 1 of volume b+1, double-buffered
   16 KiB scratch).
3. **B=2048 (1.5× L3)**: if MKL stays ahead there and only there, try a create-time
   *third* store variant that uses regular stores for the first ~⅓ of the batch and NT
   for the rest (the L3 can absorb part of the output); cheap to add to the tuner.
4. Still do not revisit the codelet arithmetic: 52 instructions issues the proven-minimal
   56-flop module with every ±i free; the only way below 1248 p0 instructions is a
   different factorization, and every candidate in the corpus adds shuffles.

---

## Round panel_r3

### Where round 2 landed (node, panel_r2)

B=1 0.583 µs (2nd, vs fusedaxes 0.573, batchsimd 0.598 — the third consecutive
statistical tie; the monitor's verdict declares B=1 a wall and says stop cutting
instructions).  Batched is where I lost, and badly: B=64 **0.706** vs batchsimd 0.636
(+11 %), B=2048 **1.526** vs batchsimd 1.205 (+27 %), B=16384 **1.778** vs 1.557
(+14 %).  My r2 codelet/copy-elimination work moved B=1 by −0.7 %/nothing, confirming
the verdict's reading: the p0-instruction-count model does not predict L=8 on this
node.  So this round is spent entirely on the batched regime, where the deficit is
structural, not arithmetic.

### Diagnosis (from reading L8_batchsimd's promoted code, not from guessing)

batchsimd's LANEX — which owns every batched cell — differs from my kernel in exactly
two DRAM-facing ways:

1. **Output store order.**  Its last DFT axis ends with the output written plane by
   plane, FRONT TO BACK — the whole 8 KiB volume is stored sequentially.  My 2-pass
   shape does the x axis last, so k0 (stride 1 KiB) scans the registers and the volume
   is written as 64 strided 128-B chunks.  Sequential vs strided makes no difference
   in L1 (B=1 was always fine) and a large one to the DRAM/NT write stream.
2. **Prefetch policy.**  Its `lanex_run` prefetches the next volume whenever
   `nvol > 1`, with regular stores too; mine was gated on NT.  Its node B=64 win
   (0.636, prefetch on, NT off) says prefetch+regular-stores is right on the node.

### What changed (all of it aimed at batch)

1. **A second kernel shape, "3-pass", with sequential output stores** (idea borrowed
   from **L8_batchsimd**'s LANEX pass structure).  Pass 1 per x-plane: transpose-in,
   z-DFT, store scratch `[x][k2]` (lanes = y).  Pass 2 per k2: x-DFT **in place in the
   scratch column** (reads scr[x][k2], writes scr[k0][k2] — the same 16 slots, legal
   because the codelet consumes all inputs in stage 1; zero shuffles).  Pass 3 per k0:
   transpose (y into registers), y-DFT, interleave, store the contiguous 1-KiB output
   plane — k0 ascending, so the volume is written sequentially.  Operation count is
   IDENTICAL to the 2-pass shape (1248 FP, 896 shuffles: pass 1's second transpose
   moves to pass 3); the cost is +128 loads +128 stores per volume, all L1-resident.
   The AVX-512 lane-residue algebra needed no new pieces: pass 1 uses z_tr8a (lanes
   become SW(y), invisible to the elementwise z- and x-codelets), pass 3 uses z_tr8b
   (registers renamed natural, lanes carry SW(k2)) and the existing z_ilv index vectors
   already compose SW⁻¹.  Verified by the same lane-tracking derivation as r2, then
   numerically on all three backends.
2. **Prefetch untied from NT** (borrowed from **L8_batchsimd**'s lanex_run rule):
   (shape, nt, pf) are now independent tuner dimensions; candidates are
   2p / 3p / 3p-pf / 3p-nt / 3p-nt-pf on the best available ISA.  Hint stays t0
   distance 1 volume (node-validated by batchsimd's r2 1.205; fusedaxes measured t1
   better on wallaby but t0 on CLX — do not chase the wallaby-only inversion).
3. **Scratch plane stride padded 128 → 144 doubles** (`-DL8R_SCRX=128` to revert):
   the pass-2 column accesses at 1-KiB stride land on only 4 L1 sets dense; at
   1152 B they spread over 8.  This is batchsimd's LPZ=9 padding and §4.5's question.
   Wallaby A/B at B=16384 was inconclusive (padded 10.22/11.04/10.80 ms vs dense
   10.74/11.14/10.77 — inside the machine-state noise), but it is free, and the
   set-conflict argument targets the node's 32-KiB L1, not wallaby's 48-KiB.
   **Monitor: one-flag A/B `-DL8R_SCRX=128` at B=2048/16384 would settle §4.5 here.**
4. **The tuner's pick is reported through `fft3d_description()`** (the panel_r2
   verdict's cross-cutting request #2): the string ends
   `tuner pick[B=<batch>]=<candidate>`.  Also added `L8R_TUNE_DEBUG=1` (prints all
   candidate times per create) and `L8R_FORCE=<name>` (create-time override for A/B
   runs).  Execute stays branch-free and repeatable; both hooks act at plan time only.
5. B=1 keeps the 2-pass shape as the default candidate, but the tuner decides: on
   wallaby the 3-pass actually wins B=1 too (0.6027 vs 0.6303 same-state in-tuner,
   and 0.309 vs r2's 0.323 across runs), because the extra 256 L1 accesses cost less
   than the better load/store scheduling of three short balanced passes.

### Operation count per volume (3-pass shape)

1248 vector FP (24 × 52, unchanged), 896 shuffles (2 × 24 per plane in pass 1 +
2 × 24 in pass 3 + 128 interleave, unchanged), 384 loads + 384 stores (was 256+256),
0 copies, 0 spills.  Classic flop count still 10752 = 192 lines × 56.  The extra 256
accesses are to a 9-KiB L1-resident scratch; DRAM-facing traffic is unchanged at
8 KiB read + 8 KiB written per volume (16 KiB with NT), but the write stream is now
sequential.

### What was measured (wallaby, Gold 6448Y SPR, min over ≥3 runs, per transform)

| B | r2 best | this round | tuner pick | vs batchsimd same session |
|---|---|---|---|---|
| 1     | 0.323 µs | **0.309 µs** | 3p (beat 2p in-tuner 0.603 vs 0.630) | — |
| 8     | 0.311    | **0.314**    | 3p-pf | — |
| 64    | 0.347    | **0.320**    | 3p-pf (in-tuner: 3p-pf 0.606 vs 2p 0.671) | — |
| 512   | 0.469    | **0.473**    | 3p-nt-pf | — |
| 2048  | 0.885 (straddles L3) | **0.474** | 3p-nt-pf | — |
| 5632 (node-B=2048 analog, WS 1.53×L3) | — | **0.476** | 3p-nt-pf | batchsimd 0.632 → **I lead 25 %** |
| 16384 | 0.623 | 0.638 (best 10.45 ms) | 3p-nt-pf | batchsimd 0.59 → **I trail 8–10 %** |

In-tuner candidate tables (same process, so drift-free): B=5632 arena
2p 1.259 / 3p 1.019 / 3p-pf 0.933 / 3p-nt 0.581 / **3p-nt-pf 0.474** — the
sequential-store NT path is 2× the non-NT path there, and prefetch is worth 18 % on
top of NT.  Correctness: PASS at B = 1, 3, 8, 64, 512, 2048, 5632, 16384,
rel_l2 = 1.87–1.92e-16 (the axis order changed z,y,x → z,x,y, which moves the last
digit exactly as batchsimd's r1 record predicts; 4 orders inside tolerance),
rel_max ≤ 3.1e-16, bit-identical re-runs everywhere.  AVX2 path (`-mno-avx512f`)
and portable path (`-mno-avx512f -mno-avx2 -mno-fma`) both PASS; builds clean
under `-Wall -Wextra`.

### What was tried and did NOT work

* **Scratch padding as a wallaby-measurable win**: see above — indistinguishable from
  machine-state noise at B=16384 (three alternating pairs).  Kept on the set-conflict
  argument, flagged for a node A/B.  Do not claim §4.5 answered by this.
* **Closing the B=16384 gap to batchsimd on wallaby.**  After the restructure my
  kernel is pass-for-pass isomorphic to their LANEX (same 1248 FP + 896 shuffles +
  384+384 L1 accesses per volume, both sequential-in sequential-out NT+t0-pf), yet
  three alternating pairs show them stably ~8–10 % ahead there (9.41–9.78 ms vs
  10.22–11.10 ms), while I lead by 25 % at B=5632.  Remaining structural differences
  I could find by reading their code: their per-volume loop lives inside the kernel
  translation unit (direct call, inlinable) where mine goes through a function
  pointer per volume; and their scratch is split-planar (SR ‖ SI) where mine
  interleaves re/im blocks per plane.  Neither is obviously worth 8 %.  Not resolved
  this round; the node may not reproduce it (wallaby's B=16384 is 4× its L3, the
  node's is 11×).
* Nothing else new was attempted; the r1/r2 failure lists (interleaved complex, pass
  reorderings, lane-axis DFT, batch-in-lanes, vector radix, perfect-shuffle lane
  order) all still stand and no other entry's record contradicts them.

### Attribution summary

Sequential-store pass structure and the prefetch-without-NT rule: **L8_batchsimd**
(its LANEX code and node numbers, read directly).  Scratch padding: **L8_batchsimd**
(LPZ=9) via the corpus §04 §7.3.  Tuner-pick reporting: the **panel_r2 VERDICT**'s
cross-cutting request.  The 3-pass in-place pass 2 (x-DFT overwriting its own scratch
column) and the SW-residue bookkeeping across the split passes are mine.

### Next

1. **Read the node's pick strings off the leaderboard** — every case now reports
   `tuner pick[B=…]`.  Expected: 2p or 3p at B=1 (either is fine, it is measured),
   3p-pf at B=64, 3p-nt-pf at B=2048/16384.  Predictions: B=64 recovers to
   ≤0.64 µs (the +3.5 % r2 regression was the missing prefetch), B=2048 lands
   1.20–1.35 µs (sequential NT writes + L3-resident input), B=16384 1.55–1.75 µs.
   B=1: unchanged-to-slightly-better; on a 1-FMA part the 2p/3p choice is the
   tuner's to make and both shapes are compiled.
2. **Ask the monitor for `-DL8R_SCRX=128` at B=2048/16384** — one flag, settles
   §4.5 for this entry on the machine where the L1 argument actually applies.
3. **If batchsimd still leads B=16384 on the node**, test the two remaining
   differences one at a time: move the volume loop inside the kernel TU (kill the
   per-volume indirect call), then split-planar scratch.  Each is a small, separately
  measurable edit.
4. Still do not touch the codelet: three entries and two rounds of node data say
   B=1 is not instruction-bound; the 52-instruction module stays.

---

## Round panel_r4

### Where round 3 landed (node, panel_r3)

**Won B=2048 outright: 1.243 µs** (batchsimd 1.283, fusedaxes 1.291, MKL 1.335) — the
3-pass sequential-store restructure did what it was built to do.  B=16384 **1.647 —
second** (fusedaxes 1.580, floor ≈1.365 at the node's 12 GB/s).  B=64 **0.671 —
third** (fusedaxes 0.642).  B=1 0.583 — third in the standing three-way tie.  The
node's pick strings (read from the r3 `t_*.json`): **2p at B=1 (3/3)**, 3p/3p-pf
flip-flopping at B=64, **3p-pf — regular stores, NOT NT — at B=2048 and B=16384
(3/3 each)**.  Two diagnoses fall straight out of those strings plus the VERDICT:

1. **Tuner instability cost real time at B=64**: run 1 picked 3p (0.671, the cell
   min), runs 2–3 picked 3p-pf (0.716, 0.715) — a +6.7 % coin-flip that the VERDICT
   called out by name.  Prefetch at a cache-resident batch size is a *documented*
   loser (fusedaxes r2: prefetching L3-resident input = 1.6× disaster), and my tuner
   kept offering it there anyway, letting arena noise ship it.
2. **My NT candidates lost on the node even at B=16384** (ws = 11.6×L3!), where
   fusedaxes WON the cell with NT+pf_t0 — and their r2 record documents the exact
   mechanism that separates us: their prefetches are embedded in compute, spread
   through the volume; mine were a 128-line burst in pass 1, and **a prefetch burst
   and the NT store drain fight for the same ~12 fill buffers**.  So it was not
   "NT loses on this node" (my r3 reading); it was "NT loses *with a prefetch
   burst*".

### What changed (three edits)

1. **Spread prefetch placement** (borrowed from **L8_fusedaxes**, whose
   compute-embedded prefetch + NT owns B=16384): new kernel instantiations issue the
   same 128 lines of volume b+1 as 6/5/5 lines per iteration of passes 1/2/3
   (48+40+40, each 64-B line exactly once) instead of 16 per pass-1 iteration —
   ~1 prefetch per 11 cycles, never competing with the pass-1 demand-load burst or
   the NT drain.  Burst placement is kept for the plain-store path (it is what won
   B=2048 on the node, 3/3 picks).  The burst+NT combination is deliberately no
   longer a candidate — that is the documented fill-buffer clog.  Candidates now:
   3p-pf (burst, plain), 3p-pfs (spread, plain), 3p-nt-pfs (spread, NT), 3p-nt, 3p.
2. **Regime-gated candidate sets** (the VERDICT's tuner-instability ask, plus
   fusedaxes' L3-relative gating): pf/NT candidates are offered **only when
   in+out > 0.9×L3** (`sysconf(_SC_LEVEL3_CACHE_SIZE)`, fallback 22 MiB).  Below the
   gate the set is just {3p, 2p} (B>1) or {2p, 3p} (B=1); at B=64 the pf coin-flip
   is now structurally impossible.  The FIRST candidate is the node-validated
   default for the regime (2p at B=1, 3p small-batch, 3p-pf big — the r3 pick
   strings), and displacing it requires a >2 % argmin win.  No scored node cell sits
   in the excluded 0.25–0.9×L3 band (B=64 = 0.045×L3, B=2048 = 1.45×L3), so the
   gate cannot mis-regime a scored cell.  `L8R_ALLCAND=1` un-gates for dev A/Bs.
3. **Per-candidate run functions** (r3 "Next" item 3, closing the last structural
   difference to L8_batchsimd's `lanex_run`): the batch loop now lives in a
   `DEFRUN`-generated function per candidate, kernel `always_inline`d, prefetch
   flag a compile-time constant — one indirect call per *execute* instead of one
   per volume, and no per-volume `(pf && b+1<nb)` branch.  The tuner times these
   exact run functions, so what is measured is what ships.  Tuner protocol also
   adopts fusedaxes r3's **one untimed pass before each timed block** (a candidate
   is measured against its own cache state — plain and NT leave different L3
   contents) and their machine-relative arena cap (4×L3 of volumes, clamped to
   [4096, 8192]; 5632 on the node, so the B=16384 arena is a faithful 4×L3 stream).

Arithmetic, transposes, scratch padding (SCRX=144), t0 hint, and both kernel shapes
are untouched: 1248 vector FP + 896 shuffles per volume, rel_l2 unchanged.

### Operation count per volume

Unchanged from r3 (3-pass: 1248 FP, 896 shuffles, 384+384 L1 accesses, 0 copies,
0 spills; 2-pass: 256+256).  The spread-pf kernels issue the identical 128
`prefetcht0` uops, just distributed; run functions delete ~5 instructions of
call/branch overhead per volume.  DRAM-facing traffic unchanged.

### What was measured (wallaby, Gold 6448Y SPR, tryout.sh, best of ≥3 runs)

Wallaby was strongly bimodal again this session (B=64 38.8 → 19.8 µs across
back-to-back runs, states ~2× apart); every number below is best-of-runs, sd within
a run ≤0.7 %.

| B | r3 best | this round | pick (in-tuner table) |
|---|---|---|---|
| 1     | 0.309 µs | **0.308 µs** | 3p, tuned (0.601 vs 2p 0.633 in-tuner; node default stays 2p) |
| 64    | 0.320    | **0.310**    | 3p default, **stable by construction** (pf not offered) |
| 2048  | 0.474    | 0.502        | 3p (wallaby gate: ws=0.53×L3 → small set; NODE gets the big set here) |
| 5632 (node-B2048 analog) | 0.476 | **0.436** | 3p-nt-pfs (in-tuner: nt-pfs 0.490, nt 0.570, pfs 0.948, pf 0.965, 3p 0.960) |
| 16384 | 0.638    | **0.575**    | 3p-nt-pfs (in-tuner: nt-pfs 0.495, nt 0.570, plain ≈1.03–1.07) |

**Spread prefetch is worth 13–14 % on top of NT** (0.490 vs 0.570 at B=5632,
0.495 vs 0.570 at B=16384, in-tuner same-process) — that is the fill-buffer
mechanism, measured in isolation for the first time.  The r3 wallaby B=16384 gap to
batchsimd (~8–10 %) is closed: 9.43 ms best vs their r3-session 9.41–9.78 ms.
Correctness: PASS at B = 1, 3, 64, 2048, 5632, 16384, rel_l2 = 1.32e-16 … 1.92e-16,
rel_max ≤ 2.9e-16, bit-identical re-runs everywhere, including the NT+spread path at
B=5632/16384 and the alignment-fallback twins.  AVX2 path (`-mno-avx512f`, incl.
NT+spread at B=5632) and portable path (`-mno-avx512f -mno-avx2 -mno-fma`) PASS.
Builds warning-free under `-Wall -Wextra` at cascadelake / haswell / x86-64.
Setup grew 0.39 → 0.55–0.82 s at the largest batches (arena 4×L3 + state passes);
plan time is unscored but noted per the VERDICT's watch item.

### What was tried and did NOT work

* **Wallaby's B=2048 regressed 0.474 → 0.502 by design**: the 0.9×L3 gate puts
  wallaby's B=2048 (ws = 0.53×L3) in the small set, and NT genuinely wins there on
  wallaby (fusedaxes r3 measured the same inversion and killed their 0.55×L3 gate
  for it).  I kept the 0.9 gate anyway because on the *node* no scored cell sits in
  the 0.25–0.9×L3 band, and the gate is what buys the B=64 stability.  If the
  driver ever scores a cell in that band, lower the gate to 0.25×L3 and re-check
  B=64 stability.  Recorded so the next round does not read the wallaby 2048 number
  as a regression.
* Nothing else new failed; the r1–r3 failure lists all stand.

### Attribution summary

Spread/compute-embedded prefetch placement and its fill-buffer rationale, the
L3-relative gating idea, the own-cache-state untimed pass, and the 4×L3 arena cap:
**L8_fusedaxes** (r2 failure 4, r3 protocol).  Run-function/direct-call structure:
**L8_batchsimd**'s `lanex_run` (via my r3 "Next" list).  Regime-default-first with
2 % displacement hysteresis: the **panel_r3 VERDICT**'s tuner-instability finding.
The burst-vs-spread split by store type (burst kept for plain stores, spread
required for NT) is mine, from reading my own r3 node pick strings against
fusedaxes'.

### Node predictions (stated to be scored)

* **B=1: 0.583 µs, unchanged** — default 2p, same kernel; 3p would need a >2 % win
  to displace, which three rounds of node data say it does not have.
* **B=64: 0.67 µs median-stable** (the min was already 0.671; the fix is that 2/3
  runs no longer ship 0.716).  Any improvement beyond that is the direct-call gain,
  ≤2 %.
* **B=2048: 1.20–1.25 µs.**  Default 3p-pf (the 1.243 winner) is protected by
  hysteresis; 3p-nt-pfs is the live challenger and wins only if it beats 1.243 by
  2 % on the node itself.
* **B=16384: 1.55–1.62 µs** if the spread fix transfers (fusedaxes' NT+pf_t0 =
  1.580 is the existence proof on this exact node and cell); 1.647 (the r3 number)
  if the node's tuner keeps plain 3p-pf.  The pick string will say which.

### Next

1. **Read the node's pick at B=16384 first.**  If it is 3p-nt-pfs and the time is
   ≈1.58, the fill-buffer story is confirmed on CLX and the remaining gap to the
   1.365 bandwidth floor (~14 %) is read-side scheduling — then try issuing the
   spread prefetches from pass 2 only (pure L1 compute, load ports idlest) as a
   sixth placement, or a 2-volume software pipeline.  If it stays 3p-pf/1.647, the
   clog theory is wrong on CLX and the honest next step is `L8R_ALLCAND` forced-run
   A/Bs on the node.
2. **The `-DL8R_SCRX=128` node A/B is still outstanding** (asked in r3, seconded by
   the VERDICT §5, not run).  One flag, B=2048/16384, settles LITERATURE §4.5.
3. **B=64's remaining 4 % to fusedaxes (0.642)** is structural (their fused 2-phase
   does 256+256 L1 accesses vs my 384+384, zero-shuffle y/x DFTs): not reachable by
   tuning.  Closing it means adopting their lane-in-z fused shape as a third kernel
   — a full rewrite, only worth it if the direct-call gain does not already close
   most of the gap.
4. Still do not touch the codelet or the transpose networks.

---

## Round panel_r5

### Where round 4 landed (node, panel_r4)

The round's cleanest result, per the VERDICT: **won B=2048 outright at 1.136 µs**
(batchsimd 1.215, fusedaxes 1.313, best library 1.324) and **won B=16384 at 1.418**
(fusedaxes 1.585, batchsimd 1.642, mkl2026 1.804) — both all-time cell bests, both from
the spread-prefetch adoption; node pick avx512-3p-pfs 3/3 in both cells (plain stores +
spread t0; NT lost again).  **B=1 0.570 — tied first** with batchsimd to the last digit.
**B=64 0.680 — third** (fusedaxes 0.623, batchsimd 0.665), and the pick strings show
something useful: run 1 tuned **2p** in and produced the cell min 0.680, runs 2–3
shipped the 3p default at 0.686/0.691.  So 2p > 3p at node B=64, and the VERDICT names
the L=8 priority: attack the B=64 "L2 cliff" (ws = 1.00 MiB = exactly the node's
per-core L2; every entry is slower there than at B=1).

### What changed (all of it aimed at B=64; the winning B=1 and streaming paths are untouched)

1. **A third kernel shape, "1f" — L8_fusedaxes' fused single-sweep, ported into this
   file** (attribution: the phase structure, the T1/T2/T3 non-destructive lane
   primitives, the trans8 network order, the PI/piinv lane bookkeeping, and the
   out_off store table are all theirs, from their r1 record and code; their 0.623 µs
   at node B=64, three rounds running, is the existence proof this round acts on).
   Phase A per x-plane: deinterleave each z-pencil with one vunpcklo/hi pair (16
   shuffles per plane, lane l = z = PI[l], PI = 0,4,1,5,2,6,3,7), radix-8 along y
   elementwise, store split-planar scratch sr[k1][x] / si[k1][x].  Phase B per k1:
   16 contiguous loads, radix-8 along x elementwise, trans8 pair (T2 0x44/0xEE, T3
   0x88/0xDD, T1 unpck — the same primitive vocabulary my copy-free transposes
   already use, so the port is 3 macros), register rename through PI⁻¹, radix-8
   along z, then their 48-op fused untranspose+interleave over all 16 registers and
   16 half-pencil stores through the f_off table.  **The codelet inside is my
   52-instruction RADIX8** (natural-in/natural-out like their dft8s, so it drops in
   unchanged and keeps my 1248-FP bill vs their 1296).  Same totals as my 2p
   otherwise: 896 shuffles, 256+256 loads/stores.  The structural difference is
   *where the shuffles sit*: my 2p puts 96 transpose shuffles in pass 1 right next
   to the input demand-loads (L2 misses at B=64); 1f's phase A is only 16 shuffles
   per plane and the heavy networks run in phase B against L1-resident scratch.
2. **Spread-t0-prefetch variants offered in the mid regime for the first time.**
   New gate structure: B=1 → {2p, 3p, 1f}; mid (B>1, in+out ≤ 0.9·L3) →
   {2p, 1f, 1f-pfs, 2p-pfs, 3p} with **2p now the default** (node r4: 2p 0.680 beat
   3p 0.686–0.691); big (> 0.9·L3) → unchanged {3p-pf, 3p-pfs, 3p-nt-pfs, 3p-nt, 3p}.
   The documented B=64 prefetch losers were the **burst** placement (node r3,
   +6.7 %) and the prefetch-vs-NT-drain clog (fusedaxes r2, 1.6×) — neither is
   spread+plain-stores, which was simply never offered below 0.9·L3.  New spread
   placements: 2p and 1f issue 8 lines of volume b+1 per iteration of each of
   their 2×8 iterations (KPF1_SPREAD2/KPF2_SPREAD2 / inline in kernel1f); the
   KERNEL2_BODY prefetch is now hook-parametrized like KERNEL3_BODY's.

### Operation count per volume (1f shape)

1248 vector FP (24 × 52, my codelet), 896 shuffles (128 deinterleave + 8×(48
transpose + 48 fused untranspose/interleave)), 256 loads + 256 stores, 0 stack
spills (asm audit of every avx512 run function: 0 vmov to rsp/rbp; 12 loop-edge
zmm copies in run_1f vs 21–23 in the others).  2p/3p bills unchanged.  Scratch:
1f uses 1024 doubles of the existing 1152-double allocation (split-planar
sr‖si), no new memory.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4, tryout.sh; the known ~2×
bimodality was present — in-tuner tables are same-process and drift-free)

| B | r4 best | this round | pick | note |
|---|---|---|---|---|
| 1 | 0.308 µs | **0.308 µs** | 3p (wallaby) 3/3, both states; in-tuner 3p 0.602 < 2p 0.633 < 1f 0.657 | 1f is 5–9 % behind at B=1 here — no instability risk |
| 64 | 0.310 | **0.310** (fast state) | 3p (wallaby; ws=0.5×L2 — resident) | in-tuner: 3p 0.605, 2p-pfs 0.633, 2p 0.647, 1f-pfs 0.658, 1f 0.681 |
| 128 (**node-B=64 analog**: ws = wallaby's 2 MB L2) | — | pfs wins 3/3 | 2p-pfs/1f-pfs (flip, ≤1 % apart) | **spread pf worth 10–13 % over the plain twins** (0.653–0.667 vs 0.713–0.750) |
| 256 (2× wallaby L2) | — | 1f-pfs 3/3 | 1f-pfs | 1f-pfs 0.439–0.448 vs 1f 0.502–0.512: **12–17 %**; beats 2p-pfs in 3/3 tables |
| 5632 (node-B2048 analog) | 0.436 | **0.432** | 3p-nt-pfs | unchanged code path |
| 16384 | 0.575 | 0.604 | 3p-nt-pfs | unchanged code path; day drift, sd 0.93 % |

The B=128/256 tables are the round's finding: **once the batch's working set reaches
the L2 boundary, spread t0 prefetch of the next volume's input is worth 10–17 %
with plain stores** — the same fill-buffer-friendly placement that won the DRAM
cells in r4, now measured in the L2/L3-latency regime nobody had offered it in.
That is exactly the regime the node's B=64 cell sits in (ws = its 1 MiB L2).

Correctness: PASS at B = 1, 3, 8, 64, 128, 256, 5632, 16384.  Every mid-regime
candidate force-run via L8R_FORCE and checked against numpy individually at
B = 3/8/64: 1f and 1f-pfs rel_l2 = 2.27e-16 (the y,x,z axis order — the same
last-digit signature as L8_fusedaxes' entry, as batchsimd's axis-order note
predicts), 2p/2p-pfs 1.34e-16, 3p 1.9e-16; prefetch twins bit-match their plain
kernels; bit-identical re-runs everywhere.  AVX2 (`-mno-avx512f`) and portable
(`-mno-avx512f -mno-avx2 -mno-fma`) builds PASS (1f is AVX-512-only and simply
absent there).  Warning-free under `-Wall -Wextra` at native and cascadelake.

### What was tried and did NOT work

* **1f as a wallaby win.**  On wallaby it loses to 3p at B=1 (0.657 vs 0.602) and
  at L2-resident B=64 (0.681 vs 0.605), and only pulls ahead of 2p once the working
  set passes L2 (B=256: 1f-pfs < 2p-pfs 3/3, by ~2 %).  This is the same
  machine-inversion fusedaxes documented from the other side (their shape wins CLX
  cells while wallaby prefers other shapes), so wallaby cannot validate the port's
  node value — only the node tuner can.  Recorded so the next round does not read
  the wallaby tables as "1f failed".
* **Nothing else new failed**; the r1–r4 failure lists all stand.  The B=64
  pick-flip between 1f-pfs and 2p-pfs at wallaby-B=128 costs ≤1 % (they are twins
  in that regime) — accepted rather than fought with more hysteresis.

### Attribution summary

Fused single-sweep shape, T-primitive networks, PI/piinv bookkeeping, out_off
store table: **L8_fusedaxes** (r1 record + code, read directly).  The 2p-default
flip at mid regime: the **panel_r4 node pick strings**.  Spread placement in the
mid regime: my own r4 result transplanted; the burst-vs-spread distinction is
**L8_fusedaxes r2**'s fill-buffer analysis.  Keeping my 52-instr codelet inside
their shape (1248 vs 1296 FP): mine.

### Node predictions (stated to be scored)

* **B=1: 0.570, unchanged** — default 2p; 3p and 1f need >2 % to displace and three
  rounds of node data say they do not have it.
* **B=64: 0.60–0.65, pick = 1f-pfs or 2p-pfs.**  Decomposition: 2p is already
  0.680; spread pf was worth 10–13 % at the wallaby L2 boundary and the node's
  B=64 has *more* latency to hide (1 MB L2, older core); 1f adds fusedaxes'
  node-proven shape (0.623 plain) with 48 fewer FP instructions.  If only the
  pf mechanism transfers: ~0.62.  If both transfer: below 0.60, which would take
  the cell.  If the node keeps plain 2p, the mid-regime experiment failed and the
  next lever is serpentine batch traversal (see Next).
* **B=2048: 1.13–1.15** and **B=16384: 1.40–1.45** — byte-identical candidate
  sets and defaults to r4's winners.

### Next

1. **Read the node's B=64 pick string first.**  1f-pfs vs 2p-pfs decides whether
   the shape or the prefetch carried the cell; plain 2p means neither transferred
   and the residual is not latency-shaped.
2. **If B=64 lands ≥0.65**: the remaining idea in the VERDICT's direction is
   traversal order — the driver re-executes the same 1 MiB working set, and a
   forward stream through an exactly-L2-sized set is LRU's worst case; a
   serpentine (alternating-direction) batch order across executes would keep the
   turn-end volumes hot.  It is repeatable (bit-identical output) but stateful
   across calls; ask the monitor whether a per-execute direction flip is within
   the rules before building it.
3. **The `-DL8R_SCRX=128` node A/B is still outstanding** (asked in r3 and r4;
   settles LITERATURE §4.5 for the 3p scratch).  1f does not use SCRX.
4. Still do not touch the codelet or the transpose networks; B=1 remains frozen
   pending the monitor's clock measurement.

---

## Round panel_r7

(There is no panel_r6 entry: round 6 was abandoned before its timing pass — see
`results/panel_r6_abandoned_no_timing/WHY.md` — and this entry's round-6 file is
byte-identical to its r5 file, so panel_r5's leaderboard is still the latest node data.)

### Where round 5 landed (node, panel_r5)

B=1 **0.570 — first** (fusedaxes 0.573, batchsimd 0.574; five rounds of statistical tie).
Everything else was a lead change against me: **L8_fusedaxes took all three batched
cells** — B=64 0.594 (me 0.619, 3rd→2nd… pick `avx512-1f-pfs` 3/3), B=2048 **0.910** (me
1.116, my own numbers improved 1.136→1.116 but I fell 1st→3rd), B=16384 **1.254** (me
1.402).  Their winning pick in both streaming cells was **`fused+pfs+pfw`** — spread t0
input prefetch *plus write-intent prefetch (`prefetchw`) of the next volume's output
lines*.  The panel_r5 VERDICT (§4.5) generalizes it: on this node, **hide the RFO
(prefetchw), do not avoid it (NT)** — NT stores lost every streaming cell for the fourth
consecutive round.  The VERDICT's stated L=8 priority (§6): **separate fusion from pfw**
— fusedaxes changed shape and store policy in one round, so nobody knows which variable
earned the 0.910; it explicitly asks for a pfw candidate on my `3p-pfs`.  Also settled:
**clk512 = 2.89 GHz** (4 of 5 probes agree), so B=1's 0.570 µs = 1648 cycles against my
1248-cycle p0 floor = 1.32× — headroom nominally exists but five rounds of tied entries
say it is not reachable by instruction counting; B=1 stays frozen.

### What changed (one mechanism adopted, one retirement, one default flip)

1. **Write-intent prefetch (`pfw`) of the next volume's output** (borrowed from
   **L8_fusedaxes**' panel_r5 node win, mechanism originally **L6_unrolled**/**L36_pfa**):
   `__builtin_prefetch(p, 1, 3)` (emits `prefetchw` on CLX) issued for the 128 output
   lines of volume b+1 at exactly the same spread cadence as the existing t0 input
   prefetches, mirrored onto `out + NDBL` (valid whenever the input pf pointer is
   non-NULL, since in and out advance together).  Two new kernels: `kernel3sw_avx512`
   (3-pass, 6/5/5 in+out lines per iteration of passes 1/2/3) and `kernel1fw_avx512`
   (fused, 8+8 lines per iteration of each phase; the 1f body is now a two-hook macro,
   `KERNEL1F_BODY(PFA, PFB)`, instantiated twice).  New candidates `1f-pfs-pfw` and
   `3p-pfs-pfw`.
2. **Streaming candidate set rebuilt; NT retired.**  New big-regime set (ws > 0.9·L3):
   `{1f-pfs-pfw (default), 3p-pfs-pfw, 3p-pfs, 1f-pfs, 3p-pf}`.  The default is the
   family that won both node streaming cells in r5; having {1f, 3p} × {pfw, no-pfw} in
   one same-process tuner table **is** the VERDICT's fusion-vs-pfw isolating experiment.
   The NT candidates (3p-nt, 3p-nt-pfs) are no longer offered anywhere (kernels stay
   compiled); four rounds of node tournaments never picked them.
3. **Mid-regime default flipped 2p → 1f-pfs** (the node's 3/3 tuned pick at B=64 in r5,
   0.619): the pick the node keeps making now ships without tuner-noise exposure.  pfw is
   NOT offered in the mid regime — prefetchw on cache-resident output is a documented
   loser (L36_pfa +13 %/+11 % in-arena at B=1/B=4; fusedaxes +3 % at an L3-resident
   wallaby cell).  B=1 set and code path are byte-identical to r5.

### Operation count per volume

Arithmetic unchanged (1248 vector FP + 896 shuffles; 3p 384+384 / 1f 256+256 L1
accesses; 0 copies, 0 spills).  The pfw variants add exactly 128 `prefetchw` uops per
volume (ports 2/3, and prefetchw dispatches like a store-address uop so port 7 is also
available — fusedaxes r5's note); DRAM traffic is unchanged in volume but the store
stream's RFO reads are issued one volume early instead of at store-retire time.

### What was measured (wallaby, Gold 6448Y SPR; the machine sat in its ~1.9×-slow clock
state for most of this session — per the panel_r5 VERDICT's clock-lottery finding, only
same-process in-tuner tables are quoted for comparisons; driver numbers are given for
correctness context only)

In-tuner tables (same process, drift-free), per transform:

| B (regime) | 1f-pfs-pfw | 3p-pfs-pfw | 3p-pfs | 1f-pfs | 3p-pf | pick |
|---|---|---|---|---|---|---|
| 5632 (1.5× wallaby L3) | 0.6164 | **0.6030** | 0.8543 | 0.9538 | 0.8365 | 3p-pfs-pfw |
| 16384 (4.4× L3) | **0.7405** | 0.7375 | 1.0248 | 1.1898 | 0.9807 | 1f-pfs-pfw (hysteresis; twins 0.4 % apart) |

**The round's finding: pfw is worth −27 % to −38 % over its no-pfw twin at streaming
sizes** (3p: 0.854→0.603 at B=5632, 1.025→0.738 at B=16384; 1f: 0.954→0.616,
1.190→0.741), and **the isolating experiment answers "pfw, not fusion"**: with pfw on
both shapes, 3p and 1f are statistical twins (≤2 % apart, either side), while 1f
*without* pfw is the worst candidate in both tables.  Fusion did not carry fusedaxes'
0.910; the RFO hiding did.  (Wallaby caveat: the same tables said NT wins on wallaby in
r4–r5 while the node disagreed — the node's tuner decides, but the *ratio* pfw/no-pfw is
same-process and machine-consistent with fusedaxes' node result.)

Mid/B=1 tables this session (slow state): B=1 `3p 0.602 < 2p 0.634 < 1f 0.656` (wallaby
prefers 3p as in every prior round; node default stays 2p); B=64 `3p 0.605 < 2p-pfs
0.634 < 2p 0.647 < 1f-pfs 0.659 < 1f 0.682` — the known wallaby/node inversion at this
cell (node r5 picked 1f-pfs 3/3 at 0.619), recorded so nobody reads it as "the default
flip failed"; B=2048 (wallaby mid regime, ws=0.53×its L3) picked 2p-pfs 0.472.

Driver runs (slow-state absolute values): B=1 0.308 µs, B=8 0.311, B=64 0.604,
B=5632 0.595, B=16384 0.918.  Correctness: **PASS at B = 1, 8, 64, 2048, 4096, 5632,
16384** (rel_l2 1.87e-16 … 2.27e-16, rel_max ≤ 3.4e-16, tol 1e-12), bit-identical
re-runs everywhere.  Each new streaming candidate was additionally force-run via
`L8R_FORCE` at B=4096 and checked against numpy individually, and **each pfw kernel's
output is bit-identical (`cmp`) to its plain twin's** — the prefetch is provably a pure
hint.  Fingerprints as expected (1f family 2.273e-16 = fusedaxes' y,x,z order; 3p family
1.915e-16).  Builds warning-free under `-Wall -Wextra` at native (SPR), `-mno-avx512f`,
`-march=x86-64`, and `-march=cascadelake` (the node's arch, run-verified on wallaby);
AVX2 and portable backends PASS at B=64.

### What was tried and did NOT work

* Nothing new failed this round — the change is a targeted adoption of a node-proven
  mechanism plus set hygiene.  The r1–r5 failure lists all stand.
* Standing wallaby traps re-confirmed: the whole session ran in the slow clock state
  (driver B=64 read 0.604 vs 0.310 fast-state in r5 — never compare across windows), and
  wallaby's B=64 tuner still inverts the node's shape preference.
* The 1f-pfs-pfw / 3p-pfs-pfw pick may flip between node runs (they sit inside the 2 %
  hysteresis band on wallaby).  Accepted, not fought: they are twins where it matters,
  and whichever string the node reports answers the fusion-vs-pfw question — both carry
  pfw.

### Attribution summary

Write-intent prefetch of the output stream, its spread pairing, and the node evidence
that made adopting it obligatory (fused+pfs+pfw, 0.910/1.254, 3/3): **L8_fusedaxes**
(panel_r5), tracing to **L6_unrolled**'s `fused_pfw` and **L36_pfa**'s `pf=2`.  NT
retirement and the "hide the RFO, don't avoid it" rule: the **panel_r5 VERDICT §4.5**.
The fusion-vs-pfw isolating experiment run inside one tuner table: the **panel_r5
VERDICT §6**'s design, implemented here.  Everything else (codelet, transposes, shapes,
tuner protocol) is unchanged from my r2–r5 work.

### Node predictions (stated to be scored)

* **B=1: 0.570, unchanged** — byte-identical instructions and candidate set.
* **B=64: 0.615–0.625, pick = 1f-pfs (default)** — same config the node tuned to 3/3 in
  r5, now shipped without the flip risk.  Beating fusedaxes' 0.594 is not predicted; the
  4 % gap is not explained by anything I can see in the two codes (same shape, same
  cadence, my codelet 48 FP instructions lighter), and I decline to guess.
* **B=2048: 0.90–0.96, pick = 1f-pfs-pfw or 3p-pfs-pfw** — fusedaxes' 0.910 is the
  existence proof for the family on this exact cell; DRAM-bound, so shape and codelet
  differences should wash out.  If my number lands ≈1.10 instead, pfw did not transfer
  to my kernels and the difference is in something unexamined (their run-loop clamps the
  last volume's prefetch to itself where mine branches to NULL, is the only structural
  difference left).
* **B=16384: 1.25–1.32** — same reasoning against their 1.254.
* If both pfw picks land at fusedaxes-parity, the L=8 streaming story is converged
  (three entries, one technique) and §4.3's r5 confusion resolves as "pfw, not fusion" —
  the wallaby tables here already say that, pending node confirmation.

### Next

1. **Read the node pick strings and the 3p-pfs-pfw vs 1f-pfs-pfw split first** — that
   one line closes the VERDICT's §4.3/§4.5 question at L=8.
2. **B=1 is 1.32× its p0 floor at the now-settled 2.89 GHz** (1648 vs 1248 cycles).  The
   only untried lever consistent with the round's licence-clock synthesis ("once inside
   the 512-bit licence, mix widths freely" — L17's node-proven −7.4 %) is a mixed-width
   experiment: e.g. the 2p pass-2 interleave+store stage in ymm halves to take pressure
   off p5 while the codelets stay zmm.  Cheap to build as a fourth B=1 candidate; expect
   ≤5 %, but B=1 has been immovable for five rounds and this is the one mechanism class
   no L=8 entry has tried.
3. **The `-DL8R_SCRX=128` node A/B remains outstanding** (asked r3–r5; settles
   LITERATURE §4.5's padding question for the 3p scratch).
4. If the node's B=64 stays 0.61+ against fusedaxes' 0.594 with identical shape+prefetch,
   ask the monitor for one `perf stat -e ld_blocks_partial.address_alias` on both
   binaries at B=64 — the VERDICT names 4K aliasing (8192-byte volume stride) as the
   uninvestigated suspect for the whole L2-cliff band, and the two entries' different
   scratch geometries would show up exactly there.

---

## Round panel_r8

### Where round 7 landed (node, panel_r7)

Third in all four cells for the first time: B=1 **0.572** (batchsimd 0.558, fusedaxes
0.571), B=64 **0.612** (fusedaxes 0.587, batchsimd 0.588), B=2048 **0.980** (fusedaxes
0.930, batchsimd 0.945 — read honestly ≈0.984 per the VERDICT §3b), B=16384 **1.277**
(batchsimd 1.232, fusedaxes 1.234).  Three of my four predictions hit (VERDICT §4 item 7),
and my crossed `{1f,3p} × {pfw,no-pfw}` table is what withdrew the r5 "fusion wins"
reading (VERDICT §4.3: the answer is pfw, not fusion; my node picks split 3p-pfs-pfw at
B=2048 / 1f-pfs-pfw at B=16384, both carrying pfw).  The B=1 news is the actionable item:
**L8_batchsimd took the cell (0.5577/0.5647, picks FUSED 2/3) by offering the fused shape
at B=1** — the shape I have carried as `1f` since r5, with the same 52-instr codelet class
— while my node tuner shipped the `2p` default 3/3 (pick strings read `avx512-2p
(default)`), i.e. `1f` never displaced `2p` past the 2 % hysteresis bar.  fusedaxes' B=1
tuned to `fused` 3/3 at 0.571 with the heavier 56-op codelet.  The arithmetic is
consistent: their 0.571 − 48 FP instructions at 2.89 GHz ≈ 0.554–0.558 = batchsimd's
number, and my `1f` is that same configuration.

### What changed (two candidate-policy edits; zero kernel changes)

1. **B=1 default flipped `2p` → `1f`** (borrowed from **L8_batchsimd**'s panel_r7 B=1 win,
   which is itself their port of **L8_fusedaxes**' shape; my `1f` has been compiled and
   correctness-verified since r5).  Set is now {1f (default), 2p, 3p}.  Rationale: two
   rivals' node tuners independently converged on the fused shape at B=1, and default-first
   + hysteresis ships the node-proven config without exposure to in-tuner noise — the exact
   mechanism (r7: my own tuner could not see a 2 % win for `1f` in its own arena) that kept
   me on 2p while the cell moved.
2. **`3p-pfs` added to the mid-regime set** (borrowed from **L8_batchsimd**'s LANEX3+s0,
   which read **0.588 at node B=64** in r7, 2/3 picks).  My mid set had every other
   {shape} × {plain, pfs} combination but never 3-pass + spread-t0; batchsimd's number says
   it is competitive exactly in the L2-cliff cell where I trail by 4 %.  Set is now
   {1f-pfs (default), 3p-pfs, 2p, 1f, 2p-pfs, 3p}.  Default unchanged: 1f-pfs is still the
   node's 3/3 tuned pick from r5; if 3p-pfs is genuinely 0.588-fast in my code the tuner
   takes it (it needs >2 %, and the r7 gap is 4 %).

Streaming (big regime) is **byte-identical to r7** — candidate set, defaults, kernels,
prefetch cadences untouched.  The VERDICT says L=8 streaming is converged on one technique
across three entries; I am not churning a working path for the residual 3–4 %.

Also read but deliberately NOT adopted this round:
* **fusedaxes' last-volume prefetch clamp** (pf pointer clamped to the last volume instead
  of my NULL branch): the only structural difference between our fused kernels.  Its cost
  in my code is 16 fully-predicted branches per volume ≈ 0.2 %; their fusedAA/seq3AA work
  (the actual alias mechanisms) was declined by their own node tuner and is the monitor's
  §6 counter experiment now.  Not worth code churn ahead of that counter.
* **The mixed-width (ymm interleave) B=1 candidate** from my own r7 "Next" list: the r7
  VERDICT §5 falsified instruction-count reasoning for L1-resident kernels three ways at
  L=6 (17–25 % uop reductions, licence-fair race, measured 2.89 GHz clock: zero picks in
  eight cells — "width buys nothing where the kernel is not front-end-bound").  My B=1 is
  1.24–1.29× its p0 floor for the same class of reason; I decline to spend the round
  rediscovering L=6's null.

### Operation count per volume

Unchanged in every shape: 1248 vector FP (24 × 52-instr codelets, 4 real mul + 52 add =
56-flop optimum per codelet), 896 shuffles, 256+256 (2p/1f) or 384+384 (3p) L1
loads/stores, 0 copies, 0 spills.  This round moved plan-time policy only; `fft3d_execute`
code paths are bit-for-bit the r7 kernels.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4; clock state mixed across the
session — driver minima quoted from fast-state runs, comparisons only from same-process
in-tuner tables)

| B | r7 | this round | pick (fingerprint) | note |
|---|---|---|---|---|
| 1 | 0.308 µs | **0.308** | wallaby tunes away from the 1f default (in-tuner, one noisy process: 2p 0.446 < 3p 0.482 < 1f 0.496) | the known wallaby/node B=1 inversion, unchanged from r5–r7: wallaby cannot validate the flip; the basis is the two rivals' node numbers |
| 64 | 0.310 | **0.309** | **3p-pfs, tuned** (rel_l2 1.910e-16 = 3p family) | in-tuner same-process: **3p-pfs 0.602** < 3p 0.604 < 2p-pfs 0.633 < 2p 0.647 < 1f-pfs 0.658 < 1f 0.681 — the new candidate wins the wallaby table outright |
| 2048 | — | 0.522 µs/vol | mid set (wallaby ws = 0.53×its L3) | PASS, repeatable |
| 5632 | 0.595 | 0.594 | 1f-pfs-pfw / 3p-pfs-pfw (flip, see below) | streaming path byte-identical to r7 |

Correctness: PASS at B = 1, 64, 2048, 5632 (rel_l2 1.87e-16 … 2.27e-16, tol 1e-12).
Forced runs of the two configs the node may newly ship, each checked against numpy
individually: `L8R_FORCE=avx512-1f` at B=1 (rel_l2 2.269e-16, the 1f-family fingerprint)
and `L8R_FORCE=avx512-3p-pfs` at B=64 (rel_l2 1.910e-16, bit-matches plain 3p — the
prefetch is a pure hint, as established in r4).  Builds warning-free under `-Wall -Wextra`
at cascadelake, haswell, and x86-64.

**A tryout artifact worth recording so nobody reads it as a bug**: at B=5632 tryout.sh
printed `NOT REPEATABLE` across two *process* runs.  Traced with three tune-debug runs: it
is the 1f-pfs-pfw ↔ 3p-pfs-pfw pick flip (runs 1–2 picked 1f-pfs-pfw and are bit-identical
to each other; run 3's table read 3p-pfs-pfw 0.8913 vs 1f-pfs-pfw 0.9101 = 2.06 % — just
past the hysteresis bar — and its output is the 3p-family bit pattern, also PASS).  Each
plan is repeatable within itself, which is the contract (rule 4); the twins straddling the
2 % line on wallaby was documented and accepted in r7.  The node's r7 behavior (3p-pfs-pfw
3/3 at B=2048, 1f-pfs-pfw 3/3 at B=16384) shows its per-cell preferences are stable and
>2 % apart, so the flip risk is a wallaby phenomenon.

### What was tried and did NOT work

* Nothing new failed — the round is deliberately two low-risk, node-evidenced policy
  edits.  The r1–r7 failure lists all stand; the wallaby B=1 inversion (wallaby prefers
  2p/3p over 1f) was re-confirmed and remains a non-result for node purposes.
* Known residual, deliberately not attacked: my 1f-pfs at node B=64 runs 4 % behind
  fusedaxes' isomorphic fused+pfs (0.612 vs 0.587).  I diffed our fused kernels line by
  line this round: same phase structure, same store tables, same 8+8 spread cadence, my
  codelet 48 FP lighter; the only structural difference is the prefetch NULL branch
  (≈0.2 %).  The remaining suspects are the scratch-vs-out 4K-alias allocation lottery
  (fusedaxes' r7 model: 12–16 blocked loads/volume set by `(scratch − out) mod 4096`) and
  measurement spread; the VERDICT §6 counter run (`ld_blocks_partial.address_alias`)
  adjudicates exactly this, so building anything ahead of it would be guessing.  3p-pfs
  in the mid set is this round's hedge: batchsimd's 0.588 shows the 3-pass family dodges
  whatever the fused family pays there.

### Attribution summary

Fused shape at B=1: **L8_batchsimd** (panel_r7 B=1 win, 0.558), which is itself
**L8_fusedaxes**' structure — my `1f` port of it (r5) needed only the default flip.
3p-pfs at mid: **L8_batchsimd**'s LANEX3+s0 node number.  The decision NOT to build
mixed-width at B=1: **L6_pfa / L6_unrolled**'s three-way falsification (r7 VERDICT §5).
The decision NOT to build alias-avoidance: **L8_fusedaxes**' r7 declined-candidate result
plus the VERDICT §6 pending counter.

### Node predictions (stated to be scored)

* **B=1: 0.555–0.565, pick = 1f (default).**  Basis: batchsimd's FUSED (same shape, same
  codelet class) measured 0.5577/0.5647 on this node in r7; my 1f is that configuration
  and ships without needing a tuner win.  If it lands ≈0.572 instead, the 4 %
  B=64-style gap between my 1f and their FUSED exists at B=1 too and is then almost
  certainly the scratch-alias lottery — one more datum for the §6 counter.  If 2p
  displaces 1f (needs >2 % in-tuner), read the pick string: that would mean my 1f
  genuinely loses to 2p on the node and the batchsimd analogy fails inside my file.
* **B=64: 0.590–0.615.**  If 3p-pfs behaves like batchsimd's LANEX3+s0 (0.588), the
  tuner takes it (4 % > the 2 % bar) and the cell lands ≈0.59; if my 3-pass pays the
  same tax my fused does, the 1f-pfs default ships ≈0.612 unchanged.  The pick string
  is the diagnostic either way.
* **B=2048: 0.96–1.00, pick 3p-pfs-pfw; B=16384: 1.25–1.30, pick 1f-pfs-pfw** —
  byte-identical code and candidate sets to r7.
* Standing asks for the monitor, unchanged: the §6 `ld_blocks_partial.address_alias`
  counter at L=8 B=1/B=64 (fusedaxes' variants, but the answer transfers to all three
  entries), and the `-DL8R_SCRX=128` A/B (r3–r7, settles LITERATURE §4.5's padding
  question for my 3p scratch).

### Next

1. **Read the B=1 and B=64 pick strings first** — each of this round's two edits carries
   its own diagnostic (see predictions above).
2. **If the §6 alias counter lands and shows my scratch drew a bad residue**, the fix is
   cheap and already designed: allocate the scratch with 4 KiB of slack and choose the
   base at create time against the tuner arena's residue — but only *after* the counter
   says the mechanism is real and mine, because fusedaxes' node tuner already declined
   their execute-time version of exactly this.
3. **If B=1 lands ≈0.558**, the three entries have converged on one B=1 configuration
   and the cell is done at 1.24× floor pending the alias counter; propose to the panel
   that L=8 B=1, like streaming, stops being tuned.
4. Still do not touch the codelet, the transpose networks, or the streaming candidate
   set.

---

## Round panel_r9

### Where round 8 landed (node, panel_r8)

Third in all four cells: B=1 **0.570** (fusedaxes 0.552, batchsimd 0.564), B=64 **0.618**
(fusedaxes 0.575 min / median-tie with batchsimd 0.589), B=2048 **0.973** (batchsimd 0.912
min, honestly ≈0.93–0.98 per VERDICT §3b — a three-way tie), B=16384 **1.263** (fusedaxes
1.236, batchsimd 1.241).  Two findings own this round's decisions:

1. **My r8 B=1 bet was measured wrong, by my own pick strings.**  The three node runs
   read `2p (tuned)` 0.5700, `1f (default)` 0.5829, `1f (default)` 0.5813 — the fused
   shape I promoted to default on the strength of both rivals' node wins is **2 % slower
   than the 2p it displaced, in this file**, and the reported 0.570 came from the one run
   whose arena tuned back to 2p.  My own pre-registered branch fired ("if it lands ≈0.572
   the 4 % gap exists at B=1 too").  The VERDICT (§2) sharpens it: adopting a rival's
   winning *configuration* did not transfer between files even with the arithmetic now
   bit-identical.
2. **L=8 has collapsed to one algorithm** (VERDICT §3b): all three entries produce
   bit-identical output in all four cells, all publish 1248 vector FP per volume.  The
   0.552 / 0.564 / 0.570 B=1 spread is therefore a **pure measurement of non-arithmetic
   cost** — prefetch branch shape, scratch placement, code layout — and the pending
   monitor counter run (`ld_blocks_partial.address_alias`, VERDICT §6 ask 1) is the only
   named instrument that can attribute it.  The VERDICT also names this entry "the
   natural donor" if the panel cuts L=8 to two slots.

### What changed (one revert, one diagnostic; zero kernel changes)

1. **B=1 default flipped back `1f` → `2p`** (set now {2p (default), 1f, 3p}).  Basis: the
   r8 node data above — this file's own three-run measurement of exactly the two configs.
   The r8 flip's basis (rivals' node numbers for the same shape+codelet) is now a
   documented non-transfer; the r9 rule I take from it is **only this file's node numbers
   pick this file's defaults**.  1f stays offered (it needs >2 % in-arena to displace,
   which r7/r8 arenas say it does not have — so 2p should ship 3/3).
2. **The in-arena candidate table and the scratch residue are now published through
   `fft3d_description()`**, e.g. `pick[B=1]=avx512-2p (default) arena{2p=0.633 1f=0.655
   3p=0.604} scr@0x540` (first three candidates, µs/volume from the same `best[]` the
   pick logic uses; `scr@` = the plan scratch's address mod 4096).  This is L36_pfa's
   create-side-measurement pattern, which the r8 VERDICT endorses panel-wide ("requires
   nothing from the monitor... should become the panel's default").  What it buys, for
   free, on the next leaderboard:
   * the node's **arena-vs-driver ranking discrepancy becomes visible per run**
     (L8_batchsimd r8 documented arenas reproducibly inverting driver rankings at B=64;
     my r7/r8 story at B=1 is the same shape).  If the node's arena says 2p ≈ 1f while
     the driver splits them by 2 %, that is direct evidence the tax is tied to the
     *driver's* buffer layout, which the arena cannot see — exactly the discrimination
     the alias-counter ask needs, obtained without a counter.
   * the **scratch residue that each scored run actually drew** is on the record, so
     fusedaxes' blocked-load model ((scr−out) mod 4096 lottery) can be checked against
     the three runs' numbers post hoc.

Mid and streaming candidate sets, defaults, kernels, prefetch cadences: **byte-identical
to r7/r8** for the third round — the VERDICT declares streaming converged (three entries
within 2.2 % at B=16384) and B=2048 a tie once batchsimd's outlier min is read honestly.

Also read and deliberately NOT done: (a) no alias-avoidance/scratch-repositioning build —
the r8 evidence pile is now 0-for-4 across the panel (L17_matrixsimd's model-chosen twins
picked everywhere and worth −0.6 %…+0.4 %; L6_pfa's rotation 0-for-8 picks; L36_mixedradix's
always-on pin +1.2 % at B=1; fusedaxes' fusedAA/seq3AA declined in-band r7–r8), and my r8
record already gated this on the counter run saying the mechanism is real and mine;
(b) no `-funroll-loops` pragma — the r8 VERDICT §3c shows the build-flag gap never existed
and batchsimd's A/B at L=8 measured the null directly; (c) no new B=1 shapes or codelet
work — five falsified uop-deletion classes at L=17/L=6 (§4/§5) plus eight rounds of flat
B=1 here say the residue is not instruction-shaped.

### Operation count per volume

Unchanged in every shape: 1248 vector FP (24 × 52-instr codelets, the 56-flop optimum),
896 shuffles, 256+256 (2p/1f) or 384+384 (3p) L1 loads/stores, 0 copies, 0 spills.
Re-audited this round under the node's exact flags (`-O3 -march=cascadelake
-fno-math-errno -funroll-loops`, gcc 11.4): every avx512 run function shows **0 zmm stack
spills/reloads, 12–23 loop-edge zmm copies, ≤12 leas** — L45_pfa's r8 lea-spill pathology
(48 leas + 37 GPR spills from IVOPTS, fixed there by an opaque-base asm barrier) does
**not** exist in this file, so the barrier was checked and not adopted.  The description
snprintf runs once per create; execute paths are bit-for-bit the r7 kernels.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4, tryout.sh; fast clock state)

| B | r8 | this round | pick (wallaby) | arena table (same process) |
|---|---|---|---|---|
| 1 | 0.308 µs | **0.308** | 3p (tuned; the standing wallaby/node inversion — node keeps 2p) | 2p=0.633 1f=0.655 3p=0.602 |
| 64 | 0.309 | **0.313** (20.045/64) | 3p-pfs (tuned; also the known inversion — node ships 1f-pfs) | 1f-pfs=0.661 3p-pfs=0.603 2p=0.651 |
| 5632 | 0.594 | **0.582** | 1f-pfs-pfw (default, 3/3 in a dedicated pick-stability check) | 1f-pfs-pfw=0.589 3p-pfs-pfw=0.587 3p-pfs=0.814 |

Correctness: PASS at B = 1, 8, 64, 5632 (rel_l2 1.87e-16 … 1.92e-16, tol 1e-12).  The
three B=1 shapes were additionally force-run via `L8R_FORCE` and checked against numpy
individually: 2p 1.308e-16, 1f 2.269e-16, 3p 1.874e-16 — the unchanged family
fingerprints, so the default revert ships a byte-identical known kernel.  One tryout at
B=5632 printed `NOT REPEATABLE` across two *processes*: traced (again) to the documented
r8 artifact — one process's arena crossed the 2 % hysteresis line to 3p-pfs-pfw (its
output is the 3p-family bit pattern, also PASS); a dedicated 3-process check then picked
1f-pfs-pfw 3/3 with the twins 0.3–1 % apart in-arena.  Each plan remains bit-identical
across its own runs, which is the contract.  AVX2 path (`-mno-avx512f`, B=64, 1.19 µs/vol)
and portable path (`-mno-avx512f -mno-avx2 -mno-fma`, B=8, 1.67 µs/vol) PASS and are
repeatable.  Builds warning-free under `-Wall -Wextra` at cascadelake / haswell / x86-64.

### What was tried and did NOT work

* **L45_pfa's opaque-base barrier: checked, not applicable** (the objdump audit above —
  nothing to fix; recorded so nobody re-runs the audit at L=8).
* Nothing else was attempted, deliberately.  The r1–r8 failure lists all stand.  The
  wallaby B=1/B=64 inversions (wallaby prefers 3p/3p-pfs over the node's 2p/1f-pfs) were
  re-confirmed and remain non-results for node purposes — and are now self-documenting on
  the leaderboard via the arena string.

### Attribution summary

The B=1 revert basis: the **panel_r8 node pick strings** for my own file (via VERDICT §2/
§3d).  Description-side arena publication: **L36_pfa**'s create-side measurement pattern,
elevated by the **panel_r8 VERDICT** ("should become the panel's default"); the
arena-vs-driver inversion it is aimed at is **L8_batchsimd**'s r8 lending.  The decision
not to build alias avoidance: the r8 panel-wide 0-for-4 (L17_matrixsimd, L6_pfa,
L36_mixedradix, L8_fusedaxes).  The decision not to ship an unroll pragma:
**L8_batchsimd**'s r8 null + VERDICT §3c.

### Node predictions (stated to be scored)

* **B=1: 0.570 ± 0.002, pick = 2p (default) 3/3.**  This is the r7 configuration shipped
  without the r8 lottery; 1f displacing it would need the >2 % in-arena win it has never
  shown.  The prize this round is the *arena string*: if the node's arena reads
  2p ≈ 1f (within ~1 %) while the driver holds the 2 % gap, the file tax is
  driver-buffer-specific and the alias-counter ask gains a second, independent line of
  evidence; if the arena also reads 1f +2 %, the tax lives in my 1f code path itself and
  the counter should look at code layout instead.
* **B=64: 0.612–0.620, pick = 1f-pfs (default)** — unchanged config; the cell is not
  reachable from inside this file until the tax is attributed.
* **B=2048: 0.95–1.00** and **B=16384: 1.25–1.28** — byte-identical code and candidate
  sets for the third round; both cells are ties per the VERDICT's honest-minimum reading.
* Standing asks, unchanged in substance: the §6 alias counter at B=1 (now with the
  per-run `scr@` residue on the leaderboard to correlate against), and the
  `-DL8R_SCRX=128` A/B (r3–r8; touches only the 2p/3p scratch stride).

### Next

1. **Read the node's B=1 arena strings against the driver numbers first** — that
   comparison is this round's experiment, and either outcome narrows the tax to
   buffer-layout or code-layout before any counter runs.
2. **If the counter run lands and attributes the tax to (scr−out) residues**, the fix is
   the execute-time base selection already sketched in r8 (4 KiB slack, deterministic
   per (in,out), cached) — build it only then, because the panel's alias interventions
   are 0-for-4 without counter evidence.
3. **If the panel cuts L=8 to two slots** (VERDICT §7 names this entry the donor), the
   transferable assets are documented: the regime-gated tuner protocol, the spread/pfw
   cadences, and the arena-publication pattern — all attributed and reproducible from
   this record.
4. Still do not touch the codelet, the transpose networks, or the streaming candidate
   set.

---

## Round panel_r10

### Where round 9 landed (node, panel_r9)

Third in all four cells for the second round, not promoted, and named the consolidation
donor (VERDICT §6/§7).  B=1 **0.5784** (batchsimd 0.5527, fusedaxes 0.5530); my prediction
failed on both the number (0.570 ± 0.002) and the pick (2p shipped 1/3, not 3/3 — the
arena displaced the 2p default to `1f` in runs 1–2).  But the arena strings I built in r9
delivered exactly the discrimination they were built for: **the node's create() arena
ranks `1f` above `2p` in 5 of 6 runs across r8–r9 (r9: 1f 0.565–0.574 vs 2p 0.561–0.603)
while the driver ranks `1f` 0.6–2 % slower (r8: 1f 0.5813/0.5829 vs 2p 0.5700)** — the
arena-inverts-the-driver failure, now measured in one binary.  B=64 **0.5971 min /
0.6369 median, spread 11.3 % at IDENTICAL picks (1f-pfs 3/3) and identical scratch residue
(scr@0x4c0 all runs)** — so the B=64 spread is the driver-buffer allocation lottery, not
pick noise.  B=2048 0.9774 / B=16384 1.269, both within 1.5–3 % of fusedaxes, streaming
declared converged.

Two rival results define this round's moves:

1. **L8_batchsimd won B=1 (0.5527, "fixed, no tuner" 3/3, tail eliminated)** with three
   changes: SI moved from scr+512 to scr+520 doubles (breaking an **exact 4096-byte**
   relation between the two halves of its own scratch), a page-aligned arena, and **no
   B=1 tournament at all** (its arena had picked a driver-slower variant in 4 of 6
   creates).  The VERDICT (§5) elevates the first into the §4.5 refinement: **the 4K-alias
   mechanism is reachable when both colliding addresses are inside the plan's own scratch,
   and unreachable when one is a driver buffer.**  My `1f` has exactly the reachable
   defect: `sr = scr, si = scr + 512` doubles — si is EXACTLY 4096 B after sr, so every
   phase-A store pair ST(sr+o)/ST(si+o) and every phase-B load pair LD(rrow+o)/LD(irow+o)
   shares its bits-11:6 line residue, and phase-B loads from sr can be 4K-blocked by
   phase-A's in-flight si stores.
2. **The bill**: batchsimd applied the layout change to every mode and paid **+5.2 % at
   B=2048** and +2.5 % at B=64 (VERDICT §2); its r10 file now gates SI=520 to B=1 only.
   L8_fusedaxes has likewise added de-aliased `fusedSI` variants at +520 this round.

### What changed (three edits)

1. **De-aliased fused-shape twins `1f520`** (borrowed from **L8_batchsimd**'s panel_r9
   B=1 node win; same fix L8_fusedaxes adopted as `fusedSI`): `KERNEL1F_BODY` gained an
   SI-offset parameter; the new `kernel1f520` places si at **scr + 520 doubles (+4160 B,
   still 64-B aligned)** instead of +512.  Arithmetic, shuffles, prefetch cadence:
   byte-identical to `1f`; output is bit-identical (verified by `cmp` on wallaby).
   **Gated to B=1 and the mid regime** — the streaming kernels keep the +512 layout
   byte-identical to r7–r9, because batchsimd's streaming regression is the documented
   cost of applying it everywhere and the attribution A/B is still pending in their file.
   `-DL8R_SIOFF=512` reverts.  The scratch allocation grows to max(8·SCRX, SIOFF+512)
   doubles so the standing `-DL8R_SCRX=128` A/B flag cannot underallocate.
2. **B=1 tournament removed** (borrowed from **L8_batchsimd**'s "fixed, no tuner" rule:
   after two rounds of the arena inverting the driver at this cell, the right amount of
   arena trust is zero).  B=1 ships **avx512-1f520 hardwired**.  The candidates
   {1f520, 1f, 2p, 1f520j} are still timed with the full r4 protocol and published
   through `fft3d_description()` (`arena{...}`, now 4 entries), and `L8R_FORCE` still
   works, but nothing can displace the pick.  Shipping the fused shape rather than 2p is
   the mechanism bet: my r8 flip to unfixed `1f` measured 0.5813–0.5829 vs 2p 0.5700, and
   the difference between my `1f` and batchsimd's winning FUSED (0.5527) that this round
   can act on is the scratch alias.
3. **Mid default → `1f520-pfs`**; set trimmed to {1f520-pfs, 3p-pfs, 1f-pfs, 2p} (1f,
   2p-pfs, 3p dropped — never picked at node B=64 in five rounds).  **Association-order
   probe `1f520j`** added at B=1 only (the r9 VERDICT §6 directive to propagate L=6's
   codelet-association result to L=8's radix-8): `RADIX8J` issues the 16 final add/subs
   feeding outputs 0,4,2,6 as FMA/FNMA with a broadcast 1.0, so **every output join is
   FMA-class** at identical DAG, depth, and count.  round(1.0·x+y) = round(x+y), so the
   output is bit-identical (cmp-verified).  Timed and published, never picked; if the
   node reads `1f520j` ≤ −2 % vs `1f520`, r11 adopts it.  (Honest caveat recorded: at
   L=6 the winning joins fed stores directly; my joins feed the ZUNTRI shuffle network,
   and on CLX all 512-bit FP shares one port — the probe measures whether any of the L=6
   effect survives those two differences.)

### Operation count per volume

Unchanged in every shipped shape: 1248 vector FP (24 × 52-instr codelets, the 56-flop
optimum), 896 shuffles, 256+256 (2p/1f/1f520) or 384+384 (3p) L1 loads/stores, 0 copies,
0 spills.  `1f520` differs from `1f` only in the si base register value; `RADIX8J` swaps
16 add/sub for 16 FMA-class ops (probe only, never shipped).  DRAM-facing traffic
unchanged; streaming paths bit-for-bit the r7 kernels.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4, tryout.sh)

| B | r9 | this round | pick (fingerprint) | note |
|---|---|---|---|---|
| 1 | 0.308 µs | **0.337** | 1f520 (fixed) — rel_l2 2.269e-16, the 1f family | wallaby prefers 3p/2p here (in-tuner 2p 0.6335 < 1f ≈ 1f520 0.655 — the standing inversion); the hardwire basis is node data, not this number |
| 64 | 0.313 | **0.603/vol** | 3p-pfs (tuned; the known wallaby inversion — node should ship 1f520-pfs default) | forced runs: all four mid candidates PASS individually |
| 2048 | 0.522 | **0.862/vol** | mid set (wallaby ws = 0.53×its L3), 1f-family fingerprint | PASS, repeatable |
| 5632 | 0.582 | **0.597/vol** | 1f-pfs-pfw (default) 3/3 in a dedicated tune-debug check (twins 0.3–2 % apart in-arena) | streaming byte-identical to r9 |

Key wallaby readings for the two experiments:

* **si520 is wallaby-invisible, as expected**: in-tuner same-process, 1f = 0.6549/0.6551
  vs 1f520 = 0.6555/0.6559 — dead even, exactly batchsimd's r9 wallaby null (SPR's alias
  penalty is small and its heap offsets differ).  Only the node can read this experiment;
  the published arena string will carry its answer on the next leaderboard.
* **1f520j (FMA-join) reads +0.5–0.6 % on wallaby** (0.6593/0.6594 vs 1f520's
  0.6555–0.6590) — flat within session noise.  The L=6 association result was likewise
  wallaby-invisible and sign-inverted on the node (SPR's extra add ports hide join-order
  effects), so this null does not decide the probe; the node's arena string does.

Correctness: PASS at B = 1, 64, 2048, 5632 via tryout (rel_l2 1.910e-16 … 2.271e-16, tol
1e-12), repeatable within every plan.  One B=5632 tryout printed `NOT REPEATABLE` across
two *processes* — traced (again, third round running) to the documented 1f-pfs-pfw ↔
3p-pfs-pfw twin flip at the 2 % hysteresis line; a dedicated 3-process tune-debug check
picked 1f-pfs-pfw 3/3 and each plan is bit-identical across its own runs, which is the
contract.  Forced runs, each checked against numpy individually at B=64: 1f520-pfs =
1f-pfs = 2.268e-16 (same bit class, confirming the de-alias is address-only), 3p-pfs
1.916e-16, 2p 1.337e-16.  `cmp`-verified at B=1: 1f520j ≡ 1f520 ≡ 1f bit-identical.
Builds warning-free under `-Wall -Wextra` at native (SPR), `-march=cascadelake`
(run-verified on wallaby: B=1 PASS + repeatable), `-mno-avx512f` (AVX2, B=64 PASS), and
`-march=x86-64` (portable, B=8 PASS).

### What was tried and did NOT work

* Nothing new failed on wallaby — but two null readings are recorded above so nobody
  mistakes them for failures: the si520 wallaby null (expected; node-only experiment) and
  the 1f520j wallaby flat (expected; the L=6 precedent is that only the node can read
  association order).
* **Not done, deliberately**: page-aligning the scratch (the second half of batchsimd's
  r9 change) — my scr residue is already published per run (scr@0x540/0x4c0, stable per
  build), there is no evidence which residue is good, and page-aligning is the prime
  suspect for their streaming bill.  Execute-time scratch-base selection against the
  driver's buffers (fusedaxes' fusedAA, first node pick at B=64 r9 run 3) — one weak
  signal against a panel record of 0-for-4 alias interventions aimed at driver buffers;
  my B=64 lottery evidence says the mechanism is real there, but the in-scratch fix must
  be attributed first or the two changes confound, exactly like batchsimd's r9 pair.

### Attribution summary

In-scratch de-alias (si +512 → +520) and the no-tournament-at-B=1 rule:
**L8_batchsimd** (panel_r9 node win, 0.5527, "fixed no tuner" 3/3), elevated by the
**panel_r9 VERDICT §4.5/§5**; the same fix appears as **L8_fusedaxes**' fusedSI this
round.  The gate keeping the streaming kernels at +512: batchsimd's own +5.2 % B=2048
bill (**VERDICT §2**) and their r10 gating, read directly from their file.  The
association-order probe: **L6_pfa / L6_unrolled**'s panel_r9 experiment (store-feeding
FMAs beat adds 3–6 % on CLX, sign-inverted on SPR), propagated here on the **VERDICT
§6**'s explicit instruction.  The probe-but-never-pick pattern: **L36_pfa**'s
create-side measurement, panel default since r8.

### Node predictions (stated to be scored)

* **B=1, pick string reads `avx512-1f520 (fixed)` 3/3 by construction.**  Pre-registered
  fork: **if ≤ 0.565, the sr/si 4K alias was the 1f driver tax** and the fused shape +
  de-alias transfers (batchsimd's 0.5527 is the existence proof at this exact cell);
  **if ~0.578–0.583, the fix bought nothing in my file** and the residual 1f-vs-2p gap
  is code layout — in that case r11 hardwires 2p at ~0.570–0.578 and B=1 is closed here.
  Either way the arena string carries 1f520-vs-1f same-process, which reads the alias
  mechanism independently of the driver lottery.
* **B=64: 0.59–0.62, pick = 1f520-pfs (default) 3/3.**  The de-alias attacks the only
  reachable part of the 11.3 % r9 spread; the driver-buffer lottery part is untouched, so
  the *median* may stay noisy even if the min improves.
* **B=2048: 0.95–1.00; B=16384: 1.25–1.28** — streaming byte-identical for the fourth
  round.
* **1f520j arena reading**: expected within ±1 % of 1f520 (null) given the shuffle-fed
  stores; if it reads ≤ −2 %, r11 adopts RADIX8J in all shapes — it is bit-identical, so
  adoption is free of accuracy risk.

### Next

1. **Read the B=1 driver number against the pre-registered fork above** — it is a clean
   one-bit answer on whether §4.5's reachable case transfers into this file.
2. **If B=1 lands ≤ 0.565**: propose the same si-offset audit to the streaming shapes,
   but only via a create-time published A/B (1f520-pfs-pfw twin), never by default —
   batchsimd's bill says the streaming sign may be opposite.
3. **If the 1f520j probe reads ≤ −2 %**: adopt RADIX8J everywhere (bit-identical, zero
   risk) and record L=8 as the second geometry where association order pays on CLX.
4. **If the panel cuts L=8 to two slots**, this record now carries the donor assets in
   final form: the regime-gated tuner, the spread/pfw cadences, the arena-publication
   pattern, the arena-inverts-driver evidence, and both of this round's mechanism
   experiments with their node readings.

---

## Round panel_r11

### Where round 10 landed (node, panel_r10)

Third in three of four cells for the third round; named the donor for the second round;
not promoted.  B=1 **0.5760** (batchsimd 0.5510, fusedaxes 0.5560) — my pre-registered
fork read its **second branch exactly** ("if ~0.578–0.583 the fix bought nothing … r11
hardwires 2p and B=1 is closed here"), with the arena confirming `1f520 ≡ 1f` to 0.3 %
(0.569/0.567, 0.574/0.572 ×2).  The in-scratch de-alias transfer is dead in this file
(1-for-3 across the geometry — VERDICT §2: "not a portable mechanism; a property of a
particular scratch layout").  The `1f520j` probe read **+0.5 to +0.7 %** against a ≤ −2 %
adoption bar — null, and the VERDICT closed the association-order propagation ask with it
(§5: the L=6 effect is a property of joins that feed *stores*; mine feed the ZUNTRI
shuffle network).  B=64 0.6166 min (+3.4 %) but **median 0.6195 (−2.7 %) and spread
11.3 % → 2.5 %** — read by the monitor as the driver-buffer lottery drawing differently,
not a mechanism.  B=2048 0.9838, B=16384 1.2497.

Two findings from the round own this round's edits:

1. **VERDICT §3(a), charged against this file by name**: my B=2048 leaderboard minimum
   (0.9838) was produced by run 2's pick `3p-pfs-pfw` while the round's correctness check
   only ever saw run 3's `1f-pfs-pfw` — **different bit classes** (1.92e-16 vs 2.27e-16
   fingerprints).  The streaming pool spanning two bit classes is what made that possible;
   it had already produced three rounds of cross-process `NOT REPEATABLE` tryout artifacts.
   The panel-side fix is L36_mixedradix's rule, elevated by the VERDICT: **a tuner pool
   must be one bit class; cross-class comparisons ride the description string, never the
   pick.**
2. **The node's own arena data in my r10 JSONs points every default the same way**:
   B=1 arena ranked `2p` FIRST 3/3 (0.560/0.569/0.569 vs 1f520 0.569/0.574/0.574) —
   agreeing with the r8 driver (2p 0.5700 vs 1f 0.5813/0.5829) for the first time;
   B=64 arena ranked `1f-pfs` ahead of `1f520-pfs` in 2 of 3 runs (0.605/0.606 vs
   0.614/0.613); B=2048 arena ranked `3p-pfs-pfw` ahead of `1f-pfs-pfw` in ALL THREE runs
   (0.900–0.907 vs 0.913–0.926) and the two driver runs that shipped it read 0.984/0.993
   vs 1.012 for the 1f run; B=16384 arena read the twins dead even (1.121–1.145 both).

### What changed (three plan-time policy edits; zero kernel changes)

1. **B=1 hardwired `1f520` → `2p`** — executing my own r10 pre-registered fork, second
   branch.  No tournament (L8_batchsimd's r9 rule stands); probes {1f520, 1f, 3p} are
   timed and published, `L8R_FORCE` still works.  The `1f520j` probe is retired from the
   timed set (node-answered in r10; kernel stays compiled for the record).
2. **Single-bit-class installable pools** (the §3(a) fix; rule borrowed from
   **L36_mixedradix** via the r10 VERDICT §2/§3(a)).  Every candidate now carries an
   `installable` flag; the argmin runs over installable candidates only; non-installable
   probes are timed and published with a `*` marker in the arena string.
   * Streaming pool → **3p family only** {3p-pfs-pfw (default), 3p-pfs, 3p-pf}, probe
     {1f-pfs-pfw}.  The default family flip 1f→3p is the node's own r10 data (item 2
     above): the class the pool keeps is also the measured-faster one at B=2048 and a tie
     at B=16384.  Installable pool `cmp`-verified bit-identical on wallaby.
   * Mid pool → **1f family only** {1f-pfs (default), 1f520-pfs}, probes {3p-pfs, 2p}.
     Default reverted 1f520-pfs → 1f-pfs on the r10 node arena (2-of-3 adverse); the two
     installable twins are `cmp`-verified bit-identical, so any residual flip is
     check-safe by construction.
   * AVX2/portable paths get the same treatment (their pools were also cross-class).
3. **Description format**: arena entries now carry the `*` probe marker, so the
   leaderboard shows per-run cross-class comparisons that can no longer be picked.

### Considered and deliberately NOT done, with the numbers

* **Porting L8_fusedaxes' `fusedAA`** (the VERDICT §6 L=8 item, addressed to
  L8_batchsimd: "Port it").  Read their code and record; declined here for three reasons.
  (a) Its node arena readings in its own file are a lottery, not a win: B=1 fusedAA
  {0.600, 0.566, 0.574} vs fused {0.558, 0.579, 0.560} — ahead 1 of 3, behind 4–7 % in
  the other two; B=64 fusedAA+pfs {0.631, 0.592, 0.617} vs fused+pfs {0.604, 0.605,
  0.593} — ahead 1 of 3.  (b) The VERDICT orders batchsimd to port it, so the mechanism
  gets its second-file pricing this round regardless; a third simultaneous port adds no
  information.  (c) My r9/r10 rule — alias interventions only on counter evidence or a
  consistent node signal — still binds; the panel's alias-fix record is now 1-for-many.
  If batchsimd's port wins a cell, r12 adopts from their file with two existence proofs.
* **fusedaxes' last-volume prefetch clamp** (the one structural difference between our
  fused kernels): estimated at ~0.2 % in r8, far below the ±2 % noise floor the r9/r10
  VERDICTs enforce.  Not worth code churn on the streaming path in the same round that
  changes its default.

### Operation count per volume

Unchanged in every shape: 1248 vector FP (24 × 52-instr codelets, the 56-flop optimum),
896 shuffles, 256+256 (2p/1f) or 384+384 (3p) L1 loads/stores, 0 copies, 0 spills.  All
execute paths are bit-for-bit the r7 kernels; this round moved plan-time policy only.

### What was measured (wallaby, Gold 6448Y SPR, gcc 11.4, tryout.sh; mixed clock states —
driver minima quoted, comparisons only from same-process in-tuner tables)

| B | r10 | this round | pick (fingerprint) | note |
|---|---|---|---|---|
| 1 | 0.337 µs | **0.324** | 2p (fixed) — rel_l2 1.308e-16, the 2p family | in-tuner: 3p 0.6015 < 2p 0.6331 < 1f≈1f520 0.656 — the standing wallaby B=1 inversion (wallaby prefers 3p, node prefers 2p); hardwire holds by construction |
| 64 | 0.603/vol | **0.340/vol** (fast state) | 1f-pfs (default) — 2.267e-16 | in-tuner: 3p-pfs* 0.6037 < 2p* 0.6478 < 1f-pfs 0.6602 ≈ 1f520-pfs 0.6603 — the known wallaby B=64 inversion, now structurally unable to flip the pick |
| 2048 (wallaby mid, ws=0.53×its L3) | 0.862/vol | **0.429/vol** | 1f family (2.271e-16) | PASS, repeatable |
| 5632 (node-B2048 analog) | 0.597/vol | **0.595/vol** | 3p-pfs-pfw (default) — 1.915e-16 | in-tuner: 3p-pfs-pfw 0.6101 < 1f-pfs-pfw* 0.6201 (−1.6 %) < 3p-pf 0.8176 < 3p-pfs 0.8213 |
| 16384 | — | **0.908/vol** | 3p-pfs-pfw — 1.915e-16 | PASS, repeatable |

Correctness: PASS at B = 1, 8, 64, 2048, 5632, 16384 (rel_l2 1.308e-16 … 2.273e-16, tol
1e-12), **repeatable across processes at every size tried** — the r8–r10 cross-process
`NOT REPEATABLE` twin-flip artifact is gone by construction, since any pick flip now
stays inside one bit class.  Every candidate force-run via `L8R_FORCE` and checked
against numpy individually: B=1 {2p 1.308e-16, 1f520 = 1f 2.269e-16, 3p 1.874e-16};
B=64 {1f-pfs = 1f520-pfs 2.267e-16 and `cmp` bit-identical, 3p-pfs 1.910e-16, 2p
1.324e-16}; B=5632 {3p-pfs-pfw = 3p-pfs = 3p-pf 1.915e-16 and `cmp` bit-identical
pairwise, 1f-pfs-pfw 2.273e-16}.  AVX2 path (`-mno-avx512f`, B=64, 1.910e-16) and
portable path (`-mno-avx512f -mno-avx2 -mno-fma`, B=8) PASS and repeatable.  Builds
warning-free under `-Wall -Wextra` at native (SPR), `-march=cascadelake` (run-verified
on wallaby, B=1 PASS), haswell, and x86-64.

### What was tried and did NOT work

* Nothing new failed — the round is three node-evidenced policy edits plus two documented
  declines (fusedAA, prefetch clamp — see above, with the numbers that priced them).
  The r1–r10 failure lists all stand.  The wallaby B=1/B=64 inversions were re-confirmed
  (in-tuner tables above) and remain non-results for node purposes; they are now also
  harmless, because the pools they used to flip are single-class or hardwired.

### Attribution summary

The installable/one-bit-class-pool rule: **L36_mixedradix** (its r10 `NOT REPEATABLE`
hard failure and the `installable` fix), elevated panel-wide by the **panel_r10 VERDICT
§2/§3(a)** — applied here because §3(a) charged this file's B=2048 provenance by name.
The B=1 hardwire-2p and mid-default reversion: this file's own r10 node arena strings and
pre-registered fork (the r9 rule — only this file's node numbers pick this file's
defaults).  The streaming default flip to 3p-pfs-pfw: this file's own r10 node arena
(3/3 at B=2048) and driver splits.  The decision not to port fusedAA: **L8_fusedaxes**'
own r10 node arena tables, plus the VERDICT §6 assignment of the port to
**L8_batchsimd**.

### Node predictions (stated to be scored)

* **B=1: 0.570–0.578, pick string `avx512-2p (fixed)` 3/3 by construction.**  Basis:
  r8's driver measurement of 2p (0.5700) and the r10 arena consensus (2p 0.560–0.569).
  The arena string will show 2p vs the 1f/3p probes same-process.  **B=1 is closed in
  this file either way** — every shape, both de-alias variants, and the association
  order have now been driver-priced here; the residual vs batchsimd's 0.551 is
  code-layout/allocation noise this file cannot reach.
* **B=64: 0.60–0.62 min, pick `1f-pfs` or `1f520-pfs` (bit-identical, either string is
  fine).**  The lottery's reachable part was priced null; the goal here is the median
  and spread staying at r10's improved level (0.6195, 2.5 %) with the default now
  matching the arena's 2-of-3 preference.
* **B=2048: 0.94–0.99, pick `3p-pfs-pfw` (default) 3/3 expected** (arena preferred it
  3/3 in r10; it now starts with hysteresis instead of fighting it).  Whatever number
  appears, it is for the first time in three rounds guaranteed to be the checked bit
  class.
* **B=16384: 1.24–1.28, pick `3p-pfs-pfw`.**  The twins were dead even in the r10 arena
  (≤0.2 %); the family flip is a provenance fix, not a performance bet — if it costs
  more than ~1 %, the arena string will show 1f-pfs-pfw* ahead and r12 should consider
  a 1f-only streaming pool instead (still single-class, the other way).

### Next

1. **Read the four pick strings and the probe deltas in the arena strings first.**  The
   `1f-pfs-pfw*` probe at the streaming cells is the one number that could send r12 back
   (see the B=16384 branch above).
2. **If L8_batchsimd's fusedAA port wins B=64 or B=1**, adopt it from their file in r12 —
   at that point the mechanism has a second-file existence proof, which is exactly the
   bar my r9/r10 rules set for alias-class interventions.
3. **If the panel executes the donor consolidation**, this record's transferable assets
   are final and now include the single-bit-class tuner protocol with published
   cross-class probes — the concrete §3(a) fix any surviving entry can copy.
4. Still do not touch the codelet, the transpose networks, or the prefetch cadences.
