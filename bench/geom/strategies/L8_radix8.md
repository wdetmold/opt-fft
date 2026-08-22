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
