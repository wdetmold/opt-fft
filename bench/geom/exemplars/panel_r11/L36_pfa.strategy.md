# L36_pfa — Good–Thomas PFA 4×9 per axis + slab fusion + blocked intermediate

`impl/L36_pfa.c`, `fft3d_name() == "L36_pfa"`, L = 36 only.

---

## Round 1 (first implementer)

### Technique

Row–column 3D DFT, each 1-D 36-point transform done by the **Good–Thomas /
prime-factor algorithm with N1 = 4, N2 = 9**. gcd(4,9)=1, so with

```
input  (Ruritanian) : n = (9*n1 + 4*n2) mod 36        n1 in [0,4), n2 in [0,9)
output (CRT)        : k = (9*k1 + 28*k2) mod 36       ([9^-1]_4 = 1, [4^-1]_9 = 7)
```

`W_36^{nk} = W_4^{n1 k1} · W_9^{n2 k2}` exactly — the whole inter-stage twiddle
array vanishes. Derivation: substituting both maps, the two cross terms carry a
factor `N1*N2 ≡ 0 mod 36` and drop out, and `N2^2 σ1 ≡ N2`, `N1^2 σ2 ≡ N1 (mod 36)`.
So the 36-point DFT is **9 independent 4-point DFTs, then 4 independent 9-point
DFTs, with nothing in between**. Both index maps are folded into the address
arithmetic of the surrounding pass (`SOFF`/`DOFF` macros), so the permutation
costs *zero instructions* — exactly the claim in §02 §5.5, and it holds.

Layout: **split-complex (SoA)** everywhere internally. Consequence: not one
shuffle inside any butterfly; every codelet instruction is `vaddpd`/`vsubpd`/
`vmulpd`/`vfma*pd`. The only shuffles in the transform are the input
deinterleave, one VL×VL register transpose that lifts z out of the lanes, and
one VL×VL register transpose that re-interleaves the answer into `out`.

Pass structure (the part that took the most iterating):

```
for s in 0..NSL-1:                       # z-slab, VL z-values per slab
    load  in[x][y][z=s*VL..] -> P[x*36+y]   (deinterleave, VL z per lane group)
    x pass: 36 codelets, axis x (stride 36 vectors), lanes over z, in place in P
    y pass: 36/KXG codelets, axis y (stride 1), lanes over z
            + VL×VL transposes -> store into T
z pass:  NTB=1296/VL codelets, axis z out of T, lanes over VL consecutive (kx,ky)
            + VL×VL transposes -> interleave straight into `out`
```

`P` (36×36 vectors, 166 KB at VL=8) is written, x-transformed and y-transformed
entirely inside L2, so the x and y passes cost no L3 traffic at all. `kx` is
processed in groups of `KXG = VL/4` so that the flat transverse index
`t = kx*36 + ky` always starts on a VL boundary (2·36 = 72 = 9·8).

`T` is blocked as **`T[slab][tile][lane-within-slab]`**. This is the single most
important layout decision in the file and it is *not* the obvious one — see
"what did not work".

### Operation count

Per 36-point 1-D transform, on split-complex vector data:

| module | instrs | flops | note |
|---|---|---|---|
| 4-point | **16** | 16 | all adds; ×(±i) is a register rename in SoA, so **zero multiplies** |
| DFT3 (FMA form) | 12 | 20 | `t,q,y0` = 6 adds; `u` = 2 FMA; `y1,y2` = 4 FMA |
| 9-point = CT 3×3 | **88** | 144 | 6·DFT3 = 72 (36 add + 36 fma) + 4 complex twiddles ×4 = 16 (8 mul + 8 fma) |
| **PFA-36** | **496** | **720** | 9·16 + 4·88 → **288 adds + 208 mul/fma** |

Comparison (FFTW's own plan accounting for 36, §02 §5.4): CT-36 = 552 instrs /
756 flops. So PFA is **−10.1 % instructions, −4.8 % flops** here, and it removes
**15 nontrivial complex multiplies per 36-point transform** = 90 flops; over one
volume that is `3·1296·15 = 58 320` complex multiplies (≈350 k flops) never done.
Per volume: `3·1296·496 = 1 928 448` FP instructions scalar-equivalent
(2 799 360 flops, 60.0 flops/point).

**Verified in the object code**: the AVX-512 build of `exec_volume` contains
exactly **1488 = 3 × 496** FP instructions, i.e. all three codelet sites hit the
predicted count with the loops fully unrolled and both index maps folded to
immediates. There is no residual index arithmetic.

The corpus quotes FFTW's `n1_9` at 80 instrs / 136 flops, 8 instructions better
than my 88. I tried three ways to close that gap and all were worse — see below.

### Layout and SIMD decisions

* `VL` doubles per vector: **8 (zmm) when `__AVX512F__`, else 4 (ymm)**. One
  source, GCC vector extensions (`vector_size(64)` + `__builtin_shuffle`), so the
  AVX-512 build is *the same code* that was verified numerically here, not a
  separately hand-written intrinsics path. `-DFFT_VL=4|8` forces either width;
  both were run and both give 3.8e-16.
* Live vectors: the 4-point stage needs 8; the 9-point stage needs 18 (`g[9]`
  complex) + ~8 temps ≈ 26–30. That fits 32 zmm; on a 16-register AVX2 host it
  spills hard (see below). The 4×9 split *is* the register-pressure fix that
  §01 §7.2 predicts — a fused 36-point kernel would need ~72 vectors.
* The 4↔9 stage boundary needs a 36-complex transpose that cannot live in
  registers: it is a 36+36-vector stack array `Ar/Ai` (4.6 KB at VL=8, L1-hot,
  reused by every codelet). Cost 144 L1 ops per codelet on top of the 496 FP ops.
  This is irreducible: stage 2 for any `k1` needs *all nine* stage-1 outputs.
* `out` is written as one fully sequential stream: per z-tile the 8 result
  pencils are 288 consecutive complex = 4608 contiguous bytes, and every store is
  64-byte aligned (`(t*36+kz)*2` doubles, and 72 = 9·8).
* Every load and store in the whole file is naturally 64-byte aligned except the
  `in` deinterleave reads, which are 16-byte aligned and use an `aligned(8)`
  vector typedef.

### What was measured

Dev host = **Haswell Xeon E5-2680 v3, 2.5 GHz, AVX2 only, 16 ymm, 256 KB L2,
30 MB L3**. This host runs the `VL=4` path natively; the `VL=8` path is *emulated*
as two ymm halves and its local time is meaningless (register file too small).
Per-transform, `min` over 16 samples, `gcc 11.4 -O3 -march=native`:

| build | B=1 | B=4 | B=32 | B=64 |
|---|---|---|---|---|
| VL=4 (native AVX2 here, = default build on this host) | **283.1 µs** | **284.1 µs** | 402.3 µs | **386.5 µs** |
| VL=8 (emulated as 2×ymm, time NOT meaningful) | 868.6 µs | 881.4 µs | — | 928.3 µs |

(best of two 20-sample runs; run-to-run spread ~2 %, sd within a run 0.5–2 %.)

Correctness (`check.py` vs numpy): **rel_l2 = 3.83e-16, rel_max = 5.1e-16** at
B=1, 4, 32, 64, for *both* VL=4 and VL=8. Repeated `fft3d_execute` on one plan is
**bit-identical**, and `in` is verified unmodified (separate harness).
`fft3d_create` ≈ 0.3 ms (two `posix_memalign` + `memset`).

Phase breakdown, VL=4, per volume (measured by compiling with passes stubbed out):

| phase | µs/volume | what it moves |
|---|---|---|
| load + deinterleave | 39 | reads `in`, 746 KB, 128-B chunks at 576-B stride |
| x pass | 53 | pure compute; P is L2-resident |
| y pass + transpose + store T | 108 | + 746 KB T write (+RFO) |
| z pass + transpose + store out | 104 | + 746 KB T read, 746 KB out write (+RFO) |

So on this host the transform is **memory-bound, not FP-bound**: 3 × 53 = 159 µs
of codelet work against ~4.5 MB of L2↔L3 traffic per volume (in 746 KB R +
T 830 KB W + T 830 KB R + out 746 KB W + 1.6 MB of RFO) at the ~22 GB/s this
core sustains to L3. That 4.5 MB is the floor for *any* two-pass 3D FFT that does
not fit in cache.

**Node prediction, stated before the monitor measures it, so it can be scored.**
Gold 5218 is a Gold-5xxx, i.e. **one AVX-512 FMA unit**, so 512-bit
add/mul/FMA all issue 1/cycle on port 0 while 256-bit ops issue 2/cycle on
ports 0+1 — AVX-512 buys *no* FP throughput on this SKU, only half the µops,
half the L1 accesses and 32 registers. Port-0 bound at VL=8:
`522 codelets × 496 = 258 912 cycles = 113 µs @2.3 GHz` (fewer if turbo holds).
The node's **1 MB L2** is the real difference from this dev host: T (830 KB) +
P (166 KB) = 996 KB, so T has a real chance of staying in L2 across the barrier,
which would cut node L3 traffic from 4.5 MB to ~2.2 MB. **Expect 110–160 µs at
B=1** against MKL's 163.6 µs — a win, but not a large one, and it hinges on that
L2 residency.

### What was tried and did NOT work — with the number that killed it

1. **`T[z][transverse]`, the obvious "z-major plane" layout.** The y pass then
   writes 4 (VL=4) perfectly sequential streams — fine — but the z pass reads
   **72 concurrent strided streams** (36 z × re/im, stride 1296 doubles). z pass
   452 µs (B=4). Blocking it as `T[tile][z]` fixed the read (428 µs) but wrecked
   the write: y pass **411 → 616 µs**, because the write became 128-B chunks at
   1152-B stride. Only the three-level form `T[slab][tile][lane]` gets both:
   the write is *one* sequential stream, the read is `NSL` sequential streams.
   y 473, z 415. **Do not "simplify" this layout back.**
2. **Loading each z-slab straight out of `in` without software prefetch.**
   Slab `s` reads 128 B of every 576 B, which is a constant stride that the L2
   streamer will not follow across page boundaries: **223 µs → 158 µs (B=4)**
   with `__builtin_prefetch(q + 12*72, 0, 0)`. Prefetch distance swept 4…64,
   flat minimum at 10–12 pencils, 24+ is worse. This one change is 5 % of the
   whole transform.
3. **Non-temporal stores on `out`** (§05 §8.2 says NT on the final write only).
   Measured **1268 µs vs 1238 µs (B=4), i.e. 2 % *worse***. Reason: in the
   driver's repeat loop the whole footprint (in+out+T+P = 2.5 MB) is L3-resident,
   so ordinary stores never touch DRAM and the RFO is an L3 hit, whereas NT
   forces a 746 KB DRAM write every execute. Left in behind `-DWANT_NT` because
   the calculus flips if `T`'s L2 residency on the node turns out to matter more
   than the DRAM write — that is the one experiment I could not run.
4. **Read-prefetching the T stream in the z pass** (`ZPF`): 1244 → 1260 µs,
   *worse*. It is already one sequential stream per slab; the extra µops cost
   more than they save. **Write**-prefetching the T stores in the y pass is the
   opposite: 1244 → 1173 µs (**−5.6 %**), kept, `TPF = 8` cache lines.
5. **Sizing T exactly (746 KB) instead of the padded 830 KB.** Tried; produced
   `rel_l2 = 5.4e-1`. The slab-blocked layout structurally reserves VL lanes per
   slab, and slab 4 only has 4 valid z, so the exact size is not reachable
   without a per-slab base pointer that `SOFF` cannot express. Reverted. The
   84 KB of padding is 8 % of the node's L2 and is the most annoying open item.
6. **Closing the 88 → 80 instruction gap on the 9-point module.** Three attempts,
   all counted worse: (a) FMA-accumulating `t = w1·G1 + w2·G2` and
   `q = w1·G1 − w2·G2` directly from the untwiddled values: 16 instrs for t,q
   vs 8 (two complex multiplies) + 4 (adds) = 12 — **+4**; (b) folding `C3` into
   the twiddles so that `C3·t` comes out of the same FMA chain: needs both `t`
   and `C3·t`, so nothing is saved; (c) folding the twiddle into the *producing*
   first-layer DFT3 output: 8 instrs vs 2 (FMA) + 4 (complex multiply) = 6 —
   **+2**. Also: a Winograd/Burrus 9-point trades multiplies for adds and lands
   near 112 *instructions* even though it has fewer flops — on FMA hardware that
   is a loss, exactly as §02 §2.7 and §03 §3.3 warn. 88 stands.
7. **Fusing the deinterleave into the x pass** (so `in → registers` instead of
   `in → P → registers`). Counted, not built: the op count is *identical*
   (36·(2 ld + 2 shuf) + 72 ld = 288 either way), the only gain is 332 KB/slab of
   **L2** traffic ≈ 11 µs, and the price is turning the `in` read into a
   20736-B stride. Judged not worth the risk. Someone with more time should
   re-count this.
8. **Vector-radix / 6-D `4×9×4×9×4×9` factorisation** (§02 §5.5 item 3,
   §03): not attempted, and §03's own conclusion is "don't". It increases the
   number of passes over memory, which is the binding constraint here.
9. **`gcc-12` (1236 µs) and `clang-18` (1255 µs) vs `gcc-11` (1238 µs)**: no
   difference worth a compiler request. A `__builtin_shufflevector` shim is in
   the file so clang builds, since clang lacks `__builtin_shuffle`.

### Known waste, quantified

* 36 = 4.5 × 8. At VL=8 the z-slab loop runs 5 slabs of which the last uses 4 of
  8 lanes, so the **x and y passes do 11 % dead arithmetic** (522 codelet calls
  instead of 486). The z pass has none (1296/8 = 162 exactly). At VL=4 there is
  no waste anywhere (36 = 9×4) — which is why VL=4's port-bound estimate
  (241 k cycles) is actually *better* than VL=8's (259 k) on a 1-FMA SKU.
  Whether VL=8's halved µop count and halved L1 traffic beats that is the
  open question the node will answer.

### Next

1. **Flip `FFT_VL` to 4 and re-measure on the node.** One token. On a 1-FMA
   Gold 5218 the two widths are within 7 % on the FP ports by the model above,
   AVX2 does not downclock, and VL=4 has zero padding waste. If the node reports
   VL=8 near 113 µs, VL=4 is the next thing to try; if VL=8 comes in near 60 µs
   the SKU has two FMA units and this whole paragraph is void.
2. **Kill the last-slab waste** by instantiating the codelet a second time on a
   4-wide vector type for slab 4 only. 256-bit ops issue 2/cycle where 512-bit
   issue 1/cycle, so that slab gets ~2× cheaper: ~8 µs, ~6 %.
3. **Attack the 4.5 MB.** The only structural lever left is keeping T in the
   node's 1 MB L2: fix item 5 above (exact 746 KB T), and consider
   `prefetchnta`-only reads for `in` so the input stream does not evict T.
4. **Huge pages** (`madvise(MADV_HUGEPAGE)` on T and P in `fft3d_create`) —
   §05 §7 argues this is unconditional at L=36 and it is free to try. Not done.
5. Batched B≫1 is DRAM-bound (397 µs/transform at B=64 locally vs 291 at B=1);
   nothing here is tuned for it. Using the batch index for the *lanes* would
   remove all three transposes and the 11 % padding waste at once, and is
   probably the right answer for large B — but it is a different program.

---

## Round panel_r2

### Where round 1 stood, and what this round is

Node, panel_r1, B=1: **225.3 µs — last of the three L36 entries** (L36_mixedradix
118.4, L36_pencilfused 194.2, MKL 163.6). My round-1 node prediction (110–160 µs)
was wrong by 40%: the three-pass structure through the 830 KB slab-blocked `T`
pays a full extra volume round-trip that the winner simply does not have, and no
amount of codelet quality buys that back. At B=256 everyone lost to MKL (246.3);
my 312.0 was 27% behind.

So this round is a **structural rewrite, not a tune**: I kept the PFA 4×9
mathematics and threw away my pass structure in favour of the one that won.

### Technique (round 2)

Same Good–Thomas 4×9 index maps as round 1 (still folded into compile-time
addressing, zero twiddles between stages), but now on **interleaved complex**
vectors (PW complex per vector, lanes always a spectator axis) in **two sweeps**:

```
phase 1, per x-plane (36×36 complex = 20.25 KB, L1-resident):
    z transform  lanes = PW y-rows, PW×PW complex-granule register transposes
                 on load and store (the one unavoidable transpose pair)
                 into a stack plane pl[y][kz]
    y transform  lanes = PW kz (contiguous in pl), store to mid[x][ky][kz]
phase 2:
    x transform  lanes = PW kz, stride 20736 B between the 36 x-streams,
                 mid -> out, results stored straight from registers
```

with `mid` selected at plan time from two options:

* **in-place** (`mid = out`): smallest resident set, 746 KB, fits the node's
  1 MB L2 — the L36_mixedradix arrangement, best at small B;
* **reused scratch + NT** (`mid = S`, one plan-owned volume reused for *every*
  volume of the batch, phase 2 writes `out` non-temporally): S stays
  cache-resident across the whole batch, so at large B the DRAM traffic per
  volume collapses to the compulsory read-`in` + write-`out` = 1.5 MB, where
  in-place pays read + RFO + writeback ≈ 2.2 MB. This is the piece neither
  rival has: mixedradix rejected a scratch on "identical traffic" grounds
  (their round-1 item 7) — true per volume in isolation, **false when the
  scratch is reused across the batch**, because a reused scratch never costs
  DRAM and it frees the final write to be non-temporal.

`fft3d_create()` times {PW=2 ymm, PW=4 zmm} × {inplace, scratch, scratch+NT}
on a private arena, verifies every candidate against a reference to 1e-13
relative before it is eligible, and installs the winner as (pw, mode). Setup
0.02 s at B=1, ~0.4 s at B=256 — excluded from the score.

### Attribution (what I borrowed and from whom)

* **Two-sweep plane-fused structure, interleaved-complex lanes, and the
  6-op DFT3 / 8-op DFT4 / 2-op CMUL instruction forms — from L36_mixedradix
  round 1.** I verified their claim that split vs interleaved is exactly equal
  on FMA-port pressure (my round-1 SoA codelet: 496 ops / 8 lanes = 62 per
  line; theirs: 248 / 4 = 62) — so my round-1 layout bought nothing and cost a
  deinterleave pass plus 11% zmm lane waste (36/8 = 4.5 vs 36/4 = 9 exact).
* **NT stores on the final write at large batch, and the create-time
  correctness interlock over variants — from L36_pencilfused round 1** (their
  measured 1.53× at B=32 from killing the write-allocate).
* Everything about the reused-scratch-plus-NT combination, and the tuner
  choosing (pw, mode) jointly per batch size, is new this round.

### Operation count

Per 36-point line over PW interleaved lanes: 9 DFT4 (8 ops) + 4 DFT9
(6 DFT3 × 6 + 4 CMUL × 2 = 44) = **248 FMA-port ops + 49 swaps (port 5)**.
3888 lines/volume → 241 056 FMA-port vector ops at PW=4. Transposes: 144
port-5 shuffles per z-pass line-group (324 groups) ≈ 47k, on top of 48k swaps
— port 5 stays far below port 0. On a 1-FMA Gold 5218, 512-bit FMA issues
1/cycle → **the arithmetic floor is ~241k cycles ≈ 105 µs at 2.3 GHz**, and
256-bit at 2/cycle has the *same* floor, so the pw2/pw4 choice on the node is
purely front-end + licence-clock, which is what the tuner measures.

**Verified in the object code under the monitor's exact flags** (-O3
-march=... -std=gnu11, no -funroll-loops): phase1_pw4 and phase2_pw4 each
contain exactly 496 = 2×248 arithmetic ops (two codelet sites each), ≤5
integer index instructions, ~80 stack spills. The model is exact; FMA
contraction via explicit `_mm512_fmadd_pd` (a plain-expression DFT3 would have
CSE'd the shared product and emitted 7 ops, not 6 — write the FMAs explicitly).

### What was measured (wallaby, Sapphire Rapids Gold 6448Y)

**Caveat: wallaby was NOT near-idle this round** — it toggled between a fast
and a ~2×-slower state across minutes (load from other panel agents;
min 50.9 vs median 102.5 within one run, sd 24%). Within-process comparisons
(the tuner's, and min-of-run in the fast state) are still meaningful;
cross-run deltas under ~10% are not.

Per transform, driver min, rel_l2 vs numpy 3.64–3.65e-16 at every batch:

| B | µs/vol (fast state) | tuner's pick | MKL same case |
|---|---|---|---|
| 1 | **50.9** | pw4, in-place | 74.7 |
| 4 | ~120 (noisy window) | pw4, in-place | ~100–182 (noisy) |
| 32 | **81.0** | pw4, scratch+NT | 106.4 |
| 256 | **130.7** (later windows 146.7) | pw4, scratch+NT | 157.1 |

Round 1 on this same host class was not measured (round 1 developed on the
Haswell), but vs round 1's *node* 225 µs the structure is transparently ~2×.
Tuner internals at B=1 (one process, slow state, so relative only):
pw4-inplace 104.3, pw4-scratch 128.4, pw4-NT 152.0, pw2-inplace 141.7 —
in-place beats scratch by 23% at B=1, exactly the L2-residency argument.
At B=256 forcing modes end-to-end: NT 126.6 vs in-place 189.2 µs/vol (+49%),
the reused-scratch+NT design doing what it was built for; it also beat MKL's
157 on the same case, which no L36 entry managed at B=256 on the node in r1.

Phase split (diagnostic builds, fast state, pw4 in-place B=1): phase 1 ≈ 70%,
phase 2 ≈ 25% — phase 1 carries 2/3 of the codelets plus all the transposes.

### What was tried and did NOT work — with the number that killed it

1. **Tuning arena of 16 volumes.** The tuner initially mis-picked cached
   stores over NT at B=256: a 16-volume arena (24 MB in+out) still fits
   wallaby's 60 MB L3, so NT looked 12% slower in the tuner while being 40%
   faster in the real 384 MB run (45.5 ms vs forced-NT 32.4 ms per call).
   Fixed by capping the arena at 64 volumes (96 MB): tuner then picks NT and
   the full run hits 33.4 ms. **If you autotune a streaming-regime decision,
   the tuning set must actually stream on the machine with the largest L3
   you will meet.**
2. **Dropping the phase-2 prefetch** (36 software prefetches, one line ahead,
   one per x-stream per tile): B=1 50.9 → 59.1 µs (**−14%**), B=256
   130.7 → 136.3. 36+36 = 72 concurrent streams is beyond the L2 streamer;
   keep the explicit prefetch. (Matches mixedradix's finding; distance 1 line
   was their best too, and I saw no sensitivity worth a sweep in the noise.)
3. **Plain-expression FMAs in DFT3/DFT4** (letting -ffp-contract do it): gcc
   CSEs the twice-used product and emits mul+add+sub (7 ops) instead of two
   FMAs (6). Caught by counting the disassembly, worth ~6% of arithmetic;
   the explicit-intrinsic forms are load-bearing, do not "clean them up".
4. **Staging phase-2 results in a buffer then storing** (the first version):
   direct stores from the codelet's stage 2 are safe even in-place (all 36
   loads precede the first store) and save 72 L1 ops per tile. Measured only
   in the noise band (~1–2%), kept because it is free. The PW=2 NT path still
   stages, deliberately: two z-blocks are paired so every NT write completes
   a full 64-byte line (half-line NT stores thrash the WC buffers).
5. Round 1's whole three-pass + slab-blocked-T architecture, in retrospect:
   the T round-trip plus deinterleave was ~90 µs of node time that phase
   fusion simply deletes. The lesson is in the round-1 record's own phase
   table — 108 of 304 µs were the T write and read — I just did not act on it.

### Open questions for the node (state predictions so they can be scored)

* **pw2 vs pw4 on Gold 5218**: identical FMA-port floor; pw4 halves the µop
  count, pw2 avoids the AVX-512 licence clock. On wallaby (2 FMA units,
  no licence penalty) pw4 wins by 26–36%. My prediction for the node: **pw4
  still wins** — the licence drop on mostly-512-bit code should cost less
  than 2× front-end — but it is the tuner's call, and whichever way it goes
  is the §4.8-gap-6 data point the corpus lacks. Read it out of the
  leaderboard's variant, or run with FFT36_VERBOSE=1 / -DFFT36_LOUD.
* **B=1 node time**: structure now matches L36_mixedradix r1 (118.4 µs) with
  equal op count; I expect **110–125 µs**, i.e. parity or slightly better
  (their r1 had no phase-2 direct stores), and the batched cases to move a
  lot: B=32 from 294.6 to **~150–180**, B=256 from 312.0 to **~170–200 and
  ahead of MKL's 246**, on the strength of scratch+NT alone.

### Next

1. **Phase 1 is the remaining 70%.** The z-pass carries 144 transpose
   shuffles + 72 extra L1 round-trips per line-group. Two candidates:
   (a) software-pipeline two yb-groups so the transpose of group g+1 overlaps
   the DFT9 stage of group g (mixedradix's untried item 2); (b) write the
   z-pass output transposes straight into `pl` (TRNC with an lvalue-macro
   destination instead of staging through r_[]), and the input transposes
   straight from `px`. gcc already kills most of the staging (78 stack movs
   total), so measure before believing either.
2. **If the node tuner picks pw2**, revisit the PW=2 z-pass: at PW=2 the
   granule transpose is 2 shuffles per 2 vectors and the plane fits twice as
   many lane-groups — the balance between the phases shifts.
3. **The 88→80 DFT9 gap is still there** (FFTW n1_9, 4 modules × 2 ops × 972
   groups ≈ 3% of the transform). Round 1 burned three attempts on it; only
   worth it via transcribing genfft's actual DAG, not by hand.
4. At B=4 the working set (5.7 MB in+out) fits the node L3 but not L2; the
   tuner saw in-place and scratch within noise there. If the node disagrees,
   trust the node.

---

## Round panel_r3

### Where round 2 stood, and what this round is

Node, panel_r2: **first in all four L=36 cells** (B=1 119.266, B=4 128.460,
B=32 202.746, B=256 238.796 µs; 1.28–1.36× ahead of MKL). The monitor's verdict
quantified the remaining gap precisely: B=256 − B=1 = 119.5 µs/volume of
**un-overlapped memory time**, against compulsory traffic of only 1.49 MB/volume;
other entries sustain 10.5–12.1 GB/s single-core streaming on the same node
(L6_unrolled B=32768, L8_batchsimd B=16384) while I ran at 6.25 GB/s effective.
Ceiling if reads/writes hide under compute: ~123 µs/volume, i.e. ~1.9× at B=256
— "the largest single quantified gap on the whole board."

So this round is **memory-level parallelism only**: no arithmetic change, no
layout change, no new pass structure. Two prefetch mechanisms plus a tuner
dimension.

### Technique (round 3 delta)

1. **Paced software prefetch of the phase-1 input stream** (`PFIN`). Phase 1
   reads `in` perfectly linearly over the volume (planes consecutive, rows
   consecutive within the yb-subloop), but on the node those reads were demand
   misses serialized against the codelet — the L2 streamer does not carry a
   stream across 4 KB page boundaries (same mechanism L8_batchsimd documented).
   A cursor runs `FFT36_PFD = 4096` doubles (32 KB) ahead of the plane being
   consumed; each of the 2·NVR loop iterations per plane (both the yb z-subloop
   and the zb y-subloop, which touches no `in` bytes at all) issues
   `PFSTEP = 36·PW` doubles = 18 (PW=4) / 9 (PW=2) line-prefetches and advances
   the cursor, so exactly one plane's prefetches issue per plane processed and
   the DRAM read stream stays busy through the y-transform's compute-only
   stretch. Pacing detail that matters: the yb-subloop consumes `in` at 2× the
   cursor rate and the zb-subloop at 0×, so the cursor falls up to 10.4 KB
   behind mid-plane — PFD must exceed that deficit, which kills very small
   distances structurally (16 KB works on wallaby but leaves only 6 KB of true
   lead; 32 KB is the safe choice for the node's longer DRAM latency).
2. **Cross-volume pre-coverage from phase 2** (`PFNX`). The cursor scheme
   leaves the first PFD+deficit bytes of each volume cold. Phase 2 of volume b
   prefetches `FFT36_PFN = 3` lines per tile of `in[b+1]` — 324 tiles × 192 B
   = 62 KB at PW=4, exactly covering the cold window — so phase 1 of the next
   volume never starts against raw DRAM latency. Phase 2 is store-bound with
   idle read bandwidth, which is where this traffic belongs. Last volume gets
   `pnext = NULL` (three predictable branches per tile, free).
3. **`pf` is a tuner dimension**, not a policy: the create-time tournament is
   now {pw2, pw4} × {inplace, scratch, scratch+NT} × {pf0, pf1} = 12 candidates,
   every one correctness-gated at 1e-13 before it may win, same 64-volume
   arena cap as r2 (the arena must stream). At B=1 the tuner rejects pf (the
   ~1.3k extra prefetch µops/volume buy nothing when `in` is L3-resident);
   at B=256 it takes it. Overridables for the monitor: `FFT36_FORCE_PF`,
   `FFT36_PFD`, `FFT36_PFH` (prefetch hint), `FFT36_PFN`.

Prefetch hint is **T1** (`__builtin_prefetch(..., 0, 2)`, fills L2 not L1):
measured better than T0 (which churns L1 against the 20.25 KB plane buffer)
and far better than NTA — see below. µop cost: 11 664 prefetches + ~1 000
next-volume prefetches per volume ≈ 4% of the ~30k-vector-op volume, on
otherwise idle load-port slots.

### Attribution

The shape of the fix — prefetcht-class hint, *coverage and distance* as the
variables that matter, hooked into a compute pass, crossing the volume
boundary — is **L6_pfa's panel_r1 `fused+nt+prefetch` variant as re-proven by
L6_unrolled in r2** (their 1.4–1.9× at DRAM batch sizes, and their "prefetch
'did not work' in r1 because distance and coverage were wrong" correction).
Their NTA catastrophe (0.53–0.65 vs 0.19–0.20 µs/vol, "do not re-try NTA at
L=6") steered me to measure NTA once and default to T1. The monitor's r2
verdict supplied the target and the 123 µs ceiling arithmetic.

### Operation count

Unchanged from r2: 248 FMA-port vector ops per 36-point line, 241 056 per
volume at PW=4, floor ~105 µs at 2.3 GHz on a 1-FMA-unit Gold 5218. This round
adds zero arithmetic (prefetches only). Verified under the monitor's exact
flags (`-O3 -march=cascadelake -std=gnu11`): builds clean; disassembly shows
180 `prefetcht0` (the r2 phase-2 source-stream prefetches, 36 × 5 sites) + 66
`prefetcht1` (the new paced-stream and next-volume prefetches).

### What was measured (wallaby, Sapphire Rapids Gold 6448Y, quiet windows)

Driver min per transform, rel_l2 vs numpy 3.644–3.654e-16 at every batch,
bit-identical re-runs, `in` unmodified; AVX2-only build exercised end-to-end
on the Haswell login node (PASS 3.651e-16 at B=4):

| B | r2 code (baseline, same host/day) | r3 code | tuner's pick | Δ |
|---|---|---|---|---|
| 1 | 52.7 µs | **51.9 µs** | pw4, inplace, **pf=0** | unchanged, by design |
| 4 | — | **72.2 µs** | pw4, inplace, pf=0 | wallaby-cached regime |
| 32 | — | **80.5 µs** | pw4, inplace, pf=0 | fits wallaby's 60 MB L3 (not the node's 22) |
| 256 | 137.6 µs | **104.4 µs** | pw4, scratch, **pf=1** | **−24%** |

Within one tuner run at B=256 (same arena, back-to-back, load-immune): pf=1
beats pf=0 on *every* mode — pw4-scratch 140.4 → 101.9 (−27%), pw4-NT
159.0 → 117.3, pw2-scratch 170.7 → 117.3. MKL on the same case: 155.9 µs/vol
(quiet window); r3 code is 1.49× ahead of it on wallaby.

Knob sweeps at B=256 (full driver runs):
* **Hint**: T1 104.4 / T0 110.2 / **NTA 135.1** µs/vol — NTA erases the whole
  gain (matches L6_unrolled at L=6; now measured at L=36 too — on SPR, NTA'd
  lines are gone from every useful level before the yb-subloop returns).
* **Distance** FFT36_PFD: 2048 → 104.1, 4096 → 104.4, 8192 → 106.2. Flat
  minimum at 16–32 KB; shipped 4096 doubles = 32 KB for node-latency margin
  (16 KB leaves only 6 KB of true lead after the pacing deficit).
* **Next-volume depth** FFT36_PFN: 3 vs 6 vs 9 = 106.2 / 115.0 / 102.9 in a
  noisy window (sd 5–6%) — indistinguishable; shipped 3 (lowest µop cost,
  exact cold-window coverage).

Note wallaby picked cached-scratch over scratch+NT at B=256 this round
(101.9 vs 117.3 in-arena) — with reads prefetched, wallaby's fat memory system
absorbs the RFO. On the node (r2: NT beat inplace by 49% end-to-end) I expect
NT+pf to win instead; the 12-candidate tuner measures exactly that on the node
at create time, so the right answer installs itself either way.

### What was tried and did NOT work — with the number that killed it

1. **NTA hint on the in-stream** (mixedradix's r2 "Next" item 3, my own r1
   item): 135.1 vs 104.4 µs/vol at B=256 — the L2-pollution theory loses to
   the lost-before-use reality by 29%. Both L=6 and L=36 now carry this
   number; nobody should try NTA a third time on these parts.
2. **T0 hint**: 110.2 vs 104.4 (+6%). L1 churn against the plane buffer.
3. **Deeper next-volume coverage (PFN=6)**: 115.0 vs 106.2 in the same noisy
   window — never better than exact coverage, sometimes worse (read traffic
   pulled too far ahead of use evicts before phase 1 arrives).
4. **PFD=8192 (64 KB)**: 106.2 vs 104.4 — mildly past the flat minimum.
5. **Cross-volume ping-pong-scratch software pipelining** (interleave
   phase1(b+1) chunks with phase2(b) tiles, the literal reading of the
   verdict's suggestion): designed, then dropped *before* building, on
   arithmetic: it needs 2 scratch volumes = 1.5 MB resident, blowing the
   node's 1 MB L2 and demoting every scratch access to L3; and it fixes only
   the phase-boundary serialization, not the demand-miss serialization inside
   phase 1, which the schedule model says is the dominant term (phase 1 is
   read-bound: 62 µs of DRAM reads under 70 µs of compute; phase 2's NT
   stores already drain asynchronously). The prefetch scheme attacks the
   dominant term with no working-set growth. If the node result lands well
   short of ~150 µs at B=256, the pipeline is the next thing to build on top.

### Node predictions (stated so they can be scored)

* **B=256: 150–175 µs** (from 238.8). The wallaby gain was −24% with a much
  faster memory system leaving less to hide; the node has more exposed
  latency to remove but a slower sustained rate (~12 GB/s ceiling → 123 µs
  floor + imperfect overlap). Anything under 180 beats every r2 number and
  MKL's 307.8 by >1.7×.
* **B=32: 145–175 µs** (from 202.7) — same mechanism; its 45.6 MB working set
  streams on the node even though it cached on wallaby, and the node's tuner
  arena (32 volumes = 46 MB > 22 MB L3) sees the streaming regime correctly.
* **B=1: unchanged, 118–121 µs**; pf=0 should win the node tournament as it
  did on wallaby. If the node picks pf=1 at B=1, the interesting datum is by
  how much (L3-resident `in` there, unlike wallaby's noisy-window L2 story).
* **Mode at B=256: scratch+NT+pf** (unlike wallaby's cached-scratch+pf pick).

### Next

1. **If B=256 lands ≥180 µs on the node**: build the ping-pong pipeline
   (item 5 above) on top of pf — the two compose: prefetch fixes intra-phase
   stalls, interleaving fixes the phase-boundary burst structure. Costs L2
   residency; only the node can price that trade.
2. **If B=256 lands ≤150 µs**: the batched regime is within ~20% of its
   floor; switch effort to B=1, where 119.3 vs the 105 µs port floor leaves
   ~12%: the z-subloop's staged transposes (r2 Next 1b, still unmeasured) and
   the 88→80 DFT9 instruction gap via genfft's DAG (~3%) are the only levers
   left.
3. **Ask the monitor** (if the leaderboard doesn't show it): which (pw, mode,
   pf) the node tuner chose per batch — one `FFT36_VERBOSE=1` line. The
   pf0→pf1 in-arena delta on the node is the direct measurement of how much
   demand-miss serialization the Gold 5218 actually had; wallaby says 27%.

---

## Round panel_r4

### Where round 3 stood, and what this round is

Node, panel_r3: held **B=32 (218.351 µs, 1.01× over MKL — after a real +7.7%
regression from r1/r2's 202.7-204.7)** and **B=256 (227.497, 1.09×)**; second at
B=1 (120.042 vs mixedradix 118.626) and third at B=4 (129.242 vs 127.304), both
inside the run spread. The r3 verdict's orders, followed literally this round:
(a) `fft3d_description()` MUST report the tuner's pick — my B=32 regression was
undiagnosable because it didn't; (b) B=256 is still the board's largest prize
(~124 µs bandwidth ceiling vs 227.5 measured, 1.83×) and blind prefetch variants
are exhausted; (c) the process lesson from L8: **add candidates, never replace
structures** — the two entries that shipped both structures as tuner candidates
hit every prediction, the one that replaced lost three cells.

### Technique (round 4 delta) — no arithmetic change anywhere

1. **`M_PIPE`, the cross-volume ping-pong pipeline, as a NEW tuner candidate**
   (the design I costed and deferred in r3 item 5). Two plan-owned scratch
   volumes S0/S1. Phase 1 and phase 2 both have exactly 36 outer iterations
   per volume, so they interleave at plane granularity with zero restructuring
   of either kernel:

   ```
   prologue: phase1(in[0] -> Sa)            (36 x-planes)
   for b in 0..B-1:
       for u in 0..35:
           phase2_yplane(Sa -> out[b], y=u, NT stores)
           phase1_plane (in[b+1] -> Sb, x=u, paced prefetch)   # if b+1 < B
       swap(Sa, Sb)
   ```

   Why: the r3 node result says the compulsory read stream (phase 1) and the NT
   write stream (phase 2) still run as *alternating full-rate bursts* — phase 2
   needs ~20 GB/s of write for its 30% of the time while the read stream idles,
   which is why prefetch alone bought 4.7% on the node against 24% on wallaby.
   Interleaved, each stream runs at ~6.3 GB/s across 100% of the time; the sum
   (12.6 GB/s) is exactly the node's demonstrated single-core rate, i.e. the
   ~124 µs ceiling becomes reachable in principle. Phase 2 leads inside each
   unit so its NT stores drain asynchronously under phase 1's compute. PFNX
   (next-volume pre-coverage) is off in pipe mode — phase 1 of b+1 is itself
   touching in[b+1] concurrently, superseding it. Cost: 1.5 MB of scratch (does
   NOT fit the node's 1 MB L2 — the reason r3 deferred it), plus one extra
   746 KB allocation in the plan. It ships as 4 more candidates
   ({pw2,pw4}×{pf0,pf1}), correctness-gated at 1e-13 like everything else, so
   the node's own tuner prices the L2-residency trade — the only machine that
   can.
2. **Tuner pick reported**: `fft3d_description()` now returns
   "…tuner pick: pw=%d mode=%s pf=%d (B=%d, nv=%d)" filled at create() time
   (static-buffer mechanism borrowed from **L36_mixedradix r2 / L6_pfa**).
3. **Hysteresis**: among all correct candidates within **3%** of the best
   in-arena time, the *simplest* installs (rank: pf=0 before pf=1, then
   inplace < scratch < scratch+nt < pipe). Directly addresses (i) the verdict's
   tuner-instability finding (coin-flips cost 3.9–6.7% elsewhere) and (ii) the
   leading hypothesis for my own B=32 regression — pf=1 winning a marginal
   in-arena comparison and losing end-to-end. Measured on wallaby: at B=32 the
   pf=1-inplace vs pf=0-inplace gap was 2.4% (a coin-flip regime; now resolves
   to pf=0), while every genuine win (pf at B=256: −11%; pipe vs scratch+nt
   in-arena: −11%) is ≥10%, so the band gives up nothing real.
4. **Run-time forcing for the monitor**: `FFT36_PW`, `FFT36_MODE`
   (name or number), `FFT36_PF` environment variables filter the candidate list
   at plan time — the forced-variant control runs the verdict asked for at B=32
   need no recompile. Compile-time `FFT_FORCE_*` still works. Timing rounds
   3 → 5 (min-of-5) for pick stability; setup stayed ~1.0 s at B=256.

Mechanically, phase1/phase2 were split into per-plane functions
(`phase1_plane`, `phase2_yplane`) with thin 36-iteration wrappers; the paced
prefetch cursor and the PFNX cursor are recomputed per plane
(cursor(x) = in + PFD + x·2592 etc.), reproducing the r3 address streams
*exactly* — verified by the unchanged pf=0/pf=1 numbers and picks at B=1/B=4.

### Attribution

* Description-string plumbing: **L36_mixedradix** (g_desc mechanism, their r2).
* "Add candidates, do not replace structures": the **r3 verdict's** reading of
  **L8_fusedaxes/L8_radix8** (shipped both, hit predictions) vs
  **L8_batchsimd** (replaced, lost three cells).
* The pipeline itself is my own r3 item 5, finally built; the burst-vs-
  concurrent-streams arithmetic is the monitor's r2/r3 ceiling analysis.

### Operation count

Unchanged: 248 FMA-port vector ops per 36-point line, 241 056 per volume at
PW=4, ~105 µs port floor at 2.3 GHz. M_PIPE adds zero arithmetic — the same 72
plane calls per volume, reordered across the volume boundary.

### What was measured (wallaby, Gold 6448Y; windows varied — MKL swung 6%
between runs, so cross-run deltas <6% are soft)

All 16 candidates pass the 1e-13 create-time gate; end-to-end
rel_l2 = 3.644–3.654e-16 at B=1/4/32/256; bit-identical re-runs; AVX2-only
build verified end-to-end on wombat (PASS 3.651e-16, B=4).

| B | end-to-end µs/vol | pick | notes |
|---|---|---|---|
| 1 | **51.2** (fast window) | pw4, inplace, pf=0 | = r3's 51.9; nothing regressed |
| 4 | **74.6** | pw4, inplace, pf=0 | |
| 32 | **83.0** | pw4, inplace, pf=1→(3% band)→pf edge was 2.4% | L3-resident on wallaby, streams on node |
| 256 | **107.6** | pw4, scratch, pf=1 | MKL same window 212.7 → 1.96× |

In-arena at B=256 (one tournament, back-to-back): pipe-pf1 **101.4** vs
scratch+nt-pf1 113.8 (−11%) vs scratch-pf1 95.8. Forced end-to-end at B=256:
scratch+nt **97.5**, pipe 109.3, tuner's scratch pick 107.6 — i.e. wallaby's
**in-arena ranking inverts end-to-end** (NT worst in-arena, best end-to-end).
Cause: the 64-volume arena (96 MB) is only marginally streaming against
wallaby's 60 MB L3 — the exact effect my r3 record item 1 documented, one level
up the hierarchy. On the node the same arena is 4.4× its 22 MB L3 and genuinely
streams, so the node tournament is representative where this one is not. (Raising
the cap to 128 volumes would fix wallaby's fidelity at 2× the plan time; not
worth it — wallaby's pick is not the scored pick.)

### What was tried and did NOT work — with the number that killed it

1. **Judging the pipe on wallaby end-to-end**: forced pipe 109.3 vs forced
   scratch+nt 97.5 µs/vol at B=256 (−11% for NT). Expected in hindsight: the
   pipe's whole value is smoothing alternating DRAM bursts through a limited
   single-core queue, and wallaby's memory system doesn't saturate on either
   burst (it absorbed even the RFO traffic in r3). Both scratches also fit
   wallaby's 2 MB L2, so wallaby cannot price the node's L2-eviction cost
   either direction. This number is NOT evidence about the node; the node
   tournament decides. Recorded so nobody reads the wallaby delta as a verdict.
2. **A 2% hysteresis band**: at B=32 the pf=1-vs-pf=0 in-arena gap measured
   2.4% — precisely inside the coin-flip zone it was meant to catch — so 2%
   let the flip through. Widened to 3% after checking every genuine win on the
   board is ≥10%.

### Node predictions (stated so they can be scored)

* **Pick reporting**: every leaderboard cell now shows pw/mode/pf. If B=32
  shows pf=0 where r3's (invisible) pick was pf=1, the hysteresis fixed the
  regression; expect **B=32 ≈ 203–212 µs** (recovery to r1/r2 level ±noise).
* **B=256**: the interesting cell. If the node's tuner takes the pipe
  (pw4-pipe-pf1) AND the burst-serialization theory is right, **185–205 µs**;
  if it stays with scratch+nt+pf, expect **215–228** (r3 ±). I predict the
  pipe is selected at B=256 and not at B=32 (at B=32 the out stream is
  L3-marginal, weakening NT, and pipe is NT-based).
* **B=1/B=4: unchanged**, 118–122 / 127–131, pick pw4-inplace-pf0 in all three
  runs (hysteresis should also kill any pf coin-flip mixedradix-style).

### Next

1. If the node picks the pipe and lands ≤205 at B=256: the remaining gap to
   ~124 is scratch-traffic latency (S now lives in L3, not L2) — try a
   **half-volume pipe** (interleave at half-plane granularity with S split so
   the hot halves fit 1 MB L2) or software-prefetch the S read streams in
   phase 2 (they are 36 sequential streams; PF36 already covers the first line).
2. If the node rejects the pipe everywhere: the B=256 story is closed from
   this side; move to B=1, where the 88→80 DFT9 gap via genfft's DAG (~3%) and
   the z-subloop staged transposes are the only levers left (r2 Next 1).
3. The forced-variant control run at B=32/B=256 the verdict scheduled needs
   only `FFT36_MODE`/`FFT36_PF`/`FFT36_PW` env vars now — no recompile.

---

## Round panel_r5

### Where round 4 stood, and what this round is

Node, panel_r4: **first at B=32 (174.226 µs, −20.2%, the round's largest move)
and B=256 (218.899, −3.8%)**; second at B=1 (121.866 vs mixedradix 119.021,
itself +1.5% over my r3 121.9→120.0); third at B=4 (132.347, a real +2.4%
regression over r3's 129.242 that the verdict attributes to the per-plane
refactor — same pick string, 0.5% spread). The verdict's L=36 priority was to
*explain* the B=32 win. The pick strings in `results/panel_r4/c_L36_pfa_*.json`
already answer most of it: **the node chose `pw=4 mode=inplace pf=1` at BOTH
B=32 and B=256** — every scratch, scratch+NT and pipe candidate lost on the
node, in both streaming cells. r3's invisible config was (by its own record's
prediction) scratch+NT+pf, so the 20% at B=32 is mostly the *mode moving to
inplace*, which only became possible when r4's hysteresis ranked inplace
simplest. The r4 pipe machinery: built, gated, and rejected by the node
exactly as the tournament is designed to do.

That reading exposes the round's target. In INPLACE mode phase 1 stores to a
**cold `out` volume**: at streaming batch sizes every one of the 11 664
64-byte lines (746 KB) costs a demand read-for-ownership from DRAM that
nothing overlaps — the read stream is paced by PFIN since r3, but the write
stream has never been prefetched. Meanwhile L6_unrolled's r3 headline
(adopted by L6_pfa in r4, and *selected by the node* at every DRAM batch size
at L=6) is precisely this fix: `__builtin_prefetch(p,1,3)` emits `prefetchw`,
acquiring the line exclusive ahead of the store while keeping the
normal-store shape this node demonstrably prefers over NT (it has now
rejected NT at L=6, L=8 and L=36).

### Technique (round 5 delta) — no arithmetic change anywhere

1. **`pf=2`, write-intent prefetch, as a third pf level in the tuner.**
   Two mechanisms, each attached to the pass whose normal stores hit cold
   lines, so the level means something in every mode that has it:
   * **M_INPLACE (`PFWMID`)**: phase 1 paces a write-intent cursor over the
     mid==out store stream with the *same pacing arithmetic as PFIN* (18
     lines per loop iteration, 2·NVR iterations per plane = exactly one
     plane's worth per plane processed), at distance `FFT36_PFWD` = 2592
     doubles = one 20.25 KB plane. Lead per line: 18–27 iterations
     (0.5–1.5 planes ≈ 1.5–4 µs at node speed) — enough to cover a DRAM RFO,
     short enough that L2 (1 MB = 50 planes) never evicts a line between
     prefetchw and store.
   * **M_SCRATCH (`PFW36`)**: phase 2 write-prefetches its 36 out-streams
     one 64-B line ahead of the tile being stored, mirroring PF36 on the
     read side (out is the cold stream in scratch mode; S is warm).
   * NT and PIPE modes gain nothing (NT stores do not RFO), so pf=2 is not
     instantiated for them: candidate count 16 → 20 per tournament, all
     still gated at 1e-13 vs the reference before eligibility.
2. **Per-plane functions inlined away.** `phase1_plane`/`phase2_yplane` are
   now `always_inline`, so the non-pipe paths compile back to r4-r3's
   monolithic loop bodies (36×2 calls/volume with a 20 KB stack frame each
   are gone) while M_PIPE keeps its per-plane interleaving. This is the
   direct response to the verdict naming the r4 refactor as the only
   candidate cause of my +2.4% B=4 node regression.

Cost accounting for pf=2: 11 664 `prefetchw`/volume in phase 1 (inplace) or
11 664/volume in phase 2 (scratch, 36 per tile × 324 tiles) — ~4% extra µops
on otherwise idle store-port slots, zero flops. Knobs for the monitor:
`FFT36_PF=2` forces it, `FFT36_PFWD` moves the phase-1 write distance.

### Attribution

* **`prefetchw` on the cold output stream: L6_unrolled round panel_r3**
  (their `fused_pfw`, −29% at DRAM sizes, "first positive prefetchw datum in
  the corpus"), **as re-proven by L6_pfa in r4** (adopted it, won B=4096) and
  by the node's own picks (prefetchw variants selected at all L=6 DRAM
  cells). Their cache-resident warning — pfw was **17% WORSE** at L3-resident
  sizes on wallaby — is respected structurally: pf=2 is a gated candidate,
  never a default, and the measurements below confirm the tuner rejects it at
  B=1/B=4 exactly as their record predicts.
* The read-side pacing arithmetic that PFWMID reuses is my own r3 PFIN.
* "Add candidates, never replace structures" (r3 verdict) applied again: all
  r4 candidates survive unchanged; pf=2 is purely additive.

### Operation count

Unchanged: 248 FMA-port vector ops per 36-point line over PW lanes, 241 056
per volume at PW=4. Verified under the monitor's flags
(`-O3 -march=cascadelake -std=gnu11`): builds clean; disassembly carries 126
`prefetchw` sites alongside the r4 t0/t1 population (288/174); bare `-O2`
and Haswell `-march=native` also build clean (mixedradix's r4 latent-break
lesson checked against).

### What was measured (wallaby, Gold 6448Y — toggling fast/slow ~2× windows
all session, like r2; in-arena numbers are back-to-back and comparable,
cross-run deltas are not unless the window is quiet)

End-to-end, driver min; rel_l2 vs numpy 3.644–3.654e-16 at B=1/4/32/256;
bit-identical re-runs; AVX2-only build verified end-to-end on wombat
(PASS 3.651e-16 at B=4, bit-identical):

| B | end-to-end µs/vol | pick | window |
|---|---|---|---|
| 1 | **51.4** | pw4, inplace, pf=0 | fast (= r4's 51.2: nothing regressed) |
| 4 | **80.4** (min) | pw4, inplace, pf=0 | noisy, sd 17% — not comparable to r4's 74.6 |
| 32 | **71.4** | pw4, inplace, **pf=2** | quiet, sd 0.04% (r4 same host: 83.0 → **−14%**) |
| 256 | **101.6** | pw4, scratch, **pf=2** | sd 0.6% (r4: 107.6 → −5.6%); MKL same window 197.2 → 1.94× |

In-arena (one tournament each, back-to-back, the load-immune comparison):

* **B=256**: scratch-pf2 **84.8** vs scratch-pf1 97.7 (**−13%**);
  inplace-pf2 **90.5** vs inplace-pf1 156.6 (**−42%** — the phase-1 RFO was
  indeed the dominant exposed cost of the mode the node runs); pw2-inplace-pf2
  97.7 vs pw2-inplace-pf1 172.7. Every pf=2 candidate beats its pf=1 twin.
* **B=32**: inplace-pf2 **70.1** vs inplace-pf1 75.0 (−6.5%); scratch-pf2
  71.7 vs scratch-pf1 91.6.
* **B=1**: inplace-pf2 117.4 vs inplace-pf0 104.2 (+13%, cache-resident, as
  L6 warned) — tuner keeps pf=0. **B=4**: inplace-pf2 138.7 vs pf0 124.9 —
  rejected likewise. The hysteresis (unchanged, 3%) has nothing to do at
  either end: pf=2's wins and losses are all ≥6%.

### What was tried and did NOT work — with the number that killed it

1. **pf=2 at cache-resident batch (B=1/B=4)**: +13% and +11% in-arena (above).
   Not a surprise — L6_unrolled measured +17% in the same regime — but now
   confirmed at L=36: prefetchw on lines that are already in cache is pure
   µop tax plus L1 churn. The candidate structure (tournament-gated, never
   default) is what makes it safe to carry.
2. **Isolating the r4 per-plane call overhead on wallaby**: not resolvable
   this session — the host toggled 2× windows and the effect is ≤2.4% (node).
   The inline change is shipped on the verdict's attribution, is zero-risk
   (same address streams, verified bit-identical output at every batch), and
   the node's B=4 cell will score it.
3. Nothing else was touched. Deliberately narrow: one new mechanism, one
   restoration.

### Node predictions (stated so they can be scored)

* **B=256: 190–210 µs** (from 218.9). The node's inplace-pf1 has ~119 µs/vol
  of un-overlapped memory time (r2 verdict arithmetic still holds) of which
  the 746 KB RFO stream is roughly a third; wallaby's in-arena said −13/−42%,
  the node has slower DRAM (more latency to hide — bigger prefetch upside)
  but a lower bandwidth ceiling (~124 µs floor). Pick: inplace-pf2 or
  scratch-pf2 — wallaby chose scratch-pf2, but the node has preferred
  inplace in both streaming cells and inplace-pf2's RFO removal is the
  larger relative gain there.
* **B=32: 158–172 µs** (from 174.2), pick inplace-pf2, by the same mechanism
  at the −6.5% in-arena scale.
* **B=1: 119–122** (pick inplace-pf0, pf=2 correctly rejected); **B=4:
  128–132** — if the inline restoration recovers the refactor cost, B=4
  returns to ~129 and B=1 shaves ~1 µs; if B=4 stays at 132 the refactor
  attribution was wrong and the cause is elsewhere (worth one forced
  `FFT36_MODE=inplace FFT36_PF=0` control against the r3 binary if the
  monitor keeps them).
* If the node picks scratch-pf2 at B=256 while inplace-pf2 wins B=32, that
  is the L2-residency crossover (S + out working set vs 1 MB) moving with
  batch size, and it is real information, not tuner noise.

### Next

1. **`FFT36_PFWD` sweep on the node** (1296 / 2592 / 5184): the 2592 default
   is pacing arithmetic, not measurement — wallaby's windows were too noisy
   to resolve it. One env-var A/B per value at B=256.
2. If B=256 lands ≤195, the batched story is within ~1.5× of the 124 µs
   bandwidth floor and the marginal return drops; move to B=1, where the
   only levers left are arithmetic: the 88→80 DFT9 instruction gap via
   transcribing genfft's actual n1_9 DAG into interleaved-lane form (~3%,
   three hand attempts already failed in r1 — do not hand-derive a fourth),
   and the z-subloop staged transposes (r2 Next 1b, still unmeasured).
3. If pf=2 is rejected on the node everywhere despite the wallaby margins,
   measure whether `prefetchw` on Cascade Lake demotes to a plain prefetch
   (it is PRFCHW, not 3DNOW — it should not), e.g. one forced
   `FFT36_PF=2` vs `FFT36_PF=1` pair at B=256 settles it.

---

## Round panel_r6

### Where round 5 stood, and this round's diagnosis

Node, panel_r5: **first in all four cells for the first time** — B=1 120.358,
B=4 129.295, B=32 168.565 (−3.2%), B=256 182.598 (−16.6%, the pf=2 write-intent
prefetch doing what it was built for). Node picks: B=1/B=4 `pw4 inplace pf=0`,
B=32 `pw4 inplace pf=2`, B=256 `pw2 inplace pf=2` (pw2! — first node cell where
256-bit won; noted, not acted on). The r5 verdict's caveat stands: the 1.70×
margin at B=256 is partly MKL regressing unexplained; don't bank it.

The round's target follows from the corrected clock. The r4/r5 verdicts settled
the node at **~3.89 GHz sustained** (L6_unrolled's in-plan probe), so the B=1
port floor is 241k cycles / 3.89 GHz ≈ **62 µs** against 120.4 measured — 1.94×.
Where do the extra ~230k cycles go? Two machine-side accounting checks:

* **wallaby B=1 sits exactly on its port-5 model.** Gold 6448Y has two 512-bit
  FMA pipes (ports 0 and 5), and port 5 also serves every 512-bit shuffle.
  Port-5 load = 120.5k (half the FMAs) + ~95k shuffles/transposes ≈ 215k cycles
  ≈ 54 µs at ~4 GHz — measured 50.7–51.4. So wallaby B=1 is **port-5-bound**,
  and shuffle reduction is a wallaby lever, not a node lever (the node's port 5
  carries no FP: its port-0 floor is 241k and port 5 sits at ~95k, idle).
* **the node's B=1 difference from wallaby is one number: L2 = 1 MB vs 2 MB,
  against in+out = 1.5 MB.** On wallaby both volumes are L2-resident at B=1;
  on the node the sequential in-read (demand or T1 prefetch) allocates every
  line into L2 and evicts `out` mid-execute, so the in-place phase-1 stores
  RFO from L3, phase 2 re-reads from L3, and modified out lines write back —
  2–3 MB of L2↔L3 round trips per execute against 746 KB compulsory. This also
  explains why all three L36 entries cluster at 120–123 µs at B=1: they share
  the two-sweep structure, so they share the wall.

### Technique (round 6 delta) — no arithmetic change anywhere

1. **pf=3 and pf=4: NTA-hinted in-read prefetch**, to keep the read stream out
   of L2 entirely (prefetchNTA fills L1 and bypasses L2 on SKX-class cores) so
   `out` (in-place) or `S` (scratch) stays L2-resident across the execute —
   and at B=1, across executes: steady state would leave out's lines
   L2-modified, deleting the RFO *and* the writeback, collapsing L3 traffic to
   the compulsory 746 KB in-read. Mechanics: a constant-lead cursor
   (`FFT36_PFDN` = 512 doubles = 4 KB) paced at exactly the consumption rate
   inside the yb-subloop only (2·PFSTEP per iteration; the zb-subloop issues
   nothing), so the lead never swings — unlike the pf=1 T1 cursor, whose
   ±10 KB swing is fine for L2 but fatal for L1-resident NTA lines. pf=3 =
   NTA read + the pf=2 write-intent mechanism (streaming cells); pf=4 = NTA
   read alone (the B=1/B=4 bet, where out should be L2-M already — ranked
   SIMPLER than pf=3 in the hysteresis). Only for INPLACE/SCRATCH; PFNX off
   for both (cache-resident target). Candidates 20 → 28, all 1e-13-gated.
   The r3 NTA catastrophe (135.1 vs 104.4 at B=256) is not being re-tried:
   that was a 32 KB lead on DRAM-latency streams with nothing behind the L1
   drop; this is a 4 KB constant lead on L3-resident data — and it ships as a
   gated candidate, never a default.
2. **Self-warming tuner (adopted from L36_pencilfused r5,** their 167.4-vs-89.8
   phantom-penalty diagnosis): one untimed exec of each candidate before its
   timed reps, so every candidate is timed from its own steady-state cache.
   My rotation had exactly the flaw they documented — at nv≥8 each candidate
   is timed with R=1, and NT/pipe candidates flush `tout`, deterministically
   overcharging their successor every round. This matters doubly now: it is
   the difference between the node reading pf=3/4's B=1 steady state honestly
   or charging it a predecessor's cold cache. Setup cost ~2× (B=32: 0.75 →
   1.44 s on wallaby), excluded from the score.

### Attribution

* **Self-warming tournament: L36_pencilfused round panel_r5**, verbatim
  mechanism, their diagnosis.
* **3.89 GHz clock and the recomputed floors: L6_unrolled r4's probe** via the
  r4/r5 verdicts (also both rivals' r5 records, which flagged B=1's ~2× floor
  gap as the board's largest prize).
* The NTA read-stream idea itself is this file's, but the *pacing discipline*
  (constant lead, consumption-rate issue) is the lesson of my own r3 PFIN
  deficit arithmetic, applied to a 32 KB-smaller budget.
* Negative guidance used: L36_mixedradix r5's "wallaby cannot arbitrate store
  policy for the node" — extended here to "wallaby cannot price NTA's B=1
  prize at all" (its 2 MB L2 holds in+out at B=1, so there is nothing to fix
  on the machine I can measure).

### Operation count

Unchanged: 248 FMA-port vector ops per 36-point line, 241 056 per volume at
PW=4. pf=3/4 add 11 664 prefetchnta µops per volume (36/iteration, yb-subloop
only) and zero flops. Recomputed floors at the verdict's 3.89 GHz: port floor
~62 µs/volume; B=256's in-place traffic floor (2.2 MB at the ~12 GB/s
demonstrated single-core rate) ~175–183 µs — **B=256 measured 182.6 is AT its
own traffic floor**; only an NT-shaped structure could go lower and the node
has rejected NT three rounds running. B=256 is closed from this side; hence
the B=1 focus.

### What was measured (wallaby, Gold 6448Y; fast/slow ~2× windows toggling as
in r2/r5 — in-arena tables are back-to-back and comparable, cross-window
deltas are not; quiet-window runs marked)

End-to-end, driver min; rel_l2 = 3.644–3.654e-16 at B=1/4/32/256,
bit-identical re-runs everywhere; AVX2-only build verified end-to-end on
wombat (PASS 3.646e-16 at B=2, bit-identical):

| B | end-to-end µs/vol | pick | window |
|---|---|---|---|
| 1 | **50.7** | pw4, inplace, pf=0 | fast (= r5's 51.4) |
| 4 | **68.9** (sd 0.02%) | pw4, inplace, **pf=2** | quiet — best-ever B=4 here (r5: 74.6) |
| 32 | **71.2** (sd 0.07%) | pw4, **scratch**, pf=2 | quiet (r5: 71.4; mode flip is the self-warmed table) |
| 256 | **100.4** | pw4, scratch, pf=2 | (r5: 101.6) |

Findings, in-arena (back-to-back within one tournament):

* **B=4 quiet window: pf=2 beats pf=0 by 8%** (inplace 70.7 vs 76.9; pf=1
  74.9). r5's B=4 rejection of pf=2 (138.7 vs 124.9) was taken in a 17%-sd
  window and looks like a window artifact. With the self-warmed table the
  node's B=4 tournament may now flip to pf=2 — worth watching.
* **NTA (pf=3/4) loses on wallaby in every regime wallaby can see**, with the
  numbers in the next section. All eight new candidates pass the 1e-13 gate
  everywhere (the mechanism is correct, just unprofitable on this memory
  system).
* Self-warming moved wallaby's B=32 in-arena from inplace-pf2 71.7 / scratch-pf2
  75.3 (r6 pre-fix) to scratch-pf2 69.4 / inplace-pf2 72.3 — i.e. the r5-style
  table had been overcharging scratch (it follows an NT candidate in rotation).
  End-to-end B=32 unchanged (71.2 vs 71.4): on wallaby the two modes are near
  precisely because its L3 absorbs both; the node's call is the real one.

### What was tried and did NOT work — with the number that killed it

1. **NTA in-read at B=32 on wallaby**: pf=4-inplace 109.4 vs pf=0-inplace 95.7
   (+14% — worse than NO prefetch at all) vs pf=2's 71.7. NTA prefetch on this
   memory system is actively harmful when the stream comes from L3/DRAM at
   rate: NTA lines are inserted into L1 with a quick-evict policy, and a line
   dropped before use was never put in L2, so it is re-read from L3 — the read
   costs twice and protects nothing.
2. **Shortening the NTA lead to 1 KB** (FFT36_PFDN=128, B=32): pf=4-inplace
   135.7 — worse still (now too late: the demand loads outrun the prefetch and
   the L1-hit fraction collapses). 4 KB (512 doubles) was the best of
   {128, 256, 512} tried; 256 ≈ 512 within noise at B=4 (85.0 vs 82.0).
3. **NTA at B=4 on wallaby, the closest local proxy for the node's B=1 regime**
   (out[b] cycles out of wallaby's 2 MB L2 between executes exactly as node
   B=1 does with 1 MB): quiet window, pf=4-inplace 82.0 / pf=3-inplace 76.7 vs
   pf=2-inplace **70.7**. Even where NTA's L2-protection prize exists (phase-2
   re-reads), its L1 fill-buffer tax exceeds it on this machine. Caveat for
   the node, recorded honestly: the B=4 proxy CANNOT show the full node-B=1
   prize (at B≥4 out is cold each execute, so the RFO/writeback deletion —
   the big term — never materializes locally; at wallaby-B=1 in+out fit L2 and
   there is nothing to fix). The node's own nv=1 steady-state tournament is
   the only honest measurement of pf=4's actual bet, which is why it ships as
   a candidate despite losing every wallaby cell.
4. Nothing else was touched: no arithmetic, no layout, no mode changes. B=1's
   hot path (pw4-inplace-pf0) is instruction-identical to r5 (verified:
   wallaby B=1 unchanged at 50.7 and bit-identical output at every batch).

### Node predictions (stated so they can be scored)

* **B=1, the cell this round is aimed at.** If the L2-eviction diagnosis is
  right AND the NTA L1-tax stays below the L3-round-trip savings on Cascade
  Lake (slower L3, smaller L1 than wallaby — both cut the wrong way), the
  tuner picks `pw4 inplace pf=4` and B=1 lands **95–112 µs**. If the wallaby
  pathology transfers, pf=4 loses in-arena, the pick stays `inplace pf=0`,
  and B=1 reads 119–122 — a null result that would close the NTA idea for
  good with one line in the pick string. I genuinely cannot call it; the
  candidate structure means being wrong costs nothing.
* **B=4**: with self-warmed tables, pick moves to `pw4 inplace pf=2` and
  **124–129 µs**, or stays pf=0 at 128–131.
* **B=32/B=256**: picks `inplace pf=2` (possibly `scratch pf=2` at B=32 now
  that the table no longer overcharges scratch — if it flips AND improves,
  that was the self-warming, not noise), times flat ±3% (168–174 / 178–188).
  Both cells are at/near their in-place traffic floors at 3.89 GHz.
* Setup roughly doubles (self-warming); still ≤5 s at B=256.

### Next

1. **If pf=4 is rejected at node B=1**: the NTA route is closed (both L1-tax
   regimes measured, both machines). The remaining B=1 levers are then (a) ask
   the monitor for one `perf stat -e L2-misses,LLC-loads` at B=1 to size the
   memory term directly — if it is small, the 2× floor gap is front-end/spill
   and the fix is codelet scheduling, not caching; (b) the 88→80 DFT9 gap via
   genfft's n1_9 DAG (~3%, transcribe-don't-derive, r1 item 6 stands); (c) a
   PW=2 B=1 look — the node chose pw2 at B=256, and pw2 halves the transpose
   shuffle count per line while keeping the same port-0 floor.
2. **If pf=4 wins at node B=1**: sweep FFT36_PFDN (256/512/1024) there, and
   try pf=4 semantics for B=4 (out[b] re-residency across the 4-volume cycle
   is marginal: 3 MB working set vs 1 MB L2 — probably lost, but the tuner
   already prices it).
3. **FFT36_PFWD node sweep** (1296/2592/5184 at B=256) is still unrun from r5;
   one env-var A/B per value if the monitor has cycles for it.
4. Wallaby-only note for whoever inherits this file: wallaby B=1 is port-5
   bound (215k-cycle model matches measurement to 5%); if wallaby numbers ever
   matter for their own sake, transpose-shuffle reduction is where its B=1
   time is, not FMA count.

---

## Round panel_r7

### Where things stand, and what this round is

panel_r6 was **abandoned before its timing pass** (stale-runner incident; see
`results/panel_r6_abandoned_no_timing/WHY.md`), so the r6 code — the pf=3/4
NTA in-read candidates aimed at node B=1 — exists on disk (preserved as
impl_6) but has **never been measured on the node**. The standing leaderboard
is still panel_r5: first in all four L=36 cells (B=1 120.358, B=4 129.295,
B=32 168.565, B=256 182.598 µs). So the node's r7 run will score r6's bet and
r7's together; both entries' predictions below are therefore live at once.

This round's own change is one new candidate combination, from re-reading my
r6 arithmetic: **scratch + NT stores + NTA-protected in-read (`scratch+nt
pf=4`), which no round has ever fielded.**

### Technique (round 7 delta) — no arithmetic change anywhere

The argument, in three steps that were all already in this file separately:

1. **B=256 in-place is AT its own traffic floor** (r6 §op-count: 2.2 MB/vol
   × ~12 GB/s ≈ 175–183 µs vs 182.598 measured). Nothing tuned inside the
   in-place shape can move that cell any more. The only structure with a
   lower floor is scratch+NT: compulsory 1.5 MB/vol ≈ 125 µs.
2. **Every node loss NT has suffered (r3–r5, four rounds panel-wide) shares
   one unaddressed defect**: the T1/demand in-read fills L2 and continuously
   evicts the 746 KB scratch S — which alone is 3/4 of the node's 1 MB L2 —
   so S's phase-1 writes and phase-2 re-reads become ~1.5 MB/vol of extra
   L2↔L3 round trips that erase NT's DRAM-traffic advantage. (Wallaby never
   shows this: its 2 MB L2 holds S regardless, which is exactly why scratch
   modes over-perform there.)
3. **r6's pf=4 NTA in-read (L1-fill, L2-bypass) removes precisely that
   defect** — but r6 gated pf≥2 to INPLACE/SCRATCH on the "NT stores do not
   RFO, nothing to hide" argument. True, and the wrong filter: NT mode needed
   the READ-side protection, not the write-side. The combination was simply
   never instantiated.

New candidates `{pw2,pw4} × M_SCRATCH_NT × pf=4` (28 → 30): phase 1 NTA-reads
`in` so S stays L2-resident, phase 2 reads S from L2 and NT-writes `out`;
per-volume DRAM traffic = the compulsory 1.5 MB. Implementation cost: the
candidate-list gate only — `run_vols` already routes pf=4 to the NTA read
cursor, mode-gates the write-intent halves off, and disables PFNX (correct
here: the cold window is only FFT36_PFDN = 4 KB/volume, and T1 pre-coverage
would refill L2 with in-stream lines, defeating the design). pf=3 for NT
would be behaviorally identical to pf=4, so only the pf=4 spelling exists.
Ranked most complex in the hysteresis (must beat every simpler candidate
by >3%). Everything else — kernels, pacing constants, all 28 r6 candidates —
is untouched; B=1's hot path is instruction-identical to r5/r6.

**Known design limit, stated up front:** the NTA cursor keeps r6's
yb-subloop-only pacing at a 4 KB constant lead, so the DRAM read stream is
bursty in NT-pf4 mode (~2× average rate during the yb half of phase 1, idle
during zb and phase 2). Pacing across both subloops would need a ≥12–16 KB
lead to absorb the yb consumption deficit, and that many quick-evict NTA
lines + the 20.25 KB plane buffer overflow the 32 KB L1 — the r3 catastrophe
mechanism. So yb-only is the L1-constrained optimum for NTA, and the burst
cost is inherent. If NT-pf4 reaches ~1.5 MB/vol of traffic but serializes the
bursts, it lands ~135–160 rather than ~125 µs; still well under 182.6.

### Attribution

* The scratch+NT structure is this file's own r2; the NTA read mechanism and
  its pacing discipline are this file's own r6; the diagnosis of *why* NT
  kept losing (S eviction by the in-read) is r6's L2-eviction analysis
  applied to scratch mode. The combination is new this round.
* **Negative guidance used, saving this round from two dead ends on my own
  Next lists**: L36_mixedradix r6 measured source-interleaved transform pairs
  (sp2) at **+7.7%** B=1 wallaby / +5% Haswell — kills the "interleave two
  lines" idea; L36_pencilfused r6 measured two-group software pipelining
  (PFA36X2) at **+1–3% pw4, −17% pw2** — kills mixedradix r1 item 2 for the
  whole board (the OOO window already covers cross-group overlap at PW=4).
  Both were candidate B=1 compute levers here; neither will be rediscovered.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 port-5 swaps per 36-point line over
PW lanes; 241 056 FMA-port vector ops per volume at PW=4; port floor ~83 µs
at the 2.89 GHz AVX-512 licence clock (r5 verdict's number, which supersedes
my r6 header's 3.89 GHz non-AVX arithmetic — 3.89 GHz is the non-AVX clock;
512-bit code runs at 2.9). NT-pf4 adds zero arithmetic.

### What was measured (wallaby, Gold 6448Y, mostly quiet windows this session)

End-to-end, driver min; rel_l2 vs numpy 3.644–3.654e-16 at B=1/32/256;
bit-identical re-runs everywhere; AVX2-only build verified end-to-end on
wombat (PASS 3.646e-16 at B=2, bit-identical — the candidate-list change also
touches the no-AVX512 build, hence the check):

| B | end-to-end µs/vol | pick | notes |
|---|---|---|---|
| 1 | **51.8** (sd 0.9%) | pw4, inplace, pf=0 | = r6's 50.7/r5's 51.4; hot path untouched |
| 32 | **70.8** | (not dumped) | = r6's 71.2 |
| 256 | **95.7–97.8** (sd 0.5–1.4%) | pw4, scratch, pf=2 | r6: 100.4; MKL same windows 200–211 → 2.1× |

Full 30-candidate B=256 tuner table (one tournament, back-to-back): all 30
pass the 1e-13 gate including both new NT-pf4 entries. Best: pw4-scratch-pf2
77.0; **pw4-scratch+nt-pf4 122.2, pw2 118.8** — the new candidate loses by
~59% on wallaby, exactly as predicted (same epistemic situation as r6's pf=4
itself: wallaby's 2 MB L2 holds S without protection, and its NTA is harmful
in every regime it can see — all pf=3/4 rows sit at 107–127 vs 77–90 for
their pf=1/2 twins). Wallaby cannot price this bet either direction; the
node tournament is the only honest judge. The tuner correctly rejects it
here, so shipping it risks nothing.

### What was tried and did NOT work — with the number that killed it

1. **Pacing the NTA cursor across both subloops for the NT-pf4 case**
   (to smooth the read bursts): killed by arithmetic before building, see
   the design-limit paragraph — a ≥12 KB swinging lead of quick-evict NTA
   lines + the 20.25 KB plane buffer overflows L1; this is the r3 NTA
   catastrophe (135.1 vs 104.4) mechanism re-derived, not re-measured.
2. Nothing else was touched. Deliberately the narrowest round yet (one
   candidate-list gate changed); the untested r6 surface on the node is
   already large.

### Node predictions (stated so they can be scored; r6's still stand too)

* **B=256, the round's target.** If the S-eviction diagnosis is right and
  the burst serialization is mild, the tuner picks `pw4 (or pw2)
  scratch+nt pf=4` and the cell lands **130–160 µs** (from 182.598; floor
  ~125). If NTA's L1-tax transfers to Cascade Lake the way wallaby suggests,
  the pick stays `inplace/scratch pf=2` at **178–188** and the NT story is
  closed for good — with the pick string as the one-line verdict either way.
* **B=32**: same mechanism, same fork: NT-pf4 selected → **135–165**
  (from 168.565); rejected → 165–175.
* **B=1**: this round adds nothing; r6's fork stands — `inplace pf=4`
  selected → **95–112**, rejected → 119–122.
* **B=4**: r6's fork stands — pf=2 selected → **124–129**, else 128–132.
* Setup: 30 candidates ≤ ~5.5 s at B=256 (measured 2.7 s on wallaby).

### Next

1. **Read the node's B=256/B=32 pick strings first.** They adjudicate three
   bets at once (pf=2 at B=4, pf=4-inplace at B=1, NT-pf4 at streaming).
2. If NT-pf4 is selected but lands >160 at B=256: the burst serialization is
   the residual; the fix direction is a *partial* NTA policy (NTA only every
   second line, T1 for the rest — halves the L2 pollution instead of
   removing it, but lets the cursor pace both subloops at an 8 KB lead
   within L1 budget). Only worth designing with the node number in hand.
3. If NT-pf4 is rejected everywhere: NT is closed permanently at L=36; both
   streaming cells are then at their in-place floors and the only remaining
   lever anywhere is B=1 compute — front-end/DSB measurement via the
   monitor (`perf stat -e idq.dsb_uops,idq.mite_uops`, pencilfused r6's
   suggestion) before touching anything.
4. Still unrun from r5/r6: the FFT36_PFWD node sweep (1296/2592/5184 at
   B=256) and one `perf stat -e L2-misses,LLC-loads` at B=1 — both
   monitor-only, both listed again so they are not forgotten.

---

## Round panel_r8

### Where round 7 stood, and this round's diagnosis

Node, panel_r7: **B=256 held (183.529 µs, by 0.03% over mixedradix — a tie)**;
B=1 120.252 (2nd, mixedradix 118.532), B=4 129.764 (tie), B=32 170.083 (3rd,
+0.9% inside a 1.0% spread; the reported min was a pw4 run, the checked runs
picked pw2 at 170.7 — same rank). Both r6/r7 bets were **rejected by the node
in every cell**: NTA in-read (pf=4, all three entries in three forms — zero
picks) and scratch+NT-pf4 (the read-protected NT — rejected; the verdict now
states NT as 0-for-everything on this node and elevates "hide the RFO with
prefetchw, do not avoid it with NT" to a rule). B=1 sits at **1.45× the ~83 µs
port floor**, all three L36 entries cluster at 118.5–124.0, the L2-thrash
diagnosis is now *unsupported by any positive result*, and the discriminating
counter (`perf stat` at B=1, requested by three records for three consecutive
rounds and ordered by the r7 verdict) **was still not run**. The verdict's
other order: stop tuning the batched cells (B=256 is at its modelled traffic
floor; B=32/B=256 moved ≤0.5%/cell in two rounds).

So this round: (a) get the missing measurement *without* the monitor, by
instrumenting `fft3d_create()` itself; (b) one new B=1 mechanism the panel has
not tried, aimed at a defect visible in the *schedule* rather than the cache
budget; (c) remove a structural bias in my own tuner that has plausibly been
donating 1–3% at B=1 for four rounds. Batched cells untouched, per the order.

### Technique (round 8 delta) — no arithmetic change anywhere

1. **pf=5/6: two-level (deep-T1 + near-T0) prefetch of phase 2's 36 source
   streams, INPLACE-only tuner candidates.** The existing `PF36` (T0, one
   64-B line = one tile ahead, load-bearing since r2: removing it cost 14% at
   B=1) issues its 36 prefetches in a burst at each tile boundary. Those 36
   target lines all contend for the **~12 L1 fill buffers**, so when they miss
   L2 (at node B=1, `out`'s early planes have been evicted to L3 by the
   phase-1 in-read), they drain at ~12-at-a-time × ~70 cycles ≈ 210 cycles —
   against a 248-cycle port-0 tile: *marginally too late, every tile*. pf=5
   adds `prefetcht1` at `FFT36_PF2D` = 4 lines (= 4 tiles ≈ 1000 cycles)
   ahead per stream: T1 fills L2 through the **L2 superqueue (16+ entries, no
   L1-FB pressure)**, so by the time the near-T0 runs, it only moves L2→L1.
   pf=6 = pf=5 + the pf=1 paced T1 in-read, in case two ~1–2% terms add.
   Cost: 11 664 `prefetcht1`/volume in phase 2, zero flops. Both gated,
   correctness-checked, rankable by the hysteresis like everything else.
2. **Regime-aware hysteresis band.** The 3% simplest-wins band (r4) was built
   for streaming cells where the arena misrepresents the end-to-end run (the
   r3 B=32 coin-flip regression). At B=1/B=4 the arena **is** the scored
   regime — same buffers, same steady loop the driver times, self-warmed
   since r6, in-arena sd ~0.5% — yet the 3% band has been installing pf=0
   over anything that wins by 1–3%, *which is the entire size of the gap to
   mixedradix at B=1* (and their node pick string shows `pf1` winning there
   some rounds). New rule: band = 1% when `nv == batch <= 8` (B=1, B=4),
   3% otherwise. B=32/B=256 keep the full band.
3. **In-plan node probes, reported through `fft3d_description()`.** After the
   tournament, create() times four fixed configurations at nv=1 steady state
   (pw4-inplace-pf0, the node's standing B=1 pick, so numbers are comparable
   across rounds): `p1` = phase 1 alone (honest: repeating it re-streams
   in→out, the same 1.5 MB L2 pressure as a real execute), `p2w` = phase 2
   alone repeated in place (**optimistic by construction**: its 746 KB fits
   1 MB L2 once warm — this is phase 2's compute+L1/L2 floor with no phase-1
   eviction), `p2wd` = same + deep prefetch (isolates pf=5's pure µop tax),
   `fu` = the full pf0-inplace execute. **`fu − p1 − p2w` is the phase-
   boundary memory penalty — a direct, node-measured discrimination of the
   L2-thrash story from the front-end story**, the exact fork the r7 verdict
   said must be settled before anyone writes another L=36 candidate. It rides
   the description string onto the leaderboard, so the monitor need do
   nothing. Cost ~7 ms of setup, min-of-3 rounds × 4 reps each.

### Attribution

* **Deep staged T1 + near T0 (pf=5): shape from L64_radix8's `slabpf`**
  (next-slab `prefetcht1` staging, node-selected at B=2/B=8 alongside pfw;
  their r7 "Next" explicitly proposes a deeper lead for uncovered L3
  latency). The fill-buffer-vs-superqueue burst analysis applied to my 36
  stream-heads-per-tile is this file's.
* **The band change is prompted by L36_mixedradix's node pick strings**
  (`v1-cached-pf0/pf1` at B=1 — pf1 winning there some rounds while my 3%
  band structurally forbids a 1–3% pf1 win from ever installing).
* Probes-in-description: the g_desc channel is L36_mixedradix's r2 mechanism;
  using it to carry *measurements* rather than picks follows L6_pfa/L6_unrolled
  reporting kclk clock probes through the same string.
* Restraint on batched cells: the r7 verdict's explicit order.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 port-5 swaps per 36-point line over
PW lanes; 241 056 FMA-port ops/volume at PW=4; port floor ~83 µs at the
2.89 GHz AVX-512 licence clock. pf=5/6 add 11 664 `prefetcht1`/volume (phase 2
only; +1 predictable branch per tile on all paths, same pattern as the r5 pfw
flag). Build hygiene: clean under the monitor's flags
(`-O3 -march=cascadelake -std=gnu11`) and bare `-O2`; disassembly carries 282
`prefetcht1` sites (r7: 66) — the PFT1D unrollings are all present.

### What was measured (wallaby, Gold 6448Y — toggling fast/slow ~2× windows
again this session; in-arena tables back-to-back, cross-window deltas soft)

End-to-end, driver min; rel_l2 vs numpy 3.644–3.654e-16 at B=1/2/4/256,
bit-identical re-runs everywhere; AVX2-only build verified end-to-end on
wombat (PASS 3.646e-16 at B=2, bit-identical, probes running the pw2 path):

| B | end-to-end µs/vol | pick | notes |
|---|---|---|---|
| 1 | **52.96** (sd 3.9%) | pw4, inplace, pf=0 | = r7's 51.8; hot path untouched |
| 4 | **70.5** (sd 1.0%) | pw4, inplace, pf=0 | = r6's best-ever 68.9 band |
| 256 | **97.75** (sd 1.2%) | pw4, scratch, pf=2 | = r7's 95.7–97.8; setup 3.2 s |

All 34 candidates pass the 1e-13 gate (incl. the four new pf=5/6). Wallaby
B=1 in-arena, one tournament: pf=5 **111.6** vs pf=0 104.8 (+6.5%), pf=6
112.1, pf=1 105.7, pf=2 112.8 — **wallaby taxes the deep prefetch and shows
no prize, exactly as it must: its 2 MB L2 holds in+out at B=1, so there are
no phase-2 L3 misses to stage.** Same epistemic situation as pf=4 in r6 and
NT-pf4 in r7: only the node tournament can price the bet, and the local tuner
correctly rejecting it means shipping it risks nothing.

Probes on wallaby (fast window): `p1=35.0 p2w=17.4 p2wd=19.5 fu=52.3` —
**p1 + p2w = 52.4 ≈ fu = 52.3: zero phase-boundary penalty on wallaby**,
which is the predicted signature of a 2 MB L2 that holds both volumes, and
validates the probe arithmetic end-to-end. p2wd − p2w = +2.1 µs is the deep
prefetch's pure tax on this host. Forced-path control verified:
`FFT36_PF=5` at B=1 runs, picks pf=5, description reports it.

### What was tried and did NOT work — with the number that killed it

1. Nothing was removed or replaced this round; all 30 r7 candidates survive.
   The only local negative is the expected one: pf=5/6 lose on wallaby at B=1
   by 6–7% (111.6/112.1 vs 104.8, table above) — recorded so nobody reads a
   wallaby rejection of pf=5 as the node's answer.
2. Not attempted, deliberately: any batched-cell tuning (verdict's order), a
   third NTA variant (closed by r7's board-wide null), and the 88→80 DFT9
   transcription (needs a front-end verdict from the probes first — if the
   B=1 residual is memory, 3% of arithmetic buys nothing).

### Node predictions (stated so they can be scored)

* **The probe line is the round's primary deliverable.** Prediction if the
  L2-thrash story is right: `fu − p1 − p2w ≈ 20–35 µs` (phase 2 re-reading
  ~0.5 MB of evicted `out` from L3, plus phase-1 RFO exposure inside p1
  itself). If instead `fu ≈ p1 + p2w` on the node too, the thrash story is
  **dead** — the residual is front-end/scheduling inside the phases — and r9
  should attack code layout (pw2 at B=1, DSB behaviour), not caches.
* **B=1**: fork. If the thrash story holds AND the staged T1 covers it: pick
  `pw4 inplace pf=5` (or pf=6) and **112–119 µs**. If only the band change
  bites (pf=1/pf=2 winning by 1–3%): pick pf=1 or pf=2 at **117–120**. Null:
  pick pf=0, 119–122, and the probe line still pays for the round.
* **B=4**: the 1% band may now install pf=2 (r6's quiet-window −8% at B=4,
  rejected by the node at 3%-band in r7): pick pf=2 → **124–129**; null:
  pf=0, 128–132.
* **B=32/B=256: unchanged**, 168–172 / 180–188, picks `inplace pf=2` /
  `scratch|inplace pf=2` as in r7. pf=5/6 should NOT appear at streaming
  (they lack the write-intent cover that is worth −42% in-arena there); if
  pf=6 does appear, it beat pf=2 by >0 within a 3% band on the node's own
  table, which is information, not an accident.
* Setup: 34 candidates + probes ≤ ~3.5 s at B=256 (measured 3.2 s wallaby).

### Next

1. **Read the node's probe line first** (`probe us p1=… p2w=… p2wd=… fu=…`
   in this entry's description on the leaderboard). It settles the
   thrash-vs-front-end fork with this round's numbers even if every candidate
   is rejected.
2. If `fu − p1 − p2w` is large and pf=5 still lost: the misses are real but
   T1 staging did not help — the remaining suspect is the *phase-1 side*
   (p1 ≫ its ~55 µs floor share would show this), where the in-read and the
   mid-store RFO interleave; try pf=2+deep (pfw for phase 1 + PFT1D for
   phase 2) as one composite candidate.
3. If the front-end story wins: measure a pw2-primary B=1 variant seriously
   (the node already chose pw2 at B=256-r6 and B=32-r7-checked-runs; pw2
   halves instruction bytes through MITE), and only then the genfft n1_9
   DAG transcription (~3% of arithmetic).
4. Still open for the monitor: the FFT36_PFWD sweep (1296/2592/5184, B=256)
   and — if the probe line contradicts itself — the real
   `perf stat -e l2_rqsts.all_demand_miss,LLC-loads,idq.dsb_uops,idq.mite_uops`.
   `FFT36_PF2D` (deep-prefetch lead, lines) is compile-time overridable for a
   node sweep {2,4,8} if pf=5 is selected.

---

## Round panel_r9

### Where round 8 stood, and this round's diagnosis

Node, panel_r8: B=1 122.099 (2nd, mixedradix 119.951), B=4 132.453 (3rd, a real
+2.1% regression over r7 against a 0.8% spread, unchanged pick — verdict class:
refactor-around-an-untouched-hot-path), B=32 168.583 (2nd), **B=256 185.973
(1st)**. The round's real product was the probe line: node, three processes,
`fu − p1 − p2w = −2.6 / −3.2 / −3.6 µs`. **Zero phase-boundary penalty; the
L2-thrash story is dead**; p1 = 90–93 µs carries the whole excess over its ~56 µs
port share, and every prefetch instrument is 0-for-three-tournaments at B=1. The
verdict's order for L=36, verbatim: *"stop looking at caches; go at the front
end. Code size, not caches"* — with the DSB/MITE counter named as "the single
thing", and the batched cells ordered frozen (both at modelled floors).

So this round is a pure front-end round: no arithmetic change, no candidate-list
change, nothing touched on the streaming paths.

### Technique (round 9 delta)

1. **Compile-time specialization of the pf=0 hot path** (the node's B=1/B=4
   pick every round since r5). `run_vols` threads pfr/p1w/pfw/p2d as RUNTIME
   ints into the always_inline plane functions, so the scored body carried
   every prefetch mechanism of rounds r3–r8 as live code: under the monitor's
   flags, general `phase1_pw4` = **1482 instrs** (108 prefetch instrs + flag
   tests in both subloop bodies), general `phase2_pw4` = **1357 instrs with two
   full codelet copies** (fp count 496 = 2×248 — the NT loop lives in the same
   function). New `phase1_pf0`/`phase2_pf0` call the plane functions with
   literal-constant flags; gcc dead-codes everything: **phase1 1482 → 1336,
   phase2 1357 → 633 instrs**; walked footprint per x-plane 8.8 → 7.9 KB, per
   volume-loop 2839 → 1969 instrs, against a ~1.5k-µop CLX DSB and 16 B/cycle
   MITE. Arithmetic order is identical → output bit-identical (verified).
   The r8 `if (pfd)` tile-loop branch — the leading suspect for the B=4
   regression — is gone from the B=4 body. pf>0 and NT candidates route
   through the general bodies, unchanged from r8, per the freeze order.
2. **Code-size A/B measured by the node itself**: the probe line now carries
   `fu` (specialized pf0 execute) and `fug` (general-body pf0 execute forced
   via opaque-register flag args — same arithmetic, same address streams, ~2×
   walked code). `fug − fu` on the leaderboard is a direct node measurement of
   what instruction footprint costs at B=1, whatever the tuner picks and even
   if perf is unavailable. p2wd is retired from the string (r8 answered it:
   +6.5 µs tax, zero picks).
3. **In-plan front-end counters**: raw `perf_event_open` (syscall, no library,
   create-time only) on {cycles, IDQ.DSB_UOPS 0x0879, IDQ.MITE_UOPS 0x0479,
   UOPS_ISSUED.ANY 0x010e} as one group around 8 reps of the specialized fu at
   nv=1, reported as `fe/vol kcyc= kdsb= kmite= kiss=` in the description —
   the counter the verdict has requested for three rounds, taken without the
   monitor if the node's perf_event_paranoid allows it (every dev host here is
   at 4 → `fe=na`; a slurm compute node may be more permissive; costs nothing
   and fails closed).

### Attribution

* **Compile-time exec-variant specialization: L45_pfa round panel_r8** (their
  3535 → 3087-instr scoring path; all three L=45 cells taken). Their second r8
  mechanism, the **opaque-base asm barrier, was checked for and is NOT needed
  here**: this file's streams are single-base + disp32 addressing — 5–8 leas
  and no GPR spill-reload chains in either phase under node flags (their
  pathology was 48 leas + 37 spills). Checked before changing anything;
  recorded so nobody ports the barrier here expecting L45's 7%.
* The fu/fug differential and the perf-group probe extend **this file's own r8
  probe-through-description mechanism** (which the verdict asked the panel to
  copy); the fe counter list is the r8 verdict's own ask 2.
* Freeze on batched cells: the r8 verdict's explicit order.

### Operation count

Unchanged: 248 FMA-port vector ops + 49 port-5 swaps per 36-point line over PW
lanes; 241 056 FMA-port ops/volume at PW=4; port floor ~83 µs at the 2.89 GHz
AVX-512 licence clock. This round adds and removes zero arithmetic; the pf=0
path's µop count per volume drops by the dead flag tests/branches (~1.6k
branches/volume) and its I-footprint by ~30%. Build hygiene: clean under the
monitor's flags and bare -O2; FFT36_SKIP1/2, FFT36_LOUD, FFT_VL=4,
FFT36_FORCE_PF all still compile; .text 106 KB (was ~90; the two specialized
bodies per PW).

### What was measured (wallaby, Gold 6448Y — toggling fast/slow windows again;
pairwise A/B within alternating back-to-back runs, cross-window deltas soft)

Correctness: rel_l2 vs numpy **3.644e-16 (B=1), 3.651e-16 (B=4), 3.654e-16
(B=256)**, bit-identical re-runs everywhere; AVX2-only build verified
end-to-end on wombat (PASS 3.646e-16 at B=2, bit-identical). All 34 tuner
candidates pass the 1e-13 gate at B=1 (full table dumped; picks pw4-inplace-pf0
at B=1, pw4-scratch-pf2-class at B=256 as before).

* End-to-end: B=1 **52.9–55.0** (fast windows; = r8's 52.96 band), B=4
  **270.1–285.6** /4 = 67.5–71.4 µs/vol (r6–r8 band), B=256 **100.2 µs/vol**
  (r7/r8 band, 1.96× MKL same window). Nothing regressed.
* **r8-vs-r9 alternating A/B** (same binary pair, same window): B=1 clean
  pairs r9 −0.4/−0.45 µs (~−0.8%); B=4 nine pairs, median pairwise delta
  −0.1% — **wallaby cannot resolve the change**, exactly as it must: SPR has a
  6-wide decode and a much larger DSB, so the CLX front-end bet is invisible
  here (same epistemic situation as pf=4/r6, NT-pf4/r7, pf=5/r8 — the node
  tournament + probe line is the judge).
* Probe line (mid-speed window): `p1=48.5 p2w=17.6 fu=57.9 fug=57.6; fe=na` —
  fug ≈ fu on wallaby confirms the A/B machinery and wallaby's blindness at
  once. fe=na is correct here (perf_event_paranoid=4 on wallaby, wombat, and
  the login node).

### What was tried and did NOT work — with the number that killed it

1. **The naive fug probe measured nothing**: calling the general bodies with
   constant flags made gcc constprop them into clones IDENTICAL to the
   specialized bodies (verified on the .o: `phase1_pw4.constprop.0` = 1336
   instrs = the specialized body). Fixed by passing the zero flags through an
   opaque register (`__asm__("" : "+r"(z0))`). **Any A/B probe of a
   specialization must launder its "unspecialized" arguments**, or the
   compiler quietly runs the same code on both sides.
2. **The L45 opaque-base asm barrier is a no-op here** — checked in the
   disassembly before porting (5–8 leas, no spill-reload chains); not applied.
   Counted as a checked negative, not a failure.
3. `restrict` on the specialized phase-2 wrapper was wrong (in-place mode
   aliases mid==out); caught by -Wrestrict at build, not by measurement.
   The general phase-2 signature is deliberately unqualified — keep it so.

### Node predictions (stated so they can be scored)

* **The probe line is again a primary deliverable.** `fug − fu` at node B=1:
  if the front-end story is real, **+2 to +6 µs** (the general body walks
  ~870 more instrs/plane through a 16 B/cycle MITE when the DSB overflows);
  if ≈ 0, code size is NOT the B=1 residual and the front-end fork closes
  negative. If `fe` reports numbers: kmite ≫ kdsb on the fug side of ~350k
  cycles/vol would confirm directly; predict kcyc ≈ 345–360 at 2.89 GHz.
* **B=1**: pick `pw4 inplace pf=0` (now the specialized body). If the
  front-end effect is the 2–4% class the instruction accounting suggests,
  **117–121 µs** and likely first (mixedradix at ~120); null branch: 120–123.
* **B=4**: the pfd-branch deletion should recover the r8 regression:
  **128–132**, pick pf=0 (or pf=2 within the 1% band).
* **B=32/B=256: unchanged by construction** (picks are pf=2-class, which
  route through byte-identical general bodies): 166–172 / 182–189. Any move
  >2% there is layout/alignment luck, not mechanism.
* Setup: +4 probe timings and the perf group, ≤3.5 s at B=256 (measured 3.3 s
  wallaby).

### Next

1. **Read fug − fu (and fe if present) off the node leaderboard first.**
   * If fug − fu ≥ ~2 µs: front-end confirmed; the follow-ups are (a)
     specialize the streaming picks too (pf=2 inplace/scratch — the freeze
     order should be lifted once the mechanism is proven on B=1), and (b)
     shrink the phase-1 plane body further — the yb-subloop (4.6 KB) and
     zb-subloop (3.2 KB) together still overflow the DSB; splitting their
     alternation or compressing TRNC staging is the next 1–2k µops.
   * If fug − fu ≈ 0 AND fe shows DSB-served: the front-end story is dead
     too; B=1 is then at a memory+port structural floor shared by all three
     entries, and the only lever left anywhere is arithmetic — the genfft
     n1_9 DAG transcription (~2% of port-0 ops; transcribe, don't hand-derive,
     r1 item 6 stands) — or accepting the cell.
2. If the node's B=4 does NOT recover to ≤131 with the pfd branch gone, the
   r8 regression attribution was wrong twice; ask the monitor for one
   `FFT36_MODE=inplace FFT36_PF=0` control against the r7 binary.
3. Still open, monitor-only: perf_event_paranoid on the node (one `cat`); if
   ≤2, the fe line self-serves every future front-end question at create time.

---

## Round panel_r10

### Where round 9 stood, and this round's diagnosis

Node, panel_r9: B=1 122.576 (2nd, mixedradix 120.478), B=4 131.953 (leaderboard;
but see below), B=32 168.253 (1st by 0.06% — the verdict says not real), B=256
185.818 (2nd by 0.9% — the verdict's median reading says dead tie). The verdict's
L=36 findings, all three decisive for this round:

1. **My r9 `fug − fu` A/B landed its null branch**: +0.4/+0.2/+0.3 µs on ~123 —
   doubling the walked instruction footprint through MITE costs 0.3%. Combined
   with mixedradix's `roll` probe (+22–24% the WRONG way) and pencilfused's flat
   `cs4 ≈ ip4`, "the front-end theory is not merely unconfirmed — it is measured
   absent." Both named B=1 theories (cache r8, front-end r9) are now dead by
   node measurement.
2. **My pick lottery is the board's worst measurement**: B=4 runs of
   131.95 / 184.11 / 132.48 (39.5% spread) and B=1 of 122.58 / 131.06 / 122.64
   (6.9%). Verdict: "no L=36 conclusion should be drawn from L36_pfa's B=1 or
   B=4 numbers until the pick lottery behind them is closed."
3. **The single thing for L=36: transcribe genfft's n1_9 FMA DAG** — L45_pfa did
   it in r9 (44 → 40 FMA-port vector ops per DFT9, correct first build, accuracy
   improved), wrote the transcription rule out mechanically, and their record
   says it "transfers verbatim to L36_pfa". It is the only lever at this
   geometry not yet falsified. (Their caveat, carried honestly: on their L=45
   B=1 cell the −5.5% port-0 cut bought 1.2%, so port 0 may not fully bind.)
4. Also acted on: "±2% at these cells is the code-layout noise floor of
   recompiling the file... stop building fixes for it."

### Technique (round 10 delta)

1. **DFT9 module replaced by genfft's n1_9 FMA DAG (`DFT9F`), taken VERBATIM
   from L45_pfa r9** — their transcription is in this file's own macro dialect
   (vec/VPAIR/SWAP/VFMA; their modules are this file's lineage), so the macro
   dropped in unchanged. Inside `PFA36`, the per-k1 CT 3×3 block (6 DFT3M +
   4 CMULW = 44 FMA-port ops + 10 swaps) becomes one `DFT9F` call (40 + 12);
   outputs come out in natural DFT order m = 0..8 and fan to the same CRT map
   `ST((9*k1 + 28*m) % 36)`. DFT3M, CMULW and the three CT twiddle constants
   are deleted; the 8 n1_9 sign-pair constants come in. r1's three failed HAND
   derivations of this exact result stand as the reason to transcribe.
2. **Pick lottery closed structurally.** At `nv == batch <= 8` the candidate
   list is restricted to the shapes the node has actually picked at B=1/B=4 in
   five consecutive scored rounds (r5–r9): {inplace, scratch} × pf≤2 × {pw2,
   pw4} = 12 candidates. NT, pipe, NTA (pf=3/4) and deep-T1 (pf=5/6) are
   0-for-every-tournament in those cells across five rounds of node data and
   contribute only tail risk there; all of them remain candidates at streaming
   batch, and any FFT36_PW/MODE/PF env or FFT_FORCE_* compile override restores
   the full list so the monitor's forced controls still work. Two supporting
   changes: the r8 regime-aware 1% band is reverted to the uniform 3% (the 1%
   band existed to admit 1–3% pf wins that r9 measured as nonexistent — all it
   bought was instability), and small-nv tournaments take min over 9 timing
   rounds instead of 5.
3. **Retired instrumentation**: the `fug` general-body probe (its question is
   answered: +0.3%) and the perf_event group (fe=na on the node too — the r9
   verdict withdrew the counters). The probe line keeps p1/p2w/fu for
   cross-round phase-split comparability and now also reports the candidate
   count (`nc=`), so the leaderboard shows whether a run used the restricted
   list.

### Attribution

* **DFT9F = genfft n1_9 FMA DAG, from L45_pfa round panel_r9, verbatim** —
  constants, comment and all. Their record specified the transfer and the
  spill caveat (+23 rsp-relative moves in their phase body, gcc's allocator
  left alone on their advice, and mine).
* The candidate-restriction rationale is the r9 verdict's own pick-lottery
  order plus five rounds of this file's node pick strings; the 3% band
  reversion is the verdict's ±2%-noise-floor rule applied to my own r8 change.

### Operation count

Per 36-point line over PW lanes: 9 DFT4 (8 ops) + 4 DFT9F (40 ops) =
**232 FMA-port vector ops + 57 swaps** (was 248 + 49). Per volume at PW=4:
972 codelet calls × 232 = **225 504 port-0 vector ops** (was 241 056, −6.45%).
Node port floor at the 2.89 GHz AVX-512 licence clock: **~78 µs** (was ~83.4).
**Verified in the object code under node flags** (`gcc -O3 -march=cascadelake
-funroll-loops -fno-math-errno`): `phase2_pf0_pw4` arith = 232 exactly (one
codelet site, every twiddle vmulpd gone), `phase1_pf0_pw4` arith = 464 = 2×232;
shuffles 57 and 258 = 2×57 + 144 transposes — the model is exact. The DAG's
longer live ranges cost spills, as L45_pfa warned: phase2 22 rsp-relative
vmovs (body 633 → 679 instrs), phase1 78. Net trade per codelet: −16 port-0
ops for +8 port-5 swaps and a few spills.

### What was measured (wallaby, Gold 6448Y — windows toggled again; quiet
windows marked by their sd)

Correctness: rel_l2 vs numpy **3.591e-16 (B=1), 3.577e-16 (B=4), 3.586e-16
(B=32, B=256)** — all slightly better than the CT form's 3.644–3.654e-16 (the
DAG is FFTW's accuracy-tested form, same effect L45_pfa saw). Bit-identical
re-runs everywhere. AVX2-only build verified end-to-end on wombat (PASS
3.575e-16 at B=2, bit-identical).

| B | end-to-end µs/vol | window | r9 band same host |
|---|---|---|---|
| 1 | **50.9** (51.6 in a sd-4% run) | fast | 52.9–55.0 |
| 4 | **75.2** (sd 0.06%, quiet) | quiet | 67.5–71.4 (r6–r9, other windows) |
| 32 | **69.7** (sd 0.22%, quiet) | quiet | 70.8–71.4 |
| 256 | **98.2** (sd 1.8%) | ok | 95.7–100.4 |

**Wallaby is flat on the arithmetic change, exactly as pre-stated and as
L45_pfa found**: wallaby B=1 is port-5 bound (r6 analysis, 215k-cycle model),
and the DAG leaves port 5 unchanged — per line, old 124 half-FMAs + 49 swaps =
173 vs new 116 + 57 = 173 — while cutting only port 0, which wallaby has two
of. The node's single 512-bit FMA unit is the only machine that can price the
−15.5k port-0 cycles/volume (~5.4 µs at 2.89 GHz if port 0 fully binds).

Tuner behaviour with the restricted list (FFT36_LOUD, B=1): exactly 12
candidates, all pass the 1e-13 gate, pick `pw4 inplace pf=0` with the nearest
rival (pf=1 twin) 4.4% behind — deterministic under the 3% band by a clear
margin. Setup at B=1 0.14 s, B=256 3.1 s.

### What was tried and did NOT work — with the number that killed it

1. Nothing failed this round: the transcription passed the reference gate and
   numpy on the first build (the point of transcribing the generated DAG —
   r1's three hand-derivation failures remain the counterfactual).
2. Not attempted, deliberately: spill surgery on DFT9F's live ranges (L45_pfa's
   "first suspect if the cut does not land"; gcc's allocator has beaten this
   file's hand scheduling three times in r1), any new prefetch/cache mechanism
   (two theories dead by measurement), and any batched-cell tuning beyond the
   shared arithmetic (both streaming cells sit at modelled traffic floors).

### Node predictions (stated so they can be scored)

* **B=1, the target.** Port-0 model: −5.4 µs if port 0 fully binds → ~117;
  L45_pfa's measured discount (their −5.5% bought 1.2%) → ~121. Prediction:
  **117–121 µs, pick `pw4 inplace pf=0` (nc=12 in the description)**. Below
  120.5 beats mixedradix's r9 number with the same codelet family they have
  not yet adopted. If B=1 lands ≥122 flat, port 0 does not bind at L=36 B=1
  either, and the honest residual model is memory latency inside phase 1
  (p1 = 90–93 of fu = 123 on the node, from my own r8/r9 probes) — in that
  case the next lever is phase-1 structure, not arithmetic.
* **B=4: 126–131, and the real deliverable is the SPREAD.** Three runs within
  ~1% of each other (restricted list + 3% band + 9 rounds) closes the lottery;
  a repeat of a 130+ outlier at an `nc=12` pick string would mean the lottery
  was never in the tuner at all but in the node run itself — also worth
  knowing, and now distinguishable.
* **B=32: 164–170** (traffic-floor-bound; the arithmetic cut is mostly hidden
  under memory, expect 0–2 µs). **B=256: 181–188** (at its 2.2 MB traffic
  floor; expect no change beyond noise). Picks pf=2-class as before.
* Probe line: p1 should drop by ~2–3.5 µs (phase 1 carries 2/3 of the
  codelets), p2w by ~1–2, if port 0 binds anywhere.

### Next

1. If B=1 lands in the 117–121 band: arithmetic is done (DFT4 is optimal,
   DFT9 is at genfft's count); the remaining B=1 gap to ~78 µs is phase-1
   memory latency, and the only untried shapes are structural (e.g. fusing
   the two phase-1 subloop passes over `pl` to shorten the plane's live
   window). If it lands flat: same conclusion, stronger.
2. If the B=4 spread stays wild at nc=12: the instability is the node run,
   not my tuner — hand the monitor that datum; it changes how every close
   L=36 cell should be read.
3. mixedradix can take DFT9F verbatim from this file or L45_pfa's; if the
   monitor cuts to two arms, whichever survives should carry it.

---

## Round panel_r11

### Where round 10 stood, and this round's diagnosis

Node, panel_r10: **B=1 113.128 — tied first with mixedradix (113.423), lowest
minimum on the board**; B=4 124.221 min (1st on min, but median 129.395 — the
verdict rules the remaining B=4 spread a HARNESS bimodality, not mine: picks
were identical 3/3, the r10 lottery fix worked); B=32 165.182 (3rd, mixedradix
162.137); B=256 181.643 (2nd, 179.219). The n1_9 DAG priced at ~full value
(−7.7%, the round's deepest cut): **port 0 binds at L=36**, the floor fell to
78 µs with the time, and the ratio is stuck at 1.45×. The verdict's L=36
directive: *"attack phase 1's structure — the plane's live window, the z/y
subloop fusion, the split-access toll on the odd stride"*, and check
L45_mixedradix's r10 three-term costing of exactly that (plane round trip /
split accesses / compulsory L3) at L=36 **before** building the fusion rewrite.
My phase split from the node: p1 = 86.3–89.0 of fu = 115–117 — phase 1 is 76%
of B=1 at ~1.7× its 52 µs port share; p2w = 33.3 vs a 26 µs share.

Checking L45's costing at L=36 took no code:

* **Split-access toll: structurally ZERO at L=36.** L45's largest term (~15–20
  µs there) comes from 45's odd row stride (720 B ≡ 16 mod 64 — 75% of z-pass
  accesses split a cache line). At L=36 the y-row stride is 576 B = exactly 9
  lines; every vector access in both phases is 64-B aligned (r1 record,
  re-verified). Nothing to fix, nothing to gain.
* **z/y subloop fusion: cannot be built at all inside a plane.** The y
  transform at any kz needs all 36 y outputs of the z pass — a true transpose
  barrier per plane. The only fusable seam is *across* planes, and that is the
  two-group software pipelining L36_pencilfused measured dead in r6 (+1–3%
  at pw4). So of the verdict's three named targets, two are closed by
  inspection at this geometry and the third (compulsory L2/L3 movement) is
  exactly what has never been *located* within phase 1.

So this round locates it, and fields the one prefetch shape that has never
been tried, plus the PFWD sweep the records have requested since r5.

### Technique (round 11 delta) — hot pf=0 path untouched, all changes additive

1. **Probes p1z/p1y: phase 1's two subloops timed separately** at nv=1,
   riding the description string (r8 mechanism). `phase1_zonly` = the
   yb-subloop of every plane exactly as shipped (in-read + TRNC pair + PFA36
   + pl writes; an asm sink keeps the dead pl stores — r9's lesson: any probe
   must launder what the compiler could remove). `phase1_yonly` = the
   zb-subloop exactly as shipped (PFA36 + mid stores = the RFO-exposed side),
   reading a fixed L1-resident 20.25 KB plane from the plan's S instead of a
   just-written pl. Port shares at 2.89 GHz: ~26 µs each (port-5: z 22.5,
   y 6.4). Each probe is a LOWER bound for its side (the other stream's L2
   pressure absent), like p2w. Object-code audit under node flags:
   phase1_zonly_pw4 = 232 arith + 201 shuffles, phase1_yonly_pw4 = 232 + 57 —
   both exactly on the per-group model.
2. **pf=7: write-intent prefetch UNBUNDLED from the read cursor.** pf=2 =
   paced T1 read + prefetchw; at B=1 the read half is pure µop tax (the
   in-read is one linear, HW-prefetchable stream) and has ridden along in
   every pf=2 tournament since r5 — the write-intent half ALONE has never
   been fielded anywhere. run_vols: pf=7 sets p1w (INPLACE) / p2w (SCRATCH)
   with pfr=0, no PFNX. Ranked one-mechanism (just above pf=1); joins the
   restricted small-nv list (INPLACE only; nc 12 → 14) where the 3% band
   still demands a real >3% win over pf=0 to install. Rationale: at node B=1
   in-place, in+out = 1.5 MB vs 1 MB L2, so part of `out` is evicted every
   execute and phase 1's stores to it RFO from L3; if p1y confirms that term,
   pf=7 is its cover at half pf=2's µop cost.
3. **FFT36_PFWD is now a runtime plan parameter, swept by the tournament at
   streaming batch**: {pw2,pw4} × inplace-pf2 × pfwd ∈ {1296, 5184} beside
   the 2592 default (nc 34 → 42 at batch>8). The 2592 default was r5 pacing
   arithmetic, never measured anywhere; this is the node sweep asked of the
   monitor in r5/r6/r8, now self-served. Non-default pfwd ranks as a
   tie-break below its default twin (same bit class — prefetch-only
   difference, so the mixedradix r10 one-bit-class-per-pool rule is
   respected). New env override `FFT36_PFWDF=<doubles>` forces a distance
   onto all write-intent candidates for the monitor's controls.

### Attribution

* The three-term phase-1 costing this round tests is **L45_mixedradix r10's**
  (their item 2), checked at L=36 as the verdict directed — with the
  split-access term found structurally absent here (different geometry
  arithmetic, not a disagreement).
* The z/y-fusion dead end is **L36_pencilfused r6's** measurement (+1–3%
  pw4 two-group pipelining), not re-run.
* prefetchw itself is still **L6_unrolled r3's** result; the unbundling is
  this file's own reading of five rounds of its bundled rejections at B=1.
* Probe-through-description is this file's r8 mechanism (originally from
  L36_mixedradix's g_desc); the asm-laundering rule is this file's r9 item 1.

### Operation count

Unchanged: 232 FMA-port vector ops + 57 port-5 swaps per 36-point line at
PW=4; 225 504 port-0 ops/volume; port floor ~78 µs at 2.89 GHz. This round
adds zero arithmetic to any scored path. Object-code audit under node flags
(gcc -O3 -march=cascadelake -funroll-loops): phase1_pf0_pw4 arith = 464 =
2×232, shuffles 258 = 2×57+144; phase2_pf0_pw4 = 232/57 — **the B=1/B=4
scored bodies are instruction-identical to r10** (the new pfwd argument is a
literal constant there and folds away). Builds clean under bare -O2, Haswell
-march=native, and cascadelake flags.

### What was measured (wallaby, Gold 6448Y — fast/slow ~2× windows toggling
all session; quiet windows marked by sd)

Correctness: rel_l2 vs numpy **3.591e-16 (B=1), 3.577e-16 (B=4), 3.586e-16
(B=32, B=256)** — the r10 n1_9 fingerprints, bit-identical re-runs everywhere.
Forced controls exercised end-to-end through the driver: `-DFFT36_FORCE_PF=7`
at B=1 (PASS, repeatable) and `-DFFT_FORCE_PW=2` at B=4 (PASS, repeatable) —
both new paths driver-verified, not just gate-verified.

| B | end-to-end µs/vol | pick | window |
|---|---|---|---|
| 1 | **50.8** | pw4-inplace-pf0, nc=14 | fast (= r10's 50.9: hot path untouched, confirmed) |
| 4 | 79.9 min | pw4-inplace-pf0 | noisy (sd 24%), not a claim |
| 32 | **67.1** (sd 0.38%, quiet) | pw4-inplace-pf2-**pfwd=1296** | **best-ever B=32 here** (r10 quiet: 69.7) |
| 256 | 101.9–105.4 (sd 1.2–1.5%) | pw4-inplace-pf2-**pfwd=1296** | r7–r10 band 95.7–105; MKL same windows 169.6–210.8 → 1.7–2.0× |

* **The PFWD sweep pays on its first outing**: B=32 in-arena, one quiet
  tournament, all 42 candidates gate-passed: pfwd=1296 **67.4** vs default
  2592 **70.0** (−3.7%, an outright win, not a tie-break) vs 5184 69.6.
  Half-a-plane (10.125 KB) of write-intent lead beats the full plane on this
  memory system; the node's own tournament now prices {1296, 2592, 5184}
  directly. End-to-end B=32 improved 69.7 → 67.1 on the same host class.
* **pf=7 behaves exactly as the epistemics predict on wallaby**: B=1
  in-arena 114.8 vs pf0's 98.6 (+16% — wallaby's 2 MB L2 holds in+out at
  B=1, so there is no RFO to hide and only the µop tax shows; same
  cannot-price-it-here class as pf=4/r6 and NT-pf4/r7). At B=32 streaming it
  loses to pf=2 (78.4 vs 70.0) — the read half genuinely matters there. The
  node's B=1 tournament is the only honest judge, and being wrong costs
  nothing (gated candidate).
* **Probe line, fast window**: `p1=39.6 p1z=22.3 p1y=13.9 p2w=17.1 fu=54.6`.
  Sanity: p1z+p1y = 36.2 ≈ p1 − 3.4 (mild positive overlap inside the real
  interleaved body); both subloops sit exactly on wallaby's port-5 model
  (z: (116 half-FMA + 201 shuf)/group → ~25 µs bound, measured 22.3;
  y: 173/group → ~14, measured 13.9) — the probes measure what they claim.
  Slow-window runs show p1z rock-stable (43.5–43.8) while p1 swings 73–90,
  i.e. wallaby's slow state hits the interleaved body hardest — noted, not
  chased.
* Setup: 0.16 s at B=1, 1.7 s at B=32, ~4.0 s at B=256 (was 3.1 — the 8 new
  streaming candidates; still trivially inside budget).

### What was tried and did NOT work — with the number that killed it

1. Nothing failed in code this round. Two of the verdict's three named
   phase-1 targets were closed by paper before building: the split-access
   toll is zero at L=36 (576 B ≡ 0 mod 64 — every access aligned), and
   intra-plane z/y fusion is impossible (transpose barrier; the cross-plane
   variant is pencilfused r6's measured +1–3% dead end). Recorded so nobody
   builds the L=36 fusion rewrite expecting L45's terms.
2. pf=7 at wallaby B=1: +16% (above) — expected, not evidence about the
   node; recorded so a wallaby rejection is not read as the verdict.

### Node predictions (stated so they can be scored)

* **The probe line is a primary deliverable.** Node port shares: p1z ~26 µs,
  p1y ~26 µs, sum 52 vs p1 = 86–89. Three pre-registered readings:
  (a) p1y ≫ 30 → the mid-store RFO is the named term, and pf=7 should be
  winning its tournament in the same run (cross-check); (b) p1z ≫ 30 with
  p1y ≈ 27 → the in-read latency is the term and, with every read-prefetch
  instrument already 0-for-B=1, the honest conclusion is that B=1 is at this
  structure's memory floor; (c) p1z + p1y ≈ 55–65 ≪ p1 → neither stream
  alone but their COEXISTENCE in 1 MB L2 is the cost — the two-sweep's
  in+out working set is the wall, no intra-structure fix exists, and the
  cell should be declared closed at ~113.
* **B=1**: pick `pw4 inplace pf=0 (nc=14)` and **111–115** (unchanged) unless
  pf=7 clears the 3% band, in which case **104–111** and the RFO story is
  confirmed. I put the odds against pf=7 (five rounds of pf-instrument
  rejections at B=1), but it is the last unfielded shape and branch (a)
  would make it the mechanism.
* **B=4**: 124–130, pick pf0-class; the verdict says the residual spread
  here is the harness's, so the median is the number to read.
* **B=32**: pick `pw4 inplace pf=2 pfwd=1296` — if wallaby's −3.7% in-arena
  transfers even at half strength, **159–165** (from 165.182), which takes
  the cell back from mixedradix's 162.137. If the node prefers 2592 or 5184,
  the pick string says so and the r5 pacing arithmetic gets its answer
  either way.
* **B=256**: pick pf=2-class with pfwd read from the string; **176–184**
  (from 181.643) — the same mechanism at the cell's traffic floor has less
  room; parity is the null.
* Setup ≤ ~4.5 s at B=256; rel_l2 fingerprints unchanged (3.58–3.59e-16).

### Next

1. **Read p1z/p1y off the node leaderboard first** — whichever branch fires
   decides r12: (a) → keep/extend pf=7 (maybe a phase-2-side prefetchw for
   inplace's re-read is next); (b)/(c) → stop hunting B=1 mechanisms at this
   geometry and say so; the two-sweep in+out working set (1.5 MB) against
   1 MB L2 is structural, and the only structure with a smaller resident set
   would be a (never-built) three-pass with a HALF-volume intermediate,
   which r1 already showed loses more to the extra sweep than L2 residency
   returns.
2. If pfwd=1296 is picked and B=32 moves: sweep one step finer ({864, 1728})
   next round; if 2592 holds on the node, the wallaby delta was
   memory-system-specific — record and stop.
3. The B=4 bimodality is the harness's (r10 verdict §3b); if it persists at
   nc=14 with identical picks, repeat the ask for median-based reads at the
   tight L=36 cells rather than building anything.
