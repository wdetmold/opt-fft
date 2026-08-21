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
