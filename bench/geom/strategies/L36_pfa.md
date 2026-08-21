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
