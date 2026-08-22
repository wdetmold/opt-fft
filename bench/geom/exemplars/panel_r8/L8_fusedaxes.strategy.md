# strategies/L8_fusedaxes.md

Geometry **L = 8** (8^3 = 512 complex doubles = 8 KiB per volume).
File: `impl/L8_fusedaxes.c`. `fft3d_name()` = `L8_fusedaxes`.

---

## Round 1 (panel round 1)

### Technique

Fuse all three axes: one trip in from memory, one trip out, and an 8 KiB L1 scratch in
between. **The SIMD width is filled by a spatial axis, never by the batch**, so B=1 is
exactly as wide as B=2048 and there is no batch-major repacking anywhere.

Split complex. One 64-byte vector (`v8d`, zmm on the node) holds the 8 values of one
axis, so every 8-point DFT is 8 lanes wide with *lane-invariant* twiddles and contains
zero shuffles. The volume is carried in two lane assignments; changing assignment is an
in-register transpose.

```
phase A, per x-plane (8 iterations)          lane = z, register index = y
    a z-pencil is 8 contiguous complex = 2 vectors;
    vunpcklpd/vunpckhpd split it into Re/Im, lane l holding z = PI[l],
    PI = (0,4,1,5,2,6,3,7)                                     [2 shuffles/pencil]
    y-axis DFT: elementwise across the 8 registers             [0 shuffles]
    store to the 8 KiB scratch, reindexed [y][x]

phase B, per y-slab (8 iterations)           lane = z, register index = x
    load 16 vectors from scratch (contiguous)
    x-axis DFT: elementwise                                    [0 shuffles]
    trans8 on Re and on Im: (reg=x,lane=z) -> (reg=z,lane=x)   [2 x 24 shuffles]
    z-axis DFT: elementwise                                    [0 shuffles]
    untrans_interleave over all 16 registers: the inverse
    transpose AND the complex re-interleave in one network     [48 shuffles]
    16 stores, straight into the driver's interleaved layout
```

**Index algebra (this is the part to reuse).** Three non-destructive 2-in/2-out lane
primitives, written as bit maps on (register bit r acted on; lane bits l2,l1,l0):

| name | instruction | masks (2-input, 0-15) | effect |
|---|---|---|---|
| `T1` | `vunpcklpd`/`vunpckhpd` | `0,8,2,10,4,12,6,14` / `1,9,3,11,5,13,7,15` | `r <-> l0` |
| `T2` | `vshuff64x2 (0,1|0,1)`/`(2,3|2,3)` | `0,1,2,3,8,9,10,11` / `4,5,6,7,12,13,14,15` | `r <-> l2` |
| `T3` | `vshuff64x2 (0,2|0,2)`/`(1,3|1,3)` | `0,1,4,5,8,9,12,13` / `2,3,6,7,10,11,14,15` | `r -> l2 -> l1 -> r` |

*There is no non-destructive primitive that swaps `r` with `l1`* (it would need blocks 0
and 1 of the destination to come from different sources, which `vshuff64x2` forbids) —
that is why the textbook transpose needs `vpermt2pd` in its middle stage, and why using
`T3`'s 3-cycle instead is the whole trick.

* `trans8` = `T2` on register bit 2, `T3` on bit 1, `T1` on bit 0. Verified map:
  output register `j` holds `z = PI[j]`; output lane `l` holds `x = (0,1,4,5,2,3,6,7)[l]`.
  The register permutation is absorbed by calling `dft8s` through
  `piinv = (0,2,4,6,1,3,5,7)`; the lane permutation is never observed.
* `untrans_interleave` (16 registers, index `m = ri*8 + k2`) = `T3` on k2 bit 0, `T3` on
  k2 bit 1, `T1` on the real/imag bit. That lands `(l2,l1,l0) = (k2_1, k2_0, ri)`, i.e.
  **exactly interleaved complex**, and leaves each output register as a ready-to-store
  half-pencil; the `(k0, half)` each one belongs to is the `out_off[16]` table.
  Derivation and full verification: the symbolic simulation in the session
  (`stage(regs, bit, prim)` over labelled lanes) — reproduce it before changing a mask.

Only the combination `T2,T3,T1` (and `T3,T3,T1` for the 16-register network) is a valid
transpose; `T3` then `T2` is **not** (l1 never receives a register bit). The reachable
lane permutations `sigma` of a 3-stage non-destructive network are only `identity` and
`swap(l1,l2)` — never a 3-cycle — which is why the *interleave* on the way out cannot be
made an `unpck` pair on its own and had to be fused into the transpose instead.

### Derivation / operation count

8-point complex DFT, radix-8 DIF, every trivial twiddle written out:

```
stage 1   t_j = x_j + x_{j+4}, s_j = x_j - x_{j+4}                  16 adds
twiddles  b_j = s_j W8^j ; W8^0 = 1 free, W8^2 = -i free (rename +
          sign folded into the DFT4 sums), W8^1 and W8^3 = 2 mul +
          2 add each                                              4 mul + 4 add
DFT4(t) -> X[2k]                                                    16 adds
DFT4(b) -> X[2k+1]  (the internal W4 = -i is free in split layout)  16 adds
                                                        total  52 adds + 4 mul = 56
```

56 real flops is the published optimum (Burrus T7.1 = T9.1 radix-8 = split-radix =
FFTW `n1_8`; split-radix closed form `4n log2 n - 6n + 8 = 56` at n=8), and
`LITERATURE.md` §3.3 says stop there. GCC contracts two of the multiplies into FMAs
(`u1r = SQ*A + SQ*B` etc.), emitting **54** vector FP instructions per DFT.

Per 8^3 volume: 3 axes x 64 lines = **192 line DFTs = 10 752 real flops**, issued as

| | count | port on the node |
|---|---|---|
| vector FP instructions | 1296 (24 x 54; **eight useful lanes each, zero waste**) | p0 (p0+p5 on a 2-FMA SKU) |
| vector shuffles | 896 = 128 deinterleave + 384 transpose + 384 fused untranspose/interleave | p5 only |
| vector loads / stores | 256 / 256 | p2,p3 / p4,p7 |
| register copies, spills | **6 static / 0** | — |

Nominal `5 N log2 N` yardstick (what the driver prints): 23 040 flops, i.e. the driver's
GF/s figure is 2.14x the honest one.

Measured static mix of the emitted AVX-512 code (`gcc -O3 -march=cascadelake`,
`volume_fused`): 162 arith + 114 shuffle + 64 vmovupd + 6 vmovapd, `0` stack traffic —
the §4.1 spill worry does not materialise, because only 16-24 vectors are live at a time
and the node has 32.

### Layout and SIMD decisions

* **Split complex, spatial axis in the lanes.** §04's `A (x) I_v` argument, but with the
  vector index taken from the *z axis* rather than the batch. Consequence: `x(-i)` is
  free everywhere (register rename + a sign folded into the next add), all twiddles are
  lane-invariant broadcasts, and the same kernel is full width at B=1. The cost is the
  two lane-assignment changes, which is what the transposes are.
* **One code path for every ISA.** GCC/Clang generic 64-byte vectors: `v8d` becomes zmm
  on the node and a 2 x ymm pair on the AVX2 dev machine with *identical arithmetic in
  identical order*, so what is verified locally is what runs there. No `#ifdef` around
  any arithmetic; the only guarded code is the optional `_mm512_stream_pd`.
* **Every shuffle in a 3-operand non-destructive form.** Port 5 is the scarce resource
  and a destructive `vpermt2pd` costs a `vmovapd` that also lands on port 5. See the
  failure list.
* **Scratch = 8 KiB** (`sr[64][8]`, `si[64][8]`, indexed `[y*8+x]`), so in + out + scratch
  = 24 KiB against a 32 KiB L1d. No padding: the scratch is addressed by register index,
  the volume stride never appears in an inner loop, so §4.5's power-of-two set-conflict
  hazard is structurally absent (this is §05's framing of the fix, not §04's padding).
* **Non-temporal final store** when `16384 * batch > 26 MB` (past the node's 22 MiB L3),
  guarded by a runtime 64-byte alignment re-check on `out`. Every half-pencil offset is
  `1024*k0 + 128*y + 64*half` bytes, so all NT stores are full-line and correctly
  aligned. At small batch NT is *off*: the driver re-writes the same 8 KiB `out` buffer
  every iteration and it should stay in L1.

### What was measured

**Local machine is Haswell E5-2680 v3 (AVX2 only, 2.5 GHz, shared 48-core login node),
so local absolute numbers are NOT node predictions.** Per transform, min of 20 samples:

| build | B=1 | B=64 | B=2048 |
|---|---|---|---|
| gcc 11.4 `-O3 -march=native` (mode 0) | 3.395 us | 3.755 us | 4.98 us |
| clang 18 `-O3 -march=native` (mode 0) | **1.214 us** | 1.456 us | 2.196 us |
| clang 18, mode 1 (3 passes via L1 scratch) | 1.247 us | 1.407 us | 2.137 us |
| clang 18, mode 2 (3 passes over the batch) | 1.633 us | 2.293 us | 2.719 us |

The gcc/clang gap is **entirely GCC 11's lowering of 8-lane 64-byte shuffles without
AVX-512**: it emits element-wise `vmovhpd`/`vmovlpd`/`vmovsd` and **1100 stack
spill/reload pairs per volume**; clang lowers them to `vperm2f128`/`vunpck` pairs. On
AVX-512 the two compilers are equivalent (gcc 162 arith + 114 shuffle + 6 copies; clang
168 arith + 114 shuffle + 0 copies, clang not contracting the twiddle FMAs). **So the
graded build does not care, but do not tune on local gcc timings.**

**Node prediction** — `llvm-mca-18 -mcpu=cascadelake` run on the actually emitted
AVX-512 loop bodies (the only honest proxy available without an AVX-512 machine):

| | phase A x8 | phase B x8 | per volume | at 2.30 GHz |
|---|---|---|---|---|
| mca simulated cycles | 40.1 | 127.2 | **1338** | **0.582 us** |
| mca Block RThroughput (ideal) | 28 | 96 | 992 | 0.431 us |
| analytic bound, 1 FMA unit (p0 = 1296 FP) | | | 1296 | 0.564 us |
| analytic bound, 2 FMA units (p0/p5 balance) | | | 1120 | 0.487 us |

Against the round-1 SOTA bar (MKL 2022, isolated node): **0.653 us at B=1**, so the
expectation is 10-25% ahead at B=1. At B=2048 MKL is 1.349 us = 12 GB/s of
read+write traffic per volume, i.e. already DRAM-bound; this kernel moves the optimal
8 KiB in + 8 KiB out with NT stores (16 KiB/volume against 24 KiB if the RFO is paid),
so B=2048 is a bandwidth tie at best and a 1.5x win only if MKL pays the RFO.

Accuracy, every configuration tested (gcc and clang builds, modes 0/1/2, B = 1, 8, 64,
2048, 4096): `rel_l2` = **2.29e-16 to 2.32e-16**, `rel_max` <= 3.3e-16, tolerance 1e-12.
Repeatability: `--warmup 0` and `--warmup 50` outputs are **bit-identical**.

**The panel's §4.3 question ("is axis fusion worth 3x or 3%?") answered for L=8:**

* fused vs **three passes through the same L1 scratch** (mode 1: identical arithmetic,
  identical shuffle count, one extra 8 KiB L1 round trip = +128 stores +128 loads):
  **+2.7% at B=1, and nothing at all (-3%, i.e. noise) at B=64 and B=2048.**
* fused vs **three separate passes over the whole batch in the driver's interleaved
  layout** (mode 2, the naive row-column shape): **+34% at B=1, +57% at B=64, +24% at
  B=2048** — but most of that is not memory traffic, it is the 3x de/interleave work
  (mode 2 spends 1408 shuffles/volume against 896) plus the extra `vmovupd` traffic.

So §07 §1.4's TurboFNO prior (3-5%) is the right one and §05 §5.4's "3x arithmetic
intensity" does **not** cash out at L=8, because the intermediate was already L1
resident. The thing actually worth fusing at this size is not the axes, it is the
*deinterleave and the transpose* — do the whole volume in one lane assignment sweep and
never write an interleaved intermediate.

**New AVX-512 information for the corpus (gap 6, which had no measurement at all).**
On a Cascade Lake SKU with **one** 512-bit FMA unit (Gold 52xx is believed to be one;
Gold 62xx/Platinum are two), 512-bit FP issues 1/cycle on port 0 = 8 doubles/cycle,
which is *exactly* what two 256-bit ops/cycle on ports 0+1 already give. **AVX-512 buys
no FP throughput at all on such a part.** Its entire value at L=8 is (a) the shuffle
network — `vshuff64x2`/`vpermt2pd` do in one port-5 op what AVX2 needs 3-4 for, and this
kernel is shuffle-heavy — and (b) 32 registers, which is what keeps a 16-vector working
set spill-free. Hence AVX-512 here even if it downclocks: the AVX2 form of this kernel
needs 2048+ port-5 ops per volume against 896, so the licence penalty would have to
exceed ~2x to flip the decision.

### What was tried and did NOT work

1. **The textbook 8x8 transpose network (`r<->l2`, `r<->l1`, `r<->l0`) plus a
   `vpermt2pd` interleave.** The middle stage has no non-destructive encoding, so gcc
   inserted a `vmovapd` per use: **38 static register copies, and port 5 pressure of
   147 slots per phase-B iteration against only 112 shuffles** (llvm-mca). Replacing it
   with the `T3` 3-cycle network took phase B from 147.3 to 127.2 cycles and the volume
   from 1517 to 1338 cycles (-12%). *The copies, not the shuffles, were the cost.*
2. **Separate transpose-back then interleave** (24+24+16 = 64 ops) instead of the fused
   48-op `untrans_interleave`: **+16 ops per phase-B iteration = +128 per volume**, all
   on port 5. The fused form is 3 bit-swaps over 16 registers; the split form does 3 over
   8 registers twice, then a 4th pass for the interleave.
3. **Doing the z axis with in-lane butterflies instead of transposes** (`vshuff64x2` +
   `fmadd` with a per-lane sign vector, twiddles as per-lane `c,s` broadcasts). Analytic
   count: 384 shuffles + 896 FP per volume against 768 shuffles + 448 FP. On a 1-FMA
   part port 0 binds, so this is **1792 vs 1344 FP instructions per volume, 33% worse**.
   The reason is that a lane-dependent twiddle costs 4 ops per pencil in split layout
   where an elementwise one costs 1. Rejected on the count.
4. **Interleaved complex in the lanes (4 complex per zmm) for the x and y axes**, which
   would remove the 256 de/interleave shuffles: the FP count is identical (56 per 8
   lines, the `x(-i)` folds into FMAs) but each 8-point DFT gains 5 shuffles per 4 lines
   = +160/volume, so the total goes **896 -> 928**. Worse once the fused output network
   exists. Rejected on the count.
5. **Batch in the lanes** (the §5 tier-1 recommendation, for the batched regime only):
   repacking 8 volumes into batch-minor split costs one 8x8 transpose per 8 doubles per
   8 volumes = **768 shuffles per volume in and out — the same 768 the axis transposes
   cost** — while needing 64 KiB of scratch (2x L1, so it runs out of L2 not L1) and not
   working at B=1 at all. There is no shuffle saving to be had from batching at L=8.
   Rejected analytically; someone should still measure it, since it has a longer
   dependency-free run.
6. **Trusting local timings.** gcc 11 on AVX2 makes this source 2.8x slower than clang 18
   on the *same machine and same source* (3.395 vs 1.214 us at B=1) purely through
   64-byte-vector shuffle lowering, and both are ~2-6x off the AVX-512 node. Everything
   quantitative above about the node comes from llvm-mca on the emitted node asm.
7. **`prefetchnta` on the next volume's first two lines** in the NT-store path: removed.
   The read stream is perfectly sequential so the L2 streamer covers it, and
   `prefetchnta` risks bypassing L2 for data that phase A reads twice per line.
8. **Padding.** Not needed and not used: the scratch is addressed by register index and
   the volume stride never enters an inner loop, so §04 §7.3's "z-stride = exactly one
   L1 set" pathology (which is created by the batch-minor layout §04 recommends) never
   arises. This is §05 §6's framing and it costs 0% memory rather than 27%.

### Next

llvm-mca's bottleneck analysis on the final phase-B body says **95.6% resource pressure,
11.8% data dependencies** — this kernel is *port*-bound, not latency-bound, so ILP
tricks are worth little. And both ports are at their floor:

* **Shuffles are provably minimal at 896/volume.** A 2-input lane shuffle can resolve
  only one bit of source-register index, so moving `k` register bits into the lanes costs
  `k` stages x (number of output registers). Phase A needs 1 bit (re/im) over 16
  registers = 16; phase B needs 3 bits in (transpose) and 3 bits out
  (untranspose+interleave) over 16 registers = 96. There is nothing left to remove.
* **FP is at the published optimum** (54 instructions per 8-point DFT after GCC's two
  twiddle FMAs, against a 56-flop lower bound), with all eight lanes useful.

So the ordered list is:

1. **Settle the FMA-unit count on the node** — time a long independent `vaddpd %zmm`
   chain, or read `AVX-512 FMA units` off the SKU. With one unit port 0 binds at 1296
   cycles (0.564 us) and this kernel is finished; with two units port 5 binds at ~944 and
   it is worth revisiting failure-list item 4 (interleaved lanes trade shuffles the wrong
   way *only* when FP is scarce) and item 3.
2. **Software-pipeline phase A of volume b+1 against phase B of volume b** (16 KiB
   double-buffered scratch). Not for ILP but for *port balance*: A is 54 FP : 16 shuffles,
   B is 108 : 96, so interleaving them gives 162 : 112 per combined iteration and lets
   port 5 absorb FP overflow on a 2-FMA part. Worth ~15% there, ~0% on a 1-FMA part.
3. **Check whether MKL uses NT stores at B=16384.** If it pays the read-for-ownership and
   this kernel does not, that is a 24 KiB -> 16 KiB traffic win (1.5x) in the batched
   regime the leaderboard scores separately. Conversely, verify on the node that NT is
   not *hurting* at B=2048 (33 MiB working set against a 22 MiB L3 is only 1.5x over, so
   the threshold `16384*batch > 26 MB` is the one number here that was chosen by
   arithmetic rather than by measurement).
4. **Measure the batch-in-the-lanes variant anyway** (failure-list item 5). It costs the
   same 768 shuffles per volume on paper, but its dependency graph is 8 independent
   volumes wide instead of one, so if the node turns out to be latency-bound in a way
   llvm-mca does not model, that is where the slack would be. It cannot serve B=1.
5. **Do not spend time on arithmetic.** 3 x 64 x 56 real flops is the optimum for
   row-column at L=8, vector-radix saves 4.2% of operations at 8x8 while destroying the
   zero-shuffle property (§03 §2.3), and this kernel is not FP-bound anyway on a 2-FMA
   part.

---

## Round panel_r2

### Where round 1 landed (node, panel_r1)

B=1 **0.570 µs — first**, but a statistical tie (batchsimd 0.573, radix8 0.576, all
within the 1–2% spread; every entry is sitting on the same port-0 bound, exactly as the
r1 analysis predicted for a 1-FMA Gold 5218). B=64 **0.625 — first** (MKL 0.710).
B=2048 **1.548 — fourth** (MKL 1.338, batchsimd 1.432). B=16384 **1.899 — fourth**
(mkl2026 1.772, batchsimd 1.782). So this round was spent entirely on the memory-bound
batched regime; the arithmetic kernel is untouched and B=1/B=64 run byte-identical code.

### What changed

**1. Software prefetch of the next volume's input, issued from phase B — idea adopted
from L8_batchsimd** (its strategy record and its `PF()` macro: prefetch the next
volume's 128 lines because the L2 streamer stops at each 4 KiB page boundary and an
8 KiB volume spans two pages). My r1 record (failure item 7) had dismissed prefetch on a
sequential-stream argument; batchsimd's node numbers (1.432 vs my 1.548 at B=2048 with
the same NT stores) show that argument was wrong on the node, and my own wallaby
measurement below shows it is worth ~21% in the DRAM regime. Placement is the one part
that is mine: the 16 prefetches per y-iteration sit in phase B's loop bodies, which are
~900 cycles of pure L1-resident compute with the load ports idle, so they cost nothing.
Differences from batchsimd: **prefetcht1, not prefetcht0** (measured, below), distance
one volume, clamped to the last volume so no prefetch ever leaves the mapping.

**2. Both regime switches are now L3-relative, decided at plan time** via
`sysconf(_SC_LEVEL3_CACHE_SIZE)` (fallback: the node's 22 MiB): NT stores when
`in+out > 1.18*L3` (identical to r1's fixed 26 MB on the node, but no longer wrong by
2.7x on a 60 MiB-L3 machine), and **prefetch iff NT** (batchsimd's rule). The NT gate
guarantees `in > 0.59*L3` whenever prefetch runs, which keeps prefetch out of the regime
where it is actively harmful (see failures). Compile-time overrides `-DL8_NT=0/1`,
`-DL8_PFSEL=0/1`, `-DL8_PF=0`, `-DL8_PF_LOC`, `-DL8_PF_DIST` exist so the monitor can
A/B any of this on the node with one flag.

### Operation count

Unchanged: 1296 vector FP + 896 shuffles + 256 loads + 256 stores per volume, plus (NT
path only) **128 `prefetcht1` uops per volume** on ports 2/3, which have ~1000 free
slots per volume. Nothing else moved.

### What was measured (wallaby, Gold 6448Y SPR, 60 MiB L3)

**Caveat: wallaby was bimodal this session** — two persistent speed states ~1.7x apart
(B=1 is 0.342 µs in one and 0.669 µs in the other, sd inside a run as low as 0.03%).
Every A/B below is best-of-3 alternating runs, and the decisive pairs were also
same-state back-to-back. Per-transform, best observed:

| B | regime on wallaby | r2 (auto) | r1 code | note |
|---|---|---|---|---|
| 1 | plain | 0.342 µs | same path | unchanged code |
| 64 | plain | 0.370 µs | same path | unchanged code |
| 2048 | in = 0.27 L3, all L3-resident | 0.469 µs (NT, no PF) | 0.510 | NT-gate change only |
| 5632 | in = 0.76 L3 (**node-B=2048 analog**, WS 1.53 L3 vs node's 1.45) | 0.494 µs | — | NT+PF ties NT-noPF here |
| 16384 | in = 2.1 L3, DRAM-bound | **0.588 µs** (NT+PF t1) | 0.749 | **+21%, the round's result** |

Correctness every configuration (auto and all four forced NT×PF combos):
rel_l2 = 2.29e-16 … 2.33e-16, tolerance 1e-12; output bit-identical across runs.

The prefetch-hint sweep at B=16384 (first pass, same slow state): **t1 9.77 ms**,
t0 13.5–15.2 ms, nta 16.1 ms, no-PF 11.8 ms. Confirmed best-of-3 alternating:
PF-t1 {9.40, 10.37, 11.54} vs no-PF {12.27, 12.64, 14.72} — PF wins every pairing.

### What was tried and did NOT work

1. **prefetcht0 (batchsimd's hint) — on wallaby.** 13.5–15.2 ms at B=16384 against
   t1's 9.77 ms, i.e. *worse than no prefetch at all* (11.8). On SPR, pulling the next
   volume into L1 while NT stores drain fights for fill buffers; into L2 it does not.
   Caveat for the monitor: batchsimd's t0 demonstrably helped on the *node* (CLX), so
   t0-vs-t1 on CLX is still open — it is `-DL8_PF_LOC=3` if worth one run.
2. **prefetchnta for the input stream**: 16.1 ms at B=16384, the worst variant. The
   driver re-reads `in` every call, so bypassing the cache hierarchy on a buffer you
   want resident is exactly backwards.
3. **Prefetch distance 2 volumes** (`-DL8_PF_DIST=2`): 10.1/10.8 ms vs distance 1's
   9.2–9.4 ms. One volume (~1300 cycles) already covers DRAM latency ~4x; distance 2
   just holds lines in L2 longer for no benefit.
4. **Prefetch when the input is L3-resident** (the old fixed-26 MB NT gate put wallaby's
   B=2048, in = 0.27 L3, on the NT+PF path): **1511 vs 959 µs back-to-back same-state —
   a 1.6x disaster**, sd < 0.7% on both sides. 128 prefetches/volume clog the
   LFB/L2-queue path that the NT stores drain through, and buy nothing because the reads
   were L3 hits anyway. This is why the shipped rule ties prefetch to the NT gate rather
   than enabling it unconditionally.
5. **Regular stores at the node-B=2048 analog** (B=5632, WS = 1.53 L3, forced
   `-DL8_NT=0`): NT wins every same-state pairing, best-of-3 2833 vs 5098 µs. So MKL's
   1.338 at node B=2048 is *not* explained by it skipping NT in favour of L3-resident
   output, at least not in a way this kernel can copy. The r1 "Next" item 3 question is
   answered: keep NT there.
6. **Tuning on single runs.** The bimodal machine state (1.7x!) makes any single number
   meaningless this session; the first prefetch measurement looked like a 28% regression
   (15.2 ms) purely because of it. Everything above is best-of-3 alternating or
   same-state back-to-back.

### Node predictions

* B=1, B=64: byte-identical code paths to r1 → 0.570 / 0.625 µs stand.
* B=2048: NT+PF-t1 now runs (r1 ran NT only). batchsimd's t0 prefetch was worth 7.5%
  on the node here (1.432 vs my 1.548); t1 should match or beat that → **≤1.43 µs**,
  MKL at 1.338 may still lead — the honest remaining gap is read-side scheduling, see Next.
* B=16384: wallaby says +21% over r1's 1.899 → **~1.55–1.6 µs**, which would clear
  mkl2026's 1.772 and batchsimd's 1.782 to first place.

### Next

1. **If B=2048 still trails MKL on the node**: the one structural lever left is
   software-pipelining phase A of volume b+1 against phase B of volume b (double-buffered
   16 KiB scratch), not for port balance (dead idea on a 1-FMA part, r1 record) but to
   put *demand loads* of the next volume under phase B alongside the prefetches, halving
   exposure to any prefetch that arrives late.
2. **Ask the monitor for one flag run if the batched cases disappoint**:
   `-DL8_PF_LOC=3` (t0 on CLX) at B=2048/16384 settles the hint question on the machine
   that actually scores.
3. Do not touch B=1: three entries within 1% for two rounds means the port-0 floor is
   real; the only thing that would move it is a 2-FMA machine.

---

## Round panel_r3

### Where round 2 landed (node, panel_r2)

B=1 **0.573 µs — first** (batchsimd 0.598 after its 2× wallaby win turned into a node
regression; radix8 0.583 after its 54→52 codelet cut measured +0.7% *worse*). B=64
0.638 — statistical tie for first. **B=2048 1.503 — fourth**, the real gap: batchsimd
1.205, MKL 1.325. B=16384 1.614 — second (batchsimd 1.557). The VERDICT's L=8 reading:
two rounds of instruction-cutting moved B=1 not at all, so the compute regimes are a
wall and the batched DRAM/L3 regimes are the only live territory. This round was spent
entirely there. **The arithmetic kernel is byte-identical for the third round running.**

### What changed

**1. Fixed a genuine r2 bug: the code did `pf = nt` while its own comment promised
"prefetch only when the INPUT alone exceeds L3".** Consequence on the node at B=2048:
input = 16 MiB < 22 MiB L3, i.e. reads were L3-resident, and the kernel prefetched them
anyway — exactly the regime my own r2 wallaby measurement (failure item 4) showed
costing **1.6×** by clogging the fill buffers the NT stores drain through. That is a
plausible mechanism for most of the 1.503-vs-1.205 gap to batchsimd at node B=2048.
The rule-based default is now: NT when `in+out > 1.18·L3`, prefetch (t1) only when
`in alone > 1.05·L3`.

**2. Store policy and prefetch are now MEASURED at plan time, not ruled.**
`fft3d_create()` times all six variants — (plain | NT) × (no-pf | pf-t1 | pf-t0) —
on a surrogate batch on the machine that scores, whenever `in+out > 0.25·L3`.
Motivation is the panel_r2 VERDICT: NT stores *lose on the node at L=6 at every batch
size* (two independent entries, against wallaby showing NT winning 1.6×), and §4.6's
"search beats model" scored twice. The store-policy crossover cannot be predicted from
arithmetic on this evidence, so I stopped trying. Protocol (drift lessons borrowed from
**L8_batchsimd** r2 and **L6_pfa**): candidates round-robin interleaved so slow machine
states hit all of them; **one untimed pass per trial to set the candidate's own cache
state** (my addition — plain and NT leave different L3 contents behind, so timing
candidate c after candidate c′ biases the sample); min of 5 timed trials each; 2%
hysteresis toward the rule default. Surrogate batch capped at `min(B, 4·L3/16 KiB, 8192)`
volumes (same residency regime as any larger batch); deterministic LCG fill; setup cost
~0.1 s at B=2048, ~0.7 s at B=16384. Tuner skipped below 0.25·L3, so **B=1 and B=64 run
byte-identical code to rounds 1–2** and their plan time stays 0.

**3. The pick is reported through `fft3d_description()`** (static buffer written by
create; the driver reads description after create, so the per-case choice lands in the
`t_*.json`). This was the VERDICT's cross-cutting ask #2, credited to L17_winograd /
L6_pfa — the entries that plumbed their picks produced the round's only decisive
node-vs-wallaby answers. Format: `"... B=%d pick=nt+pf_t0 (tuned|rule|forced)"`.
Compile-time forces (`-DL8_NT`, `-DL8_PFSEL`, `-DL8_PF_LOC=3`) still work and skip the
tuner (verified: setup=0.000 under forced flags).

**What was deliberately NOT done: the 54→52 codelet.** L8_radix8 measured exactly that
change on the node in r2: 0.576 → 0.583 µs, i.e. worse. The VERDICT's L=8 instruction
("stop cutting instructions") is followed; per the brief, a documented dead end is not
re-run.

### Operation count

Unchanged: 1296 vector FP + 896 shuffles + 256 loads + 256 stores per volume; prefetch
variants add 128 `prefetcht0/t1` uops per volume on ports 2/3 (~1000 free slots). The
only new execute-path cost is one `switch` dispatch per *batch pass* (not per volume).

### What was measured (wallaby, Gold 6448Y SPR, 60 MiB L3, best of ≥3 runs)

| B | regime | this round | r2 best | pick (tuned unless noted) |
|---|---|---|---|---|
| 1 | plain, no tuner | **0.343 µs** | 0.342 | plain (rule) — unchanged code |
| 64 | plain, no tuner | **0.354 µs** | 0.370 | plain (rule) — unchanged code |
| 2048 | ws = 0.53·L3 | **0.441 µs** | 0.469 | nt+pf_t0, 3/3 runs |
| 5632 | ws = 1.5·L3 (node-B2048 analog) | **0.434 µs** | 0.494 | nt+pf_t0 (0.717 vs rule-default nt 0.870 in-tuner, **−18%**) |
| 16384 | ws = 4.4·L3, DRAM | **0.633 µs** | 0.588 | nt+pf_t0 twice, nt+pf_t1 once (within 2% hysteresis) |

Correctness everywhere (all six variants exercised by the tuner, plus forced-flag and
mode-1/2 and AVX2-local builds): rel_l2 = 2.28e-16 … 2.33e-16, tolerance 1e-12; output
bit-identical across runs. The B=16384 spread (0.633 vs r2's 0.588) is wallaby DRAM
contention on the day, not the code: the in-tuner numbers for the same variant drifted
0.496 → 0.531 µs across runs an hour apart.

### What was tried and did NOT work

1. **Tuner gate at 0.55·L3.** First shipped value; at wallaby B=2048 (ws = 0.53·L3) the
   gate left the rule's *plain* pick in place and the driver measured **0.901 µs/vol vs
   0.469 for NT** — the store-policy crossover sits far below L3 capacity. Gate moved to
   0.25·L3. (Number that killed it: 0.901 vs 0.469, 1.9×.)
2. **Trusting any single wallaby run, again.** One B=2048 tuning run measured plain at
   0.735 while the two beside it said 0.49; one B=16384 driver min was 21% off its
   sibling. Every number above is best-of-3. The tuner's round-robin + own-state-pass
   protocol survived this: it picked nt+pf_t0 in all three B=2048 runs *including the
   slow-state one* (all six candidates slowed together, ordering preserved).
3. **t1-vs-t0 as a stable fact about a machine.** r2 measured t1 ≫ t0 on wallaby at
   B=16384 (9.77 vs 13.5 ms); this round's tuner measured t0 ≤ t1 there twice
   (0.531 vs 0.542/0.554). The hint preference is not even day-stable on a shared
   machine — one more reason it must be decided at plan time on the scoring node, and
   why both hints stay in the candidate set.

### Borrowed

* **L8_batchsimd**: round-robin interleaved trial protocol for the plan-time tuner
  (their r2 tuner rewrite, after their sequential tuner mis-picked a 1.5× slower mode).
* **L17_winograd / L6_pfa** (via the panel_r2 VERDICT): plumbing the tuner's pick into
  `fft3d_description()` so the monitor can read per-case choices from the JSON.
* The VERDICT's L=6 node result (NT loses on the node at every batch size) is the
  direct motivation for including plain-store candidates at every batched size rather
  than gating NT by arithmetic.

### Node predictions

* B=1, B=64: byte-identical code → **0.573 / ~0.64 µs** stand.
* B=2048: the honest statement is that the node now *chooses*; the r2 bug (prefetching
  an L3-resident input) is gone even in the rule fallback. If the L6 inversion carries
  to L=8, plain or plain+pf wins and lands near MKL's 1.325; if batchsimd's regime
  transfers, nt+pf_t0 wins (their t0 was worth 7.5% on the node in r1). Either way the
  1.503 outlier should close toward **≤1.35**. The pick will be in the description
  string — monitor, please read it off `t_*.json`.
* B=16384: nt+pf_t0 or t1 → **~1.55–1.62 µs**, parity with batchsimd's 1.557 (same
  16 KiB/volume NT floor; the remaining difference is prefetch scheduling).

### Next

1. **If node B=2048 still trails batchsimd**: software-pipeline phase A of volume b+1
   against phase B of volume b (double-buffered 16 KiB scratch) to put demand loads
   under compute alongside the prefetches. This is the one structural lever left and
   it is two rounds deferred.
2. **If the node picks a plain variant anywhere batched**: add plain+`prefetchw`-on-out
   candidates (hide the RFO instead of avoiding it) — cheap to add now that variants
   are a table.
3. **B=1 is frozen** until someone measures the node's actual clock under this code
   (the VERDICT's L=6 `perf stat -e cycles,ref-cycles` ask applies verbatim to L=8:
   0.573 µs is 1318 cycles at base but ~1900 at full turbo, and which one it is decides
   whether there is 30% headroom or none).

---

## Round panel_r4

### Where round 3 landed (node, panel_r3)

B=1 0.577 — second, inside the three-entry tie (batchsimd 0.570, radix8 0.583); the
B=1 wall stands for the third round. B=64 **0.642 — first**. B=16384 **1.580 — first**
(radix8 1.647, batchsimd 1.748), already within 16% of the 12 GB/s bandwidth bound per
the VERDICT. **B=2048 1.291 — third**: L8_radix8 took the cell at 1.243 with its new
3-pass shape whose one functional difference is that the output volume is written
*sequentially* (my fused phase B writes 8 interleaved streams, 2 lines per stream per
y-iteration). The VERDICT's §4.3 verdict is that at these sizes store *order* is worth
18% where pass-count fusion is worth ~3%, and its process lesson (from batchsimd's
regression) is: **add candidates, do not replace structures**. Both r3 predictions I
made landed; every code path this round keeps that discipline.

### What changed

**One change: a "seq3" shape joins the plan-time tuner as six new candidates
(shape × {plain,nt} × {no-pf, pf-t1, pf-t0}, 12 total), adopted from L8_radix8's
round-3 sequential-store 3-pass** (its node B=2048 win, 1.243 vs my 1.291, is the
measurement this round acts on). Per volume:

* **phase A unchanged** (deinterleave + y-axis DFT into scratch1 `[k1][x]`, lane=z).
* **pass B1, per k1 (8×):** 16 contiguous loads from scratch1 row k1, x-axis DFT
  along the registers (zero shuffles), 16 stores to a second 8 KiB scratch at slot
  `(k0*8+k1)*16` (re;im interleaved per slot, so B2 reads contiguously). The write
  column is strided 1 KiB but L1-resident (16 lines over 8 sets, 2 ways each — no
  conflict on an 8-way L1; radix8's 144-double padding question does not arise in
  this slot layout).
* **pass B2, per k0 (8×):** 16 contiguous loads → registers indexed k1, lane = z in
  PI order — *exactly the state my fused phase B is in after its x-DFT*, so the
  existing trans8 → piinv → dft8s → untrans_interleave chain is reused verbatim with
  k1 playing k0's role (the index algebra transfers by isomorphism; no new shuffle
  network was derived). The 16 output half-pencils all land in the one 1 KiB
  k0-plane and are stored in **ascending address order** (the r1 out_off table,
  sorted: zr0,zr4,zq0,zq4,zr2,zr6,... at offsets 0,8,16,...,120), so the volume's
  write stream is fully sequential across the k0 loop.

The fused shape stays candidates 0–5 and stays the rule default (the known-good
anchor for the 2% hysteresis). Compile-time force `-DL8_SHAPE=0/1` added beside
L8_NT/L8_PFSEL for node A/Bs. B=1 and B=64 remain **byte-identical to rounds 1–3**
(tuner still gated at ws > 0.25·L3; rule default is fused-plain below it).

### Operation count (seq3 shape)

Identical FP and shuffle bill to the fused shape: 1296 vector FP (24 × 54) and 896
shuffles (128 deinterleave in A + 8×(48 transpose + 48 fused untranspose/interleave)
in B2; B1 has zero). The cost is +128 loads +128 stores per volume (384+384 total),
all to L1-resident scratch — the same arithmetic radix8 paid. DRAM-facing traffic
unchanged: 8 KiB in + 8 KiB out (16 KiB with NT); only the write *order* changes.
Scratch grows 8 → 16 KiB; in + scr + out = 32 KiB against a 32 KiB L1 (radix8's
3-pass sits at ~25 KiB and won B=2048, so this is judged survivable; the NT
variants bypass the out allocation entirely).

### What was measured (wallaby, Gold 6448Y SPR, 60 MiB L3; the 2× bimodality from
r2/r3 is still present — mins quoted, in-tuner tables are same-process and drift-free)

| B | this round | pick | in-tuner fused-best vs seq3-best |
|---|---|---|---|
| 1 | **0.345 µs** | plain (rule, no tuner) | unchanged code |
| 64 | **0.356 µs** | plain (rule, no tuner) | unchanged code |
| 2048 (ws=0.53·L3) | **0.436 µs** | nt+pf_t0, both runs | 0.472 vs 0.489 — fused wins |
| 5632 (ws=1.5·L3, node-B2048 analog) | **0.439 µs** | **seq3-nt+pf_t0** | 0.441 vs 0.439 — seq3 edges it |
| 16384 (ws=4.4·L3, DRAM) | **0.620 µs** | nt+pf_t1 (hysteresis) | 0.499 vs 0.491 — seq3 fastest but inside the 2% band |

Two readings worth recording. (a) **The shape ranking inverts with residency even on
one machine** — fused wins at 0.53·L3, seq3 at 1.5·L3 and (marginally) at 4.4·L3 —
which is exactly why it ships as a tuner candidate and not a replacement. (b) With
**plain** stores the sequential write order is worth far more than with NT:
seq3-plain 0.920 vs fused-plain 1.172 at B=5632 (−21%), because the fused scatter
pays RFOs on 8 interleaved streams. If the node's L=6-style NT inversion ever shows
up at L=8, seq3 is the safety net.

Correctness: PASS at B = 1, 8 (forced seq3), 64, 2048, 5632, 16384 —
rel_l2 = 2.286e-16 … 2.334e-16 (tolerance 1e-12), rel_max ≤ 3.0e-16, output
bit-identical across runs everywhere, including the runs whose plan picked seq3-nt.
All 12 variants are exercised by every tuning create; the picked-variant output is
what check.py verifies, and picks covering seq3-nt+pf_t0 (B=5632), nt+pf_t0
(B=2048) and plain/forced-seq3 (B=1/8/64) all passed.

### What was tried and did NOT work

Nothing failed outright this round; the near-misses that matter:

1. **seq3 did not beat fused on wallaby at B=2048** (0.489 vs 0.472 in-tuner,
   ws=0.53·L3): when in+out fit L3 comfortably the store order is irrelevant and
   the extra 256 L1 accesses are pure cost. Do not expect seq3 to be picked below
   ~1×L3; that regime keeps the fused shape.
2. **At B=16384 the 2% hysteresis kept the rule default** (nt+pf_t1 0.499) over the
   measured-fastest seq3-nt+pf_t0 (0.491). Deliberate: the VERDICT's tuner-instability
   note says close calls should not flip on machine noise, and 1.6% is inside
   wallaby's day drift. On the node the same tournament runs on the machine that
   scores, which is the only table that matters.
3. **Software pipelining (phase A of vol b+1 under phase B of vol b) deferred a third
   time**, now deliberately: batchsimd's r2 double-buffer measured neutral, B=16384
   is already within 16% of the bandwidth bound (VERDICT §6), and the seq-store
   experiment was the round's one measured-on-node idea. If seq3 takes B=2048 and
   B=16384 still sits ≥1.55, pipelining is the only lever left and should be next.

### Borrowed

* **L8_radix8**: the sequential-output-store 3-pass structure and its node B=2048
  evidence (1.243), including the placement of the extra round trip in L1 and the
  "store order beats pass count" reading its r3 record and the VERDICT §4.3 agree on.
  The B2 reuse of my own untrans_interleave under the k0→k1 renaming, the ascending
  store schedule, and the re;im-interleaved scratch2 slot layout (contiguous B2
  reads, no padding question) are mine.
* **panel_r3 VERDICT**: "add candidates, do not replace structures" — the entire
  shape of this round.

### Node predictions

* B=1 **0.577** and B=64 **~0.64** stand (byte-identical code, tuner gated off).
* B=2048: the tuner chooses between fused nt+pf_t0 (measured 1.291 on the node in
  r3) and seq3-nt+pf_t0, whose store behaviour matches radix8's 1.243 winner while
  keeping my kernel. Predict **1.24–1.29**, seq3 picked. If seq3 is *not* picked,
  the shape difference between my B1/B2 split and radix8's pass structure is the
  residue to chase.
* B=16384: both shapes within 2% on wallaby; fused nt holds the cell at 1.580.
  Predict **1.55–1.62**, either pick — the cell is ~16% off the bandwidth wall and
  this round does not claim to move it.

### Next

1. **If seq3 takes node B=2048**: the L=8 batched story is closed to within the
   bandwidth bound; spend the next round on the deferred cross-volume software
   pipeline only if B=16384 still shows ≥0.2 µs over the 1.365 µs traffic floor.
2. **If the node ever picks a plain variant at batch** (L=6 keeps rejecting NT),
   check whether it picked seq3-plain — wallaby says sequential order is worth 21%
   there, and plain+prefetchw candidates (hide the RFO) become worth adding.
3. **B=1 stays frozen** pending the monitor's `perf stat -e cycles,ref-cycles`
   clock measurement (asked three rounds running, still the only thing that says
   whether 0.57 µs has headroom).

---

## Round panel_r5

### Where round 4 landed (node, panel_r4)

B=1 0.579 — third in the standing tie (batchsimd/radix8 0.570); four rounds now say the
compute regime is a wall. B=64 **0.623 — first** (the only cell I hold). B=2048 **1.313 —
third** and B=16384 **1.585 — second**, both cells taken by L8_radix8 (1.136 / 1.418)
with **plain stores + spread prefetch** — my own r2 prefetch placement, spread finer
(6/5/5 lines per pass iteration ~ 1 line per 11 cycles) and paired with its sequential
store order. The VERDICT's two headline facts for this round: (a) **NT stores lost on the
node in every batched L=8 cell** — plain+spread beat my NT+chunk configs even at a
256 MiB working set; (b) **the node's AVX2 clock is 3.89 GHz, not 2.30** (L6_unrolled's
probe), so every cycle count in rounds 1–4 of this record is off by ~1.7× and the B=1
"port floor" argument is unproven until someone measures the AVX-512 licence clock.
The VERDICT's L=8 priority: the B=64 L2 cliff (ws = 1.00 MiB = exactly the node's L2;
every entry is ≥9% slower per transform than at B=1).

### What changed (four things, all tuner candidates or unscored instrumentation —
the arithmetic kernel is byte-identical for the fifth round running)

1. **Spread prefetch placement, adopted from L8_radix8's r4 node wins** (1.136/1.418).
   The next volume's 128 input lines are issued t0, ~5–8 lines at the top of every loop
   iteration: 8/8 per iteration over the fused shape's 16 iterations (phase A covers
   doubles [0,512) of volume b+1, phase B [512,1024)), 6/5/5 over the seq3 shape's 24
   iterations (offsets 0/384/704, radix8's exact cadence). My old placement — 16-line
   chunks at the top of each phase-B/B2 iteration — measured 7–12% worse in-tuner on
   wallaby (fused-nt: spread 0.417 vs chunk-t1 0.474 at B=2048; seq3-nt: spread 0.481
   vs chunk-t0 0.519 at B=16384) and is retained only as the two NT continuity
   candidates that were node picks in r3/r4.
2. **Write-intent prefetch (prefetchw) of the next volume's OUTPUT lines, adopted from
   L6_unrolled / L6_pfa** (their `fused_pfw` won the L=6 DRAM cells; L6_pfa closed 13%
   at B=32768 by adopting it). With plain stores every output line pays an RFO;
   `__builtin_prefetch(p,1,3)` (emits prefetchw on CLX) issues it one volume early at
   the same spread cadence. Two candidates: fused+pfs+pfw, seq3+pfs+pfw. Wallaby data:
   pfw recovers ~40% of the plain-store RFO cost in the DRAM regime (fused-plain
   1.169 → 0.730 at B=16384-arena) but still loses to NT *on wallaby*; on the node,
   where plain already beats NT, plain+pfs+pfw is the natural challenger to radix8's
   plain+pfs. That comparison can only happen on the node; both are in the set.
3. **The tuner now covers the B=64 L2 cliff** (VERDICT ask). New gate: ws ≥ 0.5·L2
   (sysconf, fallback 1 MiB) tunes over a deliberately small plain-only set
   {fused, fused+pfs, seq3, seq3+pfs} with 3% hysteresis toward the four-round
   incumbent (fused-plain), surrogate = the exact batch so the tuner sees the driver's
   steady state, and a new inner repeat count (reps = 4096/bsur) so a timed trial
   covers ≥2 ms — a single B=64 pass is ~35 µs and untrustable. NT and pfw are
   excluded there by construction (L2/L3-resident output; radix8's r3 pf coin-flip
   cost 6.7% at this exact cell — small set = stable pick). Wallaby result: seq3+pfs
   wins the cell by 9% in-tuner (0.631 vs 0.691) and the driver confirms
   **0.337 µs/t vs r4's 0.356** — the first movement at B=64 in three rounds.
4. **A 256-bit AND 512-bit clock probe in create()**, extending L6_unrolled's method
   (serial FMA chain, latency 4), reported as `clk256=`/`clk512=` in
   `fft3d_description()` → leaderboard JSON. The 512-bit number does not exist
   anywhere in the corpus for the node (LITERATURE §4.8 gap 6) and decides whether
   B=1's 0.57 µs is at its 1296-cycle port floor: 0.579 µs = 1332 cycles at 2.30 GHz
   (floor, B=1 closed) but 2252 cycles at 3.89 GHz (40% headroom, and a ymm kernel or
   mixed-width shapes à la L17_rader become the next move). Probe details that cost
   three iterations to get right (see failures): 5 ms chunks, track the max, stop on
   100 ms max-stagnation, ~0.2–0.5 s, unscored.

Candidate set is now 10 (was 12): {fused, seq3} × {plain, plain+pfs, nt+pfs} +
{fused+pfs+pfw, seq3+pfs+pfw} + the two NT-chunk continuity variants. Retired: all
plain+chunk variants (never picked anywhere in three rounds of node and wallaby
tables) and NT-no-pf (dominated by NT+spread in every wallaby table this round).
Regime anchors for the hysteresis: ws > 1.18·L3 → seq3+pfs (the node's r4 evidence via
radix8); 0.25–1.18·L3 → fused-nt+pfs (wallaby's stable pick; no scored node cell in
this band); L2 band → fused-plain; below → fused-plain untuned, so **B=1 runs the
same instructions as rounds 1–4**. `-DL8_VARIANT=0..9` forces any variant directly
and supersedes the old L8_NT/L8_PFSEL/L8_SHAPE triplet (mapping in the file header);
`-DL8_PF=0` and `-DL8_TUNE=0` unchanged.

### Operation count

Per volume, unchanged: fused 1296 vector FP + 896 shuffles + 256 loads + 256 stores;
seq3 +128 loads +128 stores, all L1-resident. Spread variants add the same 128
`prefetcht0` uops as before, just distributed; pfw variants add 128 `prefetchw` uops
(ports 2/3, ~1000 free slots — and prefetchw dispatches like a store-address uop, so
it also has port 7 available). No arithmetic change anywhere.

### What was measured (wallaby, Gold 6448Y SPR, 60 MiB L3; bimodal again this
session — two states ~1.9× apart at B=64/2048; mins and same-process in-tuner tables
quoted)

| B | driver best this round | r4 best | pick | key in-tuner comparison |
|---|---|---|---|---|
| 1     | **0.343 µs** | 0.345 | fused (rule, untuned) | unchanged code |
| 64    | **0.337 µs** | 0.356 | **seq3+pfs (tuned, new)** | seq3+pfs 0.631 vs fused 0.691 (−9%) |
| 2048 (0.53·L3) | **0.425 µs** | 0.436 | fused-nt+pfs | spread vs chunk under NT: 0.417 vs 0.474 (−12%) |
| 5632 (1.5·L3)  | 0.446 µs | 0.439 | fused-nt+pfs | plain+pfs+pfw 0.637 vs plain+pfs 1.107 (pfw −42%) |
| 16384 (4.4·L3) | 0.626 µs | 0.620 | seq3-nt+pfs | seq3-nt spread 0.481 vs chunk-t0 0.519 (−7%) |

Correctness: PASS at B = 1, 8, 64, 2048, 5632, 16384, rel_l2 = 2.286e-16 … 2.334e-16
(tolerance 1e-12), rel_max ≤ 3.0e-16; output bit-identical across re-runs at every
size including the NT-spread and forced-pfw (`-DL8_VARIANT=2,7`) paths; AVX2 local
build PASS; builds warning-free. Setup: 0.22 s at B=1 (probe only), 0.27 s at B=64,
0.5–0.85 s at the big batches (tuner + probe), all unscored.

Wallaby clock probe: clk256 = 3.45, clk512 = 3.36 — i.e. no licence downclock on SPR
(2.6% is thermal jitter), consistent with the corpus. The absolute value is the
*shared-machine all-core* turbo, not the 4.10 an idle core gives; on the exclusive
node the reading will be clean.

### What was tried and did NOT work

1. **Clock probe, attempt 1 (L6's literal method: 4 ms warm + best-of-5 trials).**
   Read clk256 = **2.10 GHz on a 4.1 GHz wallaby core** — exactly the Gold 6448Y base
   clock. A powersave-governed core does not ramp inside 4 ms.
2. **Attempt 2 (consecutive-stability test, 5 ms chunks, 1% agreement).** Also read
   2.10: the core sits *stably at base* for ~100 ms before the governor jumps it, so
   two agreeing readings prove nothing. Attempt 3 (fixed 150 ms warm) read 3.45 once
   and 1.76 in a third process (landed on a busy core). Shipped: track the running
   max of 5 ms chunks, stop when the max stagnates 100 ms. The general lesson for
   anyone probing frequency on these machines: **warm until the reading stops
   improving, not for a fixed time, and report the max sustained chunk.**
3. **pfw at cache-resident batch sizes.** fused+pfs+pfw 0.463 vs fused+pfs 0.449 at
   wallaby B=2048 (ws = 0.53·L3): when the output is L3-resident the RFO is cheap and
   128 extra prefetch uops are pure cost. pfw is only offered where the tuner runs it
   against the alternatives, and the anchor never includes it.
4. **First B=64 tryout of the day read 0.646 µs/t** — pure slow-state wallaby (the
   in-tuner table in the same process showed all four candidates at 1.9× their
   fast-state values with ordering preserved, and the next run measured 0.337). Fifth
   round of the same lesson: never read a single wallaby number.

### Borrowed

* **L8_radix8**: the spread prefetch cadence (6/5/5 per iteration, t0, plain-store
  pairing) and the node evidence (1.136/1.418) that made adopting it obligatory; also
  the reps-style small-batch timing discipline echoes its per-candidate run functions.
* **L6_unrolled**: the clock-probe method (FMA latency chain) — extended here to
  512-bit, which no entry has measured; and (with **L6_pfa**) the write-intent
  prefetch idea behind the pfw candidates.
* **panel_r4 VERDICT**: the B=64 L2-cliff priority, the "NT lost everywhere on the
  node" reading, and the 3.89 GHz clock fact that motivates the probe.

### Node predictions (stated to be scored)

* B=1: **0.577–0.579, unchanged** (same instructions; the probe's clk512 number is
  the round's actual B=1 deliverable — it decides whether this cell is closed).
* B=64: if wallaby's seq3+pfs transfers, **0.59–0.61** and the cell is defended with
  margin; if the node keeps fused (3% hysteresis), 0.623 stands. Either way the pick
  string says which.
* B=2048: the set now contains the exact winning family (seq3+pfs ≈ radix8's 3p-pfs).
  Predict **1.13–1.20**, pick seq3+pfs or seq3+pfs+pfw; pfw beating pfs here would be
  the round's new information for the corpus (RFO hiding vs RFO avoidance on CLX).
* B=16384: same family. Predict **1.40–1.48** (radix8's 1.418 is the existence proof;
  my phase A does 128 fewer L1 loads than its pass 1, so parity or slightly better).

### Next

1. **Read clk512 off the node JSON first.** If clk512 ≈ 2.3–2.5, B=1 is at its
   1296-cycle port floor and the cell is permanently closed — record it and stop. If
   clk512 ≳ 3.3, B=1 has ~40% of unexplained headroom and the next round should build
   the mixed-width experiment (L17_rader's node-confirmed lever): e.g. do phase A in
   ymm halves (its shuffles are T1-only and cheap in ymm) while phase B stays zmm,
   trading licence clock against port-5 count.
2. **If seq3+pfs takes node B=2048/16384 at ≈ radix8's numbers**, the L=8 streaming
   story is converged (two entries, same technique, bandwidth-bound) and the only
   remaining lever is the deferred cross-volume software pipeline — which five r4
   schemes across the panel failed to get selected, so demand node evidence before
   building it.
3. **If the node picks pfw anywhere**, propagate the same candidates to the NT-less
   small-batch band (B=64) next round; if it never picks pfw, retire it with the
   number and keep the set at 8.

---

## Round panel_r7

(There is no round-6 entry: panel_r6 was halted before its timing pass to retire a stale
runner — see `results/panel_r6_abandoned_no_timing/WHY.md` — and L=8 was not in its
roster. This round starts from the panel_r5 leaderboard and VERDICT.)

### Where round 5 landed (node, panel_r5)

The best round yet: **B=64 0.594 — first**, **B=2048 0.910 — first (−31% for the entry,
the round's largest headline)**, **B=16384 1.254 — first**. Node picks: fused+pfs at
B=64, **fused+pfs+pfw 3/3 in both streaming cells** — the first selection of
write-intent prefetch at L=8, now a settled CLX rule per the VERDICT §4.5 ("hide the
RFO, don't avoid it"; NT lost for the fourth round running). B=1 **0.573 — second by
0.5%** in the standing three-way tie (radix8 0.570). Two VERDICT items drive this round:

* **clk512 = 2.89 GHz is settled** (4/5 probes; Intel's spec table agrees). So B=1's
  0.573 µs = **1656 cycles against the 1296-cycle port-0 floor — 1.28×, ~360 cycles of
  unexplained non-FP time**. Five rounds of "B=1 is a port-floor wall" were computed at
  the wrong clock; the wall was never real. My own probe read 3.27/2.43 (0.84× the
  consensus) — diagnosed below and fixed this round.
* **VERDICT §6 (L=8)**: the never-read `ld_blocks_partial.address_alias` counter, and
  L=8's 8192-byte volume stride being "maximally degenerate" for 4K aliasing.
* My r5 prediction miss is also named in §4: I mapped wallaby→node cells by B instead
  of by L3-relative working set. Predictions below use ws/L3 mapping.

### Technique: where the 360 cycles at B=1 can be, and what was built

**(1) 4K-alias model of the kernel, at 64-byte-line granularity.** All my loads/stores
are 64-aligned full vectors, so a load is falsely blocked (bits 11:6 match, ~5–30 cy)
iff its *line* residue mod 64 equals an in-flight older store's. Enumerating residues
(python brute force, reproduced in the session):

* **fused phase A: 14 blocked loads/volume, for ANY scratch placement.** The [y][x]
  scratch layout stores column x at lines {x + 8y + s}: a comb of spacing 8 that puts
  exactly 2 stores of the previous iteration into every 16-line input-load window.
  Structural — no base offset fixes a spacing-8 comb.
* **fused phase B: 12–16 blocked loads/volume, set by (scratch − out) mod 4096** — an
  allocation lottery the plan never controlled.
* **seq3 pass B2 is worse: 0 or up to ~128 blocked loads/volume**, because its
  contiguous loads [16k0+s2,+16) sit at a *constant* line offset from its contiguous
  out stores [16k0+o,+16): all-or-nothing by allocator luck. A plausible cause of
  seq3's pick flakiness in the r4/r5 tuner tables.

So ~26–30 stalls per ~1650-cycle B=1 volume in the shipping shape — a candidate for
most of the 360-cycle gap.

**(2) Anti-aliased shapes, all with byte-identical arithmetic (same dft8s order, so
all variants still produce bit-identical output):**

* **fusedAA (variant 10, +pfs = 11)**: phase A stores contiguously, layout
  [x][re/im][k1] (1 KiB per x-plane) so store lines are [16x+σ,+16) marching in step
  with the load lines [16x,+16); the scratch base is chosen **at execute time** from a
  4 KiB slack so σ = (scr−in)/64 ≡ 48 (mod 64), which keeps the store windows of
  iterations x−1..x−3 disjoint from the loads → 0 phase-A collisions. Phase B (loads
  now strided 1 KiB) runs k1 in a **permuted order** from an 8-row table indexed by
  c = (out−scr)/64 mod 8: forbidden successor q of p is (q−2p−c) mod 16 ∈ {0,1,8,9}
  (the {8,9} from the imag half-row sitting 8 lines up); rows brute-forced, rows for c
  and c+8 coincide → 8 rows. 0 phase-B collisions for every c. Out slabs are disjoint
  so any k1 order is correct; choices are cached per (in,out) pair (2 pointer compares
  per call), deterministic → repeatable.
* **seq3AA (variant 12)**: AA phase A shared; pass B1 reads the AA layout (strided)
  and keeps seq3's scratch2 slot layout so PASS_B2_BODY is reused verbatim; B1's k1
  order comes from the same table at c1 = (scr2−scr1)/64 mod 8 (same forbidden-set
  algebra); scratch2's base is pinned at (out−scr2)/64 ≡ 48, which makes B2's
  constant-offset load/store pair **alias-free in natural order with no permutation**
  — it converts seq3's lottery into a deterministic 0.
* Cost: +12 KiB × 2 arenas in the plan (execute-time base slack), one aa_setup of ~20
  ALU ops on pointer change, zero new FP/shuffle/load/store ops per volume.

**(3) The tiny regime (ws < 0.5·L2, incl. B=1) is TUNED for the first time in six
rounds**: {fused, seq3, fusedAA, seq3AA}, surrogate = exact batch (the driver's
steady state), reps ≥ ~2 ms/trial, min-of-5, **1% hysteresis** anchored at fused
(the B=1 cell is decided by 0.5% margins; min-of-5 on the exclusive node is stable
well below 1%). Rationale for seq3 at B=1 beyond aliasing: fused phase B is ~280
uops/iteration against the node's 224-entry ROB (zero cross-iteration overlap);
seq3's bodies (~90/~190) fit. L6_pfa's r5 OoO-window rejection at L=6 B=1 noted —
but L=6's bodies never exceeded the ROB; L=8's fused phase B does.

**(4) Dual-design clock probe** — the VERDICT §5 ask verbatim: SERIAL latency-4 chain
(0.25 FMA/cy, the design that reads 3.89) and SATURATING 4-chain (1 FMA/cy, the
design that reads 2.89), at 256 then 512 bits, back to back in one process, all four
numbers in `fft3d_description()` → leaderboard JSON as `clk256s/p=… clk512s/p=…`.
My r5 probe's 3.27/2.43 under-read is also fixed: the 100 ms max-stagnation stop
fired mid-governor-ramp (at B=1 no tuner runs first, so the core was cold and
powersave steps are ~100 ms apart). Now: 200 ms stagnation, 0.9 s first-probe cap,
probes after the tuner.

**(5) Anchors updated to the node's own r5 picks**: streaming and mid band →
fused+pfs+pfw (was seq3+pfs / fused-nt+pfs), L2 band → fused+pfs (was fused). NT
anchors are gone everywhere — it has lost on the node four rounds running.
Candidate sets: tiny {0,5,10,12}, L2 band {0,1,5,6,10,11,12}, streaming unchanged
{0–9} (there the store buffer is full of DRAM-bound out-stores whose in-load
aliasing is set by the driver's buffer addresses, which I cannot control).

### Operation count

Unchanged everywhere: 1296 vector FP + 896 shuffles per volume; fused 256+256
loads/stores, seq3/seq3AA +128+128 (L1-resident); AA shapes identical to their
parents. The only new execute-path cost is aa_setup's cached-pointer check (2
compares per batch pass).

### What was measured (wallaby, Gold 6448Y SPR; FAST-state session, all A/Bs
same-window per the r5 VERDICT's clock-lottery rule; driver mins + in-tuner tables)

| B | r5 best | this round | pick | note |
|---|---|---|---|---|
| 1 | 0.343 µs | **0.319–0.320 µs (−7%)** | **seq3AA** (tuned) | in-tuner: fused 0.344, fusedAA 0.344, seq3 0.319, seq3AA 0.319 — **the first B=1 movement in six rounds** |
| 64 | 0.337 | 0.337 (21.57 µs/call) | seq3+pfs | seq3AA 0.643 vs seq3+pfs 0.635 in-tuner — pfs still pays here |
| 2048 (0.53·L3) | 0.425 | 0.442 | seq3-nt+pf_t0 | wallaby still loves NT; node re-tunes on itself |
| 5632 (1.5·L3) | 0.446 | 0.437 | fused-nt+pfs | parity |
| 16384 (4.4·L3) | 0.620 | 0.607 | (rule streaming) | parity |

Correctness: PASS at B = 1, 8, 64, 2048, 5632, 16384, and forced −DL8_VARIANT=10/11/12
at B=1/8/64: rel_l2 = 2.286e-16 … 2.334e-16 (tol 1e-12), output bit-identical across
runs everywhere (variants share the arithmetic order, so tuner pick-flips across
processes stay check-safe). AVX2 (wombat) build PASS; `-Wall -Wextra` clean; modes
1/2 and the no-SIMD portable build compile.

Wallaby probe readings: clk256s/p = 3.44/3.44, clk512s/p = 3.35/3.31 — no
width-licence split and no density split on SPR, consistent with the corpus; the
interesting readings will be the node's.

### What was tried and did NOT work (or could not be tested)

1. **fusedAA shows no gain on wallaby** (0.344 vs fused 0.344 in-tuner, 0.343 driver
   both). Expected, not a refutation: Sapphire Rapids' 4K-alias penalty is far
   smaller than Cascade Lake's, and wallaby's ROB is 512. Wallaby *cannot* validate
   either mechanism this round is built on; only the node tuner can. Recorded so the
   next round does not read the wallaby table as "AA failed".
2. **Reading the alias counter myself: impossible on wallaby.** No perf binary
   anywhere on it and `perf_event_paranoid=4`. The measurement that decides *why*
   seq3AA wins (if it wins) belongs to the monitor — see the ask below.
3. **seq4 (splitting B2 into transpose-store + dft-untranspose passes) rejected on
   the count**: seq3's bodies (~90/~190 uops) already fit the node's 224-entry ROB;
   a fourth pass adds +256 L1 accesses for no window relief.
4. The r5 probe post-mortem (its 3.27/2.43): not a design flaw of the serial chain
   but a stopping-rule bug — max-stagnation 100 ms < the governor's ~100 ms step
   cadence on a cold core. The general lesson from r5 ("warm until the reading stops
   improving") needs the stagnation window to EXCEED the governor step interval.

### Borrowed

* **panel_r5 VERDICT §5**: the dual-probe experiment design (serial vs saturating,
  256 before 512, one process) — L17_winograd's saturating probe design is adopted
  as the parallel half. §6: the 4K-alias/`ld_blocks_partial.address_alias` lead that
  this round's main mechanism is built on. §4: the L3-relative wallaby→node mapping
  rule (my own named error).
* **L8_radix8 r5**: the "wallaby cannot validate a node-targeted shape, only the
  node tuner can" framing (their 1f port lost on wallaby and won the node cell).
* seq3's presence at B=1 is my own r4 structure finally offered in a regime nobody
  (including me) had tuned it in.

### For the monitor (cheap, and they decide two open questions)

1. `perf stat -e ld_blocks_partial.address_alias,cycles` on forced `-DL8_VARIANT=0`
   vs `-DL8_VARIANT=12` at B=1 — the direct test of the alias model (predicts
   ~26–30/volume vs ~2, and the cycle delta says what each stall costs on CLX).
2. The §6 fusion-vs-pfw isolation at B=2048/16384 needs only `-DL8_VARIANT=1`
   (fused+pfs, no pfw) and `-DL8_VARIANT=7` (seq3+pfs+pfw) forced runs of this file.

### Node predictions (stated to be scored; mapping by ws/L3 this time)

* **B=1: pick seq3AA, 0.50–0.55 µs.** Decomposition: wallaby transfers the shape at
  −7% (→ ~0.53); the node adds what wallaby cannot show — ROB relief (280-uop bodies
  vs 224 ROB at fused; seq3 fits) and the CLX-sized alias penalty (~26–30 stalls
  removed) — each worth 0–5% more. If neither node-only mechanism exists, wallaby's
  −7% alone still takes the cell at ~0.53. If NOTHING transfers, the 1% hysteresis
  keeps fused and 0.573 stands; the pick string will say which branch happened.
* **B=64: pick seq3+pfs or seq3AA, 0.57–0.60** (node r5 anchor fused+pfs 0.594 is
  displaced only by a >3% measured win; wallaby says the seq3 family is 5–8% ahead
  in-tuner there).
* **B=2048: 0.90–0.93, pick fused+pfs+pfw again** (candidates and code byte-identical
  to r5's winners; anchor now matches the node's own pick).
* **B=16384: 1.24–1.27, same reasoning.**
* Probe: if clk256s reads ~3.89 and clk256p ~2.89 in one process, the licence is
  density-triggered and the corpus's clk256 question closes; if all four read
  2.89/2.89, L=6's B=1 overhead shrinks by the ratio and their roadmap changes.

### Next

1. **If seq3AA takes B=1 on the node**: ask for the counter run (ask #1 above) to
   attribute the win between ROB relief and alias removal before anyone builds more
   of either; then propagate AA to the B=64 pfs twin (a seq3AA+pfs variant is one
   DEF_VOL line) if B=64's pick was AA-less.
2. **If seq3AA is picked but wins <3%**: the remaining B=1 gap is front-end or
   store-forward latency, not the memory system — the next probe is an LSD/DSB
   check (uops_issued vs uops_not_delivered on the node), monitor-side.
3. **If the streaming cells hold at 0.90/1.25**: L=8 batched is converged at two
   independent mechanisms (store order + pfw); stop touching it and spend rounds on
   B=1/B=64 only.

---

## Round panel_r8

### Where round 7 landed (node, panel_r7)

B=1 **0.571 — second**: L8_batchsimd took the cell at **0.558**, the first B=1 movement in
six rounds, with `MODE_FUSED` — **a wholesale port of MY fused structure with its own
52-instruction FMA codelet substituted for my 54-instruction dft8s, and nothing else
different** (its record and the VERDICT §2 both say so). That is the cleanest on-node A/B
the panel has ever produced for a single substitution: same shape, same shuffle networks,
same store schedule, codelets differing by 2 instructions × 24 codelets = 48 FP
instructions/volume, and it read 0.558 vs my 0.571 (−2.3%; 48 cycles at 1/cy on the 1-FMA
part predicts −2.9% — the numbers agree). B=64 **0.587 — first** (held). B=2048 **0.930 —
first** but +2.2% over my own r5 number, the round's largest cell regression, with an
unchanged pick string (`fused+pfs+pfw`) — the VERDICT reads it as part noise, part the
same "refactor around an untouched hot path" class as r4/r5. B=16384 1.234 — second by
0.2%. My r7 bet (fusedAA/seq3AA) was **declined in every cell** — the VERDICT §4.8 notes
"declined inside a 1% hysteresis band is not the same as losing" and keeps the forced
counter run as L=8's single open item (§6), monitor-side. Streaming is declared converged
("three entries within 3.6% on one technique — stop tuning it").

### What changed (two things)

**1. The 52-instruction FMA codelet, adopted from L8_batchsimd's `r8`** — taking back my
own structure's missing piece. `dft8s` is now radix-8 **DIT** (E = DFT4(evens),
O = DFT4(odds), X_k = E_k ± W^k O_k): ±i is a rename + sign fold as before, and the two
c = 1/√2 twiddles fold into the last butterfly as fmadd/fnmadd — **44 add/sub + 8 FMA =
52 instructions**, against the old DIF form's 52 add + 4 mul → 54 after contraction.
Written as `a + SQ*s` / `a - SQ*s` expressions in the generic-vector path (no intrinsics,
so the one-arithmetic-path-for-every-ISA property survives); gcc's -ffp-contract forms
all 8 FMAs — **verified by objdump on `-march=cascadelake` output: every one of the 13
variant functions emits exactly 132 add/sub + 24 FMA = 156 FP per volume loop-body set
(3 codelets), identical mix everywhere**, so cross-variant bit-identity is preserved
(confirmed on wallaby: forced variants 0 and 12 print identical rel_l2/rel_max digits).
Since every shape and every variant calls the same `dft8s`, the cut lands in all 13
variants including the AA shapes. The old codelet is kept behind **`-DL8_CODELET=54`**
(one-flag node A/B for the monitor; default 52). Why this is not L8_radix8's r2 dead end
("54→52 measured +0.7% worse"): that was a different codelet in radix8's r2 structure;
the r7 evidence is the same substitution inside *this* structure, on the node, winning
the cell. The node-validated datapoint supersedes the three-round-old one.

**2. NT variants retired from the streaming candidate set** (cand_big 10 → 6:
{fused, +pfs, +pfs+pfw, seq3, +pfs, +pfs+pfw}). The panel_r7 VERDICT elevates it to a
rule: NT stores are **0-for-everything on this node across five rounds, eight geometries
and nineteen entries' own tournaments**, including r7's purpose-built read-protected NT
at L=36. The four NT variants stay compiled and forceable (`-DL8_VARIANT=3/4/8/9`) and
the misalignment fallback still maps onto their plain twins. A 6-candidate tournament is
also less pick-flip exposure at the cells the VERDICT says to stop tuning.

Candidate sets now: tiny {0,5,10,12} (unchanged), L2 band {0,1,5,6,10,11,12}
(unchanged), streaming {0,1,2,5,6,7}. Anchors unchanged (streaming/mid fused+pfs+pfw,
L2 fused+pfs, tiny fused). Not done, deliberately: no new shapes, no AA propagation
(declined once, unmeasured — the counter run decides it, not another round of building),
no touching the streaming schedule (VERDICT: converged).

### Operation count

Per volume: **1248 vector FP** (24 codelets × 52; was 1296 × since r1) + 896 shuffles +
256/256 loads/stores (fused; seq3/AA deltas unchanged). The 52 is Yavne's 52-add minimum
with the 4 multiplies FMA-folded; on the 1-FMA node every one of the 48 removed
instructions is a port-0 cycle, so the predicted B=1 gain is ~48/1650 ≈ 2.9%.

### What was measured (wallaby, Gold 6448Y SPR; states drifted between windows this
session — MKL's own B=1 read 0.643 in one window and 0.330 in another — so the decisive
pairs are same-window alternating A/B/A/B, per the standing rule)

| measurement | c54 (r1–r7 codelet) | c52 (this round) | Δ |
|---|---|---|---|
| B=1 forced fused, alternating ×2 | 0.343 / 0.344 µs | **0.329 / 0.328 µs** | **−4.4%** |
| B=1 forced seq3AA, adjacent | 0.319 | **0.317** | −0.6% |

The c54 fused number reproduces r7's fused 0.344 exactly, so the window is calibrated
against last round's table. The fused shape takes the full codelet cut; seq3AA barely
moves — on a **two**-FMA-unit SPR core seq3AA is evidently not p0-bound, so wallaby
*cannot* show the full value of a p0 cut there; the 1-FMA node is where all shapes are
p0-bound and should all take it. Auto-tuned driver numbers, best of the session:

| B | this round (auto) | r7 best | pick |
|---|---|---|---|
| 1 | **0.317–0.318 µs** | 0.319 | seq3AA (tuned) |
| 64 | **0.328 µs/t** (20.98 µs/call; 2nd sample 0.343) | 0.337 | tuned, plain family |
| 2048 | 0.462 / 0.508 µs/t (two windows) | 0.442 | drift, memory-bound |
| 16384 | 0.921 µs/t (single sample) | 0.607 | wallaby DRAM day-noise; see below |

The B=16384 wallaby number is 50% over r7's — single sample, on the machine whose DRAM
regime has drifted 20%+ between sessions in r2/r3/r5 records; MKL's adjacent B=16384
read 1.26 µs/t in the same window (vs its own better days), so the *machine* was slow,
not the code (the codelet cut cannot cost 50%, and B=2048's two windows straddle r7's
number). Node evidence rules the streaming cells; the candidate set still contains r7's
node winners byte-comparable (`+pfs+pfw` twins).

Correctness: PASS at B = 1 (auto, forced 0, forced 12, forced 0/12 × c54), 8 (wombat
AVX2 local build), 64, 2048, 16384: rel_l2 = 2.267e-16 … 2.272e-16 (tol 1e-12; the
digits moved from the r1–r7 2.29–2.33e-16 band because the summation order changed —
expected and healthy). Output bit-identical across re-runs everywhere; forced variants
0 and 12 agree to the last printed digit at B=1, so tuner pick-flips remain check-safe.
Builds warning-free (`-Wall -Wextra`): modes 0/1/2 × codelets 52/54, cascadelake and
haswell targets.

### What was tried and did NOT work

Nothing failed outright; three readings worth recording:

1. **The codelet cut is nearly invisible in seq3AA on wallaby** (0.319 → 0.317, −0.6%,
   vs fused's −4.4%). Not a refutation of the cut — a statement that seq3AA on a 2-FMA
   SPR is bound elsewhere (its +256 L1 accesses). Do not read a wallaby shape ranking
   as a node shape ranking; on the 1-FMA node the p0 bill is 1248 for every shape.
2. **B=2048/16384 wallaby numbers are not a regression signal** — two windows straddle
   r7 at B=2048 and the B=16384 window was globally slow (MKL slow in the same window).
   Recorded so the next round does not chase it.
3. **`-funroll-loops` build-flag gap: already closed.** L45_pfa's r7 finding (scored
   build lacked it) prompted checking; the shared Makefile now carries it, so tryout
   and the scored build agree and no pragma is needed.

### Borrowed

* **L8_batchsimd**: the 52-instruction FMA codelet (`r8`, 44 add/sub + 8 FMA, DIT
  even/odd DFT4s with the √2 twiddles FMA-folded) — the round's entire arithmetic
  change, adopted after its MODE_FUSED (= my structure + this codelet) took node B=1
  at 0.558 in r7. Repaying r7 in kind: they took my shape, I take their codelet.
* **panel_r7 VERDICT §4.5/§5**: the NT-is-dead-on-this-node rule behind retiring the
  four NT candidates from the streaming set; §6's "stop tuning streaming".
* **L45_pfa r7** (via VERDICT §4): the scalar-instruction/build-flag audit habit —
  objdump-verified the FMA formation and the per-variant FP mix before timing anything.

### For the monitor (both one-flag, both cheap)

1. The r7 VERDICT §6 counter run stands and is unchanged:
   `perf stat -e ld_blocks_partial.address_alias,cycles` on `-DL8_VARIANT=0` vs
   `-DL8_VARIANT=12` at B=1 — it decides whether the AA mechanism is real and whether
   L=8's alias exposure is reachable from inside a plan at all.
2. New this round: `-DL8_CODELET=54` is the exact r1–r7 arithmetic in this round's
   file — one forced run at B=1 gives the codelet's isolated node value if the
   attribution is ever questioned.

### Node predictions (stated to be scored; ws/L3 mapping)

* **B=1: 0.552–0.562, pick fused or seq3AA** (inside the 1% hysteresis they are twins
  once both carry c52). Decomposition: batchsimd's r7 0.558 is the same arithmetic in
  the same shape, so parity is the floor of the prediction; my tuner additionally offers
  seq3AA which on wallaby is 3% ahead of fused. If the node again keeps fused and reads
  ≈0.558, the cell is a three-way tie at the new arithmetic floor and the next lever is
  the §6 counter run, not another kernel.
* **B=64: 0.575–0.590, plain-family pick** — r7's 0.587 minus the codelet's ~17
  cycles/volume (~6 ns), partially hidden under the L2-resident store traffic.
* **B=2048: 0.90–0.94, pick fused+pfs+pfw** (unchanged candidates minus NT; the r7
  regression's noise component should mean-revert; the codelet is invisible here).
* **B=16384: 1.23–1.27, pick +pfs+pfw family** (bandwidth-bound; parity with batchsimd).

### Next

1. **If B=1 lands ≈0.558 with fused picked**: the arithmetic is at Yavne's minimum, the
   shapes are tied, and the remaining ~360−48 cycles of non-FP time at B=1 are only
   attributable by the monitor's counter run (ask #1). Do not build more B=1 shapes
   until it happens.
2. **If seq3AA is picked at B=1 and wins ≥2%**: propagate AA to the B=64 twin
   (seq3AA+pfs is one DEF_VOL line) — the r7 "Next" item that was gated on exactly this.
3. **If B=2048 does not mean-revert toward 0.91 with the same pick**: the r5→r7 drift
   in that cell is real and process-level (arena layout, link order), not code — hand
   the cell to the monitor with both binaries rather than another candidate set.
