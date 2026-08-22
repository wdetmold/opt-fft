# L23_matrixsimd — strategy record

Entry: dense conjugate-symmetric 23x23 DFT matrix per axis, SIMD across lines.
Geometry: L = 23 (prime), 12167 points, 190.1 KiB per volume (one volume ≈ 6x
the node's 32 KiB L1d, 19% of its 1 MiB L2; a 23x23 plane is 8.3 KiB and
L1-resident).  First implemented in round panel_r6 — there are no earlier
rounds for this entry.

## Round panel_r6 (first implementation)

### Technique

Row-column 3D: one dense length-23 DFT matrix applied along each axis, with
the j <-> 23-j conjugate pair folded first (FFTW's dft-generic form):

    u_j = x_j + x_{23-j},  v_j = x_j - x_{23-j}          (j = 1..11)
    P_k = x_0 + sum_{j=1..11} cos(2pi kj/23) u_j         (k = 0..11)
    R_k =       sum_{j=1..11} sin(2pi kj/23) (-i v_j)    (k = 1..11)
    X_k = P_k + R_k,  X_{23-k} = P_k - R_k,  X_0 = P_0

All coefficients are REAL, so the driver's interleaved complex layout is
already the right SIMD layout (lanes = WC independent lines from a contiguous
run of a free index; a coefficient is a lane-invariant operand; the only
permutes are the re/im swap in (-i v), done as shuffle + XOR on port 5, and
the WCxWC tile transposes feeding axes 2 and 3).  Passes: Y and Z run plane
by plane through a padded 23x24-complex plane buffer with transposing stores;
the X pass runs lanes over the 529 contiguous (ky,kz) points with plain
stores.  A 23-long index is covered by 6 zmm chunks at offsets 0,4,8,12,16,19
(the tail recomputes 1 line, bit-identically); the 529-long X index by 132
zmm chunks + 1 tail at 525 (recomputes 3).

**The architecture is adopted wholesale from L17_matrixsimd** (attribution:
its rounds 1–5 records): pre-splatted coefficient tables, fused transposing
stores, plane-buffer padding, k-blocked kernel option, X-first pass order for
batch, cross-volume software pipelining (its exec18), plane-wise NT-staged
output (its exec20/l17_ntcopy), the licence-clock probe, and the bit-class
tuner discipline.  Everything was re-derived and re-measured at L = 23.

### The one genuinely new idea at this size: pin the WHOLE matrix (22 constants)

The 23x23 DFT matrix has only **11 distinct cosine and 11 distinct sine
magnitudes**: cos(2pi r/23) = cos(2pi (23-r)/23) with NO sign change, and
sin(2pi r/23) = -sin(2pi (23-r)/23), where the sign is a compile-time choice
between vfmadd and vfnmadd (free).  So a fully-unrolled kernel (generated,
253 statements) runs the entire matrix from 22 register-resident constants,
and the ~253 coefficient-table loads per chunk of a rolled kernel disappear
(~298 loads/chunk -> ~90).  This generalises L17_matrixsimd r3's "pinned sine
constants" (which could pin only the 8 sines of its nested kernel) to the
whole dense matrix — at L=17 the same fold gives 8+8 = 16 constants and would
likely pay there too (note for the L17 entries).  No asm-pinning is used: 22
pinned + 12 accumulators exceeds even the EVEX file, so the constants are
plain locals loaded once per chunk and gcc keeps what fits (measured: better
than explicitly parking P in L1, see below).

### Operation count

Per line (594 real flop): 253 vector FMA per chunk = (12 cos accumulators +
11 sin accumulators) x 11 rank-1 updates, + 22 butterfly + 22 combine
add/subs.  Naive complex 23x23 matvec is 4232 flop/line: 7.1x more.
Per volume: 3*529 lines * 594 = 943 kflop (driver yardstick 5 N log2 N = 824
kflop, so reported GF/s ≈ real GF/s).  As vector work: 409 zmm chunks/volume
(2 passes * 23 planes * 6 + 133), 297 FP-port ops each = 121.5k FP-port ops.
Node floor (one 512-bit FMA unit, 2.9 GHz): **41.9 us at B=1**.  Wallaby
floor (two units, ~3.7 GHz): ~16.4 us.

### What was measured (wallaby, Xeon Gold 6448Y, SPR; MKL on same data)

wallaby was intermittently contended during this round (MKL's own B=1 number
swung 143 <-> 279 us between runs); numbers below are best-of-several.

| case | this entry | us/vol | MKL best seen | ratio |
|---|---|---|---|---|
| B=1    | 24.5 us   | 24.5 | 142.7 us | 5.8x |
| B=8    | 227.8 us  | 28.5 | 1150.9 us | 5.1x |
| B=64   | 1824.6 us | 28.5 | 9234.4 us | 5.1x |
| B=256  | 8681.1 us | 33.9 | 38269.0 us | 4.4x |
| B=512  | 17705.2 us| 34.6 | 78576.4 us | 4.4x |
| B=2048 | 75938.4 us| 37.1 | 318.0 ms | 4.2x |

rel L2 error 3.8e-16 in every regime; repeatable (bit-identical across
processes) after the bit-class fix described below.  AVX2-only build verified
on wombat (Haswell): 94 us/vol at B=2, correct, 3.2x MKL there.

Streaming-regime tuner table (nv=403 volumes, >L3), the interesting rows:
512-bit pinned X-last 46.8 / X-first 43.4 / X-first+pipelined 44.2 /
**X-first+NT planes 34.1 (kept)** / X-first+pipelined+NT 35.3.  256-bit
pinned X-first 43.2 — the widths nearly tie in the DRAM-bound regime, as
expected.

### What did NOT work / negative results (with numbers)

* **Rolled table kernels lose to the pinned kernel everywhere.**  512-bit
  unblocked table kernel: 38.1 us at B=1 on wallaby vs 24.5 pinned (-36%).
  In the same tuner table (equal conditions): 52.6 vs 46.8 units X-last
  streaming, 48.8 vs 43.4 X-first streaming.  Cause: ~253 coefficient-table
  loads per chunk saturate the load ports; the FMA ports never see them.
* **k-blocking (register-pressure relief) is not needed at this size on
  AVX-512**: k-blocked 56.7 vs unblocked 52.6 tuner units at B=1 (-7% for
  blocking); k-blocked + L1-parked butterflies worse still (60.0).  The
  unblocked kernel's 23 accumulators fit the EVEX file; do not retry.
* **Explicitly parking P0..P11 in L1 across the R sweep (pc=1) does not beat
  letting gcc spill what it wants**: 59.7 vs 57.8 (B=1), 43.32 vs 43.39
  (streaming) — a wash at best.  Both stay as tuner candidates.
* **The pinned kernel is NOT bit-identical to the table kernels** (cmp on
  full outputs; gcc contracts the two differently) — exactly the failure
  L17_matrixsimd r3 documented.  First submission tripped the panel's
  repeatability check because two processes' tuners picked across the
  boundary.  Fix: only the pinned family is selectable; within it, width,
  P-parking, pipelining and NT staging are cmp-verified bit-identical
  (12 forced-variant cmp runs at B=4 and B=64 on wallaby).  X-last and
  X-first reassociate the triple sum and are separate classes, chosen
  deterministically by batch (<64: X-last, >=64: X-first).
* **Cross-volume pipelining did not add on top of NT staging on wallaby**
  (35.3 vs 34.1) and was ~flat without it (44.2 vs 43.4).  Kept as tuner
  candidates because L17_matrixsimd r4 measured the OPPOSITE interplay on
  the node (NT lost without pipelining, won 14% with it) — the node tuner
  must judge this itself on its 1 MiB L2 / 22 MiB L3.
* **Cross-volume input prefetch (pf) lost at streaming** with the NT winner:
  33.9 (pf=0) vs 36.7 (pf=1) us/vol.  A/B'd at plan time; picked per machine.
* **Mixed-width ymm tail (L17_rader's "512t", via L17_matrixsimd r5):
  evaluated on paper, not built.**  Covering 23 needs 6 zmm, or 5 zmm + 2 ymm
  = 6 equivalents on the node's single 512-bit unit (a tie), and 3.5 vs 3.0
  equivalents on wallaby's two units (a loss).  23 = 4*5+3 lacks L17's lucky
  4*4+1 shape; there is nothing to win.  Do not retry.

### What worked, in order of impact

1. **Pinned 22-constant fully-unrolled kernel**: -36% at B=1 vs the rolled
   dense kernel.
2. **NT-staged plane output** (adopted from L17_matrixsimd exec20): -21% in
   the streaming regime (43.4 -> 34.1 us/vol at nv=403).
3. **X-first pass order at batch** (adopted from L17_matrixsimd r3): -7%
   streaming before NT (46.8 -> 43.4); X-last stays better under batch 64.
4. 512-bit everywhere on wallaby at B=1 (-14% vs 256-bit); near-tie streaming.

### Where this stands / the crossover question this entry exists to answer

The brief asks where dense-matrix loses to Rader at a bigger prime.  This
round's answer so far: the dense approach lands at 24.5 us B=1 on wallaby =
1.5x its FP-port floor there, and ~5.8x faster than MKL; the node floor is
41.9 us.  L23_rader is still a stub, so the crossover cannot be measured yet.
For the comparison when it lands: dense-folded spends 594 flop/line; Rader-23
(length-22 cyclic convolution, = 2 x length-11, each needing Rader-11 or
Winograd-11 again) has a much lower flop count on paper but pays gathers for
the index permutation and needs two levels of prime machinery.  At L=17 the
same fight ended 15.2 vs 17.1 us in dense's favour.

### Next round

1. **Nested cyclic-11 split of the cosine side.**  With j,k reindexed by
   powers of 5 mod 23 (quotient by +-1), the cosine half is a length-11
   CYCLIC correlation and the sine half a length-11 negacyclic one.  Unlike
   L17's length-8 case there are no sign-only splits (x^11-1 = (x-1)*Phi11),
   but the (x-1) factor still peels the DC term, and a Winograd-style
   cyclic-11 by 2x Toom/Karatsuba-ish halves could cut the 253 FMA/chunk
   meaningfully.  Count the adds first — at 512-bit they share the FMA port,
   which is what killed every such split for L17_winograd.
2. **Node feedback**: check which variant the node tuner picked per batch
   cell (description string carries it plus measured clk512/256), especially
   the NT-vs-pipelined interplay, and whether 41.9 us B=1 floor + measured
   wallaby 1.5x-of-floor scheduling overhead predicts the node number
   (~55-60 us expected).
3. **B=1 gap**: still 1.5x above the wallaby FP floor.  Suspects: the X pass
   writes 23 strided streams (rows are 16-mod-64 aligned so half the vector
   stores split cache lines), and the Y-pass loads have 368-byte row stride.
   A line-splitting fix (pad t1's X-pass layout to 64-byte rows) is worth one
   experiment.
4. If the monitor can spare it: `perf stat -e cycles,ref-cycles` on the node
   run to confirm the 2.9 GHz licence clock assumption for this kernel mix
   (the clk512/256 probe in the description string reports it already).
