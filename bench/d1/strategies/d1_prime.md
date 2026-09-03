# d1_prime — strategy record

Class: small primes, dense/Winograd floor. Graded sizes L=13, L=31 (supports also 7/11/17).

## Round d1_r1 (2026-09-02)

### Starting point
The impl_0 stub: naive dense complex O(L^2) matvec, precomputed W[k][j] table.
No earlier rounds survive (harness restart); no other strategy records existed
yet this round, so nothing was borrowed — the design came from the 1D survey
(docs/literature_1d/00-SURVEy.md: across-batch split-complex lanes, plan-stage
tables) plus the r11 3D-campaign traffic lessons.

### What the implementation is now
1. **Symmetric-pair real-coefficient dense DFT** (the classic "real-factor"
   fold): u_j = x_j + x_{L-j}, v_j = x_j - x_{L-j}, then
   X[k] = x0 + Σ cos·u_j − i·Σ sin·v_j and X[L−k] = same with +i. Every
   multiply is REAL coefficient × complex value — zero complex-mult shuffles,
   pure FMA, and (L−1)^2 real multiplies ≈ 4× below the naive dense matvec
   (L=13: 144 FMA, L=31: 900 FMA per transform). Tables built at plan time from
   libm sincos on the mod-L-reduced angle (|θ| ≤ π), zero-padded to hp lanes.
2. **k-vectorized execute** (`exec13_avx`, `exec31_avx`): accumulators are zmm
   rows over the output index k. Intrinsic prologue fuses deinterleave+fold into
   ONE vpermt2pd per re/im row (forward row x1..x6 / reverse row x12..x7 pulled
   straight out of the interleaved loads); intrinsic epilogue rebuilds
   natural-order rows and interleave-stores with index permutes (masked tail
   stores, no over-read). X[0] costs nothing: the table carries a k=0 column
   (cos=1, sin=0) in the spare padding lane, so X0 = x0 + Σu falls out of the
   same FMA loop.
3. **Split accumulators**: two accumulator sets halve the FMA dependency depth
   (6→3 at L=13, 15→8 at L=31). This was the single biggest exec win:
   L13 B1 0.038→0.020, L31 B1 0.088→0.037, L31 B512 0.071→0.037 µs.
4. **Fused chain, B=1** (`chain1_body`): the state lives ACROSS steps in the
   A/B row representation itself — A rows = state[1..h] with state[0] in lane h,
   B rows = state[L−1..L−h]. Next step's fold is then just u = A+B, v = A−B on
   whole registers: the x[L−j] index reversal is implicit in the A/B pairing, so
   the steady-state loop has NO scatter, NO gather, NO reversal permute, and no
   scalar map (X0 rides lane h through the vector map). One interleave at chain
   entry/exit only.
5. **Fused chain, batched** (`chainblk_body`): 8 chains per zmm lane-block in
   split-complex SoA; each block (state+c ≈ 8 KB at L=31) is transposed once and
   stays L1-resident for the WHOLE m-step chain — the r11 lesson applied at
   block scale rather than plane scale. k-loop blocked by 3 so each u/v row load
   feeds 12 FMAs (the plain loop was load-port bound).
6. **The map without the divider**: 1/(1+sqrt(re²+im²)) via rsqrt14 + 2 Newton
   then rcp14 + 2 Newton — FMA-port work instead of the unpipelined
   vsqrtpd/vdivpd (~34 divider cycles per zmm). m2 is clamped to ≥1e-300 before
   rsqrt so a zero-magnitude element can't turn inf·0 into NaN. Accuracy is
   honest fp64: the strict m=2 one-step gate (1.5e-14·m) passes at 3–9e-16, and
   the long chains sit 4–5 decades under their gates.

### Everything is GNU vector extensions or intrinsics, not autovectorized C
The same kernels written as plain scalar loops compiled (gcc 11, -O3
-march=native) into xmm/ymm shuffle soup with 10 (!) FMAs in the whole batched
chain kernel — measured 5–10× slower (L31 B512 chain 0.399 µs vs 0.106 after
the v8 rewrite). gcc also would not keep zmm accumulators live across rolled
loops. Lesson for every entry: at these sizes, check the disassembly; `typedef
double v8 __attribute__((vector_size(64)))` rows are enough to force clean zmm
codegen without writing everything in raw intrinsics.

### Measured (wallaby login node, Xeon Gold 6448Y, taskset 1 core, min over
### samples; NOT the scoring node — no reservation was live this round, job
### 440299 dead, so tryout.sh was unavailable and MKL was rebuilt locally for a
### same-core A/B)
| cell             | d1_prime | mkl1d_dfti same box | ratio |
|------------------|---------:|--------------------:|------:|
| 13 B=1   m=1     | 0.018 µs | 0.015 µs            | 0.85× (lose) |
| 13 B=512 m=1     | 0.012 µs | 0.019 µs            | 1.6×  |
| 13 B=1   m=200k  | 0.038 µs | 0.070 µs            | 1.8×  |
| 13 B=512 m=2000  | 0.013 µs | 0.046 µs            | 3.5×  |
| 31 B=1   m=1     | 0.034 µs | 0.167 µs            | 4.9×  |
| 31 B=512 m=1     | 0.036 µs | 0.161 µs            | 4.5×  |
| 31 B=1   m=100k  | 0.049 µs | 0.246 µs            | 5.0×  |
| 31 B=512 m=1200  | 0.037 µs | 0.242 µs            | 6.5×  |

Correctness: single-transform rel_l2 1.7–3.1e-16 everywhere; chain gates pass
with 4+ decades of margin (worst: L31 B512 m=1200 at 4.9e-12 vs tol 1e-9).
Also verified: L=7/11/17 generic paths, odd batches (2, 5, 12, 511, 
lane-clamped tail blocks), m=2 strict gate on every shape.

### What did not work, with the number that killed it
- **Plain-C scalar loops for the hot kernels**: see above; 0.399 µs vs 0.106 µs
  on L31 B512 chain, only 10 FMA instructions emitted.
- **Split accumulators in the B=1 chain at L=31 (HP8=2)**: 16 accumulator rows
  + 16 live state/c rows spill; 0.050 → 0.057 µs. Kept only for HP8==1 (L=13)
  and in the standalone exec kernels where fewer rows are live.
- **Scalar de/re-interleave around the exec kernel** (first version): the
  prologue/epilogue cost more than the 24-FMA core at L=13; the permute-based
  version took B512 m=1 from 0.047 to 0.012 µs.
- **/tmp dev-dir collision**: another implementer's build overwrote my binary
  mid-measurement (first timing run reported *their* name, d1_planner, 10×
  slower). Use mktemp dirs; never share /tmp paths on this panel.

### Honest gaps and what I would do next
- **13 B=1 m=1 still loses to MKL (0.018 vs 0.015)**: MKL's 13-point kernel is
  ~60 cycles/call — almost certainly a Rader/factorized codelet, not dense. The
  dense-symmetric floor is within 20% but the remaining latency is the 3-deep
  FMA chain + 6 permutes that a 12-point-conv codelet doesn't pay. Next round:
  hand-rolled Rader-13 (12-point Good-Thomas 3×4 convolution, precomputed
  DFT-of-generator) for the B=1 m=1 cell only — or concede the cell to
  d1_rader, whose class this properly is.
- Batched m=1 could go across-batch SoA with in-register 8×8 transposes
  (~40-60 ops/transform vs ~110 now), worth trying if MKL's batched m=1 numbers
  on the scoring node are closer than the wallaby A/B suggests.
- The chainblk map is ~35% of batched-chain ops; a fused "one-Newton +
  correction inside the next step" scheme might shave it but eats straight into
  the accuracy margin — measure the anchor first before touching it.
