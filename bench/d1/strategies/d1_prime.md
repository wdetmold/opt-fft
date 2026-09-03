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

## Round d1_r2 (2026-09-02)

### Where r1 left me (ICX leaderboard)
Won 7 of 8 cells; LOST 13 B=512 m=1 (0.0146 vs fftw1d_measure 0.0123, 1.18x
behind), and the 13 B=1 m=1 win was 1% with 15.9% spread — a coin flip. This
round attacked those two, plus opportunistic gains elsewhere. The Ice Lake
reservation was dead again (job 440371 not running), so like r1 all numbers are
wallaby (SPR 6448Y) pinned-core A/B against the same library binaries in
build/wallaby/bin; ratios, not absolutes.

### The one big finding: gcc turns u/v broadcasts into port-5 vpermpd
Disassembly of the fully-unrolled exec13_avx showed the whole port-5 budget
gone before any useful shuffle ran: gcc kept the folded u/v rows in zmm and
compiled every `_mm512_set1_pd(ur[j])` into a register `vpermpd` — 20+ of
them, all port-5-only, on top of the 12 boundary permutes, while the 24 FMAs
queued behind them (both zmm FMA pipes share p0/p5 on ICX/SPR). exec31 and the
generic kernel did NOT have the disease (rolled loops index arrays at runtime,
so the broadcasts already came from memory).

Fix: force the u/v arrays through memory with a targeted compiler barrier
after the fold stores —
    __asm__("" : "+m"(ur), "+m"(ui), "+m"(vr), "+m"(vi));
gcc then folds each broadcast into the FMA as a `{1to8}` memory operand
(load-port work, zero p5). Targeted "+m", not a full `"memory"` clobber, so
table loads can still be hoisted (measured identical, but the full clobber
pessimizes table reuse in principle). Same barrier added to chain1_body.
Numbers (wallaby GF/s, warmed core, min over runs):
  13 B512 m1 exec: 21.5 -> 27.6.  13 B1 m1: 14.6 -> 15.6/15.8.
  13 B1 chain m=200k: 6.44 -> 7.68 (+19%).
LESSON FOR EVERY ENTRY with an unrolled kernel + scalar-from-vector
broadcasts: grep your disasm for `vpermpd %zmm` — each is a stolen port-5
slot; `{1to8}` FMA operands are free by comparison.

### Second wave: pair kernel + single accumulator set (batched paths only)
- exec13_avx_b: single accumulator set for B>=8 (depth 6 is hidden by
  cross-transform overlap; saves the 4 combine adds + register pressure):
  27.6 -> 28.0.
- exec13_avx_b2: TWO transforms interleaved in one always_inline body, table
  rows loaded once per pair: 28.0 -> 30.5. Without always_inline gcc outlined
  it and re-loaded 8 permute-index constants + stack-canary per call — the
  inline attribute alone was worth ~9%.
- exec31_avx_b: single-set variant of exec31 for B>=8 (the 2-set version holds
  16 accumulators and spills — 44 reg-reg vmovapd in the emitted loop):
  31 B512 m1 23.0 -> 24.4.
- x0r/x0i broadcasts read through an asm-hidden pointer so they can't CSE
  against the z0 register load into vpermpd (neutral on wallaby, keeps p5
  clean on ICX where p5 pressure is relatively higher).

### Measured (wallaby, core 10 physical, min over 3 runs x 8 samples, core
### pre-warmed 2 s — see gotcha below), GF/s and us/transform:
| cell             | r1 wallaby | r2 wallaby | best lib (which)     | ratio |
|------------------|-----------:|-----------:|---------------------:|------:|
| 13 B=1    m=1    | 14.6       | 15.5 (0.0155us) | 16.6 mkl        | 0.94x |
| 13 B=512  m=1    | 21.5       | 30.4 (0.0079us) | 29.5 fftw_meas  | 1.03x |
| 13 B=1    m=200k | 6.4        | 7.7  (0.0313us) | 4.7 fftw_cust   | 1.64x |
| 13 B=512  m=2000 | 20.1       | 20.1 (0.0120us) | 11.6 fftw_soa   | 1.74x |
| 31 B=1    m=1    | 23.7       | 23.5 (0.0327us) | 7.1 fftw_cust   | 3.3x  |
| 31 B=512  m=1    | 23.0       | 24.4 (0.0315us) | 10.8 fftw_soa   | 2.26x |
| 31 B=1    m=100k | 15.7       | 17.0 (0.0452us) | 5.6 fftw_cust   | 3.0x  |
| 31 B=512  m=1200 | 22.3       | 22.3 (0.0345us) | 13.6 fftw_soa   | 1.64x |
(13 B1 m1 loses to MKL on SPR-wallaby but won on the ICX scoring node in r1,
where MKL ran 0.0259; the +6% here should widen that ICX win. fftw_measure at
13 B512 is plan-unstable on wallaby: 21.7 one run, 29.9 the next — its ICX
number was 18.8 GF/s, so 30.4 projects to a clear win there.)

Correctness: single-call rel_l2 1.5-4.1e-16 across L=7/11/13/17/31,
B=1/2/3/5/9/12/16/511/512 (pair-kernel remainder paths included); strict m=2
gate 3-5e-16; all graded chains pass with 2-5 decades of margin; output
bit-identical across processes.

### What did NOT work, with the number that killed it
- Pinning broadcasts to registers with "+v" so all FMAs become
  non-destructive 231 forms: 27.4 -> 25.65 GF/s. The {1to8} fold is better
  even when half the FMAs come out as 132-with-vmovapd-copy.
- The pair kernel WITHOUT always_inline: no gain over single (28.0) — call
  overhead ate exactly what table sharing saved.
- Single accumulator set at B=1: 15.50 -> 14.59 (L13), 23.53 -> 23.11 (L31).
  Kept the 2-set kernels for B<8; dispatch splits at B>=8.
- always_inline on exec31_avx_b: no change (24.4) — its calls were not the
  bottleneck at L=31; the 120-FMA core is (~60 cy floor on 2 FMA pipes).

### Measurement gotcha that cost an hour
wallaby's schedutil governor ramps 2.1 -> 4.1 GHz over ~1-2 s of load: the
first timing run of a binary lands anywhere between, and alternating A/B runs
hand each other a half-warmed core (fftw measured 0.011 then 0.022 us on
adjacent runs). Warm the core with a 2 s spin BEFORE the series and take min
over >=3 alternating runs. Core numbering: 10 and 74 are SMT siblings —
pin to a physical core and check /sys/.../thread_siblings_list.

### Borrowed
Read d1_batchlane's and d1_race's r1 records before starting. Nothing adopted
verbatim this round — batchlane's transpose-floor analysis (24 shuffles per
8x8) is what convinced me NOT to try across-batch SoA for the m=1 batched
cells and to fix the AoS kernel's port-5 waste instead; their "check the
disassembly of any hot auto-vectorized loop" lesson is the same one that found
the vpermpd disease here.

### What I would do next
- 13 B512: an interleaved-pair accumulation scheme (accumulate complex pairs
  (re,im) per lane, s-table signs pre-baked, vbroadcastf64x2 for u/v pairs)
  counts to ~43 p05-ops/transform vs the current ~46 with cheaper epilogue —
  maybe 10-15%, needs new dup-tables and a full kernel rewrite; the risk is 24
  table-row loads/transform (vs 12) going load-bound on ICX.
- 31: the dense 120-FMA core is 6% off its own lane floor, so the next real
  step is algorithmic — Rader-31 (30-point conv = twiddle-free 2x3x5
  Good-Thomas) in across-batch SoA for the BATCHED cells only, where the
  transpose cost amortizes over the batch. Margin is already 1.6-3.3x, so
  only worth it if a rival closes in.
- 13 B1 m=1: per-call overhead (driver loop + dispatch + canary) is now ~half
  the 62-cycle budget; kernel-side there is little left. If fftw_estimate
  closes the 1% ICX gap, the fix is a leaner fft1d_execute entry (single
  compare dispatch, no frame when B==1).
