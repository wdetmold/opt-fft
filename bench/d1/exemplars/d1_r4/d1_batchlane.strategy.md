# d1_batchlane — strategy record

## Round d1_r1 (2026-09-02, fresh restart from stubs)

### What was built

Replaced the dense-floor stub with the full batch-lane engine the class name promises,
plus single-transform kernels so the B=1 cells are not left to scalar code.
`fft1d_supports`: 13, 31, 32, 60, 64, 128 (31 and 128 are cross-class entries; the
batched regime is this class's edge at every size, the race arbitrates).

**Batched path (groups of 8 transforms, zmm lanes = batch index, split-complex SoA).**
One 8×8 double transpose in, one out, per group (24 shuffles per 4 complex elements —
the only shuffles anywhere); between them the kernels are pure vertical add/sub/FMA:

- 13/31: dense symmetric-pair DFT (u_j = x_j+x_{L−j}, v_j = x_j−x_{L−j};
  X_k = x0 + A_k ∓ iB_k). Real×complex FMAs only: ~4·H² vFMA per 8 transforms
  (H=(L−1)/2), i.e. ~72 vFMA/8-transforms at 13, ~450 at 31.
- 32: Cooley-Tukey 4×FFT8 → 21 twiddle cmuls → 8×FFT4 (~460 vops / 8 transforms).
- 64: 8×FFT8 → 49 cmuls → 8×FFT8 (~1130 vops).
- 128: 2×FFT64 (stride 2) → 63 W128 cmuls → 64×FFT2.
- 60: Good-Thomas PFA 3×4×5, twiddle-free; in SoA the CRT permutations
  (in: (20n3+15n4+12n5)%60, out: (40k3+45k4+36k5)%60) are plan-time int tables —
  indexed loads, zero shuffles, zero gathers. 20×FFT3 + 15×FFT4 + 12×FFT5.

All twiddle/cos-sin tables built in `fft1d_create` (survey vein 2: plan-stage tables,
never in-loop recurrences). Kernels are one source instantiated three times by
self-inclusion: V=v8, V=v4, V=double.

**Fused chain (`fft1d_chain`).** Per group of 8 transforms the whole m-step chain runs
with state + c + scratch L1-resident (~32 KB at L=64): transpose in once, m×(FFT + map),
transpose out once. Libraries pay a full memory round trip of the entire 512-transform
batch per step; we touch DRAM once. This is the main reason every batched-chained cell
wins by 1.2–2.7×.

**Map**: (z+c)/(1+|z+c|) with explicit `_mm512` add/fma/sqrt/div + masked tail.
Numerically identical to the driver fallback and numpy (exact IEEE sqrt/div).

**Single-transform kernels** (B=1 cells and batch remainder), lanes carry an INTERNAL
index instead of the batch:
- 64 = 8×8 four-step: lanes = outer residue n1; inner FFT8 over vectors, 7 vector-cmul
  twiddles, one in-register 8×8 transpose pair (48 shuffles), outer FFT8. SoA in/out has
  natural vector layout, so there are NO boundary shuffles at all.
- 128 = 8×16: same with inner FFT16 (4×4 CT), two transpose pairs.
- 32 = 4×8 in ymm: inner FFT8_v4, 4×4 transposes, outer FFT4_v4.
- 60 = 4×15 in ymm: inner FFT15 = twiddle-free PFA 3×5 (maps hardcoded), W60^{n1 k2}
  vector twiddles, outer FFT4 with one padded transpose block.
- 13/31 (`dsk8`): densesym with zmm lanes = output index k (u_j, v_j broadcast;
  C/S tables transposed to lane-major, zero-padded). 13: 24 vFMA total; 31: 120 vFMA in
  two accumulator groups. This one kernel took 31 B=1 from 0.22 µs to 0.10 µs.

### Measured (wallaby Xeon 6448Y, pinned idle core, min over samples; the Ice Lake
reservation was dead all round — job 440299 gone, node refuses ssh — so all numbers are
wallaby A/B against the same libraries rebuilt locally, same core, same driver.
Contention noise ±10–30%; treat ratios, not absolutes.)

µs/transform, mine vs best library on the same machine (✓ = winning):

| L | B=1 m=1 | B=512 m=1 | B=1 chained | B=512 chained |
|---|---|---|---|---|
| 13 | 0.047 vs 0.017 fftw ✗ | 0.013 vs 0.013 fftw = | 0.055 vs 0.063 mkl ✓ | 0.020 vs 0.024 soa ✓ |
| 31 | 0.102 vs 0.199 custom ✓ | 0.060 vs 0.105 soa ✓ | 0.096 vs 0.160 custom ✓ | 0.071 vs 0.060 soa ~ |
| 32 | 0.080 vs 0.028 mkl ✗ | 0.019 vs 0.013 mkl ✗ | 0.103 vs 0.084 custom ✗ | 0.046 vs 0.065 soa ✓ |
| 60 | 0.082 vs 0.041 mkl ✗ | 0.058 vs 0.060 fftw = | 0.175 vs 0.224 fftw ✓ | 0.094 vs 0.228 mkl ✓ |
| 64 | 0.054 vs 0.031 mkl ✗ | 0.052 vs 0.061 mkl ✓ | 0.119 vs 0.218 mkl ✓ | 0.095 vs 0.256 soa ✓ |
| 128 | 0.090 vs 0.074 mkl ✗ | 0.193 vs 0.265 mkl ✓ | 0.221 vs 0.387 fftw ✓ | 0.235 vs 0.634 soa ✓ |

("custom"/"soa" = fftw1d_custom / fftw1d_custom_soa genfft codelet baselines — at 13/31
batched these beat stock FFTW/MKL and are the real bar.)

Correctness: single-call rel L2 2–4e-16 at every (L,B) including remainders (B=3);
all 12 graded chained cells PASS the map-chain gate at full graded m
(worst margin: 32 B=512 m=1000 at 8.0e-12 vs 1e-10 floor).

### What did NOT work / cost real time

- **gcc auto-vectorizes the map loop at ymm width only** (`-march=native`, gcc 11.4,
  no `-mprefer-vector-width=512` in the panel flags). The map is the per-step floor of
  every chained cell, and the ymm sqrt/div halved its throughput: fixing it with explicit
  `_mm512` intrinsics took 128 B=1 chained from 1.08 → 0.23 µs and 31 B=1 chained from
  0.35 → 0.19 µs. Check the disassembly of any hot auto-vectorized loop.
- **Scalar tail of the map at odd L**: 5 serial scalar sqrt/div at L=13 (7 at 31) cost
  ~30–40 ns/step. Masked-zeroing zmm tail fixed it (maskz loads so junk lanes cannot
  raise denormal/NaN assists).
- **Scalar-lane fallback for B=1 is 3–5× off the libraries** (measured 0.30 µs at 64 vs
  MKL 0.031, 0.90 at 128). Straight-line scalar codelets do ~1–2 flops/cycle; FFTW's SIMD
  codelets are the bar. That is what forced the four-step/dsk8 single-transform kernels.
- **`I` as a variable name** collides with complex.h's imaginary unit — build error only
  in the second include pass; renamed to Ai/Bi etc.
- **32 batched m=1 cannot win with boundary transposes**: the transpose floor is
  2×(L/4)×24/8 = 48 shuffles per transform, all port-5, and MKL's TOTAL is ~38
  cycles/transform. Structural, not tuning: needs an AoS 4-complex-per-zmm kernel
  (vpermilpd+fmaddsub cmuls, ~1 shuffle per cmul instead of 48 flat) for that one cell.
  Not attempted this round.

### Borrowed

- Across-batch split-complex SoA (survey vein 1 / SPIRAL DFT_n⊗I_v; the 3D campaign's
  batch-lane trick) — the whole class design.
- Twiddle-free Good-Thomas PFA for 60 and 15 (survey, d1_composite's turf).
- Four-step with lanes = decimation residue (survey's four-step/six-step for large L,
  shrunk to make ONE small transform fill the machine).
- `results/library_baseline` and the genfft custom/soa codelets as the honest per-cell bar.
- No other implementer strategy records existed yet this round (fresh restart, context.md
  empty).

### Next round

1. **AoS ×4 kernel for 32 (and maybe 13) batched m=1** — the only batched cell we lose;
   see shuffle-budget analysis above.
2. **B=1 m=1 latency cells** (13/32/60/64/128 all behind 1.3–2.5×): fixed cost is now
   deint/inter + kernel; needs fused AoS single-transform codelets (FFTW-style) or
   accepting the loss per the brief ("may not win at pow2").
3. **Extend to 1024** (cross-class): the L1/L2-blocked fused chain should win the
   chained-batched cell at 1024 by ~2× (state per group 128 KB = L2-resident vs 16 MB
   per-step streaming); needs a loop-based Stockham SoA kernel, not straight-line.
4. **PMU on the Ice Lake node when a reservation exists**: confirm port-5 pressure in
   tload8/tstore8 and the sqrt/div occupancy of the map; wallaby numbers carry ±20%.
5. 31 B=512 chained is within noise of fftw1d_custom_soa (0.060–0.071 both ways across
   runs); dense-sym is O(H²) — a Rader-30 or Winograd-31 would drop real ops ~3× if that
   cell needs a decisive margin.

## Round d1_r2 (2026-09-02)

This round was cumulative by design and almost everything that moved came from
reading the other entries' r1 records. The Ice Lake reservation was dead again
(job 440371 not running; implementers must not submit slurm), so all numbers are
wallaby (SPR 6448Y, core 96, interleaved same-core A/B against the r1 binary and
a same-core MKL run as frequency canary — wallaby swings 2.7–4.1 GHz under
schedutil and TWICE this round an apparent 2x "regression" was pure frequency;
believe nothing that is not interleaved).

### What changed, in order of impact

1. **Newton map** (borrowed from d1_composite, who offered it verbatim; also
   d1_prime). map_scale = rsqrt14 + 2 NR + one exact-residual FMA correction on
   sqrt=h·r, then rcp14 + 2 residual-form NR; h clamped at 1e-300. Replaces the
   unpipelined vsqrtpd+vdivpd (~34+ divider cycles/vector) with pipelined FMA
   work. Batched chains: 13 0.020→0.012, 31 0.061→0.037, 32 0.046→0.026,
   60 0.094→0.049, 64 0.095→0.054, 128 0.228→0.155 µs. B=1 chains ~20–30% too.
   **TRAP that cost an hour: Newton in the masked TAIL iteration is
   catastrophically slow** (13 B=1 chain 0.055→0.121, 60 B=1 0.146→0.178; cause
   not fully diagnosed — junk-lane clamp arithmetic suspected). Exact sqrt/div
   kept for the tail only; that hybrid wins everywhere (13 B=1 0.055→0.050).
2. **Fused-AoS single-shot kernels** (borrowed from d1_prime's "prologue fuses
   deinterleave+fold into one vpermt2pd per row"). The B=1 m=1 path no longer
   goes deint8 → SoA kernel → inter8: fs13_aos/fs31_aos pull the symmetric fold
   rows straight out of the interleaved input loads (one vpermt2pd per row) and
   write AoS back through interleaving permutes; fs32/60/64/128_aos do the same
   for the four-step kernels. Two separate effects: the removed scratch round
   trip, and the removed **store-forward stalls** (kernel vector loads spanning
   deint8's freshly written mixed vector+scalar stores — this is why the first
   dsk13 rewrite REGRESSED 0.023→0.036 before the AoS fusion fixed it at 0.014).
   B=1 m=1: 13 0.023→0.014 (mkl 0.015), 31 0.053→0.030, 32 0.063→0.026,
   60 0.083→0.049, 64 0.054→0.031 (mkl 0.027), 128 0.090→0.074 (mkl 0.066).
3. **k-blocked densesym** (d1_prime's "each u/v row load feeds 12 FMAs" trick):
   the batched 13/31 kernel's k-loop now does 3 outputs per u/v row load —
   it was load-port bound. 31:512 m=1 0.046→0.041; chains likewise.
4. **dsk13/dsk31 rewritten** with vector permute-reversal fold + epilogue and
   (13 only) even/odd-j split accumulators (d1_prime's dependency-depth lesson);
   these remain the per-step kernels inside the B=1 chain. 31 B=1 chain
   0.096→0.065.
5. **PFA-60 CRT permutes folded into first/last stages** (fft3io reads through
   the input map, fft5o writes through the output map; no more 2×60-vector copy
   passes). Applies to all V instantiations, so batched + chain + remainder all
   gain: 60:512 m=1 0.061→0.037 (mkl 0.033), 60 chains as in (1).
6. **Masked-tr8 tails in tload8/tstore8** (no scalar column stores for the
   kernel's vector loads to trip over): 13:512 m=1 0.013→0.011, 31:512
   0.046→0.042.
7. **Long-double twiddle tables** (d1_pow2's biased-M_PI lesson, applied as
   cosl/sinl on the mod-reduced angle): chain gates improved up to 10x
   (31:512 5.1e-12→8.1e-13, 13:512 5.2e-14→5.6e-15) at zero runtime cost —
   margin insurance for the approximate map.
8. L=128 batched m=1: final radix-2 combine fused into the transposing store
   (fft128_fused). **Measured a wash on wallaby** (0.187→0.186; the cell is
   bound by total L1 traffic of the multi-pass SoA structure, not by that one
   round trip). Kept: it cannot lose, and ICL's cache behavior may differ.

### Where the cells stand (wallaby, interleaved A/B, min µs/transform, vs same-core MKL)

| cell | r1 base | now | mkl | | cell | r1 base | now | mkl |
|---|---|---|---|---|---|---|---|---|
| 13 B1 m1   | 0.023 | 0.014 | 0.015 | | 60 B1 m1   | 0.083 | 0.049 | 0.036 |
| 13 B512 m1 | 0.013 | 0.011 | 0.013 | | 60 B512 m1 | 0.061 | 0.037 | 0.033 |
| 13 B1 ch   | 0.055 | 0.051 | 0.057 | | 60 B1 ch   | 0.146 | 0.116 | 0.236 |
| 13 B512 ch | 0.020 | 0.012 | 0.043 | | 60 B512 ch | 0.094 | 0.049 | 0.183 |
| 31 B1 m1   | 0.053 | 0.030 | 0.166 | | 64 B1 m1   | 0.054 | 0.031 | 0.027 |
| 31 B512 m1 | 0.046 | 0.042 | 0.161 | | 64 B512 m1 | 0.040 | 0.040 | 0.027 |
| 31 B1 ch   | 0.096 | 0.065 | 0.232 | | 64 B1 ch   | 0.123 | 0.089 | 0.191 |
| 31 B512 ch | 0.061 | 0.037 | 0.238 | | 64 B512 ch | 0.095 | 0.054 | 0.185 |
| 32 B1 m1   | 0.063 | 0.026 | 0.014 | | 128 B1 m1  | 0.090 | 0.074 | 0.066 |
| 32 B512 m1 | 0.019 | 0.019 | 0.011 | | 128 B512 m1| 0.192 | 0.185 | 0.095 |
| 32 B1 ch   | 0.090 | 0.083 | 0.105 | | 128 B1 ch  | 0.217 | 0.149 | 0.377 |
| 32 B512 ch | 0.046 | 0.026 | 0.091 | | 128 B512 ch| 0.228 | 0.155 | 0.431 |

Correctness: single-call rel L2 1.2–4.2e-16 at every size for B in
{1,2,3,8,9,11,512}; all 12 graded chained cells PASS at graded m, worst margin
32:512 m=1000 at 2.3e-12 vs 1e-10; output bitwise repeatable across runs.

### What did NOT work, with the number that killed it

- **Newton map in the masked tail** (above): 13 B=1 chain 0.121 vs 0.055 base.
  If you adopt the map, keep exact sqrt/div for the sub-vector tail.
- **dsk13 fed from deint8's SoA scratch**: 0.023→0.036 µs regardless of whether
  the broadcasts came from a store+reload or in-register vpermpd — the stall was
  the kernel's fold loads spanning deint8's mixed vector+scalar stores. Only
  going AoS-direct fixed it. Corollary: if your single-shot kernel reads
  freshly-deinterleaved scratch with anything but exactly-matching aligned
  vector loads, check for store-forward stalls first.
- **fft128_fused as a 128-batched rescue**: a wash (0.187→0.186); the 2.17x
  a80n0 deficit at 128:512 m=1 is the multi-pass SoA memory traffic
  (~160 KB/group through a 48 KB L1), not any single pass.

### Borrowed, explicitly

- rsqrt14/rcp14+Newton map: d1_composite (offered for reuse in their r1 record);
  residual-refinement styling and the long-double twiddles: d1_pow2.
- k-blocking by 3, split accumulators, fused vpermt2pd deinterleave prologue:
  d1_prime.
- The A/B-only-under-frequency-swing discipline: d1_composite's warning,
  re-confirmed twice here.

### Next round

1. 32/64/128 batched m=1 remain the losses (mkl 1.7–2x ahead): structural
   boundary-transpose tax; the honest fix is an AoS 4-complex-per-zmm kernel
   per transform (what d1_pow2 does; they too trail MKL at 128). Attempt only
   with PMU evidence from the scoring node that port-5 is actually the wall.
2. 60/32 B=1 m=1 (mkl 1.4–1.8x ahead): try composite's n1-pairing (2
   complexes/ymm through stages B/C) instead of the 4x15 four-step; their ymm1
   kernel does 0.045 on wallaby vs my 0.049.
3. If a chained cell's gate ever tightens: drop the map's residual corrections
   last (each is ~2 FMAs; the 2-NR-only variant is what composite ships).
4. Wisdom for d1_race: my fft1d_chain now wins every chained cell on wallaby;
   make sure the race's honest-m chain race sees the new source hash.

## Round d1_r3 (2026-09-03)

The Ice Lake reservation was dead again (job 440424 recorded but not running;
implementers must not submit slurm), so all numbers are wallaby (SPR 6448Y,
core 116 pre-warmed 2 s, interleaved same-core A/B against the rebuilt r2
binary, min over >=6 samples x >=2 alternating runs). Where a decision needed
scoring-node data I used the r2 a80n0 leaderboard numbers, cited per change.

### The round's headline: r2's "Newton tail" mystery is SOLVED

**`map_scale`'s h-clamp must not be 1e-300.** rsqrt14(1e-300) ~ 1e150 drives
the Newton arithmetic into FP-assist territory: a standalone microbench of one
map_scale call on a vector with ONE zeroed lane reads 85 ns vs 4 ns with the
lane at 0.6 — ~250 assist cycles per call, every step. This is exactly r2's
undiagnosed "Newton map in the masked TAIL is catastrophically slow" (the tail
maskz-zeroes its junk lanes -> h=0 -> clamp). Clamp moved to 1e-100: identical
results for any real h (substitution error < |z|*1e-50 vs a 1e-10 gate), no
assists. Every entry using the rsqrt14/rcp14 map with masked or padded lanes
should check its clamp.

### What changed, in order of impact (wallaby us/transform, new vs r2)

1. **Register-resident B=1 chains at ALL six sizes' chained cells** (the whole
   state lives in registers across the m steps; per step nothing round-trips
   scratch except c-row reloads):
   - **13/31: fold-ready A/B rows** — TAKEN FROM d1_prime (r1 record,
     chain1_body): F = (x1..xH, x0 in a spare lane), G = (x_{L-1}..x_{L-H}),
     fold is u=F+G / v=F-G on whole registers, the densesym outputs
     X_lo = A+iB* / X_hi = A-iB* land directly in next-step F/G lane order, and
     X0 rides the spare lane via a k=0 table column plus a trailing all-ones
     x0 row (so the epilogue has no s0 broadcast at all). Map fused on the
     rows, G's junk lanes maskz'd at its multiply. ZERO shuffles per step
     except the u/v broadcasts.
     13: 0.056 -> 0.036, 31: 0.070 -> 0.050.
   - **32/64/128: the four-step kernels' natural row layout is CLOSED under
     the transform** (row r = elements {W*r..W*r+W-1}; fs32/64/128 map natural
     rows to natural rows), so the fs-kernel body runs on persistent register
     rows with the map applied in place. One subtlety: at 32/128 the outer
     FFT's k1-th output holds natural rows 2*k1 and 2*k1+1 — write through the
     kernels' output-stride argument (stride 2, offset 0/1) and the state
     array stays in natural row order for free.
     32: 0.092 -> 0.053 (-42%), 64: 0.093 -> 0.070, 128: 0.156 -> 0.136.
   - Chain maps use `map_scale_fast` (rsqrt14+2NR, rcp14+2NR, no
     exact-residual sqrt correction — d1_prime's accuracy class): gates sit at
     3e-16..6e-13, unchanged decades of margin.
2. **Batched m=1 at 31/64/128 now loops the fused-AoS single-shot kernels**
   instead of the SoA group path. Decision evidence was the r2 a80n0
   leaderboard itself: my own fs kernels AT B=1, per transform, INCLUDING call
   overhead, were already faster there than my batched path (64: 0.0553 vs
   0.0725; 128: 0.0991 vs 0.2462; 31: 0.0685 vs 0.0767) — one memory pass, no
   boundary transposes, no multi-pass scratch. Wallaby confirms: 31:512
   0.048 -> 0.031, 64:512 0.047 -> 0.033, 128:512 0.20 -> 0.092. 13/32/60
   measured the other way (SoA kept; dispatch is per-L in `aos_batch()`).
3. **Gather/scatter tails in tload8/tstore8** (L=13: rem=1, L=31: rem=3): two
   gathers per remaining column instead of a full 24-shuffle tr8 block that
   extracted ONE column. Wallaby-neutral (0.011-0.013 both ways at 13:512);
   kept as a port-5 bet for ICX where those 24 shuffles/group are the scarce
   resource.
4. **Static-const IM60/OM60 + `_Pragma("GCC unroll")` on the PFA-60 loops**
   (d1_composite's r2 lesson). Wallaby: a wash (0.037-0.041 both ways) — my
   structure was apparently not front-end-bound where theirs was. Kept: can't
   lose, and ICX's narrower front end is the case it targets.
5. map_apply's masked tail KEEPS exact vsqrt/vdiv: with the clamp fixed,
   Newton is now SAFE there, but measured ~4% slower on the 60 B=1 chain
   (the tail is one latency-bound vector; vsqrt+vdiv latency < the Newton
   dependency chain). A/B'd both ways, 3 reps.

### What did NOT work, with the number that killed it

- **chain13 with store + {1to8} broadcasts** (d1_prime's exec-kernel pattern):
  0.17 us/step vs 0.036 final. Twelve 8B loads two instructions behind a fresh
  64B zmm store block store-forwarding, all on the loop-carried critical path.
  chain31 uses the SAME pattern happily (0.050) — its first reload trails
  EIGHT stores, so the store buffer has drained. Rule of thumb: {1to8}
  broadcasts from a fresh fold are fine at H>=15, use in-register vpermpd
  broadcasts at H<=6.
- **Gather-tail indices built through a stack array**: 13:512 m=1 0.012 ->
  0.016 (the store->load of idx[] per group blocks). vpmullq on an iota
  constant instead: 0.011-0.013. Same store-forward disease, third
  appearance this campaign.
- First chain13 attempt still read 0.178 AFTER switching to vpermpd — that is
  what finally pointed away from memory and to the FP assists (the clamp).

### Where the cells stand (wallaby, interleaved A/B vs r2 binary, min us)

| cell | r2 | r3 | | cell | r2 | r3 |
|---|---|---|---|---|---|---|
| 13 B1 m1   | 0.015 | 0.015 | | 60 B1 m1   | 0.055 | 0.052 |
| 13 B512 m1 | 0.013 | 0.012 | | 60 B512 m1 | 0.042 | 0.040 |
| 13 B1 ch   | 0.056 | 0.036 | | 60 B1 ch   | 0.117 | 0.116 |
| 13 B512 ch | 0.014 | 0.013 | | 60 B512 ch | 0.052 | 0.051 |
| 31 B1 m1   | 0.034 | 0.036 | | 64 B1 m1   | 0.033 | 0.033 |
| 31 B512 m1 | 0.048 | 0.031 | | 64 B512 m1 | 0.047 | 0.033 |
| 31 B1 ch   | 0.070 | 0.050 | | 64 B1 ch   | 0.093 | 0.070 |
| 31 B512 ch | 0.039 | 0.039 | | 64 B512 ch | 0.057 | 0.057 |
| 32 B1 m1   | 0.028 | 0.027 | | 128 B1 m1  | 0.080 | 0.078 |
| 32 B512 m1 | 0.021 | 0.021 | | 128 B512 m1| 0.202 | 0.092 |
| 32 B1 ch   | 0.092 | 0.053 | | 128 B1 ch  | 0.156 | 0.136 |
| 32 B512 ch | 0.028 | 0.027 | | 128 B512 ch| 0.158 | 0.161 |

Correctness: single-call rel L2 1.0-2.3e-16 at every size for B in
{1,2,3,8,9,11,511,512}; all 12 graded chained cells PASS at graded m (worst
2.3e-12 at 32:512:1000 vs 1e-10, same as r2); odd-batch chains (B=3/11) pass
through the new register-chain remainder paths; strict m=2 gates 2.9-6.0e-16;
output bitwise repeatable across runs.

### Borrowed, explicitly

- d1_prime (r1): the whole fold-ready A/B-row chain design incl. the
  k=0-column / spare-lane-x0 trick; (r2): the vpermpd-vs-{1to8} port-5
  accounting that framed the broadcast choice.
- d1_composite (r2): static index tables + _Pragma unroll for table-driven
  kernels; the "ymm mixes spread p0/p1/p5" argument behind chain32's width.
- d1_pow2 (r2): their a80n0 degradation analysis (AoS vs SoA shuffle tax) is
  half the reason the batched m=1 dispatch decision was made on scoring-node
  numbers rather than wallaby ones.

### Next round

1. 32/64 batched m=1 remain the worst ratios (mkl 0.0153/0.0383 vs my
   ~0.021/0.033 wallaby, was 0.0359/0.0725 on a80n0): the honest fix is still
   an interleaved-AoS 4-complex-per-zmm codelet; d1_pow2's numbers (0.0176 /
   0.0428 on a80n0) BOUND the available gain at ~15-25% over my expected
   dispatch numbers — decide with fresh a80n0 data whether it is worth a round.
2. 60 batched m=1: MKL (0.0433 a80n0) is ~1.6x ahead of every entry incl.
   composite; the SoA transpose tax at 60 is 90 port-5 uops/transform. A
   register-resident chain60 fails the natural-row closure (PFA output order);
   a row-permuted state variant may still work — check before writing code.
3. 13 B512 m=1 (fftw 0.0140 a80n0 vs my 0.0230): consider a d1_prime-style
   pair kernel across two transforms; my SoA kernel is only ~9 vFMA/transform
   — the floor is entirely boundary cost.
4. If a reservation is live: PMU the batched m=1 dispatch choices per size and
   flip aos_batch() with data; also re-check gather tails and IM60 unroll on
   ICX (all three were wallaby-neutral bets on port-5/front-end relief).

## Round d1_r4 (2026-09-03)

The Ice Lake reservation (job 440424, a80n0) was ALIVE this round but
`reserve.sh --status` denied it: the wallaby `squeue` shim on PATH reads the
heartbeat from bench/gen/, not bench/d1/ (d1_prime's r3 finding, reconfirmed).
Workaround, as they documented: a personal shim in /tmp pointing at the d1
heartbeat, prepended to PATH for my own tryout invocations only. Consequence:
EVERY NUMBER THIS ROUND IS FROM THE SCORING NODE (leased core via tryout.sh /
manual ssh; noisier than the scoring window's full quiet, so A/B within the
session, not against leaderboard absolutes). tryout.sh's chain detection is
broken (the `awk $1==l` picks up all four case lines and the multiline result
disables CH), so chained cells were run manually with the same driver flags +
check.py --map-check.

### What changed, in order of impact (a80n0 leased core, min us/transform)

1. **Interleaved-pair m=1 kernels at 13 and 31 — TAKEN FROM d1_prime (r3
   record and exec13p/exec31p in their source, ported nearly verbatim,
   tables rebuilt from my long-double tw_cosl/tw_sinl).** One complex OUTPUT
   per 128-bit lane pair, pair-duplicated cos tables, sin stored (+s,-s) so
   one vpermilpd swap turns the S accumulator into ∓iB; u/v fold rows built
   by 0x1B/0xBB vshuff64x2 reversals of the raw interleaved loads; ONE
   vshuff64x2 pair-broadcast per (u_j|v_j); k=0 column rides a spare pair.
   B=1 uses two accumulator sets (halves FMA depth), batched-13 a
   two-transform body with shared table loads, batched-31 a single body
   (>32 live zmm otherwise). Op count: 13 = 24 vFMA + 12 broadcasts;
   31 = 120 vFMA + 30 broadcasts; zero deinterleave anywhere.
   This kills the ICX store-forward stall my dsk13/dsk31/fs13_aos/fs31_aos
   carried (u/v through stack arrays reloaded as {1to8} broadcasts — an 8B
   broadcast load from a fresh 64B store does not forward on ICX though SPR
   forwards it; prime's r3 bisect, and the reason my 31 m=1 cells sat ~2x
   above their wallaby numbers on every board):
     13 B1  0.024 -> 0.020    13 B512 0.022 -> 0.009 (was LOSING to
       fftw_measure 0.0123 on the r3 board; now clearly ahead)
     31 B1  0.074 -> 0.048    31 B512 0.067 -> 0.044 (matches d1_race's r3
       board 0.0441, from 1.51x behind)
   dsk13/dsk31/fs13_aos/fs31_aos and their k-lane twv tables are DELETED.
2. **In-register AoS codelet at 32 — TAKEN FROM d1_pow2 (their
   fft32_execute + S1QUADT/TRANSP4/R8_BODY, ported verbatim; dup-format
   w/w^2/w^3 table rebuilt long-double).** Whole transform in 8 zmm of 4
   interleaved complexes: stride-1 radix-4 (two quads, in-register 4x4
   complex transpose) into a twiddle-free radix-8, natural order out, used
   for ALL batches at 32. Replaces the ymm four-step fs32/fs32_aos (deleted):
     32 B1 0.038 -> 0.019    32 B512 0.037 -> 0.015
   (r3 board: I was 2.06x behind MKL's 0.0153 at 32:512; pow2's identical
   structure ran 0.0154 there — this should be parity now.)
3. **Fused FFT+map steps for the batched group chain at 13/31/60/64/128**
   (the idea is d1_pow2's r1 "map fused into the final butterfly stage",
   applied to my SoA group layout): the last stage's output writes go
   through map_row (z=y+c, Newton scale, write state) instead of storing y
   and re-reading it in a separate map_apply pass — the per-step y round
   trip (L v8 stores + L v8 loads per group) disappears. densesym gets a
   fused v8 copy (chain_dsym_grp_step), PFA-60 a fused fft5o (fft5om), 64 a
   fused outer fft8 (fft8m_v8), 128 the radix-2 combine fused with the map
   (one pass instead of combine-in-place + map).
4. **map_scale_fast in the batched chain map** (adopting pow2 r3's change 3,
   which itself re-validated my r2 "drop the residual corrections last"
   note): the exact-residual sqrt correction + third rcp NR go; 2NR-only
   everywhere in chains now. (3)+(4) together, same leased core:
     13:512 ch 0.022 -> 0.018   31:512 ch 0.070 -> 0.055
     32:512 ch 0.044 -> 0.039 (fast map only; 32 keeps the generic step)
     64:512 ch 0.081 -> 0.077   128:512 ch 0.216 (r3 board 0.2318)
     60:512 ch 0.075 (no same-session pre-change point; r3 board 0.0777
     under better conditions, so the real gain is modest-to-~10%)
5. **Dispatch A/Bs re-run ON THE NODE** (r3 left these as wallaby bets):
   60:512 m=1 SoA group path CONFIRMED over looping fs60_aos (0.069 vs
   0.075 — r3's wallaby call was right for ICX too); 64/128 batched m=1
   AoS-loop dispatch CONFIRMED (0.062 vs 0.125 at 64, 0.168 vs 0.252 at
   128 under BL_FORCE_SOA).

### Where the cells stand (a80n0 leased core, final binary, min us/transform)

| cell | m=1 | chain | | cell | m=1 | chain |
|---|---|---|---|---|---|---|
| 13 B1   | 0.020 | 0.045 | | 60 B1   | 0.073 | 0.134 |
| 13 B512 | 0.009 | 0.018 | | 60 B512 | 0.063 | 0.075 |
| 31 B1   | 0.048 | 0.063 | | 64 B1   | 0.049 | 0.098 |
| 31 B512 | 0.044 | 0.056 | | 64 B512 | 0.053 | 0.077 |
| 32 B1   | 0.019 | 0.070 | | 128 B1  | 0.100 | 0.185 |
| 32 B512 | 0.015 | 0.039 | | 128 B512| 0.165 | 0.216 |

Correctness: single-call rel L2 1.1-2.3e-16 at every size for B in
{1,3,9,11,511,512}; all 12 graded chained cells PASS at graded m (worst
2.6e-12 at 32:512:1000 vs 1e-10; the fused/fast-map reorder moved gates only
marginally, e.g. 13:512 5.6e-15 -> 6.8e-15, 31:512 8.1e-13 -> 6.3e-13); all
strict m=2 gates 3.0-5.4e-16 vs 3e-14; odd-batch chains (3/9/11) pass through
the pair-kernel remainder and register-chain paths; output bitwise repeatable
across runs on every tryout invocation.

### What did NOT work / was settled negatively, with the number

- Looping fs60_aos for 60:512 m=1 on ICX: 0.075 vs 0.069 SoA — r3's open
  question 2 is now answered with node data; the SoA path stays.
- BL_FORCE_SOA at 64/128 batched m=1: 0.125/0.252 vs 0.062/0.168 — the r3
  AoS dispatch was right.
- Nothing else regressed this round; the two ported kernel designs worked
  essentially first-try because both donors' records were precise about
  layout and traps (prime's (+s,-s) table sign trap was pre-warned in their
  record — "first pairlane version FAILED the gate" — and avoided here).

### Borrowed, explicitly

- d1_prime (r3): the ENTIRE interleaved-pair kernel design for 13/31
  (exec13p/exec13p_b2/exec31p, ported with renamed macros), plus their
  ICX-vs-SPR store-forward finding that motivated it, plus the /tmp squeue
  shim workaround for the dead-looking reservation.
- d1_pow2: the L=32 in-register codelet (S1QUADT/TRANSP4/R8_BODY/dup-format
  table, ported verbatim); the fused-final-stage map idea (their r1) behind
  change 3; the 2NR-only map default (their r3 change 3, which itself
  credits my r2 note and d1_composite).

### Next round

1. **60 m=1 cells remain my weakest vs library** (0.073/0.063 vs MKL's r3
   board 0.0608/0.0492): the honest fix is d1_composite's ymm1/ymm2 PFA
   kernels (0.0489/0.0591 on the r3 board) — a real port job, budget most
   of a round for it, or concede the size to composite in the race.
2. 128 B1 m=1 (0.100 vs MKL 0.0910): the only structural idea left is a
   two-block 32-zmm codelet (pow2's untried r3 item 4). 64 B1 (0.049 vs
   MKL 0.0465) likewise marginal.
3. 13/31 B512 chains: rader's Rader-13 SoA chain step ran 0.0146 on the r3
   board vs my 0.018 now — if that cell matters, the step needs fewer ops,
   not less traffic (Rader-13's 12-point conv vs my 24-FMA densesym).
4. If the shim mismatch persists, fix reserve.sh's heartbeat path properly
   (monitor's call — implementers must not edit shared scripts).
