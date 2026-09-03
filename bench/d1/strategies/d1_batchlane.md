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
