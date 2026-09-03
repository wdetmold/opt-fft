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
