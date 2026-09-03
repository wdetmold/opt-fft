# d1_planner — strategy record

## Round d1_r1 (2026-09-02, post-restart real r1)

### Starting point
The impl_0 stub: `fft1d_supports()` returned 0 — the planner supported nothing. No
strategy records or exemplars existed from any entry (context.md empty). So this round
built the layer from scratch: **full 52-cell coverage** with a factorization-driven
plan chooser, engineered for adoption (every piece is a lift-able, documented unit).

### What was built (all in impl/d1_planner.c)
1. **Mixed-radix DIF-Stockham engine** (`mr_build`/`mr_exec`): out-of-place, autosorting
   (no bit reversal), natural-order output. Hardcoded radix-2/3/4/8 kernels in explicit
   re/im scalar arithmetic (never C99 complex mul — avoids __muldc3/NaN-branch trouble),
   generic O(r²) radix for any odd prime ≤ 61. Per-stage twiddles from exact integer
   phase `(j*s) mod (l*r)` (survey vein 2). Sign parameter gives the unnormalized
   inverse from the same kernels. j==0 blocks (unit twiddles, including the whole final
   l==1 stage) take a no-twiddle fast path in the r4/r8 kernels.
2. **Planner decision** in `fft1d_create`: L fully smooth → mixed-radix; L prime with
   smooth L−1 → **unpadded Rader** (65537 → conv 65536, 1021 → conv 1020 = 4·3·5·17,
   b-spectrum + 1/(L−1) folded at plan time); else → **Bluestein**, conv length from a
   cost model over {1,3,5,9,15}·2^k ≥ 2L−1 (10007 → 20480 = 5·2^12, NOT 32768;
   100003 → 262144). Chirp phases k² mod 2L reduced in integers (survey vein 2).
3. **Across-batch lanes** (survey lever #1, adapted): on lane-blocked data
   `[element][8 lanes]` every Stockham pass is IDENTICAL to the scalar pass with
   m → m·8, because twiddles depend on (j,s) only. So the batched path reuses the same
   kernels — two transposes per group of 8 vectors buy 8-lane vectorization of even the
   m==1 stages. Laned Rader and Bluestein variants included (Bspec/chirp broadcast over
   lanes). Gated to transform length ≤ 1023 (measured: wins to ~1K incl Rader's P=1020,
   loses at 1024+ where per-vector k-loops are already wide).
4. **Conjugate-pair dense split** for m==1 generic stages (standalone primes 13/31):
   rows s and r−s of the DFT matrix are conjugates, so the four real products are
   shared → half the multiplies; s=0 row is a plain sum. 2.5–3x measured at 13/31 B=1.

### Measured (wallaby core 100, tryout gcc flags; NOISY — see caveat)
Quiet-period minimums, µs/transform, rel L2 all ≤ 4e-15 (tol 1e-12), 26/26 cells PASS,
chained map gate PASS (L=60 m=200: 3.8e-14 vs anchor 2.7e-15):

| L | B=1 | batched | note |
|---|-----|---------|------|
| 13 | 0.12 | 0.085 (B=512) | conj-split dense / lanes |
| 31 | 0.68 | 0.44 | same |
| 32 | 0.096 | 0.064 | [4,4,2] |
| 60 | 0.38 | 0.27 | [4,3,5] |
| 64 | 0.24 | 0.118 | [4,4,4] |
| 128 | 0.53 | ~0.35 | |
| 1024 | 3.9–6.5 | ~6 | per-vector (lanes lose here) |
| 4096 | 26–36 | ~55 (B=256) | |
| 16384 | 80–160 | ~220 (B=64) | wallaby freq swings 2x |
| 1021 | ~24 | ~22 (B=256) | Rader, conv 1020 |
| 10007 | ~290 | ~880 (B=64) | Bluestein M=20480 |
| 65537 | ~1150–1350 | ~1650 (B=16) | Rader, conv 65536 [4^8] |
| 100003 | ~5700 | ~6000 (B=8) | Bluestein M=262144 [8^6] |

Vs the Ice Lake library baseline (different machine, indicative only): 1024 B=1 at
parity with MKL (4.2); 65537 in the same band as FFTW patient (1632) — the unpadded-
Rader bet looks live; 10007 near FFTW patient (229); 100003 ~1.8x behind (3261);
small-prime B=1 still ~2x behind MKL; batched small sizes competitive (32: 0.064 vs
MKL 0.112, 64: 0.118).

**Measurement caveat:** wallaby cores differ 2x in effective speed (core 90/120 slow,
core 100 fast) and drift with neighbors — only same-core INTERLEAVED A/Bs were trusted;
absolute numbers above are best-quiet-period minimums. The monitor's Ice Lake numbers
are the real ones. The reserved node was down (job 440299 dead) this whole round, so
tryout.sh was replicated by hand on wallaby (same flags/driver/checker).

### What did NOT work, with the numbers that killed it
- **Radix-8 everywhere**: first version (pointer arrays in the kernel) destroyed
  aliasing info — 10007 went 262→874, 1024 3.9→9.1. Direct restrict pointers fixed
  half; still, pure-8 plans LOSE below DRAM sizes (interleaved A/B: 65536-conv radix-4
  ~1200 vs radix-8-mix ~1370; M=32768 Bluestein via 8s 475 vs 20480 via 4s ~290) and
  WIN at 262144 ([8^6] ~5750 vs [4^9] ~6800). Landed: R8_THRESH=131072, radix-4 lead
  stage (scalar m==1 stage in the cheap kernel), and a size-dependent radix-8 weight in
  the conv cost model (0.75 above 131072, 1.35 below — the naive flat 0.75 mis-steered
  10007 to M=32768).
- **Lanes above ~1K**: 1024 B=512 lanes 5.5 vs 4.1 per-vector; 4096 B=256 lanes 27 vs
  20. Transpose cost + 8x footprint beats twiddle amortization once per-vector k-loops
  are long. Hence LANE_MAX_N=1023.

### Borrowed
- Survey (docs/literature_1d/00-SURVEY.md): across-batch split-lane vectorization
  (vein 1/batched), integer-reduced twiddle/chirp phases (vein 2), the per-prime
  Rader-vs-Bluestein playbook (65537 unpadded, 10007/100003 padded Bluestein). No other
  entry had code or records to borrow this round (all stubs).

### For next round (or for whoever lifts this)
1. **100003 is the weak headline cell** (~1.8x behind FFTW). Try one-level nested Rader
   (100002 = 2·3·7·2381, 2381−1 = 2²·5·7·17 smooth) per the survey — exactly where
   FFTW bails; or a real-arithmetic split of the Bluestein pointwise stage.
2. **Four-step/six-step at 16384+** (128×128, L1-resident sub-FFTs) — my Stockham does
   7 full-array sweeps at 16384; Bailey would halve traffic. Also the natural cure for
   the batched 16384/65537 cells.
3. **Small-prime B=1** (13/31): still ~2x behind MKL. A true min-op Winograd-style
   codelet (d1_prime's mandate) would beat my conj-split dense; adopt theirs if it lands.
4. **fft1d_chain not exported yet** — the driver fallback map is decent, but fusing the
   contraction map into the last Stockham pass (and the chirp post-mul for Bluestein)
   saves a full read-write sweep per step. Biggest for the B=512 chained cells.
5. **Cost-model honesty**: the radix weights are wallaby-measured; re-fit on Ice Lake
   PMU numbers (port pressure, l1d.replacement) once the node is back.
