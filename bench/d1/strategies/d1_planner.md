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

## Round d1_r2 (2026-09-02) — the split-complex rebuild (cumulative round working as designed)

### What changed
The r1 leaderboard said it plainly: my scalar interleaved kernels were 2–10x behind at
EVERY cell while d1_bluestein/d1_rader's split-complex core was winning cells outright.
So this round I threw away my r1 engine and rebuilt on the proven core, keeping the
planner's dispatch as the added value. r1's mixed-radix engine, lane machinery and cost
model are gone; the planner logic (smooth → direct, prime with smooth L−1 → unpadded
Rader, else Bluestein) survives on much faster legs.

**Adopted from d1_bluestein (impl + record), nearly verbatim:** the split-complex
stage kernel family st2/3/4/5/8, per-stage twiddle layout `[j-1][p]`,
inverse-as-forward-on-swapped-planes with 1/M folded into the kernel spectrum,
`st4_first_chirp` / `st4_first_bhat` / pruned `st{2,4}_last_chirp` Bluestein fusions,
`choose_M` (minimal 3^a5^b2^c pad, 4|M: 10007→20480, 100003→204800 — replaces my r1
{1,3,5,9,15}·2^k cost model), the chirp-premultiplied chain state, `#pragma GCC ivdep`
on every hot loop, and the per-function
`target("arch=icelake-server,prefer-vector-width=512")` attribute.
**Adopted from d1_prime (record):** the symmetric-pair real-coefficient fold — used
twice: as a generic split-complex stage `stg` for ANY prime radix ≤ 61 (new; extends
"smooth" so 1020 = [4,3,5,17] runs unpadded as 1021's Rader conv, and primes ≤ 61 are
a single dense stage), and as `dense_sym`, the interleaved B=1 fast path for those
single-stage plans.
**Adopted from d1_pow2 (record):** long-double twiddle/chirp/table generation
(cosl/sinl after exact integer phase reduction — M_PI's rounding is a biased phase
error), and the transform-outer chain (all m steps back-to-back per transform,
cache-resident) — my exported fft1d_chain (new this round) is built on it.
**Own additions this round:** fused deinterleave entry (`st{2,4}_first_deint`) and
fused interleave exit (`st{2,3,4,5,8}_last_int`) for the direct path — the final
stage has m=1/unit twiddles, so folding the interleaved store in deletes one full
read+write pass (measured below); the Rader plan on the shared core (fused
kernel-multiply inverse entry via st4_first_bhat when the conv leads with radix 4);
split-state fused chains for all three kinds with interleaved output only at the
final step; 8-lane across-batch path retained from r1 but now gated to dense plans
only (see "did not work").

### Measured (wallaby core 100, QUIET windows, tryout gcc flags, µs/transform;
### all 52 graded cells PASS check.py, worst rel L2 1.2e-15, all chain gates ≥ 25x margin)

| L | B=1 | batched | B=1 chain | batched chain | r1 scored B=1 | speedup |
|---|---|---|---|---|---|---|
| 13 | 0.043 | 0.036 (512) | 0.062 | 0.039 | 0.090 | 2.1x |
| 31 | 0.134 | 0.138 | 0.264 | 0.143 | 0.532 | 4.0x |
| 32 | 0.025 | 0.025 | 0.077 | 0.077 | 0.151 | 6.0x |
| 60 | 0.081 | 0.080 | 0.172 | 0.159 | 0.405 | 5.0x |
| 64 | 0.069 | 0.078 | 0.144 | 0.145 | 0.228 | 3.3x |
| 128 | 0.123 | 0.151 | 0.274 | 0.275 | 0.540 | 4.4x |
| 1024 | 1.66 | 2.21 (512) | 2.46 | 2.47 | 5.28 | 3.2x |
| 4096 | 8.84 | 10.6 (256) | 14.0 | 14.1 | 22.5 | 2.5x |
| 16384 | 37.7 | 44.7 (64) | 57.9 | 57.8 | 116.3 | 3.1x |
| 1021 | 8.19 | 9.39 (256) | 9.76 | 9.75 | 36.7 | 4.5x |
| 10007 | 113 | 116 (64) | 122 | 123 | 375 | 3.3x |
| 65537 | 813 | 815 (16) | 965 | 965 | 1614 | 2.0x |
| 100003 | 3000 | 3028 (8) | 3146 | 3152 | 6291 | 2.1x |

Vs the r1 library baselines (cross-machine, monitor arbitrates): 10007 ~1.7x AHEAD of
FFTW patient (195), 65537 ~1.8x ahead (1465), 1021 at/ahead of MKL (8.28), 31 ~2x
ahead, 100003 near parity (patient 2690); pow2/60 B=1 still behind MKL's codelets
(that is d1_pow2/d1_composite territory — the planner now loses those by ~1.2–2x
instead of 4–8x). Setup ≤ 0.052 s even at 100003.

The fused interleave exit alone (same core, interleaved A/B): 32 B=1 0.044→0.025,
32 B=512 0.041→0.026, 60 0.099→0.081, 1024 1.87→1.66, 4096 10.5→8.8, 16384 43→37.7.

### What did NOT work, with the numbers that killed it
- **Lanes for multi-stage smooth plans**: the r1 lane trick (enter the same kernels at
  stride 8 on lane-blocked data) now LOSES everywhere the plan has ≥ 2 stages, because
  the split kernels already vectorize per-vector and the two transposes are pure cost
  (interleaved A/B at B=512: 32: 0.049 lanes vs 0.041; 64: 0.108 vs 0.080; 128: 0.248
  vs 0.180; 1024: 5.1 vs 2.4). Lanes stay ONLY for dense single-generic-stage plans,
  where the per-vector path is scalar-ish (13: 0.037 vs 0.042; 31: 0.139 vs 0.160).
- **Rader fused gather-entry / scatter-exit** (d1_rader's shape rebuilt on my core):
  a WASH at 65537 (fused 828–831 vs unfused 795–843, interleaved A/B) — the
  random-access fusion de-vectorizes the stage it joins, cancelling the saved pass.
  Code kept behind `p->rentry/p->rexit = 0 &&` for a retry on the scoring node's
  1.25 MB L2, where the saved traffic should matter more than it does on wallaby.
- Bug caught by review before it shipped: after switching the chain to per-transform
  state (length L, not L·B), one leftover `pre_r + b*L` index in the kind-0 branch
  was an out-of-bounds write. The chain smoke test in PLANNER_TEST (execute+map vs
  fft1d_chain at every self-test size, B=19) is there to catch that class.

### Where it stands / for whoever lifts this
- The planner now covers all 52 cells with ONE engine and is the only entry that
  runs 1021 unpadded ([4,3,5,17] via the generic-17 stage) AND 65537 unpadded AND
  the smooth-padded Bluestein sizes from the same kernel family. Lift `stg` if you
  need an odd-prime stage; lift `st*_last_int` if your Stockham still pays a
  separate interleave pass.
- Honest gaps: pow2 B=1 vs MKL/d1_pow2 (their in-register codelets; a generic
  stage loop cannot reach 0.012 µs at L=32); 100003 near-parity — the remaining gap
  is memory traffic at M=204800, and six-step/Bailey (d1_bluestein's next-step) is
  the known cure; the chained pow2 cells pay ~25% for the exact sqrt/div map where
  d1_pow2 uses rsqrt+Newton (their accuracy fight documented — I did not spend the
  margin this round).
- Next: (1) re-A/B rentry/rexit and LANE_MAX_N on the scoring node; (2) conv-order
  chain state for Rader (d1_rader's index-reversal trick — my 65537 chain still pays
  gather+scatter per step: 965 vs 813 execute); (3) Bailey four-step at M ≥ 10^5;
  (4) adopt d1_pow2's fused rsqrt map if the graded-seed margins allow.
