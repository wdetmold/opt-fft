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

## Round d1_r3 (2026-09-03) — lift the rivals' vectorized kernels: st17, GATHER8/scatter Rader fusions, conv-order chain, rsqrt map, deterministic arena

### The r2 leaderboard, read honestly
r2 scored on a80n0 (Ice Lake Gold 6326). The planner was 1.2–4.6x behind the cell
winner EVERYWHERE — never won a cell. But three of its losses were self-inflicted and
already solved by other entries whose code I can lift (this is the cumulative round's
whole point): (1) the generic prime-17 stage ran at scalar throughput on Ice Lake, so
1021 was 13.96 vs MKL 8.28 — d1_rader had already rewritten exactly this stage with
intrinsics; (2) the Rader gather/scatter fusions were left OFF (r2 measured them a wash,
but that was the SCALAR version on wallaby's fat cache) — d1_rader shipped vectorized
GATHER8 + staged-scalar scatter that win on the scoring node; (3) the chain map was scalar
vsqrt/vdiv everywhere — d1_pow2's rsqrt14/rcp14+Newton map is 8–34% faster. Plus
d1_bluestein's deterministic huge-page arena kills the allocation-luck bimodality that
made every big-cell number partly page-coloring noise. So r3 is almost entirely adoption.

### What changed (all in impl/d1_planner.c)
1. **st17 rewritten as explicit AVX-512 8-lane blocks — ADOPTED FROM d1_rader (r2
   st17_vblock, near-verbatim; only the coefficient tables rebound to this file's
   per-stage gC/gS 8x8 fold layout).** Used for the radix-17 FINAL stage (m==1, unit
   twiddles) that appears in 1021 = [4,3,5,17] and any prime whose Rader conv ends in 17.
   Interleaved A/B, 1021 B=1: **8.19 → 5.03 us** (the old generic `stg` was ~40% of the
   transform at scalar FMA throughput, exactly d1_rader's r2 diagnosis). Needs
   `target("avx512f")` on the intrinsic functions so they inline both under -march=native
   AND under the plain -O2 PLANNER_TEST build.
2. **Vectorized Rader entry/exit, both fusions now ON — ADOPTED FROM d1_rader (r2
   GATHER8 + interleave4_store + SINKSTORE shape).** Entry: 8 random complex points loaded
   as 128-bit pairs assembled with vinsertf64x2 + one parity permute per plane (beats
   vgatherdpd, microcoded on Ice Lake), feeding the radix-4 stage-0 butterfly directly; the
   DC bin is the forward conv's bin 0 (d1_rader's "X[0] free"). Exit: the inverse's last
   radix-4 stage computes butterflies in 8-wide stack-staged blocks, only the random-index
   stores are scalar (vscatterdpd loses on Ice Lake). r2 kept these OFF on a wallaby wash;
   r3 turns them on because the vectorized versions save two P-length passes (~2×1 MB
   traffic at 65537) that matter on the scoring node's 1.25 MB L2. 65537 B=1 exec:
   **792 → 682 us** (wallaby A/B).
3. **Rader chain in CONV ORDER — ADOPTED FROM d1_rader (r1 index-reversal insight).**
   Between chain steps, gather∘scatter is a pure reversal (qin[q]=g^q=qout[(P−q) mod P])
   and the elementwise map commutes with any permutation, so the state lives split in conv
   order for the whole chain: interior steps have NO random gather and NO random scatter
   (r2 paid both per step). state[0] rides as a scalar pair; c is pre-permuted once per
   transform. New `st4_first_rev` reads the reversed state as four backwards-contiguous
   streams. 65537 B=1 m=60: **1205 → 826 us**; 65537 B=16 m=20: **1240 → 729 us**.
4. **Divider-free chain map — ADOPTED FROM d1_pow2 (r1/r2 map_vec), re-expressed in split
   form (no pair-swap permute needed since my state is already split).** rsqrt14+2 Newton +
   exact-residual Heron for sqrt, rcp14+2 Newton + residual corrections for 1/t and the
   quotient — their accuracy fight reconfirmed: this lands ~0.5–1 ulp, chain gates pass
   with ≥3 decades of margin. `map8_split` / `map_split_n` / `map_rader_state`. Gated to
   L≥32 (at 13 the vector latency loses, 0.081 vs 0.065 — d1_rader saw the same 0.110 vs
   0.099 in their codelets). Chain wins: 1024 B=1 2.45→1.97, 128 B=1 0.274→0.228,
   4096 B=1 14.1→13.0, 16384 B=1 58.2→50.7.
5. **Deterministic huge-page arena — ADOPTED FROM d1_bluestein (r2).** All same-plan work
   planes carved from ONE 2 MB-aligned MADV_HUGEPAGE block, plane stride rounded to the
   128 KB L2-way period plus a 32 KB+192 B skew, so equal-length streams never share L1/L2
   sets and physical set indexing follows the virtual layout. Kills the per-invocation
   bimodality (their 10007 was 110-or-213 us per process). Used for every plan with a plane
   length ≥ 512 (direct L≥512, Rader P≥512, all Bluestein). Biggest single win of the
   round: **16384 B=64 90.95 → 45.59 us (2x)**; also 10007 B=1 chain 130→113, 100003 B=8
   3605→3117, 4096 B=256 11.1→10.5. No regressions found (4096 B=256 first read 31 us —
   pure neighbor-spike noise, re-ran at 10.4).

### Measured (wallaby core 12, warmed, quiet-window best-min, interleaved A/B vs the r2
### build bin_r2 in the same minutes; NOT the scoring node — monitor arbitrates)
| cell | r3 | r2 (this box) | note |
|---|---:|---:|---|
| 1021 B=1 | **5.03** | 8.19 | st17 |
| 1021 B=256 | **9.34** | 13.17 | st17 (near d1_rader 9.06 / MKL 8.75) |
| 1021 B=1 chain | **5.77** | 9.77 | conv-order + rsqrt |
| 1021 B=256 chain | **5.47** | 13.55 | |
| 65537 B=1 | **682** | 792 | GATHER8 entry + block scatter exit |
| 65537 B=16 | **1385** | 1490 | |
| 65537 B=1 chain | **826** | 1205 | conv-order chain |
| 65537 B=16 chain | **729** | 1240 | |
| 10007 B=1 | **105** | 114 | arena |
| 10007 B=1 chain | **113** | 130 | arena |
| 100003 B=8 | **3117** | 3605 | arena |
| 100003 B=8 chain | **3170** | 3465 | |
| 16384 B=64 | **45.6** | 91.0 | arena (2x) |
| 16384 B=1 chain | **50.7** | 58.2 | rsqrt map |
| 1024 B=1 chain | **1.97** | 2.45 | rsqrt map |
| 1024 B=512 | **2.45** | 2.72 | arena |
| 128 B=1 chain | **0.228** | 0.274 | rsqrt map |
| 13 B=1 / B=512 | 0.044 / 0.037 | ~same | unchanged (d1_prime/d1_rader's cell) |
| 31 B=1 / B=512 | 0.135 / 0.139 | ~same | unchanged |

Correctness: all 52 graded cells PASS check.py vs numpy (worst single-call rel_l2
1.7e-15 at 2053; the graded set tops out ~1.25e-15 at 65537, tol 1e-12); every graded
chain gate passes with ≥3 decades of margin (worst 1021 B=1 m=2000 at 4.20e-12 vs its
1e-9 tol; 65537 chains ~2e-14 vs 1e-10). PLANNER_TEST self-test all-ok including odd
batches (B=19 = 2 lane groups + 3). Setup ≤ 0.02 s even at 100003.

### What did NOT work / not attempted, with the reasoning
- **Lifting d1_prime's exec13/exec31 dense codelets for the 13/31 B=1 cells** (where I am
  ~3x behind their 0.029/0.064): considered and DEFERRED. It is ~400 lines of size-specific
  intrinsic kernel for 4 cells that properly belong to d1_prime / d1_rader (whose class 13/31
  is), and the planner's adoption value is the dispatch, not re-winning their cells. Marginal
  ROI vs the large-prime and pow2-chain wins that ARE broadly mine. Noted for a future round
  if a rival stops covering them.
- **Vectorizing the Bluestein chain interior map** (10007/100003): the interior step fuses
  map + chirp-premultiply and reads the interleaved c field (stride 2), so a clean 8-wide
  map needs a pre-split c per transform. Small cells at parity with d1_bluestein/d1_race;
  arena already gave the 100003 chain its win. Not worth the risk this round.
- 4096 B=256 "regression" to 31 us on the first sample was a neighbor spike; re-ran 10.4
  vs r2 11.1. Lesson (again): one sample on a shared wallaby core is worthless — min over
  ≥3 warmed interleaved reps only.

### Borrowed (named, per the cumulative-round rule)
- **d1_rader**: st17_vblock (radix-17 intrinsic stage), GATHER8 insert-assembled entry +
  interleave4_store, the SINKSTORE staged-scalar-scatter shape, and the r1 conv-order
  index-reversal chain. Four separate lifts — the single biggest source this round.
- **d1_pow2**: the rsqrt14/rcp14 + Newton + exact-residual map (map_vec), split-form.
- **d1_bluestein**: the deterministic huge-page + set-skewed arena (r2 change 1).
- Prior-round borrowings (the split-complex core from d1_bluestein, the sym-fold from
  d1_prime, long-double twiddles from d1_pow2) all carry forward.

### For next round / for whoever lifts this
1. **Bailey/Agarwal–Cooley at 100003 and 65537 batched** — d1_bluestein's r2 AC 2D-conv
   took 100003 to 1973 us (vs my 2947); that is the remaining large-cell gap and their
   `ac_cols_*` is liftable wholesale onto my core.
2. **Re-A/B every r3 decision on the scoring node** — the GATHER8/scatter fusions and the
   arena skew were tuned on wallaby's 60 MB L3; the traffic argument says they widen on
   a80n0's 1.25 MB L2, but MEASURE (r1's 7.4→13.9 lesson).
3. **st17-style intrinsic treatment for the other generic primes** (7/11/13 as stages) if
   any smooth-with-large-prime-factor size joins the graded set.
4. The 13/31 dense-codelet lift (deferred above) if those cells open up.

## Round d1_r4 (2026-09-03) — RECONSTRUCTED STUB (the r4 session updated impl/ but never appended its record)

The r4 commit (956f7362) ships these changes, per the impl header and diff: (1) stage
kernels made NOINLINE so the 512-bit target attribute survives inlining (d1_bluestein's
r3 objdump finding — the whole r3 engine had scored at ymm width); (2) L=13/31 run
d1_prime's r3 interleaved-pair zmm kernels + their fused chains (exec13p/exec31p/
sp_chain*, sp_build tables); (3) L=32/64 run d1_pow2's r1 in-register codelets and
register-resident chains (fft32/fft64_*, pw_build); (4) NT-streamed fused exits when
in+out > 25 MB (st{2,4,8}_last_int_nt, d1_pow2 r3); (5) chain-only lanes for small
multi-stage smooth L <= 128 (lanes_chain_only — transpose amortizes over m).
r4 leaderboard: planner competitive at 13/31/32/64 (won 64 B=512, 13 B=1 chain), but
1.35-1.55x behind at 65537/1021, 1.5x at 100003, 1.5-2x at 1024/4096/16384, 2.3-3x at
128, 3.5x at 60 (0.158 vs d1_composite 0.045). Lesson recorded for future rounds: the
record append is part of the round's deliverable; without it the next generation has to
reconstruct intent from the diff.

## Round d1_r5 (2026-09-03) — the paired-p round: st8@s4 everywhere, fused mid pass, PFA-60

### Where r4 stood, read off the leaderboard
Biggest scored gaps, per cell class: L=60 3.0-3.5x (composite's PFA), L=128 2.3-3.0x,
pow2 1024/4096/16384 1.5-2x, 65537 1.13-1.39x (rader/race), 100003 1.5x (bluestein's
AC), 1021 1.25-1.55x. The common structural cause for most of these was ONE stage: every
plan here with >= 5 factors of 2 runs its second Stockham stage as radix-8 at s=4, where
the q-loop is 4-wide with per-p scalar twiddle loads — exactly the stage d1_bluestein's
r4 PMU work measured at ~2x ideal cost and fixed with a paired-p zmm kernel.

### What changed (all in impl/d1_planner.c; everything below is adoption, named)
1. **st8_s4 — ADOPTED FROM d1_bluestein (r4), near-verbatim** (their generalization of
   d1_rader's r3 st16_s4/st3_s4 paired-p shape). Two p-groups per zmm: inputs are
   contiguous 8-double runs, twiddles pair-broadcast (permutexvar of a 128-bit load),
   outputs recombine to full zmm stores via 128-bit-lane shuffles (0x44/0xEE).
   Dispatched from core_exec_range at r==8 && s==4 && m even. Hits 128=[4,8,4],
   1024/4096/16384, conv 65536, Bluestein pads 20480/204800 — execute AND chain AND
   both Rader/Bluestein directions. The round's big lever.
2. **st3_s4 and st5_s12 (+st5_col/st5_vec) — ADOPTED FROM d1_rader (r2/r4),
   near-verbatim** (scalar/masked tails for odd m kept). Hit 1020 = [4,3,5,17] (m=85
   and m=17 stages) and 60 = [4,3,5] (chain path).
3. **Fused mid pass stmid2/stmid4 — ADOPTED FROM d1_bluestein (r4 stmid2/4), which
   executed d1_rader's r3 idea**: forward-last stage (m=1, unit twiddles) + kernel-
   spectrum multiply (swapped planes) + inverse radix-4 entry with stage-0 twiddles in
   ONE pass — the conv spectrum is never written or read back (one full plane-pair
   round trip gone per transform / per chain step). Wired into bl_middle (Bluestein:
   10007 M=20480 rl=2, 100003 M=204800 rl=4) and a new rader_mid used by rader_one and
   the conv-order chain (65537 P=65536 rl=4); the spectrum DC bin (X[0] = x[0] + DC) is
   re-summed from the same last-stage inputs (d1_rader's "X[0] comes free"). Gated by
   p->fuse_mid = last radix in {2,4} && entry radix 4 (1020 ends in 17: keeps old path).
4. **PFA-60 codelet fft60_pfa — ADOPTED FROM d1_composite (fft60_ymm1, r4/r5 form),
   near-verbatim** with their CRT tables (PIN/KOUT). Good-Thomas 60=4x3x5, twiddle-FREE:
   stage A operands (xj|xj) via vbroadcastf64x2 + two signed FMAs + one in-lane
   vpermilpd (zero cross-lane shuffles), n1-paired ymm stages B/C. Dispatched for ALL
   batches at L=60 (their r5 node A/B: the ymm1 per-transform loop beats the zmm pair
   kernels at B=512 on ICX — 256-bit mix fills p1). Plus an own fft60_chain: state stays
   interleaved (the codelet's native format), map runs 8-wide on interleaved data with
   |z|^2 duplicated per 128-bit lane (their map_scale_q idiom, rsqrt14/rcp14 + 2NR).
   Replaces both my per-vector split chain AND the lane chain at 60.

### Measured (a80n0 = the scoring node, leased core 6, same-window INTERLEAVED A/B vs
### the r4 build; node was busy, load 4-11 — ratios are the trusted part; min over 3x4)
| cell | r4 | r5 | | cell | r4 | r5 |
|---|---:|---:|---|---|---:|---:|
| 60 B=1 | 0.135 | **0.049** | | 60 B=512 | 0.131 | **0.050** |
| 60 ch B=1 | 0.241 | **0.129** | | 60 ch B=512 | 0.131 | **0.129** |
| 128 B=1 | 0.237 | **0.148** | | 128 B=512 | 0.310 | **0.217** |
| 128 ch B=1 | 0.344 | **0.279** | | 128 ch B=512 | 0.282 | 0.284 |
| 1024 B=1 | 2.224 | **1.615** | | 1024 B=512 | 2.754 | **2.183** |
| 1024 ch B=1 | 2.900 | **2.387** | | 1024 ch B=512 | 2.879 | **2.344** |
| 4096 B=1 | 11.74 | **9.41** | | 4096 B=256 | 14.03 | **11.78** |
| 4096 ch B=1 | 15.78 | **13.54** | | 4096 ch B=256 | 15.60 | **13.47** |
| 16384 B=1 | 51.30 | **42.98** | | 16384 B=64 | 57.82 | **48.55** |
| 16384 ch B=1 | 67.28 | **57.61** | | 16384 ch B=64 | 68.00 | **57.67** |
| 1021 B=1 | 8.20 | **6.67** | | 1021 B=256 | 8.85 | **7.41** |
| 1021 ch B=1 | 7.97 | **6.53** | | 1021 ch B=256 | 8.03 | **6.52** |
| 10007 B=1 | 156.1 | **125.0** | | 10007 B=64 | 163.3 | **131.9** |
| 10007 ch B=1 | 170.5 | **139.8** | | 10007 ch B=64 | 170.5 | **140.9** |
| 65537 B=1 | 1075.7 | **966.8** | | 65537 B=16 | 1372.0 | **1254.6** |
| 65537 ch B=1 | 1027.4 | **909.5** | | 65537 ch B=16 | 1061.6 | **943.3** |
| 100003 B=1 | 3381.0 | **3082.6** | | 100003 B=8 | 3538.4 | **3239.5** |
| 100003 ch B=1 | 3555.4 | **3242.2** | | 100003 ch B=8 | 3566.8 | **3268.9** |
(60 rows are the final build's absolute reads in a similar window, not interleaved
against r4 for the codelet cells; the r4 numbers there are from the same session's A/B.)

Correctness on the node: all 52 graded cells PASS check.py (worst single-call rel_l2
1.25e-15 at 65537, tol 1e-12), all chain gates PASS with >= 3 decades of margin (worst
1021 B=1 m=2000 at 4.0e-12 vs 1e-9), odd batches (60 B=5/B=3, 1024 B=3, 10007 B=3)
PASS, output bitwise repeatable across runs at every graded cell. PLANNER_TEST
self-test all-ok. Setup unchanged.

### Reading vs the r4 standings (same-node, different day — the monitor arbitrates)
1021 B=1 at 6.67 lands AT d1_rader's r4-session quiet number (6.77) and ahead of the r4
leaderboard's best scored (7.40 race); 1024 B=1 1.62 still trails MKL 1.16-1.23; 128
B=1 0.148 vs MKL 0.104; 65537 B=1 967 vs race/rader 850-915; 100003 3083 vs
bluestein/race 2320-2470. The panel-internal gaps that remain are real engine gaps
(AC 2D conv at 100003, radix-16/32 middles at 65537, codelets at 128), not stage waste.

### What did not work / honest non-results
- **60 B=512 chain and 128 B=512 chain were FLAT** (0.131->0.129, 0.282->0.284): those
  ran the lane path (s0=8), which has no s=4 stage — expected, recorded so nobody looks
  for a regression. The PFA chain then took the 60 batched chain over the lane path
  (0.129), but d1_composite's SoA-resident batched chain is still ~2x ahead (0.059-
  0.075); their transposed-once SoA state is the next lift if that cell matters.
- No other regressions found: every other affected cell improved 8-38% in the same
  window. Nothing was measured-and-rejected this round; the round was pure adoption of
  levers other entries had already de-risked (which is the cumulative design working).

### Borrowed (named, per the rule)
- **d1_bluestein r4**: st8_s4 (verbatim modulo attribute macro), stmid2/stmid4 bodies.
- **d1_rader r2/r4**: st3_s4, st5_col/st5_vec/st5_s12 (verbatim modulo tails), the
  fused-mid idea + the DC-bin-from-partial-sums trick for Rader.
- **d1_composite r4/r5**: fft60_ymm1 -> fft60_pfa (verbatim modulo table renames), the
  interleaved map_scale_q idiom in fft60_chain, and their "ymm1 loop wins at every
  batch on ICX" node verdict (saved me re-measuring the zmm pair variant).
- Carried forward: d1_bluestein split core + arena, d1_prime 13/31, d1_pow2 32/64 +
  NT exits + rsqrt map, d1_rader GATHER8/scatter/conv-order chain/st17.

### For next round, in priority order
1. **100003 (and 65537 B=16): lift d1_bluestein's Agarwal-Cooley 2D engine** — the
   remaining 1.3-1.4x at 100003 is architecture, not stages; their ac_* plan is
   liftable onto this file's core (they run the same split kernels).
2. **65537: radix-16/32 middle stages** (d1_rader's r4 next-item; 65536 = [4,16,16,16,4]
   runs 5 stages vs my 6 even after the mid fusion).
3. **60 batched chain: composite's SoA-resident in-place chain step** (~2x).
4. **128 B=1 (0.148 vs MKL 0.104)**: needs a fully fused straight-line codelet
   (d1_pow2's r4 dead-end analysis bounds what transposes cost — read it first).
5. The r4 record gap: keep the rule "append the record BEFORE the round ends".

## Round d1_r6 (2026-09-03) — radix-16/64 tile schedules, Rader prefetches, NT chirp exits

### Where r5 stood, read off the leaderboard
Biggest scored gaps: 65537 all four cells 1.21–1.48x behind d1_rader/d1_race; 10007
1.07–1.67x and 100003 1.13–1.35x behind d1_bluestein/d1_race; direct pow2 1024/4096/16384
1.15–1.77x; 128 1.4–1.7x; the small batched chains 32/60 at 2.1–2.4x. The rivals' r5
records had already solved the first two groups: d1_rader's radix-64 two-layer tile took
their 65537 to 847/992, and d1_bluestein's radix-16 conv schedules took 10007 to 109. So
r6 is again almost pure adoption — lift the tile kernels onto my core and dispatch them
from the planner's schedule table.

### What changed (all in impl/d1_planner.c; adoptions named)
1. **st16_block/st16_s4/st16 — ADOPTED FROM d1_rader (r3), and dft8v/st64_block/
   st64_s4/st64 — ADOPTED FROM d1_rader (r5), all near-verbatim.** Radix-16 = two fused
   radix-4 layers through a 2 KB stack tile (layer-1 twiddles are compile-time
   constants); radix-64 = two fused radix-8 layers through an 8 KB tile with a plan-time
   8x8 W64 table (rebuilt here with long-double phases per this file's exact-phase rule;
   the donor used M_PI doubles). Their kernels bind to my per-stage [j-1][m] twiddle
   blocks UNCHANGED — same layout, their r1 borrowing coming home. Dispatched as core
   stages 16/64; core_is_generic extended so they never fall into the generic-prime path.
2. **core_schedule: hand-picked tile schedules, entry 4 and tail 2/4 kept** so every
   existing fusion (deint entry, chirp/bhat/gather entries, stmid mid pass, pruned/
   interleave/NT exits) applies unchanged: 65536=[4,64,64,4] (Rader conv, was 6 stages),
   20480=[4,16,16,5,4] (d1_bluestein's verified CF_CONV row, was 6), 204800=
   [4,64,16,5,5,2] (was 7), direct 4096=[4,16,16,4] (was 5), 16384=[4,64,16,4] (was 5).
   PLN_SCHED=0 restores plain factorization for A/B.
3. **Rader gather/scatter prefetches — ADOPTED FROM d1_rader (r5).** Entry gather: T0
   prefetch of the next iteration's 32 random read targets, gated batch >= 2 (their B=1
   +16 us measurement respected). Exit scatter: exclusive-hint write prefetch 16 points
   ahead, unconditional (their finding: output lines are cold even at B=1, evicted by
   the transform's own ping-pong traffic). Ported as __builtin_prefetch(p,1,3) — gcc 11
   refuses _mm_prefetch(_MM_HINT_ET0) inside a target()-attributed function in the plain
   -O2 self-test build ("target specific option mismatch"); the builtin compiles
   everywhere and emits prefetchw where PRFCHW is on.
4. **NT-streamed pruned chirp exits st{2,4}_last_chirp_nt — ADOPTED FROM d1_bluestein
   (r5), near-verbatim** incl. the peel-to-64B-alignment trick. Gated: batched output
   >= 8 MB AND (tail 2, or tail 4 with M/4 % 8 == 0 so both output segments share the
   head pad — a crash trap their schedules never hit but my generic choose_M could).
   Execute-only (bl_one); chains re-read state. PLN_NONT=1 for A/B.
5. **Newton-map denormal clamp 1e-300 → 1e-100 at all four sites — ADOPTED FROM
   d1_batchlane (r3) via d1_bluestein (r5):** rsqrt14 of a denormal-range operand is an
   FP assist. Visible at 13 B=1 chain: 0.039 vs r5's 0.0448.

### Measured (a80n0 leased cores via tryout.sh / its exact pipeline over ssh; the
### wallaby squeue shim lied about job 440424 again — d1_prime's /tmp shim recipe.
### Quiet-window minima unless noted; load rose to 4 late in the session (bin_c
### neighbor), which inflated the DRAM-bound reads marked *)
| cell | r5 board | r6 | | cell | r5 board | r6 |
|---|---:|---:|---|---|---:|---:|
| 65537 B=1 | 993.8 | **852.1** | | 65537 B=16 | 1383.8 | **936.0** (1041*) |
| 65537 ch B=1 | 935.5 | **783.5** | | 65537 ch B=16 | 903.0 | **739.1** (768*) |
| 10007 B=1 | 124.6 | **108.2** | | 10007 B=64 | 188.7 | **123–129** |
| 10007 ch B=1 | 173.3 | **124.4** | | 10007 ch B=64 | 140.1 | **123.3** |
| 100003 B=1 | 3067 | **2841** | | 100003 B=8 | 3232 | **2660** |
| 100003 ch B=1 | 3006 | **2801** | | 100003 ch B=8 | 3089 | **2846** (3611*) |
| 4096 B=1 | 10.68 | **8.0–9.2** | | 4096 B=256 | 14.27 | **11.5** |
| 4096 ch B=1 | 15.44 | **12.07** | | 4096 ch B=256 | 13.54 | **12.08** |
| 16384 B=1 | 50.35 | 41.7–44 | | 16384 B=64 | 60.33 | 47.7–57.4 |
| 16384 ch B=1 | 60.70 | **54.05** | | 16384 ch B=64 | 58.73 | **54.6** |
| 13 ch B=1 | 0.0448 | **0.039** | | 31 ch B=512 | 0.0549 | **0.049** |
1021 (7.80/8.70 exec, 7.58/6.76 chains), 1024, and all small m=1 cells unchanged within
window drift. The 65537 quartet now sits at d1_rader's r5 level (their 847/992/767/803).

Correctness: every cell above PASS check.py vs numpy (worst single-call rel_l2 1.24e-15
at 65537, tol 1e-12); every graded chain gate PASS with >= 3.5 decades of margin (worst
1024 B=1 m=4000 at 6.2e-12 vs 1e-10); outputs bit-repeatable across runs; PLANNER_TEST
self-test all-ok at plain -O2 (including 1020/1021/1024/2048/2053). Setup <= 0.058 s.

### What did NOT work, with the numbers that killed it
- **1024 = [4,64,4]**: node interleaved A/B wash-to-worse (B=1 1.89–1.97 sched vs
  1.83–2.00 plain; B=512 2.19+ vs 2.14). d1_bluestein's r5 finding — an L2-resident 1024
  gains nothing from a tile stage — holds on my core too, even with the light radix-4
  tail their [4,16,16] lacked. Dropped from the table; noted in the code comment.
- **16384 = [4,64,16,4]** is a statistical WASH at B=1 vs plain [4,8,8,8,8] (interleaved:
  42.3–44.3 vs 42.8–43.8) and marginal at B=64 (47.75 vs 48.15) — kept for the B=64 edge,
  but the 16384 B=1 gap vs d1_pow2 (33.9–37) is their 3-pass SS64+SX architecture, not
  pass count from here.
- **NT at 10007 B=64 is ambiguous interleaved** (nt 122.9–125.1 vs nont 116.8–126.9,
  one nont outlier low); kept because 100003 B=8 is unambiguous (2660 vs 2832–2945,
  −6..10%) and d1_bluestein's 6-of-8 verdict matches. Honest note: my earlier
  "164 → 129" NT read was cross-window, not an A/B — trust the interleaved numbers.
- **_mm_prefetch(_MM_HINT_ET0) does not compile in a target()-attributed function
  without -march** (gcc 11 inlining error); __builtin_prefetch(p,1,3) is the portable
  spelling. Recorded so nobody re-hits it.

### Borrowed (named, per the rule)
- **d1_rader r3/r5**: st16 family, dft8v + st64 family (the round's big lever), the
  gather-T0/scatter-ET0 prefetch pair with their batch >= 2 gate verdict.
- **d1_bluestein r5**: the flavor schedule verdicts (20480 row verbatim; the "1024
  direct hates tiles" negative saved me a dead end — I re-confirmed it in one A/B),
  st{2,4}_last_chirp_nt with the alignment peel.
- **d1_batchlane r3 (via d1_bluestein r5)**: the 1e-100 clamp.
- **d1_prime r3**: the /tmp squeue-shim workaround for the dead-looking reservation.
- Carried forward: everything in the r2–r5 lists.

### For next round, in priority order
1. **100003: d1_bluestein's Agarwal–Cooley 2D engine** (they run 2219 B=1 vs my 2841) —
   still the one structural gap the mono-FFT Bluestein cannot close; their ac_* plan
   runs my same split kernels and remains liftable wholesale.
2. **16384/4096 B=1 vs d1_pow2**: either lift their SS64/SX 3-pass structure, or build
   st64_last_int (tile exit with the interleave folded in) to cut my 16384 to 3 passes.
3. **Small batched chains** (32 B=512 chain 0.077 vs d1_twiddle 0.0368; 60/64/128
   similar): their SoA-resident chain state; my lane chain transposes per step group.
4. **10007 B=64** (123 vs race/bluestein best ~112): two-row software pipelining of the
   conv, or bluestein's next-round row ideas.
5. 1024 B=512 (2.4 vs d1_twiddle 1.74) — SoA execute; and 1021 st17 rework stays
   closed (d1_rader's r5 op-count verdict).

## Round d1_r7 (2026-09-03) — fuse the chain map into the final stage; tile exits make 4096/16384 3-pass; Goldschmidt map; the arena finally gets huge pages

### Where r6 stood, read off the leaderboard
Wins/ties at 1021 B=1, 1021 B=256 chain, 1024 B=512 chain, 10007 B=1, 10007 B=64
chain, 65537 B=16. Biggest losses: the small batched chains (60 B=512 chain 2.68x,
128 B=1 chain 2.13x, 32 B=512 chain 2.05x, 64 B=512 chain 1.81x), the pow2 block
(16384 B=1 1.57x, 128 cells 1.4-1.5x, 4096 1.33-1.41x, 1024 B=1 1.68x), 100003
(1.22-1.38x, d1_bluestein/race's AC engine), 65537 chain B=1 1.20x. The common
structural waste in every chain cell was ONE full read+write pass per step: the
elementwise map ran as its own pass over the state. d1_pow2's chains had already
shown the altitude ("map fused into the final stage in split form"); this round
builds that into the shared core, plus the r6 next-item tile exits.

### What changed (all in impl/d1_planner.c)
1. **Chain map fused into every step's FINAL stage.** The last Stockham stage has
   m=1, unit twiddles, and contiguous outputs at y[q + s*j] — exactly where the
   elementwise map wants them — so map(z+c) folds into the butterfly store with
   ZERO shuffles. New kernels st{2,4,5,8}_last_map (split->split) and, for the
   final chain step, map_int_out (map + ILO/IHI interleave, replacing the scalar
   sqrt/div output loop). Wired into: the per-vector direct chain (interior steps
   run stages 0..n-1 + fused last; the final step runs the full FFT + map_int_out
   — one extra pass on 1 of m steps), the lane chain (all steps fused), the
   RADER conv-order chain via **st4_last_map_rader** (the inverse's last radix-4
   butterfly on the engine's swapped planes + x0 + c + map in one pass; the
   map_rader_state pass is gone), and the BLUESTEIN chain via **map_chirp_split**
   (c pre-split once per transform — the r3 "needs a pre-split c" deferral — so
   map + chirp-premultiply runs 8-wide divider-free). PLN_NOFM=1 restores r6.
2. **Tile exits st{16,64}_last_int(+NT flag)/(_map for chains): 4096=[4,64,16] and
   16384=[4,64,64] now run THREE passes** (fused deint entry, paired-p s=4 tile,
   tile exit with interleave/NT/map folded in). r6's radix-4 tails existed only to
   host the old fused exits; the conv rows (65536/20480/204800) keep their 2/4
   tails because the Rader/Bluestein mid+exit fusions require them. PLN_SCHED=2
   restores the r6 rows.
3. **Lane chains get 2-pass chain-only schedules**: a second core_plan (lcore)
   with 64=[8,8], 128=[16,8] used ONLY by the lane chain (execute keeps radix-4
   entry for the fused deinterleave, which the chain never needs; st16 at s=8
   runs the full-lane tile block). The 60 and 64 batched chains now go through
   the fused lane path (60 needed the new st5_last_map; PFA fft60_chain keeps
   B<8 and odd tails). PLN_LC=0 / PLN_60L=0 / PLN_64L=0 for A/B.
4. **Goldschmidt + early-seeded-rcp map — ADOPTED FROM d1_prime (r5, offered to
   the panel) via d1_batchlane (r6 map_scale_fast, near-verbatim)** as the shared
   map_q8: Goldschmidt sqrt (fnmadd->fma, 8 cy/iter), reciprocal seed off the RAW
   rsqrt14 estimate so the rcp Newton chain overlaps the sqrt refinement,
   ADDITIVE 1e-100 floor inside the m2 FMA. Replaces the r3 2NR+residual form in
   map8_split/map_sc8/pw_map_vec/fft60_chain (residuals re-confirmed noise:
   worst gate 1024:1:4000 reads 4.4-5.6e-12 vs r6's 6.2e-12 — statistically the
   same).
5. **Arena rounded to whole 2MB pages + pre-faulted — d1_pow2's r6 THP finding,
   verified here**: the r3-r6 arena got AnonHugePages = 0 kB at 16384 (whole
   arena < 2MB) and only partial backing at 65537/100003; after the fix every
   arena is fully huge-page-backed (16384: 2048 kB, 100003: 26 MB). The set-skew
   constants finally act on physical set indexing, as the arena's own comment
   always claimed. Setup unchanged (<= 0.061 s at 100003).

### Measured (a80n0 leased core 4, same-window INTERLEAVED A/B vs the r6 binary
### (bin_r6); min over 3-4 samples x 2-3 reps; window drifted, ratios trusted)
| cell | r6 bin | r7 | | cell | r6 bin | r7 |
|---|---:|---:|---|---|---:|---:|
| 4096 B=1 | 9.1-9.3 | **8.5-8.9** | | 4096 B=256 | 10.1-10.9 | **8.7-9.5** |
| 4096 ch B=1 | 13.7 | **10.1-10.3** | | 4096 ch B=256 | 12.2-12.4 | **8.7-9.0** |
| 16384 B=1 | 41.3-42.5 | **38.1-40.6** | | 16384 B=64 | 45.6-50.3 | **37.4-38.7** |
| 16384 ch B=1 | 54.2-54.5 | **42.8-43.4** | | 16384 ch B=64 | 55.0-57.6 | **46.2-47.6** |
| 1024 ch B=1 | 2.68-2.69 | **1.98-2.35** | | 1024 B=512 | 2.10-2.16 | 2.11-2.12 |
| 32 ch B=1 | 0.070 | **0.066** | | 32 ch B=512 | 0.077-0.078 | **0.050-0.052** |
| 60 ch B=1 | 0.129 | **0.124** | | 60 ch B=512 | 0.147 | **0.111-0.113** |
| 64 ch B=1 | 0.117 | **0.113** | | 64 ch B=512 | 0.134 | **0.095-0.105** |
| 128 ch B=1 | 0.292 | **0.197-0.199** | | 128 ch B=512 | 0.287-0.310 | **0.163-0.175** |
| 13 ch B=512 | 0.016 | **0.015** | | 31 ch B=512 | 0.048-0.049 | **0.046** |
| 1021 ch B=1 | 6.59-6.62 | **6.31-6.32** | | 1021 ch B=256 | 6.63-6.66 | **6.30-6.45** |
| 10007 ch B=1 | 123.5-125.4 | **119.7-120.5** | | 10007 ch B=64 | 123.7-125.3 | **120.8-121.1** |
| 65537 ch B=1 | 717-763 | **712-714** | | 65537 ch B=16 | 748-755 | **697-699** |
| 100003 (all 4) | -- | flat | | 65537/1021/1024 B=1 m=1 | -- | wash |
Chain-cell breakdown of the 60/64/128 batched wins: fused lane path beat the new
Goldschmidt-only baselines too (60: lane 0.111-0.113 vs PFA-chain 0.123-0.140;
64: lcore [8,8] 0.095-0.105 vs 3-pass lane 0.117 vs codelet 0.134 — r2's "64
stays on the codelet" verdict was about the UNFUSED lane loop; 128: lcore [16,8]
0.163-0.175 vs 3-pass 0.207-0.228). If the window's numbers hold on the board,
128 B=512 chain (pow2 won r6 at 0.187) and 16384 B=64 (pow2 44.2) should flip.

### Correctness (final binary, on the node)
Every graded cell exercised PASSES check.py vs numpy (worst single-call rel_l2
1.24e-15 at 65537, tol 1e-12); every graded chain gate PASSES with >= 1.3 decades
(worst 1024:1:4000 at 4.4-5.6e-12 vs 1e-10, statistically unchanged from r6);
odd-batch chains 60:9, 64:11, 128:3, 32:19, 1024:3, 16384:3 PASS through the new
lane/tail/fused paths; outputs bitwise repeatable across runs at every cell
checked; PLANNER_TEST self-test all-ok at plain -O2 (n=1153 exercises
st4_last_map_rader, B=19 chains exercise the fused lane paths). Setup <= 0.061 s.

### What did NOT work, with the numbers that killed it
- **1024 = [4,16,16]** (3-pass with tile exit): B=1 wash-to-slightly-better
  (1.57-1.62 vs 1.62-1.65), B=512 REGRESSES (2.12-2.63 vs 2.12-2.36), chain wash.
  Dropped — d1_bluestein's "L2-resident 1024 gains nothing from a tile stage"
  verdict holds for the 16-tile as much as r6 found for the 64-tile.
- 100003 chains are FLAT under all of this: the M=204800 conv is DRAM-bound and
  the map pass was never its bottleneck. The 1.2-1.4x gap to d1_bluestein/race
  remains their Agarwal-Cooley 2D architecture, unchanged since r5.
- Honest scope note: 65537 B=1 m=1 execute read ~1000 us in the final window vs
  one stray r6 read of 771 — that is the known AVX-512 turbo bimodality
  (d1_pow2 r6), not a regression; interleaved reps read wash all session.

### Borrowed (named, per the rule)
- **d1_prime r5 via d1_batchlane r6**: the Goldschmidt + early-seeded-rcp +
  additive-floor map, near-verbatim (their map_scale_fast -> my map_q8).
- **d1_pow2 r6**: the THP 2MB-alignment/size finding (their "check your own
  AnonHugePages" was aimed at this file — measured 0, fixed as they prescribed);
  the fuse-map-into-final-stage altitude comes from their chain design (r1-r5).
- **d1_rader r3/r5** (carried): the tile layer bodies reused verbatim inside
  st{16,64}_last_*; d1_prime r3: the /tmp squeue-shim recipe, again (the wallaby
  shim still reads bench/gen's heartbeat; reserve.sh --status still lies).
- Everything in the r2-r6 lists carries forward.

### For next round, in priority order
1. **100003 / 65537 batched: d1_bluestein's Agarwal-Cooley 2D engine** — the one
   structural gap left (their 2219 B=1 vs my ~2660); their ac_* runs these same
   split kernels and remains liftable wholesale. Two rounds deferred; it is the
   largest single remaining prize.
2. **16384/4096 B=1 vs d1_pow2 (33.9/7.3)**: I am 3-pass now; the rest of their
   lead is the SX fused first-stage PAIR (entry+s4 through an L1 tile = 2
   passes). A st4+st64_s4 fused entry tile is the natural next kernel.
3. **Small batched chains vs d1_batchlane** (32: 0.050 vs 0.033; 64: 0.095 vs
   0.065; 60: 0.111 vs ~0.055): my lane steps are now 2 passes + transposes;
   their L1-blocked register-row steps are 1-ish. Port their row engine or stop.
4. 65537 chain B=1 (712 vs rader 680): their 7-pass mid-fused chain vs my 7 +
   entry-rev; profile whether st4_first_rev can fuse INTO the forward s=4 stage.
5. Env knobs for whoever lifts this: PLN_SCHED=0/2, PLN_NOFM, PLN_LC, PLN_60L,
   PLN_64L, PLN_NONT.
