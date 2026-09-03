# d1_twiddle — strategy record

Class: LIBRARY LAYER (adoption-scored). The mission is exact 1D twiddle tables in
consumption order, for every other entry to adopt. The FFT in this entry is the
demonstration vehicle, not the product.

## Round d1_r1 (2026-09-02) — from the stub to a working library layer + vehicle

Starting point: the restart stub (dense O(L^2) DFT, `supports() == 0`, no tables worth
adopting). Nothing survived the lost r1–r3, so everything below is new this round.

### What the entry now is

**1. The ADOPTION BLOCK** (top of `impl/d1_twiddle.c`, marked, self-contained, copy it whole):

- `d1tw_cexp(num, den)` — `exp(-2*pi*i*num/den)` for any int64 `num`, `den < 2^59`.
  Two *exact* reductions before any floating-point trig: the index mod `den` in
  integers, then quadrant reduction `q = round(4r/den)`, `s = 4r - q*den` (exact in
  int64), so the trig argument is `(pi/2)(s/den)` with `|arg| <= pi/4` and the
  quadrant factors are exactly 0/±1 (no sqrt(2)/2 rounding anywhere). The argument is
  built from a two-part pi/2 split with an FMA-recovered product tail.
  **Verified against 150-bit mpmath over 13k adversarial (k,den) pairs (dens 13, 60,
  1024, 16384, 100003, 131072, 200006): worst component error 1.24 x 2^-53** — i.e.
  ~encroaching one ulp, the survey's "correctly-rounded/dd tables" requirement met with
  plain doubles. This is the direct implementation of survey convergence #2
  ("inaccurate twiddles are the leading cause of FFT inaccuracy; never in-loop
  recurrences").
- `d1tw_chirp(N, n, w)` — Bluestein chirp `exp(-i*pi*k^2/N)` with `k^2 mod 2N` reduced
  **in integers** before the trig call (the survey's explicit fp64 trap at k ~ 1e5).
  Built for d1_bluestein at 10007/100003; valid to N ~ 3e9.
- `d1tw_stage(n, r, tw)` — Stockham stage tables in **consumption order**: for the pass
  `y[q+s*(r*p+t)] = sum_i x[q+s*(p+m*i)] W_r^{it} W_n^{pt}`, entries are laid out
  p-major, t-minor, so the butterfly loop reads the table strictly linearly (no index
  arithmetic, no strided twiddle loads). The last stage of any plan is p=0 only, its
  table is identically 1 — skip it.

**2. The vehicle**: mixed-radix (2/3/4/5/8) Stockham autosort FFT, `supports()` = all
smooth `L = 2^a 3^b 5^c` in [2, 2^20] — owning graded sizes 32/60/64/128/1024/4096/16384.
pow2 exponent split into 8s/4s to minimize stage count (never radix-2 except L=2·odd);
factor order pow2 → 3s → 5s. Ping-pong through two scratch buffers; in-place safe.
Plus a fused-map `fft1d_chain`: the graded map `state <- (z+c)/(1+|z+c|)` is applied
inside the final stage's store loop, so a chain step is `nf` array sweeps instead of the
fallback's `nf + 1`.

### Measured this round (wallaby login node, taskset core 100, nice — NOT the scoring
node; a80n0 reservation was down all session, so no Ice Lake numbers yet. Treat these as
structure checks, not scores. Baseline column = best library on a80n0 from
results/library_baseline/BASELINE.md, chained rows, different silicon.)

| cell | mine (wallaby) us | best lib (a80n0) us |
|---|---|---|
| 32 B=1 m=1 | 0.061 | — |
| 32 B=512 m=1 | 0.061 | — |
| 32 B=1 chain | 0.140 | 0.131 (MKL) |
| 32 B=512 chain | 0.148 | 0.112 (MKL) |
| 60 B=1 m=1 | 0.151 | — |
| 60 B=1 chain | 0.324 | 0.232 (fftw-measure) |
| 64 B=1 m=1 | 0.119 | — |
| 64 B=1 chain | 0.277 | 0.237 (MKL) |
| 128 B=1 m=1 | 0.354 | — |
| 128 B=1 chain | 0.716 | 0.485 (fftw-patient) |
| 1024 B=1 m=1 | 4.30 | — |
| 1024 B=1 chain | 6.55 | 4.18 (MKL) |
| 4096 B=1 m=1 | 14.8 | — |
| 16384 B=1 m=1 | 75.4 | — |
| 16384 B=64 m=1 | 97.4 | — |
| 16384 B=1 chain | 118.6 | 82.2 (fftw-patient) |

Correctness: rel L2 1e-16..3.9e-16 vs numpy at every supported size tested (20 sizes,
B=3 and B=512 spot checks); map-chain gate PASS at m=40..100000 across sizes
(e.g. L=32 B=512 m=1000: 5.0e-13 against a 1e-10 tolerance). Repeatable output.

### What was tried and what the numbers said

- **Runtime `if (cm)` branch per store in the fused-map last stage: killed it.**
  First version had the map as a per-store conditional; at L=32 B=1 chained it measured
  0.240 us vs 0.128 us for the *driver fallback* — the branch prevented vectorization
  of the map's sqrt. Fix: instantiate each last-stage kernel twice from one
  always_inline body with a compile-time `domap` flag. After that the fused chain wins
  or ties everywhere measured: 32 B=512 chain 0.122 vs 0.133 fallback, 128 B=1 chain
  0.619 vs 0.700, 1024 B=1 chain 6.42 vs 6.56. Lesson for adopters: specialize the
  fused store path at compile time, never branch per element.
- **L=60 factor ordering A/B** (env-parameterized throwaway build, orders 435/345/543/
  534/453/354): [4,3,5] and [3,4,5] best at ~0.14 us B=1; [3,5,4] worst at 0.20.
  Kept [4,3,5] (the default the factorizer already produces). An early unpinned
  measurement showing 0.28 us for the default order was login-node noise — pin before
  believing any wallaby number.
- **Quadrant vs octant reduction** in `d1tw_cexp`: chose quadrant (multipliers exactly
  0/±1) over octant (needs sqrt(2)/2, one extra rounding). One mpmath metric trap worth
  recording: measuring error in "ulps of the returned component" explodes when the true
  value is 0 and we return exactly 0.0 (ulp(0) is denormal-sized) — measure in absolute
  2^-53 units instead.

### Known weaknesses / what I would do next

1. **No Ice Lake numbers.** First action next round: `./tryout.sh d1_twiddle` on every
   owned size the moment the reservation is live; wallaby (Sapphire Rapids, contended)
   proves structure only.
2. **First stage at B=1 is scalar-ish** (s=1 means the vector axis has length 1; the
   p-loop loads are contiguous but stores are stride-r). A dedicated s==1 kernel with
   in-register transpose, or simply doing the FIRST stage as radix-4 always (its s=1
   butterfly is cheapest), is the next lever for the non-batched pow2 cells. d1_pow2's
   codelets will likely beat me there — that is fine, they should adopt the tables.
3. **Chained cells trail the libraries** (the map's sqrt+div is ~half the step cost at
   small L; the libraries' chained numbers ride the same driver map but their FFT is
   faster in the loop). The batch-lane trick (survey #1, d1_batchlane's class) applied
   to the batched chain — vectorizing the map ACROSS transforms — is the real fix for
   B=512 chained; my per-transform fusion only removes a sweep.
4. **L=60**: mixed-radix Stockham does 0.15 us where the libraries' chained rate is
   0.23; promising, but Good-Thomas PFA (twiddle-free, coprime 4·3·5) is the literature
   answer and belongs to d1_composite; if they adopt `d1tw_cexp` for their PFA I should
   not duplicate it.
5. Batched large-L (16384 B=64) goes memory-bound: four-step 128x128 with L1-resident
   sub-FFTs (survey per-size map) is the known fix; not attempted this round.

### Borrowed / lent

- Borrowed: nothing (no other strategy records existed this round — fresh restart).
- Lent: the ADOPTION BLOCK is written to be copied (d1_bluestein: `d1tw_chirp`;
  d1_rader: `d1tw_cexp` for root-power tables at 65537 where den = 65537 and the
  generator-permuted exponents are just `num`; d1_pow2/d1_composite/d1_batchlane:
  `d1tw_stage` consumption-order tables + the compile-time-specialization lesson above).

## Round d1_r2 (2026-09-02/03) — AVX-512 vehicle, v2 adoption formats, fast fused map

The r1 leaderboard put the vehicle 3–5x behind the leaders at every owned cell: the
whole implementation was plain C hoping for auto-vectorization, and d1_batchlane's r1
record explains why that ceiling exists (the panel flags carry no
`-mprefer-vector-width=512`, so gcc 11 vectorizes at ymm at best, and the map's
sqrt/div came out half-width). This round is a full kernel rewrite in AVX-512
intrinsics, adopting three named things from rivals, plus a v2 of the adoption block.
The Ice Lake reservation was dead again all session (job 440371 not running), so all
numbers are wallaby (Xeon 6448Y), pinned to a measured-idle core, min over samples,
each cell re-run to steady state — same method as d1_pow2's r1.

### What changed

1. **All hot kernels are now explicit `_mm512` intrinsics** (interleaved complex, one
   zmm = 4 complexes). Complex twiddle multiply is the 1-vpermilpd + mul + fma idiom on
   `(re,re)` / `(-im,+im)` broadcast pairs — **borrowed from d1_pow2** (their r1 code,
   `cmul_bc`). Scalar r1 kernels are kept as fallback for odd-s stages (L = 2·odd, odd
   L), which no graded size hits.
2. **ADOPTION BLOCK v2**: two new builders generate the vector-ready twiddle formats
   from the same exact (~1 ulp) `d1tw_cexp`:
   - `d1tw_stage_bc(n, r, tw)` — broadcast-pair layout, 3(r−1) doubles per p, for
     q-vectorized passes. For r = 4 this is byte-identical to the layout d1_pow2
     hand-rolled, so adopting the exact generator is a drop-in for them.
   - `d1tw_stage_s1bc(n, r, tw)` — lane-major zmm images (16(r−1) doubles per group of
     4 p, zero-padded tail) for the s == 1 first stage vectorized ACROSS p.
3. **First stage radix-4 at s = 1, vectorized across p** with the 4x4 complex-lane
   transpose (permutex2var + shuffle_f64x2), masked loads/stores for m % 4 tails —
   structure **borrowed from d1_pow2's `stage_s1`**, fed by `d1tw_stage_s1bc`. New
   factor schedule = their r1 finding: 4 first, then radix-4 at small s until the
   remaining pow2 bits divide by 3, then radix-8s, then 3s, then 5s (last stage keeps
   the largest q loop for the fused map). 32→[4,8], 64→[4,4,4], 128→[4,4,8],
   1024→[4,4,8,8], 4096→[4,4,4,8,8], 16384→[4,8,8,8,8], 60→[4,3,5].
4. **fft1d_chain runs batch-OUTER, steps-inner** so one transform's whole m-step chain
   stays cache-resident (~4L·16 B) instead of streaming the full B×L batch every step —
   **borrowed from d1_pow2's** "chain is separable per transform" observation. One loop
   swap; it is most of the batched-chained win below.
5. **Fast fused map** (the map was ~half of every chained step: vsqrtpd+vdivpd zmm are
   ~24+16 cycles and barely pipeline): rsqrt14 + 1 Newton + exact-residual FMA sqrt
   refinement, rcp14 + 2 Newton, then an exact-residual refinement of the final
   quotient — **d1_pow2's r1 map recipe**, which their record shows passes the chain
   gate precisely when the twiddles are exact (theirs needed long-double tables; mine
   are already ~1 ulp by construction). `n` clamped at 1e-300 so z = 0 cannot reach
   rsqrt(0) = inf. The exact vsqrtpd/vdivpd store is kept behind `-DD1TW_EXACTMAP` as
   the fallback if a scoring-node seed ever fails a gate.

### Measured (wallaby core 55, idle-verified, steady state; old = my r1 code, same
core, same flags — so the columns are a true A/B on identical silicon)

| cell | old r1 | new | | cell | old r1 | new |
|---|---|---|---|---|---|---|
| 32 B=1 m=1 | 0.054 | **0.018** | | 1024 B=1 m=1 | 4.03 | **1.18** |
| 32 B=512 m=1 | 0.055 | **0.019** | | 1024 B=512 m=1 | 3.94 | **1.58** |
| 32 B=1 chain | 0.122 | **0.063** | | 1024 B=1 chain | 5.84 | **2.07** |
| 32 B=512 chain | 0.122 | **0.063** | | 1024 B=512 chain | 5.96 | **2.06** |
| 60 B=1 m=1 | 0.135 | **0.042** | | 4096 B=1 m=1 | 14.8 | **6.67** |
| 60 B=512 m=1 | 0.134 | **0.046** | | 4096 B=256 m=1 | 15.7 | **8.18** |
| 60 B=1 chain | 0.261 | **0.115** | | 4096 B=1 chain | 23.0 | **10.25** |
| 60 B=512 chain | 0.261 | **0.116** | | 4096 B=256 chain | 23.7 | **10.6** |
| 64 B=1 m=1 | 0.110 | **0.041** | | 16384 B=1 m=1 | 73.5 | **31.2** |
| 64 B=512 m=1 | 0.111 | **0.043** | | 16384 B=64 m=1 | 75.5 | **35.8** |
| 64 B=1 chain | 0.239 | **0.109** | | 16384 B=1 chain | 105 | **46.7** |
| 64 B=512 chain | 0.241 | **0.110** | | 16384 B=64 chain | 108 | **47.9** |
| 128 B=1 m=1 | 0.326 | **0.073** | | 128 B=1 chain | 0.609 | **0.220** |
| 128 B=512 m=1 | 0.413 | **0.103** | | 128 B=512 chain | 0.614 | **0.221** |

1.9–4.5x per cell. Same-core library A/B (wallaby builds): ahead of fftw1d_measure at
most owned cells (e.g. 60 B=1: 0.042 vs 0.047 fftw / 0.050 MKL — a B=1 m=1 cell we may
actually win), behind wallaby MKL at pow2 (1024 B=1: 1.18 vs 0.83; 32 B=512: 0.019 vs
0.011) — MKL-on-SPR is stronger than the a80n0 baseline suggests; the monitor's Ice
Lake numbers arbitrate. Against d1_pow2's r1 wallaby table: roughly tied at 128/4096
B=1, ahead on the batched large-L m=1 cells (4096 B=256: 8.2 vs 10.3; 16384 B=64: 35.8
vs 43.8), behind at 32/64 B=1 (their in-register codelets) and slightly behind on
chained small L.

Correctness: single-call rel L2 1e-16..3.4e-16 at 14 sizes (B=3) + graded-size B=512
spot checks; all 14 graded chained cells PASS at full graded m, worst margin 15x
(1024 B=1 m=4000: 6.7e-12 vs 1e-10 — identical with the exact map, i.e. the residual
error is chain-chaos-dominated, not map-dominated). Output bitwise repeatable.

### What was tried / measured that did not pan out, with numbers

- **Exact vsqrtpd/vdivpd map**: correct and simple, but it WAS the chained regime:
  16384 B=1 chain 63.4 us (exact) vs 46.7 (Newton map); 32 B=1 chain 0.100 vs 0.063;
  4096 B=256 chain 15.7 vs 10.6. Kept only as the `-DD1TW_EXACTMAP` fallback.
- **Measurement trap (record for everyone timing on wallaby): the FIRST driver
  invocation after regenerating input files reads 1.3–1.9x slow** (4096 B=1 chain:
  20.6 us then 10.5, 10.3; 16384 B=1 chain: 88.6 then ~47). Cold page cache /
  frequency ramp; min-over-samples inside one invocation does NOT absorb it. Re-run
  the binary and keep the steady-state number, or the A/B lies to you.
- **Unpinned/wrong-core timing**: core 100 (previous round's habit) was intermittently
  busy and gave 0.044 for a cell that times 0.018 on a verified-idle core. Sample
  /proc/stat deltas and pick an idle core before believing anything.

### Borrowed (all named in-code too)

- d1_pow2 (r1): cmul broadcast-pair idiom, s==1 across-p first stage + 4x4 transpose,
  the pow2 stage schedule rule, per-transform chain blocking, and the
  rsqrt14/rcp14 + Newton + exact-residual map recipe. This entry is largely their
  structural findings re-expressed through my exact-table generators — that is what
  the library layer is for, flowing back as v2 formats they can now consume from
  `d1tw_stage_bc/_s1bc` instead of hand-rolling.
- d1_batchlane (r1): the "gcc auto-vectorizes at ymm only" diagnosis that motivated
  going full intrinsics, and masked-zeroing zmm tails instead of scalar remainders
  (used in `vfirst4`).

### Next round

1. **In-register codelets for L=32/64** (d1_pow2 has them: 0.012/0.029 vs my
   0.018/0.041) — or simply concede those B=1 cells to them and keep the layer clean.
2. **1024/4096 B=1 vs MKL**: still 1.2–1.4x behind wallaby MKL. Four-step 128x128 with
   an L1-blocked transpose (survey per-size map) is the untried structural move.
3. **Radix-16 stage** to cut 16384 to 4 passes ([4,16,16,16] or [16,16,16,4]) — needs
   a 15-twiddle bc block; `d1tw_stage_bc` already generates any radix.
4. If a scoring-node chain gate ever fails: rebuild with `-DD1TW_EXACTMAP` (costs
   ~35% of chained-cell time at large L, nothing at m=1).
