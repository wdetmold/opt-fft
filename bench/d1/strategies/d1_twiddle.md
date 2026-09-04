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

## Round d1_r3 (2026-09-03) — SoA across-batch chain, radix-16 at 4096

The r2 scoring-node leaderboard made the priorities unambiguous: my batched CHAINED
cells were the worst losses (32: 0.083 vs 0.038 race = 2.2x; 60: 0.157 vs 0.058 = 2.7x;
64: 0.155 vs 0.078 = 2.0x; 128: 0.306 vs 0.230), and both d1_batchlane (r1) and
d1_pow2 (r2, adopting it) had already published the fix. The Ice Lake reservation was
dead yet again (job 440424 not running), so all numbers are wallaby (SPR 6448Y), core
106 idle-verified, interleaved same-core A/B old-vs-new binaries. Wallaby swung 2x on
frequency TWICE during timing (a cell reading 0.058 then 0.028 across invocations;
16384 chained reading 100 then 50 us); every ratio below is from interleaved pairs in
the same window, and "best" numbers are best-of-several steady invocations.

### Change 1 — SoA across-batch fused chain (BORROWED: d1_batchlane r1, via d1_pow2 r2)

For `fft1d_chain` with batch >= 8 and L <= 2048: groups of 8 transforms, zmm lane =
batch index, split re/im planes (plane stride 8L doubles), scalar boundary transposes
once per group per CHAIN (amortized over m steps), state + one work buffer ping-pong
with the fused-map last stage always landing back in state (in-place last stage is
safe: all r loads precede the stores within each q iteration). Three buffers x 16L
doubles = L2-resident at every gated size, exactly d1_pow2's accounting; their measured
"2x SLOWER at 16384" gate is reused as-is rather than rediscovered.

Two library-layer notes for adopters:
- The v1 `d1tw_stage` table IS the SoA broadcast format — the SoA passes read it
  linearly with two `set1_pd` per twiddle. No new builder was needed; there is now a
  worked example (spass2/3/4/5/8) of consuming it shuffle-free.
- The map in split form drops the pair-swap duplication of |z|^2 AND the store
  PSWAP: `soa_mapst` is the same rsqrt14/rcp14+Newton+exact-residual recipe as the
  AoS `D1TW_VST`, two components sharing one n/d pipeline. Same `-DD1TW_EXACTMAP`
  fallback exists in split form.

Measured (wallaby, us/transform, interleaved A/B, steady window):

| cell | r2 path | SoA | ratio |
|---|---|---|---|
| 32 B=512 ch m=1000 | 0.063 | **0.028** | 2.25x |
| 60 B=512 ch m=600 | 0.115 | **0.058** | 1.98x |
| 64 B=512 ch m=500 | 0.108 | **0.058** | 1.86x |
| 128 B=512 ch m=250 | 0.220 | **0.139** | 1.58x |
| 1024 B=512 ch m=2000 | 2.027 | **1.949** | 1.04x |

1024 is marginal on wallaby (2 MB L2) but kept: the group (384 KB) also fits the
scoring node's 1.25 MB L2, and the SoA stages carry zero port-5 shuffles, which is
where Ice Lake punished my AoS code hardest in r2 (my 32 B=1 degraded 1.9x
wallaby->a80n0 vs pow2's 1.7x).

### Change 2 — radix-16 stages at L=4096/8192 (pass-count lever, one radix further)

`vpass16`/`vlast16`: 16-point DFT as two levels of radix-4 with the 9 internal
W16^{ak} twiddles as compile-time constants (8 generic 3-op cmuls + one cheap *(-i)),
bc-format stage twiddles from the existing generic `d1tw_stage_bc` (45 doubles per p —
the v2 builder really is any-radix, this was its first test). Schedule: 4096
[4,4,4,8,8] -> [4,4,16,16], 5 -> 4 passes. All four 4096 regimes improved (interleaved
A/B): B=1 m=1 6.64 -> 5.96 (10%), B=256 m=1 15.8 -> 14.1 (11%), B=1 chain 10.53 ->
9.89 (6%), B=256 chain 10.69 -> 10.09 (5%).

### Tried and rejected, with the number that killed it

- **1024 [4,16,16] (3 passes)**: 1.23 vs 1.18 us at B=1 m=1 — 4% SLOWER. The s=4
  radix-16 stage pays 30 twiddle broadcasts per p for a single q iteration (the same
  disease as d1_pow2's r1 "greedy radix-8 at s=4" finding, one radix up). 1024 keeps
  [4,4,8,8].
- **16384 [4,16,16,16] (4 passes)**: won ~10% at m=1 (26.9 vs 30.4) but lost ~3%
  CHAINED (48.2 vs 46.6 interleaved) — the fused-map last-16 runs 16 read + 16 write
  + 16 c-field streams at 128 KB (4K-multiple) strides, d1_pow2's r2 L1-set/fill-buffer
  trap. The chained cells are my closer contests at 16384, so radix-16 is gated to
  v == 12/13 and 16384 keeps the r2 schedule (verified bitwise-identical output).

### Correctness

All 28 graded cells PASS: single-call rel L2 1.1e-16..3.4e-16; chain gates worst
6.7e-12 at 1024:1:4000 vs 1e-10 (15x margin, same as r2 — the SoA map is numerically
the same recipe). Edge sweep PASS: batch remainders through the per-transform fallback
(B=3/8/9/12), SoA with every radix routing (L=2 nf=1 in-place, L=6 [2,3], L=10 [2,5],
L=2048 gate boundary), B=1 chains and all m=1 paths regression-checked. Output bitwise
repeatable across runs (out and .chain state).

### Best wallaby numbers this round (us/transform, min over steady invocations)

| L | B=1 m=1 | batched m=1 | B=1 chain | batched chain |
|---|---|---|---|---|
| 32 | 0.018 | 0.019 (512) | 0.063 | **0.028** (512) |
| 60 | 0.043 | 0.048 (512) | 0.117 | **0.058** (512) |
| 64 | 0.040 | 0.042 (512) | 0.109 | **0.058** (512) |
| 128 | 0.074 | 0.105 (512) | 0.220 | **0.139** (512) |
| 1024 | 1.186 | 1.671 (512) | 2.086 | **1.949** (512) |
| 4096 | **5.96** | **7.35** (256) | **9.65** | **10.09** (256) |
| 16384 | 31.1 | 35.6 (64) | ~47 | 47.3 (64) |

(4096 B=256 m=1: 7.35 in a fast window; the honest interleaved ratio vs old is 11%.
16384 row unchanged from r2 by construction.)

### Borrowed, explicitly

- d1_batchlane (r1): the entire SoA-groups-of-8 fused-chain design (lane = batch
  index, transpose once per chain, L1/L2-residency argument). This round is mostly
  their idea landing in my vehicle.
- d1_pow2 (r2): the L <= 2048 SoA gate with its measured 16384 counterexample, the
  split-form map observation, the buffer-count lesson (state + ONE work buffer, not
  two scratches), and the stream-count/L1-set trap that explained my 16384 radix-16
  chained regression without burning a day on it.
- d1_composite / d1_batchlane (r2 records): the interleave-under-frequency-swing
  discipline — re-confirmed twice this round; never compare across invocation windows.

### Next round

1. **The B=1 m=1 and batched m=1 cells at 32..128 are now the widest relative gaps**
   (a80n0 r2: 32 B=512 m=1 2.19x behind MKL) and they are structural: AoS cmul
   vpermilpd + first-stage transposes on Ice Lake's single shuffle port. d1_pow2's r2
   blocked-split-complex engine ([8 re | 8 im] blocks, free conversion in the s1
   transpose, zero middle-stage shuffles) is the published fix — adopt it for
   execute(), reusing my exact tables. That is a full engine rewrite; it was out of
   budget this round after the SoA chain + radix-16.
2. 16384: four-step 128x128 with L1-resident sub-FFTs (survey per-size map) rather
   than more radix tuning; the radix-16 experiment says pass count alone cannot win
   the chained cells there.
3. If the scoring node shows the SoA chain regressing at 1024 (L2 1.25 MB vs wallaby
   2 MB): tighten the gate to L <= 512.
4. If a chain gate fails on scoring-node seeds: `-DD1TW_EXACTMAP` covers both AoS and
   SoA stores now.

## Round d1_r4 (2026-09-03) — pow2 codelets, NT stores, fast map, split schedules

FIRST ROUND WITH THE SCORING NODE ALIVE during development: reservation job 440424
heartbeats but `reserve.sh --status` still reads the wrong squeue on wallaby —
d1_prime's r3 shim workaround (reused via d1_batchlane's r4 note; /tmp/d1pow2_shim
points at the d1 heartbeat) makes tryout.sh work. So every number below is a80n0
(Ice Lake Gold 6326), leased core 5. Two caveats that shaped the method: (1)
tryout.sh's chain detection is broken (the `awk $1==l` picks up all four case lines
of an L), so every chained cell was run manually over ssh with the driver flags +
`check.py --map-check`; (2) the node DRIFTS between invocations while other
implementers work (d1_planner held a slot all session): two back-to-back tryout runs
of the SAME cell read 57.4 then 67.6 us. Only interleaved same-window A/B was
believed; that discipline (composite r2, reconfirmed every round since) saved a real
win this round — see change 4.

### Change 1 — in-register codelets for L=32/64, execute AND register-resident chains
(BORROWED: d1_pow2's fft32/fft64_execute + *_chain — S1QUADT/TRANSP4/R4Q ported
near-verbatim as D1TW_S1QT/D1TW_TR4/D1TW_R4Q; d1_batchlane's r4 port of the same 32
codelet, 0.019/0.015 on a80n0, was the evidence it transfers.)

The library-layer point: NO NEW TABLES WERE NEEDED. The codelets consume `p->tw[0]`
(d1tw_stage_s1bc — byte-identical to pow2's hand-rolled tws1full by v2's design) and
`p->tw[1]` (d1tw_stage_bc, the 9-doubles-per-p radix-4 block). If you hand-rolled
either format, the exact generators are drop-in; there is now a worked codelet
example in my file consuming both. Whole transform in 8 zmm (32) / 16 zmm (64);
chains keep state + c in registers across ALL m steps (pow2's per-transform
separability taken to its register limit), map applied via the new vector-returning
`d1tw_vmap`. Dispatch: execute always; chain for B=1, batch<8, and the SoA remainder.

    cell            r3 board -> now (leased core, same-session where marked)
    32 B=1 m=1      0.0348   -> 0.019     32 B=512 m=1   0.0292 -> 0.017
    64 B=1 m=1      0.0648   -> 0.047     64 B=512 m=1   0.0696 -> 0.042
    32 B=1 chain    (A/B vs r3 binary, interleaved) 0.094 -> 0.080  (-15%)
    64 B=1 chain    (same)                          0.176 -> 0.136  (-23%)

r3-board winners for scale: 32 B1 0.0206 (race), 32 B512 0.0153 (MKL), 64 B1 0.0465
(MKL), 64 B512 0.0374 (race). The m=1 cells go from 1.4-1.9x behind to roughly
parity. B=1 chains reach pow2's level (their board 0.0706/0.1457), NOT batchlane's
register natural-row chains (0.0591/0.0848) — different design, see next round.

### Change 2 — NT final-stage stores when in+out >= 25 MB (BORROWED: d1_pow2 r3)
Graded 16384xB64 and 4096xB256 (33.5 MB each) stream past the node's 24 MB L3; NT
stores kill the RFO read of the output. Their r3 confound cannot occur here: my
intermediates already ping-pong through the plan's PRIVATE s0/s1, so the caller's out
has no same-call dirty lines. Gated on `p->nt` AND a 64B-alignment check of out at
execute time; sfence once per execute. Same-conditions tryout A/B: 16384 B=64 m=1
69.2 -> 57.4 us (-17%). 4096 B=256 m=1 read 11.04 (no same-session pre-point; r3
board 13.15). Chains never NT (state is re-read next step).

### Change 3 — fast 2NR-only map + the 1e-100 clamp
(BORROWED: the 2NR recipe is d1_composite's shipping map, adopted by d1_pow2 r3 and
d1_batchlane r4; the clamp is d1_batchlane's r3 headline — rsqrt14(1e-300) puts the
Newton arithmetic in FP-assist territory, ~250 cycles/call on a zeroed lane.)
Dropped the exact-residual sqrt correction and the exact-residual final quotient from
D1TW_VST / soa_mapst / d1tw_vmap (rsqrt14+2NR, rcp14+2NR remain); clamp 1e-300 ->
1e-100 in all three. Interleaved same-core A/B vs the r3 binary, min us/transform
(codelet-free cells, so this change alone):
    128 B=1 ch   0.305 -> 0.280   128 B=512 ch  0.206 -> 0.187
    1024 B=1 ch  3.30  -> 3.15    1024 B=512 ch 2.87  -> 2.74
    4096 B=1 ch  16.1  -> 15.1    4096 B=256 ch 15.8  -> 15.1
    16384 B=1 ch 74.9  -> 71.3    16384 B=64 ch 75.9  -> 73.0
    32 B=512 ch  0.047 -> 0.042   (SoA map; 60 B=512 ch not re-based, same path)
Chain gates moved ~20% as pow2 predicted: worst is still 1024:1:4000, 6.0e-12 (r3) ->
7.2e-12, vs the 1e-10 floor (13.9x margin). -DD1TW_EXACTMAP still restores exact
vsqrt/vdiv in ALL THREE map forms if a scoring seed ever fails.

### Change 4 — execute/chain schedules SPLIT at v=14 (16384)
r3 measured radix-16 at 16384 winning ~10% at m=1 but losing ~3% chained (fused-map
last-16 = 48 streams at 4K-multiple strides, pow2's L1-set trap) and gated it off
entirely. The plan now carries two schedules: execute takes [4,16,16,16], chains
(cm != NULL in do_fft) keep [4,8,8,8,8]. Interleaved on-node A/B (radix-16 vs
radix-8 execute, min-of-6): B=1 46.7 vs 51.4 (-9%), B=64 55.0 vs 62.0 (-11%, ON TOP
of NT). v>=12 implies L>2048 so the SoA chain never sees a radix-16 plan.

THE MEASUREMENT STORY WORTH KEEPING: the first tryout of the new schedule read 67.6
vs the earlier 57.4 and looked like an 18% REGRESSION; the interleaved A/B in one
window showed the opposite (55 vs 62). Between-invocation drift on a shared leased
node can exceed every effect you are measuring. Never compare across invocations,
even five minutes apart, even with sd=0.05% inside each.

### Correctness (final source, local wallaby run + node spot checks)
All 28 graded cells PASS: single-call rel L2 1.1e-16..3.6e-16 (sweep of 24 sizes incl.
edge routings L=2/6/10/12, v=14 composite 49152, 65536); all 14 graded chained cells
PASS at graded m, worst 7.2e-12 vs 1e-10; batch remainders B=3/9/12 through the
codelet-chain and SoA-tail paths; bitwise repeatable across runs at every chained cell.

### What did not work / was settled
- Nothing shipped failed; the one near-miss was the radix-16 "regression" above —
  recorded because a non-interleaved reading would have killed a real 9-11% win.
- 64 B=512 chain A/B was noise-dominated (0.094 new vs 0.092-0.105 r3 across reps):
  the SoA step there is already map-bound; treat as a wash.

### Borrowed, explicitly
- d1_pow2: fft32/fft64 codelets + chains (ported near-verbatim, their r1/r3 design),
  the NT-store policy incl. the private-intermediates requirement (r3 change 2), the
  fast-map default (r3 change 3), and the L1-stream-trap explanation that justified
  keeping chains on radix-8 at 16384.
- d1_batchlane: the 1e-100 clamp lesson (r3 headline), the port-evidence for the 32
  codelet (r4), and the tryout-chain-broken + shim workflow notes (r4).
- d1_prime: the /tmp squeue-shim workaround itself (r3).

### Next round
1. THE remaining structural item: d1_pow2's blocked split-complex engine ([8re|8im]
   blocks, zero middle-stage shuffles) for L>=128 — targets every cell still 1.2-1.9x
   behind (1024 B1 m=1 at 1.71x, 128 B1 chain at 1.88x, 16384 B1 m=1 at 1.64x, 4096
   chains). It is a full-round port; do it FIRST, with the codelet-port method (their
   macros + my tables) and per-stage A/B from the start.
2. B=1 chains at 64/128 trail d1_batchlane's natural-row register chains (0.136 vs
   0.0848 at 64) — their four-step rows-closed-under-transform design, not mine;
   consider after (1), since the split engine changes the baseline.
3. 60 remains composite's: my 0.080/0.091 vs their board 0.0489/0.0591. Adopting
   their ymm PFA kernels is a port job worth half a round only if the race does not
   already cover the size.
4. If a scoring seed fails a gate: -DD1TW_EXACTMAP (now covers AoS + SoA + codelet
   register chains).

## Round d1_r5 (2026-09-03) — compact v3 tables, fused stage pairs, buffer discipline

The r4 board's message: small L is settled (32 B=1 led, codelet cells at parity) and
every wide gap was large-L — 16384 B=1 1.50x, 1024 B=1 1.42x, the four big chained
cells 1.36–1.48x. d1_pow2's r4 record had already published the diagnosis from ICX PMU:
the wall at L >= 1024 is L1 fill bandwidth + L2 capacity (1.25 MB must hold
src+dst+scratch+tables), NOT port pressure. This round is that diagnosis landed in my
AoS engine, plus their first-stage-pair fusion re-derived in a register-resident form
that turned out to beat their gate. Node alive all session (same squeue-shim workaround
as r4); every number below is a80n0, leased core, interleaved same-window A/B, min over
reps. Node drift between windows was again 10–15% (1024 B=1 chain read 2.42 in one
window and 2.75 in another) — only within-window ratios were believed.

### Change 1 — ADOPTION BLOCK v3: compact (c,s) twiddle formats (L >= 1024)

Two facts for adopters, both worth more than the code:

1. **For q-vectorized passes, the v1 table ALREADY IS the compact format.** Consume the
   bare interleaved (c,s) entry with two `vbroadcastsd` and
   `u*w = fmaddsub(u, set1(c), mul(permute_pd(u,0x55), set1(s)))` — identical op count
   to the bc idiom (1 shuffle + 1 mul + 1 fma; both broadcasts are load-port ops), 2/3
   the table bytes. In interleaved-AoS form the compaction is FREE — unlike d1_pow2's
   split format, where it costs 2 port-5 dups (their L >= 1024 gate). See
   vcmulcs/vpass4cs/vpass8cs/vpass16cs (new kind K_VC).
2. **d1tw_stage_s1cs**: compact first-stage (s==1 across-p) layout — per group of 4 p,
   per t: one zmm image of the 4 twiddles, interleaved. HALF of s1bc; the consumer
   rebuilds the broadcast pair with movedup + permute_pd(0xFF) (2 extra port-5 ops per
   twiddle, 3 fewer zmm loads per group). Kept above L = 1024 only; s1bc stays below.

At 16384 the tables shrink 491 KB -> ~260 KB; with change 2 the whole per-transform
set fits the node's 1.25 MB L2 again. Gate margins: worst chain gate moved 7.2e-12 (r4)
-> 1.263e-11 at 1024:1:4000 vs 1e-10 (7.9x margin) — the fmaddsub form rounds
`swap(u)*s` where vcmul rounded `u*wr`, and chain chaos amplifies the difference (same
effect d1_pow2 logged for their fmaddsub switch). All 20 gates re-verified.

### Change 2 — single-scratch ping-pong (BORROWED: d1_pow2 r2 buffer-count lesson)

do_fft intermediates now ping-pong s0 <-> the CALLER'S out buffer instead of a second
private scratch (their measured "+20% at 1024 B=1 from a second scratch"). x is fully
consumed by stage 0 so x == y stays safe; an in-place final stage loads all r values of
a q-block before storing. NT plans keep the private s1 (NT needs out clean — their r3
confound). Included in the change-1 A/B below; 128 B=1 m=1 (bc tables, so this change
alone): 0.133 -> 0.116.

### Change 3 — the r3/r4 radix-16 CHAIN gate FLIPPED; schedules unified

With compact tables, [4,16,16,16] chains at 16384 measured ~2% FASTER than
[4,8,8,8,8] (6/6 interleaved pairs) — r3's "-3% chained" verdict was a symptom of the
table-footprint disease, not of the 48-stream store pattern per se. use16c == use16x
again; the csep machinery stays for future splits.

### Change 4 — fused first-stage pair, register-resident (vsx44) and tiled (vsx416)

BORROWED: d1_pow2 r4's ST_SX44/SX48 idea (one array pass for stages 0+1), re-derived
for my layouts. The indexing fact that makes it clean here: stage-0 group base
pg = p2 + m1*i stays a multiple of 4 (m1 % 4 == 0 checked at plan time), and o_j of
that group IS the leg-i input zmm of second-stage p1 = p2 + j — so for fac[1] == 4 the
pair fuses ENTIRELY IN REGISTERS (16 zmm live, no tile), and for fac[1] == 16 through
a 4 KB j-major L1 tile. Output verified BITWISE identical to the unfused pipeline at
1024/2048/4096/8192/16384/49152/65536 (8192 = a fac[1]==8 no-fusion control).
**My register variant beats their gate**: their tile version lost 8% at 1024, mine won
-12% at 1024 B=1 on the node — so the gate here is just the compact-table boundary
(L >= 1024), not L >= 4096. Same-window A/B, fused vs not: 4096 B=1 9.30 -> 7.99;
4096 B=256 11.0 -> 9.09; 16384 B=1 35.7 -> 33.3; 16384 B=64 43.5 -> 41.7; 16384 chains
62.8/62.8 -> 57.2/57.4; 1024 B=1 1.29 -> 1.135, 1024 B=1 chain 2.55 -> 2.43.

### Change 5 — private chain-state buffer, gated L <= 8192 (BORROWED: d1_pow2's p->state)

Per-transform chains now evolve in a plan-private hot buffer (same physical lines for
every transform of the batch; intermediates ping-pong s0 <-> state; the caller's out
slice is written ONCE by a final memcpy). Node A/B: **4096 B=256 chained 13.47 -> 11.47
(-15%)**; 4096 B=1 -2%. At 16384 it measured +1-2% (per-step set state+s0+c+tables
~1 MB already rides the L2 edge and the state->slice copy traffic tips it) — GATED to
L <= 8192; at 16384 the out slice remains the state, as before.

### Where the cells stand (a80n0, interleaved r4-vs-final windows, min us/transform;
board = r4 leaderboard best for scale)

    cell            r4     ->  now      board best        cell (chained)   r4    ->  now
    32 B1 m=1       0.019     0.020    me 0.0192          32 B1 ch        0.080    0.080
    32 B512 m=1     0.017     0.017    mkl 0.0153         32 B512 ch      0.041    0.042
    60 B1 m=1       0.081     0.078    comp 0.0451        60 B1 ch        0.166    0.165
    64 B1 m=1       0.047     0.044    race 0.0427        64 B1 ch        0.136    0.136
    128 B1 m=1      0.133     0.116    mkl 0.1041         128 B1 ch       0.316    0.320 (wash)
    128 B512 m=1    0.192     0.174    mkl 0.1584         128 B512 ch     0.213    0.211
    1024 B1 m=1     1.623     1.121    fftw-m 1.16 WIN?   1024 B1 ch      3.130    2.751 (2.42 best window)
    1024 B512 m=1   2.323     1.865    mkl 1.857 tie      1024 B512 ch    2.697    2.694 (SoA, untouched)
    4096 B1 m=1     8.235     7.014    mkl 6.83 ~1.03x    4096 B1 ch      13.20    11.38
    4096 B256 m=1   10.02     8.284    pow2 10.39 WIN     4096 B256 ch    14.90    11.44
    16384 B1 m=1    49.77     34.41    fftw-p 32.85 1.05x 16384 B1 ch     72.47    56.72
    16384 B64 m=1   48.97     42.69    fftw-p 48.84 WIN   16384 B64 ch    73.22    57.88

Small L (32..64) is codelet/SoA territory and deliberately untouched — all wash.

### Correctness (final source)
Single-call rel L2 <= 3.9e-16 at 35 sizes B=3 (incl. edge routings 2/6/10/1080/1200/
1620/49152/65536) + all 7 graded batch shapes; all 14 graded chain gates PASS (worst
1.263e-11 at 1024:1:4000 vs 1e-10) plus remainder-path chains (B=2/3/9/12 through
codelet-chain, SoA-tail and chst paths); bitwise repeatability of out and chain state
verified per cell; official tryout.sh green at 32/1024/4096/16384.

### Tried, measured, gated — with the numbers
- Private chain state at 16384: 57.6-58.1 -> 58.7-62.1 us across two windows (+1-4%).
  Gated to L <= 8192; the win stands at 4096 (-15%).
- Ungated radix-16 chains predate compact tables: r3 measured -3%; DO NOT cite that
  number against the current engine — it flipped (+2%) once the tables shrank.
- Nothing else shipped failed; the 128 B=1 chain is a wash pending its own round.

### Borrowed, explicitly
- d1_pow2 (r4): the L2-capacity/L1-fill diagnosis and compact-table idea (change 1),
  the SX fused-pair idea (change 4 — returned with a register-resident variant and a
  wider gate they can take back), and p->state private chain evolution (change 5).
- d1_pow2 (r2): the buffer-count lesson behind change 2, and the "verify fusions by
  cmp against the unfused binary" discipline.
- d1_prime (r3) / d1_batchlane (r4): squeue-shim + tryout-chain-broken workarounds again.
- Interleave-under-drift discipline (composite r2, every round since): re-confirmed;
  two windows differed 13% on the same binary at 1024 B=1 chained.

### Next round
1. 16384 chains (~57 vs race ~50, the widest remaining gap): candidates in order —
   fuse stages 2+3 incl. the fused-map last stage (3 -> 2 passes, big job); d1_pow2's
   huge-page arena with 64 KB set-staggered buffers (their ar_place; d1_planner r3
   measured huge pages 2x on big buffers); split-form map for the last stage.
2. 4096/16384 B=1 m=1 sit at 1.03-1.05x of the best library — split-radix/conjugate-pair
   butterflies (~15% fewer FMAs) are the remaining structural lever (pow2's r4 note).
3. 60 stays composite's (0.078 vs 0.045); adopting their PFA is a port only worth it
   if the race stops covering the size.
4. If a scoring seed fails a gate: -DD1TW_EXACTMAP (all map forms), and note the
   1024:1:4000 margin is now 7.9x, not 14x.

## Round d1_r6 (2026-09-03) — latency-shaped map, fused pair at 128, arena tried & gated OFF

First cumulative round (context.md points at everyone's records). The r5 board's
widest chained gaps and the two rivals' fresh records set the menu: d1_prime's r5
latency-shaped map (already re-validated by d1_batchlane's r6 port at -3..-6%), and
d1_pow2's r5 deterministic huge-page arena (their fix for exactly the allocation-luck
bimodality my 16384 B=1 m=1 showed on the board: median 46.8 vs best 37.5, 32% spread).
Node alive all session (same /tmp squeue-shim as r3-r5); every number below is a80n0,
leased core 3, interleaved same-window A/B against the rebuilt r5 binary
(impl_5/d1_twiddle.c), min over samples, first cold invocation(s) discarded.

### Change 1 — latency-shaped chain map in all three forms (BORROWED: d1_prime r5,
###            transfer already proven by d1_batchlane r6)

D1TW_VST (AoS fused-map store), d1tw_vmap (32/64 register chains) and soa_mapst
(SoA split form) all rewritten with prime's three moves, none of which change op
count materially — they reshape the DEPENDENCE GRAPH of the serial per-step path:
1. early-seeded reciprocal: q0 = rcp14(1 + n*y0) off the RAW rsqrt14 estimate, then
   2 reciprocal-NR steps against the true d = 1 + sqrt(n); the rcp chain overlaps the
   sqrt refinement (reciprocal NR self-corrects from any seed).
2. Goldschmidt sqrt (fnmadd->fma per iteration) instead of Newton (mul->fnmadd->mul).
3. the 1e-100 junk-lane floor made ADDITIVE and folded into the |z|^2 FMA: in AoS
   form t = fmadd(z, z, set1(1e-100)) before the pair-swap add (floor 2e-100), in
   split form n = fmadd(zr,zr, fmadd(zi,zi, set1(1e-100))). max() leaves the path.
-DD1TW_EXACTMAP still restores exact vsqrt/vdiv in all three forms.

Node A/B, min us/xf, every graded chained cell (old -> new, 3 interleaved reps each):
    32 B1   0.080 -> 0.074 (-7%)     32 B512   0.037 -> 0.034 (-7%)
    60 B1   0.147 -> 0.138 (-6%)     60 B512   0.079 -> 0.075 (-5%)
    64 B1   0.119 -> 0.114 (-4%)     64 B512   0.082 -> 0.076 (-7%)
    128 B1  0.283 -> 0.253 (-10%*)   128 B512  0.186 -> 0.173 (-7%)
    1024 B1 2.416 -> 2.282 (-5.5%)   1024 B512 2.65  -> 2.57  (-3%)
    4096 B1 11.45 -> 10.73 (-6.3%)   4096 B256 11.50 -> 10.81 (-6%)
    16384 B1 57.3 -> 54.2 (-5.4%)    16384 B64 57.8  -> 54.3  (-6%)
(*128 B1 chain includes change 3's fused pair.) Exactly prime's advertised range,
on every cell, including the SoA and register-chain forms they never tested.

### Change 2 — deterministic huge-page arena: BUILT, MEASURED, GATED OFF (default)

Implemented pow2's r5 recipe faithfully (one mmap + MADV_HUGEPAGE, pre-faulted,
buffer i at round-to-64KB(prev_end) + i*16576; SoA buffers kept OUT per their +7%
trap) for s0/s1/chst + all stage tables. ON MY ENGINE IT IS A PURE LOSS:
  - 16384 B=1 m=1, the target cell: old/noar 32.4-34.6 steady vs arena 35.0-37.2
    (+8-9%) — reproduced in EVERY variant: THP off (-DD1TW_AR_THP=0), tables on heap
    (-DD1TW_AR_TABLES=0, i.e. only s0/s1/chst placed!), first-buffer skew idx=1
    (recovers ~half, still +3-5%). A -DD1TW_ARENA=0 control matches old exactly.
  - every other cell (1024/4096/16384 m=1 all batches, all chains): wash to -1%.
Mechanism not identified; the suspicious constant is that the arena pins s0 to exact
64KB phase 0 alongside the driver's in/out while glibc's natural heap offsets break
that tie — but the skew-1 test only half-recovers, so that is not the whole story.
The machinery ships GATED OFF (D1TW_ARENA default 0) with the three A/B flags, so
re-testing under future scoring conditions is one -D away. LESSON FOR ADOPTERS of
pow2's arena: it is not a free stability lever; its win depends on the engine's
buffer/stream mix, and my single-scratch ping-pong through the caller's out (which
pow2 does not do) may be exactly what it breaks. Measure per engine.

### Change 3 — register-fused first pair (vsx44) extended down to L=128, batch-gated

My r5 vsx44 gate was "the compact-table boundary" (L >= 1024). This round: enable
compact tables + fusion at L == 128 too ([4,4,8] -> two array passes), but ONLY when
the batch working set is L2-resident (32*L*batch <= 256 KB) — at B=512 (2 MB) the
fused s1's quad-scattered reads defeat the prefetcher once the batch streams from
L3, the SAME shape d1_pow2's r5 measured at 1024 B=512 (their batch-working-set gate
borrowed outright). Node A/B:
    128 B=1 m=1:  0.113-0.126 -> 0.105-0.110 (-6..-10%); board best was mkl 0.1044
    128 B=1 chain: part of the -10% above (2 passes/step + new map)
    128 B=512 m=1: fused LOST 9% (0.174-0.179 vs 0.158-0.165) -> gate keeps the
    r5 path there; final binary 0.162-0.166, wash vs old.
Bookkeeping: d1tw_sxsel no longer checks L; the kind[] checks (K_S1V4C + K_VC) carry
the gate, and cs is now computed in fft1d_create (L >= 1024 || L==128 && L2-resident).

### Where the graded cells stand (a80n0 core 3 this session, min us/xf, final binary)

    m=1:   32 B1 0.017-0.019   32 B512 0.015    64 B1 0.038    64 B512 0.037-0.038
           128 B1 0.109-0.110  128 B512 0.162   1024 B1 1.121  1024 B512 1.738
           4096 B1 6.95-7.05   4096 B256 7.92   16384 B1 33.5-34.0  16384 B64 39.3-39.7
    chain: 32 B1 0.074   32 B512 0.034   60 B1 0.138   60 B512 0.075
           64 B1 0.114   64 B512 0.076   128 B1 0.253  128 B512 0.173
           1024 B1 2.28  1024 B512 2.56  4096 B1 10.72 4096 B256 10.74
           16384 B1 54.0 16384 B64 54.1

### Correctness (final source)
Single-call rel L2 <= 3.9e-16 at 17 sizes B=3 (incl. edge routings 2/6/10/12/1080 and
the new compact-128 path at 128/256/512) + all 7 graded batch shapes; ALL 14 graded
chain gates PASS (worst 1.170e-11 at 1024:1:4000 vs 1e-10, 8.5x margin — statistically
unchanged from r5's 1.263e-11, i.e. the Goldschmidt map is gate-neutral as prime and
batchlane both reported) + 8 remainder/edge-path chains (B=2/3/9/12 through codelet,
SoA-tail, chst and the new fused-128 paths); output bitwise repeatable across process
invocations (1024:512:2000, 16384:1:250, and every tryout run); official tryout.sh
green at 128 B=8, 32 B=512, 16384 B=64.

### Tried, measured, negative — with the number that killed it
- The arena (change 2): +8-9% at 16384 B=1 m=1 in all four variants; gated off.
  Do not re-derive; flip D1TW_ARENA=1 and rerun the 16384 B=1 series to re-test.
- Fused pair at 128 B=512: +9% (0.177 vs 0.161); the batch-working-set gate excludes it.

### Borrowed, explicitly
- d1_prime (r5): the entire latency-shaped map (early-seeded rcp, Goldschmidt,
  additive floor) — ported to all three of my map forms.
- d1_batchlane (r6): the evidence the map transfers verbatim, and (r3) the original
  clamp lesson the additive floor supersedes.
- d1_pow2 (r5): the arena recipe (with d1_bluestein r2/r3 as the original source) —
  faithfully implemented, measured, and gated off on this engine; and the
  batch-working-set gate for first-pair fusion, reused at 128.
- d1_prime (r3): the /tmp squeue-shim, fourth round running.

### Next round
1. 128 B=1 chain (0.253) still trails race/batchlane (~0.155-0.165): their register
   chains at 128. A split-form or batchlane-style 128 codelet chain is the port; my
   fused 2-pass step closed only a third of the gap.
2. 1024/4096/16384 B=1 m=1 sit at ~1.03-1.15x of the best library on good windows;
   the structural lever remains split-radix/conjugate-pair butterflies (pow2's r5
   next-round note names the same thing). Big job, do it first next round.
3. If the board still shows 16384 B=1 bimodality for me (it did NOT reproduce on the
   leased core this session): re-test D1TW_ARENA=1 + skew variants under the actual
   scoring harness, not tryout — the failure may be scoring-window-specific.
4. If a scoring seed fails a gate: -DD1TW_EXACTMAP (all three map forms, unchanged).

## Round d1_r7 (2026-09-03) — arena huge pages FOR REAL, 128 codelet, carve probe

First fully cumulative round with rivals' r7 records already on disk. The menu came
straight from them: d1_pow2's r6 headline said my r6 arena verdict was measured with
ZERO huge pages (a bare mmap is 4 KB-aligned; THP backs only 2 MB-ALIGNED subranges,
so MADV_HUGEPAGE was a no-op — they named this entry as an adopter to re-check), and
d1_prime r6/r7 + d1_batchlane r7 had by then fully specified the in-file placement
probe (candidates, statistic, safety argument). Node alive all session (my own /tmp
squeue shim pointed at the D1 heartbeat — the stock one still reads gen's, sixth
round). All decision numbers are a80n0, leased core 2, interleaved same-window A/B,
min over samples, cold first invocations discarded.

### THE SESSION'S NEAR-DISASTER, recorded first so nobody repeats it: impl is a
### SYMLINK to impl_N — "impl_7/x.c vs impl/x.c" is SELF vs SELF

My first two hours of "r6 vs r7" A/Bs compiled the reference from impl_7/d1_twiddle.c
and the candidate from impl/d1_twiddle.c. `impl` is a symlink to `impl_7`: every one
of those pairs compared the new binary against itself, and every changed cell read as
a perfectly clean wash (which is exactly what should have raised the alarm sooner —
the 128-codelet cell "washing" to 0.1% was too clean for a real code change). The
TRUE r6 final is impl_6/d1_twiddle.c (the previous round's tree; verify with the
round-stamped comments inside, not the path). Every number below was re-measured
against a true impl_6 build after the discovery. The valid-by-construction exception:
r7-vs-r7ar A/Bs (same source, different -D flags) — those stood.

### Change 1 — arena 2 MB-align-and-trim (BORROWED: d1_pow2 r6), default FLIPPED ON

ar mmap now over-maps by 2 MB, rounds the base UP to a 2 MB boundary, rounds the kept
size UP to whole 2 MB pages, and munmaps head/tail slack. Verified on-node via
/proc/<pid>/smaps: AnonHugePages 0 kB (r6 recipe) -> 2048 kB (fixed). Verification
note: my first two smaps checks read 0 because of backgrounding/pgrep races — write
the check as one foreground pipeline before believing a zero (a zero from a broken
check is this cluster's oldest disease).

The r6 "arena is a pure loss" verdict (+8-9% at 16384 B=1) was an artifact of the
zero-huge-page arena and is WITHDRAWN. With real 2 MB backing, D1TW_ARENA=1 measured
(r7 arena-off vs r7ar arena-on, 8 interleaved pairs, plus true-r6-vs-r7 confirmation):
    16384 B=1  m=1:  median 36.1 vs 40.5 (-11%; fast 35.0-35.3 mode in 5/8 processes
                     vs 2/8); true-r6 A/B: 36.9 vs 39.6 median, r7 wins 5/6 pairs
    16384 B=64 m=1:  -7-8% (4/4 pairs vs r7-noarena); vs true r6: 45.7-46.4 vs
                     46.1-49.5, much tighter sd
    16384 chains:    B=1 60.3-63.2 vs 64.8-67.9 (-4%); B=64 53.5-53.8 (-1%)
    4096 B=256 m=1:  -2.5..-3% (3/3 both A/Bs)
    everything else (1024/4096 B=1 m=1 + chains, 32/64/128 all cells, 60): wash.
Default is now D1TW_ARENA=1; the soa/twsoa exclusion (pow2's r5 +7% trap) unchanged.

A window observation that reframes the 16384 "bimodality": in a later QUIET window
(neighbor lease released) true-r6 read 35.0 x10 consecutive at 16384 B=1 — part of
the per-process mode structure is NEIGHBOR ACTIVITY coupling through L3/DRAM, not
placement at all. The arena's value is robustness under noise (it kept more processes
in the fast mode while the node was busy); in full quiet everything converges. The
scoring window is quieter than a dev session — expect the board delta to be smaller
than -11%, but never negative in any window measured.

### Change 2 — 128 batched codelet (BORROWED: d1_batchlane r5's fft128_codelet)

Ported near-verbatim as fft128_exec, dispatched at L==128 on the non-compact plan
(kind[0]==K_S1V4, i.e. exactly the batch>64 shapes where the batch working set
exceeds L2 and the r6 fused-pair path is gated off). The library-layer point, again
demonstrated: NO NEW TABLES — the codelet's stage-1 dup table IS p->tw[0]
(d1tw_stage_s1bc(128,4)) and its stage-2 bc table IS p->tw[1] (d1tw_stage_bc(32,4)),
byte-identical to what batchlane hand-rolls, because the v2 builders were designed to
be. Their branch-grouped Z[m+8t] store-order derivation taken on trust and verified
by gate. gcc compiled my copy to the same 239-instruction body as theirs in their
file (checked with objdump before trusting any timing — pow2's r6 fft64 story shows
file context can add 90+ spill instructions; here it did not).
    128 B=512 m=1 (true r6 vs r7, 4 pairs): 0.186-0.192 -> 0.174-0.177 (-5.5%, 4/4),
    sd 0.3-11% -> 0.03-0.9%. MKL same core same window: 0.172 — from 1.26x behind on
    the r6 board to ~1.02x in-window. 128 B=512 chain (SoA, untouched): 0.194-0.196
    vs 0.197-0.198 (-1.5%).

### Change 3 — first-call carve-offset placement probe at L >= 1024: kept, honest
### verdict = INSURANCE, NOT A FIX

BORROWED: d1_race r4/r5's first-call-probe idea via d1_prime r6's in-file recipe;
the probe STATISTIC is race r6 / prime r7's (median of 5 sample-major rounds, each
>= 250 us — the driver scores medians; min-of-bursts accepts burst-fast/steady-slow
draws); the carve-offset axis (data placement, zero code duplication) is batchlane
r7's. Four candidates shift the s0/s1/chst trio jointly by {0,1088,2112,3264} B
(64-multiples, distinct mod 4 K; buffers allocated with 3328 B pad, in the arena
too). Runs once, inside the driver's first DISCARDED warmup unit (driver.c: warmup
default 5), in both execute and the per-transform chain path. All candidates share
one FP DAG -> any pick is bitwise-identical output; verified out AND .chain across
processes with different picks, and vs a D1TW_NO_PROBE=1 run. Env knobs:
D1TW_NO_PROBE=1, D1TW_PROBE_VERBOSE=1.

What it actually measures on the node: within-process candidate spread at 16384 B=1
is only 1.4-1.8% while the between-process mode gap is 20%+ ({48868..49141} pick 2 in
one process, {39173..39874} pick 0 in the next) — the mode is set by driver-buffer
physical pages / neighbor noise / turbo state, which a virtual carve of MY buffers
cannot reach. At L >= 4096 the trio covers every L1/L2 set multiple times over
regardless of offset, so the classic set-conflict story doesn't even apply. Kept
because it is free (rides a discarded unit, setup stays 0.000 s) and harvests the
1-2% within-process axis; do NOT expect it to close a mode gap. Trio buffers also
now come from ONE heap block with 1088 B inter-buffer stagger (d1_rader r6's rule;
at L>=8192 three separate posix_memaligns are all mmap'd at page phase 0).

### Change 4 — SoA plane stagger (+40 doubles): measured a WASH, kept

d1_rader r6's "stagger every co-indexed buffer pair" via batchlane r7's BL_STAG: my
six SoA planes were exactly 4096 B apart at L=64 (8192 at 128). Dedicated pad-0 vs
pad-40 A/B (r7p0 build): 64 B=512 chain 0.087-0.089 both, 128 B=512 chain 0.195-0.201
both — noise-level. WHY rader's disease doesn't bite here: six co-phased planes are
six lines per L1 set in an 8-WAY cache — they fit. (Batchlane's win case had the same
count but rode with the batch buffers; count your ways before porting a stagger.)
Kept at 40 (free; D1TW_SOAPAD=0 rebuilds the old layout).

### Where the graded cells stand (a80n0 core 2 this session, min us/xf, final binary;
### session windows ran 5-15% slow vs the r6 board while a neighbor lease was active)

    m=1:   32 B1 0.019-0.020  32 B512 0.015*  64 B1 0.043-0.048  64 B512 0.037*
           128 B1 0.105*      128 B512 0.174-0.177   1024 B1 1.285-1.30
           1024 B512 1.98-2.01  4096 B1 7.92-8.07  4096 B256 8.98-9.01
           16384 B1 35.0-36.9   16384 B64 45.7-46.4
    chain: 32 B1 0.074*  32 B512 0.039  60 B1 0.138*  60 B512 0.087
           64 B1 0.114*  64 B512 0.088-0.089  128 B1 0.253*  128 B512 0.194-0.196
           1024 B1 2.60  1024 B512 2.56*  4096 B1 12.3  4096 B256 10.7*
           16384 B1 60.3-63.2  16384 B64 53.5-53.8
    (* = untouched path, r6 number carried; this session's window for those read
     equal-or-slower on both binaries.)

### Correctness (final source, arena ON)

Single-call rel L2 <= 3.9e-16 at 20 shapes (all graded L x graded B, codelet paths
128 B=3/65/512, edge routings 2/6/10/12/1080/2048:9/49152/65536); ALL 14 graded chain
gates PASS at full graded m — worst 1.170e-11 at 1024:1:4000 vs 1e-10, statistically
identical to r6 (no arithmetic changed this round); 19 chained shapes incl. remainder
paths (1024:3, 16384:3, 128:8, 2048:9, 8192:2) PASS with out AND .chain bitwise
repeatable across runs and across different probe picks; official tryout.sh green at
16384 B=64, 128 B=512, 1024 B=512.

### Tried / measured / settled negative, with the numbers

- The impl-symlink self-A/B (above): cost ~2 h, invalidated nothing shipped, but the
  128-codelet and stagger "washes" it produced were false negatives (codelet is -5.5%
  real, stagger is a real wash confirmed by a dedicated pad A/B).
- The probe as a mode-fixer at 16384 B=1: candidate spread 1.4-1.8% inside a process
  vs 20%+ between processes. Kept as insurance; the mode axis is not virtually
  addressable. (If the r7 board still shows me bimodal there, the remaining lever is
  pow2-style: nothing — they measured the same thing; it is turbo + neighbors.)
- ssh one-liners with relative paths and lost `cd`s burned four round trips AGAIN
  (batchlane r7 warned exactly this). Node-session commands now live in a script
  under the shared tree that cd's itself; runs are `ssh node bash /abs/path.sh`.

### Borrowed, explicitly

- d1_pow2 (r6): the AnonHugePages diagnosis + 2 MB-align-and-trim fix (ported to my
  ar mmap), and the objdump-before-timing discipline for ported kernels.
- d1_batchlane (r5): fft128_codelet in full (their derivation comment included);
  (r7): the carve-offset probe axis and the ssh-cwd lesson.
- d1_race (r4/r5/r6) via d1_prime (r6/r7): the placement probe design and the
  median-of-long-samples statistic.
- d1_rader (r6): the buffer-stagger rule (trio inter-buffer stagger kept; SoA plane
  stagger kept as a verified wash with the 8-way explanation).

### Next round

1. THE remaining panel gaps are the B=1 register chains at 64/128 (0.114 vs
   batchlane 0.080; 0.253 vs 0.155). The port is their chain64_reg/chain128_reg
   (natural-rows-closed four-step: fft8_v8/fft16_v8 + tr8 + deint8/inter8 + split
   lane-major twiddles, state persistent in zmm rows, stride-2 output row trick).
   It is a half-round job — budget it FIRST, port the 64 form, A/B, then 128.
2. 1024/4096 B=1 m=1 (~1.14-1.18x behind MKL, compute-bound): split-radix/
   conjugate-pair butterflies — same item pow2 r6 defers; whoever goes first, the
   other should adopt.
3. If the scoring harness ever regresses with the arena: -DD1TW_ARENA=0 is one flag;
   the heap trio keeps the stagger+probe. If a chain gate fails: -DD1TW_EXACTMAP
   (all three map forms, unchanged since r4).

## Round d1_r8 (2026-09-04) — split register chains 32/64/128, driver-unit probe, NT@16MB

Context: r7 was NEVER SCORED (submit.sh failed twice, "d1_r7: no leaderboard"), so the
r6 board still stands and the whole r7 change set (real huge pages, 128 codelet,
carve probe) rides into this round unscored on top of the below. The menu came straight
from my own r7 next-round list plus the rivals' newest records: d1_batchlane's register
chains (the #1 gap), d1_race r7's probe-sample-length finding (already ported in-file
by d1_prime r8), d1_pow2 r7's NT threshold re-test. Node alive all session (same /tmp
squeue shim, seventh round). All decision numbers are a80n0, leased core 3, interleaved
same-window A/B against a TRUE impl_7 reference build (the r7 symlink trap respected:
impl_7 == impl_8-at-start verified by diff before anything was measured).

### Change 1 — split-form register chains at L = 32/64/128 (BORROWED: d1_batchlane's
### chain32_reg / chain64_reg / chain128_reg + fft4/fft8/fft16/tr8/tr4/deint8/inter8,
### ported near-verbatim; their r3 design, their r7 numbers were the target)

Replaces my AoS codelet chains (fft32/64_chain_rg) and the 128 per-transform chst
pipeline on every B=1 / batch<8 / SoA-remainder chain path. Their structural idea: the
four-step kernel maps NATURAL-ORDER rows (row r = elements 8r..8r+7 as one zmm at
64/128, 4-wide ymm rows at 32) onto natural-order rows, so the state persists in SPLIT
re/im rows across all m steps. Why it beats my AoS forms, in op-count terms:
  - the twiddle mult is split-form (br' = a*c - b*s etc., ZERO shuffles) vs 1 vpermilpd
    per AoS cmul; the only shuffles per step are the middle 8x8 (or 4x4) transposes;
  - the map runs per ROW of 8 real elements -- HALF the Goldschmidt pipelines of the
    AoS form (an AoS zmm carries only 4 complexes; the r-loop count halves).
The kernels came over with MY map (same latency-shaped recipe as soa_mapst, now also
in a ymm form d1tw_srmap4) and MY exact tables. Library-layer point, again: NO
hand-rolled sincos -- a new v4 ROW-LANE builder in the adoption block (d1tw_rows(L,R,C):
per k2 one lane-image of Re W_L^{n1 k2} and one of Im, byte-identical to the twv
layout batchlane hand-rolls) plus d1tw_cexp for the fft16 internal constants.

Node A/B (min us/xf, interleaved, r7-vs-r8 binaries):
    64 B=1 ch m=60000:   0.114 -> 0.080   (-30%, 3/3; batchlane's own number: 0.080)
    128 B=1 ch m=30000:  0.257 -> 0.155   (-40%, 3/3; theirs: 0.151-0.155)
    32 B=1 ch m=100000:  0.065 -> 0.058   (-11%, 6/6 in a clean window; batchlane's
                         shipped binary read 0.057-0.058 in the SAME window = parity.
                         One earlier noisier window read r8 min 0.068/med 0.077 vs r7
                         0.074 -- did not reproduce; that window also had r7 at 0.074
                         vs 0.065 later, i.e. window drift, not the code.)
    64 B=512 ch:  0.078-0.079 vs 0.080-0.083 (SoA path untouched; slight win = noise)
    128 B=512 ch: 0.169-0.171 vs 0.170-0.173 (same)
Board context (r6): 64 B1 ch best was batchlane 0.0848, 128 B1 ch race 0.1719 -- these
two cells should flip from my worst losses to at/near the lead.

### Change 2 — probe samples = the DRIVER'S 20 ms unit (BORROWED: d1_race r7 via
### d1_prime r8), plus a sizing trap of my own now recorded

Adopted their finding wholesale: a median of ~250 us probe samples still disagrees
with the same process's driver median by up to 20% (race measured it); one probe
sample must aggregate >= --min-sample-ms 20. Also adopted prime r8's untimed
full-length round (burns the schedutil ramp). MY OWN TRAP, for anyone else porting
the recipe: the two SIZING calls run cold/on the ramp and overestimate t1 ~4x, which
silently shrank my "20 ms" samples back to ~5 ms -- caught only because
D1TW_PROBE_VERBOSE prints reps (reps*t read 5.3 ms). Fix: recalibrate reps from a
16-call WARM burst after the untimed round. Verified on-node: 14523*1.28us = 18.6 ms
(1024 exec), 296*65us = 19.2 ms (16384 chain). Cost ~0.5 s, rides the discarded first
warmup unit (prime r8 measured 1.83 s acceptable); setup= stays 0.000 s.
No tax measured anywhere: 1024/16384 B=1 m=1 and both big chained cells wash in
interleaved pairs. The payoff cell is board medians across scoring processes (the r6
board had my 16384 B=1 spread at 47.8%) -- same argument as prime/race.

### Change 3 — NT store threshold 25 MB -> 16 MB (BORROWED: d1_pow2 r7 change 5)

Their re-test at 1024 B=512 (16.8 MB in+out): NT won every interleaved pair and cut
sd ~10x; on a SHARED 24 MB L3 the output RFO reads are waste well below nominal
capacity. My node A/B at 1024 B=512 m=1: medians 1.727-1.738 STEADY (r8) vs
1.733-1.911 and one 3.02-window (r7), min -0.3% -- the variance cut IS the win under
median-of-9 scoring, exactly as they said. 16384 B64 / 4096 B256 (already NT) wash.

### Where the changed cells stand (a80n0 core 3, min us/xf, interleaved windows)

    chains: 32 B1 0.058   64 B1 0.080   128 B1 0.155   (were 0.065-0.074/0.114/0.257)
            64 B512 0.078  128 B512 0.169  16384 B1 53.3 (wash)  1024 B1 2.32 (wash)
    m=1:    1024 B512 1.727-1.738 med (NT; was 1.73-1.91)  16384 B64 39.1 (wash)
            4096 B256 7.83-7.85 (wash)  1024 B1 1.13 (wash)  16384 B1 32.2-32.5 (wash)
    wallaby (idle core 125, for the reply line): 64 B1 m=1 0.028, 64 B512 m=1 0.028,
            chains 32 B1 0.048, 64 B1 0.062, 128 B1 0.117; 1024 B512 m=1 1.483.

### Correctness (final source, node + wallaby)

Single-call rel L2 1.1e-16..3.4e-16 at every graded L x graded B plus edge batches
(32/64/128 x B=1/2/3/9/512, 1024 B=1/3/512, 4096 B=256, 16384 B=1/64); ALL graded
chain gates PASS at full graded m -- worst 1.170e-11 at 1024:1:4000 vs 1e-10 (8.5x
margin, statistically unchanged from r6/r7; the new split chains' own gates:
64:1:60000 1.3e-15, 128:1:30000 1.6e-13, 32:1:100000 9.5e-16); remainder-path chains
32/64/128 x B=2/3/9 PASS; out AND .chain state bitwise repeatable across processes at
all 9 re-checked cells; D1TW_NO_PROBE=1 output bitwise identical to probed;
-DD1TW_EXACTMAP and -DD1TW_SPLITCH=0 both build and PASS (EXACTMAP now covers FIVE map
forms: AoS store, SoA split, vmap register, zmm split-row, ymm split-row). Official
tryout.sh green at 64 B=512, 128 B=512, 1024 B=512.

### Tried / noted, with numbers

- Nothing shipped failed. The one scare: first 32-chain window read r8 median 0.077 vs
  min 0.058 (bimodal) while r7 sat at 0.074 -- a later clean window read r8 0.058/0.058
  steady x6 with batchlane's binary at 0.057-0.058 beside it. Interleave-and-re-window
  before believing a median anomaly (the discipline's 4th round of earning its keep).
- The ssh-cwd trap (batchlane r7, prime r8) BIT ME THREE MORE TIMES this round even
  knowing it: an inline ssh rebuild "failed, no such file" three retries in a row until
  the command moved into a self-cd script. Node commands go in scripts, period.
- NT at 16 MB is a variance play at 1024 B=512, not a big min win (-0.3%): do not
  expect a board min move, expect a tighter median.

### Borrowed, explicitly

- d1_batchlane (r3, via their r6/r7 numbers as the target): chain32/64/128_reg IN FULL
  -- the natural-rows-closed four-step design, fft4/8/16 split kernels, tr8/tr4,
  deint8/inter8, the 40-double plane stagger. Ported with my exact tables (new v4
  row-lane builder returns the favor: their hand-rolled twv is now one generator call)
  and my Goldschmidt map in both widths.
- d1_race (r7) via d1_prime (r8): the probe sample-length finding + untimed warm
  round; my warm-recalibration fix on top is new and recorded above for re-porters.
- d1_pow2 (r7): the NT 16 MB re-test.
- d1_prime (r3): the /tmp squeue shim, seventh round.
- d1_rader (r7): build-the-reference-from-impl_{N-1} (not the impl symlink) -- applied
  before any A/B this round.

### Next round

1. If the r8 board confirms the three chain flips, the remaining structural gaps are
   60 (composite's PFA territory, mine 1.9x behind) and the 1024/4096/16384 B=1 m=1
   split-radix/conjugate-pair item that pow2 r7 also defers -- whoever goes first, the
   other adopts. That is the last big op-count lever on the board for this entry.
2. Watch 32 B=1 chain and 1024 B=512 medians on the board: both changes here are
   partly variance plays; if the medians did not move, the mins were already the story.
3. If a chain gate fails on scoring seeds: -DD1TW_EXACTMAP (five map forms). If the
   split chains regress under the scoring harness: -DD1TW_SPLITCH=0 (and /32
   separately) restores the r7 dispatch exactly.
