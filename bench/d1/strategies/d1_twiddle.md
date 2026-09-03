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
