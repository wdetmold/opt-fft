# L13_rader — strategy record

Geometry: **L = 13**, cube 13³ = 2197 complex doubles per volume (34.3 KiB),
forward, unnormalised, out-of-place, batched, single-threaded.
Implementation: `impl/L13_rader.c`. `fft3d_name()` → `L13_rader`.
Scored cases (cases.txt): B = 1, 16, 512.

First implementation for this geometry (wave-2 expansion); this file replaces
the round-panel_r6 stub. The rival first-round entry is `L13_direct`
(dense conjugate-symmetric matvec in interleaved layout); read both records
together — we converged on nearly identical wall times through different
structures, which says the geometry is near its practical floor on wallaby.

---

## Round panel_r6 (2026-08-21)

### Technique

**1D module: Rader-13 with the length-12 cyclic convolution evaluated by
CRT/symmetry splitting, not by FFT-12s.** g = 2 is a primitive root mod 13;
fold over the order-6 quotient of (Z/13)\* by {±1}:

```
g^t mod 13, t=0..5:  [1, 2, 4, 8, 3, 6]
u_t = x[g^t] + x[13-g^t],   v_t = x[g^t] - x[13-g^t]
X[g^n]    = x0 + CC_n + i·SS_n          n = 0..5
X[13-g^n] = x0 + CC_n - i·SS_n
X[0]      = x0 + Σ u_t
```

CC is a **cyclic-6 correlation** of u with C_t = cos(2π g^t/13), split once
more along the Z2 factor of Z6 into an **x0-seeded cyclic-3** (kernel
CP_t = (C_t + C_{t+3})/2) plus a **negacyclic-3** (CM_t = (C_t − C_{t+3})/2)
— the ½ is baked into the constants, and seeding the cyclic-3 accumulators
with x0 makes the DC fold free. SS is a **negacyclic-6 correlation** of v
with SN_t = −sin(2π g^t/13), done densely in two halves of three outputs.
This is the L17_winograd module scheme (adopted via **L17_rader panel_r2**,
attribution there), re-derived for p = 13 and verified against numpy in
Python before writing C (max err 2e-15 on a single 13-point line). Rader's
permutations are compile-time load/store row offsets and cost nothing.

**Operation count per 13-point transform (split re/im, per lane-vector):
186 vector FP instructions = 108 FMA + 78 add/sub (294 real flops/point-set),**
against 204 for the dense conjugate-symmetric matvec (what L13_direct runs,
102 ops on 4-complex lanes = 25.5/line vs mine 186/8 = 23.25/line) and my
count of 238 for Rader with two PFA FFT-12s (96+96 for the transforms, 44
pointwise with Bhat[0] = −1/12 and Bhat[6] = √13/12 real, 2 DC). The FFT-12
route was **rejected on paper, never built**: more instructions and worse
regularity, consistent with L17_rader r1 item 3 (splitting Rader's
convolution by FFT identities just re-derives the FFT).

Constants (12 doubles total, `#define`d as 17-digit literals):
CP = {0.0684726387410543, 0.3443007134932395, −0.6627733522342938},
CM = {0.8169833869121557, 0.2237640332379165, 0.3081684651917583},
SN = {−0.4647231720437685, −0.8229838658936564, −0.9350162426854148,
0.663122658240795, −0.992708874098054, −0.2393156642875577}.

### 3D architecture (X-first, plane-fused, fused transposing z-kernel)

All GNU C vector extensions, no intrinsics; VW = 8 under `__AVX512F__`,
VW = 4 otherwise (correctness fallback, runs on wombat).

```
x pass FIRST (22 blocks, 169 contiguous lanes m = y*13+z):
    interleaved loads straight from `in` (DLE/DLO deinterleave pair per
    input row), kernel, ALIGNED split stores into A[kx][y*13+z], pitch 176
per kx plane, software-pipelined one plane deep (U double-buffered):
    zkern13_plane ×2 (lane blocks y0 = 0, 5):
        transposing 8x16 LOADS from the A plane (two 8x8 zmm transposes per
        component, 24 shuffles each), 13-point kernel on register arrays,
        transposing 8x16 STORES into U[y][kz] (stride 16)
    y pass ×2 (lane blocks kz0 = 0, 5, from the PREVIOUS plane's U):
        loads U rows, kernel, ILO/IHI interleaving store into out[kx][ky][kz]
        (direct when in+out fit this machine's L2; staged through a hot
        2.7 KB plane buffer + sequential memcpy burst when they stream)
```

Blocks per volume: 22 + 26 + 26 = 74 × 186 FP = **13.8k vector FP**, 592
lane-slots for 507 line transforms (86% — the 13-wide plane passes run 13 of
16 lanes; the x pass runs 96%). Shuffles: 572 (x deint) + 4992 (z transposes)
+ 676 (y interleave) ≈ 6.2k, **all inside kernel FMA streams** — there is no
standalone transpose pass, no T buffer, and zero scalar edge operations in
the VW=8 path. Node floor: CLX Gold 5218 (one 512-bit FMA unit on the fused
port 0+1, port 5 then free for all shuffles) ≈ 13.8k cycles ≈ **4.75 µs at
2.9 GHz**; wallaby (two FMA units, shuffles compete on port 5) ≈
(13.8k + 6.2k)/2 = 10.0k ≈ 2.6 µs at 3.9 GHz.

Plan-time decisions are **deterministic** (no timing in create, no tuner —
L13_direct's stance, adopted): staging on iff `32·B·2197 > sysconf(L2)`;
cross-volume input prefetch on iff the batch exceeds `sysconf(L3)`. On the
node that gives: B=1 direct/no-pf, B=16 **staged** (1.12 MB > 1 MB L2 —
untestable from wallaby whose L2 is 2 MB, see "for the monitor"), B=512
staged+prefetch.

Zero-pad discipline: A's pad lanes (169..175 per plane) are written only by
the clamped last x block (real lanes, unaligned) and never read; U columns
13..15 receive deterministic garbage from the transposing store that nothing
reads; repeated executes are bit-identical (verified by tryout at every
scored batch, plus B=3).

### Measurement methodology — the round's hardest-won lesson

**Unpinned tryout numbers on wallaby were 2× wrong.** The same binary
measured 7.14 µs (tryout, unpinned, sd 0.1% *within* the run) and 3.31 µs
(`taskset -c 17`, min of ≥4 runs). This is worse than the ±30% clock
bimodality L17_rader r2 documented — early in this round it sent me chasing
a phantom "the x-pass store to out costs 4.8 µs" diagnosis whose absolute
numbers were garbage (the *relative* probe ranking was still right, see
below). Every number in this record is pinned min-of-≥3, alternating with
MKL in the same windows. L13_direct's record independently re-learned the
same thing the same day (their item 3).

### What was measured — wallaby (Gold 6448Y, gcc 11.4, panel flags, pinned min-of-≥4, alternating with MKL)

| case | this file | MKL same windows | ratio |
|---|---|---|---|
| B=1 | **3.313 µs** | 3.565–3.799 | **1.08× ahead** |
| B=16 | **3.325 µs/t** (53.20 µs/call) | 3.58–3.88 | **1.08× ahead** |
| B=512 | **3.99 µs/t** (2043 µs/call) | 3.73 (fast windows) – 4.14 | 0.94× (behind MKL's best windows, ties its slow ones) |

rel L2 vs numpy: 3.97e-16 (B=1), 4.05e-16 (B=16), 4.03e-16 (B=512), 4.00e-16
(B=3); bit-identical across re-runs at every batch; `-Wall -Wextra` clean on
wombat and wallaby; AVX2 fallback verified end-to-end on wombat (12.4 µs/t at
B=8, unpinned, unscored). Setup ≈ 0 (one posix_memalign + memset + two
sysconf calls).

**Unlike L=17, MKL is genuinely strong at 13** (its rader_min boundary —
FFTW switches to Rader exactly here, and MKL clearly ships a real prime
codelet): the margin is 8%, not the 5× the L=17 entries enjoy. Rival
`L13_direct` same day, same machine: 3.296 / 3.347 / 3.935 — we are tied at
B=1 (3.31 vs 3.30), I am marginally ahead at B=16, marginally behind at
B=512. Two structurally different implementations landing within 2% is
evidence the remaining headroom on wallaby is small.

### The round's structural findings (what moved the needle, in order)

Starting point (plane-fused X-last port of the L17 architecture with
standalone 4×4-tile transposes): pinned ~3.5 µs at B=1 but **9.97 µs/t at
B=512**, 2.7× worse than MKL batched.

1. **Kernel stores to `out` are only cheap into cache-hot lines** (~4.3 µs
   of a 7.5 µs unpinned volume in the X-last form; probe ranking XV1/XV2/XV3:
   identical strided stores cost nothing into the just-written A, the
   interleave permutes cost nothing into a hot 1.7 KB scratch, but ANY
   kernel-store pattern into the 35 KB `out` — even with zero permutes —
   was catastrophic once the target left L1). out rows are 2704 B ≡ 16
   (mod 64), so 3/4 of interleaved zmm stores are line-splitting, touching
   ~2 lines each; each cold touch is an RFO. Fixes, both adopted:
   **X-first** (final stores confined to a 2.7 KB per-plane window —
   L17_matrixsimd panel_r3's reorder) and, past L2, the **staged burst
   copy** (L8_radix8's shape, via L17_rader r4): B=512 went 9.97 → ~4.2
   µs/t with X-first + staging.
2. **Fusing both plane transposes into the z kernel as in-register 8×8
   transposing loads/stores** (L17_rader panel_r3 "next" item 2 — proposed
   there, built here first). Kills the T buffer, both standalone transpose13
   passes, all ~2.6k scalar edge ops per volume, and one full store+reload
   round trip; the 8 TR8s per call (192 shuffles) hide under the kernel's
   186 FP on wallaby's second port and should be nearly free on the node
   (port 5 idle under a 1-FMA-unit stream). B=512: 4.2 → 4.20→**4.0**;
   B=1: 3.66 → 3.45.
3. **Software-pipelining zkern(kx) with ykern(kx−1)** (disjoint
   double-buffered U): the shuffle-heavy and FMA-heavy stages share ports
   instead of alternating in humps. B=1: 3.45 → **3.29**.
4. **Plan-time deterministic ys/pf gates from sysconf cache sizes** — the
   wallaby↔node L2/L3 differences make any fixed threshold wrong on one
   machine; reading the machine's own sizes keeps the decision right on
   both without a timing tuner.

### What was tried and did NOT work — with the numbers

1. **X-last with a full-volume interleaved staging buffer + one 35 KB
   memcpy (XV4)**: B=1 went 7.5 → 8.7 (unpinned window) — at B=1 `out` is
   already L2-warm, so staging pays the copy for nothing; it only won
   batched (5.6 vs 10.0 µs/t unpinned). Superseded by X-first + per-plane
   staging, which wins in both regimes.
2. **Double-buffering T/U alone and prefetching the next A plane** (before
   the fused kernel existed): no change at all (3.66 vs 3.66 B=1; 2492 vs
   2425 B=512) — the OOO core was already overlapping planes; the WAR
   hazard I theorised was not binding. (Double-buffering became necessary
   later anyway for the software pipeline, item 3 above.)
3. **Cross-volume input prefetch on wallaby**: B=512 2136–2187 with pf vs
   2046–2116 without (−4%); B=16 55.8–57.1 vs 53.7–54.2. wallaby's 60 MB L3
   holds the whole B=512 batch, so prefetches are pure overhead — exactly
   L17_rader r3's "pf verdicts do not transfer between cache-resident and
   streaming regimes". The pf code ships but only activates past the
   machine's L3 (node B=512 = 36 MB > 22 MB, where L17_winograd measured
   −4.4% for the same trick in a genuinely streaming regime).
4. **Pinning broadcast constants in registers** (KPIN, from L17_winograd
   via L17_matrixsimd; L13_direct measured all-pinning worth −10% for their
   kernel): for THIS kernel, pinning the 6 SN constants changed nothing
   (3.32–3.40 both ways) and pinning all 12 was slightly worse (3.35–3.48).
   My two-stage kernel keeps ~28 values live in the fold stage, so there is
   no register slack; their single-sweep kernel has 29 live *including* 12
   pinned. SN-pinning ships enabled (`L13R_PIN=1`) as a free bet on the
   node's 2 load ports (SPR has 3 — L17_rader r2 flagged exactly this
   asymmetry). `-DL13R_PIN=0/2` selects the variants.
5. **Mixed zmm+ymm tail blocks** (L17_rader r4's node winner): rejected on
   arithmetic for the plane passes — 13 lanes = one zmm + two ymm = the same
   16 half-unit slots as two zmm on a 1-FMA-unit part, unlike 17 = 2×8+1
   where the ymm tail saves 4 slots. Only the x pass would save (21 zmm +
   1 ymm vs 22 zmm ≈ 0.7% of the volume); not worth the dual-width
   instantiation machinery this round.
6. **Skip-phase probe timings (`-DL13R_SKIP`) as a phase profile**: isolated
   phases summed to 5.1 µs of a 7.5 µs whole, and single-phase runs
   mis-ranked badly (skipping a phase changes what stays cache-hot). The
   rdtsc in-situ counters (`-DL13R_TSC`) were the trustworthy instrument;
   both stay in the file as dev flags.

### Borrowed this round (attribution)

* **L17_winograd** (via **L17_rader** panel_r2): the symmetrised
  cyclic/negacyclic prime module scheme, re-derived for p = 13; the
  cross-volume prefetch idea (their r2).
* **L17_rader**: the plane-fused split re/im architecture, TR/pad-lane
  discipline, overlap-and-recompute lane blocks, the vfnmadd `-=`-same-
  constant spelling, the KPIN mechanism, and their r3 "next" item 2 (the
  fused transposing kernel — built here, it works).
* **L17_matrixsimd** (partly via L17_rader r4): X-first pass order; the
  measure-in-blocks discipline.
* **L8_radix8 / L17_rader r4**: finish a plane in hot scratch, burst-copy
  sequentially to `out`.
* **L13_direct** (same-round rival): the "pin everything at small L" claim —
  tested honestly, did not transfer to this kernel (item 4); their
  deterministic-plan stance (no timing tuner) — adopted; their min-of-3+
  pinned methodology — independently converged on.

### For the monitor / next round, in order

1. **Node B=1 prediction: ~4.8–5.5 µs** (13.8k-cycle port-0 floor at
   2.9 GHz; the 6.2k shuffles should hide on the free port 5, which they
   cannot on wallaby — so the node may be *relatively kinder* to this entry
   than to shuffle-light designs). MKL's node number at L=13 is the real
   unknown for everyone.
2. **B=16 on the node uses the STAGED path** (1.12 MB > 1 MB L2, from
   sysconf) — this is a bet wallaby cannot test (its 2 MB L2 says direct,
   measured 54 vs 60 µs). If the node leaderboard shows B=16 lagging B=1 by
   more than ~10%, A/B `-DL13R_FORCE_YS=0` there.
3. **B=512 node**: pf activates (36 MB > 22 MB L3). If it regresses, A/B
   `-DL13R_FORCE_PF=0`; L17_rader r4 ended with pf=0 everywhere, and my gate
   is a bet on L17_winograd's contrary streaming measurement.
4. The remaining B=1 gap to the wallaby floor (~0.7 µs over the port
   balance) looks like x-pass load latency from L2 plus kernel spill traffic
   (~87 rsp refs). The untried lever: give the x pass the zkern treatment —
   batch 2–3 blocks per call so their load streams interleave. Modest
   expected payoff (~5%).
5. If arithmetic ever matters again: 186 is not the floor — the negacyclic-6
   could split via z⁶+1 = (z²+1)(z⁴−z²+1) into 12+18 FMAs against the dense
   36 per component-pair-half, but L17's every-round lesson (arithmetic is
   closed; −12% ops bought −0.8%) says spend the round elsewhere.
