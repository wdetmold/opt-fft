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

---

## Round panel_r7 (2026-08-22)

No node numbers exist yet for L=13 (round panel_r6 shipped no leaderboard), so
every r6 "for the monitor" question is still open. This round therefore spent
its budget on the two levers that are strong bets on the node's port/memory
model and verifiable (or at least proxy-verifiable) on wallaby.

### What changed (two things, plus plumbing)

1. **prefetchw on the out plane, one pipeline step ahead of the staged burst
   copy** (adopted from **L6_unrolled panel_r3** (`fused_pfw`) via the **r5
   panel verdict §4.5** "hide the RFO with prefetchw, do not avoid it with NT
   stores" — node-selected at every streaming cell at L=6/8/36, while NT lost
   there four consecutive rounds; also via **L17_matrixsimd r6**'s `pw`,
   whose ~0.25 µs lead scheme this copies). 43 `__builtin_prefetch(line,1,3)`
   per plane, issued just before the PREVIOUS plane's y pass, so the lead is
   one full z+y plane step (~0.35 µs at node speed). Gated by the same
   plan-time L3 rule as the cross-volume input prefetch (`p->pw = p->pf`),
   because pfw at L3-resident sizes is a documented loss (L6_unrolled r3:
   17% worse; re-measured here, see below). `-DL13R_FORCE_PW=0/1` overrides.
   On the node this activates at B=512 (36 MB > 22 MB L3) and nowhere else.

2. **Global-row z pass: 26 -> 22 blocks per volume.** The z pass used 2
   blocks per kx plane (16 lane-slots for 13 y-rows, 81% full, r6's known
   waste). The volume's z-rows are really one space of 169 global rows
   g = kx*13+y, so blocks of 8 that STRADDLE plane boundaries need only
   ceil(169/8) = 22 blocks (96% full; the last block overlaps rows 161..167
   and recomputes them bit-identically). `zkern13_rows` takes per-row source
   offsets into A (rows of one block are no longer equidistant: +13 within a
   plane, +20 across the PS=176 pad) and per-row dest offsets that encode
   both y*TR and WHICH double-buffered U plane. The y pass still fires
   per-plane the moment its U completes (`znc[]` table says how far it may
   advance after each block), so the r6 z/y software pipeline survives; a
   block spans at most 2 planes and same-parity U reuse is provably ≥ 2
   blocks away, so the double buffer stays safe. All offsets and the
   plane-completion counts are **plan-time tables** (`p->zs/zd/znc`) — the
   first version computed them per block with runtime div-by-13 and was
   +1% at B=1; the tables recovered it and more. `-DL13R_ZG=0` rebuilds the
   old 26-block per-plane table through the same code path, for the monitor
   to A/B on the node.

### Operation count

Blocks per volume: x 22 + z 22 + y 26 = **70 × 186 FP = 13.0k vector FP**
(r6: 74 × 186 = 13.8k, so −5.4%). Shuffles: 572 (x deint) + 4224 (z
transposes, 22×8 TR8) + 676 (y interleave) ≈ **5.5k** (r6: 6.2k, −12%).
Lane-slot utilization 560/592 slots for 507 lines + 8 recomputed rows ≈ 92%
(r6: 86%). Node port floor (one fused 512-bit FP port): 13.0k cycles ≈
**4.49 µs at 2.9 GHz** (r6 floor: 4.75). The y pass keeps its 3-idle-lane
waste: reclaiming it needs lanes = (kx,kz) across planes, and out's index
kx*169+ky*13+kz is not affine in kx*13+kz — the interleaved store would
straddle discontiguously. Checked and abandoned on paper this round.

### What was measured — wallaby (Gold 6448Y, gcc 11.4, panel flags, taskset
pinned min-of-≥3, alternating with MKL in the same windows)

| case | this file | r6 | MKL same windows | ratio |
|---|---|---|---|---|
| B=1 | **3.21–3.29 µs** | 3.313 | 3.578–3.825 | ~1.10× ahead |
| B=16 | **3.277 µs/t** (52.43 µs/call) | 3.325 | 57.5–60.7 µs/call | 1.10× ahead |
| B=512 | **4.01–4.10 µs/t** (2054–2100 µs/call) | 3.99 | 1913 (fast windows)–2140 | 0.93× (unchanged story) |
| B=2048 (unscored streaming proxy) | **5.56 µs/t** (11 388 µs) | 6.23 (pw off, pre-ZG) | — | −11% round total |

rel L2 vs numpy: 3.97e-16 (B=1), 4.05e-16 (B=16), 4.03e-16 (B=512), 4.00e-16
(B=3); bit-identical across re-runs at every batch; ZG=0 vs ZG=1 outputs
bit-identical; pw on/off bit-identical. `-Wall -Wextra` clean on wombat and
wallaby; AVX2 fallback verified end-to-end on wombat (12.9 µs/t at B=8,
unpinned, unscored).

The **B=2048 A/B for pw alone** (same code state, streaming regime, the honest
proxy for node B=512 which wallaby's 60 MB L3 hides): 11 996 µs with pw vs
12 767 without = **−6.0%**, bit-identical. ZG then took the same case to
11 388. Note wallaby's scored B=512 numbers above run with pw OFF (gate
closed, 36 MB < 60 MB L3); the node's B=512 will run with it ON.

The wallaby B=1 gain from ZG is real but small (3.29 vs 3.35 same-window,
best windows 3.21 vs 3.29): wallaby has two FMA ports and is latency-bound
here, so removing 744 port-ops barely shows. The node has one 512-bit FP
port and was predicted port-bound at B=1 — **this change is aimed there**
(−5.4% of the floor), which wallaby structurally cannot confirm. That is the
round's main bet; `-DL13R_ZG=0` is the rollback if the node disagrees.

### What was tried and did NOT work — with the numbers

1. **Software-prefetching the next x-block's 26 input lines (t0)**: B=1
   3.382–3.591 with vs 3.399–3.595 without — identically nothing. The x
   pass's 2.6×-above-floor time on wallaby (TSC: 27% of the volume) is NOT
   simple L2 miss latency; OOO already hides it. Ships gated OFF
   (`-DL13R_XPF` to re-enable for node experiments). The stall is dependency
   chains + front-end, and the next lever there is interleaving two blocks
   per call (not attempted this round — 28 live values × 2 certainly spills;
   see Next).
2. **Forced pfw at L3-resident batch** (wallaby B=512): 2101–2230 with vs
   2054–2099 without, ~**−3%** — re-confirms L6_unrolled r3's warning on a
   second machine and validates keeping the L3 gate rather than tying pw to
   the staging (`ys`) gate.
3. **Runtime-computed global-row offsets** (div-by-13 per row per block):
   +1% at B=1 vs r6 despite the FP savings — ~50 scalar µops per block
   compete with the vector stream. Plan-time tables (`p->zs/zd/znc`) fixed
   it. Lesson: at 186-op kernel granularity, per-block scalar bookkeeping is
   visible; precompute it.
4. **PIN re-sweep under the new structure** (PIN=0/1/2 at B=1): 3.167 /
   3.189 / 3.198 best-window mins — still within noise of each other,
   consistent with r6 item 4. PIN=1 stays (node 2-load-port bet unchanged).
5. **Reclaiming the y pass's idle lanes** (22 blocks instead of 26 via
   global (kx,kz) lanes): killed on paper — needs U in [y][kx*13+kz] layout,
   whose z-side stores want masked writes (fine) but whose out-side
   interleaved stores go discontiguous at every plane boundary (fatal).
   Documented so nobody re-derives it.

### Borrowed this round (attribution)

* **L6_unrolled panel_r3** (via the r5 panel verdict §4.5 and
  **L17_matrixsimd r6**): prefetchw on the cold out stream, its ~0.3 µs lead
  placement, and the L3-resident warning that shaped the gate.
* **L17_matrixsimd r1** (via its r6 restatement): "NT lost on the node
  everywhere, four rounds running" — NT staging was NOT implemented here on
  that record, saving the round the experiment.
* The global-row z pass is this entry's own extension of the r6 fused
  transposing kernel (itself from L17_rader r3's proposal).

### For the monitor / next round, in order

1. **Node A/Bs this round shipped the knobs for**: `-DL13R_ZG=0` (if the
   global-row z pass somehow loses on CLX), `-DL13R_FORCE_PW=0` (if pfw at
   B=512 regresses — L17_matrixsimd r6 measured pw mildly negative on
   wallaby-resident cases but the node picked it at every genuinely
   streaming cell), `-DL13R_FORCE_YS=0` (r6's untestable B=16 staging bet,
   still untestable from wallaby: node L2 = 1 MB vs wallaby 2 MB).
2. **Node B=1 prediction: ~4.5–5.2 µs** (13.0k-cycle port floor at 2.9 GHz;
   r6 predicted 4.8–5.5 on the 13.8k floor). If the node lands near 4.5 the
   port model is confirmed and the next FP cut is the only lever; if it
   lands near 5.2, latency dominates there too and the x-pass 2-block
   interleave (accepting spills for MLP) is the experiment to run.
3. **Node B=512**: expect the pfw win to exceed wallaby-B2048's −6% (CLX has
   less MLP to hide RFOs). If B=512 still trails B=16 by ≫ the ~35 KB/volume
   copy cost, the next lever is fusing the burst copy into the y-pass stores
   (write out directly with pfw'd lines, skipping the staging round trip) —
   i.e. direct + pfw as a third mode.
4. Arithmetic stays closed: 186/kernel and now 70 blocks; the only remaining
   block waste is the y pass's 12 slots/volume, blocked by the out layout
   (item 5 above). Do not spend a round there.

---

## Round panel_r8 (2026-08-22)

First round with node numbers for L=13 (panel_r7 leaderboard), and they ruled:
**B=1 6.054 µs** (predicted 4.5–5.2 — +16% over the top, latency-bound like
every first-contact geometry per verdict item 10), **B=16 7.279** (the r6
staged-store bet LOST: ys was the only config difference from B=1, so staging
alone cost +20%), **B=512 9.469** — behind L13_direct (8.396) and, uniquely on
the whole board, behind both MKL builds (9.106/9.237). The r7 prediction
"expect the pfw win to exceed −6%" was wrong in direction: the entry *degraded*
with batch. This round therefore attacked the store policy and nothing else
structural.

### What changed: staging is dead; direct stores + prefetchw everywhere it streams

The full ys×pw matrix, pinned min-of-≥3 process instances on wallaby
(`taskset -c 49`, MKL canary 3.569–3.574 at B=1 — machine state matches r6/r7):

| config | B=512 (µs/call) | B=2048 (µs/call) |
|---|---|---|
| direct, no pw (y0p0) | 2386 | 15933 |
| **direct + pw (y0p1)** | **1923** | **10774** |
| staged, no pw (y1p0) | 2032 | 12160 |
| staged + pw (y1p1, the r7 shipped state) | 2094 | 11483 |

Direct+pw wins both streaming regimes outright (−5.4% vs r7's config at B=512,
−6.2% at the B=2048 node-B512 proxy), and reconciles everything: the r6 "staging
wins at B=512" measurement predates pw — staging was only ever a workaround for
un-prefetched cold-line RFOs, and once pw hides the RFO the 35 KB/volume burst
copy is pure overhead. This is r7's "next" item 3 built and confirmed, and it
matches **L13_direct r7**'s independent finding (staged-Z +16% on their
architecture) and the node's own B=16 verdict.

Shipped gates (still deterministic, no tuner):
* **ys = 0 always** (`-DL13R_FORCE_YS=1` re-enables for A/B; sb buffers kept).
* **pw moved from the L3 gate to the L2 gate**: on iff `32·B·2197 > sysconf(L2)`.
  Direct stores without pw are catastrophic as soon as out streams past L2
  (2386 vs 1923 at wallaby B=512, which is L2-streaming but L3-*resident* —
  so the old "pw only past L3" rule was an artifact of the staged copy, and
  L6_unrolled r3's "pfw 17% worse when L3-resident" does not apply to strided
  kernel stores). pw at L2-resident sizes costs ~2% (B=1: 3.25 vs 3.18), so
  the gate keeps it off there. On the node: B=1 off, **B=16 ON** (2.24 MB >
  1 MB), B=512 ON.
* pf (cross-volume input prefetch) unchanged, L3 gate — re-confirmed under the
  new store mode: B=2048 pf0 11970–12000 vs pf1 10702–10904, i.e. pf is worth
  −9% streaming.

### Operation count

Unchanged: 70 blocks × 186 vector FP = 13.0k FP, ~5.5k shuffles, node port
floor 4.49 µs at 2.9 GHz. What changed is traffic: the staged path's extra
35 KB/volume L1 round trip (memcpy) is gone from every batched case.

### What was measured — wallaby (Gold 6448Y, gcc 11.4, tryout flags, pinned min-of-≥3 process instances, MKL in the same windows)

| case | this round | r7 | MKL same windows |
|---|---|---|---|
| B=1 | **3.185 µs** | 3.21–3.29 | 3.569 (1.12× ahead) |
| B=16 | **51.69 µs/call (3.231/t)** | 52.43 | 57.58 (1.11× ahead) |
| B=512 | **1923.8 µs (3.757/t)** | 2054–2100 | 1910 (0.99× — tied, was 0.93×) |
| B=2048 (unscored streaming proxy) | **10 901 µs (5.32/t)** | 11 388 | — |

rel L2 vs numpy: 3.973e-16 (B=1), 3.999e-16 (B=3), 4.049e-16 (B=16),
4.025e-16 (B=512); bit-identical across re-runs at every batch; `-Wall
-Wextra` clean on wombat and wallaby; AVX2 fallback verified end-to-end on
wombat (12.5 µs/t at B=8, unpinned, unscored). B=1 path is byte-for-byte the
r7 configuration (ys was already 0, pw already off), so node B=1 should
reproduce ~6.05.

### What was tried and did NOT work — with the numbers

1. **The L23_rader odd-cache-line pad (PS 176 → 184, i.e. 22 → 23 lines)**,
   the r7 verdict's biggest single mechanism (−25–30% at L=23): here a wash at
   B=1 (mins 3.191 vs 3.220, medians overlap) and slightly WORSE at B=512
   direct+pw (1994 vs 1926). My x pass's in-row stride is 2704 B (gcd 16 with
   4096), so its load residues already sweep the space instead of sitting in a
   fixed mod-4096 relation to the PS-strided stores — L=23's self-inflicted
   alignment does not exist here to fix. Default stays 176; `-DL13R_PS`
   re-opens it.
2. **x-pass 2-block interleave** (`-DL13R_X2`, r7's B=1 "next" item): built,
   bit-identical output, and neutral-to-slightly-worse on wallaby across 5
   interleaved pinned instances (best 3.188 vs 3.168, medians worse). wallaby
   was already known not load-bound in the x pass (r7's XPF null), so this is
   the expected null *there*; the node (weaker OOO, 2 load ports) could still
   differ. Ships default-OFF; the knob is for a node A/B at B=1.
3. **prefetchw lead of 2 planes instead of 1**: B=2048 10748 vs 10702 (wash),
   B=512 1960 vs 1917 (worse). Lead stays 1 plane.
4. **The scored-build unroll gap** (L45_pfa's r7 discovery that the node build
   lacks `-funroll-loops`, worth 10% to them): simulated with a no-unroll
   build — 3.194 vs 3.186 µs at B=1, i.e. this file has no dependence on the
   flag (everything hot is macro-unrolled or always_inline). No pragma needed;
   recorded so nobody re-checks.

### Borrowed this round (attribution)

* **L13_direct panel_r7**: the staged-Z-loses-on-this-geometry finding (their
  FORCE=10 measurement, +16%), which corroborated killing my default staging.
* **L23_rader panel_r7** (via the r8 context/verdict): the odd-cache-line pad
  experiment — tested honestly, did not transfer (item 1 above).
* **L45_pfa panel_r7**: the build-flag audit idea — null here (item 4).
* The direct+pw third mode itself is this entry's own r7 "next" item 3,
  executed.

### For the monitor / next round, in order

1. **Node predictions, anchored on measured r7 ratios per verdict item 10**
   (not re-derived floors): **B=1 ≈ 6.0–6.1** (code path unchanged);
   **B=16 ≈ 5.9–6.3** (staging's +20% removed; pw now active there — wallaby's
   analogous L2-streaming/L3-resident regime measured pw at −19%); **B=512 ≈
   8.7–9.1** (−6.2% on the honest wallaby proxy; CLX has less MLP so pw may
   help more). That should retake B=16/B=512 from MKL; L13_direct (8.396 at
   B=512) likely stays ahead there.
2. **Node A/B knobs shipped**: `-DL13R_FORCE_PW=0` (if pw at B=16 regresses on
   the node), `-DL13R_X2` (B=1, the one lever wallaby structurally cannot
   evaluate), `-DL13R_PS=184` (aliasing pad, in case the node's L2 behaves
   differently), `-DL13R_FORCE_YS=1` (rollback to staged, not expected).
3. **B=1 is the open front**: 6.05 = 1.35× the port floor and the phase profile
   (TSC, wallaby) puts 28% in the x pass and 72% in the z+y pipeline with no
   single dominant stall. If X2 nulls on the node too, the next candidates are
   (a) reordering z blocks y-major to software-pipeline the x pass into the z
   pass (costs the z/y pipeline and 13 live U planes ≈ 54 KB — probably a bad
   trade, sized here so nobody starts it casually), and (b) accepting that the
   architecture is converged and the remaining 2.5% to L13_direct at B=1 is
   their lower shuffle count.
4. NT stores stay unbuilt: five geometries' node evidence ("hide the RFO,
   don't avoid it") plus pw's measured win here settle it.

---

## Round panel_r9 (2026-08-22)

Node r8 standings: second behind L13_direct in all three cells (6.074 vs
5.725 at B=1, 6.683 vs 6.039 at B=16, 8.809 vs 8.110 at B=512), ahead of
both MKLs everywhere. This round is one structural change aimed at the
node's execution engine, plus one measured null that closes an r8 question.

### The diagnosis: the z blocks serialize on the node, and the model says so

r8's node B=1 (6.074 µs = 17.6k cycles at 2.89 GHz) sits 4.6k cycles above
the 13.0k port-0 floor. New explanation, quantitative: a z block is ~450
uops of dependency-serial, port-HOMOGENEOUS phases — 96 shuffles (port 5) →
186 FMA (port 0) → 96 dependent shuffles (port 5) — and 450 uops exceeds
CLX's ~224-entry ROB, so the next block's independent work never enters the
window and each port idles while the other runs. Serial cost model:
z block ≈ 96+186+96 ≈ 378 cycles ×22 = 8.3k, y pass 26×186 = 4.8k, x pass
≈ 4.1k → **17.2k cycles ≈ the measured 17.6k**. The same model explains
wallaby (12.7k measured; SPR's ~512-entry ROB + shuffles sharing port 5
with FMA make it port-5-bound instead, ≈ 11–12k). r6's assumption "the
shuffles hide on the free port 5" is falsified by arithmetic on both
machines — a shuffle can only hide if FMA work is inside the OOO window,
and in the r8 schedule it never is.

### What changed: z blocks port-fused with y lane-blocks (`zykern13_f`)

Each z block now carries **one y lane-block interleaved at source-stage
level in the same instruction stream**: stages alternate ~50–130-uop
port-homogeneous chunks (z re-transposes / y fold+cyclic / z re-compute /
y negacyclic half 1 + stores / z im-transposes / y half 2 + stores / z
im-compute / mix+store-transposes), so every 48-shuffle burst has ≥54 FMAs
within one ROB span. To make the z kernel divisible, its math is
restructured into **independent re/im halves** (`Z_HALF`: 81 FP each; the
components only meet in the final ±i·SS mix of 24 adds) — same operation
trees as `kern13_regs`, so outputs are **bit-identical to the r8 code**
(verified by cmp at B = 1, 3, 16, 512). `restrict` on all fused-call
pointers is load-bearing: without it gcc must order the y stores to `out`
against the z loads from A and the compile-time interleave dies.

Schedule (plan-time, `p->zy[]`, deterministic): consume the oldest
available y lane-block (plane complete as of the *previous* z call) in
each fused call → z blocks 0,1 run unfused, blocks 2–21 each carry one y
block, 6 y blocks trail unhidden. Consumption now lags plane completion by
up to ~4 planes, so **U is 8-deep buffered** (ur[0..7], 3.3 KB per plane
pair, 26.6 KB total — still L2-trivial). `create()` verifies the reuse
invariant (no plane written by call b may share a buffer with an
unconsumed or same-call-consumed y plane) and **falls back to the unfused
loop if it ever fails**; `-DL13R_FUSE=0` forces the fallback for node A/Bs.

### Operation count

Unchanged: 70 blocks × 186 vector FP = 13.0k, ~5.5k shuffles, port floor
4.49 µs at 2.9 GHz. What changed is the *schedule*: 20 of 22 z blocks'
shuffle bursts now have FMA partners in-window. The serial model that
predicted r8's 17.6k now predicts ≈ 13.6–14.5k cycles ≈ **4.7–5.0 µs at
B=1 on the node** (x pass unchanged and still ~1.2× its own floor; 2
unfused z blocks + 6 trailing y blocks stay serial). The fused call adds
~10–20 register spills (ports 2/3/4, off the critical ports) — accepted
deliberately; this is a *pairing* of complementary ports, not a reordering
of same-port work, which is how it escapes the r7/r8 "delete uops, don't
reschedule them" verdict against the L=17 scheduling attacks.

### What was measured — wallaby (Gold 6448Y, gcc 11.4, tryout flags, taskset -c 49, interleaved variant pairs run-by-run, min of ≥3 instances)

Pairwise FUSE=0 vs FUSE=1 in the same windows (today's windows run ~4%
slower than r8's; the pairwise deltas are the signal):

| case | FUSE=0 (r8 code) | FUSE=1 | pairwise Δ |
|---|---|---|---|
| B=1 | 3.460–3.628 | **3.183–3.317** | **−7.5%** |
| B=16 | 54.01–56.84 | **51.23–54.58** | **−4%** |
| B=512 | 2021–2184 | **1999–2070** | −1–2% |
| B=2048 (streaming proxy) | 11404–12810 | **10750–10885** | **−5%** |

Best-window finals, MKL alternating in the same windows: **B=1 3.193 µs**
(MKL 3.579–3.781, 1.12× ahead), **B=16 51.23 µs/call = 3.202/t** (MKL
57.35, 1.12×), **B=512 2006 µs = 3.918/t** (MKL 1913–2129, ~tied).
Wallaby gains at all four batches even though the ROB argument is
node-specific — SPR gets the port-5 contention relief instead. rel L2 vs
numpy 3.973e-16 / 3.982e-16 / 4.049e-16 / 4.025e-16 (B=1/3/16/512),
bit-identical re-runs at B = 1, 16, 512, 2048, `-Wall -Wextra` clean, all
knob combos (ZG=0, FORCE_YS=1, X2, PIN=0) build and PASS, AVX2 fallback
PASS on wombat (12.7 µs/t at B=8, unpinned, unscored — path untouched).

### What was tried and did NOT work — with the numbers

1. **4-deep U buffering for the fused schedule**: my hand simulation said
   max lag 3 planes; the plan-time verifier said no — z block 17's row
   g=143 already touches plane 11 (buffer 3) in the very call that
   consumes plane 7's y block (buffer 7&3 = 3), a genuine within-call WAR
   hazard that `restrict` would have turned into silent corruption. 8-deep
   fixed it. **Keep the verifier; do not trust schedule arithmetic done in
   prose.**
2. **The measurement trap that followed**: with fuse silently falling back
   (fuse=0), my first "A/B" measured FUSE=0 against FUSE=0 and produced a
   perfect tie at every batch. Nothing looked wrong. Lesson for every
   entry with a create()-time fallback: **print/verify the engaged
   configuration before believing any A/B** (a 1-line scratch-copy debug
   print settled it; the real A/B then showed −7.5%).
3. **pw hysteresis for node B=16 (the r8 prediction miss): NULL, gate
   unchanged.** Hypothesis was that prefetchw hurts at *marginal* L2
   overflow (node B=16 ws = 1.12×L2). Wallaby B=32 has the identical
   ratio (2.25 MB / 2 MB L2): pw ON 113.4–119.7 µs vs OFF 126.1–131.3 —
   **pw wins by 10%** exactly where the hypothesis said it should lose.
   The r8 L2 gate stands; the node B=16 shortfall was evidently the same
   z-block serialization this round attacks (B=16 runs the B=1 code path
   plus L2 streaming), not the prefetch policy.

### Borrowed this round (attribution)

* **L45_mixedradix / the r8 verdict's L=45 line** ("the next lever is the
  fuse-z-store-with-y-load rewrite or nothing"): corroborated the
  direction; the mechanism here (ROB-span port pairing, re/im-split prime
  kernel) is this entry's own.
* **L36_pfa panel_r8** (via the r8 verdict §6): the "verify in create(),
  fall back deterministically, never hope" pattern — implemented as the
  U-reuse verifier, which promptly caught a real bug (item 1).
* **L17 r8 verdict synthesis** ("delete uops, don't reschedule") was
  treated as a constraint to argue against, not ignore: this change adds
  uops (spills) and wins by pairing ports, which is the one thing the
  falsified L=17 scheduling attacks never did.

### For the monitor / next round, in order

1. **Node predictions.** Anchored on r8's measured cells with the wallaby
   pairwise deltas as the floor and the serial-model arithmetic as the
   ceiling: **B=1 4.9–5.7 µs** (r8: 6.074), **B=16 5.7–6.4** (r8: 6.683),
   **B=512 8.2–8.7** (r8: 8.809). If B=1 lands ≤5.4 the ROB model is
   confirmed and L13_direct's 5.725 should fall; if it lands ≥5.9 the
   model is falsified on CLX and `-DL13R_FUSE=0` is the rollback (outputs
   bit-identical, so the A/B is timing-only).
2. **Knobs shipped**: `-DL13R_FUSE=0` (unfuse), plus everything from r8
   (`-DL13R_FORCE_PW=0`, `-DL13R_X2`, `-DL13R_PS`, `-DL13R_ZG=0`,
   `-DL13R_FORCE_YS=1` — all still build and PASS under the new
   structure).
3. **If the fused B=1 works, the next serial residues in order**: the x
   pass (22 FP-heavy blocks, ~28% of the volume, nothing to pair against
   inside one volume — but at B>1 the previous volume's 6 trailing y
   blocks could fuse with the next volume's first x blocks only for MLP,
   not ports; probably dead), the 2 unfused z blocks and 6 trailing y
   blocks (~0.4 µs of unpaired port time; fusing z0 with z1 is
   port-complementary and worth ~130 cycles — cheap to try), and the
   negacyclic-6 z⁶+1 split (still parked per the L=17 arithmetic verdicts).
4. The pw/pf gates are now both measurement-backed at their boundary
   regimes (r8 matrix + this round's B=32 null). Do not revisit without a
   node number that contradicts them.

---

## Round panel_r10 (2026-08-22)

Node r9 standings ruled this round before it started: **B=1 6.030 µs
(fused schedule −0.7% vs r8 — inside the ±2% layout-noise floor the r9
VERDICT §3(d) established), B=16 6.963 (+4.2% REGRESSION), B=512 9.055
(+2.8% REGRESSION)** — the ROB port-fusion model was falsified at my own
pre-registered threshold, and the verdict's L=13 line ordered the
`-DL13R_FUSE=0` rollback. Still second behind L13_direct in all three
cells (5.739 / 5.957 / 7.965). This round executes the rollback
precisely, plus one adopted pacing detail, and closes two deletion
questions by audit rather than by hope.

### What changed (three things, all knob-reversible)

1. **The fused zy schedule now runs at B=1 ONLY** (`want_fuse` requires
   `batch == 1` in create()). This takes the verdict's rollback exactly
   where the node priced the fusion as a loss (both batched cells) and
   keeps it in the one cell where the node measured it ahead (6.030 vs
   r8's 6.074 — a margin I explicitly do NOT claim as real, per the
   noise-floor doctrine; it is simply the measured-not-worse
   configuration). `-DL13R_FUSE=0` now unfuses B=1 as well.
2. **U buffer depth follows the schedule** (new plan field `p->um`: index
   mask 7 fused, 1 unfused; the zd offset table is built with it).
   r9 had silently 8-deepened U for BOTH paths because the zd table baked
   in `kx & 7` — so a naive FUSE=0 at batch would NOT have reproduced
   r8's batch code: it would have dragged 26.6 KB of hot U scratch (vs
   r8's 6.7 KB) through the streaming cells. Wallaby evidence that this
   footprint mattered: in today's same-window pairs, unfused-2-deep at
   B=16 (51.5–51.8 µs/call) TIES the fused r9 binary (51.8–52.6), while
   r9's own record shows unfused-8-deep at 54.0–56.8 in its same-window
   pairs — i.e. part of what r9 booked as "the fused schedule wins at
   batch on wallaby" was actually "8-deep U loses unfused".
3. **Prefetch pacing split in the batched loop** (`L13R_PACE=1`,
   default): prefetchw keeps its one-plane lead ahead of the y stores,
   but the cross-volume INPUT pf slice moves to AFTER the plane's y
   stores (between plane pl's y and the next z blocks) — adopted from
   **L13_direct**'s node-winning pf exec, which issues its read-side
   prefetches between the Y and Z groups. Wallaby, pinned, interleaved
   run-by-run: B=2048 pace1 wins 6/8 pairwise, min-of-8 10426 vs 10930 µs
   (−4.6%); B=512 pace1 5/8, mins tied (1930.0 vs 1932.3), medians −1%.
   At B=16 and B=1 the two spellings are semantically identical (pf gate
   closed), so this risks nothing outside the streaming cells.
   `-DL13R_PACE=0` restores the r8 joint placement.

Also: **create() now prints the engaged configuration**
(`fuse/um/ys/pf/pw/pace/znb`) through `fft3d_description()` into the
per-run JSONs — my own r9 lesson ("print the engaged configuration
before believing any A/B") plus the verdict's finding that in-plan
description strings are the panel's only working instrument; static-
buffer pattern from L13_direct.

### Operation count

Unchanged: 70 blocks × 186 vector FP = 13.0k FP, ~5.5k shuffles, node
port floor 4.49 µs at 2.9 GHz. The B=1 instruction stream is the r9
fused one; the batched stream is exactly r8's (2-deep U, unfused) plus
the pacing split. Outputs of the shipped binary are **bit-identical to
the r9 binary's at B = 1, 3, 16, 512** (verified by cmp on wallaby),
and all eight knob builds (`FUSE=0, ZG=0, FORCE_YS=1, PACE=0, X2,
PIN=0, FORCE_PW=1, FORCE_PF=1`) build clean and remain bit-identical.

### What was measured — wallaby (Gold 6448Y, gcc 11.4, tryout flags,
taskset -c 49, min over ≥4 (8 for the pace A/B) interleaved process
instances, MKL alternating in the same windows)

| case | this round (shipped) | r9 binary same windows | MKL same windows |
|---|---|---|---|
| B=1 | **3.152 µs** | 3.223 | 3.577–3.813 (1.13× ahead) |
| B=16 | **51.51–53.46 µs/call (3.22–3.34/t)** | 51.82 | 60.67 (1.13×) |
| B=512 | **1918–1930 µs (3.75/t)** | 2004 (−4% pairwise) | 1913–2134 (~tied) |
| B=2048 (streaming proxy) | **10426 µs (5.09/t)** | 11221 (−7% pairwise) | — |

(B=16 windows drifted ~4% slower between the morning A/B set and the
final set — both binaries and MKL moved together; the pairwise readings
are the signal, as ever.) rel L2 vs numpy 3.973e-16 / 3.999e-16 /
4.049e-16 / 4.025e-16 (B=1/3/16/512) — unchanged, as it must be for a
bit-identical rollback; repeatable (bit-identical re-runs) at every
batch tried; `-Wall -Wextra` clean on wombat and wallaby; AVX2 fallback
PASS end-to-end on wombat (12.8 µs/t at B=8, unpinned, unscored).

### What was tried / audited and did NOT yield — with the numbers

1. **L13_direct's r9 XOR-deletion class has no analog here — closed by
   disassembly, not assumption.** Full-object audit of the CLX-ISA
   build: **0 `vpxor`/`vxorpd` in the entire object.** My signs were
   already folded at derivation time (SN = −sin constants, ± spelled in
   the add/sub mixes, negated wrap terms as `-=`-same-constant emitting
   vfnmadd). Nothing to delete.
2. **L13_direct's r8 split-access pad class does not transfer either,
   and the reason is structural**: my z pass's A loads are ~87%
   cache-line-splitting (in-plane row stride is 13 doubles = 104 B, so
   row starts cycle through all eight mod-64 residues), which is the
   same disease their t1 pad (338→344 doubles) cured — but A's row
   stride is FORCED to 13 by the x pass, whose lanes are the contiguous
   m = y*13+z index: padding rows to 16 doubles would turn the x pass's
   aligned contiguous stores into scatters. The one own-stride knob I do
   have (plane pitch PS) was already swept in r8 (PS=184: wash/worse).
   Documented so nobody re-derives it; the split loads are the price of
   the shuffle-free x pass.
3. **Fusing the 2 unfused z blocks with each other at B=1** (r9 "next"
   item 3, ~130 cycles ≈ 0.7% of the cell): not built — below the ±2%
   noise floor the r9 verdict told the panel to stop reading, so it
   cannot be evaluated even if done.
4. Constant-load pressure noted for the record: 380 rip-relative
   constant loads in execute() (each KPIN asm barrier defeats CSE, by
   design). Pinning variants measured neutral twice before (r6 item 4,
   r7 item 4) — left alone.

### Borrowed this round (attribution)

* **L13_direct**: the read-side prefetch placement (input slices between
  the Y and Z groups — their node-winning pf exec), and the
  description-buffer pattern for printing the engaged config.
* **r9 VERDICT**: the rollback directive itself, and the ±2%
  noise-floor doctrine that shaped what was NOT attempted.

### For the monitor / next round, in order

1. **Node predictions.** B=1 ≈ **6.0–6.1** (instruction stream unchanged
   from r9's fused path). B=16 ≈ **6.6–6.8** (exact restoration of r8's
   batch code, which measured 6.683; the pacing split is inert at B=16).
   B=512 ≈ **8.5–8.8** (r8's 8.809 restored, minus the pacing win if
   wallaby's −1% to −4.6% streaming reading transfers; the two-machine
   history of memory mechanisms says discount it). If B=16 or B=512 land
   ABOVE r8's 6.683/8.809, something beyond the rollback is wrong —
   check the JSON description strings first (`fuse=0 um=1 ... pace=1`
   expected at batch; `fuse=1 um=7` at B=1).
2. **Standing A/B asks, one build each**: `-DL13R_FORCE_PW=0` at B=16
   (third round of asking; my B=16/B=1 ratio is 1.10 vs L13_direct's
   1.055 and pw is the only config difference between my two cells —
   wallaby's analog regime says pw wins there, but it has never been
   priced on CLX); `-DL13R_PACE=0` at B=512 if the cell disappoints;
   `-DL13R_FUSE=0` at B=1 (prices the fused schedule's B=1 value
   honestly — if it reads ≤6.03 the fusion carries nothing and the 1198-
   line fused kernel should be deleted next round for code-layout
   hygiene).
3. **B=1 remains the open front and is now mechanism-free**: 6.03 µs =
   1.34× the port floor, every named theory (ROB serialization,
   x-pass load latency, pinning, pad aliasing, unroll flags) measured
   null or falsified on the node. The honest candidates left are (a)
   the ~87% split-load rate on the z pass's A reads (structural, see
   item 2 above — a fix needs a different x-pass store shape, i.e. a
   real rewrite), and (b) conceding that 1.2–1.5× floor is what this
   machine gives every geometry (the r9 verdict's board-wide pattern)
   and L13_direct's 1.22× is the shuffle-count difference. I lean (b);
   if a rewrite is ever funded it should be (a).
4. The pw/pf gates stay measurement-backed (r8 matrix, r9 B=32 null,
   this round's pace A/B). Do not revisit without a node number.
