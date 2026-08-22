# VERDICT — panel round `panel_r8`

Monitor's judgement on the measurements taken on `p55n3` (Intel Xeon Gold 5218, Cascade
Lake, 2×16c, exclusive, slurm job 438524, 2026-08-22T02:52), gcc 11.4.0,
`-O3 -march=native -funroll-loops`, `governor=powersave`. Roster: **19 implementations, all
built, all ran, all correct, none missing.** Sources for this round are in
`bench/geom/impl_8/` (`impl` → `impl_8`).

**Three things about this round's shape before any numbers.**

1. **This was a flat round for the panel and a noisy one for MKL.** Sixteen of the 28
   scored cells moved less than 1% at the panel's own best entry. The apparent widening of
   several margins is **MKL getting slower, not us getting faster** — §3 quantifies it and
   §1 quotes both figures wherever it matters. Do not bank the raw L=36/L=45/L=64 batched
   margins.
2. **The round's three positive results are not kernels.** They are (a) L45_pfa's
   opaque-base asm barrier, which took all three L=45 cells back after r7's reversal;
   (b) L36_pfa's in-plan node probe, which answered a question three records had asked for
   three rounds; and (c) L64_radix8's tiled structure, which closed the corpus's largest
   untried structural move by losing cleanly. Against those, the round's *kernel* work —
   three independent uop deletions at L=17, seven ports of the L=23 padding result, one
   ERMS RFO-deletion scheme — produced nulls almost across the board (§5).
3. **The brief names L = 6, 8, 17, 36.** Eight geometries were scored. §1 gives those four
   in full and adds a compact table for L = 13, 23, 45, 64, because the promotion decision
   in §7 needs them and `CURATION.md` rule 1 is per-geometry, not per-brief.

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are per transform, minimum across three independent processes, as reported by
`leaderboard.txt`. "Best library" is whichever of FFTW ×3 / MKL 2022 / MKL 2026 / ducc0 was
fastest **in that exact cell**.

### L = 6 (volume 216)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L6_unrolled 0.217 µs** (L6_pfa 0.220) | mkl_dfti 0.370 µs | **1.71×** |
| B=64 (0.42 MiB) | **L6_unrolled 0.214 µs** (L6_pfa 0.222) | mkl_dfti 0.392 µs | **1.83×** |
| B=4096 (27 MiB) | **L6_pfa 0.389 µs** (L6_unrolled 0.391) | mkl_dfti 0.561 µs | **1.44×** |
| B=32768 (216 MiB) | **L6_unrolled 0.565 µs** (L6_pfa 0.572) | mkl_dfti 0.700 µs | **1.24×** |

**Read B=1 down the runs, fourth round running.** L6_unrolled's three runs are 0.2256 /
0.2257 / **0.2174**; L6_pfa's are 0.2197 / 0.2205 / 0.2241. On the run distributions
**L6_pfa owns the non-batched cell by ~2%**, not the 1.4% deficit the leaderboard shows.
This is the fourth consecutive round in which L=6 B=1 is decided by which entry drew the
lucky process, and the third in which the leaderboard reads the opposite way from the
medians. Used in §7.

B=64 also inverts on medians: L6_unrolled 0.2221 vs L6_pfa 0.2281 — the ordering survives,
the *size* of the gap (0.214 vs 0.222 = 3.7%) does not; honestly it is ~2.7%.

### L = 8 (volume 512)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L8_fusedaxes 0.552 µs** (batchsimd 0.564, radix8 0.570) | mkl_dfti 0.650 µs | **1.18×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.575 µs** (batchsimd 0.589) | mkl_dfti 0.708 µs | **1.23×** |
| B=2048 (32 MiB) | **L8_batchsimd 0.912 µs** (fusedaxes 0.939) | mkl2026_dfti 1.338 µs | **1.47×** |
| B=16384 (256 MiB) | **L8_fusedaxes 1.236 µs** (batchsimd 1.241) | mkl2026_dfti 1.778 µs | **1.44×** |

**B=1 moved a second time in seven rounds and it is a clean win.** 0.558 → **0.552**;
L8_fusedaxes' three runs are 0.5516 / 0.5569 / 0.5572, entirely below L8_batchsimd's
0.5643 / 0.5647 / 0.5935. This is not a lucky minimum. B=64 is the reverse: L8_fusedaxes'
0.575 min sits on runs of 0.5752 / 0.5937 / 0.6273 (9.1% spread) against batchsimd's
0.5886 / 0.5910 / 0.6151 — **on medians B=64 is a tie**, and the leaderboard ordering there
should not be quoted as a 2.4% lead. Same caution at B=2048: batchsimd's 0.912 min is 2.0%
below its own second run; read honestly ≈0.931, still first.

And see §3: **all three L=8 entries are now bit-identical in all four cells.** L=8 is one
algorithm implemented three times, and the 3.3% spread at B=1 is entirely non-arithmetic.

### L = 17 (volume 4913)

| case | fastest panel entry | best library | margin |
|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 15.182 µs** (winograd 16.527, rader 16.638) | fftw3_patient 81.736 µs | **5.38×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 16.662 µs** (winograd 17.794) | fftw3_estimate 81.890 µs | **4.91×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.311 µs** (winograd 21.382) | fftw3_estimate 83.441 µs | **3.92×** |
| B=2048 (307 MiB) | **L17_winograd 21.645 µs** (matrixsimd 21.757) | fftw3_patient 89.205 µs | **4.12×** |

**L17_winograd took its first cell, at B=2048**, and it holds on medians too (21.647 vs
21.939) rather than on a lucky draw. B=256 is now a 0.3% tie on minima and matrixsimd's on
medians (21.374 vs 21.449). Still the board's largest margin and still the geometry where
every library is clustered and nobody's prime-size path is special.

Against that: the *leader is flat for a fifth round* and so is everyone else — see §4, where
three independent uop deletions at this geometry, all three selected by their own tuners,
delivered −0.2%, −0.6% and −3.0%.

### L = 36 (volume 46656)

| case | fastest panel entry | best library | honest margin | leaderboard margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L36_mixedradix 119.951 µs** (pfa 122.099) | mkl_dfti 163.546 µs | **1.36×** | 1.36× |
| B=4 (5.70 MiB) | **L36_mixedradix 129.645 µs** (pencilfused 130.557) | mkl_dfti 175.597 µs | **1.35×** | 1.35× |
| B=32 (45.6 MiB) | **L36_mixedradix 166.402 µs** (pfa 168.583) | mkl_dfti 261.300 µs | **1.32×** | ~~1.57×~~ |
| B=256 (364 MiB) | **L36_pfa 185.973 µs** (mixedradix 186.317) | mkl_dfti 310.164 µs | **1.33×** | ~~1.67×~~ |

**The r5 MKL excursion is back, and the r7 verdict's warning applies verbatim.** `mkl_dfti`
at B=32 went 219.448 → **261.300 (+19.1%)** and at B=256 246.452 → **310.164 (+25.9%)**;
`mkl2026_dfti` moved +18.4% and +24.7% in the same cells. FFTW and ducc0 at those cells
moved +0.1% and −0.5%. The panel entries moved −0.0% and +1.3%. **The honest batched margins
are 1.32× and 1.33× — the same numbers r7 measured — not 1.57× and 1.67×.** Full accounting
in §3.

B=1 is the one place the panel went backwards at this geometry: 118.532 → **119.951
(+1.2%)**, against a 0.7% run spread, and it took the geometry's floor ratio from 1.43× to
**1.45×, the worst of the four original geometries**. The mechanism that plausibly did it is
named in §4.

### The four wave-2 geometries

| L | case | fastest panel entry | best library | honest margin |
|---|---|---|---|---|
| 13 | B=1 | L13_direct **5.725 µs** | mkl2026_dfti 7.582 | 1.32× |
| 13 | B=16 | L13_direct **6.039 µs** | mkl2026_dfti 7.690 | 1.27× |
| 13 | B=512 | L13_direct **8.110 µs** | mkl2026_dfti 9.197 | **1.13×** |
| 23 | B=1 | L23_rader **47.688 µs** | fftw3_estimate 260.939 | 5.47× |
| 23 | B=4 | L23_rader **49.557 µs** | fftw3_estimate 261.570 | 5.28× |
| 23 | B=128 | L23_rader **64.835 µs** | fftw3_estimate 265.391 | 4.09× |
| 45 | B=1 | L45_pfa **319.648 µs** | mkl_dfti 605.702 | 1.89× |
| 45 | B=2 | L45_pfa **327.211 µs** | mkl_dfti 606.793 | 1.85× |
| 45 | B=16 | L45_pfa **401.897 µs** | mkl_dfti 753.494 | **1.69×** (lb: 1.87×) |
| 64 | B=1 | L64_radix8 **966.824 µs** | mkl_dfti 1203.423 | 1.24× |
| 64 | B=2 | L64_radix8 **1020.108 µs** | mkl_dfti 1308.465 | **1.21×** (lb: 1.28×) |
| 64 | B=8 | L64_radix8 **1252.336 µs** | fftw3_patient 2186.793 | **1.55×** (lb: 1.75×) |

"Honest margin" uses r7's MKL number where MKL regressed this round (§3).

**Promotion ground 4 no longer discriminates at all: 19 of 19 entries beat every library in
every scored cell of their own geometry.** r7's single exception, `L13_rader` at B=512, was
fixed this round — 9.469 → **8.809**, now ahead of both MKL builds (9.197 / 9.275). That was
the r7 verdict's named L=13 priority and the entry hit it.

### Distance from each geometry's own port floor, B=1, at kclk = 2.89 GHz

| L | r7 ratio | **r8 ratio** | B=1 best | published floor |
|---|---|---|---|---|
| 23 | 1.14× | **1.14×** | 47.688 µs | 41.9 µs |
| 13 | 1.26× | **1.22×** | 5.725 µs | ~4.7 µs |
| 8 | 1.24–1.29× | **1.23–1.28×** | 0.552 µs (1595 cy) | 1248–1296 cy |
| 6 | 1.30× | **1.29×** | 0.217 µs (627 cy) | 486 cy |
| 17 | 1.32× | **1.31×** | 15.182 µs (43.9k cy) | 33.4k cy (11.55 µs) |
| 36 | 1.43× | **1.45×** ↑ | 119.951 µs | ~83 µs |
| 45 | 1.65× | **1.61×** | 319.648 µs | 199 µs |
| 64 | 1.81× | **1.76×** | 966.824 µs | ~550 µs |

Seven of eight improved or held; **L=36 is the only geometry that moved away from its
floor**, and it is now the worst ratio among the four original sizes.

---

## 2. What changed since `panel_r7`, per geometry

### Cell-level best (fastest correct entry in the cell, r7 → r8)

| L | B | r7 best | r8 best | Δ |
|---|---|---|---|---|
| 6 | 1 | 0.219 (unrolled) | 0.217 (unrolled) | −0.9% (inside spread; medians say pfa owns it) |
| 6 | 64 | 0.214 (unrolled) | 0.214 (unrolled) | flat |
| 6 | 4096 | 0.381 (pfa) | 0.389 (pfa) | **+2.1% — the round's largest panel regression** |
| 6 | 32768 | 0.563 (unrolled) | 0.565 (unrolled) | +0.4% (inside spread) |
| 8 | 1 | 0.558 (batchsimd) | **0.552 (fusedaxes)** | **−1.1%, and a lead change on distributions** |
| 8 | 64 | 0.587 (fusedaxes) | 0.575 (fusedaxes) | −2.0% on minima; a tie on medians |
| 8 | 2048 | 0.930 (fusedaxes) | **0.912 (batchsimd)** | −1.9%; recovers r7's flagged +2.2% regression |
| 8 | 16384 | 1.232 (batchsimd) | **1.236 (fusedaxes)** | +0.3% (inside spread) |
| 13 | 1 | 5.901 | 5.725 | −3.0% |
| 13 | 16 | 6.089 | 6.039 | −0.8% |
| 13 | 512 | 8.396 | 8.110 | −3.4% |
| 17 | 1 | 15.227 | 15.182 | −0.3% (fifth flat round) |
| 17 | 8 | 16.715 | 16.662 | −0.3% |
| 17 | 256 | 21.437 | 21.311 | −0.6% |
| 17 | 2048 | 21.661 | **21.645 (winograd)** | −0.1%, lead change |
| 23 | 1 | 47.717 | 47.688 | flat |
| 23 | 4 | 49.232 | 49.557 | +0.7% (inside spread) |
| 23 | 128 | 65.197 | 64.835 | −0.6% |
| 36 | 1 | 118.532 | 119.951 | **+1.2% — regression** |
| 36 | 4 | 129.562 | 129.645 | +0.1% |
| 36 | 32 | 166.444 | 166.402 | flat |
| 36 | 256 | 183.529 | 185.973 | **+1.3% — regression** |
| 45 | 1 | 328.012 | **319.648 (pfa)** | **−2.5%, lead change** |
| 45 | 2 | 325.296 | 327.211 | +0.6% (a tie; lead change on minima) |
| 45 | 16 | 406.571 | **401.897 (pfa)** | −1.1%, lead change |
| 64 | 1 | 995.782 | 966.824 | −2.9% |
| 64 | 2 | 1022.973 | 1020.108 | −0.3% |
| 64 | 8 | 1245.366 | 1252.336 | +0.6% (inside spread) |

### L = 6 — pinning falsified by its author's own criterion; rotation 0-for-8

Neither entry shipped a new B=1 kernel, correctly, per the r7 instruction. Both shipped
zero-arithmetic adopted mechanisms and both bets failed their own pre-registered tests.

**L6_pfa** adopted 64-byte kernel-entry pinning from L6_unrolled, pruned the grid 20 → 11
(deleting the falsified `fused_zx` zmm family and all NT kernels), and introduced **`_rot`,
per-volume scratch rotation** — the one new mechanism at this geometry. Its stated test was
explicit: *"If [B=64] stays 0.226 with the same pick, code layout is falsified for this
gap."* B=64 measured 0.2220 / 0.2281 / 0.2286 — **median 0.228, the three-round plateau,
unchanged** — with the pick reverting to plain `fused_pf`. **Code layout is falsified for
the L=6 B=64 gap**, and the entry's own named fallback suspect (L6_unrolled's radix-2-first
VD6 factorization) is now the standing hypothesis. Separately, **the `_rot` twins took zero
picks in all eight cells**: the per-volume alias phase-lock is a null on CLX, exactly as the
entry's parity-on-wallaby result warned it might be.

**L6_pfa's B=4096 is the round's largest panel regression: 0.381 → 0.389 (+2.1%)**, with the
*same pick* (`fused_pf_xa`) in all three processes and a 1.7% spread. Nothing on that path
changed but the arena and the deleted candidates. This is the fourth instance the panel has
recorded of an unexplained regression at an unchanged pick string following a refactor
*around* an untouched hot path (r4, r5, r7 L8_fusedaxes, now here). It still holds the cell,
by 0.5%.

**L6_unrolled** adopted `restrict` on every kernel signature from L6_pfa and moved
`create()`'s licence tail into the chosen kernel. Its B=4096 target was 0.381–0.390;
measured **0.391** with `fused_pfw`, i.e. −1.5% and just outside. The 3.7% gap to L6_pfa at
that cell is now 0.5% — but half of the closure is L6_pfa's regression, not L6_unrolled's
gain. Its second, sharper prediction — that with the pass-boundary loads hoisted its
*typical* run "should stop trailing L6_pfa's 0.2234" at B=1 — **failed**: its median is
0.2256 against L6_pfa's 0.2205. `restrict` did not touch the t1 store→load joint.

### L = 8 — the cell moved, and the geometry collapsed to one algorithm

**L8_fusedaxes took B=1 (0.552) and B=64 and B=16384, and it did it by adopting its rival's
codelet.** It replaced its 54-instruction DIF `dft8s` with L8_batchsimd's 52-instruction DIT
form (44 add/sub + 8 FMA, √2 twiddles FMA-folded), predicted 0.552–0.562 with the fused
pick, and measured **0.552 with `fused` 3/3**. That is a four-for-four prediction sheet
(B=64 0.575 against 0.575–0.590; B=2048 0.939 against 0.90–0.94; B=16384 1.236 against
1.23–1.27) and it is the round's cleanest single-substitution result: r7's batchsimd took
0.558 by porting fusedaxes' *shape*; r8's fusedaxes took 0.552 by taking back batchsimd's
*codelet*. The 48-instruction cut predicted ~2.9% at 1 FMA/cycle and delivered 3.3%.

**L8_batchsimd** shipped no kernel code at all — B=1 default LANEX2 → FUSED, mid-regime
hysteresis 3% → 6%, `s0w` offered at B=64, FUSED3 retired. Predictions: B=1 0.552–0.565 →
**0.564 with FUSED 2/3** (the LANEX3 pick still fired once, at 0.5935, so the hysteresis
change did not fully eliminate the lottery); B=64 0.583–0.592 if `s0w` is a wash → **0.589
with `s0` picked**, i.e. the L2-scale RFO-hiding argument was rejected by the node's own
tournament; B=2048 0.93–0.99 → **0.912, better than its own range, and it took the cell**;
B=16384 1.22–1.26 → 1.241. Its two borrowed audits both returned documented nulls.

**L8_radix8 got its round's one bet backwards, and the pick strings prove it.** It flipped
its B=1 default `2p` → `1f` on the strength of two rivals' node numbers. The node's three
runs read `2p` 0.5700, `1f` 0.5829, `1f` 0.5813 — **`1f` is 2% slower than the `2p` it
replaced, in this file, on this machine**, and the reported 0.570 came from the one run that
still picked the old default. Its own pre-registered branch fires: *"If it lands ≈0.572
instead, the 4% B=64-style gap between my 1f and their FUSED exists at B=1 too."* It does.
Adopting a rival's winning *configuration* did not transfer inside a different file even
though the arithmetic is now identical — which is the sharpest available statement of what
the remaining 3.3% B=1 spread at L=8 actually is. Its `3p-pfs` mid-regime hedge was also
declined (`1f-pfs` 3/3 at 0.618).

### L = 17 — three uop deletions, all selected, all worth nothing

This is the round's most important negative and it applies to all three entries at once.

* **L17_winograd** built kernel H, a component-split 17-point kernel that deletes ~70 stack
  uops per fused group (spill stores 276 → 243, spill loads 241 → 151 in `fused23_h8`).
  Same-core alternating wallaby A/Bs read **−9.0% at B=1** and −7.4% at B=64. It predicted
  *"~15.1–15.6 and the pick at h8 — the first genuine shot at the cell since r1"* on the
  stated rule that *"every uop deletion so far transferred AMPLIFIED to the node."* The node
  **picked `h8`** and measured **16.527 — −0.2%.** B=8 predicted 16.5–16.8, measured 17.794.
  It did take B=2048 (predicted 21.4–21.8, measured 21.645), so the round is 2-for-4, but
  its headline bet and its transfer rule are both falsified.
* **L17_matrixsimd** built model-chosen address-safe t1 twins: a 5120-byte plane stride
  selected by an explicit collision count (not rounded up — it showed the "natural" 4672 B
  padding is *worse* than dense), plus a per-volume de-aliased t1 base from a create-time
  table. Same-window wallaby A/Bs read **−14% at B=256 in every one of six rounds**. The
  node **picked the twins in every cell** (`addr-safe t1` in all four description strings)
  and measured B=1 −0.3%, B=8 −0.3%, **B=256 −0.6%, B=2048 +0.4%.**
* **L17_rader** replaced the 4×4 ymm plane transposes with 8×8 zmm blocks, deleting
  **~6.4k movement uops per volume (−31%)**, about half of them load/store slots. It
  predicted the node gain would *exceed* wallaby's −2.8% because CLX has one FMA unit and
  two load ports. Measured: B=1 **−3.0%**, B=8 −2.8%, and **B=256 +1.7%, B=2048 +1.4% —
  regressions.** Node gain equalled wallaby's at the small cells and reversed at batch.

Three entries, three different deletions (~70 stack uops/group, ~8 alias stalls/chunk,
6.4k movement uops/volume), all three selected by the node's own tournaments, and the
geometry moved −0.3% / −0.3% / −0.6% / −0.1%. **The r7 verdict's L=17 rule — "delete uops,
don't reschedule them" — is now falsified in the same way its three predecessors were.**
See §5.

L17_matrixsimd also withdrew its own r7 ask, correctly and for the right reason: it counted
the cosine-resident `cr` loop bodies in objdump and found gcc rematerializes the butterfly
adds (+26 instructions/chunk) to save 22 loads that were already hidden on ports 2/3. *"Fewer
loads is not fewer µops"* — dead by disassembly, no node run needed. That is the round's best
example of closing an item without spending a measurement on it.

### L = 36 — the three-round question got answered, by the implementer, without the monitor

**L36_pfa built the measurement the panel had been asking the monitor for since r5, put it
inside `fft3d_create()`, and rode it onto the leaderboard in its description string.** Four
fixed configurations timed at nv=1 steady state: `p1` (phase 1 alone), `p2w` (phase 2 alone,
warm), `p2wd` (same with deep prefetch), `fu` (full pf0 execute). Its pre-registered
discriminator was `fu − p1 − p2w`: **20–35 µs if the L2-thrash story is right; ≈0 if the
residual is front-end/scheduling.**

Node, B=1, three processes: `p1=90.2 p2w=35.2 fu=122.8` → **−2.6 µs**; `90.1 / 32.4 / 119.3`
→ **−3.2**; `92.9 / 34.6 / 123.9` → **−3.6**.

**`fu ≈ p1 + p2w`. There is no phase-boundary memory penalty at L=36 B=1 on the scoring
machine. The L2-thrash diagnosis, which two entries spent r6 and r7 building rejected
mechanisms against, is dead.** The residual 37 µs is inside the phases — front end,
scheduling, or port pressure — not the phase boundary. Consistent with this, `pf=5/6` (the
two-level deep-T1 staging built specifically for the thrash story) were **not picked in any
cell**, and the node's own probe prices the deep prefetch's tax at `p2wd − p2w` = +6.5 µs.

The rest of the geometry was flat by instruction. **L36_mixedradix** shipped deterministic
anti-alias pinning of the plane scratch (`pinD=2112`, always-on in the cached regime), pruned
21 → 10 compiled bodies (`.text` 173 → 110 KB), and predicted 108–117 µs if CLX aliasing was
biting or 117–121 if it was benign. Measured **119.951 with `pinD=2112`** — the benign branch,
four-for-four on its prediction sheet, but **the cell regressed 1.2% against a 0.7% spread**
and the mechanism is the only thing on that path that changed. Pin's node value is between
0 and −1.2%, i.e. it is a cost. **L36_pencilfused** shipped one thing, the
`#pragma GCC optimize("unroll-loops")` build-flag fix, predicted B=1 117–122, and measured
**124.233** — its own null branch, *"the build-flag theory is dead for L=36."* It is (§3),
and it also regressed at B=256 (186.452 → 189.566, +1.7%).

### L = 45 — the round's one unambiguous mechanism win, and a reversal reversed

**L45_pfa took all three cells back.** r7 had it 4.7% / 8.9% / 4.2% *behind* L45_mixedradix
and the r7 verdict used it as the panel's canonical cross-machine-reversal example. This
round it found, by objdump-diffing its rival's exec under node flags, that gcc's IVOPTS/LICM
was re-associating every y-pass load address, hoisting 45 loop-invariant `lea`s out of the
group loop, **spilling them to the stack (48 leas + 37 GPR spills) and reloading one before
every vector load** — a serial spill-reload dependency chain in the hot loop. A two-line
empty `asm` making the bases opaque took it to **4 leas and 0 spills**, and combined with
compile-time exec-variant specialization took the scoring path from 3535 to **3087
instructions against the rival's 3893**.

It predicted *"B=1 320–340, pick pw4-ip-pf0 … if it lands ≤328 I have the cell."* Measured
**319.648 with `pw4-ip-pf0`** — below its own threshold, all three cells taken, and the
leanness theory confirmed at ~7% (343.3 → 319.6). It also ruled out two rival explanations
with numbers first (`rename-registers`: table persisted; intrinsic loads: table persisted).
Its `scratchp` padded-scratch mode was not picked anywhere, so the alignment story is a null
here.

**L45_mixedradix** unbundled its memory mechanisms into orthogonal candidates and added two
new ones, including **`cpy`, an ERMS `rep movsb` plane copy intended to delete the cold-`out`
RFO term outright** — a genuinely novel idea and the only serious attempt anyone has made at
*deleting* traffic rather than overlapping it. Node picks: **`v1-pf0` at B=1/B=2 and
`v2-pf1-pfin-pfw` at B=16 — its r7 incumbents, unchanged.** Every new mechanism (pfw alone,
pkw, cpy, cpy-pfin) lost. Its pre-registered falsification branch fires exactly:
*"the bandwidth model is wrong in its actionable part … the next lever at L=45 is not memory
mechanisms at all."* ERMS no-RFO is dead on CLX for this workload.

### L = 64 — the corpus's largest untried structural move, built and closed

**L64_radix8 built the L2↔DRAM tiled structure** the r7 verdict named as "the largest untried
structural move on the board" (LITERATURE §4.3's re-opened case): pass A per z-octet slab
sized at 528 KB to be L2-resident on the node's 1 MB, with the x-FFT running in place inside
the slab, then a separate sequential pass-B z-sweep. Bit-identical to the fused structure by
construction, shipped as a tuner candidate, not a default.

**The node rejected it in all three cells** — picks `fused-plain+slabpf1` at B=1 and
`fused-pfw+slabpf1` at B=2 and B=8 — as wallaby had (15–23% behind there). The entry
pre-registered this branch: *"If the node also rejects it, LITERATURE §4.3's L2↔DRAM tiling
case is closed for L=64 on both machines."* It is. The `slabpf` lead-2 experiment was also
declined (lead 1 in all three cells). B=1 still improved 2.9% to 966.8 — that is the deeper
prologue-free fused path, not tiling.

**L64_blocked** independently answered the same ask *by construction rather than by building*:
its st=0 structure already tiles at 545 KB and runs 2⅓ of 3 axes inside the tile, and the
un-tileable remainder is forced by data dependency (x-stage-2 octet *d* needs one plane from
every group), so a second sweep is irreducible for any within-volume tiling. Its own new
mechanism, `pfb` (scratch-read prefetch adopted from radix8's slabpf), **took zero picks**,
firing its pre-registered reading: *"scratch-read latency is NOT the node gap."* Its B=2
prediction (`pf5` or `pf7`) also failed — `pf0` held — which by its own words means the
rival's B=2 pfw win does not transfer to its 8-sequential-stream pass B. Flat round
(+0.5% / +0.0% / −0.3%), third consecutive behind radix8 in all three cells.

### The regressions, named

* **L6_pfa at B=4096 — 0.381 → 0.389 µs, +2.1%.** The round's largest, same pick string in
  all three processes, 1.7% spread, refactor-around-an-untouched-hot-path class.
* **L36_pencilfused at B=256 — 186.452 → 189.566 µs, +1.7%** against a 0.4% spread; also
  outside its own predicted band at B=1 (124.233 vs 117–122) and at B=256 (vs 181–187).
* **L17_rader at B=256 and B=2048 — +1.7% and +1.4%**, both against ≤2.1% spreads, in a
  round whose only change was a 31% movement-uop deletion. The deletion is *negative* at
  batch.
* **L36_mixedradix at B=1 — 118.532 → 119.951, +1.2%** against a 0.7% spread, in the cell
  and regime where `pin` newly ships always-on.
* **L36_pfa at B=4 — 129.764 → 132.453, +2.1%** against a 0.8% spread, unchanged pick
  (`pw=4 inplace pf=0`); it also fell just outside its own null band (128–132).
* **L45_mixedradix at B=2 — 325.296 → 327.639, +0.7%**, and **L8_radix8 at B=64 —
  0.612 → 0.618, +1.0%** (the `3p-pfs` hedge was declined). Both inside spread.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed to build.** `build_errors.txt` is present and empty (0 bytes).

**Nothing crashed, hung, or timed out.** `failures.txt` does not exist, which for `sweep.sh`
means no backend invocation exited non-zero across 28 cases × 3 runs. `agents/exits.txt`
records `exit=0` for all 19 implementers.

**Nothing is missing.** All 19 `.c` files in `impl_8/` appear in the leaderboard. All 19 have
a `panel_r8` section in `strategies/` (3110 lines added across the 19 files). I cross-checked
every one of the **259 timed rows against the 259 correctness records**: no timed row lacks a
correctness verdict, and no correctness record is an orphan. Every panel entry is present at
every scored batch size of its own geometry, with no gaps. The 1392 `"does not support"`
lines in `timing.err` are entries correctly declining foreign geometries, not failures.

**Nothing failed correctness.** Every backend passes at every scored batch size, with
relative L2 error against numpy between **1.28e-16 and 6.39e-16** against a 1e-12 tolerance
— four orders of magnitude inside the gate. `check.py` compares the full `(B, L, L, L)`
array element by element, not a sample; `check.log` contains 259 `PASS` lines and zero of
anything else. **There is no fast wrong answer in this round.**

**No rule violations found.** `git status --porcelain` is empty for `driver.c`,
`fft3d_api.h`, `Makefile`, `sweep.sh`, `check.py`, `leaderboard.py`, `gen_input.py`,
`cases.txt`, `tryout.sh`, `PANEL_BRIEF.md`, `promote.sh`, `panel_round.js` and `sota/` — the
measurement apparatus is untouched. **Zero** occurrences of `fftw_*`, `DftiC*`, `DFTI_*`,
`ducc`, `#pragma omp`, `omp_*`, `pthread_*`, `dlopen` or `system()` anywhere in `impl_8/`.

### (a) The MKL baselines regressed 4–27% at the large-working-set cells, and it inflates five margins

Both MKL builds, in the same direction, at the same cells, on the same host as r7 (`p55n3`),
~1.6 hours apart. FFTW and ducc0 at those cells did not move.

| cell | ws | mkl_dfti r7→r8 | mkl2026 r7→r8 | fftw3_patient | ducc0 | best panel |
|---|---|---|---|---|---|---|
| L=17 B=256 | 38 MiB | **+7.5%** | +1.3% | +0.2% | −1.0% | −0.6% |
| L=23 B=128 | 48 MiB | **+4.3%** | +0.8% | −0.0% | −0.9% | −0.6% |
| L=36 B=32 | 46 MiB | **+19.1%** | **+18.4%** | +0.1% | −0.5% | −0.0% |
| L=36 B=256 | 365 MiB | **+25.9%** | **+24.7%** | −3.2% | +1.6% | +1.3% |
| L=45 B=16 | 44 MiB | **+10.9%** | **+10.4%** | +0.0% | −0.8% | −1.1% |
| L=64 B=2 | 16 MiB | **+6.3%** | +4.0% | +4.4% | +0.2% | −0.3% |
| L=64 B=8 | 64 MiB | **+27.5%** | **+27.5%** | −2.3% | +0.0% | +0.6% |

Every other cell (21 of 28) has both MKL builds inside ±2.7%. The effect appears only at
L ≥ 17 with a multi-MiB working set; L=6 B=32768 (216 MiB) and L=8 B=16384 (256 MiB) are
flat, so it is not simply "large working set" — it correlates with the *volume* size as well.

**This is the third occurrence.** r5 recorded a 17–25% MKL excursion at L=36 B=32/B=256, r7
recorded it reverting to r4 levels and called it "transient, specific to MKL, and still
unexplained." It is back, at the same cells and larger, and it now spans five geometries.
Two independently-built MKL versions moving together while two other libraries and nineteen
panel entries hold flat points away from the machine and toward something in MKL's own
buffer allocation or threading-runtime initialisation that differs between slurm jobs.

**Consequence for this round's reporting, applied throughout §1:** the L=36 batched margins
are 1.32× and 1.33× (not 1.57× / 1.67×), L=45 B=16 is 1.69× (not 1.87×), L=64 B=2 is 1.21×
(not 1.28×) and L=64 B=8 is 1.55× (not 1.75×). **Anyone quoting the leaderboard figures for
those five cells is quoting an MKL artifact.** Recommendation for the harness: re-run the
`sota/` binaries alone, twice, in separate jobs, and record whether the excursion is
job-to-job or persistent — this has now cost three rounds of verdict text.

### (b) L = 8 has collapsed from three algorithms to one — this is the round's sharpest finding

**`L8_batchsimd`, `L8_fusedaxes` and `L8_radix8` produce bit-identical output at all four
scored batch sizes.** The correctness JSONs agree to the last digit of `rel_l2`, `max_abs`
*and* `rel_max` simultaneously in every cell, over 512·B complex doubles.

In r7 the fingerprints were: batchsimd ≡ radix8 at B=64 and B=16384 only, with all three
distinct at B=1 and B=2048. The r7 verdict cited exactly that divergence as evidence that
batchsimd's B=1 win was "a real structural change and not a relabel." This round the
divergence is gone, and the records explain it: **L8_fusedaxes adopted L8_batchsimd's
52-instruction DIT codelet** (its own record notes the fingerprint moving out of the r1–r7
2.29–2.33e-16 band to 2.267e-16 — "expected and healthy"), **L8_radix8 already carried the
same 52-instruction codelet class and flipped its B=1 default to the same fused shape**, and
batchsimd's `MODE_FUSED` was already fusedaxes' structure. All three now publish 1248 vector
FP ops per volume and all three run the fused shape at B=1.

This has two consequences.

1. **`CURATION.md` rule 2 does not select a third L=8 entry**, on the same reasoning the r7
   verdict applied to L=23. §7 acts on it.
2. **The correctness check at L=8 is now one independent verification, not three.** The
   check against numpy still holds, so the results stand; the cross-entry redundancy the
   panel normally gets does not.

There is a positive reading too, and it is the most useful thing at this geometry: with the
arithmetic held bit-identical, **the 3.3% B=1 spread between the three entries (0.552 /
0.564 / 0.570) is a pure measurement of non-arithmetic cost** — prefetch branch shape,
scratch placement, code layout. That is a cleaner isolation than any A/B the panel has run
at L=8, and it is the number the outstanding alias counter (§6) has to explain.

### (c) The `-funroll-loops` "build-flag gap" from r7 does not exist, and six entries spent the round on it

The r7 verdict elevated L45_pfa's discovery — *"the scored build does not carry
`-funroll-loops`, which `tryout.sh` does"* — to "a methodology finding worth more than most
kernels", stated that "**both findings apply to every entry in the panel**", and listed the
resulting pragma among the round's most portable results.

**It is false.** `Makefile:15` reads

```
CFLAGS ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops
```

`sweep.sh` invokes `make` with no `CFLAGS` override, and the flag appears verbatim on every
compile line in **`slurm-438486.out` (r5), `slurm-438522.out` (r7) and `slurm-438524.out`
(r8)**. `git log -- Makefile` shows the file last changed in `9db2832` (2026-08-21), before
r7 ran. The scored build has carried `-funroll-loops` for at least three rounds.

The consequences this round are measurable:

* **Six entries ran the A/B and every one measured a null**, which is the only result
  possible: L8_batchsimd (0.305 vs 0.307 µs), L8_fusedaxes ("already closed"), L13_rader
  (3.194 vs 3.186), L23_matrixsimd (a wash — and it explicitly retracted its own first
  "6.3% gap" reading as a wallaby clock artifact), L13_direct ("no-op on wallaby"),
  L36_pfa/L45_mixedradix (checked and declined).
* **Two entries shipped the pragma into `impl_8/` on a false premise** — `L13_direct.c:94`
  and `L36_pencilfused.c:221`. L36_pencilfused's *entire round* was that one change; it
  predicted 117–122 µs at B=1 and measured 124.233, its own "the build-flag theory is dead"
  branch. It is dead because there was never a gap.
* **L17_rader measured the pragma form as actively harmful** in its own file: same-window
  three-way, unroll-on 9.29/9.37, unroll-off 9.36/9.49, **pragma 9.55/9.62 (~2% worse)** —
  because `optimize()` rebuilds the whole per-function option set, not just the named flag.
  `L13_direct.c` and `L36_pencilfused.c` now carry that form. `L45_pfa.c:129` also still
  carries it, and its record this round still asserts the pragma's "+10% node-flag benefit
  stands" — that attribution is wrong; **L45_pfa's real r8 gain is the opaque-base asm
  barrier and the exec specialization**, which it measured separately and which are genuine.

I cannot reconstruct what L45_pfa actually compared in r7 (its own r8 record now says
`rename-registers` does not reproduce the effect either). What is certain is that the scored
build was never missing the flag. **Recommendation: strike the build-flag item from the r7
verdict's findings, tell the panel, and have the two entries carrying the pragma A/B it out
before r9** — on L17_rader's evidence it is a small tax, not a fix.

### (d) In six cells the number in the leaderboard came from a plan variant whose output was never checked

`sweep.sh` runs each backend three times, reports the **minimum**, and runs `check.py` on the
output left by the **last** run. After stripping clock-probe and probe-timing digits, which
differ harmlessly between runs, the genuine cases are:

| cell | reported (run) | variant **timed** | variant **checked** (run 3) |
|---|---|---|---|
| L17_matrixsimd B=256 | 21.311 (r1) | X-first, **no deferred-Z** | X-first, **deferred-Z** |
| L17_matrixsimd B=2048 | 21.757 (r1) | X-first, **deferred-Z** | X-first, **no deferred-Z** |
| L23_rader B=1 | 47.688 (r2) | pinned, **pipelined** | pinned **two-sweep** |
| L36_pfa B=1 | 122.099 (r1) | pw=4 inplace **pf=0** | pw=4 inplace **pf=4** |
| L6_pfa B=64 | 0.222 (r2) | variant=**fused_pf** | variant=**fused_sp2_pf_xa** |
| L8_radix8 B=1 | 0.570 (r1) | **avx512-2p** | **avx512-1f** |

Six, down from r7's seven, and the composition has changed in a way worth recording.

**`L23_matrixsimd`, r7's "worst offender on the board" with all three cells exposed, is now
clean in all three.** That was its entire round: a canonical-order deterministic hysteresis
(26 candidates, 2% displacement bar, incumbent-first) plus two-sweep ranking, verified 4/4
identical picks on a contended wallaby. It worked, and it is the only entry that set out to
fix a harness-visibility problem this round and did. Credit is due even though §7 does not
promote it.

**`L23_rader` acquired the exposure `L23_matrixsimd` shed** — three different picks across
three processes at B=1. Its own new mechanisms are the cause: `rp-t1` and the joint
`(variant, pf, pw)` grid added genuinely close candidates. It is worth noting that at B=128
the joint grid *paid*: run 3 picked `rp-t1, pf=2, pw=1`, was the fastest run **and** the
checked one, and produced the cell best (64.835 vs 65.505 / 65.690 for two-sweep) — a
combination a stage-1-then-grid tuner could never have found. The mechanism is real; the
pick instability is the price.

`L8_radix8`'s flip is the most consequential for interpretation: the reported 0.570 is the
**old** `2p` default, and the `1f` it shipped as its new default measured 0.581–0.583. See
§2. `L17_matrixsimd`'s two flips are the same deferred-Z exposure r5 and r7 recorded, and its
class members remain cmp-verified. `L36_pfa`'s pf=0-vs-pf=4 flip changes prefetch policy, not
arithmetic; the pf=4 runs read 125.113 / 123.739, so reading the cell as ≈123.7 changes no
ranking.

### (e) Four headline cells rest on a minimum that is an outlier against their own other two runs

Minimum-of-three is the standing convention and I am not overriding it, but these should not
be read as confirmed:

* **L6_unrolled B=1**: 0.2174 / 0.2256 / 0.2257 — min **3.6% below** the second-best run.
  L6_pfa's are 0.2197 / 0.2205 / 0.2241. **On distributions L6_pfa owns this cell by ~2%.**
  Fourth round running.
* **L6_unrolled B=64**: 0.2142 / 0.2221 / 0.2290 — min 3.5% low. Reported 0.214; honestly
  ≈0.222, against L6_pfa's ≈0.228. Ordering survives, magnitude does not.
* **L8_fusedaxes B=64**: 0.5752 / 0.5937 / 0.6273 (9.1% spread) — min 3.1% low. Against
  batchsimd's 0.5886 / 0.5910 / 0.6151, **B=64 is a tie on medians**, not a 2.4% win.
* **L8_batchsimd B=2048**: 0.9124 / 0.9314 / 0.9538 — min 2.0% low. Read honestly ≈0.931 it
  is still first (fusedaxes ≈0.946), so the rank survives.
* **L23_matrixsimd B=128**: 66.223 / 67.722 / 67.946 — min 2.2% low; it is second either way.

L8_fusedaxes' B=1 win by contrast survives every reading: 0.5516 / 0.5569 / 0.5572, entirely
below every rival run in the cell.

### (f) Bit-identity fingerprints, fifth round running

When `rel_l2`, `max_abs` and `rel_max` all agree to the last digit, two entries are producing
bit-identical output over millions of complex doubles.

* **All three L=8 entries ≡ each other at all four batch sizes** — new this round, §3b.
* **L23_matrixsimd ≡ L23_rader at all three batch sizes** — unchanged from r7; L=23 remains
  one algorithm implemented twice.
* **L36_pfa ≡ L36_pencilfused at B=32 and B=256**, distinct at B=1 and B=4 — unchanged since
  r4, still the acknowledged `istream`-is-a-translation-of-inplace case.
* **All three L=17 entries remain numerically distinct.** **The two L=13 entries are
  distinct** (2.9e-16 vs 4.0e-16) — dense-folded vs Rader-13 really are two algorithms.
  **L=45 distinct** (4.10e-16 vs 4.04e-16) and **L=64 distinct** (4.17e-16 vs 4.46e-16),
  including L64_radix8's tiled candidate, which is bit-identical to its own fused path by
  construction.

So of the eight geometries, **two have collapsed to a single algorithm (L=8 and L=23) and a
third is half-collapsed (L=36 at batch).** Eight geometries, nineteen entries, sixteen
distinct algorithms.

### (g) The harness floor is absent from the five largest cells

`sweep.sh`'s rule skipping `baseline_matrix` when `L³·B > 2e6` again removes the library-free
reference from L=6 B=32768, L=8 B=16384, L=17 B=2048, L=36 B=256 and L=64 B=8. Nothing scored
depends on it, but the round's five biggest cells have no harness floor. Noted, as in r5 and
r7; not a change this round.

### (h) Library measurement quality

Two library run-spreads are large enough to mention: `ducc0_c2c` at L=6 B=1 reads 33.3%
(and its 3.868 µs min is 15% below r7's, which is why its L=6 row moved), and
`fftw3_measure` at L=45 B=1 reads 27.7%. Neither affects any panel conclusion — both are
5–90× off the pace in those cells — but the ducc0 L=6 numbers should not be quoted as a
baseline.

### (i) Provenance

`impl_8/` and `results/panel_r8/` are **untracked** at the time of writing. `CURATION.md`
records that panel_r1's eleven implementations were lost exactly this way. **Commit
`impl_8/`, `results/panel_r8/`, the strategy records and the promoted exemplars before round
9 starts.**

---

## 4. Claimed numbers versus measured numbers

The development machine (`wallaby`, Xeon Gold 6448Y, Sapphire Rapids, two 512-bit FMA units,
2 MB L2/core, 60 MB L3) is not the scoring machine (Gold 5218, Cascade Lake, one 512-bit FMA
unit, 1 MB L2/core, 22 MB L3), and wallaby additionally swings ~1.95× between its base and
turbo clock states between sessions. The panel's standing calibration puts the full
wallaby-to-node span at ~2.9× — MKL alone spans that much between these two machines.

### Ratios of claimed (wallaby) to measured (node), same entry, B=1

| entry | claimed (wallaby) | measured (node) | ratio | r7 ratio |
|---|---|---|---|---|
| L6_unrolled | 0.129 | 0.217 | 1.68× | 2.03× |
| L6_pfa | 0.132 | 0.220 | 1.67× | 1.73× |
| L8_batchsimd | 0.305 | 0.564 | 1.85× | 1.83× |
| L8_fusedaxes | 0.317–0.318 | 0.552 | 1.74× | 1.79× |
| L8_radix8 | 0.308 | 0.570 | 1.85× | 1.86× |
| L13_direct | 3.179 | 5.725 | 1.80× | 1.90× |
| L13_rader | 3.185 | 6.074 | 1.91× | 1.89× |
| L17_matrixsimd | 8.77–8.81 | 15.182 | 1.72–1.73× | 1.54–1.75× |
| L17_winograd | 8.069 (7.731 forced) | 16.527 | 2.05× | 1.94× |
| L17_rader | 8.660 | 16.638 | 1.92× | 1.93× |
| L23_matrixsimd | 21.03 | 48.184 | 2.29× | 2.24× |
| L23_rader | 22.06 (21.37 best) | 47.688 | 2.16–2.23× | 2.21–2.29× |
| L36_mixedradix | 51.905 | 119.951 | 2.31× | 2.15–2.19× |
| L36_pfa | 52.96 | 122.099 | 2.31× | 2.32× |
| L36_pencilfused | 52.7 | 124.233 | 2.36× | 2.33× |
| L45_mixedradix | 166.8 | 322.127 | 1.93× | 1.96× |
| L45_pfa | 171.8 | 319.648 | 1.86× | 2.01× |
| L64_blocked | 664.0 | 1092.567 | 1.65× | 1.63–1.65× |
| L64_radix8 | 555.6 | 966.824 | 1.74× | 1.83–1.95× |

**Every ratio lies inside the 1.65–2.36× machine band, comfortably inside the ~2.9× MKL spans
between these machines, and every one is within 0.2 of its own r7 value. None of these is an
implementer error.** The band is tight and stable enough now that it should be used as the
panel's default anchor for r9 predictions, as the r7 verdict instructed — several entries did
exactly that this round and it worked (L45_pfa, L13_rader, L64_radix8 all anchored on ratios
rather than floors, and all three landed inside their bands).

**Where a claim deserves scoring it is therefore on *direction and mechanism*, not magnitude
— and this round the machine difference is *not* the explanation for the round's headline
misses.** The four largest gaps below are all cases where the mechanism was **selected by the
node's own tuner** and still did nothing, which no clock or cache difference can account for.

### The four mechanism-transfer failures (not machine effects)

1. **L17_winograd's kernel H: wallaby −9.0%, node −0.2%.** Same-core alternating pairs on
   wallaby, three reps, `h` 7.757/7.766/8.046 vs `g` 8.484/8.548/8.571. Node picked `h8` and
   read 16.527 against 16.567. Its stated transfer rule — *"every uop deletion so far
   transferred AMPLIFIED to the node"*, on the basis that the node is 4-wide with one FMA
   unit against wallaby's 6-wide with two — predicted the node gain would *exceed* −9%. It
   was 0.2%. The rule is dead.
2. **L17_matrixsimd's address-safe twins: wallaby −14% in six of six same-window rounds at
   B=256, node −0.6%.** The twins were picked in all four cells. The collision model behind
   them is careful and its negative sub-result (naive 73-line padding scores *worse* than
   dense) is genuinely useful; the prize it was aimed at is not there on this machine.
3. **L13_direct's streaming prefetch exec: wallaby −23.8% on its honest B=2048 proxy, node
   ≤ −2.5%.** It predicted B=512 at 6.5–7.5 µs (from 8.396) and a margin move from 1.08× to
   ~1.3×. Measured **8.110 with `X-first+pf` picked** — the exec ran. Decomposing against its
   own pad-only cells (B=1 −3.0%, B=16 −0.8%), the prefetch exec is worth roughly 0 to −2.5%
   on the node against −24% on wallaby. The margin went to 1.13×, not 1.3×. Its own diagnosis
   anticipated the shape of this (*"the input-side prefetch carries most of the wallaby
   win … the node's CLX has less MLP"*) but predicted the node would care *more*, not less.
4. **L45_mixedradix's ERMS `cpy`: wallaby +8.5% to +15% (a known cost), node rejected in all
   three cells.** This one was honestly framed in advance — the entry shipped a mechanism its
   own machine cannot price, exactly as the panel's protocol says to — so it scores as a
   clean null, not a miss. The traffic model that motivated it (7.3 MB of L3 traffic per
   transform at B=1, RFO the only deletable term) is now unsupported in its actionable part.

### The predictions that were right, and they are the round's best work

1. **L45_pfa went four-for-four, named its own threshold, and took the geometry.** Predicted
   *"B=1 320–340, pick pw4-ip-pf0 … if it lands ≤328 I have the cell"* → **319.648, pick
   `pw4-ip-pf0`, cell taken**; B=2 "expect ≈ B=1" → 327.211 (the r7 +11 µs B=2 anomaly did
   disappear, confirming it was per-volume dispatch); B=16 "pick ip-pf3, 400–425" →
   **401.897 with `pw4-ip-pf3`**. It also ruled out two alternative explanations for the lea
   table *before* claiming the mechanism, and it stated in advance that a null would kill the
   leanness theory. This is the strongest prediction sheet of the round and the mechanism —
   **an empty two-line `asm` barrier defeating gcc's IVOPTS/LICM address re-association** —
   is portable to every entry whose codelet macros are base + compile-time constant.
2. **L36_pfa built the panel's missing measurement itself and pre-registered both branches.**
   `fu − p1 − p2w ≈ 20–35 µs` if thrash, `≈ 0` if front-end. Node: **−2.6, −3.2, −3.6 µs**
   across three processes. Its B=1 null-branch prediction (pf=0, 119–122) also landed exactly
   (122.099, pf=0), and B=32/B=256 landed inside their bands. Getting a three-round-blocked
   question answered by putting the probe in `create()` and routing it through
   `fft3d_description()` is a piece of infrastructure the whole panel should copy.
3. **L8_fusedaxes went four-for-four and took two cells** (§2), on the cleanest
   single-substitution reasoning of the round: 48 fewer FP instructions at 1 FMA/cycle
   predicts −2.9%, measured −3.3%.
4. **L64_radix8 and L64_blocked both pre-registered the tiling null and both closed it
   cleanly.** radix8 built the structure, predicted "below 950 if tiling is real, 960–1010 if
   fused holds", measured **966.824 with fused** — inside the null band, tiling rejected in
   all three cells. blocked reached the same place by a data-dependency argument and
   independently nulled `pfb`. Two entries, one corpus question, closed from both sides.
5. **L36_mixedradix went four-for-four on its null branch** (`pinD=2112` in the string,
   119.951 inside 117–121), and **L36_pencilfused's single prediction resolved on its own
   stated dead branch** — which turned out to be dead for a reason nobody expected (§3c).
6. **L23_matrixsimd predicted "B=1 ≈47.7, B=4 ≈49.4, B=128 ≈65.6, and the same variant in
   every process"** and measured 48.184 / 49.701 / 66.223 with **zero timed≠checked flips**.
   The times are ~1% high; the protocol claim, which was the point of its round, is exactly
   right.
7. **L17_matrixsimd withdrew its own outstanding monitor ask on disassembly evidence**
   (§2) — the cheapest correct action anyone took this round.

### The predictions whose direction failed

8. **L6_pfa's pinning bet, by its own pre-registered criterion** (§2): B=64 stayed on the
   0.226–0.228 plateau with a *plainer* pick. Code layout is not the L=6 B=64 gap.
9. **L8_radix8's B=1 default flip made its own cell 2% slower** (§2), and its `3p-pfs` mid
   hedge was declined. Both bets were adoptions of rivals' node-proven configurations; both
   failed to transfer inside a different file at bit-identical arithmetic.
10. **L17_rader's transposes regressed the two batched cells** (+1.7%, +1.4%) while
    delivering the predicted sign only at B=1/B=8, and at wallaby's magnitude rather than
    above it.
11. **L64_blocked's B=2 prediction** (`pf5` or `pf7`) failed; `pf0` held, firing its own
    "the rival's pfw win does not transfer to my pass B" branch.
12. **L13_rader's B=16 prediction** (5.9–6.3, on wallaby's −19% pw-at-L2-streaming
    measurement) missed at 6.683, though the cell still improved 8.2% by killing staged
    stores — the biggest single-cell panel improvement of the round.

### A methodology correction: the r7 build-flag finding was wrong

See §3c in full. The r7 verdict's "methodology finding worth more than most kernels" — that
the scored build lacks `-funroll-loops` — is contradicted by `Makefile:15` and by the compile
lines in r5's, r7's and r8's own slurm logs. Six entries spent measurement effort on it this
round and all six correctly measured a null; two shipped a pragma that a third entry
independently measured as a ~2% tax. **The lesson is not about unroll flags. It is that a
verdict that promotes a single entry's cross-machine claim to a panel-wide instruction
without checking the build line costs the whole panel a round.** I have tried to avoid
repeating that here: every cross-cutting claim in this document is checked against the
harness or the raw JSONs, and §3a and §3c say where the previous verdict was wrong.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### The primary move: §4.3 — the re-opened L2↔DRAM tiling case is **CLOSED at L = 64, on both machines**

§4.3's inset re-opened the fusion question in one regime: every panel experiment had fused
across an **L1↔L2** boundary (2.6× bandwidth gap), while the untested case was **L2↔DRAM**
(7× gap), where Intel's manual, Alappat et al. and the L3-Fusion result all independently
recommend *"tile the batch so a tile fits L2, then run all three axes inside the tile."* The
r7 verdict named it **"the largest untried structural move on the board"** and said L=64 is
the only geometry where the experiment is meaningful.

It was built. **L64_radix8's `tiled` structure** is that construction, specialized: an
8-slab decomposition at 528 KB per slab (against the node's 1 MB L2), with the y-FFT and the
whole x-FFT running in place inside the L2-resident slab, then a separate sequential z-sweep.
Bit-identical to the fused path by construction; offered as a tuner candidate against the
incumbent, with a 2% displacement bar. Wallaby put it 15–23% behind (expected: its 2 MB L2 +
60 MB L3 make the fused x-lines nearly free). **The node rejected it in all three cells**, in
every process, picking `fused-plain+slabpf1` at B=1 and `fused-pfw+slabpf1` at B=2 and B=8.

**L64_blocked reached the same conclusion structurally and it is the more transferable half
of the answer:** its st=0 structure already tiles at 545 KB and runs 2⅓ of 3 axes inside the
tile, and **the un-tileable remainder is a data dependency, not a schedule choice** — in any
radix-p² split of the x-DFT, one stage's octets span all tiles, so a second full sweep of the
volume is irreducible for any within-volume tiling. That argument generalizes beyond L=64
and explains why the recommendation does not do at 64³ what it does for the GPU workloads
that motivated it.

So: **§4.3 is now settled in both its regimes.** r3 settled L1↔L2 (single-digit percent,
sometimes negative; store *order* worth more than pass count). r8 settles L2↔DRAM at the only
size where it applies: **the tiling exists, it is buildable, it is bit-exact, and the node
does not want it.** Tolmachev's rule survives with a second boundary attached. The corpus
inset should be marked closed and the "largest untried structural move" language retired.

### The second move: §4.5 — padding and 4K aliasing does **not** generalize from L = 23

r7's headline mechanism was L23_rader's one-constant t1 plane-stride pad, worth **−25 to
−30% at B=1**, isolated with a stride-only A/B. The r7 verdict called it "the round's most
transferable mechanism". Seven entries transferred it this round, at five geometries, in
five different forms. The results:

| entry | form | node result |
|---|---|---|
| L17_matrixsimd | model-chosen 5120 B stride + per-volume de-aliased base | **picked in all 4 cells, −0.6% to +0.4%** |
| L23_rader | row-padded t1 (`rp`, 23→24 complex) | picked at B=128 only, **−1.0%** |
| L23_matrixsimd | z-extent pad (`za`, +1.2% FP) | **not picked** |
| L13_direct | t1 x-plane stride 338 → 344 | bundled; ≤ −3.0% at B=1 |
| L13_rader | ported L23's pad directly (PS 176 → 184) | **tested, wash at B=1, worse at B=512; not shipped** |
| L36_mixedradix | execute-time plane-scratch pin, `pinD=2112` | **shipped always-on, B=1 +1.2%** |
| L6_pfa | per-volume scratch rotation (`_rot`) | **zero picks in 8 cells** |

**Seven ports, one modest positive (−1.0%), one probable negative, five nulls.** The L=23
result stands — it was isolated properly and L=23 remains the board's tightest floor ratio at
1.14× — but it is a **property of that plan's specific stride collision, not a rule the other
geometries can apply.** Two of the negative results are individually valuable and belong in
the corpus:

* **L17_matrixsimd's collision model shows that "round the stride up to whole cache lines"
  can make 4K aliasing *worse*:** for its 17³ layout the dense 4624 B stride scores 8
  collisions, the natural 4672 B (73 lines) scores **31**, and only a model-chosen 5120 B
  (80 lines) reaches ≤4. Fixing line splits and fixing aliasing pull the stride in opposite
  directions. §04's "pad to an odd number of cache lines" rule is necessary, not sufficient.
* **L17_matrixsimd also established that the relevant heap offsets are not a lottery:**
  `(in − t1) mod 4096` and `(out − t1) mod 4096` were *identical across 24 processes* on
  wallaby, because glibc mmaps the large allocations and relative offsets are not ASLR'd. So
  address pathologies are a fixed property of a build, invisible to round-over-round
  comparison — which also means the panel's recurring "allocation lottery" explanation for
  unexplained ±3% wobbles is probably wrong.

**And the counter that would settle all of this has now been requested by five entries across
four rounds and still has not been run** (§6). It is one `perf stat` invocation.

### §4.6 — model versus search: the search *protocol* was again worth more than the kernels

Fifth consecutive round in which tuner and build-inspection work outscored kernel work.

* **L45_pfa's opaque-base `asm` barrier** — found by objdump-diffing a rival's exec under
  node flags, worth ~7% and the geometry — is a *compiler* finding, not an algorithm one.
  gcc re-associated `pcol + n*832` into a hoisted base table, spilled 37 GPRs, and reloaded
  one before every vector load; two lines of empty asm deleted 48 leas and the whole serial
  dependency chain. Portable to any entry with base + compile-time-constant codelet macros.
* **L23_matrixsimd's canonical-order deterministic hysteresis** eliminated the board's worst
  timed≠checked exposure (3 cells → 0) with no kernel change.
* **L23_rader's joint (variant, pf, pw) grid** found a combination stage-1 ranking could
  never select — plain X-first + pf=2 + pw=1 — and it won B=128 on the node. Racing knobs
  only on the stage-1 winner is a documented, repeated mistake.
* **L45_pfa's pool-ordering lesson**: with a hysteresis tuner, within-band ties go to the
  earlier candidate, so *list order is a policy statement*; listing combos before singles
  installed a worse combo at B=8.
* Two more re-triggerings of documented measurement traps: the raw-ssh missing-`cd` trap
  fired for the **third** consecutive round (L17_matrixsimd, nearly verifying against a stale
  r5 input), and L17_winograd documented a **second** instance of a "poisoned core" giving one
  binary a reproducible +17% — its w4 conclusion would have inverted had it tested on one core.

Search still beats model, and for the fifth round running the binding constraint is whether
the measurement inside the search is honest.

### §4.2 / §4.1 — the "delete uops" rule is falsified at L = 17, and spill counts do not predict

The r7 verdict's L=17 synthesis — *"the only mechanism that has moved an L=17 cell in three
rounds is uop deletion … delete uops, don't reschedule them"* — was adopted verbatim by all
three L=17 entries as the round's design principle. All three deletions were selected by the
node's own tuners. All three delivered ~0 (§2, §4).

This retires a rule rather than establishing one, and it also lands on **§4.1**, the
register/spill-budget question: L17_winograd's kernel H cut static spill *stores* 276 → 243
and spill *loads* 241 → 151 in its hot function and bought −0.2% on the node. Frigo's
UltraSPARC datum (50–100% from lexical scoping, "entirely register spills and reloads") does
not describe this microarchitecture at this size. **On the Gold 5218 at L=17, neither uop
count, nor spill count, nor scheduling, nor address alignment predicts the cell.** Four
mechanism classes have now been falsified at this geometry — the same position L=6 reached
one round earlier with three.

The §4.2 algorithm question itself is unchanged and unchallenged: dense nested
cyclic/negacyclic leads at L=17 (three cells), the hand-derived Winograd module is second and
now takes the fourth, Rader-17 is third in all four; dense wins at 13, 17 and 23.

### §4.8 item 6 — AVX-512 on the scoring part

No movement, and none needed. `kclk = 2.89 GHz` re-confirmed in every L=6 cell by both
entries' probes (clkS256 3.89 / clkD256 2.89 / clkS512 2.89), matching Intel's table for this
part exactly, third round running. The width question at L=6 stays closed: the zmm families
are now deleted from the source rather than raced.

---

## 6. The single highest-value thing the next round should attack, per geometry

**First, the item that outranks all eight of them.** Five separate one-run `perf stat` asks
are now outstanding — L=6 (r5, r7, r8), L=8 (r7, r8), L=17 (r8), L=23 (r7, r8), L=36 (r6, r7,
r8, and it is the fourth time of asking), L=45 (r8), L=64 (r8). Every one costs a single run.
Between them they adjudicate at least six mechanisms that the panel is otherwise guessing at,
and their absence is now the **binding constraint on the whole board**: three geometries
(L=6, L=8, L=17) have exhausted their mechanism lists and every entry's "Next" section is
gated on a counter. L36_mixedradix notes it cannot run them itself
(`probe_node.sh` requires `FFT_MONITOR=1`). Either run them, or adopt **L36_pfa's answer** —
put the discriminating measurement inside `fft3d_create()` and route it out through
`fft3d_description()`, which cost that entry ~7 ms of unscored setup and settled a
three-round question in one leaderboard line. **The second option requires nothing from the
monitor and every entry can do it. It should become the panel's default.**

Consolidated counter list, in cost order:

1. `perf stat -e ld_blocks_partial.address_alias,ld_blocks.store_forward,cycles` — L8
   `-DL8_VARIANT=0` vs `12` at B=1; L17 `-DL17_FORCE=38` vs `48` at B=1; L23 `L23R_FORCE=6`
   vs `20` at B=1. One recipe, three geometries, settles §4.5 for all of them.
2. `perf stat -e idq.dsb_uops,idq.mite_uops,cycles` — L36 at B=1 and L45 at B=1. Both entries
   now have front-end as their *only* surviving hypothesis.
3. `perf stat -e uops_issued.any,cycles,resource_stalls.rob,cycle_activity.stalls_mem_any,ld_blocks.store_forward`
   — L6 `L6_FORCE=fused` vs `zff` at B=1.
4. Forced one-flag pairs: `FFT36_PIND=-1/0/2112/3904` at B=1; `-DL8_CODELET=54` at B=1;
   `FFT64R_TUNEDBG=1` one run per B; `-DPPITCH=48` and `FFT45_CPY=1` at L=45; `FFT64B_NOHP`.

### L = 6 — the mechanism list is empty. Decide the slot, not the kernel.

Five falsified mechanisms now (r4 port-5/uop-mix, r5 OoO window, r7 width/uop count, r8
kernel-entry pinning by its author's own criterion, r8 per-volume scratch rotation 0-for-8),
plus `restrict` failing its own B=1 test. B=1 has read 0.217–0.223 for six rounds, the
geometry is 1.71× the best library and 1.29× a floor whose remaining prize is ~141 cycles.

**The single thing: the counter run (ask 3), or — better — L36_pfa's trick.** Both L=6
entries already ship `L6_FORCE`; a `create()`-side timed A/B of `fused` against `zff` with the
result in the description string would give the panel the same discrimination without a
monitor. **Secondary, and it is now a panel decision rather than an implementer one: both
entries independently seconded the r7 verdict's question about whether a second L=6 slot is
still earning itself.** L6_unrolled proposes redeploying to L=13 (thinnest margin) or L=64
(worst floor ratio). I agree, with one caveat: L=6 B=1 has been decided by process luck for
four rounds, so if a slot is cut, cut it on the *medians*, which say L6_pfa owns B=1 and
B=4096 and L6_unrolled owns B=64 and B=32768 — i.e. it is genuinely 2–2 and neither is
redundant. If the panel wants one L=6 entry, it should first run three more processes on the
existing binaries and settle the split properly.

### L = 8 — three entries, one algorithm. Explain the 3.3%, then cut to two.

The geometry has converged: bit-identical output in all four cells across all three entries,
1248 vector FP ops everywhere, the same fused shape at B=1, and B=1 at 1.23–1.28× floor.
Streaming has been declared converged for two rounds and is (three entries within 2.2% at
B=16384).

**The single thing: the alias counter (ask 1) on the three entries' B=1 binaries, because the
collapse has handed the panel a controlled experiment it never had.** With arithmetic held
bit-identical, the 0.552 / 0.564 / 0.570 spread is *entirely* non-arithmetic — prefetch branch
shape, scratch placement, code layout — and L8_fusedaxes' model predicts ~26–30 blocked loads
per volume against ~2. If the counter confirms it, the mechanism is real and reachable; if it
shows the aliasing is between the driver's `in` and `out`, three entries can stop spending
rounds on something structurally out of reach from inside a plan. Either answer closes L=8.
**Secondary, and it is a resourcing question like L=6's: L=8 does not need three slots to run
one algorithm.** L8_radix8's round produced a 2% self-inflicted regression and no new
mechanism; it is the natural donor.

### L = 17 — uop deletion is exhausted. Stop optimizing and measure, or rewrite.

Four mechanism classes falsified on the node: rescheduling (rader `ov` r5 0/4, rader `dz` r7
0/4, matrixsimd deferred-Z), address alignment (matrixsimd's twins, picked, −0.6%), spill
deletion (winograd's H, picked, −0.2%), and movement-uop deletion (rader's 8×8 transposes,
−31% uops, −3.0% at B=1 and *positive* at batch). B=1 has been 15.18–15.23 for five rounds at
1.31× floor.

**The single thing: the alias counter at B=1 (ask 1), `-DL17_FORCE=38` vs `48`** — it is
L17_matrixsimd's own request, it is one run, and it is the only way to distinguish "the node's
fixed heap layout happens to be benign so the twins are a no-op" from "address stalls are not
what the 43.9k−33.4k = 10.5k residual cycles are." Given four falsified classes I expect the
latter, in which case the honest reading is that **B=1 is at its structural window-drain
limit for this kernel family** and the only remaining move is the one both matrixsimd and
winograd have costed and neither has built: the interleaved-complex rewrite (winograd costs
it at +11% FP floor against ~4.5k deleted de/interleave uops — "likely a wash"). If the
counter reads clean, the panel should say so plainly and stop funding three arms at a
geometry that has not moved 1% in five rounds while sitting at 5.4× the best library.

### L = 36 — the fork is resolved. Stop looking at caches; go at the front end.

**L36_pfa's probe settled it: `fu − p1 − p2w ≈ −3 µs` across three processes. There is no
phase-boundary memory penalty. The L2-thrash story that motivated NTA (r7, three forms, zero
picks), `scratch+NT pf=4` (r7, rejected) and the deep-T1 staging pf=5/6 (r8, zero picks) is
dead.** Every prefetch instrument has now been rejected at B=1 by three independent
tournaments, and the geometry moved *away* from its floor this round (1.43× → 1.45×, the
worst of the four original sizes).

**The single thing: `perf stat -e idq.dsb_uops,idq.mite_uops,cycles` at B=1 (ask 2), and then
front-end work — code size, not caches.** All three records now converge on the same next
lever and it is well specified: L36_pencilfused's unrolled `exec_v4` is ~4080 instructions
against a ~1.5k-uop DSB, and L36_pfa's own r9 plan names `pw2` (which halves instruction
bytes through MITE and which the node *has* picked at B=256-r6 and in r7's B=32 checked runs)
as the thing to measure seriously. If MITE dominates, the correct move is **shrinking** the
unrolled bodies, which inverts four rounds of instinct. Two secondary items, both cheap:
**A/B `FFT36_PIND=-1` vs `2112` at B=1** — pinning ships always-on and coincides with a 1.2%
regression, so it must be priced before r9 builds on it; and **the batched cells stay frozen**
(B=32 flat and B=256 +1.3% in two rounds, both at modelled floors).

### The four wave-2 geometries, in one line each

* **L = 13 — the wallaby prefetch win did not transfer and B=512 is still the thinnest cell
  on the board (1.13×).** L13_direct's `pf` exec was selected and bought ≤2.5% against a
  −24% wallaby proxy; the entry ships `-DL13_PW=0` and `-DL13_PFIN=0` to split the halves and
  that A/B (two builds, two runs) is the whole item — CLX has less MLP than SPR and the entry
  itself suspected the balance would differ. Everything else at L=13 improved this round.
* **L = 23 — algorithmically finished at 1.14× floor; B=128 (1.55× floor) is the only cell
  with headroom, and the joint grid just found the first thing that moves it.** `rp-t1 +
  pf=2 + pw=1` took B=128 by 1.0% in the one process that picked it; make it the incumbent so
  it is picked 3/3, and run the alias counter (ask 1) to convert r7's −25–30% pad from a
  hypothesis into the rule five other geometries just failed to apply. Also: **L23_rader has
  inherited the timed≠checked exposure L23_matrixsimd just fixed** — it should adopt
  matrixsimd's canonical-order hysteresis, which is sitting in the same directory.
* **L = 45 — leanness paid 7%; the remaining 1.61× is split accesses, transposes and L3
  latency, and the memory-mechanism space is now empty.** L45_mixedradix's pre-registered
  branch fired: pfw, pkw and ERMS `cpy` all lost, so *"the next lever at L=45 is not memory
  mechanisms at all — it is the fuse-z-store-with-y-load rewrite or nothing."* L45_pfa's own
  transpose-conservation argument says that rewrite can relocate but not delete the two
  granule transposes per element, which narrows it usefully. **The single thing: the DSB/MITE
  counter at B=1 (ask 2)** — `x_ip0_pw4` is a 3087-instruction body against a ~1.5k-uop DSB,
  the same shape as L=36's, and both geometries can be settled by one recipe. Then propagate
  the opaque-base `asm` barrier to every entry with runtime-offset codelet macros; it is two
  lines and it was worth 7% here.
* **L = 64 — the tiling question is closed, the floor ratio is still the board's worst
  (1.76×), and `pfb` says scratch-read latency is not the gap.** L64_radix8's own next item is
  the right one and it is ~20 lines: **a one-slab prologue prefetch**, because `slabpf`
  prefetches slab *k+1* while z-lining slab *k*, so the first slab of every volume is always
  cold. Cheap standing sweeps: `FFT64B_NOHP` (hugepages measured at +3.3–3.7% in r7, never
  swept on the node's smaller STLB) and `FFT64R_TUNEDBG=1` for one node table per B.

---

## 7. Promotion

Against `docs/CURATION.md`, in its stated order. Sources are in `bench/geom/impl_8/`.

**1. Fastest correct entry per geometry.** The brief scores non-batched and batched
separately, so where a geometry splits cleanly by regime, both cell-owners qualify.

* **L = 6 — both.** `L6_unrolled` holds B=1, B=64 and B=32768 on the leaderboard minima;
  `L6_pfa` holds B=4096 and — on the run distributions, which is the honest statistic and
  which say so by ~2% — **also B=1** (§3e). Fourth round decided by process luck; the split
  is genuinely 2–2. Both also carry pre-registered measured negatives that close questions
  (rule 3): pinning falsified at B=64 by its author's own criterion, and `_rot` per-volume
  scratch rotation 0-for-8.
* **L = 8 — `L8_fusedaxes` and `L8_batchsimd`.** fusedaxes takes B=1 (0.552, a clean
  distribution win and the second B=1 movement in seven rounds), B=64 and B=16384; batchsimd
  takes B=2048 (0.912) and is tied at B=64 on medians.
* **L = 13 — `L13_direct`**, fastest in all three cells for a second round.
* **L = 17 — `L17_matrixsimd`** (B=1, B=8, B=256, sixth round as leader) and
  **`L17_winograd`**, which takes B=2048 on both minima and medians — its first cell.
* **L = 23 — `L23_rader`**, fastest in all three cells.
* **L = 36 — `L36_mixedradix`** (B=1, B=4, B=32) and **`L36_pfa`** (B=256, and on medians it
  owns that cell by 0.5% rather than the 0.2% the minima show).
* **L = 45 — `L45_pfa`**, fastest in all three cells, reversing r7 completely.
* **L = 64 — `L64_radix8`**, fastest in all three cells.

**2. Structurally different runner-up, close behind.**

* **`L13_rader`** — 6.1% behind at B=1 and *verified* structurally different by fingerprint
  (4.02e-16 vs 2.91e-16), so L=13 really is Rader against a dense conj-folded kernel. It also
  produced the round's biggest single-cell panel improvement (B=16 −8.2%, B=512 −7.0%) by
  killing its staged-store path outright, and **it fixed the one library loss on the r7
  board.** Its ys×pw matrix — the measurement that reconciles r6's "staging wins" with r7's
  "staging loses" by showing staging was only ever a workaround for un-prefetched cold-line
  RFOs — is the transferable part.
* **`L45_mixedradix`** — within 0.1% to 2.0% of the leader in all three cells, distinct
  fingerprint, genuinely different structure (PFA 9×5 two-sweep with `body()`/`exec_v_c`
  specialization, which L45_pfa borrowed *from* it this round). It also carries the round's
  most interesting failed idea with full numbers (rule 3): **ERMS `rep movsb` RFO deletion**,
  the only serious attempt anyone has made at deleting memory traffic rather than overlapping
  it, rejected in all three node cells alongside `pfw` and `pkw`, firing its own
  pre-registered falsification of the L=45 bandwidth model.
* **`L64_blocked`** — 4–13% behind on a genuinely different structure (8×8 two-stage,
  three-sweep, hugepage odd-line-padded scratch, distinct fingerprint). It carries **half of
  the L2↔DRAM tiling answer** (§5): the data-dependency argument that a second volume sweep
  is irreducible for any within-volume tiling, which is the part that generalizes beyond
  L=64. Its `pfb` null is pre-registered and recorded.

**3. Instructive failures — slower for a documented and measured reason.**

* **`L36_pfa`** qualifies under rule 1 already, but its record should be read as the round's
  primary instructive document for a different reason: **the in-plan node probe**. Its
  `p1`/`p2w`/`p2wd`/`fu` decomposition, pre-registered with both branches and routed onto the
  leaderboard through `fft3d_description()`, killed a diagnosis that had driven two rounds of
  rejected mechanisms across three entries. That pattern — when the monitor cannot run your
  counter, build the discriminator into `create()` — is the single most reusable thing in
  this round and it should be called out in `NOTES.md`.
* **`L6_pfa`** and **`L6_unrolled`** qualify under rule 1; both also carry pre-registered
  nulls (above) and both fell on the falsifying side of tests they designed to be able to
  lose, which is the behaviour the panel should keep rewarding.

**4. Anything that beat a library.** Selects **19 of 19** — every entry beat every library in
every scored cell of its own geometry, up from 18 of 19 in r7. It discriminates nothing this
round; recorded as a fact about the state of the board.

### Not promoted, with reasons

* **`L8_radix8`** — **not a structurally different runner-up: it now produces bit-identical
  output to both `L8_fusedaxes` and `L8_batchsimd` at all four batch sizes** (§3b), publishes
  the same 1248-op count, carries the same 52-instruction codelet class, and defaults to the
  same fused shape at B=1. `CURATION.md` forbids near-duplicates and rule 2 does not select
  it, on exactly the reasoning the r7 verdict used to retire `L23_matrixsimd`. It is third in
  all four cells, its one bet made its own B=1 cell 2% slower (0.570 with the *old* `2p`
  default; its new `1f` default measured 0.581–0.583), and its mid-regime hedge was declined.
  Its genuinely useful content this round — that adopting a rival's node-proven configuration
  does not transfer inside a different file even at bit-identical arithmetic — is recorded
  here in §2 and §4, and its r7 exemplar (which carries the `{1f,3p}×{pfw,no-pfw}` table that
  answered §4.3) remains in the reading list.
* **`L17_rader`** — third in all four cells, and **the two batched cells regressed (+1.7%,
  +1.4%)** in a round whose only change was a 31% movement-uop deletion. That deletion is a
  real, well-executed piece of work and its 8×8 zmm transpose network is a clean artifact —
  but the lesson it teaches (uop deletion is exhausted at L=17) is now carried, with numbers,
  by *both* promoted L=17 entries, and this is the fourth consecutive round in which this
  entry's headline mechanism was rejected or negative on the node. Its kernel survives inside
  the promoted `L17_winograd`; its r5 exemplar stands.
* **`L23_matrixsimd`** — still bit-identical to `L23_rader` at all three batch sizes (§3f),
  still the same op count and the same configuration; rule 2 does not select it and rule 1
  does not reach it (second in all three cells). **Credit where it is due, though: it set out
  to fix the one thing the r7 verdict held against it and it succeeded completely** — the
  board's worst timed≠checked offender is now clean in all three cells, verified 4/4 on a
  contended machine, and its canonical-order hysteresis is the fix `L23_rader` now needs
  (§6). That design should be lifted into `L23_rader` rather than kept as a separate arm.
* **`L36_pencilfused`** — third in all four cells, bit-identical to `L36_pfa` at B=32 and
  B=256, **regressed at B=256 (+1.7%)**, and outside its own predicted band at three of four
  cells. Its entire round was the `-funroll-loops` pragma, whose premise turns out not to
  exist (§3c); the resulting null is real but it is a null about nothing. Its r5 exemplar
  remains the reference for the cross-machine transfer lesson.

---

## Provenance and housekeeping

`impl_8/` and `results/panel_r8/` are **untracked** at the time of writing. **Commit
`impl_8/`, `results/panel_r8/`, the strategy records and the promoted exemplars before round
9 starts** (§3i).

Three corrections to the `panel_r7` verdict, all checked against the harness rather than
against a record:

1. **The `-funroll-loops` build-flag gap does not exist** and never did on this harness
   (§3c). Strike the finding, tell the panel, and have `L13_direct`, `L36_pencilfused` and
   `L45_pfa` A/B the pragma out — on `L17_rader`'s measurement it is a ~2% tax.
2. **The r7 L=17 synthesis "delete uops, don't reschedule them" is falsified** (§2, §4, §5).
   Three deletions, all selected, −0.2% / −0.6% / −3.0%.
3. **The MKL batched baselines regressed again** (§3a), so the r7 verdict's own resolution of
   the r5 excursion — "transient, specific to MKL, and still unexplained" — should be read as
   "recurrent, specific to MKL, and still unexplained." Five cells' margins in this round's
   leaderboard are inflated by it and §1 quotes both figures.

PROMOTE: L6_pfa L6_unrolled L8_batchsimd L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pfa L45_mixedradix L45_pfa L64_blocked L64_radix8
