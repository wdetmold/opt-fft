# panel_r11 — monitor's verdict

Measured on `p55n3`, Intel Xeon Gold 5218 (Cascade Lake, 2.30 GHz nominal, one 512-bit FMA
unit, 1 MB L2/core, 22 MB L3), exclusive, `governor=powersave`, gcc 11.4.0, slurm job 438529,
2026-08-22T08:03. Nineteen panel entries built, ran, and passed. `build_errors.txt` is empty;
no `failures.txt` was written; all 19 agents exited 0; all **259** correctness records report
`ok: true` with `rel_l2 ≤ 8.0e-16` against a `1e-12` tolerance.

The developers work on `wallaby`, a Xeon Gold 6448Y (Sapphire Rapids): two 512-bit FMA units,
2 MB L2, DDR5, full-clock AVX-512. Everything below that compares a claimed number to a
measured one has that gap in the middle of it, and MKL alone spans 2.9× between the two
machines.

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

The four commissioned geometries first, then the four wave-2 geometries for completeness.
"Best library" is the fastest of MKL 2022, MKL 2026, FFTW ×3 planner levels and ducc0 in that
cell. Times are µs per transform.

### L = 6 (volume 216)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | **L6_pfa 0.208** (L6_unrolled 0.209) | mkl_dfti 0.371 | **1.78×** |
| B=64 | **L6_pfa 0.220** (L6_unrolled 0.221) | mkl_dfti 0.401 | **1.82×** |
| B=4096 | **L6_unrolled 0.394 ≡ L6_pfa 0.394** (dead tie) | mkl_dfti 0.562 | **1.43×** |
| B=32768 | **L6_unrolled 0.563** (L6_pfa 0.565) | mkl_dfti 0.707 | **1.26×** |

### L = 8 (volume 512)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | **L8_batchsimd 0.551** (fusedaxes 0.555, radix8 0.578) | mkl_dfti 0.652 | **1.18×** |
| B=64 | **L8_fusedaxes 0.578** (batchsimd 0.587, radix8 0.610) | mkl_dfti 0.715 | **1.24×** |
| B=2048 | **L8_fusedaxes 0.922** (batchsimd 0.959, radix8 0.960) | mkl2026_dfti 1.332 | **1.44×** |
| B=16384 | **L8_batchsimd 1.236** (fusedaxes 1.251, radix8 1.272) | mkl2026_dfti 1.795 | **1.45×** |

### L = 17 (volume 4913) — still the largest margin on the board

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | **L17_matrixsimd 15.066** (winograd 16.458, rader 16.525) | fftw3_patient 81.725 | **5.42×** |
| B=8 | **L17_matrixsimd 16.438** (winograd 17.667, rader 17.795) | fftw3_patient 81.953 | **4.99×** |
| B=256 | **L17_matrixsimd 20.903** (winograd 21.562, rader 24.348) | fftw3_estimate 83.436 | **3.99×** |
| B=2048 | **L17_winograd 21.721** (matrixsimd 21.881, rader 24.980) | fftw3_estimate 84.098 | **3.87×** |

Every library is 3.9–5.4× off at this geometry; MKL is 4.7–6.7× off. L=17 remains the place
where a hand-built prime-size kernel is worth the most.

### L = 36 (volume 46656)

| case | fastest panel entry | best library | speed-up |
|---|---|---|---|
| **non-batched** | **L36_mixedradix 114.561** (pfa 117.861, pencilfused 118.986) | mkl_dfti 163.118 | **1.42×** |
| B=4 | **L36_pencilfused 124.701** (mixedradix 125.358, pfa 127.694) | mkl_dfti 175.303 | **1.41×** |
| B=32 | **L36_pfa 161.788** (mixedradix 163.380, pencilfused 164.790) | mkl_dfti 222.813 | **1.38×** |
| B=256 | **L36_pfa 174.268** (mixedradix 179.886, pencilfused 185.071) | mkl_dfti 248.954 | **1.43×** |

Four cells, three different winners. That is new — L=36 has never split three ways before.

### Wave-2 geometries

| geometry | case | fastest panel entry | best library | speed-up |
|---|---|---|---|---|
| L=13 | B=1 / 16 / 512 | **L13_direct 5.733 / 5.992 / 8.075** | mkl2026 7.530 / 7.607 / 9.336 | 1.31× / 1.27× / **1.16×** |
| L=23 | B=1 / 4 / 128 | **L23_rader 47.614 / 49.117**, **L23_matrixsimd 64.157** | fftw3_patient 260.810 / 261.378 / 265.149 | 5.48× / 5.32× / 4.13× |
| L=45 | B=1 / 2 / 16 | **L45_pfa 304.194 / 308.874 / 391.951** | mkl_dfti 605.981 / 598.233 / 683.443 | 1.99× / 1.94× / 1.74× |
| L=64 | B=1 / 2 / 8 | **L64_blocked 952.743**, **L64_radix8 1021.689 / 1247.663** | mkl_dfti 1193.813 / 1236.183 / 1971.810 | 1.25× / 1.21× / **1.58×** |

**No panel entry lost to any library in any of the 28 cells.** The thinnest margin on the
board is L=13 B=512 at 1.16× (was 1.12× in r10 — it widened only because MKL got 3.7% slower
in that cell, not because we got faster).

---

## 2. What changed since panel_r10, per geometry — and what regressed

Round-on-round deltas for all 19 entries. The panel's own noise floor for a recompile with
unchanged hot code is **±2%** (established r8/r9); I flag movement outside it, and I state the
**library drift in the same cell** wherever it matters, because several "regressions" are the
cell moving under everybody.

### L = 6 — the round's two questions both answered, both negative, and one joint drift

* **L6_pfa's incumbency flip fired and falsified its own hypothesis.** The pick at B=4096
  flipped to `fused_pf_d2` (zp-outer) **3/3** exactly as pre-registered, and the cell did not
  move: 0.393 → 0.394 against a predicted 0.382–0.389. Their own written branch: *"if the pick
  flips but the number stays ≈0.393, the 2.6% was never the x-order."* It was not.
* **The other side confirms it.** L6_unrolled's B=4096 kernel and pick (`fused_zp_pf`) are
  byte-identical to r10 and its cell went **0.383 → 0.394 (+2.9%)**. The r10 gap closed from
  both directions at once. x-pass group order is settled as an in-plan 1–3% effect that does
  not reach the cell; the r10 B=4096 gap was the allocation/layout draw.
* **B=32768 reverted symmetrically**: L6_pfa 0.554 → 0.565 (+2.0%), L6_unrolled flat at 0.563,
  and the cell changed hands back. Same draw, opposite sign.
* **Genuine unexplained joint regression at B=64**: L6_pfa 0.215 → 0.220 (+2.3%) *and*
  L6_unrolled 0.214 → 0.221 (+3.3%), while `mkl_dfti` in that cell is flat to three decimals
  (0.401 both rounds). Two independent entries moved the same way in one cell with the
  libraries pinned. Both changed their candidate tables this round, so the ±2% code-layout die
  is the standing explanation, but it landing the same sign twice is worth one look.

### L = 8 — the AA family retired with numbers; one deliberate, priced regression

* **B=1 and B=64 flat** (batchsimd 0.551 → 0.551; fusedaxes 0.556 → 0.555, 0.575 → 0.578).
* **L8_batchsimd B=2048 +2.9%** (0.932 → 0.959) on byte-identical streaming code and an
  identical pick string (`FUSED/s0w` 3/3, arena 0.895/0.901/0.904 — rock stable). This is the
  layout die, not a mechanism; its arena says so in-process.
* **L8_fusedaxes B=16384 +2.1%** (1.225 → 1.251) on candidate sets it states are byte-identical
  to r8/r9/r10, and it lost the cell to batchsimd. Same class.
* **L8_radix8 B=16384 +1.8%** (1.250 → 1.272) — this one is **not** noise and is the honest
  price of a correctness fix. It flipped its streaming pool from the `1f` family to the `3p`
  family to make the pool single-bit-class (see §3). Its own published probe now reads
  `3p-pfs-pfw` 1.140–1.148 against `1f-pfs-pfw*` 1.140–1.149 — dead even in-arena — so the
  1.8% at the cell is layout, and the family flip itself is free. It bought −2.4% at B=2048
  (0.984 → 0.960) in the same move.

### L = 13 — best round on two cells, worst single regression on the third

* **L13_direct is flat and still owns all three cells** (+0.7 / +0.2 / +0.9%, all inside
  noise), and it widened its library margin only because MKL slipped.
* **L13_rader B=16: −5.1%** (6.740 → 6.398) — the round's cleanest instrument-to-cell transfer.
  Its new in-plan race read `pw!` (write-intent prefetch OFF) at 18150/18158/7213 ns/vol
  against the incumbent's 19082/19154/7517 — **−4.0 to −5.2%, picked 3/3** — and the cell moved
  by exactly that. Its pre-registration ("a flip that does not drop the cell below 6.6 means
  the race adopted noise") is satisfied at 6.398.
* **L13_rader B=512: +8.3%** (8.577 → 9.292) — **the largest regression in the round.** Library
  drift in that cell is +3.7% (mkl2026 9.002 → 9.336), so ~+4.6% is the entry's own. Its own
  in-plan reading exonerates the kernel: `ab[B512] i:8637 ns/vol` against r10's 8.577 cell,
  while the driver reads 9.292 — a 7.6% in-plan/at-cell divergence. The leading suspect is the
  instrument itself: the new race allocates and touches `batch` private volumes (~36 MB at
  B=512) inside `fft3d_create()`, and setup at that cell went 0.000 s → 0.178 s. That is
  precisely the allocation-placement class this panel has documented repeatedly. **One build
  settles it: `-DL13R_AB=0` at B=512.**

### L = 17 — three write-side mechanisms fielded, three declined, cells essentially frozen

* **L17_matrixsimd's staged-output twins (F54/F55) were declined 6/6** — no run at B=256 or
  B=2048 shows a staged pick. Cells: B=1 +0.5%, B=8 −1.8%, B=256 −0.5%, B=2048 +1.2%. This was
  the r10 VERDICT's named build ("stage the output densely and pace its flush under the next
  volume's compute") and the node did not take it.
* **L17_winograd's `cw` (paced CLWB writeback) was declined 9/9** (`cw=0` in every run).
* **L17_rader's `xfs` rebuild lost on the node too**, and this time we can see by how much
  because it published the margin: `xrace xl/xfs = 22.52/25.18, 22.53/25.29, 22.50/25.28` at
  B=256 and `23.35/26.24, 23.25/26.44, 23.51/26.61` at B=2048 — **+11.8% to +13.4%**. Its cells
  regressed +2.0% (B=256) and +1.5% (B=2048); its class-A path is unchanged, so that is the
  layout die on a file whose `.text` grew.
* Net: **five write-side mechanism classes at L=17 are now node-priced and all null** (NT
  stores r1–r6, pfw r7/r9, staged input r10, staged output r11, CLWB pacing r11). The cells
  did not move.

### L = 23 — frozen by design, and re-opened by its own instrument

Both arms froze every executed instruction and shipped instruments. All six cells moved <1.2%.
L23_matrixsimd held its `flat + pf=2 + pw=1` pick 3/3 for a second round, and **L23_rader's
r10 three-way pick lottery is fixed** — it now reads `pf=2 pw=1` 3/3 as well. That was its
assigned deliverable and it landed.

### L = 36 — one bet dead, one bet paid, one gate fired for the wrong reason

* **L36_pfa's PFWD sweep is the round's best-paid single change**: `pfwd=1296` (half-plane
  write-intent lead) was picked over the never-measured 2592 default at **B=32 (2/3 runs) and
  B=256 (3/3)**, taking **B=32 −2.1%** (165.182 → 161.788, cell recaptured from mixedradix) and
  **B=256 −4.1%** (181.643 → 174.268, cell recaptured). An r5-era pacing constant that had
  never been measured anywhere was worth 4%.
* **L36_pfa B=1 +4.2%** (113.128 → 117.861) and **B=4 +2.8%** (124.221 → 127.694) — a real
  regression, and it cost the entry the B=1 cell it had tied for first. Its own prediction was
  111–115 "unchanged"; it landed 2.5% above the top of its band. The scored B=1/B=4 bodies are
  claimed instruction-identical to r10; the candidate count went 12 → 14 and the streaming pool
  34 → 42. Layout/plan-side, but it is the second time this file's B=1 has moved on a
  no-op-to-the-kernel round and it needs a controlled build.
* **L36_mixedradix's `zy` cross-plane interleave is dead, cleanly and pre-registered.** Probe
  `pf0/zy` = 143.9/153.3, 144.1/153.6, 143.4/152.1 → **ratio 1.064–1.068**, i.e. branch (c) of
  its own three-way fork (">1.03: the L1 argument is wrong too … the zy bodies get retired").
  The pick never left `v1-cached-pf0` and B=1 landed 114.561 inside its (b)/(c) band of 112–117.
* **L36_pencilfused's deterministic L2-size bit-class gate fired exactly as specified** — the
  marker it named in advance is visible in the leaderboard: its B=1 correctness fingerprint
  moved from the y-first family to **3.5726e-16**, bit-identical to L36_pfa's and
  L36_mixedradix's B=1 output, while its B=4 fingerprint stayed at the y-first **3.7588e-16**,
  which is the gate holding class A at 2<B≤8 as designed. **But the mechanism it was built to
  exploit is not there**: its own probe reads `is04` = 123.2/119.0/120.4 against `ip4` =
  120.5/118.0/118.4, so on the node istream0 is 0.8–2.2% *slower* in-arena than the in-place
  mode the gate overrode, and B=1 moved only −0.5% (119.588 → 118.986) against a predicted
  113.5–117.5. Its own branch: *"≥119 flat = the y-first/z-first order was NOT the gap."* At
  118.99 it is one part in a thousand off that line. The L2-overflow theory of pencilfused's
  4 µs B=1 deficit is falsified; the residual is now attributable to the spill delta its own
  audit found (114 stack moves in `passA_plane_v4` against L36_pfa's 78).
* It did, however, win **B=4 outright** (124.701) — the first cell this entry has ever owned.

### L = 45 — the best-improved geometry of the round

* **L45_pfa −2.0% at B=1** (310.439 → 304.194, taking the cell) and **−3.0% at B=2**
  (318.434 → 308.874); B=16 −1.0% and still owned. Its `zal` split-load anti-alias candidate
  was **declined** (pick is `pw4-ip-pf0`, not `pf0a`) → the split-load class is closed at L=45.
* **L45_mixedradix recovered its two-round B=16 regression: −4.7%** (425.326 → 405.188), with
  the `oc` odd-column form picked in 2 of 3 runs — its pre-registered branch (a) ("pick `-oc`
  and the cell lands ≤412") fired. The three-round-outstanding `-DFFT45_ODDR8` monitor ask is
  thereby self-served and answered: the odd-column rework was worth ~5% and the r9→r10 drift
  was recoverable. Its B=1 missed low (312.948 against a predicted 300–310).

### L = 64 — flat, with two questions closed

All six cells within ±1.2%. L64_radix8's `xb` (external buffer) A/B read in-place 3/3
(`xb0` in every run) — **r6's Sapphire Rapids rejection transfers to Cascade Lake and the
in-place-vs-buffer question is closed on both machines**, which was its branch (a).
L64_blocked's `pf1/pf8` bet read its null branch (`pf=0 pro=0` 3/3 at B=1).

---

## 3. Adversarial review — correctness, provenance, and anything missing

**Nothing failed correctness, nothing failed to build, nothing crashed, nothing hung, nothing
is missing.** I checked this four ways rather than trusting the absence of a `failures.txt`:

1. All 19 sources in `impl_11/` appear in the leaderboard, in every cell of their own geometry.
2. `agents/exits.txt` lists all 19 at `exit=0`; `build_errors.txt` is zero bytes.
3. All **259** `c_*.json` records carry `"ok": true`; `grep -L '"ok": true' c_*.json` returns
   nothing. Worst residual on the board is `baseline_matrix` at 8.4e-16 against a 1e-12 gate.
4. `timing.err` contains only `<entry>: does not support L=<other geometry>` lines, which is
   the harness correctly skipping geometry-specific kernels. No SIGILL, no timeout, no abort.

### The r10 §3(a) provenance charge: fixed, and the fix is visible in the leaderboard

r10 charged L8_radix8 by name: its B=2048 leaderboard minimum came from a run that picked one
bit class while the correctness check only ever saw another. This round its pools are single
bit-class, and the leaderboard *proves* it rather than asserting it — the checked fingerprints
are now **1.223e-16 at B=1** (the `2p` family), **2.2746e-16 at B=64** (the `1f` family, and
bit-identical to L8_fusedaxes' B=64 output to seventeen digits), and **1.913/1.915e-16 at
B=2048/B=16384** (the `3p` family) — one class per pool, each matching the family its pick
string names. Where its picks did vary across processes (`1f-pfs` r1/r2, `1f520-pfs` r3 at
B=64) the twins are `cmp`-verified bit-identical, so the variation is provenance-harmless.
That is the concrete fix the r10 VERDICT asked the whole panel to copy, and it cost 1.8% at
one cell (§2), which is exactly the trade a correctness fix is allowed to make.

L36_pencilfused went further and removed timing from the class decision entirely (a
`sysconf(_SC_LEVEL2_CACHE_SIZE)` gate). It reproduced the failure it was guarding against
first — two consecutive Haswell processes at B=2 installing *different* bit classes across a
3% margin — which is the right way to justify a rule.

### One residual provenance exposure, named

**L45_mixedradix's B=16 pool is not demonstrably single-class and its picks are not
deterministic.** Across the three timing processes it picked `v2-pf1-pfin-pfw-oc`,
`v2-pf1-pfin-pfw-oc`, and `v1-pf1-pfin-pfw`. `v1`/`v2` are width variants (PW=4 / PW=2) by that
file's own usage, and the record contains no `cmp` evidence that they are bit-identical; the
`oc` masked-transpose twin is likewise unasserted. The correctness check ran once, in a fourth
process, and there is no way from the artifacts to know which variant it exercised. **This is
not a correctness failure** — every variant that has been checked passes at 4.06e-16 — but it
is the same unvalidated-timed-class exposure that r10 charged against L8_radix8, and it is now
the only one left on the board. L45_mixedradix should apply L8_radix8's `installable` flag or
`cmp`-verify the pair before r12, and its 405.188 should be read with that caveat attached.

Lesser instances, all benign and stated for the record: L36_pfa's B=32 picks split `pw=4` /
`pw=2` / `pw=4` and its B=256 picks are `pw=2` 3/3, though all four L=36 cells share one
fingerprint across all three arms, so the width variants are demonstrably one class here;
L64_radix8's `fo` (an `fma(x,1.0,y)`-for-`add` twin, IEEE-identical by construction) flips
1/1/0 at B=8; L8_fusedaxes' `fused`/`fusedAA`/`fusedAA2` differ in store order only and print
identical digits.

### Nothing fast-and-wrong survived

I looked specifically for the failure mode that matters — a fast number produced by an
unchecked path. The only candidate structures are the ones above, and in every case either the
variants are bit-identical or all three arms of the geometry agree to seventeen digits. No
entry improved a cell this round by changing what it computes.

---

## 4. Claimed vs. measured — where the two machines disagree, and where the claim was simply wrong

The dev/score machine gap is real and every entry now writes its predictions in node units, so
most claims landed. The interesting rows are the ones that did not.

### Attributable to the machine difference (Sapphire Rapids → Cascade Lake)

| entry | claimed on wallaby | measured on node | reading |
|---|---|---|---|
| L17_matrixsimd | staged output **+12–13%** (17.75 vs 15.78 µs/t at nv=1000) | declined 6/6, cells flat | Wallaby has two FMA units and DDR5; the entry pre-registered this as "a pure node bet". The node agreed with wallaby, for once. |
| L17_rader | `xfs` **−18%** on wallaby | **+11.8 to +13.4%** on node (`xrace`) | The sign is the same but the node is *less* adverse. The `s17/rd = 0.82` favourable-read inversion does not transfer to a 17-stream shape carrying two loads per row. |
| L17_winograd | `cw` CLWB pacing **+26% / +33–49%** on wallaby | declined 9/9 | Wallaby "cannot price the node bet either way" (DDR5, 3× L3, different clwb µarch); the node's answer happens to agree. Prior was stated as LOW; it was right. |
| L36_mixedradix | `zy` **+9–13%** on wallaby, argued not to transfer ("48 KB L1d vs the node's 32 KB") | **+6.4 to +6.8%** on node | The argument was that the node's smaller L1 already loses the scratch, so zy's cost would vanish. It shrank by a third and stayed decisively positive. The L1-capacity model was wrong in magnitude, right in direction. |
| L36_pfa | `pf=7` **+16%** on wallaby B=1 | not picked at B=1 | Correctly pre-registered as unpriceable on a 2 MB-L2 machine; the node declined it too. |
| L36_pfa | `pfwd=1296` **−3.7%** in-arena on wallaby | picked 2/3 and 3/3, **−2.1% / −4.1%** at the cells | Transferred at full strength — one of the few wallaby→node transfers this round. |
| L6_unrolled | `abL` f3/f = **0.98–1.00** on wallaby (VD63 *faster*) | **1.011–1.020** on node, sign-stable 12/12 | The cross-machine sign inversion the entry predicted, now with the node number. See §5. |
| L8 (all three) | wallaby reads `fused ≡ fusedAA ≡ fusedAA2` to ~1 ns, six runs each | node separates them by 1–2% consistently | Wallaby's 512-entry ROB and small 4K-alias penalty genuinely cannot resolve this class. Third round of the same reading. |
| L8_radix8 | wallaby B=1 prefers `3p`, B=64 prefers `3p-pfs` | node prefers `2p` and `1f-pfs` | The standing wallaby/node inversion, now structurally unable to flip a pick because the pools are hardwired or single-class. |

### Claims that missed for reasons the machine gap does not explain

* **L36_pencilfused predicted B=1 at 113.5–117.5 and measured 118.986.** Its physics gate did
  fire and its fingerprint marker did appear, so the *plumbing* claim was right; the
  *mechanism* claim ("the L2-overflow hole is the 4 µs B=1 gap") was wrong, and its own probe
  says so in the same string (`is04` above `ip4` in all three runs). This is not a machine
  difference — the node was the machine the theory was about.
* **L36_pfa predicted B=1 "111–115 (unchanged)" and measured 117.861.** It also predicted a
  hot path "instruction-identical to r10". Both cannot be true. The +4.2% needs a controlled
  build (r10 candidate table vs r11's), not an attribution.
* **L45_mixedradix predicted B=1 300–310 and measured 312.948**, having ported the rival's
  xmm-tail change that moved the rival −5.5 µs. Its own stated failure criterion was ">315
  means my xmm tail codelet is worse than theirs"; at 312.9 it is just inside, but the port
  returned +1.2% where the donor returned −2.0%. Same geometry, same machine, same idea, two
  transcriptions — this is the §4.1 comparison repeating itself and it should be audited the
  way r10's spill-count comparison was.
* **L6_pfa predicted B=4096 at 0.382–0.389 and measured 0.394.** Honest miss, pre-registered,
  and it names its own next step (the file-level prune) — see §2.
* **L17_matrixsimd's `sbw` `cp` moved on its own**: 13.05–13.31 in r10, **11.12–11.31 at B=256
  and 13.26–13.43 at B=2048** in r11. The B=2048 replication is tight; the B=256 arena is
  40 MB against a 22 MB L3, so part of that probe is not DRAM-resident. Read `cp` at B=2048
  only.

---

## 5. Which `docs/LITERATURE.md` §4 open question moved

### Primary: §4.5 — the address-aware/alias class at L=8 is now closed, negative, in two independently-built files

§4.5 asks whether L=8 needs padding and where, and the r10 VERDICT escalated it to a single
named item: *"`fusedAA`, the only unpriced shape left … the node's own tournament has now
picked an address-aware variant twice and nobody has attributed either. Port it."* Two entries
answered, from opposite directions, and both answers are in the leaderboard JSONs.

**L8_fusedaxes attributed the variance and killed it.** Its diagnosis was that the shipped
`aa_perm_tab` de-aliases each phase-B iteration against only the *previous* iteration's stores
(depth 1), while the node's 56-entry store buffer holds ~3 iterations in flight — so the
observed r10 behaviour (one big win, two losses, ~2× `fused`'s spread) was a per-process
lottery over residual depth-2/3 collisions. It brute-forced a second table clean at depths 1,
2 and 3, changing store order only. The node's verdict, three runs at B=1:

```
arena{fused, fusedAA, fusedAA2}  µs/transform
  run 1   0.555 / 0.567 / 0.563
  run 2   0.555 / 0.564 / 0.566
  run 3   0.558 / 0.564 / 0.563
```

`fusedAA2`'s spread is **0.563–0.566** where r10's `fusedAA` ran 0.566–0.600. The c-lottery is
gone exactly as predicted — **and the depth-3 table lands 1.0–1.8% *above* plain `fused` in
all three runs.** That is the entry's own "compressed-and-high" branch, written in advance:
*"AA's layout costs more than its de-aliasing buys and the whole AA family retires with
numbers."*

**L8_batchsimd ported the shape and priced it independently.** At B=1 its publish-only A/B
reads `ab{fused=1.630, fusedAA=1.632}`, `{1.629, 1.632}`, `{1.627, 1.632}` — dead even, its
branch (c). At B=64 the ported `FUSEDAA/s0` reads 0.899 / 0.597 / 0.839 against `FUSED/s0`
0.892 / 0.587 / 0.841 — behind in two runs, level in one, never picked. L8_fusedaxes' own B=64
tournament picked `fusedAA+pfs` in exactly one of three runs again (0.604 vs 0.629), which is
the lottery, not a win.

**So §4.5 should record:** on Cascade Lake, for L=8³ complex-double with a 64-byte batch
granule, the address-aware pass-A layout plus permuted pass-B order is worth **zero to −2%**,
and permutation *depth* — the one mechanism that could have explained its variance — removes
the variance without moving the mean. §04's rule ("pad so every stride is an odd number of
cache lines") is not refuted; what is refuted is that a *store-order* fix can substitute for it
in this kernel. Two files, four cells, six independent tournaments, one direction. The AA
family can be retired.

Related and worth recording next to it: the in-scratch de-alias (`SI`) was already 1-for-3
across this geometry in r10; with AA now 0-for-2, **every allocation-class intervention at L=8
has been priced and none pays.** The three arms sit inside 4.5% of each other at B=1 and the
honest statement is that the residual is code layout and allocation draw, not a mechanism.

### Second: §4.6 — the r10 codicil is bounded, and the "store-feeding" law is narrowed again

r10 closed §4.6 with a refinement — association order is worth ~5% only when *the join feeds a
store on a machine with one store port* — and left one unexplained number: L6_pfa's B=32768
cell dropped 3.3% from an association change in the fully DRAM-bound regime, where both L=6
entries' models say a codelet cannot matter. The r10 VERDICT made settling it the single L=6
item. L6_unrolled built exactly the requested instrument: a probe-only VD63 kernel, never
installable, A/B'd against VD6 on a 16384-volume (113 MB) arena, alternating rounds, published
on every line as `abL=f<ns>,f3<ns>`.

Twelve node readings:

```
f3/f = 1.013 1.019 1.015 | 1.012 1.013 1.011 | 1.011 1.012 1.020 | 1.019 1.012 1.011
```

**Sign-stable in 12 of 12, mean ≈ 1.014**, against a wallaby reading of 0.98–1.00 for the same
pair. Neither pre-registered branch fires cleanly — it is above the ±1% "codelet is invisible"
band and below the ≥2% "association order survives into DRAM" bar. The correct entry for §4.6
is therefore: **association order does survive into the DRAM-bound regime at L=6, but at ~1.4%,
not the 3.3% the r10 cell move suggested.** Roughly 40% of that cell move was mechanism and
60% was the allocation draw — a reading independently supported by the fact that the same cell
handed itself back this round (L6_pfa 0.554 → 0.565 at an unchanged pick). The panel's "codelets
are invisible at bandwidth" model is not dead, but it now has a measured 1.4% error bar at L=6.

A fourth geometry also reported on the store-feeding law. L64_radix8 isolated the
*instruction-class* half from the *graph-shape* half by building `fout`, an `fma(x,1.0,y)`
twin of the even-output adds that feed its L3-bound stores — IEEE-identical, zero extra ops.
The node picked `fo0` **3/3 at B=1**, `fo0` at B=2, and split 1/1/0 at B=8. Null. Its
pre-registered reading applies: **the L=6 effect was graph shape, not opcode class, and r9's
wording over-claims.** With L=8, L=13, L=23 and now L=64 all null, the propagation ask is
finished.

### Third: §4.3's L2↔DRAM clause — the 39% headroom replicated, and then halved by a compute floor

r10's headline was L17_matrixsimd's `sbw` four-tuple showing the batched L=17 cells running
~8 µs (39%) above the node's own copy speed for their own traffic. The r10 VERDICT asked
L17_winograd for a second, independently-built four-tuple. It delivered, verbatim in shape:

```
B=2048, µs per volume          matrixsimd (r11)      winograd (r11)
  rd  sequential read            4.84/4.62/4.87       3.99/3.79/3.83
  wr  sequential write           7.92/8.04/8.20       9.15/9.33/9.02
  cp  read burst → write burst  13.31/13.43/13.26    13.81/13.58/13.43
  s17 17-stream read             3.83/3.84/3.77       3.92/3.53/3.68
```

**`cp` agrees to within 3% between two independently written probes in two different files.**
The 39% headroom is now a panel-wide fact, not one entry's instrument, and `cp ≈ rd + wr` in
both — the node overlaps DRAM read bursts against write bursts essentially not at all.

But the same round supplied the term that was missing from the ledger. L17_winograd added
`fu4`, the bare cache-resident compute time of the `h4` kernel the batched cells actually run —
a number nobody had ever published. The node reads **`fu4` = 17.27–17.55 µs**, tight across
nine runs. So for that entry:

* cell 21.72, compute floor 17.46, traffic floor 13.5 → the recoverable overlap is
  `cell − max(fu4, cp)` ≈ **4.3 µs (20%)**, not 8.4 µs (39%).
* Its own fork was written at a 17.5 threshold and landed on it, so the funded branch is
  "~4 µs of genuine overlap gap stands" — but the *size* of the prize for this entry is halved.

That is the round's most useful correction to §4.3's clause: **the 39% figure is `cell − cp`
and it silently assumes compute is free. It is not.** For L17_winograd the honest headroom is
20%; for L17_matrixsimd, whose `b1dec` sums to ~15.5 µs of compute, it is ~29%. The L2-tiling
construction §4.3 recommends is still the right target, but it is competing for ~4–6 µs, not 8,
and that is a materially different engineering proposition. Three write-side mechanisms were
fielded against the 8 µs number this round and all three were declined; the compute floor is
why.

### Also recorded, without claiming a §4 movement

* **§4.3 gains a second geometry.** L23_matrixsimd replicated `sbw` at L=23's strides
  (190.1 KiB volumes): `rd/wr/cp/s23` = 9.09/17.39/31.42, 8.99/17.90/31.71, 9.16/17.38/31.15,
  with **`s23/rd` = 1.01–1.03**. Two things follow. First, the L=17 result that interleaved
  17-row reads are *cheaper* than sequential (`s17/rd = 0.82`) **does not generalise** — at 23
  streams the node charges par. Second, and more consequentially, the B=128 cell runs at 64.16
  µs against `cp` = 31.4 and a B=1 compute of 47.6, so the zero-overlap bound is 79.0 and the
  perfect-overlap bound is 47.6: **~15 µs of overlap is already being collected and ~16.5 µs is
  not.** The r10 VERDICT closed L=23 on schedule-exhaustion reasoning — the same reasoning that
  was wrong at L=17 — and did so without ever measuring the machine. See §6.
* **§4.1 (spill traffic).** The L=45 pair repeated its r10 pattern with the roles unchanged:
  the same xmm-tail change measured −2.0% at B=1 in L45_pfa and +1.2% in L45_mixedradix.
  L36_pencilfused audited its `passA_plane_v4` at 114 stack moves against L36_pfa's 78 for the
  same-shaped body and, after its structural theory was falsified (§2), that delta is now the
  named residual for its 4 µs B=1 deficit. Spill counting as standard practice is holding up.

---

## 6. The single highest-value thing the next round should attack, per geometry

### L = 17 — still the board's largest margin, but the target must be re-sized before another mechanism is built

Three write-side mechanisms were built against the "8 µs of headroom" number and all three were
declined by the node's own tuners. §5 says why: `fu4 = 17.46` means winograd's cell has ~4.3 µs
of overlap gap, not 8.4. **The single item is to get the same compute floor for L17_matrixsimd**
— it has `b1dec` phase decomposition but no single steady-state batched-kernel number
comparable to `fu4` — and then decide, with both floors in hand, whether ~4–6 µs justifies the
L2-tiling rewrite §4.3 names. Do not fund a fourth write-side mechanism against an unbounded
target. If matrixsimd's floor also lands near 17–18 µs, the geometry is close to compute-bound
at batch and the next lever is a cheaper `w4` fused-pass kernel, not memory scheduling.

Secondary, and cheap: L17_matrixsimd asked, in advance, that if its twins were declined the
monitor confirm the closure with one counter run (`offcore_requests_outstanding` or
store-buffer-full stalls, F51 vs F55 at B=2048). That is a reasonable ask and it is the one
measurement that would convert "declined" into "closed".

### L = 36 — the B=1 regression, not a new mechanism

Three of four cells improved or held and the geometry split three ways, but **L36_pfa's B=1 went
+4.2% on a round it claims changed no scored instruction**, and that is worth more than any
mechanism currently on the table: it cost the panel its best L=36 number (113.128 → 114.561 is
now the board best). The item is a controlled build — r10's candidate table against r11's, hot
path fixed — to separate plan-side growth from the layout die. Everything else at L=36 is
converged and the entries know it: `zy` is dead at 1.065, `pf=7` was declined, the split-access
toll is structurally zero, intra-plane z/y fusion is impossible, and L36_pfa's own probe now
reads `p1z = 39.8–41.8` against a ~26 µs port share with `p1y = 26.2–28.1` at ~26 — **its
branch (b) fired: the in-read latency is the term, every read-prefetch instrument is 0-for-B=1,
and B=1 is at this structure's memory floor.** Say so and stop hunting B=1 mechanisms.

For L36_pencilfused specifically the item is the spill surgery its own audit named (114 → 78
stack moves in `passA_plane_v4`), now that the structural theory is falsified.

### L = 8 — the geometry is engineering-complete; spend the slot elsewhere

Every allocation-class mechanism is now priced at zero (§5), three arms sit inside 4.5% at B=1,
B=1 has been 1.28× its floor for eight rounds, and the streaming cells have been converged for
four. The only item I would fund is L8_radix8's own flag: its `1f-pfs-pfw*` probe reads dead
even with the installed `3p-pfs-pfw` at B=16384 (1.140–1.149 both), so the family flip is free
and the 1.8% is layout — one controlled build confirms that and closes the last question. Then
this geometry should give up a slot. The panel has three arms producing 4.5%-wide answers here
and one arm (L=23) that has just discovered 16 µs it cannot explain.

### L = 6 — converged; execute the slot ruling

The `abL` instrument answered the r10 VERDICT's single item at 1.4% (§5), which is small enough
that no kernel work follows from it. Both entries' remaining ideas are file-level prunes. B=1 is
at 1.23× floor, B=32768 at the compulsory-traffic floor, and the round's only movement was two
cells trading hands on the allocation draw. **The r9/r10 slot ruling should now execute.** L6_pfa
keeps the geometry.

### The wave-2 geometries

* **L = 23 — re-open it.** This is my correction to my own predecessor. The r10 VERDICT declared
  L=23 closed on schedule-exhaustion reasoning and told the panel to stop funding it; the same
  reasoning had just been shown wrong at L=17 by a measurement, and L23_matrixsimd made the
  measurement rather than accept the ruling. Its `cp` = 31.4 µs against a 64.2 µs cell and a
  47.6 µs compute floor leaves **~16.5 µs unaccounted for at B=128** — proportionally the
  largest un-attributed gap on the board now that L=17's has been halved by `fu4`. The item is
  the same one L=17 needs: measure the batched steady-state compute floor, then decide.
  L=23's `s23/rd = 1.0` also says its read shape is neutral, so the residual is write-side or
  overlap there too.
* **L = 13 — the B=512 regression, and it is one build.** `-DL13R_AB=0` at B=512 for
  L13_rader separates the instrument's allocation footprint from the cell (§2). Its other two
  cells had an excellent round and both of the standing monitor asks are now answered on the
  node at zero monitor cost: `pw` at B=16 is worth −5.1% (cashed), and the fused-vs-unfused
  schedule at B=1 reads `i:17502/f0:17594`, `15620/15772`, `6557/6540` — **within 0.5%, so the
  1198-line fused kernel buys nothing at B=1 and should go.**
* **L = 45 — fix the provenance, then read B=1.** L45_mixedradix's B=16 recovery (−4.7%) is
  real and its `oc` branch fired, but its pool spans `v1`/`v2` with non-deterministic picks and
  no bit-identity evidence (§3). Apply L8_radix8's `installable` flag first; its 405.188 is
  provisional until then. The second item is the transcription question: the same xmm-tail port
  returned −2.0% in L45_pfa and +1.2% here, which is r10's spill comparison repeating, and one
  `objdump` settles it.
* **L = 64 — the geometry is flat and two of its live questions closed this round** (`xb`
  in-place on both machines; `fout` null). Floor ratio is still 1.73×, the board's worst. The
  item is unchanged from r10: B=8, where L64_blocked's `st=3` still buys nothing (1324.5) and
  L64_radix8 is 6% ahead. That is a residency boundary nobody has located.

---

## 7. Promotion

Applying `CURATION.md` in order. As in r10, **rule 4 ("anything that beat a library baseline")
selects all nineteen** — every arm beat every library in all 28 cells — so it carries no
discriminating weight, and the decision rests on rules 1–3 plus the prohibition on
near-duplicates.

**Rule 1 — the fastest correct entry per geometry.** The cells split more widely this round
than any previous one: **L6_pfa** (B=1, B=64; ties B=4096), **L8_batchsimd** (B=1, B=16384),
**L8_fusedaxes** (B=64, B=2048), **L13_direct** (all three, disjoint), **L17_matrixsimd** (B=1,
B=8, B=256), **L23_rader** (B=1, B=4 on minima), **L36_pfa** (B=32, B=256), **L45_pfa** (all
three), **L64_radix8** (B=2, B=8).

**Rule 2 — a structurally different runner-up that came close.**

* **L6_unrolled** — owns B=32768 (0.563), ties B=4096, and built the round's §4.6 instrument.
  Its `abL` probe is the only place the 1.4% DRAM-regime codelet effect exists as a number, and
  its record carries the transferable trap that made it trustworthy: gcc inlined the
  single-caller probe kernel until it was called through a `volatile` pointer, which an
  `objdump -t` audit caught and no warning would have.
* **L8_batchsimd** and **L8_fusedaxes** — two cells each, and between them they closed §4.5's
  last open mechanism from two independently-built directions (§5). L8_fusedaxes' depth-3
  permutation table is the artifact: it removed the variance and proved the mean was never
  there.
* **L17_winograd** — a genuinely different structure (296-instruction hand-derived
  cyclic+negacyclic module), owns B=2048 (21.721), and produced **`fu4`**, the compute floor
  that halves the L=17 headroom estimate and explains three rounds of declined write-side
  mechanisms. Its replicated `sbw` four-tuple is what makes the 39% figure a panel fact.
* **L36_mixedradix** — owns B=1 (114.561, the board's best L=36 number), structurally distinct
  (lanes-are-lines against GT-PFA two-sweep), and killed its own `zy` bet at a pre-registered
  1.065 probe ratio without touching the cell.
* **L36_pencilfused** — owns B=4 (124.701), the first cell this entry has ever taken; r10
  declined it on the near-duplicate rule because all three L=36 arms had shipped one identical
  change, and that no longer holds — this round the three arms shipped three different changes
  and split four cells three ways. Its deterministic `sysconf`-based bit-class gate is a new,
  transferable mechanism that takes timing out of a correctness-relevant decision entirely, and
  it is promoted *with* its honest negative (§2): the gate fired exactly as designed and
  installed the class its own probe ranks 0.8–2.2% slower.
* **L13_rader** — structurally distinct (Rader-13 against dense 13×13), took B=16 by −5.1% on a
  node-measured in-plan flip, and answered two multi-round monitor asks for free (§6).
* **L45_mixedradix** — recovered a two-round B=16 regression by −4.7% with its pre-registered
  `oc` branch firing, which self-serves the three-round-outstanding `-DFFT45_ODDR8` ask.
* **L64_blocked** — owns B=1 (952.743), structurally different (8×8 two-stage split-complex
  blocked against radix-8² per axis), and its `st=1`/`st=2`/`st=3` history is the geometry's
  record of what does not work.

**Rule 3 — instructive failures whose record documents the number that killed them.** Four
promotions rest substantially here, and one is promoted almost entirely on this ground:

* **L8_radix8** — third in three of four cells for a fourth round, and promoted anyway. r10
  charged it by name with the §3(a) provenance defect (a leaderboard minimum from a bit class
  the correctness check never saw) and named it the donor. This round it fixed the defect with
  a general mechanism (`installable` flags, single-bit-class pools, cross-class probes marked
  `*` in the description), **paid a measured 1.8% at B=16384 for it**, and the fix is verifiable
  from the leaderboard alone: four cells, three bit classes, each matching the family its pick
  string names (§3). That is precisely CURATION rule 3 — slower for a documented and measured
  reason — and it is the concrete artifact every other entry was told to copy. Two did.
* **L36_pencilfused's** falsified L2-overflow theory, **L36_mixedradix's** `zy` null,
  **L17_matrixsimd's** staged-output decline (its second consecutive round of building the
  mechanism the monitor named and having the node refuse it), and **L6_pfa's** flipped-pick /
  unmoved-cell result are all pre-registered forks where the author bet and lost in public. That
  is what makes the records worth reading.

**Not promoted, with the number:**

* **L17_rader** — third in all four cells, batched cells regressed +2.0% and +1.5%, and its
  `xfs` rebuild lost by **11.8–13.4%** on the node (`xrace` 22.5/25.2 and 23.3/26.4) after
  losing 18% on wallaby. Sixth consecutive round of node-declined batched mechanisms. Its round
  was not wasted — publishing `xrace` converted a sixth silent decline into a number, and that
  number establishes that L=17's favourable-read result does not transfer to a 2-loads-per-row
  17-stream shape — but that finding is recorded here and in a promoted file's `sbw` string.
  Its own record's branch 4 recommends the consolidation; I agree.
* **L23_matrixsimd** — bit-identical to L23_rader and overlapping in all three cells (0.07%,
  1.2%, 0.6% — every one inside the noise floor), so the near-duplicate prohibition is exactly
  on point. This is a genuinely close call, because its `sbw` four-tuple at L=23 is the round's
  second-most valuable measurement and the reason I am re-opening the geometry (§6). It is not
  promoted only because the instrument itself is already in the exemplar set verbatim, in
  L17_matrixsimd's and L17_winograd's files. **At promotion time its `sbw` numbers and its
  three-round 3/3 pick determinism must be copied into L23_rader's record**, the same
  instruction r10 gave and for the same reason.

Seventeen of nineteen. That is two more than r10 and it is not a loosening of the bar — it is
what happens when four geometries split their cells between arms and three separate entries
ship distinct, transferable mechanisms in one round. But it is also a signal in its own right:
a panel in which 17 of 19 arms are worth keeping has too many arms, and the r9/r10 slot rulings
(L=6 and L=8 each carrying a redundant arm, L17_rader named the donor) should now execute so
that r12's slots go to L=17's and L=23's unexplained microseconds instead.

**A note for `NOTES.md` at promotion time.** This was a round of honest negatives. Of the
eleven pre-registered mechanisms fielded — `zy`, `pf=7`, staged output, `cw`, `xfs`, `fusedAA`,
`fusedAA2`, `zal`, `fout`, `xb`, `istream0` — **ten were declined by the node's own tuners and
the eleventh (`istream0`) was installed by a physics gate over its own probe's objection.** The
two changes that actually moved cells were an unmeasured five-round-old constant (`pfwd`
2592 → 1296, −2.1% and −4.1% at L=36) and turning a prefetch *off* (`pw` at L=13 B=16, −5.1%).
The panel is past the point where new mechanisms pay and into the point where measuring what it
already ships pays; the round's three best artifacts — `abL`, `fu4`, and the L=23 `sbw`
four-tuple — are all instruments, and all three changed what the panel believes.

PROMOTE: L6_pfa L6_unrolled L8_batchsimd L8_fusedaxes L8_radix8 L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_pfa L36_mixedradix L36_pencilfused L45_pfa L45_mixedradix L64_radix8 L64_blocked
