# Round ice_r6 — monitor's verdict

Scored on `a80n0.lqcd.mit`, slurm job 438633, 2026-08-23T08:38:08-04:00.
19 panel entries + `baseline_matrix` + 6 library backends, 8 geometries, 3 processes each.

## 0. Three corrections to the brief, before any number is read

**0.1 The scoring machine is not a Cascade Lake Gold 5218 — same correction as ice_r5 §0.1.**
`environment.txt` reports `Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` with
`avx512_vbmi avx512ifma avx512_bitalg avx512_vpopcntdq` in the ISA string. Those four are
Ice Lake-SP (Sunny Cove); Cascade Lake has none of them. This is a dual Gold 6326
(2 × 16C/32T = the reported 64 logical cores): **1.25 MiB L2 per core, not 1 MiB, and two
512-bit FMA pipes, not one.** The round tag `ice` is correct; the brief and
`docs/LITERATURE.md` §4.8 item 6 are the stale artifacts, and they have now been stale for
two consecutive monitor reports. The panel has routed around them empirically — twelve
records this round carry measured `clk512/clk256 = 3.30/3.50 GHz` probes and reason about
"the two FMA pipes", against §4.8's documented "2.9 GHz flat, one 512-bit FMA unit".
**Reissue §4.8 item 6 and §3.1's "batch tile for the 1 MiB L2" row against the Gold 6326
before the next round tunes to them.**

**0.2 There is no Sapphire Rapids development machine, so the brief's
claimed-versus-measured premise does not apply.** Every one of the 19 strategy records
states its measurements were taken on `a80n0` — the scoring node — via `tryout.sh` or a
direct core lease. There is no cross-machine transfer in ice_r6 and no MKL spanning 2.9×:
same-window MKL controls quoted in the records read 0.94 / 2.10–2.15 / 88.8 / 281–288 µs at
L = 6/8/17/36, against scored MKL rows of 0.940 / 2.112 / 88.505 / 284.283. Those agree to
about 1%. §4 below therefore reports a genuine claim-vs-measured audit rather than a
machine-difference apology, and the result is that **every prediction landed**.

**0.3 This round contains no non-batched measurement — sixth round running.**
`cases.txt` ("THE GRADED CONFIGURATION, fixed, identical for solver/base/SOTA") has eight
rows, `6:64:4856` through `64:2:134`, and no `B=1` row; `sweep.sh` iterates exactly that
file. Every `t_*.json` in this directory carries the one batch per L. So although
`fft3d_api.h` states that "Batched (B>1) and non-batched (B=1) are separate measurements",
ice_r1 through ice_r6 have measured only the batched chain. §1 gives the scored batched
figure and, for B=1, the implementers' own numbers **clearly labelled unscored**. This is
not a cosmetic gap: at L=64 the B=1 order is *inverted* against the scored order (§1), which
is the second geometry in two rounds to show that. Adding the rows is a one-line change.

**0.4 The round's seed is not recorded anywhere in its output.** `environment.txt`,
`leaderboard.txt` and `sweep.out` contain no `--seed` value. `CURATION.md` justifies
discarding the raw per-round data on the grounds that "a round is re-runnable from its
seed"; that guarantee is currently unbacked. It matters more than it looks, because §3.1
shows the L=6 correctness gate is **dominated** by the seed.

---

## 1. Headline per geometry

Scored figures are µs per transform, minimum over three independent processes, from
`leaderboard.txt`. This was a quiet scoring window: the three per-process minima agree to
≤0.5% for almost every entry (contrast ice_r5's 33% spread at L=36), so the rankings below
are robust unless noted.

### L = 6 (V = 216, B = 64, m = 4856)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L6_unrolled 0.291 µs** (28.78 GF/s) | mkl_dfti 0.940 µs | **3.23×** |
| non-batched (unscored) | L6_unrolled ≈0.300 quiet-equivalent (0.341 loaded); L6_pfa 0.335 busy-window | not paired at B=1 | — |

> **Read this cell as void, not as a result.** No backend at L=6 — not our two, not any of
> the six libraries, not the exact dense reference — passed the map-chain correctness gate
> this round. The leaderboard prints `ok` for all nine because of an unfixed bug in
> `check.py`. See §3.1; it is the round's most serious finding. Both L=6 entries are 1.7%
> apart and the whole gap is inside their accuracy-tier choice.

### L = 8 (V = 512, B = 64, m = 2572)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L8_fusedaxes 0.555 µs** (41.54 GF/s) | mkl2026_dfti 2.112 µs | **3.81×** |
| non-batched (unscored) | L8_fusedaxes 0.556; L8_radix8 0.570; L8_batchsimd 0.574–0.576 (MKL 2.27 → 4.0×) | not paired at B=1 | — |

The fastest cell on the board in GF/s terms, and a lead change: `L8_fusedaxes` took the
geometry by executing `L8_radix8`'s own suggested idea, in the same round that `L8_radix8`
built that idea differently, measured it a 10% loss, and declared the search closed (§2).
`L8_fusedaxes`'s three process minima are 0.555 / 0.555 / 0.555 — the flattest reading on
the board.

### L = 17 (V = 4913, B = 32, m = 98)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L17_winograd 11.649 µs** (25.86 GF/s) | ducc0_c2c 86.263 µs | **7.41×** |
| non-batched (unscored) | L17_winograd 13.350 (MKL 99.5 → 7.45×); L17_matrixsimd 13.610; L17_rader 13.888 | | |

The largest margin over any library at any geometry, for the sixth consecutive round, and
the round's second lead change. Every library still agrees to within 5% here — ducc0 86.3,
MKL 88.5/88.8, FFTW 90.1 at all three planner levels — the corpus's "neither library has a
good 17" result reproduced again. The ranking survives the stricter statistic:
median-of-process-minima gives `L17_winograd` 11.753 against `L17_matrixsimd` 11.945, still
1.6%.

### L = 36 (V = 46656, B = 8, m = 64)

| | panel best | best library | margin |
|---|---|---|---|
| **batched (scored)** | **L36_mixedradix 100.801 µs** (35.89 GF/s) | mkl_dfti 284.283 µs | **2.82×** |
| non-batched (unscored) | L36_mixedradix 114.788 (MKL 315.5 → 2.75×); L36_pencilfused 121.585; L36_pfa 121.634 | | |

`L36_mixedradix` holds first cleanly this time (100.801 / 101.010 / 101.185, sd 0.4% — last
round's 33% spread is gone) and the win is now 5.4%, not 4.4% inside its own noise. The
second/third places are *not* separated: `L36_pfa` 106.249 and `L36_pencilfused` 106.908 are
0.6% apart with overlapping process spreads.

### The other four geometries, for completeness

| L | panel best | best library | margin |
|---|---|---|---|
| 13 | L13_direct 5.287 µs | mkl2026_dfti 11.798 µs | 2.23× |
| 23 | L23_matrixsimd 35.888 µs | ducc0_c2c 247.482 µs | 6.90× |
| 45 | L45_pfa 264.111 µs | mkl_dfti 756.922 µs | 2.87× |
| 64 | L64_blocked 638.757 µs | mkl_dfti 1720.604 µs | 2.69× |

**Every panel entry at every geometry beats every library backend.** The slowest panel entry
on the board relative to its cell (`L36_pencilfused`, 3rd of 3) still beats the best library
at its geometry by 2.66×. So `CURATION.md` promotion ground 4 is again nearly vacuous — with
one exception that matters this round: at L=6, where nothing is certified correct, ground 4
is the *only* ground on which anything can be kept (§7).

**One inversion worth the brief's attention.** At L=64 the unscored B=1 numbers are
`L64_radix8` 672.1 against `L64_blocked` 713.5 — the loser of the scored cell is 5.8% faster
non-batched. ice_r5 found the same inversion at L=36. Two geometries, two rounds, and the
harness still cannot see it.

---

## 2. What changed since ice_r5, per geometry

Per-transform µs, r5 scored → r6 scored. All 19 implementations changed source this round
(`impl_5` vs `impl_6`: 19 of 19 files differ, `baseline_matrix.c` byte-identical as it should
be), and all 19 strategy records were updated (2,759 insertions).

| L | entry | r5 | r6 | Δ |
|---|---|---|---|---|
| 6 | **L6_unrolled** | 0.323 | **0.291** | **−9.9%** ← lead change |
| 6 | L6_pfa | 0.304 | 0.296 | −2.6% |
| 8 | **L8_fusedaxes** | 0.585 | **0.555** | **−5.1%** ← lead change |
| 8 | L8_radix8 | 0.570 | 0.569 | −0.2% |
| 8 | L8_batchsimd | 0.596 | 0.575 | −3.5% |
| 13 | L13_direct | 5.401 | 5.287 | −2.1% |
| 13 | L13_rader | 5.808 | 5.377 | −7.4% |
| 17 | **L17_winograd** | 14.412 | **11.649** | **−19.2%** ← lead change |
| 17 | L17_matrixsimd | 12.736 | 11.935 | −6.3% |
| 17 | L17_rader | 15.052 | 12.284 | −18.4% |
| 23 | L23_matrixsimd | 36.903 | 35.888 | −2.7% |
| 23 | L23_rader | 37.780 | 37.172 | −1.6% |
| 36 | L36_mixedradix | 103.888 | 100.801 | −3.0% |
| 36 | L36_pfa | 115.437 | 106.249 | −8.0% |
| 36 | L36_pencilfused | 108.631 | 106.908 | −1.6% |
| 45 | L45_pfa | 263.807 | 264.111 | **+0.1%** |
| 45 | L45_mixedradix | 262.959 | 264.977 | **+0.8%** |
| 64 | **L64_radix8** | 1011.066 | **660.301** | **−34.7%** |
| 64 | L64_blocked | 639.508 | 638.757 | −0.1% |

### Did anything regress?

**No entry regressed in code. One geometry regressed on the board: L=45.** The panel best at
L=45 went 262.959 → 264.111, i.e. the cell is 0.4% *slower* than last round. Neither entry is
at fault and both improved in matched conditions:

* `L45_mixedradix` shipped a cache-line-padded chain state (rows 45 → 48 complex) and
  measured it **−1.0% at B=4 / −1.3% at B=1 in same-window pairs**, bit-identity
  cmp-verified. It scored +0.8%.
* `L45_pfa` shipped an x-first step with the map at the y-subpass stores and measured
  **−1.7/−1.8% in 2/2 same-window pairs** (270.2 vs 275.2). It scored +0.1%.

Both records independently document the mechanism and predicted this exact outcome: the node
toggles between a fast (~263) and a slow (~271–275) whole-run class on byte-identical code
while MKL stays flat, so cross-round absolutes at L=45 are worth less than the ~1% both
entries won. `L45_mixedradix`'s process minima (264.977 / 265.908 / 267.070) versus
`L45_pfa`'s (264.111 / 264.396 / 272.510) put the two within the class width. **L=45 is a tie
for the third consecutive round** — though for the first time the order agrees under both
min and median-of-minima, marginally favouring `L45_pfa`.

Three **rank changes**, none of them a regression:

* **L=6:** `L6_pfa` → `L6_unrolled`. Both got faster; the flip is confounded by an accuracy
  tier change and the cell is void anyway (§3.1).
* **L=8:** `L8_radix8` → `L8_fusedaxes`. `L8_radix8` did not slow down (−0.2%, and its chain
  drift reproduces r5's `2.599e-11` to every printed digit, which it can only do if the chain
  is bit-for-bit r5's); it was passed by a 5.1% gain in the fused arm.
* **L=17:** `L17_matrixsimd` → `L17_winograd`. `L17_matrixsimd` gained 6.3% and still lost
  the cell, because the other two arms gained 19% and 18%.
* At **L=36** second and third traded (`L36_pfa` passed `L36_pencilfused`) but by 0.6%,
  inside both spreads — call it unseparated rather than a flip.

### The round's structural story

Sorting the movers by mechanism gives a single, sharp result, and it is the *opposite* of the
r5 directive at three of the four headline geometries. **Every entry that gained more than
5% did so by changing pass shape, alignment, or where the chain state's axes live — not one
of them changed an operation count.**

* `L17_winograd` (−19.2%) put one spatial axis in the SIMD lanes for a whole step and
  *alternated which axis* between steps, so pass 3's output lanes simply become the next
  state's lane axis instead of being transposed back. Two full 8×8 orientation changes per
  step deleted, ~5k uops/step. FFT arithmetic unchanged at 296 FP instructions per 17-point
  kernel.
* `L17_rader` (−18.4%) found a **one-line bug**: `2·NVOL = 9826 ≡ 2 (mod 8 doubles)`, so five
  volume-sized buffers ahead of its `msr` block left the hot arena 16 B past a cache-line
  boundary and *every* cosine-splat FMA memory operand — ~7.8k loads/step — was a line-split
  load. One realignment statement, bit-identical by construction, worth ~2.3 of 15 µs.
* `L64_radix8` (−34.7%) refuted `L64_blocked`'s r5 impossibility claim ("I do not see a legal
  fusion; treat ~640 µs as this structure's floor"). Axis alternation makes the fusion legal:
  one sweep per step with the plane orientation alternating x/y, deleting a full state
  read+write per step — traffic 22.3 → 13.4 MB/step at unchanged arithmetic.
* `L8_fusedaxes` (−5.1%) split phase B into two ~150/~240-uop passes through a second grid
  scratch, paying +256 L1 memory ops per step at *unchanged* p05 uop count.
* `L36_pfa` (−8.0%) adopted `L36_mixedradix`'s eager-map protocol wholesale and then improved
  its divider:ladder ratio past it.
* `L13_rader` (−7.4%) adopted `L13_direct`'s volume-group-major chain wholesale, converting
  every X-pass load and Z-store RFO from an L3 round trip to an L2 hit at zero added FP work.

The cross-entry borrowing is now the dominant mechanism of progress, and it is properly
attributed: every record has a "Borrowed this round, named" section, and several are mutual
(`L36_mixedradix` ↔ `L36_pencilfused` credit each other's prior rounds in the same protocol).
Two entries also read structure out of the archived rival pipelines in
`ext/reference/fft_v4_solutions/` (`L17_matrixsimd` from `1760b1bf`, `L64_blocked` from
`1000f989`). I checked this for provenance hygiene: no implementation includes or links
anything outside libc, `immintrin.h` and `fft3d_api.h`. The rival references are design ideas
in comments, not code.

---

## 3. Adversarial pass: correctness, builds, crashes, omissions

### 3.1 Every backend at L=6 failed the chain gate, and the leaderboard printed "ok" for all nine

`check.log` contains nine `FAIL` lines. All nine are the L=6 map-chain check, and they cover
**every backend on the board at that geometry**:

| L=6 backend | chain rel_l2 | vs tol 4.856e-10 |
|---|---|---|
| mkl2026_dfti | 6.174e-09 | 12.7× over |
| fftw3_estimate / measure / patient | 7.952e-09 | 16.4× over |
| mkl_dfti | 8.842e-09 | 18.2× over |
| **L6_pfa** | **9.516e-09** | **19.6× over** |
| ducc0_c2c | 1.149e-08 | 23.7× over |
| **L6_unrolled** | **1.158e-08** | **23.9× over** |
| baseline_matrix | 4.389e-08 | 90.4× over |

`leaderboard.txt` lines 11–19 show `ok` for all nine and ranks all nine.

**Root cause, unchanged from ice_r5 §3.4.** `check.py` computes `ok` from the single
transform, writes it into `result`, then at the chain stage does `ok = ok and chain_ok` — but
only `chain_ok` and `chain_rel_l2` are written back; **`result["ok"]` is never reassigned.**
`leaderboard.py` line 84 reads `chk["ok"]` from the JSON, so a chain failure is structurally
incapable of reaching the ranking, and line 6's promise that "a backend that failed
correctness is shown but never ranked" is not kept. The process exit code is correct; nothing
reads it. **The r5 verdict called this "the single highest-priority harness change" and "one
line". It was not made. This round it stopped being hypothetical: the entire L=6 cell is now
ranked on a failed gate.**

**The gate is measuring the seed, and I can put a number on that.** `baseline_matrix.c` is
**byte-identical between `impl_5` and `impl_6`** (verified by `cmp`), and its L=6 chain drift
moved 1.687e-09 → 4.389e-08, a factor of **26**. All six library binaries *passed* at L=6 in
r5 (2.18e-10 – 4.16e-10) and all six *fail* in r6 (6.17e-09 – 1.15e-08), a factor of 15–28,
on code that did not change. The only thing that changed is the seed
(`sweep.sh` derives it as `SEED + L*1000 + B`, and the round's seed is not recorded, §0.4).
So the whole L=6 cell moved ~20–26× from the seed alone, on a chaotic map
`state ← FFT(state)+c` / `(1+|·|)` iterated 4,856 times.

`check.py`'s own comment states the design basis: "a 1-ulp input perturbation ends at
4.8e-12 over the longest chain, so exact implementations pass with ~8× margin." **That
conditioning claim is falsified by this round's own data by four orders of magnitude.** The
plainest exact implementation in the tree — the O(L⁴) row-column dense DFT that the harness
is validated against — lands at 4.4e-08. The constant is not merely mistuned; the model
behind it does not describe the L=6 chain.

**Did the two L=6 entries also spend accuracy for speed? Yes, and here is the decomposition.**
`L6_unrolled` flipped its runtime default from the exact FMA-Heron ladder to the
uncompensated fast tier this round (its scored description string says `tier=fast`), which is
where most of its −9.9% came from; `L6_pfa` continues to ship its fast tier. On their own
seed-42 controls the fast arms drift 2.292e-09 (`L6_unrolled`) and 3.247e-09 (`L6_pfa`),
against exact arms at 1.40e-09 and 1.227e-09. So of the ~40× degradation in the scored L=6
drift versus r5, roughly ×20 is the seed and roughly ×2 is the tier choice.

**Ruling.** I am not calling either L=6 entry a fast wrong answer, and the reason is
arithmetic, not charity: our two entries sit at 9.5e-09 and 1.16e-08, *inside* the band that
every library occupies (6.2e-09 – 1.15e-08) and **4.5× better than the exact dense
reference**. A gate that rejects `baseline_matrix` by 90× is not adjudicating our
implementations. So:

* **The L=6 cell is reported as measured but is NOT certified.** No entry at L=6 qualifies
  under `CURATION.md` ground 1 ("fastest *correct* entry") this round.
* Neither L=6 entry is disqualified, and neither is credited with a speed result. Both are
  kept for the record and for having beaten every library by 3.2× (§7, grounds 3 and 4).
* Both entries documented their tier choice openly, keep the exact arm one environment
  variable away, and asked the monitor — for the third round running — to set the policy
  *before* the round. That ask is now overdue enough to be the monitor's failure, not theirs.
* A further sign the gate is not discriminating: `L6_pfa`'s and `L6_unrolled`'s *relative*
  accuracy ordering **inverts with the seed**. On seed 42 `L6_unrolled` is the more accurate
  arm (2.292e-09 vs 3.247e-09); on the scored seed it is the less accurate one (1.158e-08 vs
  9.516e-09).

### 3.2 Nothing is missing, nothing failed to build, nothing crashed

* `failures.txt` **does not exist**. No entry crashed or hung.
* `agents/exits.txt` records **exit=0 for all 19** implementer agents.
* All 19 implementations present in `impl_6/` (+ `baseline_matrix.c`), all 19 appear in
  `leaderboard.txt`, and all 19 strategy records were updated this round. No entry is
  promotable but recordless.
* `timing.err` contains 14.6 KB of `"<entry>: does not support L=<n>"` notices (expected —
  the sweep offers every case to every binary) plus nine in-plan tuner telemetry lines from
  `L13_direct` / `L13_rader`. No errors.
* **`build_errors.txt` is non-empty this round (381 B) but contains zero errors** — it is one
  warning, and it belongs to the L=36 winner:

  ```
  ./impl/L36_mixedradix.c:1137: warning: "MAPPAIR_D" redefined
  ./impl/L36_mixedradix.c:1165: note: this is the location of the previous definition
  ```

  `L36_mixedradix.c` uses the `#include __FILE__` self-inclusion idiom, and on one pass
  `MAPPAIR_D` is aliased to `MAPPAIR_A` at line 1165 before being redefined at 1137. This is
  not a correctness finding — the entry builds, ships, and passes both gates with 424×
  margin — but *which* pair-map macro is in force at that site currently depends on include
  order rather than on intent, in the fastest entry at the geometry. It should be resolved
  rather than carried into a seventh round.

### 3.3 Every panel entry passes the single-transform gate; 17 of 19 pass the chain gate

I read the raw `c_*.json` rather than trusting the leaderboard column, precisely because the
leaderboard column cannot see chain failures (§3.1). All 19 pass the single transform
(`rel_l2` 2.27e-16 – 4.46e-16 against tol 1e-12). On the chain gate, the two L=6 entries fail
(§3.1) and the other 17 pass. Margins against `eff_tol = max(1e-12, 1e-13·m)`:

| L | tol | tightest panel entry | margin |
|---|---|---|---|
| 6 | 4.86e-10 | **L6_pfa 9.52e-09 — FAIL** | **0.05×** |
| 8 | 2.57e-10 | all three 3.296e-12 | 78× |
| 13 | 1.28e-10 | L13_direct 1.19e-13 | 1080× |
| 17 | 9.8e-12 | L17_matrixsimd 1.89e-14 | 518× |
| 23 | 1.65e-11 | L23_rader 2.67e-14 | 618× |
| 36 | 6.4e-12 | L36_pfa 1.81e-14 | 354× |
| 45 | 1.77e-11 | L45_pfa 4.77e-14 | 371× |
| 64 | 1.34e-11 | **L64_radix8 1.934e-12** | **6.9×** |

Two of those want naming, and one L=8 curiosity is worth recording.

### 3.4 L=64 has lost its accuracy-diverse arm, deliberately

ice_r5 flagged `L64_blocked`'s 7.4× margin as "the thinnest real one on the board" and noted
that its rival `L64_radix8` passed the identical case at 3.90e-14 — 46× more accurate. This
round `L64_radix8` **adopted `L64_blocked`'s 15-op cubic reciprocal ladder for −2.8%**, and
its drift moved 3.90e-14 → 1.934e-12. Both L=64 entries now sit at ~1.93e-12 against a
1.34e-11 budget: **6.9× margin, the thinnest pair on the board, and no arm left with room.**

To `L64_radix8`'s credit it did this with eyes open: it gates the tier at create time
(`CHAINBAR = 4e-12` for the cubic ladder, 1e-12 for exact), verifies the real dispatcher
against an exact scalar chain at m = 1,2,3,4 in both boundary parities, degrades
`ckind 2→1→0` on any failure, and keeps the exact ladder behind `-DFFT64R_MAPEXACT` with a
note that it is "mandatory if this code is ever pointed at L=6/8/13 chain lengths". That is
the right engineering. The board-level consequence stands anyway: **if the L=64 chain is ever
lengthened, both entries fail together**, and there is no longer a conservative arm to fall
back to. Flagged, not penalised.

### 3.5 The round's best correctness finding is `L13_direct`'s, and it is a panel-wide hazard

The panel's dominant idiom is now "race several bit-identical arms at plan time and adopt the
winner". `L13_direct` discovered that the bit-identity half of that contract is easier to
break than anyone assumed. Its first volume-major build differed from the lazy arms at
**7,381 of 70,304 points at m=2** (drift 1.799e-13 vs 1.780e-13 — *both passing*, which is
exactly why it would have shipped). Bisecting on the node: `-ffp-contract=off` builds were
identical, so the cause was gcc's `convert_mult_to_fma` making **different choices at
different inline instances of the same map** — one site contracted a map output into
`fma(u,s,b)`, so an *unrounded* map output entered the FFT there and not elsewhere.

Why this matters beyond L=13: if raced arms are not bit-identical, two scoring processes can
pick different arms, and the repeatability `cmp` breaks — or worse, does not break and the
output is process-dependent. `L13_direct` pinned the contraction-sensitive expressions and
verified by `cmp` at m = 2/20/1278. Two other entries reached the same place independently:
`L23_rader` made its new `chAB` chain race **telemetry-only, never a pick**, explicitly
because "map styles are different bit classes and a timing-based pick would let output flip
across processes"; `L45_pfa` made its two axis-order families compile-time-selected for the
same reason, with the pool's arm 0 setting the bit reference and the other family
auto-disqualifying. **"Same DAG per point" does not imply "same bits" under
`-ffp-contract=fast`.** This belongs in the panel brief.

Related good practice worth naming: `L17_winograd` shipped a from-scratch chain rewrite and
gated it at create time against the trusted r5 chain at m=5 and m=6 in both parities,
admitting the new arm only on agreement to 1e-12 and publishing `rok=1` in its description
string. In a round where the harness gate was blind at one geometry, in-entry gates like that
are the only thing standing between the panel and a fast wrong answer.

### 3.6 A minor curiosity: the three L=8 entries produce bit-identical chain output

`c_L8_radix8`, `c_L8_fusedaxes` and `c_L8_batchsimd` all report `rel_l2 = 2.267294067970892e-16`
and `chain_rel_l2 = 3.296181007855041e-12` — identical to the last digit, where the libraries
at the same case spread over 3.3e-12 – 9.0e-12. Their records explain it: all three ship the
same radix-8 codelet arithmetic in the same value order with the same
`rsqrt14 + 2 Newton + 1 divide` map ladder, and every change any of them made this round was
argued and cmp-verified to be bit-identical (`L8_fusedaxes`'s hp split, `L8_radix8`'s
broadcast-1.0 FMA trick, `L8_batchsimd`'s in-place relayout). So this is convergent design,
not a harness artifact. Recorded because a monitor seeing three identical fingerprints should
check, and because it means **L=8's three entries now differ only in schedule** — a diversity
observation for §6.

---

## 4. Claimed versus measured

Per §0.2 there is no machine difference to attribute anything to: every record measured on
the scoring node. The audit is therefore a clean one, and the result is that **all 19
predictions landed, the largest deviation is 1.7%, and the deviations go in both
directions.**

| entry | implementer's prediction | scored | Δ |
|---|---|---|---|
| L6_unrolled | 0.289–0.293 quiet (dev 0.291) | 0.291 | in range |
| L6_pfa | 0.296 (min/median, two runs) | 0.296 | 0.0% |
| L8_fusedaxes | 0.555 tuned-driver, twice | 0.555 | 0.0% |
| L8_radix8 | 0.570 warm, 4/4 readings | 0.569 | −0.2% |
| L8_batchsimd | 0.574–0.576, 3 leases | 0.575 | in range |
| L13_direct | ~5.25–5.30 (dev floor 5.265) | 5.287 | in range |
| L13_rader | ~5.38–5.40 | 5.377 | 0.1% better |
| L17_matrixsimd | ~11.6–11.9 (dev 11.905) | 11.935 | +0.3% over |
| L17_winograd | ~11.7–12.5 (dev 11.68–11.78) | 11.649 | 0.4% better |
| L17_rader | ~12.2–12.5 (dev 12.339) | 12.284 | in range |
| L23_matrixsimd | ~35.6–36.1 (dev 35.88) | 35.888 | in range |
| L23_rader | ~37.3–37.6 | 37.172 | 0.3% better |
| L36_mixedradix | ~99–103 (dev 100.5) | 100.801 | in range |
| L36_pfa | ~105–108 (dev 106.5) | 106.249 | in range |
| L36_pencilfused | 105–110 (dev 107.92) | 106.908 | in range |
| L45_pfa | ~260–265 | 264.111 | top of range |
| L45_mixedradix | fast-state 262.8–264.9 | 264.977 | just over |
| L64_blocked | 637.4–642.7, three windows | 638.757 | in range |
| L64_radix8 | dev 672.0 | 660.301 | **1.7% faster than dev** |

The one entry whose scored number is materially *better* than its own development floor is
`L64_radix8`, which is the wrong direction for any hostile-machine story. The residual
scatter is explained the same way it was in r5, and now by more records: **the node sits in
one of two whole-run clock/contention classes, and readings convert between them by
×3.3/2.9.** `L45_pfa` watched byte-equivalent builds read 264.3 → 275.2 across one day while
MKL stayed flat at 758–765; `L36_pfa` reads 121.0 = 106.4 × 1.138 in a 2.9 GHz window;
`L23_rader` and `L45_mixedradix` both document a currently-elevated B=1 cell affecting
*unmodified r5 exemplar binaries* (`L23_rader` built `exemplars/ice_r5/L23_rader.c` directly
to prove it: 42.96 at B=1 against its own 38.26 last round). Taking the min over three
processes filters most of this, which is why the table above is as tight as it is — and why
this round's much quieter window produced both a tighter table and the L=45 board regression
in §2.

**No entry's claim is far from what was measured, so there is nothing to attribute to the
machine.** The one thing worth flagging in the other direction: the in-create tuner races
report *levels* 2–4 µs above the driver's steady state at L=17 and L=36 (`L17_winograd`:
in-create mrot8 14.85 vs driver 11.7 in the same process; `L36_pfa`: in-arena ~17%
pessimistic), because `create` races on a colder core. Those strings are on the leaderboard's
`backends:` lines and are **ranks, never levels**. A future monitor should not read them as
timings.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

**§4.4 — "Split vs interleaved complex" — moved decisively, and it moved *against* the
corpus, reopening an item the corpus had marked CLOSED.**

§4.4 was annotated **"CLOSED by §08 §5.4, in the corpus's favour, with three independent
sources"** — Popovici/Franchetti/Low's 1.3–2× on `DFT_n ⊗ I_ν` kernels, FFTc 2.0's IPC 0.13
→ 2.59, and ducc0 shipping split — with the residual open question reduced to "only the
*granule*". This round four entries tested it directly on the node, and the corpus's verdict
does not survive contact with our kernel shapes.

**The decisive experiment is `L45_pfa`'s, because it is a full build rather than an
estimate.** It transcribed both PFA-45 codelet families into split form, sign-folded, with
every `SWAP+VPAIR` crossing becoming one FMA on the other component — so per-point bits are
*identical* to the interleaved arms, and the cross-arm `memcmp` gate passed on the first
build. The codegen audit confirms the promise the corpus makes: **zero shuffle-class
instructions in all six codelet blocks**, all gathers memory-folded `vbroadcastsd`, all
116,766 codelet swaps per volume deleted, two-pipe floor improved 108.9 → 96.1 µs/vol.

**It lost by 30–38%, consistently, in the same windows.** The decomposition names the
mechanism:

| probe | split | interleaved |
|---|---|---|
| p1 (z+y, no map) | 203–228 µs/vol | 153 |
| p2 (x pass) | 66.9–76.5 | 65–66 |
| p1 with staged map | 266–300 | 243.9 |
| interleave ↔ split conversion | 83–97 one-time (2/177 steps, free) | — |

The split z+y pass is **load-port-bound**: the y↔z corner turn's gathers degrade from 16-byte
`vbroadcastf64x2` granules to 8-byte scalar broadcasts, 16 per vector-pair instead of 4,
≈540k load-class uops per volume ≈ 93 µs of load-port floor against the interleaved path's
60 µs FMA floor. Both arms run at 2.4–2.5× their binding floor, so the split arm loses by
exactly its extra load uops. And the fix is provably unavailable in this shape: 16-byte
granules need a y-pair-interleaved *source* layout, and the state's writer (lanes = kz at
fixed ky) can never emit y-pairs full-width. **The conservation law is the transferable
sentence: the y↔z corner turn must cross the volume somewhere, and 8-byte lanes double its
uop cost. Deleting the port-5 shuffle term buys you a larger port-2/3 term.**

Three independent corroborations landed in the same round, at three other geometries, each
with its own arithmetic:

* **L45_mixedradix** rejected SoA before building: ~+90k load uops/volume (~15 µs of
  load-port floor) to delete 116.8k p5 swaps worth ~20 µs — net ~5 µs on paper, on passes
  measured 1.8× *above* their port floors, "i.e. the ports are not what binds."
* **L17_matrixsimd** costed it and said do not build: for a lanes-are-lines kernel the 8×8
  real transposes cost 0.375 shuffle/double against 4×4 complex tiles' 0.29, `MULI` is
  already just a sign-folded swap, and the FP count is unchanged — it loses before any
  deinterleave saving applies.
* **L6_pfa and L6_unrolled** both kept it parked for a third round, and `L6_unrolled` now
  states why the L=64 argument does not transfer down: at L=6 the ymm y+z stage's permutes,
  not the map unpacks, dominate the shuffle bill.

**What §4.4 should now say.** The corpus's sources are not wrong about their own kernels;
they measured `DFT_n ⊗ I_ν` shapes where the batch axis is in the lanes and no corner turn is
required. **The finding to record is the boundary condition: split-complex pays when the
vectorised axis is the batch, and loses when the vectorised axis is spatial and a y↔z corner
turn must cross the volume — because the shuffle term it deletes is replaced by a larger
load-port term.** That is a primary measurement at four of our eight geometries, one of them
a complete bit-exact build, and it is the first time this panel has *reopened* a closed
corpus item rather than confirming one. §4.4's "the open question is now only the granule" is
exactly backwards: the granule *is* the whole question, and 8 bytes is the answer we are
stuck with.

**Secondary movement, §4.3 (axis fusion / pass count) — the L1-regime law is now quantitative
and it is not about pass count at all.** ice_r5 closed the L1↔L2 regime against fusion with a
ROB-width mechanism. This round L=8 bracketed the law from both sides *in the same round*,
and produced the round's sharpest cautionary tale:

* `L8_radix8` built the half-pass split as an **unbalanced ~84/~260-uop cut**, measured
  **+10%**, added it to its list of negatives (v3 fused +15%, lazy map +11%, this +10%) and
  concluded: "**v2 is a genuine local optimum, and the structural search around it is
  CLOSED.**"
* `L8_fusedaxes` built the same idea as a **balanced ~150/~240-uop cut** on an alias-free
  pinned scratch frame and measured **−5.5%** — 0.555 scored, taking the geometry. It also
  built the *unbalanced* variant one stage later (`hpf`, map-only second pass) and measured
  **+10%**, matching its rival's number.

So the L=8 law is: ~150/240-uop groups beat ~380-uop groups beat ~420-uop groups, and +256
L1-resident memory ops are cheaper than one ROB-overflowing group — **but only if the cut is
balanced**; unbalanced cuts lose by ~10% in either direction. Pass count is not the variable;
group uop count and balance are. `L8_fusedaxes` moved from 1.28× to 1.21× of its p05 pool
floor on that basis. **The methodological lesson is worth as much as the number: "the
structural search is closed" was declared and refuted in the same round, at the same
geometry, on the same node, by two entries building the same idea with different cut
points.** No entry should retire a search space on one negative instance again.

**Secondary movement, §4.5 (padding and alignment).** §4.5's open question — measure L=8 at
(8,8) vs (9,9), check `L1-dcache-load-misses` — is still not answered in its literal form,
but its general form was answered emphatically and became the round's most-borrowed lever:
`L17_rader`'s 16-byte arena offset cost 2.3 of 15 µs (§2); `L23_rader` and `L23_matrixsimd`
padded state rows 23 → 24 complex so every chain store lands on whole lines (−0.3 to −1.1 µs
each); `L45_mixedradix` borrowed both and padded 45 → 48 for −1.0/−1.3% within-window;
`L17_matrixsimd` padded rows 17 → 20 with a deliberately *odd* line-count slab stride (101
lines) for set spread, and attributes to alignment plus in-place a drop from ~145 to ~81
core-cycles per chunk. And `L36_pencilfused` found the tail risk: a page-phase sweep over its
arena read 107.9–109.4 µs across seven phases and **137.5 µs (+27%) at phase 2112** — a
one-page-wide alias hole between its subloop-B stores and its broadcast reloads. §4.5 should
be updated to record that alignment and page phase are routinely worth 1–3% at every geometry
on this node, occasionally 15%, and once 27%, and that `make_noncritical()`-style guards
should be the default rather than a diagnosis.

---

## 6. The single highest-value thing the next round should attack, per geometry

**L = 6 — fix the gate. Do not tune. This is harness work and it is now the third round of
asking.** The cell is currently unscoreable (§3.1) and both entries have been asking for a
policy decision since r4. Two changes, both small:

1. Wire `chain_ok` into `result["ok"]` in `check.py`. One line. Without it the ranking
   cannot see a chain failure, and this round it ranked nine of them.
2. Recalibrate `eff_tol` from a **multi-seed worst case** across the six libraries and
   `baseline_matrix`, and **announce the floor before the round starts**. The data now
   exists: any floor ≥3.5e-9 admits both fast tiers on seed 42, ~1.5–2e-9 admits only the
   exact arms, and on the r6 seed *nothing below ~5e-8 admits `baseline_matrix`* — which is
   the number that says the current per-step model (`1e-13·m`) is the wrong functional form
   for L=6, not just the wrong constant. Both L=6 entries are one environment variable from
   either policy (`L6_TIER=exact`, `L6_CMAP`/`sep`), so re-scoring costs nothing once the
   floor is chosen.

Everything else at L=6 is worth ~1%: both records agree the only structural lever left is a
fully split-complex chain state, and §5 now argues it will not pay there either.

**L = 8 — settle the cut-point curve, and get the PMU.** The group-size law has two measured
points on one side and one on the other (§5), and the winning cut was found by accident of
which cut each entry happened to build. Sweep it: `L8_radix8`'s untried "before trans8"
116/290 cut, `L8_fusedaxes`'s 150/240, its `hpf` 84/260-equivalent, and at least one
intermediate — the search costs one round and would turn a lucky 5.5% into a curve. Second,
and jointly requested by all three entries: **the residual 1.21× over the p05 pool floor
cannot be diagnosed further without counters.** `L8_radix8` names it ("the only instrument
left is the PMU; bare-metal counters work, `perf` is absent — needs a `perf_event_open`
harness"); `L8_fusedaxes` has a dormant `-DL8_PMC=1` probe waiting on it and a specific
hypothesis (MITE vs DSB delivery: the hp step is ~5.5k instructions against ICX's ~2.3k-uop
DSB). Building that harness once serves the whole panel and is the highest-leverage
infrastructure item after the L=6 gate. Note also §3.6: all three L=8 entries now produce
bit-identical output and differ only in schedule — the geometry has no structural diversity
left to lose, so a seeded-from-scratch fourth design would be worth more than a fourth
schedule.

**L = 17 — cross-A/B the two engines; stop treating arithmetic as the wall.** The r5
directive was "attack the 17-point module's operation count; scheduling is finished."
`L17_matrixsimd` examined that directive and **rejected it with arithmetic before building**
— our kernel already *is* the symmetric/antisymmetric split, and the rival pipeline's faster
11.16 µs mark uses *more* FP per line than we do (~168 vs 148 ops/chunk). Then all three
entries gained 6–19% from pass shape, alignment and lane custody, and none from arithmetic.
So the directive was wrong and §4.2(b) is not the lever. What the round produced instead is
**two different, independently-derived answers to the same structural question, 2.5% apart**:
`L17_winograd`'s rotating-lane state (one spatial axis in the lanes, alternating per step) and
`L17_matrixsimd`'s padded-aligned arena with in-place strided passes. Each was A/B'd only
against its own predecessor. The next round's highest-value move is to **cross them** — the
rotating-lane state inside the padded arena, and `L17_rader`'s cheaper `tr=2` lane-3 tail into
both — rather than a third round of each tuning its own. This is also the geometry with the
board's largest library margin (7.41×) and the smallest remaining gap to the archived rival
pipeline (11.65 vs 11.16, and ours is exact at 1.6e-14 against their 5.7e-14).

**L = 36 — soften the map-carrying pass's call body; the frontier is now unanimous.** All
three entries converged on the same protocol this round (eager map at the final pass's store
sites + a per-volume permuted `c` copy + a 2:1 divider:ladder hybrid) and finished within 6%
of each other, and all three independently located the same residual: the map-carrying pass
runs ~465 cycles per call against a ~310-cycle port floor, and the ~155-cycle remainder is
**ROB/retire pressure** — ports, prefetch, bandwidth, dTLB and divider occupancy have each
now been separately excluded by measurement. The one untried shape, named by two of the three,
is to lag the map one call behind (map call *n−1*'s outputs from a small L1 stash while call
*n*'s DFT issues), softening the ~750-uop body so the long map chains retire against fresh
independent work. Two guardrails from this round: the z/load sites are the **wrong** home for
map work (three independent measurements now: `nSh` 124.4 / `nSb` 126.2 vs `nF` 114.9; r4's
110.7 vs 103.3; `L36_pencilfused`'s staged-map post-mortem), and **any memory round trip for
the map costs 13–14 µs regardless of buffer size or layout** (`L36_pencilfused` measured it
twice: `mr` staging 13.1, group stash 121.9 vs 110.2). `L45_pfa`'s independent +90 µs/vol
price on a *staged* map is the same finding at a fourth geometry and should be read by anyone
at L=36 still carrying staging.

**The other four, one line each.** **L=13:** 1.7% apart with both arms improving and the
engines now identical; the one unbuilt item both name is deeper pass fusion (X feeding the
plane phase without the full `t1` round trip) — and `L13_direct`'s `-ffp-contract` protocol
(§3.5) should be adopted panel-wide before anything else is raced. **L=23:** `L23_matrixsimd`
holds by 3.6% and the r5 flip is confirmed stable, so the re-race directive is discharged;
the two remaining >1 µs levers are ymm 4×4 tile transposes in the Y/Z chunks and the
transpose-free Z-store layout, and this is a compute-bound cell, which is the condition
`L17_rader` flagged as where ymm tiles might pay. **L=45:** stop optimising. Three
consecutive rounds inside 1%, a scored cell that went *backwards* while both entries improved
in matched windows (§2), and the round's genuinely valuable product was `L45_pfa`'s
split-complex post-mortem, not another 1%. Spend L=45's next round on the B=1 row and a
statistic that can separate the arms. **L=64:** both entries are now
arithmetic/latency-bound rather than traffic-bound (`L64_radix8` moves ~20 GB/s against a
~33 GB/s ceiling), so the target is the line-FFT stages (305 µs for 1024 lines ≈ 1.5
uops/cycle in a two-stage bounce) — and someone should restore an accuracy-diverse arm
(§3.4).

**And three harness items that outrank all eight geometries:**

1. `check.py`: wire `chain_ok` into `result["ok"]`, and recalibrate the L=6 gate (above).
2. Add `B=1` rows to `cases.txt` — sixth round of asking, one line, and two geometries have
   now shown the batched and non-batched orders disagree (§0.3, §1).
3. **`tryout.sh` is broken and every single entry works around it by hand.** All 19 records
   report it: line 36 expands `$W` before it is defined (aborts under `set -u`), and line 49's
   single-quoted `'$W/c.bin'` expands empty on the remote so `check.py --map-check` silently
   *skips the chain gate*, and the `&&` chain then silently skips the repeatability `cmp`.
   Five to six rounds running. Every chain-correctness number in every record this round was
   obtained by a hand-run of `check.py` with a `W=` prefix. That is a lot of trust placed in
   nineteen agents remembering to do it manually, in a round where the scored gate was also
   blind. Fix it or bless the workaround in the brief. Also record the seed in
   `environment.txt` (§0.4).

---

## 7. Curation

Applying `docs/CURATION.md`. Ground 4 ("anything that beat a library baseline") is nearly
vacuous — all 19 entries beat all 6 libraries at every geometry — with one crucial exception:
**at L=6 it is the only ground available**, because nothing there is certified correct
(§3.1). All 19 have current strategy records, so nothing is excluded for a missing record.

**Ground 1 — fastest correct entry per geometry (7, not 8):** `L8_fusedaxes`, `L13_direct`,
`L17_winograd`, `L23_matrixsimd`, `L36_mixedradix`, `L45_pfa`, `L64_blocked`. **L=6 has no
ground-1 entry this round.**

**Grounds 3 + 4 — L = 6, both entries, on the record and on the library margin (2):**
`L6_unrolled` (0.291, fastest as measured) and `L6_pfa` (0.296). Neither can be promoted as
"fastest correct"; both beat every library by 3.2×; and both records are the primary
documentation of the gate pathology, with the multi-seed numbers a recalibration will need
(`L6_pfa`: MKL 6.2e-10, FFTW 1.5e-9, ducc0 1.5e-9, numpy-vs-numpy 1.1e-9, its own exact arm
1.227e-9; `L6_unrolled`: fast 2.292e-9, exact 1.40e-9, and the matched-tier table). Promoting
only the faster one would put the tier flip in front of the next panel without the argument
for it. **The round note must say the L=6 cell is void.**

**Ground 2 — structurally different runner-up, close (7):**

* `L8_radix8` — 2.5% behind, and the entry that *defines* the L=8 design space: v3 fused
  +15%, lazy map +11%, unbalanced half-pass +10%, and the divider bracket (0 divides +6.7%,
  64 fastest, 128 +46%) that proves the shipped ladder sits at the sweet spot. It is also the
  donor of the idea that beat it. The next panel needs both sides of §5's group-size law in
  code.
* `L13_rader` — 1.7% behind; Rader-13 CRT against a dense conj-folded matrix, the §4.2
  question at a second prime.
* `L17_matrixsimd` — 2.5% behind and genuinely different from the winner: a padded aligned
  *interleaved* arena with in-place strided passes, against `L17_winograd`'s rotating-lane
  split-component state. §6's cross-A/B is impossible unless both are in front of the panel.
* `L23_rader` — 3.6% behind; Rader-23 folded pair against the dense 23×23, and the origin of
  the padded-row layout that `L45_mixedradix` then borrowed to ship a win.
* `L36_pfa` — 5.4% behind the winner, 0.6% ahead of third; the GT-PFA `n1_9` DAG arm, the
  round's independent confirmation of the 2:1 divider ratio, and an 8.0% mover.
* `L45_mixedradix` — 0.3% behind in a cell that is a tie for the third round and has now
  inverted twice. Promoting one arm would be reading noise as a result.
* `L64_radix8` — 3.4% behind, the round's biggest mover (−34.7%), and the entry that refuted
  the promoted winner's published impossibility claim by axis alternation. This is a ground-2
  promotion now, where in r5 it was a ground-3 one.

**Not promoted (3), with reasons:**

* **`L8_batchsimd`** (0.575, 3.5% behind) — improved 3.5% creditably, and its diagnosis of
  its own tuner (a 2% hysteresis band vetoing a measured 1.7% win) is a good finding. But
  §3.6 establishes that all three L=8 entries now emit bit-identical output and differ only
  in schedule, and it adds no structure the promoted pair lacks. Near-duplicate rule.
* **`L17_rader`** (12.284, 5.5% behind) — the round's joint-largest relative mover and the
  source of its most-borrowed lever (the 16-byte alignment forensics, which `L45_mixedradix`,
  `L23_rader` and `L23_matrixsimd` all cite). Excluded anyway on its own honest description:
  "the engine is now matrixsimd's minus their ymm tail plus my cheaper `tr=2` tail, aligned
  like theirs, sign-folded like theirs" — a near-duplicate of the union of two promoted
  entries. Its two original artifacts (the `tr=2` lane-3 tail store, which `L17_matrixsimd`
  is advised to take, and the alignment-audit protocol) are preserved in
  `strategies/L17_rader.md`, which is tracked independently of `exemplars/`.
* **`L36_pencilfused`** (106.908, 6.1% behind, 0.6% behind second) — a good round with real
  products: the four-way phase-split protocol (`NOMAP`/`MAPNOP` diagnostics), the second
  independent 13–14 µs price on any staged map, and the +27% page-phase alias hole. But all
  three L=36 entries now run PFA 4×9 with the same eager-map-plus-2:1-hybrid protocol, and
  the near-duplicate rule allows two. It is the slowest of the three; its findings live in
  its record. (Symmetric to r5, where `L36_pfa` was excluded for the same reason.)

The round note for `exemplars/ice_r6/NOTES.md` should record: that the L=6 cell is void and
`check.py` ranked nine failed gates (§3.1); §4.4 reopened against the corpus with a bit-exact
build and a load-port conservation law (§5); the L=8 group-size law bracketed from both sides
in one round, with "the search is closed" declared and refuted simultaneously (§5); the
`-ffp-contract=fast` bit-identity hazard that the whole panel's race-and-adopt idiom depends
on (§3.5); and that L=64 no longer has an accuracy-diverse arm (§3.4).

PROMOTE: L6_unrolled L6_pfa L8_fusedaxes L8_radix8 L13_direct L13_rader L17_winograd L17_matrixsimd L23_matrixsimd L23_rader L36_mixedradix L36_pfa L45_pfa L45_mixedradix L64_blocked L64_radix8
