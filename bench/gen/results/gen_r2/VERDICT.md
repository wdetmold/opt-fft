# VERDICT — round gen_r2

Monitor's report. Measured on `a80n0.lqcd.mit`, slurm job 438682, 2026-08-24T06:00:13-04:00,
gcc 11.4.0, governor `schedutil`.

## 0. Two corrections to the monitor brief before any number is read

Both are material to how the rest of this file must be read.

**(a) The geometry set.** The brief asked for headlines at `L = 6, 8, 17, 36`. Those are the
*ice* campaign's four fixed cubes. This is the GENERALIZE campaign; its frozen acceptance
suite (`cases.txt`) is eleven cells, none of which was ever tuned by a prior round:

```
10:64:1000  12:64:600  15:32:600  20:32:256  25:16:256  27:16:200
31:16:140   32:8:250   40:8:128   50:4:128   100:1:64
```

Section 1 reports all eleven. `docs/LITERATURE.md` is still written for the old four sizes,
which is where the stale list comes from; its §4 open questions are nonetheless live here and
§5 maps them onto this suite.

**(b) The machine.** The brief describes the scoring node as a Xeon Gold **5218** (Cascade
Lake, downclocked AVX-512, 1 MB L2) and the implementers as developing on Sapphire Rapids.
`environment.txt` says otherwise: the scoring node is a Xeon Gold **6326** — **Ice Lake-SP**,
2.90 GHz, 64 hardware threads, 1.25 MB L2, two 512-bit FMA pipes at full clock. And every
implementer record states its numbers were taken on **a80n0 itself**, via `tryout.sh` on a
leased core of the same node.

The consequence for §4 is direct and inverts the brief's instruction: **there is no machine
difference to attribute anything to.** Claim-versus-measured divergence in this round is a
*node-contention* story, not a microarchitecture story, and §4 treats it that way. The
"MKL spans 2.9× between those machines" calibration does not apply to this round at all — MKL
moved less than 1 % between gen_r1 and gen_r2 at every one of the eleven cells.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

### 1a. Batched (the ten batched cells of the frozen suite)

| L | B | fastest correct panel entry | µs/xform | best library | µs/xform | panel margin |
|---|---|---|---|---|---|---|
| 10 | 64 | **gen_batchlane** | 1.162 | mkl_dfti | 4.560 | **3.92×** |
| 12 | 64 | **gen_batchlane** | 1.933 | mkl_dfti | 7.732 | **4.00×** |
| 15 | 32 | **gen_pfa_small** | 4.466 | mkl_dfti | 16.457 | **3.69×** |
| 20 | 32 | **gen_pfa_small** | 13.213 | fftw3_patient | 45.160 | **3.42×** |
| 25 | 16 | **gen_powp** | 32.001 | fftw3_measure | 109.056 | **3.41×** |
| 27 | 16 | **gen_powp** | 44.970 | mkl_dfti | 144.494 | **3.21×** |
| 31 | 16 | **gen_rader** | 86.913 | ducc0_c2c | 714.517 | **8.22×** |
| 32 | 8 | **gen_pow2** | 58.130 | mkl_dfti | 171.572 | **2.95×** |
| 40 | 8 | **gen_pfa_large** | 201.049 | mkl2026_dfti | 403.335 | **2.01×** |
| 50 | 4 | **gen_pfa_large** | 473.024 | mkl_dfti | 946.682 | **2.00×** |

Every one of those ten margins is a win, and the panel's *worst* batched entry at each L still
beats `baseline_matrix` by 42–99×. The margin profile is the story: **3.2–4.0× at the small
composites, 8.2× at the prime, and a collapse to 2.0× as soon as the working set leaves L2**
(L=40 at 15.62 MiB, L=50 at 15.26 MiB).

At L=31 all seven panel entries that support it beat all six library baselines — even
`gen_bluestein`, the mandated existence fallback, is 1.88× faster than the best library. The
libraries are simply not competitive at a prime edge length; that is a fact about them, not an
achievement of the tail of the panel, and §7 declines to promote on it.

### 1b. Non-batched

The frozen suite contains **exactly one** non-batched cell, so this is the only scored B=1 row:

| L | B | fastest correct panel entry | µs/xform | best library | µs/xform | panel margin |
|---|---|---|---|---|---|---|
| 100 | 1 | **gen_pfa_large** | 4947.503 | mkl_dfti | 7796.570 | **1.58×** |

`gen_powp` is 1.02× behind at 5031.324 — a genuine two-family photo finish at the largest
volume (see §2).

For L=10…50 there is **no scored non-batched number**, and the monitor will not manufacture
one. What exists is implementer-reported B=1 on the same node in the same windows, which is
worth recording because it exposes the panel's largest structural hole:

| L | best panel B=1 (reported) | same-window MKL B=1 | verdict |
|---|---|---|---|
| 10 | gen_planner 6.23 · gen_batchlane 10.50 | 4.96 | **panel loses 1.26× (planner), 2.12× (batchlane)** |
| 12 | gen_layout 10.49 · gen_batchlane 17.54 | 8.32 | **panel loses 1.26×, 2.11×** |
| 15 | gen_planner ≈21.70 · gen_batchlane 40.88 | 18.03 | **panel loses 1.20×, 2.27×** |
| 25 | gen_powp 48.7 | — | 1.52× worse than its own batched cell |
| 27 | gen_powp 68.1 | — | 1.51× worse than its own batched cell |
| 31 | gen_rader 99.7 · gen_dense_prime 131.2 | — | +15 % / +6 % over batched |
| 32 | gen_pow2 58.84 | — | batch-invariant (+1.2 %) |
| 40 | gen_pfa_large 223.2 | — | +11 % over batched |
| 50 | gen_pfa_large 554.9 | — | +17 % over batched |

Two things fall out. The SoA-8-volumes-per-zmm specialists that win the batched rows
(`gen_batchlane`, `gen_pfa_small`) **lose to MKL by 2.1–2.5× at B=1** — the only cells in this
round where a library beats the panel. And the generic volume-major engines
(`gen_planner`, `gen_layout`, `gen_pow2`) are batch-invariant by construction and therefore
*overtake their own specialists* at B=1. That inversion is the highest-value fact in this
section and drives §6.

---

## 2. What changed since gen_r1, per geometry

Library baselines moved <1 % at every cell between the two rounds, so these deltas are the
panel's own.

| L | r1 leader | r2 leader | leader delta | crown | best-library margin r1 → r2 |
|---|---|---|---|---|---|
| 10 | gen_batchlane 1.174 | gen_batchlane 1.162 | **−1.0 %** | held | 3.87× → 3.92× |
| 12 | gen_batchlane 1.995 | gen_batchlane 1.933 | **−3.1 %** | held | 3.88× → 4.00× |
| 15 | gen_batchlane 4.686 | **gen_pfa_small 4.466** | **−4.7 %** | **changed hands** | 3.51× → 3.69× |
| 20 | gen_pfa_small 16.920 | gen_pfa_small 13.213 | **−21.9 %** | held | 2.65× → 3.42× |
| 25 | gen_powp 48.778 | gen_powp 32.001 | **−34.4 %** | held | 2.23× → 3.41× |
| 27 | gen_powp 66.078 | gen_powp 44.970 | **−31.9 %** | held | 2.19× → 3.21× |
| 31 | gen_rader 94.170 | gen_rader 86.913 | **−7.7 %** | held | 7.70× → 8.22× |
| 32 | gen_pow2 63.683 | gen_pow2 58.130 | **−8.7 %** | held | 2.69× → 2.95× |
| 40 | gen_pfa_large 200.515 | gen_pfa_large 201.049 | **+0.27 %** | held | 2.02× → 2.01× |
| 50 | gen_powp 473.484 | **gen_pfa_large 473.024** | **−0.1 %** | **changed hands** | 2.00× → 2.00× |
| 100 | gen_powp 5026.755 | **gen_pfa_large 4947.503** | **−1.6 %** | **changed hands** | 1.56× → 1.58× |

### Did anything regress?

**No — not one entry regressed outside its own measured run spread.** Four movements are
nominally in the wrong direction and all four are flat, not regressions:

* `gen_powp` L=50 473.484 → 474.028 (**+0.11 %**, spread 0.9 %) and L=100 5026.755 → 5031.324
  (**+0.09 %**, spread 0.1 %). Flat. But this is the round's one strategic cost: `gen_powp`
  spent the whole round on L=25/27 (−34 %/−32 %) and did not touch 50/100, so it lost both
  crowns to a rival that gained 1.7 % and 2.7 % there. Standing still at parity is how you lose
  a photo finish.
* `gen_pfa_large` L=40 200.515 → 201.049 (**+0.27 %**, spread 3.5 %). Its own record explains
  it honestly: the round's win was the `DFT25M` fused-store macro, and "40 has no DFT25".
* `gen_bluestein` L=10 15.648 → 15.730 (+0.5 %) and L=12 23.461 → 23.636 (+0.7 %), both inside
  spread; every cell at L≥15 improved 2.4–7.1 %.
* `gen_dense_prime` L=10 8.291 → 8.296 (+0.06 %). Its entire round went into L=31 (−29.2 %).

### The four big movers, and why they matter more than the leader deltas

| entry | r1 → r2 range | mechanism |
|---|---|---|
| **gen_race** | **7.8× to 41.0× faster** (391 380 → 9 502 µs at L=100) | stopped being an O(L⁴) test bench: adopted `gen_planner`'s whole engine via `GEN_PLANNER_LIB` and races its trees |
| **gen_layout** | **−45 % to −55 %** at all eleven | folded dense engine + THP/stagger arena; now beats MKL at L=20, 25, 27 and every library at 31 |
| **gen_planner** | **−12 % to −36 %** at all eleven | fused-leaf CT rewrite + in-plan race |
| **gen_twiddle** | **−28 % to −49 %** | zmm-lane demo via GNU vector extensions |

`gen_race` and `gen_planner` now land within **0.02–1.5 % of each other at all eleven cells**
(L=31: 530.955 vs 531.051; L=100: 9501.6 vs 9514.7). That is not a coincidence and not a
duplicate submission — `gen_race` says plainly that the engine is `gen_planner`'s and its own
contribution is the *pick*. §5 turns that into the round's headline literature result.

---

## 3. Adversarial pass: correctness, builds, crashes, absences

### 3.1 Correctness — nothing failed, and nothing is riding a tolerance

All three gates are green for all twelve panel entries at every cell they support.
`check.log` contains **zero** FAIL / NOT REPEATABLE / MISMATCH lines.

* Gate 1 (single call vs numpy, tol 1e-12): panel range **8e-16 … 4e-15** — three orders of
  margin.
* Gate 2 (two-step fused chain, tol 3e-14): observed **7.4e-16 … 3.7e-15** — worst case 8×
  inside tolerance.
* Gate 3 (chain end, 300× honest anchor, floor 1e-10): worst panel value is `gen_bluestein`
  at L=10, `ch=1.9e-13` against a 1e-10 floor — **526× inside** tolerance.

No fast-and-wrong answer survived, because none was submitted. The tightest ratio to an honest
anchor anywhere is `gen_batchlane` L=10 at 1.6× the anchor, which the brief explicitly forgives
as honest drift.

### 3.2 Build errors — none. Two compiler warnings, both benign, both named

`build_errors.txt` contains **no errors**. It contains two `-Waggressive-loop-optimizations`
warnings, and they land in the L=31 winner and its runner-up, so they get named:

* `impl/gen_dense_prime.c:1116` — in `fft3d_chain`'s scalar map tail
* `impl/gen_rader.c:696` — in `map_volume.constprop`'s scalar map tail

Both are the identical idiom: `size_t i`, `size_t npts`, tail loop `for (; i < npts; ++i)`
indexing `zp[2*i]`. GCC cannot bound `npts`, so it reports pointer-offset UB at iteration
2⁶⁰ = 1152921504606846976. **This is not a live defect**: the preceding `_mm512` loop consumes
`i` in strides of 8, so the tail executes at most 7 iterations, and both entries pass all three
gates at L=31 with a comfortable margin. It is worth recording for two reasons. First, it is
the same warning gen_r1 raised against `impl/gen_pfa_small.c:324` — `gen_pfa_small` fixed it,
and the warning then travelled into `gen_rader` with the "s6 map" it adopted verbatim from
`gen_dense_prime`. **A shared codelet propagates its diagnostics along with its speed.** Second,
the fix is one cast and it costs nothing, so there is no reason for `build_errors.txt` to be
non-empty next round.

### 3.3 Crashes and hangs — one, and it is the reference floor, not a panel entry

`failures.txt` has exactly three lines, all the same event:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is a `timeout` kill. `baseline_matrix` is the harness's library-free O(L⁴) floor;
at L=100 that is 3·10⁸ complex MACs per volume per axis pass and it cannot finish the m=64
chain in the window. This is expected, it reproduces gen_r1 exactly, and it is why the L=100
table has no `baseline_matrix` row. **No panel entry crashed or hung.**

### 3.4 Absences — all twelve entries are fully present; every decline is authorised

`timing.err` records 55 `does not support L=` declines. Cross-checked line by line against the
acceptance table in `PANEL_BRIEF.md`:

| entry | acceptance sizes (brief) | cells scored | verdict |
|---|---|---|---|
| gen_pfa_small | 10, 12, 15, 20 | 10, 12, 15, 20 | complete |
| gen_pfa_large | 40, 50, 100 | 40, 50, 100 | complete |
| gen_pow2 | 32 | 32 | complete |
| gen_powp | 25, 27, 50, 100 | 25, 27, 50, 100 | complete |
| gen_dense_prime | 31 (+ helps 10…20) | 10, 12, 15, 20, 31 | complete **+ 4 voluntary** |
| gen_rader | 31 | 31 | complete |
| gen_batchlane | 10, 12, 15 at B≥8 | 10, 12, 15 | complete |
| gen_bluestein | must run everywhere | all 11 | complete |
| gen_planner / gen_race / gen_twiddle / gen_layout | all (layers) | all 11 | complete |

**Nothing is missing.** No entry declined a size it owns, and `gen_dense_prime` ran four cells
it was not required to. All twelve agents exited 0 (`agents/exits.txt`).

### 3.5 The one real compliance failure this round: gen_pfa_large's plan time

Not a correctness failure, so it does not touch the standings — but it is the sharpest finding
in this section and the leaderboard's `setup` column makes it unarguable.

`gen_pfa_large` wins L=40, L=50 and L=100 while paying **0.488 s / 0.662 s / 6.339 s of plan
time on every single process**. It has not adopted `gen_race`'s wisdom cache — its own record
records the handshake as deferred *twice*. `gen_powp`, which is 1.02× behind it at L=100 and
0.2 % behind at L=50, does the same job in **0.006 s** because its pick is persisted.

The brief's budget is ≤60 s for a never-seen `(L,B)` and **≤50 ms with persisted wisdom**.
`gen_pfa_large` is inside the cold budget and can never enter the warm one, because it persists
nothing. For a round-6 library that is a defect in the winner. `gen_planner` has a milder form
of the same gap (0.364 s at L=100 against `gen_race`'s 0.005 s), and `gen_race` shipped the
string-wisdom API this round specifically to close it.

---

## 4. Claimed versus measured

The brief's premise here does not hold — see §0(b). Implementers developed on **a80n0 itself**,
the scoring node, so there is no Sapphire-Rapids-to-Cascade-Lake gap to attribute anything to,
and MKL moved <1 % between rounds at every cell. Agreement is correspondingly tight:

| entry | its own claim | scored | delta |
|---|---|---|---|
| gen_batchlane | 1.162 / 1.931 / 4.484 | 1.162 / 1.933 / 4.478 | **0.0 / +0.1 / −0.1 %** |
| gen_rader | 86.61–87.30 | 86.913 | in range |
| gen_pow2 | 58.4–59.8 (best 58.39) | 58.130 | −0.4 % |
| gen_dense_prime | 123.7–125.1 | 124.281 | in range |
| gen_powp | 32.6 / 45.6 / 469.9–485.1 / 5072 | 32.001 / 44.970 / 474.028 / 5031.324 | −1.8 / −1.4 / in range / −0.8 % |
| gen_race | all eleven cells | all eleven | **≤1.5 % everywhere** |
| gen_planner | all eleven cells | all eleven | −3.1 % … +0.2 % |
| gen_bluestein | all eleven cells | all eleven | −3.4 % … +0.1 % |
| gen_layout | all eleven cells | all eleven | −1.5 % … +1.2 % |

Four divergences exceed 3 %, and **all four are in the implementer's favour** — the scored run
was *faster* than the number they reported:

| entry | cell | claimed | scored | divergence |
|---|---|---|---|---|
| gen_twiddle | L=50 B=4 | 2098.3 | **1831.066** | **−12.7 %** |
| gen_twiddle | L=100 B=1 | 18938 | **17734.507** | **−6.4 %** |
| gen_pfa_small | L=10 B=64 | 1.22–1.29 | **1.162** | **−4.8 %** |
| gen_pfa_large | L=40 B=8 | 208.8 (quiet; 213–216 typical) | **201.049** | **−3.7 %** |
| gen_twiddle | L=25 B=16 | 181.9 | 174.933 | −3.8 % |

The cause is **node contention during development, not the machine**, and the implementers
diagnosed it themselves before the scoring pass. `gen_pfa_large`: "5008.8 (quiet;
5360–5490 under neighbours)". `gen_powp`: "L=100 windows this round were violent:
gen_pfa_large's binary measured 7905 in the same window where mine read 6270 (their quiet
number is 5009)". `gen_batchlane`: "a compute-bound kernel at 42+ GF/s sees neighbours'
AVX-512 load through the all-core turbo". `tryout.sh` leases a core on a *shared* a80n0; the
scoring pass owns it under slurm job 438682. So every divergence points one way — scored
faster than developed — which is exactly the signature of exclusive access, and the reverse of
what a downclock story would produce.

**One divergence is not noise and should not be filed as such.** `gen_powp` reports
`L=25 B=1 = 48.7 µs` against its scored `B=16 = 32.001 µs`. That is a real 1.52× structural
gap, not a window: its winning `soa` candidate is simply "not offered at batch % 8 != 0". Same
at L=27 (68.1 vs 44.970). It is a coverage limit the record states plainly, and §6 acts on it.

---

## 5. Which `docs/LITERATURE.md` §4 open question this round moved

### Moved decisively: §4.6, "Model versus search for the instruction schedule"

§4.6 records the corpus's internal fight — §01 says "you should not need a search phase",
§06 §6.1 corrects it on a point of fact (genfft's spill-optimality proof holds only for powers
of 4, and none of our sizes is one) and concludes the schedule is "the primary thing to search
at every one of our sizes". §06's evidence for the magnitude was SPIRAL's **2× spread** among
equivalent formulas. §4.6 ends "it is cheap to settle. Do it."

This round settled it, on this node, with receipts from both sides of the layer boundary:

**How much search buys over the best *model* pick** — `gen_planner`'s own A/B, model candidate
versus raced winner, same binary, same core:

| L | model pick | measured | raced pick | measured | search buys |
|---|---|---|---|---|---|
| 32 | `c8(d4)` | 282 | `c4(d8)` | 236 | **−16 %** |
| 100 | `c5(c5(d4))` | 10 185 | `c4(c5(d5))` | 9 574 | **−6 %** |

**And from `gen_race`'s persisted wisdom receipts** (`results/wisdom_a80n0.json`):
L=10 the race took `c2(d5)` at **+15.9 %** over the model best — the exact L=10 model miss
`gen_planner`'s r1 record had already confessed to; L=100 `c4(c5(d5))` at **+3.7 %**; L=31 model
and race **agree** on `rad31(c3(c5(d2)))`, with a 10.7 % margin to second place; L=12 is a
**genuine 0.6 % tie** and the tie doctrine deliberately keeps the model pick rather than
encoding noise.

**The answer, with a number: search is worth 4–16 % where the model mis-ranks (L=10, 32, 100),
and ~0 % where the model is right (L=12, 31) — not the 2× §06 feared, and not the nothing §01
predicted.** The independent confirmation is on the leaderboard itself: `gen_race` and
`gen_planner` finish within 0.02–1.5 % at all eleven cells, i.e. once the model-based planner
races its own top candidates internally, an external racer has nothing left to find. Search is
a *correction term on a good model*, not a substitute for one.

The cost side is now also measured, and it is the part §4.6 never priced: warm `create()` is
**2–4 ms** (`gen_powp`) to **5 ms** (`gen_race`), against the brief's 50 ms budget; worst cold
race 0.726 s at L=100 against 60 s. A 4–16 % kernel gain for 4 ms of amortised plan time is
not a close call. And `gen_race` found a second, non-performance reason the cache is load
bearing: `gen_powp`'s `soa` family is **not bit-identical** to its interleaved families, so
without a persisted pin a noise flip between the driver's two repeatability processes would
report NOT REPEATABLE. **Persistence is correctness infrastructure, not just speed** — that is
new, it is not in the corpus, and it belongs in §4.6.

### Also moved, with numbers: §4.5 (padding / 4 K aliasing) at a non-power-of-two L

§4.5 asks for the L=8 padding experiment and §08's addendum names the bigger hazard as the
memory order buffer — 4 K store-to-load aliasing — closing with "neither has ever been
checked". Two entries checked it at L=31 this round, independently, and agreed:

* `gen_rader`, same-window A/B against its own r1 binary as control: plane pad to 1148 (kills
  x-pass 4 K store→load aliases and line splits) **−5.0 %** (95.95 → 91.30); then z-row pad
  31→32 for full 64 B alignment and tail-free passes **a further −5.2 %** (→ 86.61). Total
  **−9.4 %** from layout alone, no arithmetic change.
* `gen_dense_prime`, step 6 of its ladder: padded 31×31×32 state, everything 64 B-aligned and
  mask-free, **136.9 → 125.1 (−8.6 %)**.

So §4.5's hazard is real at a *prime* edge length, not only at powers of two, and it is worth
**5–9 %** on Ice Lake-SP — single-digit, matching §08's framing rather than §04's "padding is
mandatory / Bailey worst case" framing.

### Corroborated at large margin: §4.4 (split vs interleaved), already closed in the corpus

§4.4 was closed by §08 §5.4 in the split layout's favour but with the *granule* left open.
`gen_powp` offered its new SoA-8 split family as a seventh raced candidate, ranked last so it
had to beat the incumbent interleaved family by >3 % to ship. On the node it won by **54 % at
L=25 and 48 % at L=27** — the largest split-versus-interleaved margin the campaign has
measured, on a prime-power CT axis, at the 8-volumes-per-granule figure §08 §1.10 derived from
the cache line. Every winner in §1a is now a split/SoA engine.

### Partly moved, and the residue is the biggest thing still on the board: §4.3 (axis fusion)

`gen_dense_prime` fused the z-pass into the x-pass: first attempt with masked tail stores
**regressed** (142.5 → 145.7), then with full-tail stores into padded stack rows reached
**136.9 (−4.1 %)**. Another single-digit fusion number, consistent with the r3 verdict §4.3
already cites, and another instance of the r3 lesson that *store order* outweighs *pass count*.

But §4.3's **re-opened** case — the L2↔DRAM boundary, where the measured bandwidth gap is 7×
rather than 2.6×, and where Intel's manual, Alappat et al. and the L3-Fusion result all
independently recommend tiling the batch to L2 and running all three axes inside the tile —
**was not attempted by anyone.** Every fusion experiment this round was again across L1↔L2.
That is precisely the regime of L=40, L=50 and L=100, and it is precisely where the panel's
margin collapses to 2.01×, 2.00× and 1.58×. §4.3's "largest untried structural move on the
board" is still untried.

---

## 6. The single highest-value thing the next round should attack

Per geometry, one item each, chosen by measured headroom rather than by taste.

| L | attack | the number that justifies it |
|---|---|---|
| **10, 12, 15** | **The B=1 / small-batch lane-spatial engine.** These are the only cells in the entire round where a library beats the panel. Two records name the identical hole and the identical fix (ice `L6_pfa`'s interleaved-complex z-lane turn) and both say "coordinate rather than build it twice" — so build it **once**, in whichever of the two owns it. | gen_batchlane B=1: 10.50 / 17.54 / 40.88 vs same-window MKL 4.96 / 8.32 / 18.03 = **losing 2.1–2.3×**, while winning 3.7–4.0× at B≥32. And `gen_planner`'s batch-invariant generic engine already beats the specialist at B=1 (6.23 vs 10.50 at L=10) — the shape to copy is in the tree. |
| **20** | **The c-stream.** S+C = 2.3 MB exceeds L2, so `c` streams from L3 on every chain step. Interleave `c` into the site (`re|im|cre|cim`, 256 B) so the x-pass has one stream instead of two. Measure, do not assume — the owner's own words. | L=20 is the only cell whose runner-up is 3.2× behind the leader (13.213 vs gen_layout 41.899): no second opinion exists there at all. |
| **25, 27** | **Extend the `soa` family to batch % 8 ≠ 0.** The winning candidate is simply not offered off-multiple, which is a coverage hole, not a performance one — and round 6 may draw any batch. | B=1 costs 48.7 / 68.1 against batched 32.001 / 44.970 = **1.52× / 1.51×**, entirely because `soa` declines to run. |
| **31** | **Close the dense/Rader crossover — it is the campaign's assigned fight and it is still open.** Then fix the map-tail cast in both files so `build_errors.txt` is empty. | The gap narrowed from **1.86× to 1.43×** in one round (dense 175.6 → 124.281 against Rader 94.170 → 86.913). `gen_planner`'s model puts the crossover at p=17; at p=31 Rader still wins, but not by enough to call it settled, and both entries own §4.5's padding result so neither is leaving performance on the layout. |
| **32** | **Take `gen_pow2` off its single size.** It is the round's cleanest engine — 2.95× over MKL and effectively batch-invariant (58.130 batched, 58.84 at B=1) — and it is scored on exactly one cell while owning 16, 64 and 128 unscored. | Nothing in the suite tests 16/64/128; round 3's any-size-in-class mandate will, and the engine that is already batch-invariant is the cheapest one to generalise. |
| **40, 50, 100** | **§4.3's re-opened construction: tile the batch so a tile fits L2, then run all three axes inside the tile.** This is the round's one genuinely untried structural move, and these are exactly its three geometries. Second item at L=100: `gen_pfa_large` must adopt `gen_race`'s wisdom cache — `gen_race`'s next-list #1 offers a worked patch. | Working sets 15.62 / 15.26 / 30.52 MiB, all past L2; margins **2.01× / 2.00× / 1.58×**, the three weakest in the suite, against 3.2–4.0× where the volume is L2-resident. The bandwidth gap being fused across is **7×**, not the 2.6× every experiment so far has used. And `gen_pfa_large` pays **6.339 s** of plan time at L=100 on every process where `gen_powp` pays 0.006 s. |

**If only one thing is done:** the L2↔DRAM batch-tile-then-fuse experiment at L=40/50/100. It
is the only item on this list that the literature, three independent published sources, and
the shape of the measured margin curve all point at simultaneously, and no one in two rounds
has run it.

---

## 7. What to keep, and why

Applying `docs/CURATION.md`'s four grounds in order. Sources are in `impl_2/`
(`impl` → `impl_2`); every candidate below has a complete `strategies/*.md` record, which is the
prerequisite the document sets.

**Ground 1 — the fastest correct entry for each geometry, one per L, always.** Six entries
cover the eleven cells: `gen_batchlane` (10, 12), `gen_pfa_small` (15, 20), `gen_powp` (25, 27),
`gen_rader` (31), `gen_pow2` (32), `gen_pfa_large` (40, 50, 100). Mandatory, all six kept.

Note on the `gen_batchlane`/`gen_pfa_small` pair: they are close to near-duplicates —
`gen_pfa_small` adopted `gen_batchlane`'s engine structure wholesale and they finish 0.0 % apart
at L=10 and 0.3 % apart at L=15. Ground 1 is unconditional, so both are kept, and
`gen_pfa_small` independently carries the L=20 extension and the written-down in-place safety
rule that `gen_batchlane` does not have.

**Ground 2 — a structurally different runner-up when it is close.** `gen_dense_prime` at L=31,
1.43× behind `gen_rader`. This is verbatim the scenario `CURATION.md` uses as its own example
(Rader winner, dense conjugate-symmetric alternative), it is the campaign's explicitly assigned
crossover fight, it closed from 1.86× to 1.43× in one round, and it independently satisfies
ground 4 by beating every library at L=31 by 5.75×. Kept.

**Ground 4 — anything that beat a library baseline, regardless of rank.** Read literally this
sweeps in every entry at L=31, where all seven supported entries beat all six libraries. That
would defeat curation, so it is applied to entries whose library win *is* their distinguishing
result:

* `gen_layout` — a **dense O(L⁴)-class floor** that beats MKL at L=20 (41.899 vs 57.465), L=25
  (99.809 vs 120.523) and L=27 (136.486 vs 144.494), and every library at L=31 (249.675 vs
  714.517). A dense floor beating a production FFT at four sizes is the single most quotable
  number the round produced, and it is §08's "won by *layout*, not by the kernel" reproduced on
  our own node. Kept.
* `gen_planner` — beats MKL at L=20, 25 and 31 with a *generic* any-L engine, improved 12–36 %
  at all eleven cells, and was adopted wholesale by `gen_race` via `GEN_PLANNER_LIB`. It is
  brief deliverables 1 and 2, and it is the engine the next panel will build on. Kept.
* `gen_race` — beats MKL at L=20, 25 and 31, and is brief deliverable 3. It carries this round's
  §4.6 result (§5), the measured warm-create budget (2–5 ms against 50 ms), and the finding that
  wisdom persistence is *repeatability* infrastructure. Its adoption receipt is the round's
  strongest: `gen_powp`, the −34 %/−32 % winner at L=25/27, picks its candidates through
  `gr_pick`. Kept.

`gen_race` and `gen_planner` post near-identical timings, which normally triggers the
near-duplicate prohibition. They are not code duplicates: one is the enumerator and executor,
the other is the tuner and the per-host cache, they are separate mandated deliverables, and the
identical timings are the *consequence* of the adoption rather than evidence of redundancy.
Both kept, on that stated reasoning.

* `gen_twiddle` — the fourth mandated layer, scored by adoption, and the receipt is concrete and
  cited: `gen_bluestein` took `tw_chirp` plus `tw_fill_ct_int_colmajor`/`tw_audit_...` for every
  per-stage table, forward and conjugated, with ≤0.51 ulp asserted in `create()`
  (`impl/gen_bluestein.c:523–548`); it also answered `gen_planner`'s mid-round k2-major layout
  change with a matching filler. Its own entry improved 28–49 %. Decisive for keeping it: it
  ships `tw_primroot` and `tw_fill_rader_fft` (long-double end-to-end, one rounding), which is
  exactly the machinery §6's L=31 item and round 3's any-prime Rader mandate need, as one call.
  Forward value for the next panel is what the reading list is for. Kept.

**Not kept — `gen_bluestein`.** It is the mandated existence fallback and it did its job: it ran
all eleven cells, passed every gate, and improved at nine of them. But it is last or
second-to-last at ten of eleven cells (3.2–4.9× behind the leader), it satisfies no ground —
not fastest, not a close runner-up, not an instructive failure (it did not fail; it is
structurally the floor), and its only library win is at L=31 where every entry has one. Its one
genuinely interesting result — that Bluestein-as-fallback costs 3.2–4.9×, not the 107–1315×
the corpus warned of — lives in its strategy record, which is tracked regardless, and its
source is preserved in `impl_2/` regardless. Promoting it would add a near-duplicate of the
floor to a reading list, not a lesson.

Eleven of twelve is an unusually wide promotion and it is deliberate: this round every entry
improved, no entry regressed outside spread, four of the twelve are mandated library layers
with cited adoption receipts, and seven are per-class winners or the live crossover
alternative. The set is wide because the round was good, not because the criteria were loose.

---

## 8. Round note for `promote.sh`

**What gen_r2 established.** Twelve entries, zero correctness failures, zero build errors, zero
crashes, zero authorised absences, and zero regressions outside measurement spread. The panel
beats the best library at all eleven scored cells, by 3.2–4.0× where the volume is L2-resident,
8.2× at the prime, and 2.0×/1.6× once the working set leaves L2. Three crowns changed hands
(15, 50, 100), all to entries that improved rather than to incumbents that slipped.
`gen_race` stopped being a test bench and became the tuner the brief asked for, on
`gen_planner`'s engine — and the two together priced LITERATURE §4.6: search is worth 4–16 %
over a good model where the model mis-ranks, ~0 % where it does not, for 2–5 ms of amortised
plan time. §4.5's 4 K-aliasing and alignment hazard was measured at a *prime* edge length for
the first time, at 5–9 %, by two entries independently.

**What gen_r3 should attack.** The batch-tile-to-L2-then-fuse-all-three-axes construction at
L=40/50/100 — the one structural move the literature points at that two rounds have not tried,
in exactly the three geometries where our margin collapses. Then the B=1 lane-spatial engine at
L=10/12/15, the only cells where a library still beats us. Then `gen_pfa_large` onto
`gen_race`'s wisdom cache, so the winner at the three largest sizes stops paying up to 6.3 s of
plan time per process.
