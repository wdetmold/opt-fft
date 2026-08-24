# VERDICT — round gen_r3

Monitor's report. Measured on `a80n0.lqcd.mit`, slurm job 438682, 2026-08-24T11:24:41-04:00,
gcc 11.4.0, governor `schedutil`. Leaderboard written 14:54:14.

## 0. Three corrections to the monitor brief before any number is read

The first two are the same two gen_r2's verdict had to make, and they are repeated because the
brief was not updated; the third is new to this round.

**(a) The geometry set.** The brief asked for headlines at `L = 6, 8, 17, 36`. Those are the
*ice* campaign's four fixed cubes. This is the GENERALIZE campaign, and its frozen acceptance
suite (`cases.txt`) is eleven cells:

```
10:64:1000  12:64:600  15:32:600  20:32:256  25:16:256  27:16:200
31:16:140   32:8:250   40:8:128   50:4:128   100:1:64
```

Section 1 reports all eleven. `docs/LITERATURE.md` is still written for the old four sizes,
which is where the stale list comes from; its §4 open questions are nonetheless live here and
§5 maps them onto this suite. Note that `L = 6` and `L = 36` are now *supported* by
`gen_pfa_small`'s new generic engine — but they are not in the frozen suite, so they are not
scored, and §3.5 treats that gap as this round's largest unverified claim.

**(b) The machine.** The brief describes the scoring node as a Xeon Gold **5218** (Cascade Lake,
downclocked AVX-512, 1 MB L2) and the implementers as developing on Sapphire Rapids.
`environment.txt` says otherwise: the scoring node is a Xeon Gold **6326** — **Ice Lake-SP**,
2.90 GHz, 64 hardware threads, 1.25 MB L2, two 512-bit FMA pipes at full clock. Every one of
the twelve implementer records states its numbers were taken on **a80n0 itself**, via
`tryout.sh` on a leased core of the same node. `PANEL_BRIEF.md` confirms the design: Ice Lake
*is* the scored tier, and Cascade Lake and Sapphire Rapids are advisory reruns that begin
after round 4 (no `XARCH.md` exists yet, correctly).

The consequence for §4 inverts the brief's instruction: **there is no machine difference to
attribute anything to.** Claim-versus-measured divergence in this round is a *node-contention*
story, and §4 treats it that way. The "MKL spans 2.9× between those machines" calibration does
not apply: MKL moved ≤ 0.4 % between gen_r2 and gen_r3 at ten of the eleven cells, and 2.6 %
at the eleventh.

**(c) This pass was measurably less quiet than gen_r2's.** gen_r2's verdict could say library
baselines moved < 1 % at every cell. This round they moved up to **+4.6 %** (`fftw3_estimate`
L=50), **+3.2 %** (`fftw3_patient` L=10) and **−8.0 %** (`fftw3_measure` L=100), and three
scored runs carry an in-run `sd` above 4 % with median far above min (`gen_batchlane` L=15 run 1
at sd 6.71 %, L=20 run 3 at 4.78 %). Six implementer records independently report the node
spent long stretches this session in a slow all-core-turbo state. **The practical rule for
reading §2: a panel movement under ±3 % at a single cell is not evidence of anything.** Two
movements clear that bar and §2 names them.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

### 1a. Batched (ten of the eleven cells)

| L | B | fastest correct panel entry | µs/xform | best library | µs/xform | panel margin |
|---|---|---|---|---|---|---|
| 10 | 64 | **gen_pfa_small** (gen_batchlane 1.157, tie) | 1.156 | mkl_dfti | 4.553 | **3.94×** |
| 12 | 64 | **gen_batchlane** | 1.915 | mkl_dfti | 7.747 | **4.05×** |
| 15 | 32 | **gen_pfa_small** | 4.464 | mkl_dfti | 16.464 | **3.69×** |
| 20 | 32 | **gen_batchlane** (gen_pfa_small 13.257) | 13.011 | fftw3_patient | 44.869 | **3.45×** |
| 25 | 16 | **gen_powp** | 32.208 | fftw3_patient | 109.921 | **3.41×** |
| 27 | 16 | **gen_powp** | 44.811 | mkl_dfti | 144.315 | **3.22×** |
| 31 | 16 | **gen_rader** | 85.088 | ducc0_c2c | 715.949 | **8.41×** |
| 32 | 8 | **gen_pow2** | 57.322 | mkl_dfti | 176.100 | **3.07×** |
| 40 | 8 | **gen_pfa_large** | 202.382 | mkl2026_dfti | 404.881 | **2.00×** |
| 50 | 4 | **gen_powp** (gen_pfa_large 473.988, tie) | 473.678 | mkl_dfti | 947.159 | **2.00×** |

Geomean margin **3.20×**, worst **2.00×**. gen_r2's were 3.18× and 2.00×. **At the top of the
board this round changed essentially nothing** — and that is the round's central fact, not a
disappointment; see §1c.

### 1b. Non-batched

The frozen suite contains **exactly one** non-batched cell, so this is the only scored B=1 row:

| L | B | fastest correct panel entry | µs/xform | best library | µs/xform | panel margin |
|---|---|---|---|---|---|---|
| 100 | 1 | **gen_powp** | 5021.039 | mkl_dfti | 7784.041 | **1.55×** |

`gen_pfa_large` is 1.4 % behind at 5089.048. The crown changed hands back to `gen_powp`, and it
changed hands because `gen_pfa_large` got *slower* — the only cell in the suite where that is
true of the leader (§2).

For L=10…50 there is **no scored non-batched number** and the monitor will not manufacture one.
What exists is implementer-reported B=1 on the same node, and it is much better than gen_r2's:

| L | best panel B=1 (reported) | gen_r2 equivalent | note |
|---|---|---|---|
| 10 | gen_race **5.42** · gen_batchlane 10.52 | 6.23 · 10.50 | vs same-window MKL ≈ 4.96: the generic layer is now **1.09×** behind, was 1.26× |
| 12 | gen_twiddle 17.50 · gen_batchlane 17.42 | 10.49 · 17.54 | |
| 15 | gen_batchlane 44.2 | 40.88 | worse; tracks the L=15 regression in §2 |
| 25 | gen_powp **42.93** · gen_race 69.30 | 48.7 | −11.9 %, from the in-place race arena, not a new kernel |
| 27 | gen_powp **59.71** | 68.1 | −12.3 % |
| 31 | gen_rader **85.34** · gen_dense_prime 128.8 | 99.7 · 131.2 | Rader's B=1 now reads *below* its own B=16 |
| 32 | gen_pow2 58.0–58.2 | 58.84 | batch-invariant, as before |
| 40 | gen_pfa_large **215.1** | 223.2 | −3.6 % |
| 50 | gen_powp **484.5** · gen_pfa_large 530.7 | 615 · 554.9 | −21.2 % |

**The B=1 hole gen_r2 named as its #1 item at L=10/12/15 is half closed, and not by the fix it
prescribed.** Nobody built the lane-spatial engine. Instead the *generic* engines got 25–37 %
faster, and at L=10 the panel's best B=1 is now 1.09× behind MKL instead of 1.26×. But the
specialists' remainder-lane gap is untouched — `gen_batchlane` still reads 10.52 at L=10 against
1.157 batched, and it says so plainly. The fix was routed around, not made.

### 1c. What actually moved: the generic engine

The leaders plateaued. The middle of the board did not.

| entry | gen_r2 → gen_r3 range across all eleven cells | mechanism |
|---|---|---|
| **gen_race** | **−25 % to −63 %** | gen_planner's new intrinsic executor + engine-generation wisdom salt + a second tile race |
| **gen_planner** | **−24 % to −62 %** | explicit AVX-512 complex layer; the r2 executor was compiling half-scalar |
| **gen_dense_prime** | **−35 % to −46 %** at L=10…20 | register-tiled fold, z-pass fused into the x-contraction |
| **gen_twiddle** | **−17 % to −36 %** | owned in-place volume-major chain with the map fused into the axis-2 scatter |
| **gen_bluestein** | **−9 % to −22 %** | owned chain + size-gated map placement + vectorized seam groups |
| **gen_layout** | **−1 % to −18 %** | cross-plane axis-1 lane packing, axis-2 through a 4-plane collision-picked window |

Measured against the **best library at each cell**, the generic any-L engine (`gen_race`) went
from a geomean of **0.87×** in gen_r2 — i.e. 15 % *slower* than the best library — to **1.43×**,
winning **10 of 11 cells** (it loses only L=10, 4.718 vs MKL 4.553). `gen_planner` is the same
picture at 10 of 11. The distance from the generic engine to the hand-tuned class winner
collapsed everywhere:

| L | gen_race behind the winner, r2 → r3 |
|---|---|
| 20 | 3.47× → **2.14×** |
| 31 | 6.11× → **2.30×** |
| 32 | 4.03× → **2.22×** |
| 40 | 2.49× → **1.33×** |
| 50 | 2.48× → **1.63×** |
| 100 | 1.92× → **1.23×** |

That is the result that matters for round 6, which scores the *assembled library* on three
sizes nobody has seen, and it is the strongest thing this round produced.

---

## 2. What changed since gen_r2, per geometry

| L | r2 leader | r3 leader | leader delta | crown |
|---|---|---|---|---|
| 10 | gen_batchlane 1.162 | **gen_pfa_small 1.156** | −0.5 % | nominally changed hands (0.09 % apart — a tie) |
| 12 | gen_batchlane 1.933 | gen_batchlane 1.915 | −0.9 % | held |
| 15 | gen_pfa_small 4.466 | gen_pfa_small 4.464 | −0.04 % | held |
| 20 | gen_pfa_small 13.213 | **gen_batchlane 13.011** | −1.5 % | **changed hands** (batchlane's new size) |
| 25 | gen_powp 32.001 | gen_powp 32.208 | +0.6 % | held |
| 27 | gen_powp 44.970 | gen_powp 44.811 | −0.4 % | held |
| 31 | gen_rader 86.913 | gen_rader 85.088 | −2.1 % | held |
| 32 | gen_pow2 58.130 | gen_pow2 57.322 | −1.4 % | held |
| 40 | gen_pfa_large 201.049 | gen_pfa_large 202.382 | +0.7 % | held |
| 50 | gen_pfa_large 473.024 | **gen_powp 473.678** | +0.1 % | **changed hands** (0.07 % apart — a tie) |
| 100 | gen_pfa_large 4947.503 | **gen_powp 5021.039** | **+1.5 %** | **changed hands, on a regression** |

### Did anything regress?

**Two movements clear the ±3 % noise bar of §0(c), and one of them is a real regression.**

**1. `gen_batchlane` at L=15: 4.478 → 4.771, +6.5 %. This is a genuine regression and the
implementer's own explanation does not survive.** The three scored runs read 4.771 / 4.771 /
4.883 — two runs identical to three decimals — against gen_r2's 4.495 / 4.478 / 4.498. Its
record claims **4.456** and attributes the higher numbers it saw to the slow all-core-turbo
state. That defence fails on the direct control: **`gen_pfa_small`, which runs the same engine
structure at the same size, measured 4.464 / 4.465 / 4.476 in the same scoring pass** — i.e.
its gen_r2 number, unchanged. The node was fine at L=15; `gen_batchlane`'s r3 change was not.
The most likely mechanism is the round's headline change itself: the register-explicit pencil
form (R2L/R3L/R5ST macros, named `xr<k>`/`xi<k>` registers) puts 2L live vector names in flight,
and at L=15 that is 30 zmm against 32 — the register-pressure cliff its own record says it
declined to cross at L=20. **It won L=12 and L=20 with that change and lost L=15 to it.** The
cell is still the panel's, because `gen_pfa_small` holds it; but `gen_batchlane` shipped a
regression at a scored size, its own claimed number was never reproduced, and it should default
back to the r2 memory-round-trip form at L=15 unless it can beat 4.464 with the register form.

**2. `gen_pfa_large` at L=100: 4947.503 → 5089.048, +2.9 %; and the cell leader went
4947.503 → 5021.039, +1.5 %.** At 2.9 % against a 1.9–2.9 % spread this sits exactly on the
noise floor and I will not call it a code regression — its record reports 5374.6 in windows it
describes as 4–6 % slow, and the scoring pass read it 5.3 % faster than that. But it is the
**first time in the campaign that a cell's best number got worse**, and it cost `gen_pfa_large`
the L=100 crown to a rival that stood still (`gen_powp`, −0.2 %). §6 acts on it.

Everything else nominally in the wrong direction is inside spread and inside the library noise
band of §0(c): `gen_powp` L=25 +0.6 % (spread 6.0 %), `gen_pfa_large` L=40 +0.7 % (spread
0.8 %) and L=50 +0.2 %, `gen_pfa_small` L=20 +0.3 %. Nine of twelve panel entries improved at
every cell they support.

### The compliance defect gen_r2 named is fixed, with a receipt

gen_r2 §3.5's sharpest finding was that `gen_pfa_large` won the three largest sizes while paying
0.488 / 0.662 / **6.339 s** of plan time on *every* process, because it persisted nothing.
`timing.log`'s per-run setup column settles it:

| cell | gen_r2 runs 1–4 | gen_r3 runs 1–4 |
|---|---|---|
| L=40 | 0.366 / 0.413 / 0.488 / 0.354 | **0.578 / 0.005 / 0.003 / 0.002** |
| L=50 | 0.662 / 0.849 / 0.894 / 0.671 | **0.899 / 0.003 / 0.003 / 0.003** |
| L=100 | 6.339 / 6.436 / 6.484 / 6.093 | **7.605 / 0.003 / 0.004 / 0.003** |

Cold race 7.6 s against the brief's 60 s; warm create 3 ms against 50 ms. `results/wisdom_a80n0.json`
carries the three new keys (`gen_pfa_large/chain3/L{40,50,100}/...`). `gen_planner` closed the
same gap the same way (`gen_planner/tree/L*`, warm 4 ms), so all four wisdom-bearing entries are
now inside the warm budget. That was gen_r2's third-ranked next-round item and it is done.

---

## 3. Adversarial pass: correctness, builds, crashes, absences

### 3.1 Correctness — nothing failed, and no panel entry is riding a tolerance

`check.log` holds **614 lines, all `PASS`**. Zero FAIL, MISMATCH or NOT REPEATABLE anywhere.

* Gate 1 (single call vs numpy, tol 1e-12): leaderboard `1s` column spans **8e-16 … 4e-15** —
  250× inside at worst, at L=100.
* Gate 2 (two-step fused chain, tol 3e-14 = the 1.5e-14/step precision contract): worst value in
  the whole log is **5.164e-15**, and it belongs to **`baseline_matrix` at L=100**, the harness's
  library-free O(L⁴) floor, not to a panel entry. The worst *panel* value is `gen_bluestein` at
  L=100, **3.933e-15** — **7.6× inside** tolerance, statistically identical to gen_r2's 3.7e-15.
  No entry tightened this gate.
* Gate 3 (chain end, 300× honest-anchor allowance, 1e-10 floor): worst overall is again
  `baseline_matrix` (L=10, 3.417e-13 against a 9.337e-14 anchor = 3.66×). Worst panel value is
  `gen_bluestein` L=10 at 2.1e-13, **476× inside** the floor and 2.25× its anchor.

**No fast-and-wrong answer survived, because none was submitted.** Six records additionally
report bit-identical outputs across independent node processes, which is the property the
wisdom cache makes load-bearing (§5).

### 3.2 Builds — no errors. Three warnings, and gen_r2's instruction was ignored

`build_errors.txt` contains **no errors**. It contains three warnings and all three get named,
because two of them are gen_r2's warnings, unfixed.

* `impl/gen_dense_prime.c:1402` and `impl/gen_rader.c:917` (twice, in `fft3d_chain` and
  `fft3d_create`) — `-Waggressive-loop-optimizations`, "iteration 1152921504606846976 invokes
  undefined behavior". Identical idiom to gen_r2's: `size_t i`, `size_t npts`, tail loop
  `for (; i < npts; ++i)` indexing `zp[2*i]`; GCC cannot bound `npts` and reports pointer-offset
  UB at iteration 2⁶⁰. **Not a live defect** — the preceding `_mm512` loop consumes `i` in
  strides of 8, so the tail runs at most 7 iterations, and both entries pass all three gates at
  L=31 with wide margin. But gen_r2 wrote: *"the fix is one cast and it costs nothing, so there
  is no reason for `build_errors.txt` to be non-empty next round."* Both files were heavily
  edited this round — the line numbers moved 1116 → 1402 and 696 → 917 — and **neither
  implementer added the cast.** That is a direct non-response to a monitor instruction, in the
  L=31 winner and its runner-up, and it is recorded as such.
* `impl/gen_race.c:887` — **new this round.** `-Wformat-truncation`: `snprintf(gk, sizeof gk,
  "gen_race/chaingate3/L%d/%s", L, p->picked)` where `gk` is `GR_KEY_MAX` = 160 and `picked` is
  `GR_NAME_MAX + 8` = 136, so GCC computes 24–161 bytes into 160. I checked the arithmetic:
  with `L ≤ 128` (three digits) and a tree name ≤ 127 chars the worst real key is 156 bytes, so
  **this is not currently reachable.** It is worth naming anyway, because of what it would do if
  it ever were: a truncated key makes two *different* plans collide on one wisdom entry, so the
  race would replay the wrong winner while the wisdom file looked healthy. That is precisely the
  failure class `gen_race` spent this entire round discovering and fixing from the other
  direction (§5). Widen `gk` or bound `L` in the format; it is a one-line change in the file
  that owns the hazard.

### 3.3 Crashes and hangs — one, and it is the reference floor, not a panel entry

`failures.txt` has exactly three lines, all one event:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is a `timeout` kill. `baseline_matrix` is the O(L⁴) floor; at L=100 it cannot finish
the m=64 chain in the window. This reproduces gen_r1 and gen_r2 exactly and is why the L=100
table has no `baseline_matrix` row. **No panel entry crashed or hung.**

### 3.4 Absences — every panel decline is authorised; one baseline is missing

`timing.err` records 220 lines = **55 distinct** `does not support L=` declines, cross-checked
line by line against `PANEL_BRIEF.md`'s acceptance table. Every one is authorised. The set is
gen_r2's minus exactly one entry: **`gen_batchlane: does not support L=20` is gone** — it
adopted the size and won it. No entry lost coverage. All twelve agents exited 0
(`agents/exits.txt`).

**But one baseline is missing from the leaderboard, and it is the one this round was supposed to
add.** Commit 6178871 introduced `fftw3_guru` — "guru split-array API with fused split chain —
the strongest FFTW configuration, in gen (live from r3)". It is absent from
`leaderboard.txt`, from `leaderboard.md`, and from the `backends:` legend. The evidence for what
happened is in the timestamps:

* `leaderboard.txt` and `sweep.out` were written at **14:54:14**, and `sweep.out` ends
  `== round gen_r3 complete ==`.
* `t_fftw3_guru_L10_B64_r1.json` was written at **14:55:19**, `_r2.json` at **14:55:21**, and
  `timing.log`'s last line — line 618 of 618 — is a `fftw3_guru` L=10 run at 14:55:32.
* Only two of the three runs exist, and only for **L=10**. There is no guru row for any of the
  other ten cells, and no guru line in `check.log`.

So the guru baseline was started *after* the leaderboard had already been computed and was cut
off two runs into the first cell. **No headline in §1 changes** — at L=10 guru measured 6.515–
6.555 µs, which is slower than `fftw3_measure`'s 5.188 and far behind MKL's 4.553, so it would
not have been the best library there even had it finished. But two things follow. First, the
claim that guru is "the strongest FFTW configuration" is **unsupported at the only cell where a
number exists** — it lost to plain `plan_many_dft` + FFTW_MEASURE by 26 %. Second, until it runs
all eleven cells, no statement of the form "we beat the strongest FFTW" is scored. Re-run it
before gen_r4 and put it in the legend.

### 3.5 The round-3 generality mandate is delivered on paper and unscored in fact

`PANEL_BRIEF.md` adds, from round 3, "ANY size the driver asks for within your class", and
round 6 zeroes the *whole assembled library* if one surprise size fails to plan, races past
60 s, or misses a gate. Every class owner shipped generic machinery this round and says so in
the leaderboard legend: `gen_pfa_small` a runtime-table coprime P×Q engine for
6/14/18/21/24/28/35/36/45/56/63; `gen_rader` a generic folded half-system engine for primes
3…127; `gen_dense_prime` any prime p ≤ 31; `gen_pow2` any 2^k in 2…128; `gen_planner` any
2 ≤ L ≤ 128.

**None of it is scored.** The suite is eleven frozen cells and the harness ran exactly those.
The off-suite evidence is entirely implementer-attested — `gen_rader`'s p = 3…29 and 37 table,
`gen_powp`'s L = 49/81/121/125, `gen_dense_prime`'s primes 13…29, `gen_pfa_small`'s L=14 at
2.0× MKL, `gen_pfa_large`'s L=80, `gen_pow2`'s 2^k regression at L = 2…128 — all of it taken on
leased cores by the authors of the code under test. It is well documented and I have no reason
to doubt it; it is also exactly the material that round 6 will convert into a score, and it has
never been through the harness's three gates on the scoring pass. **This is the round's largest
unverified claim and it should not be allowed to reach round 6 untested.** §6 makes it an item.

### 3.6 Two housekeeping findings worth a line each

* **An implementer edited the scoring artifact, disclosed.** `gen_planner` states it stripped all
  `gen_planner/` keys from `results/wisdom_a80n0.json` at round end so the scoring run would cold-race
  in a quiet window. It is disclosed, it follows `gen_powp`'s precedent, and cold-racing is the
  more honest of the two options — but it means `gen_planner`'s scored `setup` column (0.002–
  0.083 s) is a cold number and its 4 ms warm-create claim is **unscored**. The file now holds 51
  entries and all eleven planner trees were repopulated by the scoring pass itself.
* **A stale orphaned source sits in the round's provenance directory.**
  `impl_3/gen_pfa_large.c.RESCUED-see-pfa_small-r3-notes` (65 424 bytes, 10:04) is a
  `git stash` casualty from the mid-session `impl` → `impl_3` rotation, parked verbatim by
  `gen_pfa_small` and never reconciled by `gen_pfa_large`, whose own record confirms six edits
  vanished. It does not build (the Makefile globs `*.c`) and cannot affect any number here. It
  should be diffed and deleted or merged before this round's sources are promoted, because
  `impl_N/` is the campaign's provenance and an unresolved shadow copy of the L=40/50/100 winner
  is the worst possible thing to leave in it. The standing rule the incident produced — **never
  `git stash` in this repo; A/B with a file copy** — is now in `gen_pfa_small`'s record and
  should be in the brief.

---

## 4. Claimed versus measured

The brief's premise does not hold — see §0(b). Implementers developed on **a80n0 itself**, so
there is no Sapphire-Rapids-to-Cascade-Lake gap to attribute anything to. Agreement is
correspondingly tight, and **the divergences point overwhelmingly one way: the scored run was
faster than the number the implementer reported**, which is the signature of the scoring pass
owning the node and the reverse of what a downclock story would produce.

| entry | its own claim | scored | delta |
|---|---|---|---|
| gen_pfa_small | 1.157–1.194 / 1.970–1.975 / 4.469–4.480 / 13.24–13.53 | 1.156 / 1.970 / 4.464 / 13.257 | in range or a hair better |
| gen_pow2 | best 57.41 | 57.322 | −0.15 % |
| gen_rader | 85.50 / 85.57 / 85.62 | 85.088 | −0.5 % |
| gen_powp | 32.40 / 45.03 / 482.8–486.6 / 5070–5123 | 32.208 / 44.811 / 473.678 / 5021.039 | −0.6 / −0.5 / −1.9 / −1.0 % |
| gen_layout | 4.87 / 7.98 / 19.07 / 42.02 / 94.54 | 4.904 / 8.009 / 18.621 / 40.346 / 92.999 | +0.7 … −4.0 % |
| gen_planner | all eleven | all eleven | −6.5 % … +2.5 % |
| gen_race | all eleven | all eleven | −4.1 % … −0.3 % (all in its favour) |
| gen_twiddle | all eleven | all eleven | −4.1 % … +0.8 % |
| gen_bluestein | all eleven | all eleven | −3.0 % … +4.4 % |
| gen_dense_prime | 5.838 / 8.205 / 16.177 / 43.691 / ~123–127 | 5.414 / 8.368 / 16.328 / 43.509 / 123.828 | −7.3 / +2.0 / +0.9 / −0.4 / in range |
| gen_pfa_large | 211.3 / 483.2 / 5374.6 | 202.382 / 473.988 / 5089.048 | −4.2 / −1.9 / −5.3 % |
| **gen_batchlane** | 1.156 / 1.919 / **4.456** / 12.99–13.17 | 1.157 / 1.915 / **4.771** / 13.011 | 0.0 / −0.2 / **+7.1 %** / in range |

Divergences beyond 3 %, in the implementer's favour — the scored run beat the claim:
`gen_dense_prime` L=10 (−7.3 %), `gen_planner` L=100 (6655.8 → 6219.999, −6.5 %),
`gen_pfa_large` L=100 (−5.3 %) and L=40 (−4.2 %), `gen_race` L=40 (−4.1 %), `gen_twiddle` L=20
(−4.1 %), `gen_layout` L=20 (−4.0 %). Every one is attributable to node contention during
development, and every one of those records diagnoses it before the scoring pass: `gen_pfa_large`
"this round's windows ran 4–6 % slow by the MKL yardstick"; `gen_powp` "the windows this session
ran 3–6 % hot"; `gen_pow2` "busy windows this round read 62–73 µs with median ≫ min";
`gen_batchlane` "the leased cores spent long stretches in a slow all-core-turbo state". They
were right, and §0(c) shows the effect had not entirely left by scoring time.

**Two divergences are not window noise and must not be filed as such.**

* **`gen_batchlane` L=15, claimed 4.456, scored 4.771 (+7.1 %).** Refuted in §2 by the
  same-pass control: `gen_pfa_small` measured 4.464 at the same cell in the same pass. This is
  a code regression, not a window.
* **`gen_bluestein` L=50, claimed 1851, scored 1931.920 (+4.4 %)** — the round's only
  *adverse* divergence besides the above, and the sign matters because every other entry
  scored faster than claimed. Its L=50 number is the one where its new 15 MiB map-placement
  gate flips regime (state = 15.26 MiB, right on the threshold), and its own same-window race
  data there is the noisiest it reports (1851 / 2118 / 1877 separate). The gate is set at a
  cell boundary; that is worth one measurement to move off.

**One claim about a mechanism failed to reproduce in the scoring pass, and the wisdom file
records it.** `gen_race` reports its second-stage tile race found "**t64 at L=50 (+2.3 %)**".
The scoring run's own race wrote `gen_race/tile3/L50/B4 → {winner: t32, tie: 1, margin:
−0.00995}` — t32 won, as a **tie**, by 1.0 %. Its L=31 claim did reproduce
(`t16`, margin +4.1 % against a claimed +3.9 %). No number in §1 depends on it and the tie
doctrine correctly kept the engine default, but a +2.3 % knob that inverts between the
development window and the scoring pass is a knob that was never real.

---

## 5. Which `docs/LITERATURE.md` §4 open question this round moved

### Moved decisively, and reversed a published expectation: §4.2, "L = 17: dense-symmetric, Rader, or Winograd?"

§4.2 lays out four positions and asks: *"(a) Which of dense-symmetric and Rader wins on this
hardware, batch-vectorised?"* — with §02 §7 saying Rader "is not the lever" and §03/§06 saying
benchmark the direct conjugate-symmetric form because McFarlin found direct computation
preferable to n ≈ 20 and FFTW keeps a Θ(n²) plan "useful for n ≲ 100". gen_r2 could only report
that the specialists' gap narrowed from 1.86× to 1.43×.

**This round produced two contradictory answers at the same prime, from two engines, and the
contradiction is the result.**

*In the generic engine, dense won and Rader was displaced.* `gen_planner` rewrote its dense path
as a register-tiled fold (4 output rows × 2 zmm of columns = 16 accumulators in registers), and
its race flipped: `d31` over `rad31(c3(c5(d2)))`, 204.9 vs 533. This is not an implementer's
claim — it is written into the scoring pass's own wisdom file:

```
gen_race/chain3/L31/B16  ->  {winner: 'd31', margin: 0.7923, us: 216.558}
gen_race/tile3/L31/B16   ->  {winner: 't16', margin: 0.04125, us: 200.893}
gen_planner/tree/L31     ->  {winner: 'd31'}
```

A **79 % margin** for dense over Rader, raced live on the scoring node during the scored run.
`gen_planner` recalibrated its fold constant 5.5n² → 3.0n² and reports the crossover has moved
*past* p = 31.

*In the specialist engines, Rader still won, and by more than last round.* `gen_rader` 85.088
against `gen_dense_prime` 123.828 — **1.46×**, slightly wider than gen_r2's 1.43×, because Rader
improved 2.1 % and dense spent its round on L=10…20 (−35 to −46 %) instead.

**The reconciliation is the finding, and it is exactly what §4.2's own text predicted without
being able to test it.** The dense-vs-Rader crossover is not a property of the *arithmetic* —
Rader's 388 instructions against dense's 336 do not decide it — it is a property of **how well
each kernel is written**. Improve the dense matrix kernel (register tiling, 16 accumulators,
one broadcast per FMA pair) and dense wins by 79 %; leave both hand-tuned to the same standard
and Rader wins by 46 %. §02 §7's instruction — "implement both and measure" — is now the only
defensible position, and the corpus's attempts to rank them a priori are refuted from both
sides in a single round. `gen_race`'s own words are the operational form: *"their own r2 record
predicted the d31/rad31 crossover was kernel-dependent."* It is.

### Moved, with a mechanism the corpus does not contain: §4.6, "model versus search"

gen_r2 priced §4.6 (search is worth 4–16 % where the model mis-ranks, ~0 % where it is right).
gen_r3 added the failure mode, and it is new. `gen_race` salted its wisdom tags
(`chain` → `chain3`, `chaingate` → `chaingate3`, new `tile3`) after finding that gen_planner's
new engine **reordered the candidate trees while the candidate names stayed identical**, so
`gr_sig` alone could not invalidate the r2 verdicts. The receipt is the §4.2 flip above: the
node's r2 wisdom pinned L=31 to `rad31(c3(c5(d2)))`; the fresh race takes `d31` at a 79 %
margin. **Replaying stale wisdom would have silently cost ~2× at a scored size while the wisdom
file looked perfectly healthy.** The rule now in the header — *if your engine generation changes
and your candidate names don't, salt your key tag* — is a cache-invalidation result, not an FFT
result, and it is nowhere in §06's account of FFTW's planner or SPIRAL's search. It belongs in
§4.6 alongside gen_r2's finding that persistence is *repeatability* infrastructure.

Also in §4.6's ledger: `gen_planner`'s diagnosis that its r2 executor compiled to **277 `vmovsd`
/ 68 `vmulsd` / 63 `vsubsd` — half scalar** — plus 240 shuffle-class ops against ~130 packed
FMAs. gcc-11 `-O3 -march=native` cannot vectorise interleaved-complex butterflies. §4.6 frames
the search question as *which formula*; this round says the prior question is *whether the
compiler produced vectors at all*, and the answer was worth 24–62 %. Three entries now report
that an `objdump` audit paid more than any A/B they ran (`gen_planner` 24–62 %, `gen_batchlane`
its dead-store and out-lining finding, `gen_powp` a clean audit that correctly warranted no
surgery). **Audit the assembly before believing any timing** was §07 §7.8's cheap check; it is
now the panel's highest-yield habit.

### Partly moved: §4.3 (axis fusion) — and the re-opened case is *still* untried

§4.3's re-opened residue, which gen_r2 called "the largest untried structural move on the
board", is: **tile the batch so a tile fits L2, then run all three axes inside the tile**,
across the L2↔DRAM boundary where the measured bandwidth gap is 7× rather than the 2.6× every
panel experiment has used. I searched all twelve r3 records for it. **Nobody attempted it, for
the third round running.** It is the round's clearest non-response, and it is still the item at
the three cells where our margin is worst (2.00× / 2.00× / 1.55×).

What the round *did* produce at that boundary is three independent, converging negative results
about the adjacent question — where to put the map:

* `gen_bluestein` gated map placement by state size and measured the crossover: at L=100
  (30.5 MiB) a **separate** sequential 3-stream map sweep beat the fused axis-0 scatter 4/4 —
  15.98/16.04 ms vs 18.86/19.25 — and at L=50 (15.26 MiB) 1851 vs 2460; at L=25 (7.6 MiB) fused
  wins; at L=31 (14.55 MiB) a wash. Gate at 15 MiB splits the suite exactly.
* `gen_pfa_large` killed its `ipm` fused-map family at L=100 (+11–12 %) and L=50 (+15 %), and
  states why the traffic model was wrong: "p1 at 100 is **NOT** purely DRAM-bound — its port-5 /
  issue budget is where the ladder lands."
* `gen_powp` killed its `ipm` independently with an op count: the map is ~21 FMA-port ops per
  loaded vector against phase 1's own ~48k per plane, so it **doubles phase-1 compute**, and
  phase 1 at L=100 is only about half miss-bound. At L=81 the same family *wins* (−7 %).

**Together these say the L2↔DRAM regime at our sizes is not bandwidth-saturated, it is
issue-limited — so the "hide it under the misses" argument that motivates fusion there does not
close.** That is a genuine, thrice-confirmed constraint on §4.3's re-opened construction, and it
should be read *before* the batch-tile experiment rather than instead of it: tiling to L2 changes
the traffic, but if port 5 is the binding resource at L=100 the tile will not buy the 7× the
bandwidth gap advertises. This is the first CPU-side number the corpus has on that point.

### Corroborated again: §4.4 (split vs interleaved) and the "adopt on faith" prohibition

Every winner in §1a remains a split/SoA engine. More usefully, the round produced the campaign's
cleanest statement of a rule the corpus does not have: **the div-vs-reciprocal-ladder and
map-fusion choices are properties of the surrounding codelet's port pressure, not of the
technique**, and they were measured in both directions in the same round. `gen_pfa_small` took
`gen_batchlane`'s rcp14+2NR ladder (−8.1/−8.8/−4.7 % on their engine) and lost 2.4–4.3 % on its
own; `gen_powp` took `gen_batchlane`'s sched-pressure attribute (−5.7…−10 % on theirs) and lost
**32 % at L=25 and 22 % at L=27**; `gen_rader` fused the map into its generic y-pass and lost
2.3–9 %, the third engine on the panel where eager map fusion loses. `gen_pfa_small`'s wording is
the transferable form: *"A/B it in place, never adopt on faith."* Five measured non-transfers in
one round is a stronger result than any of the individual wins.

---

## 6. The single highest-value thing the next round should attack

Per geometry, one item each, chosen by measured headroom.

| L | attack | the number that justifies it |
|---|---|---|
| **10, 12** | **Finish the B=1 job the generic layer started.** `gen_race`/`gen_planner` cut the panel's best B=1 at L=10 from 1.26× behind MKL to **1.09×** without anyone building the prescribed engine. Close the last 9 % — and give the specialists a remainder-lane path, since they are the ones that win batched. | gen_batchlane B=1 10.52 against its own 1.157 batched = **9.1×** batch sensitivity, unchanged in two rounds. L=10 is now the *only* cell where any library beats any panel entry (MKL 4.553 vs gen_race 4.718). |
| **15** | **Revert `gen_batchlane`'s register-explicit pencil at L=15, or beat 4.464 with it.** This is the round's only real regression and the only cell where a panel entry got worse than its own previous round outside spread. | 4.478 → **4.771 (+6.5 %)**, three runs, against `gen_pfa_small`'s 4.464 in the same pass on the same engine structure. 2L = 30 live zmm against 32 is the mechanism its own record identifies at L=20. |
| **20** | **Get a second opinion at this cell.** `gen_batchlane` and `gen_pfa_small` finish 1.9 % apart on the same engine, and the nearest structurally different entry is `gen_race` at 2.14× behind. Also: the c-stream idea is now dead and should stay dead — it was measured. | gen_batchlane's L=20 c-interleave (site = re8\|im8\|cre8\|cim8, 256 B) measured **+40 %** (18.39 vs 13.17). gen_pfa_small's consumption-order c layout measured **+4–10 %**. Two independent refutations of gen_r2's L=20 prescription; do not spend a third round on it. |
| **25, 27** | **Verify the off-suite prime powers on the scoring pass, not on a leased core.** `gen_powp` closed gen_r2's `soa`-at-B≠8 hole (B=1 now 42.93/59.71, was 48.7/68.1) and reports L=49/81/121/125. None of it has been through the harness. | Round 6 zeroes the whole library on one failing size, and gen_powp's off-suite numbers include one loss (L=81 execute 6060 vs MKL 3770 = **1.6× behind**) that has never been gated. |
| **31** | **Bring the register-tiled fold into `gen_dense_prime` and re-run the crossover.** The campaign's assigned fight now has two answers pointing opposite ways in the same round (§5): dense wins by 79 % inside `gen_planner`, Rader wins by 46 % between the specialists. The difference is kernel quality, so make the kernels equal and ask again. Then add the one-line cast so `build_errors.txt` is finally empty. | `gen_planner`'s `d31` at 203.733 against `gen_dense_prime`'s 123.828 says the specialist is still 1.65× *ahead* of the generic dense path — so the fold constant that flipped planner's race (5.5n² → 3.0n²) has more to give in the entry that owns the algorithm. |
| **32** | **Score `gen_pow2` on more than one cell.** It is the cleanest engine on the panel — 3.07× over MKL, batch-invariant (57.322 batched, 58.0 at B=1), three independent refutations of manual cross-codelet scheduling — and it owns 16, 64 and 128 with nothing testing them. | Its own 2^k regression at L = 2/4/8/16/64/128 is single-call only (rel L2 0 … 4.1e-16); no chain, no gate 2, no timing. Round 6 may draw 64 or 128. |
| **40, 50, 100** | **§4.3's re-opened construction, third time asked: tile the batch to L2, then run all three axes inside the tile — but instrument port 5 first.** This round produced the reason it might not work (three independent findings that L=100's phase 1 is issue-limited, not miss-limited) which makes the experiment *more* informative, not less: it is now a test of a specific hypothesis rather than a hopeful port. Second item at L=100: find the 2.9 % `gen_pfa_large` gave back. | Margins **2.00 / 2.00 / 1.55×**, still the three weakest in the suite against 3.1–4.1× where the volume is L2-resident. Working sets 15.62 / 15.26 / 30.52 MiB. And L=100 is the first cell in the campaign whose best number went *backwards*: 4947.503 → 5021.039. |
| **all** | **Run `fftw3_guru` across the eleven cells, and put the eleven generic off-suite sizes through the harness.** | Guru has **one** cell of data out of eleven, and at that cell it lost to plain `plan_many_dft` by 26 % — so the baseline set the round advertised does not exist yet. And every generality claim in §3.5 is author-attested on a shared node with round 6 three rounds away. |

**If only one thing is done:** the L2↔DRAM batch-tile-then-fuse experiment at L=40/50/100,
with a port-5/issue-rate counter alongside the miss counters. Three rounds have declined it, the
literature and three independent panel findings now point at the *same* cells from opposite
directions, and it is the only item on this list that can move the suite's worst margins.

---

## 7. What to keep, and why

Applying `docs/CURATION.md`'s four grounds in order. Sources are in `impl_3/` (`impl` →
`impl_3`); every candidate below has a complete `strategies/*.md` record with its r3 section
written, which is the prerequisite the document sets.

**Note on standing:** gen_r2's verdict recommended eleven promotions and `promote.sh` was never
run — there is no `exemplars/` directory. So this is the campaign's **first** promotion, the
near-duplicate prohibition has nothing on disk to measure against, and this set has to carry
both rounds' lessons rather than only this round's.

**Ground 1 — the fastest correct entry for each geometry, one per L, always.** Six entries cover
the eleven cells: `gen_pfa_small` (10, 15), `gen_batchlane` (12, 20), `gen_powp` (25, 27, 50,
100), `gen_rader` (31), `gen_pow2` (32), `gen_pfa_large` (40). Mandatory, all six kept.

Three of those crowns are ties inside measurement (`gen_pfa_small`/`gen_batchlane` 0.09 % apart
at L=10; `gen_powp`/`gen_pfa_large` 0.07 % apart at L=50 and 1.4 % at L=100). Both members of
each pair are kept under ground 1 regardless, so nothing turns on breaking them. `gen_batchlane`
is kept **despite** shipping the round's one regression (§2): it holds L=12 and L=20 outright,
and its record documents the regression's mechanism and two of the round's best negative results
(the +40 % c-interleave, the dead-store/out-lining asm audit) — which is exactly what makes a
record useful later.

**Ground 2 — a structurally different runner-up when it is close.** `gen_dense_prime` at L=31,
123.828, 1.46× behind `gen_rader`. This is verbatim `CURATION.md`'s own worked example (Rader
winner, dense conjugate-symmetric alternative), it is the campaign's assigned crossover fight,
and this round it became the most interesting open question on the panel rather than a narrowing
gap: the scoring pass's own wisdom file picked *dense* over Rader at a 79 % margin inside
`gen_planner` while the specialists went the other way by 46 % (§5). Keeping only the winner
would throw away half of the round's headline literature result. It also independently satisfies
ground 4 — it beats the best library at L=15 (16.328 vs 16.464), L=20 (43.509 vs 44.869) and
L=31 (123.828 vs 715.949), which for an O(L⁴)-class dense kernel is the point. Kept.

**Ground 4 — anything that beat a library baseline, regardless of rank.** Read literally this
sweeps in every entry at L=31, where all eight supported entries beat all six libraries. That
would defeat curation, so — following gen_r2's precedent — it is applied to entries whose
library win *is* their distinguishing result.

* **`gen_planner`** — beats the best library at **10 of 11 cells** with a *generic* any-L engine
  (losing only L=10, 4.741 vs 4.553), improved 24–62 % everywhere, and closed the distance to
  the hand-tuned class winners from 1.9–6.1× to 1.2–2.3×. It is brief deliverables 1 and 2, it
  is the engine `gen_race` and `gen_pfa_large` now include, and it carries the round's most
  reusable diagnosis (the executor was compiling half-scalar; intrinsics were worth 24–62 %).
  This is the entry round 6 is actually scored on. Kept.
* **`gen_race`** — beats the best library at 10 of 11, and is brief deliverable 3. It carries the
  §4.2 crossover receipt *and* the round's cache-invalidation result (engine-generation salting,
  with the ~2×-at-a-scored-size hazard it caught), the second-stage tile race, and the measured
  warm-create budget (5–7 ms against 50 ms). Its adoption receipt is now three entries deep
  (`gen_powp`, `gen_planner`, `gen_pfa_large`), and the `gen_pfa_large` adoption is what fixed
  gen_r2's one compliance failure. Kept.
  *`gen_race` and `gen_planner` again post near-identical timings (0.3–4.1 % apart at all
  eleven). They are not code duplicates: one is the enumerator and executor, the other the tuner
  and the per-host cache; they are separate mandated deliverables; and the identical timings are
  the consequence of the adoption, not evidence of redundancy. Both kept, on that stated
  reasoning — the same reasoning gen_r2 gave.*
* **`gen_layout`** — a dense O(L⁴)-class floor that beats the **best** library at L=20 (40.346 vs
  44.869), L=25 (92.999 vs 109.921), L=27 (124.173 vs 144.315), L=31 (226.722 vs 715.949) and
  L=40 (486.922 vs 520.222). A dense floor beating a production FFT at five sizes is §08's "won
  by *layout*, not by the kernel" reproduced on our own node. It also articulated the doctrine
  the other three layers adopted this round — *"churn in a layer others `#include` is its own
  cost"* — froze its API, and spent the round dogfooding its own collision model so adopters have
  a worked example. Kept.
* **`gen_twiddle`** — the fourth mandated layer, scored by adoption, and this round's receipts are
  the strongest yet and are *verified*, not asserted: it built exact fillers and audits for both
  of `gen_planner`'s new table layouts and checked them **slot-for-slot against gen_planner's own
  builders** at n = 7, 11, 13, 31 (MATCH); it shipped `tw_fill_rader_half`/`tw_audit_rader_half`
  for `gen_rader`'s any-prime mandate; `gen_bluestein`'s r2 adoption still compiles clean against
  the new file. It beats no library and does not need to — the mandate is adoption, and
  `tw_primroot` + `tw_fill_rader_half` + `tw_fill_rader_fft` is exactly the plan-time machinery
  round 6's surprise primes will need, as three calls. Its own entry improved 17–36 %. Kept.

**Ground 3 — instructive failures.** The round produced an unusually strong crop of documented,
measured negative results, and **every one of them lives inside an entry already promoted**: the
+40 % c-interleave and the register-pressure regression (`gen_batchlane`), five measured
non-transfers of borrowed techniques (`gen_pfa_small`, `gen_powp`, `gen_rader`), three
independent losses for manual cross-codelet scheduling (`gen_pow2`), the whole anti-alias pitch
program (`gen_dense_prime`), and the `ipm` fused-map refutations (`gen_pfa_large`, `gen_powp`).
No additional entry needs promoting to preserve them.

**Not kept — `gen_bluestein`.** It is the mandated existence fallback and it did its job again:
all eleven cells, every gate passed, improved 8.7–22 % at every one, and it now owns the chain
like everyone else. But it satisfies no ground — not fastest anywhere, not a close runner-up
(3.5–5.6× behind the leader at ten of eleven cells), and not an instructive failure, because it
did not fail; it is the floor by construction. Its one library win is at L=31, where every entry
has one, which is precisely the case gen_r2 declined to read literally.

This is not a costless decision this round, and the reason is worth writing down: **`gen_bluestein`
produced the only direct measurement anyone made at the L2↔DRAM boundary** — the 15 MiB
map-placement gate, 15.98/16.04 ms separate against 18.86/19.25 fused at L=100, 4/4 reps, with
the crossover bracketed at L=25 (fused wins), L=31 (wash) and L=50 (separate wins). That result
contradicts the panel's universal fuse-the-map doctrine and it is directly relevant to §6's
top item. It is recorded in §5 of this verdict, its record is tracked regardless, and its source
is preserved in `impl_3/` regardless. Promoting the Bluestein engine would put a near-duplicate
of the floor on a reading list to carry a lesson that belongs — and now is — in the verdict.

Eleven of twelve is again a wide promotion. It is deliberate: eleven entries improved at every
cell they own, one regressed at one of four, no entry failed a gate, and four of the twelve are
mandated library layers with cited and in one case independently verified adoption receipts.
The set is wide because the round was good.

---

## 8. Round note for `promote.sh`

**What gen_r3 established.** Twelve entries, 614/614 correctness PASS, zero build errors, zero
panel crashes, zero unauthorised absences. The panel beats the best library at all eleven scored
cells (geomean 3.20×, worst 2.00×) — but at the *top* of the board the round changed almost
nothing: nine of eleven leaders moved less than 2 %. The round's real result is one layer down.
**The generic any-L engine went from 15 % slower than the best library (geomean 0.87×) to 1.43×
faster, winning 10 of 11 cells, and closed the distance to the hand-tuned class winners from
1.9–6.1× to 1.2–2.3×** — which is what round 6, scoring the assembled library on unannounced
sizes, actually measures. The mechanism was not a better algorithm: `gen_planner`'s r2 executor
was compiling half-scalar, and replacing the loops with AVX-512 intrinsics was worth 24–62 %.
Three entries now report that an assembly audit paid more than any A/B they ran.

LITERATURE §4.2 moved decisively and in two directions at once: inside `gen_planner`'s
register-tiled engine the scoring pass's own race picks dense-31 over Rader-31 at a **79 %
margin**, while between the specialists Rader still beats dense by **1.46×**. The dense/Rader
crossover is a property of kernel quality, not of arithmetic — which is what §4.2's own
reconciliation guessed and could not test. §4.6 gained a failure mode the corpus does not
contain: if the engine generation changes and the candidate names don't, stale wisdom replays
silently, and here it would have cost ~2× at a scored size. gen_r2's compliance failure is
fixed — `gen_pfa_large` adopted the wisdom cache and its L=100 plan time went 6.339 s on every
process to 7.6 s cold / 3 ms warm.

**Three things gen_r3 did not do.** §4.3's re-opened construction — tile the batch to L2, then
run all three axes inside the tile — went untried for the third round running, at exactly the
three cells where our margin is worst (2.00× / 2.00× / 1.55×); the round did, however, produce
three independent findings that L=100's phase 1 is issue-limited rather than miss-limited, which
makes that experiment a sharper test than before. The `fftw3_guru` baseline the round added ran
**one of eleven cells** and lost to plain `plan_many_dft` there by 26 %, so the stronger-FFTW
claim is unscored. And the round-3 generality mandate — any size within class — is delivered in
code and **entirely unscored**: eleven off-suite sizes exist only as author-attested numbers from
leased cores, with round 6 three rounds away and a whole-library zero riding on them.

**What gen_r4 should attack.** The L2-tile-then-fuse experiment at L=40/50/100 with an issue-rate
counter alongside the miss counters. Then the two hygiene debts that have now survived a monitor
instruction: revert or beat `gen_batchlane`'s L=15 register-explicit pencil (the round's one real
regression, +6.5 %), and put the off-suite generic sizes plus `fftw3_guru` through the harness so
round 6 is not the first time anyone measures them.

---

*Adversarial summary: no entry failed correctness, none crashed, none is missing that was
required. The three things a reader should not take on trust from this round are*
`gen_batchlane`'s *L=15 claim of 4.456 (scored 4.771, refuted by a same-pass control),*
`gen_race`'s *t64-at-L=50 tile win (+2.3 % claimed; the scoring pass's own race recorded t32
winning as a tie), and every off-suite generality number in §3.5 (author-measured, never gated).
The* `fftw3_guru` *baseline named in this round's commit log does not exist in this round's
leaderboard.*

PROMOTE: gen_pfa_small gen_batchlane gen_powp gen_rader gen_pow2 gen_pfa_large gen_dense_prime gen_planner gen_race gen_layout gen_twiddle
