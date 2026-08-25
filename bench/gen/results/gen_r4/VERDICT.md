# VERDICT — round gen_r4

Monitor's judgement on the measured round. Source of record:
`results/gen_r4/leaderboard.txt`, `environment.txt`, `failures.txt`,
`build_errors.txt`, `results/wisdom_a80n0.json` (as written by the scoring race),
`strategies/*.md` §"Round gen_r4", and `docs/CURATION.md`.

---

## 0. Three corrections to the framing I was given

These change what can honestly be reported, so they go first.

**0.1 The geometries are not 6, 8, 17, 36.** Those were the *previous*
(fixed-size) campaign's sizes. This is the GENERALIZE campaign, whose acceptance
suite is frozen in `cases.txt` and is **L = 10, 12, 15, 20, 25, 27, 31, 32, 40,
50, 100**. All eleven were measured. I report on those. The only r4 data touching
the named sizes is unscored: `gen_pfa_small` measured L=36 through its generic
coprime engine (252.6 µs in-place vs 257.4 buffered, −1.9%), and
`gen_dense_prime` measured L=17 in its prime sweep (29.96–31.22 µs at B=4 m=8,
−7% vs r3). Neither is on the board.

**0.2 The scoring machine is not Cascade Lake.** `environment.txt` says
`Intel(R) Xeon(R) Gold 6326 @ 2.90GHz` with `avx512_vbmi / avx512_ifma /
avx512_vnni` — that is **Ice Lake-SP** (node a80n0), which is what
`PANEL_BRIEF.md` specifies as the scoring host. Cascade Lake and Sapphire Rapids
are the *advisory* cross-architecture reruns; per the brief they land **after**
this round's board, and `XARCH.md` does not exist yet. There is therefore no
cross-machine evidence in this round at all.

**0.3 There is no batched/non-batched pair per geometry.** Each acceptance cell
is scored at exactly one batch: B=64 at 10/12, B=32 at 15/20, B=16 at 25/27/31,
B=8 at 32/40, B=4 at 50, and **B=1 at L=100 only**. Where implementers measured
B=1 at a batched cell, those numbers live in the strategy records and are quoted
below as such — they are not scored.

The consequence for item 4 of my brief is spelled out in §4: implementers develop
on *leased cores of this same node*, so a claimed-vs-measured gap here cannot be
attributed to a machine difference. The real confound is node state, and this
round the panel independently discovered and named it.

---

## 1. Headline per geometry

Fastest correct panel entry vs. the strongest library configuration in the same
cell. Every backend at every cell reported `ok` (see §3).

| L | B | winner | µs/transform | GF/s | spread | best library | µs | win vs library |
|---|---|---|---|---|---|---|---|---|
| 10 | 64 | **gen_pfa_small** | 1.153 | 43.22 | 0.1% | mkl_dfti | 4.553 | **3.95×** |
| 12 | 64 | **gen_batchlane** | 1.912 | 48.60 | 0.2% | mkl_dfti | 7.737 | **4.05×** |
| 15 | 32 | **gen_pfa_small** | 4.429 | 44.65 | 0.2% | mkl_dfti | 16.559 | **3.74×** |
| 20 | 32 | **gen_pfa_small** | 13.059 | 39.71 | 1.7% | fftw3_measure | 44.889 | **3.44×** |
| 25 | 16 | **gen_powp** | 32.100 | 33.91 | 5.8% | fftw3_patient | 108.437 | **3.38×** |
| 27 | 16 | **gen_powp** | 44.281 | 31.70 | 6.8% | mkl_dfti | 144.393 | **3.26×** |
| 31 | 16 | **gen_rader** | 84.603 | 26.17 | 1.2% | ducc0_c2c | 720.156 | **8.51×** |
| 32 | 8 | **gen_pow2** | 55.746 | 44.09 | 1.0% | mkl_dfti | 171.200 | **3.07×** |
| 40 | 8 | **gen_pfa_large** | 188.718 | 27.07 | 0.4% | mkl2026_dfti | 404.827 | **2.15×** |
| 50 | 4 | **gen_pfa_large** | 466.040 | 22.71 | 0.7% | mkl_dfti | 945.688 | **2.03×** |
| 100 | 1 | **gen_pfa_large** | 4827.566 | 20.64 | 0.8% | mkl_dfti | 7813.913 | **1.62×** |

**Geomean over the eleven cells: 3.30× the best library; 3.01× excluding L=31.**
Worst cell is L=100 at 1.62×; best real cell is L=12 at 4.05×.

Three qualifications on this table, all adverse:

- **L=31's 8.51× is a property of the size, not of the entry.** Every library
  collapses on a prime edge: MKL 2022/2026 and all three FFTW planners land at
  848–883 µs, 10× off gen_rader, and ducc0 is only marginally better. *All seven*
  gen entries present at L=31 beat the best library there, including
  `gen_bluestein` (299 µs) and `gen_twiddle`'s demo (463 µs). I do not credit
  "beat a library at L=31" as an achievement anywhere in §7.
- **L=100 is a statistical tie, not a win.** `gen_pfa_large` 4827.566 vs
  `gen_powp` 4828.216 is a **0.013%** gap, and `gen_powp`'s run spread at that
  cell is **14.1%** — the widest of any gen entry on the board. The reported
  minimum comes from one lucky run. The honest statement is that the two engines
  are indistinguishable at L=100; the rank order flipped from r3 (where gen_powp
  led by 1.4%) and means nothing.
- **L=50 is a near-tie too**: 466.040 vs 472.873 is 1.47%, and `gen_pfa_large`'s
  own record says of this cell, verbatim, *"do not claim 50 B=4 as a win on this
  evidence."* I honour that. See §3.5 for why gen_powp probably lost 1.5% it did
  not have to.

**B=1 outside L=100** (record numbers, unscored, for the round-6 generality
question): `gen_pfa_small` 3.869 / 5.258 / 14.140 / 31.966 µs at 10/12/15/20 via
its split path; `gen_batchlane` 10.52 / 17.44 / 41.66 / 116.5 µs at the same
sizes — a 2.5–3.6× hole against its own batched numbers and roughly 2.3× behind
MKL, on the panel's list for four rounds running. `gen_rader` B=1 85.663 vs
84.725 batched, `gen_dense_prime` B=1 ≈ B=16, `gen_planner` B=1 ≈ batched
everywhere. The B=1 weakness is confined to the two SoA-8 lane engines.

---

## 2. What changed since gen_r3, per geometry

Deltas are r3 board → r4 board, same node, same harness, same seed.

### The round's real content

**L=10 / 12 / 15 / 20 — `gen_planner`'s fused CT codelets, and nothing else.**
The tuned winners moved by ≤1.5%; the *generic* layer moved 11–27%:
gen_planner −27.0 / −23.6 / −21.8 / −10.6%, gen_race tracking it. This is the
single largest algorithmic delivery of the round — a CT node with n = r·m ≤ 25 and
both stages hard leaves now runs as one register-resident kernel per column chunk
(zero arena round trip, zero per-leaf dispatch), with arithmetic byte-identical to
the staged leaves. gen_planner is now ahead of MKL at **all eleven** cells (r3:
ten of eleven; L=10 was the last red cell).

**L=25 / 27 — same story, generic layer only.** gen_planner/gen_race −13% and
−18%; gen_powp's tuned SoA engine moved −0.3% and −1.2%, i.e. not at all. The
class winner's margin over the generic layer is still 1.87× and 1.94×.

**L=31 — nothing moved, and that is the finding.** gen_rader −0.6%,
gen_dense_prime −2.7%, gen_planner −0.4%. gen_rader costed the shipped step
against the issue-port budget before touching code and found the measured 85 µs
*is* the model (~250k cycles at 2.9 GHz), then shipped a plane-custody change
worth −0.35% and struck three items off its own backlog with arithmetic instead
of burned windows. gen_dense_prime independently concluded its 1.42× gap to Rader
"is arithmetic and will not close by engineering." Both are right; the cell is
saturated.

**L=32 — `gen_pow2` −2.7%** from storing the custody `c` volume x-fastest so the
x-pass reads one contiguous 4 KB stream instead of 32 touches at 18.5 KB stride.
Bit-identical output, confirmed in three same-window interleaved sets.

**L=40 — `gen_pfa_large` −6.8%**, the round's best tuned-size gain, from adopting
the volume-major chain schedule (all m steps on one volume before the next). This
entry was the last chain owner still step-major. `gen_bluestein` −7.5% at the
same cell from fixing a shipped gate (§3.4).

**L=50 — `gen_pfa_large` −1.7%, `gen_powp` −0.2%**; the generic layer −17%.
Volume-major did not help here (16 MB step-major already rode the 24 MB LLC).

**L=100 — `gen_pfa_large` −5.1%, `gen_powp` −3.8%** from the new plane-granularity
deferred-map family (`ipp`), and **`gen_layout` −18.9%** from non-temporal stores
(§5). The generic layer −12.1%.

### Regressions

Three cells regressed against a **flat or faster** library baseline, which is
what distinguishes a regression from a hot window here.

| entry | cell | r3 → r4 | libraries in the same cell | verdict |
|---|---|---|---|---|
| **gen_race** | L=40 B=8 | 269.776 → 284.364 (**+5.4%**) | mkl −0.2%, mkl2026 0.0%, fftw3_measure 0.0% | **real regression** |
| **gen_layout** | L=40 B=8 | 486.922 → 507.702 (**+4.3%**) | as above | **real regression, unremarked** |
| **gen_race** | L=32 B=8 | 127.314 → 131.527 (**+3.3%**) | mkl **−2.8%**, mkl2026 **−2.7%**, gen_pow2 −2.7% | **real regression** |

`gen_race`'s record predicted exactly this and named the falsifier: *"L=32 and
L=40 read above their r3 boards (+4–5%) — but MKL read +5% in the same windows, so
I claim window heat, not regression. If the r4 board disagrees, these two cells
are where to look first."* **The board disagrees.** In the scored windows the L=32
cell ran ~2.7% *faster* for every other backend and L=40 was flat. The window-heat
defence does not survive. Note also the rank inversion: in r3 the race layer beat
gen_planner's own default at both cells (127.3 vs 128.3, 269.8 vs 277.8); in r4 it
is behind at both (131.5 vs 129.8, 284.4 vs 282.6). The race layer stopped adding
value at these two cells and started subtracting it.

`gen_layout`'s L=40 regression is worse in one respect: its r4 record **does not
measure L=40 at all**. Every other cell it touched is documented; this one is a
4.3% loss with no A/B, no attribution, and no mention.

Smaller moves I judge to be noise, not regressions, because the record contains a
same-core interleaved A/B on unchanged code and the board number falls inside or
below the implementer's own measured range: gen_batchlane L=20 +1.0%,
gen_dense_prime L=20 +1.5%, gen_twiddle L=10 +2.1% / L=20 +2.3%, gen_planner
L=32 +1.2% / L=40 +1.8% (board 282.620 is *faster* than the whole 283–292 range
they measured), gen_bluestein L=32 +1.2% on a byte-identical code path.

### Lead changes

- **L=20: `gen_batchlane` → `gen_pfa_small`** (13.145 vs 13.059, 0.7%). Not a
  regression — gen_batchlane's path is unchanged and its board number is *better*
  than the 13.394 it measured. gen_pfa_small took the cell by moving the m-loop
  inside the SCHED step function (−1.3%). A 0.7% lead at 1.3–1.7% spread is a tie
  in substance.
- **L=50 and L=100: `gen_powp` → `gen_pfa_large`.** Both are ties (§1). At L=50
  the flip has an identifiable cause that is not performance — see §3.5.

---

## 3. Adversarial pass: failures, correctness, builds, absences

### 3.1 Correctness — clean, and I checked it rather than assuming it

Every one of the 16 backends at every one of the 11 cells reports `ok` against the
two-part gate. Single-call residuals sit at 1.9e-14…3.4e-13 against a 1e-10
tolerance; two-step-chain values are 7e-16…4e-15 against 3e-14. Nothing is near a
limit and nothing is suspiciously *exact*. Independent corroboration in the
records: `gen_pow2`, `gen_rader`, `gen_dense_prime`, `gen_pfa_large`, `gen_powp`,
`gen_twiddle` and `gen_bluestein` each verified chain outputs **bit-identical
across independent node processes**, and four of them verified bit-identity
against their own r3 binary (which is the strongest available evidence that a
structural change moved no arithmetic). **No fast wrong answer is present on this
board.**

### 3.2 Failures — one, and it is the reference floor

`failures.txt` contains exactly three lines, all the same entry:

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is `timeout`. `baseline_matrix` is the O(L⁴)/volume/axis library-free
reference; at L=100 that is 10⁶ points × 100 per axis, and it took 34.3 ms/transform
at L=50 already. It timed out on all three runs and is correctly absent from the
L=100 table. This is not a panel entry and not a defect — but it is worth
recording that **L=100 is the one cell with no library-free harness floor**, so
its correctness rests entirely on the numpy gate. That gate did run and did pass
for all fourteen backends present.

**No panel entry crashed, hung, or was killed at any cell.**

### 3.3 Build — two undefined-behaviour warnings that must be fixed

`build_errors.txt` contains no errors. It contains two warnings, and they are the
same warning in the same borrowed routine:

```
impl/gen_dense_prime.c:1512:23: warning: iteration 1152921504606846976 invokes
    undefined behavior [-Waggressive-loop-optimizations]
 1512 |         double re = zp[2 * i] + cp[2 * i];
impl/gen_rader.c:1023:23: warning: iteration 1152921504606846976 invokes
    undefined behavior [-Waggressive-loop-optimizations]
 1023 |         double re = zp[2 * i] + cp[2 * i];
```

1152921504606846976 is 2⁶⁰: GCC has proved that `2 * i` overflows the signed index
type at that iteration, i.e. the scalar map tail loop `for (; i < npts; ++i)`
contains signed-overflow UB that the optimiser is entitled to exploit. It does not
fire today — `npts` is nowhere near 2⁶⁰, the gates pass, and gen_rader's L=31
output is bit-identical to r3 — so **this does not invalidate any number on this
board**. But it is real UB in a hot-path tail, GCC has already noticed it, and a
different inlining decision or compiler version can turn it into a wrong answer or
a truncated loop with no diagnostic.

The aggravating detail: `gen_rader`'s backend line says *"s6 map adopted from
gen_dense_prime"*, and the two loops are character-identical. **The defect
propagated by adoption**, which means every future adopter of that map tail
inherits it. Both entries are on my promotion list; the fix (index through
`size_t`, or hoist to `zp[i2]` with `i2 += 2`) is one line each and should land
before either is used as exemplar material. Named here so the next panel does not
copy it a third time.

### 3.4 Instructive failure of the round: `gen_bluestein`'s gate shipped wrong for a round

Not a correctness failure — a *policy* failure, and the best-documented negative
result on the board. `gen_bluestein`'s r3 record claimed its map-regime gate
"splits the suite exactly: 10..32 fused, 40/50/100 separate." The shipped code
gated on state bytes alone (`nrows*L*16`) while the quantity the r3 race actually
measured was state + c combined (`nrows*L*32`). L=40 and L=50 therefore sailed
under the threshold and ran the **fused** regime — the one its own r3 race had
measured 15–30% slower. Verified on the node before the fix: L=50 B=4 default
1986.8 µs vs forced-separate 1822.2; L=40 forced-fused 1381.9 vs blocked-separate
1086.6. **A one-character bug cost ~21% at L=40 and ~9% at L=50 for a full round**,
and it was invisible because the policy *code paths* were all tested — only the
gate itself was not. The lesson the record draws is the right one and worth
promoting for: when a race sets a policy gate, run one forced-knob measurement per
side of the gate.

### 3.5 `gen_powp` lost L=50 to its own hysteresis policy, not to its kernel

This is visible in `results/wisdom_a80n0.json` as written by the scoring race:

```
gen_powp/chain4/L50/B4#5e475f32 -> l50-ipp0   us=475.91  tie=1  margin=-0.0182
```

Margin **−1.82%**: the scoring race measured `ipp0` as 1.8% *slower* than the
alternative and installed it anyway, because gen_powp promoted `ipp` to rank-first
this round so it wins inside the 3% hysteresis band. Their record declares this
tax openly — *"the rank reorder has a measurable worst case… rank-first means we
eat ≤3% when ipp genuinely trails"* — and accepted it on the grounds that both
scoring-relevant cells showed ipp ahead in paired A/Bs. At L=50 the scoring race
says otherwise, and gen_powp lost the cell to gen_pfa_large by **1.47%**. The tax
they budgeted is larger than the margin they lost by. That is a fair trade they
chose with eyes open, but the board is entitled to record that the policy, not the
kernel, decided the cell.

### 3.6 Nothing is missing

I checked every entry against the class table in `PANEL_BRIEF.md`. Presence is
exactly what each `support()` should declare:

| entry | cells present | class coverage | verdict |
|---|---|---|---|
| gen_pfa_small | 10,12,15,20 | PFA coprime small | correct |
| gen_batchlane | 10,12,15,20 | SoA batch-lane, 10/12/15 + adopted 20 | correct |
| gen_dense_prime | 10,12,15,20,31 | dense prime p≤31 + helps small composites | correct |
| gen_powp | 25,27,50,100 | p^k CT | correct |
| gen_pfa_large | 40,50,100 | PFA coprime large | correct |
| gen_pow2 | 32 | 2^k | correct |
| gen_rader | 31 | Rader primes | correct |
| gen_bluestein | all 11 | must run everywhere | correct |
| gen_planner / gen_race / gen_layout / gen_twiddle | all 11 | library layers | correct |

**No entry is silently absent from a cell it owns.** Every declined cell is a
declared class boundary, not a failure hiding as an absence. That is a clean sheet
and worth stating explicitly, because it is the failure mode this check exists for.

---

## 4. Claimed vs. measured

**The premise I was given does not apply to this round, and I will not invent a
machine difference to explain gaps that have a documented cause.** Per
`PANEL_BRIEF.md`, implementers develop via `tryout.sh` on **leased cores of a80n0
itself** — the same Ice Lake node the score is taken on. Sapphire Rapids and
Cascade Lake are advisory reruns that have not happened. The dev-vs-score
agreement confirms this: across all twelve records, **most claimed numbers land
within 1–2% of the scored value**, which no cross-machine comparison could
produce.

| entry | cell | claimed in record | scored | Δ |
|---|---|---|---|---|
| gen_rader | 31 | 84.725 | 84.603 | −0.1% |
| gen_pfa_large | 40 | 188.8 | 188.718 | −0.0% |
| gen_planner | 20 | 24.69 | 24.685 | −0.0% |
| gen_dense_prime | 31 | 119.91–121.05 | 120.490 | in range |
| gen_pow2 | 32 | 55.27–55.52 | 55.746 | +0.4% |
| gen_batchlane | 15 | 4.567–4.589 | 4.566 | −0.0% |
| gen_pfa_small | 20 | 13.123 | 13.059 | −0.5% |
| gen_bluestein | 100 | 15545–15582 | 15621.150 | +0.3% |

Where a gap does exist it runs in the **opposite direction to a machine penalty** —
the scored number is usually *faster* than what the implementer could measure —
and the cause is the one the panel itself diagnosed this round:

**`gen_powp` at L=27: claimed 47.6–47.8 µs "every window this session", scored
44.281 µs — the score is 7.4% faster than anything they could reproduce.** Their
code at this cell is bit-identical to r3, whose board read 44.811. Their own
conclusion — *"the quiet floor remains r3's 45.0"* — is confirmed by the board. The
gap is node state, not code and not silicon.

Same direction, same cause, smaller: `gen_layout` L=100 claimed 15606–15845,
scored 15083.150 (−3.3%); L=50 claimed 1290, scored 1204.407 (−6.6%); L=31 claimed
231.4, scored 223.461 (−3.4%). `gen_pfa_large` L=100 claimed 4923 quiet, scored
4827.566; L=50 claimed 491.7 raw / ~476 adjusted, scored 466.040.
`gen_dense_prime` L=10 claimed 5.40–5.44, scored 5.282.

**The mechanism, which is this round's most valuable methodological finding, was
found by `gen_batchlane` and independently confirmed by four others.** `tryout.sh`
acquires a *fresh slot lease per invocation*, so consecutive A/B runs often land on
**different cores in different turbo/neighbour states** — states measured 10–25%
apart while MKL (memory-bound) moved <2%. `gen_pfa_small` measured the leased core
flipping between two sustained states ~13.5% apart with in-run sd of 0.03–0.07%
*inside each state*, so a single clean-looking run can be entirely in the slow
state and read as a code regression. The consequences were not hypothetical:

- gen_batchlane **flipped a shipped default backwards for two rounds** (SCHED15) on
  cross-invocation pairs; same-core interleaving reversed it (−4.3%).
- gen_batchlane's `map_col` epilogue measured a −4.5%/−8.7% *win* cross-invocation
  and a +6…+9% *loss* under same-core interleaving. It was nearly shipped.
- gen_pfa_large *"nearly killed ipp on a window-drift artifact"*: sequential forced
  A/B read +12% for ip1, tight alternation reversed it to ipp1 winning 4 of 5.
- gen_twiddle: *"my first tryout read at L=100 was 11225 (looked like a win); the
  first same-core interleave then showed it LOSING 3/3."*
- gen_bluestein's prefetch re-race read a 1.8 ms *win* in a dirty window and lost
  3/3 under control-first pairs.
- gen_powp read L=27 B=1 at 68.0 µs (+14%, looked like a regression); the held-lease
  rerun read 59.85.

The correct attribution for every claimed-vs-measured gap on this board is
**bimodal core state on a shared leased node**, and the fix — hold one lease,
build all arms side by side, alternate the same binaries on the same core, take
min-of-mins per config — is already institutionalized (§5.3). The machine-difference
question genuinely opens when `XARCH.md` lands after this board.

### 4.1 But the scoring race did not reproduce three implementer verdicts

This is the one place where "claimed vs measured" is a real problem rather than
node noise, and it is a property of the **race layer**, not of the kernels. Several
entries strip their wisdom keys at round end so the monitor cold-races in its own
quiet window; `wisdom_a80n0.json` therefore records the picks actually used at
score time. Three disagree with held-lease measurements the implementers reported:

| cell | implementer's measured verdict | scoring race picked | cost |
|---|---|---|---|
| L=12 | `c4(d3)`, +2.0% real margin, picked twice interleaved (2.5%, 3.7%) | `c3(d4)`, **tie, margin −0.0196** | claimed 4.993 → scored 5.107 (+2.3%) |
| L=40 | tile `t16`, "+2.2% NON-tie" | `t32`, **tie, margin −0.0076** | part of the +5.4% regression |
| L=100 | tile `t32`, "5524@32 vs 5553@64 — tile 32 stays, third round running" | **`t64`, non-tie, margin +0.0335** | gen_race 5640.202 vs gen_planner's t32 default 5466.892 (**+3.2%**) |

The L=100 case is the sharp one: the cold race declared t64 a confident 3.3%
winner against a tile choice that has now survived three rounds of A/B, and the
resulting binary finished 3.2% behind the planner running its own default. So
`gen_race`'s headline claim needs qualifying: interleaving made the racer
**self-consistent within a session** (the L=12 SEQ flip-flop is genuinely fixed),
but it did **not** make it agree with held-lease same-core A/B across sessions.
Race-picks-worse-than-default now appears at three cells (12, 40, 100) and
coincides with two of the round's three regressions.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### §4.3 — "Is axis fusion worth 3× or 3%?" — moved decisively, in its *re-opened* regime

§4.3 was marked *"SETTLED IN PART by panel_r3/VERDICT.md §5, and re-opened in one
regime by §08 §1.9."* The re-opened regime is named precisely: every panel
experiment to date fused across an **L1↔L2** boundary (measured gap 2.6×); the
untested case is **L2↔DRAM** (gap 7×), and the recommended construction is
*"tile the batch so a tile fits L2, then run all three axes inside the tile."*
§4.3 calls it **"the largest untried structural move on the board."**

**Round gen_r4 tried it, in five entries, independently, and the answer is now
quantified.**

The construction appears in three forms:

1. **Volume-major chain schedule** (`gen_pfa_large`, adopted by `gen_powp` from
   the earlier `gen_dense_prime`/`gen_rader`/`gen_layout` records): run all m
   chain steps on one volume before touching the next, dropping the per-step
   working set from the whole batch's state+c to one volume's slice. Outputs
   bit-identical. **L=40 B=8: 211.3 → 188.8 µs (−11%); L=50 B=1: 530.7 → 482.9
   (−9%); L=50 B=4: no gain** (its 16 MB step-major set already rode the 24 MB
   LLC).
2. **k-plane-blocked custody** (`gen_bluestein`, from `gen_layout` r3's plane
   window; adopted onward by `gen_twiddle`): axis 0 globally, then axes 2, 1 and
   the map sweep per block of k = 8/gcd(L,8) planes while the block is L2-hot.
   Block size chosen so the 8-row group decomposition is *provably identical* to
   the unblocked pass, hence bit-identical output. **L=100 −1.2% board, L=40
   −4.5%, L=50 −6.5%.**
3. **Plane custody in the pass schedule** (`gen_rader`): step s+1's z-pass runs
   per plane immediately after that plane's map. **L=31 −0.35%** (L2-resident) but
   **p=127 −0.9%, 3/3** (32 MB, DRAM-resident).

**The measured answer, and it is a refinement of Tolmachev's rule rather than a
confirmation of the 7× hope:**

- **The payoff is set by which cache level the deleted round trip actually lived
  at, and it is gated by residency.** gen_bluestein states the mechanism from its
  own data: *"the win is smaller than the traffic model because the 16 MB state
  already fits the 24 MB LLC: the deleted round trips were LLC-level, not DRAM."*
  Per-step traffic at L=100 fell 144 → 96 MB of touched volume-passes for a 1.2%
  gain, because the traffic was never going to DRAM.
- **Below L2 residency, blocking is a measured loss.** `gen_twiddle` shipped the
  same custody change ungated and measured **+0.6 / +1.9 / +3.8 / +0.8% at L=27
  (4/4 same-core pairs, 7/7 counting earlier sessions)**, then added a residency
  gate (`32·L³ > 1.25 MB`) that restored parity. `gen_bluestein` measured the
  fused-regime variant at **+10% at L=31**. `gen_rader` recorded the boundary in
  one sentence: the win is at DRAM-resident p, not at L2-resident 31.
- **And the largest effect at the one genuinely DRAM-resident size was not pass
  fusion at all.** `gen_layout` took L=100 from 19509 → 15606–15845 µs
  (**−19%**) and then did the knob knockout that settles the attribution:
  default 15829; prefetch disabled 15845 (**wash**); **NT stores disabled
  19590**. *The entire win is non-temporal stores* — eliminating the
  read-for-ownership line read on three full-volume store streams, 48 MB of
  eliminated read traffic per chain step. Their r3 diagnosis (that the ~100
  concurrent fold row streams were the binder) *"stands corrected in detail: the
  fold LOADS were never the binder at L=100 — out-of-order execution rides them —
  it was the three full-volume store streams whose RFO reads doubled the write
  traffic."*

**So the number §4.3 asked for, on CPU, for this size range:** tiling the batch to
L2 and fusing the axes inside the tile is worth **9–11% where the batch working
set genuinely exceeds LLC** (volume-major at L=40 B=8, L=50 B=1), **1–6.5% where
the deleted round trip was only LLC-resident**, and **−0.6 to −10% where the
volume was already L2-resident** — it must be gated on residency, not applied. And
at the one DRAM-resident cell, **store-side RFO elimination beat pass-count fusion
by roughly 4×** (−19% vs −1.2% at the same L=100). That last point is new: the
corpus's framing of fusion is entirely about avoided read passes, and on a
write-heavy in-place chain the write traffic was the bigger prize.

### Secondary movement

**§4.8 item 5** — *"No quantified comparison of batch/vector-loop placement for
batched small 3D transforms… our largest untapped search axis, and the literature
only gestures at it."* Volume-major vs step-major **is** batch-loop placement, and
it is now quantified with bit-identical outputs: −11% at L=40 B=8, −9% at L=50
B=1, ~0 at L=50 B=4. Partially closed.

**§4.6** — *"model versus search for the instruction schedule"* — moved
methodologically rather than substantively. §4.6 says the schedule is "the primary
thing to search"; this round showed that on a shared leased node the *search
protocol itself* was invalid (§4), that fixing it reversed a shipped default and
killed a false positive, and (§4.1) that the fixed racer still disagrees with
careful same-core A/B at three cells. The corpus's confidence in cheap search
should be discounted by the cost of a trustworthy measurement.

**§4.5 / §4.1** — untouched this round. No entry checked
`ld_blocks_partial.address_alias`, and `gen_layout`'s r4 addition
`gl_alias_drained4k` is a *model* of when the 4K-alias fix should pay (derived from
gen_dense_prime's negative result), not a measurement of the counter. §4.5's
"neither has ever been checked" still stands.

---

## 6. The single highest-value thing the next round should attack, per geometry

| L | attack | why it is the highest-value one |
|---|---|---|
| **10, 12** | **The B=1 lane-spatial engine**, built once and shared between `gen_batchlane` and `gen_pfa_small` | Fourth round on both entries' lists. gen_batchlane is 2.3× behind MKL at B=1 while being 4× ahead at B=64; gen_pfa_small's split path is already 2.8× better at B=1. Round 6 draws unknown batches, so this is the one hole that can zero a whole cell. Build it in whichever entry moves first; the other adopts. |
| **15** | **The ~54 rsp ops per x-pencil**, or concede | Both engines are within 3% of each other and ~15% off the port-utilization floor (~3.8 µs). gen_pfa_small has *written the rule* — register-explicit pencils win only when 2L + module temps ≤ ~32 live, so 10/12 fit and 15 (30 live) does not. The remaining lever is a hand-scheduled two-block stage 2. Expected value 1–2%; if it does not pay, declare the cell finished. |
| **20** | **L2 residency: the bounce buffer, and nothing else** | S+C = 1.7 MiB against a 1.25 MiB L2 is the only structural fact left. gen_pfa_small has already killed prefetch (+14%), split-layout c-interleave (+4%), site-interleave (+40%) and consumption-order c (+4–10%). One untested idea remains (bulk-copy plane x+1's c during plane x's sweep). Their own estimate is that it loses. Measure it in a quiet window and close the cell either way. |
| **25** | **Nothing. Spend the window elsewhere.** | gen_powp leads by 87% over the next engine and 3.38× over the best library, and moved 0.3% in a round. It is the healthiest cell on the board. |
| **27** | **The x-pass two-column pipelining item** | gen_powp's own diagnosis: 27 is its weakest ratio (3.26× vs 3.38× at 25), the item is three rounds old and untried, and it is blocked on the same ~32-live-vector register cliff that gen_pfa_small just characterized at L=15. One honest same-core A/B, one afternoon, then close it. |
| **31** | **Stop optimizing 31. Port the effort to generic primes.** | Both engines are at their models: gen_rader's measured 85 µs *is* its issue-port budget (~250k cycles), gen_dense_prime is at ~1.55× its FMA-port floor and bought 3% in three rounds. Map placement at L=31 is now closed at every granularity (five measured negatives, the fifth being the first load-side one). The real target is **true generalized Rader for 3∤h primes (7, 19, 43, 67, 79, 103, 127)** — a ~45% FMA cut at 127, where the engine is only at MKL parity. That is round-6 insurance and the biggest arithmetic lever left anywhere in the panel. |
| **32** | **The z+y skew phase** | The x-pass has now taken three consecutive rounds of wins (fusion, DSB residency, c-transpose); the z/y phase has had one. The concrete shape is folding the zpair's 16 `vshuff64x2` re-forms into the following y-pass loads, which read the rows the z-pairs just wrote. |
| **40** | **Find the regression first, then the fused (5,8) codelet** | L=40 is the only cell where *two* entries lost ground against a flat library baseline (gen_race +5.4%, gen_layout +4.3%) and neither documented a cause; gen_layout did not measure the cell at all. That is a day's work and it is owed. Then the structural item: L=40's picked tree `c5(d8)` contains no fusable node (40 > the n≤25 fusion cap), so a fused (5,8) with ~20 planned spills is the one shape that would bring the round's big generic-layer win to this cell. |
| **50** | **A 2-volume-pair schedule at B=4** | The only graded cell volume-major did not move, and it is a dead heat between two engines that both suspect the same fix (4 MB working set, still L3-safe, or SoA-style 2-lane packing at B=4). Nobody has measured either. |
| **100** | **A PMU session before any more candidates** | Three separate records now ask for the same thing and none has run it: attribute the residual to **port 5 versus DRAM** before proposing another chain family. This round's −19% came from a store-traffic insight (§5) and this round's *only* 3.2% regression came from a tile pick made on timing alone; both argue that L=100 is being tuned blind. It is also the worst cell on the board (1.62× vs 4.05× at L=12), so the headroom is real. Second priority: apply gen_layout's NT-store recipe (full aligned lines, ≥4 MiB per component, sfence per pass) to gen_pfa_large's and gen_powp's map stores — the −19% has a measured basis now and nobody outside gen_layout has adopted it. |
| **panel-wide** | **Reconcile the race layer with same-core A/B** | §4.1: the cold scoring race disagreed with held-lease verdicts at L=12, 40 and 100, costing 2–3% at each, and coincides with two of three regressions. The racer is drift-immune *within* a session but not *across* sessions. Until that is fixed, every pick it persists is a 2–3% liability, and round 6 scores the assembled trunk through exactly this layer. |

---

## 7. Curation decision

Applying `docs/CURATION.md` in its stated order. I note first that `impl_4/` is
complete provenance and every strategy record is tracked regardless of promotion —
so declining an entry does not lose its lesson, and `exemplars/gen_r4/` should stay
a *reading list*, not an archive.

**Rule 1 — fastest correct entry per geometry, always.** Eleven cells collapse to
six distinct class entries: `gen_pfa_small` (10, 15, 20), `gen_batchlane` (12),
`gen_powp` (25, 27), `gen_rader` (31), `gen_pow2` (32), `gen_pfa_large` (40, 50,
100). All six promote.

**Rule 2 — structurally different runner-up when close.**
- `gen_dense_prime` at L=31 (1.42×) is outside the "~20%" guideline, but this is
  the crossover the brief explicitly commissioned (*"gen_rader — 31 (crossover
  fight vs dense)"*), the alternative is genuinely structural (conjugate-pair
  dense fold vs Rader cyclic convolution), and **both sides have now independently
  measured the gap to be arithmetic and permanent**. That is exactly the "next
  panel needs to see the alternative actually written down" case. **Promote.**
- The other runners-up (`gen_batchlane` at 10/15/20, `gen_pfa_small` at 12,
  `gen_powp` at 50/100) are inside 0.02–3.1% and already promoted under rule 1.

**Rule 3 — instructive failures whose record documents the number that killed
them.** Three qualify, and each documents a different class of mistake:
- `gen_bluestein` — the shipped gate that was never tested against its own policy
  (§3.4): a one-character error worth 21% at L=40 and 9% at L=50 for a full round.
  It is also the only any-L existence guarantee (correct at all eleven cells and
  across a 2..128 sweep), which is what round 6's surprise draw rests on.
  **Promote.**
- `gen_race` — carries two of the round's three regressions *and* the round's most
  consequential library change. Its record predicted the falsification of its own
  window-heat defence and named the cells to check. That is precisely the record
  the next panel needs in front of it. **Promote.**
- `gen_layout` — the −19% NT-store result with the knob-knockout attribution that
  *corrected its own r3 diagnosis* is the single most instructive measurement of
  the round, and it is what moved §4.3. Its unremarked L=40 regression is a mark
  against the entry, not against the record's value. **Promote.**

**Rule 4 — anything that beat a library.** Applied with judgement, because at
L=31 every gen entry beats every library by virtue of the size (§1); I do not let
L=31 alone carry a promotion.
- `gen_planner` beat the best library at **all eleven** cells and delivered the
  round's largest algorithmic gain (fused register-resident CT codelets, −12 to
  −27% at seven cells). It is also the round-6 trunk. **Promote.**
- `gen_layout` beat the best library at 20, 25 and 27 on its own merits (in
  addition to 31). Already promoted.
- `gen_race` beat the best library at 10, 12, 20, 25, 27, 50 and 100. Already
  promoted.

**Declined: `gen_twiddle`.** Not because its work is bad — its residency-gate
negative (+0.6…+3.8% at L=27, 7/7) is real evidence in §5, and its prefetch
rejection is a clean cross-engine falsification. But: it is a library layer scored
by adoption and gained **no new adopter this round** (gen_bluestein's r3 adoption
merely stands); its demo entry is 4–9× off the winner at every cell and beats a
library only at L=31, where everything does; and its one structural r4 change is
`gen_bluestein`'s k-plane block size, borrowed — so promoting it would promote a
near-duplicate of an already-promoted entry, which `CURATION.md` forbids by name.
Its lessons survive intact in `strategies/gen_twiddle.md`, which is tracked.

**Condition on promotion:** the signed-overflow UB in the shared map tail (§3.3)
must be fixed in both `gen_dense_prime.c:1512` and `gen_rader.c:1023` before
either is put in front of the next panel as exemplar material. It is a one-line
change in each, it propagated once already by adoption, and exemplars are read to
be copied.

---

PROMOTE: gen_pfa_small gen_batchlane gen_powp gen_pfa_large gen_pow2 gen_rader gen_dense_prime gen_planner gen_race gen_layout gen_bluestein
