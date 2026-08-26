# VERDICT — round gen_r9

Monitor's judgement on the measured standings in `results/gen_r9/leaderboard.txt`.

---

## 0. Two corrections to the brief, before anything else

The monitor prompt for this round (`results/gen_r9/monitor_prompt.txt`) is stale boilerplate
carried over from the earlier `bench/geom` campaign. Two of its premises are false for this
round, and both change what the numbers mean:

**(a) The geometry set is wrong.** The brief asks for headlines at **L = 6, 8, 17, 36**.
None of those has ever been measured in this campaign. Every `gen_r*` round, including this
one, sweeps **L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100** (verified across all nine
round leaderboards). `docs/LITERATURE.md` is likewise written for 6/8/17/36 — it is the
*geom* campaign's corpus. I report the eleven geometries that were actually measured.

**(b) The scoring machine is wrong, and so is the correction factor.** The brief says
"Xeon Gold 5218, Cascade Lake" with "downclocked AVX-512 and 1 MB L2", and that "MKL alone
spans 2.9x" between dev and score hosts. `environment.txt` says:

```
host: a81n2.lqcd.mit   slurm_job: 438854
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: ... avx512_bitalg avx512ifma avx512_vbmi avx512_vnni avx512_vpopcntdq ...
```

That is **Ice Lake-SP**, not Cascade Lake — VBMI/IFMA/VNNI/BITALG/VPOPCNTDQ are all absent
on CLX. Ice Lake-SP has **1.25 MB L2 per core** and mild AVX-512 licence downclocking, not
CLX's 1 MB and severe downclocking. A Cascade Lake 5218 *does* exist in this fleet — `p52n1`,
used for the `xarch_clx_r6` cross-arch round — but it was not used this round and is not the
scoring node.

The real dev→score correction, measured from the library baselines that ran on all three
machines (SPR = wallaby Gold 6448Y, the dev host; ICL = a81n2, the scoring host; CLX = p52n1):

| backend | SPR→ICL | SPR→CLX | ICL→CLX |
|---|---|---|---|
| `mkl_dfti` | 1.21–1.47× | 1.51–2.13× | 1.20–1.44× |
| `ducc0_c2c` | 1.36–1.68× | 1.77–2.20× | 1.30–1.44× |
| `fftw3_patient` | 1.23–1.46× | 1.51–1.91× | 1.14–1.41× |
| `baseline_matrix` | 1.17–1.22× | 1.46–1.61× | 1.20–1.32× |

So **the factor to apply to implementers' wallaby numbers this round is ≈1.2–1.5×, not 2.9×.**
The 2.9× figure would only be reachable SPR→CLX at the very worst cell, and CLX was not scored.
I use the measured 1.2–1.5× band in §4.

**One node-comparability caveat.** r8 was scored on `a80n0`, r9 on `a81n2` — different physical
silicon, same CPU model (Gold 6326). `baseline_matrix`, which is fixed code, reads within
±0.3% across the two rounds at every cell, so the two nodes are calibrated to well under 1%.
Cross-round deltas below are therefore real, not node drift.

---

## 1. Headline per geometry

**Batched / non-batched:** the brief asks for both cases per geometry. The sweep measures
exactly **one** batch per L — L=10…50 batched at B=64…4, L=100 non-batched at B=1. **No
geometry has both cases measured**, so no geometry can have both reported. That is a gap in
the sweep, not in the entries: every batch-lane entry publishes B=1 numbers in its own record
(e.g. `gen_batchlane` reports B=1 m=64 chains at 10.31 / 17.61 / 39.86 / 117.5 µs at
10/12/15/20 against its batched 1.12 / 1.92 / 4.34 / 12.67 — a 6–9× per-transform gap that
the scored suite never sees). **Adding a B=1 column to the sweep is the cheapest structural
improvement available to the harness**, and is a prerequisite for scoring any of the B=1 work
five different records now list as their top unattacked item.

Fastest correct panel entry vs. best library baseline, all `ok` on correctness:

| L | case | fastest panel entry | µs/xform | GF/s | best library | µs/xform | **speedup** |
|---|---|---|---|---|---|---|---|
| 10 | batched B=64 | `gen_batchlane` | **1.121** | 44.43 | `fftw3_custom_soa` | 4.535 | **4.05×** |
| 12 | batched B=64 | `gen_race` → *batchlane* | **1.913** | 48.57 | `mkl_dfti` | 7.755 | **4.05×** |
| 15 | batched B=32 | `gen_race` → *batchlane* | **4.326** | 45.72 | `fftw3_custom_soa` | 15.551 | **3.59×** |
| 20 | batched B=32 | `gen_race` → *batchlane* | **12.549** | 41.33 | `fftw3_custom_soa` | 43.087 | **3.43×** |
| 25 | batched B=16 | `gen_powp` | **31.054** | 35.05 | `fftw3_custom_soa` | 77.748 | **2.50×** |
| 27 | batched B=16 | `gen_powp` | **43.607** | 32.19 | `fftw3_custom_soa` | 97.315 | **2.23×** |
| 31 | batched B=16 | `gen_race` → *rader* | **84.694** | 26.14 | `fftw3_custom_soa` | 216.444 | **2.56×** |
| 32 | batched B=8 | `gen_race` → *pow2* | **53.999** | 45.51 | `mkl_dfti` | 172.105 | **3.19×** |
| 40 | batched B=8 | `gen_race` → *pfa_large* | **159.721** | 31.99 | `mkl2026_dfti` | 406.933 | **2.55×** |
| 50 | batched B=4 | `gen_powp` | **417.400** | 25.35 | `mkl_dfti` | 951.936 | **2.28×** |
| 100 | **non-batched** B=1 | `gen_race` → *pfa_large* | **4516.288** | 22.07 | `mkl_dfti` | 7816.123 | **1.73×** |

### `gen_race` is a wrapper — read its wins accordingly

`gen_race` tops six cells, but it is the meta-layer: it compiles the other class entries as
`.so` at plan time, races them, and ships the winner by vtable forwarding. The banked verdicts
in `results/wisdom_a81n2.json` say exactly what it ran:

```
gen_race/eng9/L10,12,15,20 -> gen_batchlane   (tie=1, margins  1.3 / -0.4 / 0.9 / 1.9 %)
gen_race/eng9/L25,L27      -> gen_powp        (margins 48.1 / 56.2 %)
gen_race/eng9/L31          -> gen_rader       (margin  27.5 %)
gen_race/eng9/L32          -> gen_pow2        (margin  84.5 %)
gen_race/eng9/L40, L100    -> gen_pfa_large   (margins 35.1 / 2.4 %)
gen_race/eng9/L50          -> gen_powp        (tie=1,  margin  0.3 %)
```

So the *algorithms* winning the board are `gen_batchlane` (10/12/15/20), `gen_powp` (25/27/50),
`gen_rader` (31), `gen_pow2` (32) and `gen_pfa_large` (40/100). The wrapper's contribution is
picking correctly and cheaply — which this round it did, and last round it did not (§2).

### The top of L=10/12/15/20 is a statistical tie, not a ranking

At these four cells `gen_race`, `gen_batchlane` and `gen_pfa_small` finish within **0.2–3.7%**
of each other while carrying **run spreads of up to 14.1%** (`gen_race` 14.0% at L=10 and 14.1%
at L=15; `gen_batchlane` 13.8% at L=12; `gen_pfa_small` 13.7% at L=10). They are also *the same
structure* — `gen_pfa_small`'s own record calls its 10/15/20 engines "converged copies" of
`gen_batchlane`'s, and `gen_race` is forwarding to `gen_batchlane` at all four. **The rank
order at 10/12/15/20 is window luck and must not be reported as a result.** Three consecutive
rounds have shuffled these three entries at these cells with no code justification.

---

## 2. What changed since gen_r8, per geometry

Deltas are r8 (`a80n0`) → r9 (`a81n2`), calibrated by `baseline_matrix` to <1%.

### Improvements

| L | entry | r8 | r9 | Δ | cause (from the record) |
|---|---|---|---|---|---|
| 40 | `gen_race` | 241.810 | **159.721** | **−33.9%** | prefetch-all `.so` compiles: r8's bounded 30 s poll gave up on pfa_large's 83 s gcc and banked the partial verdict (setup 30.653 s → 0.565 s) |
| 25 | `gen_powp` | 41.025 | **31.054** | **−24.3%** | wisdom tag `chain6`→`chain7` retired the poisoned r8 pick |
| 25 | `gen_race` | 40.083 | **31.486** | **−21.4%** | same two fixes, downstream |
| 10 | `gen_twiddle` | 6.460 | **5.319** | **−17.7%** | packed axis-1 across plane seams (20→13 rec groups) + ax-2 w=4 tail |
| 12 | `gen_twiddle` | 9.125 | **7.553** | **−17.2%** | packed axis-1 (24→18) |
| 20 | `gen_twiddle` | 37.016 | **32.839** | **−11.3%** | packed axis-1 (60→50) |
| 15 | `gen_twiddle` | 19.759 | **17.979** | **−9.0%** | w=7 scalar tails deleted + jt=3 ax-2 tails |
| 25 | `gen_twiddle` | 80.971 | **74.963** | **−7.4%** | packed axis-1 (100→79) |
| 31 | `gen_bluestein` | 292.564 | **272.594** | **−6.8%** | undocumented (see §3) |
| 27 | `gen_twiddle` | 128.318 | **121.330** | **−5.4%** | masked gf ax-1 tails |
| 31 | `gen_dense_prime` | 113.507 | **109.955** | **−3.1%** | 24-accumulator split C/S z-phase GEMM, 0.75 → 0.417 loads/FMA |
| 20 | `gen_race` | 12.964 | **12.549** | −3.2% | (inside the tie band, §1) |
| 10 | `gen_batchlane` | 1.148 | **1.121** | −2.4% | factor-swapped map x-pencils (`BL_SWAP10` + div tail) |
| 32 | `gen_pow2` | 56.455 | **55.161** | −2.3% | z-codelet re-form via 256-bit extract-to-memory stores (−16 p5 uops/zpair) |

`gen_twiddle` and `gen_race` are the round. `gen_twiddle` delivered a bit-identical
−5 to −18% at six of eleven cells; `gen_race` + `gen_powp` between them recovered two cells
that had been *lost to their own search machinery*, not to slow code.

### Regressions

Three, all real, in descending order of importance:

**(i) `gen_powp` at L=100: 4596.649 → 4807.365, +4.6%.** The cause is in the banked wisdom
and it is the most important finding of the round — see §3, finding **A**.

**(ii) `gen_pfa_large` at L=50: 416.640 → 437.059, +4.9%** (spreads 2.1% / 1.2%, so outside
noise). `gen_pfa_large`'s r9 change was its noise gate: an upset may displace the rank-0 prior
only with margin > `max(spread, 6%)` *and* a fresh-evidence confirmation; otherwise it reverts.
Its own record notes that "quiet-floor gaps among the leading families are 1-3%" at exactly
these cells. The banked entry is `gen_pfa_large/chain7/L50/B4 -> l50-ip1.ch` with
`us=0, margin=0, widx=-1` — a *forced rank-0 default*, not a measured verdict. **Hypothesis
(unconfirmed, and the L=50 item for r10): the 6% floor is now conservative enough to lock out
a genuine 1–3% winner.** That is the cost side of the determinism ledger, and nobody has
priced it yet. The trade is still strongly positive — a 4.9% conservatism cost against the
24–34% recovered at L=25/40 — but it should be measured rather than assumed.

**(iii) `gen_dense_prime` at L=12: 7.715 → 8.147, +5.6%** (spreads 2.6% / 1.9%). Its r9 record
documents no L=12 change and claims bit-identity at all 13 dev cells. `gen_dense_prime` is 4×
off the pace at L=12 and is not a contender there, so this is a curiosity rather than a
problem — but it is unexplained by the record, and code that claims bit-identity should not
move 5.6%. Most likely code-layout tax from the new `gdp31_zuv3` inlining (the failure mode
`gen_twiddle` explicitly tested for with `nm -S` and `gen_pfa_large` with a function-size diff;
`gen_dense_prime` did neither).

`gen_layout` at L=20 (+3.4%) and `fftw3_guru` at L=32 (+4.6%) are inside their own spreads.
`gen_twiddle` at L=100 reads +2.2% on the board while its record claims −0.3%; its control was
measured in a different window (its r8 ctl read 7430/7447 where the r8 board read 7317) — this
is window drift on the campaign's noisiest cell, not a regression.

### Cells that did not move at all

**L=12 has not moved in two rounds** (1.911 → 1.913). **L=27 has not moved** (43.357 → 43.607).
**L=31 has not moved** (84.745 → 84.694 at the top; `gen_rader` itself +0.5%, i.e. flat).
`gen_rader` shipped nothing to the scored cell this round by design — the PMU audit named L=31
"the champion signature" (IPC 2.15, p0+p5 = 1.60/2.0) and it kept the cell bit-frozen, spending
the round on p=103, which is not a scored geometry.

---

## 3. Failures, misses and things that must not survive — adversarial pass

### Correctness: nothing failed, and nothing is close to the line

`check.log` is **734 lines, 734 `PASS`, zero non-`PASS`**. Every leaderboard row reads `ok`.
The margins are not marginal: single-call `rel_l2` runs 2.2e-16 – 5.2e-16 against a 1e-12
tolerance (≈2000× margin), two-step gates 7.6e-16 – 1.3e-15 against 3e-14 (≈25×), and graded
chains 2.0e-14 – 1.6e-13 against 1e-10 (≈600×). **No entry is riding the tolerance edge, so
there is no fast-wrong-answer candidate on this board.**

The worst chain drift belongs to `gen_bluestein` at every cell (1.6e-13 at L=10, ~2× the next
entry) — expected for a chirp-Z convolution and still 600× inside tolerance. Noted, not
faulted.

### A. `gen_powp` at L=100 replayed, in r9, the exact failure r9 was supposed to close

This is the finding of the round. The banked verdict on the scoring host reads:

```
gen_powp/chain7/L100/B1#080a458a -> {winner: 'l100-ipk1', tie: 1, us: 4699.06, margin: -0.01596}
```

Compare the r8 poison that `gen_powp`'s own record spent the round removing, still on disk in
`results/wisdom_a80n0.json`:

```
gen_powp/chain6/L25/B16#6bb92654 -> {winner: 'l25-ip0',   tie: 1, us: 46.987,  margin: -0.00974}
```

**Identical signature: `tie=1`, margin ≈ −1%, a coin flip banked as a verdict.** In r8 it cost
L=25 +32%. In r9 it cost L=100 +4.6% — `gen_powp` ran 4807.365 where its own dev-host
determinism test picked `l100-ipp1` on 4 of 5 cold cycles and never once picked `ipk1`
(`ipk1` is the variant its records name as the *CLX/SPR* cross-host upset — the wrong-host
arm). Setup was 0.003 s, i.e. a warm wisdom hit: the timing runs replayed a pick made once,
in one window, by a coin flip.

The mechanism the fix missed is stated plainly in `gen_race`'s own r9 record: *"Tight verdicts
**and ties** store as always."* The provisional-expiry machinery (`~q<pct>@<time>`,
`GENPWP_NQHORIZON`) only fires for verdicts classified *noisy*. **A tie is classified tight and
banked unconditionally — and a tie is precisely the case where the pick is window luck.**
The r9 gate closed the noisy-verdict hole and left the tie hole open.

This affects more than L=100. Six banked entries on `a81n2` carry `tie=1`:
`gen_powp/L100` (−1.60%), `gen_powp/L50` (−0.36%), `gen_race/chain9/L40` (−1.53%),
`gen_race/chain9/L50` (+1.97%), `gen_race/eng9/L10/L12/L15/L20/L50` (+1.3 / −0.4 / +0.9 /
+1.9 / +0.3%). **Every one of these will warm-hit in r10 and replay r9's coin flip** — which
is also the cleanest available explanation for why L=10/12/15/20 keep reshuffling between
rounds (§1).

*Fix, and it is one line of policy:* store a tie as provisional, not tight. Ties are the one
class of verdict that carries no evidence at all.

### B. Two entries shipped code with no strategy record — `gen_planner` and `gen_bluestein`

`docs/CURATION.md` is explicit: *"Do not promote … entries whose strategy record is missing —
the record is what makes the code useful later."*

| entry | source change r8→r9 | last heading in `strategies/*.md` | `git status` |
|---|---|---|---|
| `gen_planner` | `gen_planner.c` 3778 → 3958 lines, **250 changed lines** | `## Round gen_r8` | record **not modified** |
| `gen_bluestein` | `gen_bluestein.c` 2786 → 2905 lines, **123 changed lines** | `## Round gen_r8` | record **not modified** |

Both exited 0 (`agents/exits.txt`). `agents/gen_bluestein.log` is **1 byte — empty**: the
entry produced no account of its session at all, yet shipped 123 lines of change and moved
−4% to −7% at six cells. `agents/gen_planner.log` shows the work was done and locally
validated (a noise-gated `create()` race; `clflushopt` c-line custody for L=100) and that the
agent was still waiting on the node when the session ended — the record was never written.

Neither entry wins a cell, so nothing is lost from the standings. But `gen_planner`'s
`clflushopt` c-line custody at L=100 is a direct attack on the round's weakest cell (1.73× over
MKL) and is now **shipped, unmeasured on the node, and undocumented** — precisely the state
`CURATION.md` exists to prevent. **Both are disqualified from promotion**, and `gen_planner`'s
r9 work should be re-documented in r10 before it is trusted.

### C. `fftw3_custom_soa` failed to create at L=50 B=4 — a library baseline is missing a cell

`timing.err` lines 181–184:

```
fftw3_custom_soa: fft3d_create failed for L=50 batch=4   (×4 — the check run and all 3 timing runs)
```

No `t_`/`o_` JSON was produced, so L=50 has no `fftw3_custom_soa` row. This is a *hard create
failure*, categorically different from the declared refusals below, and it is unreported in
`failures.txt` (which only logs non-zero exits from runs that started). At L=25/27/31
`fftw3_custom_soa` is the **best library on the board**, so its absence at L=50 means the L=50
speedup of 2.28× vs `mkl_dfti` is against a possibly-weaker reference than the other cells.
Worth one build session in r10; it does not change any panel ranking.

### D. `baseline_matrix` timed out at L=100 — expected, third round running

`failures.txt`: `baseline_matrix L=100 B=1 run={1,2,3} exited 124`. Exit 124 is the harness
timeout. `baseline_matrix` is O(L⁴)/volume/axis and at L=100 it would take ~30 s per transform;
r8 recorded the identical three lines. **Not a defect** — the harness floor simply cannot reach
L=100 in the time budget. It should be excluded from the L=100 case explicitly rather than
timing out every round.

### E. Every other absence is a declared refusal, verified

All 57 other missing cells are `fft3d_supports()` returning false, which `driver.c:118`
documents as *"Not an error: implementations are allowed to specialize for one size."* I
checked the full list in `timing.err` against r8: **no entry lost coverage between rounds**.
`gen_batchlane`/`gen_pfa_small` (10/12/15/20), `gen_dense_prime` (10/12/15/20/31),
`gen_powp` (25/27/50/100), `gen_pfa_large` (40/50/100), `gen_pow2` (32), `gen_rader` (31) —
all identical to r8.

### F. Two build warnings, unchanged from r8, still not fixed

`build_errors.txt` (warnings only, both entries built):

```
impl/gen_dense_prime.c:2246  iteration 1152921504606846976 invokes undefined behavior
impl/gen_rader.c:2385        iteration 1152921504606846976 invokes undefined behavior
```

Both are the same `for (; i < npts; ++i) { zp[2*i] + cp[2*i] }` map tail, and both fired
identically in r8 (at lines 1913 / 2226). GCC has proven the loop trip count can exceed
`PTRDIFF_MAX/2` and is entitled to delete the loop. It does not currently, and correctness
passes — but this is **live undefined behaviour in the two entries that own L=31**, carried
across two rounds by entries that otherwise verify bit-identity to the last digit. Cast the
index or bound `npts`; it is a one-line fix in each file.

### G. Minor: `gen_race`'s dlopened copy of `gen_pow2` runs 2.1% faster than the harness build

At L=32 `gen_race` reads 53.999 while forwarding to `gen_pow2` (wisdom margin 84.5% over self),
and the harness's own build of `gen_pow2` reads 55.161 — a 2.1% gap between two builds of the
same source, against spreads of 1.3% and 0.4%. At the other forwarded cells the gap is
−0.6% (L=31), −0.3% (L=40), +1.4% (L=25), i.e. noise. L=32 is the outlier. Most likely a
compile-flag difference between `gen_race`'s plan-time `.so` build and the Makefile's. One
`diff` of the two command lines settles it, and if the `.so` flags are better, `gen_pow2`
should adopt them.

---

## 4. Claimed vs. measured

The honest headline: **the implementers were unusually accurate this round, and where they were
wrong they were wrong in the direction they predicted.** Round conditions forced this — both
Ice Lake nodes were held by an external user's ~2-day jobs for most of the session, so seven
of twelve entries developed on wallaby (SPR) with no node window at all, and said so up front.

Applying the measured SPR→ICL band from §0 (**1.2–1.5×**, *not* the brief's 2.9×):

| entry | claimed (dev host) | measured (ICL board) | verdict |
|---|---|---|---|
| `gen_twiddle` | −16..−20 / −15..−18 / −9..−11 / −10 / −6..−9 / −5.7 / −2.3 / −2.1% at 10/12/15/20/25/27/31/50 | −17.7 / −17.2 / −9.0 / −11.3 / −7.4 / −5.4 / −1.6 / −1.2% | **exact at all eight.** It had node time on `a81n2` at 23:13, the same node as the score run |
| `gen_batchlane` | −2.2 / −1.0 / −1.4..1.6% at 10/15/20 | −2.4 / −0.8 / −0.8% | **exact.** Also raced on `a81n2` (core 2, held lease) |
| `gen_pow2` | "−2%ish" at L=32, extrapolated from a −1.5% SPR reading | −2.3% | **exact**, and the extrapolation direction was right |
| `gen_dense_prime` | −3.3% at L=31 on SPR, "may be LARGER on ICX" | −3.1% | **within half a point.** The "larger on ICX" guess did not pay, but was flagged as unproven |
| `gen_pfa_small` | −0.8 / −0.7 / −1.0% at 10/15/20 (from llvm-mca ICL model) | −0.9 / −0.6 / −0.2% | right at 10/15, **nothing at 20** |
| `gen_powp` | L=25 recovery to ~31 µs | 31.054 | **exact** |
| `gen_race` | L=40 and L=25 recovered by prefetch-all | 159.721 / 31.486 | **exact** |
| `gen_rader` | nothing claimed at L=31 (cell bit-frozen) | +0.5%, flat | consistent |
| `gen_pfa_large` | nothing claimed at 40/50/100 (tune()-only round) | 40 flat, **50 +4.9%**, 100 −1.0% | the L=50 regression was **not** predicted (§2-ii) |

### Absolute levels, where dev numbers can be checked against the board

These are the cases where machine difference *is* the explanation, and it is a 1.2–1.5× effect:

| cell | dev host reading | ICL board | ratio | MKL's ratio at the same cell |
|---|---|---|---|---|
| `gen_pfa_small` L=10 | 0.914–0.921 (SPR) | 1.142 | 1.25× | 1.27× |
| `gen_pfa_small` L=15 | 3.542–3.546 (SPR) | 4.388 | 1.24× | 1.21× |
| `gen_pfa_small` L=20 | 8.634–8.665 (SPR) | 13.019 | **1.50×** | 1.37× |
| `gen_pow2` L=32 | 40.72–41.44 (SPR) | 55.161 | 1.34× | 1.24× |
| `gen_dense_prime` L=31 | 83.43–84.08 (SPR) | 109.955 | 1.31× | 1.47× |

Every dev-to-score gap sits inside the band the library baselines set, so **no implementer's
absolute claim is anomalous**. The one entry that scales *worse* than MKL is `gen_pfa_small` at
L=20 (1.50× vs MKL's 1.37×) — L=20's 7.81 MiB working set against Ice Lake's 1.25 MB L2 is the
plausible cause, and it is the one place where the brief's "smaller L2 on the score host"
intuition is actually visible in the data.

### One case worth singling out: a dev-host reading correctly overruled

`gen_pfa_small` measured its lifted DFT5 as a **loss** at L=10 on SPR (+0.6%) and shipped it
**on anyway**, on the strength of an llvm-mca `icelake-server` model (−1.0%) and
`gen_batchlane`'s r7 node measurement of the same codelet. The board says −0.9%. That is the
correct methodology — *model for the scoring machine, do not trust the dev host's sign* — and
it is worth propagating. `gen_rader` used the same discipline in the other direction: it built
2-wide `RP3_WINO` at p=103, measured +35% on SPR, calibrated the known dev-host anti-pairing
bias at ~3–11 points on that kernel family, concluded 35% could not be bias, and **defaulted
the feature OFF** rather than shipping a wallaby-tuned regression.

---

## 5. Which LITERATURE §4 open question moved

**Caveat, per §0:** `docs/LITERATURE.md` is the *geom* campaign's corpus, written for
L = 6/8/17/36. Its §4 questions are microarchitectural and transfer; its size-specific claims
do not. Three moved.

### §4.6 "Model versus search for the instruction schedule" — moved decisively, and reframed

§4.6 concludes the schedule is "the primary thing to search" and that a near-exhaustive search
at our sizes "costs minutes. Do it." Nine rounds of this panel took that advice. **r9 supplies
the half §4.6 never states: a search is worth exactly what its evidence discipline is worth,
and a bad pick costs more than any kernel win the campaign has produced.**

The receipts are on the board, and they are large:

* L=25: **41.025 → 31.054, −24.3%**, with *zero* kernel arithmetic changed. The r8 number was
  a banked coin flip (`tie=1, margin −0.97%`); r9 retired the tag and re-raced.
* L=40: **241.810 → 159.721, −33.9%**, also zero kernel change. r8's bounded 30 s compile poll
  gave up on an 83 s gcc and banked the partial verdict; r9 prefetches all class compiles on
  the first cold create.

Four entries (`gen_race`, `gen_pfa_large`, `gen_powp`, `gen_planner`) independently implemented
noise-gated search with 5-cycle determinism proofs. `gen_pfa_large` recorded the round's most
transferable negative, with a receipt: its first gate design (margin > `max(spread, 3%)`, no
floor, no confirmation) flipped `ip0/ipp1/ip0/ip1/ip1` across five cold cycles. Its conclusion:
**"a noise gate keyed only to in-window spread cannot deliver determinism; it needs a margin
floor calibrated to between-window drift and a confirmation on evidence the challenger has not
seen."** That is a genuine addition to §4.6 — the literature discusses *what* to search and
never *how to know the search's answer is real*.

And §3-A is the same question's unfinished business: the r9 gate still banks ties, so the
mechanism that cost 32% at L=25 in r8 cost 4.6% at L=100 in r9.

### §4.8 item 6 "no primary AVX-512 measurement for Ice Lake-SP or later server parts" — closed for port structure

§4.8.6 says there is "no primary measurement in the corpus for Ice Lake-SP or later *server*
parts" and instructs "measure it on the node." `gen_pfa_large` built `portcal3.c`, a
TSC-calibrated microbenchmark of 8 independent zmm FMA chains plus K independent ymm
side-streams, and measured on SPR:

```
ZMM8 baseline                4.00 cyc/iter
8×ZMM + K ymm FMA   K=2 5.02  K=4 6.00  K=8 8.00  K=12 10.00   ⇒ exactly (8+K)/2
pure ymm FMA (P=8)           4.00                              ⇒ 2/cycle, no third port
```

**256-bit FP steals 512-bit FMA slots 1:1 whenever zmm is in flight.** `gen_batchlane` reached
the same conclusion for ICL-SP from the microarchitecture (port 1's FP pipe *is* the lower
256-bit half of port 0's fused 512-bit unit). This kills the PMU audit's avenue 4 ("port 1
idles, side-work could co-issue nearly free") as a throughput lever on both graded
architectures — and it retroactively explains `gen_layout`'s result, whose new ymm `gl_map4`
tail had to be plan-gated to `L <= 16` because it only pays where the tail dominates.

A 4-lane ymm path remains valuable for **coverage** (B=4 at L=50, B%8 remainder lanes), which
is a different claim and still open.

### §4.1 "How many registers does a batch-vectorised codelet actually need?" — moved, and inverted

§4.1 asks how much spill traffic the batch-vectorised codelets generate and prescribes the
cheap check: *"count stack traffic in the generated assembly … before believing any timing."*
`gen_batchlane` ran exactly that experiment across four sizes and the answer is that **the
prescribed check does not predict the outcome**:

| L | `%rsp` touches, ship → swapped | measured |
|---|---|---|
| 12 | **35 → 16** (biggest cut) | **LOSES +3.5..4.7%** |
| 10 | 12 → 12 (no change) | wins −2.2% |
| 15 | 18 → **22** (worse) | wins −1.0% |
| 20 | 40 → **44** (worse) | wins −1.4..1.6% |

The size with the largest spill reduction lost; two sizes that *added* spill traffic won. The
record's conclusion — *"the win is dependency-shape, not spill count … do not trust rsp counts
as a proxy for time on this engine"* — directly contradicts §4.1's methodology note. §4.1's
underlying trade (a few spills versus zero shuffles) still stands; its proposed *instrument*
does not.

`gen_twiddle` added an adjacent boundary from the same round, which the corpus does not state
anywhere: **"the divider unit prices `vsqrtpd`/`vdivpd` per OP, not per useful lane"** — 8 zmm
sqrt+div for 8 scalar sites loses ~3× to 8 scalar sqrt/div. Its rule: *pack divider work
densely or leave it scalar; shuffles and loads vectorise at any occupancy, sqrt/div does not.*
That is worth writing into the corpus.

### Not moved

§4.3 (axis fusion, and specifically the **L2↔DRAM** case re-opened by §08 §1.9 — "the largest
untried structural move on the board") was not attempted by anyone. `gen_planner` shipped
`clflushopt` c-line custody at L=100, which is adjacent, but it is undocumented and unmeasured
on the node (§3-B). §4.4, §4.5 and §4.7 were untouched.

---

## 6. Highest-value attack for the next round, per geometry

**Cross-cutting first — two things that gate everything else:**

1. **Store ties as provisional, not tight** (§3-A). One line of policy in `gr_pick` and
   `gen_powp`'s `tune()`. Six banked ties on `a81n2` will otherwise replay r9's coin flips into
   r10, and this mechanism has now cost the board 32% (r8 L=25) and 4.6% (r9 L=100).
2. **Add a B=1 column to the sweep** (§1). Every batch-lane entry runs 6–9× worse per transform
   at B=1, five records name the B=1 engine as their top unattacked item, and the scored suite
   cannot see any of it.

| L | the single highest-value attack |
|---|---|
| **10, 12, 15, 20** | **Make the cell measurable before optimising it further.** The top three are within 0.2–3.7% at spreads up to 14.1%, are the same code, and have reshuffled for three rounds. Raise the sample count and pin the protocol at these four cells so a 1% effect is detectable at all — otherwise every future round's work here is unscoreable. Then attack the one term that is left: the graded map's `sqrt`/`div`, using `gen_twiddle`'s r9 rule (pack divider work densely) plus `gen_batchlane`'s finding that the div-vs-`rcp14` verdict is a property of the surrounding codelet and must be re-raced per form. |
| **25, 27** | **Arithmetic, not scheduling.** These cells run at 35.05 and 32.19 GF/s against the board's 45.51 at L=32 — a 25–30% arithmetic deficit that both `gen_powp`'s and `gen_planner`'s records attribute to real op counts. Build the **Winograd / real-factor DFT9 and DFT25 modules** that sidestep twiddles entirely. Also: verify that r9's L=25 recovery *replays* from banked wisdom rather than being re-won by luck — the whole point of the r9 design is cross-round determinism, and it has never been demonstrated across a round boundary. |
| **31** | **The divider floor.** `gen_rader` is at the port-floor signature the PMU audit calls "what done looks like" (IPC 2.15, p0+p5 = 1.60/2.0) and further scheduling will not pay. `gen_dense_prime` puts the remaining divider floor at **~21 µs of the 85** — 25% of the cell, and the largest non-FFT term anywhere on the board. Attack it with `gen_twiddle`'s density rule. Second, cheap: fix the undefined-behaviour map tail in both entries (§3-F). |
| **32** | **Resolve the 2.1% build gap** (§3-G): `gen_race`'s dlopened `gen_pow2` beats the harness's own build of the same source by more than either spread. If the `.so` flags are better, `gen_pow2` adopts them for free. Then confirm r9's `GP2_ZST` on the node with port-5 dispatch counters — the record predicts p5 uops per step drop ~8K, and that is the counter-signature the round could not take. |
| **40** | **The L2↔DRAM tiled-fusion experiment** — LITERATURE §4.3's re-opened case and, in its own words, "the largest untried structural move on the board." L=40's 15.62 MiB working set against a 1.25 MB L2 is squarely in the regime where the measured bandwidth gap is 7× rather than the 2.6× every previous panel experiment fused across. Tile the batch so a tile fits L2, run all three axes inside the tile. |
| **50** | **Price the noise gate's conservatism** (§2-ii): find out whether the +4.9% regression is the 6% upset floor locking out a genuine 1–3% winner, and if so give the gate a low-margin path with confirmation instead of an outright revert. Then **land the 4-lane batch-lane path**: `gen_layout` built and bit-verified `gl_pack4/gl_map4/gl_tr4x4` this round precisely because L=50 at B=4 is the one cell where the 8-lane form cannot run, and it was never dogfooded there. |
| **100** | **The weakest cell on the board — 1.73× over MKL against 2.2–4.1× everywhere else.** It is traffic-bound (30.52 MiB working set, 24 MB L3) and it is the same §4.3 L2↔DRAM question as L=40. Start by measuring `gen_planner`'s already-shipped `clflushopt` c-line custody on the node and writing its record (§3-B) — the work exists, unmeasured. Then the tiled all-axes-in-tile construction. Also fix the `baseline_matrix` timeout by excluding it from this cell (§3-D). |

---

## 7. What to keep

Applying `docs/CURATION.md`'s four grounds, in order.

### Promoted

| entry | ground | evidence |
|---|---|---|
| **`gen_race`** | 1 (tops 6 cells) + 4 | The round's largest deltas: L=40 −33.9%, L=25 −21.4%, both from the prefetch-all compile fix, both with zero kernel change. Also shipped the noise-gate library three other entries adopted, and found + fixed a silent data-loss bug in the wisdom parser (a compacted-JSON file made `drop_prefix` re-emit nothing). |
| **`gen_batchlane`** | 1 (L=10; the engine behind `gen_race` at 12/15/20) + 4 | 4.05× over the best library at L=10 and L=12. Its factor-swap result and the **inverted §4.1 finding** (biggest spill cut = the only loss) is the round's most transferable negative. |
| **`gen_powp`** | 1 (L=25, 27, 50) + 4 | Recovered L=25 by −24.3% through search discipline alone. Its record is the campaign's clearest write-up of the wisdom-poisoning failure mode — and, read against §3-A, of that fix's remaining hole. |
| **`gen_rader`** | 1 (the engine behind `gen_race` at L=31) + 4 | 2.56× over the best library; the PMU audit's "champion signature". Kept the cell bit-frozen on purpose and **defaulted its own new feature OFF** on a +35% dev-host reading it could not attribute to host bias — exemplary discipline. |
| **`gen_pfa_large`** | 1 (the engine behind `gen_race` at L=40 and L=100) + 4 | Holds 2.55× at L=40 and the only entry within reach at L=100. Contributed the round's most reusable methodological negative (a spread-only noise gate is not deterministic, with a 5-cycle receipt) and `portcal3`, which closed LITERATURE §4.8.6 for port structure. |
| **`gen_pow2`** | 1 (the engine behind `gen_race` at L=32) + 4 | Best GF/s on the board (45.51). Its r9 store-path re-form is a clean, bit-identical, correctly-predicted −2.3%, derived from documented port behaviour rather than search. |
| **`gen_dense_prime`** | 3 (instructive) + near-miss on 2 | The structurally different L=31 alternative (folded dense GEMM vs. Rader) at 1.30×. Its record **documents the number that kills it** and adds a second measured negative (the same 24-accumulator lever *rejected* at generic L: +4–7% at 17, +5% at 29, shipped default-off behind a knob). Exactly what ground 3 exists for. |
| **`gen_twiddle`** | 4 (beats the best library at 12, 20, 25, 40, 50, 100) | The round's second-largest delivery: bit-identical −5% to −18% at six cells from lane composition alone. Also the library layer `gen_bluestein` adopts, and the source of the divider-density rule that §6 hands to three geometries. |
| **`gen_layout`** | 4 (beats the best library at 20, 31, 32, 40, 50) | The arena/placement layer that `gen_batchlane`, `gen_rader` and `gen_pfa_small` all `#include`; promoting the adopters without it leaves the exemplar set incomplete as a reading list. Delivered the monitor-requested 4-lane family, bit-equality-asserted against the 8-lane form at `create()`, and honestly plan-gated it to `L <= 16` by measurement. |

### Not promoted, with reasons

* **`gen_pfa_small`** — near-duplicate. Its own record calls its 10/15/20 engines "converged
  copies" of `gen_batchlane`'s, and its single r9 idea (the φ-lifted DFT5) was borrowed whole
  from `gen_batchlane` r7. `CURATION.md` excludes near-duplicates of an already-promoted entry.
  This is a close call — it carries a genuinely broader generic engine (53 sizes vs
  `gen_batchlane`'s 12) — but at every *scored* cell it is the same structure, and it never
  leads one. Its one distinct contribution (correctly overruling a dev-host sign using the ICL
  model, §4) is recorded here instead.
* **`gen_planner`** — **disqualified: no r9 strategy record** despite 250 changed lines
  (§3-B). Ranks 4th at every cell and never leads one. Re-document in r10.
* **`gen_bluestein`** — **disqualified: no r9 strategy record** despite 123 changed lines, and
  an empty agent log (§3-B). Also fails ground 4: it never beats the best library at any of the
  eleven cells. Its −4% to −7% improvements this round are consequently unattributable.

---

**PROMOTE:** `gen_race` `gen_batchlane` `gen_powp` `gen_rader` `gen_pfa_large` `gen_pow2`
`gen_dense_prime` `gen_twiddle` `gen_layout`
