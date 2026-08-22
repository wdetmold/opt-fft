# VERDICT — panel round `panel_r10`

Monitor's judgement on the measurements taken on `p55n3` (Intel Xeon Gold 5218, Cascade
Lake, 2×16c, exclusive, slurm job 438527, 2026-08-22T06:18), gcc 11.4.0, `governor=powersave`.
Roster: **19 implementations plus the harness floor, all built, all ran, all correct, none
missing, none crashed, no anti-memoization trip.** Sources are in `bench/geom/impl_10/`
(`impl` → `impl_10`); all 19 differ from `impl_9` (60–722 changed lines each) and
`baseline_matrix.c` is byte-identical, as it must be.

**Four things about this round's shape before the numbers.**

1. **This is the fastest round on record.** Thirty-two panel cells improved by ≥1% on
   minima against five that regressed. `panel_r9` — the round the panel stopped guessing —
   had six gains and seven regressions. r9 bought the information; r10 spent it. The two
   largest single moves are **L64_blocked −13.2% at B=1** (the split-complex rewrite it had
   deferred for two rounds) and **L36_pfa −7.7% at B=1** (genfft's `n1_9` FMA DAG, which
   r9 §6 ordered into all three L=36 arms).
2. **The monitor owes the panel a correction, and it reverses my predecessor's.** `panel_r8`
   §3(a) flagged MKL as having regressed 10–29% at the large-working-set cells; `panel_r9`
   §3(e) concluded "two rounds now agree on the higher value, so **r7 was the anomaly** and
   the margins quoted in §1 are the honest ones." **That is wrong.** r10 reproduces r7 to
   within 0.7% at all four affected cells, in both MKL builds, while every other backend at
   those same cells moved <3% across all four rounds. **r8 and r9 were the anomalous rounds.**
   Four margins in each of the last two verdicts were inflated by 17–30%. §3(e) below.
3. **Two entries put an unvalidated bit class behind a leaderboard number.** r9 flagged the
   timed≠checked exposure and recorded that it "happened to be benign" — the minimum run and
   the checked run coincided. This round it did not: **L8_radix8 at B=2048 and L64_blocked at
   B=8** both report a minimum produced by a configuration that r10's correctness check never
   saw, and in both cases the two configurations are demonstrably different bit classes.
   §3(a). Neither changes a standing, but the exposure is no longer hypothetical.
4. **L=17's batched cells, which r9 was ready to close as bandwidth-bound, are re-opened by
   a measurement — and the entry that made it had bet against its own result.**
   L17_matrixsimd's `sbw` probe times the node's own read-burst-then-write-burst floor for
   its exact traffic mix: **13.05–13.31 µs per volume against a 21.6 µs cell.** The
   "two independent implementations tie at 10.9 GB/s, so it must be an external wall"
   reading is false; the wall is at 17.9 GB/s. §5.

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are µs per transform. "min" is the leaderboard number (minimum over three independent
processes). "med" is the median of the three per-run medians. "Best library" is whichever of
FFTW ×3 / MKL 2022 / MKL 2026 / ducc0 was fastest **in that exact cell, this round** — read
§3(e) before comparing any margin here to r8's or r9's. **Every panel entry in every cell
beat every library**, so CURATION rule 4 selects all nineteen and cannot discriminate (§7).

### L = 6 (volume 216) — L6_pfa takes the DRAM cell it predicted it would only tie

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L6_pfa 0.2070** (med 0.2150) | L6_unrolled 0.2086 (med 0.2170) | mkl_dfti 0.371 | **1.79×** |
| B=64 (0.42 MiB) | **L6_pfa** (med 0.2214; min 0.2150) | L6_unrolled min 0.2142, med 0.2221 | mkl_dfti 0.401 | **1.87×** |
| B=4096 (27 MiB) | **L6_unrolled 0.3826** (med 0.3904) | L6_pfa 0.3926 (med 0.3988) | mkl_dfti 0.549 | **1.43×** |
| B=32768 (216 MiB) | **L6_pfa 0.5540** (med 0.5627) | L6_unrolled 0.5627 (med 0.5687) | mkl_dfti 0.685 | **1.24×** |

**Two cells changed hands and both are non-overlapping.** B=4096 was a genuine tie in r9;
L6_unrolled took it by −3.3% on minima (medians 0.3852–0.3947 against pfa's 0.3949–0.4002 —
disjoint by 0.05%). B=32768 was L6_unrolled's in r9; **L6_pfa took it by −3.3%** (minima
disjoint by 1.6%; medians overlap by 0.0003). B=1 stays L6_pfa's on disjoint medians but the
margin collapsed from r9's 9.1% to **0.9%** — L6_unrolled improved 3.7% on minima there.
B=64 inverts between minima and medians for the third round running: the leaderboard hands it
to L6_unrolled by 0.4%, the medians to L6_pfa by 0.3%, and the distributions are disjoint both
times. Read on distributions the geometry is **L6_pfa 3 – L6_unrolled 1**, the same score as
r9 with a different cell in each column.

### L = 8 (volume 512) — B=1 is a three-way pileup; B=2048 came back

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L8_batchsimd 0.5510** (med 0.5580) | L8_fusedaxes 0.5560 (med 0.5560) — tie | mkl_dfti 0.652 | **1.18×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.5747** (med 0.5767) | batchsimd 0.5819 (med 0.5957) | mkl_dfti 0.704 | **1.22×** |
| B=2048 (32 MiB) | **L8_fusedaxes 0.9244** min / **L8_batchsimd** on medians (0.9472 vs 0.9558) | — | mkl2026_dfti 1.329 | **1.44×** |
| B=16384 (256 MiB) | **L8_fusedaxes 1.2245** (med 1.2272) | batchsimd 1.2432 (med 1.2596) | mkl2026_dfti 1.769 | **1.44×** |

**B=1 is a dead tie for the second round and is now genuinely three-way**: batchsimd has the
lowest minimum on the board (0.5510) and fusedaxes the lowest medians (0.5560 ×2), with
L8_radix8 0.5760 only 4.5% behind. The driver reports L=8 B=1 to three decimals, which is
0.2% of the cell — the reader should treat 0.551 vs 0.556 as unresolved.

**B=64 and B=16384 are L8_fusedaxes' on disjoint distributions.** **B=2048 splits**:
fusedaxes owns the minimum, batchsimd the medians (0.9472 against 0.9558, disjoint). That is
the cell r9 §6 named as the round's single L=8 question, and it is answered — see §2.

### L = 17 (volume 4913) — L17_matrixsimd takes three cells; the batched story changed

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 14.995** (med 15.077) | L17_winograd 16.451, L17_rader 16.515 | fftw3_patient 81.718 | **5.45×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 16.735** (med 16.882) | winograd 17.900 | fftw3_measure 81.860 | **4.89×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.001** (med 21.378) | winograd 21.526 (med 21.594) | fftw3_measure 83.462 | **3.97×** |
| B=2048 (307 MiB) | matrixsimd 21.621 min / **L17_winograd** on medians (21.723 vs 21.940) | rader 24.608 | fftw3_measure 84.070 | **3.89×** |

B=1 and B=8 are matrixsimd's on disjoint distributions (9.7% and 5.5%). B=256 is matrixsimd's
by 1.0% on medians with overlapping tails. **B=2048 inverts from r9's dead tie to L17_winograd
on disjoint medians** (21.680–21.735 against 21.795–22.004) while the minima stay tied to
0.1%. The separation is 0.8% — below the panel's own documented ±2% recompile noise floor —
so the honest reading is **a tie that happens to be statistically clean inside one round**,
not a change of ownership. L=17 remains the panel's largest margin over the state of the art
anywhere on the board.

One flag: **all three L=17 arms regressed at B=8** (matrixsimd +1.4%, winograd +2.2%, rader
+0.4%) while every library at that cell moved <0.2%. Three files, three different changes,
one cell, same sign — that is a property of the 1.20 MiB cell, not of any file.

### L = 36 (volume 46656) — the round's biggest geometry-wide move

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L36_pfa 113.128 / L36_mixedradix 113.423 — tie** | pencilfused 119.588 | mkl_dfti 163.022 | **1.44×** |
| B=4 (5.70 MiB) | **L36_mixedradix 124.274** (med 124.795) | pencilfused 124.621, pfa 124.221 min but med 129.395 | mkl_dfti 175.447 | **1.41×** |
| B=32 (45.6 MiB) | **L36_mixedradix 162.137** (med 163.581) | pencilfused 164.561, pfa 165.182 | mkl_dfti 218.069 | **1.34×** |
| B=256 (365 MiB) | **L36_mixedradix 179.219** (med 179.633) | pfa 181.643 (med 182.251) | mkl_dfti 244.927 | **1.37×** |

**Every L=36 cell improved 1.8–7.7%, in all three arms, from one shared change** (§2).
B=1 is a clean two-way tie (medians 113.59–116.52 against 113.75–117.02, fully overlapping).
**L36_mixedradix owns the other three** — B=4 on the tightest distribution in the geometry
(124.71–124.98), B=32 by 2.3% on medians, B=256 by 1.4% with the tails touching. That is
3–0–1, against r9's zero separated cells. The change of state is real; the individual margins
are still thin.

### The four wave-2 geometries

| L | case | fastest panel entry | best library | margin |
|---|---|---|---|---|
| 13 | B=1 | **L13_direct 5.692** (rader 5.982) | mkl2026 7.654 | 1.34× |
| 13 | B=16 | **L13_direct 5.983** (rader 6.740) | mkl2026 7.610 | 1.27× |
| 13 | B=512 | **L13_direct 8.006** (rader 8.577) | mkl2026 9.002 | **1.12× — thinnest cell on the board** |
| 23 | B=1 | **L23_rader 47.469** (matrixsimd 47.733 — tie) | fftw3_estimate 260.97 | 5.50× |
| 23 | B=4 | **L23_matrixsimd 49.216** (rader 49.678 — tie) | fftw3_patient 261.55 | 5.31× |
| 23 | B=128 | **L23_rader 64.793** (matrixsimd 64.874 — tie) | fftw3_estimate 265.21 | 4.09× |
| 45 | B=1 | **L45_mixedradix 309.153** (pfa 310.439 — tie) | mkl_dfti 606.26 | 1.96× |
| 45 | B=2 | **L45_pfa 318.434** (mixedradix 318.598 — tie) | mkl_dfti 607.22 | 1.91× |
| 45 | B=16 | **L45_pfa 395.760** (mixedradix 425.326) | mkl_dfti 675.35 | 1.71× |
| 64 | B=1 | **L64_radix8 949.904** (blocked 952.936 — tie) | mkl_dfti 1191.44 | 1.25× |
| 64 | B=2 | **L64_radix8 1026.618** (blocked 1032.288) | mkl_dfti 1237.49 | 1.21× |
| 64 | B=8 | **L64_radix8 1262.872** (blocked 1311.471) | mkl_dfti 1934.54 | **1.53× (was quoted 2.01× in r9 — see §3e)** |

L=13 and L=45 B=16 are non-overlapping in every cell. All three L=23 cells overlap, as they
have for three rounds. **L=64 B=1 is a tie for the first time in six rounds** (0.3% on
minima; radix8 ahead 1.1% on medians) — L64_blocked closed a 15% deficit in one move.

### Distance from each geometry's own port floor, B=1, at the measured 2.89 GHz

Floors as published by the entries themselves. **The L=36 floor fell this round** (83 → 78 µs)
because the arithmetic did.

| L | floor | r8 | r9 | **r10** | move |
|---|---|---|---|---|---|
| 6 | 486 cy = 0.1682 µs | 1.29× | 1.23× | **1.23×** | flat; nothing left (§6) |
| 8 | 1248 cy = 0.4318 µs | 1.28× | 1.28× | **1.28×** | flat, seventh round |
| 13 | 4.7 µs | 1.22× | 1.22× | **1.21×** | flat |
| 17 | 33 374 cy = 11.548 µs | 1.31× | 1.29× | **1.30×** | flat (B=1 closed in r9) |
| 23 | 41.9 µs | 1.14× | 1.14× | **1.13×** | tightest on the board; closed |
| 36 | **78 µs** (was ~83 — the floor fell) | 1.45× | 1.46× | **1.45×** | the time fell 6% and the floor fell 6% |
| 45 | 188 µs | — | 1.68× | **1.64×** | toward the floor, first time in three rounds |
| 64 | ~550 µs | 1.76× | 1.73× | **1.73×** | still the board's worst |

---

## 2. What changed since `panel_r9`, per geometry — and what regressed

### Cell-level moves on minima, r9 → r10, all panel entries, ≥1%

**Thirty-two gains, five regressions.** The full gain list is long; the regressions are:

| regression | | |
|---|---|---|
| L8_radix8 B=64 | 0.597 → 0.617 | **+3.4%** |
| L17_winograd B=8 | 17.508 → 17.900 | **+2.2%** |
| L45_mixedradix B=16 | 417.913 → 425.326 | **+1.8%** (second consecutive round) |
| L17_matrixsimd B=8 | 16.512 → 16.735 | +1.4% |
| L64_radix8 B=8 | 1249.923 → 1262.872 | +1.0% |

Three of the five are at the two cells §1 already flagged as cell-level rather than
file-level (L=17 B=8, L=64 B=8). The two that are real and attributable are L8_radix8 B=64
and L45_mixedradix B=16, and both are named in §6.

### L = 36 — the round's headline: genfft's `n1_9` DAG, and port 0 does bind here

r9 §6 ordered one thing at this geometry: "transcribe genfft's `n1_9` FMA DAG into all three
L=36 arms — 44 → 40 FMA-port vector ops per DFT9 … transcribe, do not derive." **All three
did, from L45_pfa's r9 record, and all three passed numpy on the first build** — the point of
transcribing rather than deriving, against L36_pfa's r1 record of three failed hand
derivations of the same result. Accuracy improved in every arm (rel_l2 3.95e-16 → 3.58e-16;
the leaderboard's `3.6e-16` fingerprint is the marker that the `n1_9` path is what was
scored).

The counted change is **248 → 232 FMA-port vector ops per 36-point line (−6.45%)**, +8
port-5 shuffles, verified in the object code by two arms independently (`phase2_pf0_pw4`
arith = 232 exactly; `halfplane_v4` = 232, `passA_plane_v4` = 464). What it bought:

| entry | B=1 r9 → r10 (min) | B=1 medians | delta |
|---|---|---|---|
| L36_pfa | 122.576 → **113.128** | 123.52 → 114.38 | **−7.7% / −7.4%** |
| L36_mixedradix | 120.478 → **113.423** | 121.72 → 114.73 | **−5.9% / −5.7%** |
| L36_pencilfused | 123.657 → **119.588** | — → 120.52 | **−3.3%** |

**A 6.45% cut in port-0 vector ops bought 5.7–7.7% of wall time at B=1. Port 0 binds at
L=36.** That is the opposite of the L=45 result r9 recorded (−5.5% bought +1.2%) and it is
the cleanest arithmetic result the panel has produced. The floor fell with the time
(83 → 78 µs), so the *ratio* is unchanged at 1.45× — the residual is undiminished, but it is
now 35 µs of a smaller total rather than 37 of a larger one, and the entries were right that
this was the one unfalsified lever.

Every arm's pre-registered branch fired, and two of the three cells came in **better than the
band the entry itself predicted**:

* **L36_mixedradix's three-way probe fork was the discriminator.** It shipped the old
  Cooley–Tukey 3×3 DFT9 as a probe-only twin and published the in-plan A/B:
  **`n19/ct9` = 141.4/145.6, 141.0/147.2, 143.1/148.4 → ratio 0.958–0.971**, inside its
  pre-registered 0.94–0.98 "the FMA cut prices" band. Its cell prediction was 114–119;
  it landed **113.4**.
* **L36_pfa** predicted 117–121 with pick `pw4 inplace pf=0 nc=12`; got **113.1** with that
  pick 3/3. Its phase probe moved as predicted but harder: **p1 = 93.0 → 86.3–88.4 µs**
  (predicted −2 to −3.5, measured −4.6 to −6.7).
* **L36_pencilfused** predicted 118.5–123 with "≤121.5 = the port-0 cut prices at L=36";
  got **119.588**, and its disassembly audit is the exact one (232 FP-port instructions per
  line body, "every one of the old vmulpd is gone").

**And L36_pfa's pick lottery — the board's worst measurement in r9 — is closed at the
tuner, but not at the run.** It restricted the small-batch candidate list to the twelve
shapes the node has actually picked in five rounds and raised the timing rounds 5 → 9.
Result: **identical pick strings 3/3 in every cell, and r9's 39.5%-spread B=4 outlier is
gone.** But B=4 still reads **124.22 / 129.26 / 129.39 (4.2%) at an identical `nc=12` pick
string**, in two clusters. By the entry's own pre-registration that means "the lottery was
never in the tuner at all but in the node run itself — also worth knowing." It is worth
knowing, and it is why L36_pfa's B=4 number still cannot be used (§3b).

**One methodological finding from this geometry is worth more than a percent.**
L36_mixedradix's first build offered `ct9` as an ordinary tuner candidate and tryout returned
`NOT REPEATABLE`: two processes picked *different DFT9 forms*, which are **different bit
classes** (fingerprints 3.960e-16 vs 3.577e-16). That is r9 §3(a)'s timed≠checked exposure
realized as a hard failure inside the entry's own harness. The fix — an `installable` flag,
so cross-class candidates are timed and published but can never be picked — yields a rule
the whole panel should adopt: **a tuner pool must be one bit class; cross-class comparisons
ride the description string, never the pick.** §3(a) shows what happens where it was not.

### L = 8 — the r9 attribution was right, and the fix is regime-local

r9 §6's single L=8 item: "A/B out L8_batchsimd's allocation changes at the streaming cells …
If the regression is the alias fix, the fix should be regime-gated and §4.5 gains its second
refinement." L8_batchsimd did exactly that — B=1 keeps the r9 page-aligned/`SI+520` layout
in a dedicated runner, batch>1 restores the r8 layout byte-for-byte, and
`fft3d_description()` publishes which regime a number was taken under
(`alloc=r9(a4096,si520)` vs `alloc=r8(a64,si512)`, correct in all twelve JSONs).

**Every branch fired:**

| cell | r9 median | prediction | **r10 median** | verdict |
|---|---|---|---|---|
| B=1 | 0.5588 | 0.552–0.560, `fixed, no tuner` 3/3 | **0.5580** (min 0.5510) | fired; pick 3/3 |
| B=64 | 0.6056 | ≤0.595 if allocation, ~0.605 if noise | **0.5957** (min 0.5819) | fired at the boundary |
| B=2048 | 0.9800 | 0.91–0.95 if allocation, ≥0.97 if noise | **0.9472** (min 0.9323) | **fired: −3.3%** |
| B=16384 | 1.2599 | 1.23–1.26 | **1.2596** | fired |

**So r9's attribution was correct: the global alias fix bought ~1% at B=1 and cost ~3.3% at
B=2048, and gating it by residency recovers both.** §4.5 gains the refinement r9 asked for.

**And the same fix transferred to neither sibling.** L8_fusedaxes ported it as raced
`fusedSI` twins and predicted "0.542–0.548, pick fusedSI ≥2/3" if it transferred; the node's
own arena ranked fusedSI ahead of fused in 2 of 3 runs (0.555 vs 0.558; 0.576 vs 0.579) and
behind in the third — **and the tuner picked fusedSI in none of them.** The cell read
**0.5560, above its band and 0.5% worse than r9.** L8_radix8 hardwired its de-aliased
`1f520` at B=1 behind a pre-registered fork ("≤0.565 = the sr/si 4K alias was the 1f driver
tax; ~0.578–0.583 = the fix bought nothing in my file") and measured **0.5760** — its second
branch, with the node's own arena reading `1f520` and `1f` dead even (0.569/0.567,
0.574/0.572, 0.574/0.572). **The reachable in-scratch alias class is 1-for-3 across three
implementations of the same geometry.** It is not a portable mechanism; it is a property of a
particular scratch layout.

**The L=6 association-order mechanism does not transfer to L=8.** r9 §6 asked the panel to
propagate it. L8_radix8 built the probe (`1f520j`: the 16 output joins re-issued as FMA/FNMA
with a broadcast 1.0, `cmp`-verified bit-identical) and published the node's own arena
reading: **1f520j = 0.573/0.576/0.577 against 1f520's 0.569/0.574/0.574 — +0.5 to +0.7%**,
against a pre-registered "≤ −2% and r11 adopts it." Null on the node, at the machine the
mechanism was discovered on. The entry named the reason in advance: at L=6 the winning joins
fed *stores*; here they feed the ZUNTRI shuffle network. **L8_batchsimd's standing monitor
ask for a forced `-DL8_JOIN_FMA` run is therefore already answered by its neighbour and
should not be run** — this is the third round it has been asked for, and the answer arrived
from one file over at zero monitor cost.

L8_radix8's B=64 regressed +3.4% on minima (0.597 → 0.617) at an identical pick — but its
*median* improved 2.7% (0.6369 → 0.6195) and its spread collapsed 11.3% → 2.5%. Its own r9
finding was that the B=64 spread is the driver-buffer allocation lottery, not pick noise; the
r10 numbers are consistent with the lottery having drawn differently, not with a mechanism.

### L = 17 — the batched cells are not at the bandwidth wall, and the entry that proved it bet the other way

r9 closed B=1 panel-wide and said of the batched cells: "this needs a new idea or an honest
'closed'." L17_matrixsimd built the instrument that decides, pre-registered three branches,
and stated its honest prior: **branch (a), cp ≈ 21 ± 1 → the cells are bandwidth-closed.**

Node readings of `sbw[rd/wr/cp/s17]`, µs per 78.6 KiB volume-equivalent, three processes:

```
B=2048:  rd 4.81 / 4.61 / 5.10    wr 7.95 / 8.22 / 7.56
         cp 13.31 / 13.20 / 13.05  s17 3.98 / 3.72 / 4.16       cell = 21.62
B=256:   rd 3.28 / 3.29 / 3.30     cp 11.32 / 11.28 / 11.24     cell = 21.00
```

* **`cp` = 13.05–13.31 µs against a 21.62 µs cell.** `cp` is the exec's own
  read-burst-then-write-burst phase alternation with the compute deleted — 236 KB of
  compulsory traffic at **17.9 GB/s**, not the 10.9 GB/s the cell runs at. Branch (a)
  required cp ≥ 20.5. **It read 13.2. The streaming cells have ~39% headroom over the node's
  own floor for their own traffic, and the "two implementations tie, so it must be an
  external wall" argument is refuted by direct measurement.**
* **`s17/rd` = 0.81–0.83.** The X pass's 17-interleaved-row-stream read shape — the
  mechanism the round's staged twins were built to fix, and which reads 1.28–1.31 *adverse*
  on Sapphire Rapids — is **faster** than a sequential read of the same bytes on the node.
  Branch (b) required ≥1.3.
* **Branch (c) fired**: "the headroom is write-side/overlap, staged declines, and next
  round's item is the write stream." The staged twins were **declined 3/3 in both batched
  cells** and the cells are flat (B=256 21.001 vs r9's 21.073; B=2048 21.621 vs 21.718).

`b1dec` reproduced r9 exactly (`kyz` = 10.18–10.52 against the 7.83 µs port floor; `pt=0`
picked 4/4), so B=1 stays closed and its +0.9% is the noise floor.

**L17_winograd's `i4` fork closed negative.** Its split-free pass-1 load path
(8 xmm+`vinsertf128` pairs replacing exactly the 8 splitting loads, 576 cold splits per
volume deleted, `cmp`-verified bit-identical to `h4`) was offered in all three batched cells
and **picked in 1 of 9 runs** (B=256 run 2). Its own fork: "h4 kept = the node has now priced
the mechanism in both L=17 structures and this geometry is closed in every cell." Its cells
moved −0.3% (B=2048) and +0.1% (B=256). Note the honest reading of the B=2048 median gain
that gives it the cell in §1: nothing changed on the picked path, so it is drift, not `i4`.
Its B=8 pick also moved h4 → h8 and the cell regressed 2.2% — worth one look before r11.

**L17_rader's traffic-deletion bet died 6/6, and its record says so in advance.** It
diagnosed 30–50 KB/volume of partial-line refetch waste from its 17 concurrent
16-B-misaligned k-row output streams and built two staged dense-flush shapes (`st`, and
`stp` with the flush paced under the next volume's plane phase; both bit-identical). Its
pre-registered fork: bet fires → 25.0 toward 22.5–23.5 and a staged pick; bet dies → picks
stay `xl 512t dy pf=0 pfw=0` and cells stay ~24.6/25.0. **Picks stayed `xl 512t dy pf=0
pfw=0` 3/3 in both cells; cells read 23.874 and 24.608.** By its own words: "the
partial-line-waste theory joins spreading in the falsified list … the honest position is the
r9 VERDICT's: this slot is the donor. Say so." Said. That is the fifth consecutive round of
node-declined batched mechanisms from this structure. Its ~3% and ~1.5% improvements are
real but it remains 10–14% behind in all four cells.

### L = 6 — the codelet mattered in DRAM, which nobody predicted

L6_pfa shipped one change: **a pure reorder of the candidate table**, giving every
radix-2-first `_d2` twin incumbency over its radix-3-first parent, on the reasoning that a
~1.4% node-measured advantage cannot clear a 1.5% takeover margin from a trailing slot. `.text`
byte-identical to r9, so the recompile-noise class cannot fire. **Four picks predicted, four
picks delivered 3/3** (`fused_d2` at B=1, `fused_pf_xa_d2` at B=64 and B=4096,
`fused_pfw_xa_d2` at B=32768), three cells in band.

The fourth is the round's one genuinely unexplained number. **B=32768 was predicted at
0.565–0.568 with the explicit alternative "if the pick flips but the number stays 0.573, the
codelet does NOT matter in DRAM." It read 0.5540 — the pick flipped and the cell dropped
3.3%, well past the band.** So on this machine the store-feeding-FMA association order is
worth ~3% even in the fully DRAM-bound regime, where the entry's own model said it should be
invisible. That is the only L=6 number in ten rounds that beat its own prediction, and there
is no mechanism written down for it.

L6_unrolled adopted L6_pfa's zp-outer/y-inner x-pass group order as bit-identical twins,
deleted ~390 lines of node-answered dead weight (the whole AVX-512 section, the fused3 twins,
the clock ladder; 1432 → 1047 lines), and retargeted its `ab1` instrument at the new
question. **Node reading: `fx − f` = −0.6% / −3.1% / −1.2% at B=1** against a branch (i)
threshold of ≥3%; the tuner picked `fused_zp` in 2 of 3 runs; the cell's minimum entered the
predicted band (**0.2086**, from the run that read −3.1%) but its medians did not move
(0.2170/0.2170/0.2260 against r9's 0.2174/0.2257/0.2305). Its `xod` field reads +3.5% /
−3.5% / −1.6% — too noisy to use. Honest verdict: **the x-pass group order is worth 1–3% at
nvol=1 on the node, sign-stable in the same direction as L6_pfa's file has carried since
round 1, and below the cell-level noise floor.** Its own branch (iii) prescribes the next
move (give the zp twins incumbency, exactly as L6_pfa did for d2), and L6_pfa's own record
supplies the generalizable lesson: **re-derive candidate *order* from node pick strings each
round, because a node-proven sub-margin winner in a trailing slot is invisible to the tuner
by construction.**

### The wave-2 geometries

**L = 13.** L13_direct's B=1 bet — X-first replacing X-last — read its own null branch. It
pre-registered "if the cell reads ≥5.74 the flip bought nothing"; the cell read **5.692 min /
5.742–5.760 median** against r9's 5.739 / 5.759. Its `ab` field shows why the reasoning was
sound and the cell still did not move: X-first is **2.6–2.7% faster than X-last in-plan at
B=1 and 5.5% at B=16** (`xf/xl` = 0.973–0.981 and 0.945), a real effect that does not reach
the driver. It owns all three cells for the fourth round. **Its `ab` field also priced the
L=6 association twins at a third geometry and they lost**: `q/xf` = +0.5 to +0.7%,
`s/xf` = +3.7 to +3.8%, every reading. Its `-DL13_PW=0` / `-DL13_PFIN=0` ask at B=512 is
now four rounds outstanding.

L13_rader executed the rollback r9 ordered, precisely, and it worked. All three cells landed
in their pre-registered bands with the predicted pick strings (`fuse=1 um=7` at B=1;
`fuse=0 um=1 pace=1` at batch): B=1 **5.982** (band 6.0–6.1), B=16 **6.740** (band 6.6–6.8,
**−3.2%**, r8's level restored), B=512 **8.577** (band 8.5–8.8, **−5.3%**, now *below* r8's
8.809). **Both r9 regressions recovered.** It also found and corrected a real defect in its
own r9 code: the U buffer was silently 8-deepened on *both* schedules, so a naive `FUSE=0`
would have dragged 26.6 KB of hot scratch through the streaming cells rather than r8's
6.7 KB — meaning part of what r9 booked as "the fused schedule wins at batch" was a footprint
change. That is a retraction of a prior-round attribution, made by the entry against itself.

**L = 23.** The geometry is closed and this round says so with the last untried schedule
priced. L23_matrixsimd built the tail-paced pipeline (its own r8 next-item, never raced on
the node) and published `tune[pick=… inc=… tp=…]`: **tp = 62.28 / 63.29 / 63.17 against
pick = 60.90 / 59.87 / 59.79 — the tail-paced schedule is 2.3–5.7% slower in all three
processes.** Its pre-registration: "if tp lands within 4% of pick, the tail schedule joins
NT/dz/pf1/uniform-pipe as a documented streaming null and the schedule space is empty."
**The L=23 streaming schedule space is exhausted.** It also met its determinism goal — pick
== inc, `flat + pf=2 + pw=1`, 3/3 — after r9's head-slot correction.

L23_rader shipped one pure deletion (the −i rotation's `vpxor` folded into the sine tables:
**4499 wide XORs deleted per volume**, 539 zmm + 539 ymm `vpxor` → 0 in the binary,
bit-identity verified by full-output `cmp` against `impl_9` across twelve forced variants).
It predicted −1 to −2% at B=128 and flat at B=1, and honestly pre-committed that "anything
inside ±2% … should be read as flat." Measured: **B=1 −0.7%, B=4 +0.3%, B=128 −0.1%.** Flat.
Third instance of the rule that only deletions transfer, and this time not even modestly. It
also **counted the L=6 association mechanism out before building it** (+11 ops per line-group
= +3.7% on a port-0-bound kernel to buy a stall effect measured only at identical counts) —
the right call, documented with arithmetic.

One thing moved the wrong way: **L23_rader's B=128 pick flipped three ways** (`rp-t1 pf2 pw0`
/ `rp-t1 pf2 pw1` / `plain pf2 pw1`) while L23_matrixsimd held 3/3. The lottery has moved to
the file that used to be the deterministic one, and its head slot is still rp-shaped.

**L = 45.** Both arms now run the identical `n1_9` DFT9 at the identical 188 µs floor, and
the geometry supplies the round's cleanest cross-entry attribution. L45_mixedradix
transcribed the DAG and predicted B=1 ≈314 anchored on the rival's r9 delta; it landed
**309.153 (−2.6%)**, at the low edge of its own band, and it said why in advance: **its audit
shows +3 stack moves where L45_pfa's paid +23 for the same DAG.** So the same 5.5% port-0 cut
bought **+1.2% for L45_pfa in r9 and −2.6% for L45_mixedradix in r10.** The floor ratio
improved 1.68× → 1.64×, the first movement at L=45 in three rounds. L45_pfa's own r10 bet —
`pfp`, the partial-tail phase-1 groups — took **zero picks** (`pw4-ip-pf0` 3/3 at B=1,
`pw4-ip-pf3` 3/3 at B=16), which is its pre-registered "r6's dead end stands on the node too
and the idea is closed for good." Its cells still improved 1.6–1.9% and it holds B=2 and B=16
(the latter by 7.5%, disjoint).

**L45_mixedradix's B=16 regressed +1.8%, second consecutive round, now +4.6% above r7**, at
an unchanged pick, with the one-flag A/B (`-DFFT45_ODDR8`) shipped and not run. That is the
one live monitor ask at this geometry.

**L = 64.** L64_blocked delivered the split-complex rewrite it had deferred for two rounds
and it is the largest single-entry gain in the round. `st=3` — split-complex currency, SIMD
lanes = 8 adjacent z, zero shuffles in the arithmetic, shuffle bill **0.67 M → 0.33 M per
volume** — was predicted at 890–980 µs with pick `cached pf0/pf1/pf8 st3`. Measured **952.936
with `cached pf=0 st=3` 3/3: −13.2% at B=1 and −12.5% at B=2**, turning a 15% deficit into a
0.3% tie. Its own framing was honest ("parity is the realistic outcome; the point is deleting
the 13% structural deficit and letting the schedule axes fight it out") and that is exactly
what happened. B=8 is the exception: 1311.471, above its band, flat against r9, with the pick
falling back to `st=0(3-sweep)` in run 3.

L64_radix8's `scst` A/B — its store-mode twin aimed at the pass-1 split-complex store RFOs,
the residual r9 named — read **`sc0` (plain) 3/3 in every cell**. Its own criterion: "if the
node also reads plain in all three cells, the SC-RFO theory is dead by direct test on the
machine it was written for." Dead. Cell flat at 949.904 (−0.3%). It still owns all three
cells on medians and the geometry still has the board's worst floor ratio (1.73×) for the
sixth round.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed, and I checked rather than assumed.**

* `build_errors.txt` is **0 bytes**; the slurm log shows all 20 implementations and all 6
  library backends compiling on the node itself.
* `failures.txt` **does not exist**. `sweep.sh:85–87` appends to it on any exit code other
  than 0 or 3 (3 = geometry unsupported), so no entry crashed, hung, or hit the 600 s
  `timeout`.
* `timing.err` is 1392 lines and **every one of them is a benign `does not support L=n`**
  line — `grep -v` returns empty.
* `check.log` is **259 lines, 259 of them `PASS`, zero anything else**, and all 259
  `c_*.json` carry `"ok": true`. 259 is exactly the expected count: every geometry × every
  batch × every supported backend, `baseline_matrix` skipped at the five cells where
  `L³·B > 2·10⁶`. **No leaderboard row lacks a correctness check, and no check lacks a
  leaderboard row.** Worst rel_l2 anywhere is 8.4e-16 (`baseline_matrix`); the worst panel
  entry is `L64_radix8`/`L64_blocked` at 4.5e-16.
* **The anti-memoization gate is live and untripped.** `driver.c:169–188` copies the output,
  perturbs `in[0]`, re-executes, and returns exit 5 if the output does not change. No exit-5
  line anywhere, and none could hide: it would have created `failures.txt`. **No fast wrong
  answer survived, because none was offered.**
* Inputs are regenerated per cell from `--seed SEED + L*1000 + B` (`sweep.sh:67`), so no
  entry could have been tuned to a cached dataset even if the gate were absent.
* All 19 sources differ from `impl_9`; `baseline_matrix.c` is byte-identical, as the harness
  floor must be.

### (a) Timed ≠ checked: this round it bit, twice, and both times on a bit-class boundary

`sweep.sh:90–92` checks correctness on the output of the **last run only**, while the
leaderboard reports the **minimum over three**. Pick determinism improved sharply this round —
**14 of 68 panel cells have a configuration that differs across the three runs, against r9's
44 of 76** — but two of those fourteen span genuine bit classes, and in both the reported
minimum came from a run whose bits were never checked:

* **L8_radix8, B=2048.** Run 2 produced the reported **0.9838** with pick
  `avx512-3p-pfs-pfw`. Runs 1 and 3 picked `3p` and `1f-pfs-pfw` respectively; run 3 is the
  checked run. The entry's own fingerprints: `1f-pfs` = 2.268e-16, `3p-pfs` = 1.916e-16 —
  **different bit classes.** `c_L8_radix8_L8_B2048.json` reads **2.2736e-16, the `1f`
  fingerprint.** So the leaderboard number for this cell was produced by the `3p` shape,
  which nothing in `panel_r10` validated. (`3p` has passed in earlier rounds and passes
  forced on the dev host; the defect is the round's provenance, not a suspicion of wrong
  arithmetic.)
* **L64_blocked, B=8.** Run 2 produced the reported **1311.471** with pick
  `pf=9 st=3(split-sc)`; run 3, the checked run, fell back to `pf=2 st=0(3-sweep)`. `st=3`
  is a new reassociation this round and **is a different bit class** — confirmed by the
  node's own fingerprints, not by assumption: L64_blocked reads **4.5e-16 at B=1 and B=2**
  (where `st=3` was picked 3/3, so the checked run used it) and **4.2e-16 at B=8** (r9's
  `st=0` value). The B=1/B=2 numbers are validated; **the B=8 number is not.**

Neither changes a standing: L8_radix8 is third in its cell either way, and L64_radix8 owns
L=64 B=8 by 3.7% against both of L64_blocked's readings. But r9 recorded this exposure as
"one unlucky pick away" and the round drew it twice. **The structural fix is unchanged and
now overdue: check every run, or check the run that produced the reported minimum.** Until
then, the panel-side mitigation that works is the one L36_mixedradix was forced to invent
(§2): keep tuner pools to one bit class.

The other twelve differing cells are all within a verified-bit-identical class (L=6's zp
twins, L=8's fusedSI/fusedAA, L=17's deferred-Z and i4, L=23's rp/flat, L=36's ip/cs twins)
or prefetch-only (L=64's pf and pro fields), each `cmp`-verified in the corresponding record.

### (b) One headline cell still rests on a bimodal measurement

| entry | cell | three runs (µs) | spread | median reading |
|---|---|---|---|---|
| **L36_pfa** | **B=4** | 124.22 / 129.26 / 129.39 | **4.2%** | 129.40 — third, not first |

The leaderboard prints **124.221**, which is one run of three; the other two agree with each
other at 129.3. **L36_pfa's B=4 cell should be read as ~129, which puts it third**, and no
L=36 B=4 conclusion should rest on it. This is a large improvement on r9 (131.95 / 184.11 /
132.48, 39.5%) and the tuner side is genuinely fixed — the pick string is identical in all
three runs — so the residual is in the run, exactly as the entry pre-registered. Everything
else on the board sits at 0.1–2.9% except L=45 B=1 L45_mixedradix (309.15 / 311.76 / 319.77,
3.4%) and L=64 B=2 L64_blocked (1032.3 / 1051.7 / 1072.3, 3.9%).

### (c) Where medians change a headline reading

Four cells, and unusually all four cut against the leaderboard's minima:

* **L=6 B=64** flips from L6_unrolled (0.4% on minima) to **L6_pfa** (0.3% on medians),
  disjoint both ways — third round with the wrong winner printed.
* **L=8 B=2048** splits: fusedaxes owns the minimum (0.9244), **batchsimd the medians**
  (0.9472 vs 0.9558, disjoint).
* **L=17 B=2048** flips from a 0.1% minima tie to **L17_winograd** on disjoint medians
  (0.8%) — below the ±2% cross-round noise floor, so I read it as a tie (§1).
* **L=36 B=4** flips from L36_pfa to **L36_mixedradix**, because of §3(b).

### (d) The ±2% recompile-noise rule held, and the panel used it correctly

r9 concluded that ±2% at these cells is the code-layout noise floor and told the panel to
stop reading it as mechanism. **Nine entries explicitly invoked that rule this round**, three
of them to decline shipping a change on a sub-2% dev-host reading, and two (L23_rader,
L36_pencilfused) pre-committed to calling their own result flat if it landed inside it — which
L23_rader's did, and it said so. The rule is also why five of this round's regressions are
reported here as cell- or lottery-level rather than attributed: **L17_matrixsimd B=8 +1.4%,
L17_winograd B=8 +2.2% and L17_rader B=8 +0.4% moved together at one cell while its libraries
did not.** Set against that discipline, one attribution in this round *is* now firmly
established and it is r9's: the L=8 B=2048 allocation bill, confirmed by reverting it (§2).

### (e) The library baselines: r8 and r9 were the anomalous rounds, and I am correcting my predecessor

r8 §3(a) reported MKL regressing 4–27% at the large-working-set cells. r9 §3(e) concluded
"two rounds now agree on the higher value, so **r7 was the anomaly** and the margins quoted
in §1 are the honest ones." r10 says otherwise:

| cell | r7 | r8 | r9 | **r10** | r10 vs r7 |
|---|---|---|---|---|---|
| L=36 B=32 `mkl_dfti` | 219.45 | 261.30 | 261.49 | **218.07** | **−0.6%** |
| L=36 B=32 `mkl2026_dfti` | 227.65 | 269.49 | 269.97 | **224.17** | −1.5% |
| L=36 B=256 `mkl_dfti` | 246.45 | 310.16 | 309.42 | **244.93** | **−0.6%** |
| L=36 B=256 `mkl2026_dfti` | 254.64 | 317.43 | 318.23 | **252.76** | −0.7% |
| L=45 B=16 `mkl_dfti` | 679.25 | 753.49 | 753.34 | **675.35** | **−0.6%** |
| L=45 B=16 `mkl2026_dfti` | 697.94 | 770.38 | 764.86 | **693.22** | −0.7% |
| L=64 B=8 `mkl_dfti` | 1943.53 | 2477.87 | 2516.14 | **1934.54** | **−0.5%** |
| L=64 B=8 `mkl2026_dfti` | 2068.48 | 2637.19 | 2667.56 | **2073.46** | +0.2% |

Eight readings, four cells, two independent MKL builds, all back within 1.5% of r7. At those
same four cells **every other backend moved <3% across all four rounds** — `baseline_matrix`
0.1–1.1%, `ducc0` 0.2–1.0%, `fftw3_estimate` 0.1–0.7%, and the panel entries themselves
1–5%. A 19–29% excursion confined to both MKL builds, present in exactly two consecutive
rounds and absent either side of them, is an MKL-specific measurement artifact, not the
machine.

**Consequences.** The r8 and r9 verdicts' margins at those four cells were inflated by 17–30%
and should be struck: r9's "L=36 B=32 1.55×" is really ~1.30× on an r7/r10 basis, "L=36 B=256
1.68×" ~1.34×, "L=45 B=16 1.87×" ~1.71×, "L=64 B=8 2.01×" ~1.53×. The §1 margins in *this*
verdict are quoted against r10's own libraries and are the low, honest ones. Nothing in the
panel's standings depends on this — the ordering of panel entries is untouched — but the
project's claim about how far ahead of MKL it is at large working sets was overstated for two
rounds, and the direction of the error was flattering. **Any future round that sees an MKL
excursion of this size at these cells should suspect the MKL run and not conclude anything
from two rounds agreeing.**

Two smaller library notes: `fftw3_measure` remains the noisiest backend (L=45 B=16 −16.3%
with a 24.6% run spread; L=64 B=8 +15.8% in r9 and still there) and is best library in no
cell that matters. `ducc0` at L=6 B=1 moved −12.9%, continuing to prove it is unusable as a
reference at the smallest geometry; it is never the best library anywhere.

### (f) Five of the 28 cells still have no library-free floor

`sweep.sh:72–76` skips `baseline_matrix` once `L³·B > 2·10⁶`, so L=6 B=32768, L=8 B=16384,
L=17 B=2048, L=36 B=256 and L=64 B=8 have no from-scratch reference in the table. Unchanged
from r8/r9 and defensible on cost, but the largest cell at four of the eight geometries is
anchored only by `check.py`.

### (g) The clock consensus is now unanimous, and the dissenter retired its own proxy

Seven entries publish 3.89 GHz (256-bit/scalar) and 2.89 GHz (512-bit) again, unanimously.
r9's one dissenter, L8_fusedaxes, **defaulted its condemned FMA-chain proxy off and publishes
nothing rather than a wrong clock** (`-DL8_PROBE=1` restores it, labelled "known 0.84× low"),
and replaced it with the tuner's own arena table in the description string — which is what
made §2's fusedSI null readable. Correct response to a monitor finding, executed in one round.

---

## 4. Claimed numbers versus measured numbers

Wallaby (Sapphire Rapids Gold 6448Y, full-clock AVX-512, **two** 512-bit FMA units, 2 MB L2,
60 MB L3, DDR5, six-wide decode, ~512-entry ROB) against the scoring node (Cascade Lake Gold
5218, 2.89 GHz under AVX-512, **one** 512-bit FMA unit, 1 MB L2, 22 MB L3, DDR4, four-wide
decode, ~224-entry ROB).

### The translation ratio, again, and the panel has learned to stop using it

Claimed dev-host B=1 number ÷ measured node B=1 number: **1.58× (L6_pfa) to 2.34×
(L36_pencilfused)**, ordered by working set, exactly as r9 found. The brief's frame — MKL
alone spans 2.9× between these machines — is the right one. What is new is that **the panel
has largely stopped translating**. Eleven of nineteen entries anchored their predictions on a
*mechanism* and a per-geometry ratio band rather than on a dev-host delta, and those are the
predictions that landed. The five entries whose headline mechanism was wallaby-invisible by
construction (L6_pfa's d2 ordering, L8's SI de-alias ×3, L36's `n1_9` ×3, L64_blocked's
`st=3`, L17_matrixsimd's staged twins) all said so in advance and shipped node-only bets. Four
of those five were right.

### Where a claimed number is far from measured, and whether the machine explains it

**Attributable to the machine difference — and these are the round's results:**

* **The `n1_9` FMA DAG at L=36: wallaby-flat by construction, −5.7 to −7.7% on the node.**
  All three arms measured parity or slightly adverse on wallaby (L36_mixedradix's forced
  pairs read *ct9 1% faster*; L36_pfa and L36_pencilfused read flat) and all three predicted
  it, with the mechanism stated: **wallaby has two 512-bit FMA units, so a port-0 cut has
  nothing to buy there; the node has one.** L36_pfa's arithmetic is the clearest statement of
  it — per line, old 124 half-FMAs + 49 shuffles = 173 port-5 ops, new 116 + 57 = 173, "the
  DAG leaves port 5 unchanged and cuts only port 0, which wallaby has two of." The node paid
  the full model. This is the machine difference working as designed and correctly
  pre-priced by three independent files.
* **L64_blocked's `st=3`: wallaby NT-loving and prefetch-indifferent, node −13.2%.** Its
  wallaby table honestly showed `nt pf0 st3` winning there and `cached pf0 st3` only pulled
  ahead by hysteresis; the node picked `cached pf0 st3` 3/3 and paid 13%. The mechanism —
  halving a 0.67 M-shuffle-per-volume bill on a one-512-bit-FMA part where 512-bit shuffles
  and FMAs contend — is a node property.
* **The L=8 in-scratch de-alias: wallaby cannot resolve it, third round running.** Three
  entries ran it independently (L8_batchsimd: one slow-state pair; L8_fusedaxes: five
  fast-state alternating pairs, dead even to 1 ns; L8_radix8: in-tuner 0.6549/0.6551 vs
  0.6555/0.6559). All three said the node decides. The node then answered **yes for one file
  and no for two** — which is a real result about the mechanism's locality, not about the
  machines.
* **L17_matrixsimd's `s17/rd`: 1.28–1.31 on wallaby, 0.81–0.83 on the node — a sign
  inversion.** The 17-interleaved-stream read shape that costs 28–31% on Sapphire Rapids'
  prefetchers is *cheaper* than a sequential read on Cascade Lake. This is the single largest
  cross-machine inversion the panel has measured and it is why the staged twins, which lost
  13–14% on wallaby, were also declined on the node: **the mechanism they attack does not
  exist there.** The entry built the discriminator that says so before the bet could cost a
  round, which is the pattern to copy.

**Not attributable to the machine — genuine mechanism-transfer failures, in cost order:**

* **L17_rader's staged dense flush: −22 to −27% on wallaby (adverse, expected), declined
  6/6 on the node.** The bet was that the node's 12 fill buffers and 2 load ports make its
  17 concurrent misaligned output streams pay a partial-line refetch tax worth 30–50 KB per
  volume. The node's tuner never took a staged shape in either batched cell. Fifth
  consecutive round; the entry's own record calls the consequence.
* **L8_fusedaxes' fusedSI: predicted 0.542–0.548 and an outright B=1 win, measured 0.5560
  and never picked.** The node's own arena ranked fusedSI ahead in 2 of 3 runs and the tuner
  still declined it inside hysteresis. Note what this is *not*: it is not an arena-inverts-
  driver failure (the arena's ranking was ~0.5%, below the 1% margin by design). It is the
  mechanism failing to reach the size the sibling measured.
* **L13_direct's X-first flip: 2.6–5.5% in-plan on the node, 0% at the driver.** Its `ab`
  field measures the effect it bet on, on the scoring machine, at the right magnitude — and
  the cell does not move. Third instance at this geometry of an in-plan win that does not
  reach the driver, and the strongest argument yet that in-plan discrimination measures the
  kernel and not the cell.
* **L8_radix8's `1f520` fork: predicted ≤0.565 if the alias was the tax, measured 0.5760.**
  Its own second branch, cleanly taken, with the arena confirming 1f520 ≡ 1f.
* **L45_pfa's `pfp` tails: predicted −8 to −12 µs and a `pf0p` pick, measured zero picks.**
  Its own words: "if pfp takes zero picks, r6's dead end stands on the node too and the idea
  is closed for good."
* **L64_radix8's `scst`: predicted −0.5 to −6% if the node picked a non-plain store mode,
  measured `sc0` 3/3.** The SC-store-RFO theory — the residual r9 named for this entry — is
  dead by direct test on the machine it was written for.

**Predictions that landed, and there are a lot of them.** L6_pfa (4/4 picks 3/3, three cells
in band, one better); L6_unrolled's branch-(i)/(iii) split with the minimum in band;
L8_batchsimd's four-cell two-branch fork, all four fired; L8_fusedaxes at B=64/B=2048/B=16384;
L8_radix8's B=1 second branch and its `1f520j` null; L13_direct's B=1 null branch and both
batched bands; **L13_rader's three cells all in band with both predicted pick strings**;
L23_matrixsimd's three cells, its 3/3 determinism goal and its `tp` null; L23_rader's flat-
everywhere pre-commitment; L17_matrixsimd's branch (c) — against its own stated prior;
L17_winograd's `h4`-kept branch; L17_rader's "bet dies" branch; **all three L=36 arms'
port-0 branches, two of them beating their own bands**; L36_pfa's `nc=12` pick strings 3/3
and its p1 drop; L45_mixedradix's B=1 low-edge landing; L45_pfa's B=2/B=16 bands and both
picks; L64_blocked's B=1/B=2 bands and pick 3/3; L64_radix8's three bands and its `scst`
null. **Twenty-two pre-registered branches fired correctly, and eleven of nineteen entries
had every cell inside its own band.** Combined with §2's 32-to-5 gain-to-regression ratio,
this is both the fastest and the best-calibrated round on record — and the calibration is
what produced the speed, one round later.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### Primary: **§4.3, the re-opened L2↔DRAM clause. Measured, and the geometry the panel was about to close is not closed.**

§4.3's original question (three L1-resident axis passes vs one fused pass) was settled in
`panel_r3`. What remained open is the clause added afterwards:

> "every panel experiment fused across an **L1↔L2** boundary, where §08 §1.9's measured
> bandwidth gap is 2.6×. The untested case is **L2↔DRAM**, where the gap is 7× … **tile the
> batch so a tile fits L2, then run all three axes inside the tile.** That is not the same
> experiment the panel ran, and it is the largest untried structural move on the board."

Acting on that clause has always required a number nobody had: **what can this node actually
move, for this stream mix, at these sizes?** L17_matrixsimd's `sbw` probe supplies it, in
plan time, in every batched JSON, from the exec's own instruction shapes:

```
node, B=2048 arena (157 MB), µs per 78.6 KiB volume-equivalent, three processes
  rd  (sequential zmm read)              4.81 / 4.61 / 5.10   → 16.3 GB/s
  wr  (sequential zmm store, RFO+WB)     7.95 / 8.22 / 7.56   → 19.8 GB/s of DRAM traffic
  cp  (read burst then write burst —
       the exec's own phase alternation
       with the compute deleted)        13.31 / 13.20 / 13.05  → 17.9 GB/s
  s17 (the X pass's real 17-row-stream
       read pattern)                     3.98 / 3.72 / 4.16   → s17/rd = 0.81–0.83
  the scored cell                                     21.62   → 10.9 GB/s
```

Three things follow, and two of them overturn a prior conclusion.

1. **The batched L=17 cells are not at a bandwidth wall.** `cp` — the node's own floor for
   the identical 236 KB of compulsory traffic in the identical phase order — is **13.2 µs
   against a 21.6 µs cell.** r9 §6 framed those cells as "~21.72 µs = ~10.9 GB/s against
   ~236 KB/volume of compulsory traffic … NT stores and ERMS have both failed, so this needs
   a new idea or an honest 'closed'." The honest answer is neither: **there is ~8 µs, 39%,
   of headroom between the transform and the machine's own copy speed for its own traffic**,
   and the reason two independent implementations tie at 10.9 GB/s is not that they are both
   at the wall.
2. **The read shape is not where it is.** `s17/rd` = 0.82 on the node against 1.28–1.31 on
   Sapphire Rapids: the 17-interleaved-stream read that the whole staged-input mechanism was
   designed to fix is *cheaper* than a sequential read here. Both L=17 read-side mechanisms
   this round (matrixsimd's staged input, and rader's staged output flush from the other
   side) were declined by the node's own tuners, consistently with that.
3. **So the residual is write-side/overlap**, which is branch (c) of the entry's own
   pre-registration and the first positive localisation of the L=17 batched residual in ten
   rounds. And it is *exactly* the regime §4.3's clause names: the transform's compute
   (14.9 µs at B=1) and its DRAM traffic (13.2 µs) sum to 28.1 against a 21.6 µs cell, so
   ~7 µs of overlap is already happening and ~8 µs is not. **The L2-tiling construction the
   corpus recommends now has a measured target instead of a hypothesis, and this is the
   cell to try it on.**

§4.3's clause should record: on Cascade Lake at 4913-point volumes, single core, the
achievable read/write/copy rates for this mix are 16.3/19.8/17.9 GB/s, the batched transform
runs at 10.9, and the gap is on the write/overlap side, not the read shape. The instrument
that produced it is ~60 lines in `fft3d_create()` and portable to any entry with a phase
structure — which is the second thing to record, because it cost no monitor time at all.

### Second: **§4.6 — last round's headline is now bounded, and the bound is sharp.**

r9 settled §4.6 with a number: schedule *association order alone*, on arithmetic identical in
every countable unit, is worth **3–6% on Cascade Lake** at L=6 and is **sign-inverted** on
Sapphire Rapids. r9 §6 then asked the panel to propagate the search to every other unrolled
codelet on the board. Three entries did, and **all three read null on the node**:

| geometry | probe | node reading | threshold |
|---|---|---|---|
| L=8 (`radix8`) | `1f520j`: 16 output joins → FMA/FNMA ×1.0, bit-identical | **+0.5 to +0.7%** (arena, 3/3) | ≤ −2% to adopt |
| L=13 (`direct`) | `q`, `s` association twins in `ab[B1]`/`ab[B16]` | **q +0.5 to +0.7%, s +3.7 to +3.8%** | — |
| L=23 (`rader`) | counted out before building: +11 ops/line-group | **+3.7% cost** for a 3–6% stall effect | — |

The common structure of all three negatives is stated by the entries themselves and it is the
bound: **at L=6 the winning joins fed *stores* directly. At L=8 they feed a shuffle network;
at L=13 the twins feed a spilled-constant tail; at L=23 two thirds of the stores are already
shuffle-fed, and making the rest FMA-fed costs ops.** So §4.6's answer, refined: search
association order *on the scoring machine* — but the ~5% is not a property of association
order in general, it is a property of the **last operation before a store on a machine with
one store port**. Where the store is not what the join feeds, the effect is zero. That is a
much more useful statement than "search, not model," and it retires the propagation ask: it
has now been priced at three geometries and paid at none of them.

A codicil, and it is the round's one unexplained number: **L6_pfa's own B=32768 cell dropped
3.3% from the same association change, in the fully DRAM-bound regime, against its explicit
pre-registration that the codelet should be irrelevant there** (§2). The bound above does not
explain that, and nothing else does either.

### Third: **§4.1 — spill traffic finally has a price, from the same DAG in two files.**

§4.1's open question: "how much spill traffic do the batch-vectorised codelets actually
generate, and does it cost more than the shuffles it avoids?" The `n1_9` DAG is a natural
experiment because five entries transcribed the *identical* DAG into five different
surroundings and three of them audited their own stack traffic:

| entry | port-0 ops cut | added stack moves | cell delta at B=1 | fraction of the model collected |
|---|---|---|---|---|
| L45_pfa (r9) | −5.5% | **+23** (phase body) | **+1.2%** | ~0 |
| L45_mixedradix (r10) | −5.5% | **+3** | **−2.6%** | ~47% |
| L36_pfa (r10) | −6.45% | +22 (phase2), +78 (phase1) | **−7.7%** | ~100% |
| L36_mixedradix (r10) | −6.45% | not audited | −5.9% | ~90% |
| L36_pencilfused (r10) | −6.45% | not audited | −3.3% | ~50% |

The two L=45 rows are the cleanest comparison the project has: **same DAG, same op cut, same
geometry, same machine, +23 spills against +3, and 3.8 points of cell time between them.**
L45_mixedradix predicted exactly that in advance ("if anything I should land at the low edge
of their delta: the audit shows I took the DAG with +3 spill moves where they paid +23") and
landed there. That is a real §4.1 data point — spills are *not* free at these sizes, and a
~20-move-per-body difference is worth several points of a 5% arithmetic win.

But the L=36 rows forbid the simple conclusion: L36_pfa paid +22 and +78 spills and still
collected the whole model. So the honest §4.1 entry is: **spill cost is real and measurable at
the scale of the arithmetic win it is trading against, and it is not a function of spill count
alone.** §07 §7.8's cheap check (count `vmovupd` against `%rsp`) should become standard
practice for any entry shipping a generated DAG — three entries did it unasked this round and
it is why the spread above is interpretable at all.

### Also recorded, without claiming a §4 movement

* **§4.5** gains the refinement r9 asked for — **alias fixes are regime-local: gate them to
  the residency they were measured in** (L8_batchsimd, +1% at B=1, −3.3% recovered at
  B=2048) — and one r9 hoped for and did not get: **even the "reachable" in-scratch class
  does not transfer between implementations of the same geometry** (1-for-3, §2).
* **The `perf_event_paranoid = 4` withdrawal held.** No entry gated a result on counters this
  round. Two entries still carry a dormant probe printing `fe=na` and say so.

---

## 6. The single highest-value thing the next round should attack, per geometry

### L = 17 — the batched cells, and this is the highest-value item on the whole board

`cp` = 13.2 µs against a 21.6 µs cell (§5). **Do not close this geometry.** The item is
narrow and pre-specified by the entry that found it: `cell − cp` is ~8 µs, `s17 ≈ 0.82·rd`
says it is not the read, `b1dec` says it is not compute, and the phases add — so it is the
write stream's interaction with the read stream. The named candidate is the **pipelined +
staged hybrid** L17_matrixsimd deliberately did not build this round ("three interacting
schedules in one exec with no node signal for either half — if the node picks staged AND the
s17 gap is large, that is next round's one candidate"). The node said staged is not it and
the gap is on the other side, so the correct next build is **the write-side analogue**: stage
the *output* densely and pace its flush under the next volume's compute, with `sbw`'s `wr`
and `cp` as the in-plan discriminator. Note L17_rader built exactly that shape (`stp`) and
the node declined it — but in a structure whose x pass emits 17 concurrent misaligned
streams, i.e. the wrong baseline. In matrixsimd's dense-store structure the same construction
starts from a different place and is worth exactly one candidate.

Second, cheaper: **get a second `sbw` four-tuple.** L17_winograd's `ph`/`xp` decomposition
maps 1:1 and the entry offers it. Two independent four-tuples would make "39% headroom" a
panel-wide fact rather than one file's probe.

### L = 36 — phase 1, now that the arithmetic is done and the cell is readable

The arithmetic lever is spent: DFT9 is at genfft's count, DFT4 is at 8+1, and the cut priced
at ~full value, so **port 0 binds and B=1 is now 113 µs against a 78 µs floor.** The phase
split says where the remaining 35 µs is: **`p1` = 86.3–88.4 of `fu` = 115, i.e. phase 1 is
76% of B=1 and runs at ~1.9× its own port share while phase 2 runs at ~1.35×.** Both L=36 and
L=45 now report that same shape from independent probes. So: **attack phase 1's structure —
the plane's live window, the z/y subloop fusion, the split-access toll on the odd stride.**
L45_mixedradix has already costed the three terms (plane round trip, split accesses ~40k per
volume, compulsory L3) and argues the fusion rewrite only deletes the smallest; that analysis
should be checked at L=36, where the volume is half the size, before anyone builds it.

**Precondition, and it is now the only thing blocking readable L=36 numbers:** L36_pfa's B=4
reads 124.2 / 129.3 / 129.4 at an *identical* pick string (§3b). The tuner lottery is closed;
what is left is a run-level bimodality that no L=36 entry can see from inside its own plan.
That is a **harness** item, not an entry item: two of the four L=36 cells are decided inside
2%, and one arm's numbers are bimodal at 4%. Raising `--runs` at the tightest cells, or
reporting the median rather than the minimum, would buy more than any mechanism at this
geometry.

### L = 8 — `fusedAA`, the only unpriced shape left

B=1 has been 1.28× its floor for seven rounds and every allocation mechanism is now priced:
the in-scratch de-alias is 1-for-3, the page-align is charged, the association order is null
(§5), the arena inverts the driver at two entries, and three arms sit inside 4.5% of each
other. **The one live lead is the address-aware `fusedAA` shape, which the node's own
tournament has now picked at B=1 (L8_fusedaxes run 2, arena 0.566 against fused's 0.579) as
well as at B=64 (r9 run 3) — two first-ever picks in two consecutive rounds, in two different
cells, and no entry has attributed either.** L8_batchsimd declined to port it twice on the
grounds that "the node's own tournament declined them both times." It has now not declined it,
twice. Its own r10 rule fires: "if B=64 is still lost after the gate lands AND the node picks
AA again in their file, r11 ports it." B=64 is still lost and the node picked AA again. Port it.

**Do not run L8_batchsimd's `-DL8_JOIN_FMA` ask** — L8_radix8's node arena already answered
it at +0.5 to +0.7% (§2). That is the third round the ask has been outstanding and it is now
closed at zero cost.

**Secondary, and I am repeating r9:** L8_radix8 is third in three of four cells for the third
round with both of its pre-registered forks reading null. It is the donor.

### L = 6 — one number, then converge

There is no kernel work left and this round confirmed it from both sides: seven falsified
mechanisms, the boundary probes reading zero again (`bf − bsp` = +0.1 to +1.8 ns), the uop
theory dead, B=1 at 1.23× floor, the association order found and adopted by both arms, and
the x-pass group order measured at 1–3% in-plan and zero at the cell. **The single item is the
one number nobody predicted: the d2 codelet bought 3.3% at B=32768, where both entries'
models say a codelet cannot matter.** One in-plan A/B at nvol ≫ L3 settles whether that is
the codelet or the allocation draw. If it is the codelet, the panel has a second mechanism
class (association order survives into the bandwidth-bound regime) and §4.6's bound needs
another clause. If it is the draw, **L=6 is converged in all four cells and the r9 slot
ruling should finally be executed** — L6_pfa keeps the geometry, and the second slot goes to
L=17's batched cells (the board's largest measured headroom, §5) rather than to L=64 or L=13
as r9 suggested, because the L=17 number is new and the other two are not.

### The four wave-2 geometries, one line each

* **L = 13 — L13_direct owns all three cells for the fourth round and B=512 is the thinnest
  library margin on the board (1.12×).** The single item is unchanged and is *mine to
  satisfy*: the `-DL13_PW=0` / `-DL13_PFIN=0` split at B=512, two builds and two runs, now
  four rounds outstanding — and it is the last standing monitor ask on the board that can
  actually be run. Also: L13_rader's rollback worked in all three cells and its own
  `-DL13R_FUSE=0` ask at B=1 is one build; if the fused schedule reads ≤6.03 there, 1198
  lines of dead kernel should go.
* **L = 23 — closed, and this round closed the last opening.** `tp` = pick +2.3 to +5.7% in
  all three processes: the tail-paced schedule joins NT, deferred-Z, pf=1 and uniform
  pipelining as documented streaming nulls, and **the L=23 streaming schedule space is
  exhausted.** 1.13× floor at B=1, tightest on the board, two bit-identical arms overlapping
  in all three cells. **Stop funding it.** One housekeeping item: the B=128 pick lottery has
  moved to L23_rader (three different picks in three runs) while L23_matrixsimd holds 3/3 —
  if the surviving arm is rader, its head slot needs matrixsimd's correction applied.
* **L = 45 — the floor ratio finally moved (1.68× → 1.64×) and the reason is the leaner
  transcription** (§5, §4.1). L45_pfa owns B=2/B=16 and closed its own `pfp` idea for good.
  The single item is the one-flag A/B nobody has run: **`-DFFT45_ODDR8` at B=16**, where
  L45_mixedradix has now regressed two rounds running (+1.8% this round, +4.6% above r7) at
  an unchanged pick with the odd-column rework as the only change on that path. One build,
  one run, third round of asking.
* **L = 64 — L64_blocked's split-complex rewrite is the round's largest gain and the geometry
  is a tie at B=1 for the first time.** But the floor ratio is still 1.73×, the board's worst,
  and both entries' B=1 residual mechanisms are now dead by their own node tests
  (L64_radix8's `scst` plain 3/3; L64_blocked's `st=1`/`st=2` in earlier rounds). The single
  item is **B=8, where L64_blocked's `st=3` bought nothing (1311 against a 940–1130 band) and
  its pick fell back to `st=0` in one run** — the split-complex win is real at B=1/B=2 and
  absent at 64 MB, which is a residency boundary worth locating, and it is also where §3(a)'s
  unvalidated-bit-class exposure sits.

---

## 7. Promotion

Applying `CURATION.md` in order. **Rule 4 ("anything that beat a library baseline") selects
all nineteen** — every arm beat every library in every cell — so it carries no discriminating
weight and the decision rests on rules 1–3 and on the explicit prohibition against
near-duplicates.

**Rule 1 — the fastest correct entry per geometry, one per L, always.** On distributions,
with §3(b)/§3(c) applied: **L6_pfa** (B=1, B=64, B=32768), **L8_fusedaxes** (B=64 and
B=16384 disjoint; B=1 tied; B=2048 owns the minimum), **L17_matrixsimd** (B=1 and B=8
disjoint, B=256 by 1.0%), **L36_mixedradix** (B=4, B=32, B=256; B=1 tied), **L13_direct**
(all three, disjoint), **L23_rader** (minima in 2 of 3, all three overlapping),
**L45_pfa** (owns the geometry's one separated cell, B=16, by 7.5%), **L64_radix8** (all
three on medians).

**Rule 2 — a structurally different runner-up that came close.**

* **L6_unrolled** — owns B=4096 outright on disjoint distributions (−3.3%), closed its B=1
  median deficit from 9.1% to 0.9%, and carries the round's `ab1` x-pass-group-order
  instrument (`fx − f` = −0.6 to −3.1% on the node, published on every line). The 1–2%
  in-plan / 0% at-cell split is only readable from this file.
* **L8_batchsimd** — lowest minimum on the board at B=1 (0.5510), owns B=2048 on medians,
  and **is the file that confirmed r9's attribution by reverting it**: the regime-gated
  allocation is §4.5's new refinement and the four-cell two-branch fork fired in all four.
* **L17_winograd** — a genuinely different structure (296-instruction hand-derived
  cyclic+negacyclic module), holds B=2048 on medians and is within 2.5% at B=256, and its
  `i4` split-free load path — `cmp`-verified bit-identical, picked 1 of 9 batched runs —
  **closes the split-load mechanism in the second of L=17's two structures**, which is what
  makes it closed rather than untested.
* **L36_pfa** — a different decomposition (GT-PFA two-sweep against lanes-are-lines), ties
  B=1 with the lowest minimum on the board (113.128), took the deepest cut of the round
  (−7.7%), **closed r9's worst measurement at the tuner** (identical picks 3/3, the 39.5%
  outlier gone), and produced the phase split that names L=36's next item (`p1` 93.0 →
  86.3–88.4). Promoted with §3(b) recorded against its B=4 number.
* **L13_rader** — structurally distinct (Rader-13 against dense 13×13), executed r9's ordered
  rollback with all three cells in band and **both r9 regressions recovered** (−3.2% at B=16,
  −5.3% at B=512), and **retracted part of r9's own attribution** by finding that its U buffer
  had been silently 8-deepened on both schedules, so r9's "the fused schedule wins at batch"
  was partly measuring a 26.6 KB footprint change.
* **L45_mixedradix** — takes B=1 on minima, and is one half of the round's §4.1 result: the
  same DAG, +3 stack moves against the rival's +23, and **2× the port-0 payoff, predicted in
  advance from its own audit.** Also carries an instructive unresolved failure (B=16, two
  rounds, +4.6% above r7, one-flag A/B shipped and unrun).
* **L64_blocked** — **the round's largest single gain (−13.2% at B=1, −12.5% at B=2)**,
  landed inside its pre-registered band with its predicted pick 3/3, closing a 15% structural
  deficit to a 0.3% tie after two refused structural bets in r8 and r9. This is what a
  promoted "instructive failure" turning into a win looks like, and the two dead alternatives
  (`st=1`, `st=2`) are in the same record.

**Rule 3 — instructive failures whose record documents the number that killed them.** Three
of the rule-2 promotions rest substantially on this ground (L13_rader's retraction,
L45_mixedradix's B=16, L64_blocked's `st=1`/`st=2` history), and one rule-1 promotion does
too: **L17_matrixsimd's staged twins were declined 3/3 and its own honest prior (branch (a),
"the cells are bandwidth-closed") was refuted by its own instrument.** That record — a
pre-registered three-way fork where the author bet on the branch that lost — is the single
most valuable artifact in the round (§5) and the reason the geometry stays open.

**Not promoted, with the number:**

* **L8_radix8** — third in three of four cells for the third round; B=64 +3.4% on minima;
  its B=1 fork read its null branch (0.5760 against a ≤0.565 criterion) and its `1f520j`
  association probe read +0.5 to +0.7% against a ≤ −2% criterion. Both nulls are genuinely
  valuable and both are **already carried in promoted records**: the §4.5 in-scratch-transfer
  negative in L8_fusedaxes' `arena{fused,fusedSI,…}` string, and the association-order
  negative in L13_direct's `ab[]` q/s fields at a second geometry. Named the donor for the
  second round.
* **L17_rader** — third in all four cells; `st`/`stp` declined 6/6 with the cells landing in
  its own "bet dies" band (23.874 / 24.608); fifth consecutive round of node-rejected batched
  mechanisms. Its record's own recommendation is the consolidation, and its partial-line-waste
  theory is superseded by the sbw measurement in a promoted file, which shows the headroom is
  not where `st`/`stp` looked for it.
* **L23_matrixsimd** — bit-identical to L23_rader, overlapping in all three cells; the
  near-duplicate prohibition is exactly on point. Its two real deliverables — the `tp` null
  that empties the L=23 streaming schedule space, and its 3/3 determinism against rader's
  three-way flip — are recorded in §2 and §6 and should be copied into the surviving arm's
  record at promotion time.
* **L36_pencilfused** — third at B=1/B=32/B=256; its own port-0 branch fired (119.588 against
  a ≤121.5 criterion) and it recovered its standing r7→r9 B=256 regression (189.32 → 184.46,
  now below r7). Declined on the near-duplicate rule, which bites hardest here: **all three
  L=36 arms shipped the identical single change this round**, two are promoted, and the
  arithmetic result is triply recorded. Its exact-disassembly verification of the new 78 µs
  floor ("232 FP-port instructions per line body … the model is exact") is carried in §1 and
  §2.

Fifteen entries, unchanged in membership from `panel_r9` — but the round underneath is not
the same round: r9 promoted three wave-2 arms primarily as instructive failures, and r10
promotes them for having recovered, retracted, or won.

**A note for `NOTES.md` at promotion time.** This is the round where the previous round's
information turned into speed: 32 cells improved against 5, 22 pre-registered branches fired,
eleven of nineteen entries landed every cell inside their own band, and the four biggest moves
all came from a monitor directive or a rival's record rather than from a new idea
(`n1_9` × 5 files, `st=3` from the rival's currency, the regime-gated allocation, the
rollback). Three things should be recorded as established: **port 0 binds at L=36 and the
arithmetic lever is spent** (−6.45% ops bought −5.7 to −7.7%, floor 83 → 78 µs);
**the L=6 association-order result is bounded to joins that feed stores** and is null at L=8,
L=13 and L=23; and **the batched L=17 cells have 39% headroom over the node's own copy speed
for their own traffic**, which reopens the geometry the last verdict was ready to close. Two
things should be recorded as corrections: **r8 and r9, not r7, were the anomalous MKL rounds**,
and four margins in each of those verdicts were inflated by 17–30%; and **the timed≠checked
exposure is no longer hypothetical** — two leaderboard numbers this round were produced by
bit classes that no r10 check validated. The harness item is now more valuable than any
entry's next mechanism: at three geometries the leading cells are decided inside 2% and at
least one arm is bimodal at 4%.

---

## Provenance and housekeeping

Sources: `bench/geom/impl_10/` (`impl` → `impl_10`), all 19 changed from `impl_9` (60–722
lines each), `baseline_matrix.c` byte-identical. Measurements:
`results/panel_r10/leaderboard.txt`, `environment.txt`, 2169 `t_*.json` and 259 `c_*.json`,
`check.log` (259/259 PASS), `timing.log` (782 lines, three runs per cell with per-run minima
and medians), `timing.err` (1392 benign unsupported-geometry lines, zero others),
`build_errors.txt` (empty). No `failures.txt` was created, which under `sweep.sh`'s exit-code
rule means no entry crashed, hung, timed out, or tripped `driver.c`'s anti-memoization gate.
Strategy records for all 19 entries are updated in `bench/geom/strategies/` (+3113 lines).

Retracted this round: **`panel_r9` §3(e)'s conclusion that r7 was the anomalous MKL round.**
r10 reproduces r7 within 0.7% at all four affected cells in both MKL builds while every other
backend at those cells moved <3% across four rounds. The margins quoted at L=36 B=32, L=36
B=256, L=45 B=16 and L=64 B=8 in the `panel_r8` and `panel_r9` verdicts are inflated by
17–30% and should be read against r10's libraries instead.

Closed this round, by the panel's own pre-registered criteria: the association-order
propagation ask (null at L=8, L=13, L=23 — §5); L8_batchsimd's three-round
`-DL8_JOIN_FMA` monitor ask (answered by L8_radix8's node arena at zero cost); the L=23
streaming schedule space (`tp` = pick +2.3 to +5.7%); L45_pfa's `pfp` (zero picks);
L64_radix8's SC-store-RFO theory (`sc0` 3/3); L17_rader's partial-line-refetch theory
(declined 6/6); L8_radix8's and L8_fusedaxes' in-scratch de-alias transfers (null);
L13_direct's X-first driver bet (null); L17_winograd's `i4` (1 pick of 9).
Still outstanding on me: `-DL13_PW=0` / `-DL13_PFIN=0` at L=13 B=512 (fourth round), and
`-DFFT45_ODDR8` at L=45 B=16 (second round, now with two regressions behind it).

PROMOTE: L6_pfa L6_unrolled L8_batchsimd L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pfa L45_mixedradix L45_pfa L64_blocked L64_radix8
