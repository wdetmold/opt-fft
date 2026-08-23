# ice_r4 — monitor's verdict

Round measured on **a80n0.lqcd.mit**, `slurm_job 438572`, 2026-08-23T01:16 EDT.
CPU per `environment.txt`: **Intel Xeon Gold 6326 @ 2.90 GHz — Ice Lake-SP**, 64 threads,
gcc 11.4.0, governor `schedutil`.

## 0. Two facts that frame everything below

**(a) The graded task changed between r3 and r4.** Commit `fdfe70b` ("Ice panel switches to
the full rival step, gated on whole-chain correctness"), timestamped 21:41 — thirteen
minutes after r3's timing pass and three and a half hours before r4's — replaced the chain
step. r3 timed `--chain m --unitary` (FFT plus a scalar 1/L^1.5 scale). r4 times
`--chain m --map --cin`, the rivals' real step:

    state <- (z + c) / (1 + |z + c|),   z = FFT(state)

an elementwise complex magnitude and divide per point per step. **r3 and r4 per-transform
numbers are not comparable**, and every apparent library "regression" in §2 is the map being
added to the work, not a slowdown. Corroboration: r3's per-call figures sum to 0.9716 s,
which is exactly the "our FFT-only chain (0.97 s)" datum in `fdfe70b`'s own message.

**(b) The scoring node is Ice Lake, and the implementers develop on it.** The monitor prompt
for this round states the node is a "Xeon Gold 5218, Cascade Lake" and that implementers
"develop on Sapphire Rapids with full-clock AVX-512 and 2 MB L2". Both are wrong.
`environment.txt` and `PANEL_BRIEF.md` agree the node is a80n0 / Gold 6326 (Ice Lake-SP,
2×512-bit FMA pipes, 48 KB L1d, 1.25 MB L2); `tryout.sh` builds `-march=native` **on that
same node** and runs dev timings on a leased pinned core there. There is no cross-machine
gap to attribute anything to. See §4 — the consequence is that claimed and measured numbers
agree to ~1%, and the residual noise source is core-lease contention, not microarchitecture.

---

## 1. Headline per geometry

**Non-batched (B=1) was not measured this round.** `cases.txt` pins exactly one batch per
size (`6:64:4856, 8:64:2572, 13:32:1278, 17:32:98, 23:16:165, 36:8:64, 45:4:177, 64:2:134`),
and no `*_B1_*.json` exists in `results/ice_r4/`. Several implementers report B=1 dev numbers
in their records, but those are unscored and were taken in un-drained windows; I am not going
to launder them into a headline. The batched cells below are the whole of the round's evidence.

Both metrics are from `leaderboard.txt`: `per-transform` = chain time / (B·m); `per-call` =
the whole graded point, directly comparable to the rivals' per-size marks in `PANEL_BRIEF.md`.

| L | fastest correct panel entry | per-transform | graded point | best library | ratio | vs rivals' mark |
|---|---|---|---|---|---|---|
| 6  | `L6_unrolled` **‡ uncertified** | 0.332 us | 0.1033 s | `mkl_dfti` 0.941 us | **2.83×** | 0.328 us → 1.3% behind |
| 8  | `L8_radix8` | 0.564 us | 0.0929 s | `mkl_dfti` 2.102 us | **3.73×** | 0.699 us → **1.24× ahead** |
| 17 | `L17_matrixsimd` | 13.009 us | 0.0408 s | `ducc0_c2c` 86.405 us | **6.64×** | 11.16 us → 1.17× behind |
| 36 | `L36_mixedradix` | 111.425 us | 0.0570 s | `mkl_dfti` 283.322 us | **2.54×** | 115.2 us → **1.03× ahead** |

‡ `L6_unrolled` failed the whole-chain gate — as did every other backend at L=6, including
MKL, FFTW, ducc0 and the harness's own `baseline_matrix`. See §3.1; the cell is void, not lost.

The other four geometries, for completeness (all certified):

| L | fastest correct | per-transform | graded point | best library | ratio | vs rivals' mark |
|---|---|---|---|---|---|---|
| 13 | `L13_direct` | 5.837 us | 0.2387 s | `mkl2026_dfti` 11.795 us | 2.02× | 4.010 us → 1.46× behind |
| 23 | `L23_rader` | 38.105 us | 0.1006 s | `ducc0_c2c` 247.547 us | 6.50× | 39.02 us → **1.02× ahead** |
| 45 | `L45_pfa` | 283.339 us | 0.2006 s | `mkl_dfti` 756.802 us | 2.67× | 283.9 us → **parity** |
| 64 | `L64_radix8` | 1042.956 us | 0.2795 s | `mkl_dfti` 1723.169 us | 1.65× | 843.3 us → 1.24× behind |

**Suite total: 1.113 s** against **1.005 s** for the rivals' reconstructed code on this node,
and **3.188 s** for the best library at each point (2.86×). The configuration this round
replaced — the same kernels with the driver-side unfused map — measured **2.24 s** in the
head-to-head recorded in `fdfe70b`. So the round took the suite from 2.24× the rivals' time
to **1.11×**, and won four of eight points outright.

---

## 2. What changed since ice_r3, per geometry

Because the metric changed (§0a), the honest comparison is *the marginal cost of the map*:
what each geometry pays for the map now that entries own the chain, versus what it costs
unfused. The fused increment is r4's winner minus r3's winner at that L; the unfused
increment is MKL's r3→r4 delta, MKL being the same library code timed through the
driver-side map both times.

| L | r3 winner (FFT only) | r4 winner (FFT+map) | fused map costs | unfused map costs (MKL) |
|---|---|---|---|---|
| 6  | 0.213 us | 0.332 us | **+0.119 us (+56%)** | +0.601 us (+177%) |
| 8  | 0.544 us | 0.564 us | +0.020 us (+3.7%) | +1.476 us (+236%) |
| 13 | 4.619 us | 5.837 us | **+1.218 us (+26%)** | +5.770 us (+93%) |
| 17 | 13.061 us | 13.009 us | **free** (−0.4%) | +12.55 us (+16%) |
| 23 | 39.502 us | 38.105 us | **free** (−3.5%) | +31.04 us (+13%) |
| 36 | 109.619 us | 111.425 us | +1.806 us (+1.6%) | +122.65 us (+76%) |
| 45 | 259.287 us | 283.339 us | +24.05 us (+9.3%) | +235.98 us (+45%) |
| 64 | 902.885 us | 1042.956 us | +140.07 us (+15.5%) | +703.20 us (+69%) |
| **suite** | **0.9716 s** | **1.1135 s** | **+0.142 s (+14.6%)** | **+1.27 s (+131%)** |

**Nothing regressed.** Every entry is doing strictly more work than its r3 self and seven of
eight geometries absorbed the map for under 16%; at L=17, L=23 and L=8 it is free or nearly
so, meaning the map's uops ride entirely in the shadow of existing stalls. Fusion bought a
factor **8.9×** on the map's cost suite-wide (0.142 s against 1.27 s).

Three rank changes worth naming, all caused by the metric change rather than by anything
getting slower:

* **L=8 inverted.** In r3 the order was `L8_fusedaxes` 0.544, `L8_batchsimd` 0.547,
  `L8_radix8` 0.567. In r4 it is `L8_radix8` 0.564, `L8_fusedaxes` 0.744, `L8_batchsimd`
  0.779. The two entries that won the FFT-only race are now 1.32× and 1.38× behind the one
  that lost it. **The FFT-only winner is not the fused-chain winner** — the single most
  transferable lesson of the round.
* **L=36 reordered** into a dead heat: `L36_mixedradix` 111.425 and `L36_pencilfused`
  111.962 are 0.5% apart (r3: pencilfused 109.6, pfa 111.6, mixedradix 117.4). `L36_pfa`
  fell to 1.15× — its fused map is the weakest of the three.
* **L=45 converged**: r3 had mixedradix 259.3 ahead of pfa 293.7; r4 has pfa 283.3 and
  mixedradix 283.8. pfa improved 3.5% while absorbing the map; mixedradix paid 9.3%.

---

## 3. Correctness, failures, and things that must not survive

Build: `build_errors.txt` is **empty**. No `failures.txt` exists. `agents/exits.txt` shows
`exit=0` for all 19 implementers. All 19 sources in `impl_4/` appear in the leaderboard and
all 19 have a strategy record in `strategies/`. Nothing crashed, hung, or is missing.

Single-transform correctness passes everywhere, 1.8e-16 to 8.4e-16.

### 3.1 Every backend at L=6 fails the whole-chain gate — including the reference

`check.py --map-check` sets `eff_tol = max(1e-12, 1e-13·m)`; at L=6, m=4856 gives
**4.856e-10**. From `check.log` and the `c_*.json` records, the m=4856 end-state drift:

| backend | chain rel_l2 | verdict |
|---|---|---|
| `mkl_dfti` | 6.249e-10 | FAIL (1.3×) |
| `mkl2026_dfti` | 1.057e-09 | FAIL |
| `L6_unrolled` | 1.388e-09 | FAIL |
| `ducc0_c2c` | 1.477e-09 | FAIL |
| `fftw3_estimate` / `_measure` / `_patient` | 1.545e-09 | FAIL |
| `L6_pfa` | 1.896e-09 | FAIL |
| **`baseline_matrix`** (the harness's own library-free reference) | **4.004e-09** | **FAIL (8.2×)** |

Seven independently written, exact double-precision FFTs — three of them shipped libraries,
one of them this project's own O(L⁴) reference — span 6.2e-10 to 4.0e-9 and **all** exceed
the gate. `L6_pfa`'s record adds the decisive control: **pure numpy against pure numpy**,
substituting `sqrt(re²+im²)` for `np.abs` (a sub-ulp difference; `np.abs` is `hypot`), lands
at 1.117e-9 — the reference implementation fails its own gate. `L6_unrolled` independently
measured the drift's growth with chain length: 2.5e-11 at m=2572, 2.9e-10 at m=3600,
1.40e-9 at m=4856.

**This is a harness calibration failure at L=6, not an implementation defect.** `fdfe70b`'s
own message says the 1e-13/step budget was "Verified end to end on the node at the graded
**L=17** point" (m=98) — it was never validated at m=4856, where the chain amplifies per-step
rounding by ~1e7 rather than the assumed ~4.4e4. The same binaries pass comfortably at every
other size: margins are 48–56× at L=8, 236–284× at L=45, 477× at L=36, 620× at L=23, 684× at
L=17, 1040× at L=13.

Consequences I am enforcing:
* **No L=6 result is certified this round**, panel or library. The 0.332 us headline stands
  as a measurement, not as a passed entry.
* **No L=6 entry is blamed.** Rejecting `L6_unrolled` and `L6_pfa` for a gate that MKL, FFTW,
  ducc0, `baseline_matrix` and numpy-vs-numpy all fail would be a false conviction.
* The gate must be recalibrated before ice_r5, to a floor derived from measurement rather
  than from the L=17 extrapolation. `L6_pfa`'s record proposes ≥4e-13/step to admit MKL
  through the fallback and ≥1e-12/step to be seed-robust; `L6_unrolled` proposes ~5e-9 at
  this cell. Both are defensible; the requirement is that the recalibrated gate admit at
  least one independent exact implementation, which the current one does not.

### 3.2 The leaderboard silently displays failed-chain entries as passing — fix before r5

`check.py` builds `result = dict(ok=ok, …)` *before* the chain check, then in the
`--map-check` block does `ok = ok and chain_ok` and writes `result["chain_ok"]` /
`result["chain_rel_l2"]` — but **never reassigns `result["ok"]`**. `leaderboard.py` reads
`chk["ok"]`. So both L=6 entries are printed `ok 2.4e-16` and ranked `1.00x` / `1.09x`
despite failing the gate this round is explicitly "gated on". `sweep.sh` does not check
`check.py`'s exit status either, so nothing gates.

This round the bug was harmless — everything at L=6 failed together and I caught it in the
raw JSON. Next round it is exactly the hole a fast-but-drifting entry walks through: it
would take first place with "ok" beside it. One line (`result["ok"] = ok`) closes it. **The
round's stated gate was not actually enforced by any code in the pipeline.**

### 3.3 `L64_blocked` — the only passing entry with a single-digit margin

`chain_rel_l2` 1.827e-12 against tol 1.34e-11: **7.3× margin**, versus 304× for its
stablemate `L64_radix8` (4.408e-14) on the identical cell. 41× more drift, and it buys
nothing — blocked is 1048.605 us against radix8's 1042.956. Its own record documents the
choice (`MAPDIV=0 1107.7 / MAPDIV=1 1161.5 / MAPDIV=2 1159.3`), i.e. the cheapest map arm
was taken and the extra Newton refinement was priced at 5%. It passes and it is promoted,
but flagged: at L=64 the m=134 chain is short, so 7.3× is thinner than it looks if a future
round lengthens the chain or reseeds. Tighten it or spend the 5%.

### 3.4 `L8_batchsimd` and `L8_fusedaxes` are numerically the same code

Identical to all printed digits at both checks: `rel_l2` 2.269334791e-16 and `chain_rel_l2`
**4.593166421760876e-12**, while `L8_radix8` differs (5.297e-12). Both records independently
claim 1.660e-11 dev drift; `L8_radix8`'s record explains the split — "pure reassociation from
the rotated DFT axis order". Bit-identical output across 2572 steps of a nonlinear map is not
a coincidence: these two entries are one numerical path in two schedules, and they run 1.32×
and 1.38× behind `L8_radix8`. Not a correctness problem; a curation one (§7).

### 3.5 No fast-wrong-answer survivors

Every promoted entry outside L=6 passes with ≥7.3× margin, and the map is verified through
the *end state* of the whole chain, so an entry cannot skip work: a truncated or cached step
shows up at 1e0, not 1e-14. The one structural risk left in the pipeline is §3.2, which is a
harness fix, not an entry.

---

## 4. Claimed versus measured

The premise supplied to me for this section does not hold: implementers develop on **the same
node** they are scored on (`tryout.sh` builds `-march=native` on a80n0 and runs on a leased
pinned core there), so there is no Sapphire-Rapids-vs-Cascade-Lake gap to attribute anything
to, and MKL's "2.9× span" between machines is not in play. What the records actually show is
close agreement, which is the expected result of same-machine development:

| entry | claimed (record, quiet window) | measured | delta |
|---|---|---|---|
| `L6_unrolled` | 0.331–0.332 us | 0.332 | 0% |
| `L6_pfa` | 0.364 us | 0.363 | −0.3% |
| `L8_radix8` | 0.565–0.568 us | 0.564 | −0.2% |
| `L8_fusedaxes` | 0.742–0.759 us (5 processes) | 0.744 | within stated band |
| `L8_batchsimd` | 0.778–0.800 us | 0.779 | within stated band |
| `L17_matrixsimd` | 12.971 us | 13.009 | +0.3% |
| `L17_winograd` | 14.870 us | 14.854 | −0.1% |
| `L17_rader` | 17.551 us | 17.521 | −0.2% |
| `L36_mixedradix` | 112.78, predicted 110–114 | 111.425 | inside prediction |
| `L36_pencilfused` | 112.9–113.3, predicted 110–116 | 111.962 | inside prediction, at the good end |
| `L36_pfa` | 129.4–130.4 us | 128.551 | −0.7% |
| `L64_radix8` | 1048.5–1160, quiet floor lower | 1042.956 | at/below the floor |

The only sizeable gaps are between *probe* numbers and end-to-end numbers, and the records
label them as such — e.g. `L36_pfa`'s in-plan probe `fu=91.7 us` against 128.551 measured, and
`L36_mixedradix`'s two-pipe port floor of 56.5 us/vol against 111.425. Those are model-versus-
reality gaps (§4.6 of the literature), not measurement disagreements. Two records
(`L8_radix8`, `L17_rader`) document **bimodal windows** on this node — the identical binary
reading 0.565 or 0.644, and 18.86 then 28.35 us/step — with MKL steady through the swing.
That is co-tenant L3 pressure on a 24 MB shared L3, and it is the reason the drained scoring
window matters. It is also why I treat the L=45 gap (0.16%) as a tie: it is one-ninth of
`L45_pfa`'s own 4.2% run spread.

No implementer overclaimed. Two under-claimed (`L36_mixedradix`, `L64_radix8`) by predicting
their quiet-window numbers conservatively.

---

## 5. Which open question moved

**§4.3, "Is axis fusion worth 3× or 3%?" — moved decisively, in the regime the r3 verdict
left open, and by fusing something other than an axis.**

r3's answer (recorded in §4.3's own block quote) was "single-digit percent, sometimes
negative", measured across L1↔L2 boundaries where §08 §1.9 puts the bandwidth gap at 2.6×.
The re-opened case was L2↔DRAM, where the gap is 7×. This round fused an **elementwise
nonlinear epilogue** — a full extra streaming pass with a divide — into the FFT's own passes,
and measured it at eight sizes against an identical unfused control (the driver-side map, run
by the libraries):

* Suite-wide, the map's marginal cost fell from **1.27 s unfused to 0.142 s fused — 8.9×**,
  taking the whole task from 2.24 s to 1.113 s, a **2.01× end-to-end win from fusion alone**.
  That is not 3–5%. TurboFNO's prior does not describe this case.
* The size dependence is exactly **Tolmachev's rule** (§07 §1.6: payoff = avoided passes ×
  the bandwidth gap of the levels involved), and it cuts *both* ways:
  - Where the FFT has instruction-level slack and the volume is L2-resident (L=17: 4.80 MiB,
    L=23: 5.94 MiB, L=8: 1.00 MiB), the map is **free** — it hides completely in existing
    stalls. Unfused it costs 13–236%.
  - Where the FFT is already minimal-instruction and traffic-bound (**L=6**, 36 instructions
    per codelet, 0.42 MiB), fusion still leaves **+56%**: there is no slack to hide a divide
    in. This is the sharpest counterexample in the project to "fusion is always small".
  - Where the working set exceeds L2 and the chain cycles buffers through L3/DRAM (**L=64**,
    16.00 MiB, +15.5%; **L=13**, 2.15 MiB against a 1.25 MB L2, +26%), fusion pays but does
    not fully absorb.
* So the answer to §4.3 is neither 3× nor 3%: **it is the ratio of the fused pass's uops to
  the host kernel's spare issue slots**, and it ranges from 0% to 56% across one order of
  magnitude in L on one machine.

§4.3's specific untried construction — tile the batch so a tile fits L2, then run all three
axes inside the tile — was **not** tested and remains open; the L=36/45/64 entries fuse
within a volume, not across an L2-sized batch tile. See §6.

Secondary movement: **§4.2 (L=17 dense vs Rader vs Winograd)** got a third consecutive
measurement, now under the real metric, and the ordering is unchanged and widened —
dense-symmetric 13.009, Winograd 14.854 (1.14×), Rader 17.521 (1.35×). §02 §7's "Rader is not
the lever at L=17" is now three-for-three. §4.2's part (b), the symmetric/antisymmetric
convolution split, is still untried. Note the ordering **flips at L=23**, where Rader wins
(38.105) over dense (44.872) by 1.18× — the crossover is between 17 and 23, which no source
in the corpus predicts.

---

## 6. The single highest-value thing for ice_r5, per geometry

**L=6 — recalibrate the gate, then take the fdiv arm.** This is one lever, not two. The cell
is 1.3% off the rivals' mark with a *compensated* map chosen purely for drift; `L6_pfa`'s
record shows the uncompensated `fdiv` arm at **0.304–0.328 us**, which beats the rivals'
0.328 outright, at 2.6× more drift. Since no exact implementation can pass the present gate
(§3.1), the compensation is currently buying nothing that is being measured. Fix the
tolerance to a measured floor, re-run both arms against it, and L=6 likely converts from a
1.3% loss into a win. Nothing else at L=6 is worth a round.

**L=8 — count the spill traffic in the fused chain (§4.1).** The cell is won (1.24× ahead of
the rivals, 3.7× ahead of MKL) and the map is free, so all remaining headroom is in the
codelet. §4.1's open question — §01 §7.2 measures 19 live registers for n=8 against 16
architectural, and predicts spill — has never been checked, and the fused map *adds* live
values to a codelet already over budget. Run §07 §7.8's cheap check on `L8_radix8`'s fused
chain: count `vmovupd` against `%rsp`/`%rbp` in the generated assembly before believing any
further timing. Second priority: retire one of `L8_batchsimd`/`L8_fusedaxes` (§3.4) and give
the seat to a size that is losing.

**L=17 — attack the 17-point kernel, not the chain.** `L17_matrixsimd`'s own record prices
the FFT skeleton at **11.33 us/step** with the map fully hidden, and ships 12.971; the rivals
are at **11.16**. So the entire 1.17× deficit is kernel structure, and the map is already
optimal. The untried move is §4.2(b): the symmetric/antisymmetric convolution split on top of
the dense conjugate-symmetric form, so each sub-transform is real-input and half cost —
orthogonal to the multiply/add tradeoff and never measured here. Read the rivals' phase-split
codelet alongside it (`PANEL_BRIEF.md` prices it at 1.6× on their 23-point kernel).

**L=36 — the L2 batch tile (§4.3's untried construction).** L=36 is ahead of the rivals and
the two leaders are in a 0.5% dead heat, so kernel micro-tuning is exhausted; but
`L36_mixedradix`'s own two-pipe port floor is **56.5 us/vol** against 111.425 measured —
a factor of two unexplained by arithmetic. One volume in+out is 1.49 MB, already past the
1.25 MB L2, and the graded cell's working set is 11.39 MiB: this is precisely the L2↔DRAM
regime §4.3 re-opened and the construction Intel's manual, Alappat et al. and L3-Fusion all
independently recommend. Tile the batch to fit L2 and run all three axes plus the map inside
the tile. Largest untried structural move on the board, and L=36 is the cleanest place to
test it.

For the record, the other four: **L=13** is the worst deficit on the board (1.46× behind) and
pays +26% for its fused map — the map, not the kernel, is the lever there. **L=64** is the
second-worst (1.24×) and pays +15.5%; the z-split layout the brief aimed at it is in
`L64_radix8` and has not yet closed the gap. **L=23** and **L=45** are at parity and need
nothing but defence.

---

## 7. Curation

Applying `docs/CURATION.md`. Note that **no exemplar from ice_r1–r3 contains a fused map
chain at all** — the entry point did not exist before this round — so no r4 entry is a
near-duplicate of an existing exemplar, however familiar its FFT lineage.

**Rule 1 — fastest correct per geometry (8).** `L6_unrolled`*, `L8_radix8`, `L13_direct`,
`L17_matrixsimd`, `L23_rader`, `L36_mixedradix`, `L45_pfa`, `L64_radix8`.
(*L=6 has no certified entry; `L6_unrolled` is the fastest measurement at the cell and its
record independently documents the gate failure. Promoted as **uncertified**.)

**Rule 2 — close, structurally different runner-up (4).**
`L17_winograd` (1.14×; hand-derived Winograd module vs dense conjugate-symmetric — §4.2 needs
both written down); `L36_pencilfused` (1.005×, a dead heat; plane-fused pencils vs 2-sweep
PFA); `L64_blocked` (1.005×; 8×8 blocked + hugepage scratch vs split-complex z-split layout —
promoted with the §3.3 drift flag attached); `L13_rader` (1.09×; Rader CRT vs dense conj-folded,
and two *different* fused-map strategies at the geometry with the worst deficit).

**Rule 3 — instructive failures whose record documents the killing number (2).**
`L6_pfa` — slower (1.09×) and it failed the gate, but its record is the round's single most
valuable document: a six-row A/B of chain drift including the pure-numpy-against-numpy
control at 1.117e-9, the mechanism (~1e7 amplification at m=4856, per-volume Lyapunov luck —
the same binary passes at B=1 and fails at B=64), the two numpy-semantics traps (`np.abs` is
`hypot`; numpy divides complex-by-real by Smith's algorithm, so a *more* accurate quotient
scores *worse*), and a concrete recalibration proposal. Without this in front of the next
panel, ice_r5 re-runs the entire L=6 cell blind.
`L8_fusedaxes` — the r3 winner at 0.544 that came 1.32× last-but-one at 0.744 once the map
moved inside, with its map-arena table (`slot+pfs=0.756, rr46=0.753, div=0.781, fma=0.831 …`)
recording which map families it tried and what each cost. This is the "FFT-only winner is not
the fused-chain winner" lesson with code attached.

**Rule 4 — beat a library.** Every panel entry beat every library at its geometry (1.65× to
6.83×), so this adds nobody.

**Not promoted, and why.** `L8_batchsimd` — bit-identical output to the promoted
`L8_fusedaxes` and slower (§3.4); one numerical path needs one exemplar. `L17_rader` — 1.35×,
outside the runner-up window, third consecutive loss at this size, and already promoted in
ice_r1. `L36_pfa` — 1.15× and the same PFA 4×9 two-sweep decomposition as the promoted
`L36_mixedradix`; its distinguishing feature is a *worse* fused map. `L45_mixedradix` — same
PFA 9×5 decomposition as the promoted `L45_pfa` and a statistical tie with it (0.16%, inside
`L45_pfa`'s 4.2% spread); noted here that it holds the same number at 1/3 the spread and
would be the safer pick if the tie is ever scored again, and it remains in `impl_4/` for
provenance.

PROMOTE: L6_unrolled L6_pfa L8_radix8 L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pencilfused L45_pfa L64_radix8 L64_blocked
