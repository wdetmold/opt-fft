# gen_r8 — monitor's verdict

Round `gen_r8`, scored on `a80n0.lqcd.mit`, slurm job 438682, 2026-08-25T15:40:06-04:00.
Sources: `impl_8/`. Statistic throughout: **min over 3 runs of the per-execute median**
(`leaderboard.py:124`), with the run spread reported alongside.

---

## 0. Two corrections to the brief, before any numbers

**(a) The geometries in the brief were not measured in this round.** The brief asks for
L = 6, 8, 17, 36. `gen_r8` measured **L = 10, 12, 15, 20, 25, 27, 31, 32, 40, 50, 100**.
There is no timing or correctness record for 6, 8, 17 or 36 anywhere in this round
(`ls results/gen_r8 | grep -E '_L(6|8|17|36)_'` returns nothing). L = 6/8/17/36 is the
geometry set of the **earlier `bench/geom` panel campaign** — see
`bench/geom/exemplars/panel_r11/{L6_pfa.c,L8_batchsimd.c,L17_matrixsimd.c,L36_pfa.c}`.
The `bench/gen` campaign is the any-L successor and uses the eleven-cell suite above.
This verdict reports the geometries that were actually measured. (Two records carry
*unscored dev-window* numbers at 17 — e.g. `gen_dense_prime` reads 27.2–32.3 µs at
L=17 B=4 m=8 — but those were never run against a library in the same window and are
not board results.)

**(b) The scoring machine is not Cascade Lake.** The brief describes a Xeon Gold 5218
(Cascade Lake, downclocked AVX-512, 1 MB L2). `environment.txt` says
`Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` — **Ice Lake-SP**, 1.25 MB L2/core. The ISA
line confirms it independently: `avx512_vbmi avx512_vnni avx512_vpopcntdq avx512_bitalg
avx512ifma` are Ice Lake features absent on Cascade Lake. And this round *measured* the
part directly (§4): **two** 512-bit FMA pipes at 2.0 zmm-FMA/cycle, core running
~3.26 GHz under sustained 512-bit license-2 load. There is no AVX-512 downclock penalty
to attribute anything to here.

**(c) The implementers developed on this same node**, not on Sapphire Rapids. Every
strategy record this round uses the panel's standing protocol — one held slot lease on
`a80n0`, same-core interleaved control-first pairs. SPR ("wallaby") appears only as a
scalar-build and pre-race sanity host. Consequently the machine-difference explanation
offered in the brief's item 4 does not apply to this round's discrepancies; the real
causes are identified in §4 and they are not hardware.

**(d) Only one cell ran non-batched.** L=100 is B=1; the other ten cells are batched
only (B = 64/64/32/32/16/16/16/8/8/4). There is no non-batched row for L = 10…50 in this
round, so the "both cases" split the brief asks for exists only at L=100.

---

## 1. Headline per geometry: fastest correct panel entry vs. best library

"Best library" = min over MKL 2022, MKL 2026, FFTW estimate/measure/patient/guru, ducc0.
`baseline_matrix` is excluded — per `docs/CURATION.md` it is the harness floor, not a
library. All entries listed passed correctness (§3).

| L | B | fastest correct panel entry | µs/transform | best library | µs/transform | speed-up |
|---|---|---|---|---|---|---|
| 10 | 64 | `gen_race` → **`gen_batchlane`** | **1.146** (1.148 direct) | mkl_dfti | 4.577 | **3.99×** |
| 12 | 64 | **`gen_pfa_small`** | **1.911** | mkl_dfti | 7.770 | **4.07×** |
| 15 | 32 | `gen_race` → **`gen_batchlane`** | **4.374** (4.376 direct) | mkl_dfti | 16.502 | **3.77×** |
| 20 | 32 | **`gen_batchlane`** | **12.770** | fftw3_measure | 44.947 | **3.52×** |
| 25 | 16 | `gen_race` (self / planner trunk) | **40.083** | fftw3_measure | 108.681 | **2.71×** |
| 27 | 16 | **`gen_powp`** | **43.357** | mkl_dfti | 144.242 | **3.33×** |
| 31 | 16 | **`gen_rader`** | **84.745** | ducc0_c2c | 717.096 | **8.46×** |
| 32 | 8 | `gen_race` → **`gen_pow2`** | **55.326** (56.455 direct) | mkl_dfti | 173.351 | **3.13×** |
| 40 | 8 | **`gen_pfa_large`** | **159.708** | mkl_dfti | 404.384 | **2.53×** |
| 50 | 4 | `gen_race` → **`gen_pfa_large`** | **410.896** (416.640 direct) | mkl_dfti | 946.694 | **2.30×** |
| 100 | **1 (non-batched)** | `gen_race` → **`gen_powp`** | **4562.285** | mkl_dfti | 7802.441 | **1.71×** |

**Read `gen_race`'s rows correctly.** `gen_race` is a routing layer, not a kernel. Its new
`eng8` stage compiles the class entries as shared objects at plan time, races them as whole
graded chains, and ships the winner by vtable forwarding. The wisdom file records what it
actually shipped at leaderboard time:

```
L=10  B=64  gen_batchlane.a015f14e   (tie, +0.04%)      L=27  B=16  gen_powp.c309e387     (+58.1%)
L=12  B=64  gen_batchlane.a015f14e   (tie, -0.16%)      L=31  B=16  gen_rader.8819da78    (+28.4%)
L=15  B=32  gen_batchlane.a015f14e   (tie, +0.72%)      L=32  B=8   gen_pow2.a310b7a0     (+76.0%)
L=20  B=32  gen_batchlane.a015f14e   (tie, +0.61%)      L=40  B=8   self                  (margin 0)
L=25  B=16  self                     (margin 0)         L=50  B=4   gen_pfa_large.ebf25eee (+2.2%)
                                                        L=100 B=1   gen_powp.c309e387     (tie, -0.3%)
```

So at 10/15/32/50/100 `gen_race`'s board line **is** the class engine it forwards to,
measured through one extra indirect call. The 0.2–2.0% by which `gen_race` reads *faster*
than the engine it is running is measurement noise, and at L=32 `gen_race` carries a
**26.5% run spread** against `gen_pow2`'s 1.0% — the same code in a dirtier window. Do
not read those five rows as `gen_race` outperforming the engines. Read them as the routing
working. Where it did **not** work is §2 and §4.

The libraries are essentially unchanged from r7 (mkl_dfti moved ≤ 1.6% at every one of the
eleven cells), so r7 and r8 are directly comparable and the deltas in §2 are real.

---

## 2. What changed since gen_r7, per geometry

| L | r7 best | r8 best | Δ | verdict |
|---|---|---|---|---|
| 10 | 1.147 `gen_batchlane` | 1.146 `gen_race`→batchlane | −0.1% | flat |
| 12 | 1.915 `gen_batchlane` | 1.911 `gen_pfa_small` | −0.2% | flat |
| 15 | 4.381 `gen_batchlane` | 4.374 `gen_race`→batchlane | −0.2% | flat |
| 20 | 12.855 `gen_batchlane` | 12.770 `gen_batchlane` | −0.7% | flat |
| **25** | **30.882 `gen_powp`** | **40.083 `gen_race`** | **+29.8%** | **REGRESSION** |
| 27 | 43.966 `gen_powp` | 43.357 `gen_powp` | −1.4% | small win |
| 31 | 84.544 `gen_rader` | 84.745 `gen_rader` | +0.2% | flat (bit-identical) |
| 32 | 56.378 `gen_pow2` | 55.326 `gen_race`→pow2 | −1.9% | flat (bit-identical) |
| 40 | 159.959 `gen_pfa_large` | 159.708 `gen_pfa_large` | −0.2% | flat (instruction-identical) |
| 50 | 413.958 `gen_pfa_large` | 410.896 `gen_race`→pfa_large | −0.7% | flat |
| **100** | 4475.279 `gen_pfa_large` | 4562.285 `gen_race`→powp | **+1.9%** | mild regression |

### The L=25 regression is the round's headline, and it is not noise

Two independent, both-documented plan-time race failures stacked in the same cell.

**(i) `gen_powp` shipped the wrong candidate at L=25.** Its own record states the mechanism
outright: *"a late-session tryout cold race (busy neighbors) stored l25-ip0 as a 'tie'
(ip0 trial 46.77 us/vol, soa playoff margin -0.7%) and the graded chain shipped 41.1 us
where soa's same-day quiet number is 31.4-31.8."* The wisdom file confirms this is exactly
what the leaderboard run shipped:

```
gen_powp/chain6/L25/B16#6bb92654  winner = l25-ip0   tie = 1   margin = -0.005719   us = 46.6593
gen_powp/chain6/L27/B16#2e449ac0  winner = l27-soa   tie = 0   margin = +0.1682     us = 43.8234
```

L=27 picked `soa` correctly and scored 43.357. L=25 picked `ip0` and scored **41.025**
with a **0.0% run spread** — repeatably, deterministically the slow candidate. The
tie-doctrine accepted a −0.57% margin from a contended window as a verdict.

**(ii) `gen_race`'s cross-class race never ran at L=25.** There is **no**
`gen_race/enggate8/L25/B16/*` key in the wisdom file — gates exist for every other cell
that has a class arm (L10/L12/L15/L20 batchlane+pfa_small, L27 powp, L31 rader+dense_prime,
L32 pow2, L50 pfa_large+powp, L100 pfa_large+powp) but none at L=25. The `eng8` verdict is
`self`, `margin 0`: no foreign arm was raced at all. `gen_race`'s own record predicts this
failure precisely — *"a cold-cache first create races WITHOUT the heavyweight arms and its
verdict then STICKS for the round on that (L,B)"* — against measured gcc times of
**115 s for gen_powp** and a **30 s** poll budget.

The sweep timestamps show the pattern cleanly: **each heavyweight engine is skipped at
the first cell that needs it, and works from the next cell on.**

| engine | .so compile | first cell needing it | result | next cell | result |
|---|---|---|---|---|---|
| `gen_powp` | 115 s | L=25 (16:36) | **skipped** → self | L=27 (16:52) | gated, raced, won +58% |
| `gen_pfa_large` | 83 s | L=40 (17:58) | **skipped** → self | L=50 (18:23) | gated, raced, won +2.2% |

Both failures had to be corrected to recover r7's number: even if `gen_race` had raced
`gen_powp`, the arm it would have raced was the poisoned 41.0 µs one, which loses to
`gen_race`'s own 40.083. The cell needed `gen_powp` to pick `soa` *and* `gen_race` to see it.

**Cost:** L=25's best went from 30.882 to 40.083. Against the best library the cell fell
from 3.52× (r7) to **2.71×**. The `gen_race` L=40 skip cost that entry 241.810 vs the
160.19 it measured in its own window (+51%), though it did not move the cell's headline
because `gen_pfa_large` scored directly.

### L=100

`gen_pfa_large` ships `.text` byte-identical to r7 (verified by `objcopy` compare in its
record) yet reads 4475.279 → 4567.719 (+2.1%). That is window drift in the DRAM regime
(1.7–2.0% spread on a 30.5 MiB working set). A further 0.6% came from the `eng8` tie
doctrine handing L=100 to `gen_powp` (4596.649) over `gen_pfa_large` (4567.719) on a
−0.3% margin. `gen_race`'s record flagged this cell in advance: *"If the r8 board
disagrees with my window, this cell is where to look."* It does; it is small.

### Entry-level regressions inside otherwise-flat cells

* `gen_race` L=40: 224.856 → 241.810 (+7.5%) — the `eng8` cold-cache skip above.
* `gen_layout` L=10 4.738 → 4.888 (+3.2%), L=12 7.819 → 8.092 (+3.5%), L=50 920.627 →
  943.965 (+2.5%). Its record reports all three as wins or flat; see §4.
* `gen_twiddle` L=10 +2.2%, L=12 +1.9%, L=20 +1.4% — its **disclosed** code-layout tax at
  gate-off sizes, documented with 15 A/B pairs across three builds and shipped knowingly
  because the gated-on cells (27 −1.3%, 50 −2.7%, 100 −6.2%) dwarf it. Honest.
* `gen_dense_prime` L=15 14.520 → 15.218 (+4.8%); its record calls 15 a wash
  (14.72–15.60 vs control 14.78–15.42). Board landed at the top of its own band.

### Genuine engine improvements this round

* `gen_dense_prime` L=31: 120.347 → **113.507**, −5.7%, exactly the claim, bit-identical
  outputs. The 24-accumulator GEMM tile (rotating broadcast register, ~29 live zmm, zero
  spills) refuted its own r7 "boundary count is invariant under any 32-register tiling".
* `gen_twiddle` L=100: 7797.055 → **7317.669**, −6.2%; L=50 −2.7%; L=27 −1.3%. Split-group
  axis-1→2 handoff, axis-2 gather deleted, bit-identical.
* `gen_layout` L=25 98.515 → **94.095** (−4.5%), L=32 154.051 → **149.143** (−3.2%),
  L=20 35.789 → **34.200** (−4.4%). Insert-load 8×8 transpose + exit spill diet.
* `gen_rader` off-board class duty: **−25.5% at p=127**, −25% at 107, −20% at 59, −15% at
  23, with PMU attribution (load uops −23%, l1d.replacement −19%).
* `gen_pfa_small` B=1 / B%8 at the 64 generic sizes: **4.5–5.6× over r7**, and now beats
  MKL at B=1 (L=14 1.14×, L=21 1.77×, L=34 3.40×, L=44 1.11×) where r7 lost 3–4×.
* `gen_batchlane`: DFT11 module → four new class sizes 22/33/44/55, all beating MKL
  (3.1× / 3.7× / 2.5× / 1.5×).
* `gen_pow2`: L=128 (G=16) DRAM-regime custody engine, 63 ms → **13.42 ms/step-vol**
  (4.7×), 1.60× MKL on the chain.

---

## 3. Correctness, build, crash and coverage audit

**No entry failed correctness.** All twelve exited 0 (`agents/exits.txt`). Every panel row
on every one of the eleven cells reports `ok`, with chain error 2.1e-14…2.0e-13 against a
**1e-10** tolerance and single-step error 7e-16…4e-15. No `NOT REPEATABLE` flag anywhere,
which matters more than usual this round: `gen_race` now dlopens foreign engines, and the
driver's two independent processes still produced bit-identical chains at all twelve cases.

**Adversarial check on "a fast wrong answer must not survive":** in every cell the fastest
entry's chain error sits inside the same band as the rest of the field and as the libraries
(L=10: race 1.1e-13, batchlane 1.1e-13, pfa_small 8.8e-14, mkl 1.4e-13, fftw 1.1e-13). No
entry is fast *and* anomalously loose, and none is anomalously tight in a way that would
suggest it computed something cheaper than the specified transform. `gen_bluestein` runs
~2× looser than the field (2.0e-13 at L=10 vs 1.1e-13) — expected for chirp-Z's longer
convolution, still ~500× inside tolerance, and it is the slowest entry in every cell it
enters, so nothing rides on it.

**Nothing crashed or hung among the panel entries.**

### `failures.txt` — one entry, and it is not a panel entry

```
baseline_matrix L=100 B=1 run=1 exited 124
baseline_matrix L=100 B=1 run=2 exited 124
baseline_matrix L=100 B=1 run=3 exited 124
```

Exit 124 is a timeout, all three runs. `baseline_matrix` is the O(L⁴)/volume/axis harness
floor, which `CURATION.md` classifies as the library-free reference the harness is
validated with. It reads 33.96 ms/transform at L=50; at L=100 the same scaling puts it
past 250 ms/transform × m=64, well past the driver timeout. Expected, not a defect — but
it does mean **L=100 has no harness floor row**, so the L=100 column is validated only by
`check.py` against the numpy oracle, not by a cross-check against the dense reference.

### `build_errors.txt` — no build failures; two warnings, both benign

```
impl/gen_dense_prime.c:1913  iteration 1152921504606846976 invokes undefined behavior
impl/gen_rader.c:2226        iteration 1152921504606846976 invokes undefined behavior
```

Both are `-Waggressive-loop-optimizations` on the **scalar tail of `map_volume`**, which is
identical code in both files:

```c
static void map_volume(const cplx *z, const cplx *restrict c, cplx *o, size_t npts)
...
    for (; i < npts; ++i) {
        double re = zp[2 * i] + cp[2 * i];   /* <-- warned line */
```

`i` and `npts` are `size_t`, so there is no signed overflow. What GCC is flagging is that
`zp + 2*i` is only defined while the byte offset `16*i` stays inside the object, and its
trip-count analysis notes the loop *could* reach i = 2⁶⁰. The largest `npts` any scored
geometry produces is L³·B = 10⁶ at L=100 — sixty orders of magnitude short. **This cannot
fire on any cell in this suite and is not a correctness risk.** It should still be silenced
(hoist to a stepping pointer, or bound the tail explicitly) so that a real diagnostic is
not lost in the noise next round. I am promoting both files with this recorded.

### Coverage gap: `gen_pfa_small` silently declines L=40

`t_gen_pfa_small_L40_B8_r1.json` is `{"supported":false}`. But **40 = 5 × 8**, and both
modules are in its set — `gfactor()` (impl_8/gen_pfa_small.c:1811) already carries
24 = 3×8, 48 = 3×16, 56 = 7×8, 72 = 8×9, 88 = 8×11, 104 = 8×13, 120 = 8×15. There is no
`case 40:`. Its own strategy record claims *"all coprime P*Q in 14..127 except 50/80/100
(pfa_large/powp cells)"* — 40 is an **undocumented** hole, not a declared exception. This
cost L=40 a second structurally-different challenger to `gen_pfa_large`; the fix is one
line. Not a correctness failure, but the record overstates the entry's coverage.

Everything else absent from a cell is a correct, declared `supports()` restriction
(`pow2` off non-2^k, `rader`/`dense_prime` off composites, `pfa_large` below L=30,
`batchlane` off its twelve announced sizes, `powp` off non-p^k except 50/100).

---

## 4. Claimed vs. measured

**The headline finding here is negative in the useful sense: with three exceptions, every
entry reproduced its own claim on the board to within 1–2%.** The panel's held-lease
same-core interleaved-pair protocol transfers to the scoring window. Because implementers
and the scoring pass ran on the *same* Ice Lake-SP node (§0c), **none** of the three
exceptions is attributable to a machine difference, and I decline to attribute them to one.

**Reproduced (representative):**

| entry | claimed | board | |
|---|---|---|---|
| `gen_dense_prime` L=31 | 113.11–115.07 (−5.7%) | 113.507 (−5.7%) | exact |
| `gen_rader` L=31 | 85.27–85.88 | 84.745 | board quieter |
| `gen_pow2` L=32 | 56.37 quiet | 56.455 | +0.2% |
| `gen_batchlane` L=10 | 1.147–1.148 | 1.148 | exact |
| `gen_twiddle` L=50 / L=100 | 725.3–730.6 / 7354–7451 | 725.371 / 7317.669 | exact / better |
| `gen_pfa_large` 40/50/100 | 161.3 / 421.2 / 4832 | 159.708 / 416.640 / 4567.719 | board quieter |
| `gen_pfa_small` 10/15/20 | 1.229 / 4.414 / 13.162 ("noisy window") | 1.152 / 4.416 / 13.048 | board quieter |
| `gen_layout` L=25 | 93.1–95.3 | 94.095 | exact |

**Diverged — three cases, all plan-time race misfires, none of them hardware:**

1. **`gen_powp` at L=25: claimed 31.4–31.8, measured 41.025 (+29%).** Cause is in the
   wisdom file, quoted in §2: its own `chain6` race stored `l25-ip0` as a tie on a
   −0.57% margin from a contended window. The entry documented this itself before the
   board ran, and explicitly warned against "fixing" it by pre-storing a `soa` verdict —
   that discipline is right, and the fix belongs in the race, not the record.
2. **`gen_race` at L=25: claimed 30.958, measured 40.083 (+29%).** Cause: no `enggate8`
   key at L=25 → the `gen_powp` arm never compiled in time and `self` stuck for the round.
3. **`gen_race` at L=40: claimed 160.19, measured 241.810 (+51%).** Same mechanism with
   `gen_pfa_large`.

**One bookkeeping problem worth naming.** `gen_layout`'s r8 table carries a column labelled
"r7 board" that is in fact a same-window rebuilt r7 control arm, and it runs up to 4.5%
slow. At L=10 it cites 4.98 (the real r7 board is 4.738 — 4.976 is `fftw3_measure`'s r7
number); at L=12 it cites 8.00–8.02 (real: 7.819); at L=50 it cites 961.7 (real: 920.627).
The consequence is that **three cells it recorded as flat or as −1.9% wins are +2.5% to
+3.5% regressions on the board.** Its L=20/25/27/31/32/40/100 citations are correct, and
its two real wins (25, 32) are real. Fix the column label; the entry's method is sound.

---

## 5. Which open question from `docs/LITERATURE.md` §4 this round moved

### Moved decisively: **§4.8 item 6 — AVX-512 on Ice Lake-SP server parts**

That item closes with the corpus admitting it has no primary data and instructing:
*"there is no primary measurement in the corpus for Ice Lake-SP or later **server** parts,
which is the hardware most likely to be in a current LQCD cluster. **Measure it on the
node.**"* This round did exactly that, and the answer inverted the tooling's assumption.

`gen_pfa_large` built a leased-core rdtscp microbenchmark
(`build/tryout/gen_pfa_large/portcal2.c`, left for the panel) and measured the Gold 6326:

* **zmm FMA: latency 4, throughput 2.0/cycle** — two real 512-bit pipes, ports 0 **and** 5.
* **zmm shuffle (vshufpd): latency 1, throughput 1.0/cycle** (port 5).
* Mixed 8 independent FMA + K independent shuffles: K = 4/8/12 → 6.0/8.0/12.0 cycles —
  exactly the p0/p5 model. **Shuffles steal FMA slots 1:1 past 2 uops/cycle.**
* Core clock ~3.26 GHz under sustained 512-bit load; `gen_powp`'s PMU run confirms
  **512-bit license-2 for ~100% of cycles** with no observed downclock cliff.

Four entries independently corroborated it (`gen_pfa_large`, `gen_pow2`, `gen_powp`,
`gen_rader`), and it **falsifies llvm-mca's `icelake-server` model**, which dispatches all
512-bit FP to port 0 and models `vdivpd zmm` at 16-cycle reciprocal throughput (real ICL
≈ 8). `gen_pow2` cross-checked with OSACA, which has the correct dual-pipe model
(xcol1: P0 = 484 / P5 = 484, port floor 484 cyc/column vs 695 measured).

The practical consequence, now measured rather than inferred: on this part the
**shuffle/FMA trade is 1:1 on port 5**, which is precisely why the round's three biggest
wins are all shuffle-deletion or broadcast-sharing at *identical FLOP count* —
`gen_layout`'s insert-load transpose (48→32 shuffles per block, VINSERTF64X4-from-memory
being a pure load-port uop), `gen_twiddle`'s deleted axis-2 gather (−16 shuffles, −16
stores, −16 loads per 64 complex), `gen_rader`'s paired-column chunks (broadcast traffic
halved, −25.5% at p=127). §4.8.6's historical text worried about "licence-based
downclocking … severe"; on Ice Lake-SP the binding resource is **total uops against a
~2.1 vector-uop/cycle dispatch cap**, not frequency. `gen_pfa_large` states it flatly
after four rounds of asking "port 5 or DRAM?": *"the answer is NEITHER: total uops."*

### Moved secondarily: **§4.6 — model versus search for the instruction schedule**

§4.6 records §01's claim that "you should not need a search phase" against §06's
correction that the spill-optimality proof holds only at powers of 4 and the schedule is
therefore "the primary thing to search". This round ran the experiment with static
analyzers for the first time, and the answer is **neither pole**: *models filter, the node
scores*.

* Models earned their keep as **relative** filters. `gen_bluestein` used llvm-mca to price
  a DFT-13 convolution grid (conv_mid13 460 cyc/block vs conv_mid14 313, +47%) and
  predicted the kill **before spending a lease**; the node then confirmed 3/3 pairs, +10%.
  `gen_powp` killed a split-complex DFT5 rewrite with one `-S` compile and four greps
  (+72 zmm reg-reg movs), node-confirmed 6/6 pairs, +2.9%. `gen_dense_prime` and
  `gen_layout` used them to *choose* schedules that then won on the node.
* Models are unusable as **absolute** predictors here (the port-map error above), and
  two of the three advertised tools do not run on this cluster at all: **uiCA is broken**
  on every entry that tried it (`ext/tools/uiCA` has no `instrData/`; `setup2.log` shows
  the uops.info `instructions.xml` download timing out — no outbound net from the nodes),
  and **OSACA is partially broken** (`osaca-pkg/osaca/` has only `data/parser/semantics`,
  no top-level modules) though `gen_pow2` got the ICX model to run. This is a tooling
  defect the panel should fix or stop advertising.
* §06 survives on the substance: **every shipped win this round moved zero FLOPs.** The
  schedule and the instruction stream *were* the lever, at every size.

### Also moved: **§4.1 — register liveness**, from both sides

§4.1 asks how much spill traffic batch-vectorised codelets actually generate and warns
that "2L is a data-only lower bound, not a budget". Two entries answered with numbers,
and they cut in opposite directions:

* `gen_dense_prime` **raised** the accumulator count past the assumed 16-register ceiling
  and got −5.7%: a 6-column × 4-zmm group holds 24 accumulators at ~29 live zmm because a
  broadcast consumed immediately costs no register. Its lesson, verbatim: *"'fixed by the
  register file' arguments must count **live ranges**, not named values."* An automated
  innermost-loop scan of the whole `.s` confirmed **zero spills** in every instantiation —
  which is exactly §07 §7.8's prescribed check ("count stack traffic before believing any
  timing"), finally run.
* `gen_powp` measured the **cost side**: a DFT5 form with 10% *fewer* FP ops lost 2.9%
  because every folded output FMA is destructive and its addend is live in the ± partner,
  forcing +72 zmm copies that Ice Lake does **not** eliminate at rename. Its lesson:
  *"in split complex, 'FP op count' is the wrong metric — **port uops** is the right one."*
  It is a genuine cross-arch fork: the same form is a candidate *win* on Golden Cove,
  which does eliminate vector movs (368 vs 442 p05 uops, −17%).

### Touched: **§4.3's re-opened L2↔DRAM regime**

The §4.3 note re-opens axis fusion at the L2↔DRAM boundary and calls "tile the batch so a
tile fits L2, then run all three axes inside the tile" *"the largest untried structural
move on the board."* Partial answer: `gen_planner` fused axes 0+1 per z-slab at @s1/@s3
and measured a **7–12% LOSS**, 3/3 pairs, at every size it engages (25: +10–12%,
31: +7%, 20: wash-to-loss, 15: +2%). The mechanism is specific and worth keeping: the
unfused pass walks pencils contiguously (L perfectly-sequential streams — ideal L2-streamer
food); the slab replaces that with pencil walks strided 128·L bytes (3200 B at L=25), past
the ICX streamer's ~2 KB reach, turning every line into a demand miss. Meanwhile
`gen_pow2`'s L=128 engine — the same construction done as **custody** rather than as
pass-merging — won 4.7× over generic. The distinction the round draws: what pays in the
DRAM regime is keeping the working set resident *across chain steps*, not reducing the
pass count *within* one step. Tolmachev's rule (§07 §1.6) survives with the streamer's
stride reach as a new side condition.

---

## 6. The single highest-value thing gen_r9 should attack, per geometry

| L | highest-value attack |
|---|---|
| **25** | **Fix the two race misfires — it is free and it is worth 23%.** (a) `gen_powp`'s `chain6` tie doctrine must not accept a −0.6% margin from a contended window; require a quiet-window playoff or refuse to store on a tie. (b) `gen_race` must not freeze a `self` verdict when the `.so` cache is cold: prewarm the cache before the sweep, or let `GEN_RACE_REFRESH` fire when the cache completes. Recovering `soa` alone restores ~31 µs and the cell's 3.5× library ratio, with **zero new kernel code**. Nothing else on the board pays this well. |
| **40** | **Restore the field.** One line — `case 40: *P = 5; *Q = 8; return 1;` in `gen_pfa_small`'s `gfactor()` — puts a second structurally-different engine into a cell that currently has one, and closes the undocumented coverage hole in §3. Second: `eng8` must actually gate `gen_pfa_large` here (it shipped `self` at 241.810 against a 159.708 engine, a 51% miss). |
| **10, 12, 15, 20** | **The B=1 / small-batch lane-spatial engine** — named as the top unbuilt structure by `gen_batchlane` (3 rounds), `gen_planner` (8 rounds) and `gen_pfa_small`. `gen_pfa_small` closed it this round for the 64 *generic* sizes (4.5–5.6×, now beating MKL at B=1); the tuned 10/12/15/20 paths and `gen_batchlane` still lane-replicate. Port `gstep_split` + the `GS_DEF` pencil family into the tuned paths and into `execute()`. The batched kernels themselves are saturated — three rounds bit-identical; do not chase them. |
| **27** | The 1.39× gap between `gen_planner` (60.241) and `gen_powp` (43.357) is **real arithmetic, not scheduling** — `gen_powp` measures 60% port utilisation and calls it saturated on three independent verdicts. The named lever is a **Winograd DFT9/DFT27 module** in `gen_planner`'s fused codelets, which sidesteps twiddles entirely (`gen_powp`'s 3-shear twiddles will not transfer to runtime-table codelets — analysed and declined this round). |
| **31** | `gen_rader` has been port-model-saturated since r4 and bit-identical for four rounds. The movable term is `gen_dense_prime`'s **z-phase**: ~30 µs against a ~20 µs floor, running 0.75 loads/FMA. A 3-row-pair variant is the same 24-accumulator lever one layer down; the reversed-half masked stores make the drain hairy, so cost it with OSACA before spending a window. |
| **32** | **Protect it, attack nothing.** The residual is now *model-attributed*, not merely measured: `[port ∥ L2]`, with the L2 share structurally non-overlappable because a ~350-entry ROB cannot span a 1.3 K-uop column body. All three shapes of fix were raced out in r2/r3 (+8%, +4%, +7 K cycles). If 2^k *single calls* ever score, the open lever is `gen_pow2`'s L=128 conversion fusion (~35% of 16.7 ms). |
| **50** | The cell is a 1.3% coin flip between `gen_pfa_large` and `gen_powp`, both of which declare themselves saturated. **Make the tie deterministic** (a playoff, as at L=25) rather than chasing 1%. The one non-saturated term anywhere is the map's L3-bandwidth share — 29% of the cell, three streams past 2 MB — and `c` is read-once-per-step by contract, so a real idea would have to cut the stream count itself. |
| **100** | Narrowest margin on the board (1.71×) and the **only non-batched cell**. Both attributions agree arithmetic will not move it: PMU shows LLC misses of only 9–14 MB/step against a 32 MB state+c (custody works) at 45% port utilisation, and the transform subpasses sit at the node's ~2.1 vector-uop/cycle dispatch cap. The honest lever is **fewer FLOPs per butterfly** — split-radix (2,8) chains in the radix-4 bodies — plus the B=1 engine above, since this cell is where the panel's oldest unbuilt structure is actually scored. |
| **all** | **Harness debt, seventh round running.** `tryout.sh`'s remote map-check leg still dies on an unexpanded `'$W/c.bin'`, so every entry runs its gates by hand; `reserve.sh --status` needs a slurm PATH shim; `tryout.sh` now fails *silently* without it. And `uiCA` has never worked on this cluster (missing `instrData/`) while `OSACA` is half-installed — drop uops.info's `instructions.xml` in from a machine with net access, or stop listing them in `TOOLS.md`. |

---

## 7. What to keep

Applying `docs/CURATION.md`'s four grounds, in order. Note first that the entries are
**mutually dependent at build time** — `gen_batchlane`, `gen_rader`, `gen_powp` and
`gen_twiddle` all `#include "gen_layout.c"`; `gen_pfa_large`, `gen_powp` and `gen_planner`
all `#include "gen_race.c"`; `gen_race` and `gen_planner` include each other;
`gen_bluestein` includes `gen_twiddle.c`. An exemplars directory that drops a library layer
leaves four winners that will not compile.

**1. Fastest correct entry per geometry (mandatory):**
`gen_batchlane` (10, 15, 20) · `gen_pfa_small` (12) · `gen_powp` (27, and the *true* L=25
winner at ~31 µs) · `gen_rader` (31) · `gen_pow2` (32) · `gen_pfa_large` (40, 50, 100) ·
`gen_race` (holds the board's top line at 10, 15, 25, 32, 50, 100 as the routing layer) ·
`gen_planner` (its trunk **is** the engine behind `gen_race`'s self-verdict at L=25 and 40).

**2. Structurally different runner-up:** `gen_dense_prime` at L=31 — 113.507 vs Rader's
84.745. At 1.34× it is outside the doc's ~20% yardstick, but it is the *only* structurally
different prime kernel on the board, it moved −5.7% this round, and it qualifies
independently under ground 4 (6.3× the best library at L=31). `gen_powp` at L=50 (1.3%
behind `gen_pfa_large`, CT-p^k vs GT-PFA) is the cleanest ground-2 case and is already in.

**3. Instructive failures, promoted *with* the number that killed them:**
* `gen_race` — the `eng8` cold-cache skip: 115 s and 83 s compiles against a 30 s poll,
  costing L=25 and L=40. The mechanism is in the shipped source, not just the prose.
* `gen_powp` — the tie-doctrine poisoning: `l25-ip0` stored at margin −0.57%, 41.0 µs
  shipped where 31.4–31.8 was available.
* `gen_planner` — the z-slab fusion refutation, 7–12% loss 3/3 with the streamer-stride
  mechanism; the `LITERATURE` §4.3 datum.
* `gen_bluestein` — the DFT-13 / M=208 grid: model-predicted, node-confirmed, +10% 3/3.
  The killed machinery is **still in the shipped source behind `-DBST_M13`**, so promoting
  the code preserves the corpse; the record alone would not.
* `gen_pfa_large` — its corpse (the rolled-codelet frontend diet, +11% at 50) was reverted
  and survives only in prose, but the entry carries `portcal2.c`, the machine calibration
  that corrected every static model on the panel (§5). That alone earns it twice over.

**4. Beat a library baseline, regardless of rank:** all twelve did somewhere, so this
ground alone does not discriminate. It is decisive for two that no other ground reaches:
`gen_twiddle` (L=25 80.971 vs 108.681; L=31 267.735 vs 717.096; L=100 7317.669 vs
7802.441) and `gen_bluestein` (L=31 292.564 vs ducc0's 717.096 — 2.45×, and 2.9× MKL),
which is also the only chirp-Z on the board and the any-L fallback for sizes outside every
other class. `gen_layout` clears it at 20/25/32 and is in any case a hard build dependency
of four promoted entries.

**Excluded:** nothing. Every entry passed correctness, every entry has a strategy record,
and none is a near-duplicate of another — this is a library-assembly campaign whose twelve
files are a dependency-closed set covering every L class plus four shared layers. Promoting
a proper subset would hand the next panel a reading list that does not build. I record that
this is a near-total promotion and that the discriminating judgement in this round went
into §2–§4, not into thinning the roster.

**Two conditions to carry into `NOTES.md` at promotion time:**
1. `gen_powp`'s and `gen_race`'s promoted records must state on their face that the L=25
   board number is a race artifact, not the engine's speed — otherwise r9 reads 41.0 µs as
   `gen_powp`'s L=25 capability and re-derives a solved cell.
2. The `-Waggressive-loop-optimizations` warning in `gen_dense_prime.c:1913` and
   `gen_rader.c:2226` should be silenced before those files are held up as exemplars.

PROMOTE: gen_batchlane gen_pfa_small gen_powp gen_rader gen_pow2 gen_pfa_large gen_race gen_planner gen_dense_prime gen_twiddle gen_layout gen_bluestein
