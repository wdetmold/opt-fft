# VERDICT — panel round `panel_r9`

Monitor's judgement on the measurements taken on `p55n3` (Intel Xeon Gold 5218, Cascade
Lake, 2×16c, exclusive, slurm job 438526, 2026-08-22T04:36), gcc 11.4.0,
`-O3 -march=native -mtune=native -fno-math-errno -funroll-loops`, `governor=powersave`.
Roster: **19 implementations plus the harness floor, all built, all ran, all correct, none
missing, none crashed.** Sources are in `bench/geom/impl_9/` (`impl` → `impl_9`); all 19
differ from `impl_8` (55–512 changed lines each), and `baseline_matrix.c` is byte-identical,
as it must be.

**Three things about this round's shape before any numbers.**

1. **This was the round in which the panel stopped guessing.** Eleven of the nineteen
   entries shipped a discriminating measurement inside `fft3d_create()` and routed it out
   through `fft3d_description()` — the pattern r8 §6 asked the whole panel to adopt. Every
   one of them produced a readable node number, and **seven of them read negative against
   their own pre-registered hypothesis.** The round's product is not speed; it is four
   theories killed with node evidence and one killed *and replaced* at L=6.
2. **The monitor owes the panel a correction, and it is the largest single finding here.**
   Three entries independently built raw `perf_event_open` probes into their plans, and all
   three came back `na` on the scoring node (`pmc=na`, `fe=na`, `fe=na`). `perf_event_paranoid`
   is 4 on the compute node as well as every dev host. **The consolidated counter list that
   r8 §6 declared "the item that outranks all eight" — five entries, four rounds of asking —
   was never obtainable.** It is withdrawn, permanently. Timed in-plan discrimination is the
   only instrument this cluster permits, and it worked: see §5.
3. **Minima and distributions disagree in four headline cells, and this round the
   distributions are decisive rather than merely cautionary.** `leaderboard.txt` reports the
   minimum over three processes; §1 quotes it, and then quotes the test that actually
   separates entries — whether one entry's *worst* per-run median beats the other's *best*.
   That test flips L=6 B=64, dissolves L=17 B=2048 and L=36 entirely, and turns L=6 B=1 from
   a four-round coin-flip into a 9% decision.

---

## 1. Headline per geometry — fastest correct panel entry vs. the best library

Times are µs per transform. "min" is the leaderboard number (minimum over three independent
processes). "med" is the median of the three per-run medians. "Best library" is whichever of
FFTW ×3 / MKL 2022 / MKL 2026 / ducc0 was fastest **in that exact cell**. **Every panel entry
in every cell beat every library**, including the third-place arms — so CURATION rule 4 is
satisfied by all nineteen and cannot by itself decide promotion (§7).

### L = 6 (volume 216) — non-batched decided outright for the first time

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L6_pfa 0.2068** (med 0.2152) | L6_unrolled 0.2174 (med 0.2257) | mkl_dfti 0.3733 | **1.80×** |
| B=64 (0.42 MiB) | L6_unrolled 0.2142 / **L6_pfa 0.2150** | — see below | mkl_dfti 0.3909 | **1.82×** |
| B=4096 (27 MiB) | **L6_pfa 0.3954** (unrolled 0.3957) | tie | mkl_dfti 0.5602 | **1.42×** |
| B=32768 (216 MiB) | **L6_unrolled 0.5656** (pfa 0.5733) | — | mkl_dfti 0.7001 | **1.24×** |

**B=1 is not a coin flip any more.** L6_pfa's three processes returned per-transform minima
of 0.206833 / 0.206777 / 0.206808 µs — three independent processes agreeing to **0.03%** —
and its *worst* per-run median (0.2153) is below L6_unrolled's *best* per-run median (0.2174).
The distributions do not overlap. On minima the gap is 5.1%; on medians it is 9.1%. After
four rounds in which this cell was decided by which entry drew the lucky process, and three
in which the leaderboard read opposite to the medians, **L6_pfa owns L=6 B=1.**

**B=64 inverts, and the inversion is also non-overlapping.** The leaderboard hands the cell to
L6_unrolled by 0.4% on minima (0.2142 vs 0.2150). But L6_pfa's per-run medians are
0.2214/0.2214/0.2214 and L6_unrolled's are 0.2222/0.2222/0.2231 — again disjoint, the other
way. **On distributions L6_pfa owns B=64 as well**, and it got there by closing a three-round
3% deficit (r8 medians 0.2281 vs 0.2221) in one move. §2 names the move.

Reading all four cells on distributions: **L6_pfa owns B=1 and B=64, B=4096 is a genuine tie
(0.3988–0.4099 against 0.3991–0.4033), and L6_unrolled owns B=32768** (worst median 0.5712
against pfa's best 0.5742, disjoint). That is **3–1, not the 2–2 r8 recorded.** It is the
input the slot decision needs (§6).

### L = 8 (volume 512) — B=1 is a dead tie; the streaming cells changed hands

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L8_batchsimd 0.5527** / **L8_fusedaxes 0.5530** — tie | L8_radix8 0.5784 | mkl_dfti 0.6511 | **1.18×** |
| B=64 (1.00 MiB) | **L8_fusedaxes 0.5805** (med 0.5932) | batchsimd 0.5938 (med 0.6087) | mkl_dfti 0.7196 | **1.24×** |
| B=2048 (32 MiB) | **L8_fusedaxes 0.9473** (med 0.9646) | radix8 0.9774, batchsimd 0.9782 | mkl2026_dfti 1.3425 | **1.42×** |
| B=16384 (256 MiB) | **L8_fusedaxes 1.2510** (med 1.2535) | batchsimd 1.2599 | mkl2026_dfti 1.7915 | **1.43×** |

**B=1 is now indistinguishable.** batchsimd 0.5577/0.5589/0.5605 against fusedaxes
0.5581/0.5581/0.5583 — fully overlapping, 0.05% apart on minima. L8_batchsimd closed a 2.2%
gap to zero and eliminated its 0.5935 tail (spread 5.2% → 1.1%) by a pre-registered
mechanism; §2 and §5 give it.

**B=2048 and B=16384 are L8_fusedaxes' on distributions**, both disjoint from the field
(B=2048: fusedaxes 0.9575–0.9718 against batchsimd 0.9949–1.0103 and radix8 0.9944–1.0142).
B=2048 was L8_batchsimd's in r8. It did not lose it to a better rival — it regressed into
third; §2.

### L = 17 (volume 4913) — L17_matrixsimd takes all four cells

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L17_matrixsimd 14.866** (med 15.055) | L17_winograd 16.460, L17_rader 16.543 | fftw3_patient 81.695 | **5.50×** |
| B=8 (1.20 MiB) | **L17_matrixsimd 16.512** | winograd 17.508 | fftw3_measure 81.949 | **4.96×** |
| B=256 (38 MiB) | **L17_matrixsimd 21.073** | winograd 21.512 | fftw3_estimate 83.447 | **3.96×** |
| B=2048 (307 MiB) | **L17_winograd 21.716 / L17_matrixsimd 21.718 — tie** | rader 24.983 | fftw3_estimate 84.071 | **3.87×** |

B=2048 is a dead tie on any reading: 2 ns apart on minima (0.008%), and the per-run medians
overlap completely (winograd 21.778–21.909, matrixsimd 21.852–22.010). **L17_winograd's one
cell from r8 does not survive the distributions**, so L=17 is L17_matrixsimd's in all four
cells, three of them by non-overlapping margins. L=17 remains the panel's largest margin over
the state of the art anywhere on the board — 5.50× FFTW and 6.65× MKL at B=1.

### L = 36 (volume 46656) — no cell is separated

| case | fastest panel entry | runner-up | best library | margin |
|---|---|---|---|---|
| **B=1 (non-batched)** | **L36_mixedradix 120.478** (med 121.724) | L36_pfa 122.576 (med 123.522) | mkl_dfti 160.994 | **1.34×** |
| B=4 (5.70 MiB) | **L36_mixedradix 128.957** (med 132.207) | pencilfused 130.993, pfa 131.953 | mkl_dfti 175.247 | **1.36×** |
| B=32 (45.6 MiB) | **L36_pfa 168.253** (med 169.495) | mixedradix 168.357 (med 170.067) | mkl_dfti 261.493 | **1.55×** |
| B=256 (365 MiB) | **L36_mixedradix 184.140** (med 186.741) | pfa 185.818 (med 186.841) | mkl_dfti 309.424 | **1.68×** |

**Every L=36 cell is a tie once the runs are read.** B=1: mixedradix 121.18–123.00 against
pfa 122.87–131.58 — overlapping (mixedradix ahead by 0.1% at the boundary). B=4: three-way
overlap, 129.27–132.66 / 131.82–133.51 / 132.14–184.41. B=32: 169.13–171.01 against
169.42–170.28 — pfa's 0.06% leaderboard win is not real. B=256: 184.44–187.56 against
186.40–186.88, and the medians are 186.218 vs 186.223 — a dead tie, so **B=256 did not
actually change hands from L36_pfa to L36_mixedradix.** Three arms, four cells, zero
separated cells: that is the argument in §6.

### The four wave-2 geometries

| L | case | fastest panel entry | best library | margin |
|---|---|---|---|---|
| 13 | B=1 | **L13_direct 5.739** (rader 6.030) | mkl2026 7.595 | 1.32× |
| 13 | B=16 | **L13_direct 5.957** (rader 6.963) | mkl2026 7.651 | 1.28× |
| 13 | B=512 | **L13_direct 7.965** (rader 9.055) | mkl2026 9.107 | **1.14× — thinnest cell on the board** |
| 23 | B=1 | **L23_rader 47.796** (matrixsimd 47.945 — tie) | fftw3_estimate 261.10 | 5.46× |
| 23 | B=4 | **L23_rader 49.524** (matrixsimd 49.632 — tie) | fftw3_estimate 261.55 | 5.28× |
| 23 | B=128 | **L23_rader 64.882** (matrixsimd 65.112 — tie) | fftw3_patient 264.93 | 4.08× |
| 45 | B=1 | **L45_pfa 315.898** (mixedradix 317.518 — tie) | mkl_dfti 606.17 | 1.92× |
| 45 | B=2 | **L45_pfa 324.441** (mixedradix 325.971 — tie) | mkl_dfti 607.22 | 1.87× |
| 45 | B=16 | **L45_pfa 402.154** (mixedradix 417.913) | mkl_dfti 753.34 | 1.87× |
| 64 | B=1 | **L64_radix8 952.944** (blocked 1098.376) | mkl_dfti 1192.05 | 1.25× |
| 64 | B=2 | **L64_radix8 1021.050** (blocked 1179.955) | mkl_dfti 1265.23 | 1.24× |
| 64 | B=8 | **L64_radix8 1249.923** (blocked 1304.242) | fftw3_patient 2405.75 | 1.93× |

Both L=23 cells and L=45 B=1/B=2 are overlapping distributions — consistent with r8's finding
that each of those geometries is one algorithm implemented twice. L=13 and L=64 are
non-overlapping in every cell.

### Distance from each geometry's own port floor, B=1, at the measured 2.89 GHz

Floors as published by the entries themselves.

| L | floor | r8 | **r9** | move |
|---|---|---|---|---|
| 6 | 486 cy = 0.1682 µs (two 256-bit FMA ports) | 1.29× | **1.23×** | **the only geometry that moved toward its floor by more than 1%** |
| 8 | 1248 cy = 0.4318 µs | 1.28× | 1.28× | flat, sixth round |
| 13 | 4.7 µs | 1.22× | 1.22× | flat |
| 17 | 33 374 cy = 11.548 µs | 1.31× | **1.29×** | first movement in six rounds, and §5 explains why it is the last |
| 23 | 41.9 µs | 1.14× | 1.14× | tightest on the board; closed |
| 36 | ~82.7 µs | 1.45× | **1.46×** | away from the floor, second consecutive round |
| 45 | **188 µs** (was 199 — the floor itself fell, see §2) | 1.61× | **1.68×** | away, because the floor fell faster than the time |
| 64 | ~550 µs | 1.76× | **1.73×** | still the board's worst |

---

## 2. What changed since `panel_r8`, per geometry — and what regressed

### Cell-level medians, r8 → r9, panel best in the cell (the four brief geometries)

Only moves ≥1% on medians are listed. Six gains, seven regressions.

| gain | | regression | |
|---|---|---|---|
| L6_pfa B=1 | **−6.2%** | L8_batchsimd B=2048 | **+5.2%** |
| L6_pfa B=64 | **−3.0%** | L8_radix8 B=64 | +3.1% |
| L17_rader B=2048 | −2.8% | L8_batchsimd B=64 | +2.5% |
| L17_matrixsimd B=1 | −2.1% | L8_radix8 B=2048 | +1.5% |
| L17_rader B=256 | −1.5% | L36_mixedradix B=32 | +1.2% |
| L8_batchsimd B=1 | −1.0% | L36_mixedradix B=4 | +1.0% |
| | | L8_fusedaxes B=16384 | +1.0% |

**Five of the seven regressions are at L=8, in the round L=8 shipped its one positive
mechanism.** That is not a coincidence and §2's L=8 entry attributes it.

### L = 6 — the one mechanism that worked, found by two entries running the same A/B backwards

This is the round's headline result and it is worth stating precisely, because both entries
pre-registered it and both got a clean answer.

The 6-point PFA 2×3 codelet is arithmetically closed — 48 flops / 36 scalar-shaped
instructions, provably optimal since round 1, and identical in every unit between the two
entries (12 add/sub + 6 FMA-class + 2 shuffles per vector codelet, depth-4 critical path).
The only remaining freedom is **association order**:

* `VD6` / `_d2` — **radix-2-first**: the three DFT2s run first, then two conjugate DFT3s
  whose FMA results feed the stores directly. The last op before every store is an FMA.
* `DFT6V` / `fused3` — **radix-3-first**: two conjugate DFT3s over the CRT groups, then six
  add/sub joins at the end feeding the stores. The last op before every store is an add.

L6_pfa adopted L6_unrolled's radix-2-first graph as `_d2` twins, placed after the incumbents
behind a 1.5% takeover margin. L6_unrolled adopted L6_pfa's radix-3-first graph as `fused3`
twins inside its own kernels, behind a 2.5% margin, and shipped a licence-fair `create()`-time
A/B (`ab1`, `f3d`) so the node would publish the delta whichever way the picks fell.

**The node's answer, from both sides:**

* L6_pfa's tuner picked **`fused_d2` at B=1 (3/3)** and **`fused_pf_xa_d2` at B=64 (3/3)**.
  B=1 went 0.2205 → **0.2068** median (−6.2%), B=64 went 0.2281 → **0.2213** (−3.0%). Its own
  pre-registered criterion — "if the node picks a d2 twin and B=64 drops to ≤0.222, the
  factorization was the three-round gap" — fired exactly.
* L6_unrolled's `f3d` field (fused3-family best vs fused-family best, from the node's own
  tournament) read **+3.3% to +6.2%** at B=1 and B=64 across six readings, and its `ab1`
  nvol=1 A/B read **f = 212.5–221.3 ns against f3 = 225.2–228.6 ns**, every reading. Its
  pre-registered branch (i) — "f3d ≥ +2% confirms the codelet-ordering hypothesis; L6_pfa
  should adopt VD6" — fired, against its own stated bet of (ii) f3d ≈ 0.

So: **on Cascade Lake, store-feeding FMAs beat store-feeding adds by 3–6% on identical
arithmetic**, and that is the whole of the L=6 B=64 gap that survived six rounds of other
explanations. The mechanism is now in both files.

**And the last surviving B=1 theory died in the same round.** L6_pfa's boundary probe severed
the cross-pass store→load joint at identical instruction counts (`bsp` = the dependence-broken
twin of `bf`) and measured, at B=1: **bf = 219.6–220.9 ns, bsp = 219.0–220.8 ns → the joint
costs +0.1 to +0.6 ns**, against a pre-registered "45–55 ns if the t1 joint is the B=1
overhead". And `bx + byz` = 66.0 + 155.2 = 221.2 ≈ `bf` — no fusion-scheduling deficit either.
Both branches read zero. That is the **seventh** falsified L=6 mechanism (r4 port-5/uop mix,
r5 OoO window, r7 width/uop count, r8 pinning, r8 scratch rotation, `restrict`, and now the
t1 joint). L6_unrolled's `ab1` closed the eighth from the other side: `zf/f` = **1.14–1.18**,
i.e. the zmm kernel with 25% fewer uops at the same FP floor is 14–18% *slower*, so the
front-end/uop theory is dead with a number printed on every leaderboard line.

L=6 B=1 is now 1.23× its port floor and there is no named mechanism left anywhere in either
record.

### L = 8 — the alias mechanism paid at B=1 and was charged for at B=2048

L8_batchsimd shipped two allocation-side changes and removed one tournament:

1. `SI` moved from `scr+512` to `scr+520` doubles, breaking an **exact 4096-byte** relation
   between the two halves of its own scratch (`-DL8_SI_OFF=512` restores it);
2. the arena page-aligned (`posix_memalign(4096)`, was 64);
3. **B=1 runs no tournament at all** — FUSED/plain/no-pf hardwired, because its `create()`
   arena had picked LANEX3 in 4 of 6 B=1 creates across r7/r8 while every LANEX3-picked driver
   run was 3–6% slower.

Result: B=1 0.5647 → **0.5588** median (−1.0%), minimum 0.5643 → **0.5527** (−2.1%), and the
0.5935-class tail is gone (spread 5.2% → 1.1%, pick string reads `fixed, no tuner` 3/3). Its
pre-registered branch (b) — "min ≤0.558 → the SI/page changes had node value" — fired. Wallaby
could not resolve the SI change at all (three alternating pairs, 0.330/0.330, 0.645/0.645,
0.330/0.330 — dead even).

**The bill arrived at the streaming cells.** B=2048 went 0.9314 → **0.9800** median (+5.2%,
+7.2% on minima) and B=64 0.5910 → 0.6056 (+2.5%), against a record that describes the
streaming candidate sets and defaults as byte-identical to r8. The two changes that are *not*
byte-identical — the page-aligned arena and the +64 B `SI` displacement — apply to every mode,
not just B=1. **The plausible and testable attribution is that the alias fix bought 1% at B=1
and cost 5% at B=2048**, and the entry already ships the one-flag A/B (`-DL8_SI_OFF=512`, plus
reverting the 64-byte alignment) that settles it. This is the only unexplained ≥5% number at
L=8 and it is the geometry's next item.

L8_fusedaxes changed nothing that mattered: its `B1DIRECT` dispatch shortcut (~10–20 cycles of
a ~1600-cycle call) landed at 0.5530 against a prediction of 0.546–0.552 — flat. Its real
deliverable was the PMC probe, which returned `pmc=na` (§5).

L8_radix8 reverted its B=1 default `1f` → `2p` on the strength of r8's own node pick strings
and predicted "0.570 ± 0.002, pick = 2p (default) 3/3". **It got 0.5784 (+1.4% regression) and
2p in only 1 of 3 runs** — the arena picked `1f` twice, and the arena string now on the
leaderboard shows why: the node's `create()` arena reads **1f 2–5% *faster* than 2p**
(`arena{2p=0.603 1f=0.572}`, `{2p=0.587 1f=0.574}`) while the driver reads 1f **0.6% slower**
(0.5813/0.5829 against 0.5784). That is the arena-inverts-the-driver finding the entry asked
for, obtained in one binary, with the fix already demonstrated one file over. Its own
pre-registered prediction failed on both the number and the pick.

### L = 17 — the residual was located, and it is not reachable

Three arms, three in-plan phase decompositions, one answer.

**L17_matrixsimd's `b1dec` is the decisive measurement of the round at this geometry.** It
times the *identical instruction stream* twice, once walking the real 78.6 KiB volume
(L2-resident) and once with the stride arguments collapsed so the same code re-reads one
4.6 KiB plane (L1-hot). Node readings, three processes:

```
b1dec[yz / kyz / x / kx] = 10.90 / 10.30 / 4.08 / 4.03   µs per volume-equivalent
port floors at 2.89 GHz:    7.83   7.83   3.71   3.71
```

* `yz / kyz` = **1.06**, `x / kx` = **1.01** — going from L2 to L1-hot buys 1–6%. **Memory
  fill latency is not the residual.**
* `kyz / floor` = **1.30–1.34×** and `kx / floor` = **1.09×**. The plane kernel is a third
  above its port floor *from L1, with nothing to miss*.
* `yz + x` = 14.98 ≈ the scored B=1 time of 14.87. The phases add; there is no boundary cost.

The entry's pre-registered branch (b) fired exactly: "kyz ≳ 9.5 → the kernel is >1.25× its
port floor from L1 → front-end/dependency-bound → B=1 is at its structural limit for this
kernel family." Its `pt` mechanism — the one stall class nothing had attacked, same-volume
in-pass load prefetch — was offered in all four cells and the node's tuner chose **pt=0 in
every one**. The 1.31× that L=17 B=1 has read for six rounds is now positively identified as
intra-kernel issue limitation, uniformly distributed across the phases. That is the first
positive identification of this residual in nine rounds.

**L17_winograd's `p1/f23/fu` probe closed its own rewrite gate, negative.** Node: p1 = 6.17,
f23 = 10.26, fu = 16.50; `p1 + f23` = 16.43 ≈ `fu`, and **p1/fu = 37.3–38.0%** against pass 1's
33.3% FP share. Its pre-registered gate was "p1/fu ≥ 42% funds the interleaved-complex
rewrite; ~36% or below means the residual is inside the fused groups and B=1 is closed for
this family." 37% is below the gate. **The interleaved-complex rewrite — the last named idea
in the corpus for L=17 — is not funded by measurement**, and the node agrees with wallaby (36%)
to within 1.5 points, which is itself notable given how little else transferred.

**L17_rader recovered its r8 batch regression and lost its bet.** `dy` (ymm deinterleave tiles
inside the otherwise-zmm pipeline, aimed at cold split-loads on a 2-load-port machine) was
picked at B=8/B=256/B=2048 and B=2048 went 25.904 → **25.179** median (−2.8%), B=256 25.012 →
24.631 (−1.5%) — the split-load story confirmed at reduced magnitude, matching its own
fallback branch. Its headline bet, the joint (variant, pf, pfw) grid, was **declined 6/6**: the
node picked `pf=0 pfw=0` in every batched run. §4 prices what that cost.

L17_matrixsimd's B=1 also improved 2.1% with **pt=0 picked and no scored path changed** — the
inverse of the refactor-regression disease the panel has documented four times. Both signs of
that ±2% now have instances; it is code-layout noise, not mechanism, and §3 says so.

### L = 36 — the front-end fork closed negative, three independent ways, in one round

r8's verdict ordered this geometry to "stop looking at caches; go at the front end — code
size, not caches. If MITE dominates, the correct move is shrinking the unrolled bodies, which
inverts four rounds of instinct." All three arms did exactly that, by three different
mechanisms, and **all three read null or negative on the node.**

1. **L36_mixedradix — `roll`, a 3.5× static shrink.** Unrolled exec bodies of 1922–2287
   instructions (~14 KB) re-expressed as two short DSB-resident loops per codelet, verified
   still rolled under `-funroll-loops` at 575–592 instructions. Node probe, printed on the
   leaderboard: **`unrolled = 148.8 / 149.2 / 152.8`, `rolled = 184.7 / 185.3 / 182.8` µs/vol
   at B=1** — rolled is **+22–24% slower**, and +23% at B=4. The tuner declined it in every
   cell. Its own branch (c) fired: "probe ratio ≥ 1 on the node too — the front-end theory is
   dead at this granularity alongside the cache theory." Note the sign: it predicted 95–112 µs
   if the MITE story held; the node measured the *opposite* direction at *larger* magnitude
   than wallaby's +16–21%.
2. **L36_pfa — `fug − fu`, a direct 2× walked-footprint A/B taken by the node itself.** It
   compile-time-specialised the pf=0 hot path (phase1 1482 → 1336 instrs, phase2 1357 → 633;
   per-volume-loop 2839 → 1969) and then forced the *general* bodies through an opaque-register
   flag launder so the comparison could not be constant-propagated away. Node: **fug − fu =
   +0.4 / +0.2 / +0.3 µs on ~123 µs — 0.3%**, against a pre-registered +2 to +6 µs. Its own
   branch: "if ≈ 0, code size is NOT the B=1 residual and the front-end fork closes negative."
   Doubling the instruction footprint through a 16 B/cycle MITE costs 0.3%.
3. **L36_pencilfused — `ip4` vs `cs4`, halving the hot footprint by code sharing.** It found
   that its two pass-A subloops are the same code (PST == 36) and replaced two
   610-instruction copies with one shared `halfplane()`, plus an unroll-fenced `passB_small`;
   B=1 hot footprint 6.9+7.1 KB → 3.4+3.1 KB, output bit-identical. Node probe:
   **cs4 − ip4 = +3.0 / +0.4 / −0.6 µs — flat**, picks split 2:1 on a coin flip, against a
   prediction that CLX's 1.5k-µop DSB would make the gap *larger* than wallaby's −3%. Its own
   words: "122–125 with `probe us ip4 ≈ cs4` is the clean null branch — and that null would
   kill the code-size theory for L=36 outright." B=1 landed at 123.657.

With r8's probe having already killed the cache story (`fu − p1 − p2w ≈ −3 µs`), **both named
theories at L=36 B=1 are now dead by node measurement**, and the cell has moved away from its
floor for two consecutive rounds (1.43× → 1.45× → 1.46×).

Two attribution corrections fall out, and both were wrong in *my* r8 verdict:

* **`pin` was not the cause of r8's +1.2% B=1 regression.** L36_mixedradix defaulted it off on
  my pricing; B=1 still read 120.478 (r7 was 118.532). The regression did not revert.
* **The `if (pfd)` tile-loop branch was not the cause of r8's +2.1% B=4 regression.**
  L36_pfa deleted it from the B=4 body and predicted recovery to ≤131; B=4 read **131.953**.
  By its own pre-registered criterion, "the r8 regression attribution was wrong twice."
* Likewise L36_pencilfused removed the r8 `optimize("unroll-loops")` pragma expecting the
  +1.7% B=256 regression to revert; B=256 read 189.323 against r7's 186.452. It did not.

Three unexplained ~1–2% regressions at unchanged picks, three failed attributions. §3 names
the honest conclusion.

### The wave-2 geometries

**L = 13.** L13_direct shipped three pure, bit-exact deletions — the r8 pragma out, an
all-pinned ymm tail kernel (−810 L1 loads/volume), and the −i sign folded into the sine tables
(**−882 vector XORs/volume**, off contended ports p0/p5). Wallaby measured −3.8 / −4.0 / −4.7%;
the node delivered **+0.2 / −1.4 / −1.8%** at B=1/B=16/B=512, all three slightly outside its
own predicted bands. Third data point for the rule that only deletions transfer, and modestly.
It owns all three cells for the third round.

L13_rader built the round's largest structural bet: z blocks port-fused with y lane-blocks
(each 450-uop port-homogeneous z block now carries a y lane-block interleaved at source-stage
level, so every 48-shuffle burst has ≥54 FMAs inside one ROB span), with a plan-time reuse
verifier and 8-deep U buffering. Its quantitative model predicted 4.7–5.0 µs at B=1 and it
pre-registered "≥5.9 falsifies the model on CLX." **Measured: 6.030 at B=1 (−0.7%), and
regressions of +4.2% at B=16 and +2.8% at B=512.** The ROB-serialization model is falsified by
its author's own criterion; `-DL13R_FUSE=0` is the rollback.

**L = 23.** The round achieved its stated protocol goal and corrected my r8 attribution.
L23_rader's deterministic joint-cell hysteretic tuner delivered **timed == checked, identical
pick cells 3/3, in all three cells** (`pick=` and `inc=` now printed per run). Levels flat as
designed: 47.796 / 49.524 / 64.882 against r8's 47.688 / 49.557 / 64.835. But I had instructed
the panel to lock `rp-t1 + pf=2 + pw=1` as the B=128 incumbent so it would be picked 3/3; they
did, and **the node's own tuner then displaced `rp` with plain two-sweep, 3/3, by 3.4%**
(`pick=59.54–60.20` against `inc=61.63–62.36` µs/t in the arena) at an unchanged cell time.
**The r8 B=128 win was the knobs (pf=2, pw=1), not the folded-pair layout.** L23_matrixsimd
reached the same place independently — its own za-vs-plain variant pick still flips at B=128
(plain in run 1, za in runs 2–3), so its stated determinism goal was not met, and the knob
combination is what is stable. Both entries remain bit-identical and overlapping in all three
cells: one algorithm twice, at 1.14× floor, closed.

**L = 45.** L45_pfa transcribed genfft's `n1_9` FMA DAG from the FFTW source on this
filesystem — **44 → 40 FMA-port vector ops per DFT9, −5.5% of the volume's port-0 budget**,
correct on the first build (rel_l2 improved to 3.99e-16), with the transcription rule written
out mechanically in the record. It pre-registered "≤312 confirms port-0 sensitivity; ≥318
means the −5.5% bought nothing." **Measured 315.898 — between the branches: a 5.5% arithmetic
cut bought 1.2%.** The floor fell from 199 to 188 µs, so the floor ratio got *worse*
(1.61× → 1.68×). The honest reading is that **port 0 is not the binding resource at L=45 B=1**,
and the phase probe says where the time is instead: `p1 = 243.9, p2w = 79.9, fu = 319.2` —
**phase 1 is 76% of B=1**, and p1 + p2w ≈ 1.01×fu, so the phases add here too. L45_pfa took all
three cells.

L45_mixedradix's odd-column insert/extract rework (−8.1k instructions, −5.4k shuffles, bit-
identical) landed B=1 at 317.518 as predicted, and **regressed B=16 by 2.0%** (409.808 →
417.913) at an unchanged pick, outside its own 400–410 band. It is the only change on that
path. Two of its findings matter more than its numbers: `#pragma GCC target` **adds** to
`-march` rather than restricting it, so its "v0" AVX2 kernel has compiled as EVEX and been
ICF-folded into the AVX-512 body in every scored binary since r6 (its r6 width A/B was
measuring identical code — retracted), and wallaby's slow state is **per-core and invisible in
the MKL sentinel**, so "MKL is fast so the window is fast" is an invalid inference.

**L = 64.** L64_radix8 took all three cells for the fifth round and improved B=1 by 1.4% to
952.944 (1.73× floor, still the board's worst). The interesting part is which mechanism paid:
**`propf` — the prologue prefetch I named in r8 §6 as this entry's move — read a coin flip**
(`pro0` in two runs, `pro1` in one; its own create-time A/B, ±0). **`p1pf` — pass-1 next-plane
prefetch, which the entry's own r6 wallaby gate had killed at −2.3% — was picked by the node
at B=1 (`p11`, 3/3)** and the cell landed at the low end of its predicted band. A wallaby gate
reversed by the node, and my named directive priced at zero.

L64_blocked's `st=2` x-first 2-sweep — a genuine structural rewrite moving ~9 MB of exposure
from L3 to an L2-hot octet buffer, with a corrected traffic ledger showing its 3-sweep and the
rival's 2-sweep move identical bytes — was **declined 3/3**; the node picked `cached pf=0
st=0(3-sweep)` in every cell and B=1 read 1098.376 against a "≤1030 if the theory is right"
threshold. By its own pre-registration, `st=2` joins `st=1` in the dead list. That file has
now had two structural bets refused and is 4–15% behind in all three cells for the fifth
round.

---

## 3. Adversarial pass: failures, correctness, and what the harness did *not* prove

**Nothing failed, and I checked this rather than assuming it.**

* `build_errors.txt` is **0 bytes**; the slurm log shows all 20 implementations and all 6
  library backends compiling cleanly with `-march=native` on the node itself.
* `failures.txt` **does not exist**. `sweep.sh` creates it on any exit code other than 0 or 3
  (3 = "geometry not supported"), so no entry crashed, hung, or hit the 600 s timeout.
* `timing.err` is 48 KB of exactly 240 benign `does not support L=n` lines and nothing else.
* `check.log` is **259 lines, 259 of them `PASS`, zero anything else** — every backend × every
  supported cell, against `numpy.fft.fftn` on the same input file, tolerance 1e-12. Worst
  observed rel_l2 anywhere is 8.0e-16 (`baseline_matrix`); the worst panel entry is
  `L64_radix8` at 4.5e-16. All 259 `c_*.json` files carry `"ok": true`.
* **The anti-memoization gate is live and untripped.** `driver.c:169–188` copies the output,
  perturbs `in[0]`, re-executes, and returns exit 5 if the output does not change. Zero
  occurrences in any log, and no exit-5 entries in the (absent) failures file. **No fast wrong
  answer survived, because none was offered.**
* Inputs are regenerated per round from `--seed SEED + L*1000 + B`, so no entry could have
  been tuned to a cached dataset even if the gate were absent.

### (a) Timed ≠ checked: the exposure is still open, and this round it happened to be benign

`sweep.sh` runs each backend three times, reports the **minimum**, and then checks
correctness **on the output of the last run only**. Where a plan's tuner picks a different
variant in run 3 than in the run that produced the minimum, the validated bits are not the
timed bits.

Pick strings differ across the three runs in **44 of the 76 panel cells**. I checked every
one against the entries' own bit-class records. **In no cell did the minimum run and the
checked run use different bit classes.** Two calls were close:

* **L8_radix8 at B=1 and B=2048 does span genuine bit classes** — its own forced runs give 2p
  = 1.308e-16, 1f = 2.269e-16, 3p = 1.874e-16. At B=1 the arena picked 1f in runs 1–2 and 2p
  in run 3; at B=2048 it picked 1f in run 2 and 3p in runs 1 and 3. In both cells run 3
  happened to be the minimum (0.5784 and 0.9774), so the reported number and the reported
  digit agree — `c_L8_radix8_L8_B1.json` reads 1.3197e-16, the 2p fingerprint, matching. **That
  is luck, not design.**
* **L8_fusedaxes at B=64**: the reported 0.5805 came from `fused+pfs` (run 1) while the checked
  run ran **`fusedAA+pfs`** at 0.5953. The entry's own `cmp` record establishes cross-variant
  bit-identity, so this is a labelling defect rather than a correctness hole — but it means the
  cell's headline number was produced by a configuration that was not the one validated in
  that process. Worth noting separately: this is the **first time the node's own tournament has
  selected an address-aware variant at L=8**, after L8_batchsimd declined to port `fusedAA` on
  the explicit grounds that "the node's own tournament declined them both times."

The structural fix remains what it was in r8: check every run, or check the run that produced
the reported minimum. Until then this exposure is one unlucky pick away from putting an
unvalidated number in a leaderboard.

### (b) Four headline cells rest on a minimum that is an outlier against their own runs

| entry | cell | three runs (µs) | spread | median reading |
|---|---|---|---|---|
| **L36_pfa** | **B=4** | 131.95 / **184.11** / 132.48 | **39.5%** | 132.48 — third, not second |
| L36_pfa | B=1 | 122.58 / 131.06 / 122.64 | 6.9% | 122.64 |
| L8_radix8 | B=64 | 0.5971 / 0.6369 / 0.6646 | 11.3% | 0.6369 |
| L8_batchsimd | B=64 | 0.5938 / 0.6056 / 0.6416 | 8.1% | 0.6056 |
| L6_unrolled | B=1 | 0.2174 / 0.2257 / 0.2305 | 6.1% | 0.2257 |

**L36_pfa's B=4 is the worst measurement on the board.** A 184.1 µs run against two runs at
132 µs is a 40% outlier, and it is the only one of its kind anywhere in this round. The
leaderboard prints 131.953 with a `39.5%` spread column that is easy to skim past. **That cell
should not be read as 131.95**, and no L=36 conclusion should be drawn from L36_pfa's B=1 or
B=4 numbers until the pick lottery behind them is closed. Everything else on the board sits at
0.1–2.9%.

### (c) The medians change four headline readings

Three in the panel's favour and one against:

* **L=6 B=64** flips from L6_unrolled (by 0.4% on minima) to **L6_pfa** (by 0.4% on medians),
  and the distributions are disjoint both times — so the leaderboard has the wrong winner.
* **L=17 B=2048** goes from "L17_winograd holds its one cell by 0.008%" to a fully overlapping
  tie. Winograd holds no cell outright.
* **L=36 B=256** goes from "L36_mixedradix takes it by 0.9%" to a dead tie (186.218 vs
  186.223) — the cell did not change hands.
* **L=36 B=4** goes from a 2.3% ordering to a three-way tie within 0.9%.
* **L=6 B=1** goes the other way and gets *stronger*: 5.1% on minima, 9.1% on medians,
  non-overlapping.

### (d) Three regressions at unchanged picks, and three failed attributions of last round's

L36_mixedradix B=1 (+0.4%, `pin` now off — the r8 attribution was wrong), L36_pfa B=4
(+0.4% from r8's already-regressed level, `pfd` branch deleted — its own criterion says the
attribution was wrong twice), L36_pencilfused B=256 (still 1.5% above r7, pragma removed).
Add L45_mixedradix B=16 (+2.0%) and L8_radix8 B=64/B=2048 (+3.1%/+1.5% with nothing changed on
those paths), and set against L17_matrixsimd B=1 (**−2.1% with nothing changed on that path
either**). The panel has now accumulated enough instances of both signs to draw the conclusion
it has been avoiding: **±2% at these cells is the code-layout noise floor of recompiling the
file, not a mechanism.** Reading a ±2% move as a result — in either direction — has cost the
panel at least four rounds of misattributed effort. Stop doing it, and stop building fixes for
it.

### (e) The library baselines are stable this round, and r7 was the outlier

r8's §3(a) warned that MKL had regressed 4–27% at the large-working-set cells and that five
margins were inflated. **r8 → r9 is stable at every cell** (largest library move among the
non-ducc0 backends is 3.1%). MKL 2022 at L=36 B=32 reads 219.4 / 261.3 / 261.5 across
r7/r8/r9, at L=36 B=256 246.5 / 310.2 / 309.4, at L=45 B=16 679.2 / 753.5 / 753.3, at L=64 B=8
1943.5 / 2477.9 / 2516.1. Two rounds now agree on the higher value, so **r7 was the anomaly and
the margins quoted in §1 are the honest ones.** One caveat stands: ducc0 at L=6 moved +30% at
B=1 (3.868 → 5.037) and +19% at B=64 while no other backend moved. ducc0 does no planning and
is never the best library in any cell, so nothing in §1 depends on it — but its measurement
quality at the smallest geometry is visibly poor and it should not be cited.

### (f) Five of the 28 cells have no library-free floor

`sweep.sh` skips `baseline_matrix` once `L³·B > 2·10⁶`, so L=6 B=32768, L=8 B=16384, L=17
B=2048, L=36 B=256 and L=64 B=8 have no from-scratch reference in the table. Unchanged from r8
and defensible on cost, but it means the largest cell at four of the eight geometries is
validated only by `check.py` and not anchored by the harness floor.

### (g) The clock consensus has one dissenter, and its fix was unavailable

Eight entries published a licence-clock measurement this round. Seven agree on **3.89 GHz
(256-bit/scalar) and 2.89 GHz (512-bit)** — L6_pfa, L6_unrolled, L17_matrixsimd, L17_rader,
L23_matrixsimd, L23_rader, and L17_winograd (which reads 2.89 on both, plausibly because its
256-bit probe runs mixed with 512-bit code). **L8_fusedaxes' FMA-chain proxy reads
`clk256s/p = 3.26/2.42` and `clk512s/p = 2.43/2.42`** — a 16–20% under-read against the
consensus. Its intended fix was the PMU `cycles` counter, which returned `na`. **Every
cycle-denominated claim in that file rests on the wrong clock**, and it should not be used as a
cross-check on anyone else's floor arithmetic until the proxy is replaced with something that
agrees with the other seven.

---

## 4. Claimed numbers versus measured numbers

Wallaby (Sapphire Rapids Gold 6448Y, full-clock AVX-512, 2 MB L2, 60 MB L3, DDR5, six-wide
decode, two 512-bit FMA units, ~512-entry ROB) against the scoring node (Cascade Lake Gold
5218, 2.89 GHz under AVX-512, 1 MB L2, 22 MB L3, DDR4, four-wide decode, **one** 512-bit FMA
unit, ~224-entry ROB).

### The translation ratio is not a constant, and it correlates with working set

Claimed wallaby B=1 number ÷ measured node B=1 number, this round:

| entry | ratio | | entry | ratio |
|---|---|---|---|---|
| L6_pfa | 1.58 | | L64_radix8 | 1.81 |
| L6_unrolled | 1.69 | | L45_pfa | 1.82 |
| L64_blocked | 1.70 | | L13_direct | 1.87 |
| L17_matrixsimd | 1.73 | | L17_rader | 1.87 |
| L8_fusedaxes | 1.75 | | L13_rader | 1.89 |
| L8_batchsimd | 1.68 | | L17_winograd | 2.10 |
| L45_mixedradix | 1.84 | | L23_rader | 2.18 |
| L8_radix8 | 1.88 | | L36_mixedradix | 2.22 |
| | | | L36_pfa | 2.23–2.32 |
| | | | L23_matrixsimd | 2.26 |
| | | | L36_pencilfused | 2.34 |

**1.58× to 2.34× — a 1.48× spread, ordered by working set.** L=6 and L=8 (L1-resident) sit at
1.6–1.9; L=23 and L=36 (L2/L3-pressured) sit at 2.2–2.34. The brief's note that MKL alone
spans 2.9× between these machines is the right frame: **a claimed wallaby number cannot be
translated to the node by any single constant**, and the two entries that anchored predictions
on a per-geometry ratio band (L45_pfa, L64_radix8) both landed inside their predicted ranges
while entries that reasoned from a wallaby *delta* mostly did not.

### Where a claimed number is far from measured, and whether the machine explains it

**Attributable to the machine difference — and the round's best result:**

* **L=6 codelet association, sign-inverted between the machines.** Both entries measured the
  radix-3-first ordering as **0.4–1.5% faster** on wallaby (`f3d` = −0.0 to −1.5%, `_d2` twins
  0.4–1.0% behind their parents). On the node it is **3.3–6.2% slower** (`f3d` = +3.3 to +6.2%,
  `ab1` f 212–221 ns vs f3 225–229 ns), and adopting the radix-2-first graph moved L6_pfa's
  B=1 median 6.2%. This is a clean sign inversion on arithmetic that is identical in every
  countable unit, and **L6_unrolled wrote down the correct physical reason in advance**: SPR
  runs FP adds at latency 3 on two extra ports, so a bottom-heavy add join is free there,
  whereas CLX has one store port and no such asymmetry, so store-feeding FMAs win. It bet on
  the wrong branch of its own hypothesis (it predicted f3d ≈ 0) but its mechanism was right.
  L6_pfa's diagnosis was blunter and also right: "wallaby is codelet-blind; the node decides
  this."
* **L=8 scratch de-aliasing.** Wallaby resolved nothing (three alternating pairs, dead even);
  the node paid −2.1% at B=1. SPR's alias penalty is smaller and its heap offsets differ. The
  entry said so before measuring.
* **L=45 DFT9 port-0 cut.** Predicted wallaby-flat (two 512-bit FMA units, port 0 not binding)
  and −11 µs on the node (one unit). Wallaby: 173.8 vs 171.8, flat as predicted. Node: −3.75 µs.
  **Direction right, magnitude one-third of the model.** The machine difference is real here and
  working as designed; the model over-credited it.
* **L=64 `p1pf`.** Declined on wallaby at −1.1% (its own r6 gate), picked by the node 3/3.
  Wallaby's 2 MB L2 and fast 60 MB L3 make the pass-1 L3 exposure nearly free; the node's do not.

**Not attributable to the machine — genuine mechanism-transfer failures, in cost order:**

* **L17_rader's joint grid: −12.4% claimed, 0 measured.** Wallaby's grid found `xl 512t sp +
  pf=1 + pfw=1` at 13.631 against the incumbent's 15.554 at (0,0) — −12.4%, and −7.9% against
  the incumbent's own best config. The node's grid, running the same 4-config × 2-variant
  search, picked **`pf=0 pfw=0` in all six batched runs.** The mechanism (smoothing a mixed
  read+RFO stream) targets DRAM behaviour, and the node's DRAM is *slower*, so the machine
  difference predicts a *larger* win, not zero. This is the round's largest single divergence
  and it is not a clock or cache artifact.
* **L17_winograd's `q4+pfw`: −5 to −8% claimed, −11.5% against the shipping config, 0
  measured.** Same mechanism class, same result: declined 6/6, cells flat (+0.3% / +0.6%). Two
  independent entries, two independent implementations, one shared conclusion — **the L=17
  write path does not respond to cross-volume input spreading plus write-intent prefetch on
  this machine, whatever it does on SPR.** Both entries pre-registered the null branch and both
  should now take it.
* **L13_rader's ROB port-fusion: −7.5% claimed at B=1, −0.7% measured, and +4.2% / +2.8% at
  batch.** The mechanism was designed *for the node* — CLX's 224-entry ROB against SPR's ~512
  is the entire premise — with a serial cost model that reproduced r8's measured 17.6k cycles
  and predicted 13.6–14.5k. It got 17.4k. **A model that fits the old data, predicts the new
  data, and is wrong** is the most expensive kind, and the entry's own falsification criterion
  (≥5.9 µs) fired cleanly, which is what makes the round's cost recoverable.
* **L36_mixedradix's `roll`: predicted 95–112 µs, measured probe +22–24% the wrong way.** Again
  the mechanism was node-specific by construction (CLX's ~1.5k-µop DSB against SPR's ~4k), and
  again the node measured the opposite sign at larger magnitude than the dev machine's own
  adverse reading. Combined with L36_pfa's `fug − fu` = 0.3% and L36_pencilfused's flat
  `cs4 − ip4`, the front-end theory is not merely unconfirmed — it is measured absent.
* **L64_blocked's `st=2`: predicted ≤1030 µs, measured 1098 with the variant declined 3/3.**
  Its wallaby table honestly showed `st=2` 2.5–4% behind at B=1 and ~11% behind at B=8 and it
  shipped anyway as a stated node bet. The bet lost. Recorded as designed.

**Claimed numbers retracted by static audit rather than by the machine** — the cleanest
methodological work in the round, and both from L45_mixedradix:

* **`#pragma GCC target` adds to `-march` instead of restricting it.** Its "v0" AVX2 body has
  compiled as EVEX with ymm16–31 on every AVX-512 build host and been folded by gcc's ICF into
  `jmp exec_0_*`, in the r6, r7 and r8 binaries as well as this one. Its r6 "width A/B: V0 253
  vs V2 190" was pure window noise on byte-identical code and is retracted. **Any entry that
  believes it ships a narrower per-function kernel on an AVX-512 build host should check the
  disassembly.**
* **L45_pfa's recorded "3087 instructions" for `x_ip0_pw4` does not reproduce under the
  Makefile's flag set** (3967 same-flags, against L45_mixedradix's 3855). L45_pfa
  independently reached the same correction. The two L=45 entries are equally lean to ±3%, and
  the leanness gap the panel has cited for two rounds did not exist at the size claimed.

**Predictions that landed, and they are the round's best work.** L6_pfa's B=64 d2 threshold
(≤0.222 → 0.2213), L6_unrolled's f3d branch (i) (≥+2% → +3.3 to +6.2%), L8_batchsimd's B=1
branch (b) (≤0.558 → 0.5527) with the tail eliminated as designed, L17_matrixsimd's `b1dec`
branch (b) (kyz ≳ 9.5 → 10.30, pt=0 picked in all four cells), L17_winograd's `p1/fu` gate
(<42% → 37.4%), L36_pfa's `fug − fu` null branch (≈0 → +0.3), L36_pencilfused's `ip4 ≈ cs4`
null branch with B=1 in its stated 122–125, L36_mixedradix's branch (c), L45_pfa's B=1 and
B=16 bands and both picks, L64_radix8's three cells and all three pick strings, L64_blocked's
`st=0`-holds branch, L23_rader's timed==checked goal, and L13_rader's own falsification
threshold. **Fourteen pre-registered branches fired correctly this round.** That is why a round
with almost no speed in it is nonetheless the most informative on record.

---

## 5. Which open question from `docs/LITERATURE.md` §4 moved

### Primary: **§4.6 — model versus search for the instruction schedule. Settled, with a number, and with a constraint the corpus does not state.**

§4.6 records genfft's claim that its cache-oblivious schedule is "no slower than
machine-specific codelets generated by SPIRAL", §06's correction that the spill-optimality
proof holds only for powers of 4 (**none of 6, 8, 17, 36 is one**), Frigo's own "it merely
works", and SPIRAL's measured **2× runtime spread among 10 000 arithmetically-equivalent
formulas on a transform small enough that no cache problems arise**. Its instruction was: "At
our sizes a near-exhaustive search over {schedule variant × unroll depth × batch-loop
placement × copy-or-not × compiler flags} costs minutes. Do it."

**The panel did it, at the one size where the arithmetic is provably closed, and it paid.**
This round is the corpus's missing CPU measurement of that spread, run from two directions on
the same kernel:

* The quantity varied is **association order alone**. Both codelets are 48 flops / 36
  instructions / 12 add-sub + 6 FMA-class + 2 shuffles per vector codelet / depth-4 critical
  path / identical register budget / 972 vector FP uops and 108 shuffles per volume. Nothing
  else differs — not the count, not the depth, not the width, not the addresses.
* **Measured spread on Cascade Lake: 3.3–6.2%** (L6_unrolled's `f3d`, four cells × three
  processes, plus `ab1` at nvol=1: f 212.5–221.3 ns against f3 225.2–228.6 ns). Realised as
  **−6.2% at B=1 and −3.0% at B=64** when L6_pfa adopted the winning order.
* **Measured spread on Sapphire Rapids: 0.0–1.5%, with the opposite sign.**

So §4.6's answer for this project is: **search, not model — the spread from schedule
association alone at our smallest size is ~5%, not 2×, but it is larger than every one of the
seven mechanisms falsified at this geometry across rounds 4–8 combined, and it is the only
thing that has moved L=6 in six rounds.** And the constraint the corpus does not state, which
this round supplies: **the search must be run on the scoring machine.** A dev host with a wider
front end, more FP ports and lower add latency reads the same A/B backwards. Every
schedule-search result in the literature is implicitly a per-microarchitecture result.

Cost of the experiment: one macro-parameterised codelet and three raced twins per entry —
about 40 lines, which L6_unrolled has written down as a reusable pattern.

### Second: **§4.5 — padding and 4 KiB aliasing. The first positive port, and the refinement that explains r8's 0-for-7.**

r8 recorded the L=23 padding result failing to generalise across seven ports at five
geometries and concluded the mechanism does not transfer. This round L8_batchsimd found `SR`
and `SI` **exactly 4096 B apart inside its own scratch**, applied §4.5's own rule (`+64 B`, so
the relation breaks for every same-index pair — the ducc0 `if ((dstride & 256) == 0) dstride
+= 16;` idiom in a different dress), page-aligned the arena, and took B=1 from a 0.5647 median
with a 0.5935 tail to a **0.5588 median with a 1.1% spread**.

The refinement that makes this compatible with r8's negative result: **the alias mechanism is
reachable when both colliding addresses are inside the plan's own scratch, and unreachable when
one of them is a driver buffer.** Five of r8's seven failed ports were aiming at
`(scratch − out)` residues the plan does not control — which is also exactly what L8_radix8's
`scr@0x540` / `scr@0x4c0` publication and L8_fusedaxes' allocation-lottery model were
independently circling. §4.5 should record the distinction; it is the difference between a
one-line fix and an unreachable one.

It also has a price: see §2's L=8 entry and the +5.2% at B=2048. A stride change that helps a
cache-resident cell can hurt a streaming one, and the entry ships the flag to A/B it.

### Third, partial: **§4.2 — L = 17. The part that can be closed by measurement is closed, negatively.**

§4.2's four positions (dense-symmetric, Rader with the alternate-convolution split, no
Winograd module at all, direct O(n²)) all argue about which 17-point module to build. The
panel built all three and this round measured *where the time actually is*, three times
independently on the node, and the answers agree:

* the phases add in all three structures (`yz + x` ≈ fu; `p1 + f23` ≈ fu; `ph + xp` ≈ fu ± 0.5
  at B=2048) — **no phase-boundary cost anywhere**;
* the kernel is **1.30–1.34× its port floor with its entire working set in L1**
  (L17_matrixsimd's `kyz` = 10.16–10.52 against 7.83), so the six-round 1.31× is intra-kernel
  issue limitation, not memory;
* pass 1's non-FP overweight is **+4 points** (37.4% of a 33.3% FP share), below the 42% gate
  L17_winograd pre-registered for the interleaved-complex rewrite.

**§4.2's practical answer: the module choice is worth ≤11% (nested dense over Rader), the
symmetric/antisymmetric convolution split adds nothing measurable, and the residual 1.29× is
not arithmetic-shaped, so no further module work is justified.** §4.2 item (c) — the missing
exact op count for a full 17-point Winograd module — remains unobtainable and no longer
matters.

### And the item that is not a §4 question but retires the panel's top priority

r8 §6 opened with "the item that outranks all eight of them": five outstanding one-run `perf
stat` asks across six geometries, four rounds of asking, named as "the binding constraint on
the whole board." **It was never obtainable.** `perf_event_paranoid` is 4 on the compute node,
not just the login and dev hosts, and three entries proved it independently by shipping raw
`perf_event_open` groups into `fft3d_create()` — L8_fusedaxes (`pmc=na`, with the group parse,
multiplex detection and fallback all validated against a mocked syscall harness first),
L36_pfa (`fe=na`), L45_mixedradix (`fe=na`). L6_unrolled found the setting and said so in
advance: "if the monitor's `perf stat` has been failing silently, that is why."

**The consolidated counter list is withdrawn.** No entry should gate a "Next" item on it
again, and I should not have made it the board's top priority for three rounds without
checking that it could be run. The replacement is already proven: eleven entries routed a
timed discriminator through `fft3d_description()` this round and every one of them produced a
readable node number. **That is the panel's instrument now, and §4.6's result, §4.2's closure
and all three L=36 nulls came out of it.**

---

## 6. The single highest-value thing the next round should attack, per geometry

### L = 6 — cut to one slot. There is no kernel work left.

Seven falsified mechanisms, plus the boundary probe reading zero on both of its branches
(`bf − bsp` = +0.1 to +0.6 ns against a 45–55 ns criterion; `bx + byz` ≈ `bf`), plus the uop
theory dead with `zf/f` = 1.14–1.18 printed on every line. B=1 is at 1.23× floor with three
processes agreeing to 0.03%. The one mechanism that ever worked here — association order — has
now been found, measured, and adopted by both entries, so they have converged on the same
codelet.

**The slot decision is no longer 2–2.** On non-overlapping distributions L6_pfa owns B=1 (9.1%)
and B=64 (0.4%), B=4096 is a tie, and L6_unrolled owns only B=32768 (0.8%). **Keep L6_pfa,
redeploy the other slot** — to L=64 (worst floor ratio at 1.73×, and its second arm has now had
two structural bets refused) or L=13 (thinnest library margin on the board at 1.14×). Whichever
entry remains should spend nothing further on B=1.

**If one kernel item is wanted at L=6, it is not at L=6:** propagate the association-order
search to every other unrolled codelet on the board — L=8's `dft8s` (the same join-at-the-
bottom shape L6_pfa's `DFT6V` had), L=36's DFT4/DFT9, L=13's `chunk13`. L6_unrolled has
specified the macro parameterisation that makes each one a three-candidate, ~40-line
experiment, and it is the only mechanism class with a positive node result at a closed-
arithmetic geometry.

### L = 8 — explain the 5% you paid for the 1% you won, then cut to two arms.

**The single thing: A/B out L8_batchsimd's allocation changes at the streaming cells.** B=1
improved 1.0% on medians and B=2048 regressed 5.2% in the same round, on a path the record
calls byte-identical; the only non-identical changes (page-aligned arena, `SI` +64 B) apply to
every mode. `-DL8_SI_OFF=512` plus reverting the 64-byte alignment is one build and two runs,
the flags already exist, and it is the only unexplained ≥5% number at this geometry. If the
regression is the alias fix, the fix should be regime-gated and §4.5 gains its second
refinement; if it is not, the panel has a fifth instance of the layout noise floor and should
say so.

**Secondary: cut to two slots, and L8_radix8 is the donor** — third in all four cells for the
second round, +1.4% at B=1, and a pre-registered prediction that failed on both the number
(0.578 against 0.570 ± 0.002) and the pick (2p in 1 of 3, not 3/3). Its contribution is the
evidence, and the evidence is now recorded: **the node's `create()` arena ranks `1f` 2–5%
faster than `2p` while the driver ranks it 0.6% slower.** L8_batchsimd already acted on that
and won the B=1 cell with it.

One live lead nobody has: **`fusedAA` was picked by the node's own tournament for the first
time** (L8_fusedaxes, B=64, run 3). L8_batchsimd declined to port the address-aware shapes
specifically because "the node's own tournament declined them both times." It has now not
declined it.

### L = 17 — the residual is located and it is not reachable. Stop funding B=1; cut to two arms.

`b1dec` settled it: 1.30–1.34× the port floor from L1, phases additive, `pt` declined in all
four cells, and the interleaved-complex rewrite unfunded by winograd's own 42% gate at a
measured 37.4%. Four mechanism classes were already falsified; this round adds same-volume
load prefetch and closes the memory side entirely. **There is nothing left at B=1 that is not a
rewrite, and the rewrite is measured not to pay.** Say so, and stop spending rounds there.

**L17_rader is the donor.** Third in all four cells, 13–15% behind at batch, and both of its
round-9 bets declined 6/6 by the node's own tuner after a −12.4% wallaby reading. Its
`ph/xp/fu` probe says why the batched deficit will not yield: `ph` = 16.4 µs of a 21.7 µs
whole-volume rival, `xp` = 6.8 against a ~4 µs compute share, `fu − ph − xp` = +0.5 at B=2048
and +2.4 at B=256 — **the excess is spread across both phases plus contention, with no
localised culprit**, which is the same shape as everything else at this geometry.

**If one L=17 item is funded, fund traffic deletion at B=256/B=2048**, where L17_matrixsimd and
L17_winograd are a dead tie at ~21.72 µs = ~10.9 GB/s against ~236 KB/volume of compulsory
traffic. NT stores (dead four rounds) and ERMS (dead, L45 r8) have both failed, so this needs a
new idea or an honest "closed". Everything else at L=17 is done, at 5.50× the best library.

### L = 36 — both theories are dead. The only untouched lever is arithmetic, and it is now written down.

Caches closed in r8 (`fu − p1 − p2w ≈ −3 µs`); front end closed this round three independent
ways (a 3.5× static shrink at +23%, a 2× walked-footprint A/B at +0.3%, a code-sharing halving
at ±1%). Every prefetch instrument has been tournament-rejected at B=1 by three separate
tuners. The cell has moved *away* from its floor twice running and is now the board's second
worst at 1.46×.

**The single thing: transcribe genfft's `n1_9` FMA DAG into all three L=36 arms.** L45_pfa did
it this round — 44 → 40 FMA-port vector ops per DFT9, correct on the first build, accuracy
improved — and wrote the transcription rule out mechanically ("every scalar re/im line pair is
one interleaved-vector op; every re↔im crossing is one SWAP with signs folded into VPAIR
constants", with the five crossings enumerated). All three L=36 arms run a Cooley–Tukey 3×3
DFT9 in exactly that family. It is worth ~5% of the port-0 budget, it is the only lever at this
geometry that has not been falsified, and L36_pfa's r1 record documents three failed *hand*
derivations of the same result — so transcribe, do not derive.

Two conditions on reading any of it: **close L36_pfa's pick lottery first** (B=4 runs of
131.95 / 184.11 / 132.48 and B=1 of 122.58 / 131.06 / 122.64 make that entry's numbers
unusable), and **cut to two arms** — on distributions not one of the four L=36 cells is
separated, three arms have produced three versions of the same null, and L36_pencilfused is
third in three of four cells with an unrecovered r7→r9 regression at B=256.

### The four wave-2 geometries, one line each

* **L = 13 — L13_direct owns all three cells for the third round; the deletions rule holds
  (−1.8% at B=512 against −4.7% on wallaby).** The single thing is the B=512 `-DL13_PW=0` /
  `-DL13_PFIN=0` split — two builds, two runs, asked for two rounds, and the **only outstanding
  monitor ask on the board that I can actually satisfy** now that the counters are withdrawn.
  Also: **take L13_rader's `-DL13R_FUSE=0` rollback** — its ROB model is falsified by its own
  criterion and the fused schedule costs +4.2% at B=16 and +2.8% at B=512.
* **L = 23 — done, and the r8 attribution corrected: the B=128 win was `pf=2 + pw=1`, not the
  folded-pair layout**, which the node displaced 3/3 by 3.4% when it was made the incumbent.
  1.14× floor at B=1, tightest ratio on the board, timed == checked in all three cells, two
  bit-identical arms with overlapping distributions. **Cut to one arm** and stop funding it.
* **L = 45 — L45_pfa owns all three cells, and produced the round's one reusable arithmetic
  asset and its cleanest arithmetic negative in the same move:** −5.5% of port-0 ops bought
  1.2%, so port 0 does not bind here and the floor ratio worsened to 1.68×. The node-measured
  phase split says where it does bind — **phase 1 is 76% of B=1** (p1 = 243.9 of fu = 319.2,
  additive) — so the next work is inside phase 1's z/y bodies or nowhere. Also A/B out
  L45_mixedradix's odd-column rework, the only change on its B=16 path, which coincided with a
  +2.0% regression at an unchanged pick.
* **L = 64 — L64_radix8 owns all three cells for the fifth round at the board's worst floor
  ratio (1.73×), and the mechanism that paid was the one its own dev-machine gate had killed
  (`p1pf`, picked 3/3) while the one I named (`propf`) priced at zero.** The remaining B=1
  residual is the SC store RFOs — pass 1 writes 4.5 MB scattered, the one component nothing has
  ever hidden — and with the counters withdrawn the only route is the entry's own in-plan
  create-time A/B applied to a store-mode twin. **L64_blocked should be judged on the
  split-complex rewrite it has deferred for two rounds, or the slot moved**: `st=1` and `st=2`
  are both dead, and L64_radix8's node numbers are standing evidence that the shuffle bill is
  not free even on a one-FMA-pipe part.

---

## 7. Promotion

Applying `CURATION.md` in order. Note that **rule 4 ("anything that beat a library baseline")
selects all nineteen entries** this round — every arm at every geometry beat every library in
every cell — so it carries no discriminating weight and the decision rests on rules 1–3 and on
the explicit prohibition against near-duplicates of an already-promoted entry.

**Rule 1 — the fastest correct entry per geometry, one per L, always.** On distributions:
L6_pfa (L=6), L8_fusedaxes (L=8: B=64/B=2048/B=16384 non-overlapping, B=1 tied),
L17_matrixsimd (L=17: three of four cells non-overlapping, B=2048 tied), L36_mixedradix (L=36:
marginally ahead at B=1, ties elsewhere), L13_direct, L23_rader, L45_pfa, L64_radix8.

**Rule 2 — a structurally different runner-up that came close.**

* **L6_unrolled** — owns B=32768 outright, ties B=4096, and *is the other half of the round's
  headline experiment*: the association-order result exists only because both entries ran it
  from opposite directions, and L6_unrolled's record carries the `ab1`/`f3d` fields that
  measured it, the uop theory's obituary (`zf/f` = 1.14–1.18), and the
  `perf_event_paranoid = 4` finding. The next panel cannot read the result from one file.
* **L8_batchsimd** — tied for B=1 and the source of the round's only positive §4.5 port, plus
  the sharpest statement of the arena-versus-driver failure mode ("if your default is
  driver-verified across two rounds, the right amount of arena trust at that cell is zero, not
  a wider band") and the regression that prices it.
* **L17_winograd** — a genuinely different structure (hand-derived 17-point cyclic+negacyclic
  module, 296 FP instructions) tied at B=2048 and within 2.1% at B=256, whose `p1/f23/fu`
  probe closed the interleaved-complex rewrite question for the whole geometry.
* **L36_pfa** — a different decomposition (GT-PFA two-sweep vs lanes-are-lines), tied at B=32
  and B=256, and the author of the `fug − fu` measurement that killed the front-end theory
  cleanly. Promoted despite its pick lottery, which §3(b) records against it.
* **L13_rader**, **L45_mixedradix**, **L64_blocked** — each within 5%, 0.5% and 4% of its
  geometry's leader respectively, each structurally distinct, and each carrying a
  pre-registered falsification the next panel needs (§ below).

**Rule 3 — instructive failures whose record documents the number that killed them.** All three
rule-2 wave-2 promotions qualify on this ground independently:

* **L13_rader**: the ROB port-fusion model — the most quantitative mechanism argument any entry
  has produced — falsified at its own threshold (predicted 4.7–5.0 µs, measured 6.030; ≥5.9 was
  the criterion), with +4.2% and +2.8% batch regressions recorded. Also documents two
  transferable process lessons: keep the plan-time schedule verifier (it caught a real WAR
  hazard that `restrict` would have turned into silent corruption), and **print the engaged
  configuration before believing any A/B** (its first "A/B" measured FUSE=0 against FUSE=0 and
  produced a perfect tie).
* **L45_mixedradix**: the `#pragma GCC target` / ICF discovery, which retracts its own r6
  claim and invalidates the same construction for anyone else; the audit showing the L=45
  leanness gap never existed at the claimed size; and the finding that wallaby's slow state is
  per-core and invisible in the MKL sentinel.
* **L64_blocked**: `st=2` declined 3/3 at 1098 against a ≤1030 criterion — a real structural
  rewrite with a corrected traffic ledger, killed by the node, alongside `st=1`. Its ledger
  correction (the fused win is temporal locality on the last axis's reads, not fewer memory
  ops) is the useful residue.
* **L36_mixedradix**'s `roll` probe (+22–24%) and **L36_pencilfused**'s `cs4 ≈ ip4` are the
  same class, but the front-end null is already documented in two promoted records
  (L36_mixedradix's own and L36_pfa's), which is where the near-duplicate rule bites.

**Not promoted, with the number:**

* **L8_radix8** — third in all four cells for the second round, +1.4% at B=1, prediction failed
  on both the number (0.578 vs 0.570 ± 0.002) and the pick (2p 1/3, not 3/3), and named the
  consolidation donor. Its one genuine contribution — the node's arena ranking `1f` 2–5% faster
  than the driver does — is carried, with the winning intervention attached, in L8_batchsimd's
  promoted record.
* **L17_rader** — third in all four cells, both round-9 bets declined 6/6 after a −12.4%
  wallaby reading, fourth consecutive round of node-rejected headline mechanisms. Its
  transfer-failure lesson is a near-duplicate of L17_winograd's `q4+pfw` null, which is
  promoted.
* **L23_matrixsimd** — bit-identical to L23_rader, overlapping in all three cells, and its
  stated determinism goal (3/3 picks at B=128) was not met. One algorithm twice; the promoted
  arm carries it.
* **L36_pencilfused** — third in three of four cells, its r7→r9 B=256 regression unrecovered
  (186.452 → 189.323), and its front-end null is the third instance of a result already
  recorded in two promoted files.

Fifteen entries, unchanged in membership from `panel_r8` — but the *reasons* have moved: L6_pfa
and L8_fusedaxes are now rule-1 winners on non-overlapping distributions rather than coin-flip
minima, L17_winograd drops from cell-holder to rule-2 runner-up, and three of the wave-2
promotions now rest primarily on rule 3.

**A note for `NOTES.md` at promotion time.** This round should be recorded as the one where the
panel's method changed: eleven entries measured instead of guessing, fourteen pre-registered
branches fired correctly, four theories died with node numbers attached (the L=6 store→load
joint, the L=6 uop count, the L=17 memory-side residual, the L=36 front end), one mechanism was
found and adopted by both of its geometry's arms (codelet association order, worth 3–6% on
identical arithmetic and *sign-inverted between the two machines*), and the monitor's own
top-priority ask of the last three rounds turned out to have been impossible all along. Four of
the eight geometries — L=6, L=17, L=23, and arguably L=8 — should now be described as closed
rather than slow, and the panel should be resourced accordingly: **eight geometries currently
hold nineteen slots for what the measurements say is eleven distinct algorithms.**

---

## Provenance and housekeeping

Sources: `bench/geom/impl_9/` (`impl` → `impl_9`), all 19 changed from `impl_8`,
`baseline_matrix.c` byte-identical. Measurements: `results/panel_r9/leaderboard.txt`,
`environment.txt`, 2428 raw `t_*.json` / `c_*.json` files, `check.log` (259/259 PASS),
`timing.err` (240 benign unsupported-geometry lines), `build_errors.txt` (empty). No
`failures.txt` was created, which under `sweep.sh`'s exit-code rule means no entry crashed,
hung, or tripped the driver's anti-memoization gate. Strategy records for all 19 entries are
updated in `bench/geom/strategies/` (+3158 lines this round).

Withdrawn this round: the consolidated `perf stat` counter list from `panel_r8` §6.
`perf_event_paranoid = 4` on `p55n3` as well as every dev host, proven three times over by
in-plan `perf_event_open` probes returning `na`. In-plan timed discrimination through
`fft3d_description()` is the panel's instrument; it produced every substantive result above.

PROMOTE: L6_pfa L6_unrolled L8_batchsimd L8_fusedaxes L13_direct L13_rader L17_matrixsimd L17_winograd L23_rader L36_mixedradix L36_pfa L45_mixedradix L45_pfa L64_blocked L64_radix8
