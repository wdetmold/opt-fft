# Round ice_r1 — monitor's verdict

Scored on `a80n0.lqcd.mit`, slurm job 438572, 2026-08-22T16:15:05-04:00.
Sources: `results/ice_r1/leaderboard.txt`, `environment.txt`, `build_errors.txt` (empty),
`agents/exits.txt`, `strategies/*.md`, `docs/CURATION.md`, `docs/LITERATURE.md` §4.

---

## 0. Two corrections to the round's own framing, before any numbers

**0a. The scoring machine is Ice Lake-SP, not Cascade Lake.** `environment.txt` reads
`Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz` — Ice Lake-SP: **two** 512-bit FMA pipes,
**1.25 MB** L2 per core, and no AVX-512 licence downclock. It is not a Gold 5218 (Cascade
Lake, one 512-bit pipe, 1 MB L2). Two independent in-plan clock probes on the node confirm
it: `L17_matrixsimd` and `L23_matrixsimd` both report `clk512/256 = 2.90/2.90 GHz`, and
`L17_matrixsimd`'s ramped runs read 3.30/3.50 GHz — 512-bit is never below 256-bit.
Every conclusion below is an Ice Lake conclusion.

**0b. Implementers developed on the *same* node they were scored on.** `PANEL_BRIEF.md`:
`tryout.sh` "builds **on the reserved node** (`-march=native` = Ice Lake) ... runs the
graded chain for your L". There is no Sapphire-Rapids-versus-Cascade-Lake gap to attribute
anything to in this round, and the "MKL spans 2.9× between those machines" figure is a
cross-machine statement that cannot explain anything measured here — MKL reproduces to
**±0.2%** between this round and `graded_icelake` on the same node (L=36: 82270.2 →
82272.0 µs/call). §4 below re-attributes the claimed-vs-measured gaps to their actual cause.

---

## 1. Headline per geometry — fastest correct panel entry vs. best library

### Batched — the graded chain (this round's measurement)

`cases.txt` is batched-only; per-transform times are per-call ÷ (B × m).

| L | B, chain m | fastest correct panel entry | best library | panel speed-up |
|---|---|---|---|---|
| **6** | B=64, m=4856 | **L6_unrolled 0.219 µs/xform, 38.25 GF/s** (spread 3.2%) | mkl_dfti 0.341 µs, 24.54 GF/s | **1.56×** |
| **8** | B=64, m=2572 | **L8_batchsimd 0.550 µs/xform, 41.91 GF/s** (spread 3.2%) | mkl_dfti 0.623 µs, 36.95 GF/s | **1.13×** |
| **17** | B=32, m=98 | **L17_matrixsimd 14.471 µs/xform, 20.82 GF/s** (spread 2.3%) | mkl2026_dfti 76.008 µs, 3.96 GF/s | **5.25×** |
| **36** | B=8, m=64 | **L36_pfa 119.163 µs/xform, 30.36 GF/s** (spread 1.4%) | mkl_dfti 160.688 µs, 22.52 GF/s | **1.35×** |

The other four graded geometries, for completeness: L=13 `L13_direct` 4.661 µs vs
mkl2026 6.066 (**1.30×**); L=23 `L23_rader` 39.142 vs ducc0 220.216 (**5.63×**);
L=45 `L45_mixedradix` 284.202 vs mkl_dfti 521.591 (**1.84×**); **L=64 is still the one
loss** — `L64_radix8` 1184.003 vs mkl_dfti 1016.396, i.e. MKL is **1.16× ahead of us**.

### Non-batched — NOT MEASURED THIS ROUND

`cases.txt` contains no B=1 case (`6:64`, `8:64`, `13:32`, `17:32`, `23:16`, `36:8`,
`45:4`, `64:2`), and no `t_*_B1.json` exists in `results/ice_r1/`. I will not manufacture
numbers for it. The nearest real measurement is **`geom/results/xarch_icelake`** — same
host `a80n0`, same day (10:25:53, job 438538), and for all four of these geometries
**byte-identical binaries** to the ones scored here (see §3). It is a single unchained
transform, not the graded chain, and its run spreads are 13–53%, so treat it as
indicative only:

| L | fastest correct panel entry (B=1) | best library (B=1) | panel speed-up |
|---|---|---|---|
| 6 | L6_pfa 0.166 µs, 50.61 GF/s (spread 13.7%) — L6_unrolled ties at 0.166 | mkl_dfti 0.261 µs | 1.58× |
| 8 | L8_batchsimd 0.404 µs, 57.10 GF/s (spread 14.6%) | mkl_dfti 0.464 µs | 1.15× |
| 17 | L17_matrixsimd 10.683 µs, 28.20 GF/s (spread 14.3%) | fftw3_measure 75.464 µs | 7.06× (7.74× vs MKL) |
| 36 | L36_mixedradix 76.976 µs, 47.00 GF/s (spread 53.6%) | mkl_dfti 149.263 µs | 1.94× |

Two selection inversions are visible and matter for `bench/geom/best/`: at L=36 the
non-batched winner is `L36_mixedradix` but the batched winner is `L36_pfa`; at L=6
`L6_pfa` leads unbatched and `L6_unrolled` leads batched. **If the next round is meant to
be graded on the non-batched case too, add B=1 rows to `cases.txt` — right now that column
of the scorecard does not exist.**

---

## 2. What changed since the previous round — and the drift floor that governs the answer

The comparable prior round is **`geom/results/graded_icelake`** (same node a80n0, same
`cases.txt`, 10:47 the same morning; MKL agrees to ±0.2%, so the workload is identical).
`ice_smoke` is *not* a usable baseline — see §4c.

**The controlling fact of this round: only 4 of 19 sources were edited.** `impl_1/*.c`
mtimes are 15:47–16:12 for `L13_direct`, `L13_rader`, `L17_matrixsimd`, `L17_rader`, and
**15:15 — the seed timestamp — for the other 15**, which `cmp` confirms are byte-identical
to `geom/impl_11`. So 15 entries re-measured the same binaries on the same node, and their
round-over-round spread *is* the harness's reproducibility floor:

| | min-of-mins delta, graded_icelake → ice_r1 |
|---|---|
| 15 untouched panel entries (identical binaries) | **−11.93% … +1.73%**, mean −2.61% |
| 56 library / baseline entries | −19.45% … +3.21% |

Worst cases: `L36_pencilfused` **−11.1%** and `L6_pfa` **−11.9%** with *not one byte
changed*; `mkl2026_dfti` at L=36 **−19.5%**. `L36_pencilfused`'s and `L36_mixedradix`'s
own internal probes swung 116.0→99.4 and 140.9→116.9 µs/vol on identical code — these are
runtime-autotuner plan races, not timing jitter. **Nothing under ~12% is a result on this
harness unless the entry's plan is pinned.**

### Per geometry

- **L=6 — no work done, no real change.** Both agents crashed. `L6_unrolled` −3.1%,
  `L6_pfa` −11.9%, both untouched code; both inside the drift band. `L6_pfa`'s 13.5% run
  spread against `L6_unrolled`'s 3.2%, for two implementations of the *same* PFA 2×3
  structure, is a plan race, not an algorithmic difference. The 1.56× lead over MKL holds.
- **L=8 — no work done, no real change, and the ordering flipped inside the noise.**
  `L8_fusedaxes` led `graded_icelake` (89973.7) and lost here (91529.5, **+1.7%**), while
  `L8_batchsimd` took the top slot on a −0.5% move. Both deltas are far inside drift; the
  three L=8 entries (0.550 / 0.556 / 0.561 µs) are a **statistical tie**. `L8_fusedaxes`
  also degraded in stability — run spread 1.8% → 16.6%, worst-run median 91647 → 106784.
  1.13× over MKL is the thinnest margin of the four headline geometries and it did not move.
- **L=17 — the only geometry where measurable work landed, and its value is in the
  variance, not the minimum.** `L17_matrixsimd` improved the reported min only 3.2%
  (46864 → 45382 µs/call), which is inside drift — but the min is the wrong statistic here.
  Across runs its **median-of-medians went 56486 → 45674 µs/call (−19.1%)** and its
  worst run 65951 → 46508 (**−29.5%**), with run spread collapsing **40.6% → 2.3%**. That
  is a real, large, defensible win: the entry's record diagnoses a tuner that probed an
  unramped `schedutil` core, read 2.90 GHz for everything, and therefore picked a
  256-bit-flavoured plan on a two-512-bit-pipe machine; the fixes (clock-settle spin,
  chain-shaped tuner stage at 17 ≤ batch < 64, class-D X-first from batch 17) removed the
  bad plan rather than speeding up the good one. `L17_winograd` (untouched) −2.3% min /
  +1.1% median — stable. `L17_rader` **+1.0% (a regression, inside drift)** despite the
  round's largest volume of new machinery.
- **L=36 — no work done.** `L36_pfa` −1.9%, `L36_mixedradix` −2.3%, `L36_pencilfused`
  −11.1% — all untouched code, all drift. `L36_pfa` and `L36_mixedradix` are within 0.3%
  of each other (119.163 vs 119.530) and cannot be separated. The `mkl2026_dfti` baseline
  moved −19.5% here, making L=36 the least stable geometry on the board.
- **Also, off the headline four:** `L13_direct` −1.6% min / −0.9% median (edited, no
  measurable gain); `L13_rader` +0.3% min / 0.0% median (edited, no gain — but see §5, its
  record is the round's best artifact); L=23, L=45, L=64 all untouched and all inside drift.

### Did anything regress?

Nothing regressed beyond the drift floor, and no entry lost correctness. The three nominal
regressions — `L8_fusedaxes` +1.7%, `L17_rader` +1.0%, `L8_radix8` +0.4% — are all smaller
than the −11.9% swing an *unmodified* binary produced elsewhere in the same table. The one
regression that is real and not a timing number is **`L8_fusedaxes`'s stability**: run
spread 1.8% → 16.6% on unchanged code, which is a plan race that has got worse.

---

## 3. Failures, missing entries, and correctness — the adversarial pass

**Correctness: all 26 backends pass, at every geometry, under both gates.** `check.py`
verifies each output against `numpy.fft.fftn` (rel_l2 < 1e-12) **and** verifies the end
state of the full unitary-normalised chain of m steps against its closed form, which
compounds an error at any one of up to 4856 steps into the final answer. Measured rel_l2
runs 1.3e-16 … 8.4e-16 and chain rel_l2 1.0e-14 … 5.9e-14, all far under tolerance.
`check.log` contains no FAIL. **No fast wrong answer is hiding in this leaderboard.**

**Nothing is missing from the measurement.** 624 `t_*.json` files = 26 backends × 8 cases
× 3 runs, exactly. 399 are `supported:false` (each `L<n>_*` entry correctly declines the
other seven geometries) and **zero** are a supported timing at the wrong L, so the
leaderboard's per-L blocks drop nothing. `build_errors.txt` is empty, `failures.txt` does
not exist, and `sweep.out` contains no error, warning, skip, timeout or signal.
**No entry crashed, hung, or failed to build.**

### The real failure of this round is upstream of the benchmark: 15 of 19 implementer agents died in the first minute

`agents/exits.txt` — 13 × `exit=134` (SIGABRT), 5 × `exit=139` (SIGSEGV), and every one of
those 15 agent logs is timestamped 15:30, the minute the round started. The evidence is
unambiguous that these are **agent-harness crashes, not FFT bugs**:

- `L8_batchsimd.log`: `panic: Failed to start HTTP Client thread: Resource temporarily
  unavailable (os error 11)` … `oh no: Bun has crashed.`
- `L17_winograd.log`: `ASSERTION FAILED: MemoryExhaustion: Crash intentionally because
  memory is exhausted.` (JavaScriptCore `LocalAllocator::allocateSlowCase`)
- `L36_pencilfused.log`: `memory allocation of 32 bytes failed`

The host ran out of memory and thread slots launching 19 concurrent Claude Code workers.
Named, with evidence, the agents that produced **nothing** this round:

> `L17_winograd`, `L23_matrixsimd`, `L23_rader`, `L36_mixedradix`, `L36_pencilfused`,
> `L36_pfa`, `L45_mixedradix`, `L45_pfa`, `L64_blocked`, `L64_radix8`, `L6_pfa`,
> `L6_unrolled`, `L8_batchsimd`, `L8_fusedaxes`, `L8_radix8`

Corroboration beyond the exit codes: all 15 sources still carry the 15:15 seed mtime and
`cmp` clean against `geom/impl_11`, and none of them wrote a strategy record — `strategies/`
holds exactly the four files belonging to the four agents that exited 0. This is the third
occurrence of the same failure mode (`git log`: "Ice panel workers move off wombat: strict
overcommit killed a whole round of agents", "Worker-crash storm detector in the round
runner"). **The crash-storm detector did not stop this round from being scored as if it
were a full round.** Four fifths of ice_r1 is a re-measurement of panel_r11.

### One reporting defect worth fixing in the harness

The `backends:` description block does not necessarily describe the plan that produced the
reported number. `L17_matrixsimd`'s three runs self-describe as `deferred-Z, pt=0,
clk512/256=3.30/3.37` (r1, min 45382 — the reported one), `extract-store, pt=1,
clk512/256=2.90/2.90` (r2, min 46447), and `deferred-Z, pt=0, 3.30/3.50` (r3, min 45651).
The leaderboard prints **r2's** description next to **r1's** time. Same binary — the
divergence is the runtime tuner choosing differently depending on how ramped the core was
when it probed. `leaderboard.py` should carry the description from the run that supplied
the min, or print all distinct plans.

---

## 4. Claimed numbers vs. measured numbers

**The stated attribution — "they develop on Sapphire Rapids, you score on Cascade Lake" —
does not apply to this round** (§0). Development and scoring were the same Ice Lake node.
The gaps have three other causes, and they are separable.

**(a) The dominant cause is the graded chain's scale pass, absent from every private
probe — a uniform 25–35% optimism at L=8.** All three L=8 entries claim arena numbers of
0.409–0.458 µs/transform and all three measure 0.550–0.561:

| entry | own claimed µs/xform | measured | optimism |
|---|---|---|---|
| L8_fusedaxes | 0.409–0.415 (`arena{fusedAA2+pfs=0.409}`) | 0.556 | +34% |
| L8_batchsimd | 0.426–0.442 (`arena{FUSEDAA/s0=0.426}`) | 0.550 | +26% |
| L8_radix8 | 0.457–0.458 (`arena{1f-pfs=0.458}`) | 0.561 | +22% |
| L36_pfa | `fu=81.0` µs/vol | 119.163 | +47% |
| L36_pencilfused | `ip4=99.4` µs/vol | 123.594 | +24% |

Three independent implementers landing on the same sign and magnitude is a systematic
term, not three mistakes. The graded chain scales the whole output unitarily between
steps *inside the timed unit*, and the private arenas do not: `L13_rader`'s record measures
"~1.0 µs/xform of driver-side unitary scaling inside the timed unit", and
`L17_matrixsimd`'s measures the chain overhead at "~2.8 µs/step" for Winograd against
"~8.7" for itself. At L=8, 0.550 − 0.42 ≈ 0.13 µs is about what a unitary scale of 512
complex doubles costs. **The claims are not wrong about the kernel; they are measuring a
different unit of work.** The fix is the one `L17_matrixsimd` already made — tune in the
chain regime — not a machine-difference excuse.

**(b) The implementers who did tune in-regime were accurate to a few percent**, which
confirms (a): `L17_matrixsimd` claims `xfda 14.29` µs/step *including the scale pass* and
measures **14.471** (+1.3%); `L17_rader` claims `xl 19.72` and measures **19.099** (−3.2%,
conservative); `L23_rader` claims 38.55 µs/t → 39.142 (+1.5%); `L45_mixedradix` claims
`fu=273.5` → 284.2 (+3.9%); `L23_matrixsimd` claims 37.74 → 39.584 (+4.9%);
`L36_mixedradix` claims `pf0=116.9` → 119.530 (+2.2%).

**(c) `ice_smoke` is a contended measurement and two implementers reasoned from it.**
`L17_matrixsimd`'s record opens from "ice_smoke … 21.03 µs/step, 1.30× BEHIND
L17_winograd" and `L17_rader`'s from "23.7 µs per transform step, 3rd of 3". But
`graded_icelake`, on the same node with the same binaries, already had `L17_matrixsimd` at
14.94 µs/step — the smoke figure was 41% high because the smoke run was not taken in a
drained window. Their *diagnoses* were nonetheless correct (unramped-core clock probes,
untuned batch-32 path) and their fixes produced the −19% median improvement in §2. But
the headline gains those records imply — "21.03 → 14.47" — should be read as **14.94 →
14.47 on the minimum, 56.5 → 45.7 on the median**. `context.md` points implementers at
`ice_smoke` as the only prior leaderboard; it should point at `graded_icelake` instead.

**(d) Where a machine difference *is* real, it is inside the code, not in the
measurement.** All 15 untouched entries carry gates and thresholds derived on the CLX Gold
5218 (one 512-bit FMA pipe, port 5 free for shuffles). `L13_rader` measured one of them
costing **+7.4%**: the `prefetchw` gate fired on `batch·vol > L2`, but the chain keeps
`out` L3-resident, so pw=1 was wrong on this node — the rule is now `pw` only past L3.
That is the transferable finding, and 15 entries have not had it applied.

---

## 5. Which `LITERATURE.md` §4 open question moved

### §4.2 — "L = 17: dense-symmetric, Rader, or a hand-derived Winograd module?" — MOVED, and now answered on a second microarchitecture with a mechanism attached.

The section's open question (a) is *"which of dense-symmetric and Rader-17 wins on this
hardware, batch-vectorised?"* and (b) is *"does the symmetric/antisymmetric convolution
split add anything on top?"* This round measured all three structures side by side, at the
same geometry, on the same core, under the graded chain, all correct to 3.2–3.3e-16:

| structure | entry | µs/xform | vs. best |
|---|---|---|---|
| dense conjugate-symmetric (§02 §7, §03 §6.4/§06 §6.4a's "direct O(n²)") | **L17_matrixsimd** | **14.471** | 1.00× |
| hand-derived Winograd module, 296 FP instr, sym/antisym split (§01 §8) | L17_winograd | 16.113 | 1.11× |
| Rader-17, cyclic/negacyclic (§01 §8, §02 §7) | L17_rader | 19.099 | 1.32× |

**(a) Dense-symmetric wins, and §02 §7's "Rader is not the lever at L=17" now holds on
both Cascade Lake and Ice Lake** — the ordering is identical to the CLX rounds, so it is
not a microarchitectural accident. **(b) The symmetric/antisymmetric split does not pay
for itself**: `L17_winograd` implements exactly §01's structural suggestion and still
loses to plain dense-symmetric by 11%.

What is genuinely new is **the mechanism**, and it is an Ice Lake finding the corpus did
not have. §4.2's framing is about flop and instruction counts; on this machine the
ordering is set by **port 5**. `L17_matrixsimd`'s record: the kernel runs at ~1.03 FP
ops/cycle on a machine with two 512-bit FMA pipes on ports 0 and 5, and "the 40
`vshuff64x2` per Y/Z chunk are port-5-only on ICX and now steal slots from the second FMA
pipe; the CLX-era claim 'transposes hide under the FMA stream' is machine-specific and
FALSE here." `L13_rader` independently derives the same port floor —
`(13.0k FP + 5.5k shuf)/2 ≈ 9.25k cyc/vol`. **The scarce resource at L=17 on Ice Lake is
not multiplications and not instructions; it is port-5 issue slots, and the ranking of the
three structures tracks their shuffle count.** That is a sharper statement than anything in
§4.2 and it should be written back into the section.

### §4.8 item 6 — "no primary measurement in the corpus for Ice Lake-SP or later server parts … Measure it on the node." — CLOSED.

Measured, by two entries' in-plan probes: on the Gold 6326, `clk256 = clk512 = 2.90 GHz`
under load and 3.30/3.50 GHz on a ramped otherwise-idle core. **512-bit never runs slower
than 256-bit; there is no AVX-512 licence cliff on this part.** The corroborating
consequence is that every 256-bit variant lost: `L17_matrixsimd` measured "every 256-bit
variant ≥ 19.1" against 14.29 for the 512-bit X-first plan. Two ancillary findings worth
recording in §4.8: the `schedutil` governor makes short `create()` functions probe an
unramped core and read the base clock, which silently mis-selects plans (the ~150 ms
clock-settle spin is the fix); and non-temporal stores are **catastrophic** in this
L3-resident chain regime — `L17_matrixsimd` measured NT at 28.9–30.8 µs/step against 14.29
in-place, i.e. "RFO avoidance pays only at DRAM."

### Not moved

**§4.3** (axis fusion, and specifically the re-opened **L2↔DRAM** tiling case) — every
graded working set here is 0.42–16 MiB and L3-resident, so no experiment this round
crossed the boundary the section says is untested. It remains, in the section's own words,
"the largest untried structural move on the board." **§4.1** (batch-codelet register
pressure / spill audit), **§4.5** (L=8 padding and the 4 KB store-forward alias) — both
belong to L=6 and L=8, whose agents all crashed; untouched.

---

## 6. The single highest-value thing the next round should attack

**Above all four geometries: fix the worker launch.** 15 of 19 agents died in the first
60 seconds from host memory and thread exhaustion (§3), for the third round running. Stage
the launches or cap concurrency, and make the crash-storm detector **abort the round**
rather than let it be scored — four fifths of ice_r1 is a re-measurement of panel_r11 wearing
a new round name. Second: pin the plan races before chasing kernels. An unmodified binary
moved 11.9% between two scoring windows, so a plan-raced entry cannot demonstrate a 3% win
at all; port `L17_matrixsimd`'s clock-settle spin and chain-shaped tuner stage to every
entry — it is the one change this round that produced a measurable result.

- **L = 6** — *(1.56× over MKL; nothing done this round.)* Kill the plan race first:
  `L6_pfa` shows 13.5% run spread against `L6_unrolled`'s 3.2% for the *same* PFA 2×3
  structure, so the 0.5% gap between them is unmeasurable and no kernel work can be
  validated until that is fixed. Then settle **§4.1** on this machine, which is cheap and
  overdue: §01 measured 17 live registers for n=6 including temporaries and concluded AVX2
  "will spill a little" — but AVX-512 has **32** registers, so the premise may simply be
  dead here. §07 §7.8 gives the check: count `vmovupd` against `%rsp`/`%rbp` in the
  generated assembly before believing any timing, then try the wider batch granule the
  spill argument was blocking.
- **L = 8** — *(1.13× over MKL — the thinnest margin of the four, and it did not move.)*
  Read one PMU counter: **`ld_blocks_partial.address_alias`**. §4.5/§08 §1.8 says L=8's
  volume stride is exactly 8192 B = 2 × 4096, so with two page-aligned buffers *every* load
  from `in` falsely aliases a recent store to `out` in the low 12 bits — maximally
  degenerate at this size, a documented re-issue penalty, and **never once checked**. The
  brief states the PMU is exposed on this node (`perf_event_open` works). If it fires, the
  fix is ducc0's one-line guard (`if ((dstride & 256) == 0) dstride += 16;` /
  `make_noncritical()`), not a new kernel. This is the cheapest available shot at the
  weakest margin on the board, and it also settles §4.5.
- **L = 17** — *(5.25× over MKL, the panel's biggest win; defend and extend it.)* Attack
  port 5. §5 established that the kernel runs at ~1.03 FP ops/cycle on a two-512-bit-pipe
  machine because port-5-only `vshuff64x2` displaces the second FMA pipe. `L17_rader` already
  built the countermeasure — the `ty` candidates, ymm 4×4 tile transposes that cost ~1.6×
  the shuffle µops but dual-issue on p1/p5 instead of monopolising p5 — and it is sitting
  in the entry that is 1.32× *behind*. **Port the ymm-tile transposes into
  `L17_matrixsimd`**, which already has the lowest instruction count and the stable plan.
  Heed `L13_rader`'s measured warning while doing it: p5 relief pays only if it does not
  lengthen load dependency chains (its merge-broadcast load variant cost +1.36 µs/xform).
- **L = 36** — *(1.35× over MKL; nothing done this round.)* This is the geometry the §4.3
  re-opening was written for, and the only one on the board where it applies: at B=8 the
  working set is **11.39 MiB against a 1.25 MB L2**, a ~9× overflow, which is the
  **L2↔DRAM** boundary (measured gap 7×) that every panel fusion experiment so far has
  missed by fusing across L1↔L2 (gap 2.6×). Run the construction Intel's manual, Alappat
  et al. and L3-Fusion all independently recommend: **tile the batch so one tile fits L2,
  then run all three axes inside the tile**, and measure it against the current two-sweep
  PFA. Precondition: make the plan deterministic first — identical L=36 binaries scored
  62180 and 61011 µs/call while the internal probe swung 140.9 → 116.9 µs/vol, and
  `L36_pfa` and `L36_mixedradix` are currently within 0.3% of each other and inseparable.

---

## 7. Curation — what to keep, and why

Applying `docs/CURATION.md`'s four grounds in order.

**1 — Fastest correct entry per geometry (mandatory, one per L).**
`L6_unrolled`, `L8_batchsimd`, `L13_direct`, `L17_matrixsimd`, `L23_rader`, `L36_pfa`,
`L45_mixedradix`, `L64_radix8`. All correct at 2.3e-16…4.5e-16 with chain checks passing.

**2 — A structurally different runner-up that came close.**
- `L17_winograd` (1.11×) — CURATION's own worked example, verbatim: a genuinely different
  structure (hand-derived 296-instruction Winograd module with the sym/antisym split)
  within ~20% of a dense conjugate-symmetric winner. Keeping both is what makes §4.2(b)
  legible to the next panel.
- `L23_matrixsimd` (1.01×) — dense 23×23 conj-folded against Rader-23, a **dead heat**
  (39.584 vs 39.142 µs, well inside drift). This is the same dense-vs-Rader question as
  L=17 at a larger prime, with the *opposite* answer, and it was **not** promoted at
  panel_r11 — so this is new information in the exemplar set, not a duplicate.
- Deliberately **not** promoted as runner-ups, because CURATION forbids near-duplicates of
  an already-promoted entry: `L6_pfa` (also PFA 2×3), `L8_fusedaxes` and `L8_radix8` (both
  radix-8 cousins of the winner), `L36_mixedradix` (also PFA 4×9), `L45_pfa` (also PFA
  9×5), `L64_blocked` (also 8×8 two-stage). Their leaderboard rows preserve the numbers;
  their code adds nothing the winner's does not already show.

**3 — Instructive failures whose record documents the number that killed them.**
- `L17_rader` (1.32×) — outside the ~20% runner-up band, and kept precisely for that: it
  is the measured refutation of Rader at L=17 on a second microarchitecture, with a fresh,
  detailed ice-panel record. Promoting it is what stops the next panel spending a round
  rediscovering §4.2(a).
- `L13_rader` (1.32×) — the round's most transferable artifact even though its time did
  not move (+0.3%). Its record carries two clean, measured, bit-identical negatives:
  merge-masked `vbroadcastsd` transposing loads **rejected at 7.874 vs 6.514 µs/xform
  (+1.36)** because the 8-deep merge chains serialise behind L1/L2 latency, and
  extract-form transposing stores a **wash (6.514 vs 6.531)**. Plus the CLX→ICX gate fix
  (`prefetchw` keyed on L2 instead of L3, **+7.4%**) that the other 15 entries still need.

**4 — Anything that beat a library baseline.** Every promoted entry above beats at least
one library at its geometry except `L64_radix8`, which is in on ground 1 and is the
standing 1.16× loss to MKL that the brief names as the panel's one defeat.

### Operational note before running `promote.sh`

`promote.sh` copies `strategies/<name>.md` and warns that an entry "should not be promoted
without one". **`bench/ice/strategies/` contains only four records** — the four agents that
survived. Eight of the twelve names below have no ice-panel record because their agent died
at 15:30. Their full lineage records do exist, at `bench/geom/strategies/<name>.md`
(rounds panel_r1…r11) and `bench/mt/strategies/<name>.md`, and the four ice records
explicitly cite those as the authoritative history. **Copy the `geom` record into
`bench/ice/strategies/` for each of the eight before promoting**, so the exemplar ships
with its record as CURATION requires — and note in `exemplars/ice_r1/NOTES.md` that those
eight are byte-identical carry-overs of `geom/impl_11`, re-measured but not revised, so a
later reader does not mistake them for ice_r1 work.

PROMOTE: L6_unrolled L8_batchsimd L13_direct L13_rader L17_matrixsimd L17_winograd L17_rader L23_rader L23_matrixsimd L36_pfa L45_mixedradix L64_radix8
