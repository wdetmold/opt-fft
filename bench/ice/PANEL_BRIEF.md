# Implementer brief — single-threaded, ICE LAKE, the graded chain

A parallel panel to the multicore one: back to **one core**, on a different
microarchitecture, measuring **the graded workload** — because the goal now is a
**hardware-aware code**, and because we intend to **beat the other code-generation
pipelines** whose best graded attempt stands at C_opt = 1.500 s.

Everything from the phase-1 brief still holds (ABI `fft3d_api.h`, layout, correctness
gates, no FFT libraries, single-threaded) with these changes:

## The machine: bare-metal Ice Lake-SP (Xeon Gold 6326), reserved whole

| | this panel's node (a80n0) | where your kernels were tuned (Gold 5218, CLX) |
|---|---|---|
| 512-bit FMA pipes | **2** | 1 |
| loads per cycle | 2×64B (bare metal — **not** the VM tier's ~1.4) | 2×64B |
| L1d / L2 per core | 48 KB / **1.25 MB** | 32 KB / 1 MB |
| PMU | **exposed** (`perf_event_open` works; the `perf` tool is absent but counters are readable) | exposed |

Corpus §10 (`docs/literature/10-icelake-under-glass.md`) is your primary reading — seven
rival pipelines' forensics on a *virtualized* slice of this microarchitecture — **but read
its provenance note carefully**: their ~2.1-uop/cycle feed cap and load-throughput collapse
were VM artifacts. On this bare-metal node the second FMA pipe is genuinely feedable. What
transfers regardless: phase-split register-resident codelets (1.6× on the 23-point kernel),
sign-folded constant tables, the z-split L=64 layout (7× working-set compression), denormal
padding traps, 4K-aliasing hygiene, and the GCC 13.2 spill cures. Their actual sources are
in `ext/reference/fft_v4_solutions/` — the 1.00-scorer is 1,045 readable lines; the fastest
(1.500 s) is a generator worth reading at `1760b1bf_score0.96/generator.py`.

## The workload IS the graded chain

`cases.txt` is the graded configuration: chains of m unitary-normalized transforms
(`--chain m --unitary` in the driver), all cache-resident (0.42–16 MiB). Every dev timing
and every scored number uses it. Consequences the corpus and our own grading both flag:
iterate a volume through steps while it is cache-resident; the end state is verified in
closed form, so an error anywhere in m steps is caught.

## Where you work

`./tryout.sh <name> [L] [batch]` builds **on the reserved node** (`-march=native` = Ice
Lake), leases you one pinned core out of 24, runs the graded chain for your L, verifies
single-transform + whole-chain, checks repeatability, and prints MKL on the same case and
core. The node is held by a placeholder job; do not submit slurm jobs yourself. The
monitor scores inside a window that drains all core leases, so the node is quiet for the
numbers that count.

Priorities from the grading data at these exact points: **L=64 is our one loss** (1.18× on
Ice Lake under chain semantics — the z-split layout is aimed straight at it); L=8/13/36
margins are thin (0.76–0.87); L=17/23 are strong (0.18–0.20) but the rivals' phase-split
codelets attack exactly those, so defend them. Selection matters as much as kernels: the
winning entry differs across CLX/ICX/Haswell at four of eight sizes, and what you learn
here feeds the machine-aware selection model in `bench/geom/best/`.

---

# TASK CHANGE (from ice_r4): the full rival step, with a correctness gate they never faced

The chain step is now **exactly the rival pipelines' graded step**:

    state <- (z + c) / (1 + |z + c|),   z = FFT(state)

where `c` is a fixed 0.1-scaled complex gaussian field (supplied by the harness, same
layout as the data). The head-to-head that forced this change: on our own bare-metal node,
the rivals' fastest code runs the full task in **1.00 s** end-to-end while our suite —
kernels that WIN the FFT-only comparison — takes **2.24 s**, because ~57% of our time is an
unfused map pass. The map is the battleground. Their per-size times to beat (seconds, full
graded points): 6: 0.102, 8: 0.115, 13: 0.164, 17: 0.035, 23: 0.103, 36: 0.059, 45: 0.201,
64: 0.226.

## How to fuse: the optional chain entry point

Export (in addition to the normal ABI — keep it working, the single-transform check uses it):

    void fft3d_chain(fft3d_plan *plan, const double _Complex *x0,
                     const double _Complex *c, double _Complex *final_out, int m);

If present (it is detected as a weak symbol), the driver times YOUR whole m-step chain.
If absent, you are timed through the fallback — your fft3d_execute plus a driver-side
vectorized map — which is exactly the 2.24 s configuration. The fusion technique that won
for the rivals is in §10 §2 and in their sources (`ext/reference/fft_v4_solutions/`): one
hardware divide per point with everything else as Newton on the FMA pipes, and the *lazy
map* — keep the buffer raw between steps and apply the map during the next step's first
contiguous pass, where `c` streams sequentially.

## Correctness: where we beat them by DESIGN, not just speed

Two gates, both enforced by the harness:

1. Single transform vs numpy, rel L2 < 1e-12 (unchanged).
2. **Whole-chain end state vs a numpy reference chain, budget 1e-13 per step**
   (tol = max(1e-12, 1e-13·m)).

The budget is calibrated by measurement, not taste: a 1-ulp input perturbation propagated
through the longest chain (m=4856) ends at 4.8e-12 — the map is a contraction, not a chaos
amplifier — and exact implementations differ only by reassociation (~3e-11 observed worst).
So a full-double-precision entry passes with ~8× margin. The rivals' fastest code drifts to
**1.28e-8** over that chain — and the mechanism is now verified in their source: their map
has an exact variant (`pw_full`, 3 Newton steps) used at just 7 boundary call sites, while
the bulk of every step runs `pw_full_fast` — a **single-precision `rcp` seed + 2 Newtons**,
~1e-12 residual per application. Step 1 of their chain is exact to 2.8e-16, so any
single-call check sees a perfect answer *by construction*; the drift lives only in the
chain (2.3e-14 at m=32 → 1.8e-11 at m=2048 → 1.28e-8 at m=4856). Their grader never looked
past one call. Ours checks the chain. **A fast entry that drifts past the budget is a
rejected entry.**

The nuance that makes this a tool rather than only a trap: precision-tiered maps are LEGAL
here wherever they stay inside the 1e-13/step budget. Their cheap 2-Newton map measured
against the exact chain passes our gate at the short-chain sizes (5.7e-14 at L=17's m=98)
and fails it at the long ones (L=6's m=4856 by ~26×). So tier by (L, m): the float-seed +
2-Newton map is a legitimate speed lever at L=17/23/36/45/64, and the third Newton step (or
one exact divide per point) is mandatory at L=6/8/13. Do the arithmetic for your point and
write it in your strategy record.

## The roofline (read ../../ROOFLINE.md)

A measured-ceiling roofline now exists for the graded workload: r = 0.137 vs MKL (7.3×) on
the grading VM, with full marks at r ≤ 1/6 — about 82% of it. Current ladder: our unfused
suite r = 0.705, best rival r ≈ 0.34. Two of its findings are design directives here:

* **FFT kernels roofline near the ADD port, not FMA peak** — the graphs are only 14–36%
  FMA-pairable (L=8: 14%, effective peak ≈57% of nominal). Stop optimizing for FMA count;
  optimize total vector-uop count and add-port pressure.
* **On this bare-metal node, L=6 and L=8 are L1-traffic-bound, not compute-bound** (38
  fl/pt with the map sits under the 112 B/pt/iter floor once the issue ceiling is
  bare-metal). Below ~1.75 cyc/pt at those sizes, only traffic reduction — deeper fusion,
  fewer passes — moves anything.


---

# FINAL STANDINGS (series complete, ice_r6, 2026-08-23)

Best gate-passing entry per size across rounds, full rival step, bare-metal ICX:

| L | entry | chain s | rival | ratio |
|---|---|---|---|---|
| 6 | L6_pfa (r5; both r6 entries REJECTED by the chain gate at ~1e-8 drift) | 0.0944 | 0.1018 | **0.93** |
| 8 | L8_fusedaxes | 0.0913 | 0.1147 | **0.80** |
| 13 | L13_direct | 0.2162 | 0.1636 | 1.32 |
| 17 | L17_winograd | 0.0365 | 0.0347 | 1.05 |
| 23 | L23_matrixsimd | 0.0947 | 0.1030 | **0.92** |
| 36 | L36_mixedradix | 0.0516 | 0.0586 | **0.88** |
| 45 | L45_pfa | 0.1870 | 0.2007 | **0.93** |
| 64 | L64_blocked | 0.1712 | 0.2263 | **0.76** |

**End-to-end 0.943 s vs the rivals' 1.00 s — ahead overall and at 6 of 8 sizes — under a
chain gate 10^4-10^6x stricter than any rival campaign's (v6's own per-size gates: 1e-4 at
L=6 against our 4.9e-10).** From 2.24 s in three rounds. Remaining gaps: L=13 (1.32) and
L=17 (1.05) — the sixteen v5/v6 solutions in `fft_v5v6_solutions/` (chain-gated, pure fp64)
include Hartley-split prime kernels worth mining for both.

---

# ROUNDS 7-8: mine the competition, close the last three cells

The rival campaigns' full reconstructed sources are now in the repo root — read them:

* `fft_v5v6_solutions/` — **16 chain-gated, pure-fp64 attempts** (v5 flat gates, v6 per-size
  gates). The top one, `v6_f40c5e25_score0.91/`, is a generator (`dev_generators/gen.py`)
  shipped with config `HSTYLE=bcastv H13=reg6 H17=s44 H23=s65`: PFA 6/36/45, CT 8×8 for 64,
  and **direct symmetric "Hartley-split" DFTs for the primes (~2h² FMAs per pencil,
  h=(N−1)/2)** — its own README calls the primes "the decisive win over MKL". SoA
  8-volumes-per-zmm below L=36 with within-volume fallbacks at small batch remainders.
* `fft_v4_solutions/` — the 7 earlier attempts (looser tier; the 1.500 s record and the
  1,045-line 1.00-scorer live here).
* `results/rivals_icelake/` — the rivals' codes **re-benchmarked on THIS node** (may still
  be filling in when you start; use whatever rows exist). These are the honest targets:
  same silicon you are scored on, not their tier's walls.

Mission, in priority order:

1. **L=13 (we trail 1.32×)** — the one real gap. The v6 Hartley-split 13-point pencil
   (`H13=reg6` in the generator) is the thing to understand and beat, not just match.
2. **L=17 (we trail 1.05×)** — 5% to find; compare `H17=s44` against our `L17_winograd`.
3. **L=6: regain a passing fast entry.** Both r6 entries were REJECTED at ~1e-8 chain
   drift — the exact cheap-map shortcut the gate exists to catch. At m=4856 the third
   Newton step (or one exact divide per point) is not optional. r5's `L6_pfa` (0.0944 s)
   is the floor to beat *legitimately*.
4. Protect the five winning sizes; the score anchor is now the roofline itself
   (problem v7: full marks at r ≤ 0.137), so every size still has headroom worth taking.

## The rivalry is symmetric — plan accordingly

The rival pipelines have been given OUR best-in-class sources, just as we have theirs. Two
consequences: the targets in `results/rivals_icelake/` are a *snapshot*, not a wall — their
next generation will incorporate our kernels (expect our chain-true L=6 map, the z-split
L=64 work, and the PFA wins to show up on their side); and information is not a moat.
What compounds is (a) rate of improvement per round, (b) the strict-gate correctness
discipline — a rival generation raised under loose gates keeps its cheap-map habits, and
(c) hardware-awareness: kernels that re-select per machine travel better than kernels
tuned to one tier. Win on those.

## RIVAL UPDATE (Aug 23, for round 8): a rival scored 0.99 — and our L=6 gate was wrong

A rival pipeline scored 0.99 on the rubric (r ~ 0.147 vs MKL on the grader tier, 93% of
the roofline pace, ~0.44-0.47 s bare-metal-equivalent vs our ~0.85). Its code is at
fft_warm_solutions/warm_d43251c2_score0.99/ with four siblings — READ THEM. They were
seeded with OUR r5/r6 kernels, so expect to recognize the bones; what you are mining is
what they changed.

**Gate correction (important).** The map chain is weakly CHAOTIC: two correct fp64
implementations diverge roughly exponentially with step count. Measured on r7's own
chain at L=6, m=4856: MKL/FFTW/ducc0 land at 1.7e-9 … 3.1e-9 rel L2 vs the numpy
reference — ABOVE the old gate (4.9e-10). The old gate was a coin flip on the seed
(r5's libraries: 2-4e-10; r6's: 6e-9-1.1e-8). Consequences:
1. r6/r7 L=6 "gate failures" at ~1e-9 were FALSE REJECTIONS — those entries sat closer
   to numpy than MKL did on the same chain. L6_unrolled's 0.0740 s from r7 is back in
   play, pending a one-step exactness check.
2. The gate is now two-part, per the corrected scheme in docs/GRADER.md:
   - ONE-STEP gate: a single map step (driver --map --chain 1) must match the numpy
     reference to 1e-14 rel L2. This carries the precision contract: it catches
     fp32-seeded maps (~2.6e-12 one-step) and every cheaper shortcut, chaos-free.
   - CHAIN gate: chain_rel <= 300x the worst library chain_rel measured on the SAME
     chain (same seed, same m). This only exists to catch gross cheats (fp32 interior,
     skipped steps) and can no longer punish honest rounding.
3. Directives: exact one-step map (plain divide or a Newton ladder that reaches 1e-14
   one-step) remains MANDATORY at every size. Given that, stop spending flops on chain
   drift — it is not your error, it is chaos amplifying everyone's roundoff equally.

Speed picture unchanged: mine fft_warm_solutions/ and fft_v5v6_solutions/ structure.
Round 7 flipped L=8/13/17 by doing exactly that. Remaining targets: L=6 (validate the
0.074-class entry under the one-step gate, then push), and every cell vs the warm
cohort numbers which we are measuring on our node now.
