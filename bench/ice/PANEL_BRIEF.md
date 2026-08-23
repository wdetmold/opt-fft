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
