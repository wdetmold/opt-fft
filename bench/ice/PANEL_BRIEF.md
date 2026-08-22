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
