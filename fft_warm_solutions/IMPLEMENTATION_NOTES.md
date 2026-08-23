# Implementation notes from prior campaigns (16 attempts, this machine class)

Distilled from the microbenchmark forensics and final solutions of sixteen
earlier attempts at this exact problem on this VM class (virtualized Ice
Lake-SP, "Intel Xeon Processor @ 2.60GHz", fam 6 model 106, 4 vCPU). These are
measurements made by prior sessions: treat them as strong priors, and
re-verify anything you make load-bearing.

## Machine model (as measured by prior attempts)

- **Two genuine 512-bit FMA pipes**: 1.88–2.2 FMA/cyc sustained reg–reg with 8
  independent accumulators; ymm rate = zmm rate; no AVX-512 license downclock.
- **You cannot feed them from memory**: embedded-broadcast `{1to8}` FMA
  1.0–1.25/cyc; full 64B memory-operand FMA ~1/cyc; 512-bit loads 1.0–1.4/cyc;
  one session measured a global cap of **~2.1 vector uops/cycle total** (any
  mix; ymm does not escape it). Table-driven twiddle sweeps equilibrate near
  1.25 FMA/cyc. Winning responses: register-resident constants for primes
  (only (p−1)/2 distinct cos/sin), phase-split sweeps with zero in-sweep
  loads, j-outer matvecs so each loaded element feeds all accumulators.
- **Clock**: brand string 2.60 GHz; latency-anchored calibrations gave
  2.9–3.11 GHz under sustained zmm FMA (one session measured 2.5). rdtsc
  ticks at 2.6 — convert carefully.
- **No PMU**: `perf` does not exist and cannot be installed. Prior sessions
  used rdtscp/clock_gettime microbenchmarks + `objdump` opcode histograms.
  Disassemble your own benchmark loops — GCC has CSE'd probe kernels into
  impossible numbers more than once.
- **Divider/map**: zmm `vsqrtpd`+`vdivpd` ~34 cyc/slot throughput; the map
  `z/(1+|z|)` is best done with rsqrt14 + 2 Newton on the FMA pipes plus ONE
  divider-unit op per point (one exact `vdivpd` or one hw sqrt),
  software-pipelined behind the FFT so divider latency hides. Caution:
  `vrsqrt14pd`/`vrcp14pd` measured ~1.3 cyc in one session and ~10 cyc
  ("microcoded") in another — benchmark before relying; the float-seeded
  fallback (`cvtpd2ps`→`vrsqrtps`→Newton in fp64) is safe.
- **Denormal assists are the stealth killer**: transient operands take
  assists even when stores are masked (one attempt lost 3× at L=23 to an
  out-of-bounds constant-table read whose garbage bits were denormal).
  Zero-pad blocked tables; FTZ/DAZ in benchmark harnesses; never iterate a
  contractive map in place in a timing loop.
- **Memory**: 48 KB L1d / 1.25 MB L2 / 54 MB L3 nominal (effective L3 less;
  single-core L3 BW ~12–48 GB/s, neighbor-dependent — it halved mid-run for
  one attempt). 4K aliasing is epidemic at these strides: pad plane strides
  off 4K multiples, keep 64B alignment (line-crossing 512-bit loads ~4 cyc),
  skew re/im/c allocations. THP is madvise-mode and sometimes not granted —
  verify via smaps; when denied, 2MB-aligned allocations HURT via set
  conflicts. Software prefetch mostly loses; restructure instead (tiling to
  cut concurrent streams, lazy map application in the next contiguous pass,
  laying out c in consumption order).
- **Codegen**: GCC 13.2. Straight-line SSA giants spill (constant hoisting
  exhausts 32 zmm); cures that worked: staged two-phase codelets through L1
  scratch, `#pragma GCC unroll 1` loop kernels with runtime-indexed constant
  tables, j-outer accumulator sweeps, generator-level text-order interleaving
  (GCC does no pre-RA scheduling on x86). PGO, clang, and zig were each tried
  and rejected. `-O3 -march=native -ffp-contract=fast -fno-math-errno` is the
  common core.

## What the best prior solutions did (see their READMEs for detail)

- Lane-major SoA batching: 8 volumes per zmm register for the small sizes —
  every 1-D pass pure vertical SIMD, zero shuffles/masks.
- PFA (Good–Thomas) for 6/36/45, Cooley–Tukey 8×8 for 64 (one attempt used a
  digit-transposed layout to fuse the transpose into the codelet's natural
  8×8 register tiles), symmetric-folded direct DFTs for primes 13/17/23 with
  register tiling (straight-line prime codelets were frontend/store-bound).
- Twiddles baked as hex-literal doubles computed in long double.
- Hugepage arenas, padded strides, NT stores for write-once outputs,
  prebuilt .so shipped alongside the source (avoids compile-at-import).
- **Keep import fast and deterministic**: grading times `transform()` against
  a startup baseline; heavy variable import-time work has voided attempts'
  measurements (env failure) in prior cohorts.

## The roofline (why the score curve tops where it does)

Best-known DFT op counts (PFA/Winograd 6/36/45, split-radix 8, tangent-FFT
64, Rader/Winograd primes) plus the map, executed at the measured ~2.1-vuop
issue ceiling: **1.83 s for the graded workload = 7.3× the MKL reference
(ratio 0.137)**. That is where score = 1.0 sits. The best prior attempt
reached ~4.0× MKL (~53% of roofline); the unclaimed factor of ~2 lives in the
feed/issue equilibrium (prior sessions estimated ~20% more from hand-written
asm on the sweeps and declined it — the rest requires beating the equilibrium
structurally).

## Caveats on the provided code

The solution directories were reconstructed from session transcripts. Where
`implementation.c` was generator-emitted, the generators are included and
regenerate it deterministically; constants regenerated off-host can differ in
the last ulp (x86 80-bit long double vs other hosts) — regenerate on this
machine or verify against the gates. Prebuilt `.so` files are not included;
every `solution.py` rebuilds from source at import. All provided code
complied with the no-FFT-library rule; that rule applies unchanged to
whatever you ship.
