# 10 — Ice Lake Under Glass: forensics from the grading tier itself

> **Provenance.** Transcribed from Will's artifact "Ice Lake Under Glass"
> (claude.ai/code artifact `3eaceed9`, 2026-08-22): a hand-optimizer's digest of the
> microarchitectural forensics performed by **seven independent optimization agents** on the
> **Taiga grading tier** — a KVM/Firecracker slice of an Ice Lake-SP Xeon (family 6, model
> 106, "Intel Xeon Processor @ 2.60GHz", 4 vCPU of an n2-standard-128). All seven optimized
> the same batched 3D complex-double FFT workload, single core, AVX-512. Sources: full
> session transcripts of the seven graded attempts (main-v4, 2026-08-22), independently
> extracted and cross-checked; reconstructed final sources in `fft_v4_solutions/`.
>
> **Why this section outranks §08/§09 where they conflict:** it is measured on the machine
> that grades us, which is a *virtualized* Ice Lake whose behaviour differs from bare metal
> in ways that decide kernel design (no PMU, halved load throughput, session-dependent
> bandwidth). Evidence grades used throughout, as in the original:
> **[consensus]** ≥3 independent measurements agree · **[single]** one session's
> measurement · **[contested]** sessions disagree.

## 0 · Ground rules the machine imposes

**[consensus]** There is **no PMU**. `perf` does not exist in the VM, `perf_event_open`
fails, and it cannot be installed. Every port model below is inferred from
`rdtscp`/`clock_gettime` microbenchmarks plus `objdump` instruction counting. Nobody ever
read a real port counter.

**[contested]** **Clock accounting is the largest systematic error.** Nominal is 2.60 GHz
(TSC ticks at 2.6). Four sessions calibrated the true core clock four ways and got four
answers: **3.107 GHz** from a 4-cycle-latency dependent FMA chain (the cleanest method),
**"fixed 2.5 GHz"** from a 1-cycle dependent scalar-add chain, **~3.0** assumed via a
1.155×TSC factor, and **2.33 GHz effective** under sustained load. The spread is real
(different hours, different steal), so treat any absolute cycles-per-element figure from
this tier with ±15% salt. No session detected an AVX-512 license downclock (three checked
explicitly: zmm, ymm, and scalar chains run at the same derived clock).

**[consensus]** Run-to-run noise is severe (up to 2–3× on identical runs; one grader rep of
the same reference spanned 2.65–15.98 s). The only trustworthy protocol: build all variants
as separate `.so`s, load them into **one** process, interleave calls round-robin, take
best-of-6–12. Every session independently converged on this.

## 1 · The FMA engine, in three layers

### Layer 1 — the pipes are all there

**[consensus 7/7]** The VM exposes **two genuine 512-bit FMA pipes**. Every session ran the
same probe shape — 8 independent zmm accumulator chains, `a_i = fmadd(a_i, b, c)`, values
pinned away from denormals — and measured **1.88–2.2 FMA/cycle** register-register, with
**ymm exactly equal to zmm** (no width halving, so this is a Gold/Platinum-class 2×FMA die,
not a Silver). Eight chains is the right depth: 2 pipes × 4-cycle latency = 8 in flight.
Nobody measured FMA latency directly; the 4-cycle assumption is implied by saturation.

### Layer 2 — but you cannot feed them

**[consensus 5/7, independently]** The headline discovery, found five separate times with
five different benchmark designs: **FMA throughput collapses the moment operands come from
memory.**

| operand form | throughput | measured by |
|---|---|---|
| reg-reg zmm FMA, no spills | 1.88–2.2 / cyc | all seven |
| embedded-broadcast `{1to8}` FMA | 1.0–1.25 / cyc | 1760b1bf (1.1), f7f192ab (1.2), a31f5f85 (1.0), dd9fa88c (1.2), 1000f989 (1.25 in-context) |
| full 64-byte memory-operand FMA | ~1 / cyc | a31f5f85, 1760b1bf |
| 512-bit loads (pure) | 1.0–1.4 / cyc | a31f5f85 (1.4), dd9fa88c (~1.0) |
| mixed zmm load+store pairs | 4.8 cyc / pair | dd9fa88c (batching 8 loads then 8 stores recovers most of it) |
| any vector-uop mix, total | ~2.1 uops / cyc | a31f5f85 (`ipctest.c`; ymm does not escape the cap) |

Bare-metal Ice Lake-SP does 2×64B loads + 1×64B store per cycle and folds broadcasts into
the load µop for free; **this tier does not**. Two sessions explicitly pinned the gap on the
VM. The most useful formulation is a31f5f85's global model: **~2.1 vector uops/cycle total,
any mix** — under which a fused memory-FMA costs like two uops, and the objective flips
from port balancing to **total vector-uop minimization**. 1000f989 measured the practical
equilibrium for the canonical FFT twiddle sweep — one broadcast + two FMAs per (j,k) — at
**1.25 FMA/cycle**, against 2.1 in an idealized microbenchmark on the same machine.

### Layer 3 — what it is not

**[single source, refuted hypothesis]** f7f192ab chased the broadcast penalty into the
front end: at ~11 bytes per disp32 broadcast-FMA and a conjectured 16 B/cycle legacy decode
("µop cache disabled by mitigation"), the numbers fit. It then engineered EVEX disp8
compression through the whole generator — asm-laundered window pointers, offsets held under
1016 bytes, encodings verified at 7 bytes in the disassembly — and **performance did not
move**. Hypothesis retracted. The cap is in the load/issue path, not decode. Also ruled
out: value dependence (FMA core is value-independent; 8dc1a96d), frequency effects
(interleaved fixed-work probes).

### The two winning responses to Layer 2

Both scored at the top, and they pull in opposite directions:

**(a) Refuse memory operands.** A prime-p DFT needs only (p−1)/2 distinct cosines and
sines. 1760b1bf's phase-structured codelets split the sweep into an A-phase (all cos
constants + inputs live in registers, zero loads inside the FMA sweep) and a B-phase (same
for sin), with k-blocks sized to the 32-register file — **1.6× on the isolated 23-point
kernel**. a31f5f85 and 1000f989 reached the same place via register-resident tables and
j-outer matvec (each loaded element feeds **all** k accumulators).

**(b) Accept 1.2/cyc broadcasts to kill spills.** 8dc1a96d went the other way:
runtime-indexed constant tables inside `#pragma GCC unroll 1` loops, compiling to `{1to8}`
embedded-broadcast FMAs — because the alternative (GCC hoisting every constant into a
register) spilled zmm to stack, which is far worse. Its uop audit put the resulting prime
kernels at 93% (L=13) and 74% (L=17, register file exactly full) of the machine ceiling.

The ordering that reconciles them: **reg-resident-without-spills > embedded-broadcast >
spilling**. Choose (a) when the constant set fits the register file after phase-splitting;
fall back to (b) the moment it doesn't.

### What this means coming from Cascade Lake

Our kernels were tuned on a 1×512-FMA Cascade Lake, where the FMA pipe itself was the wall.
On this tier the FMA capacity doubles but the **feed does not** (≤1.4 loads/cyc, ~2.1
uops/cyc issue). Any twiddle sweep that streams constants from memory — fine on CLX, where
one FMA/cycle left load slack — lands feed-bound here at ~1–1.25 FMA/cyc, leaving nearly
half the second pipe idle. The transferable fixes are structural, not scheduling:
phase-split register residency, j-major reuse, and sign-folded constants (bake the negated
sines into tables; kill the per-use `XOR`+rebroadcast). 1000f989's estimate of remaining
headroom over GCC for hand-written asm on the equilibrium-limited sweeps: **~20%** — it
declined to take it.

## 2 · Divider, rsqrt, and the nonlinear map

**[consensus]** Hardware `vsqrtpd`/`vdivpd` zmm are slow in throughput terms — sqrt ~20–24
cyc/vec, div ~16 cyc/vec, combined sqrt+div ~34 cyc/slot (1000f989, after twice fixing its
own benchmark: div-by-1.0 constant-folded, then iterated sqrt converging to a fixed point)
— **but the divider is a separate unit that runs in parallel with the FMA pipes**, and on
this workload the FMA pipes are the scarce resource.

**[consensus 4/7 convergent]** The winning shape for `z/(1+|z|)`, discovered independently
four times: **burn the divider exactly once per point and do everything else with Newton on
the FMA pipes**, software-pipelined so divider latency hides under FFT work. Two equivalent
instantiations: rsqrt-Newton for the magnitude + one exact `vdivpd` (1000f989's mapF,
8dc1a96d), or one hardware `vsqrtpd` + rcp14-Newton for the reciprocal (a31f5f85's V2;
0f45aeae alternates the two per output at L≥36). The exact final divide also protects the
1e-14 gate. Montgomery batch inversion (one divide per 4 slots) is correct but a wash once
the divider is hidden — measured and rejected.

**[contested]** **`vrsqrt14pd`/`vrcp14pd` throughput is genuinely disputed**: dd9fa88c
measured ~1.3 cyc/op (4-chain throughput test) and used them happily; 1760b1bf isolated
them at **~10 cyc each, "likely microcoded on this core"** (its all-FMA Newton map came out
slower than hardware sqrt, which is how it noticed) and switched to legacy float seeds —
`cvtpd_ps → vrsqrtps/vrcpps → cvtps_pd` — plus pure-FMA Newton. Two other sessions used the
14-bit seeds without complaint. If a Newton ladder underperforms on this tier, benchmark
the seed instruction before blaming the ladder.

**Denormal assists are the stealth killer.** Three sessions lost time to them. The
masterpiece: 8dc1a96d's L=23 kernel ran 3× slow (17.9 vs 5.1 ns/el) **only on realistic
data**. Cause: the padded 12th k-slot's FMAs — executed, outputs discarded — read past the
trig table into adjacent int32 index tables whose bit patterns are denormal doubles; every
such FMA took an assist. **Guarding the store does not guard the arithmetic.** Zero-pad
constant tables to the blocked size. Also: set FTZ/DAZ (`MXCSR |= 0x8040`) in benchmark
harnesses, and never iterate a contractive map in place in a timing loop — two sessions
watched their own benchmark data drift into denormal range and poison the numbers.

## 3 · Memory system

**[consensus]** Topology: 48 KB L1d / 1.25 MB L2 per core / 54 MB shared L3 (nominal).
Measured bandwidth is far below bare metal and **session-dependent** (neighbor-dependent):
L3 12–48 GB/s, DRAM 8–23 GB/s across the seven sessions. Effective L3 capacity is below
nominal (1000f989: a 34 MB working set already misbehaved). Design consequence everyone
reached: **iterate each volume through all m steps while cache-resident; never sweep
passes across volumes.**

**[consensus 4/7 independently]** **4K aliasing and cache-set conflicts are epidemic at
these strides.** Greatest hits: a 64 KB row stride at L=64 putting all 64 rows in one
L1/L2 set; a 1 KB stride mapping 64 rows onto 4 sets; plane strides near multiples of 4096
falsely aliasing load/store pairs in the store buffer; two **identical builds** timing
differently because BSS placement changed the re/im arrays' page offsets (fix: one
page-aligned arena, arrays skewed 1 KB apart mod 4K). Everyone ended up with empirical
padding tables; 1000f989's verdict after scanning: "padding matters more than set-math
suggests." Keep strides at 64B multiples (unaligned 512-bit loads crossing a line cost
~4 cyc — 8dc1a96d), then skew planes off 4K by odd line steps.

**[contested]** **Transparent huge pages are machine-state roulette.** THP is in `madvise`
mode: two sessions got hugepages (verified via `smaps` `AnonHugePages` after
`madvise(MADV_HUGEPAGE)` on a 2MB-aligned arena), one session was silently denied — and for
it, forcing 2MB-aligned allocations actively **hurt** (every buffer landing on the same
sets; +17% from deliberately staggering allocations by page-plus-5-lines instead). Verify
with smaps; never assume.

**[consensus]** **Software prefetch mostly loses; restructuring wins.** Across sessions,
explicit prefetch was neutral-to-harmful in nearly every placement tried (the exceptions:
one plane-ahead T0 pattern at distance 2–3, and a "lite" 16-line variant). The hardware
prefetchers are strong — f7f192ab measured its worst-case 64-stream strided pattern reading
at 243 GB/s. What actually moved large-L: cutting concurrent streams (two-stage tiled
x-passes: 128+ streams → 16–32), the **lazy map** (keep the buffer raw between iterations;
apply the map at the next iteration's contiguous pass, where `c` streams sequentially),
re-laying `c` into exact consumption order, and 8dc1a96d's L=64 z-split layout (zmm lanes =
z-octants, alternating natural/bit-reversed forms so no bit-reversal pass is ever
materialized: working set 68 → 9.3 MB/volume, 2.6× at B=1). NT stores only for write-once
outputs of L3 scale, alignment-gated (they fault on odd-L strides and cost a DRAM
round-trip on small outputs).

## 4 · Fighting GCC 13.2

**[consensus]** The universal failure mode: on big straight-line AVX-512 codelets GCC
hoists every broadcast constant into a register, exhausts the 32-zmm file, and spills (one
session counted ~1700 stack moves in a single kernel; another 410). Five cures shipped,
matched to kernel shape:

1. **Staged emission** — split 36/45/64 codelets into two stages through an L1 scratch
   array so each stage's live set fits (1760b1bf, dd9fa88c, 1000f989).
2. **Unroll-1 loop kernels with runtime-indexed tables** → `{1to8}` embedded-broadcast
   FMAs, no hoisting (8dc1a96d; needs `#pragma GCC unroll 1` on every loop or GCC
   re-unrolls and re-hoists).
3. **Per-use table slots** — deliberately un-deduplicated constants so each is folded as a
   one-use memory operand (f7f192ab), plus sign-folded (negated-sine) tables.
4. **j-outer matvec with k-indexed accumulators** — the one shape GCC allocates cleanly at
   24 live accumulators (1000f989: after this, nothing beat GCC — not PGO, not
   explicit-register generated intrinsics, not flag sweeps).
5. **Generator-level ILP** — GCC does no pre-RA scheduling on x86, so it keeps your text
   order: round-robin-interleave the SSA lines of 3–4 independent k-blocks (0f45aeae's
   ZIPP, +6–23% on primes).

Adjacent facts: PGO was tried twice, no gain either time. `-fschedule-insns
-fsched-pressure` helps prime passes ~20% but hurts 45/64 — resolve with per-function
`__attribute__((optimize(...)))`. Move elimination is disabled for SIMD on ICX (the known
erratum), so reg-reg `vmovapd` copies burn real port slots. Empty inline asm
(`__asm__("" : "+r"(ptr))`) is the precision tool for laundering pointers/values past the
constant folder. The universal methodology substitute for PMU: `gcc -S`/`objdump` + regex
histograms of `vfmadd`/`{1to8}`/`rsp` counts — and grep your **benchmark's** disassembly
too: two sessions shipped bogus numbers when GCC CSE'd the probe loop (one "measured" a
physically impossible 17 FMA/ns before checking).

## 5 · Scorecard

| attempt | score | C_opt | signature contribution |
|---|---|---|---|
| 1000f989 | 1.00 | 2.014 s | Lane-interleaved SoA, zero-shuffle passes; L=64 z-split; the 1.25 FMA/cyc sweep-equilibrium analysis |
| 8dc1a96d | 1.00\* | 1e-6 s\* | Embedded-broadcast loop kernels; denormal-assist detective work; allocator tuning (\*C_opt hit the timer floor — measurement artifact, not cheating) |
| 1760b1bf | 0.96 | 1.500 s | Phase-structured register-resident prime codelets; rsqrt14/rcp14 microcode discovery; float-seeded Newton |
| a31f5f85 | 0.81 | 2.306 s | The ~2.1 vector-uop/cyc global-cap model; grouped 8-volumes-in-lanes mode; allocation staggering |
| dd9fa88c | 0.76 | 2.038 s | ~1×64B/cyc L1 budget discovery; j-major emission; expression-DAG generator with fused three-variant leaves |
| 0f45aeae | 0.44† | 2.355 s | Generation-time numeric verification of every codelet; ZIPP text-order ILP (†score crushed by a noise-deflated SOTA measurement) |
| f7f192ab | 0.00‡ | — | Decode-bound hypothesis rigorously built and refuted via disp8 compression (‡zeroed on the 500 MiB /workdir limit, not on quality) |

## 6 · Direct transfers to the fft3d_best kernels

* **Prime kernels (17, 23):** if the twiddle sweeps consume broadcast/memory constants,
  they are feed-bound at ~1–1.25 FMA/cyc on this tier. Phase-split them: cos sweep with all
  (p−1)/2 constants register-resident, then sin sweep, k-blocked to the register file.
  Measured value: **1.6× on the isolated 23-point codelet**. Bake negated sines into the
  tables.
* **L=64:** the z-split octant layout (natural/bit-reversed alternation, cross-lane masked
  DFT8 butterflies, form-B constants derived by in-register transposes) compresses the
  working set 7× and removes the bit-reversal pass. This was the single biggest structural
  win of the best run.
* **The map:** one divider op per point, rest Newton on FMA, pipelined a row ahead.
  Benchmark `vrsqrt14pd` on the target before trusting it; keep the float-seed fallback in
  your pocket.
* **Layout hygiene:** audit every stride against 4K and 64B; skew re/im/c allocations off
  shared page offsets; verify THP took (or deliberately stagger if it didn't).
* **Benchmark protocol on this tier:** interleaved best-of-N in one process, FTZ on, fresh
  realistic data each round, disassemble the probe before believing it, and convert cycles
  at a clock you calibrated that hour — not at 2.6.
