# ROOFLINE — theoretical-perfect performance for the geom benchmark workload

Purpose: a principled goal post for the chained-FFT benchmark (the Taiga
`fft3d-fixed-geometry-opt` problem and the `bench/geom` kernel set): best-known
DFT operation counts per size, executed at the *measured* ceilings of one core
of the grading VM (virtualized Ice Lake-SP, family 6 model 106). "Theoretically
perfect" here is a modeled bound, not an information-theoretic one — the true
complexity of the DFT is open — so the bound is: cheapest credible algorithm
flop count / measured machine throughput, vs. the minimum L1 traffic, whichever
is larger.

Date: 2026-08-22. Method at the bottom.

## Workload

Per size L: B_L batched volumes, m_L iterations of
`z = FFT3(x) + c ; x <- z/(1+|z|)` (forward unnormalized 3D DFT, complex128,
map elementwise). The graded (3×) chain lengths:

| L | 6 | 8 | 13 | 17 | 23 | 36 | 45 | 64 |
|---|---|---|----|----|----|----|----|----|
| B_L | 64 | 64 | 32 | 32 | 16 | 8 | 4 | 2 |
| m_L (3×) | 14568 | 7716 | 3834 | 294 | 495 | 192 | 531 | 402 |

Total ≈ 1.36e9 point·iterations.

## Best-known 1D DFT operation counts (real flops, complex input)

All "derived-exact" counts were constructed from the algorithm and verified by
an instrumented op-counting implementation against `numpy.fft.fft` (errors
≲ 2e-15 over random complex128 trials). Candidates per size, best in bold:

| L | best F1D | m + a | algorithm (best) | brackets (other candidates) |
|---|---------|-------|------------------|------------------------------|
| 6 | **44** | 8m+36a | Good–Thomas PFA 2×3, Winograd modules (= FFTW n=6 codelet) | CT-with-twiddles 56; direct fold 46 |
| 8 | **56** | 4m+52a | split-radix exact (4N lgN − 6N + 8); = Winograd-8 | radix-2 bracket 120 |
| 13 | **228** | 40m+188a | Winograd 13 module (Rader + minimal nesting) *(literature)* | Rader via PFA(3,4) FFT-12: 292 (derived); direct symmetric 336 |
| 17 | **384** | 70m+314a | Winograd/Johnson–Burrus 17 module *(literature)* | Rader via split-radix-16: 428 (derived); direct symmetric 576 |
| 23 | **1056** | 248m+808a | Rader via two FFT-22 (Good–Thomas 2×11) *(formula)* | direct symmetric 1056 (derived; tie); Winograd-style 1250 |
| 36 | **560** | 80m+480a | Good–Thomas PFA 4×9, derived Winograd 9 module | Nussbaumer-table variant 576; CT 6×6 672 |
| 45 | **936** | 190m+746a | Good–Thomas PFA 9×5, derived modules | WFTA nested 944; CT 1128 |
| 64 | **1152** | — | modified split-radix / tangent FFT (Johnson–Frigo) *(literature)* | split-radix (Yavne) 1160 (derived); radix-2 bracket 1920 |

3D cost per complex point = 3·F1D(L)/L (three axis passes of L² transforms):

| L | 6 | 8 | 13 | 17 | 23 | 36 | 45 | 64 |
|---|---|---|----|----|----|----|----|----|
| FFT fl/pt | 22.0 | 21.0 | 52.6 | 75.5 | 137.7 | 46.7 | 62.4 | 54.0 |

Map cost: taken as 16 fl/pt (|z|² via mult+FMA, rsqrt seed + 2 Newton steps,
sqrt reconstruction, 1+r·q, one exact divide-equivalent, two scaling mults —
matches the mapF/RSQRTN_DIV designs the optimized kernels converged on).

Note on primes in practice: the minimal-flop prime algorithms (Rader/Winograd)
have poor locality and irregular data flow; the kernels that actually win on
this machine use the direct symmetric half-matrix (more flops, perfect
register/SIMD behavior). The roofline deliberately uses the *cheapest* counts,
so it is generous to the solver — a true lower envelope.

## Machine model (measured, not nominal)

Consolidated from the microbenchmark forensics of seven independent
optimization sessions on the grading VM (rdtscp/clock_gettime probes +
objdump audits; no PMU exists in the VM):

- Two full 512-bit FMA pipes; 8 independent-accumulator probes sustain
  ~1.9–2.2 FMA/cyc reg–reg in every session; ymm = zmm rate; no AVX-512
  license downclock observed.
- Clock: brand string 2.60 GHz, but latency-anchored calibrations give
  2.9–3.11 GHz sustained under zmm FMA (one session measured a fixed 2.5).
  Model uses 2.9 GHz (mid).
- Ceiling ladder (flops/cycle, FMA = 2 flops):
  1. **ISA ideal**: 32 — unreachable in real kernels.
  2. **Measured reg–reg sustained**: ~30 (1.88 FMA/cyc with register-resident
     constants).
  3. **Measured issue-cap model**: ~16.8 — the VM caps at ~2.1 vector
     uops/cycle *total* (any mix); with a realistic FFT mix of FMA/add/load/
     store this is the binding constraint.
  4. Table-sweep equilibrium (broadcast+2FMA triples): ~20.
- L1: ~64 B/cycle total load+store; mixed same-cycle load/store pairs cost
  extra; 512-bit loads crossing a cache line ~4 cycles (pad strides).
- Minimum L1 traffic for a cache-resident in-place 3-pass FFT + streamed c +
  fused map: ~112 B per point per iteration → 1.75 cyc/pt at 64 B/cyc.

FMA pairing matters: these DFT graphs are add-dominated (pairable fraction
0.14–0.36 of flops), so a pairing-aware FP-port bound sits well below the
ISA-ideal 32 fl/cyc.

## The roofline

cycles/pt(L) = max( (FFT fl/pt + 16) / ceiling , 1.75 ) ;
T = Σ_L B_L · m_L · L³ · cycles/pt / 2.9 GHz.

The memory bound (1.75 cyc/pt) never binds — the workload is compute-limited
at every size on this core.

| ceiling model | T_total (3× workload) | ratio r vs MKL (13.39 s) | speedup over MKL |
|---|---|---|---|
| ISA ideal (32 fl/cyc) | 1.05 s | 0.078 | 12.8× |
| FP-port, pairing-aware | 1.64 s | 0.123 | 8.1× |
| **measured issue-cap (≈"perfect on this VM")** | **1.83 s** | **0.137** | **7.3×** |

Reference points on the same (3×) workload, same grader:

| implementation | time | r vs MKL |
|---|---|---|
| MKL DFTI sequential (SOTA) | 13.39 s | 1.000 |
| fft3d_best kernels (this repo, Cascade-Lake-tuned) | 9.44 s | 0.705 |
| best AI attempt observed (main-v4, scaled) | ≈4.5 s | ≈0.34 |
| measured-ceiling roofline | 1.83 s | 0.137 |

## Goal-post interpretation

ADOPTED (problem v7, 2026-08-23): full marks are anchored AT the
measured-ceiling roofline — score 0 above MKL, 0.1 at parity, linear to 1.0 at
r ≤ 0.137 (7.3×). A score of 1.0 therefore means "theoretically perfect on
this VM"; the best AI attempts to date sit at ≈53% of the roofline (r ≈ 0.25,
score ≈ 0.88 on this curve). Problem versions ≤6 used a full-marks anchor of
r ≤ 1/6 = 0.167 (≈82% of the roofline); their recorded scores are on that
older, slightly more generous curve.

Precision caveat: the bound above is for a chain carried entirely in fp64. A
solver exploiting reduced precision on the short-chain sizes (where chaotic
amplification is small) could undercut the fp64 roofline on ~70% of the
workload; the benchmark's per-size chain gates must exclude that (verified:
complex64 chains pass a flat 1e-3 final-state gate at L = 13–64 but fail
per-size gates set at ~10³ × the measured honest fp64 divergence).

## Caveats

- Prime counts at 13/17 (Winograd modules) and 23 (Rader-22) carry
  literature/formula confidence, not instrumented derivation; brackets from
  derived-exact alternatives are +28%/+11%/tie respectively, so the totals move
  by at most a few percent under the conservative choice.
- The map count (16 fl/pt) is a design choice; hardware sqrt+div formulations
  trade flops for divider-port time and land similar.
- Ceilings are for THIS VM (issue cap ~2.1 vuops/cyc is below bare-metal
  Ice Lake-SP front-end width; L1 ~1×64B/cyc vs 2 loads + 1 store bare-metal).
  On bare metal the roofline drops by roughly the issue-cap/port ratio.
- Honest-divergence and fp32 numbers are pair- and machine-dependent; margins
  of ~10³ were used throughout.

## Method / provenance

Op counts: eight independent derivations (one per size), each constructing the
algorithm, counting adds/mults symbolically, and verifying the instrumented
implementation against numpy to ≲2e-15; formula brackets (radix-2 5N lgN,
CT-with-twiddles) computed alongside as sanity envelopes. Machine model:
consolidated from the measured microbenchmarks in the seven graded-attempt
transcripts of main-v4 (independent sessions, same VM), including the
2.1-vuop issue cap, the 1.88 FMA/cyc reg–reg sweep, the ~64 B/cyc L1 limit,
and three independent clock calibrations. Assembly script and inputs:
Taiga-side scratch (`roofline_inputs.json`, `roofline_full.json`); grader
timings from bestcheck v14 and the controlled interleaved probes.
