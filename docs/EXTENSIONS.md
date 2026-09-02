# Extensions roadmap — where this work goes next

Recorded 2026-09-02, after the generalize campaign closed (arbitrary-L 3D c2c fp64,
3.5–3.6× best stock library across 11 cells + B=1 hardened via r14, validated on two
Ice Lake nodes, Cascade Lake, and Frigo–Johnson's benchFFT). These are candidate
directions, not committed campaigns; each notes what transfers for free and what is
genuinely new.

## What every extension inherits for free
- The 1D length-L codelets (each 3D axis pass already IS a batched 1D DFT), the
  genfft custom codelets in bench/gen/sota/codelets/.
- The factorization planner, the plan-time race, and the per-host wisdom cache
  (all dimension- and hardware-agnostic).
- The two-part chaotic-chain gate (self-calibrates per chain, so it re-anchors for
  any d / core count / device automatically).
- The batch-lane SoA split-complex layout, and the r14 shared-execute()/chain() fast path.
- The measurement stack: gated chain harness + PMU toolkit (/tmp/perf, tools/pmu.sh)
  + benchFFT (bench/gen/benchfft_ours/) + the rival-forensics kit (rivaltime/gradercheck).

## What every extension must rebuild
- The 3D-hardcoded surface: fft3d_* API (single --L, L:B:m cases), the driver's fixed
  3-axis loop, check.py's fftn(axes=(-3,-2,-1)), and each kernel's PASS STRUCTURE
  (pencil sweeps + in-register 8×8 inter-axis transposes are woven per-kernel even
  though the codelets inside are not).
- The cache/roofline regime map and "which technique wins where" conclusions are
  entirely working-set-specific and do NOT transfer — each target needs its own PMU
  audit (cheap with the toolkit, not free).

---

## A. Other dimensions (d = 1, 2, 4)

Working sets (in+out, B=1, 16 B/pt) — the number that decides difficulty:

| L | 1D | 2D | 3D | 4D |
|---|---|---|---|---|
| 32 | 1 KB | 32 KB | 1 MB | 32 MB |
| 64 | 2 KB | 128 KB | 8 MB | 512 MB |
| 100 | 3 KB | 313 KB | 30 MB | 3 GB |
| 128 | 4 KB | 512 KB | 64 MB | 8 GB |
(Ice Lake-SP: L1d 48 KB, L2 1.25 MB, L3 ~54 MB.)

- **d=1 — least work, least payoff.** Bare batched 1D DFT: no transposes, no pencils,
  codelets apply directly. But L1-resident everywhere (compute-bound), and 1D is where
  FFTW/MKL are most obsessively tuned AND where our structural edge disappears (a prime
  axis-DFT is no longer amortized L² times per volume — only across the batch). Margins
  shrink toward parity except at awkward primes. Build for completeness only.

- **d=2 — the clean retarget (recommended first).** Strict subset of 3D (two passes,
  one transpose); kernels nearly drop in. Working sets are the comfortable middle
  (L2-resident to L≈64, L3 at 100–128) — the regime our kernels are best tuned for, with
  the r11 traffic work applying at the top end. Widely used (imaging, PDE), less saturated
  than 1D, small-prime advantage carries. Best effort-to-differentiation ratio.

- **d=4 — hard, and the domain-relevant case (lattice QCD).** Working sets explode:
  L=32 already 32 MB, L=64 512 MB, L=100 3 GB — MEMORY-BOUND from modest L, even at B=1.
  This inverts the campaign: r11's L=100 traffic optimization becomes the central game,
  and the still-unspent two-axes-per-pass fusion lever becomes MANDATORY. Two 4D-specific
  new requirements:
  1. **Asymmetric L³×T.** Real LQCD lattices are not cubic; the whole project assumes
     L^d. Per-axis different lengths is expressible by the planner but NO kernel was built
     for it — a genuine new capability, not a retarget. This is the load-bearing item.
  2. B×L⁴ is so large that **B=1 is the only feasible regime** at meaningful L — exactly
     the path r14 just hardened, so that groundwork is already in place.

## B. More multicore work

The earlier mt (multicore) campaign was single-socket-bounded and did not exhaust the
regime. Open items:
- **NUMA / multi-socket**: wallaby is 2× Xeon Gold 6448Y; the mt harness never crossed
  the socket boundary. Cross-socket 3D FFT is a communication problem (transpose = the
  all-to-all), and the memory-hierarchy lit (staging/04) applies directly.
- **Thread-scaling of the chain**: the fused FFT+map chain is the optimization unit; a
  multicore chain wants the map parallelized without a barrier per step (the "hard fusion
  barrier" the GPU vendors solve with callbacks — no CPU analog exists, so owning the code
  is again the only route).
- **MANDATORY for 4D**: even single-node 4D at usable L exceeds one core's practical reach;
  4D and multicore are not independent axes — a serious 4D effort is a multicore effort.

## C. More GPU / multi-GPU work

The GPU phase (docs/CAMPAIGN.md) is designed but unbuilt beyond API/driver/two baselines:
slab decomposition, single process, L=64/96/128/192/256, cufftXt distributed baseline,
8-GPU a100l reservation. Open items:
- Finish and arm the mgpu harness (bench/mgpu/ is half-built: needs cufftXt baseline,
  sweep/submit/tryout, cases, brief).
- The FP64 tensor-core (DMMA) small-dense-DFT route (L=13/17 GEMM-style) proved out at
  0.33× in the earlier GPU work — carry it into the mgpu phase.
- **MANDATORY for 4D**: 4D at production L is a multi-GPU / multi-node problem by memory
  alone (L=100 4D = 3 GB per field before batching); a single device cannot hold it.
  4D therefore forces BOTH multicore (host side, decomposition/comm) and multi-GPU
  (device compute) — they are prerequisites, not parallel options.

---

## Dependency summary
- d=2: standalone, cleanest next win.
- Multicore-NUMA: standalone, extends the mt campaign.
- Multi-GPU: standalone, finishes the designed GPU phase.
- **d=4 (the physics target): depends on ALL THREE** — asymmetric-lattice kernels +
  multicore host decomposition + multi-GPU device compute, with traffic optimization
  (the unspent two-axes-fusion lever) as the through-line. It is the capstone, not a
  parallel track.
