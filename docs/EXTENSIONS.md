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

## D. Real-data FFTs (r2c / c2r) -- a transverse extension, applies to every d

Real input has conjugate-symmetric spectrum (X[N-k] = conj(X[k])), so an N-point real FFT
needs ~HALF the flops and half the memory of a complex one -- the standard "modest but real"
win. Methods: pack two real signals into one complex FFT; pack one length-N real into a
length-N/2 complex FFT + O(N) split/recombine; specialized real split-radix (Sorensen 1987).
NOT a new dimension -- it is a transverse variant that multiplies onto d=1/2/4: an r2c 3D
FFT is one real axis pass then complex passes on the rest.

Why it belongs here and why it is NOT free:
- The theoretical ~2x is eaten by (a) the split/recombine post-processing (O(N), serial-ish,
  shuffle-heavy), (b) conjugate-symmetric HALF-COMPLEX output storage (CCS/Perm formats)
  which complicates SIMD lane packing and batching -- exactly the layouts our batch-lane
  engines depend on. So the realized win is typically 1.3-1.7x, not 2x, and can vanish at
  small N where the recombine dominates.
- Relevance to our chains: our graded workload is complex (FFT+|z| map), so real FFT does
  NOT apply to the current chain metric; it is for a DIFFERENT (real-input) workload.
- NOT motivated by the LQCD capstone. QCD lattice fields are COMPLEX in position space
  (gauge links Uμ(x) are SU(3) matrices; quark fields/propagators are complex spinor-color
  objects), so the transform that matters for the physics target is c2c, not r2c. Real FFT
  is justified only by genuinely real-input domains (audio, images, real signals), not by
  pairing with the d=4 lattice work. [corrected 2026-09-02]

Survey done (2026-09-02; full report docs/literature/staging_1d/06-real-vs-complex.md).
Payoff-ranked verdicts:
1. **Batched two-real-in-one = the easy ~2x, do first.** Pack B real transforms as B/2
   complex (z=x+iy, recombine X[k]=(Z[k]+conj Z[N-k])/2); the recombine's index reversal is
   along the transform axis so it vectorizes ACROSS the batch axis -- our batch-lane SoA
   engine runs UNCHANGED. Near-full 2x, negligible overhead.
2. **Multi-D r2c (3D) = the robust structural win.** One real axis pass (-> N/2+1 complex,
   halves the data) then complex pencils on the other axes over the HALVED volume: ~1.7-1.9x
   flops and a HARD 2x memory that never gets eaten. One new kernel into the existing
   pencil/8x8-transpose architecture. The memory halving is the headline in any memory-bound
   large-d REAL-INPUT regime -- but note this is a DIFFERENT workload from QCD (complex).
3. **B=1 single = modest 1.3-1.6x** (NR realft N/2-pack; serial reversed-index recombine
   dominates, -> parity at small N). Completeness, not excitement.
4. **Large-prime real = TRAP, skip.** Rader re-mixes the symmetry, Bluestein's chirp
   complexifies real input; libraries do complex-then-halve = memory only, ~1x flops. Given
   the d1 large-prime focus, do not spend real-FFT effort here.
5. **DHT route = dead since 1987.** Do not pursue.

Design rule (the one that decides whether the win survives): keep the COMPUTE path
full-complex and pack to half-complex/CCE only at the API boundary -- never thread the
non-pow2 N/2+1 storage through the SIMD kernels (that is exactly what erodes FFTW/MKL/cuFFT
on real). Accuracy is a non-blocker (r2c >= c2c; recombine twiddles fold into our dd table;
gate unchanged). Vendor opening: FFTW/MKL tune real LESS than complex (benchFFT r2c is
~1.4-1.7x c2c, not 2x), so our relative edge on real could match or exceed our ~3.5x c2c
edge -- to be measured, not assumed. Realistic speedup over our OWN c2c: ~2x batched,
~1.7-1.9x flops + 2x memory multi-D, ~1.3-1.6x B=1, ~1x primes.

## E. Distributed / multi-node FFT -- what the field looks like, and where we fit

Survey done 2026-09-02 (five veins + completeness critic + 3 gap-fills; synthesis in
docs/literature_dist/00-SURVEY.md, per-vein reports in docs/literature/staging_dist/).
The headline is structural and it cuts both ways.

**The good half.** Every distributed FFT that exists -- heFFTe, FFTW-MPI, P3DFFT, PFFT,
2DECOMP&FFT, fftMPI, SWFFT, FFTE, cuFFTMp, the Frontier 32768^3 record code, TensorFlow
DTensor, jaxDecomp, Grid -- is a **batched-1D local kernel plus transposes**. Our kernels
are the commodity slot of the whole field. For heFFTe the drop-in is fully specified from
source (commit 4d8d4597): a backend is ~100 lines of header-only template specialization,
the in-tree AVX-512 `stock` backend is the precedent, and with `use_reorder=true` the
executor is *guaranteed* contiguous data (stride 1, dist=size, howmany=osize(1)*osize(2)),
verified through plan_logic/reshape/setup rather than inferred from comments. Omitting the
2D/3D constructors is a compile error, never a silent wrong plan. One real cost: heFFTe
hands over INTERLEAVED std::complex, so our split-complex SoA layout needs an AoS<->SoA
shim per call -- and no distributed library anywhere documents a split-complex local layout,
so that shim is unavoidable at every integration boundary in the field.

**The sobering half (the Amdahl ceiling).** Measured, from figures recovered by rendering
the ICL tech-report pages: at 4 CPU nodes / 168 cores / 1024^3, local 1D FFT is **9.3%** of
runtime (MPI 84.3%); five libraries at 16 ranks / 256^3 all sit at **11-13%** FFT; on GPUs
it is 3.6-5% (MPI 92-95%) already at 8-24 devices. **A 3.5x local-kernel-only drop-in caps
at 1.07-1.09x end-to-end at >=4 CPU nodes and at essentially nothing multi-node on GPUs.**
Never pitch the kernel advantage as if it survives a large multi-node transform.

Three ways the value does survive, and they are all design instructions rather than
integrations:
1. **Absorb the transpose pack/unpack, not just the butterflies.** Under collective
   communication the local fraction is FFT+pack+unpack = 62% (46% for heFFTe in the
   five-library chart), projecting a ~1.5-1.8x cap instead of 1.07x. This feeds straight
   back into the 3D and 1D kernel design whether or not we ever ship anything distributed.
   It is the same observation as the vendors' measured strided-input weakness (cuFFT spikes;
   GESTS measured up to ~7x strided-vs-contiguous at L=32768), which our batch-lane SoA
   layout dissolves by construction.
2. **Batched distributed transforms are the field's admitted gap.** heFFTe's own data:
   batched 64^3 gives >2x per transform from comm/comp overlap; the SC'25 Kokkos paper names
   the absence of batching in distributed libraries outright; cuFFTMp forbids plan_many.
   And nobody has published a communication analysis of BATCHED distributed FFT -- the
   message-aggregation argument (batching multiplies per-pair message size, deferring the
   latency wall) is genuinely open. Caveat: overlap hides min(comm,comp), so a faster local
   kernel REDUCES hideable compute -- batching is what restores it.
3. **Spend surplus flops on communication-avoiding, not on raw speed.** FFTU
   (Koopman-Bisseling, SIAM 2023) does a multidimensional FFT with ONE all-to-all, valid to
   p <= sqrt(N), starting and finishing in the SAME distribution -- so forward -> elementwise
   map -> inverse needs no reordering at all. That is our fused chain regime, and the price is
   12N/p extra twiddle flops fused into packing. SOI FFT (SC'13) does the same trade for
   distributed 1D: one all-to-all instead of three at mu=5/4 oversampling. Cheap flops are
   exactly our comparative advantage.

**The LQCD lane is verified empty.** No library anywhere does 4D (an L^3xT transform is always
composed by the caller). QUDA's FFT gauge fixing is single-GPU only -- multi-GPU has been open
issue #255 since 2015. Grid's 4D FFT is a per-dimension Cshift barrel-shift gather plus a serial
FFTW/cuFFT call (source read), i.e. four full gather/scatters instead of pencil all-to-alls, and
its performance has never been characterized publicly. Chroma does momentum projection as an
O(V*n_momenta) phase sum, not an FFT -- which explains why LQCD has not been an FFT market: the
FFT case is where ALL momenta are needed (gauge fixing, momentum-space propagators, Fourier-
accelerated solvers). Independent corroboration from a different field: GROMACS reports heFFTe
"lacks the scalability for small 3D FFTs", so PME cannot scale past ~8 GCDs on AMD, and PME grids
are the same size class as an LQCD local volume. **Small distributed FFTs are where the
incumbents actually fail.**

Integration targets, ranked: heFFTe (primary -- pluggable by design, only library on AMD and
Intel GPUs, slow upstream churn, and its own benchmarks/speed3d_c2c is a ready measurement rig);
fftMPI (default local kernel is KISS FFT, compile-time pluggable, LAMMPS install base);
2DECOMP&FFT v2 (alive, pluggable, cuFFT+NCCL GPU path); Grid (the domain one). Not targets:
PFFT/P3DFFT/FFTE/SWFFT (dormant), AccFFT (failed correctness on Summit in ICL testing), cuFFTMp
(closed). Watch FFTX/SpectralPACK -- SPIRAL-generated fused batched kernels are our thesis by
another route, with a thin distributed layer and undocumented CPU fp64 performance.

Two corrections the survey's own gap-fill passes forced, recorded so they are not repeated:
the widely-quoted "communication ~50% at 512 cores" figure is **verified absent** from both ICL
reports (use 84.26% at 168 cores instead); and "cuFFTMp does not scale" is **wrong** -- the
Kokkos authors retracted it as a CPU-binding misconfiguration (cuFFTMp needs >=2 cores per
process), and correctly configured it beats their native MPI_Alltoall path by ~1.5x at 64 A100s
on 8-GPU-per-node hardware. cuFFTMp is a real incumbent.

The one number the literature does not contain, and the cheapest high-value experiment on this
list: **the comm-vs-compute fraction at 1-2 CPU nodes.** No published breakdown exists below 4
nodes; heFFTe is already built in ext/install; one axxxl reservation with a 2-node timer split
settles what our entire distributed pitch depends on.

## Dependency summary
- d=2: standalone, cleanest next win.
- Multicore-NUMA: standalone, extends the mt campaign.
- Multi-GPU: standalone, finishes the designed GPU phase.
- **d=4 (the physics target): depends on ALL THREE** — asymmetric-lattice kernels +
  multicore host decomposition + multi-GPU device compute, with traffic optimization
  (the unspent two-axes-fusion lever) as the through-line. It is the capstone, not a
  parallel track.
- Distributed (E): the natural *host* for a 4D transform, since no library offers one and
  Grid's is naive — but note the ordering. The distributed layer is where our kernel edge is
  WORTH LEAST (Amdahl-capped by the transpose) and where the design lessons are worth most
  (absorb pack/unpack, batch to aggregate messages, buy fewer all-to-alls with redundant
  flops). Take the lessons into the kernel work now; take the integration only after the
  1–2-node fraction is measured.
