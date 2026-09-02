# Distributed / large-scale FFT — heFFTe and the exascale landscape (survey, 2026-09-02)

Five-vein parallel survey plus a completeness critic and three gap-fill passes, run because
"we need to be aware of large distributed systems FFTs too". Full per-vein reports with
citations and per-claim verification tags in `../literature/staging_dist/`
(01 heFFTe internals, 02 competing libraries, 03 communication algorithms and theory,
04 GPU/exascale practice, 05 post-2020 developments, 06 critique + gap-fills). This file is
the synthesis: what the veins agreed on, what they corrected, and what it means for us.

Two of the gap-fill passes **corrected claims the survey itself had made** — those corrections
are carried here, not buried (see "Two things we would have got wrong").

---

## 1. The one structural fact everything rests on

**Every distributed FFT in existence is a batched-1D local kernel plus transposes.** This is
not an inference; it was read out of source or primary text for heFFTe, FFTW-MPI, P3DFFT,
PFFT, 2DECOMP&FFT, fftMPI, SWFFT, FFTE, cuFFTMp, GESTS (the Frontier 32768³ record code),
TensorFlow DTensor, jaxDecomp, DaggerFFT and Grid.

For heFFTe specifically (source read at commit `4d8d4597`): a 3D transform is
`std::array<unique_ptr<reshape3d_base>,4> shaper` alternating with
`std::array<executor_base*,3> executor`, and each executor is literally
`fftw_plan_many_dft(1, &size, howmany, ...)` / `DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS)` /
`cufftMakePlanMany(...)` / `rocfft_plan_create(...)` on one pencil box.

So the thing we have spent this project building **is the commodity slot of the entire field.**
That is the good news. The rest of this document is the arithmetic of how much that is worth,
and it is less than the slot's centrality suggests.

## 2. The heFFTe drop-in: fully specified, and cheaper than expected

The load-bearing question was whether our batched-1D fp64 kernel could be registered as a
heFFTe local-kernel backend and what interface it must satisfy. **Verified from plan/reshape
source, not from documentation** (there is none — no dynamic plugin registry exists; a backend
is a set of compile-time template specializations):

1. **What to write.** Tag struct in `namespace heffte::backend`; `is_enabled<tag>`;
   `buffer_traits<tag>` (a CPU default is provided); `backend::name<tag>()`;
   `one_dim_backend<tag>{ executor; executor_r2c; }`; `default_plan_options<tag>{ use_reorder }`.
   The executor derives from `executor_base` and overrides
   `forward(std::complex<double>[], std::complex<double>*)` / `backward`, plus `box_size()`
   and `workspace_size()`. **The in-tree `stock` backend — heFFTe's own hand-written AVX/AVX-512
   FFT — is the exact precedent**, and its adapter is ~100 lines.
2. **The data contract, with `use_reorder=true`.** Traced through `next_pencils_shape`,
   `plan_pencil_reshapes`, `reorder_slabs`, `fft3d::setup` and `make_reshape3d`: every shape
   handed to an executor has `order[0] == the FFT dimension`, therefore
   **stride = 1, dist = size, blocks = 1, howmany = osize(1)*osize(2)** — one contiguous run of
   back-to-back length-L lines. The strided middle-dimension path exists but is reachable only
   with `use_reorder=false`. `use_reorder=true` is already the default for the fftw and mkl
   backends.
3. **The out-of-tree hazard, resolved.** `has_executor2d/3d` are hardcoded exclusion lists and
   default to *true* for an unknown tag; but `setup()` uses ordinary runtime `if`s, so all three
   `make_executor` overloads are ODR-used and every constructor signature is instantiated.
   Omitting the 2D/3D constructors is therefore a **compile error, never a silent slab plan**.
   Copy `stock`'s throwing placeholders (and optionally specialize `has_executor2d/3d` to false).
4. **Batched calls preserve the contract.** `forward(batch_size, ...)` lays the batch out as
   back-to-back copies of the local box and calls the executor once per element at
   `data + j*box_size()`; the geometry per call is unchanged. heFFTe does **not** fuse the batch
   into one bigger plan_many — but the batching we exploit is the `howmany` lines *inside* one
   call, which for a realistic LQCD pencil (32³×64 on 64 ranks → local box 32×8×16) is hundreds
   of lines: ample width for 8-wide fp64.

**The one real cost:** heFFTe hands the executor **interleaved** `std::complex<double>`,
in-place. Our split-complex SoA batch-lane layout needs an AoS↔SoA shim per call, sized into
`workspace_size()`. No distributed library anywhere documents a split-complex local layout —
this shim is unavoidable at *every* integration boundary in the field, and its cost must be
counted inside any speedup we claim.

**Mandatory in our adapter:** `use_reorder=true` is a hard requirement, not a preference (with
`use_reorder=false`, a slab-shaped decomposition can legitimately produce a null middle shaper
and call our throwing 2D placeholder). Also assert `box.order[0]==dimension` and throw — the
contiguity guarantee is real but no test in the tree asserts it, so pin the commit.

## 3. The Amdahl ceiling, now quantified on CPUs as well as GPUs

This is the finding that reframes the whole ambition. Recovered by rendering the ICL tech-report
pages to images (their PDFs embed digits in fonts without ToUnicode maps, which is why these
numbers were previously unquotable):

| Configuration | local 1D FFT | pack/unpack | MPI |
|---|---|---|---|
| 1024³, 4 Summit CPU nodes, 168 Power9 cores, pipelined Isend/Irecv | **9.29%** | 5.58% | 84.26% |
| same, MPI_Alltoall (1.63× slower overall, 44.47 GFlop/s) | **5.65%** | 56.4% | 37.9% |
| 256³, 16 ranks, 4 nodes — five libraries (heFFTe, fftMPI, P3DFFT, FFTE, ADVMPI) | **11–13%** | 29–46% | 41–77% |
| 1024³, 16 V100 GPUs, pipelined Isend/Irecv (540 GFlop/s) | 5.2% | <5% | 92.57% |
| same, MPI_Alltoall (313 GFlop/s) | ~4% | <5% | 95.13% |
| 512³, 24 V100 (4 Summit nodes) | — | <10% | **>90%** |
| 10K×10K 2D, 8 V100, TensorFlow DTensor | **3.6%** | — | rest |
| 1024³ fp64, 64 A100 (8/node), Kokkos native path | 0.044 s of 1.21 s (**~4%**) | 0.27 s | 1.07 s |

**Consequence, stated plainly: a 3.5× local-kernel-only drop-in caps at 1.07–1.09× end-to-end
at ≥4 CPU nodes, and at essentially nothing multi-node on GPUs.** Our kernel advantage does not
survive contact with a large multi-node transform, and we should never pitch it as if it does.

### Our own measurement (2026-09-02) — the small-node end the literature never decomposed

`bench/dist/` now measures this directly, via heFFTe built with `Heffte_ENABLE_TRACING` so the
split comes from `MPI_Wtime` events rather than inference. First run: 2× Xeon Gold 5218
(Cascade Lake, 32 physical cores/node), OpenMPI 4.1.5, FFTW backend, fp64 c2c, MPI_Alltoallv.
The bucketing is verified by closure against heFFTe's own reported time (traced total =
2·nruns·per-transform, worst deviation 2.4%), so the percentages are complete and each event
counted once:

| grid | ranks | nodes | fft% | pack% | mpi% | unpack% | cap@3.5× | cap if pack absorbed |
|---|---|---|---|---|---|---|---|
| 128³ | 1 | 1 | **39.5** | – | 0 | – | 1.39 | 3.50 |
| 128³ | 8 | 1 | **22.1** | 24.9 | 20.7 | 30.9 | 1.19 | 2.30 |
| 128³ | 32 | 1 | **19.8** | 20.0 | 31.2 | 28.1 | 1.17 | 1.96 |
| 256³ | 1 | 1 | **34.4** | – | 0 | – | 1.33 | 3.50 |
| 256³ | 8 | 1 | **19.4** | 24.2 | 20.4 | 34.6 | 1.16 | 2.32 |
| 256³ | 32 | 1 | **15.7** | 30.1 | 22.3 | 30.5 | 1.13 | 2.25 |

Four things this settles:

1. **Single-node local-FFT share is 16–22% at 8–32 ranks** — higher than the 11–13% ICL
   measured at four nodes, as expected, but a kernel-only drop-in still caps at **1.13–1.19×**.
   The small-node regime is friendlier than the published numbers, and not nearly friendly
   enough to carry the claim on its own.
2. **Even at ONE RANK, with no MPI whatsoever, the local FFT is only 34–40% of the transform.**
   The other 60–66% is heFFTe's own local reshape copies. This is the most useful number here:
   it means a kernel-only drop-in is capped at ~1.39× even in the completely serial case, so
   the ceiling is *not* mainly about the network at all — it is about data movement.
3. **Absorbing the transpose pack/unpack raises the cap to 1.96–2.32×** — now measured on our
   hardware rather than projected from the ICL charts. This is the design instruction of §3
   confirmed: the win is in fusing the data movement into the kernel, not in faster butterflies.
4. **Shared-memory MPI is already 20–31% at one node.** The transpose is expensive before any
   network is involved.

**The two-node rows from that run are unusable and are deliberately not quoted.** The same 32
ranks and same grid took 0.0060 s inside one node and 0.645 s split across two — 107×, with MPI
at 98–99% of the trace. That is a transport fault on the `devel`/`prod` nodes, not a fabric
latency penalty, and it is exactly the trap the survey warns about: read off the FFT alone it
looks like a textbook confirmation of "communication dominates beyond one node". It was caught
only because the ladder holds rank count fixed while varying node count. `bench/dist` now runs
a ping-pong and all-to-all fabric probe before every multi-node sweep so this can never be
reported as an FFT property. **The Ice Lake two-node number remains open.**

Three ways the value survives, in descending order of confidence:

- **Absorb the pack/unpack, not just the butterflies.** Under collective communication on CPU
  the *local* fraction is FFT + pack + unpack = **62.1%** (or 46% for heFFTe in the five-library
  chart). A fused SoA kernel that eats the transpose-pack projects to a **~1.5–1.8× end-to-end
  cap** rather than 1.07×. This is a design instruction, and it is a projection, not a
  measurement.
- **Stay on CPUs, and stay small.** The Kokkos group's own Ice Lake row at 16 ranks —
  All2All 8.93, pack 7.63, unpack 4.99, FFT 10.39, transpose 11.20 — has **compute plus local
  data movement well above 75%** on our exact hardware class. The GPU papers' pessimism is a
  GPU fact, not a universal one.
- **1–2 nodes is unmeasured territory.** No published breakdown exists below 4 CPU nodes in
  either ICL report (the closest is total-time-only strong scaling at 4 ranks/node, showing
  near-ideal 1→2 node scaling, which *suggests* comm is not yet saturating but cannot be quoted
  as a percentage). heFFTe is already built in `ext/install`; **one axxxl reservation with a
  2-node timer split would close this ourselves**, and it is the number our entire distributed
  pitch depends on.

## 4. Where the surplus flops should actually go: communication-avoiding

The literature's own answer to "the transpose owns the runtime" is not faster kernels — it is
**fewer transposes, bought with redundant local computation.** We are the party that can most
afford to buy.

- **FFTU (Koopman–Bisseling, SIAM J. Sci. Comput. 45(6) 2023).** A multidimensional cyclic-to-
  cyclic FFT with **exactly one all-to-all**, valid to p ≤ √N, that **starts and finishes in the
  same distribution**. Cost: 12N/p extra twiddle flops, fused into the packing. The
  same-distribution property means forward → elementwise map → inverse runs **with no data
  reordering at all** — which is our fused FFT+map chain, described by someone else, at 4096
  cores. Measured on Snellius at 1024³: FFTU 149× sequential FFTW vs heFFTe 119× and PFFT 98.5×.
  This is the communication-minimal template our chain work should be measured against.
- **SOI FFT (Park et al., SC'13).** Distributed **1D** FFT with one all-to-all instead of three,
  at μ=5/4 oversampling — i.e. ~3× less communication for ~25% more work. 6.7 TFlop/s on 512
  nodes. Directly about the transform our current campaign is building, and the μ tradeoff moves
  further in favour of redundant computation the cheaper our flops are.
- **Pruned / zero-padded transforms.** Free communication reduction when part of the input is
  known-zero (LQCD correlators are convolutions). Caveat worth heeding: the 2026 distributed
  NUFFT work found pruning **wins on Slingshot and loses 2.4× on InfiniBand** because of
  concurrent sub-communicator traffic — our fabric is InfiniBand, so measure, do not assume.

Theory ceiling for context (Czechowski et al., ICS'12): on a torus the placement-optimal bound
is T_net = Ω(n³/(P^{5/6}·β)) and the realistic global-all-to-all estimate is bisection-limited
2n³/(P^{2/3}·β), with flop time trailing communication by ~three orders of magnitude at
exascale. Slab beats pencil below ~64 nodes (modelled *and* measured). Per-pair message size
falls as ~1/P², so strong scaling always walks out of the bandwidth regime into the
Bruck/latency regime — the measured walls sit at ~512 nodes for 1024³ on both the CPU and GPU
Summit configurations.

## 5. Batching: the compounding lever, and the genuinely open one

- heFFTe's own IPDPS-2022 data: batched 64³ 3D transforms give **>2× per transform** versus
  unbatched, "from the overlap of communication and computation… the more transforms per MPI
  unit generates more overlap"; the benefit is "considerably reduced" at 512³ where
  communication dominates regardless.
- The SC'25 Kokkos paper names the field's gap outright: "a further limitation of existing
  distributed FFT libraries is the absence of batching capabilities". cuFFTMp explicitly
  **forbids** `plan_many`; P3DFFT, PFFT, SWFFT, FFTE and 2DECOMP&FFT document none.
- **Nobody has published a communication analysis of batched distributed FFT.** The
  message-aggregation argument — batching multiplies every per-pair message by B, pushing the
  transpose back out of the latency regime — appears nowhere as a measured or proven result.
  That is an open, publishable lane that our batch-lane design walks into naturally.
- The honest tension: overlap hides min(comm, comp) (Song & Hollingsworth, PPoPP'14, measured
  1.10–1.76× over FFTW-MPI). **A faster local kernel reduces the compute available to hide
  communication behind.** Batching independent transforms is what restores hideable work — so
  kernel speed and overlap are complements only through batching, not in general.
- Precedent for both halves of our design at extreme scale: **nb3dfft** (RWTH/psOpen, LNCS 2017)
  overlaps many concurrent 3D FFTs *and* fuses the dealiasing filter into the transform to skip
  a data pass, at 8192³ on full JUQUEEN. Not publicly available, but it is the citation that our
  fused-chain-plus-batching design is not novel-for-novelty's-sake.

## 6. Two things we would have got wrong

The gap-fill passes overturned two claims the main veins had asserted. Both are recorded here so
they are not repeated:

1. **"Communication is ~50% at 512 cores for heFFTe CPU"** — an often-repeated figure. It is
   **verified absent** from both ICL reports (every page scanned, text and rendered). Its likely
   home is an unfetched PaCT-2021 paper. Use the verified numbers instead: 84.26% at 168 cores,
   41–77% MPI at 16 ranks.
2. **"cuFFTMp does not scale"** — the only apparently independent evidence, and it is wrong. The
   quote ("performance remained unchanged as the number of GPUs increased") belongs to the SC'25
   Kokkos workshop paper, not to DaggerFFT as first attributed; and the **same authors retract it
   in effect** in May-2026 slides: the flat scaling was a CPU-binding misconfiguration
   (cuFFTMp needs ≥2 cores per process). Correctly configured, **cuFFTMp beats their competent
   native GPU-aware MPI_Alltoall path by ~1.5× at 64 A100s** at 1024³ fp64 on 8-GPU-per-node
   hardware — our exact topology. cuFFTMp is a real incumbent. (Also from those slides: pencil
   8×8 beats slabs on an A100 node thanks to NVLink.)

## 7. Numbers to aim at, if we ever go GPU-distributed

- **The published 8×A100 single-node baseline** (Verma et al., DGX A100-80GB, cuFFTXt, fp64,
  forward+inverse pair): 512³ **5.1 ms**, 1024³ **19 ms**, 2048³ **132 ms**. Our a100l/a100r
  parts are 40 GB SXM4 — expect somewhat slower.
- **fp64 FFT on A100 is bandwidth-bound, not fp64-rate-bound**: measured DP time = 2.0× SP at
  fixed grid/GPU count.
- **The one-node→two-node cliff is real**: "best performance is realized either within a single
  DGX box for a relatively small grid, or with many more GPUs for a large grid" — the fat single
  node is the sweet spot where local kernel quality still dominates, and where an L³×T lattice
  plausibly fits entirely in HBM.
- **2080 Ti fp64 FFT has never been published by anyone.** Roofline projection (ours, unverified):
  arithmetic intensity ~0.5–0.6 fp64 flop/byte against 616 GB/s demands 290–385 GFlop/s versus
  ~420 GFlop/s fp64 peak — i.e. the 1/32-rate part sits at the bandwidth/compute *crossover*,
  so expect ~2–4× worse than a bandwidth-only estimate, **not** 32×. A ten-minute measurement on
  a `prod` node would settle it, and the transpose does not care about fp64 rate at all, so those
  22 nodes remain fully usable for communication-pattern work.
- **MPI stack choice is a first-order term**, the same magnitude as algorithm choice:
  MVAPICH-GDR beats Spectrum MPI to 64 nodes and loses above; Intel MPI→OpenMPI moved FFTU by
  22%; disabling GPU-aware MPI costs ~30% of communication; NCCL-vs-MPI flips with message size.
  Any distributed number we publish must pin the MPI configuration or it means nothing.

## 8. The lattice-QCD lane is empty — verified, not assumed

- **No library anywhere does 4D.** Every one is 2D/3D (+batch); an L³×T transform is always
  composed by the caller. Searches for four-dimensional distributed FFT return nothing.
- **QUDA**: `computeGaugeFixingFFTQuda` (Fourier-accelerated Landau gauge fixing) is
  **single-GPU only**; multi-GPU has been open issue #255 since 2015. The one place QUDA wants a
  full 4D FFT, it cannot scale past one device.
- **Grid**: `Grid/algorithms/FFT.h` read directly — the lattice FFT loops over dimensions and,
  for each, gathers the full extent locally via repeated `Cshift` barrel-shifts across the
  processor grid, runs a serial FFTW/cuFFT plan, and scatters back. Four dimensions, four full
  gather/scatters, no pencil all-to-all. **Ripe for replacement**, and no performance
  characterisation of it exists publicly.
- **Chroma**: `SftMom` does momentum projection as an explicit phase-factor sum
  (`sumMulti(cf*phases, getSet())`), O(V·n_momenta) per timeslice. This explains why LQCD has
  never been an FFT market: production correlator work wants a handful of momenta, and phase sums
  win there. **The FFT case in LQCD is where all momenta are needed** — gauge fixing,
  momentum-space propagators, quasi-PDF-style constructions, Fourier-accelerated solvers.
- **Independent corroboration from a different field**: GROMACS reports that heFFTe "lacks the
  scalability for small 3D FFTs", so PME "can hardly be scaled past 8 GCDs" on AMD hardware, and
  no portable cuFFTMp equivalent exists. PME grids are ~10² per side — the same size class as an
  LQCD local volume. **Small distributed FFTs are where the incumbents actually fail.**

## 9. Secondary integration targets, ranked

1. **heFFTe** — primary. Pluggable by design, the `stock` precedent, batch API since v2.3,
   the only library on AMD *and* Intel GPUs, and slow upstream churn (v2.4.1 Oct 2023, quiet
   since ECP wound down) means we would not be chasing a moving target. Its in-tree
   `benchmarks/speed3d_c2c` is a ready-made measurement rig, and changing backend is one
   template parameter — the cleanest possible A/B demonstration of the 1D campaign's results.
2. **fftMPI** (Plimpton, Sandia) — highest leverage per unit effort after heFFTe. Its *default*
   local kernel is KISS FFT, deliberately simple; FFTW and MKL are compile-time alternatives, so
   the plugin slot already exists. It is LAMMPS/SPARTA infrastructure, actively maintained, and
   the delta would be immediately visible in a standard LAMMPS benchmark (heFFTe already
   demonstrated a 40% KSPACE reduction against it).
3. **2DECOMP&FFT v2** — alive (JOSS 2023), modern Fortran, explicitly pluggable FFT backends
   (FFTW/MKL/cuFFT/generic), GPU via cuFFT+NCCL. Fortran-first CFD audience.
4. **Grid** — not a library integration but the domain one: replacing the Cshift-gather FFT with
   pencil transposes plus our batched-1D kernel is a credible upgrade for 4D lattice transforms.

Not targets: PFFT and P3DFFT (dormant, FFTW hardwired), FFTE (dormant, own Fortran kernels, no
general prime support at all), SWFFT (frozen), AccFFT (bit-rotten — failed correctness on Summit
in ICL's testing), cuFFTMp (closed, cuFFT-only).

The intellectual competitor to watch is **FFTX/SpectralPACK** (SPIRAL code generation): it
generates fused, batched kernels — including a batched-1D C2C library, `fftx_dftbat` — instead of
hand-writing them, which is our thesis by another route. Its distributed layer is thin and its
CPU AVX-512 fp64 performance is undocumented, so it is not an established baseline we must beat;
running it on our shapes is a cheap experiment worth doing.

## 10. What the field is *not* doing (so we needn't)

- **Task-based runtimes are a measured negative.** HPX work states outright that "the FFT
  application does not profit from asynchronous task execution" — enforced synchronisation gives
  better cache behaviour. The 5× they did find came from swapping the *transport* (LCI
  parcelport). Our hand-scheduled synchronous passes are vindicated; the exploitable slack is in
  the transpose transport, orthogonal to kernel work.
- **Mixed precision is applied only to the wire, never to the butterflies.** ZFP-compressed
  transposes (heFFTe authors' own 2022 research branch, never merged into a release) and MFFT's
  shared-exponent format keep the transform itself in fp32/fp64. Lossy transposes are research
  practice, not production practice. Our fp64 positioning is uncontested.
- **Tensor-core FFT stays single-GPU and low-precision** (tcFFT fp16, FlashFFTConv for ML
  sequence convolutions). No distributed library uses a matrix engine as its local kernel, and
  for complex fp64 lattice fields they are not an alternative at all.
- **Nobody has beaten strided-input weakness.** Vendors' batched-1D calls spike on strided input
  (cuFFT; and GESTS measured up to ~7× between strided and contiguous at length 32768), so
  libraries repack to avoid it — paying pack cost to dodge stride cost. Our batch-lane SoA layout
  dissolves that dilemma by construction. This is a real, citable differentiator, and it is the
  same observation as §3's "absorb the pack".

---

## Actionable list, ranked by leverage

1. ~~Measure the 1–2 node CPU comm fraction ourselves.~~ **Single-node done** (§3, `bench/dist/`):
   16–22% local FFT at 8–32 ranks, and only 34–40% even at one rank. **Two-node still open** —
   the `devel`/`prod` inter-node transport is broken (107× penalty), so it needs `axxxl`, which
   means one window with both Ice Lake nodes in a single allocation.
2. **Design the kernel to absorb transpose pack/unpack**, not just the butterflies — now
   promoted to the top item, because the measurement moved it from "plausible" to "the whole
   ballgame": kernel-only caps at 1.13–1.19× on one node, kernel-plus-data-movement at
   1.96–2.32×, and the serial one-rank row shows 60–66% of the transform is data movement even
   with zero MPI. Feeds straight back into the 3D/1D kernel design, independent of whether we
   ever ship anything distributed.
3. **Write the heFFTe backend adapter** (~100 lines: 1D ctor, throwing 2D/3D placeholders,
   optional `has_executor2d/3d=false`, AoS↔SoA shim, `use_reorder=true` assert). Cheap, fully
   specified above, and gives an A/B against fftw/mkl/cufft by changing one template parameter
   using heFFTe's own benchmark harness.
4. **Take the batched-distributed lane seriously** — it is the one place the field admits a gap
   (no library batches; nobody has analysed batched communication) and it is where our design is
   already pointed.
5. **Two ten-minute measurements**: cuFFT fp64 batched-1D on a 2080 Ti vs an A100 (to replace our
   roofline projection with a number nobody has published), and FFTX's `fftx_dftbat` on our
   shapes (to check the code-generation competitor).
6. **Read FFTU and SOI properly before any distributed algorithm work.** Communication-avoiding
   at the cost of redundant flops is the strategy our cost structure most favours, and FFTU's
   same-distribution single-all-to-all property is our fused chain regime exactly.
