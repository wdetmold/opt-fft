# Post-2020 developments: releases, mixed precision, compression, task runtimes, ML frameworks, LQCD codes

**Scope as the agent recorded it:** Post-2020 distributed FFT: what changed since heFFTe's paper, new libraries and rewrites, ML-framework distributed FFT, new hardware, mixed precision and compression of the transpose, task-based runtimes, higher-dimensional transforms, and how lattice QCD codes actually do momentum-space transforms

## 1. heFFTe releases since the 2020 paper: batched FFTs landed in 2021, then development slowed

*Citation:* heFFTe releases, icl-utk-edu/heffte on GitHub, https://github.com/icl-utk-edu/heffte/releases ; heFFTe project page https://icl.utk.edu/fft/

**Claim.** GitHub releases page (fetched): v2.3 (6 Mar 2021) added "batch-FFTs (multiple signals with one command)" and sub-communicator support to reduce MPI communication; v2.4.0 (27 Oct 2022) added real-to-real sin/cos type I/II transforms and DPC++/SYCL sync improvements; v2.4.1 (25 Oct 2023) improved real-to-real support across backends plus bug fixes. Nothing newer appears on the releases page — after ECP wound down the cadence is roughly one maintenance release per year, then quiet (Fedora's Dec 2024 heffte-2.4.1-2 is a repackage, not a new version).

**Numbers.** No new performance numbers in the release notes; the per-rank local kernel remains a call into FFTW/MKL/cuFFT/rocFFT/oneMKL in all these versions.

**Relevance to us.** Confirms the drop-in thesis is still live in 2026: heFFTe's local kernel is still a vendor batched-1D call and the batch API (since 2.3) is exactly the shape our batch-lane kernels present. Slow upstream churn means an interposed faster batched-1D fp64 CPU kernel would not be chasing a moving target.

**Verification status.** VERIFIED (releases page fetched). heFFTe source is public; the batched API exists in released code.

## 2. cuFFTMp: NVIDIA's own multi-node FFT, NVSHMEM-based — the vendor answer on our A100 partitions

*Citation:* NVIDIA, "Multinode Multi-GPU: Using NVIDIA cuFFTMp FFTs at Scale", NVIDIA Technical Blog (2022), https://developer.nvidia.com/blog/multinode-multi-gpu-using-nvidia-cufftmp-ffts-at-scale/ ; docs https://docs.nvidia.com/cuda/cufftmp/

**Claim.** cuFFTMp (early access Jan 2022, now shipped in the HPC SDK) does distributed 2D/3D FFTs with slab and pencil decompositions and replaces MPI_Alltoallv with kernel-initiated NVSHMEM communication, explicitly because "MPI_Alltoallv is the main bottleneck for distributed FFTs".

**Numbers.** NVIDIA's own blog numbers: >1.8 PFlop/s (~70% of machine bandwidth-limited peak) transforming >4 trillion complex points on 4096 A100-80GB (Selene); strong-scales a 4096^3 single-precision transform from 78 ms on 8 GPUs to 4 ms on 2048 GPUs. Vendor-measured, not independently reproduced by us.

**Relevance to us.** On our 8xA100 nodes this is the incumbent for any GPU distributed-FFT ambition, and its design message matches everything else in this vein: at scale the game is the all-to-all, not the local kernel. Our CPU batched-1D advantage translates to the distributed regime mainly at small node counts or fat single nodes where compute is still a visible fraction.

**Verification status.** VERIFIED that the library and claims exist (NVIDIA blog/docs are the primary source); numbers are the vendor's measurements. Binary-only distribution — no source to inspect for the local kernel.

## 3. Grid (lattice QCD) distributed FFT: per-dimension barrel-shift gather + serial FFTW/cuFFT — verified in source, and it is naive

*Citation:* Grid, Grid/algorithms/FFT.h, paboyle/Grid develop branch, https://github.com/paboyle/Grid/blob/develop/Grid/algorithms/FFT.h ; background: Boyle et al., "Grid: A next generation data parallel C++ QCD library", arXiv:1512.03487

**Claim.** Fetched the source: Grid's lattice-wide FFT loops over dimensions; for each dimension it gathers the full extent locally via repeated Cshift barrel-shifts across the processor grid ("temp = Cshift(result, dim, L)"), runs a local FFTW plan (fftw_execute_dft; cuFFT/hipFFT variants under ifdef), then scatters back. Both ComplexD and ComplexF are supported; a PlannedFFT class caches per-dimension plans.

**Numbers.** No published performance numbers for Grid's FFT path found; the structure implies each of the 4 dimensions pays a full gather/scatter of the lattice rather than a pencil all-to-all.

**Relevance to us.** This is the closest thing to 'distributed FFT in production LQCD', and it is a per-dimension gather + vendor-1D-FFT design — i.e., four passes of exactly the batched-1D problem we are optimizing, wrapped in suboptimal communication. A hand-written batched-1D kernel plus pencil transposes is a credible upgrade path for 4D L^3xT transforms in Grid-style codes.

**Verification status.** VERIFIED (source file read via fetch). Performance characterization: not documented.

## 4. QUDA: FFT-based (Fourier-accelerated) gauge fixing is single-GPU only; multi-GPU falls back to overrelaxation

*Citation:* lattice/quda issue #255 "Missing MPI support if we want to use FFT gauge fixing w/ multi GPUs", https://github.com/lattice/quda/issues/255 ; QUDA wiki/docs https://github.com/lattice/quda/wiki ; underlying algorithm: Cardoso & Bicudo, "GPU implementation of a Landau gauge fixing algorithm", arXiv:1210.8018

**Claim.** QUDA's computeGaugeFixingFFTQuda (steepest-descent with Fourier acceleration, cuFFT) "does not support multi-GPUs"; only the relaxation path computeGaugeFixingOVRQuda runs multi-GPU. The GPU literature it builds on notes cuFFT has no 4D FFT, so the 4D transform is composed from 1D/2D pieces (4x1D, 2x2D, or 3D+1D).

**Numbers.** Cardoso-era numbers only (one Tesla C2070 ~ 256 CPU cores on 32^4 Landau gauge fixing); no modern multi-GPU FFT gauge-fixing numbers exist because the capability does not exist in QUDA.

**Relevance to us.** A concrete, named gap in the LQCD GPU stack: the one place QUDA wants a full 4D FFT, it cannot scale past one GPU. Also confirms no vendor library offers 4D FFTs — a 4D transform is always composed of batched lower-dimensional passes, which is our kernel's regime.

**Verification status.** VERIFIED (issue and docs are primary sources); Cardoso numbers ABSTRACT-ONLY.

## 5. Chroma momentum projection (SftMom) is an explicit phase-factor sum, not an FFT

*Citation:* Chroma, lib/util/ft/sftmom.h, JeffersonLab/chroma, https://github.com/JeffersonLab/chroma/blob/master/lib/util/ft/sftmom.h

**Claim.** Fetched the header: SftMom precomputes phase factors and does momentum projection as "sumMulti(cf*phases, getSet())" — a slow Fourier transform over spatial sites for a user-selected set of momenta (mom2_max), with averaging over equivalent momenta. No FFT anywhere in the path.

**Numbers.** N/A — cost is O(V * n_momenta) per timeslice, which is why production analyses restrict to small mom2_max.

**Relevance to us.** Explains why LQCD production codes have not been an FFT market: correlator workflows want a handful of momenta, and O(V*n_mom) phase sums beat setting up a distributed FFT. The FFT case in LQCD is where ALL momenta are needed (gauge fixing, momentum-space propagators/quasi-PDF-style constructions, Fourier-accelerated solvers) — that is where a fast 4D=batched-1D capability would actually land.

**Verification status.** VERIFIED (source header read via fetch).

## 6. Compression of the transpose: heFFTe's own mixed-precision/approximate-FFT study (ZFP et al.)

*Citation:* S. Cayrols, J. Li, G. Bosilca, S. Tomov, A. Ayala, J. Dongarra, "Mixed precision and approximate 3D FFTs: Speed for accuracy trade-off with GPU-aware MPI and run-time data compression", ICL Tech Report ICL-UT-22-04, May 2022 (also ScalAH/SC workshop), https://icl.utk.edu/files/publications/2022/icl-utk-1555-2022.pdf

**Claim.** Reimplements the all-to-all at the core of heFFTe's 3D FFT with one-sided MPI and in-flight lossy compression (ZFP among the codecs); designs an approximate FFT with user-controlled accuracy. Key stated result: 3D FFT speeds up roughly proportionally to the compression rate, and when communication volume is compressed to the size of reduced-precision data the approximate FFT is as fast as reduced-precision execution while being about one order of magnitude more accurate.

**Numbers.** Measured on GPU systems with GPU-direct MPI (Summit-class); I could not extract the exact tables (PDF text extraction unavailable on this cluster), so specific speedup figures are not quoted here.

**Relevance to us.** Direct answer to 'does anyone compress the transposed data': yes, the heFFTe authors themselves — but as a 2022 research branch, not merged as a default in any heFFTe release (release notes through 2.4.1 do not mention it). Lossy-compressed transposes are research practice, not accepted production practice; fp64 end-to-end remains the norm, which suits our fp64-kernel positioning.

**Verification status.** VERIFIED existence, authors, method, and headline claim (ICL publication page + report); detailed numbers ABSTRACT-ONLY. Not in released heFFTe as far as release notes show — plainly: integration status undocumented.

## 7. MFFT: 'high-precision compute, low-precision communication' via shared-exponent compression at 4096-GPU scale

*Citation:* Zhao et al., "MFFT: A GPU Accelerated Highly Efficient Mixed-Precision Large-Scale FFT Framework", ACM TACO 2023, DOI 10.1145/3605148

**Claim.** Distributed FFT framework that keeps the local transform in high precision but compresses the transpose traffic with a shared-exponent floating-point format plus two-phase normalization to control round-off.

**Numbers.** On a 4096-GPU (AMD-based, Sunway/ORISE-class Chinese system) run: shared-exponent MFFT 1.23x faster on average than its own fp64 mode; fp64 MFFT reported 3.53x over CPU 2DECOMP&FFT and 9.48x over heFFTe's ROCm backend on that machine.

**Relevance to us.** The clearest post-2022 datapoint that mixed precision in DISTRIBUTED FFT is applied only to the wire format, never to the butterflies — everyone keeps the transform itself in high precision and shrinks the transpose. Our fp64 kernels are compatible with that pattern; also note the 9.48x-over-heFFTe figure says vendor-backend heFFTe can be far from optimal on non-NVIDIA GPUs.

**Verification status.** ABSTRACT-ONLY (ACM abstract and summaries; full text not fetched). Code availability not established.

## 8. TensorFlow gained a native distributed FFT (DTensor, 2023) — and measured local FFT at 3.6% of total time

*Citation:* Ruijiao Sun (Google DTensor team), "Distributed Fast Fourier Transform in TensorFlow", TensorFlow Blog, 24 Aug 2023, https://blog.tensorflow.org/2023/08/distributed-fast-fourier-transform-in-tensorflow.html

**Claim.** TF v2 exposes distributed FFT through the standard tf.signal API on DTensor-sharded tensors, using "the simplest distributed FFT algorithm" — shuffle (ncclAllToAll) + local FFT — and the post openly lists it as unoptimized.

**Numbers.** 8x V100, 10Kx10K 2D FFT: local FFT ops are 15 ms = 3.6% of total distributed-FFT time; everything else is data shuffling.

**Relevance to us.** Two lessons: (a) ML frameworks now ship distributed FFT but at a naive level, decomposing to local FFT + all-to-all like everyone else; (b) their own measurement — local compute 3.6% of wall time — quantifies the ceiling on what a faster local kernel buys in the communication-dominated distributed regime. Our 3.5x kernel advantage is worth the most on-node (batched/chained regimes), least in a multi-node transpose-bound transform.

**Verification status.** VERIFIED (blog fetched; first-party measurement at small scale).

## 9. jaxDecomp: differentiable distributed 3D FFT for JAX (pencil decomposition, cuDecomp/NCCL under the hood)

*Citation:* Balkenhol et al. (DifferentiableUniverseInitiative), "jaxDecomp: JAX Library for 3D Domain Decomposition and Parallel FFTs", JOSS (2025), DOI 10.21105/joss.08852; docs https://jaxdecomp.readthedocs.io ; https://github.com/DifferentiableUniverseInitiative/jaxDecomp

**Claim.** Provides multi-node pencil-decomposed 3D FFTs and halo exchanges inside jitted JAX code, differentiable end-to-end, with backends wrapping NVIDIA's cuDecomp transpose library / NCCL; driven by cosmology particle-mesh simulation (JaxPM).

**Numbers.** JOSS paper and docs claim superior asymptotic scaling vs prior JAX-reachable FFT paths; no independent absolute GFLOP/s figures extracted.

**Relevance to us.** This is where the ML-framework world got a real (non-naive) distributed FFT: pencils + autotuned transposes, local kernel still cuFFT via XLA. It confirms the universal architecture (batched 1D + transpose) but offers no CPU fp64 story — no competition for our niche, and another host whose local kernel is a swappable vendor call.

**Verification status.** VERIFIED existence and design (docs fetched via search results, JOSS entry); performance claims ABSTRACT-ONLY.

## 10. Task-based runtimes: HPX studies find FFT does NOT benefit from asynchronous tasking — the win is in the communication layer

*Citation:* A. Strack, D. Pflüger et al., "Experiences Porting Distributed Applications to Asynchronous Tasks: A Multidimensional FFT Case-study", arXiv:2405.00015 (2024); follow-up "A HPX Communication Benchmark: Distributed FFT using Collectives", arXiv:2504.03657 (2025, ISC workshop)

**Claim.** Added an HPX backend to FFTW and built HPX-native distributed FFTs. Verbatim finding: "the FFT application does not profit from asynchronous task execution" — enforced synchronization gives better cache behavior; but swapping HPX's MPI parcelport for the LCI parcelport accelerated communication up to 5x (up to 3x total vs an MPI+OpenMP FFTW reference on 16 nodes).

**Numbers.** Up to 5x communication speedup / 3x end-to-end vs FFTW MPI+X reference, 16-node cluster (CPU).

**Relevance to us.** Answers the Legion/StarPU/HPX question with a measured negative: FFT's dataflow is too regular for task runtimes to help compute, so hand-scheduled synchronous passes (our design) are vindicated; the exploitable slack is in the transpose transport (LCI-style), orthogonal to and composable with a faster local kernel.

**Verification status.** VERIFIED (arXiv abstracts fetched via search; findings quoted from the papers' own summaries). Full experimental tables not independently rechecked.

## 11. 2DECOMP&FFT v2: the classic CFD pencil framework got a modern-Fortran + GPU rewrite (2023)

*Citation:* Rolfo et al., "The 2DECOMP&FFT library: an update with new CPU/GPU capabilities", JOSS 8(91):5813, 2023, DOI 10.21105/joss.05813; https://github.com/2decomp-fft/2decomp-fft

**Claim.** Complete modernization of the widely used pencil-decomposition framework: modern Fortran, slab as special case of pencil, GPU offload (one MPI rank per GPU, cuFFT backend), pluggable FFT backends (FFTW, MKL, cuFFT, generic).

**Numbers.** JOSS paper is a software paper; no headline scaling numbers extracted.

**Relevance to us.** Second major distributed-FFT rewrite since 2021 (alongside heFFTe's maintenance releases) and the same drop-in structure: the 1D/serial FFT engine is an explicitly pluggable backend. A CPU AVX-512 batched-1D backend is architecturally trivial to slot into 2DECOMP&FFT for its FFTW path.

**Verification status.** VERIFIED (JOSS paper + repo exist; performant open code exists at the GitHub link).

## 12. FFTX 1.0: the SPIRAL code-generation successor to FFTW shipped, but distributed capability and adoption remain thin

*Citation:* FFTX docs https://spiral-software.github.io/fftx/ ; http://www.spiral.net/software/fftx.html ; Franchetti et al., "FFTX and SpectralPack: A First Look", IEEE HiPC-W 2018

**Claim.** FFTX (ECP project, LBNL/CMU/SpiralGen) released 1.0.0: build-time SPIRAL code generation for FFTs and FFT-composed operators (convolutions), CPU + NVIDIA/AMD GPU, single and double precision. Its pitch — fusing FFTs with surrounding pointwise operators at code-gen time — is exactly our 'fused FFT+pointwise-map chain' idea, done by a generator.

**Numbers.** No competitive published benchmark numbers vs FFTW/MKL/cuFFT found for the 1.0 release; distributed (multi-node) support is not the headline capability. Plainly: performance at our problem shapes is not documented.

**Relevance to us.** The one project philosophically closest to ours (kernel fusion, specialization to fixed geometry). Its existence is a citation point for why fusion wins; its thin adoption and undocumented CPU AVX-512 fp64 performance mean it is not established as a baseline we must beat — though running it on our shapes would be a cheap experiment.

**Verification status.** VERIFIED that release 1.0.0 and docs exist; performance UNVERIFIED/undocumented. Open source at spiral-software GitHub.

## 13. New hardware: distributed FFT practice on GH200/MI250X at 1000-GPU scale (portable NUFFT), and Grace CPU FFT parity with x86

*Citation:* Fischill, Adelmann, Muralikrishnan, "A Performance-Portable, Massively Parallel Distributed Nonuniform FFT", arXiv:2605.10678 (May 2026); Simakov et al., "First Impressions of the NVIDIA Grace CPU Superchip and NVIDIA Grace Hopper Superchip for Scientific Workloads", HPC Asia-W 2024, DOI 10.1145/3636480.3637097

**Claim.** The NUFFT paper is the first distributed NUFFT for heterogeneous supercomputers: Kokkos-portable, measured on Alps (GH200), JUWELS Booster (A100), LUMI (MI250X) up to 1024 GPUs, up to 1024^3 modes / 8.6e9 particles; the uniform distributed FFT underneath (heFFTe-class) remains 'a significant runtime portion' at scale. The Grace evaluation measured Grace CPU ~6% SLOWER than Sapphire Rapids+HBM on the HPCC FFT benchmark.

**Numbers.** 1024 GPUs, 1024^3 modes (NUFFT paper); Grace vs SPR-HBM: -6% on HPCC FFT (HPC Asia-W 2024).

**Relevance to us.** Post-2024 hardware has not changed the FFT picture: NVLink-C2C/superchips shift the transpose bottleneck but nobody reports a local-kernel revolution, and on the CPU side ARM Grace does not beat AVX-512 x86 at FFT — supporting the continued value of an AVX-512-specialized fp64 kernel. No evidence found of matrix engines (tensor cores, AMX) used for the LOCAL kernel inside any distributed FFT library.

**Verification status.** VERIFIED abstracts (arXiv page fetched; HPC Asia paper via search summary). MI300A/Aurora-PVC-specific distributed-FFT numbers: not found — see negative space.

## 14. Tensor-core FFT stays single-GPU and low-precision: tcFFT, FlashFFTConv — no distributed uptake

*Citation:* B. Li et al., "tcFFT: Accelerating Half-Precision FFT through Tensor Cores", arXiv:2104.11471 / IEEE 2021; D. Fu et al., "FlashFFTConv: Efficient Convolutions for Long Sequences with Tensor Cores", ICLR 2024, https://openreview.net/forum?id=gPKTTAfYBp

**Claim.** tcFFT: half-precision 1D/2D FFTs on tensor cores up to 3.24x over cuFFT fp16. FlashFFTConv: FFT-as-Monarch-matrix-decomposition on tensor cores for long-sequence convolutions (ML inference/training, single GPU). Neither is fp64, neither is distributed, and no distributed library is documented to use either as its local kernel.

**Numbers.** tcFFT: up to 3.24x vs cuFFT (fp16, single GPU, 2021-era hardware). FlashFFTConv: up to ~7x over PyTorch FFT conv on sequence-convolution workloads (their measurement, ML setting).

**Relevance to us.** Answers the 'matrix engines for the local kernel in a distributed setting' question: it has not happened, largely because the precision loss is unacceptable for fp64 HPC users — the precision-vs-throughput tradeoff everyone dodges by compressing communication instead. Our fp64 SIMD approach and tensor-core FFTs are not in competition today; for LQCD (complex fp64 fields) they are not an alternative at all.

**Verification status.** VERIFIED existence and headline numbers (arXiv/OpenReview); the 'nobody uses them distributed' part is a documented absence — see negative space.

## 15. What's next for heFFTe-style portability: a distributed FFT interface on the Kokkos ecosystem (SC'25 workshop)

*Citation:* "Development of a performance portable distributed FFT interface on top of the Kokkos ecosystem", SC'25 Workshops, DOI 10.1145/3731599.3767494; kokkos-fft issue #203 "Distributed support on top of heffte", https://github.com/kokkos/kokkos-fft/issues/203

**Claim.** kokkos-fft (shared-memory FFT wrapper for Kokkos, 2024-25) is growing a distributed layer; the open design discussion (Nov 2024) explicitly considers building it on top of heFFTe rather than reimplementing. Direction of travel: distributed FFT as a thin portability veneer over the same vendor local kernels.

**Numbers.** None published yet for the distributed layer.

**Relevance to us.** The ecosystem's 'next' is more interface consolidation, not faster kernels — reinforcing that the local batched-1D kernel is a stable, swappable commodity slot. Whoever owns the fastest kernel for a fixed shape can sit under all of these interfaces at once.

**Verification status.** ABSTRACT-ONLY (ACM listing + GitHub issue read via search); code for the distributed layer not yet released as far as found.

## Negative space (searched for, not found)

Searched for and did NOT find: (1) Any 4D or higher-dimensional DISTRIBUTED FFT library or paper — searches for four-dimensional distributed/parallel FFT return only 2D/3D pencil work; the only 4D practice found is LQCD gauge-fixing code composing cuFFT 1D/2D calls on a single GPU (QUDA's multi-GPU 4D FFT gauge fixing is an open issue since 2015, still unresolved). The L^3xT lattice case is genuinely unserved. (2) Any distributed FFT that computes the BUTTERFLIES in reduced precision with correction — all mixed-precision distributed work (Cayrols/heFFTe 2022, MFFT 2023) reduces precision only on the wire (compression/shared-exponent) and keeps the transform in fp32/fp64; tensor-core FFTs (tcFFT, FlashFFTConv, Sorna et al.) remain single-GPU and are not used as the local kernel of any distributed library I could find. (3) No evidence that lossy compression of transposes is accepted production practice — it remains research branches (ICL tech report, TACO paper, ZCCL 2025 arXiv); no release notes of heFFTe/cuFFTMp/2DECOMP&FFT ship it on by default. (4) No MI300A- or Aurora/PVC-specific distributed FFT benchmark paper (only MPI microbenchmarks on MI300A and general superchip evaluations); oneMKL/SYCL heFFTe backend exists but I found no published Aurora-scale FFT numbers. (5) No PyTorch-native distributed FFT — only DistDL-based model-parallel FNO research (arXiv:2204.01205) and TensorFlow's admittedly naive DTensor implementation; JAX's jaxDecomp is the only polished ML-framework option. (6) No independent (non-NVIDIA) at-scale evaluation of cuFFTMp against heFFTe was located. (7) Could not extract the detailed speedup tables of the Cayrols compression paper (no PDF text tooling on this cluster; WebFetch returned binary) — its headline claims are quoted from the ICL abstract, and rerunning that comparison locally would need the PDF read elsewhere. (8) FFTW itself: no release found after 3.3.10 (Sept 2021), but I could not verify this from a primary changelog page (fftw.org not fetched, GitHub releases page malformed) — treat 'FFTW is dormant' as UNVERIFIED though widely believed.

## Bottom line

Since 2020 the distributed-FFT world has consolidated, not innovated, around one architecture — batched 1D vendor kernels + pencil transposes — with all the post-2021 energy going into the communication layer (NVSHMEM in cuFFTMp, LCI in HPX, lossy/shared-exponent compression in research branches) because at scale the local kernel is a few percent of wall time (TensorFlow measured 3.6% on 8 GPUs). That cuts both ways for us: our 3.5x kernel win matters most in exactly the regimes we benchmark (single-node batched/chained, fat-node runs), and the 'drop-in local kernel' slot is real and stable (heFFTe batch API since v2.3, 2DECOMP&FFT pluggable backends, kokkos-fft building yet another veneer) but buys little in transpose-bound multi-node runs. The genuinely open territory that matches this cluster and lattice QCD: 4D distributed FFTs do not exist anywhere (QUDA's multi-GPU FFT gauge fixing has been an open issue for a decade; Grid's 4D FFT is a verified per-dimension barrel-shift + serial FFTW, ripe for replacement), fp64 CPU kernels face no tensor-core competition, and LQCD production codes only avoid FFTs because SftMom-style phase sums suffice for few momenta — the all-momenta use cases (gauge fixing, momentum-space propagators) are exactly where a fast batched-1D fp64 kernel composed into a 4D transform has no incumbent.
