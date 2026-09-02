# The rest of the distributed-FFT landscape: P3DFFT, PFFT, 2DECOMP&FFT, fftMPI, SWFFT, FFTE, cuFFTMp, FFTX

**Scope as the agent recorded it:** The distributed-FFT library landscape besides heFFTe: decompositions, local kernels, communication strategies, GPU capability, batching, maintenance, and measured head-to-heads

## 1. FFTW MPI transforms — slab-only, CPU-only, but has a real batched interface

*Citation:* FFTW 3.3.11 manual, 'Distributed-memory FFTW with MPI', fftw.org/fftw3_doc (MPI Data Distribution; MPI Plan Creation pages)

**Claim.** Decomposition is slab only: 'FFTW uses a 1d block distribution of the data, distributed along the first dimension.' Local kernel is FFTW itself (not pluggable). Communication is FFTW's internal MPI transpose ('intermediate steps of FFTW's algorithms involve transposing the array and redistributing the data'), with FFTW_MPI_TRANSPOSED_OUT/IN flags to skip the final transpose and FFTW_MPI_SCRAMBLED_OUT/IN for distributed 1D transforms. Batching exists: fftw_mpi_plan_many_dft (+ r2c/c2r variants) with a 'howmany' argument that transforms 'contiguous howmany-tuples rather than individual complex numbers' — i.e. an interleaved (AoS-style) batch along the fastest axis, not a split/SoA batch. No GPU support. FFTW itself is in maintenance mode (last release 3.3.10, Sept 2021 [UNVERIFIED exact date; 3.3.11 docs exist]).

**Numbers.** None published by FFTW itself for MPI scale. In ICL's 2022 exascale benchmark report it is one of the CPU baselines on Summit; the heFFTe ICCS-2020 paper's positioning: 'FFTW supports MPI via slab decomposition... [scalability limited to] a small number of nodes' (slab count <= N).

**Relevance to us.** The 'howmany contiguous tuples' batch layout is exactly the interleaved layout our split-complex SoA batch-lane kernels avoid — a drop-in into FFTW-MPI-shaped code implies a pack/unpack or a layout conversion at the boundary. Slab-only also means at most L ranks for an L^3 lattice.

**Verification status.** VERIFIED (fetched fftw.org manual pages). Performant code exists: FFTW 3.3.x, MPI transforms in libfftw3_mpi, CPU-only.

## 2. P3DFFT 2.x and P3DFFT++ (Pekurovsky) — the classic CPU pencil library, compile-time FFTW/ESSL kernel, largely dormant

*Citation:* D. Pekurovsky, 'P3DFFT: A framework for parallel computations of Fourier transforms in three dimensions', SIAM J. Sci. Comput. 34(4) C192 (2012), DOI 10.1137/11082748X; github.com/sdsc/p3dfft (v2.7) and github.com/sdsc/p3dfft.3 (P3DFFT++)

**Claim.** Classic P3DFFT: '2D (a.k.a. pencils) data decomposition', which lets processor counts reach 'the square of the linear grid size'; it is 'built on top of an externally provided 1D FFT library' — ESSL or FFTW chosen at compile time; Fortran90+MPI; no GPU support anywhere in the docs. P3DFFT++ is the C++ rewrite with '1D, 2D and 3D domain decomposition schemes' (3D listed as 'to come soon') and user-specifiable memory ordering/processor layout; local kernel is FFTW. Communication is MPI collectives in row/column subcommunicators [the MPI_Alltoall specifics are ABSTRACT-ONLY — 'the documentation doesn't explicitly name MPI_Alltoall']. Batching: not documented. Maintenance: classic repo at v2.7, copyright through 2019, 'limited recent activity'; P3DFFT++ ~305 commits, activity dates not verified.

**Numbers.** Historic scaling to 65,536 cores (Cray XT5/BG) is claimed in the SIAM paper [ABSTRACT-ONLY]. In the heFFTe ICCS-2020 paper P3DFFT is characterized as extending FFTW with pencil+slab support; no fresh head-to-head numbers vs heFFTe were found in what we fetched.

**Relevance to us.** The archetype of what heFFTe replaced on CPU: pencil transposes around a compile-time-selected 1D kernel. Its kernel interface (FFTW-or-ESSL at compile time) shows a second plausible integration point for our batched-1D kernel, but the project is close to dormant — heFFTe is the better target.

**Verification status.** VERIFIED for decomposition/kernels/maintenance (fetched both GitHub READMEs); ABSTRACT-ONLY for the alltoall detail and the 2012 scaling numbers. CPU-only.

## 3. PFFT (Pippig) — generalization of FFTW-MPI to multidimensional (pencil+) decompositions, built literally on FFTW, effectively frozen

*Citation:* M. Pippig, 'PFFT: An Extension of FFTW to Massively Parallel Architectures', SIAM J. Sci. Comput. 35(3) C213 (2013), DOI 10.1137/120885887; github.com/mpip/pfft

**Claim.** 'PFFT ... can be understood as a generalization of FFTW-MPI to multidimensional data decomposition' — i.e. pencil (and general d-1 dimensional) decompositions of d-dimensional transforms, computing local FFTs and transposes with FFTW-3.3's own routines (the serial kernels AND fftw's MPI transpose). Extras: ghost-cell exchange, pruned FFTs, Fortran interface; 'hybrid OpenMP/MPI support under development'. No GPU support. Batching: FFTW-style howmany plans [the README does not document it explicitly — UNVERIFIED]. Maintenance: repo references patches 'until release FFTW-3.3.5', ~199 commits; development essentially stopped years ago.

**Numbers.** SIAM paper reports scaling on BlueGene/P JUGENE up to 262,144 cores [ABSTRACT-ONLY, not fetched]. No heFFTe head-to-head found.

**Relevance to us.** Demonstrates that a 'better transpose scheduling around FFTW' library can be a thin layer — everything except the communication plan is FFTW. Local kernel not pluggable (FFTW hardwired), so not a drop-in target for us; useful mainly as a communication-strategy reference (pruned transforms map onto our fused chain idea).

**Verification status.** VERIFIED for architecture/status (fetched GitHub README); ABSTRACT-ONLY for JUGENE scaling. CPU-only.

## 4. 2DECOMP&FFT v2 (Rolfo, Flageul, Bartholomew, Spiga, Laizet) — actively maintained Fortran pencil library, now GPU-capable via cuFFT + NCCL, with a pluggable backend layer

*Citation:* S. Rolfo et al., 'The 2DECOMP&FFT library: an update with new CPU/GPU capabilities', JOSS 8(91):5813 (Nov 21, 2023), DOI 10.21105/joss.05813; github.com/2decomp-fft/2decomp-fft

**Claim.** Provides 2D-pencil and slab decompositions over MPI; 'The library provides a built-in FFT engine and supports various FFT backends: FFTW, Intel oneMKL, Nvidia cuFFT.' GPU: 'The library can perform multi GPU offloading using the NVHPC compiler suite for NVIDIA hardware', implemented with CUDA-aware MPI and NCCL, local transforms via cuFFT (JOSS paper: tested with NVHPC 23.11 / CUDA 12.3 / NCCL 2.18.5). Maintenance: active (dev branch ~1,574 commits, semantic releases v2.x). Batching of many independent 3D transforms: not documented. Also has a documented non-blocking-communication API (MPI-3 overlap) inherited from the NAG-era work.

**Numbers.** The JOSS paper is a software paper; no heFFTe head-to-head numbers found. (Historic 2DECOMP&FFT powered Incompact3d at O(10^5) cores [UNVERIFIED here].)

**Relevance to us.** One of only three libraries besides heFFTe with a genuinely pluggable local-kernel abstraction (generic/FFTW/MKL/cuFFT), and it is alive. Fortran-first API aimed at CFD codes; complex-to-complex is supported but the ecosystem is R2C/CFD-flavored. A plausible second integration target for our batched-1D CPU kernel after heFFTe.

**Verification status.** VERIFIED (fetched GitHub README; JOSS paper located and key claims cross-checked in two sources). CPU + NVIDIA GPU.

## 5. AccFFT (Gholami, Malhotra, Hill, Biros) — CPU/GPU pencil library with comm/compute overlap; unmaintained and failed correctness in ICL's Summit tests

*Citation:* A. Gholami, J. Hill, D. Malhotra, G. Biros, 'AccFFT: A library for distributed-memory FFT on CPU and GPU architectures', arXiv:1506.07933 (2015); github.com/amirgholami/accfft

**Claim.** Pencil decomposition with 'local FFT kernels (FFTW for CPUs, cuFFT for GPUs) with an optimized communication strategy' and 'overlapping communication techniques to minimize overhead from PCIe data transfers'; C2C and R2C; GPL-2. Maintenance: stale (177 commits total; no recent releases; the local kernels are fixed, not pluggable). Critically, the heFFTe ICCS-2020 paper reports: 'We do not include results with AccFFT library, since its GPU version did not verify correctness on several experiments performed in Summit.'

**Numbers.** Paper claims 'scaling of the library up to 4,096 K20 GPUs of Titan' [VERIFIED as a claim from the arXiv abstract; the measured GFLOPS are in the paper body, not fetched]. Ayala et al. relay AccFFT's own claim of 'a fairly constant speedup of ~1.5 compared with FFTE' with similar scalability. ICL's 2022 report also measured AccFFT's FFT planning/setup time as the slowest of all libraries tested (several times heFFTe's).

**Relevance to us.** Cautionary tale rather than competitor: the one older library that tried aggressive overlap is now bit-rotten and produced wrong answers on modern GPUs in independent testing. Its overlap idea, though, matches our fused-chain motivation: hiding transpose communication behind batched local work.

**Verification status.** VERIFIED for design and Titan-scale claim (arXiv abstract + GitHub); the correctness failure and 1.5x-vs-FFTE figures are VERIFIED as statements in the fetched heFFTe ICCS-2020 text. CPU + NVIDIA GPU (broken).

## 6. SWFFT (HACC / Pope et al., ANL) — cosmology's brick-to-pencil 3D FFT over FFTW, CPU-only

*Citation:* SWFFT, git.cels.anl.gov/hacc/SWFFT (stand-alone version of HACC's FFT); described in Ayala et al., heFFTe ICCS 2020 (DOI 10.1007/978-3-030-50371-0_19) and the AMReX docs

**Claim.** SWFFT 'takes three-dimensional arrays of data distributed across block-structured grids, and redistributes the data into pencil grids in z, x, and then y ... a 1D FFT is performed on the data along the pencil direction using calls to the FFTW library' — i.e. its distinguishing feature is starting from a 3D block (brick) decomposition natural to a particle code, not from slabs/pencils. Communication (per the heFFTe paper's Table): MPI_Sendrecv / MPI_Isend / MPI_Allreduce with MPI_Cart_create process grids; no collective alltoall. C++/F90, CPU-only; no batching documented; power-of-two-ish process-grid constraints [UNVERIFIED detail]. Maintenance: essentially frozen at the ECP-era release (v1.0 in ICL's 2022 table).

**Numbers.** heFFTe ICCS 2020: heFFTe's CPU version 'was based on improved versions of kernels from FFTMPI and SWFFT libraries, then its performance is at least as good as them. Therefore, heFFTe GPU is also ~2x faster than FFTMPI and SWFFT libraries' (1024^3, Summit, up to 1024 nodes).

**Relevance to us.** The brick-to-pencil reshape is exactly the shape of a lattice-QCD domain decomposition (4D bricks) feeding batched 1D FFT passes — SWFFT is the existence proof that applications write this themselves when libraries don't fit. Its local kernel is plain FFTW calls: another place a faster batched-1D kernel is the whole compute story.

**Verification status.** VERIFIED via secondary sources (heFFTe paper text fetched; GitLab repo located but its README fetched only via search snippets) — call it VERIFIED for architecture, ABSTRACT-ONLY for repo details. CPU-only.

## 7. FFTE 7.0 (Takahashi) — Fortran radix-2/3/5 library with MPI 2D-decomposition and CUDA routines; beaten >2x by heFFTe on Summit GPUs

*Citation:* D. Takahashi, FFTE 7.0, ffte.jp (released Aug 20, 2020); comparison in Ayala et al., heFFTe, ICCS 2020, DOI 10.1007/978-3-030-50371-0_19

**Claim.** Computes DFTs 'of 1-, 2- and 3-dimensional sequences of length (2^p)*(3^q)*(5^r)' (radix 2,3,4,5,8; a SPIRAL-generated variant adds 6,9,10,12,15,16); the MPI directory has parallel 1D/2D/3D routines, some 3D variants 'offering 2-D decomposition'; CUDA-based GPU routines (cuzfft{1,2,3}d.f) exist including in the MPI version. Communication: MPI_Alltoallv (per the heFFTe paper's primitives table: 'FFTE: MPI_Alltoallv, MPI_Bcast', no point-to-point machinery). License: free including commercial use but 'You may distribute this ORIGINAL package' only. Last release 2020 — dormant. Local kernels are FFTE's own Fortran — not pluggable. No batching of independent transforms documented. Note: no general prime support — 1021/10007-class sizes are outside FFTE entirely.

**Numbers.** heFFTe ICCS 2020, Summit, strong+weak scaling (Fig. 7): 'heFFTe overcomes FFTE in performance (by a factor > 2) and having better scalability' — FFTE_GPU vs heFFTe_GPU curves span 1-64 Summit nodes.

**Relevance to us.** FFTE is the closest existing thing to 'hand-written Fortran kernels inside a distributed wrapper' — and it loses to heFFTe's framework not on local compute but on communication flexibility. Reinforces: put our kernels inside heFFTe's reshape machinery rather than writing our own alltoall layer first.

**Verification status.** VERIFIED (fetched ffte.jp and the ICCS paper text). CPU + NVIDIA GPU (dated CUDA).

## 8. fftMPI (Plimpton, Sandia / LAMMPS) — arbitrary-tiling CPU FFT with an explicitly pluggable 1D kernel (KISS/FFTW/MKL)

*Citation:* S. Plimpton et al., fftMPI, lammps.github.io/fftmpi (open source, modified-BSD); LAMMPS result in Ayala et al., IPDPS 2022, 'Performance Analysis of Parallel FFT on Large Multi-GPU Systems'

**Claim.** Performs '3d and 2d FFTs in parallel as sets of 1d FFTs in each dimension of the FFT grid, interleaved with MPI communication to move data between processors', supports 'arbitrary tiling across MPI tasks' (bricks, slabs, pencils), double or single precision, and can use KISS FFT (bundled default), FFTW, or MKL for the 1D transforms — a compile-time pluggable local-kernel interface. 'CPU only execution, currently no OpenMP or GPU support.' Actively maintained as LAMMPS/SPARTA infrastructure. Batching not documented.

**Numbers.** IPDPS 2022 (fetched text): in a standard LAMMPS benchmark on 32 Summit nodes with a fixed 512^3 grid, 'the runtime for the KSPACE computation is reduced around 40% when switching from its default fftMPI (with pencils approach) to heFFTe'. ICCS 2020: heFFTe GPU ~2x faster than fftMPI (see SWFFT entry).

**Relevance to us.** Highest-relevance small target after heFFTe: its default local kernel is KISS FFT — a deliberately simple kernel — so a hand-written AVX-512 batched-1D kernel slots into an existing compile-time plugin slot and the delta would be immediately visible in LAMMPS-style workloads. Its interleaved-complex data model is the caveat for our SoA layout.

**Verification status.** VERIFIED (fetched lammps.github.io/fftmpi and the IPDPS 2022 text). CPU-only.

## 9. nb3dfft (RWTH Aachen, psOpen) — nonblocking overlap of many concurrent FFTs, fused with the dealiasing filter; in-house, not public

*Citation:* J.H. Göbbert, M. Gauding, et al., 'Overlapping of Communication and Computation in nb3dfft for 3D Fast Fourier Transformations', LNCS (2017), DOI 10.1007/978-3-319-53862-4_13; psOpen DNS papers (IOS Press)

**Claim.** An 'inhouse-developed 3D-FFT library optimized for pseudo-spectral DNS' that 'integrates the dealiasing cut-off filter into the 3D-FFT to reduce operations and data transposition, and it allows overlapping of communication and computation of multiple FFTs at the same time' — i.e. it exploits the fact that a spectral solver needs a batch of transforms per step to pipeline transposes of one against compute of another. Scaled with psOpen at 4096^3-8192^3 grids 'on JUQUEEN up to the full machine' (BlueGene/Q). CPU-only, Fortran, tied to psOpen; no public repository or maintained release found.

**Numbers.** Full-JUQUEEN (458,752 cores) scaling at 8192^3 is claimed in the psOpen/nb3dfft papers [ABSTRACT-ONLY — figures not fetched].

**Relevance to us.** The clearest published precedent for two ideas we already hold: (a) batching independent transforms to overlap transposes with compute (heFFTe reinvented this in 2022), and (b) fusing a pointwise spectral operation (their dealiasing filter, our pointwise map) into the transform to skip a data pass. Validates our fused FFT+map chain design at extreme scale.

**Verification status.** ABSTRACT-ONLY (search-level verification of the Springer chapter and psOpen abstracts; primary PDFs not fetched). CPU-only, not publicly available.

## 10. cuFFTMp (NVIDIA) — the GPU incumbent: NVSHMEM communication, slab+pencil, cuFFT-only local kernel, no batching

*Citation:* NVIDIA cuFFTMp documentation, docs.nvidia.com/hpc-sdk/cufftmp (latest noted version 12.1.3 EA, ships in NVIDIA HPC SDK); NVIDIA Technical Blog 'Multinode Multi-GPU: Using NVIDIA cuFFTMp FFTs at Scale' (2022)

**Claim.** Supports 'Slabs (1D) and pencils (2D) data decomposition, with arbitrary block sizes' (the 2022 EA blog said slabs-only; pencils/arbitrary boxes came later via cufftMpMakePlanDecomposition with lower/upper-corner 3D box descriptors), with a 'low-latency implementation using NVSHMEM' and an 'MPI-compatible interface'. C2C, R2C, C2R. Batching: the docs warn 'cufftMakePlanMany and cufftXtMakePlanMany should not be used with cuFFTMp. If they are used, any stride information will be ignored' — i.e. no supported batch>1 distributed transforms. Local kernel is cuFFT, closed, not pluggable. x86_64 and aarch64; NVIDIA GPUs only; actively shipped.

**Numbers.** NVIDIA blog, Selene (DGX-A100, 8x A100-80GB/node, NVSwitch 300 GB/s/GPU, HDR IB 200 GB/s/node): 'cuFFTMp reaching over 1.8 PFlop/s, more than 70% of the peak machine bandwidth' at 16,384^3 FP32 on 4096 A100s; strong scaling of a 2048^3 FP32 transform 'from 78ms with 8 GPUs (1 node) to 4ms with 2048 GPUs (256 nodes)', maintaining 'roughly 75% peak as the number of GPUs are doubled'. No NVIDIA-published heFFTe comparison. Independent note: the DaggerFFT preprint (arXiv 2601.12209) reports cases where native implementations outperformed cuFFTMp, 'whose performance remained unchanged as the number of GPUs increased' [ABSTRACT-ONLY, search snippet].

**Relevance to us.** On our A100 nodes cuFFTMp is the strongest baseline for any distributed-GPU ambition — but its two hard gaps (no batched transforms; closed cuFFT-only local kernel) are exactly where our batched-1D thesis lives. On CPUs it is irrelevant. Any 'drop-in local kernel' story must go through heFFTe or app code, not cuFFTMp.

**Verification status.** VERIFIED (fetched NVIDIA docs index, API-usage page, and the technical blog). GPU-only (NVIDIA), maintained, closed-source.

## 11. FFTX / SpectralPACK (Franchetti et al.) — SPIRAL code generation, strong on batched 1D and fused transforms, distributed layer thin

*Citation:* F. Franchetti et al., 'FFTX and SpectralPack: A First Look', IEEE HiPC-W 2018 (arXiv:1808.02618); github.com/spiral-software/fftx (BSD, ~1,224 commits, copyright through 2025)

**Claim.** FFTX 'generates C++ source code for transforms that are then compiled into libraries', targeting 'a CPU or a GPU (either CUDA or HIP or SYCL)'. Its shipped libraries include batched 1D FFTs ('fftx_dftbat — Forward batch of 1D FFT complex to complex' plus inverses), 3D C2C/R2C, and real 3D convolution — the code-generation analogue of our hand-writing approach, including fusing FFTs with pointwise operations (the SpectralPACK pitch). Distributed: 'MPI support exists as a SPIRAL package (spiral-package-mpi), though the README doesn't detail specific MPI capabilities' — the multi-node story is thin and not benchmarked publicly against heFFTe as far as we found. Actively developed (ECP project).

**Numbers.** No public head-to-head distributed numbers vs heFFTe or cuFFTMp found. Single-node generated-kernel numbers exist in FFTX papers [not fetched].

**Relevance to us.** The direct intellectual competitor to our approach: they generate fused, batched kernels rather than hand-writing them. Their batched-1D C2C library (dftbat) is the thing to benchmark our AVX-512 kernels against on Ice Lake; their weak distributed layer means they don't threaten the heFFTe-drop-in angle.

**Verification status.** VERIFIED for repo capabilities (fetched GitHub README); ABSTRACT-ONLY for the 2018 paper's claims; distributed capability UNVERIFIED beyond package existence.

## 12. DiGPUFFT (Czechowski, Vuduc et al.) — 2012 proof-of-concept GPU patches to P3DFFT; origin of the communication-bound analysis; abandoned

*Citation:* K. Czechowski et al., 'On the communication complexity of 3D FFTs and its implications for Exascale', ICS'12, DOI 10.1145/2304576.2304604; hpcgarage.org DiGPUFFT page

**Claim.** DiGPUFFT is 'a set of code patches that add GPU capabilities to P3DFFT' (cuFFT local kernels inside P3DFFT's pencil transposes), built 'for experimental validation of the theoretical analysis of the complexity of 3-D FFTs' — the paper that formalized distributed FFT as network-bandwidth-bound at exascale. Measured on Keeneland (ORNL). Long abandoned; superseded by AccFFT and then heFFTe.

**Numbers.** From the Keeneland profile: 'the local transpose (16%) was nearly as costly as the FFT computation (21%) itself' — i.e. even in 2012, local FFT compute was ~1/5 of runtime on GPU nodes.

**Relevance to us.** Sets the honest ceiling on our drop-in pitch for DISTRIBUTED runs: if local 1D FFT compute is ~20% of a pencil 3D FFT, a 3.5x local kernel buys at most ~15% end-to-end — unless batching lets the kernel hide inside communication (the heFFTe-batched/nb3dfft observation). Our strongest distributed story is therefore batched-overlap plus fused maps, not raw kernel speed.

**Verification status.** ABSTRACT-ONLY (ICS'12 paper located, percentages taken from search-surfaced text of the paper/report, primary PDF not fetched). GPU (2012-era), dead code.

## 13. heFFTe ICCS 2020 measurements — the reference head-to-head set on Summit

*Citation:* A. Ayala, S. Tomov, A. Haidar, J. Dongarra, 'heFFTe: Highly Efficient FFT for Exascale', ICCS 2020, DOI 10.1007/978-3-030-50371-0_19 (full text fetched and extracted)

**Claim.** heFFTe = pluggable local backends ('To compute low-dimensional FFTs, heFFTe supports several open-source and vendor libraries for single node') + optimized reshape ('Data reshape ... essentially a tensor transposition ... takes a great part of the computation time'; packing/unpacking 'account for less than 10% of the reshaping time') + tunable communication (binary point-to-point vs collective; 'heFFTe supports standard MPI_Alltoallv' plus its own heffte_alltoallv overlapping pack/unpack, 'proved efficient for up to 32 nodes'). Its comparison table records each rival's primitives: FFTMPI (MPI_Send/Irecv/Allreduce), SWFFT (Sendrecv/Isend/Allreduce/Cart_create), AccFFT (Sendrecv/Isend/Alltoallv/Cart_create), FFTE (Alltoallv/Bcast only).

**Numbers.** Summit, 1024^3 C2C fp64, pencil decomposition, 1-1024 nodes: heFFTe GPU ~2x heFFTe CPU (strong scaling, Fig. 4); local kernels accelerated 40-42x by GPU but total speedup only 2x because communication dominates (Fig. 6 profile: comm 0.4s/0.36s vs FFT compute 0.15s->0.008s on 32 nodes); heFFTe GPU ~2x fftMPI and SWFFT; heFFTe > 2x FFTE with better scalability; AccFFT GPU excluded for failing correctness; ~90% of a 25 GB/s-injection roofline model at scale; CPU version degrades past 512 nodes (12,288 cores) at 1024^3 from message-latency effects.

**Relevance to us.** This is the quantitative frame for our project: on Summit-class machines the local kernel is 2-4% of distributed runtime at scale (0.008s of ~0.8s), so our 3.5x CPU-kernel edge matters distributedly only in the batched/overlapped and few-node regimes — and matters enormously in the single-node/batched regimes our 1D campaign targets.

**Verification status.** VERIFIED (PDF fetched, text extracted, numbers quoted from the extraction).

## 14. ICL 2022 exascale FFT benchmark report + IPDPS 2022 — nine-library harness; heFFTe the only AMD/Intel-GPU option; batched 3D gives >2x

*Citation:* A. Ayala, S. Tomov, P. Luszczek, S. Cayrols, G. Ragghianti, J. Dongarra, 'FFT Benchmark Performance Experiments on Systems Targeting Exascale', ICL-UT-22-02, March 2022 (icl.utk.edu); and Ayala et al., 'Performance Analysis of Parallel FFT on Large Multi-GPU Systems', IPDPS 2022 (both PDFs fetched and extracted)

**Claim.** The 2022 report benchmarks AccFFT, 2DECOMP&FFT, FFTE, FFTW, fftMPI, heFFTe, SWFFT, P3DFFT, and 'FFTADVMPI' (an MPI_Alltoallw-based research code) on Summit (Power9+V100) and Spock (AMD MI100, Frontier precursor) with a common open-source harness. Key structural findings: 'heFFTe, AccFFT, and FFTE are the only libraries ... that provide support for distributed-memory systems with NVIDIA GPUs. Finally, heFFTe is the only library that provides support for distributed-memory systems with AMD GPUs' (and Intel GPUs, per the Discussion). All CPU libraries show 'a similar behavior' with the all-to-all dominating; AccFFT has the slowest FFT planning time, heFFTe the fastest. (Note: cuFFTMp was NOT in this comparison.)

**Numbers.** IPDPS 2022, verified from extracted text: batched 3D FFTs of size 64^3 (1 MPI/GPU, NVIDIA and AMD) show 'over 2x speedups ... comparing the cost of a single 3-D transform within a batch, to an isolated not batched computation. These speedups come from the overlap of communication and computation. The more transforms per MPI unit generates more overlap'; the advantage 'is considerably reduced' at 512^3 where communication dominates. LAMMPS KSPACE on 32 Summit nodes, 512^3: ~40% runtime reduction switching fftMPI -> heFFTe. (Caveat: the report PDF's digits were lost to a font-encoding problem in extraction; qualitative conclusions verified, per-figure numbers from that report not quoted here.)

**Relevance to us.** Two direct supports for our thesis: (1) batching many small-to-mid transforms is where distributed FFT still has >2x headroom, and it comes precisely from keeping local batched-1D compute overlapped with transposes — a faster local kernel extends the size range where overlap wins; (2) the whole field's local kernels reduce to FFTW/MKL/cuFFT calls, so heFFTe's backend slot is the one sanctioned entry point.

**Verification status.** VERIFIED (both PDFs fetched and text-extracted; one caveat on lost digits noted above).

## 15. Verma et al. GPU-FFT on Selene — slab-decomposed academic multi-GPU library (context for cuFFTMp claims)

*Citation:* M. Verma, S. Chatterjee, G. Garg, B. Sharma, N. Arya, S. Kumar, A. Saxena, M.K. Verma, 'Scalable Multi-node Fast Fourier Transform on GPUs', arXiv:2202.12756 (2022)

**Claim.** A slab-decomposition multi-node GPU FFT ('slab decomposition for data division and MPI for communication among GPUs') tested to 512 A100s on Selene at 1024^3-4096^3, with 'good scaling for 4096^3 grid with 64 to 512 GPUs'; local kernel cuFFT. No heFFTe comparison in the abstract.

**Numbers.** 'the timings of multicore FFT of 1536^3 grid with 196608 cores of Cray XC40 is comparable to that of GPU-FFT of 2048^3 grid with 128 GPUs.'

**Relevance to us.** Minor: shows academic groups still build one-off slab GPU FFTs over cuFFT rather than adopting heFFTe; the 128-GPUs ~ 200k-CPU-cores equivalence is a useful scale intuition for our A100 nodes.

**Verification status.** ABSTRACT-ONLY (arXiv abstract fetched). NVIDIA GPU.

## Negative space (searched for, not found)

Searched for but did NOT find: (1) any published head-to-head heFFTe-vs-cuFFTMp benchmark from ICL, NVIDIA, or a third party with numbers — the ICL 2022 report predates cuFFTMp's inclusion, and the only independent signal is a 2026 DaggerFFT preprint snippet saying cuFFTMp scaled flat in their tests (not fetched, weight it lightly). (2) Any distributed-FFT library documenting a split-complex / SoA local data layout — every interface found (FFTW-MPI howmany tuples, heFFTe backends, cuFFTMp descriptors, fftMPI) assumes interleaved complex, so our batch-lane layout needs a pack/unpack shim or a layout-aware backend at any integration boundary. (3) Any distributed library besides FFTW-MPI and heFFTe with documented batching of independent transforms — cuFFTMp explicitly disallows plan_many; 2DECOMP&FFT, P3DFFT, PFFT, SWFFT, FFTE document none. (4) Any lattice-QCD-specific distributed FFT benchmark or a 4D (L^3 x T) transform in any of these libraries — nobody covers 4D natively; all would compose it from lower-dim passes. (5) Public code or repository for nb3dfft (in-house at RWTH/JSC). (6) Current SWFFT GPU status — undocumented publicly. (7) Digit-level numbers from the ICL-UT-22-02 report — the PDF's embedded font defeated text extraction, so its per-figure timings are not quoted (qualitative conclusions were extracted verbatim); the figures could be recovered by rendering pages to images if needed. (8) FFTX's distributed MPI capability beyond the existence of spiral-package-mpi — no docs, no benchmarks. (9) Recent (post-2020) FFTE or P3DFFT releases — both appear dormant.

## Bottom line

The field splits into three tiers: dormant CPU pencil libraries that are all thin transpose layers over FFTW/MKL (P3DFFT, PFFT, SWFFT, fftMPI, FFTE — only fftMPI and 2DECOMP&FFT are both alive and expose a pluggable local-kernel slot like heFFTe's); a closed GPU incumbent (cuFFTMp: NVSHMEM, slab+pencil, strong Selene numbers, but cuFFT-only and no batching); and code generation (FFTX, our real intellectual competitor on fused batched-1D kernels, with a thin distributed layer). The verified Summit profile (local FFT ~2-4% of a large distributed transform once communication dominates) caps what a 3.5x local kernel buys at scale — EXCEPT in exactly the regime we target: heFFTe's own IPDPS-2022 data shows batched small/mid 3D transforms gain >2x from overlapping batched-1D compute with transposes, which is where a faster batched-1D kernel directly extends the win. Practical consequence: heFFTe's backend interface remains the primary drop-in target, fftMPI (KISS-FFT default, LAMMPS install base) and 2DECOMP&FFT (active, cuFFT/NCCL GPU port) are the two secondary targets, and every interface will require an interleaved<->SoA shim at the boundary.
