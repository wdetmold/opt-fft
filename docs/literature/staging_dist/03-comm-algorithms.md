# Communication algorithms and theory: decompositions, all-to-all volume, lower bounds, overlap, batching

**Scope as the agent recorded it:** Algorithms and theory of distributed FFT communication: decompositions (slab/pencil/brick), all-to-all volume and lower bounds, communication-avoiding and single-all-to-all schemes, overlap, topology/MPI effects, batching, and the bandwidth-vs-latency transition

## 1. Communication complexity of 3D FFTs at exascale (torus lower bound, Ω(n^3/(P^{5/6}·β)))

*Citation:* K. Czechowski, C. McClanahan, C. Battaglino, K. Iyer, P.-K. Yeung, R. Vuduc, "On the communication complexity of 3D FFTs and its implications for Exascale", ICS 2012, DOI 10.1145/2304576.2304604 (PDF: vuduc.org/pubs/czechowski2012-ics-xfft.pdf)

**Claim.** Each node in a pencil 3D FFT sends ~n^3/P words per exchange; on a fully connected network T_net ≈ 2·n^3/(P·β_link). On a 3D torus the optimal-placement lower bound is "T_net = Ω(n^3 / (P^(5/6)·β_link))" (modeled as simultaneous √P-node all-to-alls within P^{1/6}×P^{1/6}×P^{1/6} subblocks), while the realistic global-all-to-all estimate is bisection-limited, T_net ≈ 2·n^3/(P^{2/3}·β_link). Memory-hierarchy traffic obeys the Hong–Kung-style Ω(n/√Z)-per-element bound (their Tmem model carries the log_Z n factor). Predictions: a 3D FFT reaches only "0.08% of the 4 EF/s peak" in 2020 vs ~0.5% of peak then-current; on their GPU-like design "network time dominates memory time by 2.8×" and flop time is "practically negligible, as communication time dominates it by roughly three orders of magnitude"; "the slopes of these lines suggest that a 4D torus is well-balanced for an FFT".

**Numbers.** Calibration on 4,096 nodes (98,304 cores) of Hopper (Cray XE6, Gemini 3D torus 17×8×24): "network communication time dominates computation time" across problem sizes, with computation growing faster with size (O(n^3 log n) vs O(n^3)); "There is no overlap" between the measured phases. Exascale projections (not measurements): comm/flop ratio ~1000:1.

**Relevance to us.** This is the theory ceiling for our drop-in ambition: past a handful of nodes the transform is network-bound regardless of local-kernel speed, and the bound scales as P^{-5/6} (placement-optimal) to P^{-2/3} (bisection). Our 3.5x local win matters exactly where their model says flops still matter — small node counts and fat nodes — which is our 1–8 node A100 regime.

**Verification status.** VERIFIED (full PDF fetched and read). Analytical models + measured calibration; the exascale numbers are explicit projections. No performant code deliverable — it's a modeling paper.

## 2. heFFTe (ICCS 2020): local kernel = vendor batched-1D/FFT calls; roofline set by network bandwidth

*Citation:* A. Ayala, S. Tomov, A. Haidar, J. Dongarra, "heFFTe: Highly Efficient FFT for Exascale", ICCS 2020, LNCS 12137, DOI 10.1007/978-3-030-50371-0_19 (netlib.org/utk/people/JackDongarra/PAPERS/heffte.pdf)

**Claim.** heFFTe's structure is reshapes (pack → MPI all-to-all → unpack) around local FFT calls into FFTW/MKL/cuFFT-style backends ("API follows styles from FFTW3 and CUFFT"); its comm roofline is Ψ = 5P·log(N)·B/(α·r) with r = number of reshapes (they use r=4 for brick input/output on Summit, α=16 bytes for fp64 complex), i.e., peak FFT rate is set entirely by inter-node bandwidth B and reshape count, not flops: "Ψ_Summit = 1.953·P·log(N)" GFlop/s, and measured performance gets "about to 90% close to peak value" of that comm roofline.

**Numbers.** Summit (4,608 nodes, 2 P9 + 6 V100): GPU version ~2x faster than CPU version and hence ~2x vs FFTMPI/SWFFT on 1024^3 strong scaling; local kernels >40x faster on GPU yet total speedup only ~2x — communication absorbs the rest (Fig. 6 profile at 32 nodes). CPU version stops scaling past 512 nodes (12,288 cores) on 1024^3: "due to latency impact ... the number of messages become very large while their size becomes very small". LAMMPS Rhodopsin 128^3: heFFTe-GPU ~2x KSPACE speedup vs FFTMPI.

**Relevance to us.** Confirms our key premise: heFFTe's per-rank work is exactly a vendor batched-1D call plus pack/unpack, so our faster batched-1D kernel is drop-in — but the roofline formula says total win is capped by B and r once the all-to-all saturates. It also pins the latency wall: 1024^3 over ~12k ranks is already message-size-starved on a CPU machine, which sets the useful rank count for LQCD-sized L^3×T lattices far lower.

**Verification status.** VERIFIED (full PDF read). Performant open-source code: github.com/icl-utk-edu/heffte.

## 3. Measured comm fraction >90% on multi-GPU; slab-vs-pencil crossover at 64 Summit nodes; batched 3D FFTs 2x via overlap

*Citation:* A. Ayala, S. Tomov, M. Stoyanov, A. Haidar, J. Dongarra, "Performance Analysis of Parallel FFT on Large Multi-GPU Systems", IPDPS 2022 (netlib.org/utk/people/JackDongarra/PAPERS/Performance_Analysis-fft-ipdps22.pdf)

**Claim.** (1) Comm fraction: for a 512^3 c2c FFT on 24 V100s (4 Summit nodes), "being communication for this problem over 90% of runtime"; pack/unpack "account for less than 10% of runtime" on GPUs. (2) Cost models: T_slabs = (Π−1)(L + 16N/(B·Π²)); T_pencils = (P−1)(L + 16N/(B·P·Π)) + (Q−1)(L + 16N/(B·Q·Π)) — per-pair message size shrinks quadratically with rank count, which is the bandwidth→latency transition mechanism. (3) Using these models they "predict that the slabs decomposition should be faster than the pencil approach when using less than 64 nodes" (384 GPUs) and verify it experimentally. (4) "Network saturation causes an exponential decrease in the average bandwidth achieved by each process". (5) All-to-all beats point-to-point at large node counts; MPI_Alltoallw (the zero-copy datatype route) "is far less optimized compared to MPI_Alltoall(v)" and wasn't even GPU-aware in SpectrumMPI. (6) Batched distributed FFTs: 64^3 3D FFTs in a batch show "over 2× speedups ... These speedups come from the overlap of communication and computation. The more transforms per MPI unit generates more overlap with network exchanges"; but for 512^3 "the advantage of batching ... is considerably reduced since computation cost becomes negligible compared to the communication cost".

**Numbers.** Summit, up to 3,072 V100s; 512^3 c2c: comm >90% of runtime at 24 GPUs; total ~0.09 s at 24 GPUs for both p2p and all-to-all; strided vs contiguous cuFFT batch-1D calls show a timing spike for strided input; LAMMPS 512^3 on 32 nodes: 40% KSPACE reduction switching fftMPI→tuned heFFTe; batched 64^3: >2x vs unbatched on NVIDIA and AMD.

**Relevance to us.** The single most useful measured paper for us. It quantifies (a) how little local flops matter at scale (>90% comm at just 4 nodes for 512^3), (b) that slab wins at small node counts — our likely deployment regime, (c) that batching many independent transforms (the LQCD case: many propagators/sources) buys a measured 2x through comm/comp overlap precisely when local compute is non-negligible — which our 3.5x-faster kernel makes *harder* (less compute to hide comm behind), and (d) that strided local batch-1D input is a measured cuFFT weakness our SoA batch-lane layout is designed to avoid.

**Verification status.** VERIFIED (full PDF read). Code is heFFTe (open source).

## 4. FFTU (Koopman–Bisseling): single all-to-all, cyclic-to-cyclic, same input/output distribution, p ≤ √N

*Citation:* T. Koopman, R. H. Bisseling, "Minimizing communication in the multidimensional FFT", arXiv:2203.11795v2 (Dec 2023); published in SIAM J. Sci. Comput. 45(6), 2023. Code: gitlab.com/Thomas637/FFT (GPL)

**Claim.** A multidimensional generalization of the cyclic-to-cyclic 1D parallel FFT: "(i) has only a single all-to-all communication step; (ii) works for up to p_max = √N processors...; (iii) starts and finishes in the same distribution." BSP cost T_FFT = 5(N/p)·log N + 12(N/p) + (N/p)·g + l — i.e., each element communicated at most once (N/p words per processor, one synchronization), at the price of 12N/p extra twiddle flops (two extra complex multiplies per element, fused into packing). Slab needs p ≤ n1; pencil p ≤ N^{2/3} with 2 redistributions for d=3; cyclic reaches √N (32,768 procs for 1024^3) with one. Same-distribution property means "the inverse FFT can be performed directly after the FFT, possibly with an elementwise local operation interspersed, without the need for data reordering."

**Numbers.** Snellius (dual 64-core AMD Rome 7H12/node, HDR100 InfiniBand fat tree), up to 4,096 cores, N = 2^30: 1024^3 FFTU speedup 149x vs sequential FFTW (0.118 s) vs PFFT 98.5x and heFFTe 119x (vs MKL); 64^5 FFTU 176x, PFFT 225x (superlinear). Switching Intel MPI→OpenMPI 4.1.1 cut FFTU p=512 time 0.664 s→0.515 s (22%) — MPI implementation alone. For same-distribution runs FFTU beat PFFT at all p; with transposed output allowed, PFFT/FFTW hide their extra step well and roughly tie.

**Relevance to us.** Directly matches our fused FFT→pointwise-map→inverse chain regime: a cyclic distribution does forward + elementwise map + inverse with ONE all-to-all each way and zero reordering, and the extra cost is exactly a twiddle-multiply fused into packing — the kind of fused pointwise stage our kernels already do at ~zero cost. For a batched-1D distributed variant this is the communication-minimal template to beat.

**Verification status.** VERIFIED (full paper read). Performant open-source MPI + BSPlib code exists (uses FFTW locally).

## 5. SOI FFT (SC'13): distributed 1D FFT with one all-to-all instead of three, at a μ=5/4 oversampling cost

*Citation:* J. Park, G. Bikshandi, K. Vaidyanathan, P. T. P. Tang, P. Dubey, D. Kim, "Tera-scale 1D FFT with low-communication algorithm and Intel Xeon Phi coprocessors", SC'13, DOI 10.1145/2503210.2503242. Code: github.com/IntelLabs/SOI_FFT

**Claim.** Conventional distributed 1D FFT (Cooley-Tukey N=MP factorization) "fundamentally requires three all-to-all communication steps. This all-to-all communication can account for anywhere from 50% to over 90% of the overall running time." The segment-of-interest algorithm replaces the top-level decomposition with convolution-and-oversampling so "one all-to-all communication step suffices", where subproblems grow to M' = μM with "μ ... typically chosen to be 5/4 or less" — i.e., ~3x less communication for ~25% extra data/flops. In their model the standard algorithm makes coprocessors pointless (14% speedup) while SOI gets ~70%, because communication is the limiter.

**Numbers.** 1 TFLOP/s (fp64 1D FFT) on 64 Xeon Phi nodes; 6.7 TFLOP/s on 512 nodes (N ≈ 2^27 per node weak scaling); modeled 32-node instance: T_fft=0.50 s, T_conv=0.64 s, T_mpi=0.67 s at 3 GB/s per node.

**Relevance to us.** The canonical 'trade redundant computation for fewer all-to-alls' result, and it is specifically about the 1D transform we are building. If our batched-1D kernel ever goes distributed (very long L), SOI-style oversampling is the known way to cut the three transposes to one — and our surplus of cheap local flops (we're 3.5x faster than the libraries these tradeoffs were tuned against) shifts the μ tradeoff further in favor of redundant computation.

**Verification status.** VERIFIED (full paper text read). Performant code exists (IntelLabs/SOI_FFT, CPU/KNC era).

## 6. Zero-copy transposes via MPI_Alltoallw datatypes (Dalcin–Mortensen–Keyes; mpi4py-fft)

*Citation:* L. Dalcin, M. Mortensen, D. E. Keyes, "Fast parallel multidimensional FFT using advanced MPI", J. Parallel Distrib. Comput. 128 (2019); arXiv:1804.09536

**Claim.** Global redistribution implemented directly with MPI_Alltoallw + subarray datatypes, eliminating explicit pack/unpack ("economizes in local work" despite "all-to-all communication of discontiguous data being generally slower"); works for slab, pencil, and higher-dimensional decompositions in ~hundreds of lines. Measured on a dragonfly network (Cray Aries).

**Numbers.** Shaheen II (Cray XC40, 6,174 nodes, 32 Haswell cores/node, Aries dragonfly): slab weak scaling — their global redistributions "approximately 40-50% faster than for P3DFFT"; pencil strong scaling 700^3/2048^3 — "5-10% faster than P3DFFT and 1-5% faster than 2DECOMP&FFT"; 4D 128^4 array: 5-15% faster than PFFT from 128 to 4096 cores. Caveat: in mixed intra/inter-node mode with a large per-node mesh, "the MPI_ALLTOALL(V) based global redistribution is faster". FFTW-MPI scales poorly at high core counts in their slab tests.

**Relevance to us.** Two lessons: pack/unpack can be pushed into the network stack, but only pays off when the MPI implementation optimizes Alltoallw — which the ICL GPU papers show it does not on GPU systems; and CPU-side, the pack/twiddle fusion we already do in our kernels is the portable version of the same saving. Also the only dragonfly-measured FFT study in this set.

**Verification status.** VERIFIED (full paper read). Performant code: mpi4py-fft (and the C prototype).

## 7. Overlap that actually works: tiled MPI_Ialltoall + autotuning (PPoPP'14), and its balance limit

*Citation:* S. Song, J. K. Hollingsworth, "Designing and Auto-Tuning Parallel 3-D FFT for Computation-Communication Overlap", PPoPP 2014, DOI 10.1145/2555243.2555249 (cs.umd.edu/~hollings/papers/ppopp14.pdf)

**Claim.** Divide the volume into tiles along z; pipeline FFTy+pack / MPI_Ialltoall / unpack+FFTx over tiles with a window parameter W, autotuning tile size (small tiles = more overlap, larger messages = better bandwidth). Software-only, "fully asynchronous communication without any support from special hardware". Overlap benefit is bounded by comm/comp balance: on the faster network the win drops because there is less exposed communication to hide, and there is less overlappable compute.

**Numbers.** Speedup vs FFTW-MPI: 1.23x–1.68x on UMD cluster (Myrinet), 1.10x–1.40x on Hopper small scale, 1.48x–1.76x on Hopper at larger scale (up to 20483 on 256+ processes). Direct overlap evidence (p=32, 640^3): all-to-all Wait time 1.6 s, overlappable compute 1.2 s; overlap cuts Wait to 0.4 s — "nearly achieves the perfect computation-communication overlap". The rival overlap scheme (TH, Hoefler-style) peaked at 1.17x and was sometimes slower than FFTW.

**Relevance to us.** The honest quantification of overlap: it hides min(comm, comp), so it roughly doubles throughput only when the two are balanced. Our 3.5x faster local kernel *shrinks* the compute available to hide communication behind — batching independent transforms (the LQCD many-propagator case) is the way to restore overlappable work, exactly as heFFTe's batched 2x result shows.

**Verification status.** VERIFIED (full paper read). Code was research-grade; the technique lives on in libraries (heFFTe pipelining, 2DECOMP&FFT nonblocking API).

## 8. Hardware-offloaded nonblocking all-to-all with P3DFFT (Kandalla et al.)

*Citation:* K. Kandalla, H. Subramoni, K. Tomko, D. Pekurovsky, S. Sur, D. K. Panda, "High-performance and scalable non-blocking all-to-all with collective offload on InfiniBand clusters: a study with parallel 3D FFT", Computer Science – R&D (ISC 2011), DOI 10.1007/s00450-011-0170-4

**Claim.** First scalable nonblocking all-to-all personalized exchange using InfiniBand ConnectX-2 CORE-Direct task-list offload, with P3DFFT restructured to overlap the transpose with application-level computation; reported near-total overlap of the offloaded collective and ~23% application-level runtime improvement around 512 processes.

**Numbers.** ~99% comm/comp overlap for the offloaded nonblocking all-to-all beyond 512 processes; ~23% overall P3DFFT kernel runtime improvement (per abstract/secondary summaries).

**Relevance to us.** Overlap without burning CPU/GPU cycles on progression is possible when the NIC executes the collective — relevant on our InfiniBand cluster, where MPI progression competing with our hand-tuned compute loops would erode the kernel advantage we've built.

**Verification status.** ABSTRACT-ONLY (paywalled; numbers from the abstract and citing texts, not the full PDF). The offload mechanism's descendants exist in hardware (Mellanox SHARP / BlueField NBC offload — see BluesMPI, ISC 2021).

## 9. All-to-all algorithmics: Bruck (log P, latency-regime) vs pairwise (bandwidth-regime), and runtime algorithm switching

*Citation:* J. Bruck, C.-T. Ho, S. Kipnis, E. Upfal, D. Weathersby, "Efficient algorithms for all-to-all communications in multiport message-passing systems", IEEE TPDS 8(11), 1997, DOI 10.1109/71.642949; N. Netterville, K. Fan, S. Kumar, T. Gilray, "A Visual Guide to MPI All-to-all", IEEE HiPC-workshops 2022, DOI 10.1109/HiPCW57629.2022.00008

**Claim.** Bruck's algorithm runs an all-to-all in ⌈log2 P⌉ rounds, sending each element up to log P times (total volume ~(N/2)·log P vs N for pairwise): optimal when the per-pair message is latency-dominated. Pairwise/spread-out sends each element once in P−1 rounds: optimal when bandwidth-dominated. MPICH/Open MPI select between them at runtime by message size and P (MPICH's documented cutover is small messages ≤256B–1KB → Bruck). For a pencil FFT the per-pair message is ~16N/(P·Π) bytes, so strong-scaling a fixed lattice slides the transpose from the pairwise regime into the Bruck/latency regime — the mechanism behind the measured 512-node scaling walls.

**Numbers.** No new measurements here; the regime boundary instantiated for FFT: 1024^3 fp64-complex over 6,144 ranks (pencil P=24,Q=32... per heFFTe SC'21 data) gives per-pair messages of order tens of KB and measured scaling collapse (see SC'21 entry).

**Relevance to us.** This is the theory of when OUR distributed regime becomes latency-bound: batching B independent 1D transforms multiplies every per-pair message by B — batching is a message-aggregation strategy, moving the transpose back into the bandwidth regime where the network is actually usable. That is a communication argument for the batched-1D design independent of SIMD efficiency.

**Verification status.** UNVERIFIED for the primary Bruck paper (not fetched; textbook material); the MPI-switching behavior is documented in MPICH source/docs and the cited visual-guide paper (fetched abstract only).

## 10. MPI implementation and tuning dominate the transpose: SC'21 ExaMPI measurements on Summit

*Citation:* A. Ayala, S. Tomov, M. Stoyanov, A. Haidar, J. Dongarra, "Accelerating Multi-Process Communication for Parallel 3-D FFT", ExaMPI Workshop @ SC 2021, DOI 10.1109/ExaMPI54564.2021.00011 (netlib.org acc-fft-sc21.pdf)

**Claim.** For the same heFFTe binary and 1024^3 FFT, switching MPI distribution changes the communication runtime materially: "MVAPICH with GDR support performs better than the other two, for up to 192 GPUs (64 nodes); while for a greater number of resources ... the default MPI in Summit (IBM's Spectrum) tends to be faster." Strong scaling "stops linearly scaling" at 512 nodes (3,072 GPUs, 6,144 ranks) "due to the impact of latency ... small size message become difficult to be managed by the MPI distribution". Collective all-to-all overtakes point-to-point at scale. Multi-rail/NCCL-class transports were work-in-progress for the transpose.

**Numbers.** Summit, up to 512 nodes / 3,072 V100; 1024^3 c2c; box-plot MPI-time variability grows sharply once nodes are saturated (6 ranks/node).

**Relevance to us.** Reinforces that on our cluster the choice and tuning of the MPI stack over InfiniBand is a first-order term — the same magnitude as algorithmic choices — and that any distributed benchmark of our kernel must control MPI library/GDR settings or the comparison is meaningless (echoes FFTU's 22% swing from Intel MPI→OpenMPI).

**Verification status.** VERIFIED (full PDF read). heFFTe v2.1 is the shipped artifact.

## 11. No batching in existing distributed FFT libraries (SC'25 workshops), while applications want batches

*Citation:* Y. Asahi et al., "Development of a performance portable distributed FFT interface on top of the Kokkos ecosystem", SC'25 Workshops, DOI 10.1145/3731599.3767494; code github.com/yasahi-hpc/distributed-FFT-for-kokkos

**Claim.** "A further limitation of existing distributed FFT libraries is the absence of batching capabilities, a feature commonly available in shared-memory vendor libraries" — users must write glue code around single 3D transforms; plasma-turbulence codes batch 2D FFTs over poloidal planes (batched = embarrassingly parallel along remaining axes). (heFFTe gained 2D/3D batched transforms in the IPDPS'22 work, but no user-facing batched *1D* distributed transform exists in any mainstream library.)

**Numbers.** None extracted (ACM PDF 403-blocked; claims from indexed full-text snippets).

**Relevance to us.** Names the gap our project sits in: distributed *batched* transforms are demanded by applications (plasma, LQCD-many-sources) and unserved. A hand-tuned batched-1D kernel plus a batched-aware transpose (aggregate the batch into each all-to-all message) is a coherent library proposition, not just a kernel swap.

**Verification status.** ABSTRACT-ONLY. Kokkos distributed-FFT code exists publicly but performance numbers unverified.

## 12. Pruned / zero-padded transforms to cut transpose volume in convolution workloads

*Citation:* FFTW notes, "Pruned FFTs with FFTW" (fftw.org/pruned.html); P3DFFT documentation (pruned transforms, D. Pekurovsky, SIAM J. Sci. Comput. 34(4), 2012); T. Blattner et al. framework paper "A framework for low communication approaches for large scale 3D convolution", ICPP-W 2022, DOI 10.1145/3547276.3548626; distributed NUFFT arXiv:2605.10678

**Claim.** In zero-padded (linear) convolution, ~half or more of FFT inputs are zeros; skipping 1D sub-transforms over all-zero lines saves "a factor of two or so" locally (FFTW note) and — the distributed point — a pruned first pass shrinks the data entering the transposes, so the all-to-all volume drops proportionally. P3DFFT exposes pruned transforms so the library "avoids unnecessary processor communication" for wave-spectrum data; the 2026 distributed-NUFFT work implements σ=2 oversampling as 2^d independent (M/2)^d FFTs on separate communicators/streams, overlapping their transposes.

**Numbers.** ~2x local-work saving for 3D zero-padded convolution (FFTW estimate); no published end-to-end distributed comm-fraction number found for pruning specifically.

**Relevance to us.** LQCD correlator constructions are convolutions: if a chained FFT→map→inverse pipeline has known-zero padding or only a segment of interest in the output, pruning is free communication reduction that composes with our fused chain path — the transposes carry only live lines.

**Verification status.** ABSTRACT-ONLY / secondary (fftw.org page and P3DFFT docs not fetched this session; NUFFT paper skimmed via search excerpts).

## 13. Topology and congestion: dragonfly/Slingshot vs fat-tree for all-to-all

*Citation:* D. De Sensi, S. Di Girolamo, K. H. McMahon, D. Roweth, T. Hoefler, "An In-Depth Analysis of the Slingshot Interconnect", SC 2020, arXiv:2008.08886; K. Czechowski et al. ICS'12 (torus analysis, above); H. Jagode et al., task placement of parallel multidimensional FFTs on mesh networks (UT-CS tech report / ISPA 2008)

**Claim.** On Slingshot (dragonfly, adaptive routing + congestion control), measured congestion impact on a victim workload drops from 2.3x to 1.5x when co-running allocation shrinks 512→128 nodes, and Slingshot "is less affected by congestion" than prior networks — i.e., modern dragonflies substantially tame, but do not eliminate, the all-to-all's sensitivity to placement and interference. Czechowski's torus analysis shows optimal sub-block placement is worth up to ~10x vs naive bisection-limited estimates; the Jagode line of work showed BG/L-era FFT all-to-alls gained materially from custom rank-to-torus mappings. Fat-tree (Summit, and our InfiniBand cluster) is the benign case: FFTU/heFFTe results above were all measured on fat trees.

**Numbers.** Slingshot congestion factors 1.5–2.3x (SC'20, general traffic, not FFT-specific); no per-FFT comm-fraction-by-topology table exists in the sources fetched.

**Relevance to us.** Our cluster's InfiniBand fat tree is the friendly topology for transposes — published fat-tree results (Summit, Snellius) transfer to us better than dragonfly-era caveats. If results are ever ported to a dragonfly machine (Frontier-class), expect placement/congestion noise of tens of percent on the transpose that no kernel work can recover.

**Verification status.** ABSTRACT-ONLY for Slingshot and Jagode (not fetched in full); the torus math is VERIFIED via Czechowski.

## Negative space (searched for, not found)

Searched for but NOT found: (1) Any published communication analysis or lower bound specific to BATCHED distributed FFTs — the message-aggregation argument (batch multiplies per-pair message size, deferring the latency wall) appears nowhere as a measured or proven result; the closest is heFFTe's batched 2x at 64^3 (attributed to overlap, not aggregation). This looks genuinely open and is directly monetizable by our project. (2) Any modern LQCD-scale distributed FFT communication study — the only domain-specific paper surfaced is the 2005 'Performance of the 3D FFT on the 6D network torus QCDOC' (not fetched); nothing on 4D L^3×T complex transforms on GPU-era machines. (3) A primary-source number for the oft-quoted 'communication is ~50% at 512 cores' for heFFTe CPU — it appears in ECP summaries but I could not verify it in a fetched primary text (the ICL-UT-22-02 report PDF extracted with unreadable digit glyphs; its qualitative conclusion — 3-D FFTs are communication-bound and inter-node bandwidth ratio translates ~1:1 into FFT speed — was readable and is consistent). (4) Frontier/dragonfly measured comm fractions for heFFTe (only Spock precursor data at ≤36 nodes exists in what I could read). (5) A rigorous distributed-memory communication lower bound matching the single-all-to-all upper bound (Koopman–Bisseling prove the algorithm, not a matching Ω(N/p)-one-round optimality theorem for p ≤ √N; the BSP folklore is that one round of N/p is optimal but I found no formal proof fetched). (6) 'FFT entirely without all-to-all' — nothing credible beyond redundant-computation schemes (SOI, cyclic) and wafer-scale hardware (Cerebras arXiv:2209.15040, not fetched/verified). Also not covered for lack of verifiable numbers: NCCL/NVSHMEM-based transposes (cuFFTMp uses NVSHMEM; NVIDIA's docs claim strong multi-node scaling but I fetched no primary benchmark), and Fugaku Tofu-D FFT campaigns.

## Bottom line

The literature is unanimous and quantitative: beyond a few nodes the transpose owns the distributed FFT (measured >90% of runtime at just 4 Summit nodes for 512^3; 50–90% for distributed 1D on CPU clusters), the transpose's per-pair message shrinks as 1/P^2 so strong scaling always hits a latency wall (measured at ~512 nodes for 1024^3 on both CPU and GPU Summit configurations), and the levers that actually move the needle are: fewer all-to-alls (cyclic/FFTU: one, same in/out distribution — a perfect match for our fused FFT+map chain; SOI: one instead of three for 1D at 25% redundant work), slab-not-pencil below ~64 nodes, MPI stack tuning (20–40% swings), overlap (up to 1.76x, but only when comm≈comp — our faster kernel reduces hideable compute), and batching (measured 2x, and it aggregates messages — the unpublished, open opportunity for a batched-1D library). Consequence for us: the drop-in-to-heFFTe claim is architecturally correct (its local kernel is exactly a vendor batched-1D call), but its end-to-end payoff is Amdahl-capped by the comm fraction — our 3.5x matters most at 1–8 nodes, in batched/chained regimes, and as surplus compute to spend on communication-avoiding redundancy (SOI-style) rather than as raw speed at scale.
