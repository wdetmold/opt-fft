# GPU and exascale practice: cuFFTMp/NVSHMEM, FFT-ECP, Frontier/Aurora, fp64 rates, record-scale runs

**Scope as the agent recorded it:** GPU and exascale distributed FFT practice: cuFFTMp/NVSHMEM, heFFTe/FFT-ECP, Frontier/Aurora, MPI-vs-NCCL-vs-NVSHMEM for the transpose, single-node 8-GPU vs multi-node, fp64-rate effects, and record-scale runs

## 1. cuFFTMp: NVSHMEM-based multi-node FFT, the current NVIDIA flagship

*Citation:* NVIDIA Technical Blog, "Multinode Multi-GPU: Using NVIDIA cuFFTMp FFTs at Scale" (2022), https://developer.nvidia.com/blog/multinode-multi-gpu-using-nvidia-cufftmp-ffts-at-scale/ ; current docs https://docs.nvidia.com/hpc-sdk/cufftmp/

**Claim.** cuFFTMp is a closed-source multi-process extension of cuFFT whose transposes are done with NVSHMEM kernel-initiated communication over a global GPU address space, making it 'independent of the quality of the MPI implementation'. Docs (fetched): supports slab (1D) and pencil (2D) decompositions with arbitrary block sizes, has an MPI-compatible bootstrap interface, ships in the HPC SDK for x86_64/aarch64. Early-access supported only optimized slab decomposition; pencil came later.

**Numbers.** Selene (A100-80GB, DGX SuperPOD): weak scaling to 1.8 PFlop/s at 16384^3 SINGLE precision on 4096 GPUs (~70-75% of a bandwidth-derived peak); strong scaling 78 ms on 8 GPUs (1 node) -> 4 ms on 2048 GPUs (blog does not state the grid size for that curve next to the extracted text). Summit (V100): >50 TFlop/s FP32 and >40 TFlop/s FP64 on a 4096^3 transform using 1536 V100s (256 nodes). Note the fp64 headline at scale is ~40 TF on 1536 GPUs, i.e. ~26 GFlop/s/GPU — communication-bound, ~0.3% of local fp64 peak.

**Relevance to us.** cuFFTMp is what a GPU port of our pencil-decomposition thesis would compete with on NVIDIA hardware, and the 8-GPU/1-node starting point of its strong-scaling curve is our 8x A100 node. Its per-GPU throughput at scale (tens of GFlop/s fp64) shows the distributed regime is bandwidth/network-limited — a hand-written local kernel buys little there; the leverage is single-node and batched-small regimes.

**Verification status.** VERIFIED (blog and docs fetched). Performant binary exists in NVIDIA HPC SDK; source not available. Marketing numbers are NVIDIA's own measurements, no independent replication found.

## 2. ECP FFT benchmark report (ICL-UT-22-02): the transpose is 90-95% of a distributed GPU FFT

*Citation:* A. Ayala, S. Tomov, P. Luszczek, S. Cayrols, G. Ragghianti, J. Dongarra, "FFT Benchmark Performance Experiments on Systems Targeting Exascale", ICL Tech Report ICL-UT-22-02, March 2022, https://icl.utk.edu/files/publications/2022/icl-utk-1548-2022.pdf

**Claim.** Full text extracted. Vampir traces of back-to-back 1024^3 3D FFTs with heFFTe+cuFFT on Summit GPUs (process grids (1,4,4)->(4,1,4)->(4,4,1), i.e. 16 ranks, 1 GPU/rank): pipelined Isend/Irecv occupies 84-93% and MPI_Alltoall 95.13% of runtime; measured 540 GFlop/s (pipelined) vs 313 GFlop/s (alltoall). On GPUs, packing/unpacking drops to under ~5% (vs ~40% on CPU with Alltoall). Conclusions verbatim: '3-D FFTs are communication-bound and therefore the local (single GPU) performance is less important in the distributed computing setting'; Summit's ~2x inter-node bandwidth over Spock translated to ~2x faster 3D FFTs; rocFFT on MI100 was up to ~3x slower than cuFFT on V100 for single-GPU 3D FFTs, and Spock (single NIC) scaling deteriorated past a few nodes as all-to-all dominates.

**Numbers.** 1024^3 c2c fp64, 16 V100 GPUs on Summit: 540 GFlop/s pipelined P2P vs 313 GFlop/s MPI_Alltoall, MPI 90-95% of runtime; 168 Power9 cores: 72.5 GFlop/s (pipelined) vs 44.5 GFlop/s (Alltoall, pack+unpack ~40%).

**Relevance to us.** This is the strongest documented statement of the fact our project leans on: heFFTe's local per-rank work is a vendor batched-1D call and is NOT the bottleneck at multi-node scale — but it IS ~all of the time at 1 node/few GPUs and in the batched-small regime, which is exactly where a faster hand-written batched-1D kernel is a drop-in win.

**Verification status.** VERIFIED (PDF fetched and text-extracted; some digits lost to font ligatures — GPU count inferred from the printed process grids). heFFTe code is open (bitbucket.org/icl/heffte, now github).

## 3. Ayala et al. IPDPSW 2022: MPI-routine choice, GPU-awareness, and batched distributed FFTs, measured to 3072 GPUs

*Citation:* A. Ayala, S. Tomov, M. Stoyanov, A. Haidar, J. Dongarra, "Performance Analysis of Parallel FFT on Large Multi-GPU Systems", IPDPSW 2022, DOI 10.1109/IPDPSW55747.2022.00072, https://www.netlib.org/utk/people/JackDongarra/PAPERS/Performance_Analysis-fft-ipdps22.pdf

**Claim.** Full text extracted. For 512^3 c2c on Summit V100s: communication is >90% of runtime at 24 GPUs; P2P (Isend/Irecv) fastest below ~64 nodes, Alltoall(v) fastest above; GPU-aware P2P stops scaling past ~768 GPUs while Alltoall keeps scaling; disabling GPU-aware MPI adds ~30% communication cost at 16 nodes; MPI_Alltoallw (non-contiguous datatypes) loses badly on GPUs because it is unoptimized and not even GPU-aware in SpectrumMPI. Slab beats pencil below ~64 Summit nodes (predicted by their bandwidth model, verified experimentally). cuFFT on strided input shows large time spikes vs contiguous (~15us contiguous per batched-1D call), yet strided+Alltoallv still wins at >=64 nodes because packing costs dominate. Their batched distributed 3D FFT (64^3, batch across transforms) gives >2x speedup per transform vs unbatched, from communication/computation overlap; batching advantage vanishes at 512^3 where communication dominates. Careful grid tuning yields linear scaling to 3072 GPUs; LAMMPS KSPACE time cut ~40% switching fftMPI->tuned heFFTe at 32 nodes.

**Numbers.** 512^3 c2c: total ~0.09 s on 24 V100 (both P2P and A2A); comm >90% of runtime; GPU-aware on/off delta ~30% at 96 GPUs; batched 64^3: >2x per-transform speedup on both V100 (Summit) and MI100 (Spock, max 4 nodes allowed).

**Relevance to us.** Two direct hits for us: (1) batching across transforms to overlap the transpose is the distributed analogue of our batch-lane design and is documented to give >2x for small grids — LQCD-sized per-rank transforms are 'small' in this sense; (2) strided-vs-contiguous local FFT costs matter enough that libraries repack, which our SoA split-complex layout avoids by construction.

**Verification status.** VERIFIED (PDF text-extracted). heFFTe batched implementation is in the open heFFTe repo.

## 4. GPU-FFT on Selene (Verma et al.): the only open single-node 8xA100 3D-FFT table, and the single-node vs multi-node crossover

*Citation:* M. Verma, S. Chatterjee, G. Garg, B. Sharma et al., "Scalable Multi-node Fast Fourier Transform on GPUs", arXiv:2202.12756; SN Computer Science 4, 625 (2023), DOI 10.1007/s42979-023-02109-0; code https://github.com/Manthan-Verma/GPU_FFT

**Claim.** Full text extracted. Slab-decomposed CUDA+CUDA-aware-MPI FFT (R2C pipeline) on Selene (DGX A100-80GB, 600 GB/s NVSwitch, HDR IB). Communication fraction rises from ~0.51 (8 GPUs, 1 node) to 0.97-0.98 (largest counts); computation scales as p^-1.0 but communication exponent is only 0.11-0.75 depending on grid; scaling collapses when the all-to-all packet size drops to ~1 MB. DP time = 2.0x SP time at fixed config (e.g. 2048^3/128 GPUs: 38 ms SP vs 74 ms DP per forward+inverse pair) — i.e. A100 FFT is bandwidth-bound, not fp64-rate-bound. Conclusion verbatim: best performance is realized 'either within a single DGX box for a relatively small grid, or with many more GPUs for a large grid. The computation performance drops significantly when we go from one DGX box to two boxes.'

**Numbers.** Single node, 8x A100-80GB, forward+inverse pair (ms): cuFFTXt 512^3: 1.7 SP / 5.1 DP; 1024^3: 11 SP / 19 DP; 2048^3: 76 SP / 132 DP. Their GPU-FFT: 1024^3 17.6 SP / 32 DP; 2048^3 150 SP / 273 DP (their MPI version ~2x slower than cuFFTXt in-node). Aggregate best: 162 TFlop/s SP / 159 TFlop/s DP at 4096^3 on 256-512 GPUs; 8-GPU 1024^3 DP ~10 TFlop/s (their code) — ~13% of the node's fp64 peak. 128 A100s matched 196608 Haswell cores of Shaheen II (1536^3 CPU 42 ms vs 2048^3 GPU DP 74 ms).

**Relevance to us.** This is the published 8-GPU-single-node baseline we asked for: on our a100l/a100r nodes, a 1024^3 fp64 c2c 3D FFT should take roughly 9-10 ms via cuFFTXt (19 ms per forward+inverse pair, on 80GB parts — our 40GB SXM4 parts have 1.55 TB/s HBM so expect slightly slower). The one-node-to-two-nodes cliff means our 8xA100 node is a sweet spot where LOCAL kernel quality (the thing we know how to beat) still dominates — and where an L^3 x T lattice FFT for LQCD plausibly fits entirely in 320 GB of HBM.

**Verification status.** VERIFIED (PDF text-extracted). Open-source code exists (GPU_FFT on GitHub); cuFFTXt numbers are their measurements of the closed vendor library.

## 5. GESTS on Frontier (Yeung et al.): the largest published FFT-based production runs — 32768^3 on up to 8192 nodes

*Citation:* P.K. Yeung, K. Ravikumar, S. Nichols, R. Uma-Vaideswaran, "GPU-enabled extreme-scale turbulence simulations: Fourier pseudo-spectral algorithms at the exascale using OpenMP offloading", Computer Physics Communications 306 (2025) 109364, DOI 10.1016/j.cpc.2024.109364, OSTI 2498458

**Claim.** Full text extracted. Hand-rolled slab/pencil pseudo-spectral code: batched 1D FFT plans via the AMD ROCm library (hipFFT/rocFFT), pack/unpack on GPU, transpose via plain MPI_ALLTOALL called directly on device pointers (GPU-aware cray-mpich 8.1.23), 8 MPI ranks/node = 1 rank per MI250X GCD. SINGLE precision ('8 bytes per single-precision complex number'). Weak scaling of the full DNS step is essentially perfect from 4096^3 on 8 nodes to 32768^3 on 4096 nodes (96.6-104.6%), because the all-to-all time per step stays ~8.5 s while FFT compute grows only as log N. MPI all-to-all is 65-77% of the step everywhere beyond one node; at 2048^3 on 1 node (all intra-node Infinity Fabric) the step is 4.09 s with MPI only 1.85 s. Non-strided vs strided 1D FFT library calls differ up to ~7x at 32768-length transforms (their Table 1: 0.306 s vs 0.947 s class entries), so they restructure to feed contiguous batches.

**Numbers.** Wall time per full DNS step (3 variables, multiple FFT round-trips, fp32): 2048^3/1 node 4.09 s; 4096^3/8 nodes 10.11 s (MPI 7.76 s); 8192^3/64 nodes 10.75 s; 16384^3/512 nodes 11.53 s; 32768^3/4096 nodes 12.07-13.07 s (MPI ~8.5 s, FFT 2.7-3.4 s). Figure of merit ~2.91e12 grid points/s at 32768^3 on 4096 nodes, 6.6x their Summit-era result; 35 trillion grid points is the largest FFT-centric production resolution published.

**Relevance to us.** State of the art at the top end: even a hero exascale code uses vendor BATCHED-1D FFTs plus MPI_ALLTOALL — structurally identical to heFFTe and to our decomposition thesis, at 4096-node scale. Also a caution: their scaling works because per-node communication volume is held constant (weak scaling); LQCD-sized lattices (small N, fp64, complex) live at the opposite, latency/small-message end where their 32768^3 small-message troubles appear immediately.

**Verification status.** VERIFIED (PDF text-extracted). Code is a domain application, not a released library.

## 6. GROMACS on AMD platforms: the documented small-grid gap — heFFTe cannot do what cuFFTMp does

*Citation:* A. Alekseenko, S. Páll, E. Lindahl, "GROMACS on AMD GPU-Based HPC Platforms: Using SYCL for Performance and Portability", arXiv:2405.01420 (2024)

**Claim.** Fetched: 'GROMACS' scalability with PME decomposition has been demonstrated using the cuFFTMp library for NVIDIA GPUs, but no comparably scalable option exists for AMD GPUs'; heFFTe 'lacks the scalability for small 3D FFTs' — the efficiency of heFFTe for small grids is much lower than the CUDA path with cuFFTMp, drastically limiting PME strong scaling on Frontier/LUMI, where 'a system with PME electrostatics can hardly be scaled past 8 GCDs with only one PME rank'. Local single-GPU FFTs in GROMACS use VkFFT (a HIP fork with extra optimizations), not rocFFT.

**Numbers.** Qualitative in the fetched section; the operative measured fact is the scaling wall at ~8 GCDs for PME-decomposed runs on AMD.

**Relevance to us.** This is independent, application-side confirmation of the market gap our project's thesis points at: SMALL distributed FFTs (PME grids are ~10^2 per side, like LQCD L^3 x T local volumes) are where existing libraries fail, and the NVIDIA answer (cuFFTMp/NVSHMEM) has no AMD/portable equivalent. A fast batched-1D kernel plus a better small-message transpose is precisely the missing piece named here.

**Verification status.** VERIFIED (HTML fetched; quotes above are from the fetched text). GROMACS + VkFFT + heFFTe are all open source.

## 7. GPU-aware MPI vs NCCL/RCCL for all-to-all: neither dominates; message size decides

*Citation:* D. De Sensi, L. Pichetti, F. Vella et al., "Exploring GPU-to-GPU Communication: Insights into Supercomputer Interconnects", arXiv:2408.14090 (SC24)

**Claim.** Full text extracted. Measured on Alps (GH200), LUMI (MI250X), Leonardo (A100): for intra-node alltoall, NCCL/RCCL wins at large transfers (topology-aware pipelining MPI lacks) on Alps and LUMI, while on Leonardo MPI is slightly better; for small transfers GPU-aware MPI is up to 3x faster than RCCL on LUMI; for inter-node point-to-point, MPI outperforms *CCL by up to an order of magnitude on small transfers and up to 3x on large; RCCL's native alltoall showed no advantage over a trivial send/recv loop. Heavy env-var tuning (NCCL_IGNORE_CPU_AFFINITY, NCCL_NET_GDR_LEVEL, NCCL_NCHANNELS_PER_PEER) changed alltoall performance by 1.6-2x.

**Numbers.** Per-GCD alltoall goodput ceiling on LUMI ~600 Gb/s (Infinity Fabric topology); Leonardo GPU pair nominal 800 Gb/s; small-message regime: MPI up to 3x (intra-node, LUMI) and up to 10x (inter-node p2p) faster than *CCL.

**Relevance to us.** For an FFT transpose — many simultaneous medium/small messages — this says: on a single 8xA100 node use NCCL or NVSHMEM-style device-initiated copies for large slabs, but the small-message regime (small lattices, high GPU counts) belongs to GPU-aware MPI or custom kernels. It also explains why cuFFTMp bet on NVSHMEM: to escape exactly this MPI-vs-CCL tuning lottery. No paper was found that A/B-tests NVSHMEM vs NCCL vs MPI inside the SAME FFT — that comparison does not exist publicly.

**Verification status.** VERIFIED (PDF text-extracted). Benchmark code released by the authors.

## 8. FFT-ECP outcomes: heFFTe is the deliverable; portability achieved, vendor-gap on AMD/Intel remains

*Citation:* A. Ayala, S. Tomov, A. Haidar, J. Dongarra, "heFFTe: Highly Efficient FFT for Exascale", ICCS 2020, LNCS 12137, DOI 10.1007/978-3-030-50371-0_19; plus ICL-UT-22-02 (above) and ECP milestone reports (e.g. FFT-ECP ST-MS-10-1410, 2019)

**Claim.** The ECP FFT effort (FFT-ECP) delivered heFFTe: a portable distributed FFT with FFTW/MKL/cuFFT/rocFFT/oneMKL backends, arbitrary brick input grids, batched transforms, and MPI Alltoall(v)/P2P options. ICCS-2020 headline claims: >40x speedup of local kernels vs CPU-library local kernels and ~2x speedup of the whole FFT vs FFTE/FFTMPI-class libraries, linear scaling demonstrated on Summit (up to 24576 Power9 cores; later work ~6000 GPUs). The 2022 benchmark report (verified above) is the de-facto capability assessment: heFFTe was then the ONLY distributed FFT running on AMD and Intel GPUs. FFTX (SPIRAL-based code generation) was the other ECP FFT project; no large-scale distributed FFTX numbers comparable to heFFTe's were found.

**Numbers.** heFFTe vs FFTE: >2x overall on Summit at 1024^3 (ICCS 2020, 40x is local-kernel-only, CPU-library baseline); heFFTe setup/planning fastest among the nine benchmarked libraries (ICL-UT-22-02).

**Relevance to us.** heFFTe is the concrete integration target for our drop-in claim: its local kernel is literally a backend call into a vendor batched-1D library, and it already exposes the plumbing (grids, batching, MPI variants) around it. The ECP program's own conclusion — the network, not the local kernel, limits exascale FFT — bounds what a local-kernel replacement can claim at scale, and points our GPU-side value at 1-node and batched-small workloads.

**Verification status.** ICCS-2020 numbers ABSTRACT-ONLY (paper located, headline figures taken from abstract/secondary summaries); capability claims VERIFIED via ICL-UT-22-02 full text. heFFTe code open and maintained.

## 9. Intel/Aurora: no vendor distributed GPU FFT exists; local oneMKL batched FFT weak at large sizes

*Citation:* I. Sfiligoi, J. Candy, E.A. Belli, "Evaluation of Intel Max GPUs for CGYRO-based fusion simulations", arXiv:2410.05510; Intel oneMKL docs (cluster FFT); Aurora architecture paper arXiv:2509.08207

**Claim.** oneMKL's 'cluster FFT' is a CPU/MPI feature; searches found no Intel distributed GPU FFT product — on Aurora, distributed FFT means heFFTe with the oneMKL single-device backend (heFFTe was the only library supporting Intel GPUs per ICL-UT-22-02). CGYRO paper (full text extracted): porting batched 2D FFTs to oneMKL on Max 1550 was near-trivial (cuFFT-like plan_many semantics, reversed rank order, omp dispatch), and Max 1550 matches A100/MI250X for small/medium batched FFTs but is 'significantly slower at larger FFT sizes' — FFT being CGYRO's dominant cost.

**Numbers.** CGYRO batched 2D FFT sizes 288x96..2016x576, batches 2048-18432, across Stampede3 (4x Max 1550/node), Perlmutter (4x A100), Frontier (4x MI250X): Intel comparable at small sizes, noticeably slower at large; exact ratios in their Appendix B tables (relative-performance figures, not extracted numerically).

**Relevance to us.** Confirms the pattern on the third vendor: the distributed layer is community software (heFFTe) wrapping a vendor batched-1D/2D local call. Also a reminder that per-rank batched FFT sizes in real applications (CGYRO: 10^2-10^3 per side, batch 10^3-10^4) sit exactly in our campaign's batched regime, not at hero 4096^3 sizes.

**Verification status.** CGYRO VERIFIED (PDF text-extracted); 'no Intel distributed GPU FFT' is a negative finding from search, marked as such — no Aurora distributed-FFT scaling publication was found at all.

## 10. fp64 rate and GPU FFT: bandwidth-bound on 1/2-rate datacenter parts; our 1/32-rate 2080 Ti sits at the compute/bandwidth crossover

*Citation:* Measured basis: Verma et al. arXiv:2202.12756 (SP-vs-DP timings on A100); NVIDIA cuFFTMp blog (Summit FP32 50 TF vs FP64 40 TF at identical config); roofline arithmetic ours

**Claim.** On A100 (fp64 = 1/2 fp32 rate), DP 3D FFT time is measured at almost exactly 2x SP at fixed grid/GPU count (74 vs 38 ms at 2048^3/128 GPUs) — i.e. cost scales with bytes moved, not flops: FFT is bandwidth-bound there (~11-13% of compute peak either precision). Roofline for a radix-8/16 Stockham pass gives arithmetic intensity ~0.5-0.6 fp64 flop/byte; at 2080 Ti's 616 GB/s that demands ~290-385 GFlop/s against an fp64 peak of only ~420 GFlop/s (13.45 TF fp32 / 32). So on the 2080 Ti fp64 FFT transitions from bandwidth-bound to fp64-unit-bound: expect roughly 2-4x worse than the bandwidth-only expectation, not 32x — but with zero headroom, and any non-fused twiddle arithmetic shows up directly. In the DISTRIBUTED setting this matters even less: with communication at 76-95% of runtime, local fp64 throughput is a second-order effect (ICL-UT-22-02 conclusion).

**Numbers.** Measured: A100 DP/SP time ratio 2.0 (Selene, 2048^3, 128 GPUs); Summit V100 (1/2-rate fp64) cuFFTMp: 40+ TF DP vs 50+ TF SP at 1536 GPUs (ratio 1.25 — communication-dominated). Projected (OURS, UNVERIFIED): 2080 Ti fp64 1024^3-class cuFFT at roughly 200-350 GFlop/s/GPU vs ~700+ GFlop/s if it were purely bandwidth-bound.

**Relevance to us.** For our cluster: the 22 2080 Ti nodes are usable for fp64 FFT correctness and even for communication-pattern studies (the transpose doesn't care about fp64 rate), but any fp64 kernel-throughput number from them is compute-throttled and must not be extrapolated to A100. Recommend one calibration run: cuFFT fp64 batched-1D on a 2080 Ti vs A100 to pin the actual ratio.

**Verification status.** A100/V100 SP-vs-DP ratios VERIFIED (measured, sources above). No published 2080 Ti (or any 1/32-rate GeForce) fp64 FFT benchmark was found anywhere — VkFFT's published fp64 curves cover A100/MI250 only. The 2080 Ti statement is our roofline projection, directly measurable on our prod nodes in minutes.

## 11. Distributed NUFFT (2026) — current best practice still wraps heFFTe, and FFT+halo is 60-80% of runtime at a few hundred GPUs

*Citation:* "A Performance-Portable, Massively Parallel Distributed Nonuniform FFT", arXiv:2605.10678 (2026)

**Claim.** Benchmarked on Alps (GH200), JUWELS Booster (A100), LUMI (MI250X); uses heFFTe internally for the uniform-FFT stage; MPI only (no NCCL/RCCL evaluated). At 256 GPUs on Alps the distributed FFT is 61% of Type-1 NUFFT runtime with halo exchange another 22%; parallel efficiency 60% (Alps, 32->256 GPUs), 80% (JUWELS 16->256), 91.9% at 1024^3 from 256->1024 GPUs; LUMI 72.4% (16->512) but hurt by compiler register spilling in kernels. Pruned-FFT variant wins on Slingshot (Alps) and loses 2.4x on InfiniBand (JUWELS) due to concurrent sub-communicator traffic.

**Numbers.** 512^3 grid on 256 GH200 GPUs: full distributed FFT path 44-48 ms per NUFFT; 1024 GPUs at 1024^3 sustains 91.9% relative efficiency.

**Relevance to us.** Shows the 2026-era pattern unchanged: new distributed-FFT-adjacent codes still delegate to heFFTe + vendor batched-1D locals, and communication share grows monotonically with GPU count. Also a concrete data point that network TYPE (Slingshot vs InfiniBand — ours) flips which transpose strategy wins, so any distributed claim we make must be measured on our IB fabric.

**Verification status.** VERIFIED (HTML fetched; numbers from the fetched summary of the paper's tables).

## 12. Largest published distributed FFT runs and what limited them

*Citation:* Aggregation: NVIDIA cuFFTMp blog (Selene); Yeung et al. CPC 306 (2025) 109364 (Frontier); Ayala MUG'23 slides https://mug.mvapich.cse.ohio-state.edu/static/media/mug/presentations/23/MUG23TuesdayAlanAyala.pdf (HACC); Chatterjee et al. JPDC 113 (2018) (Shaheen II CPU)

**Claim.** Largest documented: (1) cuFFTMp 16384^3 c2c fp32, 4096 A100s, Selene — 1.8 PFlop/s, ~70% of a bandwidth-limited peak; limit: inter-node bandwidth. (2) GESTS/Frontier 32768^3 (35T points, fp32 pseudo-spectral, forward+inverse FFTs every step), 4096-8192 nodes (32768-65536 GCDs); limit: MPI_ALLTOALL time (~8.5 s/step, 65-77%) and small-message effects at 8192 nodes. (3) HACC cosmology FFTs >20000^3 with SWFFT scaling past 1.5M MPI ranks (CPU-era, slide claim). (4) CPU reference: FFTK 3072^3 on 196608 Cray XC40 cores, 179 ms/pair fp64. Common failure mode across all: all-to-all bandwidth first, then message-count latency at extreme rank counts ('at very large process counts (e.g., millions), it is the associated latency of MPI_Alltoall(v) that will produce scaling failures' — ICL-UT-22-02).

**Numbers.** See per-item; the fp64 record-class number remains cuFFTMp's 40+ TFlop/s at 4096^3 on 1536 V100s and GPU-FFT's 159 TFlop/s DP at 4096^3 on 512 A100s (Selene).

**Relevance to us.** Calibrates ambition: nobody wins the top end on local-kernel speed — records are set by communication engineering (NVSHMEM, device-resident alltoall, constant per-node volume). Our differentiation must therefore be stated for the regimes records ignore: fp64, complex, small-to-mid lattices, batched, single node or few nodes — the LQCD case.

**Verification status.** Selene/Frontier/Shaheen items VERIFIED (primary sources fetched); HACC >20000^3 / 1.5M-rank figure ABSTRACT-ONLY (MUG'23 slide citing Mniszewski et al. 2021, not independently checked).

## Negative space (searched for, not found)

Searched for but NOT found: (1) any published fp64 FFT benchmark on a 1/32-rate GeForce part (2080 Ti or similar) — VkFFT's public fp64 curves cover A100/MI250 only, TurboFFT/tcFFT papers target A100/T4; our 2080 Ti statements are roofline projections and should be replaced by a 10-minute measurement on a prod node. (2) Any controlled head-to-head heFFTe-vs-cuFFTMp benchmark paper with numbers on the same machine — the only public signals are NVIDIA's own blog and the GROMACS team's qualitative verdict that heFFTe is much less efficient for small grids; ICL folks never published a cuFFTMp comparison (Ayala moved to AMD in 2023). (3) Any NVSHMEM-vs-NCCL-vs-GPU-aware-MPI A/B test inside the same FFT code — the communication comparisons that exist (De Sensi SC24) are raw collectives, not FFT transposes. (4) Any AMD distributed FFT product (no 'rocFFTMp' exists; Frontier records use hand-rolled MPI_ALLTOALL or heFFTe+rocFFT), and any Intel distributed GPU FFT or Aurora distributed-FFT scaling publication. (5) Published 8x-A100-40GB (our exact SKU) single-node 3D FFT numbers — the closest is Verma et al.'s 8x A100-80GB DGX table (cuFFTXt 19 ms fp64 1024^3 pair); 40GB HBM is ~20% slower on paper. (6) Any published LQCD-specific distributed-FFT benchmark (4D L^3 x T complex transforms) on GPUs — the field's FFT usage (e.g. gauge fixing, momentum projections) appears nowhere in the exascale FFT literature, which is a genuine open lane. (7) cuFFTMp fp64 weak-scaling data on A100-class hardware (the fp64 numbers NVIDIA published are Summit/V100); and the grid size for the blog's 78 ms/8-GPU strong-scaling point is not stated in the text we could extract.

## Bottom line

Every serious distributed GPU FFT — cuFFTMp, heFFTe, GESTS's Frontier record code — is vendor batched-1D local kernels plus an all-to-all transpose, and beyond one node the transpose is 65-95% of runtime, so local-kernel speed (our strength) only pays off single-node, batched, or at small grids; precisely there the published record shows a real gap (GROMACS: heFFTe collapses on small grids, no portable cuFFTMp equivalent exists), our 8xA100 nodes have a concrete published baseline to beat (cuFFTXt: 19 ms fp64 1024^3 forward+inverse pair on 8x A100-80GB), fp64 FFT on A100 is purely bandwidth-bound (DP = 2.0x SP measured) while on our 1/32-rate 2080 Ti it is compute-crossover territory with zero published measurements, and the largest runs ever published (cuFFTMp 16384^3 fp32/4096 A100s at 1.8 PFlop/s; GESTS 32768^3 fp32 on 4096-8192 Frontier nodes) were all limited by network bandwidth then small-message latency — never by the FFT kernels themselves.
