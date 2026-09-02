# heFFTe internals: architecture, the local-kernel/backend contract, communication layer, measured performance

**Scope as the agent recorded it:** heFFTe in depth: architecture, backend/local-kernel interface, communication layer, and measured performance (sources: heFFTe 2.4.1 source at commit 4d8d4597 2025-11-19, cloned to /tmp/heffte_inspect; ICCS 2020 paper; IPDPS 2022 workshop paper; ICL-UT-22-07 report — extracted texts at /tmp/heffte_iccs2020.txt, /tmp/heffte_ipdps22.txt, /tmp/heffte_commcost.txt)

## 1. Architecture: brick -> pencil pipeline of at most 4 reshapes + 3 local batched-1D stages

*Citation:* Ayala, Tomov, Haidar, Dongarra, "heFFTe: Highly Efficient FFT for Exascale", ICCS 2020, DOI 10.1007/978-3-030-50371-0_19; source: /tmp/heffte_inspect/src/heffte_plan_logic.cpp, src/heffte_compute_transform.cpp

**Claim.** A distributed 3D FFT is Algorithm 1 of the ICCS paper: input brick -> reshape to x-pencils -> 1D FFTs -> reshape to y-pencils -> 1D FFTs -> reshape to z-pencils -> 1D FFTs -> reshape to output brick. In code this is exactly 'std::array<std::unique_ptr<reshape3d_base>,4> shaper' and 'std::array<executor_base*,3> executor' applied alternately in compute_transform(); reshapes are skipped when input/output are already pencils/slabs. Slab decomposition is auto-detected (plan_slab_reshapes): one 2D-FFT executor + one 1D executor, saving a reshape — the paper and IPDPS22 both state slabs win at small node counts. plan_options: use_pencils (default true), use_reorder (reshape also transposes so the transform dimension becomes contiguous; default true for fftw/stock, FALSE for mkl/cufft which handle strided batches natively), reshape_algorithm, use_gpu_aware, use_subcomm (shrink the FFT to fewer ranks).

**Numbers.** Message-count scaling quoted in ICCS 2020: brick->pencil O(P^{1/3}) messages, pencil->pencil O(P^{1/2}), brick->slab O(P^{2/3}), slab->pencil O(P). IPDPS22 (Summit, 512^3): slabs faster than pencils below 64 nodes, verified experimentally.

**Relevance to us.** Confirms our standing premise precisely: the compute in heFFTe is nothing but three batched-1D passes (or one 2D+one 1D for slabs) between transposes. A faster local batched-1D kernel drops into exactly one architectural slot with no change to the communication machinery.

**Verification status.** VERIFIED (paper text extracted in full + source code read); performant code exists: github.com/icl-utk-edu/heffte, BSD-3, v2.4.1

## 2. The local kernel IS a vendor batched-1D call, one per pencil box

*Citation:* heFFTe 2.4.1 source: include/heffte_backend_fftw.h (lines 102-343), heffte_backend_mkl.h, heffte_backend_cuda.h, heffte_backend_rocm.h, heffte_geometry.h (fft1d_get_howmany/fft1d_get_stride)

**Claim.** Per backend, the executor wraps exactly: FFTW 'fftw_plan_many_dft(1, &size, howmanyffts, ..., stride, dist, ...)'; MKL DFTI with 'DftiSetValue(plan, DFTI_NUMBER_OF_TRANSFORMS, howmanyffts)' + DFTI_INPUT_DISTANCE; cuFFT 'cufftMakePlanMany(..., batch, ...)'; rocFFT 'rocfft_plan_create(..., 1, &size, batch, desc)'; oneMKL descriptors. Batch geometry from the box: transform along the contiguous dimension -> stride=1, dist=size, howmany=osize(1)*osize(2) (one fully contiguous batched-1D call over the whole pencil box); along the middle dimension -> a loop over 'blocks' (one strided plan_many per slice); along the slow dimension -> stride=osize(0)*osize(1), dist=1. Execution is in-place on interleaved std::complex data: 'fftwf_execute_dft(*cforward, block_data, block_data)'.

**Numbers.** For a per-rank pencil box of a 1024^3 FFT on Q0=Q1=32 grid (1024 ranks): local call is a batched 1D of length 1024 x howmany=1024 per rank per pass — squarely in our measured batched-1D regime (L=1024, large batch).

**Relevance to us.** This is the drop-in surface: our batch-lane fp64 kernel replaces a plan_many call with L in {app sizes we already benchmark} and batch = pencil cross-section. Caveat: heFFTe hands the executor interleaved complex, in-place, sometimes strided (when use_reorder=false); our split-complex SoA kernel needs an interleave<->SoA conversion (or shuffle-based strided loads) inside the executor, whose cost must be counted in any claimed speedup.

**Verification status.** VERIFIED (read directly from source)

## 3. Backend registration: header-only template specializations; the 'stock' SIMD backend is the exact precedent for plugging in our kernel

*Citation:* heFFTe 2.4.1 source: include/heffte_common.h (executor_base lines 561-595, one_dim_backend line 663, make_executor 669-708, has_executor2d/3d 714+), include/heffte_backend_stock.h, include/stock_fft/

**Claim.** fft3d<backend_tag> is a header-only class template; a backend is a set of specializations, all demonstrated by the bundled 'stock' backend (heFFTe's own from-scratch SIMD FFT with AVX and AVX-512 paths under Heffte_ENABLE_AVX512 — proof a hand-written CPU kernel backend is a supported pattern): (1) tag struct in namespace heffte::backend; (2) backend::is_enabled<tag> : true_type; (3) buffer_traits<tag> (CPU default provided: location=tag::cpu, container=std::vector); (4) backend::name<tag>(); (5) one_dim_backend<tag>{ using executor = ...; using executor_r2c = ...; }; (6) default_plan_options<tag>{ static const bool use_reorder; }. The executor subclasses executor_base and overrides 'virtual void forward(std::complex<double>[], std::complex<double>*) const' / backward (in-place, second arg is scratch workspace), plus box_size() and workspace_size(); constructors take (stream, box3d, dimension) for 1D, and optionally (stream, box, dir1, dir2) for fused-2D and (stream, box) for 3D. Gotchas: has_executor2d()/has_executor3d() are hardcoded constexpr exclusion lists (stock is excluded), so a custom tag added OUTSIDE the heFFTe tree is presumed to have 2D/3D constructors — either provide trivial ones or set use_pencils=true and note plan_logic may still request slabs; setting use_reorder=true guarantees the executor always sees contiguous batched data (stride 1, dist=size), which is the friendliest contract for our SoA kernel.

**Numbers.** The stock executor shows the minimal footprint: ~100 lines of adapter around the actual kernel. No dynamic plugin/registry exists — it is compile-time template specialization only, undocumented as an extension API but requiring no modification of heFFTe sources except (optionally) the two constexpr trait functions.

**Relevance to us.** Our kernel can be a heFFTe backend with a one-header adapter: subclass executor_base, convert interleaved->SoA into our aligned scratch (workspace_size() sized), run the batch-lane FFT, convert back. use_reorder=true gives us contiguous batch-major input, which matches our across-batch vectorization layout after a transpose-in-register step.

**Verification status.** VERIFIED (source); performant reference implementation of the pattern (stock AVX512) ships in-tree

## 4. Communication layer: pure MPI (4 algorithms), pack/unpack kernels, optional GPU-aware; no NCCL/NVSHMEM; overlap only via pipelined p2p and batching

*Citation:* heFFTe 2.4.1 source: include/heffte_plan_logic.h (reshape_algorithm enum), include/heffte_reshape3d.h, src/heffte_reshape3d.cpp; Ayala et al. ExaMPI 2019 'Impacts of Multi-GPU MPI Collective Communications on Large FFT Computation'

**Claim.** reshape_algorithm in {alltoallv (default; 'larger FFT, many ranks'), alltoall ('smaller FFT, many ranks'), p2p_plined ('larger FFT, fewer ranks'), p2p} — comments in the header state that expected ranking verbatim. alltoall/alltoallv run MPI_Alltoall(v) on a subcommunicator of only the ranks that intersect the reshape; p2p posts all MPI_Irecv up front, then per-destination pack->MPI_Isend (p2p_plined) or pack->blocking MPI_Send (p2p), then an MPI_Waitany loop that unpacks each message as it lands (receive/unpack pipelining — the only intra-transform overlap). Packing is direct_packer or transpose_packer (fuses the reorder into unpack); self-to-self overlap bypasses MPI. GPU path: use_gpu_aware=true passes device pointers to MPI, false stages through pinned CPU buffers (gpu::transfer::unload/load). grep confirms zero occurrences of NCCL or NVSHMEM anywhere in the tree; MPI is the only transport. There is NO overlap of communication with the local FFT within a single transform — shaper and executor calls strictly alternate in compute_transform; overlap across transforms exists only via the batch API.

**Numbers.** ICCS 2020: packing/unpacking <10% of reshape time; heFFTe_alltoallv (multi-rail, async pack+comm) effective only up to ~32 Summit nodes. IPDPS22: disabling GPU-aware MPI costs ~30% more communication time at 16 Summit nodes; MPI_Alltoallw (non-contiguous datatypes, Dalcin-style) rejected — far less optimized in real MPI libraries, and packing it avoids is only ~10% of runtime.

**Relevance to us.** For a pencil-decomposed LQCD 4D transform on our IB cluster, heFFTe brings tuned alltoall-subcomm machinery we would otherwise rewrite; but since local FFT and comm never overlap within a transform, a 3.5x faster local kernel shrinks the compute slice only — end-to-end gain is bounded by the local-FFT fraction (see performance entries). The batched API is where local-kernel speed and comm overlap compound.

**Verification status.** VERIFIED (source read; the four apply() paths inspected)

## 5. Batched distributed transforms: whole batch per MPI message, executor looped per batch element

*Citation:* heFFTe 2.4.1 source: include/heffte_fft3d.h (forward(int batch_size,...) lines 388-413), src/heffte_compute_transform.cpp (apply_fft lambda), src/heffte_reshape3d.cpp (batch_size*send_size message sizes); IPDPS22 paper Fig. 13

**Claim.** fft3d::forward(batch_size, input, output, workspace) transforms batch_size independent 3D arrays in one call: every reshape sends 'batch_size * send_size[i]' elements in a single message (packing loops j over the batch), amortizing latency; the local executor however is invoked in a plain loop 'for j < batch_size: executor[i]->forward(data + j*box_size(), workspace)' — the batch is NOT fused into one bigger vendor plan_many call. IPDPS22: 'These speedups come from the overlap of communication and computation. The more transforms per MPI unit generates more overlap with network exchanges.'

**Numbers.** IPDPS22 Fig. 13: batched 3D FFT of size 64^3 on NVIDIA (Summit) and AMD (Spock, 4 nodes max allowed) GPUs: >2x speedup per transform vs unbatched. Benefit 'considerably reduced' at 512^3 where communication dominates regardless.

**Relevance to us.** Two openings for us: (a) the executor loop over batch elements is exactly where a batch-lane kernel that fuses the batch dimension into SIMD lanes beats a looped vendor call — heFFTe leaves that fusion on the table; (b) LQCD workloads (many propagator components = natural batch) fit the batched API, the regime where heFFTe itself says the wins live.

**Verification status.** VERIFIED (source + paper)

## 6. ICCS 2020 measured performance on Summit: linear scaling to 1024 nodes, GPU local kernels 40-42x over CPU, whole FFT ~2x, ~90% of a bandwidth-roofline

*Citation:* Ayala, Tomov, Haidar, Dongarra, ICCS 2020, DOI 10.1007/978-3-030-50371-0_19 (full text extracted from the camera-ready PDF)

**Claim.** Strong scaling of a 1024^3 complex-to-complex fp64 3D FFT on Summit, 1 to 1024 nodes (up to 24,576 Power9 cores / 6,144 V100s): heFFTe GPU ~2x faster than heFFTe CPU (itself built on improved fftMPI/SWFFT kernels, so also ~2x over those); CPU curve stops scaling past 512 nodes (12,288 cores) from latency (messages too small); >2x over FFTE in both strong and weak scaling. Fig. 6 profile (1024^3, 32 nodes, all-to-all): CPU version spends 0.08s packing, 0.10s unpacking, 0.15s(+0.4s/0.36s across phases) with FFT compute dropping to 0.008s per phase on GPUs — 'Local kernels accelerated 42x using GPUs, total speedup 2x', i.e. after GPU acceleration the runtime is overwhelmingly MPI. Roofline model Psi_Summit = 1.953*P*log(N) GFlops (B=25GB/s, r=4 reshapes, 16-byte elements); measured GPU performance reaches ~90% of that peak (Fig. 8, 4-256 nodes). Weak scaling 64^3*...*8192^3 up to 1024 nodes: ~linear, GPU >2x CPU throughout (Fig. 5 tops out around 25-30 TFlop/s at 1024 nodes). LAMMPS Rhodopsin (128^3 grid, 2 nodes): KSPACE 2x faster, application 1.5x.

**Numbers.** All figures from Summit (2xPower9 + 6xV100/node, EDR IB dual-rail ~23.5 GB/s practical inter-node). 1024^3 fp64 c2c ~ 161 GFlop per transform.

**Relevance to us.** The 42x local / 2x total arithmetic is the cautionary tale for our drop-in thesis at scale: once the local kernel is fast, the transform is bandwidth/latency-bound and further local speedup is invisible. Our 3.5x CPU win over FFTW/MKL matters end-to-end only where compute is a large fraction: few nodes, CPU-only clusters (our axxxl nodes), or fused FFT+map chains.

**Verification status.** VERIFIED (full text extracted; numbers partly read from figure captions/labels — the 0.08/0.10/0.15/0.4/0.36/0.008s breakdown is from Fig. 6 label text whose exact phase mapping is ambiguous in extraction)

## 7. IPDPS22 measurements: communication >90% of runtime at just 24 GPUs; cuFFT local call ~15us; all-to-all wins at >=64 nodes; linear scaling case study to 3072 GPUs

*Citation:* Ayala, Tomov, Stoyanov, Haidar, Dongarra, "Performance Analysis of Parallel FFT on Large Multi-GPU Systems", IPDPSW 2022 (HCW/AsHES), netlib.org/utk/people/JackDongarra/PAPERS/Performance_Analysis-fft-ipdps22.pdf

**Claim.** 512^3 c2c on Summit, 24 V100 (4 nodes): 'communication for this problem over 90% of runtime'; packing/unpacking <10%; total ~0.09s for the 3D FFT with either p2p or all-to-all at that size. Per-call cuFFT time inside the 3D pipeline ~15us for a batch of length-512 1D FFTs (contiguous); a visible spike when input is strided — 'this also happens when using FFTW and rocFFT' — yet the strided+Alltoallv variant is still fastest at >=64 nodes because avoiding it costs more in packing. GPU-aware MPI off -> +~30% communication at 16 nodes; GPU-aware p2p 'fails' (stops scaling) at large node counts while all-to-all keeps scaling to 768+ GPUs. Tuned grid sequence given for 6..3072 GPUs (Table III); 'a careful tuning of the algorithm yields to linear scalability... case study using 3072 GPUs'. heFFTe and AccFFT 'have shown good linear scaling for large number of GPUs (~6,000)'. LAMMPS 512^3 on 32 nodes: KSPACE time -40% switching fftMPI->heFFTe.

**Numbers.** Machine: Summit (6 V100/node, NVLink 50GB/s CPU-GPU, ~23.5 GB/s practical inter-node IB). Model used for slab-vs-pencil choice: B=23.5 GB/s, latency 1us; slabs predicted and measured faster below 64 nodes for 512^3.

**Relevance to us.** Directly calibrates the drop-in value: at 4 nodes the local FFT is already <10% of runtime on GPUs, so on our A100 nodes a faster local kernel buys little for big distributed transforms; the leverage is single-node/few-node runs and the strided-input inefficiency they document (our kernels could eat the transpose via batch-lane layout, removing the pack-vs-strided dilemma they measured).

**Verification status.** VERIFIED (full text extracted with readable digits)

## 8. ICL-UT-22-07 Vampir traces: MPI is 92.6-95.1% of runtime for 1024^3 on 16 V100s; CPU-vs-GPU local balance quantified

*Citation:* Ayala, Tomov, Luszczek, Cayrols, Ragghianti, Dongarra, "Analysis of the Communication and Computation Cost of FFT Libraries towards Exascale", ICL Tech Report ICL-UT-22-07, July 2022, icl.utk.edu/files/publications/2022/icl-utk-1558-2022.pdf

**Claim.** Back-to-back 3D FFTs (4 fwd + 4 bwd), grids (1,4,4)-(4,1,4)-(4,4,1) = 16 MPI ranks, heFFTe+cuFFT on Summit V100s: pipelined Isend/Irecv variant — MPI_Waitall 92.57%, backward 1D FFT 2.94%, forward 1D FFT 2.22%, 0.298 s/transform, 540.04 GFlops/s, 5120MB/rank, max error 4.57e-15; MPI_Alltoall variant — 95.13% in MPI_Alltoall, 1.8-2.2% in the 1D FFTs, 0.514 s/transform, 313.38 GFlops/s. Same experiment with heFFTe+FFTW on Power9 CPUs (MPI_Alltoall): 3.62 s/transform, 44.47 GFlops/s, 499MB/rank. Report further states tuned grids ensure 'linear scaling to over forty thousand processes', and predicts that at very large process counts (millions) MPI_Alltoall(v) LATENCY (small-message exchanges) is what breaks scaling — naming all-to-all optimization for small volumes as the critical exascale need.

**Numbers.** The 540.04 GFlops/s / 0.298s and 44.47 GFlops/s / 3.62s pairs are self-consistent with N=1024^3 (5N log2 N = 161 GFlop), confirming the transform size despite the report's digits being unextractable in body text (embedded fonts without ToUnicode maps; figure-label text survived). GPU:CPU local-throughput ratio at 16 ranks: 540/44 ~ 12x whole-transform.

**Relevance to us.** Hard numbers for the ceiling: on GPUs the local 1D FFT is ~5% of a distributed transform even at only 16 ranks, so a local-kernel win must target the CPU regime (where local compute is 10x more of the budget) or single-node work. Also note their p2p-pipelined beats alltoall by 1.7x at this scale — algorithm choice dwarfs local-kernel choice here.

**Verification status.** VERIFIED for the quoted trace percentages and GFlops (verbatim from extracted figure text); ABSTRACT-ONLY for the report's other tables (digits lost to font encoding — the companion ICL-UT-22-02 benchmark report has the same defect and its numbers could not be extracted at all)

## 9. The C++ API surface our integration would live behind (fft3d class, C/Fortran/Python wrappers)

*Citation:* heFFTe 2.4.1 source: include/heffte_fft3d.h, include/heffte_c.h, fortran/, python/; doxygen at icl-utk-edu.github.io/heffte

**Claim.** User-facing contract: construct 'heffte::fft3d<backend_tag>(inbox, outbox, comm, options)' from two box3d (low/high corner + optional internal order) and an MPI communicator; then 'fft.forward(input, output, scale::none|full|symmetric)' with std::complex<float/double> (also float/double r2c via fft3d_r2c, and cos/sin r2r variants). Batched overload takes leading int batch_size. The plan is reusable; workspace can be user-provided to avoid allocation. C API (heffte_c.h), Fortran and Python wrappers exist. Everything is templated on the backend tag, so switching vendor kernel = changing one template parameter.

**Numbers.** Not a performance claim.

**Relevance to us.** If we ship our kernel as a heffte backend tag, every existing heFFTe user (and the C/Fortran/Python bindings) can A/B it against fftw/mkl by changing one type — the cleanest possible demonstration vehicle for the 1D campaign results, and the benchmark harness (benchmarks/speed3d_c2c in-tree) gives us their own measurement rig.

**Verification status.** VERIFIED (source)

## Negative space (searched for, not found)

Searched for but NOT found: (1) any NCCL, NVSHMEM, or non-MPI transport in heFFTe 2.4.1 — grep over the whole tree returns nothing; claims elsewhere that heFFTe uses NCCL are false as of this version. (2) Any communication/computation overlap within a single transform — compute_transform strictly alternates reshape and FFT; only the p2p receive/unpack pipeline and the batched API overlap anything. (3) A documented mechanism for registering an external backend — the pattern is clear from backend_stock.h but is not documented as a public extension API; has_executor2d/has_executor3d are hardcoded exclusion lists that an out-of-tree backend cannot extend without providing 2D/3D constructors. (4) Frontier/Crusher/Perlmutter production-scale heFFTe numbers: the two ICL 2022 reports (ICL-UT-22-02 benchmark report, ICL-UT-22-07 cost analysis) embed all digits in fonts without ToUnicode maps, and their figures are raster images — body-text numbers are unrecoverable by text extraction; only figure-label text (the Vampir trace percentages quoted above) survived. IPDPS22 used Spock (4xMI100/node, max 4 nodes permitted) as the only AMD data point. I did not locate a journal (IJHPCA-style) superset of the ICCS paper with Frontier measurements. (5) Any split-complex/SoA data-layout option for the local executor — the contract is interleaved std::complex, in-place; our kernel needs a layout conversion whose cost is unmeasured. (6) 4D transform support — fft3d/fft2d (+batch) only; an L^3xT LQCD transform would be batch-of-3D plus a separate 1D pass in T, composed by the caller. (7) Fraction-of-time-in-communication for CPU-only clusters at our scale (2 nodes): not reported anywhere found; their CPU trace (44 GFlops/s at ~168 cores) suggests local compute is a much larger fraction on CPUs, but no percentage breakdown was extractable. Local artifacts kept for follow-up: /tmp/heffte_inspect (source clone, on wallaby /tmp — invisible to compute nodes, delete when done), /tmp/heffte_iccs2020.txt, /tmp/heffte_ipdps22.txt, /tmp/heffte_commcost.txt, /tmp/heffte_bench2022.txt.

## Bottom line

Verified from source: heFFTe's local kernel is exactly a vendor batched-1D call (fftw_plan_many_dft / MKL DFTI NUMBER_OF_TRANSFORMS / cufftMakePlanMany) on each pencil box, and a new backend is a ~100-line header-only template specialization (the in-tree AVX-512 'stock' backend is the precedent) — our batched-1D fp64 kernel can be plugged in, with one real cost (interleaved-complex <-> SoA conversion at the executor boundary) and one hard ceiling: their own measurements put the local 1D FFT at ~2-5% of distributed runtime on GPUs (MPI 92-95%) and communication >90% already at 4 Summit nodes, so the drop-in win is real on CPU nodes, single/few-node runs, and batched/chained workloads, and negligible for large multi-node GPU transforms.
