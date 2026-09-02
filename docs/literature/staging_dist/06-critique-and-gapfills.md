# Completeness critique and the three gap-fill passes

A sixth agent read all five veins and named the gaps that mattered for the decision;
three further agents were dispatched to close them. Their findings CORRECT the veins
above in two places, noted inline.

## Critic's assessment

The survey is strong where it matters most: the heFFTe local-kernel contract is read from source, the comm-dominance numbers are primary-sourced, and the negative-space sections are honest. Three things remain thin for our decision: (1) the CPU/few-node regime — the ONLY regime where our 3.5x kernel is claimed to pay off end-to-end — has zero verified communication-fraction numbers (all quantified profiles are Summit GPUs at >=4 nodes; the ICL-UT-22-02 CPU digits were lost to font encoding and the oft-cited 'comm ~50% at 512 cores' remains unverified); (2) cuFFTMp, the incumbent on our a100l/a100r partitions, has no independent evaluation in the survey — only NVIDIA's blog and one unfetched preprint snippet — and that same preprint (DaggerFFT, 2026) is itself a distributed-FFT library the landscape vein missed entirely; (3) the load-bearing plug-in claim 'set use_reorder=true and our executor always sees contiguous batched data (stride 1, dist=size)' rests on a header comment plus an inference, and the failure mode for an out-of-tree backend when plan_logic requests slabs (compile error? silent 2D executor instantiation?) was flagged but never resolved — this is the difference between a ~100-line adapter and a strided-batch kernel variant we don't have. The interleaved<->SoA conversion cost and the 2080 Ti/A100-40GB fp64 baselines are also missing but are local 10-minute measurements, not literature gaps, and the survey correctly says so.

## Gaps named

### Gap 1

**The gap.** No verified communication-vs-compute fraction for CPU-only clusters at small node counts (1-2 nodes, our axxxl/Ice Lake regime). Every quantified profile in all five veins is Summit GPUs at >=4 nodes (MPI 90-95%); the CPU-side numbers in ICL-UT-22-02 and the quoted 'communication ~50% at 512 cores' were lost to the PDF's broken font encoding and remain ABSTRACT-ONLY/UNVERIFIED. The 44.5 GFlop/s CPU trace hints local compute dominates on CPUs but no percentage was extracted.

**Why it matters.** This fraction is the Amdahl denominator for our entire drop-in pitch: the bottom lines claim the 3.5x local-kernel win 'is real on CPU nodes and 1-8 node runs' without a single verified number for how much of a 2-node CPU heFFTe transform is local FFT. If comm is already 60% at 2 CPU nodes, the end-to-end win caps at ~1.3x, not 3.5x.

**How to close it.** Re-fetch ICL-UT-22-07 and ICL-UT-22-02 PDFs and render the relevant pages to images (pdftoppm -png, then read the figures/tables visually) — the survey itself notes figure-label text survived and 'the figures could be recovered by rendering pages to images'. The CPU Vampir-trace figure in ICL-UT-22-07 (Power9+FFTW, 3.62 s/transform) contains the missing CPU MPI-percentage breakdown.

### Gap 2

**The gap.** cuFFTMp has no independent (non-NVIDIA) evaluation anywhere in the survey, and the one source that exists — the DaggerFFT preprint (arXiv 2601.12209, 2026), which reportedly shows cuFFTMp 'performance remained unchanged as the number of GPUs increased' — was never fetched. DaggerFFT itself is also a new distributed-FFT library that the landscape vein does not cover as an entry at all.

**Why it matters.** cuFFTMp is the incumbent to beat on our 8x A100 partitions; every cuFFTMp number in the survey is NVIDIA marketing (Selene, fp32, 80GB parts). Whether a heFFTe+our-kernel path or a native path can beat it single-node/few-node — and whether the GROMACS 'heFFTe collapses on small grids' verdict generalizes — hinges on the only independent data point, which sits unread. One fetch also closes the missing-library hole.

**How to close it.** Fetch arXiv 2601.12209 (DaggerFFT) full text and extract its measured cuFFTMp-vs-heFFTe-vs-native numbers, machine, grid sizes, and precision.

### Gap 3

**The gap.** The friendliest-contract claim for the plug-in interface — 'use_reorder=true guarantees the executor always sees contiguous batched data (stride 1, dist=size)' — is sourced to a header comment plus inference, not to the plan/reshape logic itself; and the flagged out-of-tree-backend hazard (has_executor2d/has_executor3d hardcoded exclusion lists mean plan_logic may instantiate a 2D/3D executor our backend lacks — compile failure vs silent slab plan is undetermined) was raised and left open. Whether the batched fft3d::forward(batch_size,...) per-element executor loop preserves that contiguity per batch element is likewise unchecked.

**Why it matters.** This is the stated load-bearing question. If the guarantee holds, our adapter is ~100 lines around the existing contiguous batch-lane kernel (plus one SoA shim); if the middle-dimension pass can still hand us strided blocks or plan_logic can demand a 2D executor, we need a strided kernel variant or in-tree patches — a materially different integration cost.

**How to close it.** One targeted source read in the existing clone (/tmp/heffte_inspect on wallaby, or re-clone github.com/icl-utk-edu/heffte @ 4d8d4597): heffte_plan_logic.cpp (plan_operations) + fft3d.h compute_transform, tracing what box geometry/stride each executor constructor receives when use_reorder=true and use_pencils=true, and what happens when plan_logic selects slabs for a backend with no 2D executor.

---

## Gap-fill results

## Fill 1: Distributed-FFT communication-vs-compute fraction on multi-core CPU clusters — recovery of the broken-font figures in the heFFTe/ICL tech reports ICL-UT-22-07 and ICL-UT-22-02 (both PDFs re-fetched and page-rendered; staged at /home/lqcd/wdetmold/fft/docs/literature/staging/pdfs/)

### 1.1 CPU Vampir trace, pipelined Isend/Irecv: 1D FFT compute is 9.3% of runtime, MPI 84.3%, at 168 Power9 cores / 4 nodes

*Citation:* Ayala, Tomov, Luszczek, Cayrols, Ragghianti, Dongarra, 'Analysis of the Communication and Computation Cost of FFT Libraries towards Exascale', ICL Tech Report ICL-UT-22-07, July 2022, Fig. 7.1 p.25 (identical figure in ICL-UT-22-02, Fig. A.1 p.26). https://icl.utk.edu/files/publications/2022/icl-utk-1558-2022.pdf and .../icl-utk-1548-2022.pdf

**Claim.** Prose (digits recovered from rendered page): 'for a small count of cores (168) the communication cost (tensor transpose) is ≈ 85% even when overlapping the packing and unpacking with the MPI exchange.' Figure legend (survived encoding): Pipelined Isend/Irecv 84.26% (18.802 s), Pack 5.58% (1.245 s), Backward 1D FFT 5.13% (1.144 s), Forward 1D FFT 4.16% (0.929 s), Others 0.87%. Setup: back-to-back 3-D FFTs of size 1024^3 (5 fwd + 5 bwd), 4 Summit nodes, 168 IBM Power9 cores, 42 MPI ranks/node, heFFTe with FFTW backend.

**Numbers.** Total 1D FFT compute = 9.29% of wall time; pack = 5.58%; MPI = 84.26%. Avg time per direction 2.221 s; 72.51 GFlop/s aggregate; 497 MB/rank; max error 4.2e-15. Machine: Summit CPU partition (Power9, dual-NIC EDR IB), 4 nodes.

**Relevance to us.** This is the missing CPU Amdahl denominator, at the smallest CPU scale either report documents. A 3.5x speedup on the local 1D FFT kernel alone gives an end-to-end cap of 1/(1-0.0929+0.0929/3.5) = 1.07x at this 4-node point. Our drop-in pitch cannot rest on the local kernel alone at >=4 CPU nodes.

**Verification status.** VERIFIED (PDF fetched, page 25 rendered to image and read visually; legend percentages cross-check against the legend's absolute seconds to <0.1%). Measured numbers, not projections. heFFTe code is open (bitbucket.org/icl/hefte / github icl-utk-edu), FFTW backend.

### 1.2 CPU Vampir trace, MPI_Alltoall: 1D FFT compute 5.6%, pack+unpack 56.4%, MPI 37.9%, same 168 cores

*Citation:* ICL-UT-22-07, Fig. 7.2 p.26 (identical in ICL-UT-22-02 p.27). Same URLs as above.

**Claim.** Figure legend: Unpack 52.6% (19.089 s), MPI_Alltoall 36.7% (13.306 s), Pack 3.85% (1.398 s), Backward 1D FFT 3% (1.087 s), Forward 1D FFT 2.65% (0.956 s), MPI_Barrier 1.2% (0.416 s). Recovered prose: 'the pack and unpack kernels take a considerable amount of time (around 40%)' — note the legend actually sums pack+unpack to 56.4%; the prose '40%' is the authors' own rounding/inconsistency, both now readable. Same 1024^3, 4 Summit nodes, 168 Power9 cores, heFFTe+FFTW.

**Numbers.** 3.622 s/direction, 44.47 GFlop/s (this is the '44.5 GFlop/s CPU trace' the survey had flagged). 1D FFT = 5.65% of runtime; pack+unpack = 56.4%; MPI proper = 37.9%. Alltoall variant is 1.63x slower than the Isend/Irecv variant on identical hardware.

**Relevance to us.** Second CPU regime: with collective communication, the dominant cost is not even MPI but the pack/unpack (local data movement). If our split-complex SoA kernel fuses transpose-pack/unpack into the FFT passes, the addressable fraction becomes FFT+pack+unpack = 62.1%, and a 3.5x on all of it projects to ~1.8x end-to-end (projection, not measured).

**Verification status.** VERIFIED (page rendered and read visually; legend seconds/percentages self-consistent). Measured. Note the internal prose-vs-legend discrepancy (40% vs 56.4%) documented above.

### 1.3 Per-library CPU kernel breakdown at 4 nodes: heFFTe ~54% MPI / ~12% FFT / ~34% pack+unpack (256^3, 16 ranks)

*Citation:* ICL-UT-22-07, Fig. 5.3 p.20, 'FFT kernels breakdown on 4 nodes' (chart title; the printed caption 'Comparison of setup (planning) time' is a copy-paste error — the prose in section 5.3 ties this figure to the kernel breakdown). https://icl.utk.edu/files/publications/2022/icl-utk-1558-2022.pdf

**Claim.** Stacked 100%-runtime bars for five libraries on 4 Summit CPU nodes (config of Fig. 4.1: 256^3 FFT, 4 MPI/node, 1 MPI per Power9 core, i.e. 16 ranks): ADVMPI ~77% MPI / ~23% FFT; FFTE ~60% MPI / ~11% FFT / ~29% pack+unpack; fftMPI ~41% MPI / ~13% FFT / ~46% pack+unpack; heFFTe ~54% MPI / ~12% FFT / ~34% pack+unpack; P3DFFT ~57% MPI / ~13% FFT / ~30% pack+unpack.

**Numbers.** Read off the rendered bar chart to ~±2%: local 1D FFT compute is 11-13% of runtime for every conventional library; MPI 41-77%; pack+unpack 29-46%. Machine: Summit Power9, 4 nodes, only 4 ranks/node (lightly loaded nodes), 256^3.

**Relevance to us.** Confirms across five independent implementations and a second problem size/rank count that local FFT compute is ~12% of a distributed CPU 3-D FFT at 4 nodes. Kernel-only drop-in cap ~1.09x; kernel+pack/unpack (46% local for heFFTe) cap ~1.49x at 3.5x. Also shows FFT-ADVMPI (Dalcin et al., MPI_Alltoallw with packing pushed into MPI datatypes) as the structural alternative the authors themselves endorse.

**Verification status.** VERIFIED figure (rendered and read visually); percentages are approximate chart readings, not tabulated numbers — the report prints no table for this figure.

### 1.4 Authors' summary: MPI cost 'around 90%' dominates both collective and binary exchanges (their previous profiling)

*Citation:* ICL-UT-22-07, Chapter 3 'Analysis of FFT Communication cost', p.13, citing [6] = ICL-UT-22-02 Appendix. https://icl.utk.edu/files/publications/2022/icl-utk-1558-2022.pdf

**Claim.** Recovered prose: 'We observed how the dominance of MPI cost in runtime, around 90%, slows down the computation for both collective and binary exchanges.' This 90% is their blanket summary; the CPU-specific traces above (84.3% binary / 37.9%+56.4% collective) are the actual per-architecture numbers behind it.

**Numbers.** No new measurement; contextualizes the 90% figure previously known only from GPU traces (which we re-confirmed: 92.57% MPI at 16 GPUs Isend/Irecv, 95.13% Alltoall, Figs. 7.3/7.4 pp.27-28).

**Relevance to us.** The oft-quoted '90% MPI' is a summary spanning CPU and GPU cases; on CPUs the verified number is 84-85% comm with overlap at 168 cores, lower (41-77% MPI) at 16 lightly-packed ranks. It is scale- and packing-dependent, which is exactly why our 1-2-node regime needs its own measurement.

**Verification status.** VERIFIED (page rendered and read visually).

### 1.5 CPU strong scaling 1-4 nodes exists but carries no comm/compute breakdown at 1-2 nodes

*Citation:* ICL-UT-22-07, Figs. 4.1/4.2 p.16: 'FFT of size 256^3 on IBM Power9 cores' and 'on EPYC-7662 cores', 1-4 Summit/Spock nodes, 4 MPI/node.

**Claim.** All eleven library/decomposition variants take ~0.21-0.29 s at 1 Summit node, ~0.10-0.15 s at 2 nodes, ~0.07-0.09 s at 3, ~0.055-0.075 s at 4 — i.e. roughly ideal 2x from 1 to 2 nodes at this very low rank count (4 ranks/node). Spock EPYC-7662: ~0.17-0.24 s at 1 node, ~0.07-0.12 s at 2. Total-time bars only; no component breakdown below 4 nodes anywhere in either report.

**Numbers.** Chart readings (±10%), 256^3 fp64 complex, 4 MPI/node. Near-linear 1->2 node scaling weakly suggests comm is NOT yet saturating at 2 nodes in this lightly-packed configuration, but the reports do not decompose it.

**Relevance to us.** This is the closest either report comes to our 1-2-node axxxl regime, and it is total time only. The near-ideal 1->2-node scaling is consistent with a lower comm fraction at 2 nodes than the 84% measured at 4 fully-packed nodes, but no percentage can honestly be quoted from it.

**Verification status.** VERIFIED figure (rendered, read visually); the 1-2-node comm fraction itself remains UNDOCUMENTED in both reports.

### 1.6 'Communication ~50% at 512 cores' — NOT in either ICL report; likely origin is the PaCT 2021 paper (ref [19])

*Citation:* Ayala, Tomov, Stoyanov, Dongarra, 'Scalability Issues in FFT Computation', Parallel Computing Technologies (PaCT) 2021, Springer LNCS — cited as [19] in ICL-UT-22-07 and as the source of the Fugaku-class CPU scalability discussion.

**Claim.** I scanned every page of both re-fetched PDFs (keyword scan of extracted text plus visual reads of all percentage-bearing pages). No '512 cores' / '~50% communication' statement exists in ICL-UT-22-02 or ICL-UT-22-07. The CPU-side numbers those reports actually contain are the ones above (84.26% at 168 cores; 41-77% MPI at 16 ranks). The 512-core/50% quote in the survey was mis-attributed; the plausible source is ref [19], which we have not fetched.

**Numbers.** None — the claim is absent from these two documents.

**Relevance to us.** The survey should be corrected: replace the ABSTRACT-ONLY 512-core figure with the now-verified 168-core/84.26% and 16-rank/41-77% numbers, and either fetch the PaCT 2021 paper or drop the 512-core quote.

**Verification status.** UNVERIFIED (and specifically verified ABSENT from both ICL reports); performant code n/a.

#### Negative space

Searched: full text and rendered-page visual reads of ICL-UT-22-07 (30 pp) and ICL-UT-22-02 (29 pp), every page flagged by keyword scan for Vampir/trace/communication/MPI_/%/Alltoall/FFTW/Power9. NOT found anywhere: (1) any comm-vs-compute breakdown at 1 or 2 CPU nodes — the smallest documented CPU breakdowns are 4 nodes (168 ranks in the Vampir traces; 16 ranks in Fig. 5.3) — so the exact number for our axxxl 1-2-node Ice Lake regime remains unmeasured in the literature we hold and must be measured ourselves (heFFTe is already built in ext/install; a 2-node axxxl run with heFFTe's built-in tracing or a simple timer split would close it in one reservation); (2) the '~50% at 512 cores' figure — absent from both reports, presumed to live in the unfetched PaCT 2021 paper (ref [19]); (3) any Ice Lake/AVX-512 CPU data — CPU platforms in these reports are IBM Power9 (Summit) and AMD EPYC-7662 (Spock) only, both with different FFTW throughput and NIC configs than our nodes (a faster local kernel on Ice Lake pushes the comm fraction HIGHER than Power9 at equal network, so 84% at 4 nodes is if anything optimistic about our compute share); (4) any table of the Fig. 5.3 percentages — chart-read values only. Caveat on configs: the 84.26% trace is 1024^3 at 42 ranks/node; Fig. 5.3 is 256^3 at 4 ranks/node — comm fraction depends strongly on both, spanning 41-85% across the report's own CPU cases. Artifacts staged: /home/lqcd/wdetmold/fft/docs/literature/staging/pdfs/{icl-ut-22-07.pdf,icl-ut-22-02.pdf,r07_p15.png,r07_p16.png,r07_p18.png,r07_p22.png,r07_p23.png,r07_p27.png,r07_p28.png,r07_p29.png,r07_p30.png}; pymupdf installed in /home/lqcd/wdetmold/fft/venv (text extraction works; only prose DIGITS are broken as U+FFFD — figure-embedded text is intact and rendered pages recover everything).

#### Bottom line

Gap closed at the 4-node scale, still open at 1-2 nodes: the recovered CPU Vampir traces show heFFTe+FFTW at 168 Power9 cores spends only 5.6-9.3% of runtime in local 1D FFT compute (comm 84.3% with overlap; pack+unpack 56.4% under Alltoall), and five libraries at 16 ranks all sit at ~12% FFT / 41-77% MPI. A 3.5x local-kernel-only drop-in therefore caps at ~1.07-1.09x end-to-end at >=4 CPU nodes — the pitch survives only if our fused SoA kernel also absorbs pack/unpack (addressable fraction 46-62%, projecting a 1.5-1.8x cap) or targets 1-2 nodes, where no published breakdown exists and we should measure it ourselves on axxxl with the already-built heFFTe.

## Fill 2: Distributed-FFT landscape: independent (non-NVIDIA) evaluation of cuFFTMp, plus the missing DaggerFFT library entry

### 2.1 DaggerFFT (arXiv 2601.12209) — the missing landscape entry; contains NO cuFFTMp data at all

*Citation:* S. Taghipour Anvari, J. Samaroo, M. Raayai Ardakani, D. Kaeli, "DaggerFFT: A Distributed FFT Framework Using Task Scheduling in Julia", arXiv:2601.12209, submitted 2026-01-18. https://arxiv.org/abs/2601.12209

**Claim.** The gap statement's premise is FALSE: the full text of this paper contains zero occurrences of 'cuFFTMp', 'NVSHMEM', 'multi-process', or the phrase 'remained unchanged' (grepped the extracted PDF text of all 11 pages myself, saved at /tmp/daggerfft.txt). Its sole distributed baseline is heFFTe: "we compare against heFFTe, using cuFFT as the local backend for both frameworks" (GPU) and "Both libraries use FFTW as the local backend" (CPU). DaggerFFT itself: Julia/Dagger.jl dynamic task graph over pencil/slab-partitioned DArrays with work stealing, point-to-point MPI, C2C transforms only.

**Numbers.** Hardware: CPU cluster = Xeon Gold 6240R (48c, 2.4 GHz), InfiniBand HDR; GPU cluster = V100-SXM2 32GB, 4 GPUs/node, NVLink, scaled 4-24 GPUs (NOT A100). CUDA 12.3, Open MPI 4.1.6, FFTW 3.3.10/cuFFT local backends. GPU vs heFFTe: 840^3 FP32 slab 1.04-1.35x faster across all GPU counts (0.370 s @ 4 GPUs -> 0.181 s @ 24 GPUs, 2.04x self-scaling); 720^3 FP64 pencil 1.13-1.36x for most GPU counts; 480^3 FP64 pencil scales only 1.22x from 4 to 24 GPUs (communication-bound). CPU vs heFFTe: 2.37x (512^3 pencil) and 2.67x (1024x512^2 slab) at 4 ranks; 2.40x/2.68x at 16 ranks for 1024^3; shrinks to 1.16-1.37x at 256 ranks. Oceananigans Poisson solver: 3.19x peak (small grids), 1.3-1.8x at larger scales.

**Relevance to us.** Fills the missing landscape entry, but does NOT close the cuFFTMp gap. Useful to us anyway: its 2.4-2.7x-over-heFFTe CPU wins come entirely from scheduling/overlap while calling the same FFTW local kernel — orthogonal to (and stackable with) our faster batched-1D kernel; and it independently confirms heFFTe's local per-rank work is just vendor cuFFT/FFTW calls. Caveat: GPU results are V100-era, 4 GPU/node, so they say nothing directly about our 8x A100 nodes.

**Verification status.** VERIFIED (full PDF fetched from arXiv, text-extracted and grepped directly; not just the small-model summary). Code: paper describes a Julia implementation integrated into Oceananigans.jl; repository link not given in the extracted text.

### 2.2 Kokkos distributed-FFT paper (SC'25 Workshops) — the ACTUAL source of the misattributed cuFFTMp quote

*Citation:* Y. Asahi, T. Morvany, T. Padioleau, J. Bigot, "Development of a performance portable distributed FFT interface on top of the Kokkos ecosystem", Proc. SC'25 Workshops, Nov 2025. DOI 10.1145/3731599.3767494

**Claim.** The quote our survey attributed to DaggerFFT belongs to THIS paper: "the native implementation outperformed the cuFFTMp implementation, whose performance remained unchanged as the number of GPUs increased", with the authors adding that "better configuration of cuFFTMp needs to be investigated". Their cuFFTMp interface wraps cufftXtExecDescriptor with Kokkos::deep_copy into/out of cuFFTMp-managed buffers. Also reported: good scalability on MI250X and Icelake, "whereas the performance on the A100 scales poorly"; All2all is 60-90% of total execution time on GPUs.

**Numbers.** Snippet-level only (paywalled): Pack/Unpack at 64 MPI processes: 0.141 s (MI250X), 0.064 s (A100), 2.926 s (Icelake). Machines: NVIDIA A100 (8 GPUs/node) and AMD MI250X (Adastra); grid sizes/precision for the paper's cuFFTMp figure not extractable from snippets. ACM full text returned HTTP 403 to every fetch attempt (landing page, PDF, DOI redirect); no arXiv/HAL/OSTI preprint exists as of 2026-09-02.

**Relevance to us.** This is the only independent cuFFTMp evaluation in existence, and it is weaker evidence than the survey assumed: the flat-scaling verdict is (a) at snippet level because the full text is paywalled, and (b) explicitly retracted-in-effect by the same authors' later slides (next entry) as a CPU-binding misconfiguration. Do NOT cite 'cuFFTMp doesn't scale' as an established independent finding.

**Verification status.** ABSTRACT-ONLY (quotes come from search-engine-indexed snippets of the ACM page; full PDF is behind the ACM paywall and returned 403). Performant code exists: https://github.com/yasahi-hpc/distributed-FFT-for-kokkos (from the authors' slides), with a prototype migrating into kokkos/kokkos-fft.

### 2.3 Kokkos-FFT distributed slides (CExA tea-time, May 2026) — corrected cuFFTMp numbers: it BEATS their native path at 64 A100s

*Citation:* Y. Asahi et al., "KokkosFFT: Performance-Portable FFT interface for Kokkos Applications", CExA tea-time slides, 2026-05-20. https://cexa-project.org/kokkos-tea-time/2026-05-20-tea-time-kokkosfft/slides.pdf (fetched, 25 pp; text at /tmp/kokkosfft_slides.txt)

**Claim.** The same group's post-paper measurements reverse the SC'25 verdict. Verbatim: "We initially encountered an issue in cuFFTMp on A100 machine (8 GPUs per node) due to inappropriate CPU bindings (every process needs access to at least 2 cores)" — i.e., the paper's 'performance remained unchanged as GPUs increased' was a launcher misconfiguration. With bindings fixed, cuFFTMp is FASTER than their native GPU-aware MPI_Alltoall implementation at both 16 and 64 GPUs. Context: 3D Navier-Stokes, "Problem size fixed (1024^3, FP64)", strong scaling over 2-8 nodes, X-slab -> Y-slab; "MPI All2all costs 60-90% of the execution time on MI250X and A100"; "Thanks to NVLink inside a node, pencil (8x8) decomposition gives the best performance on A100".

**Numbers.** Per-phase table (units are the slide's timing rows, presumably seconds; summed by me): 16 A100: native total ~1.94 (All2All 1.412, FFT 0.157, pack+unpack 0.272) vs cuFFTMp ~1.59 (1.489 cufftXtExecDescriptor + 0.059 deep_copy + 0.038 normalize). 64 A100: native ~1.21 (All2All 1.074, FFT 0.044) vs cuFFTMp ~0.82 (0.791 + 0.014 + 0.009) — cuFFTMp ~1.5x faster. 64 H200: native ~0.87 vs cuFFTMp ~0.65. 16-rank Icelake CPU: All2All 8.93, unpack 4.99+pack 7.63, FFT 10.39, transpose 11.20 — compute dominates on CPU. A100 machine is 8 GPUs/node (matches our a100l/a100r topology); specific cluster names not stated for A100/H200 (MI250X is Adastra).

**Relevance to us.** Directly recalibrates our incumbent assessment: on 8-GPU-per-node A100 machines at 1024^3 FP64 — the closest published proxy to our partitions — cuFFTMp with correct CPU bindings beats a competent native MPI_Alltoall implementation by ~1.5x at 64 GPUs, so cuFFTMp is a real incumbent, not marketing vapor. Equally important for our drop-in thesis: local FFT is 0.044 s of ~1.21 s (~4%) at 64 A100s, so a faster local batched-1D kernel buys almost nothing multi-node on GPUs; on Icelake CPUs, FFT+pack/transpose dominate (>75%), which is exactly where our AVX-512 batch-lane kernel has headroom. Practical trap worth recording: cuFFTMp needs >=2 CPU cores per process or it silently underperforms.

**Verification status.** VERIFIED (slides PDF downloaded from cexa-project.org and text-extracted directly). These are the authors' own follow-up numbers, self-published slides — not peer-reviewed; the per-row units are not labeled in the extracted text (consistent with seconds given the NS strong-scaling plots). Code: https://github.com/yasahi-hpc/distributed-FFT-for-kokkos.git

#### Negative space

Searched and did NOT find: (1) any mention of cuFFTMp, NVSHMEM, or 'remained unchanged' in the full text of arXiv 2601.12209 — the gap statement misattributed the quote to DaggerFFT; the quote is from the Kokkos SC'25 workshop paper (DOI 10.1145/3731599.3767494). (2) Any open-access copy of that Kokkos SC'25 paper: ACM returned 403 on landing page, PDF, and DOI redirect; no arXiv preprint (arXiv API), no HAL deposit (HAL API returned 0 hits despite CEA authorship), nothing on OSTI. Its exact grid sizes/precision for the cuFFTMp figure therefore remain unread. (3) Any independent heFFTe-vs-cuFFTMp head-to-head anywhere — DaggerFFT compares only to heFFTe, the Kokkos work compares cuFFTMp only to its own native path, so the survey still has NO source pitting the two incumbents against each other. (4) Any independent cuFFTMp measurement on A100-40GB specifically, in fp64, at 1-8 GPUs (single node) — the Kokkos slides start at 16 GPUs / 2 nodes; the single-node few-GPU regime relevant to a first head-to-head on our a100l partition is unmeasured in all public independent sources. (5) The name of the A100 and H200 clusters in the slides (only Adastra/MI250X is named). Open gap remaining: an independent single-node cuFFTMp-vs-heFFTe-vs-native fp64 C2C measurement — we would be producing the first one.

#### Bottom line

The gap closes, but not the way the survey expected: DaggerFFT (arXiv 2601.12209, fetched in full) contains zero cuFFTMp content — the 'performance remained unchanged as GPUs increased' quote belongs to the Kokkos distributed-FFT SC'25 workshop paper (Asahi et al., DOI 10.1145/3731599.3767494, paywalled), and the same authors' May-2026 slides (fetched in full) retract it in effect: the flat scaling was a CPU-binding misconfiguration, and correctly configured cuFFTMp beats their native MPI path by ~1.5x at 64 A100s (8 GPUs/node, 1024^3 FP64) — so cuFFTMp is a genuinely strong incumbent on hardware matching ours, and the sole 'independent evidence it doesn't scale' should be struck from the survey. For our kernel thesis the same table cuts both ways: at 64 A100s local FFT is ~4% of distributed 3D FFT time (All2All is 60-90%), so a faster batched-1D kernel matters little multi-node on GPUs, while on Icelake CPUs compute+pack dominates — our CPU kernel's value proposition survives, and the single-node/few-GPU cuFFTMp regime remains publicly unmeasured, i.e., ours to measure. DaggerFFT itself enters the landscape as a Julia task-scheduling layer (V100-era, beats heFFTe 1.04-1.35x GPU / up to 2.7x CPU) that gets its wins from scheduling, not kernels — orthogonal to our work.

## Fill 3: heFFTe plug-in contract verification: direct source trace of plan_logic + fft3d::setup + compute_transform in the existing clone (/tmp/heffte_inspect on wallaby, commit 4d8d4597, current upstream main via PR #87 merge)

### 3.1 use_reorder=true invariant: every executor-facing shape has the FFT dimension as the leading (order[0]) axis

*Citation:* heFFTe source, github.com/icl-utk-edu/heffte @ 4d8d4597b479d1e4709a2b1b4cd7d8922600045f, src/heffte_plan_logic.cpp: next_pencils_shape (lines 93-120), plan_pencil_reshapes (164-254), reorder_slabs (260-267), plan_slab_reshapes (273-422)

**Claim.** In next_pencils_shape, the use_reorder branch either returns boxes_out already satisfying order[0]==dimension, reorders them via new_order(), or calls make_pencils with order=new_order(source order, dimension) — in all three paths the returned boxes have order[0] == the FFT dimension. Doc comment (line 85-86): 'If use_reorder is set, the new configuration will be transposed so that the dimension will be the new leading (fast) direction.' Same holds for slab plans via reorder_slabs. So out_shape[k][rank].order[0] == plan.fft_direction[k] for k=0,1,2, and the three fft_direction entries are pairwise distinct (get_any_valid).

**Numbers.** Not a performance claim; structural invariant traced through all plan branches (pencil plan, slab-input, slab-output, pencil-input, pencil-output, brick-brick).

**Relevance to us.** This is the first half of the friendliest-contract claim: the plan layer guarantees the box handed to each executor is ordered with the transform axis fastest. Our adapter's 1D executor will never be asked to transform along order[1] or order[2] when we set use_reorder=true.

**Verification status.** VERIFIED (full source file read; local clone at the pinned commit). Performant code exists: this is heFFTe's shipping plan logic.

### 3.2 Executor stride contract: order[0]==dimension implies stride=1, dist=size, one contiguous block of howmany=osize(1)*osize(2) back-to-back 1D FFTs

*Citation:* Same commit: include/heffte_geometry.h lines 159-174 (fft1d_get_howmany, fft1d_get_stride); include/heffte_backend_fftw.h lines 220-229 (fftw_executor 1D ctor); include/heffte_backend_stock.h lines 455-464 (stock_fft_executor ctor)

**Claim.** fft1d_get_stride returns 1 iff dimension==box.order[0] (osize(0) for order[1], osize(0)*osize(1) for order[2]). The FFTW and stock 1D executor constructors both set dist = (dimension==box.order[0]) ? size : 1 and blocks = (dimension==box.order[1]) ? box.osize(2) : 1. So when the plan invariant above holds: stride=1, dist=size, blocks=1, howmany = osize(1)*osize(2) — exactly the contiguous batched contract (line k of the pencil box starts at offset k*size). The strided/blocked middle-dimension case (stride=osize(0), blocks=osize(2), block_stride=osize(0)*osize(1)) exists in the executors but is reachable only with use_reorder=false.

**Numbers.** No timing numbers; layout facts read from constructors. plan_options doc (heffte_plan_logic.h:70-79) states the intent verbatim: 'heFFTe will reorder the data so that the backend is called for contiguous batch of 1D FFTs'; the plan printer (line 191) labels the modes 'fft1d:contiguous' vs 'fft1d:strided'.

**Relevance to us.** Second half of the contract: our batch-lane AVX-512 kernel gets, per executor call, `howmany` length-`size` complex-fp64 lines packed contiguously — an ideal shape to SIMD across lines. One caveat unchanged from our plan: data arrives as interleaved std::complex<double> (AoS), so the SoA split shim per call is still required.

**Verification status.** VERIFIED (constructors and helpers read in full).

### 3.3 The 2D/3D executor branches are runtime-unreachable when use_reorder=true — every reshape between FFT stages is non-null because consecutive shapes differ in order

*Citation:* Same commit: include/heffte_fft3d.h setup() lines 599-623; include/heffte_reshape3d.h make_reshape3d lines 504-556 and its doc (497-499: 'If the input and output are the same, then an empty unique_ptr is created. If the geometries differ only in the order, then a reshape3d_transpose instance is created.')

**Claim.** fft3d::setup builds executors from plan.out_shape[k][my_rank]. It takes the 3D-executor path only if forward_shaper[1] AND forward_shaper[2] are null, the 2D path if either is null. make_reshape3d returns nullptr only when the box sets match AND have identical order (ordered_same_as). With use_reorder=true, out_shape[0].order[0]=dir0 != out_shape[1].order[0]=dir1 != out_shape[2].order[0]=dir2, so shaper[1] and shaper[2] are always at least reshape3d_transpose (non-null), and the else branch (lines 619-622) constructs three 1D executors with make_executor(stream, out_shape[k][my_rank], fft_direction[k]). Combined with entries 1-2: the guarantee HOLDS — with use_reorder=true (and any use_pencils setting), the executor always sees contiguous stride-1, dist=size batched data. Single caveat: on a degenerate rank whose local box at some stage is EMPTY (more ranks than pencils), make_reshape3d can return nullptr despite differing orders (reshape3d.h lines 522-527), letting the 2D branch be selected; but make_executor returns nullptr for empty boxes without calling any constructor (heffte_common.h 672-674, 683-685), so a throwing placeholder is only ever hit if a rank has an empty box at one stage and a non-empty box at the adjacent stage — a pathological decomposition we would not run.

**Numbers.** Structural; no measurements.

**Relevance to us.** This closes the load-bearing question in our favor: the ~100-line adapter around the existing contiguous batch-lane kernel is sufficient. No strided kernel variant is needed as long as we construct plans with use_reorder=true (which is already the default for the fftw and mkl backends: default_plan_options<backend::fftw>::use_reorder = true, heffte_backend_fftw.h:751-754).

**Verification status.** VERIFIED (setup, make_reshape3d, make_executor, and reshape3d_transpose all read; invariant chain checked branch by branch). The header-comment-plus-inference sourcing is now replaced by the reshape/plan logic itself.

### 3.4 Out-of-tree backend hazard resolved: missing 2D/3D constructors are a COMPILE failure, not a silent slab plan; the in-tree pattern is throwing placeholders (stock backend)

*Citation:* Same commit: include/heffte_common.h has_executor2d/has_executor3d lines 714-755 (hardcoded is_same exclusion lists), make_executor overloads lines 669-697; include/heffte_fft3d.h setup() lines 607-623 (plain runtime if, no if-constexpr anywhere in the file — grep confirmed); include/heffte_backend_stock.h lines 466-472 (placeholder ctors: throw std::runtime_error("2D/3D transforms for the stock backend are not available yet!")); same pattern in heffte_r2r_executor.h lines 213-217

**Claim.** has_executor2d/3d<tag> return true for ANY tag not on the hardcoded exclusion list — an out-of-tree tag defaults to 'has 2D and 3D'. But because setup()'s branches are ordinary runtime if-statements, all three make_executor overloads — and therefore the executor constructors (stream,box,dim), (stream,box,dir1,dir2), and (stream,box) — are ODR-used and instantiated for every backend tag used with fft3d, regardless of which branch runs. A custom one_dim_backend<tag>::executor lacking the 2D/3D constructor signatures fails to COMPILE; it can never silently degrade to a slab plan. The stock backend exists on the exclusion lists precisely because its 2D/3D constructors are compile-satisfying placeholders that throw at runtime. An out-of-tree backend can additionally explicitly specialize has_executor2d<my_tag>()/has_executor3d<my_tag>() to false (they are ordinary function templates; no other opt-out mechanism exists in the tree — grep shows their only uses are the two setup() lines), making even the degenerate empty-box path select 1D executors.

**Numbers.** Structural; no measurements.

**Relevance to us.** Our adapter recipe is now fully determined: (a) 1D constructor with the real kernel; (b) 2D/3D constructors that throw, copying the stock pattern, to satisfy compilation; (c) optionally specialize has_executor2d/3d to false for belt-and-braces; (d) always plan with use_reorder=true so the throwing paths are unreachable. The feared undetermined failure mode (compile vs silent slab) is settled: compile error if we omit the ctors, loud runtime exception if a degenerate/non-reorder plan ever reaches a placeholder — never a silent wrong-speed plan.

**Verification status.** VERIFIED (all cited code read; the compile-failure conclusion follows from the runtime-if instantiation semantics, corroborated by stock's placeholders whose only reason to exist is this instantiation requirement). Not additionally demonstrated with a compile test.

### 3.5 Batched fft3d::forward(batch_size,...) preserves per-element contiguity: the executor is invoked once per batch element at data + j*box_size(), and reshapes apply per element at the same granularity

*Citation:* Same commit: include/heffte_fft3d.h lines 391-414 (batched forward calls compute_transform with batch_size); src/heffte_compute_transform.cpp lines 34-48 (apply_fft lambda: for j in 0..batch_size-1, executor[i]->forward(data + j*executor[i]->box_size(), executor_workspace)) and buffer offsets at lines 32, 70 (workspace + batch_size*executor_buffer_offset / batch_size*size_comm_buffers); include/heffte_reshape3d.h reshape3d_transpose::transpose lines 473-484 (per-element loop at j*input_size)

**Claim.** The batch is laid out as batch_size back-to-back copies of the local box (element j at offset j*box_size()). Each of the batch_size executor calls per stage sees exactly the same box geometry as the unbatched case — so the stride-1/dist=size contract of entries 1-3 holds per batch element unchanged. heFFTe does NOT fuse the batch loop into one executor call: cross-element batching would require modifying compute_transform (or detecting the fixed j-stride ourselves, which the executor cannot, since it receives only a bare pointer per call). Reshape operators apply per element too (transpose loops j; the MPI reshapes take batch_size and handle offsets internally), preserving the layout between stages.

**Numbers.** Structural; no measurements. Note the per-call batch our kernel CAN exploit is howmany = osize(1)*osize(2) contiguous lines within one pencil box — for typical LQCD pencils (e.g., a 32^3x64 lattice pencil-decomposed on 64 ranks, local box 32x8x16) that is hundreds of lines per call, ample width for 8-wide fp64 SIMD across lines.

**Relevance to us.** The third open sub-question is closed: batched fft3d calls hand our executor the same contiguous contract per element. Our SIMD-across-batch design maps onto the howmany lines within each call rather than across fft3d batch elements; no strided variant needed, but also no opportunity (without an in-tree patch) to amortize twiddle loads across fft3d batch elements beyond what one box already provides.

**Verification status.** VERIFIED (compute_transform.cpp read in full, all three variants: c2c, r2c, c2r).

#### Negative space

Searched the full clone (include/ + src/ at commit 4d8d4597) for: (1) any documented API-level statement of the contiguity guarantee beyond the plan_options doc comment (heffte_plan_logic.h:70-79) — there is none; the guarantee is real but established only by the code trace above, and no test in the tree asserts it as a contract, so it could in principle change upstream without notice (pin the commit, or add an assert in our adapter that box.order[0]==dimension and throw otherwise — cheap and loud). (2) Any extension-point documentation for out-of-tree backends — none found in include/ or src/ (did not exhaustively read doxygen/ or docker/; the practical extension surface is: specialize one_dim_backend<tag>, buffer_traits, default_plan_options, optionally has_executor2d/3d, and derive from executor_base at heffte_common.h:561-595). (3) A compile test demonstrating the missing-constructor failure — not performed; the conclusion rests on ODR/instantiation semantics plus the existence of stock's throwing placeholders, which I judge sufficient, but a 20-line negative compile check during adapter work would make it airtight. (4) The r2c pipeline (fft3d_r2c, make_executor_r2c) was not traced — irrelevant for our complex LQCD fields, flagged so nobody mistakes this report as covering it. (5) One residual runtime hazard remains OUTSIDE our configuration: with use_reorder=false, shaper[1] can legitimately be null on slab-shaped decompositions (e.g., 2 ranks splitting z gives boxes that are pencils in both x and y), in which case an out-of-tree backend with has_executor2d defaulting true gets its 2D placeholder constructor CALLED on a non-empty box → runtime throw. Our adapter must therefore treat use_reorder=true as a hard requirement, not a preference.

#### Bottom line

The friendliest-contract claim is now VERIFIED from plan/reshape source, not header comments: with use_reorder=true, every shape heFFTe hands an executor has the FFT axis as order[0], every inter-stage reshape is non-null (orders differ between consecutive stages, and make_reshape3d nulls only on identical geometry AND order), so the 2D/3D executor branches are unreachable and each 1D executor call receives stride=1, dist=size, howmany=osize(1)*osize(2) contiguous complex data — per batch element, since the batched path loops j at data + j*box_size(). The out-of-tree hazard resolves to: compile failure if the 2D/3D constructor signatures are absent (all make_executor overloads are ODR-used by setup's runtime ifs), loud runtime throw if stock-style placeholders are ever reached — never a silent slab plan. Adapter cost stays at the ~100-line estimate: 1D ctor + throwing 2D/3D placeholders + optional has_executor2d/3d=false specializations + the AoS-to-SoA shim, with use_reorder=true mandatory.
