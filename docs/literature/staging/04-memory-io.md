# Vein 4: memory-hierarchy & data-movement theory (agent report, 2026-08-24)

## Theory: single-transform I/O is CLOSED; the chain is open
- Bilardi & De Stefani FSTTCS 2022 (arXiv:2210.01897): DAG-visit/boundary-complexity framework;
  NO new FFT-specific bound. Hong-Kung 1981 Theta(n logn / logM) remains tight for one transform
  and IS MET by blocked/four-step FFTs. No theoretical headroom inside one transform.
- VERIFIED NEGATIVE: no published I/O lower bound for the composed chain FFT->pointwise->FFT.
  The chain floor is strictly below the sum of per-kernel floors; nobody has formalized it. Our
  measured traffic numbers could speak to this open theory question directly.
- IOLB/IOOpt (PLDI 2020/2021): automated parametric I/O bounds + tiling recommendations for AFFINE
  programs. FFT recursion isn't affine, but a fixed-L chain as batched small-DFT matmuls +
  transposes + pointwise IS — could certify how close a fused-chain schedule is to the floor at
  S=48KB/1.25MB. Never run on an FFT chain.
- Hourglass pattern (SPAA 2024, arXiv:2404.16443): tightened bounds for hourglass dependency
  graphs — an FFT->pointwise->iFFT chain is literally hourglass-shaped; machinery exists, never
  applied to Fourier chains.
- Multiprocessor red-blue pebbling (arXiv:2409.03898): optimal I/O inapproximable in general —
  principled justification for schedule SEARCH (our race) over analytic optima.

## Practice: every post-2020 win is chain fusion, and all are GPU-only
- FlashFFTConv (ICLR 2024): Monarch order-p decomposition trades FLOPs for I/O, whole
  fwd-FFT->pointwise->inv-FFT as one fused pass, up to 7.93x over PyTorch. GPU BF16/FP16, 1D.
  The "pick decomposition order to match L1/L2 capacity" knob transfers to AVX-512 fp64 — no
  equivalent exists.
- TurboFNO (arXiv:2504.11681): first fully fused FFT-GEMM-iFFT kernels; FFT loop order
  restructured specifically so the chain composes ("thread block iterates the hidden dim,
  aligning with the GEMM k-loop"); beats cuFFT/cuBLAS by up to 150% on A100. The design question
  "which FFT dataflow makes the CHAIN fuse best" has no published CPU answer.
- cuFFT LTO callbacks (documented in arXiv:2604.06085): library FFTs are "a hard fusion barrier";
  vendor GPU stacks solved it with JIT callbacks. NO CPU analog — FFTW/MKL have no fusion hook;
  owning the FFT code (us) is the only CPU route.
- FFTX/EVPFFTX (HPEC 2023): the one framework whose THESIS is chain-as-unit (SPIRAL OL
  co-optimization of DFT calls + pointwise glue); EVPFFTX is "a first look" with NO performance
  results — the gap we'd fill.

## CPU structure ideas, unported
- MDFFT transpose-free 3D FFT (HPCC 2021, ARM): column-order strided butterflies + cache-aware
  blocking eliminate transposes; "generally better than FFTW and ARMPL on ARM". No AVX-512 port.
- Vector-radix / dimension-fused 3D FFT: DEAD post-2015 in performant software. Nearest ancestor:
  two-pass 3D FFT (Xeon Phi 2014) pairing axes per DRAM pass. Unclaimed territory for L2-resident
  pencil-pair passes at L=36..128.
- Cache-oblivious: no advances since Frigo-Leiserson 1999.
- QTT "superfast FFT" revival (arXiv:2412.11566 homogenization): only if data provably low-rank;
  logged because FFT homogenization is workload-isomorphic to our chain.

## Synthesis
A fp64 AVX-512 implementation combining (a) map fused into last/first butterfly stages of adjacent
transforms, (b) two axes per DRAM pass with L2-resident tiles, (c) transpose-free column-order —
would combine ideas the literature validated separately but never in CPU software.
