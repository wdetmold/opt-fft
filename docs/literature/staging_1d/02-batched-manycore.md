# 1D vein: batched & multicore/manycore CPU (agent, 2026-09-02)

## The core principle (our regime's default)
DFT_n (x) I_v (Franchetti/Puschel SPIRAL): run v transforms side-by-side, every scalar op
of ANY inner algorithm becomes one v-wide vector op, zero reorg — vectorization orthogonal
to algorithm. Wins over within-transform SIMD when B >= SIMD width (8 fp64) and pack/unpack
amortizes (our B=1 remainder path losing 2-3x to MKL is exactly the B<width failure).
- Popovici/Franchetti/Low HPEC 2017 (mixed/split layout): keep hot arithmetic split (SoA)
  so complex-mul needs no shuffle/addsub; convert only at stage boundaries. fp64 x86, up to
  2x FFTW. THE published justification for our split-complex batch-lane layout — closes
  split-vs-interleaved in split's favor.

## Multicore is SIMPLER than the headline results (a simplifying finding)
- For a BATCH of small/medium transforms the parallel axis IS the batch: embarrassingly
  parallel, no false-sharing rewrite, no cross-core transpose. SPIRAL-SMP (SC06) and
  Ali/Johnsson ICS07 solve SINGLE-transform threading — NOT needed for us. Cite to justify
  NOT doing the complicated thing.
- Our multicore problem collapses to: (a) NUMA-local first-touch + thread pinning of the
  BATCH (arXiv:2109.12259 NUMA-aware batched conv: allocate+first-touch from the threads
  that transform it; partition the batch not the transform), (b) maximize single-core
  batch-lane throughput. OpenFFT ICS23's thread-allocation-BY-TASK-SIZE is the one manycore
  knob worth importing. (We already measured a 2x THP/NUMA swing on our own node.)
- "One huge transform" (Takahashi six-step, FFTE Xeon Phi 91 GFlops 2^29; SVE/A64FX
  register-resident) is NOT our regime — parallelism goes inside one transform. Touches us
  only at the top of pow2 (16384 is L2-resident, no six-step needed) and for the
  register-residency lesson: 32x512-bit regs -> prefer higher radix (transfers to AVX-512).

## Vendor batched weaknesses = our openings
- FFTW plan_many: SIMD codelets vectorize ACROSS the howmany loop when batch stride is
  regular; collapses to a SCALAR codelet in a plain loop for no-vector-codelet sizes (large
  primes, odd L) or irregular dist. That fallback is where hand-written batch-lane wins.
- MKL DFTI: Intel's own guidance says it LEANS on the batch to use threads + hide latency
  -> single small transform is a known weak spot (matches our B=1 finding); cannot express
  ragged per-transform distances (must pad to max); no padded batch in DPC++/oneMKL.

## Downstream pull (real consumers)
heFFTe (ICCS20): distributed 3D FFT = 3 batched-1D passes; local per-rank kernel is just a
vendor batched-1D call -> a faster hand-written batched-1D fp64 kernel is a DROP-IN
replacement. OpenFFT, DaCe batched-DFT case study (HPC Asia 23) = batched-DFT now a
first-class kernel class. LQCD: same pencil/batched-1D structure = momentum-space transforms.

## The clearest open gap (our large primes)
NO published wide-SIMD batch-vectorized Rader/Bluestein at ~10^5 primes on AVX-512. In a
batch, Rader's inner (p-1)-length FFTs and the pointwise multiply all run DFT (x) I_v across
the batch — batch-lane vectorization one level in, plus extended-precision (p-1) twiddle
tables. The win is not a cleverer prime algorithm, it's applying the batch-lane trick to the
inner convolution. This is the natural large-prime experiment.
