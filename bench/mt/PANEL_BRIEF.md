# Implementer brief — MULTICORE complex 3D FFT for one fixed geometry

This is phase 2 of the project. Phase 1 built single-threaded kernels that beat every CPU
library at every geometry (1.15× to 4.97×). Your job is to make them use **32 cores**.

Everything in the phase-1 brief still holds — the ABI in `fft3d_api.h`, the layout, the
correctness gate, the timing method — with these changes:

## What is different

1. **You get 32 cores.** The harness runs every backend with
   `OMP_NUM_THREADS=32`, `OMP_PROC_BIND=close`, `OMP_PLACES=cores`,
   `OMP_DYNAMIC=false`. OpenMP is expected; raw pthreads are allowed.
2. **Do not take more than you are given.** Never call `omp_set_num_threads()` to raise the
   count, and never spin up threads beyond it. The comparison depends on every backend
   getting the same cores.
3. **Thread pools belong in `fft3d_create()`.** Creating threads, first-touching NUMA-local
   scratch, and computing a work decomposition are all setup and are excluded from your
   time. `fft3d_execute()` should ideally not create a single thread.
4. **The library baselines are threaded too**: FFTW with `fftw_plan_with_nthreads(32)`
   at three planner levels, oneMKL 2022 and 2026 with the GNU OpenMP threading layer, and
   ducc0 with `nthreads=32`. Beating a sequential library with 32 cores would
   prove nothing, so you are not being asked to.

## What is hard here, and what to think about

* **NUMA.** The benchmark node is two sockets. A buffer first-touched by one thread lives in
  that socket's memory, and a thread on the other socket pays the interconnect to reach it.
  The driver allocates and fills `in` and `out` before calling you, so you do **not**
  control first touch of the caller's buffers — but you do control your own scratch, and you
  control which thread touches which part of the caller's data. Think about whether your
  decomposition keeps each thread on data its socket owns.
* **B=1 may not scale at all.** One $L^3$ volume is 3.4 KB at L=6 and 746 KB at L=36; split
  across 32 cores that is 108 bytes to 23 KB per thread, and the synchronisation may
  cost more than the work. A well-argued "B=1 does not parallelise past N cores, and here is
  the measurement" is a real result — do not fake scaling by adding overhead-free-looking
  barriers that do nothing.
* **The batch is the obvious axis, and it is probably the right one.** Independent volumes
  are embarrassingly parallel with no communication at all. The interesting question is
  whether you should also split *within* a volume at large L (36, 45, 64) to keep each
  thread's working set inside its own L2.
* **Memory bandwidth is shared.** Phase 1's literature (`docs/literature/08-*.md`) found
  that a single core is limited by outstanding-miss concurrency (~10 line fill buffers), not
  by DRAM — which is precisely why more cores can help a bandwidth-bound case: more cores
  means more fill buffers. Expect the batched cases to scale better than the arithmetic
  suggests, and the small-batch cases worse.
* **False sharing.** Two threads writing different volumes that share a cache line will
  destroy your scaling. Volume sizes here are all multiples of 64 bytes, so this bites at
  the boundaries of *your own* scratch, not the caller's buffers.

## Reporting

Report **per-transform time** as always, and in your strategy record also report **parallel
efficiency**: your time at 32 threads against the same kernel's single-thread time
(the phase-1 leaderboards in `../geom/results/` give you the latter). A kernel that is 8×
faster on 32 cores is a more useful result than one that is 9× faster but whose author
cannot say where the other 23 cores went.

## Records

Independent of phase 1: your code is in `impl_N/`, your record in `strategies/`, the
leaderboards in `results/mt_rN/`. Read `../geom/strategies/` for how your kernel reached
its current form, and `../../docs/LITERATURE.md` (all nine sections) for the technique
corpus.
