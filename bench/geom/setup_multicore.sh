#!/bin/bash
# Phase 2: build the MULTICORE competition harness, then hand the baton to it.
#
# Installed as the single-thread harness's after_series.sh, so it runs when that series
# finishes. It derives bench/mt from bench/geom at that moment -- inheriting whatever fixes
# the single-thread phase accumulated -- applies the multicore deltas, seeds the new panel
# with the single-thread champions, and points bench/PHASE at the new harness.
#
# Records stay INDEPENDENT: bench/mt has its own impl_N, strategies, results, exemplars,
# leaderboards and git history, and its rounds are named mt_rN. A multicore number and a
# single-thread number are not comparable and must never share a leaderboard.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GEOM=$(pwd)
BENCH=$(readlink -f "$GEOM/..")
MT=$BENCH/mt
LOG() { printf '[%s] setup-mt: %s\n' "$(date '+%F %T')" "$*"; }

THREADS=${FFT_MT_THREADS:-32}     # Gold 5218 benchmark node: 2x16 physical cores

if [ -d "$MT" ]; then LOG "$MT already exists -- leaving it alone"; else

LOG "deriving the multicore harness from $GEOM"
mkdir -p "$MT/sota" "$MT/results" "$MT/strategies" "$MT/exemplars" "$MT/logs"

# ---- shared machinery, copied as-is -------------------------------------------------
for f in fft3d_api.h driver.c gen_input.py check.py leaderboard.py promote.sh \
         sweep.sh submit.sh tryout.sh probe_node.sh Makefile; do
  cp "$GEOM/$f" "$MT/$f"
done
cp "$GEOM"/sota/*.c "$GEOM"/sota/*.cc "$MT/sota/" 2>/dev/null || true
chmod +x "$MT"/*.sh
cp "$GEOM/exemplars/README.md" "$MT/exemplars/" 2>/dev/null || true

# ---- the contract note: threads are now allowed ------------------------------------
python3 - "$MT/fft3d_api.h" "$THREADS" <<'PY'
import sys
path, threads = sys.argv[1], sys.argv[2]
s = open(path).read()
s = s.replace(
"""*   * Single-threaded.  No OpenMP, no pthreads, in this round.""",
f"""*   * MULTICORE.  OpenMP is expected; pthreads are allowed.  The harness fixes the thread
 *     count at OMP_NUM_THREADS={threads} with OMP_PROC_BIND=close and OMP_PLACES=cores, and
 *     every backend is measured under identical settings.  Do NOT call omp_set_num_threads()
 *     to grab more than you were given -- the comparison depends on everyone getting the
 *     same cores.  Creating your own thread pool in fft3d_create() is fine and is the point:
 *     thread creation is setup, not transform time.""")
open(path, 'w').write(s)
PY

# ---- Makefile: OpenMP everywhere, plus threaded library baselines -------------------
python3 - "$MT/Makefile" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("CFLAGS     ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops",
              "CFLAGS     ?= -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops -fopenmp")
s = s.replace("CXXFLAGS   ?= -O3 -march=native -mtune=native -std=c++17 -fno-math-errno",
              "CXXFLAGS   ?= -O3 -march=native -mtune=native -std=c++17 -fno-math-errno -fopenmp")
# threaded FFTW: libfftw3_omp on top of libfftw3, and MKL's OpenMP threading layer
s = s.replace("FFTW_LIBS  := -L$(FFT_PREFIX)/lib -Wl,-rpath,$(FFT_PREFIX)/lib -lfftw3",
              "FFTW_LIBS  := -L$(FFT_PREFIX)/lib -Wl,-rpath,$(FFT_PREFIX)/lib -lfftw3_omp -lfftw3")
s = s.replace("-lmkl_intel_lp64 -lmkl_sequential -lmkl_core $(LDLIBS)",
              "-lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core $(LDLIBS)")
s = s.replace("$(MKL2026)/lib/libmkl_intel_lp64.so.3 $(MKL2026)/lib/libmkl_sequential.so.3",
              "$(MKL2026)/lib/libmkl_intel_lp64.so.3 $(MKL2026)/lib/libmkl_gnu_thread.so.3")
open(p, 'w').write(s)
PY

# FFTW needs an explicit init + plan_with_nthreads call; patch the copied backend.
python3 - "$MT/sota/fftw3.c" "$THREADS" <<'PY'
import sys
p, threads = sys.argv[1], sys.argv[2]
s = open(p).read()
s = s.replace("""    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;""",
f"""    fft3d_plan *p = malloc(sizeof *p);
    if (!p) return NULL;

    /* Threaded FFTW: init once, then tell the planner how many threads to plan for.
       Planning is setup, so this is outside the measured region. */
    static int threads_ready = 0;
    if (!threads_ready) {{
        if (!fftw_init_threads()) {{ free(p); return NULL; }}
        threads_ready = 1;
    }}
    fftw_plan_with_nthreads({threads});""")
s = s.replace('return "FFTW 3.3.10 plan_many_dft, " PLAN_NAME;',
              'return "FFTW 3.3.10 plan_many_dft, threaded, " PLAN_NAME;')
open(p, 'w').write(s)
PY

# MKL: threading layer is chosen at link time; make the description honest.
sed -i 's/oneMKL 2022.0.2 DFTI, sequential, batched/oneMKL 2022.0.2 DFTI, GNU OpenMP threading, batched/;
        s/oneMKL 2026.1 (pip wheel) DFTI, sequential, batched/oneMKL 2026.1 DFTI, GNU OpenMP threading, batched/' \
  "$MT/sota/mkl_dfti.c" 2>/dev/null || true
sed -i 's/DFTI, sequential/DFTI, threaded/' "$MT/Makefile" 2>/dev/null || true

# ducc0 takes its thread count as an argument.
sed -i "s|/\\*nthreads=\\*/1|/*nthreads=*/$THREADS|; s|ducc0 0.41 c2c, no planning, 1 thread|ducc0 0.41 c2c, no planning, $THREADS threads|" \
  "$MT/sota/ducc0_c2c.cc" 2>/dev/null || true

# ---- sweep: pin the threads identically for every backend --------------------------
python3 - "$MT/sweep.sh" "$THREADS" <<'PY'
import sys
p, threads = sys.argv[1], sys.argv[2]
s = open(p).read()
s = s.replace('source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1',
f"""source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

# Every backend is measured under identical threading. Pinning matters more than the count:
# without PROC_BIND the OS migrates threads across the two sockets mid-measurement and the
# run-to-run spread swamps the differences we are trying to see.
export OMP_NUM_THREADS={threads}
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export MKL_NUM_THREADS={threads}
export MKL_DYNAMIC=false""")
s = s.replace('echo "cores: $(nproc)   governor:',
              f'echo "threads: $OMP_NUM_THREADS of $(nproc) (PROC_BIND=$OMP_PROC_BIND)   governor:')
open(p, 'w').write(s)
PY

# ---- cases: multicore needs enough work to divide -----------------------------------
cat > "$MT/cases.txt" <<'CASES'
# Multicore sweep. B=1 is kept because it is the hard case for parallelism -- one small
# volume across many cores is latency-bound and may not scale at all, which is itself a
# result worth recording. The large batches are where the cores should pay.
6:1     6:4096    6:65536
8:1     8:2048    8:32768
17:1    17:256    17:4096
36:1    36:32     36:512
13:1    13:512    13:8192
23:1    23:128    23:2048
45:1    45:16     45:256
64:1    64:8      64:128
CASES

# ---- seed the panel with the single-thread champions -------------------------------
mkdir -p "$MT/impl_1"
for f in "$GEOM"/impl/*.c; do
  base=$(basename "$f")
  { echo "/* Carried over from the SINGLE-THREAD competition, where this file finished as"
    echo " * written below. Your job in the multicore phase is to parallelise it across"
    echo " * $THREADS cores without losing its single-core efficiency -- read"
    echo " * ../PANEL_BRIEF.md, and read ../../geom/strategies/${base%.c}.md for the full"
    echo " * history of how this kernel got here."
    echo " */"
    cat "$f"
  } > "$MT/impl_1/$base"
done
ln -sfn impl_1 "$MT/impl"
LOG "seeded impl_1 with $(ls "$MT/impl_1"/*.c | wc -l) single-thread champions"

# ---- the brief ----------------------------------------------------------------------
cat > "$MT/PANEL_BRIEF.md" <<BRIEF
# Implementer brief — MULTICORE complex 3D FFT for one fixed geometry

This is phase 2 of the project. Phase 1 built single-threaded kernels that beat every CPU
library at every geometry (1.15× to 4.97×). Your job is to make them use **$THREADS cores**.

Everything in the phase-1 brief still holds — the ABI in \`fft3d_api.h\`, the layout, the
correctness gate, the timing method — with these changes:

## What is different

1. **You get $THREADS cores.** The harness runs every backend with
   \`OMP_NUM_THREADS=$THREADS\`, \`OMP_PROC_BIND=close\`, \`OMP_PLACES=cores\`,
   \`OMP_DYNAMIC=false\`. OpenMP is expected; raw pthreads are allowed.
2. **Do not take more than you are given.** Never call \`omp_set_num_threads()\` to raise the
   count, and never spin up threads beyond it. The comparison depends on every backend
   getting the same cores.
3. **Thread pools belong in \`fft3d_create()\`.** Creating threads, first-touching NUMA-local
   scratch, and computing a work decomposition are all setup and are excluded from your
   time. \`fft3d_execute()\` should ideally not create a single thread.
4. **The library baselines are threaded too**: FFTW with \`fftw_plan_with_nthreads($THREADS)\`
   at three planner levels, oneMKL 2022 and 2026 with the GNU OpenMP threading layer, and
   ducc0 with \`nthreads=$THREADS\`. Beating a sequential library with $THREADS cores would
   prove nothing, so you are not being asked to.

## What is hard here, and what to think about

* **NUMA.** The benchmark node is two sockets. A buffer first-touched by one thread lives in
  that socket's memory, and a thread on the other socket pays the interconnect to reach it.
  The driver allocates and fills \`in\` and \`out\` before calling you, so you do **not**
  control first touch of the caller's buffers — but you do control your own scratch, and you
  control which thread touches which part of the caller's data. Think about whether your
  decomposition keeps each thread on data its socket owns.
* **B=1 may not scale at all.** One \$L^3\$ volume is 3.4 KB at L=6 and 746 KB at L=36; split
  across $THREADS cores that is 108 bytes to 23 KB per thread, and the synchronisation may
  cost more than the work. A well-argued "B=1 does not parallelise past N cores, and here is
  the measurement" is a real result — do not fake scaling by adding overhead-free-looking
  barriers that do nothing.
* **The batch is the obvious axis, and it is probably the right one.** Independent volumes
  are embarrassingly parallel with no communication at all. The interesting question is
  whether you should also split *within* a volume at large L (36, 45, 64) to keep each
  thread's working set inside its own L2.
* **Memory bandwidth is shared.** Phase 1's literature (\`docs/literature/08-*.md\`) found
  that a single core is limited by outstanding-miss concurrency (~10 line fill buffers), not
  by DRAM — which is precisely why more cores can help a bandwidth-bound case: more cores
  means more fill buffers. Expect the batched cases to scale better than the arithmetic
  suggests, and the small-batch cases worse.
* **False sharing.** Two threads writing different volumes that share a cache line will
  destroy your scaling. Volume sizes here are all multiples of 64 bytes, so this bites at
  the boundaries of *your own* scratch, not the caller's buffers.

## Reporting

Report **per-transform time** as always, and in your strategy record also report **parallel
efficiency**: your time at $THREADS threads against the same kernel's single-thread time
(the phase-1 leaderboards in \`../geom/results/\` give you the latter). A kernel that is 8×
faster on 32 cores is a more useful result than one that is 9× faster but whose author
cannot say where the other 23 cores went.

## Records

Independent of phase 1: your code is in \`impl_N/\`, your record in \`strategies/\`, the
leaderboards in \`results/mt_rN/\`. Read \`../geom/strategies/\` for how your kernel reached
its current form, and \`../../docs/LITERATURE.md\` (all nine sections) for the technique
corpus.
BRIEF

# ---- config, state, and the baton ---------------------------------------------------
cat > "$MT/results/.rounds_config" <<CFG
FFT_ROUND_PREFIX=mt_r
FFT_FIRST_ROUND=1
FFT_PARTITION=devel
FFT_TIME=59
CFG
printf '1 4\n' > "$MT/results/.rounds_state"
LOG "armed mt_r1 .. mt_r4"

# The GPU phase takes over when the multicore series finishes.
cp "$GEOM/setup_gpu.sh" "$MT/setup_gpu.sh" 2>/dev/null && chmod +x "$MT/setup_gpu.sh" \
  && ln -sfn setup_gpu.sh "$MT/after_series.sh" \
  && LOG "armed the GPU phase as the multicore series' successor" \
  || LOG "WARNING: setup_gpu.sh not found in $GEOM -- the chain stops after multicore"

fi   # end of "already exists"

printf 'mt\n' > "$BENCH/PHASE"
LOG "bench/PHASE -> mt; cron will start mt_r1 on its next tick"
rm -f "$GEOM/after_series.sh"
LOG "disarmed the single-thread harness's hook"
