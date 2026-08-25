# Performance analysis tools available to implementers

The PMU is locked on the nodes (perf_event_paranoid=4, no perf binary) — hardware counters
are NOT available. These static/emulation tools are, and they answer most of the same
questions deterministically (no noisy-window discipline needed):

## 1. llvm-mca (LLVM 22, knows Ice Lake Server) — per-loop port pressure & throughput
Compile your kernel to asm and mark the loop of interest:
    /opt/software/llvm-22.1.8/bin/clang -O3 -march=icelake-server -S kernel.c -o kernel.s
    # add comments around the hot loop:   # LLVM-MCA-BEGIN mykernel  /  # LLVM-MCA-END
    /opt/software/llvm-22.1.8/bin/llvm-mca -mcpu=icelake-server -iterations=200 \
        -timeline -bottleneck-analysis kernel.s
Gives: cycles/iteration, uops/cycle, PER-PORT dispatch counts (port 0/1 FMA vs port 5
shuffle pressure — the number you have been inferring from rdtsc probes), bottleneck
attribution. Model, not measurement: trust it for RELATIVE choices between two schedules.

## 2. uiCA (ext/tools/uiCA, Ice Lake model from uops.info) — the most accurate ICL model
Analyzes RAW MACHINE CODE of a function in your built binary:
    bench/gen/tools/uica_fn.sh build/$(hostname -s)/bin/<entry> <function_symbol>
Cross-check llvm-mca with it; where they disagree on a port count, uiCA is usually right
for Intel. Also predicts frontend effects mca misses.

## 3. OSACA (ext/tools/osaca-pkg) — third opinion, ICX support
    PYTHONPATH=/home/lqcd/wdetmold/fft/ext/tools/osaca-pkg python3 -m osaca \
        --arch ICX kernel.s

## Discipline
- These are MODELS. The benchmark node's measured chain time remains the only score.
  Use the models to choose between candidate schedules BEFORE spending a lease slot,
  and to explain a measured gap (port 5 saturation, dispatch-width limits, dependency
  chains) — record model-vs-measured in your strategy file when they disagree.
- The known model blind spots on our workload: L1-bank/4K-aliasing effects, huge-page
  TLB behavior, and the ~2.1 vector-uops/cycle global cap under mixed loads (models
  assume 5-6 dispatch; the node measures ~2.1 under memory pressure) — anything
  traffic-bound must still be decided by measurement.

## 4. HARDWARE COUNTERS — /tmp/perf on the node (live as of Aug 25)
perf_event_paranoid is now 2 on a80n0 and a node-local perf 5.15 sits at /tmp/perf
(EPHEMERAL: gone at reboot; rebuild = scp ext/tools/perf-install/bin/perf a80n0:/tmp/perf).
    tools/pmu.sh taskset -c $CORE ./bin --L 32 --batch 8 --chain 250 ...
Default event set answers the standing questions:
  - uops_dispatched.port_0/port_1 (FMA pipes) vs port_5 (shuffles): the real FMA:shuffle
    balance you have been estimating with rdtsc probes;
  - uops_dispatched.port_2_3 / port_4_9: load and store pressure;
  - l1d.replacement: L1 traffic (the 112 B/pt/iter floor question at small L);
  - core_power.lvl2_turbo_license: cycles at the 512-bit license level (the zmm-vs-ymm
    downclock question the CLX advisory raised).
Add events with -e (see /tmp/perf list). MEASURE ON A LEASED CORE. Counters are per-process
and now ground truth — where a model (mca/uiCA) and the counters disagree, believe the
counters and record the disagreement in your strategy file.
