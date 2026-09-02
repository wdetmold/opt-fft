#!/bin/bash
# Runs on the compute node(s).  Sweeps (grid x rank-count x communication algorithm) through
# heFFTe's own speed3d_c2c benchmark, built with tracing, and leaves one trace directory per
# configuration for parse_traces.py.  See README.md for what is being measured and why.
#
# usage: commfrac.sh --tag TAG [--isa avx512|avx2] [--nruns 5] [--grids "256 512"]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

TAG=""; ISA=avx512; NRUNS=5; GRIDS="256 512"
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --isa) ISA=$2; shift 2 ;;
    --nruns) NRUNS=$2; shift 2 ;;
    --grids) GRIDS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "commfrac.sh: --tag is required" >&2; exit 2; }

BIN=/home/lqcd/wdetmold/fft/ext/src/heffte/build-trace-$ISA/benchmarks/speed3d_c2c
[ -x "$BIN" ] || { echo "commfrac.sh: no traced binary at $BIN -- run ext/build_heffte_trace.sh $ISA" >&2; exit 3; }
# env.sh puts ext/install/lib on LD_LIBRARY_PATH, and that holds the NON-tracing libheffte
# from the production install -- it wins over the binary's rpath, so the trace globals come
# up undefined at exec.  Pin the traced library ahead of it, and hand the variable to the
# ranks explicitly, since OpenMPI does not forward LD_LIBRARY_PATH on every launcher.
TRACELIB=$(dirname "$(dirname "$BIN")")
export LD_LIBRARY_PATH="$TRACELIB:${LD_LIBRARY_PATH:-}"
# ...and verify it, rather than trusting the ordering: loading the wrong libheffte is the
# one failure here that could produce plausible-looking timings with no trace files at all.
if ! ldd "$BIN" | grep -q "libheffte.so.2 => $TRACELIB/libheffte.so.2"; then
  echo "commfrac.sh: ABORT -- $BIN resolves libheffte to:" >&2
  ldd "$BIN" | grep libheffte >&2
  echo "expected $TRACELIB/libheffte.so.2 (the tracing build)" >&2
  exit 4
fi

OUT=results/$TAG
mkdir -p "$OUT"
NODES=${SLURM_JOB_NUM_NODES:-1}
# PHYSICAL cores per node, not nproc: these parts have hyperthreading on, so nproc reports
# 64 where OpenMPI counts 32 bindable cores and refuses "--bind-to core" beyond that.  Two
# MPI ranks sharing a core would also make the timing meaningless.
CPN=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
[ "${CPN:-0}" -ge 1 ] || { echo "commfrac.sh: could not determine physical core count" >&2; exit 5; }

{
  echo "# commfrac $TAG"
  echo "date: $(date -Is)   slurm_job: ${SLURM_JOB_ID:-none}   nodes: $NODES"
  echo "nodelist: ${SLURM_JOB_NODELIST:-$(hostname)}"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "cores/node: $CPN   isa build: $ISA"
  echo "heffte: $BIN"
  # The MPI stack is a first-order term for the transpose (the survey records 20-40% swings
  # from MPI choice alone), so pin down exactly which one produced these numbers.
  echo "mpi: $(mpirun --version 2>&1 | head -1)"
  echo "modules: $(module list 2>&1 | tr '\n' ' ' | sed 's/  */ /g')"
} | tee "$OUT/environment.txt"

# Rank ladder.  1 rank does NO MPI, so its reshape time is pure local pack/copy -- that is
# the control that makes the multi-rank reshape growth attributable to communication.
#
# Configurations are "totalranks:ranksPerNode", because the load-bearing comparison is not a
# rank count -- it is the SAME rank count with the transpose confined to one node versus
# crossing the fabric.  Anything else confounds "more ranks" with "slower transport".
CONFIGS="1:1 8:8"
[ "$CPN" -gt 8 ] && CONFIGS="$CONFIGS $CPN:$CPN"                    # one node, fully packed
if [ "$NODES" -ge 2 ]; then
  CONFIGS="$CONFIGS $CPN:$((CPN / 2))"                              # same ranks, 2 nodes
  CONFIGS="$CONFIGS $((CPN * 2)):$CPN"                              # both nodes fully packed
fi
ALGOS_NOTE="a2av p2p_pl"
echo "configs (ranks:perNode): $CONFIGS   grids: $GRIDS   algos: $ALGOS_NOTE" \
  | tee -a "$OUT/environment.txt"
ALGOS="a2av p2p_pl"

# Characterize the fabric BEFORE the FFT sweep, so no result from here can be misread as an
# FFT property when it is really a transport property.  This is not decoration: on the
# devel/prod nodes the same 32-rank 128^3 transpose cost 0.006 s within one node and 0.645 s
# across two, and without a direct latency/bandwidth number that 107x looks exactly like the
# "communication dominates" conclusion we are trying to test.
if [ "$NODES" -ge 2 ]; then
  if mpicc -O2 -o "$OUT/fabric_probe" fabric_probe.c 2>"$OUT/probe_build.log"; then
    mpirun -np 2 --map-by ppr:1:node -x LD_LIBRARY_PATH "$OUT/fabric_probe" \
      > "$OUT/fabric_probe.txt" 2>&1 || true
    mpirun -np $((CPN * NODES)) --map-by "ppr:$CPN:node" --bind-to core -x LD_LIBRARY_PATH \
      "$OUT/fabric_probe" >> "$OUT/fabric_probe.txt" 2>&1 || true
    grep -E "probe-summary|probe: pingpong|probe: alltoall|probe: [0-9]" "$OUT/fabric_probe.txt" \
      | tee -a "$OUT/environment.txt"
  else
    echo "fabric probe FAILED to build -- see $OUT/probe_build.log" | tee -a "$OUT/environment.txt"
  fi
fi

run_one() {
  local grid=$1 ranks=$2 ppn=$3 algo=$4
  local nnodes=$(( (ranks + ppn - 1) / ppn ))
  local d="$OUT/g${grid}_r${ranks}_n${nnodes}_${algo}"
  # heFFTe writes its per-rank trace logs into the CWD under a name keyed only on
  # backend/precision/grid, so each configuration needs its own directory or they collide.
  mkdir -p "$d"
  # ppr:<n>:node pins how many ranks land on each node, which is the whole point of the
  # ladder: 32:32 keeps the transpose inside one node, 32:16 puts it on the fabric.
  ( cd "$d" && mpirun -np "$ranks" --map-by "ppr:$ppn:node" --bind-to core -x LD_LIBRARY_PATH \
      "$BIN" fftw double "$grid" "$grid" "$grid" "-$algo" "-nruns$NRUNS" -no-error \
      > stdout.txt 2>&1 )
  local rc=$?
  local nlogs=$(ls "$d"/*.log 2>/dev/null | wc -l)
  # A run that produced no trace files is a FAILURE, not an empty result: make it loud,
  # because a silent zero here would read as "no communication cost".
  if [ "$nlogs" -ne "$ranks" ]; then
    echo "   FAIL grid=$grid ranks=$ranks nodes=$nnodes algo=$algo rc=$rc logs=$nlogs/$ranks" \
      | tee -a "$OUT/failures.txt"
    return 1
  fi
  echo "   ok   grid=$grid ranks=$ranks nodes=$nnodes algo=$algo  $(grep -i 'Time per run\|GFlops' "$d/stdout.txt" | tr '\n' ' ')"
}

for grid in $GRIDS; do
  for cfg in $CONFIGS; do
    ranks=${cfg%%:*}; ppn=${cfg##*:}
    [ "$ranks" -gt $((CPN * NODES)) ] && continue
    for algo in $ALGOS; do
      # 1 rank has no communication at all, so the algorithm knob is meaningless there:
      # run it once rather than producing two identical rows.
      [ "$ranks" = 1 ] && [ "$algo" != "a2av" ] && continue
      run_one "$grid" "$ranks" "$ppn" "$algo"
    done
  done
done

python3 parse_traces.py --tag "$TAG" | tee "$OUT/FRACTIONS.txt"
echo "== commfrac $TAG complete: $OUT =="
