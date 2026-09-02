#!/bin/bash
# Queue a comm-fraction measurement.  Unlike the panel harnesses this takes a plain slurm
# allocation rather than a reservation: an MPI job spanning two nodes needs ONE --nodes=2
# allocation and cannot be assembled from two independent single-node holds.
#
# usage: submit.sh --tag TAG --nodes N --partition P [--time MIN] [--isa avx512|avx2]
#                  [--nruns 5] [--grids "256 512"] [--nodelist LIST]
#
# Partition guidance for this cluster:
#   axxxl  -- 2 Ice Lake nodes (the hardware every other campaign here is measured on).
#             A two-node run needs BOTH, so it cannot coexist with a single-node hold.
#   devel  -- 2 idle nodes, 1 h cap, prod-class CPU.  Right for validating the multi-node
#             path and for a first (non-Ice-Lake) data point.
#   prod/long -- 22 shared nodes; check `squeue` for queued multi-node jobs before taking any,
#             since prod and long draw on the same physical machines.
set -eu
cd "$(dirname "$0")"
TAG=""; NODES=1; PART=devel; TIME=55; ISA=""; NRUNS=5; GRIDS="256 512"; NODELIST=""; PML=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tag) TAG=$2; shift 2 ;;
    --nodes) NODES=$2; shift 2 ;;
    --partition) PART=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --isa) ISA=$2; shift 2 ;;
    --nruns) NRUNS=$2; shift 2 ;;
    --grids) GRIDS=$2; shift 2 ;;
    --nodelist) NODELIST=$2; shift 2 ;;
    --pml) PML=$2; shift 2 ;;
    *) echo "submit.sh: unknown argument $1" >&2; exit 2 ;;
  esac
done
[ -n "$TAG" ] || { echo "submit.sh: --tag required" >&2; exit 2; }
# heFFTe does NOT runtime-dispatch its AVX-512 kernels, so the ISA must match the partition
# or the binary SIGILLs.  Default from the partition rather than making the caller remember.
if [ -z "$ISA" ]; then
  case "$PART" in
    axxxl|a100l|a100r) ISA=avx512 ;;
    *)                 ISA=avx2   ;;
  esac
fi
SBATCH=$(command -v sbatch || echo /opt/software/slurm-19.05.8.1-cuda-11.8/bin/sbatch)
mkdir -p "results/$TAG"
# Snapshot the payload, the way bench/d1/submit.sh does.  bash reads a script incrementally,
# so editing commfrac.sh while a job is executing it kills the job mid-sweep -- the first run
# here died with "error reading input file: Stale file handle" after all 18 configurations had
# completed, losing only the parse step but for no good reason.
JOBSCRIPT=$(pwd)/.commfrac_snapshot_$TAG.sh
cp commfrac.sh "$JOBSCRIPT"; chmod +x "$JOBSCRIPT"
# Job name deliberately avoids the panels' prefixes: bench/geom/run_rounds.sh treats a queued
# job carrying its own prefix as "a round is in flight" and would stall itself behind this.
set -x
"$SBATCH" --job-name="commfrac" --partition="$PART" --nodes="$NODES" --exclusive \
  --time="$TIME" ${NODELIST:+--nodelist="$NODELIST"} \
  --output="$(pwd)/results/$TAG/slurm-%j.out" \
  --wrap="$JOBSCRIPT --tag $TAG --isa $ISA --nruns $NRUNS --grids '$GRIDS' ${PML:+--pml $PML}"
