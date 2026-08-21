#!/bin/bash
# Submit a benchmark round to an ISOLATED node (--exclusive), so timings are not
# polluted by other work.  Everything -- build, data generation, timing -- happens there.
#
# usage: submit.sh --round TAG --seed N [--partition devel] [--time 60] [extra sweep args]
set -eu
cd "$(dirname "$0")"
PARTITION=devel; TIME=60; ROUND=""; SEED=1; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --partition) PARTITION=$2; shift 2 ;;
    --time) TIME=$2; shift 2 ;;
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    *) EXTRA="$EXTRA $1"; shift ;;
  esac
done
[ -n "$ROUND" ] || { echo "submit.sh: --round is required" >&2; exit 2; }
mkdir -p results/$ROUND

# Submit a COPY of sweep.sh: bash reads a script incrementally, so editing sweep.sh while
# a job is executing it corrupts that job mid-run -- that is what invalidated round
# sota_r2.  The copy is what the job runs, so later edits are harmless.
#
# The copy must live on the SHARED filesystem and in THIS directory: /tmp is local to each
# node (a /tmp copy gives the job "not found", exit 127), and sweep.sh locates its own
# working directory with dirname "$0", so it has to sit beside the real one.
JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

# --cpu-freq=Performance where the site allows it: the devel nodes default to the
# powersave governor, which adds run-to-run spread to short measurements.
sbatch --job-name="fft-$ROUND" --partition="$PARTITION" --exclusive --nodes=1 \
       --cpu-freq=Performance \
       --time="$TIME" --output="results/$ROUND/slurm-%j.out" \
       --wrap="$JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
