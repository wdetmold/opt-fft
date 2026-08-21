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
# --cpu-freq=Performance where the site allows it: the devel nodes default to the
# powersave governor, which adds run-to-run spread to short measurements.
sbatch --job-name="fft-$ROUND" --partition="$PARTITION" --exclusive --nodes=1 \
       --cpu-freq=Performance \
       --time="$TIME" --output="results/$ROUND/slurm-%j.out" \
       --wrap="$JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
