#!/bin/bash
# Submit a GPU benchmark round. One A100, requested explicitly, on a partition that has
# them. Not --exclusive: an 8-GPU node reserved whole for a single-GPU measurement would be
# antisocial, and a GPU's own clocks and memory are not shared with the other GPUs. What
# does bleed across is host memory bandwidth and PCIe, which is why H2D/D2H are reported
# separately and excluded from the scored number.
#
# usage: submit.sh --round TAG --seed N [--partition a100l] [--time 120] [extra sweep args]
set -eu
cd "$(dirname "$0")"
PARTITION=a100l; TIME=120; ROUND=""; SEED=1; EXTRA=""
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

# A copy, so editing sweep.sh cannot corrupt a running job, and on the shared filesystem,
# because /tmp is node-local.
JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

sbatch --job-name="fft-$ROUND" --partition="$PARTITION" --nodes=1 --gres=gpu:1 \
       --cpus-per-task=8 --time="$TIME" --output="results/$ROUND/slurm-%j.out" \
       --wrap="$JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
