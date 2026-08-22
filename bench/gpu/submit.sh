#!/bin/bash
# Score a GPU round.
#
# Preferred path: the project holds an 8-GPU node (see reserve.sh), so the scored sweep runs
# there directly, inside a SCORING WINDOW that holds all eight leases -- no implementer is
# on the node while the numbers are taken. This is synchronous: it returns when the
# leaderboard exists, which is what the round runner waits for anyway.
#
# Fallback: no live reservation, so queue a whole-node job instead. (This cluster is
# select/linear: per-GPU --gres requests are rejected, so a node is the only granularity.)
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
mkdir -p "results/$ROUND"

JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"
chmod +x "$JOBSCRIPT"

# If the reservation died (time limit, preemption) try once to get it back before falling
# back to the queue: the phase may run longer than one reservation's walltime.
if ! ./reserve.sh --status >/dev/null 2>&1; then
  echo "no live reservation -- attempting to claim a node"
  ./reserve.sh --hours "${FFT_GPU_HOURS:-6}" >/dev/null 2>&1 || true
fi

if [ -f RESERVATION ] && ./reserve.sh --status >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . ./RESERVATION
  echo "scoring $ROUND on the reserved node $RES_NODE (job $RES_JOB)"
  ./gpu_lease.sh acquire-all --label "monitor:$ROUND"
  trap './gpu_lease.sh release-all >/dev/null 2>&1' EXIT INT TERM
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" \
    "cd '$(pwd)' && CUDA_VISIBLE_DEVICES=0 '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA" \
    > "results/$ROUND/sweep.out" 2>&1
  rc=$?
  ./gpu_lease.sh release-all >/dev/null 2>&1
  trap - EXIT INT TERM
  tail -3 "results/$ROUND/sweep.out"
  exit $rc
fi

echo "no live reservation -- queueing a whole-node job instead"
sbatch --job-name="fft-$ROUND" --partition="$PARTITION" --nodes=1 --exclusive \
       --time="$TIME" --output="results/$ROUND/slurm-%j.out" \
       --wrap="$JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
