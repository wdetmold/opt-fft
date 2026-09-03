#!/bin/bash
# $1 is "extra" for a node claimed only to shard grading across, absent for the primary node
# the implementers develop on.  It is a MODE, not a path: a path cannot be passed in, because
# sbatch --wrap runs in a shell where neither $GPU nor $SLURM_JOB_ID is defined yet, so an
# interpolated filename arrives as "/RESERVATION.extra.<jobid>" and the write fails on /.
GPU=$(dirname "$(readlink -f "$0")")
if [ "${1:-}" = extra ]; then
  TARGET=$GPU/RESERVATION.extra.$SLURM_JOB_ID
else
  TARGET=$GPU/RESERVATION
fi
{
  echo "RES_JOB=$SLURM_JOB_ID"
  echo "RES_NODE=$(hostname -s)"
  echo "RES_PARTITION=$SLURM_JOB_PARTITION"
  echo "RES_STARTED=$(date -Is)"
  echo "RES_GPUS=$({ lscpu -p=CORE 2>/dev/null | grep -vc "^#" || nproc; })"
} > "$TARGET"
# Heartbeat so a crashed or preempted reservation is detectable rather than silently stale.
# Only the primary hold owns the shared heartbeat file; an extra beats beside its own record.
while true; do
  case "$TARGET" in
    *RESERVATION) date +%s > "$GPU/RESERVATION.heartbeat" ;;
    *)            date +%s > "$TARGET.heartbeat" ;;
  esac
  sleep 60
done
