#!/bin/bash
GPU=$(dirname "$(readlink -f "$0")")
{
  echo "RES_JOB=$SLURM_JOB_ID"
  echo "RES_NODE=$(hostname -s)"
  echo "RES_PARTITION=$SLURM_JOB_PARTITION"
  echo "RES_STARTED=$(date -Is)"
  echo "RES_GPUS=$({ lscpu -p=CORE 2>/dev/null | grep -vc "^#" || nproc; })"
} > "$GPU/RESERVATION"
# Heartbeat so a crashed or preempted reservation is detectable rather than silently stale.
while true; do
  date +%s > "$GPU/RESERVATION.heartbeat"
  sleep 60
done
