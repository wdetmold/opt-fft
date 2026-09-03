#!/bin/bash
# Hold-or-queue guard: keeps exactly one icehold RUNNING or PENDING. Unlike reserve.sh
# (which cancels its job if the node stays busy), a pending hold KEEPS OUR QUEUE POSITION.
# When the pending job starts, this writes RESERVATION so tryout/submit find the node.
export PATH=/opt/software/slurm-19.05.8.1-cuda-11.8/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")"
# 24 h, matching reserve.sh: a 4-round d1 campaign is about a day of implementer plus
# scoring time, and the previous 12 h hold expired mid-scoring.  axxxl caps at 48 h.  The
# sleep is DERIVED from the hours so the two cannot drift apart -- they are the same number.
HOURS=${FFT_ICE_HOURS:-24}
read -r JOB STATE NODE <<< $(squeue -h -u $USER -n icehold -t RUNNING,PENDING -o "%i %T %N" | head -1)
if [ -z "$JOB" ]; then
  JOB=$(sbatch -p axxxl -J icehold -t "${HOURS}:00:00" -o /dev/null \
        --wrap "sleep $((HOURS * 3600))" | awk '{print $4}')
  echo "$(date '+%F %T') queued new icehold $JOB (position held while nodes busy)"
  exit 0
fi
if [ "$STATE" = RUNNING ] && [ -n "$NODE" ]; then
  grep -q "RES_JOB=$JOB" RESERVATION 2>/dev/null || {
    printf 'RES_JOB=%s\nRES_NODE=%s\n' "$JOB" "$NODE" > RESERVATION
    echo "$(date '+%F %T') icehold $JOB running on $NODE -- RESERVATION updated"
  }
  date +%s > RESERVATION.heartbeat
fi
