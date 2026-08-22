#!/bin/bash
# Hold an 8-GPU node for the GPU competition, so implementer agents get a GPU each.
#
# Why: this cluster is SelectType=select/linear, so slurm cannot hand out one GPU -- the
# only granularity is a whole node. Queueing one sweep job per round would also mean the
# panel never gets to iterate on real hardware. Instead we claim a node once with a
# placeholder job and then, because ssh to a node is permitted while you hold an allocation
# on it, agents run their own work there over ssh, one GPU each, coordinated by leases
# (see gpu_lease.sh).
#
#   ./reserve.sh                 claim a node for the default duration
#   ./reserve.sh --hours 10      claim it for longer
#   ./reserve.sh --status        where is it, and is it alive
#   ./reserve.sh --release       give it back
#
# The job is deliberately NOT named fft-* : the CPU phase's runner treats any queued
# fft-*/probe-* job as "a benchmark round is in flight" and would stall itself for as long
# as this reservation lives.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GPU=$(pwd)
RES=$GPU/RESERVATION
BEAT=$GPU/RESERVATION.heartbeat
PARTITION=${FFT_GPU_PARTITION:-a100l}
HOURS=${FFT_GPU_HOURS:-10}
ACTION=claim

while [ $# -gt 0 ]; do
  case "$1" in
    --hours) HOURS=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    --status) ACTION=status; shift ;;
    --release) ACTION=release; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

read_res() {
  [ -f "$RES" ] || return 1
  # shellcheck disable=SC1090
  . "$RES"
  [ -n "${RES_JOB:-}" ] && [ -n "${RES_NODE:-}" ]
}

alive() {
  read_res || return 1
  squeue -h -j "$RES_JOB" -o '%t' 2>/dev/null | grep -q '^R$'
}

case "$ACTION" in
status)
  if alive; then
    age=$(( $(date +%s) - $(cat "$BEAT" 2>/dev/null || echo 0) ))
    echo "reservation ALIVE: job $RES_JOB on $RES_NODE (heartbeat ${age}s ago)"
    echo "  time left: $(squeue -h -j "$RES_JOB" -o '%L' 2>/dev/null)"
    echo "  gpus:"
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" \
      'nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader' \
      2>/dev/null | sed 's/^/    /' || echo "    (ssh to $RES_NODE failed)"
    exit 0
  fi
  if read_res; then echo "reservation RECORDED but job $RES_JOB is not running"; else echo "no reservation"; fi
  exit 1
  ;;
release)
  if read_res; then
    scancel "$RES_JOB" 2>/dev/null && echo "cancelled reservation job $RES_JOB on $RES_NODE"
    rm -f "$RES" "$BEAT"
  else
    echo "no reservation to release"
  fi
  exit 0
  ;;
esac

if alive; then
  echo "reservation already alive: job $RES_JOB on $RES_NODE"
  exit 0
fi

# The payload records where it landed and then heartbeats, so the tooling can tell a live
# reservation from a stale file. It holds the node until cancelled or the time limit.
PAYLOAD=$GPU/.reservation_payload.sh
cat > "$PAYLOAD" <<'INNER'
#!/bin/bash
GPU=$(dirname "$(readlink -f "$0")")
{
  echo "RES_JOB=$SLURM_JOB_ID"
  echo "RES_NODE=$(hostname -s)"
  echo "RES_PARTITION=$SLURM_JOB_PARTITION"
  echo "RES_STARTED=$(date -Is)"
  echo "RES_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)"
} > "$GPU/RESERVATION"
# Heartbeat so a crashed or preempted reservation is detectable rather than silently stale.
while true; do
  date +%s > "$GPU/RESERVATION.heartbeat"
  sleep 60
done
INNER
chmod +x "$PAYLOAD"

jid=$(sbatch --parsable --job-name=gpuhold --partition="$PARTITION" --nodes=1 --exclusive \
             --time="${HOURS}:00:00" --output="$GPU/logs/reservation-%j.out" \
             --wrap="$PAYLOAD" 2>&1)
if ! echo "$jid" | grep -qE '^[0-9]+$'; then
  echo "reservation submission failed: $jid" >&2
  exit 1
fi
echo "reservation job $jid submitted to $PARTITION for ${HOURS}h; waiting for it to start"

for i in $(seq 1 120); do
  state=$(squeue -h -j "$jid" -o '%t' 2>/dev/null)
  if [ "$state" = "R" ]; then
    for j in $(seq 1 30); do [ -f "$RES" ] && break; sleep 2; done
    if read_res; then
      echo "reservation live: job $RES_JOB on $RES_NODE with ${RES_GPUS:-?} GPUs"
      mkdir -p "$GPU/leases"
      exit 0
    fi
  fi
  [ -z "$state" ] && { echo "reservation job $jid left the queue before starting" >&2; exit 1; }
  sleep 10
done
echo "reservation job $jid did not start within 20 minutes; it stays queued" >&2
exit 1
