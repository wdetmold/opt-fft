#!/bin/bash
# Score an Ice Lake round: on the reserved node, inside a scoring window that drains the
# implementers' core leases -- no slurm job at all on the primary path, so this panel never
# collides with the multicore panel's queue. Fallback: a whole-node axxxl job named
# ice-<round> (NOT fft-<round>: the other panel's runner treats fft-* as its own).
#
# usage: submit.sh --round TAG --seed N [--samples 8] [--runs 3] [--cases FILE] ...
set -eu
cd "$(dirname "$0")"
ROUND=""; SEED=1; EXTRA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --partition|--time) shift 2 ;;   # accepted for interface parity; unused on the primary path
    *) EXTRA="$EXTRA $1"; shift ;;
  esac
done
[ -n "$ROUND" ] || { echo "submit.sh: --round required" >&2; exit 2; }
# Env knobs the sweep reads, forwarded EXPLICITLY rather than relying on srun/ssh to
# propagate them: BACKENDS restricts which built binaries are measured, ALLOW_NO_PANEL
# permits a library-only run (the SOTA baseline table).  Both are empty for normal rounds.
ENVPFX=""
[ -n "${BACKENDS:-}" ] && ENVPFX="$ENVPFX BACKENDS='$BACKENDS'"
[ -n "${ALLOW_NO_PANEL:-}" ] && ENVPFX="$ENVPFX ALLOW_NO_PANEL='$ALLOW_NO_PANEL'"
mkdir -p "results/$ROUND"
JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"; chmod +x "$JOBSCRIPT"

if [ -f RESERVATION ] && ./reserve.sh --status >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . ./RESERVATION
  echo "scoring $ROUND on reserved Ice Lake node $RES_NODE"
  ./slot_lease.sh acquire-all --label "monitor:$ROUND"
  trap './slot_lease.sh release-all >/dev/null 2>&1' EXIT INT TERM
  # Run the sweep INSIDE the allocation via srun rather than ssh. pam_slurm_adopt has
  # been observed refusing ssh ("you have no active jobs on this node") on a node where
  # our job is demonstrably RUNNING with a live batch step -- which silently killed two
  # d1 rounds and a gen rescore. srun needs no pam adoption, so it is the primary path;
  # ssh stays as a fallback for the case where srun cannot get a step.
  SRUN=$(command -v srun || echo /opt/software/slurm-19.05.8.1-cuda-11.8/bin/srun)
  rc=1
  if [ -x "$SRUN" ] && [ -n "${RES_JOB:-}" ]; then
    "$SRUN" --jobid="$RES_JOB" bash -lc \
      "cd '$(pwd)' && env $ENVPFX '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA" \
      > "results/$ROUND/sweep.out" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "srun path failed (rc=$rc); falling back to ssh" >> "results/$ROUND/sweep.out"
  fi
  if [ $rc -ne 0 ]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" \
      "cd '$(pwd)' && env $ENVPFX '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA" \
      >> "results/$ROUND/sweep.out" 2>&1
    rc=$?
  fi
  ./slot_lease.sh release-all >/dev/null 2>&1; trap - EXIT INT TERM
  tail -3 "results/$ROUND/sweep.out"; exit $rc
fi

echo "no live reservation -- trying to claim one"
./reserve.sh --hours 8 >/dev/null 2>&1 && exec "$0" --round "$ROUND" --seed "$SEED" $EXTRA
echo "claim failed -- queueing a whole-node axxxl job"
sbatch --job-name="ice-$ROUND" --partition=axxxl --nodes=1 --exclusive --time=100 \
  --output="results/$ROUND/slurm-%j.out" \
  --wrap="env $ENVPFX $JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
