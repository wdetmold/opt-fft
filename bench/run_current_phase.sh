#!/bin/bash
# Cron entry point for the whole project: run whichever competition phase we are in.
#
# The project runs three competitions in sequence, each with its own harness directory and
# its own independent records:
#
#   bench/geom   single-threaded CPU   rounds panel_rN
#   bench/mt     multicore CPU        rounds mt_rN
#   bench/gpu    one A100             rounds gpu_rN
#
# bench/PHASE names the current one. Each phase's series-completion hook builds the next
# phase and rewrites PHASE, so the chain advances with nobody watching. flock (in the cron
# entry) guarantees only one phase runs at a time.
set -u
cd "$(dirname "$(readlink -f "$0")")"
BENCH=$(pwd)

PHASE=$(cat "$BENCH/PHASE" 2>/dev/null || echo geom)
HARNESS=$BENCH/$PHASE
if [ ! -d "$HARNESS" ]; then
  echo "run_current_phase: PHASE says '$PHASE' but $HARNESS does not exist" >&2
  exit 2
fi
if [ -f "$BENCH/STOP_ALL" ]; then
  echo "run_current_phase: STOP_ALL present -- not starting $PHASE"
  exit 0
fi

# run_rounds.sh lives in the first harness and drives all of them via --harness.
exec "$BENCH/geom/run_rounds.sh" --harness "$HARNESS" --resume
