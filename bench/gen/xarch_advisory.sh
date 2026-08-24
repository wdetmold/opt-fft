#!/bin/bash
# Cross-architecture ADVISORY sweep (docs/CAMPAIGN_GENERALIZE.md): after round N's Ice Lake
# leaderboard, rerun the acceptance suite at reduced sampling on another microarchitecture.
# Advisory only -- these numbers never enter a leaderboard (numbers from different machines
# are never compared directly; only ours-vs-lib RATIOS on the same machine travel).
# usage: xarch_advisory.sh <round-number> <clx|spr>
set -u
cd "$(dirname "$(readlink -f "$0")")"
N=$1; ARCH=$2
ROUND=xarch_${ARCH}_r$N
SEED=$((90260000 + N))
[ -d "results/$ROUND" ] && { echo "$ROUND already exists"; exit 0; }
JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"; chmod +x "$JOBSCRIPT"
mkdir -p "results/$ROUND"

if [ "$ARCH" = clx ]; then
  # a fresh exclusive Cascade Lake node from the devel partition; name must not match
  # the benchmark job pattern or the round runner stalls on it
  JOB=$(sbatch -p devel -J xarchhold -t 3:00:00 -o /dev/null --wrap "sleep 10800" | awk '{print $4}')
  [ -n "$JOB" ] || { echo "sbatch failed"; exit 1; }
  trap "scancel $JOB 2>/dev/null" EXIT
  for i in $(seq 1 60); do
    NODE=$(squeue -h -j "$JOB" -o %N 2>/dev/null); ST=$(squeue -h -j "$JOB" -o %T 2>/dev/null)
    [ "$ST" = RUNNING ] && [ -n "$NODE" ] && break; sleep 20
  done
  [ "$ST" = RUNNING ] || { echo "devel node never started"; exit 1; }
  echo "advisory $ROUND on $NODE (job $JOB)"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" \
    "cd '$(pwd)' && '$JOBSCRIPT' --round $ROUND --seed $SEED --runs 1 --samples 6" \
    > "results/$ROUND/sweep.out" 2>&1
  scancel "$JOB" 2>/dev/null; trap - EXIT
else
  # SPR spot-check on wallaby: implementers develop there, so run nice'd and accept noise
  echo "advisory $ROUND on wallaby (nice, reduced sampling)"
  ssh -o BatchMode=yes wallaby \
    "cd '$(pwd)' && nice -n 10 '$JOBSCRIPT' --round $ROUND --seed $SEED --runs 1 --samples 4" \
    > "results/$ROUND/sweep.out" 2>&1
fi
rm -f "$JOBSCRIPT"
python3 xarch_report_gen.py "gen_r$N" "$ROUND" | tee "results/$ROUND/XARCH_REPORT.txt"
python3 xarch_report_gen.py "gen_r$N" "$ROUND" --md > XARCH.md
echo "$ROUND done"
