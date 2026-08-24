#!/bin/bash
# Cron guard for the cross-arch advisories: CLX after r4 and r6 leaderboards, one SPR
# spot-check after r5. Stateless -- fires once per target because the advisory refuses
# to rerun when its results dir exists.
cd "$(dirname "$(readlink -f "$0")")"
for N in 4 6; do
  if [ -f "results/gen_r$N/leaderboard.txt" ] && [ ! -d "results/xarch_clx_r$N" ]; then
    ./xarch_advisory.sh "$N" clx; exit
  fi
done
if [ -f "results/gen_r5/leaderboard.txt" ] && [ ! -d "results/xarch_spr_r5" ]; then
  ./xarch_advisory.sh 5 spr
fi
