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
# sweep.out is APPENDED to across attempts: a round can now be resumed after losing its
# node, and truncating the log would erase the record of what the earlier attempt measured.
{ echo; echo "===== submit.sh attempt $(date -Is) round=$ROUND seed=$SEED extra=$EXTRA ====="; } \
  >> "results/$ROUND/sweep.out"
JOBSCRIPT=$(pwd)/.sweep_snapshot_$ROUND.sh
cp sweep.sh "$JOBSCRIPT"; chmod +x "$JOBSCRIPT"

if [ -f RESERVATION ] && ./reserve.sh --status >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  . ./RESERVATION
  # ---- grading nodes -------------------------------------------------------------------
  # Grading is sharded across every node the panel holds.  Default is to use BOTH axxxl
  # nodes: the two were calibrated against each other on 244 identical library cells and
  # agree to 0.03% at the median, and in any case each cell's panel-vs-library comparison
  # happens entirely on one node, so a shard boundary cannot bias a gain ratio.
  # FFT_GRADE_SHARDS=1 forces the old single-node behaviour.
  WANT_SHARDS=${FFT_GRADE_SHARDS:-2}
  if [ "$WANT_SHARDS" -gt 1 ]; then
    ./reserve.sh --extra >/dev/null 2>&1 || true
    for _ in $(seq 1 20); do            # a just-submitted hold needs a moment to start
      ls RESERVATION.extra.* >/dev/null 2>&1 && break
      sleep 15
    done
  fi
  GJOBS="$RES_JOB"; GNODES="$RES_NODE"
  for x in RESERVATION.extra.*; do
    case "$x" in *'*'*) continue ;; esac
    [ -f "$x" ] || continue
    xj=$(sed -n 's/^RES_JOB=//p' "$x"); xn=$(sed -n 's/^RES_NODE=//p' "$x")
    [ -n "$xj" ] || continue
    squeue -h -j "$xj" -o '%t' 2>/dev/null | grep -q '^R$' || continue
    [ "${#GJOBS}" -gt 0 ] && GJOBS="$GJOBS $xj" && GNODES="$GNODES $xn"
  done
  NSHARD=$(echo $GNODES | wc -w)
  [ "$NSHARD" -gt "$WANT_SHARDS" ] && NSHARD=$WANT_SHARDS
  echo "scoring $ROUND across $NSHARD node(s): $GNODES"
  ./slot_lease.sh acquire-all --label "monitor:$ROUND"
  trap './slot_lease.sh release-all >/dev/null 2>&1' EXIT INT TERM
  # Run the sweep INSIDE the allocation via srun rather than ssh. pam_slurm_adopt has
  # been observed refusing ssh ("you have no active jobs on this node") on a node where
  # our job is demonstrably RUNNING with a live batch step -- which silently killed two
  # d1 rounds and a gen rescore. srun needs no pam adoption, so it is the primary path;
  # ssh stays as a fallback for the case where srun cannot get a step.
  SRUN=$(command -v srun || echo /opt/software/slurm-19.05.8.1-cuda-11.8/bin/srun)
  rc=1
  if [ -x "$SRUN" ] && [ -n "${RES_JOB:-}" ] && [ "$NSHARD" -gt 1 ]; then
    # Deal the cases out in DESCENDING size order, alternating nodes.  cases.txt is written
    # smallest-first and the cost is dominated by its tail -- the large primes, and FFTW
    # patient planning that alone takes ~58 s per run at L=65537 -- so a contiguous split
    # would give one node nearly all the work and save nothing.
    CF=$(echo "$EXTRA" | sed -n 's/.*--cases \([^ ]*\).*/\1/p'); CF=${CF:-cases.txt}
    i=0
    for sh in $(seq 1 "$NSHARD"); do : > "results/$ROUND/.shard_$sh.cases"; done
    # Deal by DESCENDING estimated cost, not by L: sorting on L alone is a stable sort, so
    # each size's four regimes stay adjacent, the alternation lines up with that period, and
    # every single-call cell lands on one node while every chained cell lands on the other --
    # which is both wildly unbalanced (a chained cell runs m transforms per call) and puts a
    # whole regime on one machine for no reason.  Cost proxy is the transform count times
    # the per-transform work, L*log2(L)*B*m, which is what per-call time is proportional to.
    # Balance the shards by ESTIMATED WALL TIME, from the previous round's own measured
    # per-call and setup times where they exist (see shard_cases.py for why the obvious
    # proxies do not work).  Sharding is only a speedup if the shards finish together.
    PREVR="${ROUND%r*}r$(( ${ROUND##*r} - 1 ))"
    [ -d "results/$PREVR" ] || PREVR=""
    ./shard_cases.py --cases "$CF" --shards "$NSHARD" --runs 3 --samples 12 \
        ${PREVR:+--prev-round $PREVR} 2>>"results/$ROUND/sweep.out" \
      | grep -v '^#' \
      | while read -r sh c; do echo "$c" >> "results/$ROUND/.shard_$sh.cases"; done
    pids=""; sh=0
    for gj in $GJOBS; do
      sh=$((sh+1)); [ "$sh" -le "$NSHARD" ] || break
      gn=$(echo $GNODES | cut -d' ' -f$sh)
      echo "   shard $sh -> $gn ($(wc -l < "results/$ROUND/.shard_$sh.cases") cells)"
      "$SRUN" --jobid="$gj" bash -lc \
        "cd '$(pwd)' && env $ENVPFX '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA \
           --cases results/$ROUND/.shard_$sh.cases --shard $sh --no-board" \
        >> "results/$ROUND/sweep_$sh.out" 2>&1 &
      pids="$pids $!"
    done
    rc=0
    for pd in $pids; do wait "$pd" || rc=$?; done
    # One board over the merged per-cell JSONs, after every shard has finished.
    [ $rc -eq 0 ] && python3 leaderboard.py --round "$ROUND" \
      | tee "results/$ROUND/leaderboard.txt" >> "results/$ROUND/sweep.out"
    cat "results/$ROUND"/sweep_*.out >> "results/$ROUND/sweep.out" 2>/dev/null
  elif [ -x "$SRUN" ] && [ -n "${RES_JOB:-}" ]; then
    "$SRUN" --jobid="$RES_JOB" bash -lc \
      "cd '$(pwd)' && env $ENVPFX '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA" \
      >> "results/$ROUND/sweep.out" 2>&1
    rc=$?
    [ $rc -ne 0 ] && echo "srun path failed (rc=$rc); falling back to ssh" >> "results/$ROUND/sweep.out"
  fi
  if [ $rc -ne 0 ] && [ "$NSHARD" -le 1 ]; then
    ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" \
      "cd '$(pwd)' && env $ENVPFX '$JOBSCRIPT' --round $ROUND --seed $SEED $EXTRA" \
      >> "results/$ROUND/sweep.out" 2>&1
    rc=$?
  fi
  ./slot_lease.sh release-all >/dev/null 2>&1; trap - EXIT INT TERM
  tail -3 "results/$ROUND/sweep.out"; exit $rc
fi

echo "no live reservation -- trying to claim one"
./reserve.sh >/dev/null 2>&1 && exec "$0" --round "$ROUND" --seed "$SEED" $EXTRA
echo "claim failed -- queueing a whole-node axxxl job"
sbatch --job-name="ice-$ROUND" --partition=axxxl --nodes=1 --exclusive --time=100 \
  --output="results/$ROUND/slurm-%j.out" \
  --wrap="env $ENVPFX $JOBSCRIPT --round $ROUND --seed $SEED $EXTRA"
