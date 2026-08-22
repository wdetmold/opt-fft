#!/bin/bash
# Unattended development -> timing cycle for the FFT optimization panel.
#
# One round =  12 implementer agents revise their code (each in its own headless claude
#              session, developing and timing on wallaby)
#           -> a purely mechanical timing pass on the exclusive benchmark node
#           -> one monitor agent interprets the leaderboard and picks what to keep
#           -> promote exemplars, commit, advance the state file.
#
# The timing pass is deliberately plain shell: the numbers must not depend on a model
# behaving well. Only the two judgement-shaped jobs (writing code, interpreting results)
# use an agent.
#
#   ./run_rounds.sh --rounds 6            start a fresh series of 6 rounds
#   ./run_rounds.sh --resume              continue where the state file says (cron uses this)
#   touch STOP                            stop cleanly after the current phase
#
# Survives logout: launch with setsid+nohup, and a cron watchdog restarts it if the
# machine or the process dies. Both use flock, so a second copy can never run.
set -u

# Re-exec from a private copy. A round takes hours, and bash reads a script incrementally,
# so editing this file while it runs makes the running process execute shifted content --
# which has already bitten this project twice. Working from a snapshot means edits only
# take effect at the next cron tick, which is exactly the desired semantics.
if [ "${FFT_ROUNDS_REEXEC:-0}" != "1" ]; then
  SELF=$(readlink -f "$0")
  FFT_ROUNDS_HOME=$(dirname "$SELF")
  COPY=$(mktemp "${TMPDIR:-/tmp}/fft_rounds_XXXXXX.sh") || exit 2
  cp "$SELF" "$COPY" && chmod +x "$COPY" || exit 2
  export FFT_ROUNDS_REEXEC=1 FFT_ROUNDS_HOME FFT_ROUNDS_COPY="$COPY"
  exec "$COPY" "$@"
fi
[ -n "${FFT_ROUNDS_COPY:-}" ] && trap 'rm -f "$FFT_ROUNDS_COPY"' EXIT

cd "${FFT_ROUNDS_HOME:-$(dirname "$(readlink -f "$0")")}"
GEOM=$(pwd)
ROOT=$(readlink -f ../..)

# cron gives almost no environment, so nothing here may rely on an interactive shell.
# The slurm binaries are NOT in /usr/bin on this cluster -- leaving them off PATH makes
# sbatch fail silently under cron, which kills the timing phase while the development
# phase carries on looking healthy.
SLURM_BIN=${SLURM_BIN:-/opt/software/slurm-19.05.8.1-cuda-11.8/bin}
export PATH="/home/lqcd/wdetmold/.local/bin:$SLURM_BIN:/usr/local/bin:/usr/bin:/bin"
export HOME=${HOME:-/home/lqcd/wdetmold}
CLAUDE=${CLAUDE:-/home/lqcd/wdetmold/.local/bin/claude}

for tool in "$CLAUDE" "$SLURM_BIN/sbatch" "$SLURM_BIN/squeue"; do
  [ -x "$tool" ] || { echo "run_rounds.sh: required tool missing: $tool" >&2; exit 2; }
done

# Implementers write and tune the kernels; the monitor reads results and judges. They are
# different jobs, so they get different models.
IMPL_MODEL=${FFT_IMPL_MODEL:-claude-fable-5}
MONITOR_MODEL=${FFT_MONITOR_MODEL:-claude-opus-5}
JOBS=${FFT_JOBS:-6}                 # implementer agents in flight at once
AGENT_TIMEOUT=${FFT_AGENT_TIMEOUT:-5400}    # 90 min per implementer
TIMING_TIMEOUT=${FFT_TIMING_TIMEOUT:-7200}  # 2 h for the benchmark job to produce a leaderboard
STATE=$GEOM/results/.rounds_state
STOPFILE=$GEOM/STOP
LOGDIR=$GEOM/logs
mkdir -p "$LOGDIR" "$GEOM/results"

ROUNDS=6
RESUME=0
DRYRUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --rounds) ROUNDS=$2; shift 2 ;;
    --resume) RESUME=1; shift ;;
    --dry-run) DRYRUN=1; shift ;;
    --jobs) JOBS=$2; shift 2 ;;
    --impl-model) IMPL_MODEL=$2; shift 2 ;;
    --monitor-model) MONITOR_MODEL=$2; shift 2 ;;
    --model) IMPL_MODEL=$2; MONITOR_MODEL=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOGDIR/rounds.log"; }

# ---------------------------------------------------------------- state

# state file: "<next_round_index> <last_round_index_of_series>"
if [ "$RESUME" = 1 ]; then
  if [ ! -f "$STATE" ]; then log "no state file at $STATE; nothing to resume"; exit 0; fi
  read -r NEXT LAST < "$STATE"
else
  NEXT=2                          # panel_r1 was the manual first round
  [ -f "$STATE" ] && read -r EXISTING _ < "$STATE" && NEXT=$EXISTING
  LAST=$((NEXT + ROUNDS - 1))
  printf '%s %s\n' "$NEXT" "$LAST" > "$STATE"
fi

log "=== run_rounds: next=panel_r$NEXT last=panel_r$LAST jobs=$JOBS ==="
log "    implementers: $IMPL_MODEL    monitor: $MONITOR_MODEL"

# Never collide with a benchmark round already in flight -- including one submitted from
# outside this script. Two sweeps would build in the same per-host directory and contend
# for the node, so both sets of numbers would be junk.
round_in_flight() {
  squeue -h -u "$(whoami)" -o '%j' 2>/dev/null | grep -qE '^(fft-|probe-)'
}

wait_for_quiet() {
  local waited=0 cap=${1:-10800}
  while round_in_flight; do
    if [ "$waited" = 0 ]; then log "a benchmark job is already in flight; waiting for it"; fi
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$cap" ]; then
      log "still busy after ${cap}s -- leaving it to the next cron tick"
      return 1
    fi
  done
  [ "$waited" -gt 0 ] && log "node quiet after ${waited}s"
  return 0
}

stop_requested() {
  if [ -f "$STOPFILE" ]; then
    log "STOP file present -- halting cleanly (remove $STOPFILE to continue)"
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------- per-round source dirs

# Round N's implementers work in impl_N, so every round's code is preserved instead of
# being overwritten in place. `impl` is a symlink to the current round's directory, which
# means the Makefile, sweep.sh, tryout.sh, promote.sh and probe_node.sh all keep working
# unchanged -- they glob impl/*.c and resolve through the link.
#
# This exists because panel_r1's sources were lost: its monitor died before promoting
# anything, and the next round's implementers then rewrote all eleven files in place.
setup_impl_dir() {
  local round=$1
  local n=${round##*_r}
  local prev=$((n - 1))
  local dir="$GEOM/impl_$n"

  if [ ! -L "$GEOM/impl" ]; then
    # First cutover: rename the live directory to the round that produced its contents,
    # which preserves that round's code, then branch this round off it.
    if [ -d "$GEOM/impl" ]; then
      if [ -d "$GEOM/impl_$prev" ]; then
        cp "$GEOM"/impl/*.c "$GEOM/impl_$prev/" 2>/dev/null || true
        rm -rf "$GEOM/impl"
      else
        mv "$GEOM/impl" "$GEOM/impl_$prev"
        log "preserved the sources that produced panel_r$prev in impl_$prev"
      fi
    fi
    mkdir -p "$dir"
    cp "$GEOM/impl_$prev"/*.c "$dir/" 2>/dev/null || true
    ln -sfn "impl_$n" "$GEOM/impl"
  else
    if [ ! -d "$dir" ]; then
      mkdir -p "$dir"
      cp "$GEOM/impl_$prev"/*.c "$dir/" 2>/dev/null || true
    fi
    ln -sfn "impl_$n" "$GEOM/impl"
  fi
  log "$round works in impl_$n ($(ls "$dir"/*.c 2>/dev/null | wc -l) sources carried over from impl_$prev)"
}

# ---------------------------------------------------------------- context for implementers

# What previous generations did. Every implementer is pointed at ALL of it, not just its
# own lineage: a technique that won at one geometry frequently transfers.
build_context() {
  local round=$1 out=$2
  {
    echo "# What previous generations produced (round $round is the current one)"
    echo
    echo "## Leaderboards from earlier rounds"
    local found=0
    for lb in $(ls -1 "$GEOM"/results/*/leaderboard.txt 2>/dev/null | sort); do
      echo "  $lb"
      found=1
    done
    [ $found = 0 ] && echo "  (none yet)"
    echo
    echo "## Strategy records -- every implementation's own account of what it tried"
    if ls "$GEOM"/strategies/*.md >/dev/null 2>&1; then
      for f in "$GEOM"/strategies/*.md; do
        printf '  %-52s %s lines\n' "$f" "$(wc -l < "$f")"
      done
    else
      echo "  (none yet)"
    fi
    echo
    echo "## Promoted exemplars -- code kept from earlier rounds because it was worth keeping"
    if ls -d "$GEOM"/exemplars/*/ >/dev/null 2>&1; then
      for d in "$GEOM"/exemplars/*/; do
        echo "  $d"
        sed -n '1,12p' "$d/NOTES.md" 2>/dev/null | sed 's/^/      /'
      done
    else
      echo "  (none yet)"
    fi
    echo
    echo "## Current standings (most recent leaderboard)"
    local latest
    latest=$(ls -1t "$GEOM"/results/*/leaderboard.txt 2>/dev/null | head -1)
    if [ -n "$latest" ]; then sed -n '1,200p' "$latest"; else echo "  (none yet)"; fi
  } > "$out"
}

# ---------------------------------------------------------------- phase 1: implementers

run_implementers() {
  local round=$1 ctx=$2
  local pdir=$GEOM/results/$round/prompts
  local adir=$GEOM/results/$round/agents
  mkdir -p "$pdir" "$adir"

  # The roster is whatever is on disk: rounds after the first revise existing entries.
  local roster
  roster=$(cd "$GEOM/impl" && ls *.c 2>/dev/null | sed 's/\.c$//' | grep -v '^baseline_matrix$')
  [ -n "$roster" ] || { log "no implementations found in impl/ -- aborting round"; return 1; }
  log "roster ($(echo "$roster" | wc -w)): $(echo $roster)"

  local running=0
  for name in $roster; do
    stop_requested && break
    local L
    L=$(echo "$name" | sed -n 's/^L\([0-9]\+\)_.*/\1/p')
    [ -n "$L" ] || { log "skipping $name: cannot infer L from the name"; continue; }

    cat > "$pdir/$name.txt" <<PROMPT
You are the implementer responsible for $name.c in the FFT optimization panel, working on
round $round. Your geometry is L = $L (a cube ${L}^3 of complex doubles, batched).

Read first:
  $GEOM/PANEL_BRIEF.md            the rules, the contract, where to develop, how you are timed
  $GEOM/impl/$name.c              YOUR current implementation -- this round you improve it
  $GEOM/strategies/$name.md       YOUR own record of what you have already tried
  $ROOT/docs/LITERATURE.md        the cited corpus, with a per-size strategy table

Then read what OTHER generations did, listed in:
  $pdir/../context.md

That context file is the point of this round. You are explicitly encouraged to study the
other implementations' strategy records and promoted exemplars -- INCLUDING those for other
geometries and those of your rivals on your own geometry -- and to take any technique that
looks like it would help you. This is not independent competition any more; it is
cumulative. If you adopt an idea from another entry, say so plainly in your strategy record
and name the entry you took it from. If another entry's record shows an approach already
failed and why, do not spend this round rediscovering that.

Your job this round:
  1. Understand where your implementation currently stands (your record, and the standings
     at the end of the context file).
  2. Make it faster, without breaking correctness. Relative L2 error against numpy must
     stay below 1e-12; a fast wrong answer scores nothing.
  3. Iterate with:  cd $GEOM && ./tryout.sh --on wallaby $name $L <batch>
     wallaby is a near-idle Sapphire Rapids node with full AVX-512 that shares this
     filesystem. Use it constantly. Check both B=1 and a large batch.
  4. Do NOT submit slurm jobs. The exclusive benchmark node belongs to the monitor; it will
     measure you there and the result will appear in the next leaderboard.
  5. APPEND a new "Round $round" section to $GEOM/strategies/$name.md: what you changed,
     the operation count, what you measured on wallaby, what you tried that did not work
     with the number that killed it, anything you borrowed from another entry, and what you
     would do next. Never overwrite earlier rounds -- the history is the point.

Write only these two files: impl/$name.c and strategies/$name.md. Do not touch driver.c,
fft3d_api.h, the Makefile, the sweep/submit/tryout scripts, another implementer's files, or
anything under python/. Do not run 'make' in $GEOM (it is shared; tryout.sh is safe).

When you are done, reply with one line: the technique you ended on, your best wallaby
microseconds per transform at B=1 and batched, and your rel L2 error.
PROMPT

    if [ "$DRYRUN" = 1 ]; then log "  [dry-run] would launch implementer $name (L=$L)"; continue; fi

    ( timeout "$AGENT_TIMEOUT" "$CLAUDE" -p "$(cat "$pdir/$name.txt")" \
        --model "$IMPL_MODEL" --allowedTools Bash Read Write Edit Glob Grep \
        --permission-mode acceptEdits > "$adir/$name.log" 2>&1
      printf '%s exit=%s\n' "$name" "$?" >> "$adir/exits.txt" ) &

    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
    sleep 3      # stagger the launches
  done
  wait
  log "implementers finished: $(grep -c . "$adir/exits.txt" 2>/dev/null || echo 0) reported"
  grep -v 'exit=0' "$adir/exits.txt" 2>/dev/null | sed 's/^/    nonzero: /' | tee -a "$LOGDIR/rounds.log"
  return 0
}

# ---------------------------------------------------------------- phase 2: timing (no LLM)

run_timing() {
  local round=$1 seed=$2
  if [ "$DRYRUN" = 1 ]; then log "  [dry-run] would submit timing round $round (seed $seed)"; return 1; fi
  log "submitting timing round $round (seed $seed) to the exclusive benchmark node"
  rm -f "$GEOM/results/$round/leaderboard.txt"
  if ! ./submit.sh --round "$round" --seed "$seed" --partition devel --time 55 \
        --samples 12 --runs 3 >> "$LOGDIR/rounds.log" 2>&1; then
    log "sbatch submission failed; retrying once in 120s"
    sleep 120
    ./submit.sh --round "$round" --seed "$seed" --partition devel --time 55 \
        --samples 12 --runs 3 >> "$LOGDIR/rounds.log" 2>&1 || {
      log "submission failed twice -- skipping the timing pass for $round"; return 1; }
  fi

  local waited=0
  while [ ! -f "$GEOM/results/$round/leaderboard.txt" ]; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$TIMING_TIMEOUT" ]; then
      log "timing round $round did not produce a leaderboard within ${TIMING_TIMEOUT}s"
      squeue -u "$(whoami)" -o '%i %j %t %M' 2>/dev/null | tee -a "$LOGDIR/rounds.log"
      return 1
    fi
    if [ $((waited % 600)) = 0 ]; then log "  still waiting on $round (${waited}s)"; fi
  done
  log "leaderboard for $round is in"
  python3 leaderboard.py --round "$round" \
      --markdown "$GEOM/results/$round/leaderboard.md" >/dev/null 2>&1
  return 0
}

# ---------------------------------------------------------------- phase 3: monitor verdict

run_monitor() {
  local round=$1
  local mprompt=$GEOM/results/$round/monitor_prompt.txt
  cat > "$mprompt" <<PROMPT
You are the monitor for round $round of the FFT optimization panel. The timing pass has
already run on the exclusive benchmark node (Xeon Gold 5218, Cascade Lake, AVX-512) and
written its results. Your job is judgement, not measurement.

Read:
  $GEOM/results/$round/leaderboard.txt      the measured standings, with correctness
  $GEOM/results/$round/environment.txt      the machine it was measured on
  $GEOM/results/$round/failures.txt         (may not exist) entries that crashed or hung
  $GEOM/results/$round/build_errors.txt     (may not exist) entries that failed to build
  $GEOM/strategies/*.md                     what each implementer says it did
  $ROOT/docs/CURATION.md                    the criteria for what is worth keeping

Produce $GEOM/results/$round/VERDICT.md containing:
  1. Headline per geometry (L = 6, 8, 17, 36): fastest correct panel entry versus the best
     library, for BOTH the non-batched and the batched cases, with the numbers.
  2. What changed since the previous round, per geometry. Did anything regress?
  3. Any entry that failed correctness, failed to build, crashed, or is missing -- named,
     with the evidence. Be adversarial here: a fast wrong answer must not survive.
  4. Where an implementer's own claimed number is far from what was measured, note it and
     attribute it to the machine difference where that is the plausible cause (they develop
     on Sapphire Rapids with full-clock AVX-512 and 2 MB L2; you score on Cascade Lake with
     downclocked AVX-512 and 1 MB L2 -- MKL alone spans 2.9x between those machines).
  5. Which open question from $ROOT/docs/LITERATURE.md section 4 this round moved, and how.
  6. The single highest-value thing the next round should attack, per geometry.

Then decide what to keep, using the criteria in docs/CURATION.md: the fastest correct entry
per geometry always; a structurally different runner-up when it is close; instructive
failures whose record documents the number that killed them; anything that beat a library.

Finish your reply with exactly one line of this form, naming the implementations to keep:

PROMOTE: name1 name2 name3

Use the implementation names as they appear in impl/ (without .c). If nothing is worth
promoting, write "PROMOTE: none".
PROMPT

  local mlog=$GEOM/results/$round/monitor.log
  if [ "$DRYRUN" = 1 ]; then log "  [dry-run] would run the monitor agent"; return 0; fi
  timeout "$AGENT_TIMEOUT" "$CLAUDE" -p "$(cat "$mprompt")" \
      --model "$MONITOR_MODEL" --allowedTools Bash Read Write Edit Glob Grep \
      --permission-mode acceptEdits > "$mlog" 2>&1
  log "monitor exit=$? (log: $mlog)"

  local promote
  promote=$(grep -h '^PROMOTE:' "$mlog" 2>/dev/null | tail -1 | sed 's/^PROMOTE:[[:space:]]*//')
  if [ -n "$promote" ] && [ "$promote" != "none" ]; then
    log "promoting: $promote"
    # shellcheck disable=SC2086
    ./promote.sh "$round" $promote >> "$LOGDIR/rounds.log" 2>&1 || log "promote.sh reported a problem"
  else
    log "monitor promoted nothing this round"
  fi
}

# ---------------------------------------------------------------- phase 4: commit

commit_round() {
  local round=$1
  if [ "$DRYRUN" = 1 ]; then log "  [dry-run] would commit $round"; return 0; fi
  cd "$ROOT" || return 1
  git add -A strategies bench/geom/strategies bench/geom/exemplars 2>/dev/null
  git add -A bench/geom/results/"$round"/leaderboard.txt \
             bench/geom/results/"$round"/leaderboard.md \
             bench/geom/results/"$round"/environment.txt \
             bench/geom/results/"$round"/failures.txt \
             bench/geom/results/"$round"/build_errors.txt \
             bench/geom/results/"$round"/VERDICT.md 2>/dev/null
  git add -A 2>/dev/null
  if git diff --cached --quiet; then
    log "nothing to commit for $round"
  else
    git -c user.name="William Detmold" -c user.email="wdetmold@mit.edu" \
        commit -q -m "Panel round $round: implementations revised, measured, curated

Automated round from run_rounds.sh: implementers revised their entries with access to
every previous generation's strategy record and promoted exemplars, the benchmark node
measured them all on fresh random data, and the monitor's verdict is in
bench/geom/results/$round/VERDICT.md.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" && log "committed round $round"
  fi
  cd "$GEOM" || return 1
}

# ---------------------------------------------------------------- the cycle

while [ "$NEXT" -le "$LAST" ]; do
  stop_requested && break
  ROUND="panel_r$NEXT"
  SEED=$((20260821 + NEXT * 1013))
  log "---------- $ROUND begins ----------"

  # Serialize on the previous round's actual artifact. This also keeps us clear of the
  # first round, which was launched from an interactive workflow rather than from here:
  # its implementers may still be writing into impl/, and starting ours now would mean two
  # agents editing the same file.
  PREV="panel_r$((NEXT - 1))"
  if [ "$DRYRUN" = 1 ] && [ ! -f "$GEOM/results/$PREV/leaderboard.txt" ]; then
    log "  [dry-run] would wait for $PREV to finish"
  elif [ -d "$GEOM/results/$PREV" ] && [ ! -f "$GEOM/results/$PREV/leaderboard.txt" ]; then
    log "$PREV has no leaderboard yet -- waiting (will proceed if the panel goes quiet)"
    waited=0
    while [ ! -f "$GEOM/results/$PREV/leaderboard.txt" ]; do
      stop_requested && break 2

      # Liveness rather than a bare timeout: if nothing has written to impl/ for 45 minutes
      # and no benchmark job is in flight, the previous round is stuck, not working -- one
      # hung agent must not stall the whole series.
      newest=$(stat -c %Y "$GEOM"/impl/*.c 2>/dev/null | sort -n | tail -1)
      if [ -n "$newest" ]; then
        quiet_for=$(( $(date +%s) - newest ))
        if [ "$quiet_for" -gt 2700 ] && ! round_in_flight; then
          log "$PREV looks stalled (impl/ untouched for ${quiet_for}s, no job in flight) -- starting $ROUND anyway"
          break
        fi
      fi

      sleep 60
      waited=$((waited + 60))
      if [ "$waited" -ge 7200 ]; then
        log "$PREV still incomplete after ${waited}s -- starting $ROUND anyway"
        break
      fi
    done
    [ -f "$GEOM/results/$PREV/leaderboard.txt" ] && log "$PREV completed after ${waited}s"
  fi

  if [ "$DRYRUN" = 1 ]; then
    round_in_flight && log "  [dry-run] would wait for the in-flight benchmark job"
  else
    wait_for_quiet || break
  fi
  mkdir -p "$GEOM/results/$ROUND"

  [ "$DRYRUN" != 1 ] && setup_impl_dir "$ROUND"

  build_context "$ROUND" "$GEOM/results/$ROUND/context.md"
  log "context pack: $(wc -l < "$GEOM/results/$ROUND/context.md") lines"

  run_implementers "$ROUND" "$GEOM/results/$ROUND/context.md"
  stop_requested && break

  if run_timing "$ROUND" "$SEED"; then
    run_monitor "$ROUND"
  else
    log "$ROUND: no leaderboard, skipping the monitor verdict"
  fi

  commit_round "$ROUND"

  NEXT=$((NEXT + 1))
  if [ "$DRYRUN" = 1 ]; then log "[dry-run] stopping after one simulated round"; break; fi
  printf '%s %s\n' "$NEXT" "$LAST" > "$STATE"
  log "---------- $ROUND complete; next is panel_r$NEXT of panel_r$LAST ----------"
done

if [ "$NEXT" -gt "$LAST" ]; then
  log "=== series complete: rounds through panel_r$LAST are done ==="
else
  log "=== halted before panel_r$NEXT (STOP file); 'rm $STOPFILE' then resume ==="
fi
