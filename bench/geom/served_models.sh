#!/bin/bash
# Report which models ACTUALLY served a round's worker requests, from the workers' own
# transcripts -- not what the runner requested. Exists because Claude Code can re-run a
# flagged request on a fallback model (documented: fable-5 + cyber flag -> opus-4-8), and
# in headless -p mode that substitution is not surfaced anywhere the runner can see. A
# round whose composition differs from its configuration is a confound, so record it.
# usage: served_models.sh --harness <dir> --round <tag>
set -u
HARNESS=""; ROUND=""
while [ $# -gt 0 ]; do
  case "$1" in
    --harness) HARNESS=$2; shift 2 ;;
    --round) ROUND=$2; shift 2 ;;
    *) shift ;;
  esac
done
[ -n "$HARNESS" ] && [ -n "$ROUND" ] || { echo "usage: $0 --harness <dir> --round <tag>" >&2; exit 2; }
RD="$HARNESS/results/$ROUND"
[ -d "$RD" ] || { echo "no such round dir: $RD" >&2; exit 2; }
SLUG=$(echo "$HARNESS" | sed 's|^/||; s|/|-|g')
PROJ="$HOME/.claude/projects/-$SLUG"
OUT="$RD/served_models.txt"
{
  echo "# models that actually served $ROUND's workers (from worker transcripts)"
  echo "# transcript dir: $PROJ"
  if [ ! -d "$PROJ" ]; then
    echo "NO TRANSCRIPT DIR -- cannot verify served models"
  else
    # anchor on the round's START time (from the runner log), not on a dir whose mtime
    # keeps moving while the round writes into it
    START=$(grep -h -- "---------- $ROUND begins" "$HARNESS/logs/rounds.log" 2>/dev/null \
            | tail -1 | sed -n 's/^\[\([0-9-]* [0-9:]*\)\].*/\1/p')
    if [ -n "$START" ]; then
      echo "# round start (log): $START"
      find "$PROJ" -name '*.jsonl' -newermt "$START" 2>/dev/null > /tmp/sm_files_$$ || true
    else
      echo "# no round-start line in log; falling back to context.md mtime"
      A="$RD/context.md"; [ -f "$A" ] || A="$RD"
      find "$PROJ" -name '*.jsonl' -newer "$A" 2>/dev/null > /tmp/sm_files_$$ || true
    fi
    N=$(wc -l < /tmp/sm_files_$$)
    echo "# transcripts newer than the round anchor: $N"
    if [ "$N" -gt 0 ]; then
      while read -r f; do grep -ho '"model":"[^"]*"' "$f" 2>/dev/null; done < /tmp/sm_files_$$ \
        | sed 's/"model":"//; s/"$//' | sort | uniq -c | sort -rn
      echo "# --- fallback/flag notices (real events, not doc text) ---"
      while read -r f; do
        grep -hoE "Model '[a-z0-9.\-]+' is available, but a request was flagged" "$f" 2>/dev/null
      done < /tmp/sm_files_$$ | sort | uniq -c
    fi
    rm -f /tmp/sm_files_$$
  fi
} > "$OUT" 2>&1
cat "$OUT"
