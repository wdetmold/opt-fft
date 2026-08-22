#!/bin/bash
# Promote panel entries worth keeping into exemplars/<round>/, with their strategy
# records, so the next panel starts from real code.  See ../../docs/CURATION.md for the
# grounds on which to promote.
#
# usage: ./promote.sh <round> <impl-name> [<impl-name> ...]
#   e.g. ./promote.sh panel_r1 L17_rader L8_batchsimd
set -eu
cd "$(dirname "$0")"

[ $# -ge 2 ] || { echo "usage: $0 <round> <impl-name> [<impl-name> ...]" >&2; exit 2; }
ROUND=$1; shift
DEST=exemplars/$ROUND
mkdir -p "$DEST"

for name in "$@"; do
  # this harness's sources are .cu; the CPU harnesses' are .c
  src=impl/$name.cu
  [ -f "$src" ] || src=impl/$name.c
  if [ ! -f "$src" ]; then
    echo "!! no impl/$name.{cu,c} -- skipping" >&2
    continue
  fi
  cp "$src" "$DEST/$(basename "$src")"
  if [ -f "strategies/$name.md" ]; then
    cp "strategies/$name.md" "$DEST/$name.strategy.md"
  else
    echo "!! no strategy record for $name -- per docs/CURATION.md it should not be" \
         "promoted without one" >&2
  fi
  # Pull this entry's measured numbers out of the round's leaderboard, so the exemplar
  # carries its own evidence rather than relying on a file somewhere else.
  if [ -f "results/$ROUND/leaderboard.txt" ]; then
    grep -h "$name" "results/$ROUND/leaderboard.txt" > "$DEST/$name.results.txt" || true
  fi
  echo "promoted $name"
done

if [ ! -f "$DEST/NOTES.md" ]; then
  cat > "$DEST/NOTES.md" <<NOTE
# Round $ROUND — what it established

Promoted: $*

## Result

(Fill in: the leaderboard headline per geometry, panel best vs best library.)

## What this round settled

(Which open question from docs/LITERATURE.md section 4 now has an answer, and what it is.)

## What the next round should attack

(The specific thing, and why it is the highest-value next move.)

## Dead ends worth not repeating

(Approach, and the measured number that killed it.)
NOTE
  echo "wrote $DEST/NOTES.md skeleton -- fill it in before committing"
fi

echo
echo "next: git add $DEST strategies && git commit"
