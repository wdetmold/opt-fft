#!/bin/bash
# Development loop for the ICE LAKE panel: build + run + verify + time, ON THE RESERVED
# ICE LAKE NODE, pinned to a leased core so 19 implementers do not trample each other.
#
#   ./tryout.sh <impl-name> [L] [batch] [extra gcc flags...]
#
# Everything happens on the reserved node over ssh (the filesystem is shared): the build
# (so -march=native means Ice Lake), the run, and the checks. Timings here are from a
# single pinned core on otherwise-quiet silicon -- the same conditions you are scored
# under, minus the scoring window's full quiet.
set -u
cd "$(dirname "$0")"
ICE=$(pwd)
NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: $0 <impl-name> [L] [batch] [gcc flags...]" >&2; exit 2; }
SRC=impl/$NAME.c
[ -f "$SRC" ] || { echo "$SRC does not exist" >&2; exit 2; }
L=${2:-$(echo "$NAME" | sed -n 's/^L\([0-9]\+\)_.*/\1/p')}
[ -n "$L" ] || { echo "cannot infer L from '$NAME'" >&2; exit 2; }
B=${3:-8}
[ $# -ge 3 ] && shift 3 || shift $#

if [ ! -f RESERVATION ] || ! ./reserve.sh --status >/dev/null 2>&1; then
  echo "no live Ice Lake reservation -- ./reserve.sh first (or ask the monitor)" >&2
  exit 2
fi
# shellcheck disable=SC1091
. ./RESERVATION

SLOT=$(./slot_lease.sh acquire --label "$NAME") || exit 1
trap './slot_lease.sh release "$SLOT" >/dev/null 2>&1' EXIT INT TERM
CORE=$((SLOT + 2))

# the graded chain length for this L, so dev timings measure what the score measures
M=$(awk -F: -v l="$L" '$1==l {print $3}' cases.txt 2>/dev/null); M=${M:-1}
CH=""; [ "${M:-1}" -gt 1 ] && CH="--chain $M --unitary"

W=$ICE/build/tryout/$NAME
mkdir -p "$W"
ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" "
  cd '$ICE' && source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1 &&
  echo '== building on '\$(hostname -s)' (Ice Lake, -march=native) ==' &&
  gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops \
      -I. -o '$W/bin' '$SRC' driver.c -lm $* 2>'$W/build.err' || { head -25 '$W/build.err'; exit 1; }
  python3 gen_input.py --L $L --batch $B --seed 42 --out '$W/in.bin' >/dev/null &&
  echo '== L=$L B=$B m=$M on core $CORE ==' &&
  taskset -c $CORE '$W/bin' --L $L --batch $B $CH --in '$W/in.bin' --out '$W/out.bin' --samples 8 &&
  python3 check.py --input '$W/in.bin' --output '$W/out.bin' --L $L --batch $B ${M:+$([ $M -gt 1 ] && echo --chain-check $M)} &&
  taskset -c $CORE '$W/bin' --L $L --batch $B $CH --in '$W/in.bin' --out '$W/out2.bin' --samples 2 >/dev/null 2>&1 &&
  { cmp -s '$W/out.bin' '$W/out2.bin' && echo 'repeatable: identical output across runs' || echo '!! NOT REPEATABLE'; }
  BINDIR='$ICE/build/'\$(hostname -s)'/bin'
  if [ -x \"\$BINDIR/mkl_dfti\" ]; then
    echo '== reference: MKL on the same case, same core =='
    taskset -c $CORE \"\$BINDIR/mkl_dfti\" --L $L --batch $B $CH --in '$W/in.bin' --samples 8
  fi"
