#!/bin/bash
# Fast local development loop for ONE implementation: build, run, verify, report.
# Safe to run repeatedly and concurrently with other implementers -- everything lands in
# a per-implementation scratch directory and nothing touches the shared bin/ or driver.o.
#
# usage: ./tryout.sh [--on HOST] <impl-name> [L] [batch] [extra gcc flags...]
#   e.g. ./tryout.sh --on wallaby L17_rader          # RECOMMENDED: wallaby is idle and
#                                                    # has full AVX-512
#        ./tryout.sh L17_rader 17 64
#        ./tryout.sh L8_batchsimd 8 512 -funroll-all-loops
#
# Prefer --on wallaby.  It is a newer, near-idle login node (Xeon Gold 6448Y, Sapphire
# Rapids, 64 cores, full AVX-512 including fp16/bf16), it shares this filesystem, and its
# run-to-run spread is ~0.04% against ~0.4% on wombat -- so you can actually see small
# changes, and you can RUN your AVX-512 path rather than only compile it.
#
# Either way, these are DEVELOPMENT timings, relative only.  The scored numbers come from
# the monitor on the exclusive benchmark node (Cascade Lake), whose cache and AVX-512
# frequency behaviour differ from both login nodes -- see PANEL_BRIEF.md.
set -u
cd "$(dirname "$0")"

# --on HOST: rerun this same script over there (shared filesystem, per-host build dirs).
if [ "${1:-}" = "--on" ]; then
  TARGET=${2:-}
  [ -n "$TARGET" ] || { echo "usage: $0 --on HOST <impl-name> [...]" >&2; exit 2; }
  shift 2
  if [ "$TARGET" != "$(hostname -s)" ]; then
    exec ssh -o BatchMode=yes "$TARGET" "cd $(pwd) && ./tryout.sh $*"
  fi
fi

source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: $0 <impl-name> [L] [batch] [gcc flags...]" >&2; exit 2; }
SRC=impl/$NAME.c
[ -f "$SRC" ] || { echo "$SRC does not exist" >&2; exit 2; }

# Default L from the filename (L17_rader -> 17), default batch 8.
L=${2:-$(echo "$NAME" | sed -n 's/^L\([0-9]\+\)_.*/\1/p')}
[ -n "$L" ] || { echo "could not infer L from '$NAME'; pass it explicitly" >&2; exit 2; }
B=${3:-8}
[ $# -ge 3 ] && shift 3 || shift $#

WORK=${TMPDIR:-/tmp}/fft_tryout_$NAME
mkdir -p "$WORK"

ISA=$(lscpu | grep -oE "avx512f" | head -1)
echo "== building $SRC on $(hostname -s) (${ISA:-no avx512}) =="
if ! gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops \
        -I. -o "$WORK/bin" "$SRC" driver.c -lm "$@" 2>"$WORK/build.err"; then
  echo "BUILD FAILED:"; sed 's/^/  /' "$WORK/build.err" | head -30; exit 1
fi
[ -s "$WORK/build.err" ] && { echo "warnings:"; sed 's/^/  /' "$WORK/build.err" | head -10; }

echo "== L=$L B=$B =="
python3 gen_input.py --L "$L" --batch "$B" --seed 42 --out "$WORK/in.bin" >/dev/null || exit 1
"$WORK/bin" --L "$L" --batch "$B" --in "$WORK/in.bin" --out "$WORK/out.bin" \
            --samples 10 --warmup 5 || exit 1
python3 check.py --input "$WORK/in.bin" --output "$WORK/out.bin" --L "$L" --batch "$B"
RC=$?

# Repeatability: the contract says one plan must give the same answer every call, and the
# driver's timing loop calls execute thousands of times.  A plan that mutates its own
# scratch in a way that breaks the second call shows up here and nowhere else.
"$WORK/bin" --L "$L" --batch "$B" --in "$WORK/in.bin" --out "$WORK/out2.bin" \
            --samples 2 --warmup 3 >/dev/null 2>&1
if cmp -s "$WORK/out.bin" "$WORK/out2.bin"; then
  echo "repeatable: identical output across runs"
else
  echo "!! NOT REPEATABLE: a second run produced different output"
  RC=1
fi

echo "== reference: best library on the same case =="
BINDIR=build/$(hostname -s)/bin
if [ -x "$BINDIR/mkl_dfti" ]; then
  "$BINDIR/mkl_dfti" --L "$L" --batch "$B" --in "$WORK/in.bin" --samples 10 2>/dev/null
else
  echo "  (run 'make sota' on THIS host once to get a local comparison line)"
fi
exit $RC
