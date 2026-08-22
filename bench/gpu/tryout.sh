#!/bin/bash
# Fast development loop for ONE GPU implementation: build, run, verify, report.
#
# Runs where you are. The login node HAS an A100, so unlike the CPU phases you develop on
# the same architecture you are scored on -- but it is shared, so treat its timings as
# relative only (cuFFT's spread here is ~30% against ~0.1% on a job-allocated GPU).
#
# usage: ./tryout.sh <impl-name> [L] [batch] [extra nvcc flags...]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: $0 <impl-name> [L] [batch] [nvcc flags...]" >&2; exit 2; }
SRC=impl/$NAME.cu
[ -f "$SRC" ] || { echo "$SRC does not exist" >&2; exit 2; }
L=${2:-$(echo "$NAME" | sed -n 's/^L\([0-9]\+\)_.*/\1/p')}
[ -n "$L" ] || { echo "could not infer L from '$NAME'; pass it explicitly" >&2; exit 2; }
B=${3:-64}
[ $# -ge 3 ] && shift 3 || shift $#

WORK=${TMPDIR:-/tmp}/fft_gpu_tryout_$NAME
mkdir -p "$WORK"
NVCC=${CUDA_HOME:-/opt/software/cuda-12.2.1}/bin/nvcc

echo "== building $SRC for sm_80 =="
if ! $NVCC -O3 -std=c++14 -arch=sm_80 -lineinfo -I. -o "$WORK/bin" "$SRC" driver.cu -lm "$@" \
        2>"$WORK/build.err"; then
  echo "BUILD FAILED:"; sed 's/^/  /' "$WORK/build.err" | head -30; exit 1
fi
grep -i 'warning' "$WORK/build.err" | head -5

echo "== L=$L B=$B on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) =="
python3 gen_input.py --L "$L" --batch "$B" --seed 42 --out "$WORK/in.bin" >/dev/null || exit 1
"$WORK/bin" --L "$L" --batch "$B" --in "$WORK/in.bin" --out "$WORK/out.bin" --samples 10 || exit 1
python3 check.py --input "$WORK/in.bin" --output "$WORK/out.bin" --L "$L" --batch "$B"
RC=$?

"$WORK/bin" --L "$L" --batch "$B" --in "$WORK/in.bin" --out "$WORK/out2.bin" --samples 2 >/dev/null 2>&1
if cmp -s "$WORK/out.bin" "$WORK/out2.bin"; then
  echo "repeatable: identical output across runs"
else
  echo "!! NOT REPEATABLE: a second run produced different output"; RC=1
fi

echo "== reference: cuFFT on the same case =="
BINDIR=build/$(hostname -s)/bin
if [ -x "$BINDIR/cufft" ]; then
  "$BINDIR/cufft" --L "$L" --batch "$B" --in "$WORK/in.bin" --samples 10
else
  echo "  (run 'make sota' once to get a cuFFT comparison line)"
fi
echo
echo "worth knowing: compute-sanitizer --tool memcheck \"$WORK/bin\" --L $L --batch $B --in $WORK/in.bin"
echo "               ncu --set full \"$WORK/bin\" ...   for occupancy and memory analysis"
exit $RC
