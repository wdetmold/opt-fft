#!/bin/bash
set -u
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
cd /home/lqcd/wdetmold/fft/bench/mt || exit 1
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
W=${TMPDIR:-/tmp}/fft_tryout_L23_matrixsimd
BIN=$W/bin

echo "=== B=128 verbose walk ==="
L23_VERBOSE=1 "$BIN" --L 23 --batch 128 --in $W/in128.bin --out $W/o_v128.bin \
    --samples 2 --warmup 1 2>&1 >/dev/null | grep "mt tune"

echo "=== forced si=2 cmp (degenerates to si=0 on one socket, must be bit-identical) ==="
L23_MODE=2 L23_NT=1 L23_SI=2 "$BIN" --L 23 --batch 128 --in $W/in128.bin \
    --out $W/o_si2.bin --samples 2 --warmup 1 2>/dev/null | head -1
if cmp -s $W/o_ref.bin $W/o_si2.bin; then echo "cmp si2: IDENTICAL"; else echo "cmp si2: !! DIFFERS"; fi

echo "=== B=1 description string (park telemetry) ==="
"$BIN" --L 23 --batch 1 --in $W/in1.bin --out $W/o_d1.bin --samples 2 --warmup 1 2>/dev/null | head -1
grep -o '"description": *"[^"]*"' $W/o_d1.bin 2>/dev/null || true
