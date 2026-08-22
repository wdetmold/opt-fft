#!/bin/bash
# Pinned min-over-N-instances measurement, alternating with MKL in the same
# windows (the record's methodology). usage: meas.sh B [N]
set -u
cd /home/lqcd/wdetmold/fft/bench/geom
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
B=$1; N=${2:-4}
W=${TMPDIR:-/tmp}/fft_tryout_L13_rader
MKL=build/$(hostname -s)/bin/mkl_dfti
python3 gen_input.py --L 13 --batch "$B" --seed 42 --out "$W/in$B.bin" >/dev/null
for i in $(seq 1 "$N"); do
  taskset -c 49 "$W/bin" --L 13 --batch "$B" --in "$W/in$B.bin" \
    --out /dev/null --samples 8 --warmup 3 2>/dev/null | grep -o "min=[^u]*us" | head -1 | sed "s/^/  mine  /"
  taskset -c 49 "$MKL" --L 13 --batch "$B" --in "$W/in$B.bin" \
    --samples 8 --warmup 3 2>/dev/null | grep -o "min=[^u]*us" | head -1 | sed "s/^/  mkl   /"
done
