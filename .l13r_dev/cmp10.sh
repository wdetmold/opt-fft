#!/bin/bash
# Bit-identity of this round's outputs against the r10 exemplar binary.
set -u
cd /home/lqcd/wdetmold/fft/bench/geom
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
W=${TMPDIR:-/tmp}/l13r_cmp10; mkdir -p "$W"
gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops \
    -I. -o "$W/new" impl/L13_rader.c driver.c -lm || exit 1
mkdir -p "$W/x/impl"
cp exemplars/panel_r10/L13_rader.c "$W/x/impl/"
cp fft3d_api.h "$W/x/"
gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops \
    -I. -o "$W/old" "$W/x/impl/L13_rader.c" driver.c -lm || exit 1
for B in 1 3 16 512; do
  python3 gen_input.py --L 13 --batch "$B" --seed 42 --out "$W/in.bin" >/dev/null
  "$W/new" --L 13 --batch "$B" --in "$W/in.bin" --out "$W/a.bin" --samples 2 --warmup 1 >/dev/null 2>&1
  "$W/old" --L 13 --batch "$B" --in "$W/in.bin" --out "$W/b.bin" --samples 2 --warmup 1 >/dev/null 2>&1
  if cmp -s "$W/a.bin" "$W/b.bin"; then echo "B=$B bit-identical to r10"; else echo "B=$B DIFFERS"; fi
done
