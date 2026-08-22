#!/bin/bash
# Build every knob combo with -Wall -Wextra and run a quick PASS check.
set -u
cd /home/lqcd/wdetmold/fft/bench/geom
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
W=${TMPDIR:-/tmp}/l13r_knobs; mkdir -p "$W"
python3 gen_input.py --L 13 --batch 3 --seed 7 --out "$W/in.bin" >/dev/null
for K in "" "-DL13R_AB=0" "-DL13R_FUSE=0" "-DL13R_ZG=0" "-DL13R_FORCE_YS=1" \
         "-DL13R_PACE=0" "-DL13R_X2" "-DL13R_PIN=0" "-DL13R_FORCE_PW=1" \
         "-DL13R_FORCE_PW=0" "-DL13R_FORCE_PF=1" "-DL13R_PS=184"; do
  if ! gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno \
       -funroll-loops -Wall -Wextra -I. -o "$W/bin" $K impl/L13_rader.c \
       driver.c -lm 2>"$W/err"; then
    echo "BUILD FAIL [$K]"; head -5 "$W/err"; continue
  fi
  [ -s "$W/err" ] && { echo "WARNINGS [$K]:"; head -5 "$W/err"; }
  "$W/bin" --L 13 --batch 3 --in "$W/in.bin" --out "$W/out.bin" \
      --samples 2 --warmup 1 >/dev/null 2>&1
  R=$(python3 check.py --input "$W/in.bin" --output "$W/out.bin" --L 13 --batch 3 2>&1 | head -1)
  echo "[$K] $R"
done
