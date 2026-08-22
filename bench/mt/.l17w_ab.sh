#!/bin/bash
# L17_winograd mt_r4 dev A/B helper (temporary; removed before round end).
set -u
cd /home/lqcd/wdetmold/fft/bench/mt
W=${TMPDIR:-/tmp}/fft_tryout_L17_winograd
B=${1:-4096}
shift || true
CFG=${@:-"0 8 16"}
for pf in $CFG; do
  gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops -fopenmp -I. \
      -DL17_FORCE_VARIANT=${FV:-15} -DL17_FORCE_PF=$pf -DL17_FORCE_BTHR=32 \
      -o "$W/bin_pf$pf" impl/L17_winograd.c driver.c -lm 2>"$W/be_$pf" \
      || { echo "BUILD FAIL pf=$pf"; head -5 "$W/be_$pf"; exit 1; }
done
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
for r in 1 2 3; do
  for pf in $CFG; do
    "$W/bin_pf$pf" --L 17 --batch "$B" --in "$W/in.bin" --out "$W/o_$pf.bin" \
                   --samples 5 --warmup 2 | sed "s/^/pf=$pf /"
  done
done
first=""
for pf in $CFG; do
  if [ -z "$first" ]; then first=$pf; continue; fi
  cmp "$W/o_$first.bin" "$W/o_$pf.bin" || { echo "MISMATCH pf=$pf"; exit 1; }
done
echo "outputs bit-identical across configs"
