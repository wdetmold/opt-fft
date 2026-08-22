#!/bin/bash
# mt_r4 dev loop for L23_matrixsimd: forced-mode cmp bit-identity + A/Bs.
# Run as: ssh wallaby /home/lqcd/wdetmold/fft/tmp_l23r4/dev.sh
set -u
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
cd /home/lqcd/wdetmold/fft/bench/mt || exit 1
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
W=${TMPDIR:-/tmp}/fft_tryout_L23_matrixsimd
BIN=$W/bin
[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }

run() { # run <tag> <L> <B> <infile> <outfile> [env...]
  local tag=$1 L=$2 B=$3 inf=$4 outf=$5; shift 5
  env "$@" "$BIN" --L "$L" --batch "$B" --in "$inf" --out "$outf" \
      --samples 6 --warmup 3 2>/dev/null | sed "s/^/[$tag] /"
  test -s "$outf" || echo "[$tag] !! EMPTY OUTPUT"
}

echo "=== gen inputs ==="
python3 gen_input.py --L 23 --batch 128 --seed 7 --out $W/in128.bin >/dev/null
python3 gen_input.py --L 23 --batch 2048 --seed 7 --out $W/in2048.bin >/dev/null
python3 gen_input.py --L 23 --batch 1 --seed 7 --out $W/in1.bin >/dev/null

echo "=== B=128 bit-class cmp across forced modes ==="
run ref    23 128 $W/in128.bin $W/o_ref.bin    L23_MODE=2 L23_NT=0 L23_SI=0 L23_DYN=0
run si1    23 128 $W/in128.bin $W/o_si1.bin    L23_MODE=2 L23_NT=0 L23_SI=1 L23_DYN=0
run ntsi   23 128 $W/in128.bin $W/o_ntsi.bin   L23_MODE=2 L23_NT=1 L23_SI=1 L23_DYN=0
run ntsidy 23 128 $W/in128.bin $W/o_ntsidy.bin L23_MODE=2 L23_NT=1 L23_SI=1 L23_DYN=1
run w2si   23 128 $W/in128.bin $W/o_w2si.bin   L23_MODE=5 L23_NT=1 L23_SI=1
run t7si   23 128 $W/in128.bin $W/o_t7si.bin   L23_MODE=2 L23_NT=1 L23_SI=1 L23_T=7
run pk0    23 128 $W/in128.bin $W/o_pk0.bin    L23_MODE=2 L23_NT=1 L23_SI=1 L23_T=7 L23_PARK=0
for f in si1 ntsi ntsidy w2si t7si pk0; do
  if cmp -s $W/o_ref.bin $W/o_$f.bin; then echo "cmp $f: IDENTICAL"; else echo "cmp $f: !! DIFFERS"; fi
done

echo "=== B=1 park A/B at forced fused T=16 (both bit-cmp'd vs T=32) ==="
run f32    23 1 $W/in1.bin $W/o_f32.bin  L23_MODE=1 L23_T=32
run f16pk1 23 1 $W/in1.bin $W/o_f16a.bin L23_MODE=1 L23_T=16 L23_PARK=1
run f16pk0 23 1 $W/in1.bin $W/o_f16b.bin L23_MODE=1 L23_T=16 L23_PARK=0
for f in f16a f16b; do
  if cmp -s $W/o_f32.bin $W/o_$f.bin; then echo "cmp $f: IDENTICAL"; else echo "cmp $f: !! DIFFERS"; fi
done

echo "=== B=2048 driver-level si A/B (nt1 static, one socket) ==="
run nt1si0 23 2048 $W/in2048.bin $W/o_a.bin L23_MODE=2 L23_NT=1 L23_SI=0 L23_DYN=0
run nt1si1 23 2048 $W/in2048.bin $W/o_b.bin L23_MODE=2 L23_NT=1 L23_SI=1 L23_DYN=0
if cmp -s $W/o_a.bin $W/o_b.bin; then echo "cmp b2048 si: IDENTICAL"; else echo "cmp b2048 si: !! DIFFERS"; fi

echo "=== verbose tuner tables, B=2048 ==="
L23_VERBOSE=1 "$BIN" --L 23 --batch 2048 --in $W/in2048.bin --out $W/o_v.bin \
    --samples 2 --warmup 1 2>&1 >/dev/null | grep "mt tune"
