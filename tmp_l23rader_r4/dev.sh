#!/bin/bash
# mt_r4 dev loop for L23_rader: bit-identity cmp across forced cells + si/pv A/Bs.
# Run as: ssh wallaby /home/lqcd/wdetmold/fft/tmp_l23rader_r4/dev.sh
set -u
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
cd /home/lqcd/wdetmold/fft/bench/mt || exit 1
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
W=${TMPDIR:-/tmp}/fft_tryout_L23_rader
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

echo "=== B=128 bit-class cmp across forced knobs (cell 0 = plain static ref) ==="
run ref    23 128 $W/in128.bin $W/o_ref.bin   L23R_FORCE=0
run si1    23 128 $W/in128.bin $W/o_si1.bin   L23R_FORCE=0 L23R_SI=1
run pv1    23 128 $W/in128.bin $W/o_pv1.bin   L23R_FORCE=0 L23R_PV=1
run ntsi   23 128 $W/in128.bin $W/o_ntsi.bin  L23R_FORCE=2 L23R_SI=1
run ntsipv 23 128 $W/in128.bin $W/o_nsp.bin   L23R_FORCE=2 L23R_SI=1 L23R_PV=1
run w2     23 128 $W/in128.bin $W/o_w2.bin    L23R_FORCE=10
run si2fk  23 128 $W/in128.bin $W/o_si2.bin   L23R_FORCE=2 L23R_SI=2 L23R_FAKESOCK=1
run si1fk  23 128 $W/in128.bin $W/o_si1f.bin  L23R_FORCE=2 L23R_SI=1 L23R_FAKESOCK=1
for f in si1 pv1 ntsi nsp w2 si2 si1f; do
  if cmp -s $W/o_ref.bin $W/o_$f.bin; then echo "cmp $f: IDENTICAL"; else echo "cmp $f: !! DIFFERS"; fi
done

echo "=== B=1 park A/B at forced fused T=16 (bit-cmp'd vs T=32) ==="
run f32    23 1 $W/in1.bin $W/o_f32.bin  L23R_FORCE=1
run f16pk1 23 1 $W/in1.bin $W/o_f16a.bin L23R_FORCE=0 L23R_PARK=1
run f16pk0 23 1 $W/in1.bin $W/o_f16b.bin L23R_FORCE=0 L23R_PARK=0
for f in f16a f16b; do
  if cmp -s $W/o_f32.bin $W/o_$f.bin; then echo "cmp $f: IDENTICAL"; else echo "cmp $f: !! DIFFERS"; fi
done

echo "=== B=2048 driver-level si/pv A/B (batchNT static) ==="
run nt_si0 23 2048 $W/in2048.bin $W/o_a.bin L23R_FORCE=2
run nt_si1 23 2048 $W/in2048.bin $W/o_b.bin L23R_FORCE=2 L23R_SI=1
run nt_pv1 23 2048 $W/in2048.bin $W/o_c.bin L23R_FORCE=2 L23R_PV=1
for f in b c; do
  if cmp -s $W/o_a.bin $W/o_$f.bin; then echo "cmp 2048/$f: IDENTICAL"; else echo "cmp 2048/$f: !! DIFFERS"; fi
done

echo "=== verbose tuner tables ==="
L23R_VERBOSE=1 "$BIN" --L 23 --batch 2048 --in $W/in2048.bin --out $W/o_v.bin \
    --samples 2 --warmup 1 2>&1 >/dev/null | grep "tune"
L23R_VERBOSE=1 "$BIN" --L 23 --batch 1 --in $W/in1.bin --out $W/o_v1.bin \
    --samples 2 --warmup 1 2>&1 >/dev/null | grep "tune"
