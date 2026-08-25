#!/bin/bash
# Static port/throughput analysis of ONE function in a built binary, via uiCA's Ice Lake model.
# usage: tools/uica_fn.sh <binary> <symbol> [arch=ICL]
set -e
BIN=$1; SYM=$2; ARCH=${3:-ICL}
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
UICA=/home/lqcd/wdetmold/fft/ext/tools/uiCA
read START SIZE <<< $(nm -S "$BIN" 2>/dev/null | awk -v s="$SYM" '$4==s || $3==s {print strtonum("0x"$1), strtonum("0x"$2)}')
[ -n "$START" ] || { echo "symbol $SYM not found (try nm $BIN | grep ...)"; exit 1; }
OFF=$(objdump -h "$BIN" | awk '$2==".text" {print strtonum("0x"$6) - strtonum("0x"$4)}')
dd if="$BIN" of=/tmp/uica_$$.bin bs=1 skip=$((START+OFF)) count=$SIZE 2>/dev/null
python3 "$UICA/uiCA.py" /tmp/uica_$$.bin -arch "$ARCH"
rm -f /tmp/uica_$$.bin
