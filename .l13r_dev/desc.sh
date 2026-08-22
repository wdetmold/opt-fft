#!/bin/bash
# Dev helper for L13_rader panel_r11: run the tryout-built binary pinned,
# with --json, and print the description (race readings) per batch.
set -u
cd /home/lqcd/wdetmold/fft/bench/geom
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
W=${TMPDIR:-/tmp}/fft_tryout_L13_rader
CPU=${CPU:-49}
for B in "$@"; do
  python3 gen_input.py --L 13 --batch "$B" --seed 42 --out "$W/in$B.bin" >/dev/null
  taskset -c "$CPU" "$W/bin" --L 13 --batch "$B" --in "$W/in$B.bin" \
      --out "$W/o$B.bin" --samples 6 --warmup 3 --json "$W/j$B.json" >/dev/null 2>&1
  python3 - "$W/j$B.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print(d["batch"], "|", d["description"], "|",
      round(d["per_transform_seconds_min"]*1e9), "ns/t setup",
      round(d["setup_seconds"], 3))
EOF
done
