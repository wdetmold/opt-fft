#!/bin/bash
# Runs on the benchmark node itself.  Rebuilds locally (so -march=native targets THIS
# cpu), generates fresh random data for the round, then measures every backend on every
# case in several independent processes.
#
# usage: sweep.sh --round TAG --seed N [--runs 3] [--samples 20] [--quick]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

ROUND=""; SEED=0; RUNS=3; SAMPLES=20; QUICK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --samples) SAMPLES=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROUND" ] || { echo "sweep.sh: --round is required" >&2; exit 2; }

OUT=results/$ROUND
mkdir -p "$OUT"

{
  echo "# round $ROUND"
  echo "host: $(hostname)   date: $(date -Is)   slurm_job: ${SLURM_JOB_ID:-none}"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "isa: $(lscpu | grep -o -E 'avx512[a-z_]*|avx2|fma' | sort -u | tr '\n' ' ')"
  echo "cores: $(nproc)   governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "gcc: $(gcc --version | head -1)"
} | tee "$OUT/environment.txt"

# Compilation happens here, outside every timed region.
echo "== building on $(hostname) =="
make clean >/dev/null 2>&1
if ! make -k 2>"$OUT/build_errors.txt"; then
  echo "!! some backends failed to build; see $OUT/build_errors.txt"
  grep -E 'error' "$OUT/build_errors.txt" | head -20
fi
make list

# Cases: batch=1 is the NON-BATCHED problem, reported separately from the batched ones.
# The batched points are chosen per L to sit (a) inside L2 and (b) well past L3, since
# small-transform performance is decided by where the working set lives.
#   bytes per volume = 16 * L^3
if [ "$QUICK" = 1 ]; then
  CASES="6:1 6:512 8:1 8:512 17:1 17:64 36:1 36:8"
else
  CASES="6:1 6:64 6:4096 6:32768 8:1 8:64 8:2048 8:16384 \
         17:1 17:8 17:256 17:2048 36:1 36:4 36:32 36:256"
fi

BINDIR=build/$(hostname -s)/bin
BACKENDS=$(cd "$BINDIR" && ls)
echo "== backends: $BACKENDS =="

for case in $CASES; do
  L=${case%%:*}; B=${case##*:}
  IN=$OUT/in_L${L}_B${B}.bin
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + L * 1000 + B)) --out "$IN" >/dev/null
  for backend in $BACKENDS; do
    for run in $(seq 1 "$RUNS"); do
      # A panel entry that hangs or crashes must not take the round down with it.
      timeout 600 "$BINDIR/$backend" --L "$L" --batch "$B" --in "$IN" \
        --out "$OUT/out_${backend}_L${L}_B${B}.bin" \
        --json "$OUT/t_${backend}_L${L}_B${B}_r${run}.json" \
        --samples "$SAMPLES" --warmup 5 --min-sample-ms 20 --run-index "$run" \
        >>"$OUT/timing.log" 2>>"$OUT/timing.err"
      rc=$?
      if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
        echo "$backend L=$L B=$B run=$run exited $rc" >>"$OUT/failures.txt"
      fi
    done
    # correctness on the output of the last run, against numpy
    if [ -f "$OUT/out_${backend}_L${L}_B${B}.bin" ]; then
      python3 check.py --input "$IN" --output "$OUT/out_${backend}_L${L}_B${B}.bin" \
        --L "$L" --batch "$B" --json "$OUT/c_${backend}_L${L}_B${B}.json" \
        >>"$OUT/check.log" 2>&1
      rm -f "$OUT/out_${backend}_L${L}_B${B}.bin"   # outputs are large; keep the verdicts
    fi
  done
  rm -f "$IN"
  echo "   done L=$L B=$B"
done

python3 leaderboard.py --round "$ROUND" | tee "$OUT/leaderboard.txt"
echo "== round $ROUND complete: $OUT =="
