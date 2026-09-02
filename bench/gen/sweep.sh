#!/bin/bash
# Runs on the benchmark node itself.  Rebuilds locally (so -march=native targets THIS
# cpu), generates fresh random data for the round, then measures every backend on every
# case in several independent processes.
#
# usage: sweep.sh --round TAG --seed N [--runs 3] [--samples 20] [--quick]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

ROUND=""; SEED=0; RUNS=3; SAMPLES=20; QUICK=0; CASEFILE=cases.txt
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --samples) SAMPLES=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --cases) CASEFILE=$2; shift 2 ;;
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
if [ -f "$CASEFILE" ]; then
  # A geometry wave can extend the sweep by dropping a cases.txt here, rather than by
  # editing this script (which a running job may be executing).
  CASES=$(grep -vE '^\s*(#|$)' "$CASEFILE" | tr '\n' ' ')
elif [ "$QUICK" = 1 ]; then
  CASES="6:1 6:512 8:1 8:512 17:1 17:64 36:1 36:8"
else
  CASES="6:1 6:64 6:4096 6:32768 8:1 8:64 8:2048 8:16384 \
         17:1 17:8 17:256 17:2048 36:1 36:4 36:32 36:256"
fi

BINDIR=build/$(hostname -s)/bin
BACKENDS=$(cd "$BINDIR" && ls)
# Refuse to produce a "leaderboard" that contains only libraries: if no panel binary was
# built, the round measured nothing of ours and a board would read as a legitimate loss.
npanel=$(cd "$BINDIR" && ls | grep -vcE "^(mkl|fftw|ducc|baseline)" || true)
if [ "${npanel:-0}" -eq 0 ]; then
  echo "ABORT: no PANEL BINARIES in $BINDIR -- impl/ is empty or the build failed" | tee -a "$OUT/build_errors.txt"
  exit 4
fi
echo "== backends: $BACKENDS =="

for case in $CASES; do
  L=${case%%:*}
  rest=${case#*:}
  B=${rest%%:*}
  # optional third field: chain length m (the graded call); absent means 1
  M=1; case "$rest" in *:*) M=${rest##*:} ;; esac
  IN=$OUT/in_L${L}_B${B}.bin
  CIN=$OUT/c_L${L}_B${B}.bin
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + L * 1000 + B)) --out "$IN" >/dev/null
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + 900000 + L * 1000 + B)) --scale 0.1 --out "$CIN" >/dev/null
  for backend in $BACKENDS; do
    # The dense-matrix floor is O(L^4) per volume per axis, so on a big case it costs more
    # wall clock than every real backend combined (2.8 s per call at 36^3 x 256). It is a
    # sanity floor, not a contender: skip it once the case is large.
    if [ "$backend" = "baseline_matrix" ] && [ $((L * L * L * B)) -gt 2000000 ]; then
      echo "   skipping baseline_matrix at L=$L B=$B (too expensive to be informative)" \
        >> "$OUT/timing.log"
      continue
    fi
    for run in $(seq 1 "$RUNS"); do
      # A panel entry that hangs or crashes must not take the round down with it.
      CHAINARGS=""
      [ "$M" -gt 1 ] && CHAINARGS="--chain $M --map --cin $CIN"
      timeout 600 "$BINDIR/$backend" --L "$L" --batch "$B" $CHAINARGS --in "$IN" \
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
      CHKARGS=""
      [ "$M" -gt 1 ] && CHKARGS="--map-check $M --cin $CIN"
      python3 check.py --input "$IN" --output "$OUT/out_${backend}_L${L}_B${B}.bin" \
        --L "$L" --batch "$B" $CHKARGS --json "$OUT/c_${backend}_L${L}_B${B}.json" \
        >>"$OUT/check.log" 2>&1
      rm -f "$OUT/out_${backend}_L${L}_B${B}.bin" "$OUT/out_${backend}_L${L}_B${B}.bin.chain"   # outputs are large; keep the verdicts
    fi
    # ONE-STEP map gate (chained cases only): two steps of the graded map must match
    # numpy to 3e-14 (chaos cannot amplify in 2 steps; fp32-seeded maps land ~5e-12). This carries the precision contract; the chain gate above only
    # catches gross cheats, because the chain is chaotic (docs/GRADER.md).
    if [ "$M" -gt 1 ]; then
      timeout 120 "$BINDIR/$backend" --L "$L" --batch "$B" --chain 2 --map --cin "$CIN" \
        --in "$IN" --out "$OUT/one_${backend}_L${L}_B${B}.bin" \
        --json "$OUT/t1_${backend}_L${L}_B${B}.json" \
        --samples 1 --warmup 1 --min-sample-ms 1 --run-index 1 \
        >>"$OUT/timing.log" 2>>"$OUT/timing.err"
      if [ -f "$OUT/one_${backend}_L${L}_B${B}.bin" ]; then
        python3 check.py --input "$IN" --output "$OUT/one_${backend}_L${L}_B${B}.bin" \
          --L "$L" --batch "$B" --map-check 2 --cin "$CIN" \
          --json "$OUT/o_${backend}_L${L}_B${B}.json" >>"$OUT/check.log" 2>&1
      fi
      rm -f "$OUT/one_${backend}_L${L}_B${B}.bin" "$OUT/one_${backend}_L${L}_B${B}.bin.chain" "$OUT/t1_${backend}_L${L}_B${B}.json"
    fi
  done
  rm -f "$IN" "$CIN"
  echo "   done L=$L B=$B"
done

python3 leaderboard.py --round "$ROUND" | tee "$OUT/leaderboard.txt"
echo "== round $ROUND complete: $OUT =="
