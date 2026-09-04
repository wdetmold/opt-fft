#!/bin/bash
# Runs on the GPU benchmark node. Rebuilds locally, generates fresh random data for the
# round, then measures every backend on every case in several independent processes.
#
# usage: sweep.sh --round TAG --seed N [--runs 3] [--samples 20]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

# This is a single-GPU competition, and the allocation is a whole 8-GPU node (the cluster's
# select/linear plugin gives no finer granularity), so pin to one device explicitly. Without
# this a backend could quietly use a second GPU and its number would mean nothing.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ROUND=""; SEED=0; RUNS=3; SAMPLES=20
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --samples) SAMPLES=$2; shift 2 ;;
    --quick) shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROUND" ] || { echo "sweep.sh: --round is required" >&2; exit 2; }

OUT=results/$ROUND
mkdir -p "$OUT"
# ---- guard: this harness cannot distinguish two cases that share (L,B) --------------------
# Result files here are named t_<backend>_L<L>_B<B>_r<run>.json with NO chain tag, so two
# cases differing only in chain length write the SAME filenames and the later one silently
# overwrites the earlier.  That is not hypothetical: a comparison run with both "6:64:4856"
# and "6:64" in one case list lost the entire chained half -- 624 surviving files all read
# chain=1 and not one of 134 correctness verdicts carried a chain result.  d1/sweep.sh carries
# the _m fix; these harnesses do not, so refuse the input rather than produce a mixed round.
_dups=$(printf '%s\n' $CASES | awk -F: '{print $1":"$2}' | sort | uniq -d)
if [ -n "$_dups" ]; then
  echo "ABORT: case list has (L,B) pairs appearing more than once:" >&2
  printf '   %s\n' $_dups >&2
  echo "   This harness's filenames carry no chain tag, so these cases would overwrite each" >&2
  echo "   other.  Split them into separate rounds, or port the _m naming from d1/sweep.sh." >&2
  exit 5
fi

BINDIR=build/$(hostname -s)/bin

{
  echo "# round $ROUND"
  echo "host: $(hostname)   date: $(date -Is)   slurm_job: ${SLURM_JOB_ID:-none}"
  nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem,persistence_mode \
             --format=csv,noheader 2>/dev/null | sed 's/^/gpu: /'
  echo "visible devices: ${CUDA_VISIBLE_DEVICES:-all}  (pinned to one A100 on purpose)"
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | sed 's/^/all gpus on node: /'
  echo "nvcc: $(nvcc --version | tail -1)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
} | tee "$OUT/environment.txt"

echo "== building on $(hostname) =="
make clean >/dev/null 2>&1
if ! make -k all 2>"$OUT/build_errors.txt"; then
  echo "!! some backends failed to build; see $OUT/build_errors.txt"
  grep -E 'error' "$OUT/build_errors.txt" | head -20
fi
make list

if [ -f cases.txt ]; then
  CASES=$(grep -vE '^\s*(#|$)' cases.txt | tr '\n' ' ')
else
  CASES="6:1 6:4096 8:1 8:2048 17:1 17:256 36:1 36:32"
fi

BACKENDS=$(cd "$BINDIR" && ls)
echo "== backends: $BACKENDS =="

for case in $CASES; do
  L=${case%%:*}; B=${case##*:}
  IN=$OUT/in_L${L}_B${B}.bin
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + L * 1000 + B)) --out "$IN" >/dev/null
  for backend in $BACKENDS; do
    # The dense floor is O(L^4) per volume per axis; past this it only burns wall clock.
    if [ "$backend" = "baseline_gpu" ] && [ $((L * L * L * B)) -gt 40000000 ]; then
      echo "   skipping baseline_gpu at L=$L B=$B (too expensive to be informative)" >> "$OUT/timing.log"
      continue
    fi
    for run in $(seq 1 "$RUNS"); do
      timeout 900 "$BINDIR/$backend" --L "$L" --batch "$B" --in "$IN" \
        --out "$OUT/out_${backend}_L${L}_B${B}.bin" \
        --json "$OUT/t_${backend}_L${L}_B${B}_r${run}.json" \
        --samples "$SAMPLES" --warmup 5 --min-sample-ms 20 --run-index "$run" \
        >>"$OUT/timing.log" 2>>"$OUT/timing.err"
      rc=$?
      if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
        echo "$backend L=$L B=$B run=$run exited $rc" >>"$OUT/failures.txt"
      fi
    done
    if [ -f "$OUT/out_${backend}_L${L}_B${B}.bin" ]; then
      python3 check.py --input "$IN" --output "$OUT/out_${backend}_L${L}_B${B}.bin" \
        --L "$L" --batch "$B" --json "$OUT/c_${backend}_L${L}_B${B}.json" \
        >>"$OUT/check.log" 2>&1
      rm -f "$OUT/out_${backend}_L${L}_B${B}.bin"
    fi
  done
  rm -f "$IN"
  echo "   done L=$L B=$B"
done

# Record the clock state the numbers were taken at: we cannot lock clocks (no permission),
# so this is the only evidence of whether the GPU was boosting throughout.
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active \
           --format=csv,noheader 2>/dev/null | sed 's/^/clocks after sweep: /' >> "$OUT/environment.txt"

python3 leaderboard.py --round "$ROUND" | tee "$OUT/leaderboard.txt"
echo "== round $ROUND complete: $OUT =="
