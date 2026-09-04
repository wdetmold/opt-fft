#!/bin/bash
# Runs on the benchmark node itself.  Rebuilds locally (so -march=native targets THIS
# cpu), generates fresh random data for the round, then measures every backend on every
# case in several independent processes.
#
# usage: sweep.sh --round TAG --seed N [--runs 3] [--samples 20] [--quick]
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

# Every backend is measured under identical threading. Pinning matters more than the count:
# without PROC_BIND the OS migrates threads across the two sockets mid-measurement and the
# run-to-run spread swamps the differences we are trying to see.
export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export MKL_NUM_THREADS=32
export MKL_DYNAMIC=false

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
  echo "threads: $OMP_NUM_THREADS of $(nproc) (PROC_BIND=$OMP_PROC_BIND)   governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
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
if [ -f cases.txt ]; then
  # A geometry wave can extend the sweep by dropping a cases.txt here, rather than by
  # editing this script (which a running job may be executing).
  CASES=$(grep -vE '^\s*(#|$)' cases.txt | tr '\n' ' ')
elif [ "$QUICK" = 1 ]; then
  CASES="6:1 6:512 8:1 8:512 17:1 17:64 36:1 36:8"
else
  CASES="6:1 6:64 6:4096 6:32768 8:1 8:64 8:2048 8:16384 \
         17:1 17:8 17:256 17:2048 36:1 36:4 36:32 36:256"
fi

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
BACKENDS=$(cd "$BINDIR" && ls)
echo "== backends: $BACKENDS =="

for case in $CASES; do
  L=${case%%:*}; B=${case##*:}
  IN=$OUT/in_L${L}_B${B}.bin
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + L * 1000 + B)) --out "$IN" >/dev/null
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
