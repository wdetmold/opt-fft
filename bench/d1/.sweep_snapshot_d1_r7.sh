#!/bin/bash
# Runs on the benchmark node itself.  Rebuilds locally (so -march=native targets THIS
# cpu), generates fresh random data for the round, then measures every backend on every
# case in several independent processes.
#
# usage: sweep.sh --round TAG --seed N [--runs 3] [--samples 20] [--quick] [--fresh]
#
# The sweep is RESUMABLE: an interrupted round can be re-run with the same arguments and
# it will skip the cells it already measured.  See the resume block below for why that is
# safe only when the measurement is provably identical.
set -u
cd "$(dirname "$0")"
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1

ROUND=""; SEED=0; RUNS=3; SAMPLES=20; QUICK=0; CASEFILE=cases.txt; FRESH=0
SHARD=""; BOARD=1
# Single-call cells are measured 3x more often than chained ones.  The driver's WITHIN-process
# precision is fine either way (m=1 uses ~1.2M inner reps, sd/min 0.5%), but the spread ACROSS
# independent processes -- core placement, cache state -- is 9.0% median for m=1 against 1.1%
# for chained, so verdicts within ~10% of parity were not resolvable at 3 runs.  Extra runs
# only cost wall clock, which sharding across both nodes has now freed up.
RUNS_ONCE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --round) ROUND=$2; shift 2 ;;
    --seed) SEED=$2; shift 2 ;;
    --runs) RUNS=$2; shift 2 ;;
    --runs-once) RUNS_ONCE=$2; shift 2 ;;   # independent runs for m=1 cells
    --samples) SAMPLES=$2; shift 2 ;;
    --quick) QUICK=1; shift ;;
    --cases) CASEFILE=$2; shift 2 ;;
    --fresh) FRESH=1; shift ;;      # ignore any existing results and remeasure everything
    # Grading is sharded across the reserved nodes (see submit.sh).  Each shard measures a
    # DISJOINT set of cells into the shared round directory: the per-cell JSON names are
    # unique, but every other file this script writes -- manifest, done-list, logs,
    # environment record -- is per-shard and must be suffixed, or two nodes racing on one
    # NFS directory would clobber each other's state.  The manifest is the dangerous one:
    # sharing it would let one shard's resume decision be judged against the other's.
    --shard) SHARD="_$2"; shift 2 ;;
    --no-board) BOARD=0; shift ;;   # submit.sh builds one leaderboard after all shards finish
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$ROUND" ] || { echo "sweep.sh: --round is required" >&2; exit 2; }
# Default single-call cells to 3x the chained run count.  Defaulting rather than requiring a
# flag means the runner needs no change, and 3x is what the noise asks for: the inter-process
# spread is ~9% for m=1 against ~1.1% chained, and a median over 3x the processes cuts that by
# about sqrt(3).  In r3 that would move 6 of the 26 once cells from "verdict inside the noise"
# to resolved -- cells like L=64 B=512/once, reported as a 1.11x win with +/-18% of noise.
RUNS_ONCE=${RUNS_ONCE:-$((RUNS * 3))}

OUT=results/$ROUND
mkdir -p "$OUT"

# ---- resume ------------------------------------------------------------------------------
# A scoring sweep runs for hours inside a reservation that can expire underneath it.  d1_r1
# lost its node 3.5 h in; the retry restarted from cell 1 and measured everything twice.
# Every per-cell artifact is already named deterministically, so the work is resumable -- but
# ONLY if the files on disk provably come from the same measurement.  Resuming across a
# different seed, case list, run/sample count or a changed implementation would silently
# splice two generations of results into one leaderboard, which is worse than redoing the
# work.  So a manifest records what produced the directory, and a mismatch moves the old
# artifacts aside instead of mixing them.
MANIFEST="$OUT/.manifest$SHARD"
# The SOURCES decide the numbers, not the binaries: make clean rebuilds those every attempt,
# so their hashes differ even when nothing changed.
SRCFP=$(cat driver.c fft1d_api.h sota/*.c impl/*.c 2>/dev/null | md5sum | cut -d" " -f1)
WANT="round=$ROUND seed=$SEED runs=$RUNS/$RUNS_ONCE samples=$SAMPLES cases=$CASEFILE src=$SRCFP"
RESUME=0
if [ -f "$MANIFEST" ] && [ "$FRESH" = 0 ]; then
  HAVE=$(cat "$MANIFEST")
  if [ "$HAVE" = "$WANT" ]; then
    RESUME=1
  else
    echo "== NOT resuming $ROUND: manifest differs, so the existing results are from a"
    echo "   DIFFERENT measurement and must not be mixed into one leaderboard."
    echo "     was: $HAVE"
    echo "     now: $WANT"
    if [ -n "$SHARD" ]; then
      # A shard must NOT sweep the round directory: the per-cell JSON names are unique but
      # the glob is round-wide, so moving "the old artifacts" aside would take the OTHER
      # shard's results with them -- while that shard is still running.  A shard therefore
      # just remeasures its own disjoint cells, overwriting them in place.
      echo "   shard${SHARD}: remeasuring this shard's own cells in place (round-wide"
      echo "   artifacts left alone -- another shard may be using them)"
    else
      STALE="$OUT/stale_$(date +%Y%m%d-%H%M%S)"
      mkdir -p "$STALE"
      mv "$OUT"/t_*.json "$OUT"/c_*.json "$OUT"/o_*.json "$STALE/" 2>/dev/null
      echo "   moved the old artifacts to $STALE"
    fi
  fi
fi
printf "%s" "$WANT" > "$MANIFEST"

# Which (backend,L,B,m) units are already complete.  Scanned ONCE into a file rather than
# re-tested per unit, so resume costs one python call instead of thousands.
DONELIST="$OUT/.done_units$SHARD"
: > "$DONELIST"
if [ "$RESUME" = 1 ]; then
  python3 - "$OUT" "$RUNS" "$RUNS_ONCE" > "$DONELIST" <<'PY'
import glob, json, os, re, sys
out, runs, runs_once = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
pat = re.compile(r"^t_(?P<b>.+)_L(?P<L>\d+)_B(?P<B>\d+)_m(?P<m>\d+)_r(?P<r>\d+)\.json$")

def load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:          # missing, truncated by a kill mid-write, or unparseable
        return None

units = {}
for path in glob.glob(os.path.join(out, "t_*_r*.json")):
    mo = pat.match(os.path.basename(path))
    if mo:
        units.setdefault((mo["b"], mo["L"], mo["B"], mo["m"]), set()).add(int(mo["r"]))

for (b, L, B, m), have in sorted(units.items()):
    want = runs_once if int(m) == 1 else runs      # m=1 cells are run more often
    if set(range(1, want + 1)) - have:
        continue                                    # not every run recorded
    ds = [load(os.path.join(out, f"t_{b}_L{L}_B{B}_m{m}_r{r}.json"))
          for r in range(1, want + 1)]
    if any(d is None for d in ds):
        continue                                    # a partial write is NOT a result
    # An unsupported backend writes timing JSONs with supported=false and produces no
    # output to check, so it is complete with no verdict files at all.
    if all(not d.get("supported", False) for d in ds):
        print(b, L, B, m)
        continue
    if load(os.path.join(out, f"c_{b}_L{L}_B{B}_m{m}.json")) is None:
        continue                                    # correctness verdict missing/partial
    if int(m) > 1 and load(os.path.join(out, f"o_{b}_L{L}_B{B}_m{m}.json")) is None:
        continue                                    # one-step gate missing/partial
    print(b, L, B, m)
PY
  echo "== RESUMING $ROUND: $(wc -l < "$DONELIST") complete units on disk will be skipped =="
fi

unit_done() {   # backend L B m
  [ "$RESUME" = 1 ] || return 1
  grep -qxF "$1 $2 $3 $4" "$DONELIST"
}

{
  echo "# round $ROUND"
  echo "host: $(hostname)   date: $(date -Is)   slurm_job: ${SLURM_JOB_ID:-none}"
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "isa: $(lscpu | grep -o -E 'avx512[a-z_]*|avx2|fma' | sort -u | tr '\n' ' ')"
  echo "cores: $(nproc)   governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
  echo "gcc: $(gcc --version | head -1)"
} | tee "$OUT/environment$SHARD.txt"

# Compilation happens here, outside every timed region.
echo "== building on $(hostname) =="
make clean >/dev/null 2>&1
if ! make -k 2>"$OUT/build_errors$SHARD.txt"; then
  echo "!! some backends failed to build; see $OUT/build_errors$SHARD.txt"
  grep -E 'error' "$OUT/build_errors$SHARD.txt" | head -20
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
BACKENDS=${BACKENDS:-$(cd "$BINDIR" && ls)}
# Refuse to produce a "leaderboard" that contains only libraries: if no panel binary was
# built, the round measured nothing of ours and a board would read as a legitimate loss.
# ALLOW_NO_PANEL=1 opts out, for the one legitimate library-only run: measuring the SOTA
# baseline table itself, which by definition has no panel entries.  Do not set it for a
# scoring round -- that is precisely the silent-zero the guard exists to catch.
npanel=$(cd "$BINDIR" && ls | grep -vcE "^(mkl|fftw|ducc|baseline)" || true)
if [ "${npanel:-0}" -eq 0 ] && [ "${ALLOW_NO_PANEL:-0}" != 1 ]; then
  echo "ABORT: no PANEL BINARIES in $BINDIR -- impl/ is empty or the build failed" | tee -a "$OUT/build_errors$SHARD.txt"
  exit 4
fi
echo "== backends: $BACKENDS =="

for case in $CASES; do
  L=${case%%:*}
  rest=${case#*:}
  B=${rest%%:*}
  # optional third field: chain length m (the graded call); absent means 1
  M=1; case "$rest" in *:*) M=${rest##*:} ;; esac
  # Resume at CASE granularity first: if every backend for this case is already complete
  # there is no reason to regenerate its inputs, which for L=100003 is not cheap.
  if [ "$RESUME" = 1 ]; then
    pending=0
    for backend in $BACKENDS; do
      if [ "$backend" = "baseline_dft" ] && [ $((L * L * B)) -gt 8000000 ]; then continue; fi
      unit_done "$backend" "$L" "$B" "$M" || pending=1
    done
    if [ "$pending" = 0 ]; then
      echo "   skip L=$L B=$B m=$M (already complete)"
      continue
    fi
  fi
  IN=$OUT/in_L${L}_B${B}.bin
  CIN=$OUT/c_L${L}_B${B}.bin
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + L * 1000 + B)) --out "$IN" >/dev/null
  python3 gen_input.py --L "$L" --batch "$B" --seed $((SEED + 900000 + L * 1000 + B)) --scale 0.1 --out "$CIN" >/dev/null
  for backend in $BACKENDS; do
    # The dense-matrix floor is O(L^4) per volume per axis, so on a big case it costs more
    # wall clock than every real backend combined (2.8 s per call at 36^3 x 256). It is a
    # sanity floor, not a contender: skip it once the case is large.
    if [ "$backend" = "baseline_dft" ] && [ $((L * L * B)) -gt 8000000 ]; then
      echo "   skipping baseline_dft at L=$L B=$B (too expensive to be informative)" \
        >> "$OUT/timing$SHARD.log"
      continue
    fi
    if unit_done "$backend" "$L" "$B" "$M"; then
      echo "   skip $backend L=$L B=$B m=$M (already measured)" >> "$OUT/timing$SHARD.log"
      continue
    fi
    nruns=$RUNS; [ "$M" -eq 1 ] && nruns=$RUNS_ONCE
    for run in $(seq 1 "$nruns"); do
      # A panel entry that hangs or crashes must not take the round down with it.
      CHAINARGS=""
      [ "$M" -gt 1 ] && CHAINARGS="--chain $M --map --cin $CIN"
      timeout 600 "$BINDIR/$backend" --L "$L" --batch "$B" $CHAINARGS --in "$IN" \
        --out "$OUT/out_${backend}_L${L}_B${B}_m${M}.bin" \
        --json "$OUT/t_${backend}_L${L}_B${B}_m${M}_r${run}.json" \
        --samples "$SAMPLES" --warmup 5 --min-sample-ms 20 --run-index "$run" \
        >>"$OUT/timing$SHARD.log" 2>>"$OUT/timing$SHARD.err"
      rc=$?
      if [ $rc -ne 0 ] && [ $rc -ne 3 ]; then
        echo "$backend L=$L B=$B run=$run exited $rc" >>"$OUT/failures$SHARD.txt"
      fi
    done
    # correctness on the output of the last run, against numpy
    if [ -f "$OUT/out_${backend}_L${L}_B${B}_m${M}.bin" ]; then
      CHKARGS=""
      [ "$M" -gt 1 ] && CHKARGS="--map-check $M --cin $CIN"
      python3 check.py --input "$IN" --output "$OUT/out_${backend}_L${L}_B${B}_m${M}.bin" \
        --L "$L" --batch "$B" $CHKARGS --json "$OUT/c_${backend}_L${L}_B${B}_m${M}.json" \
        >>"$OUT/check$SHARD.log" 2>&1
      rm -f "$OUT/out_${backend}_L${L}_B${B}_m${M}.bin" "$OUT/out_${backend}_L${L}_B${B}_m${M}.bin.chain"   # outputs are large; keep the verdicts
    fi
    # ONE-STEP map gate (chained cases only): two steps of the graded map must match
    # numpy to 3e-14 (chaos cannot amplify in 2 steps; fp32-seeded maps land ~5e-12). This carries the precision contract; the chain gate above only
    # catches gross cheats, because the chain is chaotic (docs/GRADER.md).
    if [ "$M" -gt 1 ]; then
      timeout 120 "$BINDIR/$backend" --L "$L" --batch "$B" --chain 2 --map --cin "$CIN" \
        --in "$IN" --out "$OUT/one_${backend}_L${L}_B${B}_m${M}.bin" \
        --json "$OUT/t1_${backend}_L${L}_B${B}_m${M}.json" \
        --samples 1 --warmup 1 --min-sample-ms 1 --run-index 1 \
        >>"$OUT/timing$SHARD.log" 2>>"$OUT/timing$SHARD.err"
      if [ -f "$OUT/one_${backend}_L${L}_B${B}_m${M}.bin" ]; then
        python3 check.py --input "$IN" --output "$OUT/one_${backend}_L${L}_B${B}_m${M}.bin" \
          --L "$L" --batch "$B" --map-check 2 --cin "$CIN" \
          --json "$OUT/o_${backend}_L${L}_B${B}_m${M}.json" >>"$OUT/check$SHARD.log" 2>&1
      fi
      rm -f "$OUT/one_${backend}_L${L}_B${B}_m${M}.bin" "$OUT/one_${backend}_L${L}_B${B}_m${M}.bin.chain" "$OUT/t1_${backend}_L${L}_B${B}_m${M}.json"
    fi
  done
  rm -f "$IN" "$CIN"
  echo "   done L=$L B=$B"
done

if [ "$BOARD" = 1 ]; then
  python3 leaderboard.py --round "$ROUND" | tee "$OUT/leaderboard.txt"
else
  echo "== shard${SHARD:-} done; leaderboard deferred to submit.sh =="
fi
echo "== round $ROUND complete: $OUT =="
