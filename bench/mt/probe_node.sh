#!/bin/bash
# Take YOUR OWN measurement on the isolated benchmark node, for one implementation.
#
# This is the only way to see how your code behaves on the hardware you are scored on:
# Cascade Lake with AVX-512, a 1 MB L2 (the login node's Haswell has 256 KB), and no
# other jobs competing.  It builds only your file plus one library baseline, on that
# node, and times both on identical data.
#
# usage: ./probe_node.sh <impl-name> [--L N] [--batches "1 64 2048"] [--partition devel]
#
# MONITOR ONLY.  The exclusive benchmark node is reserved for the monitor agent's
# cross-checks, so that the scored numbers are taken on an uncontended machine by one
# party using one method.  Implementers develop on wallaby (see tryout.sh --on wallaby):
# it is near-idle, shares this filesystem, and has full AVX-512, so an AVX-512 path can
# be run and measured there -- just not scored there.
#
# Set FFT_MONITOR=1 to confirm you are the monitor.
set -eu
cd "$(dirname "$0")"

if [ "${FFT_MONITOR:-0}" != "1" ]; then
  cat >&2 <<'DENY'
probe_node.sh is reserved for the monitor agent.

The exclusive benchmark node stays uncontended so that every scored number is taken the
same way by one party.  For your own iteration use wallaby, which is idle, shares this
filesystem and has full AVX-512:

    ./tryout.sh --on wallaby <impl-name> [L] [batch]

If you really are the monitor, re-run with FFT_MONITOR=1.
DENY
  exit 3
fi

NAME=${1:-}
[ -n "$NAME" ] || { echo "usage: $0 <impl-name> [--L N] [--batches \"...\"] [--partition P]" >&2; exit 2; }
shift
[ -f "impl/$NAME.c" ] || { echo "impl/$NAME.c does not exist" >&2; exit 2; }

L=$(echo "$NAME" | sed -n 's/^L\([0-9]\+\)_.*/\1/p')
BATCHES=""
PARTITION=devel
while [ $# -gt 0 ]; do
  case "$1" in
    --L) L=$2; shift 2 ;;
    --batches) BATCHES=$2; shift 2 ;;
    --partition) PARTITION=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$L" ] || { echo "could not infer L from '$NAME'; pass --L" >&2; exit 2; }
if [ -z "$BATCHES" ]; then
  # non-batched plus one cache-resident and one memory-resident batched point
  case "$L" in
    6)  BATCHES="1 64 4096" ;;
    8)  BATCHES="1 64 2048" ;;
    17) BATCHES="1 8 256" ;;
    36) BATCHES="1 4 32" ;;
    *)  BATCHES="1 64" ;;
  esac
fi

OUT=results/probe_${NAME}
mkdir -p "$OUT"

# The job runs a COPY of this logic, so that editing these scripts while a job is in
# flight cannot corrupt a running job (bash reads scripts incrementally -- this has
# already bitten us once).
# On the shared filesystem, not /tmp: /tmp is node-local, so a job handed a /tmp path
# fails with "not found" (exit 127).
JOB=$(pwd)/.probe_snapshot_$NAME.sh
cat > "$JOB" <<INNER
#!/bin/bash
set -u
cd $(pwd)
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
W=\$(mktemp -d)
{
  echo "host: \$(hostname)"
  lscpu | sed -n 's/^Model name: *//p'
  echo "isa: \$(lscpu | grep -o -E 'avx512[a-z_]*|avx2|fma' | sort -u | tr '\n' ' ')"
  echo "L2: \$(lscpu | sed -n 's/^L2 cache: *//p')   L3: \$(lscpu | sed -n 's/^L3 cache: *//p')"
} | tee $OUT/environment.txt

echo "== building on this node with -march=native (AVX-512 available) =="
gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -funroll-loops \\
    -I. -o \$W/mine impl/$NAME.c driver.c -lm || { echo "BUILD FAILED"; exit 1; }
gcc -O3 -march=native -mtune=native -std=gnu11 -fno-math-errno -I. -I\$MKLROOT/include \\
    -o \$W/mkl sota/mkl_dfti.c driver.c -L\$MKLROOT/lib/intel64 \\
    -Wl,-rpath,\$MKLROOT/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm \\
    2>/dev/null || echo "(baseline build failed; continuing without it)"

for B in $BATCHES; do
  python3 gen_input.py --L $L --batch \$B --seed 31337 --out \$W/in.bin >/dev/null
  echo "-- L=$L B=\$B --"
  for exe in \$W/mine \$W/mkl; do
    [ -x "\$exe" ] || continue
    \$exe --L $L --batch \$B --in \$W/in.bin --out \$W/out.bin --samples 20 --warmup 5
    python3 check.py --input \$W/in.bin --output \$W/out.bin --L $L --batch \$B | sed 's/^/      /'
  done
done
rm -rf \$W
echo "== probe of $NAME complete =="
INNER
chmod +x "$JOB"

sbatch --job-name="probe-$NAME" --partition="$PARTITION" --exclusive --nodes=1 \
       --time=15 --cpu-freq=Performance --output="$OUT/probe-%j.out" "$JOB"
echo
echo "watch it:   squeue -u $USER | grep probe"
echo "results in: $OUT/probe-<jobid>.out"
