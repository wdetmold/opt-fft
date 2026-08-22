#!/bin/bash
# Run a command on the reserved GPU node, on one leased GPU.
#
#   ./on_gpu.sh [--label NAME] -- <command...>
#
# It leases a GPU, ssh's to the reserved node, runs the command there with
# CUDA_VISIBLE_DEVICES pointing at that GPU and this directory as the working directory (the
# filesystem is shared, so nothing needs copying), and releases the lease afterwards --
# including if the command fails or you interrupt it.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GPU=$(pwd)
LABEL=${FFT_AGENT_LABEL:-${USER:-agent}}

[ "${1:-}" = "--label" ] && { LABEL=$2; shift 2; }
[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: $0 [--label NAME] -- <command...>" >&2; exit 2; }

if [ ! -f "$GPU/RESERVATION" ]; then
  echo "no GPU reservation: run ./reserve.sh first (or ask the monitor to)" >&2
  exit 2
fi
# shellcheck disable=SC1090
. "$GPU/RESERVATION"
if ! squeue -h -j "${RES_JOB:-0}" -o '%t' 2>/dev/null | grep -q '^R$'; then
  echo "reservation job ${RES_JOB:-?} is no longer running; ./reserve.sh --status" >&2
  exit 2
fi

IDX=$("$GPU/gpu_lease.sh" acquire --label "$LABEL") || exit 1
trap '"$GPU/gpu_lease.sh" release "$IDX" >/dev/null 2>&1' EXIT INT TERM

ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$RES_NODE" \
  "cd '$GPU' && source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1 && \
   export CUDA_VISIBLE_DEVICES=$IDX && $*"
rc=$?
exit $rc
