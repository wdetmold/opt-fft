#!/bin/bash
# GPU leases: how eight agents share the eight CPU core slots of the reserved node without colliding.
#
# A lease is a DIRECTORY, because mkdir is atomic even over NFS -- flock is not reliable
# here, and a "check then create" file test is a race. Whoever succeeds in creating
# leases/slotN owns slot N until it removes it.
#
#   ./slot_lease.sh acquire [--label NAME]   claim any free slot; prints its index
#   ./slot_lease.sh release N                give slot N back
#   ./slot_lease.sh status                   who holds what
#   ./slot_lease.sh acquire-all [--label X]  claim ALL GPUs (the monitor's scoring window)
#   ./slot_lease.sh release-all
#   ./slot_lease.sh reap                     drop leases whose owner is long gone
#
# Two rules that matter:
#   * Hold a lease only while you are actually running on the GPU. Acquire, run, release.
#   * The monitor's scored measurements need the whole node quiet, so acquire-all raises a
#     SCORING flag that makes new single-GPU acquisitions wait. Without that the monitor
#     would starve behind eight agents that each re-acquire immediately.
set -u
cd "$(dirname "$(readlink -f "$0")")"
GPU=$(pwd)
LEASES=$GPU/leases
RES=$GPU/RESERVATION
STALE_MIN=${FFT_LEASE_STALE_MIN:-420}
WAIT_MAX=${FFT_LEASE_WAIT:-1800}

mkdir -p "$LEASES"

# Dev slots are physical cores 2..25 of the reserved node: 24 slots, leaving cores 0-1
# for system noise and 26-31 as headroom. A lease's index IS the core to taskset to.
ngpus() { echo 24; }
slot_to_core() { echo $(( $1 + 2 )); }

reap() {
  local now=$(date +%s) owner ts
  for d in "$LEASES"/slot*; do
    [ -d "$d" ] || continue
    ts=$(stat -c %Y "$d/owner" 2>/dev/null || echo 0)
    if [ $(( (now - ts) / 60 )) -ge "$STALE_MIN" ]; then
      owner=$(head -1 "$d/owner" 2>/dev/null || echo unknown)
      echo "reaping stale lease $(basename "$d") held by $owner for $(( (now - ts) / 60 )) min" >&2
      rm -rf "$d"
    fi
  done
}

claim_one() {  # claim_one <label> -> prints index, or fails
  local label=$1 n
  n=$(ngpus)
  for i in $(seq 0 $((n - 1))); do
    if mkdir "$LEASES/slot$i" 2>/dev/null; then
      printf '%s\n%s\n%s\n%s\n' "$label" "$(hostname -s)" "$$" "$(date -Is)" > "$LEASES/slot$i/owner"
      echo "$i"
      return 0
    fi
  done
  return 1
}

case "${1:-}" in
acquire)
  shift
  LABEL=${USER:-agent}
  [ "${1:-}" = "--label" ] && { LABEL=$2; shift 2; }
  reap
  waited=0
  while :; do
    # Yield to a scoring window rather than competing with it.
    if [ -e "$LEASES/SCORING" ]; then
      [ "$waited" = 0 ] && echo "a scoring window is open; waiting for it to finish" >&2
    else
      idx=$(claim_one "$LABEL") && { echo "$idx"; exit 0; }
    fi
    sleep 10
    waited=$((waited + 10))
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      echo "no GPU free after ${WAIT_MAX}s" >&2
      exit 1
    fi
    [ $((waited % 120)) = 0 ] && reap
  done
  ;;
release)
  idx=${2:-}
  [ -n "$idx" ] || { echo "usage: $0 release N" >&2; exit 2; }
  rm -rf "$LEASES/slot$idx"
  ;;
acquire-all)
  shift
  LABEL=${USER:-monitor}
  [ "${1:-}" = "--label" ] && { LABEL=$2; shift 2; }
  reap
  # Raise the flag FIRST so no new single-GPU claims start while we drain the in-flight ones.
  printf '%s\n%s\n' "$LABEL" "$(date -Is)" > "$LEASES/SCORING"
  n=$(ngpus)
  waited=0
  while :; do
    held=0
    for i in $(seq 0 $((n - 1))); do [ -d "$LEASES/slot$i" ] && held=$((held + 1)); done
    [ "$held" = 0 ] && break
    [ "$waited" = 0 ] && echo "waiting for $held in-flight lease(s) to finish" >&2
    sleep 10
    waited=$((waited + 10))
    if [ "$waited" -ge "$WAIT_MAX" ]; then
      echo "still $held lease(s) held after ${WAIT_MAX}s; reaping and proceeding" >&2
      STALE_MIN=0 reap
      break
    fi
  done
  for i in $(seq 0 $((n - 1))); do
    mkdir -p "$LEASES/slot$i"
    printf '%s\n%s\n%s\n%s\n' "$LABEL (scoring)" "$(hostname -s)" "$$" "$(date -Is)" \
      > "$LEASES/slot$i/owner"
  done
  echo "scoring window open: all $n slots held"
  ;;
release-all)
  rm -rf "$LEASES"/slot* "$LEASES/SCORING"
  echo "scoring window closed"
  ;;
reap)
  reap
  ;;
status)
  n=$(ngpus)
  [ -e "$LEASES/SCORING" ] && echo "SCORING WINDOW OPEN: $(head -1 "$LEASES/SCORING")"
  for i in $(seq 0 $((n - 1))); do
    if [ -d "$LEASES/slot$i" ]; then
      printf '  slot%-2s held by %-28s since %s\n' "$i" \
        "$(sed -n 1p "$LEASES/slot$i/owner" 2>/dev/null)" \
        "$(sed -n 4p "$LEASES/slot$i/owner" 2>/dev/null)"
    else
      printf '  slot%-2s free\n' "$i"
    fi
  done
  ;;
*)
  echo "usage: $0 {acquire [--label NAME]|release N|acquire-all|release-all|reap|status}" >&2
  exit 2
  ;;
esac
