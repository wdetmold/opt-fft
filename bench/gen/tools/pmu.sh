#!/bin/bash
# Hardware-counter run of one command on this node (a80n0; perf_event_paranoid=2, node-local
# /tmp/perf). Counting mode only — negligible overhead, but still do timing runs and counter
# runs separately. usage: tools/pmu.sh [-e extra,events] <command...>
PERF=/tmp/perf
[ -x "$PERF" ] || { echo "ERROR: $PERF missing (node rebooted?) — scp ext/tools/perf-install/bin/perf a80n0:/tmp/perf" >&2; exit 1; }
EXTRA=""
[ "$1" = "-e" ] && { EXTRA=",$2"; shift 2; }
exec "$PERF" stat -e cycles,instructions,uops_dispatched.port_0,uops_dispatched.port_1,uops_dispatched.port_5,uops_dispatched.port_2_3,uops_dispatched.port_4_9,l1d.replacement,core_power.lvl2_turbo_license$EXTRA -- "$@"
