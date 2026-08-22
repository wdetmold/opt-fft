#!/bin/bash
# Run the claude CLI on a remote worker host, transparently to the caller.
#
# Exists because wombat runs vm.overcommit_memory=2 with no swap: each claude worker
# reserves several GB of address space at startup, and two panels' workers launched
# together pushed Committed_AS past CommitLimit -- every new worker died at birth with
# JSC "MemoryExhaustion" (exit 134/139) while 100+ GB of RAM sat free.
#
# The host is a parameter: wallaby (503 GB, API access verified) by default; switch to the
# reserved Ice Lake node itself by setting FFT_WORKER_HOST=a80n0 once outbound API access
# is enabled there -- then the workers sit on the same silicon they measure.
set -u
HOST=${FFT_WORKER_HOST:-wallaby}
DIR=$(pwd)
args=$(printf '%q ' "$@")
exec ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$HOST" \
  "cd $(printf '%q' "$DIR") && exec /home/lqcd/wdetmold/.local/bin/claude $args"
