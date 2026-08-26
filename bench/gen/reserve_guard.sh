#!/bin/bash
# Keep exactly one icehold running or queued; claim only when neither exists.
cd "$(dirname "$(readlink -f "$0")")"
squeue -h -u $USER -n icehold -t RUNNING,PENDING -o %i | grep -q . && exit 0
FFT_ICE_HOURS=10 ./reserve.sh
