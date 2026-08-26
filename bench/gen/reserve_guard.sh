#!/bin/bash
# Keep exactly one icehold running or queued; claim only when neither exists.
export PATH=/opt/software/slurm-19.05.8.1-cuda-11.8/bin:$PATH
cd "$(dirname "$(readlink -f "$0")")"
squeue -h -u $USER -n icehold -t RUNNING,PENDING -o %i | grep -q . && exit 0
FFT_ICE_HOURS=10 ./reserve.sh
