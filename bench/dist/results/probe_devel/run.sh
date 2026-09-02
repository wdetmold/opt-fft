#!/bin/bash
# Why is inter-node MPI on the prod-class nodes ~100x slower than intra-node for the same
# transpose?  Measure the fabric directly, and enumerate what transports OpenMPI can see.
set -u
cd /home/lqcd/wdetmold/fft/bench/dist
source /home/lqcd/wdetmold/fft/env.sh >/dev/null 2>&1
O=results/probe_devel
mpicc -O2 -o $O/fabric_probe fabric_probe.c
echo "== node interfaces and IB devices =="
for h in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
  echo "-- $h"
  srun --nodes=1 --ntasks=1 -w "$h" bash -c \
    'ip -o link show | awk "{print \$2, \$9}"; echo "ib:"; ls /sys/class/infiniband 2>/dev/null || echo none'
done
echo "== openmpi transports available =="
ompi_info | grep -E "^ *MCA (btl|pml|mtl|osc)" | sort -u
echo "== default (whatever OpenMPI picks) =="
mpirun -np 2 --map-by ppr:1:node -x LD_LIBRARY_PATH $O/fabric_probe
echo "== forced ucx =="
mpirun -np 2 --map-by ppr:1:node --mca pml ucx -x LD_LIBRARY_PATH $O/fabric_probe 2>&1 | tail -8
echo "== forced tcp =="
mpirun -np 2 --map-by ppr:1:node --mca pml ob1 --mca btl tcp,self -x LD_LIBRARY_PATH $O/fabric_probe 2>&1 | tail -8
