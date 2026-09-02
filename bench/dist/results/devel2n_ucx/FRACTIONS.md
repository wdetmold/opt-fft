# Comm-vs-compute fractions: devel2n_ucx

```
=== comm-vs-compute fractions: devel2n_ucx ===
  date: 2026-09-02T14:18:56-04:00   slurm_job: 440340   nodes: 2
  nodelist: p51n1,p55n3
  cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
  cores/node: 32   isa build: avx2
  mpi: mpirun (Open MPI) 4.1.5rc4
  probe-summary: latency_us=1.784 peak_bw_GBs=8.249 alltoall_ms=0.121
  probe-summary: latency_us=4.680 peak_bw_GBs=10.354 alltoall_ms=403.500

  grid ranks nds algo     fft%   pack%    mpi%  unpack%  local%  resid%   traced_s  per_run_s  cap@3.5x  cap_local
   128     1   1 a2av    38.98    0.00    0.00     0.00   61.02    0.00     0.7812    0.0781     1.386     3.500
   128     8   1 a2av    21.95   24.81   20.98    30.91    1.28    0.06     0.1973    0.0197     1.186     2.294
   128     8   1 p2p_pl   23.58   24.45   21.44    28.52    1.16    0.84     0.1920    0.0192     1.203     2.248
   128    32   1 a2av    19.92   19.84   30.96    28.29    0.86    0.13     0.0596    0.0060     1.166     1.969
   128    32   1 p2p_pl   20.85   16.78   30.87    29.59    0.90    1.01     0.0577    0.0060     1.175     1.948
   128    32   2 a2av     0.13    0.12   99.55     0.19    0.00    0.00     9.1647    0.9165     1.001     1.003
   128    32   2 p2p_pl    0.12    0.10   76.78     0.14    0.01   22.85    10.7361    1.0736     1.001     1.003
   128    64   2 a2av     0.05    0.03   99.86     0.05    0.00    0.00    10.4518    1.0452     1.000     1.001
   128    64   2 p2p_pl    0.10    0.04   77.00     0.09    0.00   22.77     5.9474    0.5948     1.001     1.002
   256     1   1 a2av    34.24    0.00    0.00     0.00   65.76    0.00     7.0240    0.7024     1.324     3.500
   256     8   1 a2av    19.88   24.10   19.97    34.77    1.28    0.01     1.7644    0.1765     1.165     2.334
   256     8   1 p2p_pl   21.19   22.87   20.37    33.47    1.06    1.04     1.7898    0.1790     1.178     2.280
   256    32   1 a2av    15.76   30.62   21.39    30.78    1.42    0.03     0.7109    0.0712     1.127     2.279
   256    32   1 p2p_pl   16.94   25.65   24.77    30.27    1.07    1.30     0.7107    0.0712     1.138     2.119
   256    32   2 a2av     0.53    0.81   97.62     1.00    0.04    0.00    20.4090    2.0410     1.004     1.017
   256    32   2 p2p_pl    0.46    0.61   77.31     0.80    0.03   20.79    24.6479    2.5716     1.003     1.014
   256    64   2 a2av     0.25    0.36   98.93     0.44    0.01    0.00    22.0582    2.2060     1.002     1.008
   256    64   2 p2p_pl    0.23    0.29   82.45     0.37    0.01   16.65    25.1714    2.5172     1.002     1.006

  verification -- traced total vs the benchmark's OWN reported time
  (speed3d times nruns forward + nruns backward, so closure should be 1.000;
   a mis-resolved nesting would still print a tidy table, so this is the check
   that the percentages above are actually complete and counted once each)
    g128   r1    n1 a2av    traced    0.7812s  closure 1.000  nest  leaf
    g128   r8    n1 a2av    traced    0.1973s  closure 0.999  nest 1.000
    g128   r8    n1 p2p_pl  traced    0.1920s  closure 0.999  nest 1.000
    g128   r32   n1 a2av    traced    0.0596s  closure 0.997  nest 1.000
    g128   r32   n1 p2p_pl  traced    0.0577s  closure 0.961  nest 1.000   <-- CLOSURE FAIL
    g128   r32   n2 a2av    traced    9.1647s  closure 1.000  nest 1.000
    g128   r32   n2 p2p_pl  traced   10.7361s  closure 1.000  nest 1.000
    g128   r64   n2 a2av    traced   10.4518s  closure 1.000  nest 1.000
    g128   r64   n2 p2p_pl  traced    5.9474s  closure 1.000  nest 1.000
    g256   r1    n1 a2av    traced    7.0240s  closure 1.000  nest  leaf
    g256   r8    n1 a2av    traced    1.7644s  closure 1.000  nest 1.000
    g256   r8    n1 p2p_pl  traced    1.7898s  closure 1.000  nest 1.000
    g256   r32   n1 a2av    traced    0.7109s  closure 0.999  nest 1.000
    g256   r32   n1 p2p_pl  traced    0.7107s  closure 0.999  nest 1.000
    g256   r32   n2 a2av    traced   20.4090s  closure 1.000  nest 1.000
    g256   r32   n2 p2p_pl  traced   24.6479s  closure 0.958  nest 1.000   <-- CLOSURE FAIL
    g256   r64   n2 a2av    traced   22.0582s  closure 1.000  nest 1.000
    g256   r64   n2 p2p_pl  traced   25.1714s  closure 1.000  nest 1.000
    worst closure deviation: 4.15%

  fft%     = local batched-1D executor call -- the slot our kernel would occupy
  pack%/unpack% = gather into / scatter out of the communication buffers
  mpi%     = the wire (all2allv, or isend/irecv/waitany on the p2p paths)
  local%   = copy/scale outside the transpose, plus childless reshapes (1-rank local copies)
  resid%   = time inside a reshape not covered by its own child events (large on the p2p
             paths); deliberately NOT folded into mpi%, since what it is was not established
  traced_s = summed over the TIMED runs only; the warmup transform is excluded
  cap      = end-to-end Amdahl limit from speeding up ONLY the fft slot
  cap_local= same if a fused kernel also absorbs pack+unpack+local
  The 1-rank rows do no MPI at all: that is the pure-local control.
```
