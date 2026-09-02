# Comm-vs-compute fractions: devel2n

```
=== comm-vs-compute fractions: devel2n ===
  date: 2026-09-02T14:08:41-04:00   slurm_job: 440337   nodes: 2
  nodelist: p51n1,p55n3
  cpu: Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz
  cores/node: 32   isa build: avx2
  mpi: mpirun (Open MPI) 4.1.5rc4

  grid ranks nds algo     fft%   pack%    mpi%  unpack%  local%  resid%   traced_s  per_run_s  cap@3.5x  cap_local
   128     1   1 a2av    39.51    0.00    0.00     0.00   60.49    0.00     0.7944    0.0794     1.393     3.500
   128     8   1 a2av    22.12   24.85   20.73    30.91    1.32    0.06     0.1982    0.0198     1.188     2.303
   128     8   1 p2p_pl   23.45   24.69   21.53    28.30    1.16    0.86     0.1929    0.0193     1.201     2.244
   128    32   1 a2av    19.82   20.00   31.17    28.05    0.82    0.13     0.0598    0.0060     1.165     1.963
   128    32   1 p2p_pl   21.03   16.79   29.82    30.24    0.94    1.19     0.0571    0.0057     1.177     1.972
   128    32   2 a2av     0.19    0.17   99.36     0.28    0.01    0.00     6.4478    0.6448     1.001     1.005
   128    32   2 p2p_pl    0.11    0.09   81.40     0.13    0.01   18.26    11.5148    1.1515     1.001     1.002
   128    64   2 a2av     0.05    0.03   99.87     0.05    0.00    0.00    10.5634    1.0564     1.000     1.001
   128    64   2 p2p_pl    0.04    0.02   82.45     0.04    0.00   17.44    13.9531    1.3954     1.000     1.001
   256     1   1 a2av    34.35    0.00    0.00     0.00   65.65    0.00     7.0403    0.7040     1.325     3.500
   256     8   1 a2av    19.44   24.23   20.40    34.58    1.33    0.01     1.7630    0.1764     1.161     2.318
   256     8   1 p2p_pl   20.23   22.41   21.86    33.34    1.03    1.12     1.7791    0.1780     1.169     2.223
   256    32   1 a2av    15.66   30.06   22.33    30.50    1.42    0.03     0.7167    0.0717     1.126     2.245
   256    32   1 p2p_pl   17.49   25.94   22.57    31.61    1.08    1.33     0.6876    0.0689     1.143     2.191
   256    32   2 a2av     0.47    0.76   97.80     0.93    0.04    0.00    22.5531    2.2555     1.003     1.016
   256    32   2 p2p_pl    0.42    0.54   75.33     0.71    0.03   22.98    27.9262    2.7928     1.003     1.012
   256    64   2 a2av     0.31    0.42   98.74     0.52    0.02    0.00    18.7571    1.8758     1.002     1.009
   256    64   2 p2p_pl    0.26    0.33   81.88     0.43    0.02   17.08    21.9929    2.2526     1.002     1.008

  verification -- traced total vs the benchmark's OWN reported time
  (speed3d times nruns forward + nruns backward, so closure should be 1.000;
   a mis-resolved nesting would still print a tidy table, so this is the check
   that the percentages above are actually complete and counted once each)
    g128   r1    n1 a2av    traced    0.7944s  closure 1.000  nest  leaf
    g128   r8    n1 a2av    traced    0.1982s  closure 0.999  nest 1.000
    g128   r8    n1 p2p_pl  traced    0.1929s  closure 0.999  nest 1.000
    g128   r32   n1 a2av    traced    0.0598s  closure 0.996  nest 1.000
    g128   r32   n1 p2p_pl  traced    0.0571s  closure 0.996  nest 1.000
    g128   r32   n2 a2av    traced    6.4478s  closure 1.000  nest 1.000
    g128   r32   n2 p2p_pl  traced   11.5148s  closure 1.000  nest 1.000
    g128   r64   n2 a2av    traced   10.5634s  closure 1.000  nest 1.000
    g128   r64   n2 p2p_pl  traced   13.9531s  closure 1.000  nest 1.000
    g256   r1    n1 a2av    traced    7.0403s  closure 1.000  nest  leaf
    g256   r8    n1 a2av    traced    1.7630s  closure 1.000  nest 1.000
    g256   r8    n1 p2p_pl  traced    1.7791s  closure 0.999  nest 1.000
    g256   r32   n1 a2av    traced    0.7167s  closure 0.999  nest 1.000
    g256   r32   n1 p2p_pl  traced    0.6876s  closure 0.999  nest 1.000
    g256   r32   n2 a2av    traced   22.5531s  closure 1.000  nest 1.000
    g256   r32   n2 p2p_pl  traced   27.9262s  closure 1.000  nest 1.000
    g256   r64   n2 a2av    traced   18.7571s  closure 1.000  nest 1.000
    g256   r64   n2 p2p_pl  traced   21.9929s  closure 0.976  nest 1.000
    worst closure deviation: 2.37%

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
