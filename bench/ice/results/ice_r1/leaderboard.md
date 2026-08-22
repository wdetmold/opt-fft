```
=== round ice_r1 ===
# round ice_r1
host: a80n0.lqcd.mit   date: 2026-08-22T16:15:05-04:00   slurm_job: 438572
cpu: Intel(R) Xeon(R) Gold 6326 CPU @ 2.90GHz
isa: avx2 avx512_bitalg avx512bw avx512cd avx512dq avx512f avx512ifma avx512_vbmi avx512vbmi avx512vl avx512_vnni avx512_vpopcntdq fma 
cores: 64   governor: schedutil
gcc: gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0

-- L=6 (batched B=64, chain m=4856), volume 216, working set 0.42 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L6_unrolled                    0.219 us 68044.280 us    38.25       3.2%    0.445s  ok 2.4e-16       1.00x
   L6_pfa                         0.220 us 68269.476 us    38.13      13.5%    0.431s  ok 2.4e-16       1.00x
   mkl_dfti                       0.341 us 106071.288 us    24.54       2.2%    0.003s  ok 2.4e-16       1.56x
   mkl2026_dfti                   0.358 us 111256.961 us    23.40       4.6%    0.003s  ok 2.5e-16       1.64x
   fftw3_patient                  0.442 us 137425.987 us    18.94       2.5%    0.020s  ok 2.0e-16       2.02x
   fftw3_measure                  0.450 us 139722.846 us    18.63       3.9%    0.012s  ok 2.0e-16       2.05x
   fftw3_estimate                 0.957 us 297438.833 us     8.75       1.4%    0.001s  ok 2.0e-16       4.37x
   ducc0_c2c                      2.025 us 629445.903 us     4.14       4.1%    0.000s  ok 1.9e-16       9.25x
   baseline_matrix                7.179 us 2231132.520 us     1.17       0.0%    0.000s  ok 6.0e-16       32.79x

-- L=8 (batched B=64, chain m=2572), volume 512, working set 1.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L8_batchsimd                   0.550 us 90482.886 us    41.91       3.2%    0.022s  ok 2.3e-16       1.00x
   L8_fusedaxes                   0.556 us 91529.529 us    41.44      16.6%    0.054s  ok 2.3e-16       1.01x
   L8_radix8                      0.561 us 92378.585 us    41.05       3.6%    0.032s  ok 2.3e-16       1.02x
   mkl_dfti                       0.623 us 102632.203 us    36.95       2.9%    0.001s  ok 1.6e-16       1.13x
   mkl2026_dfti                   0.638 us 105022.209 us    36.11       8.6%    0.001s  ok 1.6e-16       1.16x
   fftw3_patient                  1.003 us 165150.134 us    22.96       6.3%    0.022s  ok 1.8e-16       1.83x
   fftw3_measure                  1.014 us 166991.127 us    22.71      11.2%    0.012s  ok 1.8e-16       1.85x
   ducc0_c2c                      2.919 us 480478.000 us     7.89       3.8%    0.000s  ok 1.3e-16       5.31x
   fftw3_estimate                 4.359 us 717565.470 us     5.29       0.1%    0.001s  ok 1.7e-16       7.93x
   baseline_matrix               21.433 us 3528116.740 us     1.07       0.3%    0.000s  ok 3.9e-16       38.99x

-- L=13 (batched B=32, chain m=1278), volume 2197, working set 2.15 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L13_direct                     4.661 us 190614.767 us    26.16       1.7%    0.008s  ok 2.9e-16       1.00x
   mkl2026_dfti                   6.066 us 248063.175 us    20.10       1.3%    0.003s  ok 3.2e-16       1.30x
   L13_rader                      6.131 us 250732.974 us    19.89       0.8%    0.009s  ok 4.0e-16       1.32x
   mkl_dfti                       6.241 us 255225.334 us    19.54       2.6%    0.002s  ok 3.2e-16       1.34x
   fftw3_measure                  7.692 us 314583.241 us    15.85       8.4%    0.012s  ok 3.2e-16       1.65x
   fftw3_patient                  7.772 us 317852.775 us    15.69       1.5%    0.022s  ok 3.2e-16       1.67x
   fftw3_estimate                10.031 us 410215.027 us    12.16       1.0%    0.001s  ok 3.2e-16       2.15x
   ducc0_c2c                     29.438 us 1203913.770 us     4.14       2.2%    0.000s  ok 2.5e-16       6.32x
   baseline_matrix              149.906 us 6130551.870 us     0.81       0.1%    0.000s  ok 7.9e-16       32.16x

-- L=17 (batched B=32, chain m=98), volume 4913, working set 4.80 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L17_matrixsimd                14.471 us 45382.137 us    20.82       2.3%    1.352s  ok 3.3e-16       1.00x
   L17_winograd                  16.113 us 50529.111 us    18.70       3.8%    0.737s  ok 3.3e-16       1.11x
   L17_rader                     19.099 us 59895.989 us    15.77       3.4%    0.701s  ok 3.2e-16       1.32x
   mkl2026_dfti                  76.008 us 238359.815 us     3.96       0.2%    0.049s  ok 3.1e-16       5.25x
   mkl_dfti                      76.297 us 239266.412 us     3.95       0.1%    0.047s  ok 3.1e-16       5.27x
   ducc0_c2c                     77.145 us 241926.489 us     3.90       1.0%    0.000s  ok 2.6e-16       5.33x
   fftw3_measure                 80.056 us 251056.474 us     3.76       0.3%    0.007s  ok 3.0e-16       5.53x
   fftw3_estimate                80.089 us 251158.741 us     3.76       0.3%    0.002s  ok 3.0e-16       5.53x
   fftw3_patient                 80.366 us 252026.534 us     3.75       0.8%    0.017s  ok 3.0e-16       5.55x
   baseline_matrix              436.625 us 1369256.980 us     0.69       0.0%    0.000s  ok 8.4e-16       30.17x

-- L=23 (batched B=16, chain m=165), volume 12167, working set 5.94 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L23_rader                     39.142 us 103336.061 us    21.09       0.7%    0.487s  ok 3.8e-16       1.00x
   L23_matrixsimd                39.584 us 104501.055 us    20.86       1.8%    0.395s  ok 3.8e-16       1.01x
   ducc0_c2c                    220.216 us 581370.397 us     3.75       2.8%    0.000s  ok 2.8e-16       5.63x
   mkl_dfti                     231.111 us 610132.195 us     3.57       0.0%    0.054s  ok 4.2e-16       5.90x
   fftw3_estimate               243.153 us 641924.670 us     3.40       0.1%    0.002s  ok 3.7e-16       6.21x
   fftw3_measure                243.433 us 642662.113 us     3.39       0.9%    0.011s  ok 3.7e-16       6.22x
   fftw3_patient                243.594 us 643088.793 us     3.39       0.2%    0.028s  ok 3.7e-16       6.22x
   mkl2026_dfti                 252.535 us 666691.871 us     3.27       0.1%    0.050s  ok 4.2e-16       6.45x
   baseline_matrix             1454.337 us 3839448.480 us     0.57       0.1%    0.000s  ok 7.4e-16       37.15x

-- L=36 (batched B=8, chain m=64), volume 46656, working set 11.39 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L36_pfa                      119.163 us 61011.429 us    30.36       1.4%    0.264s  ok 3.6e-16       1.00x
   L36_mixedradix               119.530 us 61199.578 us    30.27      12.0%    0.203s  ok 3.6e-16       1.00x
   L36_pencilfused              123.594 us 63279.963 us    29.27       1.4%    0.372s  ok 3.8e-16       1.04x
   mkl_dfti                     160.688 us 82272.045 us    22.52       0.3%    0.054s  ok 3.9e-16       1.35x
   mkl2026_dfti                 163.529 us 83726.812 us    22.13       0.0%    0.050s  ok 4.0e-16       1.37x
   fftw3_patient                255.561 us 130847.252 us    14.16       2.0%    0.938s  ok 3.6e-16       2.14x
   fftw3_measure                299.435 us 153310.603 us    12.08       3.3%    0.124s  ok 3.8e-16       2.51x
   ducc0_c2c                    315.951 us 161766.982 us    11.45       4.7%    0.000s  ok 3.1e-16       2.65x
   fftw3_estimate               366.283 us 187536.814 us     9.88       1.6%    0.004s  ok 3.5e-16       3.07x
   baseline_matrix             8697.588 us 4453165.290 us     0.42       0.0%    0.000s  ok 8.0e-16       72.99x

-- L=45 (batched B=4, chain m=177), volume 91125, working set 11.12 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   L45_mixedradix               284.202 us 201215.054 us    26.41       2.6%    0.176s  ok 4.1e-16       1.00x
   L45_pfa                      295.054 us 208898.020 us    25.44      12.0%    0.534s  ok 4.0e-16       1.04x
   mkl_dfti                     521.591 us 369286.291 us    14.39       0.1%    0.090s  ok 4.4e-16       1.84x
   mkl2026_dfti                 541.638 us 383479.979 us    13.86       0.1%    0.048s  ok 4.5e-16       1.91x
   ducc0_c2c                    654.135 us 463127.530 us    11.48       1.9%    0.000s  ok 3.7e-16       2.30x
   fftw3_patient                679.896 us 481366.211 us    11.04       3.5%    0.546s  ok 4.3e-16       2.39x
   fftw3_measure                777.548 us 550503.789 us     9.65       2.6%    0.063s  ok 4.2e-16       2.74x
   fftw3_estimate               977.751 us 692247.652 us     7.68       0.5%    0.002s  ok 4.3e-16       3.44x
   baseline_matrix            21447.922 us 15185128.600 us     0.35       0.3%    0.000s  ok 8.0e-16       75.47x

-- L=64 (batched B=2, chain m=134), volume 262144, working set 16.00 MiB --
   backend                   per-transform     per-call     GF/s  run spread     setup  correctness
   mkl_dfti                    1016.396 us 272394.206 us    23.21       1.7%    0.055s  ok 3.4e-16       1.00x
   mkl2026_dfti                1147.289 us 307473.509 us    20.56       0.5%    0.050s  ok 3.4e-16       1.13x
   L64_radix8                  1184.003 us 317312.847 us    19.93       3.2%    0.337s  ok 4.5e-16       1.16x
   L64_blocked                 1205.721 us 323133.240 us    19.57       2.3%    0.349s  ok 4.5e-16       1.19x
   fftw3_patient               1390.117 us 372551.470 us    16.97       6.5%    3.762s  ok 3.6e-16       1.37x
   fftw3_measure               1408.347 us 377436.959 us    16.75       3.8%    0.268s  ok 3.5e-16       1.39x
   ducc0_c2c                   1870.285 us 501236.378 us    12.61       3.4%    0.000s  ok 3.0e-16       1.84x
   fftw3_estimate              2746.435 us 736044.513 us     8.59       0.9%    0.001s  ok 3.5e-16       2.70x
   baseline_matrix            94501.457 us 25326390.400 us     0.25       1.0%    0.000s  ok 7.8e-16       92.98x

backends:
   L13_direct               conj-folded dense 13x13 per axis, lanes=lines, pinned sines; 512b all-pinned zsolidY+xmm-tail X-first; ab[B16]=y23989,zs3947,p74077,xl4946 ns/vol
   L13_rader                Rader-13 split cyc/nega (186 FP/pt), X-first, 512-bit; fuse=0 um=1 ys=0 pf=0 pw=1 pace=1 znb=22 ab[B32]=i:5748,pw!:5455,pf!:5902,f1:5854 pick=pw!
   L17_matrixsimd           nested cyclic/negacyclic 17-pt/axis, 512-bit+ymm tail, pinned, X-first, addr-safe t1, extract-store, pf=0, pw=0, pt=1, b1dec[yz/kyz/x/kx]=8.67/7.90/4.30/3.91, clk512/256=2.90/2.90 GHz, d256=2.90
   L17_rader                Rader-17 cyclic/negacyclic (kernel from L17_winograd), plane-fused; tuned: xl 512 pin, pf=0, pfw=0, clk256=2.90 clk512=2.90, xrace xl/xfs=19.72/24.48, probe ph/xp/fu=15.35/7.07/22.75 us/vol
   L17_winograd             17-pt cyclic+negacyclic module (296 FP instr), 3 rotating passes, var=h8, pf=0, pfw=0, cw=0, clk256=3.50GHz, clk512=3.30GHz, p1=5.14 f23=7.57 fu=12.65 fu4=13.40
   L23_matrixsimd           dense 23x23/axis conj-folded, 512-bit, pinned, X-first, za, pf=0, pw=0, tune[pick=37.74 inc=40.33 us/t nv=8], clk512/256=2.90/2.90 GHz
   L23_rader                rader23 folded pair, 512-bit, pinned two-sweep, X-first, pf=0 pw=0, tuner pick=38.55 inc=38.55 us/t nv=8
   L36_mixedradix           PFA 4x9 2-sweep, lanes=lines, n1_9 DFT9; pick=v1-cached-pf0 (B=8, arena=4 vol, stream=0, 14 cand, pinD=-1) probe us/vol pf0=116.9 zy=137.9
   L36_pencilfused          L=36 plane-fused y+z then strided x, PFA4x9 interleaved lanes; tuner picked pw=4 mode=inplace (B=8); probe us ip4=99.4 cs4=99.6 is04=102.8
   L36_pfa                  GT-PFA 4x9 (n1_9 DAG) two-sweep; tuner pick: pw=4 mode=inplace pf=0 (B=8, nv=8, nc=14); probe us p1=63.1 p1z=28.2 p1y=19.9 p2w=20.8 fu=81.0
   L45_mixedradix           PFA 9x5 2-sweep; pick=v1-pf0 (B=4, arena=4, stream=0, 4 cand); nv1 us fu=273.5 p1=208.1 p2w=67.9; fe=na
   L45_pfa                  GT-PFA 9x5 two-sweep; tuner pick: pw4-ip-pf0 (B=4, nv=4)
   L64_blocked              L64 8x8 two-stage, hugepage odd-line-padded scratch; tuner pick: pw=4 mode=nt pf=0 st=3(split-sc) pro=0 (B=2, nv=2)
   L64_radix8               radix-8^2 per axis, split-complex AVX-512, padded scratch; tuner pick[B=2]=fused-nt+slabpf1+pro0+p10+sc0+xb0+fo1
   L6_pfa                   Good-Thomas PFA 2x3 per axis, no twiddles, 2 cplx/ymm, plan-raced; variant=fused_pf_d2 clkS256=2.90 clkD256=2.90 clkS512=2.90 kclk=2.90GHz bf=195.6 bsp=194.6 bx=56.3 byz=137.8ns
   L6_unrolled              L=6: unrolled PFA 2x3 codelet (48 flop/36 instr, no twiddles), radix-2-first VD6, ymm; variant=fused_zp kclk=3.30GHz ab1=f163.9,fx163.5ns abL=f434.3,f3440.3ns xod=-0.2%
   L8_batchsimd             radix-8 split; pick[B=64]: mode=FUSED nt=0 pf=s0 alloc=r8(a64,si512) arena{FUSED/s0=0.429,FUSEDAA/s0=0.426,FUSED/none=0.442}
   L8_fusedaxes             8^3 fused/AA/AA2 c52; B=64 pick=fused+pfs (tuned) arena{fused+pfs=0.415,fusedAA+pfs=0.412,fusedAA2+pfs=0.409}
   L8_radix8                radix-8 52-instr codelet; 2p/1f/3p shapes; pick[B=64]=avx512-1f-pfs (default) arena{1f-pfs=0.458 1f520-pfs=0.457 3p-pfs*=0.457 2p*=0.495} scr@0x4c0
   baseline_matrix          row-column dense DFT matrix, O(L^4)/volume/axis
   ducc0_c2c                ducc0 0.41 c2c, no planning, 1 thread
   fftw3_estimate           FFTW 3.3.10 plan_many_dft, fftw3_estimate
   fftw3_measure            FFTW 3.3.10 plan_many_dft, fftw3_measure
   fftw3_patient            FFTW 3.3.10 plan_many_dft, fftw3_patient
   mkl2026_dfti             oneMKL 2026.1 (pip wheel) DFTI, sequential, batched
   mkl_dfti                 oneMKL 2022.0.2 DFTI, sequential, batched
```
